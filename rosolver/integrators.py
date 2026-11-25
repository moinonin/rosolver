"""
Fixed-step time integrators and linear-stability utilities.

Adaptive solvers choose their own steps, which hides the effect of the time step. Time-step
sensitivity studies therefore need integrators whose step is *exactly* what the user prescribes.
This module provides explicit Runge-Kutta schemes (Euler, SSP-RK2/3, Bogacki-Shampine 3,
classical RK4, Dormand-Prince 5) and implicit one-step schemes (backward Euler, Crank-Nicolson)
plus the stability function R(z) of each scheme, used to turn Jacobian eigenvalues into explicit
step-size limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.linalg import lu_factor, lu_solve

RHS = Callable[[float, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ButcherTableau:
    name: str
    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    order: int


def _tab(name, A, b, c, order) -> ButcherTableau:
    return ButcherTableau(name, np.array(A, float), np.array(b, float), np.array(c, float), order)


EXPLICIT_TABLEAUS = {
    "euler": _tab("euler", [[0]], [1], [0], 1),
    "ssprk2": _tab("ssprk2", [[0, 0], [1, 0]], [0.5, 0.5], [0, 1], 2),
    "ssprk3": _tab("ssprk3", [[0, 0, 0], [1, 0, 0], [0.25, 0.25, 0]], [1 / 6, 1 / 6, 2 / 3], [0, 1, 0.5], 3),
    "bs3": _tab("bs3", [[0, 0, 0], [0.5, 0, 0], [0, 0.75, 0]], [2 / 9, 1 / 3, 4 / 9], [0, 0.5, 0.75], 3),
    "rk4": _tab(
        "rk4",
        [[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]],
        [1 / 6, 1 / 3, 1 / 3, 1 / 6],
        [0, 0.5, 0.5, 1],
        4,
    ),
    "dopri5": _tab(
        "dopri5",
        [
            [0, 0, 0, 0, 0, 0, 0],
            [1 / 5, 0, 0, 0, 0, 0, 0],
            [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
            [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
        ],
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
        [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1],
        5,
    ),
}
IMPLICIT_FIXED = {"beuler": 1, "cn": 2}
_ALIASES = {"heun": "ssprk2", "rk2": "ssprk2", "rk3": "ssprk3", "forward_euler": "euler",
            "backward_euler": "beuler", "crank_nicolson": "cn", "trapezoidal": "cn"}
# scipy.integrate.solve_ivp names -> tableau used for stability analysis
_SCIPY_EXPLICIT = {"RK45": "dopri5", "RK23": "bs3"}
_SCIPY_IMPLICIT = {"BDF", "Radau", "LSODA"}
_SCIPY_ORDER = {"RK45": 5, "RK23": 3, "DOP853": 8, "Radau": 5}


def canonical_scheme(name: str) -> str:
    if name in _SCIPY_EXPLICIT:
        return _SCIPY_EXPLICIT[name]
    if name in _SCIPY_IMPLICIT or name == "DOP853":
        return name
    low = name.lower()
    return _ALIASES.get(low, low)


def is_fixed_scheme(name: str) -> bool:
    if name in _SCIPY_EXPLICIT or name in _SCIPY_IMPLICIT or name == "DOP853":
        return False
    low = _ALIASES.get(name.lower(), name.lower())
    return low in EXPLICIT_TABLEAUS or low in IMPLICIT_FIXED


def is_implicit(name: str) -> bool:
    c = canonical_scheme(name)
    return c in IMPLICIT_FIXED or c in _SCIPY_IMPLICIT


def scheme_order(name: str) -> Optional[int]:
    if name in _SCIPY_ORDER:
        return _SCIPY_ORDER[name]
    c = canonical_scheme(name)
    if c in EXPLICIT_TABLEAUS:
        return EXPLICIT_TABLEAUS[c].order
    return IMPLICIT_FIXED.get(c)


def stability_function(scheme: str, z) -> np.ndarray:
    """Linear stability function R(z) so that y_{n+1} = R(dt*lambda) y_n for y' = lambda y."""
    z = np.asarray(z, dtype=complex)
    c = canonical_scheme(scheme)
    if c == "beuler":
        return 1.0 / (1.0 - z)
    if c == "cn":
        return (1.0 + z / 2.0) / (1.0 - z / 2.0)
    if c not in EXPLICIT_TABLEAUS:
        raise ValueError(f"No stability function available for scheme '{scheme}'.")
    tab = EXPLICIT_TABLEAUS[c]
    stages = []
    for i in range(len(tab.b)):
        acc = np.ones_like(z)
        for j in range(i):
            if tab.A[i, j] != 0.0:
                acc = acc + z * tab.A[i, j] * stages[j]
        stages.append(acc)
    out = np.ones_like(z)
    for bi, yi in zip(tab.b, stages):
        if bi != 0.0:
            out = out + z * bi * yi
    return out


def max_stable_dt(scheme: str, eigenvalues: Sequence[complex], growth_tol: float = 1e-10) -> float:
    """
    Largest dt with |R(dt*lambda)| <= 1 for every non-growing eigenvalue.

    Growing modes (Re(lambda) > 0) are physical instabilities, not step-size constraints, and
    are ignored. Returns inf for A-stable/implicit schemes and 0.0 when no positive dt is
    stable (e.g. forward Euler on purely imaginary eigenvalues).
    """
    c = canonical_scheme(scheme)
    if c in IMPLICIT_FIXED or c in _SCIPY_IMPLICIT:
        return float("inf")
    if c not in EXPLICIT_TABLEAUS:
        raise ValueError(f"Unknown scheme '{scheme}' for stability analysis.")
    limits = []
    for lam in np.atleast_1d(np.asarray(eigenvalues, dtype=complex)):
        mag = abs(lam)
        if mag < 1e-300 or lam.real > growth_tol * mag:
            continue
        dts = np.logspace(-5, 3, 600) / mag
        stable = np.abs(stability_function(c, dts * lam)) <= 1.0 + 1e-12
        if not stable[0]:
            limits.append(0.0)
            continue
        bad = np.nonzero(~stable)[0]
        if bad.size == 0:
            continue
        lo, hi = dts[bad[0] - 1], dts[bad[0]]
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if abs(stability_function(c, mid * lam)) <= 1.0 + 1e-12:
                lo = mid
            else:
                hi = mid
        limits.append(float(lo))
    return min(limits) if limits else float("inf")


def fd_jacobian(f: RHS, t: float, y: np.ndarray, f0: Optional[np.ndarray] = None) -> np.ndarray:
    """Forward-difference Jacobian df/dy."""
    y = np.asarray(y, dtype=float)
    f0 = np.asarray(f(t, y), dtype=float) if f0 is None else f0
    J = np.empty((f0.size, y.size))
    for j in range(y.size):
        eps = 1.4901161193847656e-08 * max(1.0, abs(y[j]))
        yp = y.copy()
        yp[j] += eps
        J[:, j] = (np.asarray(f(t, yp), dtype=float) - f0) / eps
    return J


@dataclass
class FixedStepResult:
    t: np.ndarray
    y: np.ndarray
    dt: float
    n_steps: int
    scheme: str
    nfev: int
    newton_failures: int = 0


def integrate_fixed(
    f: RHS,
    t_span: Sequence[float],
    y0: Sequence[float],
    dt: float,
    scheme: str = "rk4",
    n_out: Optional[int] = None,
    jac: Optional[Callable[[float, np.ndarray], np.ndarray]] = None,
    newton_tol: float = 1e-12,
    newton_maxiter: int = 30,
) -> FixedStepResult:
    """
    Integrate y' = f(t, y) with a constant step.

    The step is adjusted down to dt_used = (t1 - t0) / ceil((t1 - t0) / dt) so that the final
    time is hit exactly; `dt_used` is reported. `n_out=None` stores every step, otherwise
    `n_out + 1` evenly spaced step indices are stored.
    """
    t0, t1 = float(t_span[0]), float(t_span[1])
    if not t1 > t0:
        raise ValueError("t_span must satisfy t1 > t0.")
    if not dt > 0:
        raise ValueError("dt must be positive.")
    name = canonical_scheme(scheme)
    if not is_fixed_scheme(scheme):
        raise ValueError(
            f"Unknown fixed-step scheme '{scheme}'. Choose from "
            f"{sorted(EXPLICIT_TABLEAUS) + sorted(IMPLICIT_FIXED)}."
        )
    n_steps = max(1, int(np.ceil((t1 - t0) / dt - 1e-9)))
    h = (t1 - t0) / n_steps
    save = np.zeros(n_steps + 1, dtype=bool)
    if n_out is None:
        save[:] = True
    else:
        save[np.unique(np.round(np.linspace(0, n_steps, int(n_out) + 1)).astype(int))] = True

    y = np.array(y0, dtype=float).reshape(-1).copy()
    ts, ys = [], []
    if save[0]:
        ts.append(t0)
        ys.append(y.copy())
    nfev = 0
    failures = 0

    if name in EXPLICIT_TABLEAUS:
        tab = EXPLICIT_TABLEAUS[name]
        s = len(tab.b)
        K = np.empty((s, y.size))
        for n in range(n_steps):
            t = t0 + n * h
            for i in range(s):
                yi = y if i == 0 else y + h * (tab.A[i, :i] @ K[:i])
                K[i] = f(t + tab.c[i] * h, yi)
            y = y + h * (tab.b @ K)
            nfev += s
            if save[n + 1]:
                ts.append(t0 + (n + 1) * h)
                ys.append(y.copy())
    else:
        theta = 1.0 if name == "beuler" else 0.5
        eye = np.eye(y.size)
        for n in range(n_steps):
            t = t0 + n * h
            tn1 = t + h
            base = y.copy()
            if theta < 1.0:
                base = base + (1.0 - theta) * h * np.asarray(f(t, y), dtype=float)
                nfev += 1
            Y = y.copy()
            J = jac(tn1, Y) if jac is not None else fd_jacobian(f, tn1, Y)
            nfev += 0 if jac is not None else y.size + 1
            lu = lu_factor(eye - theta * h * np.asarray(J, dtype=float))
            for _ in range(newton_maxiter):
                G = Y - base - theta * h * np.asarray(f(tn1, Y), dtype=float)
                nfev += 1
                dY = lu_solve(lu, -G)
                Y = Y + dY
                if np.linalg.norm(dY, np.inf) <= newton_tol * (1.0 + np.linalg.norm(Y, np.inf)):
                    break
            else:
                failures += 1
            y = Y
            if save[n + 1]:
                ts.append(tn1)
                ys.append(y.copy())

    return FixedStepResult(
        t=np.array(ts), y=np.array(ys), dt=h, n_steps=n_steps, scheme=name, nfev=nfev,
        newton_failures=failures,
    )
