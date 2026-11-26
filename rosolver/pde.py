"""
Generic 1D conservative finite-volume solver for systems of balance laws

    du_k/dt + d/dx [ f_k(u, x, t) - D_k(u, x, t) du_k/dx ] = S_k(u, x, t),   k = 1..m

on a uniform grid of N cells. Covers advection, (nonlinear) diffusion, reaction, Burgers,
acoustics/shallow-water-type hyperbolic systems, Fisher-KPP, heat conduction, pollutant
transport, porous-media and population models.

Discretisation
--------------
* Advective flux: Rusanov (local Lax-Friedrichs) with MUSCL reconstruction
  ('constant' = 1st order, 'linear' = unlimited 2nd order, 'minmod' / 'vanleer' / 'mc' = TVD).
  Wave speeds are spectral radii of the symbolic flux Jacobian df/du.
* Diffusive flux: central gradient at faces, D evaluated at the face average.
* Boundary conditions via two ghost cells: 'dirichlet' (2nd-order quadratic extrapolation),
  'neumann' (prescribed du/dx), 'outflow' (zero gradient), 'periodic'.
* Time integration: fixed-step schemes from `integrators` (dt from a CFL number or given),
  or any scipy `solve_ivp` method (with a banded Jacobian sparsity pattern for implicit ones).

Discrete conservation accounting
------------------------------
The boundary fluxes and the integrated source are appended to the state vector and advanced
with the *same* Runge-Kutta stages, so the discrete budget
    total(t) - total(0) = inflow_left - outflow_right + source
closes to round-off for any conservative scheme. `verify.audit_pde_conservation` checks this
internal accounting consistency; independent analytical and MMS benchmarks are required to verify
discrete PDE convergence and physical correctness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix

from ._parsing import ExprLike, names, parse_expr, real_symbol
from .integrators import integrate_fixed, is_fixed_scheme

_GAUSS_X = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
_GAUSS_W = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
_BC_KINDS = ("dirichlet", "neumann", "periodic", "outflow")
_IMPLICIT_SCIPY = ("BDF", "Radau", "LSODA")


def cell_average(fn: Callable[[np.ndarray], np.ndarray], edges: np.ndarray) -> np.ndarray:
    """Cell averages of fn over [edges[i], edges[i+1]] by 3-point Gauss-Legendre quadrature."""
    edges = np.asarray(edges, dtype=float)
    mid = 0.5 * (edges[:-1] + edges[1:])
    half = 0.5 * (edges[1:] - edges[:-1])
    nodes = mid[:, None] + half[:, None] * _GAUSS_X[None, :]
    vals = np.broadcast_to(np.asarray(fn(nodes.ravel()), dtype=float), (nodes.size,)).reshape(nodes.shape)
    return 0.5 * vals @ _GAUSS_W


def _minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(a * b > 0.0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def _vanleer(dl: np.ndarray, dr: np.ndarray) -> np.ndarray:
    s = dl + dr
    return np.where(dl * dr > 0.0, 2.0 * dl * dr / np.where(s == 0.0, 1.0, s), 0.0)


LIMITERS: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "constant": lambda dl, dr: np.zeros_like(dl),
    "linear": lambda dl, dr: 0.5 * (dl + dr),
    "minmod": _minmod,
    "vanleer": _vanleer,
    "mc": lambda dl, dr: _minmod(0.5 * (dl + dr), _minmod(2.0 * dl, 2.0 * dr)),
}


@dataclass
class BC:
    """Boundary condition. `value` may be a number, an expression in t, a callable(t), or a
    dict/list giving one such value per field."""

    kind: str
    value: Any = 0.0

    def __post_init__(self) -> None:
        self.kind = self.kind.lower()
        if self.kind not in _BC_KINDS:
            raise ValueError(f"Unknown boundary condition '{self.kind}'. Choose from {_BC_KINDS}.")

    @classmethod
    def dirichlet(cls, value: Any) -> "BC":
        return cls("dirichlet", value)

    @classmethod
    def neumann(cls, gradient: Any) -> "BC":
        return cls("neumann", gradient)

    @classmethod
    def periodic(cls) -> "BC":
        return cls("periodic", None)

    @classmethod
    def outflow(cls) -> "BC":
        return cls("outflow", 0.0)


def _as_bc(spec: Any) -> BC:
    if isinstance(spec, BC):
        return spec
    if isinstance(spec, str):
        kind = spec.lower()
        return BC(kind, None if kind == "periodic" else 0.0)
    if isinstance(spec, (tuple, list)) and len(spec) == 2 and isinstance(spec[0], str):
        return BC(spec[0], spec[1])
    raise ValueError(f"Cannot interpret boundary condition {spec!r}.")


def _is_bc_like(obj: Any) -> bool:
    return isinstance(obj, (BC, tuple, list)) or (isinstance(obj, str) and obj.lower() in _BC_KINDS)


def parse_bcs(spec: Any) -> Tuple[BC, BC]:
    """Accepts a single BC (both sides), a (left, right) pair, or {'left': .., 'right': ..}."""
    if isinstance(spec, Mapping) and set(spec) <= {"left", "right"}:
        return _as_bc(spec.get("left", "outflow")), _as_bc(spec.get("right", "outflow"))
    if isinstance(spec, (tuple, list)) and len(spec) == 2 and _is_bc_like(spec[1]):
        return _as_bc(spec[0]), _as_bc(spec[1])
    bc = _as_bc(spec)
    return bc, bc


@dataclass
class PDESolution:
    fields: List[str]
    x: np.ndarray
    edges: np.ndarray
    t: np.ndarray
    U: Dict[str, np.ndarray]  # field -> (n_t, N)
    h: float
    scheme: str
    reconstruction: str
    dt: Optional[float]
    n_steps: int
    nfev: int
    cfl_max: float
    formal_order: int
    budget: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    success: bool = True
    newton_failures: int = 0
    warnings: List[str] = field(default_factory=list)

    @property
    def n_cells(self) -> int:
        return int(self.x.size)

    def _name(self, name: Optional[str]) -> str:
        return self.fields[0] if name is None else name

    def __getitem__(self, name: str) -> np.ndarray:
        return self.U[name]

    def final(self, name: Optional[str] = None) -> np.ndarray:
        return self.U[self._name(name)][-1]

    def probe(self, x0: float, name: Optional[str] = None, index: int = -1) -> float:
        """Linear interpolation of cell values at position x0."""
        return float(np.interp(x0, self.x, self.U[self._name(name)][index]))

    def integral(self, name: Optional[str] = None, index: int = -1) -> float:
        return float(self.h * self.U[self._name(name)][index].sum())

    def level_crossing(self, level: float, name: Optional[str] = None, index: int = -1) -> float:
        """First x (from the left) where the profile crosses `level` (e.g. a front position)."""
        u = self.U[self._name(name)][index] - level
        idx = np.nonzero(np.sign(u[:-1]) * np.sign(u[1:]) <= 0)[0]
        if idx.size == 0:
            return float("nan")
        i = idx[0]
        if u[i + 1] == u[i]:
            return float(self.x[i])
        return float(self.x[i] - u[i] * (self.x[i + 1] - self.x[i]) / (u[i + 1] - u[i]))


class ConservationLaw1D:
    """
    Example (viscous Burgers):
        law = ConservationLaw1D("u", flux="u**2/2", diffusivity="nu")
        sol = law.solve((0, 1), 200, t_end=0.5, initial="sin(2*pi*x)",
                        bc="periodic", params={"nu": 0.01})
    """

    def __init__(
        self,
        fields: Union[str, Sequence[str]],
        flux: Any = None,
        diffusivity: Any = None,
        source: Any = None,
        x: str = "x",
        t: str = "t",
    ) -> None:
        self.fields: List[str] = [fields] if isinstance(fields, str) else list(fields)
        if not self.fields:
            raise ValueError("At least one field is required.")
        self.xname, self.tname = x, t
        self.x, self.t = real_symbol(x), real_symbol(t)
        self._base = {f: real_symbol(f) for f in self.fields}
        self._base[x], self._base[t] = self.x, self.t
        self.field_syms = [self._base[f] for f in self.fields]
        self.flux = self._per_field(flux, "flux")
        self.diffusivity = self._per_field(diffusivity, "diffusivity")
        self.source = self._per_field(source, "source")
        used = set().union(*[names(e) for e in self.flux + self.diffusivity + self.source])
        self.params: List[str] = sorted(used - set(self.fields) - {x, t})
        self._cache: Dict[Tuple, Any] = {}

    def __repr__(self) -> str:
        parts = []
        for f, F, D, S in zip(self.fields, self.flux, self.diffusivity, self.source):
            parts.append(f"d{f}/d{self.tname} + d/d{self.xname}[{F} - ({D})*d{f}/d{self.xname}] = {S}")
        return "ConservationLaw1D(" + "; ".join(parts) + ")"

    def _per_field(self, spec: Any, label: str) -> List[sp.Expr]:
        m = len(self.fields)
        if spec is None:
            return [sp.Integer(0)] * m
        if isinstance(spec, Mapping):
            unknown = set(spec) - set(self.fields)
            if unknown:
                raise KeyError(f"{label}: unknown field(s) {sorted(unknown)}.")
            return [parse_expr(spec.get(f, 0), self._base) for f in self.fields]
        if isinstance(spec, (list, tuple)):
            if len(spec) != m:
                raise ValueError(f"{label}: expected {m} entries, got {len(spec)}.")
            return [parse_expr(s, self._base) for s in spec]
        if m != 1:
            raise ValueError(f"{label} for a multi-field law must be a dict or list (one entry per field).")
        return [parse_expr(spec, self._base)]

    def _param_subs(self, params: Optional[Mapping[str, float]]) -> Dict[sp.Symbol, sp.Expr]:
        params = dict(params or {})
        missing = [p for p in self.params if p not in params]
        if missing:
            raise KeyError(f"Missing values for parameters {missing}.")
        return {real_symbol(k): sp.Float(float(v)) for k, v in params.items()}

    def manufactured_source(self, exact: Any, params: Optional[Mapping[str, float]] = None) -> Dict[str, sp.Expr]:
        """
        Method of Manufactured Solutions: the extra source that makes `exact(x, t)` an exact
        solution, S_mms = du_e/dt + d/dx[f(u_e) - D(u_e) du_e/dx] - S(u_e).
        """
        ue = self._per_field(exact, "exact")
        psubs = self._param_subs(params) if params is not None else {}
        ue = [e.xreplace(psubs) for e in ue]
        fsubs = dict(zip(self.field_syms, ue))
        out: Dict[str, sp.Expr] = {}
        for k, f in enumerate(self.fields):
            F = self.flux[k].xreplace(psubs).xreplace(fsubs)
            D = self.diffusivity[k].xreplace(psubs).xreplace(fsubs)
            S = self.source[k].xreplace(psubs).xreplace(fsubs)
            out[f] = sp.diff(ue[k], self.t) + sp.diff(F - D * sp.diff(ue[k], self.x), self.x) - S
        return out

    def _compile(self, params: Optional[Mapping[str, float]]) -> "_Compiled":
        key = tuple(sorted((k, float(v)) for k, v in (params or {}).items()))
        if key not in self._cache:
            psubs = self._param_subs(params)
            self._cache[key] = _Compiled(self, psubs)
        return self._cache[key]

    def semidiscrete(
        self,
        domain: Tuple[float, float],
        n_cells: int,
        initial: Any,
        bc: Any = "outflow",
        params: Optional[Mapping[str, float]] = None,
        reconstruction: str = "minmod",
        t0: float = 0.0,
    ) -> Tuple[Callable[[float, np.ndarray], np.ndarray], np.ndarray]:
        """Return the method-of-lines RHS f(t, U_flat) and the initial cell averages (for
        stiffness/eigenvalue analysis of the spatial discretisation)."""
        fv = _FVM(self, domain, n_cells, bc, params, reconstruction, None, t0)
        U0 = fv.initial(initial)
        m, N = U0.shape

        def f(t, y):
            dU, _, _ = fv.rhs_parts(t, np.asarray(y, dtype=float).reshape(m, N))
            return dU.ravel()

        return f, U0.ravel()

    def solve(
        self,
        domain: Tuple[float, float],
        n_cells: int,
        t_end: float,
        initial: Any,
        bc: Any = "outflow",
        params: Optional[Mapping[str, float]] = None,
        t0: float = 0.0,
        scheme: str = "ssprk3",
        dt: Optional[float] = None,
        cfl: float = 0.4,
        reconstruction: str = "minmod",
        n_out: int = 50,
        rtol: float = 1e-8,
        atol: float = 1e-10,
        extra_source: Any = None,
    ) -> PDESolution:
        """
        Solve on `domain` with `n_cells` uniform cells up to `t_end`.

        initial: per field, a number, an expression in x (params allowed) or callable(x);
                 converted to exact cell averages.
        bc: e.g. "periodic", ("dirichlet", 1.0), {"left": BC.dirichlet("1+sin(t)"),
            "right": "outflow"}, or a (left, right) pair.
        scheme: fixed-step ('ssprk3', 'rk4', 'euler', 'beuler', 'cn', ...) with dt = cfl /
                (a_max/h + 2 D_max/h^2) unless dt is given; or a scipy method ('RK45', 'BDF',...).
        extra_source: optional per-field expression/callable in (x, t) added to S (used by MMS).
        """
        if not t_end > t0:
            raise ValueError("t_end must be greater than t0.")
        fv = _FVM(self, domain, n_cells, bc, params, reconstruction, extra_source, t0)
        U0 = fv.initial(initial)
        m, N = U0.shape
        mN = m * N
        y0 = np.concatenate([U0.ravel(), np.zeros(3 * m)])

        newton_failures = 0
        warnings: List[str] = []
        if is_fixed_scheme(scheme):
            if dt is None:
                rate = fv.max_rate(U0, t0)
                dt = cfl / rate if rate > 0 else (t_end - t0) / 100.0
            res = integrate_fixed(fv.rhs_aug, (t0, t_end), y0, dt, scheme=scheme, n_out=n_out)
            t, Y, dt_used, n_steps, nfev = res.t, res.y, res.dt, res.n_steps, res.nfev
            newton_failures = res.newton_failures
            if newton_failures > 0:
                warnings.append(
                    f"Implicit scheme '{scheme}' encountered {newton_failures} Newton convergence failure(s)."
                )
        else:
            te = np.linspace(t0, t_end, n_out + 1)
            kw = {"jac_sparsity": fv.sparsity()} if scheme in _IMPLICIT_SCIPY else {}
            sol = solve_ivp(fv.rhs_aug, (t0, t_end), y0, method=scheme, t_eval=te, rtol=rtol, atol=atol, **kw)
            if not sol.success:
                raise RuntimeError(f"Time integration failed: {sol.message}")
            t, Y, dt_used, n_steps, nfev = sol.t, sol.y.T, None, len(sol.t) - 1, int(sol.nfev)

        Us = Y[:, :mN].reshape(len(t), m, N)
        cfl_max = float("nan")
        if dt_used is not None:
            cfl_max = float(max(dt_used * fv.max_rate(Us[i], t[i]) for i in range(len(t))))
        U = {f: Us[:, k, :] for k, f in enumerate(self.fields)}
        budget = {}
        for k, f in enumerate(self.fields):
            budget[f] = {
                "total": fv.h * Us[:, k, :].sum(axis=1),
                "inflow_left": Y[:, mN + k],
                "outflow_right": Y[:, mN + m + k],
                "source": Y[:, mN + 2 * m + k],
            }
        return PDESolution(
            fields=list(self.fields), x=fv.xc, edges=fv.xf, t=np.asarray(t), U=U, h=fv.h,
            scheme=scheme, reconstruction=reconstruction, dt=dt_used, n_steps=n_steps, nfev=nfev,
            cfl_max=cfl_max, formal_order=1 if reconstruction == "constant" else 2, budget=budget,
            success=(newton_failures == 0), newton_failures=newton_failures, warnings=warnings,
        )


class _Compiled:
    def __init__(self, law: ConservationLaw1D, psubs: Dict[sp.Symbol, sp.Expr]) -> None:
        args = [*law.field_syms, law.x, law.t]
        flux = [e.xreplace(psubs) for e in law.flux]
        diff = [e.xreplace(psubs) for e in law.diffusivity]
        src = [e.xreplace(psubs) for e in law.source]
        self.m = len(law.fields)
        self.has_flux = any(e != 0 for e in flux)
        self.has_diff = any(e != 0 for e in diff)
        self.has_src = any(e != 0 for e in src)
        self.flux_fn = sp.lambdify(args, flux, "numpy")
        self.diff_fn = sp.lambdify(args, diff, "numpy")
        self.src_fn = sp.lambdify(args, src, "numpy")
        self.jac_fn = sp.lambdify(args, sp.Matrix(flux).jacobian(law.field_syms).tolist(), "numpy")

    @staticmethod
    def _stack(vals, n: int) -> np.ndarray:
        return np.array([np.broadcast_to(np.asarray(v, dtype=float), (n,)) for v in vals])

    def flux(self, U, x, t):
        return self._stack(self.flux_fn(*U, x, t), U.shape[1])

    def diffusivity(self, U, x, t):
        return self._stack(self.diff_fn(*U, x, t), U.shape[1])

    def source(self, U, x, t):
        return self._stack(self.src_fn(*U, x, t), U.shape[1])

    def speed(self, U, x, t) -> np.ndarray:
        n = U.shape[1]
        if not self.has_flux:
            return np.zeros(n)
        J = self.jac_fn(*U, x, t)
        if self.m == 1:
            return np.abs(np.broadcast_to(np.asarray(J[0][0], dtype=float), (n,)))
        arr = np.empty((self.m, self.m, n))
        for i in range(self.m):
            for j in range(self.m):
                arr[i, j] = np.broadcast_to(np.asarray(J[i][j], dtype=float), (n,))
        return np.abs(np.linalg.eigvals(np.moveaxis(arr, 2, 0))).max(axis=1)


class _FVM:
    def __init__(self, law, domain, n_cells, bc, params, reconstruction, extra_source, t0) -> None:
        a, b = float(domain[0]), float(domain[1])
        if not b > a:
            raise ValueError("domain must satisfy b > a.")
        N = int(n_cells)
        if N < 3:
            raise ValueError("At least 3 cells are required.")
        if reconstruction not in LIMITERS:
            raise ValueError(f"Unknown reconstruction '{reconstruction}'. Choose from {sorted(LIMITERS)}.")
        self.law, self.N, self.m = law, N, len(law.fields)
        self.a, self.b = a, b
        self.h = (b - a) / N
        self.xc = a + (np.arange(N) + 0.5) * self.h
        self.xf = a + np.arange(N + 1) * self.h
        self.t0 = float(t0)
        self.params = dict(params or {})
        self.c = law._compile(params)
        self.left, self.right = parse_bcs(bc)
        if (self.left.kind == "periodic") != (self.right.kind == "periodic"):
            raise ValueError("Periodic boundary conditions must be applied on both sides.")
        self.left_val = self._bc_fn(self.left)
        self.right_val = self._bc_fn(self.right)
        self.limiter = LIMITERS[reconstruction]
        self.extra = self._extra_fn(extra_source)

    # ---- user value helpers
    def _items(self, spec: Any, label: str) -> List[Any]:
        m = self.m
        if isinstance(spec, Mapping):
            missing = [f for f in self.law.fields if f not in spec]
            if missing:
                raise KeyError(f"{label}: missing value for field(s) {missing}.")
            return [spec[f] for f in self.law.fields]
        if isinstance(spec, (list, tuple)) and len(spec) == m and m > 1:
            return list(spec)
        return [spec] * m

    def _symbols(self) -> Dict[str, sp.Symbol]:
        return {self.law.xname: self.law.x, self.law.tname: self.law.t}

    def _expr(self, item: Any) -> sp.Expr:
        e = parse_expr(item, self._symbols())
        psubs = {real_symbol(k): sp.Float(float(v)) for k, v in self.params.items()}
        e = e.xreplace(psubs)
        unknown = names(e) - {self.law.xname, self.law.tname}
        if unknown:
            raise KeyError(f"Unknown symbols {sorted(unknown)} in '{item}' (pass them via params).")
        return e

    def _bc_fn(self, bc: BC) -> Optional[Callable[[float], np.ndarray]]:
        if bc.kind == "periodic":
            return None
        fns = []
        for item in self._items(bc.value if bc.value is not None else 0.0, f"{bc.kind} BC"):
            if callable(item) and not isinstance(item, sp.Basic):
                fns.append(lambda t, g=item: float(g(t)))
            else:
                e = self._expr(item)
                g = sp.lambdify([self.law.t], e, "numpy")
                fns.append(lambda t, g=g: float(g(t)))
        return lambda t: np.array([fn(t) for fn in fns])

    def _extra_fn(self, spec: Any) -> Optional[Callable[[np.ndarray, float], np.ndarray]]:
        if spec is None:
            return None
        fns = []
        for item in self._items(spec, "extra_source"):
            if item is None:
                fns.append(lambda x, t: np.zeros_like(x))
            elif callable(item) and not isinstance(item, sp.Basic):
                fns.append(item)
            else:
                g = sp.lambdify([self.law.x, self.law.t], self._expr(item), "numpy")
                fns.append(g)
        n = self.N
        return lambda x, t: np.array([np.broadcast_to(np.asarray(fn(x, t), dtype=float), (n,)) for fn in fns])

    def initial(self, spec: Any) -> np.ndarray:
        U0 = np.empty((self.m, self.N))
        for k, item in enumerate(self._items(spec, "initial")):
            if callable(item) and not isinstance(item, sp.Basic):
                fn = item
            else:
                e = self._expr(item).xreplace({self.law.t: sp.Float(self.t0)})
                fn = sp.lambdify([self.law.x], e, "numpy")
            U0[k] = cell_average(fn, self.xf)
        return U0

    # ---- discretisation
    def ghost(self, U: np.ndarray, t: float) -> np.ndarray:
        N, h = self.N, self.h
        Ue = np.empty((self.m, N + 4))
        Ue[:, 2:N + 2] = U
        if self.left.kind == "periodic":
            Ue[:, 0:2] = U[:, N - 2:N]
            Ue[:, N + 2:] = U[:, 0:2]
            return Ue
        for side in ("left", "right"):
            bc = self.left if side == "left" else self.right
            val = (self.left_val if side == "left" else self.right_val)(t)
            u1, u2 = (U[:, 0], U[:, 1]) if side == "left" else (U[:, N - 1], U[:, N - 2])
            if bc.kind == "dirichlet":
                g1 = 8.0 / 3.0 * val - 2.0 * u1 + u2 / 3.0
                g2 = 8.0 * val - 9.0 * u1 + 2.0 * u2
            else:  # neumann / outflow: val = du/dx at the boundary
                g = val if bc.kind == "neumann" else 0.0
                sgn = -1.0 if side == "left" else 1.0
                g1 = u1 + sgn * g * h
                g2 = u2 + sgn * 3.0 * g * h
            if side == "left":
                Ue[:, 1], Ue[:, 0] = g1, g2
            else:
                Ue[:, N + 2], Ue[:, N + 3] = g1, g2
        return Ue

    def rhs_parts(self, t: float, U: np.ndarray):
        N, h, c = self.N, self.h, self.c
        Ue = self.ghost(U, t)
        left_c, right_c = Ue[:, 1:N + 2], Ue[:, 2:N + 3]
        F = np.zeros((self.m, N + 1))
        if c.has_flux:
            dl = Ue[:, 1:-1] - Ue[:, :-2]
            dr = Ue[:, 2:] - Ue[:, 1:-1]
            slope = self.limiter(dl, dr)
            uL = left_c + 0.5 * slope[:, 0:N + 1]
            uR = right_c - 0.5 * slope[:, 1:N + 2]
            fL, fR = c.flux(uL, self.xf, t), c.flux(uR, self.xf, t)
            a = np.maximum(c.speed(uL, self.xf, t), c.speed(uR, self.xf, t))
            F += 0.5 * (fL + fR) - 0.5 * a * (uR - uL)
        if c.has_diff:
            uf = 0.5 * (left_c + right_c)
            F -= c.diffusivity(uf, self.xf, t) * (right_c - left_c) / h
        S = c.source(U, self.xc, t) if c.has_src else np.zeros_like(U)
        if self.extra is not None:
            S = S + self.extra(self.xc, t)
        dU = -(F[:, 1:] - F[:, :-1]) / h + S
        return dU, F, S

    def rhs_aug(self, t: float, y: np.ndarray) -> np.ndarray:
        m, N = self.m, self.N
        dU, F, S = self.rhs_parts(t, y[: m * N].reshape(m, N))
        return np.concatenate([dU.ravel(), F[:, 0], F[:, -1], self.h * S.sum(axis=1)])

    def max_rate(self, U: np.ndarray, t: float) -> float:
        """a_max/h + 2 D_max/h^2: dt * max_rate is the effective CFL/Fourier number."""
        a = float(np.max(self.c.speed(U, self.xc, t))) if self.c.has_flux else 0.0
        D = float(np.max(np.abs(self.c.diffusivity(U, self.xc, t)))) if self.c.has_diff else 0.0
        return a / self.h + 2.0 * D / self.h ** 2

    def sparsity(self):
        m, N = self.m, self.N
        n = m * N + 3 * m
        S = lil_matrix((n, n))
        periodic = self.left.kind == "periodic"
        for k in range(m):
            for i in range(N):
                row = k * N + i
                for j in range(i - 3, i + 4):
                    jj = j % N if periodic else min(max(j, 0), N - 1)
                    for kk in range(m):
                        S[row, kk * N + jj] = 1
        S[m * N:, : m * N] = 1
        return S.tocsr()
