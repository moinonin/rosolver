"""
Symbolic multi-state ODE systems  dy/dt = f(t, y; p).

Any domain (kinetics, epidemiology, mechanics, economics) can be declared from strings. The
class provides exact symbolic Jacobians (stiffness/stability analysis, implicit solvers),
model-level invariant checks (is a quantity conserved by the equations themselves?),
fixed-step and adaptive integration, and construction from rosolver derivations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from ._parsing import ExprLike, names, parse_expr, real_symbol
from .integrators import integrate_fixed, is_fixed_scheme
from .results import ResultMixin

ParamValue = Union[float, Callable[[float], float]]
_IMPLICIT_SCIPY = ("BDF", "Radau", "LSODA")


@dataclass
class EventSpec:
    """Zero-crossing event specification for `ODESystem.solve`."""

    name: str
    expr: Union[ExprLike, Callable[[float, np.ndarray], float]]
    terminal: bool = False
    direction: float = 0.0


@dataclass
class Trajectory(ResultMixin):
    """Time history of an ODE solve."""

    t: np.ndarray
    y: np.ndarray  # (n_t, n_states)
    states: List[str]
    method: str
    nfev: int
    success: bool = True
    message: str = ""
    dt: Optional[float] = None
    njev: int = 0
    newton_failures: int = 0
    warnings: List[str] = field(default_factory=list)
    t_events: Dict[str, np.ndarray] = field(default_factory=dict)
    y_events: Dict[str, np.ndarray] = field(default_factory=dict)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.y[:, self.states.index(name)]

    @property
    def final(self) -> Dict[str, float]:
        return {s: float(self.y[-1, i]) for i, s in enumerate(self.states)}

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {s: self.y[:, i] for i, s in enumerate(self.states)}


class ODESystem:
    """
    Example:
        sir = ODESystem({
            "S": "-beta*S*I/N",
            "I": "beta*S*I/N - gamma*I",
            "R": "gamma*I",
        })
        sir.check_invariant("S + I + R")      # (True, 0)
        traj = sir.solve((0, 160), {"S": 990, "I": 10, "R": 0},
                         params={"beta": 0.3, "gamma": 0.1, "N": 1000})
    """

    def __init__(self, rhs: Mapping[str, ExprLike], indep: str = "t") -> None:
        if not rhs:
            raise ValueError("ODESystem needs at least one state equation.")
        self.indep = indep
        self.states: List[str] = list(rhs.keys())
        if indep in self.states:
            raise ValueError(f"Independent variable '{indep}' cannot also be a state.")
        self.t = real_symbol(indep)
        base = {n: real_symbol(n) for n in self.states}
        base[indep] = self.t
        self.state_syms = [base[n] for n in self.states]
        self.rhs: List[sp.Expr] = [parse_expr(rhs[n], base) for n in self.states]
        pnames = set().union(*[names(e) for e in self.rhs]) - set(self.states) - {indep}
        self.params: List[str] = sorted(pnames)
        self.param_syms = [real_symbol(p) for p in self.params]
        self._symbols = {**base, **dict(zip(self.params, self.param_syms))}
        args = [self.t, *self.state_syms, *self.param_syms]
        self._f = sp.lambdify(args, self.rhs, "numpy")
        self.jacobian_expr = sp.Matrix(self.rhs).jacobian(self.state_syms)
        self._J = sp.lambdify(args, self.jacobian_expr.tolist(), "numpy")

    def __repr__(self) -> str:
        eqs = ", ".join(f"d{s}/d{self.indep} = {e}" for s, e in zip(self.states, self.rhs))
        return f"ODESystem({eqs})"

    @classmethod
    def from_derivation(cls, result) -> "ODESystem":
        """Build the ODE d(target)/d(wrt) = RHS from a rosolver DerivationResult."""
        if result.differential_equation is None or result.wrt is None:
            raise ValueError("DerivationResult has no differential equation (derive with wrt=...).")
        return cls({result.target: result.differential_equation.rhs}, indep=result.wrt)

    # ------------------------------------------------------------------ numerics
    def _param_getter(self, params: Optional[Mapping[str, ParamValue]], only: Optional[Sequence[str]] = None):
        params = dict(params or {})
        needed = list(self.params) if only is None else list(only)
        missing = [p for p in needed if p not in params]
        if missing:
            raise KeyError(f"Missing values for parameters {missing}.")
        vals = [params[p] for p in needed]
        if not any(callable(v) for v in vals):
            static = [float(v) for v in vals]
            return lambda t: static
        return lambda t: [v(t) if callable(v) else float(v) for v in vals]

    def _y0(self, y0) -> np.ndarray:
        if isinstance(y0, Mapping):
            missing = [s for s in self.states if s not in y0]
            if missing:
                raise KeyError(f"Missing initial values for states {missing}.")
            return np.array([float(y0[s]) for s in self.states])
        arr = np.asarray(y0, dtype=float).reshape(-1)
        if arr.size != len(self.states):
            raise ValueError(f"Expected {len(self.states)} initial values, got {arr.size}.")
        return arr

    def rhs_function(self, params: Optional[Mapping[str, ParamValue]] = None) -> Callable:
        pv = self._param_getter(params)
        fn = self._f

        def f(t, y):
            return np.asarray(fn(t, *y, *pv(t)), dtype=float).reshape(-1)

        return f

    def jac_function(self, params: Optional[Mapping[str, ParamValue]] = None) -> Callable:
        pv = self._param_getter(params)
        fn = self._J
        m = len(self.states)

        def J(t, y):
            return np.asarray(fn(t, *y, *pv(t)), dtype=float).reshape(m, m)

        return J

    def _compile_events(
        self,
        events: Optional[Sequence[Union[EventSpec, Callable]]],
        params: Optional[Mapping[str, ParamValue]],
    ) -> Tuple[List[str], List[Callable]]:
        if not events:
            return [], []
        pv = self._param_getter(params)
        ev_names: List[str] = []
        ev_fns: List[Callable] = []
        for idx, ev in enumerate(events):
            if isinstance(ev, EventSpec):
                ev_names.append(ev.name)
                if callable(ev.expr) and not isinstance(ev.expr, sp.Basic):
                    raw_fn = ev.expr

                    def _fn(t, y, g=raw_fn):
                        return float(g(t, y))
                else:
                    e = parse_expr(ev.expr, self._symbols)
                    unknown = names(e) - set(self.states) - set(self.params) - {self.indep}
                    if unknown:
                        raise KeyError(f"Unknown symbols {sorted(unknown)} in event '{ev.name}'.")
                    args = [self.t, *self.state_syms, *self.param_syms]
                    compiled = sp.lambdify(args, e, "numpy")

                    def _fn(t, y, g=compiled, get_p=pv):
                        return float(g(t, *y, *get_p(t)))

                _fn.terminal = bool(ev.terminal)  # type: ignore[attr-defined]
                _fn.direction = float(ev.direction)  # type: ignore[attr-defined]
                ev_fns.append(_fn)
            elif callable(ev):
                ev_names.append(getattr(ev, "__name__", f"event_{idx}"))
                ev_fns.append(ev)
            else:
                raise TypeError(f"Unsupported event specification type: {type(ev)!r}.")
        return ev_names, ev_fns

    def solve(
        self,
        t_span: Tuple[float, float],
        y0,
        params: Optional[Mapping[str, ParamValue]] = None,
        method: str = "RK45",
        rtol: float = 1e-8,
        atol: float = 1e-10,
        n_out: Optional[int] = 200,
        t_eval: Optional[Sequence[float]] = None,
        dt: Optional[float] = None,
        max_step: float = np.inf,
        raise_on_failure: bool = False,
        events: Optional[Sequence[Union[EventSpec, Callable]]] = None,
    ) -> Trajectory:
        """
        Integrate the system. `method` is a scipy name ('RK45', 'Radau', 'BDF', 'LSODA', ...)
        or a fixed-step scheme ('euler', 'ssprk3', 'rk4', 'beuler', 'cn', ...) which needs `dt`.
        """
        y0v = self._y0(y0)
        f = self.rhs_function(params)
        if is_fixed_scheme(method):
            if events:
                raise ValueError("Event detection requires a scipy solve_ivp method (fixed-step schemes do not support events).")
            if dt is None:
                raise ValueError(f"Fixed-step scheme '{method}' requires dt.")
            res = integrate_fixed(f, t_span, y0v, dt, scheme=method, n_out=n_out if t_eval is None else None,
                                  jac=self.jac_function(params))
            finite_ok = (not res.diverged) and bool(np.all(np.isfinite(res.y)))
            ok = (res.newton_failures == 0) and finite_ok
            warns: List[str] = []
            if res.newton_failures > 0:
                warns.append(
                    f"Fixed-step implicit solver '{res.scheme}' encountered {res.newton_failures} Newton convergence failure(s)."
                )
            if not finite_ok:
                warns.append(
                    f"Fixed-step solver '{res.scheme}' diverged to non-finite values (Inf/NaN); reduce dt or use an implicit solver."
                )
            msg = "fixed-step" if ok else warns[0]
            if raise_on_failure and not ok:
                raise RuntimeError(msg)
            t_out, y_out = res.t, res.y
            if t_eval is not None:
                te_arr = np.asarray(t_eval, dtype=float).reshape(-1)
                span_tol = 1e-12 * max(abs(float(t_span[1]) - float(t_span[0])), 1.0)
                if te_arr.size > 0 and (float(np.min(te_arr)) < float(t_span[0]) - span_tol or float(np.max(te_arr)) > float(t_span[1]) + span_tol):
                    raise ValueError("All `t_eval` entries must lie within `t_span`.")
                y_interp = np.column_stack([np.interp(te_arr, res.t, res.y[:, j]) for j in range(res.y.shape[1])])
                t_out, y_out = te_arr, y_interp
            return Trajectory(t_out, y_out, list(self.states), res.scheme, res.nfev, ok,
                              msg, dt=res.dt, newton_failures=res.newton_failures, warnings=warns)
        te = np.linspace(t_span[0], t_span[1], (n_out or 200) + 1) if t_eval is None else np.asarray(t_eval)
        kw: Dict[str, object] = {"jac": self.jac_function(params)} if method in _IMPLICIT_SCIPY else {}
        ev_names, ev_fns = self._compile_events(events, params)
        if ev_fns:
            kw["events"] = ev_fns
        sol = solve_ivp(f, t_span, y0v, method=method, t_eval=te, rtol=rtol, atol=atol, max_step=max_step, **kw)
        ok = bool(sol.success)
        msg = str(sol.message)
        warns = [] if ok else [f"ODE solver '{method}' did not succeed: {msg}"]
        if raise_on_failure and not ok:
            raise RuntimeError(msg)
        t_ev_map: Dict[str, np.ndarray] = {}
        y_ev_map: Dict[str, np.ndarray] = {}
        if ev_names and sol.t_events is not None:
            for i, ename in enumerate(ev_names):
                t_ev_map[ename] = np.asarray(sol.t_events[i], dtype=float)
                y_ev_map[ename] = np.asarray(sol.y_events[i], dtype=float)
        return Trajectory(
            sol.t, sol.y.T, list(self.states), method, int(sol.nfev), ok,
            msg, njev=int(getattr(sol, "njev", 0) or 0), warnings=warns,
            t_events=t_ev_map, y_events=y_ev_map,
        )

    def step(self, t0: float, t1: float, y0, params=None, method: str = "RK45",
             rtol: float = 1e-10, atol: float = 1e-12) -> np.ndarray:
        """Advance from t0 to t1 and return only the end state (used by co-simulation)."""
        f = self.rhs_function(params)
        kw = {"jac": self.jac_function(params)} if method in _IMPLICIT_SCIPY else {}
        sol = solve_ivp(f, (t0, t1), self._y0(y0), method=method, rtol=rtol, atol=atol, **kw)
        if not sol.success:
            raise RuntimeError(f"Sub-step integration failed: {sol.message}")
        return sol.y[:, -1]

    def eigenvalues(self, y, params=None, t: float = 0.0) -> np.ndarray:
        return np.linalg.eigvals(self.jac_function(params)(t, self._y0(y)))

    # ------------------------------------------------------------------ symbolic checks
    def time_derivative(self, expr: ExprLike) -> sp.Expr:
        """d/dt of a state function along solutions: dI/dt = grad(I) . f + dI/dt|explicit."""
        e = parse_expr(expr, self._symbols)
        total = sp.diff(e, self.t)
        for s, rhs in zip(self.state_syms, self.rhs):
            total += sp.diff(e, s) * rhs
        return sp.simplify(total)

    def check_invariant(self, expr: ExprLike) -> Tuple[bool, sp.Expr]:
        """True if `expr` is conserved by the model equations (dI/dt simplifies to 0)."""
        d = self.time_derivative(expr)
        return bool(d == 0), d

    def evaluate(self, expr: ExprLike, traj: Trajectory, params=None) -> np.ndarray:
        """Evaluate a function of (t, states, params) along a trajectory."""
        e = parse_expr(expr, self._symbols)
        used = names(e)
        unknown = used - set(self.states) - set(self.params) - {self.indep}
        if unknown:
            raise KeyError(f"Unknown symbols {sorted(unknown)} in expression '{expr}'.")
        pnames = [p for p in self.params if p in used]
        fn = sp.lambdify([self.t, *self.state_syms, *[real_symbol(p) for p in pnames]], e, "numpy")
        params = dict(params or {})
        missing = [p for p in pnames if p not in params]
        if missing:
            raise KeyError(f"Missing values for parameters {missing}.")
        pvals = []
        for p in pnames:
            pv = params[p]
            if callable(pv):
                try:
                    arr_p = np.broadcast_to(np.asarray(pv(traj.t), dtype=float), traj.t.shape).copy()
                except (TypeError, ValueError):
                    arr_p = np.array([float(pv(float(ti))) for ti in traj.t], dtype=float)
                pvals.append(arr_p)
            else:
                pvals.append(float(pv))
        out = fn(traj.t, *traj.y.T, *pvals)
        return np.broadcast_to(np.asarray(out, dtype=float), traj.t.shape).copy()
