"""
Numerical solver and empirical calibration engine for formulated symbolic equations.

Capabilities:
  1. Vectorized compilation of symbolic CandidateEquation / Expr to NumPy callables.
  2. Exact OLS (for parameter-linear models) & Trust-Region Nonlinear Least Squares
     with analytic SymPy Jacobians (for Pi-group exponents and nested nonlinear models).
  3. Differential regression fitting (fitting dg/df = beta_0 + sum beta_i dx_i/df directly).
  4. Automated BIC / AIC / RMSE candidate model selection & ranking.
  5. Numerical ODE integration (scipy.integrate.solve_ivp) for non-elementary rate laws.
  6. Local dimensionless elasticity / sensitivity analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import sympy as sp

from .symbolic import CandidateEquation, DerivationResult


@dataclass
class FitResult:
    """Empirical calibration statistics and calibrated symbolic equation."""

    candidate_name: str
    params: Dict[str, float]
    std_errors: Dict[str, float]
    r2: float
    rmse: float
    mae: float
    aic: float
    bic: float
    n_obs: int
    calibrated_equation: sp.Eq
    predictions: np.ndarray
    residuals: np.ndarray
    converged: bool = True
    message: str = ""

    def summary(self) -> str:
        p_str = ", ".join(
            f"{k}={v:.6g} (±{self.std_errors.get(k, float('nan')):.3g})"
            for k, v in self.params.items()
        )
        return (
            f"[{self.candidate_name}] R²={self.r2:.6f}, RMSE={self.rmse:.6g}, BIC={self.bic:.3f}\n"
            f"  Params: {p_str}\n"
            f"  Calibrated: {self.calibrated_equation.lhs} = {self.calibrated_equation.rhs}"
        )


@dataclass
class ODESolution:
    """Result of numerically integrating a formulated rate equation dg/df = RHS."""

    wrt_values: np.ndarray
    target_values: np.ndarray
    rate_values: np.ndarray
    success: bool
    message: str


@dataclass
class SheddingEventSpec:
    """
    Threshold failure trigger for hybrid accretion-shedding ODE integration.

    Event fires when `margin_expr(t, y)` crosses 0 upward (`direction = +1`), e.g.:
      - Interfacial debonding: G_interface - G_a(t) == 0  (shed_fraction ~ 0.95 for slab detachment)
      - Cohesive fracture:     K_I - K_Ic == 0            (shed_fraction ~ 0.40 for chunk fracture)
    """

    name: str
    margin_expr: Union[sp.Expr, Callable[[float, float], float]]
    shed_fraction: float = 0.95
    direction: float = 1.0


@dataclass
class SheddingRecord:
    """Single discrete shedding/fracture detachment event along a hybrid trajectory."""

    time: float
    mode: str
    pre_value: float
    post_value: float
    shed_amount: float


@dataclass
class HybridODESolution:
    """Piecewise-smooth trajectory with discontinuous threshold-triggered shedding resets."""

    wrt_values: np.ndarray
    target_values: np.ndarray
    events: List[SheddingRecord]
    success: bool
    message: str

    @property
    def n_events(self) -> int:
        return len(self.events)


class NumericalSolver:
    """Numerical evaluation, parameter estimation, model selection, and ODE integration."""

    @staticmethod
    def _extract_expr_and_params(
        model: Union[CandidateEquation, DerivationResult, sp.Eq, sp.Expr],
        free_params: Optional[Sequence[Union[str, sp.Symbol]]] = None,
    ) -> Tuple[str, sp.Symbol, sp.Expr, List[sp.Symbol]]:
        if isinstance(model, DerivationResult):
            cand = model.primary
            lhs = cand.equation.lhs
            rhs = cand.rhs
            params = list(cand.free_params)
            name = cand.name
        elif isinstance(model, CandidateEquation):
            lhs = model.equation.lhs
            rhs = model.rhs
            params = list(model.free_params)
            name = model.name
        elif isinstance(model, sp.Eq):
            lhs = model.lhs
            rhs = model.rhs
            params = []
            name = "custom_eq"
        elif isinstance(model, sp.Expr):
            lhs = sp.Symbol("y")
            rhs = model
            params = []
            name = "custom_expr"
        else:
            raise TypeError(f"Unsupported model type: {type(model)}")

        if free_params is not None:
            sym_pool = {str(s): s for s in rhs.free_symbols}
            params = [
                sym_pool[p] if isinstance(p, str) and p in sym_pool else (sp.Symbol(p, real=True) if isinstance(p, str) else p)
                for p in free_params
            ]

        return name, lhs, rhs, params  # type: ignore[return-value]

    @classmethod
    def compile_model(
        cls,
        model: Union[CandidateEquation, DerivationResult, sp.Eq, sp.Expr],
        var_order: Sequence[str],
        param_order: Optional[Sequence[str]] = None,
    ) -> Callable[..., np.ndarray]:
        """Compile symbolic RHS into a fast vectorized NumPy function f(*vars, *params)."""
        _, _, rhs, default_params = cls._extract_expr_and_params(model)
        sym_lookup = {str(s): s for s in rhs.free_symbols}
        var_syms = [sym_lookup.get(v, sp.Symbol(v, real=True)) for v in var_order]
        if param_order is None:
            p_syms = default_params
        else:
            p_syms = [sym_lookup.get(p, sp.Symbol(p, real=True)) for p in param_order]
        fn = sp.lambdify(var_syms + p_syms, rhs, modules="numpy")
        return fn

    @classmethod
    def evaluate(
        cls,
        model: Union[CandidateEquation, DerivationResult, sp.Eq, sp.Expr],
        data: Mapping[str, Union[float, np.ndarray, Sequence[float]]],
        params: Optional[Mapping[str, float]] = None,
    ) -> np.ndarray:
        """Evaluate symbolic equation over scalar or array input dictionary."""
        _, _, rhs, _ = cls._extract_expr_and_params(model)
        merged: Dict[str, np.ndarray] = {}
        max_len = 1
        for k, v in data.items():
            arr = np.asarray(v, dtype=float)
            if arr.ndim > 0 and arr.size > max_len:
                max_len = arr.size
            merged[k] = arr
        if params:
            for k, v in params.items():
                merged[k] = np.asarray(v, dtype=float)

        ordered_syms = sorted(rhs.free_symbols, key=str)
        missing = [str(s) for s in ordered_syms if str(s) not in merged]
        if missing:
            raise KeyError(f"Missing values for symbols: {missing}")

        fn = sp.lambdify(ordered_syms, rhs, modules="numpy")
        raw = fn(*[merged[str(s)] for s in ordered_syms])
        out = np.asarray(raw, dtype=float)
        if out.ndim == 0 and max_len > 1:
            out = np.full(max_len, float(out))
        return out

    @staticmethod
    def _is_linear_in_params(expr: sp.Expr, param_syms: Sequence[sp.Symbol]) -> bool:
        if not param_syms:
            return True
        for p in param_syms:
            d1 = sp.diff(expr, p)
            if any(d1.has(q) for q in param_syms):
                return False
        return True

    @classmethod
    def fit(
        cls,
        model: Union[CandidateEquation, DerivationResult, sp.Eq, sp.Expr],
        data: Mapping[str, Union[float, Sequence[float], np.ndarray]],
        target_col: Optional[str] = None,
        free_params: Optional[Sequence[Union[str, sp.Symbol]]] = None,
        fixed_params: Optional[Mapping[str, float]] = None,
        initial_guess: Optional[Mapping[str, float]] = None,
        bounds: Optional[Mapping[str, Tuple[float, float]]] = None,
        loss: str = "linear",
        clean_tol: float = 1e-10,
    ) -> FitResult:
        """
        Calibrate unknown constants / Pi-group exponents from observational data.

        Automatically dispatches to exact SVD Ordinary Least Squares when linear in
        parameters (without bounds), and analytic-Jacobian Trust-Region Nonlinear
        Least Squares otherwise.
        """
        cand_name, lhs_sym, rhs, param_syms = cls._extract_expr_and_params(model, free_params)

        if fixed_params:
            sym_map = {str(s): s for s in rhs.free_symbols}
            subs_fixed = {sym_map[k]: sp.Float(v) for k, v in fixed_params.items() if k in sym_map}
            rhs = rhs.subs(subs_fixed)
            param_syms = [p for p in param_syms if str(p) not in fixed_params]

        t_name = target_col or str(lhs_sym)
        if t_name not in data:
            raise KeyError(f"Target column '{t_name}' not found in data keys {list(data.keys())}.")

        y = np.asarray(data[t_name], dtype=float).reshape(-1)
        n_obs = y.size
        if n_obs == 0:
            raise ValueError("Cannot fit on empty dataset.")

        data_syms = [s for s in sorted(rhs.free_symbols, key=str) if s not in param_syms]
        missing_cols = [str(s) for s in data_syms if str(s) not in data]
        if missing_cols:
            raise KeyError(f"Missing data columns for variables {missing_cols}.")

        x_arrays = [
            np.broadcast_to(np.asarray(data[str(s)], dtype=float), (n_obs,)).copy()
            for s in data_syms
        ]

        k_params = len(param_syms)
        if k_params == 0:
            fn_eval = sp.lambdify(data_syms, rhs, modules="numpy")
            preds = np.broadcast_to(np.asarray(fn_eval(*x_arrays), dtype=float), (n_obs,)).copy()
            return cls._build_fit_result(
                cand_name, lhs_sym, rhs, {}, {}, y, preds, True, "No free parameters.", clean_tol
            )

        is_lin = cls._is_linear_in_params(rhs, param_syms) and bounds is None and loss == "linear"

        if is_lin:
            # Exact Ordinary Least Squares: rhs = base(x) + sum_j p_j * phi_j(x)
            zero_subs = {p: 0 for p in param_syms}
            base_expr = sp.simplify(rhs.subs(zero_subs))
            base_fn = sp.lambdify(data_syms, base_expr, modules="numpy")
            base_vec = np.broadcast_to(np.asarray(base_fn(*x_arrays), dtype=float), (n_obs,)).copy()

            cols = []
            for p in param_syms:
                phi_expr = sp.simplify(sp.diff(rhs, p))
                phi_fn = sp.lambdify(data_syms, phi_expr, modules="numpy")
                col = np.broadcast_to(np.asarray(phi_fn(*x_arrays), dtype=float), (n_obs,)).copy()
                cols.append(col)
            X_mat = np.column_stack(cols)
            y_adj = y - base_vec

            theta_hat, _, _, _ = np.linalg.lstsq(X_mat, y_adj, rcond=None)
            preds = base_vec + X_mat @ theta_hat
            residuals = y - preds
            dof = max(n_obs - k_params, 1)
            sigma2 = float(np.sum(residuals ** 2) / dof)
            cov = sigma2 * np.linalg.pinv(X_mat.T @ X_mat)
            ses = np.sqrt(np.maximum(np.diag(cov), 0.0))

            p_dict = {str(p): float(val) for p, val in zip(param_syms, theta_hat)}
            se_dict = {str(p): float(se) for p, se in zip(param_syms, ses)}
            return cls._build_fit_result(
                cand_name, lhs_sym, rhs, p_dict, se_dict, y, preds, True, "Exact OLS fit.", clean_tol
            )

        # Nonlinear Least Squares with analytic SymPy Jacobian & multi-start
        all_syms = data_syms + list(param_syms)
        f_pred = sp.lambdify(all_syms, rhs, modules="numpy")
        jac_exprs = [sp.diff(rhs, p) for p in param_syms]
        f_jac = [sp.lambdify(all_syms, je, modules="numpy") for je in jac_exprs]

        lb = np.array(
            [
                bounds[str(p)][0]
                if (bounds and str(p) in bounds)
                else (1e-9 if str(p).startswith("gam_") else -np.inf)
                for p in param_syms
            ],
            dtype=float,
        )
        ub = np.array(
            [bounds[str(p)][1] if (bounds and str(p) in bounds) else np.inf for p in param_syms],
            dtype=float,
        )

        def residual_fn(theta: np.ndarray) -> np.ndarray:
            with np.errstate(all="ignore"):
                pred = np.broadcast_to(
                    np.asarray(f_pred(*x_arrays, *theta), dtype=float), (n_obs,)
                )
            res = pred - y
            if not np.all(np.isfinite(res)):
                return np.full(n_obs, 1e12)
            return res

        def jacobian_fn(theta: np.ndarray) -> np.ndarray:
            cols_j = []
            with np.errstate(all="ignore"):
                for jf in f_jac:
                    c = np.broadcast_to(
                        np.asarray(jf(*x_arrays, *theta), dtype=float), (n_obs,)
                    ).copy()
                    if not np.all(np.isfinite(c)):
                        c = np.nan_to_num(c, nan=0.0, posinf=1e6, neginf=-1e6)
                    cols_j.append(c)
            return np.column_stack(cols_j)

        starts: List[np.ndarray] = []
        if initial_guess:
            g0 = np.array([initial_guess.get(str(p), 1.0) for p in param_syms], dtype=float)
            starts.append(np.clip(g0, lb + 1e-12, ub - 1e-12))
        else:
            # Smart defaults: offsets near 0, gam_* positive, scaling kappas near 1, exponents near {-0.5, 0.5, 1.0}
            s1 = np.array(
                [
                    0.0
                    if str(p).endswith("_0")
                    else (0.5 if str(p).startswith("nu_") else (0.2 if str(p).startswith("gam_") else 1.0))
                    for p in param_syms
                ],
                dtype=float,
            )
            s2 = np.array(
                [
                    0.0
                    if str(p).endswith("_0")
                    else (-0.5 if str(p).startswith("nu_") else (1.0 if str(p).startswith("gam_") else 0.5))
                    for p in param_syms
                ],
                dtype=float,
            )
            s3 = np.array(
                [
                    0.0
                    if str(p).endswith("_0")
                    else (1.5 if str(p).startswith("nu_") else (0.05 if str(p).startswith("gam_") else 1.0))
                    for p in param_syms
                ],
                dtype=float,
            )
            starts.extend(
                [
                    np.clip(s1, lb + 1e-12, ub - 1e-12),
                    np.clip(s2, lb + 1e-12, ub - 1e-12),
                    np.clip(s3, lb + 1e-12, ub - 1e-12),
                ]
            )

        best_opt = None
        for x0 in starts:
            opt = least_squares(
                residual_fn,
                x0=x0,
                jac=jacobian_fn,
                bounds=(lb, ub),
                loss=loss,
                max_nfev=2000,
            )
            if best_opt is None or opt.cost < best_opt.cost:
                best_opt = opt

        assert best_opt is not None
        theta_hat = best_opt.x
        preds = np.broadcast_to(
            np.asarray(f_pred(*x_arrays, *theta_hat), dtype=float), (n_obs,)
        ).copy()
        residuals = y - preds
        dof = max(n_obs - k_params, 1)
        sigma2 = float(np.sum(residuals ** 2) / dof)
        J = best_opt.jac
        cov = sigma2 * np.linalg.pinv(J.T @ J)
        ses = np.sqrt(np.maximum(np.diag(cov), 0.0))

        p_dict = {str(p): float(val) for p, val in zip(param_syms, theta_hat)}
        se_dict = {str(p): float(se) for p, se in zip(param_syms, ses)}
        return cls._build_fit_result(
            cand_name,
            lhs_sym,
            rhs,
            p_dict,
            se_dict,
            y,
            preds,
            bool(best_opt.success),
            best_opt.message,
            clean_tol,
        )

    @classmethod
    def fit_differential(
        cls,
        result: DerivationResult,
        data: Mapping[str, Union[Sequence[float], np.ndarray]],
        wrt_col: Optional[str] = None,
    ) -> FitResult:
        """
        Calibrate parameters directly on the differential equation dg/df = RHS
        using numerical gradients d(target)/d(wrt) when direct rate measurements are absent.
        """
        if result.differential_equation is None:
            raise ValueError("DerivationResult has no differential_equation to fit.")
        w_name = wrt_col or result.wrt
        if w_name is None or w_name not in data:
            raise KeyError(f"Coordinate '{w_name}' required in data for differential regression.")
        t_name = result.target
        rate_col = f"d{t_name}_d{w_name}"

        aug_data = {k: np.asarray(v, dtype=float) for k, v in data.items()}
        if rate_col not in aug_data:
            aug_data[rate_col] = np.gradient(aug_data[t_name], aug_data[w_name])

        lhs_rate_sym = sp.Symbol(rate_col, real=True)
        diff_rhs = result.differential_equation.rhs
        free_in_diff = [p for p in result.free_parameters if diff_rhs.has(p)]
        eq_for_fit = sp.Eq(lhs_rate_sym, diff_rhs)
        return cls.fit(eq_for_fit, aug_data, target_col=rate_col, free_params=free_in_diff)

    @classmethod
    def rank_candidates(
        cls,
        result: DerivationResult,
        data: Mapping[str, Union[float, Sequence[float], np.ndarray]],
        target_col: Optional[str] = None,
        criterion: str = "bic",
    ) -> List[FitResult]:
        """Fit all candidate equations in a DerivationResult and rank by BIC, AIC, or RMSE."""
        fits: List[FitResult] = []
        for cand in result.candidates:
            try:
                fr = cls.fit(cand, data, target_col=target_col)
                fits.append(fr)
            except Exception:
                continue

        key_map = {
            "bic": lambda r: r.bic,
            "aic": lambda r: r.aic,
            "rmse": lambda r: r.rmse,
            "r2": lambda r: -r.r2,
        }
        if criterion not in key_map:
            raise ValueError(f"Unsupported ranking criterion '{criterion}'. Choose from {list(key_map)}.")
        fits.sort(key=key_map[criterion])
        return fits

    @classmethod
    def select_sparse_pi(
        cls,
        result: DerivationResult,
        data: Mapping[str, Union[float, Sequence[float], np.ndarray]],
        target_col: Optional[str] = None,
        max_groups: int = 2,
        form: str = "power",
        criterion: str = "bic",
    ) -> List[FitResult]:
        """
        Sparse Buckingham-Pi subset selection for high-nullity multi-physics systems.

        Evaluates Pi-subsets of cardinality 0..max_groups (using 'power', 'hill', or 'exp_sat'
        modulation) and ranks parsimonious laws by BIC/AIC/RMSE to prevent collinearity.
        """
        all_pis: List[sp.Expr] = []
        for pis in result.pi_groups.values():
            for p in pis:
                if p not in all_pis:
                    all_pis.append(p)

        base_cand = result.primary
        lhs_sym = base_cand.equation.lhs
        init_sym = sp.Symbol(f"{result.target}_0", real=True)
        has_offset = init_sym in base_cand.free_params
        core_expr = sp.simplify(base_cand.rhs - init_sym) if has_offset else base_cand.rhs
        offset_expr = init_sym if has_offset else sp.Integer(0)

        subset_cands: List[CandidateEquation] = [base_cand]
        m = len(all_pis)
        cap = min(max_groups, m)

        for r in range(1, cap + 1):
            for idxs in combinations(range(m), r):
                mod_factor = sp.Integer(1)
                extra_params: List[sp.Symbol] = []
                label_parts = [f"Pi{i + 1}" for i in idxs]
                for i in idxs:
                    pi_e = all_pis[i]
                    nu_s = sp.Symbol(f"nu_{i + 1}", real=True)
                    if form == "power":
                        mod_factor *= pi_e ** nu_s
                        extra_params.append(nu_s)
                    elif form == "hill":
                        gam_s = sp.Symbol(f"gam_{i + 1}", positive=True, real=True)
                        scaled = gam_s * (pi_e ** nu_s)
                        mod_factor *= scaled / (sp.Integer(1) + scaled)
                        extra_params.extend([gam_s, nu_s])
                    elif form == "exp_sat":
                        gam_s = sp.Symbol(f"gam_{i + 1}", positive=True, real=True)
                        scaled = gam_s * (pi_e ** nu_s)
                        mod_factor *= sp.Integer(1) - sp.exp(-scaled)
                        extra_params.extend([gam_s, nu_s])
                    else:
                        raise ValueError(f"Unsupported form '{form}'; choose 'power', 'hill', or 'exp_sat'.")

                rhs = offset_expr + core_expr * mod_factor
                eq = sp.Eq(lhs_sym, rhs)
                free_p = [p for p in (list(base_cand.free_params) + extra_params) if rhs.has(p)]
                subset_cands.append(
                    CandidateEquation(
                        name=f"sparse_{form}_{'_'.join(label_parts)}",
                        kind=f"sparse_{form}",
                        equation=eq,
                        rhs=rhs,
                        free_params=free_p,
                        pi_groups=[all_pis[i] for i in idxs],
                        is_exact_state=base_cand.is_exact_state,
                    )
                )

        fits: List[FitResult] = []
        for cand in subset_cands:
            try:
                fits.append(cls.fit(cand, data, target_col=target_col))
            except Exception:
                continue

        key_map = {
            "bic": lambda r: r.bic,
            "aic": lambda r: r.aic,
            "rmse": lambda r: r.rmse,
            "r2": lambda r: -r.r2,
        }
        fits.sort(key=key_map[criterion])
        return fits

    @classmethod
    def integrate_hybrid_ode(
        cls,
        rate_expr_or_result: Union[DerivationResult, sp.Eq, sp.Expr],
        wrt: str,
        span: Tuple[float, float],
        y0: float,
        events: Sequence[SheddingEventSpec],
        context: Optional[Mapping[str, Union[float, Callable[[float], float]]]] = None,
        params: Optional[Mapping[str, float]] = None,
        target_name: Optional[str] = None,
        method: str = "RK45",
        dt_max: float = 0.5,
        max_events: int = 100,
    ) -> HybridODESolution:
        """
        Integrate piecewise-smooth accretion ODEs with discrete threshold-triggered shedding events.

        Supports competing failure modes (e.g. interfacial adhesive slab detachment vs. cohesive
        bulk chunk fracture) and records every detachment timestamp, mode, and reset jump.
        """
        if isinstance(rate_expr_or_result, DerivationResult):
            if rate_expr_or_result.differential_equation is None:
                raise ValueError("DerivationResult has no differential_equation.")
            rhs = rate_expr_or_result.differential_equation.rhs
            t_name = target_name or rate_expr_or_result.target
        elif isinstance(rate_expr_or_result, sp.Eq):
            rhs = rate_expr_or_result.rhs
            t_name = target_name or "y"
        else:
            rhs = rate_expr_or_result
            t_name = target_name or "y"

        ctx = dict(context or {})
        if params:
            ctx.update(params)

        def build_scalar_eval(expr_or_fn: Union[sp.Expr, Callable[[float, float], float]]) -> Callable[[float, float], float]:
            if callable(expr_or_fn):
                return expr_or_fn
            syms = sorted(expr_or_fn.free_symbols, key=str)
            s_names = [str(s) for s in syms]
            compiled = sp.lambdify(syms, expr_or_fn, modules="numpy")

            def _eval(f_val: float, g_val: float) -> float:
                args = []
                for nm in s_names:
                    if nm == wrt:
                        args.append(f_val)
                    elif nm == t_name:
                        args.append(g_val)
                    elif nm in ctx:
                        val = ctx[nm]
                        args.append(float(val(f_val)) if callable(val) else float(val))
                    else:
                        raise KeyError(f"Symbol '{nm}' missing from hybrid ODE context/params.")
                return float(compiled(*args))

            return _eval

        eval_rhs = build_scalar_eval(rhs)
        compiled_events = []
        for spec in events:
            ev_eval = build_scalar_eval(spec.margin_expr)

            def make_ev(fn_ev: Callable[[float, float], float], dir_val: float) -> Callable[[float, Sequence[float]], float]:
                def _scipy_ev(t_v: float, y_v: Sequence[float]) -> float:
                    return fn_ev(float(t_v), float(y_v[0]))

                _scipy_ev.terminal = True  # type: ignore[attr-defined]
                _scipy_ev.direction = dir_val  # type: ignore[attr-defined]
                return _scipy_ev

            compiled_events.append(make_ev(ev_eval, spec.direction))

        t_curr = float(span[0])
        t_end = float(span[1])
        y_curr = float(y0)
        t_segments: List[np.ndarray] = []
        y_segments: List[np.ndarray] = []
        records: List[SheddingRecord] = []

        while t_curr < t_end - 1e-11 and len(records) < max_events:
            n_pts = max(int(np.ceil((t_end - t_curr) / max(dt_max * 0.5, 1e-3))) + 2, 10)
            t_grid = np.linspace(t_curr, t_end, n_pts)
            sol = solve_ivp(
                fun=lambda tv, yv: [eval_rhs(float(tv), float(yv[0]))],
                t_span=(t_curr, t_end),
                y0=[y_curr],
                method=method,
                t_eval=t_grid,
                events=compiled_events if compiled_events else None,
                max_step=dt_max,
                rtol=1e-8,
                atol=1e-10,
            )
            if not sol.success and sol.status != 1:
                return HybridODESolution(
                    wrt_values=np.concatenate(t_segments) if t_segments else sol.t,
                    target_values=np.concatenate(y_segments) if y_segments else sol.y[0],
                    events=records,
                    success=False,
                    message=sol.message,
                )

            if sol.status == 1 and sol.t_events is not None:
                # Find earliest triggered event
                fired_idx = -1
                t_ev = t_end
                y_pre = y_curr
                for idx, tev_arr in enumerate(sol.t_events):
                    if tev_arr.size > 0 and float(tev_arr[0]) <= t_ev:
                        fired_idx = idx
                        t_ev = float(tev_arr[0])
                        y_pre = float(sol.y_events[idx][0][0])

                seg_t = np.append(sol.t[sol.t < t_ev], t_ev)
                seg_y = np.append(sol.y[0][sol.t < t_ev], y_pre)
                t_segments.append(seg_t)
                y_segments.append(seg_y)

                fired_spec = events[fired_idx]
                frac = float(np.clip(fired_spec.shed_fraction, 0.0, 1.0))
                y_post = (1.0 - frac) * y_pre
                records.append(
                    SheddingRecord(
                        time=t_ev,
                        mode=fired_spec.name,
                        pre_value=y_pre,
                        post_value=y_post,
                        shed_amount=y_pre - y_post,
                    )
                )
                # Append immediate post-detachment state for crisp sawtooth visualization
                t_segments.append(np.array([t_ev]))
                y_segments.append(np.array([y_post]))
                y_curr = y_post
                t_curr = t_ev + 1e-9
            else:
                t_segments.append(sol.t)
                y_segments.append(sol.y[0])
                break

        return HybridODESolution(
            wrt_values=np.concatenate(t_segments) if t_segments else np.array([span[0]]),
            target_values=np.concatenate(y_segments) if y_segments else np.array([y0]),
            events=records,
            success=True,
            message=f"Completed with {len(records)} shedding event(s).",
        )

    @classmethod
    def integrate_ode(
        cls,
        rate_expr_or_result: Union[DerivationResult, sp.Eq, sp.Expr],
        wrt: str,
        span: Tuple[float, float],
        y0: float,
        context: Optional[Mapping[str, Union[float, Callable[[float], float]]]] = None,
        params: Optional[Mapping[str, float]] = None,
        target_name: Optional[str] = None,
        method: str = "RK45",
        num_points: int = 200,
    ) -> ODESolution:
        """
        Numerically integrate dg/df = RHS(f, g, ...) over `span = (f_start, f_end)`.

        `context` may provide either constant floats or callable trajectories `x(f)`
        for auxiliary variables (such as variable mass m(t) or angular jerk zeta(t)).
        """
        if isinstance(rate_expr_or_result, DerivationResult):
            if rate_expr_or_result.differential_equation is None:
                raise ValueError("DerivationResult has no differential_equation.")
            rhs = rate_expr_or_result.differential_equation.rhs
            t_name = target_name or rate_expr_or_result.target
        elif isinstance(rate_expr_or_result, sp.Eq):
            rhs = rate_expr_or_result.rhs
            t_name = target_name or "y"
        else:
            rhs = rate_expr_or_result
            t_name = target_name or "y"

        ctx = dict(context or {})
        if params:
            ctx.update(params)

        ordered_syms = sorted(rhs.free_symbols, key=str)
        sym_names = [str(s) for s in ordered_syms]
        fn_rhs = sp.lambdify(ordered_syms, rhs, modules="numpy")

        def eval_rhs(f_val: float, g_val: float) -> float:
            args = []
            for name in sym_names:
                if name == wrt:
                    args.append(f_val)
                elif name == t_name:
                    args.append(g_val)
                elif name in ctx:
                    v = ctx[name]
                    args.append(float(v(f_val)) if callable(v) else float(v))
                else:
                    raise KeyError(f"Symbol '{name}' missing from ODE context/params.")
            return float(fn_rhs(*args))

        f_eval = np.linspace(span[0], span[1], num_points)
        sol = solve_ivp(
            fun=lambda f_v, y_v: [eval_rhs(float(f_v), float(y_v[0]))],
            t_span=span,
            y0=[float(y0)],
            method=method,
            t_eval=f_eval,
            rtol=1e-8,
            atol=1e-10,
        )
        g_vals = sol.y[0] if sol.y.size else np.array([], dtype=float)
        rates = np.array([eval_rhs(float(fv), float(gv)) for fv, gv in zip(sol.t, g_vals)], dtype=float)
        return ODESolution(
            wrt_values=sol.t,
            target_values=g_vals,
            rate_values=rates,
            success=bool(sol.success),
            message=sol.message,
        )

    @classmethod
    def elasticities(
        cls,
        model: Union[CandidateEquation, DerivationResult, sp.Eq, sp.Expr],
        point: Mapping[str, float],
        params: Optional[Mapping[str, float]] = None,
    ) -> Dict[str, float]:
        """Compute dimensionless local elasticities E_{g, x} = (x / g) * (dg / dx) at `point`."""
        _, _, rhs, _ = cls._extract_expr_and_params(model)
        full_pt = dict(point)
        if params:
            full_pt.update(params)

        g_val = float(cls.evaluate(rhs, full_pt))
        if g_val == 0.0:
            raise ZeroDivisionError("Target evaluates to 0 at given point; dimensionless elasticity undefined.")

        sym_map = {str(s): s for s in rhs.free_symbols}
        out: Dict[str, float] = {}
        for var_name, x_val in point.items():
            if var_name not in sym_map:
                out[var_name] = 0.0
                continue
            d_expr = sp.diff(rhs, sym_map[var_name])
            dg_dx = float(cls.evaluate(d_expr, full_pt))
            out[var_name] = (float(x_val) / g_val) * dg_dx
        return out

    @staticmethod
    def _build_fit_result(
        cand_name: str,
        lhs_sym: sp.Symbol,
        rhs: sp.Expr,
        p_dict: Dict[str, float],
        se_dict: Dict[str, float],
        y: np.ndarray,
        preds: np.ndarray,
        converged: bool,
        message: str,
        clean_tol: float,
    ) -> FitResult:
        n_obs = y.size
        k = len(p_dict)
        residuals = y - preds
        sse = float(np.sum(residuals ** 2))
        sst = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - (sse / sst) if sst > 0 else (1.0 if sse < 1e-24 else 0.0)
        rmse = float(np.sqrt(sse / n_obs))
        mae = float(np.mean(np.abs(residuals)))

        mse_floor = max(sse / n_obs, 1e-30)
        log_lik = -0.5 * n_obs * (np.log(2 * np.pi * mse_floor) + 1.0)
        aic = float(2 * k - 2 * log_lik)
        bic = float(k * np.log(max(n_obs, 1)) - 2 * log_lik)

        sym_map = {str(s): s for s in rhs.free_symbols}
        subs_cal: Dict[sp.Symbol, sp.Expr] = {}
        for k_name, val in p_dict.items():
            if k_name in sym_map:
                cleaned = 0.0 if abs(val) < clean_tol else val
                # Snap near-exact small-denominator rationals (e.g. 1/2, -1/2, 3/2, 5); keep clean Float otherwise
                frac = Fraction(cleaned).limit_denominator(24)
                if abs(float(frac) - cleaned) <= clean_tol:
                    subs_cal[sym_map[k_name]] = sp.Rational(frac.numerator, frac.denominator)
                else:
                    subs_cal[sym_map[k_name]] = sp.Float(cleaned, 6)

        cal_rhs = sp.simplify(rhs.subs(subs_cal))
        cal_eq = sp.Eq(lhs_sym, cal_rhs)

        return FitResult(
            candidate_name=cand_name,
            params=p_dict,
            std_errors=se_dict,
            r2=r2,
            rmse=rmse,
            mae=mae,
            aic=aic,
            bic=bic,
            n_obs=n_obs,
            calibrated_equation=cal_eq,
            predictions=preds,
            residuals=residuals,
            converged=converged,
            message=message,
        )
