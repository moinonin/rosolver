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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.stats import t as student_t
import sympy as sp

from .results import ResultMixin
from .symbolic import CandidateEquation, DerivationResult
from .templates import TEMPLATE_REGISTRY


@dataclass
class IdentifiabilityReport(ResultMixin):
    """Jacobian conditioning, rank, and parameter-correlation identifiability diagnostics."""

    identifiable: bool
    condition_number: float
    singular_values: np.ndarray
    numerical_rank: int
    n_params: int
    max_abs_correlation: float
    correlation_matrix: Optional[np.ndarray] = None
    collinear_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class RejectedCandidate:
    """Candidate formulation that failed calibration or screening, with recorded reason."""

    candidate_name: str
    reason: str
    stage: str = "fitting"


@dataclass
class FitResult(ResultMixin):
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
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    covariance_matrix: Optional[np.ndarray] = None
    correlation_matrix: Optional[np.ndarray] = None
    identifiability: Optional[IdentifiabilityReport] = None
    weighted: bool = False
    objective_value: float = float("nan")
    n_dropped_missing: int = 0
    optimizer_config: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    admissibility_tier: str = "statistically_supported"

    def summary(self) -> str:
        p_str = ", ".join(
            f"{k}={v:.6g} (±{self.std_errors.get(k, float('nan')):.3g})"
            for k, v in self.params.items()
        )
        method_tag = self.optimizer_config.get("method", "WLS" if self.weighted else "fit")
        id_tag = ""
        if self.identifiability is not None:
            id_tag = f", identifiable={self.identifiability.identifiable}"
        lines = [
            f"[{self.candidate_name}] ({method_tag}{id_tag}) R²={self.r2:.6f}, RMSE={self.rmse:.6g}, BIC={self.bic:.3f}",
            f"  Params: {p_str}",
            f"  Calibrated: {self.calibrated_equation.lhs} = {self.calibrated_equation.rhs}",
        ]
        for w in self.warnings:
            lines.append(f"  Warning: {w}")
        return "\n".join(lines)

    def _repr_markdown_(self) -> str:
        lines = [
            f"### Parameter Calibration: `{self.candidate_name}`",
            "",
            f"- **R²:** {self.r2:.6f} | **RMSE:** {self.rmse:.6g} | **BIC:** {self.bic:.3f} | **Admissibility:** `{self.admissibility_tier}`",
            f"- **Calibrated Equation:** `{self.calibrated_equation.lhs} = {self.calibrated_equation.rhs}`",
            "",
            "| Parameter | Estimate | Std. Error |",
            "| --- | --- | --- |",
        ]
        for k, v in self.params.items():
            se = self.std_errors.get(k, float("nan"))
            lines.append(f"| `{k}` | {v:.6g} | ±{se:.3g} |")
        if self.warnings:
            lines.append("")
            for w in self.warnings:
                lines.append(f"- **Warning:** {w}")
        return "\n".join(lines)


@dataclass
class CandidateRankingReport(ResultMixin):
    """Multi-criterion candidate comparison separating admissibility, statistical fit, and rejections."""

    ranked: List[FitResult]
    rejected: List[RejectedCandidate]
    criterion: str
    categories: Dict[str, List[str]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"=== Candidate Ranking ({self.criterion.upper()}) ==="]
        for idx, fr in enumerate(self.ranked, 1):
            lines.append(f"{idx}. {fr.summary()}")
        if self.rejected:
            lines.append("Rejected candidates:")
            for r in self.rejected:
                lines.append(f"  - [{r.candidate_name}] ({r.stage}): {r.reason}")
        return "\n".join(lines)


@dataclass
class ODESolution(ResultMixin):
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
class HybridODESolution(ResultMixin):
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
            sym_pool = {s.name: s for s in rhs.free_symbols if isinstance(s, sp.Symbol)}
            resolved_p: List[sp.Symbol] = []
            for p in free_params:
                p_name = p.name if isinstance(p, sp.Symbol) else str(p)
                if p_name in sym_pool:
                    resolved_p.append(sym_pool[p_name])
                elif isinstance(p, sp.Symbol):
                    resolved_p.append(p)
                else:
                    resolved_p.append(sp.Symbol(p_name, real=True))
            params = resolved_p

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

    @staticmethod
    def _analyze_identifiability(
        J_w: np.ndarray,
        param_names: Sequence[str],
        pinv_info: np.ndarray,
        n_obs: int,
        cond_threshold: float = 1e8,
        corr_threshold: float = 0.98,
    ) -> IdentifiabilityReport:
        k_params = len(param_names)
        norms = np.linalg.norm(J_w, axis=0)
        zero_cols = [param_names[i] for i, nm in enumerate(norms) if nm <= 1e-15]
        J_norm = J_w / np.where(norms > 1e-15, norms, 1.0)
        s = np.linalg.svd(J_norm, compute_uv=False)
        s_max = float(s[0]) if s.size else 0.0
        numerical_rank = int(np.sum(s > 1e-12 * max(s_max, 1.0)))
        if zero_cols or len(s) < k_params or (s.size > 0 and s[-1] <= 1e-15):
            cond = float("inf")
        else:
            cond = float(s[0] / s[-1])

        diag = np.diag(pinv_info)
        denom = np.sqrt(np.outer(np.maximum(diag, 0.0), np.maximum(diag, 0.0)))
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.where(denom > 1e-30, pinv_info / denom, np.nan)
        for i in range(k_params):
            if diag[i] > 1e-30:
                corr[i, i] = 1.0

        collinear_pairs: List[Tuple[str, str, float]] = []
        max_abs_corr = 0.0
        for i in range(k_params):
            for j in range(i + 1, k_params):
                rij = float(corr[i, j])
                if np.isfinite(rij):
                    abs_r = abs(rij)
                    if abs_r > max_abs_corr:
                        max_abs_corr = abs_r
                    if abs_r >= corr_threshold:
                        collinear_pairs.append((param_names[i], param_names[j], rij))

        warnings: List[str] = []
        for zname in zero_cols:
            warnings.append(f"Parameter '{zname}' has zero local sensitivity (structurally unidentifiable).")
        if n_obs <= k_params:
            warnings.append(
                f"Degrees of freedom exhausted (n_obs={n_obs} <= n_params={k_params}); parameter uncertainty unreliable."
            )
        if numerical_rank < k_params:
            warnings.append(
                f"Jacobian is rank-deficient (rank {numerical_rank} < {k_params} parameters); "
                "parameters are not jointly identifiable."
            )
        elif cond > cond_threshold:
            warnings.append(
                f"Ill-conditioned parameter Jacobian (cond={cond:.2e} > {cond_threshold:.2e})."
            )
        for p1, p2, rij in collinear_pairs:
            warnings.append(f"Near-collinear parameters ({p1}, {p2}): correlation r={rij:+.4f}.")

        return IdentifiabilityReport(
            identifiable=(len(warnings) == 0),
            condition_number=cond,
            singular_values=s,
            numerical_rank=numerical_rank,
            n_params=k_params,
            max_abs_correlation=max_abs_corr,
            correlation_matrix=corr,
            collinear_pairs=collinear_pairs,
            warnings=warnings,
        )

    @staticmethod
    def _confidence_intervals(
        param_names: Sequence[str],
        theta_hat: np.ndarray,
        ses: np.ndarray,
        n_obs: int,
        k_params: int,
        confidence_level: float,
    ) -> Dict[str, Tuple[float, float]]:
        if n_obs <= k_params or k_params == 0:
            return {nm: (float("nan"), float("nan")) for nm in param_names}
        dof = n_obs - k_params
        t_crit = float(student_t.ppf(0.5 * (1.0 + confidence_level), dof))
        return {
            nm: (float(val - t_crit * se), float(val + t_crit * se))
            for nm, val, se in zip(param_names, theta_hat, ses)
        }

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
        weights: Optional[Union[Sequence[float], np.ndarray]] = None,
        sigma_y: Optional[Union[Sequence[float], np.ndarray]] = None,
        missing_policy: str = "raise",
        confidence_level: float = 0.95,
        cond_threshold: float = 1e8,
        corr_threshold: float = 0.98,
    ) -> FitResult:
        """
        Calibrate unknown constants / Pi-group exponents from observational data.

        Automatically dispatches to exact SVD Ordinary / Weighted Least Squares when linear in
        parameters (without bounds), and analytic-Jacobian Trust-Region Nonlinear Least Squares
        otherwise. Evaluates Jacobian conditioning, rank, correlation, and confidence intervals.
        """
        if missing_policy not in ("raise", "drop", "mask", "nan"):
            raise ValueError(f"Invalid missing_policy '{missing_policy}'. Choose 'raise', 'drop', 'mask', or 'nan'.")
        if weights is not None and sigma_y is not None:
            raise ValueError("Provide either `weights` or `sigma_y`, not both.")
        if not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence_level must satisfy 0 < confidence_level < 1.")

        tmpl_obj = (
            TEMPLATE_REGISTRY.get(model.template_name)
            if isinstance(model, CandidateEquation) and model.template_name
            else None
        )
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
        n_raw = y.size
        if n_raw == 0:
            raise ValueError("Cannot fit on empty dataset.")

        data_syms = [s for s in sorted(rhs.free_symbols, key=str) if s not in param_syms]
        missing_cols = [str(s) for s in data_syms if str(s) not in data]
        if missing_cols:
            raise KeyError(f"Missing data columns for variables {missing_cols}.")

        x_arrays = [
            np.broadcast_to(np.asarray(data[str(s)], dtype=float), (n_raw,)).copy()
            for s in data_syms
        ]

        is_weighted = (weights is not None) or (sigma_y is not None)
        if weights is not None:
            w_arr = np.broadcast_to(np.asarray(weights, dtype=float).reshape(-1), (n_raw,)).copy()
        elif sigma_y is not None:
            s_arr = np.broadcast_to(np.asarray(sigma_y, dtype=float).reshape(-1), (n_raw,)).copy()
            w_arr = None
        else:
            w_arr = None
            s_arr = None

        # Check non-finite values across target, predictors, and weights/uncertainties
        bad_cols: List[str] = []
        valid_mask = np.isfinite(y)
        if not np.all(valid_mask):
            bad_cols.append(t_name)
        for s, arr in zip(data_syms, x_arrays):
            col_ok = np.isfinite(arr)
            if not np.all(col_ok):
                bad_cols.append(str(s))
            valid_mask &= col_ok
        if w_arr is not None:
            col_ok = np.isfinite(w_arr)
            if not np.all(col_ok):
                bad_cols.append("weights")
            valid_mask &= col_ok
        if s_arr is not None:
            col_ok = np.isfinite(s_arr)
            if not np.all(col_ok):
                bad_cols.append("sigma_y")
            valid_mask &= col_ok

        n_dropped = int(n_raw - np.count_nonzero(valid_mask))
        pad_mask = valid_mask.copy() if (n_dropped > 0 and missing_policy in ("mask", "nan")) else None
        if n_dropped > 0:
            if missing_policy == "raise":
                raise ValueError(
                    f"Non-finite (NaN/Inf) entries found in column(s) {bad_cols} "
                    f"({n_dropped} row(s)); set missing_policy='drop' or 'mask' to handle incomplete rows."
                )
            y = y[valid_mask]
            x_arrays = [arr[valid_mask] for arr in x_arrays]
            if w_arr is not None:
                w_arr = w_arr[valid_mask]
            if s_arr is not None:
                s_arr = s_arr[valid_mask]

        n_obs = y.size
        if n_obs == 0:
            raise ValueError("No finite observations remain after filtering missing rows.")

        if w_arr is not None:
            if np.any(w_arr <= 0.0):
                raise ValueError("All observation weights must be strictly positive.")
            sqrt_w = np.sqrt(w_arr)
        elif s_arr is not None:
            if np.any(s_arr <= 0.0):
                raise ValueError("All measurement uncertainties sigma_y must be strictly positive.")
            sqrt_w = 1.0 / s_arr
        else:
            sqrt_w = np.ones(n_obs, dtype=float)

        fit_warnings: List[str] = []
        if n_dropped > 0:
            fit_warnings.append(f"Dropped {n_dropped} observation(s) with non-finite values (policy='{missing_policy}').")

        k_params = len(param_syms)
        param_names = [str(p) for p in param_syms]
        if k_params == 0:
            fn_eval = sp.lambdify(data_syms, rhs, modules="numpy")
            preds = np.broadcast_to(np.asarray(fn_eval(*x_arrays), dtype=float), (n_obs,)).copy()
            obj_val = float(np.sum((sqrt_w * (y - preds)) ** 2))
            return cls._build_fit_result(
                cand_name, lhs_sym, rhs, {}, {}, y, preds, True, "No free parameters.", clean_tol,
                weighted=is_weighted, sqrt_w=sqrt_w if is_weighted else None, pad_mask=pad_mask,
                objective_value=obj_val, n_dropped_missing=n_dropped,
                optimizer_config={"method": "direct_eval", "loss": loss, "weighted": is_weighted},
                warnings=fit_warnings,
            )

        is_lin = cls._is_linear_in_params(rhs, param_syms) and bounds is None and loss == "linear"

        if is_lin:
            # Exact Ordinary / Weighted Least Squares: rhs = base(x) + sum_j p_j * phi_j(x)
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

            X_w = X_mat * sqrt_w[:, None]
            y_w = y_adj * sqrt_w
            theta_hat, _, _, _ = np.linalg.lstsq(X_w, y_w, rcond=None)
            preds = base_vec + X_mat @ theta_hat
            residuals = y - preds
            w_residuals = sqrt_w * residuals
            obj_val = float(np.sum(w_residuals ** 2))
            dof = max(n_obs - k_params, 1)
            sigma2 = obj_val / dof
            pinv_info = np.linalg.pinv(X_w.T @ X_w)
            cov = sigma2 * pinv_info
            ses = np.sqrt(np.maximum(np.diag(cov), 0.0))

            id_rep = cls._analyze_identifiability(
                X_w, param_names, pinv_info, n_obs, cond_threshold, corr_threshold
            )
            ci_dict = cls._confidence_intervals(param_names, theta_hat, ses, n_obs, k_params, confidence_level)
            all_warnings = fit_warnings + list(id_rep.warnings)
            p_dict = {nm: float(val) for nm, val in zip(param_names, theta_hat)}
            se_dict = {nm: float(se) for nm, se in zip(param_names, ses)}
            method_name = "WLS" if is_weighted else "OLS"
            return cls._build_fit_result(
                cand_name, lhs_sym, rhs, p_dict, se_dict, y, preds, True,
                f"Exact {method_name} fit.", clean_tol,
                confidence_intervals=ci_dict, covariance_matrix=cov,
                correlation_matrix=id_rep.correlation_matrix, identifiability=id_rep,
                weighted=is_weighted, sqrt_w=sqrt_w if is_weighted else None, pad_mask=pad_mask,
                objective_value=obj_val, n_dropped_missing=n_dropped,
                optimizer_config={"method": method_name, "loss": loss, "weighted": is_weighted, "missing_policy": missing_policy},
                warnings=all_warnings,
            )

        # Nonlinear Least Squares (optionally weighted) with analytic SymPy Jacobian & multi-start
        all_syms = data_syms + list(param_syms)
        f_pred = sp.lambdify(all_syms, rhs, modules="numpy")
        jac_exprs = [sp.diff(rhs, p) for p in param_syms]
        f_jac = [sp.lambdify(all_syms, je, modules="numpy") for je in jac_exprs]

        def _default_bound(p_name: str) -> Tuple[float, float]:
            if tmpl_obj is not None:
                return tmpl_obj.bound_for(p_name)
            return (1e-9, np.inf) if p_name.startswith("gam_") else (-np.inf, np.inf)

        lb = np.array(
            [
                bounds[nm][0] if (bounds and nm in bounds) else _default_bound(nm)[0]
                for nm in param_names
            ],
            dtype=float,
        )
        ub = np.array(
            [
                bounds[nm][1] if (bounds and nm in bounds) else _default_bound(nm)[1]
                for nm in param_names
            ],
            dtype=float,
        )

        def residual_fn(theta: np.ndarray) -> np.ndarray:
            with np.errstate(all="ignore"):
                pred = np.broadcast_to(
                    np.asarray(f_pred(*x_arrays, *theta), dtype=float), (n_obs,)
                )
            res = sqrt_w * (pred - y)
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
                    cols_j.append(sqrt_w * c)
            return np.column_stack(cols_j)

        starts: List[np.ndarray] = []
        if initial_guess:
            g0 = np.array([initial_guess.get(nm, 1.0) for nm in param_names], dtype=float)
            starts.append(np.clip(g0, lb + 1e-12, ub - 1e-12))
        else:
            s1 = np.array(
                [
                    0.0
                    if nm.endswith("_0")
                    else (0.5 if nm.startswith("nu_") else (0.2 if nm.startswith("gam_") else 1.0))
                    for nm in param_names
                ],
                dtype=float,
            )
            s2 = np.array(
                [
                    0.0
                    if nm.endswith("_0")
                    else (-0.5 if nm.startswith("nu_") else (1.0 if nm.startswith("gam_") else 0.5))
                    for nm in param_names
                ],
                dtype=float,
            )
            s3 = np.array(
                [
                    0.0
                    if nm.endswith("_0")
                    else (1.5 if nm.startswith("nu_") else (0.05 if nm.startswith("gam_") else 1.0))
                    for nm in param_names
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
        w_residuals = sqrt_w * residuals
        obj_val = float(np.sum(w_residuals ** 2))
        dof = max(n_obs - k_params, 1)
        sigma2 = obj_val / dof
        J_w = best_opt.jac
        pinv_info = np.linalg.pinv(J_w.T @ J_w)
        cov = sigma2 * pinv_info
        ses = np.sqrt(np.maximum(np.diag(cov), 0.0))

        id_rep = cls._analyze_identifiability(
            J_w, param_names, pinv_info, n_obs, cond_threshold, corr_threshold
        )
        ci_dict = cls._confidence_intervals(param_names, theta_hat, ses, n_obs, k_params, confidence_level)
        all_warnings = fit_warnings + list(id_rep.warnings)
        if not best_opt.success:
            all_warnings.append(f"Optimizer did not converge: {best_opt.message}")

        p_dict = {nm: float(val) for nm, val in zip(param_names, theta_hat)}
        se_dict = {nm: float(se) for nm, se in zip(param_names, ses)}
        method_name = "WNLLS" if is_weighted else "NLLS"
        return cls._build_fit_result(
            cand_name, lhs_sym, rhs, p_dict, se_dict, y, preds,
            bool(best_opt.success), best_opt.message, clean_tol,
            confidence_intervals=ci_dict, covariance_matrix=cov,
            correlation_matrix=id_rep.correlation_matrix, identifiability=id_rep,
            weighted=is_weighted, sqrt_w=sqrt_w if is_weighted else None, pad_mask=pad_mask,
            objective_value=obj_val, n_dropped_missing=n_dropped,
            optimizer_config={
                "method": method_name,
                "loss": loss,
                "weighted": is_weighted,
                "n_starts": len(starts),
                "missing_policy": missing_policy,
            },
            warnings=all_warnings,
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
    def rank_candidates_report(
        cls,
        result: DerivationResult,
        data: Mapping[str, Union[float, Sequence[float], np.ndarray]],
        target_col: Optional[str] = None,
        criterion: str = "bic",
        weights: Optional[Union[Sequence[float], np.ndarray]] = None,
        sigma_y: Optional[Union[Sequence[float], np.ndarray]] = None,
        missing_policy: str = "raise",
    ) -> CandidateRankingReport:
        """Fit all candidate equations, capture rejection reasons, and categorize results."""
        key_map = {
            "bic": lambda r: r.bic,
            "aic": lambda r: r.aic,
            "rmse": lambda r: r.rmse,
            "r2": lambda r: -r.r2,
        }
        if criterion not in key_map:
            raise ValueError(f"Unsupported ranking criterion '{criterion}'. Choose from {list(key_map)}.")

        fits: List[FitResult] = []
        rejected: List[RejectedCandidate] = [
            RejectedCandidate(
                candidate_name=item.get("name", "unknown"),
                reason=item.get("reason", "Rejected during symbolic formulation."),
                stage="symbolic",
            )
            for item in getattr(result, "rejected_candidates", [])
        ]
        mathematically_admissible = [
            c.name for c in result.candidates if getattr(c, "dimensionally_consistent", True)
        ]

        for cand in result.candidates:
            try:
                fr = cls.fit(
                    cand,
                    data,
                    target_col=target_col,
                    weights=weights,
                    sigma_y=sigma_y,
                    missing_policy=missing_policy,
                )
                fits.append(fr)
            except Exception as exc:
                rejected.append(
                    RejectedCandidate(
                        candidate_name=cand.name,
                        reason=f"{type(exc).__name__}: {exc}",
                        stage="fitting",
                    )
                )

        fits.sort(key=key_map[criterion])
        categories = {
            "mathematically_admissible": mathematically_admissible,
            "statistically_supported": [
                fr.candidate_name
                for fr in fits
                if fr.converged and (fr.identifiability is None or fr.identifiability.identifiable)
            ],
            "identifiability_flagged": [
                fr.candidate_name
                for fr in fits
                if fr.identifiability is not None and not fr.identifiability.identifiable
            ],
            "rejected": [r.candidate_name for r in rejected],
        }
        return CandidateRankingReport(
            ranked=fits,
            rejected=rejected,
            criterion=criterion,
            categories=categories,
        )

    @classmethod
    def rank_candidates(
        cls,
        result: DerivationResult,
        data: Mapping[str, Union[float, Sequence[float], np.ndarray]],
        target_col: Optional[str] = None,
        criterion: str = "bic",
        weights: Optional[Union[Sequence[float], np.ndarray]] = None,
        sigma_y: Optional[Union[Sequence[float], np.ndarray]] = None,
        missing_policy: str = "raise",
    ) -> List[FitResult]:
        """Fit all candidate equations in a DerivationResult and rank by BIC, AIC, or RMSE."""
        return cls.rank_candidates_report(
            result,
            data,
            target_col=target_col,
            criterion=criterion,
            weights=weights,
            sigma_y=sigma_y,
            missing_policy=missing_policy,
        ).ranked

    @classmethod
    def select_sparse_pi_report(
        cls,
        result: DerivationResult,
        data: Mapping[str, Union[float, Sequence[float], np.ndarray]],
        target_col: Optional[str] = None,
        max_groups: int = 2,
        form: str = "power",
        criterion: str = "bic",
        weights: Optional[Union[Sequence[float], np.ndarray]] = None,
        sigma_y: Optional[Union[Sequence[float], np.ndarray]] = None,
        missing_policy: str = "raise",
    ) -> CandidateRankingReport:
        """
        Sparse Buckingham-Pi subset selection via `TEMPLATE_REGISTRY`, returning ranked fits
        alongside explicit rejection diagnostics and admissibility categories.
        """
        key_map = {
            "bic": lambda r: r.bic,
            "aic": lambda r: r.aic,
            "rmse": lambda r: r.rmse,
            "r2": lambda r: -r.r2,
        }
        if criterion not in key_map:
            raise ValueError(f"Unsupported ranking criterion '{criterion}'. Choose from {list(key_map)}.")

        try:
            tmpl = TEMPLATE_REGISTRY.get(form)
        except KeyError as exc:
            raise ValueError(
                f"Unsupported form '{form}'; choose from {TEMPLATE_REGISTRY.names()}."
            ) from exc

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
                label_parts = [f"Pi{i + 1}" for i in idxs]
                selected_pis = [all_pis[i] for i in idxs]
                mod_factor, extra_params = tmpl.build_factor(
                    selected_pis, indices=[i + 1 for i in idxs]
                )
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
                        pi_groups=selected_pis,
                        is_exact_state=base_cand.is_exact_state,
                        dimensionally_consistent=getattr(base_cand, "dimensionally_consistent", True),
                        assumptions=list(tmpl.assumptions),
                        template_name=tmpl.name,
                    )
                )

        fits: List[FitResult] = []
        rejected: List[RejectedCandidate] = []
        for cand in subset_cands:
            try:
                fits.append(
                    cls.fit(
                        cand,
                        data,
                        target_col=target_col,
                        weights=weights,
                        sigma_y=sigma_y,
                        missing_policy=missing_policy,
                    )
                )
            except Exception as exc:
                rejected.append(
                    RejectedCandidate(
                        candidate_name=cand.name,
                        reason=f"{type(exc).__name__}: {exc}",
                        stage="fitting",
                    )
                )

        fits.sort(key=key_map[criterion])
        categories = {
            "mathematically_admissible": [
                c.name for c in subset_cands if getattr(c, "dimensionally_consistent", True)
            ],
            "statistically_supported": [
                fr.candidate_name
                for fr in fits
                if fr.converged and (fr.identifiability is None or fr.identifiability.identifiable)
            ],
            "identifiability_flagged": [
                fr.candidate_name
                for fr in fits
                if fr.identifiability is not None and not fr.identifiability.identifiable
            ],
            "rejected": [r.candidate_name for r in rejected],
        }
        return CandidateRankingReport(
            ranked=fits,
            rejected=rejected,
            criterion=criterion,
            categories=categories,
        )

    @classmethod
    def select_sparse_pi(
        cls,
        result: DerivationResult,
        data: Mapping[str, Union[float, Sequence[float], np.ndarray]],
        target_col: Optional[str] = None,
        max_groups: int = 2,
        form: str = "power",
        criterion: str = "bic",
        weights: Optional[Union[Sequence[float], np.ndarray]] = None,
        sigma_y: Optional[Union[Sequence[float], np.ndarray]] = None,
        missing_policy: str = "raise",
    ) -> List[FitResult]:
        """
        Sparse Buckingham-Pi subset selection for high-nullity multi-physics systems.

        Evaluates Pi-subsets of cardinality 0..max_groups (using any registered template in
        `TEMPLATE_REGISTRY`) and ranks parsimonious laws by BIC/AIC/RMSE to prevent collinearity.
        """
        return cls.select_sparse_pi_report(
            result,
            data,
            target_col=target_col,
            max_groups=max_groups,
            form=form,
            criterion=criterion,
            weights=weights,
            sigma_y=sigma_y,
            missing_policy=missing_policy,
        ).ranked

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
                t_curr = t_ev + max(1e-12 * max(abs(t_end - float(span[0])), abs(t_ev)), 1e-15)
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
        *,
        confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None,
        covariance_matrix: Optional[np.ndarray] = None,
        correlation_matrix: Optional[np.ndarray] = None,
        identifiability: Optional[IdentifiabilityReport] = None,
        weighted: bool = False,
        sqrt_w: Optional[np.ndarray] = None,
        pad_mask: Optional[np.ndarray] = None,
        objective_value: Optional[float] = None,
        n_dropped_missing: int = 0,
        optimizer_config: Optional[Dict[str, Any]] = None,
        warnings: Optional[List[str]] = None,
    ) -> FitResult:
        n_obs = y.size
        k = len(p_dict)
        residuals = y - preds
        out_warnings = list(warnings or [])

        if weighted and sqrt_w is not None:
            w_raw = sqrt_w ** 2
            w_mean = float(np.mean(w_raw))
            w_norm = w_raw / w_mean if w_mean > 0.0 else np.ones_like(w_raw)
            y_bar = float(np.sum(w_raw * y) / np.sum(w_raw)) if np.sum(w_raw) > 0.0 else float(np.mean(y))
            sse_w = float(np.sum(w_norm * (residuals ** 2)))
            sst_w = float(np.sum(w_norm * ((y - y_bar) ** 2)))
            r2 = 1.0 - (sse_w / sst_w) if sst_w > 0.0 else (1.0 if sse_w < 1e-24 else 0.0)
            rmse = float(np.sqrt(sse_w / n_obs))
            mae = float(np.sum(w_norm * np.abs(residuals)) / n_obs)
            c2_hat = max(float(np.mean(w_raw * (residuals ** 2))), 1e-30)
            sum_log_w = float(np.sum(np.log(np.maximum(w_raw, 1e-300))))
            log_lik = -0.5 * n_obs * (np.log(2.0 * np.pi * c2_hat) + 1.0) + 0.5 * sum_log_w
            sse_default = float(np.sum(w_raw * (residuals ** 2)))
        else:
            sse = float(np.sum(residuals ** 2))
            sst = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - (sse / sst) if sst > 0 else (1.0 if sse < 1e-24 else 0.0)
            rmse = float(np.sqrt(sse / n_obs))
            mae = float(np.mean(np.abs(residuals)))
            mse_floor = max(sse / n_obs, 1e-30)
            log_lik = -0.5 * n_obs * (np.log(2 * np.pi * mse_floor) + 1.0)
            sse_default = sse

        underdetermined = k > 0 and n_obs <= k
        if underdetermined:
            aic = float("inf")
            bic = float("inf")
            se_dict = {nm: float("nan") for nm in p_dict}
            msg_ud = f"Saturated/underdetermined fit (n_obs={n_obs} <= n_params={k}): AIC/BIC set to inf."
            if msg_ud not in out_warnings:
                out_warnings.append(msg_ud)
        else:
            aic = float(2 * k - 2 * log_lik)
            bic = float(k * np.log(max(n_obs, 1)) - 2 * log_lik)

        if pad_mask is not None:
            full_len = int(pad_mask.size)
            preds_out = np.full(full_len, np.nan, dtype=float)
            res_out = np.full(full_len, np.nan, dtype=float)
            preds_out[pad_mask] = preds
            res_out[pad_mask] = residuals
        else:
            preds_out = preds
            res_out = residuals

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

        if not converged:
            tier = "fit_failed"
        elif underdetermined:
            tier = "underdetermined"
        elif identifiability is not None and not identifiability.identifiable:
            tier = "identifiability_warning"
        else:
            tier = "statistically_supported"

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
            predictions=preds_out,
            residuals=res_out,
            converged=converged,
            message=message,
            confidence_intervals=dict(confidence_intervals or {}),
            covariance_matrix=covariance_matrix,
            correlation_matrix=correlation_matrix,
            identifiability=identifiability,
            weighted=weighted,
            objective_value=sse_default if objective_value is None else objective_value,
            n_dropped_missing=n_dropped_missing,
            optimizer_config=dict(optimizer_config or {}),
            warnings=out_warnings,
            admissibility_tier=tier,
        )
