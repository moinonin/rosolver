"""Discretization vs. Claimed-Error Budget & Scheme-Ambiguity Auditor (Sprint 10 / S10.3).

Prevents coarse-grid or coarse-timestep discretization error and undeclared numerical
integrators from confounding reported experimental validation error claims (FM-3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ..model import Model
from ..odesys import ODESystem
from ..results import ResultMixin, VerificationStatus, to_jsonable
from ..verify import gci_from_values


@dataclass
class ErrorBudgetReport(ResultMixin):
    """Audit comparing discretization uncertainty and scheme spread against a claimed error."""

    claimed_rel_error: float
    declared_scheme: Optional[str]
    dt_paper: float
    reference_value: float
    discretization_error_paper: float
    budget_ratio: float  # discretization_error_paper / claimed_rel_error
    scheme_values: Dict[str, float] = field(default_factory=dict)
    scheme_errors: Dict[str, float] = field(default_factory=dict)
    scheme_spread: float = 0.0
    gci_observed_order: Optional[float] = None
    gci_fine_21: Optional[float] = None
    scheme_dependent: bool = False
    diagnostic_code: str = "error_budget_resolved"
    status: VerificationStatus = VerificationStatus.PASSED
    passed: bool = True
    warnings: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=== Discretization vs. Claimed-Error Budget Audit ===",
            f"  Status: {self.status.value} | Code: {self.diagnostic_code} | Scheme: {self.declared_scheme or 'UNSPECIFIED'}",
            f"  Claimed rel. error: {self.claimed_rel_error * 100:.4g}% | Paper dt: {self.dt_paper:g}",
            f"  Discretization err at paper dt: {self.discretization_error_paper * 100:.4g}% "
            f"(budget ratio = {self.budget_ratio:.3f})",
        ]
        if self.scheme_errors:
            parts = ", ".join(f"{k}={v * 100:.4g}%" for k, v in self.scheme_errors.items())
            lines.append(f"  Candidate scheme errors: {parts} (spread = {self.scheme_spread * 100:.4g}%)")
        if self.gci_observed_order is not None:
            lines.append(
                f"  3-level GCI observed order p = {self.gci_observed_order:.3f}, "
                f"GCI_fine = {(self.gci_fine_21 or 0.0) * 100:.4g}%"
            )
        for w in self.warnings:
            lines.append(f"  Warning: {w}")
        return "\n".join(lines)


_FORMAL_ORDERS: Dict[str, float] = {
    "euler": 1.0,
    "beuler": 1.0,
    "ssprk2": 2.0,
    "cn": 2.0,
    "ssprk3": 3.0,
    "rk4": 4.0,
    "dopri5": 5.0,
}


def audit_claimed_accuracy(
    model_or_solver: Union[Model, ODESystem, Callable[[float, str], float]],
    *,
    claimed_rel_error: float,
    dt_paper: float,
    declared_scheme: Optional[str] = None,
    t_span: Optional[Tuple[float, float]] = None,
    y0: Optional[Mapping[str, float]] = None,
    params: Optional[Mapping[str, Any]] = None,
    qoi_field: Optional[str] = None,
    reference_value: Optional[float] = None,
    candidate_schemes: Sequence[str] = ("euler", "cn", "rk4"),
    r_factor: float = 2.0,
    warning_budget_fraction: float = 0.25,
) -> ErrorBudgetReport:
    """Compare discretization error at a manuscript's stated timestep against `claimed_rel_error`.

    Args:
        model_or_solver: A validated ODE `Model`, an `ODESystem`, or a callable ``fn(dt, scheme) -> float``.
        claimed_rel_error: Claimed relative error as a dimensionless fraction (e.g. ``0.0035`` for 0.35%).
        dt_paper: Time step reported in the manuscript (e.g. ``6000.0`` s for ``n=12`` over ``72000`` s).
        declared_scheme: Integrator name if stated by the authors, or ``None`` if omitted.
        t_span: Integration interval ``(t0, t_end)`` (required for `Model`/`ODESystem`).
        y0: Initial state dict (auto-extracted from `Model` if omitted).
        params: Parameter dict (auto-extracted from `Model` if omitted).
        qoi_field: State name whose final value is audited (defaults to first state).
        reference_value: Optional exact/reference QoI value; otherwise computed via high-order ``DOP853``.
        candidate_schemes: Candidate fixed-step schemes tested when ``declared_scheme is None``.
        r_factor: Refinement ratio for 3-level Roache/Celik GCI (default 2.0).
        warning_budget_fraction: Fraction of ``claimed_rel_error`` above which discretization
            confounds validation (default 0.25, i.e., 25%).
    """
    if claimed_rel_error <= 0.0:
        raise ValueError("claimed_rel_error must be strictly positive (fraction, e.g. 0.0035 for 0.35%).")
    if dt_paper <= 0.0:
        raise ValueError("dt_paper must be strictly positive.")

    # Normalize solver callable
    if isinstance(model_or_solver, Model):
        ode_sys = model_or_solver.to_ode_system()
        eff_params = model_or_solver.parameter_values(params)
        eff_y0 = model_or_solver.resolved_ode_initial_conditions(y0, ode_sys.states, params=eff_params)
        field_name = qoi_field or ode_sys.states[0]
        if t_span is None:
            raise ValueError("t_span=(t0, t_end) is required when auditing a Model.")

        def _eval_fn(dt_v: float, sc: str) -> float:
            tr = ode_sys.solve(t_span, eff_y0, params=eff_params, method=sc, dt=dt_v)
            return float(tr.final[field_name])

        if reference_value is None:
            ref_tr = ode_sys.solve(t_span, eff_y0, params=eff_params, method="DOP853", rtol=1e-11, atol=1e-14)
            ref_val = float(ref_tr.final[field_name])
        else:
            ref_val = float(reference_value)

    elif isinstance(model_or_solver, ODESystem):
        if t_span is None or y0 is None:
            raise ValueError("t_span and y0 are required when auditing an ODESystem.")
        ode_sys = model_or_solver
        eff_params = dict(params or {})
        eff_y0 = dict(y0)
        field_name = qoi_field or ode_sys.states[0]

        def _eval_fn(dt_v: float, sc: str) -> float:
            tr = ode_sys.solve(t_span, eff_y0, params=eff_params, method=sc, dt=dt_v)
            return float(tr.final[field_name])

        if reference_value is None:
            ref_tr = ode_sys.solve(t_span, eff_y0, params=eff_params, method="DOP853", rtol=1e-11, atol=1e-14)
            ref_val = float(ref_tr.final[field_name])
        else:
            ref_val = float(reference_value)

    elif callable(model_or_solver):
        _eval_fn = model_or_solver
        if reference_value is None:
            # Estimate reference via fine RK4 at dt_paper / 16
            ref_val = float(_eval_fn(dt_paper / 16.0, declared_scheme or "rk4"))
        else:
            ref_val = float(reference_value)
    else:
        raise TypeError(f"Unsupported model_or_solver type: {type(model_or_solver)}")

    scale_ref = max(abs(ref_val), 1e-30)
    warnings: List[str] = []

    if declared_scheme is not None:
        # Case A: Declared scheme -> 3-level refinement & exact error check
        dts = (dt_paper, dt_paper / r_factor, dt_paper / (r_factor ** 2))
        vals = [float(_eval_fn(dt_v, declared_scheme)) for dt_v in dts]
        formal_p = _FORMAL_ORDERS.get(declared_scheme.lower())
        gci = gci_from_values(spacings=dts, quantities=vals, formal_order=formal_p)

        err_paper = abs(vals[0] - ref_val) / scale_ref
        ratio = err_paper / claimed_rel_error
        scheme_vals = {declared_scheme: vals[0]}
        scheme_errs = {declared_scheme: err_paper}
        macroscopic_spread = (max(vals) - min(vals)) / scale_ref > 1e-8

        if not np.all(np.isfinite(vals)):
            status = VerificationStatus.FAILED
            code = "solver_diverged_nonfinite"
            warnings.append(f"Declared scheme '{declared_scheme}' produced non-finite values around dt={dt_paper:g}.")
        elif macroscopic_spread and gci.regime == "divergent":
            status = VerificationStatus.FAILED
            code = "refinement_divergent"
            warnings.extend(gci.warnings)
        elif ratio >= 1.0:
            status = VerificationStatus.FAILED
            code = "discretization_exceeds_claimed_error"
            warnings.append(
                f"Discretization error at dt={dt_paper:g} ({err_paper * 100:.4g}%) exceeds "
                f"claimed validation error ({claimed_rel_error * 100:.4g}%)."
            )
        elif ratio >= warning_budget_fraction:
            status = VerificationStatus.WARNING
            code = "discretization_consumes_error_budget"
            warnings.append(
                f"Discretization error at dt={dt_paper:g} ({err_paper * 100:.4g}%) consumes "
                f"{ratio * 100:.1f}% of claimed error budget ({claimed_rel_error * 100:.4g}%)."
            )
        elif macroscopic_spread and gci.status == VerificationStatus.WARNING:
            status = VerificationStatus.WARNING
            code = "refinement_warning"
            warnings.extend(gci.warnings)
        else:
            status = VerificationStatus.PASSED
            code = "error_budget_resolved"

        return ErrorBudgetReport(
            claimed_rel_error=float(claimed_rel_error),
            declared_scheme=declared_scheme,
            dt_paper=float(dt_paper),
            reference_value=float(ref_val),
            discretization_error_paper=float(err_paper),
            budget_ratio=float(ratio),
            scheme_values=scheme_vals,
            scheme_errors=scheme_errs,
            scheme_spread=0.0,
            gci_observed_order=float(gci.observed_order) if np.isfinite(gci.observed_order) else None,
            gci_fine_21=float(gci.gci_fine_21) if np.isfinite(gci.gci_fine_21) else None,
            scheme_dependent=False,
            diagnostic_code=code,
            status=status,
            passed=(status == VerificationStatus.PASSED),
            warnings=warnings,
            assumptions=[
                f"3-level temporal refinement with ratio r={r_factor:g} around dt={dt_paper:g}.",
                f"Budget warning threshold: discretization error >= {warning_budget_fraction * 100:.0f}% of claimed error.",
            ],
        )

    # Case B: Scheme omitted in manuscript -> evaluate candidate schemes at dt_paper
    scheme_vals = {}
    scheme_errs = {}
    for sc in candidate_schemes:
        try:
            v_sc = float(_eval_fn(dt_paper, sc))
            scheme_vals[sc] = v_sc
            scheme_errs[sc] = abs(v_sc - ref_val) / scale_ref
        except Exception as exc:
            warnings.append(f"Candidate scheme '{sc}' failed at dt={dt_paper:g}: {exc}")

    if not scheme_errs:
        return ErrorBudgetReport(
            claimed_rel_error=float(claimed_rel_error),
            declared_scheme=None,
            dt_paper=float(dt_paper),
            reference_value=float(ref_val),
            discretization_error_paper=float("nan"),
            budget_ratio=float("nan"),
            scheme_dependent=True,
            diagnostic_code="all_candidate_schemes_failed",
            status=VerificationStatus.FAILED,
            passed=False,
            warnings=warnings,
        )

    u_min = min(scheme_errs.values())
    u_max = max(scheme_errs.values())
    spread = (max(scheme_vals.values()) - min(scheme_vals.values())) / scale_ref
    ratio_max = u_max / claimed_rel_error

    if u_min >= claimed_rel_error:
        status = VerificationStatus.FAILED
        code = "discretization_exceeds_claimed_error"
        scheme_dep = False
        warnings.append(
            f"Even the most accurate candidate scheme has discretization error {u_min * 100:.4g}% >= "
            f"claimed error {claimed_rel_error * 100:.4g}% at dt={dt_paper:g}."
        )
    elif ratio_max >= warning_budget_fraction:
        status = VerificationStatus.WARNING
        code = "scheme_unspecified_error_budget_confounded"
        scheme_dep = True
        worst_sc = max(scheme_errs, key=lambda k: scheme_errs[k])
        best_sc = min(scheme_errs, key=lambda k: scheme_errs[k])
        warnings.append(
            f"Time-integration scheme is unspecified at dt={dt_paper:g}: '{worst_sc}' error "
            f"({u_max * 100:.4g}%) consumes {ratio_max * 100:.1f}% of the claimed "
            f"{claimed_rel_error * 100:.4g}% error budget, whereas '{best_sc}' gives {u_min * 100:.4g}%."
        )
    else:
        status = VerificationStatus.PASSED
        code = "error_budget_resolved_across_schemes"
        scheme_dep = False

    return ErrorBudgetReport(
        claimed_rel_error=float(claimed_rel_error),
        declared_scheme=None,
        dt_paper=float(dt_paper),
        reference_value=float(ref_val),
        discretization_error_paper=float(u_max),
        budget_ratio=float(ratio_max),
        scheme_values=scheme_vals,
        scheme_errors=scheme_errs,
        scheme_spread=float(spread),
        gci_observed_order=None,
        gci_fine_21=None,
        scheme_dependent=scheme_dep,
        diagnostic_code=code,
        status=status,
        passed=(status == VerificationStatus.PASSED),
        warnings=warnings,
        assumptions=[
            f"Evaluated candidate schemes {list(candidate_schemes)} at manuscript step dt={dt_paper:g}.",
            "Scheme-unspecified ambiguity triggers warning when worst-case error >= 25% of claimed error.",
        ],
    )
