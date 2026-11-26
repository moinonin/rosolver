"""Experimental model validation engine with numerical and experimental uncertainties (ENG_SPEC.md P2.4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from ..data import Dataset
from ..dimensions import Dimension
from ..results import ResultMixin, VerificationStatus


@dataclass
class ValidationPoint(ResultMixin):
    """Single validation point comparing simulation prediction S against experiment D."""

    label: str
    coordinate: Optional[float]
    simulation: float
    experiment: float
    error: float  # E = S - D
    abs_error: float
    rel_error: float
    u_num: Optional[float]
    u_exp: Optional[float]
    u_input: Optional[float]
    u_val: Optional[float]
    normalized_error: Optional[float]  # |E| / u_val
    within_uncertainty: Optional[bool]
    conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport(ResultMixin):
    """Experimental model validation report across one or more operating conditions."""

    study_name: str
    field: str
    unit: str
    points: List[ValidationPoint]
    rmse: float
    mae: float
    max_abs_error: float
    mean_bias: float
    max_normalized_error: Optional[float]
    fraction_within_uncertainty: Optional[float]
    coverage_factor: float
    operating_conditions: Dict[str, Any]
    status: VerificationStatus
    passed: bool
    warnings: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"=== Model Validation Report: {self.study_name} [{self.field} ({self.unit})] ===",
            f"  Status: {self.status.value} | Passed: {self.passed} | Points: {len(self.points)}",
            f"  RMSE = {self.rmse:.4e} | MAE = {self.mae:.4e} | Max |E| = {self.max_abs_error:.4e} | Bias = {self.mean_bias:.4e}",
        ]
        if self.max_normalized_error is not None:
            lines.append(
                f"  Max |E|/u_val = {self.max_normalized_error:.3f} (limit k={self.coverage_factor:g}) | "
                f"Within band = {self.fraction_within_uncertainty * 100:.1f}%"
            )
        for w in self.warnings:
            lines.append(f"  Warning: {w}")
        return "\n".join(lines)

    def _repr_markdown_(self) -> str:
        lines = [
            f"### Experimental Validation: `{self.study_name}` (`{self.status.value.upper()}`)",
            "",
            f"- **Field:** `{self.field}` ({self.unit}) | **Points:** {len(self.points)} | **Passed:** `{self.passed}`",
            f"- **RMSE:** {self.rmse:.4e} | **MAE:** {self.mae:.4e} | **Max |E|:** {self.max_abs_error:.4e} | **Mean Bias:** {self.mean_bias:.4e}",
        ]
        if self.max_normalized_error is not None and self.fraction_within_uncertainty is not None:
            lines.append(
                f"- **Max |E|/u_val:** {self.max_normalized_error:.3f} (k={self.coverage_factor:g}) | "
                f"**Within Band:** {self.fraction_within_uncertainty * 100:.1f}%"
            )
        if self.warnings:
            lines.append("")
            for w in self.warnings:
                lines.append(f"- **Warning:** {w}")
        return "\n".join(lines)


def _expand_uncertainty(
    val: Union[None, float, Sequence[float], np.ndarray], n: int, name: str
) -> Optional[np.ndarray]:
    if val is None:
        return None
    arr = np.broadcast_to(np.asarray(val, dtype=float), (n,)).copy()
    if np.any(arr < 0.0) or not np.all(np.isfinite(arr)):
        raise ValueError(f"Uncertainty '{name}' must contain finite non-negative values.")
    return arr


class ValidationStudy:
    """Compare simulation predictions against experimental measurements and uncertainties."""

    @staticmethod
    def compare(
        simulation: Union[float, Sequence[float], np.ndarray],
        experiment: Union[float, Sequence[float], np.ndarray],
        *,
        field: str = "value",
        unit: str = "1",
        u_num: Union[None, float, Sequence[float], np.ndarray] = None,
        u_exp: Union[None, float, Sequence[float], np.ndarray] = None,
        u_input: Union[None, float, Sequence[float], np.ndarray] = None,
        coordinates: Optional[Sequence[float]] = None,
        coord_name: str = "x",
        labels: Optional[Sequence[str]] = None,
        operating_conditions: Optional[Mapping[str, Any]] = None,
        coverage_factor: float = 1.0,
        error_tol: Optional[float] = None,
        study_name: str = "validation_study",
    ) -> ValidationReport:
        Dimension.from_unit(unit)
        sim_arr = np.atleast_1d(np.asarray(simulation, dtype=float)).reshape(-1)
        exp_arr = np.atleast_1d(np.asarray(experiment, dtype=float)).reshape(-1)
        if sim_arr.size != exp_arr.size or sim_arr.size == 0:
            raise ValueError("Simulation and experiment arrays must be non-empty and have equal length.")
        if coverage_factor <= 0.0:
            raise ValueError("coverage_factor must be positive.")

        n = sim_arr.size
        coords_arr = None
        if coordinates is not None:
            coords_arr = np.atleast_1d(np.asarray(coordinates, dtype=float)).reshape(-1)
            if coords_arr.size != n:
                raise ValueError("coordinates length must match simulation/experiment length.")

        u_n = _expand_uncertainty(u_num, n, "u_num")
        u_e = _expand_uncertainty(u_exp, n, "u_exp")
        u_i = _expand_uncertainty(u_input, n, "u_input")

        has_num = u_n is not None
        has_exp = u_e is not None
        has_any_u = has_num or has_exp or (u_i is not None)

        points: List[ValidationPoint] = []
        errors = sim_arr - exp_arr
        abs_errors = np.abs(errors)
        norm_errors: List[float] = []
        within_flags: List[bool] = []

        for idx in range(n):
            s_val = float(sim_arr[idx])
            d_val = float(exp_arr[idx])
            err = float(errors[idx])
            aerr = float(abs_errors[idx])
            rerr = aerr / abs(d_val) if abs(d_val) > 1e-30 else float("nan")
            un_i = float(u_n[idx]) if u_n is not None else None
            ue_i = float(u_e[idx]) if u_e is not None else None
            ui_i = float(u_i[idx]) if u_i is not None else None

            if has_any_u:
                uv_sq = (un_i or 0.0) ** 2 + (ue_i or 0.0) ** 2 + (ui_i or 0.0) ** 2
                uv_i: Optional[float] = float(np.sqrt(uv_sq))
                if uv_i > 0.0:
                    ne_i: Optional[float] = aerr / uv_i
                    w_i: Optional[bool] = bool(aerr <= coverage_factor * uv_i)
                    norm_errors.append(ne_i)
                    within_flags.append(w_i)
                else:
                    ne_i = float("inf") if aerr > 0.0 else 0.0
                    w_i = bool(aerr == 0.0)
                    norm_errors.append(ne_i)
                    within_flags.append(w_i)
            else:
                uv_i = None
                ne_i = None
                w_i = None

            lbl = labels[idx] if labels is not None else f"pt_{idx + 1}"
            c_val = float(coords_arr[idx]) if coords_arr is not None else None
            cond_pt = {coord_name: c_val} if c_val is not None else {}
            points.append(
                ValidationPoint(
                    label=lbl,
                    coordinate=c_val,
                    simulation=s_val,
                    experiment=d_val,
                    error=err,
                    abs_error=aerr,
                    rel_error=rerr,
                    u_num=un_i,
                    u_exp=ue_i,
                    u_input=ui_i,
                    u_val=uv_i,
                    normalized_error=ne_i,
                    within_uncertainty=w_i,
                    conditions=cond_pt,
                )
            )

        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(abs_errors))
        max_abs = float(np.max(abs_errors))
        mean_bias = float(np.mean(errors))
        max_norm = float(max(norm_errors)) if norm_errors else None
        frac_within = float(sum(within_flags) / len(within_flags)) if within_flags else None

        warnings: List[str] = []
        failed = False
        if error_tol is not None and max_abs > error_tol:
            failed = True
            warnings.append(f"Max absolute error {max_abs:.4e} exceeds error_tol {error_tol:.4e}.")
        if within_flags and not all(within_flags):
            failed = True
            warnings.append(
                f"{len(within_flags) - sum(within_flags)}/{len(within_flags)} point(s) exceed "
                f"validation uncertainty band (k={coverage_factor:g})."
            )

        if failed:
            status = VerificationStatus.FAILED
        elif not has_any_u and error_tol is None:
            status = VerificationStatus.INCONCLUSIVE
            warnings.append(
                "Neither numerical/experimental uncertainties nor error_tol were supplied; validation is inconclusive."
            )
        elif has_exp and not has_num:
            status = VerificationStatus.WARNING
            warnings.append(
                "Numerical solution uncertainty u_num was not provided; validation uses experimental uncertainty only."
            )
        elif has_num and not has_exp and error_tol is None:
            status = VerificationStatus.WARNING
            warnings.append(
                "Experimental uncertainty u_exp was not provided; validation uses numerical uncertainty only."
            )
        else:
            status = VerificationStatus.PASSED

        assumptions = [
            "Validation metric E = S - D and combined u_val = sqrt(u_num^2 + u_exp^2 + u_input^2) "
            "designed with ASME V&V 20 principles in mind; does not constitute formal standards compliance.",
            f"Coverage factor k={coverage_factor:g}.",
        ]
        return ValidationReport(
            study_name=study_name,
            field=field,
            unit=unit,
            points=points,
            rmse=rmse,
            mae=mae,
            max_abs_error=max_abs,
            mean_bias=mean_bias,
            max_normalized_error=max_norm,
            fraction_within_uncertainty=frac_within,
            coverage_factor=coverage_factor,
            operating_conditions=dict(operating_conditions or {}),
            status=status,
            passed=(status == VerificationStatus.PASSED),
            warnings=warnings,
            assumptions=assumptions,
        )

    @classmethod
    def from_datasets(
        cls,
        sim_dataset: Dataset,
        exp_dataset: Dataset,
        *,
        field: str,
        coord: Optional[str] = None,
        u_num: Union[None, float, Sequence[float], np.ndarray] = None,
        u_input: Union[None, float, Sequence[float], np.ndarray] = None,
        operating_conditions: Optional[Mapping[str, Any]] = None,
        coverage_factor: float = 1.0,
        error_tol: Optional[float] = None,
        study_name: Optional[str] = None,
    ) -> ValidationReport:
        sim_u_str = sim_dataset.units.get(field)
        exp_u_str = exp_dataset.units.get(field)
        if sim_u_str and exp_u_str:
            if Dimension.from_unit(sim_u_str) != Dimension.from_unit(exp_u_str):
                raise ValueError(
                    f"Unit mismatch for field '{field}': simulation '{sim_u_str}' vs experiment '{exp_u_str}'."
                )
        unit = exp_u_str or sim_u_str or "1"

        sim_vals = sim_dataset.column(field)
        exp_vals = exp_dataset.column(field)
        exp_u = exp_dataset.uncertainty_for(field)

        coord_name = coord or (next(iter(exp_dataset.coordinates), None))
        coords = None
        u_num_resolved = u_num if u_num is not None else sim_dataset.uncertainty_for(field)

        if coord_name is not None and coord_name in exp_dataset.coordinates:
            exp_c = exp_dataset.column(coord_name)
            coords = exp_c
            if coord_name in sim_dataset.coordinates:
                sim_c = sim_dataset.column(coord_name)
                if sim_c.shape != exp_c.shape or not np.allclose(sim_c, exp_c, rtol=1e-12, atol=1e-12):
                    order = np.argsort(sim_c)
                    sim_vals = np.interp(exp_c, sim_c[order], sim_vals[order])
                    if isinstance(u_num_resolved, (Sequence, np.ndarray)) and np.asarray(u_num_resolved).size == sim_c.size:
                        u_arr = np.asarray(u_num_resolved, dtype=float)
                        u_num_resolved = np.interp(exp_c, sim_c[order], u_arr[order])

        return cls.compare(
            simulation=sim_vals,
            experiment=exp_vals,
            field=field,
            unit=unit,
            u_num=u_num_resolved,
            u_exp=exp_u,
            u_input=u_input,
            coordinates=coords,
            coord_name=coord_name or "x",
            operating_conditions=operating_conditions,
            coverage_factor=coverage_factor,
            error_tol=error_tol,
            study_name=study_name or f"{sim_dataset.name}_vs_{exp_dataset.name}",
        )
