"""PySR / External Symbolic-Regression Candidate Importer (ENG_SPEC.md P5.2 slim / Sprint 9 Slim).

Imports Pareto-front candidate expressions from PySR ``hall_of_fame.csv`` files,
duck-typed ``PySRRegressor`` objects, or raw expression lists, and filters every
candidate deterministically through four sequential gates:

1. **AST Whitelist Parsing (P0.5):** ``rosolver._parsing.parse_expr`` (zero ``eval``/``sympify``).
2. **Dimensional Homogeneity & Normalization Audit:** ``diagnose_normalization_shortcuts`` (Sprint 10).
3. **Order-of-Magnitude Scale-Separation Audit:** ``check_equation_scales`` (Sprint 10).
4. **WLS Calibration & Jacobian Identifiability (P2.1):** ``NumericalSolver.fit`` with BIC ranking.

Zero runtime dependency on ``pysr`` or external LLM SDKs.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import sympy as sp

from .._parsing import parse_expr, real_symbol
from ..data import Dataset
from ..dimensions import Dimension
from ..results import ResultMixin, VerificationStatus
from .derivation import CandidateEquation

if TYPE_CHECKING:
    from ..claims.schema import ClaimManifest
    from ..numerical import FitResult


@dataclass
class ImportedCandidate(ResultMixin):
    """Raw symbolic-regression candidate imported from PySR or another external tool."""

    index: int
    name: str
    raw_equation: str
    complexity: Optional[int] = None
    pysr_loss: Optional[float] = None
    pysr_score: Optional[float] = None
    source_format: str = "expression_list"


@dataclass
class ScreenedCandidate(ResultMixin):
    """Deterministic 4-stage screening outcome for one imported symbolic candidate."""

    candidate: ImportedCandidate
    admissible: bool
    status: VerificationStatus
    stage_results: Dict[str, str] = field(default_factory=dict)
    parsed_expression: Optional[str] = None
    calibrated_expression: Optional[str] = None
    rejection_stage: Optional[str] = None
    rejection_reason: Optional[str] = None
    dimensional_consistent: Optional[bool] = None
    normalization_hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    scale_log10_gap: Optional[float] = None
    scale_diagnostic_code: Optional[str] = None
    fit_result: Optional[FitResult] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def bic(self) -> float:
        if self.fit_result is not None and np.isfinite(self.fit_result.bic):
            return float(self.fit_result.bic)
        if self.candidate.pysr_loss is not None:
            return float(self.candidate.pysr_loss)
        return float("inf")

    @property
    def rmse(self) -> float:
        if self.fit_result is not None and np.isfinite(self.fit_result.rmse):
            return float(self.fit_result.rmse)
        return float("inf")


@dataclass
class CandidateScreeningReport(ResultMixin):
    """Aggregated screening and ranking report over an imported Pareto front."""

    target: str
    source: str
    total_imported: int
    admissible_candidates: List[ScreenedCandidate]
    rejected_candidates: List[ScreenedCandidate]
    status: VerificationStatus
    warnings: List[str] = field(default_factory=list)
    dims: Dict[str, str] = field(default_factory=dict)
    operating_envelope: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    @property
    def best_candidate(self) -> Optional[ScreenedCandidate]:
        return self.admissible_candidates[0] if self.admissible_candidates else None

    def summary(self) -> str:
        lines = [
            f"CandidateScreeningReport [{self.status.value.upper()}] — target='{self.target}', source='{self.source}'",
            f"  Imported: {self.total_imported} | Admissible: {len(self.admissible_candidates)} | Rejected: {len(self.rejected_candidates)}",
        ]
        for rank, sc in enumerate(self.admissible_candidates, start=1):
            expr_txt = sc.calibrated_expression or sc.parsed_expression or sc.candidate.raw_equation
            if sc.fit_result is not None:
                lines.append(
                    f"  #{rank} [{sc.status.value}] {sc.candidate.name}: {self.target} = {expr_txt} "
                    f"(R²={sc.fit_result.r2:.5f}, RMSE={sc.fit_result.rmse:.4g}, BIC={sc.fit_result.bic:.2f})"
                )
            else:
                lines.append(f"  #{rank} [{sc.status.value}] {sc.candidate.name}: {self.target} = {expr_txt}")
        if self.rejected_candidates:
            lines.append("  Rejected:")
            for rc in self.rejected_candidates:
                lines.append(
                    f"    - {rc.candidate.name} @ {rc.rejection_stage}: {rc.rejection_reason}"
                )
        return "\n".join(lines)

    def _repr_markdown_(self) -> str:
        lines = [
            f"### Symbolic Regression Screening Report (`{self.target}`) — **{self.status.value.upper()}**",
            f"- Source: `{self.source}` | Imported: `{self.total_imported}` | Admissible: `{len(self.admissible_candidates)}` | Rejected: `{len(self.rejected_candidates)}`",
            "",
            "| Rank | Candidate | Status | Expression | R² | RMSE | BIC |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for rank, sc in enumerate(self.admissible_candidates, start=1):
            expr_txt = sc.calibrated_expression or sc.parsed_expression or sc.candidate.raw_equation
            r2_s = f"{sc.fit_result.r2:.5f}" if sc.fit_result is not None else "—"
            rmse_s = f"{sc.fit_result.rmse:.4g}" if sc.fit_result is not None else "—"
            bic_s = f"{sc.fit_result.bic:.2f}" if sc.fit_result is not None else "—"
            lines.append(f"| {rank} | `{sc.candidate.name}` | **{sc.status.value}** | `{expr_txt}` | {r2_s} | {rmse_s} | {bic_s} |")
        return "\n".join(lines)

    def to_claim_manifest(
        self,
        *,
        study_title: Optional[str] = None,
        top_k: int = 3,
    ) -> "ClaimManifest":
        """Convert top admissible screened candidates into a Sprint 10 ``ClaimManifest``."""
        from ..claims.schema import ClaimKind, ClaimManifest, ClaimSpec

        claims: List[ClaimSpec] = []
        for idx, sc in enumerate(self.admissible_candidates[: max(1, int(top_k))], start=1):
            rhs_txt = sc.calibrated_expression or sc.parsed_expression or sc.candidate.raw_equation
            eq_str = f"{self.target} = {rhs_txt}"
            claimed_metrics: Dict[str, float] = {}
            if sc.fit_result is not None:
                claimed_metrics["r2"] = float(sc.fit_result.r2)
                claimed_metrics["rmse"] = float(sc.fit_result.rmse)
            claims.append(
                ClaimSpec(
                    id=f"SR_{idx:02d}_{sc.candidate.name}",
                    title=f"Screened Symbolic Candidate #{idx}: {self.target} = {rhs_txt}",
                    kind=ClaimKind.EQUATION_LAW,
                    source_excerpt=sc.candidate.raw_equation,
                    equations=[eq_str],
                    variables_units=dict(self.dims),
                    operating_envelope=dict(self.operating_envelope) if self.operating_envelope else None,
                    claimed_metrics=claimed_metrics,
                )
            )
        return ClaimManifest(
            study_title=study_title or f"Screened Symbolic Candidates ({self.target})",
            source_reference=self.source,
            claims=claims,
        )


def _lift_float_constants(expr: sp.Expr) -> Tuple[sp.Expr, List[sp.Symbol], Dict[str, float]]:
    """Replace non-integer floating-point numeric literals in ``expr`` with refittable parameters ``p_0, p_1, ...``.

    Integer exponents (like ``v**2``) and exact small integers inside powers are preserved so
    dimensional exponents remain fixed while multiplicative/additive coefficients are refit via WLS.
    """
    params: List[sp.Symbol] = []
    init_vals: Dict[str, float] = {}

    def _walk(node: sp.Expr, in_exponent: bool = False) -> sp.Expr:
        if isinstance(node, sp.Float) and not in_exponent:
            p_name = f"p_{len(params)}"
            sym = real_symbol(p_name)
            params.append(sym)
            init_vals[p_name] = float(node)
            return sym
        if isinstance(node, sp.Pow):
            base_new = _walk(node.base, in_exponent=False)
            exp_new = _walk(node.exp, in_exponent=True)
            return sp.Pow(base_new, exp_new)
        if node.args:
            return node.func(*[_walk(a, in_exponent=in_exponent) for a in node.args])
        return node

    lifted = _walk(expr)
    return lifted, params, init_vals


def _coerce_data_map(
    data: Optional[Union[Dataset, Mapping[str, Sequence[float]]]],
) -> Optional[Dict[str, np.ndarray]]:
    if data is None:
        return None
    if isinstance(data, Dataset):
        return data.as_dict()
    return {str(k): np.asarray(v, dtype=float) for k, v in data.items()}


def _infer_envelope_from_data(
    data_map: Mapping[str, np.ndarray],
    target: str,
    dims: Mapping[str, Any],
) -> Dict[str, Tuple[float, float]]:
    env: Dict[str, Tuple[float, float]] = {}
    for key in dims:
        if key in data_map:
            arr = np.asarray(data_map[key], dtype=float)
            finite = arr[np.isfinite(arr)]
            if finite.size > 0:
                env[key] = (float(np.min(finite)), float(np.max(finite)))
    if target in data_map and target not in env:
        arr = np.asarray(data_map[target], dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            env[target] = (float(np.min(finite)), float(np.max(finite)))
    return env


class CandidateImporter:
    """Deterministic gatekeeper importing and screening external symbolic-regression candidates."""

    @staticmethod
    def _parse_and_remap(
        raw_eq: str,
        *,
        variable_map: Optional[Mapping[str, str]] = None,
    ) -> sp.Expr:
        txt = str(raw_eq).strip()
        if not txt:
            raise ValueError("Empty candidate equation string.")
        if "=" in txt:
            _, txt = txt.split("=", 1)
            txt = txt.strip()
        # Normalize caret exponentiation occasionally emitted by external SR tools
        expr = parse_expr(txt)
        if variable_map:
            subs = {real_symbol(k): real_symbol(v) for k, v in variable_map.items()}
            expr = expr.subs(subs)
        return expr

    @classmethod
    def screen_candidates(
        cls,
        candidates: Sequence[ImportedCandidate],
        *,
        target: str,
        dims: Mapping[str, Union[str, Dimension]],
        data: Optional[Union[Dataset, Mapping[str, Sequence[float]]]] = None,
        operating_envelope: Optional[Mapping[str, Union[float, Tuple[float, float]]]] = None,
        variable_map: Optional[Mapping[str, str]] = None,
        sigma_y: Optional[Union[str, Sequence[float], np.ndarray]] = None,
        weights: Optional[Union[str, Sequence[float], np.ndarray]] = None,
        missing_policy: str = "raise",
        refit_numeric_constants: bool = True,
        scale_threshold: float = 100.0,
        reject_non_identifiable: bool = False,
        source: str = "candidates",
    ) -> CandidateScreeningReport:
        """Screen imported candidates through AST -> Dimensions -> Scale Balance -> WLS/Identifiability."""
        from ..claims.scales import check_equation_scales, diagnose_normalization_shortcuts
        from ..numerical import NumericalSolver

        dim_strs = {k: (str(v) if isinstance(v, Dimension) else str(v)) for k, v in dims.items()}
        if target not in dim_strs:
            raise KeyError(f"Target symbol '{target}' must be present in dims={list(dim_strs.keys())}.")
        # Validate units up front and preserve resolved SI Dimension objects for scales/normalization
        parsed_dims: Dict[str, Dimension] = {
            k: (v if isinstance(v, Dimension) else Dimension.from_unit(str(v))) for k, v in dims.items()
        }

        data_map = _coerce_data_map(data)
        env: Dict[str, Tuple[float, float]] = {}
        if operating_envelope is not None:
            for k, v in operating_envelope.items():
                if isinstance(v, (int, float)):
                    fv = float(v)
                    env[str(k)] = (fv, fv)
                else:
                    lo, hi = float(v[0]), float(v[1])
                    env[str(k)] = (min(lo, hi), max(lo, hi))
        elif data_map is not None:
            env = _infer_envelope_from_data(data_map, target, dim_strs)

        sigma_arr: Optional[np.ndarray] = None
        if sigma_y is not None:
            if isinstance(sigma_y, str):
                if data_map is None or sigma_y not in data_map:
                    raise KeyError(f"Uncertainty column '{sigma_y}' not found in data.")
                sigma_arr = data_map[sigma_y]
            else:
                sigma_arr = np.asarray(sigma_y, dtype=float)
        elif isinstance(data, Dataset) and data.uncertainty_for(target) is not None:
            sigma_arr = data.uncertainty_for(target)

        weights_arr: Optional[np.ndarray] = None
        if weights is not None:
            if isinstance(weights, str):
                if data_map is None or weights not in data_map:
                    raise KeyError(f"Weights column '{weights}' not found in data.")
                weights_arr = data_map[weights]
            else:
                weights_arr = np.asarray(weights, dtype=float)

        admissible: List[ScreenedCandidate] = []
        rejected: List[ScreenedCandidate] = []
        target_sym = real_symbol(target)

        for cand in candidates:
            stages: Dict[str, str] = {}
            warns: List[str] = []

            # --------------------------------------------------------------
            # Gate 1: Safe AST Parsing (P0.5)
            # --------------------------------------------------------------
            try:
                expr = cls._parse_and_remap(cand.raw_equation, variable_map=variable_map)
                stages["ast_parse"] = "passed"
            except Exception as exc:
                stages["ast_parse"] = "failed"
                rejected.append(
                    ScreenedCandidate(
                        candidate=cand,
                        admissible=False,
                        status=VerificationStatus.FAILED,
                        stage_results=stages,
                        rejection_stage="ast_parse",
                        rejection_reason=f"AST parse rejected: {exc}",
                    )
                )
                continue

            # Check for undeclared free symbols
            free_names = sorted(str(s) for s in expr.free_symbols)
            unknown_syms = [s for s in free_names if s not in dim_strs]
            if unknown_syms:
                # Allow symbolic parameters only if we also have data AND they aren't dimensional mismatches;
                # if no dims provided for them, treat as dimensionless parameters only when explicitly prefixed p_/c_
                non_param_unknown = [s for s in unknown_syms if not (s.startswith("p_") or s.startswith("c_"))]
                if non_param_unknown:
                    stages["dimensions"] = "failed"
                    rejected.append(
                        ScreenedCandidate(
                            candidate=cand,
                            admissible=False,
                            status=VerificationStatus.FAILED,
                            stage_results=stages,
                            parsed_expression=str(expr),
                            rejection_stage="dimensions",
                            rejection_reason=f"Undeclared symbol(s) without dimensions: {non_param_unknown}",
                        )
                    )
                    continue

            # Build per-candidate dims including dimensionless p_/c_ parameters
            cand_dims: Dict[str, Dimension] = dict(parsed_dims)
            for s in unknown_syms:
                cand_dims[s] = Dimension.dimensionless()

            # --------------------------------------------------------------
            # Gate 2: Dimensional Homogeneity & Normalization Audit
            # --------------------------------------------------------------
            eq_str = f"{target} = {expr}"
            norm_rep = diagnose_normalization_shortcuts(eq_str, cand_dims)
            if not norm_rep.consistent:
                stages["dimensions"] = "failed"
                hyps = [h.to_dict() for h in norm_rep.hypotheses]
                reason = (
                    norm_rep.hypotheses[0].explanation
                    if norm_rep.hypotheses
                    else "; ".join(norm_rep.warnings) or "Dimensionally inconsistent candidate."
                )
                rejected.append(
                    ScreenedCandidate(
                        candidate=cand,
                        admissible=False,
                        status=VerificationStatus.FAILED,
                        stage_results=stages,
                        parsed_expression=str(expr),
                        rejection_stage="dimensions",
                        rejection_reason=reason,
                        dimensional_consistent=False,
                        normalization_hypotheses=hyps,
                        warnings=list(norm_rep.warnings),
                    )
                )
                continue
            stages["dimensions"] = "passed"

            # --------------------------------------------------------------
            # Gate 3: Scale-Separation Audit (Sprint 10)
            # --------------------------------------------------------------
            scale_gap: Optional[float] = None
            scale_code: Optional[str] = None
            scale_status = VerificationStatus.PASSED
            if env and all(s in env for s in [target] + [n for n in free_names if n in dim_strs]):
                cand_env = dict(env)
                for s in unknown_syms:
                    cand_env.setdefault(s, (1.0, 1.0))
                try:
                    sc_rep = check_equation_scales(
                        eq_str,
                        cand_dims,
                        cand_env,
                        scale_threshold=scale_threshold,
                    )
                    scale_gap = float(sc_rep.log10_gap) if np.isfinite(sc_rep.log10_gap) else None
                    scale_code = sc_rep.diagnostic_code
                    scale_status = sc_rep.status
                    warns.extend(sc_rep.warnings)
                except Exception as exc:
                    scale_status = VerificationStatus.WARNING
                    warns.append(f"Scale audit skipped due to evaluation error: {exc}")

                if scale_status == VerificationStatus.FAILED:
                    stages["scales"] = "failed"
                    rejected.append(
                        ScreenedCandidate(
                            candidate=cand,
                            admissible=False,
                            status=VerificationStatus.FAILED,
                            stage_results=stages,
                            parsed_expression=str(expr),
                            rejection_stage="scales",
                            rejection_reason=(
                                f"Scale-separation failure ({scale_code}, log10_gap={scale_gap:.2f})"
                                if scale_gap is not None
                                else f"Scale-separation failure ({scale_code})"
                            ),
                            dimensional_consistent=True,
                            scale_log10_gap=scale_gap,
                            scale_diagnostic_code=scale_code,
                            warnings=warns,
                        )
                    )
                    continue
                stages["scales"] = "warning" if scale_status == VerificationStatus.WARNING else "passed"
            else:
                stages["scales"] = "not_assessed"

            # --------------------------------------------------------------
            # Gate 4: WLS Calibration & Jacobian Identifiability (P2.1)
            # --------------------------------------------------------------
            fit_res: Optional[FitResult] = None
            cal_expr_str: Optional[str] = str(expr)
            cal_status = VerificationStatus.PASSED

            if data_map is not None:
                if target not in data_map:
                    raise KeyError(f"Target column '{target}' missing from data.")
                try:
                    fit_expr = expr
                    param_syms: List[sp.Symbol] = [real_symbol(s) for s in unknown_syms]
                    init_p: Dict[str, float] = {s: 1.0 for s in unknown_syms}

                    if refit_numeric_constants and not param_syms:
                        fit_expr, lifted_syms, lifted_init = _lift_float_constants(expr)
                        param_syms = lifted_syms
                        init_p = lifted_init

                    ce = CandidateEquation(
                        name=cand.name,
                        kind="imported_sr",
                        equation=sp.Eq(target_sym, fit_expr),
                        rhs=fit_expr,
                        free_params=param_syms,
                    )
                    fit_res = NumericalSolver.fit(
                        ce,
                        data_map,
                        target_col=target,
                        initial_guess=init_p or None,
                        sigma_y=sigma_arr,
                        weights=weights_arr,
                        missing_policy=missing_policy,
                    )
                    cal_expr_str = str(fit_res.calibrated_equation.rhs)
                    warns.extend(fit_res.warnings)

                    if not fit_res.converged or not np.isfinite(fit_res.rmse):
                        stages["calibration"] = "failed"
                        rejected.append(
                            ScreenedCandidate(
                                candidate=cand,
                                admissible=False,
                                status=VerificationStatus.FAILED,
                                stage_results=stages,
                                parsed_expression=str(expr),
                                rejection_stage="calibration",
                                rejection_reason=f"WLS/NLLS fit did not converge: {fit_res.message}",
                                dimensional_consistent=True,
                                scale_log10_gap=scale_gap,
                                scale_diagnostic_code=scale_code,
                                fit_result=fit_res,
                                warnings=warns,
                            )
                        )
                        continue

                    if fit_res.identifiability is not None and not fit_res.identifiability.identifiable:
                        warns.append(
                            f"Non-identifiable parameterization (cond={fit_res.identifiability.condition_number:.2e}, "
                            f"rank={fit_res.identifiability.numerical_rank}/{fit_res.identifiability.n_params})."
                        )
                        if reject_non_identifiable:
                            stages["calibration"] = "failed"
                            rejected.append(
                                ScreenedCandidate(
                                    candidate=cand,
                                    admissible=False,
                                    status=VerificationStatus.FAILED,
                                    stage_results=stages,
                                    parsed_expression=str(expr),
                                    calibrated_expression=cal_expr_str,
                                    rejection_stage="calibration",
                                    rejection_reason="Rejected due to non-identifiable / collinear parameters.",
                                    dimensional_consistent=True,
                                    scale_log10_gap=scale_gap,
                                    scale_diagnostic_code=scale_code,
                                    fit_result=fit_res,
                                    warnings=warns,
                                )
                            )
                            continue
                        cal_status = VerificationStatus.WARNING
                    stages["calibration"] = "warning" if cal_status == VerificationStatus.WARNING else "passed"
                except Exception as exc:
                    stages["calibration"] = "failed"
                    rejected.append(
                        ScreenedCandidate(
                            candidate=cand,
                            admissible=False,
                            status=VerificationStatus.FAILED,
                            stage_results=stages,
                            parsed_expression=str(expr),
                            rejection_stage="calibration",
                            rejection_reason=f"Calibration error: {exc}",
                            dimensional_consistent=True,
                            scale_log10_gap=scale_gap,
                            scale_diagnostic_code=scale_code,
                            warnings=warns,
                        )
                    )
                    continue
            else:
                stages["calibration"] = "not_assessed"

            final_status = (
                VerificationStatus.WARNING
                if (scale_status == VerificationStatus.WARNING or cal_status == VerificationStatus.WARNING)
                else VerificationStatus.PASSED
            )
            admissible.append(
                ScreenedCandidate(
                    candidate=cand,
                    admissible=True,
                    status=final_status,
                    stage_results=stages,
                    parsed_expression=str(expr),
                    calibrated_expression=cal_expr_str,
                    dimensional_consistent=True,
                    scale_log10_gap=scale_gap,
                    scale_diagnostic_code=scale_code,
                    fit_result=fit_res,
                    warnings=warns,
                )
            )

        # Rank admissible candidates: lowest BIC -> lowest RMSE -> lowest complexity
        admissible.sort(
            key=lambda sc: (
                sc.bic,
                sc.rmse,
                sc.candidate.complexity if sc.candidate.complexity is not None else 9999,
                sc.candidate.index,
            )
        )

        if not admissible:
            overall = VerificationStatus.FAILED
            rep_warns = ["Zero imported candidates passed dimensional, scale, and calibration gates."]
        elif admissible[0].status == VerificationStatus.WARNING:
            overall = VerificationStatus.WARNING
            rep_warns = list(admissible[0].warnings)
        else:
            overall = VerificationStatus.PASSED
            rep_warns = []

        return CandidateScreeningReport(
            target=target,
            source=source,
            total_imported=len(candidates),
            admissible_candidates=admissible,
            rejected_candidates=rejected,
            status=overall,
            warnings=rep_warns,
            dims=dim_strs,
            operating_envelope=env,
        )

    @classmethod
    def from_expressions(
        cls,
        equations: Sequence[str],
        *,
        target: str,
        dims: Mapping[str, Union[str, Dimension]],
        data: Optional[Union[Dataset, Mapping[str, Sequence[float]]]] = None,
        operating_envelope: Optional[Mapping[str, Union[float, Tuple[float, float]]]] = None,
        variable_map: Optional[Mapping[str, str]] = None,
        sigma_y: Optional[Union[str, Sequence[float], np.ndarray]] = None,
        weights: Optional[Union[str, Sequence[float], np.ndarray]] = None,
        missing_policy: str = "raise",
        refit_numeric_constants: bool = True,
        scale_threshold: float = 100.0,
        reject_non_identifiable: bool = False,
    ) -> CandidateScreeningReport:
        """Screen a sequence of raw symbolic candidate expression strings."""
        cands = [
            ImportedCandidate(
                index=i,
                name=f"cand_{i + 1:02d}",
                raw_equation=str(eq),
                source_format="expression_list",
            )
            for i, eq in enumerate(equations)
        ]
        return cls.screen_candidates(
            cands,
            target=target,
            dims=dims,
            data=data,
            operating_envelope=operating_envelope,
            variable_map=variable_map,
            sigma_y=sigma_y,
            weights=weights,
            missing_policy=missing_policy,
            refit_numeric_constants=refit_numeric_constants,
            scale_threshold=scale_threshold,
            reject_non_identifiable=reject_non_identifiable,
            source="expression_list",
        )

    @classmethod
    def from_pysr_csv(
        cls,
        csv_path: Union[str, Path],
        *,
        target: str,
        dims: Mapping[str, Union[str, Dimension]],
        data: Optional[Union[Dataset, Mapping[str, Sequence[float]]]] = None,
        operating_envelope: Optional[Mapping[str, Union[float, Tuple[float, float]]]] = None,
        variable_map: Optional[Mapping[str, str]] = None,
        sigma_y: Optional[Union[str, Sequence[float], np.ndarray]] = None,
        weights: Optional[Union[str, Sequence[float], np.ndarray]] = None,
        missing_policy: str = "raise",
        refit_numeric_constants: bool = True,
        scale_threshold: float = 100.0,
        reject_non_identifiable: bool = False,
    ) -> CandidateScreeningReport:
        """Import and screen a PySR ``hall_of_fame.csv`` Pareto-front file."""
        src = Path(csv_path)
        if not src.exists():
            raise FileNotFoundError(f"PySR hall-of-fame CSV not found: '{src.name}'.")

        cands: List[ImportedCandidate] = []
        with src.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames:
                raise ValueError(f"PySR CSV '{src.name}' has no header row.")
            col_lookup = {h.strip().lower(): h for h in reader.fieldnames if h is not None}
            eq_key = col_lookup.get("equation") or col_lookup.get("sympy_format") or col_lookup.get("expr")
            if eq_key is None:
                raise ValueError(
                    f"PySR CSV '{src.name}' missing 'Equation' column (found {list(reader.fieldnames)})."
                )
            comp_key = col_lookup.get("complexity")
            loss_key = col_lookup.get("loss")
            score_key = col_lookup.get("score")

            for idx, row in enumerate(reader):
                eq_val = (row.get(eq_key) or "").strip()
                if not eq_val:
                    continue
                comp = int(float(row[comp_key])) if (comp_key and row.get(comp_key, "").strip()) else None
                loss = float(row[loss_key]) if (loss_key and row.get(loss_key, "").strip()) else None
                score = float(row[score_key]) if (score_key and row.get(score_key, "").strip()) else None
                cands.append(
                    ImportedCandidate(
                        index=idx,
                        name=f"pysr_c{comp if comp is not None else idx + 1}",
                        raw_equation=eq_val,
                        complexity=comp,
                        pysr_loss=loss,
                        pysr_score=score,
                        source_format="pysr_csv",
                    )
                )

        return cls.screen_candidates(
            cands,
            target=target,
            dims=dims,
            data=data,
            operating_envelope=operating_envelope,
            variable_map=variable_map,
            sigma_y=sigma_y,
            weights=weights,
            missing_policy=missing_policy,
            refit_numeric_constants=refit_numeric_constants,
            scale_threshold=scale_threshold,
            reject_non_identifiable=reject_non_identifiable,
            source=src.name,
        )

    @classmethod
    def from_pysr_model(
        cls,
        model: Any,
        *,
        target: str,
        dims: Mapping[str, Union[str, Dimension]],
        data: Optional[Union[Dataset, Mapping[str, Sequence[float]]]] = None,
        operating_envelope: Optional[Mapping[str, Union[float, Tuple[float, float]]]] = None,
        variable_map: Optional[Mapping[str, str]] = None,
        sigma_y: Optional[Union[str, Sequence[float], np.ndarray]] = None,
        weights: Optional[Union[str, Sequence[float], np.ndarray]] = None,
        missing_policy: str = "raise",
        refit_numeric_constants: bool = True,
        scale_threshold: float = 100.0,
        reject_non_identifiable: bool = False,
    ) -> CandidateScreeningReport:
        """Screen Pareto-front equations from a fitted ``PySRRegressor`` (duck-typed via ``.equations_``)."""
        eq_table = getattr(model, "equations_", None)
        if eq_table is None:
            raise ValueError("PySR model has no '.equations_' attribute; fit the model first.")

        rows: List[Mapping[str, Any]]
        if hasattr(eq_table, "to_dict"):
            rows = list(eq_table.to_dict(orient="records"))
        elif isinstance(eq_table, Sequence):
            rows = [dict(r) for r in eq_table]
        else:
            raise TypeError(f"Unsupported '.equations_' table type: {type(eq_table)!r}.")

        cands: List[ImportedCandidate] = []
        for idx, row in enumerate(rows):
            low_map = {str(k).lower(): v for k, v in row.items()}
            eq_val = str(low_map.get("equation") or low_map.get("sympy_format") or "").strip()
            if not eq_val:
                continue
            comp = int(low_map["complexity"]) if low_map.get("complexity") is not None else None
            loss = float(low_map["loss"]) if low_map.get("loss") is not None else None
            score = float(low_map["score"]) if low_map.get("score") is not None else None
            cands.append(
                ImportedCandidate(
                    index=idx,
                    name=f"pysr_c{comp if comp is not None else idx + 1}",
                    raw_equation=eq_val,
                    complexity=comp,
                    pysr_loss=loss,
                    pysr_score=score,
                    source_format="pysr_model",
                )
            )

        return cls.screen_candidates(
            cands,
            target=target,
            dims=dims,
            data=data,
            operating_envelope=operating_envelope,
            variable_map=variable_map,
            sigma_y=sigma_y,
            weights=weights,
            missing_policy=missing_policy,
            refit_numeric_constants=refit_numeric_constants,
            scale_threshold=scale_threshold,
            reject_non_identifiable=reject_non_identifiable,
            source="pysr_model",
        )

    @classmethod
    def run_pysr(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        *,
        feature_names: Sequence[str],
        target: str,
        dims: Mapping[str, Union[str, Dimension]],
        pysr_kwargs: Optional[Mapping[str, Any]] = None,
        **screen_kwargs: Any,
    ) -> CandidateScreeningReport:
        """Optional convenience wrapper running ``pysr.PySRRegressor`` and gating its output.

        Requires optional dependency: ``pip install 'rosolver[pysr]'``.
        """
        try:
            from pysr import PySRRegressor  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Running PySR directly requires the optional 'pysr' dependency: pip install 'rosolver[pysr]'."
            ) from exc

        reg = PySRRegressor(**dict(pysr_kwargs or {}))
        reg.fit(X, y, variable_names=list(feature_names))
        data_map: Dict[str, np.ndarray] = {target: np.asarray(y, dtype=float)}
        X_arr = np.asarray(X, dtype=float)
        for i, fname in enumerate(feature_names):
            data_map[fname] = X_arr[:, i]
        return cls.from_pysr_model(reg, target=target, dims=dims, data=data_map, **screen_kwargs)
