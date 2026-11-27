"""Deterministic Peer-Review Clarification Request Generator (Sprint 10 / S10.4).

Translates every non-Supported scientific claim audit finding (`Failed`, `Inconclusive`,
`Warning`, `SchemeDependent`, or `NotAssessed`) into structured, quantitative,
machine-verifiable referee queries with zero LLM randomness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..results import ResultMixin, VerificationStatus, to_jsonable
from .schema import ClaimVerdict, ClaimVerdictKind


@dataclass
class ArtifactRequirementSpec(ResultMixin):
    """Machine-verifiable contract specifying what an author rebuttal must supply."""

    kind: str  # "experimental_dataset" | "refinement_series" | "constitutive_equation" | "solver_metadata" | "uncertainty_specification"
    required_columns: List[str] = field(default_factory=list)
    expected_units: Dict[str, str] = field(default_factory=dict)
    min_rows: int = 0
    min_replicates: int = 0
    uncertainty_columns_required: bool = False
    required_fields: List[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(
            {
                "kind": self.kind,
                "required_columns": list(self.required_columns),
                "expected_units": dict(self.expected_units),
                "min_rows": int(self.min_rows),
                "min_replicates": int(self.min_replicates),
                "uncertainty_columns_required": bool(self.uncertainty_columns_required),
                "required_fields": list(self.required_fields),
                "description": self.description,
            }
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactRequirementSpec":
        return cls(
            kind=str(data["kind"]),
            required_columns=list(data.get("required_columns") or []),
            expected_units=dict(data.get("expected_units") or {}),
            min_rows=int(data.get("min_rows", 0)),
            min_replicates=int(data.get("min_replicates", 0)),
            uncertainty_columns_required=bool(data.get("uncertainty_columns_required", False)),
            required_fields=list(data.get("required_fields") or []),
            description=str(data.get("description", "")),
        )


@dataclass
class ClarificationRequest(ResultMixin):
    """Structured referee clarification request bound to an open audit trigger."""

    request_id: str
    round_index: int
    claim_id: str
    severity: str  # "major_falsification" | "major_missing_data" | "major_methodological" | "minor_clarification"
    trigger_code: str
    trigger_status: str
    quantitative_findings: Dict[str, Any]
    reviewer_prompt: str
    required_artifact: ArtifactRequirementSpec
    resolution_gate: Dict[str, Any] = field(default_factory=dict)
    status: str = "open"  # "open" | "resolved" | "superseded"

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(
            {
                "request_id": self.request_id,
                "round_index": int(self.round_index),
                "claim_id": self.claim_id,
                "severity": self.severity,
                "trigger_code": self.trigger_code,
                "trigger_status": self.trigger_status,
                "quantitative_findings": dict(self.quantitative_findings),
                "reviewer_prompt": self.reviewer_prompt,
                "required_artifact": self.required_artifact.to_dict(),
                "resolution_gate": dict(self.resolution_gate),
                "status": self.status,
            }
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClarificationRequest":
        return cls(
            request_id=str(data["request_id"]),
            round_index=int(data.get("round_index", 1)),
            claim_id=str(data["claim_id"]),
            severity=str(data["severity"]),
            trigger_code=str(data["trigger_code"]),
            trigger_status=str(data.get("trigger_status", "Inconclusive")),
            quantitative_findings=dict(data.get("quantitative_findings") or {}),
            reviewer_prompt=str(data["reviewer_prompt"]),
            required_artifact=ArtifactRequirementSpec.from_dict(data["required_artifact"]),
            resolution_gate=dict(data.get("resolution_gate") or {}),
            status=str(data.get("status", "open")),
        )


def _build_request_for_trigger(
    *,
    request_id: str,
    round_index: int,
    verdict: ClaimVerdict,
    trigger_code: str,
    claim_observables: Optional[Sequence[str]] = None,
) -> ClarificationRequest:
    m = verdict.metrics
    cid = verdict.claim_id

    if trigger_code == "unbalanced_dominant_terms":
        gap_log = m.get("log10_gap", "N/A")
        ratio = m.get("dominant_balance_ratio", "N/A")
        return ClarificationRequest(
            request_id=request_id,
            round_index=round_index,
            claim_id=cid,
            severity="major_falsification",
            trigger_code=trigger_code,
            trigger_status=VerificationStatus.FAILED.value,
            quantitative_findings={"log10_gap": gap_log, "dominant_balance_ratio": ratio},
            reviewer_prompt=(
                f"[Major — Scale Imbalance] For claim '{verdict.title}' ({cid}), the governing "
                f"equation terms are separated by ~10^{gap_log}x across the stated operating envelope. "
                "Please clarify the physical definition, units, and constitutive rescaling linking the "
                "competing terms, or supply the revised energy/momentum balance equation."
            ),
            required_artifact=ArtifactRequirementSpec(
                kind="constitutive_equation",
                required_fields=["revised_equation", "variables_units", "operating_envelope"],
                description="Dimensionally and scale-consistent constitutive closure replacing or rescaling the unbalanced balance law.",
            ),
            resolution_gate={"tool": "check_equation_scales", "claim_id": cid},
        )

    if trigger_code in ("implicit_unit_normalization", "differential_geometry_mismatch", "dimensionally_inhomogeneous"):
        return ClarificationRequest(
            request_id=request_id,
            round_index=round_index,
            claim_id=cid,
            severity="major_falsification",
            trigger_code=trigger_code,
            trigger_status=VerificationStatus.FAILED.value,
            quantitative_findings={"normalization_hypotheses": m.get("normalization_hypotheses", [])},
            reviewer_prompt=(
                f"[Major — Dimensional Closure] Claim '{verdict.title}' ({cid}) exhibits a dimensional "
                f"or normalization inconsistency ({trigger_code}). Please explicitly state all hidden "
                "reference scales (e.g., saturation mass/length caps) and distinguish finite cross-sectional "
                "areas from differential volume elements."
            ),
            required_artifact=ArtifactRequirementSpec(
                kind="constitutive_equation",
                required_fields=["revised_equation", "variables_units"],
                description="Explicitly dimensioned equation with all reference scales declared.",
            ),
            resolution_gate={"tool": "diagnose_normalization_shortcuts", "claim_id": cid},
        )

    if trigger_code in ("scheme_unspecified_error_budget_confounded", "discretization_exceeds_claimed_error", "discretization_consumes_error_budget"):
        claimed = m.get("claimed_rel_error_pct", "N/A")
        worst_err = m.get("discretization_error_paper_pct", "N/A")
        budget_r = m.get("budget_ratio", "N/A")
        sev = "major_falsification" if trigger_code == "discretization_exceeds_claimed_error" else "major_methodological"
        return ClarificationRequest(
            request_id=request_id,
            round_index=round_index,
            claim_id=cid,
            severity=sev,
            trigger_code=trigger_code,
            trigger_status=verdict.status.value,
            quantitative_findings={
                "claimed_rel_error_pct": claimed,
                "discretization_error_paper_pct": worst_err,
                "budget_ratio": budget_r,
                "scheme_errors": m.get("scheme_errors", {}),
            },
            reviewer_prompt=(
                f"[Major — Numerical Discretization] Claim '{verdict.title}' ({cid}) reports "
                f"{claimed}% error, but discretization uncertainty at the stated step size reaches "
                f"{worst_err}% (budget ratio = {budget_r}). Please state the exact time/space integration "
                "scheme and provide a 3-level refinement series (e.g., n, 2n, 4n) for Roache/Celik GCI."
            ),
            required_artifact=ArtifactRequirementSpec(
                kind="solver_metadata",
                required_fields=["time_integrator"],
                required_columns=["dt", "qoi"],
                min_rows=3,
                description="Explicit numerical integrator declaration plus >= 3-level refinement series for GCI.",
            ),
            resolution_gate={"tool": "audit_claimed_accuracy", "claim_id": cid},
        )

    if trigger_code == "missing_empirical_dataset":
        obs = list(claim_observables or m.get("required_observables") or ["measured_value"])
        req_cols = ["sample_id"] + obs + [f"u_{c}" for c in obs]
        return ClarificationRequest(
            request_id=request_id,
            round_index=round_index,
            claim_id=cid,
            severity="major_missing_data",
            trigger_code=trigger_code,
            trigger_status=VerificationStatus.INCONCLUSIVE.value,
            quantitative_findings={
                "mechanistic_status": verdict.assessment.mechanistic_status.value,
                "required_observables": obs,
            },
            reviewer_prompt=(
                f"[Major — Empirical Data Request] Claim '{verdict.title}' ({cid}) cannot be verified "
                "from qualitative or visual description alone. Please upload raw replicate experimental CSV "
                f"measurements with uncertainties for observable(s) {obs} (required columns: {req_cols})."
            ),
            required_artifact=ArtifactRequirementSpec(
                kind="experimental_dataset",
                required_columns=req_cols,
                min_rows=3,
                min_replicates=3,
                uncertainty_columns_required=True,
                description=f"Replicate experimental CSV for observables {obs} with measurement uncertainty columns.",
            ),
            resolution_gate={"tool": "validate_results", "claim_id": cid},
        )

    if trigger_code == "missing_experimental_uncertainty":
        return ClarificationRequest(
            request_id=request_id,
            round_index=round_index,
            claim_id=cid,
            severity="major_methodological",
            trigger_code=trigger_code,
            trigger_status=VerificationStatus.WARNING.value,
            quantitative_findings={"dataset": m.get("dataset")},
            reviewer_prompt=(
                f"[Minor/Major — Uncertainty Budget] Claim '{verdict.title}' ({cid}) supplies "
                "experimental values without measurement uncertainty u_exp. Please provide replicate "
                "standard uncertainties so ASME V&V 20 combined validation bands u_val can be evaluated."
            ),
            required_artifact=ArtifactRequirementSpec(
                kind="uncertainty_specification",
                uncertainty_columns_required=True,
                min_replicates=3,
                description="Experimental measurement uncertainties u_exp matching dataset rows.",
            ),
            resolution_gate={"tool": "validate_results", "claim_id": cid},
        )

    if trigger_code in ("euler_homogeneity_violation", "collinear_pi_design"):
        return ClarificationRequest(
            request_id=request_id,
            round_index=round_index,
            claim_id=cid,
            severity="major_methodological",
            trigger_code=trigger_code,
            trigger_status=verdict.status.value,
            quantitative_findings={
                "max_abs_euler_residual": m.get("max_abs_euler_residual"),
                "pi_collinear": m.get("pi_collinear"),
            },
            reviewer_prompt=(
                f"[Major — Scaling & Identifiability] Claim '{verdict.title}' ({cid}) triggered "
                f"'{trigger_code}'. Please include the missing dimensional baseline scale (non-zero offset) "
                "or supply multi-factor experimental data exciting independent Buckingham-Pi groups."
            ),
            required_artifact=ArtifactRequirementSpec(
                kind="experimental_dataset",
                min_rows=6,
                description="Multi-factor dataset independently exciting Pi-groups or revised non-zero baseline model.",
            ),
            resolution_gate={"tool": "check_homogeneity_elasticities", "claim_id": cid},
        )

    # Generic fallback for validation failures or other non-Supported triggers
    return ClarificationRequest(
        request_id=request_id,
        round_index=round_index,
        claim_id=cid,
        severity="major_falsification" if verdict.overall_verdict == ClaimVerdictKind.FALSIFIED else "minor_clarification",
        trigger_code=trigger_code,
        trigger_status=verdict.status.value,
        quantitative_findings=dict(m),
        reviewer_prompt=(
            f"[Review Query] Claim '{verdict.title}' ({cid}) finished with verdict "
            f"'{verdict.overall_verdict.value}' (trigger: {trigger_code}). Please provide supporting "
            "data, model equations, or numerical diagnostics addressing this discrepancy."
        ),
        required_artifact=ArtifactRequirementSpec(
            kind="experimental_dataset",
            description="Supporting evidence resolving the flagged verification discrepancy.",
        ),
        resolution_gate={"tool": "audit_claims", "claim_id": cid},
    )


def generate_clarification_requests(
    verdicts: Sequence[ClaimVerdict],
    *,
    round_index: int = 1,
    observables_by_claim: Optional[Mapping[str, Sequence[str]]] = None,
) -> List[ClarificationRequest]:
    """Deterministically generate structured ClarificationRequest items for all non-Supported claims."""
    requests: List[ClarificationRequest] = []
    seq = 0
    obs_map = observables_by_claim or {}

    for v in verdicts:
        if v.overall_verdict == ClaimVerdictKind.SUPPORTED:
            continue
        codes = list(v.trigger_codes)
        if not codes:
            if v.overall_verdict == ClaimVerdictKind.INCONCLUSIVE:
                codes = ["missing_empirical_dataset"]
            elif v.overall_verdict == ClaimVerdictKind.FALSIFIED:
                codes = ["validation_failed"]
            else:
                codes = ["incomplete_evidence"]

        for code in codes:
            seq += 1
            req_id = f"CR-R{round_index}-{seq:03d}"
            requests.append(
                _build_request_for_trigger(
                    request_id=req_id,
                    round_index=round_index,
                    verdict=v,
                    trigger_code=code,
                    claim_observables=obs_map.get(v.claim_id),
                )
            )
    return requests
