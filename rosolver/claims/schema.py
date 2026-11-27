"""Declarative Scientific Claim Manifest and Two-Axis Verdict Schema (Sprint 10 / P5.4).

Separates mechanistic/theoretical verification (`mechanistic_status`) from empirical
evidence support (`empirical_status`) and enforces the Hard Anti-Override Invariant:
a `ClaimVerdict` cannot claim `Supported` without verified `ExecutionRecord` run IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import yaml

from .._parsing import parse_expr
from ..dimensions import Dimension
from ..results import ResultMixin, VerificationStatus, to_jsonable

_CLAIM_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}$")


class ClaimKind(str, Enum):
    """Canonical categories of verifiable scientific manuscript claims."""

    EQUATION_LAW = "equation_law"
    DIMENSIONAL_SCALING = "dimensional_scaling"
    NUMERICAL_ACCURACY = "numerical_accuracy"
    CONSERVATION_INVARIANT = "conservation_invariant"
    PARAMETER_CALIBRATION = "parameter_calibration"
    EMPIRICAL_COMPARISON = "empirical_comparison"
    QUALITATIVE_OBSERVATION = "qualitative_observation"


class ClaimVerdictKind(str, Enum):
    """Synthesized claim-level verdict derived strictly from two-axis V&V outcomes."""

    SUPPORTED = "Supported"
    FALSIFIED = "Falsified"
    PARTIALLY_SUPPORTED = "PartiallySupported"
    SCHEME_DEPENDENT = "SchemeDependent"
    INCONCLUSIVE = "Inconclusive"
    NOT_ASSESSED = "NotAssessed"


def _coerce_claim_kind(raw: Union[str, ClaimKind]) -> ClaimKind:
    if isinstance(raw, ClaimKind):
        return raw
    low = str(raw).strip().lower()
    for item in ClaimKind:
        if item.value == low or item.name.lower() == low:
            return item
    raise ValueError(f"Unknown ClaimKind '{raw}'. Allowed: {[k.value for k in ClaimKind]}")


def _coerce_verdict_kind(raw: Union[str, ClaimVerdictKind]) -> ClaimVerdictKind:
    if isinstance(raw, ClaimVerdictKind):
        return raw
    low = str(raw).strip().lower()
    for item in ClaimVerdictKind:
        if item.value.lower() == low or item.name.lower() == low:
            return item
    raise ValueError(f"Unknown ClaimVerdictKind '{raw}'.")


def _coerce_vstatus(raw: Union[str, VerificationStatus]) -> VerificationStatus:
    if isinstance(raw, VerificationStatus):
        return raw
    low = str(raw).strip().lower()
    for item in VerificationStatus:
        if item.value.lower() == low or item.name.lower() == low:
            return item
    raise ValueError(f"Unknown VerificationStatus '{raw}'.")


def _validate_equation_ast(eq_text: str) -> None:
    """Validate an equation or expression strictly through Rosolver's AST whitelist parser."""
    txt = str(eq_text).strip()
    if not txt:
        raise ValueError("Equation string cannot be empty.")
    if "=" in txt:
        parts = txt.split("=", 1)
        parse_expr(parts[0].strip())
        parse_expr(parts[1].strip())
    else:
        parse_expr(txt)


def synthesize_verdict(
    mechanistic: VerificationStatus,
    empirical: VerificationStatus,
    *,
    requires_empirical: bool = False,
    scheme_dependent: bool = False,
) -> ClaimVerdictKind:
    """Deterministically synthesize a two-axis claim verdict without LLM override.

    Rules:
      1. Any FAILED on either axis -> Falsified.
      2. Scheme-dependent discretization ambiguity -> SchemeDependent.
      3. Mechanistic PASSED + empirical required + empirical missing/inconclusive -> Inconclusive.
      4. Mechanistic PASSED + (not requires_empirical or empirical PASSED) -> Supported.
      5. Any WARNING (without FAILED) -> PartiallySupported.
      6. Any INCONCLUSIVE -> Inconclusive.
      7. Otherwise -> NotAssessed.
    """
    if mechanistic == VerificationStatus.FAILED or empirical == VerificationStatus.FAILED:
        return ClaimVerdictKind.FALSIFIED
    if scheme_dependent:
        return ClaimVerdictKind.SCHEME_DEPENDENT
    if requires_empirical and empirical in (VerificationStatus.NOT_ASSESSED, VerificationStatus.INCONCLUSIVE):
        return ClaimVerdictKind.INCONCLUSIVE
    if mechanistic == VerificationStatus.INCONCLUSIVE or empirical == VerificationStatus.INCONCLUSIVE:
        return ClaimVerdictKind.INCONCLUSIVE
    if mechanistic == VerificationStatus.WARNING or empirical == VerificationStatus.WARNING:
        return ClaimVerdictKind.PARTIALLY_SUPPORTED
    if mechanistic == VerificationStatus.PASSED and (not requires_empirical or empirical == VerificationStatus.PASSED):
        return ClaimVerdictKind.SUPPORTED
    if not requires_empirical and mechanistic == VerificationStatus.NOT_ASSESSED and empirical == VerificationStatus.PASSED:
        return ClaimVerdictKind.SUPPORTED
    return ClaimVerdictKind.NOT_ASSESSED


@dataclass
class TwoAxisAssessment(ResultMixin):
    """Explicit separation of theoretical/mechanistic consistency from empirical lab proof."""

    mechanistic_status: VerificationStatus = VerificationStatus.NOT_ASSESSED
    empirical_status: VerificationStatus = VerificationStatus.NOT_ASSESSED
    requires_empirical: bool = False
    scheme_dependent: bool = False
    mechanistic_notes: List[str] = field(default_factory=list)
    empirical_notes: List[str] = field(default_factory=list)

    @property
    def overall_verdict(self) -> ClaimVerdictKind:
        return synthesize_verdict(
            self.mechanistic_status,
            self.empirical_status,
            requires_empirical=self.requires_empirical,
            scheme_dependent=self.scheme_dependent,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mechanistic_status": self.mechanistic_status.value,
            "empirical_status": self.empirical_status.value,
            "requires_empirical": self.requires_empirical,
            "scheme_dependent": self.scheme_dependent,
            "overall_verdict": self.overall_verdict.value,
            "mechanistic_notes": list(self.mechanistic_notes),
            "empirical_notes": list(self.empirical_notes),
        }


@dataclass
class ClaimSpec(ResultMixin):
    """Declarative specification of a single scientific claim extracted from a study."""

    id: str
    title: str
    kind: ClaimKind
    source_excerpt: str = ""
    equations: List[str] = field(default_factory=list)
    variables_units: Dict[str, str] = field(default_factory=dict)
    operating_envelope: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    characteristic_scales: Dict[str, float] = field(default_factory=dict)
    discretization_spec: Dict[str, Any] = field(default_factory=dict)
    claimed_metrics: Dict[str, float] = field(default_factory=dict)
    elasticities: Dict[str, float] = field(default_factory=dict)
    target_symbol: Optional[str] = None
    expected_dataset: Optional[str] = None
    required_observables: List[str] = field(default_factory=list)
    requires_empirical: Optional[bool] = None
    audit_raw_units: bool = False
    model_name: Optional[str] = None
    qoi_field: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not _CLAIM_ID_RE.match(self.id):
            raise ValueError(f"Invalid claim id '{self.id}'.")
        self.kind = _coerce_claim_kind(self.kind)
        if not self.title or not str(self.title).strip():
            raise ValueError(f"Claim '{self.id}' must define a non-empty title.")

        # AST-validate all equations up front (P0.5)
        for eq in self.equations:
            _validate_equation_ast(eq)

        # Validate units via Dimension.from_unit unless explicitly auditing raw malformed units
        if not self.audit_raw_units:
            for sym, unit_str in self.variables_units.items():
                try:
                    Dimension.from_unit(unit_str)
                except Exception:
                    Dimension.parse(unit_str)

        # Normalize operating_envelope tuples
        norm_env: Dict[str, Tuple[float, float]] = {}
        for k, bounds in self.operating_envelope.items():
            if isinstance(bounds, (int, float)):
                v = float(bounds)
                norm_env[k] = (v, v)
            elif isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                lo, hi = float(bounds[0]), float(bounds[1])
                if lo > hi:
                    lo, hi = hi, lo
                norm_env[k] = (lo, hi)
            else:
                raise ValueError(f"Invalid operating_envelope entry for '{k}': {bounds!r}")
        self.operating_envelope = norm_env

        if self.requires_empirical is None:
            self.requires_empirical = self.kind in (
                ClaimKind.EMPIRICAL_COMPARISON,
                ClaimKind.QUALITATIVE_OBSERVATION,
                ClaimKind.PARAMETER_CALIBRATION,
            )

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(
            {
                "id": self.id,
                "title": self.title,
                "kind": self.kind.value,
                "source_excerpt": self.source_excerpt,
                "equations": list(self.equations),
                "variables_units": dict(self.variables_units),
                "operating_envelope": {k: [v[0], v[1]] for k, v in self.operating_envelope.items()},
                "characteristic_scales": dict(self.characteristic_scales),
                "discretization_spec": dict(self.discretization_spec),
                "claimed_metrics": dict(self.claimed_metrics),
                "elasticities": dict(self.elasticities),
                "target_symbol": self.target_symbol,
                "expected_dataset": self.expected_dataset,
                "required_observables": list(self.required_observables),
                "requires_empirical": bool(self.requires_empirical),
                "audit_raw_units": bool(self.audit_raw_units),
                "model_name": self.model_name,
                "qoi_field": self.qoi_field,
                "metadata": dict(self.metadata),
            }
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClaimSpec":
        return cls(
            id=str(data["id"]),
            title=str(data["title"]),
            kind=_coerce_claim_kind(data["kind"]),
            source_excerpt=str(data.get("source_excerpt", "")),
            equations=list(data.get("equations") or []),
            variables_units=dict(data.get("variables_units") or {}),
            operating_envelope=dict(data.get("operating_envelope") or {}),
            characteristic_scales={k: float(v) for k, v in (data.get("characteristic_scales") or {}).items()},
            discretization_spec=dict(data.get("discretization_spec") or {}),
            claimed_metrics={k: float(v) for k, v in (data.get("claimed_metrics") or {}).items()},
            elasticities={k: float(v) for k, v in (data.get("elasticities") or {}).items()},
            target_symbol=data.get("target_symbol"),
            expected_dataset=data.get("expected_dataset"),
            required_observables=list(data.get("required_observables") or []),
            requires_empirical=data.get("requires_empirical"),
            audit_raw_units=bool(data.get("audit_raw_units", False)),
            model_name=data.get("model_name"),
            qoi_field=data.get("qoi_field"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class ClaimManifest(ResultMixin):
    """Collection of scientific claims extracted from a manuscript or engineering report."""

    study_title: str
    source_reference: str = ""
    version: str = "1.0"
    claims: List[ClaimSpec] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.study_title or not self.study_title.strip():
            raise ValueError("ClaimManifest requires a non-empty study_title.")
        seen: set[str] = set()
        for c in self.claims:
            if c.id in seen:
                raise ValueError(f"Duplicate claim id '{c.id}' in ClaimManifest.")
            seen.add(c.id)

    def get_claim(self, claim_id: str) -> ClaimSpec:
        for c in self.claims:
            if c.id == claim_id:
                return c
        raise KeyError(f"Unknown claim id '{claim_id}'.")

    def content_hash(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "study_title": self.study_title,
            "source_reference": self.source_reference,
            "version": self.version,
            "claims": [c.to_dict() for c in self.claims],
            "metadata": to_jsonable(self.metadata),
        }

    def save(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() == ".json":
            p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        else:
            p.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        return p

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClaimManifest":
        return cls(
            study_title=str(data["study_title"]),
            source_reference=str(data.get("source_reference", "")),
            version=str(data.get("version", "1.0")),
            claims=[ClaimSpec.from_dict(item) for item in (data.get("claims") or [])],
            metadata=dict(data.get("metadata") or {}),
        )

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "ClaimManifest":
        p = Path(path)
        raw = p.read_text(encoding="utf-8")
        if p.suffix.lower() == ".json":
            loaded = json.loads(raw)
        else:
            loaded = yaml.safe_load(raw)
        if not isinstance(loaded, Mapping):
            raise ValueError(f"Malformed ClaimManifest at '{p.name}': expected mapping.")
        return cls.from_dict(loaded)


@dataclass
class ClaimVerdict(ResultMixin):
    """Deterministic claim-level audit outcome tied to verified execution ledger runs."""

    claim_id: str
    title: str
    kind: ClaimKind
    assessment: TwoAxisAssessment
    evidence_run_ids: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    trigger_codes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    summary_text: str = ""
    post_hoc_recalibration: bool = False

    def __post_init__(self) -> None:
        self.kind = _coerce_claim_kind(self.kind)
        # Hard Anti-Override Invariant: Supported requires at least one recorded run ID
        if self.overall_verdict == ClaimVerdictKind.SUPPORTED and not self.evidence_run_ids:
            raise ValueError(
                f"Anti-override violation for claim '{self.claim_id}': "
                "'Supported' verdict requires at least one verified evidence_run_id."
            )

    @property
    def overall_verdict(self) -> ClaimVerdictKind:
        return self.assessment.overall_verdict

    @property
    def status(self) -> VerificationStatus:
        """Map ClaimVerdictKind back onto Rosolver's 5-state VerificationStatus."""
        v = self.overall_verdict
        if v == ClaimVerdictKind.SUPPORTED:
            return VerificationStatus.PASSED
        if v == ClaimVerdictKind.FALSIFIED:
            return VerificationStatus.FAILED
        if v in (ClaimVerdictKind.PARTIALLY_SUPPORTED, ClaimVerdictKind.SCHEME_DEPENDENT):
            return VerificationStatus.WARNING
        if v == ClaimVerdictKind.INCONCLUSIVE:
            return VerificationStatus.INCONCLUSIVE
        return VerificationStatus.NOT_ASSESSED

    def verify_against_ledger(self, ledger_records: Sequence[Any]) -> None:
        """Verify every referenced evidence_run_id exists in the project's execution ledger."""
        by_id = {rec.id: rec for rec in ledger_records}
        missing = [rid for rid in self.evidence_run_ids if rid not in by_id]
        if missing:
            raise ValueError(
                f"Anti-override violation for claim '{self.claim_id}': "
                f"evidence_run_id(s) {missing} not found in Project execution ledger."
            )
        if self.overall_verdict == ClaimVerdictKind.SUPPORTED:
            failed_runs = [
                rid
                for rid in self.evidence_run_ids
                if _coerce_vstatus(by_id[rid].status) == VerificationStatus.FAILED
            ]
            if failed_runs:
                raise ValueError(
                    f"Anti-override violation for claim '{self.claim_id}': "
                    f"cannot declare Supported when linked ledger run(s) {failed_runs} FAILED."
                )

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(
            {
                "claim_id": self.claim_id,
                "title": self.title,
                "kind": self.kind.value,
                "overall_verdict": self.overall_verdict.value,
                "status": self.status.value,
                "assessment": self.assessment.to_dict(),
                "evidence_run_ids": list(self.evidence_run_ids),
                "metrics": dict(self.metrics),
                "trigger_codes": list(self.trigger_codes),
                "warnings": list(self.warnings),
                "summary_text": self.summary_text,
                "post_hoc_recalibration": self.post_hoc_recalibration,
            }
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClaimVerdict":
        assess_raw = data.get("assessment") or {}
        assess = TwoAxisAssessment(
            mechanistic_status=_coerce_vstatus(assess_raw.get("mechanistic_status", "NotAssessed")),
            empirical_status=_coerce_vstatus(assess_raw.get("empirical_status", "NotAssessed")),
            requires_empirical=bool(assess_raw.get("requires_empirical", False)),
            scheme_dependent=bool(assess_raw.get("scheme_dependent", False)),
            mechanistic_notes=list(assess_raw.get("mechanistic_notes") or []),
            empirical_notes=list(assess_raw.get("empirical_notes") or []),
        )
        verdict = cls(
            claim_id=str(data["claim_id"]),
            title=str(data.get("title", data["claim_id"])),
            kind=_coerce_claim_kind(data["kind"]),
            assessment=assess,
            evidence_run_ids=list(data.get("evidence_run_ids") or []),
            metrics=dict(data.get("metrics") or {}),
            trigger_codes=list(data.get("trigger_codes") or []),
            warnings=list(data.get("warnings") or []),
            summary_text=str(data.get("summary_text", "")),
            post_hoc_recalibration=bool(data.get("post_hoc_recalibration", False)),
        )
        # Reject forged overall_verdict strings that disagree with deterministic synthesis
        if "overall_verdict" in data:
            claimed = _coerce_verdict_kind(data["overall_verdict"])
            if claimed != verdict.overall_verdict:
                raise ValueError(
                    f"Anti-override violation: serialized overall_verdict='{claimed.value}' "
                    f"disagrees with synthesized verdict '{verdict.overall_verdict.value}'."
                )
        return verdict
