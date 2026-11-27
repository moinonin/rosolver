"""Stateful Multi-Round Peer-Review Rebuttal & Transition Structures (Sprint 10 / S10.5).

Handles author rebuttals (uploaded CSV datasets, declared solver schemes, revised
constitutive equations), strict workspace path confinement, artifact-contract validation,
and immutable R1 -> R2 verdict transition tracking.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import yaml

from .._parsing import parse_expr
from ..results import ResultMixin, to_jsonable
from .clarifications import ArtifactRequirementSpec, ClarificationRequest
from .schema import ClaimVerdict, ClaimVerdictKind


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_confined_path(base_dir: Path, raw_path: Union[str, Path]) -> Path:
    """Resolve a user/rebuttal path strictly preventing `..` traversal outside base_dir.

    Absolute paths are rejected unless they already reside strictly inside `base_dir`.
    """
    base = base_dir.resolve()
    p = Path(raw_path)
    if ".." in p.parts:
        raise ValueError(f"Disallowed path traversal ('..') in rebuttal path '{raw_path}'.")
    candidate = p.resolve() if p.is_absolute() else (base / p).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError(
            f"Rebuttal path '{raw_path}' escapes project workspace confinement."
        ) from exc
    return candidate


@dataclass
class RebuttalSubmission(ResultMixin):
    """Structured author response targeting an open ClarificationRequest."""

    request_id: str
    claim_id: str
    response_type: str  # "dataset_upload" | "solver_metadata" | "constitutive_revision" | "parameter_clarification"
    author_note: str = ""
    dataset_path: Optional[str] = None
    dataset_name: Optional[str] = None
    declared_scheme: Optional[str] = None
    refinement_dts: List[float] = field(default_factory=list)
    revised_equation: Optional[str] = None
    revised_units: Dict[str, str] = field(default_factory=dict)
    revised_envelope: Dict[str, List[float]] = field(default_factory=dict)
    modifies_model_parameters: bool = False

    def __post_init__(self) -> None:
        if not self.request_id or not self.claim_id:
            raise ValueError("RebuttalSubmission requires non-empty request_id and claim_id.")
        if self.revised_equation is not None:
            txt = self.revised_equation.strip()
            if "=" in txt:
                l_part, r_part = txt.split("=", 1)
                parse_expr(l_part.strip())
                parse_expr(r_part.strip())
            else:
                parse_expr(txt)
        if self.dataset_path is not None and ".." in Path(self.dataset_path).parts:
            raise ValueError(f"Disallowed path traversal ('..') in dataset_path '{self.dataset_path}'.")

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(
            {
                "request_id": self.request_id,
                "claim_id": self.claim_id,
                "response_type": self.response_type,
                "author_note": self.author_note,
                "dataset_path": self.dataset_path,
                "dataset_name": self.dataset_name,
                "declared_scheme": self.declared_scheme,
                "refinement_dts": list(self.refinement_dts),
                "revised_equation": self.revised_equation,
                "revised_units": dict(self.revised_units),
                "revised_envelope": dict(self.revised_envelope),
                "modifies_model_parameters": bool(self.modifies_model_parameters),
            }
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RebuttalSubmission":
        return cls(
            request_id=str(data["request_id"]),
            claim_id=str(data["claim_id"]),
            response_type=str(data.get("response_type", "dataset_upload")),
            author_note=str(data.get("author_note", "")),
            dataset_path=data.get("dataset_path"),
            dataset_name=data.get("dataset_name"),
            declared_scheme=data.get("declared_scheme"),
            refinement_dts=[float(x) for x in (data.get("refinement_dts") or [])],
            revised_equation=data.get("revised_equation"),
            revised_units=dict(data.get("revised_units") or {}),
            revised_envelope=dict(data.get("revised_envelope") or {}),
            modifies_model_parameters=bool(data.get("modifies_model_parameters", False)),
        )

    @classmethod
    def load_list(cls, path: Union[str, Path]) -> List["RebuttalSubmission"]:
        p = Path(path)
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        if isinstance(raw, Mapping) and "submissions" in raw:
            items = raw["submissions"]
        elif isinstance(raw, list):
            items = raw
        else:
            raise ValueError(f"Malformed rebuttal file '{p.name}': expected list or mapping with 'submissions'.")
        return [cls.from_dict(item) for item in items]


def validate_rebuttal_dataset_csv(
    csv_path: Path,
    requirement: ArtifactRequirementSpec,
) -> List[str]:
    """Validate an uploaded rebuttal CSV against an ArtifactRequirementSpec contract."""
    errors: List[str] = []
    if not csv_path.exists():
        return [f"Rebuttal dataset file not found: '{csv_path.name}'."]

    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = [c.strip() for c in (reader.fieldnames or []) if c]
        rows = list(reader)

    missing_cols = [c for c in requirement.required_columns if c not in cols]
    if missing_cols:
        errors.append(f"Uploaded CSV missing required column(s): {missing_cols} (found: {cols}).")

    if requirement.uncertainty_columns_required:
        u_cols = [c for c in cols if c.startswith("u_") or c.startswith("sigma_")]
        if not u_cols:
            errors.append(
                "Uploaded CSV must include measurement uncertainty column(s) prefixed with 'u_' or 'sigma_'."
            )

    min_needed = max(requirement.min_rows, requirement.min_replicates)
    if min_needed > 0 and len(rows) < min_needed:
        errors.append(
            f"Uploaded CSV has {len(rows)} row(s), fewer than required minimum {min_needed}."
        )
    return errors


@dataclass
class VerdictTransition(ResultMixin):
    """Records how a claim's verdict changed between review rounds R_k -> R_{k+1}."""

    claim_id: str
    prev_verdict: str
    new_verdict: str
    resolved_request_ids: List[str] = field(default_factory=list)
    remaining_request_ids: List[str] = field(default_factory=list)
    post_hoc_recalibration: bool = False
    notes: str = ""


@dataclass
class ReviewRoundRecord(ResultMixin):
    """Immutable snapshot of a deterministic claim-verification peer-review round."""

    round_index: int
    round_id: str
    parent_round_id: Optional[str]
    manifest_hash: str
    timestamp: str
    verdicts: List[ClaimVerdict]
    open_requests: List[ClarificationRequest]
    transitions: List[VerdictTransition] = field(default_factory=list)
    rebuttal_hashes: Dict[str, str] = field(default_factory=dict)

    @property
    def summary_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {k.value: 0 for k in ClaimVerdictKind}
        for v in self.verdicts:
            counts[v.overall_verdict.value] = counts.get(v.overall_verdict.value, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(
            {
                "round_index": self.round_index,
                "round_id": self.round_id,
                "parent_round_id": self.parent_round_id,
                "manifest_hash": self.manifest_hash,
                "timestamp": self.timestamp,
                "summary_counts": self.summary_counts,
                "verdicts": [v.to_dict() for v in self.verdicts],
                "open_requests": [r.to_dict() for r in self.open_requests],
                "transitions": [t.to_dict() for t in self.transitions],
                "rebuttal_hashes": dict(self.rebuttal_hashes),
            }
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReviewRoundRecord":
        return cls(
            round_index=int(data["round_index"]),
            round_id=str(data["round_id"]),
            parent_round_id=data.get("parent_round_id"),
            manifest_hash=str(data["manifest_hash"]),
            timestamp=str(data.get("timestamp", _utc_now())),
            verdicts=[ClaimVerdict.from_dict(v) for v in (data.get("verdicts") or [])],
            open_requests=[ClarificationRequest.from_dict(r) for r in (data.get("open_requests") or [])],
            transitions=[
                VerdictTransition(
                    claim_id=str(t["claim_id"]),
                    prev_verdict=str(t["prev_verdict"]),
                    new_verdict=str(t["new_verdict"]),
                    resolved_request_ids=list(t.get("resolved_request_ids") or []),
                    remaining_request_ids=list(t.get("remaining_request_ids") or []),
                    post_hoc_recalibration=bool(t.get("post_hoc_recalibration", False)),
                    notes=str(t.get("notes", "")),
                )
                for t in (data.get("transitions") or [])
            ],
            rebuttal_hashes=dict(data.get("rebuttal_hashes") or {}),
        )
