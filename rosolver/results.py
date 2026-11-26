"""Standardized result, metadata, and JSON-serialization primitives (ENG_SPEC.md P0.1, P1.1)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import sympy as sp


class VerificationStatus(str, Enum):
    """Five-state scientific verification classification (P0.1)."""

    PASSED = "Passed"
    FAILED = "Failed"
    INCONCLUSIVE = "Inconclusive"
    NOT_ASSESSED = "NotAssessed"
    WARNING = "Warning"


def to_jsonable(obj: Any) -> Any:
    """Recursively convert scientific Python/NumPy/SymPy objects to JSON-compatible primitives."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if not np.isfinite(val) else val
    if isinstance(obj, np.complexfloating):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    if isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, sp.Basic):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(item) for item in obj]
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if callable(obj):
        return getattr(obj, "__name__", repr(obj))
    return str(obj)


class ResultMixin:
    """Mixin providing uniform dictionary and JSON serialization for result dataclasses."""

    def to_dict(self) -> Dict[str, Any]:
        if is_dataclass(self) and not isinstance(self, type):
            return {f.name: to_jsonable(getattr(self, f.name)) for f in fields(self)}
        return to_jsonable(vars(self))

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class ResultMetadata(ResultMixin):
    """Uniform metadata and provenance attached to standardized computational results."""

    kind: str
    status: VerificationStatus = VerificationStatus.PASSED
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    run_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    timing_s: Optional[float] = None
    solver_config: Dict[str, Any] = field(default_factory=dict)
    units: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StandardResult(ResultMixin):
    """Unified wrapper around any simulation, calibration, symbolic, or verification result."""

    kind: str
    status: VerificationStatus
    raw: Any
    metadata: ResultMetadata

    @property
    def passed(self) -> bool:
        return self.status == VerificationStatus.PASSED

    @property
    def warnings(self) -> List[str]:
        return self.metadata.warnings

    @property
    def assumptions(self) -> List[str]:
        return self.metadata.assumptions

    @property
    def solver_config(self) -> Dict[str, Any]:
        return self.metadata.solver_config

    @property
    def units(self) -> Dict[str, str]:
        return self.metadata.units

    @property
    def metrics(self) -> Dict[str, Any]:
        return self.metadata.metrics

    @property
    def timing_s(self) -> Optional[float]:
        return self.metadata.timing_s

    @property
    def run_id(self) -> Optional[str]:
        return self.metadata.run_id

    @property
    def model_version(self) -> Optional[str]:
        return self.metadata.model_version

    def summary(self) -> str:
        header = f"[{self.kind}] Status: {self.status.value}"
        if self.metadata.model_name:
            header += f" | Model: {self.metadata.model_name} (v{self.metadata.model_version or '0.1'})"
        if self.metadata.timing_s is not None:
            header += f" | Time: {self.metadata.timing_s:.4f}s"
        lines = [header]
        if hasattr(self.raw, "summary"):
            lines.append(self.raw.summary())
        elif self.warnings:
            for w in self.warnings:
                lines.append(f"  Warning: {w}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        raw_dict = (
            self.raw.to_dict()
            if hasattr(self.raw, "to_dict")
            else to_jsonable(self.raw)
        )
        return {
            "kind": self.kind,
            "status": self.status.value,
            "metadata": self.metadata.to_dict(),
            "result": raw_dict,
        }


def _infer_status(raw: Any) -> VerificationStatus:
    st = getattr(raw, "status", None)
    if isinstance(st, VerificationStatus):
        return st
    if hasattr(raw, "passed"):
        return VerificationStatus.PASSED if bool(raw.passed) else VerificationStatus.FAILED
    if hasattr(raw, "converged"):
        if not bool(raw.converged):
            return VerificationStatus.FAILED
        ident = getattr(raw, "identifiability", None)
        if ident is not None and not ident.identifiable:
            return VerificationStatus.WARNING
        return VerificationStatus.PASSED
    if hasattr(raw, "success"):
        if not bool(raw.success):
            return VerificationStatus.FAILED
        if getattr(raw, "warnings", None):
            return VerificationStatus.WARNING
        return VerificationStatus.PASSED
    return VerificationStatus.PASSED


def _infer_metrics(raw: Any) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for attr in (
        "nfev",
        "njev",
        "n_steps",
        "newton_failures",
        "cfl_max",
        "r2",
        "rmse",
        "mae",
        "aic",
        "bic",
        "objective_value",
        "observed_order",
        "gci_fine_21",
        "extrapolated_value",
        "max_error_vs_monolithic",
        "stiffness_ratio",
    ):
        if hasattr(raw, attr):
            val = getattr(raw, attr)
            if val is not None:
                metrics[attr] = to_jsonable(val)
    if hasattr(raw, "final") and not callable(raw.final):
        metrics["final"] = to_jsonable(raw.final)
    return metrics


def standardize_result(
    raw: Any,
    *,
    kind: Optional[str] = None,
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
    solver_config: Optional[Mapping[str, Any]] = None,
    units: Optional[Mapping[str, str]] = None,
    timing_s: Optional[float] = None,
    run_id: Optional[str] = None,
    assumptions: Optional[List[str]] = None,
) -> StandardResult:
    """Wrap any Rosolver computational output into a `StandardResult`."""
    if isinstance(raw, StandardResult):
        return raw

    inferred_kind = kind or type(raw).__name__
    status = _infer_status(raw)
    warnings = list(getattr(raw, "warnings", []) or [])
    raw_assumptions = list(getattr(raw, "assumptions", []) or [])
    merged_assumptions = list(dict.fromkeys(raw_assumptions + list(assumptions or [])))

    cfg = dict(solver_config or {})
    if not cfg:
        opt_cfg = getattr(raw, "optimizer_config", None)
        if isinstance(opt_cfg, Mapping):
            cfg.update(opt_cfg)
        for attr in ("method", "scheme", "reconstruction", "dt", "formal_order"):
            if hasattr(raw, attr) and getattr(raw, attr) is not None:
                cfg.setdefault(attr, to_jsonable(getattr(raw, attr)))

    meta = ResultMetadata(
        kind=inferred_kind,
        status=status,
        model_name=model_name,
        model_version=model_version,
        run_id=run_id,
        timing_s=timing_s,
        solver_config=to_jsonable(cfg),
        units={str(k): str(v) for k, v in (units or {}).items()},
        warnings=[str(w) for w in warnings],
        assumptions=[str(a) for a in merged_assumptions],
        metrics=_infer_metrics(raw),
    )
    return StandardResult(kind=inferred_kind, status=status, raw=raw, metadata=meta)
