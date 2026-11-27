"""
Structured scientific reporting engine for Rosolver (ENG_SPEC.md P3.1).

Generates deterministic, reproducible scientific reports directly from structured
Rosolver result objects, Model specifications, and Project execution ledgers—never
by scraping plain-text `.summary()` output.

Reports support 8 canonical scientific scopes (`REPORT_TYPES`) and export to
Markdown, JSON, and YAML without requiring a GUI or network access.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import yaml

from ..results import ResultMixin, StandardResult, VerificationStatus, standardize_result, to_jsonable

if TYPE_CHECKING:
    from ..model import Model
    from ..project import Project


REPORT_TYPES: Tuple[str, ...] = (
    "symbolic_derivation",
    "dimensional_analysis",
    "calibration",
    "numerical_simulation",
    "code_verification",
    "solution_verification",
    "model_validation",
    "integrated_project",
    "claim_verification",
    "peer_review_letter",
)

_KIND_TO_REPORT_TYPE: Dict[str, str] = {
    "symbolic_derivation": "symbolic_derivation",
    "dimensional_analysis": "dimensional_analysis",
    "calibration": "calibration",
    "candidate_ranking": "calibration",
    "simulation": "numerical_simulation",
    "ode_simulation": "numerical_simulation",
    "pde_simulation": "numerical_simulation",
    "stefan_simulation": "numerical_simulation",
    "mms": "code_verification",
    "code_verification": "code_verification",
    "gci": "solution_verification",
    "conservation": "solution_verification",
    "timestep": "solution_verification",
    "stiffness": "solution_verification",
    "solution_verification": "solution_verification",
    "validation": "model_validation",
    "model_validation": "model_validation",
    "integrated_project": "integrated_project",
    "claim_audit": "claim_verification",
    "claim_scale_audit": "claim_verification",
    "claim_normalization_audit": "claim_verification",
    "claim_scaling_audit": "claim_verification",
    "claim_budget_audit": "claim_verification",
    "claim_verification": "claim_verification",
    "peer_review_letter": "peer_review_letter",
}

_STATUS_SEVERITY: Dict[VerificationStatus, int] = {
    VerificationStatus.NOT_ASSESSED: 0,
    VerificationStatus.PASSED: 1,
    VerificationStatus.WARNING: 2,
    VerificationStatus.INCONCLUSIVE: 3,
    VerificationStatus.FAILED: 4,
}

_DEFAULT_NON_CERTIFICATION_NOTE = (
    "Verification and validation classifications follow ASME V&V 20-2009 / Roache / Celik "
    "engineering methodology for traceability and transparency, and do not constitute formal "
    "regulatory or standards certification."
)


def _worst_status(statuses: Sequence[VerificationStatus]) -> VerificationStatus:
    if not statuses:
        return VerificationStatus.NOT_ASSESSED
    return max(statuses, key=lambda s: _STATUS_SEVERITY.get(s, 0))


def _coerce_status(raw: Any) -> VerificationStatus:
    if isinstance(raw, VerificationStatus):
        return raw
    if isinstance(raw, str):
        low = raw.strip().lower()
        for item in VerificationStatus:
            if item.value.lower() == low or item.name.lower() == low:
                return item
    return VerificationStatus.NOT_ASSESSED


def _safe_relpath(target: Union[str, Path], base: Optional[Path] = None) -> str:
    """Return a clean relative path string without exposing user home prefixes."""
    p = Path(target)
    if not p.is_absolute():
        return p.as_posix()
    anchor = (base or Path.cwd()).resolve()
    try:
        return p.resolve().relative_to(anchor).as_posix()
    except ValueError:
        try:
            rel = os.path.relpath(str(p.resolve()), str(anchor))
            return Path(rel).as_posix()
        except ValueError:
            return p.name


def _normalize_figures(
    figures: Optional[Sequence[Union[str, Path, Mapping[str, str]]]],
    base: Optional[Path] = None,
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if not figures:
        return out
    for idx, item in enumerate(figures, 1):
        if isinstance(item, Mapping):
            raw_path = str(item.get("path", ""))
            caption = str(item.get("caption") or f"Figure {idx}")
        else:
            raw_path = str(item)
            caption = Path(raw_path).stem.replace("_", " ").title() or f"Figure {idx}"
        if raw_path:
            out.append({"path": _safe_relpath(raw_path, base=base), "caption": caption})
    return out


def _model_summary_dict(model: "Model") -> Dict[str, Any]:
    vars_table = [
        {
            "symbol": v.symbol or v.name,
            "name": v.name,
            "role": v.role,
            "unit": v.unit,
            "dimension": str(v.dim),
            "description": v.description,
        }
        for v in model.all_variables()
    ]
    params_table = [
        {
            "symbol": p.symbol or p.name,
            "name": p.name,
            "status": p.status,
            "value": p.value,
            "unit": p.unit,
            "bounds": list(p.bounds) if p.bounds is not None else None,
            "description": p.description,
        }
        for p in model.parameters
    ]
    equations = [
        {
            "id": eq.id,
            "type": eq.type,
            "expression": eq.expression,
            "field": eq.field,
            "flux": eq.flux,
            "diffusivity": eq.diffusivity,
            "source": eq.source,
            "description": eq.description,
        }
        for eq in model.equations
    ]
    return {
        "name": model.name,
        "version": model.version,
        "domain": model.domain,
        "description": model.description,
        "assumptions": list(model.assumptions),
        "dependencies": list(model.dependencies),
        "equations": equations,
        "variables": vars_table,
        "parameters": params_table,
        "numerics": model.numerics.to_dict(),
    }


def _fmt_metric_val(val: Any) -> str:
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        return f"{val:.6g}"
    if isinstance(val, Mapping):
        inner = ", ".join(f"{k}={_fmt_metric_val(v)}" for k, v in list(val.items())[:4])
        return "{" + inner + ("..." if len(val) > 4 else "") + "}"
    if isinstance(val, list):
        if len(val) <= 4:
            return "[" + ", ".join(_fmt_metric_val(x) for x in val) + "]"
        return f"[{len(val)} items]"
    return str(val)


@dataclass
class ScientificReport(ResultMixin):
    """
    Reproducible scientific study report built from structured results and ledger data.
    """

    title: str
    report_type: str
    description: str
    status: VerificationStatus
    model_info: Dict[str, Any] = field(default_factory=dict)
    datasets: List[Dict[str, Any]] = field(default_factory=list)
    numerical_config: Dict[str, Any] = field(default_factory=dict)
    runs: List[Dict[str, Any]] = field(default_factory=list)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    figures: List[Dict[str, str]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    software: Dict[str, Any] = field(default_factory=dict)
    source_files: List[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.report_type not in REPORT_TYPES:
            raise ValueError(
                f"Unsupported report_type '{self.report_type}'. Expected one of {REPORT_TYPES}."
            )
        if not isinstance(self.status, VerificationStatus):
            self.status = _coerce_status(self.status)

    def to_markdown(self) -> str:
        """Render the complete scientific report as deterministic GitHub-Flavored Markdown."""
        lines: List[str] = [
            f"# {self.title}",
            "",
            f"- **Report Scope:** `{self.report_type}`",
            f"- **Overall Classification:** `{self.status.value.upper()}`",
            f"- **Generated (UTC):** {self.generated_at}",
        ]
        if self.model_info.get("name"):
            lines.append(
                f"- **Model:** `{self.model_info['name']}` (v{self.model_info.get('version', 'n/a')}, domain: `{self.model_info.get('domain', 'general')}`)"
            )
        lines.append("")

        if self.description:
            lines.extend([self.description, ""])

        # Explicit attention callout when status is not PASSED
        if self.status in (
            VerificationStatus.FAILED,
            VerificationStatus.INCONCLUSIVE,
            VerificationStatus.WARNING,
        ):
            lines.extend(
                [
                    f"> **ATTENTION ({self.status.value.upper()}):** "
                    "One or more verification, validation, calibration, or integrity checks did not pass cleanly. "
                    "Inspect the Findings and Warnings sections below before relying on downstream outputs.",
                    "",
                ]
            )

        # Model overview & governing equations
        lines.extend(["## 1. Model & Governing Equations", ""])
        if self.model_info:
            if self.model_info.get("description"):
                lines.extend([str(self.model_info["description"]), ""])
            eqs = self.model_info.get("equations") or []
            if eqs:
                lines.append("### Governing Equations")
                lines.append("")
                for eq in eqs:
                    eq_id = eq.get("id", "eq")
                    eq_type = eq.get("type", "equation")
                    expr = eq.get("expression") or (
                        f"d({eq.get('field')})/dt + d({eq.get('flux', 0)})/dx = "
                        f"d/dx({eq.get('diffusivity', 0)} d{eq.get('field')}/dx) + ({eq.get('source', 0)})"
                    )
                    lines.append(f"- **{eq_id}** (`{eq_type}`): `{expr}`")
                lines.append("")
            deps = self.model_info.get("dependencies") or []
            if deps:
                lines.append("### Qualitative Dependencies")
                lines.append("")
                for dep in deps:
                    lines.append(f"- `{dep}`")
                lines.append("")
            assumptions = self.model_info.get("assumptions") or []
            if assumptions:
                lines.append("### Scientific Assumptions")
                lines.append("")
                for asm in assumptions:
                    lines.append(f"- {asm}")
                lines.append("")
        else:
            lines.extend(["*No formal Model specification attached to this report.*", ""])

        # Variables, units & parameters
        vars_tbl = self.model_info.get("variables") or []
        params_tbl = self.model_info.get("parameters") or []
        if vars_tbl or params_tbl:
            lines.extend(["## 2. Variables, Physical Units & Parameters", ""])
            if vars_tbl:
                lines.extend(
                    [
                        "| Symbol | Role | Unit | Dimension | Description |",
                        "| --- | --- | --- | --- | --- |",
                    ]
                )
                for v in vars_tbl:
                    lines.append(
                        f"| `{v.get('symbol')}` | {v.get('role')} | `{v.get('unit')}` | `{v.get('dimension', '')}` | {v.get('description', '')} |"
                    )
                lines.append("")
            if params_tbl:
                lines.extend(
                    [
                        "| Parameter | Status | Value | Unit | Bounds | Description |",
                        "| --- | --- | --- | --- | --- | --- |",
                    ]
                )
                for p in params_tbl:
                    b = p.get("bounds")
                    b_str = f"[{b[0]}, {b[1]}]" if b is not None else "—"
                    val_str = "—" if p.get("value") is None else f"{p['value']}"
                    lines.append(
                        f"| `{p.get('symbol')}` | {p.get('status')} | {val_str} | `{p.get('unit')}` | {b_str} | {p.get('description', '')} |"
                    )
                lines.append("")

        # Datasets & provenance hashes
        lines.extend(["## 3. Datasets & Input Provenance", ""])
        if self.datasets:
            lines.extend(
                [
                    "| Dataset | Kind | Relative Path | SHA-256 |",
                    "| --- | --- | --- | --- |",
                ]
            )
            for ds in self.datasets:
                lines.append(
                    f"| `{ds.get('name', 'dataset')}` | {ds.get('kind', 'data')} | `{ds.get('path', '')}` | `{ds.get('sha256', 'n/a')}` |"
                )
            lines.append("")
        else:
            lines.extend(["*No external datasets recorded for this report.*", ""])

        # Numerical configuration
        lines.extend(["## 4. Numerical Configuration", ""])
        if self.numerical_config:
            lines.append("```yaml")
            lines.append(yaml.safe_dump(to_jsonable(self.numerical_config), sort_keys=False).rstrip())
            lines.append("```")
            lines.append("")
        else:
            lines.extend(["*Not applicable or defaults used.*", ""])

        # Results, runs & V&V findings
        lines.extend(["## 5. Results, Verification & Validation Findings", ""])
        if self.runs:
            lines.extend(
                [
                    "| Run ID | Type | Status | Timestamp (UTC) | Key Metrics | Artifact |",
                    "| --- | --- | --- | --- | --- | --- |",
                ]
            )
            for r in self.runs:
                metrics = r.get("metrics") or {}
                m_str = ", ".join(
                    f"{k}={_fmt_metric_val(v)}" for k, v in list(metrics.items())[:5]
                ) or "—"
                art = r.get("output_result") or "—"
                lines.append(
                    f"| `{r.get('run_id', 'n/a')}` | `{r.get('run_type', 'unknown')}` | **{str(r.get('status', 'not_assessed')).upper()}** | {r.get('timestamp', 'n/a')} | {m_str} | `{art}` |"
                )
            lines.append("")

        if self.findings:
            lines.append("### Detailed Assessment Breakdown")
            lines.append("")
            for f_item in self.findings:
                f_status = str(f_item.get("status", "not_assessed")).upper()
                f_label = f_item.get("label", "Finding")
                f_detail = f_item.get("detail", "")
                run_ref = f" (run `{f_item['run_id']}`)" if f_item.get("run_id") else ""
                lines.append(f"- **[{f_status}] {f_label}**{run_ref}: {f_detail}")
            lines.append("")
        elif not self.runs:
            lines.extend(["*No computational runs recorded.*", ""])

        # Tables & Figures
        if self.tables or self.figures:
            lines.extend(["## 6. Tables & Figures", ""])
            for tbl in self.tables:
                if tbl.get("title"):
                    lines.extend([f"### {tbl['title']}", ""])
                headers = tbl.get("headers") or []
                rows = tbl.get("rows") or []
                if headers:
                    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
                    lines.append("| " + " | ".join("---" for _ in headers) + " |")
                    for row in rows:
                        lines.append("| " + " | ".join(str(c) for c in row) + " |")
                    lines.append("")
            for fig in self.figures:
                lines.append(f"![{fig['caption']}]({fig['path']})")
                lines.append(f"*Figure: {fig['caption']} (`{fig['path']}`)*")
                lines.append("")

        # Warnings & Limitations
        lines.extend(["## 7. Warnings & Limitations", ""])
        if self.warnings:
            lines.append("### Warnings")
            lines.append("")
            for w in self.warnings:
                lines.append(f"- **Warning:** {w}")
            lines.append("")
        else:
            lines.extend(["*No runtime warnings recorded.*", ""])

        lines.append("### Limitations & Scope Notice")
        lines.append("")
        lims = list(self.limitations) if self.limitations else [_DEFAULT_NON_CERTIFICATION_NOTE]
        for lim in lims:
            lines.append(f"- {lim}")
        lines.append("")

        # Software Metadata & Traceability
        lines.extend(["## 8. Software Metadata & Traceability", ""])
        if self.software:
            lines.append("```yaml")
            lines.append(yaml.safe_dump(to_jsonable(self.software), sort_keys=False).rstrip())
            lines.append("```")
            lines.append("")
        if self.source_files:
            lines.append("### Referenced Artifacts")
            lines.append("")
            for sf in self.source_files:
                lines.append(f"- `{sf}`")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    def to_latex(self) -> str:
        """Render this report as a self-contained publication-ready LaTeX document."""
        from .latex import render_report_latex

        return render_report_latex(self)

    def _repr_markdown_(self) -> str:
        return self.to_markdown()

    def save(
        self,
        path: Union[str, Path],
        fmt: Optional[str] = None,
        *,
        also_manifest: Union[bool, str, Path] = False,
        latex_engine: str = "auto",
    ) -> Path:
        """Write report to `.md`, `.json`, `.yaml`, `.tex`, or `.pdf` without mutating simulation outputs."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        resolved_fmt = (fmt or out.suffix.lstrip(".") or "md").lower()
        if resolved_fmt in ("md", "markdown"):
            out.write_text(self.to_markdown(), encoding="utf-8")
        elif resolved_fmt == "json":
            out.write_text(self.to_json(), encoding="utf-8")
        elif resolved_fmt in ("yaml", "yml"):
            out.write_text(self.to_yaml(), encoding="utf-8")
        elif resolved_fmt in ("tex", "latex"):
            out.write_text(self.to_latex(), encoding="utf-8")
        elif resolved_fmt == "pdf":
            from .latex import compile_latex_to_pdf

            tex_path = out.with_suffix(".tex")
            tex_path.write_text(self.to_latex(), encoding="utf-8")
            compile_latex_to_pdf(tex_path, output_pdf=out, engine=latex_engine)
        else:
            raise ValueError(
                f"Unsupported report format '{resolved_fmt}'. Expected 'md', 'json', 'yaml', 'tex', or 'pdf'."
            )

        if also_manifest:
            if isinstance(also_manifest, bool):
                m_path = out.with_suffix(".manifest.json")
            else:
                m_path = Path(also_manifest)
            m_path.parent.mkdir(parents=True, exist_ok=True)
            m_path.write_text(self.to_json(), encoding="utf-8")
        return out


class ReportGenerator:
    """Factory building `ScientificReport` objects from structured results or `Project` records."""

    @classmethod
    def from_result(
        cls,
        result: Any,
        *,
        title: Optional[str] = None,
        report_type: Optional[str] = None,
        description: str = "",
        model: Optional["Model"] = None,
        datasets: Optional[Sequence[Mapping[str, Any]]] = None,
        figures: Optional[Sequence[Union[str, Path, Mapping[str, str]]]] = None,
        run_id: Optional[str] = None,
        base_dir: Optional[Path] = None,
    ) -> ScientificReport:
        from ..project import capture_environment

        sr = result if isinstance(result, StandardResult) else standardize_result(result)
        inferred_type = report_type or _KIND_TO_REPORT_TYPE.get(sr.kind, "numerical_simulation")
        if inferred_type not in REPORT_TYPES:
            raise ValueError(
                f"Unsupported report_type '{inferred_type}'. Expected one of {REPORT_TYPES}."
            )

        model_info: Dict[str, Any] = _model_summary_dict(model) if model is not None else {}
        if not model_info and (sr.metadata.model_name or sr.units):
            model_info = {
                "name": sr.metadata.model_name or "unspecified",
                "version": sr.metadata.model_version or "n/a",
                "domain": "general",
                "description": "",
                "assumptions": list(sr.assumptions),
                "dependencies": [],
                "equations": [],
                "variables": [
                    {
                        "symbol": sym,
                        "name": sym,
                        "role": "variable",
                        "unit": unit,
                        "dimension": "",
                        "description": "",
                    }
                    for sym, unit in sr.units.items()
                ],
                "parameters": [],
            }

        eff_run_id = run_id or sr.metadata.run_id or f"{inferred_type}_001"
        metrics = dict(sr.metadata.metrics)
        raw = sr.raw
        if not metrics and hasattr(raw, "to_dict"):
            raw_d = raw.to_dict()
            for k in (
                "rmse",
                "mae",
                "r2",
                "bic",
                "observed_order",
                "gci_fine_21",
                "gci_coarse_32",
                "max_normalized_error",
                "passed",
            ):
                if k in raw_d:
                    metrics[k] = raw_d[k]

        runs = [
            {
                "run_id": eff_run_id,
                "run_type": sr.kind,
                "status": sr.status.value,
                "timestamp": sr.metadata.timestamp,
                "metrics": to_jsonable(metrics),
                "output_result": "in-memory",
            }
        ]

        detail_summary = raw.summary().splitlines()[0] if hasattr(raw, "summary") else sr.kind
        findings = [
            {
                "label": f"{sr.kind} ({inferred_type})",
                "status": sr.status.value,
                "run_id": eff_run_id,
                "detail": detail_summary,
            }
        ]

        limitations = list(getattr(raw, "assumptions", []) or sr.assumptions)
        if _DEFAULT_NON_CERTIFICATION_NOTE not in limitations:
            limitations.append(_DEFAULT_NON_CERTIFICATION_NOTE)

        ds_entries: List[Dict[str, Any]] = []
        for ds in (datasets or []):
            ds_entries.append(
                {
                    "name": str(ds.get("name", "dataset")),
                    "kind": str(ds.get("kind", "data")),
                    "path": _safe_relpath(str(ds.get("path", "")), base=base_dir)
                    if ds.get("path")
                    else "",
                    "sha256": str(ds.get("sha256", "n/a")),
                }
            )

        return ScientificReport(
            title=title or f"Rosolver Scientific Report: {model_info.get('name', sr.kind)}",
            report_type=inferred_type,
            description=description,
            status=sr.status,
            model_info=model_info,
            datasets=ds_entries,
            numerical_config=dict(sr.solver_config),
            runs=runs,
            findings=findings,
            figures=_normalize_figures(figures, base=base_dir),
            tables=[],
            warnings=list(sr.warnings),
            limitations=limitations,
            software=capture_environment(),
            source_files=[],
            metadata={"source_kind": sr.kind, "timing_s": sr.metadata.timing_s},
        )

    @classmethod
    def from_project(
        cls,
        project: "Project",
        *,
        title: Optional[str] = None,
        report_type: str = "integrated_project",
        description: str = "",
        run_ids: Optional[Sequence[str]] = None,
        figures: Optional[Sequence[Union[str, Path, Mapping[str, str]]]] = None,
    ) -> ScientificReport:
        from ..project import capture_environment

        if report_type not in REPORT_TYPES:
            raise ValueError(
                f"Unsupported report_type '{report_type}'. Expected one of {REPORT_TYPES}."
            )

        integrity = project.verify_integrity(strict=False)
        warnings: List[str] = []
        statuses: List[VerificationStatus] = []

        if not integrity["ok"]:
            statuses.append(VerificationStatus.FAILED)
            if integrity["missing"]:
                warnings.append(f"Project missing artifacts: {integrity['missing']}")
            if integrity["modified"]:
                warnings.append(f"Project modified artifacts (hash mismatch): {integrity['modified']}")

        # Load primary/latest model without raising on hash mismatch (already flagged above)
        model_info: Dict[str, Any] = {}
        source_files: List[str] = ["project.yaml"]
        if project.models:
            m_ref = project.models[-1]
            source_files.append(m_ref.path)
            try:
                model_obj = project.get_model(
                    name=m_ref.name, version=m_ref.version, verify_hash=False
                )
                model_info = _model_summary_dict(model_obj)
                model_info["file_sha256"] = m_ref.file_sha256
                model_info["content_hash"] = m_ref.content_hash
            except Exception as exc:
                warnings.append(f"Could not load model '{m_ref.name}': {exc}")
                statuses.append(VerificationStatus.FAILED)

        ds_entries = [
            {
                "name": d.name,
                "kind": d.category,
                "path": d.path,
                "sha256": d.sha256,
            }
            for d in project.datasets
        ]

        all_runs = project.ledger()
        if run_ids is not None:
            wanted = set(run_ids)
            all_runs = [r for r in all_runs if r.id in wanted]

        runs_summary: List[Dict[str, Any]] = []
        findings: List[Dict[str, Any]] = []
        numerical_config: Dict[str, Any] = dict(model_info.get("numerics") or {})
        software: Dict[str, Any] = {}

        if integrity["ok"]:
            findings.append(
                {
                    "label": "Project Provenance Integrity",
                    "status": VerificationStatus.PASSED.value,
                    "detail": f"Verified {len(project.models)} model(s) and {len(project.datasets)} dataset(s) via SHA-256.",
                }
            )
        else:
            findings.append(
                {
                    "label": "Project Provenance Integrity",
                    "status": VerificationStatus.FAILED.value,
                    "detail": (
                        f"Missing={integrity['missing']}, Modified={integrity['modified']}"
                    ),
                }
            )

        for rec in all_runs:
            rec_status = _coerce_status(rec.status)
            statuses.append(rec_status)
            if not software and rec.software:
                software = dict(rec.software)
            if rec.configuration:
                numerical_config.setdefault(rec.id, rec.configuration)

            res_path = rec.outputs.get("result", "")
            metrics: Dict[str, Any] = {}
            detail = f"Run type `{rec.type}` recorded at {rec.timestamp}."

            if res_path:
                full_res = (project.root / res_path).resolve()
                if full_res.exists():
                    source_files.append(res_path)
                    try:
                        payload = json.loads(full_res.read_text(encoding="utf-8"))
                        meta_blk = payload.get("metadata") or {}
                        metrics = dict(meta_blk.get("metrics") or {})
                        for rw in (meta_blk.get("warnings") or []):
                            if rw not in warnings:
                                warnings.append(f"[{rec.id}] {rw}")
                        raw_blk = payload.get("result") or {}
                        if not metrics and isinstance(raw_blk, Mapping):
                            for k in (
                                "observed_order",
                                "gci_fine_21",
                                "rmse",
                                "r2",
                                "bic",
                                "max_normalized_error",
                                "passed",
                            ):
                                if k in raw_blk:
                                    metrics[k] = raw_blk[k]
                        if isinstance(raw_blk, Mapping) and raw_blk.get("regime"):
                            detail = f"{rec.type}: regime={raw_blk['regime']}, passed={raw_blk.get('passed')}"
                    except Exception as exc:
                        warnings.append(f"[{rec.id}] Failed to parse stored result '{res_path}': {exc}")
                else:
                    warnings.append(f"[{rec.id}] Referenced result artifact missing: '{res_path}'.")
                    statuses.append(VerificationStatus.FAILED)

            runs_summary.append(
                {
                    "run_id": rec.id,
                    "run_type": rec.type,
                    "status": rec_status.value,
                    "timestamp": rec.timestamp,
                    "metrics": to_jsonable(metrics),
                    "output_result": res_path or "—",
                }
            )
            findings.append(
                {
                    "label": f"{rec.type} (`{rec.id}`)",
                    "status": rec_status.value,
                    "run_id": rec.id,
                    "detail": detail,
                }
            )

        if not software:
            software = capture_environment()

        limitations = list(model_info.get("assumptions") or [])
        if _DEFAULT_NON_CERTIFICATION_NOTE not in limitations:
            limitations.append(_DEFAULT_NON_CERTIFICATION_NOTE)

        overall_status = _worst_status(statuses)
        eff_title = title or f"{project.name} — Scientific Study Report"
        eff_desc = description or project.description

        # Preserve deterministic ordering of unique source files
        seen_sf: set[str] = set()
        unique_sf: List[str] = []
        for sf in source_files:
            if sf not in seen_sf:
                seen_sf.add(sf)
                unique_sf.append(sf)

        return ScientificReport(
            title=eff_title,
            report_type=report_type,
            description=eff_desc,
            status=overall_status,
            model_info=model_info,
            datasets=ds_entries,
            numerical_config=numerical_config,
            runs=runs_summary,
            findings=findings,
            figures=_normalize_figures(figures, base=project.root),
            tables=[],
            warnings=warnings,
            limitations=limitations,
            software=software,
            source_files=unique_sf,
            metadata={"project_id": project.project_id, "project_name": project.name},
        )

    @classmethod
    def from_claim_round(
        cls,
        project: "Project",
        round_record: Any,
        *,
        title: Optional[str] = None,
        report_type: str = "peer_review_letter",
        description: str = "",
    ) -> ScientificReport:
        """Build a structured Referee Report / Claim Verification `ScientificReport` from a `ReviewRoundRecord`."""
        from ..project import capture_environment

        if report_type not in ("claim_verification", "peer_review_letter"):
            raise ValueError("report_type must be 'claim_verification' or 'peer_review_letter'.")

        statuses: List[VerificationStatus] = [v.status for v in round_record.verdicts]
        overall_status = _worst_status(statuses)

        # Table 1: Executive Audit Scorecard
        scorecard_rows = [
            [
                v.claim_id,
                v.title,
                v.kind.value,
                v.assessment.mechanistic_status.value,
                v.assessment.empirical_status.value,
                v.overall_verdict.value,
                ", ".join(v.evidence_run_ids) or "—",
            ]
            for v in round_record.verdicts
        ]
        tables: List[Dict[str, Any]] = [
            {
                "title": f"Executive Claim Audit Scorecard ({round_record.round_id})",
                "headers": [
                    "Claim ID",
                    "Title",
                    "Kind",
                    "Mechanistic",
                    "Empirical",
                    "Verdict",
                    "Evidence Run IDs",
                ],
                "rows": scorecard_rows,
            }
        ]

        # Table 2: Open Clarification Requests (Referee Queries)
        if round_record.open_requests:
            req_rows = [
                [
                    r.request_id,
                    r.claim_id,
                    r.severity,
                    r.trigger_code,
                    r.required_artifact.kind,
                    r.reviewer_prompt,
                ]
                for r in round_record.open_requests
            ]
            tables.append(
                {
                    "title": f"Referee Clarification & Data Requests ({round_record.round_id})",
                    "headers": ["Request ID", "Claim", "Severity", "Trigger", "Required Artifact", "Reviewer Query"],
                    "rows": req_rows,
                }
            )

        # Table 3: Multi-Round Verdict Transitions (R_k -> R_{k+1})
        if round_record.transitions:
            tr_rows = [
                [
                    t.claim_id,
                    t.prev_verdict,
                    t.new_verdict,
                    ", ".join(t.resolved_request_ids) or "—",
                    ", ".join(t.remaining_request_ids) or "—",
                    "Yes" if t.post_hoc_recalibration else "No",
                ]
                for t in round_record.transitions
            ]
            tables.append(
                {
                    "title": f"Rebuttal Transitions ({round_record.parent_round_id} -> {round_record.round_id})",
                    "headers": [
                        "Claim ID",
                        "Previous",
                        "Updated",
                        "Resolved Requests",
                        "Open Requests",
                        "Post-Hoc Tuning Flag",
                    ],
                    "rows": tr_rows,
                }
            )

        findings = [
            {
                "label": f"{v.claim_id} ({v.overall_verdict.value})",
                "status": v.status.value,
                "run_id": v.evidence_run_ids[0] if v.evidence_run_ids else None,
                "detail": v.summary_text,
            }
            for v in round_record.verdicts
        ]
        warnings = [w for v in round_record.verdicts for w in v.warnings]
        runs_summary = [
            {
                "run_id": rid,
                "run_type": v.kind.value,
                "status": v.status.value,
                "timestamp": round_record.timestamp,
                "metrics": v.metrics,
                "output_result": f"claims/rounds/round_{round_record.round_index:03d}.json",
            }
            for v in round_record.verdicts
            for rid in (v.evidence_run_ids or [f"{v.claim_id}_unrun"])
        ]

        eff_title = title or f"{project.name} — Peer-Review Claim Audit ({round_record.round_id})"
        eff_desc = description or (
            f"Deterministic scientific claim verification round {round_record.round_id} "
            f"(manifest hash: {round_record.manifest_hash[:12]})."
        )
        limitations = [
            "Audits mathematical, dimensional, scale-separation, numerical, and statistical consistency against supplied evidence; does not certify wet-lab specimen authenticity.",
            _DEFAULT_NON_CERTIFICATION_NOTE,
        ]
        return ScientificReport(
            title=eff_title,
            report_type=report_type,
            description=eff_desc,
            status=overall_status,
            model_info={},
            datasets=[d.to_dict() for d in project.datasets],
            numerical_config={"round_id": round_record.round_id, "manifest_hash": round_record.manifest_hash},
            runs=runs_summary,
            findings=findings,
            figures=[],
            tables=tables,
            warnings=warnings,
            limitations=limitations,
            software=capture_environment(),
            source_files=[
                "claims/manifest.yaml",
                f"claims/rounds/round_{round_record.round_index:03d}.json",
                "claims/clarifications_open.json",
            ],
            metadata={
                "project_id": project.project_id,
                "project_name": project.name,
                "round_id": round_record.round_id,
                "summary_counts": round_record.summary_counts,
            },
        )
