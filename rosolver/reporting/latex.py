"""Headless LaTeX & PDF scientific report renderer (ENG_SPEC.md P3.1+ / Sprint 8 Slim).

Converts any structured ``ScientificReport`` (including Sprint 10 ``peer_review_letter``
and ``claim_verification`` reports) into self-contained publication-ready LaTeX (``.tex``)
and optionally compiles to PDF headlessly via ``pdflatex``, ``xelatex``, or ``tectonic``
without shell injection or hardcoded absolute user paths.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Union

import yaml

from ..results import VerificationStatus, to_jsonable

if TYPE_CHECKING:
    from .markdown import ScientificReport


_LATEX_SPECIALS = (
    ("\\", r"\textbackslash{}"),
    ("&", r"\&"),
    ("%", r"\%"),
    ("$", r"\$"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("{", r"\{"),
    ("}", r"\}"),
    ("~", r"\textasciitilde{}"),
    ("^", r"\textasciicircum{}"),
)


def escape_latex(text: Any) -> str:
    """Escape untrusted plain text safely for LaTeX body/table cells."""
    s = str(text)
    for src, dst in _LATEX_SPECIALS:
        s = s.replace(src, dst)
    return s


def _status_badge(status: Union[VerificationStatus, str]) -> str:
    val = status.value if isinstance(status, VerificationStatus) else str(status)
    low = val.strip().lower()
    label = escape_latex(val.upper())
    if low in ("passed", "supported"):
        return rf"\textcolor{{roGreen}}{{\textbf{{{label}}}}}"
    if low in ("failed", "falsified"):
        return rf"\textcolor{{roRed}}{{\textbf{{{label}}}}}"
    if low in ("warning", "supportedwithcaveats", "partiallysupported"):
        return rf"\textcolor{{roAmber}}{{\textbf{{{label}}}}}"
    if low in ("inconclusive", "empiricallyconsistentmechanisticallyunsupported"):
        return rf"\textcolor{{roBlue}}{{\textbf{{{label}}}}}"
    return rf"\textcolor{{roGray}}{{\textbf{{{label}}}}}"


def _fmt_metric_val(val: Any) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        if abs(val) >= 1e4 or (0.0 < abs(val) < 1e-3):
            return f"{val:.3e}"
        return f"{val:.4g}"
    return str(val)


def _table_col_spec(headers: Sequence[str]) -> str:
    """Build a booktabs/tabularx-safe column specification fitting within ``\\linewidth``."""
    n = len(headers)
    if n <= 2:
        return "l X" if n == 2 else "X"
    if n <= 4:
        return "l " * (n - 1) + "X"
    # Wide tables (5–7 cols): allocate compact left cols + wrapping last column
    specs = ["l"]
    for i in range(1, n - 1):
        hdr = str(headers[i]).lower()
        if any(k in hdr for k in ("title", "query", "detail", "metric", "artifact", "prompt")):
            specs.append("X")
        else:
            specs.append("l")
    specs.append("X")
    return " ".join(specs)


def _render_latex_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    title: Optional[str] = None,
) -> List[str]:
    if not headers:
        return []
    lines: List[str] = []
    if title:
        lines.append(rf"\subsection*{{{escape_latex(title)}}}")
    col_spec = _table_col_spec(headers)
    lines.append(r"\begin{center}")
    lines.append(r"\small")
    lines.append(rf"\begin{{tabularx}}{{\linewidth}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(rf"\textbf{{{escape_latex(h)}}}" for h in headers) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        cells = []
        for idx, cell in enumerate(row):
            if idx >= len(headers):
                break
            hdr_l = str(headers[idx]).lower()
            if hdr_l in ("status", "verdict", "mechanistic", "empirical", "previous", "updated"):
                cells.append(_status_badge(str(cell)))
            else:
                cells.append(escape_latex(cell))
        while len(cells) < len(headers):
            cells.append("")
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append(r"\end{center}")
    lines.append("")
    return lines


class LatexReportRenderer:
    """Deterministic LaTeX renderer for ``ScientificReport`` objects."""

    def __init__(
        self,
        *,
        document_class: str = "article",
        font_size: str = "10pt",
        margin: str = "0.85in",
    ) -> None:
        self.document_class = document_class
        self.font_size = font_size
        self.margin = margin

    def render(self, report: "ScientificReport") -> str:
        """Render ``report`` into a self-contained LaTeX document string."""
        is_referee = report.report_type in ("peer_review_letter", "claim_verification")
        banner_label = (
            "Peer-Review Scientific Claim Audit"
            if is_referee
            else f"Rosolver Scientific Report ({report.report_type.replace('_', ' ').title()})"
        )

        lines: List[str] = [
            rf"\documentclass[{self.font_size}]{{{self.document_class}}}",
            rf"\usepackage[margin={self.margin}]{{geometry}}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage{lmodern}",
            r"\usepackage{microtype}",
            r"\usepackage{booktabs}",
            r"\usepackage{tabularx}",
            r"\usepackage{xcolor}",
            r"\usepackage{graphicx}",
            r"\usepackage{enumitem}",
            r"\usepackage{fancyvrb}",
            r"\usepackage[hidelinks]{hyperref}",
            r"\definecolor{roGreen}{HTML}{1B7F3A}",
            r"\definecolor{roRed}{HTML}{B91C1C}",
            r"\definecolor{roAmber}{HTML}{B45309}",
            r"\definecolor{roBlue}{HTML}{1D4ED8}",
            r"\definecolor{roGray}{HTML}{4B5563}",
            r"\definecolor{roRule}{HTML}{D1D5DB}",
            r"\setlength{\parindent}{0pt}",
            r"\setlength{\parskip}{5pt}",
            r"\begin{document}",
            "",
            # Header block
            rf"{{\small\textsc{{\textcolor{{roGray}}{{{escape_latex(banner_label)}}}}}}}\par",
            rf"{{\LARGE\bfseries {escape_latex(report.title)}}}\par",
            r"\vspace{4pt}",
            rf"\noindent\textbf{{Overall Status:}} {_status_badge(report.status)} \quad|\quad "
            rf"\textbf{{Report Scope:}} \texttt{{{escape_latex(report.report_type)}}} \quad|\quad "
            rf"\textbf{{Generated (UTC):}} {escape_latex(report.generated_at)}\par",
            r"\vspace{2pt}\noindent\textcolor{roRule}{\rule{\linewidth}{0.6pt}}\vspace{6pt}",
            "",
        ]

        # Section 1: Executive Summary
        lines.append(r"\section{Executive Summary}")
        if report.description:
            lines.append(escape_latex(report.description))
            lines.append("")

        flagged = [
            f
            for f in report.findings
            if str(f.get("status", "")).lower() in ("failed", "inconclusive", "warning")
        ]
        if flagged:
            lines.append(r"\textbf{Attention --- Flagged Findings:}")
            lines.append(r"\begin{itemize}[leftmargin=1.4em,itemsep=2pt]")
            for item in flagged:
                st = _status_badge(str(item.get("status", "unknown")))
                lbl = escape_latex(item.get("label", "Check"))
                det = escape_latex(item.get("detail", ""))
                lines.append(rf"\item [{st}] \textbf{{{lbl}}}: {det}")
            lines.append(r"\end{itemize}")
            lines.append("")

        # Section 2: Model & System Specification
        if report.model_info:
            lines.append(r"\section{Model \& System Specification}")
            m = report.model_info
            lines.append(
                rf"\textbf{{Model:}} {escape_latex(m.get('name', 'unnamed'))} "
                rf"(v{escape_latex(m.get('version', '1.0'))}, kind: \texttt{{{escape_latex(m.get('kind', 'unknown'))}}})"
            )
            if m.get("description"):
                lines.append("")
                lines.append(escape_latex(m["description"]))
            if m.get("equations"):
                lines.append(r"\begin{itemize}[leftmargin=1.4em,itemsep=2pt]")
                for eq in m["equations"]:
                    if isinstance(eq, Mapping):
                        lhs = eq.get("lhs", "")
                        rhs = eq.get("rhs", eq.get("expr", ""))
                        lines.append(rf"\item \texttt{{{escape_latex(lhs)} = {escape_latex(rhs)}}}")
                    else:
                        lines.append(rf"\item \texttt{{{escape_latex(eq)}}}")
                lines.append(r"\end{itemize}")
            lines.append("")

        # Section 3: Datasets & Input Provenance
        if report.datasets:
            lines.append(r"\section{Datasets \& Input Provenance}")
            ds_rows = [
                [
                    ds.get("name", "dataset"),
                    ds.get("kind", "data"),
                    ds.get("path", ""),
                    str(ds.get("sha256", "n/a"))[:16],
                ]
                for ds in report.datasets
            ]
            lines.extend(_render_latex_table(["Dataset", "Kind", "Path", "SHA-256 (prefix)"], ds_rows))

        # Section 4: Structured Scorecards, Claims & Transitions
        if report.tables:
            sec_title = (
                "Claim Audit Scorecards, Clarifications \\& Transitions"
                if is_referee
                else "Structured Verification \\& Data Tables"
            )
            lines.append(rf"\section{{{sec_title}}}")
            for tbl in report.tables:
                lines.extend(
                    _render_latex_table(
                        tbl.get("headers") or [],
                        tbl.get("rows") or [],
                        title=tbl.get("title"),
                    )
                )

        # Section 5: Runs & Detailed Findings
        lines.append(r"\section{Computational Runs \& Detailed Findings}")
        if report.runs:
            run_rows = []
            for r in report.runs:
                metrics = r.get("metrics") or {}
                m_str = ", ".join(
                    f"{k}={_fmt_metric_val(v)}" for k, v in list(metrics.items())[:4]
                ) or "-"
                run_rows.append(
                    [
                        r.get("run_id", "n/a"),
                        r.get("run_type", "unknown"),
                        str(r.get("status", "not_assessed")),
                        m_str,
                    ]
                )
            lines.extend(_render_latex_table(["Run ID", "Type", "Status", "Key Metrics"], run_rows))

        if report.findings:
            lines.append(r"\subsection*{Assessment Breakdown}")
            lines.append(r"\begin{itemize}[leftmargin=1.4em,itemsep=2pt]")
            for f_item in report.findings:
                st = _status_badge(str(f_item.get("status", "not_assessed")))
                lbl = escape_latex(f_item.get("label", "Finding"))
                det = escape_latex(f_item.get("detail", ""))
                ref = f" (run \\texttt{{{escape_latex(f_item['run_id'])}}})" if f_item.get("run_id") else ""
                lines.append(rf"\item [{st}] \textbf{{{lbl}}}{ref}: {det}")
            lines.append(r"\end{itemize}")
            lines.append("")

        # Section 6: Figures (strictly relative paths; never absolute user paths)
        if report.figures:
            lines.append(r"\section{Figures}")
            for fig in report.figures:
                fig_path = str(fig.get("path", ""))
                caption = escape_latex(fig.get("caption", fig_path))
                lines.append(r"\begin{figure}[htbp]")
                lines.append(r"\centering")
                lines.append(rf"\includegraphics[width=0.9\linewidth]{{{fig_path}}}")
                lines.append(rf"\caption{{{caption}}}")
                lines.append(r"\end{figure}")
                lines.append("")

        # Section 7: Warnings & Limitations
        lines.append(r"\section{Warnings, Limitations \& Scope Notice}")
        if report.warnings:
            lines.append(r"\subsection*{Runtime Warnings}")
            lines.append(r"\begin{itemize}[leftmargin=1.4em,itemsep=2pt]")
            for w in report.warnings:
                lines.append(rf"\item {escape_latex(w)}")
            lines.append(r"\end{itemize}")
        if report.limitations:
            lines.append(r"\subsection*{Limitations \& Non-Certification Notice}")
            lines.append(r"\begin{itemize}[leftmargin=1.4em,itemsep=2pt]")
            for lim in report.limitations:
                lines.append(rf"\item {escape_latex(lim)}")
            lines.append(r"\end{itemize}")
        lines.append("")

        # Section 8: Traceability & Software Environment
        lines.append(r"\section{Software Metadata \& Provenance}")
        if report.software:
            sw_yaml = yaml.safe_dump(to_jsonable(report.software), sort_keys=False).rstrip()
            lines.append(r"\begin{Verbatim}[fontsize=\small]")
            lines.append(sw_yaml)
            lines.append(r"\end{Verbatim}")
        if report.source_files:
            lines.append(r"\subsection*{Referenced Artifacts}")
            lines.append(r"\begin{itemize}[leftmargin=1.4em,itemsep=1pt]")
            for sf in report.source_files:
                lines.append(rf"\item \texttt{{{escape_latex(sf)}}}")
            lines.append(r"\end{itemize}")

        lines.append(r"\end{document}")
        return "\n".join(lines) + "\n"


def render_report_latex(report: "ScientificReport") -> str:
    """Convenience function rendering a ``ScientificReport`` to self-contained LaTeX."""
    return LatexReportRenderer().render(report)


def _detect_latex_engine(engine: str = "auto") -> str:
    if engine != "auto":
        found = shutil.which(engine)
        if not found:
            raise RuntimeError(
                f"Requested LaTeX engine '{engine}' was not found on PATH. "
                "Install TeX Live / MacTeX / Tectonic or export '.tex' instead."
            )
        return found
    for cand in ("pdflatex", "xelatex", "tectonic"):
        found = shutil.which(cand)
        if found:
            return found
    raise RuntimeError(
        "No headless LaTeX compiler ('pdflatex', 'xelatex', or 'tectonic') found on PATH. "
        "Export to '.tex' or install a TeX distribution to compile PDF reports."
    )


def compile_latex_to_pdf(
    tex_path: Union[str, Path],
    *,
    output_pdf: Optional[Union[str, Path]] = None,
    engine: str = "auto",
    passes: int = 2,
    timeout_s: int = 60,
    clean_aux: bool = True,
) -> Path:
    """Compile a ``.tex`` file to ``.pdf`` headlessly without shell execution.

    Args:
        tex_path: Path to the ``.tex`` source file.
        output_pdf: Optional destination path for the compiled ``.pdf``.
        engine: ``"auto"`` (default), ``"pdflatex"``, ``"xelatex"``, or ``"tectonic"``.
        passes: Number of LaTeX compilation passes (ignored for ``tectonic``).
        timeout_s: Subprocess timeout in seconds per pass.
        clean_aux: Remove ``.aux``, ``.log``, ``.out`` auxiliary files after success.
    """
    src = Path(tex_path).resolve()
    if not src.exists():
        raise FileNotFoundError(f"LaTeX source file not found: '{src.name}'.")

    compiler = _detect_latex_engine(engine)
    compiler_name = Path(compiler).name.lower()
    work_dir = src.parent

    if "tectonic" in compiler_name:
        cmd = [compiler, "--outdir", str(work_dir), src.name]
        run_passes = 1
    else:
        cmd = [
            compiler,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            src.name,
        ]
        run_passes = max(1, int(passes))

    last_out = ""
    for _ in range(run_passes):
        proc = subprocess.run(
            cmd,
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            shell=False,
            check=False,
        )
        last_out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0:
            tail = "\n".join(last_out.splitlines()[-25:])
            raise RuntimeError(
                f"Headless LaTeX compilation failed ({compiler_name}, exit={proc.returncode}):\n{tail}"
            )

    built_pdf = work_dir / f"{src.stem}.pdf"
    if not built_pdf.exists():
        raise RuntimeError(f"LaTeX compiler '{compiler_name}' exited 0 but did not produce '{built_pdf.name}'.")

    if clean_aux:
        for ext in (".aux", ".log", ".out", ".toc"):
            aux = work_dir / f"{src.stem}{ext}"
            if aux.exists():
                try:
                    aux.unlink()
                except OSError:
                    pass

    if output_pdf is not None:
        dest = Path(output_pdf).resolve()
        if dest != built_pdf:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(built_pdf), str(dest))
            return dest
    return built_pdf
