# External CFD/FEM Audit Adapters & Headless LaTeX/PDF Reporting (`Sprint 8 Slim`)

This guide documents Rosolver's native post-processing audit adapters (`OpenFOAMAdapter`, `FEniCSxAdapter`, `HDF5Adapter`) and headless LaTeX/PDF report renderer (`LatexReportRenderer`, `compile_latex_to_pdf`).

---

## 1. Purpose & Scope

External CFD and FEM solvers (OpenFOAM, FEniCSx/DOLFINx, custom HDF5 codes) generate field profiles, residual histories, and multi-grid convergence datasets outside Rosolver. Sprint 8 Slim provides zero-bloat adapters that normalize those exports into Rosolver's `Dataset` and `VerificationStudy` pipelines—without altering any V&V algorithm or requiring OpenFOAM, FEniCSx, or VTK to be installed—and renders any `ScientificReport` (including Sprint 10 `peer_review_letter` reports) into publication-ready `.tex` and `.pdf` artifacts headlessly.

---

## 2. Supported Formats & Assumptions

| Component | Supported Inputs / Outputs | Optional Dependencies | Absent-Metadata Policy |
| --- | --- | --- | --- |
| `OpenFOAMAdapter` | `postProcessing/` sample sets (`.xy`, `.csv`, `.dat` incl. `(Ux Uy Uz)` vectors), `residuals.dat`, case dir (`system/controlDict`) | None | Unavailable fields recorded as `None` |
| `FEniCSxAdapter` | ParaView/VTK-CSV (`Points:0..2`, `u:0..2`), legacy ASCII `.vtk`, XML ASCII `.vtu`, inline XML `.xdmf` | None (`h5py` only for `Format="HDF"` XDMF) | Unavailable fields recorded as `None` |
| `HDF5Adapter` / `Dataset.to_hdf5` | Flat or structured (`coordinates/`, `fields/`, `uncertainties/` + `.attrs`) `.h5` / `.hdf5` | `rosolver[io]` (`h5py`) | Clean `ImportError` if `h5py` missing |
| `LatexReportRenderer` / `compile_latex_to_pdf` | Any `ScientificReport` (`10` scopes) $\to$ `.tex` and `.pdf` | TeX engine (`pdflatex`, `xelatex`, or `tectonic`) for `.pdf` only | `.tex` always works with zero external deps |

---

## 3. Python API Examples

### 3.1 OpenFOAM Post-Processing & Multi-Grid GCI
```python
from rosolver import OpenFOAMAdapter, VerificationStudy

# 1. Inspect residuals convergence from an OpenFOAM case
foam_res = OpenFOAMAdapter(function_object="residuals", residual_tol=1e-4)
res_summary = foam_res.load_residuals("path/to/case")

# 2. Multi-grid Roache/Celik GCI across coarse/medium/fine OpenFOAM cases
adapter = OpenFOAMAdapter(function_object="lineProbe", sample_file="line_T.xy")
study = VerificationStudy.from_multi_grid(
    adapter,
    ["case_coarse", "case_medium", "case_fine"],
    spacings=[0.04, 0.02, 0.01],
    field="T",
    reducer=lambda arr: float(arr[-1]),
)
gci = study.grid_convergence(formal_order=2.0)
```

### 3.2 FEniCSx / VTK-CSV / VTU Verification
```python
from rosolver import VerificationStudy

study = VerificationStudy.from_fenicsx(
    "outputs/poisson_fine.vtu",
    field="u",
    element_family="Lagrange",
    element_degree=2,
)
```

### 3.3 Headless LaTeX & PDF Referee Reports
```python
from rosolver import Project, ReportGenerator

proj = Project.open("my_audit_project")
round_rec = proj.audit_claims()
rep = ReportGenerator.from_claim_round(proj, round_rec)

# Export .tex (zero external dependencies) or compile headless .pdf
rep.save("reports/referee_r1.tex")
rep.save("reports/referee_r1.pdf")  # uses pdflatex / xelatex / tectonic
```

---

## 4. Workflow CLI & MCP Usage

```bash
# Generate LaTeX or compiled PDF directly from CLI
rosolver report my_project --output reports/final_report.tex
rosolver report my_project --output reports/final_report.pdf

# Export Sprint 10 peer-review referee letter directly as LaTeX/PDF
rosolver claim audit my_project --output reports/referee_r1.tex
```

Over `rosolver-mcp`, passing `output: "reports/final_report.tex"` (or `.pdf`) to `rosolver_report` writes the file and includes the rendered `"latex"` payload alongside `"markdown"`.

---

## 5. Limitations & Security Guarantees

- **No binary VTK/OpenFOAM parser**: Legacy `.vtk` and `.vtu` readers require ASCII encoding; binary/compressed VTU files should be exported as ASCII VTU, VTK-CSV, or HDF5.
- **No shell execution**: `compile_latex_to_pdf` invokes the TeX compiler with `subprocess.run(..., shell=False)` and `-interaction=nonstopmode -halt-on-error`.
- **Zero absolute user paths**: Generated `.tex` files reference figures strictly via relative paths and escape untrusted strings cleanly.
