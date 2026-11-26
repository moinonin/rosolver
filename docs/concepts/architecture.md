# Architecture & Reproducibility Model

Rosolver decouples scientific model definitions from numerical backends, execution workspaces, and scientific reports.

## Modular Package Layout (`P4.1`)

All public symbols remain exported at the top level (`import rosolver`) and are organized across domain-focused subpackages:

| Subpackage | Responsibility |
| --- | --- |
| `rosolver.core` | `Dimension`, `System`, validated `Model` spec, `Project`, `StandardResult`, `VerificationStatus` |
| `rosolver.symbolic` | Chain-rule DAG formulation, Buckingham-$\Pi$ closure, Clairaut exactness, `TEMPLATE_REGISTRY` |
| `rosolver.calibration` | `NumericalSolver` OLS/WLS/NLLS calibration, Jacobian identifiability diagnostics, BIC ranking, `Dataset` |
| `rosolver.numerics` | `ODESystem`, `ConservationLaw1D`, `Stefan1D`, fixed-step integrators |
| `rosolver.verification` | `mms_study`, `gci_from_values`, `grid_convergence_study`, `timestep_study`, conservation audits, stiffness & Aitken coupling |
| `rosolver.validation` | `ValidationStudy` comparing simulation + $u_{\text{num}}$ vs. experimental datasets + $u_{\text{exp}}$ |
| `rosolver.integrations` | `SolverAdapter` ABC, `GenericCSVAdapter`, `SciPyAdapter`, `VerificationStudy` |
| `rosolver.io` | Safe YAML/JSON `Model` persistence, `Dataset` CSV/JSON/YAML/NPZ/HDF5 I/O, SHA-256 project manifests |
| `rosolver.reporting` | `ScientificReport` (Markdown/JSON/YAML), `ReportGenerator`, publication-ready V&V plotting helpers |

## Result Standardization & 5-State Verification

Every subsystem wraps or classifies outcomes with `VerificationStatus`:
- `Passed`: criteria satisfied within tolerance
- `Failed`: numerical convergence failure, hash tamper, order shortfall, or residual above tolerance
- `Inconclusive`: oscillatory or indeterminate Celik/GCI order extraction
- `Warning`: partial uncertainty coverage (e.g., missing `u_num`), TVD limiter order relaxation, or conditioning warning
- `NotAssessed`: check not yet executed

## Provenance & Security Guardrails

- Equation strings in `Model` and `System` are parsed with the AST-validated safe parser in [rosolver/_parsing.py](../../rosolver/_parsing.py) (never raw `sympify()` or `eval()`).
- YAML specs load exclusively via `yaml.safe_load`; `.npz` loads enforce `allow_pickle=False`.
- `Project` hashes models and datasets with SHA-256, records software/Python/dependency versions, and writes an append-only JSONL execution ledger under `logs/ledger.jsonl`.
