# PySR & External Symbolic-Regression Candidate Importer (`Sprint 9 Slim`)

This guide documents Rosolver's deterministic symbolic-regression gatekeeper (`CandidateImporter`), which imports Pareto-front equations from PySR (`hall_of_fame.csv` or `PySRRegressor`) or external symbolic discovery tools and screens them strictly through Rosolver's security, dimensional, scale-separation, and statistical identifiability gates.

---

## 1. Purpose & Architecture

External symbolic regression tools (PySR, Operon, AI-Feynman, or AI-proposed expressions) optimize numerical loss without enforcing physical units, operating-envelope scale balance, or parameter identifiability. `CandidateImporter` enforces a 4-stage deterministic gate:

1. **Gate 1 — Safe AST Parsing (`P0.5`)**: Parses expressions exclusively via `rosolver._parsing.parse_expr` (no `eval`/`sympify`), remapping PySR feature names (`x0, x1, ...`) to physical symbols via `variable_map`.
2. **Gate 2 — Dimensional Homogeneity & Normalization Audit**: Runs `diagnose_normalization_shortcuts` to reject dimensionally inconsistent expressions and diagnose implicit unit normalizations (e.g., `1.0 + L`) with exact `MissingConstantHypothesis` diagnostics.
3. **Gate 3 — Order-of-Magnitude Scale-Separation Audit (Sprint 10)**: Runs `check_equation_scales` over the stated or data-inferred operating envelope to reject scale-impossible expressions and warn on negligible bloated terms.
4. **Gate 4 — WLS Calibration & Jacobian Identifiability (`P2.1`)**: Lifts numeric PySR constants into refittable parameters (when `refit_numeric_constants=True`), runs `NumericalSolver.fit` (WLS with `sigma_y`/`weights`), computes `IdentifiabilityReport`, and ranks admissible candidates by BIC/RMSE/complexity.

---

## 2. Python API Example

```python
from rosolver import CandidateImporter

report = CandidateImporter.from_pysr_csv(
    "outputs/hall_of_fame.csv",
    target="T",
    dims={"T": "s", "L": "m", "g": "m/s^2"},
    variable_map={"x0": "L", "x1": "g"},
    data={"L": L_arr, "g": g_arr, "T": T_arr},
    sigma_y=u_T,
    refit_numeric_constants=True,
)
print(report.summary())

# Bridge top screened candidate directly into a Sprint 10 ClaimManifest
manifest = report.to_claim_manifest(top_k=1)
```

---

## 3. Optional `rosolver[pysr]` Dependency

Reading `hall_of_fame.csv` files, raw expression lists, and duck-typed `PySRRegressor` objects requires **zero** external dependencies beyond Rosolver's core stack. To execute PySR directly via `CandidateImporter.run_pysr(...)`, install:

```bash
pip install 'rosolver[pysr]'
```

---

## 4. Security & Scope Limitations

- **Zero LLM SDK wrappers**: LLM orchestration remains strictly external via `rosolver-mcp`.
- **No silent candidate drops**: Every rejected Pareto-front expression is recorded in `report.rejected_candidates` with its exact `rejection_stage` and `rejection_reason`.
- **Integer exponents preserved**: Numeric lifting refits floating-point coefficients while keeping integer exponents fixed so dimensional structure is never distorted.
