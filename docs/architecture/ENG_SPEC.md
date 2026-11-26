# **Rosolver — Product and Engineering Specification**

**Document type:** Canonical product architecture and prioritized technical specification  
**Baseline:** Rosolver `v0.1.0` (`68` passing tests across `5` test files; Python `>=3.10`)  
**Target release:** Rosolver `v0.2` (verification-credible scientific core); `v0.3+` (GUI, AI, and extended integrations)  
**Primary audience:** Academic researchers, computational scientists, and engineering practitioners

---

## **1. Product Vision, Scope & Architecture**

### **1.1 Vision & Modelling Lifecycle**

Rosolver evolves from a collection of symbolic and numerical utilities into an integrated, verification-aware scientific modelling environment built around one core proposition:

> **Rosolver helps researchers formulate physically consistent mathematical models, calibrate them against observations, simulate their behaviour, and establish the numerical credibility of their results.**

Rosolver differentiates itself not by replacing specialist solvers, but by unifying the full scientific modelling lifecycle on top of a single deterministic Python core:

```text
1. Formulate physical system & governing dependencies
        |
        v
2. Construct dimensionally consistent candidate equations
        |
        v
3. Discover & compare admissible model structures
        |
        v
4. Estimate unknown parameters from observational data
        |
        v
5. Simulate resulting ODE, PDE, or moving-boundary systems
        |
        v
6. Assess numerical accuracy, stability, conservation & convergence
        |
        v
7. Validate predictions against experimental measurements
        |
        v
8. Generate reproducible, traceable scientific reports
```

### **1.2 Intended Users & Application Focus**

| Persona | Core Needs | Initial Priority Domains (`v0.2`) |
| --- | --- | --- |
| **Academic researchers** | Formulate governing laws, derive constitutive relations, compare models, calibrate parameters, publish reproducible studies | Coupled heat & mass transfer; reaction-transport systems |
| **Computational engineers** | Verify discretisations, run MMS/GCI/timestep studies, audit conservation, evaluate numerical stiffness & coupling stability | Constitutive modelling; 1D ODE/PDE/Stefan verification |
| **Scientific software developers** | Wrap external solvers, automate verification regressions, standardize machine-readable V&V evidence | Experimental parameter identification & external output V&V |

Representative domains across the broader platform include chemical engineering, metallurgical engineering, fluid mechanics, thermal engineering, materials science, environmental modelling, computational physics, and physics-constrained scientific machine learning.

### **1.3 Non-Objectives & Scientific Guardrails**

Rosolver explicitly does **not** aim to:

- Replace SymPy, SciPy, PySR, OpenFOAM, FEniCSx, COMSOL, or mature specialist simulation packages.
- Claim that dimensional consistency proves physical correctness or automatically discover universal physical laws from arbitrary data.
- Claim formal **ASME V&V 20** or **AIAA S-141** certification solely because GCI, MMS, or validation comparison utilities are provided.
- Become a general-purpose CAD, mesh-generation, or 3D multiphysics platform.
- Introduce a graphical desktop application before the Python computational core and public API stabilize.

### **1.4 Six-Layer Functional Architecture**

Every interface (Python API, CLI, Jupyter notebooks, future desktop UI) must invoke the same Python computational core with zero duplicated scientific logic.

```text
Researcher / Python API / CLI / Jupyter / Future Desktop UI
                         |
                 Project and Model Layer
                         |
       +-----------------+------------------+
       |                 |                  |
  Symbolic Engine   Numerical Engine   Data and Calibration
       |                 |                  |
       +-----------------+------------------+
                         |
                   V&V Engine
                         |
             Results and Evidence Layer
                         |
       Reports / Plots / Export / Reproducibility
```

| Layer | Responsibility |
| --- | --- |
| **User interfaces** | Python API, CLI, Jupyter integration, and future PySide6 desktop presentation layer |
| **Project & model** | Persistent project workspace, validated `Model` definitions, variable/parameter/unit registries, execution ledger |
| **Symbolic engine** | Dependency DAGs, rational dimensional closure, Buckingham-Pi extraction, Clairaut checks, pluggable templates |
| **Numerical engine** | ODE/PDE/moving-boundary discretisation and time integration with solver diagnostics |
| **Data & calibration** | Dataset import with uncertainties, OLS/NLLS/WLS parameter estimation, identifiability diagnostics, model ranking |
| **V&V engine** | Independent code verification (MMS), solution verification (GCI, timestep sensitivity), conservation audits, experimental validation |
| **Evidence & reporting** | Standardized result/provenance objects, diagnostic plots, reproducible Markdown/structured reports |

**Architectural Principles:**

1. **Single computational core:** No scientific algorithm may be duplicated across CLI, notebooks, or UI layers.
2. **Explicit scientific assumptions:** Units, dimensions, ICs/BCs, solver settings, and applicability limits must be recorded explicitly.
3. **Independent verification:** Verification routines must be tested against analytical and external reference solutions, never solely against themselves.
4. **Reproducibility:** Saved projects must capture inputs, hashes, environment versions, configurations, and seeds sufficient to reproduce supported workflows (explicitly distinguishing reproducible configuration from cross-platform floating-point identity).
5. **Extensibility:** Solvers, equation templates, verification methods, and external adapters plug in through defined interfaces.
6. **Deterministic core:** Symbolic and numerical calculations remain deterministic where algorithms permit; stochastic routines record seeds and settings.
7. **Separation of proposal and acceptance:** Automated discovery or AI tools only propose drafts; deterministic schema, dimensional, and V&V checks determine scientific admissibility.

---

## **2. Repository Baseline Matrix (`v0.1.0` Standing)**

| Layer / Capability | Current Modules | Status | Key Audit Findings |
| --- | --- | --- | --- |
| **Expression parsing** | `rosolver/_parsing.py` | **Needs Hardening** | Uses raw `sp.sympify()` (unsafe `eval` surface for YAML/JSON/external inputs). |
| **Dimensional algebra** | `rosolver/dimensions.py` | **Implemented (Partial Units)** | Exact `sp.Rational` arithmetic & canonical Buckingham-Pi work well; `UNIT_TO_DIM` only covers 7 base SI unit aliases (no `N`, `J`, `W`, `Pa`, `Hz`, etc.). |
| **Dependency DAG** | `rosolver/system.py` | **Implemented** | Acyclic validation, path enumeration, rate identities work; `Variable` forces `positive=True` on all symbols. |
| **Symbolic engine** | `rosolver/symbolic.py` | **Implemented (Coupled Templates)** | Chain-rule, Clairaut exactness, Poincare integration, rate-identity bridge work; Pi templates hardcoded in `symbolic.py`/`numerical.py`; `_solve_intercept_term` lacks dimensional check. |
| **Calibration & ranking** | `rosolver/numerical.py` | **Partial** | OLS & analytic-Jacobian NLLS with BIC/AIC/RMSE work; lacks WLS (`sigma_y`), missing-data policy, covariance/condition identifiability diagnostics; silently swallows fit errors. |
| **Numerical solvers** | `rosolver/odesys.py`, `rosolver/integrators.py`, `rosolver/pde.py`, `rosolver/phasechange.py`, `rosolver/simulate.py` | **Implemented (Fragmented)** | Solid FVM 1D, `ODESystem`, fixed-step integrators, and front-fixing `Stefan1D`; fragmented result dataclasses; `newton_failures` dropped; `ODESystem` lacks events; no unified `Model.solve()`. |
| **V&V engine** | `rosolver/verify.py` | **Needs Hardening** | GCI, MMS, timestep, conservation, stiffness, Aitken $\Delta^2$, co-sim present; lacks 5-state classification; `_celik_observed_order` and `gci_from_values` silently mask/clamp edge cases; conservation wording conflates accounting vs PDE correctness. |
| **Model & Project layer** | — | **Missing** | No `Model`, `Project`, YAML/JSON persistence, SHA-256 provenance, or append-only execution ledger. |
| **External data & validation** | — | **Missing** | Basic CSV column loader only in `rosolver/cli.py`; no `SolverAdapter`, HDF5/Pandas, uncertainty import, or `ValidationStudy`. |
| **Reporting & workflow CLI** | `rosolver/cli.py` | **Partial (CLI only)** | Legacy flag CLI (`--rel`, `--gci`, `--sweep`, `--plot`) works; no subcommands (`init`, `verify`, `report`), no `--json`, no Markdown/PDF report generator. |
| **Tests & benchmarks** | `tests/` (`5` files, `68` tests) | **Partial** | Strong unit/integration coverage, but missing independent Celik (2008) GCI check, 1D analytical heat-conduction Fourier benchmark, and Stefan convergence-order check. |
| **Desktop GUI & AI** | — | **Deferred (`v0.3+`)** | Intentionally out of scope until `v0.2` core stabilizes. |

---

## **3. Prioritized Engineering Requirements (`P0` → `P5`)**

Requirements are strictly ordered by engineering and scientific priority. **Tier P0–P3** define the `v0.2` release scope; **Tier P4** covers cross-cutting architecture, quality, and release gates; **Tier P5** captures post-`v0.2` extensions.

---

### **Tier P0 — Scientific Credibility, Security & Verification Core**

#### **`P0.1` Verification & Validation Taxonomy, 5-State Classification & Standards Framing**
- **Status:** Missing (currently binary `passed: bool`).
- **Specification:**
  - Strictly distinguish **Code Verification** (*are equations solved correctly?*), **Solution Verification** (*how accurate is the numerical solution on this mesh/timestep?*), and **Model Validation** (*how well does the model match physical reality?*) across APIs, reports, and docs.
  - Replace binary pass/fail reduction with a canonical 5-state classification enum across all V&V reports:

| Classification | Meaning |
| --- | --- |
| `Passed` | Defined quantitative acceptance criteria were satisfied |
| `Failed` | Defined quantitative acceptance criteria were violated |
| `Inconclusive` | Available grids, steps, or data cannot support the required test assumptions |
| `NotAssessed` | Test or check was not executed |
| `Warning` | Computation succeeded, but assumptions or diagnostics require scientist review |

  - Frame methods against **ASME V&V 20-2009 (R2021)**, **AIAA S-141-2024**, **Roache**, and **Celik et al. (2008)** without implying formal standards certification.
- **Acceptance criteria:**
  - Every verification and validation report exposes a 5-state `status`, metric definition, assumptions, warnings, and limitations.
  - Public docs and CLI/report headers clearly state non-certification boundaries.

#### **`P0.2` Code Verification: Hardened MMS & Independent Benchmark Suite**
- **Status:** Partial (`mms_study` exists; independent benchmark coverage incomplete).
- **Specification:**
  - Maintain and test `ConservationLaw1D.manufactured_source` + `mms_study` so MMS exercises the actual discretisation, boundary conditions, and time integrator (never post-processing alone).
  - Establish an automated `benchmarks` suite testing against **independently derived analytical or published reference values** (never testing an algorithm solely against itself):
    1. **Benchmark 1 — Analytical ODE:** $\frac{dy}{dt} = -ky$, $y(t) = y_0 e^{-kt}$ (accuracy, tolerance sensitivity, `dt` order, parameter recovery).
    2. **Benchmark 2 — Analytical 1D Heat Conduction:** $\frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial x^2}$ compared against closed-form Fourier series solution (IC/BC correctness, spatial/temporal order, energy balance, plus `Stefan1D` Neumann spatial convergence order).
    3. **Benchmark 3 — Manufactured PDE Solution:** e.g. $T(x, t) = e^{-t}\sin(\pi x)$ (symbolic forcing derivation, observed $L_1/L_2/L_\infty$ order, boundary handling).
    4. **Benchmark 4 — Dimensional Analysis & Buckingham-Pi:** Standard physical problems checked against manual analytical exponents, redundant group detection, and inconsistent-equation rejection.
    5. **Benchmark 5 — Synthetic Parameter Identification:** Known nonlinear model with controlled noise (parameter recovery within tolerances, bounds, identifiability warnings, BIC selection).
    6. **Benchmark 6 — External Solver & Celik (2008) GCI Verification:** Published reference GCI calculation and external CSV solver output import + GCI/report generation.
- **Acceptance criteria:**
  - All 6 benchmark problems define mathematical statement, reference solution, ICs/BCs, expected convergence order, tolerances, and known limitations, and run in CI.

#### **`P0.3` Solution Verification: Hardened GCI, Grid Refinement & Time-Step Sensitivity**
- **Status:** Needs Hardening (`rosolver/verify.py`).
- **Specification:**
  - Harden `gci_from_values` and `_celik_observed_order`:
    - Expose convergence warnings and set status to `Inconclusive` or `Warning` when `_celik_observed_order` encounters sign/bracket failure (`num * den <= 0`), oscillatory regime, $r < 1.3$, or non-asymptotic behavior instead of silently breaking or returning `Passed`.
    - Preserve sign information rather than masking divergence with `abs()` inside order estimation; never silently clamp `p_eff` to `[0.1, 10.0]` without recording a `Warning`.
    - Support $>3$ refinement levels cleanly (diagnose full sequence or explicitly report which levels are used) instead of silently discarding coarser grids.
  - Keep spatial grid refinement (`grid_convergence_study`) and time-step refinement (`timestep_study`) independently configurable, exposing refinement ratios, extrapolated value, safety factor ($F_s = 1.25$ default), $GCI_{\text{fine}}$, $GCI_{\text{coarse}}$, asymptotic ratio, CFL/Fourier numbers, and hybrid event-timestamp drift.
- **Acceptance criteria:**
  - GCI matches published Celik et al. (2008) reference values within documented tolerances.
  - Invalid, non-monotonic, or $r < 1.3$ sequences emit structured warnings and appropriate non-`Passed` status.

#### **`P0.4` Conservation Accounting Clarification & Solver Diagnostics**
- **Status:** Needs Hardening (`rosolver/pde.py`, `rosolver/verify.py`, `rosolver/integrators.py`, `rosolver/odesys.py`).
- **Specification:**
  - Explicitly distinguish three conservation levels in APIs, docstrings, and reports:
    1. **Accounting consistency:** Whether boundary flux and source ledgers integrated alongside the state equations close to round-off.
    2. **Discrete conservation:** Whether the numerical scheme preserves the intended discrete conservation law independently of ledger augmentation.
    3. **Physical correctness:** Whether equations, BCs, and physical inputs represent the physical system (requires independent benchmarks/validation).
  - Propagate `newton_failures` from `integrate_fixed` into `Trajectory`, `PDESolution`, and `StefanSolution` rather than hardcoding `success=True`.
  - Standardize failure/warning reporting across `ODESystem`, `ConservationLaw1D`, and `Stefan1D`.
  - Retain Jacobian stiffness diagnostics (`analyze_stiffness`), Aitken $\Delta^2$ fixed-point stabilization (`solve_partitioned_fixed_point`), and coupled co-simulation (`cosimulate_coupled_odes`).
- **Acceptance criteria:**
  - Failed implicit Newton solves set `success=False` and surface explicit warnings/errors.
  - Conservation reports label `pde_fvm` checks as *internal accounting consistency* and cite independent conservation benchmarks separately.

#### **`P0.5` Expression Parsing Security & Input Validation**
- **Status:** Needs Hardening (`rosolver/_parsing.py`).
- **Specification:**
  - Replace raw `sp.sympify()` execution on untrusted strings with a strict `ast.parse` whitelist (arithmetic operators, comparisons where valid, explicitly allow-listed math functions, and declared symbols only — no attribute access, comprehensions, lambdas, or `eval` side effects).
  - Prohibit loading untrusted Python `pickle` files by default.
  - Validate serialized model specifications and treat external CSV/HDF5/YAML/JSON datasets as untrusted inputs.
  - Keep API keys and credentials strictly out of project files, ledgers, and reports.
- **Acceptance criteria:**
  - Malicious or arbitrary Python expression strings (e.g., `__import__(...)`, attribute chains, subscripting) are rejected before SymPy evaluation with explicit `ValueError`.

---

### **Tier P1 — Unified Model Specification, Results & Project Foundation**

#### **`P1.1` Standardized Result & Diagnostic Objects**
- **Status:** Missing (currently 7 disconnected result dataclasses).
- **Specification:**
  - Define consistent base metadata/protocol fields across simulation, calibration, and verification results: `status` (5-state classification), `warnings`, `assumptions`, `solver_config`, `units`, `metrics`, `timing_s`, `run_id`, `model_version`, and structured serialization (`to_dict()` / `to_json()`).
  - Preserve existing attributes and `.summary()` methods on `DerivationResult`, `FitResult`, `ODESolution`, `HybridODESolution`, `Trajectory`, `PDESolution`, `StefanSolution`, `HybridStefanSolution`, `SimulationResult`, `GCIReport`, `MMSReport`, `TimestepReport`, `ConservationAuditReport`, `StiffnessReport`, `PartitionedCouplingReport`, and `CoSimulationReport`.
- **Acceptance criteria:**
  - Every computational result object can be inspected uniformly by the reporting engine and execution ledger without breaking existing attribute access.

#### **`P1.2` Validated Scientific Model Specification & Safe Serialization**
- **Status:** Missing.
- **Specification:**
  - Introduce a central `Model` specification independent of any specific numerical solver, representing:
    - Name, version, description, physical domain.
    - Independent variables, dependent/state variables, parameters (fixed or unknown with optional bounds), and explicit unit/dimension metadata.
    - Support both base SI dimensions and common derived SI units (`N`, `J`, `W`, `Pa`, `Hz`, `V`, `C`, `Ohm`, etc.) in `rosolver/dimensions.py` without breaking custom domain dimensions (`USD`, `Users`, `Cells`).
    - Dependency relationships (acyclic DAG validated; cycles rejected with clear path errors).
    - Governing equations / candidate structures, source/sink terms, constitutive relations, initial conditions, boundary conditions, and solver/V&V configuration hints.
  - Support lossless YAML and JSON serialization via a validated internal schema (`pyyaml`) without executing arbitrary YAML tags or code (`yaml.safe_load`).

Illustrative model schema:

```yaml
model:
  name: heat_transfer_1d
  version: "0.1"
  description: "One-dimensional transient heat conduction"
  domain: thermal_engineering

variables:
  independent:
    - name: x
      symbol: x
      unit: m
    - name: t
      symbol: t
      unit: s
  dependent:
    - name: temperature
      symbol: T
      unit: K

parameters:
  - name: thermal_diffusivity
    symbol: alpha
    value: 1.0e-5
    unit: m^2/s
    status: fixed

equations:
  - id: heat_equation
    expression: "dT/dt = alpha * d2T/dx2"
    type: pde

conditions:
  initial:
    - expression: "T(x, 0) = T_initial(x)"
  boundary:
    - location: x_min
      expression: "T = T_left"
    - location: x_max
      expression: "T = T_right"

numerics:
  solver: method_of_lines
  time_integrator: RK45
  tolerances:
    rtol: 1.0e-6
    atol: 1.0e-9
```

- **Acceptance criteria:**
  - `Model` objects construct programmatically or via `Model.from_file(...)`, round-trip cleanly to YAML/JSON, validate units/dimensions/acyclic DAGs, and reject incomplete or cyclic specs with actionable errors.

#### **`P1.3` Symbolic Engine Hardening & Pluggable Equation Template Registry**
- **Status:** Partial (`rosolver/symbolic.py` & `rosolver/numerical.py` hardcode templates).
- **Specification:**
  - Preserve existing Chain-Rule + Dimensional Closure, Buckingham-Pi extraction, Clairaut mixed-partial symmetry coupling, Poincare potential integration, and rate-identity closure.
  - Extract a pluggable `EquationTemplate` registry supporting initial templates (**Monomial**, **Affine**, **Saturating Hill**, **Rosin–Rammler**) where each template defines mathematical form, variables, parameters, dimensional constraints, parameter bounds, applicable assumptions, estimation hints, and differentiation/evaluation hooks.
  - Allow new templates to be registered without modifying `SymbolicSolver` or `NumericalSolver`.
  - Fix symbolic gaps: enforce dimensional consistency in `_solve_intercept_term`, avoid forcing `positive=True` where signed state variables are physically required, retain full derivation & rejection provenance on every `CandidateEquation`, and verify Clairaut prerequisites before applying mixed-partial symmetry.
- **Acceptance criteria:**
  - Third-party/test templates register via public registry API and participate in discovery + calibration.
  - Every rejected or failed candidate includes an explicit scientific/mathematical reason.

#### **`P1.4` Unified Numerical Engine Interface**
- **Status:** Partial (separate `ODESystem`, `ConservationLaw1D`, `Stefan1D`, `NumericalSolver` APIs).
- **Specification:**
  - Provide a common `model.solve(...)` execution entry point dispatching across ODE (IVP, coupled, stiff/non-stiff, events, time-dependent sources), 1D PDE (FVM/method-of-lines, Dirichlet/Neumann/outflow/periodic BCs, conservation diagnostics), and 1D moving-boundary (`Stefan1D`) models.
  - Add terminal/non-terminal event detection to `ODESystem.solve(...)` so hybrid event workflows are not limited to 1D scalar `NumericalSolver.integrate_hybrid_ode`.
  - Keep existing classes (`ODESystem`, `ConservationLaw1D`, `Stefan1D`, `NumericalSolver`) fully operational.
- **Acceptance criteria:**
  - `model.solve(method=..., t_span=..., ...)` executes ODE and 1D PDE models uniformly, preserves solver settings in result metadata, and exposes failures/warnings cleanly.

#### **`P1.5` Persistent Scientific Project & Append-Only Execution Ledger**
- **Status:** Missing.
- **Specification:**
  - Introduce a `Project` workspace manager organizing models, data, calibration, simulations, verification, validation, reports, and logs:

```text
my_rosolver_project/
├── project.yaml
├── models/
│   ├── model.yaml
│   ├── equations.yaml
│   └── dependencies.yaml
├── data/
│   ├── experimental/
│   ├── reference/
│   └── processed/
├── calibration/
│   ├── configuration.yaml
│   └── results/
├── simulations/
│   ├── configuration.yaml
│   └── results/
├── verification/
│   ├── mms/
│   ├── gci/
│   ├── conservation/
│   └── timestep/
├── validation/
│   ├── configuration.yaml
│   └── results/
├── reports/
│   ├── figures/
│   ├── tables/
│   └── final_report.md
└── logs/
    └── execution.log
```

  - Record project manifest (`project.yaml`) and an **append-only** execution ledger capturing: `run.id`, `type`, `model` + version, UTC timestamp, software metadata (Python, Rosolver, core dependency versions, OS), solver config, random seeds, SHA-256 hashes of model and input datasets, and output artifact paths.
  - Permit directory reference resolution relative to project root without breaking on project move.

Illustrative execution ledger record:

```yaml
run:
  id: simulation_001
  type: pde_simulation
  model: heat_transfer_1d
  timestamp: "2026-11-26T12:00:00Z"
  software:
    rosolver: "0.2.0"
    python: "3.11"
  configuration:
    solver: RK45
    rtol: 1.0e-6
    atol: 1.0e-9
  inputs:
    model_hash: "<sha256>"
    dataset_hash: "<sha256>"
  outputs:
    result: simulations/results/run_001.json
```

- **Acceptance criteria:**
  - Saved projects reload cleanly, detect missing/modified input hashes, append (never overwrite) execution history, and reproduce supported runs from stored configuration.

---

### **Tier P2 — Calibration, Interoperability & Experimental Validation**

#### **`P2.1` Parameter Estimation, WLS & Identifiability Diagnostics**
- **Status:** Partial (`rosolver/numerical.py`).
- **Specification:**
  - Extend `NumericalSolver` calibration into a reproducible workflow supporting:
    - OLS, analytic-Jacobian NLLS, **Weighted Least Squares (WLS)** using measurement uncertainties ($\sigma_y$), robust loss (`huber`, `soft_l1`, `cauchy`), parameter bounds, and explicit missing-data policies (`raise`, `drop`, `mask`).
    - Identifiability diagnostics: parameter correlation matrix, Jacobian singular values / condition number, rank deficiency detection, and confidence intervals — explicitly distinguishing **numerical convergence**, **parameter identifiability**, **statistical uncertainty**, and **physical plausibility**.
    - Multi-criterion candidate ranking separating: (1) mathematically admissible, (2) statistically supported (BIC/AIC/RMSE/$R^2$), (3) physically plausible, and (4) independently validated candidates.
    - Replace silent `except Exception: continue` in `rank_candidates` and `select_sparse_pi` with recorded per-candidate rejection diagnostics.
  - Post-`v0.2` extensions: MLE, Bayesian/MCMC estimation, bootstrap CIs, profile likelihood, global optimization.
- **Acceptance criteria:**
  - Collinear or unidentifiable parameterizations emit explicit identifiability warnings rather than silent `pinv` estimates.
  - Weighted fitting incorporates observational uncertainties and reproduces known synthetic parameters within tolerance.

#### **`P2.2` Standard External Data Interface**
- **Status:** Missing (only ad-hoc `_load_csv_columns` in `rosolver/cli.py`).
- **Specification:**
  - Provide a canonical dataset container carrying spatial coordinates, time coordinates, field values, mesh/grid metadata, physical units, boundary/initial metadata where available, solver metadata, measurement/numerical uncertainties, and provenance hashes.
  - Core `v0.2` formats: **CSV**, **JSON**, **YAML**, **NumPy** (`.npz`/arrays). Optional extras (`rosolver[io]`): **HDF5** (`h5py`), **Pandas** (`pandas`).
- **Acceptance criteria:**
  - External CSV datasets load with explicit coordinate/field/unit/uncertainty mapping and reject malformed or mismatched headers cleanly.

#### **`P2.3` External Solver Adapter Architecture**
- **Status:** Missing.
- **Specification:**
  - Define an abstract `SolverAdapter` interface so external numerical results enter the same V&V pipeline without rewriting external solvers or duplicating V&V logic:

```python
class SolverAdapter:
    def load_results(self, source): ...
    def extract_fields(self): ...
    def extract_mesh_metadata(self): ...
    def extract_solver_metadata(self): ...
    def export_standard_format(self): ...
```

  - Unsupported metadata fields must be marked unavailable (`None` / `NotAssessed`), never fabricated.
  - Integration target priority:

| Software | Priority | Interface | Target Milestone |
| --- | --- | --- | --- |
| **SciPy** | High | Native Python API | `v0.2` |
| **Custom research codes** | High | Generic CSV / NumPy / Python adapter | `v0.2` |
| **FiPy** | High | Python adapter | `v0.3` |
| **FEniCS / FEniCSx** | High | Python adapter | `v0.3` |
| **OpenFOAM** | Medium | Field/probe export-import adapter | `v0.3+` |
| **COMSOL / MATLAB** | Medium | CSV, MAT, or HDF5 adapter | `v0.3+` |

- **Acceptance criteria:**
  - At least one complete external CSV solver workflow runs end-to-end through `VerificationStudy` (GCI + report) in `v0.2` without modifying V&V algorithms.

#### **`P2.4` Experimental Model Validation Engine**
- **Status:** Missing.
- **Specification:**
  - Build a `ValidationStudy` subsystem comparing computational predictions (with numerical uncertainty $u_{\text{num}}$) against experimental datasets (with measurement uncertainty $u_{\text{exp}}$) across multiple operating points.
  - Compute validation error ($E = S - D$), combined validation uncertainty ($u_{\text{val}} = \sqrt{u_{\text{num}}^2 + u_{\text{exp}}^2 + u_{\text{input}}^2}$ where available), comparison metrics, operating condition records, and comparison plots.
  - Follow ASME V&V 20 framing without claiming formal compliance.
- **Acceptance criteria:**
  - Multi-point experimental comparisons produce structured validation results with explicit $u_{\text{num}}$, $u_{\text{exp}}$, 5-state classification, and comparison plots.

---

### **Tier P3 — Reporting, Workflow CLI & `v0.2` Flagship Demo**

#### **`P3.1` Structured Scientific Reporting Engine**
- **Status:** Missing (only `.summary()` text and standalone PNG plotting helpers exist).
- **Specification:**
  - Generate reproducible reports directly from structured result objects and execution ledger records (never from ad-hoc string scraping).
  - Support 8 canonical report scopes: Symbolic derivation, Dimensional analysis, Calibration, Numerical simulation, Code verification, Solution verification, Model validation, and Integrated project study.
  - Required contents: study title, model name/version, scientific assumptions, governing equations, variable/unit table, datasets & hashes, numerical settings, results, labeled plots/tables, V&V classifications, warnings/limitations, software metadata, and source run IDs.
  - `v0.2` formats: **Markdown**, **PNG/SVG** figures, **CSV**, **JSON**, **YAML**; **PDF** via optional/headless backend (`v0.2` target if lightweight, otherwise `v0.3`); LaTeX/HTML/Jupyter/DOCX later.
- **Acceptance criteria:**
  - Report generation is deterministic, headless (no GUI), preserves visible `Failed`/`Inconclusive` findings, and never mutates simulation outputs.

#### **`P3.2` High-Level Python API, Workflow CLI & Jupyter Integration**
- **Status:** Partial (legacy flat-flag CLI and `formulate()`/`simulate()` APIs exist).
- **Specification:**
  - Expose a cohesive high-level Python API alongside preserved existing functions:

```python
from rosolver import Project, Model

project = Project.create("heat_transfer_study")
model = Model.from_file("models/heat_transfer.yaml")
project.add_model(model)

symbolic = model.analyse_symbolically()
calibration = model.calibrate(data="data/experimental.csv")
simulation = model.solve()
verification = simulation.verify(methods=["mms", "gci", "conservation"])
project.report(output="reports/study.md")
```

  - Add workflow CLI subcommands while maintaining backwards compatibility for existing flags (`--rel`, `--gci`, `--sweep`, `--plot`):
    - `rosolver init <dir>`
    - `rosolver validate <project>` *(checks project structure & model spec)*
    - `rosolver model analyse <project>`
    - `rosolver calibrate <project>`
    - `rosolver simulate <project>`
    - `rosolver verify <project>`
    - `rosolver validate-results <project>` *(compares predictions vs experimental data — kept distinct from `validate`)*
    - `rosolver report <project>`
    - `rosolver info <project>`
  - CLI must support `--json`, non-interactive CI execution, progress output, and explicit exit codes.
  - Jupyter notebooks remain a first-class interactive environment (rich `_repr_markdown_` / `_repr_svg_` helpers optional, zero runtime dependency on IPython/Jupyter).
- **Acceptance criteria:**
  - All CLI subcommands execute headlessly with documented exit codes and `--json` mode, while existing CLI invocations remain green.

#### **`P3.3` `v0.2` Flagship End-to-End Demonstration**
- **Status:** Missing (examples currently exercise isolated features).
- **Specification:**
  - Deliver one complete, self-contained **1D transient heat conduction** project demonstrating the entire lifecycle in order:
    1. Model YAML/Python specification & validation
    2. Dimensional consistency check
    3. Governing equation formulation
    4. Numerical PDE solve
    5. Analytical Fourier-series comparison
    6. MMS code verification
    7. Spatial grid (`GCI`) & timestep convergence
    8. Conservation audit
    9. Reproducible Markdown report + figures
- **Acceptance criteria:**
  - Runs via Python API and CLI in CI and produces a complete traceable project folder and Markdown report.

---

### **Tier P4 — Target Package Layout, Non-Functional Requirements, Docs & Release Gate**

#### **`P4.1` Incremental Target Package Architecture**
- **Status:** Flat 14-file `rosolver/` layout.
- **Specification:**
  - Refactor incrementally toward a modular package tree while preserving all imports in `rosolver/__init__.py` via compatibility re-exports:

```text
rosolver/
├── __init__.py
├── cli.py
├── core/          # project.py, model.py, variables.py, parameters.py, units.py, dimensions.py, dependencies.py, results.py, exceptions.py
├── symbolic/      # discovery.py, dimensional.py, buckingham_pi.py, clairaut.py, templates.py, ranking.py, derivation.py
├── calibration/   # data.py, least_squares.py, nonlinear.py, uncertainty.py, diagnostics.py
├── numerics/      # ode.py, pde.py, phasechange.py, integrators.py, discretization.py, solvers.py, conservation.py, results.py
├── verification/  # mms.py, gci.py, convergence.py, timestep.py, conservation.py, stiffness.py, coupling.py
├── validation/    # experimental.py, uncertainty.py, comparison.py, metrics.py
├── integrations/  # base.py, scipy.py, generic.py (+ optional fipy.py, fenics.py, openfoam.py)
├── io/            # yaml.py, json.py, csv.py, hdf5.py, project.py
├── reporting/     # markdown.py, pdf.py, plots.py, templates/
├── ai/            # (v0.3+) interface.py, specification.py, candidate_generation.py
└── tests/         # unit/, integration/, benchmarks/, regression/
```

- **Acceptance criteria:**
  - Zero circular imports; `rosolver/__init__.__all__` passes backwards-compatibility regression tests at every step.

#### **`P4.2` Cross-Cutting Non-Functional Requirements**
- **Performance:** Cache compiled symbolic/lambdified expressions safely (`ConservationLaw1D._cache` pattern); vectorize array ops; avoid copies of large datasets; design long-running solvers to accept cancellation/progress callbacks for future GUI non-blocking execution; record wall-clock timing on expensive operations; establish targets from benchmark measurements rather than arbitrary numbers.
- **Reliability:** Validate inputs up front; fail loudly on violated assumptions; never silently invent missing units or BCs; preserve diagnostics on solver failure; keep regression tests green.
- **Maintainability:** Single-responsibility modules; typed public APIs; docstrings on public classes/functions; strict separation between algorithms, I/O, and UI; extension protocols for templates, solvers, V&V checks, and adapters.
- **Portability:** Linux, macOS, Windows; core executes completely without GUI frameworks.
- **Security:** Enforced per `P0.5`.

#### **`P4.3` Workflow-Oriented Documentation & Standards Reference**
- **Specification:**
  - Organize `docs/` around the scientific modelling workflow:
    - `docs/index.md`
    - `docs/getting_started/` (`installation.md`, `quickstart.md`, `first_project.md`)
    - `docs/concepts/` (`architecture.md`, `scientific_models.md`, `dimensional_analysis.md`, `equation_discovery.md`, `calibration.md`, `numerical_simulation.md`, `verification_validation.md`)
    - `docs/tutorials/` (`ode_modelling.md`, `heat_transfer.md`, `parameter_estimation.md`, `mms_verification.md`, `gci_analysis.md`, `external_solver.md`)
    - `docs/api/`
    - `docs/scientific_reference/` (`mathematical_methods.md`, `numerical_methods.md`, `verification_methods.md`, `assumptions_and_limitations.md`)
    - `docs/development/` (`architecture.md`, `testing.md`, `contributing.md`)
  - Every numerical/symbolic method doc must cover 8 items: mathematical definition, assumptions, applicable conditions, input requirements, output interpretation, known limitations, literature references (ASME V&V 20-2009, AIAA S-141-2024, Roache, Celik et al. 2008), and linked verification tests.

#### **`P4.4` `v0.2` Release Gate, Success Metrics & Packaging**

**`v0.2` Mandatory vs Deferred Scope:**

| Component | `v0.2` Mandatory | Deferred (`v0.3+`) |
| --- | --- | --- |
| Symbolic & dimensional engine | Preserved + pluggable templates (`P1.3`) | AI-powered equation generation (`P5.2`) |
| Calibration | OLS/NLLS/WLS + identifiability diagnostics (`P2.1`) | Bayesian/MCMC, profile likelihood, global opt |
| Simulation | Analytical ODE, PDE & Stefan benchmarks (`P0.2`, `P1.4`) | 2D/3D multiphysics simulation |
| Verification | Hardened MMS, GCI, dt, conservation, stiffness, Aitken (`P0.1`–`P0.4`) | Formal standards certification workflows |
| Project & data | Persistent `Project`, `Model` YAML, execution ledger, CSV import (`P1.2`, `P1.5`, `P2.2`, `P2.3`) | Extensive FiPy/FEniCSx/OpenFOAM/COMSOL adapters |
| Validation & reporting | `ValidationStudy` + Markdown/plot reports (`P2.4`, `P3.1`) | DOCX/HTML/notebook export |
| Interfaces & docs | High-level API, subcommand CLI, heat-transfer demo, docs (`P3.2`, `P3.3`, `P4.3`) | PySide6 desktop GUI (`P5.1`), cloud collaboration |

**Measurable `v0.2` Success Metrics:**

| Area | Metric | Target Outcome |
| --- | --- | --- |
| Scientific credibility | Independent benchmark coverage | All 6 `P0.2` benchmarks automated & passing |
| Numerical accuracy | Analytical ODE/PDE/Stefan errors | Within documented tolerances & expected convergence order |
| Code verification | MMS spatial order | Observed $L_2$ order matches formal order within tolerance |
| Solution verification | GCI consistency | Matches Celik (2008) benchmark; flags invalid/oscillatory/divergent sequences |
| Calibration | Synthetic parameter recovery | Recovers true parameters; flags collinear/unidentifiable cases |
| Reproducibility | Project reload & rerun | Reloaded project reproduces configured workflow & verifies hashes |
| Interoperability | External CSV workflow | Imports external mesh study, computes GCI, emits report |
| Maintainability | Regression stability | All `v0.1.0` public APIs & tests remain green in CI |
| Usability | End-to-end flagship demo | Complete transient heat-conduction study from model to Markdown report |
| Performance | Benchmark timing | Baseline runtimes recorded in CI (no unverified performance claims) |

**Packaging & Distribution:**
- Maintain MIT-licensed open-source core (`numpy`, `scipy`, `sympy`, `matplotlib`, `pyyaml`).
- Keep heavy integrations (`h5py`, `pandas`, `weasyprint`/PDF, `PySide6`, `pysr`, AI SDKs) as optional `pyproject.toml` extras.
- Distribute via PyPI, GitHub repository with CI, versioned docs, example projects, benchmark suite, and optional container/Docker environment.

---

### **Tier P5 — Post-`v0.2` Deferred Extensions**

#### **`P5.1` Desktop Application Strategy (`v0.3+`)**
- **Technology:** **PySide6** (Qt for Python) as a strict presentation layer over the public Python API (no duplicated scientific algorithms).
- **Major Views:** (A) Project explorer, (B) Model editor & dependency graph, (C) Symbolic analysis inspector, (D) Calibration workspace, (E) Simulation workspace, (F) V&V dashboard, (G) Reporting preview & export.
- **Acceptance criteria:** Projects are 100% interchangeable between API, CLI, and GUI; GUI executes long computations asynchronously without blocking; core remains fully functional without PySide6 installed.

#### **`P5.2` AI-Assisted Scientific Modelling & Symbolic Regression Integration (`v0.4+`)**
- **Principle:** *AI proposes. Rosolver evaluates. Scientific evidence determines acceptance.*
- **Capabilities:** Natural-language physical description → structured `Model` draft, candidate equation/BC suggestions, missing-variable detection, diagnostic summaries, report drafting, plus **PySR** / symbolic-regression candidate ingestion.
- **Guardrails:** AI outputs must map to structured `Model`/AST specs, declare all variables/units, pass deterministic schema + dimensional + V&V pipelines, require explicit human approval, record AI-proposed assumptions separately from verified facts, never auto-execute generated Python code, and remain strictly optional (zero API key required for core usage).

#### **`P5.3` Sustainability & Potential Commercial Extensions**
- Keep the full scientific modelling, simulation, calibration, verification, CLI, Python API, and Markdown reporting core MIT open-source and usable offline without subscriptions.
- Explore optional commercial extensions (e.g., Desktop Professional, institutional deployment, team workspaces, enterprise V&V packs, specialist adapters, hosted AI assistance) only after open-source user demand is validated.
