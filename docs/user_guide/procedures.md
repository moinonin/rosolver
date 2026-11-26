# Rosolver Extended User & Procedure Guide (`v0.2.0`)

This guide documents every scientific workflow in **Rosolver**—from qualitative equation **discovery** and **model specification** to **numerical solutions**, **calibration & experimentation**, **code/solution verification**, **experimental validation**, and **reproducible scientific reporting**—showing both **Python API** and **CLI** execution.

---

## Contents

1. [Module Map & Entry Points](#1-module-map--entry-points)
2. [Procedure 1: Symbolic Discovery & Dimensional Closure (`rosolver.symbolic`)](#2-procedure-1-symbolic-discovery--dimensional-closure-rosolversymbolic)
3. [Procedure 2: Declarative `Model` Specification & Validation (`rosolver.core`)](#3-procedure-2-declarative-model-specification--validation-rosolvercore)
4. [Procedure 3: Numerical Solutions (`rosolver.numerics`)](#4-procedure-3-numerical-solutions-rosolvernumerics)
5. [Procedure 4: Calibration, Datasets & Experimentation (`rosolver.calibration`, `rosolver.io`)](#5-procedure-4-calibration-datasets--experimentation-rosolvercalibration-rosolverio)
6. [Procedure 5: Verification & External Solver Ingestion (`rosolver.verification`, `rosolver.integrations`)](#6-procedure-5-verification--external-solver-ingestion-rosolververification-rosolverintegrations)
7. [Procedure 6: Experimental Model Validation (`rosolver.validation`)](#7-procedure-6-experimental-model-validation-rosolvervalidation)
8. [Procedure 7: Persistent `Project` Workspaces, Replay & Scientific Reporting (`rosolver.core`, `rosolver.reporting`)](#8-procedure-7-persistent-project-workspaces-replay--scientific-reporting-rosolvercore-rosolverreporting)
9. [Procedure 8: Workflow CLI & Jupyter Notebooks](#9-procedure-8-workflow-cli--jupyter-notebooks)

---

## 1. Module Map & Entry Points

Every public class and function is exported from top-level `rosolver`, and also grouped by domain subpackage:

| Subpackage | Purpose | Primary Classes / Functions |
| --- | --- | --- |
| `rosolver.core` | Core abstractions & persistence | `Dimension`, `System`, `Model`, `Project`, `StandardResult`, `VerificationStatus` |
| `rosolver.symbolic` | Qualitative & dimensional discovery | `formulate`, `SymbolicSolver`, `DerivationResult`, `TEMPLATE_REGISTRY` |
| `rosolver.calibration` | Parameter fitting & identifiability | `NumericalSolver`, `FitResult`, `IdentifiabilityReport`, `CandidateRankingReport`, `Dataset` |
| `rosolver.numerics` | PDE, ODE & Stefan solvers | `ConservationLaw1D`, `BC`, `ODESystem`, `EventSpec`, `Stefan1D`, `integrate_fixed` |
| `rosolver.verification` | Code & solution V&V | `mms_study`, `gci_from_values`, `grid_convergence_study`, `timestep_study`, `audit_pde_conservation`, `audit_ode_invariants`, `analyze_stiffness`, `solve_partitioned_fixed_point`, `cosimulate_coupled_odes` |
| `rosolver.validation` | Experimental comparison | `ValidationStudy`, `ValidationReport`, `ValidationPoint` |
| `rosolver.integrations` | External solver adapters | `SolverAdapter`, `GenericCSVAdapter`, `SciPyAdapter`, `VerificationStudy` |
| `rosolver.io` | Safe YAML/JSON/CSV/NPZ/HDF5 I/O | `Dataset`, `Model`, `Project`, `sha256_file`, `to_jsonable` |
| `rosolver.reporting` | Reports & V&V plotting | `ScientificReport`, `ReportGenerator`, `REPORT_TYPES`, `plot_simulation`, `plot_validation`, `plot_gci_study`, `plot_mms_convergence`, `plot_conservation_audit` |

Run built-in example suites directly:

```bash
python -m examples.heat_transfer_study.run_study --output outputs/heat_transfer_study
python -m examples.ch1_reproductions
python -m examples.icing_turbomachinery
python -m examples.auto_simulate
python -m examples.verification_suite
python -m examples.stefan_validation
```

---

## 2. Procedure 1: Symbolic Discovery & Dimensional Closure (`rosolver.symbolic`)

Use this procedure when you know **which variables depend on which** and optionally their **physical dimensions**, and want Rosolver to derive closed-form candidate laws.

### 2.1 One-Call Formulation (`formulate`)

```python
from rosolver import formulate

res = formulate(
    relations=["g = fn(a, b, c)", "b = fn(f; grav)"],
    dims={
        "g": "M*L^2/T^3",
        "a": "M/L^3",
        "b": "L/T",
        "c": "L^2",
        "f": "L",
        "grav": "L/T^2",
    },
    constants=["grav"],
    target="g",
    wrt="f",
    zero_initials=True,
)
print(res.summary())
print(res.to_latex())
```

- `relations`: list of qualitative DAG dependencies (`y = fn(x1, x2; const1)`). Cycles raise `ValueError`.
- `dims`: supports SI base dimensions (`M, L, T, I, Theta, N, J, rad`), SI unit strings (`kg, m, s, K, N, J, W, Pa, Hz, V, C, Ohm`), or custom domain dimensions (`USD`, `Cells`, `Bits`).
- `rate_identities`: optional `[("s", "t", "v")]` encoding $\mathrm{d}s/\mathrm{d}t = v$ for kinematic/dynamic closure.
- `res.candidates`: ordered `CandidateEquation` objects (`monomial`, `affine`, `hill`, `exp_sat`, etc.).
- `res.pi_groups`: extracted Buckingham-$\Pi$ dimensionless groups when dimensional nullity $> 0$.

### 2.2 Custom Discovery Templates (`TEMPLATE_REGISTRY`)

Register custom candidate functional forms without modifying core solver code:

```python
import sympy as sp
from rosolver import TEMPLATE_REGISTRY, EquationTemplate, list_templates

print(list_templates())  # built-in templates: monomial, affine, hill, exp_sat, ...

class LogisticTemplate(EquationTemplate):
    name = "logistic_pi"
    def build(self, lhs, pi_expr, params_prefix="kappa"):
        k = sp.Symbol(f"{params_prefix}_max", positive=True)
        c = sp.Symbol(f"{params_prefix}_rate", positive=True)
        return sp.Eq(lhs, k / (1 + sp.exp(-c * pi_expr)))

TEMPLATE_REGISTRY.register(LogisticTemplate())
```

### 2.3 CLI Discovery

```bash
rosolver --rel "K = fn(m, v)" --dim K="M*L^2/T^2" --dim m=M --dim v="L/T" \
  --target K --wrt v --zero-initials --latex
```

---

## 3. Procedure 2: Declarative `Model` Specification & Validation (`rosolver.core`)

`Model` decouples physical variables, units, parameters, equations, and initial/boundary conditions from solver backends and persists safely to YAML or JSON (`yaml.safe_load`, AST-validated math).

### 3.1 Authoring a `Model` (YAML or Python)

See [examples/heat_transfer_study/model.yaml](../../examples/heat_transfer_study/model.yaml) for a complete PDE specification, or construct in Python:

```python
from rosolver import ConditionSpec, EquationSpec, Model, ParameterSpec, VariableSpec

ode_model = Model(
    name="cooling_law",
    version="1.0",
    domain="heat_transfer",
    independent_variables=[VariableSpec(name="t", unit="s", role="independent")],
    dependent_variables=[VariableSpec(name="T", unit="K", role="dependent")],
    parameters=[
        ParameterSpec(name="k", unit="1/s", value=0.25, status="fixed"),
        ParameterSpec(name="T_amb", unit="K", value=300.0, status="fixed"),
    ],
    equations=[EquationSpec(id="newton_cool", type="ode", expression="dT/dt = -k * (T - T_amb)")],
    initial_conditions=[ConditionSpec(kind="initial", field="T", value=350.0)],
)
ode_model.save("outputs/cooling_law.yaml")
```

### 3.2 Unified Model Dispatch

```python
loaded = Model.from_file("outputs/cooling_law.yaml")
loaded.validate(strict=True)

# Unified entry points returning StandardResult
sim_res = loaded.solve(t_span=(0.0, 20.0))      # ODE, 1D PDE, or Stefan
# For models with dependencies: loaded.analyse_symbolically(target=..., wrt=...)
# For algebraic/constitutive models: loaded.calibrate(data=..., target=...)
```

CLI validation check:

```bash
rosolver validate outputs/cooling_law.yaml --json
```

---

## 4. Procedure 3: Numerical Solutions (`rosolver.numerics`)

### 4.1 Parametric Curve Sweeps (`simulate`, `simulate_and_plot`)

```python
from rosolver import formulate, simulate, simulate_and_plot

res = formulate(
    relations=["K = fn(m, v)"],
    dims={"K": "M*L^2/T^2", "m": "M", "v": "L/T"},
    target="K",
    wrt="v",
    zero_initials=True,
)
sim = simulate(
    res,
    sweep={"v": (0.0, 30.0, 200)},
    values={"kappa_K": 0.5},
    family={"m": [1.0, 5.0, 10.0]},
)
sim.to_csv("outputs/kinetic_curves.csv")
simulate_and_plot(res, sweep={"v": (0.0, 30.0, 200)}, values={"kappa_K": 0.5},
                  family={"m": [1.0, 5.0, 10.0]}, path="outputs/kinetic.png")
```

### 4.2 ODE Systems & Event Detection (`ODESystem`, `EventSpec`)

```python
from rosolver import EventSpec, ODESystem

fall = ODESystem({"z": "v", "v": "-g"})
traj = fall.solve(
    t_span=(0.0, 5.0),
    y0={"z": 10.0, "v": 0.0},
    params={"g": 9.81},
    events=[EventSpec(name="ground", expr="z", terminal=True, direction=-1)],
)
print("Impact time:", traj.t_events["ground"])
```

Fixed-step Butcher integrators (`euler`, `rk4`, `ssprk3`, `dopri5`, `beuler`, `cn`):

```python
traj_rk4 = fall.solve((0.0, 1.0), {"z": 10.0, "v": 0.0}, params={"g": 9.81}, method="rk4", dt=0.01)
```

### 4.3 1D Conservative FVM PDE Solver (`ConservationLaw1D`)

```python
from rosolver import BC, ConservationLaw1D

heat = ConservationLaw1D(fields="T", flux=0, diffusivity="alpha")
sol = heat.solve(
    domain=(0.0, 1.0),
    n_cells=64,
    t_end=0.5,
    initial="sin(pi * x)",
    bc={"left": BC.dirichlet(0.0), "right": BC.dirichlet(0.0)},
    params={"alpha": 0.05},
    scheme="cn",             # euler | rk4 | ssprk3 | beuler | cn
    reconstruction="linear", # constant | linear | minmod | vanleer | mc
)
print("Midpoint T(0.5):", sol.probe(0.5, "T"))
```

### 4.4 Moving-Boundary Phase Change (`Stefan1D`)

```python
from rosolver import Stefan1D, plot_stefan, stefan_shedding_event

st = Stefan1D(T_wall=-4.13, T_inf=-4.13, T_f=-1.18, h_c=160.0, mdot_imp=3e-3)
sol = st.solve(6 * 3600.0, delta0=1e-5)
print(sol.delta[-1], sol.branch[-1], sol.runback(-1))
plot_stefan(sol, st, path="outputs/stefan.png", show_surface=True)

# Hybrid shedding resets
hyb = st.solve_with_shedding([stefan_shedding_event(delta_crit=0.01, shed_fraction=0.8)], 12 * 3600.0, delta0=1e-5)
```

---

## 5. Procedure 4: Calibration, Datasets & Experimentation (`rosolver.calibration`, `rosolver.io`)

### 5.1 Loading Experimental / External Datasets (`Dataset`)

Supported core formats: **CSV**, **JSON**, **YAML**, **NumPy** (`.npz` with `allow_pickle=False`). Optional extras (`pip install -e ".[io]"`): **Pandas** (`from_dataframe`), **HDF5** (`from_hdf5`).

```python
from rosolver import Dataset

ds = Dataset.from_csv(
    "experiment.csv",
    coordinates=["x"],
    fields=["T"],
    uncertainties={"T": "sigma_T"},
    units={"x": "m", "T": "K"},
    missing_policy="drop",  # "raise" | "drop" | "nan"
)
```

### 5.2 Weighted Calibration, Identifiability & Candidate Ranking

```python
import numpy as np
import sympy as sp
from rosolver import NumericalSolver

x_sym, y_sym, a_sym, b_sym = sp.symbols("x y a b", real=True)
x_data = np.linspace(1.0, 10.0, 20)
sigma = np.full_like(x_data, 0.05)
y_data = 2.0 * x_data**1.5

fit = NumericalSolver.fit(
    sp.Eq(y_sym, a_sym * x_sym**b_sym),
    data={"x": x_data, "y": y_data},
    free_params=[a_sym, b_sym],
    target_col="y",
    sigma_y=sigma,
    bounds={"a": (0.0, 100.0), "b": (0.0, 5.0)},
    loss="soft_l1",  # linear | soft_l1 | huber | cauchy | arctan
)
print(fit.summary())
print("Identifiable:", fit.identifiability.identifiable, "Cond:", fit.identifiability.condition_number)
```

To rank symbolic discovery candidates against data and inspect any rejected forms:

```python
ranking = NumericalSolver.rank_candidates_report(sym_res, data={"x": x_data, "y": y_data}, target_col="y")
print(ranking.summary())
```

### 5.3 Sensitivity & Parameter-Sweep Experimentation

Run systematic design-of-experiments sweeps across operating conditions:

- Use `simulate(..., sweep=..., family={...})` to sweep primary and family parameters across algebraic candidates.
- Use `Stefan1D(...).solve_with_shedding(...)` to experiment with shedding thresholds (`delta_crit`, `shed_fraction`).
- Use `Project.run_simulation(..., params=...)` to log multiple parameter experiments into the append-only ledger and compare them side-by-side in a `ScientificReport`.

---

## 6. Procedure 5: Verification & External Solver Ingestion (`rosolver.verification`, `rosolver.integrations`)

All verification outputs classify outcomes with 5-state `VerificationStatus`: `Passed`, `Failed`, `Inconclusive`, `Warning`, `NotAssessed`.

### 6.1 Code Verification: Method of Manufactured Solutions (`mms_study`)

```python
from rosolver import ConservationLaw1D, mms_study, plot_mms_convergence

burgers = ConservationLaw1D("u", flux="u**2 / 2", diffusivity="nu")
mms = mms_study(
    burgers,
    exact="1.0 + 0.3 * sin(2 * pi * (x - 0.5 * t))",
    cells=(32, 64, 128, 256),
    params={"nu": 0.015},
    reconstruction="linear",
)
print(mms.summary())
plot_mms_convergence(mms, path="outputs/mms.png")
```

### 6.2 Solution Verification: Roache/Celik GCI & Timestep Sensitivity

```python
from rosolver import gci_from_values, grid_convergence_study, timestep_study, plot_gci_study

# Direct 3-grid Celik/Roache GCI
gci = gci_from_values(spacings=[0.01, 0.02, 0.04], quantities=[1.9572, 1.9480, 1.9112], formal_order=2.0)
print(gci.summary())
plot_gci_study(gci, path="outputs/gci.png")
```

### 6.3 Conservation Audits, Stiffness & Coupled Co-Simulation

```python
from rosolver import audit_pde_conservation, audit_ode_invariants, analyze_stiffness, cosimulate_coupled_odes

cons = audit_pde_conservation(sol, tol=1e-10)
print(cons.summary())
```

### 6.4 External Solver Ingestion (`GenericCSVAdapter`, `SciPyAdapter`, `VerificationStudy`)

Ingest results from external CFD/FEM/research solvers without changing V&V algorithms:

```python
from rosolver import VerificationStudy

study = VerificationStudy.from_csv(
    "external_mesh_convergence.csv",
    spacing_col="h",
    qoi_col="drag_coeff",
)
gci_ext = study.gci(formal_order=2.0)
print(gci_ext.summary())
```

CLI external GCI check:

```bash
rosolver --gci "0.01,0.02,0.04:1.9572,1.9480,1.9112" --formal-order 2 --gci-plot outputs/gci.png
```

---

## 7. Procedure 6: Experimental Model Validation (`rosolver.validation`)

Compare numerical predictions ($S \pm u_{\text{num}}$) against experimental measurements ($D \pm u_{\text{exp}}$) and optional input uncertainty ($u_{\text{input}}$):

```python
from rosolver import ValidationStudy, plot_validation

val = ValidationStudy.from_datasets(
    sim_dataset=sim_ds,
    exp_dataset=exp_ds,
    field="T",
    coord="x",
    coverage_factor=1.0,
    study_name="Slab Heat Validation",
)
print(val.summary())
plot_validation(val, path="outputs/validation.png", x_label="x (m)")
```

- Combined validation uncertainty: $u_{\text{val}} = \sqrt{u_{\text{num}}^2 + u_{\text{exp}}^2 + u_{\text{input}}^2}$.
- `Passed` requires all stations $|E_i| \le k u_{\text{val},i}$ with both $u_{\text{num}}$ and $u_{\text{exp}}$ quantified. Missing either emits `Warning` rather than silent pass.

---

## 8. Procedure 7: Persistent `Project` Workspaces, Replay & Scientific Reporting (`rosolver.core`, `rosolver.reporting`)

### 8.1 Reproducible Project Lifecycle & Integrity Verification

```python
from rosolver import Model, Project

proj = Project.create("outputs/study", name="Thermal Study")
model = Model.from_file("examples/heat_transfer_study/model.yaml")
proj.add_model(model)
proj.add_dataset("experiment.csv", name="lab_data", category="experimental")

sim_sr = proj.run_simulation(t_span=(0.0, 0.5), n_cells=64)
proj.verify_integrity(strict=True)  # verifies SHA-256 of models and datasets

# Reopen later from any directory and replay a recorded run
reopened = Project.open("outputs/study")
replayed = reopened.replay_run(reopened.ledger()[-1].id)
```

### 8.2 Generating Scientific Reports (`ScientificReport`, `ReportGenerator`)

Supports 8 canonical report scopes (`REPORT_TYPES`): `symbolic_derivation`, `dimensional_analysis`, `calibration`, `numerical_simulation`, `code_verification`, `solution_verification`, `model_validation`, `integrated_project`.

```python
from rosolver import ReportGenerator

# From an entire project (manifest + ledger + figures)
rep = proj.report(
    output="reports/final_report.md",
    title="Thermal Study Report",
    figures=[{"path": "reports/figures/analytical_comparison.png", "caption": "Model vs analytical"}],
    also_manifest=True,  # writes reports/final_report.manifest.json
)

# Or from any single result object
single_rep = ReportGenerator.from_result(sim_sr, model=model)
single_rep.save("outputs/single_run.yaml")
```

Reports visibly highlight any `Failed`, `Inconclusive`, or `Warning` status, include units, equations, dataset SHA-256 hashes, software metadata, and ASME V&V 20 non-certification framing.

---

## 9. Procedure 8: Workflow CLI & Jupyter Notebooks

### 9.1 Workflow CLI Subcommands

Every subcommand supports `--json` and deterministic CI exit codes (`0` passed, `1` check failed / inconclusive / runtime error, `2` validation / missing artifact / SHA-256 tamper error):

| Command | Purpose | Example |
| --- | --- | --- |
| `init` | Create project & optionally register model | `rosolver init my_proj --model model.yaml` |
| `validate` | Check project SHA-256 integrity & model schema | `rosolver validate my_proj --json` |
| `model analyse` | Run symbolic/dimensional formulation | `rosolver model analyse my_proj --target T --wrt t` |
| `calibrate` | Calibrate parameters from CSV/JSON/YAML | `rosolver calibrate my_proj --data exp.csv --uncertainty-col sigma_y` |
| `simulate` | Solve ODE/PDE/Stefan model | `rosolver simulate my_proj --t-end 0.5 --n-cells 64` |
| `verify` | Run GCI, MMS, or conservation checks | `rosolver verify my_proj --method all --t-end 0.2` |
| `validate-results` | Compare predictions vs experimental data | `rosolver validate-results my_proj --exp-csv exp.csv --field T --u-num 0.01 --u-exp u_T` |
| `report` | Generate Markdown/JSON/YAML scientific report | `rosolver report my_proj --output reports/final_report.md --manifest` |
| `info` | Inspect manifest, counts & integrity status | `rosolver info my_proj --json` |

### 9.2 Jupyter Notebook Rich Display

`StandardResult`, `ScientificReport`, `DerivationResult`, `FitResult`, `GCIReport`, `MMSReport`, `ConservationAuditReport`, and `ValidationReport` all implement pure-string `_repr_markdown_()` hooks (zero IPython/Jupyter dependency) so Jupyter renders formatted Markdown tables automatically.
