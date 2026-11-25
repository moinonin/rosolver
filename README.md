# `rosolver`

**Symbolic & Numerical Equation Formulator from Qualitative Dependencies and Dimensional Analysis.**

`rosolver` generalizes first-principles **Chain-Rule + Dimensional Closure** (originating in [`ch1.tex`](ch1.tex)) into a domain-agnostic equation discovery engine for students, engineers, and researchers. Even when no governing equation is known, declare qualitative factor dependencies (e.g., $g = \Phi(a, b, c)$ and $b = \psi(f)$) with optional physical or custom domain dimensions, and `rosolver` derives candidate symbolic laws $g(f)$, extracts Buckingham-$\Pi$ dimensionless groups, enforces Clairaut exactness, calibrates constants and exponents from data, and integrates hybrid event-driven ODEs.

---

## Key Capabilities

* **Qualitative Dependency DAGs**: Declare arbitrary multi-hop graphs (`g = fn(a, b, c)`, `b = fn(f; grav)`) with automatic cycle detection, active-path chain-rule pruning, and bottom-up composition.
* **Exact Rational Dimensional Engine**: Supports SI dimensions (`M, L, T, I, Theta, N, J, rad`), unit aliases (`kg, m, s`), fractional exponents (e.g., fracture toughness `M/(L^(1/2)*T^2)`), and custom non-physics dimensions (`USD`, `Users`, `Month`).
* **Automated Buckingham-$\Pi$ Discovery**: Extracts coprime integer dimensionless groups from $\ker(D)$ (e.g., Reynolds number $Re$, Stokes number $K$) and generates power-law, affine, and bounded $[0, 1]$ saturating (**Hill** and **Rosin–Rammler**) candidate laws.
* **Clairaut Exactness & Poincaré Integration**: Automatically couples partial-derivative constants ($\partial \beta_i/\partial x_j = \partial \beta_j/\partial x_i$) and reconstructs closed-form state potentials.
* **Numerical Calibration & Model Selection**: Exact SVD OLS for parameter-linear laws, analytic-Jacobian Trust-Region NLLS for $\Pi$-exponents, BIC candidate ranking, sparse $\Pi$-subset pruning (`select_sparse_pi`), and local dimensionless elasticities.
* **Hybrid Event-Driven ODE Solver**: Integrates smooth accretion/growth ODEs coupled with competing discontinuous threshold triggers (e.g., interfacial debonding vs. cohesive bulk fracture in turbomachinery ice shedding).
* **LaTeX & Pedagogical Traces**: Export step-by-step derivations directly to `.tex` via `result.to_latex()`.

---

## Quickstart

### 1. Installation
```bash
git clone https://github.com/moinonin/rosolver.git
cd rosolver
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Run Built-In Examples & Tests
```bash
# Foundational ch1.tex & multi-domain reproductions
python -m examples.ch1_reproductions

# Turbomachinery ice accretion & shedding discovery suite
python -m examples.icing_turbomachinery

# Run full test suite (26 tests)
pytest -v

# Auto-simulate final equations and save figures to ./figures
python -m examples.auto_simulate
```

### 3. Minimal Python Example (Dimension-Free & Dimensional)
```python
from rosolver import formulate, NumericalSolver

# A) Pure qualitative inference (no dimensions required)
res_qual = formulate(
    relations=["g = fn(a, b, c)", "b = fn(f)"],
    target="g",
    wrt="f",
)
print(res_qual.summary())

# B) Dimensional closure + Clairaut coupling + integration
res_dim = formulate(
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
print(res_dim.summary())
print(res_dim.to_latex())
```

### 4. Buckingham-$\Pi$ Saturation, Sparse Selection & Numerical Calibration
```python
from rosolver import formulate, NumericalSolver

# Discover Stokes number K = tau_p * v_rel / c and bounded collection efficiency
res = formulate(
    relations=["m_imp = fn(v_rel; IWC, tau_p, c)"],
    dims={"m_imp": "M/(L^2*T)", "v_rel": "L/T", "IWC": "M/L^3", "tau_p": "T", "c": "L"},
    constants=["IWC", "tau_p", "c"],
    target="m_imp",
    wrt="v_rel",
    zero_initials=True,
    include_saturating=True,
)

# Rank monomial, power-law, affine, Hill, and Rosin-Rammler candidates by BIC
ranked = NumericalSolver.rank_candidates(res, data=dataset_dict, criterion="bic")
print(ranked[0].summary())
```

### 5. Hybrid Event-Driven Accretion–Shedding ODEs
```python
import numpy as np
import sympy as sp
from rosolver import NumericalSolver, SheddingEventSpec

t, delta = sp.symbols("t delta", real=True)
rate_eq = sp.Eq(sp.Derivative(delta, t), sp.Float(1.0e-4))

sol = NumericalSolver.integrate_hybrid_ode(
    rate_eq,
    wrt="t",
    span=(0.0, 120.0),
    y0=0.0,
    events=[
        SheddingEventSpec("cohesive_fracture", lambda tv, dv: 2e7 * dv**3 - 1.28, shed_fraction=0.42),
        SheddingEventSpec("interfacial_adhesive", lambda tv, dv: 2e7 * dv**3 - 30.0 * np.exp(-0.008 * tv), shed_fraction=0.95),
    ],
    target_name="delta",
)
for ev in sol.events:
    print(f"t={ev.time:.2f}s | {ev.mode} | {ev.pre_value*1e3:.2f}mm -> {ev.post_value*1e3:.2f}mm")
```

### 6. Auto-Simulate & Plot a Final Equation
```python
from rosolver import formulate, simulate, plot_simulation

res = formulate(["K = fn(m, v)"], dims={"K": "M*L^2/T^2", "m": "M", "v": "L/T"},
                target="K", zero_initials=True)

# Sweep one variable, fix the rest, draw a family of curves over another
sim = simulate(res, sweep={"v": (0, 30, 200)},
               values={"kappa_K": 0.5},        # calibrated constant
               family={"m": [1, 5, 10]})
plot_simulation(sim, path="figures/kinetic_energy.png")   # also: sim.to_csv("curves.csv")
```
Also available: `plot_ranked(...)` (fitted candidates vs. data), `plot_hybrid(...)` (accretion/shedding
sawtooth with event markers), and `simulate_and_plot(...)` as a one-call shortcut. Unspecified free
constants default to `1.0` (offsets `*_0` to `0`) and are listed in `sim.assumed` and on the plot.
Pass a `FitResult` (e.g. `NumericalSolver.fit(...)`) as the model to simulate the *calibrated* equation.

Run the demo: `python -m examples.auto_simulate` (writes PNGs to `figures/`).

### 7. Command-Line Interface
```bash
# Installed console script or module invocation
rosolver --rel "g = fn(a, b, c)" --rel "b = fn(f)" --target g --wrt f
python -m rosolver --rel "g = fn(a, b, c)" --rel "b = fn(f)" --target g --wrt f --latex

# Derive, then simulate and plot the final equation in one command
rosolver --rel "K = fn(m, v)" --dim K="M*L^2/T^2" --dim m=M --dim v="L/T" \
  --target K --zero-initials --sweep v=0:30:200 --set kappa_K=0.5 --family m=1,5,10 \
  --plot figures/kinetic.png --sim-csv figures/kinetic.csv
```

---

## Repository Structure

```text
rosolver/
├── rosolver/               # Core library package
│   ├── dimensions.py       # Exact rational dimensional algebra & Buckingham-Pi solver
│   ├── system.py           # Dependency DAG builder, string DSL, cycle & path checks
│   ├── symbolic.py         # Chain-rule expansion, Clairaut coupling, symbolic integration
│   ├── numerical.py        # OLS/NLLS calibration, BIC ranking, sparse Pi, hybrid ODEs
│   ├── simulate.py         # Auto-simulation sweeps/families & matplotlib plotting
│   └── cli.py              # CLI entry point (`rosolver` / `python -m rosolver`)
├── examples/
│   ├── ch1_reproductions.py        # ch1.tex & multi-domain worked derivations
│   └── icing_turbomachinery.py     # 6 turbomachinery ice accretion & shedding cases
├── tests/
│   ├── test_rosolver.py            # Core symbolic, dimensional, numerical & CLI tests
│   └── test_icing_discovery.py     # Saturating Pi, sparse selection & hybrid ODE tests
├── docs/
│   ├── eqform_class_notes.pdf              # Lecture notes on generalized methodology
│   └── icing_accretion_shedding_notes.pdf  # Turbomachinery icing & shedding notes
├── ch1.tex                 # Original foundational chapter
└── pyproject.toml          # Packaging metadata & dependencies
```

## License

MIT
