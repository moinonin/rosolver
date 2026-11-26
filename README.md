# `rosolver`

**Symbolic Equation Discovery, Numerical Calibration & Verification Suite (Roache GCI / Celik 2008 / MMS).**

`rosolver` generalizes first-principles **Chain-Rule + Dimensional Closure** (originating in [`ch1.tex`](docs/refs/ch1.tex)) into a domain-agnostic scientific modeling and verification engine for students, engineers, and researchers:

1. **Symbolic & Numerical Discovery**: Declare qualitative factor dependencies (`g = fn(a, b, c)`, `b = fn(f; grav)`) and optional dimensions; `rosolver` derives closed-form laws $g(f)$, extracts Buckingham-$\Pi$ groups (monomial, affine, saturating Hill, and Rosin–Rammler templates), enforces Clairaut exactness, calibrates constants/exponents via OLS & analytic-Jacobian NLLS, ranks models by BIC, and auto-simulates/plots curves.
2. **Generic PDE, ODE & Verification Suite (`rosolver.verify`, `rosolver.pde`, `rosolver.odesys`)**: Verify any internal or external numerical model via **Method of Manufactured Solutions (MMS)**, **Roache Grid Convergence Index (GCI)** (including Celik non-uniform refinement iteration), **fixed-step time-step sensitivity**, **symbolic & machine-precision conservation audits**, and **Jacobian stiffness + Aitken $\Delta^2$ partitioned coupling stability checks**. These tools are inspired by ASME V&V 20 principles and do not by themselves constitute formal standards certification.

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

### 2. Run Built-In Suites & Tests
```bash
# Foundational ch1.tex & multi-domain reproductions
python -m examples.ch1_reproductions

# Turbomachinery ice accretion & shedding discovery suite
python -m examples.icing_turbomachinery

# Auto-simulate final equations and save figures to ./figures
python -m examples.auto_simulate

# Generic multi-domain V&V showcase (MMS, GCI, FVM conservation, stiffness, Aitken co-sim)
python -m examples.verification_suite

# Stefan moving-boundary validation + rime/glaze figure
python -m examples.stefan_validation

# Run full test & benchmark suite
pytest -v
```

---

## Core Workflows

### A. Qualitative + Dimensional Equation Formulation & Calibration
```python
from rosolver import formulate, NumericalSolver, simulate_and_plot

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

# Simulate & plot a curve family directly
simulate_and_plot(
    res,
    sweep={"f": (0.5, 25.0, 200)},
    values={"c": 2.0, "grav": 9.81, "kappa_b": 2**-0.5, "kappa_g": 0.5},
    family={"a": [1.0, 500.0, 1000.0]},
    path="figures/waterfall_power.png",
)
```

### B. Grid Independence (Roache GCI) & Automated Symbolic MMS
```python
from rosolver import ConservationLaw1D, mms_study, grid_convergence_study, gci_from_values

# 1. Standalone GCI on any external CFD / FEM / FVM mesh study (uniform or non-uniform r)
gci = gci_from_values(spacings=[0.01, 0.02, 0.04], quantities=[1.9572, 1.9480, 1.9112], formal_order=2.0)
print(gci.summary())

# 2. Automated Method of Manufactured Solutions (SymPy generates exact forcing S_mms)
burgers = ConservationLaw1D("u", flux="u**2 / 2", diffusivity="nu")
mms = mms_study(
    burgers,
    exact="1.0 + 0.3 * sin(2 * pi * (x - 0.5 * t))",
    cells=(32, 64, 128, 256),
    params={"nu": 0.015},
    reconstruction="linear",
)
print(mms.summary())  # Observed L2 order ~ 2.00
```

### C. Conservation Audits & Time-Step Sensitivity
```python
from rosolver import BC, ConservationLaw1D, ODESystem, audit_pde_conservation, audit_ode_invariants, timestep_study

# 1. Machine-precision FVM flux & source ledger audit
plume = ConservationLaw1D(
    fields=["C_A", "C_B"],
    flux={"C_A": "u * C_A", "C_B": "u * C_B"},
    diffusivity={"C_A": "D_A", "C_B": "D_B"},
    source={"C_A": "-k * C_A * C_B", "C_B": "k * C_A * C_B"},
)
sol = plume.solve((0.0, 5.0), 100, t_end=1.0, initial={"C_A": "0.2", "C_B": "0.5"},
                  bc={"left": BC.dirichlet({"C_A": "1.0", "C_B": "0.5"}), "right": "outflow"},
                  params={"u": 1.2, "D_A": 0.04, "D_B": 0.025, "k": 0.8})
print(audit_pde_conservation(sol).summary())  # max|R| ~ 1e-15

# 2. Symbolic + numerical ODE invariant check & fixed-step dt study
seir = ODESystem({"S": "-beta*S*I/N", "E": "beta*S*I/N - sigma*E", "I": "sigma*E - gamma*I", "R": "gamma*I"})
p = {"beta": 0.4, "sigma": 0.2, "gamma": 0.1, "N": 10000.0}
y0 = {"S": 9950.0, "E": 30.0, "I": 20.0, "R": 0.0}
traj = seir.solve((0.0, 60.0), y0, params=p, method="rk4", dt=0.25)
print(audit_ode_invariants(seir, traj, {"total_pop": "S + E + I + R"}, params=p).summary())
print(timestep_study(seir, dts=(1.0, 0.5, 0.25), t_span=(0.0, 60.0), y0=y0, params=p, scheme="rk4", state="I").summary())
```

### D. Stiffness Diagnostics, Aitken $\Delta^2$ Stabilization & Co-Simulation
```python
import numpy as np
from rosolver import ODESystem, analyze_stiffness, solve_partitioned_fixed_point, cosimulate_coupled_odes

# 1. Stabilize a divergent partitioned two-way interface (|G'| = 1.5 > 1) via Aitken Delta^2
fp = solve_partitioned_fixed_point(lambda u: np.array([3.0 - 1.5 * u[0]]), [0.0], acceleration="aitken")
print(fp.summary())

# 2. Loose vs. strong Aitken-iterated co-simulation benchmarked against monolithic ODE
sys_a = ODESystem({"T_f": "-3.0 * (T_f - T_s) + 1.5 * sin(2 * t)"})
sys_b = ODESystem({"T_s": "1.8 * (T_f - T_s) - 0.4 * T_s"})
cs = cosimulate_coupled_odes(sys_a, sys_b, (0.0, 4.0), {"T_f": 1.0, "T_s": 0.0}, dt_macro=0.1, mode="strong")
print(cs.summary())
```

### E. Command-Line Interface
```bash
# Derive, simulate, and plot in one command
rosolver --rel "K = fn(m, v)" --dim K="M*L^2/T^2" --dim m=M --dim v="L/T" \
  --target K --zero-initials --sweep v=0:30:200 --set kappa_K=0.5 --family m=1,5,10 \
  --plot figures/kinetic.png

# Standalone Roache/Celik GCI check on external CFD/FEM mesh study numbers
rosolver --gci "0.01,0.02,0.04:1.9572,1.9480,1.9112" --formal-order 2 --gci-plot figures/gci.png
```

### F. Moving-Boundary (Stefan) Ice Growth
```python
from rosolver import Stefan1D, plot_stefan

# 1. Exact self-similar validation (h_c = 0): delta = 2*lambda*sqrt(alpha*t)
st = Stefan1D(T_wall=-6.0, T_f=0.0, h_c=0.0, n_xi=200)
sol = st.solve(3600.0, delta0=1e-5)
print(sol.delta[-1], float(st.neumann_delta(3600.0)))   # agree to ~1e-7

# 2. Convective surface with SnowTek/WINICE numbers (h_c in the Stefan flux jump)
st = Stefan1D(T_wall=-4.13, T_inf=-4.13, T_f=-1.18, h_c=160.0)
sol = st.solve(86400.0, delta0=1e-5)
y, T = sol.profile()          # physical depth and ice temperature profile
print(sol.summary(), st.quasi_steady_delta(86400.0, 1e-5))
```
`Stefan1D` removes the front with the Landau transform `xi = y/delta(t)`, turns
`delta(t)` into an ODE state, and integrates with stiff `BDF`/`Radau` (or fixed-step
`beuler`/`cn`). The Stefan condition is enforced exactly — no latent-heat smearing.

Rime/glaze coupling: pass `mdot_imp` (impingement flux). The solver then switches between
the supply-limited **dry** branch (`T_s < T_f`, `ddelta/dt = mdot_imp/rho`, no runback) and
the energy-limited **wet** branch (`T_s = T_f`, Stefan rate, excess runs back):

```python
st = Stefan1D(T_wall=-4.13, T_inf=-4.13, T_f=-1.18, h_c=160.0, mdot_imp=3e-3)
sol = st.solve(6 * 3600.0, delta0=1e-5)
print(sol.branch[-1], sol.runback(-1))          # 'wet', kg/m^2/s
plot_stefan(sol, st, path="figures/stefan_front_validation.png", show_surface=True)
```

Runback film + impingement enthalpy: pass `tau_drain` (drainage time), `cp_imp`,
`T_imp`, `V_imp`, `mdot_evap`. The film thickness `h_f` becomes an ODE state and the
wet/dry switch is chosen by energy (`mdot_imp*Lf > q_cap`), so `T_s` stays continuous
as the film drains to zero:

```python
from rosolver import Stefan1D, plot_stefan

st = Stefan1D(T_wall=-4.13, T_inf=-4.13, T_f=-1.18, h_c=160.0,
              mdot_imp=6e-3, cp_imp=4186.0, V_imp=30.0, tau_drain=30.0)
sol = st.solve(2 * 3600.0, delta0=1e-5)
print(sol.film[-1], sol.freezing_fraction(), sol.mdot_out[-1])
plot_stefan(sol, st, show_film=True, path="figures/stefan_film.png")
```

Hybrid accretion/shedding: reuse `SheddingEventSpec` on the front position. Events fire
when `delta` crosses `delta_crit`, reset the ice thickness, and restart the integration:

```python
from rosolver import Stefan1D, stefan_shedding_event

st = Stefan1D(T_wall=-4.13, T_inf=-4.13, T_f=-1.18, h_c=160.0, mdot_imp=3e-3)
hyb = st.solve_with_shedding(
    [stefan_shedding_event(delta_crit=0.01, shed_fraction=0.8)],
    12 * 3600.0, delta0=1e-5,
)
print(hyb.summary())                            # lists every shedding event
```

See [docs/notes/stefan_front_tracking_notes.pdf](docs/notes/stefan_front_tracking_notes.pdf) for the full derivation.

---

## Repository Structure

```text
rosolver/
├── rosolver/
│   ├── dimensions.py       # Exact rational dimensional algebra & Buckingham-Pi solver
│   ├── system.py           # Dependency DAG builder, string DSL, cycle & path checks
│   ├── symbolic.py         # Chain-rule expansion, Clairaut coupling, symbolic integration
│   ├── numerical.py        # OLS/NLLS calibration, BIC ranking, sparse Pi, hybrid ODEs
│   ├── integrators.py      # Fixed-step RK/implicit integrators & stability functions R(z)
│   ├── odesys.py           # Symbolic multi-state ODE systems, Jacobians & Lie invariants
│   ├── pde.py              # 1D conservative FVM balance-law solver with flux/source ledger
│   ├── phasechange.py      # Front-fixing Stefan solver: wet/dry, runback film, shedding
│   ├── verify.py           # GCI, MMS, dt sensitivity, conservation audits & Aitken coupling
│   ├── simulate.py         # Parameter sweeps, curve families & V&V plotting helpers
│   └── cli.py              # CLI entry point (`rosolver` / `python -m rosolver`)
├── examples/
│   ├── ch1_reproductions.py        # ch1.tex & multi-domain worked derivations
│   ├── icing_turbomachinery.py     # 8 turbomachinery ice accretion & shedding cases
│   ├── stefan_validation.py        # Stefan front vs Neumann + rime/glaze figure
│   ├── auto_simulate.py            # Auto-simulation & plotting showcase
│   └── verification_suite.py       # Generic multi-domain V&V & coupling stability showcase
├── tests/                  # Unit, integration, security & scientific benchmark tests
└── docs/                   # Architecture specs, LaTeX lecture notes & reference material
```

## License

MIT
