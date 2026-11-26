# Tutorial: 1D Transient Heat Conduction End-to-End Study

This tutorial walks through [examples/heat_transfer_study/run_study.py](../../examples/heat_transfer_study/run_study.py) and its declarative model [examples/heat_transfer_study/model.yaml](../../examples/heat_transfer_study/model.yaml).

## Governing Problem

$$\frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial x^2}, \qquad x \in [0, 1]\,\text{m}, \quad \alpha = 0.05\,\text{m}^2/\text{s}$$

Initial condition $T(x, 0) = \sin(\pi x)\,\text{K}$ and homogeneous Dirichlet boundaries $T(0, t) = T(1, t) = 0\,\text{K}$ yield the exact analytical mode-1 Fourier solution:

$$T_{\text{exact}}(x, t) = e^{-\alpha \pi^2 t} \sin(\pi x)$$

## Lifecycle Executed in Order

1. **Model Specification & Validation:** `Model.from_file` loads `model.yaml` via `yaml.safe_load` and checks acyclic DAGs, parameters, equations, and conditions; `Project.create` + `project.add_model` records SHA-256 checksums.
2. **Dimensional Consistency Check:** Verifies $[\partial T / \partial t] = [\alpha \partial^2 T / \partial x^2] = \Theta T^{-1}$ and $[k / (\rho c_p)] = L^2 T^{-1}$.
3. **Symbolic & Buckingham-$\Pi$ Formulation:** `project.run_symbolic(target="T", wrt="t", zero_initials=True)` extracts the Fourier number dimensionless group $\Pi = \alpha t / x^2$.
4. **Numerical FVM PDE Solve:** `project.run_simulation(t_span=(0.0, 0.5), n_cells=64, method="cn")` solves using Crank-Nicolson time integration and linear spatial reconstruction.
5. **Spatial Grid (`GCI`) & Timestep Convergence:** `grid_convergence_study` over $N \in (32, 64, 128)$ and `timestep_study` over $\Delta t \in (0.1, 0.05, 0.025)$ confirm 2nd-order convergence and quantify numerical uncertainty `gci_fine_21`.
6. **Analytical Fourier Comparison (`ValidationStudy`):** Compares FVM nodal values against `T_exact` using `u_num` derived from `gci_fine_21` and reference tolerance `u_exp = 1e-4 K`.
7. **MMS Code Verification:** `mms_study` derives exact SymPy source $S_{\text{mms}}$ for $T_{\text{mms}}(x, t) = e^{-t}\sin(\pi x)$ across $N \in (16, 32, 64)$ and verifies spatial $L_2$ order $\approx 2.0$.
8. **Discrete Conservation Audit:** `audit_pde_conservation(sol, tol=1e-10)` checks flux/storage closure residuals across every timestep.
9. **Scientific Report:** `project.report(output="reports/final_report.md", ..., also_manifest=True)` writes the deterministic Markdown report, relative links to all 4 PNG plots under `reports/figures/`, and JSON manifest.

## Running Headlessly

```bash
python -m examples.heat_transfer_study.run_study --output outputs/heat_transfer_study
```
