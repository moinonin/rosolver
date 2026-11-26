# Scientific Reference: Mathematical, Numerical & V&V Methods

All core algorithms in Rosolver follow the 8-item engineering documentation standard (`ENG_SPEC.md` `P4.3`). Diagnostics support engineering traceability and do not by themselves constitute formal regulatory certification.

---

## 1. Qualitative Chain-Rule & Buckingham-$\Pi$ Formulation (`rosolver.symbolic`)

- **Mathematical Definition:** Given an acyclic dependency DAG $y = f(x_1, \dots, x_m; c_k)$ with declared base dimensions $[x_i] = \prod_d B_d^{a_{di}}$, finds exact rational exponent solutions $M \beta = \mathbf{b}$ and nullspace $\Pi$-groups ($\ker M$), enforces Clairaut mixed-partial symmetry ($\partial_{x_j}\beta_i = \partial_{x_i}\beta_j$), and constructs closed-form candidates by Poincaré line integration or registered `TEMPLATE_REGISTRY` forms.
- **Assumptions:** Dependency relations form a DAG (no directed cycles); governing dimensional exponents are rational and exact.
- **Applicable Conditions:** Dimensional closure applies when all participating variables/constants carry consistent dimensions; dimension-free calculus mode applies when dimensions are omitted.
- **Input Requirements:** Non-empty acyclic relation list and valid target variable (plus optional dimension map, governing constants, and rate identities).
- **Output Interpretation:** `DerivationResult` holds ordered `DerivationStep` objects, `pi_groups`, `candidates` (primary + alternatives), `diagnostics`, and `rejected_candidates`.
- **Known Limitations:** Cannot uniquely determine dimensionless shape functions when nullity $\ge 1$ without candidate templates and empirical calibration.
- **References:** Buckingham, E. (1914), *Phys. Rev.* 4(4); [Chain-Rule Formulation Notes](../../docs/notes/eqform_class_notes.pdf); [Symbolic Derivation Implementation](../../rosolver/symbolic/derivation.py).
- **Linked Verification Tests:** [tests/benchmarks/test_benchmark_dimensional.py](../../tests/benchmarks/test_benchmark_dimensional.py), [tests/test_rosolver.py](../../tests/test_rosolver.py), [tests/test_icing_discovery.py](../../tests/test_icing_discovery.py).

---

## 2. Parameter Estimation, WLS & Identifiability Diagnostics (`rosolver.calibration`)

- **Mathematical Definition:** Minimizes weighted least squares $J(\theta) = \sum_{i=1}^n w_i (y_i - \hat{y}(x_i; \theta))^2$ with $w_i = 1/\sigma_{y,i}^2$ (when $\sigma_y$ supplied) using SVD linear least squares (unbounded linear models) or analytic-Jacobian Trust-Region nonlinear least squares (`scipy.optimize.least_squares`) with optional bounds and robust losses (`linear`, `huber`, `soft_l1`, `cauchy`, `arctan`). Parameter covariance estimates use $s^2 (J_w^\top J_w)^{-1}$; scale-invariant correlation and condition number come from singular values of column-scaled $J_w$.
- **Assumptions:** Measurement errors have zero mean; weights reflect inverse variance when WLS uncertainty weighting is enabled; model is twice continuously differentiable in $\theta$.
- **Applicable Conditions:** Require $n > p$ observations for finite residual degrees of freedom and full-rank Jacobian for unambiguous parameter identification.
- **Input Requirements:** Candidate or `Model` equation, finite arrays after `missing_policy` (`raise`, `drop`, `mask`), optional `sigma_y`/`weights` and bounds.
- **Output Interpretation:** `FitResult` separates fit metrics (`r2`, `rmse`, `bic`, `aic`), parameter estimates/std errors, `IdentifiabilityReport` (`condition_number`, `rank`, `identifiable`, correlation matrix), and `admissibility_tier`.
- **Known Limitations:** Asymptotic linearized covariance breaks down near active bounds or under strong nonlinearity/collinearity (flagged explicitly via `identifiable=False`).
- **References:** Bates & Watts (1988), *Nonlinear Regression Analysis*; Seber & Wild (2003); [rosolver/numerical.py](../../rosolver/numerical.py).
- **Linked Verification Tests:** [tests/benchmarks/test_benchmark_calibration.py](../../tests/benchmarks/test_benchmark_calibration.py), [tests/test_sprint2_calibration_templates.py](../../tests/test_sprint2_calibration_templates.py).

---

## 3. Conservative 1D FVM PDE, ODE & Moving-Boundary Stefan Solvers (`rosolver.numerics`)

- **Mathematical Definition:** Discretizes 1D conservation laws $\partial_t u + \partial_x f(u) = \partial_x(D \partial_x u) + s(x, t, u)$ over uniform finite-volume cells $I_j$ with Rusanov/central interface fluxes, slope reconstruction (`constant`, `linear`, `minmod`, `vanleer`, `superbee`), and explicit (`euler`, `rk4`, `ssprk3`) or implicit (`beuler`, `cn`) time integration. `ODESystem` integrates stiff/non-stiff ODEs with event root-finding (`EventSpec`). `Stefan1D` applies the Landau front-fixing coordinate $\xi = y/\delta(t)$ to track exact latent-heat phase boundaries.
- **Assumptions:** Smooth solutions away from shocks; bounded diffusion/CFL stability on explicit schemes; uniform 1D domain grid.
- **Applicable Conditions:** 1D initial-boundary value problems and ODE initial-value problems with smooth parameters; Stefan problems with positive Stefan number (`T_f != T_wall`).
- **Input Requirements:** Well-posed governing equations, initial values, boundary conditions (`dirichlet`, `neumann`, `periodic`, `outflow`), and positive physical parameters.
- **Output Interpretation:** `PDESolution`, `Trajectory`, and `StefanSolution` store time/space arrays alongside solver `success`, `newton_failures`, CFL metrics, and FVM `budget` fluxes; wrapped cleanly via `standardize_result`.
- **Known Limitations:** Restricted to 1D spatial domains (`ConservationLaw1D`, `Stefan1D`) and deterministic differential equations.
- **References:** LeVeque (2002), *Finite Volume Methods for Hyperbolic Problems*; Carslaw & Jaeger (1959); [Stefan Notes](../../docs/notes/stefan_front_tracking_notes.pdf).
- **Linked Verification Tests:** [tests/benchmarks/test_benchmark_heat_conduction.py](../../tests/benchmarks/test_benchmark_heat_conduction.py), [tests/benchmarks/test_benchmark_ode.py](../../tests/benchmarks/test_benchmark_ode.py), [tests/test_phasechange.py](../../tests/test_phasechange.py).

---

## 4. Method of Manufactured Solutions (MMS) Code Verification (`rosolver.verification`)

- **Mathematical Definition:** Given a user-supplied smooth manufactured solution $u_{\text{mms}}(x, t)$, `mms_study` computes exact symbolic PDE residual forcing $S_{\text{mms}} = \partial_t u_{\text{mms}} + \partial_x f(u_{\text{mms}}) - \partial_x(D \partial_x u_{\text{mms}}) - s(u_{\text{mms}})$ in SymPy, solves the forced system across $M \ge 3$ meshes, measures discrete $L_1$, $L_2$, and $L_\infty$ cell-average errors, and regresses observed spatial order $p_{\text{obs}}$ via log-log slope.
- **Assumptions:** Time-step error is kept small relative to spatial discretization error (small `cfl` or resolved `dt`); manufactured field is smooth ($C^\infty$).
- **Applicable Conditions:** Code verification of differential operators and boundary implementations on `ConservationLaw1D`.
- **Input Requirements:** PDE model, symbolic manufactured expression, $\ge 3$ grid counts, and consistent boundary kind (`periodic` or `dirichlet`).
- **Output Interpretation:** `MMSReport` reports `order_l1`, `order_l2`, `order_linf`, `formal_order`, and classifies `VerificationStatus.PASSED` when $p_{L_2} \ge p_{\text{formal}} - \text{tol}$ (relaxing automatically with warning for TVD slope limiters).
- **Known Limitations:** Verifies numerical discretization accuracy of the implemented operators, not physical model fidelity against experiments.
- **References:** Roache (2002), *J. Fluids Eng.* 124(1); AIAA S-141-2024; [Verification Notes](../../docs/notes/verification_and_stability_notes.pdf).
- **Linked Verification Tests:** [tests/benchmarks/test_benchmark_mms.py](../../tests/benchmarks/test_benchmark_mms.py), [tests/test_verification.py](../../tests/test_verification.py).

---

## 5. Roache & Celik Grid / Timestep Convergence Index (`GCI`) (`rosolver.verification`)

- **Mathematical Definition:** On three ordered discretizations ($h_1 < h_2 < h_3$, refinement ratios $r_{21} = h_2/h_1$, $r_{32} = h_3/h_2$), computes differences $\varepsilon_{32} = \phi_3 - \phi_2$ and $\varepsilon_{21} = \phi_2 - \phi_1$, solves Celik's fixed-point equation $p = |\ln|\varepsilon_{32}/\varepsilon_{21}| + q(p)| / \ln r_{21}$, Richardson extrapolated limit $\phi_{\text{ext}} = (r_{21}^p \phi_1 - \phi_2)/(r_{21}^p - 1)$, and $\text{GCI}_{\text{fine}}^{21} = F_s |\varepsilon_{21}/\phi_1| / (r_{21}^p - 1)$ with $F_s = 1.25$.
- **Assumptions:** Grids lie within the asymptotic convergence range; functional QoI is sufficiently smooth across all three resolutions.
- **Applicable Conditions:** Monotonic convergence ($\varepsilon_{32}\varepsilon_{21} > 0$, $|\varepsilon_{21}| < |\varepsilon_{32}|$). Divergent, oscillatory, exact-zero-difference, or poorly separated sequences ($r < 1.05$) are explicitly detected.
- **Input Requirements:** At least 3 distinct positive grid spacings or timesteps and scalar quantities of interest (internal solver or external CSV via `VerificationStudy`).
- **Output Interpretation:** `GCIReport` outputs `observed_order`, `extrapolated_value`, `gci_fine_21`, `asymptotic_ratio`, `regime`, and `VerificationStatus` (`PASSED`, `FAILED`, `INCONCLUSIVE`, or `WARNING`).
- **Known Limitations:** Oscillatory convergence or pre-asymptotic meshes make Richardson extrapolation unreliable (flagged as `INCONCLUSIVE` or `WARNING`).
- **References:** Roache (1998), *Verification and Validation in Computational Science*; Celik et al. (2008), *ASME J. Fluids Eng.* 130(7); ASME V&V 20-2009.
- **Linked Verification Tests:** [tests/benchmarks/test_benchmark_external_gci.py](../../tests/benchmarks/test_benchmark_external_gci.py), [tests/test_verification.py](../../tests/test_verification.py).

---

## 6. Discrete Conservation & Invariant Audits (`rosolver.verification`)

- **Mathematical Definition:** Audits the time-integrated discrete FVM budget residual $R_k(t) = [I_k(t) - I_k(0)] - [\mathcal{F}_{\text{in}, k}(t) - \mathcal{F}_{\text{out}, k}(t) + \mathcal{S}_k(t)]$ across every field $k$ and timestep (`audit_pde_conservation`), symbolic Lie/algebraic ODE invariants (`audit_ode_invariants`), or piecewise hybrid jump balances (`audit_hybrid_conservation`).
- **Assumptions:** Boundary flux and source terms integrated identically to cell updates; linear/conservative discretizations conserve machine-precision balances when exact.
- **Applicable Conditions:** Closed/open 1D conservation laws, Hamiltonian or conserved ODEs, and event-reset hybrid models.
- **Input Requirements:** Completed solver trajectory (`PDESolution`, `Trajectory`, `HybridStefanSolution`) plus tolerance `tol`.
- **Output Interpretation:** `ConservationAuditReport` details `verification_scope` (internal flux-source accounting vs. exact analytical conservation), `max_abs_residual`, `max_rel_residual`, and propagates underlying `newton_failures` / solver failure as `FAILED`.
- **Known Limitations:** Closure of the discrete flux-source ledger verifies internal conservation accounting of the solver rather than physical correctness of boundary closures.
- **References:** LeVeque (2002); [rosolver/verify.py](../../rosolver/verify.py).
- **Linked Verification Tests:** [tests/test_verification.py](../../tests/test_verification.py), [tests/benchmarks/test_benchmark_ode.py](../../tests/benchmarks/test_benchmark_ode.py).

---

## 7. Experimental Model Validation (`rosolver.validation`)

- **Mathematical Definition:** Computes pointwise comparison error $E_i = S_i - D_i$, summary metrics (RMSE, MAE, max $|E|$, mean bias), and combined validation standard uncertainty $u_{\text{val}, i} = \sqrt{u_{\text{num}, i}^2 + u_{\text{exp}, i}^2 + u_{\text{input}, i}^2}$, checking whether $|E_i| \le k \cdot u_{\text{val}, i}$ for coverage factor $k$.
- **Assumptions:** Simulation numerical, experimental, and input uncertainties are independent; when coordinates differ, smooth linear interpolation onto experimental stations introduces negligible interpolation error.
- **Applicable Conditions:** Quantitative validation of deterministic predictions against experimental or reference datasets with compatible physical units.
- **Input Requirements:** Equal-dimension simulation and experimental `Dataset` objects (or arrays), field name, and ideally both $u_{\text{num}}$ and $u_{\text{exp}}$.
- **Output Interpretation:** `ValidationReport` classifies `PASSED` (all stations within $\pm k u_{\text{val}}$ and both $u_{\text{num}}, u_{\text{exp}}$ provided), `WARNING` (agreement without complete $u_{\text{num}}$/$u_{\text{exp}}$ quantification), or `FAILED` (error exceeds tolerance or $k u_{\text{val}}$).
- **Known Limitations:** Assumes normally distributed independent uncertainty sources; does not constitute formal regulatory certification.
- **References:** ASME V&V 20-2009; AIAA S-141-2024; [rosolver/validation/experimental.py](../../rosolver/validation/experimental.py).
- **Linked Verification Tests:** [tests/test_sprint5_data_adapters_validation.py](../../tests/test_sprint5_data_adapters_validation.py), [tests/test_sprint6_reporting_cli.py](../../tests/test_sprint6_reporting_cli.py).
