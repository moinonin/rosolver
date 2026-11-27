# Scientific Claim Verification & Peer-Review Rebuttal Guide (`rosolver.claims`)

`rosolver.claims` (`Sprint 10` / `P5.4`) provides deterministic verification of mathematically intensive scientific claims, order-of-magnitude scale-separation auditing, discretization-vs-claimed-error budget analysis, and a stateful `R1 -> R2` peer-review clarification and author rebuttal engine.

> **Epistemic Principle:** *AI proposes claim mappings; Rosolver deterministically evaluates dimensional, scale-balance, numerical, and empirical support; unresolved gaps generate quantitative referee clarification queries.*

---

## 1. Mathematical Definitions & Methods

1. **Order-of-Magnitude Scale Balance (`check_equation_scales`)**:  
   Given an equation $T_1 + T_2 + \dots + T_m = 0$ over operating envelope $\Omega = \prod_i [x_{i,\min}, x_{i,\max}]$, each term's magnitude interval $[|T_j|_{\min}, |T_j|_{\max}]$ and geometric typical scale $|T_j|_{\mathrm{typ}} = \sqrt{|T_j|_{\min}|T_j|_{\max}}$ are evaluated deterministically. If the leading two terms are separated by an interval gap exceeding $\tau_{\mathrm{scale}}$ (default $10^2\times$), the law is classified `Failed` (`unbalanced_dominant_terms`).
2. **Implicit Normalization & Differential Diagnostics (`diagnose_normalization_shortcuts`)**:  
   Traverses AST sum nodes `A + B` to identify dimensionless literals added to dimensional quantities (e.g., `1 - rho * V`), computing the exact missing scale dimension $\dimbr{B}/\dimbr{A}$, and detects differential-order mismatches (e.g., `sigma = A * dz`).
3. **Euler Homogeneity Elasticity Sum Rule (`check_homogeneity_elasticities`)**:  
   For any zero-intercept dimensionally homogeneous law $y = f(x_1,\dots,x_m)$, logarithmic elasticities $\mathbf{E} = [(x_i/y)\partial y/\partial x_i]$ must satisfy $D\,\mathbf{E} = \mathbf{d}_y$. Non-zero residual $\|D\mathbf{E} - \mathbf{d}_y\|_\infty > \mathrm{tol}$ proves a non-zero baseline offset $y_0$ or omitted dimensional scale is required.
4. **Discretization vs. Claimed-Error Budget (`audit_claimed_accuracy`)**:  
   Evaluates $R_{\mathrm{disc}} = u_{\mathrm{num}}(\Delta t_{\mathrm{paper}}) / \varepsilon_{\mathrm{claimed}}$. When the manuscript omits the time integrator, tests candidate schemes (`euler`, `cn`, `rk4`) against a high-order reference (`DOP853`) and flags `SchemeDependent` when worst-case discretization error consumes $\ge 25\%$ of the claimed error budget.
5. **Two-Axis Verdict Synthesis (`TwoAxisAssessment`, `synthesize_verdict`)**:  
   Separates `mechanistic_status` from `empirical_status`. Mechanistic plausibility (`Passed`) with missing lab data (`NotAssessed`) yields `Inconclusive` and emits a deterministic `ClarificationRequest` specifying exact required CSV columns, replicate counts, and uncertainty columns (`u_<obs>`).

---

## 2. Assumptions, Applicable Conditions & Input Requirements

- **Assumptions**: Smooth real-valued expressions representable in Rosolver's AST whitelist (`rosolver._parsing.parse_expr`); operating intervals reflect physical bounds; experimental rebuttals supply standard uncertainties $u_{\mathrm{exp}}$.
- **Applicable Conditions**: Algebraic balance laws, ODE/PDE/Stefan models, scaling/allometry claims, numerical discretization claims, and replicate laboratory assay datasets.
- **Input Requirements**: Validated `ClaimManifest` YAML/JSON, optional `Model` YAML registered in a `Project`, and confined project-relative rebuttal CSV paths (`..` traversal rejected).

---

## 3. Output Interpretation & Known Limitations

- **Verdicts**: `Supported`, `Falsified`, `PartiallySupported`, `SchemeDependent`, `Inconclusive`, `NotAssessed`.
- **Anti-Override Invariant**: Every `Supported` verdict must reference verified non-failing `ExecutionRecord` run IDs in `Project.ledger()`.
- **Anti-P-Hacking Flag**: Rebuttals that modify model parameters (`modifies_model_parameters=True`) are flagged with `post_hoc_recalibration=True` and downgraded from `Passed` to `Warning` until blindly validated.
- **Limitations**: Audits mathematical, dimensional, numerical, and statistical consistency against supplied evidence; does not independently authenticate raw wet-lab specimens or constitute formal regulatory certification.

---

## 4. Quickstart (CLI & MCP)

```bash
# CLI workflow
rosolver init my_review --model examples/pond_freezing_audit_study/model.yaml
rosolver claim init my_review --manifest claims.yaml
rosolver claim audit my_review --output reports/referee_r1.md --json
rosolver claim questions my_review --severity major
rosolver claim respond my_review --rebuttal rebuttal_r2.yaml --output reports/referee_r2.md --json
```

**MCP Tools (`rosolver-mcp`)**: `rosolver_check_scales`, `rosolver_audit_claims`, `rosolver_submit_rebuttal`, resource `rosolver://schema/claims`, prompt `peer_review_manuscript`.

---

## 5. References & Linked Verification Tests

- **References**: ASME V&V 20-2009; Celik et al. (2008), *ASME J. Fluids Eng.* 130; Roache (1998), *Verification and Validation in Computational Science and Engineering*; Buckingham (1914).
- **Linked Tests**: [`tests/test_sprint10_claim_verification.py`](../../tests/test_sprint10_claim_verification.py) (Benchmarks `S10-B1`–`S10-B3`, security/anti-override suite, CLI & MCP integration tests).
