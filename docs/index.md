# Rosolver Documentation (`v0.2.0`)

Rosolver is an open-source, reproducible scientific modeling, calibration, numerical simulation, verification & validation (V&V), and scientific reporting toolkit for Python and the command line.

> **Scope & Standards Note:** Rosolver's verification and validation diagnostics (MMS, Roache/Celik GCI, discrete conservation audits, and experimental comparison bands) follow ASME V&V 20-2009, AIAA S-141-2024, Roache, and Celik et al. (2008) methodologies for transparent traceability and do not by themselves constitute formal regulatory or standards certification.

---

## What's New in `v0.2.0` & Latest Improvements

- **PySR / Symbolic-Regression Candidate Importer (Sprint 9 Slim)**: Deterministic 4-stage Pareto-front gatekeeper (`CandidateImporter.from_pysr_csv`, `from_pysr_model`, `from_expressions`) enforcing AST parsing (`P0.5`) $\to$ dimensional/normalization audit $\to$ Sprint 10 scale-separation audit $\to$ WLS/identifiability BIC ranking, with direct `to_claim_manifest()` handoff and optional `rosolver[pysr]`. See [Symbolic-Regression Importer Guide](user_guide/symbolic_regression_importer.md).
- **External CFD/FEM Audit Adapters & Headless LaTeX/PDF Reporting (Sprint 8 Slim)**: Native zero-dependency `OpenFOAMAdapter` (`postProcessing/` probes, `residuals.dat`, `controlDict`), `FEniCSxAdapter` (VTK-CSV, ASCII `.vtk`/`.vtu`, inline `.xdmf`), optional `HDF5Adapter` (`rosolver[io]`), `VerificationStudy.from_multi_grid(...)`, and `LatexReportRenderer` / `compile_latex_to_pdf` supporting `.tex` and `.pdf` export across all 10 report scopes. See [External Adapters & LaTeX Guide](user_guide/external_adapters_and_latex.md).
- **Deterministic Scientific Claim Verification & Peer-Review Engine (`rosolver.claims`, Sprint 10)**: Two-axis claim audits (Mechanistic Fidelity + Empirical Concordance), Buckingham-Pi scale-separation & normalization diagnostics, Euler homogeneity/collinearity checks, discretization error budgets, and a stateful `R1 -> R2` author-rebuttal state machine with hard anti-override invariants. Exposed via `rosolver claim {init, audit, questions, respond}` and 3 new MCP tools. See [Claim Verification Guide](user_guide/claim_verification.md).
- **AI-Native MCP Server (`rosolver-mcp`)**: Zero-dependency `stdio` JSON-RPC 2.0 server (`rosolver.mcp`) exposing 15 deterministic tools (dimensions, formulation, calibration, simulation, V&V, OpenFOAM/FEniCSx/HDF5 external adapters, PySR candidate screening, claim verification & rebuttals, Markdown/LaTeX/PDF reporting), 6 schema/unit/template/adapter/status resources, and 4 guided prompts—enforcing *AI proposes, Rosolver evaluates* with compact array previews. See [MCP Server Guide](user_guide/mcp_server.md).
- **Flagship End-to-End Heat-Transfer Demo**: Complete 9-stage transient 1D heat conduction study ([examples/heat_transfer_study/run_study.py](../examples/heat_transfer_study/run_study.py)) from YAML model validation through dimensional checks, Fourier formulation, FVM PDE solve, GCI/timestep convergence, analytical comparison, MMS, conservation audit, and Markdown report generation. See [Tutorial](tutorials/heat_transfer.md).
- **Modular Package Architecture**: Organized across `core`, `symbolic`, `calibration`, `numerics`, `verification`, `validation`, `integrations`, `io`, `reporting`, and `mcp` while keeping 100% top-level backwards compatibility (`import rosolver`). See [Architecture](concepts/architecture.md).
- **Scientific Reporting, Workflow CLI & Jupyter Hooks**: Deterministic `ScientificReport` outputs (Markdown, JSON, YAML + sidecar manifests across 8 canonical scopes), 10 CLI workflow subcommands (`init`, `validate`, `model analyse`, `calibrate`, `simulate`, `verify`, `validate-results`, `report`, `info`, `mcp`) with `--json` and CI exit codes, plus pure-string `_repr_markdown_()` notebook rendering. See [Extended Procedures Guide](user_guide/procedures.md).
- **Persistent Workspaces, External Adapters & Validation**: Declarative `Model` specs (AST-safe parsing, `yaml.safe_load`), SHA-256 hash-verified `Project` workspaces with append-only execution ledgers & `replay_run`, `Dataset` multi-format I/O, `SolverAdapter`/`VerificationStudy` for external codes, and `ValidationStudy` experimental uncertainty comparisons.
- **Hardened V&V & Benchmark Coverage**: 5-state `VerificationStatus` (`Passed`, `Failed`, `Inconclusive`, `Warning`, `NotAssessed`), hardened Celik/Roache GCI, WLS + Jacobian identifiability diagnostics, and **182 passing tests** including independent benchmarks, Sprint 10 claim-audit benchmarks, Sprint 8 external-adapter/LaTeX suites, Sprint 9 PySR screening suites, and whole-codebase Carbon Audit regression suites. See [Benchmarks & Testing](development/benchmarks_and_testing.md).

---

## Documentation Map

- **Getting Started & User Guide**
  - [Quickstart (Python API & Workflow CLI)](getting_started/quickstart.md)
  - [Extended Procedures Guide (Discovery, Solutions, Calibration, V&V & Reporting)](user_guide/procedures.md)
  - [Scientific Claim Verification & Peer-Review Guide (`rosolver.claims`)](user_guide/claim_verification.md)
  - [External CFD/FEM Adapters & Headless LaTeX/PDF Reporting](user_guide/external_adapters_and_latex.md)
  - [PySR & Symbolic-Regression Candidate Importer (`CandidateImporter`)](user_guide/symbolic_regression_importer.md)
  - [MCP Server Guide (`rosolver-mcp` for AI Agents)](user_guide/mcp_server.md)
- **Concepts & Architecture**
  - [Package Architecture & Reproducibility Model](concepts/architecture.md)
  - [System Engineering Specification](architecture/ENG_SPEC.md)
  - [Sprint Roadmap](architecture/SPRINTS.md)
- **Tutorials**
  - [End-to-End 1D Transient Heat Conduction Study](tutorials/heat_transfer.md)
- **Scientific Reference**
  - [Mathematical, Numerical & V&V Methods Reference](scientific_reference/methods.md)
- **Development & Benchmarks**
  - [Testing Architecture, CI & Measured Benchmark Runtimes](development/benchmarks_and_testing.md)
- **Existing Lecture Notes & Refs**
  - [Chain-Rule Equation Formulation Notes (PDF)](notes/eqform_class_notes.pdf)
  - [Stefan Front-Tracking Notes (PDF)](notes/stefan_front_tracking_notes.pdf)
  - [Verification & Stability Notes (PDF)](notes/verification_and_stability_notes.pdf)
