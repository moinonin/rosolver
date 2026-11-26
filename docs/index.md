# Rosolver Documentation (`v0.2.0`)

Rosolver is an open-source, reproducible scientific modeling, calibration, numerical simulation, verification & validation (V&V), and scientific reporting toolkit for Python and the command line.

> **Scope & Standards Note:** Rosolver's verification and validation diagnostics (MMS, Roache/Celik GCI, discrete conservation audits, and experimental comparison bands) follow ASME V&V 20-2009, AIAA S-141-2024, Roache, and Celik et al. (2008) methodologies for transparent traceability and do not by themselves constitute formal regulatory or standards certification.

---

## Documentation Map

- **Getting Started & User Guide**
  - [Quickstart (Python API & Workflow CLI)](getting_started/quickstart.md)
  - [Extended Procedures Guide (Discovery, Solutions, Calibration, V&V & Reporting)](user_guide/procedures.md)
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
