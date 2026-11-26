# Testing Architecture, CI & Baseline Benchmarks

## Test Organization

- **Unit & Integration Suites (`tests/`):** Cover symbolic derivation & DAG security ([tests/test_security_and_api.py](../../tests/test_security_and_api.py), [tests/test_rosolver.py](../../tests/test_rosolver.py)), calibration & templates ([tests/test_sprint2_calibration_templates.py](../../tests/test_sprint2_calibration_templates.py)), validated `Model` / events / results ([tests/test_sprint3_model_results_events.py](../../tests/test_sprint3_model_results_events.py)), persistent `Project` / ledger ([tests/test_sprint4_project_ledger.py](../../tests/test_sprint4_project_ledger.py)), `Dataset` / `SolverAdapter` / `ValidationStudy` ([tests/test_sprint5_data_adapters_validation.py](../../tests/test_sprint5_data_adapters_validation.py)), scientific reporting & CLI ([tests/test_sprint6_reporting_cli.py](../../tests/test_sprint6_reporting_cli.py)), moving-boundary phase change ([tests/test_phasechange.py](../../tests/test_phasechange.py)), and flagship end-to-end verification ([tests/test_sprint7_flagship_modular_release.py](../../tests/test_sprint7_flagship_modular_release.py)).
- **Independent Verification Benchmarks (`tests/benchmarks/`):**
  1. [test_benchmark_heat_conduction.py](../../tests/benchmarks/test_benchmark_heat_conduction.py) — Exact Fourier series spatial/temporal convergence, analytical energy balance, and Neumann Stefan convergence.
  2. [test_benchmark_mms.py](../../tests/benchmarks/test_benchmark_mms.py) — Manufactured heat equation SymPy source & spatial convergence order.
  3. [test_benchmark_ode.py](../../tests/benchmarks/test_benchmark_ode.py) — Analytical ODE tolerance refinement, fixed-step observed order, parameter recovery, and Newton failure propagation.
  4. [test_benchmark_dimensional.py](../../tests/benchmarks/test_benchmark_dimensional.py) — Pendulum, pipe flow, convective Nusselt/Prandtl rank, and inconsistent equation rejection.
  5. [test_benchmark_calibration.py](../../tests/benchmarks/test_benchmark_calibration.py) — Noisy Hill-curve WLS recovery, parameter bounds, and unidentifiability detection.
  6. [test_benchmark_external_gci.py](../../tests/benchmarks/test_benchmark_external_gci.py) — Celik (2008) Table 1 Case 1 GCI reference, edge cases, and external CSV CLI pipeline.

## Measured Baseline Benchmark Runtimes (`v0.2.0`)

Measured on Apple Silicon macOS (Python 3.11.13, NumPy 2.4.6, SciPy 1.17.1, SymPy 1.14.0) via `pytest --durations=15 -q tests/benchmarks/` (**17 passed in 2.04s total**):

| Duration | Test Case |
| --- | --- |
| `0.32s` | `test_benchmark_calibration.py::test_noisy_hill_parameter_recovery_and_bounds` |
| `0.31s` | `test_benchmark_heat_conduction.py::test_stefan_neumann_spatial_convergence_order` |
| `0.15s` | `test_benchmark_heat_conduction.py::test_temporal_convergence_beuler_and_cn` |
| `0.06s` | `test_benchmark_heat_conduction.py::test_spatial_convergence_against_fourier_series` |
| `0.05s` | `test_benchmark_mms.py::test_manufactured_heat_equation_forcing_and_order` |
| `0.04s` | `test_benchmark_ode.py::test_ode_accuracy_and_tolerance_refinement` |
| `0.03s` | `test_benchmark_ode.py::test_newton_failure_propagation_sets_success_false` |
| `0.02s` | `test_benchmark_heat_conduction.py::test_independent_analytical_energy_balance` |

All remaining 9 benchmark cases executed in `<= 0.01s` each.

## Continuous Integration

GitHub Actions workflow [.github/workflows/ci.yml](../../.github/workflows/ci.yml) runs `pytest -v` across Python `3.10`, `3.11`, and `3.12` on every push and pull request to `main`, alongside a headless smoke execution of `python -m examples.heat_transfer_study.run_study`.
