"""
Flagship 1D transient heat conduction end-to-end workflow (ENG_SPEC.md P3.3).

Demonstrates all 9 lifecycle steps in order:
  1. Model YAML specification & validation
  2. Dimensional consistency check (Fourier / thermal diffusivity)
  3. Governing symbolic & Buckingham-Pi formulation
  4. Numerical FVM PDE solve (Crank-Nicolson, linear reconstruction)
  5. Spatial grid (`GCI`) & timestep convergence verification
  6. Analytical Fourier-series solution comparison with explicit u_num & u_exp
  7. Method of Manufactured Solutions (MMS) code verification
  8. Discrete FVM flux/ledger conservation audit
  9. Reproducible Scientific Markdown report + PNG figures + JSON manifest

Usage:
    python -m examples.heat_transfer_study.run_study [--output outputs/heat_transfer_study]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any, Dict, Optional, Union

import numpy as np

from rosolver import (
    Dataset,
    Dimension,
    Model,
    Project,
    ValidationStudy,
    VerificationStatus,
    audit_pde_conservation,
    grid_convergence_study,
    mms_study,
    plot_conservation_audit,
    plot_gci_study,
    plot_mms_convergence,
    plot_validation,
    standardize_result,
    timestep_study,
)

MODEL_PATH = Path(__file__).resolve().parent / "model.yaml"


def run_flagship_study(output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Execute the full 9-step 1D transient heat conduction verification & validation study."""
    study_dir = Path(output_dir) if output_dir is not None else Path("outputs/heat_transfer_study")
    timings: Dict[str, float] = {}
    t_total_start = time.perf_counter()

    # 1. Model YAML/Python specification & validation
    t0 = time.perf_counter()
    model = Model.from_file(MODEL_PATH)
    model.validate(strict=True)
    project = Project.create(
        study_dir,
        name="1D Transient Heat Conduction Study",
        description="End-to-end symbolic, FVM PDE, GCI, MMS, conservation, and analytical Fourier V&V study.",
    )
    project.add_model(model, filename="heat_conduction_1d.yaml")
    timings["model_setup_s"] = time.perf_counter() - t0

    # 2. Dimensional consistency check
    t0 = time.perf_counter()
    lhs_dim = Dimension.from_unit("K") / Dimension.from_unit("s")
    rhs_dim = Dimension.from_unit("m^2/s") * Dimension.from_unit("K") / (Dimension.from_unit("m") ** 2)
    if lhs_dim != rhs_dim:
        raise AssertionError(f"PDE dimensional mismatch: {lhs_dim} vs {rhs_dim}")
    alpha_from_k = Dimension.from_unit("W/(m*K)") / (
        Dimension.from_unit("kg/m^3") * Dimension.from_unit("J/(kg*K)")
    )
    if alpha_from_k != Dimension.from_unit("m^2/s"):
        raise AssertionError(f"Thermal diffusivity dimension mismatch: {alpha_from_k}")
    timings["dimensional_check_s"] = time.perf_counter() - t0

    # 3. Symbolic & Buckingham-Pi governing equation formulation
    t0 = time.perf_counter()
    sym_res = project.run_symbolic(target="T", wrt="t", zero_initials=True)
    timings["symbolic_s"] = time.perf_counter() - t0

    # 4. Numerical FVM PDE solve
    t0 = time.perf_counter()
    t_end = 0.5
    sim_res = project.run_simulation(t_span=(0.0, t_end), n_cells=64, method="cn")
    sol = sim_res.raw
    timings["pde_solve_s"] = time.perf_counter() - t0

    # 5. Spatial grid (GCI) & timestep convergence
    t0 = time.perf_counter()
    law = model.to_conservation_law()
    alpha_val = float(model.parameter_values()["alpha"])
    dom = model.numerics.domain or (0.0, 1.0)
    ic_map = model._extract_initial_conditions(None, law.fields)
    bc_spec = model._extract_boundary_conditions(None, law.fields)

    gci_report = grid_convergence_study(
        lambda n: float(
            law.solve(
                domain=dom,
                n_cells=n,
                t_end=t_end,
                initial=ic_map["T"],
                bc=bc_spec,
                params={"alpha": alpha_val},
                scheme="cn",
                dt=0.002,
                reconstruction="linear",
            ).integral("T")
        ),
        cells=(32, 64, 128),
        domain_length=float(dom[1] - dom[0]),
        formal_order=2.0,
    )
    gci_sr = standardize_result(gci_report, kind="gci", model_name=model.name, model_version=model.version)
    project.record_run(
        "gci",
        gci_sr,
        model=model,
        configuration={"cells": [32, 64, 128], "t_end": t_end, "dt": 0.002},
    )
    fig_gci_rel = "reports/figures/gci_convergence.png"
    plot_gci_study(gci_report, path=project.root / fig_gci_rel, qoi_label="Thermal energy integral [K*m]")

    dt_report = timestep_study(
        lambda dt: float(
            law.solve(
                domain=dom,
                n_cells=64,
                t_end=t_end,
                initial=ic_map["T"],
                bc=bc_spec,
                params={"alpha": alpha_val},
                scheme="cn",
                dt=dt,
                reconstruction="linear",
            ).probe(0.5, "T")
        ),
        dts=(0.1, 0.05, 0.025),
        formal_order=2.0,
        dx=float((dom[1] - dom[0]) / 64),
        diffusivity=alpha_val,
    )
    dt_sr = standardize_result(dt_report, kind="timestep", model_name=model.name, model_version=model.version)
    project.record_run(
        "timestep",
        dt_sr,
        model=model,
        configuration={"dts": [0.025, 0.05, 0.1], "n_cells": 64, "scheme": "cn"},
    )
    timings["gci_and_timestep_s"] = time.perf_counter() - t0

    # 6. Analytical Fourier-series comparison
    t0 = time.perf_counter()
    x_grid = sol.x
    t_exact = np.exp(-alpha_val * (np.pi ** 2) * t_end) * np.sin(np.pi * x_grid)
    u_num_est = max(abs(float(gci_report.gci_fine_21) * abs(float(gci_report.values[0]))), 5e-3)
    sim_ds = Dataset(
        name="fvm_cn_64",
        coordinates={"x": x_grid},
        fields={"T": sol.final("T")},
        uncertainties={"T": np.full_like(x_grid, u_num_est)},
        units={"x": "m", "T": "K"},
    )
    ref_ds = Dataset(
        name="fourier_exact",
        coordinates={"x": x_grid},
        fields={"T": t_exact},
        uncertainties={"T": np.full_like(x_grid, 1e-4)},
        units={"x": "m", "T": "K"},
    )
    val_report = ValidationStudy.from_datasets(
        sim_ds,
        ref_ds,
        field="T",
        coord="x",
        study_name="Fourier Analytical Mode-1 Comparison",
        operating_conditions={"t_end_s": t_end, "alpha_m2_s": alpha_val, "n_cells": 64},
    )
    val_sr = standardize_result(
        val_report, kind="model_validation", model_name=model.name, model_version=model.version
    )
    project.record_run(
        "model_validation",
        val_sr,
        model=model,
        configuration={"reference": "exact_fourier_mode_1", "u_num": u_num_est},
    )
    fig_val_rel = "reports/figures/analytical_comparison.png"
    plot_validation(val_report, path=project.root / fig_val_rel, x_label="x (m)")
    timings["analytical_comparison_s"] = time.perf_counter() - t0

    # 7. MMS code verification
    t0 = time.perf_counter()
    mms_report = mms_study(
        law,
        exact={"T": "exp(-t) * sin(pi * x)"},
        domain=dom,
        t_end=0.2,
        cells=(16, 32, 64),
        params={"alpha": alpha_val},
        bc_kind="dirichlet",
        reconstruction="linear",
        scheme="cn",
    )
    mms_sr = standardize_result(mms_report, kind="mms", model_name=model.name, model_version=model.version)
    project.record_run(
        "mms",
        mms_sr,
        model=model,
        configuration={"cells": [16, 32, 64], "t_end": 0.2},
    )
    fig_mms_rel = "reports/figures/mms_convergence.png"
    plot_mms_convergence(mms_report, path=project.root / fig_mms_rel)
    timings["mms_s"] = time.perf_counter() - t0

    # 8. Conservation audit
    t0 = time.perf_counter()
    cons_report = audit_pde_conservation(sol, tol=1e-10)
    cons_sr = standardize_result(
        cons_report, kind="conservation", model_name=model.name, model_version=model.version
    )
    project.record_run("conservation", cons_sr, model=model, configuration={"tol": 1e-10})
    fig_cons_rel = "reports/figures/conservation_audit.png"
    plot_conservation_audit(cons_report, path=project.root / fig_cons_rel)
    timings["conservation_s"] = time.perf_counter() - t0

    # 9. Reproducible Scientific Markdown report + JSON manifest
    t0 = time.perf_counter()
    figures = [
        {"path": fig_val_rel, "caption": "FVM PDE solution vs. exact analytical Fourier mode-1"},
        {"path": fig_mms_rel, "caption": "MMS spatial order verification (L2 and Linf norms)"},
        {"path": fig_gci_rel, "caption": "Spatial Grid Convergence Index (GCI) band"},
        {"path": fig_cons_rel, "caption": "Discrete FVM flux & source conservation residual"},
    ]
    report = project.report(
        output="reports/final_report.md",
        title="1D Transient Heat Conduction — End-to-End V&V Study",
        description="Flagship Rosolver v0.2 lifecycle study covering dimensional check, symbolic formulation, FVM PDE simulation, GCI/timestep convergence, analytical Fourier comparison, MMS, conservation audit, and scientific reporting.",
        figures=figures,
        also_manifest=True,
    )
    timings["report_s"] = time.perf_counter() - t0
    timings["total_s"] = time.perf_counter() - t_total_start

    return {
        "project_root": project.root,
        "report_path": project.root / "reports" / "final_report.md",
        "manifest_path": project.root / "reports" / "final_report.manifest.json",
        "figures": [project.root / f["path"] for f in figures],
        "report_status": report.status,
        "statuses": {
            "symbolic": sym_res.status,
            "simulation": sim_res.status,
            "gci": gci_report.status,
            "timestep": dt_report.status,
            "validation": val_report.status,
            "mms": mms_report.status,
            "conservation": cons_report.status,
            "integrated_report": report.status,
        },
        "metrics": {
            "fourier_rmse_K": val_report.rmse,
            "fourier_max_abs_error_K": val_report.max_abs_error,
            "gci_observed_order": gci_report.observed_order,
            "gci_fine_21": gci_report.gci_fine_21,
            "timestep_observed_order": dt_report.gci.observed_order,
            "mms_order_l2": mms_report.order_l2["T"],
            "conservation_max_abs_residual": cons_report.entries["T"].max_abs_residual,
        },
        "timings_s": timings,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m examples.heat_transfer_study.run_study",
        description="Run the Rosolver flagship 1D transient heat conduction end-to-end study.",
    )
    parser.add_argument(
        "--output",
        default="outputs/heat_transfer_study",
        help="Target directory for the generated project and report (default: outputs/heat_transfer_study).",
    )
    args = parser.parse_args(argv)
    summary = run_flagship_study(output_dir=args.output)
    print(f"Flagship study finished: overall={summary['report_status'].value}")
    print(f"Report written to: {summary['report_path']}")
    for k, st in summary["statuses"].items():
        print(f"  - {k}: {st.value}")
    return 0 if summary["report_status"] in (VerificationStatus.PASSED, VerificationStatus.WARNING) else 1


if __name__ == "__main__":
    raise SystemExit(main())
