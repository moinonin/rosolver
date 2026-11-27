"""Tests for the Ice Growth (Nucleation to Macroscopic Solid Mass) study."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from examples.ice_nucleation_growth_study import (
    exact_ice_growth_solution,
    run_ice_nucleation_study,
)
from rosolver import Model


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_exact_ice_growth_solution_scaling_and_eq16():
    t = np.array([0.0, 50.0, 200.0, 600.0])
    sol = exact_ice_growth_solution(
        t, N0=10000.0, beta=200.0, z0=0.0005, z_max=0.05, rho=917.0, sigma=0.01
    )
    # Monotonic growth bounded by z_max, strictly decreasing velocity, positive pressure gradient
    assert np.all(np.diff(sol["z"]) > 0.0)
    assert np.all(sol["z"] < 0.05)
    assert np.all(np.diff(sol["vz"]) < 0.0)
    assert np.all(sol["dp_dz"] > 0.0)
    # Exact Eq. (16) Euler relation dp/dz = -rho * d2z/dt2
    assert np.allclose(sol["dp_dz"], -917.0 * sol["az"], rtol=1e-14)
    # Exact momentum-impulse conservation rho * vz + P_imp = rho * vz(0)
    assert np.allclose(917.0 * sol["vz"] + sol["P_imp"], 917.0 * sol["vz"][0], rtol=1e-14)


def test_ice_nucleation_study_end_to_end_verification(tmp_path: Path):
    model = Model.from_file(REPO_ROOT / "examples" / "ice_nucleation_growth_study" / "model.yaml")
    model.validate(strict=True)

    summary = run_ice_nucleation_study(output_dir=tmp_path / "ice_study")

    assert summary["report_status"] == "Passed"
    assert summary["max_z_abs_err_m"] < 1e-9
    assert summary["max_vz_abs_err_ms"] < 1e-11
    assert summary["rk4_observed_order"] > 3.85
    assert summary["mass_conservation_max_res"] < 1e-13
    assert summary["momentum_impulse_max_res"] < 1e-13
    assert summary["validation_rmse_m"] < 1e-10

    # Higher nucleation rate beta monotonically shortens coalescence time t_c
    t_coals = [c["t_coal_s"] for c in summary["beta_cases"]]
    assert t_coals == sorted(t_coals, reverse=True)

    report_md = (tmp_path / "ice_study" / "reports" / "final_report.md").read_text(encoding="utf-8")
    assert str(Path.home().parent) not in report_md
