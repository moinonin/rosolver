"""Tests for the Cardiovascular Hemodynamics, Heart-Rate & Human Lifespan Formulation Study."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from examples.cardiovascular_lifespan_study import (
    gompertz_analytical_lifespan_yr,
    gompertz_asymptotic_linear_yr,
    murray_poiseuille_network_demo,
    run_cardiovascular_study,
)
from rosolver import Model, formulate


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_symbolic_strouhal_and_convective_pi_discovery():
    deriv_H = formulate(
        relations=["H = fn(a, v, D)", "a = fn(t)", "v = fn(t; c_v)", "D = fn(t; c_D)"],
        dims={"H": "1/T", "a": "T", "v": "L/T", "D": "L", "t": "T", "c_v": "L/T^2", "c_D": "L/T"},
        constants=["c_v", "c_D"],
        target="H",
        wrt="t",
    )
    assert "H" in deriv_H.pi_groups
    assert [str(p) for p in deriv_H.pi_groups["H"]] == ["a*v/D"]

    deriv_L = formulate(
        relations=["L_life = fn(H, v, D)"],
        dims={"L_life": "T", "H": "1/T", "v": "L/T", "D": "L"},
        target="L_life",
        zero_initials=False,
    )
    assert "L_life" in deriv_L.pi_groups
    assert [str(p) for p in deriv_L.pi_groups["L_life"]] == ["D*H/v"]
    cand_names = {c.name for c in deriv_L.candidates}
    assert {"leading_dimensional", "buckingham_pi_power", "buckingham_pi_affine"}.issubset(cand_names)


def test_gompertz_analytical_and_network_reduction_properties():
    model = Model.from_file(REPO_ROOT / "examples" / "cardiovascular_lifespan_study" / "model.yaml")
    params = model.parameter_values()

    hr_vec = np.array([50.0, 65.0, 80.0, 95.0])
    exact_yr = gompertz_analytical_lifespan_yr(hr_vec, params)
    asymp_yr = gompertz_asymptotic_linear_yr(hr_vec, params)

    # Strictly decreasing lifespan with increasing resting heart rate
    assert np.all(np.diff(exact_yr) < 0.0)
    # Leading Euler-Mascheroni log expansion has O(z*ln(1/z)/gamma_G) ~ 0.21-0.31 yr truncation residual
    assert np.max(np.abs(exact_yr - asymp_yr)) < 0.35
    # Slope agreement within 2%
    assert abs((exact_yr[-1] - exact_yr[0]) - (asymp_yr[-1] - asymp_yr[0])) < 0.15

    net = murray_poiseuille_network_demo()
    assert net["D_poiseuille_aged_mm"] < net["D_poiseuille_young_mm"]
    assert net["resistance_increase_factor"] > 1.5


def test_cardiovascular_study_end_to_end_verification(tmp_path: Path):
    summary = run_cardiovascular_study(output_dir=tmp_path / "cardio_study")

    assert summary["report_status"] == "Passed"
    assert 78.0 <= summary["expected_lifespan_ref_65bpm_yr"] <= 81.0
    assert summary["human_best_pi_r2"] > 0.95
    assert summary["rk4_observed_order"] > 3.85
    assert summary["invariant_max_abs_drift"] < 1e-12
    assert summary["validation_rmse_yr"] < 0.75

    # Phenotype monotonicity: athletic > normocardic > sympathetic > stenotic+tachycardic
    lifespans = [p["expected_lifespan_yr"] for p in summary["phenotypes"]]
    assert lifespans == sorted(lifespans, reverse=True)

    report_md = (tmp_path / "cardio_study" / "reports" / "final_report.md").read_text(encoding="utf-8")
    home_prefix = str(Path.home().parent)
    assert home_prefix not in report_md
