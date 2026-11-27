"""Tests for the independent verification & audit study of docs/refs/new_icd_study.txt."""

from __future__ import annotations

from pathlib import Path

from examples.pond_freezing_audit_study import (
    ashton_evaporative_exact_delta,
    clausius_clapeyron_es_pa,
    run_pond_freezing_audit,
)
from rosolver import Model


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_ashton_evaporative_physics_and_vapor_pressure():
    es0 = clausius_clapeyron_es_pa(0.0)
    esm1 = clausius_clapeyron_es_pa(-1.0)
    assert 610.0 < es0 < 612.0
    assert 560.0 < esm1 < 565.0
    assert es0 > esm1

    res_full = ashton_evaporative_exact_delta(
        72000.0, delta0=1e-4, rho=917.0, Lf=3.34e5, k_ice=2.22, h_c=13.3, dT=1.0, J_evap=7.96412e-6, L_sub=2.835e6
    )
    res_conv = ashton_evaporative_exact_delta(
        72000.0, delta0=1e-4, rho=917.0, Lf=3.34e5, k_ice=2.22, h_c=13.3, dT=1.0, J_evap=0.0, L_sub=2.835e6
    )
    assert float(res_full["delta"]) > 2.0 * float(res_conv["delta"])


def test_pond_freezing_audit_study_end_to_end(tmp_path: Path):
    model = Model.from_file(REPO_ROOT / "examples" / "pond_freezing_audit_study" / "model.yaml")
    model.validate(strict=True)

    summary = run_pond_freezing_audit(output_dir=tmp_path / "pond_audit")

    assert summary["report_status"] == "Passed"
    assert summary["claim1_environmental_flux"]["verdict"] == "SUPPORTED"
    assert summary["claim1_environmental_flux"]["evap_share_pct"] > 50.0
    assert summary["claim2_euler_momentum_eq16"]["verdict"] == "FALSIFIED_AS_LITERAL_VAPOR_PRESSURE_LAW"
    assert summary["claim2_euler_momentum_eq16"]["scale_mismatch_orders_of_mag"] > 12.0
    assert summary["claim3_timestep_n12_and_035pct"]["rk4_observed_order"] > 3.9
    assert summary["claim4_20h_pond_freezing_thickness"]["delta_full_conv_evap_mm"] > 7.5
    assert summary["claim5_peat_vs_freshwater_treatment"]["verdict"] == "INCONCLUSIVE_QUALITATIVELY_PLAUSIBLE"
    assert summary["verification"]["validation_rmse_m"] < 1e-12

    report_md = (tmp_path / "pond_audit" / "reports" / "final_report.md").read_text(encoding="utf-8")
    assert str(Path.home().parent) not in report_md
