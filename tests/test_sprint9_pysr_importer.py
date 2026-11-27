"""Sprint 9 Slim test suite: PySR / Symbolic-Regression Candidate Importer gated by Sprint 10."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from rosolver import (
    CandidateImporter,
    CandidateScreeningReport,
    ClaimVerdictKind,
    Project,
    VerificationStatus,
)


def _pendulum_data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    L = np.linspace(0.25, 2.5, 20)
    g = np.full_like(L, 9.81)
    T_true = 2.0 * np.pi * np.sqrt(L / g)
    sigma_T = np.full_like(L, 0.005)
    T_obs = T_true + rng.normal(0.0, sigma_T)
    return {"L": L, "g": g, "T": T_obs, "u_T": sigma_T}


def test_pysr_csv_four_stage_gate_and_security_rejection(tmp_path: Path):
    hof = tmp_path / "hall_of_fame.csv"
    hof.write_text(
        "Complexity,Loss,Score,Equation\n"
        "1,1.20,0.0,1.0 + x0\n"
        "3,0.45,0.2,x0 + x1\n"
        "4,0.80,0.1,1e-12 * sqrt(x0 / x1)\n"
        "5,0.0004,1.5,6.21 * sqrt(x0 / x1)\n"
        "7,0.0001,2.0,__import__('os').system('echo pwned')\n",
        encoding="utf-8",
    )

    data = _pendulum_data()
    report = CandidateImporter.from_pysr_csv(
        hof,
        target="T",
        dims={"T": "s", "L": "m", "g": "m/s^2"},
        variable_map={"x0": "L", "x1": "g"},
        data=data,
        sigma_y="u_T",
        refit_numeric_constants=True,
    )

    assert isinstance(report, CandidateScreeningReport)
    assert report.status == VerificationStatus.PASSED
    assert report.total_imported == 5
    assert len(report.admissible_candidates) == 1
    assert len(report.rejected_candidates) == 4

    # Best candidate recovers T = 2*pi*sqrt(L/g) via WLS refit
    best = report.best_candidate
    assert best is not None
    assert best.fit_result is not None
    assert best.fit_result.params["p_0"] == pytest.approx(2.0 * np.pi, rel=1e-2)
    assert best.fit_result.r2 > 0.999

    # Verify exact rejection stages for all 4 bad candidates
    rej_by_raw = {rc.candidate.raw_equation: rc for rc in report.rejected_candidates}
    assert rej_by_raw["__import__('os').system('echo pwned')"].rejection_stage == "ast_parse"
    assert rej_by_raw["1.0 + x0"].rejection_stage == "dimensions"
    assert any(
        h["issue_kind"] == "implicit_unit_normalization"
        for h in rej_by_raw["1.0 + x0"].normalization_hypotheses
    )
    assert rej_by_raw["x0 + x1"].rejection_stage == "dimensions"
    assert rej_by_raw["1e-12 * sqrt(x0 / x1)"].rejection_stage == "scales"
    assert rej_by_raw["1e-12 * sqrt(x0 / x1)"].scale_log10_gap > 10.0

    # Summary & Markdown repr render cleanly with zero absolute user paths
    md = report._repr_markdown_()
    assert "pysr_c5" in md
    assert str(Path.home().parent) not in md
    assert str(Path.home().parent) not in report.summary()


def test_non_identifiable_candidate_warning_and_strict_rejection():
    data = _pendulum_data()
    # c_0 * c_1 * sqrt(L/g) has structurally collinear parameters c_0, c_1
    rep_warn = CandidateImporter.from_expressions(
        ["c_0 * c_1 * sqrt(L / g)"],
        target="T",
        dims={"T": "s", "L": "m", "g": "m/s^2"},
        data=data,
        sigma_y="u_T",
        reject_non_identifiable=False,
    )
    assert rep_warn.status == VerificationStatus.WARNING
    assert len(rep_warn.admissible_candidates) == 1
    assert rep_warn.admissible_candidates[0].fit_result is not None
    assert rep_warn.admissible_candidates[0].fit_result.identifiability is not None
    assert rep_warn.admissible_candidates[0].fit_result.identifiability.identifiable is False

    rep_strict = CandidateImporter.from_expressions(
        ["c_0 * c_1 * sqrt(L / g)"],
        target="T",
        dims={"T": "s", "L": "m", "g": "m/s^2"},
        data=data,
        sigma_y="u_T",
        reject_non_identifiable=True,
    )
    assert rep_strict.status == VerificationStatus.FAILED
    assert len(rep_strict.admissible_candidates) == 0
    assert rep_strict.rejected_candidates[0].rejection_stage == "calibration"


def test_pysr_model_duck_typing_optional_guard_and_sprint10_manifest_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # 1. Optional pysr import guard
    monkeypatch.setitem(sys.modules, "pysr", None)
    with pytest.raises(ImportError, match="rosolver\\[pysr\\]"):
        CandidateImporter.run_pysr(
            np.ones((5, 2)),
            np.ones(5),
            feature_names=["L", "g"],
            target="T",
            dims={"T": "s", "L": "m", "g": "m/s^2"},
        )

    # 2. Duck-typed PySRRegressor (.equations_ list/DataFrame)
    fake_model = types.SimpleNamespace(
        equations_=[
            {"complexity": 5, "loss": 1e-4, "score": 1.2, "equation": "6.28 * sqrt(L / g)"},
            {"complexity": 3, "loss": 0.5, "score": 0.1, "equation": "L * g"},
        ]
    )
    data = _pendulum_data()
    rep = CandidateImporter.from_pysr_model(
        fake_model,
        target="T",
        dims={"T": "s", "L": "m", "g": "m/s^2"},
        data=data,
        sigma_y="u_T",
    )
    assert rep.status == VerificationStatus.PASSED
    assert len(rep.admissible_candidates) == 1
    assert len(rep.rejected_candidates) == 1

    # 3. Direct handoff to Sprint 10 ClaimManifest + Project.audit_claims
    manifest = rep.to_claim_manifest(top_k=1)
    assert len(manifest.claims) == 1
    proj = Project.create(tmp_path / "sr_audit_proj", name="SR Audit")
    r1 = proj.audit_claims(manifest=manifest)
    assert r1.verdicts[0].overall_verdict == ClaimVerdictKind.SUPPORTED
