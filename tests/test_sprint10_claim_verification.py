"""Sprint 10 tests: Deterministic Claim Verification, Scale Audit, Error Budget & R1->R2 Rebuttal Engine."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rosolver import (
    ClaimAuditEngine,
    ClaimKind,
    ClaimManifest,
    ClaimSpec,
    ClaimVerdict,
    ClaimVerdictKind,
    Model,
    Project,
    RebuttalSubmission,
    ReportGenerator,
    TwoAxisAssessment,
    VerificationStatus,
    audit_claimed_accuracy,
    check_equation_scales,
    check_homogeneity_elasticities,
    diagnose_normalization_shortcuts,
)
from rosolver.cli import EXIT_CHECK_FAILED, EXIT_OK, main as cli_main
from rosolver.mcp import MCPServer, TOOL_SPECIFICATIONS


REPO_ROOT = Path(__file__).resolve().parent.parent
POND_MODEL_YAML = REPO_ROOT / "examples" / "pond_freezing_audit_study" / "model.yaml"


# ---------------------------------------------------------------------------
# Benchmark S10-B2: Implicit Normalization & Differential Geometry Diagnostics
# ---------------------------------------------------------------------------


def test_s10_b2_normalization_and_differential_geometry_diagnostics():
    # 1. Draft ice growth equation: N * dm_dt = beta * (1 - rho * V)
    rep1 = diagnose_normalization_shortcuts(
        "N * dm_dt = beta * (1 - rho * V)",
        dims={
            "N": "1/m^3",
            "dm_dt": "kg/s",
            "beta": "1/(m^3*s)",
            "rho": "kg/m^3",
            "V": "m^3",
        },
    )
    assert rep1.status == VerificationStatus.FAILED
    assert any(
        h.issue_kind == "implicit_unit_normalization" and h.missing_factor_dimension == "M"
        for h in rep1.hypotheses
    )

    # 2. Draft cross-sectional area confusion: sigma = A * dz
    rep2 = diagnose_normalization_shortcuts(
        "sigma = A * dz",
        dims={"sigma": "m^2", "A": "m^2", "dz": "m"},
    )
    assert rep2.status == VerificationStatus.FAILED
    assert any(
        h.issue_kind == "differential_geometry_mismatch" and h.suggested_symbol == "dz"
        for h in rep2.hypotheses
    )

    # 3. Clean closed form passes
    rep3 = diagnose_normalization_shortcuts(
        "N * dm_dt = beta * (m_max - rho * V)",
        dims={
            "N": "1/m^3",
            "dm_dt": "kg/s",
            "beta": "1/(m^3*s)",
            "m_max": "kg",
            "rho": "kg/m^3",
            "V": "m^3",
        },
    )
    assert rep3.status == VerificationStatus.PASSED
    assert rep3.consistent is True


# ---------------------------------------------------------------------------
# Benchmark S10-B3: Euler Homogeneity Elasticity Sum Rule & Pi-Collinearity
# ---------------------------------------------------------------------------


def test_s10_b3_homogeneity_elasticity_lock_and_collinear_pi():
    # Intra-species human lifespan elasticities violate zero-intercept L_0 = 0 sum rule:
    # [L_life] = T, [H] = 1/T, [v] = L/T, [D] = L => E_H - E_D must equal -1 if L_0 = 0!
    rep_fail = check_homogeneity_elasticities(
        target_dim="T",
        input_dims={"H": "1/T", "v": "L/T", "D": "L"},
        elasticities={"H": -0.15, "v": -0.08, "D": 0.08},
    )
    assert rep_fail.status == VerificationStatus.FAILED
    assert rep_fail.zero_intercept_admissible is False
    assert rep_fail.max_abs_residual > 0.7

    # Cross-species allometric elasticities satisfy D*E = d_y, but 1D collinear data triggers warning
    h_1d = np.linspace(48.0, 98.0, 8)
    d_1d = np.full_like(h_1d, 0.024)
    v_1d = np.full_like(h_1d, 0.88)
    rep_col = check_homogeneity_elasticities(
        target_dim="T",
        input_dims={"H": "1/T", "v": "L/T", "D": "L"},
        elasticities={"H": -0.75, "v": -0.25, "D": 0.25},
        data={"H": h_1d, "v": v_1d, "D": d_1d},
    )
    assert rep_col.zero_intercept_admissible is True
    assert rep_col.pi_collinear is True
    assert rep_col.status == VerificationStatus.WARNING


# ---------------------------------------------------------------------------
# Benchmark S10-B1: Full new_icd_study.txt R1 -> R2A (Pass) & R2B (Fail)
# ---------------------------------------------------------------------------


def _build_pond_claim_manifest() -> ClaimManifest:
    return ClaimManifest(
        study_title="Winter Pond Freeze Crystallization Audit (new_icd_study.txt)",
        source_reference="docs/refs/new_icd_study.txt",
        claims=[
            ClaimSpec(
                id="C1_surface_energy",
                title="Convective + Evaporative Surface Cooling Balance",
                kind=ClaimKind.EQUATION_LAW,
                equations=["q_tot = h_c * dT + J * L_sub"],
                variables_units={
                    "q_tot": "W/m^2",
                    "h_c": "W/(m^2*K)",
                    "dT": "K",
                    "J": "kg/(m^2*s)",
                    "L_sub": "J/kg",
                },
                operating_envelope={
                    "q_tot": (34.0, 38.0),
                    "h_c": (12.0, 14.0),
                    "dT": (1.0, 1.0),
                    "J": (7.96e-6, 7.97e-6),
                    "L_sub": (2.83e6, 2.84e6),
                },
            ),
            ClaimSpec(
                id="C2_euler_vapor_law",
                title="Literal Vapor-Pressure Euler Balance rho*a + grad_P = 0",
                kind=ClaimKind.EQUATION_LAW,
                equations=["rho * a + grad_P = 0"],
                variables_units={"rho": "kg/m^3", "a": "m/s^2", "grad_P": "Pa/m"},
                operating_envelope={
                    "rho": (917.0, 917.0),
                    "a": (7.0e-14, 8.5e-14),
                    "grad_P": (884.8, 8848.0),
                },
            ),
            ClaimSpec(
                id="C3_coarse_n12_accuracy",
                title="Coarse n=12 (dt=6000 s) 0.35% Validation Error Claim",
                kind=ClaimKind.NUMERICAL_ACCURACY,
                discretization_spec={"n_steps": 12, "dt": 6000.0, "t_end": 72000.0, "scheme": None},
                claimed_metrics={"rel_error": 0.0035},
                qoi_field="delta_full",
            ),
            ClaimSpec(
                id="C5_peat_purification",
                title="Peat Settlement Pond Ice Equivalent to Freshwater Lake Ice",
                kind=ClaimKind.QUALITATIVE_OBSERVATION,
                required_observables=["solute_rejection_pct"],
                claimed_metrics={"min_solute_rejection_pct": 90.0},
                metadata={"mechanistic_plausible": True, "mechanistic_note": "Pe_s = 0.229 << 1 favors planar rejection."},
            ),
        ],
    )


def test_s10_b1_new_icd_study_r1_to_r2_positive_and_negative_rebuttals(tmp_path: Path):
    model = Model.from_file(POND_MODEL_YAML)
    proj = Project.create(tmp_path / "pond_r1_r2", name="Pond Peer Review")
    proj.add_model(model)

    # --- R1 Audit ---
    manifest = _build_pond_claim_manifest()
    r1 = proj.audit_claims(manifest=manifest)
    v_r1 = {v.claim_id: v for v in r1.verdicts}

    assert v_r1["C1_surface_energy"].overall_verdict == ClaimVerdictKind.SUPPORTED
    assert v_r1["C2_euler_vapor_law"].overall_verdict == ClaimVerdictKind.FALSIFIED
    assert v_r1["C2_euler_vapor_law"].metrics["log10_gap"] > 12.0
    assert v_r1["C3_coarse_n12_accuracy"].overall_verdict == ClaimVerdictKind.SCHEME_DEPENDENT
    assert v_r1["C5_peat_purification"].overall_verdict == ClaimVerdictKind.INCONCLUSIVE
    assert v_r1["C5_peat_purification"].assessment.mechanistic_status == VerificationStatus.PASSED
    assert v_r1["C5_peat_purification"].assessment.empirical_status == VerificationStatus.NOT_ASSESSED

    # Verify 3 deterministic clarification requests emitted (C2, C3, C5)
    req_by_claim = {r.claim_id: r for r in r1.open_requests}
    assert {"C2_euler_vapor_law", "C3_coarse_n12_accuracy", "C5_peat_purification"} == set(req_by_claim)

    # Referee report rendering
    rep_r1 = ReportGenerator.from_claim_round(proj, r1, report_type="peer_review_letter")
    md_r1 = rep_r1.to_markdown()
    assert "Executive Claim Audit Scorecard (R1)" in md_r1
    assert "Referee Clarification & Data Requests (R1)" in md_r1

    # --- R2A Positive Rebuttal (author declares RK4 + uploads pure-ice replicate CSV) ---
    good_csv = proj.root / "data" / "experimental" / "peat_assay_pure.csv"
    good_csv.write_text(
        "sample_id,solute_rejection_pct,u_solute_rejection_pct\n"
        "S1,96.8,1.1\n"
        "S2,97.4,0.9\n"
        "S3,97.1,1.0\n",
        encoding="utf-8",
    )
    r2_good = proj.submit_rebuttal(
        [
            RebuttalSubmission(
                request_id=req_by_claim["C3_coarse_n12_accuracy"].request_id,
                claim_id="C3_coarse_n12_accuracy",
                response_type="solver_metadata",
                declared_scheme="rk4",
                refinement_dts=[6000.0, 3000.0, 1500.0],
                author_note="Used classic 4th-order RK4.",
            ),
            RebuttalSubmission(
                request_id=req_by_claim["C5_peat_purification"].request_id,
                claim_id="C5_peat_purification",
                response_type="dataset_upload",
                dataset_path="data/experimental/peat_assay_pure.csv",
                author_note="Triplicate TOC/conductivity rejection assays.",
            ),
        ]
    )
    v_r2 = {v.claim_id: v for v in r2_good.verdicts}
    assert v_r2["C3_coarse_n12_accuracy"].overall_verdict == ClaimVerdictKind.SUPPORTED
    assert v_r2["C5_peat_purification"].overall_verdict == ClaimVerdictKind.SUPPORTED
    assert any(t.claim_id == "C5_peat_purification" and t.prev_verdict == "Inconclusive" and t.new_verdict == "Supported" for t in r2_good.transitions)

    # --- R2B Negative Rebuttal (separate workspace: contaminated ice CSV falsifies Claim 5) ---
    proj_neg = Project.create(tmp_path / "pond_neg", name="Pond Negative Rebuttal")
    proj_neg.add_model(model)
    r1_neg = proj_neg.audit_claims(manifest=_build_pond_claim_manifest())
    req_c5 = next(r for r in r1_neg.open_requests if r.claim_id == "C5_peat_purification")
    bad_csv = proj_neg.root / "data" / "experimental" / "peat_assay_contaminated.csv"
    bad_csv.write_text(
        "sample_id,solute_rejection_pct,u_solute_rejection_pct\n"
        "S1,34.0,2.5\n"
        "S2,36.2,2.1\n"
        "S3,35.1,2.3\n",
        encoding="utf-8",
    )
    r2_bad = proj_neg.submit_rebuttal(
        [
            RebuttalSubmission(
                request_id=req_c5.request_id,
                claim_id="C5_peat_purification",
                response_type="dataset_upload",
                dataset_path="data/experimental/peat_assay_contaminated.csv",
            )
        ]
    )
    v_bad = {v.claim_id: v for v in r2_bad.verdicts}
    assert v_bad["C5_peat_purification"].overall_verdict == ClaimVerdictKind.FALSIFIED


# ---------------------------------------------------------------------------
# Security, Path-Confinement & Anti-Override Integrity Suite
# ---------------------------------------------------------------------------


def test_security_ast_injection_path_traversal_and_anti_override(tmp_path: Path):
    # 1. Malicious AST payload rejected in ClaimSpec
    with pytest.raises(ValueError):
        ClaimSpec(
            id="C_bad",
            title="Injection",
            kind=ClaimKind.EQUATION_LAW,
            equations=["__import__('os').system('echo pwn') = 0"],
        )

    # 2. Path traversal ('..') rejected in RebuttalSubmission
    with pytest.raises(ValueError, match="traversal"):
        RebuttalSubmission(
            request_id="CR-R1-001",
            claim_id="C5",
            response_type="dataset_upload",
            dataset_path="../../etc/passwd",
        )

    # 3. Anti-override: Supported ClaimVerdict without evidence_run_ids rejected
    with pytest.raises(ValueError, match="Anti-override"):
        ClaimVerdict(
            claim_id="C_forged",
            title="Forged",
            kind=ClaimKind.EQUATION_LAW,
            assessment=TwoAxisAssessment(mechanistic_status=VerificationStatus.PASSED),
            evidence_run_ids=[],
        )

    # 4. Anti-override: Forged overall_verdict in deserialized dict rejected
    with pytest.raises(ValueError, match="Anti-override"):
        ClaimVerdict.from_dict(
            {
                "claim_id": "C_forged",
                "title": "Forged",
                "kind": "qualitative_observation",
                "overall_verdict": "Supported",
                "assessment": {
                    "mechanistic_status": "Passed",
                    "empirical_status": "NotAssessed",
                    "requires_empirical": True,
                },
                "evidence_run_ids": ["fake_001"],
            }
        )

    # 5. Anti-override: Ledger check rejects fabricated run_id
    v_fake = ClaimVerdict(
        claim_id="C_fake_run",
        title="Fake run",
        kind=ClaimKind.EQUATION_LAW,
        assessment=TwoAxisAssessment(mechanistic_status=VerificationStatus.PASSED),
        evidence_run_ids=["nonexistent_999"],
    )
    with pytest.raises(ValueError, match="not found in Project execution ledger"):
        v_fake.verify_against_ledger([])


# ---------------------------------------------------------------------------
# CLI (`rosolver claim ...`) & MCP (`13` Tools) Integration Tests
# ---------------------------------------------------------------------------


def test_cli_claim_workflow_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    proj_dir = tmp_path / "cli_claim_proj"
    assert cli_main(["init", str(proj_dir), "--model", str(POND_MODEL_YAML), "--json"]) == EXIT_OK
    capsys.readouterr()

    manifest_path = tmp_path / "claims.yaml"
    _build_pond_claim_manifest().save(manifest_path)

    assert cli_main(["claim", "init", str(proj_dir), "--manifest", str(manifest_path), "--json"]) == EXIT_OK
    init_out = json.loads(capsys.readouterr().out)
    assert init_out["n_claims"] == 4

    # R1 audit exits with EXIT_CHECK_FAILED because C2 is Falsified and C5 is Inconclusive
    rc = cli_main(["claim", "audit", str(proj_dir), "--output", "reports/referee_r1.md", "--json"])
    assert rc == EXIT_CHECK_FAILED
    audit_out = json.loads(capsys.readouterr().out)
    assert audit_out["round_id"] == "R1"
    assert (proj_dir / "reports" / "referee_r1.md").exists()
    assert str(Path.home().parent) not in (proj_dir / "reports" / "referee_r1.md").read_text(encoding="utf-8")

    # Questions listing
    assert cli_main(["claim", "questions", str(proj_dir), "--severity", "major", "--json"]) == EXIT_OK
    q_out = json.loads(capsys.readouterr().out)
    assert q_out["count"] == 3


def test_mcp_sprint10_tools_resource_and_prompt(tmp_path: Path):
    assert len(TOOL_SPECIFICATIONS) == 15
    srv = MCPServer()

    # 1. rosolver_check_scales (Falsifies 10^13 Euler gap)
    resp = srv.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "rosolver_check_scales",
                "arguments": {
                    "mode": "scales",
                    "equation": "rho * a + grad_P = 0",
                    "dims": {"rho": "kg/m^3", "a": "m/s^2", "grad_P": "Pa/m"},
                    "operating_envelope": {"rho": [917, 917], "a": [7e-14, 8e-14], "grad_P": [885, 8850]},
                },
            },
        }
    )
    sc_data = resp["result"]["structuredContent"]
    assert sc_data["status"] == "Failed"
    assert sc_data["log10_gap"] > 12.0

    # 2. rosolver_audit_claims + rosolver_submit_rebuttal over MCP
    proj = Project.create(tmp_path / "mcp_claim_proj", name="MCP Claim Audit")
    proj.add_model(Model.from_file(POND_MODEL_YAML))
    r1_resp = srv.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "rosolver_audit_claims",
                "arguments": {
                    "project_dir": str(proj.root),
                    "manifest_dict": _build_pond_claim_manifest().to_dict(),
                },
            },
        }
    )
    r1_sc = r1_resp["result"]["structuredContent"]
    assert r1_sc["round_id"] == "R1"
    assert len(r1_sc["open_requests"]) == 3

    # 3. Resource & Prompt check
    res_resp = srv.handle_message(
        {"jsonrpc": "2.0", "id": 3, "method": "resources/read", "params": {"uri": "rosolver://schema/claims"}}
    )
    assert "study_title:" in res_resp["result"]["contents"][0]["text"]

    pr_resp = srv.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "prompts/get",
            "params": {"name": "peer_review_manuscript", "arguments": {"manuscript_excerpt": "rho*a + grad_P = 0"}},
        }
    )
    assert "rosolver_check_scales" in pr_resp["result"]["messages"][0]["content"]["text"]
