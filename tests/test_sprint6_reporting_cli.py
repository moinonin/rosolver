"""Sprint 6 tests: Scientific reporting engine (P3.1), workflow CLI & Jupyter hooks (P3.2)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from rosolver import (
    REPORT_TYPES,
    ConditionSpec,
    ConservationLaw1D,
    Dataset,
    EquationSpec,
    Model,
    ParameterSpec,
    Project,
    ReportGenerator,
    ScientificReport,
    StandardResult,
    ValidationStudy,
    VariableSpec,
    VerificationStatus,
    audit_pde_conservation,
    formulate,
    gci_from_values,
)
from rosolver.cli import EXIT_CHECK_FAILED, EXIT_OK, EXIT_VALIDATION_ERROR, main as cli_main
from rosolver.numerical import NumericalSolver


def _heat_pde_model() -> Model:
    return Model(
        name="heat_1d",
        version="1.0",
        description="1D transient heat conduction test model",
        domain="heat_transfer",
        independent_variables=[
            VariableSpec(name="x", unit="m", role="independent"),
            VariableSpec(name="t", unit="s", role="independent"),
        ],
        dependent_variables=[
            VariableSpec(name="T", unit="K", role="dependent"),
        ],
        parameters=[
            ParameterSpec(name="alpha", unit="m^2/s", value=0.05, status="fixed"),
        ],
        equations=[
            EquationSpec(id="heat_eq", type="pde", expression="dT/dt = alpha * d2T/dx2"),
        ],
        initial_conditions=[
            ConditionSpec(kind="initial", field="T", expression="T(x,0) = sin(pi * x)"),
        ],
        boundary_conditions=[
            ConditionSpec(kind="dirichlet", location="left", field="T", value=0.0),
            ConditionSpec(kind="dirichlet", location="right", field="T", value=0.0),
        ],
    )


def _decay_ode_model() -> Model:
    return Model(
        name="exp_decay",
        version="0.2",
        description="First-order decay ODE + algebraic calibration model",
        domain="kinetics",
        independent_variables=[VariableSpec(name="t", unit="s", role="independent")],
        dependent_variables=[VariableSpec(name="y", unit="1", role="dependent")],
        parameters=[ParameterSpec(name="k", unit="1/s", value=0.5, status="fixed")],
        equations=[EquationSpec(id="decay_ode", type="ode", expression="dy/dt = -k * y")],
        initial_conditions=[ConditionSpec(kind="initial", field="y", value=2.0)],
        assumptions=["Linear first-order decay without external forcing."],
    )


def _constitutive_calib_model() -> Model:
    return Model(
        name="linear_flux",
        version="0.1",
        independent_variables=[VariableSpec(name="x", unit="m", role="independent")],
        dependent_variables=[VariableSpec(name="y", unit="m", role="dependent")],
        parameters=[ParameterSpec(name="m", unit="1", status="unknown")],
        equations=[EquationSpec(id="law", type="algebraic", expression="y = m * x")],
        dependencies=["y = fn(x)"],
    )


# ---------------------------------------------------------------------------
# P3.1 ScientificReport & ReportGenerator
# ---------------------------------------------------------------------------


def test_report_types_cover_canonical_scopes():
    assert len(REPORT_TYPES) == 10
    assert set(REPORT_TYPES) == {
        "symbolic_derivation",
        "dimensional_analysis",
        "calibration",
        "numerical_simulation",
        "code_verification",
        "solution_verification",
        "model_validation",
        "integrated_project",
        "claim_verification",
        "peer_review_letter",
    }


def test_report_from_result_and_save_formats(tmp_path: Path):
    model = _heat_pde_model()
    sim_res = model.solve(t_span=(0.0, 0.2), n_cells=24)

    rep = ReportGenerator.from_result(
        sim_res,
        model=model,
        title="Heat Conduction Single Run",
        figures=[{"path": "reports/figures/temp_profile.png", "caption": "Temperature profile"}],
    )
    assert isinstance(rep, ScientificReport)
    assert rep.report_type == "numerical_simulation"
    assert rep.status == VerificationStatus.PASSED
    assert rep.figures[0]["path"] == "reports/figures/temp_profile.png"

    md = rep.to_markdown()
    assert "# Heat Conduction Single Run" in md
    assert "`heat_1d`" in md
    assert "dT/dt = alpha * d2T/dx2" in md
    assert "ASME V&V 20" in md
    assert "![Temperature profile](reports/figures/temp_profile.png)" in md

    md_path = rep.save(tmp_path / "study.md", also_manifest=True)
    yaml_path = rep.save(tmp_path / "study.yaml")
    json_path = rep.save(tmp_path / "study.json")
    assert md_path.exists()
    assert (tmp_path / "study.manifest.json").exists()
    assert yaml.safe_load(yaml_path.read_text(encoding="utf-8"))["report_type"] == "numerical_simulation"
    assert json.loads(json_path.read_text(encoding="utf-8"))["status"] == VerificationStatus.PASSED.value


def test_report_surfaces_failed_and_inconclusive_findings_prominently(tmp_path: Path):
    # Oscillatory/divergent GCI -> INCONCLUSIVE or FAILED status
    bad_gci = gci_from_values([0.01, 0.02, 0.04], [1.0, 1.5, 1.2])
    rep = ReportGenerator.from_result(bad_gci, title="Non-Monotonic Grid Check")
    assert rep.status in (VerificationStatus.INCONCLUSIVE, VerificationStatus.FAILED)

    md = rep.to_markdown()
    assert f"ATTENTION ({rep.status.value.upper()})" in md
    assert rep.status.value.upper() in md

    # Project report preserves tampered hash / failed validation without mutating simulation artifacts
    proj = Project.create(tmp_path / "proj", name="Tamper Check")
    m = _heat_pde_model()
    proj.add_model(m)
    sim_res = proj.run_simulation(t_span=(0.0, 0.1), n_cells=16)
    out_file = tmp_path / "proj" / proj.ledger()[-1].outputs["result"]
    before_bytes = out_file.read_bytes()

    # Negative validation that fails
    vrep = ValidationStudy.compare(
        simulation=[1.0, 2.0],
        experiment=[5.0, 7.0],
        field="T",
        unit="K",
        u_exp=[0.1, 0.1],
    )
    assert vrep.status == VerificationStatus.FAILED
    proj.record_run("model_validation", vrep)

    integrated = proj.report(output="reports/study.md")
    assert integrated.status == VerificationStatus.FAILED
    assert "FAILED" in integrated.to_markdown()
    assert (tmp_path / "proj" / "reports" / "study.md").exists()
    # Original simulation JSON output was never mutated
    assert out_file.read_bytes() == before_bytes
    assert len(proj._manifest["reports"]) == 1


# ---------------------------------------------------------------------------
# P3.2 Jupyter _repr_markdown_ hooks (zero IPython dependency)
# ---------------------------------------------------------------------------


def test_jupyter_markdown_repr_hooks_pure_strings():
    import sympy as sp

    sym_res = formulate(["y = fn(x)"], target="y", wrt="x")
    assert "Symbolic Derivation" in sym_res._repr_markdown_()

    x = np.linspace(1.0, 5.0, 5)
    y_sym, a_sym, x_sym = sp.symbols("y a x", real=True)
    fit_res = NumericalSolver.fit(
        sp.Eq(y_sym, a_sym * x_sym),
        {"x": x, "y": 2.5 * x},
        free_params=[a_sym],
        target_col="y",
    )
    assert "Parameter Calibration" in fit_res._repr_markdown_()

    gci_rep = gci_from_values([0.01, 0.02, 0.04], [1.0001, 1.0004, 1.0016])
    assert "Grid Convergence Index" in gci_rep._repr_markdown_()

    v_rep = ValidationStudy.compare([1.0, 2.0], [1.05, 1.98], field="T", unit="K", u_exp=0.2)
    assert "Experimental Validation" in v_rep._repr_markdown_()

    law = ConservationLaw1D(fields="u", flux=0, diffusivity=0.05)
    sol = law.solve(domain=(0.0, 1.0), n_cells=20, t_end=0.1, initial="sin(pi*x)", bc="outflow")
    cons = audit_pde_conservation(sol)
    assert "Conservation Audit" in cons._repr_markdown_()

    sr = _heat_pde_model().solve(t_span=(0.0, 0.05), n_cells=12)
    assert "Rosolver `pde_simulation`" in sr._repr_markdown_()

    rep = ReportGenerator.from_result(sr)
    assert rep._repr_markdown_() == rep.to_markdown()


# ---------------------------------------------------------------------------
# P3.2 Workflow CLI subcommands + legacy backwards compatibility
# ---------------------------------------------------------------------------


def test_cli_workflow_end_to_end_subcommands(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    proj_dir = tmp_path / "heat_cli"
    model_file = tmp_path / "heat_model.yaml"
    _heat_pde_model().save(model_file)

    # 1. init
    rc = cli_main(["init", str(proj_dir), "--name", "Heat CLI", "--model", str(model_file), "--json"])
    assert rc == EXIT_OK
    init_out = json.loads(capsys.readouterr().out)
    assert init_out["project"]["n_models"] == 1

    # 2. info
    rc = cli_main(["info", str(proj_dir), "--json"])
    assert rc == EXIT_OK
    info_out = json.loads(capsys.readouterr().out)
    assert info_out["name"] == "Heat CLI"

    # 3. validate (valid project and standalone model)
    rc = cli_main(["validate", str(proj_dir), "--json"])
    assert rc == EXIT_OK
    assert json.loads(capsys.readouterr().out)["status"] == "valid"

    rc = cli_main(["validate", str(model_file), "--json"])
    assert rc == EXIT_OK
    capsys.readouterr()

    # 4. simulate
    rc = cli_main(["simulate", str(proj_dir), "--t-end", "0.1", "--n-cells", "20", "--json"])
    assert rc == EXIT_OK
    sim_out = json.loads(capsys.readouterr().out)
    assert sim_out["kind"] == "pde_simulation"
    assert sim_out["status"] == VerificationStatus.PASSED.value

    # 5. verify (conservation & GCI)
    rc = cli_main(["verify", str(proj_dir), "--method", "conservation", "--t-end", "0.1", "--json"])
    assert rc == EXIT_OK
    ver_out = json.loads(capsys.readouterr().out)
    assert ver_out["status"] == VerificationStatus.PASSED.value

    rc = cli_main(["verify", str(proj_dir), "--gci", "0.01,0.02,0.04:1.0001,1.0004,1.0016", "--json"])
    assert rc == EXIT_OK
    capsys.readouterr()

    # 6. validate-results against matching experimental CSV + failing check
    exp_ok = tmp_path / "exp_ok.csv"
    proj = Project.open(proj_dir)
    last_sim_id = [r.id for r in proj.ledger() if r.type == "pde_simulation"][-1]
    res_json = json.loads((proj.root / proj.get_run(last_sim_id).outputs["result"]).read_text(encoding="utf-8"))
    x_pts = np.asarray(res_json["result"]["x"], dtype=float)
    t_vals = np.asarray(res_json["result"]["U"]["T"], dtype=float)[-1]
    lines_csv = ["x,T,u_T"] + [f"{xv:.8g},{tv:.8g},0.05" for xv, tv in zip(x_pts[::4], t_vals[::4])]
    exp_ok.write_text("\n".join(lines_csv), encoding="utf-8")

    rc = cli_main(
        [
            "validate-results",
            str(proj_dir),
            "--run-id",
            last_sim_id,
            "--exp-csv",
            str(exp_ok),
            "--field",
            "T",
            "--coord",
            "x",
            "--u-num",
            "0.01",
            "--u-exp",
            "u_T",
            "--json",
        ]
    )
    assert rc == EXIT_OK
    vr_out = json.loads(capsys.readouterr().out)
    assert vr_out["status"] == VerificationStatus.PASSED.value

    # 7. report
    rc = cli_main(
        [
            "report",
            str(proj_dir),
            "--output",
            "reports/cli_report.md",
            "--manifest",
            "--json",
        ]
    )
    assert rc == EXIT_OK
    rep_out = json.loads(capsys.readouterr().out)
    assert rep_out["status"] == VerificationStatus.PASSED.value
    assert (proj_dir / "reports" / "cli_report.md").exists()
    assert (proj_dir / "reports" / "cli_report.manifest.json").exists()


def test_cli_model_analyse_calibrate_and_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    proj_dir = tmp_path / "cal_cli"
    model_file = tmp_path / "lin.yaml"
    _constitutive_calib_model().save(model_file)

    assert cli_main(["init", str(proj_dir), "--model", str(model_file), "--json"]) == EXIT_OK
    capsys.readouterr()

    # model analyse
    rc = cli_main(["model", "analyse", str(proj_dir), "--target", "y", "--wrt", "x", "--json"])
    assert rc == EXIT_OK
    ma_out = json.loads(capsys.readouterr().out)
    assert ma_out["kind"] == "symbolic_derivation"

    # calibrate
    cal_csv = tmp_path / "cal_data.csv"
    cal_csv.write_text("x,y,sigma_y\n1,2.0,0.05\n2,4.0,0.05\n3,6.0,0.05\n4,8.0,0.05\n", encoding="utf-8")
    rc = cli_main(
        [
            "calibrate",
            str(proj_dir),
            "--data",
            str(cal_csv),
            "--uncertainty-col",
            "sigma_y",
            "--json",
        ]
    )
    assert rc == EXIT_OK
    cal_out = json.loads(capsys.readouterr().out)
    assert cal_out["kind"] == "calibration"
    assert cal_out["result"]["params"]["m"] == pytest.approx(2.0, rel=1e-4)

    # Corrupt model hash -> validate returns EXIT_VALIDATION_ERROR (2)
    proj = Project.open(proj_dir)
    m_path = proj.root / proj.models[0].path
    m_path.write_text(m_path.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")
    rc_bad = cli_main(["validate", str(proj_dir), "--json"])
    assert rc_bad == EXIT_VALIDATION_ERROR
    assert json.loads(capsys.readouterr().out)["status"] == "invalid"

    # Non-monotonic GCI via CLI verify -> EXIT_CHECK_FAILED (1)
    rc_fail = cli_main(["verify", str(proj_dir), "--gci", "0.01,0.02,0.04:1.0,1.5,1.2", "--json"])
    assert rc_fail == EXIT_CHECK_FAILED
    capsys.readouterr()


def test_cli_preserves_legacy_flat_flags(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    plot_path = tmp_path / "legacy_gci.png"
    rc = cli_main(["--gci", "0.01,0.02,0.04:1.0001,1.0004,1.0016", "--gci-plot", str(plot_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Grid Convergence Index (GCI) Report" in out
    assert plot_path.exists()

    rc2 = cli_main(["--rel", "y = fn(x)", "--target", "y", "--wrt", "x"])
    assert rc2 == 0
    out2 = capsys.readouterr().out
    assert "Equation Formulation" in out2
