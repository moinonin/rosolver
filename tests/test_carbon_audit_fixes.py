"""Regression test suite verifying all 26 fixes from CARBON_AUDIT.md (CA-01 through CA-26)."""

from __future__ import annotations

import io
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from rosolver import (
    ClaimKind,
    ClaimManifest,
    ClaimSpec,
    ClaimVerdictKind,
    ConditionSpec,
    ConservationLaw1D,
    Dataset,
    Dimension,
    EquationSpec,
    Model,
    ModelValidationError,
    NumericalSolver,
    ODESystem,
    ParameterSpec,
    Project,
    ProjectError,
    RebuttalSubmission,
    SheddingEventSpec,
    Stefan1D,
    System,
    ValidationStudy,
    VariableSpec,
    VerificationStatus,
    audit_claimed_accuracy,
    audit_hybrid_conservation,
    check_homogeneity_elasticities,
    cosimulate_coupled_odes,
    gci_from_values,
    max_stable_dt,
    plot_gci_study,
    plot_ranked,
    simulate,
    standardize_result,
    stefan_shedding_event,
)
from rosolver._parsing import parse_expr
from rosolver.cli import EXIT_OK, EXIT_VALIDATION_ERROR, main as cli_main
from rosolver.dimensions import convert_unit_values
from rosolver.mcp.server import _read_message


# ---------------------------------------------------------------------------
# Batch 1: Security, CSV Row Alignment, MCP Framing & Project Confinement
# ---------------------------------------------------------------------------


def test_ca07_ca08_exponent_dos_and_mcp_content_length_guards():
    # CA-07: Giant numeric exponent blocked in AST parser and Dimension parser
    with pytest.raises(ValueError, match="Exponent magnitude"):
        parse_expr("10 ** 999999999")
    with pytest.raises(ValueError, match="exceeds allowed rational bounds"):
        Dimension.parse("L^999999")

    # CA-08: Negative and oversized Content-Length rejected; chunked stream read succeeds
    with pytest.raises(ValueError, match="out of allowed bounds"):
        _read_message(io.BytesIO(b"Content-Length: -1\r\n\r\n{}"))
    with pytest.raises(ValueError, match="out of allowed bounds"):
        _read_message(io.BytesIO(b"Content-Length: 999999999\r\n\r\n{}"))

    valid_frame = b'Content-Length: 17\r\n\r\n{"jsonrpc":"2.0"}'
    body, framed = _read_message(io.BytesIO(valid_frame))
    assert framed is True
    assert body == b'{"jsonrpc":"2.0"}'


def test_ca06_ca23_ca25_csv_row_alignment_project_confinement_and_overwrite(tmp_path: Path):
    # CA-06 & CA-25: Blank CSV cells must raise rather than silently misaligning columns
    bad_csv = tmp_path / "jagged.csv"
    bad_csv.write_text("x,y\n1.0,\n2.0,20.0\n,30.0\n", encoding="utf-8")

    proj = Project.create(tmp_path / "proj_hygiene", name="Hygiene")
    m = Model(
        name="lin",
        independent_variables=[VariableSpec("x", role="independent", unit="m")],
        dependent_variables=[VariableSpec("y", role="dependent", unit="m")],
        parameters=[ParameterSpec("a", unit="1", status="unknown")],
        equations=[EquationSpec("e1", type="algebraic", expression="y = a * x")],
    )
    proj.add_model(m)
    proj.add_dataset(bad_csv, name="jagged")
    with pytest.raises(ValueError, match="Missing value"):
        proj.run_calibration(dataset="jagged", target="y")

    # CA-23: Non-CSV (.json) calibration dataset works in Project.run_calibration
    json_ds = tmp_path / "clean.json"
    json_ds.write_text(
        json.dumps({"coordinates": {"x": [1.0, 2.0, 3.0]}, "fields": {"y": [2.0, 4.0, 6.0]}}),
        encoding="utf-8",
    )
    proj.add_dataset(json_ds, name="clean_json")
    cal_sr = proj.run_calibration(dataset="clean_json", target="y")
    assert cal_sr.raw.params["a"] == pytest.approx(2.0)

    # CA-23: Path traversal in add_model(filename=...) blocked
    with pytest.raises(ProjectError, match="escapes"):
        proj.add_model(m, filename="../escaped.yaml")

    # CA-23: Project.create(..., overwrite=True) clears stale ledger entries
    assert len(proj.ledger()) >= 1
    proj_fresh = Project.create(tmp_path / "proj_hygiene", name="Reborn", overwrite=True)
    assert len(proj_fresh.ledger()) == 0


# ---------------------------------------------------------------------------
# Batch 2: Dimensions, SI Units/Prefixes, ValidationStudy & Model Validation
# ---------------------------------------------------------------------------


def test_ca01_ca02_ca03_ca21_units_prefixes_degc_nan_and_extrapolation():
    # CA-01: Strict unit validation rejects unknown unit identifiers; SI prefixes & unit= work
    with pytest.raises(ValueError, match="Unknown physical unit"):
        Dimension.from_unit("banana")
    assert Dimension.from_unit("kPa") == Dimension.from_unit("Pa")
    assert Dimension.from_unit("mm") == Dimension.from_unit("m")
    assert np.isclose(convert_unit_values(101.325, "kPa", "Pa"), 101325.0)
    assert np.isclose(convert_unit_values(0.0, "degC", "K"), 273.15)

    s = System()
    v_e = s.var("E", unit="J")
    v_f = s.var("F", unit="N")
    assert v_e.dim == Dimension.parse("M*L^2/T^2")
    assert v_f.dim == Dimension.parse("M*L/T^2")
    with warnings.catch_warnings(record=True) as w_rec:
        warnings.simplefilter("always")
        s.var("N_ambig", "N")
        assert any("Ambiguous dimension" in str(item.message) for item in w_rec)

    # Verify scales._parse_dim treats "T" / "1/T" as Time base dimension, not Tesla
    hom = check_homogeneity_elasticities("T", {"H": "1/T", "L_char": "L"}, {"H": -1.0, "L_char": 0.0})
    assert hom.zero_intercept_admissible is True

    # CA-02: ValidationStudy converts K -> degC and kPa -> Pa automatically
    sim_ds = Dataset(name="sim", coordinates={"x": [0.0, 1.0]}, fields={"T": [273.15, 283.15]}, units={"x": "m", "T": "K"})
    exp_ds = Dataset(
        name="exp",
        coordinates={"x": [0.0, 1.0]},
        fields={"T": [0.0, 10.0]},
        uncertainties={"T": [0.2, 0.2]},
        units={"x": "m", "T": "degC"},
    )
    v_rep = ValidationStudy.from_datasets(sim_ds, exp_ds, field="T", u_num=0.1)
    assert v_rep.status == VerificationStatus.PASSED
    assert v_rep.max_abs_error == pytest.approx(0.0, abs=1e-12)

    # CA-03: NaN in simulation or experiment raises ValueError instead of returning PASSED
    with pytest.raises(ValueError, match="finite"):
        ValidationStudy.compare([float("nan")], [1.0], error_tol=0.01)

    # CA-21: Out-of-domain experimental coordinates trigger WARNING instead of silent PASSED
    exp_oob = Dataset(
        name="exp_oob",
        coordinates={"x": [0.0, 2.5]},
        fields={"T": [0.0, 10.0]},
        uncertainties={"T": [0.5, 0.5]},
        units={"x": "m", "T": "degC"},
    )
    v_oob = ValidationStudy.from_datasets(sim_ds, exp_oob, field="T", u_num=0.1)
    assert v_oob.status == VerificationStatus.WARNING
    assert any("outside simulation domain" in w for w in v_oob.warnings)


def test_ca10_model_dimensional_validation_and_conservative_pde_guard():
    # Dimensional mismatch in ODE caught at Model construction
    with pytest.raises(ModelValidationError, match="Dimensional"):
        Model(
            name="bad_units",
            independent_variables=[VariableSpec("t", role="independent", unit="s")],
            dependent_variables=[VariableSpec("y", role="dependent", unit="m")],
            parameters=[ParameterSpec("k", value=1.0, unit="kg", status="fixed")],
            equations=[EquationSpec("e1", type="ode", expression="dy/dt = k + y")],
        )

    # Non-divergence spatially-varying velocity `-x * du/dx` rejected instead of corrupting to `d/dx(x*u)`
    with pytest.raises(ModelValidationError, match="non-divergence advection"):
        Model(
            name="bad_adv",
            independent_variables=[
                VariableSpec("x", role="independent", unit="m"),
                VariableSpec("t", role="independent", unit="s"),
            ],
            dependent_variables=[VariableSpec("u", role="dependent", unit="1")],
            equations=[EquationSpec("e1", type="pde", expression="du/dt = -x * du/dx")],
        )


# ---------------------------------------------------------------------------
# Batch 3: WLS BIC/RMSE, Zero-DoF Guard, Symbol Matching & Missing Policy
# ---------------------------------------------------------------------------


def test_ca05_ca16_ca17_ca26_wls_bic_zero_dof_symbols_mask_and_simulate(tmp_path: Path):
    # CA-05 (Part A): Zero-DoF saturated fit sets BIC=inf and tier='underdetermined'
    x_small = np.array([1.0, 2.0])
    y_small = np.array([3.0, 5.0])
    eq_lin = sp.Eq(sp.Symbol("y", real=True), sp.Symbol("a", real=True) * sp.Symbol("x", real=True) + sp.Symbol("b", real=True))
    fit_0dof = NumericalSolver.fit(eq_lin, {"x": x_small, "y": y_small}, free_params=["a", "b"])
    assert np.isinf(fit_0dof.bic)
    assert fit_0dof.admissibility_tier == "underdetermined"

    # CA-05 (Part B): WLS RMSE/BIC downweights noisy outlier properly
    x_w = np.linspace(1.0, 5.0, 8)
    y_w = 2.0 * x_w.copy()
    sigma = np.full_like(x_w, 0.01)
    y_w[-1] += 50.0  # huge outlier with huge uncertainty sigma=1000
    sigma[-1] = 1000.0
    eq_1p = sp.Eq(sp.Symbol("y", real=True), sp.Symbol("a", positive=True, real=True) * sp.Symbol("x", real=True))
    # CA-16: Pass sp.Symbol("a", real=True) (different assumptions from positive=True in eq_1p!)
    fit_w = NumericalSolver.fit(eq_1p, {"x": x_w, "y": y_w}, free_params=[sp.Symbol("a", real=True)], sigma_y=sigma)
    assert fit_w.params["a"] == pytest.approx(2.0, rel=1e-3)
    assert fit_w.r2 > 0.999

    # CA-17: missing_policy='mask' preserves full array length with NaN at missing positions, and plot_ranked works
    x_m = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_m = np.array([2.0, np.nan, 6.0, 8.0, 10.0])
    fit_mask = NumericalSolver.fit(eq_1p, {"x": x_m, "y": y_m}, free_params=["a"], missing_policy="mask")
    assert fit_mask.predictions.shape == (5,)
    assert np.isnan(fit_mask.predictions[1])
    fit_drop = NumericalSolver.fit(eq_1p, {"x": x_m, "y": y_m}, free_params=["a"], missing_policy="drop")
    assert fit_drop.predictions.shape == (4,)
    fig_p = plot_ranked([fit_mask, fit_drop], {"x": x_m, "y": y_m}, "x", "y", path=tmp_path / "ranked.png")
    assert fig_p is not None and fig_p.exists()

    # CA-26: simulate() accepts SpecModel and standardize_result merges overrides on existing StandardResult
    spec_m = Model(
        name="alg_m",
        independent_variables=[VariableSpec("x", role="independent", unit="m")],
        dependent_variables=[VariableSpec("y", role="dependent", unit="m")],
        parameters=[ParameterSpec("k", value=3.0, unit="1", status="fixed")],
        equations=[EquationSpec("e1", type="algebraic", expression="y = k * x")],
    )
    sim_res = simulate(spec_m, sweep={"x": (0.0, 2.0, 5)})
    assert np.allclose(sim_res.curves["y"], 3.0 * np.linspace(0.0, 2.0, 5))
    sr1 = standardize_result(fit_w, kind="calibration")
    sr2 = standardize_result(sr1, run_id="cal_099", model_name="updated_m")
    assert sr2.run_id == "cal_099"
    assert sr2.metadata.model_name == "updated_m"


# ---------------------------------------------------------------------------
# Batch 4: Integrators, Solvers, V&V, Claims & CLI Fixes
# ---------------------------------------------------------------------------


def test_ca04_ca09_ca11_ca12_ca13_ca19_ca20_ca22_solver_and_vv_fixes(tmp_path: Path):
    # CA-04: ssprk2 / euler on purely imaginary axis have exact dt_max == 0.0
    assert max_stable_dt("euler", [1j]) == 0.0
    assert max_stable_dt("ssprk2", [1j]) == 0.0
    assert max_stable_dt("rk4", [1j]) == pytest.approx(2.0 * np.sqrt(2.0), rel=1e-2)

    # CA-09: Explicit fixed-step blow-up to Inf/NaN sets success=False
    tr_blow = ODESystem({"y": "y**4"}).solve((0.0, 2.5), {"y": 10.0}, method="euler", dt=0.5)
    assert tr_blow.success is False

    # CA-19: Fixed-step ODESystem.solve honors t_eval
    tr_te = ODESystem({"y": "-y"}).solve((0.0, 1.0), {"y": 1.0}, method="rk4", dt=0.05, t_eval=[0.0, 0.25, 0.5, 1.0])
    assert np.allclose(tr_te.t, [0.0, 0.25, 0.5, 1.0])
    assert np.allclose(tr_te["y"], np.exp(-np.array([0.0, 0.25, 0.5, 1.0])), atol=1e-5)

    # CA-12: Scalar-branching time-dependent callable works in ODESystem.evaluate and cosimulate_coupled_odes
    ode_sw = ODESystem({"y": "-k * y"})
    k_fn = lambda t: 1.0 if t < 0.5 else 2.0
    tr_sw = ode_sw.solve((0.0, 1.0), {"y": 1.0}, params={"k": k_fn})
    vals_ky = ode_sw.evaluate("k * y", tr_sw, params={"k": k_fn})
    assert vals_ky.shape == tr_sw.t.shape

    sys_a = ODESystem({"u": "-u + 0.1 * v + f_ext"})
    sys_b = ODESystem({"v": "-2 * v + 0.1 * u"})
    cosim = cosimulate_coupled_odes(
        sys_a, sys_b, (0.0, 1.0), {"u": 1.0, "v": 0.5}, dt_macro=0.2, params={"f_ext": k_fn}
    )
    assert cosim.converged is True

    # CA-13: ConservationLaw1D(m=1) accepts list initial=[1.0] and reaction-only auto-dt
    pde_sol = ConservationLaw1D("u", source="-10.0 * u").solve((0.0, 1.0), 8, 0.2, initial=[1.0], scheme="rk4")
    assert pde_sol.success is True
    assert np.allclose(pde_sol.final("u"), np.exp(-2.0), rtol=1e-3)

    # CA-11 & CA-20: Stefan1D clamps zero-film evaporation (dh_f/dt >= 0 at h_f=0) and blocks shed_fraction=1.0
    st = Stefan1D(T_wall=-5.0, T_inf=-2.0, T_f=0.0, h_c=20.0, mdot_imp=1e-4, mdot_evap=5e-4, tau_drain=30.0, h_film0=0.0)
    dz0 = st.rhs(0.0, st.initial_state(1e-4))
    assert dz0[-1] >= -1e-15
    with pytest.raises(ValueError, match="shed_fraction=1.0"):
        st.solve_with_shedding([stefan_shedding_event(2e-5, shed_fraction=1.0)], 100.0, delta0=1e-5)

    # CA-20: Sub-nanosecond hybrid ODE shedding & large-magnitude conservation audit
    hyb_ns = NumericalSolver.integrate_hybrid_ode(
        parse_expr("1e17"),
        wrt="t",
        span=(0.0, 1e-8),
        y0=0.0,
        events=[SheddingEventSpec("burst", lambda _t, y: y - 3e8, shed_fraction=0.5)],
        dt_max=1e-9,
    )
    assert hyb_ns.n_events >= 2
    aud_ns = audit_hybrid_conservation(hyb_ns, lambda _t, _y: 1e17, tol=1e-4)
    assert aud_ns.passed is True

    # CA-22: Oscillatory GCI with formal_order mismatch fails + plot_gci_study handles divergent NaNs
    rep_osc_fail = gci_from_values([0.01, 0.02, 0.04], [1.001, 0.996, 1.016], formal_order=4.0, order_tol=0.5)
    assert rep_osc_fail.status == VerificationStatus.FAILED
    rep_div = gci_from_values([0.01, 0.02, 0.04], [1.5, 1.1, 1.05])
    assert plot_gci_study(rep_div, path=tmp_path / "div_gci.png").exists()


def test_ca14_ca15_ca18_ca24_claims_budget_and_cli_validate(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    # CA-15: Model with parameter-dependent expression IC works in audit_claimed_accuracy
    m_ic = Model(
        name="param_ic_ode",
        independent_variables=[VariableSpec("t", role="independent", unit="s")],
        dependent_variables=[VariableSpec("y", role="dependent", unit="m")],
        parameters=[
            ParameterSpec("k", value=0.5, unit="1/s", status="fixed"),
            ParameterSpec("y_init", value=4.0, unit="m", status="fixed"),
        ],
        equations=[EquationSpec("e1", type="ode", expression="dy/dt = -k * y")],
        initial_conditions=[ConditionSpec(kind="initial", field="y", expression="y(0) = y_init")],
    )
    eb = audit_claimed_accuracy(m_ic, claimed_rel_error=0.01, dt_paper=0.05, declared_scheme="rk4", t_span=(0.0, 1.0))
    assert eb.status == VerificationStatus.PASSED

    # CA-14 & CA-18: n=1 replicate CSV without u_exp returns WARNING (Inconclusive/PartiallySupported) without crashing,
    # and revised_equation automatically sets post_hoc_recalibration=True
    proj = Project.create(tmp_path / "claim_fixes", name="Claim Fixes")
    proj.add_model(m_ic)
    one_csv = proj.root / "data" / "experimental" / "single.csv"
    one_csv.write_text("obs\n95.0\n", encoding="utf-8")
    proj.add_dataset(one_csv, name="single_ds", copy=False)

    cm = ClaimManifest(
        study_title="Fixes",
        claims=[
            ClaimSpec(
                id="C_one",
                title="Single row",
                kind=ClaimKind.QUALITATIVE_OBSERVATION,
                expected_dataset="single_ds",
                required_observables=["obs"],
                claimed_metrics={"min_obs": 90.0},
            ),
            ClaimSpec(
                id="C_eq",
                title="Imbalanced eq",
                kind=ClaimKind.EQUATION_LAW,
                equations=["a + b = 0"],
                variables_units={"a": "m", "b": "m"},
                operating_envelope={"a": (1.0, 1.0), "b": (1e6, 1e6)},
            ),
        ],
    )
    r1 = proj.audit_claims(cm)
    v_map = {v.claim_id: v for v in r1.verdicts}
    assert v_map["C_one"].assessment.empirical_status == VerificationStatus.WARNING

    req_eq = next(r for r in r1.open_requests if r.claim_id == "C_eq")
    r2 = proj.submit_rebuttal(
        [
            RebuttalSubmission(
                request_id=req_eq.request_id,
                claim_id="C_eq",
                response_type="constitutive_equation",
                revised_equation="a - c = 0",
                revised_units={"c": "m"},
                revised_envelope={"c": (1.0, 1.0)},
            )
        ]
    )
    v2_eq = next(v for v in r2.verdicts if v.claim_id == "C_eq")
    assert v2_eq.post_hoc_recalibration is True

    # CA-24: CLI `rosolver validate <invalid_model.yaml> --json` emits structured JSON errors and exits 2
    bad_yaml = tmp_path / "invalid_model.yaml"
    bad_yaml.write_text(
        "model:\n  name: broken\nvariables:\n  independent:\n    - {name: t, unit: s}\n"
        "  dependent:\n    - {name: y, unit: m}\nequations:\n  - {id: e1, type: ode, expression: 'dy/dt = y'}\n",
        encoding="utf-8",
    )
    rc = cli_main(["validate", str(bad_yaml), "--json"])
    assert rc == EXIT_VALIDATION_ERROR
    out_json = json.loads(capsys.readouterr().out)
    assert out_json["status"] == "invalid"
    assert len(out_json["errors"]) >= 1
