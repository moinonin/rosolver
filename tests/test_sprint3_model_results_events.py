"""Sprint 3 tests: derived SI units, StandardResult, ODESystem events & validated Model spec (P1.1, P1.2, P1.4)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rosolver import (
    ConservationLaw1D,
    Dimension,
    EquationSpec,
    EventSpec,
    Model,
    ModelValidationError,
    NumericsConfig,
    ODESystem,
    ParameterSpec,
    StandardResult,
    Stefan1D,
    VariableSpec,
    VerificationStatus,
    audit_pde_conservation,
    gci_from_values,
    mms_study,
    standardize_result,
)


class TestDerivedSIUnits:
    def test_derived_si_units_and_base_dimension_compatibility(self) -> None:
        assert Dimension.parse("W/(m*K)") == Dimension.parse("M*L/(T^3*Theta)")
        assert Dimension.parse("Pa*s") == Dimension.parse("M/(L*T)")
        assert Dimension.from_unit("N/m^2") == Dimension.from_unit("Pa")
        assert Dimension.from_unit("J/(kg*K)") == Dimension.parse("L^2/(T^2*Theta)")
        assert Dimension.from_unit("V") == Dimension.parse("M*L^2/(T^3*I)")

        # Preserve canonical base dimension semantics in Dimension.parse(...)
        assert Dimension.parse("N") == Dimension.from_dict({"N": 1})
        assert Dimension.parse("J") == Dimension.from_dict({"J": 1})
        assert Dimension.parse("T") == Dimension.from_dict({"T": 1})


class TestStandardizedResults:
    def test_to_dict_and_to_json_across_subsystems(self) -> None:
        ode = ODESystem({"y": "-k * y"})
        traj = ode.solve((0.0, 1.0), {"y": 1.0}, params={"k": 1.0}, n_out=10)
        sr_ode = standardize_result(traj, model_name="decay", units={"y": "1", "t": "s"})
        assert isinstance(sr_ode, StandardResult)
        assert sr_ode.status == VerificationStatus.PASSED
        assert sr_ode.passed
        parsed_ode = json.loads(sr_ode.to_json())
        assert parsed_ode["kind"] == "Trajectory"
        assert parsed_ode["metadata"]["units"]["t"] == "s"

        law = ConservationLaw1D("T", diffusivity="alpha")
        pde_sol = law.solve((0.0, 1.0), 20, t_end=0.05, initial="sin(pi*x)", bc=("dirichlet", 0.0), params={"alpha": 0.1}, n_out=3)
        aud = audit_pde_conservation(pde_sol)
        sr_aud = standardize_result(aud)
        assert sr_aud.status == VerificationStatus.PASSED
        assert "max_rel_residual" in json.loads(aud.to_json())["entries"]["T"]

        gci = gci_from_values([0.01, 0.02, 0.04], [2.0 - 0.5 * h ** 2 for h in (0.01, 0.02, 0.04)], formal_order=2.0)
        assert json.loads(gci.to_json())["status"] == "Passed"

        mms = mms_study(law, "exp(-t)*sin(pi*x)", cells=(16, 32, 64), params={"alpha": 0.1}, bc_kind="dirichlet", scheme="BDF")
        assert json.loads(mms.to_json())["status"] == "Passed"

        st_sol = Stefan1D(T_wall=-6.0, T_f=0.0, h_c=0.0, n_xi=40).solve(600.0, delta0=1e-5, n_out=5)
        assert json.loads(st_sol.to_json())["success"] is True


class TestODESystemEvents:
    def test_terminal_and_non_terminal_events(self) -> None:
        # Ball thrown upward: dy/dt = v, dv/dt = -g, hit ground y = 0 downward
        ode = ODESystem({"y": "v", "v": "-g"})
        traj = ode.solve(
            (0.0, 5.0),
            {"y": 0.0, "v": 9.81},
            params={"g": 9.81},
            events=[EventSpec(name="ground_hit", expr="y", terminal=True, direction=-1.0)],
            rtol=1e-10,
            atol=1e-12,
        )
        assert traj.success
        assert "ground_hit" in traj.t_events
        assert traj.t_events["ground_hit"].size == 1
        assert np.isclose(traj.t_events["ground_hit"][0], 2.0, atol=1e-6)

        # Non-terminal oscillatory zero crossings: d2x/dt2 = -w^2 * x
        osc = ODESystem({"x": "v", "v": "-w**2 * x"})
        traj_osc = osc.solve(
            (0.0, 2.0 * np.pi),
            {"x": 1.0, "v": 0.0},
            params={"w": 1.0},
            events=[EventSpec(name="zero_cross", expr="x", terminal=False, direction=0.0)],
            rtol=1e-9,
            atol=1e-11,
        )
        assert traj_osc.t_events["zero_cross"].size == 2
        assert np.allclose(traj_osc.t_events["zero_cross"], [0.5 * np.pi, 1.5 * np.pi], atol=1e-5)

        with pytest.raises(ValueError, match="Event detection requires a scipy"):
            osc.solve((0.0, 1.0), {"x": 1.0, "v": 0.0}, params={"w": 1.0}, method="rk4", dt=0.05, events=[EventSpec("z", "x")])


class TestValidatedModelAndUnifiedSolve:
    HEAT_YAML = """
model:
  name: heat_transfer_1d
  version: "0.1"
  description: "One-dimensional transient heat conduction"
  domain: thermal_engineering

variables:
  independent:
    - name: x
      symbol: x
      unit: m
    - name: t
      symbol: t
      unit: s
  dependent:
    - name: temperature
      symbol: T
      unit: K

parameters:
  - name: thermal_diffusivity
    symbol: alpha
    value: 0.05
    unit: m^2/s
    status: fixed

equations:
  - id: heat_equation
    expression: "dT/dt = alpha * d2T/dx2"
    type: pde

conditions:
  initial:
    - expression: "T(x, 0) = sin(pi * x)"
  boundary:
    - location: x_min
      expression: "T = 0.0"
    - location: x_max
      expression: "T = 0.0"

numerics:
  solver: method_of_lines
  time_integrator: BDF
  tolerances:
    rtol: 1.0e-8
    atol: 1.0e-10
"""

    def test_yaml_roundtrip_and_pde_solve_matches_analytical_fourier(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "heat.yaml"
        yaml_path.write_text(self.HEAT_YAML, encoding="utf-8")

        m1 = Model.from_file(yaml_path)
        saved_yaml = tmp_path / "heat_saved.yaml"
        m1.save(saved_yaml)
        m2 = Model.from_file(saved_yaml)
        assert m1.content_hash() == m2.content_hash()

        json_path = tmp_path / "heat.json"
        m1.save(json_path)
        m3 = Model.from_file(json_path)
        assert m1.content_hash() == m3.content_hash()

        res = m2.solve(t_span=(0.0, 0.25), domain=(0.0, 1.0), n_cells=80, n_out=5)
        assert res.status == VerificationStatus.PASSED
        assert res.kind == "pde_simulation"
        assert res.units["T"] == "K"
        assert res.timing_s is not None and res.timing_s >= 0.0

        pde_sol = res.raw
        exact = np.exp(-0.05 * (np.pi ** 2) * 0.25) * np.sin(np.pi * pde_sol.x)
        assert np.allclose(pde_sol.final("T"), exact, atol=5e-4)

    def test_ode_model_solve_and_symbolic_and_calibration(self) -> None:
        ode_model = Model(
            name="decay_model",
            independent_variables=[VariableSpec("t", role="independent", unit="s")],
            dependent_variables=[VariableSpec("y", role="dependent", unit="1")],
            parameters=[ParameterSpec("k", value=1.2, unit="Hz", status="fixed")],
            equations=[EquationSpec(id="decay", expression="dy/dt = -k * y", type="ode")],
            numerics=NumericsConfig(time_integrator="RK45", rtol=1e-9, atol=1e-11),
        )
        res = ode_model.solve(t_span=(0.0, 2.0), initial_conditions={"y": 3.0}, n_out=20)
        assert res.status == VerificationStatus.PASSED
        exact = 3.0 * np.exp(-1.2 * 2.0)
        assert np.isclose(res.raw.final["y"], exact, rtol=1e-7)

        # Symbolic + calibration on constitutive model
        ke_model = Model(
            name="kinetic_energy",
            independent_variables=[
                VariableSpec("m", role="independent", unit="kg"),
                VariableSpec("v", role="independent", unit="m/s"),
            ],
            dependent_variables=[VariableSpec("K", role="dependent", unit="J")],
            parameters=[ParameterSpec("kappa_K", value=None, unit="1", status="unknown", bounds=(0.0, 5.0))],
            equations=[EquationSpec(id="ke_law", expression="K = kappa_K * m * v**2", type="constitutive")],
            dependencies=["K = fn(m, v)"],
        )
        sym_res = ke_model.analyse_symbolically(target="K", zero_initials=True)
        assert sym_res.kind == "symbolic_derivation"
        assert sym_res.status == VerificationStatus.PASSED

        rng = np.random.default_rng(12)
        m_arr = rng.uniform(1.0, 5.0, 30)
        v_arr = rng.uniform(2.0, 10.0, 30)
        k_arr = 0.5 * m_arr * (v_arr ** 2)
        cal_res = ke_model.calibrate({"K": k_arr, "m": m_arr, "v": v_arr})
        assert cal_res.status == VerificationStatus.PASSED
        assert np.isclose(cal_res.raw.params["kappa_K"], 0.5, atol=1e-6)

    def test_security_and_validation_reject_cycles_unsafe_yaml_and_injection(self) -> None:
        # 1. Cycle in dependencies
        with pytest.raises(ModelValidationError, match="Cycle detected"):
            Model(
                name="cyclic",
                independent_variables=[VariableSpec("x", role="independent", unit="m")],
                dependent_variables=[VariableSpec("a", unit="1"), VariableSpec("b", unit="1")],
                dependencies=["a = fn(b)", "b = fn(a)"],
            )

        # 2. Malicious Python expression inside YAML equation
        bad_expr_yaml = self.HEAT_YAML.replace("dT/dt = alpha * d2T/dx2", "dT/dt = __import__('os').system('id')")
        with pytest.raises(ModelValidationError):
            Model.from_yaml(bad_expr_yaml)

        # 3. Unsafe YAML Python object tag rejected by yaml.safe_load
        unsafe_tag_yaml = "model:\n  name: !!python/object/apply:os.system ['echo pwn']\n"
        with pytest.raises(ModelValidationError, match="Unsafe or invalid YAML"):
            Model.from_yaml(unsafe_tag_yaml)
