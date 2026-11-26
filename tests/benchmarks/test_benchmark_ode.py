"""Benchmark 1: Analytical first-order ODE decay dy/dt = -k*y (ENG_SPEC.md P0.2)."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from rosolver import (
    NumericalSolver,
    ODESystem,
    VerificationStatus,
    integrate_fixed,
    timestep_study,
)


class TestBenchmarkAnalyticalODE:
    def test_ode_accuracy_and_tolerance_refinement(self) -> None:
        ode = ODESystem({"y": "-k * y"})
        y0_val = 2.5
        k_val = 1.4
        t_end = 2.0
        exact_end = y0_val * np.exp(-k_val * t_end)

        for method in ("RK45", "BDF", "Radau"):
            traj = ode.solve((0.0, t_end), {"y": y0_val}, params={"k": k_val}, method=method, rtol=1e-10, atol=1e-12)
            assert traj.success
            exact_curve = y0_val * np.exp(-k_val * traj.t)
            assert np.allclose(traj["y"], exact_curve, rtol=1e-7, atol=1e-9)

        errs = []
        for rtol in (1e-4, 1e-7, 1e-10):
            traj = ode.solve((0.0, t_end), {"y": y0_val}, params={"k": k_val}, method="RK45", rtol=rtol, atol=rtol * 1e-2)
            errs.append(abs(traj.final["y"] - exact_end))
        assert errs[2] < errs[1] < errs[0]

    def test_timestep_observed_orders_against_exact_decay(self) -> None:
        ode = ODESystem({"y": "-k * y"})
        params = {"k": 1.4}
        y0 = {"y": 2.5}
        t_span = (0.0, 2.0)

        for scheme, expected_p, dts in (
            ("euler", 1.0, (0.02, 0.01, 0.005)),
            ("cn", 2.0, (0.1, 0.05, 0.025)),
            ("rk4", 4.0, (0.2, 0.1, 0.05)),
        ):
            rep = timestep_study(ode, dts=dts, t_span=t_span, y0=y0, params=params, scheme=scheme)
            assert rep.status == VerificationStatus.PASSED
            assert rep.passed
            assert np.isclose(rep.gci.observed_order, expected_p, atol=0.2)
            exact = 2.5 * np.exp(-1.4 * 2.0)
            assert np.isclose(rep.gci.extrapolated_value, exact, rtol=1e-3)

    def test_parameter_recovery_from_decay_data(self) -> None:
        t = np.linspace(0.0, 3.0, 40)
        y_obs = 2.5 * np.exp(-1.4 * t)
        t_s, y0_s, k_s = sp.symbols("t y0 k", real=True)
        model = y0_s * sp.exp(-k_s * t_s)
        fit = NumericalSolver.fit(
            model,
            {"t": t, "y": y_obs},
            target_col="y",
            free_params=[y0_s, k_s],
            initial_guess={"y0": 1.0, "k": 1.0},
        )
        assert fit.converged
        assert np.isclose(fit.params["y0"], 2.5, rtol=1e-6)
        assert np.isclose(fit.params["k"], 1.4, rtol=1e-6)

    def test_newton_failure_propagation_sets_success_false(self) -> None:
        # Direct fixed-step implicit solve with insufficient Newton budget flags newton_failures > 0
        res = integrate_fixed(
            lambda _t, y: np.array([-50.0 * y[0] ** 3]),
            (0.0, 1.0),
            [5.0],
            dt=0.5,
            scheme="beuler",
            newton_maxiter=1,
            newton_tol=1e-14,
        )
        assert res.newton_failures > 0

        # ODESystem.solve propagates fixed-step newton_failures into success=False and warnings
        ode = ODESystem({"y": "-50.0 * y**3"})
        traj = ode.solve((0.0, 1.0), {"y": 10.0}, method="beuler", dt=0.5)
        assert traj.newton_failures > 0
        assert not traj.success
        assert traj.warnings

        with pytest.raises(RuntimeError):
            ode.solve((0.0, 1.0), {"y": 10.0}, method="beuler", dt=0.5, raise_on_failure=True)

        # Finite-time blow-up causes solve_ivp failure -> success=False, warnings populated, or raises
        blowup = ODESystem({"y": "y**2"})
        tr_fail = blowup.solve((0.0, 2.0), {"y": 1.0}, method="RK45")
        assert not tr_fail.success
        assert tr_fail.warnings
        with pytest.raises(RuntimeError):
            blowup.solve((0.0, 2.0), {"y": 1.0}, method="RK45", raise_on_failure=True)
