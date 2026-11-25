"""Comprehensive tests for generic Verification, Conservation, PDE, and Coupling Stability."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from rosolver import (
    BC,
    ConservationLaw1D,
    NumericalSolver,
    ODESystem,
    SheddingEventSpec,
    analyze_stiffness,
    audit_hybrid_conservation,
    audit_ode_invariants,
    audit_pde_conservation,
    cosimulate_coupled_odes,
    gci_from_values,
    grid_convergence_study,
    mms_study,
    plot_conservation_audit,
    plot_gci_study,
    plot_mms_convergence,
    plot_stability_spectrum,
    solve_partitioned_fixed_point,
    timestep_study,
)
from rosolver.cli import main as cli_main


class TestGridConvergenceAndMMS:
    def test_gci_uniform_and_nonuniform_celik_iteration(self) -> None:
        # Exact quadratic sequence phi(h) = 2.0 - 0.5 * h^2
        hs_uni = [0.01, 0.02, 0.04]
        phis_uni = [2.0 - 0.5 * h ** 2 for h in hs_uni]
        rep_u = gci_from_values(hs_uni, phis_uni, formal_order=2.0)
        assert rep_u.passed
        assert rep_u.regime == "monotonic"
        assert np.isclose(rep_u.observed_order, 2.0, atol=1e-6)
        assert np.isclose(rep_u.extrapolated_value, 2.0, atol=1e-8)
        assert np.isclose(rep_u.asymptotic_ratio, 1.0, atol=1e-3)

        # Non-uniform refinement ratios: r21 = 1.5, r32 = 2.0 (Celik fixed-point iteration!)
        hs_non = [0.02, 0.03, 0.06]
        phis_non = [10.0 + 1.2 * (h ** 2.0) for h in hs_non]
        rep_n = gci_from_values(hs_non, phis_non, formal_order=2.0)
        assert rep_n.passed
        assert np.isclose(rep_n.observed_order, 2.0, atol=1e-5)
        assert np.isclose(rep_n.extrapolated_value, 10.0, atol=1e-8)

    def test_gci_flags_divergence_and_oscillation(self) -> None:
        # Divergent sequence: fine-to-medium jump bigger than medium-to-coarse
        rep_div = gci_from_values([0.01, 0.02, 0.04], [1.50, 1.10, 1.05])
        assert rep_div.regime == "divergent"
        assert not rep_div.passed

        # Oscillatory convergence: alternating sign, shrinking amplitude
        rep_osc = gci_from_values([0.01, 0.02, 0.04], [1.001, 0.996, 1.016])
        assert rep_osc.regime == "oscillatory"

    def test_mms_viscous_burgers_orders_1_and_2(self) -> None:
        burgers = ConservationLaw1D("u", flux="u**2 / 2", diffusivity="nu")
        exact = "1.0 + 0.25 * sin(2 * pi * (x - 0.4 * t))"

        rep1 = mms_study(
            burgers,
            exact=exact,
            domain=(0.0, 1.0),
            t_end=0.08,
            cells=(24, 48, 96),
            params={"nu": 0.02},
            bc_kind="periodic",
            reconstruction="constant",
            scheme="ssprk3",
            cfl=0.2,
        )
        assert rep1.passed
        assert 0.85 <= rep1.order_l2["u"] <= 1.25

        rep2 = mms_study(
            burgers,
            exact=exact,
            domain=(0.0, 1.0),
            t_end=0.08,
            cells=(24, 48, 96),
            params={"nu": 0.02},
            bc_kind="periodic",
            reconstruction="linear",
            scheme="ssprk3",
            cfl=0.15,
        )
        assert rep2.passed
        assert rep2.order_l2["u"] >= 1.90

    def test_mms_coupled_two_field_dirichlet_system(self) -> None:
        # Coupled advection-diffusion-reaction system on [0, 1] with time-dependent Dirichlet BCs
        sys_pde = ConservationLaw1D(
            fields=["u", "v"],
            flux={"u": "c1 * u + 0.1 * v", "v": "c2 * v"},
            diffusivity={"u": "D1", "v": "D2"},
            source={"u": "-k * u * v", "v": "k * u * v"},
        )
        exact = {
            "u": "1.2 + 0.2 * sin(pi * x) * cos(t)",
            "v": "0.8 + 0.15 * cos(pi * x) * exp(-0.3 * t)",
        }
        rep = mms_study(
            sys_pde,
            exact=exact,
            domain=(0.0, 1.0),
            t_end=0.05,
            cells=(40, 80, 160),
            params={"c1": 0.6, "c2": -0.4, "D1": 0.02, "D2": 0.015, "k": 0.5},
            bc_kind="dirichlet",
            reconstruction="linear",
            cfl=0.15,
        )
        assert rep.passed
        assert rep.order_l2["u"] >= 1.85
        assert rep.order_l2["v"] >= 1.85

    def test_grid_convergence_study_callable(self) -> None:
        law = ConservationLaw1D("u", flux="a * u", diffusivity="D")
        rep = grid_convergence_study(
            lambda n: law.solve(
                (0.0, 1.0),
                n,
                t_end=0.15,
                initial="exp(-40 * (x - 0.35)**2)",
                bc="periodic",
                params={"a": 1.0, "D": 0.01},
                reconstruction="linear",
                cfl=0.2,
            ).probe(0.5),
            cells=(40, 80, 160),
            formal_order=2.0,
        )
        assert rep.passed
        assert rep.gci_fine_21 < 0.02


class TestTimestepSensitivityAndConservation:
    def test_timestep_observed_orders_euler_cn_rk4(self) -> None:
        decay = ODESystem({"y": "-lam * y + sin(t)"})
        p = {"lam": 2.0}

        r_euler = timestep_study(decay, dts=(0.05, 0.025, 0.0125), t_span=(0.0, 1.0), y0={"y": 1.0}, params=p, scheme="euler")
        assert r_euler.passed and np.isclose(r_euler.gci.observed_order, 1.0, atol=0.15)

        r_cn = timestep_study(decay, dts=(0.1, 0.05, 0.025), t_span=(0.0, 1.0), y0={"y": 1.0}, params=p, scheme="cn")
        assert r_cn.passed and np.isclose(r_cn.gci.observed_order, 2.0, atol=0.15)

        r_rk4 = timestep_study(
            decay,
            dts=(0.1, 0.05, 0.025),
            t_span=(0.0, 1.0),
            y0={"y": 1.0},
            params=p,
            scheme="rk4",
            dx=0.1,
            wave_speed=1.0,
            diffusivity=0.01,
        )
        assert r_rk4.passed and np.isclose(r_rk4.gci.observed_order, 4.0, atol=0.2)
        assert r_rk4.cfl_numbers is not None and r_rk4.fourier_numbers is not None

    def test_pde_fvm_conservation_audit_with_inflow_outflow_and_reaction(self) -> None:
        # Reversible reaction A <-> B with time-varying Dirichlet inflow and outflow right BC:
        # Each species satisfies exact flux+source balance AND A+B has zero net source!
        reactor = ConservationLaw1D(
            fields=["A", "B"],
            flux={"A": "u_vel * A", "B": "u_vel * B"},
            diffusivity={"A": "D_a", "B": "D_b"},
            source={"A": "-k_f * A + k_r * B", "B": "k_f * A - k_r * B"},
        )
        sol = reactor.solve(
            domain=(0.0, 2.0),
            n_cells=60,
            t_end=0.4,
            initial={"A": "0.5 * exp(-5 * x)", "B": "0.1"},
            bc={"left": BC.dirichlet({"A": "1.0 + 0.2*sin(3*t)", "B": "0.2"}), "right": "outflow"},
            params={"u_vel": 0.8, "D_a": 0.03, "D_b": 0.02, "k_f": 1.5, "k_r": 0.4},
            scheme="ssprk3",
            cfl=0.3,
        )
        audit = audit_pde_conservation(sol, tol=1e-12)
        assert audit.passed
        assert audit.entries["A"].max_rel_residual < 1e-13
        assert audit.entries["B"].max_rel_residual < 1e-13
        # Net integrated chemical source of A + B must cancel to machine precision!
        net_reaction = sol.budget["A"]["source"] + sol.budget["B"]["source"]
        assert np.max(np.abs(net_reaction)) < 1e-13

    def test_ode_symbolic_and_numerical_invariants_seir_and_hamiltonian(self) -> None:
        # 1. Epidemiology SEIR: S + E + I + R == N identically
        seir = ODESystem(
            {
                "S": "-beta * S * I / N",
                "E": "beta * S * I / N - sigma * E",
                "I": "sigma * E - gamma * I",
                "R": "gamma * I",
            }
        )
        is_inv, deriv = seir.check_invariant("S + E + I + R")
        assert is_inv and deriv == 0

        p_seir = {"beta": 0.45, "sigma": 0.2, "gamma": 0.1, "N": 1000.0}
        traj = seir.solve((0.0, 80.0), {"S": 990.0, "E": 5.0, "I": 5.0, "R": 0.0}, params=p_seir, rtol=1e-10, atol=1e-12)
        rep = audit_ode_invariants(seir, traj, {"total_population": "S + E + I + R"}, params=p_seir, tol=1e-9)
        assert rep.passed

        # 2. Harmonic oscillator: Crank-Nicolson preserves quadratic energy 0.5*(v^2 + w^2*x^2) to machine precision!
        osc = ODESystem({"x": "v", "v": "-w**2 * x"})
        p_osc = {"w": 2.0}
        traj_cn = osc.solve((0.0, 10.0), {"x": 1.0, "v": 0.0}, params=p_osc, method="cn", dt=0.05)
        rep_cn = audit_ode_invariants(osc, traj_cn, {"energy": "0.5 * v**2 + 0.5 * w**2 * x**2"}, params=p_osc, tol=1e-11)
        assert rep_cn.passed

        # Forward Euler violates energy conservation (artificial energy growth) -> audit catches it!
        traj_eu = osc.solve((0.0, 10.0), {"x": 1.0, "v": 0.0}, params=p_osc, method="euler", dt=0.05)
        rep_eu = audit_ode_invariants(osc, traj_eu, {"energy": "0.5 * v**2 + 0.5 * w**2 * x**2"}, params=p_osc, tol=1e-3)
        assert not rep_eu.passed

    def test_hybrid_event_conservation_audit(self) -> None:
        t_s, y_s = sp.symbols("t y", real=True)
        rate_fn = lambda tv, yv: 0.2 + 0.05 * np.sin(tv)
        rate_expr = sp.Float(0.2) + sp.Float(0.05) * sp.sin(t_s)
        sol = NumericalSolver.integrate_hybrid_ode(
            sp.Eq(sp.Derivative(y_s, t_s), rate_expr),
            wrt="t",
            span=(0.0, 30.0),
            y0=0.5,
            events=[SheddingEventSpec("discharge", lambda tv, yv: yv - 2.0, shed_fraction=0.6)],
            target_name="y",
            dt_max=0.1,
        )
        assert sol.n_events >= 2
        audit = audit_hybrid_conservation(sol, rate_fn, tol=1e-5)
        assert audit.passed


class TestStiffnessAndCouplingStability:
    def test_stiffness_detection_robertson_and_pde_fourier_limit(self) -> None:
        # Classic stiff Robertson chemical kinetics
        rob = ODESystem(
            {
                "A": "-0.04 * A + 1e4 * B * C",
                "B": "0.04 * A - 1e4 * B * C - 3e7 * B**2",
                "C": "3e7 * B**2",
            }
        )
        rep = analyze_stiffness(rob, [0.98, 3e-5, 0.02])
        assert rep.is_stiff
        assert rep.stiffness_ratio > 5e3
        assert "implicit" in rep.recommended_method

        # Semi-discrete 1D heat equation: explicit dt_max scales as h^2 / D
        heat = ConservationLaw1D("T", diffusivity="alpha")
        rhs_20, u0_20 = heat.semidiscrete((0.0, 1.0), 20, initial="sin(pi * x)", bc=("dirichlet", 0.0), params={"alpha": 0.1})
        rhs_40, u0_40 = heat.semidiscrete((0.0, 1.0), 40, initial="sin(pi * x)", bc=("dirichlet", 0.0), params={"alpha": 0.1})
        st_20 = analyze_stiffness(rhs_20, u0_20, schemes=("euler", "rk4"))
        st_40 = analyze_stiffness(rhs_40, u0_40, schemes=("euler", "rk4"))
        # Doubling N quarters explicit Euler dt_max (~4.0x ratio)
        ratio = st_20.max_stable_dts["euler"] / st_40.max_stable_dts["euler"]
        assert 3.8 <= ratio <= 4.2

    def test_aitken_stabilizes_divergent_partitioned_loop(self) -> None:
        # Coupled algebraic interface: u_B = 2.0 - 1.6 * u_A, u_A = u_B => G(u) = 2.0 - 1.6*u
        # Fixed point u* = 2.0 / 2.6 = 10/13. Unrelaxed Picard has |G'| = 1.6 > 1 -> DIVERGES!
        step_map = lambda u: np.array([2.0 - 1.6 * u[0], 3.0 - 1.4 * u[1]])
        exact = np.array([2.0 / 2.6, 3.0 / 2.4])

        raw = solve_partitioned_fixed_point(step_map, [0.0, 0.0], max_iter=25, acceleration="none")
        assert not raw.converged
        assert raw.asymptotic_contraction > 1.3

        aitken = solve_partitioned_fixed_point(step_map, [0.0, 0.0], max_iter=25, acceleration="aitken")
        assert aitken.converged
        assert aitken.iterations <= 12
        assert np.allclose(aitken.solution, exact, atol=1e-10)

    def test_cosimulation_strong_beats_loose_against_monolithic(self) -> None:
        # Coupled thermal-chemical or predator-prey-like two-way linear/nonlinear oscillator
        sys_a = ODESystem({"x": "-1.5 * x + 2.2 * y"})
        sys_b = ODESystem({"y": "-2.2 * x - 0.8 * y"})
        y0 = {"x": 1.0, "y": 0.5}

        loose = cosimulate_coupled_odes(sys_a, sys_b, (0.0, 2.0), y0, dt_macro=0.1, mode="loose")
        strong = cosimulate_coupled_odes(sys_a, sys_b, (0.0, 2.0), y0, dt_macro=0.1, mode="strong")
        strong_fine = cosimulate_coupled_odes(sys_a, sys_b, (0.0, 2.0), y0, dt_macro=0.05, mode="strong")
        assert strong.converged and strong_fine.converged
        assert strong.max_error_vs_monolithic < 0.1 * loose.max_error_vs_monolithic
        # Strong linear-ramp interface coupling is O(dt_macro^2) (~4x error drop when halving dt_macro)
        assert strong.max_error_vs_monolithic / strong_fine.max_error_vs_monolithic > 3.5

    def test_vv_plots_and_cli_gci(self, tmp_path, capsys) -> None:
        burgers = ConservationLaw1D("u", flux="u**2/2", diffusivity="nu")
        mms_rep = mms_study(
            burgers, "1 + 0.2*sin(2*pi*(x - 0.3*t))", cells=(16, 32, 64), params={"nu": 0.02}, reconstruction="linear"
        )
        assert plot_mms_convergence(mms_rep, path=tmp_path / "mms.png").exists()

        sol = burgers.solve((0.0, 1.0), 32, t_end=0.1, initial="1 + 0.2*sin(2*pi*x)", bc="periodic", params={"nu": 0.02})
        aud = audit_pde_conservation(sol)
        assert plot_conservation_audit(aud, path=tmp_path / "cons.png").exists()

        osc = ODESystem({"x": "v", "v": "-4*x - 0.5*v"})
        st = analyze_stiffness(osc, [1.0, 0.0])
        assert plot_stability_spectrum(st, dt=0.5, scheme="rk4", path=tmp_path / "stab.png").exists()

        gci_png = tmp_path / "gci.png"
        rc = cli_main(["--gci", "0.01,0.02,0.04:1.9995,1.9980,1.9920", "--formal-order", "2", "--gci-plot", str(gci_png)])
        assert rc == 0 and gci_png.exists()
        assert "Grid Convergence Index" in capsys.readouterr().out
