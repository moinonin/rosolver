"""Validation of the front-fixing Stefan solver (eq. 3.4 of the OTI cheat-sheet)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from rosolver import Stefan1D, SheddingEventSpec, neumann_lambda, stefan_shedding_event


class TestNeumannSimilarity:
    def test_lambda_reduces_to_sqrt_ste_over_two(self) -> None:
        # For Ste -> 0, erf(l) ~ 2l/sqrt(pi) => 2 l^2 ~ Ste => l ~ sqrt(Ste/2).
        for ste in (1e-4, 1e-3, 1e-2):
            lam = neumann_lambda(ste)
            assert np.isclose(lam, np.sqrt(ste / 2.0), rtol=2e-3)

    def test_neumann_delta_matches_quasi_steady_law(self) -> None:
        # delta = 2 l sqrt(alpha t) vs sqrt(2 alpha Ste t); equal to leading order in Ste.
        st = Stefan1D(k_eff=2.22, rho=917.0, cp=2100.0, Lf=3.34e5,
                      T_wall=-6.0, T_f=0.0, h_c=0.0)
        t = 3600.0
        exact = float(st.neumann_delta(t))
        quasi = float(np.sqrt(2.0 * st.alpha * st.stefan_number * t))
        assert np.isclose(exact, quasi, rtol=1.2e-2)

    def test_rejects_zero_stefan_number(self) -> None:
        with pytest.raises(ValueError):
            neumann_lambda(0.0)


class TestFrontTracking:
    def test_matches_exact_neumann_solution(self) -> None:
        # h_c = 0: the full PDE must reproduce the self-similar front.
        st = Stefan1D(k_eff=2.22, rho=917.0, cp=2100.0, Lf=3.34e5,
                      T_wall=-6.0, T_f=0.0, h_c=0.0, n_xi=200)
        t_end = 3600.0
        sol = st.solve(t_end, delta0=1e-5, method="BDF", n_out=20)
        exact = float(st.neumann_delta(t_end))
        assert np.isclose(sol.delta[-1], exact, rtol=1e-2)
        # Also check the whole history, not just the endpoint.
        mid = len(sol.t) // 2
        assert np.isclose(sol.delta[mid], float(st.neumann_delta(sol.t[mid])), rtol=1e-2)

    def test_spatial_convergence_to_neumann(self) -> None:
        errors = []
        for n_xi in (50, 100, 200):
            st = Stefan1D(T_wall=-6.0, T_f=0.0, h_c=0.0, n_xi=n_xi)
            sol = st.solve(1800.0, delta0=1e-5, n_out=5)
            errors.append(abs(sol.delta[-1] - float(st.neumann_delta(1800.0))))
        # Refinement must reduce the error monotonically.
        assert errors[2] < errors[1] < errors[0]

    def test_convective_branch_matches_quasi_steady_limit(self) -> None:
        # Small cp => tiny Ste => the linear-profile (quasi-steady) reduction is accurate.
        st = Stefan1D(k_eff=2.22, rho=917.0, cp=210.0, Lf=3.34e5,
                      T_wall=-1.0, T_inf=-4.13, T_f=0.0, h_c=160.0, n_xi=200)
        delta0 = 1e-5
        t_end = 600.0
        sol = st.solve(t_end, delta0=delta0, n_out=20)
        quasi = float(st.quasi_steady_delta(t_end, delta0))
        assert np.isclose(sol.delta[-1], quasi, rtol=3e-2)

        # And the closed-form scalar ODE integrated independently agrees with the helper.
        kappa = st.k_eff * (st.T_f - st.T_wall) / (st.rho * st.Lf)
        beta = st.h_c * (st.T_f - st.T_inf) / (st.rho * st.Lf)
        ref = solve_ivp(lambda _t, q: [2.0 * kappa + 2.0 * beta * np.sqrt(max(q[0], 1e-30))],
                        (0.0, t_end), [delta0 * delta0], method="BDF", rtol=1e-10, atol=1e-14)
        assert np.isclose(np.sqrt(ref.y[0, -1]), quasi, rtol=1e-4)

    def test_temperature_field_satisfies_both_boundary_conditions(self) -> None:
        st = Stefan1D(T_wall=-6.0, T_f=0.0, h_c=160.0, n_xi=100)
        sol = st.solve(1800.0, delta0=1e-5, n_out=10)
        y, T = sol.profile(-1)
        assert y[0] == 0.0
        assert np.isclose(y[-1], sol.delta[-1])
        assert np.isclose(T[0], st.T_wall)
        assert np.isclose(T[-1], st.T_f)
        # Cold wall, freezing surface: the profile is monotone in between.
        assert np.all(np.diff(T) >= -1e-9)

    def test_convection_thickens_front_versus_adiabatic_air(self) -> None:
        # Cold ambient (T_inf < T_f) removes latent heat => larger h_c grows ice faster.
        base = dict(T_wall=-6.0, T_f=0.0, n_xi=120)
        lo = Stefan1D(h_c=0.0, T_inf=-4.13, **base).solve(1800.0, delta0=1e-5)
        hi = Stefan1D(h_c=160.0, T_inf=-4.13, **base).solve(1800.0, delta0=1e-5)
        assert hi.delta[-1] > lo.delta[-1]

    def test_physical_numbers_snowtek_case(self) -> None:
        # Smoke test with the sheet's WINICE/SnowTek constants over 24 h.
        st = Stefan1D(k_eff=2.22, rho=917.0, cp=2100.0, Lf=3.34e5,
                      T_wall=-4.13, T_inf=-4.13, T_f=-1.18, h_c=160.0, n_xi=200)
        sol = st.solve(86400.0, delta0=1e-5, n_out=50)
        assert np.isfinite(sol.delta[-1])
        # 0 < delta < 1 m: the layer cannot exceed the conduction length scale.
        assert 1e-3 < sol.delta[-1] < 1.0
        assert "Stefan Front-Tracking" in sol.summary()

    def test_validation_and_guards(self) -> None:
        with pytest.raises(ValueError, match="T_f > T_wall"):
            Stefan1D(T_wall=1.0, T_f=0.0)
        with pytest.raises(ValueError, match="n_xi"):
            Stefan1D(n_xi=3)
        with pytest.raises(ValueError, match="mdot_imp"):
            Stefan1D(mdot_imp=-1e-4)
        st = Stefan1D()
        with pytest.raises(ValueError, match="delta0"):
            st.solve(10.0, delta0=0.0)
        with pytest.raises(ValueError, match="requires `dt`"):
            st.solve(10.0, delta0=1e-5, method="beuler")
        # Fixed-step implicit branch is available when dt is supplied.
        sol = st.solve(60.0, delta0=1e-5, method="beuler", dt=0.5, n_out=10)
        assert sol.n_steps == 120


class TestWetDryCoupling:
    def _model(self, mdot_imp: float, **kw) -> Stefan1D:
        base = dict(k_eff=2.22, rho=917.0, cp=2100.0, Lf=3.34e5,
                    T_wall=-4.13, T_inf=-4.13, T_f=-1.18, h_c=160.0,
                    n_xi=160, mdot_imp=mdot_imp)
        base.update(kw)
        return Stefan1D(**base)

    def test_dry_branch_is_mass_limited(self) -> None:
        # Tiny supply: capacity always exceeds demand, so all water freezes (rime).
        st = self._model(1e-4)
        delta0, t_end = 1e-4, 600.0
        sol = st.solve(t_end, delta0=delta0, n_out=20)
        assert np.all(sol.branch == "dry")
        assert np.all(sol.T_s <= st.T_f + 1e-10)
        assert np.allclose(sol.mdot_ice, 1e-4, rtol=1e-8)
        assert all(sol.runback(i) == 0.0 for i in range(len(sol.t)))
        # ddelta/dt = mdot_imp/rho everywhere -> exactly linear growth.
        expected = delta0 + 1e-4 / st.rho * t_end
        assert np.isclose(sol.delta[-1], expected, rtol=1e-6)

    def test_wet_branch_matches_unlimited_supply(self) -> None:
        # Abundant supply on a thick-enough layer: energy-limited branch == the pure Stefan model.
        wet = self._model(0.1).solve(3600.0, delta0=1e-3, n_out=30)
        pure = Stefan1D(k_eff=2.22, rho=917.0, cp=2100.0, Lf=3.34e5,
                        T_wall=-4.13, T_inf=-4.13, T_f=-1.18, h_c=160.0,
                        n_xi=160).solve(3600.0, delta0=1e-3, n_out=30)
        assert np.all(wet.branch == "wet")
        assert np.allclose(wet.T_s, wet.params["T_f"])
        assert np.allclose(wet.delta, pure.delta, rtol=1e-6, atol=1e-9)
        assert 0.0 < wet.mdot_ice[-1] < 0.1
        assert np.isclose(wet.runback(-1), 0.1 - wet.mdot_ice[-1], rtol=1e-8)

    def test_rime_to_glaze_transition(self) -> None:
        st = self._model(3e-3)
        sol = st.solve(6 * 3600.0, delta0=1e-5, n_out=160)
        branch = np.asarray(sol.branch)
        assert set(branch) == {"dry", "wet"}
        first_wet = int(np.argmax(branch == "wet"))
        assert first_wet > 0
        # The switch sits where the surface reaches the freezing point.
        assert np.isclose(sol.T_s[first_wet], st.T_f, atol=1e-3)
        # ...and the freezing flux is continuous across it (== supply at the switch).
        assert np.isclose(sol.mdot_ice[first_wet], st.mdot_imp, rtol=0.05)
        # Runback only appears once the surface is wet, and grows as delta thickens.
        assert all(sol.runback(i) == 0.0 for i in range(first_wet))
        assert sol.runback(-1) > sol.runback(first_wet)

    def test_summary_reports_branch_and_runback(self) -> None:
        sol = self._model(3e-3).solve(2 * 3600.0, delta0=1e-5, n_out=40)
        text = sol.summary()
        assert "Impingement" in text
        assert "runback" in text
        assert "dry" in text


class TestStefanPlot:
    def test_plot_writes_file_and_axes_variant(self, tmp_path) -> None:
        from rosolver import plot_stefan

        st = Stefan1D(k_eff=2.22, T_wall=-4.13, T_inf=-4.13, T_f=-1.18,
                      h_c=160.0, mdot_imp=3e-3, n_xi=100)
        sol = st.solve(3600.0, delta0=1e-5, n_out=40)
        out = tmp_path / "stefan.png"
        assert plot_stefan(sol, st, path=out, show_surface=True) == out
        assert out.exists() and out.stat().st_size > 0

    def test_plot_into_existing_axis(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from rosolver import plot_stefan

        st = Stefan1D(T_wall=-6.0, T_f=0.0, h_c=0.0, n_xi=80)
        sol = st.solve(600.0, delta0=1e-5, n_out=10)
        fig, ax = plt.subplots()
        assert plot_stefan(sol, st, ax=ax) is None
        assert len(ax.lines) >= 1
        plt.close(fig)


class TestMessingerClosure:
    def _model(self, **kw) -> Stefan1D:
        base = dict(k_eff=2.22, rho=917.0, cp=2100.0, Lf=3.34e5,
                    T_wall=-4.13, T_inf=-4.13, T_f=-1.18, h_c=160.0,
                    n_xi=120, mdot_imp=3e-3)
        base.update(kw)
        return Stefan1D(**base)

    def test_impingement_enthalpy_terms_shift_capacity(self) -> None:
        # Probe the wet branch (energy limited) so q_cap sets the freezing flux.
        st = self._model(mdot_imp=3e-2)
        T_int = st.initial_state(1e-3)[:-1]
        base = st._branch_state(T_int, 1e-3)[1] * st.rho * st.Lf
        assert st._branch_state(T_int, 1e-3)[3] == "wet"

        def capacity(**kw) -> float:
            s = self._model(mdot_imp=3e-2, **kw)
            return s._branch_state(T_int, 1e-3)[1] * s.rho * s.Lf

        sensible = capacity(cp_imp=4186.0)
        kinetic = capacity(V_imp=100.0)
        evap = capacity(mdot_evap=5e-4)
        # Supercooled droplets and evaporation remove heat; impact kinetic energy adds it.
        assert sensible > base
        assert evap > base
        assert kinetic < base
        assert np.isclose(sensible - base, 3e-2 * 4186.0 * (st.T_f - st.T_inf), rtol=1e-8)
        assert np.isclose(base - kinetic, 0.5 * 3e-2 * 100.0 ** 2, rtol=1e-8)
        assert np.isclose(evap - base, 5e-4 * st.L_v, rtol=1e-8)

    def test_switch_vector_field_is_continuous(self) -> None:
        # With cp_imp = 0 the capacity is independent of mdot, so the switch can be
        # approached from either side by perturbing the supply alone.
        st = self._model()
        delta = 1e-2
        T_int = st.initial_state(delta)[:-1]
        a, b = T_int[-1], T_int[-2]
        q_cap = (st.k_eff / delta * (3.0 * st.T_f - 4.0 * a + b) / (2.0 * st.dxi)
                 + st.h_c * (st.T_f - st.T_inf))
        mdot_star = q_cap / st.Lf

        def state(mdot: float):
            return self._model(mdot_imp=mdot)._branch_state(T_int, delta, 0.0)

        dry, wet = state(mdot_star * (1.0 - 1e-9)), state(mdot_star * (1.0 + 1e-9))
        assert dry[3] == "dry" and wet[3] == "wet"
        assert np.isclose(dry[0], wet[0], atol=1e-8)
        assert np.isclose(dry[1], wet[1], rtol=1e-8)
        assert np.isclose(dry[0], st.T_f, atol=1e-6)

    def test_dry_surface_energy_residual_is_zero(self) -> None:
        st = self._model(cp_imp=4186.0, V_imp=50.0, mdot_evap=1e-5)
        sol = st.solve(300.0, delta0=1e-4, n_out=20)
        i = -1
        released = sol.mdot_ice[i] * st.Lf
        removed = (sol.front_gradient(i) + st.h_c * (sol.T_s[i] - st.T_inf)
                   + sol.q_sens[i] + st.mdot_evap * st.L_v - st.q_kinetic)
        assert np.isclose(released, removed, rtol=1e-8)
        assert sol.branch[i] == "dry"

    def test_film_steady_state_satisfies_messinger_mass_balance(self) -> None:
        st = self._model(mdot_imp=6e-3, cp_imp=4186.0, tau_drain=30.0)
        sol = st.solve(3600.0, delta0=1e-5, n_out=60)
        assert sol.film is not None
        assert np.all(sol.film >= 0.0)
        assert sol.branch[-1] == "wet"
        # Steady film: rho_w*h_f/tau_drain == impingement - freezing (no evaporation).
        assert np.isclose(sol.mdot_out[-1], st.mdot_imp - sol.mdot_ice[-1], rtol=2e-2)
        assert np.isclose(sol.film[-1] * st.rho_w / st.tau_drain, sol.mdot_out[-1])
        assert 0.0 < sol.freezing_fraction(-1) < 1.0
        assert np.isclose(sol.runback(-1), st.mdot_imp - sol.mdot_ice[-1])

    def test_dry_branch_keeps_film_empty(self) -> None:
        st = self._model(mdot_imp=1e-4, tau_drain=30.0)
        sol = st.solve(600.0, delta0=1e-4, n_out=20)
        assert np.all(sol.branch == "dry")
        assert np.allclose(sol.film, 0.0)
        assert np.allclose(sol.mdot_out, 0.0)
        assert np.isclose(sol.freezing_fraction(-1), 1.0)

    def test_preexisting_film_freezes_back_without_stalling(self) -> None:
        # A thin ice layer has capacity to spare: the film freezes back and the
        # surface stays below T_f. The vector field must stay continuous at h -> 0.
        st = self._model(mdot_imp=1e-4, tau_drain=30.0, h_film0=1e-4)
        sol = st.solve(600.0, delta0=1e-5, n_out=20)
        assert np.isclose(sol.film[0], 1e-4)
        assert np.all(sol.branch == "dry")
        assert sol.T_s[0] <= st.T_f
        assert np.all(np.diff(sol.film) <= 1e-12)
        assert sol.film[-1] < 1e-8
        assert np.isclose(sol.freezing_fraction(-1), 1.0)

    def test_preexisting_film_persists_when_energy_limited(self) -> None:
        st = self._model(mdot_imp=6e-3, tau_drain=30.0, h_film0=1e-4)
        sol = st.solve(3600.0, delta0=1e-5, n_out=40)
        assert sol.branch[-1] == "wet"
        assert np.isclose(sol.T_s[-1], st.T_f)
        assert sol.film[-1] > 0.0
        assert np.isclose(sol.mdot_out[-1], st.mdot_imp - sol.mdot_ice[-1], rtol=2e-2)


class TestSheddingCoupling:
    def _model(self, **kw) -> Stefan1D:
        base = dict(k_eff=2.22, rho=917.0, cp=2100.0, Lf=3.34e5,
                    T_wall=-4.13, T_inf=-4.13, T_f=-1.18, h_c=160.0,
                    n_xi=100, mdot_imp=3e-3)
        base.update(kw)
        return Stefan1D(**base)

    def test_event_helper_and_guards(self) -> None:
        spec = stefan_shedding_event(0.01, shed_fraction=0.8)
        assert isinstance(spec, SheddingEventSpec)
        assert spec.direction == 1.0
        assert spec.margin_expr(0.0, 0.009) < 0.0 < spec.margin_expr(0.0, 0.011)
        with pytest.raises(ValueError, match="delta_crit"):
            stefan_shedding_event(0.0)
        st = self._model()
        with pytest.raises(ValueError, match="At least one"):
            st.solve_with_shedding([], 100.0)
        with pytest.raises(TypeError, match="callables"):
            st.solve_with_shedding([SheddingEventSpec("bad", "delta - 1")], 100.0)
        with pytest.raises(ValueError, match="solve_ivp method"):
            st.solve_with_shedding([spec], 100.0, method="beuler")

    def test_accretion_and_shedding_alternate(self) -> None:
        crit = 0.01
        st = self._model()
        hyb = st.solve_with_shedding(
            [stefan_shedding_event(crit, shed_fraction=0.8)],
            12 * 3600.0, delta0=1e-5, n_out=240,
        )
        assert len(hyb.events) >= 5
        times = [ev.time for ev in hyb.events]
        assert all(t2 > t1 for t1, t2 in zip(times, times[1:]))
        for ev in hyb.events:
            assert np.isclose(ev.pre_value, crit, rtol=1e-3)
            assert np.isclose(ev.post_value, 0.2 * ev.pre_value)
            assert np.isclose(ev.shed_amount, ev.pre_value - ev.post_value)
        assert np.all(hyb.delta <= crit * 1.001)
        assert hyb.delta[-1] < crit
        assert "Shedding events" in hyb.summary()

    def test_shed_resamples_inner_temperature_profile(self) -> None:
        st = self._model()
        hyb = st.solve_with_shedding(
            [stefan_shedding_event(0.01, shed_fraction=0.5)],
            7200.0, delta0=1e-5, n_out=120,
        )
        assert hyb.events
        ev = hyb.events[0]
        i_pre = int(np.argmin(np.abs(hyb.t - ev.time)))
        ratio = hyb.delta[i_pre + 1] / hyb.delta[i_pre]
        assert np.isclose(ratio, 0.5, rtol=1e-6)
        xi = hyb.xi[1:-1]
        T_pre_resampled = np.interp(xi * ratio, hyb.xi, hyb.T_xi[i_pre])
        assert np.allclose(hyb.T_xi[i_pre + 1, 1:-1], T_pre_resampled, atol=1e-3)

    def test_no_event_when_threshold_is_never_reached(self) -> None:
        st = self._model()
        hyb = st.solve_with_shedding(
            [stefan_shedding_event(1.0)], 3600.0, delta0=1e-5, n_out=40,
        )
        assert hyb.events == []
        assert np.allclose(hyb.delta, st.solve(3600.0, delta0=1e-5, n_out=40).delta)

    def test_film_and_shedding_couple(self) -> None:
        st = self._model(mdot_imp=6e-3, tau_drain=30.0)
        hyb = st.solve_with_shedding(
            [stefan_shedding_event(0.01, 0.5)], 7200.0, delta0=1e-5, n_out=120,
        )
        assert hyb.film is not None and np.all(hyb.film >= 0.0)
        assert len(hyb.events) >= 1
        assert np.all(hyb.delta <= 0.0101)


class TestHybridStefanPlot:
    def test_event_markers_and_film_twin_axis(self, tmp_path) -> None:
        from rosolver import plot_stefan

        st = Stefan1D(k_eff=2.22, T_wall=-4.13, T_inf=-4.13, T_f=-1.18,
                      h_c=160.0, mdot_imp=3e-3, tau_drain=30.0, n_xi=80)
        hyb = st.solve_with_shedding(
            [stefan_shedding_event(0.008, 0.7)], 7200.0, delta0=1e-5, n_out=80,
        )
        out = tmp_path / "hybrid_stefan.png"
        assert plot_stefan(hyb, st, path=out, show_film=True) == out
        assert out.exists() and out.stat().st_size > 0

