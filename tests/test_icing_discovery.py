"""Tests for turbomachinery ice accretion & shedding discovery and rosolver upgrades."""

from __future__ import annotations

import numpy as np
import sympy as sp

from rosolver import (
    NumericalSolver,
    SheddingEventSpec,
    formulate,
)


def test_saturating_hill_and_exp_pi_impingement() -> None:
    """Verify automatic Stokes-number extraction and saturating Hill candidate BIC selection."""
    res = formulate(
        relations=["m_imp = fn(v_rel; IWC, tau_p, c)"],
        dims={
            "m_imp": "M/(L^2*T)",
            "v_rel": "L/T",
            "IWC": "M/L^3",
            "tau_p": "T",
            "c": "L",
        },
        constants=["IWC", "tau_p", "c"],
        target="m_imp",
        wrt="v_rel",
        zero_initials=True,
        include_saturating=True,
    )
    cand_names = [c.name for c in res.candidates]
    assert "buckingham_pi_hill" in cand_names
    assert "buckingham_pi_exp_sat" in cand_names

    rng = np.random.default_rng(42)
    n = 70
    v_rel = rng.uniform(25.0, 180.0, n)
    iwc = rng.uniform(0.5e-3, 3.5e-3, n)
    tau_p = rng.uniform(5e-5, 8e-4, n)
    chord = rng.uniform(0.04, 0.12, n)
    k_stokes = v_rel * tau_p / chord
    m_imp = iwc * v_rel * ((3.0 * k_stokes) / (1.0 + 3.0 * k_stokes))

    ranked = NumericalSolver.rank_candidates(
        res, {"m_imp": m_imp, "v_rel": v_rel, "IWC": iwc, "tau_p": tau_p, "c": chord}, criterion="bic"
    )
    assert ranked[0].candidate_name == "buckingham_pi_hill"
    assert ranked[0].r2 > 0.999999


def test_sparse_pi_prunes_distractors() -> None:
    """Verify select_sparse_pi isolates the true Stokes Pi-group when distractors are present."""
    res = formulate(
        relations=["m_imp = fn(v_rel; IWC, tau_p, c, b_span, rho_i)"],
        dims={
            "m_imp": "M/(L^2*T)",
            "v_rel": "L/T",
            "IWC": "M/L^3",
            "tau_p": "T",
            "c": "L",
            "b_span": "L",
            "rho_i": "M/L^3",
        },
        constants=["IWC", "tau_p", "c", "b_span", "rho_i"],
        target="m_imp",
        wrt="v_rel",
        zero_initials=True,
    )
    assert len(res.pi_groups["m_imp"]) == 3  # 6 vars - 3 base dims = 3 Pi groups

    rng = np.random.default_rng(7)
    n = 60
    v_rel = rng.uniform(30.0, 160.0, n)
    iwc = rng.uniform(1e-3, 4e-3, n)
    tau_p = rng.uniform(1e-4, 6e-4, n)
    c = rng.uniform(0.05, 0.10, n)
    b_span = rng.uniform(0.10, 0.30, n)
    rho_i = rng.uniform(880.0, 920.0, n)
    k_stokes = v_rel * tau_p / c
    m_imp = 0.8 * iwc * v_rel * (k_stokes ** 0.5)

    sparse = NumericalSolver.select_sparse_pi(
        res,
        {
            "m_imp": m_imp,
            "v_rel": v_rel,
            "IWC": iwc,
            "tau_p": tau_p,
            "c": c,
            "b_span": b_span,
            "rho_i": rho_i,
        },
        max_groups=1,
        form="power",
        criterion="bic",
    )
    winner = sparse[0]
    assert winner.r2 > 0.999999
    # Distractor b_span and rho_i should NOT appear in the winning calibrated equation
    free_names = {s.name for s in winner.calibrated_equation.free_symbols}
    assert "b_span" not in free_names
    assert "rho_i" not in free_names


def test_stefan_ice_layer_growth_with_temperature_dim() -> None:
    """Verify unique zero-nullity derivation of Stefan sqrt(t) freezing law using Theta dimension."""
    res = formulate(
        relations=["delta = fn(t; k_eff, dT, H_vol)"],
        dims={
            "delta": "L",
            "t": "T",
            "k_eff": "M*L/(T^3*Theta)",
            "dT": "Theta",
            "H_vol": "M/(L*T^2)",
        },
        constants=["k_eff", "dT", "H_vol"],
        target="delta",
        wrt="t",
        zero_initials=True,
        unit_constants=True,
    )
    assert res.pi_groups == {}
    t_s = sp.Symbol("t", positive=True, real=True)
    k_s = sp.Symbol("k_eff", positive=True, real=True)
    dt_s = sp.Symbol("dT", positive=True, real=True)
    h_s = sp.Symbol("H_vol", positive=True, real=True)
    expected = 2 * sp.sqrt(k_s * dt_s * t_s / h_s)
    assert sp.simplify(res.primary.rhs - expected) == 0


def test_griffith_fracture_fractional_dimensions() -> None:
    """Verify Clairaut exactness on fractional base dimensions M * L^(-1/2) * T^(-2)."""
    res_ki = formulate(
        relations=["K_I = fn(sigma, a_flaw)"],
        dims={"K_I": "M/(L^(1/2)*T^2)", "sigma": "M/(L*T^2)", "a_flaw": "L"},
        target="K_I",
        zero_initials=True,
        unit_constants=True,
    )
    sigma_s = sp.Symbol("sigma", positive=True, real=True)
    a_s = sp.Symbol("a_flaw", positive=True, real=True)
    # Clairaut gives dK_I = 2*sqrt(a)*dsigma + (sigma/sqrt(a))*da => K_I = 2 * sigma * sqrt(a) (up to constant)
    ratio = sp.simplify(res_ki.primary.rhs / (sigma_s * sp.sqrt(a_s)))
    assert ratio.is_Number and float(ratio) > 0


def test_hybrid_ode_cohesive_chunks_vs_interfacial_slab() -> None:
    """Verify competing shedding triggers reproduce 40s cohesive chunks vs. 90s interfacial slab."""
    t_sym, delta_sym = sp.symbols("t delta", real=True)
    rate_eq = sp.Eq(sp.Derivative(delta_sym, t_sym), sp.Float(1.0e-4))
    c_load = 2.0e7

    sol_chunks = NumericalSolver.integrate_hybrid_ode(
        rate_eq,
        wrt="t",
        span=(0.0, 120.0),
        y0=0.0,
        events=[
            SheddingEventSpec("cohesive_fracture", lambda t, d: c_load * (d ** 3) - 1.28, shed_fraction=0.42),
            SheddingEventSpec("interfacial_adhesive", lambda t, d: c_load * (d ** 3) - 20.0, shed_fraction=0.95),
        ],
        target_name="delta",
    )
    assert sol_chunks.success
    assert sol_chunks.n_events == 5
    assert all(ev.mode == "cohesive_fracture" for ev in sol_chunks.events)
    assert np.isclose(sol_chunks.events[0].time, 40.0, atol=1e-2)
    assert np.isclose(sol_chunks.events[1].time, 56.8, atol=1e-2)

    sol_slab = NumericalSolver.integrate_hybrid_ode(
        rate_eq,
        wrt="t",
        span=(0.0, 120.0),
        y0=0.0,
        events=[
            SheddingEventSpec("cohesive_fracture", lambda t, d: c_load * (d ** 3) - 20.0, shed_fraction=0.42),
            SheddingEventSpec(
                "interfacial_adhesive",
                lambda t, d: c_load * (d ** 3) - (30.0 * np.exp(-0.008026 * t)),
                shed_fraction=0.95,
            ),
        ],
        target_name="delta",
    )
    assert sol_slab.success
    assert sol_slab.n_events == 1
    assert sol_slab.events[0].mode == "interfacial_adhesive"
    assert np.isclose(sol_slab.events[0].time, 90.0, atol=0.1)
    assert np.isclose(sol_slab.events[0].post_value / sol_slab.events[0].pre_value, 0.05, atol=1e-4)
