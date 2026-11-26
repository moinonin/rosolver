"""
Turbomachinery & Heat-Exchanger Ice Accretion and Shedding Discovery Suite.

Grounded directly in extracted physics and experimental benchmarks from the 'Academia' project:
  1. ICICLE / OTI Multi-Physics Impingement & Saturating Collection Efficiency (Langmuir / AIChE 2016)
  2. SnowTek Non-Spherical Disc Terminal & Reversed Hindered Settling Velocity (RHSV)
  3. WINICE (CEJ 2017) Stefan Conductive Layer Growth & SnowTek 50% HTC Thermal Fouling
  4. Nested Rotor Accretion -> Centrifugal & Aerodynamic Interfacial Stress DAG (OTI Slide 04 / Cross 2024)
  5. Irwin-Griffith Cohesive Fracture & WINICE Occluded-Liquid Defect Scaling (K_I >= K_Ic)
  6. Hybrid Event-Driven Multi-Cycle Shedding Diagnostic: 40s Cohesive Chunks vs. 90s Interfacial Slab
  7. Front-Fixing Stefan Moving Boundary: exact Neumann validation + wet/dry switch
  8. Coupled Stefan Solver: runback-film Messinger closure + hybrid shedding events

Run with:
    python -m examples.icing_turbomachinery
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from rosolver import (
    NumericalSolver,
    SheddingEventSpec,
    Stefan1D,
    SymbolicSolver,
    stefan_shedding_event,
    System,
    formulate,
)


def case1_impingement_and_sparse_saturating_pi() -> None:
    print("=" * 78)
    print("CASE 1: Ice-Crystal Impingement & Saturating Collection Efficiency (OTI Module 1)")
    print("        Automatic Stokes Number Discovery + Hill/Rosin-Rammler Saturation")
    print("=" * 78)
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
    print(res.summary())

    # Synthetic compressor-vane impingement data (50-180 m/s): beta(K) = K / (0.25 + K) = 4*K / (1 + 4*K)
    rng = np.random.default_rng(101)
    n = 80
    v_rel = rng.uniform(30.0, 180.0, n)
    iwc = rng.uniform(0.5e-3, 4.0e-3, n)      # 0.5 - 4.0 g/m^3
    tau_p = rng.uniform(5.0e-5, 8.0e-4, n)    # crystal relaxation time (s)
    chord = rng.uniform(0.04, 0.12, n)        # compressor vane chord (m)
    stokes_k = (v_rel * tau_p) / chord
    beta_coll = (4.0 * stokes_k) / (1.0 + 4.0 * stokes_k)
    m_imp_obs = iwc * v_rel * beta_coll

    data = {"m_imp": m_imp_obs, "v_rel": v_rel, "IWC": iwc, "tau_p": tau_p, "c": chord}
    ranked = NumericalSolver.rank_candidates(res, data=data, criterion="bic")
    print("\nBIC Ranking Across Monomial, Power-Law, Affine, Hill & Rosin-Rammler Candidates:")
    for idx, fr in enumerate(ranked, 1):
        print(f"  #{idx}: {fr.summary()}")

    # Also test high-nullity sparse Pi selection with irrelevant distractor variables (span b_s, rho_i)
    res_high = formulate(
        relations=["m_imp = fn(v_rel; IWC, tau_p, c, b_s, rho_i)"],
        dims={
            "m_imp": "M/(L^2*T)",
            "v_rel": "L/T",
            "IWC": "M/L^3",
            "tau_p": "T",
            "c": "L",
            "b_s": "L",
            "rho_i": "M/L^3",
        },
        constants=["IWC", "tau_p", "c", "b_s", "rho_i"],
        target="m_imp",
        wrt="v_rel",
        zero_initials=True,
    )
    data_high = {
        **data,
        "b_s": rng.uniform(0.05, 0.25, n),
        "rho_i": np.full(n, 917.0),
    }
    sparse_ranked = NumericalSolver.select_sparse_pi(
        res_high, data=data_high, max_groups=1, form="hill", criterion="bic"
    )
    print("\nSparse 1-Group Pi Selection (Pruning Distractor Variables b_s, rho_i):")
    print("  Winner:", sparse_ranked[0].summary())


def case2_snowtek_disc_rhsv() -> None:
    print("\n" + "=" * 78)
    print("CASE 2: Non-Spherical Disc Crystal Velocity & RHSV (SnowTek Sec. 2.2.2)")
    print("=" * 78)
    res = formulate(
        relations=["u_rhsv = fn(H_i; g_red)"],
        dims={"u_rhsv": "L/T", "H_i": "L", "g_red": "L/T^2"},
        constants=["g_red"],
        target="u_rhsv",
        wrt="H_i",
        zero_initials=True,
    )
    print(res.summary())

    # SnowTek Table 3 prototype numbers: D_i = 3 mm, aspect ratio gamma = 0.35 => H_i = 1.05 mm
    rho_i = 917.0
    rho_s = 1020.36
    g_val = 9.81
    g_red_val = g_val * (rho_s - rho_i) / rho_s
    c_d = 1.93
    eps = 0.94
    psi_p = 0.79
    prefactor = np.sqrt(2.0 / c_d) * (eps ** 2) * psi_p  # theoretical RHSV dimensionless factor

    h_grid = np.linspace(0.25e-3, 3.0e-3, 40)
    u_obs = prefactor * np.sqrt(g_red_val * h_grid)
    fit = NumericalSolver.fit(res, {"u_rhsv": u_obs, "H_i": h_grid, "g_red": np.full_like(h_grid, g_red_val)})
    print("\nCalibrated on SnowTek Disc Swarm Regime (Re=32.2, C_D=1.93, eps=0.94, Psi_p=0.79):")
    print(" ", fit.summary())

    u_at_3mm = float(
        NumericalSolver.evaluate(
            fit.calibrated_equation,
            {"H_i": 0.00105, "g_red": g_red_val},
        )
    )
    print(f"  Predicted RHSV at D_i=3mm, H/D=0.35 (H_i=1.05mm): {u_at_3mm:.4f} m/s (SnowTek Table 3: 0.0233 m/s)")


def case3_winice_stefan_growth_and_htc_fouling() -> None:
    print("\n" + "=" * 78)
    print("CASE 3: Coupled Stefan Layer Growth & Thermal Fouling (WINICE CEJ 2017 + SnowTek)")
    print("=" * 78)
    res = formulate(
        relations=["delta = fn(t; k_eff, dT, H_vol)"],
        dims={
            "delta": "L",
            "t": "T",
            "k_eff": "M*L/(T^3*Theta)",
            "dT": "Theta",
            "H_vol": "M/(L*T^2)",  # volumetric latent heat rho_i * L_f
        },
        constants=["k_eff", "dT", "H_vol"],
        target="delta",
        wrt="t",
        zero_initials=True,
    )
    print(res.summary())

    # Exact Stefan analytical prefactor: delta(t) = sqrt(2 * k_eff * dT / (rho_i * L_f) * t)
    # => 2 * kappa_delta = sqrt(2) => kappa_delta = sqrt(2)/2 = 0.707107
    t_vals = np.linspace(3600.0, 86400.0, 50)  # 1 h to 24 h (WINICE Table 1)
    k_eff = 2.22        # W/(m*K) ice thermal conductivity
    rho_i = 917.0       # kg/m^3
    l_f = 3.34e5        # J/kg latent heat of fusion
    h_vol = rho_i * l_f
    d_t = 2.95          # subcooling K (e.g. T_air = -4.13 C vs T_f = -1.18 C in WINICE)
    delta_stefan = np.sqrt(2.0 * k_eff * d_t / h_vol * t_vals)

    fit = NumericalSolver.fit(
        res,
        {
            "delta": delta_stefan,
            "t": t_vals,
            "k_eff": np.full_like(t_vals, k_eff),
            "dT": np.full_like(t_vals, d_t),
            "H_vol": np.full_like(t_vals, h_vol),
        },
    )
    print("\nStefan Moving-Boundary Calibration:")
    print(" ", fit.summary())

    # SnowTek 50% HTC drop verification: h_eff = 1 / (1/h_0 + delta/k_eff) = 0.5 * h_0 when delta = k_eff / h_0
    h_0 = 160.0  # W/(m^2*K) clean wall heat transfer coefficient
    delta_half = k_eff / h_0
    print(
        f"  SnowTek 50% HTC Fouling Threshold (h_0={h_0} W/m²K, k_eff={k_eff} W/mK): "
        f"delta_50% = {delta_half * 1e3:.2f} mm"
    )


def case4_nested_rotor_shedding_dag() -> None:
    print("\n" + "=" * 78)
    print("CASE 4: Nested Turbomachinery Rotor Accretion -> Centrifugal Shedding Stress DAG")
    print("        tau_cent = fn(m_area, omega, r),  m_area = fn(t; mdot_ice)")
    print("=" * 78)
    res = formulate(
        relations=["tau_cent = fn(m_area, omega, r)", "m_area = fn(t; mdot_ice)"],
        dims={
            "tau_cent": "M/(L*T^2)",
            "m_area": "M/L^2",
            "omega": "1/T",
            "r": "L",
            "t": "T",
            "mdot_ice": "M/(L^2*T)",
        },
        constants=["mdot_ice"],
        target="tau_cent",
        wrt="t",
        zero_initials=True,
        unit_constants=True,
    )
    print(res.summary())

    # Critical shedding time when tau_cent(t_shed) == tau_adh(T, alpha_liq)
    t_sym = sp.Symbol("t", positive=True, real=True)
    tau_adh_sym = sp.Symbol("tau_adh", positive=True, real=True)
    t_shed_sol = sp.solve(sp.Eq(res.primary.rhs, tau_adh_sym), t_sym)[0]
    print(f"\nSymbolic Critical Shedding Time (tau_cent == tau_adh):\n  t_shed = {t_shed_sol}")


def case5_cohesive_fracture_and_winice_flaws() -> None:
    print("\n" + "=" * 78)
    print("CASE 5: Cohesive Bulk Fracture & WINICE Occluded-Liquid Defect Scaling")
    print("        Fractional Base Dimensions [M * L^(-1/2) * T^(-2)] -> Griffith Law")
    print("=" * 78)
    res_ki = formulate(
        relations=["K_I = fn(sigma, a_flaw)"],
        dims={"K_I": "M/(L^(1/2)*T^2)", "sigma": "M/(L*T^2)", "a_flaw": "L"},
        target="K_I",
        zero_initials=True,
    )
    print("--- 5A. Stress Intensity Factor K_I = fn(sigma, a_flaw) ---")
    print(res_ki.summary())

    res_kic = formulate(
        relations=["K_Ic = fn(E_prime, G_c)"],
        dims={"K_Ic": "M/(L^(1/2)*T^2)", "E_prime": "M/(L*T^2)", "G_c": "M/T^2"},
        target="K_Ic",
        zero_initials=True,
    )
    print("\n--- 5B. Cohesive Fracture Toughness K_Ic = fn(E_prime, G_c) ---")
    print(res_kic.summary())

    # Calibrate geometry factor Y * sqrt(pi) (Y = 1.12 edge flaw => kappa = 1.98515)
    rng = np.random.default_rng(55)
    sigma_v = rng.uniform(0.2e6, 2.5e6, 40)
    a_v = rng.uniform(0.2e-3, 3.0e-3, 40)  # 0.2 - 3.0 mm WINICE occluded-liquid flaw size
    ki_v = 1.12 * np.sqrt(np.pi) * sigma_v * np.sqrt(a_v)
    fit_ki = NumericalSolver.fit(res_ki, {"K_I": ki_v, "sigma": sigma_v, "a_flaw": a_v})
    print("\nEdge-Flaw Stress Intensity Calibration (Y=1.12):")
    print(" ", fit_ki.summary())


def case6_hybrid_shedding_curveball_diagnostic() -> None:
    print("\n" + "=" * 78)
    print("CASE 6: Hybrid Event-Driven Shedding Diagnostic (OTI Playbook Curveball B)")
    print("        40s Repeated Cohesive Chunks vs. 90s Single Interfacial Glaze Slab")
    print("=" * 78)
    t_sym, delta_sym = sp.symbols("t delta", real=True)
    g_acc = 1.0e-4  # 0.10 mm/s net ice thickness accretion rate
    rate_eq = sp.Eq(sp.Derivative(delta_sym, t_sym), sp.Float(g_acc))

    # Aerodynamic + centrifugal energy release rates (J/m^2) scaling as C * delta^3
    # At delta = 4.0 mm (t = 40 s): G_coh = 1.28 J/m^2
    c_load = 2.0e7  # => C * (0.004)^3 = 1.28 J/m^2

    # Scenario A: Porous ice with WINICE inclusions (low bulk G_c = 1.28 J/m^2, cold strong interface G_a = 20 J/m^2)
    events_chunks = [
        SheddingEventSpec(
            name="cohesive_fracture",
            margin_expr=lambda t, d: c_load * (d ** 3) - 1.28,
            shed_fraction=0.42,  # 42% upper chunk snaps off, base layer remains
        ),
        SheddingEventSpec(
            name="interfacial_adhesive",
            margin_expr=lambda t, d: c_load * (d ** 3) - 20.0,
            shed_fraction=0.95,
        ),
    ]
    sol_a = NumericalSolver.integrate_hybrid_ode(
        rate_eq, wrt="t", span=(0.0, 120.0), y0=0.0, events=events_chunks, target_name="delta", dt_max=0.5
    )

    # Scenario B: Dense glaze ice (high bulk G_c = 20 J/m^2) with warm interfacial liquid film softening G_a(t)
    # G_a(t) decays from 30 J/m^2 as warm wall melt film forms, crossing c_load * delta(t)^3 at t = 90.0 s (delta = 9 mm => 14.58 J/m^2)
    events_slab = [
        SheddingEventSpec(
            name="cohesive_fracture",
            margin_expr=lambda t, d: c_load * (d ** 3) - 20.0,
            shed_fraction=0.42,
        ),
        SheddingEventSpec(
            name="interfacial_adhesive",
            margin_expr=lambda t, d: c_load * (d ** 3) - (30.0 * np.exp(-0.008026 * t)),
            shed_fraction=0.95,  # 95% clean interfacial slab detachment
        ),
    ]
    sol_b = NumericalSolver.integrate_hybrid_ode(
        rate_eq, wrt="t", span=(0.0, 120.0), y0=0.0, events=events_slab, target_name="delta", dt_max=0.5
    )

    print("Scenario A (Porous Bulk / Cold Bond -> Early Cohesive Chunks):")
    for ev in sol_a.events:
        print(
            f"  t = {ev.time:6.2f} s | mode = {ev.mode:20s} | "
            f"delta: {ev.pre_value * 1e3:5.2f} mm -> {ev.post_value * 1e3:5.2f} mm "
            f"(shed {ev.shed_amount * 1e3:5.2f} mm)"
        )

    print("\nScenario B (Dense Glaze / Warm Interfacial Liquid Film -> Delayed Whole-Slab Debonding):")
    for ev in sol_b.events:
        print(
            f"  t = {ev.time:6.2f} s | mode = {ev.mode:20s} | "
            f"delta: {ev.pre_value * 1e3:5.2f} mm -> {ev.post_value * 1e3:5.2f} mm "
            f"(shed {ev.shed_amount * 1e3:5.2f} mm)"
        )


def case7_stefan_front_tracking() -> None:
    print("\n" + "=" * 78)
    print("CASE 7: Stefan Moving-Boundary Ice Growth (eq. 3.4) — Front-Fixing PDE")
    print("        Exact Neumann validation + convective (h_c) branch")
    print("=" * 78)

    # (a) h_c = 0: the full PDE must reproduce the self-similar Neumann front.
    st_exact = Stefan1D(k_eff=2.22, rho=917.0, cp=2100.0, Lf=3.34e5,
                        T_wall=-6.0, T_f=0.0, h_c=0.0, n_xi=200)
    sol_exact = st_exact.solve(3600.0, delta0=1e-5, n_out=20)
    exact = float(st_exact.neumann_delta(3600.0))
    err = abs(sol_exact.delta[-1] - exact) / exact
    print(f"h_c=0, Ste={st_exact.stefan_number:.4f}:")
    print(f"  PDE delta(1 h) = {sol_exact.delta[-1] * 1e3:.4f} mm | "
          f"Neumann = {exact * 1e3:.4f} mm | rel. error = {err:.2e}")

    # (b) Convective surface with the WINICE/SnowTek numbers, over 24 h.
    st = Stefan1D(k_eff=2.22, rho=917.0, cp=2100.0, Lf=3.34e5,
                  T_wall=-4.13, T_inf=-4.13, T_f=-1.18, h_c=160.0, n_xi=200)
    sol = st.solve(86400.0, delta0=1e-5, n_out=50)
    d_star = st.k_eff * (st.T_f - st.T_wall) / (st.h_c * (st.T_f - st.T_inf))
    print(f"\nConvective branch (h_c=160 W/m^2K):")
    print(f"  Conduction/convection crossover delta* = {d_star * 1e3:.2f} mm")
    print("  t [h]   delta_PDE [mm]   delta_quasi [mm]")
    for t in (1.0, 6.0, 12.0, 24.0):
        t_s = t * 3600.0
        print(f"  {t:5.1f}   {sol.delta_at(t_s) * 1e3:12.3f}   "
              f"{float(st.quasi_steady_delta(t_s, 1e-5)) * 1e3:14.3f}")


def case8_stefan_film_and_hybrid_shedding() -> None:
    print("\n" + "=" * 78)
    print("CASE 8: Coupled Stefan Solver — Runback Film + Hybrid Shedding")
    print("=" * 78)

    common = dict(k_eff=2.22, rho=917.0, cp=2100.0, Lf=3.34e5,
                  T_wall=-4.13, T_inf=-4.13, T_f=-1.18, h_c=160.0)

    # (a) Full Messinger surface balance with a lumped runback film.
    st_film = Stefan1D(**common, mdot_imp=6e-3, cp_imp=4186.0,
                       V_imp=30.0, tau_drain=30.0, n_xi=120)
    sol_film = st_film.solve(2 * 3600.0, delta0=1e-5, n_out=80)
    print(sol_film.summary())
    h_steady = (st_film.tau_drain / st_film.rho_w
                * (st_film.mdot_imp - sol_film.mdot_ice[-1]
                   - st_film.mdot_evap))
    print(f"  Steady-film prediction h_f* = {h_steady * 1e3:.4f} mm "
          f"(computed {sol_film.film[-1] * 1e3:.4f} mm)")

    # (b) Accretion with discrete centrifugal shedding events.
    st_hyb = Stefan1D(**common, mdot_imp=3e-3, n_xi=100)
    hyb = st_hyb.solve_with_shedding(
        [stefan_shedding_event(delta_crit=0.01, shed_fraction=0.8)],
        12 * 3600.0, delta0=1e-5, n_out=240,
    )
    print(f"\nHybrid front tracking: {len(hyb.events)} shedding events "
          f"at delta_crit = 10 mm")
    for ev in hyb.events[:5]:
        print(f"  t = {ev.time:7.1f} s | delta {ev.pre_value * 1e3:5.2f} -> "
              f"{ev.post_value * 1e3:5.2f} mm")
    if len(hyb.events) > 5:
        print(f"  ... ({len(hyb.events) - 5} more)")


if __name__ == "__main__":
    case2_snowtek_disc_rhsv()
    case3_winice_stefan_growth_and_htc_fouling()
    case4_nested_rotor_shedding_dag()
    case5_cohesive_fracture_and_winice_flaws()
    case6_hybrid_shedding_curveball_diagnostic()
    case7_stefan_front_tracking()
    case8_stefan_film_and_hybrid_shedding()
