"""
Domain-Agnostic Verification & Stability Showcase (ASME V&V 20 / MMS / Conservation / Coupling).

Demonstrates rosolver's generic verification engine across four distinct scientific domains:
  1. Fluid Mechanics PDE: Viscous Burgers MMS Code Verification + 3-Grid Roache GCI Study
  2. Environmental / Chemical Engineering FVM: Coupled 2-Species Reactive Plume Conservation Audit
  3. Epidemiology & Classical Mechanics ODEs: Symbolic Invariants, Symplectic Drift & dt Sensitivity
  4. Chemical Kinetics & Multi-Physics Coupling: Stiff Jacobian Spectrum, Aitken Stabilization & Co-Sim

Run with:
    python -m examples.verification_suite
"""

from __future__ import annotations

import numpy as np

from rosolver import (
    BC,
    ConservationLaw1D,
    ODESystem,
    analyze_stiffness,
    audit_ode_invariants,
    audit_pde_conservation,
    cosimulate_coupled_odes,
    grid_convergence_study,
    mms_study,
    plot_conservation_audit,
    plot_gci_study,
    plot_mms_convergence,
    plot_stability_spectrum,
    solve_partitioned_fixed_point,
    timestep_study,
)

OUT = "figures"


def demo_1_mms_and_grid_independence() -> None:
    print("=" * 78)
    print("1. CODE & SOLUTION VERIFICATION: Viscous Burgers PDE (MMS + Roache GCI)")
    print("=" * 78)
    burgers = ConservationLaw1D("u", flux="u**2 / 2", diffusivity="nu")
    mms_rep = mms_study(
        burgers,
        exact="1.0 + 0.3 * sin(2 * pi * (x - 0.5 * t))",
        domain=(0.0, 1.0),
        t_end=0.10,
        cells=(32, 64, 128, 256),
        params={"nu": 0.015},
        bc_kind="periodic",
        reconstruction="linear",
        cfl=0.15,
    )
    print(mms_rep.summary())
    p_mms = plot_mms_convergence(mms_rep, path=f"{OUT}/vv_burgers_mms.png")
    print(f"  Saved MMS convergence figure: {p_mms}")

    # Solution verification (no exact closed form needed): probe steepening wave at x = 0.65
    gci_rep = grid_convergence_study(
        lambda n: burgers.solve(
            (0.0, 1.0),
            n,
            t_end=0.18,
            initial="1.0 + 0.4 * sin(2 * pi * x)",
            bc="periodic",
            params={"nu": 0.02},
            reconstruction="linear",
            cfl=0.2,
        ).probe(0.65),
        cells=(80, 160, 320),
        formal_order=2.0,
    )
    print("\n" + gci_rep.summary())
    p_gci = plot_gci_study(gci_rep, path=f"{OUT}/vv_burgers_gci.png", qoi_label="u(x=0.65, t=0.18)")
    print(f"  Saved GCI band figure: {p_gci}")


def demo_2_reactive_plume_conservation_audit() -> None:
    print("\n" + "=" * 78)
    print("2. DISCRETE FVM CONSERVATION AUDIT: 2-Species Advection-Diffusion-Reaction Plume")
    print("=" * 78)
    plume = ConservationLaw1D(
        fields=["C_A", "C_B"],
        flux={"C_A": "u_flow * C_A", "C_B": "u_flow * C_B"},
        diffusivity={"C_A": "D_A", "C_B": "D_B"},
        source={"C_A": "-k_rxn * C_A * C_B", "C_B": "k_rxn * C_A * C_B"},
    )
    sol = plume.solve(
        domain=(0.0, 5.0),
        n_cells=100,
        t_end=1.2,
        initial={"C_A": "0.2 * exp(-4 * (x - 1.0)**2)", "C_B": "0.5"},
        bc={
            "left": BC.dirichlet({"C_A": "1.0 + 0.3 * sin(2 * t)", "C_B": "0.5"}),
            "right": "outflow",
        },
        params={"u_flow": 1.2, "D_A": 0.04, "D_B": 0.025, "k_rxn": 0.8},
        scheme="ssprk3",
        cfl=0.35,
    )
    audit = audit_pde_conservation(sol, tol=1e-12)
    print(audit.summary())
    p_cons = plot_conservation_audit(audit, path=f"{OUT}/vv_plume_conservation.png")
    print(f"  Saved conservation residual figure: {p_cons}")


def demo_3_ode_invariants_and_timestep_sensitivity() -> None:
    print("\n" + "=" * 78)
    print("3. SYMBOLIC INVARIANTS & TIME-STEP SENSITIVITY: Epidemiology SEIR + Oscillator")
    print("=" * 78)
    seir = ODESystem(
        {
            "S": "-beta * S * I / N",
            "E": "beta * S * I / N - sigma * E",
            "I": "sigma * E - gamma * I",
            "R": "gamma * I",
        }
    )
    p_seir = {"beta": 0.4, "sigma": 0.2, "gamma": 0.1, "N": 10000.0}
    y0_seir = {"S": 9950.0, "E": 30.0, "I": 20.0, "R": 0.0}

    traj_rk4 = seir.solve((0.0, 60.0), y0_seir, params=p_seir, method="rk4", dt=0.25)
    inv_rep = audit_ode_invariants(
        seir, traj_rk4, {"total_population": "S + E + I + R"}, params=p_seir, tol=1e-10
    )
    print(inv_rep.summary())

    dt_rep = timestep_study(
        seir,
        dts=(1.0, 0.5, 0.25),
        t_span=(0.0, 60.0),
        y0=y0_seir,
        params=p_seir,
        scheme="rk4",
        state="I",
    )
    print("\n" + dt_rep.summary())


def demo_4_stiffness_and_partitioned_coupling_stability() -> None:
    print("\n" + "=" * 78)
    print("4. STIFFNESS & PARTITIONED COUPLING STABILITY: FVM Spectrum, Aitken & Co-Sim")
    print("=" * 78)
    # Semi-discrete advection-diffusion spectrum vs RK4 stability boundary
    adv_diff = ConservationLaw1D("u", flux="c * u", diffusivity="nu")
    rhs_mol, u0_mol = adv_diff.semidiscrete(
        (0.0, 1.0), 30, initial="sin(2 * pi * x)", bc="periodic", params={"c": 1.0, "nu": 0.015}
    )
    st_rep = analyze_stiffness(rhs_mol, u0_mol)
    print(st_rep.summary())
    dt_safe = 0.75 * st_rep.max_stable_dts["rk4"]
    p_stab = plot_stability_spectrum(st_rep, dt=dt_safe, scheme="rk4", path=f"{OUT}/vv_stability_spectrum.png")
    print(f"  Saved RK4 stability spectrum figure: {p_stab}")

    # Algebraic partitioned coupling with strong feedback (spectral radius 1.5 -> raw Picard blows up!)
    strong_feedback = lambda u: np.array([3.0 - 1.5 * u[0], 1.0 + 0.5 * u[0] - 1.3 * u[1]])
    fp_raw = solve_partitioned_fixed_point(strong_feedback, [0.0, 0.0], max_iter=20, acceleration="none")
    fp_ait = solve_partitioned_fixed_point(strong_feedback, [0.0, 0.0], max_iter=20, acceleration="aitken")
    print("\n" + fp_raw.summary())
    print(fp_ait.summary())

    # Dynamic loose vs strong co-simulation of two coupled ODE subsystems
    sub_fast = ODESystem({"T_f": "-3.0 * (T_f - T_s) + 1.5 * sin(2 * t)"})
    sub_solid = ODESystem({"T_s": "1.8 * (T_f - T_s) - 0.4 * T_s"})
    y0_cs = {"T_f": 1.0, "T_s": 0.0}
    cs_loose = cosimulate_coupled_odes(sub_fast, sub_solid, (0.0, 4.0), y0_cs, dt_macro=0.1, mode="loose")
    cs_strong = cosimulate_coupled_odes(sub_fast, sub_solid, (0.0, 4.0), y0_cs, dt_macro=0.1, mode="strong")
    print("\n" + cs_loose.summary())
    print(cs_strong.summary())


if __name__ == "__main__":
    demo_1_mms_and_grid_independence()
    demo_2_reactive_plume_conservation_audit()
    demo_3_ode_invariants_and_timestep_sensitivity()
    demo_4_stiffness_and_partitioned_coupling_stability()
