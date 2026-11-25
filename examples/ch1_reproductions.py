"""
End-to-end demonstrations of rosolver reproducing ch1.tex and generalized multi-hop scenarios.

Run with:
    python -m examples.ch1_reproductions
"""

from __future__ import annotations

import numpy as np

from rosolver import NumericalSolver, SymbolicSolver, System, formulate


def demo_user_nested_dag() -> None:
    print("=" * 76)
    print("1A. USER EXAMPLE (Dimension-Free): g = fn(a, b, c), b = fn(f)")
    print("=" * 76)
    res_free = formulate(
        relations=["g = fn(a, b, c)", "b = fn(f)"],
        target="g",
        wrt="f",
    )
    print(res_free.summary())

    print("\n" + "=" * 76)
    print("1B. USER EXAMPLE (Dimensional + Numerical Fit): g = fn(a, b, c), b = fn(f; grav)")
    print("=" * 76)
    res_dim = formulate(
        relations=["g = fn(a, b, c)", "b = fn(f; grav)"],
        dims={
            "g": "M*L^2/T^3",
            "a": "M/L^3",
            "b": "L/T",
            "c": "L^2",
            "f": "L",
            "grav": "L/T^2",
        },
        constants=["grav"],
        target="g",
        wrt="f",
        zero_initials=True,
    )
    print(res_dim.summary())

    # Numerical calibration against synthetic wind/waterfall dataset: g = 0.5 * a * c * (2 * grav * f)^(3/2)
    rng = np.random.default_rng(42)
    n = 60
    a_vals = rng.uniform(1.0, 1.4, n)
    c_vals = rng.uniform(2.0, 10.0, n)
    f_vals = rng.uniform(1.0, 25.0, n)
    grav_vals = np.full(n, 9.81)
    true_b = np.sqrt(2.0 * grav_vals * f_vals)
    true_g = 0.5 * a_vals * c_vals * (true_b ** 3)

    fit = NumericalSolver.fit(
        res_dim,
        data={"g": true_g, "a": a_vals, "c": c_vals, "f": f_vals, "grav": grav_vals},
        fixed_params={"kappa_b": float(np.sqrt(0.5))},  # 2 * kappa_b = sqrt(2) => b = sqrt(2*grav*f)
    )
    print("\nNumerical Calibration:")
    print(" ", fit.summary())


def demo_ch1_kinematics() -> None:
    print("\n" + "=" * 76)
    print("2. ch1.tex Sec 1: Linear & Non-Linear Motion s = fn(v), v = fn(t)")
    print("=" * 76)
    res = formulate(
        relations=["s = fn(v)", "v = fn(t)"],
        dims={"s": "L", "v": "L/T", "t": "T", "a": "L/T^2"},
        constants=["a"],
        rate_identities=[("s", "t", "v"), ("v", "t", "a")],
        target="s",
        wrt="t",
        unit_constants=True,
    )
    print(res.summary())


def demo_ch1_variable_mass_momentum() -> None:
    print("\n" + "=" * 76)
    print("3. ch1.tex Sec 2: Variable-Mass Momentum p = fn(m, v), m = fn(t), v = fn(t)")
    print("=" * 76)
    sys = System.from_spec(
        relations=["p = fn(m, v)"],
        dims={"p": "M*L/T", "m": "M", "v": "L/T"},
    )
    res = SymbolicSolver(sys).derive(target="p", zero_initials=True, unit_constants=True)
    print(res.summary())


def demo_ch1_kinetic_energy_and_calibration() -> None:
    print("\n" + "=" * 76)
    print("4. ch1.tex Sec 3.1: Kinetic Energy K = fn(m, v) + Numerical Discovery of 1/2")
    print("=" * 76)
    res = formulate(
        relations=["K = fn(m, v)"],
        dims={"K": "M*L^2/T^2", "m": "M", "v": "L/T"},
        target="K",
        zero_initials=True,
    )
    print(res.summary())

    rng = np.random.default_rng(7)
    m_data = rng.uniform(0.5, 20.0, 40)
    v_data = rng.uniform(1.0, 50.0, 40)
    k_data = 0.5 * m_data * (v_data ** 2)
    fit = NumericalSolver.fit(res, {"K": k_data, "m": m_data, "v": v_data})
    print("\nCalibrated from observations:")
    print(" ", fit.summary())


def demo_blasius_boundary_layer_pi_discovery() -> None:
    print("\n" + "=" * 76)
    print("5. ch1.tex Eq. 2.23: Blasius Boundary Layer delta = fn(x, rho, v, mu)")
    print("   Automatic Buckingham-Pi (Re) Discovery + Nonlinear Exponent Fit (-1/2)")
    print("=" * 76)
    res = formulate(
        relations=["delta = fn(x; rho, v, mu)"],
        dims={
            "delta": "L",
            "x": "L",
            "rho": "M/L^3",
            "v": "L/T",
            "mu": "M/(L*T)",
        },
        constants=["rho", "v", "mu"],
        target="delta",
        wrt="x",
        zero_initials=True,
    )
    print(res.summary())

    # Synthetic laminar plate measurements following Blasius: delta = 5.0 * x / sqrt(Re)
    rng = np.random.default_rng(99)
    n = 80
    x_v = rng.uniform(0.1, 2.0, n)
    rho_v = rng.uniform(1.0, 1.3, n)
    v_v = rng.uniform(2.0, 15.0, n)
    mu_v = rng.uniform(1.5e-5, 2.2e-5, n)
    re_num = (rho_v * v_v * x_v) / mu_v
    delta_obs = 5.0 * x_v / np.sqrt(re_num)

    ranked = NumericalSolver.rank_candidates(
        res,
        data={"delta": delta_obs, "x": x_v, "rho": rho_v, "v": v_v, "mu": mu_v},
        criterion="bic",
    )
    print("\nCandidate Ranking by BIC:")
    for rank_idx, fr in enumerate(ranked, 1):
        print(f"  #{rank_idx}: {fr.summary()}")


def demo_non_physics_domain() -> None:
    print("\n" + "=" * 76)
    print("6. Non-Physics Domain (SaaS Economics): Revenue R = fn(Users, Price), Users = fn(Spend; CAC)")
    print("=" * 76)
    res = formulate(
        relations=["R = fn(U, P)", "U = fn(S; CAC)"],
        dims={
            "R": "USD/Month",
            "U": "Users",
            "P": "USD/(Users*Month)",
            "S": "USD",
            "CAC": "USD/Users",
        },
        constants=["CAC"],
        target="R",
        wrt="S",
    )
    print(res.summary())


if __name__ == "__main__":
    demo_user_nested_dag()
    demo_ch1_kinematics()
    demo_ch1_variable_mass_momentum()
    demo_ch1_kinetic_energy_and_calibration()
    demo_blasius_boundary_layer_pi_discovery()
    demo_non_physics_domain()
