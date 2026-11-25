"""
Auto-simulate and plot final equations (figures saved to ./figures).

Run with:
    python -m examples.auto_simulate
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from rosolver import (
    NumericalSolver,
    SheddingEventSpec,
    formulate,
    plot_hybrid,
    plot_ranked,
    plot_simulation,
    simulate,
)

OUT = "figures"


def demo_waterfall_power() -> None:
    """User's g = fn(a, b, c), b = fn(f; grav): derive -> simulate -> plot a family over density."""
    res = formulate(
        relations=["g = fn(a, b, c)", "b = fn(f; grav)"],
        dims={"g": "M*L^2/T^3", "a": "M/L^3", "b": "L/T", "c": "L^2", "f": "L", "grav": "L/T^2"},
        constants=["grav"],
        target="g",
        wrt="f",
        zero_initials=True,
    )
    sim = simulate(
        res,
        sweep={"f": (0.5, 25.0, 200)},
        values={"c": 2.0, "grav": 9.81, "kappa_b": 2 ** -0.5, "kappa_g": 0.5},
        family={"a": [1.0, 500.0, 1000.0]},
    )
    print("Saved", plot_simulation(sim, path=f"{OUT}/waterfall_power.png"))


def demo_blasius_ranked() -> None:
    """Fit all candidates, then plot them against the data."""
    res = formulate(
        relations=["delta = fn(x; rho, v, mu)"],
        dims={"delta": "L", "x": "L", "rho": "M/L^3", "v": "L/T", "mu": "M/(L*T)"},
        constants=["rho", "v", "mu"],
        target="delta",
        wrt="x",
        zero_initials=True,
    )
    rng = np.random.default_rng(99)
    n = 80
    x = rng.uniform(0.1, 2.0, n)
    rho = np.full(n, 1.2)
    v = np.full(n, 8.0)
    mu = np.full(n, 1.8e-5)
    delta = 5.0 * x / np.sqrt(rho * v * x / mu)
    data = {"delta": delta, "x": x, "rho": rho, "v": v, "mu": mu}
    ranked = NumericalSolver.rank_candidates(res, data, criterion="bic")
    print("Saved", plot_ranked(ranked, data, "x", "delta", path=f"{OUT}/blasius_ranked.png"))


def demo_shedding_sawtooth() -> None:
    t, d = sp.symbols("t delta", real=True)
    rate = sp.Eq(sp.Derivative(d, t), sp.Float(1.0e-4))
    sol = NumericalSolver.integrate_hybrid_ode(
        rate, wrt="t", span=(0.0, 120.0), y0=0.0, target_name="delta",
        events=[
            SheddingEventSpec("cohesive_fracture", lambda tv, dv: 2e7 * dv ** 3 - 1.28, shed_fraction=0.42),
            SheddingEventSpec("interfacial_adhesive", lambda tv, dv: 2e7 * dv ** 3 - 20.0, shed_fraction=0.95),
        ],
    )
    print("Saved", plot_hybrid(sol, path=f"{OUT}/shedding_sawtooth.png",
                               x_label="t (s)", y_label="ice thickness", y_scale=1e3, y_unit="mm"))


if __name__ == "__main__":
    demo_waterfall_power()
    demo_blasius_ranked()
    demo_shedding_sawtooth()
