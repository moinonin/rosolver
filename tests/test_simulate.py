"""Tests for automatic simulation and plotting."""

from __future__ import annotations

import numpy as np
import pytest
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
from rosolver.cli import main as cli_main


def _waterfall():
    return formulate(
        relations=["g = fn(a, b, c)", "b = fn(f; grav)"],
        dims={"g": "M*L^2/T^3", "a": "M/L^3", "b": "L/T", "c": "L^2", "f": "L", "grav": "L/T^2"},
        constants=["grav"],
        target="g",
        wrt="f",
        zero_initials=True,
        unit_constants=True,
    )


def test_simulate_matches_closed_form_and_family() -> None:
    sim = simulate(
        _waterfall(),
        sweep={"f": (1.0, 9.0, 50)},
        values={"c": 2.0, "grav": 9.81},
        family={"a": [1.0, 2.0]},
    )
    assert set(sim.curves) == {"a=1", "a=2"}
    # unit constants => g = 8 a c (grav f)^(3/2)
    expected = 8.0 * 1.0 * 2.0 * (9.81 * sim.x) ** 1.5
    assert np.allclose(sim.curves["a=1"], expected)
    assert np.allclose(sim.curves["a=2"], 2 * expected)


def test_missing_values_and_bad_sweep_raise() -> None:
    with pytest.raises(KeyError, match="Provide values"):
        simulate(_waterfall(), sweep={"f": (1, 2, 5)}, values={"c": 1.0})
    with pytest.raises(KeyError, match="does not appear"):
        simulate(_waterfall(), sweep={"zzz": (1, 2, 5)}, values={})
    with pytest.raises(ValueError, match="exactly one"):
        simulate(_waterfall(), sweep={"f": (1, 2, 5), "a": (1, 2, 5)})


def test_assumed_defaults_reported() -> None:
    res = formulate(["K = fn(m, v)"], dims={"K": "M*L^2/T^2", "m": "M", "v": "L/T"},
                    target="K", zero_initials=True)
    sim = simulate(res, sweep={"v": (0, 10, 11)}, values={"m": 2.0})
    assert sim.assumed == {"kappa_K": 1.0}
    assert np.allclose(sim.curves["K"], 2.0 * sim.x ** 2)


def test_plots_written(tmp_path) -> None:
    sim = simulate(_waterfall(), sweep={"f": (1, 9, 30)}, values={"a": 1.0, "c": 1.0, "grav": 9.81})
    p = plot_simulation(sim, path=tmp_path / "sim.png", data=(sim.x, sim.curves["g"]))
    assert p.exists() and p.stat().st_size > 1000
    assert sim.to_csv(tmp_path / "sim.csv").exists()

    res = formulate(["K = fn(m, v)"], dims={"K": "M*L^2/T^2", "m": "M", "v": "L/T"},
                    target="K", zero_initials=True)
    rng = np.random.default_rng(0)
    m, v = rng.uniform(1, 5, 20), rng.uniform(1, 9, 20)
    data = {"K": 0.5 * m * v ** 2, "m": m, "v": v}
    ranked = NumericalSolver.rank_candidates(res, data)
    assert plot_ranked(ranked, data, "v", "K", path=tmp_path / "rank.png").exists()

    t, d = sp.symbols("t delta", real=True)
    sol = NumericalSolver.integrate_hybrid_ode(
        sp.Eq(sp.Derivative(d, t), sp.Float(1e-4)), wrt="t", span=(0, 120), y0=0.0, target_name="delta",
        events=[SheddingEventSpec("cohesive", lambda tv, dv: 2e7 * dv ** 3 - 1.28, shed_fraction=0.42)],
    )
    assert plot_hybrid(sol, path=tmp_path / "saw.png", y_scale=1e3, y_unit="mm").exists()


def test_cli_sweep_and_plot(tmp_path, capsys) -> None:
    out = tmp_path / "cli.png"
    rc = cli_main([
        "--rel", "K = fn(m, v)",
        "--dim", "K=M*L^2/T^2", "--dim", "m=M", "--dim", "v=L/T",
        "--target", "K", "--zero-initials",
        "--sweep", "v=0:10:21", "--set", "m=2", "--set", "kappa_K=0.5",
        "--plot", str(out), "--sim-csv", str(tmp_path / "cli.csv"),
    ])
    assert rc == 0 and out.exists()
    assert "Simulation: K(v)" in capsys.readouterr().out
