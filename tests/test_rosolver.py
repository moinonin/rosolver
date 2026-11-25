"""Comprehensive unit and integration tests for rosolver (Symbolic + Numerical Solver)."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from rosolver import (
    Dimension,
    DimensionalEngine,
    NumericalSolver,
    SymbolicSolver,
    System,
    formulate,
)
from rosolver.cli import main as cli_main


class TestDimensionsAndBuckinghamPi:
    def test_dimension_parsing_and_algebra(self) -> None:
        d1 = Dimension.parse("M*L^2/T^3")
        d2 = Dimension.parse("kg*m^2/s^3")
        assert d1 == d2
        assert not d1.is_dimensionless

        d_half = Dimension.parse("L^(1/2)/T")
        assert (d_half ** 2) == Dimension.parse("L/T^2")
        assert (d1 / d2).is_dimensionless

    def test_custom_non_si_dimensions(self) -> None:
        rev = Dimension.parse("USD/Month")
        users = Dimension.parse("Users")
        arpu = rev / users
        assert arpu == Dimension.parse("USD/(Users*Month)")

    def test_unique_monomial_solver(self) -> None:
        target = Dimension.parse("M*L/T^2")  # Force
        cands = [
            ("rho", Dimension.parse("M/L^3")),
            ("v", Dimension.parse("L/T")),
            ("A", Dimension.parse("L^2")),
        ]
        sol = DimensionalEngine.solve(target, cands)
        assert sol.consistent
        assert sol.is_unique
        assert sol.particular_exponents == [sp.Rational(1), sp.Rational(2), sp.Rational(1)]

    def test_buckingham_pi_reynolds_number(self) -> None:
        # Drag force F_D from (rho, v, L_char, mu) -> particular rho*v^2*L^2, Pi = Re
        target = Dimension.parse("M*L/T^2")
        cands = [
            ("rho", Dimension.parse("M/L^3")),
            ("v", Dimension.parse("L/T")),
            ("x", Dimension.parse("L")),
            ("mu", Dimension.parse("M/(L*T)")),
        ]
        sol = DimensionalEngine.solve(target, cands)
        assert sol.consistent
        assert sol.nullity == 1
        # Verify Pi-group is dimensionless!
        pi_vec = sol.pi_groups[0]
        check_dim = Dimension.dimensionless()
        for (_, d), exp in zip(cands, pi_vec):
            check_dim = check_dim * (d ** exp)
        assert check_dim.is_dimensionless
        # Re = rho * v * x / mu -> exponents [1, 1, 1, -1]
        assert pi_vec == [sp.Rational(1), sp.Rational(1), sp.Rational(1), sp.Rational(-1)]

    def test_inconsistent_dimensions_diagnostic(self) -> None:
        target = Dimension.parse("M*L/T^2")
        cands = [("x", Dimension.parse("L")), ("t", Dimension.parse("T"))]
        sol = DimensionalEngine.solve(target, cands)
        assert not sol.consistent
        assert "M" in sol.missing_dimensions
        assert "Unbalanced base dimension" in sol.diagnostic


class TestSymbolicFormulator:
    def test_user_nested_dag_dimension_free(self) -> None:
        """Test g = fn(a, b, c), b = fn(f) with zero prior equations or dimensions."""
        res = formulate(
            relations=["g = fn(a, b, c)", "b = fn(f)"],
            target="g",
            wrt="f",
        )
        syms = {s.name: s for s in res.integrated_equation.free_symbols}
        # Active chain rule should prune da/df and dc/df, leaving (dg/db)*(db/df)
        assert "a" not in str(res.chain_rule_active.rhs.args[0]) or "Derivative(b(f), f)" in str(
            res.chain_rule_active.rhs
        )
        assert "Derivative(a(f), f)" in str(res.chain_rule_general.rhs)
        assert "Derivative(a(f), f)" not in str(res.chain_rule_active.rhs)
        assert "Derivative(c(f), f)" not in str(res.chain_rule_active.rhs)
        # Expresses g in terms of f while preserving held factors a and c!
        assert {"a", "c", "f"}.issubset(set(syms.keys()))
        assert "b" not in syms  # b has been eliminated in favor of f!
        assert len(res.candidates) == 2  # affine_regression + power_law_scaling

    def test_user_nested_dag_dimensional(self) -> None:
        """Test g = fn(a, b, c), b = fn(f; grav) with full dimensional closure."""
        res = formulate(
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
            unit_constants=True,
        )
        # b(f) = int (grav^(1/2) * f^(-1/2)) df = 2 * sqrt(grav * f)
        # g(a,b,c) = a * b^3 * c => g(f) = 8 * a * c * (grav * f)^(3/2)
        syms = res.integrated_equation.free_symbols
        names = {s.name for s in syms}
        assert names == {"g", "a", "c", "f", "grav"}
        # Check algebraic scaling: doubling f multiplies g by 2^(3/2)
        val1 = NumericalSolver.evaluate(res, {"a": 1.2, "c": 3.0, "f": 4.0, "grav": 9.81})
        val2 = NumericalSolver.evaluate(res, {"a": 1.2, "c": 3.0, "f": 8.0, "grav": 9.81})
        assert np.isclose(val2 / val1, 2.0 ** 1.5)

    def test_ch1_sec1_kinematics_reproduction(self) -> None:
        """Reproduce ch1.tex Eq. 2.5 and Eq. 2.6: v(t) = v_0 + a*t, s(t) = s_0 + v_0*t + 1/2*a*t^2."""
        res = formulate(
            relations=["s = fn(v)", "v = fn(t)"],
            dims={"s": "L", "v": "L/T", "t": "T", "a": "L/T^2"},
            constants=["a"],
            rate_identities=[("s", "t", "v"), ("v", "t", "a")],
            target="s",
            wrt="t",
            unit_constants=True,
        )
        s0, v0, a, t = (
            sp.Symbol("s_0", real=True),
            sp.Symbol("v_0", real=True),
            sp.Symbol("a", positive=True, real=True),
            sp.Symbol("t", positive=True, real=True),
        )
        expected_v = v0 + a * t
        expected_s = s0 + v0 * t + sp.Rational(1, 2) * a * (t ** 2)
        assert sp.simplify(res.node_potentials["v"] - expected_v) == 0
        assert sp.simplify(res.integrated_equation.rhs - expected_s) == 0

    def test_clairaut_exactness_kinetic_energy(self) -> None:
        """Verify Clairaut coupling automatically links dK/dv = 2 * m * v to dK/dm = v^2."""
        res = formulate(
            relations=["K = fn(m, v)"],
            dims={"K": "M*L^2/T^2", "m": "M", "v": "L/T"},
            target="K",
            zero_initials=True,
        )
        # Only ONE free constant kappa_K should remain (not separate kappa_m, kappa_v)!
        assert len(res.primary.free_params) == 1
        k_sym = res.primary.free_params[0]
        m_sym = sp.Symbol("m", positive=True, real=True)
        v_sym = sp.Symbol("v", positive=True, real=True)
        assert sp.simplify(res.primary.rhs - k_sym * m_sym * (v_sym ** 2)) == 0

    def test_diamond_multipath_dag(self) -> None:
        """Test diamond graph p = fn(m, v) where BOTH m = fn(t; mdot) and v = fn(t; a) depend on t."""
        res = formulate(
            relations=["p = fn(m, v)", "m = fn(t; mdot)", "v = fn(t; a)"],
            dims={"p": "M*L/T", "m": "M", "v": "L/T", "t": "T", "mdot": "M/T", "a": "L/T^2"},
            constants=["mdot", "a"],
            target="p",
            wrt="t",
            unit_constants=True,
        )
        m0 = sp.Symbol("m_0", real=True)
        v0 = sp.Symbol("v_0", real=True)
        p0 = sp.Symbol("p_0", real=True)
        mdot = sp.Symbol("mdot", positive=True, real=True)
        a = sp.Symbol("a", positive=True, real=True)
        t = sp.Symbol("t", positive=True, real=True)
        expected = p0 + (m0 + mdot * t) * (v0 + a * t)
        assert sp.simplify(sp.expand(res.primary.rhs - expected)) == 0

    def test_cycle_detection_raises(self) -> None:
        with pytest.raises(ValueError, match="Cycle detected"):
            System.from_spec(["a = fn(b)", "b = fn(c)", "c = fn(a)"])


class TestNumericalSolver:
    def test_exact_ols_calibration_kinetic_energy(self) -> None:
        res = formulate(
            relations=["K = fn(m, v)"],
            dims={"K": "M*L^2/T^2", "m": "M", "v": "L/T"},
            target="K",
            zero_initials=True,
        )
        rng = np.random.default_rng(123)
        m = rng.uniform(1.0, 10.0, 50)
        v = rng.uniform(2.0, 20.0, 50)
        K = 0.5 * m * (v ** 2)

        fit = NumericalSolver.fit(res, {"K": K, "m": m, "v": v})
        assert fit.converged
        assert fit.r2 > 0.999999999
        assert np.isclose(fit.params["kappa_K"], 0.5, atol=1e-10)
        assert str(fit.calibrated_equation.rhs) == "m*v**2/2"

    def test_nonlinear_pi_exponent_discovery_blasius(self) -> None:
        """Discover Blasius exponent nu_1 = -0.5 and coefficient kappa = 5.0 from synthetic data."""
        res = formulate(
            relations=["delta = fn(x; rho, v, mu)"],
            dims={"delta": "L", "x": "L", "rho": "M/L^3", "v": "L/T", "mu": "M/(L*T)"},
            constants=["rho", "v", "mu"],
            target="delta",
            wrt="x",
            zero_initials=True,
        )
        rng = np.random.default_rng(2026)
        n = 60
        x = rng.uniform(0.2, 3.0, n)
        rho = rng.uniform(0.9, 1.3, n)
        v = rng.uniform(3.0, 20.0, n)
        mu = rng.uniform(1.5e-5, 2.5e-5, n)
        re = rho * v * x / mu
        delta = 5.0 * x * (re ** -0.5)

        ranked = NumericalSolver.rank_candidates(
            res, {"delta": delta, "x": x, "rho": rho, "v": v, "mu": mu}, criterion="bic"
        )
        assert "Buckingham-Pi Generalization" in res.summary()
        assert "\\begin{equation}" in res.to_latex()
        best = ranked[0]
        assert best.candidate_name == "buckingham_pi_power"
        assert best.r2 > 0.99999999
        assert np.isclose(best.params["kappa_delta"], 5.0, atol=1e-6)
        assert np.isclose(best.params["nu_1"], -0.5, atol=1e-6)

    def test_differential_regression_fit(self) -> None:
        res = formulate(
            relations=["s = fn(v)", "v = fn(t)"],
            dims={"s": "L", "v": "L/T", "t": "T", "a": "L/T^2"},
            constants=["a"],
            rate_identities=[("s", "t", "v"), ("v", "t", "a")],
            target="s",
            wrt="t",
        )
        t = np.linspace(0.0, 5.0, 101)
        v0_true = 3.5
        a_val = 2.0
        s = 10.0 + v0_true * t + 0.5 * a_val * (t ** 2)

        diff_fit = NumericalSolver.fit_differential(
            res, {"s": s, "t": t, "a": np.full_like(t, a_val)}
        )
        assert diff_fit.r2 > 0.9999
        assert np.isclose(diff_fit.params["v_0"], v0_true, atol=1e-2)
        assert np.isclose(diff_fit.params["kappa_v"], 1.0, atol=1e-2)

    def test_numerical_ode_integration(self) -> None:
        # Integrate nonlinear damped rate dg/dt = a - k * g^2 from g(0)=0 -> terminal sqrt(a/k)
        t_sym, g_sym, a_sym, k_sym = sp.symbols("t g a k", real=True)
        rate_eq = sp.Eq(sp.Derivative(g_sym, t_sym), a_sym - k_sym * (g_sym ** 2))

        sol = NumericalSolver.integrate_ode(
            rate_eq,
            wrt="t",
            span=(0.0, 10.0),
            y0=0.0,
            params={"a": 9.0, "k": 1.0},
            target_name="g",
        )
        assert sol.success
        # Exact solution: g(t) = 3 * tanh(3 * t) -> approaches 3.0
        exact = 3.0 * np.tanh(3.0 * sol.wrt_values)
        assert np.allclose(sol.target_values, exact, atol=1e-5)

    def test_elasticities_and_cli(self, capsys: pytest.CaptureFixture[str]) -> None:
        res = formulate(
            relations=["P = fn(rho, A, v)"],
            dims={"P": "M*L^2/T^3", "rho": "M/L^3", "A": "L^2", "v": "L/T"},
            target="P",
            zero_initials=True,
            unit_constants=True,
        )
        elast = NumericalSolver.elasticities(res, {"rho": 1.2, "A": 10.0, "v": 8.0})
        assert np.isclose(elast["rho"], 1.0)
        assert np.isclose(elast["A"], 1.0)
        assert np.isclose(elast["v"], 3.0)

        # Verify CLI end-to-end
        rc = cli_main(
            [
                "--rel",
                "g = fn(a, b, c)",
                "--rel",
                "b = fn(f)",
                "--target",
                "g",
                "--wrt",
                "f",
                "--latex",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "\\begin{equation}" in out
