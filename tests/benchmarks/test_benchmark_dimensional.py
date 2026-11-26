"""Benchmark 4: Independent analytical dimensional analysis & Buckingham-Pi checks (ENG_SPEC.md P0.2)."""

from __future__ import annotations

import sympy as sp

from rosolver import Dimension, DimensionalEngine


class TestBenchmarkDimensionalAnalysis:
    def test_pendulum_period_unique_closure(self) -> None:
        # T = fn(L, g) -> T ~ L^(1/2) * g^(-1/2)
        sol = DimensionalEngine.solve(
            target=Dimension.parse("T"),
            candidates=[("L_pend", Dimension.parse("L")), ("grav", Dimension.parse("L/T^2"))],
        )
        assert sol.consistent
        assert sol.is_unique
        assert sol.particular_exponents == [sp.Rational(1, 2), sp.Rational(-1, 2)]

    def test_pipe_flow_pressure_gradient_two_independent_pi_groups(self) -> None:
        # dp/dx = fn(rho, V, D, mu, eps) -> 5 vars, 3 base dims (M, L, T) => 2 independent Pi groups
        cands = [
            ("rho", Dimension.parse("M/L^3")),
            ("V", Dimension.parse("L/T")),
            ("D", Dimension.parse("L")),
            ("mu", Dimension.parse("M/(L*T)")),
            ("eps", Dimension.parse("L")),
        ]
        sol = DimensionalEngine.solve(target=Dimension.parse("M/(L^2*T^2)"), candidates=cands)
        assert sol.consistent
        assert sol.nullity == 2

        # Check no redundant Pi groups (rank == 2) and each Pi group is strictly dimensionless
        pi_matrix = sp.Matrix(sol.pi_groups)
        assert pi_matrix.rank() == 2
        for vec in sol.pi_groups:
            dim_check = Dimension.dimensionless()
            for (_, d), exp in zip(cands, vec):
                dim_check = dim_check * (d ** exp)
            assert dim_check.is_dimensionless

        # Verify particular solution matches [dp/dx] = [rho * V^2 / D] or dimensionally equivalent
        part_dim = Dimension.dimensionless()
        for (_, d), exp in zip(cands, sol.particular_exponents):
            part_dim = part_dim * (d ** exp)
        assert part_dim == Dimension.parse("M/(L^2*T^2)")

    def test_convective_heat_transfer_reynolds_and_prandtl_rank(self) -> None:
        # h = fn(k, D, rho, V, mu, cp) -> 6 vars, 4 base dims (M, L, T, Theta) => 2 Pi groups
        cands = [
            ("k", Dimension.parse("M*L/(T^3*Theta)")),
            ("D", Dimension.parse("L")),
            ("rho", Dimension.parse("M/L^3")),
            ("V", Dimension.parse("L/T")),
            ("mu", Dimension.parse("M/(L*T)")),
            ("cp", Dimension.parse("L^2/(T^2*Theta)")),
        ]
        sol = DimensionalEngine.solve(target=Dimension.parse("M/(T^3*Theta)"), candidates=cands)
        assert sol.consistent
        assert sol.nullity == 2
        assert sp.Matrix(sol.pi_groups).rank() == 2

    def test_inconsistent_equation_rejection(self) -> None:
        sol = DimensionalEngine.solve(
            target=Dimension.parse("M*L^2/T^2"),
            candidates=[("L_ref", Dimension.parse("L")), ("V_ref", Dimension.parse("L/T"))],
        )
        assert not sol.consistent
        assert "M" in sol.missing_dimensions
