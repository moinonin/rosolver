"""Security tests for safe AST expression parsing and regression tests for public API exports."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

import rosolver
from rosolver._parsing import names, parse_expr, real_symbol


EXPECTED_V01_EXPORTS = {
    "BC",
    "CandidateEquation",
    "CoSimulationReport",
    "ConservationAuditReport",
    "ConservationLaw1D",
    "DerivationResult",
    "DerivationStep",
    "Dimension",
    "DimensionalEngine",
    "DimensionalSolution",
    "FieldConservationEntry",
    "FitResult",
    "FixedStepResult",
    "GCIReport",
    "HybridODESolution",
    "HybridStefanSolution",
    "MMSReport",
    "NumericalSolver",
    "ODESolution",
    "ODESystem",
    "PDESolution",
    "PartitionedCouplingReport",
    "RateIdentity",
    "RelationSpec",
    "SheddingEventSpec",
    "SheddingRecord",
    "SimulationResult",
    "StiffnessReport",
    "Stefan1D",
    "StefanSolution",
    "SymbolicSolver",
    "System",
    "TimestepReport",
    "Trajectory",
    "Variable",
    "analyze_stiffness",
    "audit_hybrid_conservation",
    "audit_ode_invariants",
    "audit_pde_conservation",
    "cosimulate_coupled_odes",
    "formulate",
    "gci_from_values",
    "grid_convergence_study",
    "integrate_fixed",
    "max_stable_dt",
    "mms_study",
    "neumann_lambda",
    "plot_conservation_audit",
    "plot_gci_study",
    "plot_hybrid",
    "plot_mms_convergence",
    "plot_ranked",
    "plot_simulation",
    "plot_stability_spectrum",
    "plot_stefan",
    "simulate",
    "simulate_and_plot",
    "solve_partitioned_fixed_point",
    "stability_function",
    "stefan_shedding_event",
    "timestep_study",
    "cell_average",
}


class TestSafeExpressionParser:
    def test_valid_math_and_sympy_singletons_bound_as_real_symbols(self) -> None:
        expr = parse_expr("-beta * S * I / N + E + O + Q + gamma")
        assert names(expr) == {"beta", "S", "I", "N", "E", "O", "Q", "gamma"}
        for sym in expr.free_symbols:
            assert sym.is_real is True

    def test_valid_functions_and_constants(self) -> None:
        expr = parse_expr("1.0 + 0.3 * sin(2 * pi * (x - 0.5 * t)) + exp(-40 * (x - 0.35)**2)")
        assert names(expr) == {"x", "t"}
        assert expr.has(sp.pi)

        other = parse_expr("Abs(x) + ln(y) + atan2(y, x) + Min(a, b) + Max(a, b) + sign(-z)")
        assert names(other) == {"x", "y", "a", "b", "z"}

    def test_numeric_and_sympy_inputs(self) -> None:
        assert parse_expr(3) == sp.Integer(3)
        assert parse_expr(np.int64(4)) == sp.Integer(4)
        assert parse_expr(np.float64(2.5)) == sp.Float(2.5)
        x_sym = real_symbol("x")
        bound = parse_expr(sp.Symbol("x") + 1, symbols={"x": x_sym})
        assert x_sym in bound.free_symbols

    @pytest.mark.parametrize(
        "payload",
        [
            "",
            "   ",
            "__import__('os').system('echo pwn')",
            "(lambda x: x)(1)",
            "[x for x in range(5)]",
            "().__class__.__base__",
            "sin.__globals__",
            "open('pyproject.toml').read()",
            "eval('1 + 1')",
            "a[0]",
            "True",
            "False",
            "None",
            "'hello'",
            "x ^ 2",
            "x := 1",
            "1 < 2",
            "x and y",
            "x if y else z",
            "sin",
            "sin(x, evaluate=False)",
            "sin(*args)",
            "__secret__",
            "1 // 2",
            "5 % 2",
        ],
    )
    def test_disallowed_or_malicious_expressions_rejected(self, payload: str) -> None:
        with pytest.raises(ValueError):
            parse_expr(payload)

    def test_booleans_and_bad_types_rejected(self) -> None:
        with pytest.raises(TypeError):
            parse_expr(True)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            parse_expr([1, 2])  # type: ignore[arg-type]

    def test_complexity_and_length_limits_enforced(self) -> None:
        too_many_nodes = " + ".join(["x"] * 600)
        with pytest.raises(ValueError, match="complexity"):
            parse_expr(too_many_nodes)

        too_long = "(" + " " * 10_005 + "x)"
        with pytest.raises(ValueError, match="maximum length"):
            parse_expr(too_long)


class TestPublicApiFreeze:
    def test_v01_exports_preserved_and_resolvable(self) -> None:
        exported = set(rosolver.__all__)
        missing = EXPECTED_V01_EXPORTS - exported
        assert not missing, f"Missing backwards-compatible exports: {sorted(missing)}"
        for sym in exported:
            assert hasattr(rosolver, sym), f"Export '{sym}' in __all__ is not defined on rosolver"
