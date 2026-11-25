"""
rosolver: Symbolic & Numerical Equation Formulator from Qualitative Dependencies and Dimensions.

Generalizes the Chain-Rule + Dimensional Closure methodology of ch1.tex to arbitrary
Directed Acyclic Graphs (DAGs), Buckingham-Pi groups, Clairaut exactness, and numerical calibration.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union

from .dimensions import Dimension, DimensionalEngine, DimensionalSolution
from .numerical import (
    FitResult,
    HybridODESolution,
    NumericalSolver,
    ODESolution,
    SheddingEventSpec,
    SheddingRecord,
)
from .simulate import (
    SimulationResult,
    plot_hybrid,
    plot_ranked,
    plot_simulation,
    simulate,
    simulate_and_plot,
)
from .symbolic import CandidateEquation, DerivationResult, DerivationStep, SymbolicSolver
from .system import RateIdentity, RelationSpec, System, Variable


def formulate(
    relations: Sequence[str],
    target: str,
    wrt: Optional[str] = None,
    dims: Optional[Mapping[str, Union[str, Dimension, Mapping[str, int]]]] = None,
    constants: Optional[Iterable[str]] = None,
    rate_identities: Optional[Sequence[Tuple[str, str, str]]] = None,
    zero_initials: bool = False,
    unit_constants: bool = False,
    include_saturating: bool = False,
) -> DerivationResult:
    """
    One-call convenience API to formulate symbolic equation(s) from qualitative relations.

    Example:
        res = formulate(
            relations=["g = fn(a, b, c)", "b = fn(f)"],
            target="g",
            wrt="f",
        )
    """
    sys = System.from_spec(
        relations=relations,
        dims=dims,
        constants=constants,
        rate_identities=rate_identities,
    )
    return SymbolicSolver(sys).derive(
        target=target,
        wrt=wrt,
        zero_initials=zero_initials,
        unit_constants=unit_constants,
        include_saturating=include_saturating,
    )


__all__ = [
    "CandidateEquation",
    "DerivationResult",
    "DerivationStep",
    "Dimension",
    "DimensionalEngine",
    "DimensionalSolution",
    "FitResult",
    "HybridODESolution",
    "NumericalSolver",
    "ODESolution",
    "RateIdentity",
    "RelationSpec",
    "SheddingEventSpec",
    "SheddingRecord",
    "SimulationResult",
    "SymbolicSolver",
    "System",
    "Variable",
    "formulate",
    "plot_hybrid",
    "plot_ranked",
    "plot_simulation",
    "simulate",
    "simulate_and_plot",
]
