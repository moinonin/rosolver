"""
rosolver: Symbolic & Numerical Equation Formulator from Qualitative Dependencies and Dimensions.

Generalizes the Chain-Rule + Dimensional Closure methodology of ch1.tex to arbitrary
Directed Acyclic Graphs (DAGs), Buckingham-Pi groups, Clairaut exactness, and numerical calibration.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union

from .dimensions import Dimension, DimensionalEngine, DimensionalSolution
from .integrators import FixedStepResult, integrate_fixed, max_stable_dt, stability_function
from .numerical import (
    FitResult,
    HybridODESolution,
    NumericalSolver,
    ODESolution,
    SheddingEventSpec,
    SheddingRecord,
)
from .odesys import ODESystem, Trajectory
from .pde import BC, ConservationLaw1D, PDESolution, cell_average
from .phasechange import (
    HybridStefanSolution,
    Stefan1D,
    StefanSolution,
    neumann_lambda,
    stefan_shedding_event,
)
from .simulate import (
    SimulationResult,
    plot_conservation_audit,
    plot_gci_study,
    plot_hybrid,
    plot_mms_convergence,
    plot_ranked,
    plot_simulation,
    plot_stability_spectrum,
    plot_stefan,
    simulate,
    simulate_and_plot,
)
from .symbolic import CandidateEquation, DerivationResult, DerivationStep, SymbolicSolver
from .system import RateIdentity, RelationSpec, System, Variable
from .verify import (
    CoSimulationReport,
    ConservationAuditReport,
    FieldConservationEntry,
    GCIReport,
    MMSReport,
    PartitionedCouplingReport,
    StiffnessReport,
    TimestepReport,
    analyze_stiffness,
    audit_hybrid_conservation,
    audit_ode_invariants,
    audit_pde_conservation,
    cosimulate_coupled_odes,
    gci_from_values,
    grid_convergence_study,
    mms_study,
    solve_partitioned_fixed_point,
    timestep_study,
)


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
    "cell_average",
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
]
