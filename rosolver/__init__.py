"""
rosolver: Symbolic & Numerical Equation Formulator from Qualitative Dependencies and Dimensions.

Generalizes the Chain-Rule + Dimensional Closure methodology of ch1.tex to arbitrary
Directed Acyclic Graphs (DAGs), Buckingham-Pi groups, Clairaut exactness, and numerical calibration.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union

from .dimensions import Dimension, DimensionalEngine, DimensionalSolution
from .integrators import FixedStepResult, integrate_fixed, max_stable_dt, stability_function
from .model import (
    ConditionSpec,
    EquationSpec,
    Model,
    ModelValidationError,
    NumericsConfig,
    ParameterSpec,
    VariableSpec,
)
from .numerical import (
    CandidateRankingReport,
    FitResult,
    HybridODESolution,
    IdentifiabilityReport,
    NumericalSolver,
    ODESolution,
    RejectedCandidate,
    SheddingEventSpec,
    SheddingRecord,
)
from .odesys import EventSpec, ODESystem, Trajectory
from .pde import BC, ConservationLaw1D, PDESolution, cell_average
from .phasechange import (
    HybridStefanSolution,
    Stefan1D,
    StefanSolution,
    neumann_lambda,
    stefan_shedding_event,
)
from .project import (
    DatasetRef,
    ExecutionRecord,
    ModelRef,
    Project,
    ProjectError,
    ProvenanceMismatchError,
    capture_environment,
    sha256_file,
)
from .results import ResultMetadata, ResultMixin, StandardResult, standardize_result
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
from .templates import (
    TEMPLATE_REGISTRY,
    EquationTemplate,
    TemplateRegistry,
    get_template,
    list_templates,
    register_template,
)
from .verify import (
    CoSimulationReport,
    ConservationAuditReport,
    FieldConservationEntry,
    GCIReport,
    MMSReport,
    PartitionedCouplingReport,
    StiffnessReport,
    TimestepReport,
    VerificationStatus,
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
    templates: Optional[Sequence[str]] = None,
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
        templates=templates,
    )


__all__ = [
    "BC",
    "CandidateEquation",
    "CandidateRankingReport",
    "CoSimulationReport",
    "ConditionSpec",
    "ConservationAuditReport",
    "ConservationLaw1D",
    "DatasetRef",
    "DerivationResult",
    "DerivationStep",
    "Dimension",
    "DimensionalEngine",
    "DimensionalSolution",
    "EquationSpec",
    "EquationTemplate",
    "EventSpec",
    "ExecutionRecord",
    "FieldConservationEntry",
    "FitResult",
    "FixedStepResult",
    "GCIReport",
    "HybridODESolution",
    "HybridStefanSolution",
    "IdentifiabilityReport",
    "MMSReport",
    "Model",
    "ModelRef",
    "ModelValidationError",
    "NumericalSolver",
    "NumericsConfig",
    "ODESolution",
    "ODESystem",
    "PDESolution",
    "ParameterSpec",
    "PartitionedCouplingReport",
    "Project",
    "ProjectError",
    "ProvenanceMismatchError",
    "RateIdentity",
    "RejectedCandidate",
    "RelationSpec",
    "ResultMetadata",
    "ResultMixin",
    "SheddingEventSpec",
    "SheddingRecord",
    "SimulationResult",
    "StandardResult",
    "StiffnessReport",
    "Stefan1D",
    "StefanSolution",
    "SymbolicSolver",
    "System",
    "TEMPLATE_REGISTRY",
    "TemplateRegistry",
    "TimestepReport",
    "Trajectory",
    "Variable",
    "VariableSpec",
    "VerificationStatus",
    "analyze_stiffness",
    "audit_hybrid_conservation",
    "audit_ode_invariants",
    "audit_pde_conservation",
    "capture_environment",
    "cell_average",
    "cosimulate_coupled_odes",
    "formulate",
    "gci_from_values",
    "get_template",
    "grid_convergence_study",
    "integrate_fixed",
    "list_templates",
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
    "register_template",
    "sha256_file",
    "simulate",
    "simulate_and_plot",
    "solve_partitioned_fixed_point",
    "stability_function",
    "standardize_result",
    "stefan_shedding_event",
    "timestep_study",
]
