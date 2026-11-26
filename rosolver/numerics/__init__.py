"""Modular numerical solvers: 1D FVM PDE, ODE systems, Stefan phase-change, and fixed-step integrators."""

from ..integrators import FixedStepResult, integrate_fixed, max_stable_dt, stability_function
from ..numerical import (
    HybridODESolution,
    NumericalSolver,
    ODESolution,
    SheddingEventSpec,
    SheddingRecord,
)
from ..odesys import EventSpec, ODESystem, Trajectory
from ..pde import BC, ConservationLaw1D, PDESolution, cell_average
from ..phasechange import (
    HybridStefanSolution,
    Stefan1D,
    StefanSolution,
    neumann_lambda,
    stefan_shedding_event,
)

__all__ = [
    "BC",
    "ConservationLaw1D",
    "EventSpec",
    "FixedStepResult",
    "HybridODESolution",
    "HybridStefanSolution",
    "NumericalSolver",
    "ODESolution",
    "ODESystem",
    "PDESolution",
    "SheddingEventSpec",
    "SheddingRecord",
    "Stefan1D",
    "StefanSolution",
    "Trajectory",
    "cell_average",
    "integrate_fixed",
    "max_stable_dt",
    "neumann_lambda",
    "stability_function",
    "stefan_shedding_event",
]
