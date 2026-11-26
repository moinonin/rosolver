"""Modular symbolic discovery, dimensional closure, and template subsystem."""

from ..dimensions import DimensionalEngine, DimensionalSolution
from ..templates import (
    TEMPLATE_REGISTRY,
    EquationTemplate,
    TemplateRegistry,
    get_template,
    list_templates,
    register_template,
)
from .derivation import CandidateEquation, DerivationResult, DerivationStep, SymbolicSolver

__all__ = [
    "CandidateEquation",
    "DerivationResult",
    "DerivationStep",
    "DimensionalEngine",
    "DimensionalSolution",
    "EquationTemplate",
    "SymbolicSolver",
    "TEMPLATE_REGISTRY",
    "TemplateRegistry",
    "get_template",
    "list_templates",
    "register_template",
]
