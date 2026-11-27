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
from .importer import (
    CandidateImporter,
    CandidateScreeningReport,
    ImportedCandidate,
    ScreenedCandidate,
)

__all__ = [
    "CandidateEquation",
    "CandidateImporter",
    "CandidateScreeningReport",
    "DerivationResult",
    "DerivationStep",
    "DimensionalEngine",
    "DimensionalSolution",
    "EquationTemplate",
    "ImportedCandidate",
    "ScreenedCandidate",
    "SymbolicSolver",
    "TEMPLATE_REGISTRY",
    "TemplateRegistry",
    "get_template",
    "list_templates",
    "register_template",
]
