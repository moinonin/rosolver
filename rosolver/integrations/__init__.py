"""Modular external solver adapters and verification facade (ENG_SPEC.md P2.3)."""

from .adapters import GenericCSVAdapter, SciPyAdapter, SolverAdapter, VerificationStudy

__all__ = [
    "GenericCSVAdapter",
    "SciPyAdapter",
    "SolverAdapter",
    "VerificationStudy",
]
