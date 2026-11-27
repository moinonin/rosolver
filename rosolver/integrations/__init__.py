"""Modular external solver adapters and verification facade (ENG_SPEC.md P2.3 / P2.3+)."""

from .adapters import GenericCSVAdapter, SciPyAdapter, SolverAdapter, VerificationStudy
from .fenicsx import FEniCSxAdapter
from .hdf5 import HDF5Adapter
from .openfoam import OpenFOAMAdapter

__all__ = [
    "FEniCSxAdapter",
    "GenericCSVAdapter",
    "HDF5Adapter",
    "OpenFOAMAdapter",
    "SciPyAdapter",
    "SolverAdapter",
    "VerificationStudy",
]
