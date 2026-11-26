"""Modular scientific reporting and plotting subsystem (ENG_SPEC.md P3.1)."""

from ..simulate import (
    SimulationResult,
    plot_conservation_audit,
    plot_gci_study,
    plot_hybrid,
    plot_mms_convergence,
    plot_ranked,
    plot_simulation,
    plot_stability_spectrum,
    plot_stefan,
    plot_validation,
)
from .markdown import REPORT_TYPES, ReportGenerator, ScientificReport

__all__ = [
    "REPORT_TYPES",
    "ReportGenerator",
    "ScientificReport",
    "SimulationResult",
    "plot_conservation_audit",
    "plot_gci_study",
    "plot_hybrid",
    "plot_mms_convergence",
    "plot_ranked",
    "plot_simulation",
    "plot_stability_spectrum",
    "plot_stefan",
    "plot_validation",
]
