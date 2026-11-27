"""Deterministic Scientific Claim Verification, Scale-Separation & Peer-Review Engine (Sprint 10 / P5.4)."""

from .budget import ErrorBudgetReport, audit_claimed_accuracy
from .clarifications import (
    ArtifactRequirementSpec,
    ClarificationRequest,
    generate_clarification_requests,
)
from .engine import ClaimAuditEngine
from .rebuttal import (
    RebuttalSubmission,
    ReviewRoundRecord,
    VerdictTransition,
    resolve_confined_path,
    validate_rebuttal_dataset_csv,
)
from .scales import (
    HomogeneityAuditReport,
    MissingConstantHypothesis,
    NormalizationDiagnosticReport,
    ScaleBalanceReport,
    TermMagnitudeEntry,
    check_equation_scales,
    check_homogeneity_elasticities,
    diagnose_normalization_shortcuts,
)
from .schema import (
    ClaimKind,
    ClaimManifest,
    ClaimSpec,
    ClaimVerdict,
    ClaimVerdictKind,
    TwoAxisAssessment,
    synthesize_verdict,
)

__all__ = [
    "ArtifactRequirementSpec",
    "ClaimAuditEngine",
    "ClaimKind",
    "ClaimManifest",
    "ClaimSpec",
    "ClaimVerdict",
    "ClaimVerdictKind",
    "ClarificationRequest",
    "ErrorBudgetReport",
    "HomogeneityAuditReport",
    "MissingConstantHypothesis",
    "NormalizationDiagnosticReport",
    "RebuttalSubmission",
    "ReviewRoundRecord",
    "ScaleBalanceReport",
    "TermMagnitudeEntry",
    "TwoAxisAssessment",
    "VerdictTransition",
    "audit_claimed_accuracy",
    "check_equation_scales",
    "check_homogeneity_elasticities",
    "diagnose_normalization_shortcuts",
    "generate_clarification_requests",
    "resolve_confined_path",
    "synthesize_verdict",
    "validate_rebuttal_dataset_csv",
]
