"""Modular core domain abstractions: dimensions, system DAGs, models, projects, and results."""

from ..dimensions import Dimension, DimensionalEngine, DimensionalSolution
from ..model import (
    ConditionSpec,
    EquationSpec,
    Model,
    ModelValidationError,
    NumericsConfig,
    ParameterSpec,
    VariableSpec,
)
from ..project import (
    DatasetRef,
    ExecutionRecord,
    ModelRef,
    Project,
    ProjectError,
    ProvenanceMismatchError,
    capture_environment,
    sha256_file,
)
from ..results import (
    ResultMetadata,
    ResultMixin,
    StandardResult,
    VerificationStatus,
    standardize_result,
    to_jsonable,
)
from ..system import RateIdentity, RelationSpec, System, Variable

__all__ = [
    "ConditionSpec",
    "DatasetRef",
    "Dimension",
    "DimensionalEngine",
    "DimensionalSolution",
    "EquationSpec",
    "ExecutionRecord",
    "Model",
    "ModelRef",
    "ModelValidationError",
    "NumericsConfig",
    "ParameterSpec",
    "Project",
    "ProjectError",
    "ProvenanceMismatchError",
    "RateIdentity",
    "RelationSpec",
    "ResultMetadata",
    "ResultMixin",
    "StandardResult",
    "System",
    "Variable",
    "VariableSpec",
    "VerificationStatus",
    "capture_environment",
    "sha256_file",
    "standardize_result",
    "to_jsonable",
]
