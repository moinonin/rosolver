"""Flagship 1D transient heat conduction end-to-end study (ENG_SPEC.md P3.3)."""

from typing import Any

__all__ = ["run_flagship_study"]


def __getattr__(name: str) -> Any:
    if name == "run_flagship_study":
        from .run_study import run_flagship_study

        return run_flagship_study
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
