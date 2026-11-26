"""External solver adapters and verification study facade (ENG_SPEC.md P2.3)."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import sympy as sp

from ._parsing import parse_expr, real_symbol
from .data import Dataset
from .odesys import Trajectory
from .pde import PDESolution
from .results import ResultMixin, StandardResult, VerificationStatus, standardize_result
from .verify import GCIReport, gci_from_values


class SolverAdapter(ABC):
    """
    Common adapter contract for ingesting external numerical simulation results.

    Unsupported metadata must be marked unavailable (`None`) rather than fabricated.
    Adapters convert solver outputs into `Dataset` without altering V&V algorithms.
    """

    @abstractmethod
    def load_results(self, source: Any) -> Dataset:
        """Load simulation results from `source` into a standardized `Dataset`."""

    @abstractmethod
    def extract_fields(self) -> Dict[str, np.ndarray]:
        """Return field variable arrays from the loaded results."""

    @abstractmethod
    def extract_mesh_metadata(self) -> Dict[str, Any]:
        """Return mesh/grid metadata (unsupported keys set to None)."""

    @abstractmethod
    def extract_solver_metadata(self) -> Dict[str, Any]:
        """Return solver configuration metadata (unsupported keys set to None)."""

    @abstractmethod
    def export_standard_format(self, path: Union[str, Path]) -> Path:
        """Write the loaded dataset to a portable CSV or JSON file."""


class GenericCSVAdapter(SolverAdapter):
    """Adapter for CSV, JSON, or NPZ exports from custom research codes, COMSOL, or MATLAB."""

    def __init__(
        self,
        *,
        coordinates: Union[None, str, Sequence[str]] = None,
        fields: Union[None, str, Sequence[str]] = None,
        uncertainties: Union[None, Mapping[str, str], Sequence[str]] = None,
        units: Optional[Mapping[str, str]] = None,
        missing_policy: str = "raise",
    ) -> None:
        self.coordinates = coordinates
        self.fields = fields
        self.uncertainties = uncertainties
        self.units = units
        self.missing_policy = missing_policy
        self._dataset: Optional[Dataset] = None

    def load_results(self, source: Any) -> Dataset:
        if isinstance(source, Dataset):
            ds = source
        else:
            p = Path(source)
            suf = p.suffix.lower()
            if suf == ".json":
                ds = Dataset.from_json(p)
            elif suf == ".npz":
                ds = Dataset.from_numpy(
                    p,
                    coordinates=self.coordinates,
                    fields=self.fields,
                    uncertainties=self.uncertainties,
                    units=self.units,
                    missing_policy=self.missing_policy,
                )
            else:
                ds = Dataset.from_csv(
                    p,
                    coordinates=self.coordinates,
                    fields=self.fields,
                    uncertainties=self.uncertainties,
                    units=self.units,
                    missing_policy=self.missing_policy,
                )
        ds.mesh_metadata = {
            "n_points": ds.n_rows,
            "coordinates": list(ds.coordinates.keys()) or None,
            "cell_spacings": ds.mesh_metadata.get("cell_spacings"),
            "geometry": ds.mesh_metadata.get("geometry"),
        }
        ds.solver_metadata = {
            "solver": ds.solver_metadata.get("solver"),
            "method": ds.solver_metadata.get("method"),
            "tolerances": ds.solver_metadata.get("tolerances"),
            "source_format": ds.provenance.get("format"),
        }
        self._dataset = ds
        return ds

    def _require_loaded(self) -> Dataset:
        if self._dataset is None:
            raise RuntimeError("No results loaded; call load_results(source) first.")
        return self._dataset

    def extract_fields(self) -> Dict[str, np.ndarray]:
        return dict(self._require_loaded().fields)

    def extract_mesh_metadata(self) -> Dict[str, Any]:
        return dict(self._require_loaded().mesh_metadata)

    def extract_solver_metadata(self) -> Dict[str, Any]:
        return dict(self._require_loaded().solver_metadata)

    def export_standard_format(self, path: Union[str, Path]) -> Path:
        ds = self._require_loaded()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".json":
            out.write_text(ds.to_json(), encoding="utf-8")
            return out
        return ds.to_csv(out)


class SciPyAdapter(SolverAdapter):
    """Adapter for SciPy `solve_ivp` outputs or Rosolver `Trajectory` / `PDESolution` objects."""

    def __init__(self, *, units: Optional[Mapping[str, str]] = None, name: str = "scipy_solution") -> None:
        self.units = dict(units or {})
        self.name = name
        self._dataset: Optional[Dataset] = None

    def load_results(self, source: Any) -> Dataset:
        if isinstance(source, Trajectory):
            ds = Dataset(
                name=self.name,
                coordinates={ "t": source.t },
                fields=source.as_dict(),
                units=self.units,
                mesh_metadata={"n_points": len(source.t), "dt": source.dt},
                solver_metadata={
                    "solver": "ODESystem",
                    "method": source.method,
                    "nfev": source.nfev,
                    "njev": source.njev,
                    "success": source.success,
                },
            )
        elif isinstance(source, PDESolution):
            fields = {f: source.final(f) for f in source.fields}
            ds = Dataset(
                name=self.name,
                coordinates={"x": source.x},
                fields=fields,
                units=self.units,
                mesh_metadata={
                    "n_cells": source.n_cells,
                    "h": source.h,
                    "edges": [float(source.edges[0]), float(source.edges[-1])],
                },
                solver_metadata={
                    "solver": "ConservationLaw1D",
                    "method": source.scheme,
                    "reconstruction": source.reconstruction,
                    "dt": source.dt,
                    "n_steps": source.n_steps,
                    "nfev": source.nfev,
                    "formal_order": source.formal_order,
                },
            )
        elif hasattr(source, "t") and hasattr(source, "y"):
            t_arr = np.asarray(source.t, dtype=float)
            y_arr = np.asarray(source.y, dtype=float)
            if y_arr.ndim == 1:
                fields = {"y": y_arr}
            elif y_arr.shape[1] == t_arr.size:
                fields = {f"y_{i}": y_arr[i] for i in range(y_arr.shape[0])}
            else:
                fields = {f"y_{i}": y_arr[:, i] for i in range(y_arr.shape[1])}
            ds = Dataset(
                name=self.name,
                coordinates={"t": t_arr},
                fields=fields,
                units=self.units,
                mesh_metadata={"n_points": t_arr.size, "geometry": None},
                solver_metadata={
                    "solver": "scipy.integrate.solve_ivp",
                    "nfev": getattr(source, "nfev", None),
                    "njev": getattr(source, "njev", None),
                    "success": getattr(source, "success", None),
                    "message": getattr(source, "message", None),
                },
            )
        else:
            raise TypeError(f"Unsupported SciPy source type: {type(source)!r}.")
        self._dataset = ds
        return ds

    def _require_loaded(self) -> Dataset:
        if self._dataset is None:
            raise RuntimeError("No results loaded; call load_results(source) first.")
        return self._dataset

    def extract_fields(self) -> Dict[str, np.ndarray]:
        return dict(self._require_loaded().fields)

    def extract_mesh_metadata(self) -> Dict[str, Any]:
        return dict(self._require_loaded().mesh_metadata)

    def extract_solver_metadata(self) -> Dict[str, Any]:
        return dict(self._require_loaded().solver_metadata)

    def export_standard_format(self, path: Union[str, Path]) -> Path:
        ds = self._require_loaded()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(ds.to_json(), encoding="utf-8")
        return out


class VerificationStudy:
    """
    High-level verification entry point over external or internal `Dataset` results.

    Delegates all numerical verification algorithms directly to `rosolver.verify`.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        coordinate: Optional[str] = None,
        field: Optional[str] = None,
        reference: Optional[str] = None,
        spacing_col: Optional[str] = None,
        qoi_col: Optional[str] = None,
    ) -> None:
        self.dataset = dataset
        self.coordinate = coordinate or (next(iter(dataset.coordinates), None))
        self.field = field or (next(iter(dataset.fields), None))
        self.reference = reference
        self.spacing_col = spacing_col
        self.qoi_col = qoi_col

    @classmethod
    def from_csv(
        cls,
        file: Union[str, Path],
        *,
        coordinate: Optional[str] = None,
        field: Optional[str] = None,
        reference: Optional[str] = None,
        spacing_col: Optional[str] = None,
        qoi_col: Optional[str] = None,
        units: Optional[Mapping[str, str]] = None,
        missing_policy: str = "raise",
    ) -> "VerificationStudy":
        coords = [coordinate] if coordinate else ([spacing_col] if spacing_col else None)
        adapter = GenericCSVAdapter(
            coordinates=coords,
            units=units,
            missing_policy=missing_policy,
        )
        ds = adapter.load_results(file)
        return cls(
            ds,
            coordinate=coordinate,
            field=field,
            reference=reference,
            spacing_col=spacing_col,
            qoi_col=qoi_col,
        )

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        *,
        coordinate: Optional[str] = None,
        field: Optional[str] = None,
        reference: Optional[str] = None,
        spacing_col: Optional[str] = None,
        qoi_col: Optional[str] = None,
    ) -> "VerificationStudy":
        return cls(
            dataset,
            coordinate=coordinate,
            field=field,
            reference=reference,
            spacing_col=spacing_col,
            qoi_col=qoi_col,
        )

    @classmethod
    def from_adapter(
        cls,
        adapter: SolverAdapter,
        source: Any,
        *,
        coordinate: Optional[str] = None,
        field: Optional[str] = None,
        reference: Optional[str] = None,
    ) -> "VerificationStudy":
        ds = adapter.load_results(source)
        return cls(ds, coordinate=coordinate, field=field, reference=reference)

    def grid_convergence(
        self,
        *,
        spacings: Optional[Sequence[float]] = None,
        quantities: Optional[Sequence[float]] = None,
        spacing_col: Optional[str] = None,
        qoi_col: Optional[str] = None,
        refinement_levels: Optional[Sequence[int]] = None,
        formal_order: Optional[float] = None,
        safety_factor: float = 1.25,
        gci_tol: float = 0.05,
        order_tol: float = 0.5,
    ) -> StandardResult:
        """Compute Roache/Celik GCI using `rosolver.verify.gci_from_values`."""
        if spacings is not None and quantities is not None:
            hs = list(spacings)
            qs = list(quantities)
        else:
            s_col = spacing_col or self.spacing_col or self.coordinate
            q_col = qoi_col or self.qoi_col or self.field
            if s_col is None or q_col is None:
                raise ValueError(
                    "Specify (spacings, quantities) or (spacing_col, qoi_col) for grid_convergence."
                )
            hs_arr = self.dataset.column(s_col)
            qs_arr = self.dataset.column(q_col)
            if refinement_levels is not None:
                idxs = [int(i) for i in refinement_levels]
                hs_arr = hs_arr[idxs]
                qs_arr = qs_arr[idxs]
            hs = [float(x) for x in hs_arr]
            qs = [float(x) for x in qs_arr]

        rep: GCIReport = gci_from_values(
            spacings=hs,
            quantities=qs,
            formal_order=formal_order,
            safety_factor=safety_factor,
            gci_tol=gci_tol,
            order_tol=order_tol,
        )
        return standardize_result(
            rep,
            kind="gci",
            model_name=self.dataset.name,
            units=self.dataset.units,
            solver_config={"formal_order": formal_order, "safety_factor": safety_factor},
        )

    def reference_error(
        self,
        *,
        field: Optional[str] = None,
        reference: Optional[Union[str, Sequence[float], Callable[[np.ndarray], np.ndarray]]] = None,
        coordinate: Optional[str] = None,
        l2_tol: Optional[float] = None,
        linf_tol: Optional[float] = None,
    ) -> StandardResult:
        """Compute L1, L2, and Linf norms against a reference column, array, callable, or safe expression."""
        f_name = field or self.field
        if f_name is None:
            raise ValueError("Specify `field` for reference_error.")
        sim_vals = self.dataset.column(f_name)

        ref_spec = reference if reference is not None else self.reference
        if ref_spec is None:
            raise ValueError("Specify `reference` column, array, callable, or expression.")

        c_name = coordinate or self.coordinate
        if callable(ref_spec):
            if c_name is None:
                raise ValueError("Callable reference requires a coordinate column.")
            ref_vals = np.asarray(ref_spec(self.dataset.column(c_name)), dtype=float)
        elif isinstance(ref_spec, str) and ref_spec in self.dataset.as_dict():
            ref_vals = self.dataset.column(ref_spec)
        elif isinstance(ref_spec, str):
            if c_name is None:
                raise ValueError(f"Reference '{ref_spec}' is not a dataset column and requires a coordinate.")
            expr = parse_expr(ref_spec)
            x_sym = real_symbol(c_name)
            fn = sp.lambdify([x_sym], expr, "numpy")
            ref_vals = np.broadcast_to(np.asarray(fn(self.dataset.column(c_name)), dtype=float), sim_vals.shape)
        else:
            ref_vals = np.asarray(ref_spec, dtype=float)

        diff = np.abs(sim_vals - ref_vals)
        l1 = float(np.mean(diff))
        l2 = float(np.sqrt(np.mean(diff ** 2)))
        linf = float(np.max(diff))

        warnings = []
        failed = False
        if l2_tol is not None and l2 > l2_tol:
            failed = True
            warnings.append(f"L2 error {l2:.4e} exceeds tolerance {l2_tol:.4e}.")
        if linf_tol is not None and linf > linf_tol:
            failed = True
            warnings.append(f"Linf error {linf:.4e} exceeds tolerance {linf_tol:.4e}.")

        if failed:
            status = VerificationStatus.FAILED
        elif l2_tol is None and linf_tol is None:
            status = VerificationStatus.WARNING
            warnings.append("No error tolerances specified; raw norms reported without threshold pass/fail.")
        else:
            status = VerificationStatus.PASSED

        payload = _ReferenceErrorReport(
            field=f_name,
            n_points=int(sim_vals.size),
            l1=l1,
            l2=l2,
            linf=linf,
            status=status,
            passed=(status == VerificationStatus.PASSED),
            warnings=warnings,
        )
        return standardize_result(
            payload,
            kind="reference_error",
            model_name=self.dataset.name,
            units=self.dataset.units,
        )


from dataclasses import dataclass, field as _dc_field


@dataclass
class _ReferenceErrorReport(ResultMixin):
    field: str
    n_points: int
    l1: float
    l2: float
    linf: float
    status: VerificationStatus
    passed: bool
    warnings: list = _dc_field(default_factory=list)
    assumptions: list = _dc_field(
        default_factory=lambda: ["Pointwise unweighted L1/L2/Linf norm comparison against supplied reference."]
    )

    def summary(self) -> str:
        return (
            f"=== Reference Comparison Report [{self.field}] ===\n"
            f"  Status: {self.status.value} | N={self.n_points}\n"
            f"  L1={self.l1:.4e} | L2={self.l2:.4e} | Linf={self.linf:.4e}"
        )
