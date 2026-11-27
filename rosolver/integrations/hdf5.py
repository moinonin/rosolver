"""Hierarchical HDF5 solver-output adapter (ENG_SPEC.md P2.3+ / Sprint 8 Slim).

Requires the optional ``rosolver[io]`` extra (``h5py``). Supports both flat groups
(delegated to ``Dataset.from_hdf5``) and structured solver hierarchies
(``coordinates/``, ``fields/``, ``uncertainties/``, plus HDF5 ``.attrs`` metadata).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np

from ..data import Dataset, _sha256_path
from .adapters import SolverAdapter


def _import_h5py() -> Any:
    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "HDF5Adapter requires the optional 'io' dependency: pip install 'rosolver[io]' (h5py)."
        ) from exc
    return h5py


def _decode_attr(val: Any) -> Any:
    if isinstance(val, bytes):
        val = val.decode("utf-8", errors="replace")
    if isinstance(val, np.generic):
        return val.item()
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, str):
        s = val.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                return val
    return val


class HDF5Adapter(SolverAdapter):
    """Adapter for structured or flat HDF5 scientific simulation files (``.h5``, ``.hdf5``)."""

    def __init__(
        self,
        *,
        group: str = "/",
        coordinates: Union[None, str, Sequence[str]] = None,
        fields: Union[None, str, Sequence[str]] = None,
        uncertainties: Union[None, Mapping[str, str], Sequence[str]] = None,
        units: Optional[Mapping[str, str]] = None,
        missing_policy: str = "raise",
    ) -> None:
        self.group = group
        self.coordinates = coordinates
        self.fields = fields
        self.uncertainties = uncertainties
        self.units = dict(units or {})
        self.missing_policy = missing_policy
        self._dataset: Optional[Dataset] = None

    def load_results(self, source: Any) -> Dataset:
        h5py = _import_h5py()
        src = Path(source)
        if not src.exists():
            raise FileNotFoundError(f"HDF5 file not found: '{src}'.")

        with h5py.File(src, "r") as fh:
            grp = fh[self.group]
            attrs = {str(k): _decode_attr(v) for k, v in getattr(grp, "attrs", {}).items()}
            units = dict(attrs.get("units") or {}) if isinstance(attrs.get("units"), Mapping) else {}
            units.update(self.units)
            mesh_meta = (
                dict(attrs.get("mesh_metadata") or {})
                if isinstance(attrs.get("mesh_metadata"), Mapping)
                else {}
            )
            solver_meta = (
                dict(attrs.get("solver_metadata") or {})
                if isinstance(attrs.get("solver_metadata"), Mapping)
                else {}
            )
            if "solver" not in solver_meta:
                solver_meta["solver"] = attrs.get("solver")
            solver_meta.setdefault("source_format", "hdf5")

            # Structured layout: grp contains 'fields' or 'coordinates' subgroups
            has_structured = (
                "fields" in grp and hasattr(grp["fields"], "keys")
            ) or (
                "coordinates" in grp and hasattr(grp["coordinates"], "keys")
            )

            if has_structured:
                coords_map: Dict[str, np.ndarray] = {}
                fields_map: Dict[str, np.ndarray] = {}
                unc_map: Dict[str, np.ndarray] = {}

                if "coordinates" in grp and hasattr(grp["coordinates"], "keys"):
                    for k in grp["coordinates"].keys():
                        coords_map[str(k)] = np.asarray(grp["coordinates"][k][...], dtype=float)
                if "fields" in grp and hasattr(grp["fields"], "keys"):
                    for k in grp["fields"].keys():
                        fields_map[str(k)] = np.asarray(grp["fields"][k][...], dtype=float)
                if "uncertainties" in grp and hasattr(grp["uncertainties"], "keys"):
                    for k in grp["uncertainties"].keys():
                        unc_map[str(k)] = np.asarray(grp["uncertainties"][k][...], dtype=float)

                n_pts = (
                    int(next(iter(coords_map.values())).shape[0])
                    if coords_map
                    else (int(next(iter(fields_map.values())).shape[0]) if fields_map else 0)
                )
                mesh_meta.setdefault("n_points", n_pts)
                mesh_meta.setdefault("coordinates", list(coords_map.keys()) or None)

                ds = Dataset(
                    name=str(attrs.get("name") or src.stem),
                    coordinates=coords_map,
                    fields=fields_map,
                    uncertainties=unc_map,
                    units=units,
                    mesh_metadata=mesh_meta,
                    solver_metadata=solver_meta,
                    provenance={
                        "adapter": "HDF5Adapter",
                        "format": "hdf5",
                        "group": self.group,
                        "source_path": src.name,
                        "sha256": _sha256_path(src),
                    },
                )
            else:
                # Flat group: delegate to Dataset.from_mapping
                raw_cols = {str(k): np.asarray(grp[k][...], dtype=float) for k in grp.keys()}
                ds = Dataset.from_mapping(
                    raw_cols,
                    name=str(attrs.get("name") or src.stem),
                    coordinates=self.coordinates,
                    fields=self.fields,
                    uncertainties=self.uncertainties,
                    units=units,
                    missing_policy=self.missing_policy,
                    mesh_metadata=mesh_meta,
                    solver_metadata=solver_meta,
                    provenance={
                        "adapter": "HDF5Adapter",
                        "format": "hdf5",
                        "group": self.group,
                        "source_path": src.name,
                        "sha256": _sha256_path(src),
                    },
                )
                ds.mesh_metadata.setdefault("n_points", ds.n_rows)
                ds.mesh_metadata.setdefault("coordinates", list(ds.coordinates.keys()) or None)

        self._dataset = ds
        return ds

    def _require_loaded(self) -> Dataset:
        if self._dataset is None:
            raise RuntimeError("No HDF5 results loaded; call load_results(source) first.")
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
