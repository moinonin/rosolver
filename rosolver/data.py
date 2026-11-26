"""Standard external data container and safe dataset importers (ENG_SPEC.md P2.2)."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

from .dimensions import Dimension
from .results import ResultMixin, to_jsonable


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _as_name_list(spec: Union[None, str, Sequence[str]]) -> List[str]:
    if spec is None:
        return []
    if isinstance(spec, str):
        return [spec]
    return list(spec)


@dataclass
class Dataset(ResultMixin):
    """
    Canonical container for experimental measurements or external numerical solver outputs.

    Preserves coordinates, field variables, uncertainties, physical units, mesh/solver metadata,
    and input provenance without fabricating unavailable metadata.
    """

    name: str
    coordinates: Dict[str, np.ndarray] = field(default_factory=dict)
    fields: Dict[str, np.ndarray] = field(default_factory=dict)
    uncertainties: Dict[str, np.ndarray] = field(default_factory=dict)
    units: Dict[str, str] = field(default_factory=dict)
    mesh_metadata: Dict[str, Any] = field(default_factory=dict)
    solver_metadata: Dict[str, Any] = field(default_factory=dict)
    boundary_conditions: Optional[Dict[str, Any]] = None
    initial_conditions: Optional[Dict[str, Any]] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.coordinates = {k: np.asarray(v, dtype=float) for k, v in self.coordinates.items()}
        self.fields = {k: np.asarray(v, dtype=float) for k, v in self.fields.items()}
        self.uncertainties = {k: np.asarray(v, dtype=float) for k, v in self.uncertainties.items()}
        for var, unit_str in self.units.items():
            try:
                Dimension.from_unit(unit_str)
            except Exception as exc:
                raise ValueError(f"Invalid unit '{unit_str}' for '{var}': {exc}") from exc

        lengths = [
            arr.shape[0]
            for arr in (*self.coordinates.values(), *self.fields.values(), *self.uncertainties.values())
            if arr.ndim >= 1
        ]
        if lengths and len(set(lengths)) > 1:
            raise ValueError(f"Inconsistent leading lengths across dataset columns: {lengths}.")

    @property
    def n_rows(self) -> int:
        for arr in (*self.coordinates.values(), *self.fields.values(), *self.uncertainties.values()):
            if arr.ndim >= 1:
                return int(arr.shape[0])
        return 0

    def as_dict(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        out.update(self.coordinates)
        out.update(self.fields)
        for k, v in self.uncertainties.items():
            out[f"u_{k}"] = v
        return out

    def column(self, name: str) -> np.ndarray:
        if name in self.coordinates:
            return self.coordinates[name]
        if name in self.fields:
            return self.fields[name]
        if name in self.uncertainties:
            return self.uncertainties[name]
        raise KeyError(f"Column '{name}' not found in dataset '{self.name}'.")

    def uncertainty_for(self, field_name: str) -> Optional[np.ndarray]:
        return self.uncertainties.get(field_name)

    def to_csv(self, path: Union[str, Path]) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cols = self.as_dict()
        headers = list(cols.keys())
        n = self.n_rows
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(headers)
            for i in range(n):
                writer.writerow([float(cols[h][i]) for h in headers])
        return out

    def to_npz(self, path: Union[str, Path]) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **self.as_dict())
        return out

    @classmethod
    def from_mapping(
        cls,
        raw_cols: Mapping[str, Sequence[float]],
        *,
        name: str = "dataset",
        coordinates: Union[None, str, Sequence[str]] = None,
        fields: Union[None, str, Sequence[str]] = None,
        uncertainties: Union[None, Mapping[str, str], Sequence[str]] = None,
        units: Optional[Mapping[str, str]] = None,
        missing_policy: str = "raise",
        mesh_metadata: Optional[Mapping[str, Any]] = None,
        solver_metadata: Optional[Mapping[str, Any]] = None,
        boundary_conditions: Optional[Mapping[str, Any]] = None,
        initial_conditions: Optional[Mapping[str, Any]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "Dataset":
        if missing_policy not in ("raise", "drop", "nan"):
            raise ValueError("missing_policy must be 'raise', 'drop', or 'nan'.")
        if not raw_cols:
            raise ValueError("Cannot construct Dataset from empty data.")

        arr_cols = {k: np.asarray(v, dtype=float) for k, v in raw_cols.items()}
        n_raw = next(iter(arr_cols.values())).shape[0]
        if any(a.shape[0] != n_raw for a in arr_cols.values()):
            raise ValueError("All dataset columns must have equal length.")

        finite_mask = np.ones(n_raw, dtype=bool)
        bad_cols: List[str] = []
        for col, arr in arr_cols.items():
            ok = np.isfinite(arr)
            if not np.all(ok):
                bad_cols.append(col)
                finite_mask &= ok

        n_dropped = 0
        if bad_cols:
            if missing_policy == "raise":
                raise ValueError(
                    f"Non-finite values in dataset column(s) {bad_cols}; use missing_policy='drop' or 'nan'."
                )
            if missing_policy == "drop":
                n_dropped = int(n_raw - np.count_nonzero(finite_mask))
                arr_cols = {k: v[finite_mask] for k, v in arr_cols.items()}

        coord_names = _as_name_list(coordinates)
        unc_map: Dict[str, str] = {}
        if isinstance(uncertainties, Mapping):
            unc_map = {str(k): str(v) for k, v in uncertainties.items()}
        elif uncertainties is not None:
            for col in uncertainties:
                if col.startswith("u_") and col[2:] in arr_cols:
                    unc_map[col[2:]] = col
                elif col.startswith("sigma_") and col[6:] in arr_cols:
                    unc_map[col[6:]] = col
                else:
                    unc_map[col] = col

        unc_cols_used = set(unc_map.values())
        if fields is None:
            field_names = [k for k in arr_cols if k not in coord_names and k not in unc_cols_used]
        else:
            field_names = _as_name_list(fields)

        missing = [
            col for col in (*coord_names, *field_names, *unc_cols_used)
            if col not in arr_cols
        ]
        if missing:
            raise KeyError(f"Requested column(s) {missing} not found in {list(arr_cols.keys())}.")

        prov = {
            "loaded_at": datetime.now(timezone.utc).isoformat(),
            "n_rows": int(next(iter(arr_cols.values())).shape[0]),
            "n_dropped": n_dropped,
            **dict(provenance or {}),
        }
        return cls(
            name=name,
            coordinates={c: arr_cols[c] for c in coord_names},
            fields={f: arr_cols[f] for f in field_names},
            uncertainties={f: arr_cols[col] for f, col in unc_map.items()},
            units=dict(units or {}),
            mesh_metadata=dict(mesh_metadata or {}),
            solver_metadata=dict(solver_metadata or {}),
            boundary_conditions=dict(boundary_conditions) if boundary_conditions is not None else None,
            initial_conditions=dict(initial_conditions) if initial_conditions is not None else None,
            provenance=prov,
        )

    @classmethod
    def from_csv(
        cls,
        path: Union[str, Path],
        *,
        name: Optional[str] = None,
        coordinates: Union[None, str, Sequence[str]] = None,
        fields: Union[None, str, Sequence[str]] = None,
        uncertainties: Union[None, Mapping[str, str], Sequence[str]] = None,
        units: Optional[Mapping[str, str]] = None,
        delimiter: str = ",",
        missing_policy: str = "raise",
        mesh_metadata: Optional[Mapping[str, Any]] = None,
        solver_metadata: Optional[Mapping[str, Any]] = None,
    ) -> "Dataset":
        if missing_policy not in ("raise", "drop", "nan"):
            raise ValueError("missing_policy must be 'raise', 'drop', or 'nan'.")
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"CSV file not found: '{src}'.")

        cols: Dict[str, List[float]] = {}
        with src.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter=delimiter)
            if not reader.fieldnames:
                raise ValueError(f"CSV file '{src}' has no header row.")
            headers = [h.strip() for h in reader.fieldnames if h is not None]
            for h in headers:
                cols[h] = []
            for row_idx, row in enumerate(reader, start=2):
                for h_raw in reader.fieldnames:
                    if h_raw is None:
                        continue
                    h = h_raw.strip()
                    raw_val = (row.get(h_raw) or "").strip()
                    if raw_val == "" or raw_val.lower() in ("nan", "na", "null", "none"):
                        if missing_policy == "raise":
                            raise ValueError(f"Missing value in '{src}' row {row_idx}, column '{h}'.")
                        cols[h].append(float("nan"))
                    else:
                        try:
                            cols[h].append(float(raw_val))
                        except ValueError as exc:
                            if missing_policy == "raise":
                                raise ValueError(
                                    f"Non-numeric value '{raw_val}' in '{src}' row {row_idx}, column '{h}'."
                                ) from exc
                            cols[h].append(float("nan"))

        return cls.from_mapping(
            cols,
            name=name or src.stem,
            coordinates=coordinates,
            fields=fields,
            uncertainties=uncertainties,
            units=units,
            missing_policy=missing_policy,
            mesh_metadata=mesh_metadata,
            solver_metadata=solver_metadata,
            provenance={"source_path": src.name, "format": "csv", "sha256": _sha256_path(src)},
        )

    @classmethod
    def from_json(
        cls,
        source: Union[str, Path, Mapping[str, Any]],
        *,
        name: Optional[str] = None,
    ) -> "Dataset":
        prov: Dict[str, Any] = {"format": "json"}
        if isinstance(source, Mapping):
            data = dict(source)
        else:
            p = Path(source)
            if p.exists():
                prov["source_path"] = p.name
                prov["sha256"] = _sha256_path(p)
                data = json.loads(p.read_text(encoding="utf-8"))
                name = name or p.stem
            else:
                data = json.loads(str(source))
        return cls._from_structured_dict(data, name=name, provenance=prov)

    @classmethod
    def from_yaml(
        cls,
        source: Union[str, Path],
        *,
        name: Optional[str] = None,
    ) -> "Dataset":
        prov: Dict[str, Any] = {"format": "yaml"}
        p = Path(source)
        if p.exists():
            prov["source_path"] = p.name
            prov["sha256"] = _sha256_path(p)
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
            name = name or p.stem
        else:
            data = yaml.safe_load(str(source))
        if not isinstance(data, Mapping):
            raise ValueError("YAML dataset must deserialize to a mapping.")
        return cls._from_structured_dict(data, name=name, provenance=prov)

    @classmethod
    def _from_structured_dict(
        cls,
        data: Mapping[str, Any],
        *,
        name: Optional[str],
        provenance: Dict[str, Any],
    ) -> "Dataset":
        if "fields" in data or "coordinates" in data:
            return cls(
                name=str(name or data.get("name", "dataset")),
                coordinates={k: np.asarray(v, dtype=float) for k, v in (data.get("coordinates") or {}).items()},
                fields={k: np.asarray(v, dtype=float) for k, v in (data.get("fields") or {}).items()},
                uncertainties={k: np.asarray(v, dtype=float) for k, v in (data.get("uncertainties") or {}).items()},
                units=dict(data.get("units") or {}),
                mesh_metadata=dict(data.get("mesh_metadata") or {}),
                solver_metadata=dict(data.get("solver_metadata") or {}),
                boundary_conditions=data.get("boundary_conditions"),
                initial_conditions=data.get("initial_conditions"),
                provenance={**provenance, **dict(data.get("provenance") or {})},
            )
        return cls.from_mapping(data, name=name or "dataset", provenance=provenance)

    @classmethod
    def from_numpy(
        cls,
        source: Union[str, Path, Mapping[str, np.ndarray]],
        *,
        name: Optional[str] = None,
        coordinates: Union[None, str, Sequence[str]] = None,
        fields: Union[None, str, Sequence[str]] = None,
        uncertainties: Union[None, Mapping[str, str], Sequence[str]] = None,
        units: Optional[Mapping[str, str]] = None,
        missing_policy: str = "raise",
    ) -> "Dataset":
        prov: Dict[str, Any] = {"format": "numpy"}
        if isinstance(source, Mapping):
            raw = dict(source)
        else:
            p = Path(source)
            if not p.exists():
                raise FileNotFoundError(f"NumPy file not found: '{p}'.")
            prov["source_path"] = p.name
            prov["sha256"] = _sha256_path(p)
            name = name or p.stem
            with np.load(p, allow_pickle=False) as npz:
                raw = {k: npz[k] for k in npz.files}
        return cls.from_mapping(
            raw,
            name=name or "numpy_dataset",
            coordinates=coordinates,
            fields=fields,
            uncertainties=uncertainties,
            units=units,
            missing_policy=missing_policy,
            provenance=prov,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: Any,
        *,
        name: str = "dataframe",
        coordinates: Union[None, str, Sequence[str]] = None,
        fields: Union[None, str, Sequence[str]] = None,
        uncertainties: Union[None, Mapping[str, str], Sequence[str]] = None,
        units: Optional[Mapping[str, str]] = None,
        missing_policy: str = "raise",
    ) -> "Dataset":
        if not hasattr(df, "columns"):
            raise TypeError("Expected a DataFrame-like object with a `.columns` attribute.")
        raw = {str(col): np.asarray(df[col], dtype=float) for col in df.columns}
        return cls.from_mapping(
            raw,
            name=name,
            coordinates=coordinates,
            fields=fields,
            uncertainties=uncertainties,
            units=units,
            missing_policy=missing_policy,
            provenance={"format": "dataframe"},
        )

    @classmethod
    def from_hdf5(
        cls,
        path: Union[str, Path],
        *,
        group: str = "/",
        name: Optional[str] = None,
        coordinates: Union[None, str, Sequence[str]] = None,
        fields: Union[None, str, Sequence[str]] = None,
        uncertainties: Union[None, Mapping[str, str], Sequence[str]] = None,
        units: Optional[Mapping[str, str]] = None,
        missing_policy: str = "raise",
    ) -> "Dataset":
        try:
            import h5py  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Reading HDF5 files requires the optional 'io' dependency: pip install 'rosolver[io]' (h5py)."
            ) from exc

        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"HDF5 file not found: '{src}'.")
        raw: Dict[str, np.ndarray] = {}
        with h5py.File(src, "r") as fh:
            grp = fh[group]
            for key in grp.keys():
                raw[str(key)] = np.asarray(grp[key][...], dtype=float)
        return cls.from_mapping(
            raw,
            name=name or src.stem,
            coordinates=coordinates,
            fields=fields,
            uncertainties=uncertainties,
            units=units,
            missing_policy=missing_policy,
            provenance={"source_path": src.name, "format": "hdf5", "group": group, "sha256": _sha256_path(src)},
        )
