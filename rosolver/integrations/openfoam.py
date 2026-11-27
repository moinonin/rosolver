"""OpenFOAM post-processing audit adapter (ENG_SPEC.md P2.3+ / Sprint 8 Slim).

Parses OpenFOAM ``postProcessing/`` sample sets (``.xy``, ``.csv``, ``.dat``),
solver residual logs (``residuals.dat``), and case control metadata
(``system/controlDict``) without requiring an OpenFOAM runtime installation.
Unsupported or absent metadata fields are explicitly recorded as ``None``.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ..data import Dataset, _sha256_path
from .adapters import SolverAdapter

_CONTROL_KV_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s+([^;/{}]+?)\s*;\s*$")
_VECTOR_RE = re.compile(r"\(\s*([^()]+?)\s*\)")


def _strip_cpp_comments(text: str) -> str:
    """Remove C/C++ style // and /* */ comments from OpenFOAM dictionary text."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def _parse_control_dict(case_dir: Path) -> Dict[str, Any]:
    """Extract basic solver/time metadata from ``system/controlDict`` if present."""
    cd_path = case_dir / "system" / "controlDict"
    meta: Dict[str, Any] = {
        "solver": None,
        "application": None,
        "startTime": None,
        "endTime": None,
        "deltaT": None,
        "writeInterval": None,
    }
    if not cd_path.exists():
        return meta

    clean = _strip_cpp_comments(cd_path.read_text(encoding="utf-8", errors="replace"))
    for line in clean.splitlines():
        m = _CONTROL_KV_RE.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if key == "application":
            meta["application"] = val
            meta["solver"] = f"OpenFOAM:{val}"
        elif key in ("startTime", "endTime", "deltaT", "writeInterval"):
            try:
                meta[key] = float(val)
            except ValueError:
                meta[key] = val
    return meta


def _parse_time_dirs(parent: Path) -> List[Tuple[float, Path]]:
    """Return numeric OpenFOAM time subdirectories sorted ascending."""
    out: List[Tuple[float, Path]] = []
    if not parent.is_dir():
        return out
    for child in parent.iterdir():
        if child.is_dir():
            try:
                out.append((float(child.name), child))
            except ValueError:
                continue
    out.sort(key=lambda pair: pair[0])
    return out


def _tokenize_foam_row(line: str) -> List[float]:
    """Parse an OpenFOAM ASCII data row, expanding ``(vx vy vz)`` vector tuples."""
    expanded = _VECTOR_RE.sub(lambda m: " " + m.group(1) + " ", line)
    tokens = [tok for tok in re.split(r"[\s,]+", expanded.strip()) if tok]
    return [float(tok) for tok in tokens]


def _infer_column_names(
    header_tokens: Optional[List[str]],
    n_cols: int,
    *,
    is_residuals: bool = False,
) -> List[str]:
    """Build deterministic column names matching ``n_cols`` numeric columns."""
    if header_tokens and len(header_tokens) == n_cols:
        return header_tokens

    # Common OpenFOAM vector expansion: e.g. header ["x", "U"] with 4 numeric columns -> ["x", "U_x", "U_y", "U_z"]
    if header_tokens and len(header_tokens) < n_cols:
        extra = n_cols - len(header_tokens)
        if extra == 2:
            # Last column is a 3-component vector
            base = header_tokens[-1]
            return header_tokens[:-1] + [f"{base}_x", f"{base}_y", f"{base}_z"]
        if extra % 2 == 0 and len(header_tokens) >= 1:
            names: List[str] = list(header_tokens)
            idx = 0
            while len(names) < n_cols:
                names.append(f"col_{idx}")
                idx += 1
            return names

    if is_residuals:
        return ["Time"] + [f"res_{i}" for i in range(max(0, n_cols - 1))]
    if n_cols == 2:
        return ["x", "field_0"]
    if n_cols >= 4:
        return ["x", "y", "z"] + [f"field_{i}" for i in range(n_cols - 3)]
    return [f"col_{i}" for i in range(n_cols)]


def _parse_openfoam_table(
    file_path: Path,
    *,
    is_residuals: bool = False,
) -> Tuple[List[str], np.ndarray]:
    """Parse an OpenFOAM ``.xy``, ``.csv``, or ``residuals.dat`` table."""
    raw_lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_tokens: Optional[List[str]] = None
    rows: List[List[float]] = []

    if file_path.suffix.lower() == ".csv":
        non_comment = [ln for ln in raw_lines if ln.strip() and not ln.lstrip().startswith("#")]
        if not non_comment:
            raise ValueError(f"OpenFOAM CSV file '{file_path.name}' contains no data rows.")
        reader = csv.reader(non_comment)
        first = [c.strip() for c in next(reader)]
        try:
            rows.append([float(c) for c in first])
        except ValueError:
            header_tokens = first
        for r in reader:
            if not any(c.strip() for c in r):
                continue
            rows.append([float(c.strip()) for c in r])
    else:
        for line in raw_lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                candidate = stripped.lstrip("#").strip()
                # Skip title/banner comments like "# Residuals" when a multi-token header follows
                toks = [t for t in re.split(r"[\s,]+", candidate) if t]
                if len(toks) >= 2:
                    header_tokens = toks
                continue
            try:
                rows.append(_tokenize_foam_row(stripped))
            except ValueError:
                # Non-comment header line (e.g. tab-separated column labels)
                if not rows and header_tokens is None:
                    header_tokens = [t for t in re.split(r"[\s,]+", stripped) if t]
                    continue
                raise ValueError(f"Could not parse numeric OpenFOAM row in '{file_path.name}': {stripped!r}")

    if not rows:
        raise ValueError(f"OpenFOAM data file '{file_path.name}' has no numeric rows.")

    n_cols = len(rows[0])
    if any(len(r) != n_cols for r in rows):
        raise ValueError(f"Inconsistent column counts in OpenFOAM file '{file_path.name}'.")

    names = _infer_column_names(header_tokens, n_cols, is_residuals=is_residuals)
    arr = np.asarray(rows, dtype=float)
    return names, arr


class OpenFOAMAdapter(SolverAdapter):
    """Adapter for OpenFOAM ``postProcessing/`` sample sets, residuals, and case directories.

    Supports:
    - Single sample-set files (``.xy``, ``.csv``, ``.dat``) including vector ``(vx vy vz)`` columns.
    - Residual histories (``residuals.dat``) with automatic initial/final residual reduction metrics.
    - Full OpenFOAM case directories (auto-discovering ``system/controlDict`` and ``postProcessing/``).
    """

    def __init__(
        self,
        *,
        function_object: Optional[str] = None,
        time: Optional[Union[str, float]] = None,
        sample_file: Optional[str] = None,
        coordinates: Union[None, str, Sequence[str]] = None,
        units: Optional[Mapping[str, str]] = None,
        residual_tol: float = 1e-4,
    ) -> None:
        self.function_object = function_object
        self.time = time
        self.sample_file = sample_file
        self.coordinates = coordinates
        self.units = dict(units or {})
        self.residual_tol = float(residual_tol)
        self._dataset: Optional[Dataset] = None
        self._residual_summary: Optional[Dict[str, Any]] = None

    def _resolve_target_file(self, source: Path) -> Tuple[Path, Optional[Path], Optional[float]]:
        """Resolve ``source`` (file, functionObject dir, or case dir) to ``(data_file, case_dir, time_val)``."""
        if source.is_file():
            # Walk up to detect if inside <case>/postProcessing/...
            case_dir: Optional[Path] = None
            for parent in source.parents:
                if (parent / "system" / "controlDict").exists():
                    case_dir = parent
                    break
            time_val: Optional[float] = None
            try:
                time_val = float(source.parent.name)
            except ValueError:
                time_val = None
            return source, case_dir, time_val

        if not source.is_dir():
            raise FileNotFoundError(f"OpenFOAM source path not found: '{source}'.")

        case_dir = source if (source / "system" / "controlDict").exists() else None
        pp_dir = source / "postProcessing" if (source / "postProcessing").is_dir() else source

        # Select functionObject directory
        if self.function_object:
            fo_dir = pp_dir / self.function_object
            if not fo_dir.is_dir():
                raise FileNotFoundError(
                    f"OpenFOAM functionObject '{self.function_object}' not found under '{pp_dir.name}'."
                )
        else:
            time_dirs_direct = _parse_time_dirs(pp_dir)
            if time_dirs_direct:
                fo_dir = pp_dir
            else:
                subdirs = sorted([d for d in pp_dir.iterdir() if d.is_dir()], key=lambda p: p.name)
                if not subdirs:
                    raise FileNotFoundError(f"No OpenFOAM postProcessing subdirectories found in '{pp_dir.name}'.")
                # Prefer non-residual sample sets first unless only residuals exist
                non_res = [d for d in subdirs if "residual" not in d.name.lower()]
                fo_dir = non_res[0] if non_res else subdirs[0]

        time_dirs = _parse_time_dirs(fo_dir)
        if not time_dirs:
            raise FileNotFoundError(f"No numeric time directories found under '{fo_dir.name}'.")

        if self.time is not None:
            target_t = float(self.time)
            matched = [pair for pair in time_dirs if abs(pair[0] - target_t) <= 1e-12 * max(1.0, abs(target_t))]
            if not matched:
                available = [pair[1].name for pair in time_dirs]
                raise FileNotFoundError(f"Requested OpenFOAM time {self.time} not in {available}.")
            sel_time, sel_dir = matched[-1]
        else:
            sel_time, sel_dir = time_dirs[-1]

        if self.sample_file:
            target = sel_dir / self.sample_file
            if not target.is_file():
                raise FileNotFoundError(f"Sample file '{self.sample_file}' not found in time '{sel_dir.name}'.")
            return target, case_dir, sel_time

        candidates = sorted(
            [f for f in sel_dir.iterdir() if f.is_file() and f.suffix.lower() in (".xy", ".csv", ".dat", ".txt")],
            key=lambda p: p.name,
        )
        if not candidates:
            raise FileNotFoundError(f"No .xy/.csv/.dat files found in OpenFOAM time directory '{sel_dir.name}'.")
        return candidates[0], case_dir, sel_time

    def load_results(self, source: Any) -> Dataset:
        src = Path(source)
        target_file, case_dir, time_val = self._resolve_target_file(src)
        is_residuals = "residual" in target_file.name.lower() or "residual" in target_file.parent.parent.name.lower()

        col_names, arr = _parse_openfoam_table(target_file, is_residuals=is_residuals)
        raw_cols = {name: arr[:, i] for i, name in enumerate(col_names)}

        if self.coordinates is not None:
            coord_names = [self.coordinates] if isinstance(self.coordinates, str) else list(self.coordinates)
        elif is_residuals:
            coord_names = [col_names[0]]
        else:
            coord_names = [
                c for c in col_names
                if c in ("x", "y", "z", "t") or c.lower() in ("distance", "arc_length", "time")
            ]
            if not coord_names:
                coord_names = [col_names[0]]

        field_names = [c for c in col_names if c not in coord_names]
        ctrl_meta = _parse_control_dict(case_dir) if case_dir is not None else {
            "solver": "OpenFOAM",
            "application": None,
            "startTime": None,
            "endTime": None,
            "deltaT": None,
            "writeInterval": None,
        }
        if ctrl_meta.get("solver") is None:
            ctrl_meta["solver"] = "OpenFOAM"

        residual_summary: Optional[Dict[str, Any]] = None
        if is_residuals and field_names:
            per_field: Dict[str, Dict[str, Any]] = {}
            all_converged = True
            for fname in field_names:
                vals = raw_cols[fname]
                init_v = float(vals[0])
                final_v = float(vals[-1])
                ratio = final_v / init_v if init_v > 0.0 else float("nan")
                orders = float(-np.log10(ratio)) if (np.isfinite(ratio) and ratio > 0.0) else 0.0
                conv = bool(final_v <= self.residual_tol)
                all_converged = all_converged and conv
                per_field[fname] = {
                    "initial": init_v,
                    "final": final_v,
                    "orders_drop": round(orders, 4),
                    "converged": conv,
                }
            residual_summary = {
                "residual_tol": self.residual_tol,
                "all_converged": all_converged,
                "fields": per_field,
            }
            ctrl_meta["residuals"] = residual_summary
        self._residual_summary = residual_summary

        # Characteristic spacing estimate along 1D sample line if monotonic
        h_est: Optional[float] = None
        if not is_residuals and coord_names:
            c0 = raw_cols[coord_names[0]]
            if c0.size >= 2:
                diffs = np.diff(c0)
                pos = np.abs(diffs[np.abs(diffs) > 0])
                if pos.size > 0:
                    h_est = float(np.median(pos))

        mesh_meta: Dict[str, Any] = {
            "n_points": int(arr.shape[0]),
            "coordinates": coord_names,
            "h_sample": h_est,
            "time_directory": time_val,
            "geometry": None,
        }

        ds = Dataset.from_mapping(
            raw_cols,
            name=target_file.stem,
            coordinates=coord_names,
            fields=field_names,
            units=self.units,
            mesh_metadata=mesh_meta,
            solver_metadata=ctrl_meta,
            provenance={
                "adapter": "OpenFOAMAdapter",
                "format": "openfoam_residuals" if is_residuals else "openfoam_sample",
                "source_path": target_file.name,
                "sha256": _sha256_path(target_file),
            },
        )
        self._dataset = ds
        return ds

    def load_residuals(self, source: Union[str, Path]) -> Dict[str, Any]:
        """Convenience method to load an OpenFOAM ``residuals.dat`` and return its convergence summary."""
        self.load_results(source)
        if self._residual_summary is None:
            raise ValueError("Loaded OpenFOAM file did not appear to be a residuals table.")
        return dict(self._residual_summary)

    def _require_loaded(self) -> Dataset:
        if self._dataset is None:
            raise RuntimeError("No OpenFOAM results loaded; call load_results(source) first.")
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
