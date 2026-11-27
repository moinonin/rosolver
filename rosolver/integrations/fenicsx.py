"""FEniCSx / DOLFINx / VTK-CSV & ASCII VTK/VTU/XDMF audit adapter (ENG_SPEC.md P2.3+ / Sprint 8 Slim).

Ingests finite-element simulation exports without requiring ``dolfinx`` or ``vtk``:
- ParaView / VTK-CSV exports (normalizing ``Points:0..2`` -> ``x, y, z`` and ``u:0..2`` -> ``u_x, u_y, u_z``).
- Legacy ASCII VTK files (``.vtk``: ``STRUCTURED_POINTS``, ``UNSTRUCTURED_GRID``, ``POLYDATA``).
- XML VTK UnstructuredGrid ASCII files (``.vtu``).
- Inline XML XDMF files (``.xdmf``), delegating ``Format="HDF"`` references to ``HDF5Adapter``.
"""

from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ..data import Dataset, _sha256_path
from .adapters import SolverAdapter

_POINT_ALIASES = {
    "points:0": "x",
    "points:1": "y",
    "points:2": "z",
    "points_0": "x",
    "points_1": "y",
    "points_2": "z",
}
_COMP_SUFFIX = {0: "x", 1: "y", 2: "z"}


def _normalize_vtk_csv_name(col: str) -> str:
    low = col.strip().lower()
    if low in _POINT_ALIASES:
        return _POINT_ALIASES[low]
    if ":" in col:
        base, idx_str = col.rsplit(":", 1)
        if idx_str.isdigit() and int(idx_str) in _COMP_SUFFIX:
            return f"{base.strip()}_{_COMP_SUFFIX[int(idx_str)]}"
        return col.replace(":", "_").strip()
    return col.strip()


def _estimate_h_eff(coords: Dict[str, np.ndarray], n_cells: Optional[int] = None) -> Optional[float]:
    """Estimate characteristic mesh spacing from bounding box and cell/point count."""
    active_axes: List[Tuple[float, float]] = []
    for axis in ("x", "y", "z"):
        if axis in coords and coords[axis].size >= 2:
            span = float(np.max(coords[axis]) - np.min(coords[axis]))
            if span > 0.0:
                active_axes.append((float(np.min(coords[axis])), float(np.max(coords[axis]))))
    if not active_axes:
        return None
    dim = len(active_axes)
    measure = 1.0
    for lo, hi in active_axes:
        measure *= (hi - lo)
    denom = float(n_cells) if (n_cells and n_cells > 0) else float(max(1, next(iter(coords.values())).size - 1))
    return float((measure / denom) ** (1.0 / dim))


def _parse_vtk_csv(path: Path) -> Tuple[Dict[str, np.ndarray], List[str], Dict[str, Any]]:
    """Parse ParaView / VTK-CSV table into normalized coordinates + fields."""
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError(f"VTK-CSV file '{path.name}' has no header.")
        raw_headers = [h.strip() for h in reader.fieldnames if h is not None]
        norm_headers = [_normalize_vtk_csv_name(h) for h in raw_headers]
        cols: Dict[str, List[float]] = {nh: [] for nh in norm_headers}
        for row in reader:
            for rh, nh in zip(raw_headers, norm_headers):
                val = (row.get(rh) or "").strip()
                cols[nh].append(float(val))
    arr_map = {k: np.asarray(v, dtype=float) for k, v in cols.items()}
    coord_names = [c for c in ("x", "y", "z") if c in arr_map]
    if not coord_names:
        coord_names = [c for c in norm_headers if c.lower() in ("arc_length", "t", "time")]
    if not coord_names:
        coord_names = [norm_headers[0]]
    mesh_meta = {
        "n_points": int(next(iter(arr_map.values())).size),
        "n_cells": None,
        "spatial_dim": len([c for c in ("x", "y", "z") if c in arr_map]) or 1,
        "h_eff": _estimate_h_eff({c: arr_map[c] for c in coord_names}),
        "geometry": "vtk_csv",
    }
    return arr_map, coord_names, mesh_meta


def _parse_legacy_ascii_vtk(path: Path) -> Tuple[Dict[str, np.ndarray], List[str], Dict[str, Any]]:
    """Parse legacy ASCII VTK (``.vtk``) ``STRUCTURED_POINTS``, ``UNSTRUCTURED_GRID``, or ``POLYDATA``."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    tokens: List[str] = []
    header_format = None
    dataset_type = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if i == 2:
            header_format = s.upper()
        if s.upper().startswith("DATASET "):
            dataset_type = s.split(None, 1)[1].strip().upper()
        if i >= 3 and s:
            tokens.extend(s.split())

    if header_format != "ASCII":
        raise ValueError(f"Legacy VTK file '{path.name}' must be ASCII (got '{header_format}').")

    coords: Dict[str, np.ndarray] = {}
    fields: Dict[str, np.ndarray] = {}
    n_points = 0
    n_cells: Optional[int] = None

    idx = 0
    ntok = len(tokens)
    while idx < ntok:
        tok = tokens[idx].upper()
        if tok == "DIMENSIONS" and dataset_type == "STRUCTURED_POINTS":
            nx, ny, nz = int(tokens[idx + 1]), int(tokens[idx + 2]), int(tokens[idx + 3])
            n_points = nx * ny * nz
            n_cells = max(1, (max(nx - 1, 1)) * (max(ny - 1, 1)) * (max(nz - 1, 1)))
            idx += 4
            # Look ahead for ORIGIN and SPACING
            origin = [0.0, 0.0, 0.0]
            spacing = [1.0, 1.0, 1.0]
            while idx < ntok and tokens[idx].upper() in ("ORIGIN", "SPACING", "ASPECT_RATIO"):
                kw = tokens[idx].upper()
                vals = [float(tokens[idx + 1]), float(tokens[idx + 2]), float(tokens[idx + 3])]
                if kw == "ORIGIN":
                    origin = vals
                else:
                    spacing = vals
                idx += 4
            xs = origin[0] + spacing[0] * np.arange(nx, dtype=float)
            ys = origin[1] + spacing[1] * np.arange(ny, dtype=float)
            zs = origin[2] + spacing[2] * np.arange(nz, dtype=float)
            gz, gy, gx = np.meshgrid(zs, ys, xs, indexing="ij")
            coords = {"x": gx.ravel(), "y": gy.ravel(), "z": gz.ravel()}
            continue

        if tok == "POINTS":
            n_points = int(tokens[idx + 1])
            # tokens[idx + 2] is datatype (float/double)
            start = idx + 3
            end = start + 3 * n_points
            pts = np.asarray([float(v) for v in tokens[start:end]], dtype=float).reshape((n_points, 3))
            coords = {"x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2]}
            idx = end
            continue

        if tok == "CELLS":
            n_cells = int(tokens[idx + 1])
            total_ints = int(tokens[idx + 2])
            idx = idx + 3 + total_ints
            continue

        if tok == "CELL_TYPES":
            nc = int(tokens[idx + 1])
            idx = idx + 2 + nc
            continue

        if tok == "POINT_DATA":
            n_points = int(tokens[idx + 1])
            idx += 2
            continue

        if tok == "SCALARS":
            name = tokens[idx + 1]
            # tokens[idx + 2] is datatype; optional numComp before LOOKUP_TABLE
            idx += 3
            num_comp = 1
            if idx < ntok and tokens[idx].upper() != "LOOKUP_TABLE":
                try:
                    num_comp = int(tokens[idx])
                    idx += 1
                except ValueError:
                    pass
            if idx < ntok and tokens[idx].upper() == "LOOKUP_TABLE":
                idx += 2
            end = idx + n_points * num_comp
            vals = np.asarray([float(v) for v in tokens[idx:end]], dtype=float)
            if num_comp == 1:
                fields[name] = vals
            else:
                mat = vals.reshape((n_points, num_comp))
                for c in range(num_comp):
                    fields[f"{name}_{_COMP_SUFFIX.get(c, str(c))}"] = mat[:, c]
            idx = end
            continue

        if tok == "VECTORS":
            name = tokens[idx + 1]
            idx += 3  # skip datatype
            end = idx + 3 * n_points
            mat = np.asarray([float(v) for v in tokens[idx:end]], dtype=float).reshape((n_points, 3))
            fields[f"{name}_x"] = mat[:, 0]
            fields[f"{name}_y"] = mat[:, 1]
            fields[f"{name}_z"] = mat[:, 2]
            idx = end
            continue

        idx += 1

    if not coords or not fields:
        raise ValueError(f"Could not extract POINTS and POINT_DATA fields from VTK file '{path.name}'.")

    # Drop completely constant zero z/y coordinates only for spatial_dim count, keep x/y/z columns
    active_dim = sum(1 for k in ("x", "y", "z") if np.ptp(coords[k]) > 0.0) or 1
    arr_map = {**coords, **fields}
    mesh_meta = {
        "n_points": n_points,
        "n_cells": n_cells,
        "spatial_dim": active_dim,
        "h_eff": _estimate_h_eff(coords, n_cells=n_cells),
        "geometry": dataset_type or "vtk_ascii",
    }
    return arr_map, ["x", "y", "z"], mesh_meta


def _parse_vtu_ascii(path: Path) -> Tuple[Dict[str, np.ndarray], List[str], Dict[str, Any]]:
    """Parse XML VTK UnstructuredGrid ASCII (``.vtu``) file."""
    tree = ET.parse(path)
    root = tree.getroot()
    piece = root.find(".//Piece")
    if piece is None:
        raise ValueError(f"VTU file '{path.name}' missing <Piece> element.")

    n_points = int(piece.attrib.get("NumberOfPoints", "0"))
    n_cells = int(piece.attrib.get("NumberOfCells", "0")) or None

    pts_el = piece.find("./Points/DataArray")
    if pts_el is None:
        raise ValueError(f"VTU file '{path.name}' missing <Points><DataArray>.")
    if pts_el.attrib.get("format", "ascii").lower() != "ascii":
        raise ValueError(f"VTU file '{path.name}' uses non-ASCII encoding; export ASCII VTU or VTK-CSV.")
    n_comp = int(pts_el.attrib.get("NumberOfComponents", "3"))
    pts_vals = np.fromstring((pts_el.text or "").strip(), sep=" ", dtype=float)
    pts = pts_vals.reshape((n_points, n_comp))
    coord_names = ["x", "y", "z"][:n_comp]
    coords = {coord_names[i]: pts[:, i] for i in range(n_comp)}

    fields: Dict[str, np.ndarray] = {}
    pd_el = piece.find("./PointData")
    if pd_el is not None:
        for da in pd_el.findall("./DataArray"):
            if da.attrib.get("format", "ascii").lower() != "ascii":
                raise ValueError(f"VTU DataArray '{da.attrib.get('Name', '?')}' is not ASCII.")
            name = da.attrib.get("Name", f"field_{len(fields)}")
            nc = int(da.attrib.get("NumberOfComponents", "1"))
            vals = np.fromstring((da.text or "").strip(), sep=" ", dtype=float)
            if nc == 1:
                fields[name] = vals
            else:
                mat = vals.reshape((n_points, nc))
                for c in range(nc):
                    fields[f"{name}_{_COMP_SUFFIX.get(c, str(c))}"] = mat[:, c]

    if not fields:
        raise ValueError(f"VTU file '{path.name}' contains no <PointData> arrays.")

    active_dim = sum(1 for k in coord_names if np.ptp(coords[k]) > 0.0) or 1
    mesh_meta = {
        "n_points": n_points,
        "n_cells": n_cells,
        "spatial_dim": active_dim,
        "h_eff": _estimate_h_eff(coords, n_cells=n_cells),
        "geometry": "vtu_unstructured_grid",
    }
    return {**coords, **fields}, coord_names, mesh_meta


def _parse_xdmf_inline(path: Path) -> Tuple[Dict[str, np.ndarray], List[str], Dict[str, Any]]:
    """Parse XDMF (``.xdmf``) with inline ``Format="XML"`` geometry & attributes."""
    tree = ET.parse(path)
    root = tree.getroot()
    grid = root.find(".//Grid[@GridType='Uniform']") or root.find(".//Grid")
    if grid is None:
        raise ValueError(f"XDMF file '{path.name}' missing <Grid> element.")

    geom_di = grid.find("./Geometry/DataItem")
    if geom_di is None:
        raise ValueError(f"XDMF file '{path.name}' missing <Geometry><DataItem>.")
    if geom_di.attrib.get("Format", "XML").upper() == "HDF":
        raise ImportError(
            "XDMF file references external HDF5 data (Format='HDF'); use HDF5Adapter with pip install 'rosolver[io]'."
        )
    dims = [int(d) for d in geom_di.attrib.get("Dimensions", "").split()]
    pts_vals = np.fromstring((geom_di.text or "").strip(), sep=" ", dtype=float)
    n_points = dims[0] if dims else int(pts_vals.size // 3)
    n_comp = dims[1] if len(dims) > 1 else 3
    pts = pts_vals.reshape((n_points, n_comp))
    coord_names = ["x", "y", "z"][:n_comp]
    coords = {coord_names[i]: pts[:, i] for i in range(n_comp)}

    fields: Dict[str, np.ndarray] = {}
    for attr in grid.findall("./Attribute"):
        name = attr.attrib.get("Name", f"field_{len(fields)}")
        di = attr.find("./DataItem")
        if di is None:
            continue
        if di.attrib.get("Format", "XML").upper() == "HDF":
            raise ImportError(
                f"XDMF Attribute '{name}' references HDF5 (Format='HDF'); use HDF5Adapter with pip install 'rosolver[io]'."
            )
        vals = np.fromstring((di.text or "").strip(), sep=" ", dtype=float)
        if vals.size == n_points:
            fields[name] = vals
        elif vals.size % n_points == 0:
            nc = vals.size // n_points
            mat = vals.reshape((n_points, nc))
            for c in range(nc):
                fields[f"{name}_{_COMP_SUFFIX.get(c, str(c))}"] = mat[:, c]

    if not fields:
        raise ValueError(f"XDMF file '{path.name}' contains no inline <Attribute> arrays.")

    topo = grid.find("./Topology")
    n_cells = int(topo.attrib["NumberOfElements"]) if (topo is not None and "NumberOfElements" in topo.attrib) else None
    active_dim = sum(1 for k in coord_names if np.ptp(coords[k]) > 0.0) or 1
    mesh_meta = {
        "n_points": n_points,
        "n_cells": n_cells,
        "spatial_dim": active_dim,
        "h_eff": _estimate_h_eff(coords, n_cells=n_cells),
        "geometry": topo.attrib.get("TopologyType", "xdmf") if topo is not None else "xdmf",
    }
    return {**coords, **fields}, coord_names, mesh_meta


class FEniCSxAdapter(SolverAdapter):
    """Adapter for FEniCSx / DOLFINx / ParaView exports (``.csv``, ``.vtk``, ``.vtu``, ``.xdmf``).

    Zero runtime dependency on ``dolfinx`` or ``vtk``. Extracts coordinates, fields,
    characteristic mesh size ``h_eff``, and solver metadata (marking absent keys ``None``).
    """

    def __init__(
        self,
        *,
        units: Optional[Mapping[str, str]] = None,
        solver_name: str = "FEniCSx",
        element_family: Optional[str] = None,
        element_degree: Optional[int] = None,
    ) -> None:
        self.units = dict(units or {})
        self.solver_name = solver_name
        self.element_family = element_family
        self.element_degree = element_degree
        self._dataset: Optional[Dataset] = None

    def load_results(self, source: Any) -> Dataset:
        src = Path(source)
        if not src.exists():
            raise FileNotFoundError(f"FEniCSx/VTK file not found: '{src}'.")

        suf = src.suffix.lower()
        if suf == ".csv":
            arr_map, coord_names, mesh_meta = _parse_vtk_csv(src)
            fmt = "vtk_csv"
        elif suf == ".vtk":
            arr_map, coord_names, mesh_meta = _parse_legacy_ascii_vtk(src)
            fmt = "vtk_ascii"
        elif suf == ".vtu":
            arr_map, coord_names, mesh_meta = _parse_vtu_ascii(src)
            fmt = "vtu_ascii"
        elif suf == ".xdmf":
            arr_map, coord_names, mesh_meta = _parse_xdmf_inline(src)
            fmt = "xdmf_xml"
        else:
            raise ValueError(
                f"Unsupported FEniCSx export format '{suf}'. Expected '.csv', '.vtk', '.vtu', or '.xdmf'."
            )

        field_names = [k for k in arr_map if k not in coord_names]
        solver_meta: Dict[str, Any] = {
            "solver": self.solver_name,
            "element_family": self.element_family,
            "element_degree": self.element_degree,
            "formal_order": (self.element_degree + 1) if self.element_degree is not None else None,
            "source_format": fmt,
        }

        ds = Dataset.from_mapping(
            arr_map,
            name=src.stem,
            coordinates=coord_names,
            fields=field_names,
            units=self.units,
            mesh_metadata=mesh_meta,
            solver_metadata=solver_meta,
            provenance={
                "adapter": "FEniCSxAdapter",
                "format": fmt,
                "source_path": src.name,
                "sha256": _sha256_path(src),
            },
        )
        self._dataset = ds
        return ds

    def _require_loaded(self) -> Dataset:
        if self._dataset is None:
            raise RuntimeError("No FEniCSx results loaded; call load_results(source) first.")
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
