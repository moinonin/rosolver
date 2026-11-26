"""Persistent scientific project workspace and append-only execution ledger (ENG_SPEC.md P1.5)."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import platform
import shutil
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import uuid

import numpy as np
import yaml

from .model import Model
from .results import ResultMixin, StandardResult, standardize_result, to_jsonable


class ProjectError(RuntimeError):
    """Raised on invalid project workspace state or operations."""


class ProvenanceMismatchError(ProjectError):
    """Raised when stored model or dataset hashes fail integrity verification."""


_PROJECT_DIRS: Tuple[str, ...] = (
    "models",
    "data/experimental",
    "data/reference",
    "data/processed",
    "calibration/results",
    "simulations/results",
    "verification/mms",
    "verification/gci",
    "verification/conservation",
    "verification/timestep",
    "validation/results",
    "reports/figures",
    "reports/tables",
    "logs",
)

_RUN_SUBDIRS: Dict[str, str] = {
    "symbolic_derivation": "models",
    "ode_simulation": "simulations/results",
    "pde_simulation": "simulations/results",
    "stefan_simulation": "simulations/results",
    "calibration": "calibration/results",
    "mms": "verification/mms",
    "gci": "verification/gci",
    "conservation": "verification/conservation",
    "timestep": "verification/timestep",
    "validation": "validation/results",
}


def sha256_file(path: Union[str, Path]) -> str:
    """Compute SHA-256 hex digest of a file without loading it entirely into memory."""
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _pkg_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def capture_environment() -> Dict[str, Any]:
    """Capture software and dependency versions without exposing credentials or env vars."""
    deps: Dict[str, str] = {}
    for pkg in ("numpy", "scipy", "sympy", "matplotlib", "pyyaml", "pandas", "h5py"):
        ver = _pkg_version(pkg)
        if ver is not None:
            deps[pkg] = ver
    return {
        "rosolver": _pkg_version("rosolver") or "0.2.0",
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "executable": Path(sys.executable).name,
        "dependencies": deps,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_csv_data(path: Path) -> Dict[str, np.ndarray]:
    cols: Dict[str, List[float]] = {}
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            for k, v in row.items():
                if k is None or v is None or v.strip() == "":
                    continue
                cols.setdefault(k.strip(), []).append(float(v))
    return {k: np.asarray(vals, dtype=float) for k, vals in cols.items()}


@dataclass
class ModelRef(ResultMixin):
    """Registered model artifact inside a Rosolver project."""

    name: str
    version: str
    path: str  # project-relative POSIX path
    content_hash: str
    file_sha256: str
    added_at: str = field(default_factory=_utc_now)


@dataclass
class DatasetRef(ResultMixin):
    """Registered dataset artifact inside a Rosolver project."""

    name: str
    path: str  # project-relative POSIX path
    category: str  # "experimental", "reference", "processed"
    sha256: str
    added_at: str = field(default_factory=_utc_now)
    description: str = ""


@dataclass
class ExecutionRecord(ResultMixin):
    """Immutable append-only execution ledger entry."""

    id: str
    type: str
    model: Optional[str]
    model_version: Optional[str]
    timestamp: str
    status: str
    software: Dict[str, Any]
    configuration: Dict[str, Any]
    seed: Optional[int]
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    notes: str = ""
    reproducibility_note: str = (
        "Captures configuration, software versions, and input hashes; "
        "cross-platform floating-point bit-identity is not guaranteed."
    )


class Project:
    """
    Persistent scientific modelling project with content-hashed assets and an
    append-only execution ledger (ENG_SPEC.md P1.5).
    """

    def __init__(self, root: Path, manifest: Dict[str, Any]) -> None:
        self.root = Path(root).resolve()
        self._manifest = manifest

    @property
    def project_id(self) -> str:
        return str(self._manifest["project_id"])

    @property
    def name(self) -> str:
        return str(self._manifest["name"])

    @property
    def description(self) -> str:
        return str(self._manifest.get("description", ""))

    @property
    def created_at(self) -> str:
        return str(self._manifest["created_at"])

    @property
    def models(self) -> List[ModelRef]:
        return [ModelRef(**m) for m in self._manifest.get("models", [])]

    @property
    def datasets(self) -> List[DatasetRef]:
        return [DatasetRef(**d) for d in self._manifest.get("datasets", [])]

    @property
    def manifest_path(self) -> Path:
        return self.root / "project.yaml"

    @property
    def ledger_path(self) -> Path:
        return self.root / "logs" / "ledger.jsonl"

    @property
    def log_path(self) -> Path:
        return self.root / "logs" / "execution.log"

    # ------------------------------------------------------------------ lifecycle
    @classmethod
    def create(
        cls,
        path: Union[str, Path],
        name: Optional[str] = None,
        description: str = "",
        overwrite: bool = False,
    ) -> "Project":
        root = Path(path).resolve()
        manifest_file = root / "project.yaml"
        if manifest_file.exists() and not overwrite:
            raise ProjectError(f"Project already exists at '{root}'. Use Project.open(...) or overwrite=True.")

        root.mkdir(parents=True, exist_ok=True)
        for rel in _PROJECT_DIRS:
            (root / rel).mkdir(parents=True, exist_ok=True)

        env = capture_environment()
        manifest: Dict[str, Any] = {
            "project_id": uuid.uuid4().hex[:12],
            "name": name or root.name,
            "description": description,
            "rosolver_version": env["rosolver"],
            "created_at": _utc_now(),
            "software": env,
            "models": [],
            "datasets": [],
            "reports": [],
        }
        proj = cls(root, manifest)
        proj._save_manifest()
        proj.ledger_path.touch(exist_ok=True)
        proj._append_log(f"Project '{proj.name}' ({proj.project_id}) initialized.")
        return proj

    @classmethod
    def open(cls, path: Union[str, Path]) -> "Project":
        root = Path(path).resolve()
        manifest_file = root / "project.yaml"
        if not manifest_file.exists():
            raise ProjectError(f"No project.yaml found at '{root}'.")
        try:
            manifest = yaml.safe_load(manifest_file.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ProjectError(f"Invalid project.yaml at '{root}': {exc}") from exc
        if not isinstance(manifest, dict) or "project_id" not in manifest or "name" not in manifest:
            raise ProjectError(f"Malformed project manifest at '{manifest_file}'.")
        for rel in _PROJECT_DIRS:
            (root / rel).mkdir(parents=True, exist_ok=True)
        return cls(root, manifest)

    def _save_manifest(self) -> None:
        self.manifest_path.write_text(
            yaml.safe_dump(self._manifest, sort_keys=False),
            encoding="utf-8",
        )

    def _append_log(self, message: str) -> None:
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"[{_utc_now()}] {message}\n")

    # ------------------------------------------------------------------ models & datasets
    def add_model(self, model: Model, *, filename: Optional[str] = None) -> ModelRef:
        model.validate(strict=True)
        c_hash = model.content_hash()
        existing = self.models
        for m in existing:
            if m.name == model.name and m.version == model.version and m.content_hash == c_hash:
                return m

        if filename is not None:
            rel_path = Path("models") / filename
        else:
            conflict = any(
                m.name == model.name and m.version == model.version and m.content_hash != c_hash
                for m in existing
            )
            fname = (
                f"{model.name}_v{model.version}_{c_hash[:8]}.yaml"
                if conflict
                else f"{model.name}_v{model.version}.yaml"
            )
            rel_path = Path("models") / fname

        dest = self.root / rel_path
        if dest.exists() and any(m.path == rel_path.as_posix() and m.content_hash != c_hash for m in existing):
            rel_path = Path("models") / f"{dest.stem}_{c_hash[:8]}{dest.suffix}"
            dest = self.root / rel_path

        model.save(dest)
        f_sha = sha256_file(dest)
        ref = ModelRef(
            name=model.name,
            version=model.version,
            path=rel_path.as_posix(),
            content_hash=c_hash,
            file_sha256=f_sha,
        )
        self._manifest.setdefault("models", []).append(ref.to_dict())
        self._save_manifest()
        self._append_log(
            f"Added model '{ref.name}' v{ref.version} (content_hash={ref.content_hash[:12]}) -> {ref.path}"
        )
        return ref

    def _resolve_model_ref(
        self,
        name: Optional[str] = None,
        version: Optional[str] = None,
        content_hash: Optional[str] = None,
    ) -> ModelRef:
        candidates = self.models
        if not candidates:
            raise ProjectError(f"Project '{self.name}' has no registered models.")
        if content_hash is not None:
            matches = [m for m in candidates if m.content_hash == content_hash]
            if not matches:
                raise ProjectError(f"No model found with content_hash '{content_hash}'.")
            return matches[-1]
        if name is not None:
            candidates = [m for m in candidates if m.name == name]
            if not candidates:
                raise ProjectError(f"Unknown model '{name}' in project '{self.name}'.")
        if version is not None:
            candidates = [m for m in candidates if m.version == version]
            if not candidates:
                raise ProjectError(f"Model '{name}' version '{version}' not found.")
        return candidates[-1]

    def get_model(
        self,
        name: Optional[str] = None,
        version: Optional[str] = None,
        *,
        content_hash: Optional[str] = None,
        verify_hash: bool = True,
    ) -> Model:
        ref = self._resolve_model_ref(name=name, version=version, content_hash=content_hash)
        path = self.root / ref.path
        if not path.exists():
            raise ProvenanceMismatchError(f"Model file missing: '{ref.path}'.")
        if verify_hash:
            actual_fsha = sha256_file(path)
            if actual_fsha != ref.file_sha256:
                raise ProvenanceMismatchError(
                    f"Model file '{ref.path}' SHA-256 mismatch (expected {ref.file_sha256[:12]}, got {actual_fsha[:12]})."
                )
        model = Model.from_file(path)
        if verify_hash and model.content_hash() != ref.content_hash:
            raise ProvenanceMismatchError(
                f"Model '{ref.name}' content hash mismatch."
            )
        return model

    def add_dataset(
        self,
        source: Union[str, Path],
        name: Optional[str] = None,
        category: str = "experimental",
        copy: bool = True,
        description: str = "",
    ) -> DatasetRef:
        if category not in ("experimental", "reference", "processed"):
            raise ValueError("category must be 'experimental', 'reference', or 'processed'.")
        src = Path(source).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Dataset file not found: '{src}'.")

        src_sha = sha256_file(src)
        ds_name = name or src.stem
        for existing in self.datasets:
            if existing.name == ds_name and existing.sha256 == src_sha and (self.root / existing.path).exists():
                return existing

        if copy:
            rel_path = Path("data") / category / src.name
            dest = self.root / rel_path
            if dest.exists() and sha256_file(dest) != src_sha:
                rel_path = Path("data") / category / f"{src.stem}_{src_sha[:8]}{src.suffix}"
                dest = self.root / rel_path
            if src != dest:
                shutil.copy2(src, dest)
        else:
            try:
                rel_path = src.relative_to(self.root)
            except ValueError as exc:
                raise ProjectError(
                    "copy=False requires the dataset to already reside inside the project directory."
                ) from exc

        ref = DatasetRef(
            name=ds_name,
            path=rel_path.as_posix(),
            category=category,
            sha256=src_sha,
            description=description,
        )
        self._manifest.setdefault("datasets", []).append(ref.to_dict())
        self._save_manifest()
        self._append_log(f"Added dataset '{ref.name}' ({ref.category}, sha256={ref.sha256[:12]}) -> {ref.path}")
        return ref

    def get_dataset_ref(self, name: str) -> DatasetRef:
        matches = [d for d in self.datasets if d.name == name]
        if not matches:
            raise ProjectError(f"Unknown dataset '{name}' in project '{self.name}'.")
        return matches[-1]

    def get_dataset_path(self, name: str, *, verify_hash: bool = True) -> Path:
        ref = self.get_dataset_ref(name)
        path = self.root / ref.path
        if not path.exists():
            raise ProvenanceMismatchError(f"Dataset '{name}' missing at '{ref.path}'.")
        if verify_hash:
            actual = sha256_file(path)
            if actual != ref.sha256:
                raise ProvenanceMismatchError(
                    f"Dataset '{name}' SHA-256 mismatch (expected {ref.sha256[:12]}, got {actual[:12]})."
                )
        return path

    def verify_integrity(self, strict: bool = False) -> Dict[str, Any]:
        missing: List[Dict[str, str]] = []
        modified: List[Dict[str, str]] = []

        for m in self.models:
            p = self.root / m.path
            if not p.exists():
                missing.append({"kind": "model", "name": m.name, "path": m.path})
            elif sha256_file(p) != m.file_sha256:
                modified.append({"kind": "model", "name": m.name, "path": m.path})

        for d in self.datasets:
            p = self.root / d.path
            if not p.exists():
                missing.append({"kind": "dataset", "name": d.name, "path": d.path})
            elif sha256_file(p) != d.sha256:
                modified.append({"kind": "dataset", "name": d.name, "path": d.path})

        ok = not missing and not modified
        if strict and not ok:
            raise ProvenanceMismatchError(
                f"Project integrity check failed: missing={missing}, modified={modified}."
            )
        return {"ok": ok, "missing": missing, "modified": modified}

    # ------------------------------------------------------------------ append-only ledger
    def ledger(self) -> List[ExecutionRecord]:
        if not self.ledger_path.exists():
            return []
        records: List[ExecutionRecord] = []
        for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(ExecutionRecord(**json.loads(line)))
        return records

    def get_run(self, run_id: str) -> ExecutionRecord:
        for rec in self.ledger():
            if rec.id == run_id:
                return rec
        raise ProjectError(f"Execution record '{run_id}' not found in ledger.")

    def _next_run_id(self, run_type: str) -> str:
        seq = len(self.ledger()) + 1
        return f"{run_type}_{seq:03d}"

    def record_run(
        self,
        run_type: str,
        result: Any,
        *,
        model: Optional[Model] = None,
        configuration: Optional[Mapping[str, Any]] = None,
        inputs: Optional[Mapping[str, Any]] = None,
        seed: Optional[int] = None,
        output_subdir: Optional[str] = None,
        notes: str = "",
    ) -> ExecutionRecord:
        run_id = self._next_run_id(run_type)
        std = standardize_result(
            result,
            kind=run_type,
            model_name=model.name if model is not None else None,
            model_version=model.version if model is not None else None,
            run_id=run_id,
        )
        std.metadata.run_id = run_id

        subdir = output_subdir or _RUN_SUBDIRS.get(run_type, "logs")
        out_dir = self.root / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_rel = Path(subdir) / f"{run_id}.json"
        (self.root / out_rel).write_text(std.to_json(), encoding="utf-8")

        rec = ExecutionRecord(
            id=run_id,
            type=run_type,
            model=model.name if model is not None else std.metadata.model_name,
            model_version=model.version if model is not None else std.metadata.model_version,
            timestamp=std.metadata.timestamp,
            status=std.status.value,
            software=capture_environment(),
            configuration=to_jsonable(dict(configuration or std.solver_config)),
            seed=seed,
            inputs=to_jsonable(dict(inputs or {})),
            outputs={"result": out_rel.as_posix()},
            notes=notes,
        )
        # Strictly append-only write to ledger.jsonl
        with self.ledger_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec.to_dict()) + "\n")
        self._append_log(f"Recorded run {rec.id} ({rec.type}, status={rec.status}) -> {out_rel.as_posix()}")
        return rec

    # ------------------------------------------------------------------ tracked workflows
    def run_symbolic(
        self,
        model_name: Optional[str] = None,
        *,
        version: Optional[str] = None,
        content_hash: Optional[str] = None,
        target: Optional[str] = None,
        wrt: Optional[str] = None,
        **kwargs: Any,
    ) -> StandardResult:
        ref = self._resolve_model_ref(name=model_name, version=version, content_hash=content_hash)
        model = self.get_model(content_hash=ref.content_hash, verify_hash=True)
        res = model.analyse_symbolically(target=target, wrt=wrt, **kwargs)
        self.record_run(
            "symbolic_derivation",
            res,
            model=model,
            configuration={"target": target, "wrt": wrt, **kwargs},
            inputs={"model_hash": ref.content_hash, "model_path": ref.path},
        )
        return res

    def run_simulation(
        self,
        model_name: Optional[str] = None,
        *,
        version: Optional[str] = None,
        content_hash: Optional[str] = None,
        t_span: Optional[Tuple[float, float]] = None,
        initial_conditions: Any = None,
        boundary_conditions: Any = None,
        method: Optional[str] = None,
        tolerances: Optional[Mapping[str, float]] = None,
        params: Optional[Mapping[str, Any]] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> StandardResult:
        ref = self._resolve_model_ref(name=model_name, version=version, content_hash=content_hash)
        model = self.get_model(content_hash=ref.content_hash, verify_hash=True)
        res = model.solve(
            t_span=t_span,
            initial_conditions=initial_conditions,
            boundary_conditions=boundary_conditions,
            method=method,
            tolerances=tolerances,
            params=params,
            **kwargs,
        )
        cfg = {
            "t_span": list(t_span) if t_span is not None else None,
            "initial_conditions": to_jsonable(initial_conditions),
            "boundary_conditions": to_jsonable(boundary_conditions),
            "method": method,
            "tolerances": dict(tolerances) if tolerances is not None else None,
            "params": dict(params) if params is not None else None,
            **to_jsonable(kwargs),
        }
        self.record_run(
            res.kind,
            res,
            model=model,
            configuration=cfg,
            inputs={"model_hash": ref.content_hash, "model_path": ref.path},
            seed=seed,
        )
        return res

    def run_calibration(
        self,
        model_name: Optional[str] = None,
        *,
        version: Optional[str] = None,
        content_hash: Optional[str] = None,
        dataset: Optional[str] = None,
        data: Optional[Mapping[str, Any]] = None,
        target: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> StandardResult:
        if dataset is None and data is None:
            raise ProjectError("Provide either `dataset` name or in-memory `data` mapping.")
        ref = self._resolve_model_ref(name=model_name, version=version, content_hash=content_hash)
        model = self.get_model(content_hash=ref.content_hash, verify_hash=True)

        inputs: Dict[str, Any] = {"model_hash": ref.content_hash, "model_path": ref.path}
        if dataset is not None:
            ds_ref = self.get_dataset_ref(dataset)
            ds_path = self.get_dataset_path(dataset, verify_hash=True)
            cal_data: Mapping[str, Any] = _load_csv_data(ds_path)
            inputs["dataset"] = ds_ref.name
            inputs["dataset_path"] = ds_ref.path
            inputs["dataset_sha256"] = ds_ref.sha256
        else:
            assert data is not None
            cal_data = data
            serial_data = to_jsonable(dict(data))
            snap_bytes = json.dumps(serial_data, sort_keys=True).encode("utf-8")
            snap_sha = hashlib.sha256(snap_bytes).hexdigest()
            snap_rel = Path("data/processed") / f"calibration_input_{snap_sha[:12]}.json"
            (self.root / snap_rel).write_bytes(snap_bytes)
            inputs["data_snapshot"] = snap_rel.as_posix()
            inputs["data_sha256"] = snap_sha

        res = model.calibrate(cal_data, target=target, **kwargs)
        cfg = {
            "dataset": dataset,
            "target": target,
            **to_jsonable(kwargs),
        }
        self.record_run(
            "calibration",
            res,
            model=model,
            configuration=cfg,
            inputs=inputs,
            seed=seed,
        )
        return res

    def replay_run(self, run_id: str) -> StandardResult:
        """Re-execute a prior recorded run from its stored model hash, dataset hash, and configuration."""
        rec = self.get_run(run_id)
        self.verify_integrity(strict=True)
        model_hash = rec.inputs.get("model_hash")
        if not model_hash:
            raise ProjectError(f"Run '{run_id}' does not record a model_hash for replay.")
        model = self.get_model(content_hash=model_hash, verify_hash=True)
        cfg = dict(rec.configuration)

        if rec.type == "symbolic_derivation":
            res = model.analyse_symbolically(**cfg)
        elif rec.type in ("ode_simulation", "pde_simulation", "stefan_simulation"):
            t_span = tuple(cfg.pop("t_span")) if cfg.get("t_span") is not None else None
            if cfg.get("domain") is not None:
                cfg["domain"] = tuple(cfg["domain"])
            res = model.solve(t_span=t_span, **cfg)
        elif rec.type == "calibration":
            ds_name = cfg.pop("dataset", None)
            if ds_name is not None:
                ds_path = self.get_dataset_path(ds_name, verify_hash=True)
                cal_data: Mapping[str, Any] = _load_csv_data(ds_path)
            elif "data_snapshot" in rec.inputs:
                snap_path = self.root / rec.inputs["data_snapshot"]
                if not snap_path.exists():
                    raise ProvenanceMismatchError(f"Snapshot '{rec.inputs['data_snapshot']}' missing.")
                cal_data = json.loads(snap_path.read_text(encoding="utf-8"))
            else:
                raise ProjectError(f"Calibration run '{run_id}' has no dataset or data_snapshot to replay.")
            res = model.calibrate(cal_data, **cfg)
        else:
            raise ProjectError(f"Replay is not supported for run type '{rec.type}'.")

        self.record_run(
            res.kind,
            res,
            model=model,
            configuration=rec.configuration,
            inputs={**rec.inputs, "replayed_from": run_id},
            seed=rec.seed,
            notes=f"Replayed from {run_id}",
        )
        return res

    def info(self) -> Dict[str, Any]:
        integrity = self.verify_integrity(strict=False)
        runs = self.ledger()
        return {
            "project_id": self.project_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "rosolver_version": self._manifest.get("rosolver_version"),
            "n_models": len(self.models),
            "n_datasets": len(self.datasets),
            "n_runs": len(runs),
            "latest_run": runs[-1].id if runs else None,
            "integrity_ok": integrity["ok"],
        }

    def report(
        self,
        output: Optional[Union[str, Path]] = None,
        *,
        title: Optional[str] = None,
        report_type: str = "integrated_project",
        run_ids: Optional[Sequence[str]] = None,
        figures: Optional[Sequence[Union[str, Path, Mapping[str, str]]]] = None,
        description: str = "",
        also_manifest: Union[bool, str, Path] = False,
    ) -> Any:
        """
        Generate, save, and register a `ScientificReport` for this project.

        Relative `output` paths are resolved relative to the project root.
        """
        from .reporting import ReportGenerator

        rep = ReportGenerator.from_project(
            self,
            title=title,
            report_type=report_type,
            description=description,
            run_ids=run_ids,
            figures=figures,
        )
        if output is None:
            out_path = self.root / "reports" / "final_report.md"
        else:
            p = Path(output)
            out_path = p if p.is_absolute() else (self.root / p)

        rep.save(out_path, also_manifest=also_manifest)
        try:
            rel_out = out_path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            rel_out = out_path.name
        reports_list = list(self._manifest.get("reports") or [])
        reports_list.append(
            {
                "path": rel_out,
                "report_type": rep.report_type,
                "status": rep.status.value,
                "generated_at": rep.generated_at,
            }
        )
        self._manifest["reports"] = reports_list
        self._save_manifest()
        return rep

    def summary(self) -> str:
        inf = self.info()
        return (
            f"=== Rosolver Project: {inf['name']} ({inf['project_id']}) ===\n"
            f"  Created: {inf['created_at']} | Rosolver: {inf['rosolver_version']}\n"
            f"  Models: {inf['n_models']} | Datasets: {inf['n_datasets']} | Ledger runs: {inf['n_runs']}\n"
            f"  Integrity OK: {inf['integrity_ok']}"
        )
