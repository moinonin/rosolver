"""
Command-line interface for Rosolver (ENG_SPEC.md P3.2).

Supports both:
  1. Workflow project subcommands:
       rosolver init <dir>
       rosolver validate <project_or_model>
       rosolver model analyse <project>
       rosolver calibrate <project>
       rosolver simulate <project>
       rosolver verify <project>
       rosolver validate-results <project>
       rosolver report <project>
       rosolver info <project>
  2. Backwards-compatible legacy flat flags (--rel, --gci, --sweep, --plot, ...).

Exit codes (workflow subcommands):
  0  Completed and passed (including WARNING status unless rejected)
  1  Computational / V&V check FAILED, INCONCLUSIVE, or runtime failure
  2  Validation, missing artifact, or SHA-256 provenance integrity failure
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .data import Dataset
from .integrations import VerificationStudy
from .model import Model, ModelValidationError
from .numerical import NumericalSolver
from .project import Project, ProjectError, ProvenanceMismatchError
from .results import VerificationStatus, standardize_result, to_jsonable
from .symbolic import SymbolicSolver
from .system import System
from .validation import ValidationStudy

EXIT_OK = 0
EXIT_CHECK_FAILED = 1
EXIT_VALIDATION_ERROR = 2

WORKFLOW_COMMANDS = frozenset(
    {
        "init",
        "validate",
        "model",
        "calibrate",
        "simulate",
        "verify",
        "validate-results",
        "report",
        "info",
        "mcp",
        "claim",
    }
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rosolver",
        description="Formulate symbolic equations, solve models, verify/validate studies, and generate scientific reports.",
        epilog=(
            "Workflow subcommands: init, validate, model analyse, calibrate, simulate, "
            "verify, validate-results, report, info (run 'rosolver <cmd> --help' for options)."
        ),
    )
    p.add_argument(
        "--rel",
        action="append",
        default=[],
        help="Qualitative dependency relation, e.g. 'g = fn(a, b, c)' or 'b = fn(f; grav)'. Repeatable.",
    )
    p.add_argument(
        "--dim",
        action="append",
        default=[],
        help="Dimension assignment 'var=DIM', e.g. 'g=M*L^2/T^3'. Repeatable.",
    )
    p.add_argument(
        "--const",
        action="append",
        default=[],
        help="Mark identifier as governing constant parameter. Repeatable.",
    )
    p.add_argument(
        "--rate",
        action="append",
        default=[],
        help="Rate identity 'target,wrt,equals' meaning d(target)/d(wrt)=equals (e.g. 's,t,v'). Repeatable.",
    )
    p.add_argument("--target", default=None, help="Target variable to formulate.")
    p.add_argument("--wrt", default=None, help="Independent variable to express target with respect to.")
    p.add_argument("--zero-initials", action="store_true", help="Omit integration baseline offsets (x_0 = 0).")
    p.add_argument("--unit-constants", action="store_true", help="Set dimensionless scaling constants kappa = 1.")
    p.add_argument("--latex", action="store_true", help="Output derivation trace in LaTeX.")
    p.add_argument("--csv", default=None, help="Optional CSV path to fit & rank candidate equations numerically.")
    p.add_argument(
        "--sweep",
        default=None,
        help="Simulate the final equation over 'var=start:stop:n' (e.g. f=0:25:200).",
    )
    p.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        help="Fixed value 'name=value' for simulation (constants, kappas). Repeatable.",
    )
    p.add_argument(
        "--family",
        action="append",
        default=[],
        help="Curve family 'name=v1,v2,v3' for simulation. Repeatable.",
    )
    p.add_argument("--candidate", default=None, help="Candidate name to simulate (default: primary).")
    p.add_argument("--plot", default=None, help="Output image path (png/pdf/svg) for the simulation plot.")
    p.add_argument("--sim-csv", default=None, help="Also write simulated curves to this CSV path.")
    p.add_argument(
        "--gci",
        default=None,
        help="Standalone Roache/Celik GCI check (inspired by ASME V&V 20) on external solver outputs: "
        "'h1,h2,h3:phi1,phi2,phi3' (e.g. '0.01,0.02,0.04:1.957,1.948,1.912').",
    )
    p.add_argument("--formal-order", type=float, default=None, help="Optional formal order p for --gci.")
    p.add_argument("--gci-plot", default=None, help="Optional image path to plot --gci convergence band.")
    return p


def _load_csv_columns(path: Path) -> Dict[str, np.ndarray]:
    from .data import load_tabular_csv

    raw = load_tabular_csv(path, missing_policy="raise", allow_string_columns=True)
    return {k: v for k, v in raw.items() if isinstance(v, np.ndarray)}


def _build_workflow_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rosolver",
        description="Rosolver scientific project workflow CLI.",
    )
    subs = p.add_subparsers(dest="command", required=True)

    # init
    p_init = subs.add_parser("init", help="Initialize a structured Rosolver project directory.")
    p_init.add_argument("dir", help="Target directory for the new project.")
    p_init.add_argument("--name", default=None, help="Human-readable project name.")
    p_init.add_argument("--description", default="", help="Project description.")
    p_init.add_argument("--model", default=None, help="Optional initial model YAML/JSON to register.")
    p_init.add_argument("--json", action="store_true", help="Emit structured JSON output.")

    # validate
    p_val = subs.add_parser("validate", help="Validate project structure, SHA-256 integrity, and model specs.")
    p_val.add_argument("project", help="Project directory or standalone model YAML/JSON file.")
    p_val.add_argument("--strict", action="store_true", help="Treat warnings as validation failures.")
    p_val.add_argument("--json", action="store_true", help="Emit structured JSON output.")

    # model (analyse)
    p_mod = subs.add_parser("model", help="Model operations (analyse).")
    mod_subs = p_mod.add_subparsers(dest="model_command", required=True)
    p_ma = mod_subs.add_parser("analyse", help="Run symbolic analysis on a project's model.")
    p_ma.add_argument("project", help="Project directory.")
    p_ma.add_argument("--model", default=None, help="Registered model name (default: latest).")
    p_ma.add_argument("--target", default=None, help="Target variable.")
    p_ma.add_argument("--wrt", default=None, help="Independent variable.")
    p_ma.add_argument("--zero-initials", action="store_true", help="Omit integration constants.")
    p_ma.add_argument("--unit-constants", action="store_true", help="Set dimensionless kappas to 1.")
    p_ma.add_argument("--no-record", action="store_true", help="Do not record execution in project ledger.")
    p_ma.add_argument("--json", action="store_true", help="Emit structured JSON output.")

    # calibrate
    p_cal = subs.add_parser("calibrate", help="Calibrate model parameters against observational data.")
    p_cal.add_argument("project", help="Project directory.")
    p_cal.add_argument("--model", default=None, help="Registered model name (default: latest).")
    p_cal.add_argument("--data", required=True, help="CSV/JSON/YAML file path or registered dataset name.")
    p_cal.add_argument("--target", default=None, help="Target column to fit.")
    p_cal.add_argument("--uncertainty-col", default=None, help="Column containing 1-sigma measurement uncertainties.")
    p_cal.add_argument("--weights-col", default=None, help="Optional observation weights column.")
    p_cal.add_argument("--loss", default="linear", choices=("linear", "soft_l1", "huber", "cauchy", "arctan"))
    p_cal.add_argument("--no-record", action="store_true", help="Do not record execution in project ledger.")
    p_cal.add_argument("--json", action="store_true", help="Emit structured JSON output.")

    # simulate
    p_sim = subs.add_parser("simulate", help="Solve the project's governing model.")
    p_sim.add_argument("project", help="Project directory.")
    p_sim.add_argument("--model", default=None, help="Registered model name (default: latest).")
    p_sim.add_argument("--t0", type=float, default=0.0, help="Start time.")
    p_sim.add_argument("--t-end", type=float, default=1.0, help="End time.")
    p_sim.add_argument("--n-cells", type=int, default=None, help="Spatial cell count for PDEs.")
    p_sim.add_argument("--n-out", type=int, default=None, help="Number of output time points.")
    p_sim.add_argument("--method", default=None, help="Time integration method override.")
    p_sim.add_argument("--domain", default=None, help="Spatial domain override 'lo:hi'.")
    p_sim.add_argument("--param", action="append", default=[], help="Parameter override 'name=value'. Repeatable.")
    p_sim.add_argument("--no-record", action="store_true", help="Do not record execution in project ledger.")
    p_sim.add_argument("--json", action="store_true", help="Emit structured JSON output.")

    # verify
    p_ver = subs.add_parser("verify", help="Run code or solution verification (GCI, MMS, conservation).")
    p_ver.add_argument("project", help="Project directory.")
    p_ver.add_argument(
        "--method",
        default="conservation",
        choices=("gci", "mms", "conservation", "all"),
        help="Verification method to run.",
    )
    p_ver.add_argument("--model", default=None, help="Registered model name.")
    p_ver.add_argument("--gci", default=None, help="Direct GCI values 'h1,h2,h3:phi1,phi2,phi3'.")
    p_ver.add_argument("--csv", default=None, help="External CSV with spacing and QoI columns for GCI.")
    p_ver.add_argument("--spacing-col", default="h", help="Spacing column name in --csv (default: h).")
    p_ver.add_argument("--qoi-col", default="qoi", help="QoI column name in --csv (default: qoi).")
    p_ver.add_argument("--formal-order", type=float, default=None, help="Formal convergence order p.")
    p_ver.add_argument("--tol", type=float, default=1e-10, help="Conservation tolerance.")
    p_ver.add_argument("--t-end", type=float, default=0.5, help="Simulation end time for PDE verification.")
    p_ver.add_argument("--cells", default="16,32,64", help="Comma-separated cell counts for MMS.")
    p_ver.add_argument("--exact", default=None, help="Manufactured exact solution expression for MMS.")
    p_ver.add_argument("--json", action="store_true", help="Emit structured JSON output.")

    # validate-results
    p_vr = subs.add_parser(
        "validate-results",
        help="Compare simulation predictions against experimental data with uncertainty bands.",
    )
    p_vr.add_argument("project", help="Project directory.")
    p_vr.add_argument("--sim-csv", default=None, help="Simulation CSV path.")
    p_vr.add_argument("--run-id", default=None, help="Project simulation run_id to extract predictions from.")
    p_vr.add_argument("--exp-csv", default=None, help="Experimental CSV path.")
    p_vr.add_argument("--exp-dataset", default=None, help="Registered experimental dataset name in project.")
    p_vr.add_argument("--field", required=True, help="Field / quantity name to compare.")
    p_vr.add_argument("--coord", default=None, help="Coordinate column name (e.g. 'x' or 't').")
    p_vr.add_argument("--u-num", default=None, help="Simulation numerical uncertainty column or float.")
    p_vr.add_argument("--u-exp", default=None, help="Experimental uncertainty column or float.")
    p_vr.add_argument("--u-input", default=None, help="Input uncertainty column or float.")
    p_vr.add_argument("--coverage-factor", type=float, default=1.0, help="Validation coverage factor k.")
    p_vr.add_argument("--tol", type=float, default=None, help="Absolute error tolerance when no uncertainty given.")
    p_vr.add_argument("--plot", default=None, help="Optional path to save comparison plot.")
    p_vr.add_argument("--study-name", default=None, help="Validation study label.")
    p_vr.add_argument("--json", action="store_true", help="Emit structured JSON output.")

    # report
    p_rep = subs.add_parser("report", help="Generate reproducible scientific Markdown/JSON/YAML/LaTeX/PDF report.")
    p_rep.add_argument("project", help="Project directory.")
    p_rep.add_argument("--output", default="reports/final_report.md", help="Report output path (.md, .json, .yaml, .tex, or .pdf).")
    p_rep.add_argument("--title", default=None, help="Report title.")
    p_rep.add_argument("--type", dest="report_type", default="integrated_project", help="Report type.")
    p_rep.add_argument("--description", default="", help="Report description.")
    p_rep.add_argument("--run-id", action="append", default=None, help="Filter to specific run_id(s).")
    p_rep.add_argument("--figure", action="append", default=None, help="Figure path to include. Repeatable.")
    p_rep.add_argument("--manifest", action="store_true", help="Also emit sidecar .manifest.json.")
    p_rep.add_argument("--json", action="store_true", help="Emit structured JSON report metadata.")

    # info
    p_info = subs.add_parser("info", help="Display project metadata, models, datasets, and ledger summary.")
    p_info.add_argument("project", help="Project directory.")
    p_info.add_argument("--json", action="store_true", help="Emit structured JSON output.")

    # mcp
    p_mcp = subs.add_parser("mcp", help="Start the Rosolver Model Context Protocol (MCP) server over stdio.")
    p_mcp.add_argument("--list-tools", action="store_true", help="Print registered MCP tool schemas as JSON and exit.")

    # claim (Sprint 10: init, audit, questions, respond)
    p_claim = subs.add_parser("claim", help="Deterministic scientific claim verification and peer-review rebuttal workflow.")
    claim_subs = p_claim.add_subparsers(dest="claim_action", required=True)

    pc_init = claim_subs.add_parser("init", help="Register a ClaimManifest YAML/JSON file into a project.")
    pc_init.add_argument("project", help="Project directory.")
    pc_init.add_argument("--manifest", required=True, help="Path to ClaimManifest YAML/JSON file.")
    pc_init.add_argument("--json", action="store_true", help="Emit JSON output.")

    pc_audit = claim_subs.add_parser("audit", help="Run a deterministic claim-verification audit round.")
    pc_audit.add_argument("project", help="Project directory.")
    pc_audit.add_argument("--manifest", default=None, help="Optional ClaimManifest path to register before auditing.")
    pc_audit.add_argument("--output", default=None, help="Optional referee report output path (e.g. reports/referee_r1.md).")
    pc_audit.add_argument("--json", action="store_true", help="Emit JSON output.")

    pc_q = claim_subs.add_parser("questions", help="List open deterministic referee clarification requests.")
    pc_q.add_argument("project", help="Project directory.")
    pc_q.add_argument("--severity", choices=["all", "major"], default="all", help="Filter by severity.")
    pc_q.add_argument("--json", action="store_true", help="Emit JSON output.")

    pc_resp = claim_subs.add_parser("respond", help="Submit author rebuttal YAML/JSON and re-verify in Round k+1.")
    pc_resp.add_argument("project", help="Project directory.")
    pc_resp.add_argument("--rebuttal", required=True, help="Path to RebuttalSubmission YAML/JSON file.")
    pc_resp.add_argument("--output", default=None, help="Optional Round k+1 referee report output path.")
    pc_resp.add_argument("--json", action="store_true", help="Emit JSON output.")

    return p


def _emit(payload: Dict[str, Any], human_text: str, as_json: bool) -> None:
    if as_json:
        print(json.dumps(to_jsonable(payload), indent=2))
    else:
        print(human_text)


def _exit_for_status(status: VerificationStatus) -> int:
    if status in (VerificationStatus.FAILED, VerificationStatus.INCONCLUSIVE):
        return EXIT_CHECK_FAILED
    return EXIT_OK


def _parse_kv_params(pairs: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Expected 'name=value', got '{item}'.")
        k, v = item.split("=", 1)
        out[k.strip()] = float(v)
    return out


def _maybe_float_or_col(val: Optional[str], columns: Dict[str, Any]) -> Any:
    if val is None:
        return None
    try:
        return float(val)
    except ValueError:
        if val in columns:
            return val
        raise FileNotFoundError(f"Uncertainty '{val}' is neither numeric nor a CSV column.")


def _load_sim_dataset_from_run(proj: Project, run_id: Optional[str], field: str, coord: Optional[str]) -> Dataset:
    runs = proj.ledger()
    target_rec = None
    if run_id is not None:
        target_rec = proj.get_run(run_id)
    else:
        for r in reversed(runs):
            if r.type in ("pde_simulation", "ode_simulation", "stefan_simulation", "simulation"):
                target_rec = r
                break
    if target_rec is None:
        raise ProjectError("No simulation run found in project ledger (provide --sim-csv or --run-id).")

    rel_res = target_rec.outputs.get("result")
    if not rel_res:
        raise ProjectError(f"Run '{target_rec.id}' has no stored result file.")
    payload = json.loads((proj.root / rel_res).read_text(encoding="utf-8"))
    raw = payload.get("result") or {}
    meta_units = (payload.get("metadata") or {}).get("units") or {}

    if "x" in raw and isinstance(raw.get("U"), dict) and field in raw["U"]:
        arr_t = np.asarray(raw["U"][field], dtype=float)
        vals = arr_t[-1] if arr_t.ndim == 2 else arr_t
        c_name = coord or "x"
        coords = {c_name: np.asarray(raw["x"], dtype=float)}
        return Dataset(
            name=f"{target_rec.id}_{field}",
            coordinates=coords,
            fields={field: vals},
            units={c_name: meta_units.get(c_name, "m"), field: meta_units.get(field, "1")},
            provenance={"run_id": target_rec.id},
        )
    if "t" in raw and isinstance(raw.get("states"), list) and field in raw["states"] and "y" in raw:
        c_name = coord or "t"
        idx = raw["states"].index(field)
        y_arr = np.asarray(raw["y"], dtype=float)
        return Dataset(
            name=f"{target_rec.id}_{field}",
            coordinates={c_name: np.asarray(raw["t"], dtype=float)},
            fields={field: y_arr[:, idx]},
            units={c_name: meta_units.get(c_name, "s"), field: meta_units.get(field, "1")},
            provenance={"run_id": target_rec.id},
        )
    if "t" in raw and field in ("delta", "T_s", "film", "mdot_ice", "mdot_out") and raw.get(field) is not None:
        c_name = coord or "t"
        return Dataset(
            name=f"{target_rec.id}_{field}",
            coordinates={c_name: np.asarray(raw["t"], dtype=float)},
            fields={field: np.asarray(raw[field], dtype=float)},
            units={c_name: meta_units.get(c_name, "s"), field: meta_units.get(field, "m" if field in ("delta", "film") else "1")},
            provenance={"run_id": target_rec.id},
        )
    raise ProjectError(f"Could not extract field '{field}' from run '{target_rec.id}'.")


def _run_workflow_cli(argv: Sequence[str]) -> int:
    parser = _build_workflow_parser()
    args = parser.parse_args(argv)
    as_json = bool(getattr(args, "json", False))

    try:
        if args.command == "init":
            proj = Project.create(args.dir, name=args.name, description=args.description)
            added_model = None
            if args.model:
                m = Model.from_file(args.model)
                mref = proj.add_model(m, filename=Path(args.model).name)
                added_model = {
                    "name": mref.name,
                    "version": mref.version,
                    "content_hash": mref.content_hash,
                    "file_sha256": mref.file_sha256,
                }
            info = proj.info()
            if added_model:
                info["added_model"] = added_model
            _emit({"status": "ok", "project": info}, proj.summary(), as_json)
            return EXIT_OK

        if args.command == "validate":
            target = Path(args.project)
            if target.is_file():
                try:
                    model = Model.from_file(target)
                    errs = model.validate(strict=False)
                    if args.strict and not model.equations and not model.dependencies:
                        errs.append("Strict mode: model declares neither equations nor dependencies.")
                    valid = len(errs) == 0
                    m_name = model.name
                    m_ver = model.version
                except ModelValidationError as exc:
                    valid = False
                    errs = [s.strip() for s in str(exc).split(";") if s.strip()]
                    m_name = target.stem
                    m_ver = "unknown"
                payload = {"status": "valid" if valid else "invalid", "model": m_name, "errors": errs}
                human = (
                    f"Model '{m_name}' (v{m_ver}): VALID"
                    if valid
                    else f"Model '{m_name}': INVALID - {'; '.join(errs)}"
                )
                _emit(payload, human, as_json)
                return EXIT_OK if valid else EXIT_VALIDATION_ERROR

            proj = Project.open(target)
            integrity = proj.verify_integrity(strict=False)
            model_errors: Dict[str, List[str]] = {}
            for mref in proj.models:
                try:
                    m = proj.get_model(name=mref.name, version=mref.version, verify_hash=False)
                    errs = m.validate(strict=False)
                    if args.strict and not m.equations and not m.dependencies:
                        errs.append("Strict mode: model declares neither equations nor dependencies.")
                    if errs:
                        model_errors[mref.name] = errs
                except Exception as exc:
                    model_errors[mref.name] = [str(exc)]
            strict_fail = bool(args.strict and not proj.models)
            if strict_fail:
                model_errors["__project__"] = ["Strict mode: project has zero registered models."]
            ok = bool(integrity["ok"]) and not model_errors
            payload = {
                "status": "valid" if ok else "invalid",
                "integrity": integrity,
                "model_errors": model_errors,
            }
            human = (
                f"Project '{proj.name}': VALID ({len(proj.models)} model(s), {len(proj.datasets)} dataset(s))"
                if ok
                else f"Project '{proj.name}': INVALID | integrity={integrity} | model_errors={model_errors}"
            )
            _emit(payload, human, as_json)
            return EXIT_OK if ok else EXIT_VALIDATION_ERROR

        if args.command == "info":
            proj = Project.open(args.project)
            _emit(proj.info(), proj.summary(), as_json)
            return EXIT_OK

        if args.command == "model" and args.model_command == "analyse":
            proj = Project.open(args.project)
            if args.no_record:
                m = proj.get_model(name=args.model)
                sr = m.analyse_symbolically(
                    target=args.target,
                    wrt=args.wrt,
                    zero_initials=args.zero_initials,
                    unit_constants=args.unit_constants,
                )
            else:
                sr = proj.run_symbolic(
                    model_name=args.model,
                    target=args.target,
                    wrt=args.wrt,
                    zero_initials=args.zero_initials,
                    unit_constants=args.unit_constants,
                )
            _emit(sr.to_dict(), sr.raw.summary(), as_json)
            return _exit_for_status(sr.status)

        if args.command == "calibrate":
            proj = Project.open(args.project)
            data_path = Path(args.data)
            if not data_path.exists():
                data_path = proj.get_dataset_path(args.data, verify_hash=True)
            ext = data_path.suffix.lower()
            if ext == ".json":
                ds = Dataset.from_json(data_path)
            elif ext in (".yaml", ".yml"):
                ds = Dataset.from_yaml(data_path)
            else:
                # Generic column load retaining all numeric columns cleanly
                raw_cols = _load_csv_columns(data_path)
                ds = Dataset(name=data_path.stem, fields=raw_cols)

            col_map: Dict[str, Any] = dict(ds.coordinates)
            col_map.update(ds.fields)
            fit_kwargs: Dict[str, Any] = {"loss": args.loss}
            if args.uncertainty_col:
                fit_kwargs["sigma_y"] = col_map[args.uncertainty_col]
            if args.weights_col:
                fit_kwargs["weights"] = col_map[args.weights_col]

            if args.no_record:
                m = proj.get_model(name=args.model)
                sr = m.calibrate(col_map, target=args.target, **fit_kwargs)
            else:
                sr = proj.run_calibration(
                    data=col_map,
                    model_name=args.model,
                    target=args.target,
                    **fit_kwargs,
                )
            _emit(sr.to_dict(), sr.raw.summary(), as_json)
            return _exit_for_status(sr.status)

        if args.command == "simulate":
            proj = Project.open(args.project)
            params = _parse_kv_params(args.param) if args.param else None
            dom: Optional[Tuple[float, float]] = None
            if args.domain:
                lo_s, hi_s = args.domain.split(":", 1)
                dom = (float(lo_s), float(hi_s))
            kwargs: Dict[str, Any] = {"t_span": (args.t0, args.t_end)}
            if args.n_cells is not None:
                kwargs["n_cells"] = args.n_cells
            if args.n_out is not None:
                kwargs["n_out"] = args.n_out
            if args.method is not None:
                kwargs["method"] = args.method
            if dom is not None:
                kwargs["domain"] = dom
            if params:
                kwargs["params"] = params

            if args.no_record:
                m = proj.get_model(name=args.model)
                sr = m.solve(**kwargs)
            else:
                sr = proj.run_simulation(model_name=args.model, **kwargs)
            human = sr.raw.summary() if hasattr(sr.raw, "summary") else f"Simulation {sr.kind}: {sr.status.value}"
            _emit(sr.to_dict(), human, as_json)
            return _exit_for_status(sr.status)

        if args.command == "verify":
            from .verify import audit_pde_conservation, gci_from_values, mms_study

            proj = Project.open(args.project)
            results: List[Tuple[str, Any]] = []

            if args.gci:
                h_part, q_part = args.gci.split(":", 1)
                hs = [float(x) for x in h_part.split(",")]
                qs = [float(x) for x in q_part.split(",")]
                rep = gci_from_values(hs, qs, formal_order=args.formal_order)
                sr = standardize_result(rep, kind="gci")
                proj.record_run("gci", sr, configuration={"formal_order": args.formal_order})
                results.append(("gci", sr))
            elif args.csv:
                vs = VerificationStudy.from_csv(
                    args.csv,
                    spacing_col=args.spacing_col,
                    qoi_col=args.qoi_col,
                )
                rep = vs.gci(formal_order=args.formal_order)
                sr = standardize_result(rep, kind="gci")
                proj.record_run("gci", sr, configuration={"csv": Path(args.csv).name})
                results.append(("gci", sr))
            else:
                m = proj.get_model(name=args.model)
                methods = ("conservation", "mms") if args.method == "all" else (args.method,)
                if "gci" in methods:
                    # Grid convergence study on PDE model using 3 grids
                    law = m.to_conservation_law()
                    from .verify import grid_convergence_study

                    dom = m.numerics.domain or (0.0, 1.0)
                    ic = m._extract_initial_conditions(None, law.fields)
                    bc = m._extract_boundary_conditions(None, law.fields)
                    cells = [int(c.strip()) for c in args.cells.split(",")]
                    fld0 = law.fields[0]
                    rep = grid_convergence_study(
                        lambda n: float(
                            np.mean(
                                law.solve(
                                    domain=dom,
                                    n_cells=n,
                                    t_end=args.t_end,
                                    initial=ic if len(law.fields) > 1 else ic[fld0],
                                    bc=bc,
                                    params=m.parameter_values(),
                                    scheme=m.numerics.time_integrator,
                                    reconstruction=m.numerics.reconstruction,
                                ).final(fld0)
                            )
                        ),
                        cells=cells,
                        domain_length=float(dom[1] - dom[0]),
                        formal_order=args.formal_order or 2.0,
                    )
                    sr = standardize_result(rep, kind="gci", model_name=m.name, model_version=m.version)
                    proj.record_run("gci", sr, model=m, configuration={"cells": cells, "t_end": args.t_end})
                    results.append(("gci", sr))

                if "conservation" in methods:
                    sim_sr = m.solve(t_span=(0.0, args.t_end))
                    rep = audit_pde_conservation(sim_sr.raw, tol=args.tol)
                    sr = standardize_result(rep, kind="conservation", model_name=m.name, model_version=m.version)
                    proj.record_run("conservation", sr, model=m, configuration={"tol": args.tol, "t_end": args.t_end})
                    results.append(("conservation", sr))

                if "mms" in methods:
                    law = m.to_conservation_law()
                    fld0 = law.fields[0]
                    x_s = law.x.name
                    t_s = law.t.name
                    exact_expr = args.exact or f"exp(-{t_s}) * sin(pi * {x_s})"
                    dom = m.numerics.domain or (0.0, 1.0)
                    cells = [int(c.strip()) for c in args.cells.split(",")]
                    rep = mms_study(
                        law,
                        exact={fld0: exact_expr},
                        domain=dom,
                        t_end=args.t_end,
                        cells=cells,
                        params=m.parameter_values(),
                        bc_kind="dirichlet" if "sin(pi" in exact_expr else "periodic",
                    )
                    sr = standardize_result(rep, kind="mms", model_name=m.name, model_version=m.version)
                    proj.record_run("mms", sr, model=m, configuration={"cells": cells, "t_end": args.t_end})
                    results.append(("mms", sr))

            worst = EXIT_OK
            combined_payload: Dict[str, Any] = {}
            human_parts: List[str] = []
            for label, sr in results:
                combined_payload[label] = sr.to_dict()
                human_parts.append(sr.raw.summary())
                if _exit_for_status(sr.status) != EXIT_OK:
                    worst = EXIT_CHECK_FAILED
            if len(results) == 1:
                combined_payload = results[0][1].to_dict()
            _emit(combined_payload, "\n\n".join(human_parts), as_json)
            return worst

        if args.command == "validate-results":
            from .simulate import plot_validation

            proj = Project.open(args.project)
            if args.exp_csv:
                exp_cols = _load_csv_columns(Path(args.exp_csv))
                u_exp_arg = _maybe_float_or_col(args.u_exp, exp_cols)
                coord_cols = [args.coord] if args.coord and args.coord in exp_cols else None
                unc_map = {args.field: u_exp_arg} if isinstance(u_exp_arg, str) else None
                exp_ds = Dataset.from_csv(
                    args.exp_csv,
                    coordinates=coord_cols,
                    fields=[args.field],
                    uncertainties=unc_map,
                )
            elif args.exp_dataset:
                p_exp = proj.get_dataset_path(args.exp_dataset, verify_hash=True)
                exp_cols = _load_csv_columns(p_exp)
                u_exp_arg = _maybe_float_or_col(args.u_exp, exp_cols)
                coord_cols = [args.coord] if args.coord and args.coord in exp_cols else None
                unc_map = {args.field: u_exp_arg} if isinstance(u_exp_arg, str) else None
                exp_ds = Dataset.from_csv(
                    p_exp,
                    coordinates=coord_cols,
                    fields=[args.field],
                    uncertainties=unc_map,
                )
            else:
                raise ModelValidationError("Provide --exp-csv or --exp-dataset for validate-results.")

            if args.sim_csv:
                sim_cols = _load_csv_columns(Path(args.sim_csv))
                u_num_arg = _maybe_float_or_col(args.u_num, sim_cols)
                coord_cols = [args.coord] if args.coord and args.coord in sim_cols else None
                unc_map = {args.field: u_num_arg} if isinstance(u_num_arg, str) else None
                sim_ds = Dataset.from_csv(
                    args.sim_csv,
                    coordinates=coord_cols,
                    fields=[args.field],
                    uncertainties=unc_map,
                )
            else:
                sim_ds = _load_sim_dataset_from_run(proj, args.run_id, args.field, args.coord)
                u_num_arg = float(args.u_num) if args.u_num is not None else None

            u_exp_val = None
            if args.u_exp is not None:
                try:
                    u_exp_val = float(args.u_exp)
                except ValueError:
                    u_exp_val = None
            if u_exp_val is not None and exp_ds.uncertainty_for(args.field) is None:
                exp_ds = Dataset(
                    name=exp_ds.name,
                    coordinates=exp_ds.coordinates,
                    fields=exp_ds.fields,
                    units=exp_ds.units,
                    uncertainties={args.field: np.full_like(exp_ds.column(args.field), u_exp_val)},
                )
            u_num_val = u_num_arg if isinstance(u_num_arg, float) else None
            u_in_val = float(args.u_input) if args.u_input is not None else None

            vrep = ValidationStudy.from_datasets(
                sim_ds,
                exp_ds,
                field=args.field,
                coord=args.coord,
                u_num=u_num_val,
                u_input=u_in_val,
                coverage_factor=args.coverage_factor,
                error_tol=args.tol,
                study_name=args.study_name,
            )

            sr = standardize_result(vrep, kind="model_validation")
            proj.record_run("model_validation", sr, configuration={"field": args.field})
            if args.plot:
                plot_validation(vrep, path=args.plot)
            _emit(sr.to_dict(), vrep.summary(), as_json)
            return _exit_for_status(vrep.status)

        if args.command == "report":
            proj = Project.open(args.project)
            rep = proj.report(
                output=args.output,
                title=args.title,
                report_type=args.report_type,
                description=args.description,
                run_ids=args.run_id,
                figures=args.figure,
                also_manifest=args.manifest,
            )
            out_p = Path(args.output)
            rel_out = out_p.as_posix() if not out_p.is_absolute() else out_p.name
            if as_json:
                _emit(rep.to_dict(), "", True)
            else:
                print(
                    f"Generated {rep.report_type} report [{rep.status.value.upper()}] -> {rel_out}"
                )
            return _exit_for_status(rep.status)

        if args.command == "mcp":
            from .mcp.server import main as mcp_main

            return mcp_main(["--list-tools"] if args.list_tools else [])

        if args.command == "claim":
            from .claims import ClaimAuditEngine, ClaimManifest, ClaimVerdictKind, RebuttalSubmission
            from .reporting import ReportGenerator

            proj = Project.open(args.project)
            if args.claim_action == "init":
                manifest = ClaimManifest.from_file(args.manifest)
                dest = proj.register_claim_manifest(manifest)
                payload = {
                    "action": "claim_init",
                    "study_title": manifest.study_title,
                    "n_claims": len(manifest.claims),
                    "manifest_hash": manifest.content_hash(),
                    "path": dest.relative_to(proj.root).as_posix(),
                }
                _emit(payload, f"Registered ClaimManifest '{manifest.study_title}' ({len(manifest.claims)} claims).", as_json)
                return EXIT_OK

            if args.claim_action == "audit":
                manifest = ClaimManifest.from_file(args.manifest) if args.manifest else None
                rec = proj.audit_claims(manifest=manifest)
                rep_out: Optional[str] = None
                if args.output:
                    rep = ReportGenerator.from_claim_round(proj, rec, report_type="peer_review_letter")
                    out_p = Path(args.output)
                    full_out = out_p if out_p.is_absolute() else (proj.root / out_p)
                    rep.save(full_out, also_manifest=True)
                    rep_out = full_out.relative_to(proj.root).as_posix() if full_out.is_relative_to(proj.root) else full_out.name
                payload = {**rec.to_dict(), "report_path": rep_out}
                lines = [f"Claim audit {rec.round_id}: counts={rec.summary_counts}, open_requests={len(rec.open_requests)}"]
                for v in rec.verdicts:
                    lines.append(f"  - [{v.overall_verdict.value}] {v.claim_id}: {v.title}")
                _emit(payload, "\n".join(lines), as_json)
                has_block = any(
                    v.overall_verdict in (ClaimVerdictKind.FALSIFIED, ClaimVerdictKind.INCONCLUSIVE)
                    for v in rec.verdicts
                )
                return EXIT_CHECK_FAILED if has_block else EXIT_OK

            if args.claim_action == "questions":
                reqs = ClaimAuditEngine.load_open_clarifications(proj)
                if args.severity == "major":
                    reqs = [r for r in reqs if r.severity.startswith("major")]
                payload = {"count": len(reqs), "requests": [r.to_dict() for r in reqs]}
                lines = [f"Open clarification requests: {len(reqs)}"]
                for r in reqs:
                    lines.append(f"  - [{r.request_id}] ({r.severity}) {r.claim_id}: {r.reviewer_prompt}")
                _emit(payload, "\n".join(lines), as_json)
                return EXIT_OK

            if args.claim_action == "respond":
                subs = RebuttalSubmission.load_list(args.rebuttal)
                rec = proj.submit_rebuttal(subs)
                rep_out = None
                if args.output:
                    rep = ReportGenerator.from_claim_round(proj, rec, report_type="peer_review_letter")
                    out_p = Path(args.output)
                    full_out = out_p if out_p.is_absolute() else (proj.root / out_p)
                    rep.save(full_out, also_manifest=True)
                    rep_out = full_out.relative_to(proj.root).as_posix() if full_out.is_relative_to(proj.root) else full_out.name
                payload = {**rec.to_dict(), "report_path": rep_out}
                lines = [f"Rebuttal round {rec.round_id}: counts={rec.summary_counts}"]
                for tr in rec.transitions:
                    lines.append(f"  - {tr.claim_id}: {tr.prev_verdict} -> {tr.new_verdict}")
                _emit(payload, "\n".join(lines), as_json)
                has_block = any(
                    v.overall_verdict in (ClaimVerdictKind.FALSIFIED, ClaimVerdictKind.INCONCLUSIVE)
                    for v in rec.verdicts
                )
                return EXIT_CHECK_FAILED if has_block else EXIT_OK

    except (FileNotFoundError, ProjectError, ProvenanceMismatchError, ModelValidationError) as exc:
        err_payload = {"error": type(exc).__name__, "message": str(exc), "exit_code": EXIT_VALIDATION_ERROR}
        _emit(err_payload, f"Error ({type(exc).__name__}): {exc}", as_json)
        return EXIT_VALIDATION_ERROR
    except Exception as exc:
        err_payload = {"error": type(exc).__name__, "message": str(exc), "exit_code": EXIT_CHECK_FAILED}
        _emit(err_payload, f"Error ({type(exc).__name__}): {exc}", as_json)
        return EXIT_CHECK_FAILED

    return EXIT_OK


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if argv_list and argv_list[0] in WORKFLOW_COMMANDS:
        return _run_workflow_cli(argv_list)

    parser = build_parser()
    args = parser.parse_args(argv_list)

    if args.gci:
        from .simulate import plot_gci_study
        from .verify import gci_from_values

        try:
            h_part, q_part = args.gci.split(":", 1)
            hs = [float(x) for x in h_part.split(",")]
            qs = [float(x) for x in q_part.split(",")]
        except ValueError:
            parser.error("Expected --gci 'h1,h2,h3:phi1,phi2,phi3'")
        rep = gci_from_values(hs, qs, formal_order=args.formal_order)
        print(rep.summary())
        if args.gci_plot:
            print("Wrote", plot_gci_study(rep, path=args.gci_plot))
        if not args.rel:
            return 0

    if not args.rel or not args.target:
        parser.error("Provide --rel and --target (or standalone --gci), or use a workflow subcommand.")

    dims: Dict[str, str] = {}
    for item in args.dim:
        if "=" not in item:
            parser.error(f"Invalid --dim '{item}'; expected 'name=DIM'.")
        k, v = item.split("=", 1)
        dims[k.strip()] = v.strip()

    rates: List[Tuple[str, str, str]] = []
    for r_item in args.rate:
        parts = [x.strip() for x in r_item.split(",")]
        if len(parts) != 3:
            parser.error(f"Invalid --rate '{r_item}'; expected 'target,wrt,equals'.")
        rates.append((parts[0], parts[1], parts[2]))

    sys_obj = System.from_spec(
        relations=args.rel,
        dims=dims if dims else None,
        constants=args.const,
        rate_identities=rates if rates else None,
    )
    result = SymbolicSolver(sys_obj).derive(
        target=args.target,
        wrt=args.wrt,
        zero_initials=args.zero_initials,
        unit_constants=args.unit_constants,
    )

    if args.latex:
        print(result.to_latex())
    else:
        print(result.summary())

    if args.csv:
        data = _load_csv_columns(Path(args.csv))
        ranked = NumericalSolver.rank_candidates(result, data, target_col=args.target)
        print("\n=== Numerical Calibration & BIC Ranking ===")
        for idx, fit_res in enumerate(ranked, 1):
            print(f"{idx}. {fit_res.summary()}")

    if args.sweep:
        from .simulate import plot_simulation, simulate

        try:
            var, rng = args.sweep.split("=", 1)
            lo, hi, n = rng.split(":")
            sweep = {var.strip(): (float(lo), float(hi), int(n))}
            values = {k.strip(): float(v) for k, v in (s.split("=", 1) for s in args.set_values)}
            family = {
                k.strip(): [float(x) for x in v.split(",")]
                for k, v in (s.split("=", 1) for s in args.family)
            }
        except ValueError:
            parser.error("Use --sweep var=start:stop:n, --set name=value, --family name=v1,v2")
        sim = simulate(result, sweep, values=values, family=family, candidate=args.candidate)
        print(f"\n=== Simulation: {sim.y_name}({sim.x_name}) [{sim.candidate_name}] ===")
        if sim.assumed:
            print("Assumed defaults:", ", ".join(f"{k}={v:g}" for k, v in sim.assumed.items()))
        for label, y in sim.curves.items():
            print(f"  {label}: min={y.min():.6g}, max={y.max():.6g}, final={y[-1]:.6g}")
        if args.sim_csv:
            print("Wrote", sim.to_csv(args.sim_csv))
        if args.plot:
            print("Wrote", plot_simulation(sim, path=args.plot))
    elif args.plot:
        parser.error("--plot requires --sweep")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
