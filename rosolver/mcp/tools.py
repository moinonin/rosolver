"""
AI-native MCP tool definitions, JSON Schemas, resources, and prompts for Rosolver.

Designed around the principle:
    "AI proposes. Rosolver evaluates. Scientific evidence determines acceptance."

Supports both stateless inline calls (arrays, YAML strings, equations) and persistent
SHA-256 hash-verified `Project` workspaces on disk, while keeping array payloads compact
for LLM context windows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import sympy as sp

import rosolver
from .._parsing import parse_expr, real_symbol
from ..data import Dataset
from ..dimensions import DERIVED_UNIT_TO_DIM, SI_UNIT_TO_DIM, UNIT_TO_DIM, Dimension, DimensionalEngine
from ..integrations import VerificationStudy
from ..integrators import EXPLICIT_TABLEAUS
from ..model import Model, ModelValidationError
from ..numerical import NumericalSolver
from ..odesys import EventSpec, ODESystem
from ..pde import BC, ConservationLaw1D
from ..phasechange import Stefan1D, stefan_shedding_event
from ..project import Project, ProjectError
from ..reporting import REPORT_TYPES, ReportGenerator
from ..results import StandardResult, VerificationStatus, standardize_result, to_jsonable
from ..simulate import (
    plot_conservation_audit,
    plot_gci_study,
    plot_mms_convergence,
    plot_simulation,
    plot_stefan,
    plot_validation,
    simulate,
)
from ..templates import list_templates
from ..validation import ValidationStudy
from ..verify import (
    analyze_stiffness,
    audit_ode_invariants,
    audit_pde_conservation,
    cosimulate_coupled_odes,
    gci_from_values,
    grid_convergence_study,
    mms_study,
    timestep_study,
)


class MCPToolError(ValueError):
    """Raised when an MCP tool receives invalid inputs or fails precondition checks."""


def _downsample_1d(arr: np.ndarray, max_points: int = 64) -> Dict[str, Any]:
    flat = np.asarray(arr, dtype=float).reshape(-1)
    n = int(flat.size)
    if n == 0:
        return {"n_points": 0, "preview": []}
    if n <= max_points:
        preview = flat.tolist()
    else:
        idx = np.linspace(0, n - 1, max_points, dtype=int)
        preview = flat[idx].tolist()
    finite = flat[np.isfinite(flat)]
    return to_jsonable(
        {
            "n_points": n,
            "initial": float(flat[0]),
            "final": float(flat[-1]),
            "min": float(np.min(finite)) if finite.size else None,
            "max": float(np.max(finite)) if finite.size else None,
            "preview": preview,
        }
    )


def _parse_dim_entry(raw: str) -> Dimension:
    """Accept either base-dimension notation ('M*L^2/T^3') or SI unit notation ('W/(m*K)')."""
    try:
        return Dimension.from_unit(raw)
    except Exception:
        return Dimension.parse(raw)


def _eval_expr_dimension(expr: sp.Expr, sym_dims: Mapping[str, Dimension]) -> Tuple[bool, Dimension, List[str]]:
    """Recursively evaluate and verify dimensional homogeneity of a SymPy expression."""
    issues: List[str] = []

    def _walk(node: sp.Expr) -> Optional[Dimension]:
        if node.is_Number or node in (sp.pi, sp.E):
            return Dimension.dimensionless()
        if isinstance(node, sp.Symbol):
            name = node.name
            if name not in sym_dims:
                issues.append(f"Missing dimension/unit declaration for symbol '{name}'.")
                return None
            return sym_dims[name]
        if isinstance(node, sp.Add):
            child_dims = [_walk(arg) for arg in node.args]
            if any(d is None for d in child_dims):
                return None
            first = child_dims[0]
            assert first is not None
            for arg, d in zip(node.args[1:], child_dims[1:]):
                if d != first:
                    issues.append(
                        f"Additive dimension mismatch: '{node.args[0]}' has [{first}] vs '{arg}' with [{d}]."
                    )
                    return None
            return first
        if isinstance(node, sp.Mul):
            acc = Dimension.dimensionless()
            for arg in node.args:
                d = _walk(arg)
                if d is None:
                    return None
                acc = acc * d
            return acc
        if isinstance(node, sp.Pow):
            base_d = _walk(node.base)
            if base_d is None:
                return None
            exp = node.exp
            if exp.is_Rational:
                # Support rational powers when divisible, or check dimensionless base otherwise
                r = sp.Rational(exp)
                if r.q == 1:
                    return base_d ** int(r.p)
                out_map: Dict[str, int] = {}
                for k_dim, v_exp in base_d.exponents.items():
                    prod = sp.Rational(v_exp) * r
                    if prod.q != 1:
                        issues.append(
                            f"Fractional exponent {r} on '{node.base}' produces non-integer dimension {k_dim}^{prod}."
                        )
                        return None
                    out_map[k_dim] = int(prod.p)
                return Dimension.from_dict(out_map)
            exp_d = _walk(exp)
            if exp_d is not None and not exp_d.is_dimensionless:
                issues.append(f"Exponent '{exp}' in '{node}' must be dimensionless, got [{exp_d}].")
                return None
            if not base_d.is_dimensionless:
                issues.append(f"Variable-exponent base '{node.base}' must be dimensionless, got [{base_d}].")
                return None
            return Dimension.dimensionless()
        if isinstance(node, sp.Function):
            fname = node.func.__name__
            if fname == "sqrt" and len(node.args) == 1:
                return _walk(sp.Pow(node.args[0], sp.Rational(1, 2)))
            if fname in ("Abs",):
                return _walk(node.args[0])
            for arg in node.args:
                ad = _walk(arg)
                if ad is None:
                    return None
                if not ad.is_dimensionless:
                    issues.append(
                        f"Transcendental function '{fname}({arg})' requires dimensionless argument, got [{ad}]."
                    )
                    return None
            return Dimension.dimensionless()
        return Dimension.dimensionless()

    res_dim = _walk(expr)
    if issues or res_dim is None:
        return False, Dimension.dimensionless(), issues
    return True, res_dim, []


def _resolve_model(args: Mapping[str, Any]) -> Tuple[Model, Optional[Project]]:
    proj: Optional[Project] = None
    if args.get("project_dir"):
        proj = Project.open(str(args["project_dir"]))
    if args.get("model_yaml"):
        return Model.from_yaml(str(args["model_yaml"])), proj
    if args.get("model_dict"):
        return Model.from_dict(args["model_dict"]), proj
    if args.get("model_path"):
        return Model.from_file(str(args["model_path"])), proj
    if proj is not None:
        return proj.get_model(name=args.get("model_name")), proj
    raise MCPToolError(
        "Provide one of 'model_yaml', 'model_dict', 'model_path', or 'project_dir' (with registered model)."
    )


def _load_data_columns(args: Mapping[str, Any], proj: Optional[Project] = None) -> Dict[str, np.ndarray]:
    if args.get("data"):
        raw = args["data"]
        if not isinstance(raw, Mapping):
            raise MCPToolError("'data' must be an object mapping column names to numeric arrays.")
        return {str(k): np.asarray(v, dtype=float) for k, v in raw.items()}
    path_str = args.get("data_path") or args.get("csv_path")
    if not path_str and args.get("dataset_name") and proj is not None:
        p = proj.get_dataset_path(str(args["dataset_name"]), verify_hash=True)
    elif path_str:
        p = Path(str(path_str))
        if not p.exists() and proj is not None:
            p = proj.get_dataset_path(str(path_str), verify_hash=True)
    else:
        raise MCPToolError("Provide inline 'data', 'data_path'/'csv_path', or 'dataset_name' with 'project_dir'.")

    ext = p.suffix.lower()
    if ext == ".json":
        ds = Dataset.from_json(p)
    elif ext in (".yaml", ".yml"):
        ds = Dataset.from_yaml(p)
    elif ext == ".npz":
        ds = Dataset.from_numpy(p)
    else:
        ds = Dataset.from_csv(p, missing_policy=str(args.get("missing_policy", "raise")))
    out: Dict[str, np.ndarray] = dict(ds.coordinates)
    out.update(ds.fields)
    for f_k, u_arr in ds.uncertainties.items():
        out.setdefault(f"u_{f_k}", u_arr)
    return out


def _build_pde_bc(bc_raw: Any) -> Any:
    if bc_raw is None:
        return "outflow"
    if isinstance(bc_raw, str):
        return bc_raw
    if isinstance(bc_raw, Mapping):
        out: Dict[str, Any] = {}
        for side in ("left", "right"):
            spec = bc_raw.get(side)
            if spec is None or spec == "outflow":
                out[side] = "outflow"
            elif spec == "periodic":
                out[side] = BC.periodic()
            elif isinstance(spec, Mapping):
                kind = str(spec.get("kind", "dirichlet")).lower()
                val = spec.get("value", 0.0)
                out[side] = BC(kind, val)
            else:
                out[side] = BC.dirichlet(spec)
        return out
    return bc_raw


# ---------------------------------------------------------------------------
# Tool Handlers
# ---------------------------------------------------------------------------


def tool_capabilities(args: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "name": "rosolver-mcp",
        "rosolver_version": rosolver.__version__,
        "templates": [getattr(t, "name", str(t)) for t in list_templates()],
        "integrator_schemes": sorted(
            set(EXPLICIT_TABLEAUS.keys()) | {"beuler", "cn", "RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"}
        ),
        "pde_reconstructions": ["constant", "linear", "minmod", "vanleer", "mc", "superbee"],
        "report_types": list(REPORT_TYPES),
        "verification_statuses": [s.value for s in VerificationStatus],
        "base_dimensions": ["M", "L", "T", "I", "Theta", "N", "J", "rad"],
        "supported_si_units": sorted(set(UNIT_TO_DIM) | set(DERIVED_UNIT_TO_DIM) | set(SI_UNIT_TO_DIM)),
    }


def tool_check_dimensions(args: Mapping[str, Any]) -> Dict[str, Any]:
    dims_in = args.get("dims") or {}
    if not isinstance(dims_in, Mapping) or not dims_in:
        raise MCPToolError("Provide non-empty 'dims' mapping symbol names to units or dimensions.")
    parsed_dims = {str(k): _parse_dim_entry(str(v)) for k, v in dims_in.items()}
    result: Dict[str, Any] = {
        "parsed_dimensions": {k: str(d) for k, d in parsed_dims.items()},
    }

    eq_str = args.get("equation")
    if eq_str:
        if "=" in str(eq_str):
            lhs_s, rhs_s = str(eq_str).split("=", 1)
            lhs_expr = parse_expr(lhs_s)
            rhs_expr = parse_expr(rhs_s)
            ok_l, dim_l, iss_l = _eval_expr_dimension(lhs_expr, parsed_dims)
            ok_r, dim_r, iss_r = _eval_expr_dimension(rhs_expr, parsed_dims)
            issues = iss_l + iss_r
            consistent = ok_l and ok_r and (dim_l == dim_r)
            if ok_l and ok_r and dim_l != dim_r:
                issues.append(f"LHS dimension [{dim_l}] != RHS dimension [{dim_r}].")
            result["equation_check"] = {
                "equation": str(eq_str),
                "consistent": consistent,
                "status": VerificationStatus.PASSED.value if consistent else VerificationStatus.FAILED.value,
                "lhs_dimension": str(dim_l) if ok_l else None,
                "rhs_dimension": str(dim_r) if ok_r else None,
                "issues": issues,
            }
        else:
            expr = parse_expr(str(eq_str))
            ok_e, dim_e, iss_e = _eval_expr_dimension(expr, parsed_dims)
            result["equation_check"] = {
                "expression": str(eq_str),
                "consistent": ok_e,
                "status": VerificationStatus.PASSED.value if ok_e else VerificationStatus.FAILED.value,
                "dimension": str(dim_e) if ok_e else None,
                "issues": iss_e,
            }

    target_name = args.get("target")
    if target_name:
        t_str = str(target_name)
        if t_str not in parsed_dims:
            raise MCPToolError(f"Target '{t_str}' must be included in 'dims'.")
        cand_names = [str(c) for c in (args.get("candidates") or [k for k in parsed_dims if k != t_str])]
        cands = [(c, parsed_dims[c]) for c in cand_names]
        sol = DimensionalEngine.solve(parsed_dims[t_str], cands)
        sym_map = {k: real_symbol(k) for k in parsed_dims}
        result["buckingham_pi"] = {
            "target": t_str,
            "consistent": sol.consistent,
            "is_unique": sol.is_unique,
            "nullity": sol.nullity,
            "basis_dims": sol.basis_dims,
            "particular_monomial": str(sol.particular_monomial(sym_map)) if sol.consistent else None,
            "pi_groups": [str(p) for p in sol.pi_monomials(sym_map)],
            "missing_dimensions": sol.missing_dimensions,
            "diagnostic": sol.diagnostic,
        }

    return result


def tool_formulate(args: Mapping[str, Any]) -> Dict[str, Any]:
    if args.get("project_dir") and not args.get("relations"):
        proj = Project.open(str(args["project_dir"]))
        sr = proj.run_symbolic(
            model_name=args.get("model_name"),
            target=args.get("target"),
            wrt=args.get("wrt"),
            zero_initials=bool(args.get("zero_initials", False)),
            unit_constants=bool(args.get("unit_constants", False)),
            include_saturating=bool(args.get("include_saturating", False)),
            templates=args.get("templates"),
        )
        res = sr.raw
        run_id = sr.metadata.run_id
    else:
        rels = args.get("relations")
        target = args.get("target")
        if not rels or not target:
            raise MCPToolError("Provide 'relations' and 'target' (or 'project_dir' with a model having dependencies).")
        rates_in = args.get("rate_identities")
        rates = [tuple(r) for r in rates_in] if rates_in else None
        res = rosolver.formulate(
            relations=list(rels),
            target=str(target),
            wrt=args.get("wrt"),
            dims=args.get("dims"),
            constants=args.get("constants"),
            rate_identities=rates,  # type: ignore[arg-type]
            zero_initials=bool(args.get("zero_initials", False)),
            unit_constants=bool(args.get("unit_constants", False)),
            include_saturating=bool(args.get("include_saturating", False)),
            templates=args.get("templates"),
        )
        run_id = None

    payload: Dict[str, Any] = {
        "status": VerificationStatus.PASSED.value,
        "run_id": run_id,
        "target": str(res.target),
        "wrt": str(res.wrt) if res.wrt is not None else None,
        "primary_equation": f"{res.primary.equation.lhs} = {res.primary.equation.rhs}",
        "candidates": [
            {
                "name": c.name,
                "equation": f"{c.equation.lhs} = {c.equation.rhs}",
                "free_params": [str(p) for p in c.free_params],
                "template_name": getattr(c, "template_name", None),
            }
            for c in res.candidates
        ],
        "pi_groups": {k: [str(e) for e in v] for k, v in res.pi_groups.items()},
        "steps": [
            {
                "stage": st.stage,
                "explanation": st.explanation,
                "equation": f"{st.equation.lhs} = {st.equation.rhs}" if st.equation is not None else None,
            }
            for st in res.steps
        ],
        "diagnostics": list(res.diagnostics),
        "rejected_candidates": list(res.rejected_candidates),
    }
    if args.get("include_latex"):
        payload["latex"] = res.to_latex()
    return payload


def tool_validate_model(args: Mapping[str, Any]) -> Dict[str, Any]:
    proj: Optional[Project] = None
    integrity: Optional[Dict[str, Any]] = None
    if args.get("project_dir") and not (args.get("model_yaml") or args.get("model_dict") or args.get("model_path")):
        proj = Project.open(str(args["project_dir"]))
        integrity = proj.verify_integrity(strict=False)
        model_reports = []
        all_ok = bool(integrity["ok"])
        for mref in proj.models:
            m = proj.get_model(name=mref.name, version=mref.version, verify_hash=False)
            errs = m.validate(strict=False)
            if errs:
                all_ok = False
            model_reports.append(
                {
                    "name": m.name,
                    "version": m.version,
                    "valid": len(errs) == 0,
                    "errors": errs,
                    "content_hash": m.content_hash(),
                }
            )
        return {
            "valid": all_ok,
            "status": VerificationStatus.PASSED.value if all_ok else VerificationStatus.FAILED.value,
            "project_integrity": integrity,
            "models": model_reports,
        }

    try:
        model, proj = _resolve_model(args)
        errs = model.validate(strict=False)
    except ModelValidationError as exc:
        return {
            "valid": False,
            "status": VerificationStatus.FAILED.value,
            "errors": [str(exc)],
        }

    saved_to: Optional[str] = None
    if not errs and args.get("save_path"):
        out = model.save(str(args["save_path"]))
        saved_to = out.name
    registered_ref: Optional[Dict[str, Any]] = None
    if not errs and proj is not None and args.get("register_in_project"):
        mref = proj.add_model(model)
        registered_ref = mref.to_dict()

    return {
        "valid": len(errs) == 0,
        "status": VerificationStatus.PASSED.value if not errs else VerificationStatus.FAILED.value,
        "errors": errs,
        "name": model.name,
        "version": model.version,
        "domain": model.domain,
        "content_hash": model.content_hash(),
        "unit_map": model.unit_map(),
        "n_equations": len(model.equations),
        "saved_filename": saved_to,
        "registered_model": registered_ref,
    }


def tool_calibrate(args: Mapping[str, Any]) -> Dict[str, Any]:
    mode = str(args.get("mode", "single")).lower()
    proj = Project.open(str(args["project_dir"])) if args.get("project_dir") else None
    col_map = _load_data_columns(args, proj=proj)

    sigma_y = None
    if args.get("sigma_y") is not None:
        sigma_y = np.asarray(args["sigma_y"], dtype=float)
    elif args.get("uncertainty_col"):
        ucol = str(args["uncertainty_col"])
        if ucol not in col_map:
            raise MCPToolError(f"Uncertainty column '{ucol}' not in dataset columns {list(col_map)}.")
        sigma_y = col_map[ucol]

    weights = np.asarray(args["weights"], dtype=float) if args.get("weights") is not None else None
    missing_policy = str(args.get("missing_policy", "raise"))

    if mode in ("rank_candidates", "sparse_pi"):
        if args.get("relations"):
            deriv = rosolver.formulate(
                relations=list(args["relations"]),
                target=str(args["target"]),
                wrt=args.get("wrt"),
                dims=args.get("dims"),
                constants=args.get("constants"),
                zero_initials=bool(args.get("zero_initials", False)),
                include_saturating=bool(args.get("include_saturating", True)),
            )
        elif proj is not None:
            m = proj.get_model(name=args.get("model_name"))
            deriv = m.analyse_symbolically(target=args.get("target"), wrt=args.get("wrt")).raw
        else:
            raise MCPToolError(f"Mode '{mode}' requires 'relations' + 'target' or 'project_dir'.")

        criterion = str(args.get("criterion", "bic"))
        if mode == "rank_candidates":
            rep = NumericalSolver.rank_candidates_report(
                deriv,
                col_map,
                target_col=args.get("target"),
                criterion=criterion,
                weights=weights,
                sigma_y=sigma_y,
                missing_policy=missing_policy,
            )
        else:
            rep = NumericalSolver.select_sparse_pi_report(
                deriv,
                col_map,
                target_col=args.get("target"),
                max_groups=int(args.get("max_groups", 2)),
                form=str(args.get("form", "power")),
                criterion=criterion,
                weights=weights,
                sigma_y=sigma_y,
                missing_policy=missing_policy,
            )
        sr = standardize_result(rep, kind="candidate_ranking")
        if proj is not None and bool(args.get("record_run", True)):
            proj.record_run("calibration", sr, configuration={"mode": mode, "criterion": criterion})
        return to_jsonable(sr.to_dict())

    # Single-equation or Model calibration
    if args.get("equation"):
        eq_str = str(args["equation"])
        if "=" in eq_str:
            lhs_s, rhs_s = eq_str.split("=", 1)
            eq_obj: Any = sp.Eq(real_symbol(lhs_s.strip()), parse_expr(rhs_s))
            default_tgt = lhs_s.strip()
        else:
            default_tgt = str(args.get("target", "y"))
            eq_obj = sp.Eq(real_symbol(default_tgt), parse_expr(eq_str))
        bounds = {str(k): (float(v[0]), float(v[1])) for k, v in (args.get("bounds") or {}).items()} or None
        fit_res = NumericalSolver.fit(
            eq_obj,
            col_map,
            target_col=args.get("target") or default_tgt,
            free_params=args.get("free_params"),
            fixed_params=args.get("fixed_params"),
            weights=weights,
            sigma_y=sigma_y,
            bounds=bounds,
            loss=str(args.get("loss", "linear")),
            missing_policy=missing_policy,
        )
        sr = standardize_result(fit_res, kind="calibration")
        if proj is not None and bool(args.get("record_run", True)):
            proj.record_run("calibration", sr, configuration={"equation": eq_str})
        return to_jsonable(sr.to_dict())

    model, proj = _resolve_model(args)
    fit_kwargs: Dict[str, Any] = {
        "loss": str(args.get("loss", "linear")),
        "missing_policy": missing_policy,
    }
    if sigma_y is not None:
        fit_kwargs["sigma_y"] = sigma_y
    if weights is not None:
        fit_kwargs["weights"] = weights
    if args.get("bounds"):
        fit_kwargs["bounds"] = {str(k): (float(v[0]), float(v[1])) for k, v in args["bounds"].items()}

    if proj is not None and bool(args.get("record_run", True)):
        sr = proj.run_calibration(data=col_map, model_name=model.name, target=args.get("target"), **fit_kwargs)
    else:
        sr = model.calibrate(col_map, target=args.get("target"), **fit_kwargs)
    return to_jsonable(sr.to_dict())


def tool_simulate(args: Mapping[str, Any]) -> Dict[str, Any]:
    solver_kind = str(args.get("solver", "model")).lower()
    max_pts = int(args.get("max_preview_points", 64))
    proj = Project.open(str(args["project_dir"])) if args.get("project_dir") else None

    if solver_kind == "model":
        model, proj = _resolve_model(args)
        t_span_in = args.get("t_span") or [0.0, 1.0]
        t_span = (float(t_span_in[0]), float(t_span_in[1]))
        dom_in = args.get("domain")
        dom = (float(dom_in[0]), float(dom_in[1])) if dom_in is not None else None
        ev_specs = [
            EventSpec(
                name=str(ev["name"]),
                expr=str(ev["expr"]),
                terminal=bool(ev.get("terminal", False)),
                direction=float(ev.get("direction", 0.0)),
            )
            for ev in (args.get("events") or [])
        ] or None
        solve_kwargs: Dict[str, Any] = {"t_span": t_span}
        for key in ("method", "n_cells", "reconstruction", "dt", "cfl", "n_out", "params"):
            if args.get(key) is not None:
                solve_kwargs[key] = args[key]
        if dom is not None:
            solve_kwargs["domain"] = dom
        if ev_specs is not None:
            solve_kwargs["events"] = ev_specs

        if proj is not None and bool(args.get("record_run", True)):
            sr = proj.run_simulation(model_name=model.name, **solve_kwargs)
        else:
            sr = model.solve(**solve_kwargs)

        raw = sr.raw
        summary_data: Dict[str, Any] = {
            "kind": sr.kind,
            "status": sr.status.value,
            "run_id": sr.metadata.run_id,
            "metadata": sr.metadata.to_dict(),
        }
        if hasattr(raw, "U"):
            summary_data["x"] = _downsample_1d(raw.x, max_pts)
            summary_data["t"] = _downsample_1d(raw.t, max_pts)
            summary_data["fields_final"] = {f: _downsample_1d(raw.final(f), max_pts) for f in raw.fields}
        elif hasattr(raw, "states"):
            summary_data["t"] = _downsample_1d(raw.t, max_pts)
            summary_data["states"] = {s: _downsample_1d(raw[s], max_pts) for s in raw.states}
            summary_data["t_events"] = to_jsonable(raw.t_events)
        elif hasattr(raw, "delta"):
            summary_data["t"] = _downsample_1d(raw.t, max_pts)
            summary_data["delta"] = _downsample_1d(raw.delta, max_pts)
        return summary_data

    if solver_kind == "ode":
        rhs = args.get("rhs")
        y0 = args.get("y0")
        if not isinstance(rhs, Mapping) or not isinstance(y0, Mapping):
            raise MCPToolError("solver='ode' requires 'rhs' and 'y0' mappings.")
        ode = ODESystem(rhs, indep=str(args.get("indep", "t")))
        t_span_in = args.get("t_span") or [0.0, 1.0]
        ev_specs = [
            EventSpec(
                name=str(ev["name"]),
                expr=str(ev["expr"]),
                terminal=bool(ev.get("terminal", False)),
                direction=float(ev.get("direction", 0.0)),
            )
            for ev in (args.get("events") or [])
        ] or None
        traj = ode.solve(
            t_span=(float(t_span_in[0]), float(t_span_in[1])),
            y0={str(k): float(v) for k, v in y0.items()},
            params=args.get("params"),
            method=str(args.get("method", "RK45")),
            dt=float(args["dt"]) if args.get("dt") is not None else None,
            n_out=int(args.get("n_out", 100)),
            events=ev_specs,
        )
        sr = standardize_result(traj, kind="ode_simulation")
        if proj is not None and bool(args.get("record_run", True)):
            proj.record_run("ode_simulation", sr, configuration={"rhs": dict(rhs)})
        return {
            "kind": "ode_simulation",
            "status": sr.status.value,
            "run_id": sr.metadata.run_id,
            "success": traj.success,
            "newton_failures": traj.newton_failures,
            "t": _downsample_1d(traj.t, max_pts),
            "states": {s: _downsample_1d(traj[s], max_pts) for s in traj.states},
            "final": traj.final,
            "t_events": to_jsonable(traj.t_events),
            "warnings": traj.warnings,
        }

    if solver_kind == "pde":
        fields = args.get("fields", "u")
        law = ConservationLaw1D(
            fields=fields,
            flux=args.get("flux", 0),
            diffusivity=args.get("diffusivity", 0),
            source=args.get("source", 0),
            x=str(args.get("x", "x")),
            t=str(args.get("t", "t")),
        )
        dom_in = args.get("domain") or [0.0, 1.0]
        t_span_in = args.get("t_span") or [0.0, float(args.get("t_end", 1.0))]
        sol = law.solve(
            domain=(float(dom_in[0]), float(dom_in[1])),
            n_cells=int(args.get("n_cells", 64)),
            t0=float(t_span_in[0]),
            t_end=float(t_span_in[1]),
            initial=args.get("initial", "0.0"),
            bc=_build_pde_bc(args.get("bc")),
            params=args.get("params"),
            scheme=str(args.get("method") or args.get("scheme") or "ssprk3"),
            dt=float(args["dt"]) if args.get("dt") is not None else None,
            cfl=float(args.get("cfl", 0.4)),
            reconstruction=str(args.get("reconstruction", "linear")),
            n_out=int(args.get("n_out", 50)),
        )
        sr = standardize_result(sol, kind="pde_simulation")
        if proj is not None and bool(args.get("record_run", True)):
            proj.record_run("pde_simulation", sr, configuration={"fields": list(law.fields)})
        return {
            "kind": "pde_simulation",
            "status": sr.status.value,
            "run_id": sr.metadata.run_id,
            "success": sol.success,
            "newton_failures": sol.newton_failures,
            "x": _downsample_1d(sol.x, max_pts),
            "t": _downsample_1d(sol.t, max_pts),
            "fields_final": {f: _downsample_1d(sol.final(f), max_pts) for f in sol.fields},
            "integrals_final": {f: sol.integral(f) for f in sol.fields},
            "warnings": sol.warnings,
        }

    if solver_kind == "stefan":
        st_kwargs: Dict[str, Any] = {}
        for k in (
            "k_eff",
            "rho",
            "cp",
            "Lf",
            "T_wall",
            "T_inf",
            "T_f",
            "h_c",
            "n_xi",
            "mdot_imp",
            "cp_imp",
            "T_imp",
            "V_imp",
            "mdot_evap",
            "tau_drain",
        ):
            if args.get(k) is not None:
                st_kwargs[k] = args[k]
        st = Stefan1D(**st_kwargs)
        t_end = float(args.get("t_end", 3600.0))
        delta0 = float(args.get("delta0", 1e-5))
        shed_list = args.get("shedding_events")
        if shed_list:
            evs = [
                stefan_shedding_event(
                    delta_crit=float(ev["delta_crit"]),
                    shed_fraction=float(ev.get("shed_fraction", 0.8)),
                )
                for ev in shed_list
            ]
            hyb = st.solve_with_shedding(evs, t_end, delta0=delta0)
            return {
                "kind": "stefan_hybrid_simulation",
                "status": VerificationStatus.PASSED.value if hyb.success else VerificationStatus.FAILED.value,
                "t": _downsample_1d(hyb.t, max_pts),
                "delta": _downsample_1d(hyb.delta, max_pts),
                "n_events": len(hyb.events),
                "events": to_jsonable(hyb.events),
            }

        sol = st.solve(
            t_end=t_end,
            delta0=delta0,
            method=str(args.get("method", "BDF")),
            n_out=int(args.get("n_out", 100)),
        )
        sr = standardize_result(sol, kind="stefan_simulation")
        if proj is not None and bool(args.get("record_run", True)):
            proj.record_run("stefan_simulation", sr, configuration=st_kwargs)
        if args.get("plot_path"):
            plot_stefan(sol, st, path=str(args["plot_path"]))
        neumann_ref = float(st.neumann_delta(t_end)) if st.h_c == 0.0 and st.mdot_imp is None else None
        branch_arr = getattr(sol, "branch", None)
        final_branch = str(branch_arr[-1]) if branch_arr is not None and len(branch_arr) > 0 else None
        return {
            "kind": "stefan_simulation",
            "status": sr.status.value,
            "run_id": sr.metadata.run_id,
            "stefan_number": st.stefan_number,
            "t": _downsample_1d(sol.t, max_pts),
            "delta": _downsample_1d(sol.delta, max_pts),
            "final_delta_m": float(sol.delta[-1]),
            "neumann_exact_delta_m": neumann_ref,
            "final_branch": final_branch,
            "warnings": sol.warnings,
        }

    if solver_kind == "sweep":
        deriv = rosolver.formulate(
            relations=list(args["relations"]),
            target=str(args["target"]),
            wrt=args.get("wrt"),
            dims=args.get("dims"),
            constants=args.get("constants"),
            zero_initials=bool(args.get("zero_initials", True)),
        )
        sw_raw = args.get("sweep") or {}
        sweep_spec = {str(k): (float(v[0]), float(v[1]), int(v[2])) for k, v in sw_raw.items()}
        sim = simulate(
            deriv,
            sweep=sweep_spec,
            values=args.get("values"),
            family=args.get("family"),
            candidate=args.get("candidate"),
        )
        if args.get("csv_path"):
            sim.to_csv(str(args["csv_path"]))
        if args.get("plot_path"):
            plot_simulation(sim, path=str(args["plot_path"]))
        return {
            "kind": "parametric_sweep",
            "status": VerificationStatus.PASSED.value,
            "candidate_name": sim.candidate_name,
            "x_name": sim.x_name,
            "y_name": sim.y_name,
            "x": _downsample_1d(sim.x, max_pts),
            "curves": {lbl: _downsample_1d(arr, max_pts) for lbl, arr in sim.curves.items()},
            "assumed_defaults": sim.assumed,
        }

    raise MCPToolError(f"Unsupported solver='{solver_kind}'. Expected model, ode, pde, stefan, or sweep.")


def tool_verify(args: Mapping[str, Any]) -> Dict[str, Any]:
    method = str(args.get("method", "gci")).lower()
    proj = Project.open(str(args["project_dir"])) if args.get("project_dir") else None

    if method == "gci":
        if args.get("spacings") is not None and args.get("quantities") is not None:
            rep = gci_from_values(
                spacings=[float(x) for x in args["spacings"]],
                quantities=[float(x) for x in args["quantities"]],
                formal_order=float(args["formal_order"]) if args.get("formal_order") is not None else None,
            )
        elif args.get("csv_path"):
            vs = VerificationStudy.from_csv(
                str(args["csv_path"]),
                spacing_col=str(args.get("spacing_col", "h")),
                qoi_col=str(args.get("qoi_col", "qoi")),
            )
            rep = vs.gci(
                formal_order=float(args["formal_order"]) if args.get("formal_order") is not None else None
            )
        else:
            model, proj = _resolve_model(args)
            law = model.to_conservation_law()
            dom = model.numerics.domain or (0.0, 1.0)
            ic = model._extract_initial_conditions(None, law.fields)
            bc = model._extract_boundary_conditions(None, law.fields)
            cells = [int(c) for c in (args.get("cells") or [20, 40, 80])]
            t_end = float(args.get("t_end", 0.5))
            fld0 = law.fields[0]
            rep = grid_convergence_study(
                lambda n: float(
                    law.solve(
                        domain=dom,
                        n_cells=n,
                        t_end=t_end,
                        initial=ic if len(law.fields) > 1 else ic[fld0],
                        bc=bc,
                        params=model.parameter_values(),
                        scheme=model.numerics.time_integrator,
                        reconstruction=model.numerics.reconstruction,
                    ).integral(fld0)
                ),
                cells=cells,
                domain_length=float(dom[1] - dom[0]),
                formal_order=float(args.get("formal_order", 2.0)),
            )
        if args.get("plot_path"):
            plot_gci_study(rep, path=str(args["plot_path"]))
        sr = standardize_result(rep, kind="gci")
        if proj is not None and bool(args.get("record_run", True)):
            proj.record_run("gci", sr)
        return to_jsonable(sr.to_dict())

    if method == "mms":
        if args.get("fields") or args.get("flux") is not None or args.get("diffusivity") is not None:
            law = ConservationLaw1D(
                fields=args.get("fields", "u"),
                flux=args.get("flux", 0),
                diffusivity=args.get("diffusivity", 0),
                source=args.get("source", 0),
            )
            dom = tuple(float(x) for x in (args.get("domain") or [0.0, 1.0]))
            p_vals = args.get("params")
        else:
            model, proj = _resolve_model(args)
            law = model.to_conservation_law()
            dom = model.numerics.domain or (0.0, 1.0)
            p_vals = model.parameter_values(args.get("params"))
        fld0 = law.fields[0]
        exact = args.get("exact") or f"exp(-{law.t.name}) * sin(pi * {law.x.name})"
        bc_kind = str(args.get("bc_kind") or ("dirichlet" if "sin(pi" in str(exact) else "periodic"))
        mms_rep = mms_study(
            law,
            exact={fld0: exact} if isinstance(exact, str) else exact,
            domain=(dom[0], dom[1]),
            t_end=float(args.get("t_end", 0.2)),
            cells=[int(c) for c in (args.get("cells") or [16, 32, 64])],
            params=p_vals,
            bc_kind=bc_kind,
            reconstruction=str(args.get("reconstruction", "linear")),
            scheme=str(args.get("scheme", "cn")),
        )
        if args.get("plot_path"):
            plot_mms_convergence(mms_rep, path=str(args["plot_path"]))
        sr = standardize_result(mms_rep, kind="mms")
        if proj is not None and bool(args.get("record_run", True)):
            proj.record_run("mms", sr)
        return to_jsonable(sr.to_dict())

    if method == "timestep":
        dts = [float(d) for d in (args.get("dts") or [0.1, 0.05, 0.025])]
        if args.get("rhs"):
            ode = ODESystem(args["rhs"])
            t_span_in = args.get("t_span") or [0.0, 1.0]
            dt_rep = timestep_study(
                ode,
                dts=dts,
                t_span=(float(t_span_in[0]), float(t_span_in[1])),
                y0=args["y0"],
                params=args.get("params"),
                scheme=str(args.get("scheme", "rk4")),
                state=args.get("state"),
                formal_order=float(args["formal_order"]) if args.get("formal_order") is not None else None,
            )
        else:
            model, proj = _resolve_model(args)
            law = model.to_conservation_law()
            dom = model.numerics.domain or (0.0, 1.0)
            ic = model._extract_initial_conditions(None, law.fields)
            bc = model._extract_boundary_conditions(None, law.fields)
            fld0 = law.fields[0]
            t_end = float(args.get("t_end", 0.5))
            scheme = str(args.get("scheme") or model.numerics.time_integrator)
            dt_rep = timestep_study(
                lambda d: float(
                    law.solve(
                        domain=dom,
                        n_cells=int(args.get("n_cells", 64)),
                        t_end=t_end,
                        initial=ic if len(law.fields) > 1 else ic[fld0],
                        bc=bc,
                        params=model.parameter_values(),
                        scheme=scheme,
                        dt=d,
                        reconstruction=model.numerics.reconstruction,
                    ).probe(0.5 * (dom[0] + dom[1]), fld0)
                ),
                dts=dts,
                formal_order=float(args.get("formal_order", 2.0)),
            )
        sr = standardize_result(dt_rep, kind="timestep")
        if proj is not None and bool(args.get("record_run", True)):
            proj.record_run("timestep", sr)
        return to_jsonable(sr.to_dict())

    if method == "conservation_pde":
        if args.get("fields") or args.get("flux") is not None or args.get("diffusivity") is not None:
            law = ConservationLaw1D(
                fields=args.get("fields", "u"),
                flux=args.get("flux", 0),
                diffusivity=args.get("diffusivity", 0),
                source=args.get("source", 0),
            )
            dom_in = args.get("domain") or [0.0, 1.0]
            sol = law.solve(
                domain=(float(dom_in[0]), float(dom_in[1])),
                n_cells=int(args.get("n_cells", 64)),
                t_end=float(args.get("t_end", 0.5)),
                initial=args.get("initial", "sin(pi * x)"),
                bc=_build_pde_bc(args.get("bc")),
                params=args.get("params"),
                scheme=str(args.get("scheme", "cn")),
            )
        else:
            model, proj = _resolve_model(args)
            sol = model.solve(t_span=(0.0, float(args.get("t_end", 0.5)))).raw
        cons = audit_pde_conservation(sol, tol=float(args.get("tol", 1e-10)))
        if args.get("plot_path"):
            plot_conservation_audit(cons, path=str(args["plot_path"]))
        sr = standardize_result(cons, kind="conservation")
        if proj is not None and bool(args.get("record_run", True)):
            proj.record_run("conservation", sr)
        out = to_jsonable(sr.to_dict())
        # Keep residual_series compact
        for entry in out.get("result", {}).get("entries", {}).values():
            if isinstance(entry.get("residual_series"), list) and len(entry["residual_series"]) > 32:
                entry["residual_series"] = entry["residual_series"][:: max(1, len(entry["residual_series"]) // 32)]
        return out

    if method == "conservation_ode":
        rhs = args.get("rhs")
        y0 = args.get("y0")
        invs = args.get("invariants")
        if not isinstance(rhs, Mapping) or not isinstance(y0, Mapping) or not isinstance(invs, Mapping):
            raise MCPToolError("method='conservation_ode' requires 'rhs', 'y0', and 'invariants' mappings.")
        ode = ODESystem(rhs)
        t_span_in = args.get("t_span") or [0.0, 10.0]
        traj = ode.solve(
            t_span=(float(t_span_in[0]), float(t_span_in[1])),
            y0={str(k): float(v) for k, v in y0.items()},
            params=args.get("params"),
            method=str(args.get("scheme", "DOP853")),
            rtol=1e-10,
            atol=1e-12,
        )
        cons = audit_ode_invariants(
            ode,
            traj,
            invariants=invs,
            params=args.get("params"),
            tol=float(args.get("tol", 1e-6)),
        )
        sr = standardize_result(cons, kind="conservation")
        out = to_jsonable(sr.to_dict())
        for entry in out.get("result", {}).get("entries", {}).values():
            if isinstance(entry.get("residual_series"), list) and len(entry["residual_series"]) > 32:
                entry["residual_series"] = entry["residual_series"][:: max(1, len(entry["residual_series"]) // 32)]
        return out

    if method == "stiffness":
        rhs = args.get("rhs")
        state = args.get("state")
        if not isinstance(rhs, Mapping) or state is None:
            raise MCPToolError("method='stiffness' requires 'rhs' mapping and 'state' vector/mapping.")
        ode = ODESystem(rhs)
        st_vec = [float(state[s]) for s in ode.states] if isinstance(state, Mapping) else [float(x) for x in state]
        stiff_rep = analyze_stiffness(
            ode,
            state=st_vec,
            params=args.get("params"),
            stiff_threshold=float(args.get("stiff_threshold", 100.0)),
        )
        return to_jsonable(standardize_result(stiff_rep, kind="stiffness").to_dict())

    if method == "coupling":
        sys_a = ODESystem(args["sys_a_rhs"])
        sys_b = ODESystem(args["sys_b_rhs"])
        t_span_in = args.get("t_span") or [0.0, 2.0]
        cs_rep = cosimulate_coupled_odes(
            sys_a,
            sys_b,
            t_span=(float(t_span_in[0]), float(t_span_in[1])),
            y0={str(k): float(v) for k, v in args["y0"].items()},
            dt_macro=float(args.get("dt_macro", 0.1)),
            params=args.get("params"),
            mode=str(args.get("mode", "strong")),
        )
        return to_jsonable(standardize_result(cs_rep, kind="coupling").to_dict())

    raise MCPToolError(
        f"Unsupported verification method '{method}'. Choose from: gci, mms, timestep, conservation_pde, conservation_ode, stiffness, coupling."
    )


def tool_validate_results(args: Mapping[str, Any]) -> Dict[str, Any]:
    proj = Project.open(str(args["project_dir"])) if args.get("project_dir") else None
    field_name = str(args.get("field", "y"))
    unit = str(args.get("unit", "1"))
    cov = float(args.get("coverage_factor", 1.0))
    err_tol = float(args["error_tol"]) if args.get("error_tol") is not None else None

    if args.get("simulation") is not None and args.get("experiment") is not None:
        vrep = ValidationStudy.compare(
            simulation=args["simulation"],
            experiment=args["experiment"],
            field=field_name,
            unit=unit,
            u_num=args.get("u_num"),
            u_exp=args.get("u_exp"),
            u_input=args.get("u_input"),
            coordinates=args.get("coordinates"),
            coord_name=str(args.get("coord", "x")),
            coverage_factor=cov,
            error_tol=err_tol,
            study_name=str(args.get("study_name", "mcp_validation")),
        )
    else:
        coord = args.get("coord")
        coord_list = [str(coord)] if coord else None
        if args.get("sim_csv"):
            u_map_s = {field_name: str(args["u_num"])} if isinstance(args.get("u_num"), str) else None
            sim_ds = Dataset.from_csv(str(args["sim_csv"]), coordinates=coord_list, fields=[field_name], uncertainties=u_map_s)
            u_num_val = float(args["u_num"]) if isinstance(args.get("u_num"), (int, float)) else None
        elif proj is not None:
            from ..cli import _load_sim_dataset_from_run

            sim_ds = _load_sim_dataset_from_run(proj, args.get("run_id"), field_name, coord)
            u_num_val = float(args["u_num"]) if args.get("u_num") is not None else None
        else:
            raise MCPToolError("Provide inline 'simulation'+'experiment', 'sim_csv', or 'project_dir'.")

        if args.get("exp_csv"):
            u_map_e = {field_name: str(args["u_exp"])} if isinstance(args.get("u_exp"), str) else None
            exp_ds = Dataset.from_csv(str(args["exp_csv"]), coordinates=coord_list, fields=[field_name], uncertainties=u_map_e)
        elif proj is not None and args.get("exp_dataset"):
            p_exp = proj.get_dataset_path(str(args["exp_dataset"]), verify_hash=True)
            u_map_e = {field_name: str(args["u_exp"])} if isinstance(args.get("u_exp"), str) else None
            exp_ds = Dataset.from_csv(p_exp, coordinates=coord_list, fields=[field_name], uncertainties=u_map_e)
        else:
            raise MCPToolError("Provide 'exp_csv' or 'exp_dataset' (with 'project_dir').")

        if isinstance(args.get("u_exp"), (int, float)) and exp_ds.uncertainty_for(field_name) is None:
            exp_ds = Dataset(
                name=exp_ds.name,
                coordinates=exp_ds.coordinates,
                fields=exp_ds.fields,
                units=exp_ds.units,
                uncertainties={field_name: np.full_like(exp_ds.column(field_name), float(args["u_exp"]))},
            )

        vrep = ValidationStudy.from_datasets(
            sim_ds,
            exp_ds,
            field=field_name,
            coord=coord,
            u_num=u_num_val,
            u_input=args.get("u_input"),
            coverage_factor=cov,
            error_tol=err_tol,
            study_name=args.get("study_name"),
        )

    if args.get("plot_path"):
        plot_validation(vrep, path=str(args["plot_path"]))
    sr = standardize_result(vrep, kind="model_validation")
    if proj is not None and bool(args.get("record_run", True)):
        proj.record_run("model_validation", sr, configuration={"field": field_name})
    return to_jsonable(sr.to_dict())


def tool_project(args: Mapping[str, Any]) -> Dict[str, Any]:
    action = str(args.get("action", "info")).lower()
    p_dir = args.get("project_dir")
    if not p_dir:
        raise MCPToolError("'project_dir' is required for rosolver_project.")

    if action == "init":
        proj = Project.create(
            str(p_dir),
            name=args.get("name"),
            description=str(args.get("description", "")),
        )
        if args.get("model_yaml") or args.get("model_path") or args.get("model_dict"):
            m, _ = _resolve_model(args)
            proj.add_model(m)
        return to_jsonable(proj.info())

    proj = Project.open(str(p_dir))
    if action == "info":
        return to_jsonable(proj.info())
    if action == "verify_integrity":
        return to_jsonable(proj.verify_integrity(strict=False))
    if action == "add_model":
        m, _ = _resolve_model(args)
        ref = proj.add_model(m, filename=args.get("filename"))
        return to_jsonable(ref.to_dict())
    if action == "add_dataset":
        src = args.get("dataset_path")
        if not src:
            raise MCPToolError("action='add_dataset' requires 'dataset_path'.")
        dref = proj.add_dataset(
            str(src),
            name=args.get("dataset_name"),
            category=str(args.get("category", "experimental")),
            description=str(args.get("description", "")),
        )
        return to_jsonable(dref.to_dict())
    if action == "ledger":
        return {"runs": [to_jsonable(r.to_dict()) for r in proj.ledger()]}
    if action == "get_run":
        rid = str(args["run_id"])
        return to_jsonable(proj.get_run(rid).to_dict())
    if action == "replay_run":
        rid = str(args["run_id"])
        sr = proj.replay_run(rid)
        return {
            "replayed_from": rid,
            "new_run_id": sr.metadata.run_id,
            "kind": sr.kind,
            "status": sr.status.value,
        }
    raise MCPToolError(
        f"Unsupported project action '{action}'. Expected init, info, verify_integrity, add_model, add_dataset, ledger, get_run, or replay_run."
    )


def tool_report(args: Mapping[str, Any]) -> Dict[str, Any]:
    if not args.get("project_dir"):
        raise MCPToolError("'project_dir' is required for rosolver_report.")
    proj = Project.open(str(args["project_dir"]))
    out_rel = str(args.get("output", "reports/final_report.md"))
    rep = proj.report(
        output=out_rel,
        title=args.get("title"),
        report_type=str(args.get("report_type", "integrated_project")),
        run_ids=args.get("run_ids"),
        figures=args.get("figures"),
        description=str(args.get("description", "")),
        also_manifest=bool(args.get("also_manifest", True)),
    )
    return {
        "title": rep.title,
        "report_type": rep.report_type,
        "status": rep.status.value,
        "output": out_rel,
        "warnings": rep.warnings,
        "n_runs": len(rep.runs),
        "markdown": rep.to_markdown(),
    }


# ---------------------------------------------------------------------------
# Tool Catalog (Schemas + Dispatch)
# ---------------------------------------------------------------------------

TOOL_SPECIFICATIONS: List[Dict[str, Any]] = [
    {
        "name": "rosolver_capabilities",
        "description": "Inspect Rosolver version, supported SI units, base dimensions, equation templates, numerical schemes, and 5-state verification taxonomy.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "rosolver_check_dimensions",
        "description": "Verify dimensional consistency of physical equations/expressions and extract exact rational Buckingham-Pi dimensionless groups.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "dims": {
                    "type": "object",
                    "description": "Map of symbol -> SI unit ('W/(m*K)', 'm^2/s', 'Pa') or base dimension ('M*L^2/T^3').",
                },
                "equation": {
                    "type": "string",
                    "description": "Optional equation ('P = 0.5 * rho * A * V**3') or expression to check for dimensional homogeneity.",
                },
                "target": {
                    "type": "string",
                    "description": "Optional target symbol in 'dims' to solve Buckingham-Pi closure for.",
                },
                "candidates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional subset of governing symbols for Buckingham-Pi.",
                },
            },
            "required": ["dims"],
        },
    },
    {
        "name": "rosolver_formulate",
        "description": "Derive closed-form symbolic candidate equations, Clairaut potentials, and Buckingham-Pi groups from qualitative dependency DAGs.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "relations": {"type": "array", "items": {"type": "string"}},
                "target": {"type": "string"},
                "wrt": {"type": "string"},
                "dims": {"type": "object"},
                "constants": {"type": "array", "items": {"type": "string"}},
                "rate_identities": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
                "zero_initials": {"type": "boolean"},
                "unit_constants": {"type": "boolean"},
                "include_saturating": {"type": "boolean"},
                "templates": {"type": "array", "items": {"type": "string"}},
                "include_latex": {"type": "boolean"},
                "project_dir": {"type": "string"},
                "model_name": {"type": "string"},
            },
        },
    },
    {
        "name": "rosolver_validate_model",
        "description": "Deterministically validate a Rosolver Model YAML/JSON/dict or Project workspace (AST-safe math, acyclic DAGs, units, SHA-256 hashes).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_yaml": {"type": "string"},
                "model_dict": {"type": "object"},
                "model_path": {"type": "string"},
                "project_dir": {"type": "string"},
                "save_path": {"type": "string"},
                "register_in_project": {"type": "boolean"},
            },
        },
    },
    {
        "name": "rosolver_calibrate",
        "description": "Calibrate parameters with OLS/WLS/NLLS, evaluate Jacobian identifiability & correlations, or rank symbolic candidates by BIC/AIC.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["single", "rank_candidates", "sparse_pi"]},
                "equation": {"type": "string"},
                "free_params": {"type": "array", "items": {"type": "string"}},
                "fixed_params": {"type": "object"},
                "relations": {"type": "array", "items": {"type": "string"}},
                "dims": {"type": "object"},
                "constants": {"type": "array", "items": {"type": "string"}},
                "target": {"type": "string"},
                "wrt": {"type": "string"},
                "data": {"type": "object"},
                "data_path": {"type": "string"},
                "dataset_name": {"type": "string"},
                "sigma_y": {"type": "array", "items": {"type": "number"}},
                "uncertainty_col": {"type": "string"},
                "weights": {"type": "array", "items": {"type": "number"}},
                "bounds": {"type": "object"},
                "loss": {"type": "string", "enum": ["linear", "soft_l1", "huber", "cauchy", "arctan"]},
                "missing_policy": {"type": "string", "enum": ["raise", "drop", "mask"]},
                "criterion": {"type": "string", "enum": ["bic", "aic", "rmse", "r2"]},
                "project_dir": {"type": "string"},
                "model_yaml": {"type": "string"},
                "model_path": {"type": "string"},
                "model_name": {"type": "string"},
                "record_run": {"type": "boolean"},
            },
        },
    },
    {
        "name": "rosolver_simulate",
        "description": "Solve declarative Models, multi-state ODEs (+events), 1D conservative FVM PDEs, moving-boundary Stefan ice/phase-change problems, or parametric curve sweeps.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "solver": {"type": "string", "enum": ["model", "ode", "pde", "stefan", "sweep"]},
                "project_dir": {"type": "string"},
                "model_yaml": {"type": "string"},
                "model_dict": {"type": "object"},
                "model_path": {"type": "string"},
                "model_name": {"type": "string"},
                "rhs": {"type": "object"},
                "y0": {"type": "object"},
                "events": {"type": "array", "items": {"type": "object"}},
                "fields": {},
                "flux": {},
                "diffusivity": {},
                "source": {},
                "initial": {},
                "bc": {},
                "domain": {"type": "array", "items": {"type": "number"}},
                "t_span": {"type": "array", "items": {"type": "number"}},
                "t_end": {"type": "number"},
                "n_cells": {"type": "integer"},
                "method": {"type": "string"},
                "reconstruction": {"type": "string"},
                "dt": {"type": "number"},
                "cfl": {"type": "number"},
                "n_out": {"type": "integer"},
                "params": {"type": "object"},
                "T_wall": {"type": "number"},
                "T_inf": {"type": "number"},
                "T_f": {"type": "number"},
                "h_c": {"type": "number"},
                "delta0": {"type": "number"},
                "mdot_imp": {"type": "number"},
                "tau_drain": {"type": "number"},
                "shedding_events": {"type": "array", "items": {"type": "object"}},
                "relations": {"type": "array", "items": {"type": "string"}},
                "dims": {"type": "object"},
                "target": {"type": "string"},
                "wrt": {"type": "string"},
                "sweep": {"type": "object"},
                "values": {"type": "object"},
                "family": {"type": "object"},
                "plot_path": {"type": "string"},
                "csv_path": {"type": "string"},
                "max_preview_points": {"type": "integer"},
                "record_run": {"type": "boolean"},
            },
        },
    },
    {
        "name": "rosolver_verify",
        "description": "Run code and solution verification: Roache/Celik GCI (internal or external CSV), SymPy MMS, timestep convergence, PDE/ODE conservation audits, stiffness, or Aitken co-simulation.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": [
                        "gci",
                        "mms",
                        "timestep",
                        "conservation_pde",
                        "conservation_ode",
                        "stiffness",
                        "coupling",
                    ],
                },
                "spacings": {"type": "array", "items": {"type": "number"}},
                "quantities": {"type": "array", "items": {"type": "number"}},
                "csv_path": {"type": "string"},
                "spacing_col": {"type": "string"},
                "qoi_col": {"type": "string"},
                "formal_order": {"type": "number"},
                "cells": {"type": "array", "items": {"type": "integer"}},
                "dts": {"type": "array", "items": {"type": "number"}},
                "exact": {},
                "bc_kind": {"type": "string", "enum": ["periodic", "dirichlet"]},
                "fields": {},
                "flux": {},
                "diffusivity": {},
                "source": {},
                "domain": {"type": "array", "items": {"type": "number"}},
                "initial": {},
                "bc": {},
                "n_cells": {"type": "integer"},
                "t_end": {"type": "number"},
                "t_span": {"type": "array", "items": {"type": "number"}},
                "scheme": {"type": "string"},
                "reconstruction": {"type": "string"},
                "rhs": {"type": "object"},
                "sys_a_rhs": {"type": "object"},
                "sys_b_rhs": {"type": "object"},
                "y0": {"type": "object"},
                "state": {},
                "invariants": {"type": "object"},
                "params": {"type": "object"},
                "tol": {"type": "number"},
                "dt_macro": {"type": "number"},
                "mode": {"type": "string", "enum": ["loose", "strong"]},
                "project_dir": {"type": "string"},
                "model_yaml": {"type": "string"},
                "model_path": {"type": "string"},
                "model_name": {"type": "string"},
                "plot_path": {"type": "string"},
                "record_run": {"type": "boolean"},
            },
            "required": ["method"],
        },
    },
    {
        "name": "rosolver_validate_results",
        "description": "Compare simulation predictions (with u_num) against experimental datasets (with u_exp) using ASME V&V 20-style combined validation uncertainty bands.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "simulation": {"type": "array", "items": {"type": "number"}},
                "experiment": {"type": "array", "items": {"type": "number"}},
                "coordinates": {"type": "array", "items": {"type": "number"}},
                "sim_csv": {"type": "string"},
                "exp_csv": {"type": "string"},
                "project_dir": {"type": "string"},
                "run_id": {"type": "string"},
                "exp_dataset": {"type": "string"},
                "field": {"type": "string"},
                "unit": {"type": "string"},
                "coord": {"type": "string"},
                "u_num": {},
                "u_exp": {},
                "u_input": {},
                "coverage_factor": {"type": "number"},
                "error_tol": {"type": "number"},
                "study_name": {"type": "string"},
                "plot_path": {"type": "string"},
                "record_run": {"type": "boolean"},
            },
        },
    },
    {
        "name": "rosolver_project",
        "description": "Manage persistent SHA-256 hash-verified Rosolver Project workspaces, datasets, append-only execution ledgers, and deterministic run replay.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "init",
                        "info",
                        "verify_integrity",
                        "add_model",
                        "add_dataset",
                        "ledger",
                        "get_run",
                        "replay_run",
                    ],
                },
                "project_dir": {"type": "string"},
                "name": {"type": "string"},
                "description": {"type": "string"},
                "model_yaml": {"type": "string"},
                "model_dict": {"type": "object"},
                "model_path": {"type": "string"},
                "filename": {"type": "string"},
                "dataset_path": {"type": "string"},
                "dataset_name": {"type": "string"},
                "category": {"type": "string", "enum": ["experimental", "reference", "processed"]},
                "run_id": {"type": "string"},
            },
            "required": ["action", "project_dir"],
        },
    },
    {
        "name": "rosolver_report",
        "description": "Generate a deterministic ScientificReport (Markdown, JSON, or YAML + sidecar manifest) from a Rosolver Project workspace.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_dir": {"type": "string"},
                "output": {"type": "string"},
                "title": {"type": "string"},
                "report_type": {"type": "string", "enum": list(REPORT_TYPES)},
                "description": {"type": "string"},
                "run_ids": {"type": "array", "items": {"type": "string"}},
                "figures": {"type": "array", "items": {"type": "object"}},
                "also_manifest": {"type": "boolean"},
            },
            "required": ["project_dir"],
        },
    },
]

TOOL_HANDLERS: Dict[str, Callable[[Mapping[str, Any]], Dict[str, Any]]] = {
    "rosolver_capabilities": tool_capabilities,
    "rosolver_check_dimensions": tool_check_dimensions,
    "rosolver_formulate": tool_formulate,
    "rosolver_validate_model": tool_validate_model,
    "rosolver_calibrate": tool_calibrate,
    "rosolver_simulate": tool_simulate,
    "rosolver_verify": tool_verify,
    "rosolver_validate_results": tool_validate_results,
    "rosolver_project": tool_project,
    "rosolver_report": tool_report,
}


# ---------------------------------------------------------------------------
# MCP Resources & Prompts
# ---------------------------------------------------------------------------

_MODEL_YAML_EXAMPLE = """model:
  name: heat_conduction_1d
  version: "1.0"
  domain: heat_transfer
  description: 1D transient heat conduction in a slab
variables:
  independent:
    - { name: x, symbol: x, unit: m }
    - { name: t, symbol: t, unit: s }
  dependent:
    - { name: T, symbol: T, unit: K }
parameters:
  - { name: alpha, symbol: alpha, unit: m^2/s, status: fixed, value: 0.05 }
equations:
  - { id: heat_pde, type: pde, expression: "dT/dt = alpha * d2T/dx2" }
conditions:
  initial:
    - { kind: initial, field: T, expression: "T(x,0) = sin(pi * x)" }
  boundary:
    - { kind: dirichlet, location: left, field: T, value: 0.0 }
    - { kind: dirichlet, location: right, field: T, value: 0.0 }
numerics:
  domain: [0.0, 1.0]
  n_cells: 64
  time_integrator: cn
  reconstruction: linear
"""


def list_mcp_resources() -> List[Dict[str, Any]]:
    return [
        {
            "uri": "rosolver://schema/model",
            "name": "Rosolver Model YAML Example & Schema",
            "description": "Reference declarative Model YAML structure for ODE, 1D PDE, algebraic, and Stefan models.",
            "mimeType": "text/yaml",
        },
        {
            "uri": "rosolver://templates/catalog",
            "name": "Registered Discovery Templates",
            "description": "Catalog of pluggable symbolic/Pi-group candidate equation templates.",
            "mimeType": "application/json",
        },
        {
            "uri": "rosolver://units/supported",
            "name": "Supported SI Units & Base Dimensions",
            "description": "SI base dimensions and derived physical units recognized by Dimension.from_unit.",
            "mimeType": "application/json",
        },
        {
            "uri": "rosolver://status/taxonomy",
            "name": "VerificationStatus 5-State Taxonomy",
            "description": "Definitions of Passed, Failed, Inconclusive, Warning, and NotAssessed plus non-certification notice.",
            "mimeType": "application/json",
        },
    ]


def read_mcp_resource(uri: str) -> Tuple[str, str]:
    if uri == "rosolver://schema/model":
        return "text/yaml", _MODEL_YAML_EXAMPLE
    if uri == "rosolver://templates/catalog":
        import json

        return (
            "application/json",
            json.dumps(
                {
                    "templates": [
                        {
                            "name": getattr(t, "name", str(t)),
                            "description": getattr(t, "description", ""),
                        }
                        for t in list_templates()
                    ]
                },
                indent=2,
            ),
        )
    if uri == "rosolver://units/supported":
        import json

        return (
            "application/json",
            json.dumps(
                {
                    "base_dimensions": ["M", "L", "T", "I", "Theta", "N", "J", "rad"],
                    "si_units": {k: v for k, v in sorted(SI_UNIT_TO_DIM.items())},
                },
                indent=2,
            ),
        )
    if uri == "rosolver://status/taxonomy":
        import json

        return (
            "application/json",
            json.dumps(
                {
                    "statuses": {
                        "Passed": "All numerical, convergence, or uncertainty criteria satisfied.",
                        "Failed": "Convergence failure, tolerance breach, or SHA-256 hash mismatch.",
                        "Inconclusive": "Oscillatory or indeterminate Celik/GCI order extraction.",
                        "Warning": "Incomplete uncertainty budget (e.g. missing u_num) or conditioning warning.",
                        "NotAssessed": "Verification/validation check has not been executed.",
                    },
                    "scope_note": "Follows ASME V&V 20-2009 / Roache / Celik methodology for traceability; not formal regulatory certification.",
                },
                indent=2,
            ),
        )
    raise KeyError(f"Unknown resource URI: '{uri}'")


def list_mcp_prompts() -> List[Dict[str, Any]]:
    return [
        {
            "name": "formulate_and_verify_model",
            "description": "Guided scientific workflow: draft a dimensionally verified Model, simulate, run MMS/GCI/conservation V&V, and produce a ScientificReport.",
            "arguments": [
                {"name": "problem_description", "description": "Physical problem and governing assumptions.", "required": True},
                {"name": "project_dir", "description": "Target directory for the persistent Project.", "required": False},
            ],
        },
        {
            "name": "audit_external_solver",
            "description": "Audit external CFD/FEM/research CSV outputs with Roache/Celik GCI and experimental ValidationStudy.",
            "arguments": [
                {"name": "csv_path", "description": "Path to external mesh-refinement CSV.", "required": True},
                {"name": "qoi_col", "description": "Quantity of interest column name.", "required": True},
            ],
        },
    ]


def get_mcp_prompt(name: str, arguments: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    args = dict(arguments or {})
    if name == "formulate_and_verify_model":
        prob = args.get("problem_description", "<describe physical system>")
        pdir = args.get("project_dir", "outputs/mcp_study")
        text = (
            f"Model and verify the following physical problem using Rosolver MCP tools:\n\n"
            f"Problem: {prob}\nTarget project_dir: {pdir}\n\n"
            "Required sequence:\n"
            "1. Call `rosolver_check_dimensions` on every governing equation and parameter unit.\n"
            "2. Initialize workspace via `rosolver_project` (`action='init'`) and validate the YAML via `rosolver_validate_model` (`register_in_project=true`).\n"
            "3. Solve via `rosolver_simulate` (`solver='model'`).\n"
            "4. Verify discretisation & conservation via `rosolver_verify` (`gci`, `mms` if PDE, and `conservation_pde`/`conservation_ode`).\n"
            "5. Generate a traceable report via `rosolver_report` and report any `Failed`, `Inconclusive`, or `Warning` statuses verbatim."
        )
        return {
            "description": "End-to-end deterministic modeling & V&V workflow",
            "messages": [{"role": "user", "content": {"type": "text", "text": text}}],
        }
    if name == "audit_external_solver":
        csv_p = args.get("csv_path", "mesh_study.csv")
        qoi_c = args.get("qoi_col", "qoi")
        text = (
            f"Audit the external solver mesh convergence dataset at `{csv_p}` (QoI column `{qoi_c}`):\n"
            "1. Call `rosolver_verify` with `method='gci'`, `csv_path`, and `qoi_col`.\n"
            "2. Inspect observed order `p`, `gci_fine_21`, `asymptotic_ratio`, `regime`, and `status`.\n"
            "3. If experimental data is available, call `rosolver_validate_results` passing `u_num` derived from `gci_fine_21`."
        )
        return {
            "description": "External solver GCI & validation audit",
            "messages": [{"role": "user", "content": {"type": "text", "text": text}}],
        }
    raise KeyError(f"Unknown prompt name: '{name}'")
