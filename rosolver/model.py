"""Validated scientific model specification, safe YAML/JSON I/O & unified solve dispatch (ENG_SPEC.md P1.2, P1.4)."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import sympy as sp
import yaml

from ._parsing import names as expr_names, parse_expr, real_symbol
from .dimensions import Dimension
from .numerical import NumericalSolver
from .odesys import EventSpec, ODESystem
from .pde import BC, ConservationLaw1D
from .phasechange import Stefan1D
from .results import StandardResult, standardize_result
from .symbolic import SymbolicSolver
from .system import System

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ODE_LHS_RE = re.compile(r"^\s*d([A-Za-z_][A-Za-z0-9_]*)\s*/\s*d([A-Za-z_][A-Za-z0-9_]*)\s*$")
_IC_LHS_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)(?:\s*\([^)]*\))?\s*$")
_BC_DERIV_LHS_RE = re.compile(r"^\s*d([A-Za-z_][A-Za-z0-9_]*)\s*/\s*d([A-Za-z_][A-Za-z0-9_]*)\s*$")


class ModelValidationError(ValueError):
    """Raised when a scientific `Model` specification fails schema, unit, or DAG checks."""


@dataclass
class VariableSpec:
    """Independent or dependent variable declaration with unit metadata."""

    name: str
    symbol: Optional[str] = None
    role: str = "dependent"  # "independent" or "dependent"
    unit: str = "1"
    description: str = ""
    positive: Optional[bool] = None
    dim: Dimension = field(init=False)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ModelValidationError("Variable name cannot be empty.")
        if self.symbol is None:
            self.symbol = self.name
        if not _IDENT_RE.match(self.symbol):
            raise ModelValidationError(f"Invalid variable symbol '{self.symbol}'.")
        if self.role not in ("independent", "dependent"):
            raise ModelValidationError(
                f"Invalid variable role '{self.role}' for '{self.name}'; choose 'independent' or 'dependent'."
            )
        try:
            self.dim = Dimension.from_unit(self.unit)
        except Exception as exc:
            raise ModelValidationError(f"Invalid unit '{self.unit}' for variable '{self.name}': {exc}") from exc

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"name": self.name, "symbol": self.symbol, "unit": self.unit}
        if self.description:
            out["description"] = self.description
        if self.positive is not None:
            out["positive"] = self.positive
        return out


@dataclass
class ParameterSpec:
    """Model parameter declaration with fixed/unknown status, unit, and optional bounds."""

    name: str
    symbol: Optional[str] = None
    value: Optional[float] = None
    unit: str = "1"
    status: str = "fixed"  # "fixed" or "unknown"
    bounds: Optional[Tuple[float, float]] = None
    description: str = ""
    dim: Dimension = field(init=False)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ModelValidationError("Parameter name cannot be empty.")
        if self.symbol is None:
            self.symbol = self.name
        if not _IDENT_RE.match(self.symbol):
            raise ModelValidationError(f"Invalid parameter symbol '{self.symbol}'.")
        if self.status not in ("fixed", "unknown"):
            raise ModelValidationError(
                f"Invalid parameter status '{self.status}' for '{self.name}'; choose 'fixed' or 'unknown'."
            )
        if self.bounds is not None:
            if len(self.bounds) != 2 or float(self.bounds[0]) >= float(self.bounds[1]):
                raise ModelValidationError(
                    f"Invalid bounds {self.bounds!r} for parameter '{self.name}'; require low < high."
                )
            self.bounds = (float(self.bounds[0]), float(self.bounds[1]))
        if self.value is not None:
            self.value = float(self.value)
        try:
            self.dim = Dimension.from_unit(self.unit)
        except Exception as exc:
            raise ModelValidationError(f"Invalid unit '{self.unit}' for parameter '{self.name}': {exc}") from exc

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "name": self.name,
            "symbol": self.symbol,
            "unit": self.unit,
            "status": self.status,
        }
        if self.value is not None:
            out["value"] = self.value
        if self.bounds is not None:
            out["bounds"] = list(self.bounds)
        if self.description:
            out["description"] = self.description
        return out


@dataclass
class EquationSpec:
    """Governing, constitutive, or algebraic equation declaration."""

    id: str
    expression: str = ""
    type: str = "ode"  # "ode", "pde", "algebraic", "constitutive", "stefan"
    field: Optional[str] = None
    flux: Optional[str] = None
    diffusivity: Optional[str] = None
    source: Optional[str] = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise ModelValidationError("Equation id cannot be empty.")
        if self.type not in ("ode", "pde", "algebraic", "constitutive", "stefan"):
            raise ModelValidationError(
                f"Unsupported equation type '{self.type}' for '{self.id}'."
            )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"id": self.id, "type": self.type}
        if self.expression:
            out["expression"] = self.expression
        for key in ("field", "flux", "diffusivity", "source", "description"):
            val = getattr(self, key)
            if val is not None and val != "":
                out[key] = val
        return out


@dataclass
class ConditionSpec:
    """Initial or boundary condition specification."""

    kind: str  # "initial", "dirichlet", "neumann", "outflow", "periodic"
    expression: str = ""
    location: Optional[str] = None  # "left", "right", "x_min", "x_max"
    field: Optional[str] = None
    value: Any = None

    def __post_init__(self) -> None:
        self.kind = self.kind.lower()
        if self.kind not in ("initial", "dirichlet", "neumann", "outflow", "periodic"):
            raise ModelValidationError(f"Unsupported condition kind '{self.kind}'.")

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"kind": self.kind}
        if self.location is not None:
            out["location"] = self.location
        if self.field is not None:
            out["field"] = self.field
        if self.expression:
            out["expression"] = self.expression
        if self.value is not None:
            out["value"] = self.value
        return out


@dataclass
class NumericsConfig:
    """Solver-independent numerical configuration hints."""

    solver: str = "auto"
    time_integrator: str = "RK45"
    rtol: float = 1e-6
    atol: float = 1e-9
    domain: Optional[Tuple[float, float]] = None
    n_cells: int = 100
    reconstruction: str = "linear"
    dt: Optional[float] = None
    cfl: float = 0.4
    n_out: int = 100

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "solver": self.solver,
            "time_integrator": self.time_integrator,
            "tolerances": {"rtol": self.rtol, "atol": self.atol},
            "n_cells": self.n_cells,
            "reconstruction": self.reconstruction,
            "cfl": self.cfl,
            "n_out": self.n_out,
        }
        if self.domain is not None:
            out["domain"] = list(self.domain)
        if self.dt is not None:
            out["dt"] = self.dt
        return out

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "NumericsConfig":
        if not data:
            return cls()
        tols = data.get("tolerances") or {}
        dom = data.get("domain")
        return cls(
            solver=str(data.get("solver", "auto")),
            time_integrator=str(data.get("time_integrator", "RK45")),
            rtol=float(tols.get("rtol", data.get("rtol", 1e-6))),
            atol=float(tols.get("atol", data.get("atol", 1e-9))),
            domain=(float(dom[0]), float(dom[1])) if dom is not None else None,
            n_cells=int(data.get("n_cells", 100)),
            reconstruction=str(data.get("reconstruction", "linear")),
            dt=None if data.get("dt") is None else float(data["dt"]),
            cfl=float(data.get("cfl", 0.4)),
            n_out=int(data.get("n_out", 100)),
        )


def _norm_location(loc: Optional[str]) -> Optional[str]:
    if loc is None:
        return None
    low = loc.strip().lower()
    if low in ("left", "x_min", "xmin", "min"):
        return "left"
    if low in ("right", "x_max", "xmax", "max"):
        return "right"
    return low


@dataclass
class Model:
    """
    Central validated representation of a scientific model (ENG_SPEC.md P1.2).

    Decouples physical variables, units, parameters, governing equations, dependencies,
    and conditions from any concrete numerical solver backend.
    """

    name: str
    version: str = "0.1"
    description: str = ""
    domain: str = "general"
    independent_variables: List[VariableSpec] = field(default_factory=list)
    dependent_variables: List[VariableSpec] = field(default_factory=list)
    parameters: List[ParameterSpec] = field(default_factory=list)
    equations: List[EquationSpec] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    rate_identities: List[Tuple[str, str, str]] = field(default_factory=list)
    initial_conditions: List[ConditionSpec] = field(default_factory=list)
    boundary_conditions: List[ConditionSpec] = field(default_factory=list)
    numerics: NumericsConfig = field(default_factory=NumericsConfig)
    assumptions: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.validate(strict=True)

    # ------------------------------------------------------------------ inspection
    def all_variables(self) -> List[VariableSpec]:
        return list(self.independent_variables) + list(self.dependent_variables)

    def variable_symbols(self) -> List[str]:
        return [v.symbol or v.name for v in self.all_variables()]

    def parameter_symbols(self) -> List[str]:
        return [p.symbol or p.name for p in self.parameters]

    def unit_map(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for v in self.all_variables():
            out[v.symbol or v.name] = v.unit
        for p in self.parameters:
            out[p.symbol or p.name] = p.unit
        return out

    def _dim_map(self) -> Dict[str, Dimension]:
        out: Dict[str, Dimension] = {}
        for v in self.all_variables():
            out[v.symbol or v.name] = v.dim
        for p in self.parameters:
            out[p.symbol or p.name] = p.dim
        return out

    def _has_physical_units(self) -> bool:
        return any(not d.is_dimensionless for d in self._dim_map().values())

    def _check_dim_pair(self, eq_id: str, lhs: sp.Expr, rhs: sp.Expr, errors: List[str]) -> None:
        if not self._has_physical_units():
            return
        from .claims.scales import _expr_dimension

        dim_map = self._dim_map()
        if not all(s.name in dim_map for s in (lhs.free_symbols | rhs.free_symbols) if isinstance(s, sp.Symbol)):
            return
        d_l, e_l = _expr_dimension(lhs, dim_map)
        d_r, e_r = _expr_dimension(rhs, dim_map)
        for msg in e_l + e_r:
            errors.append(f"Dimensional inconsistency in equation '{eq_id}': {msg}")
        if not (e_l or e_r) and lhs != 0 and rhs != 0 and d_l is not None and d_r is not None and d_l != d_r:
            errors.append(
                f"Dimensional mismatch in equation '{eq_id}': LHS [{d_l}] != RHS [{d_r}]."
            )

    def _check_expr_has_dim(
        self,
        eq_id: str,
        expr: sp.Expr,
        expected: Dimension,
        dim_map: Mapping[str, Dimension],
        errors: List[str],
    ) -> None:
        if not self._has_physical_units() or expr == 0:
            return
        from .claims.scales import _expr_dimension

        if not all(s.name in dim_map for s in expr.free_symbols if isinstance(s, sp.Symbol)):
            return
        d_e, errs = _expr_dimension(expr, dim_map)
        for msg in errs:
            errors.append(f"Dimensional inconsistency in equation '{eq_id}': {msg}")
        if not errs and d_e is not None and d_e != expected:
            errors.append(
                f"Dimensional mismatch in equation '{eq_id}': expected [{expected}], got [{d_e}]."
            )

    def parameter_values(self, overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        vals: Dict[str, Any] = {}
        for p in self.parameters:
            sym = p.symbol or p.name
            if p.value is not None:
                vals[sym] = p.value
        if overrides:
            vals.update(overrides)
        return vals

    # ------------------------------------------------------------------ validation
    def validate(self, strict: bool = True) -> List[str]:
        errors: List[str] = []
        if not self.name or not self.name.strip():
            errors.append("Model name cannot be empty.")

        seen: Dict[str, str] = {}
        for v in self.all_variables():
            sym = v.symbol or v.name
            if sym in seen:
                errors.append(f"Duplicate symbol '{sym}' declared in variables/parameters.")
            seen[sym] = v.role
        for p in self.parameters:
            sym = p.symbol or p.name
            if sym in seen:
                errors.append(f"Duplicate symbol '{sym}' declared in variables/parameters.")
            seen[sym] = "parameter"
            if p.status == "fixed" and p.value is None:
                errors.append(f"Fixed parameter '{sym}' must declare a numeric value.")

        if self.dependencies:
            try:
                self.to_system().validate_acyclic()
            except ValueError as exc:
                errors.append(str(exc))

        for eq in self.equations:
            if eq.type in ("algebraic", "constitutive") and eq.expression:
                rhs_txt = eq.expression.split("=", 1)[1] if "=" in eq.expression else eq.expression
                try:
                    rhs_e = parse_expr(rhs_txt)
                    if "=" in eq.expression:
                        lhs_e = parse_expr(eq.expression.split("=", 1)[0])
                        self._check_dim_pair(eq.id, lhs_e, rhs_e, errors)
                except Exception as exc:
                    errors.append(f"Invalid expression in equation '{eq.id}': {exc}")
            elif eq.type == "ode" and eq.expression:
                try:
                    st_name, ind_name, rhs_e = self._parse_ode_equation(eq)
                    dim_map = self._dim_map()
                    if st_name in dim_map and ind_name in dim_map:
                        lhs_dim = dim_map[st_name] / dim_map[ind_name]
                        self._check_expr_has_dim(eq.id, rhs_e, lhs_dim, dim_map, errors)
                except Exception as exc:
                    errors.append(f"Invalid ODE equation '{eq.id}': {exc}")
            elif eq.type == "pde":
                try:
                    dim_map = self._dim_map()
                    if eq.expression and not (eq.flux or eq.diffusivity or eq.source):
                        fld, x_n, t_n, flx, diff_e, src_e = self._parse_pde_equation(eq)
                    else:
                        fld = eq.field or (self.dependent_variables[0].symbol or self.dependent_variables[0].name if self.dependent_variables else "")
                        indep_syms = [v.symbol or v.name for v in self.independent_variables]
                        x_n = indep_syms[0] if len(indep_syms) >= 1 else "x"
                        t_n = indep_syms[1] if len(indep_syms) >= 2 else "t"
                        flx = parse_expr(eq.flux) if eq.flux else sp.Integer(0)
                        diff_e = parse_expr(eq.diffusivity) if eq.diffusivity else sp.Integer(0)
                        src_e = parse_expr(eq.source) if eq.source else sp.Integer(0)
                    if fld in dim_map and x_n in dim_map and t_n in dim_map:
                        rate_dim = dim_map[fld] / dim_map[t_n]
                        if flx != 0:
                            self._check_expr_has_dim(f"{eq.id}.flux", flx, rate_dim * dim_map[x_n], dim_map, errors)
                        if diff_e != 0:
                            self._check_expr_has_dim(
                                f"{eq.id}.diffusivity", diff_e, (dim_map[x_n] ** 2) / dim_map[t_n], dim_map, errors
                            )
                        if src_e != 0:
                            self._check_expr_has_dim(f"{eq.id}.source", src_e, rate_dim, dim_map, errors)
                except Exception as exc:
                    errors.append(f"Invalid PDE equation '{eq.id}': {exc}")

        for cond in self.initial_conditions + self.boundary_conditions:
            if cond.kind in ("outflow", "periodic"):
                continue
            if cond.value is not None and isinstance(cond.value, str):
                try:
                    parse_expr(cond.value)
                except Exception as exc:
                    errors.append(f"Invalid condition value '{cond.value}': {exc}")
            elif cond.expression:
                rhs_txt = cond.expression.split("=", 1)[1] if "=" in cond.expression else cond.expression
                if rhs_txt.strip():
                    try:
                        # Allow symbolic IC placeholders like T_initial(x) during declarative validation
                        if not re.fullmatch(r"\s*[A-Za-z_][A-Za-z0-9_]*\([^)]*\)\s*", rhs_txt):
                            parse_expr(rhs_txt)
                    except Exception as exc:
                        errors.append(f"Invalid condition expression '{cond.expression}': {exc}")

        if strict and errors:
            raise ModelValidationError("; ".join(errors))
        return errors

    # ------------------------------------------------------------------ serialization
    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": {
                "name": self.name,
                "version": self.version,
                "description": self.description,
                "domain": self.domain,
            },
            "variables": {
                "independent": [v.to_dict() for v in self.independent_variables],
                "dependent": [v.to_dict() for v in self.dependent_variables],
            },
            "parameters": [p.to_dict() for p in self.parameters],
            "equations": [eq.to_dict() for eq in self.equations],
            "conditions": {
                "initial": [c.to_dict() for c in self.initial_conditions],
                "boundary": [c.to_dict() for c in self.boundary_conditions],
            },
            "numerics": self.numerics.to_dict(),
        }
        if self.dependencies:
            payload["dependencies"] = list(self.dependencies)
        if self.rate_identities:
            payload["rate_identities"] = [list(ri) for ri in self.rate_identities]
        if self.assumptions:
            payload["assumptions"] = list(self.assumptions)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Model":
        if not isinstance(data, Mapping):
            raise ModelValidationError("Model specification must be a mapping.")
        meta = data.get("model") or data
        if not isinstance(meta, Mapping):
            raise ModelValidationError("Invalid 'model' metadata block.")
        name = str(meta.get("name", "")).strip()
        if not name:
            raise ModelValidationError("Model specification must include 'model.name'.")

        vars_block = data.get("variables") or {}
        indep_list = [
            VariableSpec(
                name=str(item["name"]),
                symbol=item.get("symbol"),
                role="independent",
                unit=str(item.get("unit", "1")),
                description=str(item.get("description", "")),
                positive=item.get("positive"),
            )
            for item in (vars_block.get("independent") or [])
        ]
        dep_list = [
            VariableSpec(
                name=str(item["name"]),
                symbol=item.get("symbol"),
                role="dependent",
                unit=str(item.get("unit", "1")),
                description=str(item.get("description", "")),
                positive=item.get("positive"),
            )
            for item in (vars_block.get("dependent") or [])
        ]

        params_list = [
            ParameterSpec(
                name=str(item["name"]),
                symbol=item.get("symbol"),
                value=item.get("value"),
                unit=str(item.get("unit", "1")),
                status=str(item.get("status", "fixed")),
                bounds=tuple(item["bounds"]) if item.get("bounds") is not None else None,  # type: ignore[arg-type]
                description=str(item.get("description", "")),
            )
            for item in (data.get("parameters") or [])
        ]

        eqs_list = [
            EquationSpec(
                id=str(item["id"]),
                expression=str(item.get("expression", "")),
                type=str(item.get("type", "ode")),
                field=item.get("field"),
                flux=item.get("flux"),
                diffusivity=item.get("diffusivity"),
                source=item.get("source"),
                description=str(item.get("description", "")),
            )
            for item in (data.get("equations") or [])
        ]

        conds_block = data.get("conditions") or {}
        ic_list: List[ConditionSpec] = []
        for item in (conds_block.get("initial") or []):
            ic_list.append(
                ConditionSpec(
                    kind="initial",
                    expression=str(item.get("expression", "")),
                    field=item.get("field"),
                    value=item.get("value"),
                )
            )

        bc_list: List[ConditionSpec] = []
        for item in (conds_block.get("boundary") or []):
            expr = str(item.get("expression", ""))
            kind = item.get("kind")
            if kind is None:
                lhs = expr.split("=", 1)[0] if "=" in expr else expr
                if "outflow" in expr.lower():
                    kind = "outflow"
                elif "periodic" in expr.lower():
                    kind = "periodic"
                elif _BC_DERIV_LHS_RE.match(lhs):
                    kind = "neumann"
                else:
                    kind = "dirichlet"
            bc_list.append(
                ConditionSpec(
                    kind=str(kind),
                    expression=expr,
                    location=item.get("location"),
                    field=item.get("field"),
                    value=item.get("value"),
                )
            )

        rate_ids = [
            (str(r[0]), str(r[1]), str(r[2]))
            for r in (data.get("rate_identities") or [])
        ]

        return cls(
            name=name,
            version=str(meta.get("version", "0.1")),
            description=str(meta.get("description", "")),
            domain=str(meta.get("domain", "general")),
            independent_variables=indep_list,
            dependent_variables=dep_list,
            parameters=params_list,
            equations=eqs_list,
            dependencies=[str(d) for d in (data.get("dependencies") or [])],
            rate_identities=rate_ids,
            initial_conditions=ic_list,
            boundary_conditions=bc_list,
            numerics=NumericsConfig.from_dict(data.get("numerics")),
            assumptions=[str(a) for a in (data.get("assumptions") or [])],
        )

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_yaml(cls, text: str) -> "Model":
        try:
            loaded = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ModelValidationError(f"Unsafe or invalid YAML specification: {exc}") from exc
        return cls.from_dict(loaded)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "Model":
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ModelValidationError(f"Invalid JSON specification: {exc}") from exc
        return cls.from_dict(loaded)

    def save(self, path: Union[str, Path]) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".json":
            out.write_text(self.to_json(), encoding="utf-8")
        else:
            out.write_text(self.to_yaml(), encoding="utf-8")
        return out

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Model":
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"Model file not found: {src}")
        text = src.read_text(encoding="utf-8")
        if src.suffix.lower() == ".json":
            return cls.from_json(text)
        return cls.from_yaml(text)

    def content_hash(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------ converters
    def to_system(self) -> System:
        dims: Dict[str, Dimension] = {}
        constants: List[str] = []
        for v in self.all_variables():
            sym = v.symbol or v.name
            if not v.dim.is_dimensionless or v.unit in ("1", "dimensionless"):
                dims[sym] = v.dim
        for p in self.parameters:
            sym = p.symbol or p.name
            dims[sym] = p.dim
            constants.append(sym)
        if not self.dependencies:
            sys = System()
            for v in self.all_variables():
                sys.var(v.symbol or v.name, dim=v.dim, description=v.description, positive=v.positive if v.positive is not None else True)
            for p in self.parameters:
                sys.const(p.symbol or p.name, dim=p.dim, description=p.description)
            return sys
        return System.from_spec(
            relations=self.dependencies,
            dims=dims,
            constants=constants,
            rate_identities=self.rate_identities or None,
        )

    def _parse_ode_equation(self, eq: EquationSpec) -> Tuple[str, str, sp.Expr]:
        if "=" not in eq.expression:
            raise ModelValidationError(f"ODE equation '{eq.id}' must have '=' (e.g. 'dy/dt = -k * y').")
        lhs, rhs = eq.expression.split("=", 1)
        m = _ODE_LHS_RE.match(lhs)
        if not m:
            raise ModelValidationError(f"ODE LHS '{lhs.strip()}' must match 'd<state>/d<indep>'.")
        state, indep = m.group(1), m.group(2)
        return state, indep, parse_expr(rhs)

    def _parse_pde_equation(self, eq: EquationSpec) -> Tuple[str, str, str, sp.Expr, sp.Expr, sp.Expr]:
        if "=" not in eq.expression:
            raise ModelValidationError(f"PDE equation '{eq.id}' must have '=' (e.g. 'dT/dt = alpha * d2T/dx2').")
        lhs, rhs_raw = eq.expression.split("=", 1)
        m = _ODE_LHS_RE.match(lhs)
        if not m:
            raise ModelValidationError(f"PDE LHS '{lhs.strip()}' must match 'd<field>/d<t>'.")
        fld, t_name = m.group(1), m.group(2)

        # Infer spatial coordinate from independent variables (other than t_name) or regex d2<fld>/d<x>2
        space_candidates = [v.symbol or v.name for v in self.independent_variables if (v.symbol or v.name) != t_name]
        x_match = re.search(rf"d2{re.escape(fld)}\s*/\s*d([A-Za-z_][A-Za-z0-9_]*)2", rhs_raw)
        if not x_match:
            x_match = re.search(rf"d{re.escape(fld)}\s*/\s*d([A-Za-z_][A-Za-z0-9_]*)", rhs_raw)
        x_name = x_match.group(1) if x_match else (space_candidates[0] if space_candidates else "x")

        d2_tok = f"ro_d2_{fld}_{x_name}"
        d1_tok = f"ro_d1_{fld}_{x_name}"
        normalized = re.sub(
            rf"\bd2{re.escape(fld)}\s*/\s*d{re.escape(x_name)}2\b", d2_tok, rhs_raw
        )
        normalized = re.sub(
            rf"\bd{re.escape(fld)}\s*/\s*d{re.escape(x_name)}\b", d1_tok, normalized
        )
        expr = parse_expr(normalized)

        d2_sym = real_symbol(d2_tok)
        d1_sym = real_symbol(d1_tok)
        fld_sym = real_symbol(fld)
        x_sym = real_symbol(x_name)
        diff_expr = sp.simplify(sp.diff(expr, d2_sym))
        adv_rhs = sp.simplify(sp.diff(expr, d1_sym))
        if diff_expr.has(d2_sym) or diff_expr.has(d1_sym) or adv_rhs.has(d2_sym) or adv_rhs.has(d1_sym):
            raise ModelValidationError(
                f"PDE equation '{eq.id}' must be linear in first/second spatial derivatives or specify flux/diffusivity explicitly."
            )
        if adv_rhs != 0 and sp.simplify(sp.diff(adv_rhs, x_sym)) != 0:
            raise ModelValidationError(
                f"PDE equation '{eq.id}' has spatially varying non-divergence advection ({adv_rhs})*d{fld}/d{x_name}; "
                "specify conservative 'flux' and 'source' explicitly."
            )
        if diff_expr != 0 and (
            sp.simplify(sp.diff(diff_expr, x_sym)) != 0 or sp.simplify(sp.diff(diff_expr, fld_sym)) != 0
        ):
            raise ModelValidationError(
                f"PDE equation '{eq.id}' has variable/nonlinear non-divergence diffusion ({diff_expr})*d2{fld}/d{x_name}2; "
                "specify conservative 'diffusivity' explicitly."
            )
        flux_expr = sp.Integer(0) if adv_rhs == 0 else sp.integrate(-adv_rhs, fld_sym)
        if flux_expr.has(sp.Integral):
            raise ModelValidationError(
                f"PDE equation '{eq.id}' advection term '{adv_rhs}' could not be integrated into closed-form flux; "
                "specify 'flux' explicitly."
            )
        src_expr = sp.simplify(expr.subs({d2_sym: 0, d1_sym: 0}))
        return fld, x_name, t_name, flux_expr, diff_expr, src_expr

    def to_ode_system(self) -> ODESystem:
        ode_eqs = [eq for eq in self.equations if eq.type == "ode"]
        if not ode_eqs:
            raise ModelValidationError(f"Model '{self.name}' has no ODE equations.")
        rhs_map: Dict[str, sp.Expr] = {}
        indeps: List[str] = []
        for eq in ode_eqs:
            state, indep, rhs_expr = self._parse_ode_equation(eq)
            rhs_map[state] = rhs_expr
            if indep not in indeps:
                indeps.append(indep)
        if len(indeps) != 1:
            raise ModelValidationError(f"Coupled ODE system must share one independent variable; got {indeps}.")
        return ODESystem(rhs_map, indep=indeps[0])

    def to_conservation_law(self) -> ConservationLaw1D:
        pde_eqs = [eq for eq in self.equations if eq.type == "pde"]
        if not pde_eqs:
            raise ModelValidationError(f"Model '{self.name}' has no PDE equations.")
        fields: List[str] = []
        flux_map: Dict[str, Any] = {}
        diff_map: Dict[str, Any] = {}
        src_map: Dict[str, Any] = {}
        x_name: Optional[str] = None
        t_name: Optional[str] = None

        for eq in pde_eqs:
            if eq.flux is not None or eq.diffusivity is not None or eq.source is not None:
                fld = eq.field or (self.dependent_variables[0].symbol or self.dependent_variables[0].name)
                indep_syms = [v.symbol or v.name for v in self.independent_variables]
                x_name = x_name or (indep_syms[0] if len(indep_syms) >= 1 else "x")
                t_name = t_name or (indep_syms[1] if len(indep_syms) >= 2 else "t")
                fields.append(fld)
                flux_map[fld] = eq.flux if eq.flux is not None else 0
                diff_map[fld] = eq.diffusivity if eq.diffusivity is not None else 0
                src_map[fld] = eq.source if eq.source is not None else 0
            else:
                fld, x_n, t_n, flx, diff, src = self._parse_pde_equation(eq)
                x_name = x_name or x_n
                t_name = t_name or t_n
                fields.append(fld)
                flux_map[fld] = flx
                diff_map[fld] = diff
                src_map[fld] = src

        return ConservationLaw1D(
            fields=fields[0] if len(fields) == 1 else fields,
            flux=flux_map,
            diffusivity=diff_map,
            source=src_map,
            x=x_name or "x",
            t=t_name or "t",
        )

    # ------------------------------------------------------------------ unified workflows
    def analyse_symbolically(
        self,
        target: Optional[str] = None,
        wrt: Optional[str] = None,
        **kwargs: Any,
    ) -> StandardResult:
        if not self.dependencies:
            raise ModelValidationError(f"Model '{self.name}' has no declared dependencies for symbolic analysis.")
        t0 = time.perf_counter()
        sys = self.to_system()
        tgt = target or (self.dependent_variables[0].symbol or self.dependent_variables[0].name)
        res = SymbolicSolver(sys).derive(target=tgt, wrt=wrt, **kwargs)
        elapsed = time.perf_counter() - t0
        return standardize_result(
            res,
            kind="symbolic_derivation",
            model_name=self.name,
            model_version=self.version,
            units=self.unit_map(),
            timing_s=elapsed,
            assumptions=list(self.assumptions),
        )

    def resolved_ode_initial_conditions(
        self,
        override: Any = None,
        state_names: Optional[Sequence[str]] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, float]:
        """Resolve ODE initial conditions (both `value:` and parameter-dependent `expression:`) to floats."""
        states = list(state_names) if state_names is not None else [eq_s[0] for eq_s in [self._parse_ode_equation(e) for e in self.equations if e.type == "ode"]]
        raw_map = self._extract_initial_conditions(override, states)
        p_vals = self.parameter_values(params)
        psubs = {real_symbol(k): sp.Float(float(v)) for k, v in p_vals.items() if isinstance(v, (int, float, np.number))}
        resolved: Dict[str, float] = {}
        for s in states:
            if s not in raw_map:
                continue
            item = raw_map[s]
            if isinstance(item, str):
                expr_ic = parse_expr(item).xreplace(psubs)
                resolved[s] = float(expr_ic.evalf())
            else:
                resolved[s] = float(item)
        return resolved

    def _extract_initial_conditions(
        self,
        override: Any,
        state_names: Sequence[str],
    ) -> Dict[str, Any]:
        if isinstance(override, Mapping):
            return dict(override)
        out: Dict[str, Any] = {}
        for cond in self.initial_conditions:
            fld = cond.field
            val = cond.value
            if val is None and cond.expression:
                if "=" in cond.expression:
                    lhs, rhs = cond.expression.split("=", 1)
                    m = _IC_LHS_RE.match(lhs)
                    if fld is None and m:
                        fld = m.group(1)
                    val = rhs.strip()
                else:
                    val = cond.expression.strip()
            if fld is None and len(state_names) == 1:
                fld = state_names[0]
            if fld is not None and val is not None:
                out[fld] = val
        return out

    def _extract_boundary_conditions(
        self,
        override: Any,
        fields: Sequence[str],
    ) -> Any:
        if override is not None:
            return override
        if not self.boundary_conditions:
            return "outflow"
        sides: Dict[str, Dict[str, Tuple[str, Any]]] = {}
        for cond in self.boundary_conditions:
            loc = _norm_location(cond.location)
            if loc not in ("left", "right"):
                continue
            fld = cond.field
            val = cond.value
            if val is None and cond.expression and cond.kind not in ("outflow", "periodic"):
                if "=" in cond.expression:
                    lhs, rhs = cond.expression.split("=", 1)
                    if fld is None:
                        m_d = _BC_DERIV_LHS_RE.match(lhs)
                        if m_d:
                            fld = m_d.group(1)
                        elif _IDENT_RE.match(lhs.strip()):
                            fld = lhs.strip()
                    val = rhs.strip()
                else:
                    val = cond.expression.strip()
            if fld is None and len(fields) == 1:
                fld = fields[0]
            if fld is not None:
                sides.setdefault(loc, {})[fld] = (cond.kind, 0.0 if val is None else val)

        if not sides:
            return "outflow"
        bc_spec: Dict[str, Any] = {}
        for side in ("left", "right"):
            entries = sides.get(side)
            if not entries:
                bc_spec[side] = "outflow"
                continue
            kinds = {k for k, _ in entries.values()}
            if len(kinds) == 1:
                k = next(iter(kinds))
                if k == "periodic":
                    bc_spec[side] = BC.periodic()
                elif k == "outflow":
                    bc_spec[side] = BC.outflow()
                else:
                    val_map = {f: v for f, (_, v) in entries.items()}
                    bc_spec[side] = BC(k, val_map[fields[0]] if len(fields) == 1 else val_map)
            else:
                raise ModelValidationError(f"Mixed BC kinds on '{side}' boundary require explicit BC dict.")
        return bc_spec

    def solve(
        self,
        t_span: Optional[Tuple[float, float]] = None,
        initial_conditions: Any = None,
        boundary_conditions: Any = None,
        method: Optional[str] = None,
        tolerances: Optional[Mapping[str, float]] = None,
        params: Optional[Mapping[str, Any]] = None,
        events: Optional[Sequence[Union[EventSpec, Any]]] = None,
        domain: Optional[Tuple[float, float]] = None,
        n_cells: Optional[int] = None,
        reconstruction: Optional[str] = None,
        dt: Optional[float] = None,
        cfl: Optional[float] = None,
        n_out: Optional[int] = None,
        **kwargs: Any,
    ) -> StandardResult:
        """Unified simulation entry point dispatching across ODE, 1D PDE, and Stefan models."""
        eq_types = {eq.type for eq in self.equations}
        if not eq_types:
            raise ModelValidationError(f"Model '{self.name}' has no governing equations to solve.")

        tols = dict(tolerances or {})
        rtol = float(tols.get("rtol", self.numerics.rtol))
        atol = float(tols.get("atol", self.numerics.atol))
        integ_method = method or self.numerics.time_integrator
        p_vals = self.parameter_values(params)

        t_start = time.perf_counter()
        if "pde" in eq_types:
            law = self.to_conservation_law()
            dom = domain or self.numerics.domain or (0.0, 1.0)
            span = t_span or (0.0, 1.0)
            ic = self._extract_initial_conditions(initial_conditions, law.fields)
            if not ic:
                raise ModelValidationError("PDE solve requires initial conditions.")
            bc = self._extract_boundary_conditions(boundary_conditions, law.fields)
            sol = law.solve(
                domain=dom,
                n_cells=n_cells or self.numerics.n_cells,
                t_end=float(span[1]),
                t0=float(span[0]),
                initial=ic if len(law.fields) > 1 else ic[law.fields[0]],
                bc=bc,
                params=p_vals,
                scheme=integ_method,
                dt=dt if dt is not None else self.numerics.dt,
                cfl=cfl if cfl is not None else self.numerics.cfl,
                reconstruction=reconstruction or self.numerics.reconstruction,
                n_out=n_out or self.numerics.n_out,
                rtol=rtol,
                atol=atol,
            )
            elapsed = time.perf_counter() - t_start
            return standardize_result(
                sol,
                kind="pde_simulation",
                model_name=self.name,
                model_version=self.version,
                solver_config={
                    "solver": "ConservationLaw1D",
                    "method": integ_method,
                    "domain": list(dom),
                    "n_cells": n_cells or self.numerics.n_cells,
                    "rtol": rtol,
                    "atol": atol,
                },
                units=self.unit_map(),
                timing_s=elapsed,
                assumptions=list(self.assumptions),
            )

        if "stefan" in eq_types:
            span = t_span or (0.0, 3600.0)
            st_kwargs = dict(p_vals)
            st_kwargs.update(kwargs)
            st = Stefan1D(**{k: v for k, v in st_kwargs.items() if k != "delta0"})
            delta0 = float(st_kwargs.get("delta0", 1e-5))
            sol = st.solve(
                t_end=float(span[1]),
                t0=float(span[0]),
                delta0=delta0,
                method=integ_method,
                n_out=n_out or self.numerics.n_out,
                dt=dt if dt is not None else self.numerics.dt,
                rtol=rtol,
                atol=atol,
            )
            elapsed = time.perf_counter() - t_start
            return standardize_result(
                sol,
                kind="stefan_simulation",
                model_name=self.name,
                model_version=self.version,
                solver_config={"solver": "Stefan1D", "method": integ_method, "rtol": rtol, "atol": atol},
                units=self.unit_map(),
                timing_s=elapsed,
                assumptions=list(self.assumptions),
            )

        if "ode" in eq_types:
            ode = self.to_ode_system()
            span = t_span or (0.0, 1.0)
            y0_resolved = self.resolved_ode_initial_conditions(initial_conditions, ode.states, params=p_vals)
            missing_ic = [s for s in ode.states if s not in y0_resolved]
            if missing_ic:
                raise ModelValidationError(f"Missing initial condition(s) for ODE state(s) {missing_ic}.")
            traj = ode.solve(
                t_span=span,
                y0=y0_resolved,
                params=p_vals,
                method=integ_method,
                rtol=rtol,
                atol=atol,
                n_out=n_out or self.numerics.n_out,
                dt=dt if dt is not None else self.numerics.dt,
                events=events,
            )
            elapsed = time.perf_counter() - t_start
            return standardize_result(
                traj,
                kind="ode_simulation",
                model_name=self.name,
                model_version=self.version,
                solver_config={"solver": "ODESystem", "method": integ_method, "rtol": rtol, "atol": atol},
                units=self.unit_map(),
                timing_s=elapsed,
                assumptions=list(self.assumptions),
            )

        raise ModelValidationError(f"No solvable equation types in {sorted(eq_types)}.")

    def calibrate(
        self,
        data: Mapping[str, Union[float, Sequence[float], np.ndarray]],
        target: Optional[str] = None,
        **kwargs: Any,
    ) -> StandardResult:
        """Calibrate unknown model parameters against observational `data`."""
        alg_eqs = [eq for eq in self.equations if eq.type in ("algebraic", "constitutive") and eq.expression]
        if not alg_eqs:
            raise ModelValidationError(
                f"Model '{self.name}' requires an 'algebraic' or 'constitutive' equation for calibration."
            )
        eq_spec = alg_eqs[0]
        if "=" in eq_spec.expression:
            lhs_s, rhs_s = eq_spec.expression.split("=", 1)
            lhs_sym = real_symbol(lhs_s.strip())
            rhs_expr = parse_expr(rhs_s)
        else:
            tgt_name = target or (self.dependent_variables[0].symbol or self.dependent_variables[0].name)
            lhs_sym = real_symbol(tgt_name)
            rhs_expr = parse_expr(eq_spec.expression)

        unknown_params = [p for p in self.parameters if p.status == "unknown"]
        free_syms = [real_symbol(p.symbol or p.name) for p in unknown_params]
        fixed_vals = {
            (p.symbol or p.name): float(p.value)
            for p in self.parameters
            if p.status == "fixed" and p.value is not None
        }
        bounds_map = {
            (p.symbol or p.name): p.bounds
            for p in unknown_params
            if p.bounds is not None
        }
        if bounds_map and "bounds" not in kwargs:
            kwargs["bounds"] = bounds_map

        t0 = time.perf_counter()
        fit_res = NumericalSolver.fit(
            sp.Eq(lhs_sym, rhs_expr),
            data=data,
            target_col=target or lhs_sym.name,
            free_params=free_syms if free_syms else None,
            fixed_params=fixed_vals or None,
            **kwargs,
        )
        elapsed = time.perf_counter() - t0
        return standardize_result(
            fit_res,
            kind="calibration",
            model_name=self.name,
            model_version=self.version,
            units=self.unit_map(),
            timing_s=elapsed,
            assumptions=list(self.assumptions),
        )
