"""Pluggable equation template registry for dimensionless Buckingham-Pi candidate laws (ENG_SPEC.md P1.3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import sympy as sp

TemplateBuilder = Callable[[Sequence[sp.Expr], Sequence[int]], Tuple[sp.Expr, List[sp.Symbol]]]


@dataclass(frozen=True)
class EquationTemplate:
    """
    Self-contained candidate equation family over dimensionless Buckingham-Pi groups.

    Defines mathematical form, dimensional constraint, parameter bounds, applicable assumptions,
    and evaluation/differentiation interfaces without coupling to SymbolicSolver internals.
    """

    name: str
    candidate_name: str
    kind: str
    description: str
    builder: TemplateBuilder
    saturating: bool = False
    dimensional_constraint: str = "All Pi arguments must be dimensionless."
    assumptions: Tuple[str, ...] = ()
    param_bounds: Mapping[str, Tuple[float, float]] = field(default_factory=dict)

    def build_factor(
        self,
        pi_groups: Sequence[sp.Expr],
        indices: Optional[Sequence[int]] = None,
    ) -> Tuple[sp.Expr, List[sp.Symbol]]:
        """Build the dimensionless modulation factor and its introduced parameter symbols."""
        if not pi_groups:
            raise ValueError(f"Template '{self.name}' requires at least one Buckingham-Pi group.")
        idx_list = list(range(1, len(pi_groups) + 1)) if indices is None else list(indices)
        if len(idx_list) != len(pi_groups):
            raise ValueError(
                f"Template '{self.name}' received {len(pi_groups)} Pi groups but {len(idx_list)} indices."
            )
        return self.builder(pi_groups, idx_list)

    def bound_for(self, param_name: str) -> Tuple[float, float]:
        """Return (lb, ub) for `param_name` via exact key or prefix match."""
        if param_name in self.param_bounds:
            return self.param_bounds[param_name]
        for prefix, bnd in self.param_bounds.items():
            if param_name.startswith(prefix):
                return bnd
        return (-np.inf, np.inf)

    @staticmethod
    def evaluate(expr: sp.Expr, data: Mapping[str, Any]) -> np.ndarray:
        """Evaluate a symbolic expression vectorized over `data`."""
        syms = sorted(expr.free_symbols, key=str)
        missing = [str(s) for s in syms if str(s) not in data]
        if missing:
            raise KeyError(f"Missing values for symbols {missing}.")
        fn = sp.lambdify(syms, expr, modules="numpy")
        args = [np.asarray(data[str(s)], dtype=float) for s in syms]
        return np.asarray(fn(*args), dtype=float)

    @staticmethod
    def differentiate(expr: sp.Expr, wrt: Union[str, sp.Symbol]) -> sp.Expr:
        """Differentiate `expr` symbolically with respect to variable name or Symbol."""
        if isinstance(wrt, str):
            matches = [s for s in expr.free_symbols if isinstance(s, sp.Symbol) and s.name == wrt]
            if not matches:
                return sp.Integer(0)
            target_sym = matches[0]
        else:
            target_sym = wrt
        return sp.diff(expr, target_sym)


class TemplateRegistry:
    """Registry of Buckingham-Pi equation templates with alias support."""

    def __init__(self) -> None:
        self._templates: Dict[str, EquationTemplate] = {}
        self._aliases: Dict[str, str] = {}

    def register(
        self,
        template: EquationTemplate,
        *,
        aliases: Sequence[str] = (),
        overwrite: bool = False,
    ) -> EquationTemplate:
        key = template.name.strip()
        if not key:
            raise ValueError("Template name cannot be empty.")
        if not overwrite and (key in self._templates or key in self._aliases):
            raise ValueError(f"Template '{key}' is already registered.")
        for alias in aliases:
            if not overwrite and (alias in self._templates or alias in self._aliases):
                raise ValueError(f"Template alias '{alias}' conflicts with an existing template.")
        self._templates[key] = template
        for alias in aliases:
            self._aliases[alias] = key
        return template

    def unregister(self, name: str) -> None:
        canonical = self._aliases.get(name, name)
        if canonical not in self._templates:
            raise KeyError(f"Unknown template '{name}'.")
        del self._templates[canonical]
        for alias in [k for k, v in self._aliases.items() if v == canonical]:
            del self._aliases[alias]

    def get(self, name: str) -> EquationTemplate:
        canonical = self._aliases.get(name, name)
        if canonical not in self._templates:
            available = sorted(list(self._templates) + list(self._aliases))
            raise KeyError(f"Unknown equation template '{name}'. Available: {available}.")
        return self._templates[canonical]

    def list(self, include_saturating: bool = True) -> List[EquationTemplate]:
        return [
            t for t in self._templates.values()
            if include_saturating or not t.saturating
        ]

    def names(self, include_saturating: bool = True) -> List[str]:
        return [t.name for t in self.list(include_saturating=include_saturating)]


def _build_power(pi_groups: Sequence[sp.Expr], indices: Sequence[int]) -> Tuple[sp.Expr, List[sp.Symbol]]:
    nu_syms = [sp.Symbol(f"nu_{idx}", real=True) for idx in indices]
    factor = sp.Integer(1)
    for pi_e, nu_s in zip(pi_groups, nu_syms):
        factor *= pi_e ** nu_s
    return factor, nu_syms


def _build_affine(pi_groups: Sequence[sp.Expr], indices: Sequence[int]) -> Tuple[sp.Expr, List[sp.Symbol]]:
    lam_syms = [sp.Symbol(f"lam_{idx}", real=True) for idx in indices]
    factor = sp.Integer(1) + sum(l_s * pi_e for l_s, pi_e in zip(lam_syms, pi_groups))
    return factor, lam_syms


def _build_hill(pi_groups: Sequence[sp.Expr], indices: Sequence[int]) -> Tuple[sp.Expr, List[sp.Symbol]]:
    nu_syms = [sp.Symbol(f"nu_{idx}", real=True) for idx in indices]
    gam_syms = [sp.Symbol(f"gam_{idx}", positive=True, real=True) for idx in indices]
    factor = sp.Integer(1)
    for pi_e, nu_s, gam_s in zip(pi_groups, nu_syms, gam_syms):
        scaled = gam_s * (pi_e ** nu_s)
        factor *= scaled / (sp.Integer(1) + scaled)
    return factor, gam_syms + nu_syms


def _build_exp_sat(pi_groups: Sequence[sp.Expr], indices: Sequence[int]) -> Tuple[sp.Expr, List[sp.Symbol]]:
    nu_syms = [sp.Symbol(f"nu_{idx}", real=True) for idx in indices]
    gam_syms = [sp.Symbol(f"gam_{idx}", positive=True, real=True) for idx in indices]
    factor = sp.Integer(1)
    for pi_e, nu_s, gam_s in zip(pi_groups, nu_syms, gam_syms):
        scaled = gam_s * (pi_e ** nu_s)
        factor *= sp.Integer(1) - sp.exp(-scaled)
    return factor, gam_syms + nu_syms


TEMPLATE_REGISTRY = TemplateRegistry()

TEMPLATE_REGISTRY.register(
    EquationTemplate(
        name="power",
        candidate_name="buckingham_pi_power",
        kind="pi_power",
        description="Modulated by power-law Buckingham Pi dimensionless group(s).",
        builder=_build_power,
        saturating=False,
        assumptions=("Positive dimensionless groups Pi_k > 0 for arbitrary real exponents nu_k.",),
    ),
    aliases=("monomial",),
)

TEMPLATE_REGISTRY.register(
    EquationTemplate(
        name="affine",
        candidate_name="buckingham_pi_affine",
        kind="pi_affine",
        description="First-order affine series in Buckingham Pi dimensionless group(s).",
        builder=_build_affine,
        saturating=False,
        assumptions=("Small-perturbation linear expansion in dimensionless groups Pi_k.",),
    ),
)

TEMPLATE_REGISTRY.register(
    EquationTemplate(
        name="hill",
        candidate_name="buckingham_pi_hill",
        kind="pi_hill",
        description="Rational/Hill saturating [0,1] Pi-modulation (impingement / Messinger freezing regime).",
        builder=_build_hill,
        saturating=True,
        assumptions=("Positive saturation gain gam_k > 0 and non-negative Pi_k.",),
        param_bounds={"gam_": (1e-9, np.inf)},
    ),
)

TEMPLATE_REGISTRY.register(
    EquationTemplate(
        name="exp_sat",
        candidate_name="buckingham_pi_exp_sat",
        kind="pi_exp_sat",
        description="Exponential Rosin-Rammler saturating [0,1] Pi-modulation.",
        builder=_build_exp_sat,
        saturating=True,
        assumptions=("Positive saturation gain gam_k > 0 and non-negative Pi_k.",),
        param_bounds={"gam_": (1e-9, np.inf)},
    ),
    aliases=("rosin_rammler",),
)


def register_template(
    template: EquationTemplate,
    *,
    aliases: Sequence[str] = (),
    overwrite: bool = False,
) -> EquationTemplate:
    """Register a custom equation template in the global `TEMPLATE_REGISTRY`."""
    return TEMPLATE_REGISTRY.register(template, aliases=aliases, overwrite=overwrite)


def get_template(name: str) -> EquationTemplate:
    """Retrieve an equation template by canonical name or alias."""
    return TEMPLATE_REGISTRY.get(name)


def list_templates(include_saturating: bool = True) -> List[EquationTemplate]:
    """List registered equation templates."""
    return TEMPLATE_REGISTRY.list(include_saturating=include_saturating)
