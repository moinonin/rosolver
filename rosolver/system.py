"""
System declaration and Dependency Directed Acyclic Graph (DAG) representation.

Allows students and researchers to declare qualitative dependencies such as:
  g = fn(a, b, c)
  b = fn(f; grav)
with optional physical/domain dimensions and optional rate identities (e.g. ds/dt = v).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import sympy as sp

from .dimensions import Dimension


@dataclass
class Variable:
    """A physical or domain factor in an equation-formulation problem."""

    name: str
    dim: Optional[Dimension] = None
    role: str = "variable"  # "variable" or "constant"
    description: str = ""
    positive: Optional[bool] = True
    symbol: sp.Symbol = field(init=False)
    initial_symbol: sp.Symbol = field(init=False)

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.name):
            raise ValueError(f"Invalid variable identifier: '{self.name}'")
        if self.positive is True:
            self.symbol = sp.Symbol(self.name, positive=True, real=True)
        elif self.positive is False:
            self.symbol = sp.Symbol(self.name, negative=True, real=True)
        else:
            self.symbol = sp.Symbol(self.name, real=True)
        self.initial_symbol = sp.Symbol(f"{self.name}_0", real=True)


@dataclass
class RelationSpec:
    """Qualitative functional relation target = fn(inputs; params)."""

    target: str
    inputs: List[str]
    params: List[str] = field(default_factory=list)
    conservative: Optional[bool] = None
    include_intercept: bool = False


@dataclass
class RateIdentity:
    """Known derivative identity d(target)/d(wrt) == equals (e.g. ds/dt = v, dv/dt = a)."""

    target: str
    wrt: str
    equals: str


_REL_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:=\s*[A-Za-z_][A-Za-z0-9_]*\s*)?\(\s*([^)]*)\s*\)\s*$"
)


class System:
    """
    Qualitative + dimensional model specification over a Dependency DAG.

    Example:
        sys = System()
        sys.var("g", "M*L^2/T^3")
        sys.var("a", "M/L^3")
        sys.var("b", "L/T")
        sys.var("c", "L^2")
        sys.var("f", "L")
        sys.const("grav", "L/T^2")
        sys.depends("g", on=["a", "b", "c"])
        sys.depends("b", on=["f"], params=["grav"])
    """

    def __init__(self) -> None:
        self.variables: Dict[str, Variable] = {}
        self.relations: Dict[str, RelationSpec] = {}
        self.rate_identities: Dict[Tuple[str, str], RateIdentity] = {}

    def var(
        self,
        name: str,
        dim: Union[str, Dimension, Mapping[str, Union[int, float, str, sp.Rational]], None] = None,
        description: str = "",
        positive: Optional[bool] = True,
        *,
        unit: Optional[str] = None,
    ) -> Variable:
        if unit is not None and dim is not None:
            raise ValueError(f"Specify either `dim` or `unit` for variable '{name}', not both.")
        if unit is not None:
            parsed_dim: Optional[Dimension] = Dimension.from_unit(unit)
        else:
            parsed_dim = None if dim is None else self._parse_system_dim(name, dim)
        v = Variable(name=name, dim=parsed_dim, role="variable", description=description, positive=positive)
        self.variables[name] = v
        return v

    def const(
        self,
        name: str,
        dim: Union[str, Dimension, Mapping[str, Union[int, float, str, sp.Rational]], None] = None,
        description: str = "",
        positive: Optional[bool] = True,
        *,
        unit: Optional[str] = None,
    ) -> Variable:
        if unit is not None and dim is not None:
            raise ValueError(f"Specify either `dim` or `unit` for constant '{name}', not both.")
        if unit is not None:
            parsed_dim: Optional[Dimension] = Dimension.from_unit(unit)
        else:
            parsed_dim = None if dim is None else self._parse_system_dim(name, dim)
        v = Variable(name=name, dim=parsed_dim, role="constant", description=description, positive=positive)
        self.variables[name] = v
        return v

    def _parse_system_dim(
        self,
        name: str,
        dim: Union[str, Dimension, Mapping[str, Union[int, float, str, sp.Rational]]],
    ) -> Dimension:
        import warnings

        if isinstance(dim, str) and dim.strip() in ("J", "N"):
            warnings.warn(
                f"Ambiguous dimension string '{dim.strip()}' on '{name}': in base-dimension mode "
                f"'J' denotes luminous intensity (cd) and 'N' denotes amount of substance (mol). "
                f"Pass unit='{dim.strip()}' (or 'Joule'/'Newton') if you meant the SI derived unit.",
                UserWarning,
                stacklevel=3,
            )
        return Dimension.parse(dim)

    def _ensure_registered(self, name: str, default_role: str = "variable") -> Variable:
        if name not in self.variables:
            self.variables[name] = Variable(name=name, dim=None, role=default_role)
        return self.variables[name]

    def depends(
        self,
        target: str,
        on: Sequence[str],
        params: Optional[Sequence[str]] = None,
        conservative: Optional[bool] = None,
        include_intercept: bool = False,
    ) -> RelationSpec:
        if not on:
            raise ValueError(f"Relation for '{target}' must specify at least one input in 'on'.")
        if target in on:
            raise ValueError(f"Self-dependency '{target} -> {target}' is not allowed in a DAG.")

        self._ensure_registered(target, "variable")
        clean_inputs = list(dict.fromkeys(on))
        clean_params = list(dict.fromkeys(params or []))
        for inp in clean_inputs:
            self._ensure_registered(inp, "variable")
        for par in clean_params:
            self._ensure_registered(par, "constant")

        spec = RelationSpec(
            target=target,
            inputs=clean_inputs,
            params=clean_params,
            conservative=conservative,
            include_intercept=include_intercept,
        )
        old = self.relations.get(target)
        self.relations[target] = spec
        try:
            self.validate_acyclic()
        except ValueError:
            if old is None:
                del self.relations[target]
            else:
                self.relations[target] = old
            raise
        return spec

    def rate_identity(self, target: str, wrt: str, equals: str) -> RateIdentity:
        """Declare d(target)/d(wrt) = equals (such as ds/dt = v in ch1.tex Sec 1)."""
        self._ensure_registered(target, "variable")
        self._ensure_registered(wrt, "variable")
        self._ensure_registered(equals, "variable")
        ri = RateIdentity(target=target, wrt=wrt, equals=equals)
        self.rate_identities[(target, wrt)] = ri
        return ri

    @classmethod
    def from_spec(
        cls,
        relations: Sequence[str],
        dims: Optional[Mapping[str, Union[str, Dimension, Mapping[str, int]]]] = None,
        constants: Optional[Iterable[str]] = None,
        rate_identities: Optional[Sequence[Tuple[str, str, str]]] = None,
    ) -> "System":
        """
        Build a System from concise text relations such as:
          ["g = fn(a, b, c)", "b = fn(f; grav)"]
        """
        sys = cls()
        const_set = set(constants or [])
        if dims:
            for name, d in dims.items():
                if name in const_set:
                    sys.const(name, d)
                else:
                    sys.var(name, d)
        for c in const_set:
            if c not in sys.variables:
                sys.const(c, None)

        for raw in relations:
            m = _REL_RE.match(raw)
            if not m:
                raise ValueError(
                    f"Cannot parse relation '{raw}'. Expected format like 'g = fn(a, b, c)' or 'b = fn(f; grav)'."
                )
            target = m.group(1)
            body = m.group(2)
            if ";" in body:
                inp_part, par_part = body.split(";", 1)
            else:
                inp_part, par_part = body, ""
            inputs = [tok.strip() for tok in inp_part.split(",") if tok.strip()]
            params = [tok.strip() for tok in par_part.split(",") if tok.strip()]
            sys.depends(target, on=inputs, params=params)

        if rate_identities:
            for tgt, wrt, eq in rate_identities:
                sys.rate_identity(tgt, wrt=wrt, equals=eq)

        return sys

    @property
    def has_all_dimensions(self) -> bool:
        return bool(self.variables) and all(v.dim is not None for v in self.variables.values())

    def symbols_map(self) -> Dict[str, sp.Symbol]:
        return {name: v.symbol for name, v in self.variables.items()}

    def validate_acyclic(self) -> List[str]:
        """Return topological ordering from leaves (independent) to roots (targets), or raise on cycle."""
        visited: Dict[str, int] = {}  # 0=visiting, 1=done
        order: List[str] = []

        def dfs(node: str, stack: List[str]) -> None:
            state = visited.get(node)
            if state == 1:
                return
            if state == 0:
                cycle_path = " -> ".join(stack + [node])
                raise ValueError(f"Cycle detected in dependency graph: {cycle_path}")
            visited[node] = 0
            rel = self.relations.get(node)
            if rel is not None:
                for inp in rel.inputs:
                    dfs(inp, stack + [node])
            visited[node] = 1
            order.append(node)

        for name in self.variables:
            if name not in visited:
                dfs(name, [])
        return order

    def ancestors_of(self, target: str) -> Set[str]:
        """All upstream variables that feed into target (transitively)."""
        result: Set[str] = set()

        def walk(curr: str) -> None:
            rel = self.relations.get(curr)
            if rel is None:
                return
            for inp in rel.inputs:
                if inp not in result:
                    result.add(inp)
                    walk(inp)

        walk(target)
        return result

    def depends_on(self, node: str, wrt: str) -> bool:
        if node == wrt:
            return True
        return wrt in self.ancestors_of(node)

    def paths_between(self, source: str, target: str) -> List[List[str]]:
        """Enumerate all directed dependency paths [source, ..., target] where target depends on source."""
        paths: List[List[str]] = []

        def backtrack(curr: str, suffix: List[str]) -> None:
            if curr == source:
                paths.append([source] + suffix)
                return
            rel = self.relations.get(curr)
            if rel is None:
                return
            for inp in rel.inputs:
                backtrack(inp, [curr] + suffix)

        backtrack(target, [])
        return paths

    def subgraph_order(self, target: str) -> List[str]:
        """Topological bottom-up order restricted to target and its upstream ancestors."""
        full_order = self.validate_acyclic()
        needed = self.ancestors_of(target) | {target}
        return [n for n in full_order if n in needed]
