"""Expression parsing that treats every identifier as a real symbol.

`sympy.sympify("beta*S*I")` would silently map `S`, `I`, `E`, `N`, `beta`, `gamma` ... to sympy
singletons/functions. Models in epidemiology, kinetics or economics use exactly those names, so
all user identifiers are bound to plain real symbols here, except a small allow-list of math
functions.
"""

from __future__ import annotations

import re
from typing import Dict, Mapping, Optional, Set, Union

import numpy as np
import sympy as sp

ExprLike = Union[str, int, float, sp.Basic]

_FUNCS = {
    name: getattr(sp, name)
    for name in (
        "sin", "cos", "tan", "exp", "log", "sqrt", "tanh", "sinh", "cosh",
        "atan", "asin", "acos", "atan2", "Abs", "sign", "Min", "Max", "erf",
        "Heaviside", "floor", "ceiling",
    )
}
_FUNCS.update({"abs": sp.Abs, "ln": sp.log, "pi": sp.pi})
_IDENT = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*")


def real_symbol(name: str) -> sp.Symbol:
    return sp.Symbol(name, real=True)


def parse_expr(value: ExprLike, symbols: Optional[Mapping[str, sp.Symbol]] = None) -> sp.Expr:
    """Parse a string/number/sympy expression, binding identifiers by *name* to real symbols."""
    symbols = dict(symbols or {})
    if isinstance(value, sp.Basic):
        mapping = {
            s: symbols.get(s.name, real_symbol(s.name))
            for s in value.free_symbols
            if isinstance(s, sp.Symbol)
        }
        return sp.sympify(value).xreplace(mapping)
    if isinstance(value, bool):
        raise TypeError("Boolean is not a valid expression.")
    if isinstance(value, (int, np.integer)):
        return sp.Integer(int(value))
    if isinstance(value, (float, np.floating)):
        return sp.Float(float(value))
    if isinstance(value, str):
        local: Dict[str, object] = {}
        for ident in set(_IDENT.findall(value)):
            if ident in symbols:
                local[ident] = symbols[ident]
            elif ident in _FUNCS:
                local[ident] = _FUNCS[ident]
            else:
                local[ident] = real_symbol(ident)
        return sp.sympify(value, locals=local)
    raise TypeError(f"Cannot parse expression of type {type(value)!r}.")


def names(expr: sp.Basic) -> Set[str]:
    return {s.name for s in expr.free_symbols if isinstance(s, sp.Symbol)}
