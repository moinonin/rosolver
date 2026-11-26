"""Safe AST-validated expression parsing that treats every identifier as a real symbol.

Avoids `sympy.sympify()` on untrusted strings (which internally calls Python `eval`).
User identifiers are bound to plain real symbols (except an explicit allow-list of
math functions and `pi`), preventing silent SymPy singleton collisions on names like
`S`, `I`, `E`, `N`, `O`, `Q`, `beta`, or `gamma`.
"""

from __future__ import annotations

import ast
import operator
from typing import Callable, Dict, Mapping, Optional, Set, Union

import numpy as np
import sympy as sp

ExprLike = Union[str, int, float, sp.Basic]

_MAX_EXPR_LENGTH = 10_000
_MAX_AST_NODES = 500

_CALLABLE_FUNCS: Dict[str, Callable[..., sp.Expr]] = {
    name: getattr(sp, name)
    for name in (
        "sin", "cos", "tan", "exp", "log", "sqrt", "tanh", "sinh", "cosh",
        "atan", "asin", "acos", "atan2", "Abs", "sign", "Min", "Max", "erf",
        "Heaviside", "floor", "ceiling",
    )
}
_CALLABLE_FUNCS.update({"abs": sp.Abs, "ln": sp.log})

_CONSTANTS: Dict[str, sp.Expr] = {"pi": sp.pi}

_BIN_OPS: Dict[type, Callable[[sp.Expr, sp.Expr], sp.Expr]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

_UNARY_OPS: Dict[type, Callable[[sp.Expr], sp.Expr]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def real_symbol(name: str) -> sp.Symbol:
    """Create a real-valued SymPy Symbol."""
    return sp.Symbol(name, real=True)


def _eval_ast(node: ast.AST, symbols: Mapping[str, sp.Symbol]) -> sp.Expr:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body, symbols)

    if isinstance(node, ast.Constant):
        val = node.value
        if isinstance(val, bool):
            raise ValueError("Boolean literals are not valid mathematical expressions.")
        if isinstance(val, int):
            return sp.Integer(val)
        if isinstance(val, float):
            return sp.Float(val)
        raise ValueError(f"Unsupported constant literal of type {type(val).__name__}.")

    if isinstance(node, ast.Name):
        ident = node.id
        if ident.startswith("__") or ident.endswith("__"):
            raise ValueError(f"Disallowed dunder identifier '{ident}'.")
        if ident in symbols:
            return symbols[ident]
        if ident in _CONSTANTS:
            return _CONSTANTS[ident]
        if ident in _CALLABLE_FUNCS:
            raise ValueError(f"Function '{ident}' must be called with parentheses.")
        return real_symbol(ident)

    if isinstance(node, ast.UnaryOp):
        op_fn = _UNARY_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Disallowed unary operator: {type(node.op).__name__}.")
        return op_fn(_eval_ast(node.operand, symbols))

    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.BitXor):
            raise ValueError("Unsupported operator '^'; use '**' for exponentiation.")
        op_fn = _BIN_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Disallowed binary operator: {type(node.op).__name__}.")
        return op_fn(_eval_ast(node.left, symbols), _eval_ast(node.right, symbols))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct calls to allow-listed math functions are permitted.")
        if node.keywords:
            raise ValueError("Keyword arguments are not permitted in expressions.")
        fname = node.func.id
        fn = _CALLABLE_FUNCS.get(fname)
        if fn is None:
            raise ValueError(f"Function '{fname}' is not in the math allow-list.")
        args = []
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                raise ValueError("Starred arguments are not permitted in expressions.")
            args.append(_eval_ast(arg, symbols))
        return fn(*args)

    raise ValueError(f"Disallowed syntax in expression: {type(node).__name__}.")


def parse_expr(value: ExprLike, symbols: Optional[Mapping[str, sp.Symbol]] = None) -> sp.Expr:
    """Parse a string/number/SymPy expression safely, binding identifiers to real symbols."""
    sym_map = dict(symbols or {})
    if isinstance(value, sp.Basic):
        mapping = {
            s: sym_map.get(s.name, real_symbol(s.name))
            for s in value.free_symbols
            if isinstance(s, sp.Symbol)
        }
        return value.xreplace(mapping)
    if isinstance(value, bool):
        raise TypeError("Boolean is not a valid expression.")
    if isinstance(value, (int, np.integer)):
        return sp.Integer(int(value))
    if isinstance(value, (float, np.floating)):
        return sp.Float(float(value))
    if isinstance(value, str):
        if len(value) > _MAX_EXPR_LENGTH:
            raise ValueError(f"Expression exceeds maximum length of {_MAX_EXPR_LENGTH} characters.")
        text = value.strip()
        if not text:
            raise ValueError("Expression string cannot be empty.")
        try:
            tree = ast.parse(text, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Invalid expression syntax in '{value}': {exc.msg}") from exc
        if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
            raise ValueError(f"Expression exceeds maximum complexity ({_MAX_AST_NODES} AST nodes).")
        return _eval_ast(tree, sym_map)
    raise TypeError(f"Cannot parse expression of type {type(value)!r}.")


def names(expr: sp.Basic) -> Set[str]:
    """Return all free symbol names in a SymPy expression."""
    return {s.name for s in expr.free_symbols if isinstance(s, sp.Symbol)}
