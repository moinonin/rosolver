"""
Dimensional algebra and exact rational Buckingham-Pi / monomial solver.

Supports SI base dimensions (M, L, T, I, Theta, N, J, rad), common unit aliases
(kg, m, s, A, K, mol, cd), and arbitrary user-defined domain dimensions
(e.g., USD, Cells, Bits, Users) for non-physical sciences.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from itertools import combinations
from math import gcd
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import sympy as sp


UNIT_TO_DIM: Dict[str, str] = {
    "kg": "M",
    "m": "L",
    "s": "T",
    "sec": "T",
    "A": "I",
    "K": "Theta",
    "mol": "N",
    "cd": "J",
}

# Derived SI units whose names do not collide with canonical dimension symbols (M, L, T, I, Theta, N, J)
DERIVED_UNIT_TO_DIM: Dict[str, Dict[str, int]] = {
    "Pa": {"M": 1, "L": -1, "T": -2},
    "Pascal": {"M": 1, "L": -1, "T": -2},
    "W": {"M": 1, "L": 2, "T": -3},
    "Watt": {"M": 1, "L": 2, "T": -3},
    "Hz": {"T": -1},
    "Hertz": {"T": -1},
    "V": {"M": 1, "L": 2, "T": -3, "I": -1},
    "Volt": {"M": 1, "L": 2, "T": -3, "I": -1},
    "C": {"I": 1, "T": 1},
    "Coulomb": {"I": 1, "T": 1},
    "Ohm": {"M": 1, "L": 2, "T": -3, "I": -2},
    "Newton": {"M": 1, "L": 1, "T": -2},
    "Joule": {"M": 1, "L": 2, "T": -2},
    "degC": {"Theta": 1},
}

# Physical SI unit symbols (for Model `unit:` fields where N=Newton, J=Joule, T=Tesla, etc.)
SI_UNIT_TO_DIM: Dict[str, Dict[str, int]] = {
    **DERIVED_UNIT_TO_DIM,
    "kg": {"M": 1},
    "m": {"L": 1},
    "s": {"T": 1},
    "sec": {"T": 1},
    "A": {"I": 1},
    "K": {"Theta": 1},
    "mol": {"N": 1},
    "cd": {"J": 1},
    "rad": {"rad": 1},
    "N": {"M": 1, "L": 1, "T": -2},
    "J": {"M": 1, "L": 2, "T": -2},
    "T": {"M": 1, "T": -2, "I": -1},
    "F": {"M": -1, "L": -2, "T": 4, "I": 2},
    "H": {"M": 1, "L": 2, "T": -2, "I": -2},
    "S": {"M": -1, "L": -2, "T": 3, "I": 2},
    "Wb": {"M": 1, "L": 2, "T": -2, "I": -1},
}

CANONICAL_ORDER: Tuple[str, ...] = ("M", "L", "T", "I", "Theta", "N", "J", "rad")


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // gcd(a, b) if a and b else 0


def _canonicalize_null_vector(vec: sp.Matrix) -> List[sp.Rational]:
    """Scale rational nullspace vector to coprime integers with positive leading entry."""
    entries = [sp.Rational(v) for v in vec]
    denoms = [int(v.q) for v in entries if v != 0]
    if not denoms:
        return entries
    scale_lcm = reduce(_lcm, denoms, 1)
    ints = [int(v * scale_lcm) for v in entries]
    non_zero = [abs(x) for x in ints if x != 0]
    g = reduce(gcd, non_zero) if non_zero else 1
    ints = [x // g for x in ints]
    # Prefer first non-zero entry to be positive for deterministic Pi-groups
    for x in ints:
        if x != 0:
            if x < 0:
                ints = [-y for y in ints]
            break
    return [sp.Rational(x) for x in ints]


@dataclass(frozen=True)
class Dimension:
    """Exact rational exponent map over base dimensions."""

    exponents: Tuple[Tuple[str, sp.Rational], ...] = ()

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Union[int, float, str, sp.Rational]]) -> "Dimension":
        cleaned: Dict[str, sp.Rational] = {}
        for k, v in mapping.items():
            key = UNIT_TO_DIM.get(k, k)
            rat = sp.Rational(v)
            if rat != 0:
                cleaned[key] = cleaned.get(key, sp.Rational(0)) + rat
                if cleaned[key] == 0:
                    del cleaned[key]
        ordered = sorted(
            cleaned.items(),
            key=lambda kv: (
                CANONICAL_ORDER.index(kv[0]) if kv[0] in CANONICAL_ORDER else len(CANONICAL_ORDER),
                kv[0],
            ),
        )
        return cls(tuple(ordered))

    @classmethod
    def dimensionless(cls) -> "Dimension":
        return cls(())

    @classmethod
    def parse(
        cls,
        spec: Union[str, "Dimension", Mapping[str, Union[int, float, str, sp.Rational]], None],
        *,
        unit_mode: bool = False,
    ) -> "Dimension":
        """
        Parse dimension or SI unit specifications such as:
          - "M*L^2/T^3", "kg*m^2/s^3", "W/(m*K)", "Pa*s", "L^(1/2)/T", "1", ""
          - {"M": 1, "L": 2, "T": -3}
        """
        if spec is None:
            return cls.dimensionless()
        if isinstance(spec, Dimension):
            return spec
        if isinstance(spec, Mapping):
            return cls.from_dict(spec)
        if not isinstance(spec, str):
            raise TypeError(f"Unsupported dimension specification type: {type(spec)}")

        text = spec.strip()
        if text in ("", "1", "dimensionless", "none", "None"):
            return cls.dimensionless()

        # Normalize '**' to '^' for recursive descent parser
        text = text.replace("**", "^")
        tokens = _tokenize_dim(text)
        parser = _DimParser(tokens, unit_mode=unit_mode)
        result = parser.parse_expr()
        if parser.pos < len(tokens):
            raise ValueError(f"Unexpected token '{tokens[parser.pos]}' in dimension '{spec}'")
        return result

    @classmethod
    def from_unit(
        cls, spec: Union[str, "Dimension", Mapping[str, Union[int, float, str, sp.Rational]], None]
    ) -> "Dimension":
        """Parse a physical SI unit string (e.g. 'N/m^2', 'J/(kg*K)', 'W/(m*K)', 'm^2/s')."""
        return cls.parse(spec, unit_mode=True)

    @property
    def is_dimensionless(self) -> bool:
        return len(self.exponents) == 0

    def to_dict(self) -> Dict[str, sp.Rational]:
        return dict(self.exponents)

    def bases(self) -> List[str]:
        return [k for k, _ in self.exponents]

    def __mul__(self, other: "Dimension") -> "Dimension":
        d = self.to_dict()
        for k, v in other.exponents:
            d[k] = d.get(k, sp.Rational(0)) + v
        return Dimension.from_dict(d)

    def __truediv__(self, other: "Dimension") -> "Dimension":
        d = self.to_dict()
        for k, v in other.exponents:
            d[k] = d.get(k, sp.Rational(0)) - v
        return Dimension.from_dict(d)

    def __pow__(self, power: Union[int, str, sp.Rational]) -> "Dimension":
        p = sp.Rational(power)
        if p == 0:
            return Dimension.dimensionless()
        return Dimension.from_dict({k: v * p for k, v in self.exponents})

    def vector(self, basis: Sequence[str]) -> sp.Matrix:
        d = self.to_dict()
        return sp.Matrix([d.get(b, sp.Rational(0)) for b in basis])

    def format_str(self) -> str:
        if self.is_dimensionless:
            return "1"
        num: List[str] = []
        den: List[str] = []
        for k, v in self.exponents:
            if v > 0:
                num.append(k if v == 1 else f"{k}^{_fmt_rat(v)}")
            elif v < 0:
                av = -v
                den.append(k if av == 1 else f"{k}^{_fmt_rat(av)}")
        lhs = "*".join(num) if num else "1"
        if not den:
            return lhs
        rhs = den[0] if len(den) == 1 else f"({'*'.join(den)})"
        return f"{lhs}/{rhs}"

    def to_latex(self) -> str:
        if self.is_dimensionless:
            return "1"
        parts = []
        for k, v in self.exponents:
            sym = r"\Theta" if k == "Theta" else (r"\text{" + k + "}" if len(k) > 1 else k)
            if v == 1:
                parts.append(sym)
            else:
                parts.append(f"{sym}^{{{sp.latex(v)}}}")
        return " ".join(parts)

    def __str__(self) -> str:
        return self.format_str()


def _fmt_rat(r: sp.Rational) -> str:
    return str(int(r)) if r.q == 1 else f"({r.p}/{r.q})"


_TOKEN_RE = re.compile(r"\s*(\(|\)|\*|/|\^|[+-]?\d+|[A-Za-z_][A-Za-z0-9_]*)\s*")


def _tokenize_dim(text: str) -> List[str]:
    pos = 0
    tokens: List[str] = []
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if not m:
            raise ValueError(f"Invalid syntax near '{text[pos:]}' in dimension string '{text}'")
        tokens.append(m.group(1))
        pos = m.end()
    return tokens


class _DimParser:
    def __init__(self, tokens: List[str], unit_mode: bool = False) -> None:
        self.tokens = tokens
        self.pos = 0
        self.unit_mode = unit_mode

    def peek(self) -> Optional[str]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected: Optional[str] = None) -> str:
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected end of dimension expression")
        if expected is not None and tok != expected:
            raise ValueError(f"Expected '{expected}', got '{tok}'")
        self.pos += 1
        return tok

    def parse_expr(self) -> Dimension:
        acc = self.parse_power()
        while self.peek() in ("*", "/"):
            op = self.consume()
            rhs = self.parse_power()
            acc = acc * rhs if op == "*" else acc / rhs
        return acc

    def parse_power(self) -> Dimension:
        base = self.parse_atom()
        if self.peek() == "^":
            self.consume("^")
            exp = self.parse_exponent()
            return base ** exp
        return base

    def parse_exponent(self) -> sp.Rational:
        if self.peek() == "(":
            self.consume("(")
            sign = 1
            if self.peek() in ("+", "-"):
                if self.consume() == "-":
                    sign = -1
            num = int(self.consume())
            if self.peek() == "/":
                self.consume("/")
                den = int(self.consume())
                val = sp.Rational(sign * num, den)
            else:
                val = sp.Rational(sign * num)
            self.consume(")")
            return val
        tok = self.consume()
        return sp.Rational(int(tok))

    def parse_atom(self) -> Dimension:
        tok = self.peek()
        if tok == "(":
            self.consume("(")
            inner = self.parse_expr()
            self.consume(")")
            return inner
        ident = self.consume()
        if ident == "1":
            return Dimension.dimensionless()
        if re.fullmatch(r"[+-]?\d+", ident):
            raise ValueError(f"Unexpected numeric literal '{ident}' as base dimension")
        if self.unit_mode and ident in SI_UNIT_TO_DIM:
            return Dimension.from_dict(SI_UNIT_TO_DIM[ident])
        if ident in DERIVED_UNIT_TO_DIM:
            return Dimension.from_dict(DERIVED_UNIT_TO_DIM[ident])
        mapped = UNIT_TO_DIM.get(ident, ident)
        return Dimension.from_dict({mapped: 1})


@dataclass
class DimensionalSolution:
    """Result of solving D * alpha = target_dim over candidate variables."""

    consistent: bool
    basis_dims: List[str]
    var_names: List[str]
    particular_exponents: List[sp.Rational]
    pi_groups: List[List[sp.Rational]]
    alternative_particulars: List[List[sp.Rational]]
    missing_dimensions: List[str]
    diagnostic: str = ""

    @property
    def nullity(self) -> int:
        return len(self.pi_groups)

    @property
    def is_unique(self) -> bool:
        return self.consistent and len(self.pi_groups) == 0

    def particular_monomial(self, symbols_map: Mapping[str, sp.Symbol]) -> sp.Expr:
        expr = sp.Integer(1)
        for name, exp in zip(self.var_names, self.particular_exponents):
            if exp != 0:
                expr *= symbols_map[name] ** exp
        return expr

    def pi_monomials(self, symbols_map: Mapping[str, sp.Symbol]) -> List[sp.Expr]:
        pis: List[sp.Expr] = []
        for vec in self.pi_groups:
            expr = sp.Integer(1)
            for name, exp in zip(self.var_names, vec):
                if exp != 0:
                    expr *= symbols_map[name] ** exp
            pis.append(expr)
        return pis


class DimensionalEngine:
    """Exact rational solver for dimensional homogeneity and Buckingham Pi-groups."""

    @staticmethod
    def build_basis(dims: Iterable[Dimension]) -> List[str]:
        seen = set()
        for d in dims:
            seen.update(d.bases())
        return sorted(
            seen,
            key=lambda k: (
                CANONICAL_ORDER.index(k) if k in CANONICAL_ORDER else len(CANONICAL_ORDER),
                k,
            ),
        )

    @classmethod
    def solve(
        cls,
        target: Dimension,
        candidates: Sequence[Tuple[str, Dimension]],
        max_subset_search: int = 10,
    ) -> DimensionalSolution:
        """
        Solve sum_j alpha_j * dim(x_j) = dim(target) in exact rational arithmetic.

        Returns:
          - Minimum-complexity particular exponent vector alpha^(p)
          - Coprime integer Buckingham Pi nullspace basis vectors
          - Diagnostic details if inconsistent (e.g., missing physical constants)
        """
        var_names = [name for name, _ in candidates]
        cand_dims = [dim for _, dim in candidates]
        basis = cls.build_basis([target, *cand_dims])

        if not basis:
            # Purely dimensionless target and purely dimensionless candidates
            n = len(candidates)
            pi_vecs = []
            for i in range(n):
                v = [sp.Rational(0)] * n
                v[i] = sp.Rational(1)
                pi_vecs.append(v)
            return DimensionalSolution(
                consistent=True,
                basis_dims=[],
                var_names=var_names,
                particular_exponents=[sp.Rational(0)] * n,
                pi_groups=pi_vecs,
                alternative_particulars=[],
                missing_dimensions=[],
                diagnostic="Dimensionless target and candidates.",
            )

        if not candidates:
            return DimensionalSolution(
                consistent=target.is_dimensionless,
                basis_dims=basis,
                var_names=[],
                particular_exponents=[],
                pi_groups=[],
                alternative_particulars=[],
                missing_dimensions=target.bases(),
                diagnostic=(
                    "Dimensionless constant."
                    if target.is_dimensionless
                    else f"No scope variables available to balance dimensions {target}."
                ),
            )

        cols = [d.vector(basis) for d in cand_dims]
        D = sp.Matrix.hstack(*cols)  # shape: (k, m)
        b = target.vector(basis)     # shape: (k, 1)

        rank_D = D.rank()
        aug = D.row_join(b)
        rank_aug = aug.rank()

        if rank_D < rank_aug:
            # Identify unspanned base dimensions via left nullspace of D (nullspace of D.T)
            left_null = D.T.nullspace()
            missing: List[str] = []
            for w in left_null:
                if (w.T * b)[0] != 0:
                    for idx, coeff in enumerate(w):
                        if coeff != 0 and basis[idx] not in missing:
                            missing.append(basis[idx])
            diag = (
                f"Dimensional mismatch: target [{target}] cannot be formed from "
                f"{{{', '.join(f'{n}:[{d}]' for n, d in candidates)}}}. "
                f"Unbalanced base dimension(s): {', '.join(missing)}. "
                f"Add a governing constant or factor carrying {', '.join(missing)}."
            )
            return DimensionalSolution(
                consistent=False,
                basis_dims=basis,
                var_names=var_names,
                particular_exponents=[],
                pi_groups=[],
                alternative_particulars=[],
                missing_dimensions=missing,
                diagnostic=diag,
            )

        # Compute canonical coprime integer Buckingham-Pi nullspace basis
        raw_null = D.nullspace()
        pi_groups = [_canonicalize_null_vector(v) for v in raw_null]

        # Find cleanest particular solution(s)
        m = len(candidates)
        alternatives: List[List[sp.Rational]] = []

        if rank_D == 0:
            particular = [sp.Rational(0)] * m
        elif m <= max_subset_search and len(pi_groups) > 0:
            scored: List[Tuple[Tuple[int, sp.Rational, int, Tuple[sp.Rational, ...]], List[sp.Rational]]] = []
            seen_sols = set()
            for subset in combinations(range(m), rank_D):
                D_sub = D[:, list(subset)]
                if D_sub.rank() < rank_D:
                    continue
                sol_sub, params = D_sub.gauss_jordan_solve(b)
                if len(params) > 0:
                    continue
                full = [sp.Rational(0)] * m
                for pos_idx, col_idx in enumerate(subset):
                    full[col_idx] = sp.Rational(sol_sub[pos_idx])
                key = tuple(full)
                if key in seen_sols:
                    continue
                seen_sols.add(key)
                max_den = max((int(x.q) for x in full), default=1)
                l1_norm = sum((abs(x) for x in full), sp.Rational(0))
                neg_count = sum(1 for x in full if x < 0)
                score = (max_den, l1_norm, neg_count, tuple(-x for x in full))
                scored.append((score, full))
            scored.sort(key=lambda item: item[0])
            particular = scored[0][1]
            alternatives = [sol for _, sol in scored[1:]]
        else:
            sol, params = D.gauss_jordan_solve(b)
            subs_zero = {p: 0 for p in params}
            particular = [sp.Rational(v.subs(subs_zero)) for v in sol]

        return DimensionalSolution(
            consistent=True,
            basis_dims=basis,
            var_names=var_names,
            particular_exponents=particular,
            pi_groups=pi_groups,
            alternative_particulars=alternatives,
            missing_dimensions=[],
            diagnostic="Unique monomial closure." if not pi_groups else f"Solved with {len(pi_groups)} dimensionless Pi-group(s).",
        )
