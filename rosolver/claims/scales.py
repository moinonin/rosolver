"""Order-of-Magnitude Scale-Balance, Normalization Repair & Homogeneity Audits (Sprint 10 / S10.2).

Catches three subtle scientific failure modes missed by pure dimensional analysis:
  1. FM-1: Dimensionally homogeneous equations whose terms differ by orders of magnitude
     across the stated operating envelope (e.g., rho*a + grad_P = 0 with a 10^13x gap).
  2. FM-2: Implicit unit normalizations (e.g., 1 - rho*V hiding a mass scale m_max) and
     differential-geometry mismatches (e.g., sigma = A * dz giving L^3 instead of L^2).
  3. FM-4: Zero-intercept Euler homogeneity elasticity sum-rule violations (D * E = d_y)
     and collinear/unexcited Buckingham-Pi experimental designs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import sympy as sp

from .._parsing import parse_expr
from ..dimensions import CANONICAL_ORDER, Dimension, DimensionalEngine
from ..results import ResultMixin, VerificationStatus, to_jsonable

_BASE_DIM_IDENTS = frozenset((*CANONICAL_ORDER, "1", "dimensionless"))
_IDENT_SCAN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _parse_dim(spec: Union[str, Dimension, Mapping[str, Any]]) -> Dimension:
    if isinstance(spec, Dimension):
        return spec
    if isinstance(spec, Mapping):
        return Dimension.from_dict(spec)
    txt = str(spec).strip()
    idents = _IDENT_SCAN_RE.findall(txt)
    if idents and all(tok in _BASE_DIM_IDENTS for tok in idents):
        return Dimension.parse(txt)
    try:
        return Dimension.from_unit(txt)
    except Exception:
        return Dimension.parse(txt)


def _expr_dimension(expr: sp.Expr, dim_map: Mapping[str, Dimension]) -> Tuple[Optional[Dimension], List[str]]:
    """Recursively infer exact Dimension of a SymPy expression, recording mismatch errors."""
    errors: List[str] = []

    def _walk(node: sp.Expr) -> Optional[Dimension]:
        if isinstance(node, sp.Number) or node is sp.pi:
            return Dimension.dimensionless()
        if isinstance(node, sp.Symbol):
            if node.name in dim_map:
                return dim_map[node.name]
            errors.append(f"Undeclared dimension for symbol '{node.name}'.")
            return None
        if isinstance(node, sp.Mul):
            acc = Dimension.dimensionless()
            for arg in node.args:
                d = _walk(arg)
                if d is None:
                    return None
                acc = acc * d
            return acc
        if isinstance(node, sp.Pow):
            base, exp = node.args
            d_base = _walk(base)
            d_exp = _walk(exp)
            if d_base is None or d_exp is None:
                return None
            if not d_exp.is_dimensionless:
                errors.append(f"Exponent '{exp}' in '{node}' must be dimensionless (got {d_exp}).")
                return None
            if exp.is_number:
                try:
                    return d_base ** sp.Rational(str(exp))
                except Exception:
                    return d_base ** sp.Rational(float(exp)).limit_denominator(1000)
            if not d_base.is_dimensionless:
                errors.append(f"Symbolic exponent '{exp}' requires dimensionless base '{base}'.")
                return None
            return Dimension.dimensionless()
        if isinstance(node, sp.Add):
            dims: List[Tuple[sp.Expr, Dimension]] = []
            for arg in node.args:
                d = _walk(arg)
                if d is not None:
                    dims.append((arg, d))
            if not dims:
                return None
            ref_expr, ref_dim = dims[0]
            for term_expr, term_dim in dims[1:]:
                if term_dim != ref_dim:
                    errors.append(
                        f"Inhomogeneous sum: '{ref_expr}' [{ref_dim}] vs '{term_expr}' [{term_dim}]."
                    )
            return ref_dim
        if isinstance(node, sp.Function):
            for arg in node.args:
                d = _walk(arg)
                if d is not None and not d.is_dimensionless and node.func not in (sp.Abs, sp.sign):
                    errors.append(f"Transcendental function '{node.func.__name__}' argument '{arg}' must be dimensionless (got [{d}]).")
            if node.func is sp.Abs:
                return _walk(node.args[0])
            return Dimension.dimensionless()
        return Dimension.dimensionless()

    return _walk(expr), errors


@dataclass
class TermMagnitudeEntry(ResultMixin):
    """Evaluated magnitude interval for a single additive term in an equation."""

    term_expr: str
    dimension: str
    min_abs: float
    max_abs: float
    typical_abs: float
    ratio_to_leading: float
    role: str  # "dominant", "subleading", "negligible", "unbalanced_orphan"


@dataclass
class ScaleBalanceReport(ResultMixin):
    """Deterministic order-of-magnitude term-balance audit for a governing equation."""

    equation: str
    dimensionally_homogeneous: bool
    terms: List[TermMagnitudeEntry]
    dominant_balance_ratio: float
    log10_gap: float
    scale_threshold: float
    diagnostic_code: str
    status: VerificationStatus
    passed: bool
    warnings: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"=== Scale-Balance Audit: {self.equation} ===",
            f"  Status: {self.status.value} | Code: {self.diagnostic_code} | Dim-Homogeneous: {self.dimensionally_homogeneous}",
            f"  Dominant ratio: {self.dominant_balance_ratio:.3e} (log10 gap = {self.log10_gap:.2f}, threshold = {self.scale_threshold:g})",
        ]
        for t in self.terms:
            lines.append(
                f"  - [{t.role}] {t.term_expr} [{t.dimension}]: "
                f"|T| in [{t.min_abs:.3e}, {t.max_abs:.3e}] (typ={t.typical_abs:.3e}, ratio={t.ratio_to_leading:.3e})"
            )
        for w in self.warnings:
            lines.append(f"  Warning: {w}")
        return "\n".join(lines)


def _sample_envelope(
    envelope: Mapping[str, Tuple[float, float]],
    symbols: Sequence[sp.Symbol],
    max_samples: int = 256,
) -> Dict[str, np.ndarray]:
    """Deterministic corner + geometric/arithmetic midpoint grid across an operating envelope."""
    names = [s.name for s in symbols]
    missing = [n for n in names if n not in envelope]
    if missing:
        raise KeyError(f"Operating envelope missing bounds/values for symbol(s): {missing}")

    per_var: List[List[float]] = []
    for n in names:
        lo, hi = envelope[n]
        if abs(hi - lo) <= 1e-15 * max(abs(lo), abs(hi), 1.0):
            per_var.append([lo])
        elif lo > 0.0 and hi > 0.0:
            per_var.append([lo, math.sqrt(lo * hi), hi])
        else:
            per_var.append([lo, 0.5 * (lo + hi), hi])

    total = 1
    for pts in per_var:
        total *= len(pts)

    if total <= max_samples:
        combos = list(itertools.product(*per_var))
    else:
        # Deterministic corner/center fallback for high-dimensional envelopes
        rng = np.random.default_rng(20261127)
        combos = []
        # Include all-low, all-mid, all-high
        combos.append(tuple(pts[0] for pts in per_var))
        combos.append(tuple(pts[len(pts) // 2] for pts in per_var))
        combos.append(tuple(pts[-1] for pts in per_var))
        for _ in range(max_samples - 3):
            combos.append(tuple(pts[int(rng.integers(0, len(pts)))] for pts in per_var))

    arr = np.asarray(combos, dtype=float)
    return {n: arr[:, idx] for idx, n in enumerate(names)}


def check_equation_scales(
    equation: str,
    dims: Mapping[str, Union[str, Dimension]],
    operating_envelope: Mapping[str, Union[float, Tuple[float, float], Sequence[float]]],
    *,
    characteristic_scales: Optional[Mapping[str, float]] = None,
    scale_threshold: float = 100.0,
    negligible_ratio: float = 1e-6,
) -> ScaleBalanceReport:
    """Audit dimensional homogeneity AND order-of-magnitude term balance of an equation.

    Args:
        equation: Equation string such as ``"rho * a + grad_P = 0"`` or ``"rho * a = -grad_P"``.
        dims: Mapping of symbol name -> SI unit or base dimension string.
        operating_envelope: Mapping of symbol -> scalar or ``(min, max)`` interval.
        characteristic_scales: Optional characteristic scales (e.g. ``{"t": 72000.0, "z": 0.01}``)
            used if symbols are not already present in ``operating_envelope``.
        scale_threshold: Maximum allowable factor between balancing dominant terms (default 100x).
        negligible_ratio: Threshold below which a 3rd+ term is flagged as negligible (default 1e-6).
    """
    txt = str(equation).strip()
    if "=" in txt:
        lhs_txt, rhs_txt = txt.split("=", 1)
        lhs_expr = parse_expr(lhs_txt.strip())
        rhs_expr = parse_expr(rhs_txt.strip())
        combined = sp.expand(lhs_expr - rhs_expr)
    else:
        combined = sp.expand(parse_expr(txt))

    dim_map = {k: _parse_dim(v) for k, v in dims.items()}

    # Build normalized envelope
    env: Dict[str, Tuple[float, float]] = {}
    for k, v in (characteristic_scales or {}).items():
        fv = float(v)
        env[k] = (fv, fv)
    for k, v in operating_envelope.items():
        if isinstance(v, (int, float)):
            fv = float(v)
            env[k] = (fv, fv)
        else:
            seq = list(v)
            if len(seq) != 2:
                raise ValueError(f"Envelope entry '{k}' must be scalar or 2-sequence; got {v!r}")
            lo, hi = float(seq[0]), float(seq[1])
            env[k] = (min(lo, hi), max(lo, hi))

    raw_terms = list(sp.Add.make_args(combined))
    if not raw_terms or (len(raw_terms) == 1 and raw_terms[0] == 0):
        return ScaleBalanceReport(
            equation=equation,
            dimensionally_homogeneous=True,
            terms=[],
            dominant_balance_ratio=1.0,
            log10_gap=0.0,
            scale_threshold=scale_threshold,
            diagnostic_code="trivial_identity",
            status=VerificationStatus.WARNING,
            passed=True,
            warnings=["Equation reduces algebraically to 0 = 0."],
        )

    warnings: List[str] = []
    term_dims: List[Optional[Dimension]] = []
    dim_ok = True
    for term in raw_terms:
        tdim, errs = _expr_dimension(term, dim_map)
        if errs:
            dim_ok = False
            warnings.extend(errs)
        term_dims.append(tdim)

    valid_dims = [d for d in term_dims if d is not None]
    if valid_dims and any(d != valid_dims[0] for d in valid_dims[1:]):
        dim_ok = False
        warnings.append(
            "Terms are not dimensionally homogeneous: "
            + ", ".join(f"{t} [{d}]" for t, d in zip(raw_terms, term_dims))
        )

    all_syms = sorted(combined.free_symbols, key=lambda s: s.name)
    samples = _sample_envelope(env, all_syms) if all_syms else {}

    eval_stats: List[Tuple[sp.Expr, Optional[Dimension], float, float, float]] = []
    for term, tdim in zip(raw_terms, term_dims):
        t_syms = sorted(term.free_symbols, key=lambda s: s.name)
        if not t_syms:
            val = abs(float(term.evalf()))
            eval_stats.append((term, tdim, val, val, val))
        else:
            fn = sp.lambdify(t_syms, term, modules="numpy")
            vals = np.abs(np.asarray(fn(*[samples[s.name] for s in t_syms]), dtype=float))
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                raise ValueError(f"Term '{term}' evaluated to non-finite values over operating envelope.")
            mn = float(np.min(vals))
            mx = float(np.max(vals))
            typ = math.sqrt(mn * mx) if (mn > 0.0 and mx > 0.0) else 0.5 * (mn + mx)
            eval_stats.append((term, tdim, mn, mx, typ))

    # Rank terms by typical magnitude
    order_idx = sorted(range(len(eval_stats)), key=lambda i: eval_stats[i][4], reverse=True)
    lead_typ = max(eval_stats[order_idx[0]][4], 1e-300)

    if len(eval_stats) == 1:
        # Single non-zero term equated to zero
        t0, d0, mn0, mx0, typ0 = eval_stats[0]
        entry = TermMagnitudeEntry(
            term_expr=str(t0),
            dimension=str(d0) if d0 is not None else "unknown",
            min_abs=mn0,
            max_abs=mx0,
            typical_abs=typ0,
            ratio_to_leading=1.0,
            role="unbalanced_orphan",
        )
        return ScaleBalanceReport(
            equation=equation,
            dimensionally_homogeneous=dim_ok,
            terms=[entry],
            dominant_balance_ratio=float("inf"),
            log10_gap=float("inf"),
            scale_threshold=scale_threshold,
            diagnostic_code="single_nonzero_term",
            status=VerificationStatus.FAILED,
            passed=False,
            warnings=warnings + ["Equation has a single non-zero additive term equated to zero."],
        )

    i1, i2 = order_idx[0], order_idx[1]
    _, _, mn1, mx1, typ1 = eval_stats[i1]
    _, _, mn2, mx2, typ2 = eval_stats[i2]

    # Interval separation between the top two terms
    if mn1 > mx2 and mx2 > 0.0:
        interval_gap = mn1 / mx2
    elif mn2 > mx1 and mx1 > 0.0:
        interval_gap = mn2 / mx1
    elif mn1 > 0.0 and mx2 == 0.0:
        interval_gap = float("inf")
    else:
        interval_gap = 1.0

    dom_ratio = typ1 / max(typ2, 1e-300)
    effective_gap = max(interval_gap, 1.0)
    log10_gap = math.log10(dom_ratio) if dom_ratio > 0.0 and math.isfinite(dom_ratio) else float("inf")

    entries: List[TermMagnitudeEntry] = []
    has_negligible = False
    for idx, (term, tdim, mn, mx, typ) in enumerate(eval_stats):
        r_lead = typ / lead_typ
        if idx == i1 and effective_gap > scale_threshold:
            role = "unbalanced_orphan"
        elif r_lead >= 1.0 / scale_threshold:
            role = "dominant"
        elif r_lead >= negligible_ratio:
            role = "subleading"
        else:
            role = "negligible"
            has_negligible = True
        entries.append(
            TermMagnitudeEntry(
                term_expr=str(term),
                dimension=str(tdim) if tdim is not None else "unknown",
                min_abs=mn,
                max_abs=mx,
                typical_abs=typ,
                ratio_to_leading=r_lead,
                role=role,
            )
        )

    if not dim_ok:
        status = VerificationStatus.FAILED
        code = "dimensionally_inhomogeneous"
    elif effective_gap > scale_threshold:
        status = VerificationStatus.FAILED
        code = "unbalanced_dominant_terms"
        warnings.append(
            f"Leading term '{eval_stats[i1][0]}' exceeds runner-up '{eval_stats[i2][0]}' by "
            f"{dom_ratio:.2e}x (log10 gap = {log10_gap:.2f}, interval gap = {effective_gap:.2e}x > {scale_threshold:g}x)."
        )
    elif has_negligible:
        status = VerificationStatus.WARNING
        code = "negligible_asymptotic_term"
        warnings.append(
            f"Equation balances at leading order, but contains term(s) < {negligible_ratio:.1e} of leading scale."
        )
    else:
        status = VerificationStatus.PASSED
        code = "scale_balanced"

    return ScaleBalanceReport(
        equation=equation,
        dimensionally_homogeneous=dim_ok,
        terms=entries,
        dominant_balance_ratio=float(dom_ratio),
        log10_gap=float(log10_gap),
        scale_threshold=float(scale_threshold),
        diagnostic_code=code,
        status=status,
        passed=(status == VerificationStatus.PASSED),
        warnings=warnings,
        assumptions=[
            f"Scale-balance threshold: {scale_threshold:g}x; negligible-term threshold: {negligible_ratio:.1e}.",
            "Evaluated deterministically over operating-envelope corner and geometric-midpoint samples.",
        ],
    )


@dataclass
class MissingConstantHypothesis(ResultMixin):
    """Structured diagnostic for an implicit unit normalization or differential mismatch."""

    location: str
    issue_kind: str  # "implicit_unit_normalization" | "differential_geometry_mismatch" | "equation_dimension_mismatch"
    observed_dimension: str
    required_dimension: str
    missing_factor_dimension: str
    suggested_symbol: str
    explanation: str


@dataclass
class NormalizationDiagnosticReport(ResultMixin):
    """Report diagnosing implicit unit normalizations and differential mismatches in draft equations."""

    equation: str
    consistent: bool
    hypotheses: List[MissingConstantHypothesis]
    status: VerificationStatus
    passed: bool
    warnings: List[str] = field(default_factory=list)


_DIFF_TOKEN_RE = re.compile(r"^d([A-Za-z][A-Za-z0-9_]*)$")


def diagnose_normalization_shortcuts(
    equation: str,
    dims: Mapping[str, Union[str, Dimension]],
) -> NormalizationDiagnosticReport:
    """Inspect a draft equation for implicit normalizations (e.g. `1 - rho*V`) or
    differential-order geometry mistakes (e.g. `sigma = A * dz`).
    """
    dim_map = {k: _parse_dim(v) for k, v in dims.items()}
    txt = str(equation).strip()
    hypotheses: List[MissingConstantHypothesis] = []
    warnings: List[str] = []

    if "=" in txt:
        lhs_txt, rhs_txt = txt.split("=", 1)
        lhs_expr = parse_expr(lhs_txt.strip())
        rhs_expr = parse_expr(rhs_txt.strip())
        nodes_to_scan = [lhs_expr, rhs_expr]
    else:
        lhs_expr = None
        rhs_expr = parse_expr(txt)
        nodes_to_scan = [rhs_expr]

    # 1. Scan all Add sub-expressions for dimensionless literal + dimensional term
    visited: set[str] = set()
    for root in nodes_to_scan:
        for node in sp.preorder_traversal(root):
            if isinstance(node, sp.Add):
                key = str(node)
                if key in visited:
                    continue
                visited.add(key)
                term_infos: List[Tuple[sp.Expr, Optional[Dimension]]] = []
                for arg in node.args:
                    if arg == 0:
                        continue
                    d, _ = _expr_dimension(arg, dim_map)
                    term_infos.append((arg, d))
                dimless = [(a, d) for a, d in term_infos if d is not None and d.is_dimensionless]
                dimful = [(a, d) for a, d in term_infos if d is not None and not d.is_dimensionless]
                if dimless and dimful:
                    ref_expr, ref_dim = dimful[0]
                    for lit_expr, _ in dimless:
                        hypotheses.append(
                            MissingConstantHypothesis(
                                location=str(node),
                                issue_kind="implicit_unit_normalization",
                                observed_dimension="1",
                                required_dimension=str(ref_dim),
                                missing_factor_dimension=str(ref_dim),
                                suggested_symbol="c_ref",
                                explanation=(
                                    f"Dimensionless term '{lit_expr}' added to dimensional term '{ref_expr}' [{ref_dim}]; "
                                    f"replace '{lit_expr}' with a reference scale of dimension [{ref_dim}]."
                                ),
                            )
                        )

    # 2. Check LHS vs RHS dimensional mismatch (including differential factor like A*dz)
    # Note: literal zero (`= 0`) is dimensionally compatible with any physical dimension.
    if lhs_expr is not None and lhs_expr != 0 and rhs_expr != 0:
        d_lhs, errs_l = _expr_dimension(lhs_expr, dim_map)
        d_rhs, errs_r = _expr_dimension(rhs_expr, dim_map)
        warnings.extend(errs_l + errs_r)
        if d_lhs is not None and d_rhs is not None and d_lhs != d_rhs:
            diff_ratio = d_rhs / d_lhs
            # Check if RHS contains a differential symbol d<var> whose dimension accounts for diff_ratio
            diff_syms = [
                s for s in rhs_expr.free_symbols
                if isinstance(s, sp.Symbol) and _DIFF_TOKEN_RE.match(s.name) and s.name in dim_map
            ]
            is_diff_geom = any(dim_map[s.name] == diff_ratio for s in diff_syms)
            if is_diff_geom:
                culprit = next(s.name for s in diff_syms if dim_map[s.name] == diff_ratio)
                hypotheses.append(
                    MissingConstantHypothesis(
                        location=equation,
                        issue_kind="differential_geometry_mismatch",
                        observed_dimension=str(d_rhs),
                        required_dimension=str(d_lhs),
                        missing_factor_dimension=str(d_lhs / d_rhs),
                        suggested_symbol=culprit,
                        explanation=(
                            f"RHS '{rhs_expr}' [{d_rhs}] exceeds LHS '{lhs_expr}' [{d_lhs}] by differential "
                            f"factor '{culprit}' [{diff_ratio}]; remove '{culprit}' or treat LHS as a differential element."
                        ),
                    )
                )
            else:
                hypotheses.append(
                    MissingConstantHypothesis(
                        location=equation,
                        issue_kind="equation_dimension_mismatch",
                        observed_dimension=str(d_rhs),
                        required_dimension=str(d_lhs),
                        missing_factor_dimension=str(d_lhs / d_rhs),
                        suggested_symbol="kappa_dim",
                        explanation=(
                            f"LHS '{lhs_expr}' [{d_lhs}] != RHS '{rhs_expr}' [{d_rhs}]; "
                            f"missing dimensional factor [{d_lhs / d_rhs}]."
                        ),
                    )
                )

    consistent = (len(hypotheses) == 0) and (len(warnings) == 0)
    status = VerificationStatus.PASSED if consistent else VerificationStatus.FAILED
    return NormalizationDiagnosticReport(
        equation=equation,
        consistent=consistent,
        hypotheses=hypotheses,
        status=status,
        passed=consistent,
        warnings=warnings,
    )


@dataclass
class HomogeneityAuditReport(ResultMixin):
    """Audit of Euler dimensional elasticity sum rule (D * E = d_y) and Pi-design excitation."""

    target_dim: str
    elasticities: Dict[str, float]
    euler_residuals: Dict[str, float]
    max_abs_residual: float
    zero_intercept_admissible: bool
    pi_group_expressions: List[str] = field(default_factory=list)
    pi_design_rank: Optional[int] = None
    pi_expected_rank: Optional[int] = None
    pi_collinear: bool = False
    status: VerificationStatus = VerificationStatus.PASSED
    passed: bool = True
    warnings: List[str] = field(default_factory=list)


def check_homogeneity_elasticities(
    target_dim: Union[str, Dimension],
    input_dims: Mapping[str, Union[str, Dimension]],
    elasticities: Mapping[str, float],
    *,
    tol: float = 0.05,
    data: Optional[Mapping[str, Sequence[float]]] = None,
) -> HomogeneityAuditReport:
    """Verify Euler's dimensional homogeneity sum rule D * E == d_y and optional log-Pi excitation rank.

    If a zero-intercept law y = f(x_1, ..., x_m) has no omitted dimensional constant,
    its logarithmic elasticities E_i = (x_i/y)(dy/dx_i) MUST satisfy D * E = d_y for every
    base dimension. Violation proves a non-zero baseline offset y_0 or omitted scale is required.
    """
    d_target = _parse_dim(target_dim)
    d_inputs = {k: _parse_dim(v) for k, v in input_dims.items()}
    names = list(d_inputs.keys())
    basis_set = set(d_target.bases())
    for d in d_inputs.values():
        basis_set.update(d.bases())
    basis = sorted(basis_set)

    residuals: Dict[str, float] = {}
    for b in basis:
        lhs_b = sum(float(d_inputs[n].to_dict().get(b, 0)) * float(elasticities.get(n, 0.0)) for n in names)
        rhs_b = float(d_target.to_dict().get(b, 0))
        residuals[b] = lhs_b - rhs_b

    max_res = max((abs(v) for v in residuals.values()), default=0.0)
    admissible = max_res <= tol
    warnings: List[str] = []
    if not admissible:
        warnings.append(
            f"Empirical elasticities violate zero-intercept Euler homogeneity sum rule (max |D*E - d_y| = {max_res:.3f} > {tol:g}); "
            "a non-zero dimensional baseline offset or additional scale parameter is mathematically required."
        )

    # Extract Buckingham-Pi groups over inputs and optionally check log-Pi excitation rank
    scope_pairs = [(n, d_inputs[n]) for n in names]
    dim_sol = DimensionalEngine.solve(d_target, scope_pairs)
    sym_map = {n: sp.Symbol(n, positive=True) for n in names}
    pi_exprs = dim_sol.pi_monomials(sym_map) if dim_sol.consistent else []
    pi_strs = [str(p) for p in pi_exprs]

    pi_rank: Optional[int] = None
    pi_expected: Optional[int] = len(pi_exprs) if pi_exprs else None
    pi_collinear = False

    if data is not None and pi_exprs:
        # Evaluate log(Pi_j) and log(inputs) variation across dataset
        pi_cols: List[np.ndarray] = []
        for p_e in pi_exprs:
            p_syms = sorted(p_e.free_symbols, key=lambda s: s.name)
            fn = sp.lambdify(p_syms, p_e, modules="numpy")
            vals = np.asarray(fn(*[np.asarray(data[s.name], dtype=float) for s in p_syms]), dtype=float)
            if np.any(vals <= 0.0):
                continue
            log_v = np.log(vals)
            pi_cols.append(log_v - np.mean(log_v))

        # Check collinearity between log(primary variable) and log(Pi_j)
        if pi_cols:
            mat = np.column_stack(pi_cols)
            svals = np.linalg.svd(mat, compute_uv=False)
            smax = float(svals[0]) if svals.size else 0.0
            pi_rank = int(np.sum(svals > 1e-3 * max(smax, 1e-12)))
            # Also check if log(Pi) span is narrow (std < 0.08) or collinear with log(x_0)
            if pi_rank < len(pi_cols):
                pi_collinear = True
                warnings.append(
                    f"Buckingham-Pi design matrix is rank-deficient (rank {pi_rank} < {len(pi_cols)})."
                )
            elif len(pi_cols) == 1 and names and names[0] in data:
                x0 = np.asarray(data[names[0]], dtype=float)
                if np.all(x0 > 0.0):
                    lx0 = np.log(x0) - np.mean(np.log(x0))
                    denom = float(np.linalg.norm(lx0) * np.linalg.norm(pi_cols[0]))
                    corr = abs(float(np.dot(lx0, pi_cols[0]) / denom)) if denom > 1e-15 else 1.0
                    if corr > 0.985:
                        pi_collinear = True
                        warnings.append(
                            f"Dataset log(Pi_1) is near-collinear with log({names[0]}) (|r|={corr:.4f}); "
                            "independent multi-factor variation is required to identify Pi-group exponent."
                        )

    if not admissible:
        status = VerificationStatus.FAILED
    elif pi_collinear:
        status = VerificationStatus.WARNING
    else:
        status = VerificationStatus.PASSED

    return HomogeneityAuditReport(
        target_dim=str(d_target),
        elasticities={k: float(v) for k, v in elasticities.items()},
        euler_residuals=residuals,
        max_abs_residual=float(max_res),
        zero_intercept_admissible=admissible,
        pi_group_expressions=pi_strs,
        pi_design_rank=pi_rank,
        pi_expected_rank=pi_expected,
        pi_collinear=pi_collinear,
        status=status,
        passed=(status == VerificationStatus.PASSED),
        warnings=warnings,
    )
