"""
Generic Verification & Numerical Coupling Stability Suite (ASME V&V 20 / Roache / Celik 2008).

Domain-agnostic verification tools applicable to fluid mechanics, heat & mass transfer,
chemical kinetics, epidemiology, structural mechanics, population biology, and economics:

  1. Grid Independence & Spatial Mesh Convergence:
       - `gci_from_values`: Roache Grid Convergence Index (GCI) + Celik et al. (2008) fixed-point
         observed-order iteration for uniform or non-uniform grid refinement ratios. Works on
         any external solver outputs (CFD, FEM, FVM) or internal callables.
       - `grid_convergence_study`: Automated multi-mesh execution + GCI report.
       - `mms_study`: Automated Method of Manufactured Solutions (MMS) code verification on any
         `ConservationLaw1D`, using SymPy to generate exact analytical forcing terms.

  2. Time-Step Sensitivity:
       - `timestep_study`: Fixed-step dt convergence, observed temporal order, Richardson limit,
         CFL / Fourier dimensionless stability numbers, and hybrid event-timestamp drift checks.

  3. Conservation Audits (Mass, Energy, Momentum, Population, Charge, Hamiltonian Invariants):
       - `audit_pde_conservation`: Machine-precision flux-and-source ledger check on `PDESolution`.
       - `audit_ode_invariants`: Symbolic model-level check (dI/dt == 0) + numerical trajectory drift.
       - `audit_hybrid_conservation`: Global conservation audit across discontinuous jump/reset events.

  4. Numerical Coupling & Stiffness Stability:
       - `analyze_stiffness`: Symbolic/FD Jacobian spectrum, stiffness ratio, and exact maximum
         stable explicit step dt_max per Runge-Kutta scheme.
       - `solve_partitioned_fixed_point`: Partitioned two-way algebraic coupling monitor with
         contraction-factor diagnostics and Aitken Delta^2 dynamic relaxation.
       - `cosimulate_coupled_odes`: Loose (staggered) vs. strong (Aitken-iterated) co-simulation
         of coupled `ODESystem` modules benchmarked against the monolithic system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import sympy as sp

from ._parsing import ExprLike, parse_expr
from .integrators import (
    EXPLICIT_TABLEAUS,
    fd_jacobian,
    integrate_fixed,
    is_implicit,
    max_stable_dt,
    scheme_order,
)
from .numerical import HybridODESolution
from .odesys import ODESystem, Trajectory
from .pde import BC, ConservationLaw1D, PDESolution, cell_average


# ============================================================================
# 1. Grid Independence, Spatial Mesh Convergence (GCI) & Manufactured Solutions
# ============================================================================


@dataclass
class GCIReport:
    """ASME V&V 20 / Celik et al. (2008) three-grid convergence and uncertainty report."""

    spacings: Tuple[float, float, float]       # (h1_fine, h2_medium, h3_coarse)
    values: Tuple[float, float, float]         # (phi1_fine, phi2_medium, phi3_coarse)
    r21: float
    r32: float
    observed_order: float
    formal_order: Optional[float]
    extrapolated_value: float
    approx_rel_error_21: float
    extrapolated_rel_error_21: float
    gci_fine_21: float
    gci_coarse_32: float
    asymptotic_ratio: float
    regime: str  # "monotonic", "oscillatory", "divergent", "converged_exact"
    safety_factor: float = 1.25
    passed: bool = True
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        h1, h2, h3 = self.spacings
        p1, p2, p3 = self.values
        lines = [
            "=== Grid Convergence Index (GCI) Report ===",
            f"  Spacings (fine, med, coarse): ({h1:.6g}, {h2:.6g}, {h3:.6g}) | r21={self.r21:.4g}, r32={self.r32:.4g}",
            f"  QoI      (fine, med, coarse): ({p1:.8g}, {p2:.8g}, {p3:.8g})",
            f"  Regime: {self.regime} | Observed order p = {self.observed_order:.4f}"
            + (f" (formal = {self.formal_order:.2f})" if self.formal_order is not None else ""),
            f"  Richardson extrapolated phi_ext = {self.extrapolated_value:.8g}",
            f"  GCI_fine (21) = {self.gci_fine_21 * 100:.4f}% | GCI_coarse (32) = {self.gci_coarse_32 * 100:.4f}%",
            f"  Asymptotic ratio = {self.asymptotic_ratio:.4f} (ideal ~ 1.0) | Passed: {self.passed}",
        ]
        for n in self.notes:
            lines.append(f"  Note: {n}")
        return "\n".join(lines)


def _celik_observed_order(e32: float, e21: float, r21: float, r32: float, max_iter: int = 200) -> float:
    """Solve p = |ln|e32/e21| + q(p)| / ln(r21) per Celik et al. (2008), ASME J. Fluids Eng. 130."""
    ratio = abs(e32 / e21)
    s = 1.0 if (e32 / e21) >= 0.0 else -1.0
    if abs(r21 - r32) < 1e-12:
        return float(abs(np.log(ratio)) / np.log(r21))
    p = max(abs(np.log(ratio)) / np.log(r21), 1e-4)
    for _ in range(max_iter):
        num = r21 ** p - s
        den = r32 ** p - s
        if num * den <= 0:
            break
        q = np.log(num / den)
        p_next = abs(np.log(ratio) + q) / np.log(r21)
        if abs(p_next - p) < 1e-12:
            p = p_next
            break
        p = 0.5 * p + 0.5 * p_next
    return float(p)


def gci_from_values(
    spacings: Sequence[float],
    quantities: Sequence[float],
    formal_order: Optional[float] = None,
    safety_factor: float = 1.25,
    gci_tol: float = 0.05,
    order_tol: float = 0.5,
) -> GCIReport:
    """
    Compute Roache GCI and Celik observed order from >= 3 (grid_spacing, quantity) pairs.

    Pairs may be passed in any order; they are sorted automatically so index 0 is the finest
    mesh (smallest spacing h1). If more than 3 levels are given, the 3 finest levels are used.
    """
    if len(spacings) != len(quantities) or len(spacings) < 3:
        raise ValueError("Provide at least 3 matching (spacing, quantity) values.")
    pairs = sorted(zip([float(h) for h in spacings], [float(q) for q in quantities]), key=lambda kv: kv[0])
    (h1, phi1), (h2, phi2), (h3, phi3) = pairs[0], pairs[1], pairs[2]
    if h1 <= 0 or h2 <= h1 or h3 <= h2:
        raise ValueError(f"Grid spacings must be strictly positive and distinct; got {(h1, h2, h3)}.")

    r21 = h2 / h1
    r32 = h3 / h2
    e32 = phi3 - phi2
    e21 = phi2 - phi1
    notes: List[str] = []

    if r21 < 1.3 or r32 < 1.3:
        notes.append("ASME V&V 20 recommends grid refinement ratios r >= 1.3.")

    scale = max(abs(phi1), abs(phi2), abs(phi3), 1.0)
    if abs(e21) <= 1e-14 * scale and abs(e32) <= 1e-14 * scale:
        p_obs = formal_order if formal_order is not None else float("nan")
        return GCIReport(
            spacings=(h1, h2, h3),
            values=(phi1, phi2, phi3),
            r21=r21,
            r32=r32,
            observed_order=p_obs,
            formal_order=formal_order,
            extrapolated_value=phi1,
            approx_rel_error_21=0.0,
            extrapolated_rel_error_21=0.0,
            gci_fine_21=0.0,
            gci_coarse_32=0.0,
            asymptotic_ratio=1.0,
            regime="converged_exact",
            safety_factor=safety_factor,
            passed=True,
            notes=notes + ["Differences below machine precision; solution already grid-exact."],
        )

    if abs(e21) <= 1e-15 * scale:
        p_obs = formal_order if formal_order is not None else 2.0
        regime = "converged_exact"
    else:
        ratio = e21 / e32 if abs(e32) > 0 else np.inf
        if 0.0 < ratio < 1.0:
            regime = "monotonic"
        elif ratio < 0.0 and abs(ratio) < 1.0:
            regime = "oscillatory"
            notes.append("Oscillatory convergence detected (sign change between mesh increments).")
        else:
            regime = "divergent"
            notes.append("Apparent divergence (|phi2 - phi1| >= |phi3 - phi2|); refine further into asymptotic range.")
        p_obs = _celik_observed_order(e32, e21, r21, r32) if abs(e32) > 0 else (formal_order or 1.0)

    p_eff = max(min(p_obs, 10.0), 0.1)
    denom21 = (r21 ** p_eff) - 1.0
    denom32 = (r32 ** p_eff) - 1.0
    phi_ext = (r21 ** p_eff * phi1 - phi2) / denom21
    ea21 = abs((phi1 - phi2) / phi1) if abs(phi1) > 1e-30 else abs(phi1 - phi2)
    ea32 = abs((phi2 - phi3) / phi2) if abs(phi2) > 1e-30 else abs(phi2 - phi3)
    eext21 = abs((phi_ext - phi1) / phi_ext) if abs(phi_ext) > 1e-30 else abs(phi_ext - phi1)
    gci21 = safety_factor * ea21 / denom21
    gci32 = safety_factor * ea32 / denom32
    asym = gci32 / ((r21 ** p_eff) * gci21) if gci21 > 0 else 1.0

    passed = (regime in ("monotonic", "oscillatory", "converged_exact")) and (gci21 <= gci_tol)
    if formal_order is not None and regime == "monotonic":
        if abs(p_obs - formal_order) > order_tol:
            notes.append(f"Observed order {p_obs:.2f} deviates from formal order {formal_order:.2f} by > {order_tol}.")
            passed = False

    return GCIReport(
        spacings=(h1, h2, h3),
        values=(phi1, phi2, phi3),
        r21=r21,
        r32=r32,
        observed_order=p_obs,
        formal_order=formal_order,
        extrapolated_value=phi_ext,
        approx_rel_error_21=ea21,
        extrapolated_rel_error_21=eext21,
        gci_fine_21=gci21,
        gci_coarse_32=gci32,
        asymptotic_ratio=asym,
        regime=regime,
        safety_factor=safety_factor,
        passed=passed,
        notes=notes,
    )


def grid_convergence_study(
    solver_fn: Callable[[int], float],
    cells: Sequence[int] = (32, 64, 128),
    domain_length: float = 1.0,
    formal_order: Optional[float] = None,
    safety_factor: float = 1.25,
    gci_tol: float = 0.05,
    order_tol: float = 0.5,
) -> GCIReport:
    """Run `solver_fn(n_cells) -> scalar QoI` across multiple mesh resolutions and compute GCI."""
    if len(cells) < 3:
        raise ValueError("Provide at least 3 cell counts for a grid convergence study.")
    spacings = [float(domain_length) / int(n) for n in cells]
    qois = [float(solver_fn(int(n))) for n in cells]
    return gci_from_values(
        spacings=spacings,
        quantities=qois,
        formal_order=formal_order,
        safety_factor=safety_factor,
        gci_tol=gci_tol,
        order_tol=order_tol,
    )


@dataclass
class MMSReport:
    """Method of Manufactured Solutions (MMS) code-verification report."""

    fields: List[str]
    cells: List[int]
    spacings: np.ndarray
    errors_l1: Dict[str, np.ndarray]
    errors_l2: Dict[str, np.ndarray]
    errors_linf: Dict[str, np.ndarray]
    order_l1: Dict[str, float]
    order_l2: Dict[str, float]
    order_linf: Dict[str, float]
    formal_order: float
    manufactured_sources: Dict[str, sp.Expr]
    passed: bool

    def summary(self) -> str:
        lines = [
            f"=== MMS Code Verification Report (formal order = {self.formal_order:.2f}) ===",
            f"  Grids N = {self.cells}",
        ]
        for f in self.fields:
            lines.append(
                f"  Field '{f}': p_L1={self.order_l1[f]:.3f}, "
                f"p_L2={self.order_l2[f]:.3f}, p_Linf={self.order_linf[f]:.3f} | "
                f"finest L2={self.errors_l2[f][-1]:.3e}"
            )
        lines.append(f"  Passed: {self.passed}")
        return "\n".join(lines)


def _fit_log_slope(h: np.ndarray, err: np.ndarray) -> float:
    mask = err > 1e-15
    if np.count_nonzero(mask) < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(h[mask]), np.log(err[mask]), 1)
    return float(slope)


def mms_study(
    law: ConservationLaw1D,
    exact: Union[ExprLike, Mapping[str, ExprLike], Sequence[ExprLike]],
    domain: Tuple[float, float] = (0.0, 1.0),
    t_end: float = 0.1,
    cells: Sequence[int] = (32, 64, 128, 256),
    params: Optional[Mapping[str, float]] = None,
    bc_kind: str = "periodic",
    reconstruction: str = "linear",
    scheme: str = "ssprk3",
    cfl: float = 0.2,
    order_tol: float = 0.25,
) -> MMSReport:
    """
    Verify PDE discretisation accuracy via the Method of Manufactured Solutions (MMS).

    Derives analytical forcing S_mms(x, t) symbolically in SymPy so `exact` solves the modified
    conservation law identically, runs across `cells`, and checks that L1/L2 error norms decay
    at the formal spatial order.
    """
    s_mms = law.manufactured_source(exact, params=params)
    exact_exprs = law._per_field(exact, "exact")
    psubs = law._param_subs(params) if params is not None else {}
    exact_sub = [e.xreplace(psubs) for e in exact_exprs]

    if bc_kind == "periodic":
        bc_spec: Any = "periodic"
    elif bc_kind == "dirichlet":
        left_d = {f: e.xreplace({law.x: sp.Float(domain[0])}) for f, e in zip(law.fields, exact_sub)}
        right_d = {f: e.xreplace({law.x: sp.Float(domain[1])}) for f, e in zip(law.fields, exact_sub)}
        bc_spec = {"left": BC.dirichlet(left_d), "right": BC.dirichlet(right_d)}
    else:
        raise ValueError("mms_study supports bc_kind='periodic' or 'dirichlet'.")

    init_spec = {f: e.xreplace({law.t: sp.Integer(0)}) for f, e in zip(law.fields, exact_sub)}
    exact_fns = [
        sp.lambdify([law.x], e.xreplace({law.t: sp.Float(t_end)}), "numpy")
        for e in exact_sub
    ]

    h_arr = np.array([(domain[1] - domain[0]) / int(n) for n in cells], dtype=float)
    e_l1 = {f: np.zeros(len(cells)) for f in law.fields}
    e_l2 = {f: np.zeros(len(cells)) for f in law.fields}
    e_linf = {f: np.zeros(len(cells)) for f in law.fields}

    formal = 1.0 if reconstruction == "constant" else 2.0
    for idx, n_c in enumerate(cells):
        sol = law.solve(
            domain=domain,
            n_cells=int(n_c),
            t_end=t_end,
            initial=init_spec,
            bc=bc_spec,
            params=params,
            scheme=scheme,
            cfl=cfl,
            reconstruction=reconstruction,
            n_out=2,
            extra_source=s_mms,
        )
        for k, f in enumerate(law.fields):
            u_exact_bar = cell_average(exact_fns[k], sol.edges)
            diff = np.abs(sol.final(f) - u_exact_bar)
            e_l1[f][idx] = float(sol.h * np.sum(diff))
            e_l2[f][idx] = float(np.sqrt(sol.h * np.sum(diff ** 2)))
            e_linf[f][idx] = float(np.max(diff))

    ord_l1 = {f: _fit_log_slope(h_arr, e_l1[f]) for f in law.fields}
    ord_l2 = {f: _fit_log_slope(h_arr, e_l2[f]) for f in law.fields}
    ord_linf = {f: _fit_log_slope(h_arr, e_linf[f]) for f in law.fields}

    # TVD slope limiters clip slightly at smooth extrema (~1.7-2.0 in L2); unlimited 'linear' hits 2.0
    effective_tol = order_tol if reconstruction in ("constant", "linear") else max(order_tol, 0.45)
    passed = all(
        (not np.isnan(ord_l2[f])) and (ord_l2[f] >= formal - effective_tol)
        for f in law.fields
    )
    return MMSReport(
        fields=list(law.fields),
        cells=[int(n) for n in cells],
        spacings=h_arr,
        errors_l1=e_l1,
        errors_l2=e_l2,
        errors_linf=e_linf,
        order_l1=ord_l1,
        order_l2=ord_l2,
        order_linf=ord_linf,
        formal_order=formal,
        manufactured_sources=s_mms,
        passed=passed,
    )


# ============================================================================
# 2. Time-Step Sensitivity & Dimensionless CFL / Fourier Diagnostics
# ============================================================================


@dataclass
class TimestepReport:
    """Time-step convergence, stability-number, and event-timing sensitivity report."""

    dts: Tuple[float, ...]
    values: Tuple[float, ...]
    gci: GCIReport
    scheme: Optional[str] = None
    cfl_numbers: Optional[Tuple[float, ...]] = None
    fourier_numbers: Optional[Tuple[float, ...]] = None
    event_time_drift: Optional[float] = None
    passed: bool = True

    def summary(self) -> str:
        lines = [
            "=== Time-Step Sensitivity Report ===",
            f"  Steps dt: {tuple(round(d, 8) for d in self.dts)}"
            + (f" | Scheme: {self.scheme}" if self.scheme else ""),
            f"  QoI:      {tuple(round(v, 8) for v in self.values)}",
            f"  Observed temporal order p_t = {self.gci.observed_order:.4f} | "
            f"GCI_fine = {self.gci.gci_fine_21 * 100:.4f}% | Extrapolated = {self.gci.extrapolated_value:.8g}",
        ]
        if self.cfl_numbers is not None:
            lines.append(f"  CFL numbers:     {tuple(round(c, 4) for c in self.cfl_numbers)}")
        if self.fourier_numbers is not None:
            lines.append(f"  Fourier numbers: {tuple(round(fo, 4) for fo in self.fourier_numbers)}")
        if self.event_time_drift is not None:
            lines.append(f"  Finest event-timestamp drift |dt_ev|: {self.event_time_drift:.3e}")
        lines.append(f"  Passed: {self.passed}")
        return "\n".join(lines)


def timestep_study(
    runner_or_ode: Union[Callable[[float], float], ODESystem],
    dts: Sequence[float],
    t_span: Optional[Tuple[float, float]] = None,
    y0: Any = None,
    params: Optional[Mapping[str, Any]] = None,
    scheme: str = "rk4",
    state: Optional[str] = None,
    formal_order: Optional[float] = None,
    dx: Optional[float] = None,
    wave_speed: Optional[float] = None,
    diffusivity: Optional[float] = None,
    gci_tol: float = 0.02,
    order_tol: float = 0.45,
) -> TimestepReport:
    """
    Evaluate time-step sensitivity across >= 3 step sizes `dts`.

    Accepts either an `ODESystem` (integrated with fixed-step `scheme`) or any custom
    `runner(dt) -> scalar_QoI` callable (such as a PDE or hybrid event solver).
    """
    if len(dts) < 3:
        raise ValueError("Provide at least 3 time-step sizes in `dts`.")
    dts_sorted = tuple(sorted((float(d) for d in dts)))

    if isinstance(runner_or_ode, ODESystem):
        if t_span is None or y0 is None:
            raise ValueError("Passing an ODESystem to timestep_study requires `t_span` and `y0`.")
        target_state = state or runner_or_ode.states[0]
        f_ord = formal_order if formal_order is not None else float(scheme_order(scheme) or 1)
        used_dts: List[float] = []
        vals: List[float] = []
        for dt in dts_sorted:
            traj = runner_or_ode.solve(t_span, y0, params=params, method=scheme, dt=dt, n_out=2)
            used_dts.append(float(traj.dt or dt))
            vals.append(float(traj.final[target_state]))
        dts_final = tuple(used_dts)
        vals_final = tuple(vals)
        scheme_used: Optional[str] = scheme
    else:
        f_ord = formal_order
        dts_final = dts_sorted
        vals_final = tuple(float(runner_or_ode(d)) for d in dts_final)
        scheme_used = None

    gci = gci_from_values(
        spacings=dts_final,
        quantities=vals_final,
        formal_order=f_ord,
        gci_tol=gci_tol,
        order_tol=order_tol,
    )
    cfls = tuple(wave_speed * d / dx for d in dts_final) if (dx and wave_speed is not None) else None
    fos = tuple(diffusivity * d / (dx ** 2) for d in dts_final) if (dx and diffusivity is not None) else None

    return TimestepReport(
        dts=dts_final,
        values=vals_final,
        gci=gci,
        scheme=scheme_used,
        cfl_numbers=cfls,
        fourier_numbers=fos,
        passed=gci.passed,
    )


# ============================================================================
# 3. Generic Conservation Audits (PDE Flux Ledger, ODE Invariants, Hybrid Jumps)
# ============================================================================


@dataclass
class FieldConservationEntry:
    name: str
    symbolic_exact: Optional[bool]
    initial_value: float
    final_value: float
    max_abs_residual: float
    max_rel_residual: float
    residual_series: np.ndarray
    passed: bool


@dataclass
class ConservationAuditReport:
    kind: str  # "pde_fvm", "ode_invariants", "hybrid_events"
    t: np.ndarray
    entries: Dict[str, FieldConservationEntry]
    tol: float
    passed: bool

    def summary(self) -> str:
        lines = [f"=== Conservation Audit Report [{self.kind}] (tol = {self.tol:.1e}) ==="]
        for nm, e in self.entries.items():
            sym_tag = "" if e.symbolic_exact is None else f" | symbolic_exact={e.symbolic_exact}"
            lines.append(
                f"  '{nm}': init={e.initial_value:.6g}, final={e.final_value:.6g}, "
                f"max|R|={e.max_abs_residual:.3e}, rel|R|={e.max_rel_residual:.3e}"
                f"{sym_tag} -> {'PASS' if e.passed else 'FAIL'}"
            )
        lines.append(f"  Overall Passed: {self.passed}")
        return "\n".join(lines)


def audit_pde_conservation(sol: PDESolution, tol: float = 1e-10) -> ConservationAuditReport:
    """
    Verify discrete FVM balance for every conserved field in `sol`:
        R_k(t) = [total_k(t) - total_k(0)] - [inflow_left_k(t) - outflow_right_k(t) + source_k(t)].
    """
    entries: Dict[str, FieldConservationEntry] = {}
    for f in sol.fields:
        b = sol.budget[f]
        tot = b["total"]
        net_flux_src = b["inflow_left"] - b["outflow_right"] + b["source"]
        res = (tot - tot[0]) - net_flux_src
        scale = max(
            float(np.max(np.abs(tot))),
            float(np.max(np.abs(b["inflow_left"]))),
            float(np.max(np.abs(b["outflow_right"]))),
            float(np.max(np.abs(b["source"]))),
            1.0,
        )
        max_abs = float(np.max(np.abs(res)))
        max_rel = max_abs / scale
        entries[f] = FieldConservationEntry(
            name=f,
            symbolic_exact=True,
            initial_value=float(tot[0]),
            final_value=float(tot[-1]),
            max_abs_residual=max_abs,
            max_rel_residual=max_rel,
            residual_series=res,
            passed=bool(max_rel <= tol),
        )
    return ConservationAuditReport(
        kind="pde_fvm",
        t=sol.t,
        entries=entries,
        tol=tol,
        passed=all(e.passed for e in entries.values()),
    )


def audit_ode_invariants(
    system: ODESystem,
    traj: Trajectory,
    invariants: Mapping[str, ExprLike],
    params: Optional[Mapping[str, Any]] = None,
    tol: float = 1e-6,
    require_symbolic: bool = True,
) -> ConservationAuditReport:
    """
    Audit any conserved quantities I_j(t, y) (mass, energy, momentum, charge, population)
    both symbolically at the equation level (dI_j/dt == 0) and numerically along `traj`.
    """
    entries: Dict[str, FieldConservationEntry] = {}
    for name, expr in invariants.items():
        sym_ok, _ = system.check_invariant(expr)
        vals = system.evaluate(expr, traj, params=params)
        res = vals - vals[0]
        scale = max(float(np.max(np.abs(vals))), 1.0)
        max_abs = float(np.max(np.abs(res)))
        max_rel = max_abs / scale
        ok = (max_rel <= tol) and (sym_ok or not require_symbolic)
        entries[name] = FieldConservationEntry(
            name=name,
            symbolic_exact=sym_ok,
            initial_value=float(vals[0]),
            final_value=float(vals[-1]),
            max_abs_residual=max_abs,
            max_rel_residual=max_rel,
            residual_series=res,
            passed=bool(ok),
        )
    return ConservationAuditReport(
        kind="ode_invariants",
        t=traj.t,
        entries=entries,
        tol=tol,
        passed=all(e.passed for e in entries.values()),
    )


def audit_hybrid_conservation(
    sol: HybridODESolution,
    rate_fn: Callable[[float, float], float],
    tol: float = 1e-5,
) -> ConservationAuditReport:
    """
    Audit global conservation across piecewise-smooth ODE segments + discontinuous jump events:
        retained(t) + cumulative_removed_at_jumps(t) - [y(0) + integral_0^t rate(tau, y(tau)) dtau] == 0.
    """
    t = sol.wrt_values
    y = sol.target_values
    cum_int = np.zeros_like(t)
    for i in range(1, len(t)):
        dt_i = t[i] - t[i - 1]
        if dt_i > 0.0:
            r0 = float(rate_fn(float(t[i - 1]), float(y[i - 1])))
            r1 = float(rate_fn(float(t[i]), float(y[i])))
            cum_int[i] = cum_int[i - 1] + 0.5 * dt_i * (r0 + r1)
        else:
            cum_int[i] = cum_int[i - 1]

    # Cumulative removed mass/quantity steps up cleanly at the post-event duplicate timestamp
    cum_removed = np.zeros_like(t)
    ev_idx = 0
    evs = sol.events
    running_shed = 0.0
    for i in range(len(t)):
        if (
            ev_idx < len(evs)
            and i > 0
            and abs(t[i] - t[i - 1]) < 1e-12
            and abs((y[i - 1] - y[i]) - evs[ev_idx].shed_amount) < 1e-9
        ):
            running_shed += evs[ev_idx].shed_amount
            ev_idx += 1
        cum_removed[i] = running_shed

    res = (y + cum_removed) - (y[0] + cum_int)
    scale = max(float(np.max(np.abs(y[0] + cum_int))), 1.0)
    max_abs = float(np.max(np.abs(res)))
    max_rel = max_abs / scale
    entry = FieldConservationEntry(
        name="hybrid_total_budget",
        symbolic_exact=True,
        initial_value=float(y[0]),
        final_value=float(y[-1] + running_shed),
        max_abs_residual=max_abs,
        max_rel_residual=max_rel,
        residual_series=res,
        passed=bool(max_rel <= tol),
    )
    return ConservationAuditReport(
        kind="hybrid_events",
        t=t,
        entries={"hybrid_total_budget": entry},
        tol=tol,
        passed=entry.passed,
    )


# ============================================================================
# 4. Stiffness & Partitioned Multi-Physics Coupling Stability
# ============================================================================


@dataclass
class StiffnessReport:
    eigenvalues: np.ndarray
    spectral_radius: float
    stiffness_ratio: float
    max_real_part: float
    is_stiff: bool
    has_growing_mode: bool
    max_stable_dts: Dict[str, float]
    recommended_method: str

    def summary(self) -> str:
        dt_str = ", ".join(f"{k}:{v:.3e}" for k, v in self.max_stable_dts.items())
        return (
            "=== Jacobian Stiffness & Explicit Step-Limit Report ===\n"
            f"  Spectral radius = {self.spectral_radius:.4e} | Stiffness ratio = {self.stiffness_ratio:.4e}\n"
            f"  Max Re(lambda) = {self.max_real_part:.4e} | Stiff: {self.is_stiff} | Growing mode: {self.has_growing_mode}\n"
            f"  Explicit dt_max: {dt_str}\n"
            f"  Recommended integrator: {self.recommended_method}"
        )


def analyze_stiffness(
    system_or_rhs: Union[ODESystem, Callable[[float, np.ndarray], np.ndarray]],
    state: Sequence[float],
    params: Optional[Mapping[str, Any]] = None,
    t: float = 0.0,
    stiff_threshold: float = 100.0,
    schemes: Sequence[str] = ("euler", "ssprk2", "ssprk3", "rk4", "dopri5"),
) -> StiffnessReport:
    """
    Compute Jacobian spectrum, stiffness ratio, and exact maximum stable step sizes dt_max
    across explicit Runge-Kutta schemes.
    """
    if isinstance(system_or_rhs, ODESystem):
        eigs = system_or_rhs.eigenvalues(state, params=params, t=t)
        zero_tol = 1e-11
    else:
        y_arr = np.asarray(state, dtype=float).reshape(-1)
        J = fd_jacobian(system_or_rhs, t, y_arr)
        eigs = np.linalg.eigvals(J)
        zero_tol = 1e-6

    eigs = np.asarray(eigs, dtype=complex)
    mags = np.abs(eigs)
    spec_rad = float(np.max(mags)) if mags.size else 0.0
    reals = np.abs(eigs.real)
    nz_reals = reals[reals > zero_tol * max(spec_rad, 1.0)]
    stiff_ratio = float(np.max(nz_reals) / np.min(nz_reals)) if nz_reals.size >= 2 else 1.0
    max_re = float(np.max(eigs.real)) if eigs.size else 0.0
    has_grow = bool(max_re > zero_tol * max(spec_rad, 1.0))
    is_stiff = bool(stiff_ratio >= stiff_threshold)

    dt_limits = {sc: max_stable_dt(sc, eigs) for sc in schemes}
    rec = "Radau / BDF (implicit)" if is_stiff else ("cn / rk4" if dt_limits.get("euler", 1.0) == 0.0 else "RK45 / ssprk3")
    return StiffnessReport(
        eigenvalues=eigs,
        spectral_radius=spec_rad,
        stiffness_ratio=stiff_ratio,
        max_real_part=max_re,
        is_stiff=is_stiff,
        has_growing_mode=has_grow,
        max_stable_dts=dt_limits,
        recommended_method=rec,
    )


@dataclass
class PartitionedCouplingReport:
    converged: bool
    iterations: int
    solution: np.ndarray
    residual_history: List[float]
    contraction_factors: List[float]
    omega_history: List[float]
    asymptotic_contraction: float
    acceleration: str

    def summary(self) -> str:
        return (
            f"=== Partitioned Coupling Report [{self.acceleration}] ===\n"
            f"  Converged: {self.converged} in {self.iterations} iter(s) | "
            f"Final ||r||_inf = {self.residual_history[-1]:.3e} | "
            f"Asymptotic contraction rho = {self.asymptotic_contraction:.4f}"
        )


def solve_partitioned_fixed_point(
    step_map: Callable[[np.ndarray], np.ndarray],
    u0: Sequence[float],
    tol: float = 1e-10,
    max_iter: int = 100,
    omega: float = 1.0,
    acceleration: str = "aitken",  # "none", "constant", "aitken"
    omega_bounds: Tuple[float, float] = (0.02, 1.5),
) -> PartitionedCouplingReport:
    """
    Solve partitioned two-way interface fixed point u = G(u) (where G = G_A o G_B) with
    contraction-rate monitoring and optional Aitken Delta^2 dynamic relaxation.

    When unrelaxed Picard iteration (omega=1) diverges (spectral radius > 1, e.g. strong added-mass
    or stiff thermal-conductance feedback), `acceleration='aitken'` adapts omega_k automatically
    from two consecutive residuals to restore contraction.
    """
    acc = acceleration.lower()
    if acc not in ("none", "constant", "aitken"):
        raise ValueError("acceleration must be 'none', 'constant', or 'aitken'.")

    u = np.asarray(u0, dtype=float).reshape(-1).copy()
    w = 1.0 if acc == "none" else float(omega)
    res_hist: List[float] = []
    rho_hist: List[float] = []
    w_hist: List[float] = []
    r_prev: Optional[np.ndarray] = None
    converged = False

    for k in range(1, max_iter + 1):
        u_tilde = np.asarray(step_map(u), dtype=float).reshape(-1)
        r = u_tilde - u
        r_norm = float(np.linalg.norm(r, np.inf))
        if res_hist and res_hist[-1] > 0:
            rho_hist.append(r_norm / res_hist[-1])
        res_hist.append(r_norm)

        if r_norm <= tol * (1.0 + float(np.linalg.norm(u_tilde, np.inf))):
            u = u_tilde
            w_hist.append(w)
            converged = True
            break

        if acc == "aitken" and r_prev is not None:
            dr = r - r_prev
            denom = float(np.dot(dr, dr))
            if denom > 1e-30:
                w_raw = -w * float(np.dot(r_prev, dr)) / denom
                w = float(np.clip(w_raw, omega_bounds[0], omega_bounds[1]))

        w_hist.append(w)
        r_prev = r.copy()
        u = u + w * r

        if r_norm > 1e14 or not np.all(np.isfinite(u)):
            break

    asym_rho = float(np.median(rho_hist[-3:])) if len(rho_hist) >= 3 else (rho_hist[-1] if rho_hist else 0.0)
    return PartitionedCouplingReport(
        converged=converged,
        iterations=len(res_hist),
        solution=u,
        residual_history=res_hist,
        contraction_factors=rho_hist,
        omega_history=w_hist,
        asymptotic_contraction=asym_rho,
        acceleration=acc,
    )


@dataclass
class CoSimulationReport:
    t: np.ndarray
    states: List[str]
    y_cosim: np.ndarray
    y_monolithic: np.ndarray
    max_error_vs_monolithic: float
    total_subiterations: int
    max_subiterations_per_step: int
    mode: str
    dt_macro: float
    converged: bool

    def summary(self) -> str:
        return (
            f"=== Coupled Co-Simulation Report [{self.mode}, dt_macro={self.dt_macro:.4g}] ===\n"
            f"  Converged: {self.converged} | Max error vs. monolithic = {self.max_error_vs_monolithic:.3e}\n"
            f"  Total sub-iters = {self.total_subiterations} (max/step = {self.max_subiterations_per_step})"
        )


def cosimulate_coupled_odes(
    sys_a: ODESystem,
    sys_b: ODESystem,
    t_span: Tuple[float, float],
    y0: Mapping[str, float],
    dt_macro: float,
    params: Optional[Mapping[str, float]] = None,
    mode: str = "strong",  # "loose" (1-pass Gauss-Seidel) or "strong" (Picard + Aitken)
    sub_tol: float = 1e-9,
    max_subiter: int = 40,
    sub_method: str = "RK45",
) -> CoSimulationReport:
    """
    Co-simulate two coupled `ODESystem` modules that exchange states across macro-steps `dt_macro`,
    and quantify coupling splitting error directly against the exact monolithic combined `ODESystem`.
    """
    if mode not in ("loose", "strong"):
        raise ValueError("mode must be 'loose' or 'strong'.")
    base_params = {k: float(v) for k, v in (params or {}).items()}

    # Build monolithic reference system
    mono_rhs = {
        **dict(zip(sys_a.states, sys_a.rhs)),
        **dict(zip(sys_b.states, sys_b.rhs)),
    }
    mono = ODESystem(mono_rhs, indep=sys_a.indep)

    t0, t1 = float(t_span[0]), float(t_span[1])
    n_macro = max(1, int(np.ceil((t1 - t0) / dt_macro - 1e-9)))
    t_grid = np.linspace(t0, t1, n_macro + 1)
    dt_used = float(t_grid[1] - t_grid[0])

    mono_traj = mono.solve((t0, t1), y0, params=base_params, method="DOP853", rtol=1e-12, atol=1e-14, t_eval=t_grid)

    ya = np.array([float(y0[s]) for s in sys_a.states])
    yb = np.array([float(y0[s]) for s in sys_b.states])
    n_a = len(sys_a.states)
    n_b = len(sys_b.states)
    hist = np.zeros((n_macro + 1, n_a + n_b))
    hist[0] = np.concatenate([ya, yb])

    total_iters = 0
    max_step_iters = 0
    all_ok = True

    for step_idx in range(n_macro):
        ta, tb = float(t_grid[step_idx]), float(t_grid[step_idx + 1])

        def interp_param(v_start: float, v_end: float) -> Callable[[float], float]:
            span_dt = tb - ta
            return lambda t_q, vs=v_start, ve=v_end, t_s=ta, dt_s=span_dt: vs + ((t_q - t_s) / dt_s) * (ve - vs)

        if mode == "loose":
            # 1-pass staggered Gauss-Seidel: A holds B frozen at ta; B uses A's linear ramp [ya(ta) -> ya(tb)]
            pa = {**base_params, **{s: float(yb[j]) for j, s in enumerate(sys_b.states) if s in sys_a.params}}
            ya_next = sys_a.step(ta, tb, ya, params=pa, method=sub_method)
            pb = {
                **base_params,
                **{
                    s: interp_param(float(ya[i]), float(ya_next[i]))
                    for i, s in enumerate(sys_a.states)
                    if s in sys_b.params
                },
            }
            yb_next = sys_b.step(ta, tb, yb, params=pb, method=sub_method)
            ya, yb = ya_next, yb_next
            total_iters += 1
            max_step_iters = max(max_step_iters, 1)
        else:
            # Tight sub-iterated Gauss-Seidel map on end-of-step interface state yb_end
            def interface_map(yb_guess: np.ndarray) -> np.ndarray:
                pa_s = {
                    **base_params,
                    **{
                        s: interp_param(float(yb[j]), float(yb_guess[j]))
                        for j, s in enumerate(sys_b.states)
                        if s in sys_a.params
                    },
                }
                ya_end = sys_a.step(ta, tb, ya, params=pa_s, method=sub_method)
                pb_s = {
                    **base_params,
                    **{
                        s: interp_param(float(ya[i]), float(ya_end[i]))
                        for i, s in enumerate(sys_a.states)
                        if s in sys_b.params
                    },
                }
                return sys_b.step(ta, tb, yb, params=pb_s, method=sub_method)

            fp = solve_partitioned_fixed_point(
                interface_map, yb, tol=sub_tol, max_iter=max_subiter, acceleration="aitken"
            )
            all_ok = all_ok and fp.converged
            yb_next = fp.solution
            pa_final = {
                **base_params,
                **{
                    s: interp_param(float(yb[j]), float(yb_next[j]))
                    for j, s in enumerate(sys_b.states)
                    if s in sys_a.params
                },
            }
            ya_next = sys_a.step(ta, tb, ya, params=pa_final, method=sub_method)
            ya, yb = ya_next, yb_next
            total_iters += fp.iterations
            max_step_iters = max(max_step_iters, fp.iterations)

        hist[step_idx + 1] = np.concatenate([ya, yb])

    err = float(np.max(np.abs(hist - mono_traj.y)))
    return CoSimulationReport(
        t=t_grid,
        states=mono.states,
        y_cosim=hist,
        y_monolithic=mono_traj.y,
        max_error_vs_monolithic=err,
        total_subiterations=total_iters,
        max_subiterations_per_step=max_step_iters,
        mode=mode,
        dt_macro=dt_used,
        converged=all_ok,
    )
