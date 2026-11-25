"""
Automatic simulation and plotting of formulated equations.

Given a final equation (DerivationResult, CandidateEquation, FitResult, or sympy Eq/Expr),
sweep one variable, hold the others at supplied values, optionally draw a family of curves
over a second variable, and render publication-ready figures.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import sympy as sp

from .numerical import FitResult, HybridODESolution, NumericalSolver
from .symbolic import CandidateEquation, DerivationResult

Model = Union[DerivationResult, CandidateEquation, FitResult, sp.Eq, sp.Expr]


@dataclass
class SimulationResult:
    """Evaluated curves of an equation over a swept variable."""

    y_name: str
    x_name: str
    x: np.ndarray
    curves: Dict[str, np.ndarray]
    expression: sp.Expr
    assumed: Dict[str, float] = field(default_factory=dict)
    candidate_name: str = ""

    def to_csv(self, path: Union[str, Path]) -> Path:
        out = Path(path)
        labels = list(self.curves)
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow([self.x_name] + labels)
            for i, xv in enumerate(self.x):
                w.writerow([xv] + [self.curves[k][i] for k in labels])
        return out


def _resolve(model: Model, candidate: Optional[str]) -> Tuple[str, str, sp.Expr, List[sp.Symbol]]:
    if isinstance(model, DerivationResult):
        cands = {c.name: c for c in model.candidates}
        if candidate is not None and candidate not in cands:
            raise KeyError(f"Unknown candidate '{candidate}'. Available: {list(cands)}")
        c = cands[candidate] if candidate else model.primary
        return c.name, str(c.equation.lhs), c.rhs, list(c.free_params)
    if isinstance(model, CandidateEquation):
        return model.name, str(model.equation.lhs), model.rhs, list(model.free_params)
    if isinstance(model, FitResult):
        eq = model.calibrated_equation
        return model.candidate_name, str(eq.lhs), eq.rhs, []
    if isinstance(model, sp.Eq):
        return "equation", str(model.lhs), model.rhs, []
    if isinstance(model, sp.Expr):
        return "expression", "y", model, []
    raise TypeError(f"Unsupported model type: {type(model)}")


def simulate(
    model: Model,
    sweep: Mapping[str, Tuple[float, float, int]],
    values: Optional[Mapping[str, float]] = None,
    family: Optional[Mapping[str, Sequence[float]]] = None,
    candidate: Optional[str] = None,
) -> SimulationResult:
    """
    Evaluate an equation over one swept variable.

    Args:
        model: Final equation (e.g. result of `formulate`, or a `FitResult`).
        sweep: {"f": (start, stop, n_points)} - exactly one variable.
        values: Fixed values for every other symbol (constants, parameters, kappas).
            Unspecified free parameters default to 1.0 (offsets `*_0` default to 0.0)
            and are reported in `SimulationResult.assumed`.
        family: Optional {"a": [1, 2, 3]} to draw one curve per value (cartesian product
            when several variables are given).
        candidate: Candidate name when `model` is a DerivationResult with several.
    """
    if len(sweep) != 1:
        raise ValueError("`sweep` must contain exactly one variable, e.g. {'f': (0, 10, 200)}.")
    (x_name, (lo, hi, n)), = sweep.items()

    name, y_name, rhs, free = _resolve(model, candidate)
    symbols = {str(s) for s in rhs.free_symbols}
    if x_name not in symbols:
        raise KeyError(f"Swept variable '{x_name}' does not appear in the equation (symbols: {sorted(symbols)}).")

    fixed: Dict[str, float] = dict(values or {})
    fam = {k: list(v) for k, v in (family or {}).items()}
    assumed: Dict[str, float] = {}
    for p in free:
        pn = str(p)
        if pn in symbols and pn not in fixed and pn not in fam:
            assumed[pn] = 0.0 if pn.endswith("_0") else 1.0
            fixed[pn] = assumed[pn]

    missing = sorted(symbols - {x_name} - set(fixed) - set(fam))
    if missing:
        raise KeyError(f"Provide values for symbols {missing} via `values=` (or `family=`).")

    x = np.linspace(float(lo), float(hi), int(n))
    curves: Dict[str, np.ndarray] = {}
    fam_names = list(fam)
    combos = list(product(*[fam[k] for k in fam_names])) if fam_names else [()]
    for combo in combos:
        data = {x_name: x, **fixed, **dict(zip(fam_names, combo))}
        label = ", ".join(f"{k}={v:g}" for k, v in zip(fam_names, combo)) or y_name
        curves[label] = NumericalSolver.evaluate(rhs, data)

    return SimulationResult(
        y_name=y_name, x_name=x_name, x=x, curves=curves,
        expression=rhs, assumed=assumed, candidate_name=name,
    )


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def _finish(fig, path: Optional[Union[str, Path]], show: bool) -> Optional[Path]:
    plt = _pyplot()
    fig.tight_layout()
    out = None
    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=160)
    if show:
        plt.show()
    plt.close(fig)
    return out


def plot_simulation(
    sim: SimulationResult,
    path: Optional[Union[str, Path]] = None,
    data: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    logx: bool = False,
    logy: bool = False,
    title: Optional[str] = None,
    show: bool = False,
) -> Optional[Path]:
    """Plot simulated curve(s); `data=(x, y)` overlays observations as markers."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for label, y in sim.curves.items():
        ax.plot(sim.x, y, lw=2, label=label)
    if data is not None:
        ax.scatter(data[0], data[1], s=22, color="black", alpha=0.7, zorder=3, label="data")
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(sim.x_name)
    ax.set_ylabel(sim.y_name)
    ax.set_title(title or f"{sim.y_name}({sim.x_name})  [{sim.candidate_name}]")
    ax.grid(alpha=0.3)
    if len(sim.curves) > 1 or data is not None:
        ax.legend(frameon=False)
    if sim.assumed:
        note = "assumed: " + ", ".join(f"{k}={v:g}" for k, v in sim.assumed.items())
        ax.annotate(note, xy=(0.01, 0.01), xycoords="axes fraction", fontsize=7, color="gray")
    return _finish(fig, path, show)


def plot_ranked(
    ranked: Sequence[FitResult],
    data: Mapping[str, Sequence[float]],
    x_name: str,
    y_name: str,
    path: Optional[Union[str, Path]] = None,
    logx: bool = False,
    logy: bool = False,
    show: bool = False,
) -> Optional[Path]:
    """Compare fitted candidates against observations (predictions vs `x_name`)."""
    plt = _pyplot()
    x = np.asarray(data[x_name], dtype=float)
    y = np.asarray(data[y_name], dtype=float)
    order = np.argsort(x)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.scatter(x, y, s=22, color="black", alpha=0.7, zorder=3, label="data")
    for i, fr in enumerate(ranked):
        ax.plot(x[order], np.asarray(fr.predictions)[order], lw=2.2 if i == 0 else 1.2,
                label=f"#{i + 1} {fr.candidate_name} (R²={fr.r2:.4f}, BIC={fr.bic:.0f})")
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title("Candidate equations vs. data (ranked by BIC)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    return _finish(fig, path, show)


def plot_hybrid(
    sol: HybridODESolution,
    path: Optional[Union[str, Path]] = None,
    x_label: str = "t",
    y_label: str = "state",
    y_scale: float = 1.0,
    y_unit: str = "",
    show: bool = False,
) -> Optional[Path]:
    """Plot an accretion/shedding sawtooth and mark every detachment event by mode."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(sol.wrt_values, np.asarray(sol.target_values) * y_scale, color="tab:blue", lw=2)
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    modes = sorted({e.mode for e in sol.events})
    for i, mode in enumerate(modes):
        evs = [e for e in sol.events if e.mode == mode]
        ax.scatter([e.time for e in evs], [e.pre_value * y_scale for e in evs],
                   s=55, color=palette[(i + 1) % len(palette)], zorder=4,
                   label=f"{mode} (n={len(evs)})")
        for e in evs:
            ax.annotate(f"{e.time:.1f}", (e.time, e.pre_value * y_scale), textcoords="offset points",
                        xytext=(3, 6), fontsize=7)
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"{y_label} ({y_unit})" if y_unit else y_label)
    ax.set_title(f"Accretion with threshold shedding - {sol.n_events} event(s)")
    ax.grid(alpha=0.3)
    if modes:
        ax.legend(frameon=False)
    return _finish(fig, path, show)


def simulate_and_plot(
    model: Model,
    sweep: Mapping[str, Tuple[float, float, int]],
    path: Union[str, Path],
    values: Optional[Mapping[str, float]] = None,
    family: Optional[Mapping[str, Sequence[float]]] = None,
    candidate: Optional[str] = None,
    data: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    logx: bool = False,
    logy: bool = False,
) -> SimulationResult:
    """One call: evaluate the equation over a sweep and save the figure."""
    sim = simulate(model, sweep, values=values, family=family, candidate=candidate)
    plot_simulation(sim, path=path, data=data, logx=logx, logy=logy)
    return sim


def plot_mms_convergence(
    report,
    path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> Optional[Path]:
    """Plot log-log spatial L2 & Linf error convergence from `mms_study` vs. formal slope."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    h = report.spacings
    for f in report.fields:
        ax.loglog(h, report.errors_l2[f], "o-", lw=2, label=f"{f} L2 (p={report.order_l2[f]:.2f})")
        ax.loglog(h, report.errors_linf[f], "s--", lw=1.3, alpha=0.8, label=f"{f} L∞ (p={report.order_linf[f]:.2f})")
    ref = report.errors_l2[report.fields[0]][0] * (h / h[0]) ** report.formal_order
    ax.loglog(h, ref, "k:", lw=1.5, label=f"Reference O(h^{report.formal_order:g})")
    ax.set_xlabel("Grid spacing h")
    ax.set_ylabel("Discretisation error")
    ax.set_title(f"MMS Spatial Convergence (formal order = {report.formal_order:g})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    return _finish(fig, path, show)


def plot_gci_study(
    report,
    path: Optional[Union[str, Path]] = None,
    qoi_label: str = "Quantity of Interest",
    show: bool = False,
) -> Optional[Path]:
    """Plot coarse/medium/fine QoI values, Richardson extrapolation, and +-GCI_fine band."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    hs = np.array(report.spacings, dtype=float)
    phis = np.array(report.values, dtype=float)
    ax.plot(hs, phis, "o-", lw=2, color="tab:blue", label="Mesh / step evaluations")
    ax.axhline(report.extrapolated_value, color="tab:green", ls="--", lw=1.6,
               label=f"Richardson φ_ext = {report.extrapolated_value:.6g} (p={report.observed_order:.2f})")
    band = abs(report.values[0]) * report.gci_fine_21
    ax.axhspan(report.values[0] - band, report.values[0] + band, color="tab:green", alpha=0.15,
               label=f"±GCI_fine ({report.gci_fine_21 * 100:.3f}%)")
    ax.set_xlabel("Discretisation size h (or dt)")
    ax.set_ylabel(qoi_label)
    ax.set_title(f"Grid / Step Independence [{report.regime}]")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    return _finish(fig, path, show)


def plot_conservation_audit(
    audit_report,
    path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> Optional[Path]:
    """Plot discrete conservation residual histories R(t) across audited quantities."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for nm, entry in audit_report.entries.items():
        ax.plot(audit_report.t, entry.residual_series, lw=1.8,
                label=f"{nm} (max|R|={entry.max_abs_residual:.2e})")
    ax.axhline(0.0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("t")
    ax.set_ylabel("Conservation residual R(t)")
    ax.set_title(f"Conservation Audit [{audit_report.kind}] — Passed: {audit_report.passed}")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    return _finish(fig, path, show)


def plot_stability_spectrum(
    stiffness_report,
    dt: float,
    scheme: str = "rk4",
    path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> Optional[Path]:
    """Plot scaled eigenvalues z = dt*lambda against the stability region |R(z)| <= 1 of `scheme`."""
    from .integrators import stability_function

    plt = _pyplot()
    z_pts = np.asarray(stiffness_report.eigenvalues, dtype=complex) * float(dt)
    span_re = max(4.5, float(np.max(np.abs(z_pts.real)) * 1.25) if z_pts.size else 4.5)
    span_im = max(4.0, float(np.max(np.abs(z_pts.imag)) * 1.25) if z_pts.size else 4.0)
    xr = np.linspace(-span_re, max(1.5, 0.25 * span_re), 260)
    yr = np.linspace(-span_im, span_im, 260)
    X, Y = np.meshgrid(xr, yr)
    Z = X + 1j * Y
    mag = np.abs(stability_function(scheme, Z))

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.contourf(X, Y, mag, levels=[0.0, 1.0], colors=["#d9f0e1"], alpha=0.7)
    ax.contour(X, Y, mag, levels=[1.0], colors=["#1e5631"], linewidths=1.6)
    ax.axhline(0.0, color="gray", lw=0.7)
    ax.axvline(0.0, color="gray", lw=0.7)
    ax.scatter(z_pts.real, z_pts.imag, color="crimson", s=38, zorder=4, label=f"dt·λ (dt={dt:.3g})")
    ax.set_xlabel("Re(dt · λ)")
    ax.set_ylabel("Im(dt · λ)")
    ax.set_title(f"Linear Stability Region [{scheme}] — |R(z)| ≤ 1 shaded")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    return _finish(fig, path, show)
