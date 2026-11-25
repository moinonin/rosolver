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
