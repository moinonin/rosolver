"""
End-to-End Ice Growth Study: From Nucleation to Macroscopic Solid Mass (Eq. 16).

Explains, dimensionally closes, simulates, verifies, and visualizes the coupled system:
    dN/dt = beta
    N * dm/dt = beta * (m_max - m),   m = rho * sigma * z
    d2z/dt2 = -(1/(sigma*rho)) * |m_ddot| = -(1/rho) * dp/dz

Usage:
    python -m examples.ice_nucleation_growth_study.run_study [--output outputs/ice_nucleation_growth_study]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from rosolver import (
    Dimension,
    EventSpec,
    Model,
    ODESystem,
    Project,
    Stefan1D,
    ValidationStudy,
    VerificationStatus,
    analyze_stiffness,
    audit_ode_invariants,
    standardize_result,
    timestep_study,
)

MODEL_PATH = Path(__file__).resolve().parent / "model.yaml"


def exact_ice_growth_solution(
    t: Union[float, np.ndarray],
    *,
    N0: float,
    beta: float,
    z0: float,
    z_max: float,
    rho: float,
    sigma: float,
) -> Dict[str, np.ndarray]:
    """
    Closed-form exact analytical solution for constant nucleation rate beta:
        N(t)      = N0 + beta * t
        z(t)      = z_max - (z_max - z0) * N0 / (N0 + beta * t)
        vz(t)     = (z_max - z0) * N0 * beta / (N0 + beta * t)^2
        az(t)     = -2 * (z_max - z0) * N0 * beta^2 / (N0 + beta * t)^3
        dp_dz(t)  = -rho * az(t) > 0
        m(t)      = rho * sigma * z(t)
    """
    t_arr = np.asarray(t, dtype=float)
    N_t = N0 + beta * t_arr
    dz_span = z_max - z0
    z_t = z_max - dz_span * (N0 / N_t)
    vz_t = dz_span * N0 * beta / (N_t ** 2)
    az_t = -2.0 * dz_span * N0 * (beta ** 2) / (N_t ** 3)
    dp_dz_t = -rho * az_t
    m_t = rho * sigma * z_t
    p_imp_t = rho * (vz_t[0] - vz_t) if t_arr.ndim > 0 and t_arr.size > 0 else rho * (dz_span * beta / N0 - vz_t)
    return {
        "N": N_t,
        "m": m_t,
        "z": z_t,
        "vz": vz_t,
        "az": az_t,
        "dp_dz": dp_dz_t,
        "P_imp": p_imp_t,
    }


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.dpi": 220,
        }
    )


def run_ice_nucleation_study(output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Execute full dimensional check, ODE/Stefan simulation, V&V, and visual reporting."""
    _configure_plot_style()
    study_dir = Path(output_dir) if output_dir is not None else Path("outputs/ice_nucleation_growth_study")
    repo_figures_dir = Path("figures")
    repo_figures_dir.mkdir(parents=True, exist_ok=True)
    timings: Dict[str, float] = {}
    t_total_start = time.perf_counter()

    # 1. Validate declarative Model & initialize clean SHA-256 Project workspace
    t0 = time.perf_counter()
    if study_dir.exists():
        shutil.rmtree(study_dir)
    model = Model.from_file(MODEL_PATH)
    model.validate(strict=True)
    project = Project.create(
        study_dir,
        name="Ice Growth: Nucleation to Macroscopic Solid Mass Study",
        description=(
            "Dimensional closure, exact analytical benchmarking, coupled ODE + Stefan1D "
            "continuum comparison, conservation audits, and RK4 GCI verification."
        ),
        overwrite=True,
    )
    project.add_model(model, filename="ice_nucleation_to_solid_growth.yaml")
    base_params = model.parameter_values()
    y0_base = {c.field: float(c.value) for c in model.initial_conditions}
    z_max = base_params["m_max"] / (base_params["rho"] * base_params["sigma"])
    timings["model_setup_s"] = time.perf_counter() - t0

    # 2. Dimensional Homogeneity Verification of Eq. (16) & Growth Closure
    t0 = time.perf_counter()
    dim_acc = Dimension.from_unit("m/s^2")
    dim_mddot_term = (Dimension.from_unit("kg/s^2")) / (Dimension.from_unit("m^2") * Dimension.from_unit("kg/m^3"))
    dim_euler_rhs = Dimension.from_unit("Pa/m") / Dimension.from_unit("kg/m^3")
    if not (dim_acc == dim_mddot_term == dim_euler_rhs):
        raise AssertionError(f"Eq. (16) dimensional mismatch: {dim_acc}, {dim_mddot_term}, {dim_euler_rhs}")
    dim_growth_lhs = Dimension.from_unit("1/m^3") * Dimension.from_unit("kg/s")
    dim_growth_rhs = Dimension.from_unit("1/(m^3*s)") * Dimension.from_unit("kg")
    if dim_growth_lhs != dim_growth_rhs:
        raise AssertionError(f"Growth ODE dimensional mismatch: {dim_growth_lhs} vs {dim_growth_rhs}")
    timings["dimensional_check_s"] = time.perf_counter() - t0

    # 3. Coupled Multi-State ODE Simulation + Coalescence Event Detection
    t0 = time.perf_counter()
    ode_sys = model.to_ode_system()
    t_end = 600.0  # 10 minutes (12 nucleation timescales tau_N = N0/beta = 50 s)
    t_span = (0.0, t_end)
    dt_nom = 0.5  # s

    traj_ref = ode_sys.solve(t_span=t_span, y0=y0_base, params=base_params, method="rk4", dt=dt_nom)
    coalescence_event = EventSpec(
        name="coalescence_rcp",
        expr=f"z - {0.64 * z_max:.8f}",
        terminal=False,
        direction=1.0,
    )
    traj_ev = ode_sys.solve(
        t_span=t_span,
        y0=y0_base,
        params=base_params,
        method="DOP853",
        rtol=1e-11,
        atol=1e-14,
        events=[coalescence_event],
    )
    t_coal_s = float(traj_ev.t_events["coalescence_rcp"][0])

    sim_sr = standardize_result(traj_ref, kind="ode_simulation", model_name=model.name, model_version=model.version)
    project.record_run(
        "ode_simulation",
        sim_sr,
        model=model,
        configuration={"t_end_s": t_end, "dt_s": dt_nom, "method": "rk4"},
    )

    # Multi-beta nucleation regime sweep
    beta_cases = [
        ("Slow Nucleation ($\\beta = 50\\,\\mathrm{m}^{-3}\\mathrm{s}^{-1}$, $\\tau_N=200$ s)", 50.0, "#1E5631"),
        ("Reference ($\\beta = 200\\,\\mathrm{m}^{-3}\\mathrm{s}^{-1}$, $\\tau_N=50$ s)", 200.0, "#1B365D"),
        ("Fast Nucleation ($\\beta = 600\\,\\mathrm{m}^{-3}\\mathrm{s}^{-1}$, $\\tau_N=16.7$ s)", 600.0, "#B7791F"),
        ("Rapid Quench ($\\beta = 1500\\,\\mathrm{m}^{-3}\\mathrm{s}^{-1}$, $\\tau_N=6.7$ s)", 1500.0, "#9B2C2C"),
    ]
    beta_runs: List[Dict[str, Any]] = []
    for label, b_val, col in beta_cases:
        p_cur = {**base_params, "beta": b_val}
        vz0_cur = (b_val / y0_base["N"]) * (z_max - y0_base["z"])
        y_cur = {**y0_base, "vz": vz0_cur}
        tr = ode_sys.solve(t_span=t_span, y0=y_cur, params=p_cur, method="rk4", dt=dt_nom)
        ex = exact_ice_growth_solution(
            tr.t, N0=y_cur["N"], beta=b_val, z0=y_cur["z"], z_max=z_max, rho=p_cur["rho"], sigma=p_cur["sigma"]
        )
        dp_dz_num = 2.0 * p_cur["rho"] * (b_val / tr["N"]) * tr["vz"]
        beta_runs.append(
            {
                "label": label,
                "beta": b_val,
                "tau_N_s": y_cur["N"] / b_val,
                "color": col,
                "t": tr.t,
                "N": tr["N"],
                "m": tr["m"],
                "m_liq": tr["m_liq"],
                "z": tr["z"],
                "vz": tr["vz"],
                "dp_dz": dp_dz_num,
                "P_imp": tr["P_imp"],
                "exact": ex,
                "max_z_err_m": float(np.max(np.abs(tr["z"] - ex["z"]))),
                "t_coal_exact_s": (y_cur["N"] / b_val) * ((z_max - y_cur["z"]) / ((1.0 - 0.64) * z_max) - 1.0),
                "t_90_exact_s": (y_cur["N"] / b_val) * ((z_max - y_cur["z"]) / (0.10 * z_max) - 1.0),
            }
        )

    # Continuum thermal freezing comparison via rosolver.Stefan1D (subcooled cold wall T_wall = -15 C)
    stefan = Stefan1D(T_wall=-15.0, T_f=0.0, h_c=0.0, n_xi=80)
    stefan_sol = stefan.solve(t_end, delta0=y0_base["z"])
    timings["simulation_s"] = time.perf_counter() - t0

    # 4. Verification & Validation Suite
    t0 = time.perf_counter()
    # 4A. Exact analytical comparison via ValidationStudy (25 evenly spaced checkpoints)
    exact_ref = exact_ice_growth_solution(
        traj_ref.t,
        N0=y0_base["N"],
        beta=base_params["beta"],
        z0=y0_base["z"],
        z_max=z_max,
        rho=base_params["rho"],
        sigma=base_params["sigma"],
    )
    idx_val = np.linspace(0, len(traj_ref.t) - 1, 25, dtype=int)
    max_z_abs_err = float(np.max(np.abs(traj_ref["z"] - exact_ref["z"])))
    max_vz_abs_err = float(np.max(np.abs(traj_ref["vz"] - exact_ref["vz"])))

    # 4B. RK4 Timestep Convergence (Roache/Celik GCI)
    dt_report = timestep_study(
        lambda dt_v: float(
            ode_sys.solve(t_span=t_span, y0=y0_base, params=base_params, method="rk4", dt=dt_v).final["z"]
        ),
        dts=(4.0, 2.0, 1.0),
        formal_order=4.0,
    )
    dt_sr = standardize_result(dt_report, kind="timestep", model_name=model.name, model_version=model.version)
    project.record_run(
        "timestep",
        dt_sr,
        model=model,
        configuration={"dts_s": [1.0, 2.0, 4.0], "scheme": "rk4", "qoi": "z_final_m"},
    )

    # 4C. Dual Symbolic & Discrete Invariant Conservation Audit (Mass + Eq. 16 Momentum-Impulse)
    inv_report = audit_ode_invariants(
        ode_sys,
        traj_ref,
        invariants={
            "mass_conservation": "m + m_liq",
            "momentum_impulse_eq16": "rho * vz + P_imp",
        },
        params=base_params,
        tol=1e-12,
    )
    inv_sr = standardize_result(inv_report, kind="conservation", model_name=model.name, model_version=model.version)
    project.record_run(
        "conservation",
        inv_sr,
        model=model,
        configuration={"invariants": ["m + m_liq", "rho * vz + P_imp"], "tol": 1e-12},
    )

    # 4D. Stiffness & Analytical Validation Study
    stiff_report = analyze_stiffness(ode_sys, traj_ref.y[0], params=base_params, t=0.0)
    u_num_m = max(abs(float(dt_report.gci.gci_fine_21) * z_max), 1e-9)
    val_report = ValidationStudy.compare(
        simulation=traj_ref["z"][idx_val],
        experiment=exact_ref["z"][idx_val],
        field="growth_length_z",
        unit="m",
        u_num=u_num_m,
        u_exp=1e-10,
        coordinates=traj_ref.t[idx_val],
        coord_name="time_s",
        coverage_factor=2.0,
        study_name="Analytical Benchmark Verification of Ice Front Length z(t)",
        operating_conditions={"beta": base_params["beta"], "N0": y0_base["N"], "rho": base_params["rho"]},
    )
    val_sr = standardize_result(val_report, kind="validation", model_name=model.name, model_version=model.version)
    project.record_run(
        "validation",
        val_sr,
        model=model,
        configuration={"reference": "closed_form_exact_solution", "n_points": len(idx_val)},
    )
    timings["verification_s"] = time.perf_counter() - t0

    # 5. Generate Publication-Quality Visuals + Scientific Report
    t0 = time.perf_counter()
    fig_paths_rel = _generate_ice_figures(
        project_root=project.root,
        repo_figures_dir=repo_figures_dir,
        beta_runs=beta_runs,
        stefan_sol=stefan_sol,
        z_max=z_max,
        base_params=base_params,
        y0_base=y0_base,
        traj_ref=traj_ref,
        exact_ref=exact_ref,
        dt_report=dt_report,
        inv_report=inv_report,
    )

    figures_meta = [
        {"path": fig_paths_rel[0], "caption": "Multi-scale ice growth timeline: nucleation N(t), mass conservation m(t)+m_liq(t), front length z(t) vs exact solution, and Stefan1D continuum crossover."},
        {"path": fig_paths_rel[1], "caption": "Eq. (16) kinematics and induced pressure gradient: velocity dz/dt (t^-2), deceleration -d2z/dt2 (t^-3), and crystallization pressure gradient dp/dz = -rho*d2z/dt2."},
        {"path": fig_paths_rel[2], "caption": "2D solidification regime phase map t_90(beta, N0) and 4-stage micro-to-macro ice morphology schematic inside cross-section sigma."},
        {"path": fig_paths_rel[3], "caption": "Rosolver V&V diagnostics: analytical error history, RK4 Roache GCI convergence, dual mass & Eq. (16) momentum-impulse conservation residuals, and parity."},
    ]
    report = project.report(
        output="reports/final_report.md",
        title="Ice Growth from Nucleation to Macroscopic Solid Mass — End-to-End Study",
        description=(
            "Dimensional audit, exact analytical derivation, coupled ODE & Stefan1D simulation, "
            "dual conservation verification, and RK4 GCI convergence study."
        ),
        figures=figures_meta,
        also_manifest=True,
    )
    timings["reporting_and_figures_s"] = time.perf_counter() - t0
    timings["total_s"] = time.perf_counter() - t_total_start

    summary_payload = {
        "project_root": study_dir.as_posix(),
        "report_path": (study_dir / "reports" / "final_report.md").as_posix(),
        "report_status": report.status.value,
        "z_max_mm": round(z_max * 1e3, 3),
        "tau_N_ref_s": round(y0_base["N"] / base_params["beta"], 3),
        "t_coalescence_ref_s": round(t_coal_s, 3),
        "z_final_ref_mm": round(float(traj_ref.final["z"]) * 1e3, 4),
        "stefan_final_mm": round(float(stefan_sol.delta[-1]) * 1e3, 4),
        "initial_dp_dz_Pa_per_m": round(float(exact_ref["dp_dz"][0]), 4),
        "max_z_abs_err_m": max_z_abs_err,
        "max_vz_abs_err_ms": max_vz_abs_err,
        "rk4_observed_order": round(float(dt_report.gci.observed_order), 4),
        "rk4_gci_fine_21": float(dt_report.gci.gci_fine_21),
        "mass_conservation_max_res": float(inv_report.entries["mass_conservation"].max_abs_residual),
        "momentum_impulse_max_res": float(inv_report.entries["momentum_impulse_eq16"].max_abs_residual),
        "stiffness_ratio": round(float(stiff_report.stiffness_ratio), 3),
        "validation_rmse_m": float(val_report.rmse),
        "beta_cases": [
            {
                "beta": r["beta"],
                "tau_N_s": round(r["tau_N_s"], 2),
                "t_coal_s": round(r["t_coal_exact_s"], 2),
                "t_90_s": round(r["t_90_exact_s"], 2),
                "z_final_mm": round(float(r["z"][-1]) * 1e3, 3),
                "max_z_err_m": r["max_z_err_m"],
            }
            for r in beta_runs
        ],
        "timings_s": {k: round(v, 4) for k, v in timings.items()},
    }
    (project.root / "reports" / "study_metrics.json").write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )
    return summary_payload


def _generate_ice_figures(
    *,
    project_root: Path,
    repo_figures_dir: Path,
    beta_runs: List[Dict[str, Any]],
    stefan_sol: Any,
    z_max: float,
    base_params: Dict[str, float],
    y0_base: Dict[str, float],
    traj_ref: Any,
    exact_ref: Dict[str, np.ndarray],
    dt_report: Any,
    inv_report: Any,
) -> List[str]:
    rel_paths = [
        "reports/figures/ice_fig1_multiscale_timeline.png",
        "reports/figures/ice_fig2_kinematics_pressure.png",
        "reports/figures/ice_fig3_regime_and_morphology.png",
        "reports/figures/ice_fig4_verification_suite.png",
    ]
    repo_names = [
        "ice_fig1_multiscale_timeline.png",
        "ice_fig2_kinematics_pressure.png",
        "ice_fig3_regime_and_morphology.png",
        "ice_fig4_verification_suite.png",
    ]

    # ------------------------------------------------------------------
    # Figure 1: 2x2 Multi-Scale Ice Growth Timeline
    # ------------------------------------------------------------------
    fig1, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), constrained_layout=True)
    for r in beta_runs:
        t = r["t"]
        c = r["color"]
        axes[0, 0].plot(t, r["N"] * 1e-5, color=c, lw=2.0, label=r["label"])
        axes[0, 1].plot(t, r["m"], color=c, lw=2.0)
        axes[1, 0].plot(t, r["z"] * 1e3, color=c, lw=2.2)
        axes[1, 0].plot(t[::40], r["exact"]["z"][::40] * 1e3, "o", color=c, ms=3.5, alpha=0.8)

    # Liquid depletion curve on (0,1) for reference case
    ref_r = beta_runs[1]
    axes[0, 1].plot(ref_r["t"], ref_r["m_liq"], color="#1B365D", ls="--", lw=1.6, label="Ref. supercooled liquid $m_{\\mathrm{liq}}(t)$")
    axes[0, 1].axhline(base_params["m_max"], color="#333333", ls=":", lw=1.2, label="Total mass cap $m_{\\max} = 0.4585$ kg")
    axes[0, 1].legend(loc="center right", fontsize=8.0)

    axes[1, 0].axhline(0.64 * z_max * 1e3, color="#4A5568", ls="--", lw=1.3, label="Coalescence threshold $\\phi_c = 0.64$ ($32$ mm)")
    axes[1, 0].axhline(z_max * 1e3, color="#222222", ls=":", lw=1.1, label="Slab cap $z_{\\max} = 50$ mm")
    axes[1, 0].legend(loc="lower right", fontsize=8.0)

    # Panel (1,1): Nucleation-growth front vs Stefan1D thermal-conduction front
    axes[1, 1].plot(ref_r["t"], ref_r["z"] * 1e3, color="#1B365D", lw=2.2, label="Nucleation-growth ODE ($\\beta=200$)")
    axes[1, 1].plot(beta_runs[0]["t"], beta_runs[0]["z"] * 1e3, color="#1E5631", ls="-.", lw=1.8, label="Slow nucleation ($\\beta=50$)")
    axes[1, 1].plot(stefan_sol.t, stefan_sol.delta * 1e3, color="#C53030", ls="--", lw=2.2, label="Continuum Stefan1D ($T_{\\mathrm{wall}}=-15^\\circ$C, $\\propto \\sqrt{t}$)")
    axes[1, 1].axvline(ref_r["t_coal_exact_s"], color="#1B365D", ls=":", lw=1.3, label=f"Ref. coalescence $t_c = {ref_r['t_coal_exact_s']:.1f}$ s")
    axes[1, 1].legend(loc="lower right", fontsize=8.0)

    axes[0, 0].set_title("(a) Ice Nuclei Concentration $N(t) = N_0 + \\beta t$")
    axes[0, 0].set_xlabel("Time $t$ [s]")
    axes[0, 0].set_ylabel("Concentration $N$ [$\\times 10^5\\,\\mathrm{m}^{-3}$]")

    axes[0, 1].set_title("(b) Solid Mass $m(t)$ & Liquid Depletion $m_{\\mathrm{liq}}(t)$")
    axes[0, 1].set_xlabel("Time $t$ [s]")
    axes[0, 1].set_ylabel("Mass [kg]")

    axes[1, 0].set_title("(c) Growth Length $z(t)$ (Lines: ODE, Dots: Exact)")
    axes[1, 0].set_xlabel("Time $t$ [s]")
    axes[1, 0].set_ylabel("Front Length $z$ [mm]")

    axes[1, 1].set_title("(d) Discrete Nucleation-Growth vs. Continuum Stefan1D")
    axes[1, 1].set_xlabel("Time $t$ [s]")
    axes[1, 1].set_ylabel("Ice Thickness [mm]")

    fig1.legend(handles=axes[0, 0].get_lines(), loc="outside upper center", ncol=2, frameon=True, fontsize=8.8)
    for dst in (project_root / rel_paths[0], repo_figures_dir / repo_names[0]):
        dst.parent.mkdir(parents=True, exist_ok=True)
        fig1.savefig(dst, dpi=220)
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Figure 2: Eq. (16) Kinematics & Induced Pressure Gradient
    # ------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 3, figsize=(14.4, 4.7), constrained_layout=True)
    t_pos_mask = beta_runs[0]["t"] > 0.0
    for r in beta_runs:
        t_p = r["t"][t_pos_mask]
        c = r["color"]
        axes2[0].loglog(t_p, r["vz"][t_pos_mask] * 1e3, color=c, lw=2.0, label=f"$\\beta={int(r['beta'])}$")
        axes2[1].loglog(t_p, np.abs(r["exact"]["az"][t_pos_mask]) * 1e3, color=c, lw=2.0)
        axes2[2].loglog(t_p, r["dp_dz"][t_pos_mask], color=c, lw=2.0)

    # Reference power-law slope guides (t^-2 and t^-3)
    t_guide = np.geomspace(80.0, 550.0, 50)
    axes2[0].loglog(t_guide, 4.0e2 * (t_guide ** -2), "k--", lw=1.3, label="Asymptotic $\\propto t^{-2}$")
    axes2[1].loglog(t_guide, 1.5e4 * (t_guide ** -3), "k--", lw=1.3, label="Asymptotic $\\propto t^{-3}$")
    axes2[2].loglog(t_guide, 1.4e4 * (t_guide ** -3), "k--", lw=1.3, label="Eq. (16) $\\partial p/\\partial z \\propto t^{-3}$")

    axes2[0].set_title("(a) Front Velocity $\\dot{z}(t)$")
    axes2[0].set_xlabel("Time $t$ [s] (log)")
    axes2[0].set_ylabel("Velocity $\\dot{z}$ [mm/s]")
    axes2[0].legend(loc="lower left", fontsize=8.0)

    axes2[1].set_title("(b) Front Deceleration $-\\ddot{z}(t)$")
    axes2[1].set_xlabel("Time $t$ [s] (log)")
    axes2[1].set_ylabel("Deceleration $|\\ddot{z}|$ [mm/s$^2$]")
    axes2[1].legend(loc="lower left", fontsize=8.0)

    axes2[2].set_title("(c) Crystallization Pressure Gradient $\\partial p/\\partial z$")
    axes2[2].set_xlabel("Time $t$ [s] (log)")
    axes2[2].set_ylabel("$\\partial p/\\partial z = -\\rho\\ddot{z}$ [Pa/m]")
    axes2[2].legend(loc="lower left", fontsize=8.0)

    for dst in (project_root / rel_paths[1], repo_figures_dir / repo_names[1]):
        fig2.savefig(dst, dpi=220)
    plt.close(fig2)

    # ------------------------------------------------------------------
    # Figure 3: 2D Solidification Phase Map + 4-Stage Morphology Schematic
    # ------------------------------------------------------------------
    fig3, (ax_map, ax_morph) = plt.subplots(1, 2, figsize=(13.8, 5.4), constrained_layout=True)

    betas = np.geomspace(20.0, 2500.0, 60)
    n0s = np.geomspace(1e3, 5e4, 60)
    B_grid, N_grid = np.meshgrid(betas, n0s)
    # Exact time to 90% solidification: t_90 = (N0/beta) * ((z_max - z0)/(0.1*z_max) - 1)
    factor_90 = (z_max - y0_base["z"]) / (0.10 * z_max) - 1.0
    T90_grid = (N_grid / B_grid) * factor_90

    cf = ax_map.contourf(np.log10(B_grid), np.log10(N_grid), np.log10(T90_grid), levels=18, cmap="PuBu_r")
    cs = ax_map.contour(np.log10(B_grid), np.log10(N_grid), np.log10(T90_grid), levels=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5], colors="k", linewidths=0.8)
    ax_map.clabel(cs, fmt=lambda v: f"{10**v:.0f} s", fontsize=8)
    # Mark our 4 simulated cases
    for r in beta_runs:
        ax_map.scatter(np.log10(r["beta"]), np.log10(y0_base["N"]), color=r["color"], s=65, edgecolor="white", zorder=5)
    cbar = fig3.colorbar(cf, ax=ax_map, pad=0.02)
    cbar.set_label("$\\log_{10}(t_{90}\\,[\\mathrm{s}])$")
    ax_map.set_xlabel("$\\log_{10}(\\beta\\,[\\mathrm{m}^{-3}\\mathrm{s}^{-1}])$  (Nucleation Rate)")
    ax_map.set_ylabel("$\\log_{10}(N_0\\,[\\mathrm{m}^{-3}])$  (Initial Seed Concentration)")
    ax_map.set_title("(a) Regime Phase Map: Time to 90% Solidification $t_{90}(\\beta, N_0)$")
    ax_map.text(1.45, 4.45, "Nucleation-Limited\n(slow, coarse crystals)", color="white", fontweight="bold", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="#1B365D", alpha=0.75))
    ax_map.text(2.65, 3.20, "Growth-Limited\n(rapid, fine grains)", color="#1A202C", fontweight="bold", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="#EDF2F7", alpha=0.85))

    # Panel (b): 4-Stage Micro-to-Macro Morphology Schematic
    ax_morph.set_xlim(0, 10.4)
    ax_morph.set_ylim(0, 5.2)
    ax_morph.axis("off")
    ax_morph.set_title("(b) Cross-Section $\\sigma$ Morphology: Nucleation $\\to$ Solid")

    stage_specs = [
        (0.3, "1. Nucleation\n$t \\ll \\tau_N$", "nuclei"),
        (2.85, "2. Crystal Growth\n$t \\approx 0.6\\,\\tau_N$", "growth"),
        (5.4, "3. Coalescence\n$\\phi \\approx 0.64$", "coalesce"),
        (7.95, "4. Solid Front\nEq. (16) $-\\nabla p$", "continuum"),
    ]
    rng = np.random.default_rng(12)
    for x0, title_s, mode in stage_specs:
        # Container slab of area sigma
        rect = mpatches.FancyBboxPatch((x0, 0.9), 2.15, 3.1, boxstyle="round,pad=0.04", fc="#F7FAFC", ec="#1B365D", lw=1.5)
        ax_morph.add_patch(rect)
        # Cold base wall
        ax_morph.add_patch(mpatches.Rectangle((x0, 0.72), 2.15, 0.18, fc="#2C5282", ec="none"))
        ax_morph.text(x0 + 1.075, 4.35, title_s, ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1B365D")

        if mode == "nuclei":
            xs = x0 + rng.uniform(0.25, 1.9, 9)
            ys = 0.9 + rng.uniform(0.25, 2.7, 9)
            for px, py in zip(xs, ys):
                ax_morph.add_patch(mpatches.Circle((px, py), 0.07, fc="#63B3ED", ec="#2B6CB0", lw=0.8))
        elif mode == "growth":
            xs = x0 + rng.uniform(0.3, 1.85, 11)
            ys = 0.9 + rng.uniform(0.3, 2.5, 11)
            for px, py in zip(xs, ys):
                ax_morph.add_patch(mpatches.Circle((px, py), 0.18, fc="#90CDF4", ec="#2B6CB0", lw=1.0, alpha=0.9))
        elif mode == "coalesce":
            # Dense impinging crystals in lower 64%
            xs = x0 + np.array([0.35, 0.85, 1.35, 1.80, 0.55, 1.10, 1.65, 0.40, 0.95, 1.50, 1.85, 0.70, 1.30])
            ys = 0.9 + np.array([0.32, 0.30, 0.35, 0.32, 0.82, 0.80, 0.84, 1.32, 1.30, 1.35, 1.28, 1.75, 1.78])
            for px, py in zip(xs, ys):
                ax_morph.add_patch(mpatches.Circle((px, py), 0.30, fc="#63B3ED", ec="#1B365D", lw=1.0, alpha=0.82))
            ax_morph.plot([x0, x0 + 2.15], [0.9 + 2.0, 0.9 + 2.0], ls="--", color="#C53030", lw=1.2)
        elif mode == "continuum":
            # Solid macroscopic ice slab z(t)
            ax_morph.add_patch(mpatches.Rectangle((x0 + 0.02, 0.92), 2.11, 2.35, fc="#90CDF4", ec="#1B365D", lw=1.2, alpha=0.88))
            ax_morph.annotate("", xy=(x0 + 1.075, 3.65), xytext=(x0 + 1.075, 3.22),
                              arrowprops=dict(arrowstyle="->", color="#1E5631", lw=2.0))
            ax_morph.text(x0 + 1.20, 3.55, "$\\dot{z}(t)$", color="#1E5631", fontsize=8.5, fontweight="bold")
            # Opposing pressure gradient arrows
            for dx_a in (0.45, 1.70):
                ax_morph.annotate("", xy=(x0 + dx_a, 2.75), xytext=(x0 + dx_a, 3.25),
                                  arrowprops=dict(arrowstyle="->", color="#9B2C2C", lw=1.7))
            ax_morph.text(x0 + 1.075, 2.10, "Solid Ice\n$m=\\rho\\sigma z$\n$-\\frac{1}{\\rho}\\partial_z p$",
                          ha="center", va="center", fontsize=8.0, fontweight="bold", color="#1B365D")

    # Stage transition arrows
    for ax_x in (2.52, 5.07, 7.62):
        ax_morph.annotate("", xy=(ax_x + 0.28, 2.45), xytext=(ax_x, 2.45),
                          arrowprops=dict(arrowstyle="->", color="#4A5568", lw=1.8))
    ax_morph.text(5.2, 0.25, "Cross-sectional area $\\sigma = A$ ($\\mathsf{L}^2$), growth axis $z(t)$ ($\\mathsf{L}$), constant ice density $\\rho = 917\\,\\mathrm{kg/m}^3$",
                  ha="center", fontsize=8.5, style="italic", color="#4A5568")

    for dst in (project_root / rel_paths[2], repo_figures_dir / repo_names[2]):
        fig3.savefig(dst, dpi=220)
    plt.close(fig3)

    # ------------------------------------------------------------------
    # Figure 4: 4-Panel Verification & Conservation Suite
    # ------------------------------------------------------------------
    fig4, axes4 = plt.subplots(2, 2, figsize=(12.8, 8.2), constrained_layout=True)

    # (a) Analytical Error vs Time
    err_z = np.maximum(np.abs(traj_ref["z"] - exact_ref["z"]), 1e-17)
    err_vz = np.maximum(np.abs(traj_ref["vz"] - exact_ref["vz"]), 1e-17)
    axes4[0, 0].semilogy(traj_ref.t, err_z, color="#1B365D", lw=1.9, label="$|z_{\\mathrm{RK4}} - z_{\\mathrm{exact}}|$ [m]")
    axes4[0, 0].semilogy(traj_ref.t, err_vz, color="#B7791F", lw=1.7, ls="--", label="$|\\dot{z}_{\\mathrm{RK4}} - \\dot{z}_{\\mathrm{exact}}|$ [m/s]")
    axes4[0, 0].set_title("(a) Exact Analytical Verification Error ($\\Delta t = 0.5$ s)")
    axes4[0, 0].set_xlabel("Time $t$ [s]")
    axes4[0, 0].set_ylabel("Absolute Error")
    axes4[0, 0].legend(loc="upper right")

    # (b) RK4 Timestep GCI Convergence
    dts = np.array(dt_report.gci.spacings)
    q_vals = np.array(dt_report.gci.values)
    q_ext = float(dt_report.gci.extrapolated_value)
    disc_err = np.maximum(np.abs(q_vals - q_ext), 1e-16)
    axes4[0, 1].loglog(dts, disc_err, "o-", color="#1B365D", lw=2.1, ms=6, label="RK4 $|z(\\Delta t) - z_{\\mathrm{ext}}|$")
    axes4[0, 1].loglog(dts, disc_err[0] * ((dts / dts[0]) ** 4.0), "--", color="#C53030", lw=1.5, label="Formal $\\mathcal{O}(\\Delta t^4)$")
    axes4[0, 1].set_title(f"(b) RK4 Timestep Convergence ($p_{{\\mathrm{{obs}}}} = {dt_report.gci.observed_order:.2f}$)")
    axes4[0, 1].set_xlabel("Time Step $\\Delta t$ [s]")
    axes4[0, 1].set_ylabel("Discretization Error [m]")
    axes4[0, 1].legend(loc="upper left")

    # (c) Dual Conservation Residuals: Mass & Eq. (16) Momentum-Impulse
    res_m = np.maximum(np.abs(inv_report.entries["mass_conservation"].residual_series), 1e-17)
    res_p = np.maximum(np.abs(inv_report.entries["momentum_impulse_eq16"].residual_series), 1e-17)
    axes4[1, 0].semilogy(traj_ref.t, res_m, color="#1E5631", lw=1.8, label="Mass $|m + m_{\\mathrm{liq}} - m_{\\max}|$")
    axes4[1, 0].semilogy(traj_ref.t, res_p, color="#2C5282", lw=1.8, ls="-.", label="Eq. (16) $|\\rho\\dot{z} + P_{\\mathrm{imp}} - \\rho\\dot{z}_0|$")
    axes4[1, 0].axhline(1e-12, color="#C53030", ls="--", lw=1.2, label="Audit tolerance $10^{-12}$")
    axes4[1, 0].set_title("(c) Dual Symbolic & Discrete Conservation Audit")
    axes4[1, 0].set_xlabel("Time $t$ [s]")
    axes4[1, 0].set_ylabel("Invariant Drift")
    axes4[1, 0].set_ylim(1e-17, 1e-10)
    axes4[1, 0].legend(loc="upper right", fontsize=8.0)

    # (d) Exact Parity Plot across all 4 beta cases
    z_all_ex = np.concatenate([r["exact"]["z"] * 1e3 for r in beta_runs])
    z_all_num = np.concatenate([r["z"] * 1e3 for r in beta_runs])
    axes4[1, 1].plot([0, 52], [0, 52], "k--", lw=1.3, label="Exact 1:1 Parity")
    for r in beta_runs:
        axes4[1, 1].scatter(r["exact"]["z"][::35] * 1e3, r["z"][::35] * 1e3, color=r["color"], s=28, alpha=0.85, label=f"$\\beta={int(r['beta'])}$")
    axes4[1, 1].set_xlim(0, 52)
    axes4[1, 1].set_ylim(0, 52)
    axes4[1, 1].set_title("(d) Analytical vs. Numerical Parity Across All Regimes")
    axes4[1, 1].set_xlabel("Exact Analytical $z(t)$ [mm]")
    axes4[1, 1].set_ylabel("ODE Simulated $z(t)$ [mm]")
    axes4[1, 1].legend(loc="upper left", fontsize=8.0)

    for dst in (project_root / rel_paths[3], repo_figures_dir / repo_names[3]):
        fig4.savefig(dst, dpi=220)
    plt.close(fig4)

    return rel_paths


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m examples.ice_nucleation_growth_study.run_study",
        description="Run the Rosolver Ice Nucleation-to-Solid-Mass Growth Study.",
    )
    parser.add_argument(
        "--output",
        default="outputs/ice_nucleation_growth_study",
        help="Target workspace directory (default: outputs/ice_nucleation_growth_study).",
    )
    args = parser.parse_args(argv)
    summary = run_ice_nucleation_study(output_dir=args.output)
    print(f"Ice nucleation-growth study completed: status={summary['report_status']}")
    print(f"  Ref. coalescence time t_c   : {summary['t_coalescence_ref_s']} s")
    print(f"  Ref. final z(600 s)         : {summary['z_final_ref_mm']} mm (Stefan1D: {summary['stefan_final_mm']} mm)")
    print(f"  Max |z_RK4 - z_exact|       : {summary['max_z_abs_err_m']:.3e} m")
    print(f"  RK4 observed order p_obs    : {summary['rk4_observed_order']}")
    print(f"  Mass conservation residual  : {summary['mass_conservation_max_res']:.3e}")
    print(f"  Eq.(16) impulse residual    : {summary['momentum_impulse_max_res']:.3e}")
    print(f"  Report written to           : {summary['report_path']}")
    return 0 if summary["report_status"] in ("Passed", "Warning") else 1


if __name__ == "__main__":
    raise SystemExit(main())
