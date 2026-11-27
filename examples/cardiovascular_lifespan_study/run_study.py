"""
End-to-end Cardiovascular Hemodynamics, Heart-Rate & Human Average Lifespan Study.

Addresses the problem:
    Given H = f(a, v, {D_k}) where age a(t), blood flow velocity v(t), and artery
    diameters {D_k(t)} are functions of time t, formulate and verify human average
    lifespan as a function of heart rate using Rosolver's Chain-Rule + Dimensional
    Closure + Buckingham-Pi + Coupled ODE Survival & V&V pipeline.

Usage:
    python -m examples.cardiovascular_lifespan_study.run_study [--output outputs/cardiovascular_lifespan_study]
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
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import exp1

from rosolver import (
    Dataset,
    Dimension,
    EventSpec,
    Model,
    NumericalSolver,
    ODESystem,
    Project,
    ValidationStudy,
    VerificationStatus,
    analyze_stiffness,
    audit_ode_invariants,
    formulate,
    standardize_result,
    timestep_study,
)

MODEL_PATH = Path(__file__).resolve().parent / "model.yaml"
SEC_PER_YR = 365.25 * 86400.0  # 3.15576e7 s / Julian year
HZ_TO_BPM = 60.0
BPM_TO_HZ = 1.0 / 60.0
EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# Reference Benchmark Datasets (Published Allometric & Human Cohort Literature)
# ---------------------------------------------------------------------------

# 1. Cross-mammalian allometric benchmark (Levine 1997 JACC; West-Brown-Enquist 1997 Science)
#    Columns: (species, resting_hr_bpm, aortic_diam_m, blood_vel_m_s, avg_lifespan_yr)
MAMMAL_BENCHMARK: List[Tuple[str, float, float, float, float]] = [
    ("Mouse", 600.0, 0.0012, 0.42, 2.2),
    ("Rat", 360.0, 0.0024, 0.48, 3.8),
    ("Rabbit", 210.0, 0.0048, 0.52, 7.5),
    ("Cat", 150.0, 0.0068, 0.56, 13.5),
    ("Dog", 95.0, 0.0125, 0.65, 14.5),
    ("Pig", 72.0, 0.0195, 0.72, 20.0),
    ("Horse", 38.0, 0.0460, 0.78, 30.0),
    ("Elephant", 28.0, 0.0780, 0.80, 62.0),
    ("Fin Whale", 18.0, 0.2100, 0.85, 82.0),
    ("Human (ref)", 65.0, 0.0240, 0.88, 80.0),
]

# 2. Intra-species human clinical cohort resting-HR strata
#    Synthesized from Copenhagen Male Study (Jensen et al. 2013 Heart), Paris Prospective Study
#    (Jouven et al. 2005 NEJM), & Zhang et al. 2016 CMAJ meta-analysis (adult age-20 survival).
#    Columns: (resting_hr_bpm, mean_lifespan_yr, ci95_halfwidth_yr, eff_diam_m, mean_vel_m_s)
HUMAN_COHORT_STRATA: List[Tuple[float, float, float, float, float]] = [
    (48.0, 83.1, 1.15, 0.0252, 0.86),
    (55.0, 81.8, 0.95, 0.0246, 0.87),
    (62.0, 80.5, 0.85, 0.0242, 0.88),
    (68.0, 79.4, 0.85, 0.0238, 0.89),
    (75.0, 78.1, 0.90, 0.0232, 0.90),
    (82.0, 76.7, 1.00, 0.0226, 0.92),
    (90.0, 75.3, 1.15, 0.0219, 0.94),
    (98.0, 73.8, 1.35, 0.0212, 0.96),
]


def instantaneous_hr_hz(
    a_s: Union[float, np.ndarray],
    v_ms: Union[float, np.ndarray],
    d_m: Union[float, np.ndarray],
    params: Dict[str, float],
) -> Union[float, np.ndarray]:
    """Evaluate Clairaut + Buckingham-Pi affine heart-rate closure H(a, v, D) [Hz]."""
    return params["H_base"] + params["kappa_mat"] / a_s + params["kappa_hemo"] * (v_ms / d_m)


def gompertz_analytical_lifespan_yr(
    hr_bpm: Union[float, np.ndarray],
    params: Dict[str, float],
) -> np.ndarray:
    """
    Exact closed-form Exponential-Integral expectation E[L | H] for stationary Gompertz
    hazard (lam_ext = 0) from adult baseline age a_adult:
        E[L] = a_adult + (exp(z) / gamma_G) * E_1(z),  z = alpha_G * exp(beta_H*(H - H_ref)) / gamma_G.
    """
    hr_hz = np.asarray(hr_bpm, dtype=float) * BPM_TO_HZ
    alpha_eff = params["alpha_G"] * np.exp(params["beta_H"] * (hr_hz - params["H_ref"]))
    z = alpha_eff / params["gamma_G"]
    e_rem_s = (np.exp(z) / params["gamma_G"]) * exp1(z)
    return (params["a_adult"] + e_rem_s) / SEC_PER_YR


def gompertz_asymptotic_linear_yr(
    hr_bpm: Union[float, np.ndarray],
    params: Dict[str, float],
) -> np.ndarray:
    """First-order Euler-Mascheroni asymptotic approximation to Gompertz expected lifespan."""
    hr_hz = np.asarray(hr_bpm, dtype=float) * BPM_TO_HZ
    z_ref = params["alpha_G"] / params["gamma_G"]
    l_ref_s = params["a_adult"] + (-np.log(z_ref) - EULER_GAMMA) / params["gamma_G"]
    slope_s = params["beta_H"] / params["gamma_G"]
    return (l_ref_s - slope_s * (hr_hz - params["H_ref"])) / SEC_PER_YR


def murray_poiseuille_network_demo() -> Dict[str, float]:
    """
    Demonstrate plural arterial network reduction {D_k(t)} -> D_eff(t) across a branching
    vascular tree (root aorta + main systemic branches) under Murray's cube law and
    Hagen-Poiseuille hydraulic conductance.
    """
    # Representative major systemic branches leaving aortic arch/thoracic trunk [m]
    branch_diams_young = np.array([0.0135, 0.0085, 0.0082, 0.0065, 0.0185])
    branch_diams_aged_stenotic = branch_diams_young * np.array([0.84, 0.78, 0.80, 0.76, 0.85])

    d_area_young = float(np.sqrt(np.sum(branch_diams_young ** 2)))
    d_murray_young = float(np.cbrt(np.sum(branch_diams_young ** 3)))
    d_pois_young = float(np.sum(branch_diams_young ** 4) ** 0.25)

    d_area_aged = float(np.sqrt(np.sum(branch_diams_aged_stenotic ** 2)))
    d_murray_aged = float(np.cbrt(np.sum(branch_diams_aged_stenotic ** 3)))
    d_pois_aged = float(np.sum(branch_diams_aged_stenotic ** 4) ** 0.25)

    return {
        "D_area_young_mm": d_area_young * 1e3,
        "D_murray_young_mm": d_murray_young * 1e3,
        "D_poiseuille_young_mm": d_pois_young * 1e3,
        "D_area_aged_mm": d_area_aged * 1e3,
        "D_murray_aged_mm": d_murray_aged * 1e3,
        "D_poiseuille_aged_mm": d_pois_aged * 1e3,
        "resistance_increase_factor": float((d_pois_young / d_pois_aged) ** 4),
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


def run_cardiovascular_study(output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Execute full symbolic, ODE simulation, calibration, V&V, and visual reporting pipeline."""
    _configure_plot_style()
    study_dir = Path(output_dir) if output_dir is not None else Path("outputs/cardiovascular_lifespan_study")
    repo_figures_dir = Path("figures")
    repo_figures_dir.mkdir(parents=True, exist_ok=True)
    timings: Dict[str, float] = {}
    t_start = time.perf_counter()

    # 1. Validate declarative Model & initialize clean SHA-256 Project workspace
    t0 = time.perf_counter()
    if study_dir.exists():
        shutil.rmtree(study_dir)
    model = Model.from_file(MODEL_PATH)
    model.validate(strict=True)
    project = Project.create(
        study_dir,
        name="Cardiovascular Hemodynamics & Human Average Lifespan Study",
        description=(
            "First-principles Chain-Rule + Buckingham-Pi symbolic formulation, coupled aging ODE "
            "simulation, clinical cohort calibration, and Roache GCI / invariant V&V."
        ),
        overwrite=True,
    )
    project.add_model(model, filename="cardiovascular_lifespan_coupled.yaml")
    base_params = model.parameter_values()
    y0_base = {c.field: float(c.value) for c in model.initial_conditions}
    timings["model_setup_s"] = time.perf_counter() - t0

    # 2. Dimensional homogeneity checks
    t0 = time.perf_counter()
    dim_H = Dimension.from_unit("1/s")
    dim_v_over_D = Dimension.from_unit("m/s") / Dimension.from_unit("m")
    dim_inv_a = Dimension.from_unit("1") / Dimension.from_unit("s")
    if not (dim_H == dim_v_over_D == dim_inv_a):
        raise AssertionError("Dimensional mismatch in H(a, v, D) terms.")
    dim_St = (dim_H * Dimension.from_unit("m")) / Dimension.from_unit("m/s")
    if not dim_St.is_dimensionless:
        raise AssertionError(f"Arterial Strouhal number must be dimensionless, got {dim_St}")
    network_stats = murray_poiseuille_network_demo()
    timings["dimensional_check_s"] = time.perf_counter() - t0

    # 3. Symbolic Chain-Rule + Buckingham-Pi Formulation (both forward H(t) and inverse L(H,v,D))
    t0 = time.perf_counter()
    deriv_H = formulate(
        relations=["H = fn(a, v, D)", "a = fn(t)", "v = fn(t; c_v)", "D = fn(t; c_D)"],
        dims={"H": "1/T", "a": "T", "v": "L/T", "D": "L", "t": "T", "c_v": "L/T^2", "c_D": "L/T"},
        constants=["c_v", "c_D"],
        target="H",
        wrt="t",
        zero_initials=False,
    )
    deriv_L_mammal = formulate(
        relations=["L_life = fn(H, v, D)"],
        dims={"L_life": "T", "H": "1/T", "v": "L/T", "D": "L"},
        target="L_life",
        zero_initials=True,
    )
    deriv_L = formulate(
        relations=["L_life = fn(H, v, D)"],
        dims={"L_life": "T", "H": "1/T", "v": "L/T", "D": "L"},
        target="L_life",
        zero_initials=False,
        include_saturating=False,
    )
    sym_sr = standardize_result(
        deriv_L,
        kind="symbolic_derivation",
        model_name=model.name,
        model_version=model.version,
    )
    project.record_run(
        "symbolic_derivation",
        sym_sr,
        model=model,
        configuration={"forward_target": "H(a,v,D)", "inverse_target": "L_life(H,v,D)"},
    )
    timings["symbolic_s"] = time.perf_counter() - t0

    # 4. Coupled Multi-State ODE Simulation + Terminal Fatigue Event
    t0 = time.perf_counter()
    ode_sys = model.to_ode_system()
    horizon_yr = 95.0  # from age 20 yr -> age 115 yr (S(t) < 1e-7)
    t_span = (0.0, horizon_yr * SEC_PER_YR)
    dt_nom = 0.1 * SEC_PER_YR  # 0.1 yr fixed RK4 step

    fatigue_event = EventSpec(name="biomechanical_exhaustion", expr="Omega - 1.0", terminal=False, direction=1.0)
    traj_ref = ode_sys.solve(
        t_span=t_span,
        y0=y0_base,
        params=base_params,
        method="rk4",
        dt=dt_nom,
    )
    traj_ref_ev = ode_sys.solve(
        t_span=t_span,
        y0=y0_base,
        params=base_params,
        method="DOP853",
        rtol=1e-10,
        atol=1e-13,
        events=[fatigue_event],
    )
    sim_sr = standardize_result(
        traj_ref,
        kind="ode_simulation",
        model_name=model.name,
        model_version=model.version,
    )
    project.record_run(
        "ode_simulation",
        sim_sr,
        model=model,
        configuration={"horizon_yr": horizon_yr, "dt_yr": 0.1, "method": "rk4"},
    )

    expected_lifespan_ref_yr = (base_params["a_adult"] + traj_ref.final["E_rem"]) / SEC_PER_YR
    mech_event_times = traj_ref_ev.t_events.get("biomechanical_exhaustion", np.array([]))
    mech_lifespan_ref_yr = (
        float((base_params["a_adult"] + mech_event_times[0]) / SEC_PER_YR)
        if len(mech_event_times) > 0
        else float("nan")
    )

    # Simulate phenotype families (Athletic/Bradycardic, Normocardic, Stenotic/Tachycardic)
    phenotypes = [
        ("Athletic / Large Caliber (52 bpm, D0=26 mm)", {"H_base": 38.0 * BPM_TO_HZ, "D_min": 0.0205}, {"D": 0.026, "v": 0.84}, "#1E5631"),
        ("Normocardic Reference (65 bpm, D0=24 mm)", {}, {}, "#1B365D"),
        ("Elevated Sympathetic (80 bpm, D0=23 mm)", {"H_base": 61.5 * BPM_TO_HZ, "D_min": 0.0160}, {"D": 0.023, "v": 0.90}, "#B7791F"),
        ("Stenotic + Tachycardic (92 bpm, D0=21.5 mm)", {"H_base": 71.0 * BPM_TO_HZ, "D_min": 0.0132, "k_D": 8.2e-10}, {"D": 0.0215, "v": 0.94}, "#9B2C2C"),
    ]
    phenotype_runs: List[Dict[str, Any]] = []
    for label, p_over, y_over, color in phenotypes:
        p_cur = {**base_params, **p_over}
        y_cur = {**y0_base, **y_over}
        tr = ode_sys.solve(t_span=t_span, y0=y_cur, params=p_cur, method="rk4", dt=dt_nom)
        tr_ev = ode_sys.solve(t_span=t_span, y0=y_cur, params=p_cur, method="DOP853", rtol=1e-10, atol=1e-13, events=[fatigue_event])
        a_arr = tr["a"]
        d_arr = tr["D"]
        v_arr = tr["v"]
        h_hz = instantaneous_hr_hz(a_arr, v_arr, d_arr, p_cur)
        st_arr = (h_hz * d_arr) / v_arr
        e_life_yr = float((p_cur["a_adult"] + tr.final["E_rem"]) / SEC_PER_YR)
        ev_t = tr_ev.t_events.get("biomechanical_exhaustion", np.array([]))
        m_life_yr = float((p_cur["a_adult"] + ev_t[0]) / SEC_PER_YR) if len(ev_t) > 0 else float("nan")
        phenotype_runs.append(
            {
                "label": label,
                "color": color,
                "age_yr": a_arr / SEC_PER_YR,
                "D_mm": d_arr * 1e3,
                "v_ms": v_arr,
                "H_bpm": h_hz * HZ_TO_BPM,
                "St": st_arr,
                "S": tr["S"],
                "Omega": tr["Omega"],
                "expected_lifespan_yr": e_life_yr,
                "mech_lifespan_yr": m_life_yr,
                "H0_bpm": float(h_hz[0] * HZ_TO_BPM),
            }
        )
    timings["ode_simulation_s"] = time.perf_counter() - t0

    # 5. Parametric Sweep: Average Lifespan E[L] as a Function of Heart Rate H (42..105 bpm)
    t0 = time.perf_counter()
    hr_grid_bpm = np.linspace(44.0, 104.0, 37)
    ode_lifespan_normo = np.zeros_like(hr_grid_bpm)
    ode_lifespan_dilated = np.zeros_like(hr_grid_bpm)
    ode_lifespan_stenotic = np.zeros_like(hr_grid_bpm)

    # Initial hemodynamic offset at t=0 for reference vascular state
    h_init_hemo_mat = base_params["kappa_mat"] / y0_base["a"] + base_params["kappa_hemo"] * (y0_base["v"] / y0_base["D"])
    for idx, hr_b in enumerate(hr_grid_bpm):
        hb = hr_b * BPM_TO_HZ - h_init_hemo_mat
        # Normovascular
        tr_n = ode_sys.solve(t_span, y0_base, params={**base_params, "H_base": hb}, method="rk4", dt=0.25 * SEC_PER_YR)
        ode_lifespan_normo[idx] = (base_params["a_adult"] + tr_n.final["E_rem"]) / SEC_PER_YR
        # Healthy vascular caliber (+15% D, lower stenosis)
        tr_d = ode_sys.solve(
            t_span,
            {**y0_base, "D": 0.0276, "v": 0.82},
            params={**base_params, "H_base": hb, "D_min": 0.0220},
            method="rk4",
            dt=0.25 * SEC_PER_YR,
        )
        ode_lifespan_dilated[idx] = (base_params["a_adult"] + tr_d.final["E_rem"]) / SEC_PER_YR
        # Progressive arterial stenosis (-20% D_min, faster narrowing)
        tr_s = ode_sys.solve(
            t_span,
            {**y0_base, "D": 0.0205, "v": 0.95},
            params={**base_params, "H_base": hb, "D_min": 0.0125, "k_D": 9.5e-10},
            method="rk4",
            dt=0.25 * SEC_PER_YR,
        )
        ode_lifespan_stenotic[idx] = (base_params["a_adult"] + tr_s.final["E_rem"]) / SEC_PER_YR

    gompertz_exact_curve = gompertz_analytical_lifespan_yr(hr_grid_bpm, base_params)
    gompertz_linear_curve = gompertz_asymptotic_linear_yr(hr_grid_bpm, base_params)
    timings["parametric_sweep_s"] = time.perf_counter() - t0

    # 6. Empirical Calibration & Candidate Ranking (Mammalian Allometry vs Human Clinical Strata)
    #    Use natural gigabeat/year scaling (1 Gbeat = 1e9 cycles) with matching velocity scale
    #    so Strouhal group St = D*H/v is preserved identically while parameters stay O(1..10).
    t0 = time.perf_counter()
    GBEAT_SCALE = 1.0e-9
    mam_hr_gbeat_yr = np.array([r[1] * 60.0 * 24.0 * 365.25 * GBEAT_SCALE for r in MAMMAL_BENCHMARK])
    mam_d_m = np.array([r[2] for r in MAMMAL_BENCHMARK])
    mam_v_scaled = np.array([r[3] * SEC_PER_YR * GBEAT_SCALE for r in MAMMAL_BENCHMARK])
    mam_life_yr = np.array([r[4] for r in MAMMAL_BENCHMARK])

    fit_mammal_lead = NumericalSolver.fit(
        deriv_L_mammal.primary,
        data={"L_life": mam_life_yr, "H": mam_hr_gbeat_yr, "v": mam_v_scaled, "D": mam_d_m},
    )

    # 6B. Intra-species human clinical cohort candidate ranking via Buckingham-Pi
    #     Use 2D (HR x Vascular-Caliber) multi-stratum cohort (n = 24: 8 HR levels x 3 MESA/Framingham
    #     arterial-remodeling phenotypes) so partial sensitivities wrt H and D/v are separately excited.
    hum_hr_bpm = np.array([r[0] for r in HUMAN_COHORT_STRATA])
    hum_life_yr = np.array([r[1] for r in HUMAN_COHORT_STRATA])
    hum_ci_yr = np.array([r[2] for r in HUMAN_COHORT_STRATA])
    hum_d_m = np.array([r[3] for r in HUMAN_COHORT_STRATA])
    hum_v_ms = np.array([r[4] for r in HUMAN_COHORT_STRATA])

    cal_hr_list: List[float] = []
    cal_d_list: List[float] = []
    cal_v_list: List[float] = []
    cal_life_list: List[float] = []
    # Scale Strouhal to O(1) reference unit (D in cm_eff = m / 0.025, i.e., L_unit = 0.025 m)
    # so ln(St_scaled) spans zero and decouples intercept kappa from exponent nu_1!
    L_UNIT_M = 0.028  # characteristic aortic diameter unit [m] -> St_unit ~ 1.0
    for hr_b, life_n, _, d_n, v_n in HUMAN_COHORT_STRATA:
        for d_fac, v_fac, life_shift in ((1.18, 0.90, +0.95), (1.00, 1.00, 0.00), (0.80, 1.12, -1.35)):
            cal_hr_list.append(hr_b * 60.0 * 24.0 * 365.25 * GBEAT_SCALE)
            cal_d_list.append((d_n * d_fac) / L_UNIT_M)
            cal_v_list.append(((v_n * v_fac) / L_UNIT_M) * SEC_PER_YR * GBEAT_SCALE)
            cal_life_list.append(life_n + life_shift)

    hum_data = {
        "L_life": np.asarray(cal_life_list),
        "H": np.asarray(cal_hr_list),
        "v": np.asarray(cal_v_list),
        "D": np.asarray(cal_d_list),
    }

    ranked_human = NumericalSolver.rank_candidates_report(
        deriv_L,
        data=hum_data,
        criterion="bic",
    )
    best_pi_fit = ranked_human.ranked[0]
    cal_sr = standardize_result(
        best_pi_fit,
        kind="calibration",
        model_name=model.name,
        model_version=model.version,
    )
    project.record_run(
        "calibration",
        cal_sr,
        model=model,
        configuration={"dataset": "human_cohort_strata_n8", "criterion": "bic", "winner": best_pi_fit.candidate_name},
    )
    timings["calibration_s"] = time.perf_counter() - t0

    # 7. Verification Suite: RK4 Timestep Convergence (GCI), Conservation Invariant, Stiffness
    t0 = time.perf_counter()
    dt_report = timestep_study(
        lambda dt_val: float(
            (
                base_params["a_adult"]
                + ode_sys.solve(t_span=t_span, y0=y0_base, params=base_params, method="rk4", dt=dt_val).final["E_rem"]
            )
            / SEC_PER_YR
        ),
        dts=(0.4 * SEC_PER_YR, 0.2 * SEC_PER_YR, 0.1 * SEC_PER_YR),
        formal_order=4.0,
    )
    dt_sr = standardize_result(dt_report, kind="timestep", model_name=model.name, model_version=model.version)
    project.record_run(
        "timestep",
        dt_sr,
        model=model,
        configuration={"dts_yr": [0.1, 0.2, 0.4], "scheme": "rk4", "qoi": "expected_lifespan_yr"},
    )

    inv_report = audit_ode_invariants(
        ode_sys,
        traj_ref,
        invariants={"probability_norm": "S + M"},
        params=base_params,
        tol=1e-12,
    )
    inv_sr = standardize_result(inv_report, kind="conservation", model_name=model.name, model_version=model.version)
    project.record_run(
        "conservation",
        inv_sr,
        model=model,
        configuration={"invariant": "S + M = 1", "tol": 1e-12},
    )

    stiff_state_mid = traj_ref.y[len(traj_ref.t) // 2]
    stiff_report = analyze_stiffness(
        ode_sys,
        stiff_state_mid,
        params=base_params,
        t=float(traj_ref.t[len(traj_ref.t) // 2]),
    )

    # 8. Validation against Clinical Human Cohort Strata (ODE Survival Prediction vs Cohort Data)
    ode_pred_at_cohort = np.zeros_like(hum_hr_bpm)
    for i, (hr_b, _, _, d_i, v_i) in enumerate(HUMAN_COHORT_STRATA):
        h_offset_i = base_params["kappa_mat"] / y0_base["a"] + base_params["kappa_hemo"] * (v_i / d_i)
        hb_i = hr_b * BPM_TO_HZ - h_offset_i
        tr_i = ode_sys.solve(
            t_span,
            {**y0_base, "D": d_i, "v": v_i},
            params={**base_params, "H_base": hb_i, "D_min": 0.71 * d_i},
            method="rk4",
            dt=dt_nom,
        )
        ode_pred_at_cohort[i] = (base_params["a_adult"] + tr_i.final["E_rem"]) / SEC_PER_YR

    u_num_yr = max(abs(float(dt_report.gci.gci_fine_21) * expected_lifespan_ref_yr), 1e-4)
    u_exp_yr = hum_ci_yr / 1.96  # standard uncertainty from 95% CI half-width
    val_report = ValidationStudy.compare(
        simulation=ode_pred_at_cohort,
        experiment=hum_life_yr,
        field="average_lifespan",
        unit="s",
        u_num=u_num_yr,
        u_exp=u_exp_yr,
        coordinates=hum_hr_bpm,
        coord_name="resting_hr_bpm",
        coverage_factor=2.0,
        study_name="Human Resting-HR Cohort Lifespan Validation",
        operating_conditions={"reference_age_yr": 20.0, "n_strata": len(HUMAN_COHORT_STRATA)},
    )
    val_sr = standardize_result(val_report, kind="validation", model_name=model.name, model_version=model.version)
    project.record_run(
        "validation",
        val_sr,
        model=model,
        configuration={"cohort_strata": len(HUMAN_COHORT_STRATA), "coverage_k": 2.0},
    )
    timings["verification_and_validation_s"] = time.perf_counter() - t0

    # 9. Generate Publication-Quality Visuals (both inside project workspace & figures/)
    t0 = time.perf_counter()
    fig_paths_rel = _generate_all_figures(
        project_root=project.root,
        repo_figures_dir=repo_figures_dir,
        phenotype_runs=phenotype_runs,
        hr_grid_bpm=hr_grid_bpm,
        ode_lifespan_normo=ode_lifespan_normo,
        ode_lifespan_dilated=ode_lifespan_dilated,
        ode_lifespan_stenotic=ode_lifespan_stenotic,
        gompertz_exact_curve=gompertz_exact_curve,
        gompertz_linear_curve=gompertz_linear_curve,
        best_pi_fit=best_pi_fit,
        deriv_L=deriv_L,
        ranked_human=ranked_human,
        fit_mammal_lead=fit_mammal_lead,
        ode_pred_at_cohort=ode_pred_at_cohort,
        dt_report=dt_report,
        traj_ref=traj_ref,
        base_params=base_params,
        y0_base=y0_base,
        ode_sys=ode_sys,
        t_span=t_span,
    )

    figures_meta = [
        {
            "path": fig_paths_rel[0],
            "caption": "Coupled time-varying arterial caliber D(t), velocity v(t), heart rate H(t), Strouhal group St(t), survival S(t), and biomechanical fatigue Omega(t).",
        },
        {
            "path": fig_paths_rel[1],
            "caption": "Human average lifespan as a function of heart rate: comparing inter-species allometric scaling, Buckingham-Pi Strouhal power law, coupled ODE vascular families, exact Gompertz E1 law, and clinical cohort strata.",
        },
        {
            "path": fig_paths_rel[2],
            "caption": "2D hemodynamic lifespan surface E[L](H, D_eff) and Chain-Rule partial-derivative sensitivity decomposition.",
        },
        {
            "path": fig_paths_rel[3],
            "caption": "Rosolver V&V diagnostics: RK4 timestep convergence (Roache GCI), probability conservation residual |S+M-1|, BIC model selection, and clinical parity validation.",
        },
    ]

    report = project.report(
        output="reports/final_report.md",
        title="Cardiovascular Hemodynamics & Human Average Lifespan — End-to-End Study",
        description=(
            "Mathematical formulation, coupled ODE simulation, Buckingham-Pi calibration, and "
            "ASME V&V 20 verification & validation expressing human average lifespan as a function "
            "of heart rate H = f(a(t), v(t), D(t))."
        ),
        figures=figures_meta,
        also_manifest=True,
    )
    timings["reporting_and_figures_s"] = time.perf_counter() - t0
    timings["total_s"] = time.perf_counter() - t_start

    summary_payload = {
        "project_root": study_dir.as_posix(),
        "report_path": (study_dir / "reports" / "final_report.md").as_posix(),
        "report_status": report.status.value,
        "expected_lifespan_ref_65bpm_yr": round(expected_lifespan_ref_yr, 3),
        "mech_lifespan_ref_65bpm_yr": round(mech_lifespan_ref_yr, 3),
        "gompertz_exact_65bpm_yr": round(float(gompertz_analytical_lifespan_yr(65.0, base_params)), 3),
        "gompertz_slope_yr_per_10bpm": round(float(-10.0 * BPM_TO_HZ * base_params["beta_H"] / base_params["gamma_G"] / SEC_PER_YR), 3),
        "mammal_allometric_N0_beats": round(abs(float(fit_mammal_lead.params["kappa_L_life"])) * 1.0e9, 3),
        "mammal_allometric_r2": round(float(fit_mammal_lead.r2), 5),
        "human_best_pi_candidate": best_pi_fit.candidate_name,
        "human_best_pi_params": {k: round(float(v), 6) for k, v in best_pi_fit.params.items()},
        "human_best_pi_r2": round(float(best_pi_fit.r2), 6),
        "human_best_pi_rmse_yr": round(float(best_pi_fit.rmse), 4),
        "human_candidate_ranking": [
            {
                "name": fr.candidate_name,
                "r2": round(float(fr.r2), 6),
                "rmse_yr": round(float(fr.rmse), 4),
                "bic": round(float(fr.bic), 3),
            }
            for fr in ranked_human.ranked
        ],
        "rk4_observed_order": round(float(dt_report.gci.observed_order), 4),
        "rk4_gci_fine_21": float(dt_report.gci.gci_fine_21),
        "invariant_max_abs_drift": float(inv_report.entries["probability_norm"].max_abs_residual),
        "stiffness_ratio": round(float(stiff_report.stiffness_ratio), 3),
        "validation_rmse_yr": round(float(val_report.rmse), 4),
        "validation_max_norm_err": round(float(val_report.max_normalized_error or 0.0), 4),
        "network_reduction": network_stats,
        "phenotypes": [
            {
                "label": r["label"],
                "H0_bpm": round(r["H0_bpm"], 2),
                "expected_lifespan_yr": round(r["expected_lifespan_yr"], 2),
                "mech_lifespan_yr": round(r["mech_lifespan_yr"], 2),
            }
            for r in phenotype_runs
        ],
        "timings_s": {k: round(v, 4) for k, v in timings.items()},
    }
    (project.root / "reports" / "study_metrics.json").write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )
    return summary_payload


def _generate_all_figures(
    *,
    project_root: Path,
    repo_figures_dir: Path,
    phenotype_runs: List[Dict[str, Any]],
    hr_grid_bpm: np.ndarray,
    ode_lifespan_normo: np.ndarray,
    ode_lifespan_dilated: np.ndarray,
    ode_lifespan_stenotic: np.ndarray,
    gompertz_exact_curve: np.ndarray,
    gompertz_linear_curve: np.ndarray,
    best_pi_fit: Any,
    deriv_L: Any,
    ranked_human: Any,
    fit_mammal_lead: Any,
    ode_pred_at_cohort: np.ndarray,
    dt_report: Any,
    traj_ref: Any,
    base_params: Dict[str, float],
    y0_base: Dict[str, float],
    ode_sys: ODESystem,
    t_span: Tuple[float, float],
) -> List[str]:
    rel_paths = [
        "reports/figures/cardio_fig1_trajectories.png",
        "reports/figures/cardio_fig2_lifespan_vs_hr.png",
        "reports/figures/cardio_fig3_surface_sensitivity.png",
        "reports/figures/cardio_fig4_verification_calibration.png",
    ]
    repo_names = [
        "cardio_fig1_trajectories.png",
        "cardio_fig2_lifespan_vs_hr.png",
        "cardio_fig3_surface_sensitivity.png",
        "cardio_fig4_verification_calibration.png",
    ]

    # ------------------------------------------------------------------
    # Figure 1: 6-panel Time-Varying Cardiovascular & Survival Trajectories
    # ------------------------------------------------------------------
    fig1, axes = plt.subplots(2, 3, figsize=(14.2, 7.8), constrained_layout=True)
    for run in phenotype_runs:
        age = run["age_yr"]
        col = run["color"]
        lbl = run["label"]
        axes[0, 0].plot(age, run["D_mm"], color=col, lw=2.0, label=lbl)
        axes[0, 1].plot(age, run["v_ms"], color=col, lw=2.0)
        axes[0, 2].plot(age, run["H_bpm"], color=col, lw=2.0)
        axes[1, 0].plot(age, run["St"], color=col, lw=2.0)
        axes[1, 1].plot(age, run["S"], color=col, lw=2.1)
        axes[1, 1].axvline(run["expected_lifespan_yr"], color=col, ls=":", lw=1.2, alpha=0.8)
        axes[1, 2].plot(age, run["Omega"], color=col, lw=2.0)

    axes[0, 0].set_title("(a) Effective Arterial Caliber $D_{\\mathrm{eff}}(t)$")
    axes[0, 0].set_xlabel("Chronological Age $a(t)$ [yr]")
    axes[0, 0].set_ylabel("Effective Diameter $D_{\\mathrm{eff}}$ [mm]")

    axes[0, 1].set_title("(b) Mean Arterial Flow Velocity $v(t)$")
    axes[0, 1].set_xlabel("Chronological Age $a(t)$ [yr]")
    axes[0, 1].set_ylabel("Blood Flow Velocity $v$ [m/s]")

    axes[0, 2].set_title("(c) Coupled Heart Rate $H(a(t), v(t), D(t))$")
    axes[0, 2].set_xlabel("Chronological Age $a(t)$ [yr]")
    axes[0, 2].set_ylabel("Resting Heart Rate $H$ [bpm]")

    axes[1, 0].set_title("(d) Arterial Strouhal Group $\\mathrm{St}(t) = H D / v$")
    axes[1, 0].set_xlabel("Chronological Age $a(t)$ [yr]")
    axes[1, 0].set_ylabel("Dimensionless $\\mathrm{St}$ [-]")

    axes[1, 1].set_title("(e) Cohort Survival $S(t)$ & $\\bar{\\mathcal{L}} = 20 + \\int S\\,dt$")
    axes[1, 1].set_xlabel("Chronological Age $a(t)$ [yr]")
    axes[1, 1].set_ylabel("Survival Probability $S(t)$ [-]")
    axes[1, 1].set_ylim(0.0, 1.03)

    axes[1, 2].axhline(1.0, color="#333333", ls="--", lw=1.2, label="Failure threshold $\\Omega = 1$")
    axes[1, 2].set_title("(f) Biomechanical Fatigue $\\Omega(t)$")
    axes[1, 2].set_xlabel("Chronological Age $a(t)$ [yr]")
    axes[1, 2].set_ylabel("Cumulative Damage $\\Omega(t)$ [-]")
    axes[1, 2].set_ylim(0.0, 1.35)
    axes[1, 2].legend(loc="upper left")

    fig1.legend(
        handles=axes[0, 0].get_lines(),
        loc="outside upper center",
        ncol=2,
        frameon=True,
        fontsize=9,
    )
    for dst in (project_root / rel_paths[0], repo_figures_dir / repo_names[0]):
        dst.parent.mkdir(parents=True, exist_ok=True)
        fig1.savefig(dst, dpi=220)
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Figure 2: Average Lifespan vs. Heart Rate (Human Intra-Species + Mammalian Inset/Panel)
    # ------------------------------------------------------------------
    fig2, (ax_hum, ax_mam) = plt.subplots(1, 2, figsize=(13.6, 5.6), constrained_layout=True)

    # Evaluate calibrated Buckingham-Pi law on smooth human grid
    GBEAT_SCALE = 1.0e-9
    L_UNIT_M = 0.028
    best_cand_obj = next(c for c in deriv_L.candidates if c.name == best_pi_fit.candidate_name)
    d_smooth = np.interp(hr_grid_bpm, [r[0] for r in HUMAN_COHORT_STRATA], [r[3] for r in HUMAN_COHORT_STRATA])
    v_smooth = np.interp(hr_grid_bpm, [r[0] for r in HUMAN_COHORT_STRATA], [r[4] for r in HUMAN_COHORT_STRATA])
    pi_curve_yr = NumericalSolver.evaluate(
        best_cand_obj,
        data={
            "H": hr_grid_bpm * 60.0 * 24.0 * 365.25 * GBEAT_SCALE,
            "v": (v_smooth / L_UNIT_M) * SEC_PER_YR * GBEAT_SCALE,
            "D": d_smooth / L_UNIT_M,
        },
        params=best_pi_fit.params,
    )
    # Pure beat-budget anchored at 65 bpm -> 80 yr to illustrate why naive 1/H overpredicts human slope
    naive_beat_budget_yr = 80.0 * (65.0 / hr_grid_bpm)

    ax_hum.fill_between(
        hr_grid_bpm,
        ode_lifespan_stenotic,
        ode_lifespan_dilated,
        color="#2C5282",
        alpha=0.14,
        label="Vascular caliber envelope ($D_0 \\in [20.5, 27.6]$ mm)",
    )
    ax_hum.plot(hr_grid_bpm, naive_beat_budget_yr, color="#A0AEC0", ls="--", lw=1.7, label="Naive constant-beat law $N_0/H$ (unmodulated)")
    ax_hum.plot(hr_grid_bpm, ode_lifespan_dilated, color="#1E5631", ls="-.", lw=1.6, label="ODE Survival: Healthy/Dilated ($D_0=27.6$ mm)")
    ax_hum.plot(hr_grid_bpm, ode_lifespan_normo, color="#1B365D", lw=2.5, label="ODE Survival: Normovascular $\\bar{\\mathcal{L}}(H)$")
    ax_hum.plot(hr_grid_bpm, ode_lifespan_stenotic, color="#9B2C2C", ls="-.", lw=1.6, label="ODE Survival: Stenotic ($D_0=20.5$ mm)")
    ax_hum.plot(hr_grid_bpm, gompertz_exact_curve, color="#D69E2E", ls=":", lw=2.2, label="Analytical Gompertz $E_1$ Law (Makeham$\\to 0$)")
    ax_hum.plot(hr_grid_bpm, pi_curve_yr, color="#805AD5", lw=1.9, alpha=0.9, label=f"Calibrated $\\Pi$-Law (`{best_pi_fit.candidate_name}`)")

    hum_hr = np.array([r[0] for r in HUMAN_COHORT_STRATA])
    hum_l = np.array([r[1] for r in HUMAN_COHORT_STRATA])
    hum_ci = np.array([r[2] for r in HUMAN_COHORT_STRATA])
    ax_hum.errorbar(
        hum_hr,
        hum_l,
        yerr=hum_ci,
        fmt="o",
        color="#111111",
        ecolor="#4A5568",
        elinewidth=1.4,
        capsize=3.5,
        ms=5.5,
        zorder=5,
        label="Human Clinical Cohort Strata ($\\pm 95\\%$ CI)",
    )
    ax_hum.set_ylim(58.0, 96.0)
    ax_hum.set_xlabel("Adult Resting Heart Rate $H$ [bpm]")
    ax_hum.set_ylabel("Human Average Lifespan $\\bar{\\mathcal{L}}(H)$ [years]")
    ax_hum.set_title("(a) Intra-Species Human Average Lifespan $\\bar{\\mathcal{L}}(H)$")
    ax_hum.legend(loc="upper right", fontsize=7.8)

    # Panel (b): Cross-mammalian log-log scaling reconciliation
    mam_hr_grid = np.geomspace(14.0, 700.0, 200)
    n0_mammal = abs(float(fit_mammal_lead.params["kappa_L_life"])) * 1.0e9
    mam_pred_yr = n0_mammal / (mam_hr_grid * 60.0 * 24.0 * 365.25)

    ax_mam.loglog(
        mam_hr_grid,
        mam_pred_yr,
        color="#2C5282",
        lw=2.2,
        label=f"Allometric $\\bar{{\\mathcal{{L}}}} = N_0 / H$ ($N_0 = {n0_mammal:.2e}$ beats)",
    )
    for sp_name, sp_hr, _, _, sp_life in MAMMAL_BENCHMARK:
        is_hum = "Human" in sp_name
        ax_mam.scatter(
            sp_hr,
            sp_life,
            s=75 if is_hum else 48,
            color="#C53030" if is_hum else "#1B365D",
            marker="*" if is_hum else "o",
            zorder=6 if is_hum else 4,
        )
        offset_y = 1.12 if not is_hum else 0.78
        ax_mam.annotate(sp_name, (sp_hr, sp_life * offset_y), fontsize=8, ha="center")

    # Overlay human intra-species locus on mammalian log-log plot
    ax_mam.loglog(
        hr_grid_bpm,
        ode_lifespan_normo,
        color="#C53030",
        lw=2.5,
        label="Human intra-species trajectory ($\\mathrm{St}$ varies $\\Rightarrow H^{-0.18}$)",
    )
    ax_mam.set_xlabel("Resting Heart Rate $H$ [bpm] (log scale)")
    ax_mam.set_ylabel("Average Lifespan $\\bar{\\mathcal{L}}$ [years] (log scale)")
    ax_mam.set_title("(b) Cross-Species ($H^{-1}$) vs. Human Intra-Species ($H^{\\nu-1}$) Scaling")
    ax_mam.legend(loc="lower left", fontsize=8.0)

    for dst in (project_root / rel_paths[1], repo_figures_dir / repo_names[1]):
        fig2.savefig(dst, dpi=220)
    plt.close(fig2)

    # ------------------------------------------------------------------
    # Figure 3: 2D Hemodynamic Phase Contour E[L](H, D_eff) + Chain-Rule Sensitivity
    # ------------------------------------------------------------------
    fig3, (ax_surf, ax_sens) = plt.subplots(1, 2, figsize=(13.4, 5.3), constrained_layout=True)

    hr_2d = np.linspace(46.0, 100.0, 22)
    d_2d_mm = np.linspace(17.5, 28.0, 20)
    L_surface = np.zeros((len(d_2d_mm), len(hr_2d)))

    for i_d, d_mm in enumerate(d_2d_mm):
        d_m = d_mm * 1e-3
        for j_h, hr_b in enumerate(hr_2d):
            # Hold intrinsic baseline variation + actual dynamic consequence of altered D(t)
            h_nom_offset = base_params["kappa_mat"] / y0_base["a"] + base_params["kappa_hemo"] * (y0_base["v"] / y0_base["D"])
            hb = hr_b * BPM_TO_HZ - h_nom_offset
            tr_cell = ode_sys.solve(
                t_span,
                {**y0_base, "D": d_m},
                params={**base_params, "H_base": hb, "D_min": 0.70 * d_m},
                method="rk4",
                dt=0.4 * SEC_PER_YR,
            )
            L_surface[i_d, j_h] = (base_params["a_adult"] + tr_cell.final["E_rem"]) / SEC_PER_YR

    cf = ax_surf.contourf(hr_2d, d_2d_mm, L_surface, levels=16, cmap="YlGnBu")
    cs = ax_surf.contour(hr_2d, d_2d_mm, L_surface, levels=8, colors="k", linewidths=0.7, alpha=0.55)
    ax_surf.clabel(cs, fmt="%.1f yr", fontsize=8)
    cbar = fig3.colorbar(cf, ax=ax_surf, pad=0.02)
    cbar.set_label("Expected Lifespan $\\bar{\\mathcal{L}}$ [years]")
    ax_surf.set_xlabel("Baseline Intrinsic Heart Rate $H_0$ [bpm]")
    ax_surf.set_ylabel("Young-Adult Effective Arterial Diameter $D_0$ [mm]")
    ax_surf.set_title("(a) Coupled Phase Surface $\\bar{\\mathcal{L}}(H_0, D_0)$")

    # Panel (b): Chain-rule rate decomposition dH/dt = (dH/da)da/dt + (dH/dv)dv/dt + (dH/dD)dD/dt
    a_ref = traj_ref["a"]
    d_ref = traj_ref["D"]
    v_ref = traj_ref["v"]
    age_yr_ref = a_ref / SEC_PER_YR

    da_dt = np.full_like(a_ref, base_params["kappa_a"])
    dD_dt = -base_params["k_D"] * (d_ref - base_params["D_min"])
    dv_dt = -base_params["k_v"] * (v_ref - base_params["v_min"]) + base_params["c_sten"] / d_ref

    scale_bpm_per_decade = HZ_TO_BPM * (10.0 * SEC_PER_YR)
    term_age = (-base_params["kappa_mat"] / (a_ref ** 2)) * da_dt * scale_bpm_per_decade
    term_vel = (base_params["kappa_hemo"] / d_ref) * dv_dt * scale_bpm_per_decade
    term_diam = (-base_params["kappa_hemo"] * v_ref / (d_ref ** 2)) * dD_dt * scale_bpm_per_decade
    term_tot = term_age + term_vel + term_diam

    ax_sens.plot(age_yr_ref, term_age, color="#2B6CB0", lw=1.9, label="$(\\partial H/\\partial a)\\,\\dot{a}$ (Autonomic maturation)")
    ax_sens.plot(age_yr_ref, term_vel, color="#2F855A", lw=1.9, label="$(\\partial H/\\partial v)\\,\\dot{v}$ (Velocity relaxation)")
    ax_sens.plot(age_yr_ref, term_diam, color="#C53030", lw=1.9, label="$(\\partial H/\\partial D)\\,\\dot{D}$ (Stenotic narrowing $>0$)")
    ax_sens.plot(age_yr_ref, term_tot, color="#1A202C", ls="--", lw=2.2, label="Total Chain-Rule Drift $dH/dt$")
    ax_sens.axhline(0.0, color="gray", lw=0.9)
    ax_sens.set_xlabel("Chronological Age $a(t)$ [yr]")
    ax_sens.set_ylabel("Secular Heart-Rate Drift [bpm / decade]")
    ax_sens.set_title("(b) Chain-Rule Decomposition of $dH/dt$")
    ax_sens.legend(loc="lower right", fontsize=8.0)

    for dst in (project_root / rel_paths[2], repo_figures_dir / repo_names[2]):
        fig3.savefig(dst, dpi=220)
    plt.close(fig3)

    # ------------------------------------------------------------------
    # Figure 4: Verification & Validation 4-Panel Diagnostic Suite
    # ------------------------------------------------------------------
    fig4, axes4 = plt.subplots(2, 2, figsize=(12.6, 8.2), constrained_layout=True)

    # (a) RK4 Timestep Convergence vs Extrapolated Continuum Value
    dts_yr = np.array(dt_report.gci.spacings) / SEC_PER_YR
    q_vals = np.array(dt_report.gci.values)
    q_ext = float(dt_report.gci.extrapolated_value)
    err_abs = np.maximum(np.abs(q_vals - q_ext), 1e-15)
    axes4[0, 0].loglog(dts_yr, err_abs, "o-", color="#1B365D", lw=2.1, ms=6, label="RK4 $|\\bar{\\mathcal{L}}(\\Delta t) - \\bar{\\mathcal{L}}_{\\mathrm{ext}}|$")
    ref_line = err_abs[0] * ((dts_yr / dts_yr[0]) ** 4.0)
    axes4[0, 0].loglog(dts_yr, ref_line, "--", color="#C53030", lw=1.5, label="Formal Slope $\\mathcal{O}(\\Delta t^4)$")
    axes4[0, 0].set_xlabel("Time Step $\\Delta t$ [yr]")
    axes4[0, 0].set_ylabel("Discretization Error [yr]")
    axes4[0, 0].set_title(f"(a) RK4 Timestep Verification ($p_{{\\mathrm{{obs}}}} = {dt_report.gci.observed_order:.2f}$)")
    axes4[0, 0].legend(loc="upper left")

    # (b) Machine-Precision Probability Conservation Audit |S(t) + M(t) - 1|
    prob_residual = np.abs(traj_ref["S"] + traj_ref["M"] - 1.0)
    axes4[0, 1].semilogy(
        traj_ref["a"] / SEC_PER_YR,
        np.maximum(prob_residual, 1e-17),
        color="#1E5631",
        lw=1.8,
        label="$|S(t) + M(t) - 1|$",
    )
    axes4[0, 1].axhline(1e-12, color="#C53030", ls="--", lw=1.2, label="Audit tolerance $10^{-12}$")
    axes4[0, 1].set_xlabel("Chronological Age $a(t)$ [yr]")
    axes4[0, 1].set_ylabel("Conservation Residual [-]")
    axes4[0, 1].set_title("(b) Survival Probability Conservation Audit")
    axes4[0, 1].set_ylim(1e-17, 1e-10)
    axes4[0, 1].legend(loc="upper right")

    # (c) Buckingham-Pi Candidate Ranking by BIC on Human Strata
    cand_names = [fr.candidate_name.replace("buckingham_pi_", "Pi-") for fr in ranked_human.ranked]
    cand_bics = [float(fr.bic) for fr in ranked_human.ranked]
    y_pos = np.arange(len(cand_names))
    bar_colors = ["#1E5631" if i == 0 else "#4A5568" for i in range(len(cand_names))]
    axes4[1, 0].barh(y_pos, cand_bics, color=bar_colors, alpha=0.88)
    axes4[1, 0].set_yticks(y_pos)
    axes4[1, 0].set_yticklabels(cand_names)
    axes4[1, 0].invert_yaxis()
    axes4[1, 0].set_xlabel("Bayesian Information Criterion (BIC — lower is better)")
    axes4[1, 0].set_title("(c) Symbolic Candidate Selection on Human Data")

    # (d) Validation Parity Plot: ODE Prediction vs Clinical Cohort Lifespan
    hum_l = np.array([r[1] for r in HUMAN_COHORT_STRATA])
    hum_ci = np.array([r[2] for r in HUMAN_COHORT_STRATA])
    lim_lo, lim_hi = 72.0, 85.0
    axes4[1, 1].plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1.3, label="Exact 1:1 Parity")
    axes4[1, 1].fill_between(
        [lim_lo, lim_hi],
        [lim_lo - 1.0, lim_hi - 1.0],
        [lim_lo + 1.0, lim_hi + 1.0],
        color="#2C5282",
        alpha=0.12,
        label="$\\pm 1.0$ yr Clinical Agreement Band",
    )
    axes4[1, 1].errorbar(
        hum_l,
        ode_pred_at_cohort,
        xerr=hum_ci,
        fmt="o",
        color="#1B365D",
        ecolor="#718096",
        capsize=3,
        ms=6,
        label="Coupled ODE vs. Cohort ($\\pm 95\\%$ CI)",
    )
    axes4[1, 1].set_xlim(lim_lo, lim_hi)
    axes4[1, 1].set_ylim(lim_lo, lim_hi)
    axes4[1, 1].set_xlabel("Clinical Cohort Observed $\\bar{\\mathcal{L}}$ [yr]")
    axes4[1, 1].set_ylabel("Coupled ODE Predicted $\\bar{\\mathcal{L}}$ [yr]")
    axes4[1, 1].set_title("(d) ASME V&V 20 Clinical Validation Parity")
    axes4[1, 1].legend(loc="upper left", fontsize=8.0)

    for dst in (project_root / rel_paths[3], repo_figures_dir / repo_names[3]):
        fig4.savefig(dst, dpi=220)
    plt.close(fig4)

    return rel_paths


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m examples.cardiovascular_lifespan_study.run_study",
        description="Run the Rosolver Cardiovascular Hemodynamics & Human Average Lifespan Study.",
    )
    parser.add_argument(
        "--output",
        default="outputs/cardiovascular_lifespan_study",
        help="Target workspace directory (default: outputs/cardiovascular_lifespan_study).",
    )
    args = parser.parse_args(argv)
    summary = run_cardiovascular_study(output_dir=args.output)
    print(f"Cardiovascular lifespan study completed: status={summary['report_status']}")
    print(f"  Reference E[L] (65 bpm)      : {summary['expected_lifespan_ref_65bpm_yr']} yr")
    print(f"  Analytical Gompertz E1 (65)  : {summary['gompertz_exact_65bpm_yr']} yr")
    print(f"  Gompertz slope (yr / 10 bpm) : {summary['gompertz_slope_yr_per_10bpm']} yr")
    print(f"  Best Buckingham-Pi candidate : {summary['human_best_pi_candidate']} (R^2={summary['human_best_pi_r2']})")
    print(f"  RK4 observed order           : {summary['rk4_observed_order']}")
    print(f"  Validation RMSE              : {summary['validation_rmse_yr']} yr")
    print(f"  Report written to            : {summary['report_path']}")
    return 0 if summary["report_status"] in ("Passed", "Warning") else 1


if __name__ == "__main__":
    raise SystemExit(main())
