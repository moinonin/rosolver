"""Deterministic Scientific Claim Audit & Multi-Round Peer-Review Engine (Sprint 10 / S10.5).

Orchestrates dimensional, scale-balance, discretization error-budget, conservation,
homogeneity, and empirical validation checks over a SHA-256 `Project` workspace,
producing immutable `ReviewRoundRecord` snapshots and `ClarificationRequest` queues.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..results import VerificationStatus, standardize_result
from ..validation import ValidationStudy
from ..verify import audit_ode_invariants
from .budget import audit_claimed_accuracy
from .clarifications import ClarificationRequest, generate_clarification_requests
from .rebuttal import (
    RebuttalSubmission,
    ReviewRoundRecord,
    VerdictTransition,
    resolve_confined_path,
    validate_rebuttal_dataset_csv,
)
from .scales import (
    check_equation_scales,
    check_homogeneity_elasticities,
    diagnose_normalization_shortcuts,
)
from .schema import (
    ClaimKind,
    ClaimManifest,
    ClaimSpec,
    ClaimVerdict,
    ClaimVerdictKind,
    TwoAxisAssessment,
)

if TYPE_CHECKING:
    from ..project import Project


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_csv_columns(path: Path) -> Dict[str, Any]:
    from ..data import load_tabular_csv

    return load_tabular_csv(path, missing_policy="raise", allow_string_columns=True)


def _numeric_col(cols: Mapping[str, Any], name: str) -> np.ndarray:
    arr = cols[name]
    if isinstance(arr, np.ndarray):
        return arr
    return np.asarray([float(v) for v in arr if str(v).strip() != ""], dtype=float)


class ClaimAuditEngine:
    """Stateless orchestrator for deterministic claim auditing on a Rosolver `Project`."""

    @staticmethod
    def _claims_dir(project: "Project") -> Path:
        d = project.root / "claims"
        (d / "rounds").mkdir(parents=True, exist_ok=True)
        return d

    @classmethod
    def register_manifest(cls, project: "Project", manifest: ClaimManifest) -> Path:
        """Persist a validated `ClaimManifest` into `<project>/claims/manifest.yaml`."""
        cdir = cls._claims_dir(project)
        dest = cdir / "manifest.yaml"
        manifest.save(dest)
        project._append_log(
            f"Registered ClaimManifest '{manifest.study_title}' ({len(manifest.claims)} claims, hash={manifest.content_hash()[:12]})"
        )
        return dest

    @classmethod
    def load_manifest(cls, project: "Project") -> ClaimManifest:
        dest = cls._claims_dir(project) / "manifest.yaml"
        if not dest.exists():
            raise FileNotFoundError("No claims/manifest.yaml registered in project.")
        return ClaimManifest.from_file(dest)

    @classmethod
    def list_rounds(cls, project: "Project") -> List[ReviewRoundRecord]:
        rdir = cls._claims_dir(project) / "rounds"
        out: List[ReviewRoundRecord] = []
        for fp in sorted(rdir.glob("round_*.json")):
            out.append(ReviewRoundRecord.from_dict(json.loads(fp.read_text(encoding="utf-8"))))
        return out

    @classmethod
    def load_latest_round(cls, project: "Project") -> Optional[ReviewRoundRecord]:
        rounds = cls.list_rounds(project)
        return rounds[-1] if rounds else None

    @classmethod
    def load_open_clarifications(cls, project: "Project") -> List[ClarificationRequest]:
        fp = cls._claims_dir(project) / "clarifications_open.json"
        if not fp.exists():
            return []
        raw = json.loads(fp.read_text(encoding="utf-8"))
        return [ClarificationRequest.from_dict(item) for item in raw]

    @classmethod
    def _evaluate_claim(
        cls,
        project: "Project",
        claim: ClaimSpec,
        *,
        post_hoc_flag: bool = False,
    ) -> ClaimVerdict:
        run_ids: List[str] = []
        metrics: Dict[str, Any] = {}
        triggers: List[str] = []
        warnings: List[str] = []
        mech_notes: List[str] = []
        emp_notes: List[str] = []
        mech_status = VerificationStatus.NOT_ASSESSED
        emp_status = VerificationStatus.NOT_ASSESSED
        scheme_dep = False

        # ------------------------------------------------------------------
        # 1. EQUATION_LAW: Dimensional normalization + Scale-balance audit
        # ------------------------------------------------------------------
        if claim.kind == ClaimKind.EQUATION_LAW:
            eq_statuses: List[VerificationStatus] = []
            for eq in claim.equations:
                if claim.variables_units:
                    norm_rep = diagnose_normalization_shortcuts(eq, claim.variables_units)
                    if not norm_rep.consistent:
                        eq_statuses.append(VerificationStatus.FAILED)
                        metrics["normalization_hypotheses"] = [h.to_dict() for h in norm_rep.hypotheses]
                        for h in norm_rep.hypotheses:
                            if h.issue_kind not in triggers:
                                triggers.append(h.issue_kind)
                            mech_notes.append(h.explanation)
                        rec = project.record_run(
                            "claim_normalization_audit",
                            standardize_result(norm_rep, kind="claim_normalization_audit"),
                            configuration={"claim_id": claim.id, "equation": eq},
                        )
                        run_ids.append(rec.id)
                        continue

                if claim.variables_units and claim.operating_envelope:
                    sc_rep = check_equation_scales(
                        eq,
                        claim.variables_units,
                        claim.operating_envelope,
                        characteristic_scales=claim.characteristic_scales,
                        scale_threshold=float(claim.metadata.get("scale_threshold", 100.0)),
                    )
                    eq_statuses.append(sc_rep.status)
                    metrics["dominant_balance_ratio"] = sc_rep.dominant_balance_ratio
                    metrics["log10_gap"] = round(sc_rep.log10_gap, 4) if np.isfinite(sc_rep.log10_gap) else sc_rep.log10_gap
                    metrics["diagnostic_code"] = sc_rep.diagnostic_code
                    if sc_rep.status != VerificationStatus.PASSED:
                        triggers.append(sc_rep.diagnostic_code)
                        warnings.extend(sc_rep.warnings)
                        mech_notes.extend(sc_rep.warnings)
                    else:
                        mech_notes.append(f"Equation '{eq}' is dimensionally and scale balanced.")
                    rec = project.record_run(
                        "claim_scale_audit",
                        standardize_result(sc_rep, kind="claim_scale_audit"),
                        configuration={"claim_id": claim.id, "equation": eq},
                    )
                    run_ids.append(rec.id)
                elif claim.variables_units:
                    norm_rep = diagnose_normalization_shortcuts(eq, claim.variables_units)
                    eq_statuses.append(norm_rep.status)
                    rec = project.record_run(
                        "claim_normalization_audit",
                        standardize_result(norm_rep, kind="claim_normalization_audit"),
                        configuration={"claim_id": claim.id, "equation": eq},
                    )
                    run_ids.append(rec.id)

            if VerificationStatus.FAILED in eq_statuses:
                mech_status = VerificationStatus.FAILED
            elif VerificationStatus.WARNING in eq_statuses:
                mech_status = VerificationStatus.WARNING
            elif eq_statuses and all(s == VerificationStatus.PASSED for s in eq_statuses):
                mech_status = VerificationStatus.PASSED

        # ------------------------------------------------------------------
        # 2. DIMENSIONAL_SCALING: Euler homogeneity sum rule + Pi-excitation
        # ------------------------------------------------------------------
        elif claim.kind == ClaimKind.DIMENSIONAL_SCALING:
            if claim.elasticities and claim.target_symbol and claim.target_symbol in claim.variables_units:
                target_u = claim.variables_units[claim.target_symbol]
                inp_u = {k: v for k, v in claim.variables_units.items() if k != claim.target_symbol}
                data_map: Optional[Dict[str, np.ndarray]] = None
                if claim.expected_dataset:
                    try:
                        ds_path = project.get_dataset_path(claim.expected_dataset, verify_hash=True)
                        cols = _load_csv_columns(ds_path)
                        data_map = {k: _numeric_col(cols, k) for k in inp_u if k in cols}
                    except Exception:
                        data_map = None

                hom_rep = check_homogeneity_elasticities(
                    target_u,
                    inp_u,
                    claim.elasticities,
                    tol=float(claim.metadata.get("euler_tol", 0.05)),
                    data=data_map,
                )
                mech_status = hom_rep.status
                metrics["max_abs_euler_residual"] = hom_rep.max_abs_residual
                metrics["zero_intercept_admissible"] = hom_rep.zero_intercept_admissible
                metrics["pi_collinear"] = hom_rep.pi_collinear
                if not hom_rep.zero_intercept_admissible:
                    triggers.append("euler_homogeneity_violation")
                if hom_rep.pi_collinear:
                    triggers.append("collinear_pi_design")
                warnings.extend(hom_rep.warnings)
                mech_notes.extend(hom_rep.warnings or ["Euler dimensional elasticity sum rule satisfied."])
                rec = project.record_run(
                    "claim_scaling_audit",
                    standardize_result(hom_rep, kind="claim_scaling_audit"),
                    configuration={"claim_id": claim.id},
                )
                run_ids.append(rec.id)
            elif claim.equations and claim.variables_units:
                norm_rep = diagnose_normalization_shortcuts(claim.equations[0], claim.variables_units)
                mech_status = norm_rep.status
                if not norm_rep.consistent:
                    metrics["normalization_hypotheses"] = [h.to_dict() for h in norm_rep.hypotheses]
                    triggers.extend([h.issue_kind for h in norm_rep.hypotheses])
                rec = project.record_run(
                    "claim_normalization_audit",
                    standardize_result(norm_rep, kind="claim_normalization_audit"),
                    configuration={"claim_id": claim.id},
                )
                run_ids.append(rec.id)

        # ------------------------------------------------------------------
        # 3. NUMERICAL_ACCURACY: Error-budget & undeclared scheme audit
        # ------------------------------------------------------------------
        elif claim.kind == ClaimKind.NUMERICAL_ACCURACY:
            model = project.get_model(name=claim.model_name, verify_hash=True)
            dspec = claim.discretization_spec
            t_end = float(dspec.get("t_end", 1.0))
            n_steps = dspec.get("n_steps")
            dt_paper = float(dspec["dt"]) if "dt" in dspec else (t_end / float(n_steps) if n_steps else t_end / 10.0)
            declared_scheme = dspec.get("scheme")
            claimed_err = float(claim.claimed_metrics.get("rel_error", 0.01))
            cand_schemes = tuple(dspec.get("candidate_schemes") or ("euler", "cn", "rk4"))

            eb_rep = audit_claimed_accuracy(
                model,
                claimed_rel_error=claimed_err,
                dt_paper=dt_paper,
                declared_scheme=declared_scheme,
                t_span=(0.0, t_end),
                qoi_field=claim.qoi_field,
                candidate_schemes=cand_schemes,
            )
            mech_status = eb_rep.status
            scheme_dep = eb_rep.scheme_dependent
            metrics["claimed_rel_error_pct"] = round(claimed_err * 100.0, 4)
            metrics["discretization_error_paper_pct"] = round(eb_rep.discretization_error_paper * 100.0, 6)
            metrics["budget_ratio"] = round(eb_rep.budget_ratio, 4)
            metrics["scheme_errors"] = {k: round(v * 100.0, 6) for k, v in eb_rep.scheme_errors.items()}
            if eb_rep.gci_observed_order is not None:
                metrics["gci_observed_order"] = round(eb_rep.gci_observed_order, 4)
            if eb_rep.status != VerificationStatus.PASSED:
                triggers.append(eb_rep.diagnostic_code)
                warnings.extend(eb_rep.warnings)
                mech_notes.extend(eb_rep.warnings)
            else:
                mech_notes.append(
                    f"Discretization error ({eb_rep.discretization_error_paper * 100:.4g}%) is well within "
                    f"claimed accuracy ({claimed_err * 100:.4g}%)."
                )
            rec = project.record_run(
                "claim_budget_audit",
                standardize_result(eb_rep, kind="claim_budget_audit"),
                model=model,
                configuration={"claim_id": claim.id, "dt_paper": dt_paper, "declared_scheme": declared_scheme},
            )
            run_ids.append(rec.id)

        # ------------------------------------------------------------------
        # 4. CONSERVATION_INVARIANT: Symbolic + discrete ODE invariant check
        # ------------------------------------------------------------------
        elif claim.kind == ClaimKind.CONSERVATION_INVARIANT:
            model = project.get_model(name=claim.model_name, verify_hash=True)
            ode_sys = model.to_ode_system()
            t_end = float(claim.discretization_spec.get("t_end", 10.0))
            dt_v = float(claim.discretization_spec.get("dt", t_end / 100.0))
            p_m = model.parameter_values()
            y0_m = model.resolved_ode_initial_conditions(None, ode_sys.states, params=p_m)
            traj = ode_sys.solve((0.0, t_end), y0_m, params=p_m, method="rk4", dt=dt_v)
            invs = {f"inv_{i + 1}": expr for i, expr in enumerate(claim.equations)}
            inv_rep = audit_ode_invariants(
                ode_sys, traj, invariants=invs, params=p_m, tol=float(claim.metadata.get("tol", 1e-10))
            )
            mech_status = inv_rep.status
            metrics["max_rel_residuals"] = {k: e.max_rel_residual for k, e in inv_rep.entries.items()}
            if not inv_rep.passed:
                triggers.append("conservation_failed")
            rec = project.record_run(
                "conservation",
                standardize_result(inv_rep, kind="conservation"),
                model=model,
                configuration={"claim_id": claim.id, "invariants": invs},
            )
            run_ids.append(rec.id)

        # ------------------------------------------------------------------
        # 5. EMPIRICAL_COMPARISON / QUALITATIVE_OBSERVATION / PARAMETER_CALIBRATION
        # ------------------------------------------------------------------
        else:
            # Optional mechanistic pre-check (e.g., Peclet number / scale check)
            if claim.equations and claim.variables_units and claim.operating_envelope:
                sc_rep = check_equation_scales(
                    claim.equations[0], claim.variables_units, claim.operating_envelope
                )
                mech_status = sc_rep.status
                rec = project.record_run(
                    "claim_scale_audit",
                    standardize_result(sc_rep, kind="claim_scale_audit"),
                    configuration={"claim_id": claim.id},
                )
                run_ids.append(rec.id)
            elif claim.metadata.get("mechanistic_plausible") is True:
                if bool(claim.requires_empirical):
                    mech_status = VerificationStatus.PASSED
                    mech_notes.append(
                        str(claim.metadata.get("mechanistic_note", "Mechanistic plausibility established from transport theory."))
                    )
                else:
                    mech_status = VerificationStatus.INCONCLUSIVE
                    warnings.append(
                        "Anti-override guard: metadata['mechanistic_plausible'] cannot grant Supported without equations or empirical validation."
                    )

            # Empirical dataset check
            ds_name = claim.expected_dataset
            ds_exists = ds_name is not None and any(d.name == ds_name for d in project.datasets)
            if not ds_exists:
                emp_status = VerificationStatus.NOT_ASSESSED
                triggers.append("missing_empirical_dataset")
                metrics["required_observables"] = list(claim.required_observables)
                emp_notes.append(
                    f"No experimental dataset registered for claim '{claim.id}'; qualitative/unmeasured claim requires replicate lab data."
                )
            else:
                assert ds_name is not None
                ds_path = project.get_dataset_path(ds_name, verify_hash=True)
                cols = _load_csv_columns(ds_path)
                obs_list = list(claim.required_observables) or [
                    c for c in cols if not c.startswith(("u_", "sigma_", "sim_", "sample", "source", "id"))
                ]
                primary_obs = obs_list[0]
                exp_vals = _numeric_col(cols, primary_obs)
                u_col_name = f"u_{primary_obs}" if f"u_{primary_obs}" in cols else (
                    f"sigma_{primary_obs}" if f"sigma_{primary_obs}" in cols else None
                )
                u_exp = _numeric_col(cols, u_col_name) if u_col_name else None

                if u_exp is None:
                    emp_status = VerificationStatus.WARNING
                    triggers.append("missing_experimental_uncertainty")
                    emp_notes.append(f"Dataset '{ds_name}' lacks uncertainty column 'u_{primary_obs}'.")

                # Evaluate against claimed threshold or simulated column
                sim_col = f"sim_{primary_obs}"
                if sim_col in cols:
                    sim_vals = _numeric_col(cols, sim_col)
                    val_rep = ValidationStudy.compare(
                        simulation=sim_vals,
                        experiment=exp_vals,
                        field=primary_obs,
                        u_num=float(claim.metadata.get("u_num", 0.0)),
                        u_exp=u_exp,
                        coverage_factor=float(claim.metadata.get("coverage_factor", 2.0)),
                        study_name=f"Claim {claim.id} Empirical Validation",
                    )
                    emp_status = val_rep.status
                    metrics["validation_rmse"] = val_rep.rmse
                    metrics["max_normalized_error"] = val_rep.max_normalized_error
                    if val_rep.status == VerificationStatus.FAILED:
                        triggers.append("validation_failed")
                    rec = project.record_run(
                        "validation",
                        standardize_result(val_rep, kind="validation"),
                        configuration={"claim_id": claim.id, "dataset": ds_name},
                    )
                    run_ids.append(rec.id)
                else:
                    # Threshold check on replicate measurements (e.g., min_solute_rejection_pct or max_k_eff)
                    n_rep = int(len(exp_vals))
                    mean_val = float(np.mean(exp_vals))
                    if u_exp is not None:
                        se_val = float(np.sqrt(np.mean(u_exp ** 2)))
                    elif n_rep >= 2:
                        se_val = float(np.std(exp_vals, ddof=1) / np.sqrt(n_rep))
                    else:
                        se_val = 0.0
                        if "missing_experimental_uncertainty" not in triggers:
                            triggers.append("missing_experimental_uncertainty")
                        warnings.append(
                            f"Single replicate (n=1) in '{ds_name}' without uncertainty column 'u_{primary_obs}'."
                        )
                    if not np.isfinite(se_val):
                        se_val = 0.0
                    metrics[f"mean_{primary_obs}"] = round(mean_val, 5)
                    metrics[f"mean_u_{primary_obs}"] = round(se_val, 5)
                    metrics["n_replicates"] = n_rep

                    passed_thresh = True
                    if f"min_{primary_obs}" in claim.claimed_metrics:
                        thr = float(claim.claimed_metrics[f"min_{primary_obs}"])
                        passed_thresh = mean_val >= thr
                        metrics["threshold_min"] = thr
                    if f"max_{primary_obs}" in claim.claimed_metrics:
                        thr = float(claim.claimed_metrics[f"max_{primary_obs}"])
                        passed_thresh = passed_thresh and (mean_val <= thr)
                        metrics["threshold_max"] = thr

                    if not passed_thresh:
                        emp_status = VerificationStatus.FAILED
                        triggers.append("empirical_threshold_falsified")
                        emp_notes.append(
                            f"Replicate mean {primary_obs}={mean_val:.4g} (+/- {se_val:.3g}) violates claimed threshold."
                        )
                    elif u_exp is None or n_rep < 2:
                        emp_status = VerificationStatus.WARNING
                    else:
                        emp_status = VerificationStatus.PASSED
                        emp_notes.append(
                            f"Replicate mean {primary_obs}={mean_val:.4g} (+/- {se_val:.3g}, n={n_rep}) satisfies claimed threshold."
                        )

                    # Record deterministic validation run in ledger
                    safe_u = u_exp if u_exp is not None else np.full_like(exp_vals, float(np.fmax(se_val, 1e-12)))
                    val_ref = ValidationStudy.compare(
                        simulation=np.full_like(exp_vals, mean_val),
                        experiment=exp_vals,
                        field=primary_obs,
                        u_num=0.0,
                        u_exp=safe_u,
                        coverage_factor=2.0,
                        study_name=f"Claim {claim.id} Replicate Assay Check",
                    )
                    rec = project.record_run(
                        "validation",
                        standardize_result(val_ref, kind="validation"),
                        configuration={"claim_id": claim.id, "dataset": ds_name, "passed_threshold": passed_thresh},
                    )
                    run_ids.append(rec.id)

        # Anti-p-hacking downgrade when rebuttal modified parameters without blind validation
        if post_hoc_flag and emp_status == VerificationStatus.PASSED:
            emp_status = VerificationStatus.WARNING
            warnings.append("post_hoc_recalibration: model parameters modified during rebuttal without blind holdout.")
            triggers.append("post_hoc_recalibration")

        assessment = TwoAxisAssessment(
            mechanistic_status=mech_status,
            empirical_status=emp_status,
            requires_empirical=bool(claim.requires_empirical),
            scheme_dependent=scheme_dep,
            mechanistic_notes=mech_notes,
            empirical_notes=emp_notes,
        )

        # Ensure every Supported verdict is anchored to a ledger run
        if assessment.overall_verdict == ClaimVerdictKind.SUPPORTED and not run_ids:
            rec = project.record_run(
                "claim_audit",
                standardize_result({"claim_id": claim.id, "status": "Passed"}, kind="claim_audit"),
                configuration={"claim_id": claim.id},
            )
            run_ids.append(rec.id)

        summary_txt = "; ".join(mech_notes + emp_notes) or f"Verdict: {assessment.overall_verdict.value}"
        verdict = ClaimVerdict(
            claim_id=claim.id,
            title=claim.title,
            kind=claim.kind,
            assessment=assessment,
            evidence_run_ids=run_ids,
            metrics=metrics,
            trigger_codes=list(dict.fromkeys(triggers)),
            warnings=warnings,
            summary_text=summary_txt,
            post_hoc_recalibration=post_hoc_flag,
        )
        verdict.verify_against_ledger(project.ledger())
        return verdict

    @classmethod
    def run_round(
        cls,
        project: "Project",
        manifest: Optional[ClaimManifest] = None,
        *,
        parent_round: Optional[ReviewRoundRecord] = None,
        rebuttal_hashes: Optional[Mapping[str, str]] = None,
        post_hoc_claims: Optional[Mapping[str, bool]] = None,
        resolved_by_claim: Optional[Mapping[str, List[str]]] = None,
    ) -> ReviewRoundRecord:
        """Execute a deterministic verification round across all claims in `manifest`."""
        if manifest is not None:
            cls.register_manifest(project, manifest)
            active_manifest = manifest
        else:
            active_manifest = cls.load_manifest(project)

        existing_rounds = cls.list_rounds(project)
        round_idx = (existing_rounds[-1].round_index + 1) if existing_rounds else 1
        round_id = f"R{round_idx}"
        parent = parent_round or (existing_rounds[-1] if existing_rounds else None)

        ph_map = post_hoc_claims or {}
        verdicts: List[ClaimVerdict] = [
            cls._evaluate_claim(project, claim, post_hoc_flag=bool(ph_map.get(claim.id, False)))
            for claim in active_manifest.claims
        ]
        obs_map = {c.id: c.required_observables for c in active_manifest.claims}
        open_requests = generate_clarification_requests(
            verdicts, round_index=round_idx, observables_by_claim=obs_map
        )

        transitions: List[VerdictTransition] = []
        if parent is not None:
            prev_by_id = {v.claim_id: v for v in parent.verdicts}
            open_by_claim: Dict[str, List[str]] = {}
            for req in open_requests:
                open_by_claim.setdefault(req.claim_id, []).append(req.request_id)
            res_map = resolved_by_claim or {}
            for v in verdicts:
                prev_v = prev_by_id.get(v.claim_id)
                prev_str = prev_v.overall_verdict.value if prev_v is not None else ClaimVerdictKind.NOT_ASSESSED.value
                transitions.append(
                    VerdictTransition(
                        claim_id=v.claim_id,
                        prev_verdict=prev_str,
                        new_verdict=v.overall_verdict.value,
                        resolved_request_ids=list(res_map.get(v.claim_id, [])),
                        remaining_request_ids=open_by_claim.get(v.claim_id, []),
                        post_hoc_recalibration=v.post_hoc_recalibration,
                        notes=f"{prev_str} -> {v.overall_verdict.value}",
                    )
                )

        record = ReviewRoundRecord(
            round_index=round_idx,
            round_id=round_id,
            parent_round_id=parent.round_id if parent is not None else None,
            manifest_hash=active_manifest.content_hash(),
            timestamp=_utc_now(),
            verdicts=verdicts,
            open_requests=open_requests,
            transitions=transitions,
            rebuttal_hashes=dict(rebuttal_hashes or {}),
        )

        cdir = cls._claims_dir(project)
        round_file = cdir / "rounds" / f"round_{round_idx:03d}.json"
        round_file.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
        (cdir / "clarifications_open.json").write_text(
            json.dumps([r.to_dict() for r in open_requests], indent=2), encoding="utf-8"
        )
        project._append_log(
            f"Completed claim audit round {round_id}: counts={record.summary_counts}, open_requests={len(open_requests)}"
        )
        return record

    @classmethod
    def submit_rebuttal(
        cls,
        project: "Project",
        submissions: Sequence[RebuttalSubmission],
    ) -> ReviewRoundRecord:
        """Ingest author rebuttal submissions, lock SHA-256 hashes, and run Round k+1."""
        parent = cls.load_latest_round(project)
        if parent is None:
            raise RuntimeError("No prior claim audit round exists; run audit_claims() before submitting a rebuttal.")

        manifest = cls.load_manifest(project)
        open_by_id = {r.request_id: r for r in parent.open_requests}
        rebuttal_hashes: Dict[str, str] = {}
        post_hoc_claims: Dict[str, bool] = {}
        resolved_by_claim: Dict[str, List[str]] = {}

        for sub in submissions:
            if sub.request_id not in open_by_id:
                raise KeyError(
                    f"Unknown or already-resolved request_id '{sub.request_id}' "
                    f"(open: {sorted(open_by_id)})."
                )
            req = open_by_id[sub.request_id]
            claim = manifest.get_claim(sub.claim_id)
            if sub.modifies_model_parameters:
                post_hoc_claims[sub.claim_id] = True

            # 1. Dataset upload rebuttal
            if sub.dataset_path is not None:
                confined = resolve_confined_path(project.root, sub.dataset_path)
                val_errors = validate_rebuttal_dataset_csv(confined, req.required_artifact)
                if val_errors:
                    raise ValueError(
                        f"Rebuttal '{sub.request_id}' dataset failed contract validation: {'; '.join(val_errors)}"
                    )
                ds_name = sub.dataset_name or claim.expected_dataset or confined.stem
                ds_ref = project.add_dataset(
                    confined, name=ds_name, category="experimental", copy=True, description=sub.author_note
                )
                claim.expected_dataset = ds_ref.name
                rebuttal_hashes[sub.request_id] = ds_ref.sha256

            # 2. Solver metadata / scheme declaration rebuttal
            if sub.declared_scheme is not None:
                claim.discretization_spec["scheme"] = sub.declared_scheme
            if sub.refinement_dts:
                claim.discretization_spec["refinement_dts"] = list(sub.refinement_dts)

            # 3. Constitutive equation / envelope revision rebuttal
            if sub.revised_equation is not None:
                if [sub.revised_equation] != list(claim.equations):
                    post_hoc_claims[sub.claim_id] = True
                claim.equations = [sub.revised_equation]
            if sub.revised_units:
                claim.variables_units.update(sub.revised_units)
            if sub.revised_envelope:
                for k, bounds in sub.revised_envelope.items():
                    lo, hi = float(bounds[0]), float(bounds[1])
                    claim.operating_envelope[k] = (min(lo, hi), max(lo, hi))

            resolved_by_claim.setdefault(sub.claim_id, []).append(sub.request_id)

        # Persist updated manifest & run Round k+1
        cls.register_manifest(project, manifest)
        return cls.run_round(
            project,
            manifest,
            parent_round=parent,
            rebuttal_hashes=rebuttal_hashes,
            post_hoc_claims=post_hoc_claims,
            resolved_by_claim=resolved_by_claim,
        )
