"""Sprint 4 tests: persistent Project workspace, integrity hashes & append-only ledger (P1.5)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from rosolver import (
    EquationSpec,
    Model,
    ParameterSpec,
    Project,
    ProjectError,
    ProvenanceMismatchError,
    VariableSpec,
    VerificationStatus,
    capture_environment,
)


def _heat_model(alpha_val: float = 0.05) -> Model:
    return Model(
        name="heat_transfer_1d",
        version="0.1",
        description="1D transient heat conduction",
        domain="thermal_engineering",
        independent_variables=[
            VariableSpec("x", role="independent", unit="m"),
            VariableSpec("t", role="independent", unit="s"),
        ],
        dependent_variables=[VariableSpec("temperature", symbol="T", role="dependent", unit="K")],
        parameters=[ParameterSpec("thermal_diffusivity", symbol="alpha", value=alpha_val, unit="m^2/s")],
        equations=[EquationSpec(id="heat_eq", expression="dT/dt = alpha * d2T/dx2", type="pde")],
    )


def _ke_model() -> Model:
    return Model(
        name="kinetic_energy",
        version="0.1",
        independent_variables=[
            VariableSpec("m", role="independent", unit="kg"),
            VariableSpec("v", role="independent", unit="m/s"),
        ],
        dependent_variables=[VariableSpec("K", role="dependent", unit="J")],
        parameters=[ParameterSpec("kappa_K", value=None, unit="1", status="unknown", bounds=(0.0, 5.0))],
        equations=[EquationSpec(id="ke_law", expression="K = kappa_K * m * v**2", type="constitutive")],
        dependencies=["K = fn(m, v)"],
    )


class TestProjectLifecycleAndProvenance:
    def test_workspace_creation_manifest_and_environment_provenance(self, tmp_path: Path) -> None:
        proj_dir = tmp_path / "study_a"
        proj = Project.create(proj_dir, name="Thermal Study", description="Sprint 4 test")
        assert (proj_dir / "project.yaml").exists()
        for sub in (
            "models",
            "data/experimental",
            "data/reference",
            "data/processed",
            "calibration/results",
            "simulations/results",
            "verification/mms",
            "verification/gci",
            "verification/conservation",
            "verification/timestep",
            "validation/results",
            "reports/figures",
            "reports/tables",
            "logs/ledger.jsonl",
            "logs/execution.log",
        ):
            assert (proj_dir / sub).exists()

        env = capture_environment()
        assert "python" in env and "numpy" in env["dependencies"]
        assert "PATH" not in env and "API_KEY" not in json.dumps(env)

        with pytest.raises(ProjectError, match="already exists"):
            Project.create(proj_dir)

        reopened = Project.open(proj_dir)
        assert reopened.project_id == proj.project_id
        assert "Thermal Study" in reopened.summary()

    def test_model_versioning_and_dataset_hash_tamper_detection(self, tmp_path: Path) -> None:
        proj = Project.create(tmp_path / "proj")
        m1 = _heat_model(0.05)
        m2 = _heat_model(0.08)  # same name & version, different parameter value -> distinguishable file

        r1 = proj.add_model(m1)
        r1_dup = proj.add_model(m1)
        r2 = proj.add_model(m2)
        assert r1.content_hash == r1_dup.content_hash
        assert r1.content_hash != r2.content_hash
        assert r1.path != r2.path
        assert len(proj.models) == 2

        loaded_1 = proj.get_model(content_hash=r1.content_hash)
        loaded_2 = proj.get_model("heat_transfer_1d")
        assert loaded_1.parameter_values()["alpha"] == 0.05
        assert loaded_2.parameter_values()["alpha"] == 0.08

        # Register external CSV dataset and verify tamper / missing checks
        csv_src = tmp_path / "exp_raw.csv"
        csv_src.write_text("m,v,K\n1,2,2\n2,3,9\n", encoding="utf-8")
        ds_ref = proj.add_dataset(csv_src, name="ke_exp", category="experimental")
        assert proj.verify_integrity()["ok"] is True

        # Tamper with copied dataset on disk
        copied = proj.get_dataset_path("ke_exp")
        copied.write_text("m,v,K\n1,2,999\n2,3,9\n", encoding="utf-8")
        report = proj.verify_integrity(strict=False)
        assert not report["ok"]
        assert report["modified"][0]["name"] == "ke_exp"
        with pytest.raises(ProvenanceMismatchError):
            proj.verify_integrity(strict=True)

        # Delete dataset file -> missing detected
        copied.unlink()
        rep_miss = proj.verify_integrity()
        assert not rep_miss["ok"]
        assert rep_miss["missing"][0]["name"] == "ke_exp"
        assert ds_ref.name == "ke_exp"


class TestAppendOnlyLedgerAndReplay:
    def test_relocatable_project_runs_and_exact_replay(self, tmp_path: Path) -> None:
        orig_dir = tmp_path / "orig_proj"
        proj = Project.create(orig_dir, name="relocatable")
        proj.add_model(_heat_model(0.05))
        proj.add_model(_ke_model())

        # Create & register CSV dataset
        csv_src = tmp_path / "ke_obs.csv"
        rng = np.random.default_rng(77)
        ms = rng.uniform(1.0, 5.0, 20)
        vs = rng.uniform(2.0, 8.0, 20)
        ks = 0.5 * ms * (vs ** 2)
        with csv_src.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["m", "v", "K"])
            for m_v, v_v, k_v in zip(ms, vs, ks):
                w.writerow([m_v, v_v, k_v])
        proj.add_dataset(csv_src, name="ke_obs", category="experimental")

        # 1. Symbolic run
        sym_res = proj.run_symbolic("kinetic_energy", target="K", zero_initials=True)
        assert sym_res.status == VerificationStatus.PASSED

        # 2. PDE simulation run
        sim_res = proj.run_simulation(
            "heat_transfer_1d",
            t_span=(0.0, 0.2),
            initial_conditions={"T": "sin(pi * x)"},
            boundary_conditions={"left": ("dirichlet", 0.0), "right": ("dirichlet", 0.0)},
            method="BDF",
            domain=(0.0, 1.0),
            n_cells=40,
            n_out=5,
            seed=42,
        )
        assert sim_res.status == VerificationStatus.PASSED

        # 3. CSV-backed calibration run
        cal_res = proj.run_calibration("kinetic_energy", dataset="ke_obs", target="K")
        assert cal_res.status == VerificationStatus.PASSED
        assert np.isclose(cal_res.raw.params["kappa_K"], 0.5, atol=1e-6)

        runs_before_move = proj.ledger()
        assert len(runs_before_move) == 3
        assert [r.type for r in runs_before_move] == ["symbolic_derivation", "pde_simulation", "calibration"]

        # Verify no hardcoded absolute paths in project.yaml or ledger.jsonl
        manifest_txt = (orig_dir / "project.yaml").read_text(encoding="utf-8")
        ledger_txt = (orig_dir / "logs" / "ledger.jsonl").read_text(encoding="utf-8")
        assert str(orig_dir) not in manifest_txt
        assert str(orig_dir) not in ledger_txt

        # Move project folder to a completely different path and reopen
        moved_dir = tmp_path / "moved_location" / "relocatable_copy"
        moved_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(orig_dir), str(moved_dir))

        reopened = Project.open(moved_dir)
        assert reopened.verify_integrity()["ok"] is True

        # Replay simulation and calibration from append-only ledger
        sim_replay = reopened.replay_run(runs_before_move[1].id)
        cal_replay = reopened.replay_run(runs_before_move[2].id)

        assert np.allclose(sim_replay.raw.final("T"), sim_res.raw.final("T"), atol=1e-12)
        assert np.isclose(cal_replay.raw.params["kappa_K"], cal_res.raw.params["kappa_K"], atol=1e-12)

        runs_after_replay = reopened.ledger()
        assert len(runs_after_replay) == 5
        assert runs_after_replay[3].inputs["replayed_from"] == runs_before_move[1].id
        assert runs_after_replay[4].inputs["replayed_from"] == runs_before_move[2].id
        # Original records remained untouched
        assert runs_after_replay[0].id == runs_before_move[0].id
        assert runs_after_replay[1].id == runs_before_move[1].id
