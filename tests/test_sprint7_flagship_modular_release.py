"""Sprint 7 tests: Flagship heat-conduction study (P3.3), modular subpackages (P4.1) & v0.2.0 release gate (P4.4)."""

from __future__ import annotations

from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import subprocess
import sys

import pytest

import rosolver
from examples.heat_transfer_study import run_flagship_study
from rosolver import Project, VerificationStatus
from rosolver.cli import EXIT_OK, main as cli_main


def test_flagship_heat_conduction_study_full_lifecycle(tmp_path: Path):
    summary = run_flagship_study(output_dir=tmp_path / "heat_study")

    assert summary["report_status"] == VerificationStatus.PASSED
    for stage, st in summary["statuses"].items():
        assert st == VerificationStatus.PASSED, f"Stage '{stage}' status was {st}"

    assert summary["metrics"]["fourier_rmse_K"] < 0.01
    assert summary["metrics"]["mms_order_l2"] >= 1.75
    assert summary["metrics"]["conservation_max_abs_residual"] < 1e-10
    assert summary["metrics"]["gci_fine_21"] < 0.05

    report_path: Path = summary["report_path"]
    manifest_path: Path = summary["manifest_path"]
    assert report_path.exists()
    assert manifest_path.exists()

    md_text = report_path.read_text(encoding="utf-8")
    assert "/Users/" not in md_text
    assert "`heat_conduction_1d`" in md_text
    assert "ASME V&V 20" in md_text

    for fig in summary["figures"]:
        assert fig.exists()
        rel = fig.relative_to(summary["project_root"]).as_posix()
        assert rel in md_text

    proj = Project.open(summary["project_root"])
    run_types = {r.type for r in proj.ledger()}
    assert {
        "symbolic_derivation",
        "pde_simulation",
        "gci",
        "timestep",
        "model_validation",
        "mms",
        "conservation",
    }.issubset(run_types)


def test_flagship_model_yaml_via_workflow_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    model_yaml = Path(__file__).resolve().parent.parent / "examples" / "heat_transfer_study" / "model.yaml"
    proj_dir = tmp_path / "cli_flagship"

    assert cli_main(["init", str(proj_dir), "--model", str(model_yaml), "--json"]) == EXIT_OK
    capsys.readouterr()

    assert cli_main(["validate", str(proj_dir), "--json"]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["status"] == "valid"

    assert cli_main(["model", "analyse", str(proj_dir), "--target", "T", "--wrt", "t", "--zero-initials", "--json"]) == EXIT_OK
    capsys.readouterr()

    assert cli_main(["simulate", str(proj_dir), "--t-end", "0.2", "--n-cells", "32", "--json"]) == EXIT_OK
    capsys.readouterr()

    assert cli_main(["verify", str(proj_dir), "--method", "conservation", "--t-end", "0.2", "--json"]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["status"] == VerificationStatus.PASSED.value

    assert cli_main(["report", str(proj_dir), "--output", "reports/flagship_cli.md", "--json"]) == EXIT_OK
    assert (proj_dir / "reports" / "flagship_cli.md").exists()
    capsys.readouterr()


def test_modular_subpackages_import_cleanly_in_subprocesses():
    packages = (
        ("core", ("Model", "Project", "Dimension", "VerificationStatus")),
        ("symbolic", ("SymbolicSolver", "DerivationResult", "TEMPLATE_REGISTRY")),
        ("calibration", ("NumericalSolver", "FitResult", "Dataset")),
        ("numerics", ("ConservationLaw1D", "ODESystem", "Stefan1D")),
        ("verification", ("mms_study", "gci_from_values", "audit_pde_conservation")),
        ("validation", ("ValidationStudy", "ValidationReport")),
        ("integrations", ("SolverAdapter", "GenericCSVAdapter", "VerificationStudy")),
        ("io", ("Dataset", "Model", "Project", "to_jsonable")),
        ("reporting", ("REPORT_TYPES", "ScientificReport", "ReportGenerator")),
    )
    for subpkg, expected_symbols in packages:
        code = (
            f"import rosolver.{subpkg} as m; "
            f"syms = {expected_symbols!r}; "
            "assert all(hasattr(m, s) for s in syms)"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"Subpackage rosolver.{subpkg} failed: {proc.stderr}"


def test_version_and_metadata_match_v020():
    assert rosolver.__version__ == "0.2.0"
    assert importlib_metadata.version("rosolver") == "0.2.0"
