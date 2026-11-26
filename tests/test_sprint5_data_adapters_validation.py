"""Sprint 5 tests: Dataset I/O, SolverAdapter, VerificationStudy & ValidationStudy (P2.2, P2.3, P2.4)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rosolver import (
    ConservationLaw1D,
    Dataset,
    GenericCSVAdapter,
    ODESystem,
    Project,
    SciPyAdapter,
    ValidationStudy,
    VerificationStatus,
    VerificationStudy,
    gci_from_values,
    plot_validation,
)


class TestDatasetInterface:
    def test_csv_load_units_uncertainties_and_missing_policies(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "exp_heat.csv"
        csv_path.write_text(
            "x,T,u_T\n0.0,300.0,0.5\n0.5,320.0,0.6\n1.0,310.0,0.4\n",
            encoding="utf-8",
        )
        ds = Dataset.from_csv(
            csv_path,
            coordinates="x",
            fields="T",
            uncertainties={"T": "u_T"},
            units={"x": "m", "T": "K"},
        )
        assert ds.n_rows == 3
        assert np.allclose(ds.column("T"), [300.0, 320.0, 310.0])
        assert np.allclose(ds.uncertainty_for("T"), [0.5, 0.6, 0.4])
        assert ds.provenance["format"] == "csv"
        assert len(ds.provenance["sha256"]) == 64

        with pytest.raises(ValueError, match="Invalid unit"):
            Dataset.from_csv(csv_path, units={"T": "123_bad_unit"})

        # Missing value policies
        bad_csv = tmp_path / "missing.csv"
        bad_csv.write_text("x,T\n0.0,300.0\n0.5,\n1.0,310.0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Missing value"):
            Dataset.from_csv(bad_csv, coordinates="x", fields="T", missing_policy="raise")

        dropped = Dataset.from_csv(bad_csv, coordinates="x", fields="T", missing_policy="drop")
        assert dropped.n_rows == 2
        assert dropped.provenance["n_dropped"] == 1

    def test_json_yaml_numpy_security_and_optional_hdf5_guard(self, tmp_path: Path) -> None:
        ds = Dataset(
            name="demo",
            coordinates={"x": np.array([0.0, 1.0])},
            fields={"u": np.array([1.0, 2.0])},
            units={"x": "m", "u": "m/s"},
        )
        npz_path = ds.to_npz(tmp_path / "demo.npz")
        ds_npz = Dataset.from_numpy(npz_path, coordinates="x", fields="u", units={"x": "m", "u": "m/s"})
        assert np.allclose(ds_npz.column("u"), [1.0, 2.0])

        json_path = tmp_path / "demo.json"
        json_path.write_text(ds.to_json(), encoding="utf-8")
        ds_json = Dataset.from_json(json_path)
        assert ds_json.units["u"] == "m/s"

        yaml_path = tmp_path / "demo.yaml"
        yaml_path.write_text("coordinates:\n  x: [0.0, 1.0]\nfields:\n  u: [1.0, 2.0]\nunits:\n  u: m/s\n", encoding="utf-8")
        ds_yaml = Dataset.from_yaml(yaml_path)
        assert np.allclose(ds_yaml.column("u"), [1.0, 2.0])

        # Pickled object npz must be rejected (allow_pickle=False)
        pickled_npz = tmp_path / "evil.npz"
        np.savez(pickled_npz, bad=np.array([{"a": 1}], dtype=object))
        with pytest.raises(ValueError):
            Dataset.from_numpy(pickled_npz)

        # HDF5 raises helpful ImportError if h5py is not installed
        try:
            import h5py  # noqa: F401
        except ImportError:
            with pytest.raises(ImportError, match="rosolver\\[io\\]"):
                Dataset.from_hdf5(tmp_path / "missing.h5")


class TestSolverAdaptersAndVerificationStudy:
    def test_generic_csv_adapter_and_verification_study(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "external_solver.csv"
        csv_file.write_text(
            "x,temperature,analytical_solution\n"
            "0.0,0.0,0.0\n"
            "0.25,0.7071,0.70710678\n"
            "0.5,1.0,1.0\n"
            "0.75,0.7071,0.70710678\n"
            "1.0,0.0,0.0\n",
            encoding="utf-8",
        )
        adapter = GenericCSVAdapter(coordinates="x", fields=["temperature", "analytical_solution"], units={"x": "m", "temperature": "K"})
        ds = adapter.load_results(csv_file)
        assert "temperature" in adapter.extract_fields()
        # Unsupported metadata explicitly marked None rather than fabricated
        assert adapter.extract_mesh_metadata()["cell_spacings"] is None
        assert adapter.extract_solver_metadata()["solver"] is None
        exported = adapter.export_standard_format(tmp_path / "exported.json")
        assert json.loads(exported.read_text(encoding="utf-8"))["name"] == "external_solver"

        study = VerificationStudy.from_csv(
            csv_file,
            coordinate="x",
            field="temperature",
            reference="analytical_solution",
        )
        err_rep = study.reference_error(l2_tol=1e-4, linf_tol=1e-4)
        assert err_rep.status == VerificationStatus.PASSED
        assert err_rep.raw.linf < 1e-4

        # GCI via VerificationStudy delegates directly to gci_from_values
        mesh_csv = tmp_path / "mesh_qoi.csv"
        mesh_csv.write_text("h,nusselt\n0.04,4.912\n0.02,4.978\n0.01,4.9945\n", encoding="utf-8")
        gci_study = VerificationStudy.from_csv(mesh_csv, spacing_col="h", qoi_col="nusselt")
        sr_gci = gci_study.grid_convergence(formal_order=2.0)
        direct = gci_from_values([0.04, 0.02, 0.01], [4.912, 4.978, 4.9945], formal_order=2.0)
        assert sr_gci.status == VerificationStatus.PASSED
        assert np.isclose(sr_gci.raw.gci_fine_21, direct.gci_fine_21)

    def test_scipy_adapter_wraps_ode_and_pde(self) -> None:
        ode = ODESystem({"y": "-k * y"})
        traj = ode.solve((0.0, 1.0), {"y": 1.0}, params={"k": 1.0}, n_out=10)
        ad_ode = SciPyAdapter(units={"t": "s", "y": "1"})
        ds_ode = ad_ode.load_results(traj)
        assert ds_ode.n_rows == 11
        assert ad_ode.extract_solver_metadata()["solver"] == "ODESystem"

        law = ConservationLaw1D("T", diffusivity="alpha")
        pde_sol = law.solve((0.0, 1.0), 20, t_end=0.05, initial="sin(pi*x)", bc=("dirichlet", 0.0), params={"alpha": 0.1}, n_out=3)
        ad_pde = SciPyAdapter(units={"x": "m", "T": "K"})
        ds_pde = ad_pde.load_results(pde_sol)
        assert ad_pde.extract_mesh_metadata()["n_cells"] == 20
        assert "T" in ds_pde.fields


class TestValidationStudy:
    def test_validation_classifications_interpolation_units_plot_and_project(self, tmp_path: Path) -> None:
        coords = np.array([0.2, 0.4, 0.6, 0.8])
        exp = np.array([300.0, 310.0, 315.0, 305.0])
        sim_ok = np.array([300.3, 309.6, 315.4, 304.8])

        # 1. Both u_num and u_exp supplied & within band -> PASSED
        rep_pass = ValidationStudy.compare(
            sim_ok,
            exp,
            field="T",
            unit="K",
            u_num=0.3,
            u_exp=0.5,
            coordinates=coords,
            operating_conditions={"Re": 12000, "T_wall": 295.0},
            study_name="heated_slab",
        )
        assert rep_pass.status == VerificationStatus.PASSED
        assert rep_pass.passed
        assert rep_pass.fraction_within_uncertainty == 1.0
        assert "not constitute formal standards compliance" in " ".join(rep_pass.assumptions)

        # 2. Exceeds u_val band -> FAILED
        rep_fail = ValidationStudy.compare(sim_ok + 5.0, exp, field="T", unit="K", u_num=0.3, u_exp=0.5)
        assert rep_fail.status == VerificationStatus.FAILED
        assert not rep_fail.passed

        # 3. Missing u_num -> WARNING
        rep_warn = ValidationStudy.compare(sim_ok, exp, field="T", unit="K", u_exp=0.8)
        assert rep_warn.status == VerificationStatus.WARNING

        # 4. Neither uncertainties nor error_tol -> INCONCLUSIVE
        rep_inc = ValidationStudy.compare(sim_ok, exp, field="T", unit="K")
        assert rep_inc.status == VerificationStatus.INCONCLUSIVE

        # 5. Dataset interpolation + unit check + plot + Project ledger integration
        sim_ds = Dataset(
            name="fvm_sim",
            coordinates={"x": np.linspace(0.0, 1.0, 11)},
            fields={"T": 300.0 + 20.0 * np.sin(np.pi * np.linspace(0.0, 1.0, 11))},
            units={"x": "m", "T": "K"},
        )
        exp_ds = Dataset(
            name="thermocouples",
            coordinates={"x": np.array([0.25, 0.5, 0.75])},
            fields={"T": 300.0 + 20.0 * np.sin(np.pi * np.array([0.25, 0.5, 0.75])) + np.array([0.1, -0.1, 0.05])},
            uncertainties={"T": np.array([0.5, 0.5, 0.5])},
            units={"x": "m", "T": "K"},
        )
        rep_ds = ValidationStudy.from_datasets(sim_ds, exp_ds, field="T", coord="x", u_num=0.2)
        assert rep_ds.status == VerificationStatus.PASSED

        bad_unit_ds = Dataset(
            name="bad_exp",
            coordinates={"x": np.array([0.5])},
            fields={"T": np.array([300.0])},
            units={"T": "Pa"},
        )
        with pytest.raises(ValueError, match="Unit mismatch"):
            ValidationStudy.from_datasets(sim_ds, bad_unit_ds, field="T", coord="x")

        fig_path = plot_validation(rep_ds, path=tmp_path / "validation.png", x_label="x [m]")
        assert fig_path is not None and fig_path.exists() and fig_path.stat().st_size > 1000

        proj = Project.create(tmp_path / "val_proj")
        rec = proj.record_run("validation", rep_ds)
        assert rec.status == "Passed"
        assert (tmp_path / "val_proj" / rec.outputs["result"]).exists()
