"""Sprint 8 Slim test suite: OpenFOAM/FEniCSx/HDF5 audit adapters & headless LaTeX/PDF renderer."""

from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import rosolver
from rosolver import (
    ClaimKind,
    ClaimManifest,
    ClaimSpec,
    Dataset,
    FEniCSxAdapter,
    HDF5Adapter,
    LatexReportRenderer,
    Model,
    OpenFOAMAdapter,
    Project,
    ReportGenerator,
    VerificationStatus,
    VerificationStudy,
    compile_latex_to_pdf,
    render_report_latex,
)
from rosolver.cli import EXIT_OK, main as cli_main
from rosolver.mcp.server import MCPServer
from rosolver.reporting.latex import escape_latex

REPO_ROOT = Path(__file__).resolve().parent.parent
POND_MODEL_YAML = REPO_ROOT / "examples" / "pond_freezing_audit_study" / "model.yaml"


# ---------------------------------------------------------------------------
# 1. OpenFOAMAdapter Tests (Sample Sets, Residuals, Case Discovery, Multi-Grid GCI)
# ---------------------------------------------------------------------------


def test_openfoam_adapter_sample_vectors_residuals_and_case_discovery(tmp_path: Path):
    case_dir = tmp_path / "cavity_case"
    (case_dir / "system").mkdir(parents=True)
    (case_dir / "system" / "controlDict").write_text(
        """
        /* OpenFOAM controlDict */
        FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
        application     simpleFoam;
        startTime       0;
        endTime         500;
        deltaT          1;
        writeInterval   100;
        """,
        encoding="utf-8",
    )

    # Sample set with vector (Ux Uy Uz) column
    probe_dir = case_dir / "postProcessing" / "lineProbe" / "500"
    probe_dir.mkdir(parents=True)
    (probe_dir / "line_U.xy").write_text(
        "# x U\n"
        "0.00 (0.0 0.0 0.0)\n"
        "0.25 (0.5 0.1 0.0)\n"
        "0.50 (1.0 0.2 0.0)\n"
        "0.75 (0.5 0.1 0.0)\n"
        "1.00 (0.0 0.0 0.0)\n",
        encoding="utf-8",
    )

    # Residuals file
    res_dir = case_dir / "postProcessing" / "residuals" / "0"
    res_dir.mkdir(parents=True)
    (res_dir / "residuals.dat").write_text(
        "# Residuals\n"
        "# Time Ux Uy p\n"
        "1   1.0e-01   5.0e-02   2.0e-01\n"
        "100 1.0e-03   5.0e-04   2.0e-03\n"
        "500 1.0e-06   2.0e-06   5.0e-06\n",
        encoding="utf-8",
    )

    # Auto-discover sample set from case directory
    foam = OpenFOAMAdapter(units={"x": "m", "U_x": "m/s", "U_y": "m/s", "U_z": "m/s"})
    ds = foam.load_results(case_dir)
    assert "x" in ds.coordinates
    assert {"U_x", "U_y", "U_z"}.issubset(ds.fields.keys())
    assert np.isclose(ds.fields["U_x"][2], 1.0)
    assert foam.extract_solver_metadata()["application"] == "simpleFoam"
    assert foam.extract_solver_metadata()["endTime"] == 500.0
    assert np.isclose(foam.extract_mesh_metadata()["h_sample"], 0.25)

    # Load residuals and verify convergence metrics
    foam_res = OpenFOAMAdapter(function_object="residuals", residual_tol=1e-5)
    summary = foam_res.load_residuals(case_dir)
    assert summary["all_converged"] is True
    assert summary["fields"]["Ux"]["orders_drop"] >= 4.9
    assert summary["fields"]["p"]["final"] == pytest.approx(5.0e-6)


def test_openfoam_multi_grid_gci_verification(tmp_path: Path):
    # Construct 3 OpenFOAM grid refinement sample files with exact 2nd-order convergence:
    # Q(h) = 2.5 + 1.2 * h^2
    hs = [0.04, 0.02, 0.01]
    files = []
    for idx, h in enumerate(hs):
        p = tmp_path / f"grid_{idx}" / "postProcessing" / "probe" / "10" / "T.xy"
        p.parent.mkdir(parents=True)
        q_val = 2.5 + 1.2 * (h ** 2)
        p.write_text(f"# x T\n0.0 {q_val:.10f}\n0.5 {q_val:.10f}\n1.0 {q_val:.10f}\n", encoding="utf-8")
        files.append(p)

    adapter = OpenFOAMAdapter()
    study = VerificationStudy.from_multi_grid(
        adapter,
        files,
        spacings=hs,
        field="T",
        reducer=lambda arr: float(arr[-1]),
    )
    gci_res = study.grid_convergence(formal_order=2.0)
    assert gci_res.status == VerificationStatus.PASSED
    assert gci_res.raw.observed_order == pytest.approx(2.0, abs=1e-2)
    assert gci_res.raw.extrapolated_value == pytest.approx(2.5, abs=1e-5)


# ---------------------------------------------------------------------------
# 2. FEniCSxAdapter Tests (VTK-CSV, Legacy ASCII VTK, XML VTU, Inline XDMF)
# ---------------------------------------------------------------------------


def test_fenicsx_adapter_vtk_csv_vtk_vtu_and_xdmf(tmp_path: Path):
    # A. ParaView VTK-CSV
    csv_p = tmp_path / "fenics_probe.csv"
    csv_p.write_text(
        "Points:0,Points:1,Points:2,u:0,u:1,T\n"
        "0.0,0.0,0.0,0.0,0.0,300.0\n"
        "0.5,0.0,0.0,1.0,0.2,310.0\n"
        "1.0,0.0,0.0,2.0,0.4,320.0\n",
        encoding="utf-8",
    )
    f_csv = FEniCSxAdapter(element_family="Lagrange", element_degree=2, units={"x": "m", "T": "K"})
    ds_csv = f_csv.load_results(csv_p)
    assert {"x", "y", "z"}.issubset(ds_csv.coordinates.keys())
    assert {"u_x", "u_y", "T"}.issubset(ds_csv.fields.keys())
    assert f_csv.extract_solver_metadata()["formal_order"] == 3
    assert f_csv.extract_mesh_metadata()["h_eff"] == pytest.approx(0.5)

    # B. Legacy ASCII VTK (UNSTRUCTURED_GRID)
    vtk_p = tmp_path / "mesh.vtk"
    vtk_p.write_text(
        "# vtk DataFile Version 3.0\n"
        "FEniCSx export\n"
        "ASCII\n"
        "DATASET UNSTRUCTURED_GRID\n"
        "POINTS 3 float\n"
        "0.0 0.0 0.0\n"
        "0.5 0.0 0.0\n"
        "1.0 0.0 0.0\n"
        "CELLS 2 6\n"
        "2 0 1\n"
        "2 1 2\n"
        "CELL_TYPES 2\n"
        "3 3\n"
        "POINT_DATA 3\n"
        "SCALARS phi float 1\n"
        "LOOKUP_TABLE default\n"
        "1.0 2.0 3.0\n"
        "VECTORS vel float\n"
        "0.1 0.0 0.0\n"
        "0.2 0.0 0.0\n"
        "0.3 0.0 0.0\n",
        encoding="utf-8",
    )
    ds_vtk = FEniCSxAdapter().load_results(vtk_p)
    assert np.allclose(ds_vtk.fields["phi"], [1.0, 2.0, 3.0])
    assert np.allclose(ds_vtk.fields["vel_x"], [0.1, 0.2, 0.3])
    assert ds_vtk.mesh_metadata["n_cells"] == 2

    # C. XML ASCII VTU
    vtu_p = tmp_path / "sol.vtu"
    vtu_p.write_text(
        """<?xml version="1.0"?>
        <VTKFile type="UnstructuredGrid" version="0.1">
          <UnstructuredGrid>
            <Piece NumberOfPoints="3" NumberOfCells="2">
              <Points>
                <DataArray type="Float64" NumberOfComponents="3" format="ascii">
                  0 0 0  0.5 0 0  1 0 0
                </DataArray>
              </Points>
              <PointData>
                <DataArray type="Float64" Name="u" format="ascii">10 20 30</DataArray>
              </PointData>
            </Piece>
          </UnstructuredGrid>
        </VTKFile>
        """,
        encoding="utf-8",
    )
    study_vtu = VerificationStudy.from_fenicsx(vtu_p, field="u")
    assert np.allclose(study_vtu.dataset.fields["u"], [10.0, 20.0, 30.0])

    # D. Inline XML XDMF + HDF5 reference guard
    xdmf_p = tmp_path / "inline.xdmf"
    xdmf_p.write_text(
        """<?xml version="1.0"?>
        <Xdmf Version="3.0">
          <Domain>
            <Grid Name="mesh" GridType="Uniform">
              <Topology TopologyType="PolyLine" NumberOfElements="2"/>
              <Geometry GeometryType="XYZ">
                <DataItem Dimensions="3 3" Format="XML">0 0 0  0.5 0 0  1 0 0</DataItem>
              </Geometry>
              <Attribute Name="p" Center="Node">
                <DataItem Dimensions="3" Format="XML">101.0 102.0 103.0</DataItem>
              </Attribute>
            </Grid>
          </Domain>
        </Xdmf>
        """,
        encoding="utf-8",
    )
    ds_xdmf = FEniCSxAdapter().load_results(xdmf_p)
    assert np.allclose(ds_xdmf.fields["p"], [101.0, 102.0, 103.0])

    xdmf_h5 = tmp_path / "ref_h5.xdmf"
    xdmf_h5.write_text(
        """<?xml version="1.0"?>
        <Xdmf Version="3.0"><Domain><Grid GridType="Uniform">
          <Geometry><DataItem Format="HDF">mesh.h5:/mesh/geometry</DataItem></Geometry>
        </Grid></Domain></Xdmf>
        """,
        encoding="utf-8",
    )
    with pytest.raises(ImportError, match="rosolver\\[io\\]"):
        FEniCSxAdapter().load_results(xdmf_h5)


# ---------------------------------------------------------------------------
# 3. HDF5Adapter & Dataset.to_hdf5 Tests (Guard + Hierarchical Round-Trip)
# ---------------------------------------------------------------------------


def test_hdf5_adapter_optional_guard_and_hierarchical_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    h5_file = tmp_path / "sim_out.h5"

    # 1. Verify clean ImportError when h5py is unavailable
    monkeypatch.setitem(sys.modules, "h5py", None)
    with pytest.raises(ImportError, match="rosolver\\[io\\]"):
        HDF5Adapter().load_results(h5_file)
    with pytest.raises(ImportError, match="rosolver\\[io\\]"):
        Dataset(name="d", fields={"y": [1.0, 2.0]}).to_hdf5(h5_file)

    # 2. Verify full structured round-trip using a minimal in-memory h5py mock
    store: dict[str, dict] = {}

    class _FakeDataset:
        def __init__(self, arr: np.ndarray):
            self._arr = np.asarray(arr, dtype=float)

        def __getitem__(self, key):
            return self._arr

    class _FakeGroup(dict):
        def __init__(self):
            super().__init__()
            self.attrs: dict = {}

        def require_group(self, name: str):
            name = name.strip("/")
            if not name:
                return self
            if name not in self:
                self[name] = _FakeGroup()
            return self[name]

        def create_dataset(self, name: str, data):
            self[name] = _FakeDataset(data)
            return self[name]

        def __getitem__(self, key: str):
            if key in ("/", ""):
                return self
            return super().__getitem__(key.strip("/"))

    class _FakeFile:
        def __init__(self, path, mode="r"):
            self._path = Path(path)
            self._mode = mode
            if mode == "w":
                self._path.write_bytes(b"FAKEHDF5")
                store[str(self._path)] = _FakeGroup()
            self._root = store[str(self._path)]

        def __enter__(self):
            return self._root

        def __exit__(self, *exc):
            return False

    fake_h5py = types.ModuleType("h5py")
    fake_h5py.File = _FakeFile  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "h5py", fake_h5py)

    ds_orig = Dataset(
        name="cfd_run",
        coordinates={"x": np.array([0.0, 0.5, 1.0])},
        fields={"T": np.array([300.0, 310.0, 320.0])},
        uncertainties={"T": np.array([0.5, 0.5, 0.5])},
        units={"x": "m", "T": "K"},
        mesh_metadata={"n_cells": 2},
        solver_metadata={"solver": "CustomCFD"},
    )
    ds_orig.to_hdf5(h5_file)

    adapter = HDF5Adapter()
    ds_loaded = adapter.load_results(h5_file)
    assert ds_loaded.name == "cfd_run"
    assert np.allclose(ds_loaded.coordinates["x"], [0.0, 0.5, 1.0])
    assert np.allclose(ds_loaded.fields["T"], [300.0, 310.0, 320.0])
    assert np.allclose(ds_loaded.uncertainties["T"], [0.5, 0.5, 0.5])
    assert ds_loaded.units == {"x": "m", "T": "K"}
    assert adapter.extract_solver_metadata()["solver"] == "CustomCFD"


# ---------------------------------------------------------------------------
# 4. Headless LaTeX & PDF Renderer + CLI + MCP Integration
# ---------------------------------------------------------------------------


def test_latex_escaping_and_referee_report_render_and_pdf_compile(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    # A. Special character escaping
    raw = r"err_rel = 0.35% & #1 $x^2$ ~ {a\b}"
    esc = escape_latex(raw)
    assert r"\_" in esc and r"\%" in esc and r"\&" in esc and r"\#" in esc and r"\$" in esc

    # B. Build a real Sprint 10 claim audit project and render peer_review_letter to .tex and .pdf
    proj = Project.create(tmp_path / "latex_audit_proj", name="LaTeX Audit Study")
    proj.add_model(Model.from_file(POND_MODEL_YAML))
    manifest = ClaimManifest(
        study_title="Pond Freeze Audit & 100% Scale Check",
        claims=[
            ClaimSpec(
                id="C_euler_gap",
                title="Euler balance rho*a + grad_P = 0",
                kind=ClaimKind.EQUATION_LAW,
                equations=["rho * a + grad_P = 0"],
                variables_units={"rho": "kg/m^3", "a": "m/s^2", "grad_P": "Pa/m"},
                operating_envelope={"rho": (917.0, 917.0), "a": (7.5e-14, 7.5e-14), "grad_P": (885.0, 8850.0)},
            ),
            ClaimSpec(
                id="C_peat_ice",
                title="Peat pond ice purity",
                kind=ClaimKind.QUALITATIVE_OBSERVATION,
                required_observables=["solute_rejection_pct"],
                claimed_metrics={"min_solute_rejection_pct": 90.0},
                metadata={"mechanistic_plausible": True},
            ),
        ],
    )
    r1 = proj.audit_claims(manifest=manifest)
    rep = ReportGenerator.from_claim_round(proj, r1)

    tex_str = render_report_latex(rep)
    assert r"\documentclass[10pt]{article}" in tex_str
    assert r"\begin{tabularx}" in tex_str
    assert "Peer-Review Scientific Claim Audit" in tex_str
    assert "C\\_euler\\_gap" in tex_str
    assert str(Path.home().parent) not in tex_str

    tex_out = tmp_path / "reports" / "referee_r1.tex"
    rep.save(tex_out)
    assert tex_out.exists()

    # C. Headless PDF compilation when a TeX engine is available on PATH
    if shutil.which("pdflatex") or shutil.which("xelatex") or shutil.which("tectonic"):
        pdf_out = tmp_path / "reports" / "referee_r1.pdf"
        rep.save(pdf_out)
        assert pdf_out.exists()
        assert pdf_out.read_bytes().startswith(b"%PDF-")
        # Auxiliary files cleaned up
        assert not (tmp_path / "reports" / "referee_r1.aux").exists()

    # D. CLI `rosolver report` with .tex output + MCP `rosolver_report` returning latex
    cli_rc = cli_main(["report", str(proj.root), "--output", str(tmp_path / "cli_rep.tex"), "--json"])
    assert cli_rc in (EXIT_OK, 1)
    capsys.readouterr()
    assert (tmp_path / "cli_rep.tex").exists()

    srv = MCPServer()
    mcp_resp = srv.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 99,
            "method": "tools/call",
            "params": {
                "name": "rosolver_report",
                "arguments": {"project_dir": str(proj.root), "output": "reports/mcp_rep.tex"},
            },
        }
    )
    sc = mcp_resp["result"]["structuredContent"]
    assert r"\documentclass" in sc["latex"]
    assert (proj.root / "reports" / "mcp_rep.tex").exists()
