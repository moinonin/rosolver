"""Comprehensive tests for `rosolver-mcp` (tools, resources, prompts, security & stdio subprocess)."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict

import numpy as np
import pytest

from rosolver.mcp import MCPServer, TOOL_SPECIFICATIONS


def _call(server: MCPServer, name: str, arguments: Dict[str, Any] | None = None, req_id: int = 1) -> Dict[str, Any]:
    resp = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}},
        }
    )
    assert resp is not None and "result" in resp
    return resp["result"]


def test_mcp_initialize_tools_resources_and_prompts():
    srv = MCPServer()
    init = srv.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "clientInfo": {"name": "pytest"}},
        }
    )
    assert init is not None
    assert init["result"]["serverInfo"]["name"] == "rosolver-mcp"
    assert "tools" in init["result"]["capabilities"]

    # Notification produces None response
    assert srv.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert srv.initialized is True

    tools_resp = srv.handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert tools_resp is not None
    names = [t["name"] for t in tools_resp["result"]["tools"]]
    assert len(names) == 15
    assert len(TOOL_SPECIFICATIONS) == 15
    assert "rosolver_import_external" in names
    assert "rosolver_screen_candidates" in names

    # Resources
    res_list = srv.handle_message({"jsonrpc": "2.0", "id": 3, "method": "resources/list"})
    assert res_list is not None
    uris = [r["uri"] for r in res_list["result"]["resources"]]
    assert len(uris) == 6
    assert "rosolver://schema/model" in uris
    assert "rosolver://schema/adapters" in uris
    for uri in uris:
        r_read = srv.handle_message({"jsonrpc": "2.0", "id": 4, "method": "resources/read", "params": {"uri": uri}})
        assert r_read is not None
        assert r_read["result"]["contents"][0]["text"]

    # Prompts
    pr_list = srv.handle_message({"jsonrpc": "2.0", "id": 5, "method": "prompts/list"})
    assert pr_list is not None
    for pr in pr_list["result"]["prompts"]:
        got = srv.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "prompts/get",
                "params": {"name": pr["name"], "arguments": {"problem_description": "1D rod cooling"}},
            }
        )
        assert got is not None
        assert got["result"]["messages"][0]["content"]["text"]


def test_mcp_stateless_dimensions_formulate_calibrate_and_security():
    srv = MCPServer()

    cap = _call(srv, "rosolver_capabilities")
    assert cap["isError"] is False
    sc = cap["structuredContent"]
    assert sc["rosolver_version"] == "0.2.0"
    assert len(sc["tools"]) == 15
    assert set(["openfoam", "fenicsx", "hdf5"]).issubset(set(sc["external_adapters"]))
    assert set(["markdown", "latex", "pdf"]).issubset(set(sc["report_formats"]))
    assert "pysr_csv" in sc["symbolic_importers"]
    assert len(sc["report_types"]) == 10
    assert "k" in sc["si_prefixes"]

    # Dimensional consistency + Buckingham-Pi
    dim_ok = _call(
        srv,
        "rosolver_check_dimensions",
        {
            "dims": {"P": "W", "rho": "kg/m^3", "A": "m^2", "V": "m/s"},
            "equation": "P = 0.5 * rho * A * V**3",
            "target": "P",
        },
    )
    assert dim_ok["isError"] is False
    assert dim_ok["structuredContent"]["equation_check"]["consistent"] is True
    assert dim_ok["structuredContent"]["buckingham_pi"]["consistent"] is True

    dim_bad = _call(
        srv,
        "rosolver_check_dimensions",
        {"dims": {"P": "W", "V": "m/s"}, "equation": "P = V"},
    )
    assert dim_bad["structuredContent"]["equation_check"]["consistent"] is False
    assert dim_bad["structuredContent"]["equation_check"]["status"] == "Failed"

    # Symbolic formulation
    form = _call(
        srv,
        "rosolver_formulate",
        {
            "relations": ["K = fn(m, v)"],
            "dims": {"K": "M*L^2/T^2", "m": "M", "v": "L/T"},
            "target": "K",
            "wrt": "v",
            "zero_initials": True,
            "include_latex": True,
        },
    )
    assert form["isError"] is False
    assert "v**2" in form["structuredContent"]["primary_equation"]
    assert form["structuredContent"]["latex"]

    # WLS Calibration + candidate ranking
    x_vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_vals = [2.5 * x for x in x_vals]
    cal = _call(
        srv,
        "rosolver_calibrate",
        {
            "mode": "single",
            "equation": "y = a * x",
            "free_params": ["a"],
            "data": {"x": x_vals, "y": y_vals},
            "sigma_y": [0.05] * 5,
        },
    )
    assert cal["isError"] is False
    assert cal["structuredContent"]["result"]["params"]["a"] == pytest.approx(2.5, rel=1e-4)

    rank = _call(
        srv,
        "rosolver_calibrate",
        {
            "mode": "rank_candidates",
            "relations": ["y = fn(x)"],
            "target": "y",
            "wrt": "x",
            "zero_initials": True,
            "data": {"x": x_vals, "y": y_vals},
        },
    )
    assert rank["isError"] is False
    assert len(rank["structuredContent"]["result"]["ranked"]) >= 1

    # Security: code injection via expression must fail safely with isError=True
    inj = _call(
        srv,
        "rosolver_check_dimensions",
        {"dims": {"x": "m"}, "equation": "__import__('os').system('echo pwned')"},
    )
    assert inj["isError"] is True


def test_mcp_simulate_verify_and_validate_all_modes():
    srv = MCPServer()

    # ODE with terminal event
    ode_res = _call(
        srv,
        "rosolver_simulate",
        {
            "solver": "ode",
            "rhs": {"z": "v", "v": "-g"},
            "y0": {"z": 4.905, "v": 0.0},
            "params": {"g": 9.81},
            "t_span": [0.0, 3.0],
            "events": [{"name": "ground", "expr": "z", "terminal": True, "direction": -1}],
        },
    )
    assert ode_res["isError"] is False
    assert ode_res["structuredContent"]["t_events"]["ground"][0] == pytest.approx(1.0, rel=1e-3)

    # Standalone 1D FVM PDE
    pde_res = _call(
        srv,
        "rosolver_simulate",
        {
            "solver": "pde",
            "fields": "T",
            "diffusivity": "alpha",
            "params": {"alpha": 0.05},
            "domain": [0.0, 1.0],
            "n_cells": 32,
            "t_span": [0.0, 0.2],
            "initial": "sin(pi * x)",
            "bc": {"left": 0.0, "right": 0.0},
            "method": "cn",
            "max_preview_points": 16,
        },
    )
    assert pde_res["isError"] is False
    assert len(pde_res["structuredContent"]["fields_final"]["T"]["preview"]) == 16

    # Stefan moving-boundary vs exact Neumann
    st_res = _call(
        srv,
        "rosolver_simulate",
        {"solver": "stefan", "T_wall": -6.0, "T_f": 0.0, "h_c": 0.0, "n_xi": 80, "t_end": 600.0},
    )
    assert st_res["isError"] is False
    assert st_res["structuredContent"]["final_delta_m"] == pytest.approx(
        st_res["structuredContent"]["neumann_exact_delta_m"], rel=1e-3
    )

    # Verify: GCI, MMS, ODE conservation, stiffness, coupling
    gci = _call(
        srv,
        "rosolver_verify",
        {"method": "gci", "spacings": [0.01, 0.02, 0.04], "quantities": [1.9572, 1.9480, 1.9112], "formal_order": 2.0},
    )
    assert gci["structuredContent"]["status"] == "Passed"

    mms = _call(
        srv,
        "rosolver_verify",
        {
            "method": "mms",
            "fields": "T",
            "diffusivity": "alpha",
            "params": {"alpha": 0.05},
            "exact": "exp(-t) * sin(pi * x)",
            "cells": [16, 32, 64],
        },
    )
    assert mms["structuredContent"]["status"] == "Passed"

    cons_ode = _call(
        srv,
        "rosolver_verify",
        {
            "method": "conservation_ode",
            "rhs": {"x": "v", "v": "-x"},
            "y0": {"x": 1.0, "v": 0.0},
            "invariants": {"energy": "0.5 * (x**2 + v**2)"},
            "t_span": [0.0, 5.0],
        },
    )
    assert cons_ode["structuredContent"]["status"] == "Passed"

    stiff = _call(
        srv,
        "rosolver_verify",
        {
            "method": "stiffness",
            "rhs": {"y1": "-1000 * y1", "y2": "-0.5 * y2"},
            "state": {"y1": 1.0, "y2": 1.0},
        },
    )
    assert stiff["structuredContent"]["result"]["is_stiff"] is True

    coup = _call(
        srv,
        "rosolver_verify",
        {
            "method": "coupling",
            "sys_a_rhs": {"T_f": "-3.0 * (T_f - T_s)"},
            "sys_b_rhs": {"T_s": "1.8 * (T_f - T_s) - 0.4 * T_s"},
            "y0": {"T_f": 1.0, "T_s": 0.0},
            "t_span": [0.0, 1.0],
            "dt_macro": 0.1,
            "mode": "strong",
        },
    )
    assert coup["isError"] is False

    # Validate results
    val = _call(
        srv,
        "rosolver_validate_results",
        {
            "simulation": [1.00, 2.01, 3.02],
            "experiment": [1.01, 2.00, 3.00],
            "field": "T",
            "unit": "K",
            "u_num": 0.02,
            "u_exp": 0.03,
        },
    )
    assert val["structuredContent"]["status"] == "Passed"


def test_mcp_project_lifecycle_report_and_real_stdio_subprocess(tmp_path: Path):
    srv = MCPServer()
    pdir = tmp_path / "mcp_proj"
    model_yaml = (Path(__file__).resolve().parent.parent / "examples" / "heat_transfer_study" / "model.yaml").read_text(
        encoding="utf-8"
    )

    init_res = _call(srv, "rosolver_project", {"action": "init", "project_dir": str(pdir), "name": "MCP Heat"})
    assert init_res["isError"] is False

    vmod = _call(
        srv,
        "rosolver_validate_model",
        {"project_dir": str(pdir), "model_yaml": model_yaml, "register_in_project": True},
    )
    assert vmod["structuredContent"]["valid"] is True

    sim = _call(srv, "rosolver_simulate", {"solver": "model", "project_dir": str(pdir), "t_span": [0.0, 0.2], "n_cells": 24})
    run_id = sim["structuredContent"]["run_id"]
    assert run_id

    replay = _call(srv, "rosolver_project", {"action": "replay_run", "project_dir": str(pdir), "run_id": run_id})
    assert replay["structuredContent"]["status"] == "Passed"

    rep = _call(srv, "rosolver_report", {"project_dir": str(pdir), "output": "reports/mcp_report.md"})
    assert rep["structuredContent"]["status"] == "Passed"
    assert (pdir / "reports" / "mcp_report.md").exists()
    assert "/Users/" not in rep["structuredContent"]["markdown"]

    # End-to-end real stdio subprocess test (both newline-delimited AND Content-Length framed!)
    proc = subprocess.Popen(
        [sys.executable, "-m", "rosolver.mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None and proc.stdout is not None
    try:
        # 1. Newline-delimited initialize + tools/list
        req1 = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}) + "\n"
        notif = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
        req2 = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}) + "\n"
        proc.stdin.write((req1 + notif + req2).encode("utf-8"))
        proc.stdin.flush()

        line1 = json.loads(proc.stdout.readline().decode("utf-8"))
        line2 = json.loads(proc.stdout.readline().decode("utf-8"))
        assert line1["id"] == 1
        assert line2["id"] == 2
        assert len(line2["result"]["tools"]) == 15

        # 2. Content-Length framed tools/call
        body = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "rosolver_verify",
                    "arguments": {
                        "method": "gci",
                        "spacings": [0.01, 0.02, 0.04],
                        "quantities": [1.0001, 1.0004, 1.0016],
                    },
                },
            }
        ).encode("utf-8")
        proc.stdin.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
        proc.stdin.flush()

        hdr = proc.stdout.readline()
        assert hdr.lower().startswith(b"content-length:")
        resp_len = int(hdr.split(b":", 1)[1].strip())
        blank = proc.stdout.readline()
        assert blank in (b"\r\n", b"\n")
        framed_resp = json.loads(proc.stdout.read(resp_len).decode("utf-8"))
        assert framed_resp["id"] == 3
        assert framed_resp["result"]["structuredContent"]["status"] == "Passed"

        proc.stdin.close()
        assert proc.wait(timeout=5) == 0
    finally:
        if proc.poll() is None:
            proc.kill()


def test_mcp_external_adapters_pysr_screening_and_latex_report(tmp_path: Path):
    srv = MCPServer()
    pdir = tmp_path / "full_caps_proj"
    init_res = _call(srv, "rosolver_project", {"action": "init", "project_dir": str(pdir), "name": "FullCaps"})
    assert init_res["isError"] is False

    # 1. OpenFOAM multi-grid GCI via rosolver_import_external
    case_dirs = []
    for tag, h_val, q_val in (("coarse", 0.04, 1.0016), ("med", 0.02, 1.0004), ("fine", 0.01, 1.0001)):
        cdir = tmp_path / f"of_{tag}"
        sdir = cdir / "postProcessing" / "lineSample" / "10"
        sdir.mkdir(parents=True)
        (cdir / "system").mkdir(parents=True)
        (cdir / "system" / "controlDict").write_text("application simpleFoam;\ndeltaT 0.01;\n", encoding="utf-8")
        (sdir / "line_T.xy").write_text(f"# x T\n0.0 {q_val}\n0.5 {q_val}\n1.0 {q_val}\n", encoding="utf-8")
        case_dirs.append(str(cdir))

    of_insp = _call(srv, "rosolver_import_external", {"adapter": "openfoam", "action": "inspect", "source": case_dirs[-1]})
    assert of_insp["isError"] is False
    assert "T" in of_insp["structuredContent"]["fields"]
    assert of_insp["structuredContent"]["solver_metadata"]["application"] == "simpleFoam"

    of_gci = _call(
        srv,
        "rosolver_import_external",
        {
            "adapter": "openfoam",
            "action": "gci",
            "sources": list(reversed(case_dirs)),
            "spacings": [0.01, 0.02, 0.04],
            "field": "T",
            "reducer": "mean",
            "formal_order": 2.0,
            "project_dir": str(pdir),
        },
    )
    assert of_gci["isError"] is False
    assert of_gci["structuredContent"]["status"] == "Passed"

    # 2. FEniCSx VTK-CSV reference error via rosolver_import_external
    vtk_csv = tmp_path / "fenics_out.csv"
    vtk_csv.write_text("Points:0,Points:1,Points:2,u\n0.0,0.0,0.0,0.0\n0.5,0.0,0.0,0.25\n1.0,0.0,0.0,1.0\n", encoding="utf-8")
    fe_ref = _call(
        srv,
        "rosolver_import_external",
        {
            "adapter": "fenicsx",
            "action": "reference_error",
            "source": str(vtk_csv),
            "coordinate": "x",
            "field": "u",
            "reference": "x**2",
            "l2_tol": 1e-9,
            "project_dir": str(pdir),
        },
    )
    assert fe_ref["isError"] is False
    assert fe_ref["structuredContent"]["status"] == "Passed"

    # 3. PySR symbolic candidate screening + ClaimManifest export via rosolver_screen_candidates
    pysr_csv = tmp_path / "hall_of_fame.csv"
    pysr_csv.write_text(
        "Complexity,Loss,Equation,Score\n"
        "3,0.1,x0 + x1,0.1\n"
        "5,0.0001,0.48 * x0 * x1**2,1.2\n",
        encoding="utf-8",
    )
    m_vals = np.linspace(1.0, 5.0, 12)
    v_vals = np.linspace(2.0, 6.0, 12)
    E_vals = 0.5 * m_vals * (v_vals ** 2)
    screen = _call(
        srv,
        "rosolver_screen_candidates",
        {
            "target": "E",
            "dims": {"E": "J", "m": "kg", "v": "m/s"},
            "pysr_csv": str(pysr_csv),
            "variable_map": {"x0": "m", "x1": "v"},
            "data": {"m": m_vals.tolist(), "v": v_vals.tolist(), "E": E_vals.tolist()},
            "export_claim_manifest": True,
            "register_manifest": True,
            "project_dir": str(pdir),
        },
    )
    assert screen["isError"] is False
    s_sc = screen["structuredContent"]
    assert s_sc["n_admissible"] == 1
    assert s_sc["n_rejected"] == 1
    assert s_sc["best_candidate"]["fit"]["r2"] > 0.9999
    assert "claim_manifest" in s_sc
    assert (pdir / "claims" / "manifest.yaml").exists()

    # 4. Headless LaTeX report via rosolver_report(format="latex")
    tex_rep = _call(
        srv,
        "rosolver_report",
        {"project_dir": str(pdir), "format": "latex", "title": "Full Capabilities Report"},
    )
    assert tex_rep["isError"] is False
    assert "\\documentclass" in tex_rep["structuredContent"]["latex"]
    assert (pdir / "reports" / "final_report.tex").exists()
