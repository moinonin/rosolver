# Rosolver MCP Server (`rosolver-mcp`)

`rosolver-mcp` exposes Rosolver's entire deterministic scientific modeling, calibration, numerical simulation, verification, validation, and reporting engine over the **Model Context Protocol (MCP)** (`stdio` JSON-RPC 2.0) with **zero extra runtime dependencies**.

> **Design Principle:** *AI proposes. Rosolver evaluates. Scientific evidence determines acceptance.*  
> Every expression is AST-validated ([rosolver/_parsing.py](../../rosolver/_parsing.py)), every YAML spec loads via `yaml.safe_load`, large array outputs are automatically summarized to protect the LLM context window, and 5-state `VerificationStatus` outcomes (`Passed`, `Failed`, `Inconclusive`, `Warning`, `NotAssessed`) are returned explicitly.

---

## 1. Starting & Registering the Server

After `pip install -e .`, three equivalent entry points are available:

```bash
rosolver-mcp
python -m rosolver.mcp
rosolver mcp
```

Inspect all registered tool schemas locally:

```bash
rosolver-mcp --list-tools
```

### MCP Client Configuration (Antigravity, Claude Desktop, Cursor, Gemini CLI)

```json
{
  "mcpServers": {
    "rosolver": {
      "command": "rosolver-mcp",
      "args": []
    }
  }
}
```

Or using a virtual environment Python interpreter directly:

```json
{
  "mcpServers": {
    "rosolver": {
      "command": ".venv/bin/python",
      "args": ["-m", "rosolver.mcp"]
    }
  }
}
```

---

## 2. AI-Native Tool Surface (10 Tools)

Unlike a raw CLI wrapper, every computational tool supports **both** instant **stateless inline payloads** (arrays, equations, YAML strings) **and** persistent SHA-256 hash-verified **`Project` workspaces** (`project_dir`).

| Tool | Purpose | Key Modes / Inputs |
| --- | --- | --- |
| `rosolver_capabilities` | Discover version, SI units, templates, schemes, & status taxonomy | `{}` |
| `rosolver_check_dimensions` | Verify equation dimensional homogeneity & extract Buckingham-$\Pi$ groups | `dims`, optional `equation`, optional `target` + `candidates` |
| `rosolver_formulate` | Qualitative dependency DAG + dimensional closure + Clairaut derivation | `relations`, `target`, `wrt`, `dims`, `constants`, `templates`, or `project_dir` |
| `rosolver_validate_model` | Deterministic AST/schema/unit/DAG & SHA-256 validator for AI-drafted models | `model_yaml` / `model_dict` / `model_path` / `project_dir`, optional `register_in_project` |
| `rosolver_calibrate` | Single-equation OLS/WLS/NLLS + `IdentifiabilityReport`, BIC candidate ranking, or sparse-$\Pi$ selection | `mode` (`single` \| `rank_candidates` \| `sparse_pi`), `data` or `data_path`, `sigma_y`, `bounds`, `loss` |
| `rosolver_simulate` | Solve `Model` specs, multi-state ODEs (+`EventSpec`), 1D FVM PDEs, moving-boundary `Stefan1D` (+shedding), or parametric curve sweeps | `solver` (`model` \| `ode` \| `pde` \| `stefan` \| `sweep`), compact `max_preview_points` |
| `rosolver_verify` | 7-engine V&V suite: Roache/Celik GCI (values, external CSV, or PDE study), SymPy MMS, timestep sensitivity, PDE/ODE conservation, stiffness, & Aitken co-simulation | `method` (`gci` \| `mms` \| `timestep` \| `conservation_pde` \| `conservation_ode` \| `stiffness` \| `coupling`) |
| `rosolver_validate_results` | Experimental `ValidationStudy` ($S \pm u_{\text{num}}$ vs $D \pm u_{\text{exp}}$ + $u_{\text{input}}$) | Inline `simulation`+`experiment`, `sim_csv`+`exp_csv`, or `project_dir`+`run_id` |
| `rosolver_project` | Workspace lifecycle, SHA-256 integrity check, ledger inspection, & exact `replay_run` | `action` (`init` \| `info` \| `verify_integrity` \| `add_model` \| `add_dataset` \| `ledger` \| `get_run` \| `replay_run`) |
| `rosolver_report` | Generate deterministic Markdown/JSON/YAML `ScientificReport` + `.manifest.json` | `project_dir`, `output`, `report_type` (8 canonical scopes), `figures` |

---

## 3. Built-In MCP Resources & Prompts

### Resources (`resources/list`, `resources/read`)
- `rosolver://schema/model` — Reference `Model` YAML specification (`text/yaml`)
- `rosolver://templates/catalog` — Registered candidate templates (`application/json`)
- `rosolver://units/supported` — Base dimensions & SI derived units (`application/json`)
- `rosolver://status/taxonomy` — 5-state `VerificationStatus` semantics & ASME V&V 20 scope (`application/json`)

### Prompts (`prompts/list`, `prompts/get`)
- `formulate_and_verify_model` — Guides an AI agent through dimensional check -> `Model` validation -> simulation -> MMS/GCI/conservation verification -> `ScientificReport`.
- `audit_external_solver` — Guides an AI agent through external CSV Roache/Celik GCI & experimental `ValidationStudy`.
