# Agent Guidelines & Repository Operational Rules (`AGENTS.md`)

This repository (`rosolver`) is governed by operational rules designed for autonomous coding agents (Antigravity, Cursor, Claude Code, Gemini CLI, Codex, etc.).

---

## 1. Strict Privacy & Path Rule: No Absolute User Paths

> [!CRITICAL]
> **NEVER HARDCODE ABSOLUTE SYSTEM PATHS OR USER HOME DIRECTORIES**  
> Under no circumstances may an agent write, generate, or commit absolute local file system paths containing usernames or host directory structures (e.g., `/Users/<username>/...`, `file:///Users/<username>/...`, `/home/<username>/...`, or `C:\Users\<username>\...`).

### Path Formatting Rules
1. **Always Use Repository-Relative Paths**:
   - In all `.md` files, documentation, benchmarks, code comments, and test scripts, refer to files using relative paths from the repository root:
     - ✅ `docs/architecture/ENG_SPEC.md`
     - ✅ `examples/heat_transfer_study/run_study.py`
     - ✅ `tests/test_rosolver.py`
     - ❌ `file:///Users/username/...`
     - ❌ `/Users/username/...`
2. **Markdown Hyperlinks**:
   - Use relative repository links: `[Label](path/to/file.py)` or `[Label](path/to/file.py#L10-L20)`.
   - Never prefix markdown links with `file:///` protocols containing local disk paths.
3. **Executable Scripts & Tests**:
   - Resolve repository or sibling file paths dynamically:
     ```python
     from pathlib import Path
     REPO_ROOT = Path(__file__).resolve().parent.parent
     ```

### Strict Command & Filesystem Scoping: Zero Global Disk Scans
- **NEVER run global root filesystem searches** (`find /`, `grep -rn ... /`, `ls /`, etc.).
- Global root searches crawl OS volumes, snapshot backups, and external drives, hanging agent execution indefinitely.
- **Strictly scope all shell and search commands to the local workspace (`.`) or virtual environment (`.venv`)**:
  - ✅ `find . -name "*.py" -not -path "*/.*"`
  - ✅ `rg -n "symbol" rosolver/ tests/`
  - ✅ `.venv/bin/pip list`
  - ❌ `find / -name "rosolver"`
  - ❌ `grep -r "verify" /`

---

## 2. Engineering Rigor, Security & Surgical Edits

1. **No Assumptions & No Simplifications**:
   - Never assume, simplify, or water down scientific, numerical, or V&V requirements because a task seems complex.
   - Never replace real mathematical, numerical, calibration, or verification logic with hardcoded stubs or dummy scalars.
   - Write tests that assert meaningful, variable outputs for varied inputs and edge cases.
2. **Surgical File Modifications**:
   - When modifying existing modules, **do NOT regenerate entire files** unnecessarily; make targeted, minimal edits that preserve unrelated comments, docstrings, and backwards-compatible public exports (`rosolver/__init__.py`).
3. **Parsing & Serialization Security**:
   - Never call raw `sympy.sympify()` or `eval()`/`exec()` on untrusted string inputs; use `rosolver._parsing.parse_expr`.
   - Always use `yaml.safe_load` / `yaml.safe_dump` for YAML and `allow_pickle=False` for NumPy `.npz` loads.
   - Never commit credentials, secrets, or generated `outputs/` artifacts.

---

## 3. Pre-Completion Verification Gate

Before completing any task or ending your turn:
1. Run the test suite headlessly (`pytest -v` or `pytest -q` on affected suites + full suite before final completion).
2. Do NOT start blocking background servers, interactive GUI windows, or watch streaming logs (`tail -f`).
3. Verify zero absolute user path leaks in modified code/documentation and ensure all tests pass with exit code `0`.

---

## 4. Communication Style (Caveman Mode)

- Respect global user configuration: keep interactive responses concise, direct, and terse (`/caveman full`).
- Eliminate conversational preamble, repetition, and boilerplate while keeping code, docstrings, and technical documentation clear and complete.
