# Quickstart

## 1. Installation

```bash
pip install -e ".[dev]"
```

Optional extras:

- `pip install -e ".[io]"` — adds `pandas` and `h5py` for DataFrame and HDF5 dataset adapters.
- `pip install -e ".[gui]"` — optional PySide6 dependency declaration (`v0.3+` desktop roadmap).
- `pip install -e ".[pdf]"` — optional `weasyprint` dependency declaration (`v0.3+` PDF reporting).

## 2. Project Workflow via Python API

```python
from rosolver import Model, Project

model = Model.from_file("examples/heat_transfer_study/model.yaml")
project = Project.create("outputs/my_heat_study", name="Heat Conduction")
project.add_model(model)

sym_res = project.run_symbolic(target="T", wrt="t", zero_initials=True)
sim_res = project.run_simulation(t_span=(0.0, 0.5), n_cells=64, method="cn")
report = project.report(output="reports/final_report.md", also_manifest=True)
print(report.status.value)
```

## 3. Workflow Subcommand CLI

All subcommands support `--json` for CI automation and emit explicit exit codes (`0` passed, `1` computational/V&V check failed or runtime error, `2` schema/validation/hash integrity error):

```bash
rosolver init my_study --model examples/heat_transfer_study/model.yaml
rosolver validate my_study
rosolver model analyse my_study --target T --wrt t
rosolver simulate my_study --t-end 0.5 --n-cells 64
rosolver verify my_study --method all --t-end 0.2
rosolver report my_study --output reports/final_report.md --manifest
rosolver info my_study --json
```

Legacy flat flags remain supported (`rosolver --rel ...`, `rosolver --gci ...`, `rosolver --sweep ...`, `rosolver --plot ...`).

## 4. Flagship End-to-End Study

Run the complete 9-stage heat-transfer study from [examples/heat_transfer_study/run_study.py](../../examples/heat_transfer_study/run_study.py):

```bash
python -m examples.heat_transfer_study.run_study --output outputs/heat_transfer_study
```
