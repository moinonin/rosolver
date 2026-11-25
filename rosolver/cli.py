"""
Command-line interface for rosolver.

Usage examples:
  # Pure symbolic chain-rule + ansatz (no dimensions required):
  python -m rosolver --rel "g = fn(a, b, c)" --rel "b = fn(f)" --target g --wrt f

  # With physical dimensions:
  python -m rosolver \
    --rel "g = fn(a, b, c)" --rel "b = fn(f; grav)" \
    --dim g="M*L^2/T^3" --dim a="M/L^3" --dim b="L/T" --dim c="L^2" --dim f="L" --dim grav="L/T^2" \
    --const grav --target g --wrt f --zero-initials
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .numerical import NumericalSolver
from .symbolic import SymbolicSolver
from .system import System


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rosolver",
        description="Formulate symbolic equations and calibrate numerical models from dependency graphs.",
    )
    p.add_argument(
        "--rel",
        action="append",
        default=[],
        required=True,
        help="Qualitative dependency relation, e.g. 'g = fn(a, b, c)' or 'b = fn(f; grav)'. Repeatable.",
    )
    p.add_argument(
        "--dim",
        action="append",
        default=[],
        help="Dimension assignment 'var=DIM', e.g. 'g=M*L^2/T^3'. Repeatable.",
    )
    p.add_argument(
        "--const",
        action="append",
        default=[],
        help="Mark identifier as governing constant parameter. Repeatable.",
    )
    p.add_argument(
        "--rate",
        action="append",
        default=[],
        help="Rate identity 'target,wrt,equals' meaning d(target)/d(wrt)=equals (e.g. 's,t,v'). Repeatable.",
    )
    p.add_argument("--target", required=True, help="Target variable to formulate.")
    p.add_argument("--wrt", default=None, help="Independent variable to express target with respect to.")
    p.add_argument("--zero-initials", action="store_true", help="Omit integration baseline offsets (x_0 = 0).")
    p.add_argument("--unit-constants", action="store_true", help="Set dimensionless scaling constants kappa = 1.")
    p.add_argument("--latex", action="store_true", help="Output derivation trace in LaTeX.")
    p.add_argument("--csv", default=None, help="Optional CSV path to fit & rank candidate equations numerically.")
    return p


def _load_csv_columns(path: Path) -> Dict[str, np.ndarray]:
    cols: Dict[str, List[float]] = {}
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            for k, v in row.items():
                if k is None or v is None:
                    continue
                cols.setdefault(k.strip(), []).append(float(v))
    return {k: np.asarray(vals, dtype=float) for k, vals in cols.items()}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    dims: Dict[str, str] = {}
    for item in args.dim:
        if "=" not in item:
            parser.error(f"Invalid --dim '{item}'; expected 'name=DIM'.")
        k, v = item.split("=", 1)
        dims[k.strip()] = v.strip()

    rates: List[Tuple[str, str, str]] = []
    for r_item in args.rate:
        parts = [x.strip() for x in r_item.split(",")]
        if len(parts) != 3:
            parser.error(f"Invalid --rate '{r_item}'; expected 'target,wrt,equals'.")
        rates.append((parts[0], parts[1], parts[2]))

    sys = System.from_spec(
        relations=args.rel,
        dims=dims if dims else None,
        constants=args.const,
        rate_identities=rates if rates else None,
    )
    result = SymbolicSolver(sys).derive(
        target=args.target,
        wrt=args.wrt,
        zero_initials=args.zero_initials,
        unit_constants=args.unit_constants,
    )

    if args.latex:
        print(result.to_latex())
    else:
        print(result.summary())

    if args.csv:
        data = _load_csv_columns(Path(args.csv))
        ranked = NumericalSolver.rank_candidates(result, data, target_col=args.target)
        print("\n=== Numerical Calibration & BIC Ranking ===")
        for idx, fit_res in enumerate(ranked, 1):
            print(f"{idx}. {fit_res.summary()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
