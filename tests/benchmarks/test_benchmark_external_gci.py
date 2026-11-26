"""Benchmark 6: Celik et al. (2008) GCI reference, edge cases & external CSV workflow (ENG_SPEC.md P0.2, P0.3)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from rosolver import VerificationStatus, gci_from_values
from rosolver.cli import main as cli_main


class TestBenchmarkExternalGCI:
    def test_celik_2008_table1_case1_reference(self) -> None:
        """
        Independent reference from Celik et al. (2008), ASME J. Fluids Eng. 130(7), Table 1:
          N = (18000, 8000, 4500), h = 1/sqrt(N), phi = (6.063, 5.972, 5.863)
          r21 = 1.5, r32 = 4/3 -> p ~ 1.53, phi_ext ~ 6.17, e_a^21 ~ 1.5%, GCI_fine^21 ~ 2.2%.
        """
        hs = [1.0 / np.sqrt(18000.0), 1.0 / np.sqrt(8000.0), 1.0 / np.sqrt(4500.0)]
        phis = [6.063, 5.972, 5.863]

        # Independent root solve of Celik equation
        r21 = hs[1] / hs[0]
        r32 = hs[2] / hs[1]
        e32 = phis[2] - phis[1]
        e21 = phis[1] - phis[0]
        p_indep = brentq(
            lambda p: p - (np.log(abs(e32 / e21)) + np.log((r21 ** p - 1.0) / (r32 ** p - 1.0))) / np.log(r21),
            0.2,
            5.0,
        )
        phi_ext_indep = (r21 ** p_indep * phis[0] - phis[1]) / (r21 ** p_indep - 1.0)
        ea21_indep = abs((phis[0] - phis[1]) / phis[0])
        gci21_indep = 1.25 * ea21_indep / (r21 ** p_indep - 1.0)

        rep = gci_from_values(hs, phis)
        assert rep.status == VerificationStatus.PASSED
        assert rep.passed
        assert rep.regime == "monotonic"
        assert np.isclose(rep.observed_order, p_indep, atol=1e-7)
        assert np.isclose(rep.extrapolated_value, phi_ext_indep, atol=1e-7)
        assert np.isclose(rep.gci_fine_21, gci21_indep, atol=1e-7)
        assert np.isclose(rep.observed_order, 1.53, atol=0.02)
        assert np.isclose(rep.extrapolated_value, 6.17, atol=0.02)
        assert np.isclose(rep.gci_fine_21, 0.022, atol=0.002)

    def test_hardened_gci_edge_cases_and_statuses(self) -> None:
        # 1. r < 1.3 emits warning and sets VerificationStatus.WARNING
        rep_small_r = gci_from_values([0.01, 0.012, 0.0144], [2.0 - 0.5 * h ** 2 for h in (0.01, 0.012, 0.0144)])
        assert rep_small_r.status == VerificationStatus.WARNING
        assert not rep_small_r.passed
        assert any("1.3" in w for w in rep_small_r.warnings)

        # 2. Oscillatory convergence emits WARNING rather than PASSED
        rep_osc = gci_from_values([0.01, 0.02, 0.04], [1.001, 0.996, 1.016])
        assert rep_osc.regime == "oscillatory"
        assert rep_osc.status == VerificationStatus.WARNING
        assert not rep_osc.passed

        # 3. Divergent sequence emits FAILED and avoids fabricating finite GCI
        rep_div = gci_from_values([0.01, 0.02, 0.04], [1.50, 1.10, 1.05])
        assert rep_div.regime == "divergent"
        assert rep_div.status == VerificationStatus.FAILED
        assert not rep_div.passed
        assert np.isnan(rep_div.gci_fine_21)

        # 4. >3 grids records all_spacings and explicit note
        hs4 = [0.01, 0.02, 0.04, 0.08]
        phis4 = [2.0 - 0.5 * h ** 2 for h in hs4]
        rep4 = gci_from_values(hs4, phis4, formal_order=2.0)
        assert rep4.all_spacings == tuple(hs4)
        assert any("4 refinement levels" in n for n in rep4.notes)
        assert rep4.status == VerificationStatus.PASSED

        # 5. Extreme observed order clamping emits explicit warning and sets WARNING
        hs_clamp = [0.25, 0.5, 1.0]
        phis_clamp = [2.0 - 0.1 * (h ** 12) for h in hs_clamp]
        rep_clamp = gci_from_values(hs_clamp, phis_clamp)
        assert rep_clamp.status == VerificationStatus.WARNING
        assert not rep_clamp.passed
        assert any("clamped" in w for w in rep_clamp.warnings)

    def test_external_csv_import_and_cli_gci(self, tmp_path: Path, capsys) -> None:
        csv_path = tmp_path / "external_mesh_study.csv"
        rows = [
            {"h": 0.04, "nusselt": 4.912},
            {"h": 0.02, "nusselt": 4.978},
            {"h": 0.01, "nusselt": 4.9945},
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["h", "nusselt"])
            writer.writeheader()
            writer.writerows(rows)

        hs, qs = [], []
        with csv_path.open("r", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                hs.append(float(row["h"]))
                qs.append(float(row["nusselt"]))

        rep = gci_from_values(hs, qs, formal_order=2.0)
        assert rep.status == VerificationStatus.PASSED
        assert np.isclose(rep.observed_order, 2.0, atol=0.05)

        h_str = ",".join(str(x) for x in hs)
        q_str = ",".join(str(x) for x in qs)
        rc = cli_main(["--gci", f"{h_str}:{q_str}", "--formal-order", "2"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Status: Passed" in out
        assert "Safety factor Fs=1.25" in out
