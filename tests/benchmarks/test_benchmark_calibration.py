"""Benchmark 5: Synthetic nonlinear parameter identification & BIC selection (ENG_SPEC.md P0.2)."""

from __future__ import annotations

import numpy as np

from rosolver import NumericalSolver, formulate


class TestBenchmarkParameterIdentification:
    def test_noisy_hill_parameter_recovery_and_bounds(self) -> None:
        res = formulate(
            relations=["m_imp = fn(v_rel; IWC, tau_p, c)"],
            dims={
                "m_imp": "M/(L^2*T)",
                "v_rel": "L/T",
                "IWC": "M/L^3",
                "tau_p": "T",
                "c": "L",
            },
            constants=["IWC", "tau_p", "c"],
            target="m_imp",
            wrt="v_rel",
            zero_initials=True,
            include_saturating=True,
        )
        rng = np.random.default_rng(42)
        n = 120
        v_rel = rng.uniform(25.0, 180.0, n)
        iwc = rng.uniform(0.5e-3, 3.5e-3, n)
        tau_p = rng.uniform(5e-5, 8e-4, n)
        chord = rng.uniform(0.04, 0.12, n)
        stokes = v_rel * tau_p / chord
        kappa_true, gam_true, nu_true = 1.0, 2.5, 1.0
        clean = kappa_true * iwc * v_rel * ((gam_true * (stokes ** nu_true)) / (1.0 + gam_true * (stokes ** nu_true)))

        hill_cand = [c for c in res.candidates if c.name == "buckingham_pi_hill"][0]

        rmses = []
        for noise_rel in (0.0, 0.005, 0.02):
            noisy = clean * (1.0 + rng.normal(0.0, noise_rel, n))
            fit = NumericalSolver.fit(
                hill_cand,
                {"m_imp": noisy, "v_rel": v_rel, "IWC": iwc, "tau_p": tau_p, "c": chord},
                bounds={
                    "kappa_m_imp": (0.1, 5.0),
                    "gam_1": (0.1, 20.0),
                    "nu_1": (0.1, 3.0),
                },
            )
            assert fit.converged
            assert 0.1 <= fit.params["kappa_m_imp"] <= 5.0
            assert 0.1 <= fit.params["gam_1"] <= 20.0
            assert 0.1 <= fit.params["nu_1"] <= 3.0
            rmses.append(fit.rmse)

        assert rmses[0] < rmses[1] < rmses[2]

        # Under low noise, BIC should rank the saturating Hill model above leading monomial & affine models
        noisy_low = clean * (1.0 + rng.normal(0.0, 0.002, n))
        ranked = NumericalSolver.rank_candidates(
            res,
            {"m_imp": noisy_low, "v_rel": v_rel, "IWC": iwc, "tau_p": tau_p, "c": chord},
            criterion="bic",
        )
        assert ranked[0].candidate_name == "buckingham_pi_hill"
        assert np.isclose(ranked[0].params["gam_1"], gam_true, rtol=0.15)
        assert np.isclose(ranked[0].params["nu_1"], nu_true, rtol=0.10)
