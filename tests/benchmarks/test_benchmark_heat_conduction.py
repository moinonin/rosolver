"""Benchmark 2: 1D transient heat conduction (Fourier exact) & Stefan Neumann order (ENG_SPEC.md P0.2)."""

from __future__ import annotations

import numpy as np

from rosolver import ConservationLaw1D, Stefan1D, VerificationStatus, audit_pde_conservation, cell_average, gci_from_values


def _exact_heat_fourier(x: np.ndarray, t: float, alpha: float) -> np.ndarray:
    return (
        np.exp(-alpha * (np.pi ** 2) * t) * np.sin(np.pi * x)
        + 0.5 * np.exp(-9.0 * alpha * (np.pi ** 2) * t) * np.sin(3.0 * np.pi * x)
    )


def _exact_total_energy(t: float, alpha: float) -> float:
    return float(
        (2.0 / np.pi) * np.exp(-alpha * (np.pi ** 2) * t)
        + (1.0 / (3.0 * np.pi)) * np.exp(-9.0 * alpha * (np.pi ** 2) * t)
    )


class TestBenchmarkHeatConduction:
    def test_spatial_convergence_against_fourier_series(self) -> None:
        alpha = 0.05
        t_end = 0.25
        law = ConservationLaw1D("T", diffusivity="alpha")
        cells = (32, 64, 128)
        hs = np.array([1.0 / n for n in cells])
        l2_errs = []
        linf_errs = []

        for n in cells:
            sol = law.solve(
                (0.0, 1.0),
                n,
                t_end=t_end,
                initial="sin(pi * x) + 0.5 * sin(3 * pi * x)",
                bc=("dirichlet", 0.0),
                params={"alpha": alpha},
                scheme="BDF",
                rtol=1e-10,
                atol=1e-12,
                n_out=5,
            )
            assert sol.success
            exact_bar = cell_average(lambda x: _exact_heat_fourier(x, t_end, alpha), sol.edges)
            diff = np.abs(sol.final("T") - exact_bar)
            l2_errs.append(float(np.sqrt(sol.h * np.sum(diff ** 2))))
            linf_errs.append(float(np.max(diff)))

        order_l2, _ = np.polyfit(np.log(hs), np.log(l2_errs), 1)
        order_linf, _ = np.polyfit(np.log(hs), np.log(linf_errs), 1)
        assert 1.90 <= order_l2 <= 2.10
        assert 1.85 <= order_linf <= 2.15
        assert l2_errs[-1] < 1e-4

    def test_temporal_convergence_beuler_and_cn(self) -> None:
        alpha = 0.05
        t_end = 0.2
        law = ConservationLaw1D("T", diffusivity="alpha")
        ref = law.solve(
            (0.0, 1.0),
            64,
            t_end=t_end,
            initial="sin(pi * x) + 0.5 * sin(3 * pi * x)",
            bc=("dirichlet", 0.0),
            params={"alpha": alpha},
            scheme="BDF",
            rtol=1e-11,
            atol=1e-13,
            n_out=2,
        )
        dts = np.array([0.04, 0.02, 0.01])

        for scheme, lo, hi in (("beuler", 0.85, 1.15), ("cn", 1.85, 2.15)):
            errs = []
            for dt in dts:
                sol = law.solve(
                    (0.0, 1.0),
                    64,
                    t_end=t_end,
                    initial="sin(pi * x) + 0.5 * sin(3 * pi * x)",
                    bc=("dirichlet", 0.0),
                    params={"alpha": alpha},
                    scheme=scheme,
                    dt=float(dt),
                    n_out=2,
                )
                assert sol.success and sol.newton_failures == 0
                diff = sol.final("T") - ref.final("T")
                errs.append(float(np.sqrt(sol.h * np.sum(diff ** 2))))
            order_t, _ = np.polyfit(np.log(dts), np.log(errs), 1)
            assert lo <= order_t <= hi

    def test_independent_analytical_energy_balance(self) -> None:
        alpha = 0.05
        t_end = 0.25
        law = ConservationLaw1D("T", diffusivity="alpha")
        sol = law.solve(
            (0.0, 1.0),
            128,
            t_end=t_end,
            initial="sin(pi * x) + 0.5 * sin(3 * pi * x)",
            bc=("dirichlet", 0.0),
            params={"alpha": alpha},
            scheme="BDF",
            rtol=1e-10,
            atol=1e-12,
            n_out=10,
        )
        # Compare domain integral directly against exact analytical energy decay (independent of FVM ledger)
        num_delta_q = sol.integral("T", -1) - sol.integral("T", 0)
        exact_delta_q = _exact_total_energy(t_end, alpha) - _exact_total_energy(0.0, alpha)
        assert np.isclose(num_delta_q, exact_delta_q, rtol=5e-4, atol=1e-5)

        # Ledger audit explicitly identifies its scope as accounting consistency
        audit = audit_pde_conservation(sol)
        assert audit.status == VerificationStatus.PASSED
        assert audit.verification_scope == "accounting_consistency"

    def test_stefan_neumann_spatial_convergence_order(self) -> None:
        delta0 = 1e-5
        t_end = 1800.0
        n_xis = (40, 80, 160)
        deltas = []
        errs = []

        for n_xi in n_xis:
            st = Stefan1D(k_eff=2.22, rho=917.0, cp=2100.0, Lf=3.34e5, T_wall=-6.0, T_f=0.0, h_c=0.0, n_xi=n_xi)
            sol = st.solve(t_end, delta0=delta0, method="BDF", rtol=1e-11, atol=1e-14, n_out=5)
            assert sol.success
            deltas.append(float(sol.delta[-1]))
            exact = float(st.neumann_delta(t_end))
            errs.append(abs(float(sol.delta[-1]) - exact))

        assert errs[2] < errs[1] < errs[0]
        assert errs[-1] / exact < 1e-5

        gci = gci_from_values([1.0 / n for n in n_xis], deltas, formal_order=2.0)
        assert gci.status == VerificationStatus.PASSED
        assert np.isclose(gci.observed_order, 2.0, atol=0.1)
