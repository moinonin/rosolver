"""Benchmark 3: Manufactured PDE solution T(x, t) = exp(-t) * sin(pi * x) (ENG_SPEC.md P0.2)."""

from __future__ import annotations

import sympy as sp

from rosolver import ConservationLaw1D, VerificationStatus, mms_study


class TestBenchmarkManufacturedPDE:
    def test_manufactured_heat_equation_forcing_and_order(self) -> None:
        law = ConservationLaw1D("T", diffusivity="alpha")
        exact = "exp(-t) * sin(pi * x)"
        alpha_val = 0.1

        # Verify symbolic manufactured source matches analytical derivative:
        # dT/dt - alpha * d2T/dx2 = (alpha * pi^2 - 1) * exp(-t) * sin(pi * x)
        s_mms = law.manufactured_source(exact, params={"alpha": alpha_val})
        x_s, t_s = law.x, law.t
        expected_src = (alpha_val * (sp.pi ** 2) - 1.0) * sp.exp(-t_s) * sp.sin(sp.pi * x_s)
        assert sp.simplify(s_mms["T"] - expected_src) == 0

        rep = mms_study(
            law,
            exact=exact,
            domain=(0.0, 1.0),
            t_end=0.1,
            cells=(32, 64, 128),
            params={"alpha": alpha_val},
            bc_kind="dirichlet",
            reconstruction="linear",
            scheme="BDF",
        )
        assert rep.status == VerificationStatus.PASSED
        assert rep.passed
        assert rep.order_l2["T"] >= 1.90
        assert rep.order_linf["T"] >= 1.85
        assert rep.errors_l2["T"][-1] < 1e-4
