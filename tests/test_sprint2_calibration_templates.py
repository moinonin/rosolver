"""Sprint 2 tests: pluggable equation templates, symbolic hardening, WLS & identifiability (P1.3, P2.1)."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from rosolver import (
    TEMPLATE_REGISTRY,
    CandidateRankingReport,
    EquationTemplate,
    NumericalSolver,
    SymbolicSolver,
    System,
    formulate,
    get_template,
    list_templates,
    register_template,
)


class TestPluggableTemplateRegistry:
    def test_builtin_templates_aliases_and_hooks(self) -> None:
        names = {t.name for t in list_templates(include_saturating=True)}
        assert {"power", "affine", "hill", "exp_sat"}.issubset(names)
        assert get_template("monomial").name == "power"
        assert get_template("rosin_rammler").name == "exp_sat"

        pi_sym = sp.Symbol("Pi_1", positive=True, real=True)
        hill = get_template("hill")
        factor, params = hill.build_factor([pi_sym])
        assert hill.bound_for("gam_1") == (1e-9, np.inf)
        vals = hill.evaluate(factor, {"Pi_1": np.array([1.0, 2.0]), "gam_1": 2.0, "nu_1": 1.0})
        assert np.allclose(vals, [2.0 / 3.0, 4.0 / 5.0])
        deriv = hill.differentiate(factor, "Pi_1")
        assert deriv != 0

    def test_custom_template_participates_in_formulate_and_sparse_pi(self) -> None:
        def _tanh_builder(pis, indices):
            c_syms = [sp.Symbol(f"gam_{idx}", positive=True, real=True) for idx in indices]
            fac = sp.Integer(1)
            for pi_e, c_s in zip(pis, c_syms):
                fac *= sp.tanh(c_s * pi_e)
            return fac, c_syms

        custom = EquationTemplate(
            name="tanh_sat",
            candidate_name="buckingham_pi_tanh_sat",
            kind="pi_tanh_sat",
            description="Hyperbolic tangent saturating Pi-template.",
            builder=_tanh_builder,
            saturating=True,
            param_bounds={"gam_": (1e-9, np.inf)},
        )
        register_template(custom)
        try:
            with pytest.raises(ValueError, match="already registered"):
                register_template(custom)

            res = formulate(
                relations=["m_imp = fn(v_rel; IWC, tau_p, c)"],
                dims={"m_imp": "M/(L^2*T)", "v_rel": "L/T", "IWC": "M/L^3", "tau_p": "T", "c": "L"},
                constants=["IWC", "tau_p", "c"],
                target="m_imp",
                wrt="v_rel",
                zero_initials=True,
                templates=["power", "tanh_sat"],
            )
            assert [c.name for c in res.candidates] == [
                "leading_dimensional",
                "buckingham_pi_power",
                "buckingham_pi_tanh_sat",
            ]

            rng = np.random.default_rng(11)
            v_rel = rng.uniform(20.0, 120.0, 50)
            iwc = rng.uniform(1e-3, 3e-3, 50)
            tau_p = rng.uniform(1e-4, 4e-4, 50)
            c = rng.uniform(0.05, 0.1, 50)
            st = v_rel * tau_p / c
            m_imp = iwc * v_rel * np.tanh(2.5 * st)
            sparse = NumericalSolver.select_sparse_pi(
                res,
                {"m_imp": m_imp, "v_rel": v_rel, "IWC": iwc, "tau_p": tau_p, "c": c},
                max_groups=1,
                form="tanh_sat",
            )
            assert sparse[0].candidate_name == "sparse_tanh_sat_Pi1"
            assert sparse[0].r2 > 0.999999
            assert np.isclose(sparse[0].params["gam_1"], 2.5, rtol=1e-4)
        finally:
            TEMPLATE_REGISTRY.unregister("tanh_sat")


class TestSymbolicHardening:
    def test_signed_variable_and_clairaut_assumptions(self) -> None:
        sys = System()
        sys.var("K", "M*L^2/T^2", positive=None)
        sys.var("m", "M", positive=True)
        sys.var("v", "L/T", positive=None)
        sys.depends("K", on=["m", "v"])
        assert sys.variables["v"].symbol.is_positive is None
        res = SymbolicSolver(sys).derive("K", zero_initials=True)
        assert any("Clairaut" in a for a in res.primary.assumptions)

    def test_intercept_term_dimensional_closure_and_diagnostic(self) -> None:
        # Consistent intercept: y = fn(x; v0) with dy/dt = kappa_y_0 * v0 + ...
        sys_ok = System()
        sys_ok.var("y", "L")
        sys_ok.var("t", "T")
        sys_ok.const("v0", "L/T")
        sys_ok.depends("y", on=["t"], params=["v0"], include_intercept=True)
        res_ok = SymbolicSolver(sys_ok).derive("y", wrt="t", zero_initials=True, unit_constants=True)
        assert res_ok.differential_equation is not None
        assert res_ok.differential_equation.rhs.has(sys_ok.variables["v0"].symbol)

        # Inconsistent intercept: no parameter carries L/T when only dimensionless x and t? Use y[L], t[T], u[L]
        sys_bad = System()
        sys_bad.var("y", "L")
        sys_bad.var("x", "L")
        sys_bad.depends("y", on=["x"], include_intercept=True)
        res_bad = SymbolicSolver(sys_bad).derive("y", wrt="x", zero_initials=True)
        # [dy/dx] = 1 -> dimensionless intercept works; test truly unclosed intercept: y[M], x[L]
        sys_unclosed = System()
        sys_unclosed.var("y", "M")
        sys_unclosed.var("x", "L")
        sys_unclosed.depends("y", on=["x"], include_intercept=True)
        res_unc = SymbolicSolver(sys_unclosed).derive("y", wrt="x", zero_initials=True)
        assert any("intercept" in d for d in res_unc.diagnostics)


class TestWeightedLeastSquaresAndMissingPolicies:
    def test_linear_wls_outperforms_ols_under_heteroscedastic_noise(self) -> None:
        x_s, a_s, b_s = sp.symbols("x a b", real=True)
        model = a_s + b_s * x_s
        rng = np.random.default_rng(101)
        n = 80
        x = np.linspace(1.0, 10.0, n)
        sigma = np.where(x < 5.0, 0.05, 5.0)
        a_true, b_true = 2.0, 1.5
        y = a_true + b_true * x + rng.normal(0.0, sigma)

        ols = NumericalSolver.fit(model, {"x": x, "y": y}, target_col="y", free_params=[a_s, b_s])
        wls = NumericalSolver.fit(
            model, {"x": x, "y": y}, target_col="y", free_params=[a_s, b_s], sigma_y=sigma
        )
        assert not ols.weighted
        assert wls.weighted
        assert wls.identifiability is not None and wls.identifiability.identifiable
        wls_err = abs(wls.params["a"] - a_true) + abs(wls.params["b"] - b_true)
        ols_err = abs(ols.params["a"] - a_true) + abs(ols.params["b"] - b_true)
        assert wls_err < ols_err
        lo_b, hi_b = wls.confidence_intervals["b"]
        assert lo_b <= b_true <= hi_b

    def test_nonlinear_wls_and_validation(self) -> None:
        x_s, a_s, k_s = sp.symbols("x a k", real=True)
        model = a_s * sp.exp(-k_s * x_s)
        rng = np.random.default_rng(7)
        x = np.linspace(0.0, 3.0, 50)
        sigma = np.where(x < 1.5, 0.01, 1.0)
        y = 4.0 * np.exp(-0.8 * x) + rng.normal(0.0, sigma)

        wls = NumericalSolver.fit(
            model,
            {"x": x, "y": y},
            target_col="y",
            free_params=[a_s, k_s],
            initial_guess={"a": 3.0, "k": 1.0},
            sigma_y=sigma,
        )
        assert wls.converged and wls.weighted
        assert np.isclose(wls.params["a"], 4.0, atol=0.1)
        assert np.isclose(wls.params["k"], 0.8, atol=0.1)

        with pytest.raises(ValueError, match="not both"):
            NumericalSolver.fit(model, {"x": x, "y": y}, target_col="y", free_params=[a_s, k_s], weights=sigma, sigma_y=sigma)
        bad_sig = sigma.copy()
        bad_sig[0] = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            NumericalSolver.fit(model, {"x": x, "y": y}, target_col="y", free_params=[a_s, k_s], sigma_y=bad_sig)

    def test_missing_data_policies(self) -> None:
        x_s, b_s = sp.symbols("x b", real=True)
        model = b_s * x_s
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        y = np.array([3.0, 6.0, 9.0, np.inf, 15.0])

        with pytest.raises(ValueError, match="Non-finite"):
            NumericalSolver.fit(model, {"x": x, "y": y}, target_col="y", free_params=[b_s], missing_policy="raise")

        fit_drop = NumericalSolver.fit(
            model, {"x": x, "y": y}, target_col="y", free_params=[b_s], missing_policy="drop"
        )
        assert fit_drop.n_obs == 3
        assert fit_drop.n_dropped_missing == 2
        assert np.isclose(fit_drop.params["b"], 3.0, atol=1e-10)

        fit_mask = NumericalSolver.fit(
            model, {"x": x, "y": y}, target_col="y", free_params=[b_s], missing_policy="mask"
        )
        assert fit_mask.n_obs == 3
        assert np.isclose(fit_mask.params["b"], 3.0, atol=1e-10)


class TestIdentifiabilityAndRejectionReporting:
    def test_structurally_unidentifiable_and_collinear_models_flagged(self) -> None:
        x_s, a_s, b_s, c_s = sp.symbols("x a b c", real=True)
        x = np.linspace(1.0, 10.0, 30)

        # 1. Linear redundant sum: y = (a + b) * x
        fit_sum = NumericalSolver.fit(
            (a_s + b_s) * x_s,
            {"x": x, "y": 5.0 * x},
            target_col="y",
            free_params=[a_s, b_s],
        )
        assert fit_sum.identifiability is not None
        assert not fit_sum.identifiability.identifiable
        assert fit_sum.identifiability.numerical_rank < 2
        assert fit_sum.admissibility_tier == "identifiability_warning"
        assert fit_sum.warnings

        # 2. Nonlinear product ambiguity: y = a * b * x
        fit_prod = NumericalSolver.fit(
            a_s * b_s * x_s,
            {"x": x, "y": 6.0 * x},
            target_col="y",
            free_params=[a_s, b_s],
            initial_guess={"a": 2.0, "b": 2.5},
        )
        assert fit_prod.identifiability is not None
        assert not fit_prod.identifiability.identifiable
        assert fit_prod.identifiability.numerical_rank < 2

        # 3. Zero-sensitivity parameter c
        fit_zero = NumericalSolver.fit(
            a_s * x_s + sp.Integer(0) * c_s,
            {"x": x, "y": 2.0 * x},
            target_col="y",
            free_params=[a_s, c_s],
        )
        assert fit_zero.identifiability is not None
        assert not fit_zero.identifiability.identifiable
        assert any("zero local sensitivity" in w for w in fit_zero.identifiability.warnings)

        # 4. Near-collinear predictors
        x1 = np.linspace(1.0, 10.0, 40)
        x2 = x1 + 1e-6 * np.sin(x1)
        x1_s, x2_s = sp.symbols("x1 x2", real=True)
        fit_col = NumericalSolver.fit(
            a_s * x1_s + b_s * x2_s,
            {"x1": x1, "x2": x2, "y": 2.0 * x1 + 3.0 * x2},
            target_col="y",
            free_params=[a_s, b_s],
        )
        assert fit_col.identifiability is not None
        assert not fit_col.identifiability.identifiable
        assert fit_col.identifiability.max_abs_correlation > 0.99

    def test_ranking_reports_capture_rejections_and_categories(self) -> None:
        # Custom template whose factor evaluates to NaN (e.g. log(-Pi)) so fitting rejects it cleanly
        def _bad_builder(pis, indices):
            p_s = sp.Symbol("gam_1", real=True)
            return sp.log(-p_s * pis[0]), [p_s]

        bad_tmpl = EquationTemplate(
            name="bad_log",
            candidate_name="buckingham_pi_bad_log",
            kind="pi_bad_log",
            description="Intentional domain-failure template for rejection reporting.",
            builder=_bad_builder,
            param_bounds={"gam_": (1.0, 2.0)},
        )
        register_template(bad_tmpl)
        try:
            res = formulate(
                relations=["delta = fn(x; rho, v, mu)"],
                dims={"delta": "L", "x": "L", "rho": "M/L^3", "v": "L/T", "mu": "M/(L*T)"},
                constants=["rho", "v", "mu"],
                target="delta",
                wrt="x",
                zero_initials=True,
                templates=["power", "bad_log"],
            )
            # Trigger a fitting failure on bad_log by making target column contain NaN only for a custom check
            # OR test rank_candidates_report directly when a candidate fails evaluation
            res.candidates[-1].rhs = sp.Symbol("missing_data_var", real=True) * res.candidates[-1].rhs

            rng = np.random.default_rng(1)
            n = 30
            x = rng.uniform(0.2, 2.0, n)
            rho = rng.uniform(0.9, 1.3, n)
            v = rng.uniform(3.0, 15.0, n)
            mu = rng.uniform(1.5e-5, 2.5e-5, n)
            delta = 5.0 * x * ((rho * v * x / mu) ** -0.5)

            # Large uncentered Re (~10^5) makes (kappa_delta, nu_1) collinear -> flagged cleanly
            rep = NumericalSolver.rank_candidates_report(
                res, {"delta": delta, "x": x, "rho": rho, "v": v, "mu": mu}
            )
            assert isinstance(rep, CandidateRankingReport)
            assert rep.ranked[0].candidate_name == "buckingham_pi_power"
            assert any(r.candidate_name == "buckingham_pi_bad_log" for r in rep.rejected)
            assert "buckingham_pi_bad_log" in rep.categories["rejected"]
            assert "leading_dimensional" in rep.categories["statistically_supported"]
            assert "buckingham_pi_power" in rep.categories["identifiability_flagged"]
            assert "Rejected candidates:" in rep.summary()

            # When Pi spans O(1) across unity (Re ~ 0.04..25), (kappa_delta, nu_1) is well-conditioned
            x_u = rng.uniform(0.2, 5.0, n)
            v_u = rng.uniform(0.2, 5.0, n)
            rho_u = np.ones(n)
            mu_u = np.ones(n)
            delta_u = 5.0 * x_u * ((rho_u * v_u * x_u / mu_u) ** -0.5)
            rep_u = NumericalSolver.rank_candidates_report(
                res, {"delta": delta_u, "x": x_u, "rho": rho_u, "v": v_u, "mu": mu_u}
            )
            assert "buckingham_pi_power" in rep_u.categories["statistically_supported"]
        finally:
            TEMPLATE_REGISTRY.unregister("bad_log")
