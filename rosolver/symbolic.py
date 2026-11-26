"""
Symbolic equation formulator generalizing the ch1.tex Chain-Rule + Dimensional Closure method.

Capabilities:
  1. Arbitrary Dependency DAGs (e.g. g = fn(a, b, c), b = fn(f), diamond multi-path graphs).
  2. Pure Calculus / Dimension-Free Mode (general chain rule, affine Taylor ansatz, power-law ansatz).
  3. Full Dimensional Closure Mode:
       - Exact rational monomial solver for partial derivatives beta_i = dg/dx_i
       - Buckingham-Pi dimensionless group extraction when nullity > 0
       - Clairaut exact-differential symmetry coupling (d beta_i / dx_j == d beta_j / dx_i)
       - Poincare potential line-integration + directional chain-rule integration
       - Rate-identity kinematic/dynamic closure (reproducing ch1.tex Sec 1-3)
  4. Pedagogical step-by-step trace & LaTeX generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import sympy as sp

from .dimensions import Dimension, DimensionalEngine, DimensionalSolution
from .system import RelationSpec, System
from .templates import TEMPLATE_REGISTRY


@dataclass
class DerivationStep:
    """Single human-readable / LaTeX step in an equation formulation trace."""

    stage: str
    explanation: str
    equation: Optional[sp.Eq] = None

    def to_text(self) -> str:
        if self.equation is None:
            return f"[{self.stage}] {self.explanation}"
        return f"[{self.stage}] {self.explanation}\n    {sp.pretty(self.equation, use_unicode=True)}"

    def to_latex(self) -> str:
        if self.equation is None:
            return f"\\textbf{{{self.stage}}}: {self.explanation}"
        return (
            f"\\textbf{{{self.stage}}}: {self.explanation}\n"
            f"\\begin{{equation}}\n{sp.latex(self.equation)}\n\\end{{equation}}"
        )


@dataclass
class CandidateEquation:
    """A formulated symbolic equation ready for analytical inspection or numerical fitting."""

    name: str
    kind: str  # "dimensional_exact", "dimensional_directional", "pi_power", "pi_affine", "affine_ansatz", "power_ansatz"
    equation: sp.Eq
    rhs: sp.Expr
    free_params: List[sp.Symbol]
    pi_groups: List[sp.Expr] = field(default_factory=list)
    is_exact_state: bool = True
    notes: str = ""
    dimensionally_consistent: bool = True
    assumptions: List[str] = field(default_factory=list)
    template_name: Optional[str] = None


@dataclass
class NodeResolution:
    """Resolved symbolic structure for a single DAG node."""

    node: str
    partials: Dict[str, sp.Expr]
    dim_solutions: Dict[str, DimensionalSolution]
    pi_groups: List[sp.Expr]
    potential_expr: Optional[sp.Expr]
    is_exact: bool
    free_params: List[sp.Symbol]


@dataclass
class DerivationResult:
    """Complete output of SymbolicSolver.derive()."""

    target: str
    wrt: Optional[str]
    chain_rule_general: sp.Eq
    chain_rule_active: sp.Eq
    differential_equation: Optional[sp.Eq]
    integrated_equation: sp.Eq
    node_potentials: Dict[str, sp.Expr]
    candidates: List[CandidateEquation]
    pi_groups: Dict[str, List[sp.Expr]]
    free_parameters: List[sp.Symbol]
    steps: List[DerivationStep]
    diagnostics: List[str] = field(default_factory=list)
    rejected_candidates: List[Dict[str, str]] = field(default_factory=list)

    @property
    def primary(self) -> CandidateEquation:
        return self.candidates[0]

    def summary(self) -> str:
        lines = [
            f"=== Equation Formulation: {self.target}" + (f" as fn({self.wrt})" if self.wrt else "") + " ===",
        ]
        for idx, st in enumerate(self.steps, 1):
            lines.append(f"\nStep {idx}. [{st.stage}] {st.explanation}")
            if st.equation is not None:
                lines.append(f"  {st.equation.lhs} = {st.equation.rhs}")
        if self.diagnostics:
            lines.append("\nDiagnostics:")
            for d in self.diagnostics:
                lines.append(f"  - {d}")
        lines.append("\nCandidate Formulations:")
        for c in self.candidates:
            lines.append(f"  * [{c.name}] {c.equation.lhs} = {c.equation.rhs}")
            if c.free_params:
                lines.append(f"      free params: {[str(p) for p in c.free_params]}")
        return "\n".join(lines)

    def to_latex(self) -> str:
        blocks = [st.to_latex() for st in self.steps]
        return "\n\n".join(blocks)


class SymbolicSolver:
    """General symbolic equation formulator over qualitative + dimensional DAGs."""

    def __init__(self, system: System) -> None:
        self.system = system

    def derive(
        self,
        target: str,
        wrt: Optional[str] = None,
        zero_initials: bool = False,
        unit_constants: bool = False,
        include_intercepts: Optional[bool] = None,
        include_saturating: bool = False,
        templates: Optional[Sequence[str]] = None,
    ) -> DerivationResult:
        """
        Formulate equation(s) for `target` (optionally expressed as a function of `wrt`).

        Args:
            target: Target factor name (e.g., 'g').
            wrt: Root variable to differentiate/express with respect to (e.g., 'f' or 't').
                 If omitted and target has a single leaf ancestor, auto-detects wrt; otherwise
                 derives target in terms of its direct/leaf inputs.
            zero_initials: If True, sets integration baseline offsets (g_0, b_0, ...) to 0.
            unit_constants: If True, sets leading dimensionless scaling constants kappa = 1.
            include_intercepts: Override per-relation affine rate intercept beta_0.
            include_saturating: Include saturating Pi templates (Hill, Rosin-Rammler) when
                `templates` is not explicitly specified.
            templates: Optional sequence of template registry names/aliases to apply when
                dimensionless Pi-groups exist.
        """
        if target not in self.system.variables:
            raise KeyError(f"Unknown target variable '{target}'.")
        if target not in self.system.relations:
            raise ValueError(f"Target '{target}' has no declared dependencies.")

        if wrt is not None and wrt not in self.system.variables:
            raise KeyError(f"Unknown wrt variable '{wrt}'.")

        steps: List[DerivationStep] = []
        diagnostics: List[str] = []
        rejected_candidates: List[Dict[str, str]] = []
        syms = self.system.symbols_map()

        # Step 1: Construct general & active DAG chain-rule equations
        gen_eq, act_eq = self._build_chain_rule_equations(target, wrt)
        steps.append(
            DerivationStep(
                stage="Qualitative Chain Rule",
                explanation=(
                    f"Expand total variation of '{target}'"
                    + (f" with respect to '{wrt}' across all declared factors." if wrt else " as an exact differential.")
                ),
                equation=gen_eq,
            )
        )
        if wrt is not None and gen_eq != act_eq:
            paths = self.system.paths_between(wrt, target)
            path_desc = ", ".join(" -> ".join(p) for p in paths) if paths else "none"
            steps.append(
                DerivationStep(
                    stage="Active Path Pruning",
                    explanation=(
                        f"Restrict to active dependency path(s) from '{wrt}' to '{target}' ({path_desc}); "
                        f"factors independent of '{wrt}' are held constant during differentiation "
                        f"but remain inside partial derivative coefficients."
                    ),
                    equation=act_eq,
                )
            )

        # Check if dimensions are available across the target's upstream subgraph
        sub_nodes = self.system.subgraph_order(target)
        sub_has_dims = all(
            self.system.variables[n].dim is not None
            and (
                n not in self.system.relations
                or all(self.system.variables[p].dim is not None for p in self.system.relations[n].params)
            )
            for n in sub_nodes
        )

        # Check for ch1.tex rate-identity bridge (e.g. ds/dt = v, dv/dt = a)
        if wrt is not None and sub_has_dims and (target, wrt) in self.system.rate_identities:
            ri = self.system.rate_identities[(target, wrt)]
            if ri.equals in self.system.relations or any(
                k[0] == ri.equals and k[1] == wrt for k in self.system.rate_identities
            ):
                return self._derive_via_rate_identity(
                    target=target,
                    wrt=wrt,
                    gen_eq=gen_eq,
                    act_eq=act_eq,
                    steps=steps,
                    zero_initials=zero_initials,
                    unit_constants=unit_constants,
                )

        if not sub_has_dims:
            return self._derive_dimension_free(
                target=target,
                wrt=wrt,
                gen_eq=gen_eq,
                act_eq=act_eq,
                steps=steps,
                zero_initials=zero_initials,
            )

        # Full Dimensional Closure across bottom-up DAG
        resolutions: Dict[str, NodeResolution] = {}
        node_potentials: Dict[str, sp.Expr] = {}
        pi_map: Dict[str, List[sp.Expr]] = {}
        all_free_params: List[sp.Symbol] = []

        for node in sub_nodes:
            if node not in self.system.relations:
                continue
            rel = self.system.relations[node]
            use_intercept = rel.include_intercept if include_intercepts is None else include_intercepts
            res = self._resolve_node_dimensionally(
                rel=rel,
                zero_initials=zero_initials,
                unit_constants=unit_constants,
                include_intercept=use_intercept,
                steps=steps,
                diagnostics=diagnostics,
                rejected_candidates=rejected_candidates,
            )
            resolutions[node] = res
            if res.pi_groups:
                pi_map[node] = res.pi_groups
            for fp in res.free_params:
                if fp not in all_free_params:
                    all_free_params.append(fp)
            if res.potential_expr is not None:
                node_potentials[node] = res.potential_expr

        # Compose bottom-up expressions from leaves up to target
        composed_map: Dict[str, sp.Expr] = {}
        for node in sub_nodes:
            if node in node_potentials:
                expr = node_potentials[node]
                # Substitute any already-composed upstream expressions
                subs_dict = {syms[k]: v for k, v in composed_map.items() if k != node}
                composed_map[node] = sp.simplify(expr.subs(subs_dict))

        target_sym = syms[target]
        candidates: List[CandidateEquation] = []
        diff_eq: Optional[sp.Eq] = None

        # Build directional differential equation dg/df if wrt is given
        if wrt is not None:
            wrt_sym = syms[wrt]
            target_res = resolutions[target]
            target_rel = self.system.relations[target]
            rate_terms = sp.Integer(0)
            for inp in target_rel.inputs:
                if self.system.depends_on(inp, wrt):
                    inp_expr = composed_map.get(inp, syms[inp])
                    d_inp_dwrt = sp.diff(inp_expr, wrt_sym)
                    beta_inp = target_res.partials.get(inp, sp.Symbol(f"beta_{target}_{inp}", real=True))
                    subs_up = {syms[k]: v for k, v in composed_map.items()}
                    rate_terms += beta_inp.subs(subs_up) * d_inp_dwrt

            if (target_rel.include_intercept if include_intercepts is None else include_intercepts):
                b0_sym = self._solve_intercept_term(
                    target, wrt, target_rel, unit_constants, all_free_params, diagnostics=diagnostics
                )
                rate_terms = b0_sym + rate_terms

            rate_terms = sp.simplify(rate_terms)
            diff_eq = sp.Eq(sp.Derivative(target_sym, wrt_sym), rate_terms)
            steps.append(
                DerivationStep(
                    stage="Substituted Rate Equation",
                    explanation=f"Substitute dimensionally closed partials and inner derivatives w.r.t. '{wrt}'.",
                    equation=diff_eq,
                )
            )

        # Primary integrated equation
        if target in composed_map:
            primary_rhs = composed_map[target]
        elif wrt is not None and diff_eq is not None:
            wrt_sym = syms[wrt]
            init_term = sp.Integer(0) if zero_initials else self.system.variables[target].initial_symbol
            if not zero_initials and init_term not in all_free_params:
                all_free_params.append(init_term)
            primary_rhs = init_term + sp.integrate(diff_eq.rhs, wrt_sym)
        else:
            # Fallback differential representation
            primary_rhs = sum(
                resolutions[target].partials.get(inp, sp.Integer(0)) * syms[inp]
                for inp in self.system.relations[target].inputs
            )

        primary_rhs = sp.simplify(primary_rhs)
        integrated_eq = sp.Eq(target_sym, primary_rhs)
        steps.append(
            DerivationStep(
                stage="Integrated Closed-Form Equation",
                explanation=(
                    f"Compose upstream relations and integrate to express '{target}'"
                    + (f" in terms of '{wrt}'." if wrt else " in closed form.")
                ),
                equation=integrated_eq,
            )
        )

        primary_free = self._extract_free_params(primary_rhs, all_free_params)
        target_res = resolutions[target]
        target_consistent = all(sol.consistent for sol in target_res.dim_solutions.values())
        primary_assumptions: List[str] = []
        if len(self.system.relations[target].inputs) > 1 and target_res.is_exact:
            primary_assumptions.append("Assumes C^2 scalar potential satisfying Clairaut mixed-partial symmetry.")
        if not target_consistent:
            primary_assumptions.append("Includes dimensionally unclosed fallback coefficient(s).")
        candidates.append(
            CandidateEquation(
                name="leading_dimensional",
                kind="dimensional_exact" if target_res.is_exact else "dimensional_directional",
                equation=integrated_eq,
                rhs=primary_rhs,
                free_params=primary_free,
                pi_groups=pi_map.get(target, []),
                is_exact_state=target_res.is_exact,
                notes="Leading dimensional-closure equation.",
                dimensionally_consistent=target_consistent,
                assumptions=primary_assumptions,
            )
        )

        # Generate Buckingham-Pi enriched candidates via pluggable TEMPLATE_REGISTRY
        all_pis: List[sp.Expr] = []
        for node, pis in pi_map.items():
            subs_up = {syms[k]: v for k, v in composed_map.items() if k != node}
            for pi_expr in pis:
                subbed_pi = sp.simplify(pi_expr.subs(subs_up))
                if subbed_pi not in all_pis:
                    all_pis.append(subbed_pi)

        if all_pis:
            init_sym = self.system.variables[target].initial_symbol
            core_term = primary_rhs if zero_initials else sp.simplify(primary_rhs - init_sym)
            offset_term = sp.Integer(0) if zero_initials else init_sym

            selected_templates = (
                [TEMPLATE_REGISTRY.get(t) for t in templates]
                if templates is not None
                else TEMPLATE_REGISTRY.list(include_saturating=include_saturating)
            )
            first_pi_eq: Optional[sp.Eq] = None
            added_names: List[str] = []
            for tmpl in selected_templates:
                try:
                    mod_factor, extra_syms = tmpl.build_factor(all_pis)
                    cand_rhs = offset_term + core_term * mod_factor
                    cand_eq = sp.Eq(target_sym, cand_rhs)
                    cand_free = self._extract_free_params(cand_rhs, all_free_params + extra_syms)
                    candidates.append(
                        CandidateEquation(
                            name=tmpl.candidate_name,
                            kind=tmpl.kind,
                            equation=cand_eq,
                            rhs=cand_rhs,
                            free_params=cand_free,
                            pi_groups=all_pis,
                            is_exact_state=target_res.is_exact,
                            notes=tmpl.description,
                            dimensionally_consistent=target_consistent,
                            assumptions=list(tmpl.assumptions) + primary_assumptions,
                            template_name=tmpl.name,
                        )
                    )
                    if first_pi_eq is None:
                        first_pi_eq = cand_eq
                    added_names.append(tmpl.name)
                except Exception as exc:
                    rejected_candidates.append(
                        {
                            "name": tmpl.candidate_name,
                            "template": tmpl.name,
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )

            if first_pi_eq is not None:
                steps.append(
                    DerivationStep(
                        stage="Buckingham-Pi Generalization",
                        explanation=(
                            f"Nullity > 0 yielded dimensionless group(s) {', '.join(str(p) for p in all_pis)}; "
                            f"generated template candidate(s): {', '.join(added_names)}."
                        ),
                        equation=first_pi_eq,
                    )
                )

        return DerivationResult(
            target=target,
            wrt=wrt,
            chain_rule_general=gen_eq,
            chain_rule_active=act_eq,
            differential_equation=diff_eq,
            integrated_equation=integrated_eq,
            node_potentials=node_potentials,
            candidates=candidates,
            pi_groups=pi_map,
            free_parameters=all_free_params,
            steps=steps,
            diagnostics=diagnostics,
            rejected_candidates=rejected_candidates,
        )

    def _build_chain_rule_equations(self, target: str, wrt: Optional[str]) -> Tuple[sp.Eq, sp.Eq]:
        syms = self.system.symbols_map()
        rel = self.system.relations[target]
        t_sym = syms[target]

        if wrt is None:
            dg = sp.Symbol(f"d{target}")
            rhs = sum(
                sp.Derivative(sp.Function(target)(*[syms[i] for i in rel.inputs]), syms[inp])
                * sp.Symbol(f"d{inp}")
                for inp in rel.inputs
            )
            eq = sp.Eq(dg, rhs, evaluate=False)
            return eq, eq

        w_sym = syms[wrt]
        t_fn = sp.Function(target)(*[syms[i] for i in rel.inputs])
        lhs_total = sp.Derivative(t_sym, w_sym)

        gen_terms = []
        for inp in rel.inputs:
            part = sp.Derivative(t_fn, syms[inp])
            if inp == wrt:
                gen_terms.append(part)
            else:
                gen_terms.append(part * sp.Derivative(sp.Function(inp)(w_sym), w_sym))
        gen_eq = sp.Eq(lhs_total, sum(gen_terms), evaluate=False)

        # Active recursive chain-rule expansion down to wrt
        def expand_deriv(node: str) -> sp.Expr:
            if node == wrt:
                return sp.Integer(1)
            if node not in self.system.relations:
                return sp.Integer(0)
            n_rel = self.system.relations[node]
            n_fn = sp.Function(node)(*[syms[i] for i in n_rel.inputs])
            acc = sp.Integer(0)
            for child in n_rel.inputs:
                child_d = expand_deriv(child)
                if child_d != 0:
                    acc += sp.Derivative(n_fn, syms[child]) * child_d
            return acc

        act_rhs = expand_deriv(target)
        act_eq = sp.Eq(lhs_total, act_rhs, evaluate=False)
        return gen_eq, act_eq

    def _resolve_node_dimensionally(
        self,
        rel: RelationSpec,
        zero_initials: bool,
        unit_constants: bool,
        include_intercept: bool,
        steps: List[DerivationStep],
        diagnostics: List[str],
        rejected_candidates: Optional[List[Dict[str, str]]] = None,
    ) -> NodeResolution:
        syms = self.system.symbols_map()
        target_var = self.system.variables[rel.target]
        assert target_var.dim is not None

        scope_names = list(dict.fromkeys(rel.inputs + rel.params))
        scope_pairs = [(n, self.system.variables[n].dim) for n in scope_names]  # type: ignore[misc]

        raw_monomials: Dict[str, sp.Expr] = {}
        kappa_syms: Dict[str, sp.Symbol] = {}
        dim_sols: Dict[str, DimensionalSolution] = {}
        node_pis: List[sp.Expr] = []
        free_params: List[sp.Symbol] = []

        all_consistent = True
        for inp in rel.inputs:
            inp_dim = self.system.variables[inp].dim
            assert inp_dim is not None
            needed_dim = target_var.dim / inp_dim
            sol = DimensionalEngine.solve(needed_dim, scope_pairs)  # type: ignore[arg-type]
            dim_sols[inp] = sol

            if not sol.consistent:
                all_consistent = False
                diagnostics.append(f"[{rel.target} wrt {inp}] {sol.diagnostic}")
                k_sym = sp.Symbol(f"beta_{rel.target}_{inp}", real=True)
                kappa_syms[inp] = k_sym
                raw_monomials[inp] = sp.Integer(1)
                free_params.append(k_sym)
                continue

            for pi_e in sol.pi_monomials(syms):
                if pi_e not in node_pis:
                    node_pis.append(pi_e)

            mono = sol.particular_monomial(syms)
            raw_monomials[inp] = mono
            if len(rel.inputs) == 1:
                k_sym = sp.Symbol(f"kappa_{rel.target}", real=True)
            else:
                k_sym = sp.Symbol(f"kappa_{rel.target}_{inp}", real=True)
            kappa_syms[inp] = k_sym

        # Clairaut Exactness Check & Coupling across multiple inputs
        partials: Dict[str, sp.Expr] = {}
        is_exact = True

        if all_consistent and len(rel.inputs) > 1 and rel.conservative is not False:
            coupled_partials, coupled_free, exact_ok = self._couple_clairaut_exactness(
                target=rel.target,
                inputs=rel.inputs,
                raw_monomials=raw_monomials,
                kappa_syms=kappa_syms,
                unit_constants=unit_constants,
            )
            if exact_ok:
                partials = coupled_partials
                free_params.extend(coupled_free)
                is_exact = True
            else:
                is_exact = False
                reason = f"Node '{rel.target}' leading monomials violate Clairaut mixed-partial symmetry."
                if rel.conservative is True:
                    diagnostics.append(
                        f"Node '{rel.target}' requested conservative=True, but leading monomials violate Clairaut symmetry."
                    )
                if rejected_candidates is not None:
                    rejected_candidates.append(
                        {"name": f"{rel.target}_clairaut_exact", "template": "clairaut", "reason": reason}
                    )
                for inp in rel.inputs:
                    coeff = sp.Integer(1) if unit_constants else kappa_syms[inp]
                    partials[inp] = coeff * raw_monomials[inp]
                    if not unit_constants and kappa_syms[inp] not in free_params:
                        free_params.append(kappa_syms[inp])
        else:
            if len(rel.inputs) > 1 and rel.conservative is False:
                is_exact = False
            for inp in rel.inputs:
                if not dim_sols[inp].consistent:
                    partials[inp] = kappa_syms[inp]
                else:
                    coeff = sp.Integer(1) if unit_constants else kappa_syms[inp]
                    partials[inp] = coeff * raw_monomials[inp]
                    if not unit_constants and kappa_syms[inp] not in free_params:
                        free_params.append(kappa_syms[inp])

        # Record dimensional partial-derivatives step
        part_sum = sum(partials[inp] * sp.Symbol(f"d{inp}") for inp in rel.inputs)
        steps.append(
            DerivationStep(
                stage=f"Dimensional Closure ({rel.target})",
                explanation=(
                    f"Solve [d{rel.target}/dx_i] = [{target_var.dim}]/[x_i] over scope {{{', '.join(scope_names)}}}"
                    + (" with Clairaut exactness coupling." if (len(rel.inputs) > 1 and is_exact) else ".")
                ),
                equation=sp.Eq(sp.Symbol(f"d{rel.target}"), part_sum),
            )
        )

        # Reconstruct node potential via Poincare line integration if exact & consistent
        potential_expr: Optional[sp.Expr] = None
        if all_consistent and is_exact:
            potential_expr = self._poincare_integrate(
                target=rel.target,
                inputs=rel.inputs,
                partials=partials,
                zero_initials=zero_initials,
                free_params=free_params,
            )

        return NodeResolution(
            node=rel.target,
            partials=partials,
            dim_solutions=dim_sols,
            pi_groups=node_pis,
            potential_expr=potential_expr,
            is_exact=is_exact,
            free_params=free_params,
        )

    def _couple_clairaut_exactness(
        self,
        target: str,
        inputs: Sequence[str],
        raw_monomials: Mapping[str, sp.Expr],
        kappa_syms: Mapping[str, sp.Symbol],
        unit_constants: bool,
    ) -> Tuple[Dict[str, sp.Expr], List[sp.Symbol], bool]:
        """
        Enforce d(kappa_i * M_i)/dx_j == d(kappa_j * M_j)/dx_i for all i < j.
        Returns (coupled_partials, remaining_free_kappas, exact_succeeded).
        """
        syms = self.system.symbols_map()
        k_list = [kappa_syms[inp] for inp in inputs]
        equations: List[sp.Expr] = []

        for idx_i in range(len(inputs)):
            for idx_j in range(idx_i + 1, len(inputs)):
                inp_i = inputs[idx_i]
                inp_j = inputs[idx_j]
                diff_ij = sp.diff(raw_monomials[inp_i], syms[inp_j])
                diff_ji = sp.diff(raw_monomials[inp_j], syms[inp_i])
                expr = sp.simplify(k_list[idx_i] * diff_ij - k_list[idx_j] * diff_ji)
                if expr == 0:
                    continue
                # Factor out positive variable monomials to isolate linear equation in kappas
                num, _ = sp.fraction(sp.together(expr))
                # Since variables are positive symbols, collect coefficients of k_list
                lin_comb = sp.Integer(0)
                rem = num
                for k_s in k_list:
                    c_k = sp.diff(num, k_s)
                    # Strip common variable factors across all c_k later
                    lin_comb += c_k * k_s
                    rem = sp.simplify(rem - c_k * k_s)
                if rem != 0:
                    return {}, [], False
                # Cancel nonzero monomial factors shared across terms
                factored = sp.factor(lin_comb)
                # Keep only factors that depend on at least one kappa symbol
                kappa_factors = [
                    fac for fac in (factored.args if isinstance(factored, sp.Mul) else [factored])
                    if any(fac.has(ks) for ks in k_list)
                ]
                for kf in kappa_factors:
                    equations.append(kf)

        if not equations:
            partials = {}
            free: List[sp.Symbol] = []
            for inp in inputs:
                c = sp.Integer(1) if unit_constants else kappa_syms[inp]
                partials[inp] = c * raw_monomials[inp]
                if not unit_constants:
                    free.append(kappa_syms[inp])
            return partials, free, True

        sol = sp.linsolve(equations, k_list)
        if not sol:
            return {}, [], False
        sol_tuple = list(sol)[0]
        if all(val == 0 for val in sol_tuple):
            return {}, [], False

        # Re-parameterize single shared master constant nicely as kappa_<target>
        free_in_sol = []
        for val in sol_tuple:
            for s in val.free_symbols:
                if s in k_list and s not in free_in_sol:
                    free_in_sol.append(s)

        rename_map: Dict[sp.Symbol, sp.Expr] = {}
        final_free: List[sp.Symbol] = []
        if len(free_in_sol) == 1:
            master = sp.Symbol(f"kappa_{target}", real=True)
            # Normalize so potential coefficient is clean (e.g. divide by GCD or pick smallest integer ratio)
            ratios = [sp.Rational(val.subs(free_in_sol[0], 1)) for val in sol_tuple]
            # Check integrated potential scale: if integrating x_i^(a_i) gives 1/(a_i+1), master absorbs or stays clean
            min_pos = min((abs(r) for r in ratios if r != 0), default=sp.Rational(1))
            rename_map[free_in_sol[0]] = (sp.Integer(1) / min_pos) * (sp.Integer(1) if unit_constants else master)
            if not unit_constants:
                final_free.append(master)
        else:
            for fs in free_in_sol:
                if unit_constants:
                    rename_map[fs] = sp.Integer(1)
                else:
                    final_free.append(fs)

        partials = {}
        for inp, expr_k in zip(inputs, sol_tuple):
            subbed_k = sp.simplify(expr_k.subs(rename_map))
            partials[inp] = sp.simplify(subbed_k * raw_monomials[inp])

        # Verify exactness strictly
        for idx_i in range(len(inputs)):
            for idx_j in range(idx_i + 1, len(inputs)):
                inp_i, inp_j = inputs[idx_i], inputs[idx_j]
                check = sp.simplify(sp.diff(partials[inp_i], syms[inp_j]) - sp.diff(partials[inp_j], syms[inp_i]))
                if check != 0:
                    return {}, [], False

        return partials, final_free, True

    def _poincare_integrate(
        self,
        target: str,
        inputs: Sequence[str],
        partials: Mapping[str, sp.Expr],
        zero_initials: bool,
        free_params: List[sp.Symbol],
    ) -> Optional[sp.Expr]:
        """Integrate exact differential du = sum_i beta_i dx_i sequentially across coordinates."""
        syms = self.system.symbols_map()
        acc = sp.Integer(0)
        for inp in inputs:
            x_i = syms[inp]
            residual = sp.simplify(partials[inp] - sp.diff(acc, x_i))
            term = sp.integrate(residual, x_i)
            if term.has(sp.Integral):
                return None
            acc = sp.simplify(acc + term)

        if not zero_initials:
            init_s = self.system.variables[target].initial_symbol
            if init_s not in free_params:
                free_params.append(init_s)
            acc = init_s + acc
        return acc

    def _solve_intercept_term(
        self,
        target: str,
        wrt: str,
        rel: RelationSpec,
        unit_constants: bool,
        free_params: List[sp.Symbol],
        diagnostics: Optional[List[str]] = None,
    ) -> sp.Expr:
        target_var = self.system.variables[target]
        wrt_var = self.system.variables[wrt]
        if target_var.dim is not None and wrt_var.dim is not None:
            needed_dim = target_var.dim / wrt_var.dim
            scope_names = list(dict.fromkeys(rel.inputs + rel.params))
            scope_pairs = [
                (n, self.system.variables[n].dim)
                for n in scope_names
                if self.system.variables[n].dim is not None
            ]
            sol = DimensionalEngine.solve(needed_dim, scope_pairs)  # type: ignore[arg-type]
            if sol.consistent:
                mono = sol.particular_monomial(self.system.symbols_map())
                k0 = sp.Integer(1) if unit_constants else sp.Symbol(f"kappa_{target}_0", real=True)
                if not unit_constants and k0 not in free_params:
                    free_params.append(k0)
                return sp.simplify(k0 * mono)
            if diagnostics is not None:
                diagnostics.append(f"[{target} intercept wrt {wrt}] {sol.diagnostic}")
        b0 = sp.Symbol(f"beta_{target}_0", real=True)
        if b0 not in free_params:
            free_params.append(b0)
        return b0

    def _derive_via_rate_identity(
        self,
        target: str,
        wrt: str,
        gen_eq: sp.Eq,
        act_eq: sp.Eq,
        steps: List[DerivationStep],
        zero_initials: bool,
        unit_constants: bool,
    ) -> DerivationResult:
        """
        Handle ch1.tex Sec 1-style kinematic chains where ds/dt = v and dv/dt = a:
          1. Use chain rule ds/dt = (ds/dv)*(dv/dt) => v = v_0 + beta_1 * a
          2. Solve [beta_1] = [v]/[a] over {wrt, ...} => beta_1 = kappa * t => v(t) = v_0 + kappa * a * t
          3. Integrate ds/dt = v(t) w.r.t. t => s(t) = s_0 + v_0*t + 1/2 * kappa * a * t^2
        """
        syms = self.system.symbols_map()
        ri_outer = self.system.rate_identities[(target, wrt)]
        mid_name = ri_outer.equals
        mid_sym = syms[mid_name]
        wrt_sym = syms[wrt]
        target_sym = syms[target]

        mid_var = self.system.variables[mid_name]
        wrt_var = self.system.variables[wrt]
        assert mid_var.dim is not None and wrt_var.dim is not None

        free_params: List[sp.Symbol] = []
        node_potentials: Dict[str, sp.Expr] = {}

        if (mid_name, wrt) in self.system.rate_identities:
            accel_name = self.system.rate_identities[(mid_name, wrt)].equals
            accel_var = self.system.variables[accel_name]
            assert accel_var.dim is not None
            needed = mid_var.dim / accel_var.dim
            sol = DimensionalEngine.solve(needed, [(wrt, wrt_var.dim)])
            mono = sol.particular_monomial(syms)
            k1 = sp.Integer(1) if unit_constants else sp.Symbol(f"kappa_{mid_name}", real=True)
            if not unit_constants:
                free_params.append(k1)
            mid_init = sp.Integer(0) if zero_initials else mid_var.initial_symbol
            if not zero_initials:
                free_params.append(mid_init)
            mid_expr = mid_init + k1 * mono * syms[accel_name]
            mid_eq = sp.Eq(mid_sym, mid_expr)
            node_potentials[mid_name] = mid_expr
            steps.append(
                DerivationStep(
                    stage=f"Rate-Identity Regression ({mid_name})",
                    explanation=(
                        f"Apply d{target}/d{wrt} = {mid_name} and d{mid_name}/d{wrt} = {accel_name}; "
                        f"dimensional closure [{mid_var.dim}]/[{accel_var.dim}] = [{needed}] yields:"
                    ),
                    equation=mid_eq,
                )
            )
        else:
            # Resolve mid_name via standard node resolution
            mid_rel = self.system.relations[mid_name]
            res_mid = self._resolve_node_dimensionally(
                rel=mid_rel,
                zero_initials=zero_initials,
                unit_constants=unit_constants,
                include_intercept=mid_rel.include_intercept,
                steps=steps,
                diagnostics=[],
            )
            free_params.extend(res_mid.free_params)
            mid_expr = res_mid.potential_expr if res_mid.potential_expr is not None else mid_sym
            node_potentials[mid_name] = mid_expr

        diff_eq = sp.Eq(sp.Derivative(target_sym, wrt_sym), mid_expr)
        t_init = sp.Integer(0) if zero_initials else self.system.variables[target].initial_symbol
        if not zero_initials:
            free_params.append(t_init)
        integrated_rhs = sp.expand(t_init + sp.integrate(mid_expr, wrt_sym))
        node_potentials[target] = integrated_rhs
        integrated_eq = sp.Eq(target_sym, integrated_rhs)

        steps.append(
            DerivationStep(
                stage=f"Rate Integration ({target})",
                explanation=f"Integrate d{target}/d{wrt} = {mid_name}({wrt}) with respect to '{wrt}'.",
                equation=integrated_eq,
            )
        )

        cand = CandidateEquation(
            name="rate_identity_exact",
            kind="dimensional_exact",
            equation=integrated_eq,
            rhs=integrated_rhs,
            free_params=self._extract_free_params(integrated_rhs, free_params),
            notes="Derived via rate-identity chain rule + dimensional closure.",
        )

        return DerivationResult(
            target=target,
            wrt=wrt,
            chain_rule_general=gen_eq,
            chain_rule_active=act_eq,
            differential_equation=diff_eq,
            integrated_equation=integrated_eq,
            node_potentials=node_potentials,
            candidates=[cand],
            pi_groups={},
            free_parameters=free_params,
            steps=steps,
        )

    def _derive_dimension_free(
        self,
        target: str,
        wrt: Optional[str],
        gen_eq: sp.Eq,
        act_eq: sp.Eq,
        steps: List[DerivationStep],
        zero_initials: bool,
    ) -> DerivationResult:
        """
        Formulate structural equations when NO physical dimensions are supplied:
          1. Exact chain-rule symbolic differential equation
          2. First-order affine regression closure (ch1.tex linear regression ansatz)
          3. Multiplicative power-law (constant-elasticity) closure
        """
        syms = self.system.symbols_map()
        sub_nodes = self.system.subgraph_order(target)

        # 1. Affine regression closure bottom-up
        affine_map: Dict[str, sp.Expr] = {}
        affine_free: List[sp.Symbol] = []
        for node in sub_nodes:
            if node not in self.system.relations:
                continue
            rel = self.system.relations[node]
            base = sp.Integer(0) if zero_initials else self.system.variables[node].initial_symbol
            if not zero_initials:
                affine_free.append(base)
            expr = base
            for inp in rel.inputs:
                b_sym = sp.Symbol(f"beta_{node}_{inp}", real=True)
                affine_free.append(b_sym)
                inp_val = affine_map.get(inp, syms[inp])
                expr += b_sym * inp_val
            affine_map[node] = sp.expand(expr)

        # 2. Power-law (log-linear) closure bottom-up
        power_map: Dict[str, sp.Expr] = {}
        power_free: List[sp.Symbol] = []
        for node in sub_nodes:
            if node not in self.system.relations:
                continue
            rel = self.system.relations[node]
            k_sym = sp.Symbol(f"kappa_{node}", real=True)
            power_free.append(k_sym)
            expr = k_sym
            for inp in rel.inputs:
                a_sym = sp.Symbol(f"alpha_{node}_{inp}", real=True)
                power_free.append(a_sym)
                inp_val = power_map.get(inp, syms[inp])
                expr *= inp_val ** a_sym
            power_map[node] = sp.powsimp(expr, force=True)

        t_sym = syms[target]
        aff_eq = sp.Eq(t_sym, affine_map[target])
        pow_eq = sp.Eq(t_sym, power_map[target])

        diff_eq: Optional[sp.Eq] = None
        if wrt is not None:
            w_sym = syms[wrt]
            diff_eq = sp.Eq(sp.Derivative(t_sym, w_sym), sp.diff(affine_map[target], w_sym))

        steps.append(
            DerivationStep(
                stage="Affine Regression Closure (Dimension-Free)",
                explanation="Apply first-order Taylor / linear regression ansatz across each dependency link.",
                equation=aff_eq,
            )
        )
        steps.append(
            DerivationStep(
                stage="Power-Law Scaling Closure (Dimension-Free)",
                explanation="Apply constant-elasticity (monomial) ansatz across each dependency link.",
                equation=pow_eq,
            )
        )

        cands = [
            CandidateEquation(
                name="affine_regression",
                kind="affine_ansatz",
                equation=aff_eq,
                rhs=affine_map[target],
                free_params=self._extract_free_params(affine_map[target], affine_free),
                notes="First-order affine composition without dimensional constraints.",
            ),
            CandidateEquation(
                name="power_law_scaling",
                kind="power_ansatz",
                equation=pow_eq,
                rhs=power_map[target],
                free_params=self._extract_free_params(power_map[target], power_free),
                notes="Constant-elasticity monomial composition without dimensional constraints.",
            ),
        ]

        return DerivationResult(
            target=target,
            wrt=wrt,
            chain_rule_general=gen_eq,
            chain_rule_active=act_eq,
            differential_equation=diff_eq,
            integrated_equation=aff_eq,
            node_potentials=affine_map,
            candidates=cands,
            pi_groups={},
            free_parameters=affine_free + power_free,
            steps=steps,
            diagnostics=["Dimensions omitted: generated general chain rule, affine ansatz, and power-law ansatz."],
        )

    @staticmethod
    def _extract_free_params(expr: sp.Expr, pool: Sequence[sp.Symbol]) -> List[sp.Symbol]:
        present = expr.free_symbols
        return [p for p in pool if p in present]
