"""
Moving-boundary (Stefan) ice-growth solver: front-fixing transform + method of lines.

Solves the one-phase Stefan problem for a growing ice layer on a cold wall,

    rho*cp * dT/dt = d/dy( k * dT/dy ),                 0 < y < delta(t)
    T(0, t)   = T_wall
    T(delta,t)= T_s(t)
    rho*Lf * ddelta/dt = k * dT/dy|_delta + h_c*(T_s - T_inf)

which is eq. 3.4 of the OTI cheat-sheet written as a complete, well-posed model. The
flux-jump (Stefan) condition is *not* integrated directly: the front is removed by the
Landau/Chaplygin coordinate change xi = y/delta(t), so the ice always occupies xi in [0,1]

    dT/dt|_xi = k/(rho*cp*delta^2) * d2T/dxi2 + (xi*delta_dot/delta) * dT/dxi
    ddelta/dt = [ (k/delta) * dT/dxi|_1 + h_c*(T_s - T_inf) ] / (rho*Lf)

with T(0)=T_wall imposed. delta(t) becomes an explicit ODE state, which keeps the front
exact (no latent-heat smearing) and lets the existing ODE/event/V&V machinery be reused.
The diffusion coefficient scales as delta^-2, so the system is stiff; use an implicit
method (`BDF`/`Radau`) or a fixed-step implicit scheme (`beuler`/`cn`).

Wet/dry (Messinger) coupling
----------------------------
Passing `mdot_imp` (impingement mass flux, kg m^-2 s^-1) turns on the rime/glaze switch:

* ``q_cap = (k/delta) * dT/dxi|_1 + h_c*(T_f - T_inf)`` is the largest freezing flux the
  surface can sustain with the surface at the freezing point.
* **Dry / rime** (``mdot_imp*Lf <= q_cap``): supply limited. All impinging water freezes,
  ``ddelta/dt = mdot_imp/rho``, and ``T_s < T_f`` follows from the algebraic Robin balance
  ``(k/delta)*dT/dxi|_1 + h_c*T_s = mdot_imp*Lf + h_c*T_inf``. No runback.
* **Wet / glaze** (``mdot_imp*Lf > q_cap``): energy limited. ``T_s = T_f``, the Stefan
  condition sets ``ddelta/dt = q_cap/(rho*Lf)``, and the unfrozen remainder runs back.

The two branches agree in ``ddelta/dt`` at the crossover (the vector field is continuous),
so the switch is an ordinary change of regime rather than a jump.

Messinger film closure
----------------------
With `tau_drain` set, a liquid-film thickness `h_f` is added as an ODE state:

    rho_w * dh_f/dt = mdot_imp + mdot_in - mdot_ice - mdot_evap - rho_w*h_f/tau_drain

and the surface energy balance also carries the impingement sensible enthalpy
``mdot_imp*cp_imp*(T_s - T_imp)``, kinetic heating ``0.5*mdot_imp*V_imp^2`` and evaporation
``mdot_evap*L_v``. Wetness is decided by the energy criterion ``mdot_imp*Lf > q_cap``; a film
is stored/runback water that can freeze back (thinner film) or accumulate. The switch is
continuous at ``q_cap = mdot_imp*Lf`` so implicit solvers remain well behaved. Freezing
fraction and drainage flux are exposed on the solution.

Hybrid accretion / shedding
---------------------------
`solve_with_shedding(events, ...)` reuses `SheddingEventSpec` with a margin on the ice
thickness (see `stefan_shedding_event`). Events are terminal; the front is reset to
``(1 - shed_fraction)*delta``, the retained inner temperature profile is resampled onto
the new grid, and integration restarts, giving alternating accretion and detachment in
one hybrid trajectory.

Validation
----------
* `neumann_delta(t)` is the exact self-similar solution for h_c = 0. Its similarity exponent
  lambda solves ``lambda * exp(lambda^2) * erf(lambda) = Ste / sqrt(pi)`` and the front is
  ``delta = 2*lambda*sqrt(alpha*t)``.
* `quasi_steady_delta(t)` is the Ste -> 0 reduction, in which the temperature profile is
  linear and ``rho*Lf*delta*delta_dot = k*(T_f - T_wall) + h_c*(T_f - T_inf)*delta``.

References
----------
* Landau, H.G. (1950). Heat conduction in a melting solid. Q. Appl. Math. 8, 81-94.
* Carslaw & Jaeger, Conduction of Heat in Solids, 2nd ed., ch. XI.
* Messinger, B.L. (1953). Equilibrium temperature of an unheated icing surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.sparse import lil_matrix

from .integrators import integrate_fixed, is_fixed_scheme
from .numerical import SheddingEventSpec, SheddingRecord

_SCIPY_IMPLICIT = ("BDF", "Radau", "LSODA")


def neumann_lambda(ste: float) -> float:
    """Similarity exponent lambda: ``lambda*exp(lambda^2)*erf(lambda) = Ste/sqrt(pi)``."""
    ste = float(ste)
    if not ste > 0.0:
        raise ValueError("Stefan number must be positive.")
    target = ste / np.sqrt(np.pi)

    def residual(lam: float) -> float:
        return lam * np.exp(lam * lam) * _erf(lam) - target

    hi = 1.0
    while residual(hi) < 0.0:
        hi *= 2.0
    return float(brentq(residual, 0.0, hi, xtol=1e-14, rtol=1e-14))


def _erf(x: float) -> float:
    from math import erf

    return erf(x)


def stefan_shedding_event(
    delta_crit: float,
    shed_fraction: float = 0.95,
    name: str = "ice_shedding",
) -> SheddingEventSpec:
    """`SheddingEventSpec` firing when the ice front exceeds `delta_crit`.

    The margin is ``delta - delta_crit`` with upward crossing, so a shed reset leaves the
    ice below the threshold and growth can start again.
    """
    crit = float(delta_crit)
    if not crit > 0.0:
        raise ValueError("delta_crit must be positive.")
    return SheddingEventSpec(
        name=name,
        margin_expr=lambda _t, y, _c=crit: float(y) - _c,
        shed_fraction=shed_fraction,
        direction=1.0,
    )


@dataclass
class StefanSolution:
    """Time history of the moving front, surface state, and ice temperature field."""

    t: np.ndarray
    delta: np.ndarray
    xi: np.ndarray
    T_xi: np.ndarray  # (n_t, n_xi + 1), includes wall and (possibly algebraic) surface value
    params: Dict[str, float]
    method: str
    n_steps: int
    nfev: int
    T_s: np.ndarray
    mdot_ice: np.ndarray
    branch: np.ndarray
    mdot_imp: Optional[float] = None
    film: Optional[np.ndarray] = None
    mdot_out: Optional[np.ndarray] = None
    q_sens: Optional[np.ndarray] = None
    n_xi: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_xi = self.xi.size - 1

    # ---- queries ---------------------------------------------------------
    def profile(self, index: int = -1) -> Tuple[np.ndarray, np.ndarray]:
        """Physical coordinate y and temperature T at snapshot `index`."""
        delta = float(self.delta[index])
        return delta * self.xi, self.T_xi[index]

    def delta_at(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """Front position at (an) arbitrary time(s) by linear interpolation."""
        return np.interp(t, self.t, self.delta)

    def temperature_at(self, y: Union[float, np.ndarray], index: int = -1) -> np.ndarray:
        """Temperature at physical depth `y` (0 <= y <= delta) for snapshot `index`."""
        yy, TT = self.profile(index)
        return np.interp(y, yy, TT)

    def front_gradient(self, index: int = -1) -> float:
        """Conductive flux k*dT/dy at the ice surface, W/m^2."""
        dxi = self.xi[1] - self.xi[0]
        T = self.T_xi[index]
        grad_xi = (3.0 * T[-1] - 4.0 * T[-2] + T[-3]) / (2.0 * dxi)
        return float(self.params["k_eff"] / self.delta[index] * grad_xi)

    def runback(self, index: int = -1) -> float:
        """Unfrozen water flux, kg m^-2 s^-1 (Messinger outflow before film drainage)."""
        if self.mdot_imp is None:
            return 0.0
        supply = self.mdot_imp + float(self.params.get("mdot_in", 0.0))
        evap = float(self.params.get("mdot_evap", 0.0))
        return max(supply - float(self.mdot_ice[index]) - evap, 0.0)

    def freezing_fraction(self, index: int = -1) -> float:
        """Messinger freezing fraction n = mdot_ice / (mdot_imp + mdot_in)."""
        if self.mdot_imp is None:
            return 1.0
        supply = self.mdot_imp + float(self.params.get("mdot_in", 0.0))
        return float(self.mdot_ice[index] / supply) if supply > 0.0 else 1.0

    def summary(self) -> str:
        d0, d1 = float(self.delta[0]), float(self.delta[-1])
        p = self.params
        lines = [
            f"=== Stefan Front-Tracking Solution ({self.method}) ===",
            f"  Ste = {p['ste']:.4f} | alpha = {p['alpha']:.4e} m^2/s | n_xi = {self.n_xi}",
            f"  Front: delta({self.t[0]:.4g} s) = {d0 * 1e3:.4f} mm -> "
            f"delta({self.t[-1]:.4g} s) = {d1 * 1e3:.4f} mm",
        ]
        if self.mdot_imp is not None:
            modes = sorted(set(str(b) for b in self.branch))
            lines.append(
                f"  Impingement = {self.mdot_imp * 1e3:.4f} g/m^2/s | modes = {modes} | "
                f"final T_s = {float(self.T_s[-1]):.4f} C"
            )
            lines.append(
                f"  Final freezing flux = {float(self.mdot_ice[-1]) * 1e3:.4f} g/m^2/s | "
                f"n = {self.freezing_fraction(-1):.4f} | "
                f"runback = {self.runback(-1) * 1e3:.4f} g/m^2/s"
            )
        if self.film is not None:
            lines.append(
                f"  Film: h_f(0) = {float(self.film[0]) * 1e3:.4f} mm -> "
                f"h_f(end) = {float(self.film[-1]) * 1e3:.4f} mm | "
                f"tau_drain = {p.get('tau_drain', float('nan')):.4f} s | "
                f"tau_freeze = {p.get('tau_freeze', float('nan')):.4f} s"
            )
        lines.append(f"  Steps: {self.n_steps} | RHS evals: {self.nfev}")
        return "\n".join(lines)


@dataclass
class HybridStefanSolution(StefanSolution):
    """Front-tracking solution with discrete shedding resets along the trajectory."""

    events: List[SheddingRecord] = field(default_factory=list)

    def summary(self) -> str:
        lines = [super().summary(), f"  Shedding events: {len(self.events)}"]
        for ev in self.events:
            lines.append(
                f"    t = {ev.time:.4g} s | {ev.mode}: "
                f"delta {ev.pre_value * 1e3:.4f} -> {ev.post_value * 1e3:.4f} mm "
                f"(shed {ev.shed_amount * 1e3:.4f} mm)"
            )
        return "\n".join(lines)


class Stefan1D:
    """
    One-phase Stefan ice-growth model with front-fixing and method-of-lines.

    Example (exact Neumann validation, h_c = 0):
        st = Stefan1D(T_wall=-6.0, T_f=0.0, h_c=0.0)
        sol = st.solve(3600.0, delta0=1e-5)
        print(sol.delta[-1], st.neumann_delta(3600.0))

    Example (rime -> glaze with a finite water supply):
        st = Stefan1D(T_wall=-4.13, T_inf=-4.13, T_f=-1.18,
                      h_c=160.0, mdot_imp=3.0e-3)
        sol = st.solve(6 * 3600.0, delta0=1e-5)
        print(sol.branch[-1], sol.runback(-1))

    Example (Messinger film + impingement enthalpy):
        st = Stefan1D(T_wall=-4.13, T_inf=-4.13, T_f=-1.18, h_c=160.0,
                      mdot_imp=6e-3, cp_imp=4186.0, tau_drain=30.0)
        sol = st.solve(3600.0, delta0=1e-5)
        print(sol.film[-1] * 1e3, sol.freezing_fraction())

    Example (accretion / shedding cycles):
        from rosolver import stefan_shedding_event
        st = Stefan1D(T_wall=-4.13, T_inf=-4.13, T_f=-1.18, h_c=160.0, mdot_imp=3e-3)
        hyb = st.solve_with_shedding([stefan_shedding_event(0.01, 0.8)], 12 * 3600.0)
        print(hyb.events)
    """

    def __init__(
        self,
        k_eff: float = 2.22,
        rho: float = 917.0,
        cp: float = 2100.0,
        Lf: float = 3.34e5,
        T_wall: float = -6.0,
        T_inf: float = -4.13,
        T_f: float = 0.0,
        h_c: float = 0.0,
        n_xi: int = 200,
        mdot_imp: Optional[float] = None,
        cp_imp: float = 0.0,
        T_imp: Optional[float] = None,
        V_imp: float = 0.0,
        mdot_evap: float = 0.0,
        mdot_in: float = 0.0,
        L_v: float = 2.5e6,
        rho_w: float = 1000.0,
        tau_drain: Optional[float] = None,
        tau_freeze: float = 1.0,
        h_film0: float = 0.0,
    ) -> None:
        for name, val in (("k_eff", k_eff), ("rho", rho), ("cp", cp), ("Lf", Lf),
                          ("L_v", L_v), ("rho_w", rho_w)):
            if not float(val) > 0.0:
                raise ValueError(f"{name} must be positive.")
        if int(n_xi) < 4:
            raise ValueError("n_xi must be at least 4 (needs two interior nodes).")
        if not T_f > T_wall:
            raise ValueError("Require T_f > T_wall for a growing ice layer (cold wall).")
        if mdot_imp is not None and float(mdot_imp) < 0.0:
            raise ValueError("mdot_imp must be non-negative.")
        for name, val in (("cp_imp", cp_imp), ("V_imp", V_imp),
                          ("mdot_evap", mdot_evap), ("mdot_in", mdot_in),
                          ("h_film0", h_film0), ("tau_freeze", tau_freeze)):
            if float(val) < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        if not float(tau_freeze) > 0.0:
            raise ValueError("tau_freeze must be positive.")
        if tau_drain is not None:
            if not float(tau_drain) > 0.0:
                raise ValueError("tau_drain must be positive.")
            if mdot_imp is None:
                raise ValueError("A runback film requires mdot_imp to be set.")

        self.k_eff = float(k_eff)
        self.rho = float(rho)
        self.cp = float(cp)
        self.Lf = float(Lf)
        self.T_wall = float(T_wall)
        self.T_inf = float(T_inf)
        self.T_f = float(T_f)
        self.h_c = float(h_c)
        self.n_xi = int(n_xi)
        self.mdot_imp = None if mdot_imp is None else float(mdot_imp)
        self.cp_imp = float(cp_imp)
        self.T_imp = float(T_inf) if T_imp is None else float(T_imp)
        self.V_imp = float(V_imp)
        self.mdot_evap = float(mdot_evap)
        self.mdot_in = float(mdot_in)
        self.L_v = float(L_v)
        self.rho_w = float(rho_w)
        self.tau_drain = None if tau_drain is None else float(tau_drain)
        self.tau_freeze = float(tau_freeze)
        self.h_film0 = float(h_film0)
        self.xi = np.linspace(0.0, 1.0, self.n_xi + 1)
        self.dxi = 1.0 / self.n_xi
        self.alpha = self.k_eff / (self.rho * self.cp)

    # ---- derived quantities ---------------------------------------------
    @property
    def film_enabled(self) -> bool:
        """Whether the liquid-film thickness is an ODE state."""
        return self.tau_drain is not None

    @property
    def stefan_number(self) -> float:
        return self.cp * (self.T_f - self.T_wall) / self.Lf

    @property
    def q_kinetic(self) -> float:
        """Impingement kinetic-energy flux dissipated at the surface, W/m^2."""
        return 0.0 if self.mdot_imp is None else 0.5 * self.mdot_imp * self.V_imp ** 2

    @property
    def q_evaporative(self) -> float:
        """Evaporative heat sink, W/m^2."""
        return self.mdot_evap * self.L_v

    def sensible_flux(self, T_s: float) -> float:
        """Heat needed to bring supercooled droplets from T_imp to T_s, W/m^2."""
        if self.mdot_imp is None or self.cp_imp == 0.0:
            return 0.0
        return self.mdot_imp * self.cp_imp * (float(T_s) - self.T_imp)

    def neumann_delta(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """Exact self-similar front position for h_c = 0 (delta = 2*lambda*sqrt(alpha*t))."""
        lam = neumann_lambda(self.stefan_number)
        return 2.0 * lam * np.sqrt(self.alpha * np.asarray(t, dtype=float))

    def quasi_steady_delta(self, t: Union[float, np.ndarray], delta0: float) -> np.ndarray:
        """
        Ste -> 0 reduction with a linear profile and delta(0) = delta0.

        rho*Lf*delta*delta_dot = k*(T_f - T_wall) + h_c*(T_f - T_inf)*delta
        which for q = delta^2 is q' = 2*kappa + 2*beta*sqrt(q), integrated here for convenience.
        """
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        scalar = np.ndim(t) == 0
        kappa = self.k_eff * (self.T_f - self.T_wall) / (self.rho * self.Lf)
        beta = self.h_c * (self.T_f - self.T_inf) / (self.rho * self.Lf)
        q0 = float(delta0) ** 2

        def rhs(_t: float, q: np.ndarray) -> np.ndarray:
            qq = max(float(q[0]), 1e-30)
            return np.array([2.0 * kappa + 2.0 * beta * np.sqrt(qq)])

        sol = solve_ivp(rhs, (0.0, float(t_arr[-1])), [q0], method="BDF",
                        t_eval=t_arr, rtol=1e-10, atol=1e-14)
        if not sol.success:
            raise RuntimeError(f"quasi-steady integration failed: {sol.message}")
        out = np.sqrt(np.maximum(sol.y[0], 0.0))
        return float(out[0]) if scalar else out

    # ---- surface closure ------------------------------------------------
    def _branch_state(
        self, T_int: np.ndarray, delta: float, h_f: float = 0.0
    ) -> Tuple[float, float, float, str, float]:
        """
        Return (T_s, ddelta/dt, mdot_ice, branch, dh_f/dt) for the current state.

        The Messinger switch is energy-limited: ``wet`` iff the water demand
        ``supply*Lf`` exceeds the heat-removal capacity at T_s = T_f. A liquid film,
        once present, is additional water available to freeze; it does *not* by itself
        pin the surface at T_f. This keeps the vector field continuous through film
        exhaustion (an ``h > 0`` criterion makes an implicit solver stall at h ~ 0).

        Surface balance (linear in T_s):

            (k/delta)*(3T_s - 4T_{M-1} + T_{M-2})/(2 dxi)
              + h_c*(T_s - T_inf) + mdot_imp*cp_imp*(T_s - T_imp)
              + mdot_evap*L_v - 0.5*mdot_imp*V_imp^2 = mdot_ice*Lf

        with supply = mdot_imp + mdot_in. At the switch ``mdot_cap = supply``, so both
        branches give T_s = T_f and ddelta/dt = supply/rho: the vector field is C^0.
        """
        T_int = np.asarray(T_int, dtype=float)
        a, b = float(T_int[-1]), float(T_int[-2])
        grad_wet = (3.0 * self.T_f - 4.0 * a + b) / (2.0 * self.dxi)
        q_cap = (self.k_eff / delta * grad_wet
                 + self.h_c * (self.T_f - self.T_inf)
                 + self.sensible_flux(self.T_f)
                 + self.q_evaporative - self.q_kinetic)
        q_cap = max(q_cap, 0.0)
        mdot_cap = q_cap / self.Lf

        if self.mdot_imp is None:
            return self.T_f, mdot_cap / self.rho, mdot_cap, "wet", 0.0

        supply = self.mdot_imp + self.mdot_in
        h = max(float(h_f), 0.0)
        # A pre-existing film supplies extra water; tau_freeze regularises its freeze-out.
        mdot_avail = (supply + self.rho_w * h / self.tau_freeze
                      if self.film_enabled else supply)
        wet = supply * self.Lf > q_cap

        if wet:
            # Energy limited: T_s = T_f and the front grows at the capacity rate.
            mdot_ice = min(mdot_cap, mdot_avail)
            T_s = self.T_f
        else:
            # Supply limited: all impinging (and, if present, film) water freezes and the
            # surface cools below T_f. T_s follows from the linear Robin balance.
            mdot_ice = min(mdot_avail, mdot_cap)
            coef = (3.0 * self.k_eff / (2.0 * delta * self.dxi)
                    + self.h_c + self.mdot_imp * self.cp_imp)
            rhs = (mdot_ice * self.Lf + self.q_kinetic - self.q_evaporative
                   + self.h_c * self.T_inf + self.mdot_imp * self.cp_imp * self.T_imp
                   + (self.k_eff / (2.0 * delta * self.dxi)) * (4.0 * a - b))
            T_s = min(rhs / coef, self.T_f)  # cap guards against round-off at the switch

        film_rate = 0.0
        if self.film_enabled:
            film_rate = ((supply - mdot_ice - self.mdot_evap
                          - self.rho_w * h / self.tau_drain) / self.rho_w)
        return T_s, mdot_ice / self.rho, mdot_ice, ("wet" if wet else "dry"), film_rate

    # ---- discretisation --------------------------------------------------
    def initial_state(self, delta0: float) -> np.ndarray:
        """Linear (quasi-steady) profile on xi[1..n_xi-1], plus delta0 and optional film."""
        delta0 = float(delta0)
        if not delta0 > 0.0:
            raise ValueError("delta0 must be positive (the transform is singular at delta = 0).")
        if self.mdot_imp is None:
            T_s = self.T_f
        else:
            # Consistent start: iterate the linear profile against the same surface
            # closure used by rhs(), including any pre-existing film water.
            h0 = self.h_film0 if self.film_enabled else 0.0
            T_s = self.T_f
            for _ in range(3):
                T_prov = self.T_wall + (T_s - self.T_wall) * self.xi[1:-1]
                T_s = self._branch_state(T_prov, delta0, h0)[0]
            T_s = min(max(T_s, self.T_wall), self.T_f)
        T = self.T_wall + (T_s - self.T_wall) * self.xi[1:-1]
        z = np.concatenate((T, [delta0]))
        if self.film_enabled:
            z = np.append(z, self.h_film0)
        return z

    def rhs(self, _t: float, z: np.ndarray) -> np.ndarray:
        m = self.n_xi - 1  # index of delta in the state vector
        T_int = np.asarray(z[:m], dtype=float)
        delta = float(z[m])
        if delta <= 0.0:
            raise ValueError("delta reached zero; front-fixing requires delta > 0.")
        h_f = float(z[m + 1]) if self.film_enabled else 0.0

        T_s, d_delta, _, _, film_rate = self._branch_state(T_int, delta, h_f)
        T = np.concatenate(([self.T_wall], T_int, [T_s]))
        # Second derivative at the interior nodes.
        d2 = (T[2:] - 2.0 * T[1:-1] + T[:-2]) / self.dxi ** 2
        # Central first derivative: mesh-motion cell Peclet stays below ~1e-3 here.
        d1 = (T[2:] - T[:-2]) / (2.0 * self.dxi)

        dT = (self.alpha / delta ** 2) * d2 + (self.xi[1:-1] * d_delta / delta) * d1
        out = np.concatenate((dT, [d_delta]))
        return np.append(out, film_rate) if self.film_enabled else out

    def jac_sparsity(self):
        """Tridiagonal interior block, dense delta/film columns, local front/film rows."""
        m = self.n_xi - 1  # number of interior temperatures; delta index
        n = m + 1 + (1 if self.film_enabled else 0)
        S = lil_matrix((n, n), dtype=int)
        for i in range(m):
            for j in (i - 1, i, i + 1):
                if 0 <= j < m:
                    S[i, j] = 1
            S[i, m] = 1  # all temperatures depend on delta
            if self.film_enabled:
                S[i, m + 1] = 1  # the surface state depends on film thickness
        for j in (m - 3, m - 2, m - 1, m):
            if j >= 0:
                S[m, j] = 1
        if self.film_enabled:
            for j in (m - 3, m - 2, m - 1, m, m + 1):
                if j >= 0:
                    S[m + 1, j] = 1
        return S.tocsr()

    # ---- drivers ---------------------------------------------------------
    def _assemble(
        self,
        t: np.ndarray,
        Z: np.ndarray,
        method: str,
        n_steps: int,
        nfev: int,
    ) -> StefanSolution:
        """Post-process a raw state history into a `StefanSolution`."""
        m = self.n_xi - 1
        delta = np.asarray(Z[:, m], dtype=float)
        if np.any(delta <= 0.0):
            raise RuntimeError("Front position became non-positive during integration.")
        film_raw = np.asarray(Z[:, m + 1], dtype=float) if self.film_enabled else None
        film = np.maximum(film_raw, 0.0) if film_raw is not None else None

        n_t = len(t)
        T_s = np.empty(n_t)
        mdot_ice = np.empty(n_t)
        branch = np.empty(n_t, dtype="<U4")
        for i in range(n_t):
            h_i = float(film_raw[i]) if film_raw is not None else 0.0
            ts_i, _, m_i, b_i, _ = self._branch_state(Z[i, :m], delta[i], h_i)
            T_s[i], mdot_ice[i], branch[i] = ts_i, m_i, b_i
        T_xi = np.column_stack(
            [np.full(n_t, self.T_wall), np.asarray(Z[:, :m], dtype=float), T_s]
        )
        mdot_out = (self.rho_w * film / self.tau_drain if film is not None else None)
        q_sens = (np.asarray([self.sensible_flux(ts) for ts in T_s])
                  if self.mdot_imp is not None else None)

        params = {
            "k_eff": self.k_eff, "rho": self.rho, "cp": self.cp, "Lf": self.Lf,
            "T_wall": self.T_wall, "T_inf": self.T_inf, "T_f": self.T_f, "h_c": self.h_c,
            "alpha": self.alpha, "ste": self.stefan_number,
            "cp_imp": self.cp_imp, "T_imp": self.T_imp, "V_imp": self.V_imp,
            "q_kinetic": self.q_kinetic, "mdot_evap": self.mdot_evap,
            "mdot_in": self.mdot_in, "L_v": self.L_v, "rho_w": self.rho_w,
        }
        if self.tau_drain is not None:
            params["tau_drain"] = self.tau_drain
            params["tau_freeze"] = self.tau_freeze
        return StefanSolution(
            t=np.asarray(t), delta=delta, xi=self.xi.copy(), T_xi=T_xi,
            params=params, method=method, n_steps=n_steps, nfev=nfev,
            T_s=T_s, mdot_ice=mdot_ice, branch=branch, mdot_imp=self.mdot_imp,
            film=film, mdot_out=mdot_out, q_sens=q_sens,
        )

    def solve(
        self,
        t_end: float,
        delta0: float = 1e-5,
        method: str = "BDF",
        n_out: int = 200,
        dt: Optional[float] = None,
        rtol: float = 1e-8,
        atol: float = 1e-12,
        t0: float = 0.0,
    ) -> StefanSolution:
        """
        Integrate the front-fixed system from t0 to t_end.

        method: any scipy solve_ivp method ('BDF', 'Radau', 'LSODA', 'RK45', ...) or a
                fixed-step scheme from `integrators` ('beuler', 'cn', 'rk4', ...). The
                latter requires `dt`. Implicit methods are strongly preferred.
        """
        t_end = float(t_end)
        t0 = float(t0)
        if not t_end > t0:
            raise ValueError("t_end must be greater than t0.")
        z0 = self.initial_state(delta0)
        times = np.linspace(t0, t_end, int(n_out) + 1)

        if is_fixed_scheme(method):
            if dt is None:
                raise ValueError(f"Fixed-step scheme '{method}' requires `dt`.")
            res = integrate_fixed(self.rhs, (t0, t_end), z0, dt, scheme=method, n_out=n_out)
            t, Z, n_steps, nfev = res.t, res.y, res.n_steps, res.nfev
        else:
            kw: Dict[str, Any] = {}
            if method in _SCIPY_IMPLICIT:
                kw["jac_sparsity"] = self.jac_sparsity()
            sol = solve_ivp(self.rhs, (t0, t_end), z0, method=method, t_eval=times,
                            rtol=rtol, atol=atol, **kw)
            if not sol.success:
                raise RuntimeError(f"Stefan integration failed: {sol.message}")
            t, Z = sol.t, sol.y.T
            n_steps, nfev = max(len(t) - 1, 0), int(sol.nfev)

        return self._assemble(t, Z, method, n_steps, nfev)

    def solve_with_shedding(
        self,
        events: Sequence[SheddingEventSpec],
        t_end: float,
        delta0: float = 1e-5,
        method: str = "BDF",
        n_out: int = 200,
        max_events: int = 50,
        rtol: float = 1e-8,
        atol: float = 1e-12,
        t0: float = 0.0,
    ) -> HybridStefanSolution:
        """
        Integrate accretion with discrete shedding resets in one hybrid trajectory.

        `events` are `SheddingEventSpec` instances whose `margin_expr` is a callable
        ``(t, delta) -> float`` (see `stefan_shedding_event`). The first upward/downward
        crossing (per the spec `direction`) terminates the current segment; delta is reset
        by the spec `shed_fraction` and integration restarts. Only solve_ivp methods are
        supported here; the fixed-step integrators have no event detection.
        """
        if not events:
            raise ValueError("At least one SheddingEventSpec is required.")
        if is_fixed_scheme(method):
            raise ValueError("solve_with_shedding requires a solve_ivp method with event support.")
        events = list(events)
        for spec in events:
            if not callable(spec.margin_expr):
                raise TypeError("Stefan shedding margins must be callables (t, delta) -> float.")

        m = self.n_xi - 1  # delta index in the state vector
        compiled = []
        for spec in events:
            def make_ev(fn, direction):
                def _ev(t_v: float, z_v: np.ndarray) -> float:
                    return float(fn(float(t_v), float(z_v[m])))

                _ev.terminal = True  # type: ignore[attr-defined]
                _ev.direction = float(direction)  # type: ignore[attr-defined]
                return _ev

            compiled.append(make_ev(spec.margin_expr, spec.direction))

        t_end, t0 = float(t_end), float(t0)
        if not t_end > t0:
            raise ValueError("t_end must be greater than t0.")
        z_curr = self.initial_state(delta0)
        t_curr = t0
        records: List[SheddingRecord] = []
        t_segs: List[np.ndarray] = []
        z_segs: List[np.ndarray] = []
        n_steps = nfev = 0
        dt_out = (t_end - t0) / max(int(n_out), 1)

        while t_curr < t_end - 1e-11 and len(records) < int(max_events):
            n_pts = max(int(np.ceil((t_end - t_curr) / max(dt_out, 1e-9))) + 1, 10)
            t_grid = np.linspace(t_curr, t_end, n_pts)
            kw: Dict[str, Any] = {}
            if method in _SCIPY_IMPLICIT:
                kw["jac_sparsity"] = self.jac_sparsity()
            sol = solve_ivp(self.rhs, (t_curr, t_end), z_curr, method=method,
                            t_eval=t_grid, events=compiled or None, rtol=rtol, atol=atol, **kw)
            if not sol.success:
                raise RuntimeError(f"Hybrid Stefan integration failed: {sol.message}")
            n_steps += max(len(sol.t) - 1, 0)
            nfev += int(sol.nfev)

            fired = [(float(arr[0]), idx) for idx, arr in enumerate(sol.t_events)
                     if arr is not None and arr.size > 0]
            if not fired:
                t_segs.append(sol.t)
                z_segs.append(sol.y.T)
                break

            t_ev, idx = min(fired, key=lambda pair: pair[0])
            pre_z = np.asarray(sol.y_events[idx][0], dtype=float)
            pre = float(pre_z[m])
            post = (1.0 - float(np.clip(events[idx].shed_fraction, 0.0, 1.0))) * pre
            seg_t = np.append(sol.t[sol.t < t_ev], t_ev)
            seg_z = np.vstack([sol.y.T[sol.t < t_ev], pre_z])
            t_segs.append(seg_t)
            z_segs.append(seg_z)
            records.append(SheddingRecord(
                time=t_ev, mode=events[idx].name, pre_value=pre,
                post_value=post, shed_amount=pre - post,
            ))
            # The shed slab is the outer layer: the retained ice is the inner
            # (1 - shed_fraction) fraction of the old profile. Resample that part
            # onto the full xi grid of the thinner layer so physical temperatures
            # are continuous across the reset.
            h_pre = float(pre_z[m + 1]) if self.film_enabled else 0.0
            T_s_pre = self._branch_state(pre_z[:m], pre, h_pre)[0]
            T_old = np.concatenate(([self.T_wall], pre_z[:m], [T_s_pre]))
            pre_z[:m] = np.interp(self.xi[1:-1] * (post / pre), self.xi, T_old)
            pre_z[m] = post
            if self.film_enabled:
                pre_z[m + 1] = 0.0  # surface water leaves with the shed slab
            z_curr = pre_z
            t_curr = t_ev + 1e-9

        t_all = np.concatenate(t_segs) if t_segs else np.array([t0])
        Z_all = np.vstack(z_segs) if z_segs else self.initial_state(delta0).reshape(1, -1)
        sol_out = self._assemble(t_all, Z_all, method, n_steps, nfev)
        return HybridStefanSolution(
            t=sol_out.t, delta=sol_out.delta, xi=sol_out.xi, T_xi=sol_out.T_xi,
            params=sol_out.params, method=sol_out.method, n_steps=sol_out.n_steps,
            nfev=sol_out.nfev, T_s=sol_out.T_s, mdot_ice=sol_out.mdot_ice,
            branch=sol_out.branch, mdot_imp=sol_out.mdot_imp, film=sol_out.film,
            mdot_out=sol_out.mdot_out, q_sens=sol_out.q_sens, events=records,
        )
