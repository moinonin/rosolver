"""
Stefan moving-boundary validation & wet/dry demonstration.

Generates `figures/stefan_front_validation.png`:

  (a) h_c = 0: the front-fixed PDE against the exact Neumann similarity solution and the
      quasi-steady (Ste -> 0) reduction.
  (b) finite impingement: the rime (dry, supply-limited) -> glaze (wet, energy-limited)
      transition, with the wet-limit quasi-steady reference and the surface temperature.

Run with:
    python -m examples.stefan_validation
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rosolver import Stefan1D, plot_stefan

FIG = "figures/stefan_front_validation.png"


def main() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 8.0), sharex=False)

    # ---- (a) exact validation, h_c = 0 ---------------------------------
    st_exact = Stefan1D(k_eff=2.22, rho=917.0, cp=2100.0, Lf=3.34e5,
                        T_wall=-6.0, T_f=0.0, h_c=0.0, n_xi=200)
    sol_exact = st_exact.solve(3600.0, delta0=1e-5, n_out=80)
    plot_stefan(sol_exact, st_exact, ax=axes[0], neumann=True, quasi=True)
    err = abs(sol_exact.delta[-1] - float(st_exact.neumann_delta(3600.0))) / float(
        st_exact.neumann_delta(3600.0)
    )
    axes[0].set_title(
        f"(a) h_c = 0 validation: Ste = {st_exact.stefan_number:.4f}, "
        f"front error vs Neumann = {err:.1e}"
    )

    # ---- (b) rime -> glaze with finite impingement ----------------------
    st_wd = Stefan1D(k_eff=2.22, rho=917.0, cp=2100.0, Lf=3.34e5,
                     T_wall=-4.13, T_inf=-4.13, T_f=-1.18, h_c=160.0,
                     mdot_imp=3.0e-3, n_xi=200)
    sol_wd = st_wd.solve(6 * 3600.0, delta0=1e-5, n_out=120)
    plot_stefan(sol_wd, st_wd, ax=axes[1], neumann=False, quasi=True, show_surface=True)
    wet = np.asarray(sol_wd.branch) == "wet"
    t_switch = sol_wd.t[np.argmax(wet)] / 3600.0 if wet.any() else float("nan")
    axes[1].set_title(
        f"(b) impingement {st_wd.mdot_imp * 1e3:.2f} g/m$^2$/s: "
        f"rime -> glaze at t = {t_switch:.2f} h, runback at 6 h = "
        f"{sol_wd.runback(-1) * 1e3:.2f} g/m$^2$/s"
    )
    axes[1].set_xlabel("t [h]")

    fig.tight_layout()
    from pathlib import Path

    out = Path(FIG)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")

    print("\n(a) h_c = 0 exact validation")
    print(f"    PDE delta(1 h)     = {sol_exact.delta[-1] * 1e3:.6f} mm")
    print(f"    Neumann delta(1 h) = {float(st_exact.neumann_delta(3600.0)) * 1e3:.6f} mm")
    print(f"    relative error     = {err:.2e}")

    print("\n(b) rime/glaze transition")
    for t_h in (0.5, 1.0, 2.0, 4.0, 6.0):
        i = int(np.argmin(np.abs(sol_wd.t - t_h * 3600.0)))
        print(
            f"    t={t_h:4.1f} h | delta={sol_wd.delta[i] * 1e3:8.3f} mm | "
            f"T_s={sol_wd.T_s[i]:7.3f} C | {sol_wd.branch[i]:4s} | "
            f"freeze={sol_wd.mdot_ice[i] * 1e3:6.3f} g/m2s | "
            f"runback={sol_wd.runback(i) * 1e3:6.3f} g/m2s"
        )


if __name__ == "__main__":
    main()
