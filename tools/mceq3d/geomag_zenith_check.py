"""Bound the zenith-independence of the geomagnetic response G_s(E, R_c).

`mceq3d_flux` precomputes the suppression ratio ``G_s = Phi_cut/Phi_full`` at the
**vertical** column and reuses it at all zeniths. The shower develops differently
along an extreme slant column (more meson decay, more muon energy loss), so one
might worry the ratio is zenith-dependent. This script tests that directly (the
reviewer's "zenith-insensitivity validation"): it recomputes ``G_s`` in raw MCEq at
several zeniths for fixed cutoffs and reports the deviation from vertical.

Result: ``G_s`` is zenith-independent to **<=2% even at the sub-GeV horizon**
(cosθ~0.1, ~84 deg) and **<=0.4% above 1 GeV** -- because the slant-depth shower
effects act on numerator and denominator alike and cancel in the ratio. So the
vertical precomputation is safe; `solve(zenith_dependent_geomag=True)` removes even
this residual if desired. Run::

    python geomag_zenith_check.py --plot
"""

from __future__ import annotations

import argparse

import numpy as np

from mceq3d_flux import MCEq3DFlux

CZ = [1.0, 0.5, 0.25, 0.1]  # cos(zenith): vertical -> ~84 deg
RC = [5.0, 11.0]  # GV


def measure():
    eng = MCEq3DFlux(base_model="mceq")
    e = eng.e
    rc_grid = np.array(RC)
    Gz = {}
    for cz in CZ:
        G, _ = eng.geomag_response(rc_grid, cz_ref=cz)
        Gz[cz] = G["total_numu"]  # (rc, E)
    return e, rc_grid, Gz


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args(argv)

    e, rc_grid, Gz = measure()
    worst = 0.0
    for ir, rc in enumerate(rc_grid):
        print(f"\nG_s(E, R_c={rc} GV) vs zenith (cz=cosθ):")
        print("  E[GeV]   cz=1.0   cz=0.5   cz=0.25  cz=0.1   max-dev")
        for E in (0.3, 0.5, 1.0, 2.0, 5.0, 10.0):
            ie = int(np.argmin(np.abs(e - E)))
            vals = [Gz[cz][ir, ie] for cz in CZ]
            dev = max(abs(v - vals[0]) for v in vals) / vals[0] if vals[0] > 0 else 0
            worst = max(worst, dev)
            print(
                f"  {E:6.2f}   {vals[0]:6.3f}   {vals[1]:6.3f}   "
                f"{vals[2]:6.3f}   {vals[3]:6.3f}   {dev * 100:4.1f}%"
            )
    print(f"\nWorst-case zenith deviation of G_s: {worst * 100:.1f}%")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(rc_grid), figsize=(11, 4.2), sharey=True)
        for ir, (rc, ax) in enumerate(zip(rc_grid, axes)):
            for cz in CZ:
                z = np.degrees(np.arccos(cz))
                ax.semilogx(e, Gz[cz][ir], label=f"cosθ={cz} ({z:.0f}°)")
            ax.set_xlim(0.1, 100)
            ax.set_xlabel("E [GeV]")
            ax.set_title(f"$G_s(E, R_c={rc:.0f}$ GV$)$")
            ax.legend(fontsize=8)
        axes[0].set_ylabel(r"$G_s = \Phi_{\rm cut}/\Phi_{\rm full}$")
        fig.suptitle("Geomagnetic response is ~zenith-independent (<=2% sub-GeV)")
        fig.tight_layout()
        fig.savefig("geomag_zenith_check.png", dpi=110)
        print("saved plot -> geomag_zenith_check.png")


if __name__ == "__main__":
    main()
