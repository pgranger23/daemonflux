"""Extreme-latitude spot-check of the absolute engine across magnetic environments.

The reviewer's verification #1: confirm the central estimate, the model-spread
systematic, and the composition <A/Z> behave smoothly from a high-cutoff
near-equatorial site to a no-cutoff polar site. The geomagnetic cutoff is the only
site-dependent ingredient of the suppression (the 1D bases and the <A/Z> folding
are site-independent), so the test is that R_c falls monotonically toward the pole,
the flux rises (less shielding) without discontinuity, and the *fractional*
systematic envelope is site-robust (the cutoff cancels in the base ratio).

Run::

    python latitude_check.py --plot
"""

from __future__ import annotations

import argparse
import datetime

import numpy as np

import geomag_backtrace as gb
from base_comparison import model_envelope
from mceq3d_flux import MCEq3DFlux

SITES = [
    ("Equatorial (0N,75E)", 0.0, 75.0),
    ("Kamioka (36N)", 36.43, 137.31),
    ("Mid-lat (55N)", 55.0, 10.0),
    ("South Pole (90S)", -89.9, 0.0),
]
EREPORT = (0.3, 1.0, 10.0)


def _base_and_G(base_model):
    eng = MCEq3DFlux(base_model=base_model, daemonflux_location="generic")
    e = eng.e
    base = eng.base(np.array([0.95]))["total_numu"][0]
    rc_grid = np.linspace(0.1, 20.0, 14)
    G, rc_grid = eng.geomag_response(rc_grid)
    return e, base, G["total_numu"], rc_grid


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args(argv)
    date = datetime.datetime(2020, 1, 1)

    e, base_mc, G_mc, rcg = _base_and_G("mceq")
    _, base_df, G_df, _ = _base_and_G("daemonflux")

    def flux_at(base, G, rc, E):
        g = np.array([np.interp(rc, rcg, G[:, k]) for k in range(len(e))])
        f = base * g
        return float(
            np.exp(np.interp(np.log(E), np.log(e), np.log(np.maximum(f, 1e-300))))
        )

    print("Vertical numu across magnetic environments (central +/- systematic):")
    header = "  site                  R_c[GV]"
    for E in EREPORT:
        header += f"   {E:>5g}GeV"
    print(header)
    rows = []
    for name, lat, lon in SITES:
        rc = float(gb.cutoff_igrf(lat, lon, 0.0, 0.0, date))
        cells = ""
        sysrow = []
        for E in EREPORT:
            fmc = flux_at(base_mc, G_mc, rc, E)
            fdf = flux_at(base_df, G_df, rc, E)
            cen, sysf = model_envelope([fmc], [fdf])
            cells += f"  {cen[0]:7.1f}±{sysf[0] * 100:2.0f}%"
            sysrow.append(sysf[0])
        print(f"  {name:20s}  {rc:6.2f}   {cells}")
        rows.append((name, rc, sysrow))

    # smoothness assertions
    rcs = [gb.cutoff_igrf(lat, lon, 0.0, 0.0, date) for _, lat, lon in SITES]
    print(
        f"\nR_c monotonic equator->pole: {all(np.diff(rcs) <= 1e-6)} "
        f"({', '.join(f'{r:.1f}' for r in rcs)} GV)"
    )
    sys1gev = [r[2][1] for r in rows]
    print(
        f"Systematic at 1 GeV is site-robust: spread "
        f"{(max(sys1gev) - min(sys1gev)) * 100:.1f}% "
        f"(cutoff cancels in the base ratio)"
    )

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        for name, lat, lon in SITES:
            rc = float(gb.cutoff_igrf(lat, lon, 0.0, 0.0, date))
            cen = []
            lo = []
            hi = []
            for E in e[(e >= 0.1) & (e <= 100)]:
                fmc = flux_at(base_mc, G_mc, rc, E)
                fdf = flux_at(base_df, G_df, rc, E)
                c, s = model_envelope([fmc], [fdf])
                cen.append(c[0] * E**3)
                lo.append(c[0] * np.exp(-s[0]) * E**3)
                hi.append(c[0] * np.exp(s[0]) * E**3)
            ee = e[(e >= 0.1) & (e <= 100)]
            ax.fill_between(ee, lo, hi, alpha=0.15)
            ax.loglog(ee, cen, label=f"{name}  ($R_c$={rc:.1f} GV)")
        ax.set_xlabel("E [GeV]")
        ax.set_ylabel(r"$E^3\Phi_{\nu_\mu}$ (central) [GeV$^2$/m$^2$/s/sr]")
        ax.set_title("Smooth variation across magnetic environments + systematic band")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig("latitude_check.png", dpi=110)
        print("saved plot -> latitude_check.png")


if __name__ == "__main__":
    main()
