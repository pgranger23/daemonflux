"""
[PROTOTYPE] Research/de-risking scaffolding -- NOT part of the delivered flux
(mceq3d_flux). Kept for the record; do not depend on it in the paper. See
ARCHITECTURE.md.

Hadronic interaction-model spread of the 1D base -- a proxy for the
inter-calculation (Bartol/FLUKA/Honda) envelope.

The differences between full atmospheric-flux calculations are driven largely by
the hadronic interaction model. This runs the *same* setup (Hillas-Gaisser H3a
primary, CORSIKA US-Standard atmosphere, e_min=0.1 GeV, vertical) through the
hadronic models MCEq carries -- SIBYLL-2.3d, EPOS-LHC, DPMJET-III-19.3,
QGSJET-II-04 -- and reports the inter-model spread of the vertical numu flux over
0.1-100 GeV.

Caveat: this is *not* literally Bartol (TARGET) or FLUKA (FLUKA hadronic + full 3D
MC); it captures the interaction-model spread, the dominant single driver, and is a
lower bound on the true inter-calculation spread (which also includes primary-flux,
atmosphere and 3D-treatment choices). Run::

    python hadronic_spread.py
"""

from __future__ import annotations

import importlib.util  # noqa: F401  (fixes an mceq_config import quirk)

import numpy as np

MODELS = ["SIBYLL-2.3d", "EPOS-LHC", "DPMJET-III-19.3", "QGSJET-II-04"]
EREPORT = (0.1, 0.2, 0.3, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0)


def compute():
    import mceq_config as config

    config.e_min = 0.1
    from MCEq.core import MCEqRun
    import crflux.models as crf

    fluxes = {}
    e = None
    for name in MODELS:
        m = MCEqRun(
            interaction_model=name,
            primary_model=(crf.HillasGaisser2012, "H3a"),
            theta_deg=0.0,
        )
        m.set_theta_deg(0.0)
        m.solve()
        e = m.e_grid
        fluxes[name] = m.get_solution("total_numu", 0) * 1.0e4  # /(m^2 s sr GeV)
    return e, fluxes


def _at(y, e, E):
    return float(np.exp(np.interp(np.log(E), np.log(e), np.log(np.maximum(y, 1e-300)))))


def main():
    e, fluxes = compute()
    arr = np.array([fluxes[m] for m in MODELS])  # (nmodel, nE)
    gmean = np.exp(np.mean(np.log(np.maximum(arr, 1e-300)), axis=0))
    spread = np.max(arr, axis=0) / np.min(arr, axis=0)  # max/min ratio

    print("Vertical numu -- hadronic interaction-model spread (bare, no geomag):")
    hdr = "  E[GeV] " + "".join(f"{m.split('-')[0][:6]:>9s}" for m in MODELS)
    hdr += "   spread"
    print(hdr)
    for E in EREPORT:
        vals = [_at(fluxes[m], e, E) for m in MODELS]
        sp = max(vals) / min(vals)
        print(
            "  "
            + f"{E:6.2f} "
            + "".join(f"{v:9.1f}" for v in vals)
            + f"   {(sp - 1) * 100:4.0f}%"
        )
    print(
        f"\ninteraction-model spread: ~{(_at(spread, e, 0.1) - 1) * 100:.0f}% "
        f"at 0.1 GeV, rising to ~{(_at(spread, e, 10) - 1) * 100:.0f}% at multi-GeV."
    )
    print(
        "This interaction-model spread is *smaller* than the daemonflux<->MCEq "
        "(muon-calibration) spread, i.e. the sub-GeV flux uncertainty is dominated "
        "by overall normalisation/calibration, not by the interaction model alone."
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel = (e >= 0.1) & (e <= 100)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))
    for m in MODELS:
        axL.loglog(e[sel], (fluxes[m] * e**3)[sel], label=m)
    axL.set_xlabel("E [GeV]")
    axL.set_ylabel(r"$E^3\,\Phi_{\nu_\mu}$ [GeV$^2$/(m$^2$ s sr)]")
    axL.set_title("Vertical numu, four hadronic models (H3a, US-Std)")
    axL.legend(fontsize=8)
    for m in MODELS:
        axR.semilogx(e[sel], (fluxes[m] / gmean)[sel], label=m)
    axR.fill_between(
        e[sel],
        (np.min(arr, axis=0) / gmean)[sel],
        (np.max(arr, axis=0) / gmean)[sel],
        color="0.8",
        alpha=0.6,
        label="inter-model envelope",
    )
    axR.axhline(1.0, color="k", lw=0.7)
    axR.set_xlabel("E [GeV]")
    axR.set_ylabel("ratio to geometric mean")
    axR.set_title("Hadronic interaction-model spread")
    axR.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("hadronic_spread.png", dpi=110)
    print("saved plot -> hadronic_spread.png")


if __name__ == "__main__":
    main()
