"""Compare the 1D-base choice (MCEq vs daemonflux) against Honda, 0.1-100 GeV.

The sub-GeV-to-few-GeV region is where atmospheric-oscillation physics lives and
where atmospheric-flux models disagree most. This scans the absolute numu flux
ratio to Honda for both ``base_model`` options on a dense grid, at Kamioka
vertical (down-going), so the trustworthy energy range of each base is explicit:

* daemonflux (muon-calibrated) is the better base for E >~ 1 GeV (~10 %), but
  extrapolates above Honda below ~0.3 GeV (beyond its muon-calibration region);
* raw MCEq (SIBYLL23D+H3a) runs ~25-30 % low over 0.3-10 GeV but is closer at
  0.1-0.2 GeV.

Honda itself sits between the two over 0.3-1 GeV: that band is the genuine,
irreducible sub-GeV flux uncertainty. Run::

    python base_comparison.py
"""

from __future__ import annotations

import numpy as np

from mceq3d_flux import MCEq3DFlux

RC_KAMIOKA = 11.3  # GV, back-traced vertical cutoff (geomag_backtrace)
EGRID = np.array(
    [
        0.1,
        0.13,
        0.16,
        0.2,
        0.3,
        0.4,
        0.5,
        0.7,
        1.0,
        1.5,
        2.0,
        3.0,
        5.0,
        10.0,
        30.0,
        100.0,
    ]
)


def _vertical_numu(base_model):
    """Absolute numu at Kamioka vertical (down) = base(|cosZ|) * G(Rc), /(m2 s sr GeV)."""
    eng = MCEq3DFlux(base_model=base_model, daemonflux_location="kamioka")
    e = eng.e
    base = eng.base(np.array([0.95]))["total_numu"][0]
    G, rg = eng.geomag_response(np.linspace(2.0, 20.0, 10))
    g = np.array(
        [np.interp(RC_KAMIOKA, rg, G["total_numu"][:, k]) for k in range(len(e))]
    )
    flux = base * g
    return e, np.array(
        [
            np.exp(np.interp(np.log(E), np.log(e), np.log(np.maximum(flux, 1e-300))))
            for E in EGRID
        ]
    )


def main():
    import os

    e_mc, f_mc = _vertical_numu("mceq")
    _, f_df = _vertical_numu("daemonflux")

    h = dict(np.load("honda_kam.npz"))
    He, Hcz, nm = h["E"], h["czlo"], h["numu"]
    ihz = int(np.argmin(np.abs(Hcz - 0.9)))
    hv = nm[ihz].mean(0)
    f_h = np.array(
        [np.exp(np.interp(np.log(E), np.log(He), np.log(hv))) for E in EGRID]
    )

    print("numu vertical (Kamioka), ratio to Honda:")
    print("  E[GeV]   MCEq/Honda   daemonflux/Honda")
    for E, a, b, c in zip(EGRID, f_mc, f_df, f_h):
        print(f"  {E:6.2f}     {a / c:6.2f}        {b / c:6.2f}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axhspan(0.9, 1.1, color="0.85", label="±10 % of Honda")
    ax.axhline(1.0, color="k", lw=1)
    ax.plot(EGRID, f_mc / f_h, "C0o-", label="MCEq base / Honda")
    ax.plot(EGRID, f_df / f_h, "C3s-", label="daemonflux base / Honda")
    ax.set_xscale("log")
    ax.set_xlabel("E [GeV]")
    ax.set_ylabel(r"$\Phi_{\nu_\mu}$(this work) / Honda  (vertical)")
    ax.set_title("1D-base choice vs Honda HKKM2014 (Kamioka), 0.1-100 GeV")
    ax.set_ylim(0.5, 2.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig("base_comparison.png", dpi=110)
    print("saved plot -> base_comparison.png")
    _ = os


if __name__ == "__main__":
    main()
