"""Compare the 1D-base choice (MCEq vs daemonflux) against Honda, 0.1-100 GeV.

The sub-GeV-to-few-GeV region is where atmospheric-oscillation physics lives and
where atmospheric-flux models disagree most. This scans the absolute numu flux
ratio to Honda for both ``base_model`` options on a dense grid, at Kamioka
vertical (down-going), so the trustworthy energy range of each base is explicit:

* daemonflux (muon-calibrated) is the better base for E >~ 1 GeV (~10 % of Honda),
  but extrapolates above Honda below ~0.3 GeV (beyond its muon-calibration region);
* raw MCEq (SIBYLL23D+H3a) runs ~25-30 % low over 0.3-10 GeV but is closer at
  0.1-0.2 GeV.

The two bases **bracket Honda only below ~1 GeV** (daemonflux above, MCEq below) --
there the geometric mean reproduces Honda to ~5 % and that band is the genuine
irreducible sub-GeV uncertainty. Above ~1 GeV *both* bases lie below Honda
(daemonflux closer, ~0.91), so the geometric mean is biased ~10-18 % low there and
daemonflux should be used as the central value. Run::

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


def model_envelope(f_mc, f_df):
    """Central value + fractional systematic from the two-base spread.

    The **half log-spread** of the MCEq and daemonflux bases is a data-grounded
    one-sigma *model* systematic — large below ~1 GeV and shrinking to a few % above
    10 GeV. The **geometric mean** is returned as a convenience central value, but
    note its energy regimes differ (see `base_comparison` module docstring): the two
    bases bracket Honda only below ~1 GeV (where the geometric mean ~ Honda); above
    ~1 GeV both lie below Honda and the data-anchored daemonflux base is the better
    central. The band is an intra-framework spread (a floor on the flux
    uncertainty), not a full inter-calculation envelope.
    """
    f_mc = np.asarray(f_mc)
    f_df = np.asarray(f_df)
    central = np.sqrt(f_mc * f_df)
    sysfrac = 0.5 * np.abs(np.log(f_df / f_mc))  # half log-spread ~ fractional sigma
    return central, sysfrac


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

    # Model-spread systematic: the two bases bracket Honda, so their half-spread is
    # a defensible, data-grounded flux uncertainty (large sub-GeV, small at high E).
    central, sysfrac = model_envelope(f_mc, f_df)

    print("numu vertical (Kamioka):")
    print("  E[GeV]   MCEq/Honda   daemonflux/Honda   central/Honda   ±syst")
    for E, a, b, c, cen, sf in zip(EGRID, f_mc, f_df, f_h, central, sysfrac):
        print(
            f"  {E:6.2f}     {a / c:6.2f}        {b / c:6.2f}"
            f"           {cen / c:6.2f}        {sf * 100:4.0f}%"
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axhspan(0.9, 1.1, color="0.85", label="±10 % of Honda")
    ax.axhline(1.0, color="k", lw=1)
    lo = np.minimum(f_mc, f_df) / f_h
    hi = np.maximum(f_mc, f_df) / f_h
    ax.fill_between(
        EGRID, lo, hi, color="C2", alpha=0.2, label="model-spread systematic (envelope)"
    )
    ax.plot(EGRID, f_mc / f_h, "C0o-", label="MCEq base / Honda")
    ax.plot(EGRID, f_df / f_h, "C3s-", label="daemonflux base / Honda")
    ax.plot(EGRID, central / f_h, "C2--", lw=2, label="central (geom. mean)")
    ax.set_xscale("log")
    ax.set_xlabel("E [GeV]")
    ax.set_ylabel(r"$\Phi_{\nu_\mu}$(this work) / Honda  (vertical)")
    ax.set_title("1D-base choice vs Honda HKKM2014 (Kamioka), 0.1-100 GeV")
    ax.set_ylim(0.5, 2.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("base_comparison.png", dpi=110)
    print("saved plot -> base_comparison.png")
    _ = os


if __name__ == "__main__":
    main()
