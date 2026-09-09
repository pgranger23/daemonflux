"""Compare the 1D-base choice (MCEq vs daemonflux) against Honda, 0.1-100 GeV.

The sub-GeV-to-few-GeV region is where atmospheric-oscillation physics lives and
where atmospheric-flux models disagree most. This scans the absolute numu flux
ratio to Honda for both ``base_model`` options on a dense grid, at Kamioka
vertical (down-going), so the trustworthy energy range of each base is explicit:

* daemonflux (muon-calibrated) is the better base for E >~ 1 GeV (~10 % of Honda),
  but extrapolates above Honda below ~0.3 GeV (beyond its muon-calibration region);
* raw MCEq (SIBYLL23D+H3a) runs ~25-30 % low over 0.3-10 GeV but is closer at
  0.1-0.2 GeV.

The recommended central is the **GSF-anchored hybrid** (GSF primary below the E0
crossover, muon-calibrated daemonflux above), which reproduces Honda to ~1.0
sub-GeV. The delivered **systematic band** is the half log-spread between this
hybrid central and the (independently data-anchored) daemonflux base: +-14 % at
0.5 GeV, +-15 % at 0.3 GeV, +-5 % at 1 GeV, collapsing to ~0 above the ~2 GeV
crossover (where hybrid == daemonflux and the residual uncertainty is the
muon-calibration covariance + hadronic-model spread). This replaces the earlier
raw-H3a-MCEq <-> daemonflux band (+-31 % at 0.5 GeV), which inflated the sub-GeV
uncertainty with a base -- the H3a-primary MCEq, ~25-30 % low over 0.3-10 GeV --
that we explicitly recommend against. The legacy band is still printed for
reference. Run::

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


def _vertical_numu(base_model, primary=None):
    """Absolute numu at Kamioka vertical (down) = base(|cosZ|) * G(Rc), /(m2 s sr GeV).

    ``primary`` selects the cosmic-ray primary model; the recommended hybrid base
    uses the Global Spline Fit (``("GlobalSplineFitBeta", None)``), so it must be
    passed here to reproduce the delivered configuration (the default H3a would
    give the raw-MCEq sub-GeV deficit below the crossover)."""
    kw = dict(base_model=base_model, daemonflux_location="kamioka")
    if primary is not None:
        kw["primary"] = primary
    eng = MCEq3DFlux(**kw)
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
    # recommended central: the GSF-anchored hybrid (GSF primary below the E0
    # crossover, muon-calibrated daemonflux above).
    _, f_hy = _vertical_numu("hybrid", primary=("GlobalSplineFitBeta", None))

    h = dict(np.load("honda_kam.npz"))
    He, Hcz, nm = h["E"], h["czlo"], h["numu"]
    ihz = int(np.argmin(np.abs(Hcz - 0.9)))
    hv = nm[ihz].mean(0)
    f_h = np.array(
        [np.exp(np.interp(np.log(E), np.log(He), np.log(hv))) for E in EGRID]
    )

    # Model-spread systematic around the RECOMMENDED hybrid central: the half
    # log-spread between the GSF-hybrid and the (independently data-anchored)
    # daemonflux base. This replaces the old raw-H3a-MCEq <-> daemonflux spread,
    # which inflated the sub-GeV band with a base we recommend against. The band
    # collapses above the crossover (there hybrid == daemonflux); the residual
    # uncertainty there is the muon-calibration covariance + hadronic-model spread.
    central, sysfrac = f_hy, 0.5 * np.abs(np.log(f_df / f_hy))
    _, oldsys = model_envelope(f_mc, f_df)  # legacy band (for reference)

    print("numu vertical (Kamioka):")
    print("  E[GeV]  MCEq/H  df/H  hybrid/H  | band(old)  band(hybrid)")
    for E, a, b, hy, c, sf, osf in zip(EGRID, f_mc, f_df, f_hy, f_h, sysfrac, oldsys):
        print(
            f"  {E:6.2f}  {a / c:5.2f}  {b / c:5.2f}   {hy / c:5.2f}"
            f"    |   {osf * 100:4.0f}%       {sf * 100:4.0f}%"
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axhspan(0.9, 1.1, color="0.85", label="±10 % of Honda")
    ax.axhline(1.0, color="k", lw=1)
    # band around the recommended hybrid central: GSF-hybrid <-> daemonflux
    lo = np.minimum(f_hy, f_df) / f_h
    hi = np.maximum(f_hy, f_df) / f_h
    ax.fill_between(
        EGRID, lo, hi, color="C2", alpha=0.2,
        label="model-spread band (hybrid ↔ daemonflux)"
    )
    ax.plot(EGRID, f_mc / f_h, "C0o-", lw=1, alpha=0.6,
            label="raw MCEq (H3a) / Honda")
    ax.plot(EGRID, f_df / f_h, "C3s-", label="daemonflux base / Honda")
    ax.plot(EGRID, central / f_h, "C2D-", lw=2,
            label="recommended central (GSF-hybrid)")
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
