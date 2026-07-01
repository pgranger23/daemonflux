"""Propagate daemonflux's muon-calibration uncertainty to the directional flux.

daemonflux carries a nuisance-parameter covariance (its defining feature) and an
``error()`` method. The directional flux here is ``Phi_3D = Phi_df * G * S`` with the
geomagnetic factor ``G`` and solar factor ``S`` independent of those parameters, so

    sigma(Phi_3D)/Phi_3D = sigma(Phi_df)/Phi_df,

i.e. the *fractional* calibration uncertainty (and the full parameter covariance,
via gradient scaling by G*S) carries through unchanged. ``solve(with_calib_error=
True)`` returns ``flux_err`` (absolute 1-sigma) and ``flux_relerr`` (fractional);
``calib_hadronic_only`` isolates the hadronic-production part. This script solves at
Kamioka vertical, prints the propagated band, checks the fractional error is
preserved, and plots it. Run::

    python calib_uncertainty.py --plot
"""

from __future__ import annotations

import argparse

import numpy as np

from mceq3d_flux import MCEq3DFlux, SPECIES


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args(argv)

    eng = MCEq3DFlux(base_model="daemonflux", daemonflux_location="kamioka")
    cz = np.array([0.95])
    az = np.array([0.0])
    r = eng.solve(36.43, 137.31, cz, az, with_calib_error=True)
    rh = eng._daemonflux_relerr(cz, only_hadronic=True)  # hadronic-only fractional
    e = r["e"]

    def at(y, E):
        return float(np.interp(E, e, y))

    print(
        "Kamioka vertical numu: flux +/- calibration 1-sigma (daemonflux covariance):"
    )
    print("  E[GeV]   flux        +/-1sig      total%   hadronic%")
    fnumu = r["flux"]["total_numu"][0, 0]
    enumu = r["flux_err"]["total_numu"][0, 0]
    rel = r["flux_relerr"]["total_numu"][0]
    for E in (0.3, 0.5, 1.0, 3.0, 10.0, 100.0):
        print(
            f"  {E:6.2f}  {at(fnumu, E):9.3g}  {at(enumu, E):9.3g}   "
            f"{at(rel, E) * 100:5.1f}%   {at(rh['total_numu'][0], E) * 100:5.1f}%"
        )

    # Check: fractional error is preserved (G,S drop out) -- compare to raw daemonflux
    dfe = np.asarray(eng._df.error(e[e <= 1e9], 0.0, "numu"))
    dff = np.asarray(eng._df.flux(e[e <= 1e9], 0.0, "numu"))
    raw = dfe / dff
    prop = rel[e <= 1e9]
    print(
        f"\nfractional error preserved through G*S? max|prop-raw| = "
        f"{np.nanmax(np.abs(prop - raw)):.2e} (should be ~0)"
    )

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sel = (e >= 0.1) & (e <= 100)
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))
        f3 = fnumu * e**3
        axL.fill_between(
            e[sel],
            (f3 - enumu * e**3)[sel],
            (f3 + enumu * e**3)[sel],
            color="C0",
            alpha=0.3,
            label=r"$\pm1\sigma$ calibration",
        )
        axL.loglog(e[sel], f3[sel], "C0-", label=r"$\nu_\mu$ (daemonflux base)")
        axL.set_xlabel("E [GeV]")
        axL.set_ylabel(r"$E^3\Phi_{\nu_\mu}$ [GeV$^2$/(m$^2$ s sr)]")
        axL.set_title(
            "Directional flux with propagated calibration band (Kamioka vert)"
        )
        axL.legend(fontsize=8)
        for s in SPECIES:
            axR.semilogx(
                e[sel],
                (r["flux_relerr"][s][0] * 100)[sel],
                label=s.replace("total_", ""),
            )
        axR.set_xlabel("E [GeV]")
        axR.set_ylabel("calibration uncertainty [%]")
        axR.set_title("Propagated fractional calibration error, per species")
        axR.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig("calib_uncertainty.png", dpi=110)
        print("saved plot -> calib_uncertainty.png")


if __name__ == "__main__":
    main()
