"""
[PROTOTYPE] Research/de-risking scaffolding -- NOT part of the delivered flux
(mceq3d_flux). Kept for the record; do not depend on it in the paper. See
ARCHITECTURE.md.

Propagate daemonflux's muon-calibration uncertainty to the directional flux.

daemonflux carries a nuisance-parameter covariance (its defining feature) and an
``error()`` method. The directional flux here is ``Phi_3D = Phi_df * G * S`` with the
geomagnetic factor ``G`` and solar factor ``S`` independent of those parameters, so

    sigma(Phi_3D)/Phi_3D = sigma(Phi_df)/Phi_df,

i.e. the *fractional* calibration uncertainty and the *full correlated covariance*
(via gradient scaling by G*S) carry through unchanged. ``solve(with_calib_error=
True)`` returns ``flux_err``/``flux_relerr`` (1-sigma band); ``with_calib_jacobian=
True`` additionally returns the 24 nuisance-parameter names, their correlation
matrix, and the fractional per-parameter Jacobian, from which
:func:`mceq3d_flux.calib_covariance` forms the full energy-energy covariance for a
fit. This script solves at Kamioka vertical, prints the band, checks the fractional
error is preserved and that the covariance reproduces ``error()`` exactly, shows the
energy-bin correlations, and exports a self-contained npz. Run::

    python calib_uncertainty.py --plot
"""

from __future__ import annotations

import argparse

import numpy as np

from mceq3d_flux import MCEq3DFlux, SPECIES, calib_covariance


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args(argv)

    eng = MCEq3DFlux(base_model="daemonflux", daemonflux_location="kamioka")
    cz = np.array([0.95])
    az = np.array([0.0])
    r = eng.solve(
        36.43, 137.31, cz, az, with_calib_error=True, with_calib_jacobian=True
    )
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

    # Full correlated covariance (for a fit): Cov = (Phi*R)^T corr (Phi*R).
    print(f"\nFull calibration covariance ({len(r['calib_params'])} nuisance params):")
    C = calib_covariance(r, "total_numu", iz=0, ia=0)  # (n_E, n_E)
    band = np.sqrt(np.diag(C))
    # sqrt(diag) must equal the 1-sigma from the covariance path (self-consistency)
    print(
        "  sqrt(diag(Cov))/flux == relerr?  max diff = "
        f"{np.nanmax(np.abs(np.where(fnumu > 0, band / fnumu, 0) - rel)):.2e}"
    )
    m = e <= 1e9
    Cn = C[np.ix_(m, m)] / np.outer(band[m], band[m])  # correlation matrix
    ie1 = int(np.argmin(np.abs(e - 1.0)))
    ie3 = int(np.argmin(np.abs(e - 3.0)))
    ie30 = int(np.argmin(np.abs(e - 30.0)))
    mm = np.where(m)[0]

    def ci(i):
        return int(np.searchsorted(mm, i))

    print(
        f"  energy-bin correlation: rho(1,3 GeV)={Cn[ci(ie1), ci(ie3)]:+.2f}, "
        f"rho(1,30 GeV)={Cn[ci(ie1), ci(ie30)]:+.2f} "
        "(nearby bins correlated, far bins less)"
    )
    # export a self-contained npz an analysis can load
    np.savez(
        "calib_export_kamioka_vert.npz",
        e=e,
        flux_numu=fnumu,
        params=np.array(r["calib_params"]),
        corr=r["calib_corr"],
        jac_numu=r["calib_jac"]["total_numu"][:, 0, :],  # (n_par, n_E) fractional
    )
    print(
        "  exported -> calib_export_kamioka_vert.npz "
        "(flux, params, corr, fractional jac)"
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
