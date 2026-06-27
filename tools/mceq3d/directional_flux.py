"""Integrated directional 3D flux -- wiring all the pieces together.

This combines the validated components into a single directional neutrino flux
``Phi(E, zenith, azimuth)`` for a detector site:

* the multi-species cascade with the NA61-validated angular layer
  (:mod:`mceq3d_solver`);
* the **first-principles back-traced geomagnetic cutoff** (:mod:`geomag_backtrace`,
  full IGRF) folded into the *primary* spectrum per direction -- giving the
  rigidity suppression and the East-West asymmetry self-consistently;
* **muon bending** (:mod:`muon_bending`) added to the angular spread.

The cutoff map ``R_c(zenith, azimuth)`` is the only slow part (trajectory
integration, computed once); the cascade per direction is fast. Output: the
directional flux and the 3D/1D ratio, showing the geomagnetic cutoff, the
East-West asymmetry, and the angular structure together.

Remaining for the *complete* off-axis Honda excess (not here): the spherical
streaming that lets neutrinos from off-axis showers reach the detector (the
production-angle part of which is small, ~1-2%, per `coupled_3d_flux`) and the
coherent charge-dependent muon-bending shift -- both de-risked.

Run::

    python directional_flux.py --plot
"""

from __future__ import annotations

import argparse
import datetime

import numpy as np


def solve_directional(
    lat, lon, zeniths, azimuths, date=None, n_scan=14, moments="m_spliced.npz"
):
    """Phi_numu(E, zenith, azimuth) for a site, with back-traced geomagnetics.

    Returns dict: ``e``, ``flux`` (n_zen, n_az, n_E), ``base`` (no-geomag flux,
    n_E), ``cutoff`` (n_zen, n_az) [GV].
    """
    import geomag_backtrace
    from mceq3d_solver import solve

    date = date or datetime.datetime(2020, 1, 1)
    rc = geomag_backtrace.cutoff_map(
        lat, lon, date, np.asarray(zeniths), np.asarray(azimuths), n_scan=n_scan
    )  # (n_zen, n_az)

    base = solve(moments=moments, cutoff_GV=None, muon_bending=True)  # no-geomag ref
    e = base["e"]
    flux = np.zeros((len(zeniths), len(azimuths), len(e)))
    for iz, z in enumerate(zeniths):
        for ia in range(len(azimuths)):
            r = solve(
                moments=moments, zenith_deg=z, cutoff_GV=rc[iz, ia], muon_bending=True
            )
            flux[iz, ia] = r["numu"]
    return dict(
        e=e,
        flux=flux,
        base=base["numu"],
        cutoff=rc,
        zeniths=np.asarray(zeniths),
        azimuths=np.asarray(azimuths),
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lat", type=float, default=36.43)  # Kamioka
    p.add_argument("--lon", type=float, default=137.31)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args(argv)

    zen = np.array([0, 20, 40, 60, 75, 88])
    az = np.array([0, 45, 90, 135, 180, 225, 270, 315])
    r = solve_directional(args.lat, args.lon, zen, az)
    e = r["e"]

    ie = int(np.argmin(np.abs(e - 1.0)))  # 1 GeV
    ratio = r["flux"][:, :, ie] / r["base"][ie]
    print(f"Phi_3D/Phi_1D at 1 GeV (lat={args.lat}, lon={args.lon}):")
    print("  zenith \\ azimuth   " + "  ".join(f"{a:4.0f}" for a in az))
    for iz, z in enumerate(zen):
        print(f"   {z:5.0f}            " + "  ".join(f"{x:4.2f}" for x in ratio[iz]))
    iE_e = list(az).index(90)
    iE_w = list(az).index(270)
    print(
        f"\n  East-West at zenith 75, 1 GeV:  W/E = "
        f"{ratio[4, iE_w] / ratio[4, iE_e]:.2f}"
    )

    if args.plot:
        _plot(r, ie)


def _plot(r, ie):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zen, az = r["zeniths"], r["azimuths"]
    ratio = r["flux"][:, :, ie] / r["base"][ie]
    cosz = np.cos(np.deg2rad(zen))

    fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(15, 4.3))
    pc = axL.pcolormesh(az, cosz, ratio, shading="auto", cmap="viridis")
    axL.set_xlabel("azimuth [deg] (E=90, W=270)")
    axL.set_ylabel(r"$\cos\theta_z$")
    axL.set_title(r"$\Phi_{3D}/\Phi_{1D}$ at 1 GeV (full IGRF + cascade)")
    axL.set_xticks([0, 90, 180, 270])
    fig.colorbar(pc, ax=axL)

    axM.plot(az, ratio[4], "C0o-")  # zenith 75
    for a in (90, 270):
        axM.axvline(a, color="gray", ls=":", lw=0.6)
    axM.set_xticks([0, 90, 180, 270])
    axM.set_xlabel("azimuth [deg] (E=90, W=270)")
    axM.set_ylabel(r"$\Phi_{3D}/\Phi_{1D}$")
    axM.set_title("East-West asymmetry (zenith 75)")

    az_avg = ratio.mean(axis=1)
    axR.plot(cosz, az_avg, "C3o-")
    axR.set_xlabel(r"$\cos\theta_z$ (1=vertical, 0=horizon)")
    axR.set_ylabel(r"$\langle\Phi_{3D}/\Phi_{1D}\rangle_\phi$")
    axR.set_title("Zenith dependence (azimuth-averaged), 1 GeV")
    fig.tight_layout()
    fig.savefig("directional_flux.png", dpi=110)
    print("\nsaved plot -> directional_flux.png")


if __name__ == "__main__":
    main()
