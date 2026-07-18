"""
[PROTOTYPE] Research/de-risking scaffolding -- NOT part of the delivered flux
(mceq3d_flux). Kept for the record; do not depend on it in the paper. See
ARCHITECTURE.md.

Unify the two 3D corrections into one directional low-energy flux.

This project produced two independent 3D pieces, both expressed as multiplicative
corrections to the calibrated 1D flux:

* **geomagnetic** -- the rigidity-cutoff / East-West admittance
  ``G(E, zenith, azimuth; site)`` from ``daemonflux.geomagnetic`` (Stoermer
  scaffold; depends on azimuth -> East-West asymmetry);
* **production-angle** -- the angular redistribution ``R(E, zenith) =
  Phi_3D/Phi_1D`` from :mod:`coupled_3d_flux`, driven by the NA61-validated
  ``<theta^2>(E)`` (azimuth-independent; smears the zenith distribution).

Because both are ratios to the same 1D flux, the combined directional correction
on top of daemonflux is simply their product::

    Phi_3D(E, zenith, azimuth) / Phi_1D(E, zenith) = R(E, zenith) * G(E, zenith, azimuth)

This is the complete (within current scope) low-energy directional 3D model: a
zenith smearing AND an azimuthal East-West asymmetry AND an overall low-energy
cutoff, all on top of the muon-calibrated daemonflux 1D flux, and all reducing to
1 at high energy.

Run::

    python unified_3d_flux.py --moments m_spliced.npz --site kamioka --plot
"""

from __future__ import annotations

import argparse

import numpy as np

from coupled_3d_flux import mceq_1d_flux, sigma_theta_grid, convolve_sphere
from fokker_planck_3d import load_theta2


def angular_ratio(e_grid, cos_full, flux_full, e_sig, theta2, out_cos, e_indices):
    """R(E, cos theta) = Phi_3D/Phi_1D from the production-angle spread."""
    zeniths = np.degrees(np.arccos(out_cos))
    sigma_EZ = sigma_theta_grid(e_grid, e_sig, theta2, zeniths)
    phi3d = convolve_sphere(e_grid, cos_full, flux_full, sigma_EZ, out_cos, e_indices)
    phi1d = np.array(
        [np.interp(out_cos, cos_full, flux_full[:, ie]) for ie in e_indices]
    ).T
    return phi3d / phi1d  # (n_cos, n_E)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--moments", default="m_spliced.npz")
    p.add_argument("--site", default="kamioka")
    p.add_argument("--quantity", default="total_numu")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args(argv)

    from daemonflux.geomagnetic import GeomagneticModel

    geo = GeomagneticModel(args.site)
    print(geo)

    e_sig, theta2 = load_theta2(args.moments)
    e_grid, cos_full, flux_full = mceq_1d_flux(args.quantity)

    out_cos = np.linspace(0.05, 1.0, 25)
    e_targets = [0.5, 1.0, 2.0, 5.0, 100.0]
    e_idx = [int(np.argmin(np.abs(e_grid - e))) for e in e_targets]
    R = angular_ratio(e_grid, cos_full, flux_full, e_sig, theta2, out_cos, e_idx)

    # Combined correction at a near-horizon zenith vs azimuth (East-West).
    je = e_targets.index(1.0)
    zen_ew = 70.0
    cos_ew = np.cos(np.radians(zen_ew))
    r_ew = float(np.interp(cos_ew, out_cos, R[:, je]))
    az = np.linspace(0, 360, 73)
    g_ew = geo.admittance(args.quantity, np.array([e_targets[je]]), zen_ew, az)[:, 0]
    combined = r_ew * g_ew

    print(f"\nE=1 GeV, zenith={zen_ew} deg, site={args.site}:")
    print(f"  angular-only R           = {r_ew:.3f} (azimuth-independent)")
    print(
        f"  geomagnetic G  W / E     = {g_ew[az == 270][0]:.3f} / "
        f"{g_ew[az == 90][0]:.3f}"
    )
    print(
        f"  combined  W / E          = {combined[az == 270][0]:.3f} / "
        f"{combined[az == 90][0]:.3f}"
    )
    ew = combined[az == 270][0] / combined[az == 90][0]
    print(f"  East-West ratio (W/E)    = {ew:.3f}")

    if args.plot:
        _plot(out_cos, R, e_targets, je, az, r_ew, g_ew, combined, geo, args)


def _plot(out_cos, R, e_targets, je, az, r_ew, g_ew, combined, geo, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # Left: the three corrections vs azimuth at E=1 GeV, near horizon.
    axL.plot(az, np.full_like(az, r_ew), "C2--", label="angular only (R)")
    axL.plot(az, g_ew, "C1-.", label="geomagnetic only (G)")
    axL.plot(az, combined, "C0-", lw=2, label="combined (R$\\times$G)")
    for a, name in [(90, "E"), (180, "S"), (270, "W"), (360, "N")]:
        axL.axvline(a, color="gray", ls=":", lw=0.6)
    axL.set_xticks([0, 90, 180, 270, 360])
    axL.set_xlabel("geographic azimuth [deg] (E=90, W=270)")
    axL.set_ylabel(r"$\Phi_{3D}/\Phi_{1D}$")
    axL.set_title(f"{args.site}: 1 GeV, zenith 70 deg (East-West)")
    axL.legend()

    # Right: full sky correction map at 1 GeV, (azimuth x cos zenith).
    cos_map = out_cos
    Z = np.zeros((len(cos_map), len(az)))
    for ic, c in enumerate(cos_map):
        zen = np.degrees(np.arccos(c))
        g = geo.admittance(args.quantity, np.array([e_targets[je]]), zen, az)[:, 0]
        Z[ic] = R[ic, je] * g
    pc = axR.pcolormesh(az, cos_map, Z, shading="auto", cmap="viridis")
    axR.set_xticks([0, 90, 180, 270, 360])
    axR.set_xlabel("geographic azimuth [deg]")
    axR.set_ylabel(r"$\cos\theta_z$ (1=vertical, 0=horizon)")
    axR.set_title(r"Combined $\Phi_{3D}/\Phi_{1D}$ at 1 GeV")
    fig.colorbar(pc, ax=axR)

    fig.tight_layout()
    fig.savefig("unified_3d_flux.png", dpi=110)
    print("saved plot -> unified_3d_flux.png")


if __name__ == "__main__":
    main()
