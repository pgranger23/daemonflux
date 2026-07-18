"""
[PROTOTYPE] Research/de-risking scaffolding -- NOT part of the delivered flux
(mceq3d_flux). Kept for the record; do not depend on it in the paper. See
ARCHITECTURE.md.

Spherical cascade -- the energy cascade coupled to the curved atmosphere.

This is the final coupling: it runs the multi-species energy cascade
``N -> pi/K -> nu`` **down the curved line of sight** of each arrival direction,
so the decay-vs-interaction competition sees the *actual altitude/density profile*
of that column (from :mod:`spherical_geometry`), rather than a single slant-depth
approximation. It yields the absolute directional conventional flux
``Phi_nu(E, cos zenith)``.

What the coupling produces (and what it teaches)
-----------------------------------------------
The well-known **sec(theta) horizon enhancement** emerges -- and it is a
*high-energy* effect, made **finite** by the spherical geometry:

* **Sub-GeV: near-isotropic.** Mesons decay almost immediately regardless of the
  path, and primaries fully interact in every column, so the total production
  saturates -- the flux is ~flat in zenith (1D and curved-3D agree).
* **High energy: sec(theta) toward the horizon.** Mesons preferentially interact
  at low altitude, but along the long near-horizon path they spend more time high
  up (low density) and decay instead -> the flux rises toward the horizon,
  approaching ``sec(theta)`` at moderate zenith and **saturating** near the
  horizon (e.g. ~5x, not the divergent ``sec(theta)=25``).

So the *dominant* directional structure is this curved-cascade (1D-per-direction)
effect. The **genuinely-3D residual** -- the streaming/angular coupling *between*
arrival directions that 1D-per-direction misses -- is small (~1-2% on the
conventional flux; see `coupled_3d_flux` and the validated `spherical_streaming`
operator). The large sub-GeV 3D effects are geomagnetic (`directional_flux`) and
muon bending (`muon_bending`), not this geometric term.

What couples to what
--------------------
* energy cascade: the scaling yields / interaction lengths / decay competition of
  :mod:`mceq3d_solver`, **depth-resolved** -- marched in path length with the
  *local* density ``rho(altitude(l, zenith))``;
* spherical geometry: ``altitude(l, zenith)`` along the curved ray
  (:func:`spherical_geometry._altitude_along_ray`).

Validation
----------
* **vertical = 1** by construction (reference column);
* **high energy, moderate zenith -> sec(theta)** (the enhancement approaches
  ``1/cos(theta)`` as E grows);
* **near the horizon -> finite saturation** (no ``sec(theta)`` divergence);
* **sub-GeV -> near-isotropic** (flat in zenith).

Run::

    python spherical_cascade.py --plot
"""

from __future__ import annotations

import argparse

import numpy as np

from mceq3d_solver import scaling_matrix, log_grid, ZMAX, LAM, N, PI, K
from spherical_geometry import _altitude_along_ray, R_EARTH, H_TOP

RHO0 = 1.205e-3  # sea-level air density [g/cm^3]
H_RHO = 6.4  # density scale height [km] (matches spherical_geometry.H0)
M_MES = {PI: 0.13957, K: 0.49368}  # meson masses [GeV]
CTAU_KM = {PI: 7.804e-3, K: 3.711e-3}  # c*tau [km]
KM_TO_CM = 1.0e5


def density_gcc(alt_km):
    """Isothermal exponential air density [g/cm^3]."""
    return RHO0 * np.exp(-np.clip(alt_km, 0.0, None) / H_RHO)


def path_to_top(zenith_deg):
    """Path length [km] from the detector to the top of the atmosphere (curved)."""
    c = np.cos(np.deg2rad(zenith_deg))
    return -R_EARTH * c + np.sqrt((R_EARTH * c) ** 2 + 2 * R_EARTH * H_TOP + H_TOP**2)


def solve_column(zenith_deg, e, dlnE, yields, gamma=1.7, nsteps=4000):
    """March the cascade down one curved column; return Phi_nu(E) at the detector.

    The nucleon, pion and kaon fluxes are evolved in path length ``l`` (top ->
    detector) with the local density setting the interaction (per grammage) and
    decay (per length) rates; their competition at each step feeds neutrinos.
    """
    Y_NN, Y_Npi, Y_NK, Y_pinu, Y_Knu = yields
    lmax = path_to_top(zenith_deg)
    s_grid = np.linspace(lmax, 0.0, nsteps)  # top of atmosphere -> detector
    dl = abs(s_grid[1] - s_grid[0])  # km

    PhiN = e ** (-gamma)  # collimated primary nucleons (isotropic per direction)
    PhiPi = np.zeros_like(e)
    PhiK = np.zeros_like(e)
    PhiNu = np.zeros_like(e)

    ddec_pi = (e / M_MES[PI]) * CTAU_KM[PI]  # decay length [km]
    ddec_K = (e / M_MES[K]) * CTAU_KM[K]
    rdec_pi = 1.0 / ddec_pi  # decay rate [1/km]
    rdec_K = 1.0 / ddec_K

    for s in s_grid:
        rho = density_gcc(_altitude_along_ray(s, zenith_deg))
        dX = rho * dl * KM_TO_CM  # grammage step [g/cm^2]
        rint_pi = rho * KM_TO_CM / LAM[PI]  # meson interaction rate [1/km]
        rint_K = rho * KM_TO_CM / LAM[K]

        # nucleon interactions produce mesons
        dNi = PhiN * (1.0 - np.exp(-dX / LAM[N]))
        PhiN = PhiN - dNi + Y_NN @ dNi
        PhiPi = PhiPi + Y_Npi @ dNi
        PhiK = PhiK + Y_NK @ dNi

        # meson decay vs interaction competition (local density)
        for Phi_m, rint, rdec, Y in (
            (PhiPi, rint_pi, rdec_pi, Y_pinu),
            (PhiK, rint_K, rdec_K, Y_Knu),
        ):
            ploss = 1.0 - np.exp(-(rint + rdec) * dl)
            frac_dec = rdec / (rint + rdec)
            lost = Phi_m * ploss
            PhiNu = PhiNu + Y @ (lost * frac_dec)
            Phi_m -= lost  # in-place update of the array
    return PhiNu


def build_yields(e, dlnE):
    """Scaling production yields (same forms as mceq3d_solver)."""
    Y_NN = scaling_matrix(e, dlnE, lambda x: 0.8 * np.ones_like(x))
    Y_Npi = scaling_matrix(e, dlnE, lambda x: 5.0 * (1 - x) ** 3 / x)
    Y_NK = scaling_matrix(e, dlnE, lambda x: 0.5 * (1 - x) ** 3 / x)
    Y_pinu = scaling_matrix(
        e, dlnE, lambda x: np.where(x < ZMAX[PI], 1.0 / ZMAX[PI], 0.0)
    )
    Y_Knu = scaling_matrix(e, dlnE, lambda x: np.where(x < ZMAX[K], 1.0 / ZMAX[K], 0.0))
    return Y_NN, Y_Npi, Y_NK, Y_pinu, Y_Knu


def solve(cos_zeniths, n_e=60, nsteps=4000, gamma=1.7):
    """Phi_nu(E, cos zenith) through the curved atmosphere."""
    e, dlnE = log_grid(n_e)
    yields = build_yields(e, dlnE)
    flux = np.zeros((len(cos_zeniths), len(e)))
    for i, cz in enumerate(cos_zeniths):
        z = np.degrees(np.arccos(np.clip(cz, -1, 1)))
        flux[i] = solve_column(z, e, dlnE, yields, gamma=gamma, nsteps=nsteps)
    return dict(e=e, cos_zeniths=np.asarray(cos_zeniths), flux=flux)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--nsteps", type=int, default=4000)
    args = p.parse_args(argv)

    cz = np.array([1.0, 0.8, 0.6, 0.4, 0.25, 0.15, 0.08, 0.04])
    r = solve(cz, nsteps=args.nsteps)
    e = r["e"]
    vert = r["flux"][0]  # cos=1

    print("Directional enhancement Phi(cosZ)/Phi(vertical) vs energy:")
    print("  E[GeV] \\ cosZ  " + "  ".join(f"{c:5.2f}" for c in cz))
    for ie in range(0, len(e), 6):
        if 0.3 < e[ie] < 1e4 and vert[ie] > 0:
            row = r["flux"][:, ie] / vert[ie]
            print(f"  {e[ie]:8.2f}    " + "  ".join(f"{x:5.2f}" for x in row))

    # --- validations ---
    ielo = int(np.argmin(np.abs(e - 0.35)))
    iehi = int(np.argmin(np.abs(e - 2000.0)))
    icz = list(cz).index(0.6)  # moderate zenith, sec(theta)=1.667
    print("\nVALIDATION:")
    print(
        f"  vertical (cosZ=1) ratio        = {r['flux'][0, iehi]/vert[iehi]:.3f} "
        "(== 1 by construction)"
    )
    print(
        f"  sub-GeV isotropy (0.35 GeV, cosZ=0.04) = "
        f"{r['flux'][-1, ielo]/vert[ielo]:.3f}  (~1 -> isotropic)"
    )
    print(
        f"  high-E sec(theta) (2 TeV, cosZ=0.6): cascade "
        f"{r['flux'][icz, iehi]/vert[iehi]:.2f}  vs sec(theta)={1/0.6:.2f}"
    )
    print(
        f"  horizon saturation (2 TeV, cosZ=0.04): cascade "
        f"{r['flux'][-1, iehi]/vert[iehi]:.2f}  vs sec(theta)={1/0.04:.0f} (diverges)"
    )

    if args.plot:
        _plot(r, int(np.argmin(np.abs(e - 1.0))))


def _plot(r, ie1):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e, cz, flux = r["e"], r["cos_zeniths"], r["flux"]
    vert = flux[0]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))
    # left: enhancement vs cos zenith at a few energies + sec(theta) reference
    for E, c in [(0.5, "C0"), (10.0, "C1"), (100.0, "C2"), (2000.0, "C3")]:
        ie = int(np.argmin(np.abs(e - E)))
        axL.plot(cz, flux[:, ie] / vert[ie], c + "o-", ms=3, label=f"{E:g} GeV")
    axL.plot(cz, 1.0 / cz, "k:", label=r"1D sec$\theta$ (diverges)")
    axL.set_xlabel(r"$\cos\theta_z$ (1=vertical, 0=horizon)")
    axL.set_ylabel(r"$\Phi(\cos\theta)/\Phi_{\rm vertical}$")
    axL.set_title(r"sec$\theta$ enhancement: high-E, curved-saturated")
    axL.set_ylim(0.8, 8)
    axL.legend()
    axL.invert_xaxis()

    # right: near-horizon enhancement vs energy (grows with E, saturates)
    hor = flux[-1] / vert
    s = (e > 0.3) & (e < 1e4) & (vert > 0)
    axR.semilogx(e[s], hor[s], "C3-")
    axR.axhline(1, color="k", lw=0.6, ls=":")
    axR.axvspan(0.3, 2.0, color="orange", alpha=0.12)
    axR.text(0.6, 1.15, "sub-GeV:\nisotropic", fontsize=8, color="C1")
    axR.set_xlabel("E [GeV]")
    axR.set_ylabel(r"$\Phi_{\rm near-horizon}/\Phi_{\rm vertical}$")
    axR.set_title(r"Directional enhancement grows with E (sec$\theta$)")
    fig.tight_layout()
    fig.savefig("spherical_cascade.png", dpi=110)
    print("\nsaved plot -> spherical_cascade.png")


if __name__ == "__main__":
    main()
