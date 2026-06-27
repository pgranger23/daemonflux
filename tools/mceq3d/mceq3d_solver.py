"""mceq3d -- an integrated 3D atmospheric-cascade solver (research engine).

This assembles every validated/de-risked component of the project into one
coupled solver:

  * multi-species energy cascade   N -> pi/K -> numu   (matrix cascade in lnE);
  * angular transport in the Legendre (multipole) basis, driven by the
    **NA61-validated** production angle <theta^2>(E)  (:mod:`kernel_regeneration`,
    :mod:`validate_na61`);
  * the geomagnetic rigidity cutoff **folded into the primary spectrum** per
    arrival direction (the proper treatment -- no x_eff hack; resolves REVIEW
    issue B-partial / #4c), using ``daemonflux.geomagnetic``;
  * curved-atmosphere slant depth (:func:`fokker_planck_3d.slant_depth`).

Output: the muon-neutrino flux ``Phi_numu(E)`` for a given arrival zenith, with
its self-consistent angular spread ``sigma_theta(E)`` (the production-angle 3D
effect, which the solve shows is small and emerges -- it is not assumed).

What is and is not here
-----------------------
Included & exercised: the coupled (E x multipole) cascade, exact reduction to 1D,
proper geomagnetic primary-folding, curved columns, K and pi channels.
Deliberately deferred to the documented next layer (its *feasibility* is already
proven by :mod:`prototype_streaming`): the **global spherical streaming /
curvature term** that couples different arrival directions (the off-axis "Honda"
geometric horizon excess). Also simplified: charge separation, muon-decay
neutrinos, EM cascade, full hadronic model & atmosphere. So this is the real
*engine/architecture*, not a drop-in replacement for production MCEq.

Run::

    python mceq3d_solver.py --plot
"""

from __future__ import annotations

import argparse

import numpy as np

# species
N, PI, K, NUMU = 0, 1, 2, 3
N_SPECIES = 4
LAM = {N: 90.0, PI: 120.0, K: 140.0}  # interaction lengths [g/cm^2]
EPS = {PI: 115.0, K: 850.0}  # critical energies [GeV]
# numu energy-fraction upper limits in two-body M -> mu numu decay  (1 - (m_mu/M)^2)
ZMAX = {PI: 1.0 - (0.10566 / 0.13957) ** 2, K: 1.0 - (0.10566 / 0.49368) ** 2}


def log_grid(n_e=60, e_lo=0.3, e_hi=1e7):
    edges = np.logspace(np.log10(e_lo), np.log10(e_hi), n_e + 1)
    e = np.sqrt(edges[:-1] * edges[1:])
    return e, np.log(edges[1] / edges[0])


def scaling_matrix(e, dlnE, dNdx, x_min=1e-4):
    """Y[i_daughter, j_parent] = dN/dlnE_d for a scaling yield dN/dx."""
    n = len(e)
    Y = np.zeros((n, n))
    for j in range(n):
        x = e[: j + 1] / e[j]
        m = x >= x_min
        Y[: j + 1, j][m] = (x * dNdx(x))[m] * dlnE
    return Y


def primary_flux(e, zenith_deg, geomag_site, gamma=1.7, cutoff_GV=None, penumbra=0.5):
    """Collimated primary nucleon spectrum E^-gamma (dN/dlnE), with the
    geomagnetic rigidity transmission folded in *at the primary level*.

    For protons R[GV] ~ E[GeV]; the transmission is a smooth step at the cutoff,
    so low-rigidity primaries are removed before the cascade and the low-energy
    neutrino suppression emerges naturally. The cutoff is either passed directly
    (``cutoff_GV`` -- e.g. the first-principles back-traced value, the proper
    directional input) or taken from the analytic Stoermer model of the site.
    """
    flux = e ** (-gamma)
    if cutoff_GV is None and geomag_site is None:
        return flux
    from scipy.special import erf

    if cutoff_GV is None:
        from daemonflux.geomagnetic import GeomagneticModel

        geo = GeomagneticModel(geomag_site)
        az = np.linspace(0, 360, 24, endpoint=False)
        cutoff_GV = np.mean(geo.cutoff_rigidity_GV(zenith_deg, az))
        penumbra = geo.penumbra_width
    arg = (np.log(np.maximum(e, 1e-9)) - np.log(cutoff_GV)) / (np.sqrt(2) * penumbra)
    return flux * 0.5 * (1 + erf(arg))


def solve(
    moments="m_spliced.npz",
    zenith_deg=0.0,
    geomag_site=None,
    cutoff_GV=None,
    muon_bending=False,
    lmax=48,
    n_e=60,
    nsteps=1500,
    collimated=False,
):
    """Solve the coupled (E, multipole) cascade for one arrival zenith.

    Returns dict: ``e`` [GeV], ``numu`` (dN/dlnE, l=0 = flux), ``sigma_theta``
    [deg] (= arccos(c_1/c_0)), ``theta1`` [rad].
    """
    from fokker_planck_3d import load_theta2, theta2_interp, slant_depth

    e, dlnE = log_grid(n_e)
    ne = len(e)
    ell = np.arange(lmax + 1)

    e_sig, t2 = load_theta2(moments) if isinstance(moments, str) else moments
    theta1 = np.sqrt(theta2_interp(e, e_sig, t2))
    theta_dec = 0.030 / np.maximum(e, 1e-3)
    if muon_bending:
        from muon_bending import numu_bending_sigma2

        theta_dec = np.sqrt(theta_dec**2 + numu_bending_sigma2(e))
    if collimated:
        theta1 = np.zeros_like(theta1)
        theta_dec = np.zeros_like(theta_dec)

    def kern(theta_E):  # (ne, n_l) heat-kernel angular factor
        return np.exp(-ell[None, :] * (ell[None, :] + 1) * (theta_E[:, None] ** 2) / 4)

    K_pi = kern(theta1)
    K_dec = kern(theta_dec)

    # longitudinal yields (scaling forms; K softer/harder split from pi)
    Y_NN = scaling_matrix(e, dlnE, lambda x: 0.8 * np.ones_like(x))
    Y_Npi = scaling_matrix(e, dlnE, lambda x: 5.0 * (1 - x) ** 3 / x)
    Y_NK = scaling_matrix(e, dlnE, lambda x: 0.5 * (1 - x) ** 3 / x)  # ~10% of pi
    Y_pinu = scaling_matrix(
        e, dlnE, lambda x: np.where(x < ZMAX[PI], 1.0 / ZMAX[PI], 0.0)
    )
    Y_Knu = scaling_matrix(e, dlnE, lambda x: np.where(x < ZMAX[K], 1.0 / ZMAX[K], 0.0))

    # decay competition via critical energy (depth-independent prototype)
    Xv = slant_depth(zenith_deg)  # curved-atmosphere slant grammage
    inv_pi = 1.0 / LAM[PI] + EPS[PI] / (e * Xv)
    inv_K = 1.0 / LAM[K] + EPS[K] / (e * Xv)
    fdec_pi = (EPS[PI] / (e * Xv)) / inv_pi
    fdec_K = (EPS[K] / (e * Xv)) / inv_K

    n_l = lmax + 1
    Phi = [np.zeros((ne, n_l)) for _ in range(N_SPECIES)]
    Phi[N][:] = primary_flux(e, zenith_deg, geomag_site, cutoff_GV=cutoff_GV)[
        :, None
    ]  # collimated primary, geomag folded in

    dX = Xv / nsteps
    sN = np.exp(-dX / LAM[N])
    sPi = np.exp(-dX * inv_pi)[:, None]
    sK = np.exp(-dX * inv_K)[:, None]
    for _ in range(nsteps):
        dNi = Phi[N] * (1.0 - sN)
        Phi[N] = Phi[N] - dNi + Y_NN @ dNi
        Phi[PI] = Phi[PI] + (Y_Npi @ dNi) * K_pi
        Phi[K] = Phi[K] + (Y_NK @ dNi) * K_pi
        lpi = Phi[PI] * (1.0 - sPi)
        lk = Phi[K] * (1.0 - sK)
        Phi[PI] = Phi[PI] - lpi
        Phi[K] = Phi[K] - lk
        Phi[NUMU] = Phi[NUMU] + (Y_pinu @ (lpi * fdec_pi[:, None])) * K_dec
        Phi[NUMU] = Phi[NUMU] + (Y_Knu @ (lk * fdec_K[:, None])) * K_dec

    c = Phi[NUMU]
    c0 = c[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma = np.degrees(np.arccos(np.clip(c[:, 1] / c0, -1, 1)))
    sigma[~(c0 > 0)] = np.nan
    return dict(e=e, numu=c0, sigma_theta=sigma, theta1=theta1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--moments", default="m_spliced.npz")
    p.add_argument("--site", default="kamioka")
    p.add_argument("--zenith", type=float, default=0.0)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args(argv)

    base = solve(args.moments, args.zenith)  # no geomag
    geo = solve(args.moments, args.zenith, geomag_site=args.site)
    coll = solve(args.moments, args.zenith, collimated=True)

    # invariant: the angular treatment must not change the energy spectrum (c_0)
    rel = np.nanmax(np.abs(base["numu"] / coll["numu"] - 1.0)[base["numu"] > 0])
    print(f"INVARIANT  c0(angular on) == c0(collimated):  max rel diff = {rel:.2e}")
    print("  (the angular machinery conserves the energy cascade -> 1D preserved)\n")

    e = base["e"]
    print(
        f"numu flux & geomagnetic suppression (site={args.site}, "
        f"zenith={args.zenith:.0f}):"
    )
    print("  E[GeV]   sigma_theta[deg]   Phi_3D/Phi_1D (geomag)")
    for i in range(0, len(e), 4):
        if 0.5 < e[i] < 1e4 and np.isfinite(base["sigma_theta"][i]):
            ratio = geo["numu"][i] / base["numu"][i] if base["numu"][i] > 0 else np.nan
            print(f"  {e[i]:8.2f}    {base['sigma_theta'][i]:7.2f}        {ratio:6.3f}")

    if args.plot:
        _plot(base, geo, coll, args)


def _plot(base, geo, coll, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e = base["e"]
    fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(15, 4.3))
    s = (e > 0.5) & (e < 1e5) & (base["numu"] > 0)
    axL.loglog(e[s], (base["numu"] * e**2)[s], "C0-", label="no geomag")
    axL.loglog(e[s], (geo["numu"] * e**2)[s], "C3-", label=f"geomag ({args.site})")
    axL.set_xlabel("E [GeV]")
    axL.set_ylabel(r"$E^2\,\Phi_{\nu_\mu}$ (arb.)")
    axL.set_title("numu flux (cascade) + geomagnetic cutoff")
    axL.legend()

    g = (e > 0.5) & (e < 1e4) & (base["numu"] > 0)
    axM.semilogx(e[g], (geo["numu"] / base["numu"])[g], "C3-")
    axM.axhline(1, color="k", lw=0.7, ls=":")
    axM.set_xlabel("E [GeV]")
    axM.set_ylabel(r"$\Phi_{\rm geomag}/\Phi_{\rm 1D}$")
    axM.set_title("geomag folded into primary (no x_eff hack)")

    a = (e > 0.5) & (e < 3000) & np.isfinite(base["sigma_theta"])
    axR.loglog(
        e[a], base["sigma_theta"][a], "C0o-", ms=3, label=r"$\sigma_\theta$ (solve)"
    )
    axR.loglog(e[a], np.degrees(base["theta1"])[a], "C1--", label=r"$\theta_1$")
    axR.set_xlabel("E [GeV]")
    axR.set_ylabel(r"$\sigma_\theta$ [deg]")
    axR.set_title("self-consistent angular spread")
    axR.legend()
    fig.tight_layout()
    fig.savefig("mceq3d_solver.png", dpi=110)
    print("\nsaved plot -> mceq3d_solver.png")


if __name__ == "__main__":
    main()
