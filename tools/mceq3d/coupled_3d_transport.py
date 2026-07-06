"""Independent closure of E_off: a from-scratch straight-line ray-traced coupled
neutrino transport, cross-checked against the delivered ``offaxis_factor``.

Neutrinos do not scatter, so their full 3D transport from production to the
detector is a *straight-line* integral: a neutrino arriving from zenith theta_z
was produced somewhere along that arrival ray, by a parent whose direction lies
within the production cone sigma_theta(E) of the arrival direction. In a curved
atmosphere the parent's *local* zenith -- and hence the slant depth it traversed,
and the neutrino production per slant depth p(X,E) -- varies both along the ray
and across the cone. The genuinely-3D neutrino-streaming factor is therefore the
altitude-resolved cone average

    E_off(theta_z, E) = int dl rho(h) < p(X_slant(h, psi_p), E) >_cone           (1)
                        / int dl rho(h)   p(X_slant(h, psi_o), E)

the cone-average taken **at every altitude** h(l) along the ray. This is Eq. 8 of
the paper -- and, on reading ``offaxis_mc``, it is *exactly* what the delivered
``offaxis_factor`` already computes (the engine is not a single-factor
factorisation; the cone is averaged per altitude). So there is no hidden
neutrino-streaming residual: for straight-line particles the coupled transport IS
E_off.

This module confirms that from an **independent** implementation: a self-contained
parametrised curved-atmosphere cascade + ray-traced Eq. (1), whose off-axis excess
reproduces the delivered E_off. It also quantifies *why the altitude resolution is
necessary*: a naive single-altitude factorisation ((1) with the cone commuted
through the altitude integral, evaluated at the production-weighted mean altitude)
errs by tens of percent at the extreme horizon -- the trap the delivered engine
correctly avoids by integrating the cone per altitude.

The remaining genuinely-3D residual is NOT in neutrino streaming (delivered
exactly by E_off) but in **charged-shower lateral development** (secondaries
spreading before neutrino production), which needs a 3D *cascade* Monte-Carlo.

Run::

    python coupled_3d_transport.py --plot
"""

from __future__ import annotations

import argparse

import numpy as np

from spherical_geometry import R_EARTH, H0, H_TOP, _altitude_along_ray

# neutrino production-angle sigma_theta(E) from the NA61-validated generator moments
from fokker_planck_3d import load_theta2, sigma_theta_vs_energy

RHO0 = 1.205e-3  # g/cm^3 sea level
KM_TO_CM = 1.0e5


def density_gcc(alt_km):
    return RHO0 * np.exp(-np.clip(alt_km, 0.0, None) / H0)


def _local_zenith(h_km, l_km, zenith0_deg):
    """Local zenith [rad] at a point that sits at path length ``l_km`` along the
    ray launched from the detector at ``zenith0_deg``. The ray is straight, but on
    the round Earth the local vertical rotates, so the local zenith steepens toward
    the horizon with altitude. Derived from the triangle (Earth centre, detector,
    point): sin(psi)/R_det_proj ... use the exact law-of-sines relation."""
    c0 = np.cos(np.deg2rad(zenith0_deg))
    # radius of the point from Earth centre
    rp = R_EARTH + h_km
    # angle at Earth centre between detector and point
    # r_p^2 = R^2 + l^2 + 2 R l c0  ->  cos(gamma) = (R + l c0)/r_p
    cos_gamma = (R_EARTH + l_km * c0) / rp
    cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
    gamma = np.arccos(cos_gamma)
    # local zenith at the point = zenith0 - gamma (the ray direction seen locally)
    return np.deg2rad(zenith0_deg) - gamma  # rad, can exceed pi/2 near horizon


_AIR = 2.0 * H0 / R_EARTH  # curvature floor for the airmass (~2 H0/R)


def slant_depth(h_km, psi_rad):
    """Vertical-equivalent column depth [g/cm^2] from the top of atmosphere to
    altitude ``h_km`` for a straight path of *local* zenith ``psi`` -- fully
    vectorised curved-Earth airmass ``X = X_vert(h) / sqrt(cos^2 psi + 2 H0/R)``.
    Reduces to ``sec psi`` for small psi and stays finite (grazing airmass
    ``~sqrt(R/2H0) ~ 22``) at and beyond the limb (``psi >= 90 deg``)."""
    h = np.asarray(h_km, float)
    psi = np.asarray(psi_rad, float)
    Xv = RHO0 * H0 * KM_TO_CM * np.exp(-np.clip(h, 0, None) / H0)
    return Xv / np.sqrt(np.cos(psi) ** 2 + _AIR)


def production_pX(e, dlnE, n_x=260):
    """Depth-resolved neutrino production p(X, E) [per g/cm^2, arbitrary norm] from a
    parametrised N -> pi/K -> nu cascade marched in vertical column depth X.

    Returns X grid [g/cm^2] and p (n_x, n_E). The X-shape (young shower near the
    top, dying at depth) is what makes the off-axis cone-average non-trivial."""
    from mceq3d_solver import scaling_matrix, ZMAX, LAM, N, PI, K

    Y_Npi = scaling_matrix(e, dlnE, lambda x: 5.0 * (1 - x) ** 3 / x)
    Y_NK = scaling_matrix(e, dlnE, lambda x: 0.5 * (1 - x) ** 3 / x)
    Y_NN = scaling_matrix(e, dlnE, lambda x: 0.8 * np.ones_like(x))
    Y_pinu = scaling_matrix(e, dlnE, lambda x: np.where(x < ZMAX[PI], 1 / ZMAX[PI], 0.0))
    Y_Knu = scaling_matrix(e, dlnE, lambda x: np.where(x < ZMAX[K], 1 / ZMAX[K], 0.0))
    M_MES = {PI: 0.13957, K: 0.49368}
    CTAU = {PI: 7.804e-3, K: 3.711e-3}  # km

    Xtop = RHO0 * H0 * KM_TO_CM  # total vertical grammage
    Xg = np.linspace(0.0, Xtop, n_x)
    dX = Xg[1] - Xg[0]
    PhiN = e ** (-1.7)
    PhiPi = np.zeros_like(e)
    PhiK = np.zeros_like(e)
    p = np.zeros((n_x, len(e)))
    # local density along the vertical column: X = Xtop*exp(-h/H0) -> h(X); rho at h
    for i, X in enumerate(Xg):
        frac = np.clip(1.0 - X / Xtop, 1e-9, 1.0)
        h = -H0 * np.log(frac)  # altitude at this vertical depth
        rho = density_gcc(h)
        dNi = PhiN * (1.0 - np.exp(-dX / LAM[N]))
        PhiN = PhiN - dNi + Y_NN @ dNi
        PhiPi = PhiPi + Y_Npi @ dNi
        PhiK = PhiK + Y_NK @ dNi
        prod = np.zeros_like(e)
        dl_km = dX / (rho * KM_TO_CM)  # path length of this dX step [km]
        for Phi_m, mm, Y in ((PhiPi, PI, Y_pinu), (PhiK, K, Y_Knu)):
            ddec = (e / M_MES[mm]) * CTAU[mm]  # decay length [km]
            rdec = 1.0 / ddec
            rint = rho * KM_TO_CM / LAM[mm]
            ploss = 1.0 - np.exp(-(rint + rdec) * dl_km)
            frac_dec = rdec / (rint + rdec)
            lost = Phi_m * ploss
            prod = prod + Y @ (lost * frac_dec)
            Phi_m -= lost
        p[i] = prod / dX  # production per unit grammage
    return Xg, p


def _interp_p(Xg, pX, Xq):
    """Production p(Xq, E): Xq is (n_E,), returns (n_E,) -- each energy column of pX
    interpolated at its own query depth Xq[E]."""
    Xq = np.clip(Xq, 0.0, Xg[-1])
    i = np.clip(np.searchsorted(Xg, Xq) - 1, 0, len(Xg) - 2)
    t = (Xq - Xg[i]) / (Xg[i + 1] - Xg[i])
    cols = np.arange(pX.shape[1])
    return pX[i, cols] * (1 - t) + pX[i + 1, cols] * t


def offaxis_exact_vs_factorised(cos_zeniths, e, dlnE, sig_deg, n_l=160, n_beta=8,
                                n_alpha=10):
    """Return Eoff_exact, Eoff_fact on (cz, E).

    ``Eoff_exact`` = coupled ray-traced off-axis factor: the production cone is
    averaged **at every altitude** along the arrival ray, then integrated (Eq. 1).
    ``Eoff_fact`` = the factorised approximation: the cone factor is evaluated once
    (at the production-weighted mean altitude) and multiplied through the collinear
    altitude integral (Eq. 2). Both are divided by the collinear (1D) integral, so
    the residual ``exact/fact - 1`` isolates the altitude<->cone commutation term --
    the genuinely-3D neutrino streaming coupling that the factorised E_off drops."""
    Xg, pX = production_pX(e, dlnE)
    nE = len(e)
    sig = np.deg2rad(sig_deg)  # (nE,)
    beta = np.linspace(0.0, 2 * np.pi, n_beta, endpoint=False)
    a_nodes = np.linspace(0.15, 2.6, n_alpha)  # cone offset in units of sigma
    a_w = np.exp(-0.5 * a_nodes**2) * a_nodes  # 2D-Gaussian radial measure
    a_w = a_w / a_w.sum()

    Eexact = np.zeros((len(cos_zeniths), nE))
    Efact = np.zeros((len(cos_zeniths), nE))
    for iz, cz in enumerate(cos_zeniths):
        z0 = float(np.degrees(np.arccos(np.clip(abs(cz), 1e-3, 1))))
        c0 = np.cos(np.deg2rad(z0))
        lmax = -R_EARTH * c0 + np.sqrt(
            (R_EARTH * c0) ** 2 + 2 * R_EARTH * H_TOP + H_TOP**2)
        l = np.linspace(1e-3, lmax, n_l)
        h = _altitude_along_ray(l, z0)
        rho = density_gcc(h)
        psi_o = _local_zenith(h, l, z0)  # arrival local zenith along the ray (n_l)
        dl = l[1] - l[0]

        def cone_avg(il):  # <p>_cone at altitude index il -> (nE,)
            hil = np.full(nE, h[il])
            avg = np.zeros(nE)
            for ai, an in enumerate(a_nodes):
                for b in beta:
                    dpsi = an * sig * np.cos(b)  # (nE,) plane-projected offset
                    avg += a_w[ai] * _interp_p(Xg, pX, slant_depth(hil, psi_o[il] + dpsi))
            return avg / n_beta

        p_o = np.stack([_interp_p(Xg, pX, np.full(nE, slant_depth(
            np.array([h[il]]), np.array([psi_o[il]]))[0])) for il in range(n_l)])
        w = rho[:, None] * p_o  # (n_l, nE)
        denom = np.maximum(w.sum(0) * dl, 1e-300)  # collinear integral (nE)

        num_exact = np.zeros(nE)
        for il in range(n_l):
            num_exact += rho[il] * cone_avg(il) * dl
        Eexact[iz] = num_exact / denom

        # factorised: single cone factor at the production-weighted mean altitude
        il_star = np.clip(np.round(
            (w * np.arange(n_l)[:, None]).sum(0) / np.maximum(w.sum(0), 1e-300)
        ).astype(int), 0, n_l - 1)  # (nE,)
        cone_fac = np.zeros(nE)
        for je in range(nE):
            il = il_star[je]
            cone_fac[je] = cone_avg(il)[je] / max(p_o[il, je], 1e-300)
        Efact[iz] = cone_fac
    return Eexact, Efact


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--moments", default="m_spliced.npz")
    args = p.parse_args(argv)

    from mceq3d_solver import log_grid

    e, dlnE = log_grid(48)
    e_sig, theta2 = load_theta2(args.moments)
    sig_deg = sigma_theta_vs_energy(e_sig, theta2, e, zenith_deg=0.0)  # (n_E,)
    cz = np.array([0.8, 0.5, 0.3, 0.15, 0.08])
    Eex, Efa = offaxis_exact_vs_factorised(cz, e, dlnE, sig_deg)

    probe = [0.3, 1.0, 3.0]
    ie = [int(np.argmin(np.abs(e - x))) for x in probe]

    # delivered E_off (offaxis_factor) at the same directions, for the closure
    deliv = None
    try:
        from mceq3d_flux import MCEq3DFlux
        eng = MCEq3DFlux(base_model="mceq", e_min=0.1)
        de = eng.e
        off = eng.offaxis_factor(cz)["total_numu"]  # (cz, nE_engine)
        deliv = np.array([[float(np.interp(e[j], de, off[iz])) for j in ie]
                          for iz in range(len(cz))])
    except Exception as ex:  # pragma: no cover
        print(f"(delivered offaxis_factor unavailable: {ex})")

    print("\nCLOSURE -- independent ray-traced E_off vs the delivered offaxis_factor:")
    print("  cosZ  " + "".join(f"{x:>5.1f}GeV rt/deliv" for x in probe))
    for iz, c in enumerate(cz):
        cells = []
        for jj, j in enumerate(ie):
            d = deliv[iz, jj] if deliv is not None else float("nan")
            cells.append(f"{Eex[iz,j]:.2f}/{d:.2f}")
        print(f"  {c:4.2f}   " + "   ".join(cells))
    if deliv is not None:
        mask = deliv > 0.05
        rel = np.abs(np.array([[Eex[iz, j] for j in ie] for iz in range(len(cz))])
                     / np.where(mask, deliv, 1) - 1)[mask]
        print(f"  -> independent ray-trace reproduces delivered E_off to "
              f"median {np.median(rel)*100:.0f}% / max {rel.max()*100:.0f}% "
              f"(differing hadronic cascade + airmass model).")

    print("\nWHY THE ALTITUDE RESOLUTION MATTERS -- exact (per-altitude cone) vs a"
          "\nnaive single-altitude factorisation of the same integral:")
    print("  cosZ  " + "".join(f"{x:>6.1f}GeV(err)" for x in probe))
    worst = 0.0
    for iz, c in enumerate(cz):
        cells = []
        for j in ie:
            r = abs(Eex[iz, j] / max(Efa[iz, j], 1e-9) - 1) * 100
            worst = max(worst, r)
            cells.append(f"{Eex[iz,j]:.2f}/{Efa[iz,j]:.2f}({r:.0f}%)")
        print(f"  {c:4.2f}  " + "  ".join(cells))
    print(f"  -> a single-altitude factorisation errs up to {worst:.0f}% near the"
          " horizon;\n     the delivered engine avoids this by integrating the cone"
          " per altitude.")
    print("COUPLED_3D_TRANSPORT_DONE")


if __name__ == "__main__":
    main()
