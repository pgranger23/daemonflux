"""Muon-segment Monte Carlo: the *exact* bending offset between the neutrino
arrival direction and the parent primary direction, with energy loss and the
real IGRF-13 field along the muon's own path.

WHY
---
The delivered engine (``mceq3d_flux.cone_geff`` / ``joint_cone``) models
secondary-muon bending as a **single coherent shift** of the muon-decay cone
axis by the mean in-flight rotation

    Delta = q B tau_mu / m_mu = 5.08 deg   (|B| = 0.474 G at Kamioka)

evaluated with the field vector **at the detector**, applied with an ad-hoc
weight ``f_dec/(1 + f_dec)``.  Three things were suspected to be missing:

1. muon **energy loss** (a decelerating muon has a shrinking gyroradius, so it
   should bend more);
2. the field at the muon's **actual position** (tens of km up, hundreds of km
   away from the detector near the horizon) rather than at the detector;
3. the **distribution** of the bend rather than its mean.

THE KINEMATIC IDENTITY THAT SETTLES (1)
---------------------------------------
The velocity direction of a charged particle in a static field rotates about
``B_hat`` at the lab rate ``omega = qB/(gamma m)``.  Over a lab time ``t`` the
rotation angle is ``qBt/(gamma m) = q B T_p / m`` with ``T_p = t/gamma`` the
**proper** time.  Equivalently ``dtheta/ds = qB_perp/p`` and
``ds = beta c gamma dT_p``, ``p = gamma m beta c``, so

    dtheta = q B_perp dT_p / m        (exactly, at every energy)

The accumulated bend depends **only on the proper time elapsed and on B along
the path** -- not on the muon energy, and *not* on how much energy the muon
lost.  Energy loss changes the *path length* per unit proper time, and the
``ln(p_i/p_f)`` formula is just the same integral rewritten; it can only give a
larger answer if the muon is allowed to live longer than its proper lifetime.
Energy loss enters this problem only **indirectly**, through which muons can
still make a neutrino of the requested energy (see below).

WHAT THIS MODULE COMPUTES
-------------------------
For a fixed neutrino energy ``E_nu`` and a fixed arrival direction ``n``
(zenith/azimuth at Kamioka, the *from* direction) it evaluates the differential
flux integral

    Phi(E_nu, n) ~ Int dL Int dE_dec Int dOmega_dec Int dT_p
                   (1/tau) e^{-T_p/tau} (gb)_prod/(gb)_dec
                   S(X_prod, E_prod) J_E  x  dN/(dE_nu dOmega_nu)

by weighted Monte Carlo, and records **per sample** the rotation the muon
underwent between production and decay.  Because the neutrino free-streams, the
decay point lies exactly on the arrival ray; the scheme is therefore *exact for
the observable* -- the arrival direction and energy are fixed by construction
(a delta function), never by a rejection cone:

* the **decay point** ``r_dec = r_det + L n`` is sampled along the arrival ray
  (importance-sampled on a pilot production profile);
* the **muon direction at decay** is generated *backwards from the required
  neutrino direction*: ``u_nu`` is fixed to ``-n`` and the muon direction is
  sampled around it with the exact boosted-Michel lab density
  ``dN/(dE_nu dOmega_lab) = (E_nu/E*) g(x*)/(4 pi) (2/m_mu)`` (the sampling
  variable is ``t = 1 - beta cos theta_lab``, which is *the* variable in which
  the kinematically allowed set ``x* <= 1`` is an interval);
* the **proper time since production** is sampled from ``Exp(tau_mu)`` (exactly
  the decay law) and the trajectory is integrated **backwards** to the
  production point with continuous energy loss, full IGRF-13 at the muon's
  position, and curved-Earth geometry;
* the sample is weighted by the depth-resolved muon production density at the
  production point, by the energy-loss phase-space Jacobian
  ``exp(Int db/dE ds)``, and by the proposal densities.

The **primary-direction offset** delivered to the cutoff lookup is the
accumulated rotation applied to the arrival direction itself,
``n_prim = R_total n`` -- deliberately *not* ``-u_prod``, because the Michel
opening angle and the pi->mu / hadronic angles are already carried by the
engine's separate muon-decay cone width ``sigma_mu``
(``kinematic_kernel.mudecay_shape_mc``).  This module isolates the piece the
engine models as a single ``+-delta``.

MUON PRODUCTION
---------------
MCEq's depth-resolved **pi+/pi- and K+/K- fluxes** are folded with exact
two-body ``pi/K -> mu nu`` kinematics (flat in E_mu between ``r E_parent`` and
``E_parent``), giving a *charge-separated* muon source per unit path length,

    S(X, E_mu) = Int dE_P Phi_P(X, E_P) / (gamma beta c tau_P) / ((1-r) E_P) .

MCEq's ``total_mu+`` gradient in X is **not** a production term (the muon sink
from decay and energy loss is large at these energies), which is why the parent
route is used; this is the approximation flagged in the task statement.  K->mu
is included (BR 0.6355).  Muons from mu-pair/charm are neglected.

ASSUMPTIONS (all stated, none hidden)
-------------------------------------
* Michel decay is treated as **unpolarised** by default.  ``polarisation`` lets
  a rest-frame asymmetry ``1 + A(x*) P cos(theta*)`` be switched on to bound the
  effect (it re-weights which E_mu makes a given E_nu, so it can only move the
  bend distribution through the energy-loss selection, second order).
* the pion/kaon production profile is the MCEq ``theta_deg=0`` cascade read as a
  function of slant depth X (the ``E_off`` universality assumption of
  ``offaxis_mc``); the residual rho-dependence of the decay/interaction
  competition along tilted paths is dropped.
* dE/dx is the PDG air total stopping power (table below), +-~5%; the muon
  multiple-scattering angle is neglected (~0.3 deg over 100 g/cm2 at 1 GeV, in
  quadrature with a >3 deg coherent bend).
* spherical Earth, R = 6371 km, top of atmosphere 112.8 km (offaxis_mc).
* everything is done in the right-handed **ENU** (east, north, up) frame; the
  ``(north, east, up)`` ordering used inside ``cone_geff`` is LEFT handed and a
  cross product taken in it silently flips the charge assignment.
  :func:`check_handedness` asserts the sign against
  ``muon_bending.bending_deflection``.

Run::

    python muon_segment_mc.py --demo
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
C_M_S = 2.99792458e8
TAU_MU = 2.1969811e-6            # s
M_MU = 0.1056583745             # GeV
M_PI = 0.13957039               # GeV
M_K = 0.493677                  # GeV
CTAU_PI_CM = 780.45             # c tau_pi [cm]
CTAU_K_CM = 371.2               # c tau_K  [cm]
BR_K_MU = 0.6355
R_PI = (M_MU / M_PI) ** 2       # 0.5731
R_K = (M_MU / M_K) ** 2         # 0.0458
Q_E = 1.602176634e-19
M_MU_KG = 1.883531627e-28

RE_M = 6371.0e3
RE_CM = 6371.0e5
H_TOP_CM = 112.8e5
H_TOP_M = 112.8e3

LAT, LON = 36.43, 137.31
DATE = datetime(2020, 1, 1)

_HERE = os.path.dirname(os.path.abspath(__file__))
JOINTPROD = os.path.join(
    _HERE, ".cache3d", "jointprod_SIBYLL23D_HillasGaisser2012-H3a.npz")
SCRATCH = os.environ.get(
    "MSMC_SCRATCH",
    "/tmp/pigrange/claude-130233/-afs-cern-ch-work-p-pigrange-daemonflux/"
    "528f6ded-1da2-4729-9201-8e34867d3ae7/scratchpad")


# ---------------------------------------------------------------------------
# atmosphere and slant-depth geometry (from the engine's own cached tables)
# ---------------------------------------------------------------------------
def load_atmosphere(cache=JOINTPROD):
    """``(rho(h_cm) [g/cm3], (h_grid, psi_grid, xtable))`` from the joint-cone cache.

    Falls back to re-deriving them from MCEq if the cache is absent.
    """
    if os.path.exists(cache):
        d = np.load(cache)
        h_tab, rho_tab = d["rho_h"], d["rho_v"]

        def rho(h_cm):
            return np.interp(h_cm, h_tab, rho_tab, left=rho_tab[0], right=0.0)

        return rho, (d["h_grid"], d["psi_grid"], d["table"])
    import offaxis_mc as ox

    _, _, _, dm = ox.production_profile(n_x=40)
    rho = ox._rho_of_h(dm)
    return rho, ox.slant_depth_table(rho)


def xslant(h_cm, psi_rad, geom):
    """Slant column [g/cm2] from altitude ``h_cm`` to the top of the atmosphere
    along a line whose *local zenith* is ``psi``."""
    import offaxis_mc as ox

    return ox._interp_xslant(np.asarray(h_cm, float), np.asarray(psi_rad, float),
                             geom[0], geom[1], geom[2])


# ---------------------------------------------------------------------------
# muon source: MCEq depth-resolved pi/K flux folded with two-body decay
# ---------------------------------------------------------------------------
def muon_source_table(cache=None, n_x=90, theta_deg=0.0, e_lo=0.1, e_hi=400.0,
                      interaction_model="SIBYLL23D", primary="H3a",
                      method="mu_transport"):
    """``dict(x_grid, e_grid, s_plus, s_minus)`` -- charge-separated muon source
    per unit **slant depth**, ``S_X(X, E_mu)`` [g^-1 cm^2 s^-1 sr^-1 GeV^-1].

    Per unit *path length* the source is ``rho_local * S_X`` with ``rho_local``
    the air density at the actual production point -- **not** the density at the
    depth ``X`` in the vertical atmosphere.  The two differ by three orders of
    magnitude on a near-horizontal ray (X = 22 g/cm2 sits at 26 km vertically but
    at 83 km along a limb-grazing line), so the callers here always multiply by
    the local rho.

    Two routes, both from the same MCEq depth-resolved cascade:

    ``"mu_transport"`` (default) inverts the muon transport equation on MCEq's
    own depth-resolved ``mu+/mu-`` flux,

        S_X = dPhi/dX - d(b Phi)/dE + Phi / lambda_dec ,

    with ``b = dE/dX`` the air stopping power and
    ``lambda_dec = rho gamma beta c tau_mu``.  This is *not* the naive
    ``dPhi/dX`` used for neutrinos in ``offaxis_mc``: below a few GeV the decay
    sink alone is comparable to the gradient, and the continuous-loss divergence
    is of the same size again.  It is the only route that works at
    ``E_mu < 8 GeV``, because MCEq does not track ``pi+/K+`` as propagating
    states there (they are resonances, folded straight into their daughters), so
    ``get_solution("pi+")`` is identically zero below ~8 GeV.

    ``"pi_decay"`` folds MCEq's ``pi+/pi-`` and ``K+/K-`` fluxes with exact
    two-body kinematics.  It is only valid above the tracking threshold and is
    kept as an independent cross-check of the transport inversion there.
    """
    cache = cache or os.path.join(SCRATCH, f"muon_source_{method}.npz")
    if cache and os.path.exists(cache):
        d = np.load(cache)
        return {k: d[k] for k in ("x_grid", "e_grid", "s_plus", "s_minus")}

    import importlib.util  # noqa: F401  (MCEq config touches importlib.util)
    import mceq_config as cfg

    cfg.e_min = e_lo
    from MCEq.core import MCEqRun
    import crflux.models as crf

    mc = MCEqRun(interaction_model=interaction_model,
                 primary_model=(crf.HillasGaisser2012, primary),
                 theta_deg=theta_deg)
    xmax = float(mc.density_model.max_X)
    x_grid = np.linspace(xmax / n_x, xmax, n_x)
    mc.solve(int_grid=x_grid, grid_var="X")
    e = mc.e_grid
    sel = (e >= e_lo) & (e <= e_hi)
    eg = e[sel]

    def flux(key):
        return np.array([mc.get_solution(key, 0, grid_idx=i)[sel]
                         for i in range(n_x)])       # (n_x, nE)

    rho_x = 1.0 / mc.density_model.r_X2rho(np.clip(x_grid, 1e-6, None))  # g/cm3
    src = {}
    for sign, tag in ((+1, "plus"), (-1, "minus")):
        if method == "mu_transport":
            phi = flux(f"mu{'+' if sign > 0 else '-'}")
            p_mu = np.sqrt(np.maximum(eg**2 - M_MU**2, 1e-12))
            ctau_mu_cm = C_M_S * TAU_MU * 100.0
            # decay sink per g/cm2: 1/(rho gamma beta c tau)
            dec = M_MU / (rho_x[:, None] * p_mu[None, :] * ctau_mu_cm)
            bE = dedx_air(eg)                                   # GeV cm2/g
            dbphi = np.gradient(bE[None, :] * phi, eg, axis=1)
            s_x = np.gradient(phi, x_grid, axis=0) - dbphi + phi * dec
            tot = np.clip(s_x, 0.0, None)                       # per g/cm2
        else:
            tot = np.zeros((n_x, len(eg)))
            for name, mass, ctau, rr, br in (
                    (f"pi{'+' if sign > 0 else '-'}", M_PI, CTAU_PI_CM, R_PI, 1.0),
                    (f"K{'+' if sign > 0 else '-'}", M_K, CTAU_K_CM, R_K,
                     BR_K_MU)):
                phi = flux(name)
                above = eg > mass * 1.0001
                p_par = np.sqrt(np.maximum(eg**2 - mass**2, 1e-12))
                dec = np.where(above, mass / (p_par * ctau), 0.0)   # [1/cm]
                w = phi * dec[None, :] * br / ((1.0 - rr) * eg[None, :])
                tot += _fold_flat(w, eg, rr) / rho_x[:, None]   # -> per g/cm2
        src[tag] = tot
    out = dict(x_grid=x_grid, e_grid=eg, s_plus=src["plus"], s_minus=src["minus"])
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, **out)
    return out


def _fold_flat(w, e_grid, r, n_sub=64):
    """``S(E_mu) = Int_{E_mu}^{E_mu/r} dE_P w(E_P)`` on the log-spaced ``e_grid``.

    ``w`` is (n_x, nE) already carrying the ``1/((1-r) E_P)`` factor.  Midpoint
    quadrature in ``ln E_P`` with linear interpolation of ``w`` -- deliberately
    NOT a difference of cumulative sums, which loses all precision on a spectrum
    spanning ten decades.
    """
    le = np.log(e_grid)
    lo = le
    hi = np.minimum(le - np.log(r), le[-1])
    span = np.clip(hi - lo, 0.0, None)                    # (nE,)
    u = (np.arange(n_sub) + 0.5) / n_sub
    lp = lo[:, None] + u[None, :] * span[:, None]         # (nE, n_sub)
    j = np.clip(np.searchsorted(le, lp) - 1, 0, len(le) - 2)
    f = np.clip((lp - le[j]) / (le[j + 1] - le[j]), 0.0, 1.0)
    wl = w[:, j] * (1 - f)[None, :, :] + w[:, j + 1] * f[None, :, :]
    ep = np.exp(lp)                                       # dE_P = E_P dlnE_P
    return np.clip((wl * ep[None, :, :]).sum(-1) * (span / n_sub)[None, :],
                   0.0, None)


def source_interp(tab):
    """``S(X, E_mu, charge)`` -- bilinear in (X, lnE), zero outside."""
    xg, eg = tab["x_grid"], tab["e_grid"]
    lg = np.log(eg)
    sp, sm = tab["s_plus"], tab["s_minus"]

    def S(x, e, charge):
        s = sp if charge > 0 else sm
        x = np.asarray(x, float)
        le = np.log(np.clip(np.asarray(e, float), eg[0], eg[-1]))
        ix = np.clip(np.searchsorted(xg, x) - 1, 0, len(xg) - 2)
        ie = np.clip(np.searchsorted(lg, le) - 1, 0, len(lg) - 2)
        tx = np.clip((x - xg[ix]) / (xg[ix + 1] - xg[ix]), 0.0, 1.0)
        te = (le - lg[ie]) / (lg[ie + 1] - lg[ie])
        v = (s[ix, ie] * (1 - tx) * (1 - te) + s[ix + 1, ie] * tx * (1 - te)
             + s[ix, ie + 1] * (1 - tx) * te + s[ix + 1, ie + 1] * tx * te)
        # below the first tabulated depth the cascade has barely started:
        # production is linear in X (one interaction length has not elapsed),
        # so scale the first row down rather than dropping it.  Beyond the
        # tabulated depth (>1033 g/cm2, only reached on the low-altitude part of
        # a near-horizontal ray) the pion cascade is attenuated by >e^-8 and is
        # set to zero.
        v = np.where(x < xg[0], v * np.clip(x / xg[0], 0.0, 1.0), v)
        bad = (x > xg[-1]) | ~np.isfinite(x) | (x < 0.0)
        return np.where(bad, 0.0, np.clip(v, 0.0, None))

    return S


# ---------------------------------------------------------------------------
# energy loss: PDG total stopping power of muons in air
# ---------------------------------------------------------------------------
_DEDX_P = np.array([0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0,
                    10.0, 20.0, 50.0, 100.0, 1000.0])          # GeV/c
_DEDX_V = np.array([9.20, 6.05, 3.62, 2.35, 2.05, 1.88, 1.84, 1.92, 2.06,
                    2.19, 2.32, 2.51, 2.66, 3.30])             # MeV cm2/g


def dedx_air(e_gev):
    """Total muon stopping power in air [GeV cm2/g] vs total energy."""
    e = np.asarray(e_gev, float)
    p = np.sqrt(np.maximum(e**2 - M_MU**2, 1e-8))
    return 1e-3 * np.exp(np.interp(np.log(p), np.log(_DEDX_P), np.log(_DEDX_V)))


def _dbde(e_gev, rho):
    """``d(rho dE/dx)/dE`` [1/cm], the energy-loss phase-space compression rate."""
    h = 1e-3 * np.maximum(e_gev, 0.2)
    return rho * (dedx_air(e_gev + h) - dedx_air(e_gev - h)) / (2 * h)


# ---------------------------------------------------------------------------
# IGRF-13 field, interpolated on a local grid (ppigrf is far too slow per step)
# ---------------------------------------------------------------------------
class FieldGrid:
    """Trilinear IGRF-13 in geocentric Cartesian components [T].

    One ``ppigrf.igrf_gc`` call on a (r, colat, lon) grid covering the region a
    muon feeding a Kamioka arrival ray can occupy; the Cartesian components are
    smooth on this scale (checked to <1e-3 relative against a direct call).
    """

    def __init__(self, lat=LAT, lon=LON, date=DATE, dcolat=16.0, dlon=20.0,
                 h_max_km=130.0, n_r=14, n_t=33, n_p=41):
        import ppigrf

        colat0 = 90.0 - lat
        self.r = np.linspace(RE_M, RE_M + h_max_km * 1e3, n_r)
        self.t = np.linspace(colat0 - dcolat, colat0 + dcolat, n_t)
        self.p = np.linspace(lon - dlon, lon + dlon, n_p)
        R, T, P = np.meshgrid(self.r, self.t, self.p, indexing="ij")
        Br, Bt, Bp = (np.ravel(c) * 1e-9
                      for c in ppigrf.igrf_gc(R.ravel() / 1e3, T.ravel(),
                                              P.ravel(), date))
        th, ph = np.deg2rad(T.ravel()), np.deg2rad(P.ravel())
        st, ct, sp, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
        bx = Br * st * cp + Bt * ct * cp - Bp * sp
        by = Br * st * sp + Bt * ct * sp + Bp * cp
        bz = Br * ct - Bt * st
        self.B = np.stack([bx, by, bz], -1).reshape(n_r, n_t, n_p, 3)

    def __call__(self, xyz):
        """``B`` [T] (N,3) geocentric Cartesian at positions ``xyz`` [m] (N,3)."""
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        rr = np.sqrt(x * x + y * y + z * z)
        tt = np.degrees(np.arccos(np.clip(z / rr, -1, 1)))
        pp = np.degrees(np.arctan2(y, x))
        i = np.clip(np.searchsorted(self.r, rr) - 1, 0, len(self.r) - 2)
        j = np.clip(np.searchsorted(self.t, tt) - 1, 0, len(self.t) - 2)
        k = np.clip(np.searchsorted(self.p, pp) - 1, 0, len(self.p) - 2)
        fi = np.clip((rr - self.r[i]) / (self.r[i + 1] - self.r[i]), 0, 1)[:, None]
        fj = np.clip((tt - self.t[j]) / (self.t[j + 1] - self.t[j]), 0, 1)[:, None]
        fk = np.clip((pp - self.p[k]) / (self.p[k + 1] - self.p[k]), 0, 1)[:, None]
        B = self.B
        c00 = B[i, j, k] * (1 - fi) + B[i + 1, j, k] * fi
        c01 = B[i, j, k + 1] * (1 - fi) + B[i + 1, j, k + 1] * fi
        c10 = B[i, j + 1, k] * (1 - fi) + B[i + 1, j + 1, k] * fi
        c11 = B[i, j + 1, k + 1] * (1 - fi) + B[i + 1, j + 1, k + 1] * fi
        c0 = c00 * (1 - fj) + c10 * fj
        c1 = c01 * (1 - fj) + c11 * fj
        return c0 * (1 - fk) + c1 * fk


# ---------------------------------------------------------------------------
# local frames
# ---------------------------------------------------------------------------
def enu_basis(lat_deg, lon_deg):
    """``(east, north, up)`` unit vectors in geocentric Cartesian."""
    la, lo = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    up = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    north = np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo),
                      np.cos(la)])
    east = np.array([-np.sin(lo), np.cos(lo), 0.0])
    return east, north, up


def arrival_enu(zen_deg, az_deg):
    """Arrival (*from*) unit vector in ENU for a compass azimuth."""
    th, ph = np.deg2rad(zen_deg), np.deg2rad(az_deg)
    return np.array([np.sin(th) * np.sin(ph), np.sin(th) * np.cos(ph),
                     np.cos(th)])


def enu_angles(v):
    """``(zenith, azimuth)`` [deg, compass] of ENU vectors ``v`` (...,3)."""
    v = np.asarray(v, float)
    th = np.degrees(np.arccos(np.clip(v[..., 2], -1, 1)))
    ph = np.degrees(np.arctan2(v[..., 0], v[..., 1])) % 360.0
    return th, ph


def to_geocentric(v_enu, basis):
    e, n, u = basis
    v = np.atleast_2d(np.asarray(v_enu, float))
    return v[:, 0:1] * e + v[:, 1:2] * n + v[:, 2:3] * u


def to_enu(v_gc, basis):
    e, n, u = basis
    v = np.atleast_2d(np.asarray(v_gc, float))
    return np.stack([v @ e, v @ n, v @ u], -1)


def rodrigues(v, k, ang):
    """Rotate ``v`` (N,3) about unit axes ``k`` (N,3 or 3) by ``ang`` (N,) [rad]."""
    v = np.asarray(v, float)
    k = np.broadcast_to(np.asarray(k, float), v.shape)
    a = np.asarray(ang, float)[..., None]
    c, s = np.cos(a), np.sin(a)
    return v * c + np.cross(k, v) * s + k * (np.sum(k * v, -1)[..., None]) * (1 - c)


# ---------------------------------------------------------------------------
# Michel decay (unpolarised by default)
# ---------------------------------------------------------------------------
def michel_n(x, kind):
    """Rest-frame spectrum ``dN/dx``, ``x = 2E*/m_mu``, normalised on [0,1]."""
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    if kind == "numu":                # the same-flavour neutrino
        return 2.0 * x * x * (3.0 - 2.0 * x)
    return 12.0 * x * x * (1.0 - x)   # the electron-flavour neutrino


def michel_asym(x, kind):
    """Rest-frame asymmetry ``A(x)`` in ``1 + A(x) P cos(theta*)``."""
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    if kind == "numu":
        return (1.0 - 2.0 * x) / np.maximum(3.0 - 2.0 * x, 1e-12)
    return np.ones_like(x)


# ---------------------------------------------------------------------------
# the segment Monte Carlo
# ---------------------------------------------------------------------------
def segment_mc(zen_deg, az_deg, e_nu, charge, n=20000, seed=0, *,
               field=None, rho=None, geom=None, source=None,
               lat=LAT, lon=LON, e_ratio_max=40.0, e_index=3.0, t_max_tau=8.0,
               dt_tau=1.0 / 60.0, ds_max_m=1500.0, max_steps=1800,
               energy_loss=True, polarisation=0.0, const_field=None,
               prod_dir="muon"):
    """Weighted MC of the muon segment feeding ``(E_nu, zenith, azimuth)``.

    Returns a dict of per-sample arrays.  ``w_numu`` / ``w_nue`` are the two
    Michel weights on the *same* samples (the proposal does not depend on the
    neutrino species), so one run serves both ``nu_mu``-type and ``nu_e``-type
    daughters of the given muon ``charge``.

    ``charge`` is the muon charge (+1 / -1).  By lepton-flavour conservation the
    ``mu+`` daughters are ``(nubar_mu, nu_e)`` and the ``mu-`` daughters are
    ``(nu_mu, nubar_e)``.

    Keys: ``w_numu``, ``w_nue`` (weights), ``bend_deg`` (space angle between the
    arrival direction and the rotated primary direction), ``psi_deg`` (the
    signed rotation angle about B), ``dzen``/``daz`` [deg] (primary minus
    arrival), ``n_prim`` (N,3, ENU), ``T_p`` [s], ``e_dec``, ``e_prod``,
    ``ratio`` (= p_prod/p_dec), ``L_km``, ``h_dec_km``, ``h_prod_km``,
    ``x_prod``, ``theta_lab_deg``, ``alive``.
    """
    rng = np.random.default_rng(seed)
    if rho is None or geom is None:
        rho, geom = load_atmosphere()
    if source is None:
        source = source_interp(muon_source_table())
    if field is None and const_field is None:
        field = FieldGrid(lat, lon)
    basis = enu_basis(lat, lon)
    r_det = np.asarray(basis[2]) * RE_M
    n_enu = arrival_enu(zen_deg, az_deg)
    n_gc = to_geocentric(n_enu, basis)[0]

    # ---- pilot profile along the arrival ray, for importance sampling of L --
    L_top = _ray_length_to_alt(zen_deg, H_TOP_M)
    Lg = np.linspace(1e3, L_top, 800)
    P = r_det[None, :] + Lg[:, None] * n_gc[None, :]
    rr = np.linalg.norm(P, axis=1)
    h_g = rr - RE_M
    cos_psi = (P @ n_gc) / rr
    psi_g = np.arccos(np.clip(cos_psi, -1, 1))
    x_g = xslant(h_g * 100.0, psi_g, geom)
    pilot = rho(np.clip(h_g, 0.0, H_TOP_M) * 100.0) * source(
        x_g, 2.5 * e_nu, charge) + 1e-300
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pilot[1:] + pilot[:-1])
                                           * np.diff(Lg))])
    if cdf[-1] <= 0:
        raise RuntimeError("empty pilot production profile along the ray")
    # defensive mixture: 85% from the pilot CDF, 15% uniform along the ray, so
    # a region the pilot under-rates cannot produce an unbounded 1/q_L spike.
    span_L = Lg[-1] - Lg[0]
    pdf_L = 0.85 * pilot / cdf[-1] + 0.15 / span_L
    u = rng.random(n)
    L = np.where(rng.random(n) < 0.85,
                 np.interp(u * cdf[-1], cdf, Lg), Lg[0] + u * span_L)
    q_L = np.interp(L, Lg, pdf_L)

    # ---- decay-point kinematics --------------------------------------------
    r_dec = r_det[None, :] + L[:, None] * n_gc[None, :]
    # muon energy at decay: E^-e_index proposal on [E_nu, e_ratio_max E_nu]
    a, b = max(e_nu, M_MU * 1.001), e_ratio_max * e_nu
    k = e_index - 1.0
    ue = rng.random(n)
    e_dec = (a**-k - ue * (a**-k - b**-k)) ** (-1.0 / k)
    q_E = k * e_dec ** (-e_index) / (a**-k - b**-k)
    g_d = e_dec / M_MU
    b_d = np.sqrt(np.maximum(1.0 - 1.0 / g_d**2, 0.0))

    # ---- boosted Michel: sample t = 1 - beta cos(theta_lab) ----------------
    t_lo = 1.0 - b_d
    t_hi = np.minimum(1.0 + b_d, M_MU / (2.0 * g_d * e_nu))
    ok = t_hi > t_lo
    t_hi = np.where(ok, t_hi, t_lo + 1e-12)
    t_s = t_lo + rng.random(n) * (t_hi - t_lo)
    cos_lab = np.clip((1.0 - t_s) / np.maximum(b_d, 1e-12), -1.0, 1.0)
    e_star = g_d * e_nu * t_s
    x_star = np.clip(2.0 * e_star / M_MU, 1e-12, 1.0)
    # cos(theta*) of the neutrino w.r.t. the muon direction, rest frame
    cos_star = np.clip((cos_lab - b_d) / np.maximum(1.0 - b_d * cos_lab, 1e-12),
                       -1.0, 1.0)
    jac_dOmega = 2.0 * np.pi * (t_hi - t_lo) / np.maximum(b_d, 1e-12)
    pref = (e_nu / np.maximum(e_star, 1e-12)) * (2.0 / M_MU) / (4.0 * np.pi)
    w_mich = {k: pref * michel_n(x_star, k) for k in ("numu", "nue")}
    if polarisation:
        for k in w_mich:
            w_mich[k] = w_mich[k] * np.clip(
                1.0 + polarisation * michel_asym(x_star, k) * cos_star, 0.0, None)

    # muon travel direction at decay: rotate the neutrino travel direction
    u_nu = -n_gc
    u_dec = _cone_dirs(rng, u_nu, np.arccos(cos_lab))

    # ---- backward integration ----------------------------------------------
    T_star = -TAU_MU * np.log1p(-rng.random(n) * (1.0 - np.exp(-t_max_tau)))
    st = _backward(r_dec, u_dec, e_dec, T_star, charge,
                   field=field, rho=rho, const_field=const_field,
                   energy_loss=energy_loss, ref=np.broadcast_to(n_gc, (n, 3)),
                   dt_tau=dt_tau, ds_max_m=ds_max_m, max_steps=max_steps)

    e_prod, r_prod, n_prim_gc, lnJ, alive = (
        st["e"], st["r"], st["ref"], st["lnJ"], st["alive"])
    exhausted = st["exhausted"]
    psi = st["psi"]

    h_prod = np.linalg.norm(r_prod, axis=1) - RE_M
    u_prod = st["u"]
    # slant depth at the production point.  ``prod_dir="muon"`` (physical) uses
    # the muon's own direction there -- the parent shower axis -- so the bend and
    # the Michel opening angle feed back into how much atmosphere the primary had
    # crossed.  ``prod_dir="ray"`` freezes that direction to the arrival ray,
    # which is what the delivered engine assumes when it evaluates the muon
    # channel's production on the *unshifted* cone direction
    # (joint_cone.delivered_joint_factor); use it for a like-for-like offset
    # distribution, and "muon" to expose the correlation the engine drops.
    dir_prod = u_prod if prod_dir == "muon" else np.broadcast_to(-n_gc,
                                                                u_prod.shape)
    cos_psi_p = -np.sum(dir_prod * r_prod, axis=1) / np.linalg.norm(r_prod, axis=1)
    x_prod = xslant(h_prod * 100.0, np.arccos(np.clip(cos_psi_p, -1, 1)), geom)
    S = rho(np.clip(h_prod, 0.0, H_TOP_M) * 100.0) * source(x_prod, e_prod,
                                                            charge)

    g_p = e_prod / M_MU
    b_p = np.sqrt(np.maximum(1.0 - 1.0 / g_p**2, 0.0))
    common = (np.where(alive & ok, 1.0, 0.0) * S * np.exp(lnJ)
              * (g_p * b_p) / np.maximum(g_d * b_d, 1e-12)
              * jac_dOmega / np.maximum(q_L * q_E, 1e-300))
    n_prim = to_enu(n_prim_gc, basis)
    n_prim /= np.linalg.norm(n_prim, axis=1)[:, None]
    zp, ap = enu_angles(n_prim)
    z0, a0 = enu_angles(n_enu)
    bend = np.degrees(np.arccos(np.clip(n_prim @ n_enu, -1, 1)))
    daz = (ap - a0 + 180.0) % 360.0 - 180.0
    return dict(
        w_numu=common * w_mich["numu"], w_nue=common * w_mich["nue"],
        bend_deg=bend, psi_deg=np.degrees(psi), dzen=zp - z0,
        daz=daz * np.sin(np.radians(z0)),   # great-circle azimuth offset
        daz_raw=daz, n_prim=n_prim, T_p=st["T"], e_dec=e_dec, e_prod=e_prod,
        ratio=np.sqrt(np.maximum(e_prod**2 - M_MU**2, 0)) /
        np.maximum(np.sqrt(np.maximum(e_dec**2 - M_MU**2, 1e-12)), 1e-12),
        L_km=L / 1e3, h_dec_km=(np.linalg.norm(r_dec, axis=1) - RE_M) / 1e3,
        h_prod_km=h_prod / 1e3, x_prod=x_prod,
        theta_lab_deg=np.degrees(np.arccos(cos_lab)), alive=alive & ok,
        exhausted=exhausted,
        n_arr=n_enu, zen_deg=zen_deg, az_deg=az_deg, e_nu=e_nu, charge=charge)


def _ray_length_to_alt(zen_deg, h_m):
    c = np.cos(np.deg2rad(zen_deg))
    return -RE_M * c + np.sqrt((RE_M * c) ** 2 + 2 * RE_M * h_m + h_m**2)


def _cone_dirs(rng, axis, theta):
    """Unit vectors at polar angle ``theta`` (N,) about ``axis`` (3,), random phi."""
    axis = np.asarray(axis, float)
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(axis @ tmp) > 0.9:
        tmp = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(axis, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    ph = rng.uniform(0, 2 * np.pi, len(theta))
    st, ct = np.sin(theta), np.cos(theta)
    return (ct[:, None] * axis[None, :]
            + st[:, None] * (np.cos(ph)[:, None] * e1[None, :]
                             + np.sin(ph)[:, None] * e2[None, :]))


def _backward(r0, u0, e0, T_star, charge, *, field, rho, const_field,
              energy_loss, ref, dt_tau, ds_max_m, max_steps):
    """Integrate the muon *backwards* from decay for proper time ``T_star``.

    Returns the production-point state plus the accumulated rotation applied to
    the auxiliary vector ``ref`` (used to carry the arrival direction, so the
    bending offset is isolated from the Michel opening angle) and the
    energy-loss phase-space Jacobian ``lnJ``.

    Forward, the velocity rotates about ``B_hat`` by ``-q B T_p / m``; backwards
    it therefore rotates by ``+q B T_p / m``.  Both are exact and *independent of
    the muon energy* -- see the module docstring.
    """
    n = len(e0)
    r = np.array(r0, float)
    u = np.array(u0, float)
    e = np.array(e0, float)
    ref = np.array(ref, float)
    T = np.zeros(n)
    psi = np.zeros(n)
    lnJ = np.zeros(n)
    alive = np.ones(n, bool)
    coeff = Q_E / M_MU_KG                              # rad / (T s)
    for _ in range(max_steps):
        act = alive & (T < T_star - 1e-16)
        if not np.any(act):
            break
        idx = np.flatnonzero(act)
        ei = e[idx]
        gi = ei / M_MU
        bi = np.sqrt(np.maximum(1.0 - 1.0 / gi**2, 1e-12))
        # step in proper time, capped so the spatial step stays small
        dT = np.minimum(TAU_MU * dt_tau, ds_max_m / (bi * gi * C_M_S))
        dT = np.minimum(dT, T_star[idx] - T[idx])
        ds = bi * gi * C_M_S * dT                      # m
        ri = r[idx]
        if const_field is not None:
            B = np.broadcast_to(np.asarray(const_field, float), (len(idx), 3))
        else:
            B = field(ri)
        bmag = np.linalg.norm(B, axis=1)
        khat = B / np.maximum(bmag, 1e-30)[:, None]
        dpsi = charge * coeff * bmag * dT              # backward rotation [rad]
        u[idx] = rodrigues(u[idx], khat, dpsi)
        ref[idx] = rodrigues(ref[idx], khat, dpsi)
        psi[idx] += dpsi
        # move backwards along the (rotated) direction, midpoint in position
        r[idx] = ri - u[idx] * ds[:, None]
        hi = np.linalg.norm(r[idx], axis=1) - RE_M
        if energy_loss:
            rr = rho(np.clip(0.5 * (hi + np.linalg.norm(ri, axis=1) - RE_M),
                             0.0, H_TOP_M) * 100.0)
            de = rr * dedx_air(ei) * ds * 100.0        # ds in cm
            e[idx] = ei + de                           # backwards: gain
            lnJ[idx] += _dbde(ei, rr) * ds * 100.0
        T[idx] += dT
        dead = (hi > H_TOP_M) | (hi < -1.0) | (e[idx] > 5e3)
        if np.any(dead):
            alive[idx[dead]] = False
    ref /= np.linalg.norm(ref, axis=1)[:, None]
    u /= np.linalg.norm(u, axis=1)[:, None]
    # samples whose backward walk ran out of steps before reaching T* would have
    # a truncated rotation; flag them so the caller can check they are a
    # negligible weight fraction (they are the very high-gamma tail, whose
    # production point is above the atmosphere anyway).
    exhausted = alive & (T < T_star - 1e-15)
    return dict(r=r, u=u, e=e, ref=ref, T=T, psi=psi, lnJ=lnJ, alive=alive,
                exhausted=exhausted)


# ---------------------------------------------------------------------------
# forward companion: decay-in-flight vs ranging out
# ---------------------------------------------------------------------------
def forward_stats(zen_deg, az_deg, e_bins=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
                  n=6000, seed=1, *, field=None, rho=None, geom=None,
                  source=None, lat=LAT, lon=LON, charge=+1,
                  dt_tau=1.0 / 40.0, ds_max_m=1000.0, max_steps=3000):
    """Fraction of produced muons that decay in flight vs reach the ground /
    range out, per production-energy bin, for muons born on the arrival ray."""
    rng = np.random.default_rng(seed)
    if rho is None or geom is None:
        rho, geom = load_atmosphere()
    if source is None:
        source = source_interp(muon_source_table())
    if field is None:
        field = FieldGrid(lat, lon)
    basis = enu_basis(lat, lon)
    r_det = np.asarray(basis[2]) * RE_M
    n_gc = to_geocentric(arrival_enu(zen_deg, az_deg), basis)[0]
    L_top = _ray_length_to_alt(zen_deg, H_TOP_M)
    out = []
    for elo, ehi in zip(e_bins[:-1], e_bins[1:]):
        Lg = np.linspace(1e3, L_top, 400)
        P = r_det[None, :] + Lg[:, None] * n_gc[None, :]
        rr = np.linalg.norm(P, axis=1)
        psi_g = np.arccos(np.clip((P @ n_gc) / rr, -1, 1))
        xg = xslant((rr - RE_M) * 100.0, psi_g, geom)
        pil = rho(np.clip(rr - RE_M, 0.0, H_TOP_M) * 100.0) * source(
            xg, np.sqrt(elo * ehi), charge) + 1e-300
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pil[1:] + pil[:-1])
                                               * np.diff(Lg))])
        L = np.interp(rng.random(n) * cdf[-1], cdf, Lg)
        r0 = r_det[None, :] + L[:, None] * n_gc[None, :]
        e0 = np.exp(rng.uniform(np.log(elo), np.log(ehi), n))
        T_star = rng.exponential(TAU_MU, n)
        res = _forward(r0, np.broadcast_to(-n_gc, (n, 3)).copy(), e0, T_star,
                       charge, field=field, rho=rho, dt_tau=dt_tau,
                       ds_max_m=ds_max_m, max_steps=max_steps)
        out.append(dict(e_lo=elo, e_hi=ehi, **res))
    return out


def _forward(r0, u0, e0, T_star, charge, *, field, rho, dt_tau, ds_max_m,
             max_steps):
    n = len(e0)
    r, u, e = np.array(r0, float), np.array(u0, float), np.array(e0, float)
    T = np.zeros(n)
    status = np.zeros(n, int)   # 0 running, 1 decay in flight, 2 ground, 3 stop
    psi = np.zeros(n)
    coeff = Q_E / M_MU_KG
    for _ in range(max_steps):
        act = status == 0
        if not np.any(act):
            break
        idx = np.flatnonzero(act)
        ei = e[idx]
        gi = ei / M_MU
        bi = np.sqrt(np.maximum(1.0 - 1.0 / gi**2, 1e-12))
        dT = np.minimum(TAU_MU * dt_tau, ds_max_m / (bi * gi * C_M_S))
        dT = np.minimum(dT, T_star[idx] - T[idx])
        ds = bi * gi * C_M_S * dT
        B = field(r[idx])
        bmag = np.linalg.norm(B, axis=1)
        u[idx] = rodrigues(u[idx], B / np.maximum(bmag, 1e-30)[:, None],
                           -charge * coeff * bmag * dT)
        psi[idx] += charge * coeff * bmag * dT
        r_new = r[idx] + u[idx] * ds[:, None]
        h_mid = 0.5 * (np.linalg.norm(r[idx], axis=1)
                       + np.linalg.norm(r_new, axis=1)) - RE_M
        e[idx] = ei - rho(np.clip(h_mid, 0.0, H_TOP_M) * 100.0) \
            * dedx_air(ei) * ds * 100.0
        r[idx] = r_new
        T[idx] += dT
        h = np.linalg.norm(r[idx], axis=1) - RE_M
        status[idx[h < 0.0]] = 2
        status[idx[(e[idx] < M_MU * 1.02) & (status[idx] == 0)]] = 3
        status[idx[(T[idx] >= T_star[idx] - 1e-18) & (status[idx] == 0)]] = 1
    tot = len(status)
    return dict(f_decay=float((status == 1).mean()),
                f_ground=float((status == 2).mean()),
                f_stop=float((status == 3).mean()),
                f_running=float((status == 0).mean()), n=tot,
                mean_psi_deg=float(np.degrees(psi[status == 1]).mean())
                if np.any(status == 1) else float("nan"))


# ---------------------------------------------------------------------------
# weighted statistics
# ---------------------------------------------------------------------------
def wstats(x, w):
    """``dict(mean, rms, p16, p50, p84, neff)`` of ``x`` with weights ``w``."""
    x = np.asarray(x, float)
    w = np.clip(np.asarray(w, float), 0.0, None)
    s = w.sum()
    if s <= 0:
        return dict(mean=np.nan, rms=np.nan, p16=np.nan, p50=np.nan,
                    p84=np.nan, neff=0.0)
    m = float((w * x).sum() / s)
    v = float((w * (x - m) ** 2).sum() / s)
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / s
    q = np.interp([0.16, 0.5, 0.84], cw, x[o])
    return dict(mean=m, rms=float(np.sqrt(max(v, 0.0))), p16=float(q[0]),
                p50=float(q[1]), p84=float(q[2]),
                neff=float(s * s / max((w * w).sum(), 1e-300)))


# ---------------------------------------------------------------------------
# sanity: handedness / sign against the delivered first-order deflection
# ---------------------------------------------------------------------------
def check_handedness(zen_deg=87.0, az_deg=90.0, charge=+1, lat=LAT, lon=LON,
                     date=DATE, tol=0.05):
    """Assert the exact rotation agrees with ``muon_bending.bending_deflection``.

    ``cone_geff`` builds ``primary = n + bending_deflection(v, B, +1)`` for the
    ``mu+`` daughters; here the primary direction is ``R(B_hat, +q B tau/m) n``.
    Returns the relative discrepancy (must be O(Delta^2)).
    """
    from muon_bending import bending_deflection, local_field_enu

    b_enu = local_field_enu(lat, lon, date)
    bmag = float(np.linalg.norm(b_enu))
    khat = b_enu / bmag
    delta = Q_E * (bmag * 1e-4) * TAU_MU / M_MU_KG
    n_enu = arrival_enu(zen_deg, az_deg)
    exact = rodrigues(n_enu[None, :], khat, np.array([charge * delta]))[0] - n_enu
    first = bending_deflection(-n_enu, b_enu, charge=charge)   # v = -n (travel)
    err = float(np.linalg.norm(exact - first) / max(np.linalg.norm(first), 1e-30))
    assert err < tol, f"handedness/sign mismatch {err:.3f}: {exact} vs {first}"
    return err, np.degrees(delta), b_enu


# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--demo", action="store_true")
    p.add_argument("-n", type=int, default=20000)
    a = p.parse_args(argv)
    err, dd, b = check_handedness()
    print(f"handedness check: rel. err {err:.4f}; Delta = {dd:.2f} deg; "
          f"B_ENU = {np.round(b, 4)} G  (|B| = {np.linalg.norm(b):.4f} G)")
    if not a.demo:
        return
    rho, geom = load_atmosphere()
    src = source_interp(muon_source_table())
    fld = FieldGrid()
    for zen in (87.0, 75.0):
        for az, nm in ((81.88, "E"), (261.88, "W")):
            for q in (+1, -1):
                r = segment_mc(zen, az, 0.5, q, n=a.n, seed=3, field=fld,
                               rho=rho, geom=geom, source=src)
                s = wstats(r["bend_deg"], r["w_numu"])
                print(f"  zen {zen:4.0f} {nm} mu{'+' if q > 0 else '-'} "
                      f"E_nu=0.5: bend {s['mean']:5.2f} +- {s['rms']:4.2f} deg "
                      f"(p16/50/84 {s['p16']:.2f}/{s['p50']:.2f}/{s['p84']:.2f}) "
                      f"neff {s['neff']:.0f}")


if __name__ == "__main__":
    main()
