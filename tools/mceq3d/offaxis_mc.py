"""First-principles off-axis 3D production excess E_off(E, cosZ).

This derives the near-horizon 3D enhancement *from first principles* -- with no
anchoring to any reference flux. It is the complete genuine-3D/1D production
factor: it both redistributes flux in zenith and produces the sub-GeV horizontal
excess, and is the single 3D-production term used by ``mceq3d_flux`` (``offaxis=True``).

E_off is **flavour-independent** (the excess is geometric, inherited from meson
production and common to all pi/K/muon decay chains): one table is applied to each
flavour's own 1D base. ``--validate`` confirms this reproduces *both* the Honda and
Bartol nu_mu and nu_e zenith shapes to ~5% at 0.3 GeV and above ~3 GeV, but
OVERSHOOTS both independent references by ~6-14% at 0.5-1 GeV (beyond their
~4-6% mutual spread) -- a genuine mid-sub-GeV overshoot, see offaxis_mc
--validate and diag_eoff_conservation. The much larger nu_e flux
horizon-enhancement is entirely in the more horizon-peaked nu_e base.

Mechanism (Lipari 2000; Honda et al.). A neutrino arriving at the detector from
zenith ``theta`` was produced at a point ``P`` at altitude ``h`` along the arrival
ray. Its parent shower axis (the primary direction) need not be the arrival
direction: it lies within the production cone of half-width ``sigma_theta(E)``
(the NA61-validated kernel used by ``angular_factor``). In a *curved* atmosphere
the local zenith ``psi`` of a direction steepens with altitude along a
near-horizontal ray, and the slant depth ``X_slant(h, psi)`` -- hence the shower
age -- depends strongly on ``psi`` near the horizon. Averaging the *local*
neutrino production per slant depth ``p(X, E) = dPhi_nu/dX`` (taken from an MCEq
cascade) over the production cone therefore differs from the 1D value that assumes
the primary is collinear with the arrival direction:

    E_off(theta, E) =  int dl rho(h) < p(X_slant(h, psi_p), E) >_cone
                       ---------------------------------------------
                         int dl rho(h)   p(X_slant(h, psi_o), E)

with ``psi_o`` the arrival-direction local zenith at ``P`` and ``psi_p`` the
primary local zenith for a cone offset ``(alpha, beta)``:

    cos psi_p = cos alpha cos psi_o + sin alpha sin psi_o cos beta   (law of cosines)

Near the horizon ``psi_o -> 90 deg``: the cone reaches primaries that came in more
vertically (smaller ``X_slant``, younger shower, more sub-GeV production), while
the opposite half points below the local horizon (huge ``X_slant`` -> ``p -> 0``).
The asymmetry produces a *net* excess a flux-conserving redistribution cannot;
E_off -> 1 as ``sigma_pi -> 0`` (high E) and at the vertical, by construction.

Flavour-independent, pion-driven (important). The near-horizon excess is a
production-rate effect: it is set by *where the parent pions are produced* off-axis
in the curved atmosphere, and every daughter neutrino inherits it regardless of
the decay chain (direct ``pi->mu nu`` or ``pi->mu->e nu nu``). So E_off is the
**same for nu_mu and nu_e** -- one table applied to each flavour's own 1D base
(the large nu_e-vs-nu_mu flux difference is entirely in the more horizon-peaked,
muon-decay-origin nu_e base). The cone is therefore driven by the **pion
production angle** ``sigma_pi`` (kaons subdominant); the neutrino's own decay
opening angle does *not* enter -- it relocates individual neutrinos but not the
pion-production excess, which is why near-isotropic muon-decay neutrinos still
carry the full excess (they must NOT be treated as a flat pedestal).

Every ingredient is first-principles/data-driven, with no reference flux anywhere:
  * p(X, E)          -- MCEq depth-resolved cascade (SIBYLL-2.3d, H3a),
  * X_slant(h, psi)  -- curved-atmosphere slant depth in MCEq's CORSIKA density,
  * sigma_pi(E)      -- the pion production angle from the SIBYLL/UrQMD generator
                        moments (chromo, NA61-validated) folded with exact pi->mu
                        nu decay and the H3a spectrum (no hand-set p_T / x_F).

This reproduces the Honda and Bartol nu_mu and nu_e zenith shapes to within their
mutual ~5-15% spread across 0.3-10 GeV (``--validate``), with no anchoring. The
data-driven (NA61 +-12% pion-angle) uncertainty on the excess is +-8% at the
0.3 GeV horizon, falling to ~0 by a few GeV. A multi-generator hadronic spread
needs alternative generators (EPOS/QGSJET; chromo downloads / cluster).

CAVEAT UNDER TEST (2026-09-04): flavour-independence is an assumption, not a
result. Eq. (8) averages the production over the distribution of the **primary**
direction *given the neutrino direction*. For the direct ``pi -> mu nu`` channel
that distribution is the pion production angle folded with the two-body decay --
the cone used above. For a **muon-decay** neutrino (all nu_e, ~half of the
sub-GeV nu_mu) the primary direction is displaced by pion production **plus**
``pi -> mu`` **plus** the Michel decay **plus** the in-flight muon bend, i.e. a
25-50%% wider cone (`kinematic_kernel.mudecay_shape_mc`). The statement in the
docstrings below -- "the neutrino's own decay opening angle does not enter" --
is right for *relocating* neutrinos of a fixed parent, but it is not what Eq. (8)
conditions on. Because the near-horizon excess is strongly non-linear in the cone
width, a wider muon-decay cone makes E_off **species-dependent** (nu_e > nu_mu).
`build_channel` builds that variant (per-species tables, blended at the level of
the cone-averaged numerators) alongside the delivered flavour-independent one, so
the two can be compared; the delivered ``offaxis_excess.npz`` is unchanged.

Build the table (writes ``offaxis_excess.npz`` with keys e, cz, E_off)::

    python offaxis_mc.py --build

Build the per-species / corrected-moment scan (four new files, delivered table
untouched)::

    python offaxis_mc.py --build-channel --n-jobs 10

Validate the derived shape against Honda/Bartol (independent -- not used here)::

    python offaxis_mc.py --validate
"""

from __future__ import annotations

import argparse

import numpy as np

R_EARTH_CM = 6371.0e5  # mean Earth radius [cm]
H_TOP_CM = 112.8e5  # top of the CORSIKA atmosphere [cm]


# --------------------------------------------------------------------------
# First-principles ingredients
# --------------------------------------------------------------------------
TAG = "SIBYLL23D_HillasGaisser2012-H3a"  # hadronic/primary identity of the tables


def production_profile(e_lo=0.1, e_hi=100.0, n_x=80, theta_deg=0.0,
                       rc_cut_gv=None, species=None):
    """p(X, E) = dPhi_numu/dX from a depth-resolved MCEq cascade.

    Returns (x_grid [g/cm2], e_grid [GeV], p[n_x, nE], density_model).
    p is the local numu production per unit *slant* depth; a function of X, it is
    ~zenith-independent (the cascade lives in X; the residual rho(X)-dependent
    decay competition is checked by `verify_offaxis.py` via ``theta_deg``).
    ``rc_cut_gv`` optionally applies a rigidity cutoff to the primaries (A/Z=1
    upper bound) to check the site-independence of the E_off ratio.
    """
    import importlib.util  # noqa: F401  (MCEq config touches importlib.util)
    import mceq_config as cfg

    cfg.e_min = e_lo
    from MCEq.core import MCEqRun
    import crflux.models as crf

    mc = MCEqRun(
        interaction_model="SIBYLL23D",
        primary_model=(crf.HillasGaisser2012, "H3a"),
        theta_deg=theta_deg,
    )
    if rc_cut_gv is not None:
        from mceq3d_flux import _transmission

        t = _transmission(mc.e_grid, rc_cut_gv, az_over_z=1.0)
        for pid in ((2212, 0), (2112, 0)):
            part = mc.pman[pid]
            mc._phi0[part.lidx: part.uidx] *= t
    xmax = float(mc.density_model.max_X)
    x_grid = np.linspace(xmax / n_x, xmax, n_x)
    mc.solve(int_grid=x_grid, grid_var="X")
    e = mc.e_grid
    sel = (e >= e_lo) & (e <= e_hi)

    def prof(key):
        acc = np.array(
            [mc.get_solution(key, 0, grid_idx=i)[sel] for i in range(n_x)]
        )  # (n_x, nE) accumulated flux at each depth
        return np.clip(np.gradient(acc, x_grid, axis=0), 0.0, None)

    p = prof("total_numu")
    # per-parent split: the kaon-parent part gets its own (wider) cone kernel;
    # everything else (direct pi + muon-decay + K0 etc.) carries the pion
    # production geometry (inheritance -- see offaxis_excess).
    p_k = np.minimum(prof("k_numu"), p)
    out = {"tot": p, "k": p_k}
    # Optional per-species, per-PARENT-CHANNEL depth-resolved production. Needed
    # by the channel-weighted cone (`offaxis_excess_channel`): the muon-decay
    # component of each species is produced with a *wider* cone than the direct
    # pi -> mu nu component, and its share f_mu varies with depth (hence with
    # zenith, since the horizon ray reaches much larger X) and with species
    # (~0.4-0.5 for nu_mu, ~0.99 for nu_e).
    for s in species or ():
        tot_s = prof(f"total_{s}")
        k_s = np.minimum(prof(f"k_{s}"), tot_s)
        mu_s = np.minimum(prof(f"mu_{s}"), tot_s)
        out[f"{s}_tot"] = tot_s
        out[f"{s}_k"] = k_s
        out[f"{s}_mu"] = mu_s
        # everything that is neither kaon-parent nor muon-decay: the direct
        # pi -> mu nu (+ K0 etc.) component, which carries the pion cone.
        out[f"{s}_dir"] = np.clip(tot_s - k_s - mu_s, 0.0, None)
    return x_grid, e[sel], out, mc.density_model


def _rho_of_h(density_model):
    """Vectorised air density rho(h) [g/cm3] from MCEq's CORSIKA atmosphere."""
    h_tab = np.linspace(0.0, H_TOP_CM, 4000)
    x_tab = density_model.h2X(h_tab)  # vertical depth [g/cm2]
    inv_rho = density_model.r_X2rho(np.clip(x_tab, 1e-6, None))  # 1/rho [cm3/g]
    rho_tab = np.where(x_tab > 1e-4, 1.0 / inv_rho, 0.0)

    def rho(h_cm):
        return np.interp(h_cm, h_tab, rho_tab, left=rho_tab[0], right=0.0)

    return rho


def slant_depth_table(rho, n_h=90, n_psi=140, n_step=1400):
    """Precompute X_slant(h, psi): slant column from P(alt=h) to the top of the
    atmosphere along a straight line at local zenith ``psi``. Rays that strike the
    solid Earth (perigee below the surface) get X=inf (primary blocked)."""
    h_grid = np.concatenate(
        [np.linspace(0.0, 40e5, n_h - 25), np.linspace(41e5, H_TOP_CM, 25)]
    )
    psi_grid = np.linspace(0.0, np.deg2rad(179.0), n_psi)
    table = np.zeros((len(h_grid), n_psi))
    for i, h in enumerate(h_grid):
        r0 = R_EARTH_CM + h
        for j, psi in enumerate(psi_grid):
            cospsi = np.cos(psi)
            # geometric path length to the top of atmosphere (outer sphere)
            r_top = R_EARTH_CM + H_TOP_CM
            # solve r0^2 + s^2 + 2 r0 s cospsi = r_top^2 for s>0
            disc = (r0 * cospsi) ** 2 - (r0**2 - r_top**2)
            s_top = -r0 * cospsi + np.sqrt(max(disc, 0.0))
            s = np.linspace(0.0, s_top, n_step)
            r = np.sqrt(r0**2 + s**2 + 2 * r0 * s * cospsi)
            alt = r - R_EARTH_CM
            if np.any(alt < -1.0):  # ray dips into the solid Earth -> blocked
                table[i, j] = np.inf
                continue
            table[i, j] = np.trapezoid(rho(alt), s)
    return h_grid, psi_grid, table


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def _interp_xslant(h, psi, h_grid, psi_grid, table):
    """Bilinear lookup of X_slant(h, psi) (psi folded to [0, pi])."""
    psi = np.abs(psi)
    ih = np.clip(np.searchsorted(h_grid, h) - 1, 0, len(h_grid) - 2)
    ip = np.clip(np.searchsorted(psi_grid, psi) - 1, 0, len(psi_grid) - 2)
    th = (h - h_grid[ih]) / (h_grid[ih + 1] - h_grid[ih])
    tp = (psi - psi_grid[ip]) / (psi_grid[ip + 1] - psi_grid[ip])
    # inf marks rays blocked by the solid Earth; cap so bilinear stays finite and
    # _p_at returns 0 past the tabulated depth (production -> 0 there).
    cap = 1e7
    tbl = np.minimum(table, cap)
    c00, c01 = tbl[ih, ip], tbl[ih, ip + 1]
    c10, c11 = tbl[ih + 1, ip], tbl[ih + 1, ip + 1]
    return (
        c00 * (1 - th) * (1 - tp)
        + c01 * (1 - th) * tp
        + c10 * th * (1 - tp)
        + c11 * th * tp
    )


def _p_at(x_vals, x_grid, e_grid, p):
    """p(X, E) for a 1-D array of X values -> (len(x_vals), nE), linear in X,
    with p=0 beyond the tabulated depth (blocked / over-developed)."""
    x_vals = np.asarray(x_vals)
    out = np.zeros((x_vals.size, len(e_grid)))
    finite = np.isfinite(x_vals) & (x_vals <= x_grid[-1])
    if np.any(finite):
        xv = x_vals[finite]
        idx = np.clip(np.searchsorted(x_grid, xv) - 1, 0, len(x_grid) - 2)
        t = (xv - x_grid[idx]) / (x_grid[idx + 1] - x_grid[idx])
        out[finite] = (1 - t)[:, None] * p[idx] + t[:, None] * p[idx + 1]
    return out


# --------------------------------------------------------------------------
# Off-axis excess
# --------------------------------------------------------------------------
def cone_numden(
    cos_theta, e_grid, sigma_deg, x_grid, ep_grid, p, geom, n_alpha=60, n_beta=24,
    alpha_w=None,
):
    """(numerator, denominator) of E_off(E) for one arrival cos(zenith).

    The production cone is either Gaussian-on-the-sphere with RMS
    ``sigma_deg(E)``, or -- if ``alpha_w`` (nE, n_alpha) is given -- the
    **sampled angular distribution** on the internal alpha grid
    (linspace 0.5..89 deg, n_alpha points; weights include the measure), e.g.
    from the generator (x_L, theta) kernel histograms (`kinematic_kernel.
    pion_alpha_pdf`) instead of the second-moment Gaussian."""
    h_grid, psi_grid, table = geom
    theta = np.arccos(np.clip(cos_theta, -1, 1))

    # arrival ray sampled by altitude (monotonic for down/up-going rays)
    h_ray = np.linspace(0.0, 80e5, 260)
    r = R_EARTH_CM + h_ray
    # distance along ray from detector to altitude h (law of cosines, root >=0)
    disc = (R_EARTH_CM * np.cos(theta)) ** 2 + (r**2 - R_EARTH_CM**2)
    ell = -R_EARTH_CM * np.cos(theta) + np.sqrt(np.clip(disc, 0, None))
    cos_psi_o = np.clip((ell + R_EARTH_CM * np.cos(theta)) / r, -1, 1)
    psi_o = np.arccos(cos_psi_o)
    sin_psi_o = np.sin(psi_o)

    # air-density weight along the ray (production per length ~ rho(h));
    # rho(h) dl is common to numerator and denominator.
    rho_w = _RHO(h_ray)
    # path measure dl between altitude samples
    dl = np.gradient(ell)

    num = np.zeros(len(e_grid))
    den = np.zeros(len(e_grid))

    # denominator: primary collinear with arrival direction (1D)
    x_o = np.array(
        [_interp_xslant(h, ps, h_grid, psi_grid, table) for h, ps in zip(h_ray, psi_o)]
    )
    p_o = _p_at(x_o, x_grid, ep_grid, p)  # (n_ray, nE) at ep_grid
    p_o = _regrid(p_o, ep_grid, e_grid)
    den = np.sum((rho_w * dl)[:, None] * p_o, axis=0)

    # numerator: average p over the production cone, per energy (sigma depends on E)
    # Integrate over the FULL sphere with the proper 2D measure sin(alpha): the
    # large-angle tail (primaries pointing below the local horizon -> blocked,
    # p->0) stays in the normalisation and correctly dilutes <p>. cos psi_p from
    # the spherical law of cosines about the arrival direction.
    beta = np.linspace(0, 2 * np.pi, n_beta, endpoint=False)
    cos_b = np.cos(beta)
    # NB: offaxis_excess() drives this cone with the collimated *pion production*
    # angle (sigma_pi); the excess is a production-rate effect inherited by all
    # daughters, so it is not the (wide) neutrino angle and applies flavour-
    # independently. See offaxis_excess() for the physics.
    alpha = np.deg2rad(np.linspace(0.5, 89.0, n_alpha))
    for k, sdeg in enumerate(sigma_deg):
        s1 = np.deg2rad(sdeg) / np.sqrt(2.0)  # per-axis sigma
        if alpha_w is not None and alpha_w[k].sum() > 0:
            wa = alpha_w[k] / alpha_w[k].sum()  # sampled kernel distribution
        elif s1 < 1e-4:
            num[k] = den[k]
            continue
        else:
            wa = np.sin(alpha) * np.exp(-(alpha**2) / (2 * s1**2))
            wa /= wa.sum()
        acc = np.zeros(len(h_ray))
        for a, w in zip(alpha, wa):
            # cos psi_p over all beta for this alpha, broadcast over the ray
            cos_pp = (
                np.cos(a) * cos_psi_o[:, None]
                + np.sin(a) * sin_psi_o[:, None] * cos_b[None, :]
            )  # (n_ray, n_beta)
            psi_p = np.arccos(np.clip(cos_pp, -1, 1))
            xv = _interp_xslant(
                np.repeat(h_ray, n_beta),
                psi_p.ravel(),
                h_grid,
                psi_grid,
                table,
            ).reshape(len(h_ray), n_beta)
            pk = _p_at(xv.ravel(), x_grid, ep_grid, p)  # (n_ray*n_beta, nE)
            pk = _regrid(pk, ep_grid, e_grid)[:, k].reshape(len(h_ray), n_beta)
            acc += w * pk.mean(axis=1)
        num[k] = np.sum(rho_w * dl * acc)

    return num, den


# --------------------------------------------------------------------------
# Multi-cone / multi-profile evaluation of the same integral (fast path)
# --------------------------------------------------------------------------
def cone_numden_multi(cos_theta, e_grid, x_grid, profiles, cones, geom,
                      n_alpha=44, n_beta=18, alpha_chunk=6):
    """Numerator/denominator of Eq. (8) for MANY (cone, production-profile) pairs
    at one arrival ``cos_theta``, in one pass.

    ``profiles``: list of ``p[n_x, nE]`` production profiles on ``e_grid``.
    ``cones``:    list of ``(profile_index, sigma_deg[nE] or None,
                  alpha_w[nE, n_alpha] or None)``.
    Returns ``[(num, den), ...]`` aligned with ``cones``.

    Mathematically **identical** to calling :func:`cone_numden` once per entry
    (same alpha/beta grids, same ``sin(alpha) exp(-alpha^2 / 2 s1^2)`` weights with
    the per-axis width ``s1 = sigma / sqrt(2)``, same bilinear ``X_slant`` lookup and
    the same linear ``p(X)`` interpolation). The speed-up is purely
    organisational: ``X_slant(h, psi_p(alpha, beta))`` does not depend on the
    neutrino energy, so it -- and ``p(X)`` -- are evaluated once for the whole
    energy grid and all cones instead of once per energy per cone. That is an
    O(nE) saving (~60x here) and is what makes a per-species 2x2 table scan
    affordable. `test_offaxis_mc.py::test_multi_matches_cone_numden` pins the
    equivalence.
    """
    h_grid, psi_grid, table = geom
    theta = np.arccos(np.clip(cos_theta, -1, 1))
    nE = len(e_grid)

    h_ray = np.linspace(0.0, 80e5, 260)
    r = R_EARTH_CM + h_ray
    disc = (R_EARTH_CM * np.cos(theta)) ** 2 + (r**2 - R_EARTH_CM**2)
    ell = -R_EARTH_CM * np.cos(theta) + np.sqrt(np.clip(disc, 0, None))
    cos_psi_o = np.clip((ell + R_EARTH_CM * np.cos(theta)) / r, -1, 1)
    psi_o = np.arccos(cos_psi_o)
    sin_psi_o = np.sin(psi_o)
    rho_w = _RHO(h_ray)
    dl = np.gradient(ell)
    meas = rho_w * dl                      # (n_ray,)
    n_ray = len(h_ray)

    # ---- denominators: primary collinear with the arrival direction (1D) ----
    x_o = np.array(
        [_interp_xslant(h, ps, h_grid, psi_grid, table) for h, ps in zip(h_ray, psi_o)]
    )
    dens = [meas @ _p_at(x_o, x_grid, e_grid, p) for p in profiles]

    # ---- A[ip][alpha, E] = int dl rho <p(X_slant(h, psi_p))>_beta ----
    beta = np.linspace(0, 2 * np.pi, n_beta, endpoint=False)
    cos_b = np.cos(beta)
    alpha = np.deg2rad(np.linspace(0.5, 89.0, n_alpha))
    A = [np.zeros((n_alpha, nE)) for _ in profiles]
    for a0 in range(0, n_alpha, alpha_chunk):
        a1 = min(a0 + alpha_chunk, n_alpha)
        aa = alpha[a0:a1]
        cos_pp = (
            np.cos(aa)[:, None, None] * cos_psi_o[None, :, None]
            + np.sin(aa)[:, None, None] * sin_psi_o[None, :, None]
            * cos_b[None, None, :]
        )                                   # (na, n_ray, n_beta)
        psi_p = np.arccos(np.clip(cos_pp, -1, 1))
        xv = _interp_xslant(
            np.repeat(h_ray, n_beta)[None, :].repeat(a1 - a0, 0).ravel(),
            psi_p.ravel(), h_grid, psi_grid, table,
        )
        for ip, p in enumerate(profiles):
            pk = _p_at(xv, x_grid, e_grid, p).reshape(a1 - a0, n_ray, n_beta, nE)
            A[ip][a0:a1] = np.einsum("r,arbe->ae", meas, pk) / n_beta

    # ---- fold each cone's alpha weights ----
    out = []
    for ip, sigma_deg, alpha_w in cones:
        num = np.empty(nE)
        den = dens[ip]
        for k in range(nE):
            if alpha_w is not None and alpha_w[k].sum() > 0:
                wa = alpha_w[k] / alpha_w[k].sum()
            else:
                s1 = np.deg2rad(sigma_deg[k]) / np.sqrt(2.0)
                if s1 < 1e-4:
                    num[k] = den[k]
                    continue
                wa = np.sin(alpha) * np.exp(-(alpha**2) / (2 * s1**2))
                wa /= wa.sum()
            num[k] = wa @ A[ip][:, k]
        out.append((num, den))
    return out


def e_off_for_zenith(
    cos_theta, e_grid, sigma_deg, x_grid, ep_grid, p, geom, n_alpha=60, n_beta=24,
    alpha_w=None,
):
    """E_off(E) = num/den for one arrival cos(zenith) (see `cone_numden`)."""
    num, den = cone_numden(
        cos_theta, e_grid, sigma_deg, x_grid, ep_grid, p, geom, n_alpha, n_beta,
        alpha_w=alpha_w,
    )
    return np.where(den > 0, num / np.maximum(den, 1e-300), 1.0)


# module-level handles filled by build()
_RHO = None


def _regrid(arr, e_src, e_dst):
    """Log-log regrid columns of arr (., nE_src) onto e_dst."""
    if len(e_src) == len(e_dst) and np.allclose(e_src, e_dst):
        return arr
    out = np.empty((arr.shape[0], len(e_dst)))
    le_s, le_d = np.log(e_src), np.log(e_dst)
    for i in range(arr.shape[0]):
        out[i] = np.interp(le_d, le_s, arr[i])
    return out


def offaxis_excess(cz, x_grid, ep_grid, p, geom, n_alpha=44, n_beta=18, moments=None,
                   sigma_scale=1.0, cone_kernel="moments"):
    """Off-axis 3D-production excess E_off(cosZ, E) -- **flavour-independent**.

    The near-horizon excess is a geometric property of *where the parent pions are
    produced* in the curved atmosphere (off-axis production reaching shallower,
    more-productive columns near the horizon). It is a production-rate effect, so
    every daughter neutrino inherits it regardless of the decay chain (direct
    pi->mu nu, or pi->mu->e nu nu): E_off is therefore the **same for nu_mu and
    nu_e** -- one table applied to each flavour's own 1D base. (Verified: the
    excess implied by the Honda/Bartol tables, ref-shape/base-shape, agrees between
    nu_mu and nu_e to ~5% at every energy; the large nu_e-vs-nu_mu flux difference
    is entirely in the more horizon-peaked, muon-decay-origin nu_e base.)

    The magnitude is set by the **pion production angle** -- taken from the SIBYLL/
    UrQMD generator moments folded with exact pi->mu nu decay (``channel_shapes``,
    no hand-set p_T/x_F) -- pushed through the curved-atmosphere cone. Pions
    dominate the parent mesons; kaons (wider angle, <~25% of the flux only above a
    few GeV where the excess is small) are a subdominant refinement not folded in.
    The neutrino's *own* decay-opening angle does not enter the cone: it relocates
    individual neutrinos but does not change the pion-production excess (which is
    why the near-isotropic muon-decay neutrinos still carry the full excess, and
    must NOT be treated as a flat pedestal).

    Validated (``--validate``) to reproduce both the nu_mu and nu_e Honda/Bartol
    zenith shapes to within their mutual spread, with no reference-flux input.
    """
    from kinematic_kernel import channel_shapes, pion_alpha_pdf

    kw = {} if moments is None else {"moments": moments}
    alpha_deg = np.linspace(0.5, 89.0, n_alpha)
    # pion channel angular distribution:
    #   "sampled"  -- the full (x_L, theta) generator kernel histograms + exact
    #                 decay (pion_alpha_pdf); the original choice.
    #   "moments"  -- a Gaussian of the NA61-VALIDATED moment variance sigma_pi
    #                 (channel_shapes). Preferred at 0.5-1 GeV: the sampled kernel
    #                 (k_spliced.npz) is ~20-30% WIDER in RMS than the moment
    #                 sigma_pi (m_spliced.npz) there -- an inconsistency between the
    #                 two angular products that grows with energy and inflates the
    #                 E_off horizon excess by ~12-15% (outside the Honda-Bartol
    #                 spread). The moment sigma_pi is the NA61 <p_T>-validated input
    #                 and its Gaussian cone matches BOTH references (diag_cone_shape,
    #                 diag_kernel_stats). Same p95/RMS ~ 1.8 either way (no genuine
    #                 tail lost), so this is a consistency fix, not a Gaussian
    #                 approximation loss.
    W = None if cone_kernel == "moments" else pion_alpha_pdf(
        ep_grid, alpha_deg, scale=sigma_scale)
    sig_pi = channel_shapes(ep_grid, **kw)["pi"] * sigma_scale  # Gaussian fallback
    sig_k = channel_shapes(ep_grid, **kw)["k"] * sigma_scale
    p_nonk = p["tot"] - p["k"]  # direct pi + muon-decay + rest: pion geometry
    num = np.zeros((len(cz), len(ep_grid)))
    den = np.zeros_like(num)
    for i, c in enumerate(cz):
        n1, d1 = cone_numden(c, ep_grid, sig_pi, x_grid, ep_grid, p_nonk, geom,
                             n_alpha, n_beta, alpha_w=W)
        n2, d2 = cone_numden(c, ep_grid, sig_k, x_grid, ep_grid, p["k"], geom,
                             n_alpha, n_beta)
        num[i], den[i] = n1 + n2, d1 + d2
    return np.where(den > 0, num / np.maximum(den, 1e-300), 1.0)


# --------------------------------------------------------------------------
# Channel-weighted (per-species) off-axis excess
# --------------------------------------------------------------------------
CHANNEL_SPECIES = ("numu", "antinumu", "nue", "antinue")

# Corrected generator moments (production angle arcsin(p_T/p), not arctan(p_T/p);
# see kernel_regeneration).  Since 2026-09-04 these ARE ``kinematic_kernel``'s
# defaults, so ``MOMENTS_V2`` is an alias of ``_MOMENTS`` (kept as a name because
# the A/B scan in :func:`build_channel` and several diagnostics refer to it);
# ``MOMENTS_LEGACY`` is the pre-arcsin set, i.e. the "old" axis of that scan.
def _kk_moments():
    from kinematic_kernel import _MOMENTS, _MOMENTS_LEGACY

    return _MOMENTS, _MOMENTS_LEGACY


MOMENTS_V2, MOMENTS_LEGACY = _kk_moments()

# module-level context for the multiprocessing workers (avoids re-pickling the
# slant-depth table and the production profiles for every zenith)
_CTX = {}


def _channel_init(ctx):
    """Worker init. ``_RHO`` is inherited through fork (it is a closure over the
    MCEq density model, which does not pickle), so only the arrays travel here."""
    global _CTX
    _CTX = ctx


def _channel_worker(cz_value):
    """One arrival zenith: all (profile, cone) pairs of the 2x2 scan in one pass."""
    return cone_numden_multi(
        cz_value, _CTX["e"], _CTX["x"], _CTX["profiles"], _CTX["cones"],
        _CTX["geom"], _CTX["n_alpha"], _CTX["n_beta"],
    )


def channel_cone_widths(ep_grid, moments=None, bending=False, n_mc=4_000_000):
    """Space-angle RMS [deg] of the three production cones on ``ep_grid``.

    * ``pi``  -- direct ``pi -> mu nu`` (the delivered cone, `channel_shapes`),
    * ``k``   -- direct ``K -> mu nu``,
    * ``mu_numu`` / ``mu_nue`` -- the **muon-decay** channel, i.e. the primary
      direction given a ``pi -> mu -> e nu nu`` neutrino: pion production angle
      (+) ``pi -> mu`` decay (+) Michel decay (+) optionally in-flight bending,
      added in quadrature (`kinematic_kernel.mudecay_shape_mc`).

    The muon-decay cone is 25-50% wider than the direct one. Eq. (8) averages the
    production over the distribution of the PRIMARY direction *given the neutrino
    direction*, so the correct cone for the muon-decay component is this wider
    one -- the neutrino's own decay-opening angle DOES enter for that component
    (contrary to the flavour-independent argument in `offaxis_excess`, which is
    right only for the direct channel).
    """
    from kinematic_kernel import channel_shapes, mudecay_shape_mc

    kw = {} if moments is None else {"moments": moments}
    sh = channel_shapes(ep_grid, **kw)
    out = {"pi": sh["pi"], "k": sh["k"]}
    for fl in ("numu", "nue"):
        out[f"mu_{fl}"] = mudecay_shape_mc(
            ep_grid, species=fl, n=n_mc, bending=bending, **kw
        )
    return out


def _species_flavour(s):
    return "nue" if "nue" in s else "numu"


def build_channel(cz=None, n_alpha=44, n_beta=18, n_jobs=None, bending=False,
                  out_dir=".", verbose=True):
    """Build the **2x2** E_off table scan and write four self-contained files.

    The two axes are

      * generator moments: ``old`` (``m_spliced.npz``, the delivered tables, whose
        production angle used ``arctan(p_T/p)``) vs ``v2`` (``m_spliced_v2.npz``,
        the corrected ``arcsin(p_T/p)``);
      * cone: ``pion-only`` (the delivered assumption -- one pion cone for every
        production channel, hence flavour-independent) vs ``channel`` (the
        muon-decay component gets its own, wider cone).

    Files: ``offaxis_excess_rebuild_old.npz``, ``offaxis_excess_v2.npz``,
    ``offaxis_excess_channel.npz``, ``offaxis_excess_channel_v2.npz``.
    ``offaxis_excess.npz`` is never touched; ``offaxis_excess_rebuild_old.npz``
    is a byte-level regression of the delivered build through the new fast path.

    Combination rule (why a linear blend of the *numerators* is the right one).
    Split the depth-resolved production into its parent channels,
    ``p_s(X, E) = p_s^dir + p_s^K + p_s^mu`` (MCEq's own ``pi_/k_/mu_`` tracked
    categories, so the split is depth-resolved and needs no external fractions).
    Eq. (8) is linear in ``p``, and the denominator does not depend on the cone,
    so

        E_off_s = [ N_dir(sigma_pi) + N_K(sigma_K) + N_mu(sigma_mu) ]
                  / [ D_dir + D_K + D_mu ]
                = (1 - f_K - f_mu) E_off^pi + f_K E_off^K + f_mu,s E_off^mu ,

    with ``f_c = D_c / sum D`` the 1D-production-weighted channel fractions *at
    that zenith and energy*. This is exactly the requested
    ``(1-f_mu) E_off[direct] + f_mu E_off[muon]`` blend, with two improvements:
    the weights come out of the same cascade that supplies ``p`` (no separate
    `channel_fractions` call, no depth- or zenith-independence approximation --
    f_mu genuinely rises toward the horizon because the horizon ray samples much
    larger X), and the kaon channel keeps its own cone. Blending the *numerators*
    (equivalently, the E_off values with production weights) rather than the
    final fluxes is required because E_off is a ratio: a flux-weighted average of
    ratios with mismatched denominators would not be the ratio of the sums.

    The per-axis width convention is unchanged: ``sigma`` is a **space-angle
    RMS** and the cone uses ``s1 = sigma/sqrt(2)`` per axis (`cone_numden`,
    `cone_numden_multi`).
    """
    import multiprocessing as mp
    import os

    global _RHO
    if cz is None:
        cz = np.round(np.arange(0.05, 1.0, 0.1), 2)
    cz = np.asarray(cz, float)

    if verbose:
        print("[1/4] MCEq depth-resolved production p(X,E), per species/channel ...")
    x_grid, ep_grid, p, dm = production_profile(species=CHANNEL_SPECIES)
    _RHO = _rho_of_h(dm)
    if verbose:
        print("[2/4] curved-atmosphere slant-depth table X_slant(h,psi) ...")
    geom = slant_depth_table(_RHO)

    if verbose:
        print("[3/4] cone widths (old + v2 moments) ...")
    from kinematic_kernel import muon_shape

    # NB both axes are named EXPLICITLY: ``moments=None`` now resolves to the v2
    # set (kinematic_kernel._MOMENTS), so the "old" axis must pass
    # ``MOMENTS_LEGACY`` or the 2x2 scan would collapse onto one moment set.
    MOM = {"old": MOMENTS_LEGACY, "v2": MOMENTS_V2}
    widths = {
        m: channel_cone_widths(ep_grid, moments=MOM[m], bending=bending)
        for m in ("old", "v2")
    }
    # bending-on variant of the muon cone (site-dependent -> reported, not shipped)
    widths_bend = {
        m: channel_cone_widths(ep_grid, moments=MOM[m], bending=True)
        for m in ("old", "v2")
    }
    sig_mu_kernel = muon_shape(ep_grid)          # muon-calibration closure kernel

    # ---- profile list -----------------------------------------------------
    profiles = [p["tot"] - p["k"], p["k"], p["tot"]]
    pidx = {}
    for s in CHANNEL_SPECIES:
        pidx[s] = (len(profiles), len(profiles) + 1, len(profiles) + 2)
        profiles += [p[f"{s}_dir"], p[f"{s}_k"], p[f"{s}_mu"]]

    # ---- cone list --------------------------------------------------------
    SCALES = {"": 1.0, "_hi": 1.12, "_lo": 0.88}
    cones, tags = [], []

    def add(tag, ip, sigma):
        tags.append(tag)
        cones.append((ip, np.asarray(sigma, float), None))

    for m in ("old", "v2"):
        for suf, sc in SCALES.items():
            w = widths[m]
            add(f"{m}{suf}|flat|pi", 0, w["pi"] * sc)
            add(f"{m}{suf}|flat|k", 1, w["k"] * sc)
            for s in CHANNEL_SPECIES:
                i_dir, i_k, i_mu = pidx[s]
                fl = _species_flavour(s)
                add(f"{m}{suf}|{s}|dir", i_dir, w["pi"] * sc)
                add(f"{m}{suf}|{s}|k", i_k, w["k"] * sc)
                # delivered assumption: muon-decay component gets the pion cone
                add(f"{m}{suf}|{s}|mu_pi", i_mu, w["pi"] * sc)
                add(f"{m}{suf}|{s}|mu_mu", i_mu, w[f"mu_{fl}"] * sc)
                if suf == "":
                    add(f"{m}|{s}|mu_bend", i_mu,
                        widths_bend[m][f"mu_{fl}"] * sc)
    add("closure|mu", 2, sig_mu_kernel)

    if verbose:
        print(f"[4/4] {len(cones)} cones x {len(cz)} zeniths "
              f"({len(profiles)} production profiles) ...")
    ctx = dict(e=ep_grid, x=x_grid, profiles=profiles, cones=cones, geom=geom,
               n_alpha=n_alpha, n_beta=n_beta)
    n_jobs = n_jobs or min(len(cz), max(1, (os.cpu_count() or 8) // 3))
    if n_jobs > 1:
        with mp.get_context("fork").Pool(n_jobs, initializer=_channel_init,
                                         initargs=(ctx,)) as pool:
            res = pool.map(_channel_worker, list(cz))
    else:
        _channel_init(ctx)
        res = [_channel_worker(c) for c in cz]

    # ---- assemble ---------------------------------------------------------
    ti = {t: i for i, t in enumerate(tags)}
    nE = len(ep_grid)

    def ratio(keys, iz):
        num = np.zeros(nE)
        den = np.zeros(nE)
        for k in keys:
            n_, d_ = res[iz][ti[k]]
            num += n_
            den += d_
        return np.where(den > 0, num / np.maximum(den, 1e-300), 1.0)

    def table(keyfun):
        return np.array([ratio(keyfun(iz), iz) for iz in range(len(cz))])

    out = {}
    for m in ("old", "v2"):
        for mode in ("flat", "channel"):
            d = {"e": ep_grid, "cz": cz, "tag": TAG, "moments": m,
                 "cone_mode": mode, "species": np.array(CHANNEL_SPECIES)}
            for suf in SCALES:
                key = f"E_off{suf}"
                d[key] = table(lambda iz, m=m, suf=suf:
                               [f"{m}{suf}|flat|pi", f"{m}{suf}|flat|k"])
                mu_tag = "mu_pi" if mode == "flat" else "mu_mu"
                d[key + "_s"] = np.array([
                    table(lambda iz, m=m, suf=suf, s=s, mt=mu_tag:
                          [f"{m}{suf}|{s}|dir", f"{m}{suf}|{s}|k",
                           f"{m}{suf}|{s}|{mt}"])
                    for s in CHANNEL_SPECIES
                ])
            # bending-on variant of the channel cone (reported, not delivered)
            if mode == "channel":
                d["E_off_s_bend"] = np.array([
                    table(lambda iz, m=m, s=s:
                          [f"{m}|{s}|dir", f"{m}|{s}|k", f"{m}|{s}|mu_bend"])
                    for s in CHANNEL_SPECIES
                ])
            # muon-calibration closure (horizon, vertical rows)
            iz_h = int(np.argmin(cz))
            iz_v = int(np.argmax(cz))
            d["E_off_mu"] = np.array([ratio(["closure|mu"], iz_h),
                                      ratio(["closure|mu"], iz_v)])
            for cn, cv in (("sigma_pi", widths[m]["pi"]), ("sigma_k", widths[m]["k"]),
                           ("sigma_mu_numu", widths[m]["mu_numu"]),
                           ("sigma_mu_nue", widths[m]["mu_nue"])):
                d[cn] = cv
            name = {("old", "flat"): "offaxis_excess_rebuild_old.npz",
                    ("v2", "flat"): "offaxis_excess_v2.npz",
                    ("old", "channel"): "offaxis_excess_channel.npz",
                    ("v2", "channel"): "offaxis_excess_channel_v2.npz"}[(m, mode)]
            path = os.path.join(out_dir, name)
            np.savez(path, **d)
            out[(m, mode)] = path
            if verbose:
                j = int(np.argmin(abs(ep_grid - 0.3)))
                print(f"  saved {path}  E_off(0.3GeV, horizon) flat="
                      f"{d['E_off'][0, j]:.3f}  numu={d['E_off_s'][0, 0, j]:.3f}  "
                      f"nue={d['E_off_s'][2, 0, j]:.3f}")
    return out


def cone_excess(cz, sigma_deg, x_grid, ep_grid, p, geom, n_alpha=44, n_beta=18):
    """E_off(cz, E) for an arbitrary production-angle kernel sigma_deg(E)."""
    return np.array(
        [
            e_off_for_zenith(
                c, ep_grid, sigma_deg, x_grid, ep_grid, p, geom, n_alpha, n_beta
            )
            for c in cz
        ]
    )


def build(out="offaxis_excess.npz", cone_kernel="moments", moments=None):
    """Build the LEGACY flat E_off table (``mceq3d_flux.EOFF_TABLE_FLAT``).

    Since 2026-09-04 the delivered table is the species-resolved
    ``offaxis_excess_channel_v2.npz`` from :func:`build_channel`; this builder
    stays as the A/B reference, so ``moments`` defaults to the **pre-arcsin**
    ``MOMENTS_LEGACY`` set -- otherwise re-running ``--build`` would silently
    replace the historical A/B file with a v2-moment one under the same name.
    Pass ``moments=MOMENTS_V2`` for the v2 flat build (which is exactly what
    ``build_channel`` writes to ``offaxis_excess_v2.npz``).

    ``cone_kernel``: "moments" (DEFAULT, recommended --
    Gaussian of the NA61-validated moment sigma_pi) or "sampled" (legacy -- the full
    (x_L,theta) generator kernel).

    Why "moments" is now the default (root-caused 2026-07-16, diag_kernel_consistency
    / diag_kernel_stats / diag_cone_fix): the sampled kernel k_spliced.npz is
    ~16-37% WIDER in per-secondary MESON-angle RMS than the exact moment m_spliced.npz,
    the discrepancy growing with energy as the true angle shrinks toward the kernel's
    coarse 0.667-deg theta bins -- a histogram bin-center inflation of the forward-
    peaked production angle. m_spliced is exact (theta=arctan(pT/pL) per secondary, no
    binning) and NA61-<pT>-validated, so it is the accurate input. Using it removes
    the sampled kernel's 0.5-1 GeV horizon overshoot (nu_mu +12.5%->+4.8%, nu_e
    +5.3%->-1.9% vs Honda; matches Bartol too, both flavours), leaving 0.3 GeV and
    high-E agreement intact. Rebuild the legacy table with cone_kernel="sampled"."""
    global _RHO

    print("[1/3] MCEq depth-resolved production profile p(X,E) ...")
    x_grid, ep_grid, p, dm = production_profile()
    _RHO = _rho_of_h(dm)
    print(f"      p(X,E): X in [{x_grid[0]:.1f},{x_grid[-1]:.1f}] g/cm2, "
          f"nE={len(ep_grid)}")

    print("[2/3] curved-atmosphere slant-depth table X_slant(h,psi) ...")
    geom = slant_depth_table(_RHO)

    print(f"[3/3] off-axis excess (cone_kernel={cone_kernel!r} pion + Gaussian kaon)")
    from kinematic_kernel import muon_shape

    cz = np.round(np.arange(0.05, 1.0, 0.1), 2)  # Honda-style bin centres
    ck = {"cone_kernel": cone_kernel,
          "moments": MOMENTS_LEGACY if moments is None else moments}
    E_off = offaxis_excess(cz, x_grid, ep_grid, p, geom, **ck)
    # NA61 +-12% pion-angle variants -> nuisance-parameter Jacobian for the
    # engine's covariance (sigma_pi_NA61 pull, see mceq3d_flux.solve).
    print("      +-12% NA61 sigma variants (covariance pull) ...")
    E_hi = offaxis_excess(cz, x_grid, ep_grid, p, geom, sigma_scale=1.12, **ck)
    E_lo = offaxis_excess(cz, x_grid, ep_grid, p, geom, sigma_scale=0.88, **ck)
    # Muon-calibration closure: the same off-axis factor evaluated with the
    # muon angular kernel must be ~1 in daemonflux's calibration region
    # (E_mu >~ 5 GeV), otherwise folding E_off onto the muon-calibrated base
    # would break the muon/neutrino consistency the calibration relies on.
    sig_mu = muon_shape(ep_grid, moments=ck["moments"])
    E_mu = cone_excess(
        np.array([0.05, 0.95]), sig_mu, x_grid, ep_grid, p["tot"], geom
    )
    print("      muon closure E_off_mu (horizon, vertical):")
    for E in (5.0, 10.0, 30.0):
        ie = int(np.argmin(abs(ep_grid - E)))
        print(f"        E={E:5.1f} GeV  {E_mu[0, ie]:.4f}  {E_mu[1, ie]:.4f}")
    j = int(np.argmin(abs(ep_grid - 0.3)))
    for i, c in enumerate(cz):
        print(f"      cosZ={c:.2f}  E_off(0.3GeV)={E_off[i, j]:.3f}")
    np.savez(
        out, e=ep_grid, cz=cz, E_off=E_off, E_off_hi=E_hi, E_off_lo=E_lo,
        E_off_mu=E_mu, tag=TAG,
    )
    print(f"saved -> {out}")
    return ep_grid, cz, E_off


def validate(plot=False):
    """Compare the *derived* delivered zenith shape (base x E_off) to Honda and
    Bartol, for both nu_mu and nu_e. Neither reference is used to build E_off, so
    both are independent tests. The 1D base carries the ordinary sec-theta
    enhancement (which dominates the horizon/vertical ratio above ~2 GeV), so the
    honest comparison folds the MCEq base in."""
    import validate_bartol as vb
    from mceq3d_flux import MCEq3DFlux

    d = np.load("offaxis_excess.npz")
    e, cz, E_off = d["e"], d["cz"], d["E_off"]  # single flavour-independent table
    iv = int(np.argmin(np.abs(cz - 0.95)))

    eng = MCEq3DFlux(base_model="mceq", daemonflux_location="generic")
    allbase = eng.base(cz)
    h = dict(np.load("honda_kam.npz"))
    He, Hcz = h["E"], h["czlo"]

    def _at(y, x, E):
        return np.exp(np.interp(np.log(E), np.log(x), np.log(np.maximum(y, 1e-300))))

    fig_data = {}
    for fl, hkey, bkey in (("numu", "numu", "num"), ("nue", "nue", "nue")):
        base = allbase[f"total_{fl}"]
        base_e = np.array(
            [np.exp(np.interp(np.log(e), np.log(eng.e), np.log(base[i])))
             for i in range(len(cz))]
        )
        deliv = base_e * E_off  # same E_off, per-flavour base
        nm = h[hkey]
        Eb, czb, gb = vb.load_bartol(bkey, "fmin")

        def hon(c, nm=nm):
            return _at(nm[int(np.argmin(np.abs(Hcz - (c - 0.05))))].mean(0), He, e)

        def bar(c, gb=gb, Eb=Eb, czb=czb):
            return _at(gb[int(np.argmin(np.abs(czb - c)))], Eb, e)

        hon_v, bar_v = hon(0.95), bar(0.95)
        print(f"\n[{fl}] delivered zenith shape (base x E_off) vs Honda/Bartol:")
        print("  E[GeV] cosZ  base*Eoff  Honda  Bartol  ours/Honda")
        for E in (0.3, 0.5, 1.0, 3.0, 10.0):
            ie = int(np.argmin(np.abs(e - E)))
            for i in (0, 1, 2):
                ds = deliv[i, ie] / deliv[iv, ie]
                hs = hon(cz[i])[ie] / hon_v[ie]
                print(
                    f"  {E:5.2f} {cz[i]:.2f}   {ds:6.3f}   {hs:5.3f}  "
                    f"{bar(cz[i])[ie] / bar_v[ie]:5.3f}   {ds / hs:5.2f}"
                )
        fig_data[fl] = (base_e / base_e[iv], deliv / deliv[iv],
                        {c: hon(c) / hon_v for c in cz},
                        {c: bar(c) / bar_v for c in cz})

    if plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        labels = {"numu": r"\nu_\mu", "nue": r"\nu_e"}
        for row, fl in enumerate(("numu", "nue")):
            base_sh, deliv_sh, hond, bart = fig_data[fl]
            for ax, E in zip(axes[row], (0.3, 1.0)):
                ie = int(np.argmin(np.abs(e - E)))
                ax.plot(cz, base_sh[:, ie], "C0o--", label="1D base (no E_off)")
                ax.plot(cz, deliv_sh[:, ie], "C3o-", lw=2,
                        label="this work (base x E_off)")
                ax.plot(cz, [hond[c][ie] for c in cz], "k-", label="Honda")
                ax.plot(cz, [bart[c][ie] for c in cz], "C2s:", label="Bartol")
                ax.set_xlabel(r"$\cos\theta_z$ (1=vertical, 0=horizon)")
                ax.set_ylabel(rf"$\Phi_{{{labels[fl]}}}/\Phi(\rm vert)$")
                ax.set_title(rf"${labels[fl]}$ zenith shape at {E:g} GeV (Kamioka)")
                ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig("offaxis_excess.png", dpi=110)
        print("saved plot -> offaxis_excess.png")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--cone-kernel", choices=("sampled", "moments"),
                    default="moments",
                    help="pion cone: 'moments' (default, NA61-validated sigma_pi "
                         "Gaussian) or 'sampled' (legacy full kernel, ~20-30%% too "
                         "wide at 0.5-1 GeV -> horizon overshoot)")
    ap.add_argument("--build-channel", action="store_true",
                    help="build the 2x2 {old,v2 moments} x {pion-only, "
                         "channel-weighted cone} E_off scan (four new files; "
                         "offaxis_excess.npz is never touched)")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--plot", action="store_true", help="save offaxis_excess.png")
    args = ap.parse_args(argv)
    if args.build:
        build(cone_kernel=args.cone_kernel)
    if args.build_channel:
        build_channel(n_jobs=args.n_jobs)
    if args.validate or args.plot:
        validate(plot=args.plot)
    if not (args.build or args.build_channel or args.validate or args.plot):
        ap.print_help()


if __name__ == "__main__":
    main()
