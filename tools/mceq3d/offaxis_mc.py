"""First-principles off-axis 3D production excess E_off(E, cosZ).

This derives the near-horizon 3D enhancement *from first principles* -- with no
anchoring to any reference flux. It is the complete genuine-3D/1D production
factor: it both redistributes flux in zenith and produces the sub-GeV horizontal
excess, and is the single 3D-production term used by ``mceq3d_flux`` (``offaxis=True``).

E_off is **flavour-independent** (the excess is geometric, inherited from meson
production and common to all pi/K/muon decay chains): one table is applied to each
flavour's own 1D base. ``--validate`` confirms this reproduces *both* the Honda and
Bartol nu_mu and nu_e zenith shapes to ~5% sub-GeV; the much larger nu_e flux
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

Build the table (writes ``offaxis_excess.npz`` with keys e, cz, E_off)::

    python offaxis_mc.py --build

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


def production_profile(e_lo=0.1, e_hi=100.0, n_x=80, theta_deg=0.0, rc_cut_gv=None):
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
    acc = np.array(
        [mc.get_solution("total_numu", 0, grid_idx=i)[sel] for i in range(n_x)]
    )  # (n_x, nE) accumulated numu flux at each depth
    p = np.gradient(acc, x_grid, axis=0)
    p = np.clip(p, 0.0, None)
    return x_grid, e[sel], p, mc.density_model


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
def e_off_for_zenith(
    cos_theta, e_grid, sigma_deg, x_grid, ep_grid, p, geom, n_alpha=60, n_beta=24
):
    """E_off(E) for one arrival cos(zenith), integrated along the ray with a
    Gaussian-on-the-sphere production cone of RMS sigma_theta(E)."""
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
        if s1 < 1e-4:
            num[k] = den[k]
            continue
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


def offaxis_excess(cz, x_grid, ep_grid, p, geom, n_alpha=44, n_beta=18, moments=None):
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
    from kinematic_kernel import channel_shapes

    kw = {} if moments is None else {"moments": moments}
    sig_pi = channel_shapes(ep_grid, **kw)["pi"]  # generator pion prod. angle [deg]
    return cone_excess(cz, sig_pi, x_grid, ep_grid, p, geom, n_alpha, n_beta)


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


def build(out="offaxis_excess.npz"):
    global _RHO

    print("[1/3] MCEq depth-resolved production profile p(X,E) ...")
    x_grid, ep_grid, p, dm = production_profile()
    _RHO = _rho_of_h(dm)
    print(f"      p(X,E): X in [{x_grid[0]:.1f},{x_grid[-1]:.1f}] g/cm2, "
          f"nE={len(ep_grid)}")

    print("[2/3] curved-atmosphere slant-depth table X_slant(h,psi) ...")
    geom = slant_depth_table(_RHO)

    print("[3/3] off-axis excess (flavour-independent pion cone) ...")
    from kinematic_kernel import channel_shapes, muon_shape

    sig_pi = channel_shapes(ep_grid)["pi"]
    cz = np.round(np.arange(0.05, 1.0, 0.1), 2)  # Honda-style bin centres
    E_off = cone_excess(cz, sig_pi, x_grid, ep_grid, p, geom)
    # NA61 +-12% pion-angle variants -> nuisance-parameter Jacobian for the
    # engine's covariance (sigma_pi_NA61 pull, see mceq3d_flux.solve).
    print("      +-12% NA61 sigma variants (covariance pull) ...")
    E_hi = cone_excess(cz, sig_pi * 1.12, x_grid, ep_grid, p, geom)
    E_lo = cone_excess(cz, sig_pi * 0.88, x_grid, ep_grid, p, geom)
    # Muon-calibration closure: the same off-axis factor evaluated with the
    # muon angular kernel must be ~1 in daemonflux's calibration region
    # (E_mu >~ 5 GeV), otherwise folding E_off onto the muon-calibrated base
    # would break the muon/neutrino consistency the calibration relies on.
    sig_mu = muon_shape(ep_grid)
    E_mu = cone_excess(np.array([0.05, 0.95]), sig_mu, x_grid, ep_grid, p, geom)
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
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--plot", action="store_true", help="save offaxis_excess.png")
    args = ap.parse_args(argv)
    if args.build:
        build()
    if args.validate or args.plot:
        validate(plot=args.plot)
    if not (args.build or args.validate or args.plot):
        ap.print_help()


if __name__ == "__main__":
    main()
