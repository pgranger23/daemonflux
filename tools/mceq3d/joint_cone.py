"""Joint production x geomagnetic-cutoff cone integral (Phase 1, step 3).

WHY THIS EXISTS
---------------
The delivered engine (:mod:`mceq3d_flux`) writes the directional flux as a
*product of two independently cone-averaged factors*

    Phi_s(E, n) = Phi_1D_s(E,|cosZ|) * E_off(E, cosZ) * <G_s>_cone(E, n)      (P)

where (``offaxis_mc.cone_numden``)

    E_off = <p>_cone / p_axis,
    <p>_cone = Int dl rho(h) Int dOmega_p W(alpha;E) p(X_slant(l, psi_p(n_p)), E)
    p_axis   = Int dl rho(h)                        p(X_slant(l, psi_o), E)

and (``mceq3d_flux.cone_geff``)

    <G_s>_cone = Int dOmega_p W(alpha;E) G_s(E, R_c(n_p)).

The physical object is the SINGLE joint integral over the production altitude
``l`` along the arrival ray and over the primary direction ``n_p``:

    J_s(E, n) = Int dl rho(h) Int dOmega_p W(alpha;E)
                    p(X_slant(l, psi_p(n_p)), E) * G_s(E, R_c(n_p))           (J)

so that the delivered product (P) drops the cone covariance

    Cov_cone(p, G) = J_s - <p>_cone * <G_s>_cone ,

and the correction factor this module measures is

    C_s(E, n) = J_s / ( <p>_cone * <G_s>_cone )                               (C)

with the "joint 3D factor" that would REPLACE ``E_off * <G>_cone`` being

    F_s(E, n) = J_s / p_axis  =  E_off * <G>_cone * C_s.                      (F)

The term is exactly zero at the vertical (there G is flat over the cone and p is
symmetric about the axis) and maximal at the horizon, for two reasons that pull
in opposite directions:

  * ``offaxis_mc.cone_numden`` BLOCKS Earth-shadowed / sub-limb primaries
    (``X_slant = inf`` -> ``p = 0``, offaxis_mc.py:169-172, :187-206), whereas
    ``mceq3d_flux.cone_geff`` substitutes the ANTIPODAL far-side cutoff for the
    same directions (mceq3d_flux.py:735-742 + the full-sphere map). At 87 deg
    30-41% of the cone solid-angle weight is sub-limb. In (J) those samples
    carry ``p = 0`` and drop out of BOTH the numerator and (through ``<p>``) the
    normalisation -- they cannot contribute their far-side ``G`` any more.
  * Above the limb the cone is anti-correlated in the East (a more vertical
    primary has a *smaller* X_slant, i.e. higher p, AND a *lower* R_c, i.e.
    higher G) so ``Cov > 0``; in the West the cutoff gradient flips sign.

WHAT IS REUSED
--------------
Everything physical is imported, not re-derived:

  * ``offaxis_mc.production_profile``  -- p(X, E) from a depth-resolved MCEq run
  * ``offaxis_mc._rho_of_h``          -- rho(h) from MCEq's CORSIKA atmosphere
  * ``offaxis_mc.slant_depth_table``  -- curved-atmosphere X_slant(h, psi)
                                         (this is what blocks sub-limb rays)
  * ``offaxis_mc._interp_xslant`` / ``_p_at`` / ``_regrid``
  * ``kinematic_kernel.channel_shapes`` -- the NA61-validated moment sigma_pi,
    sigma_K used by ``offaxis_mc.offaxis_excess`` (offaxis_mc.py:379-381)
  * ``mceq3d_flux.MCEq3DFlux.geomag_response`` -- the cascade-correct G_s(E,R_c)
  * ``mceq3d_flux.MCEq3DFlux.finemap_rc`` / any cached map array

Only two things are re-expressed here, because both live inside function bodies
in the delivered code and are not importable:

  * :func:`arrival_ray`  -- the 8-line altitude parametrisation of the arrival
    ray at the top of ``offaxis_mc.cone_numden`` (offaxis_mc.py:246-262);
  * :func:`rc_bilinear`  -- the ``rc_at`` closure of ``mceq3d_flux.cone_geff``
    (mceq3d_flux.py:888-901), verbatim but vectorised.

Both are covered by gates in ``test_joint_cone.py``: with ``G == 1`` the joint
factor (F) must reproduce ``offaxis_mc.e_off_for_zenith`` (hence the delivered
``offaxis_excess.npz``) on the same quadrature.

CONE CONVENTION (why one construction serves both integrals)
------------------------------------------------------------
``offaxis_mc`` defines the cone offset ``(alpha, beta)`` about the arrival
direction with ``beta`` measured from the plane containing the arrival direction
and the local vertical AT THE PRODUCTION POINT, and uses the spherical law of
cosines ``cos psi_p = cos a cos psi_o + sin a sin psi_o cos b``.
``mceq3d_flux.cone_geff`` builds the same offset as a 3-vector in the DETECTOR
frame with ``e1 ~ z_hat - n_z n`` ("toward the vertical").  Detector, production
point, Earth centre and the whole arrival ray are coplanar, so the local
vertical at the production point lies in that same plane; the unit vector in the
plane orthogonal to ``n`` is unique up to sign, hence ``e1_prod == e1_det``.
The two constructions therefore describe the SAME 3-vector ``n_p``: we build it
once (detector frame, for R_c) and read its production-point zenith from the law
of cosines (for X_slant). This is asserted numerically in the tests.

DEFAULTS / KNOWN CAVEATS
------------------------
* The cutoff map is an INPUT: pass ``(zen_deg, az_deg, rc)`` from
  :func:`load_rc_map` (any ``.npz`` written by ``MCEq3DFlux.finemap_rc``) or
  from a fresh ``finemap_rc`` call.  The **repaired** map (bisection scan, dense
  limb nodes, 42x25, no saturation) is what ``MCEq3DFlux.solve`` now builds; run
  ``diag_joint_cone.py --map .cache3d/finerc_<hash>.npz`` to use it.  The module
  default is still the *committed/paper* full-sphere Kamioka map
  ``.cache3d/finerc_ced98f306edd8e46.npz`` (13+13 zeniths x 25 azimuths,
  ``r_hi = 40 GV``, ``n_scan = 20``, vertical R_c = 11.93 GV).  It carries the
  defects listed in the audit (2.08 GV rigidity quantisation, bilinear
  interpolation over a 7.4 deg zenith grid, saturation at 40 GV at 89 deg East,
  the 89-deg seam).  A finer map can be dropped in unchanged.
* The cone used here is the **offaxis_mc** one (Gaussian of the NA61-validated
  moment ``sigma_pi``, per-axis width ``sigma/sqrt(2)``, ``sin alpha`` measure,
  plus the separate kaon channel).  It is deliberately NOT ``cone_geff``'s
  (sampled ``k_spliced`` kernel, missing ``1/sqrt(2)``, 70-deg tail bin), which
  the audit shows is 41% too wide: ``C_s`` must be measured on ONE self-
  consistent cone or it mixes the covariance with a cone-width error.
* Muon bending and the channel-blended (mu-decay) cone of ``cone_geff`` are not
  applied.  They shift/widen the cone axis but do not create the covariance;
  a wider cone makes |C_s - 1| larger, so the numbers here are conservative.
  Extra channels can be supplied through ``channels=``.
* The production-point displacement of the R_c lookup (``prod_displacement`` in
  ``cone_geff``) is off, matching the delivered default.
"""

from __future__ import annotations

import os

import numpy as np

import offaxis_mc as ox
from mceq3d_flux import SPECIES, RC_MAX_GV  # noqa: F401  (RC_MAX_GV re-exported)

R_EARTH_CM = ox.R_EARTH_CM
HERE = os.path.dirname(os.path.abspath(__file__))

#: The full-sphere Kamioka cutoff map behind the *committed* (paper) numbers:
#: 13 down-going + 13 up-going zeniths x 25 azimuths, r_hi = 40 GV, n_scan = 20,
#: vertical R_c = 11.93 GV, 89-deg East saturated at the 40 GV ceiling.
PAPER_MAP = os.path.join(HERE, ".cache3d", "finerc_ced98f306edd8e46.npz")
#: The working-tree map (r_hi = 55 GV, n_scan = 24): vertical R_c = 8.79 GV
#: (the regression the audit flags), 49.08 GV at 89 deg East, 9.5 GV limb seam.
WORKTREE_MAP = os.path.join(HERE, ".cache3d", "finerc_8e83b6248bd62681.npz")


# ---------------------------------------------------------------------------
# geometry (mirrors of two non-importable internals of the delivered code)
# ---------------------------------------------------------------------------
def arrival_ray(cos_theta, n_ray=260, h_max_cm=80e5, r_earth=None):
    """Arrival-ray parametrisation by altitude, as in ``offaxis_mc.cone_numden``.

    Returns ``(h_ray, ell, psi_o, w)`` where ``h_ray`` [cm] are the sample
    altitudes, ``ell`` [cm] the distance along the ray from the detector,
    ``psi_o`` [rad] the LOCAL zenith of the arrival direction at that altitude
    (it steepens with altitude on a near-horizontal ray -- the curved-atmosphere
    effect that makes E_off), and ``w = rho(h) dl`` the production-length
    weight common to numerator and denominator.

    Verbatim geometry of ``offaxis_mc.cone_numden`` (offaxis_mc.py:246-262);
    ``offaxis_mc._RHO`` must have been set (see :func:`load_production`).
    """
    re_cm = R_EARTH_CM if r_earth is None else float(r_earth)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    h_ray = np.linspace(0.0, h_max_cm, n_ray)
    r = re_cm + h_ray
    disc = (re_cm * np.cos(theta)) ** 2 + (r**2 - re_cm**2)
    ell = -re_cm * np.cos(theta) + np.sqrt(np.clip(disc, 0, None))
    cos_psi_o = np.clip((ell + re_cm * np.cos(theta)) / r, -1, 1)
    psi_o = np.arccos(cos_psi_o)
    w = ox._RHO(h_ray) * np.gradient(ell)
    return h_ray, ell, psi_o, w


def rc_bilinear(zen_f, az_f, rc_f, th_deg, ph_deg):
    """Vectorised copy of the ``rc_at`` closure in ``mceq3d_flux.cone_geff``.

    Bilinear in (zenith, azimuth), azimuth periodic, zenith clamped to the map
    range (mceq3d_flux.py:888-901).  ``rc_f`` is (n_zen, n_az) [GV].
    """
    th = np.clip(np.asarray(th_deg, float), zen_f[0], zen_f[-1])
    ph = np.asarray(ph_deg, float) % 360.0
    iz = np.clip(np.searchsorted(zen_f, th) - 1, 0, len(zen_f) - 2)
    ja = np.clip(np.searchsorted(az_f, ph) - 1, 0, len(az_f) - 2)
    tz = (th - zen_f[iz]) / (zen_f[iz + 1] - zen_f[iz])
    ta = (ph - az_f[ja]) / (az_f[ja + 1] - az_f[ja])
    return (
        rc_f[iz, ja] * (1 - tz) * (1 - ta)
        + rc_f[iz + 1, ja] * tz * (1 - ta)
        + rc_f[iz, ja + 1] * (1 - tz) * ta
        + rc_f[iz + 1, ja + 1] * tz * ta
    )


def cone_directions(cos_theta, az_deg, alpha_rad, beta_rad):
    """Unit primary directions ``n_p`` (n_alpha, n_beta, 3) in the detector
    (north, east, up) frame -- the construction of ``mceq3d_flux.cone_geff``
    (mceq3d_flux.py:932-941).

    ``e1`` points "toward the vertical" in the plane spanned by the arrival
    direction and the local vertical; because that plane also contains the
    production point's local vertical, ``beta`` here is the SAME azimuthal
    reference as the ``cos beta`` of ``offaxis_mc``'s law of cosines.
    """
    th = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    phi = np.radians(az_deg)
    n = np.array([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)])
    e1 = np.array([0.0, 0.0, 1.0]) - n[2] * n
    nn = np.linalg.norm(e1)
    e1 = e1 / nn if nn > 1e-9 else np.array([1.0, 0.0, 0.0])
    e2 = np.cross(n, e1)
    ca, sa = np.cos(alpha_rad)[:, None, None], np.sin(alpha_rad)[:, None, None]
    cb, sb = np.cos(beta_rad)[None, :, None], np.sin(beta_rad)[None, :, None]
    nb, e1b, e2b = n[None, None, :], e1[None, None, :], e2[None, None, :]
    return ca * nb + sa * (cb * e1b + sb * e2b)


# ---------------------------------------------------------------------------
# cone weights
# ---------------------------------------------------------------------------
def alpha_grid(n_alpha=44):
    """Cone polar grid [rad] with an extra exact-axis node prepended.

    ``alpha[1:]`` is exactly ``offaxis_mc.cone_numden``'s
    ``deg2rad(linspace(0.5, 89, n_alpha))`` (offaxis_mc.py:277); ``alpha[0] = 0``
    is used only for the degenerate ``sigma -> 0`` branch, which
    ``offaxis_mc.cone_numden`` handles by short-circuiting ``num = den``
    (offaxis_mc.py:281-283).  For any finite sigma its weight is zero, so the
    quadrature is unchanged.
    """
    return np.concatenate([[0.0], np.deg2rad(np.linspace(0.5, 89.0, n_alpha))])


def gauss_alpha_weights(sigma_deg, alpha_rad, sigma_floor=1e-4):
    """``W[nE, n_alpha]`` -- normalised Gaussian-on-the-sphere cone weights.

    ``wa = sin(alpha) exp(-alpha^2 / (2 s1^2))`` with the per-axis width
    ``s1 = sigma / sqrt(2)`` of a space-angle RMS ``sigma`` -- exactly
    ``offaxis_mc.cone_numden`` (offaxis_mc.py:279-289).  Energies whose
    ``s1 < sigma_floor`` put all weight on the axis node ``alpha[0] = 0``,
    reproducing that function's ``num = den`` short circuit.
    """
    sigma_deg = np.atleast_1d(np.asarray(sigma_deg, float))
    W = np.zeros((len(sigma_deg), len(alpha_rad)))
    a = alpha_rad[1:]
    for k, sdeg in enumerate(sigma_deg):
        s1 = np.deg2rad(sdeg) / np.sqrt(2.0)
        if s1 < sigma_floor:
            W[k, 0] = 1.0
            continue
        wa = np.sin(a) * np.exp(-(a**2) / (2 * s1**2))
        tot = wa.sum()
        if tot <= 0:  # numerically underflowed -> treat as collimated
            W[k, 0] = 1.0
        else:
            W[k, 1:] = wa / tot
    return W


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def load_rc_map(path=PAPER_MAP):
    """``(zen_deg, az_deg, rc_gv)`` from any ``finemap_rc`` cache file.

    Accepts anything produced by ``MCEq3DFlux.finemap_rc`` (keys ``zen``, ``az``,
    ``rc``), so a rebuilt map with a finer rigidity scan and denser limb nodes
    drops in with no code change.
    """
    d = np.load(path)
    return np.asarray(d["zen"], float), np.asarray(d["az"], float), np.asarray(
        d["rc"], float
    )


def load_production(cache=None, species=(), **kw):
    """p(X,E), rho(h) and the curved-geometry table, memoised to ``cache``.

    Thin wrapper on ``offaxis_mc.production_profile`` + ``_rho_of_h`` +
    ``slant_depth_table``; also sets the ``offaxis_mc._RHO`` module handle that
    ``arrival_ray`` (and ``offaxis_mc.cone_numden``) need.

    ``species`` (e.g. ``offaxis_mc.CHANNEL_SPECIES``) additionally pulls MCEq's
    depth-resolved **per-parent-channel** profiles ``{s}_dir`` / ``{s}_k`` /
    ``{s}_mu`` (``offaxis_mc.production_profile``, offaxis_mc.py:115).  Those are
    what the channel-resolved joint factor blends: because Eq. (8) is linear in
    ``p``, summing the per-channel numerators (each with its own cone) over a
    common denominator IS the production-weighted channel blend, with the
    fractions ``f_c = D_c / sum_c D_c`` coming out of the same cascade --
    depth-resolved, so ``f_mu`` genuinely rises toward the horizon.

    Returns ``dict(x_grid, ep_grid, p_tot, p_k, geom[, chan])`` where ``chan`` is
    ``{species: (p_dir, p_k, p_mu)}``.
    """
    keys = tuple(species)
    if cache is not None and os.path.exists(cache):
        d = np.load(cache)
        geom = (d["h_grid"], d["psi_grid"], d["table"])
        rho_h, rho_v = d["rho_h"], d["rho_v"]
        ox._RHO = lambda h, _h=rho_h, _v=rho_v: np.interp(
            h, _h, _v, left=_v[0], right=0.0
        )
        out = dict(x_grid=d["x_grid"], ep_grid=d["ep_grid"], p_tot=d["p_tot"],
                   p_k=d["p_k"], geom=geom)
        if keys and all(f"p_{s}_dir" in d for s in keys):
            out["chan"] = {s: (d[f"p_{s}_dir"], d[f"p_{s}_k"], d[f"p_{s}_mu"])
                           for s in keys}
            return out
        if not keys:
            return out  # cached file predates the channel split; rebuild below
    x_grid, ep_grid, p, dm = ox.production_profile(species=keys, **kw)
    ox._RHO = ox._rho_of_h(dm)
    geom = ox.slant_depth_table(ox._RHO)
    out = dict(x_grid=x_grid, ep_grid=ep_grid, p_tot=p["tot"], p_k=p["k"], geom=geom)
    extra = {}
    if keys:
        out["chan"] = {s: (p[f"{s}_dir"], p[f"{s}_k"], p[f"{s}_mu"]) for s in keys}
        for s in keys:
            for c in ("dir", "k", "mu"):
                extra[f"p_{s}_{c}"] = p[f"{s}_{c}"]
    if cache is not None:
        h_tab = np.linspace(0.0, ox.H_TOP_CM, 4000)
        np.savez(
            cache,
            x_grid=x_grid,
            ep_grid=ep_grid,
            p_tot=p["tot"],
            p_k=p["k"],
            h_grid=geom[0],
            psi_grid=geom[1],
            table=np.minimum(geom[2], 1e30),
            rho_h=h_tab,
            rho_v=ox._RHO(h_tab),
            **extra,
        )
    return out


def gs_on_grid(engine, rc_grid, cz_ref, e_out, cache_dir=None, sigma_lnr=None):
    """``{species: G[n_rc, len(e_out)]}`` -- ``MCEq3DFlux.geomag_response``
    re-gridded (log-E) onto ``e_out``.  ``sigma_lnr`` is the cutoff penumbra
    width (``None`` = ``mceq3d_flux.SIGMA_LNR``)."""
    G, _ = engine.geomag_response(rc_grid, cz_ref=cz_ref, cache_dir=cache_dir,
                                  sigma_lnr=sigma_lnr)
    return {s: ox._regrid(G[s], engine.e, e_out) for s in SPECIES}


# ---------------------------------------------------------------------------
# the joint integral
# ---------------------------------------------------------------------------
def _interp_G(rc_vals, rc_grid, gmat):
    """Linear interp of ``gmat`` (n_rc, nE) at an array of R_c -> (N, nE).

    Same clamping as ``mceq3d_flux._interp_rc``, vectorised over samples."""
    rc = np.clip(np.asarray(rc_vals, float).ravel(), rc_grid[0], rc_grid[-1])
    j = np.clip(np.searchsorted(rc_grid, rc), 1, len(rc_grid) - 1)
    w = (rc - rc_grid[j - 1]) / (rc_grid[j] - rc_grid[j - 1])
    return (1.0 - w)[:, None] * gmat[j - 1] + w[:, None] * gmat[j]


def cone_production(cos_theta, alpha_rad, beta_rad, x_grid, ep_grid, p, geom,
                    n_ray=260, ray=None):
    """``(q[n_alpha, n_beta, nE], q_axis[nE])`` -- depth-integrated production.

        q[a,b,E]  = Int dl rho(h) p(X_slant(l, psi_p(a,b,l)), E)
        q_axis[E] = Int dl rho(h) p(X_slant(l, psi_o(l)),     E)

    ``psi_p`` from the spherical law of cosines about the arrival direction,
    ``X_slant`` from ``offaxis_mc``'s curved table -- so Earth-shadowed /
    sub-limb primaries get ``X = inf`` and contribute exactly zero, identically
    to ``offaxis_mc.cone_numden``.
    """
    h_ray, _, psi_o, w = (arrival_ray(cos_theta, n_ray=n_ray) if ray is None
                          else ray)
    h_grid, psi_grid, table = geom
    cpo, spo = np.cos(psi_o), np.sin(psi_o)
    x_o = ox._interp_xslant(h_ray, psi_o, h_grid, psi_grid, table)
    q_axis = w @ ox._p_at(x_o, x_grid, ep_grid, p)
    nb, nray = len(beta_rad), len(h_ray)
    q = np.zeros((len(alpha_rad), nb, len(ep_grid)))
    cb = np.cos(beta_rad)
    h_rep = np.repeat(h_ray, nb)
    for ia, a in enumerate(alpha_rad):
        if a == 0.0:  # axis node: beta-independent, = the 1D denominator
            q[ia, :] = q_axis
            continue
        ca, sa = np.cos(a), np.sin(a)
        # all beta at once (n_ray, n_beta): ~n_beta x faster than one at a time
        psi_p = np.arccos(np.clip(
            ca * cpo[:, None] + sa * spo[:, None] * cb[None, :], -1, 1))
        xv = ox._interp_xslant(h_rep, psi_p.ravel(), h_grid, psi_grid, table)
        pk = ox._p_at(xv, x_grid, ep_grid, p).reshape(nray, nb, -1)
        q[ia] = np.einsum("l,lbe->be", w, pk)
    return q, q_axis


def cone_production_multi(cos_theta, alpha_rad, beta_rad, x_grid, ep_grid,
                          profiles, geom, n_ray=260, ray=None):
    """:func:`cone_production` for a LIST of production profiles at once.

    The cone geometry (``X_slant`` lookup, the ray quadrature) is identical for
    every profile and is by far the expensive part, so the profiles are stacked
    along the energy axis and gathered in a single pass.  Returns
    ``([q_i], [q_axis_i])``.
    """
    nE = len(ep_grid)
    P = np.concatenate([np.asarray(pi, float) for pi in profiles], axis=1)
    e_stack = np.arange(P.shape[1], dtype=float) + 1.0  # shape carrier only
    q, qa = cone_production(cos_theta, alpha_rad, beta_rad, x_grid, e_stack, P,
                            geom, n_ray=n_ray, ray=ray)
    return ([q[..., i * nE:(i + 1) * nE] for i in range(len(profiles))],
            [qa[i * nE:(i + 1) * nE] for i in range(len(profiles))])


def prod_point_frame(cos_theta, az_deg, h_prod_km):
    """Local (up, north) at the production point, expressed in the detector's
    (north, east, up) frame -- the mirror of ``mceq3d_flux.cone_geff._prod_frame``
    (mceq3d_flux.py, nested in :meth:`MCEq3DFlux.cone_geff`).

    Returns ``None`` when the displacement is negligible (near-vertical), in
    which case the caller uses the detector frame.
    """
    th = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    phi = np.radians(az_deg)
    n = np.array([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)])
    re = R_EARTH_CM / 1e5  # km
    r = re + h_prod_km
    disc = (re * n[2]) ** 2 + (r * r - re * re)
    L = -re * n[2] + np.sqrt(max(disc, 0.0))
    if L < 1.0:
        return None
    P = np.array([L * n[0], L * n[1], re + L * n[2]])
    up_p = P / np.linalg.norm(P)
    north_p = np.array([1.0, 0.0, 0.0]) - up_p[0] * up_p
    nn = np.linalg.norm(north_p)
    if nn < 1e-9:
        return None
    return up_p, north_p / nn


def local_angles(vec, frame):
    """(zenith, azimuth) [deg] of ``vec`` (..., 3) in a production-point frame --
    the mirror of ``cone_geff._local_angles``."""
    up_p, north_p = frame
    east_p = np.cross(up_p, north_p)
    cz = np.tensordot(vec, up_p, axes=([-1], [0]))
    th = np.degrees(np.arccos(np.clip(cz, -1.0, 1.0)))
    ph = np.degrees(np.arctan2(np.tensordot(vec, east_p, axes=([-1], [0])),
                               np.tensordot(vec, north_p, axes=([-1], [0])))) % 360.0
    return th, ph


def zenith_terms(
    cos_theta,
    prod,
    *,
    channels=None,
    moments=None,
    sigma_scale=1.0,
    n_alpha=44,
    n_beta=18,
    n_ray=260,
):
    """Azimuth-INDEPENDENT half of the joint integral, for one arrival zenith.

    The cone production term ``q[a,b,E]`` depends on the arrival direction only
    through the local zenith ``psi_o(l)`` along the ray, so it can be computed
    once per ``cos_theta`` and reused for every azimuth (only ``R_c`` -- and
    therefore ``G`` -- is azimuth-dependent).  Returns a dict consumed by
    :func:`joint_cone` through its ``terms=`` argument.
    """
    x_grid, ep_grid = prod["x_grid"], prod["ep_grid"]
    if channels is None:
        from kinematic_kernel import channel_shapes

        kw = {} if moments is None else {"moments": moments}
        sh = channel_shapes(ep_grid, **kw)
        channels = [
            (sh["pi"], prod["p_tot"] - prod["p_k"]),
            (sh["k"], prod["p_k"]),
        ]
    alpha = alpha_grid(n_alpha)
    beta = np.linspace(0.0, 2 * np.pi, n_beta, endpoint=False)
    terms = []
    for sigma_deg, p_ch in channels:
        W = gauss_alpha_weights(np.asarray(sigma_deg, float) * sigma_scale, alpha)
        q, q_ax = cone_production(
            cos_theta, alpha, beta, x_grid, ep_grid, p_ch, prod["geom"], n_ray=n_ray
        )
        terms.append((W, q, q_ax))
    return dict(cos_theta=float(cos_theta), alpha=alpha, beta=beta, terms=terms,
                ep_grid=ep_grid)


def joint_cone(
    cos_theta,
    az_deg,
    prod,
    fine,
    rc_grid,
    G,
    *,
    channels=None,
    moments=None,
    sigma_scale=1.0,
    n_alpha=44,
    n_beta=18,
    n_ray=260,
    terms=None,
):
    """Joint / factorised cone integrals for one arrival direction.

    Parameters
    ----------
    cos_theta, az_deg
        Neutrino arrival direction at the site (azimuth measured from north
        toward east, the convention of the cutoff map and of ``cone_geff``).
    prod
        Output of :func:`load_production` (p(X,E), curved geometry).
    fine
        ``(zen_deg, az_deg, rc_gv)`` cutoff map -- any array/grid triple, e.g.
        from :func:`load_rc_map` or ``MCEq3DFlux.finemap_rc``.
    rc_grid, G
        The rigidity nodes and ``{species: G_s[n_rc, nE]}`` cascade-correct
        suppression on the SAME energy grid as ``prod['ep_grid']``
        (see :func:`gs_on_grid`).
    channels
        ``[(sigma_deg[nE], p[n_x, nE]), ...]``.  Default = the two channels of
        ``offaxis_mc.offaxis_excess``: the pion cone on ``p_tot - p_k`` and the
        (wider) kaon cone on ``p_k``.
    sigma_scale
        Multiplies every cone width (the NA61 +-12% pull; 0 collapses the cone).
    terms
        Optional output of :func:`zenith_terms` for the same ``cos_theta`` and
        cone settings, to avoid recomputing the production integral per azimuth.

    Returns
    -------
    dict with, on ``e`` (= ``prod['ep_grid']``):
        ``p_axis``   Int dl rho p(X(psi_o))                    -- the 1D denominator
        ``p_cone``   <p>_cone (the E_off numerator)
        ``E_off``    p_cone / p_axis
        ``G_cone``   {s: <G_s>_cone}    -- production-share-weighted over channels
        ``J``        {s: joint integral (J)}
        ``joint``    {s: J / p_axis}    -- the factor that replaces E_off * <G>
        ``product``  {s: E_off * G_cone}  (what the engine delivers)
        ``C``        {s: joint / product}  -- the covariance correction (C)
        ``rho_pG``   {s: Cov_cone(p,G)/(sigma_p sigma_G)} for the FIRST (pion)
                     channel only -- a normalised diagnostic of the correlation
        ``w_blocked`` cone weight fraction whose primary is fully Earth-shadowed
                     (``p == 0`` at every altitude on the ray)
        ``w_sublimb`` cone weight fraction whose primary direction is BELOW the
                     detector horizon, i.e. the samples for which ``cone_geff``
                     reads the antipodal far-side cutoff off the full-sphere map
        ``f_sublimb`` share of ``<p>_cone`` carried by those same samples
        ``rc``       R_c at each cone sample [GV]; ``rc_axis`` the axis value
        ``W``        the (composite, production-weighted) cone weights
    """
    if terms is None:
        terms = zenith_terms(
            cos_theta, prod, channels=channels, moments=moments,
            sigma_scale=sigma_scale, n_alpha=n_alpha, n_beta=n_beta, n_ray=n_ray,
        )
    ep_grid = terms["ep_grid"]
    alpha, beta = terms["alpha"], terms["beta"]

    # --- geomagnetic side: one R_c per cone sample (detector-frame lookup, the
    #     delivered default; prod_displacement is off, as in cone_geff) --------
    zen_f, az_f, rc_f = fine
    npv = cone_directions(cos_theta, az_deg, alpha, beta)  # (na, nb, 3) N,E,U
    th_p = np.degrees(np.arccos(np.clip(npv[..., 2], -1, 1)))
    ph_p = np.degrees(np.arctan2(npv[..., 1], npv[..., 0]))
    rc = rc_bilinear(zen_f, az_f, rc_f, th_p, ph_p)  # (na, nb)
    G_ab = {
        s: _interp_G(rc, rc_grid, G[s]).reshape(len(alpha), len(beta), -1)
        for s in SPECIES
    }
    rc_axis = float(rc[0, 0])
    sub = (th_p > 90.0).astype(float)  # (na, nb): reads the far-side map half

    # --- accumulate per channel -------------------------------------------
    nE = len(ep_grid)
    p_axis = np.zeros(nE)
    p_cone = np.zeros(nE)
    J = {s: np.zeros(nE) for s in SPECIES}
    Gnum = {s: np.zeros(nE) for s in SPECIES}  # sum_ch p_cone_ch * <G>_ch
    W_tot = np.zeros((nE, len(alpha)))
    blocked = np.zeros(nE)
    w_sub = np.zeros(nE)
    p_sub = np.zeros(nE)
    first = {}  # second moments of the first (pion) channel, for rho_pG
    for ich, (W, q, q_ax) in enumerate(terms["terms"]):
        # <.>_cone = sum_alpha W[E, alpha] * mean_beta(.)
        qb = q.mean(axis=1)  # (na, nE)  -- beta-averaged production
        pc = np.einsum("ea,ae->e", W, qb)
        p_axis += q_ax
        p_cone += pc
        W_tot += W * np.maximum(pc, 0.0)[:, None]
        blk = (q <= 0.0).mean(axis=1)  # fraction of beta blocked, per (alpha, E)
        blocked += np.einsum("ea,ae->e", W, blk) * np.maximum(pc, 0.0)
        w_sub += np.einsum("ea,a->e", W, sub.mean(axis=1)) * np.maximum(pc, 0.0)
        p_sub += np.einsum("ea,ae->e", W, (q * sub[:, :, None]).mean(axis=1))
        if ich == 0:
            first["p2"] = np.einsum("ea,ae->e", W, (q**2).mean(axis=1))
            first["p1"] = pc
            first["G1"], first["G2"], first["J"] = {}, {}, {}
        for s in SPECIES:
            gb = G_ab[s].mean(axis=1)  # (na, nE)
            gcone = np.einsum("ea,ae->e", W, gb)
            jc = np.einsum("ea,ae->e", W, (q * G_ab[s]).mean(axis=1))
            J[s] += jc
            Gnum[s] += pc * gcone
            if ich == 0:
                first["G1"][s] = gcone
                first["G2"][s] = np.einsum("ea,ae->e", W, (G_ab[s] ** 2).mean(axis=1))
                first["J"][s] = jc

    tiny = 1e-300
    pc_safe = np.maximum(p_cone, tiny)
    W_tot /= np.maximum(W_tot.sum(axis=1, keepdims=True), tiny)
    blocked /= pc_safe
    w_sub /= pc_safe
    E_off = p_cone / np.maximum(p_axis, tiny)
    G_cone = {s: Gnum[s] / pc_safe for s in SPECIES}
    out = dict(
        e=ep_grid,
        cos_theta=float(cos_theta),
        az_deg=float(az_deg),
        p_axis=p_axis,
        p_cone=p_cone,
        E_off=E_off,
        G_cone=G_cone,
        J=J,
        joint={s: J[s] / np.maximum(p_axis, tiny) for s in SPECIES},
        product={s: E_off * G_cone[s] for s in SPECIES},
        C={s: J[s] / np.maximum(p_cone * G_cone[s], tiny) for s in SPECIES},
        w_blocked=blocked,
        w_sublimb=w_sub,
        f_sublimb=p_sub / pc_safe,
        rc=rc,
        rc_axis=rc_axis,
        W=W_tot,
        alpha=alpha,
        beta=beta,
    )
    # cone-measure correlation coefficient of p and G, FIRST (pion) channel only
    # -- a normalised diagnostic; the delivered correction is ``C`` above.
    var_p = np.maximum(first["p2"] - first["p1"] ** 2, 0.0)
    out["rho_pG"] = {}
    for s in SPECIES:
        var_g = np.maximum(first["G2"][s] - first["G1"][s] ** 2, 0.0)
        cov = first["J"][s] - first["p1"] * first["G1"][s]
        den = np.sqrt(var_p * var_g)
        out["rho_pG"][s] = np.where(den > 0, cov / np.maximum(den, tiny), 0.0)
    return out


def at_energy(res, key, species, e_val):
    """Log-interpolate one of the returned arrays at a single energy."""
    arr = res[key][species] if isinstance(res[key], dict) else res[key]
    return float(np.interp(np.log(e_val), np.log(res["e"]), arr))


# ---------------------------------------------------------------------------
# the delivered-path object: F_s = J_s / p_axis, replacing E_off x <G>_cone
# ---------------------------------------------------------------------------
def delivered_joint_factor(
    cos_theta,
    azimuths,
    prod,
    fine,
    rc_grid,
    G,
    *,
    sigma_pi,
    sigma_k,
    sigma_mu=None,
    fmu=None,
    d_cone=None,
    mu_plus=(),
    prod_frame=True,
    h_prod_km=30.0,
    n_alpha=44,
    n_beta=18,
    n_ray=260,
    sigma_scale=1.0,
    q_cache=None,
    channels_by_species=None,
    sigma_mu_by_species=None,
    rc_family=None,
    site_lat=None,
    ray=None,
    renorm=None,
):
    """``{species: F[n_az, nE]}`` with ``F_s = J_s / p_axis`` -- the single joint
    integral that replaces the delivered product ``E_off x <G_s>_cone``.

    This is the engine-facing entry point (``MCEq3DFlux.joint_cone_factor``); the
    diagnostic/gate entry point is :func:`joint_cone`.

    Channels (matching what the delivered engine composes, but inside ONE
    integral instead of two averaged separately):

    * **pion** cone ``sigma_pi`` on the non-kaon production profile
      ``p_tot - p_k``  -- the ``offaxis_mc.offaxis_excess`` decomposition;
    * **kaon** cone ``sigma_k`` on ``p_k``;
    * **muon-decay** cone ``sigma_mu`` (wider) on the same non-kaon profile,
      blended against the pion channel by the per-species muon-decay flux
      fraction ``fmu[s]`` -- the ``cone_geff`` channel blend.  Its cutoff is read
      at the **charge-signed bending-shifted** direction ``n_p +- d_cone``
      (species in ``mu_plus`` take ``+``), exactly as ``cone_geff`` does; the
      *production* is left on the unshifted primary, because the bend happens
      after the shower has developed.

    Pass ``sigma_mu=None`` (or ``fmu=None``) for the pion+kaon-only cone: that
    configuration reduces, at ``G == 1``, to the delivered ``E_off`` table
    exactly (the gate in ``test_joint_cone.py``).

    Earth shadow is enforced by the production profile itself: a primary
    direction whose slant path strikes the solid Earth has ``X_slant = inf`` and
    hence ``q = 0`` (``offaxis_mc.slant_depth_table``), so it drops out of the
    numerator *and* of the normalisation with no separate geometric test -- the
    same physics as ``cone_geff(sublimb="prod_point")``'s explicit block, applied
    at the exact limb rather than at a 90-deg proxy.

    ``prod_frame`` (default True): read ``R_c`` in the local frame of the
    production point ``h_prod_km`` up the arrival ray, on the **down-going** part
    of ``fine`` only -- consistent with ``cone_geff(sublimb="prod_point")``.

    ``rc_family`` / ``site_lat`` (``cutoff_anchor="prod_point"``): read ``R_c``
    off a map LAUNCHED AT the production point instead of at the detector.
    ``prod_frame`` alone rotates the local vertical to P but still charges every
    cone sample the detector's cutoff; the primary, however, enters the
    atmosphere at P, which for a near-horizon arrival is 368 km (87 deg) to
    564 km (89.5 deg) away -- 3.3 to 5.1 deg of geomagnetic latitude, i.e. a
    12-16% change in ``R_c`` and an ~11 deg rotation of its azimuthal pattern.
    ``renorm`` (default ``None``, i.e. off): ``{species: g[nE]}`` multiplying
    ``F_s``.  This is the plug-point for the **Liouville constraint** of
    :func:`liouville_renorm` -- a renormalisation of the redistribution so that a
    chosen sky average of ``F`` hits a target.  It is a constraint, NOT a
    derivation: ``g`` depends on energy only, so it changes the absolute flux at
    every zenith by the same factor and cancels identically in any zenith ratio
    (horizon/vertical, East/West).

    ``ray`` (diagnostic only, default ``None``): a pre-built
    ``(h_ray, ell, psi_o, w)`` arrival-ray quadrature replacing
    :func:`arrival_ray`, used by ``diag_conservation.py`` to evaluate the same
    integral in a *flat* atmosphere where the exact answer is known analytically
    (``F = <cos alpha>``).  The delivered path never passes it.

    ``rc_family`` is the displaced-site map family of
    ``MCEq3DFlux.prod_family_rc``; for each arrival azimuth the map is
    interpolated to the production point's ground displacement
    (``mceq3d_flux.prod_point_offset_km``) and the cone samples are expressed in
    P's TRUE local frame (``mceq3d_flux.prod_frame_exact``, which -- unlike
    :func:`prod_point_frame` -- includes the convergence of the meridians, up to
    4 deg of azimuth for an east-west displacement).  ``site_lat`` is the
    detector latitude, needed for that frame.  The displacement is exactly zero
    at the vertical, so ``fine`` and the family agree there by construction.

    CHANNEL-RESOLVED MODE (``channels_by_species``, the delivered path since
    2026-09-04).  Pass ``{species: (p_dir, p_k, p_mu)}`` -- MCEq's own
    depth-resolved parent-channel profiles (``offaxis_mc.production_profile``,
    offaxis_mc.py:115) -- together with ``sigma_mu_by_species``.  Then

        J_s = N_dir(sigma_pi) + N_K(sigma_K) + N_mu(sigma_mu,s)
        F_s = J_s / (D_dir + D_K + D_mu)

    exactly as ``offaxis_mc.build_channel`` (offaxis_mc.py:591) composes its
    per-species table: the blend weights ``f_c = D_c / sum D`` are implicit in
    the profiles, so they are depth-resolved (``f_mu`` rises toward the horizon
    on its own) and species-resolved, replacing the single ``f_mu[s]`` from
    ``channel_fractions``.  ``fmu``/``sigma_mu`` are ignored in this mode.  The
    charge-signed bending shift still displaces the **muon channel's cutoff
    lookup** by ``+-d_cone``; use bending-free ``sigma_mu`` widths with it, or
    the coherent shift and the width floor double-count.
    """
    ep_grid = prod["ep_grid"]
    nE = len(ep_grid)
    alpha = alpha_grid(n_alpha)
    beta = np.linspace(0.0, 2 * np.pi, n_beta, endpoint=False)
    zen_f, az_f, rc_f = fine
    if prod_frame:  # no cone sample may read the antipodal far-side cutoff
        dn = zen_f <= 89.9
        zen_f, rc_f = zen_f[dn], rc_f[dn]

    # --- production side (azimuth-independent, memoised per zenith) -------
    chan = channels_by_species
    sp_order = tuple(sorted(chan)) if chan else ()
    key = (round(float(cos_theta), 9), n_alpha, n_beta, n_ray, sp_order,
           None if ray is None else id(ray))
    got = None if q_cache is None else q_cache.get(key)
    if got is None:
        if chan is None:
            plist = [prod["p_tot"] - prod["p_k"], prod["p_k"]]
        else:
            plist = [pc for s in sp_order for pc in chan[s]]
        qs, qas = cone_production_multi(cos_theta, alpha, beta, prod["x_grid"],
                                        ep_grid, plist, prod["geom"], n_ray=n_ray,
                                        ray=ray)
        got = (qs, qas)
        if q_cache is not None:
            q_cache[key] = got
    qs, qas = got

    W_pi = gauss_alpha_weights(np.asarray(sigma_pi, float) * sigma_scale, alpha)
    W_k = gauss_alpha_weights(np.asarray(sigma_k, float) * sigma_scale, alpha)
    if chan is None:
        q_pi, q_k = qs
        qa_pi, qa_k = qas
        p_axis = np.maximum(qa_pi + qa_k, 1e-300)
        W_mu = (None if sigma_mu is None else
                gauss_alpha_weights(np.asarray(sigma_mu, float) * sigma_scale, alpha))
    else:
        idx = {s: 3 * i for i, s in enumerate(sp_order)}
        p_axis_s = {s: np.maximum(qas[idx[s]] + qas[idx[s] + 1] + qas[idx[s] + 2],
                                  1e-300) for s in sp_order}
        W_mu_s = {s: gauss_alpha_weights(
            np.asarray(sigma_mu_by_species[s], float) * sigma_scale, alpha)
            for s in sp_order}
        p_axis = p_axis_s[sp_order[0]]

    out = {s: np.zeros((len(azimuths), nE)) for s in G}
    eoff_eq = np.zeros((len(azimuths), nE))
    blocked = np.zeros((len(azimuths), nE))
    zen_deg = float(np.degrees(np.arccos(np.clip(float(cos_theta), -1.0, 1.0))))
    for ia, azd in enumerate(azimuths):
        npv = cone_directions(cos_theta, azd, alpha, beta)  # (na, nb, 3)
        if rc_family is not None:  # cutoff_anchor="prod_point"
            import mceq3d_flux as _mf

            dn, de = _mf.prod_point_offset_km(zen_deg, azd, h_prod_km)
            zen_u = np.asarray(rc_family["zen"], float)
            az_u = np.asarray(rc_family["az"], float)
            rc_u = _mf.interp_site_rc(rc_family, dn, de)
            frame = _mf.prod_frame_exact(cos_theta, azd, h_prod_km, site_lat)
        else:
            zen_u, az_u, rc_u = zen_f, az_f, rc_f
            frame = (prod_point_frame(cos_theta, azd, h_prod_km)
                     if prod_frame else None)

        def _rc(v, _z=zen_u, _a=az_u, _r=rc_u, _f=frame):
            if _f is not None:
                th, ph = local_angles(v, _f)
            else:
                th = np.degrees(np.arccos(np.clip(v[..., 2], -1, 1)))
                ph = np.degrees(np.arctan2(v[..., 1], v[..., 0]))
            return rc_bilinear(_z, _a, _r, th, ph)

        rc0 = _rc(npv)
        Gab = {s: _interp_G(rc0, rc_grid, G[s]).reshape(len(alpha), len(beta), nE)
               for s in G}
        Gsh = None
        has_mu = (chan is not None) or (W_mu is not None)
        if has_mu and d_cone is not None:
            pp = npv + d_cone
            pp /= np.linalg.norm(pp, axis=-1, keepdims=True)
            pm = npv - d_cone
            pm /= np.linalg.norm(pm, axis=-1, keepdims=True)
            Gp = {s: _interp_G(_rc(pp), rc_grid, G[s]).reshape(
                len(alpha), len(beta), nE) for s in G}
            Gm = {s: _interp_G(_rc(pm), rc_grid, G[s]).reshape(
                len(alpha), len(beta), nE) for s in G}
            Gsh = {s: (Gp[s] if s in mu_plus else Gm[s]) for s in G}

        # channel integrals; <.>_cone = sum_alpha W[E,a] mean_beta(.)
        if chan is None:
            for s in G:
                j_pi = np.einsum("ea,ae->e", W_pi, (q_pi * Gab[s]).mean(axis=1))
                j_k = np.einsum("ea,ae->e", W_k, (q_k * Gab[s]).mean(axis=1))
                if W_mu is None or fmu is None:
                    J = j_pi + j_k
                else:
                    g_mu = Gsh[s] if Gsh is not None else Gab[s]
                    j_mu = np.einsum("ea,ae->e", W_mu, (q_pi * g_mu).mean(axis=1))
                    f = np.asarray(fmu[s], float)
                    J = (1.0 - f) * j_pi + f * j_mu + j_k
                out[s][ia] = J / p_axis
                if renorm is not None:
                    out[s][ia] *= renorm[s]
            e_pi = np.einsum("ea,ae->e", W_pi, q_pi.mean(axis=1))
            e_k = np.einsum("ea,ae->e", W_k, q_k.mean(axis=1))
            eoff_eq[ia] = (e_pi + e_k) / p_axis
            blocked[ia] = np.einsum("ea,ae->e", W_pi,
                                    (q_pi <= 0.0).mean(axis=1))
        else:
            for s in G:
                qd, qk, qm = qs[idx[s]], qs[idx[s] + 1], qs[idx[s] + 2]
                g_mu = Gsh[s] if Gsh is not None else Gab[s]
                J = (np.einsum("ea,ae->e", W_pi, (qd * Gab[s]).mean(axis=1))
                     + np.einsum("ea,ae->e", W_k, (qk * Gab[s]).mean(axis=1))
                     + np.einsum("ea,ae->e", W_mu_s[s], (qm * g_mu).mean(axis=1)))
                out[s][ia] = J / p_axis_s[s]
                if renorm is not None:
                    out[s][ia] *= renorm[s]
                eoff_eq[ia] = (
                    np.einsum("ea,ae->e", W_pi, qd.mean(axis=1))
                    + np.einsum("ea,ae->e", W_k, qk.mean(axis=1))
                    + np.einsum("ea,ae->e", W_mu_s[s], qm.mean(axis=1))
                ) / p_axis_s[s]  # last species wins; per-species via G == 1
            s0 = sp_order[0]
            blocked[ia] = np.einsum("ea,ae->e", W_pi,
                                    (qs[idx[s0]] <= 0.0).mean(axis=1))
    return dict(e=ep_grid, F=out, E_off_equiv=eoff_eq, p_axis=p_axis,
                w_blocked=blocked)


#: species order of ``offaxis_mc.CHANNEL_SPECIES`` mapped to engine species names
CHANNEL_SPECIES_MAP = {
    "total_numu": "numu", "total_antinumu": "antinumu",
    "total_nue": "nue", "total_antinue": "antinue",
}


# ---------------------------------------------------------------------------
# Liouville constraint (a CONSTRAINT, not a derivation)
# ---------------------------------------------------------------------------
def sky_scan(prod, widths, czs, rc_grid=None, G=None, **kw):
    """``(czs, {species: F[n_cz, nE]})`` at ``G == 1`` -- the production-only
    joint factor over the down-going hemisphere, used by
    :func:`liouville_renorm`.  ``kw`` is forwarded to
    :func:`delivered_joint_factor` (``sigma_scale``, ``n_alpha``, ...).
    """
    ep = prod["ep_grid"]
    chan = {s: prod["chan"][CHANNEL_SPECIES_MAP[s]] for s in SPECIES}
    smu = {s: widths["mu_nue" if "nue" in s else "mu_numu"] for s in SPECIES}
    if rc_grid is None:
        rc_grid = np.linspace(0.1, RC_MAX_GV, 8)
    if G is None:
        G = {s: np.ones((len(rc_grid), len(ep))) for s in SPECIES}
    flat = (np.array([0.0, 45.0, 89.0]), np.array([0.0, 180.0, 360.0]),
            np.ones((3, 3)))
    czs = np.asarray(czs, float)
    out = {s: np.zeros((len(czs), len(ep))) for s in SPECIES}
    q_cache = {}
    for i, c in enumerate(czs):
        r = delivered_joint_factor(
            float(c), [0.0], prod, flat, rc_grid, G,
            sigma_pi=widths["pi"], sigma_k=widths["k"],
            channels_by_species=chan, sigma_mu_by_species=smu,
            q_cache=q_cache, **kw)
        for s in SPECIES:
            out[s][i] = r["F"][s][0]
    return czs, out


def liouville_renorm(prod, widths, czs=None, measure="cos", target=1.0, **kw):
    """``{species: g[nE]}`` forcing a chosen sky average of ``F`` to ``target``.

    ``measure="cos"``    ``Int F cos psi dOmega / Int cos psi dOmega`` -- the
                         neutrinos that actually cross a horizontal unit area,
                         i.e. the average a flux-conservation argument
                         constrains.  In a FLAT atmosphere the exact kernel gives
                         ``1 - f_esc`` here, with ``f_esc`` the fraction emitted
                         into the upper hemisphere by near-horizon primaries
                         (4.4% for nu_mu at 0.3 GeV), so ``target = 1`` already
                         over-constrains; pass the measured flat value if you
                         want to keep that physics.
    ``measure="omega"``  the plain solid-angle average the paper quotes.  This is
                         NOT conserved even for the exact kernel: in a flat
                         atmosphere it is 1.15 at 0.3 GeV and grows without bound
                         as the horizon bin is refined (``F ~ 1/cos psi``).

    This is a constraint imposed on the answer, not a correction derived from the
    physics: it is offered so that the size of the renormalisation can be
    reported, and because it is a pure function of energy it cannot change any
    zenith or azimuth ratio.
    """
    czs = np.round(np.arange(0.05, 1.0, 0.1), 2) if czs is None else np.asarray(czs)
    czs, F = sky_scan(prod, widths, czs, **kw)
    w = (czs / czs.sum()) if measure == "cos" else np.full(len(czs), 1.0 / len(czs))
    return {s: target / np.maximum(w @ F[s], 1e-300) for s in SPECIES}
