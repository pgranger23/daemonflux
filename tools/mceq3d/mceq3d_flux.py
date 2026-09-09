"""mceq3d_flux -- a trustable, absolute, directional atmospheric-neutrino flux.

This is the production engine: it returns the **absolute** flux
``Phi(E, cos zenith, azimuth)`` for all four species (numu, antinumu, nue,
antinue) at a detector site, down to ~0.5 GeV, built so that every piece comes
from a *trusted* source and the 3D corrections are *validated against Honda*.

Construction (each factor trusted / validated)
---------------------------------------------
``Phi_3D = Phi_base(E,|cosZ|,s) * E_off(E,cosZ) * G_s(E,R_c(cosZ,az)) * S(E)``

* **Phi_base** -- MCEq (or daemonflux) solved per zenith with the **curved
  atmosphere**: absolute normalization, flavour/charge content, spectra, and the
  sec(theta) horizon enhancement (1D-per-direction-with-curvature).
* **E_off(E, cosZ)** -- the **first-principles off-axis 3D-production factor**
  ``Phi_3D/Phi_1D`` (:meth:`offaxis_factor`, ``offaxis=True``): the complete
  genuine-3D/1D ratio that both redistributes flux in zenith and produces the
  sub-GeV near-horizon excess (horizon/vertical ~1.8 at 0.3 GeV). Built in
  `offaxis_mc.py` from MCEq per-parent depth-resolved production (pi-geometry +
  kaon term), the curved-atmosphere slant geometry, and the **sampled** angular
  distribution from the generator's (x_L, theta) pion kernel folded with exact
  decay (Gaussian sigma_K for the kaon term). **No reference flux is used**;
  validated to reproduce Honda and Bartol (numu and nue) to their mutual ~5-15%.
  ->1 at high E and at the vertical. **On by default** since 2026-09-04
  (``offaxis=True``); pass ``offaxis=False`` for the fast geomagnetic-only path.
* **G_s(E, R_c)** -- the geomagnetic suppression = ``MCEq(primary cut at R_c) /
  MCEq(full)``, the **cascade-correct** response to removing sub-cutoff primaries
  (NO ``x_eff`` hack); ~zenith-independent, interpolated on a small R_c grid.
* **R_c(cosZ, az)** -- the first-principles **back-traced full-IGRF cutoff**
  (:mod:`geomag_backtrace`, validated: Kamioka 11.3 GV).
* **S(E)** -- optional solar-modulation factor (:meth:`solar_factor`).

With ``offaxis=True`` this is the complete first-principles 3D construction (curved
per-zenith cascade + off-axis 3D production + geomagnetic + solar), validated
absolutely against Honda/Bartol, and it is what a bare ``solve()`` now returns;
``offaxis=False`` drops E_off (the fast factorised path, ~1.8x lower near the
horizon sub-GeV, and it also switches off the joint cone). The legacy ``full_3d``
option applies only the flux-conserving redistribution R (:meth:`angular_factor`)
and is superseded by ``offaxis`` (do not combine them).

The rigidity cut is applied per-nucleus to the primary nucleons: the proton flux
is split into **free protons** (A/Z=1, R=E) and **bound protons** (in nuclei,
A/Z~2, R~2E) via MCEq's own p,n fluxes (free p = p-n by isospin), and neutrons are
all bound (R~2E).

The 1D base can be raw MCEq (default) or daemonflux's **muon-calibrated** flux
(``base_model="daemonflux"``) for a data-anchored absolute normalization.

Validation (`validate_honda.py`, and ``--validate`` here): the absolute Phi(E,
cosZ, az) for numu and nue matches the Honda HKKM2014 Kamioka tables in
normalization, zenith (sec theta) and azimuth (East-West).

Full sky
--------
* **Down-going (cosZ >= 0):** detector geomagnetic cutoff.
* **Up-going (cosZ < 0):** the production is up/down symmetric (same atmosphere at
  |cosZ|), but the geomagnetic cutoff is evaluated at the **far-side production
  point** (:func:`farside_production`) -- a global geomagnetic treatment, so the
  up-going hemisphere is included (essential for oscillation analyses).

Trust boundary (documented, not hidden)
---------------------------------------
* Up-going uses a single representative far-side production point per direction
  (the production region has finite extent; this is the leading geometric term).
* Nuclei: per-nucleus rigidity via the free/bound split (above); the bound part
  uses <A/Z>=2.0 (Fe is 2.08, a ~1% sub-component).
* Inter-direction streaming residual (~1-2%) and muon-bending E-W (charge-split,
  ~3 deg sub-GeV) are separate small corrections (see `spherical_streaming`,
  `muon_bending`).

Run::

    python mceq3d_flux.py --validate --plot
"""

from __future__ import annotations

import argparse
import hashlib
import os

import numpy as np

# Site-independent G_s and per-site cutoff maps are cached here (see solve(use_cache)).
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flux_cache")

CM2_PER_M2 = 1.0e4  # MCEq flux is per cm^2; Honda/this engine report per m^2
RE_KM = 6371.0  # mean Earth radius [km] (production-point displacement geometry)
# Rigidity ceiling for the back-traced cutoff scan and the G_s(R_c) grid.
# MUST exceed the true maximum cutoff over the sky, or the map SATURATES: with
# the previous 40 GV value the six near-horizon East cells at Kamioka were all
# pinned at exactly 40.00 GV, while direct back-tracing gives 40.5-49.5 GV there
# (peak at zenith 89 deg, azimuth 75 deg). A saturated map has ZERO azimuthal
# gradient, which silently destroys every effect that depends on sampling the
# cutoff at a shifted direction -- including the charge-signed muon-bending
# East-West asymmetry.
#
# Do NOT set this far above the true maximum: rigidities ABOVE a direction's
# cutoff are ALLOWED and their trajectories escape to r_escape (~1200 RK4 steps),
# whereas forbidden ones return in ~28, so every GV of excess headroom is paid
# for in expensive traces. 55 GV gives ~5.5 GV of margin over the measured
# Kamioka maximum at ~20% extra cost; 80 GV cost >60% and did not finish in
# 2+ hours. `finemap_rc` warns if any cell still saturates, so a site needing
# more will say so rather than silently flat-lining.
RC_MAX_GV = 55.0
# Effective neutrino production altitude [km] for the production-point
# displacement in cone_geff. The MCEq depth-resolved production weight along a
# near-horizon slant path peaks at h ~ 24-32 km (diag_ew_prodregion.py), well
# above the ~15-20 km often quoted for the vertical, because the near-horizon
# ray traverses the upper atmosphere at grazing incidence.
H_PROD_KM = 30.0
# ---------------------------------------------------------------------------
# Penumbra width sigma in ln R of the rigidity-cutoff transmission, Eq. (7).
# ---------------------------------------------------------------------------
# MEASURED, not tuned.  The back-traced admittance A(R) of a FIXED direction is
# binary; what an erf of width sigma is meant to represent is the penumbral band
# [R_L, R_U] in which allowed and forbidden rigidities alternate.  A 0.1 GV
# ladder at Kamioka (2020-01-01, 8 directions, ~1000 trajectories;
# ``diag_penumbra_width.py``) measures that band directly -- the allowed
# rigidity width below R_U, and the sigma whose erf opens the same width:
#
#   direction   R_U [GV]   R_L [GV]   allowed width < R_U   equivalent sigma
#   vertical      11.45       9.55        0.70 GV (6.1%)        0.17
#   60 E          23.56      23.66        0.00                  < 0.011
#   75 E          31.59      31.69        0.00                  < 0.008
#   81 E          36.01      36.11        0.00                  < 0.007
#   87 E          41.64      41.74        0.00                  < 0.006
#   87 N          26.73      26.83        0.00                  < 0.009
#   87 S           9.80       9.30        0.10 GV (1.0%)        0.026
#   87 W           7.41       7.11        0.20 GV (2.7%)        0.071
#
# In every East / high-cutoff direction -- the ones that make the East-West
# asymmetry -- R_L > R_U: the lowest allowed rigidity lies ABOVE the highest
# forbidden one, i.e. the transition is a sharp step with no penumbral
# structure (confirming commit 5a5ef11 at 2.5x finer resolution).  Only the
# LOW-cutoff directions show any band, and there it is a couple of narrow
# allowed islands -- a small downward shift of the effective cutoff rather than
# a smooth width (adopting that shift as R_eff was measured and refused in
# 5a5ef11: it would lower the already-correct vertical cutoff).
#
# The delivered value is therefore the sharp step, sigma = 0.  It is exact for
# the five high-cutoff directions and, because ``_transmission`` averages the
# step over the primary energy bin, it is not infinitely sharp in practice: the
# grid imposes a ramp of width dlnE ~ 0.115, i.e. an RMS of 0.033 -- which is
# itself at the level of the widths measured at 87 S / 87 W.  Nothing narrower
# is representable on the MCEq grid, and nothing wider is supported by the
# back-tracer.
#
# The legacy value was 0.5 -- hand-set (inherited from the daemonflux package's
# analytic model, where it also stood in for the primary-energy spread that
# G_s now folds in exactly), never derived, and far too wide: it passes 8% of
# the primaries at R_c/2 and 21% at R_c/1.5, opening 6.3 GV of allowed
# rigidity below the 41.6 GV East cutoff where the back-tracer measures none.
# Against the steep primary spectrum that tail carried ~40% of the surviving
# East flux at 0.5 GeV.
SIGMA_LNR = 0.0
SPECIES = ("total_numu", "total_antinumu", "total_nue", "total_antinue")
#: default E_off table read by :meth:`MCEq3DFlux.offaxis_factor` -- the
#: species-resolved, channel-weighted build on the arcsin-corrected v2 moments
#: (``offaxis_mc.build_channel``).  Changed 2026-09-04 from the flat pion-only
#: July build, which stays reachable as :data:`EOFF_TABLE_FLAT`.
EOFF_TABLE = "offaxis_excess_channel_v2.npz"
#: the original flavour-blind single-pion-cone table (A/B only)
EOFF_TABLE_FLAT = "offaxis_excess.npz"
SP_LABEL = {
    "total_numu": "numu",
    "total_antinumu": "antinumu",
    "total_nue": "nue",
    "total_antinue": "antinue",
}


def _transmission(e_grid, rc_gv, sigma_lnr=None, az_over_z=1.0, dlne=None):
    """Rigidity-cutoff transmission at ``R = az_over_z * E``.

    Eq. (7) of the paper: an erf step of width ``sigma_lnr`` in ``ln R``,
    **bin-averaged** over the (log-uniform) primary energy grid,

        T(E; R_c) = <1/2 [1 + erf((ln R - ln R_c) / (sqrt2 sigma))]>_bin ,

    with the average taken analytically over the bin
    ``[ln E - dlnE/2, ln E + dlnE/2]``.  ``dlne`` defaults to the grid's own
    spacing (:func:`_grid_dlne`); pass ``0`` for the pure point-sampled erf.

    Why the bin average.  ``MCEq._phi0`` holds *bin-averaged* primary fluxes, so
    the exact representation of a cut falling inside a bin is that bin's overlap
    fraction, not the erf sampled at the bin centre.  This matters only in the
    sharp limit: it keeps ``sigma_lnr -> 0`` (a hard rigidity step, which is what
    the back-tracer measures in the East -- see ``diag_penumbra_width``/commit 5a5ef11)
    numerically well behaved and, crucially, **continuous in R_c**.  Sampling a
    hard step at the grid points would make ``G_s(R_c)`` a staircase that only
    moves when ``R_c`` crosses a bin edge (``dlnR = dlnE ~ 0.115``, i.e. ~4.6 GV
    at 40 GV), destroying exactly the near-limb response the East-West asymmetry
    is made of.  At the legacy ``sigma_lnr = 0.5`` the bin average shifts T by
    ``<= 6e-4`` absolute, so the delivered baseline is unchanged.

    ``sigma_lnr=None`` uses the module default :data:`SIGMA_LNR`.
    """
    from scipy.special import erf

    if sigma_lnr is None:
        sigma_lnr = SIGMA_LNR
    s = float(sigma_lnr)
    e = np.maximum(np.asarray(e_grid, float), 1e-9)
    x = np.log(az_over_z * e) - np.log(float(rc_gv))  # ln(R/R_c)
    w = _grid_dlne(e) if dlne is None else np.asarray(dlne, float)
    if np.all(w <= 0):  # no bin average: point-sampled erf (or a hard step)
        if s <= 0:
            return (x > 0).astype(float)
        return 0.5 * (1.0 + erf(x / (np.sqrt(2) * s)))
    a, b = x - 0.5 * w, x + 0.5 * w
    if s <= 0:  # exact hard step, bin-averaged -> overlap fraction
        return np.clip(b / w, 0.0, 1.0)

    def prim(t):  # antiderivative of erf(t/(sqrt2 s))
        return t * erf(t / (np.sqrt(2) * s)) + s * np.sqrt(2 / np.pi) * np.exp(
            -t * t / (2 * s * s)
        )

    return np.clip(0.5 + (prim(b) - prim(a)) / (2.0 * w), 0.0, 1.0)


def gs_cache_name(tag, cz_ref, rc_grid, sigma_lnr):
    """File name of the ``geomag_response`` disk cache entry.

    The key is the model identity ``tag`` (interaction model / primary /
    atmosphere / e_min), the reference zenith, the rigidity grid **and the
    penumbra width** -- a response built with a different width is a different
    physical object and must never be silently reused.
    """
    h = hashlib.md5(
        f"{tag}|{float(cz_ref):.4f}|{np.asarray(rc_grid, float).tobytes()}"
        f"|sig{float(sigma_lnr):.6g}".encode()
    ).hexdigest()[:16]
    return f"gs_{h}.npz"


def _grid_dlne(e_grid):
    """Bin width ``dlnE`` of a (log-uniform) energy grid, as an array."""
    le = np.log(np.asarray(e_grid, float))
    if le.size < 2:
        return np.zeros_like(le)
    d = np.diff(le)
    return np.concatenate([[d[0]], 0.5 * (d[:-1] + d[1:]), [d[-1]]])


def farside_production(lat, lon, cos_zenith, azimuth, h_prod_km=20.0):
    """Far-side production point for an UP-going arrival (cos_zenith < 0).

    Up-going neutrinos are produced on the *opposite* side of the Earth and travel
    straight through it. The atmospheric production is up/down symmetric (same
    atmosphere at |cosZ|), but the **geomagnetic cutoff is set at the far-side
    production point**, not at the detector. This returns that point's
    ``(lat, lon)`` and the local primary arrival ``(cosZ, azimuth)`` there, so the
    detector's up-going geomagnetics can be evaluated globally.
    """
    from geomag_backtrace import _local_frame, arrival_direction, RE

    up = _local_frame(lat, lon)[0]
    P = RE * up
    d = arrival_direction(
        lat, lon, np.degrees(np.arccos(np.clip(cos_zenith, -1, 1))), azimuth
    )  # neutrino velocity (up-going for cosZ<0)
    Rh = RE + h_prod_km * 1e3
    Pd = P @ d
    s = Pd + np.sqrt(max(Pd**2 + (Rh**2 - RE**2), 0.0))  # far-side intersection
    Q = P - s * d
    qhat = Q / np.linalg.norm(Q)
    lat_q = np.degrees(np.arcsin(np.clip(qhat[2], -1, 1)))
    lon_q = np.degrees(np.arctan2(qhat[1], qhat[0]))
    upq, northq, eastq = _local_frame(lat_q, lon_q)
    src = -d  # primary source direction at Q
    cosz_q = float(src @ upq)
    horiz = src - cosz_q * upq
    az_q = np.degrees(np.arctan2(horiz @ eastq, horiz @ northq)) % 360.0
    return lat_q, lon_q, cosz_q, az_q


# Zenith nodes packed into the last ~10 deg before the limb.  R_c(zenith) is
# nearly flat to ~60 deg and then rises convexly (~1.5 GV/deg in the East above
# 84 deg at Kamioka), so a uniform grid's interpolation error is concentrated
# entirely here; see MCEq3DFlux.finemap_rc.
LIMB_ZENITH_NODES = (82.0, 84.0, 85.5, 87.0, 88.0, 88.5, 89.0, 89.5)


def _gauss_cone(alpha_deg, sigma_deg, per_axis=True):
    """Gaussian-on-the-sphere cone weights ``W[nE, n_alpha]``.

    ``sin(alpha) exp(-alpha^2 / (2 s1^2))`` with the **per-axis** width
    ``s1 = sigma / sqrt(2)`` of a space-angle RMS ``sigma`` -- byte-for-byte the
    convention of ``offaxis_mc.cone_numden`` (and of
    ``joint_cone.gauss_alpha_weights``), so ``E_off`` and the geomagnetic cone
    finally use ONE definition of the production angle.  ``per_axis=False``
    reproduces the legacy ``s1 = sigma`` cone -- 41% too wide -- and is kept only
    for the ``cone_kernel="legacy"`` A/B.
    """
    a = np.deg2rad(np.asarray(alpha_deg, float))
    s1 = np.deg2rad(np.maximum(np.asarray(sigma_deg, float), 1e-3))
    if per_axis:
        s1 = s1 / np.sqrt(2.0)
    return np.sin(a)[None, :] * np.exp(-(a[None, :] ** 2) / (2.0 * s1[:, None] ** 2))


def _zenith_nodes(n_zen=13, limb_nodes=True):
    """Down-going zenith nodes [deg] of the cutoff map: uniform to 80 deg plus
    (``limb_nodes``) the dense limb set.  ``limb_nodes=False`` reproduces the
    legacy uniform ``linspace(0, 89, n_zen)`` grid."""
    if not limb_nodes:
        return np.linspace(0.0, 89.0, n_zen)
    return np.unique(
        np.concatenate([np.linspace(0.0, 80.0, n_zen), np.array(LIMB_ZENITH_NODES)])
    )


def farside_cutoff_map(
    lat, lon, date, zeniths_deg, azimuths, n_scan=None, r_lo=0.5, r_hi=RC_MAX_GV,
    m_hat=None, n_jobs=None, **kw
):
    """Far-side (up-going) full-IGRF cutoff [GV] on a (zenith x azimuth) grid.

    For each up-going zenith (deg, >90) the parent primary is produced on the
    *opposite* limb and the neutrino travels straight through the Earth, so the
    cutoff is a trajectory back-trace from the far-side production point ``Q`` with
    ``u0 = -d`` (the neutrino direction) -- the global geomagnetic treatment used
    for genuinely up-going arrivals in :meth:`MCEq3DFlux.cutoff_grid` and by
    :func:`farside_production`. Factored out so the up-going hemisphere of the
    production-cone map (:meth:`MCEq3DFlux.finemap_rc`) and the solve-grid cutoff
    share one implementation.

    Uses the same coarse-ladder + bisection rigidity scan as the detector map
    (:func:`geomag_backtrace.scan_upper_cutoff`), so both hemispheres carry the
    same ~0.05 GV rigidity resolution and neither can land in a penumbral allowed
    island.

    NOTE (2026-09): this map is now used only for **genuinely up-going neutrino
    directions**.  The sub-limb part of a near-horizon *down-going* production
    cone is no longer read off it -- that stitch produced a 9.5 GV discontinuity
    at 89/90 deg; see :meth:`MCEq3DFlux.cone_geff` (``sublimb``).
    """
    import geomag_backtrace as gb

    r0, u0 = [], []
    for zdeg in zeniths_deg:
        cz = float(np.cos(np.radians(zdeg)))
        for az in azimuths:
            latq, lonq, _, _ = farside_production(lat, lon, cz, az)
            r0.append((gb.RE + 20e3) * gb._local_frame(latq, lonq)[0])
            u0.append(-gb.arrival_direction(lat, lon, zdeg, az))  # -neutrino dir
    coarse = gb.COARSE_STEP_GV
    if n_scan is not None:
        coarse = min(coarse, (r_hi - r_lo) / max(int(n_scan) - 1, 1))
    rc, _sat = gb.cutoff_from_states(
        np.array(r0), np.array(u0), date, m_hat, r_lo=r_lo, r_hi=r_hi,
        coarse_step=coarse, n_jobs=n_jobs, **kw
    )
    return rc.reshape(len(zeniths_deg), len(azimuths))


# ---------------------------------------------------------------------------
# Production-point-ANCHORED cutoff  (``cutoff_anchor``)
# ---------------------------------------------------------------------------
# The cutoff map of :func:`geomag_backtrace.cutoff_map` is anchored at the
# DETECTOR: every trajectory is launched from ``RE * up`` at ``(lat, lon)``.
# The primary that makes a near-horizon neutrino does not enter the atmosphere
# there -- it enters it at the production point P, ``L = 370 km`` up the arrival
# ray at zenith 87 deg (619 km at the exact horizon) for ``h_prod = 30 km``.
# Over that distance the *site* moves by ``d_ground / R_E`` = 3.3 deg
# (5.6 deg at 90 deg, 5.1 deg at 89.5 deg) in
# geomagnetic latitude, which changes the cutoff by ~12-16% and rotates the
# azimuthal pattern by ~11 deg (``diag_dipole_phase.py --sections prodpoint``):
#
#   zen 87, Kamioka   N      E      S      W    N/S   1st-harm
#   detector       23.78  42.99  10.26   7.60   2.32     61.7
#   prod_point     19.91  37.13  12.10   7.95   1.65     72.4
#
# ``cone_geff(sublimb="prod_point")`` / ``joint_cone.delivered_joint_factor
# (prod_frame=True)`` rotate the local VERTICAL to P and then read the
# detector-anchored map, i.e. they model the frame rotation but not the site
# displacement.  ``cutoff_anchor="prod_point"`` supplies the missing half: the
# cone samples of an arrival direction are read off a cutoff map built at P
# itself, interpolated from a small family of maps at displaced sites.
#
#: Default launch anchor for the cone samples' cutoff: ``"detector"`` (the
#: pre-2026-09-04 behaviour) or ``"prod_point"``.  ``solve(cutoff_anchor=None)``
#: resolves to this, so an A/B can be driven either per call or module-wide.
#:
#: MEASURED A/B (2026-09-05, Kamioka, delivered engine, bare ``solve`` defaults
#: plus the anchor; ``scratchpad/anchor/ab_tables.txt``).  ``"prod_point"`` is
#: the physically correct treatment -- it is what Honda does, back-tracing each
#: primary from its own injection point -- and it removes essentially the whole
#: azimuthal-phase error against Honda: the first-harmonic phase offset at
#: cosZ 0.05 / 0.5 GeV goes from -12.7 / -11.8 / -12.8 / -14.7 deg to
#: +1.3 / +1.3 / +0.5 / -0.2 deg for nu_mu / anti-nu_mu / nu_e / anti-nu_e, and
#: the North/South model-over-Honda pair from 0.87 / 1.07 to 0.95 / 0.91
#: (nu_mu, 0.5 GeV).  The grid-wide 90th-percentile |log10(ours/Honda)| drops
#: 0.067 -> 0.061, 0.086 -> 0.076, 0.101 -> 0.090, 0.066 -> 0.061.  W/E, H/V and
#: the absolute normalisation move by <=1.5%.
#:
#: It is NOT the default yet, for two reasons that are engineering, not physics:
#: (a) a new site/date costs one 4.1 h back-trace family (:meth:`prod_family_rc`)
#: before the first flux comes out; (b) the bilinear site interpolation has
#: measured 8-22% outliers in the low-cutoff west/south-west cells (the median
#: cell is ~1%; see ``diag_cutoff_anchor.py`` section 3 and
#: ``test_cutoff_anchor.py``), which an ``n_side=5`` family would quarter.
#: The default was flipped to "prod_point" on 2026-09-05 (project decision): the
#: anchor is the physically correct treatment and the family build is a one-time
#: per-site precompute like the map itself; "detector" remains the fast A/B path.
CUTOFF_ANCHOR = "prod_point"  # flipped 2026-09-05: physically correct; the Kamioka family is cached
#: Scheme tag of the displaced-site map family; part of every cache key, so a
#: family built with a different node layout / launch altitude is never reused.
PROD_FAMILY_SCHEME = "cartNxN-bandseed-v1"


def prod_point_offset_km(zen_deg, az_deg, h_prod_km=H_PROD_KM):
    """Ground displacement ``(d_north, d_east)`` [km] of the production point.

    P is the point at altitude ``h_prod_km`` on the arrival ray of a neutrino
    seen at ``(zen_deg, az_deg)``; the displacement is the great-circle offset
    of P's ground projection from the detector, which lies at bearing
    ``az_deg`` (the arrival azimuth -- the neutrino comes FROM there, so its
    parent entered the atmosphere on that side).

    Exactly zero at the vertical and monotone in zenith: 50 km at 60 deg,
    158 km at 80 deg, 368 km at 87 deg, 564 km at 89.5 deg and 617 km at the
    exact horizon, for h = 30 km.
    """
    th = np.radians(np.asarray(zen_deg, float))
    cz, sz = np.cos(th), np.sin(th)
    r = RE_KM + h_prod_km
    L = -RE_KM * cz + np.sqrt(np.maximum((RE_KM * cz) ** 2 + r * r - RE_KM**2, 0.0))
    delta = np.arctan2(L * sz, RE_KM + L * cz)  # angle P-Earth centre-detector
    d = RE_KM * delta
    a = np.radians(np.asarray(az_deg, float))
    return d * np.cos(a), d * np.sin(a)


def prod_site_latlon(lat_deg, lon_deg, dn_km, de_km):
    """Geographic ``(lat, lon)`` of the site displaced by ``(dn, de)`` km.

    Great-circle displacement on the geocentric sphere of radius
    :data:`RE_KM` -- the same sphere ``geomag_backtrace`` works on.
    """
    import geomag_backtrace as gb

    up, north, east = gb._local_frame(lat_deg, lon_deg)
    d = float(np.hypot(dn_km, de_km))
    if d < 1e-9:
        return float(lat_deg), float(lon_deg)
    hat = (dn_km * north + de_km * east) / d
    v = np.cos(d / RE_KM) * up + np.sin(d / RE_KM) * hat
    return (float(np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0)))),
            float(np.degrees(np.arctan2(v[1], v[0]))))


def prod_frame_exact(cos_theta, az_deg, h_prod_km, lat_deg):
    """``(up_P, north_P)`` at the production point, in the DETECTOR's (N,E,U) frame.

    The TRUE local frame at P, i.e. the one a cutoff map built at
    :func:`prod_site_latlon` is indexed in.  It differs from the approximate
    frame of ``cone_geff._prod_frame`` / ``joint_cone.prod_point_frame``, which
    take "north at P" to be the detector's north projected orthogonal to
    ``up_P``: that ignores the CONVERGENCE OF THE MERIDIANS, an azimuthal
    rotation of ``dlon * sin(lat)`` = up to 4.0 deg for a 610 km east-west
    displacement at Kamioka.  That approximation is harmless while the map is
    detector-anchored (its azimuth axis is the detector's anyway), but it must
    not be used to index a map built AT P.
    """
    th = np.arccos(np.clip(float(cos_theta), -1.0, 1.0))
    phi = np.radians(float(az_deg))
    n = np.array([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)])
    r = RE_KM + h_prod_km
    L = -RE_KM * n[2] + np.sqrt(max((RE_KM * n[2]) ** 2 + r * r - RE_KM**2, 0.0))
    P = np.array([L * n[0], L * n[1], RE_KM + L * n[2]])
    up_p = P / np.linalg.norm(P)
    la = np.radians(float(lat_deg))
    z_axis = np.array([np.cos(la), 0.0, np.sin(la)])  # Earth's spin axis in (N,E,U)
    north_p = z_axis - np.dot(z_axis, up_p) * up_p
    nn = np.linalg.norm(north_p)
    if nn < 1e-9:  # geographic pole: azimuth is degenerate anyway
        return up_p, np.array([1.0, 0.0, 0.0])
    return up_p, north_p / nn


def _banded_cutoff_scan(gb, r0, u0, date, seed, r_lo, r_hi, n_jobs, r_floor,
                        n_bands=16, seed_margin=0.45, coarse_step=None):
    """Back-traced cutoffs with a per-band rigidity window seeded by ``seed``.

    ``seed[i]`` is a prior estimate of direction ``i``'s cutoff (for the
    displaced-site family: the same direction's DETECTOR cutoff, which is within
    ~20% of it).  Directions are sorted by the seed and split into ``n_bands``
    equal-count groups; group ``k`` is scanned over
    ``[(1-m) min_seed_k, (1+m) max_seed_k]`` clipped to ``[r_lo, r_hi]``.

    The scan itself is unchanged -- :func:`geomag_backtrace.cutoff_from_states`
    with the same coarse ladder and bisection -- so the only way a band could
    bias a result is by starting below the true cutoff or ending above it.  Both
    show up as a result sitting at a band edge, and those directions are
    re-scanned over the FULL ``[r_lo, r_hi]`` range before returning.

    ``seed=None`` (or ``n_bands=0``) is the plain full-range scan.
    """
    n = len(r0)
    kw = dict(r_floor=r_floor)
    if coarse_step is not None:
        kw["coarse_step"] = coarse_step
    if seed is None or not n_bands or n == 0:
        return gb.cutoff_from_states(r0, u0, date, r_lo=r_lo, r_hi=r_hi,
                                     n_jobs=n_jobs, **kw)
    order = np.argsort(np.asarray(seed, float))
    groups = np.array_split(order, min(int(n_bands), n))
    rc = np.zeros(n)
    sat = np.zeros(n, bool)
    edge = np.zeros(n, bool)
    for g in groups:
        if not len(g):
            continue
        lo = max(r_lo, (1.0 - seed_margin) * float(np.min(seed[g])))
        hi = min(r_hi, (1.0 + seed_margin) * float(np.max(seed[g])))
        if hi <= lo + 2.0:  # degenerate window -> full range
            lo, hi = r_lo, r_hi
        rcg, satg = gb.cutoff_from_states(r0[g], u0[g], date, r_lo=lo, r_hi=hi,
                                          n_jobs=n_jobs, **kw)
        rc[g], sat[g] = rcg, satg
        # a cutoff at (or within one coarse step of) a band edge is not trusted
        edge[g] = (rcg <= lo + gb.COARSE_STEP_GV) | (rcg >= hi - gb.COARSE_STEP_GV)
    bad = np.where(edge)[0]
    if len(bad):
        rcb, satb = gb.cutoff_from_states(r0[bad], u0[bad], date, r_lo=r_lo,
                                          r_hi=r_hi, n_jobs=n_jobs, **kw)
        rc[bad], sat[bad] = rcb, satb
    return rc, sat


def interp_site_rc(family, dn_km, de_km):
    """Cutoff map ``rc[n_zen, n_az]`` at an arbitrary displacement, bilinear in
    the site offset ``(d_north, d_east)`` [km].

    The DIRECTION axes are never interpolated here (they keep the family's own
    fine zenith/azimuth nodes); only the *site* is, and the site enters
    ``R_c`` smoothly -- ``R_c ~ cos^4(lambda_mag)`` with
    ``d ln R_c / d lambda ~ -0.034/deg`` at Kamioka -- so a node spacing equal
    to the whole displacement range costs ``|f''| D^2 / 8`` ~ 0.6%.
    """
    off = np.asarray(family["off"], float)
    rc = np.asarray(family["rc"], float)
    x = np.clip(float(dn_km), off[0], off[-1])
    y = np.clip(float(de_km), off[0], off[-1])
    i = int(np.clip(np.searchsorted(off, x) - 1, 0, len(off) - 2))
    j = int(np.clip(np.searchsorted(off, y) - 1, 0, len(off) - 2))
    tx = (x - off[i]) / (off[i + 1] - off[i])
    ty = (y - off[j]) / (off[j + 1] - off[j])
    return (rc[i, j] * (1 - tx) * (1 - ty) + rc[i + 1, j] * tx * (1 - ty)
            + rc[i, j + 1] * (1 - tx) * ty + rc[i + 1, j + 1] * tx * ty)


def _nucleon_split(mceq, pm, e_grid):
    """Per-nucleus rigidity bookkeeping of the primary nucleon flux.

    Returns ``(phi0_std, p_sl, n_sl, f_free, az_bound)`` for an ``MCEqRun``
    ``mceq`` whose primary model is ``pm = (crflux_class, tag)``.

    The geomagnetic cutoff is on RIGIDITY ``R = (A/Z) E_nucleon``, so free
    protons (A/Z=1) and bound nucleons (He/CNO/Fe, A/Z~2) are cut at different
    energies.  By isospin the bound protons ~ the neutron flux, so from MCEq's
    own p/n nucleon fluxes: free protons = p - n (A/Z=1), bound protons = n
    (A/Z~2).  ``az_bound`` is the <A/Z> of the bound nucleons, **nucleon-flux
    weighted over the real primary composition** (He/CNO/Si A/Z=2, Fe 2.077)
    instead of a fixed 2.0, so the ~0.2-0.5% Fe sub-component is included and
    energy-dependent (Fe rises with E).  Per species the nucleon flux at
    per-nucleon energy E is A^2 Phi(A E).

    Factored out of ``MCEq3DFlux.__init__`` (2026-09-09) so the optional
    ``gs_interaction_model`` / ``gs_primary`` response engine can reuse it
    verbatim; the arithmetic is unchanged.
    """
    phi0_std = mceq._phi0.copy()
    p = mceq.pman[(2212, 0)]
    n = mceq.pman[(2112, 0)]
    p_sl = slice(p.lidx, p.uidx)
    n_sl = slice(n.lidx, n.uidx)
    p_arr = phi0_std[p_sl]
    n_arr = phi0_std[n_sl]
    with np.errstate(invalid="ignore", divide="ignore"):
        f_free = np.where(p_arr > 0, np.clip((p_arr - n_arr) / p_arr, 0.0, 1.0), 1.0)
    cr = pm[0](pm[1])
    num = np.zeros_like(e_grid)
    den = np.zeros_like(e_grid)
    for cid in cr.nucleus_ids:
        Z, A = cr.Z_A(cid)
        if A <= 1:
            continue  # free protons handled separately (A/Z=1)
        w = A * A * np.ravel(cr.nucleus_flux(cid, A * e_grid))
        num += (A / Z) * w
        den += w
    az_bound = np.where(den > 0, num / den, 2.0)
    return phi0_std, p_sl, n_sl, f_free, az_bound


class MCEq3DFlux:
    """Absolute directional flux engine (full sky: down-going + up-going)."""

    def __init__(
        self,
        interaction_model="SIBYLL23D",
        primary=("HillasGaisser2012", "H3a"),
        e_min=0.1,
        atmosphere=None,
        base_model="mceq",
        daemonflux_location="generic",
        hybrid_e0=1.7,
        gs_interaction_model=None,
        gs_primary=None,
    ):
        """``atmosphere`` is an MCEq ``density_model`` tuple. Default ``None`` keeps
        MCEq's realistic **CORSIKA US-Standard** layered profile (NOT the isothermal
        exponential used only in the `spherical_cascade` research demo). For
        seasonal/site tracking pass e.g. ``("MSIS00", ("SoudanMine", "January"))``.

        ``base_model`` selects the 1D base the geomagnetic factor multiplies:
        ``"mceq"`` (default) uses raw MCEq; ``"daemonflux"`` uses daemonflux's
        **muon-calibrated, data-anchored** 1D flux (numuflux/nueflux split by the
        ratios), which removes the ~10-20% hadronic-model normalization offset.

        ``interaction_model`` / ``primary`` set the hadronic model and primary
        spectrum of BOTH the MCEq base (raw or the MCEq half of ``"hybrid"``) and
        the geomagnetic suppression response ``G_s`` (:meth:`geomag_response`),
        and are part of the ``G_s`` disk-cache key ``self._tag``, so a response
        built with one model is never silently reused for another.  Offline (no
        network) the models available in the shipped MCEq database are
        SIBYLL23D/23E/23E*variants, SIBYLL21, DPMJETIII193, EPOSLHC, EPOSLHCR,
        QGSJETII04 and QGSJETIII; ``primary`` is any ``crflux.models`` class name
        (``HillasGaisser2012``/``"H3a"``, ``GlobalSplineFitBeta``/``None``,
        ``GaisserHonda``/``None``, ...).

        ``gs_interaction_model`` / ``gs_primary`` (both ``None`` by default, i.e.
        no override and exactly the historical behaviour) decouple the **response**
        from the base: when either is given, ``G_s`` is computed from a second,
        lazily-built ``MCEqRun`` with that model/primary while the base keeps
        ``interaction_model``/``primary``.  This isolates "how much of a residual
        is the cutoff response" from "how much is the base normalisation" -- the
        two are otherwise varied together.  The override has its own cache tag.
        """
        import crflux.models as crf
        from MCEq.core import MCEqRun
        import mceq_config as config

        self.base_model = base_model
        # hybrid-blend centre [GeV]. Physics window: bounded above by the
        # daemonflux muon-calibration floor mapped to neutrinos
        # (E_mu >= 5 GeV -> E_nu ~ E_mu/3 ~ 1.7 GeV, above which the muon
        # calibration fully constrains the flux) and below by where the
        # GSF-anchored MCEq base is validated (~0.15 GeV). The delivered
        # ratios are insensitive to E0 within this window (<=6% below 1 GeV,
        # scan in docs); the default 1.7 GeV IS the derived calibration floor
        # (fixed before the scan) -- NOT tuned to any reference.
        self._hybrid_e0 = float(hybrid_e0)
        # identity for the G_s disk cache (G_s depends only on these + rc_grid, cz_ref)
        self._tag = (
            f"{interaction_model}_{primary[0]}-{primary[1]}_emin{e_min:g}"
            f"_atm{'std' if atmosphere is None else str(atmosphere)}"
        )
        self._df = None
        if base_model in ("daemonflux", "hybrid"):
            from daemonflux import Flux

            self._df = Flux(location=daemonflux_location)
        config.e_min = e_min
        pm = (getattr(crf, primary[0]), primary[1])
        self.mceq = MCEqRun(
            interaction_model=interaction_model, primary_model=pm, theta_deg=0.0
        )
        if atmosphere is not None:
            self.mceq.set_density_model(atmosphere)
        self.e = self.mceq.e_grid
        (
            self._phi0_std,
            self._p_sl,
            self._n_sl,
            self._f_free,
            self._az_bound,
        ) = _nucleon_split(self.mceq, pm, self.e)
        # Optional G_s-only model override (see the docstring). ``None`` keeps
        # the response on ``self.mceq`` -- byte-identical to the pre-2026-09-09
        # behaviour, same cache tag, no second MCEqRun ever built.
        self._gs_spec = None
        self._gs_cache = None
        self._gs_tag = self._tag
        if gs_interaction_model is not None or gs_primary is not None:
            gim = gs_interaction_model or interaction_model
            gpr = tuple(gs_primary) if gs_primary is not None else tuple(primary)
            self._gs_spec = (gim, gpr, atmosphere, float(e_min))
            self._gs_tag = (
                f"{gim}_{gpr[0]}-{gpr[1]}_emin{e_min:g}"
                f"_atm{'std' if atmosphere is None else str(atmosphere)}"
            )

    def _gs_engine(self):
        """``(mceq, phi0_std, p_sl, n_sl, f_free, az_bound)`` behind ``G_s``.

        The base engine itself unless ``gs_interaction_model`` / ``gs_primary``
        were given, in which case a second ``MCEqRun`` is built on first use (and
        kept) so the response can use a different hadronic model or primary
        spectrum from the base.  The two must share MCEq's energy grid.
        """
        if self._gs_spec is None:
            return (
                self.mceq,
                self._phi0_std,
                self._p_sl,
                self._n_sl,
                self._f_free,
                self._az_bound,
            )
        if self._gs_cache is None:
            import crflux.models as crf
            from MCEq.core import MCEqRun
            import mceq_config as config

            gim, gpr, atmosphere, e_min = self._gs_spec
            config.e_min = e_min
            pm = (getattr(crf, gpr[0]), gpr[1])
            run = MCEqRun(
                interaction_model=gim, primary_model=pm, theta_deg=0.0
            )
            if atmosphere is not None:
                run.set_density_model(atmosphere)
            if run.e_grid.shape != self.e.shape or not np.allclose(
                run.e_grid, self.e
            ):
                raise RuntimeError(
                    "the G_s override engine has a different energy grid than "
                    "the base engine; G_s could not be applied to the base"
                )
            self._gs_cache = (run,) + _nucleon_split(run, pm, self.e)
        return self._gs_cache

    # -- base: MCEq per zenith, curved atmosphere, no geomag --
    def base(self, cos_zeniths):
        """Phi_MCEq[species, |cosZ|, E] in /(m^2 s sr GeV) (curved, no geomag).

        Uses ``|cosZ|`` (production is up/down symmetric: an up-going neutrino is
        produced on the far side at the conjugate down-going slant).
        """
        if self.base_model == "daemonflux":
            return self._base_daemonflux(cos_zeniths)
        self.mceq._phi0[:] = self._phi0_std
        out = {s: np.zeros((len(cos_zeniths), len(self.e))) for s in SPECIES}
        for i, cz in enumerate(cos_zeniths):
            self.mceq.set_theta_deg(np.degrees(np.arccos(np.clip(abs(cz), 1e-3, 1))))
            self.mceq.solve()
            for s in SPECIES:
                out[s][i] = self.mceq.get_solution(s, 0) * CM2_PER_M2
        if self.base_model == "hybrid":
            # Muon-calibrated daemonflux where the calibration is valid
            # (E >~ 1 GeV), data-anchored MCEq below (recommended with the
            # GSF primary: AMS-02/BESS/PAMELA-fitted, which fixes the sub-GeV
            # primary that H3a extrapolates poorly). Smooth log-blend around
            # E0=0.8 GeV (one octave wide): the two bases agree with the 3D
            # references in complementary domains (GSF-MCEq 0.15-0.5 GeV,
            # daemonflux >=1 GeV), so the blend tracks the better one.
            df = self._base_daemonflux(cos_zeniths)
            w = hybrid_weight(self.e, e0=self._hybrid_e0)
            for s in SPECIES:
                out[s] = (1.0 - w)[None, :] * out[s] + w[None, :] * df[s]
        return out

    def _base_daemonflux(self, cos_zeniths):
        """Muon-calibrated 1D base from daemonflux [/(m^2 s sr GeV)], all flavours.

        daemonflux reports E^3-weighted *sums* (numuflux = nu_mu+nubar_mu) and the
        ratios (numuratio = nu_mu/nubar_mu); we de-weight by E^3, convert cm->m,
        and split into species with the ratios. Uses |cosZ| (up/down symmetric).

        Validity: this base matches Honda to ~10 % for E >~ 1 GeV but
        **over-predicts below ~0.3 GeV** (up to ~2x at 0.1 GeV), where it
        extrapolates past daemonflux's muon-calibration region; there the raw MCEq
        base is closer to Honda (see `base_comparison.py`). For sub-0.3-GeV work,
        cross-check both bases.
        """
        e = self.e
        m = e <= 1.0e9  # daemonflux splines are valid to ~1e9 GeV (>> 3D regime)
        ev = e[m]
        out = {s: np.zeros((len(cos_zeniths), len(e))) for s in SPECIES}
        for i, cz in enumerate(cos_zeniths):
            zen = float(np.degrees(np.arccos(np.clip(abs(cz), 1e-3, 1))))
            tot_mu = self._df.flux(ev, zen, "numuflux") / ev**3 * CM2_PER_M2
            r_mu = self._df.flux(ev, zen, "numuratio")  # nu_mu / nubar_mu
            tot_e = self._df.flux(ev, zen, "nueflux") / ev**3 * CM2_PER_M2
            r_e = self._df.flux(ev, zen, "nueratio")
            out["total_numu"][i, m] = tot_mu * r_mu / (1.0 + r_mu)
            out["total_antinumu"][i, m] = tot_mu / (1.0 + r_mu)
            out["total_nue"][i, m] = tot_e * r_e / (1.0 + r_e)
            out["total_antinue"][i, m] = tot_e / (1.0 + r_e)
        return out

    # daemonflux quantity name per engine species (conventional charge/flavour)
    _DF_Q = {
        "total_numu": "numu",
        "total_antinumu": "antinumu",
        "total_nue": "nue",
        "total_antinue": "antinue",
    }

    def _daemonflux_relerr(self, cos_zeniths, only_hadronic=False):
        """Fractional calibration uncertainty sigma/Phi[species, cosZ, E] from
        daemonflux's nuisance-parameter covariance (its ``error()``).

        The directional flux is ``Phi_df * G * S`` with ``G, S`` independent of the
        daemonflux nuisance parameters, so the *fractional* error is preserved and
        propagates unchanged: ``sigma(Phi_3D)/Phi_3D = sigma(Phi_df)/Phi_df``.
        ``only_hadronic`` isolates the hadronic-production part of the covariance.
        """
        e = self.e
        m = e <= 1.0e9
        ev = e[m]
        out = {s: np.zeros((len(cos_zeniths), len(e))) for s in SPECIES}
        for i, cz in enumerate(cos_zeniths):
            zen = float(np.degrees(np.arccos(np.clip(abs(cz), 1e-3, 1))))
            for s in SPECIES:
                q = self._DF_Q[s]
                f = np.asarray(self._df.flux(ev, zen, q))
                er = np.asarray(self._df.error(ev, zen, q, only_hadronic=only_hadronic))
                with np.errstate(invalid="ignore", divide="ignore"):
                    out[s][i, m] = np.where(f > 0, er / f, 0.0)
        return out

    def calib_jacobian(self, cos_zeniths):
        """Full correlated calibration Jacobian for propagation into a fit.

        Returns ``dict(params, corr, cov, jac)`` where ``jac[species]`` has shape
        ``(n_param, cosZ, E)`` and is the **fractional** flux response to a +1-sigma
        pull of each daemonflux nuisance parameter, ``R_i = Phi(pull_i=+1)/Phi - 1``.
        Because ``G, S`` are parameter-independent, ``R_i`` is identical for the base
        and the directional flux (and azimuth-independent). ``corr`` is the parameter
        correlation matrix (unit-variance pulls). The directional-flux calibration
        covariance for any direction is then ``(Phi*R)^T corr (Phi*R)``
        (see :func:`calib_covariance`); by construction ``sqrt(R^T corr R)`` equals
        the fractional ``error()``. Enables an oscillation fit to carry daemonflux's
        *correlated* nuisance parameters on the 3D flux, not just the 1-sigma band.
        """
        if self.base_model != "daemonflux":
            raise ValueError("calib_jacobian requires base_model='daemonflux'")
        names = list(self._df.params.known_parameters)
        cov = np.asarray(self._df.params.cov)
        sig = np.sqrt(np.diag(cov))
        corr = cov / np.outer(sig, sig)
        e = self.e
        m = e <= 1.0e9
        ev = e[m]
        jac = {s: np.zeros((len(names), len(cos_zeniths), len(e))) for s in SPECIES}
        for iz, cz in enumerate(cos_zeniths):
            zen = float(np.degrees(np.arccos(np.clip(abs(cz), 1e-3, 1))))
            for s in SPECIES:
                q = self._DF_Q[s]
                c = np.asarray(self._df.flux(ev, zen, q))
                for ip, name in enumerate(names):
                    sh = np.asarray(self._df.flux(ev, zen, q, params={name: 1.0}))
                    with np.errstate(invalid="ignore", divide="ignore"):
                        jac[s][ip, iz, m] = np.where(c > 0, sh / c - 1.0, 0.0)
        return dict(params=names, corr=corr, cov=cov, jac=jac)

    # -- geomagnetic response G_s(E, R_c) from cascade-correct cut/full --
    def geomag_response(self, rc_grid, cz_ref=1.0, cache_dir=None, sigma_lnr=None):
        """G[species, R_c, E] = MCEq(primary cut at R_c)/MCEq(full) at one zenith.

        ``G_s`` depends only on the interaction model / primary / atmosphere / e_min
        (via ``self._gs_tag`` -- ``self._tag`` unless ``gs_interaction_model`` /
        ``gs_primary`` override the response model), the rigidity grid, ``cz_ref``
        and the penumbra width
        ``sigma_lnr`` -- it is **site- and (validated) zenith-independent**. With
        ``cache_dir`` set it is memoised to ``<cache_dir>/gs_*.npz`` and reused
        across sites and calls; the width is part of the key, so caches built
        with a different penumbra are never silently reused.

        ``sigma_lnr`` (default :data:`SIGMA_LNR`) is the width in ``ln R`` of the
        cutoff transmission Eq. (7).  It is measured from the back-traced
        admittance band, not tuned; ``0`` is the exact bin-averaged hard step.
        """
        rc_grid = np.asarray(rc_grid, float)
        if sigma_lnr is None:
            sigma_lnr = SIGMA_LNR
        sigma_lnr = float(sigma_lnr)
        fpath = None
        if cache_dir is not None:
            fpath = os.path.join(
                cache_dir, gs_cache_name(self._gs_tag, cz_ref, rc_grid, sigma_lnr)
            )
            if os.path.exists(fpath):
                d = np.load(fpath)
                if d["e"].shape == self.e.shape and np.allclose(d["e"], self.e):
                    return {s: d[s] for s in SPECIES}, rc_grid
        mq, phi0_std, p_sl, n_sl, f_free, az_bound = self._gs_engine()
        mq.set_theta_deg(np.degrees(np.arccos(np.clip(cz_ref, 1e-3, 1))))
        mq._phi0[:] = phi0_std
        mq.solve()
        full = {s: mq.get_solution(s, 0).copy() for s in SPECIES}
        G = {s: np.ones((len(rc_grid), len(self.e))) for s in SPECIES}
        for j, rc in enumerate(rc_grid):
            # proper per-nucleus rigidity: split the proton flux into free
            # (A/Z=1) and bound (A/Z~2); neutrons are all bound.
            t1 = _transmission(self.e, rc, sigma_lnr, az_over_z=1.0)
            t2 = _transmission(self.e, rc, sigma_lnr, az_over_z=az_bound)
            tp = f_free * t1 + (1.0 - f_free) * t2
            tn = t2
            mq._phi0[:] = phi0_std
            mq._phi0[p_sl] *= tp
            mq._phi0[n_sl] *= tn
            mq.solve()
            for s in SPECIES:
                cut = mq.get_solution(s, 0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    G[s][j] = np.where(full[s] > 0, cut / full[s], 1.0)
        mq._phi0[:] = phi0_std
        if fpath is not None:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez(fpath, e=self.e, **{s: G[s] for s in SPECIES})
        return G, rc_grid

    # -- solar modulation: force-field on the primary -> neutrino-energy factor --
    def _modulate_phi0(self, phi_gv):
        """Force-field (Gleeson-Axford) modulation of the primary nucleon _phi0.

        Potential ``phi_gv`` [GV] shifts each nucleon by its per-nucleon energy loss
        Phi = (Z/A)*phi and applies the flux Jacobian. Free protons (Z/A=1) lose the
        full phi; bound nucleons (Z/A=1/<A/Z>~0.5) lose half -- the standard
        rigidity-dependent modulation. ``phi_gv=0`` returns the baseline unchanged.
        """
        m_n = 0.938272
        E = self.e
        out = self._phi0_std.copy()

        def ff(f0, z_over_a):
            phi = z_over_a * phi_gv
            lo = np.log(np.maximum(f0, 1e-300))
            # Negative phi = DE-modulation (e.g. to a solar-minimum epoch):
            # E+phi can drop below the grid; clip to the grid floor. This is
            # harmless for the neutrino flux: sub-threshold primaries
            # (E_kin below the single-pion production threshold ~0.29 GeV)
            # cannot produce the E_nu >= 0.1 GeV flux modelled here.
            es = np.clip(E + phi, E[0], None)
            shifted = np.exp(np.interp(np.log(es), np.log(E), lo))
            jac = (E * (E + 2 * m_n)) / (es * (es + 2 * m_n))
            return shifted * jac

        p = self._phi0_std[self._p_sl]
        n = self._phi0_std[self._n_sl]
        zoa_bound = 1.0 / self._az_bound  # Z/A of bound nucleons (~0.5)
        out[self._p_sl] = ff(p * self._f_free, 1.0) + ff(
            p * (1 - self._f_free), zoa_bound
        )
        out[self._n_sl] = ff(n, zoa_bound)
        return out

    def solar_factor(self, phi_gv, cz_ref=1.0):
        """Neutrino-energy solar-modulation factor S(E) = numu(modulated)/numu(base).

        Runs the cascade with the force-field-modulated primary (`_modulate_phi0`)
        over the unmodulated one; the ratio maps the primary modulation to neutrino
        energy through the shower (species- and ~zenith-independent, like G_s), so it
        multiplies whichever 1D base is used. ``phi_gv`` is *relative to the H3a
        baseline* (0 = baseline); the physical solar-cycle effect is the difference
        between two potentials (e.g. ~0.4 GV solar-min vs ~1.0 GV solar-max).
        """
        if phi_gv == 0:
            return np.ones(len(self.e))
        self.mceq.set_theta_deg(np.degrees(np.arccos(np.clip(cz_ref, 1e-3, 1))))
        self.mceq._phi0[:] = self._phi0_std
        self.mceq.solve()
        ref = self.mceq.get_solution("total_numu", 0).copy()
        self.mceq._phi0[:] = self._modulate_phi0(phi_gv)
        self.mceq.solve()
        mod = self.mceq.get_solution("total_numu", 0)
        self.mceq._phi0[:] = self._phi0_std
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(ref > 0, mod / ref, 1.0)

    # -- genuine-3D production-angle redistribution R(E, cosZ) = Phi_3D/Phi_1D --
    def angular_factor(
        self, cos_zeniths, moments="m_spliced.npz", n_dense=41, n_mc=4000
    ):
        """Production-angle 3D redistribution factor R[species][cosZ, E].

        The 1D base is collinear (neutrino along the primary); in 3D a neutrino from
        direction n_o is produced along a spread of parent directions about it. We
        convolve the per-zenith base with the **NA61-validated** production-angle
        spread ``sigma_theta(E)`` (from the high-statistics kernel moments) on the
        sphere (`coupled_3d_flux.convolve_sphere`), flux-conserving. R->1 at high E
        (sigma->0); sub-GeV it is a ~1-2% zenith redistribution (slight horizon
        deficit / vertical excess). Site- and azimuth-independent; the curved-
        atmosphere sec-theta rise is already in the per-zenith base and is *not*
        touched here (no double counting).
        """
        from fokker_planck_3d import load_theta2, sigma_theta_vs_energy
        from coupled_3d_flux import convolve_sphere

        e = self.e
        e_sig, theta2 = load_theta2(moments)
        sig = np.deg2rad(sigma_theta_vs_energy(e_sig, theta2, e, zenith_deg=0.0))
        dense = np.linspace(-1.0, 1.0, n_dense)  # mirrored full sphere
        base = self.base(dense)  # per species (n_dense, n_E), symmetric in cosZ
        out_cos = np.clip(np.abs(np.asarray(cos_zeniths, float)), 1e-3, 1.0)
        sigma_EZ = np.tile(sig, (len(out_cos), 1))
        eidx = list(range(len(e)))
        R = {}
        for s in SPECIES:
            phi3d = convolve_sphere(
                e, dense, base[s], sigma_EZ, out_cos, eidx, n_mc=n_mc
            )
            phi1d = np.array(
                [np.interp(out_cos, dense, base[s][:, ie]) for ie in eidx]
            ).T
            with np.errstate(invalid="ignore", divide="ignore"):
                R[s] = np.where(phi1d > 0, phi3d / phi1d, 1.0)  # (n_cos, n_E)
        return R

    def offaxis_factor(self, cos_zeniths, path=None, shape_only=False,
                       which="E_off", species_table=None):
        """First-principles off-axis 3D-production factor E_off[species][cosZ, E].

        The complete genuine-3D/1D production ratio: it both redistributes flux in
        zenith and produces the sub-GeV near-horizon excess (horizon/vertical ~1.8
        at 0.3 GeV) that the per-zenith cascade cannot. Derived *from first
        principles* in `offaxis_mc.py`: MCEq depth-resolved production p(X,E),
        curved-atmosphere slant depth, and the **pion production angle from the
        SIBYLL/UrQMD generator moments** (chromo, NA61-validated) folded with exact
        pi->mu nu decay. The excess is a pion-production-rate effect inherited by
        every daughter neutrino (muon-decay neutrinos included -- NOT a flat
        pedestal), so E_off is **flavour-independent**: one table for all species;
        the nu_e-vs-nu_mu flux difference lives entirely in the per-flavour base.
        No reference flux is used. The delivered cone uses the exact NA61-validated
        moment sigma_pi (the sampled full kernel was root-caused as ~16-37% too wide
        -- a theta-binning artefact; diag_kernel_consistency), giving nu_mu AND nu_e
        zenith shapes within ~0-8% of BOTH Honda and Bartol over 0.3-1 GeV; the
        ~7-8% residual at 1-3 GeV is the base sec-theta, not E_off. E_off is
        near-flux-conserving (<E_off>_Omega within 4-6% of 1 in 0.2-1 GeV;
        `diag_eoff_conservation.py`), i.e. a redistribution, not a pedestal (the
        residual grows to ~8% at 0.11 GeV, the factorised extrapolation edge).
        E_off->1 at high E; applied to ``|cosZ|``; =1 outside the tabulated E range
        (0.1-100 GeV -- below 0.1 GeV the excess is large and NOT modelled).

        ``shape_only`` (opt-in, off by default): divide out the vertical value,
        E_off(cosZ)/E_off(vert). This was once the default on the daemonflux base
        out of a double-counting worry, but the muon closure shows it is
        unnecessary: daemonflux's neutrino flux is genuinely 1D and E_off_mu ~ 1
        in its muon-calibration region (E_mu >~ 5 GeV), so the **full** factor is
        self-consistent with the calibration and tracks Honda better (the vertical
        sub-GeV flux lands at ~1.0x Honda with the full factor vs ~1.07x
        shape-only). Kept only for A/B comparison.

        ``which``: "E_off" (central) | "E_off_hi"/"E_off_lo" (NA61 +-12%
        pion-angle variants, used for the sigma_pi_NA61 covariance pull).

        DEFAULT TABLE (changed 2026-09-04): :data:`EOFF_TABLE`
        ``offaxis_excess_channel_v2.npz`` -- the **species-resolved,
        channel-weighted** build (``offaxis_mc.build_channel``) on the
        arcsin-corrected v2 moments.  It carries an ``E_off_s[species, cz, E]``
        axis: the muon-decay parent channel gets its own (wider, per-flavour)
        cone instead of the direct pion cone, so the factor is no longer
        flavour-blind, and it is the SAME physics the joint cone integral
        (:meth:`joint_cone_factor`) evaluates -- with ``G == 1`` the two agree to
        the quadrature accuracy (``test_joint_cone``,
        ``test_mceq3d_flux.test_table_path_matches_joint_path_at_unit_G``).
        Before this change every path that did not go through the joint factor
        (``cone_cutoff=False``, the up-going hemisphere, energies outside
        0.1-100 GeV, the ``sigma_pi_NA61`` pull) used the flat pion-only July
        table on the pre-arcsin moments, i.e. different physics from the
        down-going in-range flux next to it.

        ``species_table``: ``None`` (default) = use the per-species ``*_s`` axis
        when the file has one; ``False`` forces the flavour-blind row (for A/B
        against the flat build); ``True`` requires it.  :data:`EOFF_TABLE_FLAT`
        (``offaxis_excess.npz``, the original July build) stays readable through
        ``path=`` for A/B comparisons.
        """
        here = os.path.dirname(os.path.abspath(__file__))
        if path is None:
            path = os.path.join(here, EOFF_TABLE)
        elif not os.path.isabs(path) and not os.path.exists(path):
            path = os.path.join(here, path)  # bare table name -> next to this module
        e = self.e
        cz = np.abs(np.asarray(cos_zeniths, float))
        if not os.path.exists(path):
            flag = ("--build-channel" if "channel" in os.path.basename(path)
                    else "--build")
            raise FileNotFoundError(
                f"{path} not found; run 'python offaxis_mc.py {flag}' first"
            )
        d = np.load(path)
        if "tag" in d:
            tag = str(d["tag"])
            if not self._tag.startswith(tag):
                # interaction model must match (the kernel and p(X,E) are its
                # yields); a primary-spectrum mismatch only reweights the ratio
                # mildly -> warn (E_off is primary-insensitive to first order).
                if not self._tag.startswith(tag.split("_")[0]):
                    raise ValueError(
                        f"offaxis table built for {tag} but engine is "
                        f"{self._tag}; rebuild with offaxis_mc.py --build"
                    )
                import warnings

                warnings.warn(
                    f"offaxis table primary ({tag}) differs from engine "
                    f"({self._tag}); E_off is a primary-insensitive ratio, "
                    "but rebuild for exactness."
                )
        Ecz, Ee = d["cz"], d["e"]
        # species-resolved axis (channel-weighted builds): E_off_s[sp, cz, E]
        has_s = (which + "_s") in d and "species" in d
        if species_table and not has_s:
            raise ValueError(f"{path} has no per-species {which}_s axis")
        use_s = has_s and species_table is not False
        if use_s:
            order = [str(x) for x in d["species"]]
            arr = d[which + "_s"]
            raw = {sp: arr[order.index(SP_LABEL[sp])] for sp in SPECIES}
        else:
            raw = {sp: d[which] for sp in SPECIES}  # flavour-blind (geometric)

        def _regrid(Etab):
            if shape_only:
                Etab = Etab / Etab[np.argmax(Ecz)]  # normalise to the vertical row
            # E_off(cz, E) -> (len(cz), len(e)); 1 outside the tabulated E range
            table = np.ones((len(cz), len(e)))
            lo, hi = Ee[0], Ee[-1]
            for k, en in enumerate(e):
                if en < lo or en > hi:
                    continue
                col = np.array([np.interp(np.log(en), np.log(Ee), Etab[i])
                                for i in range(len(Ecz))])
                table[:, k] = np.interp(cz, Ecz, col)
            return table

        return {sp: _regrid(raw[sp]) for sp in SPECIES}

    def finemap_rc(self, lat, lon, date, n_zen=13, n_az=25, n_scan=None,
                   cache_dir=None, full_sphere=True, r_hi=RC_MAX_GV,
                   n_jobs=None, limb_nodes=True):
        """Cutoff map R_c(zenith[deg], azimuth[deg]) [GV] for interpolation to
        arbitrary production-cone directions (cone_cutoff). Cached per
        site/date/grid.

        With ``full_sphere=True`` (default) the map spans the **whole sphere**,
        zenith 0->180: the down-going hemisphere is the detector cutoff
        (:func:`geomag_backtrace.cutoff_map`), the up-going hemisphere (>90 deg)
        is the far-side/global cutoff (:func:`farside_cutoff_map`), used for
        genuinely up-going neutrino directions. Set ``full_sphere=False`` for the
        down-going-only map.

        Zenith nodes (``limb_nodes``, default True): a uniform 13-node grid puts a
        7.4 deg gap over 82-89 deg, exactly where the East cutoff rises convexly by
        ~1.5 GV/deg, so bilinear interpolation *over-estimated* it by +2.6 GV at
        84 deg, +4.2 GV at 87 deg and +2.9 GV at 88 deg -- an error as large as the
        whole residual the paper attributes to physics.  The node set is therefore
        ``linspace(0, 80, n_zen)`` plus dense limb nodes at 82, 84, 85.5, 87, 88,
        88.5, 89, 89.5 deg (>= 8 nodes in 80-90 deg), mirrored about 90 deg for the
        up-going hemisphere.  The extra nodes cost back-traces once per site/date;
        the cone integral that reads the map is unchanged.

        The rigidity scan is the coarse-ladder + bisection scheme of
        :func:`geomag_backtrace.scan_upper_cutoff` (~0.05 GV), parallelised over
        ``n_jobs`` forked workers (default ``os.cpu_count()//2``); ``n_scan`` is
        kept only as a backward-compatible way to *refine* the coarse step."""
        import geomag_backtrace as gb

        zen_down = _zenith_nodes(n_zen, limb_nodes)
        az = np.linspace(0.0, 360.0, n_az)
        if full_sphere:
            zen_up = 180.0 - zen_down[::-1]  # mirror: dense just below the limb too
            zen = np.concatenate([zen_down, zen_up])
        else:
            zen = zen_down
        dtag = date.isoformat() if hasattr(date, "isoformat") else str(date)
        sphtag = "sph" if full_sphere else "dn"
        # The key carries (a) r_hi -- a map built with a lower ceiling is
        # SATURATED, not merely coarser; (b) the scan-scheme tag -- a map built
        # with the old single-ladder scan has O(GV) errors and allowed-island
        # dropouts; (c) the zenith-node scheme. None may be silently reused.
        key = (f"{lat:.4f}_{lon:.4f}_{dtag}_{n_zen}x{n_az}_{n_scan}_{sphtag}"
               f"_rhi{r_hi:g}_{gb.CUTOFF_SCHEME}"
               f"_zn{'limb' if limb_nodes else 'uni'}")
        # in-memory memo: with cone_cutoff now the default, this keeps repeated
        # solve() calls in one session from rebuilding the map even without a disk
        # cache_dir (the first call still pays the one-off back-trace).
        memo = getattr(self, "_finemap_memo", None)
        if memo is None:
            memo = self._finemap_memo = {}
        if key in memo:
            return memo[key]
        fpath = None
        if cache_dir is not None:
            h = hashlib.md5(f"finrc_{key}".encode()).hexdigest()[:16]
            fpath = os.path.join(cache_dir, f"finerc_{h}.npz")
            if os.path.exists(fpath):
                d = np.load(fpath)
                out = (d["zen"], d["az"], d["rc"])
                memo[key] = out
                return out
        rc = gb.cutoff_map(lat, lon, date, zen_down, az, n_scan=n_scan, r_hi=r_hi,
                           n_jobs=n_jobs, warn_saturated=False)
        if full_sphere:
            rc_up = farside_cutoff_map(lat, lon, date, zen_up, az, n_scan=n_scan,
                                       r_hi=r_hi, n_jobs=n_jobs)
            rc = np.concatenate([rc, rc_up], axis=0)
        n_sat = int(np.sum(np.isclose(rc, r_hi)))
        if n_sat:
            import warnings

            warnings.warn(
                f"finemap_rc: {n_sat} of {rc.size} cells saturated at the "
                f"r_hi={r_hi:g} GV scan ceiling -- the cutoff map is flat (zero "
                "gradient) there, which suppresses direction-shift effects. "
                "Raise r_hi.",
                stacklevel=2,
            )
        if fpath is not None:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez(fpath, zen=zen, az=az, rc=rc)
        memo[key] = (zen, az, rc)
        return zen, az, rc

    def prod_family_rc(self, lat, lon, date, n_zen=13, n_az=25, r_hi=RC_MAX_GV,
                       n_jobs=None, cache_dir=None, limb_nodes=True,
                       h_prod_km=H_PROD_KM, h_launch_km=0.0, n_side=3,
                       d_max_km=None, warn_saturated=True, zeniths=None,
                       azimuths=None, n_bands=16, seed_margin=0.45):
        """Family of DOWN-GOING cutoff maps at sites displaced around the detector.

        Returns ``{"off": (n_side,), "zen": (n_zen,), "az": (n_az,),
        "rc": (n_side, n_side, n_zen, n_az)}`` -- ``rc[i, j]`` is the ordinary
        detector-style map (:func:`geomag_backtrace.cutoff_map`) of the site
        displaced ``off[i]`` km north and ``off[j]`` km east of ``(lat, lon)``.
        :func:`interp_site_rc` interpolates it to an arbitrary displacement and
        :func:`prod_point_offset_km` supplies the displacement of the
        production point of any arrival direction.  This is the object that
        makes ``cutoff_anchor="prod_point"`` affordable.

        WHY A SITE GRID AND NOT A MAP PER DIRECTION.  The cutoff at the
        production point depends on (a) the *direction*, through structure that
        is sharp near the limb (~1.5 GV/deg in the East above 84 deg), and
        (b) the *site*, through the geomagnetic latitude, which enters smoothly:
        ``R_c ~ C cos^4(lambda_mag)`` gives ``d ln R_c / d lambda ~ -4 tan
        lambda`` = -0.034/deg at Kamioka, with a curvature of ``-4 sec^2
        lambda`` = -4.95/rad^2.  Interpolating (b) linearly across the WHOLE
        displacement range D (620 km = 5.6 deg) therefore costs
        ``|f''| D^2 / 8`` ~ 0.6%, while (a) keeps the family's own fine nodes and
        is never interpolated in the site direction at all.  A ``3 x 3``
        Cartesian grid in ``(d_north, d_east)`` spanning ``[-D, D]^2`` (8 extra
        maps; the centre node IS the detector map) is thus enough, and the
        interpolation is in the true displacement components, so it carries no
        chord error the way a (radius, bearing) polar grid would.

        ``d_max_km`` defaults to the ground displacement at the exact horizon,
        ``R_E * atan2(L sin z, R_E + L cos z)`` at ``z = 90 deg`` (617 km for
        ``h_prod = 30 km``), so every down-going arrival direction lands inside
        the grid.

        ``h_launch_km`` (default 0) is the altitude the back-traces are launched
        from AND the absorbing floor -- 0 reproduces
        :func:`geomag_backtrace.cutoff_map` exactly at the centre node, which is
        what makes the ``cutoff_anchor`` A/B measure the site displacement and
        nothing else.  Set it to ``h_prod_km`` for the convention of
        ``diag_dipole_phase.py --sections prodpoint`` (launch at the production
        altitude), which lowers every cutoff by ~1% through the ``1/r^2`` of the
        Stoermer scaling.

        COST, AND THE SEEDED BAND SCAN.  ``(n_side^2 - 1) * n_zen_nodes * n_az``
        back-traced cutoffs -- 4200 for the delivered 3x3 / 21x25 grid.  Scanned
        naively that is ~4x the full-sphere detector map, because the cost of
        :func:`geomag_backtrace.scan_upper_cutoff` is dominated by the ALLOWED
        rigidities it tests above the cutoff (an allowed trajectory escapes in
        ~900 RK4 steps, a forbidden one returns in ~30), and the ladder starts at
        ``r_hi = 55 GV`` for every direction -- 47 expensive traces for a 8 GV
        West cell.

        A displaced site is not an unknown site, though: the SAME direction at
        the detector is a 20%-accurate predictor of it (the whole effect being
        modelled is a 12-16% shift).  So the directions are sorted by their
        detector-map cutoff and split into ``n_bands`` equal-count groups, each
        scanned over ``[(1-seed_margin) * min_seed, (1+seed_margin) * max_seed]``
        clipped to ``[r_lo, r_hi]``.  The scan *scheme* inside a band is
        untouched (1 GV coarse ladder + bisection to 0.1 GV), only its starting
        rigidity moves, and any cell that lands within one coarse step of its
        band edge is re-scanned over the full range -- so a cell whose cutoff
        moved by more than the margin cannot be silently clamped.  Measured
        saving ~4x; ``n_bands=0`` restores the plain full-range scan.
        """
        import geomag_backtrace as gb

        # ``zeniths``/``azimuths`` override the node sets (tests / A-B studies);
        # the centre-node reuse below then no longer applies, so the family is
        # built entirely from fresh back-traces.
        zen = _zenith_nodes(n_zen, limb_nodes) if zeniths is None else np.asarray(
            zeniths, float)
        az = np.linspace(0.0, 360.0, n_az) if azimuths is None else np.asarray(
            azimuths, float)
        if d_max_km is None:
            d_max_km = float(np.hypot(*prod_point_offset_km(90.0, 0.0, h_prod_km)))
        n_side = int(n_side)
        if n_side < 3 or n_side % 2 == 0:
            raise ValueError("n_side must be an odd integer >= 3")
        off = np.linspace(-d_max_km, d_max_km, n_side)
        dtag = date.isoformat() if hasattr(date, "isoformat") else str(date)
        key = (f"{lat:.4f}_{lon:.4f}_{dtag}_{n_zen}x{n_az}_rhi{r_hi:g}"
               f"_{gb.CUTOFF_SCHEME}_zn{'limb' if limb_nodes else 'uni'}"
               f"_{PROD_FAMILY_SCHEME}_ns{n_side}_D{d_max_km:.1f}"
               f"_hl{h_launch_km:g}_hp{h_prod_km:g}"
               f"_{'' if zeniths is None else np.asarray(zeniths).tobytes()}"
               f"_{'' if azimuths is None else np.asarray(azimuths).tobytes()}")
        memo = getattr(self, "_prodfam_memo", None)
        if memo is None:
            memo = self._prodfam_memo = {}
        if key in memo:
            return memo[key]
        fpath = None
        if cache_dir is not None:
            h = hashlib.md5(f"prodfam_{key}".encode()).hexdigest()[:16]
            fpath = os.path.join(cache_dir, f"prodfam_{h}.npz")
            if os.path.exists(fpath):
                d = np.load(fpath)
                out = {k: d[k] for k in ("off", "zen", "az", "rc")}
                memo[key] = out
                return out

        r_launch = gb.RE + h_launch_km * 1e3
        # The CENTRE node is, by construction, the ordinary detector map: with
        # h_launch_km == 0 it is byte-identical to the down-going half of the
        # (already cached) full-sphere `finemap_rc`, so reuse it -- that is both
        # 1/9 of the cost and the exact-agreement guarantee at zero displacement.
        c = n_side // 2
        rc_centre = None
        if (h_launch_km == 0.0 and abs(r_hi - RC_MAX_GV) < 1e-9
                and zeniths is None and azimuths is None):
            zf, azf, rcf = self.finemap_rc(
                lat, lon, date, n_zen=n_zen, n_az=n_az, cache_dir=cache_dir,
                r_hi=r_hi, n_jobs=n_jobs, limb_nodes=limb_nodes,
            )
            if np.allclose(zf[: len(zen)], zen) and np.allclose(azf, az):
                rc_centre = rcf[: len(zen)]
        r0, u0, todo = [], [], []
        for i, dn in enumerate(off):
            for j, de in enumerate(off):
                if rc_centre is not None and i == c and j == c:
                    continue
                todo.append((i, j))
                la_s, lo_s = prod_site_latlon(lat, lon, float(dn), float(de))
                up_s = gb._local_frame(la_s, lo_s)[0]
                for z in zen:
                    for a in az:
                        r0.append(r_launch * up_s)
                        u0.append(-gb.arrival_direction(la_s, lo_s, z, a))
        r0 = np.array(r0)
        u0 = np.array(u0)
        # Seeded band scan (see the docstring). The seed of every displaced
        # direction is the SAME direction's detector cutoff.
        seed = None
        if n_bands and rc_centre is not None and len(todo):
            seed = np.tile(np.asarray(rc_centre, float).ravel(), len(todo))
        rc_flat, sat = _banded_cutoff_scan(
            gb, r0, u0, date, seed, r_lo=0.5, r_hi=r_hi, n_jobs=n_jobs,
            r_floor=r_launch, n_bands=n_bands, seed_margin=seed_margin,
        )
        rc_flat = rc_flat.reshape(len(todo), len(zen), len(az))
        rc = np.zeros((n_side, n_side, len(zen), len(az)))
        for k, (i, j) in enumerate(todo):
            rc[i, j] = rc_flat[k]
        if rc_centre is not None:
            rc[c, c] = rc_centre
        if warn_saturated and sat.any():
            import warnings

            warnings.warn(
                f"prod_family_rc: {int(sat.sum())} of {sat.size} cells saturated "
                f"at r_hi={r_hi:g} GV. The displaced southern sites sit at lower "
                "geomagnetic latitude, so their East-limb cutoff exceeds the "
                "detector's; those cells are flat. G_s is clamped at the same "
                "ceiling, so the flux there (<1e-3 of the vertical below 1 GeV) "
                "is unaffected.",
                stacklevel=2,
            )
        out = dict(off=off, zen=zen, az=az, rc=rc)
        if fpath is not None:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez(fpath, **out)
        memo[key] = out
        return out

    #: module-level :func:`_gauss_cone`, exposed on the class for convenience.
    _gauss_cone = staticmethod(lambda *a, **k: _gauss_cone(*a, **k))

    def cone_geff(self, cos_zeniths, azimuths, rc_map, rc_grid, G_by_z,
                  fine, n_alpha=12, n_beta=12, sigma_scale=1.0, channel_cone=True,
                  muon_bending=True, b_enu=None, prod_displacement=False,
                  h_prod_km=H_PROD_KM, sublimb="prod_point",
                  cone_kernel="moments", cone_moments=None,
                  rc_family=None, site_lat=None):
        """Production-cone-averaged geomagnetic factor G_eff[species][cz, az, E].

        The parent primary of a neutrino from direction ``n`` arrives from a cone
        of half-width ``sigma_theta(E)`` about ``n`` (the same production angle
        that drives E_off); each primary direction has its own rigidity cutoff, so

            G_eff(n, E) = < G_s(R_c(n_p), E) >_cone

        -- the geomagnetic analogue of the off-axis atmospheric factor. This
        restores the near-horizon East-West asymmetry that a single cutoff at the
        neutrino direction under-produces. Applied to **down-going** neutrino
        directions (cosZ>=0); up-going neutrino directions keep the single
        far-side cutoff (their own cone extension is a further refinement).

        ``sublimb`` -- how cone samples that fall **below the detector's local
        horizon** are treated.  ``fine`` is (zen_deg, az_deg, rc_fine).

        * ``"prod_point"`` (default, 2026-09).  The whole cone belongs to a single
          production point P, a distance L up the arrival ray (272 km at 87 deg /
          h=20 km, 370 km at h=30 km); the local vertical at P is tilted by ~L/R_E
          relative to the detector's (3.3 deg at 87 deg, 4.7 deg at 89 deg).  Every
          cone sample is therefore converted to P's local frame (:func:`_prod_frame`)
          and read off the **down-going** part of the map at its local zenith there.
          A sample whose local zenith *at P* still exceeds 90 deg is genuinely
          Earth-shadowed -- no primary can arrive at P from below its own horizon --
          and is **blocked**: it is dropped from the numerator *and* from the cone
          normalisation, exactly as ``offaxis_mc.cone_numden`` sends p -> 0 for
          sub-horizon production directions.  (Dropping it from the normalisation
          too is what makes <G> a *conditional* average over the directions that
          actually produce; the loss of production from the blocked wedge already
          lives in ``E_off``, and counting it twice would double-suppress the
          horizon.)  Between ~30 and ~41% of the cone weight at 87 deg lies below
          the detector horizon, so this is not a marginal correction.
        * ``"farside"`` -- the legacy behaviour: sub-limb samples read the up-going
          hemisphere of the full-sphere map, i.e. the cutoff of an **antipodal**
          trajectory.  Those are not the primaries in question, and the stitch is
          discontinuous: at Kamioka the down-going 89-deg cell is 49.1 GV and the
          up-going 90-deg cell 39.6 GV, a 9.5 GV jump across 1 deg.  Any +-3 deg
          muon-bending shift at 87 deg straddles that seam, which is why both muon
          charges moved the *same* way there and the charge-split E-W signal
          collapsed.  Kept only for A/B comparison.

        The far-side map is still the right object for genuinely **up-going
        neutrino** directions (cosZ < 0), which keep their single far-side cutoff.

        ``channel_cone`` (default **True**): use a **channel-weighted** cone width.
        ~40% of sub-GeV nu_mu and ~all nu_e are born from **muon decay**, whose
        primary-to-neutrino angle is wider than the direct pion cone and carries an
        energy-independent in-flight-bending floor (:func:`kinematic_kernel.
        mudecay_shape`). The single narrow pion cone under-smeared these, leaving a
        residual near-horizon E-W overshoot (worst at the extreme horizon, where
        the bending floor dominates the vanishing pion cone). We average G over
        *both* cone widths and blend per species by the MCEq muon-decay flux
        fraction ``f_mu`` (:func:`kinematic_kernel.channel_fractions`):
        ``G_eff = (1-f_mu) <G>_pion + f_mu <G>_mudecay``. First-principles (no
        tuned parameter); set ``channel_cone=False`` for the legacy pion-only cone.

        ``muon_bending`` (default **True**, needs ``channel_cone`` and the local
        field ``b_enu``): the muon-decay cone is **centred on the bending-shifted**
        primary direction. A muon bends in-flight before decaying (energy-
        independent ``Delta_phi = qB tau/m ~ 3 deg``), and its decay neutrino
        inherits the shift, **charge-dependently**: nu_mu/anti-nu_e come from mu-,
        nu_e/anti-nu_mu from mu+, so the two shift oppositely. Because the species
        are tracked separately (not charge-summed), each carries the *full* shift
        (not the ~0.3 deg net cancellation). Implemented as a coherent shift of the
        muon-decay cone axis per species (:func:`muon_bending.bending_deflection`).

        ``prod_displacement`` (default **False**, opt-in): evaluate each cone
        sample's cutoff in the local frame of the **production point** rather than
        the detector's. The neutrino is produced hundreds of km up the arrival ray
        near the horizon (272 km at 87 deg / h=20 km; 370 km at h=30 km), where the
        local vertical has rotated by ~L/R_E, so the same primary direction has a
        less extreme local zenith there (87 -> 83.7 deg at h=30 km) and hence a
        LOWER cutoff. Evaluating at the detector therefore over-estimates R_c and
        over-suppresses the horizon. The correction is exactly zero at the vertical
        and maximal at the horizon, and is energy-independent -- matching the
        signature of the residual Honda discrepancy.

        ``rc_family`` / ``site_lat`` (``cutoff_anchor="prod_point"``): the cone
        samples are read off a map LAUNCHED at the production point (the
        displaced-site family of :meth:`prod_family_rc`, interpolated to this
        arrival direction's ground displacement) instead of the detector-anchored
        one, and their local angles are taken in the production point's TRUE
        local frame (:func:`prod_frame_exact`, which includes the convergence of
        the meridians that ``_prod_frame`` below ignores).  This is the *site*
        half of the production-point displacement; ``sublimb="prod_point"`` /
        ``prod_displacement`` are only the *frame-rotation* half.  ``None``
        (default) = the detector anchor, bit-for-bit the previous behaviour.

        MEASURED EFFECT when it was measured on the OLD 7.4-deg-uniform map
        (diag_prod_displacement.py): correctly signed but small -- it removed ~15%
        of the extreme-horizon E-W overshoot and ~5% of the sub-GeV zenith-shape
        deficit.  That understated it: a rigid ~3-5 deg axis shift read through a
        7.4 deg bilinear cell is largely smoothed away.  With the dense limb nodes
        the shift is resolved, and ``sublimb="prod_point"`` makes the production
        frame the default rather than an opt-in, so ``prod_displacement`` is now
        redundant with it (it remains as an explicit switch for
        ``sublimb="farside"`` A/B runs).
        """
        from kinematic_kernel import pion_alpha_pdf, channel_shapes

        if cone_kernel not in ("moments", "sampled", "legacy"):
            raise ValueError("cone_kernel must be 'moments', 'sampled' or "
                             f"'legacy', got {cone_kernel!r}")
        legacy_cone = cone_kernel == "legacy"
        zen_f, az_f, rc_f = fine  # deg, deg, (n_zen, n_az)
        if sublimb not in ("prod_point", "farside"):
            raise ValueError(f"sublimb must be 'prod_point' or 'farside', got "
                             f"{sublimb!r}")
        if rc_family is not None and sublimb != "prod_point":
            raise ValueError("rc_family (cutoff_anchor='prod_point') requires "
                             "sublimb='prod_point'")
        if sublimb == "prod_point":
            # Restrict the interpolant to the DOWN-GOING part of the map: with the
            # production frame no cone sample may read the antipodal far-side
            # cutoff any more (that stitch is the 89-deg seam). Samples that are
            # still sub-horizon at the production point are blocked instead.
            dn = zen_f <= 89.9
            zen_f, rc_f = zen_f[dn], rc_f[dn]
            prod_displacement = True
        block_sublimb = sublimb == "prod_point"
        # Cone polar grid: 0.5 -> 89 deg, the SAME convention as
        # ``offaxis_mc.cone_numden``. It used to stop at 70 deg, which (i)
        # truncated the cone that E_off integrates to the limb and (ii) made
        # ``pion_alpha_pdf``'s last histogram bin span 66.8-180 deg and deposit
        # all of it -- 8.8% of the weight at 0.3 GeV -- at the 70 deg node. On
        # this grid that bin is 88.2-180 deg and carries <1%.
        alpha_deg = np.linspace(0.5, 70.0 if legacy_cone else 89.0, n_alpha)
        # sigma_scale stretches the cone half-width (the NA61 +-12% pion-angle
        # pull that drives E_off's sigma_pi_NA61) -- used to test whether that
        # systematic spans the near-horizon E-W amplitude.
        #
        # ``cone_kernel`` (default "moments", 2026-09): which pion angular
        # distribution drives the geomagnetic cone.
        #   "moments" -- a Gaussian-on-the-sphere of the NA61-validated moment
        #     ``sigma_pi`` (``m_spliced.npz``), with the per-axis width
        #     ``sigma/sqrt(2)`` of a space-angle RMS. This is EXACTLY the cone
        #     ``offaxis_mc.cone_numden`` uses to build E_off, so the two factors
        #     of the delivered product finally describe the same cone.
        #   "legacy" -- EXACTLY the pre-2026-09 behaviour, for A/B only: the
        #     sampled kernel on a 0.5-70 deg grid with sigma (not sigma/sqrt2)
        #     in both Gaussians. This is what produced the PAPER_DRAFT numbers.
        #   "sampled" -- the full (x_L, theta) kernel ``k_spliced.npz``.
        #     Root-caused in July as 16-37% too wide in per-secondary meson-angle
        #     RMS (a 0.667-deg theta-binning artefact); ``offaxis_mc`` was
        #     switched to the moments then but this cone was not, so the paper's
        #     claim that the delivered cone uses the moment sigma_pi was true of
        #     E_off only. Kept for A/B.
        # The 1/sqrt(2) is the other half of the same inconsistency: the moment
        # sigma is a SPACE-ANGLE RMS, so a Gaussian on the sphere must use
        # sigma/sqrt(2) per transverse axis (``offaxis_mc.cone_numden``); using
        # sigma directly made both Gaussian cones here 41% too wide.
        # ``cone_moments``: {species: [files]} override for the generator
        # production-angle moments (``kinematic_kernel.meson_theta2``), e.g. the
        # arcsin-corrected v2 moment files. None = the delivered defaults; this
        # never mutates kinematic_kernel's own defaults.
        mkw = {} if cone_moments is None else {"moments": cone_moments}
        sig = channel_shapes(self.e, **mkw)["pi"] * sigma_scale
        gauss = _gauss_cone(alpha_deg, sig, per_axis=not legacy_cone)
        if cone_kernel in ("sampled", "legacy"):
            W = pion_alpha_pdf(self.e, alpha_deg, scale=sigma_scale)
            W = np.where(W.sum(1, keepdims=True) > 0, W, gauss)  # thin-stats rows
        else:
            W = gauss
        W = W / np.maximum(W.sum(1, keepdims=True), 1e-300)
        alpha = np.deg2rad(alpha_deg)
        beta = np.linspace(0.0, 2 * np.pi, n_beta, endpoint=False)

        # channel-weighted cone: a second (wider) muon-decay cone weight W_mu, and
        # the per-species muon-decay flux fraction f_mu to blend the two averages.
        #
        # BOTH are ZENITH-DEPENDENT and are therefore evaluated per direction below
        # (memoised on the instance per rounded zenith). The muon-decay fraction
        # f_mu rises strongly toward the horizon (longer slant path -> more decay
        # in flight: ~0.3 vertical -> ~0.5 saturated at 85 deg), and the muon-decay
        # cone width likewise depends on the decay-in-flight fraction. Using the
        # vertical values everywhere (the previous behaviour) under-weighted the
        # wide muon-decay cone exactly at the horizon, where it matters most, and
        # suppressed the delivered sub-GeV horizon/vertical ratio by ~15-19%
        # (diag_shape_decompose.py).
        if channel_cone:
            from kinematic_kernel import mudecay_shape, channel_fractions

        def _channel_at_zenith(zen_deg):
            """(W_mu, fmu) for a given zenith [deg], memoised per 10-deg bucket."""
            key = int(round(zen_deg / 10.0) * 10)
            cache = getattr(self, "_channel_by_zen", None)
            if cache is None:
                cache = self._channel_by_zen = {}
            if key in cache:
                return cache[key]
            sig_mu = np.maximum(mudecay_shape(self.e, zenith_deg=float(key))
                                * sigma_scale, 1e-3)
            w = _gauss_cone(alpha_deg, sig_mu, per_axis=not legacy_cone)
            w = w / np.maximum(w.sum(1, keepdims=True), 1e-300)
            # Per-SPECIES (not per-flavour) muon-decay fraction: nu and nubar of
            # the same flavour have different f_mu because the muon charge ratio
            # is ~1.27, and by lepton-flavour conservation each neutrino species
            # comes from a definite parent muon charge (mu_numu is pure mu-,
            # mu_antinumu pure mu+). Previously both charges shared the numu (or
            # nue) value, which blurred exactly the charge asymmetry the
            # bending-shifted cone is meant to express.
            f = {s: channel_fractions(self.e, SP_LABEL[s],
                                      zenith_deg=float(key))["mu"]
                 for s in SPECIES}
            cache[key] = (w, f)
            return cache[key]

        # coherent muon-bending shift of the muon-decay cone axis (charge-signed):
        # nu from mu+ (anti-nu_mu, nu_e) shift one way, nu from mu- (nu_mu,
        # anti-nu_e) the other. Needs the local field vector b_enu (E,N,U).
        do_bend = channel_cone and muon_bending and b_enu is not None
        MU_PLUS = ("total_antinumu", "total_nue")  # neutrinos from mu+ decay
        if do_bend:
            import muon_bending as _mb

        def _prod_frame(n_vec, h_km):
            """Local (up, north) at the production point, in the detector's frame.

            The neutrino arriving from direction ``n_vec`` (a unit vector in the
            detector's north/east/up frame) was produced at altitude ``h_km``, a
            distance L up that ray -- hundreds of km near the horizon. The local
            vertical there is tilted by ~L/R_E relative to the detector's, so the
            SAME primary direction has a less extreme local zenith at the
            production point than at the detector. Returns the production point's
            up and north unit vectors expressed in the detector frame, or None if
            the displacement is negligible.
            """
            cz_n = float(n_vec[2])
            r = RE_KM + h_km
            disc = (RE_KM * cz_n) ** 2 + (r * r - RE_KM * RE_KM)
            L = -RE_KM * cz_n + np.sqrt(max(disc, 0.0))
            if L < 1.0:
                return None
            # detector at (0,0,RE) in its own frame; P = detector + L * n
            P = np.array([L * n_vec[0], L * n_vec[1], RE_KM + L * n_vec[2]])
            up_p = P / np.linalg.norm(P)
            # local north at P: component of the detector's north orthogonal to up_p
            north_det = np.array([1.0, 0.0, 0.0])
            north_p = north_det - np.dot(north_det, up_p) * up_p
            nn = np.linalg.norm(north_p)
            if nn < 1e-9:
                return None
            return up_p, north_p / nn

        def _local_angles(vec, frame):
            """(zenith, azimuth) [deg] of ``vec`` in a production-point frame."""
            up_p, north_p = frame
            east_p = np.cross(up_p, north_p)
            cz = float(np.dot(vec, up_p))
            th = np.degrees(np.arccos(np.clip(cz, -1.0, 1.0)))
            ph = np.degrees(np.arctan2(np.dot(vec, east_p),
                                       np.dot(vec, north_p))) % 360.0
            return th, ph

        def rc_at(th_deg, ph_deg, zf=None, af=None, rf=None):
            # bilinear, periodic azimuth, clamp zenith; (zf, af, rf) default to
            # the detector-anchored map and are overridden per arrival direction
            # by the production-point-anchored one when rc_family is given.
            zf = zen_f if zf is None else zf
            af = az_f if af is None else af
            rf = rc_f if rf is None else rf
            th = np.clip(th_deg, zf[0], zf[-1])
            ph = ph_deg % 360.0
            iz = np.clip(np.searchsorted(zf, th) - 1, 0, len(zf) - 2)
            ja = np.clip(np.searchsorted(af, ph) - 1, 0, len(af) - 2)
            tz = (th - zf[iz]) / (zf[iz + 1] - zf[iz])
            ta = (ph - af[ja]) / (af[ja + 1] - af[ja])
            return (
                rf[iz, ja] * (1 - tz) * (1 - ta)
                + rf[iz + 1, ja] * tz * (1 - ta)
                + rf[iz, ja + 1] * (1 - tz) * ta
                + rf[iz + 1, ja + 1] * tz * ta
            )

        Geff = {s: np.zeros((len(cos_zeniths), len(azimuths), len(self.e)))
                for s in SPECIES}
        for iz, cz in enumerate(cos_zeniths):
            for ia, azd in enumerate(azimuths):
                single = {s: _interp_rc(rc_map[iz, ia], rc_grid, G_by_z[iz][s])
                          for s in SPECIES}
                if cz < 0:  # up-going: single far-side cutoff (no cone yet)
                    for s in SPECIES:
                        Geff[s][iz, ia] = single[s]
                    continue
                th = np.arccos(np.clip(cz, -1, 1))
                # per-direction channel cone (zenith-dependent f_mu / cone width)
                if channel_cone:
                    W_mu, fmu = _channel_at_zenith(np.degrees(th))
                phi = np.radians(azd)
                n = np.array([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi),
                              np.cos(th)])  # (north, east, up)
                # production-point frame: the cutoff seen by the parent primary is
                # the one at the PRODUCTION point, not at the detector.
                pframe = _prod_frame(n, h_prod_km) if prod_displacement else None
                zf = af = rf = None
                if rc_family is not None:  # cutoff_anchor="prod_point"
                    dn_km, de_km = prod_point_offset_km(np.degrees(th), azd,
                                                        h_prod_km)
                    zf = np.asarray(rc_family["zen"], float)
                    af = np.asarray(rc_family["az"], float)
                    rf = interp_site_rc(rc_family, dn_km, de_km)
                    pframe = prod_frame_exact(cz, azd, h_prod_km, site_lat)
                e1 = np.array([0.0, 0.0, 1.0]) - n[2] * n  # toward vertical
                nn = np.linalg.norm(e1)
                e1 = e1 / nn if nn > 1e-9 else np.array([1.0, 0.0, 0.0])
                e2 = np.cross(n, e1)
                # charge-signed bending shift of the mu-decay cone axis, in the
                # cone frame (N,E,U); mu+ primary center = n - d, mu- = n + d.
                d_cone = None
                if do_bend:
                    v = _mb.muon_velocity_enu(np.degrees(th), azd)
                    d0 = _mb.bending_deflection(v, b_enu, charge=+1)  # rad, (E,N,U)
                    d_cone = np.array([d0[1], d0[0], d0[2]])  # -> (N,E,U)
                num = {s: np.zeros(len(self.e)) for s in SPECIES}
                num_mu = {s: np.zeros(len(self.e)) for s in SPECIES}
                # Retained (unblocked) cone weight, per energy. Identically 1 when
                # nothing is blocked, so the legacy path is bit-for-bit unchanged.
                den = np.zeros(len(self.e))
                den_mu = {s: np.zeros(len(self.e)) for s in SPECIES}

                def _angles(vec):
                    if pframe is not None:
                        return _local_angles(vec, pframe)
                    return (np.degrees(np.arccos(np.clip(vec[2], -1, 1))),
                            np.degrees(np.arctan2(vec[1], vec[0])))

                for ka, a in enumerate(alpha):
                    gbar = {s: np.zeros(len(self.e)) for s in SPECIES}
                    gbar_mu = ({s: np.zeros(len(self.e)) for s in SPECIES}
                               if do_bend else None)
                    n_ok = 0
                    n_ok_mu = {s: 0 for s in SPECIES}
                    for b in beta:
                        npv = np.cos(a) * n + np.sin(a) * (
                            np.cos(b) * e1 + np.sin(b) * e2)
                        th_p, ph_p = _angles(npv)
                        # Earth shadow: at the production point no primary can
                        # arrive from below the local horizon (offaxis_mc sends
                        # p->0 there). Blocked samples leave BOTH sums.
                        if not (block_sublimb and th_p > 90.0):
                            n_ok += 1
                            rcp = float(rc_at(th_p, ph_p, zf, af, rf))
                            for s in SPECIES:
                                gbar[s] += _interp_rc(rcp, rc_grid, G_by_z[iz][s])
                        if do_bend:  # bending-shifted mu-decay samples per charge
                            # primary center = n + delta (mu+), n - delta (mu-):
                            # neutrino n = primary - delta, so primary = n + delta.
                            pp = npv + d_cone
                            pp = pp / np.linalg.norm(pp)
                            pm = npv - d_cone
                            pm = pm / np.linalg.norm(pm)
                            th_pp, ph_pp = _angles(pp)
                            th_pm, ph_pm = _angles(pm)
                            for s in SPECIES:
                                th_s, ph_s = ((th_pp, ph_pp) if s in MU_PLUS
                                              else (th_pm, ph_pm))
                                if block_sublimb and th_s > 90.0:
                                    continue
                                n_ok_mu[s] += 1
                                gbar_mu[s] += _interp_rc(
                                    float(rc_at(th_s, ph_s, zf, af, rf)), rc_grid,
                                    G_by_z[iz][s])
                    # same inner cone samples, two alpha-weights (pion + mu-decay)
                    den += W[:, ka] * (n_ok / n_beta)
                    for s in SPECIES:
                        num[s] += W[:, ka] * gbar[s] / n_beta
                        if channel_cone:
                            g_src = gbar_mu[s] if do_bend else gbar[s]
                            nk = n_ok_mu[s] if do_bend else n_ok
                            num_mu[s] += W_mu[:, ka] * g_src / n_beta
                            den_mu[s] += W_mu[:, ka] * (nk / n_beta)
                for s in SPECIES:
                    # <G> conditional on the primary direction being reachable:
                    # divide by the retained cone weight, not by 1. The production
                    # lost to the blocked wedge is already carried by E_off.
                    g_pi = np.where(den > 1e-6, num[s] / np.maximum(den, 1e-300),
                                    single[s])
                    if channel_cone:
                        d_mu = den_mu[s]
                        g_mu = np.where(d_mu > 1e-6,
                                        num_mu[s] / np.maximum(d_mu, 1e-300),
                                        single[s])
                        # blend the two cone widths by the muon-decay flux fraction
                        Geff[s][iz, ia] = (1.0 - fmu[s]) * g_pi + fmu[s] * g_mu
                    else:
                        Geff[s][iz, ia] = g_pi
        return Geff

    # -- the JOINT production x cutoff cone integral (replaces E_off x <G>) --
    def joint_prod(self, cache_dir=None):
        """Depth-resolved production profile + curved geometry for the joint cone.

        Thin memoised wrapper on :func:`joint_cone.load_production`, which calls
        ``offaxis_mc.production_profile`` / ``slant_depth_table`` -- the SAME
        ingredients ``offaxis_mc`` uses to build ``offaxis_excess.npz``, so the
        joint factor reduces to that table exactly when ``G == 1``.

        The profile is SIBYLL-2.3d / H3a (``offaxis_mc.TAG``) like the delivered
        table, independently of this engine's primary model; E_off is a
        primary-insensitive ratio (same caveat ``offaxis_factor`` warns about).
        """
        got = getattr(self, "_joint_prod", None)
        if got is not None:
            return got
        import joint_cone as jc
        import offaxis_mc as ox

        cdir = cache_dir or _CACHE_DIR
        os.makedirs(cdir, exist_ok=True)
        path = os.path.join(cdir, f"jointprod_chan_{ox.TAG}.npz")
        got = self._joint_prod = jc.load_production(
            cache=path, species=ox.CHANNEL_SPECIES)
        return got

    def joint_cone_widths(self, ep_grid, moments=None, cache_dir=None):
        """``{'pi','k','mu_numu','mu_nue'}`` space-angle RMS [deg] for the three
        production cones -- ``offaxis_mc.channel_cone_widths`` (offaxis_mc.py:559),
        memoised (it runs a 4M-event decay Monte Carlo per flavour).

        ``bending=False``: the charge-signed in-flight muon bend is applied here
        as a **coherent axis shift** of the muon channel's cutoff lookup (see
        :meth:`cone_geff`), so folding it into the cone *width* as well would
        double-count it.
        """
        import offaxis_mc as ox
        from kinematic_kernel import _MOMENTS as _KK_MOMENTS

        # Cache key: the ACTUAL moment files behind the widths, not "was an
        # override passed" -- ``moments=None`` resolves to ``kinematic_kernel.
        # _MOMENTS``, whose default changed old -> v2 on 2026-09-04, so a
        # presence-based tag would have served stale (pre-arcsin) widths out of
        # ``.cache3d`` for every default call.
        mom = _KK_MOMENTS if moments is None else moments
        tag = "|".join(f"{k}={','.join(mom[k])}" for k in sorted(mom))
        memo = getattr(self, "_joint_widths", None)
        if memo is None:
            memo = self._joint_widths = {}
        if tag in memo:
            return memo[tag]
        cdir = cache_dir or _CACHE_DIR
        os.makedirs(cdir, exist_ok=True)
        h = hashlib.md5(f"{tag}|{np.asarray(ep_grid).tobytes()}".encode())
        path = os.path.join(cdir, f"conewidths_{h.hexdigest()[:16]}.npz")
        if os.path.exists(path):
            d = np.load(path)
            memo[tag] = {k: d[k] for k in ("pi", "k", "mu_numu", "mu_nue")}
            return memo[tag]
        w = ox.channel_cone_widths(ep_grid, moments=moments, bending=False)
        np.savez(path, **w)
        memo[tag] = w
        return w

    def joint_cone_factor(self, cos_zeniths, azimuths, rc_grid, G_by_z, fine,
                          Eoff, Geff, n_alpha=44, n_beta=18, n_ray=260,
                          sigma_scale=1.0, channel_cone=True, muon_bending=True,
                          b_enu=None, h_prod_km=H_PROD_KM, prod_frame=True,
                          moments=None, cache_dir=None, joint_channels=True,
                          rc_family=None, site_lat=None):
        """``{species: F[cz, az, E]}`` -- the joint 3D factor replacing
        ``E_off(E,cosZ) x <G_s>_cone(E,n)`` in :meth:`solve`.

        The delivered engine averages the production profile and the cutoff
        suppression over the SAME production cone but in TWO separate integrals,

            E_off = <p>_cone / p_axis        (``offaxis_mc.cone_numden``)
            <G_s>_cone                       (:meth:`cone_geff`)

        and multiplies them.  That drops ``Cov_cone(p, G_s)``, which is zero at
        the vertical and large at the horizon: 30-43% of the cone weight at
        87 deg is Earth-shadowed and carries ~4-7% of the production, yet it
        enters ``<G>`` with full weight.  This method evaluates the single object

            F_s = J_s / p_axis,
            J_s = Int dl rho(h) Int dOmega_p W(alpha;E)
                      p(X_slant(l, psi_p), E) G_s(E, R_c(n_p)) ,

        so the Earth shadow removes a direction from the numerator *and* the
        normalisation automatically (``p -> 0`` there), and the far-side map is
        never read for a down-going cone.  See :mod:`joint_cone`.

        Channels are MCEq's own depth-resolved per-species parent categories
        ``{s}_dir`` / ``{s}_k`` / ``{s}_mu`` (``offaxis_mc.production_profile``),
        each with its own cone (``offaxis_mc.channel_cone_widths``: the direct
        ``sigma_pi``, the kaon ``sigma_K``, and the corrected muon-decay
        ``mudecay_shape_mc`` per flavour).  Because Eq. (8) is linear in ``p``,
        summing the per-channel numerators over the common denominator IS the
        production-weighted blend ``f_c = D_c / sum D`` -- depth-resolved (so
        ``f_mu`` rises toward the horizon on its own) and species-resolved,
        replacing the single ``channel_fractions`` ``f_mu[s]``.  This is exactly
        how ``offaxis_mc.build_channel`` composes ``offaxis_excess_channel_v2``,
        which is therefore the ``G == 1`` gate.  The charge-signed muon-bending
        shift displaces the muon channel's **cutoff lookup** by ``+-delta``.

        Outside the tabulated production range (0.1-100 GeV) and for **up-going**
        neutrino directions the delivered product ``Eoff * Geff`` is returned
        unchanged (up-going keeps its single far-side cutoff).

        ``rc_family`` / ``site_lat`` (``cutoff_anchor="prod_point"``): read the
        cone samples' cutoff off a map anchored at the PRODUCTION POINT rather
        than at the detector -- see :func:`prod_family_rc` and
        ``joint_cone.delivered_joint_factor``.  ``None`` (default) keeps the
        detector anchor.
        """
        import joint_cone as jc

        prod = self.joint_prod(cache_dir)
        ep = prod["ep_grid"]
        # NB ``moments`` here is the {species: files} dict of
        # ``kinematic_kernel.meson_theta2``, NOT solve()'s ``moments`` filename
        # (that one drives ``angular_factor``); default = the delivered moments.
        w = self.joint_cone_widths(ep, moments=moments, cache_dir=cache_dir)
        chan = None
        if joint_channels and channel_cone and "chan" in prod:
            chan = {s: prod["chan"][jc.CHANNEL_SPECIES_MAP[s]] for s in SPECIES}
            sig_mu_s = {s: w["mu_nue" if "nue" in s else "mu_numu"] for s in SPECIES}
        if chan is None and channel_cone:
            # legacy A/B path: one f_mu[s] from channel_fractions and the old
            # mudecay_shape width, evaluated per zenith below.
            from kinematic_kernel import mudecay_shape, channel_fractions
        MU_PLUS = ("total_antinumu", "total_nue")  # neutrinos from mu+ decay
        do_bend = channel_cone and muon_bending and b_enu is not None
        if do_bend:
            import muon_bending as _mb

        q_cache = getattr(self, "_joint_q", None)
        if q_cache is None:
            q_cache = self._joint_q = {}
        out = {s: np.array([[Eoff[s][iz] * Geff[s][iz, ia]
                             for ia in range(len(azimuths))]
                            for iz in range(len(cos_zeniths))])
               for s in SPECIES}
        inside = (self.e >= ep[0]) & (self.e <= ep[-1])
        for iz, cz in enumerate(cos_zeniths):
            if cz < 0:  # up-going: far-side single cutoff, product unchanged
                continue
            zen_deg = float(np.degrees(np.arccos(np.clip(cz, -1, 1))))
            sig_mu = fmu = None
            if chan is None and channel_cone:
                zkey = int(round(zen_deg / 10.0) * 10)
                sig_mu = np.maximum(mudecay_shape(ep, zenith_deg=float(zkey)), 1e-3)
                fmu = {s: channel_fractions(ep, SP_LABEL[s],
                                            zenith_deg=float(zkey))["mu"]
                       for s in SPECIES}
            d_cone = None
            Gz = {s: ox_regrid(G_by_z[iz][s], self.e, ep) for s in SPECIES}
            for ia, azd in enumerate(azimuths):
                if do_bend:
                    v = _mb.muon_velocity_enu(zen_deg, float(azd))
                    d0 = _mb.bending_deflection(v, b_enu, charge=+1)  # (E,N,U)
                    d_cone = np.array([d0[1], d0[0], d0[2]])  # -> (N,E,U)
                res = jc.delivered_joint_factor(
                    cz, [azd], prod, fine, rc_grid, Gz,
                    sigma_pi=w["pi"], sigma_k=w["k"],
                    sigma_mu=sig_mu, fmu=fmu,
                    channels_by_species=chan,
                    sigma_mu_by_species=(sig_mu_s if chan else None),
                    d_cone=d_cone, mu_plus=MU_PLUS, prod_frame=prod_frame,
                    h_prod_km=h_prod_km, n_alpha=n_alpha, n_beta=n_beta,
                    n_ray=n_ray, sigma_scale=sigma_scale, q_cache=q_cache,
                    rc_family=rc_family, site_lat=site_lat,
                )
                for s in SPECIES:
                    f = np.interp(np.log(self.e[inside]), np.log(ep),
                                  res["F"][s][0])
                    out[s][iz, ia, inside] = f
        return out

    def cutoff_grid(
        self,
        lat,
        lon,
        cos_zeniths,
        azimuths,
        date,
        n_scan=None,
        r_lo=0.5,
        r_hi=RC_MAX_GV,
        cache_dir=None,
        n_jobs=None,
    ):
        """R_c[cosZ, az] [GV]: detector cutoff (down-going), far-side (up-going).

        ``r_hi`` spans to :data:`RC_MAX_GV` (55 GV).  The previous 40 GV ceiling
        SILENTLY SATURATED: cached maps contained literal 40.00 GV cells near the
        horizon in the East, where a direct back-trace gives 40.5-49.5 GV, so the
        map was flat (zero azimuthal gradient) exactly where the East-West signal
        is generated.  Saturation is now warned about rather than hidden.

        Resolution: the coarse-ladder + bisection scan of
        :func:`geomag_backtrace.scan_upper_cutoff` (~0.05 GV).  The old default
        ``n_scan=12`` here was a 3.6 GV step -- coarse enough to step over the
        topmost forbidden band entirely.  ``n_scan`` is kept only as a
        backward-compatible way to *refine* the coarse step.

        Both hemispheres use the same batched, forked back-trace. For up-going,
        the primary's velocity at the far-side production point equals the
        (straight-line) neutrino direction ``d``, so the cutoff is a back-trace
        from the production point ``Q`` with ``u0 = -d`` -- the global geomagnetic
        treatment, batched like :func:`geomag_backtrace.cutoff_map`.

        This is the dominant cost (trajectory integration). With ``cache_dir`` set
        the resulting map is memoised to ``<cache_dir>/rc_*.npz``, keyed by site,
        date, grid, scan parameters and scan **scheme**, so repeat evaluations are
        ~instant and stale-scheme maps are never reused.
        """
        import geomag_backtrace as gb

        fpath = None
        if cache_dir is not None:
            dtag = date.isoformat() if hasattr(date, "isoformat") else str(date)
            cz_b = np.asarray(cos_zeniths).tobytes()
            az_b = np.asarray(azimuths).tobytes()
            tag = (f"{lat:.4f}_{lon:.4f}_{dtag}_{cz_b}_{az_b}_{n_scan}_{r_lo}_{r_hi}"
                   f"_{gb.CUTOFF_SCHEME}")
            h = hashlib.md5(tag.encode()).hexdigest()[:16]
            fpath = os.path.join(cache_dir, f"rc_{h}.npz")
            if os.path.exists(fpath):
                d = np.load(fpath)
                if d["rc"].shape == (len(cos_zeniths), len(azimuths)):
                    return d["rc"]

        rc = np.zeros((len(cos_zeniths), len(azimuths)))
        down = cos_zeniths >= 0
        if np.any(down):
            zen = np.degrees(np.arccos(np.clip(cos_zeniths[down], 1e-3, 1)))
            rc[down] = gb.cutoff_map(
                lat, lon, date, zen, azimuths, n_scan=n_scan, r_lo=r_lo, r_hi=r_hi,
                n_jobs=n_jobs, warn_saturated=False,
            )
        ups = np.where(~down)[0]
        if len(ups):
            zen_up = np.degrees(np.arccos(np.clip(cos_zeniths[ups], -1, 1)))
            out_up = farside_cutoff_map(
                lat, lon, date, zen_up, azimuths, n_scan=n_scan, r_lo=r_lo,
                r_hi=r_hi, n_jobs=n_jobs,
            )
            for k, iz in enumerate(ups):
                rc[iz] = out_up[k]
        n_sat = int(np.sum(np.isclose(rc, r_hi)))
        if n_sat:
            import warnings

            warnings.warn(
                f"cutoff_grid: {n_sat} of {rc.size} cells saturated at the "
                f"r_hi={r_hi:g} GV scan ceiling -- the cutoff map is flat there. "
                "Raise r_hi.",
                stacklevel=2,
            )
        if fpath is not None:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez(fpath, rc=rc)
        return rc

    def solve(
        self,
        lat,
        lon,
        cos_zeniths,
        azimuths,
        date=None,
        n_scan=None,
        rc_grid=None,
        zenith_dependent_geomag=True,
        use_cache=False,
        cache_dir=None,
        solar_modulation=0.0,
        with_calib_error=False,
        calib_hadronic_only=False,
        with_calib_jacobian=False,
        full_3d=False,
        moments="m_spliced.npz",
        offaxis=True,
        offaxis_shape_only=None,
        offaxis_table=None,
        with_eoff_jacobian=False,
        solar_sigma_gv=0.0,
        with_base_spread=False,
        cone_cutoff=True,
        cone_sigma_scale=1.0,
        channel_cone=True,
        muon_bending=True,
        prod_displacement=False,
        sublimb="prod_point",
        cone_kernel="moments",
        cone_moments=None,
        joint_cone=None,
        joint_channels=True,
        sigma_lnr=None,
        cutoff_anchor=None,
        n_jobs=None,
    ):
        """Absolute Phi[species, cosZ, az, E] for a site (full sky).

        Down-going (cosZ>=0): detector geomagnetic cutoff. Up-going (cosZ<0):
        far-side production-point cutoff (global treatment, :func:`farside_production`).

        ``zenith_dependent_geomag`` (default **True**): recompute the suppression
        ratio ``G_s(E,R_c)`` at *every* zenith band, so no slant-depth
        approximation enters. G_s is site-independent and cached per (cz_ref,
        rc_grid), so the extra cost (one cascade pair per unique |cosZ|) is a
        one-time precompute. Set False to reuse the vertical-column G_s at all
        zeniths -- validated zenith-independent to <=2% (sub-GeV horizon) by
        `geomag_zenith_check.py` -- for a faster first (uncached) evaluation.

        ``use_cache``: memoise the two heavy ingredients to ``cache_dir`` (default
        ``flux_cache/`` next to this module) -- the **site-independent** ``G_s`` and
        the **per-site** cutoff map. The first call to a site pays the full cost;
        repeat calls (and other sites, for ``G_s``) load from disk in milliseconds.
        Matches the default path to interpolation precision (a fixed wide ``rc_grid``
        is used so one cached ``G_s`` serves every site without clamping). Off by
        default so the validated path is untouched.

        ``solar_modulation`` [GV]: force-field modulation potential applied to the
        primary (`solar_factor`), rescaling the flux toward low energy; 0 (default) =
        H3a baseline. The physical solar-cycle span is the *difference* between two
        potentials (~0.4 GV solar-min vs ~1.0 GV solar-max).

        ``with_calib_error`` (daemonflux base only): also return the propagated
        muon-**calibration** 1-sigma uncertainty from daemonflux's nuisance-parameter
        covariance -- ``result["flux_err"]`` (absolute) and ``["flux_relerr"]``
        (fractional). Since ``G`` and ``S`` are parameter-independent, the fractional
        error carries through unchanged. ``calib_hadronic_only`` isolates the
        hadronic-production component.

        ``offaxis`` (recommended for 3D): fold in the first-principles off-axis
        3D-production factor ``E_off(E, cosZ)`` (`offaxis_factor`) -- the complete
        genuine-3D/1D production ratio, which both redistributes flux in zenith and
        produces the sub-GeV near-horizon excess (horizon/vertical ~1.8 at 0.3 GeV).
        Derived from first principles (MCEq production + curved geometry + decay
        kinematics; no reference flux) and validated against Honda/Bartol to their
        mutual ~5-15%. This supersedes ``full_3d`` (the two must not be combined --
        that double-counts the production angle).

        Since 2026-09-04 ``offaxis`` defaults to **True**: a bare ``solve()`` is
        the delivered 3D physics (``offaxis=True``, ``cone_cutoff=True`` ->
        ``joint_cone=True``, ``joint_channels=True``, ``cone_kernel="moments"``
        on the arcsin-corrected v2 moments, ``sublimb="prod_point"``,
        ``sigma_lnr=SIGMA_LNR=0``).  Pass ``offaxis=False`` for the fast
        geomagnetic-only path (it also disables the joint cone, so it is the
        cheapest evaluation; it is NOT the delivered flux).

        ``full_3d`` (legacy): fold in only the flux-conserving production-angle
        **redistribution** ``R(E, cosZ)`` (`angular_factor`) -- the ~1-2% sub-GeV
        zenith redistribution without the net horizontal excess. Kept for
        comparison; use ``offaxis`` for the complete 3D flux. ``full_3d=False``
        and ``offaxis=False`` is the pure factorised (geomagnetic-only) path.

        ``offaxis_table``: path override for the tabulated ``E_off``.  ``None``
        (default) = :data:`EOFF_TABLE` (``offaxis_excess_channel_v2.npz``, the
        species-resolved channel-weighted v2 build).  Pass
        :data:`EOFF_TABLE_FLAT` for the pre-2026-09-04 flavour-blind pion-only
        July table (A/B only).  This factor is what multiplies the base wherever
        the joint cone does not apply -- ``cone_cutoff=False``, the **up-going**
        hemisphere, and energies outside the tabulated 0.1-100 GeV -- and it
        drives the ``sigma_pi_NA61`` pull; with the default it is the same
        physics the joint integral evaluates (identical at ``G == 1``).

        ``offaxis_shape_only``: opt-in E_off/E_off(vertical) instead of the full
        factor (default ``None`` -> ``False``, i.e. the full factor on every
        base). daemonflux's neutrino flux is genuinely 1D, and the build-time
        muon closure (E_off with the muon kernel ~1 for E_mu >= 5 GeV,
        daemonflux's calibration region) shows the full 1D->3D factor does not
        double-count the calibration -- it tracks Honda better than shape-only
        (vertical sub-GeV ~1.0x Honda vs ~1.07x). Kept only for A/B studies.

        Additional nuisance pulls (appended to ``calib_params/corr/jac`` so
        `calib_covariance` carries the **full** uncertainty, and folded into
        ``flux_relerr``/``flux_err`` in quadrature when ``with_calib_error``):
        ``with_eoff_jacobian`` -- the hadronic E_off pull ``sigma_pi_NA61``
        (NA61 +-12% pion production angle; ~+-8% on the sub-GeV horizon excess);
        ``solar_sigma_gv`` -- a +1-sigma solar-potential pull of this size [GV];
        ``with_base_spread`` -- the fully-correlated base-model-choice pull
        (half log-spread MCEq vs daemonflux; the dominant sub-GeV systematic).

        ``cone_cutoff`` (default **True**): apply the geomagnetic factor as a
        **production-cone average** (:meth:`cone_geff`) rather than a single cutoff
        at the neutrino direction -- the geomagnetic analogue of E_off, and the
        physically-required treatment (it restores the near-horizon East-West
        asymmetry; Section 4.1). It needs a **full-sphere** cutoff map
        (:meth:`finemap_rc`, down-going detector cutoff + up-going far-side
        cutoff), memoised on the instance and cached to ``cache_dir`` per
        site/date, so the ~one-off back-trace cost is paid once. The full-sphere
        map lets a near-horizon down-going cone extend continuously across the
        limb (removing the extreme-horizon E-W overshoot; Section 6 v).
        Set ``cone_cutoff=False`` for the **fast single-cutoff approximation**
        (adequate away from the horizon, where the cone average -> the single
        cutoff). Up-going *neutrino* directions keep the single far-side cutoff
        either way (their own cone extension is a further refinement).

        ``sublimb`` (default ``"prod_point"``): how the part of a near-horizon
        down-going production cone that falls below the detector's local horizon
        is treated -- in the local frame of the **production point** (~20-30 km up
        the arrival ray), with genuinely Earth-shadowed samples blocked, rather
        than by reading the **antipodal** far-side cutoff (``"farside"``, legacy),
        which put a 9.5 GV discontinuity into the map at 89/90 deg. See
        :meth:`cone_geff`.

        ``cone_kernel`` (default ``"moments"``): the pion angular distribution
        driving the geomagnetic cone -- the NA61-validated moment ``sigma_pi``
        Gaussian with the per-axis ``sigma/sqrt(2)``, i.e. the SAME cone
        ``offaxis_mc`` uses for E_off.  ``"sampled"`` is the legacy
        ``k_spliced.npz`` kernel (16-37% too wide).  See :meth:`cone_geff`.

        ``cone_moments``: ``{species: [files]}`` override for the generator
        production-angle moments behind the cone width.  ``None`` (default) =
        ``kinematic_kernel._MOMENTS``, which since 2026-09-04 IS the
        arcsin-corrected ``m_*_v2.npz`` set; pass
        ``kinematic_kernel._MOMENTS_LEGACY`` for the pre-arcsin A/B.

        ``joint_cone`` (default ``None`` -> **on** whenever ``offaxis`` and
        ``cone_cutoff`` are both set, which is the recommended configuration):
        replace the product ``E_off x <G_s>_cone`` by the **single joint
        integral** ``J_s / p_axis`` (:meth:`joint_cone_factor`), which restores
        the cone covariance ``Cov_cone(p, G_s)`` the product drops.  It costs
        ~1.2x a cone-average solve with warm caches.  Pass ``False`` for the
        legacy factorised product.  ``joint_channels=False`` is an A/B switch
        back to the single ``channel_fractions`` ``f_mu[s]`` blend with the old
        ``mudecay_shape`` width, instead of MCEq's depth-resolved per-species
        parent-channel profiles.

        ``sigma_lnr``: penumbra width in ``ln R`` of the cutoff transmission
        (Eq. 7) behind ``G_s``; ``None`` = the measured default
        :data:`SIGMA_LNR`.  It is part of the ``G_s`` cache key.

        ``cutoff_anchor`` (``None`` -> :data:`CUTOFF_ANCHOR`, currently
        ``"detector"``): WHERE the primary whose cutoff is evaluated enters the
        atmosphere.  ``"detector"`` is the historical behaviour -- every cone
        sample of every arrival direction is charged the cutoff of a trajectory
        launched at the detector, with ``sublimb="prod_point"`` /
        ``prod_frame=True`` rotating only the local VERTICAL to the production
        point.  ``"prod_point"`` launches at the production point itself: the
        cone samples are read off a cutoff map built at
        ``prod_site_latlon(...)``, interpolated from the displaced-site family
        :meth:`prod_family_rc`, in the production point's true local frame
        (:func:`prod_frame_exact`).  The two agree exactly at the vertical
        (zero displacement) and differ by 12-16% in ``R_c`` at the limb, where
        the production point is 370-610 km away.  The family costs 8 extra
        down-going back-traced maps per site/date, cached next to the detector
        map; ``cutoff_anchor="detector"`` never builds it.

        ``n_jobs``: forked workers for the one-off cutoff back-traces (default
        ``os.cpu_count()//2``); ignored once the maps are cached.
        """
        if cutoff_anchor is None:
            cutoff_anchor = CUTOFF_ANCHOR
        if cutoff_anchor not in ("detector", "prod_point"):
            raise ValueError("cutoff_anchor must be 'detector' or 'prod_point', "
                             f"got {cutoff_anchor!r}")
        if cutoff_anchor == "prod_point" and not cone_cutoff:
            raise ValueError("cutoff_anchor='prod_point' needs cone_cutoff=True: "
                             "it re-anchors the CONE samples' cutoff, and the "
                             "single-cutoff path has no cone to re-anchor")
        if joint_cone is None:  # default ON wherever it is defined
            joint_cone = bool(offaxis and cone_cutoff)
        if joint_cone and not (offaxis and cone_cutoff):
            raise ValueError("joint_cone=True requires offaxis=True and "
                             "cone_cutoff=True (it replaces their product)")
        if offaxis and full_3d:
            raise ValueError(
                "offaxis supersedes full_3d (E_off already contains the "
                "production-angle redistribution); set only one."
            )
        cos_zeniths = np.asarray(cos_zeniths, float)
        azimuths = np.asarray(azimuths, float)
        base = self.base(cos_zeniths)

        import datetime as _dt

        date = date or _dt.datetime(2020, 1, 1)
        if use_cache and cache_dir is None:
            cache_dir = _CACHE_DIR
        rc_map = self.cutoff_grid(
            lat, lon, cos_zeniths, azimuths, date, n_scan, cache_dir=cache_dir,
            n_jobs=n_jobs,
        )

        # Build the G(R_c) interpolation grid to *span the actual cutoff map*, so the
        # suppression is never clamped: a fixed floor (e.g. 2 GV) would apply spurious
        # suppression to low-cutoff directions/sites (polar R_c<2 GV) where G->1.
        if rc_grid is None:
            if use_cache:
                # fixed wide grid -> one cached G_s is reusable across all sites.
                # Spans to RC_MAX_GV: the near-horizon East cutoff at Kamioka
                # peaks at ~49.5 GV (measured by direct back-trace at zenith 89
                # deg, azimuth 75 deg), so the previous 40 GV ceiling clamped
                # G_s(R_c) exactly where the East-West signal is generated.
                rc_grid = np.linspace(0.1, RC_MAX_GV, 40)
            else:
                lo = max(0.1, float(np.min(rc_map)) * 0.9)
                hi = max(lo + 0.5, float(np.max(rc_map)) * 1.05)
                rc_grid = np.linspace(lo, hi, 12)
        if zenith_dependent_geomag:
            G_by_z = []
            for cz in cos_zeniths:
                Gz, rc_grid = self.geomag_response(
                    rc_grid, cz_ref=max(abs(cz), 1e-3), cache_dir=cache_dir,
                    sigma_lnr=sigma_lnr,
                )
                G_by_z.append(Gz)
        else:
            G0, rc_grid = self.geomag_response(rc_grid, cache_dir=cache_dir,
                                               sigma_lnr=sigma_lnr)
            G_by_z = [G0] * len(cos_zeniths)

        # solar modulation: neutrino-energy factor S(E), applied to all species/dirs
        smod = self.solar_factor(solar_modulation) if solar_modulation else 1.0
        # legacy flux-conserving production-angle redistribution R[species][cosZ,E]
        R3d = self.angular_factor(cos_zeniths, moments=moments) if full_3d else None
        # first-principles complete off-axis 3D-production factor E_off[cosZ,E].
        # The FULL factor is applied on every base (default). daemonflux's
        # neutrino flux is genuinely 1D, and the muon-calibration region
        # (E_mu >~ 5 GeV) has E_off_mu ~ 1 (build-time closure check), so the
        # full 1D->3D factor is self-consistent with the calibration and does
        # not double-count -- verified to track Honda better than the earlier
        # shape-only heuristic (kept only as an opt-in for A/B studies).
        if offaxis_shape_only is None:
            offaxis_shape_only = False
        Eoff = (
            self.offaxis_factor(cos_zeniths, path=offaxis_table,
                                shape_only=offaxis_shape_only)
            if offaxis
            else None
        )

        # geomagnetic factor: single cutoff at the neutrino direction, or the
        # production-cone average (cone_cutoff, the geomagnetic analogue of E_off)
        Geff = None
        rc_family = None
        if cone_cutoff:
            fine = self.finemap_rc(lat, lon, date, cache_dir=cache_dir,
                                   n_jobs=n_jobs)
            rc_family = (
                self.prod_family_rc(lat, lon, date, cache_dir=cache_dir,
                                    n_jobs=n_jobs)
                if cutoff_anchor == "prod_point" else None
            )
            b_enu = None
            if channel_cone and muon_bending:
                from muon_bending import local_field_enu
                b_enu = local_field_enu(lat, lon, date)
            Geff = self.cone_geff(
                cos_zeniths, azimuths, rc_map, rc_grid, G_by_z, fine,
                sigma_scale=cone_sigma_scale, channel_cone=channel_cone,
                muon_bending=muon_bending, b_enu=b_enu,
                prod_displacement=prod_displacement, sublimb=sublimb,
                cone_kernel=cone_kernel, cone_moments=cone_moments,
                rc_family=rc_family, site_lat=lat,
            )

        # joint production x cutoff cone integral: ONE factor replacing
        # E_off * <G>_cone (see joint_cone_factor). Down-going only; up-going
        # rows come back as the unchanged product.
        Fjoint = None
        if joint_cone:
            Fjoint = self.joint_cone_factor(
                cos_zeniths, azimuths, rc_grid, G_by_z, fine, Eoff, Geff,
                sigma_scale=cone_sigma_scale, channel_cone=channel_cone,
                muon_bending=muon_bending, b_enu=b_enu, cache_dir=cache_dir,
                moments=cone_moments, joint_channels=joint_channels,
                rc_family=rc_family, site_lat=lat,
            )

        flux = {
            s: np.zeros((len(cos_zeniths), len(azimuths), len(self.e))) for s in SPECIES
        }
        for s in SPECIES:
            r3 = R3d[s] if R3d is not None else None
            eo = Eoff[s] if Eoff is not None else None
            for ia in range(len(azimuths)):
                for iz in range(len(cos_zeniths)):
                    if Fjoint is not None:
                        # ONE factor: J_s/p_axis already contains E_off and G
                        flux[s][iz, ia] = base[s][iz] * Fjoint[s][iz, ia] * smod
                        continue
                    if Geff is not None:
                        g = Geff[s][iz, ia]
                    else:
                        g = _interp_rc(rc_map[iz, ia], rc_grid, G_by_z[iz][s])
                    f = base[s][iz] * g * smod
                    if r3 is not None:
                        f = f * r3[iz]
                    if eo is not None:
                        f = f * eo[iz]
                    flux[s][iz, ia] = f
        result = dict(
            e=self.e,
            cos_zeniths=cos_zeniths,
            azimuths=azimuths,
            flux=flux,
            base=base,
            cutoff=rc_map,
        )
        if with_calib_error:
            if self.base_model != "daemonflux":
                raise ValueError("with_calib_error requires base_model='daemonflux'")
            relerr = self._daemonflux_relerr(
                cos_zeniths, only_hadronic=calib_hadronic_only
            )
            result["flux_relerr"] = relerr  # sigma/Phi [species, cosZ, E]
        if with_calib_jacobian:
            cj = self.calib_jacobian(cos_zeniths)
            result["calib_params"] = cj["params"]  # 24 nuisance-parameter names
            result["calib_corr"] = cj["corr"]  # parameter correlation (n_par, n_par)
            result["calib_cov"] = cj["cov"]
            result["calib_jac"] = cj[
                "jac"
            ]  # R[species]=dPhi/Phi per +1sig [par,cosZ,E]

        # -- extra nuisance pulls (uncorrelated with the daemonflux calibration) --
        # Each is a fractional flux response R[species][cosZ, E] to a +1-sigma pull,
        # appended to calib_params/corr/jac so calib_covariance() carries the FULL
        # uncertainty (calibration + hadronic E_off + solar + base spread).
        extras = []  # (name, R[species])
        if with_eoff_jacobian:
            if not offaxis:
                raise ValueError("with_eoff_jacobian requires offaxis=True")
            E_hi = self.offaxis_factor(
                cos_zeniths, path=offaxis_table,
                shape_only=offaxis_shape_only, which="E_off_hi"
            )
            r = {
                s: np.where(Eoff[s] > 0, E_hi[s] / Eoff[s] - 1.0, 0.0)
                for s in SPECIES
            }
            extras.append(("sigma_pi_NA61", r))
        if solar_sigma_gv:
            s_ref = smod if solar_modulation else np.ones(len(self.e))
            s_up = self.solar_factor(solar_modulation + solar_sigma_gv)
            r1 = np.where(s_ref > 0, s_up / s_ref - 1.0, 0.0)[None, :] * np.ones(
                (len(cos_zeniths), 1)
            )
            extras.append(("solar_phi", {s: r1 for s in SPECIES}))
        if with_base_spread:
            # evaluate the *other* base at the same zeniths; the half log-spread
            # is a single fully-correlated model-choice pull (base_comparison).
            if self.base_model == "daemonflux":
                saved = self.base_model
                self.base_model = "mceq"
                try:
                    other = self.base(cos_zeniths)
                finally:
                    self.base_model = saved
            else:
                if self._df is None:
                    from daemonflux import Flux

                    self._df = Flux(location="generic")
                other = self._base_daemonflux(cos_zeniths)
            r = {}
            for s in SPECIES:
                with np.errstate(invalid="ignore", divide="ignore"):
                    r[s] = np.where(
                        (base[s] > 0) & (other[s] > 0),
                        0.5 * np.log(other[s] / base[s]),
                        0.0,
                    )
            extras.append(("base_model_spread", r))
        if extras:
            names = list(result.get("calib_params", []))
            n0 = len(names)
            jac = result.get(
                "calib_jac",
                {s: np.zeros((0, len(cos_zeniths), len(self.e))) for s in SPECIES},
            )
            for name, r in extras:
                names.append(name)
                for s in SPECIES:
                    jac[s] = np.concatenate([jac[s], r[s][None]], axis=0)
            corr = np.eye(len(names))
            if n0:
                corr[:n0, :n0] = result["calib_corr"]
            result["calib_params"] = names
            result["calib_corr"] = corr
            result["calib_jac"] = jac
            # total fractional error incl. extras (extras mutually uncorrelated)
            if with_calib_error:
                for s in SPECIES:
                    q = result["flux_relerr"][s] ** 2
                    for _, r in extras:
                        q = q + r[s] ** 2
                    result["flux_relerr"][s] = np.sqrt(q)
        if with_calib_error:
            result["flux_err"] = {
                s: flux[s] * result["flux_relerr"][s][:, None, :] for s in SPECIES
            }
        return result


def ox_regrid(arr, e_src, e_dst):
    """Log-E regrid of a (n_rc, nE_src) matrix onto ``e_dst`` (thin re-export of
    ``offaxis_mc._regrid``, so ``G_s`` can be read on the production grid)."""
    import offaxis_mc as ox

    return ox._regrid(arr, e_src, e_dst)


def _interp_rc(rc, rc_grid, gmat):
    """Linear interp of G at scalar ``rc`` along axis 0 of ``gmat`` (n_rc, n_e).

    Vectorised over energy (replaces a per-energy ``np.interp`` loop); clamps to the
    grid endpoints exactly like ``np.interp``.
    """
    rc = min(max(float(rc), rc_grid[0]), rc_grid[-1])
    j = int(np.searchsorted(rc_grid, rc))
    j = min(max(j, 1), len(rc_grid) - 1)
    w = (rc - rc_grid[j - 1]) / (rc_grid[j] - rc_grid[j - 1])
    return (1.0 - w) * gmat[j - 1] + w * gmat[j]


def hybrid_weight(e, e0=1.7):
    """Daemonflux weight w(E) of the ``base_model="hybrid"`` blend.

    Smooth one-octave log-transition centred at ``e0`` [GeV]: w->0 below (the
    GSF-anchored MCEq base), w->1 above (the muon-calibrated daemonflux base);
    w(e0)=0.5. Default e0=1.7 GeV is the daemonflux muon-calibration floor
    (E_mu >= 5 GeV per its README) mapped to neutrinos via E_nu ~ E_mu/3 --
    a physics-derived choice, not a fit to any reference.
    """
    return 0.5 * (1.0 + np.tanh(np.log(np.asarray(e, float) / e0) / np.log(2.0)))


def horizon_grid(n_horizon=9, n_bulk=6, cz_max=0.95, horizon_half_width=0.2):
    """Full-sky cosθ grid **refined near the horizon** (cosθ→0).

    The flux (and the sec θ enhancement) vary fastest near cosθ=0, so a uniform
    grid interpolates poorly there. This packs ``n_horizon`` points (per
    hemisphere) into |cosθ| < ``horizon_half_width`` and ``n_bulk`` into the rest,
    symmetric in up/down. Pass the result as ``cos_zeniths`` to :meth:`solve`.
    """
    bulk = np.linspace(horizon_half_width, cz_max, n_bulk)
    horiz = np.linspace(0.02, horizon_half_width, n_horizon, endpoint=False)
    down = np.unique(np.concatenate([horiz, bulk]))
    return np.unique(np.concatenate([-down[::-1], down]))


def interp_flux(result, energy_gev, cos_zenith, azimuth_deg, species="total_numu"):
    """Evaluate a solved grid at (E, cosZ, azimuth) [/(m^2 s sr GeV)].

    Trilinear in (log E, azimuth, cosZ) on the grid from :meth:`MCEq3DFlux.solve`;
    azimuth wraps modulo 360.
    """
    e, cz, az = result["e"], result["cos_zeniths"], result["azimuths"]
    f = result["flux"][species]  # (cosZ, az, E)
    le = np.log(np.maximum(energy_gev, e[0]))
    # interpolate in log E for every (cosZ, az)
    fz = np.array(
        [
            [np.interp(le, np.log(e), f[i, j]) for j in range(len(az))]
            for i in range(len(cz))
        ]
    )  # (cosZ, az)
    az_ext = np.r_[az, az[0] + 360.0]
    row = np.array(
        [
            np.interp(azimuth_deg % 360.0, az_ext, np.r_[fz[i], fz[i, 0]])
            for i in range(len(cz))
        ]
    )  # (cosZ,) at this E, azimuth
    order = np.argsort(cz)
    return float(np.interp(cos_zenith, cz[order], row[order]))


def calib_covariance(result, species="total_numu", iz=0, ia=0):
    """Full (n_E, n_E) muon-calibration covariance of the directional flux for one
    direction, from ``solve(with_calib_jacobian=True)``.

    ``Cov = A^T . corr . A`` with the absolute per-parameter gradient
    ``A = Phi * R`` (``R`` = fractional response per +1-sigma pull). ``sqrt(diag(Cov))``
    is the absolute 1-sigma; dividing by the flux recovers the fractional ``error()``.
    An oscillation fit uses the daemonflux pulls as nuisance parameters with unit
    priors and correlation ``result["calib_corr"]``, and modifies the flux by
    ``Phi * (1 + sum_i eta_i R_i)``.
    """
    R = result["calib_jac"][species][:, iz, :]  # (n_par, n_E) fractional response
    phi = result["flux"][species][iz, ia]  # (n_E,)
    A = phi[None, :] * R  # absolute gradient (n_par, n_E)
    return A.T @ result["calib_corr"] @ A


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lat", type=float, default=36.43)
    p.add_argument("--lon", type=float, default=137.31)
    p.add_argument("--validate", action="store_true", help="compare vs Honda")
    p.add_argument("--plot", action="store_true")
    p.add_argument(
        "--base",
        choices=["mceq", "daemonflux", "hybrid"],
        default="mceq",
        help="1D base the geomag factor multiplies (daemonflux = data-anchored)",
    )
    p.add_argument(
        "--offaxis", action=argparse.BooleanOptionalAction, default=True,
        help="fold in the first-principles off-axis 3D-production factor E_off "
        "(complete genuine-3D zenith shape; ON by default, --no-offaxis for the "
        "fast geomagnetic-only path)",
    )
    p.add_argument(
        "--full3d", action="store_true",
        help="legacy: flux-conserving production-angle redistribution only "
        "(superseded by --offaxis; cannot be combined)",
    )
    p.add_argument(
        "--interaction-model", default="SIBYLL23D",
        help="MCEq hadronic model for the base AND the G_s response (default "
        "SIBYLL23D, the model daemonflux is calibrated with). Offline choices "
        "in the shipped MCEq database: SIBYLL23D/23E/21, DPMJETIII193, "
        "EPOSLHC, EPOSLHCR, QGSJETII04, QGSJETIII.",
    )
    p.add_argument(
        "--primary", default="HillasGaisser2012:H3a",
        help="primary spectrum as 'crfluxClass[:tag]' (default "
        "HillasGaisser2012:H3a; the delivered hybrid configuration uses "
        "GlobalSplineFitBeta). Part of the G_s cache key.",
    )
    args = p.parse_args(argv)

    df_loc = "kamioka" if abs(args.lat - 36.43) < 1 else "generic"
    pcls, _, ptag = args.primary.partition(":")
    eng = MCEq3DFlux(
        base_model=args.base,
        daemonflux_location=df_loc,
        interaction_model=args.interaction_model,
        primary=(pcls, ptag or None),
    )
    cz = np.array([-0.95, -0.55, -0.15, 0.15, 0.55, 0.95])  # full sky
    az = np.array([0, 45, 90, 135, 180, 225, 270, 315], float)
    r = eng.solve(
        args.lat, args.lon, cz, az, full_3d=args.full3d, offaxis=args.offaxis
    )
    e = r["e"]
    ie = int(np.argmin(np.abs(e - 1.0)))
    print(
        f"Absolute numu flux at 1 GeV (vertical, az-avg): "
        f"{r['flux']['total_numu'][0].mean(0)[ie]:.3g} /(m^2 s sr GeV)"
    )

    if args.validate or args.plot:
        _validate(r, args)


def _validate(r, args):
    import os

    if not os.path.exists("honda_kam.npz"):
        print("honda_kam.npz not present; run validate_honda.py first.")
        return
    h = dict(np.load("honda_kam.npz"))
    He, Hcz, nm = h["E"], h["czlo"], h["numu"]  # Honda numu[cosZ,az,E]
    e = r["e"]

    def _at(yvals, xgrid, E):
        """Log-log interpolate a falling flux to the exact energy E.

        Essential: the MCEq grid point nearest 1 GeV is 0.89 GeV, so comparing by
        nearest-index would pit our 0.89 GeV flux against Honda's 1.0 GeV bin (a
        ~1.5x energy mismatch on a steeply falling spectrum).
        """
        y = np.maximum(np.asarray(yvals), 1e-300)
        return float(np.exp(np.interp(np.log(E), np.log(xgrid), np.log(y))))

    # Dense low-E grid: 0.1-100 GeV is the region that matters (sub-GeV oscillation
    # physics), with extra points below 1 GeV.
    EGRID = (0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0)
    # az-averaged numu vs Honda: vertical (down 0.95) and up-going (-0.95)
    print("\nABSOLUTE numu (az-avg) vs Honda HKKM2014 [/(m^2 s sr GeV)], 0.1-100 GeV:")
    print("  E[GeV]   vert(this)   vert(Honda)  ratio | up(this) up(Honda) ratio")
    mine = {s: r["flux"][s] for s in SPECIES}
    izd = int(np.argmin(np.abs(r["cos_zeniths"] - 0.95)))
    izu = int(np.argmin(np.abs(r["cos_zeniths"] + 0.95)))
    ihzd = int(np.argmin(np.abs(Hcz - 0.9)))  # Honda down bin lo edge
    ihzu = int(np.argmin(np.abs(Hcz - (-1.0))))  # Honda up bin lo edge
    for E in EGRID:
        md = _at(mine["total_numu"][izd].mean(0), e, E)
        hd = _at(nm[ihzd].mean(0), He, E)
        mu = _at(mine["total_numu"][izu].mean(0), e, E)
        hu = _at(nm[ihzu].mean(0), He, E)
        print(
            f"  {E:6.2f}  {md:10.3g}  {hd:10.3g}  {md/hd:5.2f} |"
            f" {mu:8.3g} {hu:8.3g} {mu/hu:5.2f}"
        )

    # flavour ratio (nue+nuebar)/(numu+numubar) -- robust across models
    if "nue" in h:
        nue_h = h["nue"] + h["nuebar"]
        numu_h = h["numu"] + h["numubar"]
        print("\nFLAVOUR RATIO (nue+nuebar)/(numu+numubar) vs Honda (vertical):")
        print("  E[GeV]   this work   Honda")
        iz = int(np.argmin(np.abs(r["cos_zeniths"] - 0.95)))
        ihz = int(np.argmin(np.abs(Hcz - 0.9)))
        for E in (0.1, 0.2, 0.3, 0.5, 1.0, 3.0, 10.0):
            rm = (
                _at(mine["total_nue"][iz].mean(0), e, E)
                + _at(mine["total_antinue"][iz].mean(0), e, E)
            ) / (
                _at(mine["total_numu"][iz].mean(0), e, E)
                + _at(mine["total_antinumu"][iz].mean(0), e, E)
            )
            rh = _at(nue_h[ihz].mean(0), He, E) / _at(numu_h[ihz].mean(0), He, E)
            print(f"  {E:5.1f}     {rm:6.3f}     {rh:6.3f}")
    if args.plot:
        _plot(r, h)


def _plot(r, h):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e = r["e"]
    He, Hcz, nm = h["E"], h["czlo"], h["numu"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.4))
    # spectrum, down-going vertical: this work vs Honda, focused on 0.1-100 GeV
    iz = int(np.argmin(np.abs(r["cos_zeniths"] - 0.95)))
    mv = r["flux"]["total_numu"][iz].mean(0)
    ihz = int(np.argmin(np.abs(Hcz - 0.9)))
    s = (e >= 0.1) & (e <= 100)
    axL.loglog(e[s], (mv * e**3)[s], "C3o-", ms=3, label="this work (3D engine)")
    sH = (He >= 0.1) & (He <= 100)
    axL.loglog(He[sH], (nm[ihz].mean(0) * He**3)[sH], "k--", label="Honda HKKM2014")
    axL.set_xlabel("E [GeV]")
    axL.set_ylabel(r"$E^3\,\Phi_{\nu_\mu}$  [GeV$^2$/(m$^2$ s sr)]")
    axL.set_title("Absolute numu spectrum, vertical (Kamioka)")
    axL.legend()
    # zenith dependence at 1 GeV
    ie = int(np.argmin(np.abs(e - 1.0)))
    ih = int(np.argmin(np.abs(He - 1.0)))
    axR.plot(
        r["cos_zeniths"],
        r["flux"]["total_numu"][:, :, ie].mean(1),
        "C3o-",
        label="this work",
    )
    czc = Hcz + 0.05
    axR.plot(czc[czc > 0], nm[:, :, ih].mean(1)[czc > 0], "k--", label="Honda")
    axR.set_xlabel(r"$\cos\theta_z$")
    axR.set_ylabel(r"$\Phi_{\nu_\mu}$ at 1 GeV  [/(m$^2$ s sr GeV)]")
    axR.set_title("Zenith dependence (azimuth-averaged)")
    axR.legend()
    fig.tight_layout()
    fig.savefig("mceq3d_flux.png", dpi=110)
    print("\nsaved plot -> mceq3d_flux.png")


if __name__ == "__main__":
    main()
