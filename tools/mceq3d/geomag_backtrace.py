"""Geomagnetic rigidity cutoff by trajectory back-tracing (replaces Stoermer).

The analytic Stoermer cutoff in ``daemonflux.geomagnetic`` is a dipole *formula*.
This module computes the cutoff from first principles: it integrates a charged
particle's trajectory in the geomagnetic field and asks whether a cosmic ray of a
given rigidity can reach the detector from a given direction (the standard
back-tracing technique used by MAGNETOCOSMICS / OTSO).

* Field: a tilted centred dipole built from the **IGRF** main coefficients
  (g10, g11, h11). The field is a single pluggable function ``bfield(r)`` -- to go
  to full IGRF, add the higher-degree Gauss terms there; nothing else changes.
* Tracer: to test whether a *positive* cosmic ray of rigidity R can arrive
  travelling along ``d_hat``, launch the particle backwards (reverse velocity and
  charge) from the detector and integrate (RK4 in arc length). If it escapes to
  large radius -> **allowed**; if it returns to Earth -> **forbidden**. The cutoff
  is the allowed/forbidden transition.

Validation: for a centred *aligned* dipole this reproduces the Stoermer cutoff --
~14.9 GV vertical at the geomagnetic equator, the ``cos^4(latitude)`` fall-off,
and the East-West asymmetry (lower cutoff from the West for positive particles).

Run::

    python geomag_backtrace.py
"""

from __future__ import annotations

import argparse
import os

import numpy as np

RE = 6.371e6  # Earth radius [m]
C = 2.99792458e8  # [m/s]
B0 = 3.07e-5  # dipole equatorial surface field [T] (from IGRF dipole moment)

# IGRF-13 (epoch 2020) main dipole Gauss coefficients [nT]
G10, G11, H11 = -29404.8, -1450.9, 4652.5

# ---------------------------------------------------------------------------
# Rigidity-scan resolution (see :func:`scan_upper_cutoff`)
# ---------------------------------------------------------------------------
# Ceiling of the cutoff scan.  MUST exceed the true maximum cutoff over the sky
# or the map saturates (flat, zero gradient); Kamioka peaks at ~49.5 GV near the
# horizon in the East, so 55 GV leaves ~5 GV of margin.  Every GV of extra
# headroom costs real time (rigidities above the cutoff are ALLOWED and their
# back-traces run to r_escape, ~900 RK4 steps, vs ~30 for a forbidden one).
RC_MAX_GV = 55.0
# Step of the coarse top-down ladder.  Must be smaller than the narrowest
# forbidden band above the true cutoff, or the scan steps over it and lands in a
# penumbral allowed island below (the 2.37 GV delivered step did exactly that at
# the Kamioka vertical: island [9.63, 10.13] -> 8.79 GV instead of ~11.5 GV).
COARSE_STEP_GV = 1.0
# Bisection target: the delivered R_c is then good to ~+-0.05 GV, i.e. the
# rigidity quantisation is no longer a term in the flux error budget.
BISECT_TOL_GV = 0.1
# Tag identifying this scan scheme AND the field it integrates in; it goes into
# every cutoff-map cache key so maps built with an older scheme -- or with the
# old, 180-deg-rotated :func:`dipole_axis` -- are never silently reused.
# History: "bisect1.0-0.1" = coarse ladder + bisection, tilted-dipole far field
# with the wrong axis longitude; "-dax" = the corrected geomagnetic-north axis.
CUTOFF_SCHEME = "bisect1.0-0.1-dax"


def dipole_axis():
    """Unit vector of the geomagnetic dipole axis from the IGRF coefficients.

    Tilt ~9.4 deg from the rotation (z) axis; the moment points roughly south
    (g10 < 0). Returned axis is the geomagnetic *north* (m_hat in B ~ -m_hat),
    i.e. it points at the geomagnetic north pole -- 80.6 N, **72.7 W** for
    IGRF-13/2020.

    The dipole part of the Gauss expansion has moment direction
    ``(g11, h11, g10)``; with ``g10 < 0`` that vector points at the geomagnetic
    *SOUTH* pole (80.6 S, 107.3 E), so the geomagnetic north axis is its
    negative.  Taking only ``lon = atan2(h11, g11)`` (as this function did)
    keeps the *south* pole's longitude while flipping the latitude, which puts
    the axis at 80.6 N, 107.3 E -- 18.8 deg away from the true axis and with the
    dipole tilt leaning to the wrong side of the globe.  The tilted dipole is
    used for r > ``r_switch`` in :func:`_bfield_igrf_cart` and everywhere in the
    scalar :func:`bfield`/:func:`is_allowed` path.
    """
    m = -np.array([G11, H11, G10], dtype=float)  # geomagnetic NORTH direction
    return m / np.linalg.norm(m)


def bfield(r, m_hat):
    """Centred-dipole field [T] at position r [m] (``m_hat`` = geomagnetic NORTH).

    Sign convention: at the equator the field points north and at the geomagnetic
    north pole it points down (into the Earth) -- the real geometry -- hence the
    leading minus. (Plug higher IGRF terms here for the full field.)
    """
    rn = np.linalg.norm(r)
    rhat = r / rn
    return -B0 * (RE / rn) ** 3 * (3 * np.dot(m_hat, rhat) * rhat - m_hat)


def _local_frame(lat_deg, lon_deg):
    la, lo = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    up = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    north = np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)])
    east = np.array([-np.sin(lo), np.cos(lo), 0.0])
    return up, north, east


def arrival_direction(lat_deg, lon_deg, zenith_deg, azimuth_deg):
    """Velocity unit vector of a cosmic ray arriving at the detector.

    ``azimuth_deg`` is the standard *arrival* (from) azimuth -- the compass
    direction of the source (N=0, E=90, S=180, W=270) -- so the velocity points
    downward and *opposite* to the source horizontal direction.
    """
    up, north, east = _local_frame(lat_deg, lon_deg)
    th, az = np.deg2rad(zenith_deg), np.deg2rad(azimuth_deg)
    return -np.cos(th) * up - np.sin(th) * (np.cos(az) * north + np.sin(az) * east)


def is_allowed(
    lat_deg,
    lon_deg,
    zenith_deg,
    azimuth_deg,
    rigidity_GV,
    m_hat,
    charge=+1,
    r_escape=25.0,
    ds=2.0e4,
    max_s=3.0e9,
):
    """True if a cosmic ray of ``rigidity_GV`` can reach the detector from the
    given direction (back-traced trajectory escapes to ``r_escape`` Re)."""
    R = rigidity_GV * 1e9  # volts
    d_hat = arrival_direction(lat_deg, lon_deg, zenith_deg, azimuth_deg)
    r = RE * _local_frame(lat_deg, lon_deg)[0]  # detector position (surface)
    u = -d_hat  # back-trace: reverse the velocity
    k = -charge * C / R  # reverse charge for back-tracing
    n = int(max_s / ds)
    for _ in range(n):
        # RK4 step in arc length
        def du(rr, uu):
            return k * np.cross(uu, bfield(rr, m_hat))

        k1r, k1u = u, du(r, u)
        k2r, k2u = u + 0.5 * ds * k1u, du(r + 0.5 * ds * k1r, u + 0.5 * ds * k1u)
        k3r, k3u = u + 0.5 * ds * k2u, du(r + 0.5 * ds * k2r, u + 0.5 * ds * k2u)
        k4r, k4u = u + ds * k3u, du(r + ds * k3r, u + ds * k3u)
        r = r + ds / 6 * (k1r + 2 * k2r + 2 * k3r + k4r)
        u = u + ds / 6 * (k1u + 2 * k2u + 2 * k3u + k4u)
        u /= np.linalg.norm(u)
        rn = np.linalg.norm(r)
        if rn > r_escape * RE:
            return True  # escaped -> allowed
        if rn < RE:
            return False  # returned to Earth -> forbidden
    return False  # trapped -> forbidden


def _bfield_igrf_cart(r_cart_m, date, m_hat, r_switch=4.0):
    """Full-IGRF field [T] (Cartesian, geocentric) at points r_cart_m (N,3) [m].

    Uses the real IGRF (degree 13, via ppigrf) where it matters -- below
    ``r_switch`` Earth radii -- and the fast tilted dipole farther out, where the
    higher-order Gauss terms are negligible (they fall as (a/r)^(n+1)).
    """
    import ppigrf

    r = np.linalg.norm(r_cart_m, axis=1)  # m
    rhat = r_cart_m / r[:, None]
    # dipole everywhere (default; overwritten by real IGRF below r_switch).
    # NOTE the leading minus: same sign convention as the validated scalar
    # :func:`bfield` above (field points north at the equator, down at the
    # geomagnetic north pole). It was missing here, so the far-field branch
    # returned -B beyond r_switch. Impact is small in practice -- beyond
    # r_switch=4 R_E the field is <2% of its surface value, and patching the
    # sign leaves the escape/return verdict of a near-horizon boundary
    # trajectory unchanged (Kamioka 87 deg E, 42 GV: escapes either way,
    # 1224 vs 1218 RK4 steps) -- but the field was simply wrong there.
    B = (
        -B0
        * (RE / r)[:, None] ** 3
        * (3 * (rhat @ m_hat)[:, None] * rhat - m_hat[None, :])
    )

    near = r < r_switch * RE
    if np.any(near):
        rk = r[near] / 1e3  # km
        x, y, z = r_cart_m[near].T
        colat = np.degrees(np.arccos(np.clip(z / r[near], -1, 1)))
        lon = np.degrees(np.arctan2(y, x))
        Br, Bt, Bp = (np.ravel(c) * 1e-9 for c in ppigrf.igrf_gc(rk, colat, lon, date))
        th, ph = np.deg2rad(colat), np.deg2rad(lon)
        st, ct, sp, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
        rh = np.stack([st * cp, st * sp, ct], 1)
        thh = np.stack([ct * cp, ct * sp, -st], 1)
        phh = np.stack([-sp, cp, np.zeros_like(sp)], 1)
        B[near] = Br[:, None] * rh + Bt[:, None] * thh + Bp[:, None] * phh
    return B


def scan_upper_cutoff(
    allowed_fn,
    n_dirs,
    r_lo=0.5,
    r_hi=RC_MAX_GV,
    coarse_step=COARSE_STEP_GV,
    tol=BISECT_TOL_GV,
    return_admittance=False,
):
    """Upper cutoff ``R_U`` for ``n_dirs`` directions: coarse top-down scan then
    bisection.  Tracer-agnostic (unit-testable on a synthetic admittance).

    ``R_U`` semantics -- **unchanged**: the highest rigidity that is *forbidden*,
    i.e. the first forbidden sample met when scanning from ``r_hi`` downwards.
    (This is deliberately NOT a penumbra-averaged effective cutoff ``R_eff``;
    commit ``5a5ef11`` refuted that alternative on a fine admittance scan.)

    Why the two stages
    ------------------
    A single linear ladder of ``n_scan`` points is not a robust way to find
    ``R_U``: the topmost forbidden band can be *narrower than the step*, and the
    penumbra contains narrow **allowed islands**.  With the delivered
    ``n_scan=24`` over 0.5-55 GV (2.37 GV step) the Kamioka vertical ladder
    samples 9.98 GV, which sits inside the allowed island [9.63, 10.13] found by
    ``diag_admittance_fine.py``; the scan therefore steps straight over the
    forbidden band that ends at ~11.5 GV and reports 8.79 GV instead --
    a 2.7 GV (24%) error, on 37% of the down-going cells.

    Stage 1 walks a ladder whose step is ``<= coarse_step`` (1 GV by default,
    below the width of the topmost forbidden band everywhere on the sky at
    Kamioka) and takes the first forbidden sample.  Stage 2 bisects the bracket
    [last allowed, first forbidden] down to ``tol`` (~0.1 GV), which costs only
    ``ceil(log2(coarse_step/tol))`` ~ 4 extra traces per direction instead of the
    ~550 a 0.1 GV ladder would need.

    ``allowed_fn(idx, R) -> bool array`` must back-trace direction ``idx[j]`` at
    rigidity ``R[j]``.  Cells forbidden already at ``r_hi`` saturate (returned as
    ``r_hi``) and are flagged in the returned ``saturated`` mask.

    Returns ``(rc, saturated)``, or ``(rc, saturated, rs, A)`` with
    ``return_admittance`` (``rs`` the coarse ladder, high->low; ``A`` its
    0/1 admittance, ``(n_dirs, len(rs))``).
    """
    n_coarse = int(np.ceil((r_hi - r_lo) / float(coarse_step))) + 1
    rs = np.linspace(r_hi, r_lo, n_coarse)
    idx = np.repeat(np.arange(n_dirs), n_coarse)
    A = allowed_fn(idx, np.tile(rs, n_dirs)).reshape(n_dirs, n_coarse)

    forb_any = ~A.all(axis=1)
    first_forb = np.argmin(A, axis=1)  # index of first False (valid if forb_any)
    saturated = forb_any & (first_forb == 0)
    rc = np.full(n_dirs, r_lo)
    rc[saturated] = r_hi

    # bracket [lo (forbidden), hi (allowed)] for the bisection stage
    act = np.where(forb_any & ~saturated)[0]
    if len(act):
        lo = rs[first_forb[act]]
        hi = rs[first_forb[act] - 1]
        n_bis = max(0, int(np.ceil(np.log2(max(coarse_step, 1e-9) / max(tol, 1e-9)))))
        for _ in range(n_bis):
            mid = 0.5 * (lo + hi)
            a = allowed_fn(act, mid)
            lo = np.where(a, lo, mid)  # mid forbidden -> R_U is at/above mid
            hi = np.where(a, mid, hi)  # mid allowed   -> R_U is below mid
        rc[act] = 0.5 * (lo + hi)
    if return_admittance:
        return rc, saturated, rs, A.astype(float)
    return rc, saturated


def _states_worker(args):
    """Fork worker: run :func:`scan_upper_cutoff` on one chunk of directions."""
    r0c, u0c, date, m_hat, charge, r_lo, r_hi, coarse_step, tol, ret_a, kw = args
    n = len(r0c)

    def allowed_fn(idx, R):
        return backtrace_vec(r0c[idx], u0c[idx], R, date, m_hat, charge, **kw)

    return scan_upper_cutoff(
        allowed_fn, n, r_lo, r_hi, coarse_step, tol, return_admittance=ret_a
    )


def cutoff_from_states(
    r0_m,
    u0,
    date,
    m_hat=None,
    charge=+1,
    r_lo=0.5,
    r_hi=RC_MAX_GV,
    coarse_step=COARSE_STEP_GV,
    tol=BISECT_TOL_GV,
    n_jobs=None,
    return_admittance=False,
    **kw,
):
    """Upper cutoff ``R_U`` [GV] for N back-trace launch states ``(r0_m, u0)``.

    One entry point for every cutoff in this package: the detector map
    (:func:`cutoff_map`, launch at the surface) and the far-side / production-point
    maps (``mceq3d_flux.farside_cutoff_map``, launch at the production point) differ
    only in the launch state, so they share this scan, its resolution guarantee and
    its parallelism.

    Directions are split over ``n_jobs`` forked workers (default
    ``os.cpu_count()//2``).  Each worker keeps its own large rigidity batch, which
    matters: the IGRF field call has a ~16 ms fixed overhead and only ~45 us per
    point, so batches of a few hundred trajectories are ~100x more efficient per
    trajectory than single traces.

    Returns ``(rc, saturated)``, plus ``(rs, A)`` if ``return_admittance``.
    """
    if m_hat is None:
        m_hat = dipole_axis()
    r0_m = np.ascontiguousarray(r0_m, float)
    u0 = np.ascontiguousarray(u0, float)
    n = len(r0_m)
    if n == 0:
        empty = (np.zeros(0), np.zeros(0, bool))
        return empty + (np.zeros(0), np.zeros((0, 0))) if return_admittance else empty
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 2) // 2)
    n_jobs = max(1, min(int(n_jobs), n))

    chunks = np.array_split(np.arange(n), n_jobs)
    args = [
        (
            r0_m[c], u0[c], date, m_hat, charge, r_lo, r_hi, coarse_step, tol,
            return_admittance, kw,
        )
        for c in chunks
        if len(c)
    ]
    if len(args) == 1:
        results = [_states_worker(args[0])]
    else:
        import multiprocessing as mp

        with mp.get_context("fork").Pool(len(args)) as pool:
            results = pool.map(_states_worker, args)

    rc = np.concatenate([r[0] for r in results])
    sat = np.concatenate([r[1] for r in results])
    if return_admittance:
        return rc, sat, results[0][2], np.concatenate([r[3] for r in results])
    return rc, sat


def cutoff_map(
    lat_deg,
    lon_deg,
    date,
    zeniths,
    azimuths,
    m_hat=None,
    charge=+1,
    r_lo=0.5,
    r_hi=RC_MAX_GV,
    n_scan=None,
    coarse_step=COARSE_STEP_GV,
    tol=BISECT_TOL_GV,
    n_jobs=None,
    return_admittance=False,
    warn_saturated=True,
    **kw,
):
    """Full-IGRF cutoff [GV] on a (zenith x azimuth) sky grid for a site.

    ``r_hi`` defaults to :data:`RC_MAX_GV` (55 GV): the near-horizon *East* cutoff
    at a mid-latitude site reaches ~49.5 GV at Kamioka, and a lower ceiling
    *saturates* those cells -- a flat map with zero gradient, which silently kills
    every direction-shift effect.  Cells still forbidden at ``r_hi`` are returned
    as ``r_hi`` and warned about (``warn_saturated``).

    Resolution: see :func:`scan_upper_cutoff` -- a ``coarse_step`` (1 GV) top-down
    ladder followed by bisection to ``tol`` (0.1 GV), NOT a single linear ladder.
    ``n_scan`` is accepted for backward compatibility and may only *refine* the
    coarse step (``(r_hi-r_lo)/(n_scan-1)`` if that is smaller than
    ``coarse_step``); it can no longer coarsen it past the 1 GV guarantee, which
    is what made the delivered vertical cutoff land inside an allowed island.

    ``return_admittance``: also return ``(rs, A)`` -- ``rs`` the *coarse* rigidity
    ladder [GV] (high->low) and ``A[n_zen, n_az, len(rs)]`` its 0/1 admittance.
    Collapsing A to the highest forbidden rigidity gives (to ``coarse_step``) the
    cutoff ``out``; keeping A whole preserves the penumbra (e.g. allowed islands
    below the main forbidden band) that the analytic erf step cannot represent.

    CAVEAT -- this map is **anchored at the detector**: every trajectory is
    launched from ``RE * up`` at ``(lat, lon)``.  Near the horizon the primary
    that makes a neutrino arriving here reaches the atmosphere hundreds of km
    away (370 km at zenith 87 deg for h_prod = 30 km), where the *site* -- not
    just the local vertical -- has moved by L/R_E in geomagnetic latitude.  That
    displacement is not a small correction to the AZIMUTHAL SHAPE of the map:
    at Kamioka, zenith 87 deg (``diag_dipole_phase.py --sections prodpoint``)

        launch        N      E      S      W    N/S   peak   1st-harm
        detector   23.78  42.99  10.26   7.60   2.32  68.9     61.7
        prod_point 19.91  37.13  12.10   7.95   1.65  88.1     72.4

    i.e. the detector-anchored map's R_c(azimuth) is skewed ~11 deg further
    clockwise (and its North/South contrast is 40% larger) than the map the
    primaries actually see.  ``mceq3d_flux.MCEq3DFlux.cone_geff``'s
    ``sublimb="prod_point"`` rotates the production point's local *vertical* but
    still reads this detector-anchored map, so the latitude part of the
    displacement is currently unmodelled.  See
    :func:`cutoff_from_states`, which takes an arbitrary launch state and is the
    entry point a production-point map would use.
    """
    if n_scan is not None:
        coarse_step = min(coarse_step, (r_hi - r_lo) / max(int(n_scan) - 1, 1))
    up = _local_frame(lat_deg, lon_deg)[0]
    nz, na = len(zeniths), len(azimuths)
    r0 = np.tile(RE * up, (nz * na, 1))
    u0 = np.array(
        [
            -arrival_direction(lat_deg, lon_deg, z, a)
            for z in zeniths
            for a in azimuths
        ]
    )
    res = cutoff_from_states(
        r0, u0, date, m_hat, charge, r_lo, r_hi, coarse_step, tol, n_jobs,
        return_admittance, **kw,
    )
    rc, sat = res[0], res[1]
    if warn_saturated and sat.any():
        import warnings

        warnings.warn(
            f"cutoff_map: {int(sat.sum())} of {sat.size} cells are forbidden even "
            f"at the r_hi={r_hi:g} GV ceiling and were clamped there (flat map, "
            "zero gradient). Raise r_hi.",
            stacklevel=2,
        )
    out = rc.reshape(nz, na)
    if return_admittance:
        return out, res[2], res[3].reshape(nz, na, -1)
    return out


def backtrace_vec(
    r0_m, u0, rigidity_GV, date, m_hat, charge=+1, r_escape=15.0, ds=1.0e5,
    max_s=1.6e9, r_floor=None, r_switch=4.0,
):
    """Vectorized back-tracing of N trajectories. Returns allowed mask (N,).

    Trajectories that have already escaped or returned are **dropped from the
    batch** rather than merely masked: a forbidden trace terminates in ~30 steps
    and an allowed one in ~900, while the loop bound is ``max_s/ds`` = 16000, so
    without compaction the whole batch paid the cost of its slowest (trapped)
    member at full width.  The verdicts are identical either way.

    ``r_floor`` [m] is the absorbing radius: a back-trace that falls below it has
    "returned to Earth" and the direction is forbidden.  It defaults to
    :data:`RE`; the launch altitude (in ``r0_m``) and the floor are the same
    surface in a standard cutoff calculation, so pass both together.

    ``r_switch`` [R_E] is forwarded to :func:`_bfield_igrf_cart`: the real IGRF
    is used below it and the tilted dipole above.  ``r_switch=0`` gives a
    **pure centred dipole everywhere**, which is the regime where the analytic
    Stoermer cutoff is exact -- that is how the tracer's conventions are pinned
    in ``test_geomag_backtrace.py`` without paying for the scalar tracer.
    """
    r = np.array(r0_m, float)
    u = np.array(u0, float)
    R = np.asarray(rigidity_GV, float) * 1e9
    r_floor = RE if r_floor is None else float(r_floor)
    k = (-charge * C / R)[:, None]
    n_pts = r.shape[0]
    live = np.arange(n_pts)
    allowed = np.zeros(n_pts, bool)

    def du(rr, uu):
        return k * np.cross(uu, _bfield_igrf_cart(rr, date, m_hat, r_switch))

    for _ in range(int(max_s / ds)):
        k1u = du(r, u)
        k2u = du(r + 0.5 * ds * u, u + 0.5 * ds * k1u)
        k3u = du(r + 0.5 * ds * (u + 0.5 * ds * k1u), u + 0.5 * ds * k2u)
        k4u = du(r + ds * (u + 0.5 * ds * k2u), u + ds * k3u)
        r = r + ds * (u + ds / 6 * (k1u + k2u + k3u))
        u = u + ds / 6 * (k1u + 2 * k2u + 2 * k3u + k4u)
        u /= np.linalg.norm(u, axis=1)[:, None]
        rn = np.linalg.norm(r, axis=1)
        esc = rn > r_escape * RE
        allowed[live[esc]] = True
        keep = ~(esc | (rn < r_floor))
        if not keep.all():
            if not keep.any():
                break
            live, r, u, k = live[keep], r[keep], u[keep], k[keep]
    return allowed


def cutoff_igrf(
    lat_deg,
    lon_deg,
    zenith_deg,
    azimuth_deg,
    date,
    m_hat=None,
    charge=+1,
    r_lo=0.3,
    r_hi=RC_MAX_GV,
    n_scan=None,
    coarse_step=COARSE_STEP_GV,
    tol=BISECT_TOL_GV,
    **kw,
):
    """Full-IGRF cutoff [GV] for one direction -- the reference single-direction
    value (same coarse-ladder + bisection scheme as :func:`cutoff_map`, so a map
    cell and a direct back-trace of the same direction agree to ``tol``)."""
    up = _local_frame(lat_deg, lon_deg)[0]
    if n_scan is not None:
        coarse_step = min(coarse_step, (r_hi - r_lo) / max(int(n_scan) - 1, 1))
    d = -arrival_direction(lat_deg, lon_deg, zenith_deg, azimuth_deg)
    rc, _ = cutoff_from_states(
        (RE * up)[None, :], d[None, :], date, m_hat, charge, r_lo, r_hi,
        coarse_step, tol, n_jobs=1, **kw,
    )
    return float(rc[0])


def cutoff_source(lat_deg, lon_deg, date, **kw):
    """Return a ``f(zenith_deg, azimuth_deg) -> R_c [GV]`` back-traced cutoff.

    Convenience factory so the first-principles IGRF cutoff can be dropped
    straight into the package model::

        from geomag_backtrace import cutoff_source
        import datetime
        gm = GeomagneticModel("kamioka",
                              cutoff_source=cutoff_source(36.43, 137.31,
                                                          datetime.datetime(2020, 1, 1)))

    The returned callable broadcasts over array (zenith, azimuth) inputs. (It is
    slow -- one trajectory back-trace per direction -- so cache/precompute a map
    for production grids; this is the drop-in *selector*, not a fast path.)
    """

    def f(zenith_deg, azimuth_deg):
        z = np.atleast_1d(np.asarray(zenith_deg, dtype=float))
        a = np.atleast_1d(np.asarray(azimuth_deg, dtype=float))
        zb, ab = np.broadcast_arrays(z, a)
        out = np.array(
            [
                cutoff_igrf(lat_deg, lon_deg, float(zz), float(aa), date, **kw)
                for zz, aa in zip(zb.ravel(), ab.ravel())
            ]
        )
        return out.reshape(zb.shape)

    return f


def cached_cutoff_source(
    lat_deg,
    lon_deg,
    date,
    n_zen=19,
    n_az=25,
    charge=+1,
    cache_dir=None,
    rebuild=False,
    **kw,
):
    """Fast, **precomputed-and-cached** back-traced cutoff for a site.

    This is the production-ready counterpart to :func:`cutoff_source`: it batches a
    full (zenith x azimuth) sky map once with :func:`cutoff_map` (a few minutes),
    caches it to ``cache_dir`` as ``.npz``, and returns a callable
    ``f(zenith_deg, azimuth_deg) -> R_c [GV]`` that **bilinearly interpolates** the
    map (microseconds, azimuth-periodic). Subsequent calls/processes reuse the
    cache, so the first-principles IGRF cutoff is fast enough to be the standard
    path::

        gm = GeomagneticModel("kamioka",
                              cutoff_source=cached_cutoff_source(36.43, 137.31, date))

    The default grid is 5 deg in zenith (0-90) x 15 deg in azimuth.
    """
    from scipy.interpolate import RegularGridInterpolator

    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "cutoff_cache")
    os.makedirs(cache_dir, exist_ok=True)
    tag = getattr(date, "strftime", lambda f: str(date))("%Y%m%d")
    # the scan-scheme tag is part of the file name: a map built with the old
    # single-ladder scan is not merely coarser, it can be several GV wrong.
    fname = os.path.join(
        cache_dir,
        f"cutoff_{lat_deg:.2f}_{lon_deg:.2f}_{tag}_{n_zen}x{n_az}_q{charge:+d}"
        f"_{CUTOFF_SCHEME}.npz",
    )
    if os.path.exists(fname) and not rebuild:
        d = np.load(fname)
        zeniths, azimuths, grid = d["zeniths"], d["azimuths"], d["grid"]
    else:
        zeniths = np.linspace(0.0, 90.0, n_zen)
        azimuths = np.linspace(0.0, 360.0, n_az)
        grid = cutoff_map(
            lat_deg, lon_deg, date, zeniths, azimuths, charge=charge, **kw
        )
        np.savez(fname, zeniths=zeniths, azimuths=azimuths, grid=grid)

    interp = RegularGridInterpolator(
        (zeniths, azimuths), grid, bounds_error=False, fill_value=None
    )

    def f(zenith_deg, azimuth_deg):
        z = np.atleast_1d(np.asarray(zenith_deg, dtype=float))
        a = np.atleast_1d(np.asarray(azimuth_deg, dtype=float)) % 360.0
        zb, ab = np.broadcast_arrays(z, a)
        out = interp(np.stack([zb.ravel(), ab.ravel()], axis=-1)).reshape(zb.shape)
        return out if out.shape != (1,) else float(out[0])

    f.grid = grid  # expose for inspection/plots
    f.zeniths = zeniths
    f.azimuths = azimuths
    f.cache_file = fname
    return f


def cutoff_rigidity(
    lat_deg,
    lon_deg=0.0,
    zenith_deg=0.0,
    azimuth_deg=0.0,
    m_hat=None,
    charge=+1,
    r_lo=0.3,
    r_hi=60.0,
    n_scan=40,
):
    """Effective (upper) cutoff: highest rigidity that is forbidden when scanning
    from high to low -- i.e. the allowed/forbidden transition."""
    if m_hat is None:
        m_hat = dipole_axis()
    rs = np.linspace(r_hi, r_lo, n_scan)
    last_allowed = r_hi
    for R in rs:
        if is_allowed(lat_deg, lon_deg, zenith_deg, azimuth_deg, R, m_hat, charge):
            last_allowed = R
        else:
            return 0.5 * (last_allowed + R)  # transition (first forbidden below)
    return r_lo


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.parse_args(argv)
    # validate with an ALIGNED centred dipole (m_hat = z) -> pure Stoermer
    mz = np.array([0.0, 0.0, 1.0])

    print("Back-traced cutoff vs Stoermer (aligned dipole):")
    print("  geomag lat   vertical cutoff [GV]   Stoermer 14.9*cos^4(lat)")
    for lat in (0.0, 20.0, 40.0, 60.0):
        rc = cutoff_rigidity(lat, 0.0, 0.0, 0.0, m_hat=mz)
        st = 14.9 * np.cos(np.deg2rad(lat)) ** 4
        print(f"   {lat:5.0f}        {rc:6.2f}                {st:6.2f}")

    print("\nEast-West at the geomagnetic equator (zenith 45 deg), aligned dipole:")
    rc_e = cutoff_rigidity(0.0, 0.0, 45.0, 90.0, m_hat=mz)  # from East
    rc_w = cutoff_rigidity(0.0, 0.0, 45.0, 270.0, m_hat=mz)  # from West
    print(
        f"  East cutoff = {rc_e:.2f} GV,  West cutoff = {rc_w:.2f} GV"
        f"  -> West lower? {rc_w < rc_e}"
    )

    md = dipole_axis()
    print(
        f"\nIGRF dipole tilt = {np.rad2deg(np.arccos(md[2])):.1f} deg "
        "(full IGRF: add higher Gauss terms in bfield)"
    )


if __name__ == "__main__":
    main()
