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


def dipole_axis():
    """Unit vector of the geomagnetic dipole axis from the IGRF coefficients.

    Tilt ~9.4 deg from the rotation (z) axis; the moment points roughly south
    (g10 < 0). Returned axis is the geomagnetic *north* (m_hat in B ~ -m_hat).
    """
    # geomagnetic north pole colatitude/longitude from the dipole terms
    tilt = np.arctan2(np.hypot(G11, H11), -G10)  # from -z because g10<0
    lon = np.arctan2(H11, G11)
    return np.array(
        [np.sin(tilt) * np.cos(lon), np.sin(tilt) * np.sin(lon), np.cos(tilt)]
    )


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
    B = (
        B0
        * (RE / r)[:, None] ** 3
        * (3 * (rhat @ m_hat)[:, None] * rhat - m_hat[None, :])
    )  # dipole everywhere (default)

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


def cutoff_map(
    lat_deg,
    lon_deg,
    date,
    zeniths,
    azimuths,
    m_hat=None,
    charge=+1,
    r_lo=0.5,
    r_hi=20.0,
    n_scan=16,
    return_admittance=False,
    **kw,
):
    """Full-IGRF cutoff [GV] on a (zenith x azimuth) sky grid for a site.

    All directions and rigidities are batched into a single vectorized
    back-trace (ppigrf's per-call overhead is amortized), so a coarse sky map is
    a few minutes rather than hours.

    ``return_admittance``: also return ``(rs, A)`` -- ``rs`` the rigidity scan
    grid [GV] (high->low) and ``A[n_zen, n_az, n_scan]`` the **penumbral
    admittance** (allowed fraction, 0/1 per traced rigidity). Collapsing A to the
    highest forbidden rigidity gives the single cutoff ``out``; keeping A whole
    preserves the penumbra (e.g. allowed islands below the main forbidden band)
    that the analytic erf step cannot represent.
    """
    if m_hat is None:
        m_hat = dipole_axis()
    up = _local_frame(lat_deg, lon_deg)[0]
    rs = np.linspace(r_hi, r_lo, n_scan)
    dirs = [(z, a) for z in zeniths for a in azimuths]
    r0, u0, R = [], [], []
    for z, a in dirs:
        d_hat = arrival_direction(lat_deg, lon_deg, z, a)
        for R_i in rs:
            r0.append(RE * up)
            u0.append(-d_hat)
            R.append(R_i)
    allowed = backtrace_vec(
        np.array(r0), np.array(u0), np.array(R), date, m_hat, charge, **kw
    ).reshape(len(dirs), n_scan)
    out = np.zeros((len(zeniths), len(azimuths)))
    for d_idx, (iz, ia) in enumerate(
        [(iz, ia) for iz in range(len(zeniths)) for ia in range(len(azimuths))]
    ):
        forb = np.where(~allowed[d_idx])[0]
        out[iz, ia] = (
            r_lo
            if len(forb) == 0
            else (r_hi if forb[0] == 0 else 0.5 * (rs[forb[0] - 1] + rs[forb[0]]))
        )
    if return_admittance:
        return out, rs, allowed.astype(float).reshape(
            len(zeniths), len(azimuths), n_scan
        )
    return out


def backtrace_vec(
    r0_m, u0, rigidity_GV, date, m_hat, charge=+1, r_escape=15.0, ds=1.0e5, max_s=1.6e9
):
    """Vectorized back-tracing of N trajectories. Returns allowed mask (N,)."""
    r = np.array(r0_m, float)
    u = np.array(u0, float)
    R = np.asarray(rigidity_GV, float) * 1e9
    k = (-charge * C / R)[:, None]
    n_pts = r.shape[0]
    done = np.zeros(n_pts, bool)
    allowed = np.zeros(n_pts, bool)

    def du(rr, uu):
        return k * np.cross(uu, _bfield_igrf_cart(rr, date, m_hat))

    for _ in range(int(max_s / ds)):
        k1u = du(r, u)
        k2u = du(r + 0.5 * ds * u, u + 0.5 * ds * k1u)
        k3u = du(r + 0.5 * ds * (u + 0.5 * ds * k1u), u + 0.5 * ds * k2u)
        k4u = du(r + ds * (u + 0.5 * ds * k2u), u + ds * k3u)
        r = r + ds * (u + ds / 6 * (k1u + k2u + k3u))
        u = u + ds / 6 * (k1u + 2 * k2u + 2 * k3u + k4u)
        u /= np.linalg.norm(u, axis=1)[:, None]
        rn = np.linalg.norm(r, axis=1)
        esc = (~done) & (rn > r_escape * RE)
        ret = (~done) & (rn < RE)
        allowed[esc] = True
        done |= esc | ret
        if done.all():
            break
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
    r_hi=30.0,
    n_scan=30,
):
    """Full-IGRF effective cutoff [GV] for one direction (vectorized over R)."""
    if m_hat is None:
        m_hat = dipole_axis()
    up = _local_frame(lat_deg, lon_deg)[0]
    r0 = np.tile(RE * up, (n_scan, 1))
    u0 = np.tile(
        -arrival_direction(lat_deg, lon_deg, zenith_deg, azimuth_deg), (n_scan, 1)
    )
    rs = np.linspace(r_hi, r_lo, n_scan)
    allowed = backtrace_vec(r0, u0, rs, date, m_hat, charge)
    forb = np.where(~allowed)[0]
    if len(forb) == 0:
        return r_lo
    i = forb[0]
    return r_hi if i == 0 else 0.5 * (rs[i - 1] + rs[i])


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
    fname = os.path.join(
        cache_dir,
        f"cutoff_{lat_deg:.2f}_{lon_deg:.2f}_{tag}_{n_zen}x{n_az}_q{charge:+d}.npz",
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
