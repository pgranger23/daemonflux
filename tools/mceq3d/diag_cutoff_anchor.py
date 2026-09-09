"""Diagnostic / validation of the production-point-anchored cutoff.

``cutoff_anchor="prod_point"`` (``mceq3d_flux.CUTOFF_ANCHOR``) reads each cone
sample's rigidity cutoff off a map LAUNCHED AT the production point P -- 370 km
up the arrival ray at zenith 87 deg -- instead of at the detector.  P's map is
interpolated from the displaced-site family ``MCEq3DFlux.prod_family_rc``.

Sections
--------
``validate``
    The 48 arrival directions of ``diag_dipole_phase.py --sections prodpoint``
    (zenith 80 and 87, 24 azimuths), three ways:
      * ``family``  -- what the engine now uses: the displaced-site family
        interpolated to P's ground offset and read in P's true local frame;
      * ``direct``  -- a DIRECT back-trace launched at P's ground point
        (surface, the family's own launch convention);
      * ``direct30``-- a DIRECT back-trace launched at P itself (altitude
        ``h_prod``, absorbing floor at the same altitude) -- the convention of
        ``diag_dipole_phase.py``, whose published numbers this reproduces.
    ``family`` vs ``direct`` isolates the ONE approximation of the scheme
    (bilinear interpolation of the map in the site offset); ``direct`` vs
    ``direct30`` is the (uniform, ~1%) launch-altitude convention.

``continuity``
    R_c along the arrival ray anchor as a function of zenith at fixed azimuth:
    the displacement grows continuously from 0 (vertical) to 617 km (horizon),
    so the anchored cutoff must have no step at any zenith node of the map.

``vertical``
    The anchor must be an exact no-op at the vertical.

Run (from tools/mceq3d, needs a cached ``.cache3d/prodfam_*.npz``)::

    python diag_cutoff_anchor.py --sections vertical continuity validate
"""

from __future__ import annotations

import argparse
import datetime

import numpy as np

import geomag_backtrace as gb
import joint_cone as jc
import mceq3d_flux as mf

LAT, LON = 36.43, 137.31
DATE = datetime.datetime(2020, 1, 1)
CACHE = ".cache3d"


class _Bare(mf.MCEq3DFlux):
    """The cutoff machinery needs no MCEq cascade."""

    def __init__(self):
        pass


def _family(n_jobs=1):
    return _Bare().prod_family_rc(LAT, LON, DATE, cache_dir=CACHE, n_jobs=n_jobs,
                                  warn_saturated=False)


def anchored_rc(fam, zen_deg, az_deg, h_prod_km=mf.H_PROD_KM, lat=LAT):
    """R_c [GV] of the arrival direction itself, production-point-anchored.

    Exactly what the cone integral does for its axis sample: interpolate the
    family to P's ground offset, then read it at the arrival direction's local
    angles AT P.
    """
    cz = float(np.cos(np.radians(zen_deg)))
    dn, de = mf.prod_point_offset_km(zen_deg, az_deg, h_prod_km)
    rc_site = mf.interp_site_rc(fam, dn, de)
    frame = mf.prod_frame_exact(cz, az_deg, h_prod_km, lat)
    th = np.radians(zen_deg)
    ph = np.radians(az_deg)
    n = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    th_p, ph_p = jc.local_angles(n, frame)
    return float(jc.rc_bilinear(np.asarray(fam["zen"], float),
                                np.asarray(fam["az"], float), rc_site,
                                th_p, ph_p))


def _phase(rc_row, az_row):
    """(peak azimuth, first-harmonic phase) [deg] -- ``diag_dipole_phase._phase``."""
    a = np.deg2rad(az_row)
    c, s = np.sum(rc_row * np.cos(a)), np.sum(rc_row * np.sin(a))
    ph = np.degrees(np.arctan2(s, c)) % 360.0
    n = len(rc_row)
    j = int(np.argmax(rc_row))
    y0, y1, y2 = rc_row[(j - 1) % n], rc_row[j], rc_row[(j + 1) % n]
    d = y0 - 2 * y1 + y2
    dx = 0.5 * (y0 - y2) / d if d != 0 else 0.0
    dz = az_row[1] - az_row[0]
    return (az_row[j] + dz * dx) % 360.0, ph


def _direct(zeniths, az, h_launch_km, n_jobs):
    """Back-traced R_c launched at the production point's site, one row per zenith."""
    r0, u0 = [], []
    for z in zeniths:
        for a in az:
            dn, de = mf.prod_point_offset_km(z, a)
            la, lo = mf.prod_site_latlon(LAT, LON, dn, de)
            up = gb._local_frame(la, lo)[0]
            r0.append((gb.RE + h_launch_km * 1e3) * up)
            u0.append(-gb.arrival_direction(LAT, LON, z, a))
    rc, _ = gb.cutoff_from_states(np.array(r0), np.array(u0), DATE, n_jobs=n_jobs,
                                  r_floor=gb.RE + h_launch_km * 1e3)
    return rc.reshape(len(zeniths), len(az))


def sec_vertical(fam):
    print("=" * 78)
    print("1. The anchor is an exact no-op at the vertical")
    print("=" * 78)
    zen_f, az_f, rc_f = fam["zen"], fam["az"], fam["rc"][1, 1]
    bad = 0.0
    for azd in (0.0, 45.0, 137.0, 300.0):
        dn, de = mf.prod_point_offset_km(0.0, azd)
        bad = max(bad, abs(dn), abs(de),
                  float(np.max(np.abs(mf.interp_site_rc(fam, dn, de) - rc_f))))
        got = anchored_rc(fam, 0.0, azd)
        ref = float(jc.rc_bilinear(zen_f, az_f, rc_f, 0.0, azd))
        print(f"  az {azd:6.1f}: displacement {np.hypot(dn, de):.3e} km, "
              f"R_c anchored {got:7.4f} vs detector {ref:7.4f}")
    print(f"  worst |anchored - detector| over the whole map: {bad:.3e} GV")


def sec_continuity(fam, n_zen=180):
    print("=" * 78)
    print("2. Continuity of the anchored R_c across the map's zenith nodes")
    print("=" * 78)
    zs = np.linspace(0.0, 89.5, n_zen)
    print(f"{'az':>5} {'max |dR_c| per 0.5 deg':>24} {'at zen':>8} "
          f"{'same, detector-anchored':>26}")
    for azd in (0.0, 90.0, 180.0, 270.0):
        r_a = np.array([anchored_rc(fam, z, azd) for z in zs])
        r_d = np.array([float(jc.rc_bilinear(fam["zen"], fam["az"], fam["rc"][1, 1],
                                             z, azd)) for z in zs])
        da, dd = np.abs(np.diff(r_a)), np.abs(np.diff(r_d))
        k = int(np.argmax(da))
        print(f"{azd:5.0f} {da.max():24.3f} {zs[k]:8.1f} {dd.max():26.3f}")
    print("  (the anchored curve must be no rougher than the detector-anchored")
    print("   one; both inherit the map's own 15 deg azimuth / limb zenith nodes)")


def sec_validate(fam, n_jobs=24):
    zeniths = (80.0, 87.0)
    az = np.arange(0.0, 360.0, 15.0)
    fam_rc = np.array([[anchored_rc(fam, z, a) for a in az] for z in zeniths])
    det_rc = np.array([[float(jc.rc_bilinear(fam["zen"], fam["az"], fam["rc"][1, 1],
                                             z, a)) for a in az] for z in zeniths])
    d0 = _direct(zeniths, az, 0.0, n_jobs)
    d30 = _direct(zeniths, az, mf.H_PROD_KM, n_jobs)
    print("=" * 78)
    print("3. Production-point-anchored R_c: engine vs direct back-trace")
    print("   (48 directions of diag_dipole_phase.py --sections prodpoint)")
    print("=" * 78)
    print(f"{'zen':>5} {'source':>10} {'N':>7} {'E':>7} {'S':>7} {'W':>7} "
          f"{'N/S':>6} {'peak':>7} {'1-harm':>7}")
    rows = dict(detector=det_rc, family=fam_rc, direct=d0, direct30=d30)
    for i, z in enumerate(zeniths):
        for tag in ("detector", "family", "direct", "direct30"):
            row = rows[tag][i]
            pk, ph = _phase(row, az)
            g = np.interp([351.9, 81.9, 171.9, 261.9], list(az) + [360.0],
                          list(row) + [row[0]])
            print(f"{z:5.0f} {tag:>10} {g[0]:7.2f} {g[1]:7.2f} {g[2]:7.2f} "
                  f"{g[3]:7.2f} {g[0]/g[2]:6.2f} {pk:7.2f} {ph:7.2f}")
        for tag in ("family", "direct", "direct30"):
            print(f"      {tag:>10}: "
                  + " ".join(f"{v:5.1f}" for v in rows[tag][i]))
        e_int = np.abs(fam_rc[i] - d0[i])
        e_tot = np.abs(fam_rc[i] - d30[i])
        alt = np.abs(d0[i] - d30[i])
        print(f"      residuals at zen {z:.0f}: family-direct  max {e_int.max():.2f} "
              f"GV / {np.max(e_int / d0[i]) * 100:.1f}%  (site interpolation)")
        print(f"                          direct-direct30 max {alt.max():.2f} GV / "
              f"{np.max(alt / d0[i]) * 100:.1f}%  (launch-altitude convention)")
        print(f"                          family-direct30 max {e_tot.max():.2f} GV / "
              f"{np.max(e_tot / d30[i]) * 100:.1f}%  (vs the published table)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sections", nargs="*",
                   default=["vertical", "continuity", "validate"])
    p.add_argument("--n-jobs", type=int, default=24)
    a = p.parse_args(argv)
    fam = _family(n_jobs=a.n_jobs)
    if "vertical" in a.sections:
        sec_vertical(fam)
    if "continuity" in a.sections:
        sec_continuity(fam)
    if "validate" in a.sections:
        sec_validate(fam, n_jobs=a.n_jobs)
    print("DIAG_CUTOFF_ANCHOR_DONE")


if __name__ == "__main__":
    main()
