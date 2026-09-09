"""Diagnostic: geomagnetic frame / direction conventions and the azimuthal
dipole phase of the back-traced cutoff map.

Sections
--------
1. field   -- IGRF at Kamioka (surface, 30 km) from ``geomag_backtrace`` vs a
              direct ``ppigrf`` call and the published declination/inclination.
2. dipole  -- the IGRF centred-dipole axis from (g10, g11, h11).
3. stoermer-- back-traced lower cutoff vs the analytic Stoermer formula for an
              ALIGNED centred dipole (where Stoermer is exact): checks the
              azimuth convention (E > W for protons) and the N/S symmetry.
4. phase   -- R_c(azimuth) of a cutoff map: peak azimuth and first-harmonic
              phase (the two differ a lot -- R_c(az) is not a sinusoid).

Run::

    python diag_dipole_phase.py --sections field dipole stoermer phase
"""

from __future__ import annotations

import argparse
import datetime

import numpy as np

import geomag_backtrace as gb

LAT, LON = 36.43, 137.31
DATE = datetime.datetime(2020, 1, 1)

C_STOERMER = 59.6  # GV RE^2 -- Stoermer constant for the Earth's dipole moment


def stoermer_rc(mag_lat_deg, zenith_deg, az_from_mag_north_deg, r_re=1.0):
    """Analytic Stoermer cutoff [GV] for a centred dipole.

    ``az`` is the *arrival* (from-) azimuth measured clockwise from magnetic
    north, so ``az=90`` is from the magnetic East -- the hardest direction for
    positive particles.
    """
    la = np.deg2rad(mag_lat_deg)
    k = (
        np.cos(la) ** 3
        * np.sin(np.deg2rad(zenith_deg))
        * np.sin(np.deg2rad(az_from_mag_north_deg))
    )
    return C_STOERMER * np.cos(la) ** 4 / (r_re**2 * (1 + np.sqrt(1 - k)) ** 2)


def sec_field():
    import ppigrf

    print("=" * 78)
    print("1. IGRF field at Kamioka (36.43 N, 137.31 E), 2020.0")
    print("=" * 78)
    print(f"{'source':34s} {'Bn':>9s} {'Be':>9s} {'Bu':>9s} {'F[uT]':>7s} "
          f"{'D[deg]':>8s} {'I[deg]':>8s}")
    up, north, east = gb._local_frame(LAT, LON)
    for h_km in (0.0, 30.0):
        Be, Bn, Bu = (float(np.ravel(c)[0]) for c in ppigrf.igrf(LON, LAT, h_km, DATE))
        _row(f"ppigrf.igrf (geodetic) h={h_km:.0f}km", Bn, Be, Bu)
        B = gb._bfield_igrf_cart((gb.RE + h_km * 1e3) * up[None, :], DATE,
                                 gb.dipole_axis())[0] * 1e9
        _row(f"_bfield_igrf_cart h={h_km:.0f}km", B @ north, B @ east, B @ up)
    print("published IGRF-13 2020 at Kamioka: D ~ -7.5..-8.2, I ~ +49..+51, "
          "F ~ 46-48 uT")
    print("(the module works on a geocentric sphere of radius RE, so its "
          "inclination runs ~0.4 deg steep -- the geodetic/geocentric latitude "
          "difference at 36.4 N is 0.19 deg -- and |B| ~0.1 uT high)")


def _row(tag, Bn, Be, Bu):
    F = np.sqrt(Bn**2 + Be**2 + Bu**2)
    D = np.degrees(np.arctan2(Be, Bn))
    I = np.degrees(np.arctan2(-Bu, np.hypot(Be, Bn)))
    print(f"{tag:34s} {Bn:9.1f} {Be:9.1f} {Bu:9.1f} {F/1000:7.2f} {D:8.2f} {I:8.2f}")


def sec_dipole():
    print("=" * 78)
    print("2. IGRF centred-dipole axis")
    print("=" * 78)
    v = np.array([gb.G11, gb.H11, gb.G10], float)
    v /= np.linalg.norm(v)

    def ll(m):
        return np.degrees(np.arcsin(m[2])), np.degrees(np.arctan2(m[1], m[0]))

    print("(g11,h11,g10)^        -> lat %+7.2f lon %+8.2f  [geomagnetic SOUTH "
          "pole, expect 80.65 S, 107.32 E]" % ll(v))
    print("-(g11,h11,g10)^       -> lat %+7.2f lon %+8.2f  [geomagnetic NORTH "
          "pole, expect 80.65 N, 72.68 W]" % ll(-v))
    m = gb.dipole_axis()
    print("dipole_axis()         -> lat %+7.2f lon %+8.2f" % ll(m))
    print("angle(dipole_axis, correct geomagnetic north) = %.2f deg"
          % np.degrees(np.arccos(np.clip(m @ (-v), -1, 1))))
    up, north, east = gb._local_frame(LAT, LON)
    for tag, mm in (("dipole_axis()", m), ("correct", -v)):
        B = gb.bfield(gb.RE * up, mm)
        D = np.degrees(np.arctan2(B @ east, B @ north))
        glat = 90 - np.degrees(np.arccos(np.clip(up @ mm, -1, 1)))
        print(f"  centred dipole {tag:14s}: Kamioka geomagnetic lat {glat:6.2f}, "
              f"declination {D:+6.2f} (mag-East azimuth {90+D:6.2f})")


def _pure_dipole_field():
    """Monkeypatch context: force the vectorised tracer onto the centred dipole.

    ``_bfield_igrf_cart(..., r_switch=0)`` never takes the real-IGRF branch, so
    the fast vectorised tracer integrates in exactly the field of the scalar
    :func:`geomag_backtrace.bfield` -- the regime where Stoermer is exact.
    """
    import contextlib
    import functools

    @contextlib.contextmanager
    def cm():
        orig = gb._bfield_igrf_cart
        gb._bfield_igrf_cart = functools.partial(orig, r_switch=0.0)
        try:
            yield
        finally:
            gb._bfield_igrf_cart = orig

    return cm()


def dipole_admittance(lat, zeniths, azimuths, rigidities, m_hat=None, charge=+1,
                      lon=0.0, ds=5.0e4, r_escape=15.0, max_s=1.2e9, n_jobs=16):
    """0/1 admittance ``(n_dir, n_R)`` in a pure centred dipole (vectorised)."""
    import multiprocessing as mp

    if m_hat is None:
        m_hat = np.array([0.0, 0.0, 1.0])
    dirs = [(z, a) for z in zeniths for a in azimuths]
    u0 = np.array([-gb.arrival_direction(lat, lon, z, a) for z, a in dirs])
    r0 = np.tile(gb.RE * gb._local_frame(lat, lon)[0], (len(dirs), 1))
    idx = np.repeat(np.arange(len(dirs)), len(rigidities))
    R = np.tile(np.asarray(rigidities, float), len(dirs))
    args = [
        (r0[idx[c]], u0[idx[c]], R[c], m_hat, charge, r_escape, ds, max_s)
        for c in np.array_split(np.arange(len(idx)), n_jobs)
    ]
    with _pure_dipole_field():
        with mp.get_context("fork").Pool(len(args)) as pool:
            out = pool.map(_adm_worker, args)
    return np.concatenate(out).reshape(len(dirs), len(rigidities)), dirs


def _adm_worker(a):
    r0, u0, R, m_hat, charge, r_escape, ds, max_s = a
    return gb.backtrace_vec(r0, u0, R, DATE, m_hat, charge, r_escape=r_escape,
                            ds=ds, max_s=max_s)


def sec_stoermer(lat=30.0, zeniths=(0.0, 45.0, 80.0, 88.0), r_lo=1.0, r_hi=70.0,
                 step=0.25, n_jobs=16):
    """Aligned-dipole back-trace vs the analytic Stoermer cutoff.

    The site latitude IS the geomagnetic latitude here (aligned dipole), so the
    Stoermer formula is exact and every convention in the tracer is pinned:
    the azimuth sense (E harder than W for protons), the N/S symmetry of the
    main cone, and the absolute normalisation (59.6 cos^4 lambda / 4 vertical).
    """
    print("=" * 78)
    print(f"3. Aligned centred dipole (m_hat = +z), site at latitude {lat:g} deg")
    print("   back-traced R_L (lowest allowed) / R_U (highest forbidden) vs "
          "Stoermer")
    print("=" * 78)
    az = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
    rs = np.arange(r_hi, r_lo, -step)
    A, dirs = dipole_admittance(lat, list(zeniths), list(az), rs, n_jobs=n_jobs)
    names = {0: "N", 90: "E", 180: "S", 270: "W"}
    print(f"{'zen':>5s} {'az':>5s} {'dir':>4s} {'Stoermer':>9s} {'R_L':>7s} "
          f"{'R_U':>7s} {'pen':>6s}")
    for k, (z, a) in enumerate(dirs):
        adm = A[k]
        forb = np.where(~adm)[0]
        R_U = rs[forb[0]] if len(forb) else float("nan")
        R_L = rs[forb[-1]] - step if len(forb) else float("nan")
        pen = (np.mean(adm[(rs <= R_U) & (rs >= R_L)]) if len(forb)
               else float("nan"))
        print(f"{z:5.0f} {a:5.0f} {names.get(int(a), ''):>4s} "
              f"{stoermer_rc(lat, z, a):9.2f} {R_L:7.2f} {R_U:7.2f} {pen:6.2f}")
    print("Expected: R_L ~ Stoermer; E > W (positive particles are cut harder "
          "from the East); N == S exactly for a centred-dipole MAIN CONE "
          "(R_L); any N/S difference lives in the penumbra/shadow (R_U).")


def _phase(rc_row, az_row):
    """(peak azimuth by parabolic interpolation, first-harmonic phase) [deg]."""
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


def sec_phase(path=".cache3d/finerc_4b7188ee3e51511f.npz"):
    print("=" * 78)
    print(f"4. R_c(azimuth) structure of the cutoff map {path}")
    print("=" * 78)
    d = np.load(path)
    zen, az, rc = d["zen"], d["az"], d["rc"]
    azp = az[:-1]  # drop the duplicated 360 node
    print(f"{'zen':>6s} {'peak_az':>8s} {'1st-harm':>9s} {'R_c(N)':>7s} "
          f"{'R_c(E)':>7s} {'R_c(S)':>7s} {'R_c(W)':>7s}")
    for i, z in enumerate(zen):
        if z > 90:
            continue
        row = rc[i, :-1]
        pk, ph = _phase(row, azp)
        g = lambda a: np.interp(a, az, rc[i])  # noqa: E731
        print(f"{z:6.1f} {pk:8.2f} {ph:9.2f} {g(351.9):7.2f} {g(81.9):7.2f} "
              f"{g(171.9):7.2f} {g(261.9):7.2f}")
    print("N/E/S/W columns are the GEOMAGNETIC cardinal azimuths "
          "(declination -8.1 deg).")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sections", nargs="*",
                   default=["field", "dipole", "stoermer", "phase"])
    p.add_argument("--map", default=".cache3d/finerc_4b7188ee3e51511f.npz")
    p.add_argument("--lat", type=float, default=30.0)
    a = p.parse_args(argv)
    if "field" in a.sections:
        sec_field()
    if "dipole" in a.sections:
        sec_dipole()
    if "stoermer" in a.sections:
        sec_stoermer(lat=a.lat)
    if "phase" in a.sections:
        sec_phase(a.map)
    if "prodpoint" in a.sections:
        sec_prodpoint()




def sec_prodpoint(h_prod_km=30.0, zeniths=(80.0, 87.0), n_jobs=16):
    """R_c at the DETECTOR vs at the PRODUCTION POINT, for the same direction.

    A near-horizon neutrino is made ~L = h/cos(zenith)... = several hundred km up
    the arrival ray, where the *site* -- not just the local vertical -- has moved
    by L/R_E in latitude and longitude.  The engine's cutoff map is built by
    launching at the detector, so it charges every near-horizon direction the
    detector's cutoff.  This section measures how much of the map's North/South
    skew is an artefact of that: from the North the production point sits
    several degrees closer to the geomagnetic pole (LOWER cutoff), from the
    South several degrees closer to the equator (HIGHER cutoff).
    """
    print("=" * 78)
    print(f"5. Detector-launched vs production-point-launched R_c "
          f"(h_prod = {h_prod_km:g} km), Kamioka")
    print("=" * 78)
    az = np.arange(0.0, 360.0, 15.0)
    up = gb._local_frame(LAT, LON)[0]
    r_det = gb.RE * up
    h = h_prod_km * 1e3
    rows = {}
    for tag in ("detector", "prod_point"):
        r0, u0 = [], []
        for z in zeniths:
            for a in az:
                d = gb.arrival_direction(LAT, LON, z, a)
                if tag == "detector":
                    r0.append(r_det)
                else:
                    # walk back up the arrival ray to altitude h_prod
                    b = 2.0 * np.dot(r_det, -d)
                    c = gb.RE**2 - (gb.RE + h) ** 2
                    L = 0.5 * (-b + np.sqrt(max(b * b - 4 * c, 0.0)))
                    r0.append(r_det + L * (-d))
                u0.append(-d)
        rc, _ = gb.cutoff_from_states(
            np.array(r0), np.array(u0), DATE, charge=+1, n_jobs=n_jobs,
            r_floor=gb.RE + (0.0 if tag == "detector" else h),
        )
        rows[tag] = rc.reshape(len(zeniths), len(az))
    print(f"{'zen':>5s} {'launch':>11s} {'N':>7s} {'E':>7s} {'S':>7s} {'W':>7s} "
          f"{'N/S':>6s} {'peak':>7s} {'1-harm':>7s}")
    for i, z in enumerate(zeniths):
        for tag in ("detector", "prod_point"):
            row = rows[tag][i]
            pk, ph = _phase(row, az)
            g = np.interp([351.9, 81.9, 171.9, 261.9], list(az) + [360.0],
                          list(row) + [row[0]])
            print(f"{z:5.0f} {tag:>11s} {g[0]:7.2f} {g[1]:7.2f} {g[2]:7.2f} "
                  f"{g[3]:7.2f} {g[0]/g[2]:6.2f} {pk:7.2f} {ph:7.2f}")
        print("      det : " + " ".join(f"{v:5.1f}" for v in rows["detector"][i]))
        print("      prod: " + " ".join(f"{v:5.1f}" for v in rows["prod_point"][i]))

if __name__ == "__main__":
    main()
