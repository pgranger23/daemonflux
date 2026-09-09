"""Tests for the geomagnetic back-tracer.

The full cutoff scan is slow (trajectory integration); the module's quantitative
validation vs Stoermer is in ``geomag_backtrace.main``. Here we test the fast
helpers plus single allowed/forbidden traces at extreme rigidities.
"""

import numpy as np

from geomag_backtrace import (
    dipole_axis,
    bfield,
    arrival_direction,
    is_allowed,
    _local_frame,
    RE,
    B0,
)

MZ = np.array([0.0, 0.0, 1.0])  # aligned dipole for the trace tests


def test_dipole_tilt_is_about_9_4_deg():
    m = dipole_axis()
    assert abs(np.rad2deg(np.arccos(m[2])) - 9.4) < 0.5


def test_field_pole_is_twice_equator():
    eq = np.linalg.norm(bfield(np.array([RE, 0, 0]), MZ))  # equator
    pole = np.linalg.norm(bfield(np.array([0, 0, RE]), MZ))  # pole
    assert np.isclose(eq, B0, rtol=1e-6)
    assert np.isclose(pole, 2 * B0, rtol=1e-6)


def test_field_falls_as_r_cubed():
    b1 = np.linalg.norm(bfield(np.array([RE, 0, 0]), MZ))
    b2 = np.linalg.norm(bfield(np.array([2 * RE, 0, 0]), MZ))
    assert np.isclose(b1 / b2, 8.0, rtol=1e-6)


def test_arrival_direction_vertical_is_down():
    d = arrival_direction(0.0, 0.0, 0.0, 0.0)  # equator, vertical
    up = np.array([1.0, 0.0, 0.0])  # equator, lon 0 -> up = +x
    assert np.allclose(d, -up, atol=1e-9)


def test_high_rigidity_allowed_low_forbidden():
    # equator, vertical, aligned dipole: 100 GV >> cutoff -> allowed;
    # 3 GV << ~15 GV cutoff -> forbidden.
    kw = dict(m_hat=MZ, ds=1.0e5, max_s=4.0e8, r_escape=15.0)
    assert is_allowed(0.0, 0.0, 0.0, 0.0, 100.0, **kw) is True
    assert is_allowed(0.0, 0.0, 0.0, 0.0, 3.0, **kw) is False


def test_cutoff_source_factory(monkeypatch):
    # cutoff_source returns a (zenith, azimuth) -> R_c callable that broadcasts.
    # Mock the slow trajectory back-trace; test the factory wiring/shape only.
    import geomag_backtrace as gb

    monkeypatch.setattr(gb, "cutoff_igrf", lambda la, lo, z, a, d, **k: 10.0 + 0.01 * z)
    f = gb.cutoff_source(36.43, 137.31, "2020-01-01")
    out = f(np.array([0.0, 50.0, 80.0]), np.array([90.0, 180.0, 270.0]))
    assert out.shape == (3,)
    assert np.allclose(out, [10.0, 10.5, 10.8])


def test_cached_cutoff_source_builds_caches_and_interpolates(monkeypatch, tmp_path):
    # Mock the slow batched map; test build->cache->reload->bilinear-interp path.
    import geomag_backtrace as gb

    calls = {"n": 0}

    def fake_map(la, lo, d, zen, az, **k):
        calls["n"] += 1
        # R_c rising with zenith, independent of azimuth -> easy to check interp
        return np.tile(10.0 + 0.05 * np.asarray(zen)[:, None], (1, len(az)))

    monkeypatch.setattr(gb, "cutoff_map", fake_map)
    f = gb.cached_cutoff_source(
        36.43, 137.31, "2020-01-01", n_zen=5, n_az=5, cache_dir=str(tmp_path)
    )
    assert calls["n"] == 1  # built once
    # bilinear interpolation between zenith nodes (azimuth-independent here)
    assert np.isclose(f(0.0, 0.0), 10.0, atol=1e-6)
    assert np.isclose(f(45.0, 123.0), 10.0 + 0.05 * 45.0, atol=1e-6)
    # second call reuses the cache file (no rebuild)
    g = gb.cached_cutoff_source(
        36.43, 137.31, "2020-01-01", n_zen=5, n_az=5, cache_dir=str(tmp_path)
    )
    assert calls["n"] == 1  # not rebuilt
    assert np.isclose(g(45.0, 0.0), 10.0 + 0.05 * 45.0, atol=1e-6)


def test_igrf_field_magnitude():
    # Full-IGRF field (ppigrf) at the surface has a realistic magnitude
    # (~25-65 uT depending on location); check a near-surface point.
    import datetime
    from geomag_backtrace import _bfield_igrf_cart

    r = np.array(
        [[RE * np.cos(np.deg2rad(36.43)), 0.0, RE * np.sin(np.deg2rad(36.43))]]
    )
    B = _bfield_igrf_cart(r, datetime.datetime(2020, 1, 1), dipole_axis())
    mag_uT = np.linalg.norm(B[0]) * 1e6
    assert 25.0 < mag_uT < 65.0


def test_far_field_matches_validated_dipole_beyond_switch():
    """Beyond r_switch the vectorised field must equal the validated scalar
    :func:`bfield`, INCLUDING sign.

    Regression guard: the far-field branch of ``_bfield_igrf_cart`` originally
    omitted the leading minus of the dipole formula, so it returned -B for
    r > r_switch. It survived because the only existing coverage
    (``test_igrf_field_magnitude``) probes a near-surface point -- inside the
    switch radius, where the real IGRF overwrites the dipole -- and checks a
    magnitude, which is sign-blind. Test both branches and the sign here.
    """
    import datetime
    from geomag_backtrace import _bfield_igrf_cart, bfield

    m_hat = dipole_axis()
    date = datetime.datetime(2020, 1, 1)
    for r_re in (5.0, 8.0):  # comfortably beyond the default r_switch=4
        pos = np.array([[RE * r_re * 0.6, RE * r_re * 0.0, RE * r_re * 0.8]])
        got = _bfield_igrf_cart(pos, date, m_hat)[0]
        want = bfield(pos[0], m_hat)
        assert np.allclose(got, want, rtol=1e-10), (
            f"far-field dipole disagrees with validated bfield() at r={r_re} R_E: "
            f"{got} vs {want} (ratio {got / want})"
        )


def test_east_west_sign():
    # Positive cosmic rays are easier from the West: at the equator, zenith 45,
    # a rigidity (~18 GV) between the two cutoffs (West ~11.5, East ~24 GV) is
    # ALLOWED from the West but FORBIDDEN from the East. (from-azimuth: 90=E,270=W)
    kw = dict(m_hat=MZ, ds=8.0e4, max_s=6.0e8, r_escape=15.0)
    assert is_allowed(0.0, 0.0, 45.0, 270.0, 18.0, **kw) is True  # from West
    assert is_allowed(0.0, 0.0, 45.0, 90.0, 18.0, **kw) is False  # from East


# ---------------------------------------------------------------------------
# Frame / convention pins
# ---------------------------------------------------------------------------
def test_dipole_axis_points_at_the_geomagnetic_north_pole():
    """The axis must be the geomagnetic NORTH pole: 80.6 N, 72.7 W (2020).

    Regression guard.  ``dipole_axis`` used to take the tilt from ``-g10`` but
    the longitude from ``atan2(h11, g11)``.  ``(g11, h11, g10)`` points at the
    geomagnetic *SOUTH* pole (80.6 S, 107.3 E), so keeping that longitude while
    flipping the latitude put the axis at 80.6 N, **107.3 E** -- 18.8 deg from
    the truth, i.e. the dipole tilt leaning to the opposite side of the globe.
    Only the tilt angle was covered before (``test_dipole_tilt_is_about_9_4``),
    and the tilt is unchanged by the error.
    """
    m = dipole_axis()
    lat = np.degrees(np.arcsin(m[2]))
    lon = np.degrees(np.arctan2(m[1], m[0]))
    assert abs(lat - 80.6) < 0.5, lat
    assert abs(lon - (-72.7)) < 1.0, lon


def test_dipole_axis_gives_the_right_geomagnetic_latitude_at_kamioka():
    """Physical consequence of the axis: Kamioka sits at geomagnetic ~26-29 N.

    With the old (180-deg-rotated) axis Kamioka came out at 44.4 N, which is a
    factor ~2 in ``cos^4(lambda)`` -- i.e. a completely different Stoermer
    cutoff for the far-field part of every back-trace.
    """
    up = _local_frame(36.43, 137.31)[0]
    glat = 90.0 - np.degrees(np.arccos(np.clip(up @ dipole_axis(), -1, 1)))
    assert 25.0 < glat < 30.0, glat


def test_arrival_direction_azimuth_is_clockwise_from_north():
    """``azimuth`` is the compass bearing OF THE SOURCE (N=0, E=90, S=180,
    W=270) and the returned vector is the particle VELOCITY, so it points away
    from the source and downwards."""
    lat, lon = 0.0, 0.0
    up, north, east = _local_frame(lat, lon)
    # horizontal arrivals: velocity is exactly opposite the source bearing
    for az, src in ((0.0, north), (90.0, east), (180.0, -north), (270.0, -east)):
        d = arrival_direction(lat, lon, 90.0, az)
        assert np.allclose(d, -src, atol=1e-12), (az, d)
    # a 60 deg zenith arrival from the East: down-going, horizontal part West
    d = arrival_direction(lat, lon, 60.0, 90.0)
    assert np.isclose(d @ up, -np.cos(np.radians(60.0)))
    assert np.isclose(d @ east, -np.sin(np.radians(60.0)))
    assert np.isclose(d @ north, 0.0, atol=1e-12)


def test_backtrace_launch_state_is_the_reversed_trajectory(monkeypatch):
    """The scan must launch FROM the detector, AWAY from it, along ``-d_hat``
    (the reversed velocity); the charge reversal is in ``backtrace_vec``.

    A sign slip in either the launch direction or the charge mirrors or rotates
    the whole cutoff sky, so pin the launch state explicitly.
    """
    import geomag_backtrace as gb

    seen = {}

    def fake(r0, u0, R, date, m_hat, charge=+1, **kw):
        seen["r0"], seen["u0"], seen["q"] = np.array(r0), np.array(u0), charge
        return np.ones(len(np.atleast_1d(R)), bool)  # everything allowed

    monkeypatch.setattr(gb, "backtrace_vec", fake)
    lat, lon, zen, az = 36.43, 137.31, 60.0, 90.0
    gb.cutoff_igrf(lat, lon, zen, az, "2020-01-01")
    up, north, east = _local_frame(lat, lon)
    assert np.allclose(seen["r0"][0], RE * up)  # launched at the detector
    assert np.allclose(seen["u0"][0], -arrival_direction(lat, lon, zen, az))
    assert seen["u0"][0] @ up > 0  # leaves the Earth
    assert seen["u0"][0] @ east > 0  # towards the source (East) bearing
    assert seen["q"] == +1  # the reversal is inside backtrace_vec, not here


# ---------------------------------------------------------------------------
# Stoermer physics pins (pure centred dipole, where the analytic answer holds)
# ---------------------------------------------------------------------------
MZ_DATE = "2020-01-01"


def _pure_dipole_cutoff(lat, zeniths, azimuths, n_jobs=8):
    """Cutoff map in a centred ALIGNED dipole (``r_switch=0`` -> no IGRF)."""
    import geomag_backtrace as gb

    return gb.cutoff_map(
        lat, 0.0, MZ_DATE, zeniths, azimuths, m_hat=MZ, r_lo=0.5, r_hi=70.0,
        n_jobs=n_jobs, r_switch=0.0, warn_saturated=False,
    )


def test_vertical_cutoff_matches_stoermer_in_an_aligned_dipole():
    """Absolute normalisation: R_c(vertical) = 14.9 cos^4(lambda) GV.

    This pins B0, the rigidity-to-curvature conversion and the escape criterion
    all at once; the aligned dipole is the only case where Stoermer is exact.
    """
    for lat, want in ((0.0, 14.9), (30.0, 14.9 * np.cos(np.radians(30.0)) ** 4)):
        got = _pure_dipole_cutoff(lat, [0.0], [0.0], n_jobs=1)[0, 0]
        assert abs(got - want) / want < 0.15, (lat, got, want)


def test_east_cutoff_exceeds_west_and_peaks_at_magnetic_east():
    """Stoermer sign + axis, in the aligned dipole where magnetic East IS az 90.

    * positive particles are cut harder from the East than from the West;
    * the near-horizon maximum of R_c(azimuth) sits at magnetic East to within
      one 15-deg azimuth bin.

    NOTE what this does *not* claim: R_c(az) is strongly skewed towards the
    North (the shadow cone; from the North the reversed trajectory mirrors in
    the converging field and is blocked far above the Stoermer main cone), so
    its FIRST FOURIER HARMONIC peaks ~20 deg clockwise of magnetic East even
    here.  That phase shift is a property of the shadow cone, not a frame error.
    """
    az = np.arange(0.0, 360.0, 15.0)
    rc = _pure_dipole_cutoff(30.0, [80.0], az)[0]
    i_e, i_w = int(np.argmin(abs(az - 90))), int(np.argmin(abs(az - 270)))
    assert rc[i_e] > 3.0 * rc[i_w], (rc[i_e], rc[i_w])
    peak_az = az[int(np.argmax(rc))]
    assert min(abs(peak_az - 90.0), abs(peak_az - 90.0 + 360)) <= 15.0, peak_az
    # ... and the North/South cut is NOT symmetric: the shadow cone is.
    i_n, i_s = int(np.argmin(abs(az - 0))), int(np.argmin(abs(az - 180)))
    assert rc[i_n] > 1.5 * rc[i_s], (rc[i_n], rc[i_s])
