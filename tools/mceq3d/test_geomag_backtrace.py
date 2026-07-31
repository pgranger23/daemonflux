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
