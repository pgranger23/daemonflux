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


def test_east_west_sign():
    # Positive cosmic rays are easier from the West: at the equator, zenith 45,
    # a rigidity (~18 GV) between the two cutoffs (West ~11.5, East ~24 GV) is
    # ALLOWED from the West but FORBIDDEN from the East. (from-azimuth: 90=E,270=W)
    kw = dict(m_hat=MZ, ds=8.0e4, max_s=6.0e8, r_escape=15.0)
    assert is_allowed(0.0, 0.0, 45.0, 270.0, 18.0, **kw) is True  # from West
    assert is_allowed(0.0, 0.0, 45.0, 90.0, 18.0, **kw) is False  # from East
