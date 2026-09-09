"""Offline tests for the curved-atmosphere production geometry."""

import numpy as np

from spherical_geometry import (
    production_integral,
    horizon_enhancement,
    R_EARTH,
    H0,
)


def test_vertical_normalized():
    assert np.isclose(horizon_enhancement(1.0, spherical=True)[0], 1.0)
    assert np.isclose(horizon_enhancement(1.0, spherical=False)[0], 1.0)


def test_small_zenith_recovers_sec_theta():
    # Near vertical, spherical ~ flat ~ sec(theta).
    cz = 0.8  # ~37 deg
    sph = horizon_enhancement(cz, spherical=True)[0]
    flat = horizon_enhancement(cz, spherical=False)[0]
    assert np.isclose(sph, 1 / cz, rtol=0.05)
    assert np.isclose(sph, flat, rtol=0.02)


def test_horizon_spherical_finite_flat_diverges():
    sph = horizon_enhancement(0.02, spherical=True)[0]
    flat = horizon_enhancement(0.02, spherical=False)[0]
    assert np.isfinite(sph) and sph < flat  # spherical saturates below flat
    # saturation scale ~ sqrt(R/h0)
    assert 0.3 * np.sqrt(R_EARTH / H0) < sph < 3 * np.sqrt(R_EARTH / H0)


def test_monotonic_toward_horizon():
    cz = np.linspace(0.05, 1.0, 20)
    sph = horizon_enhancement(cz, spherical=True)
    assert np.all(np.diff(sph) < 0)  # rises as cosZ decreases (toward horizon)


def test_decay_weight_suppresses_high_alt():
    # A weight that vanishes at high altitude reduces the integral.
    full = production_integral(0.0)
    weighted = production_integral(0.0, decay_weight=lambda h: np.exp(-h / 5.0))
    assert weighted < full
