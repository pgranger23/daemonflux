"""Offline tests for the P_N / S_N large-angle angular transport."""

import numpy as np

from sn_transport import (
    sn_sigma_from_theta1,
    legendre_coeffs,
    reconstruct,
    sigma_theta_of,
    gaussian_f1,
    ISO_SIGMA_RAD,
)


def test_small_angle_matches_sqrtN():
    # For a narrow step, arccos<cos> -> sqrt(N) theta1 (the Gaussian/FP limit).
    theta1 = np.radians(2.0)
    for n in (1, 4, 9):
        sn = sn_sigma_from_theta1(theta1, n)
        assert np.isclose(sn, np.sqrt(n) * theta1, rtol=2e-2)


def test_bounded_below_90_deg():
    # Any single step / generation count gives spread < 90 deg (arccos of >0).
    for theta1_deg, n in [(30, 50), (60, 100), (45, 200)]:
        sn = np.degrees(sn_sigma_from_theta1(np.radians(theta1_deg), n))
        assert sn <= 90.0 + 1e-9  # bounded by 90 deg (saturates, never exceeds)
    # and it approaches 90 deg for very large kappa
    big = np.degrees(sn_sigma_from_theta1(np.radians(60), 500))
    assert big > 88.0


def test_monotonic_in_generations():
    th = np.radians(10.0)
    vals = [sn_sigma_from_theta1(th, n) for n in (1, 2, 5, 10, 30)]
    assert np.all(np.diff(vals) > 0)


def test_legendre_roundtrip_constant():
    # A uniform density 1/(4pi): c_0 = 1, higher c_l ~ 0; reconstruct -> 1/(4pi).
    mu = np.linspace(-1, 1, 2001)
    f = np.full_like(mu, 1.0 / (4 * np.pi))
    c = legendre_coeffs(mu, f, lmax=20)
    assert np.isclose(c[0], 1.0, atol=1e-3)
    assert np.allclose(c[1:], 0.0, atol=1e-3)
    f_rec = reconstruct(mu, c)
    # ~0.2% endpoint (Gibbs) ringing from the finite lmax; interior is exact.
    assert np.allclose(f_rec, 1.0 / (4 * np.pi), rtol=2e-2)


def test_gaussian_f1_normalized():
    mu = np.linspace(-1, 1, 4001)
    f = gaussian_f1(mu, np.radians(15.0))
    assert np.isclose(2 * np.pi * np.trapezoid(f, mu), 1.0, atol=1e-3)


def test_sigma_of_isotropic():
    mu = np.linspace(-1, 1, 4001)
    f = np.full_like(mu, 1.0 / (4 * np.pi))  # isotropic
    assert np.isclose(sigma_theta_of(mu, f), ISO_SIGMA_RAD, rtol=1e-2)
