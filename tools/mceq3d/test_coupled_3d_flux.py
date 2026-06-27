"""Offline tests for the 3D coupling (convolution + sigma grid; no MCEq)."""

import numpy as np

from coupled_3d_flux import convolve_sphere, sigma_theta_grid


def _cosgrid(n=201):
    return np.linspace(-1.0, 1.0, n)


def test_zero_sigma_returns_1d():
    cos_full = _cosgrid()
    flux = (2.0 + cos_full)[:, None]  # linear in cos, one energy
    out_cos = np.linspace(0.1, 1.0, 8)
    sigma = np.full((len(out_cos), 1), 1e-6)
    res = convolve_sphere(None, cos_full, flux, sigma, out_cos, [0], n_mc=2000)
    expected = np.interp(out_cos, cos_full, flux[:, 0])
    assert np.allclose(res[:, 0], expected, atol=1e-6)


def test_constant_flux_stays_constant():
    cos_full = _cosgrid()
    flux = np.ones((len(cos_full), 1))
    out_cos = np.linspace(0.05, 1.0, 10)
    sigma = np.full((len(out_cos), 1), 0.4)  # 23 deg spread
    res = convolve_sphere(None, cos_full, flux, sigma, out_cos, [0], n_mc=40000, seed=1)
    assert np.allclose(res[:, 0], 1.0, atol=0.02)


def test_smearing_reduces_a_peak():
    # Flux peaked at the horizon (cos=0); spreading must lower the peak.
    cos_full = _cosgrid()
    flux = np.exp(-((cos_full / 0.2) ** 2))[:, None]
    out_cos = np.array([0.02, 0.5])
    sigma = np.full((2, 1), 0.5)
    res = convolve_sphere(None, cos_full, flux, sigma, out_cos, [0], n_mc=40000, seed=2)
    f1d = np.interp(out_cos, cos_full, flux[:, 0])
    assert res[0, 0] < f1d[0]  # peak (horizon) reduced
    assert res[1, 0] > f1d[1]  # flank raised


def test_high_sigma_flattens_more_than_low():
    cos_full = _cosgrid()
    flux = np.exp(-((cos_full / 0.2) ** 2))[:, None]
    out_cos = np.array([0.02])
    lo = convolve_sphere(
        None, cos_full, flux, np.full((1, 1), 0.2), out_cos, [0], n_mc=40000, seed=3
    )
    hi = convolve_sphere(
        None, cos_full, flux, np.full((1, 1), 0.6), out_cos, [0], n_mc=40000, seed=3
    )
    assert hi[0, 0] < lo[0, 0]  # more spread -> peak more reduced


def test_sigma_grid_is_zenith_independent():
    e_sig = np.logspace(0, 3, 30)
    theta2 = (0.3 / e_sig) ** 2
    e_grid = np.array([1.0, 10.0])
    zeniths = np.array([0.0, 80.0])  # vertical, near-horizon
    sig = sigma_theta_grid(e_grid, e_sig, theta2, zeniths)
    assert sig.shape == (2, 2)
    # production-angle spread is zenith-independent (no spurious sec-theta growth)
    assert np.allclose(sig[1], sig[0])
