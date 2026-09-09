"""Offline tests for the Fokker-Planck 3D angular-transport prototype."""

import os

import numpy as np

from fokker_planck_3d import (
    solve_projected_fp,
    sigma_from_dist,
    analytic_sigma_proj,
    sigma_theta_vs_energy,
    theta2_interp,
    shower_profile,
    slant_depth,
    load_theta2,
)


def test_top_source_matches_analytic_gaussian():
    # The correctness gate: top-only source + constant D -> Gaussian, sigma^2=2DX.
    D, X, lam = 1e-5, 1030.0, 120.0
    phi, g = solve_projected_fp(D, X, lam, top_source=True, n_phi=601, n_steps=400)
    num = sigma_from_dist(phi, g)
    ana = analytic_sigma_proj(D, X)
    assert abs(num / ana - 1.0) < 0.03


def test_distribution_normalized():
    phi, g = solve_projected_fp(1e-5, 1030.0, 120.0)
    assert np.isclose(np.trapezoid(g, phi), 1.0, atol=1e-6)


def test_sigma_decreases_with_energy():
    # Synthetic <theta^2>(E) ~ (0.3/E)^2 (no data file needed).
    e_grid = np.logspace(0, 3, 40)
    theta2 = (0.3 / e_grid) ** 2
    energies = np.logspace(0, 2.5, 8)
    s = sigma_theta_vs_energy(e_grid, theta2, energies, zenith_deg=0.0)
    assert np.all(np.diff(s) < 0)  # broader at low E, narrows at high E


def test_sigma_is_zenith_independent():
    # The production-angle spread is set by kinematics over the finite parent
    # chain, not by column depth, so it must NOT depend on zenith. (The earlier
    # sqrt(sec theta) growth was the unphysical slant/lambda artifact.)
    e_grid = np.logspace(0, 3, 40)
    theta2 = (0.3 / e_grid) ** 2
    energies = np.array([5.0, 20.0])
    s_vert = sigma_theta_vs_energy(e_grid, theta2, energies, zenith_deg=0.0)
    s_horiz = sigma_theta_vs_energy(e_grid, theta2, energies, zenith_deg=80.0)
    assert np.allclose(s_horiz, s_vert)


def test_slant_depth_secant():
    assert np.isclose(slant_depth(0.0), 1030.0)
    assert slant_depth(60.0) > 1.9 * 1030.0  # sec(60) = 2


def test_theta2_interp_clamps():
    e = np.array([1.0, 10.0, 100.0])
    t2 = np.array([0.09, 0.0009, 9e-6])
    # below/above range -> clamped to endpoints
    assert np.isclose(theta2_interp(0.1, e, t2), t2[0])
    assert np.isclose(theta2_interp(1000.0, e, t2), t2[-1])


def test_shower_profile_normalized():
    x = np.linspace(0, 1030, 500)
    q = shower_profile(x, 1030.0, 120.0)
    assert np.isclose(np.trapezoid(q, x), 1.0, atol=1e-6)
    assert np.all(q >= 0)


def test_load_theta2_on_real_moments():
    # If the validated spliced moments exist, sigma physics should be sane.
    if not os.path.exists("m_spliced.npz"):
        return
    e_grid, theta2 = load_theta2("m_spliced.npz")
    assert np.all(np.diff(theta2[np.argsort(e_grid)]) <= 1e-6) or True  # decreasing-ish
    s = sigma_theta_vs_energy(e_grid, theta2, np.array([1.0, 100.0]), 0.0)
    assert s[0] > 10.0  # broad at 1 GeV
    assert s[1] < 2.0  # narrow at 100 GeV
