"""Tests for the spherical-streaming P_N transport (curvature term)."""

import numpy as np

from spherical_streaming import (
    solve_spherical,
    diffusion_spherical,
    angular_flux,
)


def test_p1_matches_spherical_diffusion():
    # The P_1 closure reduces analytically to spherical diffusion; the staggered
    # solve must reproduce an independent diffusion-ODE solve (validates curvature).
    r = np.linspace(1.0, 6.0, 300)
    st, cc = 1.0, 0.97
    src = np.exp(-((r - 3.5) ** 2) / (2 * 0.4**2))
    res = solve_spherical(r, n_l=2, sigma_t=st, c=cc, source=src)
    phi0 = res["phi"][0]
    ref = diffusion_spherical(r, st, st * (1 - cc), src, phi0[0], phi0[-1])
    m = (r > 1.5) & (r < 5.5)
    assert np.median(np.abs(phi0[m] / ref[m] - 1.0)) < 1e-2


def test_no_odd_even_decoupling():
    # The staggered grid must not produce a sawtooth in phi_0.
    r = np.linspace(1.0, 6.0, 300)
    src = np.exp(-((r - 3.5) ** 2) / (2 * 0.4**2))
    phi0 = solve_spherical(r, n_l=2, sigma_t=1.0, c=0.97, source=src)["phi"][0]
    d = np.diff(phi0[100:140])
    # fraction of sign flips between consecutive slopes (1.0 == pure sawtooth)
    flips = np.mean(np.sign(d[:-1]) != np.sign(d[1:]))
    assert flips < 0.2


def test_operator_is_sparse():
    r = np.linspace(1, 6, 300)
    A = solve_spherical(r, n_l=16, sigma_t=1.0, c=0.5)["A"]
    assert A.nnz / A.shape[0] ** 2 < 0.01  # well under 1% dense


def test_curvature_redistributes_toward_horizon():
    # Streaming-dominated shell source: detector flux is larger toward the
    # horizon (cos z -> 0) than vertical -- the off-axis curved-shell excess.
    R, H = 30.0, 5.0
    rg = np.linspace(R, R + 2 * H, 250)
    shell = np.exp(-(rg - R) / (0.35 * H))
    phi = solve_spherical(rg, n_l=24, sigma_t=0.02, c=0.0, source=shell)["phi"]
    mu = np.linspace(-0.999, -0.05, 50)
    psi = np.clip(angular_flux(phi[:, 3], mu), 0, None)
    cosz = -mu
    vert = psi[np.argmax(cosz)]
    near_horizon = psi[cosz < 0.2].mean()
    assert near_horizon > vert  # horizon excess


def test_flat_space_limit_reduces_streaming():
    # At very large radius the curvature term (~1/r) vanishes and the spherical
    # operator must approach the slab streaming result (phi_0 shape ~ symmetric).
    r = np.linspace(1000.0, 1005.0, 300)  # r >> domain size -> ~flat
    src = np.exp(-((r - 1002.5) ** 2) / (2 * 0.4**2))
    phi0 = solve_spherical(r, n_l=2, sigma_t=1.0, c=0.97, source=src)["phi"][0]
    # symmetric about the source centre to good accuracy (no curvature bias)
    assert abs(phi0[80] - phi0[-81]) / max(phi0.max(), 1e-9) < 0.05
