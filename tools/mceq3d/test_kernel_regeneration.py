"""Offline tests for the kernel-regeneration pipeline (toy backend only).

These do not require chromo or MCEq; they validate the histogramming,
marginalization, angle conversion and consistency-check logic against the toy
generator, whose marginals are known analytically.
"""

import numpy as np

from kernel_regeneration import (
    KernelGrid,
    ToySource,
    build_kernel,
    build_angular_kernel,
    build_moments,
    marginalize_pt,
    marginalize_theta,
    to_angular_kernel,
    compare_to_reference,
    M_PION,
)


def _small_grid():
    return KernelGrid(
        xl_edges=np.logspace(-3, 0, 31),
        pt_edges=np.linspace(0.0, 3.0, 21),
        proj_energies=np.array([100.0, 1000.0]),
    )


def test_marginal_matches_toy_analytic_shape():
    grid = _small_grid()
    src = ToySource()
    kernel = build_kernel(src, grid, n_interactions=60000)
    marginal = marginalize_pt(kernel, grid)
    reference = np.tile(src.analytic_xl_marginal(grid), (len(grid.proj_energies), 1))

    stats = compare_to_reference(marginal, reference, rtol=0.10)
    # Toy reference is the same distribution, so the normalization is ~1 ...
    assert 0.8 < stats["norm_factor"] < 1.25
    # ... and most populated bins agree within 10% (statistics-limited).
    assert stats["frac_within_rtol"] > 0.7


def test_marginalize_is_total_yield():
    # Integrating the 2D kernel over both x_L and p_T must equal the mean
    # multiplicity per interaction inside the grid coverage.
    grid = _small_grid()
    src = ToySource(mean_mult=6.0)
    kernel = build_kernel(src, grid, n_interactions=80000)
    dxl = np.diff(grid.xl_edges)
    total = np.sum(marginalize_pt(kernel, grid) * dxl[None, :], axis=1)
    # Grid starts at x_L=1e-3, so a fraction of the soft tail is outside it;
    # the captured multiplicity should be a sizable, stable fraction of 6.
    assert np.all(total > 2.0)
    assert np.all(total < 6.5)


def test_angle_conversion_monotonicity():
    grid = _small_grid()
    theta, jac = to_angular_kernel(np.ones(grid.shape), grid)
    # angle grows with p_T at fixed (E, x_L)
    assert np.all(np.diff(theta, axis=2) > 0)
    # angle shrinks with projectile energy at fixed (x_L, p_T)
    assert np.all(theta[0] >= theta[1] - 1e-12)
    assert np.all(jac > 0)


def test_angular_kernel_matches_pt_marginal():
    # Two fresh ToySource instances draw identical secondaries (fixed seed), so
    # the p_T kernel and the directly-binned angular kernel must yield the same
    # dN/dx_L when each is marginalized over its second axis.
    grid = _small_grid()
    m_pt = marginalize_pt(build_kernel(ToySource(), grid, n_interactions=40000), grid)
    th_edges = np.linspace(0.0, 90.0, 121)  # wide enough to capture all angles
    ang = build_angular_kernel(
        ToySource(), grid, th_edges, M_PION, n_interactions=40000
    )
    m_th = marginalize_theta(ang, th_edges)
    mask = m_pt > 1e-6
    assert np.allclose(m_pt[mask], m_th[mask], rtol=1e-6)


def test_angular_kernel_shape_and_nonneg():
    grid = _small_grid()
    th_edges = np.linspace(0.0, 40.0, 31)
    ang = build_angular_kernel(
        ToySource(), grid, th_edges, M_PION, n_interactions=20000
    )
    assert ang.shape == (len(grid.proj_energies), len(grid.xl_edges) - 1, 30)
    assert np.all(ang >= 0)


def test_build_moments_shapes_and_signs():
    grid = _small_grid()
    mom = build_moments(ToySource(), grid, M_PION, n_interactions=20000)
    nE, nxl = len(grid.proj_energies), len(grid.xl_edges) - 1
    assert mom["theta_mean"].shape == (nE, nxl)
    assert mom["theta_sq"].shape == (nE, nxl)
    assert mom["dndx"].shape == (nE, nxl)
    fin = np.isfinite(mom["theta_mean"])
    assert np.all(mom["theta_mean"][fin] >= 0)
    assert np.all(mom["theta_sq"][np.isfinite(mom["theta_sq"])] >= 0)


def test_compare_removes_constant_normalization():
    ref = np.array([[1.0, 2.0, 4.0, 8.0]])
    stats = compare_to_reference(ref * 4.37, ref, rtol=1e-6)
    assert np.isclose(stats["norm_factor"], 4.37, rtol=1e-6)
    assert stats["frac_within_rtol"] == 1.0
    assert stats["raw_frac_within_rtol"] == 0.0
