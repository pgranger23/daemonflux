"""Offline tests for the angular-kernel extraction (synthetic kernels)."""

import numpy as np

from angular_kernel import (
    production_angle,
    angle_moments,
    mean_angle_vs_energy,
    discrete_ordinate_row,
    angular_row,
    pool_moments_by_energy,
    crossover_energy,
    M_PION,
)
from kernel_regeneration import KernelGrid, ToySource, build_moments


def _angular_axes(proj_energies, theta_max=40.0, ntheta=40):
    xl_edges = np.logspace(-3, 0, 31)
    th_edges = np.linspace(0.0, theta_max, ntheta + 1)
    return {
        "proj_energies": np.asarray(proj_energies, float),
        "xl_edges": xl_edges,
        "xl_centers": np.sqrt(xl_edges[:-1] * xl_edges[1:]),
        "is_angular": True,
        "theta_edges": th_edges,
        "theta_centers": 0.5 * (th_edges[:-1] + th_edges[1:]),
    }


def _axes(proj_energies):
    xl_edges = np.logspace(-3, 0, 31)
    pt_edges = np.linspace(0.0, 3.0, 31)
    return {
        "proj_energies": np.asarray(proj_energies, float),
        "xl_edges": xl_edges,
        "pt_edges": pt_edges,
        "xl_centers": np.sqrt(xl_edges[:-1] * xl_edges[1:]),
        "pt_centers": 0.5 * (pt_edges[:-1] + pt_edges[1:]),
    }


def test_production_angle_matches_kinematics():
    axes = _axes([100.0])
    theta = production_angle(axes)
    # at a chosen (x_L, p_T) bin, theta == arctan(p_T / p_L)
    j, k = 20, 5
    e_sec = axes["xl_centers"][j] * axes["proj_energies"][0]
    p_l = np.sqrt(e_sec**2 - M_PION**2)
    expected = np.arctan2(axes["pt_centers"][k], p_l)
    assert np.isclose(theta[0, j, k], expected)


def test_angle_decreases_with_energy():
    # Same p_T spectrum at every (E, x_L): a flat kernel. The mean angle must
    # then fall monotonically with secondary energy ~ <p_T>/E.
    axes = _axes(np.logspace(2, 4, 5))
    kernel = np.ones((len(axes["proj_energies"]), 30, 30))
    e_sec_edges = np.logspace(0, 4, 20)
    curve = mean_angle_vs_energy(kernel, axes, e_sec_edges)
    th = curve["theta_mean"]
    good = np.isfinite(th)
    # monotonic non-increasing over populated bins
    assert np.all(np.diff(th[good]) <= 1e-9)
    # scaling: theta * E_sec ~ const (within a factor over the populated range)
    prod = th[good] * curve["e_sec"][good]
    assert prod.max() / prod.min() < 3.0


def test_discrete_ordinate_row_normalized():
    axes = _axes([1000.0])
    kernel = np.ones((1, 30, 30))
    mu_edges = np.cos(np.radians(np.linspace(40.0, 0.0, 21)))
    row = discrete_ordinate_row(kernel, axes, 0, e_sec_target=10.0, mu_edges=mu_edges)
    assert np.isclose(row.sum(), 1.0)
    assert np.all(row >= 0)


def test_crossover_energy_interpolation():
    e = np.array([1.0, 10.0, 100.0, 1000.0])
    theta = np.array([20.0, 5.0, 1.0, 0.1])  # decreasing
    x = crossover_energy(e, theta, threshold_deg=3.0)
    # 3 deg is between E=10 (5 deg) and E=100 (1 deg)
    assert 10.0 < x < 100.0


def test_angular_kernel_moments_use_theta_axis():
    # A kernel concentrated in one theta bin must give <theta> = that bin center.
    axes = _angular_axes([100.0], theta_max=40.0, ntheta=40)
    nxl = len(axes["xl_centers"])
    kernel = np.zeros((1, nxl, 40))
    kbin = 7
    kernel[0, :, kbin] = 1.0
    mom = angle_moments(kernel, axes)  # dispatches on is_angular
    expected = np.radians(axes["theta_centers"][kbin])
    assert np.allclose(mom["theta_mean"][np.isfinite(mom["theta_mean"])], expected)


def test_angular_row_normalized_and_smooth():
    axes = _angular_axes([100.0])
    nxl = len(axes["xl_centers"])
    rng = np.random.default_rng(1)
    kernel = rng.random((1, nxl, 40))
    th, row = angular_row(kernel, axes, 0, e_sec_target=10.0)
    assert th.shape == row.shape
    dth = np.diff(axes["theta_edges"])
    assert np.isclose(np.sum(row * dth), 1.0)  # normalized density
    assert np.all(row >= 0)


def test_angular_row_requires_angular_kernel():
    axes = _axes([100.0])  # p_T-type axes (no is_angular)
    kernel = np.ones((1, 30, 30))
    try:
        angular_row(kernel, axes, 0, 10.0)
        assert False, "should have raised"
    except ValueError:
        pass


def test_event_moments_have_no_high_energy_floor():
    # The whole point of the gridless moments: <theta> ~ <p_T>/E_sec must hold
    # up to high energy with NO plateau. (A histogram-then-moment kernel would
    # floor <theta> at the angular bin scale, making <theta>*E_sec blow up.)
    grid = KernelGrid(
        xl_edges=np.logspace(-3, 0, 31),
        pt_edges=np.linspace(0.0, 3.0, 31),
        proj_energies=np.logspace(2, 5, 5),  # 100 GeV .. 100 TeV
    )
    mom = build_moments(ToySource(), grid, M_PION, n_interactions=30000)
    pooled = pool_moments_by_energy(mom, np.logspace(1, 4.5, 12))  # 10 GeV..30 TeV
    th = pooled["theta_mean"]
    e = pooled["e_sec"]
    good = np.isfinite(th)
    prod = th[good] * e[good]  # ~ const (<p_T>) if no floor
    assert prod.max() / prod.min() < 1.8


def test_d_theta_decreases_with_energy():
    grid = KernelGrid(
        xl_edges=np.logspace(-3, 0, 31),
        pt_edges=np.linspace(0.0, 3.0, 31),
        proj_energies=np.logspace(2, 5, 5),
    )
    mom = build_moments(ToySource(), grid, M_PION, n_interactions=30000)
    pooled = pool_moments_by_energy(mom, np.logspace(1, 4.5, 10))
    d = pooled["d_theta"]
    good = np.isfinite(d)
    assert np.all(np.diff(d[good]) < 0)  # diffusion coefficient -> 0 at high E
    assert np.allclose(pooled["d_theta"][good], 0.5 * pooled["theta_sq"][good])


def test_pool_moments_weighting():
    # Two cells at the same E_sec with different yields: pooled mean is yield-wtd.
    mom = {
        "e_sec": np.array([[10.0, 10.0]]),
        "theta_mean": np.array([[0.1, 0.2]]),
        "theta_sq": np.array([[0.01, 0.04]]),
        "dndx": np.array([[3.0, 1.0]]),
    }
    pooled = pool_moments_by_energy(mom, np.array([5.0, 20.0]))
    # weighted mean = (0.1*3 + 0.2*1)/4 = 0.125
    assert np.isclose(pooled["theta_mean"][0], 0.125)


def test_moments_finite_and_positive():
    axes = _axes([500.0])
    rng = np.random.default_rng(0)
    kernel = rng.random((1, 30, 30))
    mom = angle_moments(kernel, axes)
    assert np.all(mom["theta_rms"][np.isfinite(mom["theta_rms"])] >= 0)
    assert np.all(mom["theta_mean"][np.isfinite(mom["theta_mean"])] >= 0)
