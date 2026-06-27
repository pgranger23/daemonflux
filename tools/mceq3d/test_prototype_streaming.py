"""Offline tests for the streaming feasibility prototype."""

import numpy as np

from prototype_streaming import solve_pn, diffusion_analytic, build_pn_slab


def test_p1_matches_diffusion_interior():
    r = solve_pn(n_x=400, n_l=2, c=0.99, length=40.0)
    ana = diffusion_analytic(r["x"], r["sigma_t"], r["sigma_s"], r["length"])
    m = (r["x"] > 8) & (r["x"] < 32)
    scale = np.trapezoid(r["phi0"][m], r["x"][m]) / np.trapezoid(ana[m], r["x"][m])
    rel = np.abs(r["phi0"][m] / (ana[m] * scale) - 1.0)
    assert np.median(rel) < 0.05  # diffusion limit recovered in the interior


def test_operator_is_sparse():
    # nnz per row is bounded (~5: diagonal + 4 streaming neighbours), so density
    # falls as 1/dof -> the streaming-coupled operator stays sparse.
    for n_l in (4, 16, 64):
        A, _, _ = build_pn_slab(300, n_l, 1.0, 0.9, np.zeros(300), 40.0)
        nnz_per_row = A.nnz / A.shape[0]
        assert nnz_per_row < 6.0


def test_cost_scaling_is_subquadratic():
    nls, times = [], []
    for n_l in (4, 8, 16, 32):
        r = solve_pn(n_x=300, n_l=n_l, c=0.9, length=40.0)
        nls.append(n_l)
        times.append(r["solve_time"])
    p_exp = np.polyfit(np.log(nls), np.log(np.maximum(times, 1e-6)), 1)[0]
    assert p_exp < 1.8  # not the >=2 of a dense/MC-scale blow-up


def test_density_decreases_with_size():
    A4, _, _ = build_pn_slab(300, 4, 1.0, 0.9, np.zeros(300), 40.0)
    A32, _, _ = build_pn_slab(300, 32, 1.0, 0.9, np.zeros(300), 40.0)
    d4 = A4.nnz / A4.shape[0] ** 2
    d32 = A32.nnz / A32.shape[0] ** 2
    assert d32 < d4  # sparser as it grows
