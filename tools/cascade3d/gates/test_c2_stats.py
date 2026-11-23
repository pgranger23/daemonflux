"""Unit tests of the G2.3 statistics (`c2_stats.py`); synthetic tallies only."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c2_stats as st  # noqa: E402


def _truth():
    """Smooth flux and cutoff factor on (4 species, 10 bands, 10 rows, 12 azimuths)."""
    sp = np.array([1.0, 0.8, 2.0, 1.7])[:, None, None]
    band = np.logspace(0, -3, 10)[None, :, None]
    row = np.linspace(1.5, 1.0, 10)[None, None, :]
    phi_0 = sp * band * row
    az = st.az_centres(12)
    amp = np.linspace(0.3, 0.02, 10)[None, :, None, None]
    k = (0.4 + 0.06 * np.arange(10))[None, :, None, None] * (1 - amp * np.sin(az))
    k = k * np.linspace(0.9, 1.0, 10)[None, None, :, None]
    return phi_0, k * np.ones((4, 1, 1, 1))


def _runs(rng, nrun=10, nev=400.0):
    """Runs with Poisson noise; `n` is drawn independently of `s`."""
    phi_0, k = _truth()
    scale = nev / phi_0.max()
    s = rng.poisson(phi_0[..., None] * k * scale, (nrun,) + k.shape) / scale
    n = rng.poisson(phi_0[..., None] * scale * np.ones(12), (nrun,) + k.shape) / scale
    return s.astype(float), n.astype(float)


def test_solver_tallies_and_statistics():
    phi_0, k = _truth()
    s, n = st.solver_tallies(phi_0[..., None] * k, phi_0)
    assert np.allclose(st.stat_k(s, n), k)
    assert np.allclose(st.stat_kbar(s, n), k.mean(-1))
    h = st.stat_harm(s, n)
    amp = np.linspace(0.3, 0.02, 10)
    # flux = c (1 - amp sin az): cos component 0, sin component -amp / 2
    assert np.allclose(h[..., 0], 0.0, atol=1e-12)
    assert np.allclose(h[..., 1], -amp[:, None] / 2 * np.ones(10))
    assert st.stat_hv(s, n).shape == (10,)
    # the factor of a group of rows is weighted with the no-cutoff flux of the rows
    w, f = np.linspace(1.5, 1.0, 10), np.linspace(0.9, 1.0, 10)
    hv = (w[:2] @ f[:2] / w[:2].sum()) / (w[8:] @ f[8:] / w[8:].sum())
    assert np.allclose(st.stat_hv(s, n), hv)
    level = (0.4 + 0.06 * np.arange(10)) * (w @ f) / w.sum()
    assert np.allclose(st.stat_level(s, n), level)


def test_jackknife_matches_spread_for_a_mean():
    rng = np.random.default_rng(1)
    s, n = _runs(rng)
    val, err = st.jackknife(st.stat_flux, s, n)
    assert np.allclose(val, s.mean(0))
    assert np.allclose(err, s.std(0, ddof=1) / np.sqrt(s.shape[0]))


def test_exact_solver_counts_like_the_null():
    rng = np.random.default_rng(2)
    s_runs, n_runs = _runs(rng)
    ref = st.reference(s_runs, n_runs)
    phi_0, k = _truth()
    test = st.solver_tallies(phi_0[..., None] * k, phi_0)
    null = st.null_resample(("a", "b", "c"), ref, s_runs, n_runs, 300,
                            np.random.default_rng(3))
    for part in ("a", "b", "c"):
        items = int(ref[part][2].sum())
        c = st.count(part, st.STATS[part](*test), ref)
        g = st.null_gauss(part, ref, 300, np.random.default_rng(4))
        assert 0 < st.n5(g) <= items and 0 < st.n5(null[part]) <= items
        # the true values are a typical draw of either null model
        assert c >= min(np.percentile(g, 0.5), np.percentile(null[part], 0.5))
        assert c <= items


def test_wrong_solver_is_caught():
    rng = np.random.default_rng(5)
    s_runs, n_runs = _runs(rng, nev=4000.0)
    ref = st.reference(s_runs, n_runs)
    phi_0, k = _truth()
    good = st.solver_tallies(phi_0[..., None] * k, phi_0)
    bad = st.solver_tallies(phi_0[..., None] * k * 1.08, phi_0)
    assert st.count("a", st.stat_k(*bad), ref) < st.count("a", st.stat_k(*good), ref)
    az = st.az_centres(12)
    skew = st.solver_tallies(phi_0[..., None] * k * (1 + 0.1 * np.cos(az)), phi_0)
    nb = st.count("b", st.stat_harm(*good), ref)
    assert st.count("b", st.stat_harm(*skew), ref) < nb
    tilt = np.linspace(1.06, 1.0, 10)[None, None, :, None]
    tilted = st.solver_tallies(phi_0[..., None] * k * tilt, phi_0)
    nc = st.count("c", st.stat_hv(*good), ref)
    assert st.count("c", st.stat_hv(*tilted), ref) < nc


def test_n5():
    c = np.r_[np.full(4, 90), np.full(96, 100)]
    assert st.n5(c) == 100      # P(count < 100) = 0.04
    c = np.r_[np.full(6, 90), np.full(94, 100)]
    assert st.n5(c) == 90       # P(count < 100) = 0.06, P(count < 90) = 0


def test_null_split_identity_and_counts():
    rng = np.random.default_rng(11)
    s_runs, n_runs = _runs(rng, nrun=20)
    phi_0, k = _truth()
    ref = st.reference(s_runs, n_runs)
    # for the flux itself (a mean of the runs) the error of a draw is the spread
    # within the two halves: nu err^2 - d^2 = (SS_A + SS_B) / m, so that e^2 is the
    # pooled variance within the halves over m, with m - 2 degrees of freedom
    m, h = 20, 10
    idx = np.random.default_rng(3).permutation(m)[:h]
    rest = np.setdiff1d(np.arange(m), idx)
    val, err, ok = ref["d"]
    d = (s_runs[idx].mean(0) - s_runs[rest].mean(0)) / 2
    within = (((s_runs[idx] - s_runs[idx].mean(0)) ** 2).sum(0)
              + ((s_runs[rest] - s_runs[rest].mean(0)) ** 2).sum(0))
    assert np.allclose((m - 1) * err**2 - d**2, within / m, rtol=1e-9, atol=1e-12)

    parts = ("a", "b", "c", "d")
    sp = st.null_split(parts, ref, s_runs, n_runs, 300, np.random.default_rng(4))
    exact = st.solver_tallies(phi_0[..., None] * k, phi_0)
    for p in parts:
        c = st.count(p, st.STATS[p](*exact), ref)
        lo, hi = np.percentile(sp[p], [0.5, 99.5])
        assert lo - 2 <= c <= hi + 2, (p, c, lo, hi)
    # reproducible, and a wrong solver stays far below the threshold
    again = st.null_split(("a",), ref, s_runs, n_runs, 300, np.random.default_rng(4))
    assert np.array_equal(again["a"], sp["a"])
    wrong = st.solver_tallies(phi_0[..., None] * k * 1.08, phi_0)
    assert st.count("a", st.stat_k(*wrong), ref) < st.n5(sp["a"]) - 500


def test_null_split_needs_an_even_number_of_runs():
    rng = np.random.default_rng(12)
    s_runs, n_runs = _runs(rng, nrun=9)
    ref = st.reference(s_runs, n_runs)
    with pytest.raises(ValueError):
        st.null_split(("a",), ref, s_runs, n_runs, 5, rng)
