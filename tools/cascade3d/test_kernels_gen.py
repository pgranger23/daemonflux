"""Unit tests of ``kernels_gen`` (WP 0.2).

* moment accumulation against an analytic angular distribution (von
  Mises-Fisher, ``f(mu) ~ exp(kappa mu)``, moments
  ``a_l = I_{l+1/2}(kappa) / I_{1/2}(kappa)``);
* node selection and rescaling against ``tools/mc3d/pool.py`` on the same
  events (synthetic and a real pool file, read-only);
* delta-M limits and exactness.

Run: ``$PY -m pytest tools/cascade3d/test_kernels_gen.py -q``.
"""

import os
import shutil

import numpy as np
import pytest
from scipy import special

import kernels_gen as K
import pool as P

SCRATCH = os.path.join(K.N3, "c0", "wp02_scratch", "pytest")


def vmf_moments(kappa, l_max):
    """Legendre moments of f(mu) ~ exp(kappa mu) on [-1, 1]."""
    ell = np.arange(l_max + 1)
    return special.ive(ell + 0.5, kappa) / special.ive(0.5, kappa)


def vmf_sample(rng, kappa, n):
    u = rng.random(n)
    return 1.0 + np.log(u + (1.0 - u) * np.exp(-2.0 * kappa)) / kappa


def _events_from_mu(mu, p, pdg_child=211, rng=None):
    """One particle per event, momentum p along polar cosine mu."""
    rng = rng or np.random.default_rng(0)
    phi = rng.random(mu.size) * 2 * np.pi
    s = np.sqrt(np.maximum(1 - mu * mu, 0.0))
    px, py, pz = p * s * np.cos(phi), p * s * np.sin(phi), p * mu
    off = np.arange(mu.size + 1)
    return np.full(mu.size, pdg_child), px, py, pz, off


# ---------------------------------------------------------------------------
def test_legendre_table_matches_scipy():
    x = np.linspace(-1, 1, 201)
    P_ = K.legendre_table(x, 64)
    for ell in (0, 1, 2, 7, 32, 64):
        np.testing.assert_allclose(P_[:, ell], special.eval_legendre(ell, x),
                                   atol=1e-12)


@pytest.mark.parametrize("kappa", [3.0, 40.0])
def test_moments_of_analytic_distribution(kappa):
    rng = np.random.default_rng(1)
    n = 200000
    mu = vmf_sample(rng, kappa, n)
    pid, px, py, pz, off = _events_from_mu(mu, 5.0, rng=rng)
    e_bins = np.array([0.01, 1.0, 100.0, 1e4])
    e0 = 100.0
    acc = K.NodeAccumulator(2212, 0, e_bins, None, targets=[(0, 1.0, e0)],
                            e0=e0)
    half = n // 2
    acc.add(pid[:half], px[:half], py[:half], pz[:half], off[:half + 1])
    acc.add(pid[half:], px[half:], py[half:], pz[half:],
            off[half:] - off[half])
    a = acc.arrays()
    c = K.CHILDREN.index(211)
    S = a["S"][:, 0, c, 1]                    # child bin 1 (1..100 GeV)
    Q = a["Q"][0, c, 1]
    nev = a["n_ev"].sum()
    assert a["S"][:, 0, c, 0].sum() == 0 and int(S[:, 0].sum()) == n
    mean = S.sum(0) / nev
    err = np.sqrt((Q / nev - mean ** 2) / nev)
    want = vmf_moments(kappa, 64)
    pull = (mean - want) / np.maximum(err, 1e-12)
    assert np.all(np.abs(pull[1:]) < 5.0), pull
    assert abs(mean[0] - 1.0) < 1e-12
    # halves are disjoint and complete
    assert a["n_ev"].tolist() == [n // 2, n // 2]
    np.testing.assert_allclose(S.sum(0), a["S"][:, 0, c, 1].sum(0))
    # one particle per event: Q is the plain sum of P_l^2
    P_ = K.legendre_table(mu, 64)
    np.testing.assert_allclose(Q, (P_ ** 2).sum(0), rtol=1e-9)
    # theta histogram against the analytic CDF
    h = a["H"][0, c, 1]
    th = K.THETA_EDGES
    cdf = (np.exp(kappa * np.cos(th[:-1])) - np.exp(kappa * np.cos(th[1:])))
    cdf /= np.exp(kappa) - np.exp(-kappa)
    big = cdf * n > 200
    rel = h[big] / (cdf[big] * n) - 1
    assert np.all(np.abs(rel) < 5 / np.sqrt(cdf[big] * n)), rel


def test_multiplicity_and_out_of_grid_bookkeeping():
    rng = np.random.default_rng(2)
    mu = vmf_sample(rng, 10.0, 1000)
    pid, px, py, pz, off = _events_from_mu(mu, 5.0, rng=rng)
    # child bins that miss the particle (E_kin ~ 4.86 GeV): all "high"
    acc = K.NodeAccumulator(2212, 0, np.array([0.1, 1.0]), None,
                            targets=[(0, 1.0, 100.0)], e0=100.0)
    acc.add(pid, px, py, pz, off)
    c = K.CHILDREN.index(211)
    assert acc.n_out_high[0, c] == 1000 and acc.S.sum() == 0


# ---------------------------------------------------------------------------
@pytest.mark.parametrize("e", [2.0, 2.1, 2.5, 7.3, 79.0, 85.0, 333.0, 9000.0])
def test_node_weights_match_bin_choice(e):
    pl = P.EventPool(None)
    rng = np.random.default_rng(3)
    n = 40000
    picks = np.array([pl.bin_choice(rng, e) for _ in range(n)])
    w = dict(K.node_weights(e))
    assert abs(sum(w.values()) - 1.0) < 1e-12
    for node in np.unique(picks):
        f = (picks == node).mean()
        p = w.get(int(node), 0.0)
        assert abs(f - p) < 5 * np.sqrt(p * (1 - p) / n) + 1e-9
    assert set(np.unique(picks).tolist()) <= set(w)


def test_floor():
    assert K.node_weights(1.999) == []
    pl = P.EventPool(None)
    assert pl.draw(np.random.default_rng(0), 2212, 1.999) == []
    st, _n, _w = K.projectile_status(2212, np.array([0.5, 1.2, 3.0]),
                                     set(range(59)))
    assert st.tolist() == [K.FLOOR, K.SERVED, K.SERVED]


def _inject(pl, pdg, nodes, d):
    for i in nodes:
        pl._pools[(int(pdg), int(i))] = d


def _single_event_pool(pid, px, py, pz, e0, pdg):
    return {"pid": np.asarray(pid, np.int32),
            "px": np.asarray(px, np.float32),
            "py": np.asarray(py, np.float32),
            "pz": np.asarray(pz, np.float32),
            "off": np.array([0, len(pid)]), "e0": np.array([e0]),
            "pdg": np.array([pdg]), "n": np.array([1]),
            "model": np.array(["test"])}


def _compare(pdg, e0, e, ev, rng_seed=0):
    pl = P.EventPool(None)
    d = _single_event_pool(*ev, e0=e0, pdg=pdg)
    _inject(pl, pdg, range(P.GRID.n), d)
    got = pl.draw(np.random.default_rng(rng_seed), pdg, e)
    rs = K.rescale(pdg, e0, e, d["pid"], d["px"], d["py"], d["pz"],
                   d["off"])
    assert len(got) == len(d["pid"])
    for q, (c, ec, pv) in enumerate(got):
        assert c == int(d["pid"][q])
        np.testing.assert_allclose(ec, rs["e"][q], rtol=1e-12)
        np.testing.assert_allclose(pv, [rs["px"][q], rs["py"][q],
                                        rs["pz"][q]], rtol=1e-12,
                                   atol=1e-14)
    return rs


def test_rescale_matches_pool_draw_synthetic():
    ev = ([211, -211, 2212, 2112, 321], [0.3, -0.2, 0.1, 0.5, 0.0],
          [0.1, 0.4, -0.3, 0.2, 0.25], [20.0, 3.0, 40.0, 0.2, 7.0])
    rs = _compare(2212, 100.0, 63.0, ev)
    assert rs["scale"][0] == 1.0
    # the energy guard fires
    ev2 = ([211, 211], [0.1, 0.2], [0.0, 0.1], [90.0, 80.0])
    rs2 = _compare(211, 100.0, 120.0, ev2)
    assert rs2["scale"][0] < 1.0


def test_rescale_matches_pool_draw_real_events():
    fp = K.pool_path(2212, 23)
    if not os.path.exists(fp):
        pytest.skip("MC pools not present")
    d = K.read_pool(2212, 23)
    for k_e in (178.8, 200.0, 170.0):
        for i in range(25):
            a, b = int(d["off"][i]), int(d["off"][i + 1])
            if b <= a:
                continue
            ev = (d["pid"][a:b], d["px"][a:b], d["py"][a:b], d["pz"][a:b])
            _compare(2212, float(d["e0"][0]), k_e, ev)


def test_accumulator_matches_pool_draw_on_real_pool():
    """The accumulator's per-bin counts at one bin equal the counts of the
    particles that ``pool.draw`` returns for every event of the pool."""
    fp = K.pool_path(211, 10)
    if not os.path.exists(fp):
        pytest.skip("MC pools not present")
    d = K.read_pool(211, 10)
    e_bins = np.geomspace(0.025118864315095794, 1e11, 127)
    e_grid = np.sqrt(e_bins[1:] * e_bins[:-1])
    acc = K.NodeAccumulator(211, 10, e_bins, e_grid)
    acc.add(d["pid"], d["px"], d["py"], d["pz"], d["off"])
    kk = int(np.argmax(acc.w))
    e = acc.etot[kk]
    cnt = np.zeros((len(K.CHILDREN), len(e_grid)))
    cos1 = np.zeros_like(cnt)
    pl = P.EventPool(None)
    for i in range(int(d["n"][0])):
        a, b = int(d["off"][i]), int(d["off"][i + 1])
        one = _single_event_pool(d["pid"][a:b], d["px"][a:b], d["py"][a:b],
                                 d["pz"][a:b], float(d["e0"][0]), 211)
        _inject(pl, 211, range(P.GRID.n), one)
        for c, ec, pv in pl.draw(np.random.default_rng(i), 211, e):
            m = np.searchsorted(e_bins, ec - K.MASS[c], side="right") - 1
            if 0 <= m < len(e_grid):
                ci = K.CHILDREN.index(c)
                cnt[ci, m] += 1
                cos1[ci, m] += pv[2] / np.linalg.norm(pv)
    S = acc.S.sum(0)[kk].reshape(len(K.CHILDREN), len(e_grid), -1)
    np.testing.assert_allclose(S[..., 0], cnt, atol=1e-9)
    np.testing.assert_allclose(S[..., 1], cos1, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
def test_delta_m_straight_ahead_limit():
    a = vmf_moments(20.0, 64)
    c = 3.7 * a
    g, a_star = K.delta_m(c, L=32, g=1.0)
    assert g == 1.0
    assert a_star[0] == 1.0 and np.all(a_star[1:] == 0.0)
    np.testing.assert_array_equal(K.delta_m_moments(g, a_star, 64),
                                  np.ones(65))


def test_delta_m_pure_series_and_exactness():
    a = vmf_moments(20.0, 64)
    g0, s0 = K.delta_m(a, L=32, g=0.0)
    np.testing.assert_allclose(s0, a[:33], rtol=0, atol=1e-15)
    g, s = K.delta_m(2.0 * a, L=32)
    assert 0.0 < g < 1.0
    np.testing.assert_allclose(K.delta_m_moments(g, s)[:33], a[:33],
                               rtol=0, atol=1e-13)
    # classical choice g = a_{L+1}
    g1, _ = K.delta_m(a, L=32, n_tail=1)
    assert g1 == pytest.approx(a[33])


def test_delta_m_robust_to_noise_and_empty_entries():
    rng = np.random.default_rng(4)
    a = vmf_moments(5.0, 64)
    noisy = a[None, :] + rng.normal(0, 0.2, (1000, 65))
    noisy[:, 0] = 1.0
    noisy[0] = 0.0                       # empty entry (c_0 = 0)
    noisy[1, 33:] = 1.3                  # moments above 1: g clipped to 1
    noisy[2, 33:] = -0.5                 # negative tail: g clipped to 0
    g, s = K.delta_m(noisy, L=32)
    assert np.all(np.isfinite(g)) and np.all(np.isfinite(s))
    assert np.all((g >= 0) & (g <= 1))
    assert g[0] == 1.0 and np.all(s[0, 1:] == 0)
    assert g[1] == 1.0 and g[2] == 0.0
    ok = g < 1
    np.testing.assert_allclose(K.delta_m_moments(g[ok], s[ok])[:, 1:33],
                               noisy[ok, 1:33], atol=1e-12)
    n_ent = np.full(1000, 50)
    n_ent[5] = 3
    g2, s2 = K.delta_m(noisy, L=32, n_entries=n_ent, min_entries=10)
    assert g2[5] == 1.0 and np.all(s2[5, 1:] == 0)


def test_series_normalisation():
    a = vmf_moments(8.0, 64)
    g, s = K.delta_m(a, L=32, g=0.0)
    mu = np.linspace(-1, 1, 20001)
    f = K.series(mu, s)
    assert np.trapezoid(f, mu) == pytest.approx(1.0, rel=1e-6)
    want = 8.0 * np.exp(8.0 * mu) / (2 * np.sinh(8.0))
    np.testing.assert_allclose(f, want, atol=1e-6)


# ---------------------------------------------------------------------------
def test_merge_two_nodes_weighted_mix():
    """A bin between nodes 22 and 23 is the t-weighted mix of both."""
    if not (os.path.exists(K.pool_path(2212, 22))
            and os.path.exists(K.pool_path(2212, 23))):
        pytest.skip("MC pools not present")
    kd = os.path.join(SCRATCH, "merge")
    if os.path.isdir(kd):
        shutil.rmtree(kd)
    os.makedirs(kd)
    try:
        K.run("t", "pool", nproc=2, nodes=[(2212, 22), (2212, 23)],
              kernel_dir=kd)
        fp, = K.merge("t", populations=("PHYS",), kernel_dir=kd)
        m = K.load("t", "PHYS", kernel_dir=kd)
        assert m["meta"]["population"] == "PHYS"
        jp = K.PROJECTILES.index(2212)
        k = int(np.nonzero(m["status"][jp] == K.SERVED)[0][0])
        kx = k - int(m["k_index"][0])
        nodes, w = m["node_index"][jp, k], m["node_weight"][jp, k]
        assert sorted(nodes.tolist()) == [22, 23]
        want = 0.0
        for n_, w_ in zip(nodes, w):
            d = np.load(K.node_file("t", 2212, n_, kd))
            kk = int(np.nonzero(d["k"] == k)[0][0])
            S = d["S"][:, kk].sum(0) / d["n_ev"].sum()
            pad = np.zeros(m["c_l"].shape[2:][1:])
            pad[:S.shape[1]] = S[K.CHILDREN.index(211)]
            want = want + w_ * pad
        np.testing.assert_allclose(m["c_l"][jp, K.CHILDREN.index(211), kx],
                                   want, rtol=1e-12, atol=1e-15)
        e_k, e_m = m["e_grid"][m["k_index"]], m["e_grid"][m["m_index"]]
        np.testing.assert_allclose(
            m["c0_mceq"][jp, 4, kx],
            m["c_l"][jp, 4, kx, :, 0] * e_k[kx] / e_m, rtol=1e-12)
        with np.load(fp, allow_pickle=False) as d:
            assert "c_l" in d.files
    finally:
        shutil.rmtree(kd, ignore_errors=True)


def test_nproc_cap():
    with pytest.raises(SystemExit):
        K.run("never", "pool", nproc=25)


def test_moment_errors_match_replica_spread():
    """a_l errors from (var, cov) agree with the spread over replicas when
    the multiplicity fluctuates from event to event."""
    rng = np.random.default_rng(5)
    e_bins = np.array([0.01, 1.0, 100.0])
    c = K.CHILDREN.index(211)
    a1, err1 = [], []
    for _rep in range(60):
        nper = rng.poisson(3.0, 2000)
        n = int(nper.sum())
        mu = vmf_sample(rng, 4.0, n)
        pid, px, py, pz, _ = _events_from_mu(mu, 5.0, rng=rng)
        off = np.concatenate([[0], np.cumsum(nper)])
        acc = K.NodeAccumulator(2212, 0, e_bins, None,
                                targets=[(0, 1.0, 100.0)], e0=100.0,
                                l_max=4)
        acc.add(pid, px, py, pz, off)
        a = acc.arrays()
        assert np.allclose(a["Q"][..., 0], a["R"][..., 0])
        nev = a["n_ev"].sum()
        mean = a["S"].sum(0)[0, c, 1] / nev
        var = (a["Q"][0, c, 1] / nev - mean ** 2) / nev
        cov = (a["R"][0, c, 1] / nev - mean * mean[0]) / nev
        e = K.moment_errors(mean, var, cov)
        a1.append(mean[1] / mean[0])
        err1.append(e[1])
    spread = np.std(a1, ddof=1)
    assert np.mean(err1) == pytest.approx(spread, rel=0.25)
