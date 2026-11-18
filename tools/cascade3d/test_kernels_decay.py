"""Unit tests of ``kernels_decay`` (work package 0.3).

Run with ``$PY -m pytest tools/cascade3d/test_kernels_decay.py -q``.  The test that
needs MCEq is skipped when MCEq cannot be imported.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kernels_decay as kd  # noqa: E402
from constants import M_K, M_MU, M_PI  # noqa: E402

L = kd.L_MAX


def small_grid(lo=0.01, hi=1.0e3, per_decade=10):
    n = int(round(np.log10(hi / lo) * per_decade))
    return kd.Grid(np.logspace(np.log10(lo), np.log10(hi), n + 1))


def parent_at(grid, t, m0=None):
    """One parent energy (kinetic ``t``) in bin 0, weight one."""
    T = np.full((grid.n, 1), t)
    return T, np.ones((grid.n, 1))


def test_legendre_recursion_matches_numpy():
    x = np.linspace(-1, 1, 11)
    p = kd.legendre_all(x, 20)
    for l in (0, 1, 5, 20):
        c = np.zeros(l + 1)
        c[l] = 1.0
        assert p[l] == pytest.approx(np.polynomial.legendre.legval(x, c), abs=1e-12)


def test_two_body_lab_angle_against_closed_formula():
    """pi -> mu nu: nu lab angle is cos(th) = (c* + beta) / (1 + beta c*), and the
    neutrino's energy is gamma E* (1 + beta c*).  Integrate the closed formula with a
    fine midpoint rule over c* and compare with the binned moments, summed over bins."""
    g = small_grid(0.01, 1.0e4, 20)
    t = 5.0
    gam = (t + M_PI) / M_PI
    beta = np.sqrt(1 - 1 / gam ** 2)
    T, w = parent_at(g, t)
    mom = kd.two_body_moments(g, M_PI, 0.0, M_MU, T, w, n_gl=96, l_max=8)
    k = 3                                   # all parent bins hold the same energy
    c = (np.arange(400000) + 0.5) / 400000 * 2 - 1
    x = (c + beta) / (1 + beta * c)
    e_nu = gam * (M_PI ** 2 - M_MU ** 2) / (2 * M_PI) * (1 + beta * c)
    inside = (e_nu >= g.e_bins[0]) & (e_nu < g.e_bins[-1])
    p = kd.legendre_all(x, 8)
    for l in range(9):
        ref = np.sum(p[l][inside]) / len(c)
        assert mom[:, k, l].sum() == pytest.approx(ref, abs=1e-5)
    # the bin of each c* also has to agree: compare the l = 1 moment per child bin
    ib = np.searchsorted(g.e_bins, e_nu[inside], side="right") - 1
    ref1 = np.bincount(ib, weights=x[inside], minlength=g.n) / len(c)
    assert mom[:, k, 1] == pytest.approx(ref1, abs=1e-5)


def test_isotropic_decay_at_rest_has_no_anisotropy():
    """A parent at (almost) rest: every moment l >= 1 vanishes, for the analytic
    two-body kernel and for the sampled three-body kernel."""
    g = small_grid(1e-3, 1.0, 10)
    t = 1e-9
    T, w = parent_at(g, t)
    mom = kd.two_body_moments(g, M_PI, 0.0, M_MU, T, w, n_gl=64, l_max=16)
    tot0 = mom[:, 0, 0].sum()
    assert tot0 == pytest.approx(1.0, abs=1e-6)
    assert np.abs(mom[:, 0, 1:].sum(axis=0)).max() < 2e-3        # wide-bin average
    rng = np.random.default_rng(3)
    n = 400000
    e, p = kd.sample_three_body_rest(rng, n, M_K, [M_PI, M_MU, 0.0])
    accs = kd.sampled_moments(g, M_K, [e[2]], [p[2]], [0.0], T, w, rng, l_max=16)
    s = accs[0].s.sum(axis=0)[:, 0, :]
    n_in = s[:, 0].sum()
    assert n_in > 0.9 * n
    for l in range(1, 17):
        assert abs(s[:, l].sum() / n_in) < 5.0 / np.sqrt(n_in)


def test_helicity_probabilities_sum_to_one():
    """The left and right muon moments add up to the unpolarised muon, for all l."""
    g = small_grid(0.05, 1.0e3, 10)
    T, w = kd.parent_energy_nodes(g, "centre", 1)
    for m0 in (M_PI, M_K):
        for h0 in (-1.0, 1.0):
            ml, mr = kd.two_body_moments(g, m0, M_MU, 0.0, T, w, n_gl=96, l_max=10,
                                         h0=h0)
            un = kd.two_body_moments(g, m0, M_MU, 0.0, T, w, n_gl=96, l_max=10)
            assert ml + mr == pytest.approx(un, abs=1e-12)
            for ur in (False, True):
                ml, mr = kd.two_body_moments(g, m0, M_MU, 0.0, T, w, n_gl=96,
                                             l_max=10, h0=h0, ur=ur)
                un = kd.two_body_moments(g, m0, M_MU, 0.0, T, w, n_gl=96, l_max=10,
                                         ur=ur)
                assert ml + mr == pytest.approx(un, abs=1e-12)


def test_muon_helicity_limits():
    """Rest-frame helicity is kept by the forward muon and flipped by the backward
    one (mu+: h0 = -1, forward muons are left-handed)."""
    g = small_grid(0.05, 1.0e4, 20)
    T, w = parent_at(g, 1000.0)
    ml, mr = kd.two_body_moments(g, M_PI, M_MU, 0.0, T, w, n_gl=128, l_max=0,
                                 h0=-1.0)
    top = np.nonzero(ml[:, 3, 0] + mr[:, 3, 0])[0].max()
    low = np.nonzero(ml[:, 3, 0] + mr[:, 3, 0])[0].min()
    assert ml[top - 1, 3, 0] > 8 * mr[top - 1, 3, 0]
    assert mr[low + 1, 3, 0] > 1.5 * ml[low + 1, 3, 0]
    ratio = ml[low + 1:top, 3, 0] / mr[low + 1:top, 3, 0]
    assert np.all(np.diff(ratio) > 0)             # left share grows with muon energy
    # the totals: <left> = 0.5918 for a flat x spectrum at high energy
    tot = ml[:, 3, 0].sum(), mr[:, 3, 0].sum()
    assert tot[0] == pytest.approx(0.5918, abs=2e-3)
    assert sum(tot) == pytest.approx(1.0, abs=1e-6)


def test_analytic_pion_numu_yield_is_flat_and_normalised():
    g = small_grid(0.05, 1.0e5, 10)
    T, w = kd.parent_energy_nodes(g, "centre", 1)
    mom = kd.two_body_moments(g, M_PI, 0.0, M_MU, T, w, n_gl=64, l_max=0)[:, :, 0]
    k = 50
    r = (M_MU / M_PI) ** 2
    e_p = g.e_grid[k] + M_PI
    lost = g.e_bins[0] / ((1 - r) * e_p)           # neutrinos below the grid
    assert mom[:, k].sum() == pytest.approx(1.0 - lost, abs=1e-9)
    y = mom[:, k]
    # flat in E up to (1 - r) E_p: uniform density = 1 / ((1 - r) E_p) per GeV
    full = np.nonzero((g.e_bins[1:] < (1 - r) * e_p * 0.999))[0]
    dens = y[full] / g.e_widths[full]
    assert dens == pytest.approx(1.0 / ((1 - r) * e_p), rel=2e-3)


def test_three_body_agrees_with_decays_py():
    """Kmu3, K+ -> pi0 mu+ numu: neutrino energy spectrum and mean cos(theta) of the
    vectorised sampler against ``decays.decay`` (``_finish``), at one parent energy."""
    import decays as dk
    rng = np.random.default_rng(11)
    e_k = 4.0
    p_k = np.sqrt(e_k ** 2 - M_K ** 2)
    p4 = np.array([e_k, 0.0, 0.0, p_k])
    n_ref = 30000
    e_ref, c_ref = [], []
    for _ in range(n_ref):
        res = dk._finish(rng, p4, M_K, [111, -13, 14])
        q = [r[1] for r in res if r[0] == 14][0]
        e_ref.append(q[0])
        c_ref.append(q[3] / q[0])
    e_ref, c_ref = np.array(e_ref), np.array(c_ref)
    # sampler, boosted by hand to the same parent
    n = 400000
    e, p = kd.sample_three_body_rest(rng, n, M_K, [kd.MASS[111], M_MU, 0.0], "mc3d")
    gam = e_k / M_K
    beta = p_k / e_k
    e_l = gam * (e[2] + beta * p[2][:, 2])
    pz = gam * (p[2][:, 2] + beta * e[2])
    c_l = pz / e_l
    edges = np.linspace(0.0, e_l.max() * 1.0001, 9)
    h1 = np.histogram(e_ref, edges)[0] / n_ref
    h2 = np.histogram(e_l, edges)[0] / n
    err = np.sqrt(h1 * (1 - h1) / n_ref + h2 * (1 - h2) / n)
    assert np.all(np.abs(h1 - h2) < 4.5 * err + 1e-4), (h1, h2)
    tol = 5 * c_ref.std() / np.sqrt(n_ref)
    assert c_ref.mean() == pytest.approx(c_l.mean(), abs=tol)
    # the exactly flat Dalitz plot differs measurably (decays.py weights m12 once more)
    ef, pf = kd.sample_three_body_rest(rng, n, M_K, [kd.MASS[111], M_MU, 0.0], "flat")
    e_f = gam * (ef[2] + beta * pf[2][:, 2])
    hf = np.histogram(e_f, edges)[0] / n
    assert np.abs(hf - h2).max() > 0.004


def test_michel_matches_bend_michel_decay():
    """Mean lab neutrino energy and cos about the muon direction of our rest-frame
    sampler against the vectorised ``bend.michel_decay`` for both charges, left-handed,
    right-handed and unpolarised muons."""
    from bend import michel_decay
    rng = np.random.default_rng(5)
    n = 200000
    e_mu = 3.0
    gam = e_mu / M_MU
    beta = np.sqrt(1 - 1 / gam ** 2)
    z = np.tile([0.0, 0.0, 1.0], (n, 1))
    for pdg, charge in ((-13, +1), (13, -1)):
        for pol in (-1.0, 0.0, 1.0):
            rnd = rng.random((n, 6))
            pdg_nu, e_nu, d_nu = michel_decay(np.full(n, pdg), np.full(n, e_mu), z,
                                              np.full(n, pol), rnd,
                                              np.array([1.0, 0.0, 0.0]))
            for j, kind in enumerate(("numu", "nue")):
                e_s, c_s = kd.sample_michel_rest(rng, n, kind, charge, pol)
                e_lab = gam * e_s * (1 + beta * c_s)
                x_lab = (c_s + beta) / (1 + beta * c_s)
                for ref, mine in ((e_nu[:, j], e_lab), (d_nu[:, j, 2], x_lab)):
                    err = 5 * np.hypot(ref.std(), mine.std()) / np.sqrt(n)
                    assert abs(ref.mean() - mine.mean()) < err, (pdg, pol, kind)


def test_michel_polarisation_sign():
    """For a right-handed mu+ the nu_e is emitted opposite to the spin (backward in
    the rest frame): its lab spectrum is softer than for a left-handed mu+; the
    antinu_mu spectrum goes the other way (V-A)."""
    rng = np.random.default_rng(8)
    n = 200000
    gam = 30.0
    beta = np.sqrt(1 - 1 / gam ** 2)
    mean = {}
    for pol in (-1.0, 1.0):
        for kind in ("nue", "numu"):
            e_s, c_s = kd.sample_michel_rest(rng, n, kind, +1, pol)
            mean[(pol, kind)] = np.mean(gam * e_s * (1 + beta * c_s))
    assert mean[(1.0, "nue")] < mean[(-1.0, "nue")]
    assert mean[(1.0, "numu")] > mean[(-1.0, "numu")]


def test_cp_conjugate_names():
    assert kd.cp_conjugate_name("mu+_l") == "mu-_r"
    assert kd.cp_conjugate_name("K+") == "K-"
    assert kd.cp_conjugate_name("K_L0") == "K_L0"
    assert kd.cp_conjugate_name("mu-") == "mu+"
    assert kd.cp_conjugate_name("numu") == "antinumu"
    assert kd.cp_conjugate_name("antinue") == "nue"
    assert kd.cp_conjugate_name("pi0") == "pi0"


def test_branching_ratios_are_normalised():
    for p in ("pi+", "pi-", "K+", "K-", "K_L0", "K_S0"):
        ch, raw = kd.decay_channels(p)
        assert sum(b for b, _ in ch) == pytest.approx(1.0, abs=1e-12)
        assert 0.99 < raw < 1.0002


def test_save_load_roundtrip(tmp_path):
    g = small_grid(0.05, 100.0, 5)
    g, pairs = kd.build(g, parent_mode="lnE", n_sub=3, n_samples=2000, n_gl=32,
                        l_max=6, parents=["pi+", "K_S0", "mu+_l"], workers=1,
                        verbose=False)
    path = str(tmp_path / "decay.npz")
    kd.save(path, g, pairs, kd.make_meta(g), l_max=6)
    with np.load(path, allow_pickle=False) as d:          # no pickles in the file
        assert "mom" in d.files
    k = kd.load(path)
    assert ("pi+", "numu") in k.pairs and ("mu+_l", "nue") in k.pairs
    assert k.dense("pi+", "numu").shape == (7, g.n, g.n)
    assert k.dense("pi+", "numu") == pytest.approx(pairs[("pi+", "numu")].mom
                                                   .transpose(2, 0, 1))
    assert k.dense("mu+_l", "nue", "hd").any()            # sampled: has an error
    assert not k.dense("pi+", "numu", "hd").any()         # analytic: none


@pytest.fixture(scope="module")
def mceq():
    pytest.importorskip("MCEq")
    try:
        return kd.mceq_reference()
    except Exception as e:                                # pragma: no cover
        pytest.skip(f"MCEq not usable: {e}")


def test_grid_and_convention_against_mceq(mceq):
    """With MCEq's own approximations (parent at the bin centre, beta = 1, the
    ultra-relativistic muon polarisation, MCEq's rounded masses) pi+ -> numu, mu+_l,
    mu+_r reproduce MCEq's blocks to 2e-4: this fixes the conventions of the module
    (kinetic energy, width ratio, helicity assignment)."""
    g = kd.Grid(mceq["e_bins"], mceq["e_grid"])
    wr = g.e_widths[None, :] / g.e_widths[:, None]
    T, w = kd.parent_energy_nodes(g, "centre", 1)
    mpi, mmu = 0.13957, 0.10566
    ml, mr = kd.two_body_moments(g, mpi, mmu, 0.0, T, w, 128, 0, h0=-1.0, ur=True)
    nu = kd.two_body_moments(g, mpi, 0.0, mmu, T, w, 128, 0, ur=True)
    for name, arr in (("mu+_l", ml), ("mu+_r", mr), ("numu", nu)):
        ref = mceq["blocks"][("pi+", name)]
        mine = arr[:, :, 0] * wr * 0.99988
        sig = ref > 0.01 * ref.sum(axis=0)[None, :]
        rel = np.abs(mine - ref)[sig] / ref[sig]
        assert rel.max() < 2e-4, (name, rel.max())


def test_cp_images_equal_direct_computation():
    """pi- computed directly (rest-frame helicity +1) equals the CP image of pi+."""
    g = small_grid(0.05, 100.0, 5)
    kw = dict(parent_mode="centre", n_sub=1, n_gl=48, l_max=6, workers=1,
              verbose=False)
    _, direct = kd.build(g, parents=["pi-"], cp=False, **kw)
    _, both = kd.build(g, parents=["pi+"], cp=True, **kw)
    assert set(direct) == {("pi-", "mu-_r"), ("pi-", "mu-_l"), ("pi-", "antinumu")}
    for key, pr in direct.items():
        assert pr.mom == pytest.approx(both[key].mom, abs=1e-12)


def test_production_file_if_present():
    """The delivered ``decay.npz``: readable without pickles, on MCEq's PHYS grid,
    helicity yields add up, muon decay yields are one per neutrino flavour."""
    if not os.path.exists(kd.DEFAULT_PATH):
        pytest.skip("decay.npz not built")
    k = kd.load()
    assert k.n_bins == 126 and k.l_max == 64
    w = k.e_widths
    ib = 100                                     # a high parent bin, no energy lost
    def yield_of(par, ch):
        return (k.block(par, ch)[:, ib] * w).sum() / w[ib]
    both = yield_of("pi+", "mu+_l") + yield_of("pi+", "mu+_r")
    assert both == pytest.approx(1.0, abs=1e-6)
    assert yield_of("pi+", "numu") == pytest.approx(1.0, abs=1e-6)
    for par, nus in (("mu+_l", ("antinumu", "nue")), ("mu-_r", ("numu", "antinue"))):
        for nu in nus:
            assert yield_of(par, nu) == pytest.approx(1.0, abs=5e-3)
    assert yield_of("K+", "mu+_l") + yield_of("K+", "mu+_r") == pytest.approx(
        0.6356 / 1.0001, abs=1e-6)


def test_grid_agrees_with_kernels_mceq_file_if_present():
    p = os.path.join(kd.N3, "kernels", "mceq_PHYS.npz")
    if not (os.path.exists(p) and os.path.exists(kd.DEFAULT_PATH)):
        pytest.skip("mceq_PHYS.npz or decay.npz absent")
    k = kd.load()
    with np.load(p, allow_pickle=False) as d:
        assert np.allclose(d["e_bins"], k.e_bins, rtol=1e-12, atol=0)
        assert np.allclose(d["e_grid"], k.e_grid, rtol=1e-12, atol=0)
        names = [str(s) for s in d["species"]]
    for s in k.species:
        assert s in names
