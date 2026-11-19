"""Tests of ``kernelset_mc`` (WP 1.3): the Monte Carlo-matched kernel sets.

Every Monte Carlo rule is checked against the Monte Carlo's own functions
(``tools/mc3d``), not against copied numbers: species, floor, thresholds,
interaction lengths, decay rates, muon stopping power, primary flux and range; the
l = 0 yields against the generator and decay files they came from; the angular
laws against their moments; the Lambda kernels against ``decays.decay`` sampled.
The energy and baryon-number sum rules per projectile bin are printed
(``pytest -s``).

Run: ``taskset -c 8-11 $PY -m pytest tools/cascade3d/test_kernelset_mc.py -q``.
Uses the stored sets under ``$N3/kernels/mc`` (skipped if missing) and builds the
pool2000 and one half set afresh.
"""

import os

import numpy as np
import pytest

import kernels_decay as kd
import kernels_gen as kg
import kernelset_mc as M
from kernelset import KIND_DEC, KIND_INT, ROLE_LOSS, ROLE_SINK, KernelSet

import constants as mcc
import decays as mcdec
import interactions as mci
import pool as mcpool
import primaries as mcprim
import shower as mcshower
import sky as mcsky


@pytest.fixture(scope="module")
def ks_pool():
    return M.build("pool2000", loss=True)


@pytest.fixture(scope="module")
def ks_half():
    return M.build("fresh1e5_h1", loss=False)


@pytest.fixture(scope="module")
def gen_pool():
    return kg.load("pool2000")


def _stored(gen, loss=True):
    p = M.set_path(gen, 32, loss)
    if not os.path.exists(p):
        pytest.skip(f"{p} not built")
    return KernelSet.load(p)


# ------------------------------------------------------------------ structure
@pytest.mark.parametrize("gen", list(M.GEN_SETS))
@pytest.mark.parametrize("loss", [True, False])
def test_stored_sets_validate_and_match_a_fresh_build(gen, loss, ks_pool):
    ks = _stored(gen, loss)
    assert ks.validate()
    assert ks.meta["gen"] == gen and ks.meta["loss"] == loss
    assert ks.L == 32 and ks.nA == ks.nE
    if gen == "pool2000" and loss:
        for k in ("Y", "g", "a", "inv_lambda", "inv_dec", "dedx", "phi_top"):
            np.testing.assert_array_equal(ks.d[k], ks_pool.d[k])


def test_species_are_the_tracked_ones_and_nothing_else(ks_pool):
    d = ks_pool.d
    hadrons_and_muons = set(int(p) for p in d["pdg"][d["role"] != ROLE_SINK])
    assert hadrons_and_muons == set(mcc.TRACKED)
    assert hadrons_and_muons == set(mcpool.KEEP)
    sinks = set(int(p) for p in d["pdg"][d["role"] == ROLE_SINK])
    assert sinks == {12, -12, 14, -14}        # no nu_tau source in the MC
    # every species that has a threshold in the chain scheme is tracked, and
    # nothing else (pi0, gamma, e, hyperons, charm are dropped: no species)
    thr = mcshower.thresholds(0.1, "chain")
    assert set(thr) == set(mcc.TRACKED)
    for dropped in (111, 22, 11, -11, 3112, 3222, 411, 421, 4122):
        assert dropped not in d["pdg"]
    # muons: the six MCEq helicity states, loss species
    mu = d["role"] == ROLE_LOSS
    assert sorted(d["species"][mu]) == sorted(
        ["mu+_l", "mu+_r", "mu+", "mu-_l", "mu-_r", "mu-"])
    assert np.all(d["active"])                # nothing folded


def test_generator_muons_are_helicity_zero(ks_pool):
    for p in ("p+", "pi+", "K+"):
        for c in ("mu+", "mu-"):
            ks_pool.channel("int", p, c)
        for c in ("mu+_l", "mu+_r", "mu-_l", "mu-_r"):
            with pytest.raises(KeyError):
                ks_pool.channel("int", p, c)


# ------------------------------------------------------- transport parameters
def test_interaction_lengths_are_the_monte_carlos(ks_pool):
    d = ks_pool.d
    be = mci.ChromoBackend(cache_dir=None, xs="mceq", readonly=True)
    tab = mci.MCEqYieldBackend(mci.DEFAULT_TABLES)
    for s, pdg in enumerate(d["pdg"]):
        for k in (0, 10, 14, 17, 18, 25, 40, 51):
            et = d["e_grid"][k] + mcc.MASS[int(pdg)]
            lam = be.lambda_int(int(pdg), et) if be.has_interaction(int(pdg)) \
                else np.inf
            want = 0.0 if not np.isfinite(lam) else 1.0 / lam
            assert d["inv_lambda"][s, k] == pytest.approx(want, rel=1e-12, abs=0)
            if be.has_interaction(int(pdg)):
                # same as the table interpolation with MCEq's target mass
                ref = tab.sigma_inel(int(pdg), et) / mci.M_TARGET_G
                assert d["inv_lambda"][s, k] == pytest.approx(ref, rel=1e-12)
    # zero (to 1e-15) below the table's threshold: nucleons 1.41, pi 0.71,
    # K 0.89, Lambda 1.78 GeV kinetic
    E = d["e_grid"]
    for name, t0 in (("p+", 1.41), ("n0", 1.41), ("pi+", 0.708), ("K+", 0.89),
                     ("K_L0", 0.89), ("Lambda0", 1.78)):
        s = ks_pool.index(name)
        assert np.all(d["inv_lambda"][s, E < 0.99 * t0] < 1e-15)
        assert np.all(d["inv_lambda"][s, E > 1.01 * t0] > 1e-3)


def test_decay_rates_are_the_monte_carlos(ks_pool):
    d = ks_pool.d
    for s, pdg in enumerate(d["pdg"]):
        ctau = mcc.CTAU_CM.get(int(pdg))
        m = mcc.MASS[int(pdg)]
        for k in (0, 7, 20, 45):
            e = d["e_grid"][k] + m
            if ctau is None or d["role"][s] == ROLE_SINK:
                assert d["inv_dec"][s, k] == 0.0
                continue
            p = np.sqrt(e * e - m * m)
            # transport_hadron: s_dec = -ln(u) (p/m) ctau
            assert d["inv_dec"][s, k] == pytest.approx(m / (p * ctau), rel=1e-12)
    # neutrons decay into nothing in the MC: rate kept, no decay channel
    n = ks_pool.index("n0")
    assert d["inv_dec"][n].min() > 0
    assert not np.any((d["ch_parent"] == n) & (d["ch_kind"] == KIND_DEC))
    assert mcdec.decay(np.random.default_rng(1), 2112,
                       np.array([2.0, 0.0, 0.0, np.sqrt(4 - mcc.M_N ** 2)])) == []


def test_dedx_is_shower_dedx_air_for_muons_only(ks_pool, ks_half):
    d = ks_pool.d
    for s in range(ks_pool.nS):
        if d["role"][s] == ROLE_LOSS:
            for k in (0, 5, 20, 51):
                want = mcshower.dedx_air(d["e_grid"][k] + mcc.M_MU)
                assert d["dedx"][s, k] == pytest.approx(want, rel=1e-14)
        else:
            assert np.all(d["dedx"][s] == 0)
    assert np.all(ks_half.d["dedx"] == 0)


def test_floor_zero_yield_but_rate_kept(ks_pool):
    """pool.EventPool.draw returns nothing below GRID.edges[0] (total energy):
    every interaction channel is zero there; the interaction rate is kept."""
    d = ks_pool.d
    floor = mcpool.GRID.edges[0]
    rng = np.random.default_rng(3)
    ep = mcpool.EventPool(None)
    n_abs = 0
    for s, pdg in enumerate(d["pdg"]):
        if int(pdg) not in mcpool.POOL_PROJECTILES:
            continue
        for k in range(ks_pool.nE):
            et = d["e_grid"][k] + mcc.MASS[int(pdg)]
            sel = (d["ch_parent"] == s) & (d["ch_kind"] == KIND_INT)
            if et < floor:
                assert ep.draw(rng, int(pdg), et) == []
                assert np.all(d["Y"][sel][:, :, k] == 0)
                if d["inv_lambda"][s, k] > 1e-6:
                    n_abs += 1
    # pions 0.71-1.86 GeV and kaons 0.89-1.51 GeV: absorbed with zero yield
    assert n_abs >= 8
    pi = ks_pool.index("pi+")
    k18 = int(np.argmin(np.abs(d["e_grid"] - 1.78)))
    assert d["inv_lambda"][pi, k18] > 0.005


def test_thresholds_from_shower_and_their_irrelevance(ks_pool):
    d = ks_pool.d
    E = d["e_grid"]
    thr = mcshower.thresholds(0.1, "chain")
    for s, pdg in enumerate(d["pdg"]):
        if d["role"][s] == ROLE_SINK:
            continue
        below = E + mcc.MASS[int(pdg)] < thr[int(pdg)]
        sel = d["ch_child"] == s
        assert np.all(d["Y"][sel][:, below, :] == 0)
        if below.any():
            # only nucleons are affected (T < 0.106 GeV): no interaction, no
            # decay product, so dropping them changes no flux
            assert int(pdg) in (2212, 2112)
            assert np.all(d["inv_lambda"][s, below] < 1e-15)
            par = (d["ch_parent"] == s)
            assert not np.any(d["Y"][par][:, :, below] != 0)
    # every other species' threshold lies below the grid
    eb0 = d["e_bins"][0]
    for pdg, e in thr.items():
        if abs(pdg) not in (2212, 2112) or pdg < 0:
            assert e - mcc.MASS[pdg] < eb0


# ----------------------------------------------------------------- yields
def test_l0_interaction_yields_equal_the_generator_file(ks_pool, gen_pool):
    G, d = gen_pool, ks_pool.d
    nE = ks_pool.nE
    ki, mi = G["k_index"], G["m_index"]
    for jp, ppdg in enumerate(G["projectiles"]):
        p = int(np.flatnonzero(d["pdg"] == ppdg)[0])
        served = G["status"][jp] == kg.SERVED
        for ic, cpdg in enumerate(G["children"]):
            c = [i for i in range(ks_pool.nS)
                 if d["pdg"][i] == cpdg and d["species"][i] in M.NAMES
                 and M.SPECIES[i][2] == 0][0]
            full = np.zeros((nE, nE))
            ok_k = ki < nE
            ok_m = mi < nE
            full[np.ix_(mi[ok_m], ki[ok_k])] = G["c0_mceq"][jp, ic][ok_k][:, ok_m].T
            full[:, ~served[:nE]] = 0.0
            thr = M.mc_thresholds_kinetic()[int(cpdg)]
            full[d["e_grid"] < thr] = 0.0
            try:
                j = ks_pool.channel("int", d["species"][p], d["species"][c])
                got = d["Y"][j]
            except KeyError:
                got = np.zeros((nE, nE))
            np.testing.assert_allclose(got, full, rtol=1e-12, atol=0)


def test_half_set_uses_the_half_sample(ks_half):
    G = kg.load("fresh1e5")
    d = ks_half.d
    jp = list(G["projectiles"]).index(211)
    ic = list(G["children"]).index(211)
    j = ks_half.channel("int", "pi+", "pi+")
    E = d["e_grid"]
    for kx in (5, 15, 30):
        k = int(G["k_index"][kx])
        for mx in (3, 10, k - 2):
            m = int(G["m_index"][mx])
            want = float(G["c_l_half"][1, jp, ic, kx, mx, 0]) * E[k] / E[m]
            assert d["Y"][j][m, k] == pytest.approx(want, rel=1e-12)


def test_decay_yields_equal_the_decay_files(ks_pool):
    d = ks_pool.d
    nE = ks_pool.nE
    for path in (M.DECAY_PATH, M.DECAY_EXTRA_PATH):
        dk = kd.load(path)
        for parent, child in dk.pairs:
            if child == "pi0":
                with pytest.raises(KeyError):
                    ks_pool.index(child)
                continue
            j = ks_pool.channel("dec", parent, child)
            want = dk.block(parent, child)[:nE, :nE].copy()
            c = ks_pool.index(child)
            thr = M.mc_thresholds_kinetic().get(int(d["pdg"][c]))
            if thr is not None and d["role"][c] != ROLE_SINK:
                want[d["e_grid"] < thr] = 0.0
            np.testing.assert_allclose(d["Y"][j], want, rtol=1e-12, atol=0)


def test_upscatter_entries_are_kept(ks_pool):
    """K+ -> mu+ at parent T = 0.089 GeV: most muons land above the parent bin."""
    d = ks_pool.d
    k = int(np.argmin(np.abs(d["e_grid"] - 0.089)))
    w = d["e_widths"]
    for child, lo in (("mu+_l", 0.5), ("mu+_r", 0.5)):
        Y = d["Y"][ks_pool.channel("dec", "K+", child)]
        n = Y[:, k] * w / w[k]
        assert n[k + 1:].sum() / n.sum() > lo
    Y = d["Y"][ks_pool.channel("dec", "K_S0", "pi+")]
    n = Y[:, k] * w / w[k]
    assert n[k + 1:].sum() / n.sum() > 0.5


def test_angular_law_reproduces_the_moments(ks_pool, gen_pool):
    d = ks_pool.d
    G = gen_pool
    L = ks_pool.L
    jp = list(G["projectiles"]).index(2212)
    ic = list(G["children"]).index(211)
    j = ks_pool.channel("int", "p+", "pi+")
    a_idx = int(np.flatnonzero(d["ang_channel"] == j)[0])
    for kx in (5, 13, 30):
        k = int(G["k_index"][kx])
        for mx in range(0, k):
            cl = G["c_l"][jp, ic, kx, mx]
            if cl[0] <= 0:
                continue
            a = cl / cl[0]
            g = d["g"][a_idx, mx, k]
            mom = kg.delta_m_moments(np.array(g), d["a"][a_idx, :, mx, k])
            if np.all(np.abs(d["a"][a_idx, :, mx, k]) < 1 - 1e-9):
                np.testing.assert_allclose(mom, a[:L + 1], atol=1e-10)
    # decays too: pi+ -> numu
    dk = kd.load(M.DECAY_PATH)
    mom = dk.dense("pi+", "numu")
    j = ks_pool.channel("dec", "pi+", "numu")
    a_idx = int(np.flatnonzero(d["ang_channel"] == j)[0])
    for k in (3, 15, 30):
        for m in range(0, k + 1):
            if mom[0, m, k] <= 0:
                continue
            got = kg.delta_m_moments(np.array(d["g"][a_idx, m, k]),
                                     d["a"][a_idx, :, m, k])
            np.testing.assert_allclose(got, mom[:L + 1, m, k] / mom[0, m, k],
                                       atol=1e-10)


def test_lambda_kernels_against_decays_decay_sampled():
    """decay_extra.npz against the MC's own decays.decay, sampled."""
    dx = kd.load(M.DECAY_EXTRA_PATH)
    rng = np.random.default_rng(11)
    E = dx.e_grid
    w = dx.e_widths
    m0 = mcc.MASS[3122]
    for k in (12, 25):
        e = E[k] + m0
        p = np.sqrt(e * e - m0 * m0)
        n_s = 20000
        out = {"p+": [], "pi-": [], "n0": []}
        pdg_name = {2212: "p+", -211: "pi-", 2112: "n0"}
        for _ in range(n_s):
            for pdg, q, _pol in mcdec.decay(rng, 3122, np.array([e, 0, 0, p])):
                if pdg in pdg_name:
                    out[pdg_name[pdg]].append((q[0] - mcc.MASS[pdg],
                                               q[3] / np.linalg.norm(q[1:])))
        for c, vals in out.items():
            vals = np.array(vals)
            vals = vals[vals[:, 0] >= dx.e_bins[0]]        # children on the grid
            mom = dx.dense("Lambda0", c)[:, :, k] * w[None, :] / w[k]
            n_k = mom[0].sum()
            assert n_k == pytest.approx(len(vals) / n_s, abs=4 * np.sqrt(
                n_k * (1 - n_k) / n_s) + 1e-3)
            t_mean = (mom[0] * E).sum() / n_k
            c_mean = mom[1].sum() / n_k
            # bin-centre energies: 2% on the mean kinetic energy
            assert t_mean == pytest.approx(vals[:, 0].mean(), rel=0.03)
            assert c_mean == pytest.approx(vals[:, 1].mean(), abs=0.01)


# ------------------------------------------------------------------ primaries
def test_primary_flux_and_range(ks_pool):
    d = ks_pool.d
    lo, hi = M.mc_primary_range(7)
    assert (lo, hi) == (float(mcsky.STRATA[0]), float(mcsky.STRATA[7]))
    assert hi == pytest.approx(10 ** 3.5)        # not 1000: 7 strata end at 3162
    eb = d["e_bins"]
    p, n = ks_pool.index("p+"), ks_pool.index("n0")
    inside = (eb[:-1] >= lo * (1 - 1e-9)) & (eb[1:] <= hi * (1 + 1e-9))
    assert np.all(d["phi_top"][[p, n]][:, ~inside] == 0)
    assert np.all(d["phi_top"][[p, n]][:, inside] > 0)
    others = [i for i in range(ks_pool.nS) if i not in (p, n)]
    assert np.all(d["phi_top"][others] == 0)
    # group sum = GSF's own p and n fluxes (total-energy convention)
    t = np.geomspace(1.0, 3000.0, 7)
    jp, jn = M.mc_primary_flux(t)
    gp, gn = mcprim.GSFNucleons().proton_neutron(t, "total")
    np.testing.assert_allclose(jp, gp, rtol=1e-10)
    np.testing.assert_allclose(jn, gn, rtol=1e-10)
    assert mcprim.resolve_convention(mcprim.ENERGY_CONVENTION,
                                     name="GSFG:GH") == "total"
    # bin average: bracketed by the bin-edge values and close to the centre
    k = int(np.argmin(np.abs(d["e_grid"] - 10.0)))
    a, b = M.mc_primary_flux(np.array([eb[k], eb[k + 1]]))[0]
    assert b < d["phi_top"][p, k] < a
    c = M.mc_primary_flux(d["e_grid"][k])[0][0]
    assert d["phi_top"][p, k] / c == pytest.approx(1.0, abs=0.01)


# ------------------------------------------------------------------ sum rules
def test_sum_rules_report(ks_pool):
    E = ks_pool.d["e_grid"]
    ks_list = [17, 20, 25, 30, 40, 50]
    print("\ninteractions: E_kept/E_projectile and baryon number per interaction")
    sr = M.sum_rules(ks_pool, KIND_INT)
    for p in sr:
        e, b = sr[p]
        print(f"  {p:10s}", " ".join(f"{E[k]:7.3g}:{e[k]:5.3f}/{b[k]:5.2f}"
                                     for k in ks_list))
    print("decays: E_kept/E_parent and baryon number per decay")
    sd = M.sum_rules(ks_pool, KIND_DEC)
    for p in sd:
        e, b = sd[p]
        print(f"  {p:10s}", " ".join(f"{E[k]:7.3g}:{e[k]:5.3f}/{b[k]:5.2f}"
                                     for k in (0, 5, 10, 20, 30, 45)))
    # checks that are exact up to the binning: pion decay keeps all its energy
    e, b = sd["pi+"]
    assert np.all(np.abs(e[15:45] - 1.0) < 0.012)
    # Lambda decays keep baryon number one once the nucleon is above threshold
    e, b = sd["Lambda0"]
    assert np.allclose(b[15:45], 1.0, atol=1e-9)
    e, b = sd["Lambdabar0"]
    assert np.allclose(b[15:45], -1.0, atol=1e-9)
    # nucleon interactions keep at least the projectile's baryon number
    for p in ("p+", "n0"):
        e, b = sr[p]
        assert np.all(b[20:51] > 1.0)
