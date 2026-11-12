"""Run pithr: the tracking-threshold schemes of ``shower.thresholds``.

``legacy`` (default) is every earlier run's threshold table, unchanged; ``chain``
drops a particle only if no decay / interaction chain from it can give a
neutrino above ``e_nu_min`` (runs/pithr/DERIVATION.md).  The scheme is an
option of ``Config`` / ``sky3d.Options`` and is recorded in the run metadata."""

import subprocess
import sys

import numpy as np
import pytest

import shower
import sky3d
from constants import MASS, M_MU, M_N, NEUTRINOS
from decays import decay

M_PI = MASS[211]


def _p4(pdg, e):
    p = np.sqrt(max(e * e - MASS[pdg] ** 2, 0.0))
    return np.array([e, 0.0, 0.0, p])


def _max_chain_nu(rng, pdg, e, n):
    """Largest neutrino energy over ``n`` full decay chains of one particle."""
    best = 0.0
    for _ in range(n):
        stack = [(pdg, _p4(pdg, e), 0.0)]
        while stack:
            q, v, pol = stack.pop()
            if q in NEUTRINOS:
                best = max(best, v[0])
            elif abs(q) in (13, 211, 321) or q in (130, 310):
                stack += decay(rng, q, v, pol if abs(q) == 13 else 0.0)
    return best


# ---------------------------------------------------------------------------
# legacy: unchanged
# ---------------------------------------------------------------------------
def test_legacy_values_unchanged():
    r_pi = (M_MU / M_PI) ** 2
    r_k = (M_MU / MASS[321]) ** 2
    t = shower.thresholds(0.1)
    assert t == shower.thresholds(0.1, scheme="legacy")
    for p in (211, -211):
        assert t[p] == 0.1 / (1 - r_pi)
        assert abs(t[p] - 0.2342) < 5e-5
    for p in (321, -321, 130, 310):
        assert t[p] == 0.1 / (1 - r_k)
    for p in (13, -13):
        assert t[p] == max(M_MU, 0.1)
    for p in (2212, -2212, 2112, -2112):
        assert t[p] == M_N + 0.1 / (1 - r_pi)
    for p in (3122, -3122):
        assert t[p] == MASS[3122] + 0.1 / (1 - r_pi)
    assert set(t) == {211, -211, 321, -321, 130, 310, 13, -13, 2212, -2212,
                      2112, -2112, 3122, -3122}


def test_default_scheme_is_legacy_everywhere():
    assert shower.Config().thr_scheme == "legacy"
    assert shower.Config().thr == shower.thresholds(0.1)
    assert sky3d.Options().thresholds == "legacy"


# ---------------------------------------------------------------------------
# chain: the bounds
# ---------------------------------------------------------------------------
def test_chain_pion_bound_is_E_plus_p_equal_2_enumin():
    for e_nu in (0.05, 0.1, 0.2, 0.5):
        thr = shower.thresholds(e_nu, "chain")[211]
        if 2 * e_nu > M_PI:
            p = np.sqrt(thr ** 2 - M_PI ** 2)
            assert thr + p == pytest.approx(2 * e_nu, rel=1e-12)
        else:
            assert thr == M_PI
    assert shower.thresholds(0.1, "chain")[211] == pytest.approx(0.14869, abs=1e-4)
    assert shower.thresholds(0.1, "chain")[-211] == shower.thresholds(0.1, "chain")[211]


def test_chain_never_drops_what_legacy_keeps_for_physical_energies():
    for e_nu in (0.05, 0.1, 0.2, 0.5):
        leg, ch = shower.thresholds(e_nu), shower.thresholds(e_nu, "chain")
        assert set(leg) == set(ch)
        for p in leg:
            if abs(p) == 321 or p in (130, 310):
                # a K needs E + p >= 2 e_nu_min through K -> mu -> nu: for
                # 2 e_nu_min > m_K this is *higher* than the legacy two-body
                # cut (legacy was not a bound there); equal below
                if 2 * e_nu <= MASS[p]:
                    assert ch[p] == MASS[p] and leg[p] < MASS[p]
                continue
            lo = max(MASS[p], leg[p])           # no particle has E < mass
            assert ch[p] <= lo + 1e-12, (e_nu, p, ch[p], lo)


def test_chain_kaons_and_antinucleons_are_never_dropped():
    ch = shower.thresholds(0.1, "chain")
    for p in (321, -321, 130, 310, -2212, -2112, 3122, -3122):
        assert ch[p] == MASS[p]
    assert ch[13] == ch[-13] == max(M_MU, 0.1)      # muons as in legacy


def test_chain_nucleon_bound_between_threshold_and_legacy():
    ch = shower.thresholds(0.1, "chain")[2212]
    # free-nucleon N N -> N N pi threshold is T = 2 m_pi (1 + m_pi / 4 m_N) = 0.2897
    assert M_N + 0.2897 > ch > M_N + 0.05
    assert ch < shower.thresholds(0.1)[2212]
    assert ch == shower._nucleon_pion_threshold(0.1)
    # without Fermi motion the free-nucleon kinematic threshold is recovered
    free = shower._nucleon_pion_threshold(0.1, p_fermi=0.0)
    assert free - M_N == pytest.approx(0.2897, abs=2e-3)


def test_chain_keeps_a_pion_whose_muon_can_make_a_neutrino():
    rng = np.random.default_rng(11)
    leg, ch = shower.thresholds(0.1), shower.thresholds(0.1, "chain")
    e_pi = 0.20                                  # legacy drops it (< 0.2342)
    assert e_pi < leg[211] and e_pi >= ch[211]
    assert _max_chain_nu(rng, 211, e_pi, 4000) <= (e_pi + np.sqrt(
        e_pi ** 2 - M_PI ** 2)) / 2 + 1e-12
    # the muon of this pion can reach a neutrino above 0.1 GeV (forward muon,
    # forward nu): exact bound (E + p) / 2 = 0.1833 > 0.1
    assert (e_pi + np.sqrt(e_pi ** 2 - M_PI ** 2)) / 2 > 0.1
    # a muon sample that realises it: pi -> mu (highest-energy), mu -> nu
    assert _max_chain_nu(rng, 211, e_pi, 20000) > 0.1


def test_chain_drops_only_pions_that_cannot_make_a_neutrino():
    rng = np.random.default_rng(12)
    ch = shower.thresholds(0.1, "chain")[211]
    for e_pi in (M_PI, 0.5 * (M_PI + ch), ch - 1e-6):
        assert e_pi < ch or e_pi == M_PI
        bound = (e_pi + np.sqrt(max(e_pi ** 2 - M_PI ** 2, 0.0))) / 2
        assert bound < 0.1
        assert _max_chain_nu(rng, 211, e_pi, 3000) < 0.1


@pytest.mark.parametrize("pdg", [321, 130])
def test_chain_neutrino_bound_E_plus_p_over_2_holds_for_kaons(pdg):
    rng = np.random.default_rng(13)
    m = MASS[pdg]
    for e in (m, 2 * m):
        p = np.sqrt(e * e - m * m)
        assert _max_chain_nu(rng, pdg, e, 3000) <= (e + p) / 2 + 1e-12


def test_unknown_scheme_rejected():
    with pytest.raises(ValueError):
        shower.thresholds(0.1, "nope")
    with pytest.raises(ValueError):
        sky3d.Options(thresholds="nope")


# ---------------------------------------------------------------------------
# metadata and plumbing
# ---------------------------------------------------------------------------
def test_config_records_scheme():
    c = shower.Config(thr_scheme="chain")
    assert c.thr_scheme == "chain" and c.thr == shower.thresholds(0.1, "chain")
    assert shower.Config().thr == shower.thresholds(0.1)


def test_options_signature_records_scheme_and_legacy_is_unchanged():
    base = sky3d.Options()
    assert list(base.signature())[-2:] != [-3.0, 1.0]
    chain = sky3d.Options(thresholds="chain")
    s0, s1 = base.signature(), chain.signature()
    assert len(s1) == len(s0) + 2 and list(s1[-2:]) == [-3.0, 1.0]
    assert np.array_equal(s1[:len(s0)], s0)
    # combines with the other opt-in without clobbering it
    both = sky3d.Options(thresholds="chain", upgoing=True).signature()
    assert list(both[-4:]) == [-2.0, 1.0, -3.0, 1.0]


def test_cli_exposes_thresholds_option():
    out = subprocess.run([sys.executable, sky3d.__file__, "--help"],
                         capture_output=True, text=True, timeout=120).stdout
    assert "--thresholds" in out and "legacy" in out and "chain" in out
