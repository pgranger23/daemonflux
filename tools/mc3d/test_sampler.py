"""Sampler gates (PHASE2_PLAN.md sec. 3, gate G3)."""

import os

import numpy as np
import pytest

import primaries as pr
from constants import MASS
from interactions import MCEqYieldBackend

TABLES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "mceq_tables_SIBYLL23D.npz")


# --------------------------------------------------------------------------
# primary energy sampler
# --------------------------------------------------------------------------
def test_energy_weights_reproduce_the_input_spectrum():
    """A log-uniform sample re-weighted by w_E must integrate back to the
    model's own dJ/dE in every decade."""
    pytest.importorskip("crflux")
    model = pr.primary_model("H3a")
    rng = np.random.default_rng(3)
    n = 400000
    e, lr = pr.sample_energy(rng, n, 1.0, 1.0e4)
    w = pr.energy_weights(model, e, 1, 14, lr, n)
    for lo, hi in ((1, 10), (10, 100), (100, 1000), (1000, 10000)):
        m = (e >= lo) & (e < hi)
        got = w[m].sum()
        xs = np.exp(np.linspace(np.log(lo), np.log(hi), 400))
        ref = np.trapezoid(pr.nucleon_intensity(model, 14, xs, 1), xs)
        assert got == pytest.approx(ref, rel=0.02), (lo, hi, got, ref)


def test_species_sampler_conserves_nucleons_and_rigidity():
    pytest.importorskip("crflux")
    model = pr.primary_model("H3a")
    rng = np.random.default_rng(7)
    pdg, e, rig, w = pr.sample_species(rng, model, 50000)
    assert set(np.unique(pdg)) <= {2212, 2112}
    assert (w > 0).all()
    # helium (A/Z = 2) must show up as rigidity ~= 2 x momentum-per-nucleon
    # -> the ensemble rigidity/energy ratio spans 1..~2.6 (Fe)
    # rigidity/energy-per-nucleon: ~1 for protons, ~A/Z = 2.0-2.15 for the
    # heavy groups at high energy, dropping below 1 near the nucleon rest mass
    ratio = rig / e
    assert ratio.max() < 2.2
    hi = e > 20.0
    assert np.percentile(ratio[hi], 5) > 0.95
    assert np.percentile(ratio[hi], 95) == pytest.approx(2.1, abs=0.15)


# --------------------------------------------------------------------------
# hadronic yield sampler
# --------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(TABLES), reason="tables not built")
def test_yield_sampler_reproduces_the_mceq_column():
    b = MCEqYieldBackend(TABLES)
    rng = np.random.default_rng(11)
    e_tot = 100.0 + MASS[2212]
    j = b.bin_of(e_tot, 2212)
    tab = b.y[2212][211]
    tot, _, _ = b._prep(tab)
    n = 60000
    es = []
    for _ in range(n):
        for c, e in b.interact(rng, 2212, e_tot):
            if c == 211:
                es.append(e - MASS[211])
    es = np.array(es)
    assert len(es) / n == pytest.approx(tot[j], rel=0.01)
    h, _ = np.histogram(es, bins=b.e_bins)
    ref = tab[:, j]
    m = ref * n > 500          # bins with enough statistics to test
    dev = np.abs(h[m] / n - ref[m]) / ref[m]
    assert dev.max() < 0.06, dev.max()


@pytest.mark.skipif(not os.path.exists(TABLES), reason="tables not built")
def test_interaction_length_matches_mceq():
    """lambda = <A> m_p / sigma with <A> = 14.6568, MCEq's own value."""
    b = MCEqYieldBackend(TABLES)
    lam = b.lambda_int(2212, 100.0 + MASS[2212])
    assert 80.0 < lam < 95.0            # p-air is ~86-90 g/cm2 at 100 GeV
    lam_pi = b.lambda_int(211, 100.0)
    assert 105.0 < lam_pi < 130.0       # pi-air is longer


@pytest.mark.skipif(not os.path.exists(TABLES), reason="tables not built")
def test_kinetic_vs_total_energy_conversion():
    """The table lookup must convert total -> kinetic; a proton of total energy
    m_p + 100 GeV sits in the same bin as MCEq's 100 GeV column."""
    b = MCEqYieldBackend(TABLES)
    j = b.bin_of(100.0 + MASS[2212], 2212)
    assert b.e_bins[j] <= 100.0 <= b.e_bins[j + 1]
    # a 1 GeV *total*-energy proton has only 62 MeV of kinetic energy: it must
    # land in the same bin as a 62 MeV lookup, not a 1 GeV one
    assert b.bin_of(1.0, 2212) == b.bin_of(1.0 - MASS[2212], None)
    assert b.bin_of(1.0, 2212) != b.bin_of(1.0, None)


@pytest.mark.skipif(not os.path.exists(TABLES), reason="tables not built")
def test_decay_tables_contain_the_muon():
    """Regression: MCEq stores pi/K -> mu under HELICITY states, so a naive
    export of (211,0)->(-13,0) is empty and loses every muon."""
    b = MCEqYieldBackend(TABLES, decays=True)
    eg = b.e_grid
    j = int(np.argmin(np.abs(eg - 89.0)))
    assert b.dec[211][-13][:, j].sum() == pytest.approx(1.0, abs=0.01)
    assert b.dec[321][-13][:, j].sum() == pytest.approx(0.6356 + 0.0335,
                                                        abs=0.01)
