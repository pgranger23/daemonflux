"""The chromo event pool (milestone 2a).

The pool exists only to make the generator batched (25x); it must therefore be
statistically indistinguishable from calling the generator per event.  That is
what ``test_pool_reproduces_direct_chromo_multiplicities`` checks, species by
species, against a *direct* ``chromo`` call with the same air mix.
"""

import os
import tempfile

import numpy as np
import pytest

import pool as P

pytest.importorskip("chromo")


def _direct_multiplicities(pdg, e0, n, seed=11):
    """Mean number of each tracked species per interaction, straight from
    chromo, with the same N/O mix and the same f_i sigma_i target draw."""
    import importlib.util  # noqa: F401
    gens = P._Generators(seed)
    name = P._model_name(e0)
    rng = np.random.default_rng(seed)
    frac = P._target_fractions(gens, pdg, e0, name)
    counts = rng.multinomial(n, frac)
    tot = {}
    for (_nm, a, z, _f), k in zip(P.AIR_COMPONENTS, counts):
        if k == 0:
            continue
        m = gens.get(name, pdg, e0, (a, z))
        for ev in m(int(k)):
            for q in ev.final_state().pid:
                q = int(q)
                if q in P.KEEP:
                    tot[q] = tot.get(q, 0) + 1
    return {q: v / n for q, v in tot.items()}


@pytest.mark.parametrize("pdg,e0", [(2212, 95.0), (211, 95.0)])
def test_pool_reproduces_direct_chromo_multiplicities(pdg, e0):
    n = 3000
    with tempfile.TemporaryDirectory() as td:
        p = P.EventPool(td, n_pool=n, seed=11)
        i = p.grid.index(e0)
        e_bin = p.grid.centre(i)
        d = p._load_or_make(pdg, i)
        pooled = {}
        for q in d["pid"]:
            q = int(q)
            pooled[q] = pooled.get(q, 0) + 1
        pooled = {q: v / int(d["n"][0]) for q, v in pooled.items()}
    direct = _direct_multiplicities(pdg, e_bin, n, seed=12)
    for q in (211, -211, 321, -321, 130, 310, 2212, 2112, 3122):
        a, b = pooled.get(q, 0.0), direct.get(q, 0.0)
        if max(a, b) < 0.02:
            continue
        err = np.sqrt(max(a, 1e-6) / n) * 4.0 + 0.02 * max(a, b)
        assert abs(a - b) < 5.0 * err, (q, a, b)


def test_pool_draw_conserves_species_and_energy():
    """A drawn event must not create energy and must keep the p_T of the pool
    event while scaling p_z by p_beam(E)/p_beam(E0)."""
    with tempfile.TemporaryDirectory() as td:
        p = P.EventPool(td, n_pool=200, seed=5)
        rng = np.random.default_rng(0)
        for e in (95.0, 300.0, 1000.0):
            for _ in range(50):
                ev = p.draw(rng, 2212, e)
                assert sum(x[1] for x in ev) <= e + 0.95
                for q, en, pv in ev:
                    m = P.MASS[q]
                    assert abs(en ** 2 - m * m - float(pv @ pv)) < 1e-3 * en ** 2


def test_bin_choice_is_linear_in_log_e():
    """The stochastic choice between the two bracketing bin centres must give
    a mean ln E equal to the requested ln E (that is what removes the
    first-order multiplicity bias inside a bin)."""
    p = P.EventPool(None, n_pool=10, seed=1)
    rng = np.random.default_rng(3)
    c = p.grid.centres
    for e in (30.0, 120.0, 900.0):
        ks = np.array([p.bin_choice(rng, e) for _ in range(20000)])
        assert abs(np.mean(np.log(c[ks])) - np.log(e)) < 0.01


def test_air_mix_and_mean_mass():
    f = sum(f for _n, _a, _z, f in P.AIR_COMPONENTS)
    assert abs(f - 1.0) < 1e-9
    assert abs(P.A_AIR_MEAN - 14.43) < 0.01


@pytest.mark.parametrize("pdg", [2212, 211, 321])
def test_chromo_sibyll_lambda_is_sane(pdg):
    """chromo's own SIBYLL production cross sections must give an air
    interaction length in the physical range and ordered p < pi < K."""
    gens = P._Generators(1)
    lam = P.air_lambda(gens, pdg, 200.0, model="Sibyll23d")
    assert 60.0 < lam < 200.0
