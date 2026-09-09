"""Milestone-1 closure at reduced statistics (PHASE2_PLAN.md gate G1).

The full gate lives in ``closure.py`` (200k showers per energy on 40 cores);
this is the same thing at 1/100 of the statistics so it can run in CI.
"""

import os

import numpy as np
import pytest

import atmosphere as atm
from closure import E_BINS, mceq_initial_condition
from constants import M_P, R_EARTH_CM
from interactions import MCEqYieldBackend
from scoring import YieldScorer
from shower import Config, run_shower

TABLES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "mceq_tables_SIBYLL23D.npz")
pytestmark = pytest.mark.skipif(not os.path.exists(TABLES),
                                reason="MCEq tables not built")


def _run(n, e_kin=20.0, seed=5, mode="kinematic", pol=True):
    b = MCEqYieldBackend(TABLES, decays=(mode == "mceq"))
    cfg = Config(collinear=True, e_nu_min=0.1, polarisation=pol)
    rng = np.random.default_rng(seed)
    sc = YieldScorer(E_BINS)
    r0 = np.array([0.0, 0.0, R_EARTH_CM + atm.H_TOP_CM])
    u0 = np.array([0.0, 0.0, -1.0])
    ic = mceq_initial_condition(e_kin, b.e_grid, b.e_bins, b.e_widths)
    wsum = sum(abs(w) for _, w in ic)
    for e, w in ic:
        nk = max(1, int(round(n * abs(w) / wsum)))
        for _ in range(nk):
            run_shower(rng, 2212, e + M_P, r0, u0, b, cfg, sc, w / nk)
    sc.n_prim = 1.0
    return sc


def test_initial_condition_matches_mceq():
    """Our three-bin primary representation IS MCEq's."""
    pytest.importorskip("MCEq")
    import importlib.util  # noqa: F401
    import MCEq.config as cfg
    cfg.e_min = 0.05
    cfg.debug_level = 0
    import crflux.models as crf
    from MCEq.core import MCEqRun
    mc = MCEqRun(interaction_model="SIBYLL23D",
                 primary_model=(crf.HillasGaisser2012, "H3a"), theta_deg=0.0)
    mc.set_single_primary_particle(100.0, pdg_id=2212)
    lidx = mc.pman[2212].lidx
    phi0 = mc._phi0[lidx:lidx + len(mc.e_grid)]
    nz = np.nonzero(phi0)[0]
    ref = [(float(mc.e_grid[k]), float(phi0[k] * mc.e_widths[k])) for k in nz]
    ours = mceq_initial_condition(100.0, mc.e_grid, mc.e_bins, mc.e_widths)
    assert len(ours) == len(ref)
    for (e1, w1), (e2, w2) in zip(ours, ref):
        assert e1 == pytest.approx(e2)
        assert w1 == pytest.approx(w2, rel=1e-9)
    assert sum(w for _, w in ours) == pytest.approx(1.0)


def test_shower_produces_a_sane_neutrino_yield():
    sc = _run(300, 20.0)
    tot = sum(sc.sw[s].sum() for s in sc.sw)
    # ~12 neutrinos above 0.1 GeV per 20 GeV proton (measured 12.11)
    assert 6.0 < tot < 20.0
    # nu_mu-type must dominate nu_e-type
    nmu = sc.sw[14].sum() + sc.sw[-14].sum()
    nel = sc.sw[12].sum() + sc.sw[-12].sum()
    assert nmu > 2.0 * nel


def test_no_particle_crosses_the_earth():
    """Regression for the path_to_exit near/far-root bug: with it, muons take
    thousands of steps below the surface and every one 'decays'."""
    from shower import transport_muon
    cfg = Config(collinear=True, e_nu_min=0.1)
    rng = np.random.default_rng(0)
    r0 = np.array([0.0, 0.0, R_EARTH_CM + atm.H_TOP_CM])
    u0 = np.array([0.0, 0.0, -1.0])
    from constants import CTAU_CM, M_MU
    for e in (20.0, 50.0):
        n, dec = 400, 0
        for _ in range(n):
            k, r, u, ee = transport_muon(rng, -13, e, r0.copy(), u0.copy(), cfg)
            assert np.linalg.norm(r) >= R_EARTH_CM - 10.0
            if k in ("decay", "stop"):
                dec += 1
        lam = np.sqrt(e * e - M_MU ** 2) / M_MU * CTAU_CM[13]
        naive = 1.0 - np.exp(-atm.H_TOP_CM / lam)
        assert dec / n == pytest.approx(naive, abs=0.05), e


@pytest.mark.slow
def test_collinear_closure_against_mceq():
    """The physics gate: collinear + B = 0 must reproduce MCEq's 1D yield."""
    pytest.importorskip("MCEq")
    from closure import mceq_reference, rebin_reference
    sc = _run(4000, 100.0, seed=17, mode="kinematic")
    eg, ref = mceq_reference(100.0, helicity=True)
    rb = rebin_reference(eg, ref, E_BINS)
    ec = np.sqrt(E_BINS[1:] * E_BINS[:-1])
    band = (ec >= 0.1) & (ec < 3.0)
    w = np.diff(E_BINS)
    for s in (14, -14, 12, -12):
        y, err = sc.dnde(s)
        num = np.sum(y[band] * w[band])
        den = np.sum(rb[s][band] * w[band])
        e = np.sqrt(np.sum((err[band] * w[band]) ** 2))
        r = num / den
        assert abs(r - 1.0) < max(0.08, 3 * e / den), (s, r, e / den)
