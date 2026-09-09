"""Decay-kinematics gates (PHASE2_PLAN.md sec. 4.4, gate G4)."""

import numpy as np
import pytest

import decays as dk
from constants import M_K, M_MU, M_PI
from lorentz import boost, mass_of


def _rng():
    return np.random.default_rng(20260910)


def test_two_body_cm_momentum():
    p = dk.two_body_cm(M_PI, M_MU, 0.0)
    assert p == pytest.approx((M_PI ** 2 - M_MU ** 2) / (2 * M_PI), rel=1e-12)
    assert dk.two_body_cm(M_MU, M_PI, 0.0) == 0.0


def test_two_body_conserves_energy_and_mass():
    rng = _rng()
    p4 = np.array([10.0, 0.0, 0.0, np.sqrt(100.0 - M_PI ** 2)])
    for _ in range(200):
        q1, q2, n = dk.two_body(rng, p4, M_PI, M_MU, 0.0)
        assert np.allclose(q1 + q2, p4, atol=1e-9)
        assert mass_of(q1) == pytest.approx(M_MU, abs=1e-7)
        assert mass_of(q2) == pytest.approx(0.0, abs=1e-6)
        assert np.linalg.norm(n) == pytest.approx(1.0)


def test_pi_to_mu_nu_neutrino_spectrum_is_flat():
    """For an ultra-relativistic pion the lab neutrino energy is uniform on
    [0, (1 - r) E_pi] with r = (m_mu/m_pi)^2."""
    rng = _rng()
    e_pi = 20.0
    r = (M_MU / M_PI) ** 2
    p4 = np.array([e_pi, 0.0, 0.0, np.sqrt(e_pi ** 2 - M_PI ** 2)])
    xs = []
    for _ in range(30000):
        for pdg, q, pol in dk.decay(rng, 211, p4):
            if pdg == 14:
                xs.append(q[0] / e_pi)
    xs = np.array(xs)
    assert xs.max() < (1 - r) * 1.001
    assert xs.mean() == pytest.approx(0.5 * (1 - r), rel=0.01)
    # flatness: the four quartiles of [0, 1-r] must be equally populated
    h, _ = np.histogram(xs, bins=np.linspace(0, 1 - r, 5))
    assert h.std() / h.mean() < 0.03


def test_pi_to_mu_nu_muon_spectrum():
    """The muon carries x in [r, 1] uniformly."""
    rng = _rng()
    e_pi = 20.0
    r = (M_MU / M_PI) ** 2
    p4 = np.array([e_pi, 0.0, 0.0, np.sqrt(e_pi ** 2 - M_PI ** 2)])
    xs = [q[0] / e_pi for _ in range(20000)
          for pdg, q, pol in dk.decay(rng, 211, p4) if pdg == -13]
    xs = np.array(xs)
    assert xs.min() > r * 0.999
    assert xs.mean() == pytest.approx(0.5 * (1 + r), rel=0.01)


def test_muon_from_pion_is_fully_polarised_at_the_kinematic_edges():
    """P_L -> -h at x -> 1 (forward) and +h at x -> r (backward) for mu+."""
    rng = _rng()
    e_pi = 200.0
    p4 = np.array([e_pi, 0.0, 0.0, np.sqrt(e_pi ** 2 - M_PI ** 2)])
    hi, lo = [], []
    for _ in range(4000):
        out = dk.decay(rng, 211, p4)
        q = [o for o in out if o[0] == -13][0]
        x = q[1][0] / e_pi
        if x > 0.97:
            hi.append(q[2])
        elif x < 0.60:
            lo.append(q[2])
    assert len(hi) > 20 and len(lo) > 20
    # mu+ has helicity -1 in the pion rest frame: forward (x -> 1) keeps it,
    # backward-in-the-pion-frame (x -> r) reverses the lab momentum and so
    # flips the sign of the longitudinal polarisation.
    assert np.mean(hi) < -0.9, np.mean(hi)
    assert np.mean(lo) > 0.9, np.mean(lo)
    # and |P_L| = 1 everywhere for a two-body decay
    assert max(abs(np.mean(hi)), abs(np.mean(lo))) <= 1.0 + 1e-9


def test_michel_rest_frame_spectra():
    """Sampled x = 2E*/m_mu must follow n(x)."""
    rng = _rng()
    # (a) shape: the inverse-CDF table itself, vectorised so the test is cheap
    edges = np.linspace(0, 1, 11)
    for kind in ("numu", "nue"):
        x = np.interp(rng.random(400000), dk._CDF[kind], dk._XE)
        h, _ = np.histogram(x, bins=edges, density=True)
        ref = np.array([
            np.trapezoid(dk.michel_n(np.linspace(a, b, 64), kind),
                         np.linspace(a, b, 64)) / (b - a)
            for a, b in zip(edges[:-1], edges[1:])])
        m = ref > 0.2
        assert np.max(np.abs(h[m] - ref[m]) / ref[m]) < 0.02, (kind, h, ref)
    # (b) the full sampler (boost included) reproduces <x>
    p4 = np.array([M_MU * 1.0000001, 0.0, 0.0, 1e-4])
    for kind, mean in (("numu", 0.7), ("nue", 0.6)):
        xs = np.array([2.0 * dk.michel_sample(rng, p4, +1, 0.0, kind)[0] / M_MU
                       for _ in range(20000)])
        assert xs.max() < 1.02
        assert xs.mean() == pytest.approx(mean, rel=0.01), kind


def test_michel_polarisation_asymmetry_sign():
    """A helicity -1 mu+ makes a HARDER nu_e than an unpolarised one, and a
    helicity +1 mu+ a softer one.  (This is the sign MCEq's helicity-resolved
    matrices confirm -- see test_michel_polarised_spectra_match_mceq -- and it
    is the direction that matters physically, because a mu+ from pi+ decay is
    predominantly helicity -1.)"""
    rng = _rng()
    e = 50.0
    p4 = np.array([e, 0.0, 0.0, np.sqrt(e ** 2 - M_MU ** 2)])
    means = {}
    for pol in (0.0, -1.0, +1.0):
        xs = np.array([dk.michel_sample(rng, p4, +1, pol, "nue")[0]
                       for _ in range(20000)])
        means[pol] = xs.mean()
    assert means[-1.0] > means[0.0] > means[+1.0], means
    assert means[-1.0] / means[0.0] == pytest.approx(4.0 / 3.0, rel=0.05)


def test_kaon_branching_ratios_match_mceq():
    """The BR table must reproduce MCEq's own decay matrices."""
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "mceq_tables_SIBYLL23D.npz")
    if not os.path.exists(path):
        pytest.skip("MCEq tables not built")
    d = np.load(path)
    eg = d["e_grid"]
    j = int(np.argmin(np.abs(eg - 89.0)))
    rng = _rng()
    e_k = 50.0
    p4 = np.array([e_k, 0.0, 0.0, np.sqrt(e_k ** 2 - M_K ** 2)])
    n = 20000
    cnt = {}
    for _ in range(n):
        for pdg, q, pol in dk.decay(rng, 321, p4):
            cnt[pdg] = cnt.get(pdg, 0) + 1
    # PDG branching fractions our table encodes
    pdg_ref = {14: 0.6356 + 0.0335,     # K_mu2 + K_mu3
               12: 0.0507,              # K_e3
               -13: 0.6356 + 0.0335,
               211: 0.2067 + 2 * 0.0560 + 0.0176,
               -211: 0.0560}
    for c, ref in pdg_ref.items():
        got = cnt.get(c, 0) / n
        assert got == pytest.approx(ref, abs=0.005 + 0.02 * ref), (c, got, ref)
    # ... and how they compare to MCEq's own decay matrices.  These agree for
    # every channel EXCEPT nu_mu, where MCEq's d_321_14 = 0.6353 is the K_mu2
    # branch alone -- it carries the K_mu3 MUON (d_321_-13 = 0.6692 = K_mu2 +
    # K_mu3) but not the K_mu3 neutrino.  That is a ~5% difference on the kaon
    # nu_mu yield, i.e. <1% of the total sub-GeV nu_mu, and it is the leading
    # known reason for the A1-vs-A2 offset in the closure ladder.
    for c, key, tol in ((12, "d_321_12", 0.01), (-13, "d_321_-13", 0.02),
                        (211, "d_321_211", 0.02), (-211, "d_321_-211", 0.01)):
        ref = float(d[key][:, j].sum())
        got = cnt.get(c, 0) / n
        assert got == pytest.approx(ref, abs=tol + 0.03 * ref), (c, got, ref)
    assert cnt.get(14, 0) / n == pytest.approx(
        float(d["d_321_14"][:, j].sum()) + 0.0335, abs=0.02)


def test_lambda_decay():
    rng = _rng()
    e = 20.0
    from constants import MASS
    p4 = np.array([e, 0.0, 0.0, np.sqrt(e ** 2 - MASS[3122] ** 2)])
    cnt = {}
    n = 5000
    for _ in range(n):
        for pdg, q, pol in dk.decay(rng, 3122, p4):
            cnt[pdg] = cnt.get(pdg, 0) + 1
    assert cnt[2212] / n == pytest.approx(0.639, abs=0.02)
    assert cnt[-211] / n == pytest.approx(0.639, abs=0.02)
    assert cnt[2112] / n == pytest.approx(0.358, abs=0.02)


def test_three_body_conserves_four_momentum():
    rng = _rng()
    e_k = 30.0
    p4 = np.array([e_k, 0.3, 0.0, np.sqrt(e_k ** 2 - M_K ** 2 - 0.09)])
    from constants import MASS
    for _ in range(300):
        res = dk.three_body(rng, p4, M_K,
                            [MASS[111], MASS[11], 0.0])
        assert res is not None
        tot = res[0] + res[1] + res[2]
        assert np.allclose(tot, p4, atol=1e-8)


def test_michel_matches_muon_segment_mc():
    ms = pytest.importorskip("muon_segment_mc")
    x = np.linspace(0.01, 0.99, 25)
    for kind in ("numu", "nue"):
        assert np.allclose(dk.michel_n(x, kind), ms.michel_n(x, kind))
        assert np.allclose(dk.michel_asym(x, kind), ms.michel_asym(x, kind))


def test_michel_polarised_spectra_match_mceq():
    """The polarised Michel spectra must match MCEq's helicity-resolved decay
    matrices, sign included.  This is the gate that caught the flipped
    asymmetry sign (which made the nu_e closure 14% low)."""
    pytest.importorskip("MCEq")
    import importlib.util  # noqa: F401
    import MCEq.config as cfg
    cfg.e_min = 0.05
    cfg.debug_level = 0
    cfg.muon_helicity_dependence = True
    import crflux.models as crf
    from MCEq.core import MCEqRun
    mc = MCEqRun(interaction_model="SIBYLL23D",
                 primary_model=(crf.HillasGaisser2012, "H3a"), theta_deg=0.0)
    eg = mc.e_grid
    xr = eg[:, None] / eg[None, :]
    j = int(np.argmin(np.abs(eg - 17.78)))
    rng = _rng()
    e_tot = eg[j] + M_MU
    p4 = np.array([e_tot, 0.0, 0.0, np.sqrt(e_tot ** 2 - M_MU ** 2)])
    for child, kind in ((12, "nue"), (-14, "numu")):
        for hel in (-1, 0, 1):
            col = (mc._decays.get_matrix((-13, hel), (child, 0)) * xr)[:, j]
            if col.sum() < 1e-6:
                continue
            ref = float((col * eg).sum() / col.sum())
            # MCEq's helicity index is the muon helicity; our P_L has the same
            # meaning, so P_L = hel.
            es = np.array([dk.michel_sample(rng, p4, +1, float(hel), kind)[0]
                           for _ in range(30000)])
            assert es.mean() == pytest.approx(ref, rel=0.02), (child, hel,
                                                               es.mean(), ref)


def test_stopped_muon_decays_isotropically():
    """A muon at rest must not divide by |p| = 0."""
    rng = _rng()
    p4 = np.array([M_MU, 0.0, 0.0, 0.0])
    q = np.array([dk.michel_sample(rng, p4, +1, -1.0, "nue")
                  for _ in range(2000)])
    assert np.isfinite(q).all()
    assert q[:, 0].max() < 0.5 * M_MU * 1.01
    assert np.abs(q[:, 1:].mean(axis=0)).max() < 0.002
