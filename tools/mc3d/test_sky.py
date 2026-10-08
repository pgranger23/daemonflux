"""Milestone-2b machinery: the injection patch, the nested-cap scorer and the
correlated 1D reference."""

import numpy as np
import pytest

import geometry as geo
import scoring
from constants import R_EARTH_CM
from scoring import DualCapScorer, NestedCaps

R_INJ = R_EARTH_CM + 100.0e5
LAT, LON = 36.4267, 137.31


# ---------------------------------------------------------------------------
# the injection-patch sampler
# ---------------------------------------------------------------------------
def test_patch_positions_are_uniform_per_unit_area():
    rng = np.random.default_rng(1)
    axis = np.array(geo.local_frame(LAT, LON)[0])
    th = 40.0
    r, _u = geo.sample_injection_patch(rng, R_INJ, axis, th, 200000)
    ca = (r @ axis) / R_INJ
    cmin = np.cos(np.deg2rad(th))
    assert ca.min() > cmin - 1e-9 and ca.max() <= 1.0 + 1e-9
    # uniform per unit area <=> cos(alpha) uniform on [cos theta_inj, 1]
    x = (ca - cmin) / (1.0 - cmin)
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        assert abs(np.mean(x < q) - q) < 0.005
    # and azimuthally symmetric about the axis
    t1, t2 = geo._tangent_basis(axis)
    ph = np.arctan2(r @ t2, r @ t1)
    assert abs(np.mean(ph)) < 0.02


def test_patch_directions_are_lambert_and_inward():
    rng = np.random.default_rng(2)
    axis = np.array(geo.local_frame(LAT, LON)[0])
    r, u = geo.sample_injection_patch(rng, R_INJ, axis, 40.0, 200000)
    n_hat = r / np.linalg.norm(r, axis=1)[:, None]
    mu = -np.einsum("ij,ij->i", u, n_hat)
    assert mu.min() > 0.0                      # strictly inward
    assert abs(np.mean(np.linalg.norm(u, axis=1)) - 1.0) < 1e-12
    # p(mu) = 2 mu  =>  CDF(mu) = mu^2, <mu> = 2/3, <mu^2> = 1/2
    assert abs(np.mean(mu) - 2.0 / 3.0) < 0.004
    assert abs(np.mean(mu ** 2) - 0.5) < 0.004
    for q in (0.2, 0.5, 0.8):
        assert abs(np.mean(mu < q) - q ** 2) < 0.005


def test_patch_area():
    a = geo.patch_area_cm2(R_INJ, 40.0)
    assert abs(a / (2 * np.pi * R_INJ ** 2 * (1 - np.cos(np.deg2rad(40)))) - 1) \
        < 1e-12


# ---------------------------------------------------------------------------
# the nested-cap scorer against an analytic isotropic field
# ---------------------------------------------------------------------------
def test_cap_scorer_reproduces_an_isotropic_field():
    """A synthetic isotropic neutrino field of unit intensity must come back
    as unit intensity in **every** down-going (cos Z, azimuth) bin.

    Construction: an isotropic intensity ``Phi = 1`` outside a source sphere of
    radius ``R_s > R_E`` is sampled exactly by taking positions uniform on that
    sphere and directions from the inward Lambert law, with a per-sample rate
    weight ``w = pi A_s / n``.  Every ray that reaches the ground from outside
    must have crossed the source sphere inward, so nothing is missed, and the
    estimator
    ``Phi_est = sum(w / cos psi) / (A_cap * dOmega * dE)``
    must return 1.  This tests the whole normalisation chain at once: the cap
    area division, the ``1/cos`` incidence factor, the solid-angle measure and
    the ``cos Z`` / azimuth binning -- and, because the field is isotropic, it
    also *is* the "equal counts per unit solid angle" test.
    """
    rng = np.random.default_rng(7)
    r_s = R_EARTH_CM + 50.0e5
    n = 400000
    caps = NestedCaps(LAT, LON, (5.0, 10.0))
    e_bins = np.array([1.0, 2.0])
    sc = DualCapScorer(caps, e_bins, np.linspace(-1, 1, 21),
                       np.linspace(0, 360, 13))
    a_s = 4.0 * np.pi * r_s ** 2
    w = np.pi * a_s / n
    r, u = geo.sample_injection(rng, r_s, n)
    for i in range(n):
        sc.add(14, 1.4, r[i], u[i], w)
    de = e_bins[1] - e_bins[0]
    dcz = 0.1
    daz = 2.0 * np.pi / 12.0
    dom = dcz * daz
    k = scoring.SPECIES.index(14)
    phi = sc.s1[-1, k, 0] / (dom * de)
    err = np.sqrt(sc.s1sq[-1, k, 0]) / (dom * de)
    down, derr = phi[10:, :], err[10:, :]
    assert np.all(down > 0)
    # Each bin holds only ~n * (A_ground/A_s) * (A_cap/A_ground) * dOmega/(2 pi)
    # ~ 25 samples, so the per-bin scatter is 20%: the test has to be a pull
    # test, not a fixed tolerance.  The *mean* over the 120 down-going bins is
    # what is tight.
    pull = (down - 1.0) / np.maximum(derr, 1e-30)
    assert np.all(np.abs(pull) < 5.0), np.abs(pull).max()
    assert abs(np.mean(pull)) < 3.0 / np.sqrt(pull.size), np.mean(pull)
    assert np.mean(down) == pytest.approx(1.0, rel=0.03)
    # up-going bins are empty by construction (a straight ray leaving the
    # ground sphere outward never returns) -- documented in DualCapScorer
    assert sc.s1[-1, scoring.SPECIES.index(14), 0, :10, :].sum() == 0.0


def test_nested_caps_are_nested_and_monotone():
    rng = np.random.default_rng(3)
    caps = NestedCaps(LAT, LON, (2.5, 5.0, 7.5, 10.0))
    r, u = geo.sample_injection(rng, R_EARTH_CM + 30.0e5, 4000)
    hits = np.zeros(4)
    for i in range(4000):
        loc = caps.locate(r[i], u[i])
        if loc is None:
            continue
        hits[loc[0]:] += 1
    assert np.all(np.diff(hits) >= 0)
    # areas scale as (1 - cos theta): the hit counts must too, to ~sqrt(N)
    a = np.array(caps.areas)
    ref = hits[-1] * a / a[-1]
    assert np.all(np.abs(hits - ref) < 5.0 * np.sqrt(np.maximum(ref, 1.0)))


def test_cap_scorer_local_frame_is_at_the_crossing_point():
    """A vertical ray onto the cap edge must be scored at cos Z = 1 in the
    local frame of the crossing point, not at cos Z = cos(theta_D)."""
    caps = NestedCaps(LAT, LON, (10.0,))
    up = np.array(caps.axis)
    n2, e2 = geo.local_frame(LAT, LON)[1:]
    edge = np.cos(np.deg2rad(9.0)) * up + np.sin(np.deg2rad(9.0)) * n2
    r = (R_EARTH_CM + 20.0e5) * edge
    loc = caps.locate(r, -edge)
    assert loc is not None
    assert loc[1] == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# the correlated 1D reference
# ---------------------------------------------------------------------------
def test_collinear_cascade_makes_the_two_tallies_identical():
    """In collinear mode every secondary stays on the primary's line, so the
    3D crossing and the 1D (collinear) crossing are the *same* point and the
    two tallies must agree bin by bin.  That is the sharpest available test of
    the 1D scorer, and it is the reduced-statistics version of the milestone-1
    collinear closure: ``sky.py --collinear`` is the same code path.
    """
    import os
    from interactions import MCEqYieldBackend
    from shower import Config, run_shower
    tables = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "mceq_tables_SIBYLL23D.npz")
    backend = MCEqYieldBackend(tables)
    cfg = Config(collinear=True, bfield=None, e_nu_min=0.1)
    caps = NestedCaps(LAT, LON, (5.0, 10.0))
    sc = DualCapScorer(caps, np.logspace(-1, 2, 31), np.linspace(-1, 1, 21),
                       np.linspace(0, 360, 13))
    rng = np.random.default_rng(4)
    axis = np.array(caps.axis)
    r0, u0 = geo.sample_injection_patch(rng, R_INJ, axis, 15.0, 400)
    for i in range(400):
        sc.begin_shower(r0[i], u0[i])
        run_shower(rng, 2212, 60.0, r0[i], u0[i], backend, cfg, sc, 1.0)
    assert sc.s1.sum() > 0.0
    assert np.allclose(sc.s1, sc.t1, rtol=1e-12, atol=0.0)
    assert np.allclose(sc.s0, sc.t0, rtol=1e-12, atol=0.0)


@pytest.mark.slow
def test_chromo_collinear_yield_matches_the_mceq_yield_backend():
    """Rung B at 1/2000 of the statistics: the chromo cascade's neutrino yield
    must sit within the measured generator difference of the MCEq-yield
    cascade (both with our own decay kinematics), which bounds any wiring
    error in the pool backend."""
    import os
    import atmosphere as atm
    from constants import M_P
    from interactions import ChromoBackend, MCEqYieldBackend
    from scoring import YieldScorer
    from shower import Config, run_shower
    cache = os.environ.get("MC3D_POOL_CACHE")
    if not cache or not os.path.isdir(cache):
        pytest.skip("set MC3D_POOL_CACHE to a prebuilt pool directory")
    e_bins = np.logspace(-1, 1, 5)
    r0 = np.array([0.0, 0.0, R_EARTH_CM + atm.H_TOP_CM])
    u0 = np.array([0.0, 0.0, -1.0])
    cfg = Config(collinear=True, bfield=None, e_nu_min=0.1)
    out = {}
    for nm, be in (("mceq", MCEqYieldBackend()),
                   ("chromo", ChromoBackend(cache_dir=cache, readonly=True))):
        sc = YieldScorer(e_bins)
        rng = np.random.default_rng(9)
        for _ in range(200):
            run_shower(rng, 2212, 100.0 + M_P, r0, u0, be, cfg, sc, 1.0)
        sc.n_prim = 200.0
        out[nm] = sc.dnde(14)[0]
    r = out["chromo"] / np.maximum(out["mceq"], 1e-30)
    assert np.all(r[:2] > 0.7) and np.all(r[:2] < 1.3), r
