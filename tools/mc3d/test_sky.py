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


# ---------------------------------------------------------------------------
# milestone 2b: the production cone, the shower counters, the normalisation
# ---------------------------------------------------------------------------
def _rotated(u, axis_perp, alpha_deg):
    a = np.deg2rad(alpha_deg)
    return np.cos(a) * u + np.sin(a) * axis_perp


def test_cone_scorer_bins_the_primary_neutrino_angle():
    """Synthetic: neutrinos at known angles to a known primary land in the
    right (species, band, channel, alpha) cell with exact moments, and the
    shower-level counters / injection-angle histogram see the cap hits."""
    caps = NestedCaps(LAT, LON, (5.0, 10.0))
    sc = DualCapScorer(caps, np.logspace(-1, 2, 61), np.linspace(-1, 1, 21),
                       np.linspace(0, 360, 13))
    up, north, _east = geo.local_frame(LAT, LON)
    r0 = R_INJ * np.asarray(up)
    u0 = -np.asarray(up)
    sc.begin_shower(r0, u0)
    r_prod = (R_EARTH_CM + 20.0e5) * np.asarray(up)
    cases = [  # (pdg, E, parent, gen, alpha_deg, band, channel)
        (14, 0.2, 211, 1, 5.5, 0, 0),
        (14, 0.7, 321, 2, 30.5, 2, 1),
        (12, 2.0, -13, 3, 100.5, 3, 2),
        (-14, 0.4, 2212, 1, 0.2, 1, 3),
    ]
    for pdg, e, par, gen, al, _b, _c in cases:
        sc.add(pdg, e, r_prod, _rotated(u0, np.asarray(north), al), 1.0,
               parent=par, gen=gen)
    sc.end_shower()
    h, m = sc.cone.h, sc.cone.m
    assert h.sum() == pytest.approx(4.0)
    for pdg, e, par, gen, al, b, c in cases:
        k = scoring.SPECIES.index(pdg)
        g = 0 if gen <= 1 else 1
        # vertical primary -> top cos(zenith) class
        assert h[k, b, c, 2, g, int(al)] == pytest.approx(1.0)
        ie = int(np.log10(e / 0.1) * 10)
        assert m[k, ie, c, 2, g, 0] == pytest.approx(1.0)
        assert m[k, ie, c, 2, g, 3] == pytest.approx(al * al, rel=1e-9)
        assert m[k, ie, c, 2, g, 4] == pytest.approx(np.cos(np.deg2rad(al)),
                                                     rel=1e-9)
    # produced 20 km above the site, the three down-going neutrinos land
    # within 12 km of it (inside the cap) and the 100.5-deg one goes up; in
    # the 1D view all four ride the vertical primary onto the site.  None is
    # in the East-West target region (cos Z < 0.3).
    assert sc.shcounts.tolist() == [1.0, 1.0, 1.0, 1.0, 0.0]
    assert sc.hinj[0].sum() == pytest.approx(3.0)
    assert sc.hinj[0, scoring.SPECIES.index(-14), 1, 3, 0] == pytest.approx(1.0)
    assert sc.hinj[0, scoring.SPECIES.index(12)].sum() == 0.0
    assert sc.hinj[1].sum() == pytest.approx(4.0)
    # round trip and additive merge
    d = sc.to_dict()
    sc2 = DualCapScorer.from_dict(d).merge(DualCapScorer.from_dict(d))
    assert np.allclose(sc2.cone.m, 2.0 * m)
    assert np.allclose(sc2.shcounts, 2.0 * sc.shcounts)


class _NoInteraction:
    """Backend stub: nothing interacts, so a hadron primary just decays."""

    def has_interaction(self, pdg):
        return False

    def lambda_int(self, pdg, e_tot):
        return np.inf

    def interact(self, rng, pdg, e_tot):
        return []


def test_cone_of_a_decaying_pion_matches_two_body_kinematics():
    """Physics gate of the cone tally and of the parent tagging: a 2 GeV pi+
    that cannot interact decays isotropically in its rest frame, so the
    direct nu_mu's lab angle to the primary is fixed by two-body kinematics,
    ``cos a = (c + b) / (1 + b c)`` at ``E_nu = g E* (1 + b c)`` with ``c``
    uniform.  Per E_nu band the measured <cos a> and RMS(a) must match the
    exact integral; the nu_mu must be tagged 'pi' and the muon-decay
    neutrinos 'mu'."""
    from constants import M_MU, M_PI
    from shower import Config, run_shower
    e_pi = 2.0
    g = e_pi / M_PI
    b = np.sqrt(1.0 - 1.0 / g ** 2)
    estar = (M_PI ** 2 - M_MU ** 2) / (2.0 * M_PI)
    caps = NestedCaps(LAT, LON, (10.0,))
    sc = DualCapScorer(caps, np.logspace(-1, 2, 61))
    up = np.asarray(geo.local_frame(LAT, LON)[0])
    r0 = (R_EARTH_CM + 20.0e5) * up
    u0 = -up
    cfg = Config(collinear=False, bfield=None, e_nu_min=0.1)
    rng = np.random.default_rng(12)
    n = 12000
    for _ in range(n):
        sc.begin_shower(r0, u0)
        run_shower(rng, 211, e_pi, r0, u0, _NoInteraction(), cfg, sc, 1.0)
        sc.end_shower()
    k = scoring.SPECIES.index(14)
    h = sc.cone.h
    # parent tagging: the nu_mu only from the pion, the rest only from muons
    assert h[k, :, 0].sum() > 0.9 * n * 0.6
    assert h[k, :, 1:].sum() == 0.0
    for s in (12, -14):
        ks = scoring.SPECIES.index(s)
        assert h[ks, :, 0].sum() == 0.0 and h[ks, :, 2].sum() > 0.0
    assert h[:, :, :, :2].sum() == 0.0 and h[:, :, :, :, 1].sum() == 0.0
    # exact two-body expectation (c uniform on [-1, 1])
    c = np.linspace(-1.0, 1.0, 400001)
    e_nu = g * estar * (1.0 + b * c)
    cos_a = (c + b) / (1.0 + b * c)
    alpha = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
    # (i) counts per E_nu band of the alpha histogram
    bands = scoring.CONE_BANDS
    for ib in range(len(bands) - 1):
        p = np.mean((e_nu >= bands[ib]) & (e_nu < bands[ib + 1]))
        nrow = h[k, ib, 0, 2, 0].sum()
        assert abs(nrow - n * p) < 5.0 * np.sqrt(n * p * (1 - p)) + 1e-9, ib
    # (ii) exact (unbinned) moments on the 10/decade grid: <cos a>, RMS(a)
    m = sc.cone.m[k, :, 0, 2, 0]
    eb = sc.cone.e_mom
    tested = 0
    for ie in range(len(eb) - 1):
        sel = (e_nu >= eb[ie]) & (e_nu < eb[ie + 1])
        nn = m[ie, 0]
        if sel.sum() < 2000 or nn < 300:
            continue
        tested += 1
        a2 = alpha[sel] ** 2
        mc_cos, mc_a2 = m[ie, 4] / nn, m[ie, 3] / nn
        assert abs(mc_cos - cos_a[sel].mean()) < \
            5.0 * cos_a[sel].std() / np.sqrt(nn) + 1e-6, (ie, mc_cos)
        assert abs(mc_a2 - a2.mean()) < 5.0 * a2.std() / np.sqrt(nn), \
            (ie, np.sqrt(mc_a2), np.sqrt(a2.mean()))
    assert tested >= 6


def test_stratum_weights_do_not_depend_on_the_chunking():
    """Regression for the chunk-normalisation bug: the summed primary weight
    of a stratum must be ``pi A_patch * int J dE`` however the stratum is cut
    into tasks (before the fix it scaled with the number of chunks)."""
    import primaries as prim
    import sky
    model = prim.primary_model("GSF")
    axis = np.array(geo.local_frame(LAT, LON)[0])
    ns = 2
    e = np.logspace(0.0, 1.0, 4001)
    j, _ = model.nucleon_flux(e)
    exact = np.pi * geo.patch_area_cm2(sky.R_INJ_CM, 40.0) * np.trapezoid(j, e)
    totals = []
    for chunk in (40000, 3000):
        tasks = sky.build_tasks(30000, ["N"], 40.0, chunk, 5, nstrata=ns)
        tot = 0.0
        for g, k, m, th, seed, n_tot in tasks:
            rng = np.random.default_rng(seed)
            w = sky.primary_sample(rng, model, g, k, m, n_tot, th, axis)[2]
            tot += w.sum()
        totals.append(tot)
    assert len(sky.build_tasks(30000, ["N"], 40.0, 3000, 5, nstrata=ns)) > 5
    for tot in totals:
        assert tot == pytest.approx(exact, rel=0.02)


def test_sky_run_resumes_from_task_files(tmp_path):
    """A job interrupted after some tasks resumes by re-running only the
    missing task files, and the merged total is identical (per-task seeds)."""
    import os
    import sky
    cache = os.environ.get("MC3D_POOL_CACHE")
    if not cache or not os.path.isdir(cache):
        pytest.skip("set MC3D_POOL_CACHE to a prebuilt pool directory")
    kw = dict(nshower=300, nproc=2, groups=["N"], theta_inj=40.0,
              caps=(5.0, 10.0), cache=cache, chunk=100, seed0=3,
              outdir=str(tmp_path), tag="rs", nstrata=2)
    os.chdir(tmp_path)
    fp = sky.run(**kw)
    ref = dict(np.load(fp))
    ntask = int(np.load(os.path.join(sky.task_dir(str(tmp_path), "rs"),
                                     "manifest.npz"))["ntask"][0])
    assert ntask >= 3
    os.remove(sky.task_path(str(tmp_path), "rs", 1))
    fp = sky.run(**kw)
    new = dict(np.load(fp))
    for key in ("s1", "t1", "cone_m", "hinj", "shcounts", "counts"):
        assert np.array_equal(ref[key], new[key]), key
    assert ref["shcounts"][0] == 300.0


def test_stratified_jackknife_matches_the_true_variance():
    """``analyse_sky.JackRun`` / ``jack``: for task sums drawn from two strata
    of unequal size, weight and spread, the delete-a-group jackknife variance
    of the total must reproduce the analytic stratified variance
    ``sum_k W_k^2 sigma_k^2 / N_k`` (averaged over realisations), and the
    per-stratum pieces must add up to it (in expectation: the cross-stratum
    term of one realisation vanishes only on average)."""
    import analyse_sky as A
    rng = np.random.default_rng(5)
    G, ntask, n_t = 20, (40, 70), 30
    mu, sd, wk = (1.0, 3.0), (2.0, 5.0), (1.0, 0.25)
    n_tot = [nt * n_t for nt in ntask]
    true_var = sum(wk[k] ** 2 * sd[k] ** 2 / n_tot[k] for k in range(2))
    v_all, v_parts = [], []
    for _ in range(300):
        J = object.__new__(A.JackRun)
        J.G, J.K = G, 2
        J.M = np.zeros((G, 2))
        S = np.zeros((G, 2))
        for k in range(2):
            for j in range(ntask[k]):
                x = rng.normal(mu[k], sd[k], n_t)
                S[j % G, k] += x.sum() * wk[k] / n_tot[k]
                J.M[j % G, k] += n_t
        J.S = {"x": S}
        est, sig, _ = A.jack(lambda x: x, [(J, "x")])
        v_all.append(float(sig) ** 2)
        v_parts.append(sum(float(A.jack(lambda x: x, [(J, "x")],
                                        stratum=k)[1]) ** 2 for k in range(2)))
        assert est == pytest.approx(S.sum())
    assert np.mean(v_all) == pytest.approx(true_var, rel=0.08)
    assert np.mean(v_parts) == pytest.approx(np.mean(v_all), rel=0.03)
