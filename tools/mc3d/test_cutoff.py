"""Fast tests for ``cutoff.py`` -- the geomagnetic cutoff at the injection point.

The numbers pinned here are measured (``MILESTONE3A_NOTES.md``); rigidities in
the binary-verdict tests sit well clear of the measured cutoffs so the tests do
not depend on penumbral (chaotic) trajectories.  Runtime ~1-2 min, dominated
by a handful of allowed traces (~600 RK4 steps each).
"""

from __future__ import annotations

import datetime

import numpy as np
import pytest

import cutoff as co
import geomag_backtrace as gb

LAT, LON = co.SITES["kamioka"]
#: short path budget for tests whose verdicts are "forbidden" anyway: stably
#: trapped dipole orbits would otherwise run to the full budget
SHORT = dict(max_s_m=1.0e8)


# ---------------------------------------------------------------------------
# Field
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model", ["IGRF13", "IGRF14"])
@pytest.mark.parametrize("date", [datetime.datetime(2020, 1, 1),
                                  datetime.datetime(2013, 7, 2)])
def test_fast_igrf_matches_ppigrf(model, date):
    """The cached evaluator is ppigrf's spherical-harmonic sum, to round-off."""
    from ppigrf import ppigrf as P
    rng = np.random.default_rng(1)
    n = 1500                           # crosses the packed/loop dispatch size
    r = 6371.2 * (1.0 + 4.0 * rng.random(n))
    th = 0.2 + 179.6 * rng.random(n)
    ph = 360.0 * rng.random(n) - 180.0
    ref = P.igrf_gc(r, th, ph, date, coeff_fn=co._shc_path(model))
    got = co.fast_igrf(model)(r, th, ph, date)
    bmag = np.sqrt(sum(x[0] ** 2 for x in ref))
    for g, x in zip(got, ref):
        assert g.shape == x.shape
        assert np.max(np.abs(g[0] - x[0]) / bmag) < 1e-13


def test_field_value_does_not_depend_on_the_batch():
    """The three internal evaluators (scalar <= 4 points, packed <= 1024,
    per-degree loop above) give bitwise-identical values, so a point's field
    (hence a trajectory's verdict) cannot depend on how many other
    trajectories are alive in the batch.  The scalar path relies on numpy's
    float64 sin/cos being libm's, which this test also pins."""
    f = co.fast_igrf("IGRF13")
    gs, hs = f.coefficients(co.DEFAULT_DATE)
    rng = np.random.default_rng(2)
    n = 2000
    r = 6371.2 * (1.0 + 3.0 * rng.random(n))
    th = 1.0 + 178.0 * rng.random(n)
    ph = 360.0 * rng.random(n) - 180.0
    a = f._packed(gs, hs, r, th, ph, 1, 13)
    b = f._loop(gs, hs, r, th, ph, 1, 13)
    c = f._scalar(co.DEFAULT_DATE, r[:200], th[:200], ph[:200], 1, 13)
    for x, y, z in zip(a, b, c):
        assert np.array_equal(x, y)
        assert np.array_equal(x[:200], z)
    big = f(r, th, ph, co.DEFAULT_DATE)                 # loop path
    for k in (1, 3, 600):                               # scalar, scalar, packed
        small = f(r[:k], th[:k], ph[:k], co.DEFAULT_DATE)
        for x, y in zip(small, big):
            assert np.array_equal(x[0], y[0, :k])


def test_igrf_field_context_selects_the_model_and_restores():
    import ppigrf
    from ppigrf import ppigrf as P
    orig = ppigrf.igrf_gc
    r, th, ph = np.array([6500.0]), np.array([50.0]), np.array([120.0])
    want13 = P.igrf_gc(r, th, ph, co.DEFAULT_DATE, coeff_fn=co._shc_path("IGRF13"))
    want14 = P.igrf_gc(r, th, ph, co.DEFAULT_DATE, coeff_fn=co._shc_path("IGRF14"))
    assert abs(want13[0][0, 0] - want14[0][0, 0]) > 0.1   # the models differ (nT)
    for fast in (True, False):
        with co.igrf_field("IGRF13", fast=fast):
            got = ppigrf.igrf_gc(r, th, ph, co.DEFAULT_DATE)
            assert ppigrf.igrf_gc is not orig
        assert ppigrf.igrf_gc is orig
        assert np.isclose(got[0][0, 0], want13[0][0, 0], rtol=1e-12, atol=0.0)
    # ppigrf's own default is IGRF-14 -- what geomag_backtrace gets untouched
    assert np.isclose(orig(r, th, ph, co.DEFAULT_DATE)[0][0, 0], want14[0][0, 0])


# ---------------------------------------------------------------------------
# Kinematics and geometry
# ---------------------------------------------------------------------------
def test_rigidity_superposition():
    p1 = np.sqrt(1.0 * (1.0 + 2.0 * co.M_P))           # T = 1 GeV per nucleon
    assert np.isclose(co.rigidity_gv(1.0), p1)
    assert np.isclose(co.rigidity_gv(1.0, 4, 2), 2.0 * p1)       # He: A/Z = 2
    assert np.isclose(co.rigidity_gv(1.0, 56, 26), 56 / 26 * p1)  # Fe
    assert np.isclose(co.rigidity_gv(1.0e3), 1.0e3 + co.M_P, rtol=1e-6)


def test_step_rule():
    """25 km below 5 GV, 100 km at 5-10, 200 km above (the measured
    convergence, MILESTONE3A_NOTES.md sec. 3)."""
    R = np.array([1.0, 1.7, 3.0, 4.9, 5.0, 7.0, 9.99, 10.0, 15.0, 60.0, 5e3])
    ds = co.step_m(R)
    assert np.allclose(ds / 1e3, [25, 25, 25, 25, 100, 100, 100, 200, 200, 200, 200])
    assert np.array_equal(co._step_class(ds), ds)        # already discrete


def test_copy_states_are_rotations_about_the_site_vertical():
    axis = co.site_axis(LAT, LON)
    r, u = co.state_above_site(LAT, LON, 60.0, 30.0)
    r = co.rotate_about(r, co.site_axis(LAT, LON + 90.0), np.radians(3.0))  # off-site
    rr, uu = co.copy_states(r, u, axis, 12)
    assert rr.shape == (1, 12, 3) and uu.shape == (1, 12, 3)
    assert np.allclose(rr[0, 0], r[0]) and np.allclose(uu[0, 0], u[0])
    # isometry: radius, distance to the site and local zenith are invariant
    rad = np.linalg.norm(rr[0], axis=1)
    assert np.allclose(rad, co.R_INJ_CM, rtol=1e-12)
    assert np.allclose(rr[0] @ axis / rad, (r[0] @ axis) / rad[0])
    mu = -np.einsum("ij,ij->i", uu[0], rr[0]) / rad
    assert np.allclose(mu, mu[0])
    # directly above the site: copy j arrives from compass azimuth a + 30 j
    r0, u0 = co.state_above_site(LAT, LON, 70.0, 40.0)
    rr, uu = co.copy_states(r0, u0, axis, 12)
    up, north, east = gb._local_frame(LAT, LON)
    d = -uu[0]
    az = np.degrees(np.arctan2(d @ east, d @ north)) % 360.0
    assert np.allclose(az, (40.0 + 30.0 * np.arange(12)) % 360.0, atol=1e-9)


# ---------------------------------------------------------------------------
# The re-entry rule
# ---------------------------------------------------------------------------
def test_reentry_rule_sets_the_absorbing_radius(monkeypatch):
    seen = []

    def fake(r0, u0, R, date, m_hat, charge=+1, **kw):
        seen.append((kw["r_floor"], np.linalg.norm(r0, axis=1).min(), kw["r_escape"]))
        return np.ones(len(r0), bool)

    monkeypatch.setattr(gb, "backtrace_vec", fake)
    r, u = co.state_above_site(LAT, LON, 30.0, 0.0)
    for floor, want in (("injection", None), ("sea_level", gb.RE),
                        (50.0, gb.RE + 5.0e4)):
        co.trace_allowed(r, u, 20.0, floor=floor)
        r_floor, r_launch, r_esc = seen[-1]
        if want is None:   # the launch sphere itself, 1 m slack
            assert np.isclose(r_floor, r_launch - 1.0) and r_launch > gb.RE + 9.9e4
        else:
            assert np.isclose(r_floor, want)
        assert r_esc == co.R_ESCAPE_RE == 10.0


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------
def test_batching_equivalence():
    """A state's verdict is independent of its batch: one batch, reversed
    order, singletons-in-pairs and two forked workers agree exactly."""
    rng = np.random.default_rng(3)
    zen = rng.uniform(0.0, 80.0, 16)
    az = rng.uniform(0.0, 360.0, 16)
    r, u = co.state_above_site(LAT, LON, zen, az)
    R = rng.uniform(6.0, 40.0, 16)                    # straddles the cutoffs
    a = co.trace_allowed(r, u, R, **SHORT)
    assert 0 < a.sum() < len(a)
    rev = co.trace_allowed(r[::-1], u[::-1], R[::-1], **SHORT)[::-1]
    pairs = np.concatenate([co.trace_allowed(r[i:i + 2], u[i:i + 2], R[i:i + 2], **SHORT)
                            for i in range(0, 16, 2)])
    par = co.admittance(r, u, R, n_jobs=2, **SHORT)
    assert np.array_equal(a, rev) and np.array_equal(a, pairs)
    assert np.array_equal(a, par)


def test_charge_mirror_symmetry_in_a_dipole():
    """Aligned centred dipole: reflecting y -> -y maps a charge-q trajectory
    onto a charge -q one, so the verdicts must agree state by state.  Pins
    the antiparticle reversal inside backtrace_vec (k = -charge c / R)."""
    rng = np.random.default_rng(4)
    r, u = co._random_sphere_states(rng, 24, co.R_INJ_CM)
    R = np.exp(rng.uniform(np.log(4.0), np.log(40.0), 24))
    P = np.array([1.0, -1.0, 1.0])
    kw = dict(m_hat=co.MZ, r_switch=0.0, **SHORT)
    a = co.trace_allowed(r, u, R, +1, **kw)
    b = co.trace_allowed(r * P, u * P, R, -1, **kw)
    assert 0 < a.sum() < len(a)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# The exact pre-filter
# ---------------------------------------------------------------------------
def test_stoermer_floor_values():
    """Most favourable (West-horizon) Stoermer cutoff: C/(1+sqrt2)^2 at the
    magnetic equator on the ground, 0 at the magnetic pole, r^-2 scaling."""
    m = co.MZ
    eq = co.stoermer_floor_gv(np.array([[co.R_EARTH_CM, 0.0, 0.0]]), m)[0]
    assert np.isclose(eq, co.C_ST_GV / (1.0 + np.sqrt(2.0)) ** 2)
    assert np.isclose(co.C_ST_GV, 58.64, atol=0.01)
    pole = co.stoermer_floor_gv(np.array([[0.0, 0.0, co.R_INJ_CM]]), m)[0]
    assert pole < 1e-12
    hi = co.stoermer_floor_gv(np.array([[co.R_INJ_CM, 0.0, 0.0]]), m)[0]
    assert np.isclose(hi / eq, (co.R_EARTH_CM / co.R_INJ_CM) ** 2)
    # the floor is the minimum of the Stoermer cutoff over directions
    rng = np.random.default_rng(5)
    r, u = co._random_sphere_states(rng, 200, co.R_INJ_CM)
    assert np.all(co.stoermer_gv(r, u, 1, m) >= co.stoermer_floor_gv(r, m) * (1 - 1e-12))


def test_gate_rules_and_physics():
    r, u = co.state_above_site(LAT, LON, [0.0, 45.0, 70.0, 85.0],
                               [0.0, 90.0, 270.0, 90.0])
    lo = co.stoermer_floor_gv(r)
    assert np.all((lo > 5.0) & (lo < 8.0))        # Kamioka: ~6.4 GV at 100 km
    R = np.array([0.95 * co.GATE_LO * lo[0], 1.05 * co.GATE_LO * lo[1], 61.0, 61.0])
    g = co.gate(r, u, R)
    mu = -np.einsum("ij,ij->i", u, r) / np.linalg.norm(r, axis=1)
    assert g[0] == -1 and g[1] == 0 and g[2] == 1
    assert g[3] == (1 if mu[3] > co.GATE_MU_MIN else 0)
    # what the gate rejects is indeed forbidden (deep below every cutoff here)
    rr, uu = co.state_above_site(LAT, LON, [0.0, 60.0, 85.0], [0.0, 270.0, 270.0])
    Rg = 0.95 * co.GATE_LO * co.stoermer_floor_gv(rr)
    assert np.all(co.gate(rr, uu, Rg) == -1)
    assert not co.trace_allowed(rr, uu, Rg, **SHORT).any()


# ---------------------------------------------------------------------------
# Pure centred dipole: Stoermer is exact for the main cone
# ---------------------------------------------------------------------------
DIP = dict(m_hat=co.MZ, r_switch=0.0)


def test_pure_dipole_vertical_cutoff_is_stoermer_at_the_equator():
    """Measured R_U = 14.207 GV vs Stoermer C/(4 rho^2) = 14.210 GV at the
    injection sphere (14.660 vs 14.659 on the ground; the 100 km launch is the
    Stoermer (R_E/R_inj)^2 = 0.9693 scaling)."""
    r, u = co.state_above_site(0.0, 0.0, 0.0, 0.0)
    st = co.stoermer_gv(r, u, 1, co.MZ)[0]
    assert np.isclose(st, co.C_ST_GV / 4.0 * (co.R_EARTH_CM / co.R_INJ_CM) ** 2)
    a = co.trace_allowed(np.repeat(r, 2, 0), np.repeat(u, 2, 0),
                         [0.98 * st, 1.02 * st], 1, **DIP)
    assert a.tolist() == [False, True]


def test_pure_dipole_nothing_below_stoermer_is_allowed():
    """Stoermer's theorem: below the cutoff of its own direction a state is
    bound.  Checked for R >= 2 GV (the MC's domain; below ~0.3 GV the 25 km
    minimum step exceeds the gyroradius and the integration is not resolved)."""
    rng = np.random.default_rng(6)
    r, u = co._random_sphere_states(rng, 400, co.R_INJ_CM, lat_max=50.0)
    st = co.stoermer_gv(r, u, 1, co.MZ)
    R = rng.uniform(0.6, 0.97, len(st)) * st
    keep = R >= 2.0
    a = co.trace_allowed(r[keep][:24], u[keep][:24], R[keep][:24], 1, **DIP, **SHORT)
    assert not a.any()


def test_east_west_ordering_and_charge_sign_in_a_dipole():
    """Lat 30, zenith 80: R_U East/West = 30.9/6.2 GV for positive charge
    and exactly mirrored for negative charge (measured)."""
    r, u = co.state_above_site(30.0, 0.0, [80.0, 80.0], [90.0, 270.0])
    pos = co.trace_allowed(r, u, 15.0, +1, **DIP)
    neg = co.trace_allowed(r, u, 15.0, -1, **DIP)
    assert pos.tolist() == [False, True]       # positive: cut harder from the East
    assert neg.tolist() == [True, False]


# ---------------------------------------------------------------------------
# Kamioka, full IGRF, injection point directly above the site
# ---------------------------------------------------------------------------
def test_kamioka_east_west_ordering_and_charge_sign():
    """Zenith 80 deg at the injection sphere: R_U East/West = 34.2/7.2 GV for
    positive primaries and 6.8/31.2 GV for negative ones (measured,
    ``cutoff.py kamioka`` V2/V2neg).  At 20 GV positive primaries are cut
    from the East and admitted from the West, negative ones the reverse."""
    r, u = co.state_above_site(LAT, LON, [80.0, 80.0], [90.0, 270.0])
    pos = co.trace_allowed(r, u, 20.0, +1)
    neg = co.trace_allowed(r, u, 20.0, -1)
    assert pos.tolist() == [False, True]
    assert neg.tolist() == [True, False]


def test_kamioka_vertical_cutoff_and_island():
    """Injection point directly above Kamioka, vertical: R_U = 11.10 GV on a
    0.05 GV ladder (11.16 after bisection; 11.37-11.53 at the ground, the
    Stoermer (R_E/R_inj)^2 = 0.969), with the allowed island [9.20, 9.75] GV
    below it -- the ground island [9.63, 10.13] of PHASE1_RESULTS scaled by the
    same factor.  Rigidities sit well inside each band."""
    r, u = co.state_above_site(LAT, LON, 0.0, 0.0)
    R = np.array([11.6, 10.6, 9.5, 8.6])
    a = co.trace_allowed(np.repeat(r, 4, 0), np.repeat(u, 4, 0), R)
    assert a.tolist() == [True, False, True, False]
