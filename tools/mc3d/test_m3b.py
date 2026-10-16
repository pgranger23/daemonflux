"""Milestone 3b: field grid, primary and muon bending, the vectorised Michel
decay, the per-copy machinery of ``sky3d`` and its regression against the
milestone-2b driver."""

import math
import os

import numpy as np
import pytest

import atmosphere as atm
import bend
import cutoff
import geometry as geo
from constants import M_MU, R_EARTH_CM

LAT, LON = 36.4267, 137.31
R_INJ = R_EARTH_CM + 100.0e5
TABLES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "mceq_tables_SIBYLL23D.npz")


@pytest.fixture(scope="module")
def grid(tmp_path_factory):
    import field
    cache = os.environ.get("MC3D_FIELD_CACHE") or str(
        tmp_path_factory.mktemp("field"))
    return field.FieldGrid(cache_dir=cache)


class _FakeRng:
    def __init__(self, v):
        self.v = v

    def random(self):
        return self.v


def _local_states(rng, n, h_lo=5e5, h_hi=40e5):
    up, no, ea = geo.local_frame(LAT, LON)
    h = rng.uniform(h_lo, h_hi, n)
    r = (R_EARTH_CM + h)[:, None] * up
    zen = np.radians(rng.uniform(0, 89, n))
    az = rng.uniform(0, 2 * np.pi, n)
    u = -(np.cos(zen)[:, None] * up + np.sin(zen)[:, None]
          * (np.cos(az)[:, None] * no + np.sin(az)[:, None] * ea))
    return r, u


# ---------------------------------------------------------------------------
# the field
# ---------------------------------------------------------------------------
def test_field_grid_matches_igrf(grid):
    import geomag_backtrace as gb
    rng = np.random.default_rng(1)
    n = 4000
    la = np.degrees(np.arcsin(rng.uniform(-0.95, 0.95, n)))
    lo = rng.uniform(-180, 180, n)
    h = rng.uniform(0, 125, n)
    r = (6371.0 + h) * 1e5
    x = np.stack([r * np.cos(np.radians(la)) * np.cos(np.radians(lo)),
                  r * np.cos(np.radians(la)) * np.sin(np.radians(lo)),
                  r * np.sin(np.radians(la))], 1)
    with cutoff.igrf_field("IGRF14"):
        bt = gb._bfield_igrf_cart(x * 1e-2, cutoff.DEFAULT_DATE, gb.dipole_axis())
    err = np.linalg.norm(grid(x) - bt, axis=1) / np.linalg.norm(bt, axis=1)
    assert err.max() < 6e-4 and np.median(err) < 2.5e-4
    assert grid.model == "IGRF14" == cutoff.DEFAULT_MODEL


def test_bend_rotation_sign():
    """du/ds = (q c/p) u x B: a mu+ moving +y in B = +z turns toward +x."""
    import field
    ax, ang = field.bend_rotation(np.array([[0.0, 1.0, 0.0]]),
                                  np.array([[0.0, 0.0, 5e-5]]), 1.0, 1.0, 1e5)
    u = field.rotate(np.array([[0.0, 1.0, 0.0]]), ax, ang)
    assert u[0, 0] > 0.0
    assert ang[0] == pytest.approx(-0.299792458 * 5e-5 * 1e3, rel=1e-12)


# ---------------------------------------------------------------------------
# the primary
# ---------------------------------------------------------------------------
def test_bend_primary_straight_is_advance_grammage():
    rng = np.random.default_rng(3)
    axis = np.array(geo.local_frame(LAT, LON)[0])
    r0, u0 = geo.sample_injection_patch(rng, R_INJ, axis, 40.0, 300)
    dX = -np.log(rng.random(300)) * 85.0
    st, r, u, L, b = bend.bend_primary(r0, u0, 10.0, dX, None)
    n_hit = 0
    for i in range(300):
        sb, _ = atm.path_to_exit(r0[i], u0[i])
        s, _, hit = atm.advance_grammage(r0[i], u0[i], dX[i], sb)
        assert (st[i] == bend.P_INTERACT) == hit
        if hit:
            n_hit += 1
            assert np.linalg.norm(r0[i] + s * u0[i] - r[i]) < 2.0e3   # 20 m
            assert b[i] < 1e-7               # acos rounding
    assert n_hit > 250


def test_bend_primary_uniform_field_turns_by_qcBL_over_R():
    """A uniform field perpendicular to the motion turns the direction by
    exactly ``Q_C B L / R`` over the arc ``L``, independent of the step."""
    rng = np.random.default_rng(4)
    axis = np.array(geo.local_frame(LAT, LON)[0])
    r0, u0 = geo.sample_injection_patch(rng, R_INJ, axis, 5.0, 200)
    Bv = 4e-5 * np.cross(axis, [0.0, 0.0, 1.0])
    Bv /= np.linalg.norm(Bv) / 4e-5

    def fld(r):
        return np.tile(Bv, (len(r), 1))

    R = np.exp(rng.uniform(np.log(3.0), np.log(100.0), 200))
    st, r, u, L, b = bend.bend_primary(r0, u0, R, np.full(200, 60.0), fld)
    ok = st == bend.P_INTERACT
    # the rotation is about B: the angle between the projections on the
    # plane perpendicular to B is Q_C |B| L / R
    bh = Bv / np.linalg.norm(Bv)
    p0 = u0 - np.outer(u0 @ bh, bh)
    p1 = u - np.outer(u @ bh, bh)
    cosang = np.einsum("ij,ij->i", p0, p1) / (np.linalg.norm(p0, axis=1)
                                             * np.linalg.norm(p1, axis=1))
    ang = np.arccos(np.clip(cosang, -1, 1))
    want = 0.299792458 * 4e-5 * L * 1e-2 / R
    assert ok.sum() > 150
    assert np.allclose(ang[ok], want[ok], rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------------------
# muons
# ---------------------------------------------------------------------------
def test_muon_transport_without_field_matches_shower_transport_muon():
    from shower import Config, transport_muon
    rng = np.random.default_rng(5)
    n = 300
    r, u = _local_states(rng, n)
    e = np.exp(rng.uniform(np.log(0.15), np.log(20.0), n))
    tt = -np.log(rng.random(n))
    pdg = np.where(rng.random(n) < 0.5, 13, -13)
    k, rv, uv, ev = bend.transport_muons(pdg, e, r, u, tt, None)
    cfg = Config(collinear=False, bfield=None)
    code = {"decay": 0, "ground": 1, "escape": 2, "stop": 3}
    for i in range(n):
        kind, rs, us, es = transport_muon(_FakeRng(math.exp(-tt[i])),
                                          int(pdg[i]), e[i], r[i], u[i], cfg)
        assert code[kind] == k[i]
        if k[i] == bend.MU_DECAY:
            assert np.linalg.norm(rs - rv[i]) < 1.0e4            # 100 m
            assert abs(es - ev[i]) / es < 5e-3


def test_muon_bend_is_5p35_deg_per_lifetime_and_energy_independent():
    """``0.3 B c tau / m`` = 5.354 deg per lifetime at 0.5 G, at every energy
    (``PHASE2_PLAN`` sec. 4.3), with the right-hand-rule sign."""
    rng = np.random.default_rng(6)
    m = 100
    rr = np.tile((R_EARTH_CM + 90e5) * np.array([1.0, 0, 0]), (m, 1))
    uu = np.tile([0, 1.0, 0], (m, 1))
    ee = np.exp(rng.uniform(np.log(0.5), np.log(50.0), m))
    tt = rng.uniform(0.05, 0.3, m)

    def fld(r):
        return np.tile([0.0, 0.0, 5e-5], (len(r), 1))

    for pdg, sgn in ((-13, +1.0), (13, -1.0)):
        k, _r, u2, _e = bend.transport_muons(np.full(m, pdg), ee, rr, uu, tt,
                                             fld, energy_loss=False)
        assert (k == bend.MU_DECAY).all()
        ang = np.degrees(np.arctan2(u2[:, 0], u2[:, 1]))
        want = np.degrees(0.299792458 * 5e-5 * bend.CTAU_CM[13] / 100.0 / M_MU)
        assert want == pytest.approx(5.3537, abs=1e-3)
        assert np.allclose(ang / tt, sgn * want, rtol=1e-6)


def test_michel_vectorised_matches_decays_michel_sample():
    import decays
    rng = np.random.default_rng(2)
    n, m = 100000, 6000
    for pdg, pol in ((-13, -0.8), (13, 0.6)):
        u = np.tile([0, 0, 1.0], (n, 1))
        pv, ev, dv = bend.michel_decay(np.full(n, pdg), np.full(n, 1.0), u,
                                       np.full(n, pol), rng.random((n, 6)),
                                       np.array([1.0, 0, 0]))
        p4 = np.array([1.0, 0, 0, math.sqrt(1 - M_MU ** 2)])
        ch = +1 if pdg < 0 else -1
        for j, kd in enumerate(("numu", "nue")):
            q = np.array([decays.michel_sample(rng, p4, ch, pol, kd)
                          for _ in range(m)])
            es, cs = q[:, 0], q[:, 3] / np.linalg.norm(q[:, 1:], axis=1)
            assert abs(ev[:, j].mean() - es.mean()) < 4 * es.std() / math.sqrt(m)
            assert abs(dv[:, j, 2].mean() - cs.mean()) < 4 * cs.std() / math.sqrt(m)
        assert set(pv[:, 0]) == {-14 if pdg < 0 else 14}
        assert set(pv[:, 1]) == {12 if pdg < 0 else -12}


def test_michel_frame_is_covariant_under_the_copy_rotations():
    """With the site vertical as the reference axis, rotating the muon about
    it rotates its decay neutrinos (common random numbers): the copies of a
    field-free decay are exact rotations of one another."""
    axis = cutoff.site_axis(LAT, LON)
    rng = np.random.default_rng(8)
    r, u = _local_states(rng, 50)
    rnd = rng.random((50, 6))
    p0, e0, d0 = bend.michel_decay(np.full(50, -13), np.full(50, 2.0), u,
                                   np.full(50, -0.5), rnd, axis)
    ur = cutoff.rotate_about(u, axis, -math.radians(77.0))
    p1, e1, d1 = bend.michel_decay(np.full(50, -13), np.full(50, 2.0), ur,
                                   np.full(50, -0.5), rnd, axis)
    assert np.allclose(e0, e1, rtol=1e-12)
    assert np.allclose(cutoff.rotate_about(d0, axis, -math.radians(77.0)), d1,
                       atol=1e-12)


def test_max_nu_angle():
    a = bend.max_nu_angle(np.array([0.12, 0.2, 1.0, 1e3]), 0.1)
    assert a[0] == 0.0 and 0.5 < a[1] < 0.6 and a[2] < a[1] and a[3] < 0.02


# ---------------------------------------------------------------------------
# sky3d machinery
# ---------------------------------------------------------------------------
def test_locate_vec_matches_nested_caps_and_soft_hits_is_exact_at_zero():
    import sky3d
    from scoring import NestedCaps
    caps = NestedCaps(LAT, LON, (2.5, 5.0, 7.5, 10.0))
    rng = np.random.default_rng(9)
    axis = np.array(caps.axis)
    r, u = geo.sample_injection_patch(rng, R_EARTH_CM + 20e5, axis, 25.0, 4000)
    i0, cz, az, cosi = sky3d.locate_vec(caps, r, u)
    soft = sky3d.soft_hits(caps, r, u, 0.0)
    n = 0
    for i in range(len(r)):
        loc = caps.locate(r[i], u[i])
        assert (loc is None) == (i0[i] < 0) == (not soft[i])
        if loc is not None:
            n += 1
            assert loc[0] == i0[i]
            assert loc[1] == pytest.approx(cz[i], abs=1e-12)
            d = (loc[2] - az[i] + 180) % 360 - 180
            assert abs(d) < 1e-9
    assert n > 100


def test_tally_squares_are_summed_per_primary():
    import sky
    import sky3d
    from scoring import NestedCaps
    caps = NestedCaps(LAT, LON, (10.0,))
    g = sky3d.Grid(caps, sky.E_BINS, sky.CZ_BINS, sky.AZ_BINS)
    t = sky3d.Tally(g)
    flat = np.array([5, 5, 5, 7])
    w = np.array([1.0, 2.0, 4.0, 1.0])
    prim = np.array([0, 0, 1, 1])
    t.accumulate("s", flat, w, w, prim)
    a = caps.areas[0]
    assert t.a["s0"][5] == pytest.approx(7.0 / a)
    assert t.a["s0sq"][5] == pytest.approx((9.0 + 16.0) / a ** 2)
    assert t.a["s0sq"][7] == pytest.approx(1.0 / a ** 2)


def _ctx(sky3d_mod, opts, fld=None):
    """Install a cheap (MCEq-yield) backend in both drivers' contexts."""
    import primaries as prim
    import sky
    from interactions import MCEqYieldBackend
    from scoring import NestedCaps
    from shower import Config
    caps = NestedCaps(LAT, LON, (2.5, 5.0, 7.5, 10.0))
    be = MCEqYieldBackend(TABLES)
    cfg = Config(collinear=False, bfield=None, energy_loss=True, e_nu_min=0.1,
                 decay_mode="kinematic", polarisation=True)
    model = prim.primary_model("H3A")
    sky._CTX.update(backend=be, cfg=cfg, caps=caps, model=model)
    sky3d_mod._CTX.update(backend=be, cfg=cfg, caps=caps, model=model,
                          opts=opts, field=fld, sitemap=None)


@pytest.mark.parametrize("naz", [1, 12])
def test_regression_reproduces_milestone_2b(tmp_path, naz):
    """Gate C1: cutoff off, bending off, B = 0 -> the new driver is the
    milestone-2b driver: the same neutrinos, so with one copy the tallies are
    equal bin by bin and with twelve the azimuth-summed ones are (the copies
    only redistribute azimuth); the in-run 1D and the 3D no-cutoff copy-0
    tallies equal 2b's; the cone tally is identical."""
    import sky
    import sky3d
    opts = sky3d.Options(cutoff=False, bend_primary=False, muon_field=False,
                         n_az=naz)
    _ctx(sky3d, opts)
    task = ("He", 2, 60, 40.0, 4321, 600)
    old = sky._shard(task)
    os.makedirs(sky3d.task_dir(str(tmp_path), "rg"))
    sky3d.cascade_task(task + (0, sky3d.cfile(str(tmp_path), "rg", 0)))
    sky3d.trace_replay_task(([0], str(tmp_path), "rg"))
    new = dict(np.load(sky3d.tfile(str(tmp_path), "rg", 0)))
    assert old["s1"].sum() > 0
    for a, b in (("s0", "s0"), ("s1", "s1"), ("t0", "t0"), ("t1", "t1"),
                 ("n1", "s1"), ("v1", "t1")):
        x, y = new[a], old[b]
        if naz > 1 or a == "n1":
            x, y = x.sum(-1), y.sum(-1)
        assert np.allclose(x, y, rtol=1e-10, atol=1e-14 * np.abs(y).max()), a
    assert np.array_equal(new["cone_m"], old["cone_m"])


def test_pending_test_loses_nothing(tmp_path, grid):
    """The pending test is conservative: with the primary and the muons bent
    (cutoff off), replaying every realisation (``pend_all``) gives the same
    tallies bit for bit as replaying only the pending ones."""
    import sky3d
    out = {}
    for pa in (False, True):
        opts = sky3d.Options(cutoff=False, bend_primary=True, muon_field=True,
                             r_thr=15.0, pend_all=pa)
        _ctx(sky3d, opts, grid)
        tag = f"pa{int(pa)}"
        os.makedirs(sky3d.task_dir(str(tmp_path), tag))
        sky3d.cascade_task(("p", 2, 40, 40.0, 99, 400, 0,
                            sky3d.cfile(str(tmp_path), tag, 0)))
        sky3d.trace_replay_task(([0], str(tmp_path), tag))
        out[pa] = dict(np.load(sky3d.tfile(str(tmp_path), tag, 0)))
    assert out[True]["s1"].sum() > 0
    for key in ("s1", "n1", "t1", "v1", "s0sq"):
        assert np.array_equal(out[False][key], out[True][key]), key


def test_copy_verdicts_are_charge_mirrored_in_a_dipole():
    """Gate C2 at the level of the per-copy weights: in an aligned dipole and
    for a site on the x-z plane, the verdict of copy ``j`` of a positive
    primary equals that of copy ``-j`` of the mirrored (``y -> -y``) primary
    with negative charge -- which pins the copy rotation sense against the
    charge reversal of the back-trace."""
    lat, lon = 30.0, 0.0
    axis = cutoff.site_axis(lat, lon)
    rng = np.random.default_rng(12)
    r0, u0 = geo.sample_injection_patch(rng, R_INJ, axis, 10.0, 40)
    R = rng.uniform(8.0, 30.0, 40)
    naz = 12
    rr, uu = cutoff.copy_states(r0, u0, axis, naz)
    mir = np.array([1.0, -1.0, 1.0])
    rm, um = cutoff.copy_states(r0 * mir, u0 * mir, axis, naz)
    kw = dict(r_switch=0.0, m_hat=np.array([0.0, 0.0, 1.0]))
    a_pos = cutoff.trace_allowed(rr.reshape(-1, 3), uu.reshape(-1, 3),
                                 np.repeat(R, naz), 1, **kw).reshape(-1, naz)
    a_neg = cutoff.trace_allowed(rm.reshape(-1, 3), um.reshape(-1, 3),
                                 np.repeat(R, naz), -1, **kw).reshape(-1, naz)
    perm = (-np.arange(naz)) % naz
    assert np.array_equal(a_pos, a_neg[:, perm])
    assert 0.1 < a_pos.mean() < 0.9          # a non-trivial mix of verdicts
