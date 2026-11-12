"""Run mcup: the opt-in up-going (second-root) scorer of ``sky3d`` and
``scoring.NestedCaps``.

Geometry of the second root (exit point, cos psi, cap membership, the mirror
point ``Q`` at ``2 arcsin(c)``), its vectorised form against the scalar one,
the up/down symmetry of an isotropic field of straight lines, and the driver:
off = the previous outputs bit for bit, on = the down-going tallies unchanged,
and a conservative pending test for the up-going entries."""

import math
import os

import numpy as np
import pytest

import geometry as geo
from constants import R_EARTH_CM

LAT, LON = 36.4267, 137.31
CAPS = (2.5, 5.0, 7.5, 10.0)
TABLES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "mceq_tables_SIBYLL23D.npz")


def _caps():
    from scoring import NestedCaps
    return NestedCaps(LAT, LON, CAPS)


def _site_frame():
    up, no, ea = geo.local_frame(LAT, LON)
    return np.asarray(up, float), np.asarray(no, float), np.asarray(ea, float)


def _ground_point(dist_deg, az_deg):
    """Unit vector of the ground point at angular distance ``dist_deg`` from
    the site, in compass direction ``az_deg`` (N = 0, E = 90)."""
    s, n, e = _site_frame()
    d, a = math.radians(dist_deg), math.radians(az_deg)
    return math.cos(d) * s + math.sin(d) * (math.cos(a) * n + math.sin(a) * e)


def _chord_ray(q_hat, x_hat, back_km=50.0):
    """A ray that enters the ground sphere at ``q_hat`` and leaves it at
    ``x_hat`` (unit vectors): origin ``back_km`` before the entry point."""
    Q, X = R_EARTH_CM * q_hat, R_EARTH_CM * x_hat
    d = (X - Q) / np.linalg.norm(X - Q)
    return Q - back_km * 1.0e5 * d, d


# ---------------------------------------------------------------------------
# scalar geometry
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("c", [0.05, 0.15, 0.25, 0.5, 0.9])
@pytest.mark.parametrize("az", [0.0, 37.0, 90.0, 200.0, 315.0])
def test_ray_from_Q_arrives_up_going_with_the_same_cosine(c, az):
    """A ray entering the Earth at Q, 2 arcsin(c) from the site in compass
    direction ``az``, with the chord aimed at the site: its second root is
    the site, it arrives there up-going with cos Z = -c and azimuth ``az``
    (it comes from Q's side), inside the smallest cap, with cos psi = c; at
    Q (its first root) it was down-going with the same local cosine c."""
    from scoring import NestedCaps
    caps = _caps()
    q = _ground_point(math.degrees(2.0 * math.asin(c)), az)
    s = _site_frame()[0]
    r, u = _chord_ray(q, s)
    x = np.array(caps.crossing2(r, u))
    assert np.linalg.norm(x - R_EARTH_CM * s) < 1.0           # cm
    i0, cz, a, cosi = caps.locate2(r, u)
    assert i0 == 0
    assert cz == pytest.approx(-c, abs=1e-10)
    assert cosi == pytest.approx(c, abs=1e-10)
    assert ((a - az + 180.0) % 360.0) - 180.0 == pytest.approx(0.0, abs=1e-6)
    # first root = Q, down-going there with the same cosine
    h1 = np.array(caps.crossing(r, u))
    assert np.linalg.norm(h1 - R_EARTH_CM * q) < 1.0
    lat_q = math.degrees(math.asin(q[2]))
    lon_q = math.degrees(math.atan2(q[1], q[0]))
    loc_q = NestedCaps(lat_q, lon_q, (1.0,)).locate(r, u)
    assert loc_q[0] == 0 and loc_q[1] == pytest.approx(c, abs=1e-10)


def test_exit_point_cap_membership():
    """The smallest cap holding the exit point, for exit points at known
    angular distances from the site (chords of |cos| 0.1 from all sides)."""
    caps = _caps()
    want = {1.0: 0, 2.4: 0, 2.6: 1, 4.9: 1, 5.1: 2, 7.4: 2, 7.6: 3, 9.9: 3,
            10.1: None, 14.0: None}
    rng = np.random.default_rng(3)
    for dist, i0 in want.items():
        for _ in range(4):
            phi = rng.uniform(0, 360)
            x = _ground_point(dist, phi)
            # the entry point 2 arcsin(0.1) = 11.5 deg from x, any direction
            lat_x = math.degrees(math.asin(x[2]))
            lon_x = math.degrees(math.atan2(x[1], x[0]))
            up, no, ea = (np.asarray(v, float) for v in
                          geo.local_frame(lat_x, lon_x))
            a = math.radians(rng.uniform(0, 360))
            dq = 2.0 * math.asin(0.1)
            q = math.cos(dq) * up + math.sin(dq) * (math.cos(a) * no
                                                    + math.sin(a) * ea)
            r, u = _chord_ray(q, x)
            loc = caps.locate2(r, u)
            if i0 is None:
                assert loc is None
            else:
                assert loc is not None and loc[0] == i0, (dist, loc)
                assert loc[1] == pytest.approx(-0.1, abs=1e-9)


def test_no_second_root_without_an_entry():
    """Rays moving away from the sphere, missing it, or starting inside it
    have no second root (the inside origin's single root stays the first
    root's business)."""
    caps = _caps()
    s, n, e = _site_frame()
    r_up = (R_EARTH_CM + 20e5) * s
    assert caps.crossing2(r_up, s) is None                     # outward
    assert caps.crossing2(r_up, n) is None                     # misses
    r_in = (R_EARTH_CM - 1e5) * s
    d = -(0.9 * s + math.sqrt(1 - 0.81) * n)
    assert caps.crossing2(r_in, d) is None                     # inside
    assert caps.crossing(r_in, d) is not None
    import sky3d
    R = np.array([r_up, r_up, r_in])
    U = np.array([s, n, d])
    assert (sky3d.locate_vec2(caps, R, U)[0] == -1).all()


def test_chord_symmetry_and_separation_on_random_lines():
    """For every line through the Earth: |cos| at the entry = |cos| at the
    exit (each in its own local frame), and the two points are 2 arcsin|cos|
    apart; scalar and vectorised roots agree."""
    import sky3d
    caps = _caps()
    rng = np.random.default_rng(11)
    n = 3000
    r, u = geo.sample_injection_patch(rng, R_EARTH_CM + 60e5,
                                      np.array(caps.axis), 60.0, n)
    b = np.einsum("ij,ij->i", r, u)
    c = np.einsum("ij,ij->i", r, r) - R_EARTH_CM ** 2
    disc = b * b - c
    ok = disc > 0
    assert ok.sum() > 1000
    s1 = -b - np.sqrt(np.where(ok, disc, 0))
    s2 = -b + np.sqrt(np.where(ok, disc, 0))
    x1 = r + s1[:, None] * u
    x2 = r + s2[:, None] * u
    c1 = -np.einsum("ij,ij->i", u, x1) / np.linalg.norm(x1, axis=1)
    c2 = -np.einsum("ij,ij->i", u, x2) / np.linalg.norm(x2, axis=1)
    assert np.allclose(c1[ok], -c2[ok], atol=1e-9)
    sep = np.arccos(np.clip(np.einsum("ij,ij->i", x1, x2)
                            / (np.linalg.norm(x1, axis=1)
                               * np.linalg.norm(x2, axis=1)), -1, 1))
    assert np.allclose(sep[ok], 2 * np.arcsin(np.abs(c1[ok])), atol=1e-7)
    for i in np.where(ok)[0][:300]:
        h = caps.crossing2(r[i], u[i])
        assert np.linalg.norm(np.array(h) - x2[i]) < 1.0           # cm


def test_locate_vec2_matches_locate2_and_soft_hits_up():
    """Vectorised second root = scalar, and the exact second-root pending
    test agrees with it, on rays from a 35-deg patch aimed at random ground
    points near the site (a mix of first- and second-root crossings)."""
    import sky3d
    caps = _caps()
    rng = np.random.default_rng(5)
    n = 4000
    r, _u = geo.sample_injection_patch(rng, R_EARTH_CM + 20e5,
                                       np.array(caps.axis), 35.0, n)
    x = np.array([_ground_point(rng.uniform(0, 12), rng.uniform(0, 360))
                  for _ in range(n)]) * R_EARTH_CM
    u = (x - r) / np.linalg.norm(x - r, axis=1)[:, None]
    i0, cz, az, cosi = sky3d.locate_vec2(caps, r, u)
    soft = sky3d.soft_hits_up(caps, r, u)
    nh = 0
    for i in range(n):
        loc = caps.locate2(r[i], u[i])
        assert (loc is None) == (i0[i] < 0) == (not soft[i])
        if loc is not None:
            nh += 1
            assert loc[0] == i0[i]
            assert loc[1] == pytest.approx(cz[i], abs=1e-12) and cz[i] < 0
            assert loc[3] == pytest.approx(cosi[i], abs=1e-12)
            assert abs((loc[2] - az[i] + 180) % 360 - 180) < 1e-9
    assert nh > 300


def test_isotropic_field_of_lines_gives_U_equal_D():
    """An isotropic field of straight lines (positions uniform on a 55-deg
    patch at 20 km, inward Lambert directions truncated at cos 0.32, which
    covers every line that crosses the ground at |cos| <= 0.3) scored at the
    first root (down-going) and the second root (up-going) in the nested caps
    with the scorer's weights: U = D per |cos Z| row, in both measures."""
    import sky3d
    caps = _caps()
    rng = np.random.default_rng(2026)
    ax = np.array(caps.axis)
    t1, t2 = geo._tangent_basis(ax)
    rows = np.array([0.0, 0.1, 0.2, 0.3])
    acc = np.zeros((2, 2, 3, 2))          # (down/up, measure, row, [w, w^2])
    cmax = 0.32
    for _ in range(8):
        m = 250000
        cth = 1.0 - rng.random(m) * (1.0 - math.cos(math.radians(55.0)))
        ph = rng.uniform(0, 2 * np.pi, m)
        sth = np.sqrt(1.0 - cth * cth)
        nrm = (cth[:, None] * ax + sth[:, None] * (np.cos(ph)[:, None] * t1
                                                    + np.sin(ph)[:, None] * t2))
        r = (R_EARTH_CM + 20e5) * nrm
        mu = cmax * np.sqrt(rng.random(m))                       # Lambert
        a = rng.uniform(0, 2 * np.pi, m)
        e1 = np.cross(nrm, np.where(np.abs(nrm[:, 2:3]) < 0.9,
                                    [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]]))
        e1 /= np.linalg.norm(e1, axis=1)[:, None]
        e2 = np.cross(nrm, e1)
        st = np.sqrt(1.0 - mu * mu)
        u = -(mu[:, None] * nrm) + st[:, None] * (np.cos(a)[:, None] * e1
                                                  + np.sin(a)[:, None] * e2)
        for v, fn in enumerate((sky3d.locate_vec, sky3d.locate_vec2)):
            i0, cz, _az, cosi = fn(caps, r, u)
            hit = (i0 >= 0) & (i0 <= 3)
            acz = np.abs(cz)
            ir = np.searchsorted(rows, acz, side="right") - 1
            sel = hit & (ir >= 0) & (ir < 3)
            for meas, w in ((0, np.ones(m)), (1, 1.0 / np.maximum(cosi, 1e-6))):
                for k in range(3):
                    q = sel & (ir == k)
                    acc[v, meas, k, 0] += w[q].sum()
                    acc[v, meas, k, 1] += (w[q] ** 2).sum()
    for meas in (0, 1):
        for k in range(3):
            D, U = acc[0, meas, k, 0], acc[1, meas, k, 0]
            err = math.sqrt(acc[0, meas, k, 1] / D ** 2
                            + acc[1, meas, k, 1] / U ** 2)
            assert acc[0, 0, k, 0] > 2000
            assert abs(U / D - 1.0) < 4.0 * err, (meas, k, U / D, err)


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------
def _ctx(sky3d_mod, opts, fld=None):
    """Install the cheap (MCEq-yield) backend (as ``test_m3b._ctx``)."""
    import primaries as prim
    import sky
    from interactions import MCEqYieldBackend
    from scoring import NestedCaps
    from shower import Config
    caps = NestedCaps(LAT, LON, CAPS)
    be = MCEqYieldBackend(TABLES)
    cfg = Config(collinear=False, bfield=None, energy_loss=True, e_nu_min=0.1,
                 decay_mode="kinematic", polarisation=True)
    model = prim.primary_model("H3A")
    sky._CTX.update(backend=be, cfg=cfg, caps=caps, model=model)
    sky3d_mod._CTX.update(backend=be, cfg=cfg, caps=caps, model=model,
                          opts=opts, field=fld, sitemap=None)


def _job(tmp_path, tag, opts, task, fld=None):
    import sky3d
    _ctx(sky3d, opts, fld)
    os.makedirs(sky3d.task_dir(str(tmp_path), tag))
    sky3d.cascade_task(task + (0, sky3d.cfile(str(tmp_path), tag, 0)))
    sky3d.trace_replay_task(([0], str(tmp_path), tag))
    return dict(np.load(sky3d.tfile(str(tmp_path), tag, 0)))


def test_options_signature_unchanged_when_off():
    import sky3d
    a = sky3d.Options(cutoff=False, primary_energy="kinetic")
    b = sky3d.Options(cutoff=False, primary_energy="kinetic", upgoing=True)
    assert len(a.signature()) == 9
    assert np.array_equal(b.signature()[:-2], a.signature())


def test_upgoing_on_leaves_the_down_going_tallies_bit_identical(tmp_path):
    """Field free: the run with ``upgoing`` holds every array of the run
    without it, bit for bit, plus the up-going tallies, which are filled
    only at cos Z < 0 (and ``un`` = ``us`` without a cutoff)."""
    import sky3d
    task = ("p", 2, 120, 40.0, 4321, 1200)
    out = {}
    for up in (False, True):
        o = sky3d.Options(cutoff=False, bend_primary=False, muon_field=False,
                          upgoing=up)
        out[up] = _job(tmp_path, f"u{int(up)}", o, task)
    off, on = out[False], out[True]
    for k in off:
        if k in ("meta", "m3b_counts", "m3b_opts"):
            continue
        assert np.array_equal(off[k], on[k]), k
    for k in ("us0", "us1", "us0sq", "us1sq", "un0", "un1", "hinj_up"):
        assert k in on and k not in off
    us = on["us1"]
    assert us.sum() > 0
    assert us[:, :, :, 10:].sum() == 0.0                  # up-going rows only
    assert on["s1"][:, :, :, :10].sum() == 0.0            # and s never there
    assert on["un1"].sum() > 0 and on["un1"][:, :, :, 10:].sum() == 0.0
    # hinj_up holds the up-going weight of the largest cap, both measures, in
    # its E bins (0.1-3.98 GeV = sky.E_BINS 0-31), every smallest-cap class
    a10 = on["cap_area"][-1]
    h = on["hinj_up"]
    assert h.shape == (2, 4, 4, 8, 5, 60)
    for m in (0, 1):
        assert h[m].sum() == pytest.approx(on[f"us{m}"][-1][:, :32].sum() * a10,
                                           rel=1e-9)
    # the smallest-cap classes: the 2.5-deg cap holds class 0 only
    a25 = on["cap_area"][0]
    assert h[0, 0].sum() == pytest.approx(on["us0"][0][:, :32].sum() * a25,
                                          rel=1e-9)


@pytest.fixture(scope="module")
def grid(tmp_path_factory):
    import field
    cache = os.environ.get("MC3D_FIELD_CACHE") or str(
        tmp_path_factory.mktemp("field"))
    return field.FieldGrid(cache_dir=cache)


def test_upgoing_pending_test_loses_nothing(tmp_path, grid):
    """With the primary and the muons bent (cutoff off), replaying every
    realisation gives the same up-going tallies bit for bit as replaying only
    the pending ones: the second-root pending test (exact for rigid
    neutrinos, conservative for muons) drops nothing that scores."""
    import sky3d
    out = {}
    for pa in (False, True):
        o = sky3d.Options(cutoff=False, bend_primary=True, muon_field=True,
                          r_thr=15.0, pend_all=pa, upgoing=True)
        out[pa] = _job(tmp_path, f"pu{int(pa)}", o,
                       ("p", 2, 300, 40.0, 99, 3000), grid)
    assert out[True]["us1"].sum() > 0
    for key in ("us1", "un1", "us0sq", "s1", "n1", "t1", "hinj_up"):
        assert np.array_equal(out[False][key], out[True][key]), key


def test_cone_may_hit_cap_up_is_conservative():
    """Directions sampled inside each test cone: whenever one of them leaves
    the Earth inside the largest cap, the cone test says it may."""
    import sky3d
    caps = _caps()
    rng = np.random.default_rng(17)
    n = 600
    P, _u = geo.sample_injection_patch(rng, R_EARTH_CM + 25e5,
                                       np.array(caps.axis), 30.0, n)
    # cone axes aimed roughly through the Earth toward the site region
    x = np.array([_ground_point(rng.uniform(0, 14), rng.uniform(0, 360))
                  for _ in range(n)]) * R_EARTH_CM
    ax_ = (x - P) / np.linalg.norm(x - P, axis=1)[:, None]
    alpha = rng.uniform(0.01, 0.3, n)
    may = sky3d.cone_may_hit_cap_up(caps, P, ax_, alpha)
    n_true = 0
    for i in range(n):
        t1, t2 = geo._tangent_basis(ax_[i])
        k = 200
        ca = 1.0 - rng.random(k) * (1.0 - math.cos(alpha[i]))
        ph = rng.uniform(0, 2 * np.pi, k)
        sa = np.sqrt(1 - ca * ca)
        d = (ca[:, None] * ax_[i] + sa[:, None] * (np.cos(ph)[:, None] * t1
                                                    + np.sin(ph)[:, None] * t2))
        hit = (sky3d.locate_vec2(caps, np.tile(P[i], (k, 1)), d)[0] >= 0).any()
        if hit:
            n_true += 1
            assert may[i], i
    assert n_true > 50
