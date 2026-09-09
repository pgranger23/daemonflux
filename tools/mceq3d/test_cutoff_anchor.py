"""Tests for the production-point-anchored cutoff (``cutoff_anchor``).

The engine's cutoff map is anchored at the DETECTOR: every trajectory is
launched from ``RE * up`` at the site.  The primary that makes a near-horizon
neutrino enters the atmosphere at the production point P, up to 617 km away, so
``cutoff_anchor="prod_point"`` reads the cone samples' cutoff off a map built at
P, interpolated from the displaced-site family ``MCEq3DFlux.prod_family_rc``.

Three groups here:

* pure geometry (displacement, displaced site, the exact local frame at P) --
  fast, exact, no back-tracing;
* the site interpolation and the family's centre node -- a small, cheap
  back-traced family (``r_hi = 15 GV``, 2 zeniths x 4 azimuths);
* the delivered path -- that the anchor is a no-op at the vertical and that the
  family-interpolated cutoff agrees with a DIRECT back-trace launched at the
  displaced site.

The heavy full-resolution family (``.cache3d/prodfam_*.npz``) is used only if it
is already cached; those tests skip otherwise.
"""

import datetime
import os

import numpy as np
import pytest

import geomag_backtrace as gb
import joint_cone as jc
import mceq3d_flux as mf

LAT, LON = 36.43, 137.31
DATE = datetime.datetime(2020, 1, 1)


class _Bare(mf.MCEq3DFlux):
    """``MCEq3DFlux`` without the MCEq cascade: the cutoff machinery is pure
    geometry + back-tracing and needs none of it."""

    def __init__(self):
        pass


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def test_offset_vanishes_at_the_vertical():
    dn, de = mf.prod_point_offset_km(0.0, 137.0)
    assert abs(dn) < 1e-9 and abs(de) < 1e-9


def test_offset_grows_monotonically_toward_the_horizon():
    d = [np.hypot(*mf.prod_point_offset_km(z, 0.0)) for z in (0, 30, 60, 80, 87, 90)]
    assert all(b > a for a, b in zip(d, d[1:]))
    # h_prod = 30 km: 370 km up the ray at 87 deg -> 368 km of ground track
    assert 360.0 < d[-2] < 375.0
    assert 610.0 < d[-1] < 625.0  # the exact horizon


def test_offset_bearing_is_the_arrival_azimuth():
    # the neutrino comes FROM az, so its parent entered the atmosphere on that
    # side: the displacement points along the arrival azimuth.
    d = np.hypot(*mf.prod_point_offset_km(87.0, 0.0))
    for az, exp in ((0.0, (1, 0)), (90.0, (0, 1)), (180.0, (-1, 0)), (270.0, (0, -1))):
        dn, de = mf.prod_point_offset_km(87.0, az)
        assert np.allclose([dn, de], [d * exp[0], d * exp[1]], atol=1e-6)


def test_prod_site_latlon_is_a_great_circle_step():
    for dn, de in ((367.0, 0.0), (0.0, 367.0), (-200.0, 300.0)):
        la, lo = mf.prod_site_latlon(LAT, LON, dn, de)
        u0 = gb._local_frame(LAT, LON)[0]
        u1 = gb._local_frame(la, lo)[0]
        ang = np.degrees(np.arccos(np.clip(u0 @ u1, -1, 1)))
        assert np.isclose(ang * np.pi / 180.0 * mf.RE_KM, np.hypot(dn, de), rtol=1e-9)
    # a pure northward step stays on the meridian
    la, lo = mf.prod_site_latlon(LAT, LON, 367.0, 0.0)
    assert np.isclose(lo, LON, atol=1e-9)
    assert np.isclose(la, LAT + np.degrees(367.0 / mf.RE_KM), atol=1e-9)


def test_prod_frame_exact_is_the_identity_at_the_vertical():
    up_p, north_p = mf.prod_frame_exact(1.0, 137.0, mf.H_PROD_KM, LAT)
    assert np.allclose(up_p, [0.0, 0.0, 1.0], atol=1e-12)
    assert np.allclose(north_p, [1.0, 0.0, 0.0], atol=1e-12)


def test_prod_frame_exact_matches_the_true_local_frame_at_P():
    """The frame must be the one a map built AT P is indexed in: rotate a
    (zenith, azimuth) at P back to a 3-vector and compare with
    ``gb.arrival_direction`` at the displaced site."""
    up_d, north_d, east_d = gb._local_frame(LAT, LON)
    for zen, azd in ((87.0, 90.0), (80.0, 200.0), (60.0, 0.0)):
        cz = np.cos(np.radians(zen))
        frame = mf.prod_frame_exact(cz, azd, mf.H_PROD_KM, LAT)
        dn, de = mf.prod_point_offset_km(zen, azd)
        la, lo = mf.prod_site_latlon(LAT, LON, dn, de)
        up_p, north_p = frame
        east_p = np.cross(up_p, north_p)
        for th_s, ph_s in ((10.0, 33.0), (55.0, 210.0), (85.0, 300.0)):
            # the direction with these angles in P's frame, built in the
            # detector frame from `frame` ...
            t, p = np.radians(th_s), np.radians(ph_s)
            v_det = (np.cos(t) * up_p
                     + np.sin(t) * (np.cos(p) * north_p + np.sin(p) * east_p))
            v_ecef = v_det[0] * north_d + v_det[1] * east_d + v_det[2] * up_d
            # ... must equal the same angles resolved at the displaced site
            ref = -gb.arrival_direction(la, lo, th_s, ph_s)
            assert np.allclose(v_ecef, ref, atol=1e-9)
            # and the round trip through local_angles reproduces the angles
            th_b, ph_b = jc.local_angles(v_det, frame)
            assert np.isclose(th_b, th_s, atol=1e-9)
            assert np.isclose(ph_b % 360.0, ph_s % 360.0, atol=1e-9)


def test_meridian_convergence_is_what_the_legacy_frame_misses():
    """``joint_cone.prod_point_frame`` takes "north at P" to be the detector's
    north projected onto P's tangent plane, which ignores the convergence of the
    meridians -- ``dlon * sin(lat)``, up to ~4 deg for an east-west displacement.
    That is harmless for a detector-anchored map but must not index a map at P."""
    cz = np.cos(np.radians(89.5))
    for azd, expect_small in ((0.0, True), (180.0, True), (90.0, False),
                              (270.0, False)):
        up_a, north_a = jc.prod_point_frame(cz, azd, mf.H_PROD_KM)
        up_b, north_b = mf.prod_frame_exact(cz, azd, mf.H_PROD_KM, LAT)
        assert np.allclose(up_a, up_b, atol=1e-9)  # the vertical agrees exactly
        rot = np.degrees(np.arccos(np.clip(north_a @ north_b, -1, 1)))
        if expect_small:
            assert rot < 1e-6  # north/south: same meridian, no rotation
        else:
            dn, de = mf.prod_point_offset_km(89.5, azd)
            la, lo = mf.prod_site_latlon(LAT, LON, dn, de)
            exp = abs((lo - LON) * np.sin(np.radians(LAT)))
            assert 3.0 < rot < 5.0
            assert np.isclose(rot, exp, rtol=0.05)


# ---------------------------------------------------------------------------
# site interpolation
# ---------------------------------------------------------------------------
def _toy_family(n_side=3, D=600.0, nz=4, na=5):
    rng = np.random.default_rng(0)
    off = np.linspace(-D, D, n_side)
    zen = np.linspace(0.0, 89.5, nz)
    az = np.linspace(0.0, 360.0, na)
    rc = rng.uniform(5.0, 45.0, (n_side, n_side, nz, na))
    return dict(off=off, zen=zen, az=az, rc=rc)


def test_interp_site_rc_reproduces_every_node():
    fam = _toy_family()
    for i, dn in enumerate(fam["off"]):
        for j, de in enumerate(fam["off"]):
            assert np.allclose(mf.interp_site_rc(fam, dn, de), fam["rc"][i, j])


def test_interp_site_rc_is_bilinear_and_clamped():
    fam = _toy_family()
    D = fam["off"][-1]
    mid = mf.interp_site_rc(fam, D / 2.0, 0.0)
    assert np.allclose(mid, 0.5 * (fam["rc"][1, 1] + fam["rc"][2, 1]))
    # beyond the grid the site is clamped, never extrapolated
    assert np.allclose(mf.interp_site_rc(fam, 10 * D, 10 * D), fam["rc"][2, 2])


def test_interp_site_rc_at_zero_displacement_is_the_centre_map():
    fam = _toy_family(n_side=5)
    dn, de = mf.prod_point_offset_km(0.0, 217.0)
    assert np.allclose(mf.interp_site_rc(fam, dn, de), fam["rc"][2, 2])


# ---------------------------------------------------------------------------
# the back-traced family (cheap: r_hi = 15 GV, 2 x 4 grid)
# ---------------------------------------------------------------------------
CHEAP = dict(zeniths=[0.0, 40.0], azimuths=[0.0, 90.0, 180.0, 270.0],
             n_side=3, d_max_km=400.0, r_hi=15.0, n_jobs=8, warn_saturated=False)


@pytest.fixture(scope="module")
def cheap_family():
    return _Bare().prod_family_rc(LAT, LON, DATE, cache_dir=None, **CHEAP)


def test_family_shape_and_nodes(cheap_family):
    fam = cheap_family
    assert fam["rc"].shape == (3, 3, 2, 4)
    assert np.allclose(fam["off"], [-400.0, 0.0, 400.0])


def test_family_centre_node_is_the_ordinary_detector_map(cheap_family):
    """The centre of the family IS ``geomag_backtrace.cutoff_map`` at the site,
    which is what makes ``cutoff_anchor`` a pure displacement A/B."""
    ref = gb.cutoff_map(LAT, LON, DATE, CHEAP["zeniths"], CHEAP["azimuths"],
                        r_hi=CHEAP["r_hi"], n_jobs=8, warn_saturated=False)
    assert np.allclose(cheap_family["rc"][1, 1], ref, atol=1e-9)


def test_family_northward_site_has_a_lower_cutoff(cheap_family):
    """Physics gate on the sign of the site dependence.

    400 km north is 3.6 deg closer to the geomagnetic pole, so the Stoermer
    scaling ``R_c ~ cos^4 lambda_mag`` lowers the cutoff; 400 km south raises
    it.  The gate is on the VERTICAL (where the scaling is clean) and on the
    median over the map.

    It is deliberately NOT asserted cell by cell: measured on this family, the
    low-cutoff WEST lobe does not follow the latitude scaling (at zenith 40 deg
    west the southern site comes out at 9.35 GV against the detector's 9.47 GV,
    i.e. the wrong way).  There the cutoff is set by the shape of the shadow
    cone rather than by cos^4 lambda, so the pure-latitude argument does not
    apply and a cell-wise assertion would be asserting something false.
    """
    fam = cheap_family
    v = fam["rc"][:, 1, 0, 0]  # vertical cutoff, south -> centre -> north
    assert v[0] > v[1] > v[2]  # south > detector > north
    assert 0.02 < (v[1] - v[2]) / v[1] < 0.20
    unsat = fam["rc"][1, 1] < CHEAP["r_hi"] - 1e-9
    assert np.median(fam["rc"][2, 1][unsat]) < np.median(fam["rc"][1, 1][unsat])


def test_family_interpolation_matches_a_direct_back_trace(cheap_family):
    """The one approximation the scheme makes is bilinear interpolation of the
    map in the SITE offset.  Gate it against a direct back-trace at an
    intermediate displacement (the worst case: the cell centre)."""
    fam = cheap_family
    dn, de = 200.0, 200.0  # dead centre of a 400 km cell, in both axes
    got = mf.interp_site_rc(fam, dn, de)
    la, lo = mf.prod_site_latlon(LAT, LON, dn, de)
    ref = gb.cutoff_map(la, lo, DATE, CHEAP["zeniths"], CHEAP["azimuths"],
                        r_hi=CHEAP["r_hi"], n_jobs=8, warn_saturated=False)
    ok = ref < CHEAP["r_hi"] - 1e-9  # ignore cells clamped at the test ceiling
    assert ok.any()
    err = np.abs(got[ok] / ref[ok] - 1.0)
    # MEASURED, not aspirational: the site dependence of R_c is smooth enough
    # that the median cell interpolates to ~1%, but it is NOT smooth everywhere
    # -- one cell here (zenith 40 deg north) comes out 14% high at the cell
    # centre.  The same behaviour, at the same size, is quantified on the
    # delivered 3x3 / 617 km family in ``diag_cutoff_anchor.py`` section 3
    # (cardinal directions agree with a direct back-trace to <=0.4 GV, the worst
    # of 24 azimuths to 1.2 GV).  A 5x5 site grid would quarter it.
    assert np.median(err) < 0.03
    assert err.max() < 0.20


# ---------------------------------------------------------------------------
# the delivered path
# ---------------------------------------------------------------------------
def _fine_from_family(fam):
    return (np.asarray(fam["zen"], float), np.asarray(fam["az"], float),
            np.asarray(fam["rc"][1, 1], float))


def test_anchor_is_a_no_op_at_the_vertical(cheap_family):
    """Zero displacement -> the family interpolates to its own centre node and
    the exact frame is the identity, so the prod_point anchor cannot move the
    vertical."""
    fam = cheap_family
    zen_f, az_f, rc_f = _fine_from_family(fam)
    rc_site = mf.interp_site_rc(fam, *mf.prod_point_offset_km(0.0, 123.0))
    assert np.allclose(rc_site, rc_f)
    frame = mf.prod_frame_exact(1.0, 123.0, mf.H_PROD_KM, LAT)
    v = np.array([[0.3, 0.2, np.sqrt(1 - 0.13)]])
    th_a, ph_a = jc.local_angles(v, frame)
    th_b = np.degrees(np.arccos(v[..., 2]))
    ph_b = np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360.0
    assert np.allclose(th_a, th_b, atol=1e-9)
    assert np.allclose(ph_a, ph_b, atol=1e-9)


try:  # the production profile is a build product; skip if it is not cached
    import offaxis_mc as _ox

    _PROD_CACHE = os.path.join(".cache3d", f"jointprod_chan_{_ox.TAG}.npz")
    _HAVE_PROD = os.path.exists(_PROD_CACHE)
except Exception:  # pragma: no cover
    _PROD_CACHE, _HAVE_PROD = None, False


def _tiny_joint_inputs():
    """Minimal (prod, ep, rc_grid, G) tuple for ``delivered_joint_factor``."""
    prod = jc.load_production(cache=_PROD_CACHE, species=())
    ep = prod["ep_grid"]
    rc_grid = np.linspace(0.1, mf.RC_MAX_GV, 40)
    # a monotone, smooth stand-in for the cascade response: G falls with R_c
    G = {s: np.exp(-np.outer(rc_grid, np.ones(len(ep))) / 20.0) for s in mf.SPECIES}
    return prod, ep, rc_grid, G


@pytest.mark.skipif(not _HAVE_PROD, reason="joint production profile not cached")
def test_joint_factor_uniform_family_reproduces_the_detector_anchor(cheap_family):
    """Sanity gate on the plumbing: feed a family whose nine maps are all the
    same map.  Near the vertical the exact and approximate production frames
    coincide, so the prod_point anchor must then reproduce the detector anchor
    to machine precision."""
    fam = dict(cheap_family)
    fam["rc"] = np.repeat(np.repeat(fam["rc"][1:2, 1:2], 3, 0), 3, 1)
    prod, ep, rc_grid, G = _tiny_joint_inputs()
    fine = _fine_from_family(fam)
    kw = dict(sigma_pi=np.full(len(ep), 5.0), sigma_k=np.full(len(ep), 5.0),
              n_alpha=6, n_beta=6, n_ray=60)
    cz = 0.995
    a = jc.delivered_joint_factor(cz, [40.0], prod, fine, rc_grid, G, **kw)
    b = jc.delivered_joint_factor(cz, [40.0], prod, fine, rc_grid, G,
                                  rc_family=fam, site_lat=LAT, **kw)
    for s in mf.SPECIES:
        assert np.allclose(a["F"][s], b["F"][s], rtol=2e-4)


@pytest.mark.skipif(not _HAVE_PROD, reason="joint production profile not cached")
def test_joint_factor_prod_anchor_moves_the_horizon_the_right_way(cheap_family):
    """At the horizon the North production point sits closer to the pole (lower
    R_c -> more flux) and the South one closer to the equator (higher R_c ->
    less flux), so the anchor must RAISE the North bin and LOWER the South one."""
    prod, ep, rc_grid, G = _tiny_joint_inputs()
    fine = _fine_from_family(cheap_family)
    kw = dict(sigma_pi=np.full(len(ep), 5.0), sigma_k=np.full(len(ep), 5.0),
              n_alpha=6, n_beta=6, n_ray=60)
    cz = np.cos(np.radians(87.0))
    a = jc.delivered_joint_factor(cz, [0.0, 180.0], prod, fine, rc_grid, G, **kw)
    b = jc.delivered_joint_factor(cz, [0.0, 180.0], prod, fine, rc_grid, G,
                                  rc_family=cheap_family, site_lat=LAT, **kw)
    s = "total_numu"
    assert np.all(b["F"][s][0] > a["F"][s][0])  # North: cutoff drops
    assert np.all(b["F"][s][1] < a["F"][s][1])  # South: cutoff rises


# ---------------------------------------------------------------------------
# the delivered full-resolution family, if it is cached
# ---------------------------------------------------------------------------
def test_delivered_family_agrees_with_direct_prod_point_backtraces():
    """Full-resolution gate (skipped unless the family is already cached): the
    family-interpolated, frame-transformed cutoff at the production point of a
    near-horizon arrival direction must reproduce a DIRECT back-trace launched
    at that production point."""
    import os

    eng = _Bare()
    # only proceed if the disk cache exists -- never build it inside a test
    import hashlib

    d_max = float(np.hypot(*mf.prod_point_offset_km(90.0, 0.0, mf.H_PROD_KM)))
    key = (f"{LAT:.4f}_{LON:.4f}_{DATE.isoformat()}_13x25_rhi{mf.RC_MAX_GV:g}"
           f"_{gb.CUTOFF_SCHEME}_znlimb_{mf.PROD_FAMILY_SCHEME}_ns3"
           f"_D{d_max:.1f}_hl0_hp{mf.H_PROD_KM:g}__")
    h = hashlib.md5(f"prodfam_{key}".encode()).hexdigest()[:16]
    path = os.path.join(".cache3d", f"prodfam_{h}.npz")
    if not os.path.exists(path):
        pytest.skip("full-resolution prod-point family not cached")
    fam = eng.prod_family_rc(LAT, LON, DATE, cache_dir=".cache3d", n_jobs=1)

    zen, azd = 80.0, 90.0
    dn, de = mf.prod_point_offset_km(zen, azd)
    la, lo = mf.prod_site_latlon(LAT, LON, dn, de)
    rc_site = mf.interp_site_rc(fam, dn, de)
    probe_z = [20.0, 60.0, 85.0]
    probe_a = [0.0, 90.0, 180.0, 270.0]
    ref = gb.cutoff_map(la, lo, DATE, probe_z, probe_a, n_jobs=8,
                        warn_saturated=False)
    got = jc.rc_bilinear(
        np.asarray(fam["zen"], float), np.asarray(fam["az"], float), rc_site,
        np.array(probe_z)[:, None] * np.ones(len(probe_a))[None, :],
        np.ones(len(probe_z))[:, None] * np.array(probe_a)[None, :],
    )
    err = np.abs(got / ref - 1.0)
    # Same measured behaviour as the cheap family above: most cells interpolate
    # to well under 1%, the low-cutoff west lobe to ~20%.  The delivered impact
    # is bounded by ``diag_cutoff_anchor.py`` section 3, which does this
    # comparison at the arrival directions the engine actually uses.
    assert np.median(err) < 0.01
    assert err.max() < 0.25
