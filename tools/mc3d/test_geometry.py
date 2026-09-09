"""Geometry / atmosphere gates (PHASE2_PLAN.md sec. 2, gate G2)."""

import numpy as np
import pytest

import atmosphere as atm
import geometry as geo
from constants import R_EARTH_CM


# --------------------------------------------------------------------------
# atmosphere
# --------------------------------------------------------------------------
def test_density_matches_mceq():
    """Our analytic CORSIKA BK_USStd == MCEq's own, at every altitude."""
    pytest.importorskip("MCEq")
    import importlib.util  # noqa: F401
    from MCEq.geometry.density_profiles import CorsikaAtmosphere
    ca = CorsikaAtmosphere("BK_USStd", None)
    ca.set_theta(0.0)
    for h_km in (0, 0.5, 3, 6.9, 7.1, 11.3, 20, 36.9, 37.1, 60, 99, 105, 112):
        h = h_km * 1e5
        assert atm.density(h) == pytest.approx(ca.get_density(h), rel=1e-10)
        assert atm.vertical_depth(h) == pytest.approx(
            ca.get_mass_overburden(h), rel=1e-10)


def test_scalar_fast_path_matches_array():
    for h in (0.0, 1e5, 7e5, 1.14e6, 3.7e6, 9e6, 1.1e7, 1.2e7):
        assert atm.rho_s(h) == pytest.approx(float(atm.density(h)), rel=1e-12)
    for h in (0.0, 6e5, 1e6, 2e6, 5e6):
        assert atm.hscale_s(h) == pytest.approx(atm.scale_height(h))


def test_vertical_grammage():
    r0 = np.array([0.0, 0.0, R_EARTH_CM + atm.H_TOP_CM])
    u = np.array([0.0, 0.0, -1.0])
    x = atm.grammage(r0, u, atm.H_TOP_CM)
    assert x == pytest.approx(atm.X_GROUND, rel=2e-4)


def test_advance_grammage_is_the_inverse_of_grammage():
    r0 = np.array([0.0, 0.0, R_EARTH_CM + atm.H_TOP_CM])
    u = np.array([0.0, 0.0, -1.0])
    for dX in (1.0, 50.0, 300.0, 900.0):
        s, xd, hit = atm.advance_grammage(r0, u, dX, atm.H_TOP_CM)
        assert hit
        assert atm.grammage(r0, u, s) == pytest.approx(dX, rel=1e-3)


def test_slant_depth_matches_offaxis_mc():
    """Gate against tools/mceq3d's independent slant-depth table."""
    off = pytest.importorskip("offaxis_mc")
    pytest.importorskip("MCEq")
    import importlib.util  # noqa: F401
    from MCEq.geometry.density_profiles import CorsikaAtmosphere
    ca = CorsikaAtmosphere("BK_USStd", None)
    ca.set_theta(0.0)
    rho = off._rho_of_h(ca)
    h_grid, psi_grid, table = off.slant_depth_table(rho, n_h=60, n_psi=90,
                                                    n_step=900)
    # Compare at EXACT table nodes: offaxis_mc's lookup is bilinear on a
    # 2-deg psi grid, and interpolating across the convex near-limb rise is
    # itself a ~0.5% effect (the same interpolation error the Phase-1 audit
    # found in the cutoff map).  At the nodes the two are independent
    # quadratures of the same integral.
    for ih in (0, 10, 30):
        for ip in (0, 20, 50, 70):
            h = float(h_grid[ih])
            psi = float(psi_grid[ip])
            ref = float(table[ih, ip])
            if not np.isfinite(ref):
                continue
            r0 = (R_EARTH_CM + h) * np.array([0.0, 0.0, 1.0])
            u = np.array([np.sin(psi), 0.0, np.cos(psi)])
            s_top, hit_ground = atm.path_to_exit(r0, u)
            if hit_ground:
                continue
            ours = atm.grammage(r0, u, s_top)
            assert ours == pytest.approx(ref, rel=3e-3), \
                (h / 1e5, np.degrees(psi), ours, ref)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def test_particles_do_not_pass_through_the_earth():
    """Regression: path_to_exit must take the NEAR ground root only."""
    u = np.array([0.0, 0.0, -1.0])
    for h in (1e7, 1e5, 1e3, 1.0, 0.0):
        r0 = np.array([0.0, 0.0, R_EARTH_CM + h])
        s, ground = atm.path_to_exit(r0, u)
        assert ground
        assert s <= h + 2.0, (h, s)
    # upward-going from the surface must exit through the top, not the ground
    r0 = np.array([0.0, 0.0, R_EARTH_CM])
    s, ground = atm.path_to_exit(r0, np.array([0.0, 0.0, 1.0]))
    assert not ground
    assert s == pytest.approx(atm.H_TOP_CM, rel=1e-6)


def test_cap_area_and_solid_angle():
    cap = geo.DetectorCap(36.4, 137.3, 10.0)
    assert cap.area_cm2 == pytest.approx(
        2 * np.pi * R_EARTH_CM ** 2 * (1 - np.cos(np.deg2rad(10.0))))
    # Honda's cap radius is ~1117 km
    r_km = np.deg2rad(10.0) * 6371.0
    assert r_km == pytest.approx(1112.0, abs=10.0)


def test_cap_acceptance_is_the_cap():
    """A vertical neutrino above the site hits; one above a point 20 deg away
    does not; the boundary is exactly theta_D."""
    cap = geo.DetectorCap(36.4, 137.3, 10.0)
    for off_deg, expect in ((0.0, True), (5.0, True), (9.5, True),
                            (10.5, False), (20.0, False)):
        up, north, east = geo.local_frame(36.4 + off_deg, 137.3)
        r0 = (R_EARTH_CM + 2e6) * up
        hit, zen, az, ci = cap.score(r0, -up)
        assert hit is expect, (off_deg, hit)
        if hit:
            assert zen == pytest.approx(0.0, abs=1e-6)


def test_cap_solid_angle_monte_carlo():
    """The measured hit fraction of an isotropic downward flux equals the cap
    area fraction."""
    cap = geo.DetectorCap(0.0, 0.0, 10.0)
    rng = np.random.default_rng(4)
    n = 40000
    # vertical rays from a shell just above the ground, uniform over the sphere
    pos, _ = geo.sample_injection(rng, R_EARTH_CM + 1e6, n)
    hits = 0
    for p in pos:
        u = -p / np.linalg.norm(p)
        if cap.score(p, u)[0]:
            hits += 1
    frac = hits / n
    expect = 0.5 * (1 - np.cos(np.deg2rad(10.0)))
    assert frac == pytest.approx(expect, rel=0.12)


def test_injection_sampling_is_lambert_and_isotropic():
    rng = np.random.default_rng(1)
    pos, u = geo.sample_injection(rng, 1.0, 200000)
    assert np.allclose(np.linalg.norm(pos, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(u, axis=1), 1.0, atol=1e-12)
    mu = -np.sum(u * pos, axis=1)
    assert mu.min() > 0.0
    # p(mu) = 2 mu  ->  <mu> = 2/3, <mu^2> = 1/2
    assert mu.mean() == pytest.approx(2.0 / 3.0, rel=0.01)
    assert (mu ** 2).mean() == pytest.approx(0.5, rel=0.01)
    # positions isotropic
    assert np.abs(pos.mean(axis=0)).max() < 0.01


def test_arrival_direction_convention_matches_geomag_backtrace():
    gb = pytest.importorskip("geomag_backtrace")
    for zen, az in ((0, 0), (60, 0), (60, 90), (87, 270), (30, 180)):
        a = geo.arrival_direction(36.4, 137.3, zen, az)
        b = gb.arrival_direction(36.4, 137.3, zen, az)
        assert np.allclose(a, b, atol=1e-12)


def test_zenith_azimuth_round_trip():
    up, north, east = geo.local_frame(36.4, 137.3)
    for zen, az in ((0, 0), (30, 45), (87, 275), (60, 180)):
        u = geo.arrival_direction(36.4, 137.3, zen, az)
        z2, a2 = geo.zenith_azimuth(u, up, north, east)
        assert z2 == pytest.approx(zen, abs=1e-8)
        if zen > 0:
            assert a2 == pytest.approx(az % 360.0, abs=1e-8)
