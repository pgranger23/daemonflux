"""Gates for :mod:`joint_cone` -- the joint production x cutoff cone integral.

Three gates, all on a small synthetic setup (no MCEq, < 10 s):

(a) with ``G == 1`` the joint factor must equal the off-axis excess produced by
    ``offaxis_mc`` on the same quadrature (and, if a cached real production
    profile is available, the delivered ``offaxis_excess.npz`` table);
(b) with the cone width -> 0 the joint factor must reduce to ``G_s`` at the
    axis cutoff;
(c) the covariance correction ``C_s`` -> 1 for a direction-independent cutoff
    map, -> 1 as the cone collapses (high energy), and is much smaller at the
    vertical than at the horizon.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import joint_cone as jc
import offaxis_mc as ox
from mceq3d_flux import SPECIES

N_ALPHA, N_BETA = 20, 12
E_TOY = np.array([0.3, 1.0, 10.0])


@pytest.fixture(scope="module")
def toy():
    """Exponential atmosphere + one-bump production + a toy cutoff map.

    Mirrors the fixture of ``test_offaxis_mc.py`` so the two suites agree on the
    geometry, and adds an E-W-asymmetric cutoff map and a monotone G_s(R_c).
    """
    H, rho0 = 7.0e5, 1.2e-3

    def rho(h):
        return rho0 * np.exp(-np.clip(np.asarray(h, float), 0, None) / H)

    ox._RHO = rho
    geom = ox.slant_depth_table(rho, n_h=40, n_psi=80, n_step=400)
    x_grid = np.linspace(5.0, 1000.0, 60)
    prof = np.exp(-((x_grid - 100.0) ** 2) / (2 * 60.0**2))
    p_tot = np.tile(prof[:, None], (1, len(E_TOY)))
    prod = dict(
        x_grid=x_grid,
        ep_grid=E_TOY,
        p_tot=p_tot,
        p_k=np.zeros_like(p_tot),
        geom=geom,
    )

    # cutoff map: rises with zenith and toward the East (azimuth 90 deg), like
    # a mid-latitude northern site; full sphere so the cone can cross the limb.
    zen = np.linspace(0.0, 180.0, 25)
    az = np.linspace(0.0, 360.0, 25)
    rc = (10.0 + 25.0 * np.sin(np.radians(zen))[:, None]
          * (0.5 + 0.5 * np.sin(np.radians(az))[None, :]))
    fine = (zen, az, rc)

    rc_grid = np.linspace(0.1, 55.0, 40)
    # a smooth, monotonically decreasing suppression, harder at low energy
    G = {}
    for i, s in enumerate(SPECIES):
        g = 1.0 / (1.0 + (rc_grid[:, None] / (8.0 * E_TOY[None, :] + 2.0)) ** 2)
        G[s] = np.clip(g * (1.0 - 0.03 * i), 0.0, 1.0)
    return prod, fine, rc_grid, G


def _channels(prod, sigma_deg):
    """Single-channel cone (no kaon term) at a fixed width, for the gates."""
    return [(np.full(len(prod["ep_grid"]), float(sigma_deg)), prod["p_tot"])]


# ---------------------------------------------------------------------------
# geometry mirrors
# ---------------------------------------------------------------------------
def test_rc_bilinear_reproduces_map_at_nodes(toy):
    zen, az, rc = toy[1]
    for iz in (0, 5, len(zen) - 1):
        for ja in (0, 7, len(az) - 1):
            got = jc.rc_bilinear(zen, az, rc, zen[iz], az[ja])
            assert np.isclose(got, rc[iz, ja], rtol=0, atol=1e-9)
    # azimuth is periodic, zenith is clamped (as in cone_geff's rc_at)
    assert np.isclose(jc.rc_bilinear(zen, az, rc, 30.0, 400.0),
                      jc.rc_bilinear(zen, az, rc, 30.0, 40.0))
    assert np.isclose(jc.rc_bilinear(zen, az, rc, -10.0, 12.0),
                      jc.rc_bilinear(zen, az, rc, 0.0, 12.0))


def test_cone_frame_equals_law_of_cosines():
    """The detector-frame 3-vector cone and offaxis_mc's law of cosines describe
    the SAME primary direction (the ``e1_prod == e1_det`` argument in the module
    docstring): the local zenith of ``n_p`` at any point of the arrival ray must
    equal ``arccos(cos a cos psi_o + sin a sin psi_o cos b)``."""
    alpha = np.deg2rad([0.0, 5.0, 20.0, 60.0])
    beta = np.linspace(0.0, 2 * np.pi, 7, endpoint=False)
    for cz in (0.05, 0.5, 0.95):
        for azd in (0.0, 90.0, 210.0):
            npv = jc.cone_directions(cz, azd, alpha, beta)
            th = np.arccos(np.clip(cz, -1, 1))
            phi = np.radians(azd)
            n = np.array([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi),
                          np.cos(th)])
            for L in (0.0, 100e5, 400e5):  # cm along the ray
                P = np.array([L * n[0], L * n[1], jc.R_EARTH_CM + L * n[2]])
                up = P / np.linalg.norm(P)
                psi_o = np.arccos(np.clip(float(n @ up), -1, 1))
                got = np.arccos(np.clip(npv @ up, -1, 1))
                want = np.arccos(np.clip(
                    np.cos(alpha)[:, None] * np.cos(psi_o)
                    + np.sin(alpha)[:, None] * np.sin(psi_o) * np.cos(beta)[None, :],
                    -1, 1))
                assert np.allclose(got, want, atol=1e-9)


# ---------------------------------------------------------------------------
# gate (a): G == 1  ->  the joint factor is exactly E_off
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cz", [0.05, 0.35, 0.95])
def test_gate_a_unit_G_reproduces_offaxis_mc(toy, cz):
    prod, fine, rc_grid, _ = toy
    G1 = {s: np.ones((len(rc_grid), len(E_TOY))) for s in SPECIES}
    sigma = 20.0
    r = jc.joint_cone(cz, 90.0, prod, fine, rc_grid, G1,
                      channels=_channels(prod, sigma),
                      n_alpha=N_ALPHA, n_beta=N_BETA)
    want = ox.e_off_for_zenith(
        cz, E_TOY, np.full(len(E_TOY), sigma), prod["x_grid"], E_TOY,
        prod["p_tot"], prod["geom"], N_ALPHA, N_BETA,
    )
    for s in SPECIES:
        assert np.allclose(r["joint"][s], want, rtol=1e-12, atol=0)
        assert np.allclose(r["C"][s], 1.0, rtol=1e-12, atol=0)
    assert np.allclose(r["E_off"], want, rtol=1e-12, atol=0)


PROD_CACHE = os.environ.get("JOINT_CONE_PROD_CACHE", "joint_cone_prod.npz")
TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "offaxis_excess.npz")


@pytest.mark.skipif(
    not (os.path.exists(PROD_CACHE) and os.path.exists(TABLE)),
    reason="needs a cached real production profile (JOINT_CONE_PROD_CACHE) "
           "and offaxis_excess.npz",
)
def test_gate_a_unit_G_reproduces_delivered_table():
    """With G == 1 the joint factor must equal the DELIVERED E_off table."""
    prod = jc.load_production(cache=PROD_CACHE)
    ep = prod["ep_grid"]
    rc_grid = np.linspace(0.1, jc.RC_MAX_GV, 40)
    G1 = {s: np.ones((len(rc_grid), len(ep))) for s in SPECIES}
    fine = jc.load_rc_map()
    d = np.load(TABLE)
    for cz in (0.05, 0.35, 0.95):
        r = jc.joint_cone(cz, 90.0, prod, fine, rc_grid, G1)
        i = int(np.argmin(abs(d["cz"] - cz)))
        for E in (0.3, 0.5, 1.0, 3.0):
            ref = np.interp(np.log(E), np.log(d["e"]), d["E_off"][i])
            got = np.interp(np.log(E), np.log(ep), r["joint"]["total_numu"])
            assert got == pytest.approx(ref, rel=2e-3)


# ---------------------------------------------------------------------------
# gate (b): cone -> 0  ->  the joint factor is G_s at the axis cutoff
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cz,azd", [(0.05, 90.0), (0.05, 270.0), (0.5, 0.0),
                                    (0.95, 180.0)])
def test_gate_b_collimated_limit_is_axis_G(toy, cz, azd):
    prod, fine, rc_grid, G = toy
    r = jc.joint_cone(cz, azd, prod, fine, rc_grid, G,
                      channels=_channels(prod, 20.0), sigma_scale=0.0,
                      n_alpha=N_ALPHA, n_beta=N_BETA)
    zen, az, rc = fine
    th = np.degrees(np.arccos(cz))
    rc_axis = jc.rc_bilinear(zen, az, rc, th, azd)
    assert r["rc_axis"] == pytest.approx(float(rc_axis), rel=1e-12)
    assert np.allclose(r["E_off"], 1.0, rtol=1e-12)
    for s in SPECIES:
        want = jc._interp_G([rc_axis], rc_grid, G[s])[0]
        assert np.allclose(r["joint"][s], want, rtol=1e-12, atol=0)
        assert np.allclose(r["G_cone"][s], want, rtol=1e-12, atol=0)
        assert np.allclose(r["C"][s], 1.0, rtol=1e-12, atol=0)


def test_gate_b_small_but_finite_cone_approaches_axis_G(toy):
    """A genuinely narrow (but non-degenerate) cone must approach the same
    limit through the ordinary quadrature branch, not the short circuit."""
    prod, fine, rc_grid, G = toy
    r = jc.joint_cone(0.05, 90.0, prod, fine, rc_grid, G,
                      channels=_channels(prod, 0.6), n_alpha=N_ALPHA,
                      n_beta=N_BETA)
    assert r["W"][:, 0].max() == 0.0  # the axis short-circuit was NOT used
    zen, az, rc = fine
    rc_axis = jc.rc_bilinear(zen, az, rc, np.degrees(np.arccos(0.05)), 90.0)
    for s in SPECIES:
        want = jc._interp_G([rc_axis], rc_grid, G[s])[0]
        assert np.allclose(r["joint"][s], want, rtol=3e-2)
        assert np.allclose(r["C"][s], 1.0, rtol=5e-3)


# ---------------------------------------------------------------------------
# gate (c): C -> 1 in the uncorrelated / collapsed-cone / vertical limits
# ---------------------------------------------------------------------------
def test_gate_c_constant_cutoff_map_gives_unit_C(toy):
    """A direction-independent R_c makes G constant over the cone, so the
    covariance vanishes identically at every direction and energy."""
    prod, fine, rc_grid, G = toy
    zen, az, _ = fine
    flat = (zen, az, np.full((len(zen), len(az)), 17.0))
    for cz in (0.05, 0.35, 0.95):
        r = jc.joint_cone(cz, 90.0, prod, flat, rc_grid, G,
                          channels=_channels(prod, 25.0),
                          n_alpha=N_ALPHA, n_beta=N_BETA)
        for s in SPECIES:
            assert np.allclose(r["C"][s], 1.0, rtol=1e-12, atol=0)


def test_gate_c_C_shrinks_with_the_cone(toy):
    """C -> 1 monotonically as the cone collapses (the high-energy limit)."""
    prod, fine, rc_grid, G = toy
    dev = []
    for scale in (1.0, 0.5, 0.25, 0.1):
        r = jc.joint_cone(0.05, 90.0, prod, fine, rc_grid, G,
                          channels=_channels(prod, 30.0), sigma_scale=scale,
                          n_alpha=N_ALPHA, n_beta=N_BETA)
        dev.append(float(np.max(np.abs(r["C"]["total_numu"] - 1.0))))
    assert dev[0] > dev[1] > dev[2] > dev[3]
    assert dev[-1] < 0.01


def test_gate_c_vertical_much_smaller_than_horizon(toy):
    """The covariance is a near-horizon effect: |C-1| at the vertical must be a
    small fraction of its horizon value (it is not identically zero, because
    both p and R_c fall off with the off-axis angle even at the vertical)."""
    prod, fine, rc_grid, G = toy
    ch = _channels(prod, 30.0)
    cv = jc.joint_cone(0.999, 90.0, prod, fine, rc_grid, G, channels=ch,
                       n_alpha=N_ALPHA, n_beta=N_BETA)["C"]["total_numu"]
    chz = jc.joint_cone(0.05, 90.0, prod, fine, rc_grid, G, channels=ch,
                        n_alpha=N_ALPHA, n_beta=N_BETA)["C"]["total_numu"]
    assert np.max(np.abs(cv - 1.0)) < 0.35 * np.max(np.abs(chz - 1.0))


def test_sublimb_weight_is_large_at_the_horizon_but_carries_no_production(toy):
    """The mechanism: a large fraction of the near-horizon cone points below the
    local limb -- ``cone_geff`` reads the far-side cutoff there, while
    ``offaxis_mc`` gives those directions p = 0."""
    prod, fine, rc_grid, G = toy
    r = jc.joint_cone(0.05, 90.0, prod, fine, rc_grid, G,
                      channels=_channels(prod, 35.0),
                      n_alpha=N_ALPHA, n_beta=N_BETA)
    assert r["w_sublimb"][0] > 0.2          # 0.3 GeV: wide cone crosses the limb
    assert r["f_sublimb"][0] < 0.5 * r["w_sublimb"][0]
    # near the vertical only the extreme 89-deg tail of the alpha grid can reach
    # below the limb, so the sub-limb weight is two orders of magnitude smaller
    v = jc.joint_cone(0.95, 90.0, prod, fine, rc_grid, G,
                      channels=_channels(prod, 35.0),
                      n_alpha=N_ALPHA, n_beta=N_BETA)
    assert v["w_sublimb"].max() < 0.02 * r["w_sublimb"][0]


# ---------------------------------------------------------------------------
# gates for the DELIVERED-path object (joint_cone.delivered_joint_factor,
# wired into MCEq3DFlux.joint_cone_factor / solve(joint_cone=True))
# ---------------------------------------------------------------------------
def _sig(prod, v):
    return np.full(len(prod["ep_grid"]), float(v))


def test_delivered_joint_unit_G_is_eoff(toy):
    """G == 1 -> F_s = E_off on the same quadrature, for every species."""
    prod, fine, rc_grid, _ = toy
    G1 = {s: np.ones((len(rc_grid), len(E_TOY))) for s in SPECIES}
    az = [0.0, 90.0, 270.0]
    r = jc.delivered_joint_factor(
        0.05, az, prod, fine, rc_grid, G1,
        sigma_pi=_sig(prod, 20.0), sigma_k=_sig(prod, 30.0),
        n_alpha=N_ALPHA, n_beta=N_BETA)
    # p_k is zero in the toy, so the composite reduces to the pion channel
    want = ox.e_off_for_zenith(
        0.05, E_TOY, _sig(prod, 20.0), prod["x_grid"], E_TOY, prod["p_tot"],
        prod["geom"], N_ALPHA, N_BETA)
    for s in SPECIES:
        for ia in range(len(az)):
            assert np.allclose(r["F"][s][ia], want, rtol=1e-12, atol=0)
    assert np.allclose(r["E_off_equiv"], want[None, :], rtol=1e-12, atol=0)


@pytest.mark.skipif(
    not (os.path.exists(PROD_CACHE) and os.path.exists(TABLE)),
    reason="needs a cached real production profile and offaxis_excess.npz",
)
def test_delivered_joint_unit_G_is_the_delivered_eoff_table():
    """The engine gate: with G == 1 (and no muon-decay channel) the joint factor
    must reproduce the DELIVERED offaxis_excess.npz table."""
    from kinematic_kernel import channel_shapes

    prod = jc.load_production(cache=PROD_CACHE)
    ep = prod["ep_grid"]
    sh = channel_shapes(ep)
    rc_grid = np.linspace(0.1, jc.RC_MAX_GV, 40)
    G1 = {s: np.ones((len(rc_grid), len(ep))) for s in SPECIES}
    fine = jc.load_rc_map(_new_map())
    d = np.load(TABLE)
    for cz in (0.05, 0.35, 0.95):
        r = jc.delivered_joint_factor(cz, [90.0], prod, fine, rc_grid, G1,
                                      sigma_pi=sh["pi"], sigma_k=sh["k"])
        i = int(np.argmin(abs(d["cz"] - cz)))
        for E in (0.3, 0.5, 1.0, 3.0):
            ref = np.interp(np.log(E), np.log(d["e"]), d["E_off"][i])
            got = np.interp(np.log(E), np.log(ep), r["F"]["total_numu"][0])
            assert got == pytest.approx(ref, rel=2e-3)


@pytest.mark.parametrize("cz,azd", [(0.05, 90.0), (0.05, 270.0), (0.95, 0.0)])
def test_delivered_joint_collimated_limit(toy, cz, azd):
    """Cone -> 0 -> F_s = G_s at the axis cutoff, read in the PRODUCTION-POINT
    frame (the frame ``cone_geff(sublimb='prod_point')`` uses)."""
    prod, fine, rc_grid, G = toy
    r = jc.delivered_joint_factor(
        cz, [azd], prod, fine, rc_grid, G,
        sigma_pi=_sig(prod, 20.0), sigma_k=_sig(prod, 30.0),
        sigma_mu=_sig(prod, 40.0),
        fmu={s: np.full(len(E_TOY), 0.4) for s in SPECIES},
        sigma_scale=0.0, n_alpha=N_ALPHA, n_beta=N_BETA)
    zen, az, rc = fine
    dn = zen <= 89.9
    frame = jc.prod_point_frame(cz, azd, 30.0)
    n = jc.cone_directions(cz, azd, np.array([0.0]), np.array([0.0]))[0, 0]
    th, ph = (jc.local_angles(n, frame) if frame is not None
              else (np.degrees(np.arccos(n[2])),
                    np.degrees(np.arctan2(n[1], n[0]))))
    rc_axis = jc.rc_bilinear(zen[dn], az, rc[dn], th, ph)
    assert np.allclose(r["E_off_equiv"], 1.0, rtol=1e-12)
    for s in SPECIES:
        want = jc._interp_G([rc_axis], rc_grid, G[s])[0]
        assert np.allclose(r["F"][s][0], want, rtol=1e-12, atol=0)


def test_delivered_joint_bending_splits_the_charges(toy):
    """The charge-signed muon-bending shift must move nu-from-mu+ and
    nu-from-mu- in OPPOSITE directions on an azimuthally-varying map."""
    prod, fine, rc_grid, G = toy
    mu_plus = ("total_antinumu", "total_nue")
    kw = dict(sigma_pi=_sig(prod, 20.0), sigma_k=_sig(prod, 30.0),
              sigma_mu=_sig(prod, 30.0),
              fmu={s: np.full(len(E_TOY), 1.0) for s in SPECIES},
              mu_plus=mu_plus, n_alpha=N_ALPHA, n_beta=N_BETA)
    d = np.array([0.0, 0.08, 0.0])  # ~4.6 deg shift toward the East
    a = jc.delivered_joint_factor(0.05, [90.0], prod, fine, rc_grid, G,
                                  d_cone=d, **kw)
    b = jc.delivered_joint_factor(0.05, [90.0], prod, fine, rc_grid, G,
                                  d_cone=None, **kw)
    dp = a["F"]["total_nue"][0] / b["F"]["total_nue"][0] - 1      # mu+
    dm = a["F"]["total_numu"][0] / b["F"]["total_numu"][0] - 1    # mu-
    assert np.all(dp * dm < 0)          # opposite sign at every energy
    assert np.max(np.abs(dp)) > 1e-3    # and not negligible


def _new_map():
    """The repaired full-sphere Kamioka map if present, else the paper map."""
    import glob

    for f in sorted(glob.glob(os.path.join(os.path.dirname(jc.PAPER_MAP),
                                           "finerc_*.npz"))):
        d = np.load(f)
        if len(d["zen"]) > 30:  # dense limb nodes -> the repaired map
            return f
    return jc.PAPER_MAP


# ---------------------------------------------------------------------------
# gates for the CHANNEL-RESOLVED joint factor (the delivered path since
# 2026-09-04): MCEq's depth-resolved {s}_dir / {s}_k / {s}_mu profiles carry the
# blend weights, each channel gets its own cone.
# ---------------------------------------------------------------------------
def _toy_channels(prod):
    """Split the toy profile into three parent channels with different depth
    shapes, so the implicit f_c really varies with depth and energy."""
    x = prod["x_grid"][:, None]
    p = prod["p_tot"]
    w_mu = 0.35 + 0.3 * np.tanh((x - 300.0) / 200.0)  # muon share grows with X
    w_k = np.full_like(w_mu, 0.08)
    return {s: (p * (1.0 - w_mu - w_k), p * w_k, p * w_mu) for s in SPECIES}


def test_channel_joint_sums_the_channels_exactly(toy):
    """G == 1: the channel-resolved factor must equal the production-weighted
    blend of the three per-channel E_off values on the same quadrature."""
    prod, fine, rc_grid, _ = toy
    G1 = {s: np.ones((len(rc_grid), len(E_TOY))) for s in SPECIES}
    chan = _toy_channels(prod)
    smu = {s: _sig(prod, 35.0) for s in SPECIES}
    r = jc.delivered_joint_factor(
        0.05, [90.0], prod, fine, rc_grid, G1, sigma_pi=_sig(prod, 20.0),
        sigma_k=_sig(prod, 30.0), channels_by_species=chan,
        sigma_mu_by_species=smu, n_alpha=N_ALPHA, n_beta=N_BETA)
    # independent recomputation: numerator sum over channels / denominator sum
    num = np.zeros(len(E_TOY))
    den = np.zeros(len(E_TOY))
    for p_ch, sg in zip(chan["total_numu"], (20.0, 30.0, 35.0)):
        n_, d_ = ox.cone_numden(0.05, E_TOY, _sig(prod, sg), prod["x_grid"],
                                E_TOY, p_ch, prod["geom"], N_ALPHA, N_BETA)
        num += n_
        den += d_
    assert np.allclose(r["F"]["total_numu"][0], num / den, rtol=1e-10, atol=0)


CHAN_TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "offaxis_excess_channel_v2.npz")


def _chan_prod_cache():
    """First existing channel-split production cache (engine or explicit)."""
    import offaxis_mc as _ox

    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.environ.get("JOINT_CONE_PROD_CHAN_CACHE", ""),
             os.path.join(here, ".cache3d", f"jointprod_chan_{_ox.TAG}.npz"),
             os.path.join(here, "flux_cache", f"jointprod_chan_{_ox.TAG}.npz")]
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


@pytest.mark.skipif(
    not (os.path.exists(CHAN_TABLE) and _chan_prod_cache()),
    reason="needs offaxis_excess_channel_v2.npz and a cached channel-split "
           "production profile (run diag_joint_stages.py once)",
)
def test_channel_joint_unit_G_is_the_channel_v2_table():
    """THE gate: with G == 1 the channel-resolved joint factor must reproduce
    ``offaxis_excess_channel_v2.npz`` -- the species-dependent E_off table built
    by ``offaxis_mc.build_channel`` from the same ingredients -- for every
    species and zenith."""
    import offaxis_mc as _ox

    prod = jc.load_production(cache=_chan_prod_cache(),
                              species=_ox.CHANNEL_SPECIES)
    ep = prod["ep_grid"]
    d = np.load(CHAN_TABLE)
    # the table's own widths: the cone MC is stochastic, the blend under test
    # is not -- this gate checks the composition, not the width generator
    w = {"pi": d["sigma_pi"], "k": d["sigma_k"],
         "mu_numu": d["sigma_mu_numu"], "mu_nue": d["sigma_mu_nue"]}
    chan = {s: prod["chan"][jc.CHANNEL_SPECIES_MAP[s]] for s in SPECIES}
    smu = {s: w["mu_nue" if "nue" in s else "mu_numu"] for s in SPECIES}
    rc_grid = np.linspace(0.1, jc.RC_MAX_GV, 40)
    G1 = {s: np.ones((len(rc_grid), len(ep))) for s in SPECIES}
    fine = jc.load_rc_map(_new_map())
    order = list(d["species"])
    for cz in (0.05, 0.35, 0.95):
        r = jc.delivered_joint_factor(cz, [90.0], prod, fine, rc_grid, G1,
                                      sigma_pi=w["pi"], sigma_k=w["k"],
                                      channels_by_species=chan,
                                      sigma_mu_by_species=smu)
        i = int(np.argmin(abs(d["cz"] - cz)))
        for s in SPECIES:
            k = order.index(jc.CHANNEL_SPECIES_MAP[s])
            assert np.allclose(r["F"][s][0], d["E_off_s"][k, i], rtol=1e-9,
                               atol=0), (cz, s)


@pytest.mark.parametrize("cz,azd", [(0.05, 90.0), (0.95, 0.0)])
def test_channel_joint_collimated_limit(toy, cz, azd):
    """Cone -> 0 -> G_s at the axis cutoff, channel mode (unchanged gate)."""
    prod, fine, rc_grid, G = toy
    r = jc.delivered_joint_factor(
        cz, [azd], prod, fine, rc_grid, G, sigma_pi=_sig(prod, 20.0),
        sigma_k=_sig(prod, 30.0), channels_by_species=_toy_channels(prod),
        sigma_mu_by_species={s: _sig(prod, 35.0) for s in SPECIES},
        sigma_scale=0.0, n_alpha=N_ALPHA, n_beta=N_BETA)
    zen, az, rc = fine
    dn = zen <= 89.9
    frame = jc.prod_point_frame(cz, azd, 30.0)
    n = jc.cone_directions(cz, azd, np.array([0.0]), np.array([0.0]))[0, 0]
    th, ph = (jc.local_angles(n, frame) if frame is not None
              else (np.degrees(np.arccos(n[2])),
                    np.degrees(np.arctan2(n[1], n[0]))))
    rc_axis = jc.rc_bilinear(zen[dn], az, rc[dn], th, ph)
    for s in SPECIES:
        want = jc._interp_G([rc_axis], rc_grid, G[s])[0]
        assert np.allclose(r["F"][s][0], want, rtol=1e-12, atol=0)


def test_channel_joint_bending_splits_the_charges(toy):
    """The charge-signed shift still splits the species in channel mode."""
    prod, fine, rc_grid, G = toy
    kw = dict(sigma_pi=_sig(prod, 20.0), sigma_k=_sig(prod, 30.0),
              channels_by_species=_toy_channels(prod),
              sigma_mu_by_species={s: _sig(prod, 35.0) for s in SPECIES},
              mu_plus=("total_antinumu", "total_nue"),
              n_alpha=N_ALPHA, n_beta=N_BETA)
    d = np.array([0.0, 0.08, 0.0])
    a = jc.delivered_joint_factor(0.05, [90.0], prod, fine, rc_grid, G,
                                  d_cone=d, **kw)
    b = jc.delivered_joint_factor(0.05, [90.0], prod, fine, rc_grid, G,
                                  d_cone=None, **kw)
    dp = a["F"]["total_nue"][0] / b["F"]["total_nue"][0] - 1
    dm = a["F"]["total_numu"][0] / b["F"]["total_numu"][0] - 1
    assert np.all(dp * dm < 0)
    assert np.max(np.abs(dp)) > 1e-3
