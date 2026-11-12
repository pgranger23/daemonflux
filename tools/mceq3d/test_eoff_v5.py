"""Gates for :mod:`eoff_v5` (E_off v5: local-density energy-loss kernels in the cone integral).

All on a small synthetic setup (exponential atmosphere, toy profiles, toy cutoff map; no MCEq, a few seconds):

(a) the density interpolation weights are a partition of unity, exact at the nodes, linear in ln rho, clamped;
(b) the density-split cone production sums to :func:`joint_cone.cone_production_multi` (v4's q) bit for bit, with
    and without zenith-node profiles, and each part lives where the ray's density is;
(c) :func:`eoff_v5.joint_factor_v5` with the same kernel at every density node is
    :func:`joint_cone.delivered_joint_factor` (v4's joint integral), with the cutoff, the production-point frame
    and the muon-bend shift;
(d) a kernel that narrows with density lowers the near-horizon factor more than the vertical one, the physical
    content of v5 (denser air at the off-axis production points of a near-horizontal arrival);
(e) plumbing: eoff_model validation and the v5 table guard.
"""

from __future__ import annotations

import numpy as np
import pytest

import eoff_v5 as v5
import joint_cone as jc
import offaxis_mc as ox
from mceq3d_flux import SPECIES

E_TOY = np.array([0.15, 0.3, 1.0])
NODES = (0.3e-4, 0.6e-4, 1.2e-4, 2.5e-4, 5.0e-4, 1.0e-3)


@pytest.fixture(scope="module")
def toy():
    H, rho0 = 7.0e5, 1.2e-3

    def rho(h):
        return rho0 * np.exp(-np.clip(np.asarray(h, float), 0, None) / H)

    ox._RHO = rho
    geom = ox.slant_depth_table(rho, n_h=40, n_psi=80, n_step=400)
    x_grid = np.linspace(5.0, 1000.0, 60)
    prof = np.exp(-((x_grid - 100.0) ** 2) / (2 * 60.0 ** 2))
    p = np.tile(prof[:, None], (1, len(E_TOY)))
    w_mu = 0.35 + 0.3 * np.tanh((x_grid[:, None] - 300.0) / 200.0)
    chan = {s: (p * (1.0 - w_mu - 0.08), p * 0.08, p * w_mu) for s in SPECIES}
    prod = dict(x_grid=x_grid, ep_grid=E_TOY, p_tot=p, p_k=0.08 * p, geom=geom, chan=chan)
    # two zenith nodes with different depth shapes (zenith-node path of v4)
    prof2 = np.exp(-((x_grid - 250.0) ** 2) / (2 * 120.0 ** 2))
    p2 = np.tile(prof2[:, None], (1, len(E_TOY)))
    zn = dict(cg=np.array([1.0, 0.05]), x=[x_grid, x_grid],
              chan=[chan, {s: (p2 * 0.5, p2 * 0.08, p2 * 0.42) for s in SPECIES}])
    zen = np.linspace(0.0, 180.0, 25)
    az = np.linspace(0.0, 360.0, 25)
    rc = 10.0 + 25.0 * np.sin(np.radians(zen))[:, None] * (0.5 + 0.5 * np.sin(np.radians(az))[None, :])
    rc_grid = np.linspace(0.1, 55.0, 40)
    G = {s: np.clip(1.0 / (1.0 + (rc_grid[:, None] / (8.0 * E_TOY[None, :] + 2.0)) ** 2) * (1 - 0.03 * i), 0, 1)
         for i, s in enumerate(SPECIES)}
    alpha = np.concatenate([[0.0], np.deg2rad(np.linspace(1.0, 150.0, 24))])
    return prod, zn, (zen, az, rc), rc_grid, G, alpha, rho


def _kernel(alpha, sigma_deg):
    a = alpha[1:]
    w = np.sin(a) * np.exp(-np.degrees(a) ** 2 / (2 * sigma_deg ** 2))
    out = np.zeros((len(E_TOY), len(alpha)))
    out[:, 1:] = w / w.sum()
    return out


# (a) ------------------------------------------------------------------------
def test_rho_weights_partition_nodes_and_clamp():
    rho = np.array([1e-6, 0.3e-4, 0.45e-4, 1.2e-4, 7e-4, 1.0e-3, 5e-3])
    w = v5.rho_weights(rho, NODES)
    assert w.shape == (len(NODES), len(rho))
    assert np.allclose(w.sum(0), 1.0, rtol=0, atol=1e-14)
    assert (w >= 0).all()
    assert w[0, 0] == 1.0 and w[-1, -1] == 1.0                      # clamped to the end members
    assert w[0, 1] == pytest.approx(1.0) and w[2, 3] == pytest.approx(1.0)   # exact at the nodes
    t = np.log(0.45e-4 / 0.3e-4) / np.log(0.6e-4 / 0.3e-4)          # linear in ln rho
    assert w[0, 2] == pytest.approx(1 - t) and w[1, 2] == pytest.approx(t)


def test_rho_nodes_cover_the_v4_family():
    import eoff_v4 as v4

    nodes = v5.rho_nodes()
    assert list(nodes) == sorted(nodes)
    assert set(v4.RHO_FAMILY) <= set(nodes) and max(nodes) >= 1.0e-3


# (b) ------------------------------------------------------------------------
@pytest.mark.parametrize("cz", [0.05, 0.5, 1.0])
@pytest.mark.parametrize("use_zn", [False, True])
def test_split_production_sums_to_v4(toy, cz, use_zn):
    prod, zn, _, _, _, alpha, _ = toy
    beta = np.linspace(0.0, 2 * np.pi, 12, endpoint=False)
    plist = [prod["chan"]["total_numu"][c] for c in range(3)]
    znodes = None
    if use_zn:
        znodes = dict(cg=zn["cg"], x=zn["x"], profiles=[[ch["total_numu"][c] for c in range(3)] for ch in zn["chan"]])
    q4, qa4 = jc.cone_production_multi(cz, alpha, beta, prod["x_grid"], E_TOY, plist, prod["geom"], n_ray=120,
                                       znodes=znodes)
    q5, qa5 = v5.cone_production_split(cz, alpha, beta, prod["x_grid"], E_TOY, plist, prod["geom"], NODES,
                                       n_ray=120, znodes=znodes)
    for i in range(3):
        assert q5[i].shape == (len(NODES),) + q4[i].shape
        assert np.allclose(q5[i].sum(0), q4[i], rtol=1e-12, atol=1e-300)
        assert np.allclose(qa5[i].sum(0), qa4[i], rtol=1e-12, atol=1e-300)


def test_split_production_follows_the_ray_density(toy):
    """The on-axis part at node j comes from the altitudes whose density is near rho_j: for the exponential toy
    atmosphere the density-weighted mean of the parts must increase monotonically with the node density."""
    prod, _, _, _, _, alpha, rho = toy
    beta = np.linspace(0.0, 2 * np.pi, 6, endpoint=False)
    _, qa = v5.cone_production_split(0.5, alpha[:2], beta, prod["x_grid"], E_TOY, [prod["p_tot"]], prod["geom"],
                                     NODES, n_ray=200)
    share = qa[0][:, 0] / qa[0][:, 0].sum()
    assert share.sum() == pytest.approx(1.0)
    assert (share > 0).sum() >= 4                    # the production spans several density nodes
    # a ray at the horizon samples thinner air on the axis than a vertical one (more weight on low-density nodes)
    _, qh = v5.cone_production_split(0.05, alpha[:2], beta, prod["x_grid"], E_TOY, [prod["p_tot"]], prod["geom"],
                                     NODES, n_ray=200)
    sh = qh[0][:, 0] / qh[0][:, 0].sum()
    lr = np.log(NODES)
    assert (sh * lr).sum() < (share * lr).sum()


# (c) ------------------------------------------------------------------------
@pytest.mark.parametrize("cz,azd", [(0.05, 90.0), (0.15, 270.0), (0.6, 0.0)])
@pytest.mark.parametrize("bend", [False, True])
def test_joint_v5_with_density_independent_kernel_is_v4(toy, cz, azd, bend):
    prod, zn, fine, rc_grid, G, alpha, _ = toy
    W = {s: (_kernel(alpha, 25.0), _kernel(alpha, 30.0), _kernel(alpha, 45.0)) for s in SPECIES}
    Wj = {s: tuple(np.repeat(w[None], len(NODES), 0) for w in W[s]) for s in SPECIES}
    d_cone = np.array([0.02, -0.01, 0.0]) if bend else None
    mu_plus = ("total_antinumu", "total_nue")
    r4 = jc.delivered_joint_factor(cz, [azd], prod, fine, rc_grid, G, sigma_pi=None, sigma_k=None,
                                   channels_by_species=prod["chan"], W_by_species=W, alpha_rad=alpha, d_cone=d_cone,
                                   mu_plus=mu_plus, n_beta=12, n_ray=120, znodes=zn)
    r5 = v5.joint_factor_v5(cz, [azd], prod, fine, rc_grid, G, channels_by_species=prod["chan"], Wj_by_species=Wj,
                            alpha_rad=alpha, nodes=NODES, d_cone=d_cone, mu_plus=mu_plus, n_beta=12, n_ray=120,
                            znodes=zn)
    for s in SPECIES:
        assert np.allclose(r5["J"][s], r4["J"][s], rtol=1e-12, atol=0)
        assert np.allclose(r5["p_axis_s"][s], r4["p_axis_s"][s], rtol=1e-12, atol=0)


def test_joint_v5_q_cache_is_reused(toy):
    prod, zn, fine, rc_grid, G, alpha, _ = toy
    W = {s: tuple(np.repeat(_kernel(alpha, 30.0)[None], len(NODES), 0) for _ in range(3)) for s in SPECIES}
    qc = {}
    a = v5.joint_factor_v5(0.05, [0.0, 90.0], prod, fine, rc_grid, G, channels_by_species=prod["chan"],
                           Wj_by_species=W, alpha_rad=alpha, nodes=NODES, n_beta=12, n_ray=120, q_cache=qc, znodes=zn)
    assert len(qc) == 1
    b = v5.joint_factor_v5(0.05, [90.0], prod, fine, rc_grid, G, channels_by_species=prod["chan"],
                           Wj_by_species=W, alpha_rad=alpha, nodes=NODES, n_beta=12, n_ray=120, q_cache=qc, znodes=zn)
    assert len(qc) == 1
    assert np.array_equal(a["J"]["total_numu"][1], b["J"]["total_numu"][0])


# (d) ------------------------------------------------------------------------
def test_denser_air_narrower_kernel_lowers_the_horizon_more(toy):
    """mu-channel kernel narrowing with density (as the energy-loss family does): relative to the kernel of the
    on-axis decay density, the local-density factor drops more at the horizon than at the vertical."""
    prod, zn, fine, rc_grid, G, alpha, rho = toy
    G1 = {s: np.ones_like(G[s]) for s in SPECIES}
    widths = np.array([60.0, 55.0, 50.0, 45.0, 40.0, 35.0])

    def ratio(cz):
        Wloc = {s: (np.repeat(_kernel(alpha, 25.0)[None], 6, 0), np.repeat(_kernel(alpha, 30.0)[None], 6, 0),
                    np.stack([_kernel(alpha, w) for w in widths])) for s in SPECIES}
        r = v5.joint_factor_v5(cz, [0.0], prod, fine, rc_grid, G1, channels_by_species=prod["chan"],
                               Wj_by_species=Wloc, alpha_rad=alpha, nodes=NODES, n_beta=12, n_ray=120, znodes=zn)
        # the "row" kernel: the family member at the on-axis mu-channel production-weighted density
        _, qa = v5.cone_production_split(cz, alpha[:2], np.zeros(1), prod["x_grid"], E_TOY,
                                         [prod["chan"]["total_numu"][2]], prod["geom"], NODES, n_ray=120,
                                         znodes=dict(cg=zn["cg"], x=zn["x"],
                                                     profiles=[[ch["total_numu"][2]] for ch in zn["chan"]]))
        sh = qa[0][:, 0] / qa[0][:, 0].sum()
        wrow = float(np.interp((sh * np.log(NODES)).sum(), np.log(NODES), widths))
        Wrow = {s: tuple(np.repeat(k[None], 6, 0) for k in (_kernel(alpha, 25.0), _kernel(alpha, 30.0),
                                                              _kernel(alpha, wrow))) for s in SPECIES}
        r0 = v5.joint_factor_v5(cz, [0.0], prod, fine, rc_grid, G1, channels_by_species=prod["chan"],
                                Wj_by_species=Wrow, alpha_rad=alpha, nodes=NODES, n_beta=12, n_ray=120, znodes=zn)
        return r["J"]["total_numu"][0, 0] / r0["J"]["total_numu"][0, 0]

    assert ratio(0.05) < ratio(1.0)


# (e) ------------------------------------------------------------------------
def test_eoff_model_validation():
    import mceq3d_flux as mf

    assert mf.EOFF_TABLE_V5 == "offaxis_excess_channel_v5.npz"
    with pytest.raises(ValueError, match="v5"):
        mf.MCEq3DFlux(eoff_model="v6")


def test_v5_config_and_tag_are_distinct_from_v4():
    import eoff_v4 as v4

    assert v5.SCHEME != v4.SCHEME
    assert v5.V5_CONFIG["eloss"] == "local" and v4.V4_CONFIG["eloss"] == "row"
    assert {k: v for k, v in v5.V5_CONFIG.items() if k != "eloss"} == \
        {k: v for k, v in v4.V4_CONFIG.items() if k != "eloss"}
