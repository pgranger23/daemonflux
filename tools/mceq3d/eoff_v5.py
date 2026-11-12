"""E_off v5 (opt-in, ``MCEq3DFlux(eoff_model="v5")``): E_off v4 with the muon energy-loss kernel evaluated at the
air density of EACH production sample of the cone integral, instead of one density per arrival row.

Why (2026-11-12, ``$W/runs/mgfix/HANDBACK.md``).  v4 (:mod:`eoff_v4`) builds its mu-channel kernels with continuous
muon energy loss in a medium of constant density and picks ONE member of the density family per arrival row: the
muon-decay-weighted density of the on-axis 1D production along that row's line of sight (:func:`eoff_v4.rho_dec`,
0.26e-4 g/cm3 at |cos Z| 0.05, 2.5e-4 at 0.95).  Near the horizon, however, the E_off enhancement is carried by
off-axis samples -- primaries much steeper than the arrival direction feeding production points deep in the atmosphere
-- and those sit in denser air: for the mu channel at 0.10-0.18 GeV and |cos Z| 0.05 the log-mean density of the
alpha > 47 deg samples is 1.3e-4 against 3.6e-5 on the axis (at |cos Z| 0.65 the two agree, 1.1e-4 / 1.2e-4).  The
muons there lose more energy before they decay, so their neutrinos are more collimated than the row kernel says: the
row kernel is too wide exactly for the samples that carry the horizon enhancement, and the error grows with the width
of the kernel (all nucleon generations).  v5 splits the arrival-ray quadrature by the local density:

    J = sum_j sum_alpha W_j(alpha) mean_beta[ q_j(alpha, beta) G(alpha, beta) ],
    q_j(alpha, beta) = Int dl rho(h) lambda_j(rho(h)) p(X_slant(l, psi_p), E),

with ``W_j`` the energy-loss kernel at the family density ``rho_j`` (:data:`eoff_v4.RHO_FAMILY`, extended with
:data:`RHO_EXTRA` so that the ray's whole density range is covered) and ``lambda_j`` the linear-in-ln-rho
interpolation weights of :func:`eoff_v4.kernel_W_rho` (clamped to the end members), evaluated per ray point (the
density at the neutrino production point = the muon decay point).  Summed over ``j`` the weights are 1, so
``sum_j q_j`` is v4's ``q``, and a density-independent kernel gives v4's integral (tested).  The direct and kaon
channels have no energy loss and use v4's kernels unchanged.  Everything else -- generator kernels (all generations,
polarisation, K -> mu), zenith-node profiles, geometry, cutoff, the up-going table product -- is v4's.

What this is NOT: a vertex-displacement or generation-resolved factorisation.  The field-free toy of the mgfix study
(the MC's own cascade physics on the same chains, ``$W/runs/mgfix/scripts/toy3d.py``) shows that displacement,
truncation and the alpha-X correlation of later generations each move the horizon by <= 1-2%; the remaining
factorisation error of the product form, about +2% / +4% at |cos Z| 0.05 for 0.10-0.16 / 0.16-0.25 GeV (1.4 / 2.1
sigma), ~0 at 0.15 and about -2% at 0.4-1 GeV, is not in v5 and is carried as an uncertainty.
"""

from __future__ import annotations

import os

import numpy as np

SCHEME = "v5a"  # part of the v5 table tag
#: densities [g/cm3] added below / above eoff_v4.RHO_FAMILY so that the energy-loss kernel family spans the densities
#: of the production samples (ground 1.2e-3, 80 km 1.8e-8); outside it the end member is used (clamped)
RHO_EXTRA = (1.0e-3,)
#: v5 configuration (= eoff_v4.V4_CONFIG with eloss="local")
V5_CONFIG = dict(multigen=True, polarisation=True, kmu=True, eloss="local", zdep=True)


def rho_nodes():
    import eoff_v4 as v4

    return tuple(sorted(set(float(r) for r in v4.RHO_FAMILY) | set(float(r) for r in RHO_EXTRA)))


def rho_weights(rho, nodes):
    """``(len(nodes), n)`` linear-in-ln-rho interpolation weights of densities ``rho`` over ``nodes`` (ascending),
    clamped to the end members (the convention of :func:`eoff_v4.kernel_W_rho`); columns sum to 1."""
    rs = np.asarray(nodes, float)
    lr = np.log(np.clip(np.ravel(np.asarray(rho, float)), rs[0], rs[-1]))
    lrs = np.log(rs)
    w = np.zeros((len(rs), lr.size))
    if len(rs) == 1:
        w[0] = 1.0
        return w
    j = np.clip(np.searchsorted(lrs, lr) - 1, 0, len(rs) - 2)
    t = (lr - lrs[j]) / (lrs[j + 1] - lrs[j])
    np.add.at(w, (j, np.arange(lr.size)), 1.0 - t)
    np.add.at(w, (j + 1, np.arange(lr.size)), t)
    return w


def cone_production_split(cos_theta, alpha_rad, beta_rad, x_grid, ep_grid, profiles, geom, nodes, n_ray=260,
                          ray=None, znodes=None, rho_of_h=None):
    """:func:`joint_cone.cone_production_multi` with the arrival-ray weight split by local density.

    Returns ``(q, q_axis)``: ``q[i]`` (n_rho, n_alpha, n_beta, nE) and ``q_axis[i]`` (n_rho, nE) for each profile ``i``;
    ``q[i].sum(0)`` / ``q_axis[i].sum(0)`` are the unsplit v4 quantities.  ``rho_of_h`` defaults to
    ``offaxis_mc._RHO`` (set by the band profiles)."""
    import joint_cone as jc
    import offaxis_mc as ox

    rho_of_h = ox._RHO if rho_of_h is None else rho_of_h
    nE = len(ep_grid)
    P = np.concatenate([np.asarray(pi, float) for pi in profiles], axis=1)
    e_stack = np.arange(P.shape[1], dtype=float) + 1.0
    zn = None
    if znodes is not None:
        zn = dict(cg=np.asarray(znodes["cg"], float), x=list(znodes["x"]),
                  p=[np.concatenate([np.asarray(pi, float) for pi in pl], axis=1) for pl in znodes["profiles"]])
    h_ray, _, psi_o, w = (jc.arrival_ray(cos_theta, n_ray=n_ray) if ray is None else ray)
    lam = rho_weights(rho_of_h(h_ray), nodes)                       # (n_rho, n_ray)
    wj = lam * w[None, :]
    h_grid, psi_grid, table = geom
    cpo, spo = np.cos(psi_o), np.sin(psi_o)
    x_o = ox._interp_xslant(h_ray, psi_o, h_grid, psi_grid, table)
    qa = wj @ jc._p_lookup(x_o, h_ray, psi_o, x_grid, e_stack, P, zn)
    nb, nray = len(beta_rad), len(h_ray)
    q = np.zeros((len(nodes), len(alpha_rad), nb, P.shape[1]))
    cb = np.cos(beta_rad)
    h_rep = np.repeat(h_ray, nb)
    for ia, a in enumerate(alpha_rad):
        if a == 0.0:
            q[:, ia, :] = qa[:, None, :]
            continue
        ca, sa = np.cos(a), np.sin(a)
        psi_p = np.arccos(np.clip(ca * cpo[:, None] + sa * spo[:, None] * cb[None, :], -1, 1))
        xv = ox._interp_xslant(h_rep, psi_p.ravel(), h_grid, psi_grid, table)
        pk = jc._p_lookup(xv, h_rep, psi_p.ravel(), x_grid, e_stack, P, zn).reshape(nray, nb, -1)
        q[:, ia] = np.einsum("jl,lbe->jbe", wj, pk)
    return ([q[..., i * nE:(i + 1) * nE] for i in range(len(profiles))],
            [qa[:, i * nE:(i + 1) * nE] for i in range(len(profiles))])


def family_W(family, nodes, band, species, channel, e_grid):
    """Normalised mu-channel kernels ``(len(nodes), nE, n_alpha)`` at the density ``nodes`` from the v4 energy-loss
    family (:func:`eoff_v4.kernel_W_rho`; nodes outside the family are clamped to its end members)."""
    import eoff_v4 as v4

    return np.stack([v4.kernel_W_rho(family, float(r), band, species, channel, e_grid) for r in nodes])


def engine_inputs(engine, cache_dir=None, config=None):
    """v4's inputs (:func:`eoff_v4.engine_inputs` with energy loss "row" for the shared kernel family) plus the density
    nodes and a channel-aware kernel function ``Wj(b, species, channel, e_grid) -> (n_rho, nE, n_alpha)``: the
    energy-loss family at the nodes for the mu channel, the v4 kernel replicated for the others."""
    import eoff_v3 as v3
    import eoff_v4 as v4

    cfg = dict(V5_CONFIG if config is None else config)
    if cfg.get("eloss") != "local":
        raise ValueError("eoff_v5 needs eloss='local'")
    cfg4 = dict(cfg)
    cfg4["eloss"] = "row"
    inp = v4.engine_inputs(engine, cache_dir, cfg4)
    nodes = rho_nodes()
    sw = {k: cfg[k] for k in ("multigen", "polarisation", "kmu")}
    fam = v4.kernel_family(engine, rhos=nodes, **sw)       # cached per density (eoff_v4 kernel cache keys)
    inp["family_v5"] = fam
    memo = {}

    def Wj(b, s, c, e_grid):
        key = (b, s, c, len(e_grid), float(e_grid[0]), float(e_grid[-1]))
        if key not in memo:
            if c == "mu":
                memo[key] = family_W(fam, nodes, b, s, c, e_grid)
            else:
                w = v3.kernel_W(fam[0][1], b, s, c, e_grid)    # no energy loss in dir / k: identical in every member
                memo[key] = np.repeat(w[None], len(nodes), axis=0)
        return memo[key]

    inp.update(nodes=nodes, Wj=Wj, config=cfg)
    return inp


def table_rows_v5(cz_rows, prods, zbands, Wj, nodes, n_beta=18, n_ray=260):
    """``N[b, s, c, iz, E]``, ``D[...]`` at G = 1 with local-density mu-channel kernels (:func:`eoff_v4.table_rows_v4`
    with the ray split by density)."""
    import eoff_v3 as v3

    ep = prods[0]["ep_grid"]
    alpha = np.deg2rad(v3.ALPHA_NODES_DEG)
    beta = np.linspace(0.0, 2 * np.pi, n_beta, endpoint=False)
    N = np.zeros((v3.N_BANDS, 4, 3, len(cz_rows), len(ep)))
    D = np.zeros_like(N)
    for iz, cz in enumerate(cz_rows):
        for b in range(v3.N_BANDS):
            plist = [prods[b]["chan"][s][ic] for s in v3.SPECIES for ic in range(3)]
            zn = None
            if zbands is not None:
                zb = zbands[b]
                zn = dict(cg=zb["cg"], x=zb["x"],
                          profiles=[[ch[s][ic] for s in v3.SPECIES for ic in range(3)] for ch in zb["chan"]])
            qs, qas = cone_production_split(float(cz), alpha, beta, prods[0]["x_grid"], ep, plist, prods[0]["geom"],
                                            nodes, n_ray=n_ray, znodes=zn)
            i = 0
            for si, s in enumerate(v3.SPECIES):
                for ic, c in enumerate(v3.CHANNELS):
                    W = Wj(b, s, c, ep)                              # (n_rho, nE, n_alpha)
                    N[b, si, ic, iz] = np.einsum("jea,jae->e", W, qs[i].mean(axis=2))
                    D[b, si, ic, iz] = qas[i].sum(0)
                    i += 1
    return N, D


def table_tag(engine):
    return f"{engine._tag}|{SCHEME}|nodes={','.join(f'{r:.3g}' for r in rho_nodes())}"


def build_table_v5(engine, out_path, cz_rows=None, cache_dir=None, config=None):
    """The v5 E_off table (G = 1), same schema as :func:`eoff_v4.build_table_v4` plus the density nodes.  Never
    overwrites an existing file."""
    import eoff_v3 as v3
    import eoff_v4 as v4
    import offaxis_mc as ox

    if os.path.exists(out_path):
        raise FileExistsError(f"{out_path} exists; tables are never overwritten")
    cz_rows = np.round(np.arange(0.05, 1.0, 0.1), 2) if cz_rows is None else np.asarray(cz_rows)
    inp = engine_inputs(engine, cache_dir, config)
    N, D = table_rows_v5(cz_rows, inp["prods"], inp["zbands"], inp["Wj"], inp["nodes"])
    with np.errstate(invalid="ignore", divide="ignore"):
        Eb = np.where(D.sum(2) > 0, N.sum(2) / np.maximum(D.sum(2), 1e-300), 1.0)
        Es = np.where(D.sum((0, 2)) > 0, N.sum((0, 2)) / np.maximum(D.sum((0, 2)), 1e-300), 1.0)
    np.savez(out_path, e=inp["prods"][0]["ep_grid"], cz=cz_rows, E_off_s=Es, E_off=Es[0],
             E_off_band_s=Eb, D_band_s=D.sum(2), N_bsc=N, D_bsc=D,
             species=np.array(v3.SPECIES), band_T=np.asarray(v3.BAND_T),
             alpha_nodes=v3.ALPHA_NODES_DEG, rho_nodes=np.asarray(inp["nodes"]), znodes_deg=np.asarray(v4.ZNODES_DEG),
             tag=np.array(engine._tag), table_tag=np.array(table_tag(engine)),
             cone_mode=np.array("v5-local-density-eloss"), scheme=np.array(SCHEME), config=np.array(str(inp["config"])),
             provenance=np.array(
                 "eoff_v5.build_table_v5: eoff_v4 kernels and zenith-node profiles; the mu-channel energy-loss kernel "
                 "interpolated in ln rho at the air density of every production sample of the cone integral "
                 "(eoff_v4.RHO_FAMILY + RHO_EXTRA nodes); no reference flux, no MC tally used."),
             **ox._db_keys())
    return out_path


# ---------------------------------------------------------------------------
# the joint (production x cutoff) cone integral with local-density kernels
# ---------------------------------------------------------------------------
def joint_factor_v5(cos_theta, azimuths, prod, fine, rc_grid, G, *, channels_by_species, Wj_by_species, alpha_rad,
                    nodes, d_cone=None, mu_plus=(), prod_frame=True, h_prod_km=30.0, n_beta=18, n_ray=260,
                    q_cache=None, rc_family=None, site_lat=None, znodes=None):
    """``dict(J, p_axis_s)`` per species, shapes (n_az, nE) / (nE): the channel-resolved, sampled-kernel path of
    :func:`joint_cone.delivered_joint_factor` (same cone directions, production-point frame, cutoff lookup and
    channel sum, the same operations in the same order) with the arrival ray split by local density and, per
    channel, ``Wj_by_species[s][c]`` of shape (n_rho, nE, n_alpha):

        J_s = sum_c sum_j sum_alpha W_{s,c,j}(alpha) mean_beta[q_{s,c,j}(alpha, beta) G_s(alpha, beta)].

    With ``W_{s,c,j}`` equal for all ``j`` this is ``delivered_joint_factor``'s J up to the floating-point order of
    the sum over the density nodes (tested)."""
    import joint_cone as jc

    ep_grid = prod["ep_grid"]
    nE = len(ep_grid)
    alpha = np.asarray(alpha_rad, float)
    beta = np.linspace(0.0, 2 * np.pi, n_beta, endpoint=False)
    zen_f, az_f, rc_f = fine
    if prod_frame:
        dn = zen_f <= 89.9
        zen_f, rc_f = zen_f[dn], rc_f[dn]
    chan = channels_by_species
    sp_order = tuple(sorted(chan))
    key = ("v5", round(float(cos_theta), 9), n_beta, n_ray, sp_order, len(alpha), float(alpha.sum()), id(prod),
           None if znodes is None else id(znodes), tuple(float(r) for r in nodes))
    got = None if q_cache is None else q_cache.get(key)
    if got is None:
        plist = [pc for s in sp_order for pc in chan[s]]
        zn = None
        if znodes is not None:
            zn = dict(cg=znodes["cg"], x=znodes["x"],
                      profiles=[[pc for s in sp_order for pc in ch_k[s]] for ch_k in znodes["chan"]])
        got = cone_production_split(cos_theta, alpha, beta, prod["x_grid"], ep_grid, plist, prod["geom"], nodes,
                                    n_ray=n_ray, znodes=zn)
        if q_cache is not None:
            q_cache[key] = got
    qs, qas = got
    idx = {s: 3 * i for i, s in enumerate(sp_order)}
    p_axis_s = {s: np.maximum(qas[idx[s]].sum(0) + qas[idx[s] + 1].sum(0) + qas[idx[s] + 2].sum(0), 1e-300)
                for s in sp_order}
    J_out = {s: np.zeros((len(azimuths), nE)) for s in sp_order}
    zen_deg = float(np.degrees(np.arccos(np.clip(float(cos_theta), -1.0, 1.0))))
    for ia, azd in enumerate(azimuths):
        npv = jc.cone_directions(cos_theta, azd, alpha, beta)
        if rc_family is not None:
            import mceq3d_flux as _mf

            dn_, de_ = _mf.prod_point_offset_km(zen_deg, azd, h_prod_km)
            zen_u = np.asarray(rc_family["zen"], float)
            az_u = np.asarray(rc_family["az"], float)
            rc_u = _mf.interp_site_rc(rc_family, dn_, de_)
            frame = _mf.prod_frame_exact(cos_theta, azd, h_prod_km, site_lat)
        else:
            zen_u, az_u, rc_u = zen_f, az_f, rc_f
            frame = jc.prod_point_frame(cos_theta, azd, h_prod_km) if prod_frame else None

        def _rc(v, _z=zen_u, _a=az_u, _r=rc_u, _f=frame):
            if _f is not None:
                th, ph = jc.local_angles(v, _f)
            else:
                th = np.degrees(np.arccos(np.clip(v[..., 2], -1, 1)))
                ph = np.degrees(np.arctan2(v[..., 1], v[..., 0]))
            return jc.rc_bilinear(_z, _a, _r, th, ph)

        rc0 = _rc(npv)
        Gab = {s: jc._interp_G(rc0, rc_grid, G[s]).reshape(len(alpha), len(beta), nE) for s in G}
        Gsh = None
        if d_cone is not None:
            pp = npv + d_cone
            pp /= np.linalg.norm(pp, axis=-1, keepdims=True)
            pm = npv - d_cone
            pm /= np.linalg.norm(pm, axis=-1, keepdims=True)
            Gp = {s: jc._interp_G(_rc(pp), rc_grid, G[s]).reshape(len(alpha), len(beta), nE) for s in G}
            Gm = {s: jc._interp_G(_rc(pm), rc_grid, G[s]).reshape(len(alpha), len(beta), nE) for s in G}
            Gsh = {s: (Gp[s] if s in mu_plus else Gm[s]) for s in G}
        for s in sp_order:
            g_mu = Gsh[s] if Gsh is not None else Gab[s]
            J = np.zeros(nE)
            for c, g in ((0, Gab[s]), (1, Gab[s]), (2, g_mu)):
                Wc = np.asarray(Wj_by_species[s][c], float)
                q = qs[idx[s] + c]
                J = J + np.einsum("jea,jae->e", Wc, (q * g[None]).mean(axis=2))
            J_out[s][ia] = J
    return dict(e=ep_grid, J=J_out, p_axis_s=p_axis_s)
