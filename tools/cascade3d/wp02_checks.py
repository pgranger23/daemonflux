"""Checks of WP 0.2 (angular kernels from generator events).

Writes ``$N3/c0/wp02_checks.md`` (tables) and ``wp02_checks.json``.

1. fresh versus pool kernels on the pilot nodes;
2. l = 0 against MCEq's interaction blocks as spectrum-weighted moments
   (preview of G0.3);
3. half-sample consistency (preview of G0.4);
4. energy conservation per event;
5. mean p_T of pions;
6. <cos theta> and the delta-M forward fraction g (L = 32) for p -> pi+.

Usage::

    python wp02_checks.py --pool pool2000 --fresh pilot2e4 \
        --nodes 2212:3,2212:23,211:10
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import kernels_gen as K

GAMMA = 2.7          # differential spectral index of the Z factors


def _node(tag, pdg, node):
    with np.load(K.node_file(tag, pdg, node), allow_pickle=False) as d:
        return {k: np.array(d[k]) for k in d.files}


def _node_moments(d):
    """Mean and variance of the per-interaction sums, per served bin."""
    n = float(d["n_ev"].sum())
    S = d["S"].sum(axis=0)
    mean = S / n
    var = np.maximum(d["Q"] / n - mean ** 2, 0.0) / n
    return mean, var


def _pad(a, m):
    out = np.zeros(a.shape[:-2] + (m, a.shape[-1]))
    out[..., :a.shape[-2], :] = a
    return out


# ---------------------------------------------------------------------------
def check_fresh_vs_pool(pool_tag, fresh_tag, nodes, lines, res):
    lines.append("## 1. Fresh versus pool kernels on the pilot nodes\n")
    lines.append("z = (fresh - pool) / sqrt(var_fresh + var_pool) per "
                 "(child, child bin, l), entries with at least 20 pool "
                 "particles. Expected: mean 0, rms 1.\n")
    lines.append("| node | E0 [GeV] | bin k (E_kin) | l | n cells | mean z "
                 "| rms z | frac abs(z)>3 | chi2/ndf |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    out = {}
    for pdg, node in nodes:
        dp, df = _node(pool_tag, pdg, node), _node(fresh_tag, pdg, node)
        mp, vp = _node_moments(dp)
        mf, vf = _node_moments(df)
        m = max(mp.shape[-2], mf.shape[-2])
        mp, vp, mf, vf = (_pad(x, m) for x in (mp, vp, mf, vf))
        rawp = _pad(dp["S"].sum(0), m)[..., 0]
        eg = np.load(K.node_file(pool_tag, pdg, node).rsplit("/", 1)[0]
                     + "/grid_PHYS.npz")["e_grid"]
        kk = int(np.argmax(dp["w"]))
        k = int(dp["k"][kk])
        for ell in (0, 1, 2, 8, 32, 64):
            sel = rawp[kk] >= 20
            den = np.sqrt(vp[kk, ..., ell] + vf[kk, ..., ell])
            ok = sel & (den > 0)
            z = (mf[kk, ..., ell] - mp[kk, ..., ell])[ok] / den[ok]
            r = {"n": int(z.size), "mean": float(z.mean()),
                 "rms": float(np.sqrt((z ** 2).mean())),
                 "f3": float((np.abs(z) > 3).mean()),
                 "chi2ndf": float((z ** 2).mean())}
            out[f"{pdg}_{node}_l{ell}"] = r
            lines.append(f"| {pdg}/{node} | {dp['e0']:.4g} | {k} "
                         f"({eg[k]:.3g}) | {ell} | {r['n']} | "
                         f"{r['mean']:+.3f} | {r['rms']:.3f} | "
                         f"{r['f3']:.4f} | {r['chi2ndf']:.3f} |")
        # integrated quantities per child
        nP, nF = dp["n_ev"].sum(), df["n_ev"].sum()
        for ci, c in enumerate(K.CHILDREN):
            if c not in (211, -211, 2212, 321):
                continue
            a = []
            for mean, var in ((mp, vp), (mf, vf)):
                mu = mean[kk, ci, :, 0].sum()
                c1 = mean[kk, ci, :, 1].sum()
                a.append((mu, np.sqrt(var[kk, ci, :, 0].sum()), c1 / mu))
            ptp = dp["pt_sum"][ci] / max(dp["mult"][ci], 1)
            ptf = df["pt_sum"][ci] / max(df["mult"][ci], 1)
            out[f"{pdg}_{node}_child{c}"] = {
                "mult_pool": a[0][0], "mult_fresh": a[1][0],
                "cos_pool": a[0][2], "cos_fresh": a[1][2],
                "pt_pool": ptp, "pt_fresh": ptf,
                "n_pool": int(nP), "n_fresh": int(nF)}
    lines.append("")
    lines.append("Integrated over child energy, at the bin of largest "
                 "weight (pool 2000 events / fresh 2e4 events):\n")
    lines.append("| node | child | mult pool | mult fresh | <cos> pool | "
                 "<cos> fresh | <p_T> pool | <p_T> fresh |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for key, v in out.items():
        if "_child" not in key:
            continue
        nd, c = key.split("_child")
        lines.append(f"| {nd} | {c} | {v['mult_pool']:.4f} | "
                     f"{v['mult_fresh']:.4f} | {v['cos_pool']:.4f} | "
                     f"{v['cos_fresh']:.4f} | {v['pt_pool']:.4f} | "
                     f"{v['pt_fresh']:.4f} |")
    lines.append("")
    res["fresh_vs_pool"] = out


# ---------------------------------------------------------------------------
def _mceq_numbers(population="PHYS"):
    import kernels_mceq as KM
    d = KM.load(population)
    w = d["e_widths"]
    lam = d["inv_lambda_int"]

    def n_of(parent, child):
        p = KM.species_index(d, K.MCEQ_NAME[parent])
        c = KM.species_index(d, K.MCEQ_NAME[child])
        b = KM.get_block(d, "int", p, c, loss=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.where(lam[p][None, :] > 0, b / lam[p][None, :], 0.0)
        if p == c and d["can_interact"][p]:
            y[np.diag_indices_from(y)] += np.where(lam[p] > 0, 1.0, 0.0)
        return y * w[:, None] / w[None, :]          # number per bin
    return d, n_of


def _mceq_raw_numbers(population="PHYS"):
    """MCEq's raw yield matrices (``Interactions.get_matrix``), before the
    resonance folding of ``int_m``: number per child bin."""
    import importlib.util  # noqa: F401
    import MCEq.config as cfg
    import crflux.models as crf
    import mceq3d_db
    from mceq3d_flux import _primary_floor_set
    pop = K.POPULATIONS[population]
    cfg.e_min = pop["e_min"]
    cfg.debug_level = 0
    with _primary_floor_set(pop["primary_floor"]):
        mc = mceq3d_db.make_run(interaction_model="SIBYLL23D",
                                primary_model=(crf.GlobalSplineFitBeta,
                                               None), theta_deg=0.0)
    eg = mc.e_grid
    xr = eg[:, None] / eg[None, :]

    def n_of(parent, child):
        try:
            return mc._interactions.get_matrix((parent, 0), (child, 0)) * xr
        except Exception:
            return np.zeros_like(xr)
    return eg, n_of


def check_mceq_z(gen, lines, res):
    lines.append("## 2. l = 0 against MCEq's blocks: spectrum-weighted "
                 "moments (G0.3 preview)\n")
    lines.append(f"Z = sum_m n[m, k] (E_m / E_k)^(gamma - 1), gamma = "
                 f"{GAMMA}, kinetic bin centres, E_m <= E_k; n = number per "
                 "interaction in child bin m. N = sum_m n (child bins on "
                 "the grid). Generator: `gen_pool2000`. MCEq raw: "
                 "`Interactions.get_matrix` of the v142p database (the "
                 "yields before MCEq folds mesons below e_mix into their "
                 "decay products). MCEq int_m: the blocks of "
                 "`mceq_PHYS.npz` divided by 1/lambda (these lack every "
                 "child below its e_mix: pi 8.9 GeV, K+- 89 GeV, K_L 22 "
                 "GeV, K_S 8.9 TeV, Lambda 5.6 TeV, n 1.4 GeV).\n")
    d, n_blk = _mceq_numbers()
    eg_raw, n_raw = _mceq_raw_numbers()
    eg = gen["e_grid"]
    assert np.allclose(d["e_grid"], eg, rtol=1e-12)
    assert np.allclose(eg_raw, eg, rtol=1e-12)
    pairs = [(2212, 211), (2212, -211), (2212, 321), (2212, -321),
             (2212, 2212), (2212, 2112), (2212, 3122), (2212, 130),
             (2112, 211), (211, 211), (211, -211), (211, 321),
             (211, 2212), (321, 321), (321, 211), (-2212, 211)]
    lines.append("| channel | E_k [GeV] | Z gen | Z MCEq raw | gen/raw | "
                 "Z MCEq int_m | N gen | N MCEq raw |")
    lines.append("|---|---|---|---|---|---|---|---|")
    out = {}
    ki, mi = gen["k_index"], gen["m_index"]
    for par, ch in pairs:
        jp, ci = K.PROJECTILES.index(par), K.CHILDREN.index(ch)
        nm = n_raw(par, ch)
        nb = n_blk(par, ch)
        for e in (3.0, 10.0, 30.0, 100.0, 1000.0):
            k = int(np.argmin(np.abs(np.log(eg / e))))
            if gen["status"][jp, k] != K.SERVED:
                continue
            kx = k - ki[0]
            ng = np.zeros(len(eg))
            ng[mi] = gen["c_l"][jp, ci, kx, :, 0]
            x = eg / eg[k]
            wz = np.where(x <= 1.0 + 1e-9, x ** (GAMMA - 1), 0.0)
            zg, zm = float((ng * wz).sum()), float((nm[:, k] * wz).sum())
            zb = float((nb[:, k] * wz).sum())
            Ng, Nm = float(ng.sum()), float(nm[:, k].sum())
            r = zg / zm if zm > 0 else np.nan
            out[f"{par}->{ch}@{eg[k]:.3g}"] = {"Zgen": zg, "Zmceq": zm,
                                               "ratio": r, "Zblock": zb,
                                               "Ngen": Ng, "Nmceq": Nm}
            lines.append(f"| {par} -> {ch} | {eg[k]:.3g} | {zg:.4g} | "
                         f"{zm:.4g} | {r:.3f} | {zb:.4g} | {Ng:.4g} | "
                         f"{Nm:.4g} |")
    lines.append("")
    res["mceq_z"] = out


# ---------------------------------------------------------------------------
def check_halves(gen, lines, res, label):
    lines.append(f"## 3. Half-sample consistency, `{label}` (G0.4 "
                 "preview)\n")
    lines.append("z = (half_even - half_odd) / sqrt(4 var), var the "
                 "statistical variance of the full-sample moment (from the "
                 "per-event sums of squares). Entries with at least N raw "
                 "particles. Expected: mean 0, rms 1.\n")
    lines.append("| min particles | l | n entries | mean z | rms z | "
                 "frac abs(z)>3 | frac abs(z)>5 |")
    lines.append("|---|---|---|---|---|---|---|")
    h = gen["c_l_half"].astype(float)
    var = gen["c_l_var"]
    ne = gen["n_entries"]
    out = {}
    for nmin in (20, 200):
        for ell in (0, 1, 2, 8, 32, 64):
            den = np.sqrt(4 * var[..., ell])
            ok = (ne >= nmin) & (den > 0)
            z = (h[0, ..., ell] - h[1, ..., ell])[ok] / den[ok]
            r = {"n": int(z.size), "mean": float(z.mean()),
                 "rms": float(np.sqrt((z ** 2).mean())),
                 "f3": float((np.abs(z) > 3).mean()),
                 "f5": float((np.abs(z) > 5).mean())}
            out[f"n{nmin}_l{ell}"] = r
            lines.append(f"| {nmin} | {ell} | {r['n']} | {r['mean']:+.3f} | "
                         f"{r['rms']:.3f} | {r['f3']:.4f} | {r['f5']:.5f} |")
    lines.append("")
    res[f"halves_{label}"] = out


# ---------------------------------------------------------------------------
def check_energy(pool_tag, fresh_tag, nodes, gen, lines, res):
    lines.append("## 4. Energy conservation per event\n")
    out = {}
    lines.append("Fresh pilot, complete final state (before the KEEP "
                 "filter): dE = sum E_final - E_lab per event [GeV]. The "
                 "budget of the MC's guard is E_lab + 14.9 GeV (O-16).\n")
    lines.append("| node | model | n | mean dE | rms dE | min dE | max dE | "
                 "frac dE > 14.9 | frac dE < -1 MeV |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for pdg, node in nodes:
        df = _node(fresh_tag, pdg, node)
        e = json.loads(str(df["efs_json"]))
        meta = json.loads(str(df["meta_json"]))
        n = e["n"]
        mean = e["sum"] / n
        rms = np.sqrt(max(e["sum2"] / n - mean ** 2, 0))
        out[f"fresh_{pdg}_{node}"] = e
        lines.append(f"| {pdg}/{node} | {meta['model']} | {int(n)} | "
                     f"{mean:.3f} | {rms:.3f} | {e['min']:.3f} | "
                     f"{e['max']:.3f} | {e['n_over_guard'] / n:.5f} | "
                     f"{e['n_below_elab'] / n:.4f} |")
    lines.append("")
    # pools: kept energy at the node energy
    mx, over, nn = [], 0, 0
    for p, i in K.node_list("pool"):
        d = _node(pool_tag, p, i)
        b = json.loads(str(d["ekept_node_json"]))
        mx.append((b["max"], p, i))
        over += b["n_over_guard"]
        nn += b["n"]
    mx.sort()
    lines.append(f"All {len(mx)} pool nodes, kept final state at the node "
                 f"energy: {over} of {int(nn)} events exceed E0 + 14.9 GeV; "
                 f"largest sum E_kept / E0 = {mx[-1][0]:.3f} "
                 f"(pdg {mx[-1][1]}, node {mx[-1][2]}).\n")
    gf = gen["guard_fraction"]
    st = gen["status"][:, gen["k_index"]]
    sel = st == K.SERVED
    lines.append(f"Guard firing after rescaling to the MCEq bin centres "
                 f"(pool2000): largest fraction of events {gf[sel].max():.5f}"
                 f", mean {gf[sel].mean():.2e} over {int(sel.sum())} "
                 "(projectile, bin) cells.\n")
    res["energy"] = {"fresh": out, "pool_n_over_guard": int(over),
                     "pool_n": int(nn), "pool_max_ratio": mx[-1][0],
                     "guard_max": float(gf[sel].max()),
                     "guard_mean": float(gf[sel].mean())}


# ---------------------------------------------------------------------------
def check_pt(pool_tag, lines, res):
    lines.append("## 5. Mean p_T of charged pions\n")
    lines.append("Node level (before rescaling; p_T is not rescaled), pool "
                 "files, all particles of the species.\n")
    lines.append("| projectile | node | E0 [GeV] | model | <p_T> pi+ | "
                 "<p_T> pi- | rms p_T pi+ |")
    lines.append("|---|---|---|---|---|---|---|")
    out = {}
    ip, im = K.CHILDREN.index(211), K.CHILDREN.index(-211)
    for pdg in (2212, 211):
        for node in (0, 3, 8, 15, 16, 23, 36, 46, 58):
            d = _node(pool_tag, pdg, node)
            meta = json.loads(str(d["meta_json"]))
            a = d["pt_sum"] / np.maximum(d["mult"], 1)
            rms = np.sqrt(d["pt2_sum"][ip] / max(d["mult"][ip], 1))
            out[f"{pdg}_{node}"] = {"pip": a[ip], "pim": a[im], "rms": rms}
            lines.append(f"| {pdg} | {node} | {float(d['e0']):.4g} | "
                         f"{meta['model']} | {a[ip]:.4f} | {a[im]:.4f} | "
                         f"{rms:.4f} |")
    lines.append("")
    res["pt"] = out


# ---------------------------------------------------------------------------
def check_forward(gen, lines, res, label):
    lines.append(f"## 6. p -> pi+: <cos theta> and delta-M g (L = 32), "
                 f"`{label}`\n")
    jp, ci = K.PROJECTILES.index(2212), K.CHILDREN.index(211)
    eg, ki, mi = gen["e_grid"], gen["k_index"], gen["m_index"]
    cl = gen["c_l"][jp, ci]
    a = K.normalised_moments(cl)
    g, _ = K.delta_m(cl, L=32)
    err1 = K.moment_errors(cl, gen["c_l_var"][jp, ci],
                           gen["c_l_cov0"][jp, ci])[..., 1]
    es = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
    ms = [int(np.argmin(np.abs(np.log(eg[mi] / e)))) for e in es]
    hdr = " | ".join(f"{eg[mi][m]:.3g}" for m in ms)
    out = {}
    for name, arr in (("<cos theta>", a[..., 1]), ("g(L=32)", g)):
        lines.append(f"{name}; rows: projectile E_kin [GeV]; columns: pi+ "
                     "E_kin [GeV]; '-' = no entry or E_sec > E_proj\n")
        lines.append(f"| E_proj \\ E_pi | {hdr} |")
        lines.append("|---" * (len(ms) + 1) + "|")
        for e in (3.0, 10.0, 30.0, 100.0, 1000.0, 8000.0):
            k = int(np.argmin(np.abs(np.log(eg / e))))
            kx = k - ki[0]
            if not (0 <= kx < len(ki)) or gen["status"][jp, k] != K.SERVED:
                continue
            row = []
            for m in ms:
                v = arr[kx, m]
                if cl[kx, m, 0] <= 0 or eg[mi][m] > eg[k]:
                    row.append("-")
                elif name == "g(L=32)":
                    row.append(f"{v:.3f}")
                else:
                    row.append(f"{v:.4f}+-{err1[kx, m]:.4f}")
                out[f"{name}_{eg[k]:.3g}_{eg[mi][m]:.3g}"] = float(v)
            lines.append(f"| {eg[k]:.3g} | " + " | ".join(row) + " |")
        lines.append("")
    res[f"forward_{label}"] = out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="pool2000")
    ap.add_argument("--fresh", default="pilot2e4")
    ap.add_argument("--nodes", type=K._parse_nodes,
                    default=[(2212, 3), (2212, 23), (211, 10)])
    ap.add_argument("--out", default=os.path.join(K.N3, "c0",
                                                  "wp02_checks"))
    a = ap.parse_args(argv)
    gen = K.load(a.pool, "PHYS")
    lines = ["# WP 0.2 checks (generated by tools/cascade3d/wp02_checks.py)",
             ""]
    res = {}
    check_fresh_vs_pool(a.pool, a.fresh, a.nodes, lines, res)
    check_mceq_z(gen, lines, res)
    check_halves(gen, lines, res, a.pool)
    check_energy(a.pool, a.fresh, a.nodes, gen, lines, res)
    check_pt(a.pool, lines, res)
    check_forward(gen, lines, res, a.pool)
    with open(a.out + ".md", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(a.out + ".json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
