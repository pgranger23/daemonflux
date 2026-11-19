"""Gate G0.4 (reported): kernel statistics from two disjoint half samples.

Evaluates G0.4 as written in ``tools/cascade3d/GATES.md``::

    $PY tools/cascade3d/gates/g0_4.py [--tag fresh1e5]

and writes ``$N3/c0/gates/G0.4.md``, ``G0.4.json`` and ``G0.4_nodes.csv``
(one line per node).

For each node file (projectile, energy node) and each kernel entry
(served bin, child, child bin, l)::

    z = (mean_even - mean_odd) / sqrt(s^2 (1/n_even + 1/n_odd))

``mean_h`` is the moment per interaction from half ``h`` of the events and
``s^2`` the per-event variance of the full sample.  Expected: mean 0, rms 1.
Entries need a minimum raw particle count (the l = 0 sum).
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import kernels_gen as K  # noqa: E402
from g0_3 import gate_text  # noqa: E402

N3 = os.environ.get("N3", "/home/pigrange/mc3d_work/runs/native3d")
ELLS = (0, 1, 2, 8, 32, 64)
NMINS = (20, 200)


def node_z(path):
    """z arrays of one node: dict (nmin, l) -> 1D array; plus identity."""
    with np.load(path, allow_pickle=False) as d:
        S = np.array(d["S"], float)            # (2, nk, C, md, L+1)
        Q = np.array(d["Q"], float)            # (nk, C, md, L+1)
        nh = np.array(d["n_ev"], float)
        pdg, node, e0 = int(d["pdg"]), int(d["node"]), float(d["e0"])
    n = nh.sum()
    mean = S.sum(axis=0) / n
    s2 = np.maximum(Q / n - mean ** 2, 0.0)
    den = np.sqrt(s2 * (1.0 / nh[0] + 1.0 / nh[1]))
    diff = S[0] / nh[0] - S[1] / nh[1]
    cnt = S.sum(axis=0)[..., 0]
    out = {}
    for nmin in NMINS:
        for ell in ELLS:
            ok = (cnt >= nmin) & (den[..., ell] > 0)
            out[(nmin, ell)] = diff[..., ell][ok] / den[..., ell][ok]
    # relative statistical error of the l = 0 and l = 1 entries
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.sqrt(s2 / n) / np.abs(mean)
    ok = cnt >= NMINS[1]
    rel0 = rel[..., 0][ok]
    rel1 = rel[..., 1][ok]
    return pdg, node, e0, int(n), out, rel0, rel1


def stats(z):
    if z.size == 0:
        return {"n": 0, "mean": float("nan"), "rms": float("nan"),
                "f3": float("nan"), "f5": float("nan")}
    return {"n": int(z.size), "mean": float(z.mean()),
            "rms": float(np.sqrt((z ** 2).mean())),
            "f3": float((np.abs(z) > 3).mean()),
            "f5": float((np.abs(z) > 5).mean())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="fresh1e5")
    ap.add_argument("--out", default=os.path.join(N3, "c0", "gates", "G0.4"))
    a = ap.parse_args(argv)
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(K.tag_dir(a.tag), "node_*.npz")))
    allz = {(nm, ell): [] for nm in NMINS for ell in ELLS}
    byproj = {}
    nodes = []
    for fp in files:
        pdg, node, e0, n, zz, rel0, rel1 = node_z(fp)
        for key, z in zz.items():
            allz[key].append(z)
            byproj.setdefault(pdg, {}).setdefault(key, []).append(z)
        z1 = zz[(NMINS[1], 1)]
        z0 = zz[(NMINS[1], 0)]
        nodes.append({"pdg": pdg, "node": node, "e0": e0, "n_events": n,
                      "n_z": int(z1.size), **{"l0_" + k: v for k, v in
                                              stats(z0).items() if k != "n"},
                      **{"l1_" + k: v for k, v in stats(z1).items()
                         if k != "n"},
                      "med_rel_err_l0": float(np.median(rel0))
                      if rel0.size else float("nan"),
                      "med_rel_err_l1": float(np.median(rel1))
                      if rel1.size else float("nan")})
    lines = ["# Gate G0.4: kernel statistics, half samples (reported)", "",
             "## Gate text (GATES.md)", "", gate_text("G0.4", "G1."), "",
             "## Result", "",
             "**REPORTED** (not gated). Command: `$PY tools/cascade3d/gates/"
             "g0_4.py " + " ".join(sys.argv[1:]) + "`", "",
             f"Kernel set `gen_{a.tag}`: {len(files)} node files, "
             f"{sorted({n['n_events'] for n in nodes})} events per node.", "",
             "z = (even half - odd half) / expected error of the difference. "
             "Expected: mean 0, rms 1, fraction beyond 3 about 0.003.", "",
             "### All nodes together", "",
             "| min particles | l | entries | mean z | rms z | "
             "frac abs(z)>3 | frac abs(z)>5 |", "|---|---|---|---|---|---|---|"]
    res = {"all": {}, "by_projectile": {}}
    for nm in NMINS:
        for ell in ELLS:
            s = stats(np.concatenate(allz[(nm, ell)]))
            res["all"][f"n{nm}_l{ell}"] = s
            lines.append(f"| {nm} | {ell} | {s['n']} | {s['mean']:+.4f} | "
                         f"{s['rms']:.4f} | {s['f3']:.5f} | {s['f5']:.6f} |")
    lines += ["", f"### By projectile (entries with at least {NMINS[1]} "
              "particles)", "",
              "| projectile | nodes | entries (l = 1) | rms z, l = 0 | "
              "rms z, l = 1 | rms z, l = 8 | rms z, l = 64 | "
              "frac abs(z)>3, l = 1 |", "|---|---|---|---|---|---|---|---|"]
    for pdg in K.PROJECTILES:
        if pdg not in byproj:
            continue
        st = {ell: stats(np.concatenate(byproj[pdg][(NMINS[1], ell)]))
              for ell in ELLS}
        res["by_projectile"][str(pdg)] = st
        nn = sum(1 for n in nodes if n["pdg"] == pdg)
        lines.append(f"| {K.MCEQ_NAME[pdg]} | {nn} | {st[1]['n']} | "
                     f"{st[0]['rms']:.3f} | {st[1]['rms']:.3f} | "
                     f"{st[8]['rms']:.3f} | {st[64]['rms']:.3f} | "
                     f"{st[1]['f3']:.4f} |")
    r1 = np.array([n["l1_rms"] for n in nodes])
    nz = np.array([n["n_z"] for n in nodes])
    exp = 1.0 / np.sqrt(2.0 * np.maximum(nz, 1))     # error of an rms of n
    dev = (r1 - 1.0) / exp
    ok = np.isfinite(r1) & (nz >= 50)
    lines += ["", "### Per node (l = 1, entries with at least "
              f"{NMINS[1]} particles)", "",
              f"Nodes with at least 50 such entries: {int(ok.sum())} of "
              f"{len(nodes)}. Their rms z: median {np.median(r1[ok]):.3f}, "
              f"min {r1[ok].min():.3f}, max {r1[ok].max():.3f}. "
              "Deviation of a node's rms from 1 in units of its own "
              "sampling error (entries of one node share events, so this "
              "overstates the significance): rms "
              f"{np.sqrt((dev[ok] ** 2).mean()):.2f}, "
              f"largest {np.abs(dev[ok]).max():.1f}.", "",
              "Ten nodes with the largest rms z:", "",
              "| projectile | node | E0 [GeV] | entries | rms z (l = 1) | "
              "rms z (l = 0) | frac abs(z)>3 |", "|---|---|---|---|---|---|---|"]
    order = np.argsort(-np.where(ok, r1, -np.inf))[:10]
    for i in order:
        n = nodes[i]
        lines.append(f"| {K.MCEQ_NAME[n['pdg']]} | {n['node']} | "
                     f"{n['e0']:.3g} | {n['n_z']} | {n['l1_rms']:.3f} | "
                     f"{n['l0_rms']:.3f} | {n['l1_f3']:.4f} |")
    rel0 = np.array([n["med_rel_err_l0"] for n in nodes])
    rel1 = np.array([n["med_rel_err_l1"] for n in nodes])
    f = np.isfinite(rel0)
    lines += ["", "### Size of the statistical errors", "",
              "Median relative error of a kernel entry with at least "
              f"{NMINS[1]} particles, per node; over nodes: l = 0 median "
              f"{100 * np.median(rel0[f]):.2f}% (max "
              f"{100 * rel0[f].max():.2f}%), l = 1 median "
              f"{100 * np.median(rel1[f]):.2f}% (max "
              f"{100 * rel1[f].max():.2f}%).", "",
              f"Per-node numbers: `{os.path.basename(a.out)}_nodes.csv`. "
              f"Run time {time.time() - t0:.0f} s."]
    res["nodes_summary"] = {"n_nodes": len(nodes), "n_ok": int(ok.sum()),
                            "l1_rms_median": float(np.median(r1[ok])),
                            "l1_rms_min": float(r1[ok].min()),
                            "l1_rms_max": float(r1[ok].max())}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out + ".md", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(a.out + ".json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    keys = list(nodes[0].keys())
    with open(a.out + "_nodes.csv", "w") as fh:
        fh.write(",".join(keys) + "\n")
        for n in nodes:
            fh.write(",".join(f"{n[k]:.6g}" if isinstance(n[k], float)
                              else str(n[k]) for k in keys) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
