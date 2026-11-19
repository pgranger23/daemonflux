"""Gate G0.3 (reported): generator yields against MCEq's, as Z factors.

Evaluates G0.3 as written in ``tools/cascade3d/GATES.md``::

    $PY tools/cascade3d/gates/g0_3.py [--tag fresh1e5] [--also pool2000]

and writes ``$N3/c0/gates/G0.3.md`` and ``G0.3.json``.

For every projectile of the kernel file and every secondary, at the MCEq bin
nearest to 3, 10, 30, 100 and 1000 GeV (kinetic)::

    Z = sum_m n[m, k] (E_m / E_k)^(gamma - 1),   gamma = 2.7,  E_m <= E_k

``n[m, k]`` is the number of secondaries per interaction in child bin ``m``.
Generator: the l = 0 moment of the kernel file.  MCEq: the raw yield matrix
``Interactions.get_matrix`` of the database, i.e. before MCEq folds mesons
below their mixing energy into their decay products (the folded blocks are
not comparable channel by channel).

The statistical error of the generator's Z treats child bins as independent.
The two half samples give a check of it: ``z_half = (Z_even - Z_odd) / (2 sigma)``.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import kernels_gen as K  # noqa: E402
import wp02_checks as W  # noqa: E402

N3 = os.environ.get("N3", "/home/pigrange/mc3d_work/runs/native3d")
ENERGIES = (3.0, 10.0, 30.0, 100.0, 1000.0)
GAMMA = W.GAMMA
Z_MIN = 1e-3  # channels below this Z (MCEq) are left out of the summary


def gate_text(name, nxt):
    path = os.path.join(os.path.dirname(HERE), "GATES.md")
    out, on = [], False
    for ln in open(path).read().split("\n"):
        if ln.startswith("**" + name):
            on = True
        elif on and (ln.startswith("**" + nxt) or ln.startswith("## ")):
            break
        if on:
            out.append(ln)
    return "\n".join(out).strip()


def z_table(gen, eg_raw, n_raw):
    eg = gen["e_grid"]
    assert np.allclose(eg_raw, eg, rtol=1e-12)
    ki, mi = gen["k_index"], gen["m_index"]
    rows = []
    for jp, par in enumerate(K.PROJECTILES):
        for ci, ch in enumerate(K.CHILDREN):
            nm = n_raw(par, ch)
            for e in ENERGIES:
                k = int(np.argmin(np.abs(np.log(eg / e))))
                if gen["status"][jp, k] != K.SERVED:
                    continue
                kx = k - ki[0]
                x = eg[mi] / eg[k]
                wz = np.where(x <= 1.0 + 1e-9, x ** (GAMMA - 1), 0.0)
                zg = float((gen["c_l"][jp, ci, kx, :, 0] * wz).sum())
                sg = float(np.sqrt((gen["c_l_var"][jp, ci, kx, :, 0]
                                    * wz ** 2).sum()))
                zh = [float((gen["c_l_half"][h, jp, ci, kx, :, 0].astype(float)
                             * wz).sum()) for h in (0, 1)]
                xf = eg / eg[k]
                wf = np.where(xf <= 1.0 + 1e-9, xf ** (GAMMA - 1), 0.0)
                zm = float((nm[:, k] * wf).sum())
                if zg == 0.0 and zm == 0.0:
                    continue
                rows.append({"parent": int(par), "child": int(ch),
                             "E": float(eg[k]), "Zgen": zg, "sig": sg,
                             "Zmceq": zm,
                             "ratio": zg / zm if zm > 0 else float("nan"),
                             "zhalf": (zh[0] - zh[1]) / (2 * sg)
                             if sg > 0 else float("nan")})
    return rows


def summary(rows, lines, label):
    big = [r for r in rows if r["Zmceq"] >= Z_MIN and r["Zgen"] > 0]
    lr = np.array([np.log(r["ratio"]) for r in big])
    lines.append(f"### Summary, `{label}`\n")
    lines.append(f"Channels with Z (MCEq) >= {Z_MIN:g}: {len(big)} of "
                 f"{len(rows)} rows.\n")
    lines.append("| E_k [GeV] | rows | median gen/MCEq | 16th pct | 84th pct | "
                 "min | max | within 10% | within 20% |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    es = sorted({r["E"] for r in big})
    for e in es + [None]:
        s = np.array([e is None or r["E"] == e for r in big])
        q = np.exp(lr[s])
        name = "all" if e is None else f"{e:.3g}"
        lines.append(f"| {name} | {s.sum()} | {np.median(q):.3f} | "
                     f"{np.percentile(q, 16):.3f} | {np.percentile(q, 84):.3f}"
                     f" | {q.min():.3f} | {q.max():.3f} | "
                     f"{(np.abs(q - 1) <= 0.1).mean():.2f} | "
                     f"{(np.abs(q - 1) <= 0.2).mean():.2f} |")
    zh = np.array([r["zhalf"] for r in big])
    zh = zh[np.isfinite(zh)]
    lines.append("")
    lines.append(f"Half-sample check of the quoted errors: z_half has mean "
                 f"{zh.mean():+.3f}, rms {np.sqrt((zh ** 2).mean()):.3f} "
                 f"(expected 0 and 1), n = {zh.size}.\n")
    return {"n": len(big), "median": float(np.exp(np.median(lr))),
            "p16": float(np.exp(np.percentile(lr, 16))),
            "p84": float(np.exp(np.percentile(lr, 84))),
            "min": float(np.exp(lr.min())), "max": float(np.exp(lr.max())),
            "zhalf_rms": float(np.sqrt((zh ** 2).mean()))}


def table(rows, lines):
    lines.append("| channel | E_k [GeV] | Z gen | stat. error | Z MCEq raw | "
                 "gen/MCEq | (gen-MCEq)/error |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        pull = (r["Zgen"] - r["Zmceq"]) / r["sig"] if r["sig"] > 0 \
            else float("nan")
        lines.append(f"| {K.MCEQ_NAME[r['parent']]} -> "
                     f"{K.MCEQ_NAME[r['child']]} | {r['E']:.3g} | "
                     f"{r['Zgen']:.4g} | {r['sig']:.2g} | {r['Zmceq']:.4g} | "
                     f"{r['ratio']:.3f} | {pull:+.1f} |")
    lines.append("")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="fresh1e5")
    ap.add_argument("--also", default="pool2000")
    ap.add_argument("--out", default=os.path.join(N3, "c0", "gates", "G0.3"))
    a = ap.parse_args(argv)
    t0 = time.time()
    eg_raw, n_raw = W._mceq_raw_numbers("PHYS")
    lines = ["# Gate G0.3: generator yields against MCEq's (reported)", "",
             "## Gate text (GATES.md)", "", gate_text("G0.3", "G0.4"), "",
             "## Result", "",
             "**REPORTED** (not gated). Command: `$PY tools/cascade3d/gates/"
             "g0_3.py " + " ".join(sys.argv[1:]) + "`", "",
             f"gamma = {GAMMA}; population PHYS; MCEq side: raw yields "
             "(`Interactions.get_matrix`), before the folding of mesons "
             "below their mixing energy. Errors: statistical error of the "
             "generator's Z, child bins treated as independent.", ""]
    res = {}
    full = {}
    for tag in (a.tag, a.also):
        if not tag:
            continue
        gen = K.load(tag, "PHYS")
        rows = z_table(gen, eg_raw, n_raw)
        full[tag] = rows
        res[tag] = {"summary": summary(rows, lines, "gen_" + tag),
                    "rows": rows}
    lines.append(f"## Full table, `gen_{a.tag}`\n")
    table(full[a.tag], lines)
    if a.also:
        lines.append(f"## Main channels, `gen_{a.also}` against "
                     f"`gen_{a.tag}`\n")
        lines.append("| channel | E_k [GeV] | Z " + a.also + " | Z " + a.tag
                     + " | ratio | difference / combined error |")
        lines.append("|---|---|---|---|---|---|")
        idx = {(r["parent"], r["child"], r["E"]): r for r in full[a.tag]}
        pulls = []
        for r in full[a.also]:
            o = idx.get((r["parent"], r["child"], r["E"]))
            if o is None or o["Zgen"] <= 0 or r["Zmceq"] < Z_MIN:
                continue
            s = np.hypot(r["sig"], o["sig"])
            pull = (r["Zgen"] - o["Zgen"]) / s if s > 0 else float("nan")
            pulls.append(pull)
            if r["parent"] in (2212, 2112, 211, -211, 321) and \
                    r["child"] in (211, -211, 321, -321, 2212, 2112):
                lines.append(f"| {K.MCEQ_NAME[r['parent']]} -> "
                             f"{K.MCEQ_NAME[r['child']]} | {r['E']:.3g} | "
                             f"{r['Zgen']:.4g} | {o['Zgen']:.4g} | "
                             f"{r['Zgen'] / o['Zgen']:.3f} | {pull:+.1f} |")
        pulls = np.array(pulls)
        pulls = pulls[np.isfinite(pulls)]
        lines.append("")
        lines.append(f"All channels with Z (MCEq) >= {Z_MIN:g}: pull mean "
                     f"{pulls.mean():+.3f}, rms "
                     f"{np.sqrt((pulls ** 2).mean()):.3f}, n = {pulls.size}, "
                     f"fraction beyond 3: {(np.abs(pulls) > 3).mean():.3f}.\n")
        res["pool_vs_fresh"] = {"mean": float(pulls.mean()),
                                "rms": float(np.sqrt((pulls ** 2).mean())),
                                "n": int(pulls.size)}
    lines.append(f"Run time {time.time() - t0:.0f} s.")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out + ".md", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(a.out + ".json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()
