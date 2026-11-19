"""Gate G1.4 (kernel statistics in the flux); writes G1.4.md / G1.4.json.

Two solves with the kernel sets of the two disjoint halves of the 10^5-event samples
(``ks_mc_fresh1e5_h0_L32_losson.npz``, ``..._h1_...``), otherwise the frozen G1.3(a)
configuration (``run_solve.py MC losson --r-earth 6371 --ks ...``).  The G1.3 cells
are the cell averages ``cell_<species>`` of the two files (four neutrino species, ten
bands 0.1-10 GeV, ten cos Z bins 0-1).

Measure (fixed before the run): the relative difference of a cell,
d = (Phi_h1 - Phi_h0) / ((Phi_h0 + Phi_h1) / 2); PASS if |d| <= 0.5% in every cell.
Also reported, not gated: Phi_h1 / Phi_h0 - 1 (differs from d at second order), and,
when the full 10^5 solve is given, each half against it.  No kernel regeneration is
started by this script.

Usage::

    $PY tools/cascade3d/gates/g1_4.py --h0 H0.npz --h1 H1.npz [--full FULL.npz]
        [--outdir DIR]
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from g1_3 import SPECIES, jsonable, sha  # noqa: E402

TOL = 0.005
GATE = """**G1.4 Kernel statistics in the flux (gated).** Kernels from two disjoint halves of the
10^5-event samples give fluxes within 0.5% in every G1.3 cell. If not, the nodes responsible
are regenerated at 10^6 events and the gate is re-run."""


def cells(path):
    z = np.load(path, allow_pickle=False)
    v = np.stack([np.asarray(z[f"cell_{q}"], float) for q in SPECIES])
    return v, np.asarray(z["cell_band_edges"]), np.asarray(z["cell_cosz_edges"]), \
        json.loads(str(z["cost_json"]))


def evaluate(a, b, tol=TOL):
    """``a``, ``b`` (4, nb, nz): the symmetric relative difference and the verdict."""
    d = (b - a) / (0.5 * (a + b))
    ok = np.abs(d) <= tol
    return {"d": d, "ratio_m1": b / a - 1.0, "inside": ok, "n_cells": int(d.size),
            "n_inside": int(ok.sum()), "pass": bool(ok.all()),
            "max_abs": float(np.max(np.abs(d)))}


def table(arr, be, ce, title, flag=None, fmt="+.4f"):
    L = []
    for i, q in enumerate(SPECIES):
        L += [f"{title}, {q} (rows: band [GeV]; columns: cos Z bin; bold = outside)",
              "",
              "| band | " + " | ".join(f"{ce[j]:.1f}-{ce[j + 1]:.1f}"
                                     for j in range(len(ce) - 1)) + " |",
              "|---|" + "---|" * (len(ce) - 1)]
        for k in range(len(be) - 1):
            row = []
            for j in range(len(ce) - 1):
                t = format(arr[i, k, j], fmt)
                if flag is not None and not flag[i, k, j]:
                    t = f"**{t}**"
                row.append(t)
            L.append(f"| {be[k]:.3g}-{be[k + 1]:.3g} | " + " | ".join(row) + " |")
        L.append("")
    return L


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--h0", required=True)
    ap.add_argument("--h1", required=True)
    ap.add_argument("--full", default="")
    ap.add_argument("--outdir", default=os.path.join(
        os.environ.get("N3", "/home/pigrange/mc3d_work/runs/native3d"), "c1", "g13"))
    ap.add_argument("--name", default="G1.4")
    args = ap.parse_args(argv)
    a, be, ce, ca = cells(args.h0)
    b, be1, ce1, cb = cells(args.h1)
    if not (np.array_equal(be, be1) and np.array_equal(ce, ce1)):
        print("INPUT ERROR: cell edges differ", file=sys.stderr)
        return 2
    res = evaluate(a, b)
    d = res["d"]
    order = np.argsort(-np.abs(d), axis=None)[:15]
    L = [f"# Gate {args.name}: kernel statistics in the flux", "", GATE, "",
         "Measure: d = (Phi_h1 - Phi_h0) / mean(Phi_h0, Phi_h1) per G1.3 cell; "
         f"PASS if |d| <= {TOL:.1%} in every cell.", "", "## Inputs", ""]
    files = {"h0": args.h0, "h1": args.h1}
    if args.full:
        files["full"] = args.full
    for k, p in files.items():
        c = json.loads(str(np.load(p)["cost_json"]))
        L.append(f"- {k}: `{p}` (sha256 {sha(p)[:16]}; kernel set `{c.get('ks')}`, "
                 f"R_E {c.get('r_earth')} km, straight {c.get('straight')}, "
                 f"{c['total_cpu_s'] / 60:.1f} core-min, {c['peak_rss_GB']:.2f} GB)")
    L += ["", f"## Result: **{'PASS' if res['pass'] else 'FAIL'}**", "",
          f"Cells inside: {res['n_inside']} of {res['n_cells']}; largest |d| "
          f"{res['max_abs']:.4%}.", "", "Largest cells:", "",
          "| species | band [GeV] | cos Z | d | h1/h0 - 1 |", "|---|---|---|---|---|"]
    for f in order:
        i, k, j = np.unravel_index(f, d.shape)
        L.append(f"| {SPECIES[i]} | {be[k]:.3g}-{be[k + 1]:.3g} | "
                 f"{ce[j]:.1f}-{ce[j + 1]:.1f} | {d[i, k, j]:+.4%} | "
                 f"{res['ratio_m1'][i, k, j]:+.4%} |")
    L += ["", "Largest |d| per species and band (over cos Z):", "",
          "| band [GeV] | " + " | ".join(SPECIES) + " |", "|---|" + "---|" * 4]
    mx = np.max(np.abs(d), axis=2)
    for k in range(len(be) - 1):
        L.append(f"| {be[k]:.3g}-{be[k + 1]:.3g} | " + " | ".join(
            (f"**{mx[i, k]:.4%}**" if mx[i, k] > TOL else f"{mx[i, k]:.4%}")
            for i in range(4)) + " |")
    L.append("")
    L += table(d, be, ce, "d", res["inside"])
    summary = {"pass": res["pass"], "tol": TOL, **res,
               "inputs": {k: {"path": p, "sha256": sha(p)} for k, p in files.items()}}
    if args.full:
        f, bef, cef, _ = cells(args.full)
        r0, r1 = a / f - 1, b / f - 1
        L += ["## Not gated: each half against the full 10^5 set", "",
              f"h0/full - 1: largest {np.max(np.abs(r0)):.4%}; h1/full - 1: largest "
              f"{np.max(np.abs(r1)):.4%}; (h0 + h1)/2 / full - 1: largest "
              f"{np.max(np.abs(0.5 * (a + b) / f - 1)):.4%}", ""]
        summary["h0_over_full_m1"], summary["h1_over_full_m1"] = r0, r1
    cmd = "$PY tools/cascade3d/gates/g1_4.py " + " ".join(
        argv if argv is not None else sys.argv[1:])
    L += ["## Command", "", f"`{cmd}`", ""]
    summary["command"] = cmd
    os.makedirs(args.outdir, exist_ok=True)
    md = os.path.join(args.outdir, f"{args.name}.md")
    with open(md, "w") as fh:
        fh.write("\n".join(L))
    with open(os.path.join(args.outdir, f"{args.name}.json"), "w") as fh:
        json.dump(jsonable(summary), fh, indent=1)
    print(f"{args.name}: {'PASS' if res['pass'] else 'FAIL'}; {res['n_inside']}/"
          f"{res['n_cells']} cells; largest |d| {res['max_abs']:.4%}; wrote {md}")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
