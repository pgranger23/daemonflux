"""Report of the sky tables of a location grid (cascade3d C2, "Reported with the
skies").

For the 25 tables of the 5-deg grid (h_ref = 0): sha256, statistics, cost, and the
singular values of the deficits of the locations (no solve).  The deficit of a
location is D_x(line node, sector, bin) of ``boundary.deficit_lines`` (GSFG:GH), as a
row of a matrix (25 x n); the singular values are given for the matrix itself and
for its rows minus their mean, relative to the largest, with the number of values
above 1e-2, 1e-3 and 1e-4 of the largest (the number of independent boundaries a set
of locations needs at that accuracy).  Columns are weighted equally (the deficit in
the units of phi_top).

Usage::

    taskset -c 12 $PY tools/cascade3d/gates/c2_skies.py --out SKIES.md
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
import boundary as B  # noqa: E402

N3 = os.environ.get("N3", "/home/pigrange/mc3d_work/runs/native3d")
KS = os.path.join(N3, "kernels", "mc_close", "ks_mc_pool2000_L32_losson.npz")


def main(argv=None):
    import kernelset as K

    ap = argparse.ArgumentParser()
    ap.add_argument("--sky-dir", default=os.path.join(N3, "c2", "solver", "sky"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--spacing", type=float, default=5.0)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists")
    ks = K.KernelSet.load(KS)
    names = list(ks.species)
    b0 = ks.phi_top[[names.index("p+"), names.index("n0")]]
    gw = B.GroupWeights(ks.e_bins)
    g = B.LocationGrid(a.spacing, 10.0)
    rows, stats = [], []
    for i in range(g.n):
        p = B.table_path(a.sky_dir, (g.x[i], g.y[i]), 0.0)
        t = B.SkyTable.load(p)
        s = t.summary()
        s.update({"x": float(g.x[i]), "y": float(g.y[i]), "file": os.path.basename(p),
                  "sha256": B._sha256(p), "seed": int(t.seed)})
        stats.append(s)
        rows.append(B.deficit_lines(t, gw, b0).ravel())
    X = np.array(rows)
    out = {}
    for name, Y in (("deficits", X), ("deficits minus mean", X - X.mean(0))):
        sv = np.linalg.svd(Y, compute_uv=False)
        r = sv / sv[0]
        out[name] = {"relative": r.tolist(),
                     "n_above": {e: int(np.sum(r > float(e)))
                                 for e in ("1e-2", "1e-3", "1e-4")}}
    L = [f"# Sky tables of the {a.spacing:g}-deg location grid (h_ref = 0)", "",
         "| x | y | file | sha256 | states | traced | allowed | core-s |",
         "|---|---|---|---|---|---|---|---|"]
    for s in stats:
        L.append(f"| {s['x']:g} | {s['y']:g} | `{s['file']}` | `{s['sha256']}` | "
                 f"{s['states']} | {s['traced']} | {s['allowed_fraction']:.4f} | "
                 f"{s['cpu_s']:.0f} |")
    tot = sum(s["cpu_s"] for s in stats)
    L += ["", f"Tracing CPU of the {g.n} tables (workers): {tot / 3600:.2f} core-h.",
          "",
          "## Singular values of the location deficits (relative to the largest)",
          ""]
    for name, v in out.items():
        L.append(f"- {name}: " + ", ".join(f"{x:.2e}" for x in v["relative"]))
        L.append(f"  - above 1e-2 / 1e-3 / 1e-4: {v['n_above']['1e-2']} / "
                 f"{v['n_above']['1e-3']} / {v['n_above']['1e-4']}")
    L += ["", "Command: `" + " ".join(sys.argv) + "`", ""]
    with open(a.out, "w") as fh:
        fh.write("\n".join(L))
    if os.path.exists(a.out[:-3] + ".json"):
        raise SystemExit(a.out[:-3] + ".json exists")
    with open(a.out[:-3] + ".json", "w") as fh:
        json.dump({"tables": stats, "svd": out}, fh, indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
