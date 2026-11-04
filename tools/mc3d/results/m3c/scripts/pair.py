"""Task 3 (milestone 3c): paired re-anchor vs re-simulate.

``c3anc`` (r_thr 0: every cascade reused by rotation and re-anchored on each
copy's own exact bent primary) and ``c3res`` (r_thr 65: every copy of a
primary below 65 GV re-simulated with its own bent primary; above 65 GV
re-anchored as in c3anc), both with ``--crn``: same seed, tasks and
allocation (the pilot's Neyman allocation x 0.2), every realisation of a
primary drawn from the same random stream and the same first-interaction
depth.  So copy 0 is identical in both runs, above 65 GV every copy is
identical, and below 65 GV a re-simulated copy differs from the re-anchored
one only through the geometry (and the stream divergence it causes).

    python pair.py [anc_tag res_tag] -> $M/reports/pair.txt, pair.npz
"""
import os
import sys

import numpy as np

import m3c_lib as L

M = os.environ["M"]
ta, tr = (sys.argv[1:3] if len(sys.argv) > 2 else ("c3anc", "c3res"))
lines = []


def out(s=""):
    print(s, flush=True)
    lines.append(s)


A, B = L.Run(ta), L.Run(tr)
cpu = {t: float(R.d["cpu"].sum()) / 3600 for t, R in ((ta, A), (tr, B))}
out(f"# task 3: {tr} (re-simulate below 65 GV) - {ta} (re-anchor), paired "
    f"(--crn); {A.d['nsh'].sum():.4g} primaries each; CPU {ta} {cpu[ta]:.2f} "
    f"core-h, {tr} {cpu[tr]:.2f} core-h")
res = {}
for pooled in (False, True):
    oa, ob = A.obs("s", pooled), B.obs("s", pooled)
    d, de = L.diff(ob, oa)
    ea, eb = L.jk_err(oa[1]), L.jk_err(ob[1])
    lab = "pooled cos Z 0-0.3 (12 values)" if pooled else "36 cells"
    out(f"\n## {lab}: d = a1/a0({tr}) - a1/a0({ta}), paired jackknife; "
        "unpaired error sqrt(e_a^2 + e_b^2) for comparison")
    x, xe = d[..., 0], de[..., 0]
    unp = np.sqrt(ea[..., 0] ** 2 + eb[..., 0] ** 2)
    if pooled:
        for ib, E in enumerate(L.EB):
            out(f"E {E}: " + "  ".join(
                f"{sp} {oa[0][k, ib, 0]:.3f}/{ob[0][k, ib, 0]:.3f} d {x[k, ib]:+.4f}"
                f"+-{xe[k, ib]:.4f} (unpaired {unp[k, ib]:.3f})"
                for k, sp in enumerate(L.SP)))
    else:
        for ib, E in enumerate(L.EB):
            for iz, cz in enumerate(((0, .1), (.1, .2), (.2, .3))):
                out(f"E {E} cz {cz[0]:.1f}-{cz[1]:.1f}: " + "  ".join(
                    f"{sp} d {x[k, ib, iz]:+.4f}+-{xe[k, ib, iz]:.4f} "
                    f"(unp {unp[k, ib, iz]:.3f})" for k, sp in enumerate(L.SP)))
    # the mean difference with its jackknife error (paired replicates)
    dr = ob[1][..., 0] - oa[1][..., 0]
    mean = float(np.nanmean(x))
    mean_e = float(L.jk_err(np.nanmean(dr.reshape(dr.shape[0], -1), 1)))
    c2, n = L.chi2(x, xe)
    out(f"mean d {mean:+.4f} +- {mean_e:.4f}; rms d {np.sqrt(np.nanmean(x ** 2)):.4f}; "
        f"chi2(d=0)/{n} = {c2 / n:.2f}; median paired error {np.nanmedian(xe):.4f} "
        f"(median unpaired {np.nanmedian(unp):.4f}, gain x{np.nanmedian(unp / xe):.1f})")
    # phase too
    out(f"phase dphi: mean d {np.nanmean(d[..., 1]):+.2f} deg, chi2/{n} = "
        f"{L.chi2(d[..., 1], de[..., 1])[0] / n:.2f}")
    res["pooled" if pooled else "cells"] = (d, de, unp, mean, mean_e, c2, n)
np.savez(M + "/reports/pair.npz",
         d_cells=res["cells"][0], e_cells=res["cells"][1],
         unp_cells=res["cells"][2], d_pooled=res["pooled"][0],
         e_pooled=res["pooled"][1], unp_pooled=res["pooled"][2],
         mean_cells=res["cells"][3:5], mean_pooled=res["pooled"][3:5],
         cpu=np.array([cpu[ta], cpu[tr]]))
open(M + "/reports/pair.txt", "w").write("\n".join(lines) + "\n")
