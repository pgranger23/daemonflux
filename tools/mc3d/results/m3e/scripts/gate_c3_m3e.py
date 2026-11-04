"""Redefined gate C3 (milestone 3e), part 1: the MC's own bookkeeping test.

For every 1D entry of a run's kept c-files, ``t`` = no-cutoff weight, ``v`` =
the weight with the traced cutoff verdict.  The sharp prediction is
``v_pred = t * F_bin(R_primary)``, ``F_bin(R)`` = the fraction of directions in
the bin whose site-map R_c (``cutoff.SiteCutoffMap``) is <= the primary's
nucleus rigidity (a 5 x (6 or 24) sub-grid of cos Z, azimuth).  v/t vs v_pred/t
must agree at a few % (the traced verdict differs from the sharp R_c only by
the penumbra and the exact entry geometry).  Bins: vertical (cos Z 0.9-1, all
az), 87 deg W (az 240-270, 270-300), 87 deg E (60-90, 90-120); three E bands;
per species; jackknife G = 20 over the run's tasks as ``analyse_sky.JackRun``.

    python gate_c3_m3e.py RUNDIR TAG   -> $M/reports/gate_c3_bookkeeping_<TAG>.txt
"""
import os
import pickle
import sys

import numpy as np

import sky
import sky3d
import primaries as prim

rundir, tag = sys.argv[1], sys.argv[2]
G = 20
M = os.environ["M"]
BINS = (("vertical", 19, None), ("87W 240-270", 10, 8), ("87W 270-300", 10, 9),
        ("87E 60-90", 10, 2), ("87E 90-120", 10, 3))
EB = sky3d.MUB_EBANDS
NE, NCZ, NAZ = len(sky.E_BINS) - 1, len(sky.CZ_BINS) - 1, len(sky.AZ_BINS) - 1
GID = {g: (a_, z_) for g, a_, z_, _ in prim.SPECIES}
man = dict(np.load(os.path.join(sky3d.task_dir(rundir, tag), "manifest.npz")))
ntask = int(os.environ.get("GATE_NMAX") or man["ntask"][0])
os_ = man["opts"]
import cutoff
from datetime import datetime
opts = sky3d.Options(cutoff=bool(os_[0]), n_az=int(os_[4]))
lat, lon = sky.SITES[str(man["site"][0])]
smap = sky3d.site_map(lat, lon, opts.model, opts.date, 1, os.path.join(M, "work"))


def rc_grid(izb, iab):
    cz_lo = -1.0 + (izb) * (2.0 / NCZ)
    czs = cz_lo + (np.arange(5) + 0.5) / 5 * (2.0 / NCZ)
    if iab is None:
        azs = (np.arange(24) + 0.5) * 360.0 / 24
    else:
        azs = sky.AZ_BINS[iab] + (np.arange(6) + 0.5) / 6 * (sky.AZ_BINS[iab + 1] - sky.AZ_BINS[iab])
    return np.array([float(smap(np.degrees(np.arccos(c)), a_)) for c in czs for a_ in azs])


RC = [np.sort(rc_grid(izb, iab)) for _l, izb, iab in BINS]
print("R_c quantiles per bin (min, median, max):",
      [(round(r[0], 1), round(float(np.median(r)), 1), round(r[-1], 1)) for r in RC], flush=True)

sids, pos, mg, acc = {}, {}, {}, {}
for i in range(ntask):
    with open(sky3d.cfile(rundir, tag, i), "rb") as fh:
        d = pickle.load(fh)
    gname, k = d["group"], int(d["k"])
    sid = sids.setdefault((gname, k), len(sids))
    j = pos.get(sid, 0)
    pos[sid] = j + 1
    key = (j % G, sid)
    mg[key] = mg.get(key, 0.0) + float(d["meta"][0])
    x = acc.setdefault(key, np.zeros((3, len(BINS), 4, len(EB))))   # t, v, v_pred
    e1 = d["ent"]["1d"]
    if len(e1["flat"]) == 0:
        continue
    R = np.asarray(d["R"], float)
    flat = np.asarray(e1["flat"], np.int64)
    ia = flat % NAZ
    iz = (flat // NAZ) % NCZ
    ie = (flat // (NAZ * NCZ)) % NE
    sp = (flat // (NAZ * NCZ * NE)) % 4
    ib = np.full(len(flat), -1)
    for b, (i0, i1) in enumerate(EB):
        ib[(ie >= i0) & (ie < i1)] = b
    prim_ = np.asarray(e1["prim"], np.int64)
    inv = np.asarray(e1["inv"], float)
    for nb, (_l, izb, iab) in enumerate(BINS):
        sel = (ib >= 0) & (iz == izb)
        if iab is not None:
            sel &= ia == iab
        if not sel.any():
            continue
        wt = np.asarray(e1["wt"], float)[sel] * inv[sel]
        wv = np.asarray(e1["wv"], float)[sel] * inv[sel]
        F = np.searchsorted(RC[nb], R[prim_[sel]], side="right") / len(RC[nb])
        for q, w in enumerate((wt, wv, wt * F)):
            np.add.at(x[q, nb], (sp[sel], ib[sel]), w)
K = len(sids)
S = np.zeros((G, K, 3, len(BINS), 4, len(EB)))
for (g, s), v in acc.items():
    S[g, s] = v
Mg = np.zeros((G, K))
for (g, s), m in mg.items():
    Mg[g, s] = m
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "m3c", "scripts"))
import m3c_lib as L
tot = S.sum((0, 1))
rep = L.replicates(S, Mg)
lines = [f"# gate C3 bookkeeping, run {tag}: v/t (traced cutoff) vs v_pred/t (sharp site-map R_c at the primary's nucleus rigidity)",
         "bin           species   band   v/t +- err     v_pred/t +- err    v/v_pred +- err (paired jackknife)"]
rows = []
for nb, (lab, _a, _b) in enumerate(BINS):
    for k, spn in enumerate(L.SP):
        for b, E in enumerate(L.EB):
            t, v, p = tot[:, nb, k, b]
            if t <= 0 or p <= 0:
                continue
            rv = rep[:, 1, nb, k, b] / rep[:, 0, nb, k, b]
            rp = rep[:, 2, nb, k, b] / rep[:, 0, nb, k, b]
            rr = rep[:, 1, nb, k, b] / rep[:, 2, nb, k, b]
            f = lambda r: np.sqrt((G - 1.0) / G * np.nansum((r - np.nanmean(r)) ** 2))
            rows.append((lab, spn, E, v / t, f(rv), p / t, f(rp), v / p, f(rr)))
            lines.append(f"{lab:12s} {spn:9s} {E:4.1f}  {v/t:.4f}+-{f(rv):.4f}  {p/t:.4f}+-{f(rp):.4f}   {v/p:.3f}+-{f(rr):.3f}")
r = np.array([x[7] for x in rows]); e = np.array([x[8] for x in rows])
lines.append(f"\nmean v/v_pred over {len(r)} cells: {r.mean():.3f}; median |v/v_pred - 1| {np.median(abs(r-1)):.3f}; "
             f"chi2/ndf (v/v_pred = 1, paired errors) {np.sum(((r-1)/e)**2):.1f}/{len(r)}; "
             f"cells beyond 5%: {(abs(r-1) > 0.05).sum()}")
open(f"{M}/reports/gate_c3_bookkeeping_{tag}.txt", "w").write("\n".join(lines) + "\n")
np.savez(f"{M}/work/gate_c3_{tag}.npz", S=S, M=Mg)
print("\n".join(lines[-6:]))
