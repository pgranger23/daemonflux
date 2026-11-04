"""Gate C3 (milestone 3c): the 1D tallies ``t`` (no cutoff) and ``v`` (site
cutoff) of a run, recomputed from its kept c-files with per-primary weights
under alternative primary-flux conventions.  Exact: every pending primary's
nucleus rigidity is in the c-file, so its kinetic energy per nucleon is
``T = sqrt((R Z/A)^2 + m_p^2) - m_p`` and its weight can be re-scaled.

Weights:
  orig      as run (H3a evaluated at A*T: ``primaries.nucleon_intensity``)
  mceq      x Phi(A (T + m)) / Phi(A T): crflux at the total energy, as MCEq
  mceq_t2   mceq and T >= 1.995 GeV (MCEq's lowest primary bin edge,
            ``minimal_primary_energy`` = 3 GeV total)

Bins: vertical (cos Z 0.9-1, all az), 87W 240-270, 87W 270-300 (cos Z 0-0.1),
87E 60-90, 87E 90-120; measure 1; the three E bands; per (jackknife group,
stratum) as ``analyse_sky.JackRun``; plus histograms in T (MCEq's bin edges).

    python c3_reweight.py RUNDIR TAG -> $M/work/rew_<TAG>.npz
"""
import os
import pickle
import sys

import numpy as np

import crflux.models as crf
import sky
import sky3d

rundir, tag = sys.argv[1], sys.argv[2]
G = 20
MP, MN = 0.938272, 0.939565
MBAR = 0.5 * (MP + MN)
H3A = crf.HillasGaisser2012("H3a")
GID = {"p": (1, 1, 14), "He": (4, 2, 402), "CNO": (14, 7, 1407),
       "MgSi": (25, 12, 2512), "Fe": (56, 26, 5626)}
BINS = (("vertical", 19, None), ("87W 240-270", 10, 8), ("87W 270-300", 10, 9),
        ("87E 60-90", 10, 2), ("87E 90-120", 10, 3))
EB = sky3d.MUB_EBANDS
TEDGES = 10 ** (np.arange(0, 41) / 10.0)            # 1 GeV .. 10 TeV, MCEq's
VARS = ("orig", "mceq", "mceq_t2")
NE, NCZ, NAZ = len(sky.E_BINS) - 1, len(sky.CZ_BINS) - 1, len(sky.AZ_BINS) - 1

man = dict(np.load(os.path.join(sky3d.task_dir(rundir, tag), "manifest.npz")))
ntask = int(man["ntask"][0])
sids, pos = {}, {}
# acc[var][(g, sid)] = (tv=2, bin, sp, band); hist[var] = (tv, bin, sp, band, T)
acc = {v: {} for v in VARS}
hist = {v: np.zeros((2, len(BINS), 4, len(EB), len(TEDGES) - 1)) for v in VARS}
mg = {}
for i in range(ntask):
    with open(sky3d.cfile(rundir, tag, i), "rb") as fh:
        d = pickle.load(fh)
    gname, k = d["group"], int(d["k"])
    sid = sids.setdefault((gname, k), len(sids))
    j = pos.get(sid, 0)
    pos[sid] = j + 1
    key = (j % G, sid)
    mg[key] = mg.get(key, 0.0) + float(d["meta"][0])
    a_, z_, gid = GID[gname]
    R = np.asarray(d["R"], float)
    pn = R * z_ / a_
    T = np.sqrt(pn * pn + MP * MP) - MP
    fac = np.ravel(H3A.nucleus_flux(gid, a_ * (T + MBAR))) / \
        np.maximum(np.ravel(H3A.nucleus_flux(gid, a_ * T)), 1e-300)
    pf = {"orig": np.ones_like(T), "mceq": fac, "mceq_t2": fac * (T >= 1.995)}
    e1 = d["ent"]["1d"]
    for v in VARS:
        acc[v].setdefault(key, np.zeros((2, len(BINS), 4, len(EB))))
    if len(e1["flat"]) == 0:
        continue
    flat = np.asarray(e1["flat"], np.int64)
    ia = flat % NAZ
    iz = (flat // NAZ) % NCZ
    ie = (flat // (NAZ * NCZ)) % NE
    sp = (flat // (NAZ * NCZ * NE)) % 4
    ib = np.full(len(flat), -1)
    for b, (i0, i1) in enumerate(EB):
        ib[(ie >= i0) & (ie < i1)] = b
    prim = np.asarray(e1["prim"], np.int64)
    it = np.clip(np.searchsorted(TEDGES, T[prim], side="right") - 1, 0,
                 len(TEDGES) - 2)
    inv = np.asarray(e1["inv"], float)
    for nb, (_lab, izb, iab) in enumerate(BINS):
        sel = (ib >= 0) & (iz == izb)
        if iab is not None:
            sel &= ia == iab
        if not sel.any():
            continue
        for v in VARS:
            f = pf[v][prim[sel]]
            for tv, wk in enumerate(("wt", "wv")):
                w = np.asarray(e1[wk], float)[sel] * inv[sel] * f
                np.add.at(acc[v][key], (tv, nb, sp[sel], ib[sel]), w)
                np.add.at(hist[v], (tv, nb, sp[sel], ib[sel], it[sel]), w)
    if i % 200 == 0:
        print(f"{i}/{ntask}", flush=True)
K = len(sids)
out = {"strata": np.array([f"{g}:{k}" for (g, k), s in
                           sorted(sids.items(), key=lambda x: x[1])]),
       "bins": np.array([b[0] for b in BINS]), "tedges": TEDGES}
Mg = np.zeros((G, K))
for (g, s), m in mg.items():
    Mg[g, s] = m
out["M"] = Mg
for v in VARS:
    S = np.zeros((G, K, 2, len(BINS), 4, len(EB)))
    for (g, s), x in acc[v].items():
        S[g, s] = x
    out[f"S|{v}"] = S
    out[f"hist|{v}"] = hist[v]
np.savez(os.path.join(os.environ["M"], "work", f"rew_{tag}.npz"), **out)
print("wrote", tag)
