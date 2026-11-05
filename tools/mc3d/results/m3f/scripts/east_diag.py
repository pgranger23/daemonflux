"""East 60-90 bookkeeping offset: where in R / R_c(bin) does v exceed the sharp prediction?"""
import os, sys, pickle
import numpy as np
sys.path.insert(0, os.environ["R"] + "/tools/mc3d")
import sky, sky3d, cutoff
M = os.environ["M"]; W = os.environ["W"]
run = W + "/runs/m3e/sky"; tag = "pilot2"
man = dict(np.load(sky3d.task_dir(run, tag) + "/manifest.npz"))
opts = sky3d.Options(model=None)
lat, lon = sky.SITES["kamioka"]
smap = sky3d.site_map(lat, lon, opts.model, opts.date, 1, W + "/runs/m3e/work")
NE, NCZ, NAZ = len(sky.E_BINS) - 1, len(sky.CZ_BINS) - 1, len(sky.AZ_BINS) - 1
iz_b, ia_b = 10, 2
zen_c = np.degrees(np.arccos(-1.0 + (iz_b + np.arange(5) + 0.5) / 5 * 0.1 * 1.0))  # placeholder, fixed below
czs = -1.0 + 0.1 * iz_b + (np.arange(5) + 0.5) / 5 * 0.1
def grid(nc, na):
    c = -1.0 + 0.1 * iz_b + (np.arange(nc) + 0.5) / nc * 0.1
    a = sky.AZ_BINS[ia_b] + (np.arange(na) + 0.5) / na * 30.0
    Z, A = np.meshgrid(np.degrees(np.arccos(c)), a, indexing="ij")
    return np.sort(np.asarray(smap(Z.ravel(), A.ravel())))
RC = {"5x6": grid(5, 6), "20x30": grid(20, 30)}
print("R_c sub-grid mean/min/max:", {k: (round(v.mean(), 2), round(v.min(), 1), round(v.max(), 1)) for k, v in RC.items()})
print("map grid: n_zen", len(smap.zen), "zen", smap.zen[-6:], "n_az", len(smap.az) - 1, "step", smap.az[1] - smap.az[0])
rbins = np.array([0, .5, .7, .8, .9, .95, 1.0, 1.05, 1.1, 1.2, 1.5, 2.0, 4, 100])
rmean = RC["5x6"].mean()
H = np.zeros((3, len(rbins) - 1))   # t, v, vpred(5x6)
H2 = 0.0
tot = np.zeros(4)
CAP = np.zeros((4, 3))
files = [i for i in range(int(man["ntask"][0]))]
n = 0
for i in files:
    try:
        d = pickle.load(open(sky3d.cfile(run, tag, i), "rb"))
    except Exception:
        continue
    if d["group"] != "p" or int(d["k"]) not in (3, 4):
        continue
    if n >= int(os.environ.get('EAST_NMAX', 10**9)):
        break
    n += 1
    e1 = d["ent"]["1d"]
    if len(e1["flat"]) == 0:
        continue
    flat = np.asarray(e1["flat"], np.int64)
    ia = flat % NAZ; iz = (flat // NAZ) % NCZ; ie = (flat // (NAZ * NCZ)) % NE
    i0 = flat // (4 * NE * NCZ * NAZ)
    sel = (iz == iz_b) & (ia == ia_b) & (ie >= 12) & (ie < 22)
    if not sel.any():
        continue
    R = np.asarray(d["R"], float)[np.asarray(e1["prim"])[sel]]
    inv = np.asarray(e1["inv"], float)[sel]
    wt = np.asarray(e1["wt"], float)[sel] * inv; wv = np.asarray(e1["wv"], float)[sel] * inv
    F1 = np.searchsorted(RC["5x6"], R, side="right") / len(RC["5x6"])
    F2 = np.searchsorted(RC["20x30"], R, side="right") / len(RC["20x30"])
    ib = np.searchsorted(rbins, R / rmean, side="right") - 1
    for q, w in enumerate((wt, wv, wt * F1)):
        np.add.at(H[q], ib, w)
    tot += [wt.sum(), wv.sum(), (wt * F1).sum(), (wt * F2).sum()]
    for c in range(4):
        m_ = i0[sel] <= c
        CAP[c] += [wt[m_].sum(), wv[m_].sum(), (wt * F1)[m_].sum()]
print("c-files used", n)
print("totals t, v, vpred(5x6), vpred(20x30):", tot, " v/vp(5x6) %.4f  v/vp(20x30) %.4f" % (tot[1] / tot[2], tot[1] / tot[3]))
print("R/<R_c> bin      t_share   v/t     vpred/t   (v-vp)/t_total")
for k in range(len(rbins) - 1):
    if H[0, k] > 0:
        print(f"{rbins[k]:5.2f}-{rbins[k+1]:5.2f}   {H[0,k]/tot[0]:.4f}  {H[1,k]/H[0,k]:.3f}  {H[2,k]/H[0,k]:.3f}   {(H[1,k]-H[2,k])/tot[0]:+.4f}")

print('cumulative caps (theta_D 2.5/5/7.5/10 deg): v/vpred', [round(c[1]/c[2],4) for c in CAP], ' v/t', [round(c[1]/c[0],4) for c in CAP], ' t share', [round(c[0]/CAP[3,0],3) for c in CAP])
