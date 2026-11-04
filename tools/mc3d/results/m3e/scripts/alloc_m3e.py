"""Neyman allocation for pilot-2 serving the 3D amplitudes (12 pooled a1/a0, tag s)
AND the 1D references (vertical cos Z 0.9-1 rates of t and v, 4 species x 3 bands),
with a floor per (group, stratum).

    python alloc_m3e.py STRATA.npz NTOTAL OUT.npz [cost_scale]

V_hq = jackknife variance of observable q when only stratum h is allowed to
vary; objective sum_q sum_h V_hq / V_q  (relative variances);
n_h ~ sqrt(a_h / c_h), a_h = n_h sum_q V_hq / V_q (as analyse_m3b.allocation).
Floors: n_h >= 300 expected *3D-scoring* primaries (the pilot's rate per
stratum) and >= 0.2% of the total.  Cost c_h: pilot CPU/primary x 0.25
(re-anchored) x 2.4 for p 32-100 GeV/n (hybrid re-simulation window)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "m3c", "scripts"))
import analyse_m3b as a
import m3c_lib as L

d = dict(np.load(sys.argv[1])); NT = float(sys.argv[2]); out = sys.argv[3]
S, M = d["ew"], d["M"]; V = d["vert"]
G, K = S.shape[:2]
strata = [x.split(":") for x in d["strata"]]
def obs_s(ew):
    x = ew[0].sum(2)                     # (sp, band, az)
    return np.array([[a.observables(x[k, b], L.axis())[0] for b in range(3)] for k in range(4)]).ravel()
def obs_tv(v):
    return np.concatenate([v[2].ravel(), v[3].ravel()])
def rep_ps(X):                           # leave-one-group-out per stratum (G,K,...)
    return L.replicates(X, M, per_stratum=True)
rS, rV = rep_ps(S), rep_ps(V)
totS, totV = S.sum((0, 1)), V.sum((0, 1))
def jv(vals):
    vals = np.array(vals); G_ = len(vals)
    return (G_ - 1.0) / G_ * ((vals - vals.mean(0)) ** 2).sum(0)
Vq = [jv([obs_s(totS - S[:, h].sum(0) * 0 - 0 + 0) for h in [0]])]   # placeholder (overwritten)
# total variance with all strata varying
fullS = L.replicates(S, M); fullV = L.replicates(V, M)
Vtot_s = jv([obs_s(r) for r in fullS]); Vtot_tv = jv([obs_tv(r) for r in fullV])
a_h = np.zeros(K); contrib = np.zeros((K, 2))
for h in range(K):
    base_s, base_v = S[:, h].sum(0), V[:, h].sum(0)
    vs = jv([obs_s(totS - base_s + rS[g, h]) for g in range(G)])
    vv = jv([obs_tv(totV - base_v + rV[g, h]) for g in range(G)])
    with np.errstate(invalid="ignore", divide="ignore"):
        cs = np.nansum(np.where(Vtot_s > 0, vs / Vtot_s, 0.0)) / len(Vtot_s)
        cv = np.nansum(np.where(Vtot_tv > 0, vv / Vtot_tv, 0.0)) / len(Vtot_tv)
    contrib[h] = cs, cv
    a_h[h] = d["nsh"][h] * (cs + cv)
cost = d["cpu"] / d["nsh"] * 0.25
for h, (g, k) in enumerate(strata):
    if g == "p" and k == "3":
        cost[h] *= 2.4
w = np.sqrt(np.maximum(a_h, 0) / cost)
frac = w / w.sum()
# floors
dg = dict(np.load(os.path.join(os.environ["M"], "work", "rwdiag_pilot_diag.npz")))
pend_rate = np.zeros(K)
for h, (g, k) in enumerate(strata):
    m = (dg["label"] == dg["label"][0]) & (dg["group"] == g) & (dg["k"] == int(k))
    pend_rate[h] = dg["n3"][m].sum() / d["nsh"][h]
floor_n = np.maximum(300.0 / np.maximum(pend_rate, 1e-4), 0.002 * NT)
n = np.maximum(frac * NT, floor_n)
n = n * NT / n.sum() if n.sum() > NT * 1.2 else n
for _ in range(5):                      # re-normalise the free strata
    fixed = frac * NT < floor_n
    free = ~fixed
    n = np.where(fixed, floor_n, frac * (NT - floor_n[fixed].sum()) / frac[free].sum())
n = np.round(n).astype(int)
np.savez(out, group=np.array([g for g, k in strata]), stratum=np.array([int(k) for g, k in strata]), n=n)
print("total", n.sum(), "pilot", int(d["nsh"].sum()), "est CPU core-h", float((n * cost).sum() / 3600))
print("group k  n_new  n_pilot  pend_rate  floor  contrib_s contrib_tv")
for h, (g, k) in enumerate(strata):
    print(f"{g:5s}{k:>2s} {n[h]:8d} {int(d['nsh'][h]):8d} {pend_rate[h]:8.3f} {int(floor_n[h]):7d} {contrib[h,0]:.4f} {contrib[h,1]:.4f}")
