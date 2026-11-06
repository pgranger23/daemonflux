"""Re-optimised allocation for the production table (milestone 4), from pilot-2's per-stratum information.

Observables q: relative error of G = s/t (per species) in the COARSE down-going bins
  E 5/dec over 0.1-2.5 GeV (7 bins) x cos Z 5 bins of 0.2 (0-1) x az 6 bins.
Targets: 5% in the horizon bins (cos Z 0-0.3 -> coarse bins 0 (0-0.2) and 1 (0.2-0.4), taken whole), 3% elsewhere.
Model: Var_q(n) = sum_h a_hq / n_h  with a_hq = n_h^pilot * Var_hq(pilot) (jackknife variance of the ratio when only
stratum h varies, G = 20).  Allocation: n_h ~ sqrt( sum_q w_q a_hq / c_h ), w_q = 1/T_q^2 (Neyman for the weighted
sum of relative variances), cost c_h = pilot-2 CPU per primary (hybrid, measured) x 1.14 (tracing); floors: 300 expected
3D-scoring primaries and 2e-4 of the total per cell.  The overall scale N is the smallest for which the MEDIAN over
the bins of each class (horizon / other) of the per-species relative error meets its target.  Prints the prediction,
also for 90th percentiles.
    python alloc_m4.py OUT.npz
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "m3c", "scripts"))
import m3c_lib as L
M = os.environ["M"]
d = np.load(M + "/work/design_pilot2.npz")
S, Mg, cpu, nsh = d["x"], d["M"], d["cpu"], d["nsh"]            # (G,K,3,sp,E20,cz20,az12)
strata = [x.split(":") for x in d["strata"]]
G_, K = S.shape[:2]
NEF = 14                                                          # fine E bins 0.1-2.5 GeV
def coarse(x):                                                    # (..., sp, E20, cz20, az12) -> (..., sp, 7, 5, 6) down-going
    x = x[..., :NEF, 10:, :]
    s = x.shape
    x = x.reshape(s[:-3] + (7, 2, 5, 2, 6, 2))
    return x.sum((-1, -3, -5))
Sc = coarse(S)                                                    # (G,K,3,sp,7,5,6)
tot = Sc.sum((0, 1))
def Gof(x):                                                       # x (3,sp,7,5,6) -> s/t
    with np.errstate(divide="ignore", invalid="ignore"):
        return x[0] / x[1]
G0 = Gof(tot)
rep_ps = L.replicates(Sc.reshape(G_, K, -1), Mg, per_stratum=True)   # (G,K,flat)
rep_ps = rep_ps.reshape((G_, K) + Sc.shape[2:])
fullrep = L.replicates(Sc.reshape(G_, K, -1), Mg).reshape((G_,) + Sc.shape[2:])
def jv(vals):
    v = np.array(vals); n = len(v)
    return (n - 1.0) / n * ((v - v.mean(0)) ** 2).sum(0)
Vtot = jv([Gof(r) for r in fullrep])
good = np.isfinite(G0) & (G0 > 0) & np.isfinite(Vtot)
relvar_tot = np.where(good, Vtot / np.where(good, G0, 1) ** 2, np.nan)
a = np.zeros((K,) + G0.shape)
for h in range(K):
    base = Sc[:, h].sum(0)
    vh = jv([Gof(tot - base + rep_ps[g, h]) for g in range(G_)])
    with np.errstate(invalid="ignore", divide="ignore"):
        a[h] = np.where(good, vh / np.where(good, G0, 1) ** 2, 0.0) * nsh[h]
a = np.nan_to_num(a)
hor = np.zeros(G0.shape, bool); hor[..., :2, :] = True
ebin = np.zeros(G0.shape, int); ebin[..., 2:5, :, :] = 1; ebin[..., 5:7, :, :] = 2
CLASSES = [(eb_, h_) for eb_ in range(3) for h_ in (True, False)]
def class_mask(eb_, h_):
    return good & (G0 > 0) & (ebin == eb_) & (hor == h_)
cz_idx = np.arange(5)
T = np.full(G0.shape, 0.03)
T[..., :2, :] = 0.05                                              # horizon: coarse cos Z bins 0-0.2, 0.2-0.4 (axis -2)
cost = cpu / nsh * 1.14
def make_frac(cw):
    """Neyman fractions for class weights cw (dict class -> weight)."""
    w = np.zeros(G0.shape)
    for c, wc in cw.items():
        w = np.where(class_mask(*c), wc / (0.05 if c[1] else 0.03) ** 2, w)
    num = np.array([(w * a[h]).sum() for h in range(K)])
    f = np.sqrt(np.maximum(num, 0) / cost)
    return f / f.sum()
dg = dict(np.load(M + "/work/rwdiag_pilot2_42625.npz"))
rate = np.zeros(K)
lab0 = dg["label"][0]
for h, (g, k) in enumerate(strata):
    m = (dg["label"] == lab0) & (dg["group"] == g) & (dg["k"] == int(k))
    rate[h] = dg["n3"][m].sum() / nsh[h]
def rel_err(N, frac=None):
    n = np.maximum(frac * N, np.maximum(300.0 / np.maximum(rate, 1e-4), 2e-4 * N))
    var = sum(a[h] / n[h] for h in range(K))
    return n, np.sqrt(var)
hor = np.zeros(G0.shape, bool); hor[..., :2, :] = True
ebin = np.zeros(G0.shape, int); ebin[..., 2:5, :, :] = 1; ebin[..., 5:7, :, :] = 2     # 0.1-0.25 | 0.25-1 | 1-2.5 GeV
CLASSES = [(eb_, h_) for eb_ in range(3) for h_ in (True, False)]
def class_mask(eb_, h_):
    return good & (G0 > 0) & (ebin == eb_) & (hor == h_)
def medians(N, frac):
    n, e = rel_err(N, frac)
    meds = {c: np.median(e[class_mask(*c)]) for c in CLASSES}
    return n, meds, n @ cost / 3600
def ok(meds):
    return all(m <= (0.05 if c[1] else 0.03) for c, m in meds.items())
def solve(frac):
    lo, hi = 1e5, 5e8
    for _ in range(50):
        mid = np.sqrt(lo * hi)
        if ok(medians(mid, frac)[1]):
            hi = mid
        else:
            lo = mid
    return hi
cw = {c: 1.0 for c in CLASSES}
best = None
for it in range(40):
    fr = make_frac(cw)
    N = solve(fr)
    n, meds, ch = medians(N, fr)
    if best is None or ch < best[0]:
        best = (ch, N, fr, dict(cw))
    for c in CLASSES:        # raise the weight of the class that is closest to binding, lower the slack ones
        cw[c] *= (meds[c] / (0.05 if c[1] else 0.03)) ** 1.0
    tot_w = sum(cw.values()); cw = {c: v * len(cw) / tot_w for c, v in cw.items()}
ch, N, frac, cw = best
print("best class weights:", {c: round(v, 3) for c, v in cw.items()})
n, meds, ch = medians(N, frac)
n_, e_ = rel_err(N, frac)
n = np.round(n).astype(int)
np.savez(sys.argv[1], group=np.array([g for g, k in strata]), stratum=np.array([int(k) for g, k in strata]), n=n)
print(f"N total {n.sum():.4g} primaries; predicted CPU {ch:.1f} core-h (pilot-2 cost model x1.14 for tracing)")
names = ("0.1-0.25", "0.25-1", "1-2.5")
for c, m in meds.items():
    mm = class_mask(*c)
    print(f"  class E {names[c[0]]:9s} GeV {'horizon' if c[1] else 'other  '}: median {m*100:.2f}%  90th pct {np.percentile(e_[mm], 90)*100:.1f}%  ({mm.sum()} bins)")
print("group k   n_new    n_pilot   rate3D  cost[s/prim]")
for h, (g, k) in enumerate(strata):
    print(f"{g:5s}{k:>2s} {n[h]:9d} {int(nsh[h]):9d} {rate[h]:7.3f} {cost[h]:.3f}")
# model check at the pilot allocation
n_p = nsh.astype(float)
var_model = sum(a[h] / n_p[h] for h in range(K))
m = good & (G0 > 0)
for c in CLASSES:
    mm = class_mask(*c)
    print("  pilot-2 allocation, class", c, "jackknife %.2f%% model %.2f%%" % (100*np.median(np.sqrt(relvar_tot[mm])), 100*np.median(np.sqrt(var_model[mm]))))
print("check at pilot-2 allocation: median rel err  jackknife(total) %.2f%%  vs additive model %.2f%% (horizon: %.2f%% vs %.2f%%)" % (
    100 * np.median(np.sqrt(relvar_tot[m & ~hor])), 100 * np.median(np.sqrt(var_model[m & ~hor])),
    100 * np.median(np.sqrt(relvar_tot[m & hor])), 100 * np.median(np.sqrt(var_model[m & hor]))))
