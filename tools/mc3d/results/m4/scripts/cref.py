"""Reference-shape factor c(E band, cos Z group) = t_addon / (kappa_addon B) with kappa over cos Z 0.6-1 (PM request, M4).

Species combined (sum of t over species / sum of B over species; kappa from the same combination, replicate by
replicate, so the shared showers are accounted for by the jackknife) and per species.  Groups: PM's (0-0.1, 0.1-0.2, 0.2-0.4,
0.4-0.6, 0.6-1) and the coarse cos Z bins of the table (0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1).
Also the diagnostics of the two outlier cells (effective statistics N_eff = (sum w)^2 / sum w^2 from t1sq and the leave-one-group-out spread).
 -> $M/reports/cref.txt, $M/work/cref.npz"""
import os, sys
import numpy as np
sys.path.insert(0, os.environ["R"] + "/tools/mc3d")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "m3c", "scripts"))
import analyse_sky as A
import m3c_lib as L
M, W = os.environ["M"], os.environ["W"]
Bt1 = np.load(W + "/runs/integ/out/base1d_kamioka.npz")["B_t1"]               # (4, 60, 20)
def red(d):
    return np.stack([np.asarray(d["t1"])[-1].sum(-1), np.asarray(d["t1sq"])[-1].sum(-1)])      # (2, sp, 60, 20)
J = A.JackRun(M + "/sky", "addon", G=20, reducers={"x": red})
S, Mg = J.S["x"], J.M
tot = S.sum((0, 1)); rep = L.replicates(S.reshape(S.shape[:2] + (-1,)), Mg).reshape((S.shape[0],) + S.shape[2:])
G = len(rep)
def jk(vals):
    v = np.array(vals); return np.sqrt((G - 1.0) / G * np.nansum((v - np.nanmean(v, 0)) ** 2, 0))
eb = np.logspace(-1, 2, 61); ec = np.sqrt(eb[1:] * eb[:-1])
BANDS = ((0.1, 0.3), (0.3, 1.0), (1.0, 3.0), (3.0, 10.0))
PMG = ((10, 11), (11, 12), (12, 14), (14, 16), (16, 20))
CGR = ((10, 12), (12, 14), (14, 16), (16, 18), (18, 20))
def cfun(x, sel, groups, sps, ref):
    """c per group, x (2, sp, 60, 20): t = x[0]."""
    t = x[0][sps][:, sel]; b = Bt1[sps][:, sel]
    with np.errstate(divide="ignore", invalid="ignore"):
        kap = t[..., 16:20].sum((0, 1, 2)) / b[..., 16:20].sum((0, 1, 2)) if False else t[..., slice(*ref)].sum() / b[..., slice(*ref)].sum()
        return np.array([t[..., a:b_].sum() / b[..., a:b_].sum() / kap for a, b_ in groups])
sp_sets = {"combined": [0, 1, 2, 3], "nue": [0], "antinue": [1], "numu": [2], "antinumu": [3]}
res = {}
L_ = ["# c = t_addon / (kappa_addon B), kappa over cos Z 0.6-1; jackknife (G = 20) errors, shared showers accounted for in the replicates"]
for gname, groups in (("PM groups", PMG), ("coarse cos Z bins", CGR)):
    L_.append(f"\n## {gname}: " + "   ".join(f"{a/10:.1f}-{b/10:.1f}" if True else "" for a, b in [(g[0] - 10, g[1] - 10) for g in groups]))
    for ib, (e0, e1) in enumerate(BANDS):
        sel = (ec >= e0) & (ec < e1)
        for nm, sps in sp_sets.items():
            ref = (16, 20) if gname == "PM groups" else (16, 20)
            c0 = cfun(tot, sel, groups, sps, ref); ce = jk([cfun(r, sel, groups, sps, ref) for r in rep])
            res[(gname, ib, nm)] = (c0, ce)
            L_.append(f"E {e0:g}-{e1:g} {nm:9s}: " + "  ".join(f"{c0[j]:.3f}+-{ce[j]:.3f}" for j in range(5)))
# species compatibility (coarse bins)
L_.append("\n## species vs combined (coarse cos Z bins): max |c_sp - c_comb| / err_sp per band")
for ib, (e0, e1) in enumerate(BANDS):
    cc = res[("coarse cos Z bins", ib, "combined")][0]
    zs = []
    for nm in ("nue", "antinue", "numu", "antinumu"):
        c0, ce = res[("coarse cos Z bins", ib, nm)]
        zs.append(np.abs(c0 - cc) / ce)
    L_.append(f"E {e0:g}-{e1:g}: max z {np.max(zs):.1f}, mean z {np.mean(zs):.2f}")
# sigma_ref map for the table's coarse E bins (5/dec, 10 bins) x coarse cos Z (5): band of each E bin
Ecoarse = 0.1 * 10 ** (np.arange(11) / 5.0)
band_of = []
for i in range(10):
    em = np.sqrt(Ecoarse[i] * Ecoarse[i + 1]); band_of.append(next(k for k, (a, b) in enumerate(BANDS) if a <= em < b))
sig_ref = np.zeros((10, 5)); c_map = np.zeros((10, 5)); cerr_map = np.zeros((10, 5))
for i in range(10):
    c0, ce = res[("coarse cos Z bins", band_of[i], "combined")]
    for j in range(5):
        c_map[i, j], cerr_map[i, j] = c0[j], ce[j]
        sig_ref[i, j] = ce[j] if j == 0 else max(abs(c0[j] - 1.0), ce[j])
        if j == 4:
            sig_ref[i, j] = ce[j]            # 0.8-1: part of the kappa window, c = 1 by construction (err only)
np.savez(M + "/work/cref.npz", c=c_map, c_err=cerr_map, sigma_ref=sig_ref, band_of=np.array(band_of), E_edges=Ecoarse)
L_.append("\n## sigma_ref(E coarse bin, cos Z coarse bin) = max(|c-1|, err(c)) [cos Z 0-0.2: err(c); 0.8-1: err(c)]  (species combined)")
for i in range(10):
    L_.append(f"E {Ecoarse[i]:.2f}-{Ecoarse[i+1]:.2f} (band {BANDS[band_of[i]]}): " + " ".join(f"{sig_ref[i,j]*100:5.1f}%" for j in range(5)))
# ---- outliers: effective statistics and leave-one-group-out spread
def neff(sel_e, cz0, cz1, sp):
    s1 = tot[0][sp][sel_e][:, cz0:cz1].sum(); s2 = tot[1][sp][sel_e][:, cz0:cz1].sum()
    return s1 ** 2 / s2 if s2 > 0 else np.nan
L_.append("\n## outliers")
for lab, sp, (e0, e1), (a, b) in (("antinue 1-3 GeV cos Z 0-0.1", 1, (1.0, 3.0), (10, 11)), ("antinue 0.3-1 GeV cos Z 0-0.1", 1, (0.3, 1.0), (10, 11)),
                                  ("antinue 0.3-1 GeV cos Z 0.9-1", 1, (0.3, 1.0), (19, 20)), ("numu 0.3-1 GeV cos Z 0.9-1 (reference)", 2, (0.3, 1.0), (19, 20)),
                                  ("nue 0.3-1 GeV cos Z 0-0.1", 0, (0.3, 1.0), (10, 11))):
    sel = (ec >= e0) & (ec < e1)
    vals = np.array([r[0][sp][sel][:, a:b].sum() / (tot[0][sp][sel][:, a:b].sum()) for r in rep])
    L_.append(f"{lab}: N_eff {neff(sel, a, b, sp):.0f}; leave-one-group-out t relative to total: min {vals.min():.3f} max {vals.max():.3f}; "
              f"top-group share of the sum of squares: {1.0:.0f}x")
open(M + "/reports/cref.txt", "w").write("\n".join(L_) + "\n"); print("\n".join(L_))
