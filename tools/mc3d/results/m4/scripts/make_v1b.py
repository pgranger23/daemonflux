"""v1b of the table: E 5/dec (0.1-10 GeV, 10 bins) x cos Z 0.1 (10 down-going bins) x az 12 x 30 deg, from $M/work/B_design.npz
(per-(group, stratum) sums, E 10/dec x 10 cz x 12 az); same definition as v1 (G = s / (kappa B), kappa = t/B over cos Z 0.6-1),
sigma_ref inherited from the 0.2-wide parent bin.  Also the PM's comparisons (sec. 11 of MILESTONE4_RESULTS.md).
   python make_v1b.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.environ["R"] + "/tools/mc3d")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "m3c", "scripts"))
import analyse_m3b as a
import m3c_lib as L
import sky
M, W = os.environ["M"], os.environ["W"]
d = np.load(M + "/work/B_design.npz"); S, Mg = d["x"], d["M"]                  # (G,K,4,sp,20,10,12) tags s,n,t,v
v1 = dict(np.load(M + "/out/kamioka_G_down_v1.npz"))
Bt1 = np.load(W + "/runs/integ/out/base1d_kamioka.npz")["B_t1"]                # (4,60,20)
cref = np.load(M + "/work/cref.npz")
def e5(x):                                                                   # merge E 10/dec pairs -> 5/dec: axis -3
    s = x.shape; return x.reshape(s[:-3] + (s[-3] // 2, 2) + s[-2:]).sum(-3)
S5 = e5(S)                                                                   # (G,K,4,sp,10,10,12)
Bc = Bt1[:, :40, 10:].reshape(4, 10, 4, 10).sum(2)[..., None] / 12.0 * np.ones(12)       # (sp,10 E5,10 cz,12 az)
Bk = Bt1[:, :40, 16:20].reshape(4, 10, 4, 4).sum(2)                          # (sp,10,4) for kappa
def kap(x):                                                                  # x (4tag,sp,10,10,12) totals
    return x[2][:, :, 6:10, :].sum((2, 3)) / Bk.sum(-1)                      # t over cos Z 0.6-1 (cz bins 6-9), all az
reps = L.replicates(S5.reshape(S5.shape[:2] + (-1,)), Mg).reshape((S5.shape[0],) + S5.shape[2:])
def Gof(x):
    return x[0] / (kap(x)[:, :, None, None] * Bc)
tot = S5.sum((0, 1)); G0 = Gof(tot)
Gr = np.array([Gof(r) for r in reps]); n = len(Gr)
Ge = np.sqrt((n - 1.0) / n * np.nansum((Gr - np.nanmean(Gr, 0)) ** 2, 0))
k0 = kap(tot); kr = np.array([kap(r) for r in reps]); ke = np.sqrt((n - 1.0) / n * np.nansum((kr - np.nanmean(kr, 0)) ** 2, 0))
sig = cref["sigma_ref"][:, np.arange(10) // 2]                               # (10 E5, 10 cz) from the 0.2 parent
Gt = np.sqrt(Ge ** 2 + (sig[None, :, :, None] * G0) ** 2)
out = {k: v for k, v in v1.items() if not (k.endswith("_fine") or k.endswith("_coarse"))}
out.update(E_edges=0.1 * 10 ** (np.arange(11) / 5.0), cosZ_edges=np.linspace(0, 1, 11), az_edges=np.arange(13) * 30.0,
           G=G0, G_err=Ge, G_err_total=Gt, kappa=k0, kappa_err=ke, sigma_ref=sig, s=tot[0], n=tot[1], t=tot[2], v=tot[3], B=Bc,
           note=np.array(["v1b: E 5/dec (0.1-10 GeV, bins 0-7 = 0.1-3.98 deliverable, 8-9 blend check) x cos Z 0.1 (10 down-going bins) x az 12 x 30 deg; same definition as v1 "
                          "(G = s/(kappa B), not clipped; kappa = t/B over cos Z 0.6-1; sigma_ref of the 0.2-wide parent bin; G_err_total = sqrt(G_err^2 + (sigma_ref G)^2))."]))
np.savez(M + "/out/kamioka_G_down_v1b.npz", **out)
L_ = []
def P(s=""): print(s); L_.append(s)
# ---- (a) medians
reg = {"horizon cos Z 0-0.4": range(0, 4), "mid 0.4-0.8": range(4, 8), "vertical 0.8-1": range(8, 10)}
bands = {"0.1-0.25": [0, 1], "0.25-1": [2, 3, 4], "1-2.5": [5, 6], "2.5-4": [7], "4-10 (blend)": [8, 9]}
P("(a) median relative error of G (stat -> total), v1b grid (E 5/dec x cos Z 0.1 x az 12; per species and bin)")
for r, ix in reg.items():
    ix = list(ix); cells = []
    for b, ie in bands.items():
        st = (Ge / G0)[:, ie][:, :, ix]; tt = (Gt / G0)[:, ie][:, :, ix]; ok = np.isfinite(st) & (G0[:, ie][:, :, ix] > 0)
        cells.append(f"{b}: {np.median(st[ok])*100:.1f}->{np.median(tt[ok])*100:.1f}%")
    P(f"  {r:20s} " + "  ".join(cells))
# ---- (b) observables
IDX = [0, 1, 5, 6, 7]; NM = ("a1/a0", "dphi", "N/S", "W/E")
def obs_pat(f):
    o = a.observables(f, L.axis())
    # N/S from the analyse_m3b definition: fN / fS
    return o[0], o[1], o[7]
def we(f):                                   # W/E from the 12-bin pattern in tally order: mean of the two bins centred nearest 270 and 90
    c = np.array(a.AZ_C)
    iw = [int(np.argmin(abs(((c - 270 + 180) % 360) - 180)))]; ie_ = [int(np.argmin(abs(((c - 90 + 180) % 360) - 180)))]
    return f[iw].mean() / f[ie_].mean()
def honda_pat(path, e0, e1, cz0):
    h = dict(np.load(path)); lE = np.log(h["E"]); c = np.array(a.AZ_C)
    az_c = (180.0 - (h["azlo"].astype(float) + 15.0)) % 360.0
    order = [int(np.argmin(np.abs(((az_c - ci + 180) % 360) - 180))) for ci in a.AZ_C]
    x = np.logspace(np.log10(e0), np.log10(e1), 81); icz = int(np.argmin(np.abs(h["czlo"] - cz0)))
    out_ = []
    for sp in a.SP_NAMES:
        tab = h[a.HONDA_KEY[sp]]
        out_.append([np.trapezoid(np.exp(np.interp(np.log(x), lE, np.log(np.maximum(tab[icz, j], 1e-300)))), x) for j in order])
    return np.array(out_)                                                    # (4, 12)
HMAX = W + "/runs/m3f/honda/honda_kam_solmax.npz"; HMIN = R + "/tools/mceq3d/honda_kam.npz" if (R := os.environ["R"]) else None
Eedges = out["E_edges"]
P("\n(b) azimuth observables straight from the G array (12 bins) next to Honda solmax (solmin) -- same E bin and 0.1-wide cos Z bin; E 5/dec bins containing 0.3 / 0.5 / 1 GeV")
P("    columns: MC G | Honda solmax | (Honda solmin); N/S = fN/fS (analyse_m3b), W/E = bin(270)/bin(90)")
rows = []
for icz, cz0 in enumerate((0.0, 0.1, 0.2)):
    for E, ie in ((0.3, 2), (0.5, 3), (1.0, 4)):
        for k, sp in enumerate(a.SP_NAMES):
            f = G0[k, ie, icz]; hx = honda_pat(HMAX, Eedges[ie], Eedges[ie + 1], cz0)[k]; hn = honda_pat(HMIN, Eedges[ie], Eedges[ie + 1], cz0)[k]
            om, ox, on_ = obs_pat(f), obs_pat(hx), obs_pat(hn)
            rr_ = np.array([obs_pat(g_[k, ie, icz]) for g_ in Gr]); dv = rr_ - rr_.mean(0); dv[:, 1] = (dv[:, 1] + 180) % 360 - 180
            erra = np.sqrt((n - 1.0) / n * (dv ** 2).sum(0))                 # jackknife errors of (a1/a0, dphi, N/S)
            rows.append((cz0, E, sp, om, ox, on_, we(f), we(hx), we(hn), erra))
            P(f"  cosZ {cz0+0.05:.2f} E {E:3.1f} {sp:9s} a1/a0 {om[0]:.3f}+-{erra[0]:.3f} | {ox[0]:.3f} | ({on_[0]:.3f})   dphi {om[1]:+6.1f}+-{erra[1]:.1f} | {ox[1]:+6.1f} | ({on_[1]:+6.1f})   N/S {om[2]:.3f}+-{erra[2]:.3f} | {ox[2]:.3f} | ({on_[2]:.3f})   W/E {we(f):.3f} | {we(hx):.3f} | ({we(hn):.3f})")
for icz, cz0 in enumerate((0.0, 0.1, 0.2)):
    r = [x for x in rows if x[0] == cz0]
    P(f"  cosZ {cz0+0.05:.2f} mean over species x E: a1/a0 MC {np.mean([x[3][0] for x in r]):.3f} Honda solmax {np.mean([x[4][0] for x in r]):.3f} (solmin {np.mean([x[5][0] for x in r]):.3f});  "
      f"N/S {np.mean([x[3][2] for x in r]):.3f} / {np.mean([x[4][2] for x in r]):.3f};  W/E {np.mean([x[6] for x in r]):.3f} / {np.mean([x[7] for x in r]):.3f}")
for nm_, j in (("a1/a0", 0), ("N/S", 2)):
    z = np.array([(x[3][j] - x[4][j]) / x[9][j] for x in rows]); zn = np.array([(x[3][j] - x[5][j]) / x[9][j] for x in rows])
    P(f"  chi2/36 (all cells, MC errors) of {nm_} vs Honda solmax {np.sum(z**2)/36:.2f} (solmin {np.sum(zn**2)/36:.2f}); mean MC-solmax {np.mean([x[3][j]-x[4][j] for x in rows]):+.4f}")
z = np.array([(x[3][1] - x[4][1] + 180) % 360 - 180 for x in rows]) / np.array([x[9][1] for x in rows]); P(f"  chi2/36 of dphi vs solmax {np.sum(z**2)/36:.2f}")
# ---- (c) antinue 2.5-4 GeV vertical vs engine
eng = np.load(W + "/runs/integ/out/engine_F_fine.npz")
ee = eng["e"]
def engF(sp, e0, e1, czs):
    x = np.logspace(np.log10(e0), np.log10(e1), 81)
    fl = eng[f"flux_dn6_total_{sp}"]; bs = eng[f"base_dn6_total_{sp}"]                     # (7 cz, 6 az, E) / (7 cz, E)
    out_ = []
    for ic in czs:
        f_ = np.array([np.trapezoid(np.exp(np.interp(np.log(x), np.log(ee), np.log(np.maximum(fl[ic, j], 1e-300)))), x) for j in range(fl.shape[1])]).mean()
        b_ = np.trapezoid(np.exp(np.interp(np.log(x), np.log(ee), np.log(np.maximum(bs[ic], 1e-300)))), x)
        out_.append(f_ / b_)
    return np.mean(out_)
P("\n(c) MC G / engine F (azimuth-averaged, cos Z 0.8-1 = engine bins 0.85, 0.95; and 0.4-0.8 = 0.45,0.55,0.65,0.75), v1b G, errors = MC statistical (G_err of the az-mean)")
spn = ("nue", "antinue", "numu", "antinumu")
for lab, czb, czi in (("0.8-1", (8, 9), (5, 6)), ("0.4-0.8", (4, 5, 6, 7), (1, 2, 3, 4))):
    for e_lab, ie in (("1-2.5", [5, 6]), ("2.5-4", [7])):
        rs = []
        for k, sp in enumerate(spn):
            g = np.nanmean(G0[k][ie][:, czb]); gerr = np.sqrt(np.nansum((Ge[k][ie][:, czb] ** 2))) / (len(ie) * len(czb) * 12) * np.sqrt(12) if False else \
                np.sqrt(np.nanmean(Ge[k][ie][:, czb] ** 2) / (len(ie) * len(czb) * 12))
            F = np.mean([engF(sp, Eedges[i], Eedges[i + 1], czi) for i in ie])
            rs.append((sp, g / F, gerr / F))
        P(f"  cos Z {lab} E {e_lab}: " + "  ".join(f"{sp} {r:.3f}+-{er:.3f}" for sp, r, er in rs))
# decomposition for antinue / numu at 2.5-4, vertical
ie, czb = 7, (8, 9)
P("\n    decomposition at cos Z 0.8-1, E 2.5-4 (v1b): per species  G | kappa(2.5-4) +- err | t/(kappa B) over 0.8-1 | s/t over 0.8-1")
for k, sp in enumerate(spn):
    t_ = tot[2][k, ie, 8:10].sum(); B_ = Bc[k, ie, 8:10].sum(); s_ = tot[0][k, ie, 8:10].sum()
    P(f"    {sp:9s} G {np.nanmean(G0[k, ie, 8:10]):.3f} | kappa {k0[k, ie]:.3f}+-{ke[k, ie]:.3f} | t/(kB) {t_/(k0[k, ie]*B_):.3f} | s/t {s_/t_:.3f}")
open(M + "/reports/v1b.txt", "w").write("\n".join(L_) + "\n")
