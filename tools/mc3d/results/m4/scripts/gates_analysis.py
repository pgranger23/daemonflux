"""Gates 5-7 from $M/work/B_design.npz and pilot-2's design_pilot2.npz.
 gate 5: s per coarse bin, B vs pilot-2 (independent samples, same configuration): chi2 of the difference with both jackknife errors.
 gate 6: G in the 2.5-10 GeV bins (G -> 1, not clipped).
 gate 7: split-half: halves from jackknife groups 0-9 / 10-19 (per stratum re-scaled by the primaries of the half): G_half1 vs G_half2, chi2/ndf
         against the jackknife error scaled for half samples."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "m3c", "scripts"))
import m3c_lib as L
M, W = os.environ["M"], os.environ["W"]
B = np.load(M + "/work/B_design.npz"); P = np.load(W + "/runs/m3f/work/design_pilot2.npz")
SB, MB = B["x"], B["M"]                                # (G,K,4,sp,20,10,12) tags s,n,t,v
base = np.load(W + "/runs/integ/out/base1d_kamioka.npz"); Bt1 = base["B_t1"]
def coarse(x):
    s = x.shape
    return x.reshape(s[:-3] + (s[-3] // 2, 2, s[-2] // 2, 2, s[-1] // 2, 2)).sum((-1, -3, -5))
out = []
def jk(f, S, Mg):
    tot = S.sum((0, 1)); rep = L.replicates(S.reshape(S.shape[:2] + (-1,)), Mg).reshape((S.shape[0],) + S.shape[2:])
    e0 = f(tot); er = np.array([f(r) for r in rep]); n = len(er)
    return e0, np.sqrt((n - 1.0) / n * np.nansum((er - np.nanmean(er, 0)) ** 2, 0))
# ---- gate 5: s only, E 0.1-2.5 GeV coarse (7 bins), cos Z down 5, az 6
SBc = coarse(SB)[..., :7, :, :]                         # (G,K,4,sp,7,5,6)
SPp = P["x"]                                            # pilot-2: (G,K,3 [s,t,v],sp,20,20,12) E10/dec, cz 20 -> down = 10:
SPc = coarse(SPp[..., 10:, :])[..., :7, :, :]
sB, eB = jk(lambda x: x[0], SBc, MB); sP, eP = jk(lambda x: x[0], SPc, P["M"])
z = (sB - sP) / np.sqrt(eB ** 2 + eP ** 2); ok = np.isfinite(z)
out.append(f"gate 5: s per coarse bin, B vs pilot-2 (E 0.1-2.5, down-going): bins {ok.sum()}, chi2/ndf {np.nansum(z[ok]**2)/ok.sum():.2f}, mean (B/pilot2 - 1) {np.nanmean(sB[ok]/sP[ok]-1):+.4f} (+-{np.sqrt(np.nanmean((eB/sB)[ok]**2+(eP/sP)[ok]**2)/ok.sum()):.4f}); median |z| {np.nanmedian(abs(z[ok])):.2f}")
for nm, sl in (("horizon 0-0.4", [0, 1]), ("mid 0.4-0.8", [2, 3]), ("vertical", [4])):
    zz = z[..., sl, :]; okk = np.isfinite(zz)
    out.append(f"     {nm}: chi2/ndf {np.nansum(zz[okk]**2)/okk.sum():.2f}  mean ratio-1 {np.nanmean((sB/sP)[..., sl, :]-1):+.4f}")
# ---- gate 6
tot = SB.sum((0, 1))
Bc = Bt1[:, :40, 10:].reshape(4, 20, 2, 10).sum(2)[..., None] / 12.0 * np.ones(12)
kap = np.load(M + "/out/kamioka_G_down_v1.npz")
G = kap["G_fine"]; Ge = kap["G_err_fine"]
out.append("gate 6: mean G per cos Z (fine, down-going bins 0-1 by 0.1) at E 3.2-10 GeV (fine E bins 15-19), all species/az: " +
           " ".join(f"{np.nanmean(G[:, 15:20, c]):.3f}" for c in range(10)) + f"   (mean error per bin {np.nanmean(Ge[:, 15:20]):.3f})")
# ---- gate 7: split halves
def half(S, Mg, gs):
    Sh = np.zeros_like(S); Sh[gs] = S[gs]
    Mh = np.zeros_like(Mg); Mh[gs] = Mg[gs]
    return Sh, Mh
kcoarse = kap["kappa_coarse"]; Bx = coarse(Bc[None])[0]
def Gof_tot(S):                                          # S (G,K,4,...) -> G coarse using kappa from the full sample (kappa is common)
    t = S.sum((0, 1)); return t[0] / (kcoarse[:, :, None, None] * Bx)
res = []
for gs in (np.arange(10), np.arange(10, 20)):
    Sh = coarse(SB)[gs].sum(0)                          # (K,4,sp,E,cz,az); scale each stratum so that it represents the full sample
    fac = (MB.sum(0) / MB[gs].sum(0))
    Sh = Sh * fac.reshape((-1,) + (1,) * (Sh.ndim - 1))
    res.append(Sh.sum(0)[0] / (kcoarse[:, :, None, None] * Bx))
Ga, Gb = res
Gtot = kap["G_coarse"]; Gerr = kap["G_err_coarse"]
sel = np.isfinite(Ga) & np.isfinite(Gb) & (Gtot > 0) & (np.arange(10)[None, :, None, None] < 7)
zz = (Ga - Gb) / (np.sqrt(2.0) * Gerr * np.sqrt(2.0))      # half-sample error = sqrt(2) x full error; difference of two halves: x sqrt(2)
out.append(f"gate 7: split halves (groups 0-9 vs 10-19) G coarse, E 0.1-2.5 GeV: bins {sel.sum()}, chi2/ndf of (G1-G2) with sqrt(2)*sqrt(2)*err: {np.nansum(zz[sel]**2)/sel.sum():.2f}; "
           f"mean (G1-G2)/G {np.nanmean(((Ga-Gb)/Gtot)[sel]):+.4f}")
open(M + "/reports/gates_B.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out))
