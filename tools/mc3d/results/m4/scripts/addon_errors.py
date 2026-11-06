"""R_s = F_on/F_off on the coarse down-going grid (E 5/dec 0.1-2.5 GeV x cos Z 5 x az 6, per species) from addon_<TAG>.npz,
with jackknife errors: F_off = s - mu_on + mu_cf.   python addon_errors.py TAG"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "m3c", "scripts"))
import m3c_lib as L
M = os.environ["M"]; tag = sys.argv[1]
d = np.load(f"{M}/work/addon_{tag}.npz")
S, Mg = d["x"], d["M"]                       # (G,K,3,sp,60,20,12)
def coarse(x):
    x = x[..., :28, 10:, :]
    s = x.shape
    return x.reshape(s[:-3] + (7, 4, 5, 2, 6, 2)).sum((-1, -3, -5))
def R(x):                                    # x (3,sp,7,5,6)
    with np.errstate(divide="ignore", invalid="ignore"):
        return x[0] / (x[0] - x[1] + x[2])
Sc = coarse(S)
tot = Sc.sum((0, 1)); R0 = R(tot)
rep = L.replicates(Sc.reshape(*Sc.shape[:2], -1), Mg).reshape((Sc.shape[0],) + Sc.shape[2:])
Rr = np.array([R(r) for r in rep]); G = len(Rr)
err = np.sqrt((G - 1.0) / G * np.nansum((Rr - np.nanmean(Rr, 0)) ** 2, 0))
ok = np.isfinite(R0) & np.isfinite(err) & (err > 0)
hor = np.zeros(R0.shape, bool); hor[..., :2, :] = True
print(f"{tag}: primaries {Mg.sum():.3g}; R_s over bins: median {np.median(R0[ok]):.3f}, range {R0[ok].min():.2f}-{R0[ok].max():.2f}")
for name, m in (("horizon (cosZ<0.4)", hor), ("other", ~hor)):
    mm = ok & m
    print(f"  {name}: bins {mm.sum()}  median abs error on R_s {np.median(err[mm]):.4f}  90th {np.percentile(err[mm],90):.4f}")
for ie in range(7):
    mm = ok[..., ie, :, :]
    print(f"  E bin {ie}: median err {np.median(err[..., ie, :, :][mm]):.4f}; median |R-1| {np.median(abs(R0[..., ie, :, :][mm]-1)):.3f}")
np.savez(f"{M}/reports/addon_errors_{tag}.npz", R=R0, err=err)
