"""Effect of the SIBYLL nucleon-projectile fix: a1rot (fixed pools) - a0rot (old pools), same seeds."""
import importlib.util  # noqa
import os
import numpy as np
import analyse_m3b as a
M = os.environ["M"]
J0 = a.load(M + "/sky", "a0rot", G=20)
J1 = a.load(M + "/sky", "a1rot", G=20)
assert J0.strata == J1.strata
ax = a.ew_axis()
lines = []
def out(s=""):
    print(s, flush=True); lines.append(s)
e0, r0 = a.a1_obs(J0, ax)
e1, r1 = a.a1_obs(J1, ax)
d = e1[..., 0] - e0[..., 0]
dr = r1[..., 0] - r0[..., 0]
G = dr.shape[0]
de = np.sqrt((G - 1.0) / G * np.nansum((dr - np.nanmean(dr, 0)) ** 2, 0))
out("# SIBYLL nucleon fix: a1rot (pools_fix) - a0rot (old pools), same seeds, everything on, r_thr = 0")
out(f"3D a1/a0: mean diff {np.mean(d):+.4f}, rms {np.sqrt(np.mean(d**2)):.4f}, chi2/36 {np.nansum((d/de)**2)/36:.2f}")
# azimuth-integrated fluxes and ratios, E bands, cz 0-0.3 and 0.3-1
az0, az1 = J0.total("az"), J1.total("az")
rep0, rep1 = J0.replicates("az"), J1.replicates("az")
names = [t + m for t in ("s", "n", "t", "v") for m in ("0", "1")]
i_s, i_t, i_v = names.index("s1"), names.index("t1"), names.index("v1")
for ie, (E, (i0, i1)) in enumerate(a.E_BANDS.items()):
    for czl, sl in (("cz 0-0.3", slice(10, 13)), ("cz 0.3-1", slice(13, 20))):
        row = []
        for k, sp in enumerate(a.SP_NAMES):
            f = lambda A, i: A[i, k, i0:i1, sl].sum()
            r = f(az1, i_s) / f(az0, i_s)
            rr = np.array([f(x1, i_s) / f(x0, i_s) for x0, x1 in zip(rep0, rep1)])
            er = np.sqrt((G - 1.0) / G * np.sum((rr - rr.mean()) ** 2))
            q1 = f(az1, i_s) / f(az1, i_t); q0 = f(az0, i_s) / f(az0, i_t)
            qq = np.array([f(x1, i_s) / f(x1, i_t) / (f(x0, i_s) / f(x0, i_t)) for x0, x1 in zip(rep0, rep1)])
            eq = np.sqrt((G - 1.0) / G * np.sum((qq - qq.mean()) ** 2))
            row.append(f"{sp} 3D flux new/old {r:.3f}+-{er:.3f}, (3D/1D) new/old {q1/q0:.3f}+-{eq:.3f}")
        out(f"E {E} {czl}: " + "; ".join(row))
open(M + "/reports/poolfix.txt", "w").write("\n".join(lines) + "\n")
