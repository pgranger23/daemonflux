"""Gate C5: bias of the rotation reuse vs threshold, from the paired a1rot/a1anc."""
import importlib.util  # noqa
import os
import numpy as np
import analyse_m3b as a
import cutoff, sky
M = os.environ["M"]
Jr = a.load(M + "/sky", "a1rot", G=20)
Ja = a.load(M + "/sky", "a1anc", G=20)
assert Jr.strata == Ja.strata and np.allclose(Jr.M, Ja.M)
ax = a.ew_axis()
lines = []
def out(s=""):
    print(s, flush=True); lines.append(s)
# lower-edge rigidity of each (group, stratum)
rlo = []
for g, k in Jr.strata:
    _n, A_, Z_, _ = sky.GROUPS[g]
    rlo.append(float(cutoff.rigidity_gv(sky.STRATA[k], A_, Z_)))
rlo = np.array(rlo)
e0, r0 = a.a1_obs(Jr, ax)
_, ra = a.a1_obs(Ja, ax)
err0 = np.sqrt(a.jvar(r0[..., 0]))
out("# gate C5: bias of reusing a cascade by rotation for primaries above r_thr,")
out("# = 3D observable(re-anchored above r_thr) - observable(rotated everywhere),")
out("# paired runs a1rot/a1anc (1e6 primaries, same seeds), stratified jackknife")
for thr in (0.0, 4.5, 10.9, 15.0, 20.0, 33.0, 60.0, 101.0):
    above = [h for h in range(Jr.K) if rlo[h] >= thr - 1e-9]
    d, de = a.rotation_bias(Jr, Ja, ax, above)
    # summary over the 36 (species, E, cz) cells: |d a1| relative to a1, and in units of the stat error of a1rot
    da = d[..., 0]
    rel = da / e0[..., 0]
    z = da / np.where(de[..., 0] > 0, de[..., 0], np.nan)
    out(f"r_thr {thr:6.1f} GV ({len(above)} strata reused): d(a1/a0) mean {np.mean(da):+.4f}, "
        f"rms {np.sqrt(np.mean(da**2)):.4f}, max |.| {np.max(np.abs(da)):.4f}; "
        f"mean rel {np.mean(rel):+.3f}, rms rel {np.sqrt(np.mean(rel**2)):.3f}; "
        f"rms d(dphi) {np.sqrt(np.mean(d[...,1]**2)):.2f} deg; "
        f"chi2(d=0)/36 = {np.nansum(z**2)/36:.2f}; |d|/sigma_stat(1e6) rms {np.sqrt(np.mean((da/err0)**2)):.3f}")
    if thr in (0.0, 20.0, 33.0):
        for k, sp in enumerate(a.SP_NAMES):
            out("   " + sp + " " + " ".join(f"{d[k,ie,iz,0]:+.3f}+-{de[k,ie,iz,0]:.3f}"
                                         for ie in range(3) for iz in range(3)))
open(M + "/reports/bias_c5.txt", "w").write("\n".join(lines) + "\n")
out("\n# per stratum: chi2/36 of (re-anchored - rotated) for that stratum alone, and the mean d(a1/a0)")
for h, (g, k) in enumerate(Jr.strata):
    d, de = a.rotation_bias(Jr, Ja, ax, [h])
    z = d[..., 0] / np.where(de[..., 0] > 0, de[..., 0], np.nan)
    out(f"  {g:5s}{k} R_lo {rlo[h]:7.1f} GV: chi2/36 {np.nansum(z**2)/36:6.2f}  mean d {np.mean(d[...,0]):+.4f}  rms d {np.sqrt(np.mean(d[...,0]**2)):.4f}")
open(M + "/reports/bias_c5.txt", "w").write("\n".join(lines) + "\n")
