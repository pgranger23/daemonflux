"""Milestone-3b pilot summary: species fan-out, Fourier vs Honda / engine, N/S,
3D/1D, comparisons with the rotation and re-anchored runs, cost to sharpen."""
import importlib.util  # noqa
import os, sys
import numpy as np
import analyse_m3b as a
M = os.environ["M"]
tag = sys.argv[1] if len(sys.argv) > 1 else "pilot"
ax = a.ew_axis()
lines = []
def out(s=""):
    print(s, flush=True); lines.append(s)
J = a.load(M + "/sky", tag, G=20)
tab = a.ew_table(J, ax, tags=("s", "n", "t", "v"))
ho = a.table_obs(a.honda_patterns(), ax)
eo = a.table_obs(a.engine_patterns(M + "/work/engine_ew_gsf.npz"), ax)
eo3 = None
if os.path.exists(M + "/work/engine_ew_h3a.npz"):
    eo3 = a.table_obs(a.engine_patterns(M + "/work/engine_ew_h3a.npz"), ax)
cpu = J.cpu.sum() / 3600
out(f"# {tag}: {J.n_showers:.4g} primaries, {cpu:.1f} core-h; geomagnetic East {ax:.2f} deg")
est, err, rep = tab["s"]
SP = a.SP_NAMES
EB = list(a.E_BANDS)
# 1. a1/a0 table
out("\n## 1. 3D East-West dipole amplitude a1/a0 (MC +- jackknife | Honda | engine GSF" + (" | engine H3a" if eo3 is not None else "") + ")")
for ie, E in enumerate(EB):
    for iz, cz in enumerate(a.CZ_BANDS):
        out(f"E {E} GeV cosZ {cz[0]:.1f}-{cz[1]:.1f}: " + "  ".join(
            f"{sp} {est[k,ie,iz,0]:.3f}+-{err[k,ie,iz,0]:.3f} | {ho[k,ie,iz,0]:.3f} | {eo[k,ie,iz,0]:.3f}"
            + (f" | {eo3[k,ie,iz,0]:.3f}" if eo3 is not None else "") for k, sp in enumerate(SP)))
# 2. phase
out("\n## 2. phase dphi = phi1 - geomagnetic West [deg] (MC | Honda | engine)")
for ie, E in enumerate(EB):
    for iz, cz in enumerate(a.CZ_BANDS):
        out(f"E {E} cosZ {cz[0]:.1f}-{cz[1]:.1f}: " + "  ".join(
            f"{sp} {est[k,ie,iz,1]:+.1f}+-{err[k,ie,iz,1]:.1f} | {ho[k,ie,iz,1]:+.1f} | {eo[k,ie,iz,1]:+.1f}" for k, sp in enumerate(SP)))
# 3. W/E (two-harmonic) and max/min
out("\n## 3. W/E (2-harmonic at geomagnetic W/E) and max/min over the 12 bins (MC | Honda | engine)")
for ie, E in enumerate(EB):
    for iz, cz in enumerate(a.CZ_BANDS):
        out(f"E {E} cosZ {cz[0]:.1f}-{cz[1]:.1f}: " + "  ".join(
            f"{sp} W/E {est[k,ie,iz,3]:.2f}+-{err[k,ie,iz,3]:.2f}|{ho[k,ie,iz,3]:.2f}|{eo[k,ie,iz,3]:.2f} "
            f"mm {est[k,ie,iz,4]:.2f}+-{err[k,ie,iz,4]:.2f}|{ho[k,ie,iz,4]:.2f}|{eo[k,ie,iz,4]:.2f}"
            for k, sp in enumerate(SP)))
# 4. fan-out: ratio to numu, pooled over cz 0-0.3 too
out("\n## 4. species fan-out: a1(species)/a1(numu) (MC +- jackknife | Honda | engine)")
def ratios(e_, r_):
    rr = r_[..., 0] / r_[:, 2:3, ..., 0]
    G = rr.shape[0]
    re_ = np.sqrt((G - 1.0) / G * np.nansum((rr - np.nanmean(rr, 0)) ** 2, 0))
    return e_[..., 0] / e_[2:3, ..., 0], re_
rat, rerr = ratios(est, rep)
for ie, E in enumerate(EB):
    for iz, cz in enumerate(a.CZ_BANDS):
        out(f"E {E} cosZ {cz[0]:.1f}-{cz[1]:.1f}: " + "  ".join(
            f"{sp}/numu {rat[k,ie,iz]:.2f}+-{rerr[k,ie,iz]:.2f} | {ho[k,ie,iz,0]/ho[2,ie,iz,0]:.2f} | {eo[k,ie,iz,0]/eo[2,ie,iz,0]:.2f}"
            for k, sp in enumerate(SP) if sp != "numu"))
# pooled over the three zenith bands (sum of patterns)
pats = a.band_patterns(J.total("ew"))[0].sum(2)            # (sp, E, az)
reps = np.array([a.band_patterns(r)[0].sum(2) for r in J.replicates("ew")])
hp = a.honda_patterns().sum(2)
ep = a.engine_patterns(M + "/work/engine_ew_gsf.npz").sum(2)
def obs2(p):
    o = np.zeros(p.shape[:2] + (len(a.OBS),))
    for idx in np.ndindex(p.shape[:2]):
        o[idx] = a.observables(p[idx], ax)
    return o
o_mc = obs2(pats); o_rep = np.array([obs2(r) for r in reps]); o_h = obs2(hp); o_e = obs2(ep)
G = o_rep.shape[0]
e_mc = np.sqrt((G - 1.0) / G * np.nansum((o_rep - np.nanmean(o_rep, 0)) ** 2, 0))
out("\n## 5. pooled over cos Z 0-0.3 (sum of the three bands): a1/a0 and fan-out")
for ie, E in enumerate(EB):
    r_mc = o_mc[:, ie, 0] / o_mc[2, ie, 0]
    rr = o_rep[:, :, ie, 0] / o_rep[:, 2:3, ie, 0]
    rre = np.sqrt((G - 1.0) / G * np.nansum((rr - np.nanmean(rr, 0)) ** 2, 0))
    out(f"E {E}: " + "  ".join(f"{sp} a1 {o_mc[k,ie,0]:.3f}+-{e_mc[k,ie,0]:.3f} (H {o_h[k,ie,0]:.3f}, E {o_e[k,ie,0]:.3f}) /numu {r_mc[k]:.2f}+-{rre[k]:.2f} (H {o_h[k,ie,0]/o_h[2,ie,0]:.2f}, E {o_e[k,ie,0]/o_e[2,ie,0]:.2f})" for k, sp in enumerate(SP)))
    out(f"      phase: " + "  ".join(f"{sp} {o_mc[k,ie,1]:+.1f}+-{e_mc[k,ie,1]:.1f} (H {o_h[k,ie,1]:+.1f}, E {o_e[k,ie,1]:+.1f})" for k, sp in enumerate(SP)))
    out(f"      N/S: " + "  ".join(f"{sp} fN {o_mc[k,ie,5]:.3f}+-{e_mc[k,ie,5]:.3f} (H {o_h[k,ie,5]:.3f}) fS {o_mc[k,ie,6]:.3f}+-{e_mc[k,ie,6]:.3f} (H {o_h[k,ie,6]:.3f})" for k, sp in enumerate(SP)))
# ordering significance: nue - antinumu, antinumu - numu, numu - antinue (pooled)
out("\n## 6. ordering nue > antinumu > numu > antinue: pooled differences of a1/a0 (MC +- jk | Honda)")
for ie, E in enumerate(EB):
    s_ = []
    for (i, j) in ((0, 3), (3, 2), (2, 1), (0, 1)):
        d = o_mc[i, ie, 0] - o_mc[j, ie, 0]
        dr = o_rep[:, i, ie, 0] - o_rep[:, j, ie, 0]
        de = np.sqrt((G - 1.0) / G * np.nansum((dr - np.nanmean(dr)) ** 2))
        s_.append(f"{SP[i]}-{SP[j]} {d:+.3f}+-{de:.3f} ({d/de:+.1f} sig; H {o_h[i,ie,0]-o_h[j,ie,0]:+.3f})")
    out(f"E {E}: " + "  ".join(s_))
# 7. the other tallies pooled
for tg, lab in (("v", "1D site cutoff"), ("n", "3D no cutoff, copy 0")):
    it = a.TAG_IDX[tg]
    p_ = a.band_patterns(J.total("ew"))[it].sum(2)
    r_ = np.array([a.band_patterns(r)[it].sum(2) for r in J.replicates("ew")])
    o_ = obs2(p_); orp = np.array([obs2(x) for x in r_])
    e_ = np.sqrt((G - 1.0) / G * np.nansum((orp - np.nanmean(orp, 0)) ** 2, 0))
    out(f"\n## 7. {lab} [{tg}], pooled cos Z 0-0.3: a1/a0, dphi, c1/a0, s1/a0")
    for ie, E in enumerate(EB):
        out(f"E {E}: " + "  ".join(f"{sp} a1 {o_[k,ie,0]:.3f}+-{e_[k,ie,0]:.3f} dphi {o_[k,ie,1]:+.0f}+-{e_[k,ie,1]:.0f} c1 {o_[k,ie,8]:+.3f}+-{e_[k,ie,8]:.3f} s1 {o_[k,ie,9]:+.3f}+-{e_[k,ie,9]:.3f}" for k, sp in enumerate(SP)))
np.savez(M + f"/reports/{tag}_ew_summary.npz", est=est, err=err, honda=ho, engine=eo,
         engine_h3a=(eo3 if eo3 is not None else np.zeros(0)), pooled_mc=o_mc, pooled_err=e_mc,
         pooled_honda=o_h, pooled_engine=o_e, obs=np.array(a.OBS), axis=ax,
         e_bands=np.array(EB), cz_bands=np.array(a.CZ_BANDS), cpu_h=cpu, n=J.n_showers)
open(M + f"/reports/{tag}_ew.txt", "w").write("\n".join(lines) + "\n")
