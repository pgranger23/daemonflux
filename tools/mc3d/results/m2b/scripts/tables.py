"""Render the MILESTONE2_RESULTS.md sec. 4 tables from the report summaries."""
import sys
import numpy as np
M = "/home/pigrange/mc3d_work/runs/m2b/reports/"
x = dict(np.load(M + sys.argv[1]))          # x10 summary
p = dict(np.load(M + "pilot_summary.npz"))  # pilot summary
NM = {14: "numu", -14: "antinumu", 12: "nue", -12: "antinue"}
out = []
o = out.append

o("#### (a) cost table (10x run)")
o("| species | E [GeV] | cos Z | N_eff 3D / az bin per 1e6 prim. | rel. err. Phi_3D / az bin | clus | rho(3D,1D) | rel. err. R / az bin | core-h, 3% on Phi_3D: no reuse / N_az=12 | core-h, 3% on R: no reuse / N_az=12 |")
o("|---|---|---|---|---|---|---|---|---|---|")
for r in x["cost_rows"]:
    s, e, iz = int(r[0]), r[1], int(r[2])
    if iz > 11:
        continue
    o(f"| {NM[s]} | {e:.1f} | {0.1*(iz-10):.1f}-{0.1*(iz-9):.1f} | {r[5]:.2f} | {r[6]:.2f} | {r[7]:.2f} | {r[8]:+.2f} | {r[9]:.2f} | {r[11]:.0f} / {r[12]:.0f} | {r[13]:.0f} / {r[14]:.0f} |")
o("")
o("pilot vs 10x core-h (3% on R, N_az=12), same bins: " + ", ".join(
    f"{NM[int(a[0])]} {a[1]:.1f} cz{int(a[2])-10}: {a[14]:.0f}/{b[14]:.0f}"
    for a, b in zip(p["cost_rows"], x["cost_rows"]) if a[2] <= 11 and a[1] == 0.5))
o("")
o("#### strata (10x)")
o("| stratum [GeV/nucleon] | showers | ms/shower | scored 3D | scored 1D | scored either | 3D in E-W region |")
o("|---|---|---|---|---|---|---|")
for r in x["strata_table"]:
    o(f"| {r[1]:.1f}-{r[2]:.1f} | {r[3]:.0f} | {r[5]:.1f} | {r[6]:.4f} | {r[7]:.4f} | {r[8]:.4f} | {r[9]:.5f} |")
sf = x["scored_fraction_all"]
o(f"| all | {x['n_prim']:.0f} | {x['cpu_per_shower']*1e3:.2f} | {sf[0]:.4f} | {sf[1]:.4f} | {sf[2]:.4f} | {sf[3]:.5f} |")
o("")
for k in ("ew_binding_Phi_3D", "ew_binding_R", "survey_0.1_3.0", "survey_0.1_10.0", "neyman"):
    o(f"{k}: {np.round(x[k], 4).tolist()}")
o("")
o("#### (b) R vs cos Z (10x, in-run; true and 1D residual if p1dx present)")
keys, bt = x["btab_keys"], x["btab"]
for (s, e), t in zip(keys, bt):
    s = int(s)
    if s not in (14, 12):
        continue
    o(f"{NM[s]} {e} GeV")
    o("| cos Z | " + " | ".join(f"{0.1*i:.1f}-{0.1*i+0.1:.1f}" for i in range(10)) + " |")
    o("|---" * 11 + "|")
    o("| MC in-run | " + " | ".join(f"{a:.3f}±{b:.3f}" for a, b in zip(t[0], t[1])) + " |")
    if np.all(np.isfinite(t[4])):
        o("| MC true (p3dx/p1dx) | " + " | ".join(f"{a:.3f}±{b:.3f}" for a, b in zip(t[4], t[5])) + " |")
        o("| in-run 1D / true 1D | " + " | ".join(f"{a:.3f}±{b:.3f}" for a, b in zip(t[6], t[7])) + " |")
    o("| engine E_off_s | " + " | ".join(f"{a:.3f}" for a in t[8]) + " |")
    o("")
o("H/V contrast R(0-0.1)/R(0.9-1): " + "; ".join(
    f"{NM[int(r[0])]} {r[1]}: true {r[2]:.3f}±{r[3]:.3f} in-run {r[4]:.3f}±{r[5]:.3f} engine {r[6]:.3f}"
    for r in x["hv_contrast"]))
o("")
o("Honda bands: " + "; ".join(
    f"{r[0]:.2f}-{r[1]:.2f} GeV nsp={int(r[2])} cz{int(r[3])-10}: in-run {r[4]:.3f}±{r[5]:.3f} true {r[6]:.3f}±{r[7]:.3f}"
    for r in x["honda_bands"]))
o("")
o("G7: " + "; ".join(f"{NM[int(r[0])]} {r[1]}: " + " ".join(f"{a:.3f}±{b:.3f}" for a, b in zip(r[2:6], r[6:10])) for r in x["g7"]))
o("")
o("#### (c) measures (10x)")
E_C = (0.2, 0.3, 0.5, 1.0, 2.0, 3.0)
e10 = np.logspace(-1, 2, 31)
ai, sai, at, sat = x["avg_inrun"], x["avg_inrun_err"], x["avg_true"], x["avg_true_err"]
ea = x["engine_avgs"]
ee = x["engine_e"]
mc, smc = x["cone_mean_cos"], x["cone_mean_cos_err"]
for sp, nm, esp in ((2, "numu", 0), (0, "nue", 2)):
    o(f"{nm}")
    o("| quantity | " + " | ".join(f"{e} GeV" for e in E_C) + " |")
    o("|---" * 7 + "|")
    lab = ("<F>_Omega plain", "<F>_cos", "sum s1/sum t1", "sum s0/sum t0 (through the ground)", "F(vertical)")
    for q in range(5):
        ie = [int(np.searchsorted(e10, e * (1 + 1e-9)) - 1) for e in E_C]
        o(f"| {lab[q]} in-run | " + " | ".join(f"{ai[q, sp, i]:.3f}±{sai[q, sp, i]:.3f}" for i in ie) + " |")
        if np.all(np.isfinite(at[q, sp, ie])):
            o(f"| {lab[q]} true | " + " | ".join(f"{at[q, sp, i]:.3f}±{sat[q, sp, i]:.3f}" for i in ie) + " |")
        qe = {0: 0, 1: 1, 4: 2}.get(q)
        if qe is not None:
            iee = [int(np.argmin(np.abs(ee - np.sqrt(e10[i] * e10[i + 1])))) for i in ie]
            o(f"| {lab[q]} engine | " + " | ".join(f"{ea[qe, esp, j]:.3f}" for j in iee) + " |")
    em = np.logspace(-1, 1, 21)
    iem = [int(np.searchsorted(em, e * (1 + 1e-9)) - 1) for e in E_C]
    o(f"| <cos alpha> MC (all primaries) | " + " | ".join(f"{mc[sp, i, 3]:.3f}" for i in iem) + " |")
    o("")
o("#### (d) cone (10x): RMS [deg] MC all-gen / gen1 / engine realised")
rc, src = x["cone_rms"], x["cone_rms_err"]
eng_e = ee
import sys as _s
_s.path.insert(0, "/afs/cern.ch/work/p/pigrange/daemonflux/tools/mc3d")
from analyse_sky import engine_kernel
eng = np.load("/afs/cern.ch/work/p/pigrange/daemonflux/tools/mceq3d/offaxis_excess_channel_v2.npz")
E_D = (0.224, 0.282, 0.447, 0.708, 1.12, 1.78)
em = np.logspace(-1, 1, 21)
for sp, nm, chans in ((2, "numu", (("pi", 0, "sigma_pi"), ("mu", 2, "sigma_mu_numu"), ("K", 1, "sigma_k"))),
                      (0, "nue", (("mu", 2, "sigma_mu_nue"), ("K", 1, "sigma_k")))):
    o(f"{nm}")
    o("| channel | " + " | ".join(f"{e} GeV" for e in E_D) + " |")
    o("|---" * 7 + "|")
    for cn, c, ek in chans:
        cells = []
        for e in E_D:
            ie = int(np.searchsorted(em, e * (1 + 1e-9)) - 1)
            iee = int(np.argmin(np.abs(eng_e - e)))
            rr = engine_kernel(eng[ek][iee])[1][0]
            cells.append(f"{rc[sp, ie, c]:.1f} / {rc[sp, ie, 4 + c]:.1f} / {rr:.1f} (x{rc[sp, ie, c]/rr:.2f}, f={rc[sp, ie, 8 + c]:.2f})")
        o(f"| {cn} | " + " | ".join(cells) + " |")
    o("| first-interaction fraction | " + " | ".join(
        f"{rc[sp, int(np.searchsorted(em, e*(1+1e-9))-1), 13]:.2f}" for e in E_D) + " |")
    o("")
print("\n".join(out))
