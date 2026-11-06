"""kappa_s(E) = t / B and the flatness gates (milestone 4, PM definition G = s / (kappa B)).

    python kappa.py TAG [TAG2]    # TAG: a run with 1D tally t (B = bent production, addon = no primary bend)

t1: largest cap, measure 1, azimuth-summed, (sp, E 60 bins 0.1-100 GeV, cos Z 20).  B_t1 from
``$W/runs/integ/out/base1d_kamioka.npz`` (the engine's curved-atmosphere 1D, GSF, v142p, no cutoff).  Jackknife G = 20.
Writes ``$M/work/kappa_<TAG>.npz``: for each cos Z window ('0.6-1' and '0-1'): kappa (sp, E), error, chi2 of t/B flatness
over the window's cos Z bins per (sp, E-band), and the t/B per cos Z bin (sp, E band, cz) with errors."""
import os, sys
import numpy as np
sys.path.insert(0, os.environ["R"] + "/tools/mc3d")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "m3c", "scripts"))
import analyse_sky as A
import m3c_lib as L
M, W = os.environ["M"], os.environ["W"]
tag = sys.argv[1]
base = np.load(W + "/runs/integ/out/base1d_kamioka.npz")
Bt1 = base["B_t1"]                                    # (4, 60, 20)
def red(d):
    return np.asarray(d["t1"])[-1].sum(-1)            # (sp, 60, 20)
J = A.JackRun(M + "/sky", tag, G=20, reducers={"x": red})
S, Mg = J.S["x"], J.M
tot = S.sum((0, 1))
rep = L.replicates(S.reshape(S.shape[:2] + (-1,)), Mg).reshape((S.shape[0],) + S.shape[2:])
G = len(rep)
def jkerr(vals):
    v = np.array(vals); return np.sqrt((G - 1.0) / G * np.nansum((v - np.nanmean(v, 0)) ** 2, 0))
BANDS = ((0.1, 0.3), (0.3, 1.0), (1.0, 3.0), (3.0, 10.0))
eb = np.logspace(-1, 2, 61); ec = np.sqrt(eb[1:] * eb[:-1])
out = {}
lines = [f"# kappa and flatness for {tag}: t1 (largest cap, az-summed) / B_t1 (engine 1D, GSF v142p)"]
for win, cz in (("0.6-1", slice(16, 20)), ("0-1", slice(10, 20))):
    def kap(x):
        with np.errstate(divide="ignore", invalid="ignore"):
            return x[:, :, cz].sum(-1) / Bt1[:, :, cz].sum(-1)
    k0 = kap(tot); ke = jkerr([kap(r) for r in rep])
    out[f"kappa_{win}"] = k0; out[f"kappa_err_{win}"] = ke
    lines.append(f"\n## window cos Z {win}: kappa(E) (sp x band; mean over the E bins of the band, error from the band-summed ratio)")
    for ib, (e0, e1) in enumerate(BANDS):
        sel = (ec >= e0) & (ec < e1)
        def band_ratio(x):
            with np.errstate(divide="ignore", invalid="ignore"):
                return x[:, sel][:, :, cz].sum((1, 2)) / Bt1[:, sel][:, :, cz].sum((1, 2))
        kb = band_ratio(tot); kbe = jkerr([band_ratio(r) for r in rep])
        # flatness: t/B per cos Z bin in the band, chi2 against the window's kappa
        def per_cz(x):
            with np.errstate(divide="ignore", invalid="ignore"):
                return x[:, sel][:, :, cz].sum(1) / Bt1[:, sel][:, :, cz].sum(1)       # (sp, ncz)
        r0 = per_cz(tot); re = jkerr([per_cz(r) for r in rep])
        chi = np.nansum(((r0 / kb[:, None] - 1) / (re / kb[:, None])) ** 2, 1)
        nd = r0.shape[1] - 1
        out[f"tB_{win}_band{ib}"] = r0; out[f"tB_err_{win}_band{ib}"] = re
        lines.append(f"E {e0:g}-{e1:g}: " + "  ".join(f"{s} {kb[k]:.4g}+-{kbe[k]:.2g} (flat chi2 {chi[k]:.1f}/{nd})" for k, s in enumerate(L.SP)))
        lines.append("      t/B per cos Z bin: " + "  ".join(f"{s} " + " ".join(f"{x/kb[k]:.2f}" for x in r0[k]) for k, s in enumerate(L.SP)))
np.savez(f"{M}/work/kappa_{tag}.npz", E=ec, **out)
open(f"{M}/reports/kappa_{tag}.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# ---- PM table: t/B / kappa(0.6-1) in cos Z groups, per species and E band (flatness down to the horizon)
GRP = ((10, 11, "0-0.1"), (11, 12, "0.1-0.2"), (12, 14, "0.2-0.4"), (14, 16, "0.4-0.6"), (16, 20, "0.6-1"))
BANDS3 = ((0.1, 0.3), (0.3, 1.0), (1.0, 3.0))
tl = [f"\n# {tag}: [t / B] in cos Z groups divided by kappa(cos Z 0.6-1) of the same band  (jackknife errors)"]
tl.append("band      species    " + "   ".join(f"{g[2]:>14s}" for g in GRP))
tabs = {}
for e0, e1 in BANDS3:
    sel = (ec >= e0) & (ec < e1)
    def grp(x, a=None, b=None):
        with np.errstate(divide="ignore", invalid="ignore"):
            return x[:, sel][:, :, a:b].sum((1, 2)) / Bt1[:, sel][:, :, a:b].sum((1, 2))
    def allr(x):
        k = grp(x, 16, 20)
        return np.stack([grp(x, a, b) / k for a, b, _ in GRP], -1)        # (sp, 5)
    r0 = allr(tot); re = jkerr([allr(r) for r in rep])
    tabs[f"{e0}-{e1}"] = (r0, re)
    for k, sp in enumerate(L.SP):
        tl.append(f"{e0:g}-{e1:g}  {sp:9s} " + "   ".join(f"{r0[k, j]:.3f}+-{re[k, j]:.3f}".rjust(14) for j in range(5)))
open(f"{M}/reports/kappa_{tag}.txt", "a").write("\n".join(tl) + "\n")
np.savez(f"{M}/work/kappa_groups_{tag}.npz", **{f"r_{k}": v[0] for k, v in tabs.items()}, **{f"e_{k}": v[1] for k, v in tabs.items()})
print("\n".join(tl))
