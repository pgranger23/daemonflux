"""Gate C3: MC v/t from the re-weighted 1D entries (c3_reweight.py) for the
three flux conventions, per bin, species, band; jackknife errors.
    python c3_rew_report.py TAG [TAG...] -> stdout (+ $M/reports/c3_rew_<TAG>.txt)"""
import os, sys
import numpy as np
import m3c_lib as L
M = os.environ["M"]
for tag in sys.argv[1:]:
    d = dict(np.load(f"{M}/work/rew_{tag}.npz"))
    lines = [f"# {tag}: MC v/t (1D site cutoff / 1D no cutoff), measure 1, by flux convention"]
    for v in ("orig", "mceq", "mceq_t2"):
        S, Mg = d[f"S|{v}"], d["M"]          # (G, K, tv, bin, sp, band)
        tot = S.sum((0, 1))
        rep = L.replicates(S, Mg)
        r = tot[1] / tot[0]
        rr = rep[:, 1] / rep[:, 0]
        e = L.jk_err(rr)
        lines.append(f"## weights {v}")
        for nb, lab in enumerate(d["bins"]):
            lines.append(f"{lab:12s} " + "  ".join(
                f"{sp} " + " ".join(f"{r[nb, k, ib]:.3f}+-{e[nb, k, ib]:.3f}" for ib in range(3))
                for k, sp in enumerate(L.SP)))
    # share of t from T < 2 GeV/n and from the p 1-3.16 stratum
    S = d["S|orig"]
    st = list(d["strata"])
    lines.append("## share of t (orig weights) from each lowest stratum (p:0, He:0, CNO:0)")
    for nb, lab in enumerate(d["bins"]):
        tt = S[:, :, 0, nb].sum(0)          # (K, sp, band)
        lines.append(f"{lab:12s} " + "  ".join(
            f"{s_}: " + " ".join(f"{tt[st.index(s_), 2, ib] / tt[:, 2, ib].sum():.3f}" for ib in range(3))
            for s_ in ("p:0", "He:0", "CNO:0")) + "   (numu, 3 bands)")
    open(f"{M}/reports/c3_rew_{tag}.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
