"""Relative errors of G = s/t (3D with cutoff / 1D no cutoff, same showers) per design bin from pilot-2
(jackknife G=20), for a fine and a coarse binning, and the primaries / CPU needed for 3% and 5%.
    python design_errors.py -> $M/reports/design_errors.txt"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "m3c", "scripts"))
import m3c_lib as L
M = os.environ["M"]
d = np.load(M + "/work/design_pilot2.npz")
S, Mg = d["x"], d["M"]                       # (G,K,3,sp,E20,cz20,az12)
Nprim = Mg.sum(); CPU_CORE_H = 34.2          # pilot-2
EC = 10 ** (-1 + (np.arange(20) + 0.5) / 20.0 * 2.0)
def rebin(x, fe, fc, fa):
    """x (..., sp, E, cz, az): merge fe E-bins, fc cz bins, fa az bins."""
    s = x.shape
    x = x.reshape(s[:-3] + (s[-3] // fe, fe, s[-2] // fc, fc, s[-1] // fa, fa))
    return x.sum((-1, -3, -5))
def err_map(Sx):
    tot = Sx.sum((0, 1))
    rep = L.replicates(Sx, Mg)               # (G, 3, sp, E, cz, az)
    with np.errstate(divide="ignore", invalid="ignore"):
        G0 = tot[0] / tot[1]
        Gr = rep[:, 0] / rep[:, 1]
        Gn = len(Gr)
        e = np.sqrt((Gn - 1.0) / Gn * np.nansum((Gr - np.nanmean(Gr, 0)) ** 2, 0))
        rel = e / G0
        tneff = tot[1]
    return G0, rel, tot
lines = ["# pilot-2 (2.0e6 primaries, 34.2 core-h): relative error of G = s/t per design bin (jackknife G = 20)"]
for name, (fe, fc, fa) in (("fine: E 10/dec x 20 cosZ x 12 az", (1, 1, 1)),
                          ("coarse: E 5/dec x 10 cosZ x 6 az", (2, 2, 2)),
                          ("minimal: E 2.5/dec x 5 cosZ x 4 az", (4, 4, 3))):
    Sx = rebin(S, fe, fc, fa)
    G0, rel, tot = err_map(Sx)
    ne, nc = Sx.shape[-3], Sx.shape[-2]
    ec = np.sqrt(10 ** (-1 + np.arange(ne) * 2.0 / ne) * 10 ** (-1 + (np.arange(ne) + 1) * 2.0 / ne))
    cz = -1 + (np.arange(nc) + 0.5) * 2.0 / nc
    lines.append(f"\n## {name}")
    # classes: horizon (|cosZ| < 0.3, downward cosZ 0-0.3), vertical (cosZ>0.8), by energy band; require G>0 and t>0
    for cls, sel_c in (("horizon cosZ 0-0.3", (cz > 0) & (cz < 0.3)), ("vertical cosZ 0.8-1", cz > 0.8), ("mid cosZ 0.3-0.8", (cz >= 0.3) & (cz <= 0.8))):
        for elo, ehi in ((0.1, 0.3), (0.3, 1.0), (1.0, 3.0), (3.0, 10.0)):
            sel_e = (ec >= elo) & (ec < ehi)
            r = rel[:, sel_e][:, :, sel_c]            # (sp, e, c, az)
            g = G0[:, sel_e][:, :, sel_c]
            ok = np.isfinite(r) & (g > 0)
            if ok.sum() == 0:
                continue
            rr = r[ok]
            med, p90 = np.median(rr), np.percentile(rr, 90)
            lines.append(f"{cls:20s} E {elo:g}-{ehi:g}: bins {ok.sum():5d} (empty/no-signal {np.size(ok)-ok.sum():4d}); rel. err median {med*100:5.1f}%  90th pct {p90*100:6.1f}%;"
                         f"  primaries x for median->3%: {(med/0.03)**2:6.1f}, ->5%: {(med/0.05)**2:6.1f}; 90th->5%: {(p90/0.05)**2:7.1f}")
open(M + "/reports/design_errors.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
