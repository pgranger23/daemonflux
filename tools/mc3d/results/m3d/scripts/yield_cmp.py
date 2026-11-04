"""Task 3a: per-interaction yields, chromo DPMJET-III-19.3 pool vs MCEq's
tabulated DPMJET-III-19.3 (the MC's own tables, v140), at matched projectile
kinetic energy (nearest table column), p/pi+/pi- + air.

Yields per interaction of each child above total-energy thresholds (E_tot of
the child; pions >= 0.7 GeV are the ones that can make a 0.3-1 GeV neutrino
directly: E_nu <= 0.427 E_pi); the table's within-bin shape is the sampler's
own E^-g (interactions.MCEqYieldBackend)."""
import os
import sys

import numpy as np

R = os.environ["R"]
sys.path.insert(0, R + "/tools/mc3d")
from constants import MASS  # noqa: E402
from interactions import MCEqYieldBackend  # noqa: E402

POOLS = os.environ["MC3D_POOL_CACHE"]
B = MCEqYieldBackend()
eg, eb = B.e_grid, B.e_bins
THR = (0.2, 0.7, 1.0, 2.0)
KIDS = (("pi+", 211), ("pi-", -211), ("K+", 321), ("K-", -321), ("p", 2212),
        ("n", 2112), ("Lam", 3122))


def table_above(proj, child, j, thr):
    tab = B.y.get(proj, {}).get(child)
    if tab is None:
        return 0.0
    tot, _c, g = B._prep(tab)
    mc = MASS.get(child, 0.0)
    out = 0.0
    for i in range(len(eg)):
        n = tab[i, j]
        if n <= 0:
            continue
        lo, hi = eb[i] + mc, eb[i + 1] + mc           # total energy
        a = max(lo, thr)
        if a >= hi:
            continue
        gg = g[i, j]
        if abs(gg - 1.0) < 1e-6:
            fr = np.log(hi / a) / np.log(hi / lo)
        else:
            e = 1.0 - gg
            fr = (hi ** e - a ** e) / (hi ** e - lo ** e)
        out += n * fr
    return float(out)


def table_above_i(proj, child, ek, thr):
    """``table_above`` interpolated (log-linear in T) between the two table
    columns around the projectile kinetic energy ``ek``."""
    j1 = int(np.clip(np.searchsorted(eg, ek), 1, len(eg) - 1))
    j0 = j1 - 1
    w = np.log(ek / eg[j0]) / np.log(eg[j1] / eg[j0])
    a, b = table_above(proj, child, j0, thr), table_above(proj, child, j1, thr)
    if a > 0 and b > 0:
        return float(a ** (1 - w) * b ** w)
    return float((1 - w) * a + w * b)


def pool_above(d, child, thr):
    pid = d["pid"]
    sel = pid == child
    m = MASS.get(child, 0.0)
    e = np.sqrt(m * m + d["px"][sel].astype(float) ** 2 + d["py"][sel].astype(float) ** 2
                + d["pz"][sel].astype(float) ** 2)
    nev = int(d["n"][0])
    return (e >= thr).sum() / nev, np.sqrt((e >= thr).sum()) / nev


L = []
for proj, pn in ((2212, "p"), (211, "pi+"), (-211, "pi-")):
    L.append(f"\n=== {pn} + air; per interaction, pool/table (pool count) ; thresholds E_tot of the child {THR}")
    for i in range(0, 12):
        d = dict(np.load(f"{POOLS}/pool_{proj}_{i:03d}_2000.npz"))
        e0 = float(d["e0"][0])
        ek = e0 - MASS[proj]
        j = int(np.argmin(np.abs(np.log(eg / ek))))
        L.append(f"-- pool E_tot {e0:.2f} (T {ek:.2f}) vs table (interpolated in T)")
        for nm, c in KIDS:
            row = []
            for thr in THR:
                p, _e = pool_above(d, c, thr)
                t = table_above_i(proj, c, ek, thr)
                row.append(f"{p:6.3f}/{t:6.3f}={p / t if t > 1e-4 else np.nan:5.2f}")
            L.append(f"   {nm:4s} " + "  ".join(row))
open(os.environ["M"] + "/reports/yield_cmp.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
