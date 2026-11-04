"""Gate C3 candidate (a): the penumbra of the 1D site cutoff in the gate's
bins.  Ground launch, sea-level floor, IGRF-14 -- the convention of
``cutoff.SiteCutoffMap`` (whose ``R_c`` is the upper cutoff ``R_U``).  For
each direction: the allowed verdict on a 0.1 GV ladder, ``R_L`` (lowest
allowed), ``R_U`` (highest forbidden), the effective cutoff
``R_eff = R_U - int_{R_L}^{R_U} allowed dR`` and the map's value.

    python penumbra.py  ->  $M/reports/penumbra.txt, $M/work/penumbra.npz
"""
import os
import time

import numpy as np

import cutoff
import sky3d

LAT, LON = 36.4267, 137.31
M = os.environ["M"]
smap = sky3d.site_map(LAT, LON, "IGRF14", cutoff.DEFAULT_DATE, 1, M + "/work")
RL = np.arange(3.0, 14.0001, 0.1)
bins = {"vertical": (np.array([5.0, 10.0, 15.0]), np.array([0.0, 90.0, 180.0, 270.0])),
        "87W 240-270": (np.array([84.5, 86.0, 87.0, 88.0, 89.0, 89.5]),
                        np.array([242.5, 247.5, 252.5, 257.5, 262.5, 267.5])),
        "87W 270-300": (np.array([84.5, 86.0, 87.0, 88.0, 89.0, 89.5]),
                        np.array([272.5, 277.5, 282.5, 287.5, 292.5, 297.5]))}
lines, out = [], {"R": RL}
t0 = time.time()
for lab, (zens, azs) in bins.items():
    Z, A = np.meshgrid(zens, azs, indexing="ij")
    r, u = cutoff.state_above_site(LAT, LON, Z.ravel(), A.ravel(), 0.0)
    n = len(r)
    rr = np.repeat(r, len(RL), 0)
    uu = np.repeat(u, len(RL), 0)
    RR = np.tile(RL, n)
    ok = cutoff.trace_allowed(rr, uu, RR, 1, floor="sea_level").reshape(n, len(RL))
    rmap = smap(Z.ravel(), A.ravel())
    out[lab + "|allowed"] = ok
    out[lab + "|zen"] = Z.ravel()
    out[lab + "|az"] = A.ravel()
    out[lab + "|rmap"] = rmap
    lines.append(f"## {lab}: allowed fraction vs R (0.1 GV ladder), "
                 f"{time.time() - t0:.0f} s")
    for i in range(n):
        a = ok[i]
        rl = RL[np.argmax(a)] if a.any() else np.nan
        ru = RL[np.where(~a)[0].max()] if (~a).any() else np.nan
        sel = (RL >= rl) & (RL <= ru)
        reff = ru - a[sel].sum() * 0.1 if np.isfinite(rl) else np.nan
        lines.append(f"zen {Z.ravel()[i]:5.1f} az {A.ravel()[i]:6.1f}: map R_c "
                     f"{rmap[i]:6.2f}  ladder R_L {rl:6.2f} R_U {ru:6.2f} "
                     f"R_eff {reff:6.2f}  width R_U-R_L {ru - rl:5.2f} GV")
    print("\n".join(lines[-n - 1:]), flush=True)
open(M + "/reports/penumbra.txt", "w").write("\n".join(lines) + "\n")
np.savez(M + "/work/penumbra.npz", **out)
