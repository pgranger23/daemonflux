"""Cone origin check: pion production angle per interaction, chromo pools
(DPMJET-III-19.3 < 80 GeV, SIBYLL-2.3d above) vs the factorised engine's
generator moments m_spliced_v2.npz (UrQMD-3.4 <= 80 GeV, SIBYLL-2.3d above),
p + air, theta = arcsin(p_T / p), RMS in bins of pion energy."""
import os
import sys
import numpy as np
R = "/afs/cern.ch/work/p/pigrange/daemonflux"
sys.path.insert(0, R + "/tools/mc3d")
from constants import MASS
import pool as P

os.chdir(R + "/tools/mceq3d")          # read-only: the engine's relative paths
sys.path.insert(0, R + "/tools/mceq3d")
eng = dict(np.load("m_spliced_v2.npz"))
EB = np.array([0.3, 0.5, 1.0, 2.0, 5.0])
pe = eng["proj_energies"]


def engine_row(e_proj):
    i = int(np.argmin(np.abs(np.log(pe / e_proj))))
    es, t2, w = eng["e_sec"][i], eng["theta_sq"][i], eng["dndx"][i]
    out = []
    for lo, hi in zip(EB[:-1], EB[1:]):
        m = (es >= lo) & (es < hi) & np.isfinite(t2) & (w > 0)
        out.append(np.sqrt(np.average(t2[m], weights=w[m])) if m.any() else np.nan)
    return pe[i], np.degrees(np.array(out))


print("RMS production angle [deg] of pi+ in p+air, per pion-energy bin "
      + " ".join(f"[{a},{b})" for a, b in zip(EB[:-1], EB[1:])))
for e_proj in (10.0, 20.0, 45.0, 100.0, 300.0, 1000.0):
    i = P.GRID.index(e_proj)
    d = dict(np.load(f"/home/pigrange/mc3d_work/pools/pool_2212_{i:03d}_2000.npz"))
    pid = d["pid"]
    px, py, pz = (d[k].astype(float) for k in ("px", "py", "pz"))
    p = np.sqrt(px * px + py * py + pz * pz)
    e = np.sqrt(p * p + MASS[211] ** 2)
    th = np.arcsin(np.clip(np.sqrt(px * px + py * py) / np.maximum(p, 1e-12), 0, 1))
    sel = pid == 211
    rms = []
    for lo, hi in zip(EB[:-1], EB[1:]):
        m = sel & (e >= lo) & (e < hi)
        rms.append(np.degrees(np.sqrt(np.mean(th[m] ** 2))) if m.sum() > 50 else np.nan)
    ep, er = engine_row(float(d["e0"][0]))
    print(f"  E_proj {float(d['e0'][0]):7.1f} ({d['model'][0]:12s}) pool: "
          + " ".join(f"{x:6.2f}" for x in rms)
          + f" | engine row {ep:7.1f}: " + " ".join(f"{x:6.2f}" for x in er)
          + " | ratio " + " ".join(f"{a/b:5.2f}" for a, b in zip(rms, er)))
from kinematic_kernel import meson_theta2
e_m, t2 = meson_theta2("pi")
print("engine pooled <theta^2>^0.5 (what sigma_pi is built from) at E_pi = "
      + ", ".join(f"{x}: {np.degrees(np.sqrt(np.exp(np.interp(np.log(x), np.log(e_m), np.log(t2))))):.2f}"
                  for x in (0.4, 0.7, 1.4, 3.0)))
