import os, sys
import numpy as np
sys.path.insert(0, os.environ["R"] + "/tools/mc3d")
from constants import MASS
import pool as P
pl = P.EventPool(os.environ["MC3D_POOL_CACHE"], readonly=True)
rng = np.random.default_rng(3)
print("# raw pool events at E0 (no rescale): sum of kept-product energies vs avail = E_tot + 0.94")
for i in (0, 2, 4, 5, 7, 9, 13, 20):
    d = pl._load_or_make(2212, i)
    e0 = float(d["e0"][0]); n = int(d["n"][0])
    off = d["off"]
    pid = d["pid"]; m = np.array([MASS.get(int(q), 0.0) for q in pid])
    e = np.sqrt(m**2 + d["px"].astype(float)**2 + d["py"].astype(float)**2 + d["pz"].astype(float)**2)
    tot = np.add.reduceat(e, off[:-1])[: n]
    avail = e0 + 0.94
    s = np.minimum(1, avail / tot)
    nuc = np.isin(pid, (2212, 2112))
    print(f"E0 {e0:7.2f}: fraction tot>avail {np.mean(tot > avail):.3f}; mean scale {s.mean():.3f}; mean tot {tot.mean():.2f} avail {avail:.2f}; slow nucleon rest-mass share {(np.add.reduceat((m*nuc), off[:-1])[:n]).mean():.2f}")
print("# the guard at the MC's energies (draw at E_tot): fraction of draws with scale<1, mean scale")
for ek in (1.5, 2.5, 4.0, 6.0, 10.0, 20.0, 40.0):
    et = ek + 0.938
    sc = []
    for _ in range(4000):
        out = pl.draw(rng, 2212, et)
        # recompute what the guard saw: draw() already applied it; infer scale via energy sum
    # instrumented replica of draw()
    c = pl.grid.centres
    ss = []
    for _ in range(4000):
        i = pl.bin_choice(rng, et); d = pl._load_or_make(2212, i)
        k = int(rng.integers(int(d["n"][0]))); a, b = int(d["off"][k]), int(d["off"][k + 1])
        if b <= a: continue
        f = P._p_beam(2212, et) / P._p_beam(2212, float(d["e0"][0]))
        pid = d["pid"][a:b]; m = np.array([MASS.get(int(q), 0.0) for q in pid])
        e = np.sqrt(m**2 + d["px"][a:b].astype(float)**2 + d["py"][a:b].astype(float)**2 + (d["pz"][a:b].astype(float) * f)**2)
        ss.append(min(1.0, (et + 0.94) / e.sum()))
    ss = np.array(ss)
    print(f"E_p {ek:5.1f}: fraction scaled {np.mean(ss < 1):.3f}; mean scale {ss.mean():.3f}; min {ss.min():.3f}")
