"""Mechanism of the horizon deficit of the 1D tally t: fraction of primaries whose line crosses the caps that make a
shower, straight line (what the 1D tally represents) vs bent in the field (what the cascade actually does), by
rigidity and crossing-frame cos Z.  Same sampled interaction depths for both."""
import os, sys
import numpy as np
sys.path.insert(0, os.environ["R"] + "/tools/mc3d")
import sky, sky3d, bend, geometry as geo, cutoff, field
from scoring import NestedCaps
from interactions import ChromoBackend
W = os.environ["W"]; M = os.environ["M"]
lat, lon = sky.SITES["kamioka"]
man = dict(np.load(W + "/runs/m3e/sky/pilot2_tasks/manifest.npz"))
caps = NestedCaps(lat, lon, tuple(man["caps"]))
opts = sky3d.Options()
fld = field.FieldGrid(opts.model, opts.date, cache_dir=M + "/work")
axis = np.array(caps.axis, float)
rng = np.random.default_rng(11)
backend = ChromoBackend(cache_dir=os.environ["MC3D_POOL_CACHE"], xs="mceq", readonly=True)
rs, us, czs = [], [], []
NS = int(float(sys.argv[1]))
for _ in range(NS // 500_000):
    r0, u0 = geo.sample_injection_patch(rng, sky.R_INJ_CM, axis, 40.0, 500_000)
    i0, cz, az, cosi = sky3d.locate_vec(caps, r0, u0)
    sel = (i0 >= 0) & (cz >= 0.0)
    if sel.sum() > 120_000:
        sel &= rng.random(len(sel)) < 120_000 / sel.sum() * (1.0 + 3 * (cz < 0.6))
    rs.append(r0[sel]); us.append(u0[sel]); czs.append(cz[sel])
r = np.concatenate(rs); u = np.concatenate(us); cz = np.concatenate(czs)
print("states", len(r), flush=True)
# rigidities log-uniform 2-60 GV (protons: T from p); lambda from the MC's table at the matching energy
Rg = np.exp(rng.uniform(np.log(1.0), np.log(60.0), len(r)))
T = np.sqrt(Rg ** 2 + 0.938 ** 2) - 0.938
lam = np.array([backend.lambda_int(2212, t + 0.938) for t in T])
dX = -np.log(rng.random(len(r))) * lam
st_b, *_ = bend.bend_primary(r, u, Rg, dX, fld)
st_s, *_ = bend.bend_primary(r, u, Rg, dX, None)
L = ["# P(first interaction in the atmosphere = shower) for primaries (protons) whose straight line crosses the caps",
     "# cosZ(crossing)   R[GV]      straight   bent   bent/straight   (n states)"]
for c0, c1 in ((0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.6), (0.6, 0.8), (0.8, 1.01)):
    for r0_, r1_ in ((1, 2), (2, 4), (4, 8), (8, 16), (16, 60)):
        m = (cz >= c0) & (cz < c1) & (Rg >= r0_) & (Rg < r1_)
        if m.sum() < 20:
            continue
        ps, pb = (st_s[m] == bend.P_INTERACT).mean(), (st_b[m] == bend.P_INTERACT).mean()
        L.append(f"{c0:.1f}-{c1:.1f}   {r0_:3d}-{r1_:<3d}   {ps:.3f}    {pb:.3f}    {pb/ps:.3f}   ({m.sum()})")
open(M + "/reports/bend_t_diag.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
