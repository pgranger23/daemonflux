"""Exact cutoff R_U of MC-like injection states that arrive in the East 60-90 / cos Z 0-0.1 bin,
vs the ground-anchored site map at the crossing-frame (zenith, azimuth)."""
import os, sys
import numpy as np
sys.path.insert(0, os.environ["R"] + "/tools/mc3d")
import sky, sky3d, cutoff, geometry as geo
from scoring import NestedCaps
W = os.environ["W"]; M = os.environ["M"]
lat, lon = sky.SITES["kamioka"]
man = dict(np.load(W + "/runs/m3e/sky/pilot2_tasks/manifest.npz"))
caps = NestedCaps(lat, lon, tuple(man["caps"]))
opts = sky3d.Options()
smap = sky3d.site_map(lat, lon, opts.model, opts.date, 1, W + "/runs/m3e/work")
axis = np.array(caps.axis, float)
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
AZ0 = float(os.environ.get("AZ0", "60"))
NS = int(float(sys.argv[2])) if len(sys.argv) > 2 else 4_000_000
keep_r, keep_u, keep_loc = [], [], []
for _ in range(NS // 500_000):
    r0, u0 = geo.sample_injection_patch(rng, sky.R_INJ_CM, axis, 40.0, 500_000)
    i0, cz, az, cosi = sky3d.locate_vec(caps, r0, u0)
    sel = (i0 >= 0) & (cz >= 0.0) & (cz < 0.1) & (az >= AZ0) & (az < AZ0 + 30)
    keep_r.append(r0[sel]); keep_u.append(u0[sel]); keep_loc.append(np.stack([i0[sel], cz[sel], az[sel]], 1))
r = np.concatenate(keep_r); u = np.concatenate(keep_u); loc = np.concatenate(keep_loc)
print("states in the bin:", len(r), " by i0:", np.bincount(loc[:, 0].astype(int)))
zen = np.degrees(np.arccos(loc[:, 1]))
rc_map = smap(zen, loc[:, 2])
out = {}
for fl in ("injection", "sea_level"):
    rc, sat = cutoff.upper_cutoff(r, u, 1, floor=fl, gated=True, n_jobs=1)
    out[fl] = rc
    d = rc / rc_map - 1
    print(f"floor={fl:10s}: R_U(state)/R_c(map at crossing frame) - 1: mean {d.mean():+.4f} median {np.median(d):+.4f} rms {d.std():.4f}; "
          f"mean R_U {rc.mean():.2f}, mean map {rc_map.mean():.2f}")
    for c in range(4):
        m = loc[:, 0] <= c
        if m.sum():
            print(f"   cap<= {c}: n={m.sum()} mean {d[m].mean():+.4f} rms {d[m].std():.4f}")
# flux-weighted (E^-1.7 integral) effective cutoff: <R_c^-1.7>^(-1/1.7) vs map
g = 1.7
for fl, rc in out.items():
    eff = np.mean(rc ** -g) ** (-1 / g); effm = np.mean(rc_map ** -g) ** (-1 / g)
    print(f"floor={fl:10s}: effective cutoff (integral index {g}): states {eff:.2f}  map {effm:.2f}  ratio {eff/effm:.4f}  -> v ratio ~ {(eff/effm)**-g:.4f}")
np.savez(M + "/work/states_az" + os.environ.get("AZ0", "60") + ".npz", r=r, u=u, loc=loc, rc_inj=out["injection"], rc_sea=out["sea_level"], rc_map=rc_map)
