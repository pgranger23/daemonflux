"""Quick in-process regression: sky._shard vs sky3d (cutoff/bend/field off)."""
import importlib.util  # noqa
import os, sys, time, pickle
import numpy as np
import sky, sky3d
W = os.environ["M"] + "/work/regress"
os.makedirs(W, exist_ok=True)
cache = os.environ["MC3D_POOL_CACHE"]
lat, lon = sky.SITES["kamioka"]
caps = (2.5, 5.0, 7.5, 10.0)
naz = int(sys.argv[1]) if len(sys.argv) > 1 else 12
sky._init(cache, 2000, False, 0.1, lat, lon, caps, 1, "H3A")
opts = sky3d.Options(cutoff=False, bend_primary=False, muon_field=False, n_az=naz)
sky3d._init(cache, 2000, 0.1, lat, lon, caps, "H3A", opts, None, None)
task = ("He", 2, 400, 40.0, 12345, 4000)
t = time.time(); d_old = sky._shard(task); print("old", time.time() - t)
tdir = sky3d.task_dir(W, "rg"); os.makedirs(tdir, exist_ok=True)
t = time.time()
r = sky3d.cascade_task(task + (0, sky3d.cfile(W, "rg", 0)))
print("cascade", time.time() - t, r)
t = time.time()
sky3d.trace_replay_task(([0], W, "rg"))
print("replay", time.time() - t)
d_new = dict(np.load(sky3d.tfile(W, "rg", 0)))
for a, b in (("s0", "s0"), ("s1", "s1"), ("t0", "t0"), ("t1", "t1"), ("n1", "s1"), ("v1", "t1")):
    x, y = d_new[a].sum(-1), d_old[b].sum(-1)
    rel = np.abs(x - y).max() / max(np.abs(y).max(), 1e-300)
    print(a, "vs", b, "max rel diff (az-summed)", rel, "sums", x.sum(), y.sum())
if naz == 1:
    for a in ("s1", "t1"):
        print(a, "per-bin max rel diff", np.abs(d_new[a] - d_old[a]).max() / np.abs(d_old[a]).max())
print("cone m rel", np.abs(d_new["cone_m"] - d_old["cone_m"]).max() / np.abs(d_old["cone_m"]).max())
print(dict(zip(d_new["m3b_count_names"], d_new["m3b_counts"])))
