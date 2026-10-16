"""In-process smoke test of the full milestone-3b chain on one task."""
import importlib.util  # noqa
import os, sys, time
import numpy as np
import sky, sky3d, field, cutoff
M = os.environ["M"]
W = M + "/work/smoke"
os.makedirs(W, exist_ok=True)
cache = os.environ["MC3D_POOL_CACHE"]
lat, lon = sky.SITES["kamioka"]
caps = (2.5, 5.0, 7.5, 10.0)
g, k, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rthr = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0
opts = sky3d.Options(r_thr=rthr)
fld = field.FieldGrid(opts.model, opts.date, cache_dir=M + "/work/fieldcache")
smap = sky3d.site_map(lat, lon, opts.model, opts.date, 1, M + "/work")
sky3d._init(cache, 2000, 0.1, lat, lon, caps, "H3A", opts, fld, smap)
task = (g, k, n, 40.0, 777 + k, n)
tag = f"sm_{g}{k}"
tdir = sky3d.task_dir(W, tag); os.makedirs(tdir, exist_ok=True)
t = time.time()
r = sky3d.cascade_task(task + (0, sky3d.cfile(W, tag, 0)))
print("cascade", time.time() - t, r)
t = time.time()
sky3d.trace_replay_task(([0], W, tag))
print("trace+replay", time.time() - t)
d = dict(np.load(sky3d.tfile(W, tag, 0)))
c = dict(zip(d["m3b_count_names"], d["m3b_counts"]))
print({k_: round(float(v), 3) for k_, v in c.items()})
for tg in ("s", "n", "t", "v"):
    print(tg, "sum1", d[tg + "1"][-1].sum())
