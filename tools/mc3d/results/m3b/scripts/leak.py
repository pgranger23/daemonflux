"""Pending-test leakage gate: pend_all vs the pending test on the same task."""
import importlib.util  # noqa
import os, sys, time
import numpy as np
import sky, sky3d, field
M = os.environ["M"]
W = M + "/gates/leak"
os.makedirs(W, exist_ok=True)
cache = os.environ["MC3D_POOL_CACHE"]
lat, lon = sky.SITES["kamioka"]
caps = (2.5, 5.0, 7.5, 10.0)
g, k, n, rthr = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
res = {}
for pa in (False, True):
    opts = sky3d.Options(r_thr=rthr, pend_all=pa)
    fld = field.FieldGrid(opts.model, opts.date, cache_dir=M + "/work/fieldcache")
    smap = sky3d.site_map(lat, lon, opts.model, opts.date, 1, M + "/work")
    sky3d._init(cache, 2000, 0.1, lat, lon, caps, "H3A", opts, fld, smap)
    tag = f"lk_{g}{k}_{int(pa)}"
    os.makedirs(sky3d.task_dir(W, tag), exist_ok=True)
    t = time.time()
    r = sky3d.cascade_task((g, k, n, 40.0, 4242 + k, n, 0, sky3d.cfile(W, tag, 0)))
    sky3d.trace_replay_task(([0], W, tag))
    print("pend_all", pa, "time", round(time.time() - t, 1), r)
    res[pa] = dict(np.load(sky3d.tfile(W, tag, 0)))
for tg in ("s", "n", "t", "v"):
    a, b = res[False][tg + "1"], res[True][tg + "1"]
    ew = (slice(None), slice(None), slice(8, 30), slice(10, 13))   # 0.25-3 GeV, cz<0.3
    print(tg, "total pend/all %.6f" % (a.sum() / max(b.sum(), 1e-300)),
          "  E-W region %.6f" % (a[ew].sum() / max(b[ew].sum(), 1e-300)),
          "  max |diff| / max %.2e" % (np.abs(a - b).max() / max(np.abs(b).max(), 1e-300)))
