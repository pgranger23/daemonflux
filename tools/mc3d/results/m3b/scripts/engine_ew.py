import importlib.util  # noqa
import os, sys, time
sys.path.insert(0, os.environ["R"] + "/tools/mceq3d")
os.chdir(os.environ["R"] + "/tools/mceq3d")   # engine reads data files relative to cwd (read only; cache_dir is ours)
import analyse_m3b
t = time.time()
prim = None
tag = sys.argv[1]
if tag == "h3a":
    prim = ("HillasGaisser2012", "H3a")
out = os.environ["M"] + f"/work/engine_ew_{tag}.npz"
analyse_m3b.engine_ew(out, os.environ["M"] + "/work/engine_cache", cz_sub=1, primary=prim)
print("done", out, time.time() - t)
