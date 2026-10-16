import importlib.util  # noqa
import os, sys
import numpy as np
import analyse_m3b as a, sky3d, cutoff
M = os.environ["M"]
tag = sys.argv[1]
J = a.load(M + "/sky", tag, G=20)
smap = sky3d.site_map(36.4267, 137.31, "IGRF14", cutoff.DEFAULT_DATE, 1, M + "/work")
lines = []
def out(s):
    print(s, flush=True); lines.append(s)
rows = a.gate_c3(J, M + "/work/engine_gs_h3a.npz", smap, out=out)
open(M + f"/reports/gate_c3_{tag}.txt", "w").write("\n".join(lines) + "\n")
