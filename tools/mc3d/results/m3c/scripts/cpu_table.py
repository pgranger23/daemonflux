"""CPU (sum of the tasks' process times) and wall of the milestone-3c runs."""
import os
import numpy as np
M = os.environ["M"]
tot = 0.0
for tag in ("a1nc", "a1ncnomu", "a1ncnb", "a1ancd", "c3anc", "c3res", "a1ncnbnomu"):
    fp = f"{M}/sky/{tag}_total.npz"
    if not os.path.exists(fp):
        print(f"{tag}: not finished")
        continue
    d = np.load(fp)
    tm = d["task_meta"]                 # [i, meta(9), counts]
    cpu = tm[:, 1 + 7].sum() / 3600
    nsh = tm[:, 1 + 0].sum()
    man = np.load(f"{M}/sky/{tag}_tasks/manifest.npz")
    wall = float(man["wall_s"][0]) / 3600 if "wall_s" in man.files else float("nan")
    tot += cpu
    print(f"{tag:11s} {nsh:9.4g} primaries  {cpu:6.2f} core-h  wall {wall:5.2f} h")
print(f"total MC {tot:.1f} core-h")
