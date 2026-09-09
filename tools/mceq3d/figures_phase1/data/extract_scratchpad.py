"""One-off extraction of the small arrays the figure script needs out of the
Phase-1 session scratchpad (which is temporary) into this directory, so
``make_phase1_figures.py`` is self-contained.

Source (session scratchpad, 2026-09-03..05):
  pen/penumbra.npz        -> penumbra_ladders.npz   (fig 01 inset, fig 02)
  msgrid_ray_n30000_s3.npz-> muonseg_bend.npz       (fig 10)
  pattern/pat_a.npz       -> pattern_cz005.npz      (fig 07, detector anchor)
  anchor/pat_after.npz    -> pattern_cz005.npz      (fig 07, prod_point anchor)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S = sys.argv[1] if len(sys.argv) > 1 else os.environ["SCRATCH"]
SP = ("total_numu", "total_antinumu", "total_nue", "total_antinue")

# ---- fig 01 inset + fig 02 : fine admittance ladders ---------------------
d = np.load(os.path.join(S, "pen", "penumbra.npz"))
np.savez_compressed(os.path.join(HERE, "penumbra_ladders.npz"),
                    **{k: d[k] for k in ("vertical_R", "vertical_A",
                                         "87E_R", "87E_A",
                                         "87W_R", "87W_A",
                                         "bundle_87E_R", "bundle_87E_A")})

# ---- fig 10 : muon-segment bending angles at decay -----------------------
d = np.load(os.path.join(S, "msgrid_ray_n30000_s3.npz"), allow_pickle=True)
out = {}
for zen in ("87",):
    for az in ("E", "W"):
        for q, tag in (("1", "p"), ("-1", "m")):      # muon charge
            k = f"{zen}_{az}_0.5_{q}_bend_deg"
            if k in d.files:
                out[f"bend_{zen}_{az}_{tag}"] = d[k].astype(np.float32)
np.savez_compressed(os.path.join(HERE, "muonseg_bend.npz"), **out)

# ---- fig 07 : azimuth pattern at cosZ = 0.05, both cutoff anchors --------
det = np.load(os.path.join(S, "pattern", "pat_a.npz"))
pro = np.load(os.path.join(S, "anchor", "pat_after.npz"))
i_det = int(np.argmin(np.abs(det["cz"] - 0.05)))
i_pro = int(np.argmin(np.abs(pro["cz"] - 0.05)))
np.savez_compressed(
    os.path.join(HERE, "pattern_cz005.npz"),
    e=det["e"], az=det["az"], cz=0.05,
    cutoff_det=det["cutoff"][i_det], cutoff_prod=pro["cutoff"][i_pro],
    **{f"det_{s}": det[s][i_det] for s in SP},
    **{f"prod_{s}": pro[s][i_pro] for s in SP})

for f in ("penumbra_ladders.npz", "muonseg_bend.npz", "pattern_cz005.npz"):
    print(f, os.path.getsize(os.path.join(HERE, f)), "bytes")
