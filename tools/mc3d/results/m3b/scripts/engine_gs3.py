"""G_s of the factorised engine (H3a, SIBYLL-2.3d, sharp cutoff) for gate C3:
vertical on the full grid (cached), 87 deg on a grid focused on the West and
East cutoffs (MCEq at 87 deg is ~1 min per solve)."""
import importlib.util  # noqa
import os, time
import numpy as np
os.chdir(os.environ["M"] + "/work")
from mceq3d_flux import MCEq3DFlux
t = time.time()
eng = MCEq3DFlux(interaction_model="SIBYLL23D", primary=("HillasGaisser2012", "H3a"),
                 base_model="mceq")
grids = {0.95: np.concatenate([np.arange(0.0, 20.0, 0.25), np.arange(20.0, 60.01, 1.0)]),
         0.05: np.array([0.0, 5.0, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 10.0, 12.0,
                         20.0, 25.0, 30.0, 33.0, 36.0, 39.0, 42.0, 45.0, 50.0, 60.0])}
out = {"e": eng.e}
for czr, rc in grids.items():
    G, _ = eng.geomag_response(rc, cz_ref=czr, cache_dir=os.environ["M"] + "/work/engine_cache",
                               sigma_lnr=0.0)
    out[f"rc|{czr}"] = rc
    for s, v in G.items():
        out[f"{s}|{czr}"] = v
    print("cz_ref", czr, time.time() - t, flush=True)
np.savez(os.environ["M"] + "/work/engine_gs_h3a.npz", **out)
print("done", time.time() - t)
