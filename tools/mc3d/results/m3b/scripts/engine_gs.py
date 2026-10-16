"""G_s(R_c, E) of the factorised engine (read-only use), H3a primary, sharp cutoff,
at the vertical and near the horizon -- for gate C3."""
import importlib.util  # noqa
import os, sys, time
import numpy as np
os.chdir(os.environ["M"] + "/work")
from mceq3d_flux import MCEq3DFlux
t = time.time()
eng = MCEq3DFlux(interaction_model="SIBYLL23D", primary=("HillasGaisser2012", "H3a"),
                 base_model="mceq")
rc = np.concatenate([np.arange(0.0, 20.0, 0.25), np.arange(20.0, 60.01, 1.0)])
out = {"e": eng.e, "rc": rc}
for czr in (0.95, 0.05, 0.15, 0.25):
    G, _ = eng.geomag_response(rc, cz_ref=czr, cache_dir=os.environ["M"] + "/work/engine_cache",
                               sigma_lnr=0.0)
    for s, v in G.items():
        out[f"{s}|{czr}"] = v
    print("cz_ref", czr, time.time() - t, flush=True)
np.savez(os.environ["M"] + "/work/engine_gs_h3a.npz", **out)
print("done", time.time() - t)
