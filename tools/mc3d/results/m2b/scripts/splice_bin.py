import importlib.util  # noqa: F401
import numpy as np
import MCEq.config as cfg
cfg.e_min = 0.05
cfg.debug_level = 0
import crflux.models as crf
from MCEq.core import MCEqRun
R = "/afs/cern.ch/work/p/pigrange/daemonflux"
tab = dict(np.load(R + "/tools/mc3d/mceq_tables_SIBYLL23D.npz"))
mc = MCEqRun(interaction_model="DPMJETIII193",
             primary_model=(crf.HillasGaisser2012, "H3a"), theta_deg=0.0)
eg = mc.e_grid
xr = eg[:, None] / eg[None, :]
for p, c in ((2212, 211), (2212, 2212), (211, 211)):
    N = (mc._interactions.get_matrix((p, 0), (c, 0)) * xr).astype(np.float32)
    a = tab[f"y_{p}_{c}"]
    print(f"{p}->{c}: " + "  ".join(
        f"{eg[j]:.1f}:{'=' if np.array_equal(a[:, j], N[:, j]) else 'x'}"
        f"{a[:, j].sum()/N[:, j].sum():.3f}" for j in range(len(eg))
        if 25 < eg[j] < 120))
