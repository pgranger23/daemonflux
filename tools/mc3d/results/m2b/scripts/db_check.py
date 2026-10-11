"""Task 4: which MCEq database does tools/mc3d/mceq_tables_SIBYLL23D.npz come from?

Rebuild build_tables.build()'s export for SIBYLL23D *and* DPMJETIII193 from the
installed MCEq (default config) and compare against the shipped npz."""
import importlib.util  # noqa: F401
import os
import sys
import numpy as np

R = "/afs/cern.ch/work/p/pigrange/daemonflux"
sys.path.insert(0, R + "/tools/mc3d")
import MCEq.config as cfg
print("MCEq version file:", __import__("MCEq").__file__)
print("config.mceq_db_fname =", cfg.mceq_db_fname)
print("config.data_dir      =", cfg.data_dir)
print("data_dir listing     =", sorted(os.listdir(cfg.data_dir)))

npz = dict(np.load(R + "/tools/mc3d/mceq_tables_SIBYLL23D.npz"))
print("npz model/emin:", npz["model"], npz["emin"], " n_E =", len(npz["e_grid"]))

cfg.e_min = float(npz["emin"][0])
cfg.debug_level = 0
import crflux.models as crf
from MCEq.core import MCEqRun

out = {}
for model in ("SIBYLL23D", "DPMJETIII193"):
    mc = MCEqRun(interaction_model=model,
                 primary_model=(crf.HillasGaisser2012, "H3a"), theta_deg=0.0)
    eg = mc.e_grid
    xr = eg[:, None] / eg[None, :]
    I = mc._interactions
    d = {}
    for p in (2212, 2112, 211, -211, 321):
        for c in (2212, 2112, 211, -211, 321, -321, 130, 13, -13):
            try:
                M = I.get_matrix((p, 0), (c, 0))
            except Exception:
                continue
            N = (M * xr).astype(np.float32)
            if N.sum() > 1e-12:
                d[(p, c)] = N
    d["cs_2212"] = mc._int_cs.get_cs(2212)
    out[model] = (eg, d)
    del mc

eg = out["SIBYLL23D"][0]
assert np.allclose(eg, npz["e_grid"])
print("\nE_kin grid matches the npz grid")
print("\n(1) shipped npz vs a fresh SIBYLL23D export (identity of the build):")
for (p, c) in ((2212, 211), (2212, -211), (2212, 2212), (211, 211), (2212, 321)):
    a = npz[f"y_{p}_{c}"]
    b = out["SIBYLL23D"][1][(p, c)]
    print(f"   y_{p}_{c}: max|diff| = {np.abs(a - b).max():.3e}  (sum {a.sum():.4e})")
print("\n(2) column-wise SIBYLL23D(npz) / DPMJETIII193 total yield per "
      "interaction, and max relative cell difference, vs projectile E_kin:")
for (p, c) in ((2212, 211), (2212, -211), (2212, 2212), (2212, 2112),
               (211, 211), (211, -211), (2212, 321)):
    a = npz[f"y_{p}_{c}"].astype(float)
    b = out["DPMJETIII193"][1][(p, c)].astype(float)
    print(f"  {p}->{c}")
    for e in (2.8, 4.5, 8.9, 17.8, 44.7, 70.8, 89.1, 112.0, 1000.0):
        j = int(np.argmin(np.abs(eg - e)))
        sa, sb = a[:, j].sum(), b[:, j].sum()
        m = (np.abs(a[:, j]) + np.abs(b[:, j])) > 0
        rel = np.max(np.abs(a[m, j] - b[m, j]) / np.maximum(np.abs(b[m, j]), 1e-30)) if m.any() else 0.0
        print(f"     E_kin={eg[j]:8.2f}  sum S23D={sa:9.5f}  sum DPM={sb:9.5f}  "
              f"ratio={sa/sb if sb else np.nan:8.5f}  max cell rel diff={rel:.2e}")
cs_a, cs_b = npz["cs_2212"], out["DPMJETIII193"][1]["cs_2212"]
print("\n(3) p-air inelastic cross section ratio S23D/DPMJET at E_kin = "
      + ", ".join(f"{eg[j]:.1f}: {cs_a[j]/cs_b[j]:.4f}" for j in
                  [int(np.argmin(np.abs(eg - e))) for e in (5, 20, 50, 79, 100, 1000)]))
np.savez("/home/pigrange/mc3d_work/runs/m2b/db_check_dpm.npz",
         e_grid=eg, **{f"y_{p}_{c}": v for (p, c), v in out["DPMJETIII193"][1].items()
                       if isinstance(p, int)})
