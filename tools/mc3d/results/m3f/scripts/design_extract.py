"""Per-(jackknife group, stratum) sums of the full 3D (s) and 1D no-cutoff (t) tallies of a run
on a design grid: E 0.1-10 GeV at 10/decade (20 bins, pairs of Honda's 20/decade bins), cos Z 20 bins,
az 12 bins, 4 species, largest cap, measure 1.  -> $M/work/design_<TAG>.npz
    python design_extract.py RUNDIR TAG"""
import os, sys
import numpy as np
sys.path.insert(0, os.environ["R"] + "/tools/mc3d")
import analyse_sky as A
rundir, tag = sys.argv[1], sys.argv[2]
def red(d):
    out = []
    for t in ("s1", "t1", "v1"):
        x = np.asarray(d[t])[-1][:, :40]            # (sp, E(40), cz, az)
        out.append(x.reshape(x.shape[0], 20, 2, x.shape[2], x.shape[3]).sum(2))
    return np.stack(out)                            # (3, sp, 20, 20, 12)
J = A.JackRun(rundir, tag, G=20, reducers={"x": red})
np.savez(os.path.join(os.environ["M"], "work", f"design_{tag}.npz"), x=J.S["x"], M=J.M, cpu=J.cpu, nsh=J.nsh,
         strata=np.array([f"{g}:{k}" for g, k in J.strata]))
print("done", J.S["x"].shape)
