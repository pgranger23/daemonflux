"""Per-(jackknife group, stratum) sums for the muon-bend response table:
 x = [s1 (largest cap, field on, all entries), mucf_full[0] (muon-born, field on), mucf_full[1] (muon-born, zero-field cf)]
on the fine grid (sp, E, cz, az) -> $M/work/addon_<TAG>.npz.   python addon_extract.py RUNDIR TAG"""
import os, sys
import numpy as np
sys.path.insert(0, os.environ["R"] + "/tools/mc3d")
import analyse_sky as A
rundir, tag = sys.argv[1], sys.argv[2]
area = {}
def red(d):
    a = float(np.asarray(d["cap_area"])[-1])
    s = np.asarray(d["s1"])[-1] * a            # s1 is divided by the cap area: undo, so all three are unnormalised
    mf = np.asarray(d["mucf_full"])
    area["a"] = a
    return np.stack([s, mf[0], mf[1]])
J = A.JackRun(rundir, tag, G=20, reducers={"x": red})
np.savez(os.path.join(os.environ["M"], "work", f"addon_{tag}.npz"), x=J.S["x"], M=J.M, cpu=J.cpu, nsh=J.nsh,
         strata=np.array([f"{g}:{k}" for g, k in J.strata]), area=area["a"])
print("done", J.S["x"].shape)
