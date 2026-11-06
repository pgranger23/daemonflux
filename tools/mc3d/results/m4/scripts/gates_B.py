"""Gates 5-7 for the production table (milestone 4): extracts B's per-(jackknife group, stratum) design sums once
(-> $M/work/B_design.npz, x = (G, K, 4 tag [s,n,t,v], sp, E 10/dec x20, cos Z down x10, az x12)) and compares with pilot-2."""
import os, sys
import numpy as np
sys.path.insert(0, os.environ["R"] + "/tools/mc3d")
import analyse_sky as A
M, W = os.environ["M"], os.environ["W"]
def red(d):
    out = []
    for t in ("s1", "n1", "t1", "v1"):
        x = np.asarray(d[t])[-1][:, :40, 10:, :]
        out.append(x.reshape(4, 20, 2, 10, 12).sum(2))
    return np.stack(out)
J = A.JackRun(M + "/sky", "B", G=20, reducers={"x": red})
np.savez(M + "/work/B_design.npz", x=J.S["x"], M=J.M, cpu=J.cpu, nsh=J.nsh, strata=np.array([f"{g}:{k}" for g, k in J.strata]))
print("done", J.S["x"].shape)
