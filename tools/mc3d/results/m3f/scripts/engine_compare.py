"""MC run vs engine cells (rigfix files): per species, 12 cells.
    python engine_compare.py STRATA_TAG ENGINE.npz LABEL"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "m3c", "scripts"))
import analyse_m3b as a
import m3c_lib as L
tag, eng, lab = sys.argv[1:4]
M = os.environ["M"]
IDX = [0, 1, 5, 6, 7]; NAMES = ("a1/a0", "dphi", "fN", "fS", "N/S")
def cells(ew):
    x = ew[0].sum(2)
    return np.array([[a.observables(x[k, b], L.axis())[IDX] for b in range(3)] for k in range(4)])
r = L.Run(tag, M + "/work")
est = cells(r.tot); rep = np.array([cells(x) for x in r.rep]); G = len(rep)
dev = rep - rep.mean(0); dev[..., 1] = (dev[..., 1] + 180) % 360 - 180
err = np.sqrt((G - 1.0) / G * (dev ** 2).sum(0))
pats = a.engine_patterns(eng).sum(2)
en = np.array([[a.observables(pats[k, b], L.axis())[IDX] for b in range(3)] for k in range(4)])
d = est - en; d[..., 1] = (d[..., 1] + 180) % 360 - 180
z = d / err
out = [f"# {tag} minus engine {os.path.basename(eng)} ({lab}); 12 pooled cells, jackknife errors of the MC only"]
out.append("species   " + "  ".join(f"{n:>22s}" for n in NAMES))
for k, sp in enumerate(L.SP):
    out.append(f"{sp:9s} " + "  ".join(f"{d[k,:,j].mean():+.3f}+-{np.sqrt((err[k,:,j]**2).sum())/3:.3f} chi2/3={np.sum(z[k,:,j]**2):.1f}".rjust(22) for j in range(5)))
out.append("all 12    " + "  ".join(f"{d[...,j].mean():+.3f}+-{np.sqrt((err[...,j]**2).sum())/12:.3f} chi2/12={np.sum(z[...,j]**2)/12:.2f}".rjust(22) for j in range(5)))
for (i, j, nm) in ((3, 2, "antinumu-numu"), (0, 1, "nue-antinue")):
    out.append(f"fan-out {nm}: MC " + " ".join(f"{x:+.3f}" for x in est[i,:,0]-est[j,:,0]) + "  engine " + " ".join(f"{x:+.3f}" for x in en[i,:,0]-en[j,:,0]))
open(f"{M}/reports/engine_compare_{lab}.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
