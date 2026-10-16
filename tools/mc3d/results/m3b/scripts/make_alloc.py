import importlib.util  # noqa
import os, pickle, sys
import numpy as np
import analyse_m3b as a
M = os.environ["M"]
thr, budget_h = float(sys.argv[1]), float(sys.argv[2])
J = a.load(M + "/sky", "a1rot", G=20)
ax = a.ew_axis()
names = [str(x) for x in np.load(M + "/sky/a1rot_tasks/t00000.npz")["m3b_count_names"]]
C = J.S["cnt"].sum(0)
casc = C[:, names.index("cpu_cascade")] / J.nsh
cost0 = J.cpu / J.nsh
groups = ["p", "He", "CNO", "MgSi", "Fe"]
rpp = a.realisations_per_primary(groups, 7, thr, n=3000)
mult = np.array([rpp[s][0] for s in J.strata])
cost = cost0 + casc * (mult - 1.0)
frac, a_h, Vh = a.allocation(J, ax, cost)
n = frac * budget_h * 3600.0 / (frac * cost).sum()
n = np.maximum(np.round(n), 300).astype(int)
print(f"thr {thr} GV, budget {budget_h} core-h -> N = {n.sum()}, predicted CPU {(n*cost).sum()/3600:.1f} core-h")
for (g, k), f, nn, c, m in zip(J.strata, frac, n, cost, mult):
    print(f"  {g:5s}{k} frac {f:.4f} n {nn:8d} cost {1e3*c:7.1f} ms/prim (x{m:.1f} cascades) -> {nn*c/3600:.2f} core-h")
np.savez(M + f"/reports/alloc_thr{thr:g}.npz", group=np.array([s[0] for s in J.strata]),
         stratum=np.array([s[1] for s in J.strata]), n=n, frac=frac, cost=cost, mult=mult)
