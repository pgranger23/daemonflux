import importlib.util  # noqa
import os, sys, pickle
import numpy as np
import analyse_m3b as a
M = os.environ["M"]
J = a.load(M + "/sky", "a1rot", G=20)
ax = a.ew_axis()
names = [str(x) for x in np.load(M + "/sky/a1rot_tasks/t00000.npz")["m3b_count_names"]]
C = J.S["cnt"].sum(0)
ic = names.index("cpu_cascade")
casc = C[:, ic] / J.nsh                     # cascade CPU per primary (one realisation)
cost0 = J.cpu / J.nsh                        # total CPU per primary, no resim
est, rep = a.a1_obs(J, ax)
Vq = a.jvar(rep[..., 0])
Vh = []
for h in range(J.K):
    _e, r_h = a.a1_obs(J, ax, stratum=h)
    Vh.append(a.jvar(r_h[..., 0]))
Vh = np.array(Vh)                            # (K, 4, 3, 3)
ok = Vq > 0
rel = np.where(ok, Vh / np.where(ok, Vq, 1), 0).reshape(J.K, -1)
share = rel.sum(1) / rel.sum()
groups = ["p", "He", "CNO", "MgSi", "Fe"]
out = {}
print("stratum  n  ms/prim  var-share")
for h, (g, k) in enumerate(J.strata):
    print(f"{g:5s}{k} {J.nsh[h]:8.0f} {1e3*cost0[h]:7.1f} {share[h]:.3f}")
for thr in (0.0, 10.9, 20.0, 33.0, 60.0):
    rpp = a.realisations_per_primary(groups, 7, thr, n=1500)
    mult = np.array([rpp[s][0] for s in J.strata])
    cost = cost0 + casc * (mult - 1.0)
    a_h = J.nsh * rel.sum(1)
    w = np.sqrt(np.maximum(a_h, 0) / cost)
    frac = np.maximum(w / w.sum(), 0.002)
    frac /= frac.sum()
    budget = 60 * 3600.0
    n_new = frac * budget / (frac * cost).sum()
    # predicted summed relative variance: sum_q sum_h V_hq n_h / n'_h / V_q
    pred = (rel * (J.nsh / n_new)[:, None]).sum()
    # E^-0.8 (the A allocation) at the same budget and the same cost model
    nA = J.nsh * budget / (J.nsh * cost).sum()
    predA = (rel * (J.nsh / nA)[:, None]).sum()
    nq = rel.shape[1]
    print(f"thr {thr:5.1f} GV: Neyman  N = {n_new.sum():.3g} primaries in 60 core-h, "
          f"mean rel var {pred/nq:.4f} (rms rel err x{np.sqrt(pred/nq):.3f} of A's);"
          f" E^-0.8 alloc: N = {nA.sum():.3g}, x{np.sqrt(predA/nq):.3f}")
    out[thr] = dict(frac=frac, cost=cost, n=n_new, mult=mult)
    print("   frac: " + " ".join(f"{g}{k}:{f:.3f}" for (g, k), f in zip(J.strata, frac) if f > 0.003))
pickle.dump({"strata": J.strata, "out": out, "share": share, "cost0": cost0, "casc": casc},
            open(M + "/reports/alloc_a1.pkl", "wb"))
