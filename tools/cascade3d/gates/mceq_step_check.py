"""Review check for G1.1: are MCEq's own references converged in the step size?

Independent of ``mceq_euler.py`` / ``euler_refine.py`` (which re-implement MCEq's
march): this script runs **MCEq's own solver** and only divides its two step
controls, ``config.stability_margin`` and ``config.dXmax``, by n.  MCEq's forward
Euler step is ``min(margin / (max_ldec / rho), margin / max_lint, dXmax)``.

For each requested direction it reports, for the G1.1 quantities and every MCEq bin
from 0.1 to 100 GeV:

- ``ref / limit``: the default-step flux (n = 1, which must equal the stored G1.1
  reference) over the first-order Richardson limit of the two finest runs;
- ``solver / limit``: the cascade3d flux of the stored G1.1 run over the same limit.

Usage::

    source $N3/env.sh
    taskset -c 4-5 $PY tools/cascade3d/gates/mceq_step_check.py PHYS off 0,3,5 1,4,16
    $PY tools/cascade3d/gates/mceq_step_check.py --report PHYS off   # table only

Output ``$N3/c1/review/mceq_step_<pop>_<loss>.{md,npz}``.  Cost: about 7 ms per MCEq
step; cos Z = 0.05 at n = 16 is 297 098 steps (35 minutes on one core).
"""

import importlib.util  # noqa: F401
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import kernels_mceq as K  # noqa: E402

COS_Z = np.array([1.0, 0.75, 0.5, 0.25, 0.15, 0.05])
QUANT = ["numu", "antinumu", "nue", "antinue", "mu"]
OUT_DIR = os.path.join(K.N3, "c1", "review")


def fluxes(run):
    out = {n: np.asarray(run.get_solution(n, mag=0), float) for n in QUANT[:4]}
    out["mu"] = np.asarray(run.get_solution("total_mu+", mag=0), float) + np.asarray(
        run.get_solution("total_mu-", mag=0), float)
    return out


def run(pop, loss, idx, ns):
    """Solve with MCEq at each step refinement; save the fluxes; return the path."""
    tag = "losson" if loss else "lossoff"
    from MCEq import config as cfg

    margin0, dxmax0 = cfg.stability_margin, cfg.dXmax
    mceq = K.make_mceq(pop, energy_loss=loss)
    save = {"e_grid": np.asarray(mceq.e_grid, float), "cos_z": COS_Z[idx],
            "index": np.array(idx), "n": np.array(ns)}
    t0 = time.time()
    try:
        for i in idx:
            steps = []
            for n in ns:
                cfg.stability_margin, cfg.dXmax = margin0 / n, dxmax0 / n
                mceq.set_theta_deg(float(np.degrees(np.arccos(COS_Z[i]))))
                mceq.integration_path = None
                mceq.solve()
                for q, v in fluxes(mceq).items():
                    save[f"{q}_n{n}_{i}"] = v
                steps.append(int(mceq.integration_path[0]))
                print(f"cosZ {COS_Z[i]} n {n}: {steps[-1]} steps, "
                      f"{time.time() - t0:.0f} s", flush=True)
            save[f"steps_{i}"] = np.array(steps)
    finally:
        cfg.stability_margin, cfg.dXmax = margin0, dxmax0
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"mceq_step_{pop}_{tag}.npz")
    np.savez_compressed(path, **save)
    return path


def report(pop, loss):
    """Build the result file from the saved fluxes, the reference and the solver."""
    tag = "losson" if loss else "lossoff"
    base = os.path.join(OUT_DIR, f"mceq_step_{pop}_{tag}")
    z = np.load(base + ".npz", allow_pickle=False)
    ref = np.load(os.path.join(K.N3, "c1", "ref", f"mceq_g11_{pop}_{tag}.npz"),
                  allow_pickle=False)
    sol = np.load(os.path.join(K.N3, "c1", "g11", f"g11_{pop}_{tag}.npz"),
                  allow_pickle=False)
    e, ns = z["e_grid"], [int(n) for n in z["n"]]
    idx = ([int(i) for i in z["index"]] if "index" in z.files else
           [int(np.flatnonzero(np.isclose(COS_Z, c))[0]) for c in z["cos_z"]])
    sel = (e >= 0.1 - 1e-9) & (e <= 100.0 + 1e-6)
    n_lo, n_hi = ns[-2], ns[-1]
    lines = [f"# MCEq step convergence, {pop}, energy loss {'on' if loss else 'off'}",
             "",
             "MCEq's own solver with `stability_margin` and `dXmax` divided by n = "
             f"{ns}. Limit: first-order Richardson on n = {n_lo} and {n_hi}. "
             "Deviations in %: the worst bin of 0.1 to 100 GeV, with its energy. "
             "`ref` is the stored G1.1 reference (MCEq's default step), `solver` the "
             "stored cascade3d G1.1 run. The last column is the size of the last "
             "Richardson correction, an upper bound on what the limit still misses.",
             ""]
    for i in idx:
        # the solver file is ordered by its own ``mu0``, not by its ``cos_z`` key
        j = int(np.flatnonzero(np.isclose(sol["mu0"], COS_Z[i]))[0])
        same = max(np.abs(z[f"{q}_n{ns[0]}_{i}"][sel] / ref[q][i][sel] - 1).max()
                   for q in QUANT) if ns[0] == 1 else np.nan
        st = (", ".join(f"n={n}: {s}" for n, s in zip(ns, z[f"steps_{i}"]))
              if f"steps_{i}" in z.files else "see the log")
        lines += [f"## cos Z = {COS_Z[i]}", "",
                  f"Steps: {st}. n = 1 against the stored G1.1 reference: "
                  f"{same:.1e}.", "",
                  "| quantity | ref/limit - 1 | at GeV | solver/limit - 1 | at GeV | "
                  "solver/ref - 1 | at GeV | limit bound |",
                  "|---|---|---|---|---|---|---|---|"]
        lim = {}
        for q in QUANT:
            f_lo, f_hi = z[f"{q}_n{n_lo}_{i}"], z[f"{q}_n{n_hi}_{i}"]
            lim[q] = f_hi + (f_hi - f_lo) / (n_hi / n_lo - 1.0)
            a = ref[q][i][sel] / lim[q][sel] - 1
            b = sol[q][j][sel] / lim[q][sel] - 1
            c = sol[q][j][sel] / ref[q][i][sel] - 1
            unc = np.abs(lim[q][sel] / f_hi[sel] - 1).max()
            ka, kb, kc = (int(np.abs(v).argmax()) for v in (a, b, c))
            lines.append(f"| {q} | {100 * a[ka]:+.3f} | {e[sel][ka]:.3g} | "
                         f"{100 * b[kb]:+.3f} | {e[sel][kb]:.3g} | "
                         f"{100 * c[kc]:+.3f} | {e[sel][kc]:.3g} | {100 * unc:.3f} |")
        lines += ["", "Muons bin by bin below 1 GeV (%):", "",
                  "| E [GeV] | ref/limit - 1 | solver/limit - 1 | solver/ref - 1 |",
                  "|---|---|---|---|"]
        for k in np.where(sel & (e < 1.0))[0]:
            lines.append(f"| {e[k]:.3f} | "
                         f"{100 * (ref['mu'][i][k] / lim['mu'][k] - 1):+.2f} | "
                         f"{100 * (sol['mu'][j][k] / lim['mu'][k] - 1):+.2f} | "
                         f"{100 * (sol['mu'][j][k] / ref['mu'][i][k] - 1):+.2f} |")
        lines.append("")
    with open(base + ".md", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    if sys.argv[1] == "--report":
        report(sys.argv[2], sys.argv[3] == "on")
        return
    pop, loss = sys.argv[1], sys.argv[2] == "on"
    run(pop, loss, [int(i) for i in sys.argv[3].split(",")],
        [int(n) for n in sys.argv[4].split(",")])
    report(pop, loss)


if __name__ == "__main__":
    main()
