"""Gate G1.1 (straight-ahead limit against MCEq); writes ``$N3/c1/gates/G1.1.md``.

For PHYS and REF, loss off and loss on: the straight-ahead kernel set
(``kernelset.from_mceq``) is solved on the six G1.1 lines (in the straight-ahead limit
lines do not couple, so this is exactly the per-line computation of the full family;
G1.2 checks it numerically) with the frozen default settings, and the ground fluxes of
numu, antinumu, nue, antinue and mu+ + mu- are compared with MCEq's
(``$N3/c1/ref/mceq_g11_*.npz``) in every MCEq bin from 0.1 to 100 GeV.

Diagnosis (not part of the gate decision): when present, the Euler-step study
``$N3/c1/diag/euler_*.npz`` (``euler_refine.py``: MCEq's own matrices marched with
2, 4, 8, 16 times finer steps) is used to estimate MCEq's step-converged value and the
solver is compared with it as well.

Usage::

    taskset -c 0-3 $PY tools/cascade3d/gates/g1_1.py [PHYS REF]
"""

import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import assemble as A  # noqa: E402
import grid  # noqa: E402
import kernelset as K  # noqa: E402
import mceq_euler as ME  # noqa: E402
import rays  # noqa: E402
import solver as S  # noqa: E402

N3 = ME.N3
OUT = os.path.join(N3, "c1", "gates", "G1.1.md")
RES = os.path.join(N3, "c1", "g11")
TOL = {False: 0.003, True: 0.02}

GATE = """**G1.1 Straight-ahead limit against MCEq (gated).** Same database, population,
atmosphere, primary model and observation level in both. Quantities: nu_mu, anti-nu_mu,
nu_e, anti-nu_e and mu+ + mu- at the ground. Energies: every MCEq bin from 0.1 to
100 GeV.
Directions: cos Z = 1, 0.75, 0.5, 0.25, 0.15, 0.05.
- Energy loss off in both: every cell within 0.3%.
- Energy loss on in both: every cell within 2%."""



def pinned():
    """The process affinity as a taskset list, e.g. "0-3"."""
    c = sorted(os.sched_getaffinity(0))
    parts, start = [], c[0]
    for a, b in zip(c, c[1:] + [None]):
        if b != a + 1:
            parts.append(f"{start}-{a}" if a > start else f"{a}")
            start = b
    return ",".join(parts)

def solve_case(pop, loss):
    atm = grid.CorsikaAtmosphere()
    lev = grid.make_levels(atm)
    fam = grid.LineFamily(np.array(grid.G11_COSZ), np.zeros(0))
    r = rays.Rays(atm, lev, fam)
    ks = K.from_mceq(pop, loss=loss)
    sv = S.Solver(r, ks)
    t = time.time()
    out = sv.solve()
    gf = A.ground_fluxes(sv, out)
    return gf, ks, float(np.max(out["res"])), time.time() - t, r.summary()


def euler_limit(pop, loss_s, i, q):
    """MCEq's step-converged estimate from the refinement files, or None."""
    vals = {}
    pat = os.path.join(N3, "c1", "diag", f"euler_{pop}_{loss_s}_{i}*.npz")
    for f in glob.glob(pat):
        z = np.load(f)
        for n in z["refine"]:
            if f"r{n}_{q}" in z.files:
                vals[int(n)] = z[f"r{n}_{q}"]
    if len(vals) < 2:
        return None, None
    ns = sorted(vals)
    a, b = vals[ns[-2]], vals[ns[-1]]
    # first-order Richardson on the two finest steps (ratio of step sizes 2)
    return 2 * b - a, ns


def main(argv):
    pops = argv[1:] or ["PHYS", "REF"]
    os.makedirs(RES, exist_ok=True)
    lines = ["# Gate G1.1: straight-ahead limit against MCEq", "", GATE, "",
             "Solver settings (frozen defaults, `DESIGN.md`): levels 0.4 km to 50 km, "
             "1 km to 70 km, 3 km to the top (plus layer boundaries); n_sub 4; "
             "dX_max 10 g/cm2; "
             "6 Jacobi sweeps; gamma0 2.", ""]
    summary = {}
    all_ok = True
    for pop in pops:
        for loss in (False, True):
            ls = "losson" if loss else "lossoff"
            gf, ks, res, t, summ = solve_case(pop, loss)
            z = np.load(ME.ref_path(pop, loss))
            E = ks.e_grid
            sel = np.flatnonzero((E >= 0.1 - 1e-9) & (E <= 100.0 + 1e-6))
            np.savez(os.path.join(RES, f"g11_{pop}_{ls}.npz"), e_grid=E,
                     cos_z=z["cos_z"], **{q: gf[q] for q in A.QUANTITIES},
                     mu0=gf["mu0"])
            tab = ["| quantity | " + " | ".join(f"cos Z {c:g}" for c in z["cos_z"])
                   + " |", "|---" * (1 + len(z["cos_z"])) + "|"]
            worst = (0.0, None)
            diag = []
            for q in A.QUANTITIES:
                row = [q]
                for i, c in enumerate(z["cos_z"]):
                    j = int(np.flatnonzero(np.isclose(gf["mu0"], c))[0])
                    rel = gf[q][j][sel] / z[q][i][sel] - 1
                    m = int(np.argmax(np.abs(rel)))
                    row.append(f"{rel[m]:+.2e} ({E[sel][m]:.3g})")
                    if abs(rel[m]) > abs(worst[0]):
                        worst = (rel[m], (q, c, E[sel][m]))
                    lim, ns = euler_limit(pop, ls, i, q)
                    if lim is not None:
                        rl = gf[q][j][sel] / lim[sel] - 1
                        rm = z[q][i][sel] / lim[sel] - 1
                        mm = int(np.argmax(np.abs(rl)))
                        diag.append((q, c, rl[mm], E[sel][mm], rm[mm], ns))
                tab.append("| " + " | ".join(row) + " |")
            ok = abs(worst[0]) <= TOL[loss]
            all_ok &= ok
            summary[f"{pop}_{ls}"] = {"worst": worst[0], "where": worst[1], "pass": ok}
            lines += [f"## {pop}, energy loss {'on' if loss else 'off'} "
                      f"(tolerance {TOL[loss]:.1%})", "",
                      "Largest |solver / MCEq - 1| over 0.1 to 100 GeV per quantity "
                      "and "
                      "direction (energy of that cell in GeV):", ""] + tab + [
                      "", f"Worst cell: {worst[0]:+.3%} ({worst[1][0]}, cos Z "
                      f"{worst[1][1]:g}, E = {worst[1][2]:.3g} GeV).  Same-bin "
                      f"residual max {res:.1e}.  Solve {t:.0f} s.  "
                      f"**{'PASS' if ok else 'FAIL'}**",
                      ""]
            if diag:
                lines += ["Diagnosis against MCEq's step-converged estimate "
                          "(first-order "
                          "Richardson on the two finest Euler refinements):", "",
                          "| quantity | cos Z | solver / limit - 1 (worst) | E [GeV] | "
                          "MCEq ref / limit - 1 there | refinements |",
                          "|---|---|---|---|"
                          "---|---|"]
                for q, c, a, e, b, ns in diag:
                    lines.append(f"| {q} | {c:g} | {a:+.2e} | {e:.3g} | {b:+.2e} | "
                                 f"{ns} |")
                lines.append("")
    lines += [f"## Result: **{'PASS' if all_ok else 'FAIL'}**", "",
              f"Command: `taskset -c {pinned()} $PY tools/cascade3d/gates/g1_1.py "
              f"{' '.join(pops)}`", ""]
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines))
    with open(OUT.replace(".md", ".json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    print("\n".join(lines))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
