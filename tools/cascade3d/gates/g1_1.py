"""Gate G1.1 (straight-ahead limit against MCEq); writes ``$N3/c1/gates/G1.1.md``.

For PHYS and REF, loss off and loss on: the straight-ahead kernel set
(``kernelset.from_mceq``) is solved on the six G1.1 lines (in the straight-ahead limit
lines do not couple, so this is exactly the per-line computation of the full family;
G1.2 checks it numerically) with the frozen default settings, and the ground fluxes of
numu, antinumu, nue, antinue and mu+ + mu- are compared with MCEq's
(``$N3/c1/ref/mceq_g11_*.npz``) in every MCEq bin from 0.1 to 100 GeV.

Diagnosis (not part of the gate decision): the solver is also compared with two
estimates of MCEq's step-converged value, ``$N3/c1/diag/euler_*.npz``
(``euler_refine.py``) and ``$N3/c1/review/mceq_step_*.npz`` (``mceq_step_check.py``).

Output directory: ``$C3D_OUTDIR`` (default ``$N3/c1/cycle2``): ``gates/G1.1.md`` and
``g11/g11_<pop>_<loss>.npz`` (rows in the order of the ``cos_z`` key).

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
OUTDIR = os.environ.get("C3D_OUTDIR", os.path.join(N3, "c1", "cycle2"))
OUT = os.path.join(OUTDIR, "gates", "G1.1.md")
RES = os.path.join(OUTDIR, "g11")
REVIEW = os.path.join(N3, "c1", "review")
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
    summ = r.summary()
    summ.update(settings_of(sv, r))
    return gf, ks, float(np.max(out["res"])), time.time() - t, summ


def settings_of(sv, r):
    """Numerical settings read from the run's own objects (for the header)."""
    lev = r.levels
    dh = np.diff(lev)
    return {"n_levels": int(lev.size), "dh_min_km": float(dh.min() / 1e5),
            "dh_max_km": float(dh.max() / 1e5), "n_sub": r.n_sub,
            "dX_max": r.dX_max, "n_sweeps": sv.n_sweeps, "gamma0": sv.gamma0,
            "n_up": sv.n_up, "projection": sv.projection, "k_top": sv.k_top,
            "r_e_km": float(r.r_e / 1e5), "lines": int(r.Nl), "nodes": int(r.N)}


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


def review_limit(pop, loss_s, i, q):
    """Step limit of the independent check (``gates/mceq_step_check.py``) or None."""
    f = os.path.join(REVIEW, f"mceq_step_{pop}_{loss_s}.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f)
    key = f"{q}_limit_{i}"
    return z[key] if key in z.files else None


def main(argv):
    pops = argv[1:] or ["PHYS", "REF"]
    os.makedirs(RES, exist_ok=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    lines = ["# Gate G1.1: straight-ahead limit against MCEq", "", GATE, "",
             "Solver settings: printed per case below, read from the run's own "
             "`Rays` and `Solver` objects.", ""]
    summary = {}
    all_ok = True
    for pop in pops:
        for loss in (False, True):
            ls = "losson" if loss else "lossoff"
            gf, ks, res, t, summ = solve_case(pop, loss)
            z = np.load(ME.ref_path(pop, loss))
            E = ks.e_grid
            sel = np.flatnonzero((E >= 0.1 - 1e-9) & (E <= 100.0 + 1e-6))
            # rows in the order of the cos_z key that labels them
            rows = [int(np.flatnonzero(np.isclose(gf["mu0"], c))[0])
                    for c in z["cos_z"]]
            np.savez(os.path.join(RES, f"g11_{pop}_{ls}.npz"), e_grid=E,
                     cos_z=z["cos_z"], **{q: gf[q][rows] for q in A.QUANTITIES})
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
                    rv = review_limit(pop, ls, i, q)
                    if lim is not None or rv is not None:
                        out = [q, c]
                        for L_ in (lim, rv):
                            if L_ is None:
                                out += [None, None, None]
                                continue
                            rl = gf[q][j][sel] / L_[sel] - 1
                            rm = z[q][i][sel] / L_[sel] - 1
                            mm = int(np.argmax(np.abs(rl)))
                            out += [rl[mm], E[sel][mm], rm[mm]]
                        diag.append(tuple(out))
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
                      f"**{'PASS' if ok else 'FAIL'}**", "",
                      "Settings of this run: " + ", ".join(
                          f"{k_} {v_}" for k_, v_ in summ.items()
                          if k_ in ("n_levels", "dh_min_km", "dh_max_km", "n_sub",
                                    "dX_max", "n_sweeps", "gamma0", "n_up",
                                    "projection", "k_top", "r_e_km", "lines",
                                    "nodes")) + ".",
                      ""]
            if diag:
                lines += ["Not a gate: against MCEq's step-converged estimates, (a) "
                          "this project's (`euler_refine.py`: MCEq's matrices, steps / "
                          "2..16, Richardson on the two finest) and (b) the "
                          "independent check (`mceq_step_check.py`: MCEq's solver, "
                          "steps / 4 and / 16). Worst cell of 0.1 to 100 GeV.", "",
                          "| quantity | cos Z | solver/limit(a) - 1 | E | ref/limit(a) "
                          "- 1 | solver/limit(b) - 1 | E | ref/limit(b) - 1 |",
                          "|---|---|---|---|---|---|---|---|"]

                def fmt(x, f):
                    return "n/a" if x is None else format(x, f)

                for q, c, a1, e1, b1, a2, e2, b2 in diag:
                    lines.append(f"| {q} | {c:g} | {fmt(a1, '+.2e')} | "
                                 f"{fmt(e1, '.3g')} | {fmt(b1, '+.2e')} | "
                                 f"{fmt(a2, '+.2e')} | {fmt(e2, '.3g')} | "
                                 f"{fmt(b2, '+.2e')} |")
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
