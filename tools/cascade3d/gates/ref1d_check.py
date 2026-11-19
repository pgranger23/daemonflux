"""Straight-ahead consistency check on the Monte Carlo-matched sets (cycle 2, task G).

Compares the solver's four ground neutrino fluxes (``run_solve.py --ks ... --straight
--gate-lines --r-earth 6371``) with the builder's dense reference
``$N3/c1/ref1d/ref1d_ks_mc_<set>_L32_lossoff_R6371.npz`` (keys ``<nu>_<i>``, i = 0, 1, 2
for cos Z = 1, 0.5, 0.15) per bin from 0.1 to 10 GeV.  Not a pre-registered gate.
Writes ``$N3/c1/cycle2/ref1d_check/REF1D_CHECK.md``.

Usage::

    $PY tools/cascade3d/gates/ref1d_check.py pool2000 fresh1e5
"""

import os
import sys

import numpy as np

N3 = os.environ.get("N3", "/home/pigrange/mc3d_work/runs/native3d")
DIR = os.path.join(N3, "c1", "cycle2", "ref1d_check")
NUS = ("numu", "antinumu", "nue", "antinue")
COSZ = (1.0, 0.5, 0.15)


def main(argv):
    sets = argv[1:] or ["pool2000", "fresh1e5"]
    lines = ["# Straight-ahead check: cascade3d against the builder's dense reference",
             "", __doc__.split("Usage")[0].strip(), "",
             "| set | nu | cos Z | worst solver/ref - 1 | at GeV | rms | ref "
             "convergence 2->4 |", "|---|---|---|---|---|---|---|"]
    worst_all = 0.0
    for st in sets:
        a = np.load(os.path.join(DIR, f"sa_{st}.npz"))
        b = np.load(os.path.join(N3, "c1", "ref1d",
                                 f"ref1d_ks_mc_{st}_L32_lossoff_R6371.npz"))
        E = a["e_grid"]
        assert np.allclose(E, b["e_grid"][:E.size])
        sel = (E >= 0.1 - 1e-9) & (E <= 10.0 + 1e-6)
        for q in NUS:
            for i, c in enumerate(COSZ):
                j = int(np.flatnonzero(np.isclose(a["mu0"], c))[0])
                ref = b[f"{q}_{i}"][:E.size]
                r = a[q][j][sel] / ref[sel] - 1
                m = int(np.argmax(np.abs(r)))
                conv = np.max(np.abs(b[f"{q}_{i}_coarse"][:E.size][sel] / ref[sel] - 1))
                worst_all = max(worst_all, abs(r[m]))
                lines.append(f"| {st} | {q} | {c:g} | {r[m]:+.2e} | {E[sel][m]:.3g} | "
                             f"{np.sqrt(np.mean(r * r)):.1e} | {conv:.1e} |")
    lines += ["", f"Largest |solver/ref - 1|: {worst_all:.2e} "
              f"({'within' if worst_all <= 3e-3 else 'ABOVE'} 0.3%).", "",
              "Command: `$PY tools/cascade3d/gates/ref1d_check.py "
              f"{' '.join(sets)}`", ""]
    with open(os.path.join(DIR, "REF1D_CHECK.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv)
