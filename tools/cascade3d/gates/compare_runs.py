"""Compare two ``run_solve.py`` outputs quantity by quantity (cycle 2b, task A).

Every stored quantity (the five G1.1 ground quantities at all ground lines and
energies, and the G1.3 cell arrays ``cell_*``): largest relative difference where the
reference is non-zero, and whether the reference's exact zeros stay exactly zero.
Undefined cells (NaN: the six G1.1 lines have no cos Z edges) must be NaN in both.

Usage::

    $PY tools/cascade3d/gates/compare_runs.py reference.npz candidate.npz [tol]
"""

import sys

import numpy as np

KEYS = ("numu", "antinumu", "nue", "antinue", "mu")


def compare(ref_path, new_path):
    a, b = np.load(ref_path), np.load(new_path)
    keys = [k for k in KEYS if k in a.files]
    keys += [k for k in a.files if k.startswith("cell_") and "edges" not in k]
    out = {}
    for k in keys:
        x, y = a[k], b[k]
        fin = np.isfinite(x)
        nz = fin & (x != 0)
        rel = (float(np.max(np.abs(y[nz] / x[nz] - 1))) if nz.any() else 0.0)
        zeros_ok = bool(np.all(y[fin & (x == 0)] == 0))
        # non-finite where the reference is finite (cells undefined on the six
        # G1.1 lines are NaN in both and must stay so)
        nan = bool(np.any(~np.isfinite(y[fin])) or np.any(np.isfinite(y[~fin])))
        out[k] = (rel, zeros_ok, nan)
    return out


def main(argv):
    ref, new = argv[1], argv[2]
    tol = float(argv[3]) if len(argv) > 3 else 1e-9
    res = compare(ref, new)
    worst = max(v[0] for v in res.values())
    ok = worst <= tol and all(v[1] and not v[2] for v in res.values())
    for k, (rel, z, nan) in res.items():
        print(f"{k:14s} max rel {rel:.2e}  zeros kept {z}  non-finite {nan}")
    print(f"worst {worst:.2e}; tolerance {tol:g}: {'ACCEPT' if ok else 'REJECT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
