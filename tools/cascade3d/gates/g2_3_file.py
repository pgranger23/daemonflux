"""The solver file of gate G2.3 from the cap solves (no comparison, no reference).

Stacks the cells of two cap solves (``run_solve3d.py combine``) on the cap axis of the
Monte Carlo tallies, the layout ``g2_3.py`` reads (``--solver``):

- cap of 10 degrees (index 3): the solve of G2.3;
- cap of 5 degrees (index 1): the solve of G2.3(g);
- caps of 2.5 and 7.5 degrees (indices 0 and 2): not solved, NaN.  ``g2_3.py`` reads
  the caps of 10 and 5 degrees only.

``phi_0`` (transmission 1) must be the same array in both solves.  Part (f) takes
``phi_straight_0`` from the straight-ahead solve of C1 with the same kernel set
(``c1/close/g13/pool2000_straight.npz``).  Part (h) is not provided: the solver's
ground lines end at the horizon.

Checked: cap, grid, M, guard and kernel set of each solve (``cost_json`` of the
tangents), the axes of the cells, the kernel set of the straight-ahead solve.  The
output is never overwritten.

Usage::

    $PY tools/cascade3d/gates/g2_3_file.py --cap10 CAP10_B_DIR --cap5 CAP5_B_DIR \
        --out SOLVER.npz [--cells cells_M8.npz] [--M 8] [--straight FILE]
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

CAPS = (2.5, 5.0, 7.5, 10.0)       # cap axis of the tallies of sky3d.py
SPACING = (5.0, 1.25)              # grid B
N3 = os.environ.get("N3", "/home/pigrange/mc3d_work/runs/native3d")
STRAIGHT = os.path.join(N3, "c1", "close", "g13", "pool2000_straight.npz")


def _stop(msg):
    raise SystemExit("g2_3_file: " + msg)


def solve_key(metas, cap, M):
    """Guard and kernel set of one solve, after the checks of its boundary."""
    if not metas:
        _stop("no tangents file")
    for m in metas:
        b = m["boundary"]
        if b.get("cap_deg") != cap or tuple(np.atleast_1d(b["spacing_deg"])) != SPACING:
            _stop(f"boundary {b}: expected cap {cap} on the grid {SPACING}")
        if m["M"] != M:
            _stop(f"M = {m['M']}, expected {M}")
    keys = {(json.dumps(m["guard"], sort_keys=True), m["ks_sha256"]) for m in metas}
    if len(keys) != 1:
        _stop("guard or kernel set differs between the tangents files")
    return keys.pop()


def load_cells(path, axes):
    c = np.load(path, allow_pickle=False)
    for k, v in axes.items():
        if k == "species":
            if tuple(str(x) for x in c[k]) != tuple(v):
                _stop(f"species of {path}")
        elif c[k].shape != np.shape(v) or not np.allclose(c[k], v, rtol=1e-12, atol=0):
            _stop(f"{k} of {path}")
    return np.asarray(c["phi_cut"], float), np.asarray(c["phi_0"], float)


def main(argv=None):
    import assemble as A
    import assemble3d as A3
    import g2_4

    ap = argparse.ArgumentParser()
    ap.add_argument("--cap10", required=True)
    ap.add_argument("--cap5", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", default="cells_M8.npz")
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--straight", default=STRAIGHT)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        _stop(f"{a.out} exists")
    axes = {"e_band_edges_GeV": np.asarray(A.G13_BAND_EDGES),
            "cosz_edges": np.asarray(A.G13_COSZ_EDGES),
            "azimuth_edges_deg": np.asarray(A3.AZ_EDGES),
            "species": A.G13_SPECIES}
    dirs = {10.0: a.cap10, 5.0: a.cap5}
    keys = {cap: solve_key(g2_4.solve_meta(d), cap, a.M) for cap, d in dirs.items()}
    if keys[10.0] != keys[5.0]:
        _stop("the two solves differ in guard or kernel set")
    cells = {cap: load_cells(os.path.join(d, a.cells), axes) for cap, d in dirs.items()}
    if not np.array_equal(cells[10.0][1], cells[5.0][1]):
        _stop("phi_0 differs between the two solves")
    shape = cells[10.0][0].shape
    phi_cut = np.full((len(CAPS),) + shape, np.nan)
    phi_0 = np.full((len(CAPS),) + shape[:-1], np.nan)
    for cap, (pc, p0) in cells.items():
        phi_cut[CAPS.index(cap)], phi_0[CAPS.index(cap)] = pc, p0

    s = np.load(a.straight, allow_pickle=False)
    sm = json.loads(str(s["cost_json"]))
    if not sm.get("straight") or g2_4.sha256(sm["ks"]) != keys[10.0][1]:
        _stop("the straight-ahead solve is not of the kernel set of the cap solves")
    if not (np.allclose(s["cell_band_edges"], axes["e_band_edges_GeV"], rtol=1e-12)
            and np.allclose(s["cell_cosz_edges"], axes["cosz_edges"], rtol=1e-12)):
        _stop("cells of the straight-ahead solve")
    straight = np.stack([s["cell_" + sp] for sp in A.G13_SPECIES])

    src = {"cap10": os.path.join(a.cap10, a.cells),
           "cap5": os.path.join(a.cap5, a.cells), "straight": a.straight}
    info = {"sources": {k: [p, g2_4.sha256(p)] for k, p in src.items()}, "M": a.M,
            "spacing_deg": list(SPACING), "ks_sha256": keys[10.0][1],
            "guard": json.loads(keys[10.0][0]), "caps_not_solved_deg": [2.5, 7.5]}
    np.savez(a.out, phi_cut=phi_cut, phi_0=phi_0, phi_straight_0=straight,
             cap_theta_deg=np.asarray(CAPS), info_json=json.dumps(info),
             **{k: np.asarray(v) for k, v in axes.items()})
    print(f"wrote {a.out}: phi_cut {phi_cut.shape}, caps solved 10 and 5 deg")
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
