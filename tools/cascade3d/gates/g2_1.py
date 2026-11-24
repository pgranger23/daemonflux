"""Gate G2.1 (transmission 1 reproduces C1); writes G2.1.md / .json.

Gate text: ``GATES.md`` section C2 (G2.1) and "G2.1 and G2.5(b)" of the entry of
2026-11-23 (quoted in the result file).  How it is evaluated:

- The assembly of G2.3: the cap of 10 deg on grid B (5 deg in x, 1.25 deg in y,
  ``boundary.LocationGrid((5, 1.25)).cap_weights(10)``), the 85 sky tables with their
  transmission set to 1 everywhere (``allowed`` true and no line node flagged as
  never reaching the injection sphere; the chains of the solver's line family that
  never reach it, transmission 0 in the solves, also get 1),
  ``assemble3d.grid_deficit``, the orders 0..M
  (``assemble3d.order_tangents``).  The deficit is checked to be exactly zero, so the
  response of the linear tangent map is exactly zero (``test_solver3d``:
  zero boundary tangent gives zero bit for bit) and no tangent is solved.
- Phi_C1: the primal of the cap solve of G2.3 (``tangents_*.npz`` of its directory,
  the same in every file, checked), the ground table of the C1 scheme.
- Cells: ``assemble3d.azimuth_table`` and ``azimuth_cells``, every G1.3 cell (four
  species x ten bands x ten cos Z rows) in each of the twelve azimuth bins.
- Pass: every cell within 0.2% of ``c1/close/g13/pool2000.npz`` (``cell_<species>``).

Usage::

    $PY tools/cascade3d/gates/g2_1.py --solve-dir CAP10_B_DIR --outdir OUT [--M 8]

Exit status 0 PASS, 1 FAIL, 2 input error.
"""

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

N3 = os.environ.get("N3", "/home/pigrange/mc3d_work/runs/native3d")
C1_SOLVE = os.path.join(N3, "c1", "close", "g13", "pool2000.npz")
SKY = os.path.join(N3, "c2", "solver", "sky")
TOL = 0.002
SPECIES = ("nue", "antinue", "numu", "antinumu")

GATE = """\
**G2.1 Transmission 1 reproduces C1 (gated).** The C2 solver with the transmission set
to 1 everywhere, assembled exactly as for G2.3 (location grid, interpolation, average
over the 10° cap), against the C1 solve `c1/close/g13/pool2000.npz`: every G1.3 cell, in
each of the twelve azimuth bins, within 0.2%.

**G2.1 and G2.5(b)** hold by construction in this design (D = 0 returns the C1 solve;
a rotation of the sky is a phase of the orders). They are run as written and test the
bookkeeping of the assembly only. G2.7 is the test of the m = 0 part."""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def unit_tables(grid, weights, sky_dir):
    """The tables of the cap with their transmission set to 1 everywhere."""
    import boundary as B

    out = {}
    for p, w in enumerate(weights):
        if w == 0.0:
            continue
        t = B.SkyTable.load(B.table_path(sky_dir, (grid.x[p], grid.y[p])))
        grid.check_table(t, p)
        d = dict(t.d)
        d["allowed"] = np.ones_like(d["allowed"], bool)
        d["never"] = np.zeros_like(d["never"], bool)
        d["never_state"] = np.zeros_like(d["never_state"], bool)
        out[p] = B.SkyTable(d, t.path)
    return out


def main(argv=None):
    import assemble3d as A3
    import boundary as B
    import g2_7
    import kernelset as K

    ap = argparse.ArgumentParser()
    ap.add_argument("--solve-dir", required=True)
    ap.add_argument("--outdir", default=os.path.join(N3, "c2", "solver", "gates",
                                                     "G2.1"))
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--sky-dir", default=SKY)
    ap.add_argument("--name", default="G2.1")
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)
    md, js = (os.path.join(a.outdir, a.name + s) for s in (".md", ".json"))
    for p in (md, js):
        if os.path.exists(p):
            raise SystemExit(f"{p} exists; result files are never overwritten")
    files = sorted(glob.glob(os.path.join(a.solve_dir, "tangents_*.npz")))
    if not files:
        raise SystemExit("no tangents files")
    f0 = np.load(files[0])
    I0, mu0 = f0["I"], f0["mu0"]
    for f in files[1:]:
        if not np.array_equal(np.load(f)["I"], I0):
            raise SystemExit("the primal differs between the files of the solve")
    meta = json.loads(str(f0["cost_json"]))
    if meta["boundary"].get("cap_deg") != 10.0 or tuple(
            meta["boundary"]["spacing_deg"]) != (5.0, 1.25):
        raise SystemExit("the solve is not the cap of 10 deg on grid B")
    ks = K.KernelSet.load(g2_7.KS_PATH)
    grid = B.LocationGrid((5.0, 1.25), 10.0)
    W = grid.cap_weights(10.0)
    tabs = unit_tables(grid, W, a.sky_dir)
    gw = B.GroupWeights(ks.e_bins, model="GSFG:GH")
    ry, _ks, _sha = g2_7.build_geometry()      # chains of the frozen geometry
    i0, i1, w0, w1, never = B.chain_weights(ry, B.line_nodes())
    # transmission 1 everywhere: also on the chains that never reach the injection
    # sphere (transmission 0 by the convention of the solves)
    cw = (i0, i1, w0, w1, np.zeros_like(never))
    D, used = A3.grid_deficit(a.sky_dir, grid, W, gw, ks, cw, tables=tabs)
    zero_deficit = bool(not np.any(D))
    if not zero_deficit:
        raise SystemExit("the deficit with transmission 1 is not exactly zero")
    tl = A3.order_tangents(D, a.M, ks)
    if any(np.any(t[2]) for t in tl):
        raise SystemExit("an order of the zero deficit is not zero")
    dI = np.zeros((len(tl),) + I0.shape)             # J[0] = 0 exactly
    Iaz = A3.azimuth_table(I0, dI, [(t[0], t[1]) for t in tl])
    cells = A3.azimuth_cells(mu0, Iaz, ks)            # (4, 10, 10, 12)
    ref = np.load(C1_SOLVE)
    st = np.stack([ref[f"cell_{s}"] for s in SPECIES])
    rel = np.abs(cells / st[..., None] - 1)
    ok = bool(np.all(rel <= TOL))
    i = np.unravel_index(np.argmax(rel), rel.shape)
    res = {"gate": "G2.1", "verdict": "PASS" if ok else "FAIL",
           "max_rel": float(rel.max()), "index": [int(x) for x in i],
           "n_tables": len(used), "M": a.M, "zero_deficit": zero_deficit,
           "inputs": {C1_SOLVE: sha256(C1_SOLVE), files[0]: sha256(files[0])},
           "script_sha256": sha256(os.path.abspath(__file__)),
           "command": " ".join(sys.argv)}
    with open(js, "w") as fh:
        json.dump(res, fh, indent=1)
    L = ["# Gate G2.1: transmission 1 reproduces C1", "", GATE, "",
         f"## Result: **{res['verdict']}**", "",
         f"Assembly of G2.3 (cap of 10 deg on grid B, {len(used)} tables with "
         f"non-zero weight, transmission set to 1 everywhere, orders 0..{a.M}): the "
         "deficit is exactly zero, so the tangent response is exactly zero. Largest "
         f"|cell / C1 - 1| over the 400 G1.3 cells x 12 azimuth bins: "
         f"**{rel.max():.3e}** (limit {TOL}).", "", "Inputs (sha256):", ""]
    L += [f"- `{p}` {h}" for p, h in res["inputs"].items()]
    L += ["", f"Script sha256: {res['script_sha256']}", "", "Command:", "",
          f"    {res['command']}", ""]
    with open(md, "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
