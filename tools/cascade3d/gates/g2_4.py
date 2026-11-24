"""Gate G2.4 (homogeneity), part (a): grid A against grid B; writes G2.4a.md / .json.

Gate text: ``GATES.md`` section C2 (G2.4), the entry "G2.4, the two variations"
(2026-11-23) and, replacing its pair (a), section 3 of the entry of 2026-11-24 (all
quoted in the result file).  Part (b) (reference altitude) is stage 2 and not in this
script.

- Inputs: the cells of two cap solves made with ``run_solve3d.py combine``: the cap of
  10 deg on grid B (5 deg in x, 1.25 deg in y; the grid of the solves) and on grid A
  (5 deg in x, 2.5 deg in y), both at M = 8, everything else equal (the solves'
  ``cost_json`` is checked: cap, M, guard, kernel set, code of the same commit).
- Cells: Phi_cut of G2.3, four neutrino species x ten bands (0.1 to 10 GeV) x ten
  cos Z rows x twelve azimuth bins of 30 deg (4800 cells).
- Pass: max |Phi_A - Phi_B| / Phi_B <= 0.005 over the 4800 cells (Phi_B, the finer
  grid and the grid of the solves, as the reference of the move).  Reported: the same
  with Phi_A as the denominator, cells above 0.1%, 0.3%, the worst cell per band, the
  cutoff factor range.

Usage::

    $PY tools/cascade3d/gates/g2_4.py a --cells-b CAP10_B.npz --cells-a CAP10_A.npz \
        --solve-b DIR_B --solve-a DIR_A --outdir OUT

Exit status 0 PASS, 1 FAIL, 2 input error.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

N3 = os.environ.get("N3", "/home/pigrange/mc3d_work/runs/native3d")
TOL = 0.005
SPECIES = ("nue", "antinue", "numu", "antinumu")

GATE = """\
**G2.4 Homogeneity (gated).** Halving the spacing of the location grid, and moving the
reference altitude at which a line of sight is attached to a location, each move no
cell of Φ_cut of G2.3 (cap of 10°) by more than 0.5%. The two variations (the spacings,
the altitudes) are fixed here as a dated entry once `DESIGN.md` names the location grid
and the reference altitude, before any C2 solve with the cutoff. Fixed on 2026-11-23
("C2, dated entries"), together with two additions: G2.7 (gated) and G2.8 (reported).

**G2.4, the two variations.**
- (a) Spacing of the location grid: 5° (25 nodes on [−10°, 10°]² in azimuthal
  equidistant coordinates about the site) against 2.5° (81 nodes, which contain the 25).

Pass for each, as the gate says: no cell of Φ_cut of G2.3 (cap of 10°, 4800 cells)
moves by more than 0.5%.

**3. G2.4(a): the pair is replaced before its run.** (entry of 2026-11-24)
- Not run: 5° against 2.5° (entry of 2026-11-23). G2.0 triggered on the 5° grid, the
  cap-averaged proton boundaries of the two differ by 1.65% ± 0.12% by number (cap of
  10°), so a failure was foreseen; and no solve uses the 5° grid.
- The pair now: grid A (5° in x, 2.5° in y, 45 nodes) against grid B (85 nodes, which
  contain the 45). Cap of 10°, M = 8, everything else equal.
- Pass, as the gate says: no cell of Φ_cut of G2.3 (4800 cells) moves by more than
  0.5% between A and B."""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def solve_meta(d):
    """cost_json of every tangents file of a solve directory."""
    import glob

    out = []
    for f in sorted(glob.glob(os.path.join(d, "tangents_*.npz"))):
        out.append(json.loads(str(np.load(f)["cost_json"])))
    return out


def check_pair(ma, mb):
    """Both solves: the same cap, M, guard and kernel set; grids A and B."""
    def one(ms):
        keys = {(m["boundary"].get("cap_deg"), m["M"], json.dumps(m["guard"]),
                 m["ks_sha256"]) for m in ms}
        sp = {tuple(m["boundary"]["spacing_deg"]) for m in ms}
        return keys, sp
    ka, sa = one(ma)
    kb, sb = one(mb)
    if len(ka) != 1 or ka != kb:
        raise SystemExit(f"the two solves differ in cap, M, guard or kernels: "
                         f"{ka} {kb}")
    if sa != {(5.0, 2.5)} or sb != {(5.0, 1.25)}:
        raise SystemExit(f"grids are not A and B: {sa} {sb}")
    (cap, M, _g, _k), = ka
    if cap != 10.0 or M != 8:
        raise SystemExit(f"cap {cap}, M {M}: the gate asks for 10 deg and M = 8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("a",))
    ap.add_argument("--cells-b", required=True)
    ap.add_argument("--cells-a", required=True)
    ap.add_argument("--solve-b", required=True)
    ap.add_argument("--solve-a", required=True)
    ap.add_argument("--outdir", default=os.path.join(N3, "c2", "solver", "gates",
                                                     "G2.4"))
    ap.add_argument("--name", default="G2.4a")
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)
    md, js = (os.path.join(a.outdir, a.name + s) for s in (".md", ".json"))
    for p in (md, js):
        if os.path.exists(p):
            raise SystemExit(f"{p} exists; result files are never overwritten")
    check_pair(solve_meta(a.solve_a), solve_meta(a.solve_b))
    cb, ca = np.load(a.cells_b), np.load(a.cells_a)
    for k in ("e_band_edges_GeV", "cosz_edges", "azimuth_edges_deg"):
        if not np.array_equal(cb[k], ca[k]):
            raise SystemExit(f"axes differ: {k}")
    if [str(s) for s in cb["species"]] != list(SPECIES):
        raise SystemExit("species order")
    pb, pa = cb["phi_cut"], ca["phi_cut"]
    if pb.shape != (4, 10, 10, 12) or np.any(pb <= 0) or np.any(pa <= 0):
        raise SystemExit("cells: shape or non-positive values")
    rel = np.abs(pa - pb) / pb
    rel_a = np.abs(pa - pb) / pa
    i = np.unravel_index(np.argmax(rel), rel.shape)
    be, ce, ae = cb["e_band_edges_GeV"], cb["cosz_edges"], cb["azimuth_edges_deg"]
    where = (f"{SPECIES[i[0]]}, {be[i[1]]:.3g}-{be[i[1] + 1]:.3g} GeV, cos Z "
             f"{ce[i[2]]:.1f}-{ce[i[2] + 1]:.1f}, azimuth {ae[i[3]]:.0f}-"
             f"{ae[i[3] + 1]:.0f} deg")
    ok = bool(np.all(rel <= TOL))
    K = pb / cb["phi_0"][..., None]
    res = {"gate": "G2.4(a)", "verdict": "PASS" if ok else "FAIL",
           "max_rel_to_B": float(rel.max()), "at": where, "index": [int(x) for x in i],
           "max_rel_to_A": float(rel_a.max()),
           "n_above": {t: int(np.sum(rel > float(t))) for t in ("1e-3", "3e-3",
                                                                  "5e-3")},
           "worst_per_band": [float(x) for x in rel.max(axis=(0, 2, 3))],
           "K_B_range": [float(K.min()), float(K.max())],
           "inputs": {p: sha256(p) for p in (a.cells_b, a.cells_a)},
           "script_sha256": sha256(os.path.abspath(__file__)),
           "command": " ".join(sys.argv)}
    with open(js, "w") as fh:
        json.dump(res, fh, indent=1)
    L = ["# Gate G2.4(a): location grid A against grid B (cap of 10 deg, M = 8)", "",
         GATE, "", f"## Result: **{res['verdict']}**", "",
         f"Largest move |Phi_A - Phi_B| / Phi_B over the 4800 cells: "
         f"**{rel.max():.3e}** (limit {TOL}) at {where}. With Phi_A as the "
         f"denominator: {rel_a.max():.3e}. Cells above 0.1%, 0.3%, 0.5%: "
         f"{res['n_above']['1e-3']}, {res['n_above']['3e-3']}, "
         f"{res['n_above']['5e-3']}.", "",
         "Worst per band (0.1 to 10 GeV, ascending): " + ", ".join(
             f"{x:.1e}" for x in res["worst_per_band"]) + ".", "",
         f"Cutoff factor K = Phi_cut / Phi_0 on grid B: {K.min():.3f} to "
         f"{K.max():.3f}.", "", "Inputs (sha256):", ""]
    L += [f"- `{p}` {h}" for p, h in res["inputs"].items()]
    L += ["", f"Script sha256: {res['script_sha256']}", "", "Command:", "",
          f"    {res['command']}", ""]
    with open(md, "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
