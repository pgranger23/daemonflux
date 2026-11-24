"""Gate G2.5 (numerical convergence with the azimuth); writes G2.5_<part>.md / .json.

Gate text: ``GATES.md`` section C2 (G2.5) and the entry "G2.5(a), how 'doubling the
phi nodes (or azimuthal modes)' is read" (both quoted in the result file).

Part ``orders`` (this script's first part; the mu and L doublings and part (b) follow
in later runs and get their own result files):

- Location: the grid location nearest the site (the site itself), boundary = its own
  sky table, mode path of ``run_solve3d.py`` at M = 12 (all 25 tangents).
- (a), azimuthal orders: Phi_cut at M = 6 is the partial sum of the M = 12 run
  (orders 0..6), ``assemble3d.azimuth_table(..., M=6)``; cells of G2.3 at that
  location: four neutrino species x ten bands (0.1 to 10 GeV) x ten cos Z rows x
  twelve azimuth bins of 30 deg (4800 cells), ``assemble3d.azimuth_cells``.  Pass if
  max |Phi(M=6) - Phi(M=12)| / Phi(M=12) <= 0.005 over the 4800 cells.  Cells with
  Phi(M=12) <= 0 are listed and counted as failing.
- (c): the same-bin sweep residual, per energy bin, the largest over the primal march
  and the 25 tangent marches (the tangent residual is the change of the last sweep of
  the tangent relative to the tangent's own largest value, ``solver3d``).  Pass if it
  is below 1e-4 in every energy bin.
- Reported: the same with the order 0 alone against M = 12; per-band worst of (a);
  and a check of the path against G2.7: the cells of the order-0 tangent of this run
  against ``gates/G2.7/cells_s0.1_i0.1_l0.1_mx_D2.npz`` (the same boundary, the
  azimuth average of the site's deficit, up to rounding of the clipping order).

With ``--M 16`` the same evaluation tests M = 8 against M = 16 (``GATES.md``, entry of
2026-11-24, point 1, quoted in the result file): the tangents of the orders 0 to 16 are
read from ``--solve-dir`` (those of the run of 2026-11-23 and the eight of the orders 13
to 16), and the files are named ``G2.5_orders_M8.*``.  With ``--M 12`` the names and the
numbers are those of 2026-11-23.

Usage::

    $PY tools/cascade3d/gates/g2_5.py orders --solve-dir DIR --tags b1,b2,b3,b4,b5 \
        --outdir OUT [--M 16] [--g27 CELLS.npz]

Exit status 0 = (a)-orders and (c) pass, 1 = a FAIL, 2 = input error.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

N3 = os.environ.get("N3", "/home/pigrange/mc3d_work/runs/native3d")
TOL_A, TOL_C = 0.005, 1e-4
SPECIES = ("nue", "antinue", "numu", "antinumu")

GATE = """\
**G2.5 Numerical convergence with the azimuth (gated; an addition to the plan).** At
one location, the grid location nearest the site, with the cutoff, on the ground flux
of that location in the cells of G2.3:
- (a) doubling, one at a time, the φ nodes (or azimuthal modes), the μ grid and the
  Legendre order L moves no cell by more than 0.5%;
- (b) the cutoff sky rotated by 30° about the vertical gives the flux moved by one
  azimuth bin, every cell within 0.5%;
- (c) the same-bin sweep residual is below 1e-4 in every energy bin.

**G2.5(a), how "doubling the φ nodes (or azimuthal modes)" is read.** M against 2M,
from the partial sums of one run at 2M. In the design the orders are independent. The
agent shows this once on a reduced geometry: a run at M equals the partial sum of a run
at 2M to 1e-8. The μ grid and L are doubled in separate runs. The first solve (the
site, M = 12) gives the test for M = 6. If M = 6 fails it, M = 12 for all solves takes
C2 to or beyond the plan's compute and goes to the PI with G2.6."""

GATE_1124 = """\
**Entry of 2026-11-24, 1. Azimuth order: M = 8 for every solve of C2.**
- G2.5(a) on the orders stays FAILED at M = 6 (0.507% against 0.5%). M = 6 is not used.
- The test of M = 8 is the gate's, as read on 2026-11-23: M = 8 against M = 16 at the
  site, from the partial sums of one run at 16. That run is the site solve of
  2026-11-23 (orders 0 to 12, kept as it is) plus the eight tangents of the orders 13
  to 16: same solver code (unchanged since commit d7ede51), same settings, same sky
  table (sha256 beginning `0a382659062ef54d`).
- Cells and evaluation as on 2026-11-23: `gates/g2_5.py orders --M 16`. The script is
  committed with this entry. With `--M 12` it gives the numbers of the result file of
  2026-11-23 again.
- Pass: no cell of the 4800 moves by more than 0.5% between M = 8 and M = 16.
- If it fails, it is reported as failed, and no cap solve is run before the PI has
  decided."""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def worst(rel, be, ce, ae):
    i = np.unravel_index(np.nanargmax(rel), rel.shape)
    s, b, c, a = (int(x) for x in i)
    return {"value": float(rel[i]), "species": SPECIES[s],
            "band_GeV": [float(be[b]), float(be[b + 1])],
            "cosz": [float(ce[c]), float(ce[c + 1])],
            "azimuth_deg": [float(ae[a]), float(ae[a + 1])], "index": [s, b, c, a]}


def cmd_orders(a):
    import assemble as A
    import assemble3d as A3
    import run_solve3d as RS

    stem = "G2.5_orders" if a.M == 12 else f"G2.5_orders_M{a.M // 2}"
    gate = GATE if a.M == 12 else GATE + "\n\n" + GATE_1124
    md = os.path.join(a.outdir, stem + ".md")
    js = os.path.join(a.outdir, stem + ".json")
    npz = os.path.join(a.outdir, stem + "_cells.npz")
    for p in (md, js, npz):
        if os.path.exists(p):
            raise SystemExit(f"{p} exists; result files are never overwritten")
    tags = a.tags.split(",")
    files = {t: os.path.join(a.solve_dir, f"tangents_{t}.npz") for t in tags}
    d = RS.load_tangents(a.solve_dir, tags)
    need = {(m, p) for m in range(a.M + 1) for p in "cs" if not (p == "s" and m == 0)}
    if set(d["tangents"]) != need:
        raise SystemExit(f"orders {sorted(set(d['tangents']) ^ need)} missing or "
                         "extra")
    ks = d["ks"]
    be, ce = np.asarray(A.G13_BAND_EDGES), np.asarray(A.G13_COSZ_EDGES)
    ae = np.asarray(A3.AZ_EDGES)
    cells = {}
    for M in (a.M, a.M // 2, 0):
        Iaz = A3.azimuth_table(d["I0"], d["dI"], d["tangents"], M=M)
        cells[M] = A3.azimuth_cells(d["mu0"], Iaz, ks)
    ref = cells[a.M]
    bad = ref <= 0
    rel = np.where(bad, np.inf, np.abs(cells[a.M // 2] - ref) / np.where(bad, 1, ref))
    rel0 = np.where(bad, np.inf, np.abs(cells[0] - ref) / np.where(bad, 1, ref))
    wa = worst(rel, be, ce, ae)
    pass_a = bool(np.all(rel <= TOL_A))
    # (c) residuals per energy bin
    res_p = np.asarray(d["res"])
    res_t = np.asarray(d["d_res"]).max(axis=0)
    res = np.maximum(res_p, res_t)
    pass_c = bool(np.all(res < TOL_C))
    out = {"part": "orders", "M": a.M, "M_half": a.M // 2,
           "a_orders": {"max_rel": wa, "pass": pass_a, "n_cells": int(rel.size),
                        "n_nonpositive": int(bad.sum()),
                        "n_above_1e-3": int(np.sum(rel > 1e-3)),
                        "worst_per_band": [float(x) for x in rel.max(axis=(0, 2, 3))],
                        f"order0_vs_M{a.M}": worst(rel0, be, ce, ae)},
           "c": {"max_per_bin": res.tolist(), "max": float(res.max()),
                 "argmax_bin": int(np.argmax(res)),
                 "primal_max": float(res_p.max()), "tangent_max": float(res_t.max()),
                 "pass": pass_c},
           "inputs": {t: sha256(f) for t, f in files.items()},
           "script_sha256": sha256(os.path.abspath(__file__)),
           "command": " ".join(sys.argv)}
    if a.g27:
        g = np.load(a.g27)
        i0 = d["tangents"].index((0, "c"))
        lin0 = A3.phi0_cells(d["mu0"], d["I0"] + d["dI"][i0], ks)
        c1 = A3.phi0_cells(d["mu0"], d["I0"], ks)
        out["check_g27"] = {
            "file": a.g27, "sha256": sha256(a.g27),
            "order0_lin_vs_g27_lin_max_rel": float(np.max(np.abs(lin0 / g["lin"] - 1))),
            "c1_vs_g27_c1_max_rel": float(np.max(np.abs(c1 / g["c1"] - 1)))}
    np.savez(npz, phi_0=A3.phi0_cells(d["mu0"], d["I0"], ks), e_band_edges_GeV=be,
             cosz_edges=ce, azimuth_edges_deg=ae, species=np.array(SPECIES),
             **{f"phi_M{M}": c for M, c in cells.items()})
    with open(js, "w") as fh:
        json.dump(out, fh, indent=1)
    ok = pass_a and pass_c
    L = ["# Gate G2.5, azimuthal orders and sweep residual (part of the gate)", "",
         gate, "",
         "This file covers (a) for the azimuthal orders and (c). The mu grid and L "
         "doublings of (a) and part (b) are not in it; the gate is complete only "
         "with them.", "",
         f"## (a), orders: **{'PASS' if pass_a else 'FAIL'}**", "",
         f"M = {a.M // 2} (partial sum) against M = {a.M}, 4800 cells of G2.3 at the "
         f"site: largest relative move **{wa['value']:.3e}** (limit {TOL_A}) at "
         f"{wa['species']}, {wa['band_GeV'][0]:.3g}-{wa['band_GeV'][1]:.3g} GeV, "
         f"cos Z {wa['cosz'][0]:.1f}-{wa['cosz'][1]:.1f}, azimuth "
         f"{wa['azimuth_deg'][0]:.0f}-{wa['azimuth_deg'][1]:.0f} deg. Cells above "
         f"1e-3: {out['a_orders']['n_above_1e-3']}; non-positive cells: "
         f"{int(bad.sum())}.", "",
         "Worst per band (0.1 to 10 GeV, ascending): " + ", ".join(
             f"{x:.1e}" for x in out["a_orders"]["worst_per_band"]) + ".", "",
         f"Reported: order 0 alone against M = {a.M}: largest move "
         f"{out['a_orders'][f'order0_vs_M{a.M}']['value']:.3e}.", "",
         f"## (c): **{'PASS' if pass_c else 'FAIL'}**", "",
         f"Largest same-bin sweep residual over the energy bins: {res.max():.2e} "
         f"(bin {int(np.argmax(res))}); primal {res_p.max():.2e}, tangents "
         f"{res_t.max():.2e} (limit {TOL_C} in every bin).", ""]
    if "check_g27" in out:
        c = out["check_g27"]
        L += ["## Check of the path against G2.7 (reported)", "",
              f"Order-0 tangent of this run against deficit 2 of G2.7 (`{c['file']}`): "
              "linearised cells differ by at most "
              f"{c['order0_lin_vs_g27_lin_max_rel']:.2e}; the C1 cells by "
              f"{c['c1_vs_g27_c1_max_rel']:.2e}.", ""]
    L += [f"Result of this file: (a)-orders {'PASS' if pass_a else 'FAIL'}, (c) "
          f"{'PASS' if pass_c else 'FAIL'}.", "", "Inputs (sha256):", ""]
    L += [f"- `{files[t]}` {h}" for t, h in out["inputs"].items()]
    L += ["", f"Script sha256: {out['script_sha256']}", "", "Command:", "",
          f"    {out['command']}", ""]
    with open(md, "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("orders",))
    ap.add_argument("--solve-dir", required=True)
    ap.add_argument("--tags", default="b1,b2,b3,b4,b5")
    ap.add_argument("--M", type=int, default=12)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--g27", default="")
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)
    return cmd_orders(a)


if __name__ == "__main__":
    sys.exit(main())
