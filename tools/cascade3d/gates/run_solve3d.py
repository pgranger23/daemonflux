"""Location solve of the mode path (cascade3d C2): tangents of the azimuthal orders.

Geometry, kernel set and settings: those of the frozen C1 solve
(``g2_7.build_geometry``);
guard: the setting of G2.7 (``solver3d.GUARD_PROTOTYPE``).  The boundary is the deficit
of one location's sky table (h_ref = 0) on the chains and the 36 sectors; the orders
are ``assemble3d.order_tangents``.

The boundary is one location (``--xy``) or the cap average over a grid
(``--cap 10 --spacing 5``; ``assemble3d.grid_deficit``).

Steps::

    taskset -c CORES $PY run_solve3d.py run OUTDIR --xy 0,0 --M 12 \
        --orders 0c,1c,1s --batch 3 --tag A [--cache DIR]
    $PY run_solve3d.py combine OUTDIR --tags A,B,... --M 12 --out site_M12.npz

``run`` writes ``OUTDIR/tangents_<tag>.npz`` (ground tables of every tangent, the
primal, residuals, timing and cost) and refuses to overwrite.  ``combine`` writes the
cells: ``phi_cut`` (species, bands, rows, 12 azimuth bins) with all orders up to M,
``phi_cut_M<m>`` for the partial sums asked with ``--partial``, ``phi_0`` (species,
bands, rows) and the axes.
"""

import argparse
import json
import os
import resource
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

N3 = os.environ.get("N3", "/home/pigrange/mc3d_work/runs/native3d")
SKY_DIR = os.path.join(N3, "c2", "solver", "sky")


def cpu():
    t = os.times()
    return t.user + t.system


def parse_orders(text, M):
    if text == "all":
        return None
    out = []
    for it in text.split(","):
        out.append((int(it[:-1]), it[-1]))
    for m, _p in out:
        if m > M:
            raise SystemExit("order above M")
    return out


def cmd_run(a):
    import assemble3d as A3
    import boundary as B
    import g2_7
    import solver3d as S3

    if a.cache:
        S3.enable_compilation_cache(a.cache)
    path = os.path.join(a.outdir, f"tangents_{a.tag}.npz")
    if os.path.exists(path):
        raise SystemExit(f"{path} exists")
    w0, c0 = time.time(), cpu()
    ry, ks, sha = g2_7.build_geometry()
    grid = B.LocationGrid(a.spacing, 10.0)
    if a.cap > 0:
        weights = grid.cap_weights(a.cap)
        where = {"cap_deg": a.cap, "spacing_deg": a.spacing}
    else:
        x, y = (float(v) for v in a.xy.split(","))
        weights = np.zeros(grid.n)
        weights[grid.node_index(x, y)] = 1.0
        where = {"xy": [x, y], "spacing_deg": a.spacing}
    gw = B.GroupWeights(ks.e_bins, model="GSFG:GH")
    nodes = B.line_nodes()
    cw = B.chain_weights(ry, nodes)
    Dc, used = A3.grid_deficit(SKY_DIR, grid, weights, gw, ks, cw)
    for _p, _w, tp in used:
        t = B.SkyTable.load(tp)
        if not all(np.array_equal(t.nodes[k], nodes[k]) for k in nodes):
            raise SystemExit(f"{tp}: line nodes differ from boundary.line_nodes()")
    allt = A3.order_tangents(Dc, a.M, ks)
    want = parse_orders(a.orders, a.M)
    tl = allt if want is None else [t for t in allt if (t[0], t[1]) in want]
    if want is not None and len(tl) != len(want):
        raise SystemExit("an asked order has no tangent")
    ts = S3.TangentSolver(ry, ks, guard=S3.GUARD_PROTOTYPE)
    w1, c1 = time.time(), cpu()
    r = A3.run_tangents(ts, tl, a.batch, log=lambda s: print(s, flush=True))
    w2, c2 = time.time(), cpu()
    cost = {"setup_cpu_s": c1 - c0, "solve_cpu_s": c2 - c1, "total_cpu_s": c2 - c0,
            "total_wall_s": w2 - w0,
            "peak_rss_GB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024.0 ** 2, "affinity": sorted(os.sched_getaffinity(0)),
            "batch": a.batch, "n_tangents": len(tl), "timing": r["timing"],
            "boundary": where,
            "sky": [(p, w, tp, B._sha256(tp)) for p, w, tp in used],
            "ks_sha256": sha,
            "guard": S3.GUARD_PROTOTYPE.as_dict(), "M": a.M,
            "res_max": float(r["res"].max()),
            "d_res_max": float(r["d_res"].max()),
            "d_up_res": r["d_up_res"].tolist()}
    print(json.dumps({k: v for k, v in cost.items() if k != "timing"}, indent=1),
          flush=True)
    np.savez(path, mu0=r["mu0"], I=r["I"], F_ground=r["F_ground"], res=r["res"],
             up_res=r["up_res"], dI=r["dI"], dF_ground=r["dF_ground"],
             d_res=r["d_res"], d_up_res=r["d_up_res"],
             orders=np.array([t[0] for t in r["tangents"]]),
             parts=np.array([t[1] for t in r["tangents"]]),
             cost_json=json.dumps(cost), ground_mu0=ry.ground_mu0)


def load_tangents(outdir, tags):
    import kernelset as K

    import g2_7
    ks = K.KernelSet.load(g2_7.KS_PATH)
    tl, dI, dres, dF = [], [], [], []
    I0 = mu0 = None
    for t in tags:
        f = np.load(os.path.join(outdir, f"tangents_{t}.npz"))
        if I0 is None:
            I0, mu0, F0, res0 = f["I"], f["mu0"], f["F_ground"], f["res"]
        else:
            assert np.array_equal(f["I"], I0), "primal differs between runs"
        tl += [(int(m), str(p)) for m, p in zip(f["orders"], f["parts"])]
        dI.append(f["dI"])
        dres.append(f["d_res"])
        dF.append(f["dF_ground"])
    if len(set(tl)) != len(tl):
        raise SystemExit("an order is in two files")
    return {"ks": ks, "mu0": mu0, "I0": I0, "F0": F0, "res": res0, "tangents": tl,
            "dI": np.concatenate(dI), "d_res": np.concatenate(dres),
            "dF": np.concatenate(dF)}


def cmd_combine(a):
    import assemble as A
    import assemble3d as A3

    out = os.path.join(a.outdir, a.out)
    if os.path.exists(out):
        raise SystemExit(f"{out} exists")
    d = load_tangents(a.outdir, a.tags.split(","))
    need = {(m, p) for m in range(a.M + 1) for p in "cs" if not (p == "s" and m == 0)}
    if not need <= set(d["tangents"]):
        raise SystemExit(f"missing orders: {sorted(need - set(d['tangents']))}")
    ks = d["ks"]
    cells = {}
    for M in [a.M] + [int(v) for v in a.partial.split(",") if v]:
        Iaz = A3.azimuth_table(d["I0"], d["dI"], d["tangents"], M=M)
        cells[f"phi_cut_M{M}"] = A3.azimuth_cells(d["mu0"], Iaz, ks)
    np.savez(out, phi_cut=cells[f"phi_cut_M{a.M}"],
             phi_0=A3.phi0_cells(d["mu0"], d["I0"], ks),
             e_band_edges_GeV=np.asarray(A.G13_BAND_EDGES),
             cosz_edges=np.asarray(A.G13_COSZ_EDGES),
             azimuth_edges_deg=np.asarray(A3.AZ_EDGES),
             species=np.array(A.G13_SPECIES), tags=a.tags, **cells)
    print(f"wrote {out}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "combine"))
    ap.add_argument("outdir")
    ap.add_argument("--xy", default="0,0")
    ap.add_argument("--cap", type=float, default=0.0,
                    help="cap radius [deg]: cap-average boundary (0: --xy)")
    ap.add_argument("--spacing", type=float, default=5.0)
    ap.add_argument("--M", type=int, default=12)
    ap.add_argument("--orders", default="all")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--tag", default="A")
    ap.add_argument("--tags", default="")
    ap.add_argument("--partial", default="6")
    ap.add_argument("--out", default="cells.npz")
    ap.add_argument("--cache", default="")
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)
    return {"run": cmd_run, "combine": cmd_combine}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main() or 0)
