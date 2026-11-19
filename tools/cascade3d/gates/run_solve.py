"""One cascade3d solve with the default (or a modified) geometry; prints cost.

Kernel set: MCEq's (``pop``, ``loss``; ``--angular`` attaches the synthetic law) or a
file (``--ks path``; then ``pop`` and ``loss`` are labels only); ``--straight`` drops
every angular law; ``--r-earth`` in km (MCEq 6391, the Monte Carlo 6371).

Usage::

    taskset -c 0-3 $PY tools/cascade3d/gates/run_solve.py PHYS losson [--angular]
        [--levels-refine 2] [--family-refine 2] [--n-sub 8] [--L 32] [--gate-lines]
        [--out file.npz]

Writes the ground fluxes of the G1.1 quantities for every ground line, the sweep
residuals and the measured cost (wall, CPU = user+sys of the process, peak RSS).
"""

import argparse
import json
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import assemble as A  # noqa: E402
import grid  # noqa: E402
import kernelset as K  # noqa: E402
import rays  # noqa: E402
import solver as S  # noqa: E402

DEFAULTS = {"n_ground": 48, "n_sub": 4, "dX_max": 5.0, "L": 32, "n_sweeps": 6,
            "levels_refine": 1, "family_refine": 1}


def straight_ahead(ks):
    """The same set with every angular law removed (g = 1)."""
    d = dict(ks.d)
    d["ang_channel"] = np.zeros(0, dtype=np.int64)
    d["n_ang"] = np.array(0)
    d["g"] = np.zeros((0, 0, 0))
    d["a"] = np.zeros((0, 1, 0, 0))
    return K.KernelSet(d)


def build(args):
    r_e = args.r_earth * 1e5
    atm = grid.CorsikaAtmosphere(r_e=r_e)
    spec = parse_spec(args.levels_spec)
    lev = grid.make_levels(atm, spec=spec, refine=args.levels_refine,
                           ground_layer=args.ground_layer)
    if args.gate_lines:
        fam = grid.LineFamily(np.array(grid.G11_COSZ), np.zeros(0), r_e=r_e)
    else:
        base = grid.make_levels(atm, spec=spec, ground_layer=args.ground_layer)
        fam = grid.make_line_family(base, n_ground=args.n_ground,
                                    refine=args.family_refine, r_e=r_e,
                                    **({} if args.ground_mode is None
                                       else {"mode": args.ground_mode}))
    r = rays.Rays(atm, lev, fam, n_sub=args.n_sub, dX_max=args.dX_max)
    if args.ks:
        ks = K.KernelSet.load(args.ks)
    else:
        ks = K.from_mceq(args.pop, loss=args.loss == "losson",
                         keep_upscatter=args.keep_upscatter)
        if args.angular:
            ks = K.synthetic_angular(ks, L=args.L)
    if args.straight:
        ks = straight_ahead(ks)
    if args.e_top is not None:
        d = dict(ks.d)
        d["phi_top"] = np.where(ks.e_grid[None, :] <= args.e_top, ks.phi_top, 0.0)
        ks = K.KernelSet(d)
    return r, ks


def parse_spec(text):
    """"1:0.05,3:0.1,50:0.4,70:1,inf:3" -> ((1, 0.05), ...) (km); None = default."""
    if not text:
        return None
    out = []
    for item in text.split(","):
        h, dh = item.split(":")
        out.append((float(h), float(dh)))
    return tuple(out)


def cpu():
    t = os.times()
    return t.user + t.system


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pop")
    ap.add_argument("loss")
    ap.add_argument("--angular", action="store_true")
    ap.add_argument("--gate-lines", action="store_true")
    ap.add_argument("--n-ground", type=int, default=DEFAULTS["n_ground"])
    ap.add_argument("--n-sub", type=int, default=DEFAULTS["n_sub"])
    ap.add_argument("--dX-max", type=float, default=DEFAULTS["dX_max"])
    ap.add_argument("--L", type=int, default=DEFAULTS["L"])
    ap.add_argument("--n-sweeps", type=int, default=DEFAULTS["n_sweeps"])
    ap.add_argument("--levels-refine", type=int, default=1)
    ap.add_argument("--levels-spec", default="")
    ap.add_argument("--ground-layer", action="store_true",
                    help="levels 10, 30, 100, 300 m above the ground")
    ap.add_argument("--ground-mode", default=None, choices=("mixed", "theta"),
                    help="ground-line layout (default: grid.make_line_family's)")
    ap.add_argument("--proj", default="hat", choices=("hat", "trapezoid"))
    ap.add_argument("--ks", default="", help="kernel-set npz (overrides pop/loss)")
    ap.add_argument("--straight", action="store_true",
                    help="drop every angular law (straight-ahead limit)")
    ap.add_argument("--r-earth", type=float, default=6391.0, help="Earth radius [km]")
    ap.add_argument("--keep-upscatter", action="store_true",
                    help="keep MCEq's up-scatter into transported species (passes)")
    ap.add_argument("--e-top", type=float, default=None,
                    help="zero the top intensity above this kinetic energy [GeV]")
    ap.add_argument("--family-refine", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--warm", action="store_true",
                    help="also time a second (compiled) solve in the same process")
    args = ap.parse_args(argv)
    w0, c0 = time.time(), cpu()
    r, ks = build(args)
    w1, c1 = time.time(), cpu()
    sv = S.Solver(r, ks, n_sweeps=args.n_sweeps, projection=args.proj)
    out = sv.solve(verbose=True)
    out["res"].block_until_ready()
    w2, c2 = time.time(), cpu()
    gf = A.ground_fluxes(sv, out)
    cells = A.cell_averages(sv, out)
    w3, c3 = time.time(), cpu()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 ** 2
    warm = {}
    if args.warm:
        # second solve in the same process (compiled), solve + assembly
        w4, c4 = time.time(), cpu()
        out2 = sv.solve()
        out2["res"].block_until_ready()
        A.ground_fluxes(sv, out2)
        A.cell_averages(sv, out2)
        w5, c5 = time.time(), cpu()
        warm = {"warm_wall_s": w5 - w4, "warm_cpu_s": c5 - c4,
                "peak_rss_GB_after_warm":
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 ** 2}
    cost = {"setup_wall_s": w1 - w0, "setup_cpu_s": c1 - c0,
            "solve_wall_s": w2 - w1, "solve_cpu_s": c2 - c1,
            "assemble_wall_s": w3 - w2, "assemble_cpu_s": c3 - c2,
            "total_wall_s": w3 - w0, "total_cpu_s": c3 - c0,
            "peak_rss_GB": rss, "affinity": sorted(os.sched_getaffinity(0)),
            "W": sv.W, "res_max": float(np.max(out["res"])), "k_top": sv.k_top,
            "k_up": sv.k_up, "up_res": [float(x) for x in np.asarray(out["up_res"])]}
    cost.update(warm)
    cost.update(r.summary())
    cost.update(vars(args))
    print(json.dumps(cost, indent=1), flush=True)
    if args.out:
        np.savez(args.out, e_grid=ks.e_grid, res=np.asarray(out["res"]),
                 up_res=np.asarray(out["up_res"]),
                 cost_json=np.array(json.dumps(cost)),
                 cell_band_edges=cells["band_edges"],
                 cell_cosz_edges=cells["cosz_edges"],
                 **{f"cell_{q}": cells[q] for q in A.G13_SPECIES},
                 **{q: gf[q] for q in list(A.QUANTITIES) + ["mu0"]})
    return gf, cost


if __name__ == "__main__":
    main()
