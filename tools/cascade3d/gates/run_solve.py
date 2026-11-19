"""One cascade3d solve with the default (or a modified) geometry; prints cost.

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

DEFAULTS = {"n_ground": 48, "n_sub": 4, "dX_max": 10.0, "L": 32, "n_sweeps": 6,
            "levels_refine": 1, "family_refine": 1}


def build(args):
    atm = grid.CorsikaAtmosphere()
    spec = parse_spec(args.levels_spec)
    lev = grid.make_levels(atm, spec=spec, refine=args.levels_refine)
    if args.gate_lines:
        fam = grid.LineFamily(np.array(grid.G11_COSZ), np.zeros(0))
    else:
        base = grid.make_levels(atm, spec=spec)
        fam = grid.make_line_family(base, n_ground=args.n_ground,
                                    refine=args.family_refine)
    r = rays.Rays(atm, lev, fam, n_sub=args.n_sub, dX_max=args.dX_max)
    ks = K.from_mceq(args.pop, loss=args.loss == "losson")
    if args.angular:
        ks = K.synthetic_angular(ks, L=args.L)
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
    ap.add_argument("--family-refine", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    w0, c0 = time.time(), cpu()
    r, ks = build(args)
    w1, c1 = time.time(), cpu()
    sv = S.Solver(r, ks, n_sweeps=args.n_sweeps)
    out = sv.solve(verbose=True)
    out["res"].block_until_ready()
    w2, c2 = time.time(), cpu()
    gf = A.ground_fluxes(sv, out)
    w3, c3 = time.time(), cpu()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 ** 2
    cost = {"setup_wall_s": w1 - w0, "setup_cpu_s": c1 - c0,
            "solve_wall_s": w2 - w1, "solve_cpu_s": c2 - c1,
            "assemble_wall_s": w3 - w2, "assemble_cpu_s": c3 - c2,
            "total_wall_s": w3 - w0, "total_cpu_s": c3 - c0,
            "peak_rss_GB": rss, "affinity": sorted(os.sched_getaffinity(0)),
            "W": sv.W, "res_max": float(np.max(out["res"]))}
    cost.update(r.summary())
    cost.update(vars(args))
    print(json.dumps(cost, indent=1), flush=True)
    if args.out:
        np.savez(args.out, e_grid=ks.e_grid, res=np.asarray(out["res"]),
                 cost_json=np.array(json.dumps(cost)),
                 **{q: gf[q] for q in list(A.QUANTITIES) + ["mu0"]})
    return gf, cost


if __name__ == "__main__":
    main()
