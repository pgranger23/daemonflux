"""G2.8 (mode path in the straight-ahead limit; reported, with a trigger); writes
G2.8.md / .json.

Gate text: ``GATES.md``, "C2, dated entries", 2026-11-23 (quoted in the result file).
How it is evaluated:

- Geometry, kernel set and boundary: those of G2.2 (``gates/g2_2.py``): the six G1.1
  lines (cos Z 1, 0.75, 0.5, 0.25, 0.15, 0.05; straight ahead the lines do not
  couple), the straight-ahead kernel set of G1.1 with loss on, R_E 6391 km, the
  site's sky table, H3a group weights, the binned transmission of each (line,
  sector) as in G2.2 (``group_transmission`` of the line node and sector j).
- Path A: for each of the 36 sectors one nonlinear solve with a top per chain
  (``solver3d.ModeSolver``): the six lines of that sector.
- Path B: the mode path, M = 18.  D_j = B0 (1 - tau_j) on the chains and the 36
  sectors, its orders 0..18 (``boundary.fourier_sectors``; at order 18 only the sine
  part is non-zero at the sector centres), 36 tangents of the guarded tangent solver
  (``solver3d.TangentSolver``, guard of G2.7) at the C1 state, the neutrino
  intensities by ``solver3d.sink_tangent``, then the sector values: the base plus the
  sum over the orders with the means of cos(m phi), sin(m phi) over each 10-deg sector
  (``boundary.bin_mean_basis``).  This synthesis holds the 36 sector values of the
  boundary exactly.  Reported next to it: the point values of the series at the
  sector centres (cos(m phi_j), sin(m phi_j)), which do not hold them exactly (the
  orders carry the sinc correction of the sector average).
- Trigger: |B / A - 1| > 1% for any of the four neutrino species, any line, any
  sector, any MCEq bin from 0.1 to 100 GeV.  Ground muons (mu+ + mu-) reported.

Usage::

    taskset -c 16-19 $PY tools/cascade3d/gates/g2_8.py --outdir OUT [--probe]

``--probe``: one zenith line (cos Z 0.5), both paths, timed, no result file.
Exit status 0: no trigger; 1: trigger; 2: input error.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

import g2_2 as G22  # noqa: E402

N3 = G22.N3
TRIG = 0.01
M = 18

GATE = """\
**G2.8 Mode path in the straight-ahead limit (reported, with a trigger; an addition).**
The orders m ≥ 1 have no nonlinear counterpart on the same code path. This check tests
their bookkeeping (frames, signs, phases, normalisation of the deficit). At the site,
the six zenith lines of G2.2 × 36 azimuth sectors of 10°, straight-ahead limit:
- path A: the nonlinear solve of each direction with that direction's own boundary (the
  path of G2.2);
- path B: the mode path with M = 18, which holds 36 sector values exactly, synthesised
  at the sector centres.

A difference above 1% for any of the four neutrino species, in any MCEq bin from 0.1 to
100 GeV, triggers a check of the mode path before the freeze. The check does not test
the kernel moments of order m ≥ 1: in the straight-ahead limit every order has the same
transport. If the mode path cannot be run in the straight-ahead limit at reasonable
cost, the agent says so and the check is reported as not done."""


def quantities(ks, Fg, nu_ground, solver):
    """Ground fluxes (n_lines, nE) of the neutrinos and mu+ + mu- from the ground
    sink table ``nu_ground`` (nE, nSk, n_lines) and F_ground (nE, nT, n_lines)."""
    import assemble as A

    names = list(ks.species)
    out = {}
    for q, sps in A.QUANTITIES.items():
        v = 0.0
        for s in sps:
            if s not in names:
                continue
            i = names.index(s)
            if i in solver.Sk:
                v = v + nu_ground[:, solver.Sk.index(i), :].T
            else:
                v = v + Fg[:, solver.T.index(i), :].T
        out[q] = v
    return out


def main(argv=None):
    import assemble3d as A3
    import boundary as B
    import solver3d as S3

    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(N3, "c2", "solver", "gates",
                                                     "G2.8"))
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--name", default="G2.8")
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)
    md, js = (os.path.join(a.outdir, a.name + s) for s in (".md", ".json"))
    if not a.probe:
        for p in (md, js):
            if os.path.exists(p):
                raise SystemExit(f"{p} exists; result files are never overwritten")
    w0, c0 = time.time(), G22.cpu()
    tab = B.SkyTable.load(G22.SITE_TABLE)
    ns = int(tab.n_sectors)
    cosz = (0.5,) if a.probe else G22.COSZ
    r, ks = G22.solver_setup(cosz)
    model = G22.h3a()
    cz = [float(np.round(c, 10)) for c in G22.chain_cosz(r)]
    names = list(ks.species)
    ip, iN = names.index("p+"), names.index("n0")
    taus = np.zeros((r.Nl, ns, 2, ks.nE))
    for jl, c in enumerate(cz):
        node = G22.node_of_cosz(tab, c)
        for j in range(ns):
            T = G22.group_transmission(tab, node, j, ks.e_bins, model)
            taus[jl, j] = G22.nucleon_ratio(T, ks.e_grid)
    w1, c1 = time.time(), G22.cpu()
    # ---- path A
    sv = S3.ModeSolver(r, ks, guard=S3.GUARD_NONE)
    A_q = []
    for j in range(ns):
        gf, _res = G22.solve_with_tops(sv, ks, r, [(taus[jl, j, 0], taus[jl, j, 1])
                                                   for jl in range(r.Nl)])
        A_q.append(gf)
    w2, c2 = time.time(), G22.cpu()
    # ---- path B
    b0 = ks.phi_top[[ip, iN]]                                   # (2, nE)
    D = b0.T[None, None] * (1.0 - np.moveaxis(taus, 2, 3))      # (Nl, ns, nE, 2)
    tl = A3.order_tangents(D, M, ks)
    ts = S3.TangentSolver(r, ks, guard=S3.GUARD_PROTOTYPE)
    out = ts.solve(np.stack([t[2] for t in tl]), [t[0] for t in tl])
    gn = r.ground_node
    if r.Nl != r.n_ground or not np.allclose(np.sort(cz), cz):
        raise SystemExit("expected only ground lines, in ascending cos Z")
    I, dI = S3.sink_tangent(ts.sv, out["emis"], out["d_emis"])
    I, dI = np.asarray(I)[:, :, gn], np.asarray(dI)[:, :, :, gn]
    Fg, dFg = np.asarray(out["F_ground"]), np.asarray(out["d_F_ground"])
    w3, c3 = time.time(), G22.cpu()
    sec_edges = np.linspace(0.0, 360.0, ns + 1)
    Cb, Sb = B.bin_mean_basis(M, sec_edges)
    ph = np.radians(B.sector_centres(ns))
    Cp = np.cos(np.arange(M + 1)[None] * ph[:, None])
    Sp = np.sin(np.arange(M + 1)[None] * ph[:, None])
    E = ks.e_grid
    sel = np.flatnonzero((E >= 0.1 - 1e-9) & (E <= 100.0 + 1e-6))
    worst = {"means": {}, "points": {}}
    per_line = {}
    for mode, (Cm, Sm) in (("means", (Cb, Sb)), ("points", (Cp, Sp))):
        for j in range(ns):
            Ij = I.copy()
            Fj = Fg.copy()
            for (m, part, _dt), d1, f1 in zip(tl, dI, dFg):
                wgt = Cm[j, m] if part == "c" else Sm[j, m]
                Ij = Ij + wgt * d1
                Fj = Fj + wgt * f1
            Bq = quantities(ks, Fj, Ij, ts.sv)
            for q in G22.NU + ("mu",):
                for jl, c in enumerate(cz):
                    # chains = ground lines in ascending mu0 (no tangent lines)
                    av = A_q[j][q][jl][sel]
                    bv = Bq[q][jl][sel]
                    rel = bv / av - 1
                    k = int(np.argmax(np.abs(rel)))
                    key = "mu" if q == "mu" else "nu"
                    cur = worst[mode].get(key, (0.0, None))
                    if abs(rel[k]) > abs(cur[0]):
                        worst[mode][key] = (float(rel[k]), (q, c, j, float(E[sel][k])))
                    if mode == "means":
                        pl = per_line.setdefault(f"{c:g}", {})
                        pl[q] = max(pl.get(q, 0.0), float(np.abs(rel).max()))
    w4, c4 = time.time(), G22.cpu()
    cost = {"setup_cpu_s": c1 - c0, "pathA_cpu_s": c2 - c1, "pathB_cpu_s": c3 - c2,
            "assembly_cpu_s": c4 - c3, "total_cpu_s": c4 - c0, "wall_s": w4 - w0,
            "n_tangents": len(tl), "affinity": sorted(os.sched_getaffinity(0)),
            "pathB_tangent_res_max": float(np.max(np.asarray(out["d_res"])))}
    trig = abs(worst["means"]["nu"][0]) > TRIG
    if a.probe:
        print(json.dumps({"worst": worst, "cost": cost, "per_line": per_line},
                         indent=1))
        return 0
    res = {"check": "G2.8", "outcome": "TRIGGER" if trig else "no trigger",
           "worst": worst, "per_line_max_abs_rel_means": per_line, "cost": cost,
           "sky_table": G22.SITE_TABLE, "sky_sha256": G22.sha256(G22.SITE_TABLE),
           "command": " ".join(sys.argv), "script_sha256": G22.sha256(__file__)}
    with open(js, "w") as fh:
        json.dump(res, fh, indent=1)
    wn, wm = worst["means"]["nu"], worst["means"]["mu"]
    pn = worst["points"]["nu"]
    L = ["# G2.8: mode path in the straight-ahead limit", "", GATE, "",
         f"## Outcome: **{res['outcome']}**", "",
         f"Path B (M = 18, sector means) against path A, 6 lines x 36 sectors, four "
         f"neutrino species, MCEq bins 0.1 to 100 GeV: largest |B/A - 1| = "
         f"**{abs(wn[0]):.3e}** ({wn[1][0]}, cos Z {wn[1][1]:g}, sector {wn[1][2]}, "
         f"E = {wn[1][3]:.3g} GeV); trigger above {TRIG:.0%}.", "",
         f"Ground muons (reported): {wm[0]:+.3e} ({wm[1][1]:g}, sector {wm[1][2]}, "
         f"E = {wm[1][3]:.3g} GeV).", "",
         f"Reported: point values of the series at the sector centres against path "
         f"A: largest {pn[0]:+.3e} ({pn[1][0]}, cos Z {pn[1][1]:g}, sector "
         f"{pn[1][2]}, E = {pn[1][3]:.3g} GeV).", "",
         "Largest |B/A - 1| per line and species (sector means):", "",
         "| cos Z | " + " | ".join(G22.NU) + " | mu |", "|---" * 6 + "|"]
    for c, pl in per_line.items():
        L.append(f"| {c} | " + " | ".join(f"{pl[q]:.2e}" for q in G22.NU + ("mu",))
                 + " |")
    L += ["", f"Cost: {cost['total_cpu_s'] / 60:.1f} core-min (path A "
          f"{cost['pathA_cpu_s'] / 60:.1f}, path B {cost['pathB_cpu_s'] / 60:.1f} "
          f"with {len(tl)} tangents), wall {cost['wall_s'] / 60:.1f} min, cores "
          f"{cost['affinity']}.", "",
          f"Sky table `{G22.SITE_TABLE}` (sha256 {res['sky_sha256'][:16]}).", "",
          "Command:", "", f"    {res['command']}", "",
          f"Script sha256: {res['script_sha256']}", ""]
    with open(md, "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))
    return 1 if trig else 0


if __name__ == "__main__":
    sys.exit(main())
