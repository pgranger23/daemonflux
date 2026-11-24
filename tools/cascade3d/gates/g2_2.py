"""Gate G2.2 (straight-ahead limit with the cutoff against MCEq); writes G2.2.md/.json.

Gate text: ``GATES.md`` section C2 and the entry "G2.2, how H3a is read" (quoted in
the result file).  How it is evaluated:

- Directions: Kamioka, cos Z = 1, 0.75, 0.5, 0.25, 0.15, 0.05 x azimuth 0, 90, 180,
  270 deg (21 directions; the vertical once, at azimuth 0).
- The cutoff sky of a line of sight: the site's sky table (h_ref = 0, seed
  20261123), the ground line node of that cos Z (an exact node of the table), the
  sector that contains the azimuth (``boundary.sky_transmission``'s rule: sector
  floor(az / 10 deg)).  Per H3a nucleus group (crflux ``HillasGaisser2012("H3a")``,
  groups 14, 402, 1206, 2814, 5426) the binned transmission of MCEq energy bin k is
  T_g(k) = int_bin J_g(T) t(R_g(T)) dT / int_bin J_g(T) dT, T the kinetic energy per
  nucleon, J_g the group's nucleus flux at total energy A (T + m_N) as MCEq evaluates
  it, R_g = (A/Z) p(T) (``cutoff.rigidity_gv``), t the table's verdict of the
  rigidity half-interval of R_g (1 above 60 GV).
- MCEq: ``kernels_mceq.make_mceq("PHYS", energy_loss=True)`` (database, population,
  atmosphere, H3a and observation level of G1.1 with loss on); for each direction the
  primary model is H3a with every group's nucleus flux multiplied by T_g(k) of the bin
  of its kinetic energy per nucleon (``set_primary_model`` under the population's
  floor), then ``set_theta_deg`` and ``solve``.
- Solver: the straight-ahead kernel set of G1.1 (``kernelset.from_mceq("PHYS",
  loss=True)``) on the six G1.1 lines (straight ahead the lines do not couple, as in
  G1.1), frozen default settings, R_E of MCEq (6391 km), one nonlinear solve per
  azimuth with a top intensity per chain (``solver3d.ModeSolver``):
  ``phi_top_x(k) * tau_x(k)``, tau_x the ratio of MCEq's nucleon spectrum with and
  without the transmission at the same energies (the same input as MCEq's), the
  group weights of H3a on both sides.
- Pass: the four neutrino species within 1% in every MCEq bin from 0.1 to 100 GeV, in
  all 21 directions.  Ground muons (mu+ + mu-) are reported.

Usage::

    taskset -c 16-19 $PY tools/cascade3d/gates/g2_2.py --outdir OUT [--probe]

``--probe``: one direction (cos Z 0.5, azimuth 90) on one zenith line, timed, no result
file (the cost check before the run).  Exit status 0 PASS, 1 FAIL, 2 input error.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

N3 = os.environ.get("N3", "/home/pigrange/mc3d_work/runs/native3d")
SITE_TABLE = os.path.join(N3, "c2", "solver", "sky", "sky_h0_x+00.0_y+00.0.npz")
COSZ = (1.0, 0.75, 0.5, 0.25, 0.15, 0.05)
AZIMUTHS = (0.0, 90.0, 180.0, 270.0)
NU = ("numu", "antinumu", "nue", "antinue")
TOL = 0.01

GATE = """\
**G2.2 Straight-ahead limit with the cutoff against MCEq (gated).** At Kamioka
(36.4267° N, 137.31° E), cos Z = 1, 0.75, 0.5, 0.25, 0.15, 0.05, azimuth 0°, 90°, 180°
and 270° (21 directions; the vertical once). Solver: straight-ahead limit, boundary from
the cutoff sky for that line of sight. MCEq: same zenith, the spectrum of each nucleus
group multiplied by the same binned transmission of that line. Otherwise as G1.1 with
energy loss on (same database, population, atmosphere, H3a, observation level).
Pass: the four neutrino species within 1% in every MCEq bin from 0.1 to 100 GeV.
Ground muons are reported.

**G2.2, how "H3a" is read.** The boundary of this gate uses the group weights of H3a,
as G1.1 did, on both sides. The solver side is the straight-ahead nonlinear path, one
chain per direction, not the mode path."""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def cpu():
    t = os.times()
    return t.user + t.system


# ---------------------------------------------------------------- the boundary


def h3a():
    from mceq3d_primary import primary_class
    return primary_class("HillasGaisser2012")("H3a")


def group_transmission(table, node, sector, e_bins, model, n_gl=8, n_panel=2):
    """Binned transmission T_g(k) of every crflux group of ``model`` for the line
    node ``node`` and sector ``sector`` of ``table``: dict cid -> (nE,)."""
    import boundary as B
    import cutoff
    import kernels_mceq as KM  # noqa: F401  (MCEq on the path)

    m_n = 0.5 * (0.938272 + 0.939565)
    edges = np.asarray(table.edges)
    verdict = np.asarray(table.allowed[node, sector], float)       # (n_rungs,)
    xg, wg = np.polynomial.legendre.leggauss(n_gl)
    out = {}
    for cid in model.nucleus_ids:
        Z, A = model.Z_A(cid)
        tcut = B.t_of_rigidity(edges, A / Z)
        T = np.zeros(len(e_bins) - 1)
        for k in range(T.size):
            lo, hi = e_bins[k], e_bins[k + 1]
            inner = tcut[(tcut > lo) & (tcut < hi)]
            cuts = np.unique(np.concatenate([[lo, hi], inner]))
            num = den = 0.0
            for a, b in zip(cuts[:-1], cuts[1:]):
                ed = np.linspace(np.log(a), np.log(b), n_panel + 1)
                h = np.diff(ed)
                u = ed[:-1, None] + 0.5 * h[:, None] * (xg[None] + 1.0)
                t = np.exp(u).ravel()
                w = (0.5 * h[:, None] * wg[None] * np.exp(u)).ravel()
                J = np.asarray(model.nucleus_flux(cid, A * (t + m_n)), float)
                R = cutoff.rigidity_gv(np.sqrt(a * b), A, Z)
                r = int(np.searchsorted(edges, R) - 1)
                v = 1.0 if r >= edges.size - 1 else verdict[max(r, 0)]
                s = float(np.sum(w * J))
                num += s * v
                den += s
            T[k] = num / den if den > 0 else 1.0
        out[cid] = T
    return out


def transmitted_class(T_by_cid, e_grid):
    """crflux H3a class whose group fluxes carry T_g(k) of the bin of their kinetic
    energy per nucleon (nearest MCEq bin centre in ln E); a class, because
    ``MCEqRun.set_primary_model`` accepts objects only of direct PrimaryFlux
    subclasses."""
    from mceq3d_primary import primary_class

    base = primary_class("HillasGaisser2012")
    m_n = 0.5 * (0.938272 + 0.939565)
    le = np.log(np.asarray(e_grid, float))

    class Transmitted(base):
        def nucleus_flux(self, corsika_id, E):
            f = super().nucleus_flux(corsika_id, E)
            Z, A = self.Z_A(corsika_id)
            t = np.maximum(np.asarray(E, float) / A - m_n, 1e-300)
            k = np.clip(np.rint((np.log(t) - le[0]) / (le[1] - le[0])).astype(int), 0,
                        le.size - 1)
            return f * T_by_cid[corsika_id][k]

    return Transmitted


def nucleon_ratio(T_by_cid, e_grid):
    """tau_x(k): MCEq's nucleon spectrum with the transmission over without, at the
    energies at which MCEq evaluates it (bin centre + mean nucleon mass)."""
    m_n = 0.5 * (0.938272 + 0.939565)
    e_tot = np.asarray(e_grid, float) + m_n
    base = h3a()
    _f, p0, n0 = base.p_and_n_flux(e_tot)
    _f, p1, n1 = transmitted_class(T_by_cid, e_grid)("H3a").p_and_n_flux(e_tot)
    rp = np.where(p0 > 0, p1 / np.where(p0 > 0, p0, 1), 1.0)
    rn = np.where(n0 > 0, n1 / np.where(n0 > 0, n0, 1), 1.0)
    return rp, rn


def node_of_cosz(table, c):
    kind, cz = np.asarray(table.nodes["kind"]), np.asarray(table.nodes["cosz"])
    i = np.flatnonzero((kind == 0) & np.isclose(cz, c, atol=1e-12))
    if i.size != 1:
        raise SystemExit(f"cos Z {c} is not a ground line node of the table")
    return int(i[0])


def sector_of(az, n_sectors):
    return int(np.floor(az / (360.0 / n_sectors))) % n_sectors


# ---------------------------------------------------------------- the two sides
def solver_setup(cosz=COSZ):
    import grid
    import kernelset as K
    import rays

    atm = grid.CorsikaAtmosphere()
    lev = grid.make_levels(atm)
    fam = grid.LineFamily(np.array(cosz), np.zeros(0))
    r = rays.Rays(atm, lev, fam)
    ks = K.from_mceq("PHYS", loss=True)
    return r, ks


def chain_cosz(r):
    return np.sqrt(np.clip(1 - (r.line_b / r.r_e) ** 2, 0, None))


def solve_with_tops(sv, ks, r, taus):
    """One nonlinear straight-ahead solve; ``taus[j] = (tau_p, tau_n)`` for chain j."""
    import assemble as A

    names = list(ks.species)
    ip, iN = names.index("p+"), names.index("n0")
    top = np.broadcast_to(ks.phi_top[:, :, None], ks.phi_top.shape + (r.Nl,)).copy()
    for j, (tp, tn) in enumerate(taus):
        top[ip, :, j] *= tp
        top[iN, :, j] *= tn
    prm = ks.params()
    prm["phi_top_c"] = top
    out = sv.solve(prm)
    return A.ground_fluxes(sv, out), float(np.max(np.asarray(out["res"])))


def mceq_solve(run, T_by_cid, theta_deg):
    import kernels_mceq as KM

    pop = KM.POPULATIONS["PHYS"]
    with KM._config(minimal_primary_energy=pop["primary_floor"], e_min=pop["e_min"]):
        run.set_primary_model(transmitted_class(T_by_cid, run.e_grid), "H3a")
    run.set_theta_deg(float(theta_deg))
    run.solve()
    d = {n: run.get_solution(n, mag=0) for n in NU}
    d["mu"] = run.get_solution("total_mu+", mag=0) + run.get_solution("total_mu-",
                                                                         mag=0)
    return d


def main(argv=None):
    import boundary as B
    import kernels_mceq as KM
    import solver3d as S3

    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(N3, "c2", "solver", "gates",
                                                     "G2.2"))
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--name", default="G2.2")
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)
    md, js = (os.path.join(a.outdir, a.name + s) for s in (".md", ".json"))
    if not a.probe:
        for p in (md, js):
            if os.path.exists(p):
                raise SystemExit(f"{p} exists; result files are never overwritten")
    w0, c0 = time.time(), cpu()
    tab = B.SkyTable.load(SITE_TABLE)
    ns = int(tab.n_sectors)
    cosz = (0.5,) if a.probe else COSZ
    azs = (90.0,) if a.probe else AZIMUTHS
    run = KM.make_mceq("PHYS", energy_loss=True)
    r, ks = solver_setup(cosz)
    if not np.allclose(run.e_grid, ks.e_grid, rtol=1e-12):
        raise SystemExit("MCEq and kernel-set grids differ")
    model = h3a()
    sv = S3.ModeSolver(r, ks, guard=S3.GUARD_NONE)
    cz_chain = chain_cosz(r)
    w1, c1 = time.time(), cpu()
    E = ks.e_grid
    sel = np.flatnonzero((E >= 0.1 - 1e-9) & (E <= 100.0 + 1e-6))
    rows, worst = [], {"nu": (0.0, None), "mu": (0.0, None)}
    t_solver = t_mceq = 0.0
    for az in azs:
        j_sec = sector_of(az, ns)
        Ts, taus = {}, []
        for c in cz_chain:
            node = node_of_cosz(tab, float(np.round(c, 10)))
            Ts[c] = group_transmission(tab, node, j_sec, ks.e_bins, model)
            taus.append(nucleon_ratio(Ts[c], ks.e_grid))
        ta = cpu()
        gf, res = solve_with_tops(sv, ks, r, taus)
        t_solver += cpu() - ta
        for jl, c in enumerate(cz_chain):
            cr = float(np.round(c, 10))
            if cr == 1.0 and az != azs[0]:
                continue                                  # the vertical once
            ta = cpu()
            m = mceq_solve(run, Ts[c], np.degrees(np.arccos(cr)))
            t_mceq += cpu() - ta
            j = int(np.flatnonzero(np.isclose(gf["mu0"], c))[0])
            row = {"cosz": cr, "az": az, "sector": j_sec, "res": res}
            for q in NU + ("mu",):
                rel = gf[q][j][sel] / m[q][sel] - 1
                k = int(np.argmax(np.abs(rel)))
                row[q] = (float(rel[k]), float(E[sel][k]))
                key = "mu" if q == "mu" else "nu"
                if abs(rel[k]) > abs(worst[key][0]):
                    worst[key] = (float(rel[k]), (q, cr, az, float(E[sel][k])))
            row["T_p_min"] = float(min(taus[jl][0][taus[jl][0] > 0].min(), 1.0))
            rows.append(row)
    w2, c2 = time.time(), cpu()
    cost = {"setup_cpu_s": c1 - c0, "loop_cpu_s": c2 - c1, "solver_cpu_s": t_solver,
            "mceq_cpu_s": t_mceq, "wall_s": w2 - w0, "total_cpu_s": c2 - c0,
            "affinity": sorted(os.sched_getaffinity(0)), "n_directions": len(rows)}
    if a.probe:
        print(json.dumps({"probe": rows, "cost": cost, "worst": worst}, indent=1))
        return 0
    ok = abs(worst["nu"][0]) <= TOL and len(rows) == 21
    res = {"gate": "G2.2", "verdict": "PASS" if ok else "FAIL", "rows": rows,
           "worst_neutrino": worst["nu"], "worst_muon": worst["mu"], "cost": cost,
           "sky_table": SITE_TABLE, "sky_sha256": sha256(SITE_TABLE),
           "command": " ".join(sys.argv), "script_sha256": sha256(__file__)}
    with open(js, "w") as fh:
        json.dump(res, fh, indent=1)
    L = ["# Gate G2.2: straight-ahead limit with the cutoff against MCEq", "", GATE,
         "", f"## Result: **{res['verdict']}**", "",
         f"Worst neutrino cell over the 21 directions, 0.1 to 100 GeV: "
         f"{worst['nu'][0]:+.3%} ({worst['nu'][1][0]}, cos Z {worst['nu'][1][1]:g}, "
         f"azimuth {worst['nu'][1][2]:g}, E = {worst['nu'][1][3]:.3g} GeV); "
         f"tolerance {TOL:.0%}.", "",
         f"Ground muons (reported): worst {worst['mu'][0]:+.3%} ({worst['mu'][1][1]:g}"
         f", azimuth {worst['mu'][1][2]:g}, E = {worst['mu'][1][3]:.3g} GeV).", "",
         "| cos Z | azimuth | sector | " + " | ".join(NU) + " | mu (reported) | "
         "smallest proton transmission |", "|---" * 9 + "|"]
    for row in rows:
        L.append(f"| {row['cosz']:g} | {row['az']:g} | {row['sector']} | " +
                 " | ".join(f"{row[q][0]:+.2e} ({row[q][1]:.3g})" for q in NU) +
                 f" | {row['mu'][0]:+.2e} ({row['mu'][1]:.3g}) | "
                 f"{row['T_p_min']:.3f} |")
    L += ["", "Largest |solver / MCEq - 1| per species over 0.1 to 100 GeV (energy of "
          "that cell in GeV).", "",
          f"Sky table `{SITE_TABLE}` (sha256 {res['sky_sha256'][:16]}). Cost: "
          f"{cost['total_cpu_s'] / 60:.1f} core-min (MCEq {t_mceq / 60:.1f}, solver "
          f"{t_solver / 60:.1f}), wall {cost['wall_s'] / 60:.1f} min, cores "
          f"{cost['affinity']}.", "", "Command:", "", f"    {res['command']}", "",
          f"Script sha256: {res['script_sha256']}", ""]
    with open(md, "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
