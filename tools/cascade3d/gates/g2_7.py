"""Gate G2.7 (linearisation of the guarded tangent); writes G2.7.md / .json.

Gate text: ``tools/cascade3d/GATES.md``, "C2, dated entries", 2026-11-23 (quoted in
the result file).  Settings: the frozen C1 solve (``RESULTS_C1.md`` section 5:
CORSIKA BK_USStd, R_E 6371 km, ``make_levels`` default, 48 ground lines uniform in
zenith angle plus the G1.1 and G1.3 directions, one tangent line per level,
``Rays(n_sub=4, dX_max=5)``, L = 32, hat projection, 6 sweeps, gamma0 = 2, n_up = 3)
and the kernel set of G2.3 (``kernels/mc_close/ks_mc_pool2000_L32_losson.npz``,
sha256 ``32b4215d5bf0de6d...``).

Deficits (D = B0 - B, B0 = ``phi_top``):

1. isotropic step: T = 1 / (1 + exp(-ln(R / 11.5 GV) / 0.15)) at the bin centres,
   R = p for protons and 2 p for neutrons (p = sqrt(E (E + 2 m_p)), E the kinetic
   energy per nucleon of the bin centre), the same on every chain (``DESIGN_C2.md``
   1.6, test 1; ``lin_test.transmission`` of the design study);
2. the azimuth average (order m = 0) of the deficit of the cutoff sky at the site:
   ``boundary.SkyTable`` of the site (h_ref = 0), ``boundary.GroupWeights``
   (GSFG:GH), ``boundary.deficit_lines``, mean over the 36 sectors, then
   ``boundary.chain_weights`` / ``deficit_chains`` (lines that never reach the
   injection sphere: D = B0).

Phi_lin = Phi_C1 - J[D]: ``solver3d.TangentSolver`` (guarded tangent, order 0) at the
C1 state, the tangent of the neutrino line-of-sight integral by
``solver3d.sink_tangent``, then the cell means of ``assemble.cell_table`` (PCHIP in
cos Z, bin-convention band means) applied to the intensities Phi_C1(mu0) - J[D](mu0)
of each ground direction (``DESIGN_C2.md`` 3.3).  Phi_C1: the primal of the same run
(checked against the stored C1 solve ``c1/close/g13/pool2000.npz``).  Phi_nl: the
nonlinear C1 scheme with the boundary B0 - D: ``solver.Solver`` for deficit 1 (one top
for all chains), ``solver3d.ModeSolver`` with per-chain tops for deficit 2.

Pass, for both deficits, in the 400 cells of G1.3 (four neutrino species, ten bands
0.1 to 10 GeV, ten cos Z bins 0 to 1):
(i) |Phi_lin - Phi_nl| <= 0.005 Phi_nl in every cell;
(ii) |Phi_lin - Phi_nl| <= 0.01 (Phi_C1 - Phi_nl) in every cell with
Phi_nl <= 0.95 Phi_C1.
Ground muons (the six muon states summed, ground lines, bins 0.1 to 10 GeV) are
reported.

Usage (each step one process; ``taskset`` as in ``RESULTS_C1.md``)::

    $PY g2_7.py deficits OUTDIR --sky SKY.npz
    taskset -c 12-15 $PY g2_7.py nl OUTDIR --deficit 1
    taskset -c 16-19 $PY g2_7.py nl OUTDIR --deficit 2
    taskset -c 0-7  $PY g2_7.py lin OUTDIR \
        --guard shape=0.1,integral=0.1,loss=0.1,simpson=none
    $PY g2_7.py evaluate OUTDIR --variants TAG[,TAG...] [--first TAG] [--name G2.7]

``evaluate`` writes ``OUTDIR/<name>.md`` and ``.json`` and refuses to overwrite them.
Exit status of ``evaluate``: 0 PASS (some variant passes), 1 FAIL, 2 input error.
"""

import argparse
import hashlib
import json
import os
import resource
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

N3 = os.environ.get("N3", "/home/pigrange/mc3d_work/runs/native3d")
KS_PATH = os.path.join(N3, "kernels", "mc_close", "ks_mc_pool2000_L32_losson.npz")
KS_SHA16 = "32b4215d5bf0de6d"
C1_SOLVE = os.path.join(N3, "c1", "close", "g13", "pool2000.npz")
R_EARTH_KM = 6371.0
M_P_STEP = 0.938272            # the design study's proton mass (lin_test.py)
SPECIES = ("nue", "antinue", "numu", "antinumu")
MUONS = ("mu+_l", "mu+", "mu+_r", "mu-_l", "mu-", "mu-_r")
TOL_I, TOL_II, K_MAX_II = 0.005, 0.01, 0.95

GATE = """\
**G2.7 Linearisation (gated; an addition to the plan).** The exact derivative of the C1
solve is unstable below 1.6 GeV (`DESIGN_C2.md`, 1.6; measured on a reduced geometry).
The guarded tangent is not the exact derivative of anything, so its accuracy is known
only against nonlinear solves. Test, on the full geometry and settings of the frozen C1
solve (`RESULTS_C1.md`, section 5; kernel set of G2.3), with two deficits that the
nonlinear C1 scheme can solve because they are symmetric in azimuth:
1. the isotropic step of the design's test 1: a smooth step in rigidity at 11.5 GV, 15%
   wide, R = p for protons and 2p for neutrons;
2. the azimuth average (order m = 0) of the deficit of the cutoff sky at the site, from
   `boundary.py`.

For each, Φ_lin = Φ_C1 − J[D] against Φ_nl, the nonlinear solve with the boundary
B⁰ − D, in the 400 cells of G1.3. Pass, for both deficits:
- (i) |Φ_lin − Φ_nl| ≤ 0.005 Φ_nl in every cell;
- (ii) |Φ_lin − Φ_nl| ≤ 0.01 (Φ_C1 − Φ_nl) in every cell with Φ_nl ≤ 0.95 Φ_C1.

Ground muons are reported. First setting: the prototype's (ρ = 0.1 on the source shape,
the source integral and the loss interpolation; the Simpson correction not guarded). On
the reduced geometry it gave 0.42% for (i) and 0.23% for (ii), so the margin of (i) is
small.

If it fails: one tuning cycle (guard on the Simpson correction; ρ per piece). Every
variant is run in full on both deficits and every result is reported. The setting that
passes is frozen in `RESULTS_C2.md`. If none passes, the gate is reported as failed. The
fallback of the design (chains in φ, two to three times the cost per boundary) is not
started without the PI. No solve of the mode path with a cutoff sky is run before
G2.7 has passed."""


def cpu():
    t = os.times()
    return t.user + t.system


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 ** 2


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def build_geometry():
    """Rays and kernel set of the frozen C1 solve (``run_solve.py MC losson
    --r-earth 6371 --ks ks_mc_pool2000_L32_losson.npz``)."""
    import grid
    import kernelset as K
    import rays

    r_e = R_EARTH_KM * 1e5
    atm = grid.CorsikaAtmosphere(r_e=r_e)
    lev = grid.make_levels(atm)
    fam = grid.make_line_family(lev, n_ground=48, r_e=r_e)
    ry = rays.Rays(atm, lev, fam, n_sub=4, dX_max=5.0)
    sha = sha256(KS_PATH)
    if not sha.startswith(KS_SHA16):
        raise SystemExit(f"kernel set hash {sha[:16]} != {KS_SHA16}")
    ks = K.KernelSet.load(KS_PATH)
    return ry, ks, sha


def step_transmission(ks):
    """Deficit 1: the design's isotropic step (``lin_test.transmission``)."""
    T = np.ones_like(ks.phi_top)
    E = ks.e_grid
    p = np.sqrt(E * (E + 2 * M_P_STEP))
    names = list(ks.species)
    for sp, az in (("p+", 1.0), ("n0", 2.0)):
        R = az * p
        T[names.index(sp)] = 1.0 / (1.0 + np.exp(-np.log(R / 11.5) / 0.15))
    return T


def cmd_deficits(a):
    import boundary as B

    ry, ks, sha = build_geometry()
    names = list(ks.species)
    ip, iN = names.index("p+"), names.index("n0")
    b0 = ks.phi_top[[ip, iN]]
    # deficit 1 (nS, nE): the same on every chain
    D1 = ks.phi_top * (1.0 - step_transmission(ks))
    # deficit 2 (nS, nE, Nl)
    tab = B.SkyTable.load(a.sky)
    gw = B.GroupWeights(ks.e_bins, model="GSFG:GH")
    Dl = B.deficit_lines(tab, gw, b0)                        # (nn, 36, nE, 2)
    c, _s = B.fourier_sectors(Dl, 0, axis=1)                  # order 0 = mean
    D0 = c[:, 0]                                              # (nn, nE, 2)
    cw = B.chain_weights(ry, tab.nodes)
    Dc = B.deficit_chains(D0[:, None], cw, b0)[:, 0]          # (Nl, nE, 2)
    D2 = np.zeros(ks.phi_top.shape + (ry.Nl,))
    D2[ip] = Dc[:, :, 0].T
    D2[iN] = Dc[:, :, 1].T
    out = os.path.join(a.outdir, "deficits.npz")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists")
    np.savez(out, D1=D1, D2=D2, sky=a.sky, sky_sha256=B._sha256(a.sky),
             ks_sha256=sha, never=cw[4], b0=b0, D_lines_m0=D0,
             gw_C=gw.C, gw_above=gw.above, gw_flux=gw.flux)
    with np.errstate(invalid="ignore", divide="ignore"):
        k1 = 1 - D1[[ip, iN]].sum(0) / b0.sum(0)
    print(json.dumps({"out": out, "never_chains": int(cw[4].sum()),
                      "D2_rel_max_over_ground_chains": float(np.nanmax(
                          D2[ip][:, ry.line_kind == 0] / np.where(
                              b0[0][:, None] > 0, b0[0][:, None], np.nan))),
                      "T_step_p_bins_16_33": np.round(k1[16:34], 4).tolist()},
                     indent=1))


def _ground_tables(sv, emis, F_ground):
    """Sink intensity table at the ground directions (+ horizontal node), and the
    ground fluxes of the transported species."""
    import assemble as A
    import solver3d as S3

    nodes, mu0 = S3.ground_table_nodes(sv.rays)
    I = np.asarray(A.sink_intensity(sv, emis))[:, :, nodes]
    return mu0, I, np.asarray(F_ground)


def cmd_nl(a):
    import jax.numpy as jnp
    import solver as S
    import solver3d as S3

    w0, c0 = time.time(), cpu()
    ry, ks, sha = build_geometry()
    d = np.load(os.path.join(a.outdir, "deficits.npz"))
    prm = {k: jnp.asarray(v) for k, v in ks.params().items()}
    if a.deficit == 1:
        sv = S.Solver(ry, ks)
        prm["phi_top"] = prm["phi_top"] - jnp.asarray(d["D1"])
        kind = "solver.Solver, phi_top - D1"
    else:
        sv = S3.ModeSolver(ry, ks, guard=S3.GUARD_NONE)
        top = np.broadcast_to(ks.phi_top[:, :, None], d["D2"].shape) - d["D2"]
        prm["phi_top_c"] = jnp.asarray(top)
        kind = "solver3d.ModeSolver (per-chain tops), phi_top - D2"
    w1, c1 = time.time(), cpu()
    out = sv.solve(prm, verbose=True)
    out["res"].block_until_ready()
    w2, c2 = time.time(), cpu()
    mu0, I, Fg = _ground_tables(sv, out["emis"], out["F_ground"])
    w3, c3 = time.time(), cpu()
    cost = {"kind": kind, "setup_cpu_s": c1 - c0, "solve_cpu_s": c2 - c1,
            "assemble_cpu_s": c3 - c2, "total_cpu_s": c3 - c0,
            "total_wall_s": w3 - w0, "peak_rss_GB": rss_gb(),
            "affinity": sorted(os.sched_getaffinity(0)),
            "res_max": float(np.max(np.asarray(out["res"]))),
            "up_res": [float(x) for x in np.asarray(out["up_res"])]}
    cost.update(ry.summary())
    print(json.dumps(cost, indent=1), flush=True)
    path = os.path.join(a.outdir, f"nl{a.deficit}.npz")
    if os.path.exists(path):
        raise SystemExit(f"{path} exists")
    np.savez(path, mu0=mu0, I=I, F_ground=Fg, res=np.asarray(out["res"]),
             cost_json=json.dumps(cost), ground_mu0=ry.ground_mu0)


def cmd_lin(a):
    import jax.numpy as jnp
    import solver3d as S3

    w0, c0 = time.time(), cpu()
    guard = S3.Guard.parse(a.guard)
    ry, ks, sha = build_geometry()
    d = np.load(os.path.join(a.outdir, "deficits.npz"))
    Nl = ry.Nl
    dt1 = -np.broadcast_to(d["D1"][:, :, None], d["D1"].shape + (Nl,))
    dt2 = -d["D2"]
    ts = S3.TangentSolver(ry, ks, guard=guard)
    w1, c1 = time.time(), cpu()
    out = ts.solve(np.stack([dt1, dt2]), [0, 0], verbose=True)
    out["d_emis"].block_until_ready()
    w2, c2 = time.time(), cpu()
    nodes, mu0 = S3.ground_table_nodes(ry)
    I, dI = S3.sink_tangent(ts.sv, out["emis"], out["d_emis"])
    I = np.asarray(I)[:, :, nodes]
    dI = np.asarray(dI)[:, :, :, nodes]
    w3, c3 = time.time(), cpu()
    emis = np.asarray(out["d_emis"])
    lin_emis = np.asarray(out["emis"])[None] + emis
    cost = {"guard": guard.as_dict(), "tag": guard.tag(), "setup_cpu_s": c1 - c0,
            "solve_cpu_s": c2 - c1, "assemble_cpu_s": c3 - c2,
            "total_cpu_s": c3 - c0, "total_wall_s": w3 - w0,
            "peak_rss_GB": rss_gb(), "affinity": sorted(os.sched_getaffinity(0)),
            "res_max": float(np.max(np.asarray(out["res"]))),
            "d_res_max": [float(x) for x in np.max(np.asarray(out["d_res"]), axis=1)],
            "up_res": [float(x) for x in out["up_res"]],
            "d_up_res": np.asarray(out["d_up_res"]).tolist(),
            "lin_emis_neg_frac": [float(np.mean(e < 0)) for e in lin_emis]}
    cost.update(ry.summary())
    print(json.dumps(cost, indent=1), flush=True)
    path = os.path.join(a.outdir, f"lin_{guard.tag()}.npz")
    if os.path.exists(path):
        raise SystemExit(f"{path} exists")
    np.savez(path, mu0=mu0, I=I, dI=dI, F_ground=np.asarray(out["F_ground"]),
             dF_ground=np.asarray(out["d_F_ground"]), res=np.asarray(out["res"]),
             d_res=np.asarray(out["d_res"]), cost_json=json.dumps(cost),
             ground_mu0=ry.ground_mu0)


# --------------------------------------------------------------- evaluation
def cells(mu0, I, ks, sp_index):
    """(4, 10, 10) cells of the four neutrino species from a table I (nE, nSk, n)."""
    import assemble as A

    return np.stack([A.cell_table(mu0, I[:, sp_index[s], :], ks.e_bins, ks.e_grid)
                     for s in SPECIES])


def criteria(lin, nl, c1):
    """Numbers of (i) and (ii) for cells (4, 10, 10)."""
    d = np.abs(lin - nl)
    r1 = d / nl
    i1 = np.unravel_index(np.argmax(r1), r1.shape)
    sel = nl <= K_MAX_II * c1
    den = c1 - nl
    r2 = np.where(sel, d / np.where(sel, den, 1.0), 0.0)
    i2 = np.unravel_index(np.argmax(r2), r2.shape)
    return {"i_max": float(r1.max()), "i_at": [int(x) for x in i1],
            "i_pass": bool(np.all(d <= TOL_I * nl)),
            "i_cells_above_1e-3": int(np.sum(r1 > 1e-3)),
            "i_rms": float(np.sqrt(np.mean(r1 ** 2))),
            "ii_max": float(r2.max()), "ii_at": [int(x) for x in i2],
            "ii_cells": int(sel.sum()),
            "ii_pass": bool(np.all(np.where(sel, d <= TOL_II * den, True))),
            "K_min": float((nl / c1).min()), "K_max": float((nl / c1).max())}


def muon_numbers(ks, ry_mu0, Fc1, Flin, Fnl, T_list, e_lo=0.1, e_hi=10.0):
    names = list(ks.species)
    idx = [T_list.index(names.index(m)) for m in MUONS]

    def mu(F):
        return np.asarray(F)[:, idx, :].sum(axis=1)          # (nE, n_ground)

    c1, li, nl = mu(Fc1), mu(Flin), mu(Fnl)
    kb = (ks.e_grid >= e_lo) & (ks.e_grid <= e_hi)
    lines = np.asarray(ry_mu0) >= 0.05 - 1e-12
    sel = kb[:, None] & lines[None, :] & (nl > 0)
    d = np.abs(li - nl)
    r1 = np.where(sel, d / np.where(sel, nl, 1.0), 0.0)
    i1 = np.unravel_index(np.argmax(r1), r1.shape)
    sel2 = sel & (nl <= K_MAX_II * c1)
    r2 = np.where(sel2, d / np.where(sel2, c1 - nl, 1.0), 0.0)
    i2 = np.unravel_index(np.argmax(r2), r2.shape)
    return {"rel_max": float(r1.max()), "rel_at_E_GeV": float(ks.e_grid[i1[0]]),
            "rel_at_cosz": float(ry_mu0[i1[1]]),
            "rel_to_deficit_max": float(r2.max()),
            "rel_to_deficit_at_E_GeV": float(ks.e_grid[i2[0]]),
            "rel_to_deficit_at_cosz": float(ry_mu0[i2[1]]),
            "n_points": int(sel.sum()), "n_points_deficit": int(sel2.sum())}


def cell_name(at, band_edges, cosz_edges):
    s, b, c = at
    return (f"{SPECIES[s]}, {band_edges[b]:.3g}-{band_edges[b + 1]:.3g} GeV, "
            f"cos Z {cosz_edges[c]:.1f}-{cosz_edges[c + 1]:.1f}")


def cmd_evaluate(a):
    import assemble as A
    import kernelset as K

    ks = K.KernelSet.load(KS_PATH)
    names = list(ks.species)
    sinks = [i for i in range(ks.nS) if ks.role[i] == K.ROLE_SINK]
    sp_index = {s: sinks.index(names.index(s)) for s in SPECIES}
    T_list = [i for i in range(ks.nS) if ks.role[i] == K.ROLE_HADRON] + [
        i for i in range(ks.nS) if ks.role[i] == K.ROLE_LOSS]
    be, ce = np.asarray(A.G13_BAND_EDGES), np.asarray(A.G13_COSZ_EDGES)
    nl = {j: np.load(os.path.join(a.nl_dir or a.outdir, f"nl{j}.npz"))
          for j in (1, 2)}
    c1ref = np.load(C1_SOLVE)
    variants = [v for v in a.variants.split(",") if v]
    res = {"gate": "G2.7", "variants": {}, "nl_cost": {}, "commands": a.commands,
           "kernel_set": KS_PATH, "c1_solve": C1_SOLVE}
    for j in (1, 2):
        res["nl_cost"][j] = json.loads(str(nl[j]["cost_json"]))
    d = np.load(os.path.join(a.outdir, "deficits.npz"))
    res["sky"] = str(d["sky"])
    res["sky_sha256"] = str(d["sky_sha256"])
    res["ks_sha256"] = str(d["ks_sha256"])
    any_pass = False
    for tag in variants:
        L = np.load(os.path.join(a.outdir, f"lin_{tag}.npz"))
        mu0 = L["mu0"]
        cost = json.loads(str(L["cost_json"]))
        C1 = cells(mu0, L["I"], ks, sp_index)
        # the primal against the stored frozen C1 solve
        st = np.stack([c1ref[f"cell_{s}"] for s in SPECIES])
        dev_c1 = float(np.max(np.abs(C1 / st - 1)))
        v = {"cost": cost, "primal_vs_stored_C1_cells_max_rel": dev_c1,
             "deficits": {}}
        ok_all = True
        for j in (1, 2):
            NL = cells(nl[j]["mu0"], nl[j]["I"], ks, sp_index)
            LIN = cells(mu0, L["I"] + L["dI"][j - 1], ks, sp_index)
            c = criteria(LIN, NL, C1)
            c["i_at_cell"] = cell_name(c["i_at"], be, ce)
            c["ii_at_cell"] = cell_name(c["ii_at"], be, ce)
            c["pass"] = c["i_pass"] and c["ii_pass"]
            c["muons"] = muon_numbers(ks, L["ground_mu0"], L["F_ground"],
                                      L["F_ground"] + L["dF_ground"][j - 1],
                                      nl[j]["F_ground"], T_list)
            # per-band worst of (i)
            r1 = np.abs(LIN - NL) / NL
            c["i_worst_per_band"] = [float(x) for x in r1.max(axis=(0, 2))]
            v["deficits"][j] = c
            ok_all &= c["pass"]
            np.savez(os.path.join(a.outdir, f"cells_{tag}_D{j}.npz"), lin=LIN, nl=NL,
                     c1=C1, band_edges=be, cosz_edges=ce)
        v["pass"] = bool(ok_all)
        any_pass |= ok_all
        res["variants"][tag] = v
    res["verdict"] = "PASS" if any_pass else "FAIL"
    md = os.path.join(a.outdir, a.name + ".md")
    js = os.path.join(a.outdir, a.name + ".json")
    for p in (md, js):
        if os.path.exists(p):
            raise SystemExit(f"{p} exists; result files are never overwritten")
    with open(js, "w") as fh:
        json.dump(res, fh, indent=1)
    lines = [f"# Gate G2.7: linearisation of the guarded tangent ({a.name})", "",
             GATE, "", f"## Result: **{res['verdict']}**", "",
             "Variants run (each in full on both deficits): " + ", ".join(
                 f"`{t}`" for t in variants) + ".", ""]
    for tag in variants:
        v = res["variants"][tag]
        g = v["cost"]["guard"]
        lines += [f"### Variant `{tag}`: {'PASS' if v['pass'] else 'FAIL'}", "",
                  f"Guard: rho on source shape {g['shape']}, source integral "
                  f"{g['integral']}, loss interpolation {g['loss']}, Simpson "
                  f"correction {g['simpson']} (None = exact derivative).", "",
                  "| deficit | (i) max abs(lin-nl)/nl | cell | (ii) max abs(lin-nl)"
                  "/(C1-nl) | cell | cells in (ii) | K = nl/C1 range | (i) | (ii) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for j in (1, 2):
            c = v["deficits"][j]
            lines.append(
                f"| {j} | {c['i_max']:.3e} | {c['i_at_cell']} | {c['ii_max']:.3e} | "
                f"{c['ii_at_cell']} | {c['ii_cells']} | {c['K_min']:.3f} to "
                f"{c['K_max']:.3f} | {'pass' if c['i_pass'] else 'FAIL'} | "
                f"{'pass' if c['ii_pass'] else 'FAIL'} |")
        lines += ["", "Worst (i) per energy band (bands 0.1 to 10 GeV, ascending):", ""]
        for j in (1, 2):
            c = v["deficits"][j]
            lines.append(f"- deficit {j}: " + ", ".join(
                f"{x:.1e}" for x in c["i_worst_per_band"]) +
                f"; cells above 1e-3: {c['i_cells_above_1e-3']}; rms "
                f"{c['i_rms']:.1e}")
        lines += ["", "Ground muons (reported; six muon states summed, ground lines "
                  "cos Z >= 0.05, bins 0.1 to 10 GeV):", ""]
        for j in (1, 2):
            m = v["deficits"][j]["muons"]
            lines.append(
                f"- deficit {j}: max abs(lin-nl)/nl {m['rel_max']:.3e} at "
                f"{m['rel_at_E_GeV']:.3g} GeV, cos Z {m['rel_at_cosz']:.3f}; max "
                f"abs(lin-nl)/(C1-nl) where nl <= 0.95 C1 {m['rel_to_deficit_max']:.3e}"
                f" at {m['rel_to_deficit_at_E_GeV']:.3g} GeV, cos Z "
                f"{m['rel_to_deficit_at_cosz']:.3f}")
        cst = v["cost"]
        lines += ["", f"Primal of the tangent run against the stored C1 solve "
                  f"(`{C1_SOLVE}`), 400 cells: max relative difference "
                  f"{v['primal_vs_stored_C1_cells_max_rel']:.2e}.",
                  f"Tangent run: {cst['total_cpu_s'] / 60:.1f} core-min (script), "
                  f"peak RSS {cst['peak_rss_GB']:.2f} GiB, cores {cst['affinity']}; "
                  f"same-bin sweep residual primal {cst['res_max']:.1e}, tangents "
                  f"{', '.join(f'{x:.1e}' for x in cst['d_res_max'])}; up-scatter "
                  f"passes primal {cst['up_res']}, tangents {cst['d_up_res']}.", ""]
    for j in (1, 2):
        c = res["nl_cost"][j]
        lines.append(f"Nonlinear solve {j} ({c['kind']}): {c['total_cpu_s'] / 60:.1f} "
                     f"core-min, peak RSS {c['peak_rss_GB']:.2f} GiB, sweep residual "
                     f"{c['res_max']:.1e}.")
    lines += ["", f"Sky table of deficit 2: `{res['sky']}` (sha256 "
              f"{res['sky_sha256'][:16]}); kernel set sha256 {res['ks_sha256'][:16]}.",
              "", "Commands:", ""] + [f"    {c}" for c in a.commands] + [
              "", f"    {' '.join(sys.argv)}", ""]
    with open(md, "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    return 0 if any_pass else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("deficits", "nl", "lin", "evaluate"))
    ap.add_argument("outdir")
    ap.add_argument("--sky", default="")
    ap.add_argument("--deficit", type=int, choices=(1, 2))
    ap.add_argument("--guard", default="shape=0.1,integral=0.1,loss=0.1,simpson=none")
    ap.add_argument("--variants", default="")
    ap.add_argument("--nl-dir", default="")
    ap.add_argument("--name", default="G2.7")
    ap.add_argument("--commands", nargs="*", default=[],
                    help="the commands of the runs, quoted in the result file")
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)
    return {"deficits": cmd_deficits, "nl": cmd_nl, "lin": cmd_lin,
            "evaluate": cmd_evaluate}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main() or 0)
