"""2026-11 rigidity-convention fix and the engine primary scan (``RIGIDITY_FIX.md``).

Sub-commands (outputs under ``$W/runs/rigfix``; nothing is written to AFS)::

    chunk PRIMARY CZ I N   G_s of the 40-point rigidity grid at cos Z_ref = CZ for
                           chunk I of N (the rigidity points are independent solves;
                           this is only a way to use many cores)
    assemble PRIMARY CZ    write the engine's own G_s cache file from the N chunks
                           (key and layout identical to ``geomag_response``)
    scan PRIMARY MB        delivered-engine observables for one primary and
                           ``muon_bending`` = MB (0/1), vs Honda
    m3b PRIMARY MB         the 36 cells of ``analyse_m3b`` (cz_sub = 3, band-integrated)
    tables                 markdown tables from the json files

PRIMARY: gsf | h3a | ghn (GaisserHondaNative) | ght (GaisserHondaAtTotal).
"""
import importlib.util  # noqa: F401
import json
import os
import sys
import time
import warnings
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
W = os.environ.get("MC3D_WORK", "/home/pigrange/mc3d_work")
RUN = os.path.join(W, "runs", "rigfix")
CACHE_REQ = os.path.join(W, "runs", "dbv142", "cache_old")  # DB-independent maps
PRIM = {"gsf": ("GlobalSplineFitBeta", None), "h3a": ("HillasGaisser2012", "H3a"),
        "ghn": ("GaisserHondaNative", None), "ght": ("GaisserHondaAtTotal", None)}
LAT, LON = 36.43, 137.31
DATE = datetime(2020, 1, 1)
SPEC = ("total_numu", "total_antinumu", "total_nue", "total_antinue")
E_PTS = (0.3, 0.5, 1.0)
CZ_BANDS = (0.25, 0.15, 0.05)


def out(name):
    os.makedirs(os.path.join(RUN, os.path.dirname(name)), exist_ok=True)
    return os.path.join(RUN, name)


def rc_grid():
    from mceq3d_flux import RC_MAX_GV
    return np.linspace(0.1, RC_MAX_GV, 40)


def make_engine(prim, base="hybrid"):
    from mceq3d_flux import MCEq3DFlux
    kw = dict(daemonflux_location="kamioka") if base == "hybrid" else {}
    return MCEq3DFlux(base_model=base, primary=PRIM[prim], **kw)


def cmd_chunk(prim, cz, i, n):
    cz = float(cz)
    i, n = int(i), int(n)
    eng = make_engine(prim, "mceq")
    g = rc_grid()
    idx = np.arange(len(g))[i::n]
    t = time.time()
    G, _ = eng.geomag_response(g[idx], cz_ref=cz, cache_dir=None)
    np.savez(out(f"chunks/{prim}_{cz:.4f}_{i}of{n}.npz"), idx=idx, e=eng.e,
             **{k: v for k, v in G.items()})
    print("chunk done", prim, cz, i, n, f"{time.time() - t:.0f}s")


def cmd_assemble(prim, cz, n=8):
    import glob
    import mceq3d_db as db
    from mceq3d_flux import SPECIES, SIGMA_LNR, engine_tag, gs_cache_name
    cz = float(cz)
    files = sorted(glob.glob(out(f"chunks/{prim}_{cz:.4f}_*of*.npz")))
    d0 = np.load(files[0])
    g = rc_grid()
    G = {s: np.ones((len(g), len(d0["e"]))) for s in SPECIES}
    seen = np.zeros(len(g), bool)
    for f in files:
        d = np.load(f)
        for s in SPECIES:
            G[s][d["idx"]] = d[s]
        seen[d["idx"]] = True
    assert seen.all(), "missing chunks"
    tag = engine_tag("SIBYLL23D", PRIM[prim], 0.1, None, db.db_tag())
    path = os.path.join(db.cache_dir(CACHE_REQ), gs_cache_name(tag, cz, g, float(SIGMA_LNR)))
    np.savez(path, e=d0["e"], **{s: G[s] for s in SPECIES})
    print("wrote", path)


# ---------------------------------------------------------------- scan
def _honda_pattern(h, key, cz_c, E, fz):
    icz = int(np.argmin(np.abs(h["czlo"] - round(cz_c - 0.05, 2))))
    az_c = fz.honda_azimuth_to_compass(h["azlo"].astype(float) + 15.0)
    rows = h[key][icz]
    f = np.zeros(12)
    for j in range(rows.shape[0]):
        k = int(round(((az_c[j] - 15.0) % 360.0) / 30.0)) % 12
        f[k] = fz._log_at(rows[j], h["E"], E)
    return f


def cmd_scan(prim, mb):
    import analyse_m3b as M
    import diag_ew_charge_fourier as fz
    mb = bool(int(mb))
    warnings.simplefilter("ignore")
    eng = make_engine(prim)
    ax = fz.geomagnetic_ew_axis()
    h = dict(np.load(os.path.join(HERE, "honda_kam.npz")))
    e, F, az = fz.model_flux(muon_bending=mb, cz_sub=1, cache_dir=CACHE_REQ, engine=eng)
    res = {"prim": prim, "mb": mb, "axis": ax, "obs": list(M.OBS), "cells": {}}
    for sp in SPEC:
        for icz, czc in enumerate(CZ_BANDS):
            for E in E_PTS:
                f = np.array([fz._log_at(F[sp][icz, j], e, E) for j in range(12)])
                fh = _honda_pattern(h, fz.HKEY[sp], czc, E, fz)
                res["cells"][f"{sp}|{czc}|{E}"] = {
                    "model": M.observables(f, ax).tolist(),
                    "honda": M.observables(fh, ax).tolist()}
    # Phase-1 definition of W/E: engine flux at azimuth 270 over 90, cos Z = 0.05
    r = eng.solve(LAT, LON, np.array([0.05]), np.array([90.0, 270.0]), date=DATE,
                  use_cache=True, cache_dir=CACHE_REQ, muon_bending=mb)
    ee = r["e"]
    Hcz = h["czlo"]
    ih = int(np.argmin(abs(Hcz - 0.0)))
    for sp, key in (("total_numu", "numu"), ("total_nue", "nue")):
        f = r["flux"][sp][0]
        res[f"WE_{sp}"] = {str(E): [float(np.interp(E, ee, f[1] / f[0])),
                                    float(np.interp(E, h["E"], h[key][ih].max(0)
                                                    / h[key][ih].min(0)))] for E in E_PTS}
    with open(out(f"out/scan_{prim}_mb{int(mb)}.json"), "w") as fh_:
        json.dump(res, fh_, indent=1)
    print("scan done", prim, mb)


def cmd_m3b(prim, mb):
    import analyse_m3b as M
    import diag_ew_charge_fourier as fz
    mb = bool(int(mb))
    warnings.simplefilter("ignore")
    eng = make_engine(prim)
    e, F, az = fz.model_flux(muon_bending=mb, cz_sub=3, cache_dir=CACHE_REQ, engine=eng)
    order = [2, 1, 0]
    p = out(f"out/m3b_engine_{prim}_mb{int(mb)}.npz")
    np.savez(p, e=e, az=az, **{s: F[s][order] for s in F})
    ax = fz.geomagnetic_ew_axis()
    eo = M.table_obs(M.engine_patterns(p), ax)
    ho = M.table_obs(M.honda_patterns(), ax)
    np.savez(out(f"out/m3b_obs_{prim}_mb{int(mb)}.npz"), engine=eo, honda=ho,
             obs=np.array(M.OBS), species=np.array(M.SP_NAMES))
    print("m3b done", prim, mb)


def _at(e, y, x):
    return float(np.interp(np.log(x), np.log(e), y))


def cmd_tables():
    """before/after (v142p, old kinetic-energy cut vs momentum cut) and the scan."""
    D = os.path.join(W, "runs", "dbv142", "out")
    B = json.load(open(os.path.join(D, "before", "engine_v142p.json")))
    A = json.load(open(os.path.join(D, "engine_v142p.json")))
    L = ["**A. v142p before | after the rigidity fix (delivered engine, hybrid base, GSF, Kamioka)**", "",
         "| observable | before | after | after/before - 1 | Honda |", "|---|---|---|---|---|"]

    def row(n, b, a, h=None, fmt="{:.3f}", absd=False):
        d = f"{a - b:+.3f}" if absd else f"{100 * (a / b - 1):+.2f}%"
        L.append(f"| {n} | {fmt.format(b)} | {fmt.format(a)} | {d} | "
                 f"{'' if h is None else fmt.format(h)} |")
    for fl in ("numu", "nue"):
        for i, e in enumerate((0.3, 0.5, 1.0, 3.0)):
            row(f"H/V {fl} {e} GeV", B[f"hv_{fl}"][i], A[f"hv_{fl}"][i], B[f"hv_{fl}_honda"][i])
    for fl in ("numu", "nue"):
        for e in (0.5, 1.0):
            for i, z in enumerate((87, 81, 75)):
                row(f"W/E {fl} {e} GeV {z} deg", B[f"we_{fl}_{e}"][i], A[f"we_{fl}_{e}"][i],
                    B[f"we_{fl}_{e}_honda"][i])
    for fl in ("numu", "nue"):
        for e in (0.3, 0.5):
            for q, n in (("dphi", "D(dphi) deg"), ("maxmin", "D(max/min)")):
                row(f"{fl}-anti {e} GeV {n}", B[f"split_{fl}_{e}_model_{q}"],
                    A[f"split_{fl}_{e}_model_{q}"], B[f"split_{fl}_{e}_honda_{q}"], "{:+.3f}", True)
    for sp in ("total_numu", "total_antinumu", "total_nue", "total_antinue"):
        for q in ("N", "S"):
            row(f"{q} {sp[6:]} model/Honda 0.5 GeV", B[f"{q}_{sp}"], A[f"{q}_{sp}"], 1.0)
        row(f"phase offset {sp[6:]} deg", B[f"phase_{sp}"], A[f"phase_{sp}"], 0.0, "{:+.2f}", True)
    gb = np.load(os.path.join(D, "before", "gs_v142p.npz"))
    ga = np.load(os.path.join(D, "gs_v142p.npz"))
    L += ["", "**B. G_s(E, R_c), cos Z_ref = 1, before | after (ratio after/before - 1)**", "",
          "| primary, species, R_c | 0.1 GeV | 0.3 GeV | 1 GeV | 3 GeV | 10 GeV |", "|---|---|---|---|---|---|"]
    for pr in ("GSF", "H3a"):
        for sp in ("total_numu", "total_nue"):
            for i, rc in enumerate(ga["rc"]):
                c = []
                for E in (0.1, 0.3, 1.0, 3.0, 10.0):
                    b, a = _at(gb["e"], gb[f"{pr}_{sp}"][i], E), _at(ga["e"], ga[f"{pr}_{sp}"][i], E)
                    c.append(f"{b:.4f} -> {a:.4f} ({100 * (a / b - 1):+.1f}%)")
                L.append(f"| {pr} {sp[6:]} {rc:g} GV | " + " | ".join(c) + " |")
    # scan
    sc = {}
    for p in PRIM:
        for mb in (0, 1):
            f = os.path.join(RUN, "out", f"scan_{p}_mb{mb}.json")
            if os.path.exists(f):
                sc[(p, mb)] = json.load(open(f))
    names = {"gsf": "GSF", "h3a": "H3a", "ghn": "GH-native", "ght": "GH-at-total"}
    O = ["a1_rel", "dphi", "fN", "fS", "NS", "WE", "maxmin"]
    oi = {n: i for i, n in enumerate(sc[next(iter(sc))]["obs"])} if sc else {}
    for mb in (0, 1):
        ks = [p for p in PRIM if (p, mb) in sc]
        if not ks:
            continue
        L += ["", f"**C. primary scan, muon_bending = {bool(mb)}; rows: observable of the 12-bin azimuth pattern "
              "(analyse_m3b definitions), cos Z = 0.05 (87 deg); columns: engine primary, Honda**", "",
              "| species, E | observable | " + " | ".join(names[p] for p in ks) + " | Honda |",
              "|---|---|" + "---|" * (len(ks) + 1)]
        for sp in SPEC:
            for E in E_PTS:
                for o in O:
                    key = f"{sp}|0.05|{E}"
                    v = [sc[(p, mb)]["cells"][key]["model"][oi[o]] for p in ks]
                    h = sc[(ks[0], mb)]["cells"][key]["honda"][oi[o]]
                    L.append(f"| {sp[6:]} {E} | {o} | " + " | ".join(f"{x:.3f}" for x in v) + f" | {h:.3f} |")
        L += ["", f"**D. W/E, Phase-1 definition (engine flux at az 270 / az 90, cos Z = 0.05) vs Honda max/min, muon_bending = {bool(mb)}**", "",
              "| species, E | " + " | ".join(names[p] for p in ks) + " | Honda |", "|---|" + "---|" * (len(ks) + 1)]
        for sp in ("total_numu", "total_nue"):
            for E in E_PTS:
                v = [sc[(p, mb)][f"WE_{sp}"][str(E)][0] for p in ks]
                h = sc[(ks[0], mb)][f"WE_{sp}"][str(E)][1]
                L.append(f"| {sp[6:]} {E} | " + " | ".join(f"{x:.3f} ({100 * (x / h - 1):+.0f}%)" for x in v) + f" | {h:.3f} |")
    # m3b cross-check cells
    f = os.path.join(RUN, "out", "m3b_obs_h3a_mb0.npz")
    if os.path.exists(f):
        d = np.load(f)
        names = list(d["obs"])
        sel = ["a1_rel", "dphi", "WE", "maxmin", "fN", "fS", "NS"]
        L += ["", "**E. cross-check cells for tools/mc3d/analyse_m3b: engine (rigidity fixed, H3a at total energy, "
              "muon_bending = False, cz_sub = 3, band-integrated like Honda; analyse_m3b.observables) | Honda**", "",
              "| species | E band | cos Z band | " + " | ".join(sel) + " |", "|---|---|---|" + "---|" * len(sel)]
        for k, sp in enumerate(d["species"]):
            for ie, E in enumerate((0.3, 0.5, 1.0)):
                for iz, cb in enumerate(("0-0.1", "0.1-0.2", "0.2-0.3")):
                    c = [f"{d['engine'][k, ie, iz, names.index(o)]:.3f} | {d['honda'][k, ie, iz, names.index(o)]:.3f}"
                         for o in sel]
                    L.append(f"| {sp} | {E} | {cb} | " + " | ".join(c) + " |")
    txt = "\n".join(L)
    open(out("out/tables_rig.md"), "w").write(txt)
    print(txt)


def main():
    c, a = sys.argv[1], sys.argv[2:]
    if c == "tables":
        return cmd_tables()
    {"chunk": cmd_chunk, "assemble": cmd_assemble, "scan": cmd_scan, "m3b": cmd_m3b}[c](*a)


if __name__ == "__main__":
    main()
