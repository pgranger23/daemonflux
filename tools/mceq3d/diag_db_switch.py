"""2026-11 switch of the engine's MCEq database: verify, compare, re-validate.

The engine default is the v1.4.2 database with the 56.2 GeV hole patched
(``mceq3d_db.py``; ``MCEQ_DB_HOLE.md`` section 10).  Sub-commands (all outputs go
to ``OUT`` = ``$W/runs/dbv142/out``; nothing is written to AFS)::

    verify                 task 1: v1.4.2 hole is the same bin / 0.8 factor; only
                           column 37 of the SIBYLL23D yields + cross section
                           changed (every other dataset bit-identical to the v1.4.2
                           source); sum rule; which models still carry the hole
    flux DB PRIMARY [--model M]
                           1D MCEq fluxes (SIBYLL23D by default) on database DB
                           (an MCEQ3D_DB spec: v140 | v140p | v142 | v142p | path)
    compat                 MCEq 1.4.1 reading the v1.4.2 file: noop == raw bit-wise,
                           non-SIBYLL models unchanged vs v140, isospin check of
                           the SIBYLL neutron yields (the v1.4.2 fix)
    engine DB [--cache-dir D]
                           headline Phase-1 observables in the DELIVERED engine
                           path (hybrid base + GSF + Kamioka + joint cone):
                           hybrid-base 1D, G_s(R_c), H/V (3.1), W/E (3.2),
                           charge-splitting (3.3), N/S + phase (6.3)
    tables                 old | new | new/old-1 | Honda markdown

Run with ``MCEQ3D_DB`` unset (default = v142p) or pass the spec as DB.  See
``MCEQ_DB_HOLE.md`` section 10 for the commands used.
"""

import importlib.util  # noqa: F401  (must precede the MCEq import)
import argparse
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
OUT = os.path.join(W, "runs", "dbv142", "out")
LABELS = ("v140", "v140p", "v142", "v142p")
MODEL = "SIBYLL23D"
J = 37


def _out(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


def _use(db):
    """Select the database for this process (the engine's single switch)."""
    if db:
        os.environ["MCEQ3D_DB"] = db
    import mceq3d_db

    return mceq3d_db.select()


# --------------------------------------------------------------------------
# task 1: verify
# --------------------------------------------------------------------------
def cmd_verify(args):
    import h5py
    import diag_mceq_db_hole as dh
    import mceq3d_db

    src = mceq3d_db.resolve("v142")
    pat = mceq3d_db.resolve("v142p")
    nop = mceq3d_db.resolve(os.path.join(mceq3d_db.DB_DIR,
                                         "mceq_db_lext_dpm193_v142_dbhole_noop.h5"))
    lines = []

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        lines.append(s)

    P("source v1.4.2:", src.path, src.sha256)
    P("patched interp :", pat.path, pat.sha256)
    P("noop           :", nop.path, nop.sha256)
    f0, f1, fn = (h5py.File(x.path, "r") for x in (src, pat, nop))
    eg = f0["common"].attrs["e_grid"]
    S0, D0 = dh.Table(f0, MODEL), dh.Table(f0, dh.REF_MODEL)

    # [1] the hole in the v1.4.2 source: same bin, same factor
    jj = [j for j in range(len(eg)) if 20 < eg[j] < 160]
    hole_bins = set()
    for p in sorted({k[0] for k in S0.keys}):
        eS, eD = dh.energy_fraction(S0, p, eg), dh.energy_fraction(D0, p, eg)
        for j in jj:
            if eD[j] > 0 and eS[j] / eD[j] < 0.9:
                hole_bins.add(j)
    P(f"\n[1] v1.4.2 {MODEL}: bins with >10% sum-rule deficit vs {dh.REF_MODEL} "
      f"(any projectile, 20-160 GeV): {sorted(hole_bins)} = "
      f"{[round(float(eg[j]), 1) for j in sorted(hole_bins)]} GeV")
    r = []
    for k in S0.keys:
        if k in D0.raw:
            s, d = S0.dense(k)[:, J], D0.dense(k)[:, J]
            m = d > 0
            r.extend(s[m] / d[m])
    r = np.array(r)
    P(f"    yield cells in bin {J}: n={len(r)} ratio to {dh.REF_MODEL} "
      f"min {r.min():.12f} max {r.max():.12f}")
    nid = sum(not np.array_equal(S0.dense(k)[:, :J], D0.dense(k)[:, :J])
              for k in S0.keys if k in D0.raw)
    P(f"    tuples with columns j<{J} different from {dh.REF_MODEL}: {nid}")
    cs_s = f0[f"cross_sections/air/{MODEL}"]
    cs_d = f0[f"cross_sections/air/{dh.REF_MODEL}"]
    ps, pd_ = list(cs_s.attrs["parents"]), list(cs_d.attrs["parents"])
    rr = []
    for ip, pid in enumerate(ps):
        if pid in pd_:
            c = pd_.index(pid)
            if cs_d[J, c] > 0:
                rr.append(cs_s[J, ip] / cs_d[J, c])
    P(f"    inelastic cross section, bin {J}, {len(rr)} projectiles: "
      f"ratio min {min(rr):.12f} max {max(rr):.12f}")
    ip = ps.index(2212)
    P("    p-air cross section 44.7/56.2/70.8/89.1 GeV: "
      + "/".join(f"{cs_s[j, ip]:.2f}" for j in range(J - 1, J + 3)) + " mb")

    # [2] bit-identity of everything except the SIBYLL23D column 37
    def compare(fa, fb, name, allow_hole_patch):
        bad, n = [], 0

        def walk(g, path=""):
            nonlocal n
            for key, item in g.items():
                p = f"{path}/{key}"
                if isinstance(item, h5py.Group):
                    walk(item, p)
                else:
                    n += 1
                    if p not in fb:
                        bad.append(("missing", p))
                        continue
                    a, b = item[()], fb[p][()]
                    if a.shape != b.shape or not np.array_equal(a, b):
                        bad.append(("differs", p))
        walk(fa)
        bad_attr = []
        for key in fa.attrs:
            if not np.array_equal(fa.attrs[key], fb.attrs.get(key)):
                bad_attr.append(key)
        P(f"\n[2] {name}: {n} datasets in the source; datasets that differ "
          f"(raw arrays): {[b for b in bad]}; root attrs that differ: {bad_attr}")
        return bad

    bad_p = compare(f0, f1, "interp vs v1.4.2 source", True)
    bad_n = compare(f0, f2 := fn, "noop vs v1.4.2 source", False)
    expect = {("differs", f"/hadronic_interactions/air/{MODEL}"),
              ("differs", f"/hadronic_interactions/air/{MODEL}_indptrs"),
              ("differs", f"/cross_sections/air/{MODEL}")}
    P("    interp: differing datasets are exactly the 3 SIBYLL23D ones:",
      set(bad_p) <= expect and len(bad_p) >= 1, sorted(bad_p))
    P("    noop: only the (re-packed) yield storage differs:",
      set(bad_n) <= expect, sorted(bad_n))
    # decoded content
    S1, Sn = dh.Table(f1, MODEL), dh.Table(f2, MODEL)
    assert S0.keys == S1.keys == Sn.keys
    n_other = n_noop = n_hole = 0
    for k in S0.keys:
        a, b, c = S0.dense(k), S1.dense(k), Sn.dense(k)
        m = np.ones(a.shape[1], bool)
        m[J] = False
        n_other += not np.array_equal(a[:, m], b[:, m])
        n_noop += not np.array_equal(a, c)
        n_hole += not np.array_equal(a[:, J], b[:, J])
    P(f"    decoded yields, {len(S0.keys)} tuples: interp differs outside column "
      f"{J} in {n_other}; noop differs anywhere in {n_noop}; interp changes "
      f"column {J} in {n_hole}")
    csa, csb, csc = (f[f"cross_sections/air/{MODEL}"][:] for f in (f0, f1, fn))
    rows = np.where((csa != csb).any(1))[0]
    P(f"    cross-section rows changed by interp: {rows.tolist()}; "
      f"noop identical: {np.array_equal(csa, csc)}")
    # [3] sum rule
    P(f"\n[3] energy sum rule, column {J}, per projectile (source -> interp; "
      f"neighbours {J-1}/{J+1} in brackets)")
    ok = True
    for p in sorted({k[0] for k in S0.keys}):
        a = dh.energy_fraction(S0, p, eg)
        b = dh.energy_fraction(S1, p, eg)
        if a[J] == 0:
            continue
        good = abs(b[J] / (0.5 * (b[J - 1] + b[J + 1])) - 1) < 0.02
        ok &= good
        P(f"    {p:6d}: {a[J]:.4f} -> {b[J]:.4f}  [{b[J-1]:.4f} / {b[J+1]:.4f}]"
          f"{'' if good else '   <-- outside 2% of neighbour mean'}")
    P("    all projectiles within 2% of the neighbour mean:", ok)
    # [4] models still carrying the hole
    P("\n[4] proton sum rule 44.7/56.2/70.8 GeV, patched file (HOLE = bin 37 < 0.9 x "
      "neighbour mean)")
    for med in ("air", "hydrogen"):
        for m in sorted(x for x in f1[f"hadronic_interactions/{med}"]
                        if "indptrs" not in x):
            t = dh.Table(f1, m, med)
            e = dh.energy_fraction(t, 2212, eg)
            flag = "HOLE" if e[J] < 0.9 * 0.5 * (e[J - 1] + e[J + 1]) else "ok"
            P(f"    {med:8s} {m:22s} {e[J-1]:.3f} {e[J]:.3f} {e[J+1]:.3f}  {flag}")
    for f in (f0, f1, fn):
        f.close()
    with open(_out("verify_v142.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# 1D fluxes
# --------------------------------------------------------------------------
def cmd_flux(args):
    import diag_mceq_db_hole as dh

    s = _use(args.db)
    import MCEq.config as cfg
    from mceq3d_db import make_run

    cfg.e_min = 0.1
    cfg.debug_level = 0
    cls, arg = dh._primary(args.primary)
    t0 = time.time()
    mc = make_run(interaction_model=args.model, primary_model=(cls, arg),
                  theta_deg=0.0)
    out = {"e": np.array(mc.e_grid)}
    for cz in dh.COSZ:
        mc.set_theta_deg(float(np.degrees(np.arccos(cz))))
        mc.solve()
        for sp in dh.SPECIES:
            out[f"{sp}_cz{cz}"] = np.array(mc.get_solution("total_" + sp, 0))
    lab = args.label or s.label
    fn = _out(f"flux_{lab}_{args.model}_{args.primary}.npz")
    np.savez(fn, **out)
    print(f"{fn} ({time.time() - t0:.0f} s) db={s.path}")


# --------------------------------------------------------------------------
# compat
# --------------------------------------------------------------------------
def cmd_compat(args):
    import h5py
    import diag_mceq_db_hole as dh
    import mceq3d_db

    lines = []

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        lines.append(s)

    def mx(a, b):
        a, b = np.load(_out(a)), np.load(_out(b))
        sel = (a["e"] >= 0.1) & (a["e"] <= 1e3)
        w, where = 0.0, ""
        for k in a.files:
            if k == "e":
                continue
            ok = sel & (a[k] > 0)
            d = np.abs(b[k][ok] / a[k][ok] - 1)
            if d.size and d.max() >= w:
                w, where = d.max(), f"{k} at {a['e'][ok][d.argmax()]:.3g} GeV"
        return w, where

    P("MCEq 1.4.1 reading the v1.4.2 file (MCEq 1.4.1 -> 1.4.2 code diff is "
      "config.py only: file name, checksum, old-db name)")
    for pr in ("H3a", "GSF"):
        if os.path.exists(_out(f"flux_v142_{MODEL}_{pr}.npz")):
            P(f"  {pr}: max |v142p/v142 - 1| = %.3e (%s)"
              % mx(f"flux_v142_{MODEL}_{pr}.npz", f"flux_v142p_{MODEL}_{pr}.npz"))
        if os.path.exists(_out(f"flux_noop_{MODEL}_{pr}.npz")):
            P(f"  {pr}: max |noop/v142 - 1| = %.3e (%s)"
              % mx(f"flux_v142_{MODEL}_{pr}.npz", f"flux_noop_{MODEL}_{pr}.npz"))
    for m in ("EPOSLHC", "DPMJETIII193", "QGSJETII04"):
        fa, fb = f"flux_v140_{m}_H3a.npz", f"flux_v142_{m}_H3a.npz"
        if os.path.exists(_out(fa)) and os.path.exists(_out(fb)):
            P(f"  {m} H3a: v142/v140 max |ratio - 1| = %.3e (%s)" % mx(fa, fb))
    # isospin check: n -> pi+ should mirror p -> pi- in a fixed file
    P("\nisospin check on the SIBYLL23D tables (secondaries per interaction, "
      "N = M E_i/E_j summed over i; pi+/pi- multiplicity ratio):")
    for lab in ("v140", "v142"):
        s = mceq3d_db.resolve(lab)
        with h5py.File(s.path, "r") as f:
            eg = f["common"].attrs["e_grid"]
            T = dh.Table(f, MODEL)
            xr = eg[:, None] / eg[None, :]
            for j in (45, 55):  # ~ 178 GeV, ~ 1.1 TeV
                row = {}
                for par in (2212, 2112, -2212, -2112):
                    n = {c: (T.dense((par, 0, c, 0)) * xr)[:, j].sum()
                         if (par, 0, c, 0) in T.raw else 0.0 for c in (211, -211)}
                    row[par] = n[211] / n[-211]
                P(f"  {lab} E_kin={eg[j]:.0f} GeV pi+/pi-: p {row[2212]:.3f}  "
                  f"n {row[2112]:.3f}  pbar {row[-2212]:.3f}  nbar {row[-2112]:.3f}"
                  f"   (isospin: n ~ 1/p = {1 / row[2212]:.3f})")
    with open(_out("compat.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# engine observables (delivered path)
# --------------------------------------------------------------------------
def _honda_refs():
    h = dict(np.load(os.path.join(HERE, "honda_kam.npz")))
    bt = dict(np.load(os.path.join(HERE, "bartol_kam.npz")))
    return h, bt


def cmd_engine(args):
    warnings.simplefilter("always")
    s = _use(args.db)
    import diag_joint_stages as js
    import diag_ew_charge_fourier as fz
    from mceq3d_flux import MCEq3DFlux

    cache = args.cache_dir
    res = {"db": s.label, "sha256": s.sha256, "tag": s.tag}
    t0 = time.time()
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        eng = MCEq3DFlux(base_model="hybrid", primary=("GlobalSplineFitBeta", None),
                         daemonflux_location="kamioka")
        e = eng.e
        # ---- (a) hybrid base 1D, and its MCEq half ----
        czb = np.array([1.0, 0.5, 0.1])
        hyb = eng.base(czb)
        eng.base_model = "mceq"
        raw = eng.base(czb)
        eng.base_model = "hybrid"
        np.savez(_out(f"base_{s.label}.npz"), e=e, cz=czb,
                 **{f"hyb_{k}": v for k, v in hyb.items()},
                 **{f"mceq_{k}": v for k, v in raw.items()})
        # ---- (b) G_s response (GSF delivered primary + H3a) ----
        rc = np.array([7.0, 11.0, 40.0])
        gs = {}
        for pname, eg_ in (("GSF", eng), ("H3a", None)):
            if eg_ is None:
                eg_ = MCEq3DFlux(base_model="mceq")
            G, _ = eg_.geomag_response(rc, cz_ref=1.0, cache_dir=None)
            for sp, v in G.items():
                gs[f"{pname}_{sp}"] = v
        np.savez(_out(f"gs_{s.label}.npz"), e=e, rc=rc, **gs)
        print(f"base + G_s done {time.time() - t0:.0f} s", flush=True)

        # ---- (c) 3.1 / 3.2 : diag_joint_stages S4 path ----
        h, bt = _honda_refs()
        He, Hcz = h["E"], h["czlo"]
        cz, az = np.array(js.CZ), np.array(js.AZ)
        ie, iw = int(np.argmin(abs(az - 90.0))), int(np.argmin(abs(az - 270.0)))
        r = eng.solve(js.LAT, js.LON, cz, az, use_cache=True, cache_dir=cache,
                      date=js.DATE, n_jobs=None, **js.STAGES["S4"])
        print(f"S4 solve done {time.time() - t0:.0f} s", flush=True)
        ee = r["e"]
        for fl, sp in js.FLAV:
            f = r["flux"][sp]
            res[f"hv_{fl}"] = [js.log_at(f[0].mean(0), ee, E) / js.log_at(
                f[-1].mean(0), ee, E) for E in js.HV_E]
            for E in js.EW_E:
                res[f"we_{fl}_{E}"] = [float(np.interp(
                    E, ee, f[i, iw] / np.maximum(f[i, ie], 1e-300)))
                    for i in range(len(js.EW_CZ))]
            nm = h[fl]
            ihz = int(np.argmin(abs(Hcz - 0.0)))
            ivz = int(np.argmin(abs(Hcz - 0.9)))
            res[f"hv_{fl}_honda"] = [js.log_at(nm[ihz].mean(0), He, E) / js.log_at(
                nm[ivz].mean(0), He, E) for E in js.HV_E]
            key = "num" if fl == "numu" else "nue"
            y = 0.5 * (bt[f"{key}_fmin"] + bt[f"{key}_fmax"])
            ih = int(np.argmin(abs(bt["cz"] - 0.05)))
            iv = int(np.argmin(abs(bt["cz"] - 0.95)))
            res[f"hv_{fl}_bartol"] = [js.log_at(y[ih], bt["E"], E) / js.log_at(
                y[iv], bt["E"], E) for E in js.HV_E]
            for E in js.EW_E:
                res[f"we_{fl}_{E}_honda"] = [float(np.interp(
                    E, He, nm[int(np.argmin(abs(Hcz - c)))].max(0)
                    / nm[int(np.argmin(abs(Hcz - c)))].min(0))) for c in js.EW_CZ]

        # ---- (d) 3.3 / 6.3 : charge-fourier path on Honda's bins ----
        ew_axis = fz.geomagnetic_ew_axis()
        e2, F, az12 = fz.model_flux(muon_bending=True, cz_sub=1, cache_dir=cache,
                                    engine=eng)
        print(f"fourier solve done {time.time() - t0:.0f} s", flush=True)
        obs = {}
        hobs = {}
        for icz, czc in enumerate(fz.CZ_BANDS):
            for sp in fz.SPECIES:
                for E in fz.E_POINTS:
                    f = np.array([fz._log_at(F[sp][icz, j], e2, E)
                                  for j in range(F[sp].shape[1])])
                    obs[(sp, czc, E)] = fz.ew_observables(f, az12, ew_axis)
                    hobs[(sp, czc, E)] = fz.honda_ew(h, fz.HKEY[sp], czc, E, ew_axis)
        for czc in (0.05,):
            for fl, a, b in fz.PAIRS:
                for E in (0.3, 0.5):
                    for nm_, o in (("model", obs), ("honda", hobs)):
                        res[f"split_{fl}_{E}_{nm_}_dphi"] = float(fz._wrap180(
                            o[(a, czc, E)]["dphi"] - o[(b, czc, E)]["dphi"]))
                        res[f"split_{fl}_{E}_{nm_}_maxmin"] = float(
                            o[(a, czc, E)]["maxmin"] - o[(b, czc, E)]["maxmin"])
        # N/S model/Honda (bins centred 345,15 = N; 165,195 = S) and phase offset
        az_h = fz.honda_azimuth_to_compass(h["azlo"].astype(float) + 15.0)
        icz = list(fz.CZ_BANDS).index(0.05)
        icz_h = int(np.argmin(np.abs(h["czlo"] - 0.0)))
        for sp in fz.SPECIES:
            E = 0.5
            fm = np.array([fz._log_at(F[sp][icz, j], e2, E) for j in range(12)])
            rows = h[fz.HKEY[sp]][icz_h]
            fh_ = np.array([fz._log_at(rows[j], h["E"], E) for j in range(12)])
            # put Honda on our compass grid
            fh_c = np.array([fh_[int(np.argmin(np.abs(
                (az_h - a + 180) % 360 - 180)))] for a in az12])
            def pick(v, centres):
                return np.mean([v[int(np.argmin(np.abs(az12 - c)))] for c in centres])
            for nm_, cen in (("N", (15.0, 345.0)), ("S", (165.0, 195.0))):
                res[f"{nm_}_{sp}"] = float(pick(fm, cen) / pick(fh_c, cen))
            res[f"phase_{sp}"] = float(fz._wrap180(
                obs[(sp, 0.05, E)]["phi1"] - hobs[(sp, 0.05, E)]["phi1"]))
        # all D(dphi), max/min raw model values for the record
        res["fourier"] = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in obs.items()}
    res["warnings"] = sorted({f"{w.category.__name__}: {str(w.message)[:160]}"
                              for w in wl})
    res["seconds"] = time.time() - t0
    fn = _out(f"engine_{s.label}.json")
    with open(fn, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("wrote", fn, f"({res['seconds']:.0f} s)")


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------
def _pct(a, b):
    return f"{100 * (b / a - 1):+.2f}%"


def _at(e, y, e0):
    return float(np.interp(np.log(e0), np.log(e), y))


def aux_tables(lab):
    """Hybrid-base 1D flux and G_s response tables (old | new | new/old-1)."""
    L = []
    EP = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
    old = lab[0]
    B = {k: np.load(_out(f"base_{k}.npz")) for k in lab
         if os.path.exists(_out(f"base_{k}.npz"))}
    if old in B:
        e = B[old]["e"]
        for kind, nm_ in (("hyb", "hybrid base"), ("mceq", "MCEq half of the base")):
            for ic, cz in enumerate(B[old]["cz"]):
                L += [f"\n**{nm_}, 1D flux, cos Z = {cz:g}: ratio to {old}** "
                      f"(GSF primary, SIBYLL23D; hybrid = (1-w) MCEq + w daemonflux[kamioka])\n",
                      "| species | variant | " + " | ".join(f"{x:g} GeV" for x in EP) + " |",
                      "|---|---|" + "---|" * len(EP)]
                for sp in ("total_numu", "total_antinumu", "total_nue", "total_antinue"):
                    for k in lab[1:]:
                        if k in B:
                            r = B[k][f"{kind}_{sp}"][ic] / B[old][f"{kind}_{sp}"][ic]
                            L.append(f"| {sp[6:]} | {k}/{old} | " + " | ".join(
                                f"{100 * (_at(e, r, x) - 1):+.2f}%" for x in EP) + " |")
    G = {k: np.load(_out(f"gs_{k}.npz")) for k in lab
         if os.path.exists(_out(f"gs_{k}.npz"))}
    if old in G:
        e, rc = G[old]["e"], G[old]["rc"]
        for pr in ("GSF", "H3a"):
            for sp in ("total_numu", "total_nue"):
                L += [f"\n**G_s ({pr}, {sp[6:]}, cz_ref = 1): old value, then ratio to {old}**\n",
                      "| R_c [GV] | variant | " + " | ".join(f"{x:g} GeV" for x in EP[:-2]) + " |",
                      "|---|---|" + "---|" * len(EP[:-2])]
                for i, r_c in enumerate(rc):
                    g0 = G[old][f"{pr}_{sp}"][i]
                    L.append(f"| {r_c:g} | {old} (value) | " + " | ".join(
                        f"{_at(e, g0, x):.4f}" for x in EP[:-2]) + " |")
                    for k in lab[1:]:
                        if k in G:
                            r = G[k][f"{pr}_{sp}"][i] / np.where(g0 > 0, g0, 1)
                            L.append(f"| {r_c:g} | {k}/{old}-1 | " + " | ".join(
                                f"{100 * (_at(e, r, x) - 1):+.2f}%" for x in EP[:-2]) + " |")
                L += ["", "W/E-type double ratio G_s(7 GV)/G_s(40 GV), ratio to " + old + ": "
                      + "; ".join(f"{k}: " + ", ".join(
                          f"{x:g} GeV {100 * (_at(e, (G[k][f'{pr}_{sp}'][0] / np.maximum(G[k][f'{pr}_{sp}'][2], 1e-300)) / (G[old][f'{pr}_{sp}'][0] / np.maximum(G[old][f'{pr}_{sp}'][2], 1e-300)), x) - 1):+.2f}%"
                          for x in (0.3, 1.0, 3.0)) for k in lab[1:] if k in G)]
    return L


def cmd_tables(args):
    lab = args.labels
    E = {k: json.load(open(_out(f"engine_{k}.json"))) for k in lab
         if os.path.exists(_out(f"engine_{k}.json"))}
    L = aux_tables(lab)
    old, new = lab[0], lab[-1]

    def row(name, vals, honda=None, fmt="{:.3f}"):
        cells = [fmt.format(v) for v in vals]
        d = [_pct(vals[0], v) for v in vals[1:]]
        L.append(f"| {name} | " + " | ".join(cells) + " | " + " | ".join(d) + " | "
                 + (fmt.format(honda) if honda is not None else "") + " |")

    hdr = ("| observable | " + " | ".join(lab) + " | "
           + " | ".join(f"{x}/{old}-1" for x in lab[1:]) + " | Honda |")
    sep = "|" + "---|" * (2 * len(lab) + 1)
    HV_E = (0.3, 0.5, 1.0, 3.0)
    for fl in ("numu", "nue"):
        L += [f"\n**3.1 H/V, {fl}** (azimuth-averaged horizon/vertical)\n", hdr, sep]
        for i, e in enumerate(HV_E):
            v = [E[k][f"hv_{fl}"][i] for k in lab]
            row(f"H/V {e} GeV", v, E[old][f"hv_{fl}_honda"][i])
        L += ["", "model/Honda (Bartol): " + "; ".join(
            f"{e} GeV " + " ".join(f"{E[k][f'hv_{fl}'][i] / E[old][f'hv_{fl}_honda'][i]:.3f}"
                                    f" ({E[k][f'hv_{fl}'][i] / E[old][f'hv_{fl}_bartol'][i]:.3f})"
                                    for k in lab)
            for i, e in enumerate(HV_E))]
    for fl in ("numu", "nue"):
        L += [f"\n**3.2 W/E, {fl}** (87 / 81 / 75 deg, W/E of the engine; Honda = max/min)\n",
              hdr, sep]
        for e in (0.5, 1.0):
            for i, z in enumerate((87, 81, 75)):
                v = [E[k][f"we_{fl}_{e}"][i] for k in lab]
                row(f"W/E {e} GeV {z} deg", v, E[old][f"we_{fl}_{e}_honda"][i],
                    "{:.3f}")
                L.append(f"|   (vs Honda) | " + " | ".join(
                    f"{x / E[old][f'we_{fl}_{e}_honda'][i] - 1:+.1%}" for x in v)
                    + " |" + " |" * (2 * len(lab) - len(lab)) )
    L += ["\n**3.3 charge splitting, 87 deg** (nu - nubar)\n", hdr, sep]
    for fl in ("numu", "nue"):
        for e in (0.3, 0.5):
            for q, nm_ in (("dphi", "D(dphi) [deg]"), ("maxmin", "D(max/min)")):
                v = [E[k][f"split_{fl}_{e}_model_{q}"] for k in lab]
                h_ = E[old][f"split_{fl}_{e}_honda_{q}"]
                cells = [f"{x:+.3f}" for x in v]
                dl = [f"{x - v[0]:+.3f}" if True else "" for x in v[1:]]
                L.append(f"| {fl}-anti {e} GeV {nm_} | " + " | ".join(cells)
                         + " | " + " | ".join(dl) + f" | {h_:+.3f} |")
    L += ["\n(for 3.3 the 'x/old-1' columns are ABSOLUTE differences new - old: the "
          "observables are themselves differences and pass through zero)\n"]
    L += ["\n**6.3 N/S model/Honda and first-harmonic phase offset, 0.5 GeV, cosZ 0.05**\n",
          hdr, sep]
    for sp in ("total_numu", "total_antinumu", "total_nue", "total_antinue"):
        for q in ("N", "S"):
            v = [E[k][f"{q}_{sp}"] for k in lab]
            row(f"{q} {sp[6:]}", v, 1.0)
        v = [E[k][f"phase_{sp}"] for k in lab]
        L.append(f"| phase offset {sp[6:]} [deg] | " + " | ".join(f"{x:+.2f}" for x in v)
                 + " | " + " | ".join(f"{x - v[0]:+.2f}" for x in v[1:]) + " | 0 |")
    L.append("\n(phase rows: differences are absolute, degrees)\n")
    txt = "\n".join(L)
    with open(_out("tables_engine.md"), "w") as fh:
        fh.write(txt)
    print(txt)


def cmd_doc(args):
    """Compact old | new | new/old-1 | Honda tables with the hole / v1.4.2 split
    (the tables quoted in MCEQ_DB_HOLE.md section 10.5)."""
    lab = ("v140", "v140p", "v142p")
    E = {k: json.load(open(_out(f"engine_{k}.json"))) for k in lab}
    L = []

    def line(name, v, honda=None, fmt="{:.3f}", absd=False, hon_fmt=None):
        o, h, n = v
        if absd:
            tot, hole, fix = n - o, h - o, n - h
            d = f"{tot:+.3f} | {hole:+.3f} | {fix:+.3f}"
        else:
            d = (f"{100 * (n / o - 1):+.2f}% | {100 * (h / o - 1):+.2f}% | "
                 f"{100 * (n / h - 1):+.2f}%")
        hs = "" if honda is None else (hon_fmt or fmt).format(honda)
        L.append(f"| {name} | {fmt.format(o)} | {fmt.format(n)} | {d} | {hs} |")

    def head(title):
        L.extend(["", f"**{title}**", "",
                  "| observable | old (v140) | new (v142p) | new/old - 1 | of which hole patch (v140p/v140) "
                  "| of which v1.4.2 fixes (v142p/v140p) | Honda |", "|---|---|---|---|---|---|---|"])

    for fl in ("numu", "nue"):
        head(f"3.1 H/V {fl} (azimuth-averaged horizon/vertical)")
        for i, e in enumerate((0.3, 0.5, 1.0, 3.0)):
            line(f"{e} GeV", [E[k][f"hv_{fl}"][i] for k in lab],
                 E["v140"][f"hv_{fl}_honda"][i])
    for fl in ("numu", "nue"):
        head(f"3.2 W/E {fl} (engine W/E at the E/W azimuths; Honda: max/min over azimuth)")
        for e in (0.5, 1.0):
            for i, z in enumerate((87, 81, 75)):
                line(f"{e} GeV, {z} deg", [E[k][f"we_{fl}_{e}"][i] for k in lab],
                     E["v140"][f"we_{fl}_{e}_honda"][i])
    L.extend(["", "**3.3 charge splitting, 87 deg (nu - nubar); the 'new/old' columns are absolute differences**", "",
              "| observable | old (v140) | new (v142p) | new - old | hole patch | v1.4.2 fixes | Honda |",
              "|---|---|---|---|---|---|---|"])
    for fl in ("numu", "nue"):
        for e in (0.3, 0.5):
            for q, nm_ in (("dphi", "D(dphi) [deg]"), ("maxmin", "D(max/min)")):
                line(f"{fl}-anti {e} GeV {nm_}",
                     [E[k][f"split_{fl}_{e}_model_{q}"] for k in lab],
                     E["v140"][f"split_{fl}_{e}_honda_{q}"], "{:+.3f}", absd=True)
    L.extend(["", "**6.3 N/S model/Honda and first-harmonic phase offset (deg), 0.5 GeV, cos Z 0.05**", "",
              "| observable | old (v140) | new (v142p) | new/old - 1 | hole patch | v1.4.2 fixes | Honda |",
              "|---|---|---|---|---|---|---|"])
    for sp in ("total_numu", "total_antinumu", "total_nue", "total_antinue"):
        for q in ("N", "S"):
            line(f"{q} {sp[6:]}", [E[k][f"{q}_{sp}"] for k in lab], 1.0)
        line(f"phase {sp[6:]} [deg] (abs. diff.)", [E[k][f"phase_{sp}"] for k in lab], 0.0,
             "{:+.2f}", absd=True)
    # hybrid base and G_s
    B = {k: np.load(_out(f"base_{k}.npz")) for k in lab}
    e = B["v140"]["e"]
    EP = (0.1, 0.3, 1.0, 3.0, 10.0)
    for kind, nm_ in (("hyb", "hybrid base"), ("mceq", "MCEq half of the hybrid base")):
        L.extend(["", f"**{nm_}, 1D flux, cos Z = 1, GSF, SIBYLL23D: new/old - 1 "
                  "(total | hole patch | v1.4.2 fixes)**", "",
                  "| species | " + " | ".join(f"{x:g} GeV" for x in EP) + " |",
                  "|---|" + "---|" * len(EP)])
        for sp in ("total_numu", "total_antinumu", "total_nue", "total_antinue"):
            c = {k: B[k][f"{kind}_{sp}"][0] for k in lab}
            cells = []
            for x in EP:
                o, h, n = (_at(e, c[k], x) for k in lab)
                cells.append(f"{100 * (n / o - 1):+.2f} / {100 * (h / o - 1):+.2f} / "
                             f"{100 * (n / h - 1):+.2f}%")
            L.append(f"| {sp[6:]} | " + " | ".join(cells) + " |")
    G = {k: np.load(_out(f"gs_{k}.npz")) for k in lab}
    eg, rc = G["v140"]["e"], G["v140"]["rc"]
    for pr in ("GSF", "H3a"):
        L.extend(["", f"**G_s response, {pr} primary, cos Z_ref = 1: old value | new/old - 1 "
                  "(total / hole patch / v1.4.2 fixes)**", "",
                  "| species, R_c | " + " | ".join(f"{x:g} GeV" for x in EP[:4]) + " |",
                  "|---|" + "---|" * 4])
        for sp in ("total_numu", "total_nue"):
            for i, r_c in enumerate(rc):
                cells = []
                for x in EP[:4]:
                    o, h, n = (_at(eg, G[k][f"{pr}_{sp}"][i], x) for k in lab)
                    cells.append(f"{o:.4f}: {100 * (n / o - 1):+.2f} / {100 * (h / o - 1):+.2f} / "
                                 f"{100 * (n / h - 1):+.2f}%")
                L.append(f"| {sp[6:]}, {r_c:g} GV | " + " | ".join(cells) + " |")
        for sp in ("total_numu", "total_nue"):
            r = {k: G[k][f"{pr}_{sp}"][0] / np.maximum(G[k][f"{pr}_{sp}"][2], 1e-300)
                 for k in lab}
            L.append("")
            L.append(f"G_s(7 GV)/G_s(40 GV), {sp[6:]}, {pr}: new/old - 1 = " + ", ".join(
                f"{x:g} GeV {100 * (_at(eg, r['v142p'], x) / _at(eg, r['v140'], x) - 1):+.2f}% "
                f"(hole {100 * (_at(eg, r['v140p'], x) / _at(eg, r['v140'], x) - 1):+.2f}%)"
                for x in (0.3, 1.0, 3.0)))
    txt = "\n".join(L)
    with open(_out("tables_doc.md"), "w") as fh:
        fh.write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("verify").set_defaults(fn=cmd_verify)
    q = sp.add_parser("flux")
    q.add_argument("db")
    q.add_argument("primary", choices=["H3a", "GSF"])
    q.add_argument("--model", default=MODEL)
    q.add_argument("--label", default=None)
    q.set_defaults(fn=cmd_flux)
    sp.add_parser("compat").set_defaults(fn=cmd_compat)
    q = sp.add_parser("engine")
    q.add_argument("db", nargs="?", default=None)
    q.add_argument("--cache-dir", default=os.path.join(HERE, ".cache3d"))
    q.set_defaults(fn=cmd_engine)
    sp.add_parser("doc").set_defaults(fn=cmd_doc)
    q = sp.add_parser("tables")
    q.add_argument("--labels", nargs="+", default=list(LABELS))
    q.set_defaults(fn=cmd_tables)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
