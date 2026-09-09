"""Interaction-model / primary-model / geomagnetic-epoch scan of the residuals.

Question this answers: how much of what is left between this engine and Honda's
HKKM2014 Kamioka tables is a **different input** (hadronic model, primary
spectrum, geomagnetic epoch) rather than one of our own approximations?  The two
residuals under test are

(a) the ``nu_mu`` West/East overshoot at 87 deg growing with energy
    (+12/+21/+26 % at 0.5/1/2 GeV, ``validate_ew_zenith.py``), where the
    production cone is small so the cutoff values and the per-species
    suppression response ``G_s`` dominate; and
(b) the **charge-ratio** component of the East-West species pattern -- with
    ``muon_bending`` switched off the model's four species W/E collapse onto
    ~2.9-3.1 while Honda's total pattern is 2.51/3.81/4.74/2.12.

Because the base flux is azimuth-independent, every W/E number here is a pure
``G_s`` x cone observable: changing the interaction model or the primary can
only move it through the *response*.  ``H/V`` does see the base as well.

What is and is not varied.  ``MCEq3DFlux(interaction_model=..., primary=...)``
sets both the MCEq base (the sub-1.7-GeV half of the hybrid base) and ``G_s``;
the ``G_s`` disk cache is keyed on that model identity (``_gs_tag``), so nothing
is ever silently reused across models.  The joint-cone *production profile*
(``joint_prod``) is fixed SIBYLL-2.3d/H3a by ``offaxis_mc.TAG`` independently of
this engine -- E_off is a primary-insensitive ratio -- so the cone geometry is
held constant across the scan by construction.

Sub-commands
------------
``build``    build (and cache) the ``G_s`` band of one model at one ``cz_ref``.
             This is the only expensive step: 41 MCEq solves, ~135 s each at
             87 deg.  Run one process per (model, band) in parallel.
``measure``  with warm caches, solve the delivered engine for one model and dump
             every observable to a JSON file.
``gsresp``   per-model charge-ratio response ``G_nue/G_antinue`` and
             ``G_numu/G_antinumu`` read straight off the cached ``G_s``.
``epoch``    IGRF-epoch spread of the back-traced cutoff at the East horizon.
``table``    aggregate the JSON dumps into one table per observable.

Run from ``tools/mceq3d`` with a warm ``.cache3d``.
"""

from __future__ import annotations

import argparse
import importlib.util  # noqa: F401  (mceq_config import shim; must precede MCEq)
import json
import os
import time
import warnings
from datetime import datetime

import numpy as np

LAT, LON = 36.43, 137.31
DATE = datetime(2020, 1, 1)
CACHE = ".cache3d"
RC_GRID = np.linspace(0.1, 55.0, 40)  # == solve(use_cache=True)'s own grid
CZ = (0.05, 0.15, 0.95)               # 87 / 81 / 18 deg
AZ = tuple(np.arange(8) * 45.0)       # includes 90 (E) and 270 (W)
EW_E = (0.5, 1.0, 2.0)
HV_E = (0.3, 0.5, 1.0)
SP = ("total_numu", "total_antinumu", "total_nue", "total_antinue")

#: primary-model shorthands -> ``crflux.models`` (class name, tag)
PRIMARIES = {
    "GSF": ("GlobalSplineFitBeta", None),
    "H3a": ("HillasGaisser2012", "H3a"),
    # Gaisser-Honda (2002) -- the closest thing crflux carries to Honda's own
    # primary parametrisation.  HKKM2014's actual fit (Honda et al. 2011/2015,
    # refitted to AMS-02/BESS/PAMELA) is NOT distributed with crflux.
    "GH": ("GaisserHonda", None),
}


#: the delivered configuration: base, E_off table and joint-cone production
#: profile stay here in ``response_only`` mode
BASE_MODEL = "SIBYLL23D"
BASE_PRIMARY = "GSF"


def cfg_name(model, prim):
    return f"{model}__{prim}"


def make_engine(model, prim, base_model="hybrid", response_only=False):
    """The scan engine for one (interaction model, primary) point.

    ``response_only=False`` varies the WHOLE engine: base and ``G_s`` both move
    to ``model``/``prim``.  That is the honest "what if we had used this model"
    configuration, but ``offaxis_factor``'s table guard only admits the
    interaction model the ``E_off`` table was built with (SIBYLL-2.3d), so it is
    usable for primary variations only.

    ``response_only=True`` keeps the base, the ``E_off`` table and the joint-cone
    production profile at the delivered SIBYLL-2.3d/GSF and moves ONLY the
    geomagnetic suppression response ``G_s`` to ``model``/``prim``, through the
    ``gs_interaction_model`` / ``gs_primary`` override.  Every W/E number is a
    pure ``G_s`` observable anyway (the base is azimuth-independent), so this is
    the like-for-like comparison, and it isolates the response from the base
    normalisation.  The cached ``G_s`` bands are shared between the two paths --
    ``_gs_tag`` is built with the same format string as ``_tag``.
    """
    from mceq3d_flux import MCEq3DFlux

    if response_only:
        return MCEq3DFlux(
            interaction_model=BASE_MODEL,
            primary=PRIMARIES[BASE_PRIMARY],
            base_model=base_model,
            daemonflux_location="kamioka",
            gs_interaction_model=model,
            gs_primary=PRIMARIES[prim],
        )
    return MCEq3DFlux(
        interaction_model=model,
        primary=PRIMARIES[prim],
        base_model=base_model,
        daemonflux_location="kamioka",
    )


# ----------------------------------------------------------------- build ----
def cmd_build(args):
    eng = make_engine(args.model, args.primary,
                      response_only=getattr(args, "response_only", False))
    for cz in [float(c) for c in args.cz.split(",")]:
        t0 = time.time()
        eng.geomag_response(RC_GRID, cz_ref=max(abs(cz), 1e-3), cache_dir=CACHE)
        print(f"G_s {args.model}/{args.primary} cz={cz}: {time.time() - t0:.0f}s",
              flush=True)
    print("BUILD_DONE", flush=True)


# --------------------------------------------------------------- measure ----
def _log_at(y, x, X):
    return float(np.exp(np.interp(np.log(X), np.log(x),
                                  np.log(np.maximum(y, 1e-300)))))


def cmd_measure(args):
    warnings.filterwarnings("ignore")
    eng = make_engine(args.model, args.primary,
                      response_only=args.response_only)
    cz, az = np.array(CZ), np.array(AZ)
    ie = int(np.argmin(abs(az - 90.0)))
    iw = int(np.argmin(abs(az - 270.0)))
    out = {"model": args.model, "primary": args.primary,
           "response_only": bool(args.response_only)}
    for bend, key in ((True, "bend_on"), (False, "bend_off")):
        t0 = time.time()
        r = eng.solve(LAT, LON, cz, az, use_cache=True, cache_dir=CACHE,
                      date=DATE, n_jobs=args.n_jobs, muon_bending=bend)
        e = r["e"]
        blk = {"cost_s": time.time() - t0}
        for sp in SP:
            f = r["flux"][sp]
            blk[f"we_{sp}"] = {
                f"{np.degrees(np.arccos(c)):.0f}": [
                    float(np.interp(E, e, f[i, iw]
                                    / np.maximum(f[i, ie], 1e-300)))
                    for E in EW_E]
                for i, c in enumerate(CZ[:2])}
            blk[f"hv_{sp}"] = [
                _log_at(f[0].mean(0), e, E) / _log_at(f[2].mean(0), e, E)
                for E in HV_E]
        out[key] = blk
        print(f"  {key} done in {blk['cost_s']:.0f}s", flush=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("MEASURE_DONE", flush=True)


# ---------------------------------------------------------------- gsresp ----
def cmd_gsresp(args):
    """Charge-ratio response of G_s itself: G_nue/G_antinue, G_numu/G_antinumu."""
    eng = make_engine(args.model, args.primary, base_model="mceq")
    rows = {}
    for cz in [float(c) for c in args.cz.split(",")]:
        G, rcg = eng.geomag_response(RC_GRID, cz_ref=max(abs(cz), 1e-3),
                                     cache_dir=CACHE)
        e = eng.e
        row = {}
        for rc in (7.0, 20.0, 42.0):
            j = int(np.argmin(abs(rcg - rc)))
            g = {sp: float(np.interp(args.energy, e, G[sp][j])) for sp in SP}
            row[f"{rcg[j]:.2f}"] = {
                "G_numu": g["total_numu"], "G_antinumu": g["total_antinumu"],
                "G_nue": g["total_nue"], "G_antinue": g["total_antinue"],
                "r_mu": g["total_numu"] / g["total_antinumu"],
                "r_e": g["total_nue"] / g["total_antinue"],
            }
        rows[f"{cz:g}"] = row
    res = {"model": args.model, "primary": args.primary,
           "energy": args.energy, "rows": rows}
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))
    print("GSRESP_DONE", flush=True)


# ----------------------------------------------------------------- epoch ----
def cmd_epoch(args):
    """Back-traced cutoff at a few directions for several IGRF epochs.

    The engine's own default epoch is 2020-01-01 (``validate_ew_zenith`` /
    ``validate_honda``).  Honda's HKKM2014 Kamioka tables are computed with the
    IGRF field of their own reference year; this measures how much R_c -- and
    therefore the East-West contrast -- moves across a 20-year epoch span.
    """
    import geomag_backtrace as gb

    zeniths = np.array([float(z) for z in args.zeniths.split(",")])
    azimuths = np.array([float(a) for a in args.azimuths.split(",")])
    years = [int(y) for y in args.years.split(",")]
    res = {}
    for y in years:
        t0 = time.time()
        m = gb.cutoff_map(LAT, LON, datetime(y, 1, 1), zeniths, azimuths,
                          n_jobs=args.n_jobs, warn_saturated=False)
        res[str(y)] = np.asarray(m, float).tolist()
        print(f"epoch {y}: {time.time() - t0:.0f}s", flush=True)
    print(f"\n{'zen':>5} {'az':>6} " + "".join(f"{y:>9}" for y in years)
          + f"{'spread%':>10}")
    ref = np.array(res[str(years[0])])
    for i, z in enumerate(zeniths):
        for j, a in enumerate(azimuths):
            v = [res[str(y)][i][j] for y in years]
            sp = (max(v) - min(v)) / max(np.mean(v), 1e-9) * 100
            print(f"{z:>5.0f} {a:>6.0f} " + "".join(f"{x:>9.2f}" for x in v)
                  + f"{sp:>9.1f}%")
    # East-West contrast per epoch (the observable that matters)
    ie = int(np.argmin(abs(azimuths - 90.0)))
    iw = int(np.argmin(abs(azimuths - 270.0)))
    print(f"\n{'zen':>5} " + "".join(f"{'E/W ' + str(y):>12}" for y in years))
    for i, z in enumerate(zeniths):
        print(f"{z:>5.0f} " + "".join(
            f"{res[str(y)][i][ie] / res[str(y)][i][iw]:>12.3f}" for y in years))
    del ref
    with open(args.out, "w") as fh:
        json.dump({"zeniths": zeniths.tolist(), "azimuths": azimuths.tolist(),
                   "rc": res}, fh, indent=1)
    print("EPOCH_DONE", flush=True)


# ----------------------------------------------------------------- table ----
def _honda_refs():
    h = dict(np.load("honda_kam.npz"))
    He, Hcz = h["E"], h["czlo"]
    hs = {"total_numu": h["numu"], "total_antinumu": h["numubar"],
          "total_nue": h["nue"], "total_antinue": h["nuebar"]}
    ih = int(np.argmin(abs(Hcz - 0.0)))
    iv = int(np.argmin(abs(Hcz - 0.9)))
    we, hv = {}, {}
    for sp, nm in hs.items():
        we[sp] = {}
        for c, lab in ((0.05, "87"), (0.15, "81")):
            k = int(np.argmin(abs(Hcz - (c - 0.05))))
            r = nm[k].max(0) / nm[k].min(0)
            we[sp][lab] = [float(np.interp(E, He, r)) for E in EW_E]
        hv[sp] = [_log_at(nm[ih].mean(0), He, E) / _log_at(nm[iv].mean(0), He, E)
                  for E in HV_E]
    bt = dict(np.load("bartol_kam.npz"))
    bhv = {}
    for sp, key in (("total_numu", "num"), ("total_antinumu", "nbm"),
                    ("total_nue", "nue"), ("total_antinue", "nbe")):
        y = 0.5 * (bt[f"{key}_fmin"] + bt[f"{key}_fmax"])
        ihb = int(np.argmin(abs(bt["cz"] - 0.05)))
        ivb = int(np.argmin(abs(bt["cz"] - 0.95)))
        bhv[sp] = [_log_at(y[ihb], bt["E"], E) / _log_at(y[ivb], bt["E"], E)
                   for E in HV_E]
    return we, hv, bhv


def cmd_table(args):
    files = sorted(args.files)
    cfg = []
    for f in files:
        with open(f) as fh:
            cfg.append(json.load(fh))
    names = [f"{c['model'][:9]}/{c['primary']}" for c in cfg]
    we_h, hv_h, hv_b = _honda_refs()
    w = 17

    def head(title):
        print("\n" + "=" * 78)
        print(title)
        print("=" * 78)
        print(f"{'':>16}" + "".join(f"{n:>{w}}" for n in names))

    head("A. nu_mu West/East vs Honda (max/min over azimuth), bending ON")
    for lab in ("87", "81"):
        for k, E in enumerate(EW_E):
            hval = we_h["total_numu"][lab][k]
            print(f"{lab + 'deg ' + str(E) + 'GeV':>16}"
                  + "".join(
                      f"{c['bend_on']['we_total_numu'][lab][k]:>10.2f}"
                      f"({c['bend_on']['we_total_numu'][lab][k] / hval - 1:+5.0%})"
                      for c in cfg) + f"   Honda {hval:.2f}")

    for bk, blab in (("bend_off", "OFF"), ("bend_on", "ON")):
        head(f"B. four-species W/E at 87 deg / 0.5 GeV, muon_bending {blab}")
        for sp in SP:
            hval = we_h[sp]["87"][0]
            print(f"{sp.replace('total_', ''):>16}"
                  + "".join(f"{c[bk]['we_' + sp]['87'][0]:>17.2f}" for c in cfg)
                  + f"   Honda {hval:.2f}")
        print(f"{'nue/antinue':>16}"
              + "".join(
                  f"{c[bk]['we_total_nue']['87'][0] / c[bk]['we_total_antinue']['87'][0]:>17.2f}"
                  for c in cfg)
              + f"   Honda {we_h['total_nue']['87'][0] / we_h['total_antinue']['87'][0]:.2f}")
        print(f"{'numu/antinumu':>16}"
              + "".join(
                  f"{c[bk]['we_total_numu']['87'][0] / c[bk]['we_total_antinumu']['87'][0]:>17.2f}"
                  for c in cfg)
              + f"   Honda {we_h['total_numu']['87'][0] / we_h['total_antinumu']['87'][0]:.2f}")

    for sp in ("total_numu", "total_nue"):
        head(f"C. horizon/vertical {sp.replace('total_', '')}"
             " (az-averaged); ratio to Honda in ()")
        for k, E in enumerate(HV_E):
            print(f"{str(E) + ' GeV':>16}"
                  + "".join(f"{c['bend_on']['hv_' + sp][k]:>10.2f}"
                            f"({c['bend_on']['hv_' + sp][k] / hv_h[sp][k]:5.2f})"
                            for c in cfg)
                  + f"   Honda {hv_h[sp][k]:.2f}  Bartol {hv_b[sp][k]:.2f}")
        for k, E in enumerate(HV_E):
            print(f"{'/Bartol ' + str(E):>16}"
                  + "".join(f"{c['bend_on']['hv_' + sp][k] / hv_b[sp][k]:>17.3f}"
                            for c in cfg))
    print("TABLE_DONE")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--model", default="SIBYLL23D")
    b.add_argument("--primary", default="GSF", choices=sorted(PRIMARIES))
    b.add_argument("--cz", default="0.05")
    b.set_defaults(func=cmd_build)
    m = sub.add_parser("measure")
    m.add_argument("--model", default="SIBYLL23D")
    m.add_argument("--primary", default="GSF", choices=sorted(PRIMARIES))
    m.add_argument("--n-jobs", type=int, default=4)
    m.add_argument("--response-only", action="store_true",
                   help="vary only G_s (gs_interaction_model/gs_primary); keep "
                        "the base, E_off table and cone profile at "
                        "SIBYLL23D/GSF -- the like-for-like response scan")
    m.add_argument("--out", required=True)
    m.set_defaults(func=cmd_measure)
    g = sub.add_parser("gsresp")
    g.add_argument("--model", default="SIBYLL23D")
    g.add_argument("--primary", default="GSF", choices=sorted(PRIMARIES))
    g.add_argument("--cz", default="0.05")
    g.add_argument("--energy", type=float, default=0.5)
    g.add_argument("--out", required=True)
    g.set_defaults(func=cmd_gsresp)
    ep = sub.add_parser("epoch")
    ep.add_argument("--years", default="2000,2010,2020")
    ep.add_argument("--zeniths", default="0,75,87")
    ep.add_argument("--azimuths", default="0,90,180,270")
    ep.add_argument("--n-jobs", type=int, default=12)
    ep.add_argument("--out", required=True)
    ep.set_defaults(func=cmd_epoch)
    t = sub.add_parser("table")
    t.add_argument("files", nargs="+")
    t.set_defaults(func=cmd_table)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
