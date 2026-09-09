"""Before/after for the Phase-1 cutoff repairs: zenith shape and East-West.

Runs the delivered ``solve()`` in several configurations that differ ONLY in the
cutoff map and in how the sub-limb part of the production cone is treated, and
reports the two observables the paper quotes:

* horizon/vertical  ``Phi(cosZ=0.05)/Phi(cosZ=0.95)`` (azimuth-averaged) at
  0.3/0.5/1/3 GeV, versus Honda (paper Sections 4.5 / 6);
* the West/East ratio at the Honda cos(zenith) bins 0.45/0.35/0.25/0.15/0.05
  (63/69/75/81/87 deg) at 0.5 and 1 GeV, versus Honda's max/min over azimuth
  (paper Section 4.1 / `validate_ew_zenith.py`).

Configurations
--------------
``B0``  the **committed** map (uniform 13 zenith nodes, single 20-point ladder to
        40 GV -> vertical R_c = 11.93 GV, six saturated 40.00 GV limb cells) with
        the legacy far-side sub-limb stitch.  This is what produced the numbers
        in PAPER_DRAFT.
``B1``  the **uncommitted working-tree** map (24-point ladder to 55 GV ->
        vertical R_c = 8.79 GV, the allowed-island dropout) + far-side stitch.
``A``   the new map (1 GV ladder + bisection, dense limb nodes) + production-point
        sub-limb treatment with Earth-shadow blocking.  The delivered default.
``A-fs``  new map, legacy far-side sub-limb stitch -- isolates the sub-limb change.
``A-uni`` new map resampled onto the legacy uniform 13-node zenith grid, with the
        production-point treatment -- isolates the near-limb node density.

Run from tools/mceq3d with a warm ``.cache3d``::

    python diag_cutoff_ablation.py [--configs B0,B1,A,A-fs,A-uni]
"""

from __future__ import annotations

import argparse
import importlib.util  # noqa: F401  (mceq_config import shim)
from datetime import datetime

import numpy as np

from mceq3d_flux import MCEq3DFlux, _zenith_nodes

LAT, LON = 36.43, 137.31
DATE = datetime(2020, 1, 1)
CACHE = ".cache3d"
HV_E = (0.3, 0.5, 1.0, 3.0)
EW_CZ = (0.45, 0.35, 0.25, 0.15, 0.05)
EW_E = (0.5, 1.0)

# cached full-sphere maps built by the two pre-repair schemes (see module docstring)
LEGACY_MAPS = {
    "B0": ".cache3d/finerc_ced98f306edd8e46.npz",  # committed: n_scan=20, r_hi=40
    "B1": ".cache3d/finerc_8e83b6248bd62681.npz",  # working tree: n_scan=24, r_hi=55
}


def log_at(y, x, X):
    return float(np.exp(np.interp(np.log(X), np.log(x),
                                  np.log(np.maximum(y, 1e-300)))))


def _resample_uniform(fine, n_zen=13):
    """The new map read on the LEGACY uniform zenith nodes (each hemisphere)."""
    zen, az, rc = fine
    zl = _zenith_nodes(n_zen, limb_nodes=False)  # 0..89
    zu = 180.0 - zl[::-1]
    znew = np.concatenate([zl, zu])
    out = np.empty((len(znew), len(az)))
    for j in range(len(az)):
        out[:, j] = np.interp(znew, zen, rc[:, j])
    return znew, az, out


def run(eng, tag, cz, az, fine_override=None, sublimb="prod_point", n_jobs=None):
    kw = dict(offaxis=True, use_cache=True, cone_cutoff=True, cache_dir=CACHE,
              date=DATE, sublimb=sublimb, n_jobs=n_jobs)
    if fine_override is not None:
        eng._finemap_memo = {}
        orig = eng.finemap_rc
        eng.finemap_rc = lambda *a, **k: fine_override
        try:
            r = eng.solve(LAT, LON, cz, az, **kw)
        finally:
            eng.finemap_rc = orig
    else:
        r = eng.solve(LAT, LON, cz, az, **kw)
    return r


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--configs", default="B0,B1,A,A-fs,A-uni")
    p.add_argument("--n-jobs", type=int, default=16)
    args = p.parse_args(argv)
    configs = args.configs.split(",")

    eng = MCEq3DFlux(base_model="hybrid", primary=("GlobalSplineFitBeta", None),
                     daemonflux_location="kamioka")
    h = dict(np.load("honda_kam.npz"))
    He, Hcz, nm = h["E"], h["czlo"], h["numu"]

    fine_new = None
    if any(t not in LEGACY_MAPS for t in configs):
        # only build/load the repaired map when a config actually needs it
        fine_new = eng.finemap_rc(LAT, LON, DATE, cache_dir=CACHE,
                                  n_jobs=args.n_jobs)
        print(f"new map: {fine_new[2].shape} nodes, R_c range "
              f"{fine_new[2].min():.2f}-{fine_new[2].max():.2f} GV, "
              f"vertical {fine_new[2][0, 0]:.2f} GV")

    def fine_for(tag):
        if tag in LEGACY_MAPS:
            d = np.load(LEGACY_MAPS[tag])
            return d["zen"], d["az"], d["rc"]
        if tag == "A-uni":
            return _resample_uniform(fine_new)
        return fine_new

    cz_hv = np.array([0.05, 0.95])
    az4 = np.array([0.0, 90.0, 180.0, 270.0])
    cz_ew = np.array(EW_CZ)
    az2 = np.array([90.0, 270.0])

    hv_rows, ew_rows = {}, {}
    for tag in configs:
        sub = "farside" if tag in ("B0", "B1", "A-fs") else "prod_point"
        fine = fine_for(tag)
        over = None if tag == "A" else fine
        r1 = run(eng, tag, cz_hv, az4, over, sub, args.n_jobs)
        e = r1["e"]
        f = r1["flux"]["total_numu"].mean(1)
        hv_rows[tag] = [log_at(f[0], e, E) / log_at(f[1], e, E) for E in HV_E]
        r2 = run(eng, tag, cz_ew, az2, over, sub, args.n_jobs)
        we = r2["flux"]["total_numu"][:, 1] / r2["flux"]["total_numu"][:, 0]
        ew_rows[tag] = {E: [float(np.interp(E, r2["e"], we[i]))
                            for i in range(len(EW_CZ))] for E in EW_E}
        print(f"  done {tag} (sublimb={sub}, map {fine[2].shape})", flush=True)

    hv_h = []
    for E in HV_E:
        ih = int(np.argmin(np.abs(Hcz - 0.0)))
        iv = int(np.argmin(np.abs(Hcz - 0.9)))
        hv_h.append(log_at(nm[ih].mean(0), He, E) / log_at(nm[iv].mean(0), He, E))

    print("\n== horizon/vertical (numu, az-averaged), ratio to Honda in () ==")
    print("config   " + "".join(f"{E:>16.2f} GeV" for E in HV_E))
    print("Honda    " + "".join(f"{v:>20.3f}" for v in hv_h))
    for tag in configs:
        cells = "".join(f"{v:>11.3f} ({v / hv_h[i]:5.3f})"
                        for i, v in enumerate(hv_rows[tag]))
        print(f"{tag:<8} {cells}")

    print("\n== West/East (numu) vs Honda max/min over azimuth ==")
    for E in EW_E:
        print(f"-- E = {E} GeV")
        hd = []
        for czt in EW_CZ:
            ih = int(np.argmin(np.abs(Hcz - czt)))
            hd.append(float(np.interp(E, He, nm[ih].max(0) / nm[ih].min(0))))
        print("  zenith  " + "".join(f"{np.degrees(np.arccos(c)):>13.0f}"
                                     for c in EW_CZ))
        print("  Honda   " + "".join(f"{v:>13.2f}" for v in hd))
        for tag in configs:
            cells = "".join(f"{v:>7.2f}({v / hd[i] - 1:+5.0%})"
                            for i, v in enumerate(ew_rows[tag][E]))
            print(f"  {tag:<7} {cells}")
    print("\nDIAG_CUTOFF_ABLATION_DONE")


if __name__ == "__main__":
    main()
