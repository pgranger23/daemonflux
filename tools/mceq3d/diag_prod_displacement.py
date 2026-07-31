"""Task-11 fix test: does the production-point displacement in cone_geff move the
delivered zenith shape and E-W toward Honda?

The cutoff seen by a neutrino's parent primary is the one at the PRODUCTION point,
not at the detector. Near the horizon the production point is hundreds of km up the
arrival ray (272 km at 87 deg / h=20 km, 370 km at h=30 km), where the local
vertical has rotated by ~L/R_E, so the same primary direction has a markedly less
extreme local zenith (87 -> 83.7 deg at h=30 km). Since R_c rises steeply toward the
horizon, evaluating at the detector's more-extreme zenith OVERESTIMATES R_c and
over-suppresses the horizon. `cone_geff(prod_displacement=True)` evaluates each cone
sample in the production point's local frame instead.

The effect is zero at the vertical by construction and maximal at the horizon --
matching the observed "vertical fine / horizon wrong" asymmetry -- and is
energy-independent, matching the observed energy-flat ~3 GV offset.

Reports, with the displacement OFF (delivered) and ON:
  * the numu horizon/vertical zenith shape vs Honda (the 11-15% deficit);
  * the 87 deg W/E amplitude vs Honda (the 15-24% overshoot).
Both should improve together if the mechanism is right.

Run from tools/mceq3d (PYTHONPATH=$PWD). Caches warm.
"""
from datetime import datetime

import numpy as np

from mceq3d_flux import MCEq3DFlux

LAT, LON = 36.43, 137.31
DATE = datetime(2020, 1, 1)
CACHE = ".cache3d"


def log_at(y, x, X):
    return float(np.exp(np.interp(np.log(X), np.log(x),
                                  np.log(np.maximum(y, 1e-300)))))


def main():
    eng = MCEq3DFlux(base_model="hybrid", primary=("GlobalSplineFitBeta", None),
                     daemonflux_location="kamioka")
    e = eng.e
    h = dict(np.load("honda_kam.npz"))
    He, Hcz = h["E"], h["czlo"]

    # --- zenith shape: horizon vs vertical, azimuth-averaged ---
    cz = np.array([0.05, 0.95])
    az = np.array([0.0, 90.0, 180.0, 270.0])
    ih = int(np.argmin(np.abs(Hcz - 0.0)))
    iv = int(np.argmin(np.abs(Hcz - 0.9)))

    res = {}
    for tag, pd in (("OFF (delivered)", False), ("ON  (displaced)", True)):
        r = eng.solve(LAT, LON, cz, az, offaxis=True, use_cache=True,
                     cone_cutoff=True, cache_dir=CACHE, date=DATE,
                     prod_displacement=pd)
        res[tag] = r["flux"]["total_numu"].mean(1)  # az-avg -> (2, nE)

    print("ZENITH SHAPE  Phi(horizon)/Phi(vertical), numu, vs Honda")
    print(f"{'E':>5} {'Honda':>7} {'OFF':>7} {'ON':>7} | {'OFF/H':>7} {'ON/H':>7}")
    for E in (0.3, 0.5, 1.0, 3.0):
        hon = (log_at(h["numu"][ih].mean(0), He, E)
               / log_at(h["numu"][iv].mean(0), He, E))
        row = []
        for tag in ("OFF (delivered)", "ON  (displaced)"):
            f = res[tag]
            row.append(log_at(f[0], e, E) / log_at(f[1], e, E))
        print(f"{E:5.2f} {hon:7.3f} {row[0]:7.3f} {row[1]:7.3f} | "
              f"{row[0]/hon:7.3f} {row[1]/hon:7.3f}")

    # --- E-W amplitude at 87 deg ---
    cz2 = np.array([0.05])
    naz = 24
    az2 = (np.arange(naz) + 0.5) * 360.0 / naz
    hwe = h["numu"][ih].max(0) / h["numu"][ih].min(0)
    print("\nEAST-WEST amplitude (max/min over azimuth) at 87 deg, numu, vs Honda")
    print(f"{'E':>5} {'Honda':>7} {'OFF':>7} {'ON':>7} | {'OFF/H':>7} {'ON/H':>7}")
    ew = {}
    for tag, pd in (("OFF (delivered)", False), ("ON  (displaced)", True)):
        r = eng.solve(LAT, LON, cz2, az2, offaxis=True, use_cache=True,
                     cone_cutoff=True, cache_dir=CACHE, date=DATE,
                     prod_displacement=pd)
        f = r["flux"]["total_numu"][0]
        ew[tag] = f.max(0) / np.maximum(f.min(0), 1e-300)
    for E in (0.5, 1.0, 2.0):
        hon = float(np.interp(E, He, hwe))
        a = float(np.interp(E, e, ew["OFF (delivered)"]))
        b = float(np.interp(E, e, ew["ON  (displaced)"]))
        print(f"{E:5.2f} {hon:7.2f} {a:7.2f} {b:7.2f} | {a/hon:7.3f} {b/hon:7.3f}")

    print("\nBoth ON/H columns should move toward 1.000 if the mechanism is right:")
    print("  zenith shape was ~0.83-0.89 (too low), E-W was ~1.15-1.24 (too high).")
    print("DIAG_PROD_DISPLACEMENT_DONE")


if __name__ == "__main__":
    main()
