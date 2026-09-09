"""Task-11: WHICH factor causes the 11-15% sub-GeV horizon-shape undershoot?

diag_full_comparison.py Section B found the delivered zenith shape
Phi(horizon)/Phi(vertical) undershoots Honda by 11-15% at 0.3-1 GeV, when measured
through the FULL delivered solve() path (hybrid base + GSF primary + Kamioka + G).
The earlier moment-cone validation (diag_cone_fix.py) reported 0-8% agreement -- but
it measured base*E_off only, with base_model="mceq" + generic location and NO
geomagnetic factor G. So either G's zenith dependence, or the base/site change,
is eating the horizon excess.

The delivered horizon/vertical ratio factorises exactly:
    HV_delivered = HV_base * HV_Eoff * HV_G
where HV_X = X(horizon)/X(vertical). This computes each factor separately for the
DELIVERED configuration and compares the product to Honda's HV, so the culprit is
immediately visible. Also repeats it for base_model="mceq"+generic (the old
validation config) to confirm the configs really do differ and by how much.

Run from tools/mceq3d (PYTHONPATH=$PWD). Caches should be warm.
"""
from datetime import datetime

import numpy as np

from mceq3d_flux import MCEq3DFlux

LAT, LON = 36.43, 137.31
DATE = datetime(2020, 1, 1)
CACHE = ".cache3d"
EGRID = (0.3, 0.5, 1.0, 3.0)


def log_at(y, x, X):
    return float(np.exp(np.interp(np.log(X), np.log(x),
                                  np.log(np.maximum(y, 1e-300)))))


def honda_hv(h, key, E):
    """Honda horizon/vertical (az-averaged) for species key at energy E."""
    He, Hcz = h["E"], h["czlo"]
    arr = h[key]
    ih = int(np.argmin(np.abs(Hcz - 0.0)))   # cosZ bin 0.0-0.1 -> horizon
    iv = int(np.argmin(np.abs(Hcz - 0.9)))   # cosZ bin 0.9-1.0 -> vertical
    return (log_at(arr[ih].mean(0), He, E) / log_at(arr[iv].mean(0), He, E))


def decompose(label, base_model, primary, location, use_G):
    print(f"\n{'='*76}")
    print(f"CONFIG: {label}")
    print(f"  base_model={base_model!r} primary={primary} location={location!r} "
          f"G={'ON' if use_G else 'OFF'}")
    print(f"{'='*76}")
    kw = dict(base_model=base_model, daemonflux_location=location)
    if primary is not None:
        kw["primary"] = primary
    eng = MCEq3DFlux(**kw)
    e = eng.e
    cz = np.array([0.05, 0.95])          # horizon, vertical
    az = np.array([0.0, 90.0, 180.0, 270.0])

    base = eng.base(cz)["total_numu"]     # (2, nE)
    eoff = eng.offaxis_factor(cz)["total_numu"]   # (2, nE)

    if use_G:
        r = eng.solve(LAT, LON, cz, az, use_cache=True, cache_dir=CACHE,
                      date=DATE)
        full = r["flux"]["total_numu"].mean(1)    # az-avg -> (2, nE)
    else:
        full = base * eoff

    h = dict(np.load("honda_kam.npz"))
    print(f"{'E':>5} {'HV_base':>8} {'HV_Eoff':>8} {'HV_G':>7} {'HV_full':>8} "
          f"{'Honda':>7} {'full/Honda':>11}")
    for E in EGRID:
        hv_b = log_at(base[0], e, E) / log_at(base[1], e, E)
        hv_e = log_at(eoff[0], e, E) / log_at(eoff[1], e, E)
        hv_f = log_at(full[0], e, E) / log_at(full[1], e, E)
        hv_g = hv_f / max(hv_b * hv_e, 1e-300)   # residual = G's contribution
        hv_h = honda_hv(h, "numu", E)
        print(f"{E:5.2f} {hv_b:8.3f} {hv_e:8.3f} {hv_g:7.3f} {hv_f:8.3f} "
              f"{hv_h:7.3f} {hv_f/hv_h:11.3f}")
    return


def main():
    # 1. The DELIVERED configuration (what diag_full_comparison.py measured)
    decompose("DELIVERED (what users get)", "hybrid",
              ("GlobalSplineFitBeta", None), "kamioka", use_G=True)
    # 2. Same, but with G switched off -> isolates G's zenith effect exactly
    decompose("DELIVERED base+E_off only (G OFF)", "hybrid",
              ("GlobalSplineFitBeta", None), "kamioka", use_G=False)
    # 3. The OLD validation configuration (diag_cone_fix.py style)
    decompose("OLD validation config (mceq base, generic, no G)", "mceq",
              None, "generic", use_G=False)

    print("\n" + "="*76)
    print("READ: HV_G < 1 means the geomagnetic factor SUPPRESSES the horizon")
    print("relative to the vertical, eating the E_off excess. Compare 'full/Honda'")
    print("between config 1 (delivered) and config 3 (old validation) to see how")
    print("much of the discrepancy is G vs the base/site change.")
    print("DIAG_SHAPE_DECOMPOSE_DONE")


if __name__ == "__main__":
    main()
