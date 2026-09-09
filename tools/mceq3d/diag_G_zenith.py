"""Task-11 follow-up: what actually drives HV_G ~ 0.82 (G suppressing the horizon
15-19% relative to the vertical)?

The zenith-dependent f_mu fix moved it only 0.3%, so the blend weight is not the
lever. HV_G = G(horizon)/G(vertical) is built from two ingredients:
  (1) the CUTOFF VALUES R_c(horizon) vs R_c(vertical) -- at Kamioka the az-averaged
      horizon cutoff is higher than the vertical one, which suppresses the horizon;
  (2) the CASCADE RESPONSE curve G_s(R_c, E) -- how much flux is lost for a given
      cutoff, which zenith_dependent_geomag=True recomputes per zenith.

Crucially HV_G < 1 is PHYSICALLY EXPECTED (Honda's absolute flux is geomagnetically
suppressed too). The real question is whether 0.82 is too strong. This prints:
  * R_c az-averaged at horizon vs vertical (from the actual cutoff map);
  * G_s evaluated at those cutoffs, per zenith, to show how much of HV_G comes from
    the cutoff difference vs the response-curve difference;
  * the SINGLE-cutoff G (no cone) vs the cone-averaged Geff, to isolate whether the
    production-cone averaging itself is doing the suppressing.

Run from tools/mceq3d (PYTHONPATH=$PWD). Caches warm.
"""
from datetime import datetime

import numpy as np

from mceq3d_flux import MCEq3DFlux, _interp_rc

LAT, LON = 36.43, 137.31
DATE = datetime(2020, 1, 1)
CACHE = ".cache3d"
EGRID = (0.3, 0.5, 1.0, 3.0)


def main():
    eng = MCEq3DFlux(base_model="hybrid", primary=("GlobalSplineFitBeta", None),
                     daemonflux_location="kamioka")
    e = eng.e
    cz = np.array([0.05, 0.95])          # horizon, vertical
    az = np.linspace(0.0, 360.0, 13)[:-1]

    # the actual cutoff map used by solve()
    rc_map = eng.cutoff_map_for(LAT, LON, cz, az, DATE, cache_dir=CACHE) \
        if hasattr(eng, "cutoff_map_for") else None
    if rc_map is None:
        import geomag_backtrace as gb
        zen = np.degrees(np.arccos(np.clip(cz, -1, 1)))
        rc_map = gb.cutoff_map(LAT, LON, DATE, zen, az, n_scan=32)
    print("Cutoff R_c [GV] (az-averaged):")
    print(f"  horizon (cosZ=0.05): {rc_map[0].mean():6.2f}   "
          f"(min {rc_map[0].min():.1f}, max {rc_map[0].max():.1f})")
    print(f"  vertical(cosZ=0.95): {rc_map[1].mean():6.2f}   "
          f"(min {rc_map[1].min():.1f}, max {rc_map[1].max():.1f})")

    rc_grid = np.linspace(0.1, 40.0, 40)
    # per-zenith response curves (what solve() uses with zenith_dependent_geomag)
    G_h, _ = eng.geomag_response(rc_grid, cz_ref=0.05, cache_dir=CACHE)
    G_v, _ = eng.geomag_response(rc_grid, cz_ref=0.95, cache_dir=CACHE)

    print("\nDecomposing HV_G into cutoff-value vs response-curve effects (numu):")
    print(f"{'E':>5} {'G_h(Rc_h)':>10} {'G_v(Rc_v)':>10} {'HV_single':>10} | "
          f"{'G_v(Rc_h)':>10} {'cutoffOnly':>11} {'curveOnly':>10}")
    for E in EGRID:
        je = int(np.argmin(np.abs(e - E)))
        rch, rcv = rc_map[0].mean(), rc_map[1].mean()
        gh = _interp_rc(rch, rc_grid, G_h["total_numu"])[je]
        gv = _interp_rc(rcv, rc_grid, G_v["total_numu"])[je]
        # counterfactual: vertical response curve evaluated at the horizon cutoff
        gv_at_h = _interp_rc(rch, rc_grid, G_v["total_numu"])[je]
        hv_single = gh / max(gv, 1e-300)
        cutoff_only = gv_at_h / max(gv, 1e-300)   # same curve, different cutoff
        curve_only = gh / max(gv_at_h, 1e-300)    # same cutoff, different curve
        print(f"{E:5.2f} {gh:10.4f} {gv:10.4f} {hv_single:10.3f} | "
              f"{gv_at_h:10.4f} {cutoff_only:11.3f} {curve_only:10.3f}")

    print("\nREAD:")
    print("  cutoffOnly = suppression from the horizon cutoff being higher than")
    print("               the vertical one (using ONE response curve).")
    print("  curveOnly  = extra suppression from the per-zenith cascade response")
    print("               curve differing (same cutoff).")
    print("  Their product ~ HV_single (the no-cone G ratio). Compare HV_single to")
    print("  the delivered HV_G~0.82 to see how much the CONE averaging adds.")
    print("DIAG_G_ZENITH_DONE")


if __name__ == "__main__":
    main()
