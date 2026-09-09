"""Settle the Ru-vs-Reff question with a FINE near-horizon admittance scan.

BACKGROUND. `cutoff_map` collapses the back-traced admittance to the first
forbidden rigidity scanning downward -- the UPPER cutoff R_U -- discarding any
allowed islands below it. The penumbra-averaged EFFECTIVE cutoff R_eff <= R_U, and
the gap is largest where the penumbra is widest. That would explain the residual
Honda discrepancy IF the gap is large near the horizon and small at the vertical.

A coarse scan (~1.1 GV steps) found the OPPOSITE: one allowed island at the
vertical (R_eff = 9.88 vs R_U = 11.34, i.e. -1.46 GV) and none at 87 deg E/W. But
1.1 GV cannot resolve narrow islands, and near-horizon trajectories are chaotic --
exactly where fine structure is expected. This scans finely enough to settle it.

TWO QUESTIONS, because raw GV shifts proved to be a poor proxy for delivered flux
changes (the production-point displacement gave a -2.3 GV raw shift but only ~15%
of the E-W gap, since the +-20 deg cone average and G_s saturation both compress
it):
  (a) RAW: is R_eff materially below R_U near the horizon, and is the gap larger
      there than at the vertical (the direction the hypothesis needs)?
  (b) DELIVERED: translating any measured DR_c through the cached cascade response
      G_s, how much of the zenith-shape / E-W discrepancy would it actually close?

R_eff definitions reported:
  * bandwidth:     R_U - (total allowed rigidity width below R_U)
  * flux-weighted: the sharp cutoff giving the same integrated transmission for a
      power-law primary spectrum R^-gamma, i.e.
      R_eff = [ (gamma-1) * Int T(R) R^-gamma dR ]^(1/(1-gamma))
    This is the physically meaningful one for a flux calculation.

Run from tools/mceq3d (PYTHONPATH=$PWD). Needs ppigrf. Minutes, not seconds.
"""
import time
from datetime import datetime

import numpy as np

import geomag_backtrace as gb

LAT, LON = 36.43, 137.31
DATE = datetime(2020, 1, 1)
GAMMA = 2.7  # primary integral spectral index

# (label, zenith_deg, azimuth_deg). Vertical is the control: we already know it has
# an island, and the hypothesis REQUIRES the horizon gap to exceed the vertical one.
DIRS = [
    ("vertical",   0.0,  0.0),
    ("60 deg E",  60.0, 90.0),
    ("84 deg E",  84.0, 90.0),
    ("87 deg E",  87.0, 90.0),
    ("87 deg W",  87.0, 270.0),
]
R_LO, R_HI, N_SCAN = 1.0, 55.0, 220   # ~0.246 GV steps, ceiling above the ~42 GV E cutoff


def reff_from_admittance(rs, a, gamma=GAMMA):
    """(R_U, R_L, R_eff_bandwidth, R_eff_fluxweighted) from an admittance scan.

    ``rs`` descending rigidities, ``a`` allowed fraction (0/1) at each.
    """
    order = np.argsort(rs)
    R, A = rs[order], a[order]          # ascending
    forb = R[A < 0.5]
    allw = R[A >= 0.5]
    if forb.size == 0:
        return R.min(), R.min(), R.min(), R.min()
    R_U = forb.max()                    # highest forbidden = what the code uses
    R_L = allw.min() if allw.size else R.max()
    dR = np.gradient(R)
    # allowed width strictly below R_U (the islands the scalar cutoff throws away)
    below = (R < R_U) & (A >= 0.5)
    R_eff_bw = R_U - float(np.sum(dR[below]))
    # Flux-weighted: integrate the true transmission against R^-gamma.
    # The scan only covers [R_LO, R_HI]; above R_HI everything is allowed
    # (T = 1), and that tail MUST be included or the integral is badly
    # underestimated for high-cutoff directions, which drives R_eff far ABOVE the
    # scan range (a perfectly sharp 41.68 GV cutoff scanned to 55 GV returns a
    # spurious 73 GV without the tail, versus 41.5 GV with it).
    w = R ** (-gamma)
    integral = float(np.sum(A * w * dR))
    r_top = float(R.max())
    integral += r_top ** (1.0 - gamma) / (gamma - 1.0)   # T = 1 tail above the scan
    R_eff_fw = ((gamma - 1.0) * integral) ** (1.0 / (1.0 - gamma))
    return R_U, R_L, R_eff_bw, R_eff_fw


def main():
    print(f"Fine admittance scan: {len(DIRS)} directions x {N_SCAN} rigidities "
          f"({R_LO}-{R_HI} GV, step {(R_HI-R_LO)/(N_SCAN-1):.3f} GV)")
    zen = np.array([d[1] for d in DIRS])
    az = np.array([d[2] for d in DIRS])
    t0 = time.time()
    # one call per direction keeps the (zenith x azimuth) grid from exploding
    results = {}
    for label, z, a_deg in DIRS:
        t1 = time.time()
        rc, rs, A = gb.cutoff_map(LAT, LON, DATE, [z], [a_deg],
                                  r_lo=R_LO, r_hi=R_HI, n_scan=N_SCAN,
                                  return_admittance=True)
        adm = A[0, 0]
        results[label] = (float(rc[0, 0]), rs, adm)
        n_allowed = int(np.sum(adm >= 0.5))
        print(f"  {label:10s} done in {time.time()-t1:5.0f}s  "
              f"(scalar R_c={rc[0,0]:5.1f} GV, {n_allowed}/{len(adm)} allowed)",
              flush=True)
    print(f"  total {time.time()-t0:.0f}s\n")

    print("(a) RAW admittance structure")
    print(f"{'direction':>10} {'R_U':>7} {'R_L':>7} {'Reff_bw':>8} {'Reff_fw':>8} "
          f"{'DR=R_U-Reff_fw':>15} {'islands':>8}")
    summary = {}
    for label, _, _ in DIRS:
        _, rs, adm = results[label]
        R_U, R_L, bw, fw = reff_from_admittance(rs, adm)
        order = np.argsort(rs)
        R, A = rs[order], adm[order]
        below = (R < R_U) & (A >= 0.5)
        # count contiguous allowed islands below R_U
        isl = int(np.sum(np.diff(np.concatenate([[0], below.astype(int)])) == 1))
        summary[label] = (R_U, fw, R_U - fw)
        print(f"{label:>10} {R_U:7.2f} {R_L:7.2f} {bw:8.2f} {fw:8.2f} "
              f"{R_U-fw:15.2f} {isl:8d}")

    print("\nHYPOTHESIS TEST: the gap DR must be LARGER at the horizon than at the")
    print("vertical for this to explain a horizon-specific discrepancy.")
    dv = summary["vertical"][2]
    de = summary["87 deg E"][2]
    print(f"  vertical DR = {dv:+.2f} GV ; 87 deg E DR = {de:+.2f} GV")
    if de > dv + 0.5:
        print("  -> SUPPORTS the hypothesis (horizon gap exceeds vertical).")
    elif abs(de - dv) <= 0.5:
        print("  -> NEUTRAL: gaps comparable; would shift both, not the ratio.")
    else:
        print("  -> REFUTES the hypothesis (vertical gap exceeds horizon): applying")
        print("     Reff would LOWER the already-correct vertical cutoff and make")
        print("     the horizon/vertical ratio WORSE.")

    # ---------------- (b) delivered impact ----------------
    print("\n(b) DELIVERED impact, translating DR_c through the cached G_s")
    try:
        from mceq3d_flux import MCEq3DFlux, _interp_rc
        eng = MCEq3DFlux(base_model="hybrid", primary=("GlobalSplineFitBeta", None),
                         daemonflux_location="kamioka")
        e = eng.e
        rc_grid = np.linspace(0.1, 55.0, 56)
        G_h, _ = eng.geomag_response(rc_grid, cz_ref=0.05, cache_dir=".cache3d")
        G_v, _ = eng.geomag_response(rc_grid, cz_ref=0.95, cache_dir=".cache3d")
        print(f"{'E':>5} {'HV_G(R_U)':>10} {'HV_G(Reff)':>11} {'change':>8}  "
              f"(HV_G ~ 0.82 now; Honda needs ~1.0)")
        RUv, FWv, _ = summary["vertical"]
        RUe, FWe, _ = summary["87 deg E"]
        for E in (0.3, 0.5, 1.0, 3.0):
            je = int(np.argmin(np.abs(e - E)))
            gh_u = _interp_rc(RUe, rc_grid, G_h["total_numu"])[je]
            gv_u = _interp_rc(RUv, rc_grid, G_v["total_numu"])[je]
            gh_f = _interp_rc(FWe, rc_grid, G_h["total_numu"])[je]
            gv_f = _interp_rc(FWv, rc_grid, G_v["total_numu"])[je]
            hv_u = gh_u / max(gv_u, 1e-300)
            hv_f = gh_f / max(gv_f, 1e-300)
            print(f"{E:5.2f} {hv_u:10.3f} {hv_f:11.3f} {hv_f-hv_u:+8.3f}")
        print("\n  (single-direction proxy: uses the 87 deg E cutoff vs vertical,")
        print("   NOT the azimuth-averaged/cone-averaged delivered value, so the")
        print("   real delivered change will be SMALLER still -- the cone average")
        print("   and G_s saturation both compress raw cutoff shifts.)")
    except Exception as ex:
        print(f"  (delivered-impact step skipped: {type(ex).__name__}: {ex})")

    np.savez("/tmp/pigrange/claude-130233/-afs-cern-ch-work-p-pigrange/"
             "908b6451-0fbd-441a-958d-19fe0943b711/scratchpad/admittance_fine.npz",
             **{f"{k}_rs": v[1] for k, v in results.items()},
             **{f"{k}_A": v[2] for k, v in results.items()})
    print("\nDIAG_ADMITTANCE_FINE_DONE")


if __name__ == "__main__":
    main()
