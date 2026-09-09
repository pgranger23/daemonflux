"""Task-10: sensitivity test for the charge-dependent E-W miss.

diag_full_comparison.py Section E found the delivered model predicts almost NO
charge-dependent East-West splitting (nu-nubar W/E amplitude diff ~0.00-0.02 for
numu) while Honda shows a large one (diff -0.91 to -1.53 for numu, +1.7 to +3.3
for nue). The pion-production-angle charge asymmetry hypothesis was refuted
directly from data (pi+ vs pi- widths agree to 1-3%).

UPDATE 2026-09-03: the earlier version of this script measured the splitting as
``max/min`` over a symmetric 24-point azimuth grid, which is EXACTLY invariant
under a rotation by +delta vs -delta -- i.e. blind to the very mechanism being
scaled, which is why the answer "saturated" instead of growing. It now reports
the shift-sensitive first-harmonic observables of ``diag_ew_charge_fourier``
(amplitude a1/a0, phase dphi about the geomagnetic E-W axis, transverse
component s1/a0) alongside the legacy max/min.

The ONLY charge-dependent mechanism in the delivered model is the coherent
muon-bending axis shift (`muon_bending.bending_deflection`, applied in
`mceq3d_flux.cone_geff` lines ~800-836: the mu-decay cone is centred on
n +/- d_cone depending on whether the species comes from mu+ or mu- decay).
This monkey-patches ONLY that one function with a scale multiplier (leaving the
angular WIDTH machinery, cone shape, and everything else untouched) and reruns
the delivered solve() at several scales, to see:
  (a) does a bigger coherent shift move the charge-split in the RIGHT direction
      (toward Honda's sign and magnitude)?
  (b) how big a multiplier would be needed -- if even a large one (x10-30) can't
      reach Honda's magnitude, the coherent-shift MECHANISM itself is the wrong
      lever (not just under-tuned), and something else in Honda's calculation
      (K+/K- asymmetry, or a genuinely different transport effect) is missing.

Run from tools/mceq3d (PYTHONPATH=$PWD). Needs cached G_s/finemap (warm from
diag_full_comparison.py's run at the same lat/lon/date/config).
"""
from datetime import datetime

import numpy as np

import muon_bending as mb
from mceq3d_flux import MCEq3DFlux
from diag_ew_charge_fourier import (
    ew_observables, geomagnetic_ew_axis, honda_ew, honda_table, _wrap180,
)

LAT, LON = 36.43, 137.31
DATE = datetime(2020, 1, 1)
CACHE = ".cache3d"

_orig_bending_deflection = mb.bending_deflection


def scaled_bending_deflection(scale):
    def f(vel_hat, b_gauss_enu, charge=+1):
        return _orig_bending_deflection(vel_hat, b_gauss_enu, charge=charge) * scale
    return f


def main():
    eng = MCEq3DFlux(base_model="hybrid", primary=("GlobalSplineFitBeta", None),
                     daemonflux_location="kamioka")
    cz = np.array([0.05])  # 87 deg
    naz = 24
    az = (np.arange(naz) + 0.5) * 360.0 / naz  # fine azimuth for a real max/min

    def we_diff(scale, E):
        mb.bending_deflection = (scaled_bending_deflection(scale) if scale != 1.0
                                 else _orig_bending_deflection)
        try:
            r = eng.solve(LAT, LON, cz, az, offaxis=True, use_cache=True,
                         cone_cutoff=True, cache_dir=CACHE, date=DATE)
        finally:
            mb.bending_deflection = _orig_bending_deflection
        e = r["e"]
        ew_axis = geomagnetic_ew_axis()
        out = {}
        for fl, nu, nub in (("numu", "total_numu", "total_antinumu"),
                            ("nue", "total_nue", "total_antinue")):
            o = []
            for sp in (nu, nub):
                f = r["flux"][sp][0]          # (naz, nE)
                v = np.array([np.interp(E, e, f[j]) for j in range(f.shape[0])])
                o.append(ew_observables(v, az, ew_axis))
            out[fl] = o
        return out

    h = honda_table()
    ew_axis = geomagnetic_ew_axis()
    scales = (0.0, 1.0, 3.0, 10.0, 30.0)
    for E in (0.5, 1.0):
        print(f"\n{'='*96}")
        print(f"E = {E} GeV, cosZ=0.05 (87 deg): nu MINUS nubar, vs bending scale")
        print(f"{'='*96}")
        print(f"{'flavour':>7} {'scale':>6} | {'d(a1/a0)':>9} {'d(dphi)deg':>11} "
              f"{'d(s1/a0)':>9} {'d(max/min)':>11} | {'(nu,nubar) dphi':>22}")
        for fl in ("numu", "nue"):
            for sc in scales:
                a, b = we_diff(sc, E)[fl]
                print(f"{fl:>7} {sc:6.1f} | {a['a1_rel']-b['a1_rel']:+9.4f} "
                      f"{_wrap180(a['dphi']-b['dphi']):+11.2f} "
                      f"{a['s1_rel']-b['s1_rel']:+9.4f} "
                      f"{a['maxmin']-b['maxmin']:+11.3f} | "
                      f"({a['dphi']:+7.2f},{b['dphi']:+7.2f})")
            hk = {"numu": ("numu", "numubar"), "nue": ("nue", "nuebar")}[fl]
            ha = honda_ew(h, hk[0], 0.05, E, ew_axis)
            hb = honda_ew(h, hk[1], 0.05, E, ew_axis)
            print(f"{fl:>7} {'Honda':>6} | {ha['a1_rel']-hb['a1_rel']:+9.4f} "
                  f"{_wrap180(ha['dphi']-hb['dphi']):+11.2f} "
                  f"{ha['s1_rel']-hb['s1_rel']:+9.4f} "
                  f"{ha['maxmin']-hb['maxmin']:+11.3f} | "
                  f"({ha['dphi']:+7.2f},{hb['dphi']:+7.2f})")
    print("\nscale=0 switches the coherent shift off entirely: the residual there is")
    print("the charge-blind floor (different f_mu for nu and nubar). Growth of")
    print("d(dphi)/d(s1/a0) with scale is the mechanism responding; comparison with")
    print("the Honda row says how much of the real splitting it can carry.")
    print("DIAG_EW_CHARGE_SCALE_DONE")


if __name__ == "__main__":
    main()
