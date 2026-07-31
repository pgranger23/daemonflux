"""Task-10: sensitivity test for the charge-dependent E-W miss.

diag_full_comparison.py Section E found the delivered model predicts almost NO
charge-dependent East-West splitting (nu-nubar W/E amplitude diff ~0.00-0.02 for
numu) while Honda shows a large one (diff -0.91 to -1.53 for numu, +1.7 to +3.3
for nue). The pion-production-angle charge asymmetry hypothesis was refuted
directly from data (pi+ vs pi- widths agree to 1-3%).

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
        out = {}
        for fl, nu, nub in (("numu", "total_numu", "total_antinumu"),
                            ("nue", "total_nue", "total_antinue")):
            f_nu = r["flux"][nu][0]   # (naz, nE)
            f_nb = r["flux"][nub][0]
            we_nu = float(np.interp(E, e, f_nu.max(0))) / \
                max(float(np.interp(E, e, f_nu.min(0))), 1e-300)
            we_nb = float(np.interp(E, e, f_nb.max(0))) / \
                max(float(np.interp(E, e, f_nb.min(0))), 1e-300)
            out[fl] = (we_nu, we_nb, we_nu - we_nb)
        return out

    honda = {
        0.5: {"numu": -1.30, "nue": +2.62},
        1.0: {"numu": -1.53, "nue": +3.26},
    }
    scales = (1.0, 3.0, 10.0, 30.0)
    for E in (0.5, 1.0):
        print(f"\n{'='*70}\nE = {E} GeV, cosZ=0.05 (87 deg): nu-nubar W/E diff vs bending scale")
        print(f"{'='*70}")
        print(f"{'scale':>7} | {'numu diff':>10} {'(nu,nubar)':>16} | "
              f"{'nue diff':>9} {'(nu,nubar)':>16}")
        for s in scales:
            d = we_diff(s, E)
            mn = d["numu"]; ne = d["nue"]
            print(f"{s:7.1f} | {mn[2]:10.2f} ({mn[0]:5.2f},{mn[1]:5.2f}) | "
                  f"{ne[2]:9.2f} ({ne[0]:5.2f},{ne[1]:5.2f})")
        h = honda[E]
        print(f"{'Honda':>7} | {h['numu']:10.2f} {'':>16} | {h['nue']:9.2f}")
    print("\nIf diff grows toward Honda's sign/magnitude with scale -> right")
    print("mechanism, under-tuned. If it saturates far short, or wrong sign ->")
    print("coherent muon-bending shift is NOT the (whole) explanation.")
    print("DIAG_EW_CHARGE_SCALE_DONE")


if __name__ == "__main__":
    main()
