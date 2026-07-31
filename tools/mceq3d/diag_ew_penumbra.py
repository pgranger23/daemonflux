"""Task-8: is the E-W overshoot the cutoff-transition sharpness? Compare the REAL
back-traced penumbra (chaotic allowed/forbidden islands) at 87 deg E/W to the
delivered smooth erf(penumbra=0.5) transmission that G_s uses.

The delivered model turns each direction's single cutoff R_c into a suppression via
_transmission(E, R_c) = 0.5(1+erf((lnR-lnR_c)/(sqrt2 * 0.5))) -- a hand-set width.
The true geomagnetic transition is a *band* of allowed/forbidden rigidity islands
(the penumbra) that Honda's MC samples fully. If the real band is wider than the
erf, our cutoff is too sharp -> E-W contrast too high. This measures both.

Prints, for 87 deg East and West: the main cutoff (highest forbidden R), the full-
transmission rigidity (lowest R above which all allowed), the penumbra width, and
the effective allowed-fraction vs rigidity next to the erf(0.5) it is approximated
by. Run from tools/mceq3d (PYTHONPATH=$PWD). Needs ppigrf.
"""
from datetime import datetime

import numpy as np
from scipy.special import erf

import geomag_backtrace as gb

LAT, LON = 36.43, 137.31
DATE = datetime(2020, 1, 1)


def erf_T(R, Rc, pen=0.5):
    return 0.5 * (1 + erf((np.log(R) - np.log(Rc)) / (np.sqrt(2) * pen)))


def describe(rs, A, name):
    # rs high->low; A allowed fraction (0/1) per rigidity
    order = np.argsort(rs)
    R = rs[order]
    a = A[order]
    forb = R[a < 0.5]
    Rc = forb.max() if forb.size else R.min()       # main cutoff = highest forbidden
    # full transmission: lowest R above which everything is allowed
    allowed_all = R[np.array([a[i:].min() for i in range(len(R))]) > 0.5]
    Rfull = allowed_all.min() if allowed_all.size else R.max()
    # effective penumbra width in ln R, and the erf-0.5 equivalent for reference
    width_lnR = np.log(max(Rfull, 1e-9)) - np.log(max(Rc, 1e-9))
    # erf(0.5) 10-90% spans lnR = 2*sqrt2*0.5*erfinv(0.8) ~ 1.81*0.5 ~ 0.906
    print(f"  {name}: main cutoff R_c={Rc:5.1f} GV, full-transmission R={Rfull:5.1f} GV")
    print(f"       real penumbra width dlnR={width_lnR:.2f} "
          f"(R {Rc:.0f}->{Rfull:.0f}); erf(0.5) 10-90% dlnR=0.91")
    return Rc, Rfull


def main():
    zen = 87.0
    rc, rs, A = gb.cutoff_map(LAT, LON, DATE, [zen], [90.0, 270.0],
                              r_lo=0.5, r_hi=60.0, n_scan=120,
                              return_admittance=True)
    print(f"87 deg penumbra (real back-trace) vs erf(0.5) approximation:")
    describe(rs, A[0, 0], "East")
    describe(rs, A[0, 1], "West")

    # effective allowed-fraction vs R (coarsened) next to erf, East
    print("\nEast: R[GV]  A_real  erf(0.5,Rc=40)")
    order = np.argsort(rs)[::-1]
    Rs, Ae = rs[order], A[0, 0][order]
    for Rt in (55, 48, 42, 38, 34, 30, 25, 20, 15, 10):
        i = int(np.argmin(np.abs(Rs - Rt)))
        print(f"      {Rs[i]:5.1f}   {Ae[i]:.2f}    {erf_T(Rs[i], 40.0):.2f}")
    print("\nIf A_real rises more gradually than erf (allowed islands well below the")
    print("main cutoff), the true cutoff is SOFTER than erf(0.5) -> our G_s over-sharp")
    print("-> W/E too high, growing with E. That is a concrete, first-principles")
    print("'where we differ from Honda' (its MC samples the full penumbra).")
    print("DIAG_EW_PENUMBRA_DONE")


if __name__ == "__main__":
    main()
