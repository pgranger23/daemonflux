"""Verify the three E_off approximations flagged in the global review.

1. **p(X,E) zenith-independence** -- the depth-resolved production profile is
   computed once at the vertical; the rho(X)-dependent meson decay-vs-interaction
   competition differs along inclined columns. Rebuild p at theta=60/85 deg and
   recompute E_off.
2. **Cutoff-independence (site-independence)** -- p is built with uncut
   primaries; a high-cutoff site (Kamioka ~11.3 GV, A/Z=1 upper bound) shifts the
   sub-GeV profile. Rebuild p with the rigidity-cut primary and recompute E_off.
3. **Numerical convergence** -- double the cone quadrature (n_alpha, n_beta).

Each prints the max relative E_off deviation over 0.2-10 GeV at the horizon and a
mid zenith; deviations should be << the NA61 +-8% kernel systematic. Run::

    python verify_offaxis.py
"""

from __future__ import annotations

import numpy as np

import offaxis_mc as ox
CZ = np.array([0.05, 0.55])


def _eoff(x_grid, ep_grid, p, geom, n_alpha=44, n_beta=18):
    return ox.offaxis_excess(CZ, x_grid, ep_grid, p, geom, n_alpha, n_beta)


def _maxdev(a, b, e, lo=0.2, hi=10.0):
    m = (e >= lo) & (e <= hi)
    return float(np.max(np.abs(b[:, m] / a[:, m] - 1.0)))


def main():
    print("[ref] vertical, uncut p(X,E) ...")
    x0, e0, p0, dm = ox.production_profile()
    ox._RHO = ox._rho_of_h(dm)
    geom = ox.slant_depth_table(ox._RHO)
    ref = _eoff(x0, e0, p0, geom)

    print("[1] zenith-dependence of p(X,E):")
    for th in (60.0, 85.0):
        xg, eg, pg, _ = ox.production_profile(theta_deg=th)
        dev = _maxdev(ref, _eoff(xg, eg, pg, geom), e0)
        print(f"    theta={th:4.0f} deg : max |dE_off/E_off| = {dev * 100:.1f}%")

    print("[2] rigidity-cutoff dependence of p(X,E):")
    xg, eg, pg, _ = ox.production_profile(rc_cut_gv=11.3)
    dev = _maxdev(ref, _eoff(xg, eg, pg, geom), e0)
    print(f"    R_c=11.3 GV : max |dE_off/E_off| = {dev * 100:.1f}%")

    print("[3] cone-quadrature convergence:")
    dev = _maxdev(ref, _eoff(x0, e0, p0, geom, n_alpha=88, n_beta=36), e0)
    print(f"    (n_alpha,n_beta) x2 : max |dE_off/E_off| = {dev * 100:.2f}%")
    print("done. (Compare against the NA61 kernel systematic, +-8% sub-GeV.)")


if __name__ == "__main__":
    main()
