"""Is the 0.5-1 GeV E_off overshoot the sampled cone-kernel's TAIL shape (fixable)
or the factorised geometry (a real limitation)?

The delivered E_off drives the pion cone with the *sampled* full (x_L,theta)
generator kernel (`pion_alpha_pdf`), which can carry a heavier large-angle tail
than a Gaussian of the same variance -- and near the horizon a heavier tail
reaches further into the blocked/younger region, inflating the excess. We compare,
at the same NA61-anchored variance sigma_pi(E):
  * SAMPLED kernel  (offaxis_excess, the delivered path), vs
  * GAUSSIAN cone   (cone_excess with the same sigma_pi).
Both are first-principles (same variance); the ONLY difference is the angular
SHAPE. If the Gaussian materially reduces the 0.5-1 GeV excess toward the
reference-implied target, the tail is the (fixable) cause; if not, it is the
geometry.

Targets (reference 3D excess / mceq base_HV), from diag_isolate.py:
  0.3 GeV ~1.63 [H] 1.50 [B];  0.5 ~1.27 [H] 1.22 [B];  1.0 ~1.03 [H] 1.06 [B].

Run from tools/mceq3d (PYTHONPATH=$PWD).
"""
import numpy as np

import offaxis_mc as om
from offaxis_mc import (production_profile, slant_depth_table, offaxis_excess,
                        cone_excess, _rho_of_h)


def main():
    print("building production profile + slant geometry (one MCEq solve) ...")
    x_grid, ep_grid, p, dm = production_profile()
    om._RHO = _rho_of_h(dm)
    geom = slant_depth_table(om._RHO)
    cz = np.round(np.arange(0.05, 1.0, 0.1), 2)
    ih, iv = 0, int(np.argmax(cz))

    from kinematic_kernel import channel_shapes
    sig_pi = channel_shapes(ep_grid)["pi"]

    print("computing SAMPLED (delivered) and GAUSSIAN (same variance) E_off ...")
    Eo_samp = offaxis_excess(cz, x_grid, ep_grid, p, geom)           # sampled kernel
    Eo_gaus = cone_excess(cz, sig_pi, x_grid, ep_grid, p["tot"], geom)  # Gaussian

    tgtH = {0.3: 1.63, 0.5: 1.27, 1.0: 1.03}
    tgtB = {0.3: 1.50, 0.5: 1.22, 1.0: 1.06}
    print("\nE_off horizon/vertical ratio: sampled vs Gaussian (same sigma_pi):")
    print(f"{'E':>5} {'sig_pi':>7} {'sampled':>8} {'gauss':>7} | "
          f"{'tgtHonda':>8} {'tgtBartol':>9}")
    for E in (0.3, 0.5, 1.0, 2.0):
        je = int(np.argmin(np.abs(ep_grid - E)))
        s = Eo_samp[ih, je] / Eo_samp[iv, je]
        g = Eo_gaus[ih, je] / Eo_gaus[iv, je]
        th = tgtH.get(E, float("nan"))
        tb = tgtB.get(E, float("nan"))
        print(f"{E:5.1f} {float(np.interp(E, ep_grid, sig_pi)):7.1f} "
              f"{s:8.3f} {g:7.3f} | {th:8.3f} {tb:9.3f}")
    print("\ngauss << sampled toward target => tail shape is the cause (fixable).")
    print("gauss ~ sampled (both overshoot) => factorised geometry (real limit).")
    print("DIAG_CONE_SHAPE_DONE")


if __name__ == "__main__":
    main()
