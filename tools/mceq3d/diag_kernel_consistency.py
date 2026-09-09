"""Root-cause: why is the sampled cone kernel (k_spliced.npz) wider than the
moment sigma_pi (m_spliced.npz)? -> the kernel's coarse theta binning inflates
the RMS of the forward-peaked production angle, growing with energy.

Compares the per-secondary MESON production-angle RMS (before any decay folding):
  * from the 2D kernel k_spliced.npz (finite 0.667-deg theta bins, bin centres), vs
  * from the exact moment m_spliced.npz (theta=arctan(pT/pL) per secondary, no bins).
The kernel is systematically wider, the ratio growing with E_meson as the true
angle approaches the bin resolution -- a discretisation artefact. The exact moment
is the accurate, NA61-<pT>-validated input, so the delivered cone uses it (offaxis_mc
build cone_kernel="moments", the default).

Run from tools/mceq3d (PYTHONPATH=$PWD). No MCEq needed.
"""
import numpy as np

from fokker_planck_3d import load_theta2


def main():
    k = dict(np.load("k_spliced.npz"))
    pe, xe, te, K = k["proj_energies"], k["xl_edges"], k["theta_edges"], k["kernel"]
    xc, tc = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (te[:-1] + te[1:])
    em, t2 = load_theta2("m_spliced.npz")
    mom_deg = np.degrees(np.sqrt(t2))

    rows = []
    for i in range(len(pe)):
        for j in range(len(xc)):
            Em, w = xc[j] * pe[i], K[i, j].sum()
            if xc[j] < 0.02 or w <= 0 or Em < 0.3 or Em > 200:
                continue
            rows.append((Em, np.sqrt(np.sum(K[i, j] * tc**2) / w)))
    rows = np.array(sorted(rows))

    print(f"kernel theta-bin width = {te[1]-te[0]:.3f} deg (the resolution floor)\n")
    print("meson production-angle RMS: kernel (binned) vs moment (exact)")
    print(f"{'E_meson':>8} {'kernel':>8} {'moment':>8} {'k/m':>6}")
    for Etgt in (1.0, 1.7, 2.5, 3.6, 5.3, 7.8, 11.4, 16.8, 25.0):
        sel = (rows[:, 0] > Etgt * 0.8) & (rows[:, 0] < Etgt * 1.25)
        if not sel.any():
            continue
        kr, mr = np.median(rows[sel, 1]), np.interp(Etgt, em, mom_deg)
        print(f"{Etgt:8.1f} {kr:8.2f} {mr:8.2f} {kr/max(mr,1e-9):6.2f}")
    print("\nk/m > 1 and growing with E => kernel over-wide (binning); moment is exact.")
    print("DIAG_KERNEL_CONSISTENCY_DONE")


if __name__ == "__main__":
    main()
