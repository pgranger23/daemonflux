"""Full before/after of the cone-kernel consistency fix, both flavours, both refs.

Builds E_off with the delivered SAMPLED (x_L,theta) kernel and with the NA61-
validated MOMENT sigma_pi (Gaussian), and reports the delivered horizon/vertical
zenith shape vs Honda AND Bartol for nu_mu and nu_e. Confirms the moment cone
removes the 0.5-1 GeV overshoot without breaking the 0.3 GeV / high-E agreement,
for both flavours. Run from tools/mceq3d (PYTHONPATH=$PWD).
"""
import numpy as np

import offaxis_mc as om
from offaxis_mc import production_profile, slant_depth_table, offaxis_excess, _rho_of_h


def main():
    print("building production profile + slant geometry (one MCEq solve) ...")
    x_grid, ep_grid, p, dm = production_profile()
    om._RHO = _rho_of_h(dm)
    geom = slant_depth_table(om._RHO)
    cz = np.round(np.arange(0.05, 1.0, 0.1), 2)
    ih, iv = 0, int(np.argmax(cz))

    print("computing E_off: sampled kernel vs moment sigma_pi ...")
    Es = offaxis_excess(cz, x_grid, ep_grid, p, geom, cone_kernel="sampled")
    Em = offaxis_excess(cz, x_grid, ep_grid, p, geom, cone_kernel="moments")

    from mceq3d_flux import MCEq3DFlux
    eng = MCEq3DFlux(base_model="mceq", daemonflux_location="generic")
    e = eng.e
    base = eng.base(cz)
    h = dict(np.load("honda_kam.npz"))
    He, Hcz = h["E"], h["czlo"]
    import validate_bartol as vb

    def refHV(nm, E):  # honda-style array (n_cz_bins, ...)
        a = int(np.argmin(np.abs(Hcz - (cz[ih] - 0.05))))
        b = int(np.argmin(np.abs(Hcz - (cz[iv] - 0.05))))
        return np.interp(E, He, nm[a].mean(0)) / np.interp(E, He, nm[b].mean(0))

    for fl, hkey, bkey in (("total_numu", "numu", "num"), ("total_nue", "nue", "nue")):
        b_e = np.array([np.exp(np.interp(np.log(ep_grid), np.log(e),
                        np.log(np.maximum(base[fl][i], 1e-300)))) for i in range(len(cz))])
        Eb, czb, gb = vb.load_bartol(bkey, "fmin")

        def bHV(E):
            a = int(np.argmin(np.abs(czb - cz[ih])))
            b = int(np.argmin(np.abs(czb - cz[iv])))
            return np.interp(E, Eb, gb[a]) / np.interp(E, Eb, gb[b])

        print(f"\n[{fl}] horizon/vertical: delivered vs refs, SAMPLED -> MOMENTS")
        print(f"{'E':>5} {'Honda':>6} {'Bartol':>6} | {'samp':>6} {'mom':>6} |"
              f" {'s/Hon':>6} {'m/Hon':>6} {'s/Bar':>6} {'m/Bar':>6}")
        for E in (0.3, 0.5, 1.0, 3.0):
            je = int(np.argmin(np.abs(ep_grid - E)))
            bhv = b_e[ih, je] / b_e[iv, je]
            ds = bhv * Es[ih, je] / Es[iv, je]
            dm_ = bhv * Em[ih, je] / Em[iv, je]
            H, B = refHV(h[hkey], E), bHV(E)
            print(f"{E:5.1f} {H:6.3f} {B:6.3f} | {ds:6.3f} {dm_:6.3f} |"
                  f" {ds/H:6.3f} {dm_/H:6.3f} {ds/B:6.3f} {dm_/B:6.3f}")
    print("\n(s/Hon,m/Hon = delivered/Honda for sampled/moment cone; ~1 is best.)")
    print("DIAG_CONE_FIX_DONE")


if __name__ == "__main__":
    main()
