"""Flux-conservation diagnostic for the delivered off-axis factor E_off.

Physical premise: 3D geometry only redistributes neutrinos in ARRIVAL DIRECTION;
it cannot change the number or energy of neutrinos produced by a fixed, isotropic
primary flux. Therefore, at each energy E, the solid-angle integral of the true
3D/1D ratio must be ~1 (Honda's sub-GeV horizontal enhancement is a redistribution
toward the horizon, with the all-direction average ~unchanged from 1D).

So the base-weighted, solid-angle-averaged E_off,

    W(E) = < E_off(cosZ,E) >_base = int E_off * Phi_base dcosZ / int Phi_base dcosZ

is the factor by which applying E_off inflates the angle-integrated flux at energy
E. W(E) ~ 1  => E_off redistributes (sound).  W(E) >> 1 => E_off creates flux
(overstated / not flux-conserving).

Run from tools/mceq3d (PYTHONPATH=$PWD).
"""
import numpy as np


def main():
    d = np.load("offaxis_excess.npz")
    cz, e, E_off = d["cz"], d["e"], d["E_off"]  # cz bin centres 0.05..0.95

    from mceq3d_flux import MCEq3DFlux
    eng = MCEq3DFlux(base_model="mceq", daemonflux_location="generic")
    base = eng.base(cz)["total_numu"]  # (n_cz, n_Eeng)
    # interpolate base onto the E_off energy grid, in log-log
    base_e = np.array([
        np.exp(np.interp(np.log(e), np.log(eng.e), np.log(np.maximum(base[i], 1e-300))))
        for i in range(len(cz))
    ])  # (n_cz, nE)

    # down-going hemisphere solid-angle measure: dOmega = dphi dcosZ, phi -> 2pi,
    # so weight is just dcosZ. cz bin centres are uniformly spaced (0.1), so equal.
    dcz = np.gradient(cz)  # ~0.1 each

    print("Flux-conservation check on the delivered E_off (numu, mceq base):")
    print(f"{'E[GeV]':>7} {'W(E)=<Eoff>_base':>17} {'<Eoff>_flat':>12} "
          f"{'Eoff(horiz)':>12} {'Eoff(vert)':>11}")
    for E in (0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 10.0):
        ie = int(np.argmin(np.abs(e - E)))
        eo = E_off[:, ie]
        w = base_e[:, ie]
        W = np.sum(eo * w * dcz) / np.sum(w * dcz)             # base-weighted
        Wflat = np.sum(eo * dcz) / np.sum(dcz)                  # geometric only
        print(f"{E:7.2f} {W:17.4f} {Wflat:12.4f} {eo[0]:12.3f} {eo[-1]:11.3f}")

    print("\nInterpretation:")
    print("  W(E) ~ 1.0  => E_off is a (near) flux-conserving redistribution.")
    print("  W(E) >> 1.0 => E_off inflates the angle-integrated flux at fixed E,")
    print("                 which a fixed isotropic primary flux cannot do")
    print("                 (3D only changes arrival direction, not yield).")
    print("DIAG_EOFF_DONE")


if __name__ == "__main__":
    main()
