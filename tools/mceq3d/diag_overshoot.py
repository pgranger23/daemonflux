"""Diagnose the 0.5-1 GeV E_off overshoot: cone width (sigma_pi) or geometry?

E_off matches Honda AND Bartol to ~3% at 0.3 GeV but overshoots both by ~6-14%
at 0.5-1 GeV. Two candidate causes:
  (i)  the cone width sigma_pi(E) is too large in the mid-sub-GeV band (a cone-
       width / energy-dependence problem), or
  (ii) the slant-depth-blocking geometry over-produces the excess there
       (a geometric problem, insensitive to a global cone-width rescale).

Decisive test: rescale sigma_pi by a global factor and see whether ANY single
scale brings 0.5-1 GeV onto the references WITHOUT breaking the 0.3 GeV match.
  * If yes -> cause (i): the sigma_pi(E) shape is off.
  * If no (every scale that fixes 0.5-1 GeV breaks 0.3 GeV) -> cause (ii):
    geometry, not cone width.

Run from tools/mceq3d (PYTHONPATH=$PWD).
"""
import numpy as np

import offaxis_mc as om
from offaxis_mc import production_profile, slant_depth_table, offaxis_excess, _rho_of_h


def main():
    print("building MCEq production profile + slant-depth geometry (one solve) ...")
    x_grid, ep_grid, p, dm = production_profile()
    om._RHO = _rho_of_h(dm)
    geom = slant_depth_table(om._RHO)
    cz = np.round(np.arange(0.05, 1.0, 0.1), 2)
    iv = int(np.argmax(cz))  # vertical bin
    ih = 0                    # horizon bin (cz=0.05)

    # sigma_pi(E) for reference (the delivered cone width)
    from kinematic_kernel import channel_shapes
    sig_pi = channel_shapes(ep_grid)["pi"]

    # base horizon/vertical (sec-theta) from the engine, shared by all scales
    from mceq3d_flux import MCEq3DFlux
    eng = MCEq3DFlux(base_model="mceq", daemonflux_location="generic")
    base = eng.base(cz)["total_numu"]
    base_e = np.array([np.exp(np.interp(np.log(ep_grid), np.log(eng.e),
                       np.log(np.maximum(base[i], 1e-300)))) for i in range(len(cz))])

    # references (H/V of numu)
    import validate_bartol as vb
    h = dict(np.load("honda_kam.npz"))
    He, Hcz, nm = h["E"], h["czlo"], h["numu"]
    Eb, czb, gb = vb.load_bartol("num", "fmin")

    def hHV(E):
        a = int(np.argmin(np.abs(Hcz - (cz[ih] - 0.05))))
        b = int(np.argmin(np.abs(Hcz - (cz[iv] - 0.05))))
        return (np.interp(E, He, nm[a].mean(0)) / np.interp(E, He, nm[b].mean(0)))

    def bHV(E):
        a = int(np.argmin(np.abs(czb - cz[ih])))
        b = int(np.argmin(np.abs(czb - cz[iv])))
        return np.interp(E, Eb, gb[a]) / np.interp(E, Eb, gb[b])

    scales = [0.7, 0.85, 1.0, 1.15]
    Es = [0.3, 0.5, 1.0]
    print("\nsigma_pi(E) [deg]:", {E: round(float(np.interp(E, ep_grid, sig_pi)), 1)
                                   for E in Es})
    print("\nDelivered horizon/vertical shape = base_HV x (E_off[hor]/E_off[vert])")
    print("vs Honda and Bartol, for several global sigma_pi scales:\n")
    header = f"{'E':>5} {'HondaHV':>8} {'BartolHV':>8} |" + "".join(
        f" x{s:<4}" for s in scales)
    print(header)
    for E in Es:
        je = int(np.argmin(np.abs(ep_grid - E)))
        base_hv = base_e[ih, je] / base_e[iv, je]
        cells = []
        for s in scales:
            Eo = offaxis_excess(cz, x_grid, ep_grid, p, geom, sigma_scale=s)
            dHV = base_hv * (Eo[ih, je] / max(Eo[iv, je], 1e-9))
            cells.append(dHV)
        row = f"{E:5.1f} {hHV(E):8.3f} {bHV(E):8.3f} |" + "".join(
            f" {c:5.3f}" for c in cells)
        print(row)
    print("\nRead: find the scale column closest to Honda/Bartol at EACH E.")
    print("Same scale works at all E -> geometry ok, sigma_pi(E) shape is the issue.")
    print("Different scales needed -> the overshoot is geometric, not cone width.")
    print("DIAG_OVERSHOOT_DONE")


if __name__ == "__main__":
    main()
