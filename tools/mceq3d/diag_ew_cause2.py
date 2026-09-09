"""Task-7: quantify how much the production-point displacement (candidate a) softens
the 87 deg W/E, using the cascade-correct G_s.

diag_ew_cause.py: at 87 deg the East cutoff drops 40->33 GV and West 7.1->8.1 GV
when evaluated at the ~925 km displaced production point. Here we map those cutoffs
to W/E = G_s(R_c^W, E)/G_s(R_c^E, E) with the delivered cascade-correct geomagnetic
response, and compare to Honda (2.51/2.09/1.52 at 0.5/1/2 GeV).

Run from tools/mceq3d (PYTHONPATH=$PWD).
"""
import numpy as np


def main():
    from mceq3d_flux import MCEq3DFlux
    eng = MCEq3DFlux(base_model="mceq", daemonflux_location="kamioka")
    e = eng.e
    rc_grid = np.linspace(4.0, 42.0, 24)
    G, rc_grid = eng.geomag_response(rc_grid)  # {species: G[rc, E]}
    g = G["total_numu"]  # (n_rc, nE)

    def gs(rc, E):
        je = int(np.argmin(np.abs(e - E)))
        return np.interp(rc, rc_grid, g[:, je])

    # cutoffs from diag_ew_cause.py (CORRECTED near-root geometry: production point
    # is ~272 km / +-3 deg away, East cutoff only 40 -> 38.5 GV, West ~unchanged).
    det = {"E": 40.0, "W": 7.1}
    prod = {"E": 38.5, "W": 7.1}
    honda = {0.5: 2.51, 1.0: 2.09, 2.0: 1.52}

    print("87 deg W/E from G_s: detector cutoff vs production-point cutoff vs Honda")
    print(f"{'E':>5} {'Honda':>6} {'WE_det':>7} {'WE_prod':>8} "
          f"{'ovr_det':>8} {'ovr_prod':>9}")
    for E in (0.5, 1.0, 2.0):
        we_det = gs(det["W"], E) / max(gs(det["E"], E), 1e-9)
        we_prod = gs(prod["W"], E) / max(gs(prod["E"], E), 1e-9)
        h = honda[E]
        print(f"{E:5.1f} {h:6.2f} {we_det:7.2f} {we_prod:8.2f} "
              f"{(we_det/h-1)*100:+7.0f}% {(we_prod/h-1)*100:+8.0f}%")
    print("\nRESULT (corrected geometry): the production point is only ~272 km /")
    print("+-3 deg away, so the East cutoff drops just 40->38.5 GV and W/E falls only")
    print("~4-5 points (still +23-32% vs Honda). The production-point displacement is")
    print("NOT the cause. With correct cutoffs the overshoot is in the cascade-response")
    print("ratio G_s(Rc_W)/G_s(Rc_E) and GROWS with energy -> the neutrino-production")
    print("spectrum / hadronic-primary difference vs Honda, a genuine model uncertainty")
    print("(penumbra is ~E-flat and cannot produce the growing-with-E pattern).")
    print("DIAG_EW_CAUSE2_DONE")


if __name__ == "__main__":
    main()
