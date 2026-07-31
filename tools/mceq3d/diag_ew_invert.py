"""Task-8: quantify WHERE we differ from Honda on the E-W. Invert both our delivered
W/E and Honda's, using our cascade-correct G_s and a fixed West cutoff, to the
*effective East cutoff* each implies. Interpretation:
  * inferred R_c^E roughly CONSTANT across energy, but Honda's < ours -> the
    difference is the geomagnetic cutoff VALUE/contrast (our IGRF back-trace at the
    extreme horizon is higher-contrast than Honda's effective cutoff);
  * inferred R_c^E VARIES strongly with energy (no single cutoff reproduces Honda)
    -> the difference is the cascade-response SHAPE G_s(E), not the cutoff value.

Our back-traced 87 deg cutoffs: East 42, West 7.1 GV. Delivered (cone_geff) W/E =
2.87/2.48/1.88; Honda 2.51/2.09/1.52 at 0.5/1/2 GeV. Run from tools/mceq3d.
"""
import numpy as np

RC_W = 7.1
DELIV = {0.5: 2.87, 1.0: 2.48, 2.0: 1.88}
HONDA = {0.5: 2.51, 1.0: 2.09, 2.0: 1.52}
OUR_RC_E = 42.0


def main():
    from mceq3d_flux import MCEq3DFlux
    eng = MCEq3DFlux(interaction_model="SIBYLL23D",
                     primary=("HillasGaisser2012", "H3a"),
                     base_model="mceq", daemonflux_location="kamioka")
    rc_grid = np.linspace(4.0, 60.0, 40)
    G, rc_grid = eng.geomag_response(rc_grid, cache_dir=".cache3d")
    g = G["total_numu"]
    e = eng.e

    def invert_RcE(E, we):
        """effective East cutoff s.t. G_s(RC_W)/G_s(RcE)=we (West + G_s fixed)."""
        je = int(np.argmin(np.abs(e - E)))
        gw = np.interp(RC_W, rc_grid, g[:, je])
        target = gw / we                      # required G_s(RcE)
        col = g[:, je]                        # G_s vs rc (decreasing)
        # invert monotone-decreasing G_s(rc) -> rc
        order = np.argsort(col)
        return float(np.interp(target, col[order], rc_grid[order]))

    print("Effective East cutoff implied by W/E (West + our G_s fixed):")
    print(f"{'E':>5} {'ours_WE':>8} {'RcE(ours)':>10} {'Honda_WE':>9} {'RcE(Honda)':>11}"
          f"  (back-traced RcE={OUR_RC_E:.0f})")
    for E in (0.5, 1.0, 2.0):
        rce_o = invert_RcE(E, DELIV[E])
        rce_h = invert_RcE(E, HONDA[E])
        print(f"{E:5.1f} {DELIV[E]:8.2f} {rce_o:10.1f} {HONDA[E]:9.2f} {rce_h:11.1f}")
    print("\nRead: RcE(ours) recovers the cone-averaged effective East cutoff behind")
    print("our delivered W/E; RcE(Honda) is what Honda's W/E implies with OUR G_s.")
    print("Honda ~const and < ours => cutoff-value/contrast difference.")
    print("Honda varies strongly with E => cascade-response-shape difference.")
    print("DIAG_EW_INVERT_DONE")


if __name__ == "__main__":
    main()
