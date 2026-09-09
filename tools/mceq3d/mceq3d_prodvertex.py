"""Production-vertex closure: recompute E_off from an INDEPENDENTLY marched real
cascade, to confirm E_off is the deterministic production-vertex result and not an
artefact of offaxis_mc's own machinery.

Physics (why this is the right vehicle, after three refuted reweighting shortcuts):
neutrinos free-stream, so the only 3D effect is production GEOMETRY -- a neutrino
arriving from zenith theta was made by an isotropic primary from within the
production cone, and in the curved atmosphere the cone reaches primaries of
different slant depth (younger near the horizon -> more sub-GeV production). That
is exactly the production-vertex cone integral

    Phi_3D(theta,E) = int dl rho <p(X_slant(P,psi_p),E)>_cone,
    E_off = Phi_3D / Phi_1D   (Phi_1D = collinear cone -> 0).

The independence here: p(X,E) -- the local neutrino production per slant depth --
is taken from `mceq3d_real.march_profile` (our own hand-march of MCEq's real
matrices), NOT from offaxis_mc's separate MCEqRun.solve. The curved-atmosphere
geometry (slant_depth_table) and the moment cone are shared. If this reproduces the
delivered E_off, the excess is confirmed to emerge from the real cascade production
+ curved geometry, independent of the offaxis_mc implementation.

Validation gates:
  (1) cone width -> 0  =>  E_off -> 1 exactly (collinear reduction);
  (2) high energy      =>  E_off -> 1 (cone vanishes);
  (3) sub-GeV          =>  reproduces the delivered E_off and the Honda-implied excess.

Run from tools/mceq3d (PYTHONPATH=$PWD).
"""
import numpy as np

import offaxis_mc as om
from offaxis_mc import slant_depth_table, cone_numden, _rho_of_h
from mceq3d_real import MCEqCascade3D
from kinematic_kernel import channel_shapes


def eoff_from_marched_p(cz, e, sig_deg, x_grid, p_march, geom):
    """E_off(cz,E) via the production-vertex cone (moment/Gaussian) over the marched
    production profile p_march(X,E). Gaussian cone (alpha_w=None) -> the delivered
    'moments' choice."""
    out = np.ones((len(cz), len(e)))
    for i, c in enumerate(cz):
        num, den = cone_numden(c, e, sig_deg, x_grid, e, p_march, geom,
                               n_alpha=44, n_beta=18, alpha_w=None)
        out[i] = np.where(den > 0, num / np.maximum(den, 1e-300), 1.0)
    return out


def main():
    casc = MCEqCascade3D(e_min=0.3)
    e = casc.e
    print("marching real cascade (vertical) for depth-resolved p(X,E) ...")
    Xrec, Prec = casc.march_profile(0.0, pdg=14, n_rec=80)
    # local production per slant depth p(X,E) = dPhi/dX (>=0)
    p_march = np.clip(np.gradient(Prec, Xrec, axis=0), 0.0, None)
    sel = (e >= 0.1) & (e <= 100.0)
    e = e[sel]
    p_march = p_march[:, sel]

    om._RHO = _rho_of_h(casc.mceq.density_model)
    geom = slant_depth_table(om._RHO)
    sig_pi = channel_shapes(e)["pi"]              # moment cone (delivered choice)
    cz = np.round(np.arange(0.05, 1.0, 0.1), 2)

    print("computing E_off from the marched cascade + production-vertex cone ...")
    Eoff_march = eoff_from_marched_p(cz, e, sig_pi, Xrec, p_march, geom)
    print("GATE 1: cone width -> 0 must give E_off -> 1 ...")
    Eoff_zero = eoff_from_marched_p(cz, e, sig_pi * 1e-3, Xrec, p_march, geom)

    # delivered E_off + Honda-implied
    from mceq3d_flux import MCEq3DFlux, EOFF_TABLE_FLAT
    eng = MCEq3DFlux(base_model="mceq", daemonflux_location="generic")
    # the closure target is the FLAT single-pion-cone table: this script's own
    # integral uses one pion cone, while the default table
    # (offaxis_excess_channel_v2.npz) is channel-weighted and species-resolved.
    Edeliv = eng.offaxis_factor(cz, path=EOFF_TABLE_FLAT)["total_numu"]
    de = eng.e
    h = dict(np.load("honda_kam.npz"))
    He, Hcz, nm = h["E"], h["czlo"], h["numu"]

    ih, iv = 0, len(cz) - 1
    print(f"\nGATE 1 (cone->0): max|E_off-1| = {np.nanmax(np.abs(Eoff_zero-1)):.2e}"
          "  (should be ~0)")
    print("\nGATE 3: marched-cascade E_off vs delivered E_off (numu horizon):")
    print(f"{'E':>5} {'march':>7} {'deliv':>7} {'m/d':>6} | {'HondaHV/baseless':>16}")
    for E in (0.3, 0.5, 1.0, 2.0, 3.0):
        je = int(np.argmin(np.abs(e - E)))
        m = Eoff_march[ih, je] / Eoff_march[iv, je]      # horizon/vertical of march
        d = float(np.interp(E, de, Edeliv[ih]) / np.interp(E, de, Edeliv[iv]))
        # Honda excess over its own vertical (a rough independent target on E_off)
        a = int(np.argmin(np.abs(Hcz - (cz[ih]-0.05))))
        b = int(np.argmin(np.abs(Hcz - (cz[iv]-0.05))))
        hv = np.interp(E, He, nm[a].mean(0))/np.interp(E, He, nm[b].mean(0))
        print(f"{E:5.1f} {m:7.3f} {d:7.3f} {m/max(d,1e-9):6.3f} | {hv:16.3f}")
    print("\nGATE 2: E_off -> 1 at high E (marched):",
          f"E_off(10GeV, horizon)={np.interp(10.0, e, Eoff_march[ih]):.3f}")
    print("MCEQ3D_PRODVERTEX_DONE")


if __name__ == "__main__":
    main()
