"""Task-6 diagnostic: does a COUPLED charged-sector march soften the extreme-horizon
E-W overshoot below the factorised cone_geff, or is the overshoot fundamental?

Delivered (factorised cone_geff) overshoots Honda W/E at 87 deg by +15/+19/+24% at
0.5/1/2 GeV. The coupled march applies the REAL IGRF cutoff to each direction's
primary, develops the curved per-direction cascade, spreads the produced neutrinos
over the production cone (row-norm -> azimuthal SMEARING, which is exactly what a
sharp E-W contrast needs -- the opposite of what E_off needed), and rotates the
charged species by the in-cascade Lorentz force. We isolate each ingredient:

  cone OFF, force OFF  -> single-direction cutoff, no smear (sharpest)
  cone ON,  force OFF  -> + production-cone azimuthal smear
  cone ON,  force ON   -> + in-cascade meson/muon bending (the 'small rider')

vs Honda at 87 deg. If 'cone ON' collapses toward Honda, the coupled smearing is the
fix; if even the full coupled march stays near the factorised +15-24%, the overshoot
is not a coupling artefact (hadronic/primary/cutoff-treatment vs Honda).

Run from tools/mceq3d (PYTHONPATH=$PWD). Needs ppigrf (IGRF) + honda_kam.npz.
"""
import importlib.util  # noqa: F401
import time
from datetime import datetime

import numpy as np

from mceq3d_real import MCEqCascade3D
from fokker_planck_3d import load_theta2, sigma_theta_vs_energy
import geomag_backtrace as gb
from muon_bending import local_field_enu

LAT, LON = 36.43, 137.31
DATE = datetime(2020, 1, 1)


def _transmission(e, rc, penumbra=0.5):
    from scipy.special import erf
    return 0.5 * (1 + erf((np.log(np.maximum(e, 1e-9)) - np.log(rc))
                          / (np.sqrt(2) * penumbra)))


def main():
    casc = MCEqCascade3D(e_min=0.3)
    numu = casc._slice(14)
    p_sl, n_sl = casc._slice(2212), casc._slice(2112)
    e = casc.e

    # down-going grid spanning the 87-deg cone; full azimuth ring for the cone smear
    zen_deg = np.array([69.0, 75.0, 81.0, 84.0, 87.0])
    naz = 24
    az_deg = (np.arange(naz) + 0.5) * 360.0 / naz
    print(f"IGRF cutoff map {len(zen_deg)}x{naz} (Kamioka) ...")
    t0 = time.time()
    rc = gb.cutoff_map(LAT, LON, DATE, zen_deg, az_deg, n_scan=32)  # (nzen, naz) [GV]
    print(f"  ... {time.time()-t0:.0f}s;  R_c(87deg) E={rc[-1, 6]:.1f}  W={rc[-1, 18]:.1f} GV")

    ZEN, AZ = np.meshgrid(zen_deg, az_deg, indexing="ij")
    zen = ZEN.ravel(); azr = np.radians(AZ.ravel()); th = np.radians(zen)
    RC = rc.ravel()
    nd = zen.size
    dvec_th, dvec_ph = th, azr
    # solid-angle weights for the cone normalisation
    dcz = np.abs(np.gradient(np.cos(np.radians(zen_deg))))
    Wsa = np.repeat(dcz[:, None], naz, axis=1).ravel() * (2 * np.pi / naz)

    # cutoff-modulated primary per direction (free-proton erf approx, same E/W)
    base0 = casc.mceq_primary()
    phi0 = np.repeat(base0[None, :], nd, axis=0)
    for i in range(nd):
        t = _transmission(e, RC[i])
        # broadcast the per-energy transmission onto the proton/neutron energy blocks
        phi0[i, p_sl] *= t
        phi0[i, n_sl] *= t

    sig = sigma_theta_vs_energy(*load_theta2("m_spliced.npz"), e, zenith_deg=0.0)
    b_enu = local_field_enu(LAT, LON, DATE)
    dirs = (dvec_th, dvec_ph)

    print("marching: cone off/on x force off/on ...")
    t0 = time.time()
    F_00 = casc.march_checkpoints(phi0, zen, dirs, n_check=14)
    F_10 = casc.march_checkpoints(phi0, zen, dirs, n_check=14,
                                  cone_sigma_deg=sig, weights=Wsa, cone_norm="row")
    # force ONLY on cascade-produced mesons/muons (NOT nucleons: their deflection is
    # already in the back-traced cutoff -> bending them again double-counts).
    MESON_MU = {211: +1, -211: -1, 321: +1, -321: -1, 13: -1, -13: +1}
    F_11 = casc.march_checkpoints(phi0, zen, dirs, n_check=14,
                                  cone_sigma_deg=sig, weights=Wsa, cone_norm="row",
                                  b_enu=b_enu, force_species=MESON_MU)
    print(f"  ... {time.time()-t0:.0f}s")

    def we_at(flux, iz):
        # E-W AMPLITUDE = max/min over azimuth (convention-independent, matches how
        # Honda's max(0)/min(0) is formed).
        f = flux[:, numu].reshape(len(zen_deg), naz, len(e))[iz]
        return f.max(0) / np.maximum(f.min(0), 1e-300)

    h = dict(np.load("honda_kam.npz"))
    He, Hcz, nm = h["E"], h["czlo"], h["numu"]
    iz = len(zen_deg) - 1  # 87 deg
    ih = int(np.argmin(np.abs(Hcz - 0.05)))
    hwe = nm[ih].max(0) / nm[ih].min(0)

    print("\nW/E at 87 deg: coupled march vs Honda (delivered cone_geff for ref):")
    print(f"{'E':>5} {'Honda':>6} {'deliv':>6} | {'coneOFF':>7} {'coneON':>7} {'+force':>7}")
    deliv = {0.5: 2.87, 1.0: 2.48, 2.0: 1.88}
    for E in (0.5, 1.0, 2.0):
        je = int(np.argmin(np.abs(e - E)))
        w00 = we_at(F_00, iz)[je]; w10 = we_at(F_10, iz)[je]; w11 = we_at(F_11, iz)[je]
        hv = float(np.interp(E, He, hwe))
        print(f"{E:5.1f} {hv:6.2f} {deliv[E]:6.2f} | {w00:7.2f} {w10:7.2f} {w11:7.2f}")
    print("\nconeON << coneOFF and -> Honda  => production-cone smear is the fix.")
    print("coneON ~ coneOFF ~ deliv (all overshoot) => overshoot is NOT coupling.")
    print("COUPLED_EW_DONE")


if __name__ == "__main__":
    main()
