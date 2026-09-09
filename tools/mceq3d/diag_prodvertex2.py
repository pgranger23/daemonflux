"""Task-5 experiment #2: does spreading the PARENT mesons/muons at the production
vertex (cone_target="parents") reproduce the E_off horizon excess?

Experiment #1 (diag_prodvertex.py) refuted spreading the neutrino ARRIVAL flux
(both row- and col-norm). The production-vertex physics says the cone must act on
the parents before they decay: near the horizon the younger, more-vertical columns
carry more low-E parents, and if they feed direction i's neutrino PRODUCTION the
horizon is enhanced. This runs the real-matrix curved cascade with cone_target=
"parents" (col-norm parent spread at each checkpoint) and compares the emergent
Phi_parents/Phi_nocone to the delivered E_off and Honda. Honest test.

Run from tools/mceq3d (PYTHONPATH=$PWD).
"""
import importlib.util  # noqa: F401
import time

import numpy as np

from mceq3d_real import MCEqCascade3D
from fokker_planck_3d import load_theta2, sigma_theta_vs_energy


def main():
    casc = MCEqCascade3D(e_min=0.3)
    numu = casc._slice(14)
    e = casc.e
    cz = np.array([0.1, 0.15, 0.22, 0.32, 0.45, 0.6, 0.8, 1.0])
    naz = 12
    az = (np.arange(naz) + 0.5) * 2 * np.pi / naz
    CZ, AZ = np.meshgrid(cz, az, indexing="ij")
    zen = np.degrees(np.arccos(CZ)).ravel()
    TH, PH = np.radians(zen), AZ.ravel()
    nd = TH.size
    dcz = np.gradient(cz)
    W = (np.abs(dcz)[:, None] * (2 * np.pi / naz) * np.ones((1, naz))).ravel()
    sig = sigma_theta_vs_energy(*load_theta2("m_spliced.npz"), e, zenith_deg=0.0)
    phi0 = np.repeat(casc.mceq_primary()[None, :], nd, axis=0)

    print(f"marching {nd} directions: cone off / parents ...")
    t = time.time()
    off = casc.march_checkpoints(phi0, zen, (TH, PH), n_check=14)
    par = casc.march_checkpoints(phi0, zen, (TH, PH), n_check=14,
                                 cone_sigma_deg=sig, weights=W,
                                 cone_target="parents")
    print(f"  ... done in {time.time()-t:.0f}s")

    def az_avg(f):
        return f[:, numu].reshape(len(cz), naz, len(e)).mean(1)

    Foff, Fpar = az_avg(off), az_avg(par)
    from mceq3d_flux import MCEq3DFlux, EOFF_TABLE_FLAT
    eng = MCEq3DFlux(base_model="mceq", e_min=0.3)
    # the closure target is the FLAT single-pion-cone table: this script's own
    # integral uses one pion cone, while the default table
    # (offaxis_excess_channel_v2.npz) is channel-weighted and species-resolved.
    Edeliv = eng.offaxis_factor(cz, path=EOFF_TABLE_FLAT)["total_numu"]
    de = eng.e

    print("\nEmergent parents-cone factor vs delivered E_off:")
    print(f"{'E':>5} {'cosZ':>5} {'parents':>8} {'E_off':>7}")
    for E in (0.3, 0.5, 1.0):
        je = int(np.argmin(np.abs(e - E)))
        for icz in (0, 2, 4):
            pr = Fpar[icz, je] / max(Foff[icz, je], 1e-300)
            dv = float(np.interp(E, de, Edeliv[icz]))
            print(f"{E:5.1f} {cz[icz]:5.2f} {pr:8.3f} {dv:7.3f}")
    print("\nconservation of parents operator (<par/off>_Omega, ~1 expected):")
    for E in (0.3, 0.5, 1.0):
        je = int(np.argmin(np.abs(e - E)))
        rat = Fpar[:, je] / np.maximum(Foff[:, je], 1e-300)
        print(f"  E={E:4.1f}  {np.sum(rat*np.abs(dcz))/np.sum(np.abs(dcz)):.3f}")
    print("DIAG_PRODVERTEX2_DONE")


if __name__ == "__main__":
    main()
