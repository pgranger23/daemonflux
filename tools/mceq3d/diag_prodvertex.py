"""Task-5 experiment: does a column-normalised (production-vertex) cone reproduce
the off-axis excess E_off that the row-normalised (arrival-smearing) cone misses?

`validate_3d_deterministic.py` showed the row-normalised production-cone spread
nets to ~0.97 (smearing), NOT E_off's sub-GeV horizon excess. Hypothesis: the
excess is a production-vertex effect and the correct operator is column-normalised
-- each production SOURCE distributes its neutrinos over arrival directions, so
near the horizon the domain boundary lets younger, higher-production near-vertical
columns feed the horizon arrival (a conserving redistribution WITH an excess).

This runs the SAME real-matrix, curved-column, checkpoint-coupled cascade on the
same whole-sky grid with cone_norm in {row, col} and compares the emergent factor
Phi_cone/Phi_nocone to the delivered E_off and to Honda. Honest test: col-norm
either reproduces ~1.2-1.35 at 0.3-0.5 GeV horizon (-> production-vertex closure
found) or it does not (-> the normalisation is not the whole story).

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

    print(f"marching {nd} directions, cone off / row / col ...")
    t = time.time()
    off = casc.march_checkpoints(phi0, zen, (TH, PH), n_check=14)
    row = casc.march_checkpoints(phi0, zen, (TH, PH), n_check=14,
                                 cone_sigma_deg=sig, weights=W, cone_norm="row")
    col = casc.march_checkpoints(phi0, zen, (TH, PH), n_check=14,
                                 cone_sigma_deg=sig, weights=W, cone_norm="col")
    print(f"  ... done in {time.time()-t:.0f}s")

    def az_avg(flux):
        return flux[:, numu].reshape(len(cz), naz, len(e)).mean(1)  # (n_cz, nE)

    Foff, Frow, Fcol = az_avg(off), az_avg(row), az_avg(col)

    from mceq3d_flux import MCEq3DFlux, EOFF_TABLE_FLAT
    eng = MCEq3DFlux(base_model="mceq", e_min=0.3)
    # the closure target is the FLAT single-pion-cone table: this script's own
    # integral uses one pion cone, while the default table
    # (offaxis_excess_channel_v2.npz) is channel-weighted and species-resolved.
    Eoff_deliv = eng.offaxis_factor(cz, path=EOFF_TABLE_FLAT)["total_numu"]
    de = eng.e
    h = dict(np.load("honda_kam.npz"))
    He, Hcz, nm = h["E"], h["czlo"], h["numu"]

    iv = len(cz) - 1  # vertical
    print("\nEmergent cone factor (Phi_cone/Phi_nocone) and horizon/vertical shape:")
    print(f"{'E':>5} {'cosZ':>5} | {'row':>6} {'col':>6} {'deliv':>6} |"
          f" {'row H/V':>8} {'col H/V':>8} {'delivHV':>8} {'HondaHV':>8}")
    # conservation of each operator (base-weighted omega avg of the emergent factor)
    for E in (0.3, 0.5, 1.0, 3.0):
        je = int(np.argmin(np.abs(e - E)))
        # delivered H/V of base*E_off needs the base; use Foff (nocone) as base shape
        for icz in (0, 2, 4):
            r = Frow[icz, je] / max(Foff[icz, je], 1e-300)
            c = Fcol[icz, je] / max(Foff[icz, je], 1e-300)
            dv = float(np.interp(E, de, Eoff_deliv[icz]))
            rhv = Frow[icz, je] / Frow[iv, je]
            chv = Fcol[icz, je] / Fcol[iv, je]
            # delivered H/V = (nocone base H/V) * (E_off[icz]/E_off[vert])
            base_hv = Foff[icz, je] / Foff[iv, je]
            dvv = float(np.interp(E, de, Eoff_deliv[iv]))
            dhv = base_hv * dv / max(dvv, 1e-9)
            ih = int(np.argmin(np.abs(Hcz - cz[icz])))
            ivh = int(np.argmin(np.abs(Hcz - 1.0)))
            hhv = (np.interp(E, He, nm[ih].mean(0)) / np.interp(E, He, nm[ivh].mean(0)))
            print(f"{E:5.1f} {cz[icz]:5.2f} | {r:6.3f} {c:6.3f} {dv:6.3f} |"
                  f" {rhv:8.3f} {chv:8.3f} {dhv:8.3f} {hhv:8.3f}")

    # conservation: solid-angle avg of the emergent col factor at fixed E
    print("\nConservation of the col operator (omega-avg of Phi_col/Phi_off, should ~1):")
    for E in (0.3, 0.5, 1.0):
        je = int(np.argmin(np.abs(e - E)))
        rat = Fcol[:, je] / np.maximum(Foff[:, je], 1e-300)
        w = np.abs(dcz)
        print(f"  E={E:4.1f}  <col/off>_Omega = {np.sum(rat*w)/np.sum(w):.3f}")
    print("DIAG_PRODVERTEX_DONE")


if __name__ == "__main__":
    main()
