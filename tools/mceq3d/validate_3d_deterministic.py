"""Fine-grid physics validation of the deterministic 3D MCEq: does the coupled
production-cone spread reproduce the off-axis excess E_off, from first principles?

Runs the real-matrix, curved-column, checkpoint-coupled cascade on a whole-sky
grid (refined near the horizon, where the cone reaches younger showers), with the
production cone ON and OFF, and forms the *emergent* off-axis factor

    E_off^det(cosZ, E) = Phi_cone(cosZ, E) / Phi_nocone(cosZ, E)   (azimuth-averaged)

comparing it to the delivered factorised ``offaxis_factor`` and to the excess
implied by Honda. Phi_nocone is the per-direction curved cascade (sec theta, =
MCEq per zenith); the ratio isolates the cone's inter-direction redistribution.

Run (from tools/mceq3d)::

    python validate_3d_deterministic.py
"""
import importlib.util  # noqa: F401
import numpy as np

from mceq3d_real import MCEqCascade3D
from fokker_planck_3d import load_theta2, sigma_theta_vs_energy


def main():
    casc = MCEqCascade3D(e_min=0.3)
    numu = casc._slice(14)
    e = casc.e

    # down-going grid: zenith refined near the horizon, x azimuth. Capped at
    # cosZ=0.1 (~84 deg): the extreme horizon has ~10k cascade steps and dominates
    # the runtime; the off-axis excess is already strong at 84 deg.
    cz = np.array([0.1, 0.15, 0.22, 0.32, 0.45, 0.6, 0.8, 1.0])
    naz = 12
    az = (np.arange(naz) + 0.5) * 2 * np.pi / naz
    CZ, AZ = np.meshgrid(cz, az, indexing="ij")
    zen = np.degrees(np.arccos(CZ)).ravel()
    TH = np.radians(zen)
    PH = AZ.ravel()
    nd = TH.size
    # solid-angle weights (dcosZ x dphi), for the cone normalisation
    dcz = np.gradient(cz)
    W = (np.abs(dcz)[:, None] * (2 * np.pi / naz) * np.ones((1, naz))).ravel()

    sig = sigma_theta_vs_energy(*load_theta2("m_spliced.npz"), e, zenith_deg=0.0)
    phi0 = np.repeat(casc.mceq_primary()[None, :], nd, axis=0)  # uniform primary

    print(f"marching {nd} directions ({len(cz)} zenith x {naz} az), cone off/on ...")
    import time
    t = time.time()
    off = casc.march_checkpoints(phi0, zen, (TH, PH), n_check=14)
    on = casc.march_checkpoints(phi0, zen, (TH, PH), n_check=14,
                                cone_sigma_deg=sig, weights=W)
    print(f"  ... done in {time.time()-t:.0f}s")

    # azimuth-average numu at each cosZ
    def az_avg(flux):
        f = flux[:, numu].reshape(len(cz), naz, len(e))
        return f.mean(1)  # (n_cz, nE)

    Foff, Fon = az_avg(off), az_avg(on)
    Eoff_det = Fon / np.maximum(Foff, 1e-300)  # emergent off-axis factor

    # delivered factorised E_off and Honda-implied excess
    from mceq3d_flux import MCEq3DFlux
    eng = MCEq3DFlux(base_model="mceq", e_min=0.3)
    Eoff_deliv = eng.offaxis_factor(cz)["total_numu"]  # (n_cz, nE_eng)
    de = eng.e
    h = dict(np.load("honda_kam.npz"))
    He, Hcz, nm = h["E"], h["czlo"], h["numu"]

    print("\nEmergent E_off^det vs delivered offaxis_factor vs Honda excess"
          " (horizon/vertical of the shape):")
    print(f"{'E':>6} {'cosZ':>5} {'det':>6} {'deliv':>7} | {'det H/V':>8}"
          f" {'deliv H/V':>9} {'Honda H/V':>9}")
    iv = len(cz) - 1  # vertical index (cz=1.0)
    for E in (0.3, 0.5, 1.0, 3.0):
        je = int(np.argmin(np.abs(e - E)))
        jed = int(np.argmin(np.abs(de - E)))
        for icz in (0, 2, 4):  # horizon, near-horizon, mid
            det = Eoff_det[icz, je]
            dv = float(np.interp(E, de, Eoff_deliv[icz]))
            # horizon/vertical of the full shape (base*Eoff): det uses Fon
            hv_det = Fon[icz, je] / Fon[iv, je]
            hv_dl = (np.interp(E, de, Eoff_deliv[icz]) * 0 + dv)  # shape needs base
            ih = int(np.argmin(np.abs(Hcz - cz[icz])))
            ivh = int(np.argmin(np.abs(Hcz - 1.0)))
            hv_h = (np.interp(E, He, nm[ih].mean(0))
                    / np.interp(E, He, nm[ivh].mean(0)))
            print(f"{E:6.1f} {cz[icz]:5.2f} {det:6.3f} {dv:7.3f} | {hv_det:8.3f}"
                  f" {'-':>9} {hv_h:9.3f}")
    print("\n(det = emergent cone factor; deliv = factorised offaxis_factor;"
          " H/V = horizon/vertical of the delivered directional shape.)")
    print("VALIDATE_3D_DET_DONE")


if __name__ == "__main__":
    main()
