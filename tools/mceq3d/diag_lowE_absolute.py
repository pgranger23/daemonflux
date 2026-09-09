"""Is there a 'massive' low-energy excess vs Honda in ABSOLUTE flux (not just shape)?

Every validation run so far (offaxis_mc.validate, validate_honda.py) compares
RATIOS -- horizon/vertical shape, E-W amplitude -- which cancel absolute
normalization. This checks the actual delivered ABSOLUTE numu flux (the full
solve(): hybrid base x E_off x cone-averaged geomagnetic cutoff) against Honda's
absolute table, across energy (0.1-3 GeV) and zenith (vertical to horizon), to see
where/how large any absolute excess is and whether it is a BASE effect (below the
muon-calibration floor) or an E_off effect (the off-axis cone at its most
extrapolated, near 0.1-0.15 GeV).

Run from tools/mceq3d (PYTHONPATH=$PWD). Needs the cached .cache3d/gs_* (should hit
from earlier hybrid/GSF/Kamioka solve() calls this session).
"""
from datetime import datetime

import numpy as np

from mceq3d_flux import MCEq3DFlux

LAT, LON = 36.43, 137.31
DATE = datetime(2020, 1, 1)
CACHE = ".cache3d"
EGRID = np.array([0.11, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])


def main():
    eng = MCEq3DFlux(base_model="hybrid", primary=("GlobalSplineFitBeta", None),
                     daemonflux_location="kamioka")
    h = dict(np.load("honda_kam.npz"))
    He, Hcz, nm = h["E"], h["czlo"], h["numu"]  # numu[cosZ_bin, az, E], az-avg below

    cz = np.array([0.95, 0.45, 0.15, 0.05])  # vertical, mid, near-horizon, horizon
    az = np.array([0.0, 90.0, 180.0, 270.0])
    r = eng.solve(LAT, LON, cz, az, offaxis=True, use_cache=True, cone_cutoff=True,
                 cache_dir=CACHE, date=DATE)
    e = r["e"]
    f = r["flux"]["total_numu"]  # (n_cz, n_az, nE)
    f_azavg = f.mean(1)  # (n_cz, nE)

    def honda_at(czt, E):
        i = int(np.argmin(np.abs(Hcz - (czt - 0.05))))
        return np.exp(np.interp(np.log(E), np.log(He),
                                np.log(np.maximum(nm[i].mean(0), 1e-300))))

    print("ABSOLUTE numu flux: delivered (hybrid base x E_off x cone-avg G) / Honda")
    print(f"{'E':>5} | " + " | ".join(f"cosZ={c:.2f}" for c in cz))
    for E in EGRID:
        row = []
        for i, c in enumerate(cz):
            ours = float(np.interp(E, e, f_azavg[i]))
            hon = float(honda_at(c, E))
            row.append(f"{ours/max(hon,1e-300):6.2f}")
        print(f"{E:5.2f} | " + " | ".join(f"{v:>9}" for v in row))

    print("\nFor reference, also the BASE alone (no E_off, no G) vs Honda vertical,")
    print("to separate a base-extrapolation effect from an E_off effect:")
    base_only = eng.base(np.array([0.95]))["total_numu"][0]
    be = eng.e
    print(f"{'E':>5} {'base/Honda(vert)':>18}")
    for E in EGRID:
        b = float(np.exp(np.interp(np.log(E), np.log(be),
                                   np.log(np.maximum(base_only, 1e-300)))))
        hv = float(honda_at(0.95, E))
        print(f"{E:5.2f} {b/max(hv,1e-300):18.2f}")
    print("\nDIAG_LOWE_ABS_DONE")


if __name__ == "__main__":
    main()
