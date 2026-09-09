"""Out-of-sample zenith dependence of the East-West asymmetry (Kamioka, numu).

The production-cone-averaged geomagnetic cutoff (`cone_cutoff`, Section 4.1 of
the paper) was diagnosed against the E-W amplitude at a single zenith (~75 deg).
Because the fix (the 40 GV rigidity ceiling + the cone average) was introduced
while addressing that discrepancy, this script validates it **out of sample**:
it compares the matched-zenith E-W amplitude against Honda's own cos(zenith)
bins at several *other* zeniths, at a few energies.

Finding (matched bins): the amplitude has the correct sign, peak and >10 GeV
vanishing, and overshoots Honda by ~5-6% at 63 deg and ~8-11% at 69 deg, growing
toward the extreme horizon. With the **limb-continuous full-sphere cutoff map**
(finemap_rc full_sphere=True, the geomagnetic analogue of extending the cone
across the horizon; paper Section 6, limitation v), the near-horizon production
cone no longer clamps at the limb but samples the far-side up-going cutoff
continuously, pulling the worst bins toward Honda: the extreme-horizon overshoot
(87 deg) falls from ~25-29% to ~20-25% and the 81 deg overshoot from ~18-21% to
~18%, largest where the cone crosses the limb most (lowest energy). Moderate
zeniths (63/69 deg, cone does not reach the limb) are unchanged. A residual
overshoot remains from the intrinsically sharp horizon E-W contrast. The
comparison is apples-to-apples: Honda's max/min over azimuth vs this work's W/E,
both in the same cos(zenith) bin.

Run (from tools/mceq3d, needs honda_kam.npz and a writable .cache3d)::

    python validate_ew_zenith.py
"""

from __future__ import annotations

import importlib.util  # noqa: F401  (mceq_config import shim)
from datetime import datetime

import numpy as np

from mceq3d_flux import MCEq3DFlux

CACHE = ".cache3d"
LAT, LON = 36.43, 137.31
# Epoch matches the engine default and validate_honda.py so both validation
# scripts share one cached full-IGRF cutoff map (the 5-month IGRF drift is
# negligible: the non-limb-crossing bins are identical between epochs).
DATE = datetime(2020, 1, 1)
# Honda down-going cosZ bins away from the ~0.25 (75 deg) diagnosis band
TARGETS = (0.05, 0.15, 0.35, 0.45)
PROBE_E = (0.5, 1.0, 2.0)


def main():
    eng = MCEq3DFlux(base_model="hybrid", primary=("GlobalSplineFitBeta", None),
                     daemonflux_location="kamioka")
    h = dict(np.load("honda_kam.npz"))
    He, Hcz, nm = h["E"], h["czlo"], h["numu"]  # numu[cz, az, E]

    cz = np.array(TARGETS)
    az = np.array([90.0, 270.0])  # E, W
    # bare call: offaxis / cone_cutoff / joint_cone / joint_channels /
    # cone_kernel="moments" (v2 moments) / sublimb="prod_point" are solve()'s
    # own defaults since 2026-09-04.
    r = eng.solve(LAT, LON, cz, az, use_cache=True, cache_dir=CACHE, date=DATE)
    e = r["e"]
    probe = np.array(PROBE_E)

    print("Out-of-sample E-W (W/E) vs Honda (max/min over az), matched cosZ bins:")
    print("  cosZ  zen[deg]  E[GeV]   Honda   thiswork   dev")
    for i, czt in enumerate(TARGETS):
        ih = int(np.argmin(np.abs(Hcz - czt)))
        ewh = nm[ih].max(0) / nm[ih].min(0)
        we = r["flux"]["total_numu"][i, 1] / r["flux"]["total_numu"][i, 0]
        for E in probe:
            hval = float(np.interp(E, He, ewh))
            mval = float(np.interp(E, e, we))
            print(f"  {czt:4.2f}   {np.degrees(np.arccos(czt)):5.1f}   {E:5.1f}   "
                  f"{hval:5.2f}    {mval:5.2f}    {(mval / hval - 1) * 100:+4.0f}%")


if __name__ == "__main__":
    main()
