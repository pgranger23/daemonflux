"""Out-of-sample zenith dependence of the East-West asymmetry (Kamioka, numu).

The production-cone-averaged geomagnetic cutoff (`cone_cutoff`, Section 4.1 of
the paper) was diagnosed against the E-W amplitude at a single zenith (~75 deg).
Because the fix (the 40 GV rigidity ceiling + the cone average) was introduced
while addressing that discrepancy, this script validates it **out of sample**:
it compares the matched-zenith E-W amplitude against Honda's own cos(zenith)
bins at several *other* zeniths, at a few energies.

Finding (matched bins): the amplitude has the correct sign, peak and >10 GeV
vanishing, and overshoots Honda by ~5-6% at 63 deg and ~8-11% at 69 deg, growing
to ~25-29% at the extreme horizon (87 deg). The growth toward the horizon is the
down-going-cone-truncation limitation: the ~20 deg production cone there extends
below the local horizon, where the down-going cutoff map is clamped (paper
Section 6, limitation v). The comparison is apples-to-apples: Honda's max/min
over azimuth vs this work's W/E, both in the same cos(zenith) bin.

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
DATE = datetime(2020, 6, 1)
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
    r = eng.solve(LAT, LON, cz, az, offaxis=True, use_cache=True,
                  cone_cutoff=True, cache_dir=CACHE, date=DATE)
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
