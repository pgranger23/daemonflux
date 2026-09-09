"""Live delivered-engine solve on Honda's exact down-going grid (10 cosZ x 12 az).

Used only by figure 12.  Bare `MCEq3DFlux(...).solve(...)` -- every Phase-1
default (joint cone, channel mode, v2 moments, sigma_lnR=0, prod_point cutoff
anchor) is taken from the module, nothing is overridden.  Result cached so
`make_phase1_figures.py` never has to re-solve.
"""
import importlib.util  # noqa: F401  (MCEq config touches importlib.util)
import os
import time
from datetime import datetime

import numpy as np

from mceq3d_flux import MCEq3DFlux

LAT, LON = 36.43, 137.31
DATE = datetime(2020, 1, 1)
CACHE = ".cache3d"
SPECIES = ("total_numu", "total_antinumu", "total_nue", "total_antinue")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "grid_delivered.npz")


def main():
    h = dict(np.load("honda_kam.npz"))
    cz = h["czlo"][h["czlo"] >= 0.0] + 0.05          # 0.05 .. 0.95
    az = h["azlo"].astype(float) + 15.0              # 15 .. 345
    eng = MCEq3DFlux(base_model="hybrid",
                     primary=("GlobalSplineFitBeta", None),
                     daemonflux_location="kamioka")
    t = time.time()
    r = eng.solve(LAT, LON, cz, az, use_cache=True, cache_dir=CACHE, date=DATE)
    print(f"solve {time.time() - t:.0f}s", flush=True)
    np.savez_compressed(OUT, e=r["e"], cz=cz, az=az,
                        **{s: r["flux"][s] for s in SPECIES})
    print("GRID_SOLVE_DONE", OUT, flush=True)


if __name__ == "__main__":
    main()
