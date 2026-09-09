"""Build the repaired cutoff map one hemisphere at a time, with partial caching.

A full-sphere ``finemap_rc`` build is ~60k back-traces and can take over an hour
on a busy machine; ``finemap_rc`` only writes its cache at the very end, so an
interrupted build loses everything.  This driver splits it:

    python diag_cutoff_stage.py --stage down --n-jobs 16   # detector hemisphere
    python diag_cutoff_stage.py --stage up   --n-jobs 16   # far-side hemisphere
    python diag_cutoff_stage.py --stage assemble           # write .cache3d entry

``down`` and ``up`` are independent and can run concurrently; each writes its own
partial ``.npz`` to ``--tmp``.  ``assemble`` feeds the two partials back through
``MCEq3DFlux.finemap_rc`` (with the back-tracer stubbed out) so the cache file
name and contents are exactly what a normal build would have produced.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime

import numpy as np

import geomag_backtrace as gb
import mceq3d_flux as mf
from mceq3d_flux import MCEq3DFlux, _zenith_nodes

LAT, LON = 36.43, 137.31
DATE = datetime(2020, 1, 1)
CACHE = ".cache3d"
N_ZEN, N_AZ = 13, 25


class _NoMCEq:
    """finemap_rc only touches ``self._finemap_memo``."""


def nodes():
    zen_down = _zenith_nodes(N_ZEN, True)
    return zen_down, 180.0 - zen_down[::-1], np.linspace(0.0, 360.0, N_AZ)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=("down", "up", "assemble"))
    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument("--tmp", default=".")
    a = p.parse_args(argv)
    zen_down, zen_up, az = nodes()
    f_dn = os.path.join(a.tmp, "rc_stage_down.npz")
    f_up = os.path.join(a.tmp, "rc_stage_up.npz")

    t0 = time.time()
    if a.stage == "down":
        rc = gb.cutoff_map(LAT, LON, DATE, zen_down, az, n_jobs=a.n_jobs,
                           warn_saturated=False)
        np.savez(f_dn, zen=zen_down, az=az, rc=rc)
        print(f"down done in {time.time() - t0:.0f} s -> {f_dn}  "
              f"range {rc.min():.2f}-{rc.max():.2f} GV, vertical {rc[0, 0]:.3f}")
    elif a.stage == "up":
        rc = mf.farside_cutoff_map(LAT, LON, DATE, zen_up, az, n_jobs=a.n_jobs)
        np.savez(f_up, zen=zen_up, az=az, rc=rc)
        print(f"up done in {time.time() - t0:.0f} s -> {f_up}  "
              f"range {rc.min():.2f}-{rc.max():.2f} GV")
    else:
        d, u = np.load(f_dn), np.load(f_up)
        assert np.allclose(d["zen"], zen_down) and np.allclose(u["zen"], zen_up)
        orig_dn, orig_up = gb.cutoff_map, mf.farside_cutoff_map
        gb.cutoff_map = lambda *aa, **kk: d["rc"]
        mf.farside_cutoff_map = lambda *aa, **kk: u["rc"]
        try:
            zen, azo, rc = MCEq3DFlux.finemap_rc(
                _NoMCEq(), LAT, LON, DATE, cache_dir=CACHE
            )
        finally:
            gb.cutoff_map, mf.farside_cutoff_map = orig_dn, orig_up
        print(f"assembled {rc.shape}; vertical {rc[0, 0]:.3f} GV; "
              f"saturated {int(np.sum(np.isclose(rc, mf.RC_MAX_GV)))}")
    print("DIAG_CUTOFF_STAGE_DONE")


if __name__ == "__main__":
    main()
