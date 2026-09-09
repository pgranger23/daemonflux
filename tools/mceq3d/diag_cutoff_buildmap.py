"""Build (and cache) the repaired full-sphere cutoff map, and report its quality.

Builds ``MCEq3DFlux.finemap_rc`` for a site straight into ``.cache3d`` without
instantiating MCEq (the map only needs the back-tracer), then prints:

* saturation and range checks;
* the near-limb East profile;
* the bilinear-interpolation error against direct back-traces, for the new dense
  limb nodes and for the legacy uniform 13-node grid resampled from the same map
  (so the two grids are compared on identical cutoff values).

Run::

    python diag_cutoff_buildmap.py --n-jobs 16
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime

import numpy as np

import geomag_backtrace as gb
from mceq3d_flux import MCEq3DFlux, _zenith_nodes

LAT, LON = 36.43, 137.31
DATE = datetime(2020, 1, 1)
CACHE = ".cache3d"
# direct back-traces (this scanner, tol 0.1 GV) at azimuth 90 deg = from the East
PROBE_ZEN = (81.0, 84.0, 84.75, 85.5, 86.25, 87.0, 87.5, 88.0, 88.25, 89.0, 89.5)


class _NoMCEq:
    """finemap_rc only touches ``self._finemap_memo``."""


def bilinear(zen, az, rc, th, ph):
    th = float(np.clip(th, zen[0], zen[-1]))
    ph = float(ph) % 360.0
    iz = int(np.clip(np.searchsorted(zen, th) - 1, 0, len(zen) - 2))
    ja = int(np.clip(np.searchsorted(az, ph) - 1, 0, len(az) - 2))
    tz = (th - zen[iz]) / (zen[iz + 1] - zen[iz])
    ta = (ph - az[ja]) / (az[ja + 1] - az[ja])
    return (rc[iz, ja] * (1 - tz) * (1 - ta) + rc[iz + 1, ja] * tz * (1 - ta)
            + rc[iz, ja + 1] * (1 - tz) * ta + rc[iz + 1, ja + 1] * tz * ta)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument("--probe", action="store_true",
                   help="also back-trace the probe zeniths directly (slow)")
    args = p.parse_args(argv)

    t0 = time.time()
    zen, az, rc = MCEq3DFlux.finemap_rc(
        _NoMCEq(), LAT, LON, DATE, cache_dir=CACHE, n_jobs=args.n_jobs
    )
    print(f"map built/loaded in {time.time() - t0:.1f} s  shape={rc.shape}")
    print(f"  zenith nodes: {np.array2string(zen, precision=2, max_line_width=100)}")
    n_dn = int(np.sum(zen <= 89.9))
    print(f"  down-going nodes: {n_dn}, R_c range "
          f"{rc[:n_dn].min():.2f}-{rc[:n_dn].max():.2f} GV")
    print(f"  up-going  nodes: {len(zen) - n_dn}, R_c range "
          f"{rc[n_dn:].min():.2f}-{rc[n_dn:].max():.2f} GV")
    print(f"  saturated cells at 55 GV: {int(np.sum(np.isclose(rc, 55.0)))}")
    print(f"  vertical R_c = {rc[0, 0]:.3f} GV  (literature ~11.3-11.5)")

    ie = int(np.argmin(np.abs(az - 90.0)))
    iw = int(np.argmin(np.abs(az - 270.0)))
    print("\n  zenith   R_c(East)  R_c(West)")
    for i in range(len(zen)):
        if zen[i] >= 74.0 and zen[i] <= 96.0:
            print(f"  {zen[i]:6.2f}   {rc[i, ie]:8.2f}   {rc[i, iw]:8.2f}")

    # -- interpolation error, new dense grid vs the legacy uniform grid --
    zl = _zenith_nodes(13, limb_nodes=False)
    zu_leg = np.concatenate([zl, 180.0 - zl[::-1]])
    rc_leg = np.empty((len(zu_leg), len(az)))
    for j in range(len(az)):
        rc_leg[:, j] = np.interp(zu_leg, zen, rc[:, j])

    truth = None
    if args.probe:
        up = gb._local_frame(LAT, LON)[0]
        r0 = np.tile(gb.RE * up, (len(PROBE_ZEN), 1))
        u0 = np.array([-gb.arrival_direction(LAT, LON, z, 90.0)
                       for z in PROBE_ZEN])
        truth, _ = gb.cutoff_from_states(r0, u0, DATE, n_jobs=args.n_jobs)

    print("\n  bilinear interpolation error at azimuth 90 (East) [GV]")
    print("  zenith   direct    dense-grid        legacy-uniform-grid")
    for k, z in enumerate(PROBE_ZEN):
        d = bilinear(zen, az, rc, z, 90.0)
        lg = bilinear(zu_leg, az, rc_leg, z, 90.0)
        if truth is not None:
            print(f"  {z:6.2f} {truth[k]:8.2f} {d:8.2f} ({d - truth[k]:+5.2f}) "
                  f"  {lg:8.2f} ({lg - truth[k]:+5.2f})")
        else:
            print(f"  {z:6.2f}      --- {d:8.2f}          {lg:8.2f} "
                  f"({lg - d:+5.2f} vs dense)")
    print("\nDIAG_CUTOFF_BUILDMAP_DONE")


if __name__ == "__main__":
    main()
