"""Figure 1 -- the back-traced full-IGRF geomagnetic rigidity-cutoff sky map at
Kamioka, R_c(zenith, azimuth) [GV].

Computed on a fine (zenith x azimuth) grid with a fine rigidity scan (so neither
the spatial cells nor the R_c colour show visible binning), cached to
``.cache3d`` so re-plotting is instant. Rendered as a smooth polar sky map
(zenith = radius, 0 at centre = vertical, 90 deg at the rim = horizon; azimuth =
angle, N/E/S/W marked) plus a rectangular panel. The West-low / East-high lobe
is the geomagnetic East-West asymmetry.

Run (from tools/mceq3d, needs a writable .cache3d and network-free IGRF)::

    python make_cutoff_map_fig.py [--n-zen 31 --n-az 61 --n-scan 40 --rebuild]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util  # noqa: F401  (mceq_config import shim, kept for parity)
import os
from datetime import datetime

import numpy as np

LAT, LON = 36.43, 137.31
DATE = datetime(2020, 1, 1)
CACHE = ".cache3d"


def cutoff_grid(n_zen, n_az, n_scan, r_hi=40.0, rebuild=False):
    """R_c[n_zen, n_az] [GV] on a fine sky grid, cached per (grid, scan, r_hi)."""
    import geomag_backtrace as gb

    zen = np.linspace(0.0, 89.0, n_zen)
    az = np.linspace(0.0, 360.0, n_az)
    os.makedirs(CACHE, exist_ok=True)
    key = f"cutfig_{LAT}_{LON}_{DATE.date()}_{n_zen}x{n_az}_s{n_scan}_rhi{r_hi}"
    fp = os.path.join(CACHE, f"{hashlib.md5(key.encode()).hexdigest()[:16]}.npz")
    if os.path.exists(fp) and not rebuild:
        d = np.load(fp)
        return d["zen"], d["az"], d["rc"]
    rc = gb.cutoff_map(LAT, LON, DATE, zen, az, n_scan=n_scan, r_hi=r_hi)
    np.savez(fp, zen=zen, az=az, rc=rc)
    return zen, az, rc


def plot(zen, az, rc, out="geomag_cutoff_map.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # R_c is quantised at ~1 GV by the finite rigidity scan; a light smoothing
    # (periodic in azimuth) removes the residual contour speckle in the flat
    # low-cutoff regions without moving the physics. Gouraud shading then
    # interpolates continuously between the real grid points -- no visible cells.
    try:
        from scipy.ndimage import gaussian_filter
        rcp = np.column_stack([rc, rc, rc])  # pad azimuth periodically
        rcp = gaussian_filter(rcp, sigma=(0.8, 0.8))
        rc = rcp[:, rc.shape[1]:2 * rc.shape[1]]
    except Exception:
        pass

    vmin, vmax = np.floor(rc.min()), np.ceil(rc.max())
    fig = plt.figure(figsize=(12, 5.2))

    # --- polar sky map (zenith = radius, azimuth = angle) ---
    axP = fig.add_subplot(1, 2, 1, projection="polar")
    axP.set_theta_zero_location("N")
    axP.set_theta_direction(-1)  # clockwise: N, E, S, W
    A, Z = np.meshgrid(np.deg2rad(az), zen)
    pc = axP.pcolormesh(A, Z, rc, cmap="turbo", shading="gouraud",
                        vmin=vmin, vmax=vmax)
    axP.set_rlabel_position(135)
    axP.set_xticks(np.deg2rad([0, 90, 180, 270]))
    axP.set_xticklabels(["N", "E", "S", "W"])
    axP.set_title("Rigidity cutoff sky map (Kamioka)\nzenith = radius, "
                  "horizon at rim", pad=14)
    fig.colorbar(pc, ax=axP, label=r"$R_c$ [GV]", pad=0.10, shrink=0.85)

    # --- rectangular panel (azimuth x zenith) ---
    axR = fig.add_subplot(1, 2, 2)
    pc2 = axR.pcolormesh(az, zen, rc, cmap="turbo", shading="gouraud",
                         vmin=vmin, vmax=vmax)
    axR.axvline(90, color="w", lw=0.8, ls=":")
    axR.axvline(270, color="w", lw=0.8, ls=":")
    axR.text(90, 4, "E", color="w", ha="center", va="top", fontsize=9)
    axR.text(270, 4, "W", color="w", ha="center", va="top", fontsize=9)
    axR.set_xlabel("azimuth [deg]  (N=0, E=90, S=180, W=270)")
    axR.set_ylabel("zenith [deg]")
    axR.set_title("Cutoff vs azimuth and zenith")
    fig.colorbar(pc2, ax=axR, label=r"$R_c$ [GV]")

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"saved plot -> {out}  (grid {len(zen)}x{len(az)}, "
          f"R_c range {rc.min():.1f}-{rc.max():.1f} GV)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-zen", type=int, default=31)
    p.add_argument("--n-az", type=int, default=61)
    p.add_argument("--n-scan", type=int, default=40)
    p.add_argument("--rebuild", action="store_true")
    a = p.parse_args(argv)
    zen, az, rc = cutoff_grid(a.n_zen, a.n_az, a.n_scan, rebuild=a.rebuild)
    plot(zen, az, rc)


if __name__ == "__main__":
    main()
