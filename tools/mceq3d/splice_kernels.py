"""Splice per-energy kernels from different interaction models.

No single hadronic model spans the full energy range: SIBYLL-2.3d has a hard
floor at sqrt(s) = 10 GeV (E_lab ~ 53 GeV), while low-energy models (UrQMD-3.4,
DPMJET-III) cover below it. The standard solution (as in CORSIKA and MCEq) is to
use a low-energy model below a transition energy and a high-energy model above.

Because the Fortran generators wrapped by chromo use global COMMON blocks, only
*one* model can run per Python process. So each model's kernel is generated in a
separate ``kernel_regeneration.py`` run, and this tool merges them: at each
projectile energy it takes the low-energy kernel below ``--transition`` and the
high-energy kernel above. The two inputs must share the x_L (and p_T or theta)
binning; ``proj_energies`` may differ and are concatenated.

Example::

    python kernel_regeneration.py --backend chromo --model UrQMD34  --angular \
        --emin 4   --emax 60    --ne 6 --out k_low.npz
    python kernel_regeneration.py --backend chromo --model Sibyll23d --angular \
        --emin 80  --emax 10000 --ne 6 --out k_high.npz
    python splice_kernels.py k_low.npz k_high.npz --transition 60 --out k_spliced.npz
"""

from __future__ import annotations

import argparse
import numpy as np


def splice(low_path: str, high_path: str, transition_gev: float, out_path: str):
    low = dict(np.load(low_path))
    high = dict(np.load(high_path))
    assert np.allclose(low["xl_edges"], high["xl_edges"]), "x_L grids differ"

    e_low = low["proj_energies"]
    e_high = high["proj_energies"]
    keep_low = e_low < transition_gev
    keep_high = e_high >= transition_gev
    proj = np.concatenate([e_low[keep_low], e_high[keep_high]])
    order = np.argsort(proj)

    def cat(key):
        return np.concatenate([low[key][keep_low], high[key][keep_high]], axis=0)[order]

    if "is_moments" in low:
        # Moments files: concatenate every per-(E_proj, *) array along axis 0.
        assert "is_moments" in high, "cannot splice moments with non-moments"
        out = {
            "is_moments": True,
            "proj_energies": proj[order],
            "xl_edges": low["xl_edges"],
            "xl_centers": low["xl_centers"],
            "theta_mean": cat("theta_mean"),
            "theta_sq": cat("theta_sq"),
            "dndx": cat("dndx"),
            "e_sec": cat("e_sec"),
        }
    else:
        # Kernel files: the secondary axis (p_T or theta) must match.
        axis_key = "theta_edges" if "theta_edges" in low else "pt_edges"
        assert axis_key in high, "low/high kernels use different secondary axes"
        assert np.allclose(low[axis_key], high[axis_key]), f"{axis_key} grids differ"
        out = {
            "kernel": cat("kernel"),
            "proj_energies": proj[order],
            "xl_edges": low["xl_edges"],
            axis_key: low[axis_key],
        }
        if "marginal" in low and "marginal" in high:
            out["marginal"] = cat("marginal")

    np.savez(out_path, **out)
    print(
        f"spliced {keep_low.sum()} low-E ({e_low[keep_low].min():.0f}-"
        f"{e_low[keep_low].max():.0f} GeV) + {keep_high.sum()} high-E "
        f"({e_high[keep_high].min():.0f}-{e_high[keep_high].max():.0f} GeV) "
        f"-> {out_path}  ({len(proj)} energies)"
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("low", help="low-energy kernel .npz (e.g. UrQMD34)")
    p.add_argument("high", help="high-energy kernel .npz (e.g. Sibyll23d)")
    p.add_argument("--transition", type=float, default=60.0, help="splice energy [GeV]")
    p.add_argument("--out", default="kernel_spliced.npz")
    args = p.parse_args(argv)
    splice(args.low, args.high, args.transition, args.out)


if __name__ == "__main__":
    main()
