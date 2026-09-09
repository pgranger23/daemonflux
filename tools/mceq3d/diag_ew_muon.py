"""Task-9: is the missing E-W softening the muon-decay channel? Decompose the
delivered cone_geff W/E at 87 deg by turning the muon-decay ingredients on/off, and
probe the sensitivity to the cone width -- all with the real engine machinery.

The delivered cone_geff blends a wider muon-decay cone (channel_cone) and a charge-
signed muon-bending shift onto the sharp pion cutoff cone. If the muon-decay channel
is the ~7-10 GV of missing softening, then (a) channel_cone should soften W/E a lot,
and (b) a physically-plausible widening should reach Honda (2.51/2.09/1.52). If
channel_cone already saturates well short of Honda, the first-principles muon-decay
smearing is NOT enough -> the overshoot is not (only) the muon channel.

Configs at 87 deg (cosZ=0.05), W/E = numu(W)/numu(E):
  pion-only          channel_cone=False, muon_bending=False   (sharpest)
  +channel cone      channel_cone=True,  muon_bending=False
  delivered          channel_cone=True,  muon_bending=True     (~2.87/2.48/1.88)
  wider cone x1.5    delivered + cone_sigma_scale=1.5          (sensitivity, NOT a fit)
  wider cone x2.0    delivered + cone_sigma_scale=2.0

Run from tools/mceq3d (PYTHONPATH=$PWD). Needs the .cache3d cutoff map.
"""
from datetime import datetime

import numpy as np

from mceq3d_flux import MCEq3DFlux

LAT, LON = 36.43, 137.31
DATE = datetime(2020, 1, 1)
CACHE = ".cache3d"
HONDA = {0.5: 2.51, 1.0: 2.09, 2.0: 1.52}


def main():
    eng = MCEq3DFlux(base_model="mceq", daemonflux_location="kamioka")
    cz = np.array([0.05])
    az = np.array([90.0, 270.0])  # E, W
    e = eng.e

    def we(channel_cone, muon_bending, scale):
        r = eng.solve(LAT, LON, cz, az, offaxis=True, use_cache=True,
                      cone_cutoff=True, cache_dir=CACHE, date=DATE,
                      channel_cone=channel_cone, muon_bending=muon_bending,
                      cone_sigma_scale=scale)
        f = r["flux"]["total_numu"][0]  # (naz, nE)
        return {E: float(np.interp(E, e, f[1] / np.maximum(f[0], 1e-300)))
                for E in HONDA}

    configs = [
        ("pion-only",      dict(channel_cone=False, muon_bending=False, scale=1.0)),
        ("+channel cone",  dict(channel_cone=True,  muon_bending=False, scale=1.0)),
        ("delivered",      dict(channel_cone=True,  muon_bending=True,  scale=1.0)),
        ("wider x1.5",     dict(channel_cone=True,  muon_bending=True,  scale=1.5)),
        ("wider x2.0",     dict(channel_cone=True,  muon_bending=True,  scale=2.0)),
    ]
    print("87 deg W/E vs Honda (2.51/2.09/1.52 at 0.5/1/2 GeV):")
    print(f"{'config':>15} {'0.5GeV':>7} {'1GeV':>6} {'2GeV':>6}")
    for name, kw in configs:
        w = we(kw["channel_cone"], kw["muon_bending"], kw["scale"])
        print(f"{name:>15} {w[0.5]:7.2f} {w[1.0]:6.2f} {w[2.0]:6.2f}")
    print(f"{'Honda':>15} {2.51:7.2f} {2.09:6.2f} {1.52:6.2f}")
    print("\npion->channel gap = how much the muon-decay cone already softens;")
    print("if 'delivered' >> Honda and only 'wider x2' reaches it, the first-principles")
    print("mu-decay width is too narrow (under-weighted). If even x2 overshoots, the")
    print("muon channel is NOT the missing softening.")
    print("DIAG_EW_MUON_DONE")


if __name__ == "__main__":
    main()
