"""Follow-up to coupled_ew_charge_diag.py: is the exact force-ON=force-OFF null
result a numerical artefact of too-coarse checkpointing, or real?

Per-checkpoint geomagnetic bending angle at 87 deg is ~100-2000 deg (mod 360),
vastly exceeding the 15-deg (24-point) direction-grid spacing. Each single
checkpoint's rotate-then-snap-to-nearest-direction step is individually exact and
DOES discriminate by charge sign (verified directly), but chaining 14 such large,
lossy rebinning steps could progressively destroy the coherent directional signal
through repeated coarse-grid aliasing -- a resolution artefact, not physics.

Test: rerun with MANY more checkpoints (n_check=140, 10x finer -> per-step angle
~10x smaller) for one energy, cone ON + force ON only, and see if a nonzero
charge-dependent split now survives. If yes -> the earlier null is a resolution
artefact (fixable); if still ~zero -> the coarse-checkpoint hypothesis is wrong too.

UPDATE 2026-09-03: the premise of the original scan was itself wrong twice over.
The "~100-2000 deg per checkpoint" quoted above was a 10x UNIT ERROR in
``_force_checkpoint`` (gauss treated as tesla); the true per-checkpoint rotation
is 10x smaller. And the resample was a nearest-neighbour ``argmax`` pull, so a
lossy map applied 140 times is as destructive as applied 14 times -- the scan
could not have detected the problem it was designed to detect. Both are now
fixed (interpolating k-NN resample, exact in the zero-rotation limit), and the
splitting is measured with the shift-sensitive first-harmonic observables of
``diag_ew_charge_fourier`` instead of the shift-blind ``max/min``.

Run from tools/mceq3d (PYTHONPATH=$PWD).
"""
import time
from datetime import datetime

import numpy as np

from mceq3d_real import MCEqCascade3D
from fokker_planck_3d import load_theta2, sigma_theta_vs_energy
import geomag_backtrace as gb
from muon_bending import local_field_enu
from diag_ew_charge_fourier import (
    ew_observables, geomagnetic_ew_axis, _wrap180,
)

LAT, LON = 36.43, 137.31
DATE = datetime(2020, 1, 1)


def _transmission(e, rc, penumbra=0.5):
    from scipy.special import erf
    return 0.5 * (1 + erf((np.log(np.maximum(e, 1e-9)) - np.log(rc))
                          / (np.sqrt(2) * penumbra)))


def main():
    casc = MCEqCascade3D(e_min=0.3)
    SP = {"numu": 14, "antinumu": -14}
    sl = {k: casc._slice(v) for k, v in SP.items()}
    p_sl, n_sl = casc._slice(2212), casc._slice(2112)
    e = casc.e

    zen_deg = np.array([87.0])
    naz = 24
    az_deg = (np.arange(naz) + 0.5) * 360.0 / naz
    print("IGRF cutoff (single zenith, reuse geometry) ...")
    t0 = time.time()
    rc = gb.cutoff_map(LAT, LON, DATE, zen_deg, az_deg, n_scan=32)
    print(f"  ... {time.time()-t0:.0f}s")

    ZEN, AZ = np.meshgrid(zen_deg, az_deg, indexing="ij")
    zen = ZEN.ravel(); azr = np.radians(AZ.ravel()); th = np.radians(zen)
    RC = rc.ravel()
    nd = zen.size
    dcz = np.abs(np.gradient(np.cos(np.radians(zen_deg)))) if len(zen_deg) > 1 \
        else np.array([1.0])
    Wsa = np.repeat(dcz[:, None], naz, axis=1).ravel() * (2 * np.pi / naz)

    base0 = casc.mceq_primary()
    phi0 = np.repeat(base0[None, :], nd, axis=0)
    for i in range(nd):
        t = _transmission(e, RC[i])
        phi0[i, p_sl] *= t
        phi0[i, n_sl] *= t

    sig = sigma_theta_vs_energy(*load_theta2("m_spliced.npz"), e, zenith_deg=0.0)
    b_enu = local_field_enu(LAT, LON, DATE)
    dirs = (th, azr)
    MESON_MU = {211: +1, -211: -1, 321: +1, -321: -1, 13: -1, -13: +1}

    ew_axis = geomagnetic_ew_axis()

    def obs(F, species, E):
        f = F[:, sl[species]].reshape(len(zen_deg), naz, len(e))[0]
        v = np.array([np.interp(E, e, f[j]) for j in range(naz)])
        return ew_observables(v, az_deg, ew_axis)

    for n_check in (14, 42, 140):
        print(f"\nmarching n_check={n_check} (cone ON + force ON) ...")
        t0 = time.time()
        F = casc.march_checkpoints(phi0, zen, dirs, n_check=n_check,
                                   cone_sigma_deg=sig, weights=Wsa, cone_norm="row",
                                   b_enu=b_enu, force_species=MESON_MU)
        print(f"  ... {time.time()-t0:.0f}s")
        for E in (0.5, 1.0):
            a, b = obs(F, "numu", E), obs(F, "antinumu", E)
            print(f"  E={E:.1f} GeV  numu-antinumu: d(a1/a0)="
                  f"{a['a1_rel']-b['a1_rel']:+.4f}  d(dphi)="
                  f"{_wrap180(a['dphi']-b['dphi']):+.2f} deg  d(s1/a0)="
                  f"{a['s1_rel']-b['s1_rel']:+.4f}  d(max/min)="
                  f"{a['maxmin']-b['maxmin']:+.4f}")

    print("\nIf diff grows (in magnitude, away from 0) as n_check increases,")
    print("the coarse-checkpoint hypothesis is CONFIRMED (fixable resolution issue).")
    print("If diff stays ~0 regardless of n_check, something else is going on.")
    print("DIAG_CHECKPOINT_RES_DONE")


if __name__ == "__main__":
    main()
