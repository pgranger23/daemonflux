"""Build the explicit 3D near-horizon excess correction H(E, cosZ).

The sub-GeV near-horizon flux carries a genuine 3D enhancement (off-axis
production in the curved atmosphere) that the per-zenith cascade + flux-conserving
angular redistribution do not produce: Honda and Bartol (two *independent* full-3D
Monte-Carlos) both show it, agreeing to ~5% (horizon/vertical shape ~1.8 at
0.3 GeV, ->1 by a few GeV). A first-principles deterministic derivation is the 3D-MC
problem; here we model it *explicitly but reference-anchored*: define the
multiplicative shape correction

    H(E, cosZ) = [Honda zenith shape] / [this-work zenith shape],   H(vertical)=1,

so folding H in makes the delivered zenith shape reproduce the full-3D references.
We anchor to Honda and **validate against Bartol** (independent) -> residual ~ the
Honda-Bartol spread (~5%). H is the near-horizon *geometric* excess (approx
site-independent); it is stored per (E, cosZ) at Kamioka and applied to |cosZ|.

Writes ``horizon_excess.npz`` (used by ``solve(horizon_excess=True)``). Run::

    python build_horizon_excess.py
"""

from __future__ import annotations

import numpy as np

import validate_bartol as vb
from mceq3d_flux import MCEq3DFlux


def _at(y, x, E):
    return np.exp(np.interp(np.log(E), np.log(x), np.log(np.maximum(y, 1e-300))))


def main():
    h = dict(np.load("honda_kam.npz"))
    He, Hcz, nm = h["E"], h["czlo"], h["numu"]
    Eb, czb, gb = vb.load_bartol("num", "fmin")

    czc = np.round(np.arange(0.05, 1.0, 0.10), 2)  # Honda down-going centers
    az = np.array([0, 60, 120, 180, 240, 300.0])
    eng = MCEq3DFlux(base_model="daemonflux", daemonflux_location="kamioka")
    r = eng.solve(36.43, 137.31, czc, az, full_3d=True, use_cache=True)
    e = r["e"]
    m = (e >= 0.1) & (e <= 100.0)
    eg = e[m]

    def our(iz):  # az-averaged numu shape vs vertical (last index = 0.95)
        return _at(r["flux"]["total_numu"][iz].mean(0), e, eg)

    def hon(cz):
        ih = int(np.argmin(np.abs(Hcz - (cz - 0.05))))
        return _at(nm[ih].mean(0), He, eg)

    def bar(cz):
        ib = int(np.argmin(np.abs(czb - cz)))
        return _at(gb[ib], Eb, eg)

    iv = len(czc) - 1  # vertical (0.95)
    our_v, hon_v, bar_v = our(iv), hon(0.95), bar(0.95)
    H = np.ones((len(czc), len(eg)))
    for i, cz in enumerate(czc):
        our_shape = our(i) / our_v
        hon_shape = hon(cz) / hon_v
        H[i] = np.clip(hon_shape / np.maximum(our_shape, 1e-6), 0.3, 3.0)

    np.savez("horizon_excess.npz", e=eg, cz=czc, H=H)
    print(f"wrote horizon_excess.npz  (H shape {H.shape}, cosZ {czc})")

    # Independent cross-check: corrected shape (= Honda) vs Bartol near the horizon.
    print("\nCross-validation -- corrected zenith shape vs BARTOL (independent):")
    print("  E[GeV] cosZ  ours+H  Bartol  (both / vertical)")
    for E in (0.3, 0.5, 1.0, 3.0):
        ie = int(np.argmin(np.abs(eg - E)))
        for i, cz in enumerate(czc[:3]):
            corrected = (our(i) / our_v)[ie] * H[i, ie]
            print(
                f"  {E:5.2f} {cz:.2f}  {corrected:5.3f}  "
                f"{(bar(cz) / bar_v)[ie]:5.3f}"
            )


if __name__ == "__main__":
    main()
