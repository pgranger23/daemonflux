"""Does the REAL coupled transport produce Honda's charge-dependent East-West split?

HISTORY / WHAT CHANGED (2026-09-03)
-----------------------------------
The first version of this script reported "an essentially exact null" and that
result went into the paper as a *confirmed structural gap*.  It was not a
measurement.  Three defects, all now fixed:

1. **Blind observable.**  It used ``max/min`` over a uniform 24-point azimuth
   grid.  The charge-dependent muon-bending term is a coherent *rotation* of the
   azimuthal pattern, and ``max/min`` on a grid symmetric about the magnetic
   meridian is *exactly* invariant under a rotation by ``+delta`` vs ``-delta``
   (``test_ew_charge_observable.py``).  We now use the first-azimuthal-harmonic
   observables of ``diag_ew_charge_fourier`` -- amplitude ``a1/a0``, phase
   ``dphi`` relative to geomagnetic West, and the transverse component
   ``s1/a0`` -- which are odd in the shift and linear in it for small shifts.
2. **10x unit error in the rotation.**  ``_force_checkpoint`` built the
   gyroradius with ``1e5`` instead of ``1e6``, i.e. it treated the field in gauss
   as if it were tesla: every in-cascade rotation was 10x too large (~1170 deg
   per checkpoint at 0.3 GeV / 87 deg).  Fixed in ``mceq3d_real.py``.
3. **Non-interpolating resample.**  The rotated field was resampled with an
   ``argmax`` nearest-neighbour *pull*, which is not bijective and cannot
   represent a rotation smaller than the grid spacing.  Replaced by a k-nearest
   inverse-square interpolation that is exact in the zero-rotation limit.

4. **Grid too small.**  75-87 deg x 24 azimuths is 12% of the sky, so any
   rotation pushes flux onto a band edge.  We now march the whole **down-going
   hemisphere**.  (MCEq columns exist only for ``theta < 90``, so the grid is a
   hemisphere, not a sphere: near the limb the rotation still leaks across the
   horizon.  That is a property of the scheme, not of this script.)

Everything else is as before: ``mceq3d_real.march_checkpoints`` bends every
charged secondary (pi+-, K+-, mu+-) by the real charge-signed Lorentz rotation,
accumulated over its actual path length, at every altitude checkpoint of the
curved cascade, with the production-cone inter-direction spread on.  The primary
carries the back-traced IGRF cutoff (interpolated from the engine's own cached
full-sphere map, so the input cutoff is identical to the delivered engine's).

Run from tools/mceq3d (PYTHONPATH=$PWD)::

    python coupled_ew_charge_diag.py                 # default hemisphere grid
    python coupled_ew_charge_diag.py --n-zen 6 --naz 16 --n-check 10   # cheaper
"""
import argparse
import time
from datetime import datetime

import numpy as np

from mceq3d_real import MCEqCascade3D
from fokker_planck_3d import load_theta2, sigma_theta_vs_energy
from muon_bending import local_field_enu
from diag_ew_charge_fourier import (
    ew_observables, geomagnetic_ew_axis, honda_ew, honda_table, use_legacy_map,
    _wrap180,
)

LAT, LON = 36.43, 137.31
DATE = datetime(2020, 1, 1)
CACHE = ".cache3d"
SP = {"numu": 14, "antinumu": -14, "nue": 12, "antinue": -12}
HK = {"numu": "numu", "antinumu": "numubar", "nue": "nue", "antinue": "nuebar"}
PAIRS = (("numu", "numu", "antinumu"), ("nue", "nue", "antinue"))


def _transmission(e, rc, penumbra=0.5):
    from scipy.special import erf
    return 0.5 * (1 + erf((np.log(np.maximum(e, 1e-9)) - np.log(rc))
                          / (np.sqrt(2) * penumbra)))


def _cutoff_from_map(zen_deg, az_deg, cache_dir=CACHE, legacy_map=False):
    """R_c at arbitrary (zenith, azimuth) from the engine's cached IGRF map.

    Using the delivered engine's own ``finemap_rc`` keeps the primary cutoff --
    the dominant driver of the E-W amplitude -- identical between the coupled
    march and the factorised engine, so any difference between them is the
    *transport*, not the cutoff input.
    """
    import contextlib

    from mceq3d_flux import MCEq3DFlux

    eng = MCEq3DFlux(base_model="hybrid", primary=("GlobalSplineFitBeta", None),
                     daemonflux_location="kamioka")
    with (use_legacy_map() if legacy_map else contextlib.nullcontext()):
        zf, af, rf = eng.finemap_rc(LAT, LON, DATE, cache_dir=cache_dir)
    th = np.clip(zen_deg, zf[0], zf[-1])
    ph = np.asarray(az_deg, float) % 360.0
    iz = np.clip(np.searchsorted(zf, th) - 1, 0, len(zf) - 2)
    ja = np.clip(np.searchsorted(af, ph) - 1, 0, len(af) - 2)
    tz = (th - zf[iz]) / (zf[iz + 1] - zf[iz])
    ta = (ph - af[ja]) / (af[ja + 1] - af[ja])
    return (rf[iz, ja] * (1 - tz) * (1 - ta) + rf[iz + 1, ja] * tz * (1 - ta)
            + rf[iz, ja + 1] * (1 - tz) * ta + rf[iz + 1, ja + 1] * tz * ta)


def build_grid(n_zen=10, naz=24):
    """Down-going hemisphere: equal-solid-angle cosZ bands x uniform azimuth.

    Equal-area bands put most directions where the cutoff structure is (they are
    uniform in cosZ), and the last band centre sits at ~87 deg, matching Honda's
    horizon bin.
    """
    cz = (np.arange(n_zen) + 0.5) / n_zen           # band centres in cosZ
    zen_deg = np.degrees(np.arccos(cz))[::-1]       # increasing zenith
    az_deg = (np.arange(naz) + 0.5) * 360.0 / naz
    ZEN, AZ = np.meshgrid(zen_deg, az_deg, indexing="ij")
    w = np.full(ZEN.shape, (1.0 / n_zen) * (2 * np.pi / naz))  # solid angle
    return zen_deg, az_deg, ZEN.ravel(), AZ.ravel(), w.ravel()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-zen", type=int, default=10)
    p.add_argument("--naz", type=int, default=24)
    p.add_argument("--n-check", type=int, default=14)
    p.add_argument("--cache-dir", default=CACHE)
    p.add_argument("--legacy-map", action="store_true")
    args = p.parse_args(argv)

    casc = MCEqCascade3D(e_min=0.3)
    sl = {k: casc._slice(v) for k, v in SP.items()}
    p_sl, n_sl = casc._slice(2212), casc._slice(2112)
    e = casc.e

    zen_deg, az_deg, zen, azd, Wsa = build_grid(args.n_zen, args.naz)
    nd = zen.size
    print(f"grid: {args.n_zen} zenith bands x {args.naz} azimuths = {nd} "
          f"directions over the whole down-going hemisphere")
    print(f"  zenith band centres [deg]: "
          + " ".join(f"{z:.1f}" for z in zen_deg))
    RC = _cutoff_from_map(zen, azd, cache_dir=args.cache_dir,
                          legacy_map=args.legacy_map)
    print(f"  R_c from the engine's cached IGRF map: {RC.min():.1f}-{RC.max():.1f} GV")

    base0 = casc.mceq_primary()
    phi0 = np.repeat(base0[None, :], nd, axis=0)
    for i in range(nd):
        t = _transmission(e, RC[i])
        phi0[i, p_sl] *= t
        phi0[i, n_sl] *= t

    sig = sigma_theta_vs_energy(*load_theta2("m_spliced.npz"), e, zenith_deg=0.0)
    b_enu = local_field_enu(LAT, LON, DATE)
    dirs = (np.radians(zen), np.radians(azd))
    MESON_MU = {211: +1, -211: -1, 321: +1, -321: -1, 13: -1, -13: +1}

    print("marching: cone ON, force OFF vs ON (real charge-signed Lorentz "
          "rotation on pi/K/mu at every checkpoint) ...")
    t0 = time.time()
    F_off = casc.march_checkpoints(phi0, zen, dirs, n_check=args.n_check,
                                   cone_sigma_deg=sig, weights=Wsa,
                                   cone_norm="row")
    print(f"  force OFF: {time.time()-t0:.0f}s", flush=True)
    t0 = time.time()
    F_on = casc.march_checkpoints(phi0, zen, dirs, n_check=args.n_check,
                                  cone_sigma_deg=sig, weights=Wsa, cone_norm="row",
                                  b_enu=b_enu, force_species=MESON_MU)
    print(f"  force ON : {time.time()-t0:.0f}s", flush=True)

    ew_axis = geomagnetic_ew_axis()
    h = honda_table()

    def obs(F, iz, species, E):
        f = F[:, sl[species]].reshape(len(zen_deg), len(az_deg), len(e))[iz]
        v = np.array([np.exp(np.interp(np.log(E), np.log(e),
                                       np.log(np.maximum(f[j], 1e-300))))
                      for j in range(len(az_deg))])
        return ew_observables(v, az_deg, ew_axis)

    # nearest Honda cosZ bin centre for each of our bands we report on
    report_bands = [len(zen_deg) - 1, max(len(zen_deg) - 2, 0)]
    print("\n" + "=" * 78)
    print("CHARGE-DEPENDENT EAST-WEST, coupled march vs Honda")
    print("  a1/a0 = first-harmonic amplitude; dphi = its phase relative to")
    print("  geomagnetic West [deg]; s1/a0 = transverse (shift-odd) component;")
    print("  max/min = the legacy statistic the earlier null was measured with.")
    print("=" * 78)
    for iz in report_bands:
        zdeg = zen_deg[iz]
        # Honda's bins are [k/10, (k+1)/10): FLOOR, not round -- rounding put the
        # 87.1 deg band (cosZ 0.0506) in the 0.15 bin.
        czb = np.floor(np.cos(np.radians(zdeg)) * 10) / 10 + 0.05
        czb = float(min(max(czb, 0.05), 0.95))
        print(f"\n### zenith band {zdeg:.1f} deg  (Honda cosZ bin centre {czb:.2f})")
        for fl, s_nu, s_nb in PAIRS:
            print(f"\n[{fl}]  {'E':>5} {'cfg':>10} | "
                  f"{'a1/a0 nu':>9}{'nubar':>8}{'diff':>8} | "
                  f"{'dphi nu':>8}{'nubar':>8}{'diff':>8} | "
                  f"{'s1 diff':>9} | {'max/min diff':>13}")
            for E in (0.3, 0.5, 1.0, 2.0):
                for cfg, F in (("force OFF", F_off), ("force ON", F_on)):
                    a, b = obs(F, iz, s_nu, E), obs(F, iz, s_nb, E)
                    print(f"{'':>6} {E:5.2f} {cfg:>10} | "
                          f"{a['a1_rel']:9.3f}{b['a1_rel']:8.3f}"
                          f"{a['a1_rel']-b['a1_rel']:+8.3f} | "
                          f"{a['dphi']:8.1f}{b['dphi']:8.1f}"
                          f"{_wrap180(a['dphi']-b['dphi']):+8.1f} | "
                          f"{a['s1_rel']-b['s1_rel']:+9.4f} | "
                          f"{a['maxmin']-b['maxmin']:+13.2f}")
                ha = honda_ew(h, HK[s_nu], czb, E, ew_axis)
                hb = honda_ew(h, HK[s_nb], czb, E, ew_axis)
                print(f"{'':>6} {E:5.2f} {'Honda':>10} | "
                      f"{ha['a1_rel']:9.3f}{hb['a1_rel']:8.3f}"
                      f"{ha['a1_rel']-hb['a1_rel']:+8.3f} | "
                      f"{ha['dphi']:8.1f}{hb['dphi']:8.1f}"
                      f"{_wrap180(ha['dphi']-hb['dphi']):+8.1f} | "
                      f"{ha['s1_rel']-hb['s1_rel']:+9.4f} | "
                      f"{ha['maxmin']-hb['maxmin']:+13.2f}")

    print("\nRead-out: force ON minus force OFF is the whole in-cascade charged")
    print("transport contribution. If it is far below Honda's splitting even with")
    print("the unit and resample fixed and a whole-hemisphere grid, the coupled")
    print("operator-splitting march does NOT carry the effect.")
    print("COUPLED_EW_CHARGE_DONE")


if __name__ == "__main__":
    main()
