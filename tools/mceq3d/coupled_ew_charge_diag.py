"""Task-10 follow-up: does the REAL coupled transport (not the factorized coherent-
shift approximation) produce Honda's charge-dependent East-West split?

diag_ew_charge_scale.py tested only the FACTORIZED mechanism in cone_geff: a single
small-angle coherent shift (muon_bending.bending_deflection), applied once at the
final cutoff-sampling step, computed from the energy-independent mean bending angle
Delta_phi=qBtau/m. Scaling it up to 30x still only reached ~20% of Honda's charge
split before the small-angle approximation broke down -- but that only rules out
THAT mechanism, not the physics.

The coupled march (mceq3d_real.march_checkpoints) is a materially different, richer
treatment already built and validated this session (tasks 5, 6, 9): it bends EVERY
charged secondary (pi+, pi-, K+, K-, mu+, mu-) by the REAL charge-signed Lorentz
rotation, accumulated over its ACTUAL path length, at EVERY checkpoint along the
curved cascade (up to 14 checkpoints from production altitude to the ground) -- not
one shift applied once. Near the 87-deg horizon the path length is hundreds of km,
so this could plausibly accumulate a much larger charge-dependent effect than the
factorized approximation's single small-angle shift.

This extracts ALL FOUR neutrino species (numu=14, antinumu=-14, nue=12, antinue=-12)
-- not just numu, as coupled_ew_diag.py (task 6) did -- from the SAME coupled march
that already has the charge-signed force wired in (force_species=MESON_MU, the
double-counting bug already fixed), and forms the nu-nubar W/E amplitude difference,
exactly as diag_ew_charge_scale.py did for the factorized model. Two configs:
  cone ON, force OFF  -> expect ~0 charge split (no charge-dependent mechanism at
                          all without the force -- a sanity/consistency check)
  cone ON, force ON   -> the full coupled charged-transport physics

Run from tools/mceq3d (PYTHONPATH=$PWD). Needs ppigrf (IGRF) + honda_kam.npz.
"""
import time
from datetime import datetime

import numpy as np

from mceq3d_real import MCEqCascade3D
from fokker_planck_3d import load_theta2, sigma_theta_vs_energy
import geomag_backtrace as gb
from muon_bending import local_field_enu

LAT, LON = 36.43, 137.31
DATE = datetime(2020, 1, 1)


def _transmission(e, rc, penumbra=0.5):
    from scipy.special import erf
    return 0.5 * (1 + erf((np.log(np.maximum(e, 1e-9)) - np.log(rc))
                          / (np.sqrt(2) * penumbra)))


def main():
    casc = MCEqCascade3D(e_min=0.3)
    SP = {"numu": 14, "antinumu": -14, "nue": 12, "antinue": -12}
    sl = {k: casc._slice(v) for k, v in SP.items()}
    p_sl, n_sl = casc._slice(2212), casc._slice(2112)
    e = casc.e

    zen_deg = np.array([75.0, 81.0, 84.0, 87.0])
    naz = 24
    az_deg = (np.arange(naz) + 0.5) * 360.0 / naz
    print(f"IGRF cutoff map {len(zen_deg)}x{naz} (Kamioka) ...")
    t0 = time.time()
    rc = gb.cutoff_map(LAT, LON, DATE, zen_deg, az_deg, n_scan=32)
    print(f"  ... {time.time()-t0:.0f}s")

    ZEN, AZ = np.meshgrid(zen_deg, az_deg, indexing="ij")
    zen = ZEN.ravel(); azr = np.radians(AZ.ravel()); th = np.radians(zen)
    RC = rc.ravel()
    nd = zen.size
    dcz = np.abs(np.gradient(np.cos(np.radians(zen_deg))))
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

    print("marching: cone ON, force OFF vs ON (all 4 species, real charge-signed"
         " Lorentz bending on pi/K/mu at every checkpoint) ...")
    t0 = time.time()
    F_10 = casc.march_checkpoints(phi0, zen, dirs, n_check=14,
                                  cone_sigma_deg=sig, weights=Wsa, cone_norm="row")
    F_11 = casc.march_checkpoints(phi0, zen, dirs, n_check=14,
                                  cone_sigma_deg=sig, weights=Wsa, cone_norm="row",
                                  b_enu=b_enu, force_species=MESON_MU)
    print(f"  ... {time.time()-t0:.0f}s")

    def we_at(flux, iz, species):
        f = flux[:, sl[species]].reshape(len(zen_deg), naz, len(e))[iz]
        return f.max(0) / np.maximum(f.min(0), 1e-300)

    h = dict(np.load("honda_kam.npz"))
    He, Hcz = h["E"], h["czlo"]
    ih = int(np.argmin(np.abs(Hcz - 0.05)))
    hwe = {"numu": h["numu"][ih].max(0) / h["numu"][ih].min(0),
          "antinumu": h["numubar"][ih].max(0) / h["numubar"][ih].min(0),
          "nue": h["nue"][ih].max(0) / h["nue"][ih].min(0),
          "antinue": h["nuebar"][ih].max(0) / h["nuebar"][ih].min(0)}

    iz87 = len(zen_deg) - 1
    for E in (0.5, 1.0, 2.0):
        je = int(np.argmin(np.abs(e - E)))
        print(f"\n{'='*72}\nE = {E} GeV, 87 deg")
        print(f"{'='*72}")
        for cfg, F in (("cone ON, force OFF", F_10), ("cone ON, force ON", F_11)):
            wn = we_at(F, iz87, "numu")[je]
            wnb = we_at(F, iz87, "antinumu")[je]
            we = we_at(F, iz87, "nue")[je]
            web = we_at(F, iz87, "antinue")[je]
            print(f"  [{cfg}]")
            print(f"    numu: nu={wn:.2f} nubar={wnb:.2f} diff={wn-wnb:+.2f}   "
                  f"nue: nu={we:.2f} nubar={web:.2f} diff={we-web:+.2f}")
        hn = float(np.interp(E, He, hwe["numu"]))
        hnb = float(np.interp(E, He, hwe["antinumu"]))
        hne = float(np.interp(E, He, hwe["nue"]))
        hneb = float(np.interp(E, He, hwe["antinue"]))
        print(f"  [Honda]              numu: nu={hn:.2f} nubar={hnb:.2f} "
              f"diff={hn-hnb:+.2f}   nue: nu={hne:.2f} nubar={hneb:.2f} "
              f"diff={hne-hneb:+.2f}")

    print("\nIf 'force ON' diff >> 'force OFF' diff and approaches Honda's diff,")
    print("the real multi-step coupled transport DOES capture (much more of) the")
    print("charge split -- the factorized coherent-shift approx was simply too")
    print("crude, NOT a structural limit of the physics itself.")
    print("COUPLED_EW_CHARGE_DONE")


if __name__ == "__main__":
    main()
