"""Validation of the deterministic 3D cascade first version: exact 1D reduction,
E-W emergence from a cutoff-structured source, and high-energy recovery."""
import numpy as np

from mceq3d_deterministic import solve, _cutoff_ew


def test_reduces_to_1d_without_coupling():
    # no cone, no force, isotropic source -> every direction gives the SAME cascade
    r = solve(cone=False, geomag=False, n_mu=8, n_phi=12, n_x=120)
    fl = r["flux"]
    ie = int(np.argmin(np.abs(r["e"] - 1.0)))
    assert fl[:, ie].std() / fl[:, ie].mean() < 1e-6


def test_flux_positive_and_falls_with_energy():
    r = solve(n_mu=8, n_phi=12, n_x=120)
    fl = r["flux"].mean(0)  # direction-averaged spectrum
    assert np.all(fl >= 0)
    ilo = int(np.argmin(np.abs(r["e"] - 0.5)))
    ihi = int(np.argmin(np.abs(r["e"] - 10.0)))
    assert fl[ilo] > fl[ihi]  # steeply falling spectrum


def test_east_west_emerges_and_vanishes_high_E():
    r = solve(cutoff_gv=_cutoff_ew(), geomag=True, cone=True, n_mu=10, n_phi=16,
              n_x=150)
    TH, PH, e = r["theta"], r["phi"], r["e"]
    horiz = np.cos(TH) < 0.2
    east = horiz & (np.sin(PH) > 0.7)
    west = horiz & (np.sin(PH) < -0.7)

    def we(E):
        je = int(np.argmin(np.abs(e - E)))
        return r["flux"][west, je].mean() / max(r["flux"][east, je].mean(), 1e-30)

    assert we(0.5) > 1.1          # E-W present sub-GeV
    assert we(50.0) < we(0.5)     # weaker at high energy (cutoff -> 0)
