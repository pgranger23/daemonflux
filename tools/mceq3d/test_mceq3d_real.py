"""The real-MCEq-matrix 3D cascade must reproduce MCEq exactly (validating the
hand-marched real physics) and carry an absolute, positive directional flux."""
import numpy as np
import pytest

from mceq3d_real import MCEqCascade3D


@pytest.fixture(scope="module")
def casc():
    return MCEqCascade3D(e_min=0.3)


def test_curved_columns_reproduce_mceq_sec_theta(casc):
    # each direction develops down its own slant column -> sec(theta); validated
    # against MCEq's own per-zenith solution (moderate zeniths for speed)
    numu = casc._slice(14)
    zens = [0.0, 70.0]
    phi0 = np.repeat(casc.mceq_primary()[None, :], len(zens), axis=0)
    phi = casc.march_curved(phi0, zens)
    ref = {}
    for z in zens:
        casc.mceq.set_theta_deg(z)
        casc.mceq.solve()
        ref[z] = casc.mceq.get_solution("numu", mag=0).copy()
    e = casc.e
    ihi = int(np.argmin(np.abs(e - 30.0)))
    # curved column reproduces MCEq per zenith
    m = ref[70.0] > ref[70.0].max() * 1e-6
    assert np.abs(phi[1][numu][m] / ref[70.0][m] - 1).max() < 1e-8
    # sec(theta): 70 deg horizon-enhanced over vertical at high E, matching MCEq
    mine_sec = phi[1][numu][ihi] / phi[0][numu][ihi]
    mceq_sec = ref[70.0][ihi] / ref[0.0][ihi]
    assert mine_sec > 1.2
    assert abs(mine_sec / mceq_sec - 1) < 1e-6


def test_directional_march_is_per_direction_and_absolute(casc):
    # multiple directions marched at once, same primary -> identical absolute flux
    numu = casc._slice(14)
    nsteps, dX, rho_inv = casc.path(0.0)
    ref = casc.mceq.get_solution("numu", mag=0).copy()
    phi0 = np.repeat(casc.mceq_primary()[None, :], 4, axis=0)
    phi = casc.march(phi0, nsteps, dX, rho_inv)
    fl = phi[:, numu]
    # each direction reproduces MCEq (absolute), and they are identical
    m = ref > ref.max() * 1e-8
    assert np.abs(fl[0][m] / ref[m] - 1).max() < 1e-10
    assert fl.std(0).max() / max(np.abs(fl).mean(), 1e-30) < 1e-12
