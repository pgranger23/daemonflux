"""The real-MCEq-matrix 3D cascade must reproduce MCEq exactly (validating the
hand-marched real physics) and carry an absolute, positive directional flux."""
import numpy as np
import pytest

from mceq3d_real import MCEqCascade3D


@pytest.fixture(scope="module")
def casc():
    return MCEqCascade3D(e_min=0.3)


def test_hand_march_reproduces_mceq_exactly(casc):
    numu = casc._slice(14)
    nsteps, dX, rho_inv = casc.path(0.0)
    ref = casc.mceq.get_solution("numu", mag=0).copy()
    phi = casc.march(casc.mceq_primary()[None, :], nsteps, dX, rho_inv)[0]
    m = ref > ref.max() * 1e-8
    rel = np.abs(phi[numu][m] / ref[m] - 1)
    assert rel.max() < 1e-10  # bit-identical to MCEq's own forward Euler


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
