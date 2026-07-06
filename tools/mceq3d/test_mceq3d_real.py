"""The real-MCEq-matrix 3D cascade must reproduce MCEq exactly (validating the
hand-marched real physics) and carry an absolute, positive directional flux."""
import numpy as np
import pytest

from mceq3d_real import MCEqCascade3D


@pytest.fixture(scope="module")
def casc():
    return MCEqCascade3D(e_min=0.3)


def test_checkpoint_coupling_reduces_exactly_and_couples(casc):
    # Operator-splitting at altitude checkpoints: with coupling OFF it must
    # reproduce MCEq per-zenith (curved columns + sec theta); with the geomagnetic
    # force ON it stays a bounded, self-consistent rider.
    numu = casc._slice(14)
    zens = [0.0, 70.0]
    TH, PH = np.radians(zens), np.array([1.57, 1.57])
    phi0 = np.repeat(casc.mceq_primary()[None, :], len(zens), axis=0)
    chk = casc.march_checkpoints(phi0, zens, (TH, PH), b_enu=None, n_check=12)
    ref = {}
    for z in zens:
        casc.mceq.set_theta_deg(z)
        casc.mceq.solve()
        ref[z] = casc.mceq.get_solution("numu", mag=0).copy()
    e = casc.e
    # reduction: no-coupling checkpoints reproduce MCEq per zenith (exact)
    m = ref[70.0] > ref[70.0].max() * 1e-6
    assert np.abs(chk[1][numu][m] / ref[70.0][m] - 1).max() < 1e-8
    # sec(theta) carried by the curved columns
    ihi = int(np.argmin(np.abs(e - 30.0)))
    assert chk[1][numu][ihi] / chk[0][numu][ihi] > 1.2
    # force ON: applied during the real cascade, stays a bounded small rider
    # (its magnitude needs a fine direction grid + muon tracking to be resolved;
    # on this 2-direction anchor it is grid-limited but must not blow up)
    chf = casc.march_checkpoints(phi0, zens, (TH, PH),
                                 b_enu=np.array([0.0, 0.30, -0.37]), n_check=12)
    scale = np.abs(chk[:, numu]).max()
    assert np.abs(chf[:, numu] - chk[:, numu]).max() / scale < 0.05


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
