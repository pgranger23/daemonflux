"""Sanity checks for the independent ray-traced coupled neutrino transport
(coupled_3d_transport): the exact altitude-resolved off-axis factor has the right
qualitative behaviour, and the naive single-altitude factorisation degrades near
the horizon (why the delivered engine integrates the cone per altitude)."""
import numpy as np

from coupled_3d_transport import offaxis_exact_vs_factorised
from mceq3d_solver import log_grid
from fokker_planck_3d import load_theta2, sigma_theta_vs_energy


def _run():
    e, dlnE = log_grid(40)
    e_sig, theta2 = load_theta2("m_spliced.npz")
    sig_deg = sigma_theta_vs_energy(e_sig, theta2, e, zenith_deg=0.0)
    cz = np.array([1.0, 0.5, 0.15, 0.08])
    Eex, Efa = offaxis_exact_vs_factorised(cz, e, dlnE, sig_deg)
    return e, cz, Eex, Efa


def test_vertical_is_unity():
    e, cz, Eex, _ = _run()
    # vertical (cz=1): the cone straddles the vertical, off-axis factor ~ 1
    assert abs(Eex[0].mean() - 1.0) < 0.1


def test_high_energy_recovers_1d():
    e, cz, Eex, _ = _run()
    ihi = int(np.argmin(np.abs(e - 30.0)))
    # at 30 GeV the cone is sub-degree -> E_off -> 1 at every zenith
    assert np.all(np.abs(Eex[:, ihi] - 1.0) < 0.05)


def test_subgev_horizon_excess():
    e, cz, Eex, _ = _run()
    ilo = int(np.argmin(np.abs(e - 0.3)))
    # sub-GeV near-horizon: the coupled transport produces a net excess (>1)
    assert Eex[-1, ilo] > 1.2  # cz=0.08, 0.3 GeV


def test_single_altitude_factorisation_degrades_at_horizon():
    e, cz, Eex, Efa = _run()
    ilo = int(np.argmin(np.abs(e - 0.3)))
    err_vert = abs(Eex[0, ilo] / Efa[0, ilo] - 1)
    err_hor = abs(Eex[-1, ilo] / Efa[-1, ilo] - 1)
    # the single-altitude factorisation is fine at vertical, bad at the horizon
    assert err_vert < 0.1
    assert err_hor > 0.2
