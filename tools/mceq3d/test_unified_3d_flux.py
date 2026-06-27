"""Offline tests for the unified (geomagnetic x angular) 3D composition."""

import numpy as np

from unified_3d_flux import angular_ratio
from daemonflux.geomagnetic import GeomagneticModel


def test_combined_east_west_ratio_equals_geomagnetic():
    # R is azimuth-independent, so it cancels in the East-West ratio: the
    # combined W/E asymmetry must equal the geomagnetic one alone.
    geo = GeomagneticModel("ino")  # equatorial -> strong effect
    E = np.array([1.0])
    g_w = geo.admittance("numuflux", E, 70.0, 270.0)[0]
    g_e = geo.admittance("numuflux", E, 70.0, 90.0)[0]
    R = 0.93  # any azimuth-flat angular factor
    combined_w, combined_e = R * g_w, R * g_e
    assert np.isclose(combined_w / combined_e, g_w / g_e)


def test_west_exceeds_east_at_low_energy():
    geo = GeomagneticModel("kamioka")
    E = np.array([1.0])
    g_w = geo.admittance("numuflux", E, 70.0, 270.0)[0]
    g_e = geo.admittance("numuflux", E, 70.0, 90.0)[0]
    assert g_w > g_e  # positive primaries easier from the West


def test_high_energy_correction_is_unity():
    geo = GeomagneticModel("kamioka")
    g = geo.admittance("numuflux", np.array([1e4]), 70.0, 90.0)[0]
    assert np.isclose(g, 1.0, atol=1e-6)  # cutoff negligible -> 1D recovered


def test_angular_ratio_offline_high_energy_unity():
    # Synthetic 1D flux + validated-style theta2; no MCEq needed.
    e_grid = np.array([0.5, 1.0, 5.0, 50.0, 500.0, 5000.0])
    cos_full = np.linspace(-1.0, 1.0, 101)
    # horizon-peaked flux (even in cos), same for all energies
    shape = np.exp(-((cos_full / 0.3) ** 2))
    flux_full = np.repeat(shape[:, None], len(e_grid), axis=1)
    e_sig = np.logspace(-0.2, 3.7, 30)
    theta2 = (0.3 / e_sig) ** 2
    out_cos = np.linspace(0.1, 1.0, 8)
    R = angular_ratio(e_grid, cos_full, flux_full, e_sig, theta2, out_cos, [0, 5])
    assert R.shape == (len(out_cos), 2)
    # high-energy column (5000 GeV): sigma -> 0 -> R ~ 1
    assert np.allclose(R[:, 1], 1.0, atol=0.02)
    # low-energy column (0.5 GeV): smearing of a peak -> some bins differ from 1
    assert np.any(np.abs(R[:, 0] - 1.0) > 0.02)
