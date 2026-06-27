"""Offline tests for the muon-bending module."""

import numpy as np

from muon_bending import (
    bending_angle,
    bending_deflection,
    coherent_ew_shift_deg,
    decay_length_km,
    decay_in_flight_fraction,
    numu_bending_sigma2,
    numu_ew_asymmetry,
)


def test_coherent_shift_charge_antisymmetric():
    sp = coherent_ew_shift_deg(0.30, charge=+1)
    sm = coherent_ew_shift_deg(0.30, charge=-1)
    assert sp > 0 and np.isclose(sp, -sm)  # mu+ east, mu- west
    assert 2.0 < sp < 5.0  # ~3 deg for B_north ~ 0.3 G


def test_deflection_perpendicular_to_velocity():
    vel = np.array([0.0, 0.0, -1.0])  # downward
    d = bending_deflection(vel, [0.0, 0.3, 0.0], charge=+1)
    assert abs(np.dot(d, vel)) < 1e-12  # perp to velocity
    assert d[0] > 0  # east component positive for mu+ with northward field


def test_ew_asymmetry_decays_with_energy():
    lo = numu_ew_asymmetry(0.3)
    hi = numu_ew_asymmetry(10.0)
    assert lo["charge_separated_deg"] > hi["charge_separated_deg"]
    assert abs(lo["net_shift_deg"]) < lo["charge_separated_deg"]  # R~1 -> small net


def test_bending_angle_linear_in_B_few_degrees():
    a = bending_angle(0.45)
    assert 2.0 < np.degrees(a) < 8.0  # few degrees
    assert np.isclose(bending_angle(0.90), 2 * a)  # linear in B


def test_decay_length_grows_with_energy():
    assert decay_length_km(1.0) < decay_length_km(10.0)
    # ~6 km at 1 GeV
    assert 4.0 < decay_length_km(1.0) < 9.0


def test_decay_fraction_decreases_with_energy():
    e = np.array([0.3, 1.0, 3.0, 10.0, 30.0]) * 3  # E_mu
    f = decay_in_flight_fraction(e)
    assert np.all(np.diff(f) < 0)
    assert f[0] > 0.8 and f[-1] < 0.1  # low-E decays, high-E reaches ground


def test_bending_spread_is_subgev_and_falls():
    enu = np.array([0.3, 1.0, 3.0, 10.0, 30.0])
    s = np.sqrt(numu_bending_sigma2(enu))
    assert np.all(np.diff(s) < 0)  # falls with energy (decay fraction)
    assert np.all(s >= 0)
    assert np.degrees(s[0]) > 2.0  # several degrees sub-GeV
