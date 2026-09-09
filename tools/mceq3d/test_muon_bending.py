"""Offline tests for the muon-bending module."""

import numpy as np

from muon_bending import (
    bending_angle,
    bending_deflection,
    coherent_ew_shift_deg,
    coherent_shift_deg,
    decay_length_km,
    decay_in_flight_fraction,
    muon_velocity_enu,
    numu_bending_sigma2,
    numu_ew_asymmetry,
    path_length_km,
    muon_decay_numu_fraction,
)


def test_path_length_grows_with_zenith():
    # vertical -> ~production altitude; near horizon -> hundreds of km (curved)
    assert abs(path_length_km(0.0) - 15.0) < 0.1
    assert path_length_km(60.0) > path_length_km(0.0)
    assert path_length_km(89.0) > 100.0  # long near-horizon slant


def test_muon_decay_fraction_bounded_and_falls():
    # w = f/(1+f) in [0, 0.5], larger at low E (muons decay), ->0 at high E
    w_lo = muon_decay_numu_fraction(0.3)
    w_hi = muon_decay_numu_fraction(100.0)
    assert 0.0 <= w_hi < w_lo <= 0.5
    # near-horizon path is longer -> more decay -> larger weight at fixed E
    assert muon_decay_numu_fraction(3.0, zenith_deg=85.0) > muon_decay_numu_fraction(
        3.0, zenith_deg=0.0
    )


def test_coherent_shift_charge_antisymmetric():
    sp = coherent_ew_shift_deg(0.30, charge=+1)
    sm = coherent_ew_shift_deg(0.30, charge=-1)
    assert sp > 0 and np.isclose(sp, -sm)  # mu+ east, mu- west
    assert 2.0 < sp < 5.0  # ~3 deg for B_north ~ 0.3 G


def test_muon_velocity_enu_downward():
    v = muon_velocity_enu(0.0, 0.0)  # vertical arrival
    assert np.allclose(v, [0, 0, -1])  # travels straight down
    vh = muon_velocity_enu(90.0, 90.0)  # horizontal from east
    assert vh[0] < -0.99 and abs(vh[2]) < 1e-9  # travels toward west, no vertical


def test_general_shift_recovers_vertical_special_case():
    # coherent_shift_deg with vertical muon + pure north field == the old E-W shift
    g = coherent_shift_deg(0.0, 0.0, [0.0, 0.30, 0.0], charge=+1)
    assert np.isclose(g["ew_deg"], coherent_ew_shift_deg(0.30, +1))
    # full field adds a non-zero N-S / direction dependence the vertical case lacks
    full = coherent_shift_deg(60.0, 90.0, [-0.04, 0.30, -0.37], charge=+1)
    assert abs(full["ns_deg"]) > 0.5  # inclined direction: N-S term now present


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
