"""Offline tests for kinematic_kernel (generator moments + exact decay)."""

import os

import numpy as np
import pytest

import kinematic_kernel as kk

E = np.array([0.3, 0.5, 1.0, 3.0, 10.0])


def test_meson_theta2_pion_range_and_falloff():
    e, t2 = kk.meson_theta2("pi")
    assert e.min() < 0.5 and e.max() > 1e4  # covers the sub-GeV parents
    assert np.all(t2 > 0)
    # production angle falls steeply with meson energy (~E^-0.9 overall)
    s = np.sqrt(t2)
    assert s[0] > 10 * s[-1]


def test_channel_shapes_monotone_and_sane():
    sh = kk.channel_shapes(E, n=200_000)
    for c in ("pi", "k"):
        assert np.all(np.diff(sh[c]) < 0), f"sigma_{c} must fall with E"
    # sub-GeV pion angle is tens of degrees; >=10 GeV it is ~1 degree
    assert 8.0 < sh["pi"][0] < 40.0
    assert sh["pi"][-1] < 2.0
    # kaons are the wider parent at fixed E_nu
    assert np.all(sh["k"] > sh["pi"])


def test_muon_shape_small_in_calibration_region():
    s = kk.muon_shape(E, n=200_000)
    assert np.all(np.diff(s) < 0)
    # daemonflux calibrates on muons E >~ 5 GeV: the muon angle is small there
    # (~2 deg at 10 GeV, production-angle dominated), which is what makes the
    # muon-calibration closure of E_off hold (build prints ~1.00 vertical).
    assert s[-1] < 3.0


@pytest.mark.skipif(
    not os.path.exists("channel_fractions.npz"), reason="fractions cache absent"
)
def test_channel_fractions_physics():
    fr_mu = kk.channel_fractions(E, flavour="numu")
    fr_e = kk.channel_fractions(E, flavour="nue")
    for fr in (fr_mu, fr_e):
        tot = fr["pi"] + fr["k"] + fr["mu"]
        # pi+K+mu cover ~everything for nu_mu; for nu_e the K0L (K_e3) parent
        # is outside these categories and grows with E (~12% at 10 GeV).
        assert np.all(tot > 0.8) and np.all(tot < 1.05)
    # nu_e is muon-decay dominated sub-GeV (direct pi is helicity-suppressed)
    assert fr_e["mu"][0] > 0.9
    # the nu_mu muon-decay fraction falls with energy
    assert fr_mu["mu"][0] > fr_mu["mu"][-1]
