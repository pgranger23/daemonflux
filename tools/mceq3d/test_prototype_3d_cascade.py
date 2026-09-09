"""Offline tests for the coupled (E, l) 3D-cascade prototype.

Uses a synthetic theta2(E) so no data file is needed; fast (~tens of ms).
"""

import numpy as np

from prototype_3d_cascade import run_cascade, log_grid


def _synthetic_theta2():
    e = np.logspace(-0.5, 4, 40)
    theta2 = (0.3 / e) ** 2  # rad^2 ; theta1 ~ 0.3/E
    return e, theta2


def test_reduces_to_1d_when_collimated():
    r = run_cascade(moments=_synthetic_theta2(), lmax=40, nsteps=600, collimated=True)
    good = r["c0"] > r["c0"].max() * 1e-6
    assert np.nanmax(r["sigma_theta"][good]) < 1e-9  # angular machinery inert


def test_flux_positive_with_spectral_break():
    r = run_cascade(moments=_synthetic_theta2(), lmax=40, nsteps=800)
    e, c0 = r["e"], r["c0"]
    assert np.all(c0 >= 0)
    # E^2 * dN/dlnE should peak at low E then fall (steepening past eps_pi)
    w = (c0 * e**2)[(e > 0.5) & (e < 1e4)]
    assert w[0] > w[-1]


def test_spread_below_single_production_angle():
    # The coupled solve must give sigma_theta < theta1(E_numu): the neutrino's
    # parent meson is more energetic (more forward) than the neutrino.
    r = run_cascade(moments=_synthetic_theta2(), lmax=60, nsteps=800)
    e, st, th1 = r["e"], r["sigma_theta"], np.degrees(r["theta1"])
    s = (e > 1.0) & (e < 1000) & np.isfinite(st)
    assert np.all(st[s] < th1[s])  # below the single-production angle
    assert np.all(st[s] < np.sqrt(2) * th1[s])  # well below the old N_chain=2 guess


def test_spread_decreases_with_energy():
    r = run_cascade(moments=_synthetic_theta2(), lmax=60, nsteps=800)
    e, st = r["e"], r["sigma_theta"]
    s = (e > 1.0) & (e < 1000) & np.isfinite(st)
    assert np.all(np.diff(st[s]) < 0)


def test_performance_is_cheap():
    # The angular dimension is a vectorized batch -> the whole coupled solve is
    # fast and scales sub-linearly in lmax.
    r = run_cascade(moments=_synthetic_theta2(), lmax=80, nsteps=1000)
    assert r["runtime"] < 2.0  # generous; typically ~tens of ms
    assert r["state_shape"] == (len(log_grid(60)[0]), 81)
