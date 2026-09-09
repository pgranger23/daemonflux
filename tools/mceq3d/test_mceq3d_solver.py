"""Offline tests for the integrated 3D-MCEq engine.

Synthetic <theta^2>(E) (no data file); geomagnetics via daemonflux.geomagnetic.
"""

import numpy as np

from mceq3d_solver import solve


def _theta2():
    e = np.logspace(-0.5, 5, 45)
    return e, (0.3 / e) ** 2


def test_angular_treatment_conserves_energy_spectrum():
    # THE key invariant: turning the angular machinery on must not change the
    # l=0 flux (the energy cascade) -> exact reduction to 1D.
    on = solve(moments=_theta2(), lmax=24, nsteps=800)
    off = solve(moments=_theta2(), lmax=24, nsteps=800, collimated=True)
    m = on["numu"] > 0
    assert np.allclose(on["numu"][m], off["numu"][m], rtol=1e-12)


def test_geomagnetic_suppresses_low_energy_via_primary():
    base = solve(moments=_theta2(), lmax=16, nsteps=800)
    geo = solve(moments=_theta2(), lmax=16, nsteps=800, geomag_site="ino")
    e = base["e"]
    rat = geo["numu"] / base["numu"]
    lo = (e > 0.7) & (e < 2) & (base["numu"] > 0)
    hi = (e > 200) & (e < 1e4) & (base["numu"] > 0)  # finite-flux region
    assert np.nanmean(rat[lo]) < 0.9  # suppressed at low E
    assert np.allclose(rat[hi], 1.0, atol=1e-3)  # recovered at high E


def test_flux_positive_with_peak():
    r = solve(moments=_theta2(), lmax=16, nsteps=800)
    e, f = r["e"], r["numu"]
    assert np.all(f >= 0)
    w = (f * e**2)[(e > 0.5) & (e < 1e4)]
    assert w.argmax() > 0 and w[0] < w.max() and w[-1] < w.max()  # has a peak


def test_spread_below_single_production_angle_and_decreasing():
    r = solve(moments=_theta2(), lmax=48, nsteps=800)
    e, st, th1 = r["e"], r["sigma_theta"], np.degrees(r["theta1"])
    s = (e > 1) & (e < 1000) & np.isfinite(st)
    assert np.all(st[s] < th1[s])  # parent meson is more forward
    assert np.all(np.diff(st[s]) < 0)


def test_explicit_cutoff_suppresses_low_energy():
    # Feeding a back-traced cutoff directly (cutoff_GV) folds it into the primary;
    # low-E numu is suppressed, high-E recovered.
    base = solve(moments=_theta2(), lmax=8, nsteps=800)
    geo = solve(moments=_theta2(), lmax=8, nsteps=800, cutoff_GV=12.0)
    e = base["e"]
    rat = geo["numu"] / base["numu"]
    assert np.nanmean(rat[(e > 0.7) & (e < 2) & (base["numu"] > 0)]) < 0.9
    assert np.allclose(rat[(e > 300) & (e < 1e4) & (base["numu"] > 0)], 1.0, atol=1e-3)


def test_muon_bending_broadens_subgev_spread():
    # Muon bending adds angular variance to the decay neutrinos -> larger
    # sigma_theta at sub-GeV than without it.
    a = solve(moments=_theta2(), lmax=48, nsteps=800, muon_bending=False)
    b = solve(moments=_theta2(), lmax=48, nsteps=800, muon_bending=True)
    e = a["e"]
    m = (
        (e > 0.5)
        & (e < 2)
        & np.isfinite(a["sigma_theta"])
        & np.isfinite(b["sigma_theta"])
    )
    assert np.all(b["sigma_theta"][m] > a["sigma_theta"][m])


def test_curved_slant_depth_used():
    # A near-horizon column has larger grammage -> more meson interaction ->
    # different (softer) high-E flux than vertical. Just check it runs & differs.
    v = solve(moments=_theta2(), lmax=8, nsteps=800, zenith_deg=0.0)
    h = solve(moments=_theta2(), lmax=8, nsteps=800, zenith_deg=80.0)
    assert not np.allclose(v["numu"], h["numu"])
