"""Offline tests for mceq3d_flux helpers.

The full MCEq base + geomagnetic-response solve is exercised in
``mceq3d_flux.main`` (and validated vs Honda there); it needs MCEq and is slow,
so here we test only the offline rigidity-transmission helper.
"""

import numpy as np

from mceq3d_flux import (
    _transmission,
    horizon_grid,
    interp_flux,
    farside_production,
    CM2_PER_M2,
    SPECIES,
)


def test_farside_straight_up_is_antipode():
    # straight-up from Kamioka -> far-side at the antipodal longitude, vertical down
    lat, lon = 36.43, 137.31
    latq, lonq, czq, azq = farside_production(lat, lon, -1.0, 0.0)
    assert abs(((lonq - (lon - 180)) + 180) % 360 - 180) < 2.0  # antipode longitude
    assert abs(latq + lat) < 2.0  # antipode latitude
    assert czq > 0.99  # vertical (down-going) primary at the far side


def test_farside_near_horizon_stays_near_horizon():
    # up-going near the horizon is produced near the horizon on the far side
    _, _, czq, _ = farside_production(36.43, 137.31, -0.05, 90.0)
    assert 0.0 < czq < 0.3


def test_interp_flux_on_synthetic_grid():
    e = np.logspace(-0.5, 3, 40)
    cz = np.array([0.95, 0.55, 0.05])
    az = np.array([0.0, 90, 180, 270])
    flux = {s: np.ones((3, 4, 40)) for s in SPECIES}
    flux["total_numu"][:] = (e**-2)[None, None, :]
    flux["total_numu"][0, 3] *= 2.0  # West vertical doubled
    r = dict(e=e, cos_zeniths=cz, azimuths=az, flux=flux)
    # at E=1, cosZ=0.95, az=270 -> 2 * 1^-2 = 2
    assert abs(interp_flux(r, 1.0, 0.95, 270.0, "total_numu") - 2.0) < 0.1
    # at az=0 (East) -> 1
    assert abs(interp_flux(r, 1.0, 0.95, 0.0, "total_numu") - 1.0) < 0.1
    # azimuth wraps modulo 360
    assert np.isclose(
        interp_flux(r, 1.0, 0.95, 270.0, "total_numu"),
        interp_flux(r, 1.0, 0.95, 630.0, "total_numu"),
    )


def test_transmission_is_a_rigidity_step():
    e = np.logspace(-1, 3, 200)
    t = _transmission(e, rc_gv=12.0)
    assert np.all((t >= 0) & (t <= 1))
    assert np.all(np.diff(t) >= -1e-12)  # monotonically increasing
    assert t[np.argmin(np.abs(e - 1.0))] < 0.05  # well below cutoff -> blocked
    assert t[np.argmin(np.abs(e - 100.0))] > 0.99  # well above -> transmitted
    # at the cutoff the transmission is ~0.5
    assert abs(t[np.argmin(np.abs(e - 12.0))] - 0.5) < 0.05


def test_bound_neutron_rigidity_less_suppressed():
    # neutrons (A/Z~2) have R=2E, so they are LESS cut than protons (R=E)
    e = np.logspace(-1, 3, 200)
    tp = _transmission(e, 12.0, az_over_z=1.0)
    tn = _transmission(e, 12.0, az_over_z=2.0)
    m = (e > 1) & (e < 12)
    assert np.all(tn[m] >= tp[m])


def test_unit_constant():
    assert CM2_PER_M2 == 1.0e4  # MCEq cm^-2 -> Honda/engine m^-2


def test_horizon_grid_refines_near_horizon():
    g = horizon_grid()
    assert np.all(np.diff(g) > 0)  # strictly increasing, ordered
    assert np.allclose(g, -g[::-1])  # symmetric up/down
    # more points packed near the horizon than in the bulk per unit cosθ
    near = np.sum(np.abs(g) < 0.2)
    far = np.sum(np.abs(g) >= 0.2)
    assert near > far  # horizon is refined
    assert g.min() < -0.9 and g.max() > 0.9  # spans the full sky
