"""Offline test for the directional-flux assembly.

The trajectory back-tracer (`cutoff_map`) is slow and already tested in
`test_geomag_backtrace`; here we mock it with a synthetic directional cutoff to
test the *assembly* (cascade folding + East-West propagation) quickly.
"""

import numpy as np

import geomag_backtrace
import directional_flux


def _theta2():
    e = np.logspace(-0.5, 5, 45)
    return e, (0.3 / e) ** 2


def test_directional_assembly_propagates_east_west(monkeypatch):
    zen = np.array([0.0, 75.0])
    az = np.array([90.0, 270.0])  # East, West

    # synthetic cutoff: high from East (hard), low from West (easy)
    def fake_map(lat, lon, date, zeniths, azimuths, **kw):
        rc = np.zeros((len(zeniths), len(azimuths)))
        rc[:, 0] = 18.0  # East -> high cutoff
        rc[:, 1] = 6.0  # West -> low cutoff
        return rc

    monkeypatch.setattr(geomag_backtrace, "cutoff_map", fake_map)

    r = directional_flux.solve_directional(
        36.4, 137.3, zen, az, n_scan=4, moments=_theta2()
    )
    assert r["flux"].shape == (2, 2, len(r["e"]))
    ie = int(np.argmin(np.abs(r["e"] - 1.0)))
    east = r["flux"][1, 0, ie]  # zenith 75, East (high cutoff -> suppressed)
    west = r["flux"][1, 1, ie]  # zenith 75, West (low cutoff -> less suppressed)
    assert west > east  # lower western cutoff -> higher flux (East-West effect)


def test_directional_high_energy_unaffected(monkeypatch):
    def fake_map(lat, lon, date, zeniths, azimuths, **kw):
        return np.full((len(zeniths), len(azimuths)), 12.0)

    monkeypatch.setattr(geomag_backtrace, "cutoff_map", fake_map)
    r = directional_flux.solve_directional(
        36.4,
        137.3,
        np.array([0.0]),
        np.array([0.0, 180.0]),
        n_scan=4,
        moments=_theta2(),
    )
    e = r["e"]
    hi = (e > 300) & (e < 1e4) & (r["base"] > 0)
    # well above the cutoff the directional flux equals the no-geomag base
    assert np.allclose(r["flux"][0, 0, hi] / r["base"][hi], 1.0, atol=1e-3)
