"""Tests for the spherical (curved-atmosphere) cascade coupling."""

import numpy as np

from spherical_cascade import solve, density_gcc, path_to_top


def _run():
    cz = np.array([1.0, 0.6, 0.25, 0.08, 0.04])
    return solve(cz, n_e=40, nsteps=1500)


def test_vertical_flux_is_physical():
    # the vertical column produces a positive, finite numu spectrum
    r = _run()
    e = r["e"]
    band = (e > 0.5) & (e < 1e4)
    assert np.all(r["flux"][0][band] > 0)
    assert np.all(np.isfinite(r["flux"][0][band]))


def test_subgev_near_isotropic():
    # at sub-GeV mesons fully decay & primaries fully interact -> flat in zenith
    r = _run()
    e = r["e"]
    ie = int(np.argmin(np.abs(e - 0.4)))
    ratio = r["flux"][:, ie] / r["flux"][0, ie]
    assert np.all(np.abs(ratio - 1.0) < 0.1)  # within 10% of isotropic


def test_highE_horizon_enhanced_and_monotonic():
    # at high E the flux rises toward the horizon (cos z -> 0), monotonically
    r = _run()
    e = r["e"]
    ie = int(np.argmin(np.abs(e - 1000.0)))
    ratio = r["flux"][:, ie] / r["flux"][0, ie]  # cz descending 1 -> 0.04
    assert np.all(np.diff(ratio) > 0)  # increases toward horizon
    assert ratio[-1] > 2.0  # substantial near-horizon enhancement


def test_horizon_saturates_below_sec_theta():
    # curved geometry: near-horizon enhancement is finite, well below sec(theta)
    r = _run()
    e = r["e"]
    ie = int(np.argmin(np.abs(e - 1000.0)))
    cz = r["cos_zeniths"]
    enh = r["flux"][-1, ie] / r["flux"][0, ie]
    sec_theta = 1.0 / cz[-1]
    assert enh < 0.5 * sec_theta  # saturated, not diverging


def test_enhancement_grows_with_energy():
    # the sec(theta) effect is a high-energy effect: near-horizon enhancement
    # is larger at high E than at sub-GeV
    r = _run()
    e = r["e"]
    lo = (
        r["flux"][-1, int(np.argmin(np.abs(e - 0.4)))]
        / r["flux"][0, int(np.argmin(np.abs(e - 0.4)))]
    )
    hi = (
        r["flux"][-1, int(np.argmin(np.abs(e - 1000.0)))]
        / r["flux"][0, int(np.argmin(np.abs(e - 1000.0)))]
    )
    assert hi > lo + 1.0


def test_geometry_helpers():
    assert density_gcc(0.0) > density_gcc(10.0) > 0  # density falls with altitude
    # near-horizon path to top is much longer than vertical
    assert path_to_top(85.0) > 5 * path_to_top(0.0)
