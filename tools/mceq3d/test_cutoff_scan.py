"""Regression tests for the rigidity-cutoff scan and the near-limb cutoff map.

Covers the two defects repaired in Phase 1 of the audit:

1. **Rigidity-scan resolution.**  A single linear ladder of ``n_scan`` points
   cannot find the upper cutoff ``R_U`` robustly: with the delivered 2.37 GV step
   the Kamioka vertical ladder samples 9.98 GV, which lies inside the *allowed
   island* [9.63, 10.13], so the scan walks straight past the forbidden band that
   ends at ~11.5 GV and reports 8.79 GV.  The scanner is tested on a synthetic
   admittance carrying exactly that island, and against direct back-traces.
2. **Near-limb interpolation.**  The uniform 13-node zenith grid leaves a 7.4 deg
   gap over the convex rise of the East cutoff; the dense limb nodes must bring
   the bilinear interpolation error there under ~0.5 GV.

The synthetic tests are instant.  The tests that back-trace are marked
``slow_backtrace``; they take ~1-3 min and need ``ppigrf``.
"""

from __future__ import annotations

import datetime

import numpy as np
import pytest

import geomag_backtrace as gb
from mceq3d_flux import _zenith_nodes, LIMB_ZENITH_NODES

LAT, LON = 36.43, 137.31
DATE = datetime.datetime(2020, 1, 1)

# Reference cutoffs at Kamioka from a direct 0.17 GV-resolution back-trace scan
# (arrival azimuth 90 = from the East, 270 = from the West).
REF_CUTOFF = {
    (0.0, 0.0): 11.51,
    (87.0, 90.0): 41.66,
    (87.0, 270.0): 7.40,
}


# --------------------------------------------------------------------------
# 1. the scanner itself, on a synthetic admittance (no back-tracing)
# --------------------------------------------------------------------------
def _island_admittance(R):
    """Kamioka-vertical-like admittance: main forbidden band up to 11.51 GV with a
    narrow ALLOWED island at [9.63, 10.13] below it (diag_admittance_fine.py)."""
    R = np.atleast_1d(np.asarray(R, float))
    return (R > 11.51) | ((R >= 9.63) & (R <= 10.13))


def test_scanner_survives_an_allowed_island():
    rc, sat = gb.scan_upper_cutoff(
        lambda idx, R: _island_admittance(R), 1, r_lo=0.5, r_hi=55.0
    )
    assert not sat[0]
    # R_U = first forbidden from the top, to the bisection tolerance
    assert abs(rc[0] - 11.51) < gb.BISECT_TOL_GV


def test_coarse_ladder_alone_would_fall_into_the_island():
    """Documents the defect: a >1.4 GV ladder steps over the forbidden band."""
    rc, _ = gb.scan_upper_cutoff(
        lambda idx, R: _island_admittance(R), 1, r_lo=0.5, r_hi=55.0,
        coarse_step=2.37, tol=0.1,
    )
    assert rc[0] < 10.2  # lands below the island instead of at 11.51


def test_scanner_edge_cases():
    n = 3
    allowed = gb.scan_upper_cutoff(lambda i, R: np.ones(len(np.atleast_1d(R)), bool), n)
    assert np.allclose(allowed[0], 0.5) and not allowed[1].any()
    forb = gb.scan_upper_cutoff(lambda i, R: np.zeros(len(np.atleast_1d(R)), bool), n)
    assert np.allclose(forb[0], gb.RC_MAX_GV) and forb[1].all()  # saturated flag


def test_scanner_is_per_direction():
    """Directions are scanned independently (index plumbing, not a broadcast)."""
    truth = np.array([3.0, 17.0, 44.0])

    def fn(idx, R):
        return np.asarray(R, float) > truth[np.asarray(idx)]

    rc, sat = gb.scan_upper_cutoff(fn, 3)
    assert np.allclose(rc, truth, atol=gb.BISECT_TOL_GV)
    assert not sat.any()


def test_scan_resolution_guarantee():
    assert gb.COARSE_STEP_GV <= 1.0
    assert gb.BISECT_TOL_GV <= 0.1
    assert gb.RC_MAX_GV >= 55.0  # must clear the ~49.5 GV Kamioka maximum


# --------------------------------------------------------------------------
# 2. zenith nodes
# --------------------------------------------------------------------------
def test_limb_nodes_are_dense_near_the_horizon():
    z = _zenith_nodes(13, limb_nodes=True)
    assert np.all(np.diff(z) > 0)
    assert np.sum((z >= 80.0) & (z <= 90.0)) >= 6
    assert np.max(np.diff(z[z >= 80.0])) <= 2.0  # <=2 deg spacing at the limb
    assert set(LIMB_ZENITH_NODES) <= set(z.tolist())
    assert z[-1] < 90.0  # the detector map never claims a sub-horizon cell


def test_uniform_nodes_reproduce_the_legacy_grid():
    assert np.allclose(_zenith_nodes(13, limb_nodes=False), np.linspace(0, 89, 13))


# --------------------------------------------------------------------------
# 3. real back-traces (slow)
# --------------------------------------------------------------------------
@pytest.mark.slow_backtrace
def test_kamioka_reference_cutoffs():
    """Vertical, 87 deg East and 87 deg West against the fine reference scan."""
    dirs = list(REF_CUTOFF)
    up = gb._local_frame(LAT, LON)[0]
    r0 = np.tile(gb.RE * up, (len(dirs), 1))
    u0 = np.array([-gb.arrival_direction(LAT, LON, z, a) for z, a in dirs])
    rc, sat = gb.cutoff_from_states(r0, u0, DATE, n_jobs=len(dirs))
    assert not sat.any()
    got = dict(zip(dirs, rc))
    # the vertical is the case the delivered 2.37 GV ladder got wrong (8.79 GV)
    assert 11.3 <= got[(0.0, 0.0)] <= 11.7, got
    assert abs(got[(87.0, 90.0)] - REF_CUTOFF[(87.0, 90.0)]) < 0.5, got
    assert abs(got[(87.0, 270.0)] - REF_CUTOFF[(87.0, 270.0)]) < 0.5, got
    # East-West ordering for positive primaries
    assert got[(87.0, 270.0)] < got[(87.0, 90.0)]
