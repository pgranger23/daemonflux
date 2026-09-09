"""Tests for the sub-limb treatment of the production cone (``cone_geff``).

The full-sphere cutoff map stitched the down-going *detector* cutoff onto the
up-going *antipodal far-side* cutoff, and a near-horizon production cone read
that stitch: at Kamioka the 89-deg down-going cell is 49.1 GV and the 90-deg
up-going cell 39.6 GV, a 9.5 GV jump across 1 deg of zenith.  Those far-side
values are the cutoffs of trajectories on the other side of the Earth -- not of
the primaries that made the neutrino.

``sublimb="prod_point"`` instead evaluates every cone sample in the local frame
of the production point (~30 km up the arrival ray, so the local vertical is
tilted by ~L/R_E) and reads the **down-going** map only; a sample still below the
horizon *there* is Earth-shadowed and is dropped from both the numerator and the
cone normalisation.

These tests need no MCEq run: ``cone_geff`` only uses ``self.e`` (plus the
kinematic kernels), so they drive it on a stub with a synthetic map.  The
production-angle kernels are replaced by a cheap analytic Gaussian of the same
width -- they cost ~20 s per call to rebuild from the moment tables and nothing
here tests them; what is under test is the sub-limb geometry.
"""

from __future__ import annotations

import numpy as np
import pytest

from mceq3d_flux import MCEq3DFlux, SPECIES

SIGMA_PI_DEG = 12.0  # stand-in cone width; the tests are width-independent


@pytest.fixture(autouse=True)
def _cheap_kernels(monkeypatch):
    import kinematic_kernel as kk

    def fake_pdf(e, alpha_deg, scale=1.0, **kw):
        a = np.asarray(alpha_deg, float)[None, :]
        w = (np.exp(-0.5 * (a / (SIGMA_PI_DEG * scale)) ** 2)
             * np.sin(np.radians(a)))
        return np.repeat(w / w.sum(), len(np.atleast_1d(e)), axis=0)

    monkeypatch.setattr(kk, "pion_alpha_pdf", fake_pdf)
    monkeypatch.setattr(
        kk, "channel_shapes",
        lambda e, **kw: {"pi": np.full(len(np.atleast_1d(e)), SIGMA_PI_DEG)},
    )


class _Stub:
    """Minimal stand-in: cone_geff only reads ``self.e`` and memo attributes."""

    def __init__(self, e):
        self.e = e


def _synthetic(up_going_value):
    """(fine map, rc_grid, G_by_z) with a steeply rising East limb and a
    configurable, deliberately discontinuous up-going hemisphere."""
    zen = np.concatenate([np.linspace(0.0, 80.0, 13),
                          np.array([82.0, 84.0, 85.5, 87.0, 88.0, 88.5, 89.0, 89.5]),
                          np.linspace(90.0, 180.0, 13)])
    az = np.linspace(0.0, 360.0, 25)
    rc = np.empty((len(zen), len(az)))
    for i, z in enumerate(zen):
        if z <= 89.9:  # down-going: East (az 90) high, West (az 270) low
            base = 11.0 + 40.0 * max(0.0, (z - 60.0) / 30.0) ** 2
            rc[i] = base * (1.0 + 0.35 * np.sin(np.radians(az)))
        else:
            rc[i] = up_going_value
    rc_grid = np.linspace(0.1, 55.0, 40)
    e = np.geomspace(0.3, 10.0, 8)
    # monotone, energy-dependent suppression G(R_c)
    G = np.exp(-np.outer(rc_grid, 1.0 / e) / 20.0)
    G_by_z = [{s: G for s in SPECIES}]
    return (zen, az, rc), rc_grid, G_by_z, e


def _geff(cz, azd, sublimb, up_value):
    fine, rc_grid, G_by_z, e = _synthetic(up_value)
    zen, az, rc = fine
    # single cutoff at the neutrino direction, for the fallback branch
    iz = int(np.argmin(np.abs(zen - np.degrees(np.arccos(cz)))))
    ia = int(np.argmin(np.abs(az - azd)))
    rc_map = np.array([[rc[iz, ia]]])
    out = MCEq3DFlux.cone_geff(
        _Stub(e), np.array([cz]), np.array([azd]), rc_map, rc_grid, G_by_z, fine,
        channel_cone=False, muon_bending=False, sublimb=sublimb,
    )
    return np.array([out[s][0, 0] for s in SPECIES])


def test_prod_point_ignores_the_antipodal_farside_map():
    """The 89/90-deg seam cannot influence a down-going cone any more."""
    cz = np.cos(np.radians(87.0))
    a = _geff(cz, 90.0, "prod_point", up_value=5.0)
    b = _geff(cz, 90.0, "prod_point", up_value=50.0)
    assert np.allclose(a, b), "prod_point still reads the up-going hemisphere"


def test_farside_is_sensitive_to_the_seam():
    """Control: the legacy path *does* depend on the far-side values, which is
    exactly the defect (a 10x change of an unrelated hemisphere moves <G>)."""
    cz = np.cos(np.radians(87.0))
    a = _geff(cz, 90.0, "farside", up_value=5.0)
    b = _geff(cz, 90.0, "farside", up_value=50.0)
    assert not np.allclose(a, b, rtol=1e-3)


def test_vertical_is_unchanged_by_the_sublimb_treatment():
    """At the vertical the production frame is the detector frame and nothing is
    blocked, so the two treatments must agree."""
    a = _geff(1.0, 0.0, "prod_point", up_value=5.0)
    b = _geff(1.0, 0.0, "farside", up_value=5.0)
    assert np.allclose(a, b, rtol=2e-3), (a, b)


def test_geff_stays_a_proper_average():
    """Blocking renormalises, so <G> must stay inside the range of G over the
    map -- it must not be pushed to zero by the dropped samples."""
    for zdeg in (60.0, 81.0, 87.0, 89.0):
        cz = np.cos(np.radians(zdeg))
        g = _geff(cz, 90.0, "prod_point", up_value=5.0)
        assert np.all(g > 0.0) and np.all(g < 1.0), (zdeg, g)


def test_bad_sublimb_rejected():
    with pytest.raises(ValueError):
        _geff(1.0, 0.0, "nonsense", up_value=5.0)
