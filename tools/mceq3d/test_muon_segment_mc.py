"""Tests for :mod:`muon_segment_mc` -- the muon-segment bending Monte Carlo.

The physics assertions are the ones the study rests on:

* the accumulated bend is ``q B T_p / m`` -- **exactly independent of the muon
  energy and of the energy loss** (the hypothesis this study was built to test);
* the mean bend over the exponential proper-time law is the delivered
  ``Delta = q B tau / m = 5.08 deg``;
* the rotation sense matches ``muon_bending.bending_deflection`` (the
  (north, east, up) handedness trap);
* Lipari's qualitative statement: a ``mu+`` whose decay neutrino arrives from
  the (magnetic) East near the horizon came from a primary at a *more extreme*
  zenith.

Everything here runs without MCEq except the two tests marked ``needs_source``.
"""

from __future__ import annotations

import numpy as np
import pytest

import muon_segment_mc as ms

LAT, LON = ms.LAT, ms.LON


def _const_B():
    from muon_bending import local_field_enu

    b_enu = local_field_enu(LAT, LON, ms.DATE)          # gauss, ENU
    basis = ms.enu_basis(LAT, LON)
    return b_enu, ms.to_geocentric(b_enu * 1e-4, basis)[0]   # tesla, geocentric


# ---------------------------------------------------------------------------
def test_handedness_and_delta():
    """Exact rotation == first-order deflection of the delivered engine."""
    err, delta_deg, b = ms.check_handedness(87.0, 81.88, +1)
    assert err < 0.05
    assert 5.0 < delta_deg < 5.2                     # 5.08 deg at |B|=0.474 G
    # ENU is right-handed: cross(east, north) = up
    e, n, u = np.eye(3)
    assert np.allclose(np.cross(e, n), u)
    # ... while the (north, east, up) ordering used inside cone_geff is not
    assert np.allclose(np.cross(n, e), -u)
    assert np.isclose(np.linalg.norm(b), 0.4743, atol=2e-3)


def test_zero_field_no_bend():
    n = 200
    r0 = np.tile([0.0, 0.0, ms.RE_M + 20e3], (n, 1))
    u0 = np.tile([1.0, 0.0, 0.0], (n, 1))
    st = ms._backward(r0, u0, np.full(n, 3.0), np.full(n, ms.TAU_MU), +1,
                      field=None, rho=lambda h: np.zeros_like(np.asarray(h)),
                      const_field=[0.0, 0.0, 0.0], energy_loss=False,
                      ref=u0.copy(), dt_tau=1 / 40, ds_max_m=2000.0,
                      max_steps=200)
    assert np.allclose(st["psi"], 0.0)
    assert np.allclose(st["ref"], u0)


@pytest.mark.parametrize("e_mu", [0.3, 1.0, 10.0, 100.0])
def test_bend_is_energy_independent(e_mu):
    """``psi = q B T_p / m`` at every energy -- with and without energy loss."""
    _, b_gc = _const_B()
    n = 64
    r0 = np.tile([0.0, 0.0, ms.RE_M + 20e3], (n, 1))
    u0 = np.tile([1.0, 0.0, 0.0], (n, 1))
    rho = ms.load_atmosphere()[0]
    out = {}
    for loss in (False, True):
        st = ms._backward(r0, u0, np.full(n, e_mu), np.full(n, ms.TAU_MU), +1,
                          field=None, rho=rho, const_field=b_gc,
                          energy_loss=loss, ref=u0.copy(), dt_tau=1 / 80,
                          ds_max_m=4000.0, max_steps=900)
        out[loss] = np.degrees(st["psi"].mean())
    expected = np.degrees(ms.Q_E * np.linalg.norm(b_gc) * ms.TAU_MU / ms.M_MU_KG)
    assert np.isclose(out[False], expected, rtol=1e-4)
    # the whole point: energy loss changes the path, never the rotation
    assert np.isclose(out[True], out[False], rtol=1e-6)


def test_energy_loss_does_change_the_energy():
    """Guard against a silent no-op in the dE/dx branch."""
    _, b_gc = _const_B()
    n = 8
    r0 = np.tile([0.0, 0.0, ms.RE_M + 8e3], (n, 1))
    u0 = np.tile([1.0, 0.0, 0.0], (n, 1))
    rho = ms.load_atmosphere()[0]
    st = ms._backward(r0, u0, np.full(n, 2.0), np.full(n, 3 * ms.TAU_MU), +1,
                      field=None, rho=rho, const_field=b_gc, energy_loss=True,
                      ref=u0.copy(), dt_tau=1 / 80, ds_max_m=500.0,
                      max_steps=900)
    assert (st["e"] > 2.05).all()      # backwards the muon must be more energetic


def test_exponential_mean_is_the_delivered_delta():
    """``<psi>`` over ``T ~ Exp(tau)`` is exactly the engine's single shift."""
    _, b_gc = _const_B()
    rng = np.random.default_rng(0)
    n = 4000
    T = rng.exponential(ms.TAU_MU, n)
    r0 = np.tile([0.0, 0.0, ms.RE_M + 25e3], (n, 1))
    u0 = np.tile([1.0, 0.0, 0.0], (n, 1))
    st = ms._backward(r0, u0, np.full(n, 3.0), T, +1, field=None,
                      rho=ms.load_atmosphere()[0], const_field=b_gc,
                      energy_loss=True, ref=u0.copy(), dt_tau=1 / 40,
                      ds_max_m=1500.0, max_steps=900)
    delta = np.degrees(ms.Q_E * np.linalg.norm(b_gc) * ms.TAU_MU / ms.M_MU_KG)
    assert np.isclose(np.degrees(st["psi"]).mean(), delta, rtol=0.05)


def test_michel_spectra_normalised():
    x = np.linspace(0, 1, 20001)
    for kind in ("numu", "nue"):
        assert np.isclose(np.trapezoid(ms.michel_n(x, kind), x), 1.0, atol=1e-4)


def test_dedx_air_sane():
    assert 1.7e-3 < ms.dedx_air(1.0) < 2.2e-3           # GeV cm2/g
    assert ms.dedx_air(100.0) > ms.dedx_air(1.0)         # relativistic rise
    assert ms.dedx_air(0.15) > ms.dedx_air(1.0)          # 1/beta^2 rise


def test_field_grid_matches_ppigrf():
    """The interpolated grid reproduces a direct IGRF call.

    ``muon_bending.local_field_enu`` uses ``ppigrf.igrf`` (**geodetic**) while
    the tracer -- like ``geomag_backtrace`` -- uses ``ppigrf.igrf_gc``
    (**geocentric**, spherical Earth).  At Kamioka the two latitudes differ by
    ~0.19 deg, which tilts the ENU frame by the same amount, so the components
    agree only to that rotation; the magnitude and the trilinear interpolation
    error are what this test pins.
    """
    fg = ms.FieldGrid()
    basis = ms.enu_basis(LAT, LON)
    p = np.asarray(basis[2]) * (ms.RE_M + 25e3)
    b_grid = fg(p[None, :])[0]
    from muon_bending import local_field_enu

    b_ref = ms.to_geocentric(local_field_enu(LAT, LON, ms.DATE, h_km=25.0)
                             * 1e-4, basis)[0]
    assert np.isclose(np.linalg.norm(b_grid), np.linalg.norm(b_ref), rtol=3e-3)
    ang = np.degrees(np.arccos(np.clip(
        b_grid @ b_ref / (np.linalg.norm(b_grid) * np.linalg.norm(b_ref)),
        -1, 1)))
    assert ang < 0.5, f"field direction off by {ang:.2f} deg"


def test_wstats():
    x = np.array([1.0, 2.0, 3.0])
    w = np.array([1.0, 1.0, 2.0])
    s = ms.wstats(x, w)
    assert np.isclose(s["mean"], 2.25)
    assert s["neff"] > 2.0


# ---------------------------------------------------------------------------
# the two tests that need the MCEq-derived muon source table
# ---------------------------------------------------------------------------
def _have_source():
    import os

    return os.path.exists(os.path.join(ms.SCRATCH,
                                       "muon_source_mu_transport.npz"))


needs_source = pytest.mark.skipif(not _have_source(),
                                  reason="muon source table not built")


@needs_source
def test_segment_mc_reproduces_delta_and_lipari():
    """Full MC: mean bend ~5 deg, and Lipari's East ``mu+`` bends DOWN."""
    rho, geom = ms.load_atmosphere()
    src = ms.source_interp(ms.muon_source_table())
    fld = ms.FieldGrid()
    r = ms.segment_mc(87.0, 81.88, 0.5, +1, n=4000, seed=5, field=fld,
                      rho=rho, geom=geom, source=src, prod_dir="ray")
    s = ms.wstats(r["bend_deg"], r["w_numu"])
    assert s["neff"] > 500
    assert 3.5 < s["mean"] < 5.3                       # near the 5.08 deg scale
    # Lipari (hep-ph/0003013): a mu+ arriving from the East horizontally was
    # bent DOWN, so its primary sat at a MORE extreme zenith.
    dz = ms.wstats(r["dzen"], r["w_numu"])["mean"]
    assert dz > 0.5, f"mu+ from East must push the primary zenith up, got {dz}"
    rm = ms.segment_mc(87.0, 81.88, 0.5, -1, n=4000, seed=5, field=fld,
                       rho=rho, geom=geom, source=src, prod_dir="ray")
    assert ms.wstats(rm["dzen"], rm["w_numu"])["mean"] < -0.5


@needs_source
def test_energy_loss_flag_barely_moves_the_bend():
    """Switching dE/dx off changes the bend only through the E_nu selection."""
    rho, geom = ms.load_atmosphere()
    src = ms.source_interp(ms.muon_source_table())
    fld = ms.FieldGrid()
    kw = dict(n=6000, seed=7, field=fld, rho=rho, geom=geom, source=src,
              prod_dir="ray")
    a = ms.segment_mc(87.0, 81.88, 1.0, +1, energy_loss=True, **kw)
    b = ms.segment_mc(87.0, 81.88, 1.0, +1, energy_loss=False, **kw)
    ma = ms.wstats(a["bend_deg"], a["w_numu"])["mean"]
    mb = ms.wstats(b["bend_deg"], b["w_numu"])["mean"]
    # near the horizon the selection effect is small, and it goes the way the
    # energy-loss hypothesis did NOT predict: dE/dx can only *reduce* the bend.
    assert abs(ma - mb) / mb < 0.15
    assert ma <= mb * 1.02


@needs_source
def test_source_transport_matches_pi_decay_where_both_valid():
    """The transport inversion equals the independent pi/K -> mu fold above the
    MCEq hadron-tracking threshold (the only place both are defined)."""
    import os

    alt = os.path.join(ms.SCRATCH, "muon_source_pi_decay.npz")
    if not os.path.exists(alt):
        pytest.skip("pi_decay cross-check table not built")
    a = ms.source_interp(ms.muon_source_table())
    b = ms.source_interp(ms.muon_source_table(method="pi_decay"))
    for E in (10.0, 20.0, 50.0):
        ra = float(a(130.0, E, +1))
        rb = float(b(130.0, E, +1))
        assert abs(rb / ra - 1.0) < 0.10, (E, ra, rb)
