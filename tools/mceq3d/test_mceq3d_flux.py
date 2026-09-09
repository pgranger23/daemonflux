"""Offline tests for mceq3d_flux helpers.

The full MCEq base + geomagnetic-response solve is exercised in
``mceq3d_flux.main`` (and validated vs Honda there); it needs MCEq and is slow,
so here we test only the offline rigidity-transmission helper.
"""

import numpy as np

from mceq3d_flux import (
    _transmission,
    SIGMA_LNR,
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
    for sig in (SIGMA_LNR, 0.5, 0.1, 0.0):
        t = _transmission(e, rc_gv=12.0, sigma_lnr=sig)
        assert np.all((t >= 0) & (t <= 1))
        assert np.all(np.diff(t) >= -1e-12)  # monotonically increasing
        assert t[np.argmin(np.abs(e - 1.0))] < 0.05  # below cutoff -> blocked
        assert t[np.argmin(np.abs(e - 100.0))] > 0.99  # above -> transmitted
        # the 50% point sits at the cutoff, to within one grid bin
        dlne = np.log(e[1] / e[0])
        e_half = np.exp(np.interp(0.5, t, np.log(e)))
        assert abs(np.log(e_half / 12.0)) < dlne


def test_transmission_width_is_a_free_parameter_and_orders_correctly():
    """sigma_lnR sets the penumbra: narrower -> less leakage below R_c.

    The legacy hand-set 0.5 passes 8% of the primaries at R_c/2 and 21% at
    R_c/1.5; the back-traced East transition is a sharp step (``diag_penumbra_width``),
    which passes none.
    """
    e = np.logspace(-1, 3, 400)
    lo = int(np.argmin(np.abs(e - 6.0)))  # R_c/2
    hi = int(np.argmin(np.abs(e - 24.0)))  # 2 R_c
    ts = [_transmission(e, 12.0, s) for s in (0.5, 0.25, 0.1, 0.03, 0.0)]
    for a, b in zip(ts, ts[1:]):  # sharper -> smaller below, larger above
        assert b[lo] <= a[lo] + 1e-12
        assert b[hi] >= a[hi] - 1e-12
    assert 0.05 < ts[0][lo] < 0.12  # the legacy tail this task removes
    assert ts[-1][lo] == 0.0 and ts[-1][hi] == 1.0


def test_transmission_sharp_limit_is_the_exact_bin_overlap():
    """sigma_lnR=0 is the bin-averaged hard step, not a point-sampled one.

    MCEq's ``_phi0`` holds bin-averaged fluxes, so a cut inside a bin must give
    that bin's overlap fraction; this is what keeps G_s(R_c) continuous in R_c
    instead of a staircase that only moves at bin edges.
    """
    w = 0.1
    e = np.exp(np.arange(-2.0, 5.0, w))
    for off in (-0.04, 0.0, 0.03):
        rc = float(np.exp(np.log(e[40]) - off))
        t = _transmission(e, rc, 0.0)
        assert abs(t[40] - (0.5 + off / w)) < 1e-12
        assert t[41] == 1.0 and t[39] == 0.0


def test_transmission_sharp_is_continuous_in_rc():
    """No staircase: a small change of R_c must move T by a small amount."""
    e = np.exp(np.arange(-2.0, 5.0, 0.115))  # the MCEq grid spacing
    j = int(np.argmin(np.abs(e - 40.0)))
    rc = np.linspace(35.0, 45.0, 400)
    t = np.array([_transmission(e, r, 0.0)[j] for r in rc])
    assert np.all(np.diff(t) <= 1e-12)  # T falls as the cutoff rises
    assert np.abs(np.diff(t)).max() < 0.02  # continuous, no bin-edge jump
    assert t[0] == 1.0 and t[-1] == 0.0  # and it does span the full range


def test_transmission_bin_average_is_negligible_at_the_legacy_width():
    """The bin average must not move the published sigma=0.5 baseline."""
    from scipy.special import erf

    e = np.exp(np.arange(-2.0, 5.0, 0.115))
    t = _transmission(e, 12.0, 0.5)
    t_point = 0.5 * (1 + erf((np.log(e) - np.log(12.0)) / (np.sqrt(2) * 0.5)))
    assert np.abs(t - t_point).max() < 1e-3


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


def test_hybrid_weight_blend():
    """hybrid base blend: GSF-MCEq below ~0.8 GeV, daemonflux above."""
    from mceq3d_flux import hybrid_weight

    e = np.array([0.1, 1.7, 10.0, 100.0])
    w = hybrid_weight(e)
    assert w[0] < 0.03  # sub-GeV: data-anchored (GSF) MCEq base
    # centre = daemonflux calibration floor mapped to nu (5 GeV / ~3)
    assert abs(w[1] - 0.5) < 1e-12
    assert w[2] > 0.99 and w[3] > 0.999  # >~ GeV: muon-calibrated daemonflux
    assert np.all(np.diff(hybrid_weight(np.geomspace(0.1, 100, 50))) > 0)


# ---------------------------------------------------------------------------
# Delivered geomagnetic response: cache-backed regression
# ---------------------------------------------------------------------------
# G_s(E, R_c) is a pair of MCEq cascade solves per rigidity (~12 min at the
# vertical, ~1 h at 87 deg), far too slow for CI, so this pins the DELIVERED
# artifact instead: the cached response that ``solve(use_cache=True)`` builds
# for the Kamioka vertical band with the delivered engine (hybrid base, GSF
# primary, SIBYLL-2.3d).  The key is built with the module's own helper, so a
# key change makes the test skip loudly rather than compare the wrong file.
#
# Values measured 2026-09-04 with the delivered penumbra width SIGMA_LNR = 0
# (the back-traced sharp step; diag_penumbra_width.py).  The legacy sigma = 0.5 values
# are pinned alongside because the whole point of the change is the East: at
# R_c = 41.7 GV (Kamioka, 87 deg East) the hand-set penumbra let through 14%
# more flux than the measured cutoff does, while the vertical moves by 2%.
_GS_TAG = "SIBYLL23D_GlobalSplineFitBeta-None_emin0.1_atmstd"
_GS_CZ_VERTICAL = 0.95
GS_VERTICAL_REF = {  # (sigma_lnR, species, R_c[GV], E[GeV]) -> G_s
    (0.0, "total_numu", 11.5, 1.0): 0.7228,
    (0.0, "total_numu", 11.5, 0.5): 0.5784,
    (0.0, "total_numu", 41.7, 0.5): 0.1653,
    (0.0, "total_nue", 11.5, 1.0): 0.6696,
    (0.0, "total_nue", 41.7, 0.5): 0.1432,
    (0.5, "total_numu", 11.5, 1.0): 0.7068,
    (0.5, "total_numu", 41.7, 0.5): 0.1889,
}


def _load_gs(sigma):
    """Cached delivered G_s at the Kamioka vertical band, or ``(None, rc)``."""
    import os

    from mceq3d_flux import RC_MAX_GV, gs_cache_name

    rc = np.linspace(0.1, RC_MAX_GV, 40)
    name = gs_cache_name(_GS_TAG, _GS_CZ_VERTICAL, rc, sigma)
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (os.path.join(here, ".cache3d"), os.path.join(here, "flux_cache")):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return np.load(p), rc
    return None, rc


def _gs_at(d, rc, species, rc_gv, e_gev):
    j = int(np.argmin(np.abs(d["e"] - e_gev)))
    return float(np.interp(rc_gv, rc, d[species][:, j]))


def test_geomag_response_vertical_regression():
    """Pin the delivered cascade suppression at the Kamioka vertical band."""
    import pytest

    for (sig, sp, rc_gv, e_gev), ref in GS_VERTICAL_REF.items():
        d, rc = _load_gs(sig)
        if d is None:
            pytest.skip(f"no cached G_s for sigma_lnR={sig} (run solve once)")
        got = _gs_at(d, rc, sp, rc_gv, e_gev)
        assert abs(got - ref) < 5e-3, (sig, sp, rc_gv, e_gev, got, ref)


def test_sharp_cutoff_suppresses_the_east_more_than_the_legacy_penumbra():
    """The measured (sharp) cutoff must bite harder where R_c is large.

    This is the physics of the penumbra change: at the 87 deg East cutoff the
    legacy sigma = 0.5 erf opened 6.3 GV of allowed rigidity below R_c, where
    the back-tracer measures none; at the 11.5 GV vertical cutoff the same erf
    opens only 1.7 GV, so the East moves several times more than the vertical.
    """
    import pytest

    d0, rc = _load_gs(0.0)
    d5, _ = _load_gs(0.5)
    if d0 is None or d5 is None:
        pytest.skip("need both the sharp and the legacy cached G_s")
    east0 = _gs_at(d0, rc, "total_numu", 41.7, 0.5)
    east5 = _gs_at(d5, rc, "total_numu", 41.7, 0.5)
    vert0 = _gs_at(d0, rc, "total_numu", 11.5, 0.5)
    vert5 = _gs_at(d5, rc, "total_numu", 11.5, 0.5)
    assert east0 < 0.93 * east5  # East: >7% less flux gets through
    assert abs(vert0 / vert5 - 1.0) < 0.03  # vertical: barely moves


# ---------------------------------------------------------------------------
# The two E_off paths must be the SAME physics (2026-09-04 defaults)
#
# ``solve()`` applies the joint production x cutoff integral
# (``joint_cone_factor``) on the down-going, in-range flux, and the tabulated
# ``offaxis_factor`` everywhere else (``cone_cutoff=False``, the up-going
# hemisphere, energies outside the tabulated 0.1-100 GeV).  Until 2026-09-04
# those were different physics: the joint factor used the channel-resolved cone
# on whatever moments the caller passed, the table was the flat pion-only July
# build on the pre-arcsin moments.  Both defaults now resolve to the
# arcsin-corrected v2 moments and the channel-weighted cone, so with ``G == 1``
# (no geomagnetic suppression -- a zero-cutoff site) the two must agree.
# ---------------------------------------------------------------------------
def _joint_gate_inputs():
    """(prod, fine, prod_cache_dir) or None if the cached artifacts are absent."""
    import glob
    import os

    import joint_cone as jc
    import offaxis_mc as ox

    here = os.path.dirname(os.path.abspath(__file__))
    cache = None
    for d in (os.environ.get("MCEQ3D_CACHE_DIR", ""), ".cache3d", "flux_cache"):
        if not d:
            continue
        c = os.path.join(here, d, f"jointprod_chan_{ox.TAG}.npz")
        if os.path.exists(c):
            cache = c
            break
    if cache is None:
        return None
    mp = jc.PAPER_MAP
    for f in sorted(glob.glob(os.path.join(os.path.dirname(cache),
                                           "finerc_*.npz"))
                    + glob.glob(os.path.join(here, "finerc_*.npz"))):
        if len(np.load(f)["zen"]) > 30:  # dense limb nodes -> the repaired map
            mp = f
            break
    if not os.path.exists(mp):
        return None
    prod = jc.load_production(cache=cache, species=ox.CHANNEL_SPECIES)
    return prod, jc.load_rc_map(mp), os.path.dirname(cache)


def test_table_path_matches_joint_path_at_unit_G():
    import pytest

    import joint_cone as jc
    from mceq3d_flux import MCEq3DFlux

    got = _joint_gate_inputs()
    if got is None:
        pytest.skip("needs a cached channel-split production profile and a "
                    "cutoff map (run diag_joint_stages.py once)")
    prod, fine, cdir = got
    ep = prod["ep_grid"]
    # a bare engine shell: both methods need only ``e`` and the model tag
    eng = MCEq3DFlux.__new__(MCEq3DFlux)
    eng.e = ep
    eng._tag = "SIBYLL23D_HillasGaisser2012-H3a_emin0.1_atmstd"
    eng._joint_prod = prod
    rc_grid = np.linspace(0.1, jc.RC_MAX_GV, 40)
    G1 = {s: np.ones((len(rc_grid), len(ep))) for s in SPECIES}
    cz = np.array([0.05, 0.35, 0.95])  # down-going: horizon, mid, vertical
    az = np.array([90.0])
    Eoff = eng.offaxis_factor(cz)  # the DEFAULT table path
    Geff = {s: np.ones((len(cz), len(az), len(ep))) for s in SPECIES}
    F = eng.joint_cone_factor(cz, az, rc_grid, [G1] * len(cz), fine, Eoff, Geff,
                              muon_bending=False, b_enu=None, cache_dir=cdir)
    for iz in range(len(cz)):
        for s in SPECIES:
            assert np.allclose(F[s][iz, 0], Eoff[s][iz], rtol=1e-9, atol=0), (
                cz[iz], s)


def test_default_eoff_table_is_species_resolved_and_v2():
    """The default table must be the channel-weighted v2 build (not the flat
    July one), and it must actually differ per species."""
    import os

    import pytest

    from mceq3d_flux import EOFF_TABLE, EOFF_TABLE_FLAT, MCEq3DFlux

    here = os.path.dirname(os.path.abspath(__file__))
    tab, flat = (os.path.join(here, EOFF_TABLE),
                 os.path.join(here, EOFF_TABLE_FLAT))
    if not (os.path.exists(tab) and os.path.exists(flat)):
        pytest.skip("E_off tables not built (run offaxis_mc.py --build-channel)")
    d = np.load(tab)
    assert str(d["moments"]) == "v2" and str(d["cone_mode"]) == "channel"
    eng = MCEq3DFlux.__new__(MCEq3DFlux)
    eng.e = d["e"]
    eng._tag = "SIBYLL23D_HillasGaisser2012-H3a_emin0.1_atmstd"
    cz = np.array([0.05])
    E = eng.offaxis_factor(cz)
    # nu_e is ~entirely muon-decay (the WIDE cone) -> a larger horizon excess
    # than nu_mu; the flat table cannot express that at all.
    i = int(np.argmin(abs(d["e"] - 0.3)))
    assert E["total_nue"][0, i] > E["total_numu"][0, i] * 1.01
    Eflat = eng.offaxis_factor(cz, path=flat)
    assert Eflat["total_nue"][0, i] == Eflat["total_numu"][0, i]
    assert E["total_numu"][0, i] > Eflat["total_numu"][0, i] * 1.05
