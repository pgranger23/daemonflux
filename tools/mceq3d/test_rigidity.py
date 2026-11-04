"""Rigidity convention of the G_s response (2026-11 fix) and the GH primary variable."""
import importlib.util  # noqa: F401
import numpy as np
import pytest

import mceq3d_flux as mf
from mceq3d_flux import (_nucleon_transmission, _transmission, engine_tag,
                         gs_cache_name, RIGIDITY_TAG)
from mceq3d_primary import M_NUCLEON as M, primary_class

E = 0.1 * 10 ** (np.arange(0, 60) / 20.0)      # kinetic grid, 20/decade


def _tc(rc, az=1.0):
    return np.sqrt(M * M + (rc / az) ** 2) - M


def test_cut_acts_on_momentum_not_kinetic_energy():
    """7.5 GV: protons below T = 6.62 GeV are removed (not 7.50), bound nucleons
    (A/Z = 2) below 2.93 GeV/n (not 3.75)."""
    assert _tc(7.5) == pytest.approx(6.6188, abs=2e-3)
    assert _tc(7.5, 2.0) == pytest.approx(2.9273, abs=2e-3)
    for az, tc in ((1.0, _tc(7.5)), (2.0, _tc(7.5, 2.0))):
        t = _nucleon_transmission(E, 7.5, 0.0, az_over_z=az)
        below = E * np.exp(0.5 * mf._grid_dlne(E)) < tc
        above = E * np.exp(-0.5 * mf._grid_dlne(E)) > tc
        assert np.all(t[below] == 0) and np.all(t[above] == 1)
        # the partially cut bin is the one containing the threshold
        j = int(np.argmin(abs(np.log(E / tc))))
        assert 0 <= t[j] <= 1
    # the OLD convention would have put the edge at T = R_c / (A/Z)
    t = _nucleon_transmission(E, 7.5, 0.0)
    old = _transmission(E, 7.5, 0.0)
    i = np.where((E > 6.7) & (E < 7.4))[0]
    assert np.all(t[i] == 1) and np.any(old[i] < 1)


def test_hard_step_bin_overlap_is_exact_and_continuous():
    dl = mf._grid_dlne(E)
    for rc in (3.0, 7.5, 11.0, 40.0):
        t = _nucleon_transmission(E, rc, 0.0)
        # integral over ln T of the step = ln(T_max/T_c) within the grid
        tot = np.sum(t * dl)
        edge_hi = np.log(E[-1]) + 0.5 * dl[-1]
        assert tot == pytest.approx(edge_hi - np.log(_tc(rc)), rel=1e-9)
    r = np.linspace(7.0, 7.6, 61)
    j = int(np.argmin(abs(E - 6.8)))
    tj = np.array([_nucleon_transmission(E, x, 0.0)[j] for x in r])
    assert np.all(np.diff(tj) <= 1e-12) and np.max(np.abs(np.diff(tj))) < 0.25


def test_smooth_step_matches_momentum_form_at_high_energy():
    e = 30.0 * 10 ** (np.arange(0, 40) / 20.0)
    a = _nucleon_transmission(e, 25.0, 0.5)
    b = _transmission(np.sqrt(e * (e + 2 * M)), 25.0, 0.5, dlne=mf._grid_dlne(e))
    assert np.max(np.abs(a - b)) < 3e-3
    # monotone and bounded
    t = _nucleon_transmission(E, 12.0, 0.5)
    assert np.all(np.diff(t) >= -1e-12) and t.min() >= 0 and t.max() <= 1


def test_gs_cache_identity_carries_the_rigidity_convention():
    tag = engine_tag("SIBYLL23D", ("GlobalSplineFitBeta", None), 0.1, None, "")
    assert RIGIDITY_TAG in tag
    # a key built the way the pre-fix engine did can never equal a current one
    old = "SIBYLL23D_GlobalSplineFitBeta-None_emin0.1_atmstd"
    rc = np.array([7.0, 11.0, 40.0])
    assert gs_cache_name(old, 1.0, rc, 0.0) != gs_cache_name(tag, 1.0, rc, 0.0)
    assert tag.startswith(old)       # offaxis_factor() prefix comparison still works


def test_gaisser_honda_native_variable():
    """GH must see kinetic energy per nucleon.  All-nucleon flux ratios to GSF
    at total energy, T/n = 3..100 GeV (PM's table)."""
    import crflux.models as crf

    T = np.array([3, 5, 7, 10, 15, 20, 30, 50, 100.0])
    mt = 0.5 * (0.938272 + 0.939565)
    gsf = crf.GlobalSplineFitBeta()

    def nuc(m, e):
        _, p, n = m.p_and_n_flux(e)
        return p + n

    ref = nuc(gsf, T + mt)
    nat = primary_class("GaisserHondaNative")()
    tot = primary_class("GaisserHondaAtTotal")()
    h3a = crf.HillasGaisser2012("H3a")
    want_nat = [1.05, 1.00, 0.98, 0.98, 0.98, 0.99, 1.01, 1.05, 1.11]
    want_tot = [0.64, 0.70, 0.74, 0.79, 0.84, 0.88, 0.93, 1.00, 1.08]
    want_h3a = [0.88, 0.74, 0.70, 0.68, 0.67, 0.69, 0.72, 0.78, 0.87]
    np.testing.assert_allclose(nuc(nat, T + mt) / ref, want_nat, atol=0.015)
    np.testing.assert_allclose(nuc(tot, T + mt) / ref, want_tot, atol=0.015)
    np.testing.assert_allclose(nuc(h3a, T + mt) / ref, want_h3a, atol=0.015)
    # native == crflux GH evaluated at the kinetic variable itself
    np.testing.assert_allclose(nuc(nat, T + M), nuc(tot, T), rtol=1e-12)
    with pytest.raises(ValueError):
        primary_class("GaisserHonda")
    assert primary_class("GlobalSplineFitBeta") is crf.GlobalSplineFitBeta
