"""The East-West observable must SEE a coherent azimuthal shift.

The charge-dependent muon-bending term rotates the azimuthal flux pattern by
+/- ~5 deg (mu+ vs mu-).  Every earlier charge-split diagnostic quantified the
E-W asymmetry as ``max/min`` over a uniform azimuth grid, which is *exactly*
invariant under that rotation when the grid is symmetric about the pattern's
axis -- so it reported a null by construction.  These tests pin the failure of
the old statistic and the sensitivity (and correct sign) of the new one.
"""
import numpy as np
import pytest

from diag_ew_charge_fourier import (
    azimuth_harmonics,
    ew_observables,
    honda_azimuth_to_compass,
    honda_ew,
    honda_table,
    geomagnetic_ew_axis,
)

EW_AXIS = 90.0                       # geomagnetic East for the synthetic tests
WEST = EW_AXIS + 180.0
NAZ = 24
# grid symmetric under reflection about WEST (and about WEST+180): this is the
# configuration in which max/min is provably blind to the sign of a rotation.
AZ = (np.arange(NAZ) + 0.5) * 360.0 / NAZ


def dipole(delta_deg, amp=0.4, amp2=0.06, az=AZ):
    """Synthetic E-W pattern rotated by ``delta_deg`` (max toward West+delta)."""
    ph = np.radians(az - WEST - delta_deg)
    return 1.0 + amp * np.cos(ph) + amp2 * np.cos(2 * ph)


def test_maxmin_is_blind_to_the_sign_of_the_shift():
    """The legacy observable gives the SAME number for +delta and -delta."""
    for d in (2.0, 5.0, 15.0, 30.0):
        fp, fm = dipole(+d), dipole(-d)
        op = ew_observables(fp, AZ, EW_AXIS)
        om = ew_observables(fm, AZ, EW_AXIS)
        assert op["maxmin"] == pytest.approx(om["maxmin"], rel=1e-12)
        # ... and so is a shift-blind |a1|
        assert op["a1_rel"] == pytest.approx(om["a1_rel"], rel=1e-12)


def test_phase_and_transverse_component_respond_with_opposite_sign():
    """dphi and s1/a0 are odd in the shift; dphi recovers it exactly."""
    for d in (2.0, 5.0, 15.0, 30.0):
        op = ew_observables(dipole(+d), AZ, EW_AXIS)
        om = ew_observables(dipole(-d), AZ, EW_AXIS)
        assert op["dphi"] == pytest.approx(+d, abs=1e-6)
        assert om["dphi"] == pytest.approx(-d, abs=1e-6)
        assert op["s1_rel"] == pytest.approx(-om["s1_rel"], rel=1e-9)
        assert op["s1_rel"] > 0 > om["s1_rel"]
        # the even (along-axis) part is what max/min already saw
        assert op["c1_rel"] == pytest.approx(om["c1_rel"], rel=1e-9)


def test_transverse_component_is_linear_in_a_small_shift():
    """For a ~5 deg bending shift, s1/a0 ~= (a1/a0) * delta -- a first-order,
    signed handle where max/min is second order (and here exactly zero)."""
    a1 = 0.4
    o0 = ew_observables(dipole(0.0, amp=a1), AZ, EW_AXIS)
    o5 = ew_observables(dipole(5.0, amp=a1), AZ, EW_AXIS)
    assert o0["s1_rel"] == pytest.approx(0.0, abs=1e-12)
    assert o5["s1_rel"] == pytest.approx(a1 * np.radians(5.0), rel=0.02)
    # max/min barely moves (0.5%, and it is EVEN in the shift, so it cannot
    # tell the two charges apart) while s1/a0 goes from 0 to 0.035 with a sign
    assert abs(o5["maxmin"] / o0["maxmin"] - 1) < 0.01
    assert ew_observables(dipole(-5.0, amp=a1), AZ, EW_AXIS)["maxmin"] == \
        pytest.approx(o5["maxmin"], rel=1e-12)
    assert o5["s1_rel"] > 0.03


def test_bin_average_deconvolution():
    """Honda's rows are 30-deg bin averages; the sinc correction recovers the
    underlying harmonic amplitudes (phase is unaffected by binning)."""
    az_c = np.arange(12) * 30.0 + 15.0
    fine = np.linspace(-15.0, 15.0, 201)
    binned = np.array([dipole(7.0, az=a + fine).mean() for a in az_c])
    raw = azimuth_harmonics(binned, az_c, n_harm=2)
    cor = azimuth_harmonics(binned, az_c, bin_width_deg=30.0, n_harm=2)
    assert raw["a"][1] == pytest.approx(0.4 * np.sinc(1 / 12.0), rel=2e-3)
    assert cor["a"][1] == pytest.approx(0.4, rel=2e-3)
    assert cor["a"][2] == pytest.approx(0.06, rel=5e-3)
    assert cor["phi"][1] == pytest.approx(raw["phi"][1], abs=1e-9)


def test_shift_survives_honda_binning():
    """A +/-5 deg rotation is resolvable through Honda's 12 x 30-deg binning."""
    az_c = np.arange(12) * 30.0 + 15.0
    fine = np.linspace(-15.0, 15.0, 201)
    out = {}
    for d in (+5.0, -5.0):
        b = np.array([dipole(d, az=a + fine).mean() for a in az_c])
        out[d] = ew_observables(b, az_c, EW_AXIS, bin_width_deg=30.0)
    assert out[+5.0]["maxmin"] == pytest.approx(out[-5.0]["maxmin"], rel=1e-12)
    assert out[+5.0]["dphi"] == pytest.approx(+5.0, abs=0.05)
    assert out[-5.0]["dphi"] == pytest.approx(-5.0, abs=0.05)


def test_honda_shows_a_charge_dependent_phase_split():
    """Honda's own table carries the charge-dependent AZIMUTHAL SHIFT that the
    max/min statistic cannot express.

    At 87 deg / 0.3 GeV the nu_e pattern sits 12.2 deg round from nubar_e and
    nu_mu 5.9 deg the OTHER way -- both of order the 5.08 deg per-lifetime muon
    bending, with the sign set by the parent muon charge (nu_e and nubar_mu from
    mu+, nu_mu and nubar_e from mu-), and with the ratio of the two splittings
    (2.1) matching the ratio of muon-decay channel fractions (nu_e is ~all
    mu-decay, nu_mu only ~45%).

    Sign convention: the primary of a mu+ decay neutrino arriving from compass
    azimuth phi came from phi + ~4 deg, so the mu+ species' pattern is rotated to
    SMALLER compass azimuth -- d(dphi) < 0 for nu_e minus nubar_e.  Getting this
    right requires mapping Honda's azimuth (counterclockwise from South) onto the
    compass convention; without that mapping the split comes out with the wrong
    sign, which is why this test also pins the mapping.
    """
    h = honda_table()
    ax = geomagnetic_ew_axis()
    o = {k: honda_ew(h, k, 0.05, 0.3, ax)
         for k in ("numu", "numubar", "nue", "nuebar")}
    d_nue = o["nue"]["dphi"] - o["nuebar"]["dphi"]
    d_numu = o["numu"]["dphi"] - o["numubar"]["dphi"]
    assert d_nue < -8.0                     # measured -12.2 deg
    assert d_numu > 3.0                     # measured  +5.9 deg
    assert d_nue < 0 < d_numu               # opposite, as the mu charge demands
    assert abs(d_nue) > abs(d_numu)         # nu_e is the purer mu-decay channel
    # the transverse dipole component splits too, same sign
    assert o["nue"]["s1_rel"] - o["nuebar"]["s1_rel"] < -0.03
    assert o["numu"]["s1_rel"] - o["numubar"]["s1_rel"] > 0.02


def test_honda_azimuth_mapping_is_a_mirror_about_the_ew_axis():
    """The Honda -> compass mapping must fix East and West and swap North/South."""
    assert honda_azimuth_to_compass(90.0) == pytest.approx(90.0)    # East
    assert honda_azimuth_to_compass(270.0) == pytest.approx(270.0)  # West
    assert honda_azimuth_to_compass(0.0) == pytest.approx(180.0)    # S -> S(compass)
    assert honda_azimuth_to_compass(180.0) == pytest.approx(0.0)    # N -> N(compass)
    # applying it twice is the identity
    a = np.array([15.0, 45.0, 105.0, 285.0])
    assert honda_azimuth_to_compass(honda_azimuth_to_compass(a)) == \
        pytest.approx(a)
