import numpy as np
import numpy.testing as npt
import pathlib
import pytest

from daemonflux import Flux
from daemonflux.geomagnetic import (
    GeomagneticModel,
    GeomagneticSite,
    KNOWN_SITES,
)


# ---------------------------------------------------------------------------
# Physics of the geomagnetic model (independent of the flux splines)
# ---------------------------------------------------------------------------
def test_vertical_cutoff_latitude_ordering():
    # Cutoff is largest near the geomagnetic equator and ~0 at the poles.
    eq = GeomagneticSite("eq", 0.0).vertical_cutoff_GV
    mid = GeomagneticSite("mid", 45.0).vertical_cutoff_GV
    pole = GeomagneticSite("pole", 85.0).vertical_cutoff_GV
    assert eq > mid > pole
    # Stoermer equatorial vertical cutoff is ~14.9 GV.
    npt.assert_allclose(eq, 14.9, atol=0.1)


def test_kamioka_cutoff_realistic():
    # Kamioka vertical cutoff is ~11 GV in the literature.
    m = GeomagneticModel("kamioka")
    assert 10.0 < m.site.vertical_cutoff_GV < 12.5


def test_east_west_cutoff_asymmetry():
    # For positive primaries the cutoff is lower from the West than the East,
    # strongest near the horizon.
    m = GeomagneticModel("ino")  # near equator -> strong effect
    rc_east = float(m.cutoff_rigidity_GV(80.0, 90.0))
    rc_west = float(m.cutoff_rigidity_GV(80.0, 270.0))
    assert rc_west < rc_east


def test_east_west_flux_asymmetry():
    # The admittance (hence the flux) is larger from the West at low energy and
    # symmetric (ratio -> 1) at high energy.
    m = GeomagneticModel("ino")
    E = np.array([1.0, 2.0, 5.0, 1000.0])
    g_east = m.admittance("numuflux", E, 80.0, 90.0)
    g_west = m.admittance("numuflux", E, 80.0, 270.0)
    ratio = g_west / g_east
    assert ratio[0] > 1.2  # strong asymmetry at 1 GeV
    assert ratio[-1] == pytest.approx(1.0, abs=1e-6)  # none at 1 TeV
    assert np.all(np.diff(ratio) <= 1e-9)  # monotonically decreasing with E


def test_high_energy_limit_is_unity():
    m = GeomagneticModel("ino")
    g = m.admittance("numuflux", np.array([1e4, 1e5]), 30.0, 90.0)
    npt.assert_allclose(g, 1.0, atol=1e-6)


def test_latitude_suppression_ordering():
    # Equatorial site suppresses low-energy flux more than a high-latitude one.
    E = np.array([1.0, 2.0, 5.0])
    g_ino = GeomagneticModel("ino").admittance("numuflux", E, 0.0, None)
    g_snolab = GeomagneticModel("snolab").admittance("numuflux", E, 0.0, None)
    assert np.all(g_ino <= g_snolab + 1e-9)
    assert np.all(g_ino < 1.0)


def test_polar_site_no_suppression():
    g = GeomagneticModel("southpole").admittance(
        "numuflux", np.array([0.5, 1.0, 2.0]), 0.0, None
    )
    npt.assert_allclose(g, 1.0, atol=1e-2)


def test_ratio_quantities_not_modulated():
    m = GeomagneticModel("ino")
    E = np.array([1.0, 2.0, 5.0])
    for q in ("numuratio", "muratio", "flavorratio"):
        npt.assert_array_equal(m.admittance(q, E, 80.0, 270.0), np.array(1.0))


def test_admittance_shapes():
    m = GeomagneticModel("kamioka")
    E = np.array([1.0, 2.0, 5.0])
    # scalar azimuth -> shape (nE,)
    assert m.admittance("numuflux", E, 30.0, 90.0).shape == (3,)
    # None azimuth (averaged) -> shape (nE,)
    assert m.admittance("numuflux", E, 30.0, None).shape == (3,)
    # array azimuth -> shape (naz, nE)
    az = np.array([0.0, 90.0, 180.0, 270.0])
    assert m.admittance("numuflux", E, 30.0, az).shape == (4, 3)


def test_unknown_site_raises():
    with pytest.raises(KeyError):
        GeomagneticModel("atlantis")


def test_known_sites_registry():
    assert "kamioka" in KNOWN_SITES
    assert all(isinstance(s, GeomagneticSite) for s in KNOWN_SITES.values())


# ---------------------------------------------------------------------------
# Integration with Flux (uses the bundled reduced test splines)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def flux_factory():
    basep = pathlib.Path(__file__).parent.absolute()

    def _make():
        return Flux(
            "",
            spl_file=basep / "test_daemonsplines_generic_202303_1.pkl",
            cal_file=basep / "test_calibration_default_202303_1.pkl",
            debug=0,
        )

    return _make


def test_no_model_is_backward_compatible(flux_factory):
    # Without a geomagnetic model the result is identical to the 1D model,
    # even when an azimuth is (pointlessly) supplied.
    base = flux_factory()
    E = np.logspace(0, 4, 20)
    ref = base.flux(E, "18.1949", "numuflux")
    npt.assert_array_equal(ref, base.flux(E, "18.1949", "numuflux", azimuth_deg=270.0))


def test_model_suppresses_low_energy(flux_factory):
    base = flux_factory()
    geo = flux_factory()
    geo.set_geomagnetic_model("kamioka")
    E = np.array([1.0, 2.0, 50.0, 1000.0])
    ratio = geo.flux(E, "18.1949", "numuflux", azimuth_deg=270.0) / base.flux(
        E, "18.1949", "numuflux"
    )
    assert ratio[0] < 1.0  # suppressed at 1 GeV
    npt.assert_allclose(ratio[-1], 1.0, atol=1e-6)  # untouched at 1 TeV


def test_constructor_geomag_location(flux_factory):
    basep = pathlib.Path(__file__).parent.absolute()
    geo = Flux(
        "",
        spl_file=basep / "test_daemonsplines_generic_202303_1.pkl",
        cal_file=basep / "test_calibration_default_202303_1.pkl",
        geomag_location="kamioka",
        debug=0,
    )
    g = geo[geo.supported_fluxes[0]]._geomag
    assert isinstance(g, GeomagneticModel)
    assert g.site.name == "kamioka"


def test_error_inherits_admittance(flux_factory):
    # The same admittance multiplies the absolute error, so error/flux (the
    # relative error) is preserved: the calibration covariance carries over.
    base = flux_factory()
    geo = flux_factory()
    geo.set_geomagnetic_model("kamioka")
    E = np.array([1.0, 2.0, 50.0])
    rel_1d = base.error(E, "18.1949", "numuflux") / base.flux(E, "18.1949", "numuflux")
    rel_3d = geo.error(E, "18.1949", "numuflux", azimuth_deg=270.0) / geo.flux(
        E, "18.1949", "numuflux", azimuth_deg=270.0
    )
    npt.assert_allclose(rel_1d, rel_3d, rtol=1e-9)


def test_array_zenith_with_model_raises(flux_factory):
    geo = flux_factory()
    geo.set_geomagnetic_model("kamioka")
    with pytest.raises(NotImplementedError):
        geo.flux(
            np.array([1.0, 2.0]),
            [0.0, 18.1949],
            "numuflux",
            azimuth_deg=270.0,
        )


def test_clearing_model_restores_1d(flux_factory):
    base = flux_factory()
    geo = flux_factory()
    geo.set_geomagnetic_model("kamioka")
    geo.set_geomagnetic_model(None)
    E = np.logspace(0, 3, 10)
    npt.assert_array_equal(
        base.flux(E, "18.1949", "numuflux"),
        geo.flux(E, "18.1949", "numuflux", azimuth_deg=270.0),
    )
