"""Tabulated, per-species 3D factor (``Tabulated3DModel``) and its use in ``Flux``."""

import pathlib
import warnings

import numpy as np
import numpy.testing as npt
import pytest

from daemonflux import Flux
from daemonflux.geomagnetic import Tabulated3DModel

SPECIES = ("numu", "antinumu", "nue", "antinue")
E_EDGES = np.logspace(0.0, 2.0, 11)  # 1-100 GeV, 5/decade
CZ_EDGES = np.linspace(-1.0, 1.0, 11)
AZ_EDGES = np.linspace(0.0, 360.0, 7)  # 6 bins of 60 deg, centres 30, 90, ...
A_S = {"numu": 1.10, "antinumu": 0.95, "nue": 1.30, "antinue": 0.80}


def _centres(e):
    return 0.5 * (e[:-1] + e[1:])


def toy_factor(sp, E, cz):
    """Closed form, linear in ln E and in cos Z, independent of azimuth."""
    return A_S[sp] * (1.0 + 0.05 * np.log(E / 10.0)) * (1.0 + 0.2 * cz)


def toy_table(az_dependence=0.0):
    ec = np.sqrt(E_EDGES[:-1] * E_EDGES[1:])  # geometric centres
    cc = _centres(CZ_EDGES)
    ac = _centres(AZ_EDGES)
    f = np.zeros((4, len(ec), len(cc), len(ac)))
    for k, sp in enumerate(SPECIES):
        f[k] = toy_factor(sp, ec[:, None, None], cc[None, :, None]) * (
            1.0 + az_dependence * np.cos(np.deg2rad(ac))[None, None, :]
        )
    return dict(
        species=np.array(SPECIES),
        e_edges=E_EDGES,
        cosz_edges=CZ_EDGES,
        az_edges=AZ_EDGES,
        factor=f,
        factor_error=0.01 * f,
        label="toy",
        status="TEST",
    )


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


E = np.array([1.5, 3.0, 10.0, 30.0, 300.0])
E_IN = E[:4]  # inside the table (1-100 GeV)
QUANTITIES = [
    "numu", "antinumu", "nue", "antinue", "numuflux", "nueflux", "muflux",
    "numuratio", "nueratio", "muratio", "flavorratio", "mu+", "mu-",
]  # fmt: skip
QUANTITIES += ["total_" + q for q in QUANTITIES]


# ---------------------------------------------------------------------------
# backward compatibility
# ---------------------------------------------------------------------------
def test_no_model_is_bit_identical_and_clearing_restores_it(flux_factory):
    base = flux_factory()
    tab = flux_factory()
    tab.set_geomagnetic_model(Tabulated3DModel(toy_table()))
    tab.set_geomagnetic_model(None)
    for q in QUANTITIES:
        for z in (0.0, "18.1949", 10.0):
            npt.assert_array_equal(base.flux(E, z, q), tab.flux(E, z, q))
            npt.assert_array_equal(base.error(E, z, q), tab.error(E, z, q))
            npt.assert_array_equal(
                base.flux(E, z, q), base.flux(E, z, q, azimuth_deg=10.0)
            )


def test_unit_table_reproduces_the_1d_model(flux_factory):
    t = toy_table()
    t["factor"] = np.ones_like(t["factor"])
    t["factor_error"] = np.zeros_like(t["factor"])
    base, geo = flux_factory(), flux_factory()
    geo.set_geomagnetic_model(Tabulated3DModel(t))
    for q in QUANTITIES:
        npt.assert_allclose(
            geo.flux(E, 10.0, q, azimuth_deg=100.0), base.flux(E, 10.0, q), rtol=1e-12
        )
        npt.assert_allclose(
            geo.error(E, 10.0, q, azimuth_deg=100.0), base.error(E, 10.0, q), rtol=1e-12
        )


# ---------------------------------------------------------------------------
# interpolation: closed-form toy, edges, periodicity
# ---------------------------------------------------------------------------
def test_toy_table_is_reproduced_between_centres():
    m = Tabulated3DModel(toy_table())
    ec = np.sqrt(E_EDGES[0] * E_EDGES[1])
    E = np.array([ec, 2.0, 7.3, 25.0, 60.0])
    for z in (100.0, 120.0, 150.0, 60.0, 30.0):  # cos Z between the centres
        cz = np.cos(np.deg2rad(z))
        if abs(cz) > 0.9:
            continue
        for sp in SPECIES:
            npt.assert_allclose(
                m.factor(sp, E, z, 45.0), toy_factor(sp, E, cz), rtol=1e-12
            )
            npt.assert_allclose(
                m.factor(sp, E, z, None), toy_factor(sp, E, cz), rtol=1e-12
            )


def test_value_at_a_bin_centre_is_the_table_value_and_flat_beyond():
    t = toy_table()
    m = Tabulated3DModel(t)
    ec = np.sqrt(E_EDGES[3] * E_EDGES[4])
    cz = _centres(CZ_EDGES)[7]
    az = _centres(AZ_EDGES)[2]
    z = np.rad2deg(np.arccos(cz))
    npt.assert_allclose(m.factor("nue", ec, z, az), t["factor"][2, 3, 7, 2], rtol=1e-12)
    # between the outermost centre and the edge: the outermost value (flat)
    zc = np.rad2deg(np.arccos(_centres(CZ_EDGES)[-1]))
    z_edge = np.rad2deg(np.arccos(0.999))
    npt.assert_allclose(
        m.factor("numu", 5.0, z_edge, 30.0), m.factor("numu", 5.0, zc, 30.0), rtol=1e-12
    )


def test_hemispheres_are_interpolated_separately():
    """Up- and down-going patterns are never blended across the horizon: between the
    horizon and the first centre of a hemisphere, that hemisphere's bin value is used.
    """
    t = toy_table()
    f = np.where(_centres(CZ_EDGES)[None, None, :, None] > 0, 1.0, 2.0) * np.ones_like(
        t["factor"]
    )
    m = Tabulated3DModel(dict(t, factor=f))
    for z, want in ((89.0, 1.0), (85.0, 1.0), (90.0, 1.0), (91.0, 2.0), (95.0, 2.0)):
        npt.assert_allclose(m.factor("numu", 5.0, z, 45.0), want, rtol=0, atol=1e-12)
    # within a hemisphere the interpolation is unchanged (toy linear in cos Z)
    m2 = Tabulated3DModel(toy_table())
    for z in (60.0, 100.0, 120.0):
        cz = np.cos(np.deg2rad(z))
        npt.assert_allclose(
            m2.factor("nue", 7.3, z, 45.0), toy_factor("nue", 7.3, cz), rtol=1e-12
        )
    # a down-going-only table (no bins below the horizon) is not split
    dn = dict(t, cosz_edges=np.linspace(0.0, 1.0, 11))
    assert not Tabulated3DModel(dn)._split


def test_above_emax_blends_to_one_and_below_emin_warns_or_raises():
    m = Tabulated3DModel(toy_table(), blend_decades=1.0)
    f_top = m.factor("nue", 100.0, 60.0, 30.0)
    assert f_top[0] != 1.0
    npt.assert_allclose(
        m.factor("nue", 100.0 * 10**0.5, 60.0, 30.0), 0.5 * (f_top + 1.0)
    )
    npt.assert_allclose(m.factor("nue", 1000.0, 60.0, 30.0), 1.0, atol=1e-12)
    npt.assert_allclose(m.factor("nue", 1e5, 60.0, 30.0), 1.0, atol=1e-12)
    npt.assert_allclose(m.factor_error("nue", 1e4, 60.0, 30.0), 0.0, atol=1e-12)
    with pytest.warns(UserWarning, match="below the table"):
        low = m.factor("nue", 0.5, 60.0, 30.0)
    npt.assert_allclose(low, m.factor("nue", 1.0, 60.0, 30.0))
    with pytest.raises(ValueError, match="below the table"):
        Tabulated3DModel(toy_table(), below_emin="raise").factor("nue", 0.5, 60.0, 30.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m.factor("nue", 1.0, 60.0, 30.0)  # exactly E_min: no warning


def test_zenith_outside_table_coverage_raises():
    t = toy_table()
    keep = CZ_EDGES >= 0.0  # a down-going-only table
    t["cosz_edges"] = CZ_EDGES[keep]
    t["factor"] = t["factor"][:, :, CZ_EDGES[:-1] >= 0.0, :]
    t["factor_error"] = t["factor_error"][:, :, CZ_EDGES[:-1] >= 0.0, :]
    m = Tabulated3DModel(t)
    m.factor("numu", 5.0, 30.0, 0.0)
    with pytest.raises(ValueError, match="outside the table"):
        m.factor("numu", 5.0, 120.0, 0.0)


def test_azimuth_is_periodic_and_interpolated():
    m = Tabulated3DModel(toy_table(az_dependence=0.3))
    E1 = np.array([5.0, 20.0])
    for sp in SPECIES:
        for a in (0.0, 17.0, 123.0, 359.0):
            npt.assert_allclose(
                m.factor(sp, E1, 60.0, a), m.factor(sp, E1, 60.0, a + 360.0), rtol=1e-12
            )
            npt.assert_allclose(
                m.factor(sp, E1, 60.0, a), m.factor(sp, E1, 60.0, a - 720.0), rtol=1e-12
            )
    # azimuth 0 lies midway between the centres at 330 and 30 deg: the mean of the two
    f0 = m.factor("numu", E1, 60.0, 0.0)
    npt.assert_allclose(
        f0, 0.5 * (m.factor("numu", E1, 60.0, 330.0) + m.factor("numu", E1, 60.0, 30.0))
    )
    # vector azimuth: shape (naz, nE), consistent with the scalar calls
    az = np.array([0.0, 90.0, 200.0])
    v = m.factor("numu", E1, 60.0, az)
    assert v.shape == (3, 2)
    for i, a in enumerate(az):
        npt.assert_allclose(v[i], m.factor("numu", E1, 60.0, a))
    # azimuth average = mean over the bin centres
    t = toy_table(az_dependence=0.3)
    npt.assert_allclose(
        m.factor("numu", E1, 60.0, None),
        0.5 * 0
        + toy_factor("numu", E1, np.cos(np.deg2rad(60.0)))
        * (1.0 + 0.3 * np.cos(np.deg2rad(_centres(AZ_EDGES))).mean()),
        rtol=1e-10,
    )
    del t


# ---------------------------------------------------------------------------
# quantities: species, sums, ratios, errors, up-going
# ---------------------------------------------------------------------------
def _setup(flux_factory):
    base, geo = flux_factory(), flux_factory()
    model = Tabulated3DModel(toy_table(az_dependence=0.2))
    geo.set_geomagnetic_model(model)
    return base, geo, model


def test_species_sums_and_ratios_are_recombined(flux_factory):
    base, geo, model = _setup(flux_factory)
    z, az = 10.0, 100.0
    for pre in ("", "total_"):
        n, a, e, ae = (base.flux(E, z, pre + s) for s in SPECIES)
        fn, fa, fe, fae = (model.factor(s, E, z, az) for s in SPECIES)
        g = lambda q: geo.flux(E, z, pre + q, azimuth_deg=az)  # noqa: E731
        b = lambda q: base.flux(E, z, pre + q)  # noqa: E731
        npt.assert_allclose(g("numu"), n * fn, rtol=1e-12)
        npt.assert_allclose(g("antinue"), ae * fae, rtol=1e-12)
        npt.assert_allclose(g("numuflux") / b("numuflux"), (n * fn + a * fa) / (n + a))
        npt.assert_allclose(g("nueflux") / b("nueflux"), (e * fe + ae * fae) / (e + ae))
        npt.assert_allclose(g("numuratio") / b("numuratio"), fn / fa, rtol=1e-12)
        npt.assert_allclose(g("nueratio") / b("nueratio"), fe / fae, rtol=1e-12)
        flav = ((n * fn + a * fa) / (n + a)) / ((e * fe + ae * fae) / (e + ae))
        npt.assert_allclose(g("flavorratio") / b("flavorratio"), flav, rtol=1e-12)
        # the corrected ratio is consistent with the corrected components
        npt.assert_allclose(
            g("numuratio") / b("numuratio"), (g("numu") / n) / (g("antinumu") / a)
        )
        # muons are not in a neutrino table: factor 1
        npt.assert_array_equal(g("mu+"), b("mu+"))
        npt.assert_array_equal(g("muratio"), b("muratio"))


def test_equal_factors_leave_ratios_unchanged_and_scale_sums(flux_factory):
    E = E_IN
    t = toy_table()
    t["factor"] = np.full_like(t["factor"], 1.25)
    t["factor_error"] = np.zeros_like(t["factor"])
    base, geo = flux_factory(), flux_factory()
    geo.set_geomagnetic_model(Tabulated3DModel(t))
    for q in ("numuratio", "nueratio", "flavorratio"):
        npt.assert_allclose(
            geo.flux(E, 10.0, q, azimuth_deg=90.0), base.flux(E, 10.0, q)
        )
    npt.assert_allclose(
        geo.flux(E, 10.0, "numuflux", azimuth_deg=90.0),
        1.25 * base.flux(E, 10.0, "numuflux"),
    )


def test_errors_scale_with_the_flux_and_table_error_is_separate(flux_factory):
    E = E_IN
    base, geo, model = _setup(flux_factory)
    z, az = 10.0, 100.0
    for q in ("numu", "numuflux", "numuratio", "flavorratio"):
        scale = geo.flux(E, z, q, azimuth_deg=az) / base.flux(E, z, q)
        npt.assert_allclose(
            geo.error(E, z, q, azimuth_deg=az), base.error(E, z, q) * scale, rtol=1e-12
        )
    # the table's own (1% of the factor in the toy) uncertainty is exposed separately
    rel = geo.geomagnetic_factor_error(E, z, "numu", azimuth_deg=az)
    npt.assert_allclose(rel, 0.01, rtol=1e-9)
    rel_ratio = geo.geomagnetic_factor_error(E, z, "numuratio", azimuth_deg=az)
    npt.assert_allclose(rel_ratio, 0.01 * np.sqrt(2.0), rtol=1e-9)
    rel_sum = geo.geomagnetic_factor_error(E, z, "numuflux", azimuth_deg=az)
    assert np.all(rel_sum <= 0.01 + 1e-12) and np.all(rel_sum > 0.0055)
    with pytest.raises(RuntimeError):
        base.geomagnetic_factor_error(E, z, "numu")


def test_upgoing_zenith_uses_the_complementary_1d_flux(flux_factory):
    base, geo, model = _setup(flux_factory)
    z_up = 170.0  # cos Z = -0.985; the 1D flux is that of zenith 10 deg
    for q in ("numu", "numuflux", "flavorratio"):
        got = geo.flux(E, z_up, q, azimuth_deg=250.0)
        expect = base.flux(E, 10.0, q) * model.scale(
            q,
            E,
            z_up,
            250.0,
            lambda name: base.flux(E, 10.0, name),
        )
        npt.assert_allclose(got, expect, rtol=1e-12)
    # without a tabulated model up-going zeniths remain an error (unchanged behaviour)
    with pytest.raises(ValueError):
        base.flux(E, z_up, "numu")


def test_npz_round_trip_and_set_geomagnetic_model_from_path(tmp_path, flux_factory):
    p = tmp_path / "toy.npz"
    np.savez(p, **toy_table())
    m = Tabulated3DModel(p)
    assert m.label == "toy" and m.status == "TEST" and "TEST" in repr(m)
    geo = flux_factory()
    geo.set_geomagnetic_model(p)
    assert isinstance(geo[geo.supported_fluxes[0]]._geomag, Tabulated3DModel)
    geo.set_geomagnetic_model(str(p))
    with pytest.raises(TypeError):
        geo.set_geomagnetic_model(3.0)


def test_invalid_tables_are_rejected():
    t = toy_table()
    bad = dict(t, factor=np.where(t["factor"] > 0, np.nan, 0.0))
    with pytest.raises(ValueError, match="finite"):
        Tabulated3DModel(bad)
    with pytest.raises(ValueError, match="shape"):
        Tabulated3DModel(dict(t, factor=t["factor"][:, :-1]))
    with pytest.raises(ValueError, match="span 360"):
        Tabulated3DModel(dict(t, az_edges=np.linspace(0, 300, 7)))
    with pytest.raises(ValueError, match="unknown species"):
        Tabulated3DModel(dict(t, species=np.array(["numu", "x", "nue", "antinue"])))


def test_provisional_kamioka_table_if_available(flux_factory):
    """The PROVISIONAL Kamioka table (``$DAEMONFLUX_3D_TABLE``) loads, says so, covers the
    full sphere and gives a finite flux for down- and up-going directions."""
    import os

    path = os.environ.get("DAEMONFLUX_3D_TABLE", "")
    if not path or not os.path.exists(path):
        pytest.skip("set DAEMONFLUX_3D_TABLE to the provisional table to run this")
    m = Tabulated3DModel(path)
    assert m.status == "PROVISIONAL" and "PROVISIONAL" in repr(m)
    assert m.cosz_edges[0] == -1.0 and m.cosz_edges[-1] == 1.0
    geo = flux_factory()
    geo.set_geomagnetic_model(m)
    for z in (10.0, 170.0):
        f = geo.flux(np.array([1.5, 5.0]), z, "numuflux", azimuth_deg=100.0)
        assert np.all(np.isfinite(f)) and np.all(f > 0)


def test_array_zenith_with_a_tabulated_model_is_rejected_clearly(flux_factory):
    _, geo, _ = _setup(flux_factory)
    for z in (np.array([10.0, 20.0]), [170.0, 175.0], np.array([100.0, 110.0])):
        with pytest.raises(NotImplementedError, match="single zenith"):
            geo.flux(E_IN, z, "numu", azimuth_deg=10.0)
        with pytest.raises(NotImplementedError, match="single zenith"):
            geo.error(E_IN, z, "numu", azimuth_deg=10.0)
    # a scalar up-going zenith, and the 1D model, are unaffected
    assert np.all(np.isfinite(geo.flux(E_IN, 170.0, "numu", azimuth_deg=10.0)))


def test_uncovered_quantities_pass_through_with_factor_one(flux_factory):
    base, geo, model = _setup(flux_factory)
    # muon quantities: not in a neutrino table -> identical to 1D, also with total_,
    # at any zenith (even outside the table's cos Z coverage) and any azimuth
    for q in ("muflux", "muratio", "mu+", "mu-", "total_muflux", "total_muratio"):
        for z in (10.0, 170.0):
            for az in (None, 100.0, np.array([10.0, 200.0])):
                got = geo.flux(E_IN, z, q, azimuth_deg=az)
                z1 = 180.0 - z if z > 90.0 else z
                npt.assert_array_equal(
                    got, np.broadcast_to(base.flux(E_IN, z1, q), got.shape)
                )
            npt.assert_array_equal(
                geo.error(E_IN, z, q, azimuth_deg=100.0), base.error(E_IN, z1, q)
            )
        assert np.all(
            geo.geomagnetic_factor_error(E_IN, 10.0, q, azimuth_deg=100.0) == 0
        )
    # a table that covers only part of the species: sums use 1 for the missing one
    t = toy_table()
    keep = [0, 1]  # numu, antinumu only
    t["species"] = t["species"][keep]
    t["factor"], t["factor_error"] = t["factor"][keep], t["factor_error"][keep]
    m = Tabulated3DModel(t)
    assert m.covers("numuflux") and not m.covers("nueflux") and not m.covers("muflux")
    one = m.scale("nueflux", E_IN, 10.0, 100.0, lambda q: None)
    npt.assert_array_equal(one, 1.0)
    weights = {"numu": 2.0, "antinumu": 1.0, "nue": 3.0, "antinue": 1.0}

    def comp(q):
        return weights[q]

    sc = m.scale("flavorratio", E_IN, 10.0, 100.0, comp)
    f = m.factor("numu", E_IN, 10.0, 100.0), m.factor("antinumu", E_IN, 10.0, 100.0)
    npt.assert_allclose(sc, (2 * f[0] + f[1]) / 3.0)  # nue/antinue factors are 1


def test_kamioka_v1_table_if_available(flux_factory):
    """The final Kamioka table v1 (if present): loads, carries its provenance, evaluates
    down- and up-going, no NaN, errors positive."""
    import os

    path = os.environ.get("DAEMONFLUX_3D_TABLE_V1", "")
    if not path or not os.path.exists(path):
        pytest.skip("set DAEMONFLUX_3D_TABLE_V1 to kamioka_3D_v1.npz to run this")
    d = np.load(path, allow_pickle=False)
    for k in (
        "input_sha256_G_down",
        "input_sha256_Rmu",
        "db_label",
        "route_R_up",
        "field",
    ):
        assert k in d.files
    m = Tabulated3DModel(path)
    assert m.label in ("kamioka_3D_v1", "kamioka_3D_v1b", "kamioka_3D_v1c")
    assert m.status.startswith("v1")
    assert np.all(m._df > 0) and np.all(np.isfinite(m._f))
    geo = flux_factory()
    geo.set_geomagnetic_model(m)
    for z in (10.0, 170.0):
        f = geo.flux(np.array([1.5, 5.0]), z, "numuflux", azimuth_deg=100.0)
        assert np.all(np.isfinite(f)) and np.all(f > 0)
        assert np.all(
            geo.geomagnetic_factor_error(np.array([1.5]), z, "numu", azimuth_deg=10.0)
            > 0
        )


def test_kamioka_v2h_table_if_available(flux_factory):
    """The candidate Kamioka hybrid table v2h (if present): loads, carries the hybrid
    provenance, factor = factor_3d x hybrid_correction, evaluates down- and up-going with
    finite positive values and errors."""
    import os

    path = os.environ.get("DAEMONFLUX_3D_TABLE_V2H", "")
    if not path or not os.path.exists(path):
        pytest.skip("set DAEMONFLUX_3D_TABLE_V2H to kamioka_3D_v2h.npz to run this")
    d = np.load(path, allow_pickle=False)
    for k in (
        "hybrid_correction",
        "factor_3d",
        "err_hybrid_rel",
        "hybrid_e0",
        "hybrid_w_formula",
        "engine_eoff_model",
        "input_sha256_v1c",
        "db_label",
    ):
        assert k in d.files
    npt.assert_allclose(
        d["factor"], d["factor_3d"] * d["hybrid_correction"][..., None], rtol=1e-12
    )
    m = Tabulated3DModel(path)
    assert m.label == "kamioka_3D_v2h"
    assert m.status == "candidate"
    assert np.all(m._df > 0) and np.all(np.isfinite(m._f))
    geo = flux_factory()
    geo.set_geomagnetic_model(m)
    for z in (10.0, 170.0):  # the bundled test splines cover zenith 0-18 deg
        f = geo.flux(np.array([1.5, 5.0]), z, "numuflux", azimuth_deg=100.0)
        assert np.all(np.isfinite(f)) and np.all(f > 0)
        ge = geo.geomagnetic_factor_error(np.array([1.5]), z, "numu", azimuth_deg=10.0)
        assert np.all(np.isfinite(ge)) and np.all(ge > 0)
