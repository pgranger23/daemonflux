"""Milestone 3d: per-model native energy variable, GST / GSF-groups primary
models, the generalised exact re-weighting factor."""

import numpy as np
import pytest

import primaries as prim

T = np.array([1.0, 2.0, 3.0, 10.0, 30.0, 100.0])


def test_native_energy_variable_per_model():
    """H3a, GST, GSF fit the total energy per particle; Gaisser-Honda's
    formula takes the kinetic energy per nucleon (crflux feeds it E/A)."""
    crf = pytest.importorskip("crflux.models")
    assert prim.native_convention(crf.HillasGaisser2012("H3a")) == "total"
    assert prim.native_convention(crf.GaisserStanevTilav("3-gen")) == "total"
    assert prim.native_convention(crf.GlobalSplineFitBeta()) == "total"
    assert prim.native_convention(crf.GaisserHonda()) == "kinetic"
    for nm, want in (("H3A", "total"), ("GST", "total"), ("GSF", "total"),
                     ("GH", "kinetic"), ("GSFG:GH", "total")):
        assert prim.native_for_name(nm) == want
    with pytest.raises(ValueError):
        prim.native_for_name("ZS")
    # "native" is resolved per model, "kinetic" / "total" override it
    gh, h3 = prim.primary_model("GH"), prim.primary_model("H3A")
    for gid, a in ((14, 1), (402, 4)):
        assert np.array_equal(prim.nucleon_intensity(gh, gid, T, a, "native"),
                              prim.nucleon_intensity(gh, gid, T, a, "kinetic"))
        assert np.array_equal(prim.nucleon_intensity(h3, gid, T, a, "native"),
                              prim.nucleon_intensity(h3, gid, T, a, "total"))
    assert not np.allclose(prim.nucleon_intensity(gh, 14, T, 1, "total"),
                           prim.nucleon_intensity(gh, 14, T, 1, "kinetic"))
    with pytest.raises(ValueError):
        prim.crflux_energy(T, 1, "native")          # needs the model
    # the all-nucleon flux of every model in its own variable is within 20% of
    # GSF (total) at 5-30 GeV/n for GH, and GSF's own p_and_n_flux sum
    gsf = prim.primary_model("GSF")
    ref = gsf.nucleon_flux(T)[0]
    tot = sum(prim.nucleon_intensity(gh, g, T, a, "native")
              for _n, a, _z, g in prim.SPECIES)
    assert np.all(np.abs(tot / ref - 1)[2:5] < 0.2)


def test_gst_groups_cnoc_and_mgsi():
    crf = pytest.importorskip("crflux.models")
    m = crf.GaisserStanevTilav("3-gen")
    z, a = 0, 0
    cno = prim.nucleon_intensity(m, 1407, T, 14, "total")
    want = sum(m.nucleus_flux(c, m.Z_A(c)[1] * (T + prim.M_NUCLEON))
               * m.Z_A(c)[1] ** 2 * 1e-4 for c in (1206, 1608))
    assert np.allclose(cno, want, rtol=1e-12)
    assert np.all(prim.nucleon_intensity(m, 2512, T, 25, "total") == 0)
    p = prim.nucleon_intensity(m, 14, T, 1, "total")
    assert np.allclose(p, m.nucleus_flux(14, T + prim.M_NUCLEON) * 1e-4)
    assert z == a == 0


@pytest.mark.parametrize("ref", ["H3A", "GST", "GH"])
def test_gsf_groups_reproduce_gsf_proton_and_neutron_flux(ref):
    g = prim.primary_model("GSFG:" + ref)
    gsf = prim.primary_model("GSF")
    jp = sum(g.group_split(gid, T)[0] for _n, _a, _z, gid in prim.SPECIES)
    jn = sum(g.group_split(gid, T)[1] for _n, _a, _z, gid in prim.SPECIES)
    _f, pf, nf = gsf._m.p_and_n_flux(T + prim.M_NUCLEON)
    assert np.allclose(jp, pf * 1e-4, rtol=1e-12)
    assert np.allclose(jn, nf * 1e-4, rtol=1e-12)
    # protons only in the p group have A/Z = 1: no neutrons there
    assert np.all(g.group_split(14, T)[1] == 0)
    # and the group sampler uses the split
    import sky
    rng = np.random.default_rng(5)
    axis = np.array([0.0, 0.0, 1.0])
    _e, is_p, w, _r, _u = sky.primary_sample(rng, g, "p", 2, 40, 40, 40.0, axis)
    assert is_p.all() and (w > 0).all()
    _e, is_p, w, _r, _u = sky.primary_sample(rng, g, "Fe", 2, 400, 400, 40.0,
                                             axis)
    assert 0.2 < is_p.mean() < 0.8


def test_reweight_factor_matches_the_weights_of_a_rerun():
    """``reweight_factor`` = the ratio of the weights ``primary_sample``
    gives for the same seed under the two models (exactness of the replay)."""
    import sky
    axis = np.array([0.0, 0.0, 1.0])
    new = (prim.primary_model("GH"), "native")
    old = (prim.primary_model("H3A"), "kinetic")
    for gname, gid in (("p", 14), ("He", 402), ("MgSi", 2512)):
        out = {}
        for key, (mdl, conv) in (("new", new), ("old", old)):
            prim.ENERGY_CONVENTION = conv
            try:
                out[key] = sky.primary_sample(np.random.default_rng(9), mdl,
                                              gname, 1, 60, 60, 40.0, axis)
            finally:
                prim.ENERGY_CONVENTION = "native"
        e = out["new"][0]
        f = prim.reweight_factor(new, old, gid, e)
        assert np.allclose(out["new"][2] / out["old"][2], f, rtol=1e-12)
    # identical models -> 1; GSFG needs the identity and conserves nucleons
    assert np.allclose(prim.reweight_factor(old, old, 402, T), 1.0)
    g = (prim.primary_model("GSFG"), "native")
    with pytest.raises(ValueError):
        prim.reweight_factor(g, old, 402, T)
    isp = np.zeros(len(T), bool)
    f_n = prim.reweight_factor(g, (prim.primary_model("H3A"), "total"), 402,
                               T, isp)
    f_p = prim.reweight_factor(g, (prim.primary_model("H3A"), "total"), 402,
                               T, ~isp)
    jp, jn = g[0].group_split(402, T)
    h = prim.primary_model("H3A")
    j = prim.nucleon_intensity(h, 402, T, 4, "total")
    assert np.allclose(0.5 * (f_p + f_n) * j, jp + jn, rtol=1e-12)


def test_options_signature_resolves_native_per_model():
    import sky3d
    k = sky3d.Options(primary_energy="kinetic").signature()
    t = sky3d.Options(primary_energy="total").signature()
    assert len(k) == 9 and len(t) == 12
    # native: H3a -> total (milestone 3c's default manifests), GH -> kinetic
    assert np.array_equal(sky3d.Options(primary_model="H3A").signature(), t)
    assert np.array_equal(sky3d.Options(primary_model="GSF").signature(), t)
    assert np.array_equal(sky3d.Options(primary_model="GH").signature(), k)
    assert sky3d.Options(primary_model="GH").primary_energy == "kinetic"
    assert sky3d.Options(primary_model="GH",
                         primary_energy="total").primary_energy == "total"


def _fake_pool(pl, e0, events):
    """Inject a one-bin pool of hand-made events ``[(pid, px, py, pz), ...]``
    into an ``EventPool`` (bin 0 of the 2.0-GeV-floor grid)."""
    pid, px, py, pz, off = [], [], [], [], [0]
    for ev in events:
        for q, x, y, z in ev:
            pid.append(q), px.append(x), py.append(y), pz.append(z)
        off.append(len(pid))
    pl._pools[(2212, 0)] = {
        "n": np.array([len(events)]), "off": np.array(off),
        "pid": np.array(pid), "px": np.array(px, np.float32),
        "py": np.array(py, np.float32), "pz": np.array(pz, np.float32),
        "e0": np.array([e0])}


def test_energy_guard_budget_includes_the_struck_nucleus():
    """The guard of ``EventPool.draw`` (no energy creation) must not fire on a
    physical low-energy event whose kept products include the target's slow
    fragments: p + O at E_tot = 2.4 GeV with 3 evaporation nucleons (2.8 GeV of
    rest mass) and two pions has sum(E) = 5.7 GeV > E_tot + 0.94 = 3.4 GeV, the
    single-nucleon budget of milestones 1-3c (which then scaled every energy by
    0.6), but is within E_tot + 14.9 GeV."""
    import pool
    pl = pool.EventPool(None, n_pool=1)
    ev = [(2212, 0.0, 0.0, 1.2), (2212, 0.1, 0.0, 0.1), (2112, 0.0, 0.1, 0.1),
          (2112, -0.1, 0.0, 0.05), (211, 0.2, 0.0, 0.5), (-211, 0.0, 0.2, 0.3)]
    _fake_pool(pl, 2.24, [ev])
    e_tot = 2.44
    out = pl.draw(np.random.default_rng(1), 2212, e_tot)
    f = pool._p_beam(2212, e_tot) / pool._p_beam(2212, 2.24)
    for (q, e, p), (q0, x, y, z) in zip(out, ev):
        m = pool.MASS[q0]
        assert q == q0
        assert p[2] == pytest.approx(z * f, rel=1e-6)           # unscaled
        assert e == pytest.approx(np.sqrt(m * m + x * x + y * y + (z * f) ** 2),
                                  rel=1e-6)
    assert sum(o[1] for o in out) > e_tot + 0.94      # the old budget: exceeded
    # a pathological event (more energy than projectile + heaviest nucleus) is
    # still scaled down to the budget
    bad = [(211, 0.0, 0.0, 30.0), (211, 0.0, 0.0, 25.0)]
    _fake_pool(pl, 2.24, [bad])
    out = pl.draw(np.random.default_rng(1), 2212, e_tot)
    assert sum(o[1] for o in out) == pytest.approx(e_tot + pool.GUARD_MASS,
                                                   rel=1e-9)


def test_primary_sample_draw_order_is_what_the_replay_regenerates():
    """``results/m3d/scripts/reweight_replay.py`` regenerates each pending
    primary's ``T`` and nucleon identity from the c-file's seed with two
    vectorised draws (log-uniform energy, then ``rng.random(n) < Z/A``); that
    is the draw order of ``sky.primary_sample`` for a group model."""
    import sky
    axis = np.array([0.0, 0.0, 1.0])
    model = prim.primary_model("H3A")
    for gname in ("p", "He", "Fe"):
        _nm, a_mass, z, _gid = sky.GROUPS[gname]
        k, n = 2, 300
        e, isp, _w, _r, _u = sky.primary_sample(np.random.default_rng(77), model,
                                                gname, k, n, n, 40.0, axis)
        rng = np.random.default_rng(77)
        e2 = np.exp(rng.uniform(np.log(sky.STRATA[k]), np.log(sky.STRATA[k + 1]), n))
        isp2 = rng.random(n) < (z / a_mass)
        assert np.array_equal(e, e2) and np.array_equal(isp, isp2)


def test_hybrid_resim_option_keeps_signatures_and_selects_primaries():
    import sky3d
    a = sky3d.Options(primary_energy="kinetic")
    assert np.array_equal(a.signature(), sky3d.Options(primary_energy="kinetic",
                                                       resim_rmin=0.0).signature())
    h = sky3d.Options(r_thr=65.0, reanchor=True, resim_rmin=32.0,
                      resim_groups=("p",))
    s = h.signature()
    assert s[-3] == -1.0 and s[-2] == 32.0 and s[-1] == 0.0
    assert not np.array_equal(s[:9], sky3d.Options(r_thr=65.0).signature()[:9]) \
        or len(s) > 9
    assert len(sky3d.Options(r_thr=65.0, reanchor=True).signature()) < len(s)


def test_mc_tables_record_and_check_their_database():
    import os
    import warnings
    import interactions as I
    p = I.TABLES_V142P
    if not os.path.exists(p):
        pytest.skip("v142p tables not built")
    d = np.load(p)
    assert I.table_database(d)[1] and len(I.table_database(d)[1]) == 64
    old = np.load(I.TABLES_V140)
    assert I.table_database(old) == (None, None)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        I.check_database(old, I.TABLES_V140)          # v140 table, v142p selected
    assert any("built from database" in str(x.message) for x in w)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        I.check_database(d, p)
    assert not w
    # the 56 GeV hole is gone in the new table
    i = int(np.argmin(abs(d["e_grid"] - 56.2)))
    assert d["cs_2212"][i] > 0.95 * d["cs_2212"][i - 1]
    assert old["cs_2212"][i] < 0.85 * old["cs_2212"][i - 1]
