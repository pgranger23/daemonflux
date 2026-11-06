"""Milestone 3c: the muon-bend decomposition and diagnostics, the common
random numbers of the paired re-anchor vs re-simulate test."""

import os
import pickle

import numpy as np
import pytest

import bend
import cutoff
import geometry as geo
from constants import R_EARTH_CM
from test_m3b import LAT, LON, _ctx, _local_states, grid  # noqa: F401


def _bent_pairs(n=400, seed=21):
    rng = np.random.default_rng(seed)
    r, u0 = _local_states(rng, n)
    ax = rng.normal(size=(n, 3))
    ax /= np.linalg.norm(ax, axis=1)[:, None]
    ang = rng.uniform(0.0, 0.1, n)
    u1 = np.array([cutoff.rotate_about(u0[i:i + 1], ax[i], ang[i])[0]
                   for i in range(n)])
    return r, u0, u1


def test_split_bend_keeps_one_component_each():
    import sky3d
    r, u0, u1 = _bent_pairs()
    bv, bh = sky3d.bend_components(u0, u1, r)
    uv = sky3d.split_bend(u0, u1, r, "vertical")
    uh = sky3d.split_bend(u0, u1, r, "horizontal")
    assert np.allclose(np.linalg.norm(uv, axis=1), 1.0, atol=1e-12)
    assert np.allclose(np.linalg.norm(uh, axis=1), 1.0, atol=1e-12)
    bvv, bhv = sky3d.bend_components(u0, uv, r)
    bvh, bhh = sky3d.bend_components(u0, uh, r)
    assert np.allclose(bvv, bv, atol=1e-12) and np.allclose(bhv, 0.0, atol=1e-9)
    assert np.allclose(bvh, 0.0, atol=1e-12) and np.allclose(bhh, bh, atol=1e-9)
    assert np.array_equal(sky3d.split_bend(u0, u1, r, "full"), u1)
    assert np.abs(bv).max() > 0.02 and np.abs(bh).max() > 1.0


def test_bend_components_sign_in_a_northward_field():
    """At the site, B = 30 uT North: a mu+ moving East (arriving from the
    West) turns up (``q v x B`` = up), so its arrival cos Z falls (``bv < 0``);
    a mu- does the opposite; a vertical field turns arrivals in azimuth."""
    import sky3d
    up, no, ea = geo.local_frame(LAT, LON)
    up, no, ea = (np.asarray(x, float) for x in (up, no, ea))
    m = 20
    r = np.tile((R_EARTH_CM + 15e5) * up, (m, 1))
    u = np.tile(ea, (m, 1))
    tt = np.full(m, 0.5)
    e = np.full(m, 2.0)
    for bvec, check in ((3e-5 * no, "v"), (-4e-5 * up, "h")):
        def fld(x, b=bvec):
            return np.tile(b, (len(x), 1))
        for pdg, sgn in ((-13, 1.0), (13, -1.0)):
            k, rr, uu, _e = bend.transport_muons(np.full(m, pdg), e, r, u, tt,
                                                 fld, energy_loss=False)
            assert (k == bend.MU_DECAY).all()
            bv, bh = sky3d.bend_components(u, uu, rr)
            if check == "v":
                assert (sgn * bv < 0).all() and np.abs(bh).max() < 0.05
            else:
                # B down, mu+ moving East: v x B = E x (-up) = +North, so the
                # arrival direction (West) swings towards South: az decreases
                assert (sgn * bh < 0).all() and np.abs(bv).max() < 2e-3


def test_crn_resimulated_copy_starts_where_the_reanchored_copy_does(
        tmp_path, grid):  # noqa: F811
    """With ``crn``, re-simulating a primary's copies (``r_thr`` above its
    rigidity) and re-anchoring the base cascade (``r_thr = 0``, ``reanchor``)
    are paired copy by copy: the re-simulated copy 0 *is* the base cascade
    (same stream, same first interaction, same muon decays -> identical 3D
    entries), and every re-simulated copy ``j`` starts from exactly the
    state the re-anchoring moves the base cascade to (same sampled depth,
    the copy's own bent primary)."""
    import sky3d
    task = ("p", 2, 100, 40.0, 77, 1000)
    out = {}
    for tag, thr in (("anc", 0.0), ("res", 1e9)):
        opts = sky3d.Options(cutoff=False, bend_primary=True, muon_field=True,
                             r_thr=thr, reanchor=True, crn=True,
                             keep_reals=True)
        _ctx(sky3d, opts, grid)
        os.makedirs(sky3d.task_dir(str(tmp_path), tag))
        path = sky3d.cfile(str(tmp_path), tag, 0)
        sky3d.cascade_task(task + (0, path))
        with open(path, "rb") as fh:
            out[tag] = pickle.load(fh)
    anc, res = out["anc"], out["res"]
    # copy 0: identical 3D entries (keyed by the primary's index in the task)
    def copy0(d):
        e3, ids = d["ent"]["3d"], np.array([p["i"] for p in d["prims"]])
        m = e3["copy"] == 0
        key = np.stack([ids[e3["prim"][m]], e3["flat"][m],
                        np.round(e3["e"][m] * 1e9)], 1)
        return key[np.lexsort(key.T[::-1])]
    a0, r0 = copy0(anc), copy0(res)
    assert len(a0) > 20
    assert np.array_equal(a0, r0)
    # copies j >= 1: same first-interaction state as the re-anchoring's
    tab = sky3d._reanchor_table(anc["prims"], np.array(
        cutoff.site_axis(LAT, LON)), 12, 30.0 * np.arange(12), grid)
    pa = {p["i"]: (q, p) for q, p in enumerate(anc["prims"])}
    n_cmp = 0
    for p in res["prims"]:
        if p["i"] not in pa:
            continue
        q, pa_ = pa[p["i"]]
        rb = [ri for ri, r_ in enumerate(pa_["reals"]) if r_["kind"] == "B"]
        dx0 = pa_["reals"][rb[0]]["int"][2]
        for r_ in p["reals"]:
            if r_["kind"] != "C" or "int" not in r_:
                continue
            assert r_["int"][2] == dx0                     # one depth
            if r_["j"] == 0:
                continue
            tr = tab.get((q, rb[0], int(r_["j"])))
            if tr is None or tr[0] is None:
                continue
            # the two paths start from copy_states vs _rot (1e-16 apart) and
            # are stepped independently: they end metres apart on 30-1000 km
            # (bend_primary's own tolerance against advance_grammage: 20 m)
            assert np.linalg.norm(tr[1] - r_["int"][0]) < 2.0e3    # cm
            n_cmp += 1
    assert n_cmp > 20


def test_crn_off_keeps_the_milestone_3b_stream_and_mub_diagnostics(tmp_path,
                                                                  grid):  # noqa: F811
    """``crn`` off (default), ``mu_bend="full"`` and the kinetic primary
    energy leave the 3b manifest signature; the muon-bend diagnostic
    histograms are filled and add up."""
    import sky3d
    o = sky3d.Options(cutoff=False, bend_primary=True, muon_field=True,
                      r_thr=0.0, primary_energy="kinetic")
    assert len(o.signature()) == 9
    assert len(sky3d.Options(crn=True, primary_energy="kinetic")
               .signature()) == 11
    _ctx(sky3d, o, grid)
    os.makedirs(sky3d.task_dir(str(tmp_path), "mb"))
    sky3d.cascade_task(("p", 2, 60, 40.0, 5, 600, 0,
                        sky3d.cfile(str(tmp_path), "mb", 0)))
    sky3d.trace_replay_task(([0], str(tmp_path), "mb"))
    d = dict(np.load(sky3d.tfile(str(tmp_path), "mb", 0)))
    m, h = d["mub_m"], d["mub_h"]
    assert m[..., 0].sum() > 0
    assert np.allclose(h[..., 0, :].sum(-1), m[..., 0])
    assert np.allclose(h[..., 1, :].sum(-1), m[..., 0])
    # the muon-born weight is a part of the 3D tally (largest cap, which holds
    # every entry with 1/A) in those bins, and a large part of it near the
    # horizon at these energies
    s0 = d["s0"][-1] * d["cap_area"][-1]
    frac = []
    for b, (i0, i1) in enumerate(sky3d.MUB_EBANDS):
        tot = s0[:, i0:i1, sky3d.MUB_CZ[0]:sky3d.MUB_CZ[1]].sum(1)
        assert (m[:, b, ..., 0] <= tot * (1 + 1e-9) + 1e-300).all()
        frac.append(m[:, b, ..., 0].sum() / tot.sum())
    assert 0.2 < max(frac) <= 1.0


def test_mu_bend_none_runs_the_field_pipeline_without_bending(tmp_path,
                                                             grid):  # noqa: F811
    """``mu_bend="none"``: the muons are re-transported and decayed per copy
    as with the field on (``mub`` filled), but with a zero field: every
    recorded bend is exactly zero."""
    import sky3d
    o = sky3d.Options(cutoff=False, bend_primary=False, muon_field=True,
                      r_thr=0.0, mu_bend="none", primary_energy="kinetic")
    assert len(o.signature()) == 11
    _ctx(sky3d, o, grid)
    os.makedirs(sky3d.task_dir(str(tmp_path), "mn"))
    sky3d.cascade_task(("p", 2, 40, 40.0, 6, 400, 0,
                        sky3d.cfile(str(tmp_path), "mn", 0)))
    sky3d.trace_replay_task(([0], str(tmp_path), "mn"))
    d = dict(np.load(sky3d.tfile(str(tmp_path), "mn", 0)))
    m = d["mub_m"]
    assert m[..., 0].sum() > 0
    assert np.abs(m[..., 1]).max() < 1e-12 * m[..., 0].max()
    assert np.abs(m[..., 3]).max() < 1e-9 * m[..., 0].max()


def test_primary_energy_convention_total_is_mceqs():
    """``primaries.ENERGY_CONVENTION = "total"`` evaluates crflux at the
    nucleus' total energy ``A (T + m)``: summed over the five groups it is
    exactly crflux's ``p_and_n_flux(T + m)`` -- what ``MCEqRun`` puts into
    ``_phi0`` on its kinetic grid -- while ``"kinetic"`` (milestones 1-3b)
    is ``x 2.06`` of it at 3 GeV/n.  The sampled weights follow the module
    setting, and the run signature records a non-default one."""
    import primaries as prim
    import sky
    import sky3d
    model = prim.primary_model("H3A")
    T = np.array([2.0, 3.0, 10.0, 100.0])
    # (crflux's own H3a groups: C-12, Si-28, Fe-54; primaries.SPECIES uses
    # N-14, Mg-25, Fe-56 with the same group spectra -- a ~1% difference)
    tot = 0.0
    for cid in model.nucleus_ids:
        z_, a_ = model.Z_A(cid)
        tot = tot + prim.nucleon_intensity(model, cid, T, a_, "total") * z_ / a_
    _f, p_ref, _n = model.p_and_n_flux(T + prim.M_NUCLEON)
    assert np.allclose(tot, p_ref * 1e-4, rtol=1e-12)
    kin = prim.nucleon_intensity(model, 14, T, 1, "kinetic")
    assert np.allclose(kin, model.nucleus_flux(14, T) * 1e-4, rtol=1e-12)
    r = kin / prim.nucleon_intensity(model, 14, T, 1, "total")
    assert r[1] == pytest.approx(2.06, abs=0.01) and r[-1] < 1.03
    # the drivers' weights follow the module setting
    axis = np.array(cutoff.site_axis(LAT, LON))
    old = prim.ENERGY_CONVENTION
    try:
        w = {}
        for conv in ("kinetic", "total"):
            prim.ENERGY_CONVENTION = conv
            e, _isp, w[conv], _r, _u = sky.primary_sample(
                np.random.default_rng(3), model, "He", 1, 50, 50, 40.0, axis)
        want = (model.nucleus_flux(402, 4 * (e + prim.M_NUCLEON))
                / model.nucleus_flux(402, 4 * e))
        assert np.allclose(w["total"] / w["kinetic"], want, rtol=1e-12)
    finally:
        prim.ENERGY_CONVENTION = old
    assert len(sky3d.Options(primary_energy="kinetic").signature()) == 9
    assert len(sky3d.Options(primary_energy="total").signature()) == 12
    # milestone 3d: the default is "native"; for H3a (the default primary
    # model) that resolves to "total" -- the same signature as milestone 3c's
    # default runs, so their manifests stay valid
    assert prim.ENERGY_CONVENTION == "native"
    assert len(sky3d.Options().signature()) == 12
    assert np.array_equal(sky3d.Options().signature(),
                          sky3d.Options(primary_energy="total").signature())


def test_mu_cf_counterfactual_is_exact_without_bend(tmp_path,
                                                    grid):  # noqa: F811
    """``mu_cf``: the zero-field counterfactual of every re-transported muon
    uses the same proper time and decay numbers, so with ``mu_bend="none"``
    it equals the field-on entries bin by bin (every decay in fate class 0);
    with the field on, the field-on muon part of ``mucf`` plus the non-muon
    part of ``s`` is ``s`` itself, and the physics tallies do not depend on
    ``mu_cf``."""
    import sky3d
    out = {}
    for tag, mode, cf in (("cn", "none", True), ("cf", "full", True),
                          ("c0", "full", False)):
        o = sky3d.Options(cutoff=False, bend_primary=False, muon_field=True,
                          r_thr=0.0, mu_bend=mode, mu_cf=cf)
        _ctx(sky3d, o, grid)
        os.makedirs(sky3d.task_dir(str(tmp_path), tag))
        sky3d.cascade_task(("p", 2, 60, 40.0, 8, 600, 0,
                            sky3d.cfile(str(tmp_path), tag, 0)))
        sky3d.trace_replay_task(([0], str(tmp_path), tag))
        out[tag] = dict(np.load(sky3d.tfile(str(tmp_path), tag, 0)))
    m = out["cn"]["mucf"]
    assert m[0].sum() > 0
    assert np.allclose(m[0], m[1], rtol=1e-12, atol=0)
    assert m[:, 1:].sum() == 0
    mf = out["cn"]["mucf_full"]                  # milestone 4: the full-grid version
    assert np.allclose(mf[0], mf[1], rtol=1e-12, atol=0)
    assert mf[0].sum() >= m[0].sum() * (1 - 1e-12) > 0
    for key in ("s1", "s0", "n1", "t1", "mub_m"):
        assert np.array_equal(out["cf"][key], out["c0"][key]), key
    assert out["c0"]["mucf"].sum() == pytest.approx(
        out["c0"]["mucf"][0, 0].sum(), rel=1e-12)
