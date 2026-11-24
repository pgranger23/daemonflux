"""Tests of ``boundary.py`` (cascade3d C2, WP 2.1).

Run: ``taskset -c 0-3 $PY -m pytest tools/cascade3d/test_boundary.py -q``
"""

import os

import numpy as np
import pytest

import boundary as B

N3 = os.environ.get("N3", "/home/pigrange/mc3d_work/runs/native3d")
KS = os.path.join(N3, "kernels", "mc_close", "ks_mc_pool2000_L32_losson.npz")


def mock_verdict(r, u, R):
    """Deterministic stand-in for ``cutoff.trace_gated``: the dipole Stoermer
    cutoff of the state (``cutoff.stoermer_gv``)."""
    import cutoff

    rc = cutoff.stoermer_gv(np.atleast_2d(r), np.atleast_2d(u))
    return np.asarray(R) > rc, 0


# ------------------------------------------------------------------ geometry
def test_entry_geometry_ground_lines():
    """Entry radius, central angle and injection mu against the analytic line."""
    R, Ri = B.R_EARTH_CM, B.R_INJ_CM
    rh = B.pos_from_xy(3.0, -4.0)[0]
    cz = np.array([1.0, 0.75, 0.5, 0.25, 0.05])
    nodes = {"kind": np.zeros(5, int), "cosz": cz, "ht_km": np.zeros(5)}
    az = np.array([[0.0, 90.0, 200.0]] * 5)
    r, u, never = B.chain_entry_states(rh, nodes, az)
    assert not never.any()
    rad = np.linalg.norm(r, axis=-1)
    assert np.max(np.abs(rad / Ri - 1)) < 1e-12
    b = R * np.sqrt(1 - cz ** 2)
    mu_inj = -np.einsum("ijk,ijk->ij", u, r) / rad
    assert np.max(np.abs(mu_inj - np.sqrt(1 - (b / Ri) ** 2)[:, None])) < 1e-12
    ang = np.degrees(np.arccos(np.clip(np.einsum("ijk,k->ij", r, rh) / rad, -1, 1)))
    ang_exp = np.degrees(np.arctan2(-R * cz, b) - np.arctan2(-np.sqrt(Ri ** 2 - b ** 2),
                                                             b))
    assert np.max(np.abs(ang - ang_exp[:, None])) < 1e-6
    # the design's number: cos Z 0.05 enters 7.6 deg from P with mu 0.182
    assert abs(ang[-1, 0] - 7.615) < 1e-3 and abs(mu_inj[-1, 0] - 0.1819) < 1e-4
    # arrival azimuth: the horizontal part of -u points to the azimuth
    up, north, east = (v[0] for v in B.local_frame(rh[None]))
    a = np.degrees(np.arctan2(-u[2] @ east, -u[2] @ north)) % 360
    assert np.max(np.abs(a - az[2])) < 1e-9


def test_entry_geometry_tangent_lines():
    R, Ri = B.R_EARTH_CM, B.R_INJ_CM
    ht = np.array([0.0, 15.0, 99.0, 100.0, 110.0])
    nodes = {"kind": np.ones(5, int), "cosz": np.zeros(5), "ht_km": ht}
    r, u, never = B.chain_entry_states(B.pos_from_xy(0, 0)[0], nodes,
                                       np.full((5, 2), 45.0))
    assert never.tolist() == [[False] * 2] * 3 + [[True] * 2] * 2
    ok = ~never[:, 0]
    rad = np.linalg.norm(r[ok, 0], axis=-1)
    mu = -np.einsum("ij,ij->i", u[ok, 0], r[ok, 0]) / rad
    assert np.max(np.abs(mu - np.sqrt(1 - ((R + ht[ok] * 1e5) / Ri) ** 2))) < 1e-12


def test_chain_weights_and_never():
    import grid
    import rays

    atm = grid.CorsikaAtmosphere(r_e=B.R_EARTH_CM)
    lev = grid.make_levels(atm, spec=((50.0, 5.0), (70.0, 10.0), (np.inf, 15.0)))
    mu0 = np.array([1.0, 0.5, 0.33, 0.05, 0.02])
    fam = grid.LineFamily(mu0, lev[:-1], r_e=B.R_EARTH_CM)
    r = rays.Rays(atm, lev, fam, n_sub=1, dX_max=1e9)
    nodes = B.line_nodes()
    i0, i1, w0, w1, never = B.chain_weights(r, nodes)
    assert np.allclose(w0 + w1, 1.0)
    kind, cz, ht = nodes["kind"], nodes["cosz"], nodes["ht_km"]
    for j in range(r.Nl):
        if r.line_kind[j] == 0:
            z = np.degrees(np.arcsin(r.line_b[j] / r.r_e))
            zz = np.where(kind == 0, np.degrees(np.arccos(np.clip(cz, -1, 1))),
                          np.where(ht == 0, 90.0, np.nan))
            assert abs(w0[j] * zz[i0[j]] + w1[j] * zz[i1[j]] - z) < 1e-9
            assert not never[j]
        else:
            h = (r.line_b[j] - r.r_e) / 1e5
            assert never[j] == (h >= 100.0 - 1e-5)
            if 0 <= h <= 99:
                assert abs(w0[j] * ht[i0[j]] + w1[j] * ht[i1[j]] - h) < 1e-6
    # cos Z 0.5 is a node: weight one on it
    j = int(np.flatnonzero(np.isclose(r.ground_mu0, 0.5))[0])
    k = r.ground_line[j]
    sel = i0[k] if w0[k] > 0.5 else i1[k]
    assert np.isclose(cz[sel], 0.5) and max(w0[k], w1[k]) > 1 - 1e-12


# ------------------------------------------------------------- group weights
@pytest.fixture(scope="module")
def ks():
    if not os.path.exists(KS):
        pytest.skip("kernel set not available")
    import kernelset as K
    return K.KernelSet.load(KS)


def test_group_weights_reproduce_phi_top(ks):
    """Sum over groups and rungs = 1; the group sums give phi_top (bin 17 carries
    the interacting fraction f_prim of rule M3)."""
    gw = B.GroupWeights(ks.e_bins, "GSFG:GH")
    names = list(ks.species)
    b0 = ks.phi_top[[names.index("p+"), names.index("n0")]]
    tot = gw.C.sum(-1) + gw.above
    assert np.max(np.abs(tot[gw.has_flux] - 1)) < 1e-14
    assert np.array_equal(gw.has_flux, b0 > 0)
    turn = ks.meta["turn_on"]["species"]
    fp = np.array([turn["p+"]["f_prim"], turn["n0"]["f_prim"]])
    ratio = np.where(b0 > 0, gw.flux / np.where(b0 > 0, b0, 1), 1.0)
    k17 = turn["p+"]["k"]
    others = np.delete(ratio, k17, axis=1)
    assert np.max(np.abs(others - 1)) < 1e-6
    assert np.max(np.abs(ratio[:, k17] * fp - 1)) < 1e-6
    # neutrons come from A/Z ~ 2 groups only: none below 2 x 1.696 GV (rungs 0-4,
    # whose upper edges are below 3.39 GV)
    assert np.all(gw.edges[1:6] < 3.39) and np.all(gw.C[1, :, :5] == 0)
    assert np.any(gw.C[0, :, :5] > 0)


def test_unit_transmission_gives_zero_deficit(ks):
    gw = B.GroupWeights(ks.e_bins, "GSFG:GH")
    d = B.SkyTable.states((0.0, 0.0), seed=4)
    d["allowed"] = np.ones(d["az"].shape, bool)
    tab = B.SkyTable(d)
    names = list(ks.species)
    b0 = ks.phi_top[[names.index("p+"), names.index("n0")]]
    D = B.deficit_lines(tab, gw, b0)
    assert np.all(D == 0.0)
    d["allowed"][:] = False
    D = B.deficit_lines(B.SkyTable(d), gw, b0)
    # all forbidden below 60 GV: the deficit is B0 minus the part above 60 GV
    assert np.allclose(D[0, 0], (b0 * (1 - gw.above)).T, rtol=1e-12, atol=0)
    assert np.all(D <= b0.T[None, None])
    gwh = B.GroupWeights(ks.e_bins, "H3A")
    assert np.max(np.abs((gwh.C.sum(-1) + gwh.above)[gwh.has_flux] - 1)) < 1e-14


# ------------------------------------------------------------------- orders
def test_fourier_round_trip_and_sinc():
    rng = np.random.default_rng(1)
    V = rng.random((3, 36, 4))
    c, s = B.fourier_sectors(V, 18, axis=1)
    assert np.max(np.abs(B.synth_sectors(c, s, 36, axis=1) - V)) < 1e-12
    # a known function: sector means of cos(m phi), sin(m phi) give the orders back
    ed = np.linspace(0, 360, 37)
    Cb, Sb = B.bin_mean_basis(12, ed)
    for m in (0, 1, 5, 12):
        cc, ss = B.fourier_sectors(Cb[:, m][None], 12, axis=1)
        e = np.zeros(13)
        e[m] = 1
        assert np.max(np.abs(cc[0] - e)) < 1e-12 and np.max(np.abs(ss[0])) < 1e-12
        if m:
            cc, ss = B.fourier_sectors(Sb[:, m][None], 12, axis=1)
            assert np.max(np.abs(ss[0] - e)) < 1e-12 and np.max(np.abs(cc[0])) < 1e-12


def test_rotation_by_30_degrees_rotates_orders():
    """Sky rotated by 30 deg (three sectors): (c, s) of order m rotate by 30 m."""
    rng = np.random.default_rng(2)
    V = rng.random((5, 36))
    c, s = B.fourier_sectors(V, 12, axis=1)
    c2, s2 = B.fourier_sectors(np.roll(V, 3, axis=1), 12, axis=1)
    a = np.radians(30.0) * np.arange(13)
    assert np.max(np.abs(c2 - (c * np.cos(a) - s * np.sin(a)))) < 1e-12
    assert np.max(np.abs(s2 - (c * np.sin(a) + s * np.cos(a)))) < 1e-12
    # 30-deg bins: the bin means of the rotated function are the old ones moved by one
    Cb, Sb = B.bin_mean_basis(12, np.linspace(0, 360, 13))
    old = c @ Cb.T + s @ Sb.T
    new = c2 @ Cb.T + s2 @ Sb.T
    assert np.max(np.abs(new - np.roll(old, 1, axis=1))) < 1e-12


# ------------------------------------------------------------- location grid
def test_location_grid():
    g = B.LocationGrid(5.0, 10.0)
    assert g.n == 25 and g.node_index(0, 0) == 12
    assert np.allclose(g.positions()[12], B.site_frame()[0])
    x, y = B.xy_of(g.positions())
    assert np.allclose(x, g.x) and np.allclose(y, g.y)
    idx, w, inside = g.weights_xy(np.array([1.0, -7.5, 10.0]), np.array([2.0, 3.0,
                                                                          -10.0]))
    assert inside.all() and np.allclose(w.sum(1), 1)
    pts = (g.x[idx] * w).sum(1), (g.y[idx] * w).sum(1)
    assert np.allclose(pts[0], [1.0, -7.5, 10.0]) and np.allclose(pts[1], [2, 3, -10])
    W = g.cap_weights(10.0, n_quad=60)
    assert abs(W.sum() - 1) < 1e-12 and W[12] == W.max()
    Wg = W.reshape(5, 5)
    assert np.allclose(Wg, Wg[::-1]) and np.allclose(Wg, Wg[:, ::-1])
    g2 = B.LocationGrid(2.5, 10.0)
    assert g2.n == 81 and np.allclose(g2.cap_weights(10, 60).sum(), 1)


# ------------------------------------------------------- table build and G2.0
def test_build_resume_and_no_overwrite(tmp_path):
    p = str(tmp_path / "t.npz")
    t = B.SkyTable.build(p, (5.0, 0.0), seed=7, min_chunk=8000,
                         verdict="test_boundary.mock_verdict", log=lambda s: None)
    assert np.array_equal(t.allowed, (t.R > _stoermer(t)) & ~t.never_state)
    with pytest.raises(FileExistsError):
        B.SkyTable.build(p, (5.0, 0.0), seed=7, verdict="test_boundary.mock_verdict")
    os.remove(p)
    msgs = []
    t2 = B.SkyTable.build(p, (5.0, 0.0), seed=7, min_chunk=8000,
                          verdict="test_boundary.mock_verdict", log=msgs.append)
    assert all("resumed" in m for m in msgs) and len(msgs) >= 2
    assert np.array_equal(t2.allowed, t.allowed)


def _stoermer(t):
    import cutoff

    r = t.r_inj.reshape(-1, 3)
    u = t.u.reshape(-1, 3)
    out = np.full(r.shape[0], np.inf)
    ok = ~t.never_state.ravel()
    out[ok] = cutoff.stoermer_gv(r[ok], u[ok])
    return out.reshape(t.R.shape)


@pytest.fixture(scope="module")
def mock_tables(tmp_path_factory):
    d = tmp_path_factory.mktemp("sky")
    g = B.LocationGrid(5.0, 10.0)
    tabs = {}
    for i in range(g.n):
        p = str(d / f"t{i}.npz")
        tabs[i] = B.SkyTable.build(p, (g.x[i], g.y[i]), seed=100 + i,
                                   verdict="test_boundary.mock_verdict",
                                   log=lambda s: None)
    return g, tabs


def test_g20_harness_exact_states(mock_tables):
    """A state of a table (location node, line node, its azimuth and rigidity) gets
    exactly that table's verdict."""
    g, tabs = mock_tables
    t = tabs[g.node_index(5.0, -5.0)]
    rng = np.random.default_rng(5)
    nn, ns, nr = t.allowed.shape
    ii = rng.integers(0, nn, 60)
    jj = rng.integers(0, ns, 60)
    kk = rng.integers(0, nr - 1, 60)
    keep = ~t.never_state[ii, jj, kk]
    ii, jj, kk = ii[keep], jj[keep], kk[keep]
    T, cov, hit = B.sky_transmission(t.r_inj[ii, jj, kk], t.u[ii, jj, kk],
                                     t.R[ii, jj, kk], tabs, g)
    assert cov.all()
    assert np.max(np.abs(T - t.allowed[ii, jj, kk])) < 1e-12


def test_g20_harness_200_states(mock_tables):
    """200 covered states of the reference draw: sky transmission against the
    direct verdict of the same (mock) cutoff.  The class means agree; most states
    agree individually (they differ only near the cutoff edge)."""
    import cutoff

    g, tabs = mock_tables
    st = cutoff.sample_test_states(11, 6000)
    sel = st["R"] < 60.0
    r, u, R = st["r_cm"][sel], st["u"][sel], st["R"][sel]
    T, cov, hit = B.sky_transmission(r, u, R, tabs, g)
    r, u, R, T, hit = r[cov][:200], u[cov][:200], R[cov][:200], T[cov][:200], hit[
        cov][:200]
    assert r.shape[0] == 200 and (~hit).sum() > 0
    direct = mock_verdict(r, u, R)[0].astype(float)
    assert abs(T.mean() - direct.mean()) < 0.03
    assert np.mean(np.abs(T - direct) < 0.5) > 0.9


# --------------------------------------------- location grid, spacing per axis
WREF = os.path.join(N3, "c2", "solver", "grid_ref", "weights_before_d7ede51.npz")


def test_isotropic_grid_bit_identical_to_before():
    """With one spacing the class gives the nodes, point weights and cap weights it
    gave before it took one spacing per axis (stored from the code of d7ede51)."""
    if not os.path.exists(WREF):
        pytest.skip("reference weights not available")
    d = np.load(WREF)
    for s in (5.0, 2.5):
        k = str(s).replace(".", "p")
        g = B.LocationGrid(s, 10.0)
        idx, w, ins = g.weights_xy(d["x"], d["y"])
        assert np.array_equal(idx, d["idx_" + k]) and np.array_equal(w, d["w_" + k])
        assert np.array_equal(ins, d["in_" + k])
        assert np.array_equal(g.x, d["nx_" + k]) and np.array_equal(g.y, d["ny_" + k])
        for c in (10, 5):
            assert np.array_equal(g.cap_weights(float(c)), d[f"cap{c}_" + k])
        assert np.array_equal(g.cap_weights(10.0, n_quad=60), d["cap10q60_" + k])
        assert np.array_equal(B.LocationGrid((s, s), (10.0, 10.0)).cap_weights(10.0),
                              d["cap10_" + k])


def test_anisotropic_grid_weights():
    """Grid A (5 x 2.5) and grid B (5 x 1.25): node counts, partition of unity,
    exactness for functions bilinear in x and y, node_index, A inside B."""
    ga, gb = B.LocationGrid((5.0, 2.5)), B.LocationGrid((5.0, 1.25))
    assert ga.n == 45 and gb.n == 85 and gb.label() == "5x1.25"
    rng = np.random.default_rng(3)
    x, y = rng.uniform(-10, 10, 500), rng.uniform(-10, 10, 500)
    x[:4], y[:4] = [-10, 10, 0, 5], [-10, 10, 1.25, -8.75]
    for g in (ga, gb):
        idx, w, ins = g.weights_xy(x, y)
        assert ins.all() and np.allclose(w.sum(1), 1, atol=1e-14, rtol=0)
        for f in (lambda u, v: 1 + 0 * u, lambda u, v: u, lambda u, v: v,
                  lambda u, v: 3 - 2 * u + 0.5 * v + 0.1 * u * v):
            assert np.max(np.abs((f(g.x[idx], g.y[idx]) * w).sum(1) - f(x, y))) < 1e-12
        for p in range(g.n):
            assert g.node_index(g.x[p], g.y[p]) == p
        _i, _w, out = g.weights_xy(np.array([10.5, 0]), np.array([0, -10.2]))
        assert not out.any()
    for p in range(ga.n):
        q = gb.node_index(ga.x[p], ga.y[p])
        assert ga.x[p] == gb.x[q] and ga.y[p] == gb.y[q]
    assert np.allclose(ga.positions(), gb.positions()[[gb.node_index(a, b) for a, b in
                                                        zip(ga.x, ga.y)]])


@pytest.mark.parametrize("cap", [10.0, 5.0])
def test_anisotropic_cap_weights(cap):
    """Cap weights of grids A and B: sum to one, mirror symmetric in x and in y,
    centred, zero outside the cap and its border cells; the cap mean of a function
    linear in x and y is exact on both grids."""
    for g in (B.LocationGrid((5.0, 2.5)), B.LocationGrid((5.0, 1.25))):
        W = g.cap_weights(cap)
        assert abs(W.sum() - 1) < 1e-12 and np.all(W >= 0)
        Wg = W.reshape(g.axis_x.size, g.axis_y.size)
        assert np.allclose(Wg, Wg[::-1], rtol=1e-12, atol=1e-15)
        assert np.allclose(Wg, Wg[:, ::-1], rtol=1e-12, atol=1e-15)
        assert abs(W @ g.x) < 1e-12 and abs(W @ g.y) < 1e-12
        far = (np.abs(g.x) > cap + g.spacing_xy[0]) | (np.abs(g.y) > cap +
                                                        g.spacing_xy[1])
        assert np.all(W[far] == 0)
        assert abs(W @ (2 + g.x - 3 * g.y) - 2) < 1e-12


def test_grid_b_names_and_xy_check(tmp_path):
    """table_path stays as it was (one decimal): the names of grid B are unique and
    those of the 2.5-deg grid unchanged; check_table refuses a table whose stored
    coordinates are not the node's."""
    gb, g25 = B.LocationGrid((5.0, 1.25)), B.LocationGrid(2.5)
    names = [B.table_path("d", (x, y)) for x, y in zip(gb.x, gb.y)]
    assert len(set(names)) == 85
    assert B.table_path("d", (0.0, 1.25)).endswith("sky_h0_x+00.0_y+01.2.npz")
    assert B.table_path("d", (0.0, 3.75)).endswith("sky_h0_x+00.0_y+03.8.npz")
    assert B.table_path("d", (-7.5, 2.5)).endswith("sky_h0_x-07.5_y+02.5.npz")
    n25 = {B.table_path("d", (x, y)) for x, y in zip(g25.x, g25.y)}
    shared = [B.table_path("d", (x, y)) for x, y in zip(gb.x, gb.y)
              if abs(y / 2.5 - round(y / 2.5)) < 1e-12]
    assert len(shared) == 45 and set(shared) <= n25
    d = B.SkyTable.states((0.0, 1.25), seed=1)
    t = B.SkyTable(d)
    gb.check_table(t, gb.node_index(0.0, 1.25))
    with pytest.raises(ValueError):
        gb.check_table(t, gb.node_index(0.0, 2.5))
    assert B.parse_spacing("5") == 5.0 and B.parse_spacing("5,1.25") == (5.0, 1.25)
