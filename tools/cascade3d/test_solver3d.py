"""Tests of ``solver3d.py`` (cascade3d C2, WP 2.2).

Run: ``taskset -c 0-3 $PY -m pytest tools/cascade3d/test_solver3d.py -q``

The reduced geometry below (26 levels, 5 ground and 13 tangent lines, 611 nodes) with
the kernel set of G2.3 cut at 6 GeV marches every piece of the C1 step (angular
channels, loss species, same-bin sweeps, up-scatter passes) in about 20 s per solve.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import grid
import kernelset as K
import rays
import solver as S
import solver3d as S3

N3 = os.environ.get("N3", "/home/pigrange/mc3d_work/runs/native3d")
KS = os.path.join(N3, "kernels", "mc_close", "ks_mc_pool2000_L32_losson.npz")


# ------------------------------------------------------------- mode weights
def test_assoc_legendre_against_scipy():
    from scipy.special import factorial, lpmv

    x = np.linspace(-1, 1, 9)
    for m in (0, 1, 2, 5, 12, 18):
        P = S3.assoc_legendre(x, 32, m)
        assert np.all(P[:m] == 0)
        for ll in range(m, 33):
            ref = lpmv(m, ll, x) * (-1) ** m * np.sqrt(
                (2 * ll + 1) / 2 * factorial(ll - m, exact=True)
                / factorial(ll + m, exact=True))
            assert np.max(np.abs(P[ll] - ref)) < 1e-12


@pytest.fixture(scope="module")
def coarse_rays():
    atm = grid.CorsikaAtmosphere(r_e=6371e5)
    lev = grid.make_levels(atm, spec=((50.0, 5.0), (70.0, 10.0), (np.inf, 20.0)))
    fam = grid.make_line_family(lev, n_ground=12, r_e=6371e5)
    return rays.Rays(atm, lev, fam, n_sub=1, dX_max=1e9)


def test_mode_weights_m0_is_c1(coarse_rays):
    """Order 0: C1's hat projection and synthesis (to 1e-13)."""
    r, L = coarse_rays, 32
    W, Q = S3.mode_weights(r, L, 0)
    fac = np.sqrt((2 * np.arange(L + 1) + 1) / 2)
    W0 = S._hat_weights(r, L)
    assert np.max(np.abs(W - W0 * fac)) < 1e-13 * np.abs(W0).max()
    mu = np.concatenate([r.node_mu, [0.0]])[r.lev_node]
    P = np.moveaxis(K.legendre_table(mu, L), 0, -1)
    assert np.max(np.abs(Q - P * fac)) < 1e-12


def test_mode_weights_odd_m_quadrature(coarse_rays):
    """Odd m (theta quadrature) against adaptive quadrature, and converged in the
    number of theta points."""
    from scipy.integrate import quad

    r, L = coarse_rays, 32
    e = 20
    mu_all = np.concatenate([r.node_mu, [0.0]])
    valid = r.lev_node[e] < r.N
    mus = mu_all[r.lev_node[e][valid]]
    n = mus.size
    for m in (1, 3, 7):
        W, _ = S3.mode_weights(r, L, m)
        W2, _ = S3.mode_weights(r, L, m, n_theta=3 * L)
        assert np.max(np.abs(W - W2)) < 1e-13 * np.abs(W).max()
        for slot in (0, n // 2, n - 1):
            for ll in (m, m + 4, L):
                def f(t, slot=slot, ll=ll, m=m):
                    h = np.interp(t, mus, np.eye(n)[slot])
                    return h * S3.assoc_legendre(np.array(t), L, m)[ll]
                v, _ = quad(f, -1, 1, points=list(mus), limit=400, epsabs=1e-14,
                            epsrel=1e-12)
                assert abs(v - W[e, slot, ll]) < 1e-12


@pytest.mark.parametrize("m", [0, 1, 2, 5])
def test_analytic_ylm_field(m):
    """A field Pbar_l^m(mu) cos(m phi) is an eigenfunction of every kernel: the
    order-m source synth(a . proj(f)) must be a_l f at the nodes, to the accuracy of
    the hat interpolation, converging as the line family is refined."""
    atm = grid.CorsikaAtmosphere(r_e=6371e5)
    lev = grid.make_levels(atm, spec=((50.0, 5.0), (70.0, 10.0), (np.inf, 20.0)))
    L = 16
    a = 0.9 ** np.arange(L + 1)
    errs = []
    for ng in (12, 48):
        fam = grid.make_line_family(lev, n_ground=ng, r_e=6371e5)
        r = rays.Rays(atm, lev, fam, n_sub=1, dX_max=1e9)
        W, Q = S3.mode_weights(r, L, m)
        mu = np.concatenate([r.node_mu, [0.0]])
        e = int(np.argmin(np.abs(r.levels - 15e5)))
        valid = r.lev_node[e] < r.N
        nodes = r.lev_node[e][valid]
        worst = 0.0
        for ll in (m, m + 2, m + 5):
            f = S3.assoc_legendre(mu[nodes], L, m)[ll]
            mom = f @ W[e, valid]
            src = (a * mom) @ Q[e, valid].T
            worst = max(worst, np.max(np.abs(src - a[ll] * f)))
        errs.append(worst)
    assert errs[1] < 0.5 * errs[0] and errs[1] < 2e-2, errs


# ----------------------------------------------------------- guard functions
def test_guarded_functions_rules():
    """Where the ratios are above rho the guarded derivative is the exact one; below
    it, the derivative of the linear fallback form.  Primal values unchanged."""
    gf = S3.guarded_functions(S3.Guard(0.1, 0.1, 0.1, 0.1))
    rng = np.random.default_rng(0)
    # source shape
    Aa, Ab, Ac = jnp.asarray([1.0, 1.0]), jnp.asarray([0.5, 1e-3]), jnp.asarray(
        [0.3, 1.0])
    uc, u, hc = jnp.asarray([2.0, 2.0]), jnp.asarray([0.4, 0.4]), jnp.asarray(
        [True, True])
    t = tuple(jnp.asarray(rng.random(2)) for _ in range(3))
    zero = jnp.zeros(2)
    f0 = (lambda a, b, c: S._logquad(a, b, c, uc, u, hc))  # noqa: E731
    f1 = (lambda a, b, c: gf["logquad"](a, b, c, uc, u, hc))  # noqa: E731
    y0, d0 = jax.jvp(f0, (Aa, Ab, Ac), t)
    y1, d1 = jax.jvp(f1, (Aa, Ab, Ac), t)
    assert np.array_equal(y0, y1)
    assert abs(d1[0] - d0[0]) < 1e-14
    assert abs(d1[1] - (t[0][1] + u[1] * (t[1][1] - t[0][1]))) < 1e-14
    del zero
    # source integral
    args = [jnp.asarray(v) for v in ([1.0, 1.0], [0.5, 1e-4], [1.0, 1.0], [1.0, 1.0],
                                     [0.3, 0.3], [2.0, 2.0])]
    ta = [jnp.asarray(rng.random(2)) for _ in range(6)]
    y0, d0 = jax.jvp(S._expj, tuple(args), tuple(ta))
    y1, d1 = jax.jvp(gf["expj"], tuple(args), tuple(ta))
    def lin_e(S0, S1, r0, r1, x, ds):
        p0, p1 = S._phi01(x)
        return ds * (S0 * r0 * p0 + S1 * r1 * p1)

    lin = jax.jvp(lin_e, tuple(args), tuple(ta))[1][1]
    assert np.array_equal(y0, y1) and abs(d1[0] - d0[0]) < 1e-14
    assert abs(d1[1] - lin) < 1e-12
    # loss interpolation
    vals = jnp.asarray([[1.0, 1.2, 1.5, 1.9], [1.0, 1e-2, 1e-3, 1e-4]])
    wl = jnp.asarray([[-0.1, 0.8, 0.4, -0.1]] * 2)
    wpre = jnp.asarray([[-0.12, 0.85, 0.37, -0.09]] * 2)
    dv = jnp.asarray(rng.random((2, 4)))
    y0, d0 = jax.jvp(S._interp_log, (vals, wl, wpre), (dv, 0 * wl, 0 * wpre))
    y1, d1 = jax.jvp(gf["interp_log"], (vals, wl, wpre), (dv, 0 * wl, 0 * wpre))
    assert np.array_equal(y0, y1) and abs(d1[0] - d0[0]) < 1e-14
    assert abs(d1[1] - jnp.sum(wpre[1] * dv[1])) < 1e-14
    # Simpson means
    a, b = jnp.asarray([1.0, 1.0]), jnp.asarray([0.5, 1e-3])
    da, db = jnp.asarray([0.3, 0.3]), jnp.asarray([0.7, 0.7])
    for name, ref in (("gmean", S3._c1_gmean), ("logmean", S3._c1_logmean)):
        y0, d0 = jax.jvp(ref, (a, b), (da, db))
        y1, d1 = jax.jvp(gf[name], (a, b), (da, db))
        assert np.array_equal(y0, y1) and abs(d1[0] - d0[0]) < 1e-14
        assert abs(d1[1] - 0.5 * (da[1] + db[1])) < 1e-14


# ---------------------------------------------------------------- solver path
@pytest.fixture(scope="module")
def small():
    if not os.path.exists(KS):
        pytest.skip("kernel set not available")
    atm = grid.CorsikaAtmosphere(r_e=6371e5)
    lev = grid.make_levels(atm, spec=((50.0, 5.0), (70.0, 10.0), (np.inf, 20.0)))
    fam = grid.LineFamily(np.array([1.0, 0.6, 0.3, 0.12, 0.05]), lev[:-1][::2],
                          r_e=6371e5)
    r = rays.Rays(atm, lev, fam, n_sub=2, dX_max=50.0)
    ks = K.KernelSet.load(KS)
    d = dict(ks.d)
    d["phi_top"] = np.where(ks.e_grid[None] <= 6.0, ks.phi_top, 0.0)
    ks = K.KernelSet(d)
    sv = S.Solver(r, ks)
    out = sv.solve()
    # isotropic step deficit (G2.7 deficit 1)
    T = np.ones_like(ks.phi_top)
    p = np.sqrt(ks.e_grid * (ks.e_grid + 2 * 0.938272))
    names = list(ks.species)
    for sp, az in (("p+", 1.0), ("n0", 2.0)):
        T[names.index(sp)] = 1 / (1 + np.exp(-np.log(az * p / 11.5) / 0.15))
    D = ks.phi_top * (1 - T)
    return {"r": r, "ks": ks, "sv": sv, "out": out, "D": D}


def test_primal_inside_equals_solver(small):
    """ModeSolver.solve and the primal of TangentSolver equal Solver.solve (1e-12;
    measured: bit for bit)."""
    r, ks, out = small["r"], small["ks"], small["out"]
    ms = S3.ModeSolver(r, ks)
    o2 = ms.solve()
    ts = S3.TangentSolver(r, ks)
    o3 = ts.solve(np.zeros((1,) + ks.phi_top.shape + (r.Nl,)), [3])
    for key in ("emis", "F_ground"):
        a = np.asarray(out[key])
        for o in (o2, o3):
            assert np.max(np.abs(np.asarray(o[key]) - a)) <= 1e-12 * np.abs(a).max()
    # zero boundary tangent: exactly zero (transmission 1 returns C1 bit for bit)
    assert not np.any(np.asarray(o3["d_emis"]))
    assert not np.any(np.asarray(o3["d_F_ground"]))


def test_unguarded_tangent_equals_jvp(small):
    """Order 0 without guard: jax.jvp of Solver.solve (to 1e-10; measured: bit for
    bit), through the up-scatter passes; the tangent sweep residual is reported."""
    r, ks, sv, D = small["r"], small["ks"], small["sv"], small["D"]
    prm = {k: jnp.asarray(v) for k, v in ks.params().items()}

    def f(pt):
        o = sv.solve({**prm, "phi_top": pt})
        return o["emis"], o["F_ground"]

    _, (de, dg) = jax.jvp(f, (prm["phi_top"],), (-jnp.asarray(D),))
    ts = S3.TangentSolver(r, ks, guard=S3.GUARD_NONE)
    dt = -np.broadcast_to(D[:, :, None], D.shape + (r.Nl,))
    res = ts.solve(dt[None], [0])
    for a, b in ((de, res["d_emis"][0]), (dg, res["d_F_ground"][0])):
        a, b = np.asarray(a), np.asarray(b)
        assert np.max(np.abs(a - b)) <= 1e-10 * np.abs(a).max()
    assert np.max(np.asarray(res["d_res"])) < 1e-6
    assert res["d_up_res"].shape == (1, 3)


def test_guarded_tangent_linear_and_order_independent(small):
    """The guarded map is linear in the boundary (cap average by one boundary), the
    cos and sin parts of an order share the operator, and a per-chain boundary
    differs from the uniform one."""
    r, ks, D = small["r"], small["ks"], small["D"]
    ts = S3.TangentSolver(r, ks, guard=S3.GUARD_PROTOTYPE)
    rng = np.random.default_rng(3)
    b1 = -np.broadcast_to(D[:, :, None], D.shape + (r.Nl,))
    b2 = -D[:, :, None] * rng.random(r.Nl)[None, None]
    res = ts.solve(np.stack([b1, b2, 0.3 * b1 - 2.0 * b2, b1]), [2, 2, 2, 0])
    e = np.asarray(res["d_emis"])
    sc = np.abs(e[:3]).max()
    assert np.max(np.abs(e[2] - (0.3 * e[0] - 2.0 * e[1]))) < 1e-12 * sc
    assert np.max(np.abs(e[0] - e[3])) > 1e-3 * sc        # order 2 is not order 0
    g = np.asarray(res["d_F_ground"])
    assert np.max(np.abs(g[2] - (0.3 * g[0] - 2.0 * g[1]))) < 1e-12 * np.abs(g).max()


def test_per_chain_top_straight_ahead(small):
    """Per-chain tops: in the straight-ahead limit the chains are independent and the
    scheme is homogeneous of degree one, so a top scaled on one chain scales that
    chain's outputs; equal tops on every chain are the scalar path, bit for bit."""
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gates"))
    from run_solve import straight_ahead

    r = small["r"]
    ks = straight_ahead(small["ks"])
    ms = S3.ModeSolver(r, ks, guard=S3.GUARD_NONE)
    o = ms.solve()
    c = 2.0 ** np.arange(r.Nl) / 2 ** 8
    prm = ks.params()
    prm["phi_top_c"] = ks.phi_top[:, :, None] * c[None, None]
    o2 = ms.solve(prm)
    F, F2 = np.asarray(o["F_ground"]), np.asarray(o2["F_ground"])
    assert np.max(np.abs(F2 - F * c[r.ground_line][None, None])) < 1e-13 * np.abs(
        F).max()
    e, e2 = np.asarray(o["emis"]), np.asarray(o2["emis"])
    cn = c[r.node_line]
    assert np.max(np.abs(e2 - e * cn[None, None])) < 1e-13 * np.abs(e * cn).max()
    prm["phi_top_c"] = np.broadcast_to(ks.phi_top[:, :, None],
                                       ks.phi_top.shape + (r.Nl,))
    o3 = ms.solve(prm)
    assert np.array_equal(np.asarray(o3["emis"]), e)


def test_sink_tangent_equals_jvp(small):
    """sink_tangent without guard is jax.jvp of assemble.sink_intensity."""
    import assemble as A

    sv, out = small["sv"], small["out"]
    ms = S3.ModeSolver(small["r"], small["ks"], guard=S3.GUARD_NONE)
    emis = out["emis"]
    de = jnp.asarray(np.random.default_rng(1).random(emis.shape)) * emis
    I, dI = S3.sink_tangent(ms, emis, de[None])
    y, dy = jax.jvp(lambda e: A.sink_intensity(sv, e), (emis,), (de,))
    assert np.max(np.abs(np.asarray(I) - np.asarray(y))) <= 1e-13 * np.abs(y).max()
    assert np.max(np.abs(np.asarray(dI[0]) - np.asarray(dy))) <= 1e-12 * np.abs(
        dy).max()


# ----------------------------------------------------------------- 3D toy
def _toy_reference(ms, h_d, mu_d):
    """Neutrino intensity of the toy of ``test_toy3d`` for a parent top intensity
    cos(m phi), m in ``ms``: the coefficient of cos(m phi), by direct quadrature (no
    solver code).  The angular integral of the child source carries
    cos(m (phi' - phi)); the forward part keeps the azimuth.  Returns (len(ms),)."""
    ms = np.asarray(ms)
    import test_toy3d as T

    def q_child_m(r, mu, n_mu=240, n_phi=48, chunk=64):
        r = np.atleast_1d(np.asarray(r, float))
        mu = np.broadcast_to(np.asarray(mu, float), r.shape)
        if r.size > chunk:
            return np.concatenate([q_child_m(r[i:i + chunk], mu[i:i + chunk])
                                   for i in range(0, r.size, chunk)], axis=1)
        xs, ws = T._gl(n_mu)
        pieces = [(-1.0, -0.2), (-0.2, -0.02), (-0.02, 0.0), (0.0, 0.02),
                  (0.02, 0.2), (0.2, 1.0)]
        mup = np.concatenate([a + (b - a) * xs for a, b in pieces])
        wmu = np.concatenate([(b - a) * ws for a, b in pieces])
        xp, wp = T._gl(n_phi)
        phi, wphi = np.pi * xp, np.pi * wp * 2.0
        fp = T.f_parent(r[:, None], mup[None, :])
        s = np.sqrt(np.clip(1 - mu ** 2, 0, None))
        sp_ = np.sqrt(np.clip(1 - mup ** 2, 0, None))
        ct = (mu[:, None, None] * mup[None, :, None]
              + s[:, None, None] * sp_[None, :, None] * np.cos(phi)[None, None, :])
        kern = np.einsum("nmp,kp->knm", T._vmf(ct),
                         np.cos(ms[:, None] * phi[None]) * wphi[None])
        smooth = np.sum(kern * fp[None] * wmu, axis=-1)          # (k, n)
        return T._rho(r - T.R)[None] / T.LAM * (
            T.G * T.f_parent(r, mu)[None] + (1 - T.G) * smooth)

    R = T.R
    rd = R + h_d
    b = rd * np.sqrt(1 - mu_d ** 2)
    td = -mu_d * rd
    ts = np.sqrt(R ** 2 - b ** 2) if (mu_d < 0 and b < R) else -np.sqrt(
        (R + T.TOP) ** 2 - b ** 2)
    L = td - ts
    d = np.concatenate([[0.0], T.DC * np.geomspace(1e-3, 50.0, 60)])
    d = d[d < L]
    tb = np.unique(np.concatenate([td - d, [ts], np.linspace(ts, td, 60)]))
    if ts < 0 < td:
        tb = np.unique(np.concatenate([tb, [0.0]]))
    tb = tb[(tb >= ts) & (tb <= td)]
    x, w = T._gl(16)
    t = (tb[:-1, None] + np.diff(tb)[:, None] * x).ravel()
    wt = (np.diff(tb)[:, None] * w).ravel()
    rr = np.sqrt(b ** 2 + t ** 2)
    q = q_child_m(rr, -t / rr)
    return np.sum(q * ((1 - np.exp(-(td - t) / T.DC)) * wt)[None], axis=1)


TOY_ORDERS = (0, 1, 2, 4, 8)
#: tolerance relative to the isotropic (m = 0) reference value of the same point
TOY_TOL = 0.01


def toy_modes(guard):
    """Solver side of the toy: one tangent per order, top tangent 1 on every chain."""
    import test_toy3d as T

    R = T.R
    atm = grid.ExponentialAtmosphere(T.RHO0, T.HS, T.TOP, R)
    lev = grid.make_levels(atm, spec=((np.inf, 0.5),))
    mu0, ht = [], []
    for h, m in T.POINTS:
        rr = R + h
        b = rr * np.sqrt(1 - m * m)
        if b < R:
            mu0.append(np.sqrt(1 - (b / R) ** 2))
        else:
            ht.append(b - R)
    fam = grid.make_line_family(lev, n_ground=48,
                                extra_mu0=tuple(mu0) + grid.G13_COSZ_EDGES, r_e=R)
    fam = grid.LineFamily(fam.mu0, np.concatenate([fam.h_tangent, ht]), r_e=R)
    ry = rays.Rays(atm, lev, fam, n_sub=4)
    ks = K.toy_cascade(lam_p=T.LAM, dec_c=1.0 / T.DC, phi_top=1.0, g=T.G,
                       kappa=T.KAPPA, L=32)
    ts = S3.TangentSolver(ry, ks, guard=guard)
    dt = np.broadcast_to(ks.phi_top[:, :, None], ks.phi_top.shape + (ry.Nl,))
    out = ts.solve(np.stack([dt] * len(TOY_ORDERS)), TOY_ORDERS)
    I, dI = S3.sink_tangent(ts.sv, out["emis"], out["d_emis"])
    w = ks.e_widths
    sol, prim = [], []
    for h, m in T.POINTS:
        lvl = int(np.argmin(np.abs(ry.levels - h)))
        cand = np.flatnonzero(ry.node_level == lvl)
        j = cand[np.argmin(np.abs(ry.node_mu[cand] - m))]
        sol.append(np.asarray(dI)[:, 0, 0, j] * w[0] / w[1])
        prim.append(float(np.asarray(I)[0, 0, j]) * w[0] / w[1])
    return np.array(sol).T, np.array(prim)


@pytest.fixture(scope="module")
def toy_ref():
    import test_toy3d as T

    return np.array([_toy_reference(TOY_ORDERS, h, mm) for (h, mm) in T.POINTS]).T


@pytest.mark.parametrize("guard", ["none", "prototype"])
def test_toy3d_azimuthal_orders(toy_ref, guard):
    """3D toy with a parent top intensity cos(m phi), m = 0, 1, 2, 4, 8: the tangent
    of order m against direct quadrature, at the eleven points of ``test_toy3d``
    (ground and altitude, up- and down-going), within 1% of the isotropic value."""
    g = S3.GUARD_NONE if guard == "none" else S3.GUARD_PROTOTYPE
    sol, prim = toy_modes(g)
    ref = toy_ref
    err = np.abs(sol - ref) / ref[0][None]
    assert np.all(err < TOY_TOL), np.round(err, 4)
    # the orders differ from each other (the test sees the m dependence)
    assert np.max(np.abs(ref[1] - ref[0]) / ref[0]) > 0.02


def test_straight_ahead_set_every_order_is_order_0(small):
    """A kernel set without angular channels (straight ahead, as in G2.2 and G2.8):
    the tangent solver runs, every order has the transport of order 0 (bit for bit),
    and the order-0 tangent without guard is jax.jvp of Solver.solve."""
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gates"))
    from run_solve import straight_ahead

    r, D = small["r"], small["D"]
    ks = straight_ahead(small["ks"])
    ts = S3.TangentSolver(r, ks, guard=S3.GUARD_NONE)
    dt = -np.broadcast_to(D[:, :, None], D.shape + (r.Nl,))
    res = ts.solve(np.stack([dt, dt, dt]), [0, 3, 18])
    e, g = np.asarray(res["d_emis"]), np.asarray(res["d_F_ground"])
    assert np.array_equal(e[0], e[1]) and np.array_equal(e[0], e[2])
    assert np.array_equal(g[0], g[2])
    sv = S.Solver(r, ks)
    prm = {k: jnp.asarray(v) for k, v in ks.params().items()}

    def f(pt):
        o = sv.solve({**prm, "phi_top": pt})
        return o["emis"], o["F_ground"]

    _, (de, dg) = jax.jvp(f, (prm["phi_top"],), (-jnp.asarray(D),))
    assert np.max(np.abs(np.asarray(de) - e[0])) <= 1e-10 * np.abs(e[0]).max()
