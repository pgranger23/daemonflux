"""Unit tests of the cascade3d solver core (WP 1.1 / 1.2).

Run: ``taskset -c 0-3 $PY -m pytest tools/cascade3d/test_cascade3d.py -q``
"""

import os

import numpy as np
import pytest

import grid
import kernelset as K
import rays
import solver as S

N3 = os.environ.get("N3", "/home/pigrange/mc3d_work/runs/native3d")
REF = os.path.join(N3, "c1", "ref", "mceq_g11_PHYS_lossoff.npz")


# ----------------------------------------------------------------- G1.0 (a)
def test_x64_enforced():
    """Gate G1.0 (a): fails if JAX is not in 64-bit mode."""
    import jax
    import jax.numpy as jnp

    assert S.x64_enabled()
    assert jax.config.jax_enable_x64
    assert jnp.asarray(1.0).dtype == jnp.float64
    assert (jnp.asarray(1.0) + 1e-15) != 1.0


# ----------------------------------------------------------------- geometry
@pytest.fixture(scope="module")
def ref():
    return np.load(REF)


def test_grammage_against_mceq(ref):
    """Slant depth along the six G1.1 lines vs MCEq's h2X: 1e-4 below 80 km.

    Above 80 km MCEq's 2000-point trapezoid across the 13% density step at 100 km
    dominates (relative 1e-4 to 3e-4 of X < 0.1 g/cm2); there we check instead that
    the same trapezoid on our density reproduces MCEq to 1e-10.
    """
    from scipy.integrate import cumulative_trapezoid

    atm = grid.CorsikaAtmosphere()
    R, top = grid.R_E_CM, grid.R_E_CM + grid.H_TOP_CM
    h = ref["h_tab"]
    for i, c in enumerate(ref["cos_z"]):
        b = R * np.sqrt(1 - c * c)
        m = h <= 80e5
        X = rays.slant_depth(atm, b, h[m])
        assert np.max(np.abs(X / ref["X_tab"][i][m] - 1)) < 1e-4
        assert abs(X[0] / ref["max_X"][i] - 1) < 1e-5
        L = np.sqrt(top ** 2 - b ** 2) - R * c
        dl = np.linspace(0, L, 2000)
        hh = np.sqrt(b ** 2 + (R * c + L - dl) ** 2) - R
        Xt = cumulative_trapezoid(atm.density(hh), dl)[-1]
        assert abs(Xt / ref["max_X"][i] - 1) < 1e-10


def test_path_points_against_mceq(ref):
    """MCEq's (path_h, path_X) pairs along each line: 1e-4 relative below 80 km."""
    atm = grid.CorsikaAtmosphere()
    for i, c in enumerate(ref["cos_z"]):
        b = grid.R_E_CM * np.sqrt(1 - c * c)
        ph, pX = ref[f"path_h_{i}"], ref[f"path_X_{i}"]
        sel = np.flatnonzero((ph > 0) & (ph < 80e5))[::200]
        X = rays.slant_depth(atm, b, ph[sel])
        assert np.max(np.abs(X / pX[sel] - 1)) < 1e-4


@pytest.fixture(scope="module")
def gate_rays():
    atm = grid.CorsikaAtmosphere()
    lev = grid.make_levels(atm)
    fam = grid.LineFamily(np.array(grid.G11_COSZ), np.zeros(0))
    return rays.Rays(atm, lev, fam)


def test_chain_grammage_and_local_angle(gate_rays, ref):
    """Node tables: cumulative grammage = slant_depth; mu = local cosine."""
    r = gate_rays
    atm = r.atm
    for j, c in enumerate(r.ground_mu0):
        b = r.line_b[j]
        nodes = r._chains[j][:r._breaks[j] + 1]
        X = rays.slant_depth(atm, b, r.node_h[nodes])
        assert np.max(np.abs(r.node_Xc[nodes][1:] / X[1:] - 1)) < 1e-9
        rr = grid.R_E_CM + r.node_h[nodes]
        mu = np.sqrt(1 - (b / rr) ** 2)
        assert np.max(np.abs(r.node_mu[nodes] - mu)) < 1e-12
        assert abs(r.node_mu[r.ground_node[j]] - c) < 1e-12
    # sub-point grammage sums to the segment grammage, u from 0 to 1
    assert np.allclose(r.seg_u[:, 0], 0) and np.allclose(r.seg_u[~r.seg_break, -1], 1)
    assert np.all(np.diff(r.seg_u, axis=1) >= -1e-15)
    # loss points carry the same total grammage per line
    for j in range(r.Nl):
        tot_pt = r.pt_dX[r.pt_line == j].sum()
        tot_seg = r.seg_X[r.seg_line == j].sum()
        assert abs(tot_pt / tot_seg - 1) < 1e-10


def test_tangent_lines_geometry():
    """Tangent and pass-through lines: symmetric branches, mu = 0 at the tangent."""
    atm = grid.ExponentialAtmosphere()
    lev = grid.make_levels(atm, spec=((np.inf, 2.0),))
    fam = grid.LineFamily(np.array([0.5]), np.array([lev[3], lev[3] + 0.5e5]))
    r = rays.Rays(atm, lev, fam)
    for j in (1, 2):
        nodes = np.array(r._chains[j])
        mu = r.node_mu[nodes]
        assert np.allclose(mu, -mu[::-1], atol=1e-12)
        X = r.seg_X[r.chain_seg[j, :len(nodes) - 1]]
        assert np.allclose(X, X[::-1], rtol=1e-10)
    assert np.any(r.node_branch == 0)


# --------------------------------------------------------- segment integrator
def _single_line_solver(atm, n_sub=4, levels=None):
    lev = levels if levels is not None else grid.make_levels(
        atm, spec=((np.inf, 1.0),), top_layer=False)
    fam = grid.LineFamily(np.array([1.0]), np.zeros(0), r_e=atm.r_e)
    r = rays.Rays(atm, lev, fam, n_sub=n_sub)
    ks = K.toy_cascade()
    return S.Solver(r, ks), r


def test_segment_limits():
    """Thick limit -> S/sigma, thin limit -> integral of S, constant case exact."""
    import jax.numpy as jnp

    atm = grid.ExponentialAtmosphere(h_top=20e5)
    sv, r = _single_line_solver(atm)
    ds_tot = r.seg_len
    brk = r.seg_break
    x, w = np.polynomial.legendre.leggauss(200)
    # source proportional to rho (represented exactly), constant decay rate D, from
    # thin (D L = 1e-4) to thick (D L = 1e5): Q against direct quadrature
    for D in (1e-9, 1e-6, 1e-4, 1e-2, 1.0):
        T, c = sv._seg_weights(jnp.zeros(1), jnp.array([D]))
        Q = np.asarray(sv._seg_source(jnp.asarray(r.node_rho)[None], c))[0, :-1]
        Tn = np.asarray(T)[0, :-1]
        for s in np.flatnonzero(~brk)[::7]:
            ha, hb = r.node_h[r.seg_a[s]], r.node_h[r.seg_b[s]]
            L = ds_tot[s]
            # vertical line: s from a to b, h linear in s; resolve e^{-D(L-s)} by
            # splitting [0, L] geometrically towards the end
            edges = np.unique(np.concatenate(
                [[0.0], L - np.geomspace(min(L, 1e-3 / D), L, 40)[::-1], [L]]))
            edges = edges[(edges >= 0) & (edges <= L)]
            tot = 0.0
            for a0, b0 in zip(edges[:-1], edges[1:]):
                ss = a0 + (b0 - a0) * 0.5 * (x + 1)
                hh = ha + (hb - ha) * ss / L
                tot += np.sum(atm.density(hh) * np.exp(-D * (L - ss)) * w) * 0.5 * (
                    b0 - a0)
            assert abs(Q[s] / tot - 1) < 2e-4, (D, s, Q[s], tot)
        assert np.allclose(Tn[~brk], np.exp(-D * ds_tot[~brk]), rtol=1e-12)
    # a constant source per length is not of the exact form: record its error
    T, c = sv._seg_weights(jnp.zeros(1), jnp.zeros(1))
    Q = np.asarray(sv._seg_source(jnp.ones((1, r.N)), c))[0, :-1]
    err = np.max(np.abs(Q[~brk] / ds_tot[~brk] - 1))
    assert err < 3e-3   # 1 km levels, H = 7 km: (dlnX)^2 / 8 order
    # thick: interaction-dominated, S = rho * a -> f_end = a / Lambda
    T, c = sv._seg_weights(jnp.array([1e3]), jnp.zeros(1))
    Q = np.asarray(sv._seg_source(jnp.asarray(r.node_rho)[None] * 2.0, c))[0, :-1]
    ok = ~r.seg_break
    assert np.allclose(Q[ok], 2.0 / 1e3, rtol=1e-10)
    # thin: sigma -> 0, S = rho * a exactly integrated (log-linear per grammage)
    T, c = sv._seg_weights(jnp.zeros(1), jnp.zeros(1))
    Q = np.asarray(sv._seg_source(jnp.asarray(r.node_rho)[None] * 3.0, c))[0, :-1]
    assert np.allclose(Q[ok], 3.0 * r.seg_X[ok], rtol=1e-4)


def test_segment_exponential_source():
    """Source falling exponentially in depth (attenuating parent), thin limit."""
    import jax.numpy as jnp

    atm = grid.ExponentialAtmosphere(h_top=40e5)
    sv, r = _single_line_solver(atm, n_sub=8)
    lam = 120.0
    Xn = r.node_Xc
    S = jnp.asarray(r.node_rho * np.exp(-Xn / lam))[None]
    T, c = sv._seg_weights(jnp.zeros(1), jnp.zeros(1))
    Q = np.asarray(sv._seg_source(S, c))[0, :-1]
    ok = ~r.seg_break
    Xa = Xn[r.seg_a]
    exact = lam * (np.exp(-Xa / lam) - np.exp(-(Xa + r.seg_X) / lam))
    rel = np.abs(Q[ok] / exact[ok] - 1)
    # exact up to the trapezoid on the sub-points of the rho-weighted exponential
    assert rel.max() < 2e-4


# ----------------------------------------------------------------- Legendre
@pytest.mark.parametrize("kappa", [1.0, 4.0])
def test_legendre_projection_on_family(kappa):
    """Moments of exp(kappa mu) at every level of the default family vs analytic."""
    import jax.numpy as jnp

    atm = grid.CorsikaAtmosphere()
    lev = grid.make_levels(atm)
    fam = grid.make_line_family(lev, n_ground=48)
    r = rays.Rays(atm, lev, fam, n_sub=1, dX_max=1e9)
    ks = K.synthetic_angular(K.from_mceq("PHYS", loss=False), L=16, e_cut=1.0)
    sv = S.Solver(r, ks)
    f = jnp.asarray(np.exp(kappa * r.node_mu))[None]
    Fl = np.asarray(sv._proj(f))[0]                  # (NL, L+1)
    exact = K.moments_of_law(lambda m: np.exp(kappa * (m - 1)), sv.L) * (
        2 * np.sinh(kappa) / kappa)
    err = np.abs(Fl - exact[None]) / exact[0]
    assert err[:, 0].max() < 2e-3
    # synthesis from L moments: error decreases with L (smooth function)
    errs = []
    for LL in (2, 4, 8, 16):
        M = np.zeros_like(Fl)
        M[:, :LL + 1] = exact[None, :LL + 1]
        syn = np.asarray(sv._synth(jnp.asarray(M)[None]))[0]
        errs.append(np.max(np.abs(syn / np.asarray(f)[0] - 1)))
    assert all(a > b for a, b in zip(errs[:-1], errs[1:])), errs
    assert errs[-1] < 1e-6   # truncation of exp(kappa mu) at L = 16


# ---------------------------------------------------------- continuous loss
def loss_toy(beta=2.0e-3, gamma=2.7, lam=1e-6, dec=None, nE=40, e0=0.03):
    """Parent P (no attenuation to 1e-3) -> same-bin muon-like species with loss."""
    e_bins = e0 * 10 ** (np.arange(nE + 1) / 10.0 - 0.05)
    E = np.sqrt(e_bins[1:] * e_bins[:-1])
    w = np.diff(e_bins)
    d = {
        "e_grid": E, "e_bins": e_bins, "e_widths": w,
        "species": np.array(["P", "mu", "nu"]), "pdg": np.array([2212, 13, 14]),
        "mass": np.array([0.938, 0.10566, 0.0]), "charge": np.array([1, -1, 0]),
        "role": np.array([K.ROLE_HADRON, K.ROLE_LOSS, K.ROLE_SINK]),
        "active": np.ones((3, nE), bool),
        "inv_lambda": np.array([np.full(nE, lam), np.zeros(nE), np.zeros(nE)]),
        "inv_dec": np.zeros((3, nE)),
        "dedx": np.array([np.zeros(nE), np.full(nE, beta), np.zeros(nE)]),
        "phi_top": np.array([E ** -gamma, np.zeros(nE), np.zeros(nE)]),
        "ch_kind": np.array([K.KIND_INT]), "ch_parent": np.array([0]),
        "ch_child": np.array([1]), "Y": np.eye(nE)[None],
    }
    if dec is not None:
        d["inv_dec"][1] = dec(E)
    return K.KernelSet(d)


def loss_exact(E, X, beta, gamma, lam):
    """phi(E, X) = int_0^X lam P0(E + beta (X - x)) exp(-lam x) dx (no decay)."""
    x, w = np.polynomial.legendre.leggauss(400)
    xx = 0.5 * X * (x + 1)
    ww = 0.5 * X * w
    Ep = E[:, None] + beta * (X - xx[None])
    return np.sum(lam * Ep ** -gamma * np.exp(-lam * xx[None]) * ww[None], axis=1)


@pytest.mark.parametrize("mu0", [1.0, 0.15])
def test_csda_power_law(mu0):
    """Constant dE/dX, power-law source, no decay: within 1% of the exact solution."""
    beta, gamma, lam = 2.0e-3, 2.7, 1e-6
    atm = grid.CorsikaAtmosphere()
    lev = grid.make_levels(atm)
    fam = grid.LineFamily(np.array([mu0]), np.zeros(0))
    r = rays.Rays(atm, lev, fam, dX_max=10.0)
    ks = loss_toy(beta, gamma, lam)
    sv = S.Solver(r, ks)
    out = sv.solve()
    f = np.asarray(out["F_ground"])[:, 1, 0]
    X = r.node_Xc[r.ground_node[0]]
    ex = loss_exact(ks.e_grid, X, beta, gamma, lam)
    rel = f / ex - 1
    # top bins miss the source above the grid: compare up to 6 bins below the top
    assert np.max(np.abs(rel[:-6])) < 1e-2, rel


# ------------------------------------------------------------- G1.3 cells
def test_cell_table_analytic():
    """Cell means of I = E^-gamma / sqrt(mu^2 + a^2) on the default ground lines.

    Energy: the band mean in the bin convention equals sum phi_i w_i / sum w_i by
    construction; the log-log option is exact for a power law.  Angle: PCHIP on the
    default directions (ground lines + mu = 0) against asinh, steep near the horizon.
    Measured: energy 4e-7; angle -3.5e-3 in the bin 0-0.1 (peak of width 0.08 at
    cos Z = 0), <= 6e-4 elsewhere.
    """
    import assemble as A

    ks = K.from_mceq("PHYS", loss=False)
    eb, E = ks.e_bins, ks.e_grid
    gam, a = 2.7, 0.08
    mu0 = np.concatenate([[0.0], grid.ground_mu0(48)])
    I = E[:, None] ** -gam / np.sqrt(mu0[None, :] ** 2 + a * a)
    tab = A.cell_table(mu0, I, eb, E, energy="loglog")
    be, ce = np.asarray(A.G13_BAND_EDGES), np.asarray(A.G13_COSZ_EDGES)
    eband = (be[:-1] ** (1 - gam) - be[1:] ** (1 - gam)) / (gam - 1) / np.diff(be)
    ang = (np.arcsinh(ce[1:] / a) - np.arcsinh(ce[:-1] / a)) / np.diff(ce)
    exact = eband[:, None] * ang[None, :]
    rel = tab / exact - 1
    assert np.max(np.abs(rel[:, 0])) < 5e-3, rel
    assert np.max(np.abs(rel[:, 1:])) < 1e-3, rel
    tb = A.cell_table(mu0, I, eb, E, energy="bins")
    w = np.diff(eb)
    i0 = int(np.argmin(np.abs(eb - 0.1)))
    ref = (E[i0:i0 + 2] ** -gam * w[i0:i0 + 2]).sum() / w[i0:i0 + 2].sum()
    assert abs(tb[0, -1] / (ref * ang[-1]) - 1) < 2e-3


# ------------------------------------------------- energy stencil interpolation
def test_interp_log_bound():
    """`_interp_log` is the log-cubic value, bounded only on stencils over 1e3.

    Smooth stencils (also one with a maximum between the nodes, where the cubic
    exceeds every stencil value) keep the unbounded value; a stencil that spans more
    than a factor 1e3 is bounded by its smallest and largest value; a stencil with a
    zero takes the linear branch as before.
    """
    # cubic Lagrange weights on nodes -1, 0, 1, 2 at 0.5, and extrapolating at 2.6
    def lagrange(t):
        x = np.array([-1.0, 0.0, 1.0, 2.0])
        return np.array([np.prod([(t - x[j]) / (x[i] - x[j])
                                  for j in range(4) if j != i]) for i in range(4)])

    w_in, w_out = lagrange(0.5), lagrange(2.6)
    smooth = np.array([1.0, 3.0, 3.0, 1.0])            # maximum between the nodes
    steep = np.array([1.0, 1e-2, 1e-9, 1e-30])         # spans 30 decades
    edge = np.array([1.0, 0.3, 0.02, 1.01e-3])         # just under a factor 1e3
    zero = np.array([1.0, 0.5, 0.0, 0.2])
    for vals, w in [(smooth, w_in), (edge, w_in), (edge, w_out)]:
        got = float(S._interp_log(vals[None], w[None], w[None])[0])
        assert got == pytest.approx(np.exp(np.sum(w * np.log(vals))), rel=1e-13)
    assert float(S._interp_log(smooth[None], w_in[None], w_in[None])[0]) > 3.0
    for w in (w_in, w_out):
        free = np.exp(np.sum(w * np.log(steep)))
        got = float(S._interp_log(steep[None], w[None], w[None])[0])
        assert steep.min() <= got <= steep.max()
        if steep.min() <= free <= steep.max():
            assert got == pytest.approx(free, rel=1e-13)
    # the extrapolating weights leave the range of the steep stencil without the bound
    free = np.exp(np.sum(w_out * np.log(steep)))
    assert not steep.min() <= free <= steep.max()
    assert float(S._interp_log(steep[None], w_out[None], w_out[None])[0]) in (
        pytest.approx(steep.min(), rel=1e-13), pytest.approx(steep.max(), rel=1e-13))
    got = float(S._interp_log(zero[None], w_in[None], w_in[None])[0])
    assert got == pytest.approx(np.sum(w_in * zero), rel=1e-13)
