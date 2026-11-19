"""Energy-marched transport on the line family (``cascade3d`` WP 1.2, JAX float64).

The algorithm is fixed in ``DESIGN.md`` section 4.  Entry point: :class:`Solver`.

* Geometry (``rays.Rays``) and the non-differentiable setup data (species roles,
  ``active`` masks, ``dedx`` and the CSDA range tables) are converted once to device
  arrays.
* The differentiable kernel arrays (``KernelSet.params()``: ``Y``, ``inv_lambda``,
  ``inv_dec``, ``phi_top``, ``g``, ``a``) are arguments of :meth:`Solver.solve`, so
  ``jax.jvp`` / ``jax.vjp`` of the whole solve with respect to them work.
* The energy loop is Python around one jitted per-bin step (two variants: with and
  without angular channels).  Everything a bin needs from higher bins is in the carry
  (``S_acc``, ``M_acc`` and the loss-species windows); the push into lower bins is
  linear.

JAX's CPU thread pool ignores ``OMP_NUM_THREADS``: start every process with
``taskset -c 0-3``.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import kernelset as KS  # noqa: E402


def x64_enabled():
    """True if JAX computes in float64 (gate G1.0 a)."""
    return bool(jax.config.jax_enable_x64) and jnp.zeros(1).dtype == jnp.float64


# ----------------------------------------------------------- small helpers
def _phi01(x):
    """Weights of the linear-in-optical-depth source function (see DESIGN.md)."""
    small = x < 1e-3
    xs = jnp.where(small, 1.0, x)
    em = jnp.exp(-xs)
    e1 = -jnp.expm1(-xs) / xs
    p0 = (e1 - em) / xs
    p1 = (1.0 - e1) / xs
    p0s = 0.5 - x / 3.0 + x * x / 8.0 - x ** 3 / 30.0
    p1s = 0.5 - x / 6.0 + x * x / 24.0 - x ** 3 / 120.0
    return jnp.where(small, p0s, p0), jnp.where(small, p1s, p1)


def _loglin(Aa, Ab, u):
    """exp((1-u) ln Aa + u ln Ab) where both > 0, else linear; broadcasting."""
    pos = (Aa > 0) & (Ab > 0)
    la = jnp.log(jnp.where(pos, Aa, 1.0))
    lb = jnp.log(jnp.where(pos, Ab, 1.0))
    geo = jnp.exp(la + u * (lb - la))
    lin = Aa + u * (Ab - Aa)
    return jnp.where(pos, geo, lin)


def _logquad(Aa, Ab, Ac, uc, u, has_c):
    """Per-grammage source at grammage fraction u (DESIGN 4, source between nodes).

    Quadratic in u of ln A through (0, Aa), (1, Ab), (uc, Ac) where all three are
    positive and a third node exists; log-linear between a and b if only those two
    are positive; linear otherwise.
    """
    pos2 = (Aa > 0) & (Ab > 0)
    pos3 = pos2 & (Ac > 0) & has_c
    la = jnp.log(jnp.where(Aa > 0, Aa, 1.0))
    lb = jnp.log(jnp.where(Ab > 0, Ab, 1.0))
    lc = jnp.log(jnp.where(Ac > 0, Ac, 1.0))
    ucs = jnp.where(has_c, uc, 2.0)
    # Lagrange basis on nodes 0, 1, uc
    l0 = (u - 1.0) * (u - ucs) / ucs
    l1 = u * (u - ucs) / (1.0 - ucs)
    l2 = u * (u - 1.0) / (ucs * (ucs - 1.0))
    quad = jnp.exp(la * l0 + lb * l1 + lc * l2)
    geo = jnp.exp(la + u * (lb - la))
    lin = Aa + u * (Ab - Aa)
    return jnp.where(pos3, quad, jnp.where(pos2, geo, lin))


def _expm1_over(y):
    """(e^y - 1)/y, stable near 0."""
    small = jnp.abs(y) < 1e-6
    ys = jnp.where(small, 1.0, y)
    return jnp.where(small, 1.0 + 0.5 * y + y * y / 6.0, jnp.expm1(ys) / ys)


def _expj(S0, S1, r0, r1, x, ds):
    """Sub-interval source integral with ln J linear in optical depth.

    J = S / sigma; ``r0 = sbar/sigma_0``, ``r1 = sbar/sigma_1`` (1 where sigma = 0),
    so ``ds S_i r_i = x J_i``.  Exact for J exponential in optical depth (any source
    exponential along the step at constant sigma, any rho-proportional source at
    sigma proportional to rho), and in the thick and thin limits.  If an end is
    <= 0: J linear in optical depth.
    """
    a = S0 * r0
    b = S1 * r1
    pos = (a > 0) & (b > 0)
    la = jnp.log(jnp.where(pos, a, 1.0))
    lb = jnp.log(jnp.where(pos, b, 1.0))
    y = x + lb - la
    ym = jnp.minimum(y, 0.0)
    yp = jnp.maximum(y, 0.0)
    formA = a * jnp.exp(-x) * _expm1_over(ym)
    formB = b * _expm1_over(-yp)
    ex = jnp.where(y <= 0, formA, formB)
    p0, p1 = _phi01(x)
    lin = a * p0 + b * p1
    return ds * jnp.where(pos, ex, lin)


def _safe_ratio(num, den):
    ok = den > 0
    return jnp.where(ok, num / jnp.where(ok, den, 1.0), 1.0)


_PUSH_BLOCK = 16
_GL3_X, _GL3_WW = np.polynomial.legendre.leggauss(3)
_GL3_T = tuple(float(v) for v in 0.5 * (_GL3_X + 1.0))
_GL3_W = tuple(float(v) for v in 0.5 * _GL3_WW)


def _interp_log(vals, wl, wpre):
    """Stencil interpolation: exp(sum wl ln v) if all v > 0, else sum wpre v."""
    pos = jnp.all(vals > 0, axis=-1)
    lv = jnp.log(jnp.where(vals > 0, vals, 1.0))
    geo = jnp.exp(jnp.sum(wl * lv, axis=-1))
    lin = jnp.sum(wpre * vals, axis=-1)
    return jnp.where(pos, geo, lin)


def _lagrange4(x):
    """Cubic Lagrange weights on nodes 0, 1, 2, 3 at x (last axis 4)."""
    w0 = -(x - 1.0) * (x - 2.0) * (x - 3.0) / 6.0
    w1 = x * (x - 2.0) * (x - 3.0) / 2.0
    w2 = -x * (x - 1.0) * (x - 3.0) / 2.0
    w3 = x * (x - 1.0) * (x - 2.0) / 6.0
    return jnp.stack([w0, w1, w2, w3], axis=-1)


def _range_tables(ks, loss_idx, n_fine=40, n_ext=6):
    """CSDA range tables for the loss species (numpy).

    Returns ``zf`` (fractional bin index, fine grid from 0 to nE-1+n_ext), ``Rf``
    (nL, nF) range [g/cm^2] at ``zf`` (from E_0; +inf-free), ``lnb`` (nL, nF) ln beta,
    ``lossy`` (nL,) bool.
    """
    E = ks.e_grid
    nE = E.size
    dl = np.log(E[1] / E[0])
    zf = np.linspace(0.0, nE - 1 + n_ext, (nE - 1 + n_ext) * n_fine + 1)
    Rf = np.zeros((len(loss_idx), zf.size))
    lnb = np.zeros((len(loss_idx), zf.size))
    lossy = np.zeros(len(loss_idx), dtype=bool)
    zb = np.arange(nE, dtype=float)
    for i, s in enumerate(loss_idx):
        beta = ks.dedx[s]
        if not np.any(beta > 0):
            continue
        if np.any(beta <= 0):
            raise ValueError("dedx must be > 0 in every bin of a lossy species")
        lossy[i] = True
        lb = np.log(beta)
        # log-log linear between centres, linear extrapolation in ln beta at the ends
        lbf = np.interp(zf, zb, lb)
        hi = zf > nE - 1
        lbf[hi] = lb[-1] + (lb[-1] - lb[-2]) * (zf[hi] - (nE - 1))
        Ef = E[0] * np.exp(dl * zf)
        integrand = Ef / np.exp(lbf) * dl  # dR/dz
        Rf[i] = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:]
                                                       + integrand[:-1])
                                                * np.diff(zf))])
        lnb[i] = lbf
    return zf, Rf, lnb, lossy


# ------------------------------------------------------------------ solver
class Solver:
    """Straight-ahead and delta-M angular transport on one ray table.

    Parameters
    ----------
    rays : rays.Rays
    ks : kernelset.KernelSet
        Supplies the setup data; the differentiable arrays are passed to
        :meth:`solve` (default: ``ks.params()``).
    n_sweeps : int
        Jacobi sweeps of the same-bin closure.
    gamma0 : float
        Pre-whitening exponent of the energy interpolation of loss species.
    k_top : int, optional
        Highest bin marched (default: the last bin).
    store_nodes : bool
        Keep the transported intensities at all nodes for every bin (memory!).
    quad_source : bool
        Source shape between nodes quadratic in grammage of ln(S/rho) through
        three chain nodes (default); False: log-linear between the two end nodes.
    """

    def __init__(self, rays, ks, n_sweeps=6, gamma0=2.0, k_top=None,
                 store_nodes=False, quad_source=True):
        ks.validate()
        self.rays, self.ks = rays, ks
        self.n_sweeps = int(n_sweeps)
        self.gamma0 = float(gamma0)
        self.store_nodes = bool(store_nodes)
        self.quad_source = bool(quad_source)
        nE = ks.nE
        self.nE = nE
        self.k_top = nE - 1 if k_top is None else int(k_top)
        role = ks.role
        # species lists (kernel-set indices)
        self.H = [s for s in range(ks.nS) if role[s] == KS.ROLE_HADRON]
        self.Lsp = [s for s in range(ks.nS) if role[s] == KS.ROLE_LOSS]
        self.Sk = [s for s in range(ks.nS) if role[s] == KS.ROLE_SINK]
        self.T = self.H + self.Lsp
        self.C = self.T + self.Sk
        nH, nL, nT, nC = len(self.H), len(self.Lsp), len(self.T), len(self.C)
        self.nH, self.nL, self.nT, self.nC = nH, nL, nT, nC
        posT = {s: i for i, s in enumerate(self.T)}
        posC = {s: i for i, s in enumerate(self.C)}
        # channels kept: parent transported, child in C
        keep = [j for j in range(ks.ch_kind.size)
                if ks.ch_parent[j] in posT and ks.ch_child[j] in posC]
        self.ch_keep = np.array(keep, dtype=np.int64)
        self.ch_p = np.array([posT[ks.ch_parent[j]] for j in keep], dtype=np.int64)
        self.ch_c = np.array([posC[ks.ch_child[j]] for j in keep], dtype=np.int64)
        self.ch_kind = ks.ch_kind[self.ch_keep]
        # loss species may not receive same-bin input from loss species
        for j in keep:
            p, c = ks.ch_parent[j], ks.ch_child[j]
            if role[p] == KS.ROLE_LOSS and role[c] == KS.ROLE_LOSS:
                if np.any(np.diag(ks.Y[j]) != 0):
                    raise ValueError("same-bin coupling among loss species")
            if role[p] == KS.ROLE_LOSS and role[c] == KS.ROLE_HADRON:
                raise ValueError("loss species may not feed hadrons")
        # angular channels (positions in the kept list)
        kpos = {j: i for i, j in enumerate(keep)}
        ang = [int(j) for j in ks.ang_channel]
        self.ang_sel = np.array([i for i, j in enumerate(ang) if j in kpos],
                                dtype=np.int64)
        self.ang_kept = np.array([kpos[ang[i]] for i in self.ang_sel], dtype=np.int64)
        self.nA = ks.nA if self.ang_sel.size else 0
        self.L = ks.L if self.ang_sel.size else 0
        self.active_T = np.asarray(ks.active)[self.T]
        # largest child-bin minus parent-bin with a non-zero yield (sink up-scatter)
        reach = 0
        for j in keep:
            m, kk = np.nonzero(ks.Y[j])
            if m.size:
                reach = max(reach, int((m - kk).max()))
        self.up_reach = reach
        self._setup_geometry()
        self._setup_loss()
        self._step_fwd = jax.jit(self._make_step(angular=False), donate_argnums=1)
        self._step_ang = (jax.jit(self._make_step(angular=True), donate_argnums=1)
                          if self.nA else None)

    # ---------------------------------------------------------- setup
    def _setup_geometry(self):
        r = self.rays
        g = {}
        j = jnp.asarray
        g["seg_a"], g["seg_b"] = j(r.seg_a), j(r.seg_b)
        g["seg_rho"], g["seg_u"] = j(r.seg_rho), j(r.seg_u)
        g["seg_ds"], g["seg_dX"] = j(r.seg_ds), j(r.seg_dX)
        g["seg_break"] = j(r.seg_break)
        g["rho_a"] = j(r.node_rho[r.seg_a])
        g["rho_b"] = j(r.node_rho[r.seg_b])
        segc = np.where(r.seg_c >= 0, r.seg_c, r.seg_a)
        g["seg_c"], g["seg_uc"] = j(segc), j(r.seg_uc)
        g["seg_hasc"] = j((r.seg_c >= 0) & self.quad_source)
        g["rho_c"] = j(r.node_rho[segc])
        g["chain_seg"] = j(r.chain_seg)
        g["node_pos"], g["node_line"] = j(r.node_pos), j(r.node_line)
        g["node_rho"] = j(r.node_rho)
        g["node_level"], g["node_slot"] = j(r.node_level), j(r.node_slot)
        g["lev_node"] = j(r.lev_node)
        g["lev_rho"] = j(self.rays.atm.density(r.levels))
        # points
        g["pt_prev"], g["pt_seg"] = j(r.pt_prev), j(np.maximum(r.pt_seg, 0))
        g["pt_isseg"] = j(r.pt_seg >= 0)
        g["pt_u"], g["pt_ds"], g["pt_dX"] = j(r.pt_u), j(r.pt_ds), j(r.pt_dX)
        g["pt_rho"] = j(r.pt_rho)
        g["pt_break"], g["pt_start"] = j(r.pt_break), j(r.pt_start)
        g["pt_node"] = j(np.maximum(r.pt_node, 0))
        g["pt_isnode"] = j(r.pt_node >= 0)
        g["node_point"] = j(r.node_point)
        g["buckets"] = tuple(j(b) for b in r.buckets)
        g["active_T"] = j(self.active_T)
        if self.L or self.nA:
            L = self.L
            mu = np.concatenate([r.node_mu, [0.0]])
            P = KS.legendre_table(mu[r.lev_node], L)  # (L+1, NL, M)
            w = r.lev_w
            g["WP"] = j(np.moveaxis(P * w[None], 0, -1))  # (NL, M, L+1)
            fac = (2 * np.arange(L + 1) + 1) / 2.0
            g["QP"] = j(np.moveaxis(P, 0, -1) * fac)      # (NL, M, L+1)
        self.geo = g

    def _setup_loss(self):
        ks, r = self.ks, self.rays
        # loss species with identical dedx (the six helicity muons) share the energy
        # stencil: computed once, broadcast over species
        rows = np.asarray(ks.dedx)[self.Lsp] if self.Lsp else np.zeros((0, self.nE))
        self.share_loss = bool(len(self.Lsp) > 1 and np.all(rows == rows[:1]))
        rep = self.Lsp[:1] if self.share_loss else self.Lsp
        zf, Rf, lnb, lossy = _range_tables(ks, rep)
        self.lossy = lossy
        nE = self.nE
        dl = float(np.log(ks.e_grid[1] / ks.e_grid[0]))
        self.dl = dl
        W = 4
        if np.any(lossy):
            Rb = np.array([np.interp(np.arange(nE), zf, Rf[i])
                           for i in range(len(rep))])
            dxm = float(r.pt_dX.max())
            zmax = 0.0
            for i in np.flatnonzero(lossy):
                z = np.interp(Rb[i] + dxm, Rf[i], zf) - np.arange(nE)
                zmax = max(zmax, float(z.max()))
            W = int(np.ceil(zmax)) + 5
        else:
            Rb = np.zeros((len(rep), nE))
        self.W = W
        g = self.geo
        g["zf"], g["Rf"], g["lnb"] = jnp.asarray(zf), jnp.asarray(Rf), jnp.asarray(lnb)
        g["Rb"], g["lossy"] = jnp.asarray(Rb), jnp.asarray(lossy)
        # inverse range table on a uniform grid in q = ln(R + R0): z(q) without search
        R0 = 1e-3
        nq = 40001
        qtab = np.zeros((len(rep), nq))
        q0 = np.zeros(len(rep))
        dq = np.ones(len(rep))
        for i in range(len(rep)):
            if not lossy[i]:
                continue
            qf = np.log(Rf[i] + R0)
            q0[i], q1 = qf[0], qf[-1]
            dq[i] = (q1 - q0[i]) / (nq - 1)
            qtab[i] = np.interp(q0[i] + dq[i] * np.arange(nq), qf, zf)
        g["qtab"], g["q0"] = jnp.asarray(qtab), jnp.asarray(q0)
        g["dq"] = jnp.asarray(dq)
        self.R0 = R0
        # ln beta at integer bin offsets (bins beyond the grid: extrapolated table)
        nb = nE + W + 4
        g["lnb_bins"] = jnp.asarray(np.array(
            [np.interp(np.arange(nb, dtype=float), zf, lnb[i])
             for i in range(len(rep))]).reshape(len(rep), nb))

    # --------------------------------------------------------- pieces
    def _proj(self, F):
        """Legendre moments at the levels: F (x, N) -> (x, NL, L+1)."""
        g = self.geo
        Fp = jnp.concatenate([F, jnp.zeros((F.shape[0], 1))], axis=1)
        return jnp.einsum("xqm,qml->xql", Fp[:, g["lev_node"]], g["WP"])

    def _synth(self, M):
        """Synthesis at the nodes: M (x, NL, L+1) -> (x, N)."""
        g = self.geo
        S_lev = jnp.einsum("xql,qml->xqm", M, g["QP"])
        return S_lev[:, g["node_level"], g["node_slot"]]

    def _seg_weights(self, Lam, De):
        """Transmission T (h, Ns+1) and sub-point weights c (h, Ns+1, nq)."""
        g = self.geo
        rho, ds, dX = g["seg_rho"], g["seg_ds"], g["seg_dX"]
        sig = rho[None] * Lam[:, None, None] + De[:, None, None]      # (h,Ns,nq)
        x = dX[None] * Lam[:, None, None] + ds[None] * De[:, None, None]  # (h,Ns,ns)
        okd = ds > 0
        sbar = jnp.where(okd[None], x / jnp.where(okd, ds, 1.0)[None], 0.0)
        r0 = _safe_ratio(sbar, sig[..., :-1])
        r1 = _safe_ratio(sbar, sig[..., 1:])
        p0, p1 = _phi01(x)
        t0 = ds[None] * r0 * p0
        t1 = ds[None] * r1 * p1
        tot = jnp.sum(x, axis=-1, keepdims=True)
        after = jnp.exp(-(tot - jnp.cumsum(x, axis=-1)))             # (h,Ns,ns)
        c = jnp.zeros(sig.shape)
        c = c.at[..., :-1].add(after * t0)
        c = c.at[..., 1:].add(after * t1)
        T = jnp.exp(-tot[..., 0])
        brk = g["seg_break"][None]
        T = jnp.where(brk, 0.0, T)
        c = jnp.where(brk[..., None], 0.0, c)
        after = jnp.where(brk[..., None], 0.0, after)
        nh = Lam.shape[0]
        T = jnp.concatenate([T, jnp.ones((nh, 1))], axis=1)
        c = jnp.concatenate([c, jnp.zeros((nh, 1, c.shape[-1]))], axis=1)
        w = {"c": c, "x": x, "r0": r0, "r1": r1, "after": after}
        return T, w

    def _seg_source(self, S, w):
        """Q (h, Ns+1) of node sources S (h, N).

        Source at the sub-points: quadratic (or log-linear) per grammage between the
        nodes; per sub-interval: ln J linear in optical depth (``_expj``).
        """
        g = self.geo
        Aa = S[:, g["seg_a"]] / g["rho_a"][None]
        Ab = S[:, g["seg_b"]] / g["rho_b"][None]
        Ac = S[:, g["seg_c"]] / g["rho_c"][None]
        Aq = _logquad(Aa[..., None], Ab[..., None], Ac[..., None],
                      g["seg_uc"][None, :, None], g["seg_u"][None],
                      g["seg_hasc"][None, :, None])
        Sq = g["seg_rho"][None] * Aq
        ds = g["seg_ds"][None]
        contrib = _expj(Sq[..., :-1], Sq[..., 1:], w["r0"], w["r1"], w["x"], ds)
        Q = jnp.sum(w["after"] * contrib, axis=-1)
        return jnp.concatenate([Q, jnp.zeros((S.shape[0], 1))], axis=1)

    def _seg_linear(self, w):
        """alpha, beta (h, Ns+1): Q of a same-bin source linear per grammage."""
        c = w["c"]
        g = self.geo
        u, rho = g["seg_u"], g["seg_rho"]
        al = jnp.sum(c[:, :-1] * (rho * (1.0 - u))[None], axis=-1) / g["rho_a"][None]
        be = jnp.sum(c[:, :-1] * (rho * u)[None], axis=-1) / g["rho_b"][None]
        z = jnp.zeros((c.shape[0], 1))
        return jnp.concatenate([al, z], axis=1), jnp.concatenate([be, z], axis=1)

    def _chain_scan(self, T, Q, f0):
        """f along every chain; returns node values (h, N)."""
        g = self.geo
        Tc = T[:, g["chain_seg"]]   # (h, Nl, Lmax-1)
        Qc = Q[:, g["chain_seg"]]

        def body(f, tq):
            t, q = tq
            f = t * f + q
            return f, f

        f0b = jnp.broadcast_to(f0[:, None], Tc.shape[:2])
        _, ys = jax.lax.scan(body, f0b, (jnp.moveaxis(Tc, 2, 0),
                                         jnp.moveaxis(Qc, 2, 0)))
        fch = jnp.concatenate([f0b[None], ys], axis=0)   # (Lmax, h, Nl)
        return fch[g["node_pos"], :, g["node_line"]].T

    def _point_scan(self, A, B):
        """phi_p = A_p phi_{p-1} + B_p on the loss points; A, B (l, Np+1)."""
        g = self.geo
        out = jnp.zeros(A.shape)

        def body(f, ab):
            a, b = ab
            f = a * f + b
            return f, f

        for idx in g["buckets"]:
            Ab, Bb = A[:, idx], B[:, idx]   # (l, nb, P)
            f0 = jnp.zeros(Ab.shape[:2])
            _, ys = jax.lax.scan(body, f0, (jnp.moveaxis(Ab, 2, 0),
                                            jnp.moveaxis(Bb, 2, 0)))
            out = out.at[:, idx].set(jnp.moveaxis(ys, 0, 2))
        return out

    # ------------------------------------------------------------ step
    def _make_step(self, angular):
        nH, nL, nT, nC = self.nH, self.nL, self.nT, self.nC
        nE, W, nA, L = self.nE, self.W, self.nA, self.L
        Tidx = jnp.asarray(np.array(self.T, dtype=np.int64))
        ch_p, ch_c = jnp.asarray(self.ch_p), jnp.asarray(self.ch_c)
        ch_keep = jnp.asarray(self.ch_keep)
        is_int = jnp.asarray(self.ch_kind == KS.KIND_INT)
        ang_sel, ang_kept = jnp.asarray(self.ang_sel), jnp.asarray(self.ang_kept)
        n_sweeps, gamma0, dl = self.n_sweeps, self.gamma0, self.dl
        sink_child = jnp.asarray(np.arange(nC) >= nT)
        up_reach = self.up_reach
        midx = jnp.arange(nE)

        def dense(vals_ch, mask_int):
            """(nCh_kept, ...) -> (nC, nT, ...) for int or dec channels."""
            v = jnp.where(mask_int.reshape((-1,) + (1,) * (vals_ch.ndim - 1)),
                          vals_ch, 0.0)
            out = jnp.zeros((nC, nT) + vals_ch.shape[1:])
            return out.at[ch_c, ch_p].add(v)

        def step(k, carry, prm):
            g = self.geo
            S_acc, M_acc, fwin, swin = carry
            Y = prm["Y"][ch_keep]                         # (nch, nE, nE)
            Yk = jnp.take(Y, k, axis=2)                   # (nch, nE) child m
            if angular:
                gk = jnp.take(prm["g"][ang_sel], k, axis=2)        # (nang, nA)
                ak = jnp.take(prm["a"][ang_sel], k, axis=3)        # (nang, L+1, nA)
                Yf = Yk.at[ang_kept, :nA].multiply(gk)
                Ynf = (Yk[ang_kept, None, :nA] * (1.0 - gk)[:, None, :] * ak)
                nf_full = jnp.zeros((Yk.shape[0], L + 1, nA)).at[ang_kept].set(Ynf)
                G_int = dense(nf_full, is_int)            # (nC, nT, L+1, nA)
                G_dec = dense(nf_full, ~is_int)
            else:
                Yf = Yk
            Y_int = dense(Yf, is_int)                     # (nC, nT, nE)
            Y_dec = dense(Yf, ~is_int)
            lam = jnp.take(prm["inv_lambda"][Tidx], k, axis=1)   # (nT,)
            dec = jnp.take(prm["inv_dec"][Tidx], k, axis=1)
            ftop = jnp.take(prm["phi_top"][Tidx], k, axis=1)
            act = jnp.take(g["active_T"], k, axis=1)
            rho_n = g["node_rho"]
            # same-bin couplings among transported species (forward)
            Cint = jnp.take(Y_int[:nT], k, axis=2)         # (nT c, nT p)
            Cdec = jnp.take(Y_dec[:nT], k, axis=2)
            eye = jnp.eye(nT)
            self_int = jnp.diagonal(Cint)
            self_dec = jnp.diagonal(Cdec)
            Cint_o = Cint * (1.0 - eye)
            Cdec_o = Cdec * (1.0 - eye)
            if angular:
                kk = jnp.minimum(k, nA - 1)
                Gs_int = jnp.take(G_int[:nT], kk, axis=3)  # (nT, nT, L+1)
                Gs_dec = jnp.take(G_dec[:nT], kk, axis=3)
                lev_rho = g["lev_rho"]

            def same_bin(F, rows):
                """Same-bin source (rows of T) from transported F (nT, N)."""
                Ai = lam[:, None] * F * rho_n[None]
                Ad = dec[:, None] * F
                S = Cint_o[rows] @ Ai + Cdec_o[rows] @ Ad
                if angular:
                    Fl = self._proj(F)                    # (nT, NL, L+1)
                    Ri = lam[:, None, None] * Fl * lev_rho[None, :, None]
                    Rd = dec[:, None, None] * Fl
                    Ms = (jnp.einsum("cpl,pql->cql", Gs_int[rows], Ri)
                          + jnp.einsum("cpl,pql->cql", Gs_dec[rows], Rd))
                    S = S + self._synth(Ms)
                return S

            S_fix = jnp.take(S_acc, k, axis=1)            # (nC, N)
            if angular:
                Mk = jnp.take(M_acc, jnp.minimum(k, nA - 1), axis=1)
                S_fix = S_fix + self._synth(Mk)
            # ---------------- hadrons
            hs = jnp.arange(nH)
            Lam_e = lam[:nH] * (1.0 - self_int[:nH])
            De_e = dec[:nH] * (1.0 - self_dec[:nH])
            Tseg, c = self._seg_weights(Lam_e, De_e)
            Qfix = self._seg_source(S_fix[:nH], c)
            al, be = self._seg_linear(c)
            F_H = self._chain_scan(Tseg, Qfix, ftop[:nH]) * act[:nH, None]
            zL = jnp.zeros((nL, F_H.shape[1]))

            def sweep(i, st):
                F_H, _ = st
                Fall = jnp.concatenate([F_H, zL], axis=0)
                Ss = same_bin(Fall, hs)
                Q = Qfix + jnp.concatenate(
                    [al[:, :-1] * Ss[:, g["seg_a"]] + be[:, :-1] * Ss[:, g["seg_b"]],
                     jnp.zeros((nH, 1))], axis=1)
                Fn = self._chain_scan(Tseg, Q, ftop[:nH]) * act[:nH, None]
                scale = jnp.max(jnp.abs(Fn), axis=1)
                dmax = jnp.max(jnp.abs(Fn - F_H), axis=1)
                res = jnp.max(_safe_ratio(dmax, scale) * (scale > 0))
                return Fn, res

            if n_sweeps > 0 and nH > 0:
                F_H, res = jax.lax.fori_loop(0, n_sweeps, sweep, (F_H, 0.0))
            else:
                res = jnp.asarray(0.0)
            # ---------------- loss species
            if nL:
                Fall = jnp.concatenate([F_H, zL], axis=0)
                lrows = jnp.arange(nH, nT)
                S_L = S_fix[nH:nT] + same_bin(Fall, lrows)
                F_L, fwin, swin = self._loss_transport(
                    k, S_L, lam[nH:], dec[nH:], ftop[nH:], fwin, swin, prm)
                F_L = F_L * act[nH:, None]
                F_T = jnp.concatenate([F_H, F_L], axis=0)
            else:
                F_T = F_H
            # ---------------- push into lower bins (and same bin for sinks)
            # transported children: bins below k (same bin was closed above; their
            # up-scatter is dropped); sinks: every bin, including up-scatter
            mask = jnp.where(sink_child[:, None], True, midx[None] < k)
            Ai = lam[:, None] * F_T * rho_n[None]
            Ad = dec[:, None] * F_T
            Ycat = jnp.concatenate([Y_int, Y_dec], axis=1) * mask[:, None, :]
            Acat = jnp.concatenate([Ai, Ad], axis=0)
            # blocks of child bins; a block entirely above k + up_reach gets nothing
            for m0 in range(0, nE, _PUSH_BLOCK):
                m1 = min(m0 + _PUSH_BLOCK, nE)

                def add(Sa, m0=m0, m1=m1):
                    blk = jnp.einsum("cpm,pn->cmn", Ycat[:, :, m0:m1], Acat)
                    return jax.lax.dynamic_update_slice_in_dim(
                        Sa, Sa[:, m0:m1] + blk, m0, axis=1)

                S_acc = jax.lax.cond(m0 <= k + up_reach, add, lambda Sa: Sa, S_acc)
            if angular:
                Fl = self._proj(F_T)
                Ri = lam[:, None, None] * Fl * g["lev_rho"][None, :, None]
                Rd = dec[:, None, None] * Fl
                mA = mask[:, :nA]
                Gi = G_int * mA[:, None, None, :]
                Gd = G_dec * mA[:, None, None, :]
                M_acc = M_acc + (jnp.einsum("cplm,pql->cmql", Gi, Ri)
                                 + jnp.einsum("cplm,pql->cmql", Gd, Rd))
            out = {"F_ground": F_T[:, self._gnode], "res": res}
            if self.store_nodes:
                out["F"] = F_T
            return (S_acc, M_acc, fwin, swin), out

        return step

    def _stencil(self, k, Rk, dXs):
        """Energy stencil of E* for steps of grammage ``dXs``: slots, weights, z."""
        g = self.geo
        W, dl, gamma0 = self.W, self.dl, self.gamma0
        nq = g["qtab"].shape[1]
        q = jnp.log(Rk[:, None] + dXs[None] + self.R0)
        t = (q - g["q0"][:, None]) / g["dq"][:, None]
        t = jnp.clip(t, 0.0, nq - 1.000001)
        i0 = jnp.floor(t).astype(jnp.int32)
        fr = t - i0
        tab = g["qtab"]
        lidx = jnp.arange(tab.shape[0])[:, None]
        zabs = tab[lidx, i0] * (1.0 - fr) + tab[lidx, i0 + 1] * fr
        z = jnp.where(g["lossy"][:, None], zabs - k, 0.0)
        z = jnp.clip(z, 0.0, W - 3.999)
        base = jnp.clip(jnp.floor(z) - 1.0, 0.0, W - 4.0)
        jj = base[..., None] + jnp.arange(4.0)
        wl = _lagrange4(z - base)
        wpre = wl * jnp.exp(gamma0 * dl * (jj - z[..., None]))
        slots = (k + jj.astype(jnp.int32)) % W
        return z, jj, wl, wpre, slots

    def _loss_transport(self, k, S_L, lam, dec, ftop, fwin, swin, prm):
        """CSDA transport of the loss species on the loss points (DESIGN 4).

        Energy interpolation at E*: cubic Lagrange in ln E of ln f (exact for any
        local power law); if a stencil value is <= 0, cubic in ln E of f E^gamma0.
        """
        g = self.geo
        W, nE = self.W, self.nE
        nL = S_L.shape[0]
        # sources at the points (log-linear per grammage in each segment)
        sa = g["seg_a"][g["pt_seg"]]
        sb = g["seg_b"][g["pt_seg"]]
        sc = g["seg_c"][g["pt_seg"]]
        Aa = S_L[:, sa] / g["rho_a"][g["pt_seg"]][None]
        Ab = S_L[:, sb] / g["rho_b"][g["pt_seg"]][None]
        Ac = S_L[:, sc] / g["rho_c"][g["pt_seg"]][None]
        Sp = g["pt_rho"][None] * _logquad(
            Aa, Ab, Ac, g["seg_uc"][g["pt_seg"]][None], g["pt_u"][None],
            g["seg_hasc"][g["pt_seg"]][None])
        Sp = jnp.where(g["pt_isnode"][None], S_L[:, g["pt_node"]], Sp)
        Sp = jnp.concatenate([Sp, jnp.zeros((nL, 1))], axis=1)   # dummy point
        slot = k % W
        swin = swin.at[slot].set(Sp)
        Rk = jnp.take(g["Rb"], k, axis=1)                         # (nL,)
        dX = g["pt_dX"]
        kf = jnp.asarray(k, float)
        z, jj, wl, wpre, slots = self._stencil(k, Rk, dX)
        # decay and interaction rates at E* (log-linear between bins)
        fl = jnp.broadcast_to(jnp.floor(z), (nL,) + z.shape[1:])
        ib = jnp.clip(fl.astype(jnp.int32) + k, 0, nE - 1)
        ib1 = jnp.clip(ib + 1, 0, nE - 1)
        tz = jnp.broadcast_to(z, fl.shape) - fl
        Lrow = prm["inv_lambda"][jnp.asarray(self.Lsp)]
        Drow = prm["inv_dec"][jnp.asarray(self.Lsp)]

        def at(row, i):
            return jax.vmap(lambda r, ii: r[ii])(row, i)

        D_star = _loglin(at(Drow, ib), at(Drow, ib1), tz)
        L_star = _loglin(at(Lrow, ib), at(Lrow, ib1), tz)
        prev = g["pt_prev"]
        rho_p = g["pt_rho"]
        rho_u = jnp.concatenate([g["pt_rho"], jnp.ones(1)])[prev]
        sig_star = rho_u[None] * L_star + D_star
        sig_m = rho_p[None] * lam[:, None] + dec[:, None]
        ds = g["pt_ds"]
        x = (0.5 * (D_star + dec[:, None]) * ds[None]
             + 0.5 * (L_star + lam[:, None]) * dX[None])
        okd = ds > 0
        sbar = jnp.where(okd[None], x / jnp.where(okd, ds, 1.0)[None], 0.0)
        ex = jnp.exp(-x)
        lidx = jnp.arange(nL)[:, None, None]
        pidx = prev[None, :, None]
        P1 = swin.shape[2]
        fwf, swf = fwin.reshape(-1), swin.reshape(-1)

        def wget(flat, sl, pts):
            # flat 1-D gather of window[sl, l, pts] (faster than 3-index gather)
            return jnp.take(flat, (sl * nL + lidx) * P1 + pts)

        f_up = wget(fwf, slots, pidx)                             # (nL, Np, 4)
        s_up = wget(swf, slots, pidx)
        cur = (jj == 0.0)

        lbb = g["lnb_bins"]
        lrow = jnp.arange(lbb.shape[0])[:, None, None]

        def bratio(jq):
            # beta(E_{k+j}) / beta(E_k) at the stencil bins: psi = beta f and
            # beta S are interpolated (smooth; f has the kinks of beta)
            idx = k + jq.astype(jnp.int32)
            return jnp.where(g["lossy"][:, None, None],
                             jnp.exp(lbb[lrow, idx] - lbb[lrow[:, 0], k][..., None]),
                             1.0)

        br = bratio(jj)
        # r* S* = (beta S)(E*) / beta_k
        Sstar_r = _interp_log(s_up * br, wl, wpre)
        # The energy changes along a step (by up to ~70% at 0.03 GeV), so the source
        # beta S varies much more than exponentially along it; the step integral is
        # rescaled by (quadrature along the characteristic) / (log mean of the ends).
        # Simpson along the characteristic: ends (E* upstream, E_m here) and the
        # midpoint (energy at half the step, source averaged over both points)
        here = jnp.arange(Sp.shape[1] - 1)[None, :, None]
        zq, jq, wlq, wpq, sl = self._stencil(k, Rk, 0.5 * dX)
        bq = bratio(jq)
        s_prev = _interp_log(wget(swf, sl, pidx) * bq, wlq, wpq)
        s_here = _interp_log(wget(swf, sl, here) * bq, wlq, wpq)
        quad = (Sstar_r + 2.0 * (s_prev + s_here) + Sp[:, :-1]) / 6.0
        # thin-limit mean of the end values that _expj integrates exactly (log mean)
        s0, s1 = Sstar_r, Sp[:, :-1]
        posm = (s0 > 0) & (s1 > 0)
        l0 = jnp.log(jnp.where(posm, s0, 1.0))
        l1 = jnp.log(jnp.where(posm, s1, 1.0))
        lmean = jnp.where(posm, s0 * _expm1_over(l1 - l0), 0.5 * (s0 + s1))
        corr = jnp.where(lmean > 0, quad / jnp.where(lmean > 0, lmean, 1.0), 1.0)
        C1 = ex
        B0 = corr * _expj(s0, s1, _safe_ratio(sbar, sig_star),
                          _safe_ratio(sbar, sig_m), x, ds[None])
        brk = g["pt_break"][None]
        st = g["pt_start"][None]
        C1 = jnp.where(brk | st, 0.0, C1)
        B0 = jnp.where(brk, 0.0, jnp.where(st, ftop[:, None], B0))
        # pad to the dummy point: identity step
        def pad(a, v):
            a = jnp.broadcast_to(a, (nL,) + a.shape[1:])
            return jnp.concatenate([a, jnp.full((nL, 1) + a.shape[2:], v, a.dtype)],
                                   axis=1)
        xs = {"C1": pad(C1, 1.0), "B0": pad(B0, 0.0), "fup": pad(f_up * br, 0.0),
              "cur": pad(cur, False), "wl": pad(wl, 0.0),
              "wpre": pad(wpre, 0.0)}
        xs["wl"] = xs["wl"].at[:, -1, 0].set(1.0)
        xs["wpre"] = xs["wpre"].at[:, -1, 0].set(1.0)
        xs["cur"] = xs["cur"].at[:, -1, 0].set(True)
        phi = self._point_scan_nl(xs)
        phi = phi.at[:, -1].set(0.0)
        fwin = fwin.at[slot].set(phi)
        F_L = phi[:, g["node_point"]]
        return F_L, fwin, swin

    def _point_scan_nl(self, xs):
        """phi_p = C1_p f*(phi_{p-1}) + B0_p on the loss points (nonlinear in phi)."""
        g = self.geo
        nL = xs["C1"].shape[0]
        out = jnp.zeros(xs["C1"].shape)

        def body(f, x):
            vals = jnp.where(x["cur"], f[..., None], x["fup"])
            fs = _interp_log(vals, x["wl"], x["wpre"])
            f = x["C1"] * fs + x["B0"]
            return f, f

        for idx in g["buckets"]:
            xb = {k: jnp.moveaxis(v[:, idx], 2, 0) for k, v in xs.items()}
            f0 = jnp.zeros((nL, idx.shape[0]))
            _, ys = jax.lax.scan(body, f0, xb)
            out = out.at[:, idx].set(jnp.moveaxis(ys, 0, 2))
        return out

    # ------------------------------------------------------------ solve
    def init_carry(self):
        N = self.rays.N
        S_acc = jnp.zeros((self.nC, self.nE, N))
        if self.nA:
            NL = self.rays.levels.size
            M_acc = jnp.zeros((self.nC, self.nA, NL, self.L + 1))
        else:
            M_acc = jnp.zeros((1, 1, 1, 1))
        nP = self.rays.Np + 1
        nl = max(self.nL, 1)
        fwin = jnp.zeros((self.W, nl, nP))
        swin = jnp.zeros((self.W, nl, nP))
        return (S_acc, M_acc, fwin, swin)

    @property
    def _gnode(self):
        return jnp.asarray(self.rays.ground_node)

    def solve(self, params=None, verbose=False):
        """Run the energy march.

        Parameters
        ----------
        params : dict, optional
            Differentiable kernel arrays (``KernelSet.params()`` layout), numpy or
            JAX; default the kernel set's own.

        Returns
        -------
        dict
            ``emis`` (nE, nSink, N): sink emissivity per unit length at the nodes;
            ``F_ground`` (nE, nT, n_ground): transported species at the inward ground
            nodes; ``res`` (nE,): relative change in the last same-bin sweep;
            ``F`` (nE, nT, N) if ``store_nodes``.  Bins above ``k_top`` are zero.
        """
        if params is None:
            params = self.ks.params()
        prm = {k: jnp.asarray(v) for k, v in params.items()}
        carry = self.init_carry()
        outs = {}
        t0 = time.time()
        for k in range(self.k_top, -1, -1):
            stepf = self._step_ang if (self.nA and k < self.nA) else self._step_fwd
            carry, o = stepf(jnp.int32(k), carry, prm)
            for key, v in o.items():
                outs.setdefault(key, {})[k] = v
            if verbose and k % 20 == 0:
                print(f"  bin {k}  {time.time() - t0:7.1f} s", flush=True)
        S_acc, M_acc = carry[0], carry[1]
        emis = jnp.moveaxis(S_acc[self.nT:], 1, 0)        # (nE, nSink, N)
        if self.nA:
            nf = jax.vmap(self._synth, in_axes=1)(M_acc[self.nT:])  # (nA, nSk, N)
            emis = emis.at[:self.nA].add(nf)
        res = {"emis": emis}
        for key, d in outs.items():
            ex = next(iter(d.values()))
            zero = jnp.zeros_like(ex)
            res[key] = jnp.stack([d.get(k, zero) for k in range(self.nE)])
        return res
