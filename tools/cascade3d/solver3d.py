"""Azimuthal orders and the guarded tangent of the C1 scheme (``cascade3d`` C2, WP 2.2).

Design: ``DESIGN_C2.md`` sections 1.3 to 1.6 and 8; conditions: ``GATES.md``, "C2,
dated entries", 2026-11-23.  The flux with the cutoff is the C1 solve minus a linear
map of the boundary deficit, Phi_cut = Phi_C1 - J[D].  J is the tangent of the C1
scheme at its own state (transmission 1), taken with guarded derivative rules, one
tangent field per (azimuthal order m, cos/sin part, boundary).

Pieces
------
:func:`assoc_legendre`, :func:`mode_weights`
    Normalised associated Legendre functions and the hat projection / synthesis of
    order m (``W^m[level, slot, l] = int hat_slot(mu) Pbar_l^m(mu) dmu``,
    ``Q^m = Pbar_l^m(mu_node)``).  The order-m collision source of a field
    ``f_m(mu) cos(m phi)`` is ``cos(m phi) sum_l a_l Pbar_l^m(mu) int Pbar_l^m f_m``
    (addition theorem); m = 0 is C1's ``(2l+1)/2 P_l int P_l f``.
:class:`Guard`, :func:`guarded_functions`
    The derivative rules: where a log form of the C1 scheme combines positive values
    whose ratio is below rho (source shape ``_logquad``, source integral ``_expj``),
    or where the four energy-stencil values span more than a factor 1/rho (loss
    interpolation ``_interp_log``), the derivative of the scheme's own linear fallback
    form is used.  Optional: the Simpson correction of the loss step (geometric and
    logarithmic means).  The primal values are those of ``solver.py``, unchanged.
:class:`ModeSolver`
    ``solver.Solver`` with (a) a top intensity per chain (``params["phi_top_c"]``,
    (nS, nE, Nl)), (b) the projection and synthesis taken from ``params["WPm"]``,
    ``params["QPm"]`` in the tangent (the primal always uses C1's), (c) the guarded
    rules, (d) the change of the last same-bin sweep as an output.  Without
    ``phi_top_c`` and outside a tangent it is the C1 scheme; ``ModeSolver.solve``
    is ``Solver.solve`` (equal to it to rounding, tested).
:class:`TangentSolver`
    One primal march (the C1 state) and a batch of tangents (``jax.jvp`` of each
    bin's step, ``vmap`` over the tangents, each with its own order m and per-chain
    top tangent).  Outputs the tangent emissivity, the tangent ground fluxes of the
    transported species and the tangent sweep residual per bin, and the primal ones.
:func:`sink_tangent`
    The guarded tangent of ``assemble.sink_intensity`` (neutrino line-of-sight
    integral) at the base emissivity.

The kernel arrays are not differentiated here (C4); the tangents are with respect to
the boundary only.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

import kernelset as KS
import solver as S

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# associated Legendre functions and the order-m hat projection
# ---------------------------------------------------------------------------
def assoc_legendre(x, L, m):
    """Pbar_l^m(x), l = 0..L, normalised: int_{-1}^{1} Pbar_l^m Pbar_l'^m = delta.

    No Condon-Shortley phase (it cancels in projection times synthesis).  Rows l < m
    are zero.  Returns an array (L+1,) + x.shape.
    """
    x = np.asarray(x, float)
    out = np.zeros((L + 1,) + x.shape)
    if m > L:
        return out
    s2 = np.clip(1.0 - x * x, 0.0, None)
    fac = np.prod([(2.0 * i - 1.0) / (2.0 * i) for i in range(1, m + 1)]) if m else 1.0
    pmm = np.sqrt(0.5 * (2 * m + 1) * fac) * s2 ** (0.5 * m)
    out[m] = pmm
    if m + 1 <= L:
        out[m + 1] = x * np.sqrt(2 * m + 3.0) * pmm
    for ll in range(m + 2, L + 1):
        a = np.sqrt((4.0 * ll * ll - 1.0) / (ll * ll - m * m))
        b = np.sqrt(((ll - 1.0) ** 2 - m * m) / (4.0 * (ll - 1.0) ** 2 - 1.0))
        out[ll] = a * (x * out[ll - 1] - b * out[ll - 2])
    return out


def mode_weights(rays, L, m, n_theta=None):
    """Projection W^m and synthesis Q^m of order m on the levels of ``rays``.

    ``W[e, slot, l] = int hat_slot(mu) Pbar_l^m(mu) dmu`` over [-1, 1] for the
    piecewise-linear interpolant in mu between the sorted nodes of level e, constant
    beyond the extreme nodes (C1's ``_hat_weights``); ``Q[e, slot, l] =
    Pbar_l^m(mu_slot)`` (padded slots: mu = 0, as C1).  Even m: Gauss-Legendre in mu,
    exact (polynomial of degree l + 1).  Odd m: Gauss-Legendre in theta = arccos mu
    on each interval (``Pbar_l^m(cos theta) sin theta`` times the hat is smooth),
    ``n_theta`` points per interval (default L + 8).

    With this normalisation the order-m source is ``synth(a_l proj(f))``; for m = 0,
    ``W = sqrt((2l+1)/2) W_C1`` and ``Q = sqrt((2l+1)/2) P_l`` (C1:
    ``QP = (2l+1)/2 P_l``), the same product.

    Returns numpy arrays (NL, M, L+1), (NL, M, L+1).
    """
    r = rays
    nl, Ms = r.lev_node.shape
    W = np.zeros((nl, Ms, L + 1))
    mu_all = np.concatenate([r.node_mu, [0.0]])
    Q = np.moveaxis(assoc_legendre(mu_all[r.lev_node], L, m), 0, -1)
    if m % 2 == 0:
        x, wq = np.polynomial.legendre.leggauss(L // 2 + 3)
        theta = False
    else:
        x, wq = np.polynomial.legendre.leggauss(n_theta or (L + 8))
        theta = True
    t, wt = 0.5 * (x + 1.0), 0.5 * wq

    def integ(lo, hi, left):
        """int over [lo, hi] (mu) of the hat that is 1 at lo (left) or at hi."""
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        if not theta:
            d = hi - lo
            pts = lo[:, None] + d[:, None] * t[None]
            P = assoc_legendre(pts, L, m)                   # (L+1, n, q)
            hat = (1.0 - t) if left else t
            return np.einsum("lnq,nq->nl", P, d[:, None] * wt[None] * hat[None])
        a, b = np.arccos(np.clip(hi, -1, 1)), np.arccos(np.clip(lo, -1, 1))
        th = a[:, None] + (b - a)[:, None] * t[None]       # theta from hi to lo
        mu = np.cos(th)
        P = assoc_legendre(mu, L, m)
        d = hi - lo
        dd = np.where(d > 0, d, 1.0)[:, None]
        hat = (hi[:, None] - mu) / dd if left else (mu - lo[:, None]) / dd
        w = (b - a)[:, None] * wt[None] * np.sin(th) * hat
        return np.einsum("lnq,nq->nl", P, w)

    def integ_const(lo, hi):
        lo, hi = np.asarray([lo]), np.asarray([hi])
        return integ(lo, hi, True)[0] + integ(lo, hi, False)[0]

    for e in range(nl):
        valid = r.lev_node[e] < r.N
        mu = mu_all[r.lev_node[e][valid]]
        n = mu.size
        We = np.zeros((n, L + 1))
        if n > 1:
            a, b = mu[:-1], mu[1:]
            We[:-1] += integ(a, b, True)
            We[1:] += integ(a, b, False)
        for idx, lo, hi in ((0, -1.0, mu[0]), (n - 1, mu[-1], 1.0)):
            if hi - lo > 0:
                We[idx] += integ_const(lo, hi)
        W[e, :n] = We
    return W, Q


# ---------------------------------------------------------------------------
# guarded derivative rules
# ---------------------------------------------------------------------------
class Guard:
    """Thresholds of the guarded derivative rules (None: exact derivative).

    Parameters
    ----------
    shape : float or None
        rho of the source shape between nodes (``_logquad``).
    integral : float or None
        rho of the sub-interval source integral (``_expj``).
    loss : float or None
        rho of the energy interpolation of the loss species (``_interp_log``):
        guarded where the four stencil values span more than 1/rho.
    simpson : float or None
        rho of the Simpson correction of the loss step (geometric mean of the
        midpoint, logarithmic mean of the ends); None: not guarded.
    """

    def __init__(self, shape=0.1, integral=0.1, loss=0.1, simpson=None):
        self.shape, self.integral, self.loss, self.simpson = shape, integral, loss, \
            simpson

    def tag(self):
        def f(v):
            return "x" if v is None else f"{v:g}"
        return (f"s{f(self.shape)}_i{f(self.integral)}_l{f(self.loss)}"
                f"_m{f(self.simpson)}")

    def as_dict(self):
        return {"shape": self.shape, "integral": self.integral, "loss": self.loss,
                "simpson": self.simpson}

    @classmethod
    def parse(cls, text):
        """"shape=0.1,integral=0.1,loss=0.1,simpson=none" -> Guard."""
        kw = {}
        for item in text.split(","):
            k, v = item.split("=")
            kw[k.strip()] = None if v.strip().lower() in ("none", "x", "") else float(v)
        return cls(**kw)


#: the prototype's setting (DESIGN_C2.md 1.6; first setting of G2.7)
GUARD_PROTOTYPE = Guard(0.1, 0.1, 0.1, None)
#: no guard anywhere: the exact derivative of the C1 scheme
GUARD_NONE = Guard(None, None, None, None)


def _c1_gmean(a, b):
    """C1's midpoint value of the Simpson rule (``solver._loss_transport``)."""
    posq = (a > 0) & (b > 0)
    lgm = 0.5 * (jnp.log(jnp.where(posq, a, 1.0)) + jnp.log(jnp.where(posq, b, 1.0)))
    return jnp.where(posq, jnp.exp(lgm), 0.5 * (a + b))


def _c1_logmean(s0, s1):
    """C1's thin-limit mean of the end values (``solver._loss_transport``)."""
    posm = (s0 > 0) & (s1 > 0)
    l0 = jnp.log(jnp.where(posm, s0, 1.0))
    l1 = jnp.log(jnp.where(posm, s1, 1.0))
    return jnp.where(posm, s0 * S._expm1_over(l1 - l0), 0.5 * (s0 + s1))


def guarded_functions(guard):
    """``{"logquad", "expj", "interp_log", "gmean", "logmean"}`` with the derivative
    rules of ``guard`` (primal: the C1 functions).  A piece whose rho is None is the
    C1 function itself."""
    lq, ej, il = S._logquad, S._expj, S._interp_log
    out = {"logquad": lq, "expj": ej, "interp_log": il, "gmean": _c1_gmean,
           "logmean": _c1_logmean}

    if guard.shape is not None:
        rho = float(guard.shape)

        @jax.custom_jvp
        def logquad(Aa, Ab, Ac, uc, u, has_c):
            return lq(Aa, Ab, Ac, uc, u, has_c)

        @logquad.defjvp
        def _lq_jvp(primals, tangents):
            Aa, Ab, Ac, uc, u, has_c = primals
            dAa, dAb, dAc = tangents[:3]
            y, dy = jax.jvp(lambda a, b, c: lq(a, b, c, uc, u, has_c),
                            (Aa, Ab, Ac), (dAa, dAb, dAc))
            dl = dAa + u * (dAb - dAa)
            c_ok = jnp.where(has_c, Ac, jnp.maximum(Aa, Ab))
            mx = jnp.maximum(jnp.maximum(Aa, Ab), c_ok)
            mn = jnp.minimum(jnp.minimum(Aa, Ab), c_ok)
            good = (mn > 0) & (mn >= rho * mx)
            return y, jnp.where(good, dy, dl)

        out["logquad"] = logquad

    if guard.integral is not None:
        rho_i = float(guard.integral)

        def lin_e(S0, S1, r0, r1, x, ds):
            p0, p1 = S._phi01(x)
            return ds * (S0 * r0 * p0 + S1 * r1 * p1)

        @jax.custom_jvp
        def expj(S0, S1, r0, r1, x, ds):
            return ej(S0, S1, r0, r1, x, ds)

        @expj.defjvp
        def _ej_jvp(primals, tangents):
            y, dy = jax.jvp(ej, primals, tangents)
            _, dl = jax.jvp(lin_e, primals, tangents)
            S0, S1, r0, r1, x, ds = primals
            a, b = S0 * r0, S1 * r1
            good = (a > 0) & (b > 0) & (jnp.minimum(a, b) >= rho_i * jnp.maximum(a, b))
            return y, jnp.where(good, dy, dl)

        out["expj"] = expj

    if guard.loss is not None:
        lr = float(np.log(1.0 / guard.loss))

        @jax.custom_jvp
        def interp_log(vals, wl, wpre):
            return il(vals, wl, wpre)

        @interp_log.defjvp
        def _il_jvp(primals, tangents):
            vals, wl, wpre = primals
            y, dy = jax.jvp(il, primals, tangents)
            dl = jnp.sum(wpre * tangents[0], axis=-1)
            pos = jnp.all(vals > 0, axis=-1)
            lv = jnp.log(jnp.where(vals > 0, vals, 1.0))
            span = jnp.max(lv, axis=-1) - jnp.min(lv, axis=-1)
            good = pos & (span <= lr)
            return y, jnp.where(good, dy, dl)

        out["interp_log"] = interp_log

    if guard.simpson is not None:
        rho_s = float(guard.simpson)

        @jax.custom_jvp
        def gmean(a, b):
            return _c1_gmean(a, b)

        @gmean.defjvp
        def _gm_jvp(primals, tangents):
            a, b = primals
            y, dy = jax.jvp(_c1_gmean, primals, tangents)
            dl = 0.5 * (tangents[0] + tangents[1])
            good = (a > 0) & (b > 0) & (jnp.minimum(a, b) >= rho_s * jnp.maximum(a, b))
            return y, jnp.where(good, dy, dl)

        @jax.custom_jvp
        def logmean(s0, s1):
            return _c1_logmean(s0, s1)

        @logmean.defjvp
        def _lm_jvp(primals, tangents):
            s0, s1 = primals
            y, dy = jax.jvp(_c1_logmean, primals, tangents)
            dl = 0.5 * (tangents[0] + tangents[1])
            good = (s0 > 0) & (s1 > 0) & (
                jnp.minimum(s0, s1) >= rho_s * jnp.maximum(s0, s1))
            return y, jnp.where(good, dy, dl)

        out["gmean"], out["logmean"] = gmean, logmean
    return out


# ---------------------------------------------------------------------------
# projection / synthesis with the order of the tangent
# ---------------------------------------------------------------------------
@jax.custom_jvp
def _modal_proj(Fg, W0, Wm):
    """Moments of gathered node values Fg (x, NL, M): primal with W0 (C1), tangent
    with Wm (the order of the tangent)."""
    return jnp.einsum("xqm,qml->xql", Fg, W0)


@_modal_proj.defjvp
def _modal_proj_jvp(primals, tangents):
    Fg, W0, Wm = primals
    return (jnp.einsum("xqm,qml->xql", Fg, W0),
            jnp.einsum("xqm,qml->xql", tangents[0], Wm))


@jax.custom_jvp
def _modal_synth(M, Q0, Qm):
    """Level synthesis M (x, NL, L+1) -> (x, NL, M): primal Q0 (C1), tangent Qm."""
    return jnp.einsum("xql,qml->xqm", M, Q0)


@_modal_synth.defjvp
def _modal_synth_jvp(primals, tangents):
    M, Q0, Qm = primals
    return (jnp.einsum("xql,qml->xqm", M, Q0),
            jnp.einsum("xql,qml->xqm", tangents[0], Qm))


# ---------------------------------------------------------------------------
# the C1 step with per-chain tops, modal operators and guarded rules
# ---------------------------------------------------------------------------
class ModeSolver(S.Solver):
    """``solver.Solver`` extended for C2 (see the module docstring).

    Parameters
    ----------
    rays, ks, **kw
        As ``solver.Solver``.
    guard : Guard
        Derivative rules (default :data:`GUARD_PROTOTYPE`); no effect on values.
    """

    def __init__(self, rays, ks, guard=GUARD_PROTOTYPE, **kw):
        self.guard = guard
        self._gf = guarded_functions(guard)
        self._want_dlast = False
        super().__init__(rays, ks, **kw)
        self.geo["pt_line"] = jnp.asarray(rays.pt_line)

    # -------------------------------------------------------- modal operators
    def _proj_p(self, F, prm):
        g = self.geo
        Fp = jnp.concatenate([F, jnp.zeros((F.shape[0], 1))], axis=1)
        Fg = Fp[:, g["lev_node"]]
        if prm is not None and "WPm" in prm:
            return _modal_proj(Fg, g["WP"], prm["WPm"])
        return jnp.einsum("xqm,qml->xql", Fg, g["WP"])

    def _synth_p(self, M, prm):
        g = self.geo
        if prm is not None and "QPm" in prm:
            S_lev = _modal_synth(M, g["QP"], prm["QPm"])
        else:
            S_lev = jnp.einsum("xql,qml->xqm", M, g["QP"])
        return S_lev[:, g["node_level"], g["node_slot"]]

    # -------------------------------------------------------- transport pieces
    def _chain_scan(self, T, Q, f0):
        """As C1, with ``f0`` (h,) or per chain (h, Nl)."""
        g = self.geo
        Tc = T[:, g["chain_seg"]]
        Qc = Q[:, g["chain_seg"]]

        def body(f, tq):
            t, q = tq
            f = t * f + q
            return f, f

        if f0.ndim == 1:
            f0b = jnp.broadcast_to(f0[:, None], Tc.shape[:2])
        else:
            f0b = f0
        _, ys = jax.lax.scan(body, f0b, (jnp.moveaxis(Tc, 2, 0),
                                         jnp.moveaxis(Qc, 2, 0)))
        fch = jnp.concatenate([f0b[None], ys], axis=0)
        return fch[g["node_pos"], :, g["node_line"]].T

    def _seg_source(self, S_, w):
        """C1's ``_seg_source`` with the guarded rules."""
        g = self.geo
        Aa = S_[:, g["seg_a"]] / g["rho_a"][None]
        Ab = S_[:, g["seg_b"]] / g["rho_b"][None]
        Ac = S_[:, g["seg_c"]] / g["rho_c"][None]
        Aq = self._gf["logquad"](Aa[..., None], Ab[..., None], Ac[..., None],
                                 g["seg_uc"][None, :, None], g["seg_u"][None],
                                 g["seg_hasc"][None, :, None])
        Sq = g["seg_rho"][None] * Aq
        ds = g["seg_ds"][None]
        contrib = self._gf["expj"](Sq[..., :-1], Sq[..., 1:], w["r0"], w["r1"],
                                   w["x"], ds)
        Q = jnp.sum(w["after"] * contrib, axis=-1)
        return jnp.concatenate([Q, jnp.zeros((S_.shape[0], 1))], axis=1)

    def _loss_transport(self, k, S_L, lam, dec, ftop, fwin, swin, prm):
        """C1's ``_loss_transport`` with the guarded rules and per-chain tops."""
        gf = self._gf
        g = self.geo
        W, nE = self.W, self.nE
        nL = S_L.shape[0]
        sa = g["seg_a"][g["pt_seg"]]
        sb = g["seg_b"][g["pt_seg"]]
        sc = g["seg_c"][g["pt_seg"]]
        Aa = S_L[:, sa] / g["rho_a"][g["pt_seg"]][None]
        Ab = S_L[:, sb] / g["rho_b"][g["pt_seg"]][None]
        Ac = S_L[:, sc] / g["rho_c"][g["pt_seg"]][None]
        Sp = g["pt_rho"][None] * gf["logquad"](
            Aa, Ab, Ac, g["seg_uc"][g["pt_seg"]][None], g["pt_u"][None],
            g["seg_hasc"][g["pt_seg"]][None])
        Sp = jnp.where(g["pt_isnode"][None], S_L[:, g["pt_node"]], Sp)
        Sp = jnp.concatenate([Sp, jnp.zeros((nL, 1))], axis=1)
        slot = k % W
        swin = swin.at[slot].set(Sp)
        Rk = jnp.take(g["Rb"], k, axis=1)
        dX = g["pt_dX"]
        z, jj, wl, wpre, slots = self._stencil(k, Rk, dX)
        fl = jnp.broadcast_to(jnp.floor(z), (nL,) + z.shape[1:])
        ib = jnp.clip(fl.astype(jnp.int32) + k, 0, nE - 1)
        ib1 = jnp.clip(ib + 1, 0, nE - 1)
        tz = jnp.broadcast_to(z, fl.shape) - fl
        Lrow = prm["inv_lambda"][jnp.asarray(self.Lsp)]
        Drow = prm["inv_dec"][jnp.asarray(self.Lsp)]

        def at(row, i):
            return jax.vmap(lambda r, ii: r[ii])(row, i)

        D_star = S._loglin(at(Drow, ib), at(Drow, ib1), tz)
        L_star = S._loglin(at(Lrow, ib), at(Lrow, ib1), tz)
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
            return jnp.take(flat, (sl * nL + lidx) * P1 + pts)

        f_up = wget(fwf, slots, pidx)
        s_up = wget(swf, slots, pidx)
        cur = (jj == 0.0)
        lbb = g["lnb_bins"]
        lrow = jnp.arange(lbb.shape[0])[:, None, None]

        def bratio(jq):
            idx = k + jq.astype(jnp.int32)
            return jnp.where(g["lossy"][:, None, None],
                             jnp.exp(lbb[lrow, idx] - lbb[lrow[:, 0], k][..., None]),
                             1.0)

        br = bratio(jj)
        Sstar_r = gf["interp_log"](s_up * br, wl, wpre)
        here = jnp.arange(Sp.shape[1] - 1)[None, :, None]
        zq, jq, wlq, wpq, sl = self._stencil(k, Rk, 0.5 * dX)
        bq = bratio(jq)
        s_prev = gf["interp_log"](wget(swf, sl, pidx) * bq, wlq, wpq)
        s_here = gf["interp_log"](wget(swf, sl, here) * bq, wlq, wpq)
        s_mid = gf["gmean"](s_prev, s_here)
        quad = (Sstar_r + 4.0 * s_mid + Sp[:, :-1]) / 6.0
        s0, s1 = Sstar_r, Sp[:, :-1]
        lmean = gf["logmean"](s0, s1)
        corr = jnp.where(lmean > 0, quad / jnp.where(lmean > 0, lmean, 1.0), 1.0)
        C1 = ex
        B0 = corr * gf["expj"](s0, s1, S._safe_ratio(sbar, sig_star),
                               S._safe_ratio(sbar, sig_m), x, ds[None])
        brk = g["pt_break"][None]
        st = g["pt_start"][None]
        C1 = jnp.where(brk | st, 0.0, C1)
        if ftop.ndim == 1:
            ftp = ftop[:, None]
        else:
            ftp = ftop[:, g["pt_line"]]
        B0 = jnp.where(brk, 0.0, jnp.where(st, ftp, B0))

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
        """C1's ``_point_scan_nl`` with the guarded energy interpolation."""
        g = self.geo
        nL = xs["C1"].shape[0]
        out = jnp.zeros(xs["C1"].shape)
        il = self._gf["interp_log"]

        def body(f, x):
            vals = jnp.where(x["cur"], f[..., None], x["fup"])
            fs = il(vals, x["wl"], x["wpre"])
            f = x["C1"] * fs + x["B0"]
            return f, f

        for idx in g["buckets"]:
            xb = {k: jnp.moveaxis(v[:, idx], 2, 0) for k, v in xs.items()}
            f0 = jnp.zeros((nL, idx.shape[0]))
            _, ys = jax.lax.scan(body, f0, xb)
            out = out.at[:, idx].set(jnp.moveaxis(ys, 0, 2))
        return out

    # ------------------------------------------------------------------ step
    def _make_step(self, angular, up=False):
        """C1's step (``solver.Solver._make_step``) with: per-chain tops
        (``prm["phi_top_c"]``), the modal operators of the tangent, and, if
        ``self._want_dlast``, the change of the last same-bin sweep as output
        ``"dlast"`` (hadrons, (nH, N))."""
        nH, nL, nT, nC = self.nH, self.nL, self.nT, self.nC
        nE, nA, L = self.nE, self.nA, self.L
        Tidx = jnp.asarray(np.array(self.T, dtype=np.int64))
        ch_p, ch_c = jnp.asarray(self.ch_p), jnp.asarray(self.ch_c)
        ch_keep = jnp.asarray(self.ch_keep)
        is_int = jnp.asarray(self.ch_kind == KS.KIND_INT)
        ang_sel, ang_kept = jnp.asarray(self.ang_sel), jnp.asarray(self.ang_kept)
        n_sweeps = self.n_sweeps
        sink_child = jnp.asarray(np.arange(nC) >= nT)
        up_reach = self.up_reach
        int_par = jnp.asarray(np.unique(self.ch_p[self.ch_kind == KS.KIND_INT]))
        dec_par = jnp.asarray(np.unique(self.ch_p[self.ch_kind == KS.KIND_DEC]))
        c_used = np.unique(self.ch_c)
        all_children = c_used.size == nC
        c_idx = jnp.asarray(c_used)
        c_up = np.unique(self.ch_c[self.ch_c < nT])
        c_up_idx = jnp.asarray(c_up)
        midx = jnp.arange(nE)
        want_dlast = bool(self._want_dlast)

        def dense(vals_ch, mask_int):
            v = jnp.where(mask_int.reshape((-1,) + (1,) * (vals_ch.ndim - 1)),
                          vals_ch, 0.0)
            out = jnp.zeros((nC, nT) + vals_ch.shape[1:])
            return out.at[ch_c, ch_p].add(v)

        K_up = self.K_up
        KA = min(K_up, nA)

        def step(k, carry, prm, top_on):
            g = self.geo
            S_acc, M_acc, fwin, swin, S_up, M_up = carry
            proj = (lambda F: self._proj_p(F, prm))      # noqa: E731
            synth = (lambda M: self._synth_p(M, prm))    # noqa: E731
            Y = prm["Y"][ch_keep]
            Yk = jnp.take(Y, k, axis=2)
            if angular:
                gk = jnp.take(prm["g"][ang_sel], k, axis=2)
                ak = jnp.take(prm["a"][ang_sel], k, axis=3)
                Yf = Yk.at[ang_kept, :nA].multiply(gk)
                Ynf = (Yk[ang_kept, None, :nA] * (1.0 - gk)[:, None, :] * ak)
                nf_full = jnp.zeros((Yk.shape[0], L + 1, nA)).at[ang_kept].set(Ynf)
                G_int = dense(nf_full, is_int)
                G_dec = dense(nf_full, ~is_int)
            else:
                Yf = Yk
            Y_int = dense(Yf, is_int)
            Y_dec = dense(Yf, ~is_int)
            lam = jnp.take(prm["inv_lambda"][Tidx], k, axis=1)
            dec = jnp.take(prm["inv_dec"][Tidx], k, axis=1)
            if "phi_top_c" in prm:
                ftop = jnp.take(prm["phi_top_c"], k, axis=1)[Tidx] * top_on
            else:
                ftop = jnp.take(prm["phi_top"][Tidx], k, axis=1) * top_on
            act = jnp.take(g["active_T"], k, axis=1)
            rho_n = g["node_rho"]
            Cint = jnp.take(Y_int[:nT], k, axis=2)
            Cdec = jnp.take(Y_dec[:nT], k, axis=2)
            eye = jnp.eye(nT)
            self_int = jnp.diagonal(Cint)
            self_dec = jnp.diagonal(Cdec)
            Cint_o = Cint * (1.0 - eye)
            Cdec_o = Cdec * (1.0 - eye)
            if angular:
                kk = jnp.minimum(k, nA - 1)
                Gs_int = jnp.take(G_int[:nT], kk, axis=3)
                Gs_dec = jnp.take(G_dec[:nT], kk, axis=3)
                lev_rho = g["lev_rho"]

            def same_bin(F, rows):
                Ai = lam[:, None] * F * rho_n[None]
                Ad = dec[:, None] * F
                S_ = Cint_o[rows] @ Ai + Cdec_o[rows] @ Ad
                if angular:
                    Fl = proj(F)
                    Ri = lam[:, None, None] * Fl * lev_rho[None, :, None]
                    Rd = dec[:, None, None] * Fl
                    Ms = (jnp.einsum("cpl,pql->cql", Gs_int[rows], Ri)
                          + jnp.einsum("cpl,pql->cql", Gs_dec[rows], Rd))
                    S_ = S_ + synth(Ms)
                return S_

            S_fix = jnp.take(S_acc, k, axis=1)
            if angular:
                Mk = jnp.take(M_acc, jnp.minimum(k, nA - 1), axis=1)
                S_fix = S_fix + synth(Mk)
            hs = jnp.arange(nH)
            Lam_e = lam[:nH] * (1.0 - self_int[:nH])
            De_e = dec[:nH] * (1.0 - self_dec[:nH])
            Tseg, c = self._seg_weights(Lam_e, De_e)
            Qfix = self._seg_source(S_fix[:nH], c)
            al, be = self._seg_linear(c)
            F_H = self._chain_scan(Tseg, Qfix, ftop[:nH]) * act[:nH, None]
            zL = jnp.zeros((nL, F_H.shape[1]))

            def sweep(i, st):
                F_H, _, _ = st
                Fall = jnp.concatenate([F_H, zL], axis=0)
                Ss = same_bin(Fall, hs)
                Q = Qfix + jnp.concatenate(
                    [al[:, :-1] * Ss[:, g["seg_a"]] + be[:, :-1] * Ss[:, g["seg_b"]],
                     jnp.zeros((nH, 1))], axis=1)
                Fn = self._chain_scan(Tseg, Q, ftop[:nH]) * act[:nH, None]
                scale = jnp.max(jnp.abs(Fn), axis=1)
                dmax = jnp.max(jnp.abs(Fn - F_H), axis=1)
                res = jnp.max(S._safe_ratio(dmax, scale) * (scale > 0))
                return Fn, res, (Fn - F_H if want_dlast else jnp.zeros((1,)))

            dl0 = jnp.zeros_like(F_H) if want_dlast else jnp.zeros((1,))
            if n_sweeps > 0 and nH > 0:
                F_H, res, dlast = jax.lax.fori_loop(0, n_sweeps, sweep,
                                                    (F_H, 0.0, dl0))
            else:
                res, dlast = jnp.asarray(0.0), dl0
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
            mask = jnp.where(sink_child[:, None], True, midx[None] < k)
            Ai = lam[:, None] * F_T * rho_n[None]
            Ad = dec[:, None] * F_T
            Ycat = jnp.concatenate([Y_int[:, int_par], Y_dec[:, dec_par]],
                                   axis=1) * mask[:, None, :]
            Acat = jnp.concatenate([Ai[int_par], Ad[dec_par]], axis=0)
            if not all_children:
                Ycat = Ycat[c_idx]
            for m0 in range(0, nE, S._PUSH_BLOCK):
                m1 = min(m0 + S._PUSH_BLOCK, nE)

                def add(Sa, m0=m0, m1=m1):
                    blk = jnp.einsum("cpm,pn->cmn", Ycat[:, :, m0:m1], Acat)
                    if all_children:
                        return jax.lax.dynamic_update_slice_in_dim(
                            Sa, Sa[:, m0:m1] + blk, m0, axis=1)
                    return Sa.at[c_idx, m0:m1].add(blk)

                S_acc = jax.lax.cond(m0 <= k + up_reach, add, lambda Sa: Sa, S_acc)
            if angular:
                Fl = proj(F_T)
                Ri = lam[:, None, None] * Fl * g["lev_rho"][None, :, None]
                Rd = dec[:, None, None] * Fl
                mA = mask[:, :nA]
                Gi = G_int * mA[:, None, None, :]
                Gd = G_dec * mA[:, None, None, :]
                M_acc = M_acc + (jnp.einsum("cplm,pql->cmql", Gi, Ri)
                                 + jnp.einsum("cplm,pql->cmql", Gd, Rd))
            if up:
                mup = (midx[None, :K_up] > k).astype(Ai.dtype)
                Yu = jnp.concatenate([Y_int[c_up_idx][:, int_par, :K_up],
                                      Y_dec[c_up_idx][:, dec_par, :K_up]],
                                     axis=1) * mup[:, None, :]
                S_up = S_up.at[c_up_idx].add(jnp.einsum("cpm,pn->cmn", Yu, Acat))
                if angular and KA > 0:
                    mu_a = mup[:, :KA]
                    Gi = G_int[:nT, :, :, :KA] * mu_a[:, None, None, :]
                    Gd = G_dec[:nT, :, :, :KA] * mu_a[:, None, None, :]
                    M_up = M_up + (jnp.einsum("cplm,pql->cmql", Gi, Ri)
                                   + jnp.einsum("cplm,pql->cmql", Gd, Rd))
            out = {"F_ground": F_T[:, self._gnode], "res": res}
            if want_dlast:
                out["dlast"] = dlast
                out["F_H"] = F_H
            if self.store_nodes:
                out["F"] = F_T
            return (S_acc, M_acc, fwin, swin, S_up, M_up), out

        return step

    def emissivity(self, carry, prm=None):
        """Sink emissivity (nE, nSink, N) of a final carry (as ``Solver.solve``)."""
        S_acc, M_acc = carry[0], carry[1]
        emis = jnp.moveaxis(S_acc[self.nT:], 1, 0)
        if self.nA:
            nf = jax.vmap(lambda M: self._synth_p(M, prm), in_axes=1)(
                M_acc[self.nT:])
            emis = emis.at[:self.nA].add(nf)
        return emis


# ---------------------------------------------------------------------------
# the tangent solver
# ---------------------------------------------------------------------------
def _exact_zero(*xs):
    return all(not bool(jnp.any(x != 0)) for x in xs)


class TangentSolver:
    """Primal march of the C1 state and a batch of guarded tangents.

    Parameters
    ----------
    rays, ks
        Geometry and kernel set.
    guard : Guard
    **kw
        ``solver.Solver`` options (``n_sweeps``, ``gamma0``, ``n_up``, ...).
    """

    def __init__(self, rays, ks, guard=GUARD_PROTOTYPE, **kw):
        self.sv = ModeSolver(rays, ks, guard=guard, **kw)
        self.sv._want_dlast = True
        self.rays, self.ks = rays, ks
        self._mw = {}
        self._tsteps = {}

    @property
    def guard(self):
        return self.sv.guard

    def modal_ops(self, m):
        """(W^m, Q^m) as device arrays; m = 0: C1's own ``WP``, ``QP``."""
        if m not in self._mw:
            if m == 0:
                self._mw[m] = (self.sv.geo["WP"], self.sv.geo["QP"])
            else:
                W, Q = mode_weights(self.rays, self.sv.L, m)
                self._mw[m] = (jnp.asarray(W), jnp.asarray(Q))
        return self._mw[m]

    def _tstep(self, ang, up):
        key = (ang, up)
        if key in self._tsteps:
            return self._tsteps[key]
        raw = self.sv._make_step(angular=ang, up=up)

        def tstep(k, carry, tcarry, prm, topc, dtopc, Wms, Qms, top_on):
            def one(tc, dt, Wm, Qm):
                def f(c, tp):
                    p = dict(prm)
                    p["phi_top_c"] = tp
                    p["WPm"], p["QPm"] = Wm, Qm
                    return raw(k, c, p, top_on)

                return jax.jvp(f, (carry, topc), (tc, dt))

            (c1, o1), (tc1, to1) = jax.vmap(one, in_axes=(0, 0, 0, 0),
                                            out_axes=(None, 0))(
                tcarry, dtopc, Wms, Qms)
            # tangent residual of the same-bin sweeps (hadrons)
            dF, dd = to1["F_H"], to1["dlast"]
            scale = jnp.max(jnp.abs(dF), axis=2)
            dmax = jnp.max(jnp.abs(dd), axis=2)
            tres = jnp.max(S._safe_ratio(dmax, scale) * (scale > 0), axis=1)
            o = {"F_ground": o1["F_ground"], "res": o1["res"]}
            to = {"F_ground": to1["F_ground"], "res": tres}
            return c1, tc1, o, to

        fn = jax.jit(tstep, donate_argnums=(1, 2))
        self._tsteps[key] = fn
        return fn

    def solve(self, dtop_c, ms, params=None, top_c=None, verbose=False):
        """March the C1 state and the tangents.

        Parameters
        ----------
        dtop_c : array (B, nS, nE, Nl)
            Tangent of the top intensity of every chain (per species of the kernel
            set and bin), one per tangent; -D for the deficit D.
        ms : sequence of int (B,)
            Azimuthal order of each tangent.
        params : dict, optional
            Kernel arrays (default the set's).
        top_c : array (nS, nE, Nl), optional
            Primal per-chain top (default ``phi_top`` on every chain: the C1 state).

        Returns
        -------
        dict
            ``emis`` (nE, nSink, N), ``F_ground`` (nE, nT, n_ground), ``res`` (nE,),
            ``up_res``: the primal (C1 scheme); ``d_emis`` (B, nE, nSink, N),
            ``d_F_ground`` (B, nE, nT, n_ground), ``d_res`` (B, nE): the tangents
            (residual: change of the last same-bin sweep of the tangent, relative);
            ``d_up_res`` (B, n_passes - 1); ``timing``.
        """
        sv = self.sv
        if params is None:
            params = self.ks.params()
        prm = {k: jnp.asarray(v) for k, v in params.items()}
        Nl = self.rays.Nl
        if top_c is None:
            top_c = jnp.broadcast_to(prm["phi_top"][:, :, None],
                                     prm["phi_top"].shape + (Nl,))
        top_c = jnp.asarray(top_c)
        dtop_c = jnp.asarray(dtop_c)
        B = dtop_c.shape[0]
        ms = [int(m) for m in ms]
        if len(ms) != B:
            raise ValueError("one order per tangent")
        Wms = jnp.stack([self.modal_ops(m)[0] for m in ms])
        Qms = jnp.stack([self.modal_ops(m)[1] for m in ms])
        carry = sv.init_carry()
        tcarry = jax.tree_util.tree_map(
            lambda a: jnp.zeros((B,) + a.shape, a.dtype), carry)
        nT, K_up = sv.nT, sv.K_up
        n_pass = 1 + (sv.n_up if K_up > 0 else 0)
        outs, touts = {}, {}
        up_res, dup_res = [], []
        t0 = time.time()
        for ipass in range(n_pass):
            if ipass == 0:
                ks_range = range(sv.k_top, -1, -1)
                top_on = jnp.asarray(1.0)
            else:
                S_acc, M_acc, fwin, swin, S_up, M_up = carry
                tS_acc, tM_acc, tfwin, tswin, tS_up, tM_up = tcarry
                if _exact_zero(S_up, M_up, tS_up, tM_up):
                    up_res.extend([0.0] * (n_pass - ipass))
                    dup_res.extend([np.zeros(B)] * (n_pass - ipass))
                    break
                emis_prev = S_acc[nT:]
                temis_prev = tS_acc[:, nT:]
                S_acc = S_acc.at[:nT, :K_up].set(S_up[:, :K_up])
                tS_acc = tS_acc.at[:, :nT, :K_up].set(tS_up[:, :, :K_up])
                KA = min(K_up, sv.nA)
                if KA > 0:
                    M_acc = M_acc.at[:nT, :KA].set(M_up)
                    tM_acc = tM_acc.at[:, :nT, :KA].set(tM_up)
                carry = (S_acc, M_acc, jnp.zeros_like(fwin), jnp.zeros_like(swin),
                         jnp.zeros_like(S_up), jnp.zeros_like(M_up))
                tcarry = (tS_acc, tM_acc, jnp.zeros_like(tfwin),
                          jnp.zeros_like(tswin), jnp.zeros_like(tS_up),
                          jnp.zeros_like(tM_up))
                ks_range = range(sv.k_up, -1, -1)
                top_on = jnp.asarray(0.0)
            fg_pass, tfg_pass = {}, {}
            for k in ks_range:
                ang = bool(sv.nA and k < sv.nA)
                upf = bool(K_up > 0 and k <= sv.k_up)
                fn = self._tstep(ang, upf)
                carry, tcarry, o, to = fn(jnp.int32(k), carry, tcarry, prm, top_c,
                                          dtop_c, Wms, Qms, top_on)
                for key in ("F_ground", "res"):
                    for d, v in ((outs, o[key]), (touts, to[key])):
                        if ipass == 0:
                            d.setdefault(key, {})[k] = v
                        elif key == "F_ground":
                            d[key][k] = d[key][k] + v
                        else:
                            d[key][k] = jnp.maximum(d[key][k], v)
                fg_pass[k] = o["F_ground"]
                tfg_pass[k] = to["F_ground"]
                if verbose and k % 10 == 0:
                    print(f"  pass {ipass} bin {k}  {time.time() - t0:7.1f} s",
                          flush=True)
            if ipass > 0:
                r_e = S._rel_change(carry[0][nT:] - emis_prev, carry[0][nT:])
                fgp = jnp.stack([fg_pass[k] for k in ks_range])
                fgt = jnp.stack([outs["F_ground"][k] for k in ks_range])
                up_res.append(float(jnp.maximum(r_e, S._rel_change(fgp, fgt))))
                dr = []
                for b in range(B):
                    re_b = S._rel_change(tcarry[0][b, nT:] - temis_prev[b],
                                         tcarry[0][b, nT:])
                    fp = jnp.stack([tfg_pass[k][b] for k in ks_range])
                    ft = jnp.stack([touts["F_ground"][k][b] for k in ks_range])
                    dr.append(float(jnp.maximum(re_b, S._rel_change(fp, ft))))
                dup_res.append(np.array(dr))
        emis = sv.emissivity(carry)
        d_emis = jnp.stack([self._emis_mode(tcarry, b, ms[b]) for b in range(B)])
        res = {"emis": emis, "d_emis": d_emis,
               "up_res": np.array(up_res), "d_up_res": np.array(dup_res).T
               if dup_res else np.zeros((B, 0))}
        nE = sv.nE
        for key, d in outs.items():
            zero = jnp.zeros_like(next(iter(d.values())))
            res[key] = jnp.stack([d.get(k, zero) for k in range(nE)])
        for key, d in touts.items():
            zero = jnp.zeros_like(next(iter(d.values())))
            res["d_" + key] = jnp.moveaxis(
                jnp.stack([d.get(k, zero) for k in range(nE)]), 0, 1)
        res["timing"] = {"wall_s": time.time() - t0}
        return res

    def _emis_mode(self, tcarry, b, m):
        """Tangent emissivity of tangent ``b`` (order ``m``): synthesis with Q^m."""
        sv = self.sv
        S_acc, M_acc = tcarry[0][b], tcarry[1][b]
        emis = jnp.moveaxis(S_acc[sv.nT:], 1, 0)
        if sv.nA:
            g = sv.geo
            Qm = self.modal_ops(m)[1]

            def syn(M):
                S_lev = jnp.einsum("xql,qml->xqm", M, Qm)
                return S_lev[:, g["node_level"], g["node_slot"]]

            nf = jax.vmap(syn, in_axes=1)(M_acc[sv.nT:])
            emis = emis.at[:sv.nA].add(nf)
        return emis


# ---------------------------------------------------------------------------
# neutrino line-of-sight integral and its tangent
# ---------------------------------------------------------------------------
def sink_tangent(solver, emis, d_emis, chunk=48):
    """Sink intensity at every node and its guarded tangent.

    ``assemble.sink_intensity`` with the ``_seg_source`` of ``solver`` (a
    :class:`ModeSolver`: guarded rules), differentiated in the direction
    ``d_emis``.  Parameters: ``emis`` (nE, nSink, N); ``d_emis`` (B, nE, nSink, N).
    Returns ``(I (nE, nSink, N), dI (B, nE, nSink, N))``.
    """
    emis = jnp.asarray(emis)
    d_emis = jnp.asarray(d_emis)
    nE, nSk, N = emis.shape
    B = d_emis.shape[0]
    flat = jnp.reshape(emis, (nE * nSk, N))
    dflat = jnp.reshape(d_emis, (B, nE * nSk, N))
    z = jnp.zeros(chunk)
    T, c = solver._seg_weights(z, z)

    def one(Sv):
        Q = solver._seg_source(Sv, c)
        return solver._chain_scan(T, Q, jnp.zeros(Sv.shape[0]))

    @jax.jit
    def both(Sv, dS):
        y, dy = jax.vmap(lambda d: jax.jvp(one, (Sv,), (d,)),
                         out_axes=(None, 0))(dS)
        return y, dy

    parts, dparts = [], []
    for i in range(0, flat.shape[0], chunk):
        blk = flat[i:i + chunk]
        dblk = dflat[:, i:i + chunk]
        n = blk.shape[0]
        if n < chunk:
            blk = jnp.concatenate([blk, jnp.zeros((chunk - n, N))], axis=0)
            dblk = jnp.concatenate([dblk, jnp.zeros((B, chunk - n, N))], axis=1)
        y, dy = both(blk, dblk)
        parts.append(y[:n])
        dparts.append(dy[:, :n])
    I = jnp.reshape(jnp.concatenate(parts, axis=0), (nE, nSk, N))
    dI = jnp.reshape(jnp.concatenate(dparts, axis=1), (B, nE, nSk, N))
    return I, dI


def ground_table_nodes(rays):
    """Nodes and cos Z of the ground intensity table of ``assemble.ground_sink_table``
    (the ground lines plus the horizontal node), in ascending cos Z."""
    nodes = list(rays.ground_node)
    mu0 = list(rays.ground_mu0)
    hz = np.flatnonzero((rays.node_level == 0) & (rays.node_branch == 0))
    if hz.size:
        nodes.append(int(hz[0]))
        mu0.append(0.0)
    nodes, mu0 = np.array(nodes), np.array(mu0)
    o = np.argsort(mu0)
    return nodes[o], mu0[o]
