"""Straight lines in a spherical shell: nodes, chains, segments (WP 1.1).

Numpy only, computed once per (atmosphere, levels, line family, sampling).  See
``DESIGN.md`` section 2 for the definitions.  A line of impact parameter ``b`` is
parametrised by ``t``, the signed distance from its point of closest approach, in
the direction of motion: position radius ``r(t) = sqrt(b^2 + t^2)``, altitude
``h = r - R``, local direction cosine ``mu = -t / r`` (``mu > 0`` down-going).

Tables (``N`` nodes, ``Ns`` segments, ``Nl`` lines; index ``N`` / ``Ns`` is a dummy
used for padding):

``node_level, node_mu, node_h, node_rho, node_line, node_pos, node_branch``
    level index, cos(angle to downward vertical), altitude [cm], density [g/cm^3],
    line, position in the chain, +1 inward / 0 tangent / -1 outward.
``chain_node (Nl, Lmax)``, ``chain_seg (Nl, Lmax-1)``
    nodes and segments of every line in the order of motion.
``seg_a, seg_b, seg_break``
    upstream and downstream node of each segment, ground break flag (transmission 0).
``seg_ds, seg_dX (Ns, n_sub)``; ``seg_rho, seg_u, seg_h, seg_mu (Ns, n_sub + 1)``
    sub-interval lengths [cm] and grammages [g/cm^2]; density, grammage fraction from
    the upstream node, altitude and mu at the sub-points (0 and n_sub are the nodes).
``pt_*``
    loss points (sub-points split so that no step exceeds ``dX_max``), see
    :meth:`Rays._build_points`.
``lev_node (NL, M)``, ``lev_w (NL, M)``
    nodes of each level (padded with ``N``) and their trapezoid weights in mu.
"""

import numpy as np

from grid import R_E_CM

_GL_N = 8


def _gl(n=_GL_N):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def line_grammage(atm, b, t0, t1, n_gl=_GL_N, r_e=R_E_CM):
    """Grammage [g/cm^2] along a line between parameters ``t0`` and ``t1``.

    Gauss-Legendre in ``t`` on each of the (broadcast) intervals; the intervals must not
    straddle a density boundary for full accuracy.
    """
    x, w = _gl(n_gl)
    t0, t1, b = np.broadcast_arrays(np.asarray(t0, float), np.asarray(t1, float),
                                    np.asarray(b, float))
    tt = t0[..., None] + (t1 - t0)[..., None] * x
    h = np.sqrt(b[..., None] ** 2 + tt ** 2) - r_e
    return np.sum(atm.density(h) * w, axis=-1) * (t1 - t0)


def slant_depth(atm, b, h, r_e=R_E_CM, levels=None, n_sub=16):
    """Grammage from the top down to altitude ``h`` along the inward branch.

    Independent of the node tables (used by the geometry tests).  Intervals are cut at
    the density boundaries and at ``levels`` if given, and each piece is split into
    ``n_sub`` Gauss-Legendre intervals.

    Parameters
    ----------
    b : float
        Impact parameter [cm].
    h : array_like
        Altitudes [cm] (``R + h >= b``).
    """
    h = np.atleast_1d(np.asarray(h, float))
    cuts = np.union1d(atm.boundaries, [] if levels is None else levels)
    cuts = cuts[(cuts > 0) & (cuts < atm.h_top)]
    out = np.zeros(h.size)
    for i, hh in enumerate(h):
        edges = np.concatenate([[hh], cuts[cuts > hh], [atm.h_top]])
        r = r_e + edges
        t = -np.sqrt(np.clip(r ** 2 - b ** 2, 0.0, None))
        tot = 0.0
        for ta, tb in zip(t[1:], t[:-1]):  # from top down: ta < tb <= 0
            ts = np.linspace(ta, tb, n_sub + 1)
            tot += np.sum(line_grammage(atm, b, ts[:-1], ts[1:], r_e=r_e))
        out[i] = tot
    return out


class Rays:
    """Node, chain and segment tables for one geometry.

    Parameters
    ----------
    atm : atmosphere
        ``grid.CorsikaAtmosphere`` or ``grid.ExponentialAtmosphere``.
    levels : array_like
        Altitude levels [cm] (``grid.make_levels``).
    family : grid.LineFamily
        Ground and tangent lines.
    n_sub : int
        Sub-intervals per segment (equal in ``t``).
    dX_max : float
        Largest grammage step of the loss points [g/cm^2].
    n_buckets : int
        Number of length buckets for the scans over loss points.
    """

    def __init__(self, atm, levels, family, n_sub=4, dX_max=10.0, n_buckets=4):
        self.atm = atm
        self.levels = np.asarray(levels, float)
        self.family = family
        self.r_e = family.r_e
        self.n_sub = int(n_sub)
        self.dX_max = float(dX_max)
        self.n_buckets = int(n_buckets)
        self._build_nodes()
        self._build_segments()
        self._build_levels()
        self._build_points()

    # ------------------------------------------------------------------ nodes
    def _build_nodes(self):
        lev, R = self.levels, self.r_e
        rl = R + lev
        nl = lev.size
        b_all = self.family.b
        ng = self.family.n_ground
        nodes = []  # (line, level, t, branch)
        chains, breaks, line_b, line_kind = [], [], [], []
        for j, b in enumerate(b_all):
            if j < ng:
                kind = 0
                ell = list(range(nl - 1, -1, -1))
                ins = [(e, -np.sqrt(rl[e] ** 2 - b ** 2), 1) for e in ell]
                outs = [(e, np.sqrt(rl[e] ** 2 - b ** 2), -1) for e in ell[::-1]]
                seq = ins + outs
                brk = len(ins) - 1  # segment index of the break
            else:
                ht = b - R
                if ht >= lev[-1] - 1.0:
                    continue
                hit = np.flatnonzero(np.abs(lev - ht) < 1.0)
                if hit.size:
                    kind = 1
                    lt = int(hit[0])
                    b = rl[lt]
                    above = list(range(nl - 1, lt, -1))
                    ins = [(e, -np.sqrt(rl[e] ** 2 - b ** 2), 1) for e in above]
                    outs = [(e, np.sqrt(rl[e] ** 2 - b ** 2), -1) for e in above[::-1]]
                    seq = ins + [(lt, 0.0, 0)] + outs
                else:
                    kind = 2
                    l0 = int(np.searchsorted(lev, ht) - 1)
                    above = list(range(nl - 1, l0, -1))
                    ins = [(e, -np.sqrt(rl[e] ** 2 - b ** 2), 1) for e in above]
                    outs = [(e, np.sqrt(rl[e] ** 2 - b ** 2), -1) for e in above[::-1]]
                    seq = ins + outs
                brk = -1
            jj = len(chains)
            idx = []
            for pos, (e, t, br) in enumerate(seq):
                idx.append(len(nodes))
                nodes.append((jj, e, t, br, pos))
            chains.append(idx)
            breaks.append(brk)
            line_b.append(b)
            line_kind.append(kind)
        arr = np.array(nodes, dtype=float)
        self.N = arr.shape[0]
        self.node_line = arr[:, 0].astype(np.int64)
        self.node_level = arr[:, 1].astype(np.int64)
        self.node_t = arr[:, 2]
        self.node_branch = arr[:, 3].astype(np.int64)
        self.node_pos = arr[:, 4].astype(np.int64)
        self.line_b = np.array(line_b)
        self.line_kind = np.array(line_kind)  # 0 ground, 1 tangent node, 2 pass
        self.Nl = len(chains)
        self.node_h = self.levels[self.node_level]
        r = np.sqrt(self.line_b[self.node_line] ** 2 + self.node_t ** 2)
        self.node_mu = np.clip(-self.node_t / r, -1.0, 1.0)
        self.node_mu[self.node_branch == 0] = 0.0
        self.node_rho = self.atm.density(self.node_h)
        self.Lmax = max(len(c) for c in chains)
        self.chain_node = np.full((self.Nl, self.Lmax), self.N, dtype=np.int64)
        for j, c in enumerate(chains):
            self.chain_node[j, :len(c)] = c
        self.chain_len = np.array([len(c) for c in chains])
        self._chains, self._breaks = chains, breaks
        self.n_ground = int(np.sum(self.line_kind == 0))
        # observation: inward ground node of each ground line
        g = np.flatnonzero(self.line_kind == 0)
        self.ground_line = g
        self.ground_mu0 = self.family.mu0.copy()
        self.ground_node = np.array([chains[j][breaks[j]] for j in g], dtype=np.int64)

    # --------------------------------------------------------------- segments
    def _build_segments(self):
        ns = self.n_sub
        sa, sb, brk, line = [], [], [], []
        self.chain_seg = np.zeros((self.Nl, self.Lmax - 1), dtype=np.int64)
        for j, c in enumerate(self._chains):
            for q in range(len(c) - 1):
                self.chain_seg[j, q] = len(sa)
                sa.append(c[q])
                sb.append(c[q + 1])
                brk.append(q == self._breaks[j])
                line.append(j)
        Ns = len(sa)
        self.Ns = Ns
        for j, c in enumerate(self._chains):
            self.chain_seg[j, len(c) - 1:] = Ns
        self.seg_a = np.array(sa, dtype=np.int64)
        self.seg_b = np.array(sb, dtype=np.int64)
        self.seg_break = np.array(brk, dtype=bool)
        self.seg_line = np.array(line, dtype=np.int64)
        b = self.line_b[self.seg_line]
        ta, tb = self.node_t[self.seg_a], self.node_t[self.seg_b]
        tb = np.where(self.seg_break, ta, tb)  # zero-length break
        frac = np.linspace(0.0, 1.0, ns + 1)
        tq = ta[:, None] + (tb - ta)[:, None] * frac[None, :]
        tq[:, 0], tq[:, -1] = ta, tb
        self.seg_t = tq
        self.seg_ds = np.diff(tq, axis=1)
        self.seg_dX = line_grammage(self.atm, b[:, None], tq[:, :-1], tq[:, 1:],
                                    r_e=self.r_e)
        r = np.sqrt(b[:, None] ** 2 + tq ** 2)
        self.seg_h = r - self.r_e
        self.seg_h[:, 0] = self.node_h[self.seg_a]
        self.seg_h[:, -1] = np.where(self.seg_break, self.node_h[self.seg_a],
                                     self.node_h[self.seg_b])
        self.seg_rho = self.atm.density(self.seg_h)
        self.seg_rho[:, 0] = self.node_rho[self.seg_a]
        self.seg_rho[:, -1] = np.where(self.seg_break, self.node_rho[self.seg_a],
                                       self.node_rho[self.seg_b])
        self.seg_mu = np.clip(-tq / r, -1.0, 1.0)
        cum = np.concatenate([np.zeros((Ns, 1)), np.cumsum(self.seg_dX, axis=1)],
                             axis=1)
        tot = cum[:, -1:]
        with np.errstate(invalid="ignore", divide="ignore"):
            u = np.where(tot > 0, cum / tot, frac[None, :])
        self.seg_u = u
        self.seg_X = tot[:, 0]
        self.seg_len = np.sum(self.seg_ds, axis=1)
        # cumulative grammage from the top along each chain, at the nodes
        self.node_Xc = np.zeros(self.N)
        for j, c in enumerate(self._chains):
            x = 0.0
            for q in range(len(c) - 1):
                x += self.seg_X[self.chain_seg[j, q]]
                self.node_Xc[c[q + 1]] = x
        # third node for the quadratic source shape: the next node downstream of b
        # on the same chain (not across the ground break), else the node upstream of
        # a; seg_c = -1 if neither exists.  seg_uc = its grammage offset from a in
        # units of the segment's grammage.
        self.seg_c = np.full(Ns, -1, dtype=np.int64)
        self.seg_uc = np.zeros(Ns)
        for j, c in enumerate(self._chains):
            nseg = len(c) - 1
            brk = self._breaks[j]
            for q in range(nseg):
                s = self.chain_seg[j, q]
                if q == brk or self.seg_X[s] <= 0:
                    continue
                cand = None
                if q + 1 < nseg and q + 1 != brk:
                    cand = c[q + 2]
                elif q >= 1 and q - 1 != brk:
                    cand = c[q - 1]
                if cand is None:
                    continue
                self.seg_c[s] = cand
                dxc = self.node_Xc[cand] - self.node_Xc[c[q]]
                self.seg_uc[s] = dxc / self.seg_X[s]

    # ----------------------------------------------------------------- levels
    def _build_levels(self):
        nl = self.levels.size
        per = [np.flatnonzero(self.node_level == e) for e in range(nl)]
        M = max(len(p) for p in per)
        self.lev_node = np.full((nl, M), self.N, dtype=np.int64)
        self.lev_w = np.zeros((nl, M))
        self.node_slot = np.zeros(self.N, dtype=np.int64)
        for e, p in enumerate(per):
            mu = self.node_mu[p]
            o = np.argsort(mu, kind="stable")
            p, mu = p[o], mu[o]
            w = np.zeros(mu.size)
            if mu.size > 1:
                dm = np.diff(mu)
                w[:-1] += 0.5 * dm
                w[1:] += 0.5 * dm
            # extend to [-1, 1] if the extreme nodes are not at +-1
            w[0] += mu[0] + 1.0
            w[-1] += 1.0 - mu[-1]
            self.lev_node[e, :p.size] = p
            self.lev_w[e, :p.size] = w
            self.node_slot[p] = np.arange(p.size)
        self.node_w = np.zeros(self.N)
        valid = self.lev_node < self.N
        self.node_w[self.lev_node[valid]] = self.lev_w[valid]

    # ----------------------------------------------------------------- points
    def _build_points(self):
        """Loss points: every sub-interval split into ``ceil(dX / dX_max)`` parts.

        Per point ``p`` (ordered along each chain, the chain's first point is its top
        node): ``pt_prev`` upstream point (dummy for a chain start), ``pt_ds``,
        ``pt_dX`` (step from the upstream point), ``pt_rho``, ``pt_seg`` and ``pt_u``
        (segment and grammage fraction, for interpolating sources), ``pt_node``
        (node index or -1), ``pt_break`` (ground break: transmission 0),
        ``pt_start``.
        """
        b = self.line_b[self.seg_line]
        npart = np.maximum(1, np.ceil(self.seg_dX / self.dX_max - 1e-9)).astype(int)
        npart[self.seg_break] = 1
        # per part: segment, sub-interval, part index
        seg_rep = np.repeat(np.arange(self.Ns), npart.sum(axis=1))
        q_rep = np.concatenate([np.repeat(np.arange(self.n_sub), n) for n in npart])
        k_rep = np.concatenate([np.concatenate([np.arange(m) for m in n])
                                for n in npart])
        n_rep = npart[seg_rep, q_rep]
        t0 = self.seg_t[seg_rep, q_rep]
        t1 = self.seg_t[seg_rep, q_rep + 1]
        ta = t0 + (t1 - t0) * k_rep / n_rep
        tb = t0 + (t1 - t0) * (k_rep + 1) / n_rep
        dX = line_grammage(self.atm, b[seg_rep], ta, tb, r_e=self.r_e)
        dX[self.seg_break[seg_rep]] = 0.0
        ds = np.where(self.seg_break[seg_rep], 0.0, tb - ta)
        # grammage fraction at the end of each part
        cum_seg = np.concatenate([np.zeros((self.Ns, 1)),
                                  np.cumsum(self.seg_dX, axis=1)], axis=1)
        x_before = cum_seg[seg_rep, q_rep] + self.seg_dX[seg_rep, q_rep] * (
            (k_rep + 1) / n_rep)
        tot = self.seg_X[seg_rep]
        with np.errstate(invalid="ignore", divide="ignore"):
            u_end = np.where(tot > 0, x_before / tot,
                             (q_rep + (k_rep + 1) / n_rep) / self.n_sub)
        last = (q_rep == self.n_sub - 1) & (k_rep == n_rep - 1)
        u_end[last] = 1.0
        r = np.sqrt(b[seg_rep] ** 2 + tb ** 2)
        h_end = r - self.r_e
        node_end = np.where(last, self.seg_b[seg_rep], -1)
        h_end = np.where(last, self.node_h[self.seg_b[seg_rep]], h_end)
        rho_end = self.atm.density(h_end)
        rho_end = np.where(last, self.node_rho[self.seg_b[seg_rep]], rho_end)
        # assemble per chain: start node, then the parts of its segments in order
        order_pts = []
        start = []
        part_of_seg = np.split(np.arange(seg_rep.size),
                               np.cumsum(npart.sum(axis=1))[:-1])
        pt_seg, pt_u, pt_node, pt_ds, pt_dX, pt_rho, pt_brk = ([] for _ in range(7))
        pt_line = []
        self.line_npts = np.zeros(self.Nl, dtype=np.int64)
        for j, c in enumerate(self._chains):
            n0 = len(pt_seg)
            pt_seg.append(-1)
            pt_u.append(0.0)
            pt_node.append(c[0])
            pt_ds.append(0.0)
            pt_dX.append(0.0)
            pt_rho.append(self.node_rho[c[0]])
            pt_brk.append(False)
            pt_line.append(j)
            start.append(n0)
            for q in range(len(c) - 1):
                s = self.chain_seg[j, q]
                parts = part_of_seg[s]
                pt_seg.extend([s] * parts.size)
                pt_u.extend(u_end[parts])
                pt_node.extend(node_end[parts])
                pt_ds.extend(ds[parts])
                pt_dX.extend(dX[parts])
                pt_rho.extend(rho_end[parts])
                pt_brk.extend([bool(self.seg_break[s])] * parts.size)
                pt_line.extend([j] * parts.size)
            self.line_npts[j] = len(pt_seg) - n0
            order_pts.append(np.arange(n0, len(pt_seg)))
        self.Np = len(pt_seg)
        self.pt_seg = np.array(pt_seg, dtype=np.int64)
        self.pt_u = np.array(pt_u)
        self.pt_node = np.array(pt_node, dtype=np.int64)
        self.pt_ds = np.array(pt_ds)
        self.pt_dX = np.array(pt_dX)
        self.pt_rho = np.array(pt_rho)
        self.pt_break = np.array(pt_brk, dtype=bool)
        self.pt_line = np.array(pt_line, dtype=np.int64)
        self.pt_start = np.zeros(self.Np, dtype=bool)
        self.pt_start[start] = True
        self.pt_prev = np.arange(self.Np) - 1
        self.pt_prev[self.pt_start] = self.Np
        self.node_point = np.full(self.N, -1, dtype=np.int64)
        isn = self.pt_node >= 0
        self.node_point[self.pt_node[isn]] = np.flatnonzero(isn)
        assert np.all(self.node_point >= 0), "every node must be a point"
        # length buckets for the scans
        o = np.argsort(self.line_npts, kind="stable")
        cuts = np.array_split(o, min(self.n_buckets, self.Nl))
        self.buckets = []
        for lines in cuts:
            if lines.size == 0:
                continue
            P = int(self.line_npts[lines].max())
            idx = np.full((lines.size, P), self.Np, dtype=np.int64)
            for i, j in enumerate(lines):
                idx[i, :self.line_npts[j]] = order_pts[j]
            self.buckets.append(idx)

    # --------------------------------------------------------------- summary
    def summary(self):
        """Sizes of the tables (dict)."""
        pad = sum(bk.size for bk in self.buckets)
        return {
            "levels": int(self.levels.size), "lines": int(self.Nl),
            "ground_lines": int(self.n_ground), "nodes": int(self.N),
            "segments": int(self.Ns), "n_sub": self.n_sub, "Lmax": int(self.Lmax),
            "loss_points": int(self.Np), "loss_points_padded": int(pad),
            "dX_max": self.dX_max, "max_nodes_per_level": int(self.lev_node.shape[1]),
        }
