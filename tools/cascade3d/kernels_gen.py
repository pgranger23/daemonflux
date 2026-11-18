"""Angular kernels of hadronic interactions from generator events (WP 0.2).

For every interaction ``projectile j at MCEq bin k -> secondary i in MCEq
bin m`` this module accumulates, per interaction,

* the mean multiplicity ``c_0[j, i, k, m]`` (number of secondaries in bin m),
* ``c_l[j, i, k, m] = < sum over those secondaries of P_l(cos theta) >`` for
  l = 0 .. 64, theta the lab angle between secondary and projectile,
* the same sums for two disjoint halves of the events (even / odd event
  number), a per-event sum of squares for the statistical error, and
* a coarse histogram in theta (``THETA_EDGES``, fine at small angles).

``a_l = c_l / c_0`` is the Legendre moment ``<P_l(cos theta)>`` of the
angular distribution, so the azimuth-integrated distribution in
``mu = cos theta`` is ``c_0 * sum_l (2l + 1)/2 a_l P_l(mu)``.

Two event sources go through the same accumulator (``NodeAccumulator``):

``pool``   an existing Monte-Carlo pool file
           (``$W/pools_fix/pool_<pdg>_<iii>_2000.npz`` by default), opened
           read-only.  This is the kernel set of gate G1.3(a).  The MC's
           physics runs since 2026-10-11 (m3b, m3c, fix2, mcup, ...) use
           ``$W/pools_fix`` with ``$W/pylib_fix`` (chromo 0.11.0 with the
           SIBYLL nucleon-projectile fix, ``MILESTONE3_RESULTS.md`` sec.
           0.1).  ``$W/pools`` is the pre-fix set, in which SIBYLL (nodes
           >= 80 GeV) simulated n, pbar and nbar as protons; it is used
           only by the regression environment ``runs/m3b/env_oldpools.sh``.
``fresh``  new chromo events made through ``tools/mc3d/pool.py``'s own
           generator set-up (``_Generators``, ``_target_fractions``,
           ``_model_name``), batched, never stored.

Rules replicated from the Monte Carlo (``tools/mc3d``)
-----------------------------------------------------
R1  Event pools live on ``pool.GRID`` (total projectile energy, 2 GeV to
    1e6 GeV; 10 bins/decade below 80 GeV, 20 above); node i is generated at
    the bin centre ``E0_i``.  Nodes 0..58 exist (centres up to 1.07e4 GeV;
    ``prebuild_pools.py --emax 1e4``).
R2  Node selection (``pool.EventPool.bin_choice``): a projectile of total
    energy E is served by node j with probability t and by node j - 1 with
    1 - t, ``t = ln(E/c_{j-1}) / ln(c_j/c_{j-1})``, where c_{j-1} < E <= c_j;
    node 0 alone below c_0, the last grid node above the last centre.  The
    kernel for MCEq bin k is the t-weighted mix of the two node kernels,
    evaluated at the bin centre ``E = e_grid[k] + m_proj`` (MCEq's grid is
    kinetic energy; the MC selects on total energy).
R3  Rescaling (``pool.EventPool.draw``): ``p_z -> p_z * p_beam(E) /
    p_beam(E0)``, ``p_T`` kept, ``E = sqrt(m^2 + p^2)`` with
    ``constants.MASS`` (``p_beam`` uses ``MASS.get(pdg, 0.938)``).
R4  Energy guard (``draw``): if the summed energy of the kept final state
    exceeds ``E + GUARD_MASS`` (16 * 0.931494 GeV), every kept particle's
    energy and momentum are multiplied by ``(E + GUARD_MASS) / sum``.
    Directions are unchanged; energies (hence child bins) move.
R5  Floor (``draw`` and ``EventPool.lambda_int``): a projectile below
    ``GRID.edges[0] = 2 GeV`` total energy produces nothing.  In the MC the
    interaction length is MCEq's (``ChromoBackend(xs="mceq")`` ->
    ``interactions.MCEqYieldBackend.lambda_int`` on
    ``mceq_tables_SIBYLL23D_v142p.npz``), so such a projectile is still
    absorbed at MCEq's rate.  Here: status ``FLOOR`` and an all-zero kernel
    (the solver must keep MCEq's lambda and use a zero yield).
R6  Kept species (``pool.KEEP``): p, pbar, n, nbar, pi+-, K+-, K_L, K_S,
    Lambda, Lambdabar, mu+-.  Dropped at generation: pi0, gamma, e+-,
    neutrinos (none in practice), Sigma, Xi, Omega and their antiparticles,
    eta and other unstable states that the generator did not decay, charm,
    nuclear fragments.  Nothing is decayed by the MC at production: K_S and
    Lambda are kept and transported (they decay in flight).
R7  Interaction-produced muons carry polarisation 0 in the MC
    (``shower.run_shower``) and are mapped to MCEq's helicity-0 states
    ``mu+`` / ``mu-`` (pdg -13 / 13, helicity 0).
R8  Projectiles that interact (``pool.POOL_PROJECTILES``): p, pbar, n, nbar,
    pi+-, K+-, K_L, K_S, Lambda, Lambdabar.  Muons and neutrinos never
    interact.
R9  Generators: DPMJET-III-19.3 below 80 GeV (node centre), SIBYLL-2.3d
    above (``pool._model_name``); air = N 0.7848 / O 0.2152 by atom, the
    struck nucleus drawn from ``f_i sigma_prod,i`` (``_target_fractions``).
R10 Child energy bin: kinetic energy ``E - m`` on MCEq's bin edges
    (``interactions.MCEqYieldBackend.bin_of`` uses the same convention);
    children outside MCEq's grid are counted in ``n_out_low`` /
    ``n_out_high`` and are not part of the kernel.
R11 Event source of the MC: ``$W/pools_fix`` (see above); the fresh
    source runs with ``$W/pylib_fix`` on the path (``runs/native3d/env.sh``
    via ``runs/dbv142/env.sh``), i.e. the fixed chromo.

Not replicated here, because they are transport rules that the solver must
apply itself: the MC's tracking thresholds (``shower.thresholds``, scheme
``legacy`` or ``chain``, which depend on ``e_nu_min``), MCEq's interaction
lengths (R5) and the decays.

Conventions of the merged file ``$N3/kernels/gen_<tag>.npz``
------------------------------------------------------------
See :func:`merge` and the WP02 report.  ``c0_mceq = c_l[..., 0] * E_k / E_m``
(kinetic bin centres) is MCEq's interaction-block convention
(``tools/mc3d/build_tables.py``: number per bin ``N = M * E_child /
E_parent``).

Usage::

    python kernels_gen.py run --source pool --tag pool2000 --nproc 6
    python kernels_gen.py run --source fresh --tag fresh1e5 --nevents 100000
    python kernels_gen.py merge --tag pool2000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_MC3D = os.path.join(os.path.dirname(_HERE), "mc3d")
if _MC3D not in sys.path:
    sys.path.insert(0, _MC3D)

import pool as mcpool  # noqa: E402  (tools/mc3d)
from constants import MASS  # noqa: E402  (tools/mc3d)

VERSION = "kernels_gen-1.1"
L_MAX = 64
#: theta bin edges [rad]: 0, 30 log bins 1e-4 .. 0.3, 14 linear bins to pi
THETA_EDGES = np.concatenate([[0.0], np.geomspace(1e-4, 0.3, 30),
                              np.linspace(0.3, np.pi, 15)[1:]])
N_THETA = len(THETA_EDGES) - 1

PROJECTILES = tuple(int(p) for p in mcpool.POOL_PROJECTILES)
CHILDREN = (2212, -2212, 2112, -2112, 211, -211, 321, -321, 130, 310,
            3122, -3122, 13, -13)
assert frozenset(CHILDREN) == mcpool.KEEP
#: MCEq names of the (pdg, helicity 0) states the children map to
MCEQ_NAME = {2212: "p+", -2212: "pbar-", 2112: "n0", -2112: "nbar0",
             211: "pi+", -211: "pi-", 321: "K+", -321: "K-",
             130: "K_L0", 310: "K_S0", 3122: "Lambda0",
             -3122: "Lambdabar0", 13: "mu-", -13: "mu+"}
_CIDX = {p: i for i, p in enumerate(CHILDREN)}
_CMASS = np.array([MASS[p] for p in CHILDREN])

POPULATIONS = {"PHYS": {"primary_floor": 1.9, "e_min": 0.03},
               "REF": {"primary_floor": 3.0, "e_min": 0.1}}
POOL_DIR = "/home/pigrange/mc3d_work/pools_fix"
POOL_NPOOL = 2000
N3 = os.environ.get("N3", "/home/pigrange/mc3d_work/runs/native3d")
KERNEL_DIR = os.path.join(N3, "kernels")
SEED_BASE = 20261118
MAX_NPROC = 24

FLOOR, SERVED, UNAVAILABLE = 0, 1, 2

RULES = __doc__[__doc__.index("R1 "):__doc__.index("Not replicated")]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def legendre_table(x, l_max=L_MAX):
    """``P_l(x)`` for l = 0 .. l_max by the upward recurrence.

    Parameters
    ----------
    x : array_like
        Points in [-1, 1].
    l_max : int
        Highest order.

    Returns
    -------
    ndarray, shape (len(x), l_max + 1)
    """
    x = np.asarray(x, float).ravel()
    p = np.empty((x.size, l_max + 1))
    p[:, 0] = 1.0
    if l_max >= 1:
        p[:, 1] = x
    for ell in range(1, l_max):
        p[:, ell + 1] = ((2 * ell + 1) * x * p[:, ell]
                         - ell * p[:, ell - 1]) / (ell + 1)
    return p


def _group_sum(idx, vals, n):
    """``out[g] = sum of vals[idx == g]`` for a 2-D ``vals``."""
    from scipy import sparse
    out_shape = (n, vals.shape[1])
    if idx.size == 0:
        return np.zeros(out_shape)
    a = sparse.csr_matrix((np.ones(idx.size), (idx, np.arange(idx.size))),
                          shape=(n, idx.size))
    return np.asarray(a @ vals)


def _lookup(pid, fn, dtype):
    """``fn(pdg)`` for every entry of ``pid``, vectorised over unique ids."""
    pid = np.asarray(pid)
    if pid.size == 0:
        return np.zeros(0, dtype)
    uq, inv = np.unique(pid, return_inverse=True)
    vals = np.array([fn(int(q)) for q in uq], dtype)
    return vals[inv.ravel()]


def p_beam(pdg, e_tot):
    """Projectile momentum exactly as ``pool._p_beam``."""
    return mcpool._p_beam(pdg, e_tot)


def node_weights(e_tot, grid=None):
    """Probabilities with which the MC serves total energy ``e_tot``.

    Replicates ``pool.EventPool.bin_choice`` (rule R2) and the floor of
    ``draw`` (rule R5).

    Parameters
    ----------
    e_tot : float
        Total projectile energy [GeV].
    grid : pool.EnergyGrid, optional
        Default ``pool.GRID``.

    Returns
    -------
    list of (int, float)
        ``(node, probability)`` with probability > 0; empty below the
        floor.
    """
    grid = grid or mcpool.GRID
    if e_tot < grid.edges[0]:
        return []
    c = grid.centres
    j = int(np.searchsorted(c, e_tot))
    if j <= 0:
        return [(0, 1.0)]
    if j >= len(c):
        return [(len(c) - 1, 1.0)]
    t = np.log(e_tot / c[j - 1]) / np.log(c[j] / c[j - 1])
    out = []
    if t < 1.0:
        out.append((j - 1, float(1.0 - t)))
    if t > 0.0:
        out.append((j, float(t)))
    return out


def node_targets(pdg, node, e_grid, grid=None):
    """MCEq projectile bins that node ``node`` serves, with weights.

    Returns
    -------
    list of (int, float, float)
        ``(k, weight, e_tot)``: MCEq bin, probability that the MC uses this
        node at the bin centre, and the bin centre's total energy.
    """
    m = MASS[int(pdg)]
    out = []
    for k, ek in enumerate(np.asarray(e_grid, float)):
        et = float(ek) + m
        for n, w in node_weights(et, grid):
            if n == int(node) and w > 0.0:
                out.append((k, w, et))
    return out


def projectile_status(pdg, e_grid, available, grid=None):
    """Per MCEq bin: status, the (up to) two nodes and their weights.

    Parameters
    ----------
    pdg : int
    e_grid : array_like
        MCEq kinetic bin centres.
    available : set of int
        Node indices that exist for this projectile.

    Returns
    -------
    status : ndarray of int8
        ``FLOOR`` (0), ``SERVED`` (1) or ``UNAVAILABLE`` (2: a node the MC
        would need does not exist; the MC itself would fail there).
    nodes : ndarray of int, shape (nE, 2)
        -1 where unused.
    weights : ndarray, shape (nE, 2)
    """
    m = MASS[int(pdg)]
    ne = len(e_grid)
    status = np.zeros(ne, np.int8)
    nodes = -np.ones((ne, 2), int)
    wts = np.zeros((ne, 2))
    for k, ek in enumerate(np.asarray(e_grid, float)):
        nw = node_weights(float(ek) + m, grid)
        if not nw:
            status[k] = FLOOR
            continue
        for q, (n, w) in enumerate(nw):
            nodes[k, q], wts[k, q] = n, w
        ok = all(n in available for n, _w in nw)
        status[k] = SERVED if ok else UNAVAILABLE
    return status, nodes, wts


def rescale(pdg, e0, e_tot, pid, px, py, pz, off):
    """Rescale flat event arrays from ``e0`` to ``e_tot`` (rules R3, R4).

    Parameters
    ----------
    pdg : int
        Projectile.
    e0, e_tot : float
        Node energy and target energy (total, GeV).
    pid, px, py, pz : ndarray
        Flat particle arrays (kept species only, as in a pool file).
    off : ndarray
        Event offsets, ``len(off) = n_events + 1``.

    Returns
    -------
    dict
        ``e, px, py, pz`` (float64, after the guard), ``scale`` (guard
        factor per event, 1 when it did not fire), ``ev`` (event index per
        particle).
    """
    off = np.asarray(off, np.int64)
    nev = len(off) - 1
    ev = np.repeat(np.arange(nev), np.diff(off))
    f = p_beam(pdg, e_tot) / p_beam(pdg, e0)
    x = np.asarray(px).astype(np.float64)
    y = np.asarray(py).astype(np.float64)
    z = np.asarray(pz).astype(np.float64) * f
    mm = _lookup(pid, lambda q: MASS.get(q, 0.0), float)
    e = np.sqrt(mm * mm + x * x + y * y + z * z)
    tot = np.bincount(ev, weights=e, minlength=nev)
    avail = e_tot + mcpool.GUARD_MASS
    scale = np.ones(nev)
    fire = tot > avail
    scale[fire] = avail / tot[fire]
    s = scale[ev]
    return {"e": e * s, "px": x * s, "py": y * s, "pz": z * s,
            "scale": scale, "ev": ev}


# ---------------------------------------------------------------------------
# the accumulator
# ---------------------------------------------------------------------------
class NodeAccumulator:
    """Moment sums of one (projectile, pool node) for the bins it serves.

    Parameters
    ----------
    pdg : int
        Projectile.
    node : int
        Index on ``pool.GRID``.
    e_bins : array_like
        MCEq kinetic bin edges (child binning).
    e_grid : array_like
        MCEq kinetic bin centres (projectile bins, for ``node_targets``).
    targets : list of (k, weight, e_tot), optional
        Override of ``node_targets`` (tests).
    e0 : float, optional
        Node energy; default ``pool.GRID.centre(node)``.
    l_max : int
    """

    def __init__(self, pdg, node, e_bins, e_grid, targets=None, e0=None,
                 l_max=L_MAX):
        self.pdg, self.node = int(pdg), int(node)
        self.e0 = float(mcpool.GRID.centre(node) if e0 is None else e0)
        self.e_bins = np.asarray(e_bins, float)
        self.ne = len(self.e_bins) - 1
        if targets is None:
            targets = node_targets(pdg, node, e_grid)
        self.k = np.array([t[0] for t in targets], int)
        self.w = np.array([t[1] for t in targets], float)
        self.etot = np.array([t[2] for t in targets], float)
        self.l_max = int(l_max)
        nk, nc, ng = len(self.k), len(CHILDREN), len(CHILDREN) * self.ne
        self.n_groups = ng
        nl = self.l_max + 1
        self.S = np.zeros((2, nk, ng, nl))
        self.Q = np.zeros((nk, ng, nl))
        self.R = np.zeros((nk, ng, nl))
        self.H = np.zeros((nk, ng, N_THETA))
        self.n_ev = np.zeros(2, np.int64)
        self.n_guard = np.zeros(nk, np.int64)
        self.guard_scale_sum = np.zeros(nk)
        self.e_kept_sum = np.zeros(nk)
        self.n_out_low = np.zeros((nk, nc))
        self.n_out_high = np.zeros((nk, nc))
        # node level (no rescaling): p_T and multiplicity per species
        self.pt_sum = np.zeros(nc)
        self.pt2_sum = np.zeros(nc)
        self.mult = np.zeros(nc)
        # energy bookkeeping of the full final state (fresh source only)
        self.efs = {"n": 0, "sum": 0.0, "sum2": 0.0, "min": np.inf,
                    "max": -np.inf, "n_over_guard": 0, "n_below_elab": 0}
        self.ekept_node = {"n": 0, "sum": 0.0, "max": -np.inf,
                           "n_over_guard": 0}
        self.n_dropped = 0

    # -- input ----------------------------------------------------------------
    def add(self, pid, px, py, pz, off, e_all=None):
        """Add a batch of events in pool format.

        Parameters
        ----------
        pid, px, py, pz : ndarray
            Flat arrays of the kept particles (``pool.KEEP``).
        off : ndarray
            Event offsets.
        e_all : ndarray, optional
            Summed energy of the complete final state of each event
            (before the KEEP filter), for the energy-conservation check.
        """
        pid = np.asarray(pid, np.int64)
        off = np.asarray(off, np.int64)
        nev = len(off) - 1
        if nev <= 0:
            return
        cidx = _lookup(pid, lambda q: _CIDX.get(q, -1), int)
        self.n_dropped += int((cidx < 0).sum())
        ev_glob = self.n_ev.sum() + np.arange(nev)
        half_ev = (ev_glob % 2).astype(int)
        self.n_ev[0] += int((half_ev == 0).sum())
        self.n_ev[1] += int((half_ev == 1).sum())
        ev = np.repeat(np.arange(nev), np.diff(off))
        okc = cidx >= 0
        # node-level diagnostics, unrescaled
        x0 = np.asarray(px, float)
        y0 = np.asarray(py, float)
        z0 = np.asarray(pz, float)
        pt = np.hypot(x0, y0)
        nc = len(CHILDREN)
        self.pt_sum += np.bincount(cidx[okc], pt[okc], nc)
        self.pt2_sum += np.bincount(cidx[okc], pt[okc] ** 2, nc)
        self.mult += np.bincount(cidx[okc], minlength=nc)
        mm = np.where(okc, _CMASS[np.maximum(cidx, 0)], 0.0)
        e_node = np.sqrt(mm * mm + x0 * x0 + y0 * y0 + z0 * z0)
        ek_ev = np.bincount(ev, e_node, nev)
        b = self.ekept_node
        b["n"] += nev
        b["sum"] += float((ek_ev / self.e0).sum())
        b["max"] = max(b["max"], float((ek_ev / self.e0).max()))
        b["n_over_guard"] += int((ek_ev > self.e0
                                  + mcpool.GUARD_MASS).sum())
        if e_all is not None:
            r = np.asarray(e_all, float) - self.e0
            f = self.efs
            f["n"] += nev
            f["sum"] += float(r.sum())
            f["sum2"] += float((r * r).sum())
            f["min"] = min(f["min"], float(r.min()))
            f["max"] = max(f["max"], float(r.max()))
            f["n_over_guard"] += int((r > mcpool.GUARD_MASS).sum())
            f["n_below_elab"] += int((r < -1e-3).sum())
        ng, ne = self.n_groups, self.ne
        for kk in range(len(self.k)):
            rs = rescale(self.pdg, self.e0, self.etot[kk], pid, px, py, pz,
                         off)
            fired = rs["scale"] < 1.0
            self.n_guard[kk] += int(fired.sum())
            self.guard_scale_sum[kk] += float(rs["scale"].sum())
            e_ev = np.bincount(ev, rs["e"], nev)
            self.e_kept_sum[kk] += float((e_ev / self.etot[kk]).sum())
            pabs = np.sqrt(rs["px"] ** 2 + rs["py"] ** 2 + rs["pz"] ** 2)
            cth = np.where(pabs > 0, rs["pz"] / np.where(pabs > 0, pabs, 1),
                           1.0)
            cth = np.clip(cth, -1.0, 1.0)
            ek = rs["e"] - mm
            m = np.searchsorted(self.e_bins, ek, side="right") - 1
            low = okc & (m < 0)
            high = okc & (m >= ne)
            self.n_out_low[kk] += np.bincount(cidx[low], minlength=nc)
            self.n_out_high[kk] += np.bincount(cidx[high], minlength=nc)
            ok = okc & (m >= 0) & (m < ne)
            g = cidx[ok] * ne + m[ok]
            P = legendre_table(cth[ok], self.l_max)
            hv = half_ev[ev[ok]]
            for h in (0, 1):
                s = hv == h
                self.S[h, kk] += _group_sum(g[s], P[s], ng)
            key = ev[ok].astype(np.int64) * ng + g
            uk, inv = np.unique(key, return_inverse=True)
            Y = _group_sum(inv.ravel(), P, len(uk))
            gu = (uk % ng).astype(int)
            self.Q[kk] += _group_sum(gu, Y * Y, ng)
            self.R[kk] += _group_sum(gu, Y * Y[:, :1], ng)
            th = np.arccos(cth[ok])
            tb = np.clip(np.searchsorted(THETA_EDGES, th, side="right") - 1,
                         0, N_THETA - 1)
            self.H[kk] += np.bincount(g * N_THETA + tb,
                                      minlength=ng * N_THETA
                                      ).reshape(ng, N_THETA)

    # -- output ---------------------------------------------------------------
    def arrays(self):
        """Dict of the sums, trimmed in child energy to the populated bins."""
        nk, nc, ne, nl = len(self.k), len(CHILDREN), self.ne, self.l_max + 1
        S = self.S.reshape(2, nk, nc, ne, nl)
        Q = self.Q.reshape(nk, nc, ne, nl)
        R = self.R.reshape(nk, nc, ne, nl)
        H = self.H.reshape(nk, nc, ne, N_THETA)
        occ = np.nonzero(S[:, :, :, :, 0].sum(axis=(0, 1, 2)) > 0)[0]
        m_dim = int(occ[-1]) + 1 if occ.size else 0
        efs = {k: float(v) for k, v in self.efs.items()}
        ekn = {k: float(v) for k, v in self.ekept_node.items()}
        return {
            "pdg": np.array(self.pdg), "node": np.array(self.node),
            "e0": np.array(self.e0), "l_max": np.array(self.l_max),
            "k": self.k, "w": self.w, "etot": self.etot,
            "n_ev": self.n_ev, "m_dim": np.array(m_dim),
            "S": S[..., :m_dim, :], "Q": Q[..., :m_dim, :],
            "R": R[..., :m_dim, :],
            "H": H[..., :m_dim, :].astype(np.float32),
            "n_guard": self.n_guard, "guard_scale_sum": self.guard_scale_sum,
            "e_kept_sum": self.e_kept_sum, "n_out_low": self.n_out_low,
            "n_out_high": self.n_out_high, "pt_sum": self.pt_sum,
            "pt2_sum": self.pt2_sum, "mult": self.mult,
            "n_dropped": np.array(self.n_dropped),
            "efs_json": np.array(json.dumps(efs)),
            "ekept_node_json": np.array(json.dumps(ekn)),
            "theta_edges": THETA_EDGES, "e_bins": self.e_bins,
        }


# ---------------------------------------------------------------------------
# event sources
# ---------------------------------------------------------------------------
def pool_path(pdg, node, pool_dir=POOL_DIR, n_pool=POOL_NPOOL):
    """Path of an MC pool file (same naming as ``EventPool.path``)."""
    return os.path.join(pool_dir, f"pool_{int(pdg)}_{int(node):03d}_"
                        f"{int(n_pool)}.npz")


def read_pool(pdg, node, pool_dir=POOL_DIR, n_pool=POOL_NPOOL):
    """Read one pool file (read-only) and check its header.

    Returns
    -------
    dict
        ``pid, px, py, pz, off, e0, model, n``.
    """
    fp = pool_path(pdg, node, pool_dir, n_pool)
    with np.load(fp, allow_pickle=False) as d:
        out = {k: np.array(d[k]) for k in d.files}
    if int(out["pdg"][0]) != int(pdg):
        raise ValueError(f"{fp}: pdg {out['pdg'][0]} != {pdg}")
    e0 = float(out["e0"][0])
    if abs(e0 / mcpool.GRID.centre(node) - 1.0) > 1e-12:
        raise ValueError(f"{fp}: e0 {e0} is not GRID.centre({node})")
    if len(out["off"]) != int(out["n"][0]) + 1:
        raise ValueError(f"{fp}: offsets do not match n")
    return out


def node_seed(seed_base, pdg, node):
    """Seed of a fresh node; disjoint from the pools' seeds (1 + ...)."""
    s = (int(seed_base) * 1000003 + 7919 * int(node)
         + 104729 * abs(int(pdg)) + (500009 if pdg < 0 else 0))
    return int(s % 900000000) + 1


def fresh_batches(pdg, node, n_events, seed, batch=5000, timing=None):
    """Generate events through ``pool.py``'s generator set-up, in batches.

    Yields ``(pid, px, py, pz, off, e_all)`` per sub-batch (one target
    nucleus at a time), with the KEEP filter of ``pool.generate_pool``.
    Nothing is stored.  ``timing`` (dict) receives ``t_gen``.

    The chromo generators are a process-wide singleton
    (``pool._Generators``): the seed is used only by the first construction
    in a process, so the driver runs every node in a fresh process.
    """
    gens = mcpool._Generators(seed)
    e0 = mcpool.GRID.centre(node)
    name = mcpool._model_name(e0)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    p = mcpool._target_fractions(gens, pdg, e0, name)
    if timing is not None:
        timing["t_init"] = time.time() - t0
        timing["model"] = name
        timing.setdefault("t_gen", 0.0)
    keep = np.array(sorted(mcpool.KEEP))
    done = 0
    while done < n_events:
        nb = min(int(batch), n_events - done)
        counts = rng.multinomial(nb, p)
        for (_nm, a, z, _f), k in zip(mcpool.AIR_COMPONENTS, counts):
            if k == 0:
                continue
            t1 = time.time()
            m = gens.get(name, pdg, e0, (a, z))
            pid_l, px_l, py_l, pz_l, nper, eall = [], [], [], [], [], []
            for ev in m(int(k)):
                fs = ev.final_state()
                q = np.asarray(fs.pid, np.int32)
                sel = np.isin(q, keep)
                pid_l.append(q[sel])
                px_l.append(np.asarray(fs.px, float)[sel])
                py_l.append(np.asarray(fs.py, float)[sel])
                pz_l.append(np.asarray(fs.pz, float)[sel])
                nper.append(int(sel.sum()))
                eall.append(float(np.sum(fs.en)))
            if timing is not None:
                timing["t_gen"] += time.time() - t1
            off = np.zeros(len(nper) + 1, np.int64)
            off[1:] = np.cumsum(nper)
            yield (np.concatenate(pid_l), np.concatenate(px_l),
                   np.concatenate(py_l), np.concatenate(pz_l), off,
                   np.array(eall))
        done += nb


# ---------------------------------------------------------------------------
# MCEq grid
# ---------------------------------------------------------------------------
def mceq_grid(population="PHYS"):
    """MCEq's energy grid and species list for a population.

    Built with ``mceq3d_db.make_run`` (the project's database selection),
    ``config.e_min`` and the primary floor of ``POPULATIONS``.

    Returns
    -------
    dict
        ``e_grid, e_bins, e_widths`` (kinetic, GeV), ``species`` (MCEq
        cascade-particle names in MCEq order), ``pdg``, ``helicity``,
        ``mass``, ``db_label``, ``db_sha256``.
    """
    import importlib.util  # noqa: F401  (MCEq.config needs it)
    import MCEq.config as cfg
    import crflux.models as crf
    import mceq3d_db
    from mceq3d_flux import _primary_floor_set

    pop = POPULATIONS[population]
    old = cfg.e_min
    cfg.e_min = pop["e_min"]
    cfg.debug_level = 0
    try:
        with _primary_floor_set(pop["primary_floor"]):
            mc = mceq3d_db.make_run(
                interaction_model="SIBYLL23D",
                primary_model=(crf.GlobalSplineFitBeta, None),
                theta_deg=0.0)
    finally:
        cfg.e_min = old
    cp = mc.pman.cascade_particles
    return {"e_grid": np.array(mc.e_grid), "e_bins": np.array(mc.e_bins),
            "e_widths": np.array(mc.e_widths),
            "species": np.array([p.name for p in cp]),
            "pdg": np.array([int(p.pdg_id[0]) for p in cp]),
            "helicity": np.array([int(p.pdg_id[1]) for p in cp]),
            "mass": np.array([float(p.mass) for p in cp]),
            "db_label": np.array(mc.mceq3d_db.label),
            "db_sha256": np.array(mc.mceq3d_db.sha256)}


def grid_cached(population, path):
    """``mceq_grid`` cached in an npz at ``path`` (written atomically)."""
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as d:
            return {k: np.array(d[k]) for k in d.files}
    g = mceq_grid(population)
    atomic_savez(path, g, compressed=False)
    return g


def check_mceq_kernels(grid, population, kernel_dir=KERNEL_DIR):
    """Compare with ``kernels_mceq``'s file when it exists.

    Returns
    -------
    str
        ``"absent"`` or ``"agrees"``; raises ``AssertionError`` otherwise.
    """
    fp = os.path.join(kernel_dir, f"mceq_{population}.npz")
    if not os.path.exists(fp):
        return "absent"
    with np.load(fp, allow_pickle=False) as d:
        for k in ("e_grid", "e_bins"):
            np.testing.assert_allclose(d[k], grid[k], rtol=1e-12, atol=0)
        sp = [str(s) for s in d["species"]]
    mine = [str(s) for s in grid["species"]]
    assert sp[:len(mine)] == mine or sp == mine, \
        f"species differ: {sp} vs {mine}"
    return "agrees"


# ---------------------------------------------------------------------------
# file helpers
# ---------------------------------------------------------------------------
def atomic_savez(path, d, compressed=True):
    """Write an npz atomically (temporary file in the same directory)."""
    dn = os.path.dirname(os.path.abspath(path))
    os.makedirs(dn, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dn, prefix=".tmp_", suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as fh:
            (np.savez_compressed if compressed else np.savez)(fh, **d)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def tag_dir(tag, kernel_dir=KERNEL_DIR):
    return os.path.join(kernel_dir, f"gen_{tag}")


def node_file(tag, pdg, node, kernel_dir=KERNEL_DIR):
    return os.path.join(tag_dir(tag, kernel_dir),
                        f"node_{int(pdg)}_{int(node):03d}.npz")


def node_list(source, pool_dir=POOL_DIR):
    """All (pdg, node) of a source.

    Both sources use the node set of the MC's pools (the files present in
    ``pool_dir``: 12 projectiles x nodes 0..58), so that the fresh kernels
    cover exactly the bins the MC can serve.
    """
    out = []
    for p in PROJECTILES:
        for i in range(mcpool.GRID.n):
            if os.path.exists(pool_path(p, i, pool_dir)):
                out.append((p, i))
    return out


def _code_sha():
    with open(os.path.abspath(__file__), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _generator_versions():
    try:
        import chromo
        import chromo.models as cm
        out = {"chromo": str(chromo.__version__)}
        for n in ("Sibyll23d", "DpmjetIII193"):
            c = getattr(cm, n)
            out[n] = f"{getattr(c, 'label', n)} {getattr(c, 'version', '')}"
        return out
    except Exception as exc:          # pragma: no cover
        return {"error": repr(exc)}


# ---------------------------------------------------------------------------
# per-node worker
# ---------------------------------------------------------------------------
def run_node(task):
    """Build one node file.  ``task`` is a dict (see :func:`run`)."""
    pdg, node, tag = int(task["pdg"]), int(task["node"]), task["tag"]
    out = node_file(tag, pdg, node, task["kernel_dir"])
    if os.path.exists(out):
        try:
            with np.load(out, allow_pickle=False) as d:
                if str(d["config_sha"]) == task["config_sha"]:
                    return {"pdg": pdg, "node": node, "status": "cached"}
        except Exception:
            pass
    t0 = time.time()
    with np.load(task["grid_path"], allow_pickle=False) as d:
        e_bins, e_grid = np.array(d["e_bins"]), np.array(d["e_grid"])
    acc = NodeAccumulator(pdg, node, e_bins, e_grid, l_max=task["l_max"])
    timing = {"t_gen": 0.0}
    meta = {"source": task["source"], "tag": tag}
    if task["source"] == "pool":
        d = read_pool(pdg, node, task["pool_dir"])
        acc.add(d["pid"], d["px"], d["py"], d["pz"], d["off"])
        meta["model"] = str(d["model"][0])
        meta["pool_file"] = pool_path(pdg, node, task["pool_dir"])
    else:
        log = os.path.join(tag_dir(tag, task["kernel_dir"]), "logs",
                           f"node_{pdg}_{node:03d}.log")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        seed = node_seed(task["seed"], pdg, node)
        meta["seed"] = seed
        for b in fresh_batches(pdg, node, int(task["nevents"]), seed,
                               batch=int(task["batch"]), timing=timing):
            acc.add(*b[:5], e_all=b[5])
        meta["model"] = timing.get("model", "")
        meta["generators"] = _generator_versions()
    t_tot = time.time() - t0
    timing["t_total"] = t_tot
    timing["t_acc"] = t_tot - timing.get("t_gen", 0.0) \
        - timing.get("t_init", 0.0)
    a = acc.arrays()
    a["config_sha"] = np.array(task["config_sha"])
    a["meta_json"] = np.array(json.dumps(meta))
    a["timing_json"] = np.array(json.dumps(timing))
    atomic_savez(out, a)
    return {"pdg": pdg, "node": node, "status": "built",
            "n": int(acc.n_ev.sum()), **timing}


def run(tag, source, nevents=None, nproc=12, nodes=None, seed=SEED_BASE,
        batch=5000, l_max=L_MAX, kernel_dir=KERNEL_DIR, pool_dir=POOL_DIR):
    """Driver: all (or the given) nodes in parallel, resumable.

    Parameters
    ----------
    tag : str
        Output ``$N3/kernels/gen_<tag>/``.
    source : {"pool", "fresh"}
    nevents : int
        Events per node (fresh only).
    nproc : int
        Worker processes, at most ``MAX_NPROC``.
    nodes : list of (pdg, node), optional
    """
    if nproc > MAX_NPROC:
        raise SystemExit(f"--nproc {nproc} refused: the project cap is "
                         f"{MAX_NPROC} cores")
    if nproc < 1:
        raise SystemExit("--nproc must be >= 1")
    if source not in ("pool", "fresh"):
        raise SystemExit(f"unknown source {source!r}")
    if source == "fresh" and not nevents:
        raise SystemExit("--nevents is required for --source fresh")
    td = tag_dir(tag, kernel_dir)
    os.makedirs(td, exist_ok=True)
    grid_path = os.path.join(td, "grid_PHYS.npz")
    grid = grid_cached("PHYS", grid_path)
    print("kernels_mceq check (PHYS):", check_mceq_kernels(grid, "PHYS"),
          flush=True)
    cfg = {"version": VERSION, "source": source,
           "nevents": int(nevents) if source == "fresh" else POOL_NPOOL,
           "seed": int(seed), "l_max": int(l_max),
           "theta_edges_sha": hashlib.sha256(THETA_EDGES.tobytes())
           .hexdigest(),
           "e_bins_sha": hashlib.sha256(grid["e_bins"].tobytes()).hexdigest(),
           "pool_dir": pool_dir if source == "pool" else ""}
    csha = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()) \
        .hexdigest()[:16]
    cpath = os.path.join(td, "config.json")
    if os.path.exists(cpath):
        with open(cpath) as fh:
            old = json.load(fh)
        if old.get("config_sha") != csha:
            raise SystemExit(f"{cpath} holds a different configuration; "
                             "use a new --tag")
    else:
        tmp = cpath + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({**cfg, "config_sha": csha}, fh, indent=1)
        os.replace(tmp, cpath)
    todo = list(nodes) if nodes else node_list(source, pool_dir)
    # slow (DPMJET) nodes first for load balance
    todo.sort(key=lambda t: (mcpool.GRID.centre(t[1]) >= mcpool.E_SWITCH,
                             t[1], t[0]))
    tasks = [{"pdg": p, "node": i, "tag": tag, "source": source,
              "nevents": nevents, "seed": seed, "batch": batch,
              "l_max": l_max, "kernel_dir": kernel_dir,
              "pool_dir": pool_dir, "grid_path": grid_path,
              "config_sha": csha} for p, i in todo]
    print(f"{len(tasks)} nodes, source={source}, nproc={nproc}", flush=True)
    import multiprocessing as mp
    t0 = time.time()
    res = []
    ctx = mp.get_context("fork")
    with ctx.Pool(nproc, maxtasksperchild=1) as pl:
        for r in pl.imap_unordered(run_node, tasks, chunksize=1):
            res.append(r)
            if r["status"] == "built" and source == "fresh":
                print(f"  {r['pdg']:+6d} node {r['node']:3d} "
                      f"{r.get('model', '')} n={r['n']} "
                      f"gen {r.get('t_gen', 0):.1f} s "
                      f"acc {r.get('t_acc', 0):.1f} s", flush=True)
            if len(res) % 50 == 0:
                print(f"  {len(res)}/{len(tasks)} ({time.time() - t0:.0f} s)",
                      flush=True)
    nb = sum(r["status"] == "built" for r in res)
    print(f"done: {nb} built, {len(res) - nb} cached, "
          f"{time.time() - t0:.0f} s", flush=True)
    return res


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------
def _load_node(fp):
    with np.load(fp, allow_pickle=False) as d:
        return {k: np.array(d[k]) for k in d.files}


def merge(tag, populations=("PHYS", "REF"), kernel_dir=KERNEL_DIR):
    """Merge the node files of a tag into one kernel file per population.

    Writes ``gen_<tag>.npz`` (PHYS) and ``gen_<tag>_<POP>.npz`` for the
    others.  Arrays (``P`` projectiles, ``C`` children, ``K`` projectile bins
    ``k_index``, ``M`` child bins ``m_index``, both indices into the
    population's MCEq grid):

    ``c_l`` (P, C, K, M, L+1)       per interaction, sum of P_l(cos theta)
    ``c_l_var`` (P, C, K, M, L+1)   variance of ``c_l`` (statistical)
    ``c_l_cov0`` (P, C, K, M, L+1)  covariance of ``c_l`` with ``c_0``
    ``c_l_half`` (2, P, C, K, M, L+1)  the same from the even / odd events
    ``c0_mceq`` (P, C, K, M)        ``c_l[..., 0] * E_k / E_m`` (MCEq's
                                    block convention)
    ``hist`` (P, C, K, M, T)        per interaction, number in theta bin
    ``n_entries`` (P, C, K, M)      raw particle count behind each entry
    ``status`` (P, nE)              FLOOR / SERVED / UNAVAILABLE per bin
    """
    td = tag_dir(tag, kernel_dir)
    with open(os.path.join(td, "config.json")) as fh:
        cfg = json.load(fh)
    grids = {"PHYS": grid_cached("PHYS", os.path.join(td, "grid_PHYS.npz"))}
    files = {}
    for p in PROJECTILES:
        for i in range(mcpool.GRID.n):
            fp = node_file(tag, p, i, kernel_dir)
            if os.path.exists(fp):
                files[(p, i)] = fp
    if not files:
        raise SystemExit(f"no node files under {td}")
    g = grids["PHYS"]
    e_grid = g["e_grid"]
    ne = len(e_grid)
    nl = int(cfg["l_max"]) + 1
    nP, nC = len(PROJECTILES), len(CHILDREN)
    status = np.zeros((nP, ne), np.int8)
    nodes = -np.ones((nP, ne, 2), int)
    wts = np.zeros((nP, ne, 2))
    for jp, p in enumerate(PROJECTILES):
        avail = {i for (q, i) in files if q == p}
        status[jp], nodes[jp], wts[jp] = projectile_status(p, e_grid, avail)
    served = np.nonzero((status == SERVED).any(axis=0))[0]
    if served.size == 0:
        raise SystemExit(f"{td}: no MCEq bin has all the nodes it needs")
    k_index = np.arange(served.min(), served.max() + 1)
    # load nodes once, find the child-bin range
    data = {key: _load_node(fp) for key, fp in files.items()}
    sha = {str(d["config_sha"]) for d in data.values()}
    if sha != {cfg["config_sha"]}:
        raise SystemExit(f"node files of mixed configurations: {sha}")
    m_hi = max(int(d["m_dim"]) for d in data.values())
    m_index = np.arange(m_hi)
    nK, nM = len(k_index), len(m_index)
    c_l = np.zeros((nP, nC, nK, nM, nl))
    c_var = np.zeros_like(c_l)
    c_cov = np.zeros_like(c_l)
    c_half = np.zeros((2,) + c_l.shape)
    hist = np.zeros((nP, nC, nK, nM, N_THETA))
    n_ent = np.zeros((nP, nC, nK, nM))
    guard = np.zeros((nP, nK))
    ekept = np.zeros((nP, nK))
    out_low = np.zeros((nP, nC, nK))
    n_events = np.zeros((nP, mcpool.GRID.n), np.int64)
    models = {}
    for (p, i), d in data.items():
        jp = PROJECTILES.index(p)
        n_events[jp, i] = int(d["n_ev"].sum())
        models[f"{p}_{i}"] = json.loads(str(d["meta_json"])).get("model", "")
        md = int(d["m_dim"])
        nev = float(d["n_ev"].sum())
        nh = d["n_ev"].astype(float)
        for kk, k in enumerate(d["k"]):
            if status[jp, k] != SERVED:
                continue
            q = np.nonzero(nodes[jp, k] == i)[0]
            if q.size != 1:
                raise RuntimeError(f"node {p}/{i} not used for bin {k}")
            w = wts[jp, k, q[0]]
            if abs(w - float(d["w"][kk])) > 1e-12:
                raise RuntimeError(f"weight mismatch {p}/{i}/{k}")
            kx = k - k_index[0]
            S = d["S"][:, kk]                       # (2, C, md, L+1)
            mean = S.sum(axis=0) / nev
            c_l[jp, :, kx, :md] += w * mean
            var = (d["Q"][kk] / nev - mean ** 2) / nev
            c_var[jp, :, kx, :md] += w * w * np.maximum(var, 0.0)
            cov = (d["R"][kk] / nev - mean * mean[..., :1]) / nev
            c_cov[jp, :, kx, :md] += w * w * cov
            for h in (0, 1):
                c_half[h, jp, :, kx, :md] += w * S[h] / max(nh[h], 1.0)
            hist[jp, :, kx, :md] += w * d["H"][kk] / nev
            n_ent[jp, :, kx, :md] += S.sum(axis=0)[..., 0]
            guard[jp, kx] += w * d["n_guard"][kk] / nev
            ekept[jp, kx] += w * d["e_kept_sum"][kk] / nev
            out_low[jp, :, kx] += w * d["n_out_low"][kk] / nev
    meta = {"tag": tag, "version": VERSION, "config": cfg,
            "rules": RULES, "generators": _generator_versions(),
            "node_models": models, "code_sha256": _code_sha(),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mceq_db": str(g["db_label"]), "mceq_db_sha256":
            str(g["db_sha256"])}
    written = []
    for pop in populations:
        if pop == "PHYS":
            gp, off = g, 0
        else:
            gp = grid_cached(pop, os.path.join(td, f"grid_{pop}.npz"))
            off = ne - len(gp["e_grid"])
            np.testing.assert_allclose(gp["e_grid"], e_grid[off:],
                                       rtol=1e-12)
            np.testing.assert_allclose(gp["e_bins"], g["e_bins"][off:],
                                       rtol=1e-12)
        msel = m_index >= off
        ksel = k_index >= off
        egp = gp["e_grid"]
        ki = k_index[ksel] - off
        mi = m_index[msel] - off
        cl = c_l[:, :, ksel][:, :, :, msel]
        c0m = cl[..., 0] * egp[ki][None, None, :, None] \
            / egp[mi][None, None, None, :]
        pm = dict(meta, population=pop, **POPULATIONS[pop],
                  grid_offset_from_PHYS=int(off))
        arrs = {
            "e_grid": egp, "e_bins": gp["e_bins"],
            "e_widths": gp["e_widths"],
            "mceq_species": gp["species"], "mceq_pdg": gp["pdg"],
            "mceq_helicity": gp["helicity"],
            "projectiles": np.array(PROJECTILES),
            "projectile_names": np.array([MCEQ_NAME[p]
                                          for p in PROJECTILES]),
            "children": np.array(CHILDREN),
            "child_names": np.array([MCEQ_NAME[p] for p in CHILDREN]),
            "child_helicity": np.zeros(nC, int),
            "k_index": ki, "m_index": mi,
            "l_max": np.array(nl - 1), "theta_edges": THETA_EDGES,
            "c_l": cl,
            "c_l_var": c_var[:, :, ksel][:, :, :, msel],
            "c_l_cov0": c_cov[:, :, ksel][:, :, :, msel],
            "c_l_half": c_half[:, :, :, ksel][:, :, :, :, msel]
            .astype(np.float32),
            "c0_mceq": c0m,
            "hist": hist[:, :, ksel][:, :, :, msel].astype(np.float32),
            "n_entries": n_ent[:, :, ksel][:, :, :, msel],
            "status": status[:, off:], "node_index": nodes[:, off:],
            "node_weight": wts[:, off:],
            "guard_fraction": guard[:, ksel],
            "e_kept_over_e": ekept[:, ksel],
            "out_low_per_int": out_low[:, :, ksel],
            "n_events_node": n_events,
            "grid_e0_node": mcpool.GRID.centres,
            "population": np.array(pop),
            "primary_floor": np.array(POPULATIONS[pop]["primary_floor"]),
            "e_min": np.array(POPULATIONS[pop]["e_min"]),
            "meta_json": np.array(json.dumps(pm)),
        }
        name = f"gen_{tag}.npz" if pop == "PHYS" else f"gen_{tag}_{pop}.npz"
        fp = os.path.join(kernel_dir, name)
        atomic_savez(fp, arrs)
        written.append(fp)
        print("wrote", fp, flush=True)
    return written


def load(tag, population="PHYS", kernel_dir=KERNEL_DIR):
    """Load a merged kernel file as a dict of arrays (``allow_pickle=False``).

    ``meta`` holds the decoded metadata.
    """
    name = f"gen_{tag}.npz" if population == "PHYS" \
        else f"gen_{tag}_{population}.npz"
    with np.load(os.path.join(kernel_dir, name), allow_pickle=False) as d:
        out = {k: np.array(d[k]) for k in d.files}
    out["meta"] = json.loads(str(out["meta_json"]))
    return out


# ---------------------------------------------------------------------------
# delta-M
# ---------------------------------------------------------------------------
def normalised_moments(c_l):
    """``a_l = c_l / c_0`` (``a_0 = 1``; 0 where ``c_0 <= 0``)."""
    c_l = np.asarray(c_l, float)
    c0 = c_l[..., :1]
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(c0 > 0, c_l / np.where(c0 > 0, c0, 1.0), 0.0)
    a[..., 0] = np.where(c0[..., 0] > 0, 1.0, 0.0)
    return a


def moment_errors(c_l, c_var, c_cov0):
    """Statistical error of ``a_l = c_l / c_0`` (first-order, with the
    covariance of numerator and denominator); 0 where ``c_0 <= 0``."""
    c0 = c_l[..., :1]
    a = normalised_moments(c_l)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = (c_var - 2 * a * c_cov0 + a * a * c_var[..., :1]) \
            / np.where(c0 > 0, c0 * c0, 1.0)
    return np.where(c0 > 0, np.sqrt(np.maximum(v, 0.0)), 0.0)


def delta_m(c_l, L=32, g=None, n_tail=4, n_entries=None, min_entries=0,
            eps=1e-6, clip=False):
    """Forward fraction and reduced moments of a delta-M truncation.

    The kernel ``c_0 [g delta(1 - mu) + (1 - g) sum_{l<=L} (2l+1)/2 a*_l
    P_l(mu)]`` reproduces ``a_l`` for every ``l <= L`` exactly:
    ``a_l = g + (1 - g) a*_l``.

    Parameters
    ----------
    c_l : array_like, (..., l_max + 1)
        Moments (``c_l`` or already normalised ``a_l``).
    L : int
        Truncation order.
    g : float or array_like, optional
        Forward fraction to impose.  ``g = 1`` gives exactly the
        straight-ahead kernel (``a* = (1, 0, ..., 0)``); ``g = 0`` the plain
        Legendre series.  Default: estimated from the moments.
    n_tail : int
        The estimate is the mean of ``a_{L+1} .. a_{L+n_tail}`` (fewer if
        ``l_max`` is reached), clipped to [0, 1].  ``n_tail = 1`` is the
        classical choice ``g = a_{L+1}`` (Wiscombe 1977); averaging a few
        moments reduces the statistical noise of the estimate.  Any g in
        [0, 1) keeps the low moments exact, so noise in g only moves the
        unresolved tail.
    n_entries, min_entries : array_like, int
        Where the raw particle count is below ``min_entries`` the entry is
        set to straight ahead (g = 1).
    eps : float
        If ``1 - g < eps`` the entry is set to exactly straight ahead.
    clip : bool
        Clip ``a*`` to [-1, 1] (breaks exactness where it acts; off by
        default).

    Returns
    -------
    g : ndarray, (...)
    a_star : ndarray, (..., L + 1)
    """
    a = normalised_moments(c_l)
    lmax = a.shape[-1] - 1
    if L < 0 or L > lmax:
        raise ValueError(f"L={L} outside 0..{lmax}")
    empty = a[..., 0] <= 0
    if g is None:
        hi = min(L + n_tail, lmax)
        if L + 1 > lmax:
            g = np.zeros(a.shape[:-1])
        else:
            g = a[..., L + 1:hi + 1].mean(axis=-1)
        g = np.clip(g, 0.0, 1.0)
    else:
        g = np.broadcast_to(np.asarray(g, float), a.shape[:-1]).copy()
        if np.any((g < 0) | (g > 1)):
            raise ValueError("g must lie in [0, 1]")
    if n_entries is not None and min_entries > 0:
        g = np.where(np.asarray(n_entries) < min_entries, 1.0, g)
    sa = (1.0 - g) < eps
    sa |= empty
    g = np.where(sa, 1.0, g)
    den = np.where(sa, 1.0, 1.0 - g)
    a_star = (a[..., :L + 1] - g[..., None]) / den[..., None]
    a_star[..., 0] = 1.0
    a_star = np.where(sa[..., None], 0.0, a_star)
    a_star[..., 0] = 1.0
    if clip:
        a_star = np.clip(a_star, -1.0, 1.0)
    return g, a_star


def delta_m_moments(g, a_star, l_max=None):
    """Moments ``a_l`` of a delta-M kernel (``g + (1 - g) a*_l``; beyond L,
    only the forward peak contributes: ``a_l = g``)."""
    g = np.asarray(g, float)
    L = a_star.shape[-1] - 1
    l_max = L if l_max is None else int(l_max)
    out = np.empty(g.shape + (l_max + 1,))
    out[..., :L + 1] = g[..., None] + (1.0 - g[..., None]) \
        * a_star[..., :min(L, l_max) + 1]
    out[..., L + 1:] = g[..., None]
    return out


def series(mu, a_star):
    """Smooth part ``sum_l (2l+1)/2 a*_l P_l(mu)`` (per unit cos theta)."""
    L = a_star.shape[-1] - 1
    P = legendre_table(mu, L)
    w = (2 * np.arange(L + 1) + 1) / 2.0
    return P @ (w * a_star)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_nodes(s):
    out = []
    for item in s.split(","):
        p, i = item.split(":")
        out.append((int(p), int(i)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="build node files")
    r.add_argument("--tag", required=True)
    r.add_argument("--source", choices=("pool", "fresh"), required=True)
    r.add_argument("--nevents", type=int, default=None)
    r.add_argument("--nproc", type=int, default=12)
    r.add_argument("--nodes", type=_parse_nodes, default=None,
                   help="pdg:node,pdg:node,...")
    r.add_argument("--seed", type=int, default=SEED_BASE)
    r.add_argument("--batch", type=int, default=5000)
    r.add_argument("--kernel-dir", default=KERNEL_DIR)
    r.add_argument("--pool-dir", default=POOL_DIR)
    r.add_argument("--merge", action="store_true",
                   help="merge after the run")
    m = sub.add_parser("merge", help="merge node files")
    m.add_argument("--tag", required=True)
    m.add_argument("--populations", nargs="+", default=["PHYS", "REF"])
    m.add_argument("--kernel-dir", default=KERNEL_DIR)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.tag, a.source, a.nevents, a.nproc, a.nodes, a.seed, a.batch,
            kernel_dir=a.kernel_dir, pool_dir=a.pool_dir)
        if a.merge:
            merge(a.tag, kernel_dir=a.kernel_dir)
    else:
        merge(a.tag, a.populations, a.kernel_dir)


if __name__ == "__main__":
    main()
