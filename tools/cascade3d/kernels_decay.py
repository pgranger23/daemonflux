"""Decay kernels with angular moments for the cascade3d solver (work package 0.3).

For every decay "parent species j, kinetic-energy bin k  ->  child species i, bin m"
this module gives the yield and the angular distribution of the child about the
parent's direction, as Legendre moments

    M_l[m, k] = <P_l(cos theta)>-weighted yield,   l = 0 .. 64,

where theta is the *lab-frame* angle between the child's and the parent's momentum.
The kinematics are those of ``tools/mc3d/decays.py`` (decay channels and branching
ratios, isotropic two-body decays, flat-Dalitz three-body decays, exact muon
polarisation in pi/K -> mu nu, polarised Michel decay).

Conventions (derived from MCEq's source, ``MCEq/core.py`` and ``particlemanager.py``)
-----------------------------------------------------------------------------------
* Energy grid: MCEq's own, in **kinetic** energy ``T``.  ``e_bins`` are the bin edges,
  ``e_grid`` the (geometric) bin centres, ``e_widths = diff(e_bins)``.  Parent and child
  bins are both in kinetic energy; a child of mass ``m_c`` and total energy ``E_c`` is
  in the bin containing ``T_c = E_c - m_c``.  Children outside the grid are lost.
* MCEq stores the state as ``phi[i, m] = dN_i/dT`` per GeV, and its decay block for
  (parent j, child i) is ``D[m, k] = Y[k -> m] * w_k / w_m`` (``w`` = ``e_widths``),
  with ``Y[k -> m]`` the number of children landing in child bin ``m`` per decay of a
  parent in bin ``k``.  (The decay matrix ``dec_m`` additionally multiplies column k by
  ``m / (c tau (T_k + m))`` and is non-zero only for parent bins ``k >= mix_idx``;
  below ``mix_idx`` MCEq treats the parent as a resonance.  The raw blocks, which this
  module compares with, are ``particle.decay_dists[child]``.)
* **This module stores the moments in the same convention**:
  ``mom[.., l] = (w_k / w_m) * sum over decays of P_l(cos theta) per decay`` including
  the branching ratio.  So ``l = 0`` is directly comparable with MCEq's block, and the
  per-decay yield is ``Y = D * w_m / w_k``.  A moment normalised to the yield is
  ``mom[l] / mom[0]``.
* Branching ratios are those of ``decays.py``, renormalised to sum to one for each
  parent exactly as ``decays._pick`` does (pi: 0.99988 -> 1; K_L: 0.9964 -> 1; K_S:
  0.9989 -> 1; K+-: 1.0001 -> 1), so that the kernels agree with the Monte Carlo.
  The raw table sums are stored in the metadata.

Parent energy inside its bin
----------------------------
Selectable with ``parent_mode``: ``"centre"`` (delta function at ``e_grid[k]``, the
default, here and on the command line), ``"lnE"`` (uniform in ln T), ``"T"`` (uniform
in T), or ``"pow<g>"`` (density proportional to ``T**-g``, e.g. ``"pow2.7"``).  The
default is the one that reproduces MCEq's blocks best and the one ``decay.npz`` was
built with; see ``WP03_REPORT.md`` for the comparison.

Helicity
--------
MCEq carries mu+- as helicity species ``_l`` (h = -1) and ``_r`` (h = +1) and as
unpolarised ``mu+``/``mu-`` (helicity 0).

* pi, K -> mu nu (two-body): the muon's lab energy fixes ``cos theta*`` in the parent
  rest frame; that same angle fixes the lab angle of the muon and its longitudinal
  polarisation ``P_L = (m_mu / E_lab) (s_vec . p_hat)`` (``decays.muon_polarisation``,
  evaluated analytically).  A muon born with rest-frame helicity ``h0`` (mu+: -1,
  mu-: +1) is found with lab helicity ``+1`` with probability ``(1 + P_L)/2`` and
  ``-1`` with probability ``(1 - P_L)/2``; the two helicity yields are separate child
  species, each with its own angular moments.
* Three-body muons (K_mu3, K_L mu3) are unpolarised: helicity-0 species ``mu+``/``mu-``.
* Muon decay: parents ``mu+_l``, ``mu+_r``, ``mu+`` (and the mu- ones) are fully
  left-/right-polarised (P_L = -1/+1) or unpolarised (0).  The polarised Michel
  distribution is ``n(x) [1 + s A(x) P_L cos theta*]`` (``decays.michel_sample``), ``s =
  -1`` for mu+, ``+1`` for mu-, theta* measured from the muon direction.  Children are
  ``numu``/``antinumu`` and ``nue``/``antinue``; the moments are about the muon's
  direction.

Method
------
Two-body decays (pi, K_mu2, K -> pi pi0, K_S): analytic.  For a parent of total energy
``E_p`` the child energy is linear in ``cos theta*`` (isotropic in the rest frame), so
each child bin corresponds to an interval of ``cos theta*`` that is integrated by
Gauss-Legendre quadrature, together with the lab angle and, for muons, the helicity
probabilities.  The parent energy distribution inside the bin is integrated with a
Gauss-Legendre rule in the chosen variable.

Three-body decays and muon decay: rest-frame decays are sampled once
(``N`` per channel), every sample is given a parent energy drawn from the in-bin
distribution of each parent bin ``k``, boosted along the parent direction and binned.
Samples are split into two disjoint halves (even / odd index).  ``hd = (A - B) / 2``,
with ``A``, ``B`` the half-sample estimates, is an unbiased estimator of the variance
of ``mom``: its magnitude is the 1-sigma statistical error of the full-sample moment.
``n`` is the number of samples in the (k, m) entry.  Analytic entries have ``hd = 0``.

File layout (``decay.npz``, ``allow_pickle=False``)
---------------------------------------------------
``e_grid, e_bins, e_widths``: MCEq grid.  ``l_max``.  ``meta_json``: JSON text.
Pairs ``p = 0 .. npair-1`` in CSR layout: ``pair_parent[p]``, ``pair_child[p]``
(MCEq species names), ``pair_ptr`` (npair+1 offsets), and the concatenated entries
``k`` (parent bin), ``m`` (child bin), ``mom`` (nnz, l_max+1), ``hd`` (nnz, l_max+1),
``n`` (nnz).  Also ``species_*`` arrays (``name``, ``pdg``, ``helicity``, ``mass``) for
the species that occur.  Use :func:`load`.

Species not covered (use MCEq's block, children straight ahead): Lambda, n and nbar
(neutron beta decay), pi0 -> gamma, D mesons, the 1.2e-4 pi -> e nu branch (nue/antinue
from pi), the 7e-4 K_S semileptonic branches, and everything that MCEq lists as a
tracking species (these are copies of their untracked parents).
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_MC3D = os.path.join(os.path.dirname(_HERE), "mc3d")
if _MC3D not in sys.path:
    sys.path.append(_MC3D)

import decays as _dk  # noqa: E402
from constants import MASS, M_K, M_K0, M_MU, M_PI  # noqa: E402

L_MAX = 64
N3 = os.environ.get("N3", "/home/pigrange/mc3d_work/runs/native3d")
DEFAULT_PATH = os.path.join(N3, "kernels", "decay.npz")

# --------------------------------------------------------------------------
# species names (MCEq's)
# --------------------------------------------------------------------------
_NU = {12: "nue", -12: "antinue", 14: "numu", -14: "antinumu"}
_HEL = {-1: "_l", 1: "_r", 0: ""}


def species_name(pdg, hel=0):
    """MCEq name of a species: ``mu+_l``, ``pi-``, ``antinumu`` ..."""
    pdg = int(pdg)
    if pdg in _NU:
        return _NU[pdg]
    if abs(pdg) == 13:
        return ("mu-" if pdg > 0 else "mu+") + _HEL[hel]
    if pdg == 211:
        return "pi+"
    if pdg == -211:
        return "pi-"
    if pdg == 111:
        return "pi0"
    raise KeyError(pdg)


#: parent name -> (pdg of the decaying particle, helicity, mass, decay table)
def _parent_table():
    t = {}
    t["pi+"] = (211, 0, M_PI, None)
    t["pi-"] = (-211, 0, M_PI, None)
    t["K+"] = (321, 0, M_K, _dk.BR_KP)
    t["K-"] = (-321, 0, M_K, _dk.BR_KM)
    t["K_L0"] = (130, 0, M_K0, _dk.BR_KL)
    t["K_S0"] = (310, 0, M_K0, _dk.BR_KS)
    for pdg in (13, -13):
        for h in (-1, 0, 1):
            t[species_name(pdg, h)] = (pdg, h, M_MU, None)
    return t


PARENTS = _parent_table()
#: covered parents, in a fixed order
PARENT_NAMES = ["pi+", "pi-", "K+", "K-", "K_L0", "K_S0",
                "mu+_l", "mu+_r", "mu+", "mu-_l", "mu-_r", "mu-"]
#: parents that are computed; the others are their CP images
COMPUTED_PARENTS = ["K_L0", "K+", "mu+_l", "mu+_r", "mu+", "pi+", "K_S0"]


def _mass(pdg):
    return MASS[int(pdg)]


def _norm_table(table):
    s = sum(b for b, _ in table)
    return [(b / s, ds) for b, ds in table], s


def decay_channels(parent):
    """Normalised decay channels ``[(br, [pdg daughters])]`` of a hadron parent
    (``decays.py`` tables, renormalised as ``decays._pick`` does) and the raw sum."""
    pdg = PARENTS[parent][0]
    if abs(pdg) == 211:
        # decays.decay: pi -> mu nu with probability one
        mu = -13 if pdg > 0 else 13
        nu = 14 if pdg > 0 else -14
        return [(1.0, [mu, nu])], 0.99988
    return _norm_table(PARENTS[parent][3])


# --------------------------------------------------------------------------
# grid and Legendre helpers
# --------------------------------------------------------------------------
class Grid:
    """Kinetic-energy grid (MCEq's)."""

    def __init__(self, e_bins, e_grid=None):
        self.e_bins = np.asarray(e_bins, float)
        self.e_widths = np.diff(self.e_bins)
        self.e_grid = (np.sqrt(self.e_bins[1:] * self.e_bins[:-1])
                       if e_grid is None else np.asarray(e_grid, float))
        self.n = len(self.e_grid)


def legendre_all(x, l_max=L_MAX):
    """``P_l(x)`` for ``l = 0..l_max``: array of shape ``(l_max + 1,) + x.shape``."""
    x = np.asarray(x, float)
    out = np.empty((l_max + 1,) + x.shape)
    out[0] = 1.0
    if l_max >= 1:
        out[1] = x
    for l in range(1, l_max):
        out[l + 1] = ((2 * l + 1) * x * out[l] - l * out[l - 1]) / (l + 1)
    return out


def parent_energy_nodes(grid, mode="centre", n_sub=8):
    """Parent kinetic energies and weights (sum to one) inside every bin.

    Returns ``(T, w)`` with shape ``(n_bins, n_sub)`` (``n_sub = 1`` for "centre").
    """
    lo, hi = grid.e_bins[:-1], grid.e_bins[1:]
    if mode == "centre":
        return grid.e_grid[:, None], np.ones((grid.n, 1))
    xg, wg = np.polynomial.legendre.leggauss(n_sub)
    u = 0.5 * (xg + 1.0)
    wg = 0.5 * wg
    if mode == "lnE":
        T = np.exp(np.log(lo)[:, None] + (np.log(hi) - np.log(lo))[:, None] * u)
        return T, np.broadcast_to(wg, T.shape).copy()
    if mode == "T":
        T = lo[:, None] + (hi - lo)[:, None] * u
        return T, np.broadcast_to(wg, T.shape).copy()
    if mode.startswith("pow"):
        g = float(mode[3:])
        # inverse-CDF nodes of T**-g between the edges, equal weights in u
        if abs(g - 1.0) < 1e-12:
            T = np.exp(np.log(lo)[:, None] + (np.log(hi) - np.log(lo))[:, None] * u)
        else:
            a = (1.0 - g)
            la, ha = lo[:, None] ** a, hi[:, None] ** a
            T = (la + (ha - la) * u) ** (1 / a)
        return T, np.broadcast_to(wg, T.shape).copy()
    raise ValueError(mode)


# --------------------------------------------------------------------------
# two-body decays, analytic
# --------------------------------------------------------------------------
def two_body_cm(m0, m1, m2):
    return _dk.two_body_cm(m0, m1, m2)


def muon_pl_two_body(m_parent, c_star, gam, beta, e_lab, p_perp, p_z, h0):
    """Longitudinal polarisation ``P_L`` of the muon of ``pi/K -> mu nu`` in the lab.

    Vectorised, closed form of ``decays.muon_polarisation``: the rest-frame spin
    4-vector ``s* = h0 (p*/m, E*/m n*)`` is boosted along the parent direction ``z``
    and ``P_L = (m / E_lab) (s_vec . p_hat)``.
    """
    p_star = two_body_cm(m_parent, M_MU, 0.0)
    e_star = np.sqrt(p_star ** 2 + M_MU ** 2)
    s_t = h0 * (e_star / M_MU) * np.sqrt(np.maximum(1.0 - c_star ** 2, 0.0))
    s_zs = h0 * (e_star / M_MU) * c_star
    s0s = h0 * p_star / M_MU
    s_z = gam * (s_zs + beta * s0s)
    p_abs = np.sqrt(p_z ** 2 + p_perp ** 2)
    pl = (M_MU / e_lab) * (s_z * p_z + s_t * p_perp) / np.maximum(p_abs, 1e-300)
    return np.clip(pl, -1.0, 1.0)


def two_body_moments(grid, m0, m1, m2, T_par, w_par, n_gl=128, l_max=L_MAX,
                     h0=None, ur=False):
    """Moments of the children of an isotropic two-body decay ``m0 -> m1 + m2``.

    Parameters
    ----------
    grid : Grid
        Energy grid.
    m0, m1, m2 : float
        Parent and daughter masses.  Child 1 (mass ``m1``) is the one returned.
    T_par, w_par : ndarray (n_bins, n_sub)
        Parent kinetic energies and weights, from :func:`parent_energy_nodes`.
    n_gl : int
        Gauss-Legendre nodes in cos(theta*) per (parent energy, child bin).
    h0 : float or None
        If not None, child 1 is a muon with rest-frame helicity ``h0``; the result is
        then split into the lab helicities.
    ur : bool
        Use ``beta = 1`` with ``gamma = (T + m0) / m0`` and the ultra-relativistic
        limit of the muon polarisation (MCEq's decay tables do this, see the module
        docstring); the default is the exact kinematics.

    Returns
    -------
    ndarray or tuple
        ``yield * <P_l>`` per decay, shape ``(n_bins_child, n_bins_parent, l_max+1)``
        (NOT yet multiplied by the MCEq width ratio).  For ``h0`` not None, a tuple
        ``(left, right)`` of such arrays.
    """
    nb = grid.n
    p_star = two_body_cm(m0, m1, m2)
    e_star = np.sqrt(p_star ** 2 + m1 ** 2)
    xg, wg = np.polynomial.legendre.leggauss(n_gl)
    shape = (nb, nb, l_max + 1)
    out_l = np.zeros(shape)
    out_r = np.zeros(shape) if h0 is not None else None
    edges = grid.e_bins
    for k in range(nb):
        for j in range(T_par.shape[1]):
            wj = w_par[k, j]
            e_p = T_par[k, j] + m0
            gam = e_p / m0
            beta = 1.0 if ur else np.sqrt(max(1.0 - 1.0 / gam ** 2, 0.0))
            bp = beta * p_star
            if bp <= 0:
                continue
            e_lo, e_hi = gam * (e_star - bp), gam * (e_star + bp)
            t_lo, t_hi = e_lo - m1, e_hi - m1
            m_a = int(np.searchsorted(edges, max(t_lo, edges[0]), side="right") - 1)
            m_b = int(np.searchsorted(edges, min(t_hi, edges[-1] * (1 - 1e-15)),
                                      side="right") - 1)
            m_a, m_b = max(m_a, 0), min(m_b, nb - 1)
            if t_hi <= edges[0] or t_lo >= edges[-1]:
                continue
            ms = np.arange(m_a, m_b + 1)
            e_c_lo = np.maximum(edges[ms] + m1, e_lo)
            e_c_hi = np.minimum(edges[ms + 1] + m1, e_hi)
            c_lo = (e_c_lo / gam - e_star) / bp
            c_hi = (e_c_hi / gam - e_star) / bp
            ok = c_hi > c_lo
            if not ok.any():
                continue
            ms, c_lo, c_hi = ms[ok], c_lo[ok], c_hi[ok]
            half = 0.5 * (c_hi - c_lo)[:, None]
            c = half * xg[None, :] + 0.5 * (c_hi + c_lo)[:, None]
            wq = 0.25 * (c_hi - c_lo)[:, None] * wg[None, :]      # includes dc*/2
            e_c = gam * (e_star + bp * c)
            p_z = gam * (p_star * c + beta * e_star)
            p_perp = p_star * np.sqrt(np.maximum(1.0 - c * c, 0.0))
            p_abs = np.sqrt(p_z ** 2 + p_perp ** 2)
            x = np.clip(p_z / np.maximum(p_abs, 1e-300), -1.0, 1.0)
            pls = legendre_all(x, l_max)                       # (L+1, nm, ng)
            if h0 is None:
                out_l[ms, k, :] += wj * np.einsum("lmg,mg->ml", pls, wq)
            else:
                if ur:
                    # ultra-relativistic limit, as in MCEq: P_L is a function of
                    # cos(theta*) only
                    pl = h0 * (c + p_star / e_star) / (1.0 + p_star / e_star * c)
                else:
                    pl = muon_pl_two_body(m0, c, gam, beta, e_c, p_perp, p_z, h0)
                wl = wq * 0.5 * (1.0 - pl)
                wr = wq * 0.5 * (1.0 + pl)
                out_l[ms, k, :] += wj * np.einsum("lmg,mg->ml", pls, wl)
                out_r[ms, k, :] += wj * np.einsum("lmg,mg->ml", pls, wr)
    if h0 is None:
        return out_l
    return out_l, out_r


# --------------------------------------------------------------------------
# sampled rest-frame decays
# --------------------------------------------------------------------------
def _boost_arr(e, p, beta):
    """Boost 4-vectors ``(e, p)`` (arrays, p: (n, 3)) by velocity ``beta`` (n, 3)."""
    b2 = np.sum(beta * beta, axis=1)
    gam = 1.0 / np.sqrt(1.0 - b2)
    bp = np.sum(beta * p, axis=1)
    g2 = np.where(b2 > 0, (gam - 1.0) / np.where(b2 > 0, b2, 1.0), 0.0)
    e_new = gam * (e + bp)
    p_new = p + (g2 * bp + gam * e)[:, None] * beta
    return e_new, p_new


def _random_unit(rng, n):
    c = rng.uniform(-1.0, 1.0, n)
    ph = rng.uniform(0.0, 2.0 * np.pi, n)
    s = np.sqrt(1.0 - c * c)
    return np.stack([s * np.cos(ph), s * np.sin(ph), c], axis=1)


def _kallen_p(m0, m1, m2):
    """Two-body CM momentum, vectorised in ``m0`` (zero below threshold)."""
    m0 = np.asarray(m0, float)
    a = (m0 ** 2 - (m1 + m2) ** 2) * (m0 ** 2 - (m1 - m2) ** 2)
    return np.sqrt(np.maximum(a, 0.0)) / (2.0 * m0)


def sample_three_body_rest(rng, n, m0, masses, dalitz="mc3d"):
    """Rest-frame three-body decays (constant matrix element), vectorised.

    ``dalitz="mc3d"`` reproduces ``decays.three_body``: ``m12`` is drawn uniformly in
    ``m12**2`` and accepted with probability proportional to ``p1* p3`` (so the density
    in ``m12`` is ``m12 p1* p3``).  ``dalitz="flat"`` draws uniformly in ``m12`` before
    the same acceptance (density ``p1* p3 dm12`` = uniform Dalitz plot).

    Returns ``(E, P)``: lists of three arrays, energies ``(n,)`` and momenta ``(n, 3)``.
    """
    m1, m2, m3 = masses
    lo, hi = m1 + m2, m0 - m3
    grid_m = np.linspace(lo, hi, 4001)
    wmax = float(np.max(_kallen_p(grid_m, m1, m2) * _kallen_p(m0, grid_m, m3))) * 1.001
    got = np.empty(0)
    while len(got) < n:
        nn = int(2.5 * (n - len(got))) + 100
        if dalitz == "mc3d":
            m12 = np.sqrt(rng.uniform(lo ** 2, hi ** 2, nn))
        elif dalitz == "flat":
            m12 = rng.uniform(lo, hi, nn)
        else:
            raise ValueError(dalitz)
        w = _kallen_p(m12, m1, m2) * _kallen_p(m0, m12, m3)
        got = np.concatenate([got, m12[rng.random(nn) * wmax <= w]])
    m12 = got[:n]
    p1 = _kallen_p(m12, m1, m2)
    p3 = _kallen_p(m0, m12, m3)
    n3 = _random_unit(rng, n)
    n1 = _random_unit(rng, n)
    e3 = np.sqrt(p3 ** 2 + m3 ** 2)
    e12 = np.sqrt(p3 ** 2 + m12 ** 2)
    b12 = (p3 / e12)[:, None] * n3
    e1r = np.sqrt(p1 ** 2 + m1 ** 2)
    e2r = np.sqrt(p1 ** 2 + m2 ** 2)
    e1, q1 = _boost_arr(e1r, p1[:, None] * n1, b12)
    e2, q2 = _boost_arr(e2r, -p1[:, None] * n1, b12)
    return [e1, e2, e3], [q1, q2, -p3[:, None] * n3]


def sample_michel_rest(rng, n, kind, charge, pol):
    """Rest-frame Michel neutrino: energy ``E*`` and ``cos theta*`` (about the muon
    direction), ``dN ~ n(x) [1 + s A(x) pol cos theta*]`` (``decays.michel_sample``)."""
    x = np.interp(rng.random(n), _dk._CDF[kind], _dk._XE)
    a = _dk.michel_asym(x, kind)
    s = -1.0 if charge > 0 else 1.0
    aa = s * a * pol
    v = rng.random(n)
    small = np.abs(aa) < 1e-9
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(small, 2.0 * v - 1.0,
                     (-1.0 + np.sqrt(np.maximum(
                         1.0 + 2.0 * aa * (2.0 * v - 1.0) + aa * aa, 0.0)))
                     / np.where(small, 1.0, aa))
    return 0.5 * M_MU * x, np.clip(c, -1.0, 1.0)


class _Accumulator:
    """Per-child accumulation of sum P_l over samples, per (child bin, parent bin),
    in two halves."""

    def __init__(self, nb, l_max):
        self.s = np.zeros((2, nb, nb, l_max + 1))
        self.n = np.zeros((2, nb, nb), np.int64)
        self.l_max = l_max
        self.nb = nb

    def add(self, k, mbin, x, half):
        """Add samples: child bin index ``mbin`` (-1 = outside), cos ``x``, half 0/1."""
        ok = mbin >= 0
        if not ok.any():
            return
        mbin, x, half = mbin[ok], x[ok], half[ok]
        p = legendre_all(x, self.l_max)                       # (L+1, n)
        for h in (0, 1):
            sel = half == h
            if not sel.any():
                continue
            mb, pp = mbin[sel], p[:, sel]
            order = np.argsort(mb, kind="stable")
            mb, pp = mb[order], pp[:, order]
            u, start = np.unique(mb, return_index=True)
            sums = np.add.reduceat(pp, start, axis=1)           # (L+1, nu)
            self.s[h, u, k, :] += sums.T
            self.n[h, u, k] += np.diff(np.append(start, len(mb)))


def _child_bins(grid, e_lab, m_c):
    t = e_lab - m_c
    idx = np.searchsorted(grid.e_bins, t, side="right") - 1
    idx[(t < grid.e_bins[0]) | (t >= grid.e_bins[-1])] = -1
    return idx


def sampled_moments(grid, m0, rest_e, rest_p, masses, T_par, w_par, rng,
                    l_max=L_MAX, chunk=100000, ur=False):
    """Boost rest-frame decay samples to every parent bin and bin the children.

    Parameters
    ----------
    m0 : float
        Mass of the decaying parent.
    rest_e, rest_p : lists
        Per child: rest-frame energies ``(n,)`` and momenta ``(n, 3)`` (z is the
        parent's direction of flight).
    masses : list of float
        Child masses.
    T_par, w_par : ndarray (n_bins, n_sub)
        Parent energy nodes.  Each sample draws one node per parent bin with
        probability ``w_par`` (so the in-bin distribution is sampled, not integrated).

    Returns
    -------
    list of ``_Accumulator``
        One per child, with ``s`` = sum of ``P_l`` (2 halves), ``n`` counts.
    """
    nb = grid.n
    ns = len(rest_e[0])
    accs = [_Accumulator(nb, l_max) for _ in rest_e]
    half_all = (np.arange(ns) % 2).astype(np.int8)
    for k in range(nb):
        for a in range(0, ns, chunk):
            b = min(ns, a + chunk)
            nn = b - a
            # a continuous in-bin draw: stratify by the node weights
            if T_par.shape[1] == 1:
                t = np.full(nn, T_par[k, 0])
            else:
                cdf = np.cumsum(w_par[k])
                idx = np.minimum(np.searchsorted(cdf / cdf[-1], rng.random(nn)),
                                 T_par.shape[1] - 1)
                t = T_par[k, idx]
            _bin_chunk(grid, m0, accs, rest_e, rest_p, masses, (k, a, b, t),
                       half_all, ur)
    return accs


def _bin_chunk(grid, m0, accs, rest_e, rest_p, masses, args, half_all, ur=False):
    k, a, b, t = args
    gam = (t + m0) / m0
    beta = np.ones_like(gam) if ur else np.sqrt(1.0 - 1.0 / gam ** 2)
    half = half_all[a:b]
    for acc, e_r, p_r, mc in zip(accs, rest_e, rest_p, masses):
        e_l = gam * (e_r[a:b] + beta * p_r[a:b, 2])
        p_z = gam * (p_r[a:b, 2] + beta * e_r[a:b])
        p_abs = np.sqrt(p_z ** 2 + p_r[a:b, 0] ** 2 + p_r[a:b, 1] ** 2)
        x = np.clip(p_z / np.maximum(p_abs, 1e-300), -1.0, 1.0)
        acc.add(k, _child_bins(grid, e_l, mc), x, half)


# --------------------------------------------------------------------------
# kernel assembly
# --------------------------------------------------------------------------
class Pair:
    """Moments of one (parent, child) pair as dense arrays (n_m, n_k, L+1) in MCEq's
    convention (width-ratio applied)."""

    def __init__(self, nb, l_max):
        self.mom = np.zeros((nb, nb, l_max + 1))
        self.var = np.zeros((nb, nb, l_max + 1))       # estimated variance
        self.hd = np.zeros((nb, nb, l_max + 1))
        self.n = np.zeros((nb, nb), np.int64)


def _add_analytic(pair, arr, br, wr):
    pair.mom += br * arr * wr[:, :, None]


def _add_sampled(pair, acc, br, n_samp, wr):
    """BR-weighted per-decay moments from the two halves."""
    per_half = n_samp / 2.0
    ma = acc.s[0] / per_half
    mb = acc.s[1] / per_half
    pair.mom += br * 0.5 * (ma + mb) * wr[:, :, None]
    pair.hd += br * 0.5 * (ma - mb) * wr[:, :, None]
    pair.n += acc.n.sum(axis=0)


def _task_parent(args):
    """Compute all pairs of one parent species (run in a worker process)."""
    (parent, e_bins, e_grid, parent_mode, n_sub, n_samp, seed, dalitz, n_gl,
     l_max, three_body_scale, ur) = args
    grid = Grid(e_bins, e_grid)
    nb = grid.n
    wr = grid.e_widths[None, :] / grid.e_widths[:, None]   # [m, k] = w_k / w_m
    pdg, hel, m0, _ = PARENTS[parent]
    T_par, w_par = parent_energy_nodes(grid, parent_mode, n_sub)
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    pairs = {}

    def get(child):
        if child not in pairs:
            pairs[child] = Pair(nb, l_max)
        return pairs[child]

    if abs(pdg) == 13:
        charge = +1 if pdg < 0 else -1
        pol = {-1: -1.0, 0: 0.0, 1: 1.0}[hel]
        nu_of = {"numu": -14 if charge > 0 else 14, "nue": 12 if charge > 0 else -12}
        for kind in ("numu", "nue"):
            e_st, c = sample_michel_rest(rng, n_samp, kind, charge, pol)
            st = np.sqrt(np.maximum(1.0 - c * c, 0.0))
            ph = rng.uniform(0, 2 * np.pi, n_samp)
            p_r = np.stack([e_st * st * np.cos(ph), e_st * st * np.sin(ph),
                            e_st * c], axis=1)
            accs = sampled_moments(grid, m0, [e_st], [p_r], [0.0], T_par, w_par, rng,
                                   l_max, ur=ur)
            _add_sampled(get(species_name(nu_of[kind])), accs[0], 1.0, n_samp, wr)
        return parent, pairs
    channels, _ = decay_channels(parent)
    for br, ds in channels:
        masses = [_mass(d) for d in ds]
        if len(ds) == 2:
            for i in (0, 1):
                d = ds[i]
                if abs(d) in (11,) or d == 22:
                    continue
                o = masses[1 - i]
                if abs(d) == 13:
                    h0 = -1.0 if d < 0 else +1.0       # mu+ (PDG -13): helicity -1
                    arr_l, arr_r = two_body_moments(grid, m0, masses[i], o, T_par,
                                                    w_par, n_gl, l_max, h0=h0, ur=ur)
                    _add_analytic(get(species_name(d, -1)), arr_l, br, wr)
                    _add_analytic(get(species_name(d, +1)), arr_r, br, wr)
                else:
                    arr = two_body_moments(grid, m0, masses[i], o, T_par, w_par,
                                           n_gl, l_max, ur=ur)
                    _add_analytic(get(species_name(d)), arr, br, wr)
        else:
            ns = int(n_samp * three_body_scale)
            e_r, p_r = sample_three_body_rest(rng, ns, m0, masses, dalitz)
            keep = [i for i, d in enumerate(ds) if abs(d) != 11 and d != 22]
            accs = sampled_moments(grid, m0, [e_r[i] for i in keep],
                                   [p_r[i] for i in keep],
                                   [masses[i] for i in keep], T_par, w_par, rng,
                                   l_max, ur=ur)
            for acc, i in zip(accs, keep):
                d = ds[i]
                _add_sampled(get(species_name(d, 0)), acc, br, ns, wr)
    return parent, pairs


def build(grid=None, parent_mode="centre", n_sub=8, n_samples=1000000, seed=20261118,
          dalitz="mc3d", n_gl=128, l_max=L_MAX, parents=None, workers=2,
          three_body_scale=1.0, verbose=True, ur=False, cp=True):
    """Build all kernels.  Returns a dict ``(parent, child) -> Pair`` and the grid.

    ``grid`` is a :class:`Grid` (default: MCEq's, via :func:`mceq_grid`).  ``workers``
    must not exceed the core budget (2).  With ``cp`` the kernels of pi-, K- and mu-
    are the CP images of those of pi+, K+ and mu+ (same decay tables in ``decays.py``;
    CP flips the muon helicity and conjugates the children); they share the
    statistical fluctuations of their partner.  ``ur`` selects MCEq's ultra-relativistic
    approximation (see :func:`two_body_moments`)."""
    if grid is None:
        grid = mceq_grid()
    parents = list(COMPUTED_PARENTS if parents is None else parents)
    tasks = [(p, grid.e_bins, grid.e_grid, parent_mode, n_sub, n_samples,
              seed + 1000 * i, dalitz, n_gl, l_max, three_body_scale, ur)
             for i, p in enumerate(parents)]
    t0 = time.time()
    res = {}
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            for p, pairs in pool.imap_unordered(_task_parent, tasks):
                res[p] = pairs
                if verbose:
                    print(f"  {p:6s} done ({time.time() - t0:6.0f} s)", flush=True)
    else:
        for t in tasks:
            p, pairs = _task_parent(t)
            res[p] = pairs
            if verbose:
                print(f"  {p:6s} done ({time.time() - t0:6.0f} s)", flush=True)
    out = {}
    for p in parents:
        for c, pr in res[p].items():
            out[(p, c)] = pr
            if cp and cp_conjugate_name(p) in PARENTS and cp_conjugate_name(p) != p:
                out[(cp_conjugate_name(p), cp_conjugate_name(c))] = pr
    return grid, out


def cp_conjugate_name(name):
    """CP-conjugate species name (``mu+_l -> mu-_r``: parity flips the helicity;
    ``numu -> antinumu``; ``pi+ -> pi-``; neutral kaons are unchanged)."""
    if name.startswith("anti"):
        return name[4:]
    if name in ("nue", "numu", "nutau"):
        return "anti" + name
    flip = {"_l": "_r", "_r": "_l"}
    base, suffix = name, ""
    if name.endswith(("_l", "_r")):
        base, suffix = name[:-2], flip[name[-2:]]
    if base.endswith("+"):
        return base[:-1] + "-" + suffix
    if base.endswith("-"):
        return base[:-1] + "+" + suffix
    return name


# --------------------------------------------------------------------------
# MCEq access
# --------------------------------------------------------------------------
def mceq_grid(e_min=0.03):
    """MCEq's kinetic-energy grid (builds an ``MCEqRun`` and reads its grid).

    The default ``e_min = 0.03`` GeV is the PHYS grid (126 bins); the REF grid
    (``e_min = 0.1``, 121 bins) is its last 121 bins, so the PHYS grid serves both."""
    ref = mceq_reference(blocks=False, e_min=e_min)
    return Grid(ref["e_bins"], ref["e_grid"])


def mceq_reference(blocks=True, e_min=0.03, interaction_model="SIBYLL23D"):
    """Grid, species and raw decay blocks straight from MCEq.

    Returns a dict with ``e_grid, e_bins, e_widths``; ``species`` (list of
    ``(name, pdg, helicity, mass, ctau, is_tracking)`` in MCEq order); ``mix_idx``
    (name -> first hadronic bin); ``blocks`` ((parent, child) name -> raw
    ``decay_dists`` matrix ``D[m, k]``, no ``1/lambda_dec``); ``dec_m`` (the sparse
    decay matrix) and ``db`` (the interaction database file loaded).
    """
    import importlib.util  # noqa: F401
    import MCEq.config as cfg
    cfg.e_min = e_min
    for p in (os.path.join(os.path.dirname(_HERE), "mceq3d"),):
        if p not in sys.path:
            sys.path.append(p)
    from mceq3d_db import make_run
    from mceq3d_primary import primary_class
    run = make_run(interaction_model=interaction_model,
                   primary_model=(primary_class("HillasGaisser2012"), "H3a"),
                   theta_deg=0.0)
    cps = list(run.pman.cascade_particles)
    ref = dict(e_grid=np.array(run.e_grid), e_bins=np.array(run.e_bins),
               e_widths=np.array(run.e_widths),
               species=[(p.name, int(p.pdg_id[0]), int(p.helicity), float(p.mass),
                         float(p.ctau), bool(p.is_tracking)) for p in cps],
               db=str(getattr(run, "mceq3d_db", "")))
    ref["mix_idx"] = {p.name: int(p.mix_idx) for p in cps}
    if blocks:
        bl = {}
        for p in cps:
            if p.is_stable or p.is_tracking:
                continue
            for c in p.children:
                if c.is_tracking:
                    continue
                bl[(p.name, c.name)] = np.asarray(p.decay_dists[c])
        ref["blocks"] = bl
        ref["dec_m"] = run.dec_m
    return ref


# --------------------------------------------------------------------------
# save / load
# --------------------------------------------------------------------------
def save(path, grid, pairs, meta=None, l_max=L_MAX):
    """Write kernels to ``path`` (npz, no pickles, written atomically)."""
    names, ks, ms, moms, hds, ns, ptr = [], [], [], [], [], [], [0]
    for (parent, child), pr in sorted(pairs.items()):
        nz = np.nonzero(np.any(pr.mom != 0.0, axis=2))
        m, k = nz
        names.append((parent, child))
        ks.append(k.astype(np.int32))
        ms.append(m.astype(np.int32))
        moms.append(pr.mom[m, k, :])
        hds.append(pr.hd[m, k, :].astype(np.float32))
        ns.append(pr.n[m, k].astype(np.int32))
        ptr.append(ptr[-1] + len(k))
    species = sorted({c for _, c in names} | {p for p, _ in names})
    info = _species_info(species)
    d = dict(
        e_grid=grid.e_grid, e_bins=grid.e_bins, e_widths=grid.e_widths,
        l_max=np.int64(l_max),
        pair_parent=np.array([a for a, _ in names]),
        pair_child=np.array([b for _, b in names]),
        pair_ptr=np.array(ptr, np.int64),
        k=np.concatenate(ks), m=np.concatenate(ms),
        mom=np.concatenate(moms), hd=np.concatenate(hds), n=np.concatenate(ns),
        species_name=np.array(species),
        species_pdg=np.array([info[s][0] for s in species], np.int64),
        species_helicity=np.array([info[s][1] for s in species], np.int64),
        species_mass=np.array([info[s][2] for s in species]),
        meta_json=np.array(json.dumps(meta or {}, sort_keys=True)),
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **d)
    os.replace(tmp, path)


def _species_info(names):
    out = {}
    for n in names:
        if n in PARENTS:
            pdg, hel, m, _ = PARENTS[n]
            out[n] = (pdg, hel, m)
            continue
        hel = -1 if n.endswith("_l") else (1 if n.endswith("_r") else 0)
        base = n.split("_")[0] if n.endswith(("_l", "_r")) else n
        pdg = {"nue": 12, "antinue": -12, "numu": 14, "antinumu": -14, "pi+": 211,
               "pi-": -211, "pi0": 111, "mu+": -13, "mu-": 13}[base]
        out[n] = (pdg, hel, MASS.get(pdg, 0.0))
    return out


class DecayKernels:
    """Loaded decay kernels (see the module docstring for the conventions)."""

    def __init__(self, d):
        self.e_grid = d["e_grid"]
        self.e_bins = d["e_bins"]
        self.e_widths = d["e_widths"]
        self.l_max = int(d["l_max"])
        self.meta = json.loads(str(d["meta_json"]))
        self.species = [str(s) for s in d["species_name"]]
        self.pdg = d["species_pdg"]
        self.helicity = d["species_helicity"]
        self.mass = d["species_mass"]
        self._parent = [str(s) for s in d["pair_parent"]]
        self._child = [str(s) for s in d["pair_child"]]
        self._ptr = d["pair_ptr"]
        self._d = {k: d[k] for k in ("k", "m", "mom", "hd", "n")}
        self.pairs = list(zip(self._parent, self._child))
        self._index = {pc: i for i, pc in enumerate(self.pairs)}

    @property
    def n_bins(self):
        return len(self.e_grid)

    def entries(self, parent, child):
        """``(k, m, mom, hd, n)`` of a pair (sparse entries, MCEq convention)."""
        i = self._index[(parent, child)]
        s = slice(self._ptr[i], self._ptr[i + 1])
        return tuple(self._d[x][s] for x in ("k", "m", "mom", "hd", "n"))

    def dense(self, parent, child, what="mom"):
        """Dense array ``(l_max + 1, n_m, n_k)`` of ``mom`` or ``hd``."""
        k, m, mom, hd, _ = self.entries(parent, child)
        a = np.zeros((self.l_max + 1, self.n_bins, self.n_bins))
        a[:, m, k] = (mom if what == "mom" else hd).T
        return a

    def block(self, parent, child):
        """The ``l = 0`` block ``D[m, k]``: comparable with MCEq's ``decay_dists``."""
        return self.dense(parent, child)[0]


def load(path=None):
    """Load ``decay.npz`` (default: ``$N3/kernels/decay.npz``)."""
    with np.load(path or DEFAULT_PATH, allow_pickle=False) as d:
        return DecayKernels({k: d[k] for k in d.files})


def make_meta(grid, **kw):
    """Metadata dict stored with the kernels."""
    ref_sums = {p: (decay_channels(p)[1] if p in PARENT_NAMES[:6] else 1.0)
                for p in PARENT_NAMES[:6]}
    meta = dict(
        module="cascade3d/kernels_decay.py", created=time.strftime("%Y-%m-%d %H:%M"),
        convention="mom[l,m,k] = (w_k/w_m) * sum_decays P_l(cos theta_lab) / N_decays "
                   "incl. BR; kinetic-energy grid; l=0 equals MCEq decay_dists",
        raw_br_sums=ref_sums, br_renormalised=True, n_bins=int(grid.n),
        not_covered=["Lambda", "n", "nbar", "pi0 -> gamma", "D mesons",
                     "pi -> e nu (1.2e-4)", "K_S semileptonic (7e-4)",
                     "tracking species"],
    )
    meta.update(kw)
    return meta


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=DEFAULT_PATH)
    ap.add_argument("--n-samples", type=int, default=1000000,
                    help="rest-frame samples per muon decay (and per 3-body channel "
                    "times --three-body-scale)")
    ap.add_argument("--three-body-scale", type=float, default=0.5)
    ap.add_argument("--parent-mode", default="centre")
    ap.add_argument("--dalitz", default="mc3d", choices=["mc3d", "flat"])
    ap.add_argument("--ur", action="store_true",
                    help="MCEq's ultra-relativistic approximation (diagnostic)")
    ap.add_argument("--parents", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261118)
    a = ap.parse_args(argv)
    grid = mceq_grid()
    grid, pairs = build(grid, parent_mode=a.parent_mode, n_samples=a.n_samples,
                        dalitz=a.dalitz, workers=a.workers, ur=a.ur, seed=a.seed,
                        three_body_scale=a.three_body_scale, parents=a.parents)
    meta = make_meta(grid, parent_mode=a.parent_mode, n_samples=a.n_samples,
                     three_body_scale=a.three_body_scale, dalitz=a.dalitz, n_sub=8,
                     n_gl=128, seed=a.seed, ur=a.ur, parents=a.parents)
    save(a.out, grid, pairs, meta)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
