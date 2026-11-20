"""Kernel sets matched to the Monte Carlo (``tools/mc3d``) for G1.3 and G1.4 (WP 1.3).

Gate G1.3 compares the solver with a field-free run of ``tools/mc3d/sky3d.py``
like for like: the solver must be given the physics the Monte Carlo used, so that
only the transport method differs.  This module builds that physics as a
:class:`kernelset.KernelSet` (contract: ``DESIGN.md`` section 3) from

* a generator kernel file of ``kernels_gen`` (``gen_pool2000``: the Monte Carlo's
  own event pools, G1.3(a); ``gen_fresh1e5``: 10^5 fresh events per node, G1.3(c);
  its two disjoint halves of even- and odd-numbered events, G1.4),
* the decay kernels of ``kernels_decay`` (``decay.npz``) and the Lambda kernels made
  here with the same machinery (``decay_extra.npz``, :func:`build_decay_extra`),
* the Monte Carlo's own functions, called, not copied: interaction lengths
  (``interactions.ChromoBackend(xs="mceq").lambda_int``), lifetimes and masses
  (``constants``), the tracking thresholds (``shower.thresholds``), the muon
  stopping power (``shower.dedx_air``), the primary flux (``primaries``,
  ``sky.STRATA``).

Rules (Monte Carlo source -> realisation here; ``LIKE_FOR_LIKE.md`` has the table)
------------------------------------------------------------------------------
M1  Species (``constants.TRACKED``, ``pool.KEEP``): p, pbar, n, nbar, pi+-, K+-,
    K_L, K_S, Lambda, Lambdabar (hadrons), mu+- in MCEq's helicity states
    (loss species), the four neutrinos (sinks).  Everything else has no species,
    so every yield into it is zero (pi0, gamma, e, hyperons, charm, fragments).
M2  Every hadron is transported and decays in flight at all energies
    (``shower.transport_hadron``): ``active`` is True everywhere, nothing folded.
M3  Interaction rate: ``1 / ChromoBackend.lambda_int(pdg, T + m)`` at the bin
    centre (MCEq's SIBYLL-2.3d cross sections, log-log in kinetic energy, zero
    below the table's threshold).  Muons and neutrinos never interact.
    Turn-on of the nucleon cross section (``turn_on=True``, the default; C1
    closing cycle).  For p and n the table is zero up to one node and full from
    the next, so the Monte Carlo's rate rises by many decades inside one bin k*
    (found from the function, :func:`turn_on_bin`: k* = 17, T = 1.26 to 1.58
    GeV).  With P(T) = 1 - exp(-X_ref / lambda(T)), X_ref = 2000 g/cm2
    (:data:`X_REF`), integrals over bin k* in T:

    * every source into (s, k*) is multiplied by the interacting fraction of
      that source: ``phi_top[s, k*]`` by f_prim = int J_s P / int J_s (J_s the
      primary flux per nucleon, :func:`mc_primary_flux`), the row k* of every
      interaction and decay channel with child s by f_sec = int T^-g P /
      int T^-g, g = g_s of :data:`G_SEC` (diagonal entries of s -> s included);
    * ``inv_lambda[s, k*]`` = a_s = (int J_s / lambda / int J_s) / f_prim, the
      interaction rate per g/cm2 of the bin's population at birth;
    * the yields out of (s, k*) and everything else are unchanged.

    Consequence: the nucleon flux of bin k* that the solver returns is the
    interacting part only, and the number sum rule of the channels into
    (s, k*) is short by 1 - f on purpose.  One fraction is exact at X_ref only
    (the ramp is steep but not a step).  The numbers are in ``meta["turn_on"]``.
    ``turn_on=False`` reproduces the sets of 2026-11-19 array by array.
M4  Floor (``pool.EventPool.draw``): below 2 GeV total energy an interaction
    yields nothing; the particle is still removed (zero yield, rate kept).
M5  Interaction yields and angles: the generator kernel's ``c_l`` (rules R1-R11 of
    ``kernels_gen``: pool nodes, node mixing, rescaling, guard, kept species).
    Generator muons have polarisation 0 -> helicity-0 states ``mu+``/``mu-``.
M6  Decay rate ``m / (p c tau)`` with exact ``p`` (``transport_hadron``,
    ``transport_muon``); neutrons decay into nothing (``decays.decay`` returns no
    daughter): pure absorption.
M7  Decay yields and angles: ``decay.npz`` (exact kinematics, ``decays.py``
    channels and sampler, exact muon polarisation as helicity split) plus
    ``decay_extra.npz`` (Lambda -> p pi-, n pi0 and the CP image).
M8  Tracking thresholds ``shower.thresholds(e_nu_min, scheme)``: a child whose bin
    centre lies below its species' threshold is given zero yield (only nucleons
    below T = 0.106 GeV are affected for ``chain`` and e_nu_min = 0.1; they have
    no interaction and no decay product, so this changes no flux).
M9  Continuous energy loss for muons only, ``dedx = shower.dedx_air(T + m_mu)``
    (``loss=False`` gives ``dedx = 0``).  No loss for hadrons, no multiple
    scattering (the Monte Carlo has none).
M10 Primaries: GSF p and n (``primaries.GSFGroups``, model ``GSFG:GH``, energy
    convention resolved as ``sky3d`` does: ``total``), kinetic energy per nucleon
    from ``sky.STRATA[0]`` to ``sky.STRATA[nstrata]`` (1 to 3162 GeV for
    ``--nstrata 7``), nothing outside; ``phi_top`` is the bin average of the flux
    over the part of each bin inside that range (default) or its centre value.

Angular laws: delta-M at order ``L`` (default 32) with ``g`` the mean of
``a_{L+1} .. a_{L+n_tail}`` (``kernels_gen.delta_m``, ``n_tail = 4``), for every
interaction and decay channel and every (parent, child) bin pair; ``a*`` is clipped
to [-1, 1] where the noise pushes it out (counted in ``meta``).

Usage::

    python kernelset_mc.py decay-extra
    python kernelset_mc.py build --gen pool2000 [--L 32] [--loss off]
    python kernelset_mc.py build --gen all            # the four G1.3/G1.4 sets
    python kernelset_mc.py build --gen pool2000 --turn-on off   # 2026-11-19 rule

Sets with the turn-on rule go to ``$N3/kernels/mc_close`` by default, sets
without it to ``$N3/kernels/mc`` (the file names do not carry the rule).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_MC3D = os.path.join(os.path.dirname(_HERE), "mc3d")
if _MC3D not in sys.path:
    sys.path.append(_MC3D)

import kernels_decay as kd  # noqa: E402
import kernels_gen as kg  # noqa: E402
from kernelset import (KIND_DEC, KIND_INT, ROLE_HADRON, ROLE_LOSS,  # noqa: E402
                       ROLE_SINK, KernelSet)

import constants as mcc  # noqa: E402  (tools/mc3d)
import decays as mcdec  # noqa: E402
import primaries as mcprim  # noqa: E402
import shower as mcshower  # noqa: E402

VERSION = "kernelset_mc-1.1"
N3 = os.environ.get("N3", "/home/pigrange/mc3d_work/runs/native3d")
KERNEL_DIR = os.path.join(N3, "kernels")
OUT_DIR = os.path.join(KERNEL_DIR, "mc")
OUT_DIR_CLOSE = os.path.join(KERNEL_DIR, "mc_close")

#: slant depth [g/cm2] at which the interacting fraction of rule M3 is taken
#: (cos Z = 0.5, the middle of the cells of G1.3); fixed in the brief of the
#: C1 closing cycle, not adjusted
X_REF = 2000.0
#: species with the turn-on rule of M3
TURN_ON_SPECIES = ("p+", "n0")
#: local logarithmic slope g_s of dN/dT of the secondary production of p and n
#: around the turn-on bin (bins k*-1, k*, k*+1): :func:`secondary_slope` of the
#: first generation of the primaries in the pool2000 set of 2026-11-19
#: (``kernels/mc/ks_mc_pool2000_L32_losson.npz``, f_prim in bin k*), rounded to
#: four digits (0.91256, 0.87071); the same two numbers for every set.  All
#: baryon generations without decays give 1.30 and 1.31 (cross-check only).
G_SEC = {"p+": 0.9126, "n0": 0.8707}
DECAY_PATH = os.path.join(KERNEL_DIR, "decay.npz")
DECAY_EXTRA_PATH = os.path.join(KERNEL_DIR, "decay_extra.npz")

#: generator sets: name -> (kernels_gen tag, half index or None)
GEN_SETS = {"pool2000": ("pool2000", None), "fresh1e5": ("fresh1e5", None),
            "fresh1e5_h0": ("fresh1e5", 0), "fresh1e5_h1": ("fresh1e5", 1)}

#: (MCEq name, pdg, helicity, role) in the order of the kernel set
SPECIES = (
    ("p+", 2212, 0, ROLE_HADRON), ("pbar-", -2212, 0, ROLE_HADRON),
    ("n0", 2112, 0, ROLE_HADRON), ("nbar0", -2112, 0, ROLE_HADRON),
    ("pi+", 211, 0, ROLE_HADRON), ("pi-", -211, 0, ROLE_HADRON),
    ("K+", 321, 0, ROLE_HADRON), ("K-", -321, 0, ROLE_HADRON),
    ("K_L0", 130, 0, ROLE_HADRON), ("K_S0", 310, 0, ROLE_HADRON),
    ("Lambda0", 3122, 0, ROLE_HADRON), ("Lambdabar0", -3122, 0, ROLE_HADRON),
    ("mu+_l", -13, -1, ROLE_LOSS), ("mu+_r", -13, 1, ROLE_LOSS),
    ("mu+", -13, 0, ROLE_LOSS), ("mu-_l", 13, -1, ROLE_LOSS),
    ("mu-_r", 13, 1, ROLE_LOSS), ("mu-", 13, 0, ROLE_LOSS),
    ("nue", 12, 0, ROLE_SINK), ("antinue", -12, 0, ROLE_SINK),
    ("numu", 14, 0, ROLE_SINK), ("antinumu", -14, 0, ROLE_SINK),
)
NAMES = tuple(s[0] for s in SPECIES)
_IDX = {n: i for i, n in enumerate(NAMES)}

#: Monte Carlo defaults of the G1.3 reference run (``c1/mcref/PLAN_CONFIG.txt``)
MC_DEFAULTS = {"model": "GSFG:GH", "nstrata": 7, "e_nu_min": 0.1,
               "thr_scheme": "chain", "primary_energy": None}

#: Lambda decay channels of ``decays.py`` (renormalised as ``decays._pick`` does)
_LAMBDA_TABLES = {"Lambda0": (3122, mcdec.BR_LAM),
                  "Lambdabar0": (-3122, mcdec.BR_LAMBAR)}
_EXTRA_NAMES = {2212: "p+", -2212: "pbar-", 2112: "n0", -2112: "nbar0",
                211: "pi+", -211: "pi-", 111: "pi0"}


# ---------------------------------------------------------------------------
# the Monte Carlo's rules, from its own functions
# ---------------------------------------------------------------------------
def mc_backend():
    """The interaction backend of ``sky3d`` (``ChromoBackend(xs="mceq")``), without
    a pool directory: only its interaction lengths are used."""
    import interactions as mci

    return mci.ChromoBackend(cache_dir=None, xs="mceq", readonly=True)


def mc_inv_lambda(pdg, e_kin, backend=None):
    """``1 / lambda_int`` [cm^2/g] of the Monte Carlo at kinetic energies ``e_kin``.

    ``ChromoBackend.lambda_int`` (MCEq table, log-log in kinetic energy) where
    ``has_interaction``; 0 where the length is infinite or the species never
    interacts.
    """
    be = backend or mc_backend()
    m = mcc.MASS[int(pdg)]
    out = np.zeros(np.size(e_kin))
    if not be.has_interaction(int(pdg)):
        return out
    for i, t in enumerate(np.atleast_1d(e_kin)):
        lam = be.lambda_int(int(pdg), float(t) + m)
        out[i] = 0.0 if not np.isfinite(lam) else 1.0 / lam
    return out


def mc_inv_decay(pdg, e_kin):
    """Decay rate per cm, ``m / (p c tau)``, as ``shower.transport_hadron`` and
    ``transport_muon`` sample it (exact momentum); 0 for stable species."""
    ctau = mcc.CTAU_CM.get(int(pdg), np.inf)
    m = mcc.MASS.get(int(pdg), 0.0)
    t = np.atleast_1d(np.asarray(e_kin, float))
    if not np.isfinite(ctau) or m <= 0.0:
        return np.zeros(t.size)
    p = np.sqrt(t * (t + 2.0 * m))
    return m / (p * ctau)


def mc_thresholds_kinetic(e_nu_min=0.1, scheme="chain"):
    """Tracking thresholds of ``shower.thresholds`` as kinetic energies [GeV]."""
    thr = mcshower.thresholds(e_nu_min, scheme)
    return {int(p): float(e) - mcc.MASS[int(p)] for p, e in thr.items()}


def mc_dedx(e_kin):
    """Muon stopping power ``shower.dedx_air`` [GeV cm^2/g] at kinetic energies."""
    return np.array([mcshower.dedx_air(float(t) + mcc.M_MU)
                     for t in np.atleast_1d(e_kin)])


def mc_primary_range(nstrata=MC_DEFAULTS["nstrata"]):
    """Kinetic energy per nucleon range of ``sky.build_tasks`` with ``nstrata``."""
    import sky

    ns = len(sky.STRATA) - 1 if nstrata is None else int(nstrata)
    return float(sky.STRATA[0]), float(sky.STRATA[ns])


def mc_primary_flux(t, model=MC_DEFAULTS["model"], convention=None):
    """``(J_p, J_n)`` [1/(GeV cm^2 s sr)] per kinetic energy per nucleon ``t``.

    The sum over the Monte Carlo's five groups of ``primaries.group_split`` (what
    ``sky.primary_sample`` samples: group intensity times its proton fraction),
    with the energy convention resolved exactly as ``sky3d.Options`` does.
    """
    mdl = mcprim.primary_model(model)
    conv = mcprim.resolve_convention(convention or mcprim.ENERGY_CONVENTION,
                                     name=model)
    t = np.atleast_1d(np.asarray(t, float))
    jp, jn = np.zeros(t.size), np.zeros(t.size)
    for _n, _a, _z, gid in mcprim.SPECIES:
        p, n = mcprim.group_split(mdl, gid, t, conv)
        jp, jn = jp + p, jn + n
    return jp, jn


def phi_top_mc(e_bins, model=MC_DEFAULTS["model"], nstrata=MC_DEFAULTS["nstrata"],
               mode="average", n_gl=16, e_range=None):
    """``phi_top`` rows of p and n on a kinetic grid (``mode`` "average" or
    "centre"); zero outside the Monte Carlo's primary range."""
    lo_r, hi_r = mc_primary_range(nstrata) if e_range is None else e_range
    e_bins = np.asarray(e_bins, float)
    nE = e_bins.size - 1
    out = np.zeros((2, nE))
    xg, wg = np.polynomial.legendre.leggauss(n_gl)
    for k in range(nE):
        a, b = max(e_bins[k], lo_r), min(e_bins[k + 1], hi_r)
        if b <= a * (1 + 1e-12):
            continue
        if mode == "centre":
            c = np.sqrt(e_bins[k] * e_bins[k + 1])
            if lo_r <= c <= hi_r:
                jp, jn = mc_primary_flux(c, model)
                out[:, k] = jp[0], jn[0]
            continue
        u = np.log(a) + (np.log(b) - np.log(a)) * 0.5 * (xg + 1.0)
        t = np.exp(u)
        w = 0.5 * (np.log(b) - np.log(a)) * wg * t        # dT = T d ln T
        jp, jn = mc_primary_flux(t, model)
        width = e_bins[k + 1] - e_bins[k]
        out[0, k] = np.sum(w * jp) / width
        out[1, k] = np.sum(w * jn) / width
    return out


# ---------------------------------------------------------------------------
# the turn-on of the nucleon cross section (rule M3)
# ---------------------------------------------------------------------------
def mc_interaction_prob(pdg, e_kin, x, backend=None):
    """Probability that a hadron of kinetic energy ``e_kin`` interacts over ``x``
    g/cm2 in the Monte Carlo (no energy loss of hadrons): 1 - exp(-x/lambda)."""
    return -np.expm1(-x * mc_inv_lambda(pdg, e_kin, backend))


def turn_on_bin(pdg, e_bins, x_ref=X_REF, tol=0.01, backend=None):
    """The bin in which the Monte Carlo's interaction probability over ``x_ref``
    rises by more than ``tol`` between the bin edges (the turn-on bin).

    Returns the bin index, or None if there is no such bin; raises
    ``ValueError`` if there are several.
    """
    pe = mc_interaction_prob(pdg, np.asarray(e_bins, float), x_ref, backend)
    ks = np.flatnonzero(np.diff(pe) > tol)
    if ks.size == 0:
        return None
    if ks.size > 1:
        raise ValueError(f"pdg {pdg}: interaction probability rises in bins {ks}")
    return int(ks[0])


def bin_quadrature(lo, hi, breaks=(), n_panel=64, n_gl=8):
    """Nodes ``t`` and weights ``w`` (for dT) of a composite Gauss-Legendre rule
    in ln T over [lo, hi], with panel boundaries at the ``breaks`` inside.

    The Monte Carlo's cross section is linear in log-log between its table nodes,
    so splitting at the nodes leaves only smooth (if steep) integrands.
    """
    cuts = [lo] + sorted(b for b in breaks if lo < b < hi) + [hi]
    xg, wg = np.polynomial.legendre.leggauss(n_gl)
    ts, ws = [], []
    for a, b in zip(cuts[:-1], cuts[1:]):
        edges = np.linspace(np.log(a), np.log(b), n_panel + 1)
        h = np.diff(edges)
        u = edges[:-1, None] + 0.5 * h[:, None] * (xg[None, :] + 1.0)
        t = np.exp(u)
        ts.append(t.ravel())
        ws.append((0.5 * h[:, None] * wg[None, :] * t).ravel())   # dT = T dlnT
    return np.concatenate(ts), np.concatenate(ws)


def _table_nodes(backend):
    return np.asarray(backend._cs.e_grid, float)


def turn_on_factors(name, e_bins, g, x_ref=X_REF, model=MC_DEFAULTS["model"],
                    backend=None, dg=(-0.5, 0.5)):
    """Interacting fractions and rate of rule M3 for nucleon species ``name``.

    Parameters
    ----------
    name : {"p+", "n0"}
        Species.
    e_bins : array
        Kinetic bin edges of the set [GeV].
    g : float
        Slope g_s of the secondary weight T^-g.
    x_ref : float
        Slant depth [g/cm2] of the interacting fraction.
    model : str
        Primary model (J_s).
    backend : interactions.ChromoBackend, optional
    dg : tuple of float
        Offsets of g at which f_sec is also reported.

    Returns
    -------
    dict
        ``k``, ``T_range``, ``f_prim``, ``f_sec``, ``a`` (rate [cm2/g]),
        ``inv_lambda_centre`` (the rule-off rate), ``g_sec``, ``x_ref`` and
        ``f_sec_at_g`` ({g + dg: f_sec}).
    """
    be = backend or mc_backend()
    pdg = SPECIES[_IDX[name]][1]
    if pdg not in (2212, 2112):
        raise ValueError("the turn-on rule is for p and n only")
    e_bins = np.asarray(e_bins, float)
    k = turn_on_bin(pdg, e_bins, x_ref, backend=be)
    if k is None:
        raise ValueError(f"{name}: no turn-on bin on this grid")
    lo, hi = e_bins[k], e_bins[k + 1]
    t, w = bin_quadrature(lo, hi, _table_nodes(be))
    il = mc_inv_lambda(pdg, t, be)
    P = -np.expm1(-x_ref * il)
    jp, jn = mc_primary_flux(t, model)
    J = jp if pdg == 2212 else jn
    f_prim = float(np.sum(w * J * P) / np.sum(w * J))
    a = float(np.sum(w * J * il) / np.sum(w * J)) / f_prim

    def f_sec_of(gg):
        ws = w * t ** (-gg)
        return float(np.sum(ws * P) / np.sum(ws))

    centre = float(mc_inv_lambda(pdg, np.sqrt(lo * hi), be)[0])
    return {"k": k, "T_range": [float(lo), float(hi)], "f_prim": f_prim,
            "f_sec": f_sec_of(g), "a": a, "inv_lambda_centre": centre,
            "g_sec": float(g), "x_ref": float(x_ref),
            "f_sec_at_g": {f"{g + d:.4f}": f_sec_of(g + d) for d in dg}}


def secondary_slope(ks, name, k=None, x_ref=X_REF, f_prim=None):
    """Local logarithmic slope g of dN/dT of the first-generation production of
    species ``name`` by the primaries of the set ``ks``, around bin ``k``.

    Production density in child bin m: S(m) = sum over the interaction channels
    q -> name and parent bins k' of ``Y[m, k'] phi_top[q, k'] P_q(k')``, with
    P_q(k') = 1 - exp(-x_ref inv_lambda[q, k']) the probability that the
    primary interacts over ``x_ref`` (``f_prim`` replaces it in the turn-on bin
    when given: {q name: f}).  g = -d ln S / d ln T from the bins k-1, k, k+1:
    the least-squares slope of ln S against ln T_centre over the three bins,
    which on the uniform ln T grid is the central difference
    ln(S[k-1] / S[k+1]) / ln(T[k+1] / T[k-1]).
    """
    d = ks.d
    c = ks.index(name)
    if k is None:
        k = turn_on_bin(SPECIES[_IDX[name]][1], d["e_bins"], x_ref)
    S = np.zeros(ks.nE)
    for j in np.flatnonzero((d["ch_child"] == c) & (d["ch_kind"] == KIND_INT)):
        q = int(d["ch_parent"][j])
        P = -np.expm1(-x_ref * np.asarray(d["inv_lambda"][q], float))
        if f_prim is not None and str(d["species"][q]) in f_prim:
            P = P.copy()
            P[k] = f_prim[str(d["species"][q])]
        S += d["Y"][j] @ (d["phi_top"][q] * P)
    E = d["e_grid"]
    return float(np.log(S[k - 1] / S[k + 1]) / np.log(E[k + 1] / E[k - 1]))


# ---------------------------------------------------------------------------
# decay kernels of Lambda (decay_extra.npz)
# ---------------------------------------------------------------------------
def _extra_species_info(names):
    info = {}
    for n in names:
        if n in _LAMBDA_TABLES:
            pdg = _LAMBDA_TABLES[n][0]
        else:
            pdg = {v: k for k, v in _EXTRA_NAMES.items()}[n]
        info[n] = (pdg, 0, mcc.MASS[pdg])
    return info


def build_decay_extra(path=DECAY_EXTRA_PATH, n_gl=128, l_max=kd.L_MAX,
                      parent_mode="centre", grid=None):
    """Lambda and Lambdabar decay kernels with the machinery of ``kernels_decay``.

    Channels and branching ratios of ``decays.py`` (``BR_LAM``, ``BR_LAMBAR``),
    renormalised to one as ``decays._pick`` does; both channels are isotropic
    two-body decays (``decays._finish``), computed analytically by
    ``kernels_decay.two_body_moments`` on the grid of ``decay.npz`` (parent at the
    bin centre, as ``decay.npz``).  pi0 children are kept in the file (as in
    ``decay.npz``) and dropped by the kernel-set builder.  The file has the layout
    of ``decay.npz`` and loads with ``kernels_decay.load``.
    """
    if os.path.exists(path):
        raise FileExistsError(f"{path} exists; never overwritten")
    if grid is None:
        dk = kd.load(DECAY_PATH)
        grid = kd.Grid(dk.e_bins, dk.e_grid)
    nb = grid.n
    wr = grid.e_widths[None, :] / grid.e_widths[:, None]
    T_par, w_par = kd.parent_energy_nodes(grid, parent_mode, 8)
    pairs = {}
    raw = {}
    for parent, (pdg, table) in _LAMBDA_TABLES.items():
        m0 = mcc.MASS[pdg]
        chans, s = kd._norm_table(table)
        raw[parent] = s
        for br, ds in chans:
            masses = [mcc.MASS[d] for d in ds]
            for i in (0, 1):
                child = _EXTRA_NAMES[ds[i]]
                arr = kd.two_body_moments(grid, m0, masses[i], masses[1 - i], T_par,
                                          w_par, n_gl, l_max)
                pr = pairs.setdefault((parent, child), kd.Pair(nb, l_max))
                kd._add_analytic(pr, arr, br, wr)
    names, ks, ms, moms, hds, ns, ptr = [], [], [], [], [], [], [0]
    for (parent, child), pr in sorted(pairs.items()):
        m, k = np.nonzero(np.any(pr.mom != 0.0, axis=2))
        names.append((parent, child))
        ks.append(k.astype(np.int32))
        ms.append(m.astype(np.int32))
        moms.append(pr.mom[m, k, :])
        hds.append(pr.hd[m, k, :].astype(np.float32))
        ns.append(pr.n[m, k].astype(np.int32))
        ptr.append(ptr[-1] + len(k))
    species = sorted({c for _, c in names} | {p for p, _ in names})
    info = _extra_species_info(species)
    meta = {"module": "cascade3d/kernelset_mc.py:build_decay_extra",
            "machinery": "kernels_decay.two_body_moments (analytic)",
            "created": time.strftime("%Y-%m-%d %H:%M"), "version": VERSION,
            "convention": "as decay.npz: mom[l,m,k] = (w_k/w_m) sum_decays "
                          "P_l(cos theta_lab) / N_decays incl. BR",
            "parent_mode": parent_mode, "n_gl": n_gl, "n_bins": int(nb),
            "raw_br_sums": raw, "br_renormalised": True,
            "channels": "decays.BR_LAM, decays.BR_LAMBAR (p pi-, n pi0; CP image)"}
    d = dict(e_grid=grid.e_grid, e_bins=grid.e_bins, e_widths=grid.e_widths,
             l_max=np.int64(l_max), pair_parent=np.array([a for a, _ in names]),
             pair_child=np.array([b for _, b in names]),
             pair_ptr=np.array(ptr, np.int64), k=np.concatenate(ks),
             m=np.concatenate(ms), mom=np.concatenate(moms),
             hd=np.concatenate(hds), n=np.concatenate(ns),
             species_name=np.array(species),
             species_pdg=np.array([info[s][0] for s in species], np.int64),
             species_helicity=np.array([info[s][1] for s in species], np.int64),
             species_mass=np.array([info[s][2] for s in species]),
             meta_json=np.array(json.dumps(meta, sort_keys=True)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **d)
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------
# the kernel set
# ---------------------------------------------------------------------------
def _delta_m(c_l, L, n_tail, clip):
    """``kernels_gen.delta_m`` plus clipping of ``a*`` to [-1, 1].

    Returns ``(g, a_star, n_clipped, max_excess)``."""
    g, a = kg.delta_m(c_l, L=L, n_tail=n_tail)
    exc = np.abs(a) - 1.0
    n_clip = int(np.count_nonzero(exc > 1e-12))
    max_exc = float(max(exc.max(), 0.0)) if exc.size else 0.0
    if clip:
        a = np.clip(a, -1.0, 1.0)
    return g, a, n_clip, max_exc


def _threshold_rows(child_pdg, e_grid, thr_kin):
    """Bool mask over child bins: True where the bin centre is below threshold."""
    t = thr_kin.get(int(child_pdg))
    if t is None:
        return np.ones(e_grid.size, bool)
    return e_grid < t


def build(gen="pool2000", L=32, n_tail=4, n_e=52, loss=True,
          e_nu_min=MC_DEFAULTS["e_nu_min"], thr_scheme=MC_DEFAULTS["thr_scheme"],
          model=MC_DEFAULTS["model"], nstrata=MC_DEFAULTS["nstrata"],
          phi_mode="average", clip=True, decay_path=DECAY_PATH,
          decay_extra_path=DECAY_EXTRA_PATH, kernel_dir=KERNEL_DIR, angular=True,
          turn_on=True, x_ref=X_REF, g_sec=None):
    """Build the Monte Carlo-matched kernel set.

    Parameters
    ----------
    gen : {"pool2000", "fresh1e5", "fresh1e5_h0", "fresh1e5_h1"}
        Generator kernels: the Monte Carlo's own pools (G1.3(a)), 10^5 fresh
        events per node (G1.3(c)), or one of its two disjoint halves (G1.4).
    L : int
        Legendre order of the reduced angular law (delta-M at order L).
    n_tail : int
        ``g`` is the mean of ``a_{L+1} .. a_{L+n_tail}``.
    n_e : int
        Number of energy bins kept from the bottom of the PHYS grid (52: up to
        3.98 TeV, above the top of the primaries, 3.16 TeV).
    loss : bool
        Muon energy loss on (``shower.dedx_air``) or off.
    e_nu_min, thr_scheme : float, str
        Arguments of ``shower.thresholds`` (the reference run: 0.1, ``chain``).
    model, nstrata, phi_mode
        Primary model, number of strata (sets the top of the primary energies) and
        "average" (bin average) or "centre" for ``phi_top``.
    clip : bool
        Clip ``a*`` to [-1, 1] (``KernelSet.validate`` requires it).
    angular : bool
        Attach the angular laws (False: straight-ahead set, smaller).
    turn_on : bool
        Apply the turn-on rule of M3 to p and n (default); False gives the sets
        of 2026-11-19.
    x_ref : float
        Slant depth [g/cm2] of the interacting fractions of the turn-on rule.
    g_sec : dict, optional
        Slopes g_s of the secondary weight per species (default :data:`G_SEC`).

    Returns
    -------
    KernelSet
    """
    tag, half = GEN_SETS[gen]
    G = kg.load(tag, "PHYS", kernel_dir)
    e_grid_full = G["e_grid"]
    e_grid, e_bins = e_grid_full[:n_e], G["e_bins"][:n_e + 1]
    e_widths = G["e_widths"][:n_e]
    nE, nS = n_e, len(SPECIES)
    dk = kd.load(decay_path)
    dx = kd.load(decay_extra_path)
    for d in (dk, dx):
        np.testing.assert_allclose(d.e_grid, e_grid_full, rtol=1e-12)
    thr = mc_thresholds_kinetic(e_nu_min, thr_scheme)
    pdg = np.array([s[1] for s in SPECIES], np.int64)
    role = np.array([s[3] for s in SPECIES], np.int64)
    mass = np.array([mcc.MASS[int(p)] for p in pdg])
    charge = np.array([mcc.CHARGE[int(p)] for p in pdg], np.int64)
    backend = mc_backend()
    inv_lambda = np.array([mc_inv_lambda(p, e_grid, backend) if r == ROLE_HADRON
                           else np.zeros(nE) for p, r in zip(pdg, role)])
    inv_dec = np.array([mc_inv_decay(p, e_grid) if r != ROLE_SINK
                        else np.zeros(nE) for p, r in zip(pdg, role)])
    dedx = np.zeros((nS, nE))
    if loss:
        dedx[role == ROLE_LOSS] = mc_dedx(e_grid)[None, :]
    phi_top = np.zeros((nS, nE))
    phi_top[[_IDX["p+"], _IDX["n0"]]] = phi_top_mc(e_bins, model, nstrata, phi_mode)

    kinds, par, chi, Ys, gs, As = [], [], [], [], [], []
    stats = {"clipped": 0, "max_excess": 0.0, "dropped_above_grid": {},
             "threshold_zeroed": {}}

    def add(kind, p, c, Y, clm):
        """clm: (nE, nE, l) moments [m, k, l] with clm[..., 0] the number."""
        cpdg = pdg[c]
        rows = _threshold_rows(cpdg, e_grid, thr) if role[c] != ROLE_SINK \
            else np.zeros(nE, bool)
        if rows.any() and np.any(Y[rows] != 0):
            key = f"{['int', 'dec'][kind]} {NAMES[p]}->{NAMES[c]}"
            n = Y[rows] * e_widths[rows, None] / e_widths[None, :]
            stats["threshold_zeroed"][key] = float(n.sum(axis=0).max())
            Y = Y.copy()
            Y[rows] = 0.0
        if not np.any(Y != 0):
            return
        kinds.append(kind)
        par.append(p)
        chi.append(c)
        Ys.append(Y)
        if angular:
            g, a, nc, mx = _delta_m(clm, L, n_tail, clip)
            g = np.where(Y != 0, g, 1.0)
            a = np.where((Y != 0)[..., None], a, 0.0)
            a[..., 0] = 1.0
            stats["clipped"] += nc
            stats["max_excess"] = max(stats["max_excess"], mx)
            gs.append(g)
            As.append(np.moveaxis(a, -1, 0))

    # interactions (generator kernels)
    k_index, m_index = G["k_index"], G["m_index"]
    ksel = k_index < nE
    msel = m_index < nE
    ki, mi = k_index[ksel], m_index[msel]
    for jp, ppdg in enumerate(G["projectiles"]):
        p = int(np.flatnonzero(pdg == int(ppdg))[0])
        served = G["status"][jp] == kg.SERVED
        if half is None:
            cl_p = G["c_l"][jp]
        else:
            cl_p = G["c_l_half"][half, jp].astype(np.float64)
        for ic, cpdg in enumerate(G["children"]):
            c = int(np.flatnonzero((pdg == int(cpdg)) & (np.array(
                [s[2] for s in SPECIES]) == 0))[0])
            cl = cl_p[ic][ksel]                         # (K, M, l)
            above = cl[:, ~msel, 0].sum(axis=1).max() if (~msel).any() else 0.0
            if above > 0:
                key = f"int {NAMES[p]}->{NAMES[c]}"
                stats["dropped_above_grid"][key] = float(above)
            clm = np.zeros((nE, nE, cl.shape[-1]))
            clm[np.ix_(mi, ki)] = np.moveaxis(cl[:, msel], 0, 1)
            clm[:, ~served[:nE]] = 0.0
            Y = clm[..., 0] * e_grid[None, :] / e_grid[:, None]
            add(KIND_INT, p, c, Y, clm)
        del cl_p

    # decays
    for src in (dk, dx):
        for parent, child in src.pairs:
            if parent not in _IDX or child not in _IDX:
                continue
            p, c = _IDX[parent], _IDX[child]
            mom = src.dense(parent, child)                # (l, m, k)
            wr = e_grid_full[nE:, None] / e_grid_full[None, :nE]
            above = (mom[0, nE:, :nE] * wr).sum(axis=0).max()
            if above > 0:
                stats["dropped_above_grid"][f"dec {parent}->{child}"] = float(above)
            mom = mom[:, :nE, :nE]
            Y = mom[0].copy()
            clm = np.moveaxis(mom, 0, -1)
            add(KIND_DEC, p, c, Y, clm)

    # turn-on of the nucleon cross section (rule M3)
    turn_meta = None
    if turn_on:
        g_sec = dict(G_SEC if g_sec is None else g_sec)
        turn_meta = {"rule": "M3 turn-on: sources into (s, k*) times the "
                     "interacting fraction over x_ref (phi_top: f_prim, channel "
                     "rows k* with child s: f_sec), inv_lambda[s, k*] = a_s",
                     "x_ref": float(x_ref),
                     "consequence": "the nucleon flux of bin k* returned by the "
                     "solver is the interacting part only; the number sum rule "
                     "of the channels into (s, k*) is short by 1 - f_sec on "
                     "purpose", "species": {}}
        for name in TURN_ON_SPECIES:
            if g_sec.get(name) is None:
                raise ValueError(f"no secondary slope g for {name}")
            tf = turn_on_factors(name, e_bins, g_sec[name], x_ref, model, backend)
            s, k = _IDX[name], tf["k"]
            np.testing.assert_allclose(inv_lambda[s, k], tf["inv_lambda_centre"],
                                       rtol=1e-12)
            phi_top[s, k] *= tf["f_prim"]
            inv_lambda[s, k] = tf["a"]
            n_sc = 0
            for j in range(len(Ys)):
                if chi[j] == s and np.any(Ys[j][k] != 0):
                    Ys[j] = Ys[j].copy()
                    Ys[j][k] *= tf["f_sec"]
                    n_sc += 1
            tf["n_channels_scaled"] = n_sc
            tf["channels_with_child"] = sorted(
                f"{['int', 'dec'][kinds[j]]} {NAMES[par[j]]}->{name}"
                for j in range(len(Ys)) if chi[j] == s)
            turn_meta["species"][name] = tf

    nCh = len(Ys)
    d = {"e_grid": e_grid, "e_bins": e_bins, "e_widths": e_widths,
         "species": np.array(NAMES), "pdg": pdg, "mass": mass, "charge": charge,
         "role": role, "active": np.ones((nS, nE), bool),
         "inv_lambda": inv_lambda, "inv_dec": inv_dec, "dedx": dedx,
         "phi_top": phi_top, "ch_kind": np.array(kinds, np.int64),
         "ch_parent": np.array(par, np.int64), "ch_child": np.array(chi, np.int64),
         "Y": np.array(Ys)}
    if angular:
        d["ang_channel"] = np.arange(nCh, dtype=np.int64)
        d["n_ang"] = np.array(nE)
        d["g"] = np.array(gs)
        d["a"] = np.array(As)
    lo_r, hi_r = mc_primary_range(nstrata)
    meta = {"builder": "kernelset_mc.build", "version": VERSION, "gen": gen,
            "gen_tag": tag, "gen_half": half, "gen_file_meta_created":
            G["meta"].get("created"), "decay_file": decay_path,
            "decay_extra_file": decay_extra_path, "L": L if angular else None,
            "n_tail": n_tail, "delta_m": "kernels_gen.delta_m, g = mean a_{L+1.."
            "L+n_tail}, a* clipped to [-1,1]" if clip else "not clipped",
            "a_star_clipped_entries": stats["clipped"],
            "a_star_max_excess_before_clip": stats["max_excess"],
            "loss": bool(loss), "dedx": "shower.dedx_air(T + m_mu), muons only",
            "e_nu_min": e_nu_min, "thr_scheme": thr_scheme,
            "thresholds_kinetic": {str(k): v for k, v in thr.items()},
            "threshold_zeroed_max_number_per_parent": stats["threshold_zeroed"],
            "dropped_above_grid_max_number_per_parent":
            stats["dropped_above_grid"],
            "primary_model": model, "nstrata": nstrata,
            "primary_range_T_per_nucleon": [lo_r, hi_r], "phi_mode": phi_mode,
            "interaction_lengths": "interactions.ChromoBackend(xs='mceq')"
            ".lambda_int at T_centre + m", "decay_rates": "m/(p ctau), "
            "constants.CTAU_CM", "n_e": n_e,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mceq_db": G["meta"].get("mceq_db")}
    if turn_on:
        meta["turn_on"] = turn_meta
        meta["interaction_lengths"] += (" (p, n in the turn-on bin: rule M3, "
                                        "meta['turn_on'])")
    d["meta"] = meta
    ks = KernelSet(d)
    ks.validate()
    return ks


def set_path(gen, L=32, loss=True, out_dir=None, turn_on=True):
    """File name of a stored set (``out_dir`` None: :data:`OUT_DIR_CLOSE` with
    the turn-on rule, :data:`OUT_DIR` without)."""
    if out_dir is None:
        out_dir = OUT_DIR_CLOSE if turn_on else OUT_DIR
    return os.path.join(out_dir, f"ks_mc_{gen}_L{L}_{'losson' if loss else 'lossoff'}"
                                 ".npz")


def save_new(ks, path):
    """Save ``ks`` to ``path``; refuses to overwrite an existing file."""
    if os.path.exists(path):
        raise FileExistsError(f"{path} exists; never overwritten")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return ks.save(path)


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------
_BARYON = {2212: 1, 2112: 1, 3122: 1, -2212: -1, -2112: -1, -3122: -1}


def sum_rules(ks, kind=KIND_INT):
    """Energy and baryon-number sums per parent species and parent bin.

    Returns ``{parent name: (E_out / E_in, baryon number out)}`` arrays over the
    parent bins: ``E_out = sum_c sum_m n_mk (T_m + m_c)`` (kept children, total
    energy at the child bin centre), ``E_in = T_k + m_p`` (for interactions the
    target nucleus's rest mass is not counted, so kept target nucleons can push
    the ratio above one),
    with ``n_mk = Y[m, k] w_m / w_k`` the number per interaction or decay.
    Columns with no yield at all are NaN.
    """
    d = ks.d
    E, w = d["e_grid"], d["e_widths"]
    out = {}
    for p in np.unique(d["ch_parent"]):
        sel = np.flatnonzero((d["ch_parent"] == p) & (d["ch_kind"] == kind))
        if sel.size == 0:
            continue
        eo = np.zeros(E.size)
        bo = np.zeros(E.size)
        tot = np.zeros(E.size)
        for j in sel:
            c = d["ch_child"][j]
            n = d["Y"][j] * w[:, None] / w[None, :]
            eo += ((E + d["mass"][c])[:, None] * n).sum(axis=0)
            bo += _BARYON.get(int(d["pdg"][c]), 0) * n.sum(axis=0)
            tot += n.sum(axis=0)
        ein = E + d["mass"][p]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[str(d["species"][p])] = (np.where(tot > 0, eo / ein, np.nan),
                                         np.where(tot > 0, bo, np.nan))
    return out


def g_summary(ks, parent, child, kind=KIND_INT, ks_list=(3, 10, 30, 100, 1000)):
    """Yield-weighted forward fraction ``g`` of a channel per parent energy."""
    d = ks.d
    j = ks.channel(kind, parent, child)
    a = int(np.flatnonzero(d["ang_channel"] == j)[0])
    out = {}
    for e in ks_list:
        k = int(np.argmin(np.abs(np.log(d["e_grid"] / e))))
        y = d["Y"][j][:, k] * d["e_widths"] / d["e_widths"][k]
        out[e] = float(np.sum(y * d["g"][a][:, k]) / max(y.sum(), 1e-300))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("decay-extra", help="write decay_extra.npz (Lambda kernels)")
    b = sub.add_parser("build", help="build and store kernel set(s)")
    b.add_argument("--gen", default="pool2000",
                   choices=list(GEN_SETS) + ["all"])
    b.add_argument("--L", type=int, default=32)
    b.add_argument("--n-tail", type=int, default=4)
    b.add_argument("--n-e", type=int, default=52)
    b.add_argument("--loss", default="on", choices=["on", "off"])
    b.add_argument("--phi-mode", default="average", choices=["average", "centre"])
    b.add_argument("--no-angular", action="store_true")
    b.add_argument("--turn-on", default="on", choices=["on", "off"],
                   help="turn-on rule of M3 for p and n (default on)")
    b.add_argument("--out-dir", default=None,
                   help="default: kernels/mc_close (rule on), kernels/mc (off)")
    a = ap.parse_args(argv)
    if a.cmd == "decay-extra":
        print("wrote", build_decay_extra())
        return
    gens = list(GEN_SETS) if a.gen == "all" else [a.gen]
    loss = a.loss == "on"
    turn_on = a.turn_on == "on"
    for gen in gens:
        t0 = time.time()
        ks = build(gen, L=a.L, n_tail=a.n_tail, n_e=a.n_e, loss=loss,
                   phi_mode=a.phi_mode, angular=not a.no_angular,
                   turn_on=turn_on)
        path = set_path(gen, a.L, loss, a.out_dir, turn_on)
        if a.no_angular:
            path = path.replace(".npz", "_sa.npz")
        save_new(ks, path)
        m = ks.meta
        print(f"wrote {path}: {ks.d['Y'].shape[0]} channels, L={m['L']}, "
              f"clipped {m['a_star_clipped_entries']}, "
              f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
