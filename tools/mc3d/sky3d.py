"""Milestone 3b: the 3D driver with the geomagnetic cutoff, the primary bend,
the muon bend and the azimuthal reuse.

What one job does (``run``)
---------------------------
**Phase 1 -- cascades** (one multiprocessing task per (species, ln E stratum,
chunk), exactly ``sky.build_tasks``' decomposition and seeds).  For every
primary (``sky.primary_sample``: patch position, Lambert direction, H3a
species, superposition nucleon):

* its nucleus rigidity ``R = (A/Z) p_nucleon`` (``cutoff.rigidity_gv``) and
  its ``N_AZ`` azimuthal copies, rotations about the site vertical by
  ``360 j / N_AZ`` (copy 0 is the sampled primary itself), each pre-filtered
  by the exact gate (``cutoff.gate``: forbidden / allowed / to trace);
* one or more **realisations** (hadronic cascades), run with ``B = 0`` for
  the muons and the primary bent in the field (``bend.bend_primary``,
  vectorised over the task):

  - ``R >= r_thr``: one realisation ``B``, the primary bent in copy 0's own
    field; it is **reused** for every copy by the rigid rotation;
  - ``R < r_thr``: one straight (unbent) realisation ``S`` for the 1D
    references, plus one realisation ``C_j`` per copy ``j`` (copy 0 always,
    the others unless gated forbidden), each with the *copy's own* bent
    primary -- the bend depends on the field relative to the rotated
    direction, so these copies cannot be rotations of one another;

* the **pending test** (``soft_hits``): a realisation is kept for replay if
  any neutrino can reach the largest cap in some copy -- exact for neutrinos
  not born in a muon decay (the cap is invariant under the rotations), and
  with a conservative angular margin for every muon (its bend bound
  ``Q_C |B|_max int ds/p`` plus the widest decay-neutrino angle above
  ``e_nu_min`` plus ``margin_extra``), because those are re-generated per
  copy; and kept for the 1D tallies if the primary's straight line crosses the
  cap.  Everything else contributes nothing to any tally and is dropped.

The task writes its pending showers and the copy states still to trace to
``<tag>_tasks/c<i>.pkl``.

**Phase 2 -- pooled trace and replay** (one task per group of consecutive
c-files holding >= ``trace_batch`` copy states, MILESTONE3A_NOTES sec. 8.4).
One ``cutoff.trace_gated`` call for the whole group (Honda's test at each
copy's own injection state), then each c-file is **replayed** into its final
shard ``t<i>.npz``:

* the cascade's non-muon neutrinos are rotated into every copy (``B``) or
  used as they are (``C_j``);
* every pending muon segment is **re-transported in the copy's own field**
  (``bend.transport_muons``, all copies of all showers of the file in one
  vectorised lock-step) and decayed with common random numbers across the
  copies (``bend.michel_decay``) -- the exact treatment of task A, no rotated
  bend approximation;
* four tallies are filled, each in both measures and with per-primary
  squared sums (all copies and realisations of one primary are correlated):

  ``s``  3D, weight ``w / N_AZ x allowed(copy)`` -- the measurement;
  ``n``  3D with no cutoff, copy 0 only, weight ``w`` -- diagnostic;
  ``t``  1D, the realisation's neutrinos laid on the copy's *straight*
         injection line, no cutoff, weight ``w / N_AZ`` -- the reference that
         multiplies daemonflux (which has no cutoff);
  ``v``  1D with the **site line-of-sight cutoff** (``cutoff.SiteCutoffMap``
         at the copy's 1D arrival direction), weight ``w / N_AZ x [R > R_c]``
         -- the reference that isolates the genuine 3D part.

With ``cutoff=False, bend_primary=False, muon_field=False`` the driver is the
milestone-2b driver (``sky._shard``) with the azimuthal reuse: same RNG
stream, the same neutrinos, and the azimuth-summed tallies equal
``sky._shard``'s (``test_m3b.py::test_regression_reproduces_milestone_2b``).
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import time

import numpy as np

import bend
import cutoff
import cutoff as _co
import geometry as geo
import primaries as prim
import sky
from constants import M_N, M_P, R_EARTH_CM
from field import Q_C
from scoring import CONE_BANDS, CONE_EMOM, CONE_ZP, SPECIES, ConeScorer, NestedCaps

N_AZ = 12
#: a bound on |B| over the injection patch at 0-120 km (IGRF-14 2020: <= 61 uT
#: anywhere within 45 deg of Kamioka), for the muon pending margin
B_MAX_T = 6.5e-5
TAGS = ("s", "n", "t", "v")
SUFFIX = ("0", "0sq", "1", "1sq")
SUMMED = tuple(t + s for t in TAGS for s in SUFFIX) + ("hinj", "cone_h",
                                                       "cone_h2", "cone_m")
_SIDX = {s: k for k, s in enumerate(SPECIES)}
_KIND = {"decay": bend.MU_DECAY, "ground": bend.MU_GROUND,
         "escape": bend.MU_ESCAPE, "stop": bend.MU_STOP}
_PKIND = {bend.P_INTERACT: "interact", bend.P_GROUND: "ground",
          bend.P_ESCAPE: "escape"}
_KCODE = {"B": 0, "S": 1, "C": 2}
COUNT_NAMES = ("n_prim", "n_real", "n_pend3", "n_pend1", "n_pend_prim",
               "n_resim_prim", "n_resim_copies", "n_gate_minus", "n_gate_plus",
               "n_trace", "n_traced", "n_allowed_traced", "n_mu_retrans",
               "cpu_cascade", "cpu_bend", "cpu_trace", "cpu_replay",
               "n_prim_escape", "cpu_mu")


class Options:
    """Run switches.  Defaults = everything on (milestone 3b)."""

    def __init__(self, cutoff=True, bend_primary=True, muon_field=True,
                 r_thr=30.0, n_az=N_AZ, skip_gated=True, pend_all=False,
                 margin_extra_deg=2.0, model=None, date=None,
                 trace_batch=10000, max_files_per_group=64, keep_cfiles=True,
                 keep_reals=False, reanchor=False):
        self.cutoff = bool(cutoff)
        self.bend_primary = bool(bend_primary)
        self.muon_field = bool(muon_field)
        self.r_thr = float(r_thr)
        self.n_az = int(n_az)
        self.skip_gated = bool(skip_gated)
        self.pend_all = bool(pend_all)
        self.margin_extra_deg = float(margin_extra_deg)
        self.model = (model or _co.DEFAULT_MODEL).upper()
        self.date = _co._as_datetime(date or _co.DEFAULT_DATE)
        self.trace_batch = int(trace_batch)
        self.max_files_per_group = int(max_files_per_group)
        self.keep_cfiles = bool(keep_cfiles)
        self.keep_reals = bool(keep_reals)
        self.reanchor = bool(reanchor)

    def signature(self):
        return np.array([self.cutoff, self.bend_primary, self.muon_field,
                         self.r_thr, self.n_az, self.skip_gated, self.pend_all,
                         self.margin_extra_deg, self.reanchor], float)

    def needs_field(self):
        return self.bend_primary or self.muon_field


# ---------------------------------------------------------------------------
# geometry helpers (vectorised)
# ---------------------------------------------------------------------------
def locate_vec(caps, r, u):
    """``NestedCaps.locate`` for rows: ``(i0, cz, az, cosi)``, ``i0 = -1``
    where the ray misses every cap."""
    r = np.asarray(r, float)
    u = np.asarray(u, float)
    re = caps.r_e
    b = np.einsum("ij,ij->i", r, u)
    c = np.einsum("ij,ij->i", r, r) - re * re
    disc = b * b - c
    sq = np.sqrt(np.maximum(disc, 0.0))
    s = -b - sq
    s = np.where(s <= 0.0, -b + sq, s)
    ok = (disc >= 0.0) & (s > 0.0)
    h = r + s[:, None] * u
    hn = np.linalg.norm(h, axis=1)
    n = h / np.where(hn > 0, hn, 1.0)[:, None]
    ax = np.asarray(caps.axis)
    ca = n @ ax
    cth = np.asarray(caps.cos_theta)
    i0 = (ca[:, None] < cth[None, :]).sum(1)
    i0 = np.where(ok & (i0 < len(cth)), i0, -1)
    nx, ny, nz = n[:, 0], n[:, 1], n[:, 2]
    cla = np.sqrt(nx * nx + ny * ny)
    pole = cla < 1e-12
    clo = np.where(pole, 1.0, nx / np.where(pole, 1.0, cla))
    slo = np.where(pole, 0.0, ny / np.where(pole, 1.0, cla))
    en = np.stack([-nz * clo, -nz * slo, cla], 1)
    ee = np.stack([-slo, clo, np.zeros_like(clo)], 1)
    en[pole] = (-1.0, 0.0, 0.0)
    ee[pole] = (0.0, 1.0, 0.0)
    cz = -np.einsum("ij,ij->i", u, n)
    dn = -np.einsum("ij,ij->i", u, en)
    de = -np.einsum("ij,ij->i", u, ee)
    az = np.degrees(np.arctan2(de, dn)) % 360.0
    return i0, cz, az, np.abs(cz)


def soft_hits(caps, r, u, margin_rad):
    """Could the ray ``(r, u)`` reach the largest cap after a change of
    direction of at most ``margin_rad``?  Exact (= ``locate`` hits) for a
    zero margin.  A ray that hits the ground is accepted if its crossing lies
    within ``theta_D + margin`` of the site (near grazing incidence a
    direction change of ``d`` moves the crossing by at most ``~d`` of arc); a
    ray that misses is accepted if it misses the Earth's limb by less than
    the margin and its point of closest approach projects within
    ``theta_D + margin`` of the site."""
    r = np.atleast_2d(np.asarray(r, float))
    u = np.atleast_2d(np.asarray(u, float))
    if len(r) == 0:
        return np.zeros(0, bool)
    m = np.broadcast_to(np.asarray(margin_rad, float), (len(r),))
    re = caps.r_e
    ax = np.asarray(caps.axis)
    th = math.radians(caps.thetas[-1])
    rn = np.linalg.norm(r, axis=1)
    b = np.einsum("ij,ij->i", r, u)
    disc = b * b - (rn * rn - re * re)
    sq = np.sqrt(np.maximum(disc, 0.0))
    s = -b - sq
    s2 = -b + sq
    s = np.where(s <= 0.0, s2, s)
    hit = (disc >= 0.0) & (s > 0.0)
    X = r + np.where(hit, s, 0.0)[:, None] * u
    arc_x = np.arccos(np.clip((X @ ax) / np.linalg.norm(X, axis=1), -1, 1))
    st = np.maximum(-b, 0.0)
    T = r + st[:, None] * u
    arc_t = np.arccos(np.clip((T @ ax) / np.linalg.norm(T, axis=1), -1, 1))
    gam = np.arccos(np.clip(-b / rn, -1.0, 1.0))
    gam_t = np.arcsin(np.clip(re / rn, -1.0, 1.0))
    eps = gam - gam_t
    ok_hit = hit & (arc_x <= th + m)
    ok_miss = (~hit) & (eps < m) & (arc_t <= th + m)
    return ok_hit | ok_miss


# ---------------------------------------------------------------------------
# phase 1: the recorder
# ---------------------------------------------------------------------------
class Recorder:
    """Proxy scorer for ``run_shower``: records every neutrino (with its
    parent and generation) and every muon segment (start state,
    polarisation, outcome, ``int ds/p``), and optionally fills the cone
    tally against the primary's injection direction ``u0``."""

    wants_parent = True
    wants_muons = True

    def __init__(self, cone=None, r0=None, u0=None):
        self.nu = []
        self.mu = []
        self.cone = cone
        self._u0 = None if u0 is None else tuple(float(x) for x in u0)
        self._zp = 0
        if r0 is not None and u0 is not None:
            rn = math.sqrt(float(np.dot(r0, r0)))
            czp = -float(np.dot(u0, r0)) / rn
            self._zp = min(max(int(czp * (len(CONE_ZP) - 1)), 0),
                           len(CONE_ZP) - 2)

    def add(self, pdg, e, r, u, w, parent=0, gen=1):
        k = _SIDX.get(pdg)
        if k is None:
            return
        self.nu.append((pdg, e, r[0], r[1], r[2], u[0], u[1], u[2], parent,
                        gen))
        if self.cone is not None and self._u0 is not None and e >= CONE_EMOM[0]:
            u0 = self._u0
            self.cone.add(k, e, u0[0] * u[0] + u0[1] * u[1] + u0[2] * u[2],
                          w, parent, gen, self._zp)

    def add_muon(self, pdg, e, r, u, pol, parent, gen, kind, r_end, u_end,
                 e_end, sinv, t_target=0.0):
        self.mu.append((pdg, e, r[0], r[1], r[2], u[0], u[1], u[2], pol, parent,
                        gen, _KIND[kind], r_end[0], r_end[1], r_end[2],
                        u_end[0], u_end[1], u_end[2], e_end, sinv, t_target))

    def arrays(self):
        nu = np.array(self.nu, float).reshape(-1, 10)
        mu = np.array(self.mu, float).reshape(-1, 21)
        return nu, mu


# column indices
NU_PDG, NU_E, NU_R, NU_U, NU_PAR, NU_GEN = 0, 1, slice(2, 5), slice(5, 8), 8, 9
MU_PDG, MU_E, MU_R, MU_U, MU_POL, MU_PAR, MU_GEN, MU_KIND = 0, 1, slice(2, 5), \
    slice(5, 8), 8, 9, 10, 11
MU_REND, MU_UEND, MU_EEND, MU_SINV, MU_TT = slice(12, 15), slice(15, 18), 18, \
    19, 20
#: the columns kept for the replay: pdg, e, r, u, pol, parent, gen, t_target
MU_KEEP = list(range(11)) + [MU_TT]
RM_TT = 11


def muon_margin(mu, e_nu_min, extra_deg):
    """Half-angle [rad] of the cone around a muon's end direction that holds
    every neutrino above ``e_nu_min`` its decay can make in any copy: the
    widest decay angle (``bend.max_nu_angle``) plus the most the segment can
    bend in any field orientation (``Q_C |B|_max int ds/p``) plus
    ``extra_deg``."""
    bend_max = Q_C * B_MAX_T * mu[:, MU_SINV] * 1e-2
    return (bend_max + bend.max_nu_angle(mu[:, MU_EEND], e_nu_min)
            + math.radians(extra_deg))


def cone_may_hit_cap(caps, P, u, alpha, n_edge=72):
    """Conservative: can some direction within ``alpha`` of ``u`` (rows), from
    ``P``, reach the largest cap?  Two necessary conditions: the cone reaches
    below the horizon of ``P``, and it overlaps the disc of directions
    ``{d : angle(d, d_C) <= beta}`` around the direction ``d_C`` of the site,
    ``beta`` the largest angle between ``d_C`` and the direction of a point of
    the cap's rim (``pi`` when ``P`` is above the cap)."""
    P = np.atleast_2d(np.asarray(P, float))
    u = np.atleast_2d(np.asarray(u, float))
    if len(P) == 0:
        return np.zeros(0, bool)
    re = caps.r_e
    ax = np.asarray(caps.axis, float)
    th = math.radians(caps.thetas[-1])
    # rim of the cap on the ground sphere
    t1, t2 = geo._tangent_basis(ax)
    ph = np.linspace(0.0, 2.0 * np.pi, n_edge, endpoint=False)
    rim = re * (math.cos(th) * ax[None, :]
                + math.sin(th) * (np.cos(ph)[:, None] * t1[None, :]
                                  + np.sin(ph)[:, None] * t2[None, :]))
    C = re * ax
    dC = C[None, :] - P
    dC /= np.linalg.norm(dC, axis=1)[:, None]
    dk = rim[None, :, :] - P[:, None, :]
    dk /= np.linalg.norm(dk, axis=2)[:, :, None]
    beta = np.arccos(np.clip(np.einsum("ikj,ij->ik", dk, dC), -1, 1)).max(1)
    pn = np.linalg.norm(P, axis=1)
    above = (P @ ax) / pn >= math.cos(th)
    beta = np.where(above, np.pi, beta)
    ang = np.arccos(np.clip(np.einsum("ij,ij->i", u, dC), -1, 1))
    gam = np.arccos(np.clip(-np.einsum("ij,ij->i", P, u) / pn, -1, 1))
    gam_t = np.arcsin(np.clip(re / pn, -1, 1))
    return (ang <= alpha + beta) & (gam - alpha < gam_t)


def pending_3d(caps, nu, mu, muon_field, e_nu_min, extra_deg, pend_all=False):
    """``(pending, muon_mask)``: does the realisation need a 3D replay, and
    which of its muons must be re-transported per copy."""
    if pend_all:
        return True, np.ones(len(mu), bool)
    if not muon_field:
        return bool(len(nu) and soft_hits(caps, nu[:, NU_R], nu[:, NU_U],
                                          0.0).any()), np.zeros(len(mu), bool)
    rigid = np.abs(nu[:, NU_PAR]) != 13 if len(nu) else np.zeros(0, bool)
    p = bool(rigid.any() and soft_hits(caps, nu[rigid][:, NU_R],
                                       nu[rigid][:, NU_U], 0.0).any())
    mm = np.zeros(len(mu), bool)
    if len(mu):
        live = np.isin(mu[:, MU_KIND], (bend.MU_DECAY, bend.MU_GROUND))
        if live.any():
            ml = mu[live]
            P = ml[:, MU_REND].copy()
            ue = ml[:, MU_UEND]
            gr = ml[:, MU_KIND] == bend.MU_GROUND
            P[gr] -= 1.0e5 * ue[gr]          # 1 km back up the track
            mm[live] = cone_may_hit_cap(caps, P, ue,
                                        muon_margin(ml, e_nu_min, extra_deg))
    return p or bool(mm.any()), mm


# ---------------------------------------------------------------------------
# phase 1: the cascade task
# ---------------------------------------------------------------------------
_CTX = {}


def _init(cache, n_pool, e_nu_min, lat, lon, caps, model, opts, fieldgrid,
          sitemap):
    import importlib.util  # noqa: F401
    from interactions import ChromoBackend
    from shower import Config
    _CTX["backend"] = ChromoBackend(cache_dir=cache, n_pool=n_pool,
                                    xs="mceq", readonly=True)
    _CTX["cfg"] = Config(collinear=False, bfield=None, energy_loss=True,
                         e_nu_min=e_nu_min, decay_mode="kinematic",
                         polarisation=True)
    _CTX["caps"] = NestedCaps(lat, lon, caps)
    _CTX["model"] = prim.primary_model(model)
    _CTX["opts"] = opts
    _CTX["field"] = fieldgrid
    _CTX["sitemap"] = sitemap


def task_dir(outdir, tag):
    return sky.task_dir(outdir, tag)


def cfile(outdir, tag, i):
    return os.path.join(task_dir(outdir, tag), f"c{i:05d}.pkl")


def tfile(outdir, tag, i):
    return sky.task_path(outdir, tag, i)


def _atomic_pickle(path, obj):
    tmp = path + f".{os.getpid()}.part"
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def cascade_task(task):
    """Phase 1 for one (group, stratum, chunk): see the module docstring."""
    gname, k, n, theta_inj, seed, n_tot, ti, path = task
    from shower import run_shower
    caps, cfg, backend = _CTX["caps"], _CTX["cfg"], _CTX["backend"]
    opts, fld = _CTX["opts"], _CTX["field"]
    c_start = time.process_time()
    t_start = time.time()
    rng = np.random.default_rng(seed)
    axis = np.array(caps.axis, float)
    e_nuc, is_p, w, r0, u0 = sky.primary_sample(rng, _CTX["model"], gname, k, n,
                                                n_tot, theta_inj, axis)
    _nm, a_mass, z, _gid = sky.GROUPS[gname]
    R = cutoff.rigidity_gv(e_nuc, a_mass, z)
    pdg0 = np.where(is_p, 2212, 2112)
    e_tot = e_nuc + np.where(is_p, M_P, M_N)
    naz = opts.n_az
    cnt = dict.fromkeys(COUNT_NAMES, 0.0)
    cnt["n_prim"] = float(n)
    # copies and the exact gate
    rr, uu = cutoff.copy_states(r0, u0, axis, naz, 0.0)        # (n, naz, 3)
    if opts.cutoff:
        gate = cutoff.gate(rr.reshape(-1, 3), uu.reshape(-1, 3),
                           np.repeat(R, naz)).reshape(n, naz)
    else:
        gate = np.ones((n, naz), int)
    # the realisation plan
    resim = (R < opts.r_thr) if opts.bend_primary else np.zeros(n, bool)
    plan = []                    # (i, kind, copy j, r_start, u_start, bent?)
    for i in range(n):
        if not resim[i]:
            plan.append((i, "B", 0, opts.bend_primary))
        else:
            plan.append((i, "S", 0, False))
            for j in range(naz):
                if j == 0 or not (opts.skip_gated and gate[i, j] == -1):
                    plan.append((i, "C", j, True))
    cnt["n_resim_prim"] = float(resim.sum())
    cnt["n_resim_copies"] = float(sum(1 for p in plan if p[1] == "C"))
    # the bent primaries of the whole task in one vectorised call
    first = {}
    if opts.bend_primary:
        cb = time.process_time()
        rng_b = np.random.default_rng([seed, 7919])
        bi = [p for p in plan if p[3]]
        if bi:
            ii = np.array([p[0] for p in bi])
            jj = np.array([p[2] for p in bi])
            lam = np.array([backend.lambda_int(int(pdg0[i]), float(e_tot[i]))
                            for i in ii])
            dX = -np.log(rng_b.random(len(bi))) * lam
            st, ri, ui, _L, _b = bend.bend_primary(rr[ii, jj], uu[ii, jj], R[ii],
                                                   dX, fld)
            for q, p in enumerate(bi):
                first[(p[0], p[2])] = (_PKIND[int(st[q])], ri[q], ui[q],
                                       float(dX[q]))
            cnt["n_prim_escape"] = float((st != bend.P_INTERACT).sum())
        cnt["cpu_bend"] = time.process_time() - cb
    # the cascades
    cone = ConeScorer()
    pend = {}
    for (i, kind, j, bent) in plan:
        is_base = kind in ("B", "S")
        rec = Recorder(cone if is_base else None, r0[i], u0[i])
        rs, us = (r0[i], u0[i]) if j == 0 else (rr[i, j], uu[i, j])
        fst = first.get((i, j)) if bent else None
        run_shower(rng, int(pdg0[i]), float(e_tot[i]), rs, us, backend, cfg,
                   rec, float(w[i]), first=None if fst is None else fst[:3])
        cnt["n_real"] += 1
        nu, mu = rec.arrays()
        p3, mmask = (False, None)
        if kind in ("B", "C"):
            p3, mmask = pending_3d(caps, nu, mu, opts.muon_field, cfg.e_nu_min,
                                   opts.margin_extra_deg, opts.pend_all)
        p1 = False
        if kind in ("B", "S"):
            p1 = opts.pend_all or caps.locate(r0[i], u0[i]) is not None
        if not (p3 or p1):
            continue
        cnt["n_pend3"] += p3
        cnt["n_pend1"] += p1
        real = {"kind": kind, "j": j, "p3": p3, "p1": p1, "nu": nu}
        if kind == "B" and bent and (i, j) in first:
            f = first[(i, j)]
            real["int"] = (f[1], f[2], f[3])
        if p3 and opts.muon_field:
            real["mu"] = mu[mmask][:, MU_KEEP]
            real["mu_idx"] = np.where(mmask)[0]
            real["n_mu"] = len(mu)
        pend.setdefault(i, []).append(real)
    prims = []
    for i in sorted(pend):
        prims.append({"i": int(i), "w": float(w[i]), "R": float(R[i]),
                      "r0": r0[i].copy(), "u0": u0[i].copy(),
                      "gate": gate[i].copy(), "reals": pend[i]})
    cnt["n_pend_prim"] = float(len(prims))
    cnt["cpu_cascade"] = time.process_time() - c_start - cnt["cpu_bend"]
    # the per-copy entries: rotations, muon re-transport in each copy's own
    # field, decays, cap crossings -- everything but the cutoff verdict
    c_rep = time.process_time()
    grid = Grid(caps, sky.E_BINS, sky.CZ_BINS, sky.AZ_BINS)
    ent, n_mu = build_entries(prims, seed, grid, opts, fld, _CTX["sitemap"],
                              cfg.e_nu_min, cfg.polarisation)
    cnt["n_mu_retrans"] = float(n_mu)
    cnt["cpu_mu"] = float(getattr(build_entries, "cpu_mu", 0.0))
    # the copies whose verdict matters: a 3D cap hit in that copy
    e3 = ent["3d"]
    pairs = np.unique(np.stack([e3["prim"], e3["copy"]], 1)[e3["w0"] > 0],
                      axis=0) if len(e3["prim"]) else np.zeros((0, 2), int)
    gate_p = np.array([p["gate"] for p in prims], int).reshape(-1, naz)
    if opts.cutoff and len(pairs):
        gv = gate_p[pairs[:, 0], pairs[:, 1]]
        cnt["n_gate_minus"] = float((gv == -1).sum())
        cnt["n_gate_plus"] = float((gv == 1).sum())
        tr_ij = pairs[gv == 0]
    else:
        tr_ij = np.zeros((0, 2), int)
    tr_r = np.array([rr[prims[q]["i"], j] for q, j in tr_ij]).reshape(-1, 3)
    tr_u = np.array([uu[prims[q]["i"], j] for q, j in tr_ij]).reshape(-1, 3)
    tr_R = np.array([prims[q]["R"] for q, _j in tr_ij], float)
    cnt["n_trace"] = float(len(tr_ij))
    cnt["cpu_replay"] = time.process_time() - c_rep
    a_patch = geo.patch_area_cm2(sky.R_INJ_CM, theta_inj)
    meta = np.array([n, k, sky.GROUP_FRAC.get(gname, 0.0),
                     time.time() - t_start, a_patch, theta_inj, n_tot,
                     time.process_time() - c_start, seed])
    out = {"meta": meta, "group": gname, "k": k, "seed": seed, "ti": ti,
           "gate": gate_p, "R": np.array([p["R"] for p in prims]),
           "ent": ent, "trace_ij": tr_ij, "trace_r": tr_r, "trace_u": tr_u,
           "trace_R": tr_R, "counts": cnt, "cone": cone.to_dict()}
    if opts.keep_reals:
        out["prims"] = prims
    _atomic_pickle(path, out)
    np.save(path[:-4] + ".n.npy", np.array([len(tr_ij), len(prims)]))
    return {"ti": ti, "n": n, "n_trace": len(tr_ij), "n_pend": len(prims),
            "cpu": float(meta[7]), "wall": float(meta[3])}


# ---------------------------------------------------------------------------
# the per-copy entries (phase 1) and the tallies (phase 2)
# ---------------------------------------------------------------------------
class Grid:
    """Honda's binning plus the nested caps: flat bin indices."""

    def __init__(self, caps, e_bins, cz_bins, az_bins):
        self.caps = caps
        self.e_bins = np.asarray(e_bins, float)
        self.cz_bins = np.asarray(cz_bins, float)
        self.az_bins = np.asarray(az_bins, float)
        self.ne, self.ncz, self.naz = (len(self.e_bins) - 1,
                                       len(self.cz_bins) - 1,
                                       len(self.az_bins) - 1)
        self.ncap = len(caps.thetas)
        self.shape = (self.ncap, len(SPECIES), self.ne, self.ncz, self.naz)
        self.size = int(np.prod(self.shape))
        self._lge0 = math.log(self.e_bins[0])
        self._dlge = math.log(self.e_bins[1] / self.e_bins[0])

    def bins(self, pdg, e, i0, cz, az):
        """Flat bin index in the smallest containing cap (-1 = not scored)
        and the species index."""
        pdg = np.asarray(pdg).astype(int)
        k = np.full(len(pdg), -1)
        for sp, idx in _SIDX.items():
            k[pdg == sp] = idx
        with np.errstate(divide="ignore", invalid="ignore"):
            ie = np.floor((np.log(e) - self._lge0) / self._dlge)
        ie = np.where(np.isfinite(ie), ie, -1).astype(int)
        iz = np.minimum(((np.asarray(cz) + 1.0) / 2.0 * self.ncz).astype(int),
                        self.ncz - 1)
        ia = np.clip((np.asarray(az) / 360.0 * self.naz).astype(int), 0,
                     self.naz - 1)
        ok = (k >= 0) & (ie >= 0) & (ie < self.ne) & (iz >= 0) & (i0 >= 0)
        flat = (((i0 * len(SPECIES) + k) * self.ne + ie) * self.ncz + iz) \
            * self.naz + ia
        return np.where(ok, flat, -1), k


class Tally:
    """The four tallies (``TAGS``) x two measures x squares (squares summed
    per primary: all copies and realisations of a primary are correlated),
    plus the injection-distance histogram of the cap hits."""

    HINJ_CZ = (0.0, 0.1, 0.2, 0.5, 1.0)
    HINJ_NTH = 60

    def __init__(self, grid):
        self.g = grid
        self.a = {t + s: np.zeros(grid.size) for t in TAGS for s in SUFFIX}
        self.hinj = np.zeros((2, len(SPECIES), len(CONE_BANDS) - 1,
                              len(self.HINJ_CZ) - 1, self.HINJ_NTH))

    def accumulate(self, tag, flat, w0, w1, prim):
        """Entries (smallest-cap flat index, weights in both measures, owning
        primary) go into every cap >= the smallest one, with ``1/A_cap``."""
        g = self.g
        ok = (flat >= 0) & (w0 != 0.0)
        if not ok.any():
            return
        flat, w0, w1, prim = flat[ok], w0[ok], w1[ok], prim[ok]
        per_cap = g.size // g.ncap
        i0 = flat // per_cap
        rest = flat % per_cap
        F, W0, W1, P = [], [], [], []
        for ic in range(g.ncap):
            m = i0 <= ic
            F.append(ic * per_cap + rest[m])
            W0.append(w0[m] / g.caps.areas[ic])
            W1.append(w1[m] / g.caps.areas[ic])
            P.append(prim[m])
        F, W0, W1, P = (np.concatenate(x) for x in (F, W0, W1, P))
        key = P.astype(np.int64) * g.size + F
        uk, inv = np.unique(key, return_inverse=True)
        for suf, W in (("0", W0), ("1", W1)):
            self.a[tag + suf] += np.bincount(F, weights=W, minlength=g.size)
            sums = np.bincount(inv, weights=W)
            self.a[tag + suf + "sq"] += np.bincount(uk % g.size,
                                                     weights=sums * sums,
                                                     minlength=g.size)

    def add_hinj(self, view, k, e, cz, ith, w):
        b = np.asarray(CONE_BANDS)
        ib = np.searchsorted(b, e, side="right") - 1
        hz = np.asarray(self.HINJ_CZ)
        czc = np.minimum(cz, 1.0)
        iz = np.clip(np.searchsorted(hz, czc, side="right") - 1, 0, len(hz) - 2)
        ok = (k >= 0) & (ib >= 0) & (ib < len(b) - 1) & (czc >= hz[0]) & (w != 0)
        np.add.at(self.hinj, (np.full(ok.sum(), view), k[ok], ib[ok], iz[ok],
                              ith[ok]), w[ok])

    def to_dict(self):
        g = self.g
        d = {key: v.reshape(g.shape) for key, v in self.a.items()}
        d.update(e_bins=g.e_bins, cz_bins=g.cz_bins, az_bins=g.az_bins,
                 cap_theta=np.array(g.caps.thetas),
                 cap_area=np.array(g.caps.areas),
                 site=np.array([g.caps.lat, g.caps.lon]),
                 hinj=self.hinj, hinj_cz=np.array(self.HINJ_CZ))
        return d


def _rot(v, axis, deg):
    """Rotate rows ``v`` clockwise-from-above by ``deg`` about ``axis``
    (``cutoff.copy_states``' convention: right-handed rotation by -deg)."""
    if deg == 0.0:
        return v
    return cutoff.rotate_about(v, axis, -math.radians(deg))


def _cat(lst, dtype=float, shape=(0,)):
    return np.concatenate(lst) if lst else np.zeros(shape, dtype)


def _reanchor_table(prims, axis, naz, shifts, fld):
    """For every 3D-pending ``B`` realisation and copy ``c >= 1``: the exact
    first-interaction state of the copy's *own* bent primary (same sampled
    depth, the copy's own field) against the rotation of the base one.
    Returns ``{(prim, real, copy): (r_rot, r_ex, Q)}``: an entry of the
    rotated copy is moved by ``x -> r_ex + Q (x - r_rot)``, ``d -> Q d`` --
    the cascade re-anchored on the copy's exact primary axis (the bias
    measurement of the rotation reuse; ``(None,)`` if the exact copy's
    primary leaves the atmosphere)."""
    keys, rs, us, Rs, dXs, base = [], [], [], [], [], []
    for pi, p in enumerate(prims):
        for ri, re_ in enumerate(p["reals"]):
            if not (re_["p3"] and re_["kind"] == "B" and "int" in re_):
                continue
            r_i, u_i, dX = re_["int"]
            for c in range(1, naz):
                keys.append((pi, ri, c))
                rs.append(_rot(p["r0"][None, :], axis, shifts[c])[0])
                us.append(_rot(p["u0"][None, :], axis, shifts[c])[0])
                Rs.append(p["R"])
                dXs.append(dX)
                base.append((_rot(np.asarray(r_i)[None, :], axis, shifts[c])[0],
                             _rot(np.asarray(u_i)[None, :], axis, shifts[c])[0]))
    if not keys:
        return {}
    st, rex, uex, _L, _b = bend.bend_primary(np.array(rs), np.array(us),
                                             np.array(Rs), np.array(dXs), fld)
    out = {}
    for q, key in enumerate(keys):
        if st[q] != bend.P_INTERACT:
            out[key] = (None,)
            continue
        a, b = base[q][1], uex[q]
        ax = np.cross(a, b)
        sn = np.linalg.norm(ax)
        ang = math.atan2(sn, float(np.dot(a, b)))
        Q = np.eye(3) if sn < 1e-15 else _rotmat(ax / sn, ang)
        out[key] = (base[q][0], rex[q], Q)
    return out


def _rotmat(k, ang):
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(ang) * K + (1.0 - math.cos(ang)) * (K @ K)


def _anchor_pos(x, tr):
    if tr is None:
        return x
    return tr[1][None, :] + (x - tr[0][None, :]) @ tr[2].T


def _anchor_dir(d, tr):
    if tr is None:
        return d
    return d @ tr[2].T


def build_entries(prims, seed, grid, opts, fld, sitemap, e_nu_min,
                  polarisation=True):
    """Every cap entry of every copy of the pending realisations, before the
    cutoff verdict.  Returns ``(ent, n_muons_retransported)`` with
    ``ent["3d"]`` (keys ``flat, k, e, cz, ith, w0, wn, inv, prim, copy``:
    ``w0 = w / N_AZ`` to be multiplied by the copy's verdict, ``wn`` the
    no-cutoff copy-0 weight, ``inv = 1/cos``) and ``ent["1d"]`` (``flat, k,
    e, cz, ith, wt, wv, inv, prim``)."""
    caps = grid.caps
    axis = np.asarray(caps.axis, float)
    naz = opts.n_az
    shifts = 360.0 * np.arange(naz) / naz
    E = {key: [] for key in ("pdg", "e", "r", "u", "w0", "wn", "prim", "copy",
                             "ith")}
    M = {key: [] for key in ("pdg", "e", "r", "u", "pol", "rnd", "w0", "wn",
                             "prim", "copy", "ith")}
    D = {key: [] for key in ("flat", "k", "e", "cz", "ith", "wt", "wv", "inv",
                             "prim")}
    anchor = _reanchor_table(prims, axis, naz, shifts, fld) \
        if (opts.reanchor and opts.bend_primary) else {}
    for pi, p in enumerate(prims):
        w, R = p["w"], p["R"]
        ca = float(np.dot(p["r0"], axis) / np.linalg.norm(p["r0"]))
        ith = min(int(math.degrees(math.acos(max(-1.0, min(1.0, ca))))),
                  Tally.HINJ_NTH - 1)
        for ri, re_ in enumerate(p["reals"]):
            nu = re_["nu"]
            if re_["p3"]:
                copies = np.arange(naz) if re_["kind"] == "B" \
                    else np.array([re_["j"]])
                if opts.muon_field and len(nu):
                    rig = nu[np.abs(nu[:, NU_PAR]) != 13]
                else:
                    rig = nu
                mu = re_.get("mu")
                rnd = None
                if mu is not None and len(mu):
                    # common random numbers over the copies: the base
                    # cascade's proper decay time, and one decay sample per
                    # muon, keyed by (task seed, primary, realisation, muon)
                    # so that they do not depend on what else is pending
                    rnd = np.random.default_rng(
                        [seed, p["i"], _KCODE[re_["kind"]], int(re_["j"])]
                    ).random((re_["n_mu"], 7))[re_["mu_idx"]]
                    rnd[:, 0] = 1.0 - np.exp(-mu[:, RM_TT])
                for c in copies:
                    w0 = w / naz
                    wn = w if c == 0 else 0.0
                    sh = shifts[c] if re_["kind"] == "B" else 0.0
                    tr = anchor.get((pi, ri, int(c)))
                    if tr is not None and tr[0] is None:
                        continue              # the exact copy never interacts
                    if len(rig):
                        m = len(rig)
                        E["pdg"].append(rig[:, NU_PDG])
                        E["e"].append(rig[:, NU_E])
                        E["r"].append(_anchor_pos(_rot(rig[:, NU_R], axis, sh),
                                                  tr))
                        E["u"].append(_anchor_dir(_rot(rig[:, NU_U], axis, sh),
                                                  tr))
                        E["w0"].append(np.full(m, w0))
                        E["wn"].append(np.full(m, wn))
                        E["prim"].append(np.full(m, pi))
                        E["copy"].append(np.full(m, c))
                        E["ith"].append(np.full(m, ith))
                    if rnd is not None:
                        m = len(mu)
                        M["pdg"].append(mu[:, MU_PDG])
                        M["e"].append(mu[:, MU_E])
                        M["r"].append(_anchor_pos(_rot(mu[:, MU_R], axis, sh),
                                                  tr))
                        M["u"].append(_anchor_dir(_rot(mu[:, MU_U], axis, sh),
                                                  tr))
                        M["pol"].append(mu[:, MU_POL] if polarisation
                                        else np.zeros(m))
                        M["rnd"].append(rnd)
                        M["w0"].append(np.full(m, w0))
                        M["wn"].append(np.full(m, wn))
                        M["prim"].append(np.full(m, pi))
                        M["copy"].append(np.full(m, c))
                        M["ith"].append(np.full(m, ith))
            if re_["p1"] and len(nu):
                rs = cutoff.rotate_about(np.tile(p["r0"], (naz, 1)), axis,
                                         -np.radians(shifts))
                us = cutoff.rotate_about(np.tile(p["u0"], (naz, 1)), axis,
                                         -np.radians(shifts))
                i0, cz, az, cosi = locate_vec(caps, rs, us)
                if opts.cutoff and sitemap is not None:
                    zen = np.degrees(np.arccos(np.clip(cz, -1.0, 1.0)))
                    site_ok = sitemap.allowed(R, zen, az)
                else:
                    site_ok = np.ones(naz, bool)
                m = len(nu)
                for c in range(naz):
                    if i0[c] < 0:
                        continue
                    flat, kk = grid.bins(nu[:, NU_PDG], nu[:, NU_E],
                                         np.full(m, i0[c]), np.full(m, cz[c]),
                                         np.full(m, az[c]))
                    D["flat"].append(flat)
                    D["k"].append(kk)
                    D["e"].append(nu[:, NU_E])
                    D["cz"].append(np.full(m, cz[c]))
                    D["ith"].append(np.full(m, ith))
                    D["wt"].append(np.full(m, w / naz))
                    D["wv"].append(np.full(m, w / naz * float(site_ok[c])))
                    D["inv"].append(np.full(m, 1.0 / max(cosi[c], 1e-6)))
                    D["prim"].append(np.full(m, pi))
    # --- re-transport and decay the muons, per copy, in the copy's own field
    n_mu = 0
    c_mu = time.process_time()
    if M["pdg"]:
        mp = _cat(M["pdg"]).astype(int)
        n_mu = len(mp)
        rnd = _cat(M["rnd"], shape=(0, 7))
        tt = -np.log(1.0 - rnd[:, 0])
        kind, rm, um, em = bend.transport_muons(
            mp, _cat(M["e"]), _cat(M["r"], shape=(0, 3)),
            _cat(M["u"], shape=(0, 3)), tt, fld if opts.muon_field else None)
        dec = kind == bend.MU_DECAY
        if dec.any():
            pn, en, dn = bend.michel_decay(mp[dec], em[dec], um[dec],
                                           _cat(M["pol"])[dec], rnd[dec, 1:7],
                                           axis)
            for key in ("w0", "wn", "prim", "copy", "ith"):
                E[key].append(np.repeat(_cat(M[key])[dec], 2))
            E["pdg"].append(pn.reshape(-1).astype(float))
            E["e"].append(en.reshape(-1))
            E["r"].append(np.repeat(rm[dec], 2, axis=0))
            E["u"].append(dn.reshape(-1, 3))
    build_entries.cpu_mu = time.process_time() - c_mu
    # --- the 3D cap crossings; only the hits are kept
    e3 = {key: np.zeros(0) for key in ("flat", "k", "e", "cz", "ith", "w0",
                                        "wn", "inv", "prim", "copy")}
    if E["pdg"]:
        pdg = _cat(E["pdg"]).astype(int)
        e = _cat(E["e"])
        r = _cat(E["r"], shape=(0, 3))
        u = _cat(E["u"], shape=(0, 3))
        keep = e >= e_nu_min
        i0, cz, az, cosi = locate_vec(caps, r[keep], u[keep])
        flat, kk = grid.bins(pdg[keep], e[keep], i0, cz, az)
        hit = flat >= 0
        e3 = {"flat": flat[hit].astype(np.int64), "k": kk[hit].astype(np.int8),
              "e": e[keep][hit], "cz": cz[hit], "ith": None,
              "inv": 1.0 / np.maximum(cosi[hit], 1e-6)}
        for key in ("w0", "wn", "prim", "copy", "ith"):
            e3[key] = _cat(E[key])[keep][hit]
        e3["prim"] = e3["prim"].astype(np.int64)
        e3["copy"] = e3["copy"].astype(np.int64)
        e3["ith"] = e3["ith"].astype(np.int64)
    e1 = {key: _cat(D[key]) for key in D}
    if len(e1["flat"]):
        hit = e1["flat"] >= 0
        e1 = {key: v[hit] for key, v in e1.items()}
        for key in ("flat", "prim", "ith", "k"):
            e1[key] = e1[key].astype(np.int64)
    return {"3d": e3, "1d": e1}, n_mu


def tally_entries(grid, ent, A):
    """Phase 2: the tallies from the entries and the (n_prim, n_az) verdict
    matrix ``A``."""
    tal = Tally(grid)
    e3, e1 = ent["3d"], ent["1d"]
    if len(e3["flat"]):
        ws = e3["w0"] * A[e3["prim"], e3["copy"]]
        tal.accumulate("s", e3["flat"], ws, ws * e3["inv"], e3["prim"])
        tal.accumulate("n", e3["flat"], e3["wn"], e3["wn"] * e3["inv"],
                       e3["prim"])
        tal.add_hinj(0, e3["k"].astype(int), e3["e"], e3["cz"], e3["ith"], ws)
    if len(e1["flat"]):
        tal.accumulate("t", e1["flat"], e1["wt"], e1["wt"] * e1["inv"],
                       e1["prim"])
        tal.accumulate("v", e1["flat"], e1["wv"], e1["wv"] * e1["inv"],
                       e1["prim"])
        tal.add_hinj(1, e1["k"].astype(int), e1["e"], e1["cz"], e1["ith"],
                     e1["wt"])
    return tal


def verdicts(d, verdict_traced, cut_on, naz):
    """The (n_prim, n_az) verdict matrix of a c-file: the gate where it
    decides, the trace where it does not (copies with no 3D hit keep the
    gate's value; they do not enter the 3D tally)."""
    g = np.asarray(d["gate"]).reshape(-1, naz)
    if not cut_on:
        return np.ones(g.shape, bool)
    A = g == 1
    tij = np.asarray(d["trace_ij"], int).reshape(-1, 2)
    if len(tij):
        A[tij[:, 0], tij[:, 1]] = verdict_traced
    return A


def trace_replay_task(job):
    """Phase 2 for a group of c-files: one pooled trace, then the tallies."""
    tis, outdir, tag = job
    caps, opts = _CTX["caps"], _CTX["opts"]
    datas = []
    for ti in tis:
        with open(cfile(outdir, tag, ti), "rb") as fh:
            datas.append(pickle.load(fh))
    c0 = time.process_time()
    nst = [len(d["trace_r"]) for d in datas]
    verdict = np.zeros(sum(nst), bool)
    n_traced = 0
    if opts.cutoff and sum(nst):
        verdict, n_traced = cutoff.trace_gated(
            np.concatenate([d["trace_r"] for d in datas]),
            np.concatenate([d["trace_u"] for d in datas]),
            np.concatenate([d["trace_R"] for d in datas]), 1,
            model=opts.model, date=opts.date)
    cpu_trace = time.process_time() - c0
    grid = Grid(caps, sky.E_BINS, sky.CZ_BINS, sky.AZ_BINS)
    off = 0
    out = []
    for d, ns, ti in zip(datas, nst, tis):
        c1 = time.process_time()
        v = verdict[off:off + ns]
        off += ns
        A = verdicts(d, v, opts.cutoff, opts.n_az)
        tal = tally_entries(grid, d["ent"], A)
        cnt = d["counts"]
        share = (ns / sum(nst)) if sum(nst) else 1.0 / len(tis)
        cnt["cpu_trace"] = cpu_trace * share
        cnt["n_traced"] = float(n_traced) * share
        cnt["n_allowed_traced"] = float(v.sum())
        cnt["cpu_replay"] += time.process_time() - c1
        dd = tal.to_dict()
        dd.update(d["cone"])
        meta = d["meta"].copy()
        meta[7] = meta[7] + cnt["cpu_trace"] + (time.process_time() - c1)
        dd["meta"] = meta
        dd["group"] = np.array([d["group"]])
        dd["m3b_counts"] = np.array([cnt[c] for c in COUNT_NAMES])
        dd["m3b_count_names"] = np.array(COUNT_NAMES)
        dd["m3b_opts"] = opts.signature()
        sky._atomic_savez(tfile(outdir, tag, ti), dd, compressed=True)
        out.append((ti, float(meta[7])))
        if not opts.keep_cfiles:
            os.remove(cfile(outdir, tag, ti))
    return out


# ---------------------------------------------------------------------------
# site map (1D reference cutoff), built once and cached
# ---------------------------------------------------------------------------
class SiteMap:
    """``cutoff.SiteCutoffMap``'s arrays, picklable and cheap to load."""

    def __init__(self, zen, az, rc):
        self.zen, self.az, self.rc = (np.asarray(x, float) for x in (zen, az, rc))
        self._f = None

    def __call__(self, zen_deg, az_deg):
        from scipy.interpolate import RegularGridInterpolator
        if self._f is None:
            self._f = RegularGridInterpolator((self.zen, self.az), self.rc,
                                              bounds_error=False, fill_value=None)
        z = np.clip(np.asarray(zen_deg, float), self.zen[0], self.zen[-1])
        a = (np.asarray(az_deg, float) - self.az[0]) % 360.0 + self.az[0]
        z, a = np.broadcast_arrays(z, a)
        return self._f(np.stack([z.ravel(), a.ravel()], -1)).reshape(z.shape)

    def allowed(self, rigidity, zen_deg, az_deg):
        return np.asarray(rigidity, float) > self(zen_deg, az_deg)


def site_map(lat, lon, model, date, n_jobs, cache_dir):
    fp = os.path.join(cache_dir, f"sitemap_{lat:.4f}_{lon:.4f}_{model}_"
                      f"{date:%Y%m%d}.npz")
    if os.path.exists(fp):
        d = np.load(fp)
        return SiteMap(d["zen"], d["az"], d["rc"])
    m = cutoff.SiteCutoffMap(lat, lon, n_jobs=n_jobs, model=model, date=date)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(fp, zen=m.zen, az=m.az, rc=m.rc, saturated=m.saturated)
    return SiteMap(m.zen, m.az, m.rc)


# ---------------------------------------------------------------------------
# tasks, groups, the job
# ---------------------------------------------------------------------------
def build_tasks(nshower, groups, theta_inj, chunk, seed0, nstrata=None,
                alloc=None):
    """``sky.build_tasks`` with an optional explicit allocation ``alloc``:
    a dict ``{(group, stratum): n_tot}`` replacing the group fraction x
    ``E^-0.8`` default (milestone-3b re-allocation, task B)."""
    if alloc is None:
        return sky.build_tasks(nshower, groups, theta_inj, chunk, seed0, nstrata)
    tasks = []
    for gi, g in enumerate(groups):
        for k in range(len(sky.STRATA) - 1):
            n_tot = int(alloc.get((g, k), 0))
            if n_tot <= 0:
                continue
            ck = max(100, int(chunk * sky._SC[0] / sky._SC[k]))
            left, c = n_tot, 0
            while left > 0:
                m = min(ck, left)
                tasks.append((g, k, int(m), float(theta_inj),
                              seed0 + 1000003 * gi + 10007 * k + 13 * c,
                              int(n_tot)))
                left -= m
                c += 1
    return tasks


def make_groups(counts, batch, max_files):
    """Consecutive task indices grouped until >= ``batch`` trace states (or
    ``max_files`` files)."""
    groups, cur, acc = [], [], 0
    for ti in sorted(counts):
        cur.append(ti)
        acc += counts[ti]
        if acc >= batch or len(cur) >= max_files:
            groups.append(cur)
            cur, acc = [], 0
    if cur:
        groups.append(cur)
    return groups


def run(nshower, nproc, groups, theta_inj, caps, cache, opts, n_pool=2000,
        e_nu_min=0.1, site="kamioka", chunk=2000, seed0=20261011, model="H3A",
        outdir=".", tag="m3b", nstrata=None, alloc=None, work=None,
        log=None):
    import multiprocessing as mp
    if log is None:
        def log(msg):
            print(msg, flush=True)
    lat, lon = sky.SITES[site]
    work = work or os.path.join(outdir, "..", "work")
    os.makedirs(outdir, exist_ok=True)
    tasks = build_tasks(nshower, groups, theta_inj, chunk, seed0, nstrata, alloc)
    tdir = task_dir(outdir, tag)
    os.makedirs(tdir, exist_ok=True)
    mpath = os.path.join(tdir, "manifest.npz")
    sig = np.array([[t[1], t[2], t[4], t[5]] for t in tasks], np.int64)
    if os.path.exists(mpath):
        old = np.load(mpath, allow_pickle=False)
        if not (old["sig"].shape == sig.shape and np.all(old["sig"] == sig)
                and np.allclose(old["opts"], opts.signature())
                and float(old["theta_inj"][0]) == float(theta_inj)):
            raise RuntimeError(f"{tdir} holds a different job")
    man = {"sig": sig, "ntask": np.array([len(tasks)]),
           "groups": np.array(list(groups)), "opts": opts.signature(),
           "theta_inj": np.array([float(theta_inj)]),
           "caps": np.array(caps, float), "nshower": np.array([nshower]),
           "nproc": np.array([nproc]), "site": np.array([site]),
           "model": np.array([model]), "seed0": np.array([seed0]),
           "chunk": np.array([chunk]), "collinear": np.array([False]),
           "field_model": np.array([opts.model]),
           "date": np.array([opts.date.isoformat()])}
    sky._atomic_savez(mpath, man)
    fld = None
    if opts.needs_field():
        fld = __import__("field").FieldGrid(opts.model, opts.date,
                                            cache_dir=work)
    smap = site_map(lat, lon, opts.model, opts.date, nproc, work) \
        if opts.cutoff else None
    log(f"{len(tasks)} tasks, {nshower} primaries, groups={groups}, "
        f"theta_inj={theta_inj}, caps={caps}, opts={vars(opts)}")
    t0 = time.time()
    ctx = mp.get_context("fork")
    todo = [t + (i, cfile(outdir, tag, i)) for i, t in enumerate(tasks)
            if not (os.path.exists(cfile(outdir, tag, i))
                    or os.path.exists(tfile(outdir, tag, i)))]
    counts = {}
    with ctx.Pool(nproc, initializer=_init,
                  initargs=(cache, n_pool, e_nu_min, lat, lon, caps, model,
                            opts, fld, smap)) as pool:
        done, nsh, cpu = 0, 0, 0.0
        for r in pool.imap_unordered(cascade_task, todo, chunksize=1):
            done += 1
            nsh += r["n"]
            cpu += r["cpu"]
            if done % 20 == 0 or done == len(todo):
                dt = time.time() - t0
                log(f"  phase 1: {done}/{len(todo)} tasks, {nsh} showers, "
                    f"{dt:.0f} s, {nsh / max(cpu, 1e-9):.1f} showers/s/core")
        t1 = time.time()
        for i in range(len(tasks)):
            if os.path.exists(tfile(outdir, tag, i)):
                continue
            counts[i] = int(np.load(cfile(outdir, tag, i)[:-4] + ".n.npy")[0])
        jobs = [(g, outdir, tag) for g in make_groups(
            counts, opts.trace_batch, opts.max_files_per_group)]
        log(f"phase 1 done in {t1 - t0:.0f} s; phase 2: {len(jobs)} groups, "
            f"{sum(counts.values())} copy states to trace")
        done = 0
        for r in pool.imap_unordered(trace_replay_task, jobs, chunksize=1):
            done += 1
            if done % 10 == 0 or done == len(jobs):
                log(f"  phase 2: {done}/{len(jobs)} groups, "
                    f"{time.time() - t1:.0f} s")
    wall = time.time() - t0
    man["wall_s"] = np.array([wall])
    sky._atomic_savez(mpath, man)
    fp = merge(outdir, tag)
    log(f"wrote {fp} ({wall:.0f} s)")
    return fp


def merge(outdir, tag):
    """Sum the t-files of a job into ``<tag>_total.npz``."""
    man = dict(np.load(os.path.join(task_dir(outdir, tag), "manifest.npz"),
                       allow_pickle=False))
    tot, meta = None, []
    for i in range(int(man["ntask"][0])):
        d = dict(np.load(tfile(outdir, tag, i), allow_pickle=False))
        meta.append(np.concatenate([[i], d["meta"], d["m3b_counts"]]))
        if tot is None:
            tot = {k: v.copy() for k, v in d.items() if k in SUMMED}
            keep = {k: v for k, v in d.items() if k not in tot}
        else:
            for k in tot:
                tot[k] = tot[k] + d[k]
    out = dict(keep)
    out.update(tot)
    out["task_meta"] = np.array(meta)
    out.pop("meta", None)
    fp = os.path.join(outdir, f"{tag}_total.npz")
    np.savez(fp, **out)
    return fp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nshower", type=int, default=100000)
    ap.add_argument("--nproc", type=int, default=36)
    ap.add_argument("--groups", nargs="+",
                    default=["p", "He", "CNO", "MgSi", "Fe"])
    ap.add_argument("--theta-inj", type=float, default=40.0)
    ap.add_argument("--caps", type=float, nargs="+",
                    default=[2.5, 5.0, 7.5, 10.0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--n-pool", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--model", default="H3A")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--work", default=None)
    ap.add_argument("--tag", default="m3b")
    ap.add_argument("--nstrata", type=int, default=None)
    ap.add_argument("--alloc", default=None,
                    help="npz with keys group, stratum, n (task B allocation)")
    ap.add_argument("--no-cutoff", action="store_true")
    ap.add_argument("--no-bend-primary", action="store_true")
    ap.add_argument("--no-muon-field", action="store_true")
    ap.add_argument("--r-thr", type=float, default=30.0)
    ap.add_argument("--n-az", type=int, default=N_AZ)
    ap.add_argument("--no-skip-gated", action="store_true")
    ap.add_argument("--pend-all", action="store_true")
    ap.add_argument("--margin-extra", type=float, default=2.0)
    ap.add_argument("--trace-batch", type=int, default=10000)
    ap.add_argument("--reanchor", action="store_true",
                    help="bias study: re-anchor each rotated copy of a reused "
                         "cascade on the copy's own exact bent primary")
    ap.add_argument("--merge-only", action="store_true")
    a = ap.parse_args(argv)
    if a.merge_only:
        print("wrote", merge(a.outdir, a.tag))
        return
    opts = Options(cutoff=not a.no_cutoff, bend_primary=not a.no_bend_primary,
                   muon_field=not a.no_muon_field, r_thr=a.r_thr, n_az=a.n_az,
                   skip_gated=not a.no_skip_gated, pend_all=a.pend_all,
                   margin_extra_deg=a.margin_extra, trace_batch=a.trace_batch,
                   reanchor=a.reanchor)
    alloc = None
    if a.alloc:
        d = np.load(a.alloc)
        alloc = {(str(g), int(k)): int(n) for g, k, n in
                 zip(d["group"], d["stratum"], d["n"])}
    run(a.nshower, a.nproc, a.groups, a.theta_inj, tuple(a.caps), a.cache,
        opts, n_pool=a.n_pool, chunk=a.chunk, seed0=a.seed, model=a.model,
        outdir=a.outdir, tag=a.tag, nstrata=a.nstrata, alloc=alloc,
        work=a.work)


if __name__ == "__main__":
    main()
