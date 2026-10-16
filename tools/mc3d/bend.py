"""Charged-particle transport in the geomagnetic field, vectorised (milestone 3b).

Two transports, both array-valued over many independent states so that the
``N_az`` azimuthal copies of a shower (and all the showers of a task) are
moved in lock-step with one field call per step:

``bend_primary``
    The primary from the injection sphere to its first interaction.  Honda
    bends it ("all the nucleons carried by the cosmic ray nuclei are treated
    as protons with double rigidity ... before the first interaction",
    astro-ph/0404457 sec. IV); ``shower.transport_hadron`` moves it in a
    straight line.  **Superposition approximation**: the nucleon handed to the
    cascade (``p`` or ``n``, drawn ``Z/A``) is bent with the rigidity of its
    *parent nucleus*, ``R = (A/Z) p_nucleon``, and charge ``+1``, until its
    own first interaction, which is drawn from the *nucleon* interaction
    length.  A real nucleus fragments earlier (its own interaction length is
    shorter), after which its neutrons go straight and its protons bend at
    ``p_nucleon`` -- i.e. half the rigidity for ``A/Z = 2``.  Both errors are
    confined to the ~``A^-0.3`` fraction of the first-interaction path that
    differs, and to the ~26% of nucleons that come in nuclei (H3a, 10 GeV).

``transport_muons``
    The muon segment, with continuous ``dE/dx`` and the field, from its
    production point to its decay (or the ground / the top of the atmosphere
    / range-out).  It is the exact per-copy re-transport of milestone 3b: the
    hadronic cascade of a shower is reused across its azimuthal copies by a
    rigid rotation about the site vertical, but every muon segment is
    re-transported in the copy's *own* field (task A of milestone 3b).  With
    ``field=None`` it reproduces ``shower.transport_muon``.

Plus the vectorised polarised Michel decay (``michel_decay``) that turns the
re-transported muons into neutrinos with common random numbers across copies.

Integration scheme (both transports): per step, the direction rotates about
``B_hat`` by ``theta = -q Q_C |B| ds / p`` (exact for a uniform field); the
position advances along the **chord** direction ``R(theta/2) u`` (second
order); the grammage of the chord is a 5-point Gauss-Legendre integral of the
CORSIKA density (the altitude may change by at most ~1.5 scale heights per
step, where GL5 on an exponential is exact to ~1e-8).
"""

from __future__ import annotations

import numpy as np

import atmosphere as atm
from constants import CTAU_CM, M_MU, R_EARTH_CM
from field import Q_C, rotate

R_TOP_CM = R_EARTH_CM + atm.H_TOP_CM

# Gauss-Legendre 5 on [0, 1]
_GLX = 0.5 + 0.5 * np.array([-0.906179845938664, -0.538469310105683, 0.0,
                             0.538469310105683, 0.906179845938664])
_GLW = 0.5 * np.array([0.236926885056189, 0.478628670499366,
                       0.568888888888889, 0.478628670499366,
                       0.236926885056189])

_HSCALE = np.array([atm._CATM[0], atm._CATM[1], atm._CATM[2], atm._CATM[3],
                    1.0e9])


def _alt(r):
    return np.sqrt(np.einsum("ij,ij->i", r, r)) - R_EARTH_CM


def scale_height(h):
    i = np.clip(np.searchsorted(atm._HLAY, h, side="right") - 1, 0, 4)
    return _HSCALE[i]


def chord_grammage(r, u, ds):
    """Grammage [g/cm2] along the straight chords ``r + s u``, ``s in [0, ds]``
    (rows), by 5-point Gauss-Legendre."""
    r2 = np.einsum("ij,ij->i", r, r)
    ru = np.einsum("ij,ij->i", r, u)
    s = ds[:, None] * _GLX[None, :]
    h = np.sqrt(r2[:, None] + 2.0 * s * ru[:, None] + s * s) - R_EARTH_CM
    rho = atm.density(h)
    return ds * (rho @ _GLW)


def exit_path(r, u):
    """``(s, hit_ground)``: straight-line path to the ground sphere (near
    root only -- see ``atmosphere.path_to_exit``) or to the top of the
    atmosphere, rows."""
    r2 = np.einsum("ij,ij->i", r, r)
    b = np.einsum("ij,ij->i", r, u)
    disc_g = b * b - (r2 - R_EARTH_CM ** 2)
    s_g = -b - np.sqrt(np.maximum(disc_g, 0.0))
    hit = (disc_g >= 0.0) & (s_g > 1.0)
    on_ground = (r2 <= (R_EARTH_CM + 1.0) ** 2) & (b <= 0.0)
    disc_t = b * b - (r2 - R_TOP_CM ** 2)
    s_t = np.maximum(-b + np.sqrt(np.maximum(disc_t, 0.0)), 0.0)
    s = np.where(hit, s_g, s_t)
    s = np.where(on_ground, 0.0, s)
    return s, hit | on_ground


def _invert_chord(r, u, ds, target, n_iter=40):
    """Path ``s in [0, ds]`` along each chord at which the grammage reaches
    ``target`` (bisection on the GL5 chord integral)."""
    lo = np.zeros_like(ds)
    hi = ds.copy()
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        x = chord_grammage(r, u, mid)
        below = x < target
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# the primary
# ---------------------------------------------------------------------------
#: primary status codes
P_INTERACT, P_GROUND, P_ESCAPE = 0, 1, 2


def bend_primary(r0, u0, rigidity, dX, field, charge=1.0, ds_max=2.0e6,
                 dtheta_max=0.03, max_steps=2000):
    """Transport primaries (rows) to their first interaction in the field.

    ``rigidity`` [GV] (nucleus), ``dX`` the sampled interaction depth
    [g/cm2], ``field`` a callable ``r_cm (n,3) -> B (n,3) [T]`` (or ``None``:
    straight line, then this is ``atmosphere.advance_grammage``).

    Returns ``(status, r, u, path_cm, bend_rad)`` with ``status`` in
    ``P_INTERACT / P_GROUND / P_ESCAPE``, the position and direction at the
    interaction (or where the primary left the atmosphere), the arc length and
    the angle between ``u0`` and the final direction.
    """
    r = np.array(np.atleast_2d(r0), float)
    u = np.array(np.atleast_2d(u0), float)
    n = len(r)
    R = np.broadcast_to(np.asarray(rigidity, float), (n,)).copy()
    dX = np.broadcast_to(np.asarray(dX, float), (n,)).copy()
    q = np.broadcast_to(np.asarray(charge, float), (n,)).copy()
    X = np.zeros(n)
    path = np.zeros(n)
    status = np.full(n, -1, int)
    live = np.arange(n)
    for _ in range(max_steps):
        if live.size == 0:
            break
        rl, ul = r[live], u[live]
        h = _alt(rl)
        dhds = np.einsum("ij,ij->i", rl, ul) / (h + R_EARTH_CM)
        ds = np.full(live.size, ds_max)
        ds = np.minimum(ds, 1.5 * scale_height(h) / np.maximum(np.abs(dhds), 1e-6))
        if field is not None:
            B = field(rl)
            bn = np.linalg.norm(B, axis=1)
            ds = np.minimum(ds, dtheta_max * R[live] / np.maximum(Q_C * bn, 1e-30)
                            * 100.0)
            bax = B / np.maximum(bn, 1e-30)[:, None]
            k = -q[live] * Q_C * bn * 1e-2 / R[live]     # rad per cm
        else:
            bax = np.tile([0.0, 0.0, 1.0], (live.size, 1))
            k = np.zeros(live.size)
        ds = np.maximum(ds, 1.0e2)
        um = rotate(ul, bax, 0.5 * k * ds)
        s_ex, hit_g = exit_path(rl, um)
        at_bound = s_ex <= ds
        ds = np.where(at_bound, s_ex, ds)
        um = rotate(ul, bax, 0.5 * k * ds)
        xc = chord_grammage(rl, um, ds)
        done = X[live] + xc >= dX[live]
        # interaction inside the chord
        if np.any(done):
            d = np.where(done)[0]
            s_star = _invert_chord(rl[d], um[d], ds[d], dX[live[d]] - X[live[d]])
            idx = live[d]
            r[idx] = rl[d] + s_star[:, None] * um[d]
            u[idx] = rotate(ul[d], bax[d], k[d] * s_star)
            path[idx] += s_star
            X[idx] = dX[idx]
            status[idx] = P_INTERACT
        # left the atmosphere inside the chord without interacting
        out = ~done & at_bound
        if np.any(out):
            o = np.where(out)[0]
            idx = live[o]
            r[idx] = rl[o] + ds[o][:, None] * um[o]
            u[idx] = rotate(ul[o], bax[o], k[o] * ds[o])
            path[idx] += ds[o]
            status[idx] = np.where(hit_g[o], P_GROUND, P_ESCAPE)
        go = ~done & ~at_bound
        if np.any(go):
            g = np.where(go)[0]
            idx = live[g]
            r[idx] = rl[g] + ds[g][:, None] * um[g]
            u[idx] = rotate(ul[g], bax[g], k[g] * ds[g])
            u[idx] /= np.linalg.norm(u[idx], axis=1)[:, None]
            X[idx] += xc[g]
            path[idx] += ds[g]
        live = live[go]
    status[status < 0] = P_ESCAPE
    u0n = np.atleast_2d(np.asarray(u0, float))
    cosb = np.clip(np.einsum("ij,ij->i", u, u0n), -1.0, 1.0)
    return status, r, u, path, np.arccos(cosb)


# ---------------------------------------------------------------------------
# muons
# ---------------------------------------------------------------------------
#: muon outcome codes
MU_DECAY, MU_GROUND, MU_ESCAPE, MU_STOP = 0, 1, 2, 3

_LOG_P = np.log(np.array([0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0,
                          10.0, 20.0, 50.0, 100.0, 1000.0]))
_LOG_V = np.log(np.array([9.20, 6.05, 3.62, 2.35, 2.05, 1.88, 1.84, 1.92,
                          2.06, 2.19, 2.32, 2.51, 2.66, 3.30]))


def dedx_air(e_tot):
    """Muon stopping power [GeV cm2/g] (``shower.dedx_air``, vectorised)."""
    p = np.sqrt(np.maximum(e_tot * e_tot - M_MU * M_MU, 1e-8))
    return 1e-3 * np.exp(np.interp(np.log(p), _LOG_P, _LOG_V))


_MU_DX_MAX = 30.0          # g/cm2 of energy loss per step (shower._MU_DX_MAX)
_MU_DS_BEND = 5.0e5        # 5 km cap with the field on (shower._MU_DS_CAP)


def transport_muons(pdg, e_tot, r, u, t_target, field, energy_loss=True,
                    dtheta_max=0.05, max_steps=4000):
    """Transport muon segments (rows) to decay / ground / escape / stop.

    ``pdg`` 13 (mu-) or -13 (mu+), ``e_tot`` [GeV], ``t_target`` the proper
    decay time in units of the lifetime (the same exponential variate that
    ``shower.transport_muon`` draws), ``field`` as in :func:`bend_primary`.
    Mirrors ``shower.transport_muon`` step for step in its physics (energy
    loss with a mid-point correction, proper time accumulated with the
    step-start ``beta gamma``, decay inside the step at the linearly
    interpolated point); the field rotates the direction about ``B_hat``.

    Returns ``(kind, r, u, e)`` arrays.
    """
    pdg = np.asarray(pdg)
    n = len(pdg)
    r = np.array(r, float).reshape(n, 3)
    u = np.array(u, float).reshape(n, 3)
    e = np.array(e_tot, float).reshape(n).copy()
    tt = np.asarray(t_target, float).reshape(n)
    q = np.where(pdg < 0, 1.0, -1.0)
    ctau = CTAU_CM[13]
    tacc = np.zeros(n)
    kind = np.full(n, -1, int)
    live = np.arange(n)
    m = M_MU
    for _ in range(max_steps):
        if live.size == 0:
            break
        rl, ul, el = r[live], u[live], e[live]
        h = _alt(rl)
        rn = h + R_EARTH_CM
        dhds = np.einsum("ij,ij->i", rl, ul) / rn
        p = np.sqrt(np.maximum(el * el - m * m, 0.0))
        bg = p / m
        rho = atm.density(np.clip(h, 0.0, None))
        ds = np.where(rho > 1e-30, _MU_DX_MAX / np.maximum(rho, 1e-300), np.inf)
        ds = np.minimum(ds, 1.5 * scale_height(h)
                        / np.maximum(np.abs(dhds), 1e-6))
        ds = np.minimum(ds, 0.5 * bg * ctau)
        if field is not None:
            B = field(rl)
            bn = np.linalg.norm(B, axis=1)
            ds = np.minimum(ds, _MU_DS_BEND)
            ds = np.minimum(ds, dtheta_max * p / np.maximum(Q_C * bn, 1e-30)
                            * 100.0)
            bax = B / np.maximum(bn, 1e-30)[:, None]
            k = -q[live] * Q_C * bn * 1e-2 / np.maximum(p, 1e-9)   # rad/cm
        else:
            bax = np.tile([0.0, 0.0, 1.0], (live.size, 1))
            k = np.zeros(live.size)
        ds = np.maximum(ds, 1.0e2)
        um = rotate(ul, bax, 0.5 * k * ds)
        s_ex, hit_g = exit_path(rl, um)
        at_bound = s_ex <= ds
        ds = np.where(at_bound, s_ex, ds)
        # boundary reached at the start of the step
        zero = s_ex <= 0.0
        dt = ds / np.maximum(bg * ctau, 1e-30)
        dec = (tacc[live] + dt >= tt[live]) & ~zero
        # --- decays inside the step
        if np.any(dec):
            d = np.where(dec)[0]
            idx = live[d]
            frac = (tt[idx] - tacc[idx]) / np.maximum(dt[d], 1e-30)
            dsd = ds[d] * frac
            umd = rotate(ul[d], bax[d], 0.5 * k[d] * dsd)
            if energy_loss:
                dX = chord_grammage(rl[d], umd, dsd)
                e[idx] = np.maximum(el[d] - dedx_air(el[d]) * dX, m)
            r[idx] = rl[d] + dsd[:, None] * umd
            u[idx] = rotate(ul[d], bax[d], k[d] * dsd)
            kind[idx] = MU_DECAY
        # --- at a boundary already
        if np.any(zero):
            z = np.where(zero)[0]
            kind[live[z]] = np.where(hit_g[z], MU_GROUND, MU_ESCAPE)
        go = ~dec & ~zero
        if np.any(go):
            g = np.where(go)[0]
            idx = live[g]
            if energy_loss:
                dX = chord_grammage(rl[g], um[g], ds[g])
                emid = np.maximum(el[g] - 0.5 * dedx_air(el[g]) * dX, m)
                enew = el[g] - dedx_air(emid) * dX
            else:
                enew = el[g]
            r[idx] = rl[g] + ds[g][:, None] * um[g]
            unew = rotate(ul[g], bax[g], k[g] * ds[g])
            u[idx] = unew / np.linalg.norm(unew, axis=1)[:, None]
            e[idx] = np.maximum(enew, m)
            tacc[idx] += dt[g]
            stop = enew <= m * 1.0005
            kind[idx[stop]] = MU_STOP
            # reached the boundary at the end of this step: the next step
            # sees s_exit = 0 and closes it (exactly as transport_muon)
        live = live[kind[live] < 0]
    kind[kind < 0] = MU_ESCAPE
    return kind, r, u, e


# ---------------------------------------------------------------------------
# polarised Michel decay, vectorised, with common random numbers
# ---------------------------------------------------------------------------
def michel_decay(pdg, e_tot, u, pol, rnd, ref_axis):
    """Both Michel neutrinos of each decaying muon (rows).

    ``rnd`` (n, 6) uniforms: ``(x, cos, phi)`` for the nu_mu-type and for the
    nu_e-type neutrino.  The decay frame's transverse axes are built from
    ``ref_axis`` (the site vertical), which is invariant under the azimuthal
    copy rotations, so with common random numbers a copy's decay is the exact
    rotation of the original's when the field is off.  Physics identical to
    ``decays.michel_sample`` (inverse-CDF ``x``, ``1 + s A(x) P_L cos``,
    ``s = -1`` for mu+, boost along the muon momentum).

    Returns ``(pdg_nu (n,2), e_nu (n,2), dir_nu (n,2,3))``.
    """
    from decays import _CDF, _XE, michel_asym
    pdg = np.asarray(pdg)
    n = len(pdg)
    ch = np.where(pdg < 0, 1.0, -1.0)            # mu+ has PDG -13
    s = np.where(ch > 0, -1.0, 1.0)
    e = np.asarray(e_tot, float)
    p = np.sqrt(np.maximum(e * e - M_MU * M_MU, 0.0))
    beta = p / e
    gam = e / M_MU
    z = np.asarray(u, float)
    ref = np.asarray(ref_axis, float)
    e1 = np.cross(np.broadcast_to(ref, z.shape), z)
    nr = np.linalg.norm(e1, axis=1)
    bad = nr < 1e-9
    if np.any(bad):
        alt = np.cross(np.array([1.0, 0.0, 0.0]), z[bad])
        alt_bad = np.linalg.norm(alt, axis=1) < 1e-9
        alt[alt_bad] = np.cross(np.array([0.0, 1.0, 0.0]), z[bad][alt_bad])
        e1[bad] = alt
        nr = np.linalg.norm(e1, axis=1)
    e1 = e1 / nr[:, None]
    e2 = np.cross(z, e1)
    pdg_out = np.zeros((n, 2), int)
    e_out = np.zeros((n, 2))
    d_out = np.zeros((n, 2, 3))
    pdg_out[:, 0] = np.where(ch > 0, -14, 14)
    pdg_out[:, 1] = np.where(ch > 0, 12, -12)
    for j, kindname in enumerate(("numu", "nue")):
        x = np.interp(rnd[:, 3 * j], _CDF[kindname], _XE)
        a = michel_asym(x, kindname)
        aa = s * a * np.asarray(pol, float)
        v = rnd[:, 3 * j + 1]
        small = np.abs(aa) < 1e-9
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.where(small, 2.0 * v - 1.0,
                         (-1.0 + np.sqrt(np.maximum(
                             1.0 + 2.0 * aa * (2.0 * v - 1.0) + aa * aa, 0.0)))
                         / np.where(small, 1.0, aa))
        c = np.clip(c, -1.0, 1.0)
        ph = 2.0 * np.pi * rnd[:, 3 * j + 2]
        st = np.sqrt(np.maximum(1.0 - c * c, 0.0))
        es = 0.5 * M_MU * x
        e_lab = gam * es * (1.0 + beta * c)
        p_par = gam * es * (c + beta)
        p_perp = es * st
        vec = (p_par[:, None] * z
               + p_perp[:, None] * (np.cos(ph)[:, None] * e1
                                    + np.sin(ph)[:, None] * e2))
        nv = np.linalg.norm(vec, axis=1)
        d_out[:, j] = vec / np.maximum(nv, 1e-300)[:, None]
        e_out[:, j] = e_lab
    return pdg_out, e_out, d_out


def max_nu_angle(e_mu, e_nu_min):
    """Largest lab angle [rad] between a muon of total energy ``e_mu`` and a
    decay neutrino of energy >= ``e_nu_min`` (``E_nu <= (m/2) / (gamma (1 -
    beta cos))``); 0 where no such neutrino exists, pi where every angle is
    reachable."""
    e = np.asarray(e_mu, float)
    gam = np.maximum(e / M_MU, 1.0)
    beta = np.sqrt(np.maximum(1.0 - 1.0 / gam ** 2, 1e-12))
    c = (1.0 - M_MU / (2.0 * gam * e_nu_min)) / beta
    return np.where(c > 1.0, 0.0, np.arccos(np.clip(c, -1.0, 1.0)))
