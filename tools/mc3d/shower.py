"""Shower driver: transport of one primary through the spherical atmosphere.

Modes
-----
``collinear=True``   every secondary inherits its parent's direction and no
                     magnetic field acts.  This is the *1D reference computed on
                     the same showers*: it is the sigma_theta -> 0, B -> 0 limit
                     that must reproduce MCEq (milestone-1 closure gate).
``collinear=False``  full 3D: secondaries take the generator's / decay
                     kinematics' directions and charged particles are bent by
                     the IGRF field.

Tracking thresholds (PHASE2_PLAN.md sec. 4.4).  A particle is dropped when it
cannot make a neutrino above ``e_nu_min`` at the detector:

* pi+-  below ``e_nu_min / (1 - m_mu^2/m_pi^2) = e_nu_min / 0.4270``
* K+-, K_L below ``e_nu_min / 0.9544``
* mu+-  below ``e_nu_min``  (E_nu <= E_mu)
* nucleons below ``m_N + 4 e_nu_min`` (pion-production threshold plus the
  inelasticity needed to reach the pion cut; deliberately conservative)
* pi0, gammas, electrons, hyperons and charm are dropped outright -- the EM
  component feeds nothing back into the hadronic cascade in MCEq either, so
  dropping it keeps the two calculations comparable.  (Charm matters only above
  ~10^5 GeV, far outside the 3D/1D table's range.)
"""

from __future__ import annotations

import numpy as np

import atmosphere as atm
from constants import (CTAU_CM, MASS, M_MU, M_N, NEUTRINOS, R_EARTH_CM)
from decays import decay as kin_decay

# PDG muon stopping power in air [MeV cm2/g] (same table as
# tools/mceq3d/muon_segment_mc.dedx_air)
_DEDX_P = np.array([0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0,
                    10.0, 20.0, 50.0, 100.0, 1000.0])
_DEDX_V = np.array([9.20, 6.05, 3.62, 2.35, 2.05, 1.88, 1.84, 1.92, 2.06,
                    2.19, 2.32, 2.51, 2.66, 3.30])
_LOG_P, _LOG_V = np.log(_DEDX_P), np.log(_DEDX_V)


def dedx_air(e_tot):
    """Muon total stopping power [GeV cm2/g] vs total energy."""
    p = np.sqrt(max(e_tot * e_tot - M_MU * M_MU, 1e-8))
    return 1e-3 * float(np.exp(np.interp(np.log(p), _LOG_P, _LOG_V)))


_R_PI = (M_MU / MASS[211]) ** 2
_R_K = (M_MU / MASS[321]) ** 2


def thresholds(e_nu_min):
    t = {}
    for p in (211, -211):
        t[p] = e_nu_min / (1.0 - _R_PI)
    for p in (321, -321, 130, 310):
        t[p] = e_nu_min / (1.0 - _R_K)
    for p in (13, -13):
        # E_nu <= (E_mu + p_mu)/2 <= E_mu, so E_mu < e_nu_min can be dropped.
        # (The earlier "e_nu_min + m_mu" cut threw away muons that CAN make a
        # neutrino above threshold.)
        t[p] = max(M_MU, e_nu_min)
    for p in (2212, -2212, 2112, -2112):
        t[p] = M_N + e_nu_min / (1.0 - _R_PI)
    for p in (3122, -3122):
        t[p] = MASS[3122] + e_nu_min / (1.0 - _R_PI)
    return t


class Config:
    def __init__(self, collinear=True, bfield=None, energy_loss=True,
                 e_nu_min=0.1, decay_mode="kinematic", polarisation=True,
                 max_particles=200000):
        self.collinear = bool(collinear)
        self.bfield = bfield
        self.energy_loss = bool(energy_loss)
        self.e_nu_min = float(e_nu_min)
        self.decay_mode = decay_mode
        self.polarisation = bool(polarisation)
        self.max_particles = int(max_particles)
        self.thr = thresholds(e_nu_min)


# ---------------------------------------------------------------------------
# transport of a single stable-in-flight particle
# ---------------------------------------------------------------------------
_MU_MAX_STEPS = 4000
_MU_DX_MAX = 30.0        # g/cm2 of energy loss per step
_MU_DS_CAP = 5.0e5       # 5 km cap when the field is on (bend resolution)


def _mu_step(h, dhds, s_left, l_dec, bend=False):
    """Muon step length.

    Adaptive on the *energy loss* (a fixed grammage per step) rather than on the
    density scale height: high above the atmosphere rho -> 0 and the muon can
    cross tens of km in one step, which is what makes the transport affordable.
    Additional caps: 2 scale heights of altitude change (so the exact grammage
    integrator inside the step stays cheap), half a decay length (so the
    linear-in-ds decay point stays accurate) and, with the field on, 5 km.
    """
    rho = atm.density(h)
    ds = _MU_DX_MAX / rho if rho > 1e-30 else np.inf
    hs = atm.scale_height(h)
    if abs(dhds) > 1e-6:
        ds = min(ds, 2.0 * hs / abs(dhds))
    ds = min(ds, 0.5 * l_dec)
    if bend:
        ds = min(ds, _MU_DS_CAP)
    return float(min(max(ds, 1.0e2), s_left))


def _boundary(r, u):
    """Path length to the ground or to the top of the atmosphere."""
    return atm.path_to_exit(r, u)


def transport_hadron(rng, pdg, e_tot, r, u, backend):
    """Advance a hadron to its interaction / decay / boundary.

    Returns ``(kind, r_new)`` with ``kind in {'interact', 'decay', 'ground',
    'escape'}``.
    """
    lam = backend.lambda_int(pdg, e_tot) if backend.has_interaction(pdg) else np.inf
    ctau = CTAU_CM.get(int(pdg), np.inf)
    m = MASS[int(pdg)]
    if np.isfinite(ctau) and m > 0.0:
        p = np.sqrt(max(e_tot * e_tot - m * m, 0.0))
        s_dec = -np.log(rng.random()) * (p / m) * ctau
    else:
        s_dec = np.inf
    s_bound, hit_ground = _boundary(r, u)
    s_max = min(s_dec, s_bound)
    if np.isfinite(lam):
        dX = -np.log(rng.random()) * lam
        s_int, _, hit = atm.advance_grammage(r, u, dX, s_max)
        if hit:
            return "interact", r + s_int * u
    if s_dec < s_bound:
        return "decay", r + s_dec * u
    return ("ground" if hit_ground else "escape"), r + s_bound * u


def transport_muon(rng, pdg, e_tot, r, u, cfg):
    """Advance a muon with continuous energy loss (and, in 3D mode, IGRF
    bending).  Returns ``(kind, r, u, E)``; ``kind in {'decay','ground',
    'escape','stop'}``."""
    m = M_MU
    ctau = CTAU_CM[int(pdg)]
    t_target = -np.log(rng.random())            # in units of the lifetime
    t_acc = 0.0
    charge = +1.0 if pdg < 0 else -1.0          # PDG 13 = mu-
    r = np.array(r, float)
    u = np.array(u, float)
    e = float(e_tot)
    for _ in range(_MU_MAX_STEPS):
        s_bound, hit_ground = _boundary(r, u)
        if s_bound <= 0.0:
            return ("ground" if hit_ground else "escape"), r, u, e
        rn = float(np.linalg.norm(r))
        h = rn - R_EARTH_CM
        dhds = float(np.dot(r, u)) / rn
        p = np.sqrt(max(e * e - m * m, 0.0))
        bg = p / m                                # beta*gamma
        ds = _mu_step(h, dhds, s_bound, bg * ctau,
                      bend=(not cfg.collinear) and cfg.bfield is not None)
        # remaining proper time available over this step
        dt = ds / max(bg * ctau, 1e-30)
        if t_acc + dt >= t_target:
            ds = ds * (t_target - t_acc) / max(dt, 1e-30)
            r_dec = r + ds * u
            if cfg.energy_loss:
                e = max(e - dedx_air(e) * atm.grammage(r, u, ds), m)
            return "decay", r_dec, u, e
        t_acc += dt
        r_new = r + ds * u
        if cfg.energy_loss:
            dX = atm.grammage(r, u, ds)
            if dX > 0.0:
                # midpoint correction so a large step is still accurate
                e_mid = max(e - 0.5 * dedx_air(e) * dX, m)
                e = e - dedx_air(e_mid) * dX
                if e <= m * 1.0005:
                    return "stop", r_new, u, m
        if (not cfg.collinear) and cfg.bfield is not None:
            b = cfg.bfield(np.array([r_new / 100.0]))[0]      # cm -> m, tesla
            # d u / ds = (q c / (p c [GeV])) u x B ; p in GeV/c, B in T
            p = np.sqrt(max(e * e - m * m, 0.0))
            k = charge * 2.99792458e-4 / max(p, 1e-9)        # rad per metre
            du = k * np.cross(u, b) * (ds / 100.0)
            u = u + du
            u /= np.linalg.norm(u)
        r = r_new
    return "escape", r, u, e


# ---------------------------------------------------------------------------
# the shower
# ---------------------------------------------------------------------------
def run_shower(rng, pdg0, e0, r0, u0, backend, cfg, scorer, weight=1.0):
    """Track one primary and everything it makes."""
    stack = [(int(pdg0), float(e0), np.array(r0, float), np.array(u0, float),
              float(weight), 0.0)]
    n_done = 0
    n_escape = 0
    while stack:
        pdg, e, r, u, w, pol = stack.pop()
        n_done += 1
        if n_done > cfg.max_particles:
            break
        if pdg in NEUTRINOS:
            if e >= cfg.e_nu_min:
                scorer.add(pdg, e, r, u, w)
            continue
        thr = cfg.thr.get(pdg)
        if thr is None or e < thr:
            continue
        if abs(pdg) == 13:
            kind, r, u, e = transport_muon(rng, pdg, e, r, u, cfg)
            if kind in ("ground", "escape"):
                if kind == "escape":
                    n_escape += 1
                continue
            # decay (in flight or at rest)
            p4 = _p4(pdg, e, u)
            for d_pdg, d_p4, d_pol in kin_decay(
                    rng, pdg, p4, pol if cfg.polarisation else 0.0):
                stack.append(_child(d_pdg, d_p4, r, u, w, d_pol, cfg))
            continue
        kind, r_new = transport_hadron(rng, pdg, e, r, u, backend)
        if kind == "ground":
            continue
        if kind == "escape":
            n_escape += 1
            continue
        if kind == "decay":
            p4 = _p4(pdg, e, u)
            for d_pdg, d_p4, d_pol in kin_decay(rng, pdg, p4, 0.0):
                stack.append(_child(d_pdg, d_p4, r_new, u, w, d_pol, cfg))
            continue
        # interaction: backends return (pdg, E) [inclusive] or
        # (pdg, E, p_vec) [exclusive, generator frame with z along the parent]
        for item in backend.interact(rng, pdg, e):
            c, ec = int(item[0]), float(item[1])
            if c not in cfg.thr or ec < cfg.thr[c]:
                continue
            uc = u if (cfg.collinear or len(item) < 3) \
                else _rotate_from_z(u, item[2])
            stack.append((c, ec, r_new, uc, w, 0.0))
    return n_done, n_escape


def _p4(pdg, e, u):
    m = MASS[int(pdg)]
    p = np.sqrt(max(e * e - m * m, 0.0))
    return np.concatenate([[e], p * np.asarray(u, float)])


def _child(d_pdg, d_p4, r, u_parent, w, pol, cfg):
    e = float(d_p4[0])
    if cfg.collinear:
        uc = u_parent
    else:
        pv = d_p4[1:]
        n = np.linalg.norm(pv)
        uc = pv / n if n > 0 else u_parent
    return (int(d_pdg), e, np.array(r, float), np.array(uc, float), w, float(pol))


def _rotate_from_z(u, pvec):
    """Map a generator momentum (z along the projectile) into the lab frame."""
    from lorentz import rotate_to
    n = np.linalg.norm(pvec)
    if n <= 0:
        return u
    return rotate_to(u, pvec / n)
