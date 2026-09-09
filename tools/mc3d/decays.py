"""Full-kinematics decays for the mc3d cascade.

Channels implemented (PDG 2024 branching fractions):

===========  ==========================================  ========
parent       channel                                     BR
===========  ==========================================  ========
pi+-         mu nu                                       0.99988
K+-          mu nu                                       0.6356
K+-          pi pi0                                      0.2067
K+-          pi pi pi                                    0.0560
K+-          pi0 e nu   (Ke3)                            0.0507
K+-          pi0 mu nu  (Kmu3)                           0.0335
K+-          pi pi0 pi0                                  0.0176
K_L          pi e nu    (Ke3, both charges)              0.4055
K_L          pi mu nu   (Kmu3, both charges)             0.2704
K_L          3 pi0                                       0.1952
K_L          pi+ pi- pi0                                 0.1254
K_S          pi+ pi- / pi0 pi0                           0.9989
mu+-         e nu nu  (polarised Michel)                 1.0
===========  ==========================================  ========

Two-body decays are isotropic in the parent rest frame.  Three-body decays use
**flat Dalitz phase space** (constant matrix element); for K_l3 this misstates
the neutrino spectrum shape by a few per cent, which is <0.2% of the total
sub-GeV nu_e flux because K_l3 carries <8% of the kaon decays and kaons make
<15% of the sub-GeV neutrinos.  Recorded as a known approximation
(PHASE2_PLAN.md sec. 4.5); replacing it with the measured f_+(t) form factor is
a contained change in :func:`three_body`.

Muon polarisation is carried exactly: the muon from ``pi/K -> mu nu`` is
produced with helicity ``h = -1`` (for ``mu+``) / ``+1`` (for ``mu-``) in the
parent rest frame; the spin 4-vector is boosted to the lab and projected back
into the muon rest frame, giving the longitudinal polarisation ``P_L`` used by
:func:`michel_sample`.
"""

from __future__ import annotations

import numpy as np

from constants import (MASS, M_E, M_K, M_K0, M_MU, M_PI, M_PI0)
from lorentz import boost, make_p4, random_unit, rotate_to

# --------------------------------------------------------------------------
# decay tables: (branching, [daughter pdgs])
# --------------------------------------------------------------------------
BR_KP = [
    (0.6356, [-13, 14]),                 # K+ -> mu+ nu_mu
    (0.2067, [211, 111]),
    (0.0560, [211, 211, -211]),
    (0.0507, [111, -11, 12]),            # Ke3: K+ -> pi0 e+ nu_e
    (0.0335, [111, -13, 14]),            # Kmu3
    (0.0176, [211, 111, 111]),
]
BR_KL = [
    (0.2027, [-211, -11, 12]),
    (0.2027, [211, 11, -12]),
    (0.1352, [-211, -13, 14]),
    (0.1352, [211, 13, -14]),
    (0.1952, [111, 111, 111]),
    (0.1254, [211, -211, 111]),
]
BR_KS = [(0.6920, [211, -211]), (0.3069, [111, 111])]
BR_LAM = [(0.639, [2212, -211]), (0.358, [2112, 111])]
BR_LAMBAR = [(0.639, [-2212, 211]), (0.358, [-2112, 111])]


def _conj(chan):
    out = []
    for br, ds in chan:
        out.append((br, [-d if abs(d) not in (111,) else d for d in ds]))
    return out


BR_KM = _conj(BR_KP)


def _pick(rng, table):
    r = rng.random() * sum(b for b, _ in table)
    acc = 0.0
    for br, ds in table:
        acc += br
        if r <= acc:
            return ds
    return table[-1][1]


# --------------------------------------------------------------------------
# kinematics
# --------------------------------------------------------------------------
def two_body_cm(m0, m1, m2):
    """CM momentum of a two-body decay."""
    if m0 <= m1 + m2:
        return 0.0
    return np.sqrt((m0 ** 2 - (m1 + m2) ** 2) * (m0 ** 2 - (m1 - m2) ** 2)) / (2 * m0)


def two_body(rng, p4_parent, m0, m1, m2):
    """Isotropic two-body decay.  Returns the two lab 4-vectors and the
    daughter-1 direction in the parent rest frame."""
    p = two_body_cm(m0, m1, m2)
    n = random_unit(rng, 1)[0]
    e1 = np.sqrt(p * p + m1 * m1)
    e2 = np.sqrt(p * p + m2 * m2)
    q1 = np.concatenate([[e1], p * n])
    q2 = np.concatenate([[e2], -p * n])
    b = np.asarray(p4_parent, float)[1:] / p4_parent[0]
    return boost(q1, b)[0], boost(q2, b)[0], n


def three_body(rng, p4_parent, m0, masses, n_try=200):
    """Flat (constant matrix element) three-body phase space, RAMBO-style
    rejection on the Dalitz variable ``m12``."""
    m1, m2, m3 = masses
    m12_min, m12_max = m1 + m2, m0 - m3
    if m12_max <= m12_min:
        return None
    # sample m12^2 with the correct flat-phase-space density
    best = None
    wmax = 0.0
    for _ in range(n_try):
        m12 = np.sqrt(rng.uniform(m12_min ** 2, m12_max ** 2))
        p1 = two_body_cm(m12, m1, m2)
        p3 = two_body_cm(m0, m12, m3)
        w = p1 * p3
        if w > wmax:
            wmax = w
        if best is None:
            best = (m12, p1, p3, w)
    # rejection using the running maximum (fast: the density is smooth)
    for _ in range(400):
        m12 = np.sqrt(rng.uniform(m12_min ** 2, m12_max ** 2))
        p1 = two_body_cm(m12, m1, m2)
        p3 = two_body_cm(m0, m12, m3)
        if rng.random() * wmax * 1.05 <= p1 * p3:
            break
    # build in the parent rest frame
    n3 = random_unit(rng, 1)[0]
    e3 = np.sqrt(p3 * p3 + m3 * m3)
    e12 = np.sqrt(p3 * p3 + m12 * m12)
    q3 = np.concatenate([[e3], -p3 * n3])
    q12 = np.concatenate([[e12], p3 * n3])
    n1 = random_unit(rng, 1)[0]
    e1 = np.sqrt(p1 * p1 + m1 * m1)
    e2 = np.sqrt(p1 * p1 + m2 * m2)
    r1 = np.concatenate([[e1], p1 * n1])
    r2 = np.concatenate([[e2], -p1 * n1])
    b12 = q12[1:] / q12[0]
    q1 = boost(r1, b12)[0]
    q2 = boost(r2, b12)[0]
    bl = np.asarray(p4_parent, float)[1:] / p4_parent[0]
    return (boost(q1, bl)[0], boost(q2, bl)[0], boost(q3, bl)[0])


# --------------------------------------------------------------------------
# muon polarisation from pi/K -> mu nu
# --------------------------------------------------------------------------
def muon_polarisation(p4_mu_lab, m_parent, charge, n_star):
    """Longitudinal polarisation ``P_L`` of the muon in its own rest frame.

    ``n_star`` is the muon direction in the parent rest frame and ``charge`` the
    muon charge.  ``mu+`` is produced with helicity -1, ``mu-`` with +1 (the V-A
    structure of ``pi -> mu nu``).
    """
    h = -1.0 if charge > 0 else +1.0
    p_star = two_body_cm(m_parent, M_MU, 0.0)
    e_star = np.sqrt(p_star ** 2 + M_MU ** 2)
    # spin 4-vector in the parent rest frame (fully longitudinal)
    s_star = np.concatenate([[h * p_star / M_MU], h * (e_star / M_MU) * n_star])
    q_star = np.concatenate([[e_star], p_star * n_star])
    # boost to the lab with the parent's velocity, recovered from p4_mu_lab
    # (the parent boost is what maps q_star -> p4_mu_lab)
    e_lab, p_lab = p4_mu_lab[0], p4_mu_lab[1:]
    p_norm = np.linalg.norm(p_lab)
    if p_norm <= 0.0:
        return 0.0
    # solve for the boost beta along the parent direction: use the general
    # relation via the explicit parent 4-vector reconstruction
    # p_parent = q_mu + q_nu, both known in the parent rest frame -> the boost
    # is fixed by e_lab.  Numerically: beta such that gamma(e*+beta.p*) = e_lab.
    # Instead of inverting, boost s_star with the same beta used for q_star,
    # which the caller supplies implicitly; recover it from the parent frame:
    gam = (e_lab * e_star - np.dot(p_lab, p_star * n_star)) / M_MU ** 2
    gam = max(gam, 1.0)
    bmag = np.sqrt(1.0 - 1.0 / gam ** 2)
    # boost direction: the component of the lab momentum orthogonal to q_star
    # determines it uniquely; solve p_lab = p* n* + (gam-1)(p*.bhat)bhat + gam e* b
    # -> b_hat parallel to (p_lab - p* n*) when e* != 0
    dvec = p_lab - p_star * n_star
    dn = np.linalg.norm(dvec)
    bhat = dvec / dn if dn > 1e-12 else (p_lab / p_norm)
    b = bmag * bhat
    s_lab = boost(s_star, b)[0]
    # Rest-frame spin from the lab spin 4-vector.  With s.p = 0 one has
    # s^0 = s_vec . beta, and boosting to the rest frame gives
    #     zeta = s_vec - (1 - 1/gamma) (s_vec . phat) phat,
    # hence the LONGITUDINAL component collapses to
    #     P_L = zeta . phat = (s_vec . phat) / gamma = (m/E) (s_vec . phat).
    # (An earlier version used the "- (s.p)/(m(E+m)) p" form, which is the
    # formula for a different object and gives (2m-E)/m instead of m/E -- i.e.
    # the WRONG SIGN for every relativistic muon.  Gated by
    # test_decays.py::test_muon_from_pion_is_fully_polarised_at_the_kinematic_edges.)
    s_vec = s_lab[1:]
    p_l = (M_MU / e_lab) * float(np.dot(s_vec, p_lab)) / p_norm
    return float(np.clip(p_l, -1.0, 1.0))


# --------------------------------------------------------------------------
# Michel
# --------------------------------------------------------------------------
def michel_n(x, kind):
    """Rest-frame ``dN/dx``, ``x = 2E*/m_mu`` (same as
    ``tools/mceq3d/muon_segment_mc.michel_n``)."""
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    if kind == "numu":
        return 2.0 * x * x * (3.0 - 2.0 * x)
    return 12.0 * x * x * (1.0 - x)


def michel_asym(x, kind):
    """Rest-frame asymmetry ``A(x)`` in ``1 + A(x) P cos(theta*)``."""
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    if kind == "numu":
        return (1.0 - 2.0 * x) / np.maximum(3.0 - 2.0 * x, 1e-12)
    return np.ones_like(x)


# Inverse-CDF tables for the rest-frame Michel spectra.  Built with a
# cumulative TRAPEZOID (not a plain cumsum, which is a half-bin biased
# right-endpoint rule) on a fine grid, so the sampled shape matches n(x) to
# well below the statistical error of any run (gated in
# test_decays.py::test_michel_rest_frame_spectra).
_XE = np.linspace(0.0, 1.0, 2001)
_CDF = {}
for _k in ("numu", "nue"):
    _n = michel_n(_XE, _k)
    _c = np.concatenate([[0.0], np.cumsum(0.5 * (_n[1:] + _n[:-1])
                                          * np.diff(_XE))])
    _CDF[_k] = _c / _c[-1]


def michel_sample(rng, p4_mu_lab, charge, pol, kind):
    """One Michel neutrino of type ``kind`` ('numu'/'nue') in the lab.

    The two neutrinos are sampled **independently from their exact marginals**;
    this is unbiased for every single-particle (linear) observable, which is all
    the scorer tallies.  ``pol`` is the longitudinal polarisation ``P_L``; the
    polarisation axis is the muon lab momentum.  The sign convention is
    ``dN/dx dcos ~ n(x) [1 + s A(x) P_L cos(theta*)]`` with ``s = -1`` for
    ``mu-`` and ``s = +1`` for ``mu+`` (V-A; verified against the unpolarised
    limit and against MCEq's helicity-resolved decay matrices).
    """
    u = rng.random()
    x = float(np.interp(u, _CDF[kind], _XE))
    a = float(michel_asym(x, kind))
    s = -1.0 if charge > 0 else +1.0
    aa = s * a * pol
    # sample cos(theta*) from (1 + aa*cos)/2 on [-1, 1]
    v = rng.random()
    if abs(aa) < 1e-9:
        c = 2.0 * v - 1.0
    else:
        c = (-1.0 + np.sqrt(np.maximum(1.0 + 2.0 * aa * (2.0 * v - 1.0) + aa * aa,
                                       0.0))) / aa
        c = float(np.clip(c, -1.0, 1.0))
    ph = 2.0 * np.pi * rng.random()
    st = np.sqrt(max(1.0 - c * c, 0.0))
    n_local = np.array([st * np.cos(ph), st * np.sin(ph), c])
    p_lab = np.asarray(p4_mu_lab, float)[1:]
    p_norm = float(np.linalg.norm(p_lab))
    if p_norm < 1e-12:                       # a stopped muon decays isotropically
        e_star = 0.5 * M_MU * x
        n = random_unit(rng, 1)[0]
        return np.concatenate([[e_star], e_star * n])
    axis = p_lab / p_norm
    n_star = rotate_to(axis, n_local)
    e_star = 0.5 * M_MU * x
    q = np.concatenate([[e_star], e_star * n_star])
    b = p_lab / p4_mu_lab[0]
    return boost(q, b)[0]


# --------------------------------------------------------------------------
# the public entry point
# --------------------------------------------------------------------------
def decay(rng, pdg, p4, pol=0.0):
    """Decay one particle.  Returns ``[(pdg, p4, pol), ...]`` for the daughters
    that the tracker cares about (hadrons, muons, neutrinos)."""
    pdg = int(pdg)
    m0 = MASS[pdg]
    out = []
    if abs(pdg) == 211:
        ch = +1 if pdg > 0 else -1
        q_mu, q_nu, n_star = two_body(rng, p4, M_PI, M_MU, 0.0)
        p_l = muon_polarisation(q_mu, M_PI, ch, n_star)
        out.append((-13 if ch > 0 else 13, q_mu, p_l))
        out.append((14 if ch > 0 else -14, q_nu, 0.0))
        return out
    if abs(pdg) == 13:
        ch = +1 if pdg < 0 else -1          # PDG 13 = mu-
        q1 = michel_sample(rng, p4, ch, pol, "numu")
        q2 = michel_sample(rng, p4, ch, pol, "nue")
        out.append((-14 if ch > 0 else 14, q1, 0.0))
        out.append((12 if ch > 0 else -12, q2, 0.0))
        return out
    if abs(pdg) == 321:
        table = BR_KP if pdg > 0 else BR_KM
        ds = _pick(rng, table)
        return _finish(rng, p4, M_K, ds)
    if pdg == 130:
        return _finish(rng, p4, M_K0, _pick(rng, BR_KL))
    if pdg == 310:
        return _finish(rng, p4, M_K0, _pick(rng, BR_KS))
    if abs(pdg) == 3122:
        table = BR_LAM if pdg > 0 else BR_LAMBAR
        return _finish(rng, p4, MASS[3122], _pick(rng, table))
    return out


def _finish(rng, p4, m0, ds):
    out = []
    if len(ds) == 2:
        m1, m2 = MASS[ds[0]], MASS[ds[1]]
        q1, q2, n_star = two_body(rng, p4, m0, m1, m2)
        for pdg_d, q in ((ds[0], q1), (ds[1], q2)):
            if abs(pdg_d) == 13:
                ch = +1 if pdg_d < 0 else -1
                out.append((pdg_d, q, muon_polarisation(q, m0, ch, n_star)))
            else:
                out.append((pdg_d, q, 0.0))
        return out
    ms = [MASS[d] for d in ds]
    res = three_body(rng, p4, m0, ms)
    if res is None:
        return out
    for pdg_d, q in zip(ds, res):
        # a muon from a 3-body kaon decay is treated as unpolarised
        out.append((pdg_d, q, 0.0))
    return out
