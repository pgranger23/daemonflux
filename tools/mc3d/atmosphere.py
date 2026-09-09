"""CORSIKA US-standard (Keilhauer ``BK_USStd``) atmosphere, spherical geometry.

Why a re-implementation and not ``offaxis_mc``'s machinery
---------------------------------------------------------
``offaxis_mc`` gets ``rho(h)`` by tabulating MCEq's ``density_model`` and
``X_slant(h, psi)`` by a 90x140 bilinear table built once for the *deterministic*
cone integral.  A Monte-Carlo needs the *inverse* operation -- "advance this
particle until it has traversed ``dX`` g/cm2" -- millions of times, at arbitrary
positions and directions, and needs it to be exactly invertible so that the
interaction/decay competition is unbiased.  A table with bilinear interpolation
is neither invertible nor accurate enough near the limb (it is capped at 1e7 to
keep the interpolation finite).  So this module carries the **same** analytic
CORSIKA 5-layer parameterisation MCEq itself uses (``MCEq.geometry.
atmosphere_parameters['BK_USStd']``, verified bit-for-bit by
``test_geometry.py::test_density_matches_mceq``) and adds an adaptive
grammage integrator/inverter along an arbitrary straight ray.

Layers (h in cm, ``hlay`` boundaries):
    i = 0..3:  X_v(h) = a_i + b_i exp(-h/c_i),   rho(h) = (b_i/c_i) exp(-h/c_i)
    i = 4:     X_v(h) = a_4 - h/c_4,             rho(h) = 1/c_4
    h > h_top = a_4 c_4 = 112.8 km:  rho = 0.
"""

from __future__ import annotations

import math

import numpy as np

from constants import R_EARTH_CM

# BK_USStd, identical to MCEq.geometry.atmosphere_parameters
_AATM = np.array([-149.801663, -57.932486, 0.63631894, 4.3545369e-4, 0.01128292])
_BATM = np.array([1183.6071, 1143.0425, 1322.9748, 655.69307, 1.0])
_CATM = np.array([954248.34, 800005.34, 629568.93, 737521.77, 1.0e9])
_HLAY = np.array([0.0, 7.0e5, 1.14e6, 3.7e6, 1.0e7])

H_TOP_CM = _AATM[4] * _CATM[4]          # 1.128292e7 cm = 112.8 km
X_GROUND = _AATM[0] + _BATM[0]          # 1033.8049 g/cm2 at h = 0


# --- scalar fast paths -----------------------------------------------------
# The transport inner loop calls these millions of times on *scalars*; the
# numpy versions below cost ~2 us each through asarray/searchsorted/where,
# which dominates the whole Monte-Carlo.  These are the same formulas in plain
# Python floats and are gated against the array versions in test_geometry.py.
_A = tuple(_AATM)
_B = tuple(_BATM)
_C = tuple(_CATM)
_HL = tuple(_HLAY)


def rho_s(h):
    """Scalar air density [g/cm3]."""
    if h < 0.0 or h > 1.128292e7:
        return 0.0
    if h < 7.0e5:
        return _B[0] / _C[0] * math.exp(-h / _C[0])
    if h < 1.14e6:
        return _B[1] / _C[1] * math.exp(-h / _C[1])
    if h < 3.7e6:
        return _B[2] / _C[2] * math.exp(-h / _C[2])
    if h < 1.0e7:
        return _B[3] / _C[3] * math.exp(-h / _C[3])
    return 1.0 / _C[4]


def hscale_s(h):
    """Scalar local density scale height [cm]."""
    if h < 7.0e5:
        return _C[0]
    if h < 1.14e6:
        return _C[1]
    if h < 3.7e6:
        return _C[2]
    return _C[3]


def _layer(h_cm):
    return np.clip(np.searchsorted(_HLAY, h_cm, side="right") - 1, 0, 4)


def density(h_cm):
    """Air density [g/cm3] at altitude ``h_cm`` (0 above the top / below 0)."""
    h = np.asarray(h_cm, dtype=float)
    i = _layer(h)
    rho = np.where(
        i < 4,
        _BATM[np.minimum(i, 3)] / _CATM[np.minimum(i, 3)]
        * np.exp(-h / _CATM[np.minimum(i, 3)]),
        1.0 / _CATM[4],
    )
    rho = np.where((h < 0.0) | (h > H_TOP_CM), 0.0, rho)
    return rho if rho.ndim else float(rho)


def vertical_depth(h_cm):
    """Vertical mass overburden X_v(h) [g/cm2] above altitude ``h_cm``."""
    h = np.asarray(h_cm, dtype=float)
    i = _layer(np.clip(h, 0.0, None))
    j = np.minimum(i, 3)
    x = np.where(i < 4, _AATM[j] + _BATM[j] * np.exp(-h / _CATM[j]),
                 _AATM[4] - h / _CATM[4])
    x = np.where(h > H_TOP_CM, 0.0, x)
    x = np.where(h < 0.0, X_GROUND + (-h) * density(0.0), x)
    return x if x.ndim else float(x)


def height_of_depth(x_v):
    """Invert :func:`vertical_depth` (used for 1D cross-checks)."""
    x = np.asarray(x_v, dtype=float)
    thickl = np.array([X_GROUND, vertical_depth(_HLAY[1]), vertical_depth(_HLAY[2]),
                       vertical_depth(_HLAY[3]), vertical_depth(_HLAY[4])])
    out = np.empty_like(x)
    for k in range(4):
        m = (x <= thickl[k]) & (x > thickl[k + 1])
        out[m] = _CATM[k] * np.log(_BATM[k] / (x[m] - _AATM[k]))
    m = x <= thickl[4]
    out[m] = (_AATM[4] - x[m]) * _CATM[4]
    m = x > thickl[0]
    out[m] = 0.0
    return out if out.ndim else float(out)


# ---------------------------------------------------------------------------
# Ray geometry in a spherical atmosphere
# ---------------------------------------------------------------------------
def altitude(r_vec):
    return np.linalg.norm(r_vec) - R_EARTH_CM


def ray_sphere(r0, u, radius):
    """Path lengths at which the ray ``r0 + s u`` crosses ``|r| = radius``.

    Returns ``(s_minus, s_plus)`` or ``(nan, nan)`` if it misses.
    """
    b = r0[0] * u[0] + r0[1] * u[1] + r0[2] * u[2]
    c = r0[0] * r0[0] + r0[1] * r0[1] + r0[2] * r0[2] - radius * radius
    disc = b * b - c
    if disc < 0.0:
        return (float("nan"), float("nan"))
    sq = math.sqrt(disc)
    return (-b - sq, -b + sq)


def _h_at(r0, u, s, r0n2, r0u):
    return np.sqrt(r0n2 + 2.0 * s * r0u + s * s) - R_EARTH_CM


# Adaptive-step control: fractional change of the density scale height per step.
_DS_MAX = 30.0e5          # 30 km hard cap
_DS_MIN = 1.0e2           # 1 m floor
_H_SCALE = 6.4e5          # nominal density scale height [cm]


def scale_height(h):
    """Local density scale height [cm] (the CORSIKA layer's own ``c_i``)."""
    return hscale_s(float(h))


def _step_len(h, dhds, s_left):
    """Substep length for the Simpson integrator: the altitude may change by
    ~0.3 scale heights per step (Simpson over an exponential is then good to
    ~1e-5, verified in ``test_geometry.py::test_vertical_grammage``)."""
    hs = hscale_s(h)
    ds = _DS_MAX if abs(dhds) < 1e-6 else 0.25 * hs / abs(dhds)
    if ds < _DS_MIN:
        ds = _DS_MIN
    elif ds > _DS_MAX:
        ds = _DS_MAX
    return ds if ds < s_left else s_left


def grammage(r0, u, s_end, r0n2=None, r0u=None):
    """Slant grammage [g/cm2] along ``r0 + s u`` for ``s in [0, s_end]``.

    ``r0n2 = |r0|^2`` and ``r0u = r0.u`` may be passed in when the caller
    already has them (the transport loop does).
    """
    if r0n2 is None:
        r0n2 = float(r0[0] * r0[0] + r0[1] * r0[1] + r0[2] * r0[2])
        r0u = float(r0[0] * u[0] + r0[1] * u[1] + r0[2] * u[2])
    s = 0.0
    X = 0.0
    while s < s_end - 1e-6:
        r = math.sqrt(r0n2 + 2 * s * r0u + s * s)
        h = r - R_EARTH_CM
        ds = _step_len(h, (r0u + s) / r, s_end - s)
        sm = s + 0.5 * ds
        se = s + ds
        h1 = math.sqrt(r0n2 + 2 * sm * r0u + sm * sm) - R_EARTH_CM
        h2 = math.sqrt(r0n2 + 2 * se * r0u + se * se) - R_EARTH_CM
        X += ds / 6.0 * (rho_s(h) + 4.0 * rho_s(h1) + rho_s(h2))
        s = se
    return X


def advance_grammage(r0, u, dX, s_max, r0n2=None, r0u=None, tol=1e-4):
    """Advance along ``r0 + s u`` until grammage ``dX`` is accumulated.

    Returns ``(s, X_done, hit)``: ``hit`` is True when ``dX`` was reached inside
    ``s_max``, otherwise ``s = s_max`` and ``X_done < dX`` (the particle left the
    atmosphere / reached the boundary first).
    """
    if r0n2 is None:
        r0n2 = float(r0[0] * r0[0] + r0[1] * r0[1] + r0[2] * r0[2])
        r0u = float(r0[0] * u[0] + r0[1] * u[1] + r0[2] * u[2])
    s = 0.0
    X = 0.0
    while s < s_max - 1e-6:
        r = math.sqrt(r0n2 + 2 * s * r0u + s * s)
        h = r - R_EARTH_CM
        ds = _step_len(h, (r0u + s) / r, s_max - s)
        sm = s + 0.5 * ds
        se = s + ds
        rho0 = rho_s(h)
        h1 = math.sqrt(r0n2 + 2 * sm * r0u + sm * sm) - R_EARTH_CM
        h2 = math.sqrt(r0n2 + 2 * se * r0u + se * se) - R_EARTH_CM
        dXs = ds / 6.0 * (rho0 + 4.0 * rho_s(h1) + rho_s(h2))
        if X + dXs >= dX:
            lo, hi = 0.0, ds
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                a = s + 0.5 * mid
                bpt = s + mid
                hm = math.sqrt(r0n2 + 2 * a * r0u + a * a) - R_EARTH_CM
                he = math.sqrt(r0n2 + 2 * bpt * r0u + bpt * bpt) - R_EARTH_CM
                xm = X + mid / 6.0 * (rho0 + 4 * rho_s(hm) + rho_s(he))
                if xm < dX:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < tol * (ds if ds > 1.0 else 1.0):
                    break
            return s + 0.5 * (lo + hi), dX, True
        X += dXs
        s = se
    return s_max, X, False


_EPS_CM = 1.0


def path_to_exit(r0, u, r_top=None):
    """Path length from ``r0`` to the ground or to the top of the atmosphere.

    Returns ``(s, hit_ground)``.  Only the **near** root of the ground sphere
    counts: taking the far root when the near one is <= 0 sends the particle
    straight through the Earth and out the other side (a real bug found in the
    milestone-1 closure -- every muon then "decayed" after thousands of steps
    inside the Earth, so the muon-decay neutrino yield came out ~2x too high).
    """
    r_top = (R_EARTH_CM + H_TOP_CM) if r_top is None else r_top
    rn2 = r0[0] * r0[0] + r0[1] * r0[1] + r0[2] * r0[2]
    ru = r0[0] * u[0] + r0[1] * u[1] + r0[2] * u[2]
    if rn2 <= (R_EARTH_CM + _EPS_CM) ** 2 and ru <= 0.0:
        return 0.0, True
    s0, _ = ray_sphere(r0, u, R_EARTH_CM)
    if s0 == s0 and s0 > _EPS_CM:      # not NaN
        return s0, True
    _, s1 = ray_sphere(r0, u, r_top)
    if s1 != s1:
        return 0.0, False
    return (s1 if s1 > 0.0 else 0.0), False
