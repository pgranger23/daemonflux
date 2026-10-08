"""Geomagnetic cutoff acceptance at each primary's own injection point.

Milestone 3a of ``PHASE2_PLAN.md`` (sec. 3.4).  Honda's test (astro-ph/0404457
sec. IV): a primary sampled on the injection sphere is a cosmic ray if its
time-reversed trajectory "reaches the escape sphere without touching the
injection sphere again".  This module applies that test, batched, to arbitrary
injection states ``(position, inward velocity, rigidity, charge)`` and returns a
binary verdict ``allowed`` -- the per-shower weight ``w_cutoff in {0, 1}``.

What is reused, what is new
---------------------------
The integrator is ``tools/mceq3d/geomag_backtrace.backtrace_vec`` (fixed-step
RK4 in arc length, IGRF below ``r_switch = 4 R_E``, tilted dipole above,
compaction of finished trajectories), called unchanged.  Its conventions were
verified in Phase 1 and are kept: geocentric Cartesian, ``R_E = 6371 km``, the
launch velocity is the **reversed** physical velocity and the antiparticle
reversal is done inside ``backtrace_vec`` through ``k = -charge * c / R``.  So a
primary of charge sign ``q`` moving along ``u`` is tested as
``backtrace_vec(r, -u, R, ..., charge=q)``.

New here:

* **The re-entry rule** (``floor``).  Default ``"injection"``: the back-trace is
  forbidden as soon as it falls below the launch (injection) sphere -- Honda's
  convention, and the only one consistent with sampling the injection sphere:
  every physical primary is then represented exactly once, at its *first*
  inward crossing (see ``MILESTONE3A_NOTES.md`` sec. 2 for the argument and the
  measured difference to ``geomag_backtrace``'s sea-level floor).
* **The field accelerator** (``igrf_field``).  97-98% of a back-trace is spent
  in ``ppigrf.igrf_gc``, which re-parses the coefficient file with pandas on
  every call (14.6 ms) and then costs ~30 us per point.  :class:`FastIGRF`
  evaluates the *same* spherical-harmonic sum from coefficients cached once per
  (model, date) -- equal to ``ppigrf`` to ~1e-15 relative (``test_cutoff.py``)
  -- and is swapped in for the duration of a trace by a context manager.  It
  also makes the model explicit: ``ppigrf`` 2.1.0 *defaults to IGRF-14*,
  whatever ``geomag_backtrace``'s docstrings say; here the default is IGRF-13
  (``DEFAULT_MODEL``) and either can be chosen.
* **A rigidity-dependent step and a path budget** (``step_m``, ``MAX_S_M``).
  ``backtrace_vec``'s default ``ds = 100 km`` exceeds the gyroradius of a
  1.7 GV proton near the surface (~95 km at 60 uT), and the injection patch
  reaches geomagnetic latitudes where such primaries are allowed: 25 km below
  5 GV, 100 km at 5-10 GV, 200 km above; budget 2e8 m (measured convergence:
  ``MILESTONE3A_NOTES.md`` secs. 3-4).
* **An exact pre-filter** (``gate``, ``trace_gated``): far below the Stoermer
  cutoff of the most favourable direction, or above 60 GV, the verdict is known
  without a trace (zero errors on 15 000 test states).  It matters because the
  expensive traces are the *forbidden* ones: deep below its cutoff a back-trace
  is quasi-trapped and bounces until the path budget.
* **Azimuthal copies** (``copy_states``): rotations about the site vertical,
  each needing its own trace because its direction relative to B changes.
* **Acceleration option** (``RUCache``): a cached ``R_U`` grid used as a gate,
  and **the 1D reference's cutoff** (``SiteCutoffMap``).

Units: positions in **cm** (mc3d convention; converted to m for the
back-tracer), rigidities in GV, dates as ``datetime``.  Studies:
``python cutoff.py <study> --outdir DIR`` (see ``main``).
"""

from __future__ import annotations

import contextlib
import datetime
import os
import time

import numpy as np

import geomag_backtrace as gb
from constants import H_INJ_KM, M_P, R_EARTH_CM

DEFAULT_DATE = datetime.datetime(2020, 1, 1)
DEFAULT_MODEL = "IGRF13"
#: escape sphere [R_E = 6371 km] -- Honda's 10 R_e (he uses R_e = 6378.18 km)
R_ESCAPE_RE = 10.0
#: path budget [m] (31 R_E); a trace still bound after it is trapped -> forbidden.
#: Measured (MILESTONE3A_NOTES.md sec. 4): against 8e8 m it changes 1 of 3000
#: cap-pointing verdicts; the bouncing tail a batch waits for scales with it.
MAX_S_M = 2.0e8
#: RK4 step rule: ds = STEP_TABLE[k][1] for the first k with R < STEP_TABLE[k][0]
#: -- 25 km below 5 GV, 100 km at 5-10 GV, 200 km above (MILESTONE3A_NOTES.md
#: sec. 3: the same verdicts as a 6.25 / 12.5 km reference in 5999 of 6000
#: test states, the one difference a chaotic penumbral trajectory)
STEP_TABLE = ((5.0, 2.5e4), (10.0, 1.0e5), (np.inf, 2.0e5))
DS_MAX_M = 2.0e5
R_INJ_CM = R_EARTH_CM + H_INJ_KM * 1.0e5
SITES = {"kamioka": (36.4267, 137.31)}
#: exact pre-filter (``gate``): reject without a trace below
#: GATE_LO x the Stoermer cutoff of the most favourable direction at the
#: position (allowed states measured down to 0.98 x it within 15 deg of the
#: site and 0.77 x it anywhere on the patch, sec. 5.4 of the notes);
#: accept without a trace above R_SKIP_GV if the primary reaches the dense
#: atmosphere (mu > GATE_MU_MIN; no forbidden cap-pointing state above 26.6 GV
#: in 3000 between 20 and 2000 GV)
GATE_LO = 0.70
R_SKIP_GV = 60.0
GATE_MU_MIN = 0.17
#: mceq3d's delivered cutoff-map nodes (mceq3d_flux._zenith_nodes(13, True))
MAP_ZEN = np.concatenate([np.linspace(0.0, 80.0, 13),
                          [82.0, 84.0, 85.5, 87.0, 88.0, 88.5, 89.0, 89.5]])
MAP_AZ = np.arange(0.0, 360.0, 15.0)


# ---------------------------------------------------------------------------
# Field: a cached-coefficient evaluator of the ppigrf spherical-harmonic sum
# ---------------------------------------------------------------------------
_SHC_FILE = {"IGRF13": "IGRF13.shc", "IGRF14": "IGRF14.shc"}


def _as_datetime(date):
    if isinstance(date, datetime.datetime):
        return date
    if isinstance(date, datetime.date):
        return datetime.datetime(date.year, date.month, date.day)
    return datetime.datetime.fromisoformat(str(date))


def _shc_path(model):
    from ppigrf import ppigrf as _pp
    try:
        return os.path.join(os.path.dirname(_pp.__file__), _SHC_FILE[model.upper()])
    except KeyError:
        raise ValueError(f"unknown field model {model!r}; use one of {list(_SHC_FILE)}")


class FastIGRF:
    """``ppigrf.igrf_gc`` with the coefficients parsed and time-interpolated once.

    Same inputs and outputs as ``ppigrf.igrf_gc(r_km, colat_deg, lon_deg, date)``
    -- ``(Br, Btheta, Bphi)`` in nT, each of shape ``(1, N)`` -- and the same
    numerics: the time interpolation is literally ppigrf's pandas code (run
    once per date), the Legendre functions use ppigrf's recursion and Schmidt
    factors, the sums are the same terms.  Only the bookkeeping differs: the
    coefficient file is read once, the Schmidt factors are folded into the
    coefficients, ``a^(n+2)`` is a running product, and each degree's ``m``
    terms are reduced in one array operation.  Measured agreement with
    ``ppigrf``: <= 2e-15 relative (``test_cutoff.py``).

    Three internal evaluators -- scalar Python (<= ``SCALAR_MAX`` points: the
    bouncing tails of a batch), packed arrays (<= ``PACKED_MAX``), per-degree
    loop (larger) -- perform the same IEEE operations in the same order and are
    **bitwise equal**, so a point's field never depends on the batch size.
    """

    def __init__(self, model=DEFAULT_MODEL):
        from ppigrf import ppigrf as _pp
        self._pp = _pp
        self.model = model.upper()
        self.coeff_fn = _shc_path(self.model)
        self._g, self._h = _pp.read_shc(self.coeff_fn)
        keys = [tuple(int(x) for x in k) for k in self._g.columns]
        self.nmax = max(n for n, _ in keys)
        nm = self.nmax
        expect = [(n, m) for n in range(1, nm + 1) for m in range(n + 1)]
        if keys != expect:  # the packed layout below relies on this order
            raise RuntimeError("unexpected coefficient order in " + self.coeff_fn)
        self.off = np.zeros(nm + 2, int)
        for n in range(1, nm + 1):
            self.off[n + 1] = self.off[n] + n + 1
        # Schmidt factors S[n, m] and recursion constants K[n, m] -- ppigrf.get_legendre
        S = np.zeros((nm + 1, nm + 1))
        S[0, 0] = 1.0
        K = np.zeros((nm + 1, nm + 1))
        for n in range(1, nm + 1):
            for m in range(0, n + 1):
                if m == 0:
                    S[n, 0] = S[n - 1, 0] * (2.0 * n - 1) / n
                else:
                    S[n, m] = S[n, m - 1] * np.sqrt((n - m + 1) * (int(m == 1) + 1.0)
                                                    / (n + m))
                if n > 1:
                    K[n, m] = ((n - 1) ** 2 - m ** 2) / ((2 * n - 1) * (2 * n - 3))
        self._S = np.concatenate([S[n, : n + 1] for n in range(1, nm + 1)])
        self._K = K
        self._mcol = [np.arange(n + 1, dtype=float)[:, None] for n in range(nm + 1)]
        # packed-layout helpers (small batches): order m of every (n, m) term,
        # degree exponents, segment starts of each degree
        self._mk = np.concatenate([np.arange(n + 1) for n in range(1, nm + 1)])
        self._mkf = self._mk.astype(float)[:, None]
        self._mrow = np.arange(nm + 1, dtype=float)[:, None]
        self._np2 = (np.arange(1, nm + 1) + 2.0)[:, None]
        self._n1 = (np.arange(1, nm + 1) + 1.0)[:, None]
        self._Kl = K.tolist()
        self._coef = {}
        self._coefl = {}

    def coefficients(self, date):
        """Schmidt-folded packed ``(g, h)`` at ``date`` (ppigrf's interpolation)."""
        d = _as_datetime(date)
        c = self._coef.get(d)
        if c is None:
            dd = np.array([d])
            index = self._g.index.union(dd)
            g = self._g.reindex(index).groupby(index).first()
            h = self._h.reindex(index).groupby(index).first()
            g = g.interpolate(method="time").loc[dd, :].values[0]
            h = h.interpolate(method="time").loc[dd, :].values[0]
            c = (np.asarray(g, float) * self._S, np.asarray(h, float) * self._S)
            self._coef[d] = c
        return c

    #: up to SCALAR_MAX points: pure-Python scalar evaluation (the long tails of
    #: bouncing trajectories run at one or two points, where numpy's per-call
    #: overhead dominates); up to PACKED_MAX: the packed evaluator; above: the
    #: per-degree loop (better cache behaviour).  All three are bitwise equal.
    SCALAR_MAX = 4
    PACKED_MAX = 1024

    def __call__(self, r, theta, phi, date, coeff_fn=None, min_degree=1,
                 max_degree=13):
        if coeff_fn is not None and os.path.basename(coeff_fn) != \
                os.path.basename(self.coeff_fn):
            raise ValueError(f"FastIGRF({self.model}) called with {coeff_fn}")
        gs, hs = self.coefficients(date)
        r, theta, phi = np.broadcast_arrays(np.asarray(r, float),
                                            np.asarray(theta, float),
                                            np.asarray(phi, float))
        shape = r.shape
        r, theta, phi = r.ravel(), theta.ravel(), phi.ravel()
        if r.size <= self.SCALAR_MAX:
            Br, Bt, Bp = self._scalar(date, r, theta, phi, min_degree, max_degree)
        elif r.size <= self.PACKED_MAX:
            Br, Bt, Bp = self._packed(gs, hs, r, theta, phi, min_degree, max_degree)
        else:
            Br, Bt, Bp = self._loop(gs, hs, r, theta, phi, min_degree, max_degree)
        out = (1,) + shape
        return Br.reshape(out), Bt.reshape(out), Bp.reshape(out)

    def _legendre(self, st, ct):
        """Unnormalised P_nm, dP_nm/dtheta (ppigrf's recursion), packed rows."""
        nm, off, Kc = self.nmax, self.off, self._K
        n_pt = st.size
        P = np.empty((off[nm + 1], n_pt))
        dP = np.empty((off[nm + 1], n_pt))
        P[0], dP[0] = ct, -st
        P[1], dP[1] = st, ct
        for n in range(2, nm + 1):
            o, o1 = off[n], off[n - 1]
            if n == 2:
                p2, dp2 = 1.0, 0.0          # degree 0: P00 = 1, dP00 = 0
            else:
                o2 = off[n - 2]
                p2, dp2 = P[o2:o2 + n - 1], dP[o2:o2 + n - 1]
            k = Kc[n, : n - 1, None]
            Pp, dPp = P[o1:o1 + n], dP[o1:o1 + n]
            P[o:o + n] = ct * Pp                # m = n-1 has K[n, n-1] = 0
            P[o:o + n - 1] -= k * p2
            dP[o:o + n] = ct * dPp - st * Pp
            dP[o:o + n - 1] -= k * dp2
            P[o + n] = st * Pp[n - 1]
            dP[o + n] = st * dPp[n - 1] + ct * Pp[n - 1]
        return P, dP

    def _scalar(self, date, r, theta, phi, min_degree, max_degree):
        """Pure-Python evaluation of a few points: the same IEEE operations in
        the same order as :meth:`_packed`, hence bitwise equal to it wherever
        numpy's float64 ``sin``/``cos`` are libm's (true on this AVX2 build;
        pinned by ``test_cutoff.py``)."""
        import math
        d = _as_datetime(date)
        cl = self._coefl.get(d)
        if cl is None:
            gs, hs = self.coefficients(d)
            cl = self._coefl[d] = (gs.tolist(), hs.tolist())
        gl, hl = cl
        nm, off, K = self.nmax, self.off.tolist(), self._Kl
        RE = self._pp.RE
        win = [1.0 if min_degree <= n <= max_degree else 0.0 for n in range(1, nm + 1)]
        npt = r.size
        Br, Bt, Bp = np.empty(npt), np.empty(npt), np.empty(npt)
        for i, (rr, tt, pp) in enumerate(zip(r.tolist(), theta.tolist(), phi.tolist())):
            th = math.radians(tt)
            st, ct = math.sin(th), math.cos(th)
            phr = math.radians(pp)
            cm = [math.cos(float(m) * phr) for m in range(nm + 1)]
            sm = [math.sin(float(m) * phr) for m in range(nm + 1)]
            a = RE / rr
            c = a * a
            apw = []
            for _ in range(nm):
                c = c * a
                apw.append(c)
            P = [0.0] * off[nm + 1]
            dP = [0.0] * off[nm + 1]
            P[0], dP[0], P[1], dP[1] = ct, -st, st, ct
            for n in range(2, nm + 1):
                o, o1 = off[n], off[n - 1]
                Kn = K[n]
                for m in range(n):
                    P[o + m] = ct * P[o1 + m]
                    dP[o + m] = ct * dP[o1 + m] - st * P[o1 + m]
                if n == 2:
                    P[o] = P[o] - Kn[0] * 1.0
                    dP[o] = dP[o] - Kn[0] * 0.0
                else:
                    o2 = off[n - 2]
                    for m in range(n - 1):
                        P[o + m] = P[o + m] - Kn[m] * P[o2 + m]
                        dP[o + m] = dP[o + m] - Kn[m] * dP[o2 + m]
                P[o + n] = st * P[o1 + n - 1]
                dP[o + n] = st * dP[o1 + n - 1] + ct * P[o1 + n - 1]
            br = bt = bp = None
            for n in range(1, nm + 1):
                o = off[n]
                sr = sth = sp = None
                for m in range(n + 1):
                    k = o + m
                    C, S, g, h = cm[m], sm[m], gl[k], hl[k]
                    tc = g * C + h * S
                    ts = h * C - g * S
                    x, y, z = P[k] * tc, dP[k] * tc, (float(m) * P[k]) * ts
                    if m == 0:
                        sr, sth, sp = x, y, z
                    else:
                        sr, sth, sp = sr + x, sth + y, sp + z
                an = apw[n - 1] * win[n - 1] if win[n - 1] != 1.0 else apw[n - 1]
                tr, tt_, tp = (float(n + 1) * an) * sr, an * sth, an * sp
                if n == 1:
                    br, bt, bp = tr, tt_, tp
                else:
                    br, bt, bp = br + tr, bt + tt_, bp + tp
            Br[i], Bt[i], Bp[i] = br, -bt, -bp / st
        return Br, Bt, Bp

    def _common(self, r, theta, phi):
        """Shared per-point inputs; both evaluators use exactly these, so they
        agree bitwise (the batch size never changes a field value)."""
        nm = self.nmax
        th = np.radians(theta)
        st, ct = np.sin(th), np.cos(th)
        mph = self._mrow * np.radians(phi)
        cm, sm = np.cos(mph), np.sin(mph)
        a = self._pp.RE / r
        # a^(n+2), n = 1 .. nmax, as the sequential product a*a*a*...
        apw = np.cumprod(np.broadcast_to(a, (nm + 2, r.size)), axis=0)[2:]
        P, dP = self._legendre(st, ct)
        return st, cm, sm, apw, P, dP

    def _packed(self, gs, hs, r, theta, phi, min_degree, max_degree):
        nm = self.nmax
        st, cm, sm, apw, P, dP = self._common(r, theta, phi)
        C, S = cm[self._mk], sm[self._mk]
        g, h = gs[:, None], hs[:, None]
        tc = g * C + h * S
        ts = h * C - g * S
        X, Y, Z = P * tc, dP * tc, self._mkf * P * ts
        sr = np.empty((nm, r.size))
        sth = np.empty((nm, r.size))
        sp = np.empty((nm, r.size))
        for n in range(1, nm + 1):          # sequential sums (np.add.reduceat
            o = self.off[n]                 # is not), as in _loop
            sr[n - 1] = X[o:o + n + 1].sum(axis=0)
            sth[n - 1] = Y[o:o + n + 1].sum(axis=0)
            sp[n - 1] = Z[o:o + n + 1].sum(axis=0)
        if min_degree > 1 or max_degree < nm:
            deg = np.arange(1, nm + 1)
            apw = apw * ((deg >= min_degree) & (deg <= max_degree))[:, None]
        Br = (self._n1 * apw * sr).sum(axis=0)
        Bt = -(apw * sth).sum(axis=0)
        Bp = -(apw * sp).sum(axis=0) / st
        return Br, Bt, Bp

    def _loop(self, gs, hs, r, theta, phi, min_degree, max_degree):
        nm, off = self.nmax, self.off
        st, cm, sm, apw, P, dP = self._common(r, theta, phi)
        Br = np.zeros(r.size)
        Bt = np.zeros(r.size)
        Bp = np.zeros(r.size)
        for n in range(1, nm + 1):
            an = apw[n - 1] if min_degree <= n <= max_degree else 0.0 * apw[n - 1]
            o = off[n]
            g = gs[o:o + n + 1, None]
            h = hs[o:o + n + 1, None]
            cmn, smn = cm[: n + 1], sm[: n + 1]
            tc = g * cmn + h * smn
            ts = h * cmn - g * smn
            Pn = P[o:o + n + 1]
            Br += (self._n1[n - 1] * an) * (Pn * tc).sum(axis=0)
            Bt -= an * (dP[o:o + n + 1] * tc).sum(axis=0)
            Bp -= an * (self._mcol[n] * Pn * ts).sum(axis=0)
        Bp /= st
        return Br, Bt, Bp


_FAST = {}


def fast_igrf(model=DEFAULT_MODEL):
    """Process-wide :class:`FastIGRF` instance for ``model`` (built once)."""
    key = model.upper()
    f = _FAST.get(key)
    if f is None:
        f = _FAST[key] = FastIGRF(key)
    return f


@contextlib.contextmanager
def igrf_field(model=DEFAULT_MODEL, fast=True):
    """Route ``geomag_backtrace``'s IGRF calls to ``model`` for the duration.

    ``geomag_backtrace._bfield_igrf_cart`` calls ``ppigrf.igrf_gc`` without a
    coefficient file, i.e. with ppigrf's *default* model (IGRF-14 for ppigrf
    2.1.0).  Inside this context the call goes to :class:`FastIGRF` for
    ``model`` (``fast=True``) or to the stock ``ppigrf.igrf_gc`` pinned to
    ``model``'s coefficient file (``fast=False``, the reference path).  The
    original function is restored on exit; nesting is safe.  Forked worker
    processes created inside the context inherit it.
    """
    import functools

    import ppigrf
    orig = ppigrf.igrf_gc
    if fast:
        repl = fast_igrf(model)
    else:
        base = getattr(orig, "_mc3d_orig", orig)
        repl = functools.partial(base, coeff_fn=_shc_path(model))
        repl._mc3d_orig = base
    ppigrf.igrf_gc = repl
    try:
        yield
    finally:
        ppigrf.igrf_gc = orig


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------
def rigidity_gv(e_kin_nucleon, a_mass=1, z=1, m_nucleon=M_P):
    """Rigidity [GV] of a nucleus (A, Z) with kinetic energy ``e_kin_nucleon``
    [GeV] per nucleon: ``R = (A/|Z|) p_nucleon``, ``p_nucleon = sqrt(T (T + 2m))``.

    Superposition (Honda, astro-ph/0404457 sec. IV): the A nucleons of a
    nucleus are "treated as protons with double rigidity ... in the rigidity
    cutoff test" -- the test uses the *nucleus* rigidity, while the cascade
    gets A free nucleons at ``E/A``.  ``sky._shard`` samples the kinetic energy
    per nucleon (it starts the cascade at ``e_tot = e_nuc + m``).
    """
    t = np.asarray(e_kin_nucleon, float)
    p = np.sqrt(t * (t + 2.0 * m_nucleon))
    return (float(a_mass) / abs(float(z))) * p


def step_m(rigidity, table=STEP_TABLE):
    """RK4 arc-length step [m] for rigidity ``R`` [GV] (``STEP_TABLE``).

    The gyroradius near the surface is ``rho = R / (c B) ~ 55.6 km x R[GV]`` at
    60 uT: ``geomag_backtrace``'s fixed 100 km step exceeds it for a 1.7 GV
    proton.  Measured (``MILESTONE3A_NOTES.md`` sec. 3, 2 x 3000 states, every
    state traced at every fixed step from 400 to 6.25 km): above 10 GV no
    verdict changes even at 400 km; at 5-10 GV none at 100 km (2 of 618 at
    200 km); below 5 GV 50-100 km flip 0.2-0.4% of the states, 25 km none
    but one chaotic state.
    """
    R = np.asarray(rigidity, float)
    ds = np.full(R.shape, float(table[-1][1]))
    for r_max, d in reversed(table[:-1]):
        ds = np.where(R < r_max, d, ds)
    return ds


def _step_class(ds):
    """Steps are already discrete (``STEP_TABLE``); kept for callers that pass
    an arbitrary ``ds``: round down to ``DS_MAX_M / 2^k`` (k >= 0)."""
    k = np.ceil(np.log2(DS_MAX_M / np.asarray(ds, float)) - 1e-9)
    return DS_MAX_M / 2.0 ** np.maximum(k, 0.0)


# ---------------------------------------------------------------------------
# The batched admittance test
# ---------------------------------------------------------------------------
def _floor_radius_m(floor, r_launch_m):
    if floor is None or floor == "injection":
        return r_launch_m - 1.0              # the launch sphere itself (1 m slack)
    if floor == "sea_level":
        return gb.RE
    return gb.RE + 1.0e3 * float(floor)      # an altitude [km]


def trace_allowed(r_cm, u, rigidity, charge=1, *, date=DEFAULT_DATE,
                  floor="injection", r_escape=R_ESCAPE_RE, ds_m=None,
                  max_s_m=MAX_S_M, model=DEFAULT_MODEL, fast_field=True,
                  r_switch=4.0, m_hat=None):
    """Binary cutoff verdict for N injection states (serial, this process).

    Parameters
    ----------
    r_cm : (N, 3) positions [cm], all on one launch sphere (the injection
        sphere, |r| = R_E + 100 km, for the 3D test).
    u : (N, 3) **inward** unit velocity of the primary (as sampled by
        ``geometry.sample_injection_patch``).  The back-trace is launched along
        ``-u``.
    rigidity : (N,) or scalar, nucleus rigidity [GV] (:func:`rigidity_gv`).
    charge : sign of the primary's charge, scalar or (N,) (+1 for nuclei).
    floor : re-entry rule -- ``"injection"`` (default; forbidden if the
        back-trace drops below the launch sphere: Honda), ``"sea_level"``
        (``geomag_backtrace``'s ``r < R_E``), or an altitude in km.
    r_escape : escape radius [R_E]; ``ds_m`` the step [m] (scalar, or ``None``
        for the rigidity-dependent :func:`step_m`); ``max_s_m`` the path budget
        [m] after which a still-bound trace counts as forbidden.
    model, fast_field : field model (``"IGRF13"``/``"IGRF14"``) and whether to
        use :class:`FastIGRF` (``False`` = stock ppigrf, the reference path).
    r_switch, m_hat : passed to ``backtrace_vec`` (``r_switch=0`` with
        ``m_hat`` = a unit axis is a pure centred dipole -- the Stoermer case).

    Returns
    -------
    allowed : (N,) bool.  The verdict of a state does not depend on the rest of
    the batch (every trajectory is integrated independently; gated in
    ``test_cutoff.py``).

    Validity: reaching ``r_escape`` proves escape only if the bound (Stoermer
    inner) region, of size ``R_E sqrt(C_St / R)``, lies inside it, i.e.
    ``R > C_St / r_escape^2 = 0.59 GV`` for 10 R_E.  Below that, bound orbits
    cross 10 R_E and are wrongly allowed (measured in a pure dipole, notes
    sec. 5.2).  The MC's primaries start at 1.7 GV.
    """
    r_m = np.atleast_2d(np.asarray(r_cm, float)) * 1.0e-2
    v = np.atleast_2d(np.asarray(u, float))
    n = len(r_m)
    if n == 0:
        return np.zeros(0, bool)
    R = np.broadcast_to(np.asarray(rigidity, float), (n,)).astype(float)
    q = np.broadcast_to(np.asarray(charge, float), (n,)).astype(float)
    if m_hat is None:
        m_hat = gb.dipole_axis()
    rad = np.linalg.norm(r_m, axis=1)
    # one floor per launch radius (1 m bins): the 3D test has a single radius
    rkey = np.round(rad).astype(np.int64)
    if ds_m is None:
        ds = _step_class(step_m(R))
    else:
        ds = np.broadcast_to(np.asarray(ds_m, float), (n,))
    out = np.zeros(n, bool)
    with igrf_field(model, fast_field):
        for rk in np.unique(rkey):
            sel_r = rkey == rk
            r_floor = _floor_radius_m(floor, float(np.min(rad[sel_r])))
            for d in np.unique(ds[sel_r]):
                idx = np.where(sel_r & (ds == d))[0]
                qq = q[idx]
                if np.all(qq == qq[0]):
                    qq = float(qq[0])
                out[idx] = gb.backtrace_vec(
                    r_m[idx], -v[idx], R[idx], date, m_hat, charge=qq,
                    r_escape=r_escape, ds=float(d), max_s=max_s_m,
                    r_floor=r_floor, r_switch=r_switch)
    return out


def stoermer_floor_gv(r_cm, m_hat=None):
    """Stoermer cutoff [GV] of the **most favourable** direction (horizon,
    West for positive charges) at each position, in the IGRF's tilted centred
    dipole (``geomag_backtrace.dipole_axis()``, ``B0``):
    ``C cos^4(lam) / (rho^2 [1 + sqrt(1 + cos^3(lam))]^2)``.  In a dipole no
    trajectory below it can be allowed; in IGRF allowed states were found down
    to 0.77 x it on the 40 deg patch (``GATE_LO = 0.70`` carries the margin)."""
    if m_hat is None:
        m_hat = gb.dipole_axis()
    r = np.atleast_2d(np.asarray(r_cm, float)) * 1.0e-2
    rn = np.linalg.norm(r, axis=1)
    sl = (r / rn[:, None]) @ (np.asarray(m_hat, float) / np.linalg.norm(m_hat))
    cl = np.sqrt(np.maximum(1.0 - sl * sl, 0.0))
    rho = rn / gb.RE
    return C_ST_GV * cl ** 4 / (rho ** 2 * (1.0 + np.sqrt(1.0 + cl ** 3)) ** 2)


def gate(r_cm, u, rigidity, lo=GATE_LO, r_skip=R_SKIP_GV, mu_min=GATE_MU_MIN):
    """Exact pre-filter: ``-1`` forbidden without a trace (``R < lo R_St,min``),
    ``+1`` allowed without a trace (``R > r_skip`` and ``mu > mu_min``), ``0``
    trace.  Both bounds are measured, not assumed (MILESTONE3A_NOTES.md
    sec. 5.4): no test state violates either.  The upper gate is only valid for
    states that reach the dense atmosphere, i.e. for showers that can score."""
    r_cm = np.atleast_2d(np.asarray(r_cm, float))
    u = np.atleast_2d(np.asarray(u, float))
    R = np.broadcast_to(np.asarray(rigidity, float), (len(r_cm),))
    mu = -np.einsum("ij,ij->i", u, r_cm) / np.linalg.norm(r_cm, axis=1)
    out = np.zeros(len(R), int)
    out[(R > r_skip) & (mu > mu_min)] = 1
    out[R < lo * stoermer_floor_gv(r_cm)] = -1
    return out


def trace_gated(r_cm, u, rigidity, charge=1, **kw):
    """:func:`gate`, then :func:`trace_allowed` on the undecided states only.
    Returns ``(allowed, n_traced)``."""
    g = gate(r_cm, u, rigidity)
    allowed = g == 1
    t = np.where(g == 0)[0]
    if len(t):
        r_cm = np.atleast_2d(np.asarray(r_cm, float))
        u = np.atleast_2d(np.asarray(u, float))
        R = np.broadcast_to(np.asarray(rigidity, float), (len(g),))
        q = np.broadcast_to(np.asarray(charge, float), (len(g),))
        allowed[t] = trace_allowed(r_cm[t], u[t], R[t], q[t], **kw)
    return allowed, len(t)


def _admit_worker(args):
    idx, r_cm, u, R, q, kw = args
    t0 = time.process_time()
    a = trace_allowed(r_cm, u, R, q, **kw)
    return idx, a, time.process_time() - t0


def admittance(r_cm, u, rigidity, charge=1, *, n_jobs=1, n_chunks=None,
               return_cpu=False, **kw):
    """:func:`trace_allowed` spread over ``n_jobs`` forked workers.

    States are dealt round-robin into ``n_chunks`` (default ``n_jobs``) chunks,
    so every chunk carries the same mix of states.  One large chunk per worker
    is the right default: a call lives as long as its slowest (bouncing)
    trajectory, and that tail is paid once per call and step class
    (``MILESTONE3A_NOTES.md`` sec. 6).  ``n_jobs=1`` runs in-process -- use
    that inside an existing pool worker (``sky._shard``); daemonic pool
    workers cannot fork.  ``return_cpu`` also returns the summed worker CPU
    seconds.
    """
    r_cm = np.atleast_2d(np.asarray(r_cm, float))
    u = np.atleast_2d(np.asarray(u, float))
    n = len(r_cm)
    R = np.broadcast_to(np.asarray(rigidity, float), (n,)).astype(float)
    q = np.broadcast_to(np.asarray(charge, float), (n,)).astype(float)
    if n_jobs is None or n_jobs <= 1 or n < 2:
        t0 = time.process_time()
        a = trace_allowed(r_cm, u, R, q, **kw)
        return (a, time.process_time() - t0) if return_cpu else a
    import multiprocessing as mp
    n_chunks = int(n_chunks or n_jobs)
    n_chunks = max(1, min(n_chunks, n))
    args = []
    for j in range(n_chunks):
        idx = np.arange(j, n, n_chunks)
        args.append((idx, r_cm[idx], u[idx], R[idx], q[idx], kw))
    out = np.zeros(n, bool)
    cpu = 0.0
    with mp.get_context("fork").Pool(min(int(n_jobs), n_chunks)) as pool:
        for idx, a, c in pool.imap_unordered(_admit_worker, args):
            out[idx] = a
            cpu += c
    return (out, cpu) if return_cpu else out


# ---------------------------------------------------------------------------
# Azimuthal copies: rotations about the site vertical
# ---------------------------------------------------------------------------
def site_axis(lat_deg, lon_deg):
    """Unit vector from the Earth's centre through the site (its local vertical)."""
    return gb._local_frame(lat_deg, lon_deg)[0]


def rotate_about(v, axis, angle_rad):
    """Rotate vectors ``v`` (..., 3) by ``angle_rad`` (scalar, or an array
    broadcasting against ``v``'s leading dimensions) about the unit ``axis``
    (right-handed: counter-clockwise seen from the tip of ``axis``)."""
    k = np.asarray(axis, float)
    k = k / np.linalg.norm(k)
    v = np.asarray(v, float)
    ang = np.asarray(angle_rad, float)
    c, s = np.cos(ang)[..., None], np.sin(ang)[..., None]
    kv = (v @ k)[..., None]
    return v * c + np.cross(k, v) * s + kv * k * (1.0 - c)


def copy_shifts_deg(n_az, phase_deg=0.0):
    """Compass shifts ``phase + 360 j / n_az`` [deg] of the ``n_az`` copies;
    ``phase_deg`` scalar -> shape ``(n_az,)``, array ``(N,)`` -> ``(N, n_az)``."""
    ph = np.asarray(phase_deg, float)
    return ph[..., None] + 360.0 * np.arange(int(n_az)) / int(n_az)


def copy_states(r_cm, u, axis, n_az, phase_deg=0.0):
    """Injection states of the ``n_az`` azimuthal copies of each primary.

    Copy ``j`` is the original rotated **clockwise seen from above the site** by
    ``delta_j = phase + 360 j / n_az`` degrees about the site vertical
    ``axis`` (a right-handed rotation by ``-delta_j``), so a neutrino arriving
    at the site from compass azimuth ``a`` arrives in copy ``j`` from
    ``a + delta_j`` (exactly at the site centre; elsewhere in the cap the local
    frame adds the meridian convergence, which the scorer handles by
    re-locating the rotated ray).  ``phase_deg`` may be a scalar or one phase
    per primary, shape ``(N,)`` (a random phase per shower keeps the copies off
    a fixed azimuth-bin pattern).  The rotation is an isometry of the sphere,
    so the copy's injection point stays on the injection sphere at the same
    distance from the site and its direction keeps its local zenith -- only the
    direction relative to B changes, which is why every copy needs its own
    trace.  Returns ``(r, u)`` of shape ``(N, n_az, 3)``.
    """
    r_cm = np.atleast_2d(np.asarray(r_cm, float))
    u = np.atleast_2d(np.asarray(u, float))
    n = len(r_cm)
    d = np.radians(np.broadcast_to(copy_shifts_deg(n_az, phase_deg), (n, int(n_az))))
    rr = rotate_about(np.repeat(r_cm[:, None, :], n_az, axis=1), axis, -d)
    uu = rotate_about(np.repeat(u[:, None, :], n_az, axis=1), axis, -d)
    return rr, uu


def admittance_copies(r_cm, u, rigidity, axis, n_az, charge=1, phase_deg=0.0,
                      **kw):
    """``(N, n_az)`` verdicts of every azimuthal copy, one batched call."""
    rr, uu = copy_states(r_cm, u, axis, n_az, phase_deg)
    n = rr.shape[0]
    R = np.repeat(np.broadcast_to(np.asarray(rigidity, float), (n,)), n_az)
    q = np.repeat(np.broadcast_to(np.asarray(charge, float), (n,)), n_az)
    a = admittance(rr.reshape(-1, 3), uu.reshape(-1, 3), R, q, **kw)
    return a.reshape(n, n_az)


# ---------------------------------------------------------------------------
# Upper cutoff R_U per state (geomag_backtrace.scan_upper_cutoff, our tracer)
# ---------------------------------------------------------------------------
def _scan_worker(args):
    r_cm, u, q, r_lo, r_hi, coarse, tol, ret_a, gated, kw = args

    def allowed_fn(idx, R):
        qq = q if np.ndim(q) == 0 else q[idx]
        if gated:
            return trace_gated(r_cm[idx], u[idx], R, qq, **kw)[0]
        return trace_allowed(r_cm[idx], u[idx], R, qq, **kw)

    return gb.scan_upper_cutoff(allowed_fn, len(r_cm), r_lo, r_hi, coarse, tol,
                                return_admittance=ret_a)


def upper_cutoff(r_cm, u, charge=1, *, r_lo=0.5, r_hi=70.0,
                 coarse_step=gb.COARSE_STEP_GV, tol=gb.BISECT_TOL_GV, n_jobs=1,
                 return_admittance=False, gated=False, **kw):
    """Upper cutoff ``R_U`` [GV] (highest forbidden rigidity) of each state.

    The scan is ``geomag_backtrace.scan_upper_cutoff`` unchanged (a top-down
    ladder of step <= ``coarse_step`` from ``r_hi``, then bisection to
    ``tol``); every trace goes through :func:`trace_allowed`, so the floor,
    escape radius, step rule and field model are those of the binary test.
    ``gated=True`` decides the ladder rungs that :func:`gate` can decide
    without a trace (identical result, since the gate is exact; it skips the
    expensive bouncing rungs far below the cutoff).
    Returns ``(rc, saturated)`` (+ ``(rs, A)`` with ``return_admittance``).
    """
    r_cm = np.atleast_2d(np.asarray(r_cm, float))
    u = np.atleast_2d(np.asarray(u, float))
    n = len(r_cm)
    q = np.asarray(charge, float)
    nj = max(1, min(int(n_jobs or 1), n))
    chunks = [np.arange(j, n, nj) for j in range(nj)]
    args = [(r_cm[c], u[c], q if q.ndim == 0 else q[c], r_lo, r_hi, coarse_step,
             tol, return_admittance, gated, kw) for c in chunks]
    if nj == 1:
        res = [_scan_worker(args[0])]
    else:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(nj) as pool:
            res = pool.map(_scan_worker, args)
    rc = np.zeros(n)
    sat = np.zeros(n, bool)
    A = None
    for c, x in zip(chunks, res):
        rc[c], sat[c] = x[0], x[1]
        if return_admittance:
            if A is None:
                A = np.zeros((n, x[3].shape[1]))
            A[c] = x[3]
    if return_admittance:
        return rc, sat, res[0][2], A
    return rc, sat


class SiteCutoffMap:
    """Detector-anchored cutoff map ``R_c(zenith, azimuth)`` at a site -- the
    cutoff convention of a **1D** calculation (launch at the ground, sea-level
    floor: ``geomag_backtrace.cutoff_map``), for the correlated 1D reference
    tally of ``sky.py`` (PHASE2_PLAN sec. 5.2: "the cutoff is evaluated at the
    site line-of-sight").  Built once per (site, date, model) with
    :func:`upper_cutoff`; ``allowed(R, zen, az)`` is the sharp step ``R > R_c``
    with ``R_c`` bilinear in (zenith, azimuth) -- the same sharp-``R_U``
    convention as the delivered ``mceq3d`` engine (``SIGMA_LNR = 0``)."""

    def __init__(self, lat_deg, lon_deg, zen=None, az=None, h_km=0.0,
                 floor="sea_level", n_jobs=1, **kw):
        self.zen = MAP_ZEN if zen is None else np.asarray(zen, float)
        az = MAP_AZ if az is None else np.asarray(az, float)
        Z, A = np.meshgrid(self.zen, az, indexing="ij")
        r, u = state_above_site(lat_deg, lon_deg, Z.ravel(), A.ravel(), h_km)
        kw.setdefault("gated", True)     # exact: skips the bouncing low rungs
        rc, self.saturated = upper_cutoff(r, u, 1, floor=floor, n_jobs=n_jobs, **kw)
        rc = rc.reshape(len(self.zen), len(az))
        self.az = np.append(az, az[0] + 360.0)          # periodic closure
        self.rc = np.concatenate([rc, rc[:, :1]], axis=1)

    def __call__(self, zen_deg, az_deg):
        from scipy.interpolate import RegularGridInterpolator
        f = getattr(self, "_f", None)
        if f is None:
            f = self._f = RegularGridInterpolator((self.zen, self.az), self.rc,
                                                  bounds_error=False, fill_value=None)
        z = np.clip(np.asarray(zen_deg, float), self.zen[0], self.zen[-1])
        a = (np.asarray(az_deg, float) - self.az[0]) % 360.0 + self.az[0]
        z, a = np.broadcast_arrays(z, a)
        return f(np.stack([z.ravel(), a.ravel()], -1)).reshape(z.shape)

    def allowed(self, rigidity, zen_deg, az_deg):
        return np.asarray(rigidity, float) > self(zen_deg, az_deg)


# ---------------------------------------------------------------------------
# Local-frame helpers (validation and studies)
# ---------------------------------------------------------------------------
def state_above_site(lat_deg, lon_deg, zenith_deg, azimuth_deg, h_km=H_INJ_KM):
    """Injection state directly above a site arriving from (zenith, compass
    azimuth): ``(r_cm (N,3), u (N,3))`` with ``u`` the inward velocity."""
    z = np.atleast_1d(np.asarray(zenith_deg, float))
    a = np.atleast_1d(np.asarray(azimuth_deg, float))
    z, a = np.broadcast_arrays(z, a)
    up = site_axis(lat_deg, lon_deg)
    u = np.array([gb.arrival_direction(lat_deg, lon_deg, zz, aa)
                  for zz, aa in zip(z.ravel(), a.ravel())])
    r = np.tile((R_EARTH_CM + h_km * 1.0e5) * up, (len(u), 1))
    return r, u


# ---------------------------------------------------------------------------
# Test sets and studies (``python cutoff.py <study> --outdir ...``)
# ---------------------------------------------------------------------------
# The test states mimic sky.py's sampling: positions uniform per area on the
# injection patch, inward-Lambert directions, energy per nucleon stratified in
# ln E over [1, 1e4] GeV with n_k ~ E_k^-0.8 (sky.STRATA_FRAC), H3a p/He split
# of the samples 0.8/0.2 (sky.GROUP_FRAC).  Replicated here (not imported) so
# the studies do not depend on a driver that is being edited concurrently.
STUDY_GROUPS = (("p", 1, 1, 0.8), ("He", 4, 2, 0.2))
STUDY_STRATA = np.logspace(0.0, 4.0, 9)


def _strata_frac(alloc_exp=-0.8):
    sc = np.sqrt(STUDY_STRATA[1:] * STUDY_STRATA[:-1])
    f = sc ** alloc_exp
    return f / f.sum()


def first_ground_crossing(r_cm, u):
    """First crossing of the straight ray ``r + s u`` (s > 0) with the ground
    sphere, or NaN rows when it misses."""
    r = np.atleast_2d(np.asarray(r_cm, float))
    u = np.atleast_2d(np.asarray(u, float))
    b = np.einsum("ij,ij->i", r, u)
    c = np.einsum("ij,ij->i", r, r) - R_EARTH_CM ** 2
    disc = b * b - c
    s = -b - np.sqrt(np.maximum(disc, 0.0))
    s = np.where((disc >= 0.0) & (s > 0.0), s, np.nan)
    return r + s[:, None] * u


def sample_test_states(seed, n, site="kamioka", theta_inj=40.0, cap_deg=None,
                       alloc_exp=-0.8):
    """Injection states as ``sky.py`` samples them (see above).

    ``cap_deg``: keep only states whose straight primary ray first meets the
    ground inside a cap of that radius around the site -- the 1D proxy for a
    shower that scores (the subset whose verdict matters).
    Returns a dict of arrays (``r_cm, u, R, A, Z, e_kin, stratum, alpha_deg``).
    """
    import geometry as geo
    rng = np.random.default_rng(seed)
    lat, lon = SITES[site]
    axis = site_axis(lat, lon)
    fr = _strata_frac(alloc_exp)
    gf = np.array([g[3] for g in STUDY_GROUPS])
    gf = gf / gf.sum()
    out = {k: [] for k in ("r_cm", "u", "e_kin", "A", "Z", "stratum")}
    have = 0
    while have < n:
        m = max(4 * (n - have), 1000) if cap_deg is None else 50 * (n - have) + 5000
        r, u = geo.sample_injection_patch(rng, R_INJ_CM, axis, theta_inj, m)
        k = rng.choice(len(fr), size=m, p=fr)
        e = np.exp(rng.uniform(np.log(STUDY_STRATA[k]), np.log(STUDY_STRATA[k + 1])))
        g = rng.choice(len(gf), size=m, p=gf)
        A = np.array([STUDY_GROUPS[i][1] for i in g], float)
        Z = np.array([STUDY_GROUPS[i][2] for i in g], float)
        keep = np.ones(m, bool)
        if cap_deg is not None:
            h = first_ground_crossing(r, u)
            ok = np.isfinite(h[:, 0])
            ca = np.full(m, -2.0)
            ca[ok] = (h[ok] @ axis) / np.linalg.norm(h[ok], axis=1)
            keep = ca >= np.cos(np.radians(cap_deg))
        idx = np.where(keep)[0][: n - have]
        for key, val in (("r_cm", r), ("u", u), ("e_kin", e), ("A", A), ("Z", Z),
                         ("stratum", k)):
            out[key].append(val[idx])
        have += len(idx)
    d = {k: np.concatenate(v) for k, v in out.items()}
    d["R"] = rigidity_gv(d["e_kin"]) * d["A"] / d["Z"]
    d["alpha_deg"] = np.degrees(np.arccos(np.clip(
        (d["r_cm"] @ axis) / np.linalg.norm(d["r_cm"], axis=1), -1, 1)))
    d["mu"] = (-np.einsum("ij,ij->i", d["u"], d["r_cm"])
               / np.linalg.norm(d["r_cm"], axis=1))
    return d


def _rbins_table(R, flags, labels, edges=(0, 3, 5, 10, 20, 40, 80, 1e9)):
    """Per-rigidity-bin counts of each boolean column in ``flags``."""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (R >= lo) & (R < hi)
        rows.append([lo, hi, int(s.sum())] + [int(np.sum(f[s])) for f in flags])
    hdr = "R_lo  R_hi      n " + " ".join(f"{x:>10s}" for x in labels)
    lines = [hdr] + [f"{r[0]:5.0f} {r[1]:5.0g} {r[2]:6d} " +
                     " ".join(f"{x:10d}" for x in r[3:]) for r in rows]
    return "\n".join(lines)


def _log(outdir, name, text):
    print(text, flush=True)
    with open(os.path.join(outdir, name), "a") as f:
        f.write(text + "\n")


def study_steps(outdir, n=3000, n_jobs=12, seed=11, max_s_m=4.0e8):
    """Verdict convergence in the RK4 step: fixed ds in {400..6.25} km vs a
    fine reference, on the patch set (A) and the cap-pointing set (B)."""
    sets = {"A": sample_test_states(seed, n),
            "B": sample_test_states(seed + 1, n, cap_deg=10.0)}
    ds_list = [4.0e5, 2.0e5, 1.0e5, 5.0e4, 2.5e4, 1.25e4, 6.25e3]
    res = {}
    for name, st in sets.items():
        for ds in ds_list:
            t0 = time.time()
            # one chunk per worker: a chunk lives as long as its slowest
            # (bouncing) member, so few large chunks amortise that tail
            a, cpu = admittance(st["r_cm"], st["u"], st["R"], 1, ds_m=ds,
                                max_s_m=max_s_m, n_jobs=n_jobs, n_chunks=n_jobs,
                                return_cpu=True)
            res[(name, ds)] = a
            _log(outdir, "steps.txt", f"set {name} ds={ds/1e3:7.2f} km: allowed "
                 f"{a.mean():.4f}  wall {time.time()-t0:7.1f} s  cpu {cpu:8.1f} s")
            np.savez(os.path.join(outdir, "steps.npz"),
                     **{f"{k[0]}_{k[1]:.0f}": v for k, v in res.items()},
                     **{f"{k}_{f}": v[f] for k, v in sets.items()
                        for f in ("R", "alpha_deg", "mu")})
        ref = res[(name, ds_list[-1])]
        flips = [res[(name, d)] != ref for d in ds_list[:-1]]
        _log(outdir, "steps.txt", f"set {name}: verdict flips vs ds=6.25 km, per R bin\n"
             + _rbins_table(st["R"], flips, [f"{d/1e3:.1f}km" for d in ds_list[:-1]]))
    return res


def study_floor(outdir, n=3000, n_jobs=12, seed=21):
    """Re-entry rule: floors at 0 (sea level), 20, 50, 80 km and the injection
    sphere (100 km); escape radius 10/15/25 R_E (the path budget has its own
    study)."""
    sets = {"A": sample_test_states(seed, n),
            "B": sample_test_states(seed + 1, n, cap_deg=10.0)}
    floors = ["sea_level", 20.0, 50.0, 80.0, "injection"]
    store = {}
    for name, st in sets.items():
        res = {}
        for fl in floors:
            t0 = time.time()
            a = admittance(st["r_cm"], st["u"], st["R"], 1, floor=fl,
                           n_jobs=n_jobs, n_chunks=n_jobs)
            res[fl] = a
            store[f"{name}_floor_{fl}"] = a
            _log(outdir, "floor.txt", f"set {name} floor={fl!s:>10}: allowed "
                 f"{a.mean():.4f} ({time.time()-t0:.0f} s)")
        inj = res["injection"]
        flips = [res[f] & ~inj for f in floors[:-1]]
        _log(outdir, "floor.txt", f"set {name}: allowed with floor X but forbidden "
             "with the injection-sphere floor, per R bin\n"
             + _rbins_table(st["R"], flips + [inj], [str(f) for f in floors[:-1]]
                            + ["allowed"]))
        for res_key, kw in (("esc15", dict(r_escape=15.0)),
                            ("esc25", dict(r_escape=25.0))):
            a = admittance(st["r_cm"], st["u"], st["R"], 1, n_jobs=n_jobs,
                           n_chunks=n_jobs, **kw)
            store[f"{name}_{res_key}"] = a
            _log(outdir, "floor.txt", f"set {name} {res_key}: allowed {a.mean():.4f}, "
                 f"flips vs default {int(np.sum(a != inj))} / {len(a)}")
        for f in ("R", "alpha_deg", "mu", "e_kin", "A"):
            store[f"{name}_{f}"] = st[f]
        np.savez(os.path.join(outdir, "floor.npz"), **store)
    return store


def study_budget(outdir, n=3000, n_jobs=12, seed=61,
                 budgets=(5.0e7, 1.0e8, 2.0e8, 4.0e8, 8.0e8), **_):
    """Path budget ``max_s``: verdict flips and CPU (the budget also sets the
    length of the bouncing tail that a whole batch has to wait for)."""
    sets = {"A": sample_test_states(seed, n),
            "B": sample_test_states(seed + 1, n, cap_deg=10.0)}
    store = {}
    for name, st in sets.items():
        res = {}
        for b in budgets:
            t0 = time.time()
            a, cpu = admittance(st["r_cm"], st["u"], st["R"], 1, max_s_m=b,
                                n_jobs=n_jobs, n_chunks=n_jobs, return_cpu=True)
            res[b] = a
            store[f"{name}_{b:.0e}"] = a
            _log(outdir, "budget.txt", f"set {name} max_s={b:.1e} m: allowed "
                 f"{a.mean():.4f}"
                 f", wall {time.time()-t0:6.1f} s, cpu {cpu:7.1f} s "
                 f"({cpu / len(a) * 1e3:.1f} ms/trace)")
        ref = res[budgets[-1]]
        flips = [res[b] != ref for b in budgets[:-1]]
        _log(outdir, "budget.txt", f"set {name}: flips vs max_s={budgets[-1]:.0e} m\n"
             + _rbins_table(st["R"], flips, [f"{b:.0e}" for b in budgets[:-1]]))
        store[f"{name}_R"] = st["R"]
        np.savez(os.path.join(outdir, "budget.npz"), **store)
    return store


def study_highr(outdir, n=3000, n_jobs=12, seed=71, **_):
    """Forbidden fraction of cap-pointing states at high rigidity: where can
    the trace be skipped (an exact gate if nothing scored is forbidden above
    it)?"""
    st = sample_test_states(seed, n, cap_deg=10.0)
    rng = np.random.default_rng(seed + 5)
    R = np.exp(rng.uniform(np.log(20.0), np.log(2000.0), n))
    out = {}
    for fl in ("injection", "sea_level"):
        a = admittance(st["r_cm"], st["u"], R, 1, floor=fl, n_jobs=n_jobs,
                       n_chunks=n_jobs)
        out[fl] = a
        f = ~a
        _log(outdir, "highr.txt", f"floor={fl}: {int(f.sum())} forbidden of {n} "
             f"(R in [20, 2000] GV); max forbidden R {R[f].max() if f.any() else 0:.1f}"
             f" GV; forbidden above 60/80/100 GV: {int(np.sum(f & (R > 60)))}/"
             f"{int(np.sum(f & (R > 80)))}/{int(np.sum(f & (R > 100)))}")
    np.savez(os.path.join(outdir, "highr.npz"), R=R, mu=st["mu"],
             alpha=st["alpha_deg"], **{k: v for k, v in out.items()})
    return out


def study_bend(outdir, n=3000, seed=5, lam=85.0, **_):
    """Bend of the primary between the injection sphere and its first
    interaction, ``L c B_perp / R`` with ``L`` the straight path to
    ``X = lam`` g/cm2 and ``B_perp`` at its midpoint (``shower.py`` moves the
    primary in a straight line; Honda bends it).  Geometry and field only."""
    import atmosphere as atm
    st = sample_test_states(seed, n, cap_deg=10.0)
    L = np.empty(n)
    for i in range(n):
        s_bound, _ = atm.path_to_exit(st["r_cm"][i], st["u"][i])
        sx, _, hit = atm.advance_grammage(st["r_cm"][i], st["u"][i], lam, s_bound)
        L[i] = sx if hit else s_bound
    mid = (st["r_cm"] + 0.5 * L[:, None] * st["u"]) * 1.0e-2
    with igrf_field(DEFAULT_MODEL):
        B = gb._bfield_igrf_cart(mid, DEFAULT_DATE, gb.dipole_axis())
    bperp = np.linalg.norm(np.cross(st["u"], B), axis=1)
    theta = np.degrees((L * 1.0e-2) * gb.C * bperp / (st["R"] * 1.0e9))
    z = np.degrees(np.arccos(st["mu"]))
    for z0, z1 in ((0, 45), (45, 70), (70, 85)):
        for r0, r1 in ((5, 15), (15, 40), (40, 1e9)):
            t = (z >= z0) & (z < z1) & (st["R"] >= r0) & (st["R"] < r1)
            if t.sum() > 5:
                _log(outdir, "bend.txt", f"z_inj {z0}-{z1} deg, R {r0}-{r1:g} GV: "
                     f"n={int(t.sum())} L median {np.median(L[t]) / 1e5:.0f} km, bend "
                     f"median {np.median(theta[t]):.2f} deg, 90% "
                     f"{np.percentile(theta[t], 90):.2f} deg")
    return theta


def study_escape(outdir, n=3000, n_jobs=12, seed=22, **_):
    """Escape radius 10 vs 15 R_E with a path budget long enough for both
    (8e8 m), on the cap-pointing set of the floor study."""
    st = sample_test_states(seed, n, cap_deg=10.0)
    res = {}
    for esc in (10.0, 15.0):
        res[esc] = admittance(st["r_cm"], st["u"], st["R"], 1, r_escape=esc,
                              max_s_m=8.0e8, n_jobs=n_jobs, n_chunks=n_jobs)
        _log(outdir, "escape.txt", f"r_escape {esc:4.1f} R_E, max_s 8e8 m: allowed "
             f"{res[esc].mean():.4f}")
    f = res[10.0] != res[15.0]
    _log(outdir, "escape.txt", f"flips 10 vs 15 R_E: {int(f.sum())} / {n} (R of flips: "
         f"{np.round(st['R'][f], 2).tolist()})")
    return res


# -- validation (a): injection points above Kamioka on the cutoff_map grid --


def _map_states(lat, lon, h_km, zen=MAP_ZEN, az=MAP_AZ):
    Z, A = np.meshgrid(zen, az, indexing="ij")
    return state_above_site(lat, lon, Z.ravel(), A.ravel(), h_km)


#: (label, lat, lon, launch altitude [km], kw for trace_allowed, r_lo, r_hi, charge)
KAMIOKA_VARIANTS = (
    # Phase-1 convention exactly (cutoff_map defaults; ppigrf's default IGRF-14)
    ("ph1", 36.43, 137.31, 0.0, dict(model="IGRF14", floor="sea_level",
                                     r_escape=15.0, ds_m=1.0e5, max_s_m=1.6e9),
     0.5, 55.0, 1),
    ("V0_14", 36.4267, 137.31, 0.0, dict(model="IGRF14", floor="sea_level"),
     2.0, 70.0, 1),
    ("V0", 36.4267, 137.31, 0.0, dict(floor="sea_level"), 2.0, 70.0, 1),
    ("V1", 36.4267, 137.31, 100.0, dict(floor="sea_level"), 2.0, 70.0, 1),
    ("V2", 36.4267, 137.31, 100.0, dict(floor="injection"), 2.0, 70.0, 1),
    ("V2neg", 36.4267, 137.31, 100.0, dict(floor="injection"), 2.0, 70.0, -1),
)


def study_kamioka(outdir, n_jobs=12, **_):
    """R_U maps above Kamioka: Phase-1 reproduction, model, launch altitude,
    re-entry rule and charge sign, on mceq3d's cutoff-map grid."""
    store = {"zen": MAP_ZEN, "az": MAP_AZ}
    for lab, lat, lon, h, kw, r_lo, r_hi, q in KAMIOKA_VARIANTS:
        t0 = time.time()
        r, u = _map_states(lat, lon, h)
        rc, sat = upper_cutoff(r, u, q, r_lo=r_lo, r_hi=r_hi, n_jobs=n_jobs, **kw)
        store[lab] = rc.reshape(len(MAP_ZEN), len(MAP_AZ))
        store[lab + "_sat"] = sat.reshape(len(MAP_ZEN), len(MAP_AZ))
        np.savez(os.path.join(outdir, "kamioka_maps.npz"), **store)
        m = store[lab]
        iz = {z: int(np.argmin(abs(MAP_ZEN - z))) for z in (0, 60, 80, 87)}
        ia = {a: int(np.argmin(abs(MAP_AZ - a))) for a in (0, 90, 180, 270)}
        _log(outdir, "kamioka.txt", f"{lab:6s} (h={h:.0f} km, q={q:+d}, {kw}) "
             f"{time.time()-t0:.0f} s, saturated {int(sat.sum())}\n"
             f"   vertical {m[iz[0], 0]:6.2f} GV; N/E/S/W at 60: "
             + " ".join(f"{m[iz[60], ia[a]]:6.2f}" for a in (0, 90, 180, 270))
             + "; at 80: " + " ".join(f"{m[iz[80], ia[a]]:6.2f}"
                                      for a in (0, 90, 180, 270))
             + "; at 87: " + " ".join(f"{m[iz[87], ia[a]]:6.2f}"
                                      for a in (0, 90, 180, 270)))
    return store


# -- validation (b): pure centred dipole, where Stoermer is exact -------------
MZ = np.array([0.0, 0.0, 1.0])
#: Stoermer constant of geomag_backtrace's dipole: B0 R_E c [GV] (4 x the
#: vertical equatorial cutoff, 14.66 GV for B0 = 3.07e-5 T)
C_ST_GV = gb.B0 * gb.RE * gb.C / 1.0e9


def stoermer_gv(r_cm, u, charge=1, m_hat=MZ):
    """Stoermer cutoff [GV] of each state in a centred dipole along ``m_hat``:
    ``C cos^4(lam) / (rho^2 [1 + sqrt(1 - cos^3(lam) sin(eps) sin(xi))]^2)``,
    ``lam`` the magnetic latitude, ``rho = r/R_E``, ``eps`` the zenith and
    ``xi`` the magnetic azimuth (clockwise from magnetic North, East = 90) of
    the **arrival** direction; ``xi -> -xi`` for negative charge.  Below it
    every trajectory is bound (a theorem for the dipole), so no trace may be
    allowed there."""
    r = np.atleast_2d(np.asarray(r_cm, float)) * 1.0e-2
    d = -np.atleast_2d(np.asarray(u, float))          # towards the source
    m = np.asarray(m_hat, float) / np.linalg.norm(m_hat)
    rho = np.linalg.norm(r, axis=1) / gb.RE
    up = r / (rho * gb.RE)[:, None]
    sl = up @ m
    cl = np.sqrt(np.maximum(1.0 - sl * sl, 0.0))
    north = m[None, :] - sl[:, None] * up
    north /= np.linalg.norm(north, axis=1)[:, None]
    east = np.cross(north, up)
    ce = np.einsum("ij,ij->i", d, up)
    se = np.sqrt(np.maximum(1.0 - ce * ce, 0.0))
    sx = np.where(se > 1e-12, np.einsum("ij,ij->i", d, east) / np.maximum(se, 1e-12), 0.0)
    sx = sx * np.sign(charge)
    return C_ST_GV * cl ** 4 / (rho ** 2 * (1.0 + np.sqrt(1.0 - cl ** 3 * se * sx)) ** 2)


def _random_sphere_states(rng, n, r_cm, lat_max=70.0):
    """Positions uniform on a sphere band |lat| <= lat_max, inward Lambert."""
    import geometry as geo
    out_r, out_u = [], []
    have = 0
    while have < n:
        r, u = geo.sample_injection(rng, r_cm, 2 * n)
        keep = np.abs(r[:, 2] / r_cm) <= np.sin(np.radians(lat_max))
        out_r.append(r[keep]), out_u.append(u[keep])
        have += int(keep.sum())
    return np.concatenate(out_r)[:n], np.concatenate(out_u)[:n]


def study_dipole(outdir, n_jobs=12, n=2000, seed=31, **_):
    """Pure aligned dipole (``r_switch=0``, ``m_hat=z``): the Stoermer
    theorem, the vertical R_U vs Stoermer, the r^-2 launch scaling, E/W, and
    the exact charge-mirror symmetry."""
    rng = np.random.default_rng(seed)
    kw = dict(m_hat=MZ, r_switch=0.0)
    store = {}
    # 1. theorem: below Stoermer nothing is allowed, whatever the floor
    r, u = _random_sphere_states(rng, n, R_INJ_CM)
    rst = stoermer_gv(r, u)
    f_lo = rng.uniform(0.30, 0.97, n)
    f_hi = rng.uniform(1.00, 2.00, n)
    for fl in ("injection", "sea_level"):
        a_lo = admittance(r, u, f_lo * rst, 1, floor=fl, n_jobs=n_jobs, **kw)
        a_hi = admittance(r, u, f_hi * rst, 1, floor=fl, n_jobs=n_jobs, **kw)
        store[f"theorem_{fl}_lo"], store[f"theorem_{fl}_hi"] = a_lo, a_hi
        # reaching r_escape only proves escape if the Stoermer length
        # R_E sqrt(C/R) -- the size of the bound inner region -- is inside it:
        # R > C / r_escape^2 = 0.59 GV for 10 R_E
        dom = f_lo * rst > C_ST_GV / R_ESCAPE_RE ** 2
        _log(outdir, "dipole.txt", f"theorem, floor={fl}: allowed below Stoermer "
             f"(R/R_St in [0.30, 0.97]) {int(a_lo.sum())}/{n}, of which with "
             f"R > {C_ST_GV / R_ESCAPE_RE ** 2:.2f} GV {int(a_lo[dom].sum())}/"
             f"{int(dom.sum())}; above (R/R_St in [1, 2]) {a_hi.mean():.3f}")
    store.update(theorem_rst=rst, theorem_flo=f_lo, theorem_fhi=f_hi)
    # 2. vertical R_U vs Stoermer, launch at sea level and at the injection sphere
    # <= 50 deg: R_St >= 2.5 GV, so the 25 km minimum step is <= 0.15 gyroradii
    lats = np.arange(0.0, 51.0, 10.0)
    rows = []
    for h, fl in ((0.0, "sea_level"), (H_INJ_KM, "injection"), (H_INJ_KM, "sea_level")):
        rr = np.array([state_above_site(la, 0.0, 0.0, 0.0, h)[0][0] for la in lats])
        uu = np.array([state_above_site(la, 0.0, 0.0, 0.0, h)[1][0] for la in lats])
        rc, _ = upper_cutoff(rr, uu, 1, r_lo=1.0, r_hi=20.0, coarse_step=0.25,
                             tol=0.01, n_jobs=min(n_jobs, len(lats)), floor=fl, **kw)
        st = stoermer_gv(rr, uu)
        rows.append((h, fl, rc, st))
        store[f"vert_h{h:.0f}_{fl}"] = rc
        store[f"vert_h{h:.0f}_st"] = st
        _log(outdir, "dipole.txt", f"vertical R_U, launch h={h:.0f} km, floor={fl}: "
             + "  ".join(f"lat{la:.0f}: {a:.3f}/{b:.3f}={a/b:.4f}"
                         for la, a, b in zip(lats, rc, st)))
    _log(outdir, "dipole.txt", "launch-altitude ratio R_U(100 km)/R_U(0 km) (sea-level "
         "floor) = " + " ".join(f"{x:.4f}" for x in rows[2][2] / rows[0][2])
         + f"; Stoermer (R_E/R_inj)^2 = {(gb.RE / (gb.RE + 1e5)) ** 2:.4f}")
    # 3. East/West at lat 30, zenith 60 and 80, both charges
    for q in (1, -1):
        rr, uu = [], []
        for z in (60.0, 80.0):
            for az in (90.0, 270.0):
                a, b = state_above_site(30.0, 0.0, z, az, H_INJ_KM)
                rr.append(a[0]), uu.append(b[0])
        rr, uu = np.array(rr), np.array(uu)
        rc, _ = upper_cutoff(rr, uu, q, r_lo=1.0, r_hi=70.0, n_jobs=4, **kw)
        st = stoermer_gv(rr, uu, q)
        store[f"ew_q{q:+d}"], store[f"ew_q{q:+d}_st"] = rc, st
        _log(outdir, "dipole.txt", f"lat 30, q={q:+d}: z60 E/W R_U "
             f"{rc[0]:.2f}/{rc[1]:.2f}"
             f" (St {st[0]:.2f}/{st[1]:.2f}); z80 E/W R_U {rc[2]:.2f}/{rc[3]:.2f}"
             f" (St {st[2]:.2f}/{st[3]:.2f})")
    # 4. exact mirror symmetry: (q, state) <-> (-q, P state), P: y -> -y
    r, u = _random_sphere_states(rng, n, R_INJ_CM)
    R = np.exp(rng.uniform(np.log(3.0), np.log(80.0), n))
    P = np.array([1.0, -1.0, 1.0])
    a_p = admittance(r, u, R, 1, n_jobs=n_jobs, **kw)
    a_m = admittance(r * P, u * P, R, -1, n_jobs=n_jobs, **kw)
    store.update(mirror_p=a_p, mirror_m=a_m)
    _log(outdir, "dipole.txt", f"mirror symmetry: {int(np.sum(a_p != a_m))} "
         f"disagreements / {n} (allowed fraction {a_p.mean():.3f})")
    np.savez(os.path.join(outdir, "dipole.npz"), **store)
    return store


# -- cost ---------------------------------------------------------------------
def study_cost(outdir, n_jobs=12, seed=41,
               parts="field,band,single,batch,gated,mixA,scaling",
               **_):
    """Field cost; CPU per trace by rigidity band, by verdict (fast vs stock
    field), and -- the number that matters -- vs batch size on the
    cap-pointing mix; whole-patch mix; multiprocessing scaling.  Single-process
    timings are process CPU seconds."""
    import ppigrf
    parts = set(parts.split(","))
    st = sample_test_states(seed, 10000, cap_deg=10.0)
    rng = np.random.default_rng(seed + 1)

    def timed(r, u, R, **kw):
        t0 = time.process_time()
        a = trace_allowed(r, u, R, **kw)
        return a, time.process_time() - t0

    if "field" in parts:
        fi = fast_igrf(DEFAULT_MODEL)
        for npt in (1, 4, 100, 1000):
            rr = 6471.0 + 3000.0 * rng.random(npt)
            th, ph = 1.0 + 178.0 * rng.random(npt), 360.0 * rng.random(npt)
            for lab, fn in (("ppigrf", lambda: ppigrf.igrf_gc(rr, th, ph, DEFAULT_DATE)),
                            ("FastIGRF", lambda: fi(rr, th, ph, DEFAULT_DATE))):
                k, t0 = 0, time.process_time()
                while time.process_time() - t0 < 1.0:
                    fn()
                    k += 1
                dt = (time.process_time() - t0) / k
                _log(outdir, "cost.txt", f"field {lab:8s} N={npt:5d}: {dt * 1e3:7.3f} "
                     f"ms/call = {dt / npt * 1e6:8.2f} us/point")
    if "band" in parts:   # CPU per trace by rigidity band, cap-pointing geometry
        for lo, hi in ((1.7, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 40.0),
                       (40.0, 100.0), (100.0, 1000.0)):
            R = np.exp(rng.uniform(np.log(lo), np.log(hi), 1000))
            a, cpu = timed(st["r_cm"][:1000], st["u"][:1000], R)
            _log(outdir, "cost.txt", f"band R {lo:6.1f}-{hi:6.1f} GV, batch 1000: "
                 f"allowed {a.mean():.3f}, {cpu:7.1f} s = {cpu:.2f} ms CPU/trace")
    if "single" in parts:  # all-allowed (80 GV) / mostly forbidden (4 GV), fast vs stock
        for label, R0 in (("allowed", 80.0), ("forbidden", 4.0)):
            for nb, fast in ((100, True), (100, False), (3000, True)):
                a, cpu = timed(st["r_cm"][:nb], st["u"][:nb], R0, fast_field=fast)
                _log(outdir, "cost.txt", f"{label:9s} R={R0:5.1f} GV batch {nb:5d} "
                     f"field={'fast ' if fast else 'stock'}: allowed {a.mean():.3f}, "
                     f"{cpu / nb * 1e3:8.2f} ms CPU/trace")
    if "batch" in parts:   # realistic (cap-pointing) mix vs batch size
        for nb in (100, 300, 1000, 3000, 10000):
            a, cpu = timed(st["r_cm"][:nb], st["u"][:nb], st["R"][:nb])
            _log(outdir, "cost.txt", f"mix B (cap-pointing) batch {nb:6d}: allowed "
                 f"{a.mean():.3f}, {cpu / nb * 1e3:7.2f} ms CPU/trace, "
                 f"{nb / cpu:6.1f} traces/s/core")
    if "gated" in parts:   # the same mix through the exact pre-filter
        for nb in (100, 300, 1000, 3000, 10000):
            t0 = time.process_time()
            a, n_tr = trace_gated(st["r_cm"][:nb], st["u"][:nb], st["R"][:nb])
            cpu = time.process_time() - t0
            _log(outdir, "cost.txt", f"gated mix B batch {nb:6d}: traced "
                 f"{n_tr / nb:.3f}, "
                 f"allowed {a.mean():.3f}, {cpu / nb * 1e3:7.2f} ms CPU/state "
                 f"({cpu / max(n_tr, 1) * 1e3:7.2f} ms/traced state)")
    if "mixA" in parts:
        sa = sample_test_states(seed + 2, 3000)
        a, cpu = timed(sa["r_cm"], sa["u"], sa["R"])
        _log(outdir, "cost.txt", f"mix A (whole patch) batch 3000: allowed {a.mean():.3f}"
             f", {cpu / 3000 * 1e3:.2f} ms CPU/trace")
    if "scaling" in parts:  # fixed 6000 states, one chunk per worker
        t_ref = None
        for nj in (1, 2, 4, 8, 12):
            if nj > n_jobs:
                break
            t0 = time.time()
            a, cpu = admittance(st["r_cm"][:6000], st["u"][:6000], st["R"][:6000],
                                n_jobs=nj, n_chunks=nj, return_cpu=True)
            wall = time.time() - t0
            t_ref = wall if t_ref is None else t_ref
            _log(outdir, "cost.txt", f"scaling n_jobs={nj:2d}: wall {wall:7.1f} s, cpu "
                 f"{cpu:7.1f} s, speed-up {t_ref / wall:5.2f}, efficiency "
                 f"{t_ref / wall / nj:4.2f}, {6000 / wall:6.1f} traces/s")


# -- acceleration (iii): cached R_U on a (position, direction) grid -----------
class RUCache:
    """Upper cutoffs ``R_U`` on a grid of injection states around a site, used
    as a **gate**: a state whose rigidity is far above every corner of its grid
    cell is declared allowed, far below every corner forbidden, and only the
    rest is traced exactly.

    Grid.  Positions on the injection sphere in azimuthal-equidistant
    coordinates about the site, ``(x_east, y_north) = alpha (sin b, cos b)``
    [deg] (``alpha`` the angular distance, ``b`` the compass bearing), nodes
    every ``dpos`` deg out to ``alpha_max``; at each position node the arrival
    direction in that point's own local frame, zenith nodes ``zen`` and compass
    azimuth every ``daz`` deg.  A state's 4-D cell has 16 corners.

    Gate.  With ``lo, hi`` the min/max of ``R_U`` over the 16 corners:
    ``R > hi (1 + m_hi)`` -> allowed; ``R < lo (1 - m_lo)`` -> forbidden (with a
    wider ``m_lo_islands`` where ``lo < r_islands``: the low-cutoff, vertical and
    West, directions are the only ones with allowed islands below ``R_U``,
    ``PHASE1_RESULTS``/``PAPER_DRAFT`` Table 1); else trace.
    """

    def __init__(self, site="kamioka", alpha_max=7.5, dpos=2.5,
                 zen=(0, 10, 20, 30, 40, 50, 60, 65, 70, 74, 77, 80, 83),
                 daz=15.0):
        self.lat, self.lon = SITES[site]
        self.up, self.north, self.east = gb._local_frame(self.lat, self.lon)
        k = int(np.ceil(alpha_max / dpos))
        self.xg = dpos * np.arange(-k, k + 1)
        self.dpos = float(dpos)
        self.zen = np.asarray(zen, float)
        self.daz = float(daz)
        self.az = np.arange(0.0, 360.0, daz)
        self.rc = None

    # geometry --------------------------------------------------------------
    def _pos_from_xy(self, x, y):
        al = np.radians(np.hypot(x, y))
        b = np.arctan2(x, y)
        rh = (np.cos(al)[:, None] * self.up
              + np.sin(al)[:, None] * (np.cos(b)[:, None] * self.north
                                       + np.sin(b)[:, None] * self.east))
        return rh

    def _xy_of(self, r_cm):
        rh = r_cm / np.linalg.norm(r_cm, axis=1)[:, None]
        al = np.degrees(np.arccos(np.clip(rh @ self.up, -1, 1)))
        b = np.arctan2(rh @ self.east, rh @ self.north)
        return al * np.sin(b), al * np.cos(b)

    @staticmethod
    def _frames(rh):
        lat = np.arcsin(np.clip(rh[:, 2], -1, 1))
        lon = np.arctan2(rh[:, 1], rh[:, 0])
        up = rh
        north = np.stack([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon),
                          np.cos(lat)], 1)
        east = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], 1)
        return up, north, east

    def node_states(self):
        X, Y = np.meshgrid(self.xg, self.xg, indexing="ij")
        rh = self._pos_from_xy(X.ravel(), Y.ravel())
        up, no, ea = self._frames(rh)
        Z, A = np.meshgrid(np.radians(self.zen), np.radians(self.az), indexing="ij")
        z, a = Z.ravel(), A.ravel()
        # velocity of an arrival from (z, a): -(cos z up + sin z (cos a N + sin a E))
        d = (np.cos(z)[None, :, None] * up[:, None, :]
             + np.sin(z)[None, :, None] * (np.cos(a)[None, :, None] * no[:, None, :]
                                           + np.sin(a)[None, :, None] * ea[:, None, :]))
        r = np.repeat(R_INJ_CM * rh, len(z), axis=0)
        return r, -d.reshape(-1, 3)

    def build(self, n_jobs=12, r_lo=2.0, r_hi=70.0, **kw):
        import resource
        r, u = self.node_states()
        t0 = time.time()
        c0 = (resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime
              + resource.getrusage(resource.RUSAGE_SELF).ru_utime)
        rc, sat = upper_cutoff(r, u, 1, r_lo=r_lo, r_hi=r_hi, n_jobs=n_jobs,
                               gated=True, **kw)
        self.build_wall = time.time() - t0
        self.build_cpu = (resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime
                          + resource.getrusage(resource.RUSAGE_SELF).ru_utime - c0)
        self.rc = rc.reshape(len(self.xg), len(self.xg), len(self.zen), len(self.az))
        self.sat = sat
        return self

    def bounds(self, r_cm, u):
        """``(lo, hi, inside)``: min/max ``R_U`` over each state's 16 corners."""
        x, y = self._xy_of(r_cm)
        rh = r_cm / np.linalg.norm(r_cm, axis=1)[:, None]
        up, no, ea = self._frames(rh)
        d = -u
        z = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", d, up), -1, 1)))
        a = np.degrees(np.arctan2(np.einsum("ij,ij->i", d, ea),
                                  np.einsum("ij,ij->i", d, no))) % 360.0
        g = self.xg
        ix = np.clip(np.searchsorted(g, x) - 1, 0, len(g) - 2)
        iy = np.clip(np.searchsorted(g, y) - 1, 0, len(g) - 2)
        iz = np.clip(np.searchsorted(self.zen, z) - 1, 0, len(self.zen) - 2)
        ia = (a // self.daz).astype(int) % len(self.az)
        inside = ((x >= g[0]) & (x <= g[-1]) & (y >= g[0]) & (y <= g[-1])
                  & (z <= self.zen[-1]))
        vals = []
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    for da in (0, 1):
                        vals.append(self.rc[ix + dx, iy + dy, iz + dz,
                                            (ia + da) % len(self.az)])
        v = np.stack(vals, 1)
        return v.min(1), v.max(1), inside

    def gate(self, r_cm, u, R, m_hi=0.05, m_lo=0.10, m_lo_islands=0.25,
             r_islands=20.0):
        """+1 allowed, -1 forbidden, 0 = trace exactly (also outside the grid)."""
        lo, hi, inside = self.bounds(r_cm, u)
        ml = np.where(lo < r_islands, m_lo_islands, m_lo)
        out = np.zeros(len(R), int)
        out[R > hi * (1.0 + m_hi)] = 1
        out[R < lo * (1.0 - ml)] = -1
        out[~inside] = 0
        return out


def study_cache(outdir, n_jobs=12, seed=51, n_base=600, n_az=24, **_):
    """Speed-up and error of the R_U-cache gate on cap-pointing states and
    their azimuthal copies, against exact traces of the same states."""
    cache = RUCache()
    t0 = time.time()
    r_nodes, _u = cache.node_states()
    cache.build(n_jobs=n_jobs)
    _log(outdir, "cache.txt", f"cache: {len(r_nodes)} nodes ({len(cache.xg)}^2 "
         f"positions x {len(cache.zen)} zen x {len(cache.az)} az), built in "
         f"{time.time()-t0:.0f} s "
         f"wall / {cache.build_cpu:.0f} s CPU on {n_jobs} workers "
         f"({cache.build_cpu / len(r_nodes):.2f} s/node), saturated "
         f"{int(cache.sat.sum())}; R_U range {cache.rc.min():.2f}-"
         f"{cache.rc.max():.2f} GV")
    np.savez(os.path.join(outdir, "cache_rc.npz"), rc=cache.rc, xg=cache.xg,
             zen=cache.zen, az=cache.az)
    st = sample_test_states(seed, 4 * n_base, cap_deg=10.0)
    sel = np.where(st["alpha_deg"] <= 6.0)[0][:n_base]
    axis = site_axis(cache.lat, cache.lon)
    rr, uu = copy_states(st["r_cm"][sel], st["u"][sel], axis, n_az)
    R = np.repeat(st["R"][sel], n_az)
    r2, u2 = rr.reshape(-1, 3), uu.reshape(-1, 3)
    t0 = time.time()
    exact, cpu_all = admittance(r2, u2, R, n_jobs=n_jobs, n_chunks=n_jobs,
                                return_cpu=True)
    _log(outdir, "cache.txt", f"test: {len(R)} states ({len(sel)} cap-pointing x {n_az}"
         f" copies, alpha <= 6 deg), exact allowed fraction {exact.mean():.3f}; all "
         f"traced: CPU {cpu_all:.0f} s (wall {time.time() - t0:.0f} s)")
    t0 = time.process_time()
    g_ex = gate(r2, u2, R)
    g_ca = cache.gate(r2, u2, R)
    t_gate = (time.process_time() - t0) / len(R)
    g_co = np.where(g_ex != 0, g_ex, g_ca)
    for lab, g in (("exact pre-filter", g_ex), ("cache gate", g_ca),
                   ("pre-filter + cache", g_co)):
        tr = g == 0
        a_tr, cpu_tr = admittance(r2[tr], u2[tr], R[tr], n_jobs=n_jobs,
                                  n_chunks=n_jobs, return_cpu=True)
        err_a = int(np.sum((g == 1) & ~exact))
        err_f = int(np.sum((g == -1) & exact))
        _log(outdir, "cache.txt", f"  {lab:19s}: decided allowed {np.mean(g == 1):.3f}, "
             f"forbidden {np.mean(g == -1):.3f}, traced {np.mean(tr):.3f}; errors "
             f"{err_a} + {err_f}; CPU {cpu_tr:.0f} s -> speed-up "
             f"{cpu_all / max(cpu_tr, 1e-9):.2f}")
    _log(outdir, "cache.txt", f"  gate lookup {t_gate * 1e6:.1f} us/state")
    for m_hi, m_lo, m_isl in ((0.02, 0.05, 0.15), (0.05, 0.10, 0.25),
                              (0.10, 0.15, 0.35), (0.20, 0.25, 0.50)):
        gg = cache.gate(r2, u2, R, m_hi=m_hi, m_lo=m_lo, m_lo_islands=m_isl)
        _log(outdir, "cache.txt", f"  margins hi/lo/islands {m_hi:.2f}/{m_lo:.2f}/"
             f"{m_isl:.2f}: trace {np.mean(gg == 0):.3f}, errors "
             f"{int(np.sum((gg == 1) & ~exact))} + {int(np.sum((gg == -1) & exact))}"
             f" of {len(R)}")
    g = g_ca
    np.savez(os.path.join(outdir, "cache_test.npz"), exact=exact, gate=g, R=R)
    return cache


def study_cachebatch(outdir, seed=51, n_base=600, n_az=24, cache_dir=None, **_):
    """The cache-study test states traced in ONE call per variant, one
    process: all / exact pre-filter / cache gate.  Separates the per-state
    saving from the per-call tail (the cache study's worker chunks shrink with
    the traced fraction).  Needs ``cache_rc.npz`` from ``study_cache``."""
    d = np.load(os.path.join(cache_dir or outdir, "cache_rc.npz"))
    cache = RUCache()
    assert np.allclose(cache.xg, d["xg"]) and np.allclose(cache.zen, d["zen"])
    cache.rc = d["rc"]
    st = sample_test_states(seed, 4 * n_base, cap_deg=10.0)
    sel = np.where(st["alpha_deg"] <= 6.0)[0][:n_base]
    rr, uu = copy_states(st["r_cm"][sel], st["u"][sel], site_axis(cache.lat, cache.lon),
                         n_az)
    R = np.repeat(st["R"][sel], n_az)
    r2, u2 = rr.reshape(-1, 3), uu.reshape(-1, 3)
    for lab, g in (("all traced", np.zeros(len(R), int)),
                   ("exact pre-filter", gate(r2, u2, R)),
                   ("cache gate", cache.gate(r2, u2, R))):
        tr = g == 0
        t0 = time.process_time()
        trace_allowed(r2[tr], u2[tr], R[tr])
        cpu = time.process_time() - t0
        _log(outdir, "cachebatch.txt", f"{lab:17s}: one call of {int(tr.sum()):6d} "
             "states,"
             f" CPU {cpu:7.1f} s = {cpu / len(R) * 1e3:6.2f} ms per state of the "
             f"{len(R)} ({cpu / max(tr.sum(), 1) * 1e3:6.2f} ms per traced state)")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("study", choices=["steps", "floor", "budget", "highr", "kamioka",
                                      "dipole", "cost", "cache", "bend",
                                      "escape", "cachebatch"])
    ap.add_argument("--outdir", default="/home/pigrange/mc3d_work/runs/m3a/out")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--n-jobs", type=int, default=12)
    ap.add_argument("--parts", default=None, help="cost study: comma-separated parts")
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)
    {"steps": study_steps, "floor": study_floor, "budget": study_budget,
     "highr": study_highr, "kamioka": study_kamioka, "dipole": study_dipole,
     "cost": study_cost, "cache": study_cache, "bend": study_bend,
     "escape": study_escape, "cachebatch": study_cachebatch}[a.study](
        a.outdir, n=a.n, n_jobs=a.n_jobs, **({"parts": a.parts} if a.parts else {}))


if __name__ == "__main__":
    main()
