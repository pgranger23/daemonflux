"""Neutrino scorers.

``YieldScorer``
    Collinear / 1D reference: tallies every neutrino produced, binned in
    (species, E).  With one primary injected at the top of a vertical column
    this is exactly MCEq's ``get_solution`` after
    ``set_single_primary_particle`` -- the milestone-1 closure observable.

``CapScorer``
    3D: a neutrino emitted at ``r`` along ``u`` is scored if the straight ray
    crosses the ground sphere inside the virtual-detector cap.  **Nested caps**
    are scored simultaneously on the same neutrino (a neutrino inside the 2.5
    deg cap is also inside the 5, 7.5 and 10 deg caps), so the four estimates
    are maximally correlated and their finite-size extrapolation to
    ``Omega_D -> 0`` has small variance.  Weights carry ``1 / A_cap`` so the
    tally is a flux per cm2; the ``cos`` of the incidence angle is *not*
    divided out here -- the scorer stores both the plain count and the
    ``1/cos`` -weighted count so either the "crossing the shell" or the
    "specific intensity" normalisation can be formed downstream (see
    PHASE2_PLAN.md sec. 5.2).

Both scorers accumulate the sum of weights and the sum of squared weights so
every reported number carries a statistical error, and both are additive
(``merge``) so multiprocessing shards combine exactly.
"""

from __future__ import annotations

import math

import numpy as np

from constants import NU_NAME

SPECIES = (12, -12, 14, -14)


class YieldScorer:
    def __init__(self, e_bins):
        self.e_bins = np.asarray(e_bins, float)
        n = len(self.e_bins) - 1
        self.sw = {s: np.zeros(n) for s in SPECIES}
        self.sw2 = {s: np.zeros(n) for s in SPECIES}
        self.n_prim = 0.0

    def add(self, pdg, e, r, u, w):
        if pdg not in self.sw:
            return
        i = int(np.searchsorted(self.e_bins, e) - 1)
        if 0 <= i < len(self.sw[pdg]):
            self.sw[pdg][i] += w
            self.sw2[pdg][i] += w * w

    def merge(self, other):
        for s in SPECIES:
            self.sw[s] += other.sw[s]
            self.sw2[s] += other.sw2[s]
        self.n_prim += other.n_prim
        return self

    def dnde(self, pdg):
        """(dN/dE per primary, its 1-sigma error)."""
        wid = np.diff(self.e_bins)
        n = max(self.n_prim, 1.0)
        return self.sw[pdg] / wid / n, np.sqrt(self.sw2[pdg]) / wid / n

    def to_dict(self):
        d = {"e_bins": self.e_bins, "n_prim": np.array([self.n_prim])}
        for s in SPECIES:
            d[f"sw_{s}"] = self.sw[s]
            d[f"sw2_{s}"] = self.sw2[s]
        return d

    @classmethod
    def from_dict(cls, d):
        o = cls(d["e_bins"])
        o.n_prim = float(np.atleast_1d(d["n_prim"])[0])
        for s in SPECIES:
            o.sw[s] = d[f"sw_{s}"]
            o.sw2[s] = d[f"sw2_{s}"]
        return o


class CapScorer:
    """(species, E, cosZ, azimuth) histogram over nested detector caps."""

    def __init__(self, caps, e_bins, cz_bins=None, az_bins=None):
        self.caps = list(caps)
        self.e_bins = np.asarray(e_bins, float)
        self.cz_bins = np.linspace(-1, 1, 21) if cz_bins is None \
            else np.asarray(cz_bins, float)
        self.az_bins = np.linspace(0, 360, 13) if az_bins is None \
            else np.asarray(az_bins, float)
        shape = (len(self.caps), len(SPECIES), len(self.e_bins) - 1,
                 len(self.cz_bins) - 1, len(self.az_bins) - 1)
        self.sw = np.zeros(shape)
        self.sw2 = np.zeros(shape)
        self.n_prim = 0.0
        self._sidx = {s: k for k, s in enumerate(SPECIES)}

    def add(self, pdg, e, r, u, w):
        k = self._sidx.get(pdg)
        if k is None:
            return
        ie = int(np.searchsorted(self.e_bins, e) - 1)
        if not (0 <= ie < len(self.e_bins) - 1):
            return
        for ic, cap in enumerate(self.caps):
            hit, zen, az, cosi = cap.score(r, u)
            if not hit:
                continue
            cz = np.cos(np.deg2rad(zen))
            iz = int(np.searchsorted(self.cz_bins, cz) - 1)
            ia = int(np.searchsorted(self.az_bins, az) - 1)
            if not (0 <= iz < len(self.cz_bins) - 1):
                continue
            ia = min(max(ia, 0), len(self.az_bins) - 2)
            ww = w / cap.area_cm2
            self.sw[ic, k, ie, iz, ia] += ww
            self.sw2[ic, k, ie, iz, ia] += ww * ww

    def merge(self, other):
        self.sw += other.sw
        self.sw2 += other.sw2
        self.n_prim += other.n_prim
        return self

    def to_dict(self):
        return {"sw": self.sw, "sw2": self.sw2,
                "n_prim": np.array([self.n_prim]),
                "e_bins": self.e_bins, "cz_bins": self.cz_bins,
                "az_bins": self.az_bins,
                "cap_theta": np.array([c.theta_deg for c in self.caps])}


def name(pdg):
    return NU_NAME[pdg]


# ---------------------------------------------------------------------------
# The production cone: angle between the PRIMARY and the NEUTRINO
# ---------------------------------------------------------------------------
#: E_nu bands [GeV] of the alpha histogram
CONE_BANDS = (0.1, 0.3, 0.5, 1.0, 3.0)
#: 10 bins/decade from 0.1 to 10 GeV for the moments; the bin centres are
#: exactly the factorised engine's energy grid (``offaxis_excess_channel_v2.npz``
#: key ``e``: 0.1122, 0.1413, ... GeV), so widths compare point by point
CONE_EMOM = np.logspace(-1.0, 1.0, 21)
#: parent channels, by the PDG id of the particle whose decay made the neutrino
CONE_CHANNELS = ("pi", "K", "mu", "other")
_CONE_CH = {211: 0, -211: 0, 321: 1, -321: 1, 130: 1, 310: 1, 13: 2, -13: 2}
#: classes of the primary's local cos(zenith) at the injection point
CONE_ZP = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
#: moment slots: sum w, w^2, w alpha, w alpha^2, w cos(alpha); alpha in deg
CONE_MOMENTS = ("w", "w2", "wa", "wa2", "wcos")
_RAD2DEG = 180.0 / math.pi


class ConeScorer:
    """Space angle ``alpha`` between the **primary** direction and the
    **neutrino** direction, for every neutrino produced above threshold.

    This is the cone the factorised engine (``tools/mceq3d``, Eq. 8 of the
    off-axis construction) parametrises by the space-angle RMS widths
    ``sigma_pi``, ``sigma_k``, ``sigma_mu_numu``, ``sigma_mu_nue`` of
    ``offaxis_excess_channel_v2.npz`` -- "the distribution of the primary
    direction given the neutrino direction", which, the kernel being
    azimuthally symmetric about the primary axis, is the distribution of
    ``alpha`` over the produced neutrinos.  Unlike the engine, which builds it
    from single-interaction meson moments plus decay kinematics, a cascade has
    the full multi-generation history in it, so this measures the cone
    directly.

    Every neutrino produced above ``e_nu_min`` is entered, with its physical
    weight, whether or not it reaches the detector cap: the kernel is a
    property of the production, and conditioning on the cap would fold in the
    detector geometry.

    Axes:

    ``h[species, band, channel, zp, gen, alpha]``  1-deg bins in ``alpha``,
        ``CONE_BANDS`` in ``E_nu``, ``CONE_CHANNELS`` by the decaying parent
        (direct ``pi``; ``K`` = K+-, K_L, K_S including K_l3; ``mu`` = muon
        decay whatever made the muon; ``other``), ``zp`` = the primary's
        local ``cos zenith`` at injection in ``CONE_ZP`` classes, ``gen`` =
        0 if the neutrino's ancestry holds at most one hadronic interaction
        (the parent meson came from the primary's first interaction), 1 if
        more -- which separates the single-interaction cone the engine assumes
        from the multi-generation accumulation.
    ``m[species, e10, channel, zp, gen, CONE_MOMENTS]``  exact moments on the
        ``CONE_EMOM`` grid (no binning error in the RMS).

    Both are plain weighted sums, so shards merge additively.
    """

    def __init__(self, bands=CONE_BANDS, e_mom=CONE_EMOM, n_alpha=180):
        self.bands = np.asarray(bands, float)
        self.e_mom = np.asarray(e_mom, float)
        self.n_alpha = int(n_alpha)
        nb, ne = len(self.bands) - 1, len(self.e_mom) - 1
        nz = len(CONE_ZP) - 1
        nch = len(CONE_CHANNELS)
        self.h = np.zeros((len(SPECIES), nb, nch, nz, 2, self.n_alpha))
        self.h2 = np.zeros_like(self.h)
        self.m = np.zeros((len(SPECIES), ne, nch, nz, 2, len(CONE_MOMENTS)))
        self._b = tuple(float(x) for x in self.bands)
        self._lge0 = math.log(float(self.e_mom[0]))
        self._dlge = math.log(float(self.e_mom[1] / self.e_mom[0]))
        self._ne = ne
        self._dalpha = 180.0 / self.n_alpha

    @staticmethod
    def channel(parent):
        return _CONE_CH.get(int(parent), 3)

    def add(self, k, e, cos_alpha, w, parent=0, gen=1, zp=0):
        """Enter one neutrino of species index ``k`` (``SPECIES.index``)."""
        ca = -1.0 if cos_alpha < -1.0 else (1.0 if cos_alpha > 1.0 else cos_alpha)
        a = math.acos(ca) * _RAD2DEG
        ch = _CONE_CH.get(parent, 3)
        g = 0 if gen <= 1 else 1
        ie = int((math.log(e) - self._lge0) / self._dlge)
        if 0 <= ie < self._ne:
            v = self.m[k, ie, ch, zp, g]
            v[0] += w
            v[1] += w * w
            v[2] += w * a
            v[3] += w * a * a
            v[4] += w * ca
        b = self._b
        if b[0] <= e < b[-1]:
            ib = 0
            while e >= b[ib + 1]:
                ib += 1
            ia = int(a / self._dalpha)
            if ia >= self.n_alpha:
                ia = self.n_alpha - 1
            self.h[k, ib, ch, zp, g, ia] += w
            self.h2[k, ib, ch, zp, g, ia] += w * w

    def merge(self, other):
        self.h += other.h
        self.h2 += other.h2
        self.m += other.m
        return self

    def to_dict(self, prefix="cone_"):
        return {prefix + "h": self.h, prefix + "h2": self.h2,
                prefix + "m": self.m, prefix + "bands": self.bands,
                prefix + "emom": self.e_mom,
                prefix + "channels": np.array(CONE_CHANNELS),
                prefix + "zp": np.array(CONE_ZP)}

    @classmethod
    def from_dict(cls, d, prefix="cone_"):
        o = cls(d[prefix + "bands"], d[prefix + "emom"],
                np.asarray(d[prefix + "h"]).shape[-1])
        o.h = np.array(d[prefix + "h"])
        o.h2 = np.array(d[prefix + "h2"])
        o.m = np.array(d[prefix + "m"])
        return o

    # -- summaries ----------------------------------------------------------
    @staticmethod
    def rms(m):
        """Space-angle RMS [deg] from a moment array whose last axis is
        ``CONE_MOMENTS``; NaN where empty."""
        w = m[..., 0]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(w > 0, np.sqrt(m[..., 3] / np.where(w > 0, w, 1.0)),
                            np.nan)

    @staticmethod
    def mean_cos(m):
        w = m[..., 0]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(w > 0, m[..., 4] / np.where(w > 0, w, 1.0), np.nan)


# ---------------------------------------------------------------------------
# Nested caps + the correlated 1D reference on the same showers
# ---------------------------------------------------------------------------
class NestedCaps:
    """Concentric ground-sphere caps, intersected **once** per neutrino.

    ``CapScorer`` above calls ``DetectorCap.score`` once per cap, which repeats
    the ray-sphere intersection, the ``latlon_of`` and the ``local_frame``
    four times over -- ~50 us per neutrino, and the scorer is called a few
    times per shower.  The caps are concentric, so containment is *monotone*:
    the crossing point sits inside every cap at least as large as the smallest
    one that contains it.  So the geometry is computed once and the answer is a
    single index ``i0``; caps ``i0 .. n-1`` are hit.

    Everything here is scalar Python arithmetic on purpose -- ``numpy`` on
    3-vectors costs several microseconds per call and this is the inner loop.
    """

    def __init__(self, lat_deg, lon_deg, theta_degs=(2.5, 5.0, 7.5, 10.0)):
        from constants import R_EARTH_CM
        self.thetas = tuple(sorted(float(t) for t in theta_degs))
        self.cos_theta = tuple(np.cos(np.deg2rad(t)) for t in self.thetas)
        self.areas = tuple(2.0 * np.pi * R_EARTH_CM ** 2 * (1.0 - c)
                           for c in self.cos_theta)
        self.omegas = tuple(2.0 * np.pi * (1.0 - c) for c in self.cos_theta)
        la, lo = np.deg2rad(float(lat_deg)), np.deg2rad(float(lon_deg))
        self.axis = (float(np.cos(la) * np.cos(lo)),
                     float(np.cos(la) * np.sin(lo)), float(np.sin(la)))
        self.lat, self.lon = float(lat_deg), float(lon_deg)
        self.r_e = float(R_EARTH_CM)

    def crossing(self, r, u):
        """First ground-sphere crossing of the ray, or ``None``.

        Only the **first** (near) root is taken.  For a straight ray whose
        origin is above the ground sphere this is the only physical crossing:
        ``r(s)`` has a single minimum, so a ray with ``dr/ds > 0`` never comes
        back.  Up-going neutrinos therefore never register -- see the note in
        ``DualCapScorer``.
        """
        b = r[0] * u[0] + r[1] * u[1] + r[2] * u[2]
        c = r[0] * r[0] + r[1] * r[1] + r[2] * r[2] - self.r_e * self.r_e
        disc = b * b - c
        if disc < 0.0:
            return None
        sq = disc ** 0.5
        s = -b - sq
        if s <= 0.0:
            s = -b + sq
            if s <= 0.0:
                return None
        return (r[0] + s * u[0], r[1] + s * u[1], r[2] + s * u[2])

    def locate(self, r, u):
        """``(i0, cos_zenith, azimuth_deg, cos_incidence)`` or ``None``.

        ``i0`` is the smallest cap index containing the crossing; caps
        ``i0 ..`` all contain it.  Zenith and azimuth are in the local frame of
        the **crossing point**, not of the cap centre (Honda scores at the
        crossing; the two differ by up to ``theta_D``).
        """
        h = self.crossing(r, u)
        if h is None:
            return None
        n = (h[0] ** 2 + h[1] ** 2 + h[2] ** 2) ** 0.5
        nx, ny, nz = h[0] / n, h[1] / n, h[2] / n
        ca = nx * self.axis[0] + ny * self.axis[1] + nz * self.axis[2]
        if ca < self.cos_theta[0]:
            i0 = None
            for k in range(1, len(self.cos_theta)):
                if ca >= self.cos_theta[k]:
                    i0 = k
                    break
            if i0 is None:
                return None
        else:
            i0 = 0
        # local frame at the crossing point
        sla = nz
        cla = (nx * nx + ny * ny) ** 0.5
        if cla < 1e-12:
            e_n, e_e = (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
        else:
            clo, slo = nx / cla, ny / cla
            e_n = (-sla * clo, -sla * slo, cla)
            e_e = (-slo, clo, 0.0)
        cz = -(u[0] * nx + u[1] * ny + u[2] * nz)      # cos zenith of arrival
        dn = -(u[0] * e_n[0] + u[1] * e_n[1] + u[2] * e_n[2])
        de = -(u[0] * e_e[0] + u[1] * e_e[1] + u[2] * e_e[2])
        az = np.degrees(np.arctan2(de, dn)) % 360.0
        return i0, cz, az, abs(cz)


class DualCapScorer:
    """3D tally and the correlated 1D reference, on the same neutrinos.

    For every neutrino two entries are made:

    **3D** -- the straight ray from the production point along the neutrino's
    own direction, intersected with the nested caps.

    **1D** -- the same neutrino energy delivered along the **primary's**
    direction: the ray from the injection point along ``u0``, whose ground
    crossing and local zenith/azimuth are computed once per shower in
    ``begin_shower``.  That is the collinear (1D) approximation evaluated on
    the same primary sample and the same hadronic events, so the numerator and
    denominator of ``R = Phi_3D / Phi_1D`` are strongly correlated.  (The
    cascade itself is still transported in 3D, so the secondaries' slant depth
    differs from a truly collinear cascade; ``sky.py --collinear`` re-runs the
    identical primary sample with a collinear cascade and measures that
    residual.)

    Two normalisations, both stored, because ``PHASE1_RESULTS.md`` sec. 8.2
    shows the whole "conservation excess" question is a choice of measure:

    ``s0 = sum w``            the ground-crossing (cos-weighted) measure:
                              ``int dOmega cos psi Phi = s0 / (A dE)``
    ``s1 = sum w / cos psi``  the specific intensity:
                              ``Phi = s1 / (A dOmega dE)``

    plus the sums of squares of both, so every number carries a statistical
    error, and all of it merges additively across multiprocessing shards.

    **Up-going neutrinos are not scored** (see ``NestedCaps.crossing``): a
    straight ray leaving the atmosphere upward never returns to the ground
    sphere, so the only way to populate ``cos Z < 0`` is a shower on the far
    side of the Earth whose neutrino crosses the ground sphere twice -- once at
    its own production side (the root we take) and once, up-going, under the
    detector.  That needs a second injection patch at the antipode and the
    *second* root; milestone 2b therefore reports down-going bins only, and
    ``cos Z < 0`` is empty by construction rather than by physics.

    Three auxiliary tallies (milestone 2b), all keyed off ``begin_shower``:

    ``cone``  a :class:`ConeScorer` -- the primary-to-neutrino angle of every
        produced neutrino, by species, energy, parent channel, primary zenith
        class and generation (``run_shower`` passes ``parent``/``gen`` because
        this class sets ``wants_parent``).
    ``hinj[view, species, band, czband, theta]``  the angular distance
        ``theta`` [deg, 1-deg bins] between the shower's **injection point**
        and the site, for every 3D (``view = 0``) and 1D (``view = 1``) cap
        hit, in ``CONE_BANDS`` x ``HINJ_CZ`` bands.  Its tail beyond the
        design patch is the injection-patch truncation, measured directly
        (gate G8) instead of from the difference of two noisy runs.
    ``shcounts``  showers opened / with >= 1 3D hit / with >= 1 1D hit / with
        either / with >= 1 3D hit in the East-West target region
        (``cos Z < EW_CZ_MAX``, ``EW_E_RANGE``).  The scored-shower fraction
        is what sets the cost of a per-shower geomagnetic cutoff back-trace
        (milestone 3a) if the cutoff is traced only for showers that score.
    """

    wants_parent = True
    HINJ_CZ = (0.0, 0.1, 0.2, 0.5, 1.0)
    HINJ_NTH = 60
    EW_CZ_MAX = 0.3
    EW_E_RANGE = (0.3, 2.0)

    def __init__(self, caps, e_bins, cz_bins=None, az_bins=None, cone=True):
        self.caps = caps
        self.e_bins = np.asarray(e_bins, float)
        self.cz_bins = np.linspace(-1, 1, 21) if cz_bins is None \
            else np.asarray(cz_bins, float)
        self.az_bins = np.linspace(0, 360, 13) if az_bins is None \
            else np.asarray(az_bins, float)
        self.ne = len(self.e_bins) - 1
        self.ncz = len(self.cz_bins) - 1
        self.naz = len(self.az_bins) - 1
        shape = (len(caps.thetas), len(SPECIES), self.ne, self.ncz, self.naz)
        self.shape = shape
        for nm in ("s0", "s0sq", "s1", "s1sq", "t0", "t0sq", "t1", "t1sq"):
            setattr(self, nm, np.zeros(shape))
        self.n_prim = 0.0
        self.n_nucleon = 0.0
        self.n_nu = 0.0
        self.n_hit3 = 0.0
        self.n_hit1 = 0.0
        self._sidx = {s: k for k, s in enumerate(SPECIES)}
        self._1d = None
        self._lge0 = float(np.log(self.e_bins[0]))
        self._dlge = float(np.log(self.e_bins[1] / self.e_bins[0]))
        # auxiliary tallies
        self.cone = ConeScorer() if cone else None
        self.hinj = np.zeros((2, len(SPECIES), len(CONE_BANDS) - 1,
                              len(self.HINJ_CZ) - 1, self.HINJ_NTH))
        self.shcounts = np.zeros(5)
        self._u0 = None
        self._zp = 0
        self._ith = -1
        self._open = False
        self._f3 = self._f1 = self._few = False

    # -- per-shower 1D geometry -------------------------------------------
    def begin_shower(self, r0, u0):
        """Precompute the collinear (1D) crossing of the primary's own ray,
        and open the per-shower bookkeeping of the auxiliary tallies."""
        if self._open:
            self.end_shower()
        self._open = True
        self._f3 = self._f1 = self._few = False
        self._u0 = (float(u0[0]), float(u0[1]), float(u0[2]))
        rn = math.sqrt(float(r0[0]) ** 2 + float(r0[1]) ** 2
                       + float(r0[2]) ** 2)
        ax = self.caps.axis
        czp = -(self._u0[0] * r0[0] + self._u0[1] * r0[1]
                + self._u0[2] * r0[2]) / rn
        self._zp = min(max(int(czp * (len(CONE_ZP) - 1)), 0), len(CONE_ZP) - 2)
        ca = (r0[0] * ax[0] + r0[1] * ax[1] + r0[2] * ax[2]) / rn
        th = math.degrees(math.acos(max(-1.0, min(1.0, ca))))
        self._ith = min(int(th), self.HINJ_NTH - 1)
        loc = self.caps.locate(r0, u0)
        if loc is None:
            self._1d = None
            return
        i0, cz, az, cosi = loc
        iz = min(int((cz + 1.0) / 2.0 * self.ncz), self.ncz - 1)  # cosZ=1: top bin
        ia = int(az / 360.0 * self.naz)
        if not (0 <= iz < self.ncz):
            self._1d = None
            return
        ia = min(max(ia, 0), self.naz - 1)
        self._1d = (i0, iz, ia, max(cosi, 1e-6), cz)

    def end_shower(self):
        """Close the shower opened by ``begin_shower`` (shower-level counts)."""
        if not self._open:
            return
        c = self.shcounts
        c[0] += 1.0
        c[1] += self._f3
        c[2] += self._f1
        c[3] += self._f3 or self._f1
        c[4] += self._few
        self._open = False
        self._u0 = None

    def _ebin(self, e):
        i = int((np.log(e) - self._lge0) / self._dlge)
        return i if 0 <= i < self.ne else -1

    def _hinj(self, view, k, e, cz):
        b = CONE_BANDS
        if not (b[0] <= e < b[-1]) or self._ith < 0:
            return None
        ib = 0
        while e >= b[ib + 1]:
            ib += 1
        hz = self.HINJ_CZ
        cz = min(cz, 1.0)              # 1 + 2e-16 for an exactly vertical ray
        if not (hz[0] <= cz <= hz[-1]):
            return None
        iz = 0
        while iz < len(hz) - 2 and cz >= hz[iz + 1]:
            iz += 1
        return (view, k, ib, iz, self._ith)

    def add(self, pdg, e, r, u, w, parent=0, gen=1):
        k = self._sidx.get(pdg)
        if k is None:
            return
        u0 = self._u0
        if self.cone is not None and u0 is not None and e >= CONE_EMOM[0]:
            self.cone.add(k, e, u0[0] * u[0] + u0[1] * u[1] + u0[2] * u[2],
                          w, parent, gen, self._zp)
        ie = self._ebin(e)
        if ie < 0:
            return
        self.n_nu += 1.0
        loc = self.caps.locate(r, u)
        if loc is not None:
            i0, _cz, az, cosi = loc
            iz = min(int((_cz + 1.0) / 2.0 * self.ncz), self.ncz - 1)
            if 0 <= iz < self.ncz:
                ia = min(max(int(az / 360.0 * self.naz), 0), self.naz - 1)
                self.n_hit3 += 1.0
                cosi = max(cosi, 1e-6)
                for ic in range(i0, len(self.caps.thetas)):
                    a = self.caps.areas[ic]
                    w0, w1 = w / a, w / (a * cosi)
                    self.s0[ic, k, ie, iz, ia] += w0
                    self.s0sq[ic, k, ie, iz, ia] += w0 * w0
                    self.s1[ic, k, ie, iz, ia] += w1
                    self.s1sq[ic, k, ie, iz, ia] += w1 * w1
                if self._open:
                    self._f3 = True
                    if (_cz < self.EW_CZ_MAX
                            and self.EW_E_RANGE[0] <= e < self.EW_E_RANGE[1]):
                        self._few = True
                    hi = self._hinj(0, k, e, _cz)
                    if hi is not None:
                        self.hinj[hi] += w
        if self._1d is not None:
            i0, iz, ia, cosi, cz1 = self._1d
            self.n_hit1 += 1.0
            for ic in range(i0, len(self.caps.thetas)):
                a = self.caps.areas[ic]
                w0, w1 = w / a, w / (a * cosi)
                self.t0[ic, k, ie, iz, ia] += w0
                self.t0sq[ic, k, ie, iz, ia] += w0 * w0
                self.t1[ic, k, ie, iz, ia] += w1
                self.t1sq[ic, k, ie, iz, ia] += w1 * w1
            if self._open:
                self._f1 = True
                hi = self._hinj(1, k, e, cz1)
                if hi is not None:
                    self.hinj[hi] += w

    # -- bookkeeping -------------------------------------------------------
    def merge(self, other):
        for nm in ("s0", "s0sq", "s1", "s1sq", "t0", "t0sq", "t1", "t1sq"):
            setattr(self, nm, getattr(self, nm) + getattr(other, nm))
        for nm in ("n_prim", "n_nucleon", "n_nu", "n_hit3", "n_hit1"):
            setattr(self, nm, getattr(self, nm) + getattr(other, nm))
        self.hinj = self.hinj + other.hinj
        self.shcounts = self.shcounts + other.shcounts
        if self.cone is not None and other.cone is not None:
            self.cone.merge(other.cone)
        elif other.cone is not None:
            self.cone = ConeScorer.from_dict(other.cone.to_dict())
        return self

    def to_dict(self):
        d = {nm: getattr(self, nm) for nm in
             ("s0", "s0sq", "s1", "s1sq", "t0", "t0sq", "t1", "t1sq")}
        d.update(e_bins=self.e_bins, cz_bins=self.cz_bins,
                 az_bins=self.az_bins,
                 cap_theta=np.array(self.caps.thetas),
                 cap_area=np.array(self.caps.areas),
                 counts=np.array([self.n_prim, self.n_nucleon, self.n_nu,
                                  self.n_hit3, self.n_hit1]),
                 site=np.array([self.caps.lat, self.caps.lon]),
                 hinj=self.hinj, hinj_cz=np.array(self.HINJ_CZ),
                 shcounts=self.shcounts)
        if self.cone is not None:
            d.update(self.cone.to_dict())
        return d

    @classmethod
    def from_dict(cls, d):
        caps = NestedCaps(float(d["site"][0]), float(d["site"][1]),
                          tuple(d["cap_theta"]))
        has_cone = "cone_h" in d
        o = cls(caps, d["e_bins"], d["cz_bins"], d["az_bins"], cone=has_cone)
        for nm in ("s0", "s0sq", "s1", "s1sq", "t0", "t0sq", "t1", "t1sq"):
            setattr(o, nm, np.array(d[nm]))
        (o.n_prim, o.n_nucleon, o.n_nu, o.n_hit3, o.n_hit1) = \
            [float(x) for x in d["counts"]]
        if "hinj" in d:
            o.hinj = np.array(d["hinj"])
        if "shcounts" in d:
            o.shcounts = np.array(d["shcounts"], float)
        if has_cone:
            o.cone = ConeScorer.from_dict(d)
        return o
