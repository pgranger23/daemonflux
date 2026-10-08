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
    """

    def __init__(self, caps, e_bins, cz_bins=None, az_bins=None):
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

    # -- per-shower 1D geometry -------------------------------------------
    def begin_shower(self, r0, u0):
        """Precompute the collinear (1D) crossing of the primary's own ray."""
        loc = self.caps.locate(r0, u0)
        if loc is None:
            self._1d = None
            return
        i0, cz, az, cosi = loc
        iz = int((cz + 1.0) / 2.0 * self.ncz)
        ia = int(az / 360.0 * self.naz)
        if not (0 <= iz < self.ncz):
            self._1d = None
            return
        ia = min(max(ia, 0), self.naz - 1)
        self._1d = (i0, iz, ia, max(cosi, 1e-6))

    def _ebin(self, e):
        i = int((np.log(e) - self._lge0) / self._dlge)
        return i if 0 <= i < self.ne else -1

    def add(self, pdg, e, r, u, w):
        k = self._sidx.get(pdg)
        if k is None:
            return
        ie = self._ebin(e)
        if ie < 0:
            return
        self.n_nu += 1.0
        loc = self.caps.locate(r, u)
        if loc is not None:
            i0, _cz, az, cosi = loc
            iz = int((_cz + 1.0) / 2.0 * self.ncz)
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
        if self._1d is not None:
            i0, iz, ia, cosi = self._1d
            self.n_hit1 += 1.0
            for ic in range(i0, len(self.caps.thetas)):
                a = self.caps.areas[ic]
                w0, w1 = w / a, w / (a * cosi)
                self.t0[ic, k, ie, iz, ia] += w0
                self.t0sq[ic, k, ie, iz, ia] += w0 * w0
                self.t1[ic, k, ie, iz, ia] += w1
                self.t1sq[ic, k, ie, iz, ia] += w1 * w1

    # -- bookkeeping -------------------------------------------------------
    def merge(self, other):
        for nm in ("s0", "s0sq", "s1", "s1sq", "t0", "t0sq", "t1", "t1sq"):
            setattr(self, nm, getattr(self, nm) + getattr(other, nm))
        for nm in ("n_prim", "n_nucleon", "n_nu", "n_hit3", "n_hit1"):
            setattr(self, nm, getattr(self, nm) + getattr(other, nm))
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
                 site=np.array([self.caps.lat, self.caps.lon]))
        return d

    @classmethod
    def from_dict(cls, d):
        caps = NestedCaps(float(d["site"][0]), float(d["site"][1]),
                          tuple(d["cap_theta"]))
        o = cls(caps, d["e_bins"], d["cz_bins"], d["az_bins"])
        for nm in ("s0", "s0sq", "s1", "s1sq", "t0", "t0sq", "t1", "t1sq"):
            setattr(o, nm, np.array(d[nm]))
        (o.n_prim, o.n_nucleon, o.n_nu, o.n_hit3, o.n_hit1) = \
            [float(x) for x in d["counts"]]
        return o
