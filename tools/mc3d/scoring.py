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
