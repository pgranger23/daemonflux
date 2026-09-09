"""Interaction backends for the mc3d cascade.

Two backends with the same interface:

``MCEqYieldBackend``
    Samples secondaries from MCEq's **own** inclusive yield matrices (the
    ``mceq_tables_*.npz`` written by ``build_tables.py``).  Inclusive, hence
    collinear-only in energy: it carries no p_T.  This is what makes the
    milestone-1 closure a clean test of the geometry/transport/decay machinery
    rather than of a generator.  Note that Honda's own 3D calculation uses
    exactly this kind of *inclusive* interaction code
    (astro-ph/0404457 sec. III: "the inclusive interaction code is only valid
    for the calculation of a time averaged quantity, such as the fluxes of
    atmospheric neutrinos"), so this is not a toy.

``ChromoBackend``
    Runs a real event generator (SIBYLL-2.3d by default) through ``chromo`` and
    returns the exclusive final state with full 3-momenta.  This is the backend
    the 3D physics needs, and it is gated against ``MCEqYieldBackend`` in
    collinear mode to expose the generator/database difference.

Interaction length: ``lambda_int(E) = <A> m_p / sigma_inel(E)`` with
``<A> = 14.6568`` -- byte-for-byte MCEq's ``ParticleManager.
inverse_interaction_length``.
"""

from __future__ import annotations

import os

import numpy as np

from constants import MASS

A_TARGET = 14.6568
M_TARGET_G = A_TARGET * 1.672621e-24

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TABLES = os.path.join(_HERE, "mceq_tables_SIBYLL23D.npz")


class MCEqYieldBackend:
    """Inclusive-yield sampler built on MCEq's tables."""

    def __init__(self, path=DEFAULT_TABLES, decays=False):
        d = np.load(path)
        self.e_grid = d["e_grid"]
        self.e_bins = d["e_bins"]
        self.e_widths = d["e_widths"]
        self.log_e = np.log(self.e_grid)
        self.cs = {}
        self.y = {}
        self.dec = {}
        for k in d.files:
            if k.startswith("cs_"):
                self.cs[int(k[3:])] = d[k]
            elif k.startswith("y_"):
                p, c = k[2:].rsplit("_", 1)
                self.y.setdefault(int(p), {})[int(c)] = np.asarray(d[k], float)
            elif k.startswith("d_") and decays:
                p, c = k[2:].rsplit("_", 1)
                self.dec.setdefault(int(p), {})[int(c)] = np.asarray(d[k], float)
        self._cache = {}
        self._dcache = {}

    # -- energetics --------------------------------------------------------
    def has_interaction(self, pdg):
        return int(pdg) in self.cs

    def sigma_inel(self, pdg, e_tot):
        cs = self.cs.get(int(pdg))
        if cs is None:
            return 0.0
        ek = max(e_tot - MASS.get(int(pdg), 0.0), self.e_grid[0])
        return float(np.exp(np.interp(np.log(ek),
                                      self.log_e, np.log(np.maximum(cs, 1e-40)))))

    def lambda_int(self, pdg, e_tot):
        s = self.sigma_inel(pdg, e_tot)
        return np.inf if s <= 0.0 else M_TARGET_G / s

    # -- sampling ----------------------------------------------------------
    def _prep(self, table):
        """(cdf over daughter bins, local spectral index) for every column."""
        key = id(table)
        if key in self._cache:
            return self._cache[key]
        N = table
        tot = N.sum(axis=0)
        cdf = np.cumsum(N, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            cdf = np.where(tot > 0, cdf / np.where(tot > 0, tot, 1.0), 0.0)
            f = N / self.e_widths[:, None]
            lf = np.log(np.maximum(f, 1e-300))
            g = np.zeros_like(f)
            dl = self.log_e[2:] - self.log_e[:-2]
            g[1:-1] = -(lf[2:] - lf[:-2]) / dl[:, None]
            g[0] = g[1]
            g[-1] = g[-2]
            g = np.clip(np.nan_to_num(g), -5.0, 12.0)
        self._cache[key] = (tot, cdf, g)
        return self._cache[key]

    def _sample_bins(self, rng, table, j, n):
        tot, cdf, g = self._prep(table)
        u = rng.random(n)
        idx = np.searchsorted(cdf[:, j], u)
        idx = np.clip(idx, 0, len(self.e_grid) - 1)
        lo = self.e_bins[idx]
        hi = self.e_bins[idx + 1]
        gg = g[idx, j]
        v = rng.random(n)
        # inverse CDF of E^-g on [lo, hi]
        a = 1.0 - gg
        near = np.abs(a) < 1e-6
        out = np.empty(n)
        out[near] = lo[near] * (hi[near] / lo[near]) ** v[near]
        na = ~near
        out[na] = (lo[na] ** a[na] + v[na] * (hi[na] ** a[na] - lo[na] ** a[na])) \
            ** (1.0 / a[na])
        return out

    def bin_of(self, e_tot, pdg=None):
        """Energy-grid bin of a particle of TOTAL energy ``e_tot``.

        MCEq's energy grid is **kinetic** energy (``etot_grid = e_grid + m``),
        while the cascade tracks total energy, so the conversion has to happen
        at every table lookup.  Getting this wrong shifts nucleons by ~1 GeV and
        pions by 0.14 GeV -- a large distortion of the sub-GeV spectrum.
        """
        ek = e_tot - (MASS.get(int(pdg), 0.0) if pdg is not None else 0.0)
        j = int(np.searchsorted(self.e_bins, max(ek, self.e_bins[0])) - 1)
        return int(np.clip(j, 0, len(self.e_grid) - 1))

    def interact(self, rng, pdg, e_tot, e_cut=None):
        """Inclusive secondaries of one interaction: ``[(pdg, E), ...]``."""
        ch = self.y.get(int(pdg))
        if ch is None:
            return []
        j = self.bin_of(e_tot, pdg)
        out = []
        for c, table in ch.items():
            tot, _, _ = self._prep(table)
            mu = tot[j]
            if mu <= 0.0:
                continue
            k = rng.poisson(mu)
            if k == 0:
                continue
            mc = MASS.get(int(c), 0.0)
            es = self._sample_bins(rng, table, j, k) + mc   # kinetic -> total
            if e_cut is not None:
                es = es[es >= e_cut.get(int(c), 0.0)]
            for e in es:
                out.append((int(c), float(e)))
        return out

    def decay_energies(self, rng, pdg, e_tot):
        """Daughter energies from MCEq's own decay matrices (collinear mode)."""
        ch = self.dec.get(int(pdg))
        if ch is None:
            return []
        j = self.bin_of(e_tot, pdg)
        out = []
        for c, table in ch.items():
            tot, _, _ = self._prep(table)
            mu = tot[j]
            if mu <= 0.0:
                continue
            k = rng.poisson(mu)
            if k == 0:
                continue
            mc = MASS.get(int(c), 0.0)
            for e in self._sample_bins(rng, table, j, k):
                out.append((int(c), float(e) + mc))
        return out


class ChromoBackend:
    """Real event generator via ``chromo`` (SIBYLL-2.3d / DPMJET-III / ...).

    ``model_lo`` is used below ``e_switch`` (SIBYLL-2.3d is not valid below
    ~10 GeV lab; DPMJET-III-19.3 is the same code family MCEq's shipped database
    splices in below 80 GeV, so it is the natural low-energy partner).
    """

    TARGET = (14, 7)      # nitrogen; air-average handled by the N/O mix below
    AIR_MIX = ((14, 7, 0.781 + 0.0093 * 0.0), (16, 8, 0.209))

    def __init__(self, model="Sibyll23d", model_lo="DpmjetIII193",
                 e_switch=80.0, seed=1, tables=None):
        self.model_name = model
        self.model_lo_name = model_lo
        self.e_switch = float(e_switch)
        self.seed = int(seed)
        self._models = {}
        # cross sections and lambda come from the MCEq tables for consistency
        self._cs = MCEqYieldBackend(tables or DEFAULT_TABLES)

    def has_interaction(self, pdg):
        return self._cs.has_interaction(pdg)

    def lambda_int(self, pdg, e_tot):
        return self._cs.lambda_int(pdg, e_tot)

    def _model(self, name, pdg, e_tot):
        import chromo
        from chromo.kinematics import FixedTarget, GeV
        key = name
        kin = FixedTarget(e_tot * GeV, int(pdg), self.TARGET)
        if key not in self._models:
            cls = getattr(chromo.models, name)
            self._models[key] = cls(kin, seed=self.seed)
        else:
            self._models[key].kinematics = kin
        return self._models[key]

    def interact(self, rng, pdg, e_tot, e_cut=None):
        """Exclusive final state: ``[(pdg, E, (px,py,pz)/|p|), ...]`` in the lab
        frame of the projectile (z along the projectile direction)."""
        name = self.model_name if e_tot >= self.e_switch else self.model_lo_name
        m = self._model(name, pdg, e_tot)
        for ev in m(1):
            fs = ev.final_state()
            out = []
            for pid, en, px, py, pz in zip(fs.pid, fs.en, fs.px, fs.py, fs.pz):
                out.append((int(pid), float(en),
                            np.array([float(px), float(py), float(pz)])))
            return out
        return []
