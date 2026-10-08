"""Milestone 2b: the 3D production-geometry run (no field, no cutoff).

What this driver does
---------------------
1. Samples primaries on an **injection patch** -- a cap of angular radius
   ``theta_inj = theta_D + 30 deg`` on the 100 km injection sphere around the
   site (``geometry.sample_injection_patch``), positions uniform per unit area
   and directions from the inward Lambert law.
2. Samples the energy per nucleon **stratified in ln E** (``STRATA``), one
   multiprocessing task per (species, stratum) so the per-stratum yield and
   cost come out of the run for free and the allocation can be re-tuned
   afterwards without re-deriving anything.
3. Applies **superposition**: a nucleus of mass ``A``, charge ``Z`` at energy
   per nucleon ``E`` is represented by one nucleon drawn ``p`` with
   probability ``Z/A`` and ``n`` otherwise, carrying the full nucleon-intensity
   weight; the nucleus rigidity ``A/Z * p_nucleon`` is recorded for the
   milestone-3 cutoff even though no cutoff is applied here.
4. Runs the full 3D cascade (``shower.run_shower`` with ``collinear=False``,
   ``bfield=None``) and scores every neutrino into the nested caps *and* into
   the correlated collinear 1D reference (``scoring.DualCapScorer``).

``--collinear`` re-runs the *identical* primary sample with a collinear
cascade.  In that mode every secondary stays on the primary's line, so the 3D
tally and the 1D tally coincide and the run is a genuine 1D calculation on the
same primaries -- which is how the residual of the in-run correlated 1D
reference (whose cascade is still transported in 3D) is measured.

Deliberately **not** here (milestone 3): the geomagnetic cutoff acceptance at
the injection point, and the azimuthal reuse with the rotated muon bend.  This
run measures the pure 3D production geometry.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

import geometry as geo
import primaries as prim
from constants import M_N, M_P, R_EARTH_CM
from scoring import SPECIES, DualCapScorer, NestedCaps
from shower import Config, run_shower

H_INJ_CM = 100.0e5
R_INJ_CM = R_EARTH_CM + H_INJ_CM

SITES = {"kamioka": (36.4267, 137.31), "southpole": (-90.0, 0.0),
         "gran_sasso": (42.4542, 13.5755)}

# Honda's grid: 20 E-bins/decade from 0.1 GeV, 20 cos Z bins of 0.1,
# 12 azimuth bins of 30 deg (PHASE2_PLAN.md sec. 5.1)
E_BINS = np.logspace(-1, 2, 61)
CZ_BINS = np.linspace(-1.0, 1.0, 21)
AZ_BINS = np.linspace(0.0, 360.0, 13)

# Stratified ln E: 8 half-decade-and-a-bit strata over [1, 1e4] GeV/nucleon.
#
# The allocation matters more than anything else in this file.  A log-uniform
# proposal gives every sample in stratum k a weight ~ J(E) E ~ E^-1.7, so a
# flat allocation lets the lowest stratum dominate the *weight* while carrying
# only 1/8 of the *samples*: the pilot measured an effective sample size
# N_eff = (sum w)^2 / sum w^2 of ~1 per (E, cos Z) bin from 4000 primaries,
# i.e. essentially every bin held by a single shower.  Allocating
# n_k ~ E_k^-0.8 splits the difference between Neyman's variance-optimal
# n_k ~ E_k^-1.7 (which would starve E_nu > 3 GeV completely) and a flat
# allocation.  ``shard_meta`` in the output records the samples and the CPU
# time per stratum so milestone 4 can re-tune this from the measured
# per-stratum contribution instead of from an exponent.
STRATA = np.logspace(0.0, 4.0, 9)
_SC = np.sqrt(STRATA[1:] * STRATA[:-1])
STRATA_FRAC = _SC ** -0.8
STRATA_FRAC = STRATA_FRAC / STRATA_FRAC.sum()

# (name, A, Z, crflux group id) -- the plan's five groups, subset by --species
GROUPS = {n: (n, a, z, g) for n, a, z, g in prim.SPECIES}
# "N": the all-nucleon group used by the GSF nucleon-level model (see
# primaries.GSFNucleons -- crflux's GSF has no per-nucleus splines, and with no
# cutoff a species decomposition is redundant under superposition anyway)
GROUPS["N"] = ("N", 1, 1, 0)
# sample allocation between species; the weight restores the true intensity,
# this only balances the variance (p carries ~74% of the nucleons, He ~18%)
GROUP_FRAC = {"N": 1.0, "p": 0.80, "He": 0.20, "CNO": 0.06, "MgSi": 0.03,
              "Fe": 0.02}

_CTX = {}


def _init(cache, n_pool, collinear, e_nu_min, lat, lon, caps, seed0, model):
    import importlib.util  # noqa: F401
    from interactions import ChromoBackend
    _CTX["backend"] = ChromoBackend(cache_dir=cache, n_pool=n_pool,
                                    xs="mceq", readonly=True)
    _CTX["cfg"] = Config(collinear=collinear, bfield=None, energy_loss=True,
                         e_nu_min=e_nu_min, decay_mode="kinematic",
                         polarisation=True)
    _CTX["caps"] = NestedCaps(lat, lon, caps)
    _CTX["model"] = prim.primary_model(model)


def _shard(task):
    """One (species, stratum, chunk) block of primaries."""
    gname, k, n, theta_inj, seed = task
    _name, a_mass, z, gid = GROUPS[gname]
    rng = np.random.default_rng(seed)
    caps, cfg, backend = _CTX["caps"], _CTX["cfg"], _CTX["backend"]
    sc = DualCapScorer(caps, E_BINS, CZ_BINS, AZ_BINS)
    e1, e2 = float(STRATA[k]), float(STRATA[k + 1])
    ln_range = np.log(e2 / e1)
    e_nuc = np.exp(rng.uniform(np.log(e1), np.log(e2), n))
    model = _CTX["model"]
    if getattr(model, "nucleon_level", False):
        j_nuc, p_frac = model.nucleon_flux(e_nuc)
        is_p = rng.random(n) < p_frac
    else:
        j_nuc = prim.nucleon_intensity(model, gid, e_nuc, a_mass)
        is_p = rng.random(n) < (z / a_mass)
    a_patch = geo.patch_area_cm2(R_INJ_CM, theta_inj)
    w = j_nuc * e_nuc * ln_range / n * np.pi * a_patch
    axis = np.array(caps.axis, float)
    r0, u0 = geo.sample_injection_patch(rng, R_INJ_CM, axis, theta_inj, n)
    t0 = time.time()
    for i in range(n):
        pdg = 2212 if is_p[i] else 2112
        e_tot = float(e_nuc[i]) + (M_P if is_p[i] else M_N)
        sc.begin_shower(r0[i], u0[i])
        run_shower(rng, pdg, e_tot, r0[i], u0[i], backend, cfg, sc, float(w[i]))
    sc.n_prim = float(n)
    sc.n_nucleon = float(n)
    d = sc.to_dict()
    d["meta"] = np.array([n, k, GROUP_FRAC.get(gname, 0.0), time.time() - t0,
                          a_patch, theta_inj])
    d["group"] = np.array([gname])
    return d


def build_tasks(nshower, groups, theta_inj, chunk, seed0, nstrata=None):
    """One task per (species, stratum, chunk).

    The chunk size is scaled **down** with the stratum energy as
    ``chunk * E_0 / E_k`` (floor 100), so every task costs roughly the same
    wall time.  Without that the top stratum is a single task of ~10^3 showers
    at 10^4 GeV, each ~100x the cost of a 2 GeV shower, and it holds the whole
    pool open for over an hour after every other task has finished -- measured
    the hard way.
    """
    tasks = []
    ns = len(STRATA) - 1 if nstrata is None else int(nstrata)
    fr = np.array([GROUP_FRAC[g] for g in groups], float)
    fr = fr / fr.sum()
    sf = STRATA_FRAC[:ns] / STRATA_FRAC[:ns].sum()
    for gi, g in enumerate(groups):
        for k in range(ns):
            ck = max(100, int(chunk * _SC[0] / _SC[k]))
            n_tot = int(round(nshower * fr[gi] * sf[k]))
            left, c = n_tot, 0
            while left > 0:
                m = min(ck, left)
                tasks.append((g, k, int(m), float(theta_inj),
                              seed0 + 1000003 * gi + 10007 * k + 13 * c))
                left -= m
                c += 1
    return tasks


def run(nshower, nproc, groups, theta_inj, caps, cache, n_pool=2000,
        collinear=False, e_nu_min=0.1, site="kamioka", chunk=2000,
        seed0=20260910, model="GSF", outdir=".", tag="pilot", nstrata=None):
    import multiprocessing as mp
    lat, lon = SITES[site]
    os.makedirs(outdir, exist_ok=True)
    tasks = build_tasks(nshower, groups, theta_inj, chunk, seed0, nstrata)
    print(f"{len(tasks)} tasks, {nshower} primaries, theta_inj={theta_inj} deg, "
          f"caps={caps}, collinear={collinear}", flush=True)
    acc, probes, meta = {}, [], []
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(nproc, initializer=_init,
                  initargs=(cache, n_pool, collinear, e_nu_min, lat, lon,
                            caps, seed0, model)) as pool:
        done = 0
        for d in pool.imap_unordered(_shard, tasks, chunksize=1):
            g = str(d["group"][0])
            k = int(d["meta"][1])
            probes.append(_probe(d))
            meta.append([k, d["meta"][0], d["meta"][3], _gidx(groups, g)])
            key = (g, k)
            s = DualCapScorer.from_dict(d)
            acc[key] = s if key not in acc else acc[key].merge(s)
            done += 1
            if done % 10 == 0 or done == len(tasks):
                dt = time.time() - t0
                nsh = sum(x[1] for x in meta)
                print(f"  {done}/{len(tasks)} tasks, {nsh:.0f} showers, "
                      f"{dt:.0f} s, {nsh/dt:.1f} showers/s", flush=True)
    tot = None
    for key, s in sorted(acc.items()):
        fp = os.path.join(outdir, f"{tag}_{key[0]}_s{key[1]}.npz")
        np.savez(fp, **s.to_dict())
        tot = s if tot is None else tot.merge(s)
    out = tot.to_dict()
    out["probes"] = np.array(probes)
    out["shard_meta"] = np.array(meta, float)
    out["strata"] = STRATA
    out["wall_s"] = np.array([time.time() - t0])
    out["nproc"] = np.array([nproc])
    fp = os.path.join(outdir, f"{tag}_total.npz")
    np.savez(fp, **out)
    print("wrote", fp, f"({time.time()-t0:.0f} s)")
    return fp


def _gidx(groups, g):
    return float(list(groups).index(g))


# probe bins for the 3D/1D correlation and the cost measurement:
# (species, E [GeV], cos Z band) -- azimuth-summed, largest cap
PROBES = [(14, 0.3, 0.0, 0.1), (14, 0.5, 0.0, 0.1), (14, 1.0, 0.0, 0.1),
          (12, 0.3, 0.0, 0.1), (12, 0.5, 0.0, 0.1), (12, 1.0, 0.0, 0.1),
          (14, 0.3, 0.9, 1.0), (14, 1.0, 0.9, 1.0),
          (12, 0.3, 0.9, 1.0), (12, 1.0, 0.9, 1.0)]


def probe_index(e_bins, cz_bins):
    out = []
    for s, e, c0, c1 in PROBES:
        ie = int(np.searchsorted(e_bins, e) - 1)
        i0 = int(np.searchsorted(cz_bins, c0 + 1e-9) - 1)
        i1 = int(np.searchsorted(cz_bins, c1 - 1e-9) - 1)
        out.append((SPECIES.index(s), ie, i0, i1))
    return out


def _probe(d):
    """(s1, t1, s1sq, t1sq) summed over azimuth in each probe bin, last cap."""
    idx = probe_index(d["e_bins"], d["cz_bins"])
    v = []
    for k, ie, i0, i1 in idx:
        sl = (-1, k, ie, slice(i0, i1 + 1), slice(None))
        v.extend([d["s1"][sl].sum(), d["t1"][sl].sum(),
                  d["s1sq"][sl].sum(), d["t1sq"][sl].sum()])
    return v


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--nshower", type=int, default=100000)
    ap.add_argument("--nproc", type=int, default=40)
    ap.add_argument("--groups", nargs="+", default=None,
                    help='default: ["N"] for GSF, ["p","He"] otherwise')
    ap.add_argument("--theta-inj", type=float, default=None,
                    help="default: max(cap) + 30 deg")
    ap.add_argument("--caps", type=float, nargs="+",
                    default=[2.5, 5.0, 7.5, 10.0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--n-pool", type=int, default=2000)
    ap.add_argument("--collinear", action="store_true")
    ap.add_argument("--e-nu-min", type=float, default=0.1)
    ap.add_argument("--site", default="kamioka")
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--model", default="GSF")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--nstrata", type=int, default=None,
                    help="use only the first N ln E strata (truncates the "
                         "primary energy range; fractions are renormalised)")
    a = ap.parse_args(argv)
    th = a.theta_inj if a.theta_inj is not None else max(a.caps) + 30.0
    groups = a.groups
    if groups is None:
        groups = ["N"] if a.model.upper() == "GSF" else ["p", "He"]
    run(a.nshower, a.nproc, groups, th, tuple(a.caps), a.cache,
        n_pool=a.n_pool, collinear=a.collinear, e_nu_min=a.e_nu_min,
        site=a.site, chunk=a.chunk, seed0=a.seed, model=a.model,
        outdir=a.outdir, tag=a.tag, nstrata=a.nstrata)


if __name__ == "__main__":
    main()
