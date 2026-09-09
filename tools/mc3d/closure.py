"""Milestone-1 closure gate: collinear, B = 0, vertical column, vs MCEq 1D.

The gate.  Inject a single proton of fixed energy ``E_p`` straight down at the
top of the CORSIKA atmosphere and tally every neutrino produced, ``dN/dE_nu``
per primary.  MCEq run with ``set_single_primary_particle(E_p, pdg_id=2212)``
at ``theta = 0`` and evaluated at the ground gives exactly the same quantity for
the same hadronic model.  In collinear mode every neutrino goes straight down,
so "produced" and "crossing the ground" are the same set and no geometry enters
the comparison other than the atmosphere itself.

Ladder (each rung isolates one thing):

A1  MC(MCEq yields + MCEq decay tables)   vs MCEq  -> geometry / transport /
                                                     interaction-decay
                                                     competition / dE/dx
A2  MC(MCEq yields + full decay kinematics) vs MCEq -> the decay module
B   MC(chromo SIBYLL-2.3d + kinematics)     vs MCEq -> generator difference

Usage::

    python closure.py --energies 20 100 --nshower 20000 --nproc 40 \
        --mode kinematic --out closure_kin.npz
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

import atmosphere as atm
from constants import R_EARTH_CM
from interactions import MCEqYieldBackend, DEFAULT_TABLES
from scoring import SPECIES, YieldScorer
from shower import Config, run_shower

E_BINS = np.logspace(-1, 2, 31)          # 0.1 - 100 GeV, 10 bins/decade

_CTX = {}


def mceq_initial_condition(e_kin, e_grid, e_bins, e_widths):
    """MCEq's own three-bin representation of a mono-energetic primary.

    ``set_single_primary_particle`` does not put the proton in one bin: it
    spreads it over ``cenbin-1, cenbin, cenbin+1`` with weights that match the
    first three energy moments -- and the third weight is **negative**
    (e.g. +0.4184 / +0.6692 / -0.0876 at 100 GeV).  Injecting the MC at a single
    bin centre instead compares two different primaries and shows up as a
    growing high-energy discrepancy.  Reproducing the same three (signed)
    weights makes the closure a test of the *transport* alone.

    Returns ``[(E_kin_bin_centre, weight), ...]``.
    """
    from scipy.linalg import solve
    cen = int(np.argwhere(e_kin < e_bins)[0][0] - 1)
    sl = slice(cen - 1, cen + 2)
    emat = np.vstack((e_widths[sl],
                      e_widths[sl] * e_grid[sl],
                      e_widths[sl] * e_grid[sl] ** 2))
    phi = solve(emat, np.array([1.0, e_kin, e_kin ** 2]))
    return [(float(e_grid[cen - 1 + k]), float(phi[k] * e_widths[cen - 1 + k]))
            for k in range(3)]


def _init(tables, mode, e_nu_min, pol):
    _CTX["backend"] = MCEqYieldBackend(tables, decays=(mode == "mceq"))
    _CTX["cfg"] = Config(collinear=True, bfield=None, energy_loss=True,
                         e_nu_min=e_nu_min, decay_mode=mode, polarisation=pol)
    _CTX["mode"] = mode


def _shard(task):
    e_p, n, seed = task
    rng = np.random.default_rng(seed)
    backend, cfg = _CTX["backend"], _CTX["cfg"]
    sc = YieldScorer(E_BINS)
    r0 = np.array([0.0, 0.0, R_EARTH_CM + atm.H_TOP_CM])
    u0 = np.array([0.0, 0.0, -1.0])
    from constants import M_P
    ic = mceq_initial_condition(float(e_p), backend.e_grid, backend.e_bins,
                                backend.e_widths)
    wsum = sum(abs(w) for _, w in ic)
    for e_kin, w in ic:
        nk = max(1, int(round(n * abs(w) / wsum)))
        ww = w / nk
        e_tot = e_kin + M_P
        if _CTX["mode"] == "mceq":
            _run_mceq_decays(rng, e_tot, nk, backend, cfg, sc, r0, u0, ww)
        else:
            for _ in range(nk):
                run_shower(rng, 2212, e_tot, r0, u0, backend, cfg, sc, ww)
    sc.n_prim = 1.0
    return sc.to_dict()


def _run_mceq_decays(rng, e_p, n, backend, cfg, sc, r0, u0, weight=1.0):
    """Rung A1: decays taken from MCEq's own decay matrices (collinear)."""
    from constants import NEUTRINOS
    from shower import transport_hadron, transport_muon
    for _ in range(n):
        stack = [(2212, float(e_p), r0.copy(), u0.copy(), float(weight))]
        while stack:
            pdg, e, r, u, w = stack.pop()
            if pdg in NEUTRINOS:
                if e >= cfg.e_nu_min:
                    sc.add(pdg, e, r, u, w)
                continue
            thr = cfg.thr.get(pdg)
            if thr is None or e < thr:
                continue
            if abs(pdg) == 13:
                kind, r, u, e = transport_muon(rng, pdg, e, r, u, cfg)
                if kind in ("ground", "escape"):
                    continue
                for c, ec in backend.decay_energies(rng, pdg, e):
                    stack.append((c, ec, r, u, w))
                continue
            kind, r_new = transport_hadron(rng, pdg, e, r, u, backend)
            if kind in ("ground", "escape"):
                continue
            src = (backend.decay_energies(rng, pdg, e) if kind == "decay"
                   else backend.interact(rng, pdg, e))
            for c, ec in src:
                stack.append((c, ec, r_new, u, w))


def run(energies, nshower, nproc, mode="kinematic", tables=DEFAULT_TABLES,
        e_nu_min=0.1, pol=True, shard=250):
    import multiprocessing as mp
    out = {}
    ctx = mp.get_context("fork")
    for e_p in energies:
        tasks = []
        left = nshower
        k = 0
        while left > 0:
            m = min(shard, left)
            tasks.append((float(e_p), int(m), 12345 + 1000 * k + int(e_p)))
            left -= m
            k += 1
        t0 = time.time()
        with ctx.Pool(nproc, initializer=_init,
                      initargs=(tables, mode, e_nu_min, pol)) as pool:
            acc = None
            nsh = 0
            for d in pool.imap_unordered(_shard, tasks, chunksize=1):
                s = YieldScorer.from_dict(d)
                acc = s if acc is None else acc.merge(s)
                nsh += 1
            acc.n_prim = float(nsh)   # each shard carries total weight 1
        dt = time.time() - t0
        print(f"E_p = {e_p:g} GeV: {nshower} showers in {dt:.1f} s "
              f"({nshower / dt:.0f} showers/s on {nproc} cores)")
        out[float(e_p)] = acc
    return out


# ---------------------------------------------------------------------------
def mceq_reference(e_p, model="SIBYLL23D", e_min=0.05, theta_deg=0.0,
                   helicity=True):
    """dN/dE per primary proton at the ground from MCEq's 1D solution.

    ``helicity=False`` switches off MCEq's helicity-resolved muon states -- the
    matching reference for rung A1, whose decay tables are helicity-summed
    (see ``build_tables.py``).
    """
    import importlib.util  # noqa: F401

    import MCEq.config as cfg
    cfg.e_min = e_min
    cfg.debug_level = 0
    cfg.muon_helicity_dependence = bool(helicity)
    import crflux.models as crf
    from MCEq.core import MCEqRun
    mc = MCEqRun(interaction_model=model,
                 primary_model=(crf.HillasGaisser2012, "H3a"),
                 theta_deg=theta_deg)
    mc.set_single_primary_particle(float(e_p), pdg_id=2212)
    mc.solve()
    eg = mc.e_grid
    res = {}
    for s, nm in ((14, "total_numu"), (-14, "total_antinumu"),
                  (12, "total_nue"), (-12, "total_antinue")):
        res[s] = mc.get_solution(nm, mag=0)
    return eg, res


def rebin_reference(eg, ref, e_bins):
    """Integrate the MCEq dN/dE onto ``e_bins`` and return dN/dE per bin."""
    out = {}
    lg = np.log(eg)
    for s, y in ref.items():
        f = np.maximum(y, 1e-300)
        vals = []
        for a, b in zip(e_bins[:-1], e_bins[1:]):
            xs = np.exp(np.linspace(np.log(a), np.log(b), 33))
            fs = np.exp(np.interp(np.log(xs), lg, np.log(f)))
            vals.append(np.trapezoid(fs, xs) / (b - a))
        out[s] = np.array(vals)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--energies", type=float, nargs="+", default=[20.0, 100.0])
    ap.add_argument("--nshower", type=int, default=5000)
    ap.add_argument("--nproc", type=int, default=os.cpu_count() // 2)
    ap.add_argument("--mode", default="kinematic", choices=["kinematic", "mceq"])
    ap.add_argument("--tables", default=DEFAULT_TABLES)
    ap.add_argument("--e-nu-min", type=float, default=0.1)
    ap.add_argument("--no-pol", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    res = run(a.energies, a.nshower, a.nproc, a.mode, a.tables,
              a.e_nu_min, not a.no_pol)
    store = {}
    for e_p, sc in res.items():
        eg, ref = mceq_reference(e_p, helicity=(a.mode != "mceq"))
        rb = rebin_reference(eg, ref, E_BINS)
        ec = np.sqrt(E_BINS[1:] * E_BINS[:-1])
        print(f"\n=== E_p = {e_p:g} GeV, mode={a.mode}, "
              f"{a.nshower} showers ===")
        for s in SPECIES:
            y, err = sc.dnde(s)
            print(f"  {s:+3d}")
            for i in range(0, len(ec)):
                if rb[s][i] <= 0 or (y[i] == 0 and rb[s][i] < 1e-12):
                    continue
                r = y[i] / rb[s][i] if rb[s][i] > 0 else np.nan
                dr = err[i] / rb[s][i] if rb[s][i] > 0 else np.nan
                print(f"    E={ec[i]:8.3f}  MC={y[i]:.4e}+-{err[i]:.1e}  "
                      f"MCEq={rb[s][i]:.4e}  ratio={r:.4f}+-{dr:.4f}")
            store[f"mc_{e_p}_{s}"] = y
            store[f"err_{e_p}_{s}"] = err
            store[f"ref_{e_p}_{s}"] = rb[s]
    if a.out:
        store["e_bins"] = E_BINS
        np.savez(a.out, **store)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
