"""Fill the ``pool.EventPool`` disk cache in parallel.

Every (projectile, energy-bin) pool is a self-contained npz written by an
atomic rename, so the build is resumable: re-running skips what exists and
regenerates only the missing files.  Workers of the physics runs then open the
cache ``readonly=True`` and never pay generation time.

    python prebuild_pools.py --cache <dir> --emax 1e4 --npool 2000 --nproc 40
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from pool import GRID, POOL_PROJECTILES, EventPool, generate_pool, _Generators


def _keys(emax, species):
    out = []
    for i in range(GRID.n):
        if GRID.centres[i] > emax:
            break
        for s in species:
            out.append((s, i))
    return out


def _one(task):
    import importlib.util  # noqa: F401
    pdg, i, cache, n_pool, seed = task
    p = EventPool(cache, n_pool=n_pool, seed=seed)
    fp = p.path(pdg, i)
    if os.path.exists(fp):
        return (pdg, i, 0.0, "cached")
    t0 = time.time()
    try:
        p._load_or_make(pdg, i)
    except Exception as exc:                      # a generator can refuse
        return (pdg, i, time.time() - t0, f"FAIL {type(exc).__name__}: {exc}")
    return (pdg, i, time.time() - t0, "built")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--emax", type=float, default=1.0e4)
    ap.add_argument("--npool", type=int, default=2000)
    ap.add_argument("--nproc", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--species", type=int, nargs="*", default=None)
    a = ap.parse_args(argv)
    sp = a.species or list(POOL_PROJECTILES)
    tasks = [(s, i, a.cache, a.npool, a.seed) for s, i in _keys(a.emax, sp)]
    print(f"{len(tasks)} pools, emax={a.emax:g}, npool={a.npool}", flush=True)
    import multiprocessing as mp
    t0 = time.time()
    nb = nf = 0
    with mp.get_context("fork").Pool(a.nproc) as pool:
        for pdg, i, dt, st in pool.imap_unordered(_one, tasks, chunksize=1):
            if st.startswith("FAIL"):
                nf += 1
                print(f"  {pdg:+6d} bin {i:3d} E0={GRID.centres[i]:9.2f} {st}",
                      flush=True)
            elif st == "built":
                nb += 1
                if nb % 25 == 0:
                    print(f"  built {nb}/{len(tasks)} ({time.time()-t0:.0f} s)",
                          flush=True)
    print(f"done: {nb} built, {nf} failed, {time.time()-t0:.0f} s")
    # tabulate the interaction lengths too, so physics workers never have to
    # construct a chromo generator (~1 s and a few hundred MB each)
    p = EventPool(a.cache, n_pool=a.npool, seed=a.seed)
    p.build_lambda_table(species=sp, emax=a.emax)
    print("wrote", p.lam_path)


if __name__ == "__main__":
    main()
