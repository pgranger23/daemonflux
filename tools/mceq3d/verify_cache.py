"""Verify that ``solve(use_cache=True)`` matches the default path and is fast.

Checks, on a small full-sky grid, that (i) the cached result reproduces the direct
computation to interpolation precision, (ii) the on-disk cache round-trips exactly
(cold vs warm identical), and (iii) a warm re-evaluation is milliseconds. The cache
holds the site-independent G_s and the per-site cutoff map (see
``MCEq3DFlux.solve(use_cache=...)``). Run::

    python verify_cache.py
"""

from __future__ import annotations

import datetime
import os
import shutil
import time

import numpy as np

import mceq3d_flux as m


def _max_rel_diff(a, b):
    d = 0.0
    for s in m.SPECIES:
        x, y = a["flux"][s], b["flux"][s]
        d = max(d, float(np.max(np.abs(x - y) / np.maximum(np.abs(y), 1e-300))))
    return d


def main():
    cd = "/tmp/flux_cache_verify"
    shutil.rmtree(cd, ignore_errors=True)
    eng = m.MCEq3DFlux(base_model="daemonflux", daemonflux_location="kamioka")
    cz = np.array([-0.55, 0.55])
    az = np.array([0.0, 270.0])
    date = datetime.datetime(2020, 1, 1)

    t0 = time.perf_counter()
    r_ref = eng.solve(36.43, 137.31, cz, az, date=date)
    t_ref = time.perf_counter() - t0

    t0 = time.perf_counter()
    r_cold = eng.solve(36.43, 137.31, cz, az, date=date, use_cache=True, cache_dir=cd)
    t_cold = time.perf_counter() - t0

    t0 = time.perf_counter()
    r_warm = eng.solve(36.43, 137.31, cz, az, date=date, use_cache=True, cache_dir=cd)
    t_warm = time.perf_counter() - t0

    print(
        f"timings: default={t_ref:.1f}s  cache-cold={t_cold:.1f}s  "
        f"cache-warm={t_warm * 1e3:.1f} ms  (speed-up {t_ref / t_warm:.0f}x)"
    )
    print(
        f"max rel diff default vs cached : {_max_rel_diff(r_ref, r_cold) * 100:.3f} %"
    )
    print(
        f"max rel diff cold vs warm      : {_max_rel_diff(r_cold, r_warm) * 100:.1e} %"
    )
    print("cache files:", sorted(os.listdir(cd)))

    assert _max_rel_diff(r_ref, r_cold) < 0.005, "cached path deviates >0.5%"
    assert _max_rel_diff(r_cold, r_warm) == 0.0, "cold/warm cache not identical"
    assert t_warm < 0.5, "warm solve not fast"
    print("OK: cached path matches default, round-trips exactly, and is fast.")


if __name__ == "__main__":
    main()
