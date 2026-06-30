"""Profile the full 3D directional engine against a bare MCEq 1D solve.

Reports: one-time setup, the cost of a single MCEq cascade solve (the atomic unit),
and a phase-by-phase breakdown of a full-sky ``solve()`` -- 1D base, geomagnetic
response G_s, cutoff back-tracing, assembly -- for both base models, expressed as a
multiple of one MCEq solve. Run::

    python profile_3d.py
"""

from __future__ import annotations

import time

import numpy as np


def t(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return time.perf_counter() - t0, out


def main():
    import mceq3d_flux as m
    import datetime

    date = datetime.datetime(2020, 1, 1)
    print("=" * 64)

    # --- one-time engine construction (MCEqRun init: load model, build matrices)
    t_init, eng = t(m.MCEq3DFlux, base_model="mceq")
    print(f"one-time setup  MCEq3DFlux()            : {t_init:6.1f} s")

    # --- atomic unit: a single MCEq cascade solve at one zenith
    def one_solve():
        eng.mceq.set_theta_deg(0.0)
        eng.mceq._phi0[:] = eng._phi0_std
        eng.mceq.solve()
        return eng.mceq.get_solution("total_numu", 0)

    ts = [t(one_solve)[0] for _ in range(3)]
    t_unit = float(np.median(ts))
    print(f"atomic unit     1 MCEq solve (1 zenith) : {t_unit:6.3f} s  (median of 3)")
    print("=" * 64)

    # --- grids
    cz = np.array([-0.95, -0.55, -0.15, 0.15, 0.55, 0.95])
    az = np.array([0, 45, 90, 135, 180, 225, 270, 315.0])
    ndir = len(cz) * len(az)
    print(f"grid: {len(cz)} zeniths x {len(az)} azimuths = {ndir} directions\n")

    # --- phase breakdown (MCEq base)
    t_base, _ = t(eng.base, cz)
    rc_grid = np.linspace(2.0, 14.0, 12)
    t_geo, _ = t(eng.geomag_response, rc_grid)
    t_cut, _ = t(eng.cutoff_grid, 36.43, 137.31, cz, az, date)

    def assemble():
        return eng.solve(36.43, 137.31, cz, az, date=date)

    t_full, _ = t(assemble)

    print("PHASE breakdown of a full solve() [MCEq base]:")
    print(f"  base() {len(cz)} zenith cascades        : {t_base:6.2f} s "
          f"= {t_base / t_unit:4.1f} x unit")
    print(f"  geomag_response() ({len(rc_grid)} R_c solves)  : {t_geo:6.2f} s "
          f"= {t_geo / t_unit:4.1f} x unit")
    print(f"  cutoff_grid() back-tracing ({ndir} dir): {t_cut:6.2f} s "
          f"= {t_cut / t_unit:4.1f} x unit  <-- dominant")
    print(f"  ---- full solve() end-to-end          : {t_full:6.2f} s "
          f"= {t_full / t_unit:4.1f} x unit")
    print()

    # --- daemonflux base: base() is spline eval, not cascades
    t_initdf, engdf = t(m.MCEq3DFlux, base_model="daemonflux",
                        daemonflux_location="kamioka")
    t_basedf, _ = t(engdf.base, cz)
    print("daemonflux base:")
    print(f"  base() {len(cz)} zeniths (spline eval)   : {t_basedf:6.3f} s "
          f"= {t_basedf / t_unit:4.2f} x unit (vs {t_base / t_unit:.1f}x for MCEq base)")
    print()

    # --- cached cutoff: amortised cost on re-use (one-time map build excluded)
    import geomag_backtrace as gb

    tc, f = t(gb.cached_cutoff_source, 36.43, 137.31, date, 3, 5)  # warm load
    zz = np.repeat(np.linspace(0, 90, 37), 49)
    aa = np.tile(np.linspace(0, 360, 49), 37)
    tcall, _ = t(lambda: f(zz, aa))
    print("cached cutoff (one-time map build amortised away):")
    print(f"  warm load + {len(zz)} interpolations  : "
          f"{(tc + tcall) * 1e3:6.1f} ms  (vs {t_cut:.0f} s back-tracing live)")
    print("=" * 64)
    print("Summary:")
    print(f"  Full 3D directional flux over {ndir} directions costs "
          f"~{t_full / t_unit:.0f}x a single MCEq 1D solve;")
    print("  the cost is dominated by geomagnetic cutoff back-tracing, which is a "
          "one-time")
    print("  per-site precompute -- cached, re-evaluation is ~milliseconds.")


if __name__ == "__main__":
    main()
