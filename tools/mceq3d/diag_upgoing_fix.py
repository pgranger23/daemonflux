"""Diagnosis of the engine's up-going near-horizon bin (INTEGRATION.md 11).

    modes   solve the DOWN-going horizon rows (cos Z 0.05, 0.15, 0.25 x 12 az) three ways and keep F = flux/base:
            full   joint cone + production-point anchor (the delivered down-going path, = 'hor' of engine_F_fine)
            cone   joint cone, detector-anchored cutoff            (cutoff_anchor='detector')
            single E_off x single detector cutoff, no cone          (cone_cutoff=False, joint_cone=False)
            -- the 'single' mode is what the engine does for every cos Z < 0 direction
            -> $W/runs/integ/out/updiag_modes.npz
"""
import importlib.util  # noqa: F401
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import diag_integ as D  # noqa: E402


def cmd_modes(which="hor"):
    from mceq3d_flux import MCEq3DFlux

    warnings.simplefilter("ignore")
    eng = MCEq3DFlux(base_model="mceq", primary=("GlobalSplineFitBeta", None))
    out = {"e": np.array(eng.e), "cz": D.CZ_HOR, "az": D.AZ12}
    modes = {
        "cone": dict(cutoff_anchor="detector"),
        "single": dict(cone_cutoff=False, joint_cone=False, cutoff_anchor="detector"),
    }
    for grid, cz, az in (("", D.CZ_HOR, D.AZ12), ("dn6_", D.CZ_DN6, D.AZ6)):
        if grid and which == "hor":
            continue
        if not grid and which == "dn6":
            continue
        base = eng.base(cz)
        out[f"{grid}cz"], out[f"{grid}az"] = cz, az
        for name, kw in modes.items():
            t = time.time()
            r = eng.solve(D.LAT, D.LON, cz, az, date=D.DATE, use_cache=True,
                          cache_dir=D.CACHE_REQ, muon_bending=False, n_jobs=1, **kw)
            for sp in D.SPECIES:
                out[f"{grid}{name}_{sp}"] = r["flux"][sp] / base[sp][:, None, :]
            print(grid, name, "done", f"{time.time() - t:.0f}s", flush=True)
    np.savez(os.path.join(D.RUN, "out", f"updiag_modes{'_dn6' if which == 'dn6' else ''}.npz"), **out)




def _obs_table(F, e, ax, cz_rows, label, names):
    """observables (a1/a0, dphi, fN, fS, N/S, W/E) of F[row, az, E] at E = 0.3, 0.5, 1 GeV"""
    import analyse_m3b as M
    import diag_integ as D2

    out = {}
    for sp in D2.SPECIES:
        for ir, c in enumerate(cz_rows):
            for E in (0.3, 0.5, 1.0):
                f = np.array([float(np.exp(np.interp(np.log(E), np.log(e),
                                                      np.log(np.maximum(F[sp][ir, j], 1e-300)))))
                              for j in range(12)])
                o = M.observables(f, ax)
                out[(label, sp[6:], c, E)] = {n: o[M.OBS.index(n)] for n in names}
    return out


def cmd_report():
    import diag_ew_charge_fourier as fz

    names = ["a1_rel", "dphi", "NS", "WE"]
    ax = fz.geomagnetic_ew_axis()
    m = np.load(os.path.join(D.RUN, "out", "updiag_modes.npz"))
    f = np.load(os.path.join(D.RUN, "out", "engine_F_fine.npz"))
    e = m["e"]
    pat = {}
    for mode in ("single", "cone"):
        pat[mode] = {sp: m[f"{mode}_{sp}"] for sp in D.SPECIES}
    pat["full"] = {sp: f[f"flux_hor_{sp}"] / f[f"base_hor_{sp}"][:, None, :] for sp in D.SPECIES}
    # up-going rows -0.05, -0.15, -0.25 (indices 9, 8, 7), mirrored to the same order as CZ_HOR
    pat["up (engine)"] = {sp: (f[f"flux_up_{sp}"] / f[f"base_up_{sp}"][:, None, :])[[9, 8, 7]] for sp in D.SPECIES}
    rows = []
    res = {k: _obs_table(v, e, ax, D.CZ_HOR, k, names) for k, v in pat.items()}
    L = ["| species | |cos Z| | E | observable | down single | down cone (det. anchor) | down full | **up engine** |", "|---|---|---|---|---|---|---|---|"]
    for sp in D.MC_SPECIES:
        for c in D.CZ_HOR:
            for E in (0.3, 0.5, 1.0):
                for n in names:
                    v = [res[k][(k, sp, c, E)][n] for k in ("single", "cone", "full", "up (engine)")]
                    L.append(f"| {sp} | {c} | {E} | {n} | " + " | ".join(f"{x:.3f}" for x in v) + " |")
    open(os.path.join(D.RUN, "out", "updiag_report.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    {"modes": cmd_modes, "report": cmd_report}[sys.argv[1]](*sys.argv[2:])
