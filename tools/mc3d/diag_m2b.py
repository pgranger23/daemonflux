"""Milestone-2b diagnostics (MILESTONE2_RESULTS.md sec. 2.4, 2.6 and 4.4).

    python diag_m2b.py db        # which MCEq database the yield tables come from
    python diag_m2b.py splice    # where SIBYLL23D == DPMJETIII193 stops
    python diag_m2b.py dpm       # chromo pool vs MCEq table per interaction
    python diag_m2b.py sigma     # Sigma+- in raw chromo events (dropped by KEEP)
    python diag_m2b.py ptpool    # pion production angle: pools vs the engine's
                                 # moment rows, and the engine's pooling bias

``dpm``, ``sigma`` and ``ptpool`` need ``MC3D_POOL_CACHE``; ``ptpool`` reads
``tools/mceq3d/m_spliced_v2.npz`` / ``m_piminus_v2.npz`` (read-only).
``sigma`` constructs chromo generators, so run it from a scratch directory
(chromo writes ``tables.dat`` into the cwd).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MCEQ3D = os.path.join(HERE, "..", "mceq3d")
TABLES = os.path.join(HERE, "mceq_tables_SIBYLL23D.npz")
X_EDGES = np.array([0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5,
                    0.7, 0.9, 1.01])


def _mceq_run(model, emin=0.05):
    import importlib.util  # noqa: F401
    import MCEq.config as cfg
    cfg.e_min = emin
    cfg.debug_level = 0
    import crflux.models as crf
    from MCEq.core import MCEqRun
    return MCEqRun(interaction_model=model,
                   primary_model=(crf.HillasGaisser2012, "H3a"), theta_deg=0.0)


def _export(mc, pairs):
    eg = mc.e_grid
    xr = eg[:, None] / eg[None, :]
    out = {}
    for p, c in pairs:
        try:
            out[(p, c)] = (mc._interactions.get_matrix((p, 0), (c, 0))
                           * xr).astype(np.float32)
        except Exception:
            pass
    return eg, out


PAIRS = ((2212, 211), (2212, -211), (2212, 2212), (2212, 2112), (211, 211),
         (211, -211), (2212, 321))


def cmd_db():
    """The shipped npz vs fresh SIBYLL23D / DPMJETIII193 exports."""
    import MCEq.config as cfg
    print("config.mceq_db_fname =", cfg.mceq_db_fname)
    print("data_dir listing     =", sorted(os.listdir(cfg.data_dir)))
    tab = dict(np.load(TABLES))
    eg, sib = _export(_mceq_run("SIBYLL23D", float(tab["emin"][0])), PAIRS)
    _, dpm = _export(_mceq_run("DPMJETIII193", float(tab["emin"][0])), PAIRS)
    for pc in PAIRS:
        a = tab[f"y_{pc[0]}_{pc[1]}"]
        print(f"{pc}: |npz - fresh SIBYLL23D export| max = "
              f"{np.abs(a - sib[pc]).max():.1e}")
        print("   npz/DPMJET per projectile E_kin: " + "  ".join(
            f"{eg[j]:.1f}:{a[:, j].sum() / max(dpm[pc][:, j].sum(), 1e-30):.4f}"
            f"{'=' if np.array_equal(a[:, j], dpm[pc][:, j]) else ''}"
            for j in range(len(eg)) if 2.0 < eg[j] < 1200.0 and j % 2 == 0))


def cmd_splice():
    tab = dict(np.load(TABLES))
    eg, dpm = _export(_mceq_run("DPMJETIII193"), PAIRS[:5])
    for pc, n in dpm.items():
        a = tab[f"y_{pc[0]}_{pc[1]}"]
        print(f"{pc}: " + "  ".join(
            f"{eg[j]:.1f}:{'=' if np.array_equal(a[:, j], n[:, j]) else 'x'}"
            f"{a[:, j].sum() / n[:, j].sum():.3f}"
            for j in range(len(eg)) if 25 < eg[j] < 120))


def _pool(pdg, i, cache):
    return dict(np.load(os.path.join(cache, f"pool_{pdg}_{i:03d}_2000.npz")))


def _kin(d):
    from constants import MASS
    pid = d["pid"]
    px, py, pz = (d[k].astype(float) for k in ("px", "py", "pz"))
    m = np.array([MASS.get(int(q), 0.0) for q in pid])
    p2 = px * px + py * py + pz * pz
    e = np.sqrt(m * m + p2)
    return pid, e, m, px, py, np.sqrt(p2)


def _table_x(tab, proj, child, j, xb):
    eg, eb = tab["e_grid"], tab["e_bins"]
    col = tab[f"y_{proj}_{child}"][:, j].astype(float)
    out = np.zeros(len(xb) - 1)
    for i in np.nonzero(col)[0]:
        lo, hi = eb[i] / eg[j], eb[i + 1] / eg[j]
        for k in range(len(xb) - 1):
            ov = max(0.0, min(hi, xb[k + 1]) - max(lo, xb[k]))
            out[k] += col[i] * ov / (hi - lo)
    return out


def cmd_dpm(cache, bins=(9, 10, 13, 14)):
    """Per interaction, p + air: pool (chromo) vs table (MCEq), kinetic x."""
    from constants import MASS
    tab = dict(np.load(TABLES))
    eg = tab["e_grid"]
    mp = MASS[2212]
    for i in bins:
        d = _pool(2212, i, cache)
        e0 = float(d["e0"][0])
        j = int(np.argmin(np.abs(np.log(eg / (e0 - mp)))))
        pid, e, m, *_ = _kin(d)
        x = (e - m) / (e0 - mp)
        nev = int(d["n"][0])
        print(f"\np+air pool E_tot {e0:.2f} ({d['model'][0]}) vs table E_kin "
              f"{eg[j]:.2f}")
        for name, q in (("pi+", 211), ("pi-", -211), ("p", 2212),
                        ("n", 2112), ("K+", 321)):
            hp = np.histogram(np.clip(x[pid == q], 0, 1), X_EDGES)[0] / nev
            ht = _table_x(tab, 2212, q, j, X_EDGES)
            print(f"  {name:4s} pool {hp.sum():.3f} table {ht.sum():.3f} "
                  f"ratio {hp.sum() / ht.sum():.3f}; per x bin: " + " ".join(
                      f"{hp[k] / ht[k]:.2f}" if ht[k] > 0 else "nan"
                      for k in range(len(hp))))
        nuc = np.isin(pid, (2212, 2112))
        tn = (tab["y_2212_2212"][:, j] + tab["y_2212_2112"][:, j]).astype(float)
        print(f"  nucleons E_tot > 2 GeV: pool {(nuc & (e > 2)).sum() / nev:.3f}"
              f" table {tn[eg > 2 - mp].sum():.3f}; slow: pool "
              f"{(nuc & (e <= 2)).sum() / nev:.3f} table "
              f"{tn[eg <= 2 - mp].sum():.3f}")
        off = d["off"]
        lead = np.array([((e[a:b] - m[a:b]) / (e0 - mp))[nuc[a:b]].max()
                         if nuc[a:b].any() else 0.0
                         for a, b in zip(off[:-1], off[1:])])
        xt = eg / eg[j]
        print(f"  leading nucleon x > 0.9: pool {np.mean(lead > 0.9):.3f} "
              f"table {tn[xt > 0.9].sum():.3f}")


def cmd_sigma(energies=(22.51, 44.95, 95.0), n=3000):
    import pool as P
    gens = P._Generators(seed=4242)
    for e0 in energies:
        name = P._model_name(e0)
        fr = P._target_fractions(gens, 2212, e0, name)
        rng = np.random.default_rng(1)
        n_sig, e_sig, nev = 0, [], 0
        for (_nm, a, z, _f), k in zip(P.AIR_COMPONENTS, rng.multinomial(n, fr)):
            m = gens.get(name, 2212, e0, (a, z))
            for ev in m(int(k)):
                fs = ev.final_state()
                pid, en = np.asarray(fs.pid), np.asarray(fs.en, float)
                s = np.isin(np.abs(pid), (3222, 3112))
                n_sig += int(s.sum())
                e_sig.extend(en[s].tolist())
                nev += 1
        e_sig = np.asarray(e_sig)
        # Sigma -> N pi: the nucleon carries ~E*_N/M_Sigma = 0.79 of E_Sigma
        print(f"E_tot {e0:.2f} {name}: Sigma+- per event {n_sig / nev:.4f}, "
              f"decay nucleons above 2 GeV {np.sum(0.79 * e_sig > 2) / nev:.4f}")


def cmd_ptpool(cache):
    """The engine's sigma_pi input vs the pools, and its pooling bias."""
    import pool as P
    eb = np.array([0.3, 0.5, 1.0, 2.0, 5.0])
    for f in ("m_spliced_v2.npz", "m_piminus_v2.npz"):
        d = dict(np.load(os.path.join(MCEQ3D, f)))
        pe, dx = d["proj_energies"], np.diff(d["xl_edges"])
        es, t2, w = d["e_sec"], d["theta_sq"], d["dndx"]
        print(f"{f}: rows {pe.min():.0f}-{pe.max():.0f} GeV")
        for lo, hi in zip(eb[:-1], eb[1:]):
            m = (es >= lo) & (es < hi) & np.isfinite(t2) & (w > 0)
            n = w * dx[None, :]
            sw = (pe ** -1.7)[:, None]
            a = np.degrees(np.sqrt(np.sum(t2[m] * w[m]) / np.sum(w[m])))
            c = np.degrees(np.sqrt(np.sum((t2 * n * sw)[m])
                                   / np.sum((n * sw)[m])))
            share = sum(w[i][m[i]].sum() for i in np.nonzero(pe > 1e3)[0])
            print(f"  E_pi [{lo},{hi}): engine pooling {a:6.2f} deg, "
                  f"pions-per-cell x E_proj^-1.7 {c:6.2f} deg (x{c / a:.2f}); "
                  f"rows > 1 TeV carry {share / w[m].sum():.2f} of the "
                  f"engine weight")
    eng = dict(np.load(os.path.join(MCEQ3D, "m_spliced_v2.npz")))
    pe = eng["proj_energies"]
    print("per projectile energy, RMS arcsin(p_T/p) of pi+ [deg] in E_pi "
          "bins 0.3-0.5 / 0.5-1 / 1-2 / 2-5: pool | engine row")
    for e_proj in (10.0, 20.0, 45.0, 100.0, 300.0, 1000.0):
        d = _pool(2212, P.GRID.index(e_proj), cache)
        pid, e, _m, px, py, p = _kin(d)
        th = np.arcsin(np.clip(np.hypot(px, py) / np.maximum(p, 1e-12), 0, 1))
        r_p, r_e = [], []
        i = int(np.argmin(np.abs(np.log(pe / float(d["e0"][0])))))
        for lo, hi in zip(eb[:-1], eb[1:]):
            s = (pid == 211) & (e >= lo) & (e < hi)
            r_p.append(np.degrees(np.sqrt(np.mean(th[s] ** 2))))
            mm = ((eng["e_sec"][i] >= lo) & (eng["e_sec"][i] < hi)
                  & np.isfinite(eng["theta_sq"][i]) & (eng["dndx"][i] > 0))
            r_e.append(np.degrees(np.sqrt(np.average(
                eng["theta_sq"][i][mm], weights=eng["dndx"][i][mm]))))
        print(f"  {float(d['e0'][0]):7.1f} GeV ({d['model'][0]}): "
              + " ".join(f"{x:5.1f}" for x in r_p) + " | "
              + " ".join(f"{x:5.1f}" for x in r_e) + f"  (row {pe[i]:.0f} GeV)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("db", "splice", "dpm", "sigma", "ptpool"))
    ap.add_argument("--cache", default=os.environ.get("MC3D_POOL_CACHE"))
    a = ap.parse_args(argv)
    sys.path.insert(0, HERE)
    if a.what == "db":
        cmd_db()
    elif a.what == "splice":
        cmd_splice()
    elif a.what == "dpm":
        cmd_dpm(a.cache)
    elif a.what == "sigma":
        cmd_sigma()
    else:
        cmd_ptpool(a.cache)


if __name__ == "__main__":
    main()
