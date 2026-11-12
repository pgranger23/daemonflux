"""E_off horizon gate: the field-free P' test of an E_off table against the 3D MC add-on (2026-11-12, mgfix).

The MC add-on run (tools/mc3d sky3d, no cutoff, no primary bend, muon field on with its zero-field counterfactual) gives the
field-free 3D flux S' = s - mu_on + mu_cf per (species, E, cos Z) and the straight-line reference t.  With the engine's REF 1D base
B (bin integrals) and the table's E_off (G = 1, MC population: primary bands T >= 1 GeV/n, base-weighted in E),

    P'_X = S' / (kappa_A B E_off_X),        kappa_A = sum_{cos Z 0.6-1} t / sum_{cos Z 0.6-1} B,

and the horizon departure is D(P') = ln P'(row) - <ln P'>_{rows 0.35-0.95} (species mean; 0 = the table has the MC's field-free
3D/1D zenith shape).  Gate (mgfix PREREG, criterion (i)): every cell of rows 0.05 / 0.15 / 0.25 x E bands 0.10-1.0 GeV (5/dec)
within max(3%, 2 sigma) of 0, sigma = jackknife over the MC's 20 groups (stratified replicates).  Exactly the hzn study's b3
metric ($W/runs/hzn/scripts/b3_pp_ingredients.py, hzn_lib.load_A/base_B/engine_eoff), re-implemented here without the run
directory.  fix2's closure gate tests cos Z 0.2-1 only; this one is the horizon half of the E_off gate.

Known limits of the reference (mgfix HANDBACK): the MC drops charged pions below e_nu_min/(1 - m_mu^2/m_pi^2) = 0.234 GeV, which
removes muon-decay neutrinos up to ~0.21 GeV; the add-on's row-0.05 cells carry 3-7% jackknife errors.

usage:
  diag_eoff_horizon.py --addon ADDON_ADDON.npz --sums A1_ADDON_SUMS.npz --base BASE1D.npz --fine ENGINE_F_FINE.npz TABLE [TABLE ...]
    ADDON_ADDON.npz   m4/work/addon_addon.npz (x: G, K, [s, mu_on, mu_cf], sp, 60 E 20/dec, 20 cos Z, 12 az; M; area)
    A1_ADDON_SUMS.npz hzn/out/a1_addon_sums.npz (azx: G, K, [s1, n1, t1], sp, 40 E, 10 cos Z down, 12 az; M)
    BASE1D.npz        integ/out/base1d_kamioka.npz (B_t1: sp, 60 E, 20 cos Z)
    ENGINE_F_FINE.npz integ/out/engine_F_fine.npz (fine e grid, e_widths, base_hor_*/base_dn6_* = B_ref at the row centres)
    TABLE             eoff_v3/v4/v5 table on the 10 row centres 0.05 .. 0.95 (E_off_s_mc, or N_bsc/D_bsc with bands)
"""

from __future__ import annotations

import argparse
import os

import numpy as np

SP = ("nue", "antinue", "numu", "antinumu")           # MC order
ESP = ("total_nue", "total_antinue", "total_numu", "total_antinumu")
E5 = 10.0 ** (-1.0 + 0.2 * np.arange(11))
MID = list(range(3, 10))
EB = ("0.10-0.16", "0.16-0.25", "0.25-0.40", "0.40-0.63", "0.63-1.0", "1.0-1.6", "1.6-2.5")
NE = 5                                                 # gated bands 0.10-1.0 GeV
ROWS = (0, 1, 2)                                       # cos Z 0.05 / 0.15 / 0.25


def e5_from20(x, ax):
    """Merge 20/dec E bins in groups of four (-> 5/dec) along axis ``ax``."""
    x = np.moveaxis(x, ax, -1)
    s = x.shape
    return np.moveaxis(x.reshape(s[:-1] + (s[-1] // 4, 4)).sum(-1), -1, ax)


def replicates(S, Mg):
    """Stratified leave-one-group-out totals: S (G, K, ...), Mg (G, K) primaries -> (G, ...)."""
    Sk, Mk = S.sum(0), Mg.sum(0)
    bshape = (-1,) + (1,) * (S.ndim - 2)
    out = []
    for g in range(S.shape[0]):
        rem = Mk - Mg[g]
        fac = np.where(rem > 0, Mk / np.where(rem > 0, rem, 1.0), 0.0)
        out.append(((Sk - S[g]) * fac.reshape(bshape)).sum(0))
    return np.array(out)


def load_addon(addon, sums):
    """dict s, mu_on, mu_cf, t of shape (G, K, 4 sp, 10 E5, 10 cos Z down), azimuth summed; and M (G, K)."""
    a = np.load(addon)
    xa = e5_from20(a["x"][..., :40, 10:, :] / float(a["area"]), 4).sum(-1)      # (G, K, 3, sp, 10, 10)
    b = np.load(sums)
    if not np.allclose(a["M"], b["M"]):
        raise ValueError("addon and sums files come from different groupings")
    t = e5_from20(np.asarray(b["azx"][:, :, 2]), 3).sum(-1)                     # (G, K, sp, 10, 10)
    return {"s": xa[:, :, 0], "mu_on": xa[:, :, 1], "mu_cf": xa[:, :, 2], "t": t}, a["M"]


def load_base(base):
    bt = np.load(base)["B_t1"]
    Bc = bt[:, :40, 10:].reshape(4, 10, 4, 10).sum(2)                           # (sp, 10 E5, 10 rows)
    return Bc, Bc[:, :, 6:10].sum(-1)


def table_eoff(table, fine):
    """E_off (MC population) on (4 sp, 10 E5, 10 rows), B_ref x dE weighted over the fine grid of each 5/dec bin."""
    tab = np.load(table)
    f = np.load(fine)
    e, de = f["e"], f["e_widths"]
    if not np.allclose(tab["cz"], np.round(np.arange(0.05, 1.0, 0.1), 2)):
        raise ValueError("table rows must be the 10 row centres 0.05 .. 0.95")
    tsp = [str(s) for s in tab["species"]]
    arr = (tab["E_off_s_mc"] if "E_off_s_mc" in tab.files
           else tab["N_bsc"][1:].sum((0, 2)) / tab["D_bsc"][1:].sum((0, 2)))
    out = np.zeros((4, 10, 10))
    for k, sp in enumerate(SP):
        base = np.concatenate([f[f"base_hor_{ESP[k]}"], f[f"base_dn6_{ESP[k]}"]], 0)
        eo = np.array([np.interp(np.log(e), np.log(tab["e"]), arr[tsp.index(sp), r]) for r in range(10)])
        for ie in range(10):
            m = (e >= E5[ie]) & (e < E5[ie + 1])
            w = base[:, m] * de[m]
            out[k, ie] = (eo[:, m] * w).sum(-1) / w.sum(-1)
    return out


def departure(x):
    lx = np.log(x)
    return lx - lx[..., MID].mean(-1, keepdims=True)


def p_prime_D(A, Bc, Bk, eoff):
    """species-mean D(P') (7 E5, 10 rows) from tallies A (dict of (sp, 10, 10))."""
    kA = A["t"][:, :, 6:10].sum(-1) / Bk
    x = (A["s"] - A["mu_on"] + A["mu_cf"])[:, :7] / (kA[:, :7, None] * Bc[:, :7] * eoff[:, :7])
    return departure(x).mean(0)


def horizon_test(A, M, Bc, Bk, eoff):
    """(D, sigma, pass mask (NE, 3), n_pass) for one E_off array."""
    tot = {k: v.sum((0, 1)) for k, v in A.items()}
    rep = {k: replicates(v, M) for k, v in A.items()}
    G = M.shape[0]
    d = p_prime_D(tot, Bc, Bk, eoff)
    r = np.array([p_prime_D({k: rep[k][g] for k in rep}, Bc, Bk, eoff) for g in range(G)])
    sig = np.sqrt((G - 1.0) / G * ((r - r.mean(0)) ** 2).sum(0))
    ok = np.abs(d[:NE][:, ROWS]) <= np.maximum(0.03, 2 * sig[:NE][:, ROWS])
    return d, sig, ok, int(ok.sum())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--addon", required=True)
    ap.add_argument("--sums", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--fine", required=True)
    ap.add_argument("--out", default=None, help="optional npz with D, sigma per table")
    ap.add_argument("tables", nargs="+")
    a = ap.parse_args(argv)
    A, M = load_addon(a.addon, a.sums)
    Bc, Bk = load_base(a.base)
    res = {}
    for t in a.tables:
        d, sig, ok, n = horizon_test(A, M, Bc, Bk, table_eoff(t, a.fine))
        res[os.path.basename(t)] = (d, sig)
        print(f"\n{os.path.basename(t)}: species-mean D(P') rows 0.05 / 0.15 / 0.25 (+- jackknife; * = outside max(3%, 2 sigma))")
        for ie in range(NE):
            print(f"  {EB[ie]}  " + "  ".join(f"{d[ie, ic]:+.3f}({1e3 * sig[ie, ic]:.0f}){'' if ok[ie, j] else '*'}"
                                              for j, ic in enumerate(ROWS)))
        print(f"  E_off horizon gate: {n}/{NE * len(ROWS)} cells -> {'PASS' if n == NE * len(ROWS) else 'FAIL'}")
    if a.out:
        np.savez(a.out, **{k + "_D": v[0] for k, v in res.items()}, **{k + "_sigma": v[1] for k, v in res.items()})
    return res


if __name__ == "__main__":
    main()
