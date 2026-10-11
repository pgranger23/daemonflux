"""Task 4 diagnostic: chromo DPMJET-III-19.3 pool vs MCEq's tabulated
DPMJET-III-19.3 (mceq_tables_SIBYLL23D.npz below the splice), per interaction,
p + air at ~20 and ~50 GeV.  x = E_kin(secondary) / E_kin(projectile)."""
import sys
import numpy as np
R = "/afs/cern.ch/work/p/pigrange/daemonflux"
sys.path.insert(0, R + "/tools/mc3d")
import pool as P
from constants import MASS

W = "/home/pigrange/mc3d_work/pools"
tab = dict(np.load(R + "/tools/mc3d/mceq_tables_SIBYLL23D.npz"))
eg, eb = tab["e_grid"], tab["e_bins"]
MP = MASS[2212]
XB = np.array([0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.01])


def pool_events(pdg, i):
    d = dict(np.load(f"{W}/pool_{pdg}_{i:03d}_2000.npz"))
    return d


def table_hist(proj, child, j, xb):
    """number per interaction of `child` in x bins (x = Ekin_c/Ekin_p) from
    the MCEq column j (projectile E_kin = eg[j]); bin contents split
    uniformly in E_kin within each MCEq bin."""
    N = tab.get(f"y_{proj}_{child}")
    if N is None:
        return np.zeros(len(xb) - 1)
    col = N[:, j].astype(float)
    out = np.zeros(len(xb) - 1)
    ep = eg[j]
    for i in range(len(col)):
        if col[i] == 0:
            continue
        lo, hi = eb[i] / ep, eb[i + 1] / ep
        for k in range(len(xb) - 1):
            ov = max(0.0, min(hi, xb[k + 1]) - max(lo, xb[k]))
            if ov > 0:
                out[k] += col[i] * ov / (hi - lo)
    return out


def pool_hist(d, child_set, xb):
    e0 = float(d["e0"][0])
    ekp = e0 - MP
    pid, px, py, pz = d["pid"], d["px"].astype(float), d["py"].astype(float), d["pz"].astype(float)
    m = np.array([MASS.get(int(q), 0.0) for q in pid])
    e = np.sqrt(m * m + px * px + py * py + pz * pz)
    x = (e - m) / ekp
    sel = np.isin(pid, list(child_set))
    h, _ = np.histogram(np.clip(x[sel], 0, 1.0), bins=xb)
    n_ev = int(d["n"][0])
    # leading-nucleon fraction and 'no new hadron' events
    off = d["off"]
    return h / n_ev, x, sel, e0, n_ev


for i_pool in (9, 10, 13, 14):
    d = pool_events(2212, i_pool)
    e0 = float(d["e0"][0])
    ekp = e0 - MP
    j = int(np.argmin(np.abs(np.log(eg / ekp))))
    print(f"\n=== p + air: pool bin {i_pool} E_tot={e0:.2f} (E_kin {ekp:.2f}) "
          f"<-> MCEq column E_kin={eg[j]:.2f} ({100*(eg[j]/ekp-1):+.1f}%), "
          f"pool model {d['model'][0]}, {int(d['n'][0])} events")
    for name, kids_pool, kids_tab in (("pi+", (211,), (211,)), ("pi-", (-211,), (-211,)),
                                       ("p", (2212,), (2212,)), ("n", (2112,), (2112,)),
                                       ("K+", (321,), (321,))):
        hp, x, sel, _, nev = pool_hist(d, kids_pool, XB)
        ht = sum(table_hist(2212, c, j, XB) for c in kids_tab)
        print(f"  {name:4s} total pool {hp.sum():6.3f}  table {ht.sum():6.3f}  "
              f"ratio {hp.sum()/max(ht.sum(),1e-9):5.3f} | <x> pool "
              f"{(x[sel]).mean():.4f}")
        row = "       x-bin ratio pool/table: " + " ".join(
            f"[{XB[k]:.3g},{XB[k+1]:.3g}):{hp[k]/ht[k] if ht[k]>0 else np.nan:5.2f}"
            f"({hp[k]:.3f})" for k in range(len(XB) - 1))
        print(row)
    # nucleon energy flow: the cascade-driving nucleons (E_tot > 2 GeV)
    pid = d["pid"]
    px, py, pz = d["px"].astype(float), d["py"].astype(float), d["pz"].astype(float)
    m = np.array([MASS.get(int(q), 0.0) for q in pid])
    e = np.sqrt(m * m + px * px + py * py + pz * pz)
    nuc = np.isin(pid, (2212, 2112))
    nev = int(d["n"][0])
    print(f"  nucleons above the 2 GeV re-interaction cut per event: pool "
          f"{(nuc & (e > 2.0)).sum()/nev:.3f}; below it (target fragments + "
          f"slow): {(nuc & (e <= 2.0)).sum()/nev:.3f}")
    # table: nucleons above E_tot = 2 GeV <=> E_kin > 1.06
    tN = tab["y_2212_2212"][:, j].astype(float) + tab["y_2212_2112"][:, j].astype(float)
    print(f"                                          table "
          f"{tN[eg > 2.0 - MP].sum():.3f}; below it: {tN[eg <= 2.0 - MP].sum():.3f}")
    # energy fractions
    off = d["off"]
    pi_ch = np.isin(pid, (211, -211))
    efrac_pi = e[pi_ch].sum() / nev / e0
    efrac_N = e[nuc].sum() / nev / e0
    tpi = (tab["y_2212_211"][:, j] + tab["y_2212_-211"][:, j]).astype(float)
    print(f"  energy fraction in pi+-: pool {efrac_pi:.3f}  table "
          f"{(tpi*(eg+MASS[211])).sum()/(eg[j]+MP):.3f};  in nucleons: pool "
          f"{efrac_N:.3f}  table {(tN*(eg+MP)).sum()/(eg[j]+MP):.3f}")
    # events with no charged pion at all (quasi-elastic-like) and leading nucleon x
    npi = np.add.reduceat(pi_ch.astype(int), off[:-1]) if len(pid) else np.zeros(nev)
    lead = []
    for a, b in zip(off[:-1], off[1:]):
        sl = slice(a, b)
        xx = ((e[sl] - m[sl]) / (e0 - MP))[nuc[sl]]
        lead.append(xx.max() if len(xx) else 0.0)
    lead = np.array(lead)
    print(f"  events with no charged pion: {np.mean(npi == 0):.3f};  leading "
          f"nucleon x>0.9: {np.mean(lead > 0.9):.3f}, x>0.7: {np.mean(lead > 0.7):.3f}, "
          f"<x_lead> {lead.mean():.3f}")
    xt = eg / eg[j]
    print(f"  table leading-nucleon proxy: N(x>0.9) = {tN[xt > 0.9].sum():.3f}, "
          f"N(x>0.7) = {tN[xt > 0.7].sum():.3f} per interaction")
