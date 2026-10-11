import sys
import numpy as np
R = "/afs/cern.ch/work/p/pigrange/daemonflux"
sys.path.insert(0, R + "/tools/mc3d")
from constants import MASS
tab = dict(np.load(R + "/tools/mc3d/mceq_tables_SIBYLL23D.npz"))
eg, eb = tab["e_grid"], tab["e_bins"]
for i_pool in (9, 10, 13):
    d = dict(np.load(f"/home/pigrange/mc3d_work/pools/pool_2212_{i_pool:03d}_2000.npz"))
    e0 = float(d["e0"][0]); ekp = e0 - MASS[2212]
    j = int(np.argmin(np.abs(np.log(eg / ekp))))
    pid = d["pid"]; p2 = d["px"].astype(float)**2 + d["py"].astype(float)**2 + d["pz"].astype(float)**2
    m = np.array([MASS.get(int(q), 0.0) for q in pid]); ek = np.sqrt(m*m + p2) - m
    nev = int(d["n"][0])
    for lo in (1.122, 1.778, 2.818):
        ib = int(np.argmin(np.abs(eb - lo)))
        pool_n = (np.isin(pid, (2212, 2112)) & (ek > eb[ib])).sum() / nev
        # scale the table column to the pool energy (x-scaling): count bins above eb[ib]*(eg[j]/ekp)
        tN = (tab["y_2212_2212"][:, j] + tab["y_2212_2112"][:, j]).astype(float)
        tab_n = tN[ib:].sum()
        pool_pi = (np.isin(pid, (211, -211)) & (ek > eb[ib])).sum() / nev
        tpi = (tab["y_2212_211"][:, j] + tab["y_2212_-211"][:, j]).astype(float)[ib:].sum()
        print(f"E_kin proj {ekp:6.2f} (table {eg[j]:6.2f}): E_kin > {eb[ib]:.3f}: nucleons pool {pool_n:.3f} table {tab_n:.3f} ratio {pool_n/tab_n:.3f} | pi+- pool {pool_pi:.3f} table {tpi:.3f} ratio {pool_pi/tpi:.3f}")
