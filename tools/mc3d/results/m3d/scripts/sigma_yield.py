"""Task 3c: Sigma+- in raw chromo DPMJET-III-19.3 events at low energy
(pool.KEEP drops them, MILESTONE2 item (i)), and what their decays would add
per p + air interaction: nucleons >= 2 GeV total (re-interaction floor) and
charged pions >= 0.7 GeV (can make a 0.3-1 GeV nu directly).  Sigma+ -> p pi0
(51.6%) / n pi+ (48.3%), Sigma- -> n pi- (99.85%), isotropic in the Sigma frame."""
import os
import sys

import numpy as np

sys.path.insert(0, os.environ["R"] + "/tools/mc3d")
import pool as P
from constants import MASS

MP, MN, MPI, MPI0 = 0.938272, 0.939565, 0.139570, 0.134977
rng = np.random.default_rng(7)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
L = []
gens = P._Generators(seed=4242)
for e0 in (3.56, 5.64, 7.11, 11.27, 17.88):
    name = P._model_name(e0)
    fr = P._target_fractions(gens, 2212, e0, name)
    n_s, nev = 0, 0
    nuc = []
    pic = []
    for (_nm, a, z, _f), k in zip(P.AIR_COMPONENTS, rng.multinomial(N, fr)):
        m = gens.get(name, 2212, e0, (a, z))
        for ev in m(int(k)):
            fs = ev.final_state()
            pid = np.asarray(fs.pid)
            nev += 1
            sel = np.where(np.isin(np.abs(pid), (3222, 3112)))[0]
            for q in sel:
                n_s += 1
                M = MASS.get(abs(int(pid[q])), 1.19)
                P4 = np.array([float(fs.en[q]), float(fs.px[q]), float(fs.py[q]), float(fs.pz[q])])
                if int(pid[q]) in (3222, -3112):      # Sigma+ / anti-Sigma-
                    ch = (MP, MPI0) if rng.random() < 0.516 else (MN, MPI)
                else:
                    ch = (MN, MPI)
                m1, m2 = ch
                pst = np.sqrt((M**2 - (m1 + m2)**2) * (M**2 - (m1 - m2)**2)) / (2 * M)
                c = rng.uniform(-1, 1); ph = rng.uniform(0, 2 * np.pi); s = np.sqrt(1 - c * c)
                d = np.array([s * np.cos(ph), s * np.sin(ph), c])
                b = P4[1:] / P4[0]; b2 = b @ b; g = 1 / np.sqrt(1 - b2)
                outs = []
                for mm, sg in ((m1, 1), (m2, -1)):
                    pv = sg * pst * d; e_ = np.sqrt(mm * mm + pst * pst)
                    bp = b @ pv
                    ee = g * (e_ + bp)
                    outs.append(ee)
                nuc.append(outs[0]); 
                if abs(m2 - MPI) < 1e-3:
                    pic.append(outs[1])
    nuc, pic = np.array(nuc), np.array(pic)
    L.append(f"E_tot {e0:5.2f} ({name}): Sigma+- per interaction {n_s / nev:.4f}; decay nucleons E>=2 GeV {np.sum(nuc >= 2.0) / nev:.4f}, "
             f"all {len(nuc) / nev:.4f}; decay charged pi E>=0.7 {np.sum(pic >= 0.7) / nev:.4f}  ({nev} events)")
open(os.environ["M"] + "/reports/sigma_yield.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
