"""Are the pool's missing re-interacting nucleons the dropped Sigma+- ?
Raw chromo events (no KEEP filter), p + N/O mix as in the pool."""
import sys
import numpy as np
R = "/afs/cern.ch/work/p/pigrange/daemonflux"
sys.path.insert(0, R + "/tools/mc3d")
import pool as P
from constants import MASS

gens = P._Generators(seed=4242)
HYP = {3222: "Sigma+", 3112: "Sigma-", -3222: "aSigma-", -3112: "aSigma+",
       3312: "Xi-", 3322: "Xi0", 3334: "Omega-", 3212: "Sigma0", 3122: "Lambda"}
for e0 in (22.51, 44.95, 95.0):
    name = P._model_name(e0)
    fr = P._target_fractions(gens, 2212, e0, name)
    n = 3000
    counts = {k: 0 for k in HYP}
    nuc_hi_from_sig = 0.0
    sig_e = []
    nev = 0
    rng = np.random.default_rng(1)
    for (_nm, a, z, _f), k in zip(P.AIR_COMPONENTS, rng.multinomial(n, fr)):
        m = gens.get(name, 2212, e0, (a, z))
        for ev in m(int(k)):
            fs = ev.final_state()
            pid = np.asarray(fs.pid)
            en = np.asarray(fs.en, float)
            nev += 1
            for q in HYP:
                s = pid == q
                counts[q] += int(s.sum())
                if abs(q) in (3222, 3112):
                    sig_e.extend(en[s].tolist())
    sig_e = np.array(sig_e)
    # Sigma -> N pi : the nucleon takes E_N ~ E_Sigma * E*_N / M_Sigma
    # (E*_N = 0.94 GeV in the Sigma frame, M_Sigma = 1.19) -> ~0.79 E_Sigma
    frac_hi = np.mean(0.79 * sig_e > 2.0) if len(sig_e) else 0.0
    print(f"E_tot={e0:6.2f} {name:12s} {nev} events: " + ", ".join(
        f"{HYP[q]} {counts[q]/nev:.4f}" for q in HYP))
    print(f"   Sigma+- (incl. anti) per event: {len(sig_e)/nev:.4f}; their "
          f"decay nucleon above 2 GeV: {len(sig_e)*frac_hi/nev:.4f} per event "
          f"(<E_Sigma> = {sig_e.mean() if len(sig_e) else 0:.2f} GeV)")
