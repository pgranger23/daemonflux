"""Task 3d: first-generation products at the MC's own sampling level
(ChromoBackend.interact: bin choice + p_z rescale + guard) vs the table sampler
(MCEqYieldBackend.interact), p + air at kinetic E_p, in total-energy bins."""
import os, sys
import numpy as np
sys.path.insert(0, os.environ["R"] + "/tools/mc3d")
from constants import MASS
from interactions import ChromoBackend, MCEqYieldBackend
P = ChromoBackend(cache_dir=os.environ["MC3D_POOL_CACHE"], xs="mceq", readonly=True)
T = MCEqYieldBackend()
EB = np.array([0.2, 0.3, 0.45, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 1e3])
rng = np.random.default_rng(11)
N = 20000
L = ["# per p+air interaction, products per total-energy bin (edges " + " ".join(f"{x:g}" for x in EB[:-1]) + " GeV): pool(rescaled) / table ; counts",
     "E_p(kin)  child  bin ratios pool/table (pool count per interaction)"]
for ek in (1.5, 2.5, 4.0, 6.0, 10.0, 20.0):
    et = ek + MASS[2212]
    cnt = {"P": {}, "T": {}}
    for tag, be in (("P", P), ("T", T)):
        for _ in range(N):
            for item in be.interact(rng, 2212, et):
                c, e = int(item[0]), float(item[1])
                cnt[tag].setdefault(c, []).append(e)
    L.append(f"-- E_p = {ek} GeV kinetic (E_tot {et:.3f})")
    for nm, c in (("pi+", 211), ("pi-", -211), ("K+", 321), ("p", 2212), ("n", 2112)):
        hp = np.histogram(cnt["P"].get(c, []), EB)[0] / N
        ht = np.histogram(cnt["T"].get(c, []), EB)[0] / N
        L.append(f"  {nm:4s} " + " ".join(f"{(a / b if b > 2e-3 else float('nan')):5.2f}({a:5.3f})" for a, b in zip(hp, ht)) + f"   total>0.2: {hp.sum():.3f}/{ht.sum():.3f}")
open(os.environ["M"] + "/reports/first_gen.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
