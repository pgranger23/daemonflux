"""Task 3b: the milestone-1 closure ladder at low primary energy, with an
optional re-interaction floor imposed on the MCEq-table backend.

    python closure_floor.py MODE FLOOR NSHOWER NPROC OUT E_kin [E_kin ...]

MODE = mceq (rung A1: table yields + MCEq decay tables), kinematic (A2: table
yields + decay kinematics) or chromo (B: DPMJET pools, the MC's own backend,
whose floor is ``pool.E_MIN_POOL`` = 2 GeV total).  FLOOR [GeV, total energy]:
hadrons below it make no interaction products in the table backends (as
``EventPool.draw`` does below 2 GeV); 0 = none.  The reference is MCEq's 1D
solution for the same primary (``closure.mceq_reference``, shipped v140 DB).
Primary energies are KINETIC (closure's convention)."""
import os
import sys

import numpy as np

import closure
from interactions import MCEqYieldBackend
from scoring import SPECIES

mode, floor, nsh, nproc, out = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
energies = [float(x) for x in sys.argv[6:]]


class FloorBackend(MCEqYieldBackend):
    def interact(self, rng, pdg, e_tot, e_cut=None):
        if e_tot < floor:
            return []
        return super().interact(rng, pdg, e_tot, e_cut)


if os.environ.get("GUARD_MASS"):         # 0.94 = the guard of milestones 1-3c
    import pool
    pool.GUARD_MASS = float(os.environ["GUARD_MASS"])
if floor > 0:
    closure.MCEqYieldBackend = FloorBackend
cache = os.environ["MC3D_POOL_CACHE"] if mode == "chromo" else None
res = closure.run(energies, nsh, nproc, mode, shard=250, cache=cache, xs="mceq")
store = {"e_bins": closure.E_BINS}
for e_p, sc in res.items():
    cp = os.environ["M"] + f"/work/mceqref_{e_p:g}_{int(mode != 'mceq')}.npz"
    if os.path.exists(cp):
        z = np.load(cp)
        eg, ref = z["eg"], {s: z[f"r{s}"] for s in SPECIES}
    else:
        eg, ref = closure.mceq_reference(e_p, helicity=(mode != "mceq"))
        np.savez(cp, eg=eg, **{f"r{s}": ref[s] for s in SPECIES})
    rb = closure.rebin_reference(eg, ref, closure.E_BINS)
    for s in SPECIES:
        y, err = sc.dnde(s)
        store[f"mc_{e_p}_{s}"] = y
        store[f"err_{e_p}_{s}"] = err
        store[f"ref_{e_p}_{s}"] = rb[s]
np.savez(out, **store)
print("wrote", out)
