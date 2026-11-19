"""Euler-step study of MCEq's march for gate G1.1 (diagnosis, numpy).

For each (population, loss, direction index) runs ``mceq_euler.EulerMarch`` with
MCEq's own steps refined ``n`` times (1, 2, 4, ...) and with the transported-species
up-scatter dropped, and saves the G1.1 quantities to
``$N3/c1/diag/euler_<pop>_<loss>_<i>.npz``.

Usage::

    taskset -c 0-3 $PY tools/cascade3d/gates/euler_refine.py PHYS lossoff 0,5 1,2,4
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mceq_euler as ME  # noqa: E402

OUT = os.path.join(ME.N3, "c1", "diag")


def main(argv):
    pop, loss_s = argv[1], argv[2]
    dirs = [int(x) for x in argv[3].split(",")]
    refs = [int(x) for x in argv[4].split(",")]
    loss = loss_s == "losson"
    os.makedirs(OUT, exist_ok=True)
    z = np.load(ME.ref_path(pop, loss))
    em = ME.EulerMarch(pop, loss)
    emd = ME.EulerMarch(pop, loss, drop_upscatter=True, d=em.d)
    for i in dirs:
        res = {}
        for n in refs:
            t = time.time()
            q = em.quantities(em.run(z[f"path_dX_{i}"], z[f"path_rho_inv_{i}"],
                                     refine=n))
            for k, v in q.items():
                res[f"r{n}_{k}"] = v
            print(pop, loss_s, i, "refine", n, f"{time.time() - t:.0f} s", flush=True)
        q = emd.quantities(emd.run(z[f"path_dX_{i}"], z[f"path_rho_inv_{i}"]))
        for k, v in q.items():
            res[f"noup_{k}"] = v
        q = em.quantities(em.run(z[f"path_dX_{i}"], z[f"path_rho_inv_{i}"],
                                 exact_end=float(z["max_X"][i])))
        for k, v in q.items():
            res[f"exact_{k}"] = v
        tag = "" if refs == [1, 2, 4] else "_r" + "-".join(map(str, refs))
        path = os.path.join(OUT, f"euler_{pop}_{loss_s}_{i}{tag}.npz")
        np.savez(path, e_grid=em.d["e_grid"], refine=np.array(refs), **res)
        print("wrote", path, flush=True)


if __name__ == "__main__":
    main(sys.argv)
