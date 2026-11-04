"""Per-stratum re-weighting diagnostics only (no tracing, no replay):
the same ``_diag`` records as ``reweight_replay.py``, from the c-files.

    python rw_diag_run.py SRC_RUNDIR TAG NPROC TARGET [TARGET ...] -> $M/work/rwdiag_<TAG>_diag.npz
"""
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reweight_replay as rr  # noqa: E402  (parses argv, no side effects)
import primaries as prim  # noqa: E402
import sky  # noqa: E402

targets = [rr.parse_target(s) for s in rr.args[3:]]
tag = rr.tag


def one(i):
    d = rr._load(i)
    T, is_p = rr.primaries_of(d)
    gid = sky.GROUPS[str(d["group"])][3]
    out = []
    for (_nm, _c, new, label) in targets:
        f = prim.reweight_factor(new, rr.OLD, gid, T, is_p)
        dg = rr._diag(d, f)
        dg.update(label=label, group=str(d["group"]), k=int(d["k"]), ti=i)
        out.append(dg)
    return out


if __name__ == "__main__":
    with mp.get_context("fork").Pool(rr.nproc) as pool:
        alld = [x for r in pool.imap_unordered(one, range(rr.ntask), chunksize=20)
                for x in r]
    keys = ("label", "group", "k", "ti", "n", "n3", "S1", "S2", "F1", "F2", "T1",
            "TF1", "fmin", "fmax", "fz")
    p = os.path.join(os.environ["M"], "work", f"rwdiag_{tag}_diag.npz")
    np.savez(p, **{k: np.array([x[k] for x in alld]) for k in keys})
    print("wrote", p, len(alld))
