"""Exact re-evaluation of a finished run under the corrected H3a energy
convention (milestone 3c): crflux evaluated at the nucleus' total energy
``A (T + m)`` (MCEq's convention, ``primaries.ENERGY_CONVENTION = "total"``)
instead of ``A T``.

The importance weight of a primary is linear in the flux, so changing the
convention multiplies every entry of a primary by
``f = Phi(A (T + m)) / Phi(A T)`` -- exactly what re-running with the same
seeds would give.  ``T`` follows from the nucleus rigidity stored per pending
primary in the c-file.  The cutoff verdicts of the traced copy states are not
stored, so they are recomputed (``cutoff.trace_gated``, deterministic, the
same model and date); the tallies are then replayed with
``sky3d.tally_entries``.  Output: a new task directory ``<TAG>_tot`` that the
usual analyses (``analyse_m3b``, ``final_report.py``) read, written under
``$M/sky`` (the source run is only read).

    python reweight_replay.py SRC_RUNDIR TAG NPROC [--check]

``--check`` replays a few files with f = 1 and compares with the original
t-files (must be identical).
"""
import multiprocessing as mp
import os
import pickle
import shutil
import sys

import numpy as np

import crflux.models as crf
import cutoff
import primaries as prim
import sky
import sky3d

rundir, tag, nproc = sys.argv[1], sys.argv[2], int(sys.argv[3])
check = "--check" in sys.argv
new = tag + ("_chk" if check else "_tot")
H3A = crf.HillasGaisser2012("H3a")
MP = 0.938272
GID = {g: (a_, z_, gid) for g, a_, z_, gid in prim.SPECIES}
man = dict(np.load(os.path.join(sky3d.task_dir(rundir, tag), "manifest.npz")))
ntask = int(man["ntask"][0])
os_ = man["opts"]
opts = sky3d.Options(cutoff=bool(os_[0]), bend_primary=bool(os_[1]),
                     muon_field=bool(os_[2]), r_thr=float(os_[3]),
                     n_az=int(os_[4]))
OUT = os.path.join(os.environ["M"], "sky")
from scoring import NestedCaps  # noqa: E402
caps = NestedCaps(*sky.SITES[str(man["site"][0])], tuple(man["caps"]))
grid = sky3d.Grid(caps, sky.E_BINS, sky.CZ_BINS, sky.AZ_BINS)


def factor(d):
    a_, z_, gid = GID[d["group"]]
    R = np.asarray(d["R"], float)
    pn = R * z_ / a_
    T = np.sqrt(pn * pn + MP * MP) - MP
    if check:
        return np.ones_like(T)
    return (np.ravel(H3A.nucleus_flux(gid, prim.crflux_energy(T, a_, "total")))
            / np.ravel(H3A.nucleus_flux(gid, prim.crflux_energy(T, a_, "kinetic"))))


def _load(i):
    with open(sky3d.cfile(rundir, tag, i), "rb") as fh:
        return pickle.load(fh)


def group(tis):
    """Re-weight and replay a group of consecutive c-files with ONE pooled
    trace of their copy states (as ``sky3d.trace_replay_task``: the
    vectorised tracer needs ~1e4 states per call to be efficient)."""
    tis = [i for i in tis if not os.path.exists(sky3d.tfile(OUT, new, i))]
    if not tis:
        return 0
    datas = [_load(i) for i in tis]
    nst = [len(d["trace_r"]) for d in datas]
    verdict = np.zeros(sum(nst), bool)
    if opts.cutoff and sum(nst):
        verdict, _n = cutoff.trace_gated(
            np.concatenate([d["trace_r"] for d in datas]),
            np.concatenate([d["trace_u"] for d in datas]),
            np.concatenate([d["trace_R"] for d in datas]), 1,
            model=opts.model, date=opts.date)
    off = 0
    for i, d, ns in zip(tis, datas, nst):
        v = verdict[off:off + ns]
        off += ns
        f = factor(d)
        ent = d["ent"]
        e3, e1 = ent["3d"], ent["1d"]
        if len(e3["flat"]):
            for k in ("w0", "wn"):
                e3[k] = e3[k] * f[e3["prim"]]
        if len(e1["flat"]):
            for k in ("wt", "wv"):
                e1[k] = e1[k] * f[e1["prim"]]
        if "3dcf" in ent and len(ent["3dcf"]["flat"]):
            ent["3dcf"]["w0"] = ent["3dcf"]["w0"] * f[ent["3dcf"]["prim"]]
        A = sky3d.verdicts(d, v, opts.cutoff, opts.n_az)
        tal = sky3d.tally_entries(grid, ent, A)
        old = dict(np.load(sky3d.tfile(rundir, tag, i), allow_pickle=False))
        dd = tal.to_dict()
        for k in old:
            if (k.startswith("cone") or k in ("meta", "group", "m3b_counts",
                                              "m3b_count_names", "m3b_opts")):
                dd[k] = old[k]
        sky._atomic_savez(sky3d.tfile(OUT, new, i), dd, compressed=True)
    return len(tis)


if __name__ == "__main__":
    os.makedirs(sky3d.task_dir(OUT, new), exist_ok=True)
    shutil.copy(os.path.join(sky3d.task_dir(rundir, tag), "manifest.npz"),
                os.path.join(sky3d.task_dir(OUT, new), "manifest.npz"))
    todo = list(range(ntask)) if not check else list(range(0, ntask, max(1, ntask // 6)))
    # groups of consecutive files holding >= 1e4 copy states to trace
    counts = {i: int(np.load(sky3d.cfile(rundir, tag, i)[:-4] + ".n.npy")[0])
              for i in todo}
    groups = sky3d.make_groups(counts, 10000, 64) if not check else [[i] for i in todo]
    done = 0
    with mp.get_context("fork").Pool(nproc) as pool:
        for n_ in pool.imap_unordered(group, groups, chunksize=1):
            done += n_
            print(f"{done}/{len(todo)} files", flush=True)
    if check:
        for i in todo:
            a = np.load(sky3d.tfile(rundir, tag, i))
            b = np.load(sky3d.tfile(OUT, new, i))
            for k in ("s1", "s1sq", "n1", "t1", "v1"):
                assert np.allclose(a[k], b[k], rtol=1e-12, atol=0), (i, k)
        print("check passed on", len(todo), "files")
    else:
        print("wrote", sky3d.merge(OUT, new))
