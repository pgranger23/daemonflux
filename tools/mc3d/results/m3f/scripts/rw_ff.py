"""Exact re-evaluation of a finished run under another primary model
(milestone 3d; generalises milestone 3c's H3a-only script).

The importance weight of a primary is linear in the primary flux, so changing
the model (or its energy variable) multiplies every entry of a primary by

    f = J_new(group, T [, nucleon identity]) / J_old(group, T [, identity])

(``primaries.reweight_factor``) -- exactly what re-running with the same seeds
would give.  ``T`` and the nucleon identity of each pending primary are
regenerated from the c-file's seed (``sky.primary_sample``: the same two
vectorised RNG draws; the pending primary is matched to its row by the stored
rigidity, which must match exactly).  The cutoff verdicts of the traced copy
states are not stored, so they are recomputed ONCE per group of files
(``cutoff.trace_gated``, deterministic, same model and date) and shared by all
targets; the tallies are replayed per target with ``sky3d.tally_entries``.

    python reweight_replay.py SRC_RUNDIR TAG NPROC TARGET [TARGET ...]
    python reweight_replay.py SRC_RUNDIR TAG NPROC --check

``TARGET`` = ``MODEL`` or ``MODEL:conv`` (conv = native|total|kinetic, default
native), e.g. ``H3A:total GSFG GH GST``.  The source run's own model and energy
variable (manifest) are the "old" ones.  Each target is written to the task
directory ``$M/sky/<TAG>_rw_<MODEL>_<conv>_tasks`` (conv resolved to
total/kinetic) and merged to ``..._total.npz`` -- what ``extract_strata.py``
reads; the source run is only read.  Per-stratum diagnostics (re-weighting
factor range, effective sample size) go to ``$M/work/rwdiag_<TAG>.npz``.

``--check`` replays a few files with the old model as the target (f = 1) and
compares with the original t-files (must be identical).
"""
import multiprocessing as mp
import os
import pickle
import shutil
import sys

import numpy as np

import cutoff
import primaries as prim
import sky
import sky3d

args = [a for a in sys.argv[1:] if not a.startswith("--")]
rundir, tag, nproc = args[0], args[1], int(args[2])
check = "--check" in sys.argv
man = dict(np.load(os.path.join(sky3d.task_dir(rundir, tag), "manifest.npz")))
ntask = int(man["ntask"][0])
os_ = man["opts"]
opts = sky3d.Options(cutoff=bool(os_[0]), bend_primary=bool(os_[1]),
                     muon_field=bool(os_[2]), r_thr=float(os_[3]),
                     n_az=int(os_[4]), primary_energy="kinetic")
# the source run's model and energy variable (9 = kinetic, 12 = [.., index])
OLD_NAME = str(man["model"][0])
OLD_CONV = "kinetic" if len(os_) < 12 else prim.ENERGY_CONVENTIONS[int(os_[11])]
OLD = (prim.primary_model(OLD_NAME), OLD_CONV)
OUT = os.path.join(os.environ["M"], "sky")
from scoring import NestedCaps  # noqa: E402
caps = NestedCaps(*sky.SITES[str(man["site"][0])], tuple(man["caps"]))
grid = sky3d.Grid(caps, sky.E_BINS, sky.CZ_BINS, sky.AZ_BINS)


GFIX = None


def gfix_factor(T, e):
    """Approximate correction for the milestone-1..3c energy-guard bug of
    ``pool.EventPool.draw`` (``build_gfix.py``): the ratio of the 1D neutrino
    yield with the fixed guard to the old one, a function of the primary's
    kinetic energy per nucleon ``T`` and the neutrino energy ``e``."""
    from scipy.interpolate import RegularGridInterpolator
    global GFIX
    if not isinstance(GFIX, RegularGridInterpolator):
        z = np.load(os.environ["GFIX_FILE"])
        GFIX = RegularGridInterpolator((np.log(z["ep"]), np.log(z["ec"])), z["g"],
                                       bounds_error=False, fill_value=None)
        GFIX.lo = (np.log(z["ep"][0]), np.log(z["ec"][0]))
        GFIX.hi = (np.log(z["ep"][-1]), np.log(z["ec"][-1]))
    x = np.log(np.clip(T, np.exp(GFIX.lo[0]), np.exp(GFIX.hi[0])))
    y = np.log(np.clip(e, np.exp(GFIX.lo[1]), np.exp(GFIX.hi[1])))
    return GFIX(np.stack([x, y], -1))


def parse_target(s):
    gfix = s.endswith("+gfix")
    s = s[:-5] if gfix else s
    t = _parse_target(s)
    if gfix:
        t = (t[0], t[1], t[2], t[3] + "_gfix")
    return t


class FFModel:
    """Force-field re-modulation of ``base`` from phi0 to phi1 [GV] (Gleeson-Axford):
    J1(T) = J0(T + dPhi) p^2(T) / p^2(T + dPhi), dPhi = (Z/A)(phi1 - phi0) per nucleon,
    p^2(x) = x^2 + 2 m x (exact for J0 = J_LIS modulated by phi0)."""
    nucleon_level = False

    def __init__(self, base, phi0, phi1):
        self.base, self.phi0, self.phi1 = base, phi0, phi1
        self.native_energy = prim.native_convention(base)
        self.name = f"FF({phi1})"

    def group_split(self, gid, e_nucleon, convention=None):
        a_, z_ = prim._AZ[gid]
        t = np.atleast_1d(np.asarray(e_nucleon, float))
        dphi = (z_ / a_) * (self.phi1 - self.phi0)
        t2 = np.maximum(t + dphi, 1e-3)
        m = prim.M_NUCLEON
        sc = (t * t + 2 * m * t) / (t2 * t2 + 2 * m * t2)
        jp, jn = prim.group_split(self.base, gid, t2, convention)
        return jp * sc, jn * sc


def _parse_target(s):
    if s.startswith("FF"):
        phi1 = float(s[2:])
        phi0 = float(os.environ.get("FF_PHI0", "0.55"))
        conv = OLD_CONV
        return ("FF", conv, (FFModel(OLD[0], phi0, phi1), conv), f"FF{phi1:g}")
    name, _, conv = s.partition(":")
    if name.upper().startswith("GSFG") and conv.upper() in ("GST", "GH", "H3A"):
        name, conv = s, "native"       # GSFG:REF names the composition model
    mdl = prim.primary_model(name)
    c = prim.resolve_convention(conv or "native", mdl)
    return name, c, (mdl, c), f"{name.replace(':', '-')}_{c}"


def primaries_of(d):
    """``(T, is_p)`` of the pending primaries of a c-file, regenerated from
    its seed (and checked against the stored rigidities)."""
    gname, k = str(d["group"]), int(d["k"])
    n = int(d["meta"][0])
    _nm, a_mass, z, _gid = sky.GROUPS[gname]
    e1, e2 = float(sky.STRATA[k]), float(sky.STRATA[k + 1])
    rng = np.random.default_rng(int(d["seed"]))
    e_nuc = np.exp(rng.uniform(np.log(e1), np.log(e2), n))
    if hasattr(OLD[0], "group_split"):            # GSF-groups: its own p/n split
        jp, jn = OLD[0].group_split(sky.GROUPS[gname][3], e_nuc, OLD[1])
        j = jp + jn
        is_p = rng.random(n) < np.where(j > 0, jp / np.where(j > 0, j, 1.0), 0.0)
    else:
        is_p = rng.random(n) < (z / a_mass)
    rig = cutoff.rigidity_gv(e_nuc, a_mass, z)
    order = np.argsort(rig)
    pos = np.searchsorted(rig[order], d["R"])
    idx = order[np.minimum(pos, n - 1)]
    if not np.array_equal(rig[idx], np.asarray(d["R"], float)):
        raise RuntimeError("pending primaries do not match the regenerated "
                           "sample (a different primary_sample?)")
    return e_nuc[idx], is_p[idx]


def _load(i):
    with open(sky3d.cfile(rundir, tag, i), "rb") as fh:
        return pickle.load(fh)


def _scaled(ent, f, T=None):
    """A copy of ``ent`` with every weight times ``f[prim]`` (and, with the
    primaries' ``T``, the energy-guard correction ``gfix_factor(T, e_nu)``)."""
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in ent.items()}
    e3, e1 = out["3d"], out["1d"]
    if T is not None and "3dcf" in out and len(out["3dcf"]["flat"]):
        raise NotImplementedError("gfix with the mu-cf entries")
    if len(e3["flat"]):
        g = 1.0 if T is None else gfix_factor(T[e3["prim"]], e3["e"])
        for k in ("w0", "wn"):
            e3[k] = e3[k] * f[e3["prim"]] * g
    if len(e1["flat"]):
        g = 1.0 if T is None else gfix_factor(T[e1["prim"]], e1["e"])
        for k in ("wt", "wv"):
            e1[k] = e1[k] * f[e1["prim"]] * g
    if "3dcf" in out and len(out["3dcf"]["flat"]):
        out["3dcf"]["w0"] = out["3dcf"]["w0"] * f[out["3dcf"]["prim"]]
    return out


def _diag(d, f):
    """Per-stratum re-weighting diagnostics of one c-file: c_i = the
    primary's no-cutoff 3D weight (its contribution to the tallies)."""
    e3, e1 = d["ent"]["3d"], d["ent"]["1d"]
    npr = len(f)
    c = np.bincount(e3["prim"], e3["w0"], minlength=npr) if len(e3["flat"]) \
        else np.zeros(npr)
    c1 = np.bincount(e1["prim"], e1["wt"], minlength=npr) if len(e1["flat"]) \
        else np.zeros(npr)
    cc = c + 0.0
    cf = cc * f
    return {"n": npr, "n3": int((c > 0).sum()), "S1": cc.sum(),
            "S2": (cc ** 2).sum(), "F1": cf.sum(), "F2": (cf ** 2).sum(),
            "T1": c1.sum(), "TF1": (c1 * f).sum(),
            "fmin": float(f.min()) if npr else np.nan,
            "fmax": float(f.max()) if npr else np.nan,
            "fz": int((f == 0).sum())}


def group(job):
    """Re-weight and replay a group of consecutive c-files for all targets
    with ONE pooled trace of their copy states."""
    tis, targets = job
    tis = [i for i in tis
           if any(not os.path.exists(sky3d.tfile(OUT, f"{tag}_rw_{t[3]}", i))
                  for t in targets)]
    if not tis:
        return []
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
    diags = []
    for i, d, ns in zip(tis, datas, nst):
        v = verdict[off:off + ns]
        off += ns
        T, is_p = primaries_of(d)
        gid = sky.GROUPS[str(d["group"])][3]
        A = sky3d.verdicts(d, v, opts.cutoff, opts.n_az)
        old = dict(np.load(sky3d.tfile(rundir, tag, i), allow_pickle=False))
        for (_nm, _c, new, label) in targets:
            tg = f"{tag}_rw_{label}"
            if os.path.exists(sky3d.tfile(OUT, tg, i)):
                continue
            f = prim.reweight_factor(new, OLD, gid, T, is_p)
            gf = label.endswith("_gfix")
            tal = sky3d.tally_entries(grid, _scaled(d["ent"], f, T if gf else None), A)
            dd = tal.to_dict()
            for k in old:
                if (k.startswith("cone") or k in ("meta", "group", "m3b_counts",
                                                  "m3b_count_names",
                                                  "m3b_opts")):
                    dd[k] = old[k]
            sky._atomic_savez(sky3d.tfile(OUT, tg, i), dd, compressed=True)
            dg = _diag(d, f)
            dg.update(label=label, group=str(d["group"]), k=int(d["k"]), ti=i)
            diags.append(dg)
    return diags


if __name__ == "__main__":
    if check:
        targets = [("OLD", OLD_CONV, OLD, "chk")]
        todo = list(range(0, ntask, max(1, ntask // 6)))
    else:
        targets = [parse_target(s) for s in args[3:]]
        todo = list(range(ntask))
        if os.environ.get("RW_NFILES"):          # timing tests only
            todo = todo[:int(os.environ["RW_NFILES"])]
    for _nm, _c, _m, label in targets:
        tg = f"{tag}_rw_{label}"
        os.makedirs(sky3d.task_dir(OUT, tg), exist_ok=True)
        shutil.copy(os.path.join(sky3d.task_dir(rundir, tag), "manifest.npz"),
                    os.path.join(sky3d.task_dir(OUT, tg), "manifest.npz"))
    counts = {i: int(np.load(sky3d.cfile(rundir, tag, i)[:-4] + ".n.npy")[0])
              for i in todo}
    groups = [[i] for i in todo] if check else sky3d.make_groups(counts, 10000, 64)
    jobs = [(g, targets) for g in groups]
    done, alld = 0, []
    with mp.get_context("fork").Pool(nproc) as pool:
        for r in pool.imap_unordered(group, jobs, chunksize=1):
            alld += r
            done += 1
            if done % 10 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} groups", flush=True)
    if check:
        for i in todo:
            a = np.load(sky3d.tfile(rundir, tag, i))
            b = np.load(sky3d.tfile(OUT, f"{tag}_rw_chk", i))
            for k in ("s1", "s1sq", "n1", "t1", "v1"):
                assert np.allclose(a[k], b[k], rtol=1e-12, atol=0), (i, k)
        print("check passed on", len(todo), "files")
    else:
        for _nm, _c, _m, label in (targets if (len(todo) == ntask and not os.environ.get("RW_NOMERGE")) else []):
            print("wrote", sky3d.merge(OUT, f"{tag}_rw_{label}"))
        if alld:
            keys = ("label", "group", "k", "ti", "n", "n3", "S1", "S2", "F1",
                    "F2", "T1", "TF1", "fmin", "fmax", "fz")
            np.savez(os.path.join(os.environ["M"], "work",
                                  f"rwdiag_{tag}_{len(alld)}.npz"),
                     **{k: np.array([x[k] for x in alld]) for k in keys})
