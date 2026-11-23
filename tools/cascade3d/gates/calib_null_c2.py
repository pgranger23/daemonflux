"""Calibration of the count thresholds of G2.3 on synthetic references (no Monte Carlo
data, no solver): the evidence of the amendment of 2026-11-23 in ``GATES.md``.

For R independent synthetic references of twenty runs, the count of the exactly right
solver is compared with the threshold of each candidate null model, each computed from
the reference alone.  A rule is calibrated if the exact solver falls below its
threshold in 5% of the references.  ``signflip`` is ``c2_stats.null_split``.

Models of the reference:
  indep   every tally cell an independent Poisson count
  corr    one set of neutrinos per (species, band, row); independent verdicts
  nested  one set of neutrinos per (species, band, row); the verdicts of the twelve
          azimuth bins are nested (one latent rigidity per neutrino), which is what
          twelve rotated copies of one primary give

usage: calib_null_c2.py MODEL NEV SLOPE R SEED OUT.json [REF.npz synth|ref [CAP]]
       calib_null_c2.py --summary OUT.json [OUT.json ...]
       calib_null_c2.py --calib CALIB.json OUT.json [OUT.json ...]

With REF.npz the number of events comes from a Monte Carlo reference instead of NEV
and SLOPE: per (species, band, cos Z row) the effective number of events of the ring
of twelve azimuth bins, (value / error)^2 of the ring total of ``n`` over the m runs,
times (m - 3) / (m - 1) (the estimate is otherwise too high by that factor on
average).  The cutoff factor is the synthetic one (``synth``) or that of the reference
itself (``ref``): the estimate K clipped to 0..1 and pulled to (N K + 0.5) / (N + 1),
N the effective number of events of the ring, so that a cell in which every event
passed does not become a cell without noise.  The synthetic references always have
twenty runs.

``--calib`` pools runs made with the same model on the same reference (mode ``ref``,
different seeds) into the calibration of the threshold of part (a) at the reference's
own event counts, read by ``null_count_c2.py --calib``: the false-failure rate of the
half-against-half threshold and the number of cells ``delta_a`` by which the threshold
would have to be lowered to bring the rate to 5%.  Reported next to the threshold; the
gate decides on the threshold itself (``GATES.md``, amendment of 2026-11-23).

Runs of 2026-11-20 (SLOPE -2, R 200), outputs ``c2_<model>_<nev>.json`` under
``$N3/c2/manager/calib/``: nested 60 (seed 21), nested 300 (22), indep 300 (23),
corr 300 (24), corr 60 (25), nested 1000 (26), nested 150 (27), indep 1500 (28).
Runs of 2026-11-23 (R 150) under ``$N3/c2/manager/calib/valid/``: the event counts
of the field-free dry run with the synthetic cutoff factor (``truth_s61``,
``truth_s62``) and mode ``ref`` on three synthetic references drawn from it
(``ref71_s81``, ``ref72_s82``, ``ref73_s83``).
"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c2_stats as st  # noqa: E402

NRUN, NSAMP, P = 20, 400, ("a", "b", "c")
NU = NRUN - 1


def summary(files):
    """False-failure rate of the exactly right solver for each rule."""
    for f in files:
        d = json.load(open(f))
        rows = d["rows"]
        print("== %s: model %s, nev %g, slope %g, R %d"
              % (f, d["model"], d["nev"], d["slope"], len(rows)))
        for p in P:
            def g(k):
                return np.array([r[p][k] for r in rows], float)
            ex = g("exact")
            rules = {"first rule: max(gauss, resample)":
                     np.maximum(g("gauss"), g("resample")),
                     "student": g("student"), "plugin": g("plugin"),
                     "signflip (amended rule)": g("signflip")}
            extra = ", floor cells %.0f" % g("floor_cells").mean() if p == "a" else ""
            print(" part %s: items %.0f, exact mean %.1f sd %.1f%s"
                  % (p, g("items").mean(), ex.mean(), ex.std(), extra))
            for name, thr in rules.items():
                print("   %-34s threshold mean %.1f  false-failure rate %.3f"
                      % (name, thr.mean(), (ex < thr).mean()))


def calib(out, files):
    """Pool runs on one reference into the calibration of the threshold of (a)."""
    if os.path.exists(out):
        sys.exit("refusing to overwrite %s" % out)
    ds = [json.load(open(f)) for f in files]
    same = ("model", "profile", "profile_sha256", "nrun", "nsamp")
    if any(d.get(k) != ds[0].get(k) for d in ds for k in same):
        sys.exit("the runs differ in model, reference or sampling")
    seeds = [d["seed"] for d in ds]
    prof = ds[0]["profile"]
    if len(set(seeds)) != len(seeds) or len(prof) < 2 or prof[1] != "ref":
        sys.exit("needs runs of mode ref with different seeds")
    rows = [r["a"] for d in ds for r in d["rows"]]

    def g(k):
        return np.array([r[k] for r in rows], float)
    ex, sf, first = g("exact"), g("signflip"), np.maximum(g("gauss"), g("resample"))
    delta = 0
    while (ex < sf - delta).mean() > 0.05:
        delta += 1
    res = {"model": ds[0]["model"], "references": len(rows), "seeds": seeds,
           "nrun": ds[0]["nrun"], "nsamp": ds[0]["nsamp"],
           "reference": os.path.abspath(prof[0]), "sha256": ds[0]["profile_sha256"],
           "cap_index": int(prof[2]) if len(prof) > 2 else 3,
           "items_mean": float(g("items").mean()),
           "floor_cells_mean": float(g("floor_cells").mean()),
           "exact_mean": float(ex.mean()), "exact_sd": float(ex.std()),
           "split_n5_mean": float(sf.mean()), "first_rule_mean": float(first.mean()),
           "rate_split": float((ex < sf).mean()),
           "rate_first_rule": float((ex < first).mean()),
           "delta_a": int(delta), "rate_after": float((ex < sf - delta).mean())}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


def profile(path, cap, k):
    """Events per azimuth cell and run with no cutoff, and the cutoff factor, from a
    reference: (species, band, row), (species, band, row, azimuth)."""
    d = np.load(path, allow_pickle=False)
    s_runs = np.asarray(d["s_runs"][:, cap], float)
    n_runs = np.asarray(d["n_runs"][:, cap], float)
    m = n_runs.shape[0]
    ring = n_runs.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        err = ring.std(0, ddof=1) / np.sqrt(m)
        neff = np.nan_to_num((ring.mean(0) / err) ** 2, posinf=0.0)
        neff *= (m - 3.0) / (m - 1.0)
        if k is None:
            k = np.clip(np.nan_to_num(st.stat_k(s_runs.mean(0), n_runs.mean(0))), 0, 1)
            k = (neff[..., None] * k + 0.5) / (neff[..., None] + 1.0)
    return neff / (12.0 * NRUN), k


if sys.argv[1] == "--summary":
    summary(sys.argv[2:])
    sys.exit(0)
if sys.argv[1] == "--calib":
    calib(sys.argv[2], sys.argv[3:])
    sys.exit(0)
model, nev, slope, R, seed, out = sys.argv[1:7]
nev, slope, R, seed = float(nev), float(slope), int(R), int(seed)
PROFILE = sys.argv[7:10]


def truth():
    sp = np.array([1.0, 0.8, 2.0, 1.7])[:, None, None]
    phi_0 = sp * np.logspace(0, slope, 10)[None, :, None] * np.linspace(1.5, 1.0, 10)
    az = st.az_centres(12)
    amp = np.linspace(0.3, 0.02, 10)[None, :, None, None]
    k = (0.4 + 0.06 * np.arange(10))[None, :, None, None] * (1 - amp * np.sin(az))
    k = k * np.linspace(0.9, 1.0, 10)[None, None, :, None] * np.ones((4, 1, 1, 1))
    return phi_0, k


PHI0, K = truth()
SCALE = nev / PHI0.max()
if PROFILE:
    PHI0, K = profile(PROFILE[0], int(PROFILE[2]) if len(PROFILE) > 2 else 3,
                      K if PROFILE[1] == "synth" else None)
    SCALE = 1.0
ORDER = np.argsort(K, -1)
KS = np.take_along_axis(K, ORDER, -1)
PV = np.diff(np.concatenate([np.zeros(K.shape[:-1] + (1,)), KS,
                             np.ones(K.shape[:-1] + (1,))], -1), axis=-1)
INV = np.argsort(ORDER, -1)


def gen(rng):
    lam = PHI0 * SCALE                     # events per azimuth cell with no cutoff
    if model == "indep":
        s = rng.poisson(lam[..., None] * K, (NRUN,) + K.shape)
        n = rng.poisson(lam[..., None] * np.ones(12), (NRUN,) + K.shape)
        return s / SCALE, n / SCALE
    ntot = rng.poisson(12 * lam, (NRUN,) + lam.shape)
    n = rng.multinomial(ntot, np.full(12, 1 / 12.0))
    if model == "corr":
        s = rng.binomial(ntot[..., None], K) / 12.0
    else:
        c = np.cumsum(rng.multinomial(ntot, PV), -1)[..., :12]
        s = np.take_along_axis(c, np.broadcast_to(INV, c.shape), -1) / 12.0
    return s / SCALE, n / SCALE


def ref_parts(s_runs, n_runs):
    o, cells = {}, st.used_cells(s_runs)
    for p in P:
        val, err = st.jackknife(st.STATS[p], s_runs, n_runs)
        ok = np.isfinite(val) & np.isfinite(err)
        ok = ok.all(-1) if p == "b" else (ok & cells if p == "a" else ok)
        o[p] = (val, err, ok)
    return o


def null_draw(p, ref, rng, kind):
    """Independent items.  student: value + err * t.  plugin: a new reference, value
    redrawn normal with the jackknife error, error redrawn with nu degrees of freedom;
    the exactly right solver sits at the old value."""
    val, err, ok = ref[p]
    o = np.empty(NSAMP, int)
    for i in range(NSAMP):
        with np.errstate(invalid="ignore"):
            if kind == "student":
                test = val + err * rng.standard_t(NU, val.shape)
                o[i] = (st.inside(p, test, val, err) & ok).sum()
            else:
                v2 = val + err * rng.standard_normal(val.shape)
                e2 = err * np.sqrt(rng.chisquare(NU, val.shape) / NU)
                o[i] = (st.inside(p, val, v2, e2) & ok).sum()
    return o


def null_signflip(ref, s_runs, n_runs, rng):
    """Half of the runs against the other half: d = (stat(A) - stat(B)) / 2 has the
    distribution of (reference - truth) and keeps the correlations between items; the
    error with d removed, e^2 = (nu err^2 - d^2) / (nu - 1), is independent of it."""
    m = s_runs.shape[0]
    h = m // 2
    ts, tn = s_runs.sum(0), n_runs.sum(0)
    o = {p: np.empty(NSAMP, int) for p in P}
    for i in range(NSAMP):
        idx = rng.permutation(m)[:h]
        sa, na = s_runs[idx].sum(0), n_runs[idx].sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            for p in P:
                val, err, ok = ref[p]
                fn = st.STATS[p]
                d = (fn(sa / h, na / h) - fn((ts - sa) / h, (tn - na) / h)) / 2
                e = np.sqrt(np.clip((NU * err**2 - d**2) / (NU - 1), 0, None))
                o[p][i] = (st.inside(p, val, val + d, e) & ok).sum()
    return o


rng = np.random.default_rng(seed)
test = st.solver_tallies(PHI0[..., None] * K, PHI0)
rows = []
for r in range(R):
    s_runs, n_runs = gen(rng)
    with np.errstate(invalid="ignore", divide="ignore"):
        ref = ref_parts(s_runs, n_runs)
        res = st.null_resample(P, ref, s_runs, n_runs, NSAMP, rng)
        sf = null_signflip(ref, s_runs, n_runs, rng)
        row = {}
        for p in P:
            val, err, ok = ref[p]
            exact = int((st.inside(p, st.STATS[p](*test), val, err) & ok).sum())
            g = st.null_gauss(p, ref, NSAMP, rng)
            t = null_draw(p, ref, rng, "student")
            pl = null_draw(p, ref, rng, "plugin")
            row[p] = {"items": int(ok.sum()), "exact": exact, "gauss": st.n5(g),
                      "resample": st.n5(res[p]), "student": st.n5(t),
                      "plugin": st.n5(pl), "signflip": st.n5(sf[p]),
                      "gauss_med": float(np.median(g)),
                      "student_med": float(np.median(t)),
                      "plugin_med": float(np.median(pl)),
                      "signflip_med": float(np.median(sf[p])),
                      "resample_med": float(np.median(res[p]))}
            if p == "a":
                rel = (err / np.abs(val))[ok]
                row[p]["floor_cells"] = int((2 * rel <= st.FLOOR_A).sum())
    rows.append(row)
    if r % 20 == 19:
        print(r + 1, flush=True)
sha = hashlib.sha256(open(PROFILE[0], "rb").read()).hexdigest() if PROFILE else None
json.dump({"model": model, "nev": nev, "slope": slope, "R": R, "seed": seed,
           "nrun": NRUN, "nsamp": NSAMP, "profile": PROFILE, "profile_sha256": sha,
           "rows": rows}, open(out, "w"))
