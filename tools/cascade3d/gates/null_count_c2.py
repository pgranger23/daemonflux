"""Count thresholds of G2.3 from the Monte Carlo reference alone (GATES.md, section C2).

For each counted part of G2.3 the count that an exactly right solver reaches or
exceeds in 95% of the cases, under two null models: independent Gaussian errors, and
resampling of the runs.  The larger (stricter) of the two is the threshold.  No solver
output is read.

usage: null_count_c2.py REF.npz --outdir DIR [--cap 3] [--nsamp 20000]

REF.npz holds ``s_runs`` and ``n_runs`` of shape (runs, caps, species, bands, cos Z
rows, azimuth bins) in the flux measure, runs of equal size and each normalised on
its own.  Writes ``null_count_c2.md`` and ``null_count_c2.json`` into DIR and refuses
to overwrite them.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c2_stats as st  # noqa: E402

GATED = ("a", "b", "c")
REPORTED = ("d", "f")
NAMES = {"a": "cutoff factor per cell", "b": "East-West harmonic per band and row",
         "c": "horizontal over vertical per band", "d": "flux per cell (reported)",
         "f": "azimuth-averaged cutoff factor per cell (reported)"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--cap", type=int, default=3)
    ap.add_argument("--nsamp", type=int, default=20000)
    a = ap.parse_args(argv)
    for name in ("null_count_c2.md", "null_count_c2.json"):
        if os.path.exists(os.path.join(a.outdir, name)):
            sys.exit("refusing to overwrite %s" % os.path.join(a.outdir, name))

    sha = hashlib.sha256(open(a.ref, "rb").read()).hexdigest()
    d = np.load(a.ref, allow_pickle=False)
    s_runs = np.asarray(d["s_runs"][:, a.cap], float)
    n_runs = np.asarray(d["n_runs"][:, a.cap], float)
    ref = st.reference(s_runs, n_runs)
    res = {"reference": os.path.abspath(a.ref), "sha256": sha, "cap_index": a.cap,
           "runs": int(s_runs.shape[0]), "nsamp": a.nsamp, "parts": {}}
    lines = ["# Count thresholds of G2.3 from the reference alone", "",
             "Reference `%s`, sha256 `%s`, cap index %d, %d runs."
             % (res["reference"], sha, a.cap, s_runs.shape[0]),
             "Command: `null_count_c2.py %s`" % " ".join(argv or sys.argv[1:]), "",
             "| Part | Items counted | Gaussian: median, N5 | Resampling: median, N5 |"
             " Threshold |", "|---|---|---|---|---|"]
    parts = GATED + REPORTED
    rs = st.null_resample(parts, ref, s_runs, n_runs, a.nsamp,
                          np.random.default_rng(2026112012))
    for k, p in enumerate(parts):
        ok = ref[p][2]
        cg = st.null_gauss(p, ref, a.nsamp, np.random.default_rng(2026112011 + 100 * k))
        ng, nr = st.n5(cg), st.n5(rs[p])
        res["parts"][p] = {
            "name": NAMES[p], "items": int(ok.sum()),
            "items_left_out": int((~ok).sum()),
            "gauss_median": float(np.median(cg)), "gauss_n5": ng,
            "resample_median": float(np.median(rs[p])), "resample_n5": nr,
            "threshold": max(ng, nr), "gated": p in GATED}
        lines.append("| (%s) %s | %d (%d left out) | %.0f, %d | %.0f, %d | **%d** |"
                     % (p, NAMES[p], ok.sum(), (~ok).sum(), np.median(cg), ng,
                        np.median(rs[p]), nr, max(ng, nr)))
    val, err, ok = ref["e"]
    lines += ["", "Level of the cutoff factor per band (part e), value and error:", ""]
    lines += ["- band %d: %.4f +- %.4f" % (i, v, e) for i, (v, e) in
              enumerate(zip(val, err))]
    res["level"] = {"value": val.tolist(), "error": err.tolist()}
    val, err, ok = ref["a"]
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = (err / val)[ok]
    lines += ["", "Relative error of the cutoff factor per counted cell: median %.4f, "
              "90th percentile %.4f; cells whose tolerance is the 3%% floor: %d of %d."
              % (np.median(rel), np.percentile(rel, 90),
                 int((2 * rel <= st.FLOOR_A).sum()), rel.size)]
    os.makedirs(a.outdir, exist_ok=True)
    open(os.path.join(a.outdir, "null_count_c2.md"), "w").write("\n".join(lines) + "\n")
    json.dump(res, open(os.path.join(a.outdir, "null_count_c2.json"), "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
