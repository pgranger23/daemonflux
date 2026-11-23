"""Count thresholds of G2.3 from the Monte Carlo reference alone (GATES.md, section C2).

For each counted part of G2.3 the count that an exactly right solver reaches or
exceeds in 95% of the cases.  The threshold is that of the null model of the amendment
of 2026-11-23: half of the runs against the other half (``c2_stats.null_split``).  The
two null models of the first text (independent Gaussian errors, resampling of the
runs) and the larger of their two numbers, the first rule, are computed and written
next to it.  No solver output is read.

usage: null_count_c2.py REF.npz --outdir DIR [--cap 3] [--nsamp 20000]
                           [--calib CALIB.json]

CALIB.json is the calibration of the threshold of (a) at the reference's own event
counts (``calib_null_c2.py --calib``, made from the same reference and cap): the
false-failure rate of the threshold and the number of cells ``delta_a`` that would
bring it to 5%.  Both are written next to the threshold and change nothing in it.

REF.npz holds ``s_runs`` and ``n_runs`` of shape (runs, caps, species, bands, cos Z
rows, azimuth bins) in the flux measure, an even number of runs of equal size, each
normalised on its own.  Writes ``null_count_c2.md`` and ``null_count_c2.json`` into
DIR and refuses to overwrite them.
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
#: measured false-failure rate of the threshold of (a) above which the result of (a)
#: carries a caveat for the PI (5% is intended)
RATE_FLAG = 0.10


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--cap", type=int, default=3)
    ap.add_argument("--nsamp", type=int, default=20000)
    ap.add_argument("--calib")
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
             "Threshold: N5 of the half-against-half null model (amendment of "
             "2026-11-23). First rule: the larger N5 of the Gaussian and the "
             "resampling null models (text of 2026-11-20, commit 5fa5953).", "",
             "| Part | Items counted | Half against half: median, N5 | Gaussian: "
             "median, N5 | Resampling: median, N5 | Threshold | First rule |",
             "|---|---|---|---|---|---|---|"]
    parts = GATED + REPORTED
    rs = st.null_resample(parts, ref, s_runs, n_runs, a.nsamp,
                          np.random.default_rng(2026112012))
    sp = st.null_split(parts, ref, s_runs, n_runs, a.nsamp,
                       np.random.default_rng(2026112013))
    for k, p in enumerate(parts):
        ok = ref[p][2]
        cg = st.null_gauss(p, ref, a.nsamp, np.random.default_rng(2026112011 + 100 * k))
        ng, nr, ns = st.n5(cg), st.n5(rs[p]), st.n5(sp[p])
        res["parts"][p] = {
            "name": NAMES[p], "items": int(ok.sum()),
            "items_left_out": int((~ok).sum()),
            "split_median": float(np.median(sp[p])), "split_n5": ns,
            "gauss_median": float(np.median(cg)), "gauss_n5": ng,
            "resample_median": float(np.median(rs[p])), "resample_n5": nr,
            "threshold": ns, "threshold_first_rule": max(ng, nr),
            "gated": p in GATED}
        lines.append("| (%s) %s | %d (%d left out) | %.0f, %d | %.0f, %d | %.0f, %d | "
                     "**%d** | %d |"
                     % (p, NAMES[p], ok.sum(), (~ok).sum(), np.median(sp[p]), ns,
                        np.median(cg), ng, np.median(rs[p]), nr, ns, max(ng, nr)))
    val, err, ok = ref["e"]
    lines += ["", "Level of the cutoff factor per band (part e), value and error:", ""]
    lines += ["- band %d: %.4f +- %.4f" % (i, v, e) for i, (v, e) in
              enumerate(zip(val, err))]
    res["level"] = {"value": val.tolist(), "error": err.tolist()}
    val, err, ok = ref["a"]
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = (err / val)[ok]
    nfloor = int((2 * rel <= st.FLOOR_A).sum())
    res["a_floor_cells"], res["a_floor_share"] = nfloor, nfloor / max(rel.size, 1)
    lines += ["", "Relative error of the cutoff factor per counted cell: median %.4f, "
              "90th percentile %.4f; cells whose tolerance is the 3%% floor: %d of %d "
              "(share %.3f)." % (np.median(rel), np.percentile(rel, 90), nfloor,
                                 rel.size, res["a_floor_share"])]
    cal = None
    if a.calib:
        c = json.load(open(a.calib))
        if c["sha256"] != sha or c["cap_index"] != a.cap:
            sys.exit("the calibration is not that of this reference and cap")
        keep = ("model", "references", "seeds", "rate_split", "rate_first_rule",
                "delta_a", "rate_after", "exact_mean", "exact_sd")
        cal = {k: c[k] for k in keep}
        cal["file"] = os.path.abspath(a.calib)
        cal["file_sha256"] = hashlib.sha256(open(a.calib, "rb").read()).hexdigest()
        cal["threshold_less_delta"] = res["parts"]["a"]["threshold"] - c["delta_a"]
        lines += ["", "Threshold of (a) at the reference's own event counts (`%s`, "
                  "model %s, %d synthetic references): an exactly right solver falls "
                  "below it in %.1f%% of the cases (5%% intended; first rule %.1f%%). "
                  "Lowered by %d cells, to %d, the rate is %.1f%%. Reported: the gate "
                  "decides on the threshold of the table."
                  % (cal["file"], c["model"], c["references"], 100 * c["rate_split"],
                     100 * c["rate_first_rule"], c["delta_a"],
                     cal["threshold_less_delta"], 100 * c["rate_after"])]
    else:
        lines += ["", "Threshold of (a) at the reference's own event counts: not "
                  "measured (no `--calib`)."]
    res["a_calibration"] = cal
    res["a_calibration_flag"] = bool(cal is None or cal["rate_split"] > RATE_FLAG)
    if cal is not None and res["a_calibration_flag"]:
        lines += ["**The measured rate is above %.0f%%: the result of (a) goes to the "
                  "PI with this caveat.**" % (100 * RATE_FLAG)]
    os.makedirs(a.outdir, exist_ok=True)
    open(os.path.join(a.outdir, "null_count_c2.md"), "w").write("\n".join(lines) + "\n")
    json.dump(res, open(os.path.join(a.outdir, "null_count_c2.json"), "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
