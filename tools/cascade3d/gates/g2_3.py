"""Gate G2.3 (cutoff and azimuth against the Monte Carlo); writes G2.3.md / G2.3.json.

Written 2026-11-20, before the reference of G2.3 is complete and before any C2 solver
exists.  Gate text: ``tools/cascade3d/GATES.md``, section C2.  The statistics are those
of ``c2_stats.py``, the same functions that give the count thresholds
(``null_count_c2.py``).

Inputs:

- ``--ref``: the combined reference: ``s_runs``, ``n_runs`` (and, if present,
  ``t_runs``, ``us_runs``, ``un_runs``) of shape (runs, caps, species, bands, cos Z
  rows, azimuth bins), with ``e_band_edges_GeV``, ``cosz_edges_down``,
  ``azimuth_edges_deg``, ``cap_theta_deg`` and ``species``.
- ``--null``: ``null_count_c2.json`` of the same reference (its sha256 is checked) for
  cap index 3: the thresholds N_a, N_b, N_c.
- ``--solver``: the assembled solver cells, unit of ``cell_<species>`` of
  ``run_solve.py`` (the unit of the reference):

  - ``phi_cut`` (caps, species, bands, rows, azimuth bins): with the cutoff;
  - ``phi_0`` (caps, species, bands, rows): same assembly, transmission 1;
  - ``e_band_edges_GeV``, ``cosz_edges``, ``azimuth_edges_deg``, ``cap_theta_deg``,
    ``species``: checked against the reference;
  - optional ``phi_straight_0`` (species, bands, rows): straight-ahead limit with no
    cutoff, for part (f);
  - optional ``phi_cut_up`` (caps, species, bands, 2, azimuth bins) and ``phi_0_up``
    (caps, species, bands, 2): the rows cos Z -0.2 to -0.1 and -0.1 to 0, in this
    order, for part (h).

- ``--null5``: optional thresholds for the cap of 5 degrees (``null_count_c2.py
  --cap 1``), quoted in part (g).

PASS = the counts of (a), (b) and (c) in the cap of 10 degrees each reach their
threshold.  Exit status 0 = PASS, 1 = FAIL, 2 = input error.  The result files are
never overwritten.

Usage::

    $PY tools/cascade3d/gates/g2_3.py --ref REF.npz --null null_count_c2.json \
        --solver SOLVER.npz --outdir DIR [--null5 NULL5.json] [--name G2.3]
"""

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c2_stats as st  # noqa: E402

SPECIES = ("nue", "antinue", "numu", "antinumu")
CAP, CAP5 = 3, 1
ROWS_UP = (8, 9)
GATED = ("a", "b", "c")
NAMES = {"a": "cutoff factor per cell", "b": "East-West harmonic per band and row",
         "c": "horizontal over vertical per band"}

GATE = """**G2.3 Cutoff and azimuth against the Monte Carlo** (`GATES.md`, section C2).
Reference: `sky3d.py` at Kamioka, cutoff on, no bending, twenty runs; tallies `s` and
`n`, flux measure, cap of 10°. K = Φ_cut / Φ_0; K_MC = `s`(cell) / (Σ_az `n` / 12);
σ = delete-one-run jackknife.
- (a) gated: |K_solver − K_MC| ≤ max(0.03 K_MC, 2σ) in at least N_a of the cells
  counted (a cell is counted if `s` is filled in at least half of the runs).
- (b) gated: per band and cos Z row, the four species summed, a_c = Σ Φ_cut cos(az) /
  Σ Φ_cut and a_s likewise with sin(az); both within max(0.01, 2σ), in at least N_b of
  the 100 pairs.
- (c) gated: per band, H = K(cos Z 0 to 0.2) / K(cos Z 0.8 to 1) within max(2%, 2σ),
  in at least N_c of the 10 bands.
N_a, N_b, N_c: the counts an exactly right solver reaches in 95% of the cases, from
the reference alone (`null_count_c2.py`).
Reported: (d) r = Φ_cut,solver / `s`; (e) level of K per band, a band outside
max(2%, 2σ) triggers one diagnosis cycle; (f) plan wording, solver 3D with the cutoff
over its straight-ahead limit with no cutoff against `s/t`; (g) (a) to (c) for the cap
of 5°; (h) (a) for the up-going rows cos Z −0.2 to 0."""


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _bad(msg):
    print("input error: " + msg, file=sys.stderr)
    raise SystemExit(2)


def check_axes(ref, sol):
    """The solver's cells must be the reference's cells."""
    for kr, ks in (("e_band_edges_GeV", "e_band_edges_GeV"),
                   ("cosz_edges_down", "cosz_edges"),
                   ("azimuth_edges_deg", "azimuth_edges_deg"),
                   ("cap_theta_deg", "cap_theta_deg")):
        if kr not in ref.files or ks not in sol.files:
            _bad("missing %s (reference) or %s (solver)" % (kr, ks))
        x, y = np.asarray(ref[kr], float), np.asarray(sol[ks], float)
        if x.shape != y.shape or not np.allclose(x, y, rtol=1e-6, atol=1e-9):
            _bad("%s differs between the reference and the solver" % ks)
    for name, d in (("reference", ref), ("solver", sol)):
        if tuple(str(x) for x in d["species"]) != SPECIES:
            _bad("species order of the %s is not %s" % (name, SPECIES))


def evaluate(s_runs, n_runs, phi_cut, phi_0, parts=("a", "b", "c", "d", "e")):
    """{part: solver value, Monte Carlo value, error, counted, inside} for one cap."""
    ref = st.reference(s_runs, n_runs)
    test = st.solver_tallies(phi_cut, phi_0)
    out = {}
    for p in parts:
        val, err, ok = ref[p]
        tv = st.STATS[p](*test)
        with np.errstate(invalid="ignore"):
            ins = st.inside(p, tv, val, err) & ok
        out[p] = {"solver": tv, "mc": val, "err": err, "counted": ok, "inside": ins}
    return out


def cutoff_factor(s_runs, n_runs, phi_cut, phi_0):
    """Part (a) alone, as in :func:`evaluate`, for any number of cos Z rows."""
    val, err = st.jackknife(st.stat_k, s_runs, n_runs)
    ok = np.isfinite(val) & np.isfinite(err) & st.used_cells(s_runs)
    tv = st.stat_k(*st.solver_tallies(phi_cut, phi_0))
    with np.errstate(invalid="ignore"):
        ins = st.inside("a", tv, val, err) & ok
    return {"solver": tv, "mc": val, "err": err, "counted": ok, "inside": ins}


def _outside_by_axis(e):
    out = e["counted"] & ~e["inside"]
    names = ("species", "band", "row", "azimuth")
    return {n: out.sum(tuple(j for j in range(4) if j != i)).tolist()
            for i, n in enumerate(names)}


def _cells(mask):
    return [[int(i) for i in idx] for idx in np.argwhere(mask)]


def _band_labels(edges):
    return ["%.3g-%.3g" % (a, b) for a, b in zip(edges[:-1], edges[1:])]


def _mean_r(phi_cut, s_runs):
    """Mean over cos Z and azimuth of r = solver / s per (species, band), counted
    cells only, and the number of cells within max(3%, 2 sigma)."""
    val, err, ok = st.reference(s_runs, s_runs)["d"]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(ok, phi_cut / val, np.nan)
        ins = st.inside("d", phi_cut, val, err) & ok
    return np.nanmean(r, axis=(2, 3)), int(ins.sum()), int(ok.sum()), r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--null", required=True)
    ap.add_argument("--solver", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--null5")
    ap.add_argument("--name", default="G2.3")
    a = ap.parse_args(argv)
    paths = [os.path.join(a.outdir, a.name + ext) for ext in (".md", ".json")]
    for p in paths:
        if os.path.exists(p):
            _bad("refusing to overwrite %s" % p)

    ref = np.load(a.ref, allow_pickle=False)
    sol = np.load(a.solver, allow_pickle=False)
    null = json.load(open(a.null))
    sha = _sha(a.ref)
    if null["sha256"] != sha or null["cap_index"] != CAP:
        _bad("the thresholds are not those of this reference and of cap index %d" % CAP)
    thr = {p: int(null["parts"][p]["threshold"]) for p in GATED}
    check_axes(ref, sol)
    s_runs, n_runs = np.asarray(ref["s_runs"], float), np.asarray(ref["n_runs"], float)
    phi_cut, phi_0 = np.asarray(sol["phi_cut"], float), np.asarray(sol["phi_0"], float)
    if phi_cut.shape != s_runs.shape[1:] or phi_0.shape != s_runs.shape[1:-1]:
        _bad("shape of phi_cut %s or phi_0 %s" % (phi_cut.shape, phi_0.shape))
    if not (np.isfinite(phi_cut).all() and np.isfinite(phi_0).all()
            and (phi_0 > 0).all() and (phi_cut >= 0).all()):
        _bad("phi_cut or phi_0 holds a negative or non-finite value")
    bands = _band_labels(np.asarray(ref["e_band_edges_GeV"], float))

    ev = evaluate(s_runs[:, CAP], n_runs[:, CAP], phi_cut[CAP], phi_0[CAP])
    items = {p: int(ev[p]["counted"].sum()) for p in GATED}
    counts = {p: int(ev[p]["inside"].sum()) for p in GATED}
    for p in GATED:
        if items[p] != null["parts"][p]["items"]:
            _bad("part (%s): %d items counted, the thresholds file has %d"
                 % (p, items[p], null["parts"][p]["items"]))
    passed = {p: counts[p] >= thr[p] for p in GATED}
    ok = all(passed.values())
    n95 = math.ceil(0.95 * items["a"])
    res = {"gate": "G2.3", "pass": bool(ok), "command": " ".join(argv or sys.argv[1:]),
           "reference": os.path.abspath(a.ref), "reference_sha256": sha,
           "null": os.path.abspath(a.null), "null_sha256": _sha(a.null),
           "solver": os.path.abspath(a.solver), "solver_sha256": _sha(a.solver),
           "runs": int(s_runs.shape[0]), "thresholds": thr, "items": items,
           "counts": counts, "passed": {p: bool(v) for p, v in passed.items()},
           "a_95_percent": {"needed": n95, "met": bool(counts["a"] >= n95)},
           "a_not_counted": _cells(~ev["a"]["counted"]),
           "a_outside_by_axis": _outside_by_axis(ev["a"]),
           "b_outside_by_band":
               (ev["b"]["counted"] & ~ev["b"]["inside"]).sum(1).tolist()}
    for p in ("a", "b", "c", "e"):
        res["part_" + p] = {k: np.asarray(v).tolist() for k, v in ev[p].items()}

    L = ["# %s: %s" % (a.name, "PASS" if ok else "FAIL"), "",
         "Command: `g2_3.py %s`" % res["command"], "",
         "- Reference `%s`, sha256 `%s`, %d runs." % (res["reference"], sha,
                                                    res["runs"]),
         "- Thresholds `%s`, sha256 `%s`." % (res["null"], res["null_sha256"]),
         "- Solver `%s`, sha256 `%s`." % (res["solver"], res["solver_sha256"]), "",
         GATE, "", "## Gated parts, cap of 10°", "",
         "| Part | Items counted | Inside | Needed | Result |", "|---|---|---|---|---|"]
    for p in GATED:
        L.append("| (%s) %s | %d | %d | %d | **%s** |" % (
            p, NAMES[p], items[p], counts[p], thr[p], "PASS" if passed[p] else "FAIL"))
    nc = res["a_not_counted"]
    L += ["", "(a) against 95%% of the cells counted, the plan's wording: %d needed, "
          "%s." % (n95, "met" if counts["a"] >= n95 else "not met"),
          "(a) cells not counted (species, band, row, azimuth bin): %d%s"
          % (len(nc), "" if not nc else ": " + ", ".join(str(tuple(c)) for c in nc[:60])
             + (" ... (all in the json)" if len(nc) > 60 else "")),
          "(a) cells outside, by species %(species)s, by band %(band)s, by cos Z row "
          "%(row)s, by azimuth bin %(azimuth)s." % res["a_outside_by_axis"],
          "(b) pairs outside, by band: %s." % res["b_outside_by_band"], "",
          "(c) per band: H solver, H Monte Carlo, error, inside:", ""]
    e = ev["c"]
    L += ["- %s GeV: %.4f, %.4f ± %.4f, %s" % (bands[i], e["solver"][i], e["mc"][i],
                                               e["err"][i], bool(e["inside"][i]))
          for i in range(len(bands))]

    L += ["", "## Reported", ""]
    mr, nin, nall, _ = _mean_r(phi_cut[CAP], s_runs[:, CAP])
    res["d_mean_r"] = mr.tolist()
    res["d_inside"], res["d_counted"] = nin, nall
    L += ["(d) Absolute, r = Φ_cut,solver / `s`: %d of %d cells within max(3%%, 2σ). "
          "Mean r − 1 per band (%s):" % (nin, nall, ", ".join(SPECIES)), ""]
    L += ["- %s GeV: %s" % (bands[i], " ".join("%+.4f" % (x - 1) for x in mr[:, i]))
          for i in range(len(bands))]
    e = ev["e"]
    fired = e["counted"] & ~e["inside"]
    res["e_trigger"] = bool(fired.any())
    L += ["", "(e) Level of K per band: solver, Monte Carlo, error, inside. Trigger %s."
          % ("**FIRED**" if fired.any() else "not fired"), ""]
    L += ["- %s GeV: %.4f, %.4f ± %.4f, %s" % (bands[i], e["solver"][i], e["mc"][i],
                                               e["err"][i], bool(e["inside"][i]))
          for i in range(len(bands))]

    if "phi_straight_0" in sol.files and "t_runs" in ref.files:
        t_runs = np.asarray(ref["t_runs"], float)[:, CAP]
        f = cutoff_factor(s_runs[:, CAP], t_runs, phi_cut[CAP],
                          np.asarray(sol["phi_straight_0"], float))
        with np.errstate(invalid="ignore", divide="ignore"):
            q = np.where(f["counted"], f["solver"] / f["mc"], np.nan)
        res["f"] = {"inside": int(f["inside"].sum()),
                    "counted": int(f["counted"].sum()),
                    "mean_q": np.nanmean(q, axis=(2, 3)).tolist()}
        L += ["", "(f) Plan wording, F = Φ_cut / (azimuth mean of the straight-ahead "
              "flux with no cutoff) against `s` / (Σ_az `t` / 12): %d of %d cells "
              "within max(3%%, 2σ). Mean of F_solver / F_MC − 1 per band (%s):"
              % (res["f"]["inside"], res["f"]["counted"], ", ".join(SPECIES)), ""]
        L += ["- %s GeV: %s" % (bands[i], " ".join(
            "%+.4f" % (x - 1) for x in np.array(res["f"]["mean_q"])[:, i]))
            for i in range(len(bands))]
    else:
        L += ["", "(f) Plan wording: not evaluated (no `phi_straight_0` or no "
              "`t_runs`)."]

    g = evaluate(s_runs[:, CAP5], n_runs[:, CAP5], phi_cut[CAP5], phi_0[CAP5], GATED)
    res["g"] = {p: {"inside": int(g[p]["inside"].sum()),
                    "counted": int(g[p]["counted"].sum())} for p in GATED}
    line = ", ".join("(%s) %d of %d" % (p, res["g"][p]["inside"],
                                        res["g"][p]["counted"]) for p in GATED)
    if a.null5:
        n5 = json.load(open(a.null5))
        if n5["sha256"] != sha or n5["cap_index"] != CAP5:
            _bad("--null5 is not of this reference and of cap index %d" % CAP5)
        res["g"]["thresholds"] = {p: int(n5["parts"][p]["threshold"]) for p in GATED}
        line += "; counts of an exactly right solver: " + ", ".join(
            "(%s) %d" % (p, res["g"]["thresholds"][p]) for p in GATED)
    L += ["", "(g) Cap of 5°, inside: %s." % line]

    if all(k in ref.files for k in ("us_runs", "un_runs")) and all(
            k in sol.files for k in ("phi_cut_up", "phi_0_up")):
        us = np.asarray(ref["us_runs"], float)[:, CAP][..., list(ROWS_UP), :]
        un = np.asarray(ref["un_runs"], float)[:, CAP][..., list(ROWS_UP), :]
        h = cutoff_factor(us, un, np.asarray(sol["phi_cut_up"], float)[CAP],
                          np.asarray(sol["phi_0_up"], float)[CAP])
        res["h"] = {"inside": int(h["inside"].sum()),
                    "counted": int(h["counted"].sum())}
        L += ["", "(h) Up-going rows cos Z −0.2 to 0, cutoff factor per cell: %d of %d "
              "counted cells within max(3%%, 2σ)." % (res["h"]["inside"],
                                                     res["h"]["counted"])]
    else:
        L += ["", "(h) Up-going rows: not evaluated (tallies or solver cells missing)."]

    os.makedirs(a.outdir, exist_ok=True)
    open(paths[0], "w").write("\n".join(L) + "\n")
    json.dump(res, open(paths[1], "w"), indent=1)
    print("\n".join(L))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
