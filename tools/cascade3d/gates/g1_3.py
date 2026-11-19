"""Gate G1.3 (3D transport against the field-free Monte Carlo); writes G1.3.md/.json.

Evaluation fixed before any Monte Carlo number was seen (WP 1.4, phase 1; brief
``c1/G13_BRIEF.md``, gate text ``tools/cascade3d/GATES.md``):

- Reference ``s[cap, species, band, cosz]`` of the combined file, cap index 3 (10 deg),
  species in the file's ``species`` order (checked: nue, antinue, numu, antinumu),
  bands and cos Z bins checked against the solver's cell edges.
- (a) r = Phi_solver / s per cell; rn = r / mean_cosZ(r) (arithmetic mean over the ten
  cos Z bins of the same species and band: equal solid angles).  sigma of a cell =
  ``s_err_task / s`` of that cell (the error of the normalising mean is not added).
  A cell is inside if |rn - 1| <= max(2%, 2 sigma).  Count: at least 95% of the cells.
  Row mean: arithmetic mean of rn over the four species and the bands with upper edge
  <= 1 GeV (20 cells) of one cos Z bin; its sigma = std(ddof=1) over the ten runs of
  the same statistic computed with ``s_runs[k]`` in place of ``s`` (each run normalised
  by its own cos Z mean, same solver values), divided by sqrt(n_runs).  A row passes
  if |mean - 1| <= max(1%, 2 sigma).  PASS = count and every row.
- A cell with s = 0 in the combined reference is reported and counted as outside;
  it is left out of its cos Z mean and of the row mean (it has no r).  A cell with
  s_runs = 0 in one run is left out of that run's means in the same way.
- (b) mean of r over cos Z (= the normaliser of (a)) per species and band; trigger if
  any |mean - 1| > 5%.  No diagnosis is started by this script.
- (c) (a) and (b) with the fresh 1e5-event kernels, and the difference from the pool
  kernels.  (d) q = (Phi_3D / Phi_straight) / (s / t), per cell, with the sigma of
  s / t from the spread over the runs; reported, with the caveat of the gate text.
- Reported cell by cell in every case: cos Z 0-0.1 and 0.1-0.2 below 0.4 GeV, and the
  band 0.10-0.16 GeV.
- Not gates: caps 2 against 3 per band and row (sigma of the ratio from the runs),
  and normalisation / binning diagnostics (constant factor, one-bin shifts in cos Z
  and energy, a power of cos Z).

Usage::

    $PY tools/cascade3d/gates/g1_3.py --ref REF.npz --pool POOL.npz \
        --fresh FRESH.npz --straight STRAIGHT.npz [--outdir DIR] [--name G1.3]

The solver files are ``run_solve.py --out`` files (keys ``cell_<species>``).
Exit status 0 = PASS, 1 = FAIL, 2 = input error.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

SPECIES = ("nue", "antinue", "numu", "antinumu")
CAP = 3
CAP_DEG = 10.0
TOL_CELL, TOL_ROW, FRAC_MIN, TRIGGER_B = 0.02, 0.01, 0.95, 0.05
ROW_EMAX = 1.0

GATE = """**G1.3 Three-dimensional transport against the Monte Carlo.**
Reference: a field-free run of `tools/mc3d/sky3d.py` (`--no-cutoff --no-bend-primary
--no-muon-field`), about 10^7 primaries, down-going tally `s`. Cells: the four neutrino
species x the Monte Carlo's energy bands inside 0.1 to 10 GeV x its ten cos Z bins from
0 to 1. The solver is averaged over each cell.

- **(a) Zenith shape, like for like (gated).** Solver with kernels built from the Monte
  Carlo's own event pools (...), so both use the same events.
  r = Phi_solver / s_MC per cell, divided by its solid-angle-weighted mean over the ten cos Z
  bins of the same species and energy band. Pass: at least 95% of cells within
  max(2%, 2 sigma_MC) of 1; and for each cos Z row, the mean over species and bands below 1 GeV
  within max(1%, 2 sigma_MC of that mean) of 1.
- **(b) Level, like for like (reported, with a trigger).** The mean of r before
  normalisation, per species and band. A deviation above 5% triggers one diagnosis cycle
  before decision point 1.
- **(c) Kernel statistics (reported).** (a) and (b) repeated with the 10^5-event kernels.
  The difference from the pool kernels measures the effect of the 2000-event pools on the
  Monte Carlo.
- **(d) Plan wording (reported).** Solver 3D over its own straight-ahead limit, against
  `s/t`, with the caveat above.

Reported cell by cell in every case: rows cos Z 0 to 0.1 and 0.1 to 0.2 below 0.4 GeV (the
engine's E_off v4 is about 8% high there) and the band 0.10 to 0.16 GeV (engine 5 to 9% low).

(Corrections of 2026-11-19: sigma_MC of a cell is the chunk-spread estimate `*_err_task`;
for the row means sigma is the spread over the ten runs of the row mean itself, divided by
sqrt(10).)"""


class InputError(Exception):
    pass


# ----------------------------------------------------------------------------- input
def load_ref(path):
    """The reference file, checked against the structure of the brief."""
    z = np.load(path, allow_pickle=False)
    need = ["species", "e_band_edges_GeV", "cosz_edges", "cap_theta_deg",
            "cap_area_cm2", "n_primaries", "n_primaries_per_run"]
    for x in ("s", "t", "s_cross", "t_cross"):
        need += [x, x + "_runs", x + "_err_sq", x + "_err_seed", x + "_err_task"]
    miss = [k for k in need if k not in z.files]
    if miss:
        raise InputError(f"reference lacks {miss}")
    ref = {k: np.asarray(z[k]) for k in z.files}
    sp = tuple(str(s) for s in ref["species"])
    if sp != SPECIES:
        raise InputError(f"reference species {sp} != {SPECIES}")
    if abs(float(ref["cap_theta_deg"][CAP]) - CAP_DEG) > 1e-9:
        raise InputError(f"cap {CAP} is {ref['cap_theta_deg'][CAP]} deg, not {CAP_DEG}")
    nc, ns, nb, nz = ref["s"].shape
    if ns != 4 or ref["s_runs"].shape[1:] != ref["s"].shape:
        raise InputError("unexpected shapes of s / s_runs")
    for k in ("s_err_task", "t", "t_runs"):
        if ref[k].shape[-4:] != ref["s"].shape:
            raise InputError(f"unexpected shape of {k}")
    if len(ref["e_band_edges_GeV"]) != nb + 1 or len(ref["cosz_edges"]) != nz + 1:
        raise InputError("edges do not match the tally shape")
    ref["meta"] = {k[5:]: str(ref[k]) for k in z.files if k.startswith("meta_")}
    return ref


def load_solver(path, ref):
    """Solver cells (4, n_band, n_cosz) in the reference species order; edges checked."""
    z = np.load(path, allow_pickle=False)
    be, ce = np.asarray(z["cell_band_edges"]), np.asarray(z["cell_cosz_edges"])
    if be.shape != ref["e_band_edges_GeV"].shape or not np.allclose(
            be, ref["e_band_edges_GeV"], rtol=1e-9, atol=0):
        raise InputError(f"{path}: band edges differ from the reference")
    if ce.shape != ref["cosz_edges"].shape or not np.allclose(
            ce, ref["cosz_edges"], rtol=0, atol=1e-9):
        raise InputError(f"{path}: cos Z edges differ from the reference")
    cells = np.stack([np.asarray(z[f"cell_{q}"], float) for q in SPECIES])
    cost = json.loads(str(z["cost_json"])) if "cost_json" in z.files else {}
    return cells, cost


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ----------------------------------------------------------------------- statistics
def _ratio(sol, s):
    """r = sol / s with NaN where s <= 0 (no r)."""
    s = np.asarray(s, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(s > 0, sol / np.where(s > 0, s, 1.0), np.nan)


def normalise(r):
    """rn = r / (mean over cos Z of r), mean over the cells that have an r."""
    m = np.nanmean(r, axis=-1, keepdims=True)
    return r / m, m[..., 0]


def row_bands(band_edges):
    return np.flatnonzero(np.asarray(band_edges)[1:] <= ROW_EMAX * (1 + 1e-9))


def row_means(rn, ib):
    """(n_cosz,) mean of rn over the species and the bands ``ib``."""
    return np.nanmean(rn[:, ib, :], axis=(0, 1))


def part_a(sol, ref):
    """Gate (a) and the numbers of (b) for solver cells ``sol`` (4, nb, nz)."""
    s = ref["s"][CAP]
    err = ref["s_err_task"][CAP]
    r = _ratio(sol, s)
    rn, rbar = normalise(r)
    with np.errstate(divide="ignore", invalid="ignore"):
        sig = np.where(s > 0, err / np.where(s > 0, s, 1.0), np.nan)
    tol = np.maximum(TOL_CELL, 2.0 * sig)
    zero = ~(s > 0)
    inside = (np.abs(rn - 1.0) <= tol) & ~zero
    pull = (rn - 1.0) / sig
    n_cells = int(inside.size)
    n_in = int(inside.sum())
    count_ok = n_in >= FRAC_MIN * n_cells - 1e-9
    ib = row_bands(ref["e_band_edges_GeV"])
    rows = row_means(rn, ib)
    runs = ref["s_runs"][:, CAP]
    rows_k = np.array([row_means(normalise(_ratio(sol, sk))[0], ib) for sk in runs])
    n_runs = runs.shape[0]
    rows_sig = np.std(rows_k, axis=0, ddof=1) / np.sqrt(n_runs)
    rows_tol = np.maximum(TOL_ROW, 2.0 * rows_sig)
    rows_ok = np.abs(rows - 1.0) <= rows_tol
    ok = bool(count_ok and rows_ok.all())
    trig = np.abs(rbar - 1.0) > TRIGGER_B
    return {"r": r, "rn": rn, "rbar": rbar, "sigma": sig, "tol": tol, "pull": pull,
            "inside": inside, "zero": zero, "n_cells": n_cells, "n_inside": n_in,
            "n_needed": int(np.ceil(FRAC_MIN * n_cells - 1e-9)), "count_ok": bool(count_ok),
            "row_bands": ib, "rows": rows, "rows_runs": rows_k, "rows_sigma": rows_sig,
            "rows_tol": rows_tol, "rows_ok": rows_ok, "pass": ok,
            "b_trigger": trig, "b_any": bool(trig.any()), "n_runs": int(n_runs)}


def part_d(sol3, solsa, ref):
    """q = (Phi_3D / Phi_straight) / (s / t); sigma of s/t from the runs."""
    s, t = ref["s"][CAP], ref["t"][CAP]
    st = _ratio(s, t)
    st_k = np.array([_ratio(a, b) for a, b in zip(ref["s_runs"][:, CAP],
                                                  ref["t_runs"][:, CAP])])
    n = st_k.shape[0]
    with np.errstate(invalid="ignore"):
        sig = np.nanstd(st_k, axis=0, ddof=1) / np.sqrt(n) / st
    q = (sol3 / solsa) / st
    tol = np.maximum(TOL_CELL, 2.0 * sig)
    inside = np.abs(q - 1.0) <= tol
    return {"q": q, "solver_ratio": sol3 / solsa, "mc_ratio": st, "sigma": sig,
            "inside": inside, "n_inside": int(np.nansum(inside)), "n_cells": int(q.size)}


def caps_check(ref):
    """s[cap 2] / s[cap 3] - 1 per band and row, sigma from the runs (correlated caps)."""
    a, b = ref["s"][2], ref["s"][CAP]
    q = _ratio(a, b)
    qk = np.array([_ratio(x, y) for x, y in zip(ref["s_runs"][:, 2],
                                                ref["s_runs"][:, CAP])])
    sig = np.nanstd(qk, axis=0, ddof=1) / np.sqrt(qk.shape[0])
    with np.errstate(invalid="ignore"):
        pull = (q - 1.0) / sig
    # per band, all rows and species: weighted mean of q - 1
    w = 1.0 / sig ** 2
    band = np.nansum(w * (q - 1), axis=(0, 2)) / np.nansum(w, axis=(0, 2))
    band_sig = 1.0 / np.sqrt(np.nansum(w, axis=(0, 2)))
    return {"q": q, "sigma": sig, "pull": pull, "band_dev": band, "band_sig": band_sig,
            "chi2": float(np.nansum(pull ** 2)), "ndf": int(np.isfinite(pull).sum())}


def diagnostics(sol, ref, a):
    """Signs of a normalisation or binning mismatch rather than physics."""
    s = ref["s"][CAP]
    sig = a["sigma"]
    out = {}
    rb = a["rbar"]
    out["rbar_mean"] = float(np.mean(rb))
    out["rbar_rms_about_mean"] = float(np.std(rb))
    out["rbar_min"], out["rbar_max"] = float(np.min(rb)), float(np.max(rb))

    def chi2(rn):
        p = (rn - 1.0) / sig
        return float(np.nansum(p ** 2)), int(np.isfinite(p).sum())

    out["chi2_shape"] = chi2(a["rn"])
    # one-bin shifts of the reference in cos Z / energy (interior cells only)
    sh = {}
    for name, axis, k in (("cosz+1", 2, 1), ("cosz-1", 2, -1), ("band+1", 1, 1),
                          ("band-1", 1, -1)):
        s2 = np.roll(s, k, axis=axis)
        sl = [slice(None)] * 3
        sl[axis] = slice(1, None) if k == 1 else slice(None, -1)
        sl = tuple(sl)
        r2 = _ratio(sol, s2)
        rn2, _ = normalise(r2)
        rn1 = a["rn"].copy()
        m = np.zeros_like(rn1, bool)
        m[sl] = True
        p0 = np.where(m, (rn1 - 1) / sig, np.nan)
        p1 = np.where(m, (rn2 - 1) / sig, np.nan)
        sh[name] = {"chi2_unshifted": float(np.nansum(p0 ** 2)),
                    "chi2_shifted": float(np.nansum(p1 ** 2)),
                    "n": int(np.isfinite(p0).sum())}
    out["shifts"] = sh
    # rn ~ c^alpha over the cos Z bin centres (weighted least squares of ln rn)
    ce = ref["cosz_edges"]
    cc = 0.5 * (ce[1:] + ce[:-1])
    x = np.broadcast_to(np.log(cc), a["rn"].shape)
    y = np.log(a["rn"])
    w = 1.0 / sig ** 2
    ok = np.isfinite(y) & np.isfinite(w)
    xm = np.sum(w[ok] * x[ok]) / np.sum(w[ok])
    ym = np.sum(w[ok] * y[ok]) / np.sum(w[ok])
    sxx = np.sum(w[ok] * (x[ok] - xm) ** 2)
    out["cosz_power"] = float(np.sum(w[ok] * (x[ok] - xm) * (y[ok] - ym)) / sxx)
    out["cosz_power_sigma"] = float(1.0 / np.sqrt(sxx))
    return out


# --------------------------------------------------------------------------- report
def _fmt_table(arr, ref, fmt, title, flag=None):
    be, ce = ref["e_band_edges_GeV"], ref["cosz_edges"]
    lines = []
    for i, q in enumerate(SPECIES):
        lines += [f"{title}, {q} (rows: band [GeV]; columns: cos Z bin)", "",
                  "| band | " + " | ".join(f"{ce[j]:.1f}-{ce[j + 1]:.1f}"
                                         for j in range(len(ce) - 1)) + " |",
                  "|---|" + "---|" * (len(ce) - 1)]
        for b in range(len(be) - 1):
            cells = []
            for j in range(len(ce) - 1):
                v = arr[i, b, j]
                txt = "s=0" if not np.isfinite(v) else format(v, fmt)
                if flag is not None and not flag[i, b, j]:
                    txt = f"**{txt}**"
                cells.append(txt)
            lines.append(f"| {be[b]:.3g}-{be[b + 1]:.3g} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def special_cells(ref):
    """(species, band, cosz) of the cells reported one by one."""
    be, ce = ref["e_band_edges_GeV"], ref["cosz_edges"]
    sel = set()
    for b in range(len(be) - 1):
        for j in range(len(ce) - 1):
            low_row = ce[j + 1] <= 0.2 + 1e-9 and be[b + 1] <= 0.4 * (1 + 1e-9)
            first_band = abs(be[b] - 0.1) < 1e-9
            if low_row or first_band:
                sel.add((b, j))
    return sorted(sel)


def special_lines(label, ref, cols):
    """``cols``: list of (header, array (4, nb, nz), fmt)."""
    be, ce = ref["e_band_edges_GeV"], ref["cosz_edges"]
    lines = [f"Cells reported one by one, {label}:", "",
             "| species | band [GeV] | cos Z | " + " | ".join(h for h, _, _ in cols) + " |",
             "|---|---|---|" + "---|" * len(cols)]
    for b, j in special_cells(ref):
        for i, q in enumerate(SPECIES):
            vals = []
            for _h, arr, fmt in cols:
                v = arr[i, b, j]
                vals.append("n/a" if not np.isfinite(v) else format(v, fmt))
            lines.append(f"| {q} | {be[b]:.3g}-{be[b + 1]:.3g} | {ce[j]:.1f}-{ce[j + 1]:.1f} | "
                         + " | ".join(vals) + " |")
    return lines + [""]


def block_a(label, a, ref):
    ce, be = ref["cosz_edges"], ref["e_band_edges_GeV"]
    L = [f"### {label}: zenith shape (a)", "",
         f"Cells inside max(2%, 2 sigma): **{a['n_inside']} of {a['n_cells']}** "
         f"(needed {a['n_needed']}); cells with s = 0: {int(a['zero'].sum())}; "
         f"count criterion {'met' if a['count_ok'] else 'NOT met'}.", "",
         f"Row means over the four species and the bands "
         f"{be[a['row_bands'][0]]:.3g}-{be[a['row_bands'][-1] + 1]:.3g} GeV "
         f"({4 * len(a['row_bands'])} cells), sigma from the {a['n_runs']} runs:", "",
         "| cos Z | row mean - 1 | sigma | tolerance | inside |", "|---|---|---|---|---|"]
    for j in range(len(ce) - 1):
        L.append(f"| {ce[j]:.1f}-{ce[j + 1]:.1f} | {a['rows'][j] - 1:+.4f} | "
                 f"{a['rows_sigma'][j]:.4f} | {a['rows_tol'][j]:.4f} | "
                 f"{'yes' if a['rows_ok'][j] else '**NO**'} |")
    L += ["", f"(a) for {label}: **{'PASS' if a['pass'] else 'FAIL'}**", ""]
    L += _fmt_table(a["rn"], ref, ".4f", f"Normalised r ({label}; bold = outside)",
                    a["inside"])
    L += _fmt_table(a["pull"], ref, "+.2f", f"(rn - 1) / sigma ({label})")
    L += _fmt_table(a["sigma"], ref, ".4f", f"sigma_MC = s_err_task / s ({label})")
    return L


def block_b(label, a, ref):
    be = ref["e_band_edges_GeV"]
    L = [f"### {label}: level (b), mean of r over cos Z before normalisation", "",
         "| band [GeV] | " + " | ".join(SPECIES) + " |", "|---|" + "---|" * 4]
    for b in range(len(be) - 1):
        L.append(f"| {be[b]:.3g}-{be[b + 1]:.3g} | " + " | ".join(
            (f"**{a['rbar'][i, b] - 1:+.4f}**" if a["b_trigger"][i, b]
             else f"{a['rbar'][i, b] - 1:+.4f}") for i in range(4)) + " |")
    L += ["", ("Trigger of (b): **a mean of r deviates by more than 5%** (bold above). "
               "One diagnosis cycle is due before decision point 1; it is decided by "
               "the manager, not started here." if a["b_any"] else
               "Trigger of (b): no mean of r deviates by more than 5%."), ""]
    return L


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def evaluate(ref, pool, fresh=None, straight=None):
    res = {"a": part_a(pool, ref)}
    if fresh is not None:
        res["c"] = part_a(fresh, ref)
        res["c_minus_pool_rn"] = res["c"]["rn"] - res["a"]["rn"]
        res["c_over_pool_rbar"] = res["c"]["rbar"] / res["a"]["rbar"] - 1.0
    if straight is not None:
        res["d"] = part_d(pool, straight, ref)
    res["caps"] = caps_check(ref)
    res["diag"] = diagnostics(pool, ref, res["a"])
    return res


def report(res, ref, args, inputs, cmd):
    a = res["a"]
    L = [f"# Gate {args.name}: three-dimensional transport against the Monte Carlo", "",
         GATE, "", "## Inputs", ""]
    for k, (p, h) in inputs.items():
        L.append(f"- {k}: `{p}` (sha256 {h[:16]})")
    L += [f"- cap index {CAP} ({CAP_DEG:g} deg); n_primaries "
          f"{int(ref['n_primaries'])}; runs {a['n_runs']}", ""]
    if ref.get("meta"):
        L += ["Reference metadata (verbatim):", ""]
        for k, v in sorted(ref["meta"].items()):
            L.append(f"- `meta_{k}`: {v}")
        L.append("")
    L += [f"## Result of the gated part (a): **{'PASS' if a['pass'] else 'FAIL'}**", "",
          f"- cells inside: {a['n_inside']} of {a['n_cells']} (needed {a['n_needed']})",
          f"- rows inside: {int(a['rows_ok'].sum())} of {len(a['rows_ok'])}",
          f"- (b) trigger: {'YES' if a['b_any'] else 'no'} "
          f"(mean r - 1 from {np.min(a['rbar']) - 1:+.4f} to {np.max(a['rbar']) - 1:+.4f})",
          ""]
    L += ["## (a), (b): pool kernels (gated)", ""]
    L += block_a("pool2000", a, ref)
    L += block_b("pool2000", a, ref)
    L += special_lines("pool2000", ref, [("normalised r", a["rn"], ".4f"),
                                        ("(rn-1)/sigma", a["pull"], "+.2f"),
                                        ("r before normalisation", a["r"], ".4f")])
    if "c" in res:
        c = res["c"]
        L += ["## (c): fresh 10^5-event kernels (reported)", ""]
        L += block_a("fresh1e5", c, ref)
        L += block_b("fresh1e5", c, ref)
        L += _fmt_table(res["c_minus_pool_rn"], ref, "+.4f",
                        "rn(fresh1e5) - rn(pool2000)")
        L += ["Level: mean r (fresh1e5) / mean r (pool2000) - 1:", "",
              "| band [GeV] | " + " | ".join(SPECIES) + " |", "|---|" + "---|" * 4]
        be = ref["e_band_edges_GeV"]
        for b in range(len(be) - 1):
            L.append(f"| {be[b]:.3g}-{be[b + 1]:.3g} | " + " | ".join(
                f"{res['c_over_pool_rbar'][i, b]:+.4f}" for i in range(4)) + " |")
        L.append("")
        L += special_lines("fresh1e5", ref, [("normalised r", c["rn"], ".4f"),
                                            ("(rn-1)/sigma", c["pull"], "+.2f"),
                                            ("rn fresh - rn pool", res["c_minus_pool_rn"],
                                             "+.4f")])
    if "d" in res:
        d = res["d"]
        L += ["## (d): solver 3D / straight ahead against s / t (reported)", "",
              "Caveat of the gate text: the Monte Carlo's `t` is not a one-dimensional "
              "cascade; it is the three-dimensional cascade with its neutrinos laid on "
              "the primary's line.  The comparison is not like for like.", "",
              f"Cells with |q - 1| <= max(2%, 2 sigma(s/t)): {d['n_inside']} of "
              f"{d['n_cells']} (not a gate).", ""]
        L += _fmt_table(d["q"], ref, ".4f", "q = (Phi_3D/Phi_straight)/(s/t)",
                        d["inside"])
        L += special_lines("(d)", ref, [("q", d["q"], ".4f"),
                                        ("solver 3D/straight", d["solver_ratio"], ".4f"),
                                        ("MC s/t", d["mc_ratio"], ".4f"),
                                        ("sigma(s/t) rel", d["sigma"], ".4f")])
    cp = res["caps"]
    be = ref["e_band_edges_GeV"]
    L += ["## Not gates", "", "### Caps of the reference: s(7.5 deg) / s(10 deg) - 1", "",
          f"chi2 / ndf of the cell pulls: {cp['chi2']:.1f} / {cp['ndf']}.  Per band "
          "(weighted over species and cos Z):", "",
          "| band [GeV] | deviation | sigma |", "|---|---|---|"]
    for b in range(len(be) - 1):
        L.append(f"| {be[b]:.3g}-{be[b + 1]:.3g} | {cp['band_dev'][b]:+.4f} | "
                 f"{cp['band_sig'][b]:.4f} |")
    L.append("")
    L += _fmt_table(cp["pull"], ref, "+.2f", "Cap pull (s2/s3 - 1)/sigma")
    dg = res["diag"]
    L += ["### Normalisation and binning diagnostics (pool2000)", "",
          f"- mean of r over all species and bands: {dg['rbar_mean']:.4f}, rms about it "
          f"{dg['rbar_rms_about_mean']:.4f}, range {dg['rbar_min']:.4f} to "
          f"{dg['rbar_max']:.4f} (a constant factor shows as a common offset with a "
          "small rms)",
          f"- chi2 of the shape pulls: {dg['chi2_shape'][0]:.1f} over "
          f"{dg['chi2_shape'][1]} cells",
          f"- rn as a power of cos Z (bin centres): exponent {dg['cosz_power']:+.4f} +- "
          f"{dg['cosz_power_sigma']:.4f} (a missing or extra factor cos Z gives -1 or +1)"]
    for k, v in dg["shifts"].items():
        L.append(f"- reference shifted by one bin ({k}): chi2 {v['chi2_shifted']:.1f} "
                 f"against {v['chi2_unshifted']:.1f} unshifted, over {v['n']} cells")
    L += ["", "## Command", "", f"`{cmd}`", ""]
    return L


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--fresh", default="")
    ap.add_argument("--straight", default="")
    ap.add_argument("--outdir", default=os.path.join(
        os.environ.get("N3", "/home/pigrange/mc3d_work/runs/native3d"), "c1", "g13"))
    ap.add_argument("--name", default="G1.3")
    args = ap.parse_args(argv)
    try:
        ref = load_ref(args.ref)
        pool, cpool = load_solver(args.pool, ref)
        fresh = load_solver(args.fresh, ref)[0] if args.fresh else None
        straight = load_solver(args.straight, ref)[0] if args.straight else None
    except InputError as exc:
        print("INPUT ERROR:", exc, file=sys.stderr)
        return 2
    res = evaluate(ref, pool, fresh, straight)
    inputs = {"reference": (args.ref, sha(args.ref)), "pool2000 solve": (args.pool, sha(args.pool))}
    if args.fresh:
        inputs["fresh1e5 solve"] = (args.fresh, sha(args.fresh))
    if args.straight:
        inputs["pool2000 straight solve"] = (args.straight, sha(args.straight))
    inputs["g1_3.py"] = (os.path.abspath(__file__), sha(os.path.abspath(__file__)))
    cmd = "$PY tools/cascade3d/gates/g1_3.py " + " ".join(argv if argv is not None
                                                          else sys.argv[1:])
    lines = report(res, ref, args, inputs, cmd)
    os.makedirs(args.outdir, exist_ok=True)
    md = os.path.join(args.outdir, f"{args.name}.md")
    with open(md, "w") as fh:
        fh.write("\n".join(lines))
    summary = {"pass_a": res["a"]["pass"], "command": cmd,
               "inputs": {k: {"path": p, "sha256": h} for k, (p, h) in inputs.items()},
               "a": {k: v for k, v in res["a"].items() if k not in ("rows_runs",)}}
    for k in ("c", "d", "caps", "diag", "c_minus_pool_rn", "c_over_pool_rbar"):
        if k in res:
            summary[k] = res[k]
    with open(os.path.join(args.outdir, f"{args.name}.json"), "w") as fh:
        json.dump(jsonable(summary), fh, indent=1)
    print(f"{args.name} (a): {'PASS' if res['a']['pass'] else 'FAIL'}; cells "
          f"{res['a']['n_inside']}/{res['a']['n_cells']}; rows "
          f"{int(res['a']['rows_ok'].sum())}/10; (b) trigger "
          f"{'YES' if res['a']['b_any'] else 'no'}; wrote {md}")
    return 0 if res["a"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
