"""Gate G0.2: decay kernels against MCEq's decay blocks (work package 0.3).

Evaluates G0.2 exactly as written in ``tools/cascade3d/GATES.md``::

    $PY tools/cascade3d/gates/g0_2.py [--kernels PATH] [--mceqlike PATH] [--flat PATH]

and writes ``$N3/c0/gates/G0.2.md`` (gate text, numbers, PASS or FAIL, command).

Definition used for the gated test (the gate text leaves "column" and "entry" open):
the block is MCEq's raw decay block ``D[m, k]`` of ``particle.decay_dists[child]`` (the
same array in the file ``dec_m`` is cut from, before the factor ``1/lambda_dec``); the
column is the parent bin ``k``; an entry is ``D[m, k]``; it is tested when it exceeds 1%
of ``sum_m D[m, k]`` (the sum of MCEq's column); the test is
``|ours - MCEq| <= 0.5% * MCEq``.  Parent bins ``k`` are those whose centre energy
``e_grid[k]`` (kinetic) lies in ``[0.1, 1000]`` GeV.

Gated pairs: pi+- -> mu_l, mu_r, nu_mu; K+- -> mu_l, mu_r (pure two-body blocks); and
K+- -> nu_mu on the entries where our K_mu3 contribution is below 0.1% of the entry
(MCEq's block adds the three-body neutrinos of K_mu3, which are not gated).
Everything else (three-body kaon decays, K_L, K_S, muon decay, K -> pi, ...) is
reported, not gated.
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import kernels_decay as kd  # noqa: E402

N3 = os.environ.get("N3", "/home/pigrange/mc3d_work/runs/native3d")
PARENT_SAMPLED = ("K+", "K-", "K_L0", "mu+_l", "mu+_r", "mu+", "mu-_l", "mu-_r",
                   "mu-")
E_RANGES = [(0.1, 1.0), (1.0, 10.0), (10.0, 100.0), (100.0, 1000.0)]
TOL = 0.005
FRAC = 0.01


def gate_text():
    path = os.path.join(os.path.dirname(HERE), "GATES.md")
    lines = open(path).read().split("\n")
    out = []
    on = False
    for ln in lines:
        if ln.startswith("**G0.2"):
            on = True
        elif on and ln.startswith("**G0.3"):
            break
        if on:
            out.append(ln)
    return "\n".join(out).strip()


def parent_bins(grid, lo=0.1, hi=1000.0):
    return np.nonzero((grid.e_grid >= lo) & (grid.e_grid <= hi))[0]


def compare(ours, ref, ks, hd=None, n_sel=None):
    """Per parent bin comparison of two blocks ``D[m, k]``.

    Returns dict of arrays over the entries above ``FRAC`` of MCEq's column sum.
    """
    cs = ref.sum(axis=0)
    rows = []
    for k in ks:
        sel = ref[:, k] > FRAC * cs[k]
        if n_sel is not None:
            sel &= n_sel[:, k]
        for m in np.nonzero(sel)[0]:
            rel = (ours[m, k] - ref[m, k]) / ref[m, k]
            z = np.nan
            if hd is not None and hd[m, k] != 0:
                z = (ours[m, k] - ref[m, k]) / abs(hd[m, k])
            rows.append((k, m, ref[m, k], ours[m, k], rel, z))
    return rows


def summarise(rows):
    if not rows:
        return dict(n=0, nfail=0, max=np.nan, med=np.nan, where=None)
    rel = np.array([abs(r[4]) for r in rows])
    i = int(np.argmax(rel))
    return dict(n=len(rows), nfail=int((rel > TOL).sum()), max=rel[i],
                med=float(np.median(rel)), where=(rows[i][0], rows[i][1]))


def fmt_pct(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.3g}%"


def yield_ratio(ours, ref, grid, ks):
    """Total yield (sum over child bins of D * w_m / w_k) ours / MCEq per parent bin."""
    w = grid.e_widths
    ys = (ours * w[:, None]).sum(axis=0) / w
    yr = (ref * w[:, None]).sum(axis=0) / w
    return ys[ks] / np.where(yr[ks] > 0, yr[ks], np.nan)


def mean_energy_ratio(ours, ref, grid, ks):
    """Yield-weighted mean child kinetic energy, ours / MCEq, per parent bin."""
    w = grid.e_widths
    e = grid.e_grid
    out = []
    for k in ks:
        yo = ours[:, k] * w
        yr = ref[:, k] * w
        if yo.sum() <= 0 or yr.sum() <= 0:
            out.append(np.nan)
            continue
        out.append(((yo * e).sum() / yo.sum()) / ((yr * e).sum() / yr.sum()))
    return np.array(out)


def erange_ks(grid, ks, lo, hi):
    return np.array([k for k in ks if lo <= grid.e_grid[k] < hi * (1 + 1e-12)])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", default=kd.DEFAULT_PATH)
    ap.add_argument("--mceqlike", default=os.path.join(N3, "kernels",
                                                      "decay_mceqlike.npz"))
    ap.add_argument("--flat", default=os.path.join(N3, "kernels", "decay_flat3.npz"))
    ap.add_argument("--out", default=os.path.join(N3, "c0", "gates", "G0.2.md"))
    a = ap.parse_args(argv)
    t0 = time.time()
    cmd = "$PY tools/cascade3d/gates/g0_2.py " + " ".join(sys.argv[1:])

    ref = kd.mceq_reference(blocks=True, e_min=0.03)
    grid = kd.Grid(ref["e_bins"], ref["e_grid"])
    K = kd.load(a.kernels)
    assert np.allclose(K.e_bins, grid.e_bins, rtol=1e-12, atol=0), "grid differs"
    meta = K.meta
    notes = []

    # ---- cross-check with the WP 0.1 extraction (if present) ----
    for pop in ("PHYS", "REF"):
        p = os.path.join(N3, "kernels", f"mceq_{pop}.npz")
        if not os.path.exists(p):
            notes.append(f"`mceq_{pop}.npz` not present at run time: no cross-check.")
            continue
        with np.load(p, allow_pickle=False) as d:
            eb = d["e_bins"]
            sp = [str(s) for s in d["species"]]
        nb = len(eb) - 1
        ok_grid = np.allclose(eb, K.e_bins[-len(eb):], rtol=1e-12, atol=0)
        ok_sp = sp == [s[0] for s in ref["species"]]
        assert ok_grid, f"{pop}: energy grid differs from kernels_mceq"
        assert ok_sp, f"{pop}: species list differs from kernels_mceq"
        notes.append(f"`mceq_{pop}.npz`: grid ({nb} bins, the last {nb} of the "
                     f"{grid.n} kernel bins) and the {len(sp)} species (names, order) "
                     f"agree with this module's MCEq (asserted).")

    # REF (e_min 0.1) blocks must be the trailing sub-blocks of the PHYS ones
    try:
        ref_ref = kd.mceq_reference(blocks=True, e_min=0.1)
        worst = max(np.abs(ref["blocks"][key][-ref_ref["e_grid"].size:,
                                             -ref_ref["e_grid"].size:] - blk).max()
                    / max(np.abs(blk).max(), 1e-300)
                    for key, blk in ref_ref["blocks"].items())
        notes.append(f"MCEq built with e_min = 0.1 (REF, {ref_ref['e_grid'].size} "
                     f"bins): all {len(ref_ref['blocks'])} decay blocks equal the "
                     f"trailing sub-blocks of the {grid.n}-bin blocks, largest "
                     f"relative difference {worst:.2g}; the kernels serve both "
                     "populations.")
    except Exception as exc:                              # pragma: no cover
        notes.append(f"REF sub-grid check not run: {exc}")

    ks_all = parent_bins(grid)
    k_lo, k_hi = ks_all.min(), ks_all.max()

    # ---- gated pairs ----
    gated = []
    for par in ("pi+", "pi-"):
        ch = (["mu+_l", "mu+_r", "numu"] if par == "pi+"
              else ["mu-_l", "mu-_r", "antinumu"])
        gated += [(par, c, None) for c in ch]
    for par in ("K+", "K-"):
        mu = ["mu+_l", "mu+_r"] if par == "K+" else ["mu-_l", "mu-_r"]
        nu = "numu" if par == "K+" else "antinumu"
        gated += [(par, c, None) for c in mu] + [(par, nu, "no3b")]

    def kmu3_share(par, child):
        """Our K_mu3 contribution to the neutrino block, as a fraction of the block."""
        # K_mu3 neutrino alone = total (two-body + K_mu3) - two-body-only analytic
        ch, _ = kd.decay_channels(par)
        br2 = [b for b, ds in ch if len(ds) == 2 and abs(ds[0]) == 13][0]
        pdg = kd.PARENTS[par][0]
        T, w = kd.parent_energy_nodes(grid, "centre", 1)
        nu_pdg = 14 if pdg > 0 else -14
        two = kd.two_body_moments(grid, kd.M_K, 0.0, kd.M_MU, T, w, 128, 0)[:, :, 0]
        wr = grid.e_widths[None, :] / grid.e_widths[:, None]
        two = two * wr * br2
        tot = K.block(par, child)
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(tot > 0, (tot - two) / tot, 0.0)
        return share, nu_pdg

    rows_g = {}
    lines_g = []
    all_rows = []
    for par, child, flag in gated:
        ours = K.block(par, child)
        mce = ref["blocks"][(par, child)]
        sel = None
        if flag == "no3b":
            share, _ = kmu3_share(par, child)
            sel = share < 1e-3
        rows = compare(ours, mce, ks_all, n_sel=sel)
        rows_g[(par, child)] = rows
        all_rows += rows
        s = summarise(rows)
        lines_g.append((par, child, flag, s))
    tot = summarise(all_rows)
    verdict = "PASS" if (tot["n"] > 0 and tot["nfail"] == 0) else "FAIL"

    # ---- diagnostic: MCEq's own approximations in our analytic code ----
    wr = grid.e_widths[None, :] / grid.e_widths[:, None]
    Tc, wc = kd.parent_energy_nodes(grid, "centre", 1)
    mcm = dict(pi=0.13957, mu=0.10566, K=0.49368)

    def analytic(par, child, ur, masses):
        """Analytic two-body kernel (D convention) for the gated two-body blocks."""
        m0 = masses["pi"] if par.startswith("pi") else masses["K"]
        mm = masses["mu"]
        ch, _ = kd.decay_channels(par)
        br = [b for b, ds in ch if len(ds) == 2 and abs(ds[0]) == 13][0]
        if child.startswith("mu"):
            pdg = kd.PARENTS[par][0]
            h0 = -1.0 if pdg > 0 else +1.0
            l, r = kd.two_body_moments(grid, m0, mm, 0.0, Tc, wc, 128, 0, h0=h0,
                                       ur=ur)
            arr = l if child.endswith("_l") else r
        else:
            arr = kd.two_body_moments(grid, m0, 0.0, mm, Tc, wc, 128, 0, ur=ur)
        return arr[:, :, 0] * wr * br

    diag = {}
    for label, ur, masses in (
            ("exact kinematics, project masses (as the kernel file)", False,
             dict(pi=kd.M_PI, mu=kd.M_MU, K=kd.M_K)),
            ("MCEq approximations (beta = 1, UR polarisation), project masses", True,
             dict(pi=kd.M_PI, mu=kd.M_MU, K=kd.M_K)),
            ("MCEq approximations, MCEq's rounded masses", True, mcm)):
        rr = []
        for par, child, flag in gated:
            if flag == "no3b":
                continue
            ours = analytic(par, child, ur, masses)
            if "rounded" in label:
                # MCEq's own BR ratio is 0.99988 for pi; our BR renormalises to 1
                pass
            rr += compare(ours, ref["blocks"][(par, child)], ks_all)
        diag[label] = summarise(rr)

    # ---- where does the exact-kinematics kernel fail: by parent energy range ----
    by_range = {}
    for (par, child), rows in rows_g.items():
        for lo, hi in E_RANGES:
            kk = set(erange_ks(grid, ks_all, lo, hi).tolist())
            rr = [r for r in rows if r[0] in kk]
            key = "pi" if par[0] == "p" else "K"
            by_range.setdefault((key, lo, hi), []).extend(rr)

    # ---- reported tables: three-body kaon decays, K_L, K_S, muon decay ----
    mceqlike = kd.load(a.mceqlike) if os.path.exists(a.mceqlike) else None
    flat = kd.load(a.flat) if os.path.exists(a.flat) else None
    rep = []
    names = [(p, c) for (p, c) in K.pairs if p in ("K+", "K_L0", "K_S0", "mu+_l",
                                                  "mu+_r", "mu+", "K-", "mu-_l",
                                                  "mu-_r", "mu-")]
    for par, child in sorted(names):
        if (par, child) not in ref["blocks"]:
            continue
        mce = ref["blocks"][(par, child)]
        ours = K.block(par, child)
        rep.append((par, child, "exact, mc3d Dalitz", ours, mce, K))
        if mceqlike is not None and (par, child) in mceqlike._index:
            rep.append((par, child, "MCEq-like (UR)", mceqlike.block(par, child), mce,
                        mceqlike))
        if flat is not None and (par, child) in flat._index:
            rep.append((par, child, "exact, flat Dalitz", flat.block(par, child), mce,
                        flat))

    # ---- not covered ----
    covered_parents = {p for p, _ in K.pairs}
    cov = set(K.pairs)
    missing_parents = []
    for sp in ref["species"]:
        nm, pdg, hel, mass, ctau, trk = sp
        if trk or not np.isfinite(ctau):
            continue
        if any(p == nm for (p, _) in ref["blocks"]) and nm not in covered_parents:
            missing_parents.append(nm)
    missing_children = sorted(f"{p}>{c}" for (p, c) in ref["blocks"]
                              if p in covered_parents and (p, c) not in cov)

    # ---- write ----
    L = []
    w = L.append
    w("# G0.2 Decay kernels (gated for two-body)")
    w("")
    w(f"Verdict: **{verdict}**   (run {time.strftime('%Y-%m-%d %H:%M')}, "
      f"{time.time() - t0:.0f} s)")
    w("")
    w("## Gate text (verbatim from `GATES.md`)")
    w("")
    for ln in gate_text().split("\n"):
        w("> " + ln if ln else ">")
    w("")
    w("## Definitions used")
    w("")
    w("- Kernel under test: `" + os.path.relpath(a.kernels, N3) + "` (exact kinematics "
      "of `tools/mc3d/decays.py`, parent at the geometric bin centre, "
      f"`n_samples`={meta.get('n_samples')}, three-body scale "
      f"{meta.get('three_body_scale')}, Dalitz `{meta.get('dalitz')}`, "
      f"seed {meta.get('seed')}).")
    w("- MCEq's block: `particle.decay_dists[child]` of the MCEq built by "
      "`kernels_decay.mceq_reference()` (database "
      f"`{ref['db'].split('fname=')[-1].split(',')[0]}`), `D[m, k]` per the width "
      "convention in the module docstring. The blocks carry no `1/lambda_dec` and "
      "are defined for every parent bin; `dec_m` is zero below `mix_idx` "
      "(pi: 20, K+-: 30 on the 121-bin grid), where MCEq treats the parent as a "
      "resonance, so the raw blocks are the only thing to compare with there.")
    w("- Parent bins: centre energy (kinetic) in [0.1, 1000] GeV: "
      f"k = {k_lo} .. {k_hi} ({len(ks_all)} bins, {grid.e_grid[k_lo]:.3g} .. "
      f"{grid.e_grid[k_hi]:.4g} GeV) of {grid.n}.")
    w("- Entry: `D[m, k]`; tested if > 1% of `sum_m D[m, k]` of MCEq's column; test "
      "`|ours - MCEq| <= 0.5% MCEq`. l = 0 only.")
    w("- Gated pairs: pi+- -> mu_l, mu_r, nu_mu; K+- -> mu_l, mu_r; K+- -> nu_mu on "
      "entries where our K_mu3 share is below 0.1% (MCEq's block contains the K_mu3 "
      "neutrinos, three-body, not gated).")
    w("")
    for n in notes:
        w("- " + n)
    w("")
    w("## Result of the gated test")
    w("")
    w(f"Entries tested: {tot['n']}; failing: {tot['nfail']}; largest deviation "
      f"{fmt_pct(tot['max'])} at (k, m) = {tot['where']}"
      + (f" (parent {grid.e_grid[tot['where'][0]]:.3g} GeV, child bin centre "
         f"{grid.e_grid[tot['where'][1]]:.3g} GeV)" if tot["where"] else "") + ".")
    w("")
    w("| parent | child | entries | failing | max dev | median dev |")
    w("|---|---|---|---|---|---|")
    for par, child, flag, s in lines_g:
        w(f"| {par} | {child}{' (K_mu3-free entries)' if flag else ''} | {s['n']} | "
          f"{s['nfail']} | {fmt_pct(s['max'])} | {fmt_pct(s['med'])} |")
    w("")
    w("By parent energy (kinetic, GeV), gated entries of all gated pairs:")
    w("")
    w("| parent energy | pion parents: entries / failing / max | "
      "kaon parents: entries / failing / max |")
    w("|---|---|---|")
    for lo, hi in E_RANGES:
        cells = []
        for key in ("pi", "K"):
            s = summarise(by_range.get((key, lo, hi), []))
            cells.append(f"{s['n']} / {s['nfail']} / {fmt_pct(s['max'])}")
        w(f"| {lo:g} to {hi:g} | {cells[0]} | {cells[1]} |")
    w("")
    w("### Diagnosis (analytic two-body kernels, same gated entries)")
    w("")
    w("| variant of our analytic kernel | entries | failing | max dev | median dev |")
    w("|---|---|---|---|---|")
    for label, s in diag.items():
        w(f"| {label} | {s['n']} | {s['nfail']} | {fmt_pct(s['max'])} | "
          f"{fmt_pct(s['med'])} |")
    w("")
    w("The gate is evaluated on the first row. The other rows are diagnosis, not a "
      "redefinition: they show what MCEq's blocks contain (see below).")
    w("")
    w("## Reported, not gated: three-body kaon decays, K_L, K_S, muon decay")
    w("")
    w("Per (parent, child) and parent-energy range: entries above 1% of MCEq's "
      "column sum (all parent bins with centre in 0.1 .. 1000 GeV), median and 90th "
      "percentile of |ours/MCEq - 1|; median over parent bins of the total yield "
      "ratio and of the ratio of mean child kinetic energies (ours/MCEq); "
      "`z` = median |ours - MCEq| / sigma_stat over the entries that have a "
      "statistical error (sigma_stat from the two disjoint halves).")
    w("")
    w("| parent | child | kernel | range GeV | entries | med dev | p90 dev | "
      "yield ratio | <T_c> ratio | median z |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for par, child, kind, ours, mce, kern in rep:
        hd = kern.dense(par, child, "hd")[0]
        for lo, hi in E_RANGES:
            ks = erange_ks(grid, ks_all, lo, hi)
            if len(ks) == 0:
                continue
            rows = compare(ours, mce, ks, hd=hd)
            if not rows:
                continue
            rel = np.array([abs(r[4]) for r in rows])
            z = np.array([abs(r[5]) for r in rows if np.isfinite(r[5])])
            yr = np.nanmedian(yield_ratio(ours, mce, grid, ks))
            er = np.nanmedian(mean_energy_ratio(ours, mce, grid, ks))
            w(f"| {par} | {child} | {kind} | {lo:g}-{hi:g} | {len(rows)} | "
              f"{fmt_pct(np.median(rel))} | {fmt_pct(np.percentile(rel, 90))} | "
              f"{yr:.4f} | {er:.4f} | "
              f"{(f'{np.median(z):.2f}' if len(z) else 'n/a')} |")
    w("")
    w("## Statistical error of the sampled moments (kernel file under test)")
    w("")
    w("Entries above 1% of the column sum of the same pair (parent bins as above). "
      "`hd = (A - B)/2` from the two disjoint halves estimates the 1-sigma error of "
      "the full-sample moment. Columns: median over entries of sigma(mom_l)/mom_0 "
      "(error of the normalised moment `<P_l>` times the yield error is dominated "
      "by it), and of sigma(mom_0)/mom_0, for l = 0, 1, 2, 10, 64; median number of "
      "samples per entry.")
    w("")
    w("| parent | child | n entries | sigma_0/mom_0 | l=1 | l=2 | l=10 | l=64 | "
      "median samples |")
    w("|---|---|---|---|---|---|---|---|---|")
    for par, child in sorted(K.pairs):
        if par in ("pi+", "pi-", "K_S0") or par not in PARENT_SAMPLED:
            continue
        if par.endswith("-") and not par.startswith("K_"):
            continue
        k_, m_, mom_, hd_, n_ = K.entries(par, child)
        mom0 = mom_[:, 0]
        ok = np.isin(k_, ks_all)
        colsum = np.zeros(grid.n)
        np.add.at(colsum, k_, mom0)
        ok &= mom0 > FRAC * colsum[k_]
        if not ok.any() or not np.any(hd_[ok, 0] != 0):
            continue
        r = np.abs(hd_[ok]) / mom0[ok, None]
        med = np.median(r, axis=0)
        w(f"| {par} | {child} | {int(ok.sum())} | {med[0]:.2g} | {med[1]:.2g} | "
          f"{med[2]:.2g} | {med[10]:.2g} | {med[64]:.2g} | "
          f"{int(np.median(n_[ok]))} |")
    w("")
    w("## Coverage")
    w("")
    w("Parents covered: " + ", ".join(sorted(covered_parents)) + ".")
    w("")
    w("MCEq decaying species with a block in MCEq's decay matrix but not covered "
      "(use MCEq's block, children straight ahead): " + ", ".join(missing_parents)
      + ".")
    w("")
    w("(parent > child) blocks of covered parents that we do not have: "
      + ", ".join(missing_children) + ".")
    w("")
    w("## Command")
    w("")
    w("```")
    w("cd /afs/cern.ch/work/p/pigrange/daemonflux && source "
      "/home/pigrange/mc3d_work/runs/native3d/env.sh")
    w(cmd)
    w("```")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", a.out, "verdict", verdict, "tested", tot["n"], "failing",
          tot["nfail"], "max", tot["max"])
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
