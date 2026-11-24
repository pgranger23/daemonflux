"""Gate G2.0 (cutoff sky against direct back-tracing); writes G2.0.md / .json.

Gate text: ``GATES.md`` section C2 and the entry "G2.0, the draw and the frames"
(2026-11-23), both quoted in the result file.  How it is evaluated:

- Draw: ``cutoff.sample_test_states(SEED, N_DRAW)`` with its default
  (``cap_deg=None``).  Location of a state: its line's first ground crossing or, for
  a line that misses the ground, the point under its tangent point
  (``boundary.state_location``).  Covered: that location within 10 deg of the site
  (and its four grid tables present).
- Sample: the shortest prefix of the draw that holds at least 20 000 covered states
  with R < 60 GV, of which at least 2 000 on lines that miss the ground.  Every
  covered state of that prefix is traced; the others are counted.
- Direct verdict: ``cutoff.trace_gated`` (``gate``, then ``trace_allowed``), pooled
  in calls of at least 10^4 states over forked workers.
- Sky: ``boundary.sky_transmission`` on the 25 tables of the 5-deg grid (h_ref = 0).
- Classes: R 0-5, 5-10, 10-20, 20-60 GV x cos of the zenith at injection 0-0.25,
  0.25-0.5, 0.5-1 x quadrants centred on N, E, S, W of the arrival azimuth in the
  local frame of the injection point.  Per class: mean sky transmission against the
  allowed fraction of the verdicts; trigger if they differ by more than
  max(0.02, 2 sigma), sigma = sqrt(p (1 - p) / n) of the allowed fraction p.
  Evaluated on all covered states, on those that hit the ground, and on those that
  miss it (there: the four rigidity classes only).  Empty classes are listed.
- Reported in addition: 500 covered states above 60 GV on lines that miss the
  ground (from the same draw, continued), traced; the sky gives them 1.

Usage::

    taskset -c 0-7 $PY tools/cascade3d/gates/g2_0.py --sky-dir DIR --outdir OUT \
        [--seed 2026112402] [--jobs 8] [--spacing 5]
    $PY tools/cascade3d/gates/g2_0.py --outdir OUT --spacing 5,1.25 \
        --reeval OUT --name G2.0_gridB          # no tracing

Exit status 0: no trigger; 1: a trigger; 2: input error.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
import boundary as B  # noqa: E402

N3 = os.environ.get("N3", "/home/pigrange/mc3d_work/runs/native3d")
R_CLASSES = (0.0, 5.0, 10.0, 20.0, 60.0)
MU_CLASSES = (0.0, 0.25, 0.5, 1.0)
QUADS = ("N", "E", "S", "W")
N_COVERED, N_MISS, N_HIGH = 20000, 2000, 500
COVER_DEG = 10.0

GATE = """\
**G2.0 Cutoff sky against direct back-tracing (reported, with a trigger).** The
transmission that `boundary.py` gives a primary (nucleus group, kinetic energy per
nucleon, point and direction on the injection sphere) against the verdict of
`tools/mc3d/cutoff.py` (`gate`, then `trace_gated`) for the same state. States: at least
20 000, drawn as the reference run draws its primaries (`cutoff.sample_test_states`),
rigidity below 60 GV; states that the sky does not cover are counted and left out.
Classes: rigidity 0 to 5, 5 to 10, 10 to 20 and 20 to 60 GV × cos of the zenith at
injection 0 to 0.25, 0.25 to 0.5 and 0.5 to 1 × four azimuth quadrants centred on
north, east, south and west. Per class: mean transmission of the sky against the
allowed fraction of the verdicts. A class in which they differ by more than
max(0.02, 2σ), σ the binomial error, triggers a check of the sky before the freeze.

**G2.0, the draw and the frames.** The gate text leaves two points open.
- Draw: `cutoff.sample_test_states(seed, n)` with its default (`cap_deg=None`),
  rigidity below 60 GV. With `cap_deg=10` the lines that miss the ground would be left
  out, and they have their own code path in `boundary.py`.
- Location of a state: its line's first ground crossing or, for a line that misses the
  ground, the point under its tangent point. A state is covered if that location is
  within 10° of the site. Only covered states are traced. The others are counted.
- Size: at least 20 000 covered states, of which at least 2 000 on lines that miss the
  ground.
- Quadrants: in the local frame of the injection point.
- The classes and the trigger of the gate are evaluated on all covered states and,
  separately, on those that hit and those that miss the ground. For the last subset the
  classes are the four rigidity classes only.
- Reported in addition: 500 states above 60 GV on lines that miss the ground, traced."""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_tables(sky_dir, grid):
    tabs, shas = {}, {}
    for i in range(grid.n):
        p = B.table_path(sky_dir, (grid.x[i], grid.y[i]), 0.0)
        if not os.path.exists(p):
            raise SystemExit(f"missing sky table {p}")
        tabs[i] = B.SkyTable.load(p)
        grid.check_table(tabs[i], i)
        shas[os.path.basename(p)] = sha256(p)
    return tabs, shas


def injection_frame(r, u):
    """cos of the zenith at injection and arrival azimuth [deg] in its local frame."""
    up, north, east = B.local_frame(r / np.linalg.norm(r, axis=1)[:, None])
    mu = -np.einsum("ij,ij->i", u, up)
    az = np.degrees(np.arctan2(-np.einsum("ij,ij->i", u, east),
                               -np.einsum("ij,ij->i", u, north))) % 360.0
    return mu, az


def quadrant(az):
    return (((az + 45.0) % 360.0) // 90.0).astype(int)      # 0 N, 1 E, 2 S, 3 W


def trace(r, u, R, jobs, work, log):
    """trace_gated pooled in chunks of >= 1e4 states; resumable per chunk."""
    import multiprocessing as mp

    os.makedirs(work, exist_ok=True)
    n = r.shape[0]
    nch = max(1, n // 10000)
    chunks = [np.arange(j, n, nch) for j in range(nch)]
    out = np.zeros(n, bool)
    ntr, cpu = 0, 0.0
    todo = []
    for j, idx in enumerate(chunks):
        cf = os.path.join(work, f"chunk_{j:03d}_of_{nch}.npz")
        if os.path.exists(cf):
            with np.load(cf) as f:
                if np.array_equal(f["idx"], idx) and np.array_equal(f["R"], R[idx]):
                    out[idx] = f["allowed"]
                    ntr += int(f["n_traced"])
                    cpu += float(f["cpu_s"])
                    continue
            raise SystemExit(f"{cf} belongs to another sample")
        todo.append((j, idx, r[idx], u[idx], R[idx], "trace_gated"))
    if todo:
        with mp.get_context("fork").Pool(min(jobs, len(todo))) as pool:
            for j, idx, a, nt, c in pool.imap_unordered(B._pool_worker, todo):
                cf = os.path.join(work, f"chunk_{j:03d}_of_{nch}.npz")
                np.savez(cf + ".tmp.npz", idx=idx, R=R[idx], allowed=a,
                         n_traced=np.array(nt), cpu_s=np.array(c))
                os.replace(cf + ".tmp.npz", cf)
                out[idx] = a
                ntr += nt
                cpu += c
                log(f"chunk {j}: {idx.size} states, {nt} traced, {c:.0f} s")
    return out, ntr, cpu


def class_table(sel, R, mu, quad, T, A, rigidity_only=False):
    rows = []
    ri = np.digitize(R, R_CLASSES[1:-1])
    mi = np.clip(np.digitize(mu, MU_CLASSES[1:-1]), 0, 2)
    for a in range(4):
        mus = [None] if rigidity_only else range(3)
        for b in mus:
            qs = [None] if rigidity_only else range(4)
            for q in qs:
                m = sel & (ri == a)
                if b is not None:
                    m &= mi == b
                if q is not None:
                    m &= quad == q
                n = int(m.sum())
                row = {"R": f"{R_CLASSES[a]:g}-{R_CLASSES[a + 1]:g}",
                       "mu": "all" if b is None else
                       f"{MU_CLASSES[b]:g}-{MU_CLASSES[b + 1]:g}",
                       "quad": "all" if q is None else QUADS[q], "n": n}
                if n == 0:
                    row.update({"sky": None, "allowed": None, "sigma": None,
                                "diff": None, "tol": None, "trigger": False})
                else:
                    p = float(A[m].mean())
                    s = float(np.sqrt(p * (1 - p) / n))
                    t = float(T[m].mean())
                    tol = max(0.02, 2 * s)
                    row.update({"sky": t, "allowed": p, "sigma": s, "diff": t - p,
                                "tol": tol, "trigger": bool(abs(t - p) > tol)})
                rows.append(row)
    return rows


def main(argv=None):
    import cutoff

    ap = argparse.ArgumentParser()
    ap.add_argument("--sky-dir", default=os.path.join(N3, "c2", "solver", "sky"))
    ap.add_argument("--outdir", default=os.path.join(N3, "c2", "solver", "gates",
                                                     "G2.0"))
    ap.add_argument("--seed", type=int, default=2026112402)
    ap.add_argument("--block", type=int, default=200000)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--name", default="G2.0")
    ap.add_argument("--spacing", default="5",
                    help="location grid: one spacing or x,y [deg], e.g. 5,1.25")
    ap.add_argument("--reeval", default="",
                    help="directory of an earlier evaluation: its draw is regenerated "
                    "(checked against the stored rigidities) and its stored verdicts "
                    "are used; only the sky is recomputed, nothing is traced")
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)
    md = os.path.join(a.outdir, a.name + ".md")
    js = os.path.join(a.outdir, a.name + ".json")
    for p in (md, js):
        if os.path.exists(p):
            raise SystemExit(f"{p} exists; result files are never overwritten")
    w0, c0 = time.time(), time.process_time()
    grid = B.LocationGrid(B.parse_spacing(a.spacing), 10.0)
    tabs, shas = load_tables(a.sky_dir, grid)
    prev = None
    if a.reeval:
        with open(os.path.join(a.reeval, "G2.0.json")) as fh:
            prev = json.load(fh)
        st0 = np.load(os.path.join(a.reeval, "g2_0_states.npz"))
        if prev["seed"] != a.seed or prev["block"] != a.block:
            raise SystemExit("--seed/--block differ from the evaluation re-used")

    def log(s):
        print(s, flush=True)

    # draw in blocks (one seed per block: seed, seed + 1, ...) until the counts hold;
    # the sample of the gate is the shortest prefix that holds them
    keys = ("r_cm", "u", "R", "A", "Z", "e_kin")
    pool = {k: [] for k in keys + ("hit", "cov")}
    n_cov = n_miss = n_high = 0
    blk = 0
    while n_cov < N_COVERED or n_miss < N_MISS or n_high < N_HIGH:
        st = cutoff.sample_test_states(a.seed + blk, a.block)
        q, hit, _cz, _az, _ht = B.state_location(st["r_cm"], st["u"])
        x, y = B.xy_of(q)
        cov = np.hypot(x, y) <= COVER_DEG
        for k in keys:
            pool[k].append(st[k])
        pool["hit"].append(hit)
        pool["cov"].append(cov)
        lo = st["R"] < 60.0
        n_cov += int((cov & lo).sum())
        n_miss += int((cov & lo & ~hit).sum())
        n_high += int((cov & ~lo & ~hit).sum())
        blk += 1
    d = {k: np.concatenate(v) for k, v in pool.items()}
    lo = d["R"] < 60.0
    cm = np.cumsum(d["cov"] & lo)
    mm = np.cumsum(d["cov"] & lo & ~d["hit"])
    end = int(np.flatnonzero((cm >= N_COVERED) & (mm >= N_MISS))[0]) + 1
    pre = np.arange(d["R"].size) < end
    S = pre & d["cov"] & lo
    hm = np.flatnonzero(d["cov"] & ~lo & ~d["hit"])[:N_HIGH]
    log(f"draw: {blk} blocks of {a.block}; prefix {end} states; R<60 in prefix "
        f"{int((pre & lo).sum())}, covered {int(S.sum())} (miss "
        f"{int((S & ~d['hit']).sum())}), not covered {int((pre & lo).sum() - S.sum())}")
    idx = np.flatnonzero(S)
    r, u, R = d["r_cm"][idx], d["u"][idx], d["R"][idx]
    T, cov2, hit = B.sky_transmission(r, u, R, tabs, grid, COVER_DEG)
    if not np.all(cov2):
        raise SystemExit("a covered state has no tables")
    rh, uh, Rh = d["r_cm"][hm], d["u"][hm], d["R"][hm]
    if prev is None:
        A, ntr, cpu_t = trace(r, u, R, a.jobs, os.path.join(a.outdir, "work_main"),
                              log)
        Ah, ntrh, cpu_h = trace(rh, uh, Rh, a.jobs,
                                os.path.join(a.outdir, "work_high"), log)
    else:
        same = (blk == prev["blocks"] and end == prev["prefix"]
                and np.array_equal(idx, st0["idx"]) and np.array_equal(R, st0["R"])
                and np.array_equal(Rh, st0["high_R"])
                and np.array_equal(hit, st0["hit"]))
        if not same:
            raise SystemExit("the regenerated draw differs from the stored states")
        A, Ah = st0["allowed"], st0["high_allowed"]
        ntr, cpu_t = prev["traced"], prev["trace_cpu_s"]
        ntrh, cpu_h = prev["high_R_miss"]["traced"], prev["high_R_miss"]["cpu_s"]
        log(f"re-evaluation: draw and states of {a.reeval} reproduced exactly; "
            "stored verdicts used")
    Th, covh, _ = B.sky_transmission(rh, uh, Rh, tabs, grid, COVER_DEG)
    mu, az = injection_frame(r, u)
    qd = quadrant(az)
    allm = np.ones(idx.size, bool)
    tables = {"all": class_table(allm, R, mu, qd, T, A),
              "hit": class_table(hit, R, mu, qd, T, A),
              "miss": class_table(~hit, R, mu, qd, T, A, rigidity_only=True)}
    trig = {k: [r_ for r_ in v if r_["trigger"]] for k, v in tables.items()}
    any_trig = any(trig.values())
    sp = os.path.join(a.outdir, "g2_0_states.npz" if prev is None
                      else f"g2_0_states_{a.name}.npz")
    if os.path.exists(sp):
        raise SystemExit(f"{sp} exists")
    np.savez(sp, idx=idx, R=R, mu=mu, az=az, hit=hit, T=T, allowed=A, high_R=Rh,
             high_allowed=Ah, high_T=Th)
    res = {"gate": "G2.0", "outcome": "TRIGGER" if any_trig else "no trigger",
           "seed": a.seed, "block": a.block, "blocks": blk, "prefix": end,
           "n_R_below_60_in_prefix": int((pre & lo).sum()),
           "n_covered": int(idx.size), "n_covered_hit": int(hit.sum()),
           "n_covered_miss": int((~hit).sum()),
           "n_not_covered": int((pre & lo).sum() - idx.size),
           "traced": ntr, "trace_cpu_s": cpu_t,
           "overall": {"sky": float(T.mean()), "allowed": float(A.mean())},
           "high_R_miss": {"n": int(hm.size), "allowed_fraction": float(Ah.mean()),
                           "sky_mean": float(Th.mean()), "traced": ntrh,
                           "cpu_s": cpu_h, "R_min": float(Rh.min()),
                           "R_max": float(Rh.max())},
           "classes": tables, "triggers": trig, "sky_tables_sha256": shas,
           "grid": {"spacing_deg": list(grid.spacing_xy), "nodes": grid.n},
           "reeval_of": a.reeval or None,
           "wall_s": time.time() - w0, "driver_cpu_s": time.process_time() - c0,
           "command": " ".join(sys.argv)}
    with open(js, "w") as fh:
        json.dump(res, fh, indent=1)
    L = [f"# Gate G2.0: cutoff sky against direct back-tracing ({a.name})", "", GATE,
         "", f"## Outcome: **{res['outcome']}**", "",
         f"Draw: `cutoff.sample_test_states(seed, {a.block})` with seeds {a.seed} to "
         f"{a.seed + blk - 1}, concatenated; the sample is the shortest prefix "
         f"({end} states) holding >= {N_COVERED} covered states with R < 60 GV, "
         f">= {N_MISS} of them on lines that miss the ground. In the prefix: "
         f"{res['n_R_below_60_in_prefix']} states below 60 GV, covered "
         f"{res['n_covered']} (hit {res['n_covered_hit']}, miss "
         f"{res['n_covered_miss']}), not covered {res['n_not_covered']} (counted, "
         f"left out). Traced: {ntr} (the rest decided by `gate`), "
         f"{cpu_t / 60:.1f} core-min.", "",
         f"Location grid: {grid.label()} deg ({grid.n} tables)." + (
             f" Re-evaluation of `{a.reeval}`: the draw was regenerated and checked "
             "against the stored states (indices, rigidities, hit flags equal); "
             "the stored verdicts were used; only the sky was recomputed; nothing "
             "was traced." if prev else ""), "",
         f"All covered states: mean sky transmission {T.mean():.4f}, allowed "
         f"fraction {A.mean():.4f}.", "",
         f"Triggers: all {len(trig['all'])}, hit {len(trig['hit'])}, miss "
         f"{len(trig['miss'])}.", ""]
    for name in ("all", "hit", "miss"):
        L += [f"### Classes, {name} covered states", "",
              "| R [GV] | cos Z inj | quadrant | n | sky | allowed | sigma | diff | "
              "tol | trigger |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r_ in tables[name]:
            if r_["n"] == 0:
                L.append(f"| {r_['R']} | {r_['mu']} | {r_['quad']} | 0 | - | - | - | "
                         "- | - | empty |")
                continue
            L.append(f"| {r_['R']} | {r_['mu']} | {r_['quad']} | {r_['n']} | "
                     f"{r_['sky']:.4f} | {r_['allowed']:.4f} | {r_['sigma']:.4f} | "
                     f"{r_['diff']:+.4f} | {r_['tol']:.4f} | "
                     f"{'**YES**' if r_['trigger'] else 'no'} |")
        L.append("")
    h = res["high_R_miss"]
    L += ["### Reported: states above 60 GV on lines that miss the ground", "",
          f"{h['n']} covered states (R {h['R_min']:.1f} to {h['R_max']:.1f} GV), "
          f"traced with `trace_gated`: allowed fraction {h['allowed_fraction']:.4f} "
          f"({h['traced']} traced); the sky gives {h['sky_mean']:.4f} (transmission "
          "1 above 60 GV by convention).", "",
          f"Sky tables ({grid.label()}-deg grid, h_ref = 0), sha256:", ""]
    L += [f"- `{k}` {v}" for k, v in sorted(shas.items())]
    L += ["", f"Wall {res['wall_s'] / 60:.1f} min.", "", "Command:", "",
          f"    {res['command']}", ""]
    with open(md, "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L[:30]))
    return 1 if any_trig else 0


if __name__ == "__main__":
    sys.exit(main())
