"""Gate G1.2 (numerical convergence); writes ``$N3/c1/gates/G1.2.md``.

Doubling one at a time the altitude grid, the line family (mu grid), the Legendre
order L and the ray sampling (sub-intervals per segment and loss-step grammage) must
move no G1.1 cell by more than 0.5%; the same-bin sweep residual must be below 1e-4
in every bin.  The G1.3-cell part waits for the real kernels.

Part A, straight-ahead (MCEq kernels), PHYS, loss off and on, G1.1 cells.  The six
G1.1 lines are solved alone (straight-ahead lines do not couple); the equivalence with
the full family is checked on the base setting.  L does not enter (g = 1: the angular
code is not compiled), and the line family does not change the G1.1 lines, so for
straight-ahead the family doubling is covered by the equivalence check.

Part B, synthetic angular kernels (``kernelset.synthetic_angular``: Gaussian in theta
with pT 0.35 GeV for every hadronic interaction channel below 3 TeV, L = 32), PHYS,
loss on, full family: base and the four doublings, G1.1 cells.

Every run is a separate ``run_solve.py`` process (cost and memory per run are
recorded in its npz).  Usage::

    taskset -c 0-3 $PY tools/cascade3d/gates/g1_2.py [A] [B] [C]

Part C (cycle 2): the Monte Carlo-matched set pool2000 (L = 32, loss on, R = 6371
km): the four doublings on the G1.1 cells and on the G1.3 cell averages; its base
run is the G1.5 cost configuration; L = 16 as a diagnostic.
"""

import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N3 = os.environ.get("N3", "/home/pigrange/mc3d_work/runs/native3d")
OUTDIR = os.environ.get("C3D_OUTDIR", os.path.join(N3, "c1", "cycle2"))
RUNS = os.path.join(OUTDIR, "g12")
OUT = os.path.join(OUTDIR, "gates", "G1.2.md")
REF = os.path.join(N3, "c1", "ref", "mceq_g11_PHYS_losson.npz")
QUANT = ("numu", "antinumu", "nue", "antinue", "mu")
COSZ = (1.0, 0.75, 0.5, 0.25, 0.15, 0.05)
TOL = 0.005

GATE = """**G1.2 Numerical convergence (gated).** Doubling, one at a time, the altitude
grid, the mu grid, the Legendre order L and the ray sampling moves no G1.1 or G1.3 cell
by more
than 0.5%. The same-bin sweep residual is below 1e-4 in every energy bin."""

VARIANTS = {
    "base": [],
    "levels2": ["--levels-refine", "2"],
    "family2": ["--family-refine", "2"],
    "L2": ["--L", "64"],
    "sampling2": ["--n-sub", "8", "--dX-max", "2.5"],
}



def pinned():
    """The process affinity as a taskset list, e.g. "0-3"."""
    c = sorted(os.sched_getaffinity(0))
    parts, start = [], c[0]
    for a, b in zip(c, c[1:] + [None]):
        if b != a + 1:
            parts.append(f"{start}-{a}" if a > start else f"{a}")
            start = b
    return ",".join(parts)

def run(tag, args):
    os.makedirs(RUNS, exist_ok=True)
    out = os.path.join(RUNS, f"{tag}.npz")
    if os.path.exists(out):
        return out
    cmd = [sys.executable, os.path.join(HERE, "run_solve.py")] + args + ["--out", out]
    log = open(os.path.join(RUNS, f"{tag}.log"), "w")
    subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    return out


def cells(path):
    z = np.load(path)
    E = z["e_grid"]
    sel = (E >= 0.1 - 1e-9) & (E <= 100 + 1e-6)
    rows = [int(np.flatnonzero(np.isclose(z["mu0"], c))[0]) for c in COSZ]
    v = np.stack([z[q][rows][:, sel] for q in QUANT])   # (5, 6, nsel)
    return v, E[sel], json.loads(str(z["cost_json"])), np.asarray(z["res"])


def compare(base, var):
    vb, E, _, _ = cells(base)
    vv, _, _, _ = cells(var)
    rel = vv / vb - 1
    i = np.unravel_index(np.argmax(np.abs(rel)), rel.shape)
    return float(rel[i]), (QUANT[i[0]], COSZ[i[1]], float(E[i[2]]))


def compare_g13(base, var):
    """Largest move of the G1.3 cell averages (reported, not gated here)."""
    a, b = np.load(base), np.load(var)
    if "cell_nue" not in a.files or "cell_nue" not in b.files:
        return None
    worst = (0.0, None)
    be, ce = a["cell_band_edges"], a["cell_cosz_edges"]
    for q in ("nue", "antinue", "numu", "antinumu"):
        r = b[f"cell_{q}"] / a[f"cell_{q}"] - 1
        i = np.unravel_index(np.argmax(np.abs(r)), r.shape)
        if abs(r[i]) > abs(worst[0]):
            worst = (float(r[i]), (q, float(be[i[0]]), float(be[i[0] + 1]),
                                   float(ce[i[1]]), float(ce[i[1] + 1])))
    return worst


def main(argv):
    parts = argv[1:] or ["A", "B"]
    lines = ["# Gate G1.2: numerical convergence", "", GATE, "",
             __doc__.split("Every run")[0].split("in every bin.", 1)[1].strip(), ""]
    ok_all = True
    summary = {}
    if "A" in parts:
        for loss in ("lossoff", "losson"):
            base = run(f"A_{loss}_base", ["PHYS", loss, "--gate-lines"])
            vals = []
            for tag in ("levels2", "sampling2"):
                p = run(f"A_{loss}_{tag}", ["PHYS", loss, "--gate-lines"]
                        + VARIANTS[tag])
                vals.append((tag,) + compare(base, p))
            full = run(f"A_{loss}_fullfamily", ["PHYS", loss])
            vals.append(("full family vs G1.1 lines",) + compare(base, full))
            res = float(np.max(cells(base)[3]))
            lines += [f"## A. Straight-ahead, PHYS, {loss}", "",
                      "| change | largest cell move | cell (quantity, cos Z, E GeV) | "
                      "within 0.5% |", "|---|---|---|---|"]
            for tag, m, w in vals:
                ok = abs(m) <= TOL
                ok_all &= ok
                lines.append(f"| {tag} | {m:+.2e} | {w[0]}, {w[1]:g}, {w[2]:.3g} | "
                             f"{'yes' if ok else 'NO'} |")
                summary[f"A_{loss}_{tag}"] = m
            ok_r = res < 1e-4
            ok_all &= ok_r
            lines += ["", "L and the line family: not applicable to the straight-ahead "
                      "G1.1 lines (see above).  Sweep residual, largest over bins: "
                      f"{res:.1e} ({'below' if ok_r else 'NOT below'} 1e-4).", ""]
    if "B" in parts:
        base = run("B_base", ["PHYS", "losson", "--angular"])
        vals = []
        for tag in ("levels2", "family2", "L2", "sampling2"):
            p = run(f"B_{tag}", ["PHYS", "losson", "--angular"] + VARIANTS[tag])
            vals.append((tag,) + compare(base, p))
        res = float(np.max(cells(base)[3]))
        cost = cells(base)[2]
        lines += ["## B. Synthetic angular kernels, PHYS, loss on, full family", "",
                  "| change | largest cell move | cell (quantity, cos Z, E GeV) | "
                  "within 0.5% |", "|---|---|---|---|"]
        for tag, m, w in vals:
            ok = abs(m) <= TOL
            ok_all &= ok
            lines.append(f"| {tag} | {m:+.2e} | {w[0]}, {w[1]:g}, {w[2]:.3g} | "
                         f"{'yes' if ok else 'NO'} |")
            summary[f"B_{tag}"] = m
        lines += ["", "Reported, not gated (stage B gates them with the real "
                  "kernels): largest move of the G1.3 cell averages "
                  "(`assemble.cell_averages`, 4 species x 10 bands x 10 cos Z bins).",
                  "", "| change | largest move | cell (species, band GeV, cos Z) |",
                  "|---|---|---|"]
        for tag in ("levels2", "family2", "L2", "sampling2"):
            w = compare_g13(os.path.join(RUNS, "B_base.npz"),
                            os.path.join(RUNS, f"B_{tag}.npz"))
            if w is not None and w[1] is not None:
                c = w[1]
                lines.append(f"| {tag} | {w[0]:+.2e} | {c[0]}, {c[1]:.3g}-{c[2]:.3g}, "
                             f"{c[3]:g}-{c[4]:g} |")
        ok_r = res < 1e-4
        ok_all &= ok_r
        lines += ["", f"Sweep residual, largest over bins: {res:.1e} "
                  f"({'below' if ok_r else 'NOT below'} 1e-4).  Base run: "
                  f"{cost['nodes']} nodes, {cost['total_cpu_s'] / 60:.1f} core-min, "
                  f"{cost['peak_rss_GB']:.1f} GB.", ""]
        # angular effect itself, for orientation
        sa = os.path.join(RUNS, "A_losson_fullfamily.npz")
        if os.path.exists(sa):
            m, w = compare(sa, base)
            lines += [f"For orientation: synthetic 3D over straight-ahead, largest "
                      f"cell "
                      f"ratio - 1 = {m:+.3f} ({w[0]}, cos Z {w[1]:g}, {w[2]:.3g} GeV).",
                      ""]
    if "C" in parts:
        ksd = os.path.join(N3, "kernels", "mc")
        own = os.path.join(N3, "c1", "cycle2", "kernels")
        base_args = ["MC", "losson", "--r-earth", "6371", "--ks",
                     os.path.join(ksd, "ks_mc_pool2000_L32_losson.npz")]
        varc = {"levels2": VARIANTS["levels2"], "family2": VARIANTS["family2"],
                "sampling2": VARIANTS["sampling2"]}
        base = run("C_base", base_args)
        vals = []
        for tag in ("levels2", "family2", "L2", "sampling2"):
            if tag == "L2":
                args = base_args[:-1] + [os.path.join(
                    own, "ks_mc_pool2000_L64_losson.npz")]
            else:
                args = base_args + varc[tag]
            p = run(f"C_{tag}", args)
            vals.append((tag, compare(base, p), compare_g13(base, p)))
        l16 = run("C_L16", base_args[:-1] + [os.path.join(
            own, "ks_mc_pool2000_L16_losson.npz")])
        cost = cells(base)[2]
        res = float(np.max(cells(base)[3]))
        lines += ["## C. Monte Carlo-matched set pool2000 (L = 32), loss on, "
                  "R = 6371 km, full family", "",
                  "G1.1 cells (5 quantities x 6 directions x 0.1-100 GeV) and the "
                  "G1.3 cell averages (4 species x 10 bands x 10 cos Z bins).", "",
                  "| change | G1.1 cells: largest move | where | G1.3 cells: largest "
                  "move | where | within 0.5% |", "|---|---|---|---|---|---|"]
        for tag, (m, w), g13 in vals:
            okc = abs(m) <= TOL and (g13 is None or abs(g13[0]) <= TOL)
            ok_all &= okc
            gtxt = ("n/a", "") if g13 is None else (
                f"{g13[0]:+.2e}", f"{g13[1][0]}, {g13[1][1]:.3g}-{g13[1][2]:.3g} GeV, "
                f"cos Z {g13[1][3]:g}-{g13[1][4]:g}")
            lines.append(f"| {tag} | {m:+.2e} | {w[0]}, {w[1]:g}, {w[2]:.3g} | "
                         f"{gtxt[0]} | {gtxt[1]} | {'yes' if okc else 'NO'} |")
            summary[f"C_{tag}"] = [m, None if g13 is None else g13[0]]
        m16, w16 = compare(base, l16)
        g16 = compare_g13(base, l16)
        lines += ["", f"Diagnostic: L = 16 instead of 32 moves the G1.1 cells by "
                  f"{m16:+.2e} ({w16[0]}, cos Z {w16[1]:g}, {w16[2]:.3g} GeV) and the "
                  f"G1.3 cells by {g16[0]:+.2e} ({g16[1]}).",
                  f"Sweep residual {res:.1e}; up-scatter pass sizes "
                  f"{cost.get('up_res')}.  Base run (the G1.5 configuration): "
                  f"{cost['nodes']} nodes, {cost['loss_points']} loss points, "
                  f"k_top {cost.get('k_top')}, {cost['total_cpu_s'] / 60:.1f} core-min "
                  f"(user+sys incl. compilation), {cost['peak_rss_GB']:.1f} GB, "
                  f"wall {cost['total_wall_s'] / 60:.1f} min on cores "
                  f"{cost['affinity']}.", ""]
    lines += ["Per-run cost (CPU = user + sys of the process, peak RSS):", "",
              "| run | nodes | loss points | core-min | peak GB |",
              "|---|---|---|---|---|"]
    for f in sorted(os.listdir(RUNS)):
        if f.endswith(".npz"):
            c = json.loads(str(np.load(os.path.join(RUNS, f))["cost_json"]))
            lines.append(f"| {f[:-4]} | {c['nodes']} | {c['loss_points']} | "
                         f"{c['total_cpu_s'] / 60:.1f} | {c['peak_rss_GB']:.1f} |")
    lines += ["", f"## Result: **{'PASS' if ok_all else 'FAIL'}**", "",
              f"Command: `taskset -c {pinned()} $PY tools/cascade3d/gates/g1_2.py "
              f"{' '.join(parts)}`", ""]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines))
    with open(OUT.replace(".md", ".json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print("\n".join(lines))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
