"""Diagnostic (cycle 2b, task C): the real set with a ground layer of levels.

Set pool2000 (L 32, loss on, R 6371 km), ``make_levels(ground_layer=True)`` (levels
10, 30, 100, 300 m above the ground), base and the four doublings of G1.2 part C, with
the same comparison code (``g1_2.compare``, ``g1_2.compare_g13``).  Reported: the
largest move of the G1.1 cells per quantity, the number of muon cells outside 0.5%
per direction, the largest move of the G1.3 cells, nodes, loss points, cost; and the
difference between the ground-layer base and the default base.  Nothing is adopted.

Usage::

    $PY tools/cascade3d/gates/diag_ground_layer.py [variant ...]
        (runs the missing runs; ``report`` only writes the tables)
    env TAG=passthrough EXTRA="--levels-spec ..." for a second candidate
"""

import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g1_2 as G  # noqa: E402

N3 = G.N3
TAG = os.environ.get("TAG", "ground_layer")
EXTRA = os.environ.get("EXTRA", "--ground-layer").split()
DIR = os.path.join(N3, "c1", "cycle2b", "diagC", TAG)
KS = os.path.join(N3, "kernels", "mc", "ks_mc_pool2000_L32_losson.npz")
KS64 = os.path.join(N3, "c1", "cycle2", "kernels", "ks_mc_pool2000_L64_losson.npz")
DEFAULT_BASE = os.path.join(N3, "c1", "cycle2b", "g12", "C_base.npz")
VARIANTS = {"base": [], "levels2": ["--levels-refine", "2"],
            "family2": ["--family-refine", "2"], "L2": None,
            "sampling2": ["--n-sub", "8", "--dX-max", "2.5"]}
TOL = 0.005


def args_of(tag):
    ks = KS64 if tag == "L2" else KS
    extra = [] if VARIANTS[tag] is None else VARIANTS[tag]
    return ["MC", "losson", "--r-earth", "6371", "--ks", ks] + EXTRA + extra


def run(tag, cores):
    os.makedirs(DIR, exist_ok=True)
    out = os.path.join(DIR, f"{tag}.npz")
    if os.path.exists(out):
        return out
    cmd = ["taskset", "-c", cores, sys.executable, os.path.join(HERE, "run_solve.py")]
    cmd += args_of(tag) + ["--out", out]
    with open(os.path.join(DIR, f"{tag}.log"), "w") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    return out


def per_quantity(base, var):
    vb, E, _, _ = G.cells(base)
    vv, _, _, _ = G.cells(var)
    rel = vv / vb - 1                                    # (5, 6, nE)
    rows = []
    for iq, q in enumerate(G.QUANT):
        i = np.unravel_index(np.argmax(np.abs(rel[iq])), rel[iq].shape)
        rows.append((q, float(rel[iq][i]), G.COSZ[i[0]], float(E[i[1]])))
    imu = G.QUANT.index("mu")
    n_out = [int(np.sum(np.abs(rel[imu, j]) > TOL)) for j in range(len(G.COSZ))]
    return rows, n_out


def report():
    lines = [f"# Diagnostic: real set with `{' '.join(EXTRA)}` ({TAG})", "",
             __doc__.split("Usage")[0].strip(), ""]
    base = os.path.join(DIR, "base.npz")
    cb = G.cells(base)[2]
    lines += [f"Base: {cb['nodes']} nodes, {cb['loss_points']} loss points, "
              f"{cb['total_cpu_s'] / 60:.1f} core-min, {cb['peak_rss_GB']:.1f} GB "
              f"(default base: {G.cells(DEFAULT_BASE)[2]['nodes']} nodes, "
              f"{G.cells(DEFAULT_BASE)[2]['total_cpu_s'] / 60:.1f} core-min).", ""]
    all_in = True
    for tag in ("levels2", "family2", "L2", "sampling2"):
        v = os.path.join(DIR, f"{tag}.npz")
        if not os.path.exists(v):
            lines.append(f"## {tag}: not run")
            all_in = False
            continue
        rows, n_out = per_quantity(base, v)
        g13 = G.compare_g13(base, v)
        cv = G.cells(v)[2]
        lines += [f"## {tag} ({cv['nodes']} nodes, {cv['loss_points']} loss points, "
                  f"{cv['total_cpu_s'] / 60:.1f} core-min, {cv['peak_rss_GB']:.1f} GB)",
                  "", "| quantity | largest G1.1-cell move | cos Z | E [GeV] |",
                  "|---|---|---|---|"]
        for q, m, c, e in rows:
            lines.append(f"| {q} | {m:+.2e} | {c:g} | {e:.3g} |")
            all_in &= abs(m) <= TOL
        lines += ["", "Muon cells outside 0.5% per direction (cos Z "
                  + ", ".join(f"{c:g}" for c in G.COSZ) + "): "
                  + ", ".join(str(n) for n in n_out) + ".",
                  f"G1.3 cells: largest move {g13[0]:+.2e} ({g13[1]}).", ""]
    rows, n_out = per_quantity(DEFAULT_BASE, base)
    g13 = G.compare_g13(DEFAULT_BASE, base)
    lines += ["## Ground-layer base against the default base", "",
              "| quantity | largest difference | cos Z | E [GeV] |",
              "|---|---|---|---|"]
    for q, m, c, e in rows:
        lines.append(f"| {q} | {m:+.2e} | {c:g} | {e:.3g} |")
    lines += ["", f"G1.3 cells: largest difference {g13[0]:+.2e} ({g13[1]}).", "",
              f"Every G1.1 cell inside 0.5% under all four doublings: "
              f"**{'YES' if all_in else 'NO'}**.", ""]
    path = os.path.join(DIR, "DIAG.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    return all_in


def main(argv):
    tags = [a for a in argv[1:] if a in VARIANTS]
    cores = os.environ.get("CORES", "0-3")
    for t in tags:
        run(t, cores)
    if "report" in argv[1:]:
        report()


if __name__ == "__main__":
    main(sys.argv)
