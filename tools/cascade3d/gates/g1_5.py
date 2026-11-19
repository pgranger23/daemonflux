"""Gate G1.5 (cost); writes G1.5.md / G1.5.json.

"One C1 solve, all species, measured: at most 10 core-minutes and 8 GB."

The gate figure is one cold solve of the G1.3(a) configuration (frozen settings,
``run_solve.py MC losson --r-earth 6371 --ks ks_mc_pool2000_L32_losson.npz``) in a
fresh process pinned to ``--cores``, under ``/usr/bin/time -v``.  CPU = user + system
of that process (set-up, JIT compilation, solve and cell assembly: everything the
process does); memory = its peak resident set.  The script's own figures
(``cost_json``) are reported next to ``time -v`` as a cross-check; the verdict uses the
larger of the two CPU figures and of the two memory figures.  ``uptime`` is recorded
just before and just after.  With ``--warm`` a second, separate process runs
``run_solve.py --warm`` and its second (compiled) solve is reported, labelled as not
the gate.  8 GB is read as 8 x 10^9 bytes; the GiB figure is given next to it.

Usage::

    $PY tools/cascade3d/gates/g1_5.py --cores 8-11 [--warm] [--outdir DIR]
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
N3 = os.environ.get("N3", "/home/pigrange/mc3d_work/runs/native3d")
KS = os.path.join(N3, "kernels", "mc", "ks_mc_pool2000_L32_losson.npz")
CPU_MAX_MIN, MEM_MAX_GB = 10.0, 8.0
GATE = ("**G1.5 Cost (gated).** One C1 solve, all species, measured: at most "
        "10 core-minutes and 8 GB.")


def uptime():
    return subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()


def timed(cores, out, extra, log):
    cmd = ["taskset", "-c", cores, "/usr/bin/time", "-v", "-o", out + ".time",
           sys.executable, os.path.join(HERE, "run_solve.py"), "MC", "losson",
           "--r-earth", "6371", "--ks", KS, "--out", out] + extra
    with open(log, "w") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    tv = {}
    for line in open(out + ".time"):
        if ": " in line:
            k, v = line.split(": ", 1)
            tv[k.strip()] = v.strip()
    import numpy as np
    cost = json.loads(str(np.load(out)["cost_json"]))
    return p.returncode, cmd, tv, cost


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", required=True)
    ap.add_argument("--warm", action="store_true")
    ap.add_argument("--outdir", default=os.path.join(N3, "c1", "g13", "g15"))
    args = ap.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    up0 = uptime()
    rc, cmd, tv, cost = timed(args.cores, os.path.join(args.outdir, "cold.npz"), [],
                              os.path.join(args.outdir, "cold.log"))
    up1 = uptime()
    user, sys_ = float(tv["User time (seconds)"]), float(tv["System time (seconds)"])
    rss_kb = float(tv["Maximum resident set size (kbytes)"])
    cpu_tv = (user + sys_) / 60.0
    cpu_sc = cost["total_cpu_s"] / 60.0
    mem_tv_gb = rss_kb * 1024 / 1e9
    mem_sc_gb = cost["peak_rss_GB"] * 1024 ** 3 / 1e9
    cpu, mem = max(cpu_tv, cpu_sc), max(mem_tv_gb, mem_sc_gb)
    ok = rc == 0 and cpu <= CPU_MAX_MIN and mem <= MEM_MAX_GB
    L = ["# Gate G1.5: cost", "", GATE, "",
         __doc__.split("Usage")[0].split('"\n', 1)[1].strip(), "",
         f"## Result: **{'PASS' if ok else 'FAIL'}**", "",
         "| figure | time -v | script (cost_json) | limit |", "|---|---|---|---|",
         f"| CPU, core-minutes (user+sys) | {cpu_tv:.2f} (user {user:.0f} s, "
         f"sys {sys_:.0f} s) | {cpu_sc:.2f} | {CPU_MAX_MIN:g} |",
         f"| peak RSS, GB (10^9 B) | {mem_tv_gb:.2f} ({rss_kb / 1024 ** 2:.2f} GiB) | "
         f"{mem_sc_gb:.2f} ({cost['peak_rss_GB']:.2f} GiB) | {MEM_MAX_GB:g} |",
         f"| wall | {tv.get('Elapsed (wall clock) time (h:mm:ss or m:ss)')} | "
         f"{cost['total_wall_s'] / 60:.2f} min | - |", "",
         f"Split (script): set-up {cost['setup_cpu_s'] / 60:.2f}, "
         f"solve {cost['solve_cpu_s'] / 60:.2f}, "
         f"assembly {cost['assemble_cpu_s'] / 60:.2f} core-min; CPU/wall "
         f"{cpu_sc * 60 / cost['total_wall_s']:.2f} on cores {cost['affinity']}; "
         f"exit {rc}; sweep residual {cost['res_max']:.1e}; {cost['nodes']} nodes, "
         f"{cost['levels']} levels, "
         f"{cost['lines']} lines, {cost['loss_points']} loss points.", "",
         f"Load: before `{up0}`; after `{up1}`.", "",
         f"Command (cold, the gate): `{' '.join(cmd)}`", ""]
    summary = {"pass": ok, "cpu_core_min_timev": cpu_tv, "cpu_core_min_script": cpu_sc,
               "peak_GB_timev": mem_tv_gb, "peak_GB_script": mem_sc_gb,
               "uptime_before": up0,
               "uptime_after": up1, "exit": rc, "command": cmd, "cost": cost}
    if args.warm:
        u2 = uptime()
        rcw, cmdw, tvw, cw = timed(args.cores, os.path.join(args.outdir, "warm.npz"),
                                   ["--warm"], os.path.join(args.outdir, "warm.log"))
        u3 = uptime()
        tot_w = (float(tvw["User time (seconds)"])
                 + float(tvw["System time (seconds)"])) / 60
        L += ["## Not the gate: warm second solve in the same process", "",
              f"Separate process `{' '.join(cmdw)}` (exit {rcw}). "
              f"Its first (cold) solve: "
              f"{cw['total_cpu_s'] / 60:.2f} core-min; second solve + assembly: "
              f"**{cw['warm_cpu_s'] / 60:.2f} core-min**, "
              f"wall {cw['warm_wall_s'] / 60:.2f} min; "
              f"peak RSS after both {cw['peak_rss_GB_after_warm']:.2f} GiB; "
              f"process total (time -v) {tot_w:.2f} "
              f"core-min.", "", f"Load: before `{u2}`; after `{u3}`.", ""]
        summary["warm"] = {"cost": cw, "uptime_before": u2, "uptime_after": u3,
                           "exit": rcw}
    with open(os.path.join(args.outdir, "G1.5.md"), "w") as fh:
        fh.write("\n".join(L))
    with open(os.path.join(args.outdir, "G1.5.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print("\n".join(L))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
