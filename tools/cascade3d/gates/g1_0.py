"""Gate G1.0 (JAX set-up) for cascade3d; writes ``$N3/c1/gates/G1.0.md``.

(a) ``test_cascade3d.py::test_x64_enforced`` must pass (fails if 64-bit is off).
(b) CPU time / wall time of a solve <= number of pinned cores (measured with
    ``os.times`` around the solve, affinity from ``os.sched_getaffinity``).
(c) ``jax.jvp`` of the straight-ahead solve with respect to one kernel block
    (the interaction yield p+ -> pi+, tangent direction = the block itself) against a
    central finite difference (relative step 1e-4): every cell of the five G1.1
    quantities, six G1.1 lines, 0.1 to 100 GeV, within 1e-6 relative.

The solve is PHYS with energy loss on, on the six G1.1 lines (straight-ahead lines
are independent, so this is the same per-line computation as the full family).

Usage::

    taskset -c 0-3 $PY tools/cascade3d/gates/g1_0.py
"""

import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import assemble as A  # noqa: E402
import grid  # noqa: E402
import kernelset as K  # noqa: E402
import rays  # noqa: E402
import solver as S  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

N3 = os.environ.get("N3", "/home/pigrange/mc3d_work/runs/native3d")
OUT = os.path.join(N3, "c1", "gates", "G1.0.md")
TOL_C = 1e-6
EPS = 1e-4

GATE = """**G1.0 JAX set-up (gated, before any physics gate).**
(a) A test fails if 64-bit mode is off.
(b) Measured CPU time over wall time of a solve does not exceed the number of pinned
cores.
(c) `jax.jvp` of the straight-ahead solve with respect to one kernel block agrees with a
central finite difference to 1e-6 relative."""



def pinned():
    """The process affinity as a taskset list, e.g. "0-3"."""
    c = sorted(os.sched_getaffinity(0))
    parts, start = [], c[0]
    for a, b in zip(c, c[1:] + [None]):
        if b != a + 1:
            parts.append(f"{start}-{a}" if a > start else f"{a}")
            start = b
    return ",".join(parts)

def ground(sv, out):
    """Five G1.1 quantities on the ground lines as one jnp array (5, nl, nE)."""
    nu = A.sink_intensity(sv, out["emis"])[:, :, jnp.asarray(sv.rays.ground_node)]
    Fg = out["F_ground"]
    names = list(sv.ks.species)
    res = []
    for q, sps in A.QUANTITIES.items():
        v = 0.0
        for s in sps:
            i = names.index(s)
            if i in sv.Sk:
                v = v + nu[:, sv.Sk.index(i), :].T
            else:
                v = v + Fg[:, sv.T.index(i), :].T
        res.append(v)
    return jnp.stack(res)


def main():
    lines = ["# Gate G1.0: JAX set-up", "", GATE, ""]
    # (a)
    cmd = [sys.executable, "-m", "pytest", "-q",
           os.path.join(HERE, "..", "test_cascade3d.py") + "::test_x64_enforced"]
    pa = subprocess.run(cmd, capture_output=True, text=True)
    ok_a = pa.returncode == 0
    lines += ["## (a) 64-bit enforced", "",
              f"`{' '.join(cmd[2:])}` -> return code {pa.returncode}: "
              f"`{pa.stdout.strip().splitlines()[-1] if pa.stdout else ''}`.  "
              f"`jax_enable_x64` = {jax.config.jax_enable_x64}.  "
              f"**{'PASS' if ok_a else 'FAIL'}**", ""]
    # set-up
    atm = grid.CorsikaAtmosphere()
    lev = grid.make_levels(atm)
    fam = grid.LineFamily(np.array(grid.G11_COSZ), np.zeros(0))
    r = rays.Rays(atm, lev, fam)
    ks = K.from_mceq("PHYS", loss=True)
    sv = S.Solver(r, ks)
    prm = {k: jnp.asarray(v) for k, v in ks.params().items()}
    j = ks.channel("int", "p+", "pi+")
    # (b): warm-up (compile) then a timed solve
    ground(sv, sv.solve(prm)).block_until_ready()
    aff = sorted(os.sched_getaffinity(0))
    t0, c0 = time.time(), os.times()
    ground(sv, sv.solve(prm)).block_until_ready()
    t1, c1 = time.time(), os.times()
    wall = t1 - t0
    cpu = (c1.user - c0.user) + (c1.system - c0.system)
    ratio = cpu / wall
    ok_b = ratio <= len(aff) + 1e-9
    lines += ["## (b) CPU use inside the pinned cores", "",
              f"Affinity {aff} ({len(aff)} cores).  Solve (compiled): "
              f"wall {wall:.2f} s, "
              f"CPU {cpu:.2f} s, CPU/wall = {ratio:.2f}.  "
              f"**{'PASS' if ok_b else 'FAIL'}**", ""]
    # (c)
    E = ks.e_grid
    sel = (E >= 0.1) & (E <= 100.0)

    def f(Yblk):
        p = dict(prm)
        p["Y"] = prm["Y"].at[j].set(Yblk)
        return ground(sv, sv.solve(p))

    Y0 = prm["Y"][j]
    t0 = time.time()
    val, tan = jax.jvp(f, (Y0,), (Y0,))
    tan.block_until_ready()
    t_jvp = time.time() - t0
    fp = f(Y0 * (1 + EPS))
    fm = f(Y0 * (1 - EPS))
    fd = (fp - fm) / (2 * EPS)
    tan, fd, val = (np.asarray(x)[..., sel] for x in (tan, fd, val))
    rel = np.abs(tan - fd) / np.maximum(np.abs(fd), 1e-300)
    sig = np.abs(fd) > 1e-8 * np.abs(val)   # cells the block actually moves
    worst = float(np.max(rel[sig]))
    w = np.unravel_index(np.argmax(np.where(sig, rel, 0)), rel.shape)
    qn = list(A.QUANTITIES)
    ok_c = worst <= TOL_C
    lines += ["## (c) jvp against central finite difference", "",
              f"Block: interaction yield `p+ -> pi+` (channel {j}), tangent = "
              f"the block "
              f"itself; FD relative step {EPS:g}.  Cells: 5 quantities x 6 lines x "
              f"{int(sel.sum())} bins; {int(sig.sum())} cells with "
              f"|dPhi| > 1e-8 Phi compared.",
              f"Largest |jvp - FD| / |FD| = {worst:.2e} ({qn[w[0]]}, cos Z "
              f"{r.ground_mu0[w[1]]:.2f}, E {E[sel][w[2]]:.3g} GeV); median "
              f"{np.median(rel[sig]):.1e}.  Typical d ln Phi / d ln Y: "
              f"{np.median(np.abs(fd[sig] / val[sig])):.3f}.  jvp time {t_jvp:.1f} s.  "
              f"**{'PASS' if ok_c else 'FAIL'}**", ""]
    ok = ok_a and ok_b and ok_c
    lines += [f"## Result: **{'PASS' if ok else 'FAIL'}**", "",
              f"Command: `taskset -c {pinned()} $PY tools/cascade3d/gates/g1_0.py`", ""]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    with open(OUT.replace(".md", ".json"), "w") as fh:
        json.dump({"a": ok_a, "b_ratio": ratio, "b_cores": len(aff),
                   "c_worst": worst, "pass": ok}, fh)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
