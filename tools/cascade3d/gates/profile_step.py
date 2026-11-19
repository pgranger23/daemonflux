"""Time the pieces of one energy step on the default full geometry (cost study, G1.5).

Usage::

    taskset -c 0-3 $PY tools/cascade3d/gates/profile_step.py [--angular]
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import grid  # noqa: E402
import kernelset as K  # noqa: E402
import rays  # noqa: E402
import solver as S  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402


def timeit(f, *a, n=5):
    jax.block_until_ready(f(*a))
    t = time.time()
    for _ in range(n):
        jax.block_until_ready(f(*a))
    return (time.time() - t) / n


def main():
    angular = "--angular" in sys.argv
    atm = grid.CorsikaAtmosphere()
    lev = grid.make_levels(atm)
    r = rays.Rays(atm, lev, grid.make_line_family(lev))
    ks = K.from_mceq("PHYS", loss=True)
    if angular:
        ks = K.synthetic_angular(ks)
    sv = S.Solver(r, ks)
    prm = {k: jnp.asarray(v) for k, v in ks.params().items()}
    carry = sv.init_carry()
    nH, N = sv.nH, r.N
    Lam = jnp.full(nH, 0.01)
    De = jnp.full(nH, 1e-5)
    Snod = jnp.ones((nH, N))
    sw = jax.jit(sv._seg_weights)
    T, c = sw(Lam, De)
    print("seg_weights", timeit(sw, Lam, De))
    ss = jax.jit(sv._seg_source)
    print("seg_source", timeit(ss, Snod, c))
    Q = ss(Snod, c)
    cs = jax.jit(sv._chain_scan)
    print("chain_scan", timeit(cs, T, Q, jnp.ones(nH)))
    A = jnp.ones((sv.nL, r.Np + 1)) * 0.9
    ps = jax.jit(sv._point_scan)
    print("point_scan", timeit(ps, A, A))

    def push(S_acc, Y, F):
        return S_acc + jnp.einsum("cpm,pn->cmn", Y, F)

    pj = jax.jit(push, donate_argnums=0)
    Y = jnp.ones((sv.nC, 2 * sv.nT, sv.nE))
    F = jnp.ones((2 * sv.nT, N))
    S_acc = carry[0]
    S_acc = pj(S_acc, Y, F)
    jax.block_until_ready(S_acc)
    t = time.time()
    for _ in range(3):
        S_acc = pj(S_acc, Y, F)
    jax.block_until_ready(S_acc)
    print("push", (time.time() - t) / 3)
    del S_acc
    c0 = sv.init_carry()
    nL = sv.nL
    lt = jax.jit(lambda k, S, fw, sw, p: sv._loss_transport(
        k, S, jnp.zeros(nL), jnp.full(nL, 1e-6), jnp.zeros(nL), fw, sw, p))
    SL = jnp.ones((nL, N)) * jnp.asarray(r.node_rho)[None]
    print("loss_transport", timeit(lt, jnp.int32(20), SL, c0[2], c0[3], prm, n=3))
    P1 = r.Np + 1
    xs = {"C1": jnp.full((nL, P1), 0.9), "B0": jnp.ones((nL, P1)),
          "fup": jnp.ones((nL, P1, 4)), "cur": jnp.zeros((nL, P1, 4), bool),
          "wl": jnp.full((nL, P1, 4), 0.25), "wpre": jnp.full((nL, P1, 4), 0.25)}
    pn = jax.jit(sv._point_scan_nl)
    print("point_scan_nl", timeit(pn, xs, n=3))
    print("loss points", r.Np, "window", c0[2].shape)
    for k in (100, 60, 20):
        stepf = sv._steps[(bool(sv.nA and k < sv.nA), bool(sv.K_up > 0
                                                           and k <= sv.k_up))]
        t = time.time()
        c2, o = stepf(jnp.int32(k), sv.init_carry(), prm, jnp.asarray(1.0))
        jax.block_until_ready(o["res"])
        t1 = time.time() - t
        del c2
        carry = sv.init_carry()
        jax.block_until_ready(carry)
        t = time.time()
        c2, o = stepf(jnp.int32(k), carry, prm, jnp.asarray(1.0))
        jax.block_until_ready(o["res"])
        print(f"full step k={k}: first {t1:.2f} s, then {time.time() - t:.2f} s")
        del c2


if __name__ == "__main__":
    main()
