"""Neutrino (sink) intensities from the emissivity, ground fluxes, cell averages.

Neutrinos are not transported: the intensity of a sink species at a node is the
line-of-sight integral of its emissivity upstream along the node's chain, with the
same log-linear source representation and sub-points as the transport (sigma = 0 is
the trapezoid limit of the segment formula).  The ground break resets it: in C1 no
neutrino comes up through the Earth (far side: stage C2).  ``DESIGN.md`` section 4.
"""

import jax
import jax.numpy as jnp
import numpy as np

import kernelset as KS

#: G1.1 quantities: name -> species names summed
QUANTITIES = {
    "numu": ("numu",), "antinumu": ("antinumu",), "nue": ("nue",),
    "antinue": ("antinue",),
    "mu": ("mu+_l", "mu+", "mu+_r", "mu-_l", "mu-", "mu-_r"),
}


def sink_intensity(solver, emis, chunk=48):
    """Intensity of every sink at every node.

    Parameters
    ----------
    solver : solver.Solver
    emis : array (nE, nSink, N)
        Emissivity per unit length (``Solver.solve()["emis"]``).

    Returns
    -------
    jax.Array (nE, nSink, N)
    """
    nE, nSk, N = emis.shape
    flat = jnp.reshape(emis, (nE * nSk, N))
    z = jnp.zeros(chunk)
    T, c = solver._seg_weights(z, z)

    @jax.jit
    def one(S):
        Q = solver._seg_source(S, c)
        return solver._chain_scan(T, Q, jnp.zeros(S.shape[0]))

    parts = []
    for i in range(0, flat.shape[0], chunk):
        blk = flat[i:i + chunk]
        n = blk.shape[0]
        if n < chunk:
            blk = jnp.concatenate([blk, jnp.zeros((chunk - n, N))], axis=0)
        parts.append(one(blk)[:n])
    return jnp.reshape(jnp.concatenate(parts, axis=0), (nE, nSk, N))


def ground_fluxes(solver, out, quantities=QUANTITIES):
    """Ground intensities of the G1.1 quantities for every ground line.

    Returns
    -------
    dict
        name -> numpy array (n_ground, nE) [1/(GeV cm2 s sr)], plus ``mu0``.
    """
    ks = solver.ks
    names = list(ks.species)
    gnode = solver.rays.ground_node
    res = {"mu0": solver.rays.ground_mu0.copy()}
    need_sink = any(names.index(s) in solver.Sk
                    for q in quantities.values() for s in q if s in names)
    if need_sink:
        nu = np.asarray(sink_intensity(solver, out["emis"]))[:, :, gnode]
    Fg = np.asarray(out["F_ground"])
    for q, sps in quantities.items():
        v = np.zeros((gnode.size, ks.nE))
        for s in sps:
            if s not in names:
                continue
            i = names.index(s)
            if i in solver.Sk:
                v += nu[:, solver.Sk.index(i), :].T
            else:
                v += Fg[:, solver.T.index(i), :].T
        res[q] = v
    return res


def band_average(e_grid, values, e_lo, e_hi, n=64):
    """Average of dPhi/dE over [e_lo, e_hi], log-log linear between bin centres.

    ``values`` (..., nE) positive; quadrature in ln E with ``n`` points.
    """
    v = np.asarray(values, float)
    le = np.log(e_grid)
    x = np.linspace(np.log(e_lo), np.log(e_hi), n + 1)
    xm = 0.5 * (x[1:] + x[:-1])
    lv = np.log(np.maximum(v, 1e-300))
    f = np.exp(np.apply_along_axis(lambda r: np.interp(xm, le, r), -1, lv))
    w = np.exp(xm) * np.diff(x)
    return np.sum(f * w, axis=-1) / (e_hi - e_lo)


def cosz_average(mu0, values, lo, hi):
    """Average over cos Z in [lo, hi] (trapezoid over the ground lines inside).

    The bin edges must be ground lines (``grid.G13_COSZ_EDGES`` are by default).
    ``values`` (n_ground, ...).
    """
    mu0 = np.asarray(mu0)
    sel = np.flatnonzero((mu0 >= lo - 1e-12) & (mu0 <= hi + 1e-12))
    if lo <= 1e-12:
        # cos Z = 0 is not a ground line: extrapolate flat from the lowest line
        x = np.concatenate([[0.0], mu0[sel]])
        y = np.concatenate([np.asarray(values)[sel[:1]], np.asarray(values)[sel]])
    else:
        x, y = mu0[sel], np.asarray(values)[sel]
    return np.trapezoid(y, x, axis=0) / (x[-1] - x[0])


def role_names(ks):
    """Species names by role (for reports)."""
    return {r: [str(s) for s, rr in zip(ks.species, ks.role) if rr == r]
            for r in (KS.ROLE_HADRON, KS.ROLE_LOSS, KS.ROLE_SINK)}
