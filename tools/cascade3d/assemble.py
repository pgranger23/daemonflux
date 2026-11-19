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


# ------------------------------------------------------------ G1.3 cells
#: Monte Carlo energy-band edges [GeV]: 0.1 x 10^(j/5), j = 0..10 (two MCEq bins each)
G13_BAND_EDGES = tuple(0.1 * 10.0 ** (np.arange(11) / 5.0))
#: Monte Carlo cos Z bin edges at the ground (down-going)
G13_COSZ_EDGES = tuple(np.round(np.linspace(0.0, 1.0, 11), 10))
G13_SPECIES = ("nue", "antinue", "numu", "antinumu")


def ground_sink_table(solver, out):
    """Sink intensities at the ground for every down-going direction available.

    Returns ``(mu0, I)``: ``mu0`` ascending (the ground lines plus 0, the horizontal
    node of the line tangent at h = 0, if present) and ``I`` (nE, nSink, n_mu0).
    """
    r = solver.rays
    nodes = list(r.ground_node)
    mu0 = list(r.ground_mu0)
    hz = np.flatnonzero((r.node_level == 0) & (r.node_branch == 0))
    if hz.size:
        nodes.append(int(hz[0]))
        mu0.append(0.0)
    nodes, mu0 = np.array(nodes), np.array(mu0)
    o = np.argsort(mu0)
    nu = np.asarray(sink_intensity(solver, out["emis"]))[:, :, nodes[o]]
    return mu0[o], nu


def band_mean_bins(e_bins, values, e_lo, e_hi):
    """Mean of dPhi/dE over [e_lo, e_hi] in the solver's bin convention.

    The number in bin i is ``phi_i w_i`` (the yields count children in bins), so
    the band mean is ``sum phi_i w_i / sum w_i`` over the bins inside the band; the
    band edges must be bin edges (checked).
    """
    eb = np.asarray(e_bins)
    i0 = int(np.argmin(np.abs(eb - e_lo)))
    i1 = int(np.argmin(np.abs(eb - e_hi)))
    if abs(eb[i0] / e_lo - 1) > 1e-6 or abs(eb[i1] / e_hi - 1) > 1e-6:
        raise ValueError("band edges must be energy-bin edges")
    w = np.diff(eb)[i0:i1]
    v = np.asarray(values)[..., i0:i1]
    return np.sum(v * w, axis=-1) / np.sum(w)


def cosz_mean(mu0, values, lo, hi):
    """Mean over cos Z in [lo, hi] of an intensity tabulated at ascending ``mu0``.

    Quadrature: the shape-preserving piecewise-cubic (PCHIP) interpolant through
    all tabulated directions, integrated exactly over the bin; ``values``
    (..., n_mu0).  Third order for smooth intensities; the bin edges should be
    tabulated directions (they are: ground lines and the horizontal node).
    """
    from scipy.interpolate import PchipInterpolator

    f = PchipInterpolator(np.asarray(mu0), np.asarray(values), axis=-1,
                          extrapolate=False)
    return f.integrate(lo, hi) / (hi - lo)


def cell_averages(solver, out, species=G13_SPECIES, band_edges=G13_BAND_EDGES,
                  cosz_edges=G13_COSZ_EDGES, energy="bins"):
    """Solver flux averaged over the Monte Carlo cells of gate G1.3.

    Parameters
    ----------
    energy : {"bins", "loglog"}
        Band mean in the bin convention (``band_mean_bins``, default) or by
        log-log interpolation between bin centres (``band_average``).

    Returns
    -------
    dict
        ``species`` -> array (n_band, n_cosz) of mean dN/dE [1/(GeV cm2 s sr)]
        over the band (neutrino energy) and mean specific intensity over the cos Z
        bin's solid angle (2 pi d cos Z, azimuth summed then divided by 2 pi
        Delta cos Z: i.e. the mean intensity per sr); rows in ascending band
        order (``band_edges``), columns in ascending cos Z (``cosz_edges``).  Also
        ``band_edges``, ``cosz_edges``.
    """
    ks = solver.ks
    names = list(ks.species)
    mu0, nu = ground_sink_table(solver, out)
    res = {"band_edges": np.asarray(band_edges),
           "cosz_edges": np.asarray(cosz_edges), "mu0": mu0}
    for sp in species:
        I = nu[:, solver.Sk.index(names.index(sp)), :]       # (nE, n_mu0)
        res[sp] = cell_table(mu0, I, ks.e_bins, ks.e_grid, band_edges, cosz_edges,
                             energy)
    return res


def cell_table(mu0, I, e_bins, e_grid, band_edges=G13_BAND_EDGES,
               cosz_edges=G13_COSZ_EDGES, energy="bins"):
    """Cell means (n_band, n_cosz) of an intensity table ``I`` (nE, n_mu0).

    See :func:`cell_averages`; this is its pure part (tested on analytic tables).
    """
    ang = np.stack([cosz_mean(mu0, I, lo, hi)                # (n_cosz, nE)
                    for lo, hi in zip(cosz_edges[:-1], cosz_edges[1:])])
    rows = []
    for lo, hi in zip(band_edges[:-1], band_edges[1:]):
        if energy == "bins":
            rows.append(band_mean_bins(e_bins, ang, lo, hi))
        else:
            rows.append(band_average(e_grid, ang, lo, hi))
    return np.array(rows)
