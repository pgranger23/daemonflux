"""From cutoff skies to the cells of G2.3 by azimuthal orders (``cascade3d`` C2).

``DESIGN_C2.md`` sections 2.2, 3.2 and 3.3.  The flux with the cutoff at a set of
locations (one location, or the area average over a cap) is

    Phi(mu0, phi) = Phi_C1(mu0)
                    + sum_m [dI^c_m(mu0) cos(m phi) + dI^s_m(mu0) sin(m phi)]

with ``dI^{c,s}_m`` the guarded tangent (``solver3d.TangentSolver``) of order m for the
top tangent ``-D^{c,s}_m`` on every chain, ``D`` the deficit of the boundary averaged
over the locations with their weights (the tangent map is linear and the same at every
location: the cap average is one boundary).  Per azimuth bin the sum is taken with the
analytic bin means of cos(m phi), sin(m phi); the cell means are then
``assemble.cell_table`` (PCHIP in cos Z, bin-convention band means) of each bin's
intensities, unchanged from C1.  With D = 0 the cells are ``cell_table`` of Phi_C1 bit
for bit.

Pieces: :func:`location_deficit`, :func:`cap_deficit` (chains x sectors),
:func:`order_tangents`, :func:`run_tangents` (one process, batches),
:func:`azimuth_table` (with the partial sum to an order M), :func:`azimuth_cells`,
:func:`phi0_cells`.

Azimuth: arrival azimuth, from north through east, in the local frame of the point
(``tools/mc3d/scoring.py``); bins of 30 deg from 0 deg (``AZ_EDGES``).
"""

import numpy as np

import assemble as A
import boundary as B

AZ_EDGES = tuple(np.linspace(0.0, 360.0, 13))
CAP_THETA_DEG = (2.5, 5.0, 7.5, 10.0)


def b0_rows(ks):
    """Indices of p and n in the kernel set and their no-cutoff boundary (2, nE)."""
    names = list(ks.species)
    ip, iN = names.index("p+"), names.index("n0")
    return (ip, iN), ks.phi_top[[ip, iN]]


def location_deficit(table, gw, ks, cw):
    """D on (chain, sector, bin, x) for one location's sky table."""
    _idx, b0 = b0_rows(ks)
    return B.deficit_chains(B.deficit_lines(table, gw, b0), cw, b0)


def cap_deficit(tables, weights, gw, ks, cw):
    """sum_P W_P D_P over locations with non-zero weight (each in its own frame)."""
    D = None
    for p, w in enumerate(weights):
        if w == 0.0:
            continue
        Dp = w * location_deficit(tables[p], gw, ks, cw)
        D = Dp if D is None else D + Dp
    return D


def grid_deficit(sky_dir, grid, weights, gw, ks, cw, h_ref_km=0.0, tables=None):
    """Weighted sum of the location deficits of a ``boundary.LocationGrid`` (cap
    average: ``weights = grid.cap_weights(radius)``; one location: a unit vector).

    Only the tables with non-zero weight are read (``boundary.table_path`` names in
    ``sky_dir``, or ``tables``: index -> SkyTable).  The result has the shape of one
    location's deficit whatever the number of locations, so the solve that follows
    costs the same for 25 or 81 tables.  Returns ``(D, used)`` with ``used`` the
    (index, weight, path) of the tables read.
    """
    used, tabs = [], {}
    for p, w in enumerate(weights):
        if w == 0.0:
            continue
        if tables is not None:
            tabs[p] = tables[p]
            path = getattr(tables[p], "path", None)
        else:
            path = B.table_path(sky_dir, (grid.x[p], grid.y[p]), h_ref_km)
            tabs[p] = B.SkyTable.load(path)
        grid.check_table(tabs[p], p)
        used.append((p, float(w), path))
    return cap_deficit(tabs, weights, gw, ks, cw), used


def order_tangents(Dc, M, ks):
    """Top tangents of the orders 0..M: list of ``(m, part, dtop)`` with ``part`` "c"
    or "s" and ``dtop`` (nS, nE, Nl) = -D^part_m (sine parts from m = 1; at the
    Nyquist order only the non-zero part)."""
    (ip, iN), _b0 = b0_rows(ks)
    ns = Dc.shape[1]
    c, s = B.fourier_sectors(Dc, M, axis=1)              # (Nl, M+1, nE, 2)
    out = []
    for m in range(M + 1):
        for part, arr in (("c", c), ("s", s)):
            if part == "s" and m == 0:
                continue
            if 2 * m == ns and not np.any(arr[:, m]):
                continue
            dt = np.zeros(ks.phi_top.shape + (Dc.shape[0],))
            dt[ip] = -arr[:, m, :, 0].T
            dt[iN] = -arr[:, m, :, 1].T
            out.append((m, part, dt))
    return out


def run_tangents(ts, tangents, batch, log=print):
    """Solve the tangents in batches with one ``TangentSolver`` (one process).

    Returns ``{"mu0", "I" (nE, nSk, n_mu0), "F_ground", "res", "up_res",
    "tangents": [(m, part)], "dI" (n, nE, nSk, n_mu0), "dF_ground" (n, nE, nT, ng),
    "d_res" (n, nE), "d_up_res" (n, passes - 1), "timing": [per batch]}``.  The
    primal (C1 state) of the last batch is returned; every batch recomputes it (the
    tangent of a step needs the primal of that step)."""
    import solver3d as S3

    nodes, mu0 = S3.ground_table_nodes(ts.rays)
    res = {"mu0": mu0, "tangents": [], "dI": [], "dF_ground": [], "d_res": [],
           "d_up_res": [], "timing": []}
    for i in range(0, len(tangents), batch):
        part = tangents[i:i + batch]
        out = ts.solve(np.stack([t[2] for t in part]), [t[0] for t in part])
        I, dI = S3.sink_tangent(ts.sv, out["emis"], out["d_emis"])
        res["I"] = np.asarray(I)[:, :, nodes]
        res["F_ground"] = np.asarray(out["F_ground"])
        res["res"] = np.asarray(out["res"])
        res["up_res"] = np.asarray(out["up_res"])
        res["dI"].append(np.asarray(dI)[:, :, :, nodes])
        res["dF_ground"].append(np.asarray(out["d_F_ground"]))
        res["d_res"].append(np.asarray(out["d_res"]))
        res["d_up_res"].append(np.asarray(out["d_up_res"]))
        res["tangents"] += [(t[0], t[1]) for t in part]
        res["timing"].append(dict(out["timing"], batch=len(part)))
        log(f"batch {i // batch}: orders {[(t[0], t[1]) for t in part]}, "
            f"{out['timing']['cpu_s'] / 60:.1f} core-min")
    for k in ("dI", "dF_ground", "d_res", "d_up_res"):
        res[k] = np.concatenate(res[k], axis=0)
    return res


def azimuth_table(I0, dI, tangents, az_edges=AZ_EDGES, M=None):
    """Ground intensity per azimuth bin: (n_az, nE, nSk, n_mu0).

    ``dI`` (n, ...) per tangent ``(m, part)``; orders above ``M`` are left out (the
    partial sum of G2.5 a)."""
    mmax = max(m for m, _p in tangents)
    Cb, Sb = B.bin_mean_basis(mmax, az_edges)             # (n_az, mmax + 1)
    out = np.broadcast_to(np.asarray(I0), (Cb.shape[0],) + np.shape(I0)).copy()
    for (m, part), d in zip(tangents, dI):
        if M is not None and m > M:
            continue
        w = Cb[:, m] if part == "c" else Sb[:, m]
        out += w[:, None, None, None] * np.asarray(d)[None]
    return out


def azimuth_cells(mu0, Iaz, ks, species=A.G13_SPECIES):
    """Cells (species, bands, rows, n_az) of an azimuth table (``assemble.cell_table``
    per bin)."""
    names = list(ks.species)
    sinks = [i for i in range(ks.nS) if _is_sink(ks, i)]
    out = np.zeros((len(species), len(A.G13_BAND_EDGES) - 1,
                    len(A.G13_COSZ_EDGES) - 1, Iaz.shape[0]))
    for j, sp in enumerate(species):
        isk = sinks.index(names.index(sp))
        for b in range(Iaz.shape[0]):
            out[j, :, :, b] = A.cell_table(mu0, Iaz[b, :, isk, :], ks.e_bins,
                                           ks.e_grid)
    return out


def _is_sink(ks, i):
    import kernelset as KS

    return ks.role[i] == KS.ROLE_SINK


def phi0_cells(mu0, I0, ks, species=A.G13_SPECIES):
    """Cells (species, bands, rows) of the transmission-1 table."""
    return azimuth_cells(mu0, np.asarray(I0)[None], ks, species)[..., 0]
