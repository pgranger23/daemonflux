"""Cutoff skies and boundary deficits of ``cascade3d`` stage C2 (WP 2.1).

The design is ``DESIGN_C2.md`` section 2 with the conditions of ``GATES.md``
("C2, dated entries", 2026-11-23).  Everything here is numpy; the cutoff verdicts
are those of the Monte Carlo (``tools/mc3d/cutoff.py``: ``gate``, then
``trace_allowed``, i.e. ``trace_gated``, with its defaults: IGRF-14 at 2020-01-01,
injection sphere at 100 km, re-entry rule at that sphere).

Pieces
------
:class:`LocationGrid`
    Locations in azimuthal-equidistant coordinates about the site (x east, y north,
    degrees of arc, as ``cutoff.RUCache``); bilinear weights; cap averages.
:func:`line_nodes` and :func:`chain_entry_states`
    The line nodes of a sky table (ground lines by their ground cos Z, tangent lines
    by their tangent altitude) and the injection state (r, u) of each (line node,
    azimuth) at a location: the line attached there, followed backwards to its first
    crossing of the injection sphere.  Lines that never reach it are flagged
    (transmission 0, the convention of ``GATES.md``).
:class:`GroupWeights`
    The share of each rigidity half-interval of the ladder, summed over the nucleus
    groups, in the nucleons of type x (p, n) of energy bin k:
    ``C_x[k, r] = sum_g int_{bin k, R_g(T) in r} J_{g,x} dT / int_{bin k} J_x dT``
    (``J_{g,x}`` from ``primaries.group_split``; GSFG:GH or H3a).  Above 60 GV the
    transmission is 1 (``above``).
:class:`SkyTable`
    The verdicts at one location: line nodes x 36 azimuth sectors x 31 rigidity
    half-intervals, one azimuth drawn in each sector and one ln R in each
    half-interval (fixed seed).  Built with pooled ``trace_gated`` calls over forked
    workers; resumable (chunk files); the finished table is written once, atomically,
    with its sha256, and never overwritten.
:func:`deficit_lines`, :func:`chain_weights`, :func:`deficit_chains`
    D = B0 (1 - Tbar) on (line node, sector, bin, x), and on the chains of a ray
    table (linear interpolation in ground zenith angle, resp. tangent altitude, after
    the bin average).
:func:`fourier_sectors`, :func:`synth_sectors`, :func:`bin_mean_basis`
    Orders m = 0..M of a sector-averaged azimuthal function (discrete Fourier sum
    divided by the sinc of the sector average) and back.
:func:`sky_transmission`
    The G2.0 interpolation: the transmission the tables give an arbitrary injection
    state.

Conventions: positions in cm, rigidities in GV, angles in degrees unless stated;
azimuth is the arrival azimuth (the compass direction the particle comes from, from
north through east) in the local frame of the point considered, as
``tools/mc3d/scoring.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(os.path.dirname(_HERE), "mc3d"),
           os.path.join(os.path.dirname(_HERE), "mceq3d")):
    if _p not in sys.path:
        sys.path.append(_p)

import constants as mcc  # noqa: E402  (tools/mc3d)
import cutoff  # noqa: E402
import primaries as mcprim  # noqa: E402

R_EARTH_CM = mcc.R_EARTH_CM
R_INJ_CM = cutoff.R_INJ_CM
SITE = cutoff.SITES["kamioka"]
M_NUC = mcc.M_P                        # the nucleon mass of ``cutoff.rigidity_gv``
R_SKIP_GV = cutoff.R_SKIP_GV           # transmission 1 above (GATES.md, conventions)

#: ground line nodes (cos Z at the ground): G1.1 directions, G1.3 bin edges, and the
#: ground zenith every 5 degrees (DESIGN_C2.md 2.2)
GROUND_COSZ = tuple(np.unique(np.round(np.concatenate([
    [1.0, 0.75, 0.5, 0.25, 0.15, 0.05], np.linspace(0.1, 1.0, 10),
    np.cos(np.radians(5.0 * np.arange(18)))]), 12))[::-1])
#: tangent line nodes [km]
TANGENT_HT_KM = (0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0,
                 70.0, 80.0, 90.0, 99.0)
#: azimuth sectors
N_SECTORS = 36
#: rigidity ladder [GV]: 20 half-intervals per decade from 1.7 GV, the last one
#: closed at 60 GV (above: transmission 1)
LADDER_PER_DECADE = 20
LADDER_LO_GV = 1.7


def ladder_edges(lo=LADDER_LO_GV, hi=R_SKIP_GV, per_decade=LADDER_PER_DECADE):
    """Edges of the rigidity half-intervals [GV] (31 intervals by default)."""
    n = int(np.floor(per_decade * np.log10(hi / lo) + 1e-9))
    e = lo * 10.0 ** (np.arange(n + 1) / per_decade)
    if e[-1] < hi * (1 - 1e-9):
        e = np.append(e, hi)
    else:
        e[-1] = hi
    return e


# ---------------------------------------------------------------------------
# frames and the location grid
# ---------------------------------------------------------------------------
def local_frame(rh):
    """(up, north, east) unit vectors at unit positions ``rh`` (n, 3), geocentric
    (``cutoff.RUCache._frames``; at the site the frame of
    ``geomag_backtrace._local_frame``)."""
    rh = np.atleast_2d(np.asarray(rh, float))
    lat = np.arcsin(np.clip(rh[:, 2], -1, 1))
    lon = np.arctan2(rh[:, 1], rh[:, 0])
    north = np.stack([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon),
                      np.cos(lat)], 1)
    east = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], 1)
    return rh, north, east


def site_frame(site=SITE):
    """(up, north, east) at the site, each (3,)."""
    la, lo = np.radians(site[0]), np.radians(site[1])
    up = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    north = np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo),
                      np.cos(la)])
    east = np.array([-np.sin(lo), np.cos(lo), 0.0])
    return up, north, east


def pos_from_xy(x, y, site=SITE):
    """Unit positions (n, 3) of azimuthal-equidistant coordinates (x east, y north)
    [deg of arc] about the site (``cutoff.RUCache._pos_from_xy``)."""
    up, north, east = site_frame(site)
    x, y = np.broadcast_arrays(np.atleast_1d(np.asarray(x, float)),
                               np.atleast_1d(np.asarray(y, float)))
    al = np.radians(np.hypot(x, y))
    b = np.arctan2(x, y)
    return (np.cos(al)[:, None] * up
            + np.sin(al)[:, None] * (np.cos(b)[:, None] * north
                                     + np.sin(b)[:, None] * east))


def xy_of(r, site=SITE):
    """Azimuthal-equidistant (x, y) [deg] of positions ``r`` (n, 3), any radius."""
    up, north, east = site_frame(site)
    r = np.atleast_2d(np.asarray(r, float))
    rh = r / np.linalg.norm(r, axis=1)[:, None]
    al = np.degrees(np.arccos(np.clip(rh @ up, -1, 1)))
    b = np.arctan2(rh @ east, rh @ north)
    return al * np.sin(b), al * np.cos(b)


class LocationGrid:
    """Nodes at multiples of the spacing covering [-hw_x, hw_x] x [-hw_y, hw_y].

    Parameters
    ----------
    spacing : float or (float, float)
        Degrees of arc, one value for both axes or (x east, y north): 5 (25 nodes),
        2.5 (81 nodes), (5, 2.5) grid A (45 nodes), (5, 1.25) grid B (85 nodes).
    half_width : float or (float, float)
        Degrees of arc (default 10 for both axes); a multiple of the spacing.
    site : (lat, lon)
        Centre; node (0, 0) is the site.

    With equal spacings the nodes, weights and cap weights are those of the class
    before it took one spacing per axis (``test_boundary``: bit for bit).
    Node index: ``i * ny + j`` with i along x and j along y.
    """

    def __init__(self, spacing=5.0, half_width=10.0, site=SITE):
        sx, sy = (float(v) for v in np.broadcast_to(np.asarray(spacing, float), 2))
        hx, hy = (float(v) for v in np.broadcast_to(np.asarray(half_width, float),
                                                    2))
        nx = int(round(hx / sx))
        ny = int(round(hy / sy))
        if abs(nx * sx - hx) > 1e-9 or abs(ny * sy - hy) > 1e-9:
            raise ValueError("half_width must be a multiple of spacing")
        self.spacing_xy, self.half_width_xy = (sx, sy), (hx, hy)
        self.site = tuple(site)
        self.isotropic = (sx == sy) and (hx == hy)
        # scalar attributes of the isotropic grid (kept for its callers)
        self.spacing = sx if self.isotropic else (sx, sy)
        self.half_width = hx if self.isotropic else (hx, hy)
        self.axis_x = sx * np.arange(-nx, nx + 1)
        self.axis_y = sy * np.arange(-ny, ny + 1)
        if self.isotropic:
            self.axis = self.axis_x
        X, Y = np.meshgrid(self.axis_x, self.axis_y, indexing="ij")
        self.x, self.y = X.ravel(), Y.ravel()
        self.n = self.x.size

    def label(self):
        """"5" or "5x1.25" (degrees)."""
        sx, sy = self.spacing_xy
        return f"{sx:g}" if sx == sy else f"{sx:g}x{sy:g}"

    def node_index(self, x, y):
        """Index of the node at (x, y) (exact multiples)."""
        i = int(round((x + self.half_width_xy[0]) / self.spacing_xy[0]))
        j = int(round((y + self.half_width_xy[1]) / self.spacing_xy[1]))
        return i * self.axis_y.size + j

    def check_table(self, table, p, tol=1e-9):
        """Raise unless ``table`` stores the coordinates of node ``p`` (the file
        name rounds them to one decimal)."""
        xy = np.asarray(table.xy, float)
        if abs(xy[0] - self.x[p]) > tol or abs(xy[1] - self.y[p]) > tol:
            raise ValueError(f"table at {tuple(xy)} is not node {p} "
                             f"({self.x[p]}, {self.y[p]})")

    def positions(self):
        """Unit positions (n, 3) of the nodes."""
        return pos_from_xy(self.x, self.y, self.site)

    def weights_xy(self, x, y):
        """Bilinear weights of points (x, y): ``(idx (n, 4), w (n, 4), inside)``."""
        x = np.atleast_1d(np.asarray(x, float))
        y = np.atleast_1d(np.asarray(y, float))
        nx, ny = self.axis_x.size, self.axis_y.size
        fx = (x + self.half_width_xy[0]) / self.spacing_xy[0]
        fy = (y + self.half_width_xy[1]) / self.spacing_xy[1]
        inside = (fx >= -1e-12) & (fx <= nx - 1 + 1e-12) & (fy >= -1e-12) & (
            fy <= ny - 1 + 1e-12)
        i0 = np.clip(np.floor(fx).astype(int), 0, nx - 2)
        j0 = np.clip(np.floor(fy).astype(int), 0, ny - 2)
        tx = np.clip(fx - i0, 0.0, 1.0)
        ty = np.clip(fy - j0, 0.0, 1.0)
        idx = np.stack([i0 * ny + j0, (i0 + 1) * ny + j0, i0 * ny + j0 + 1,
                        (i0 + 1) * ny + j0 + 1], 1)
        w = np.stack([(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty], 1)
        return idx, w, inside

    def weights(self, points):
        """Bilinear weights of positions ``points`` (n, 3)."""
        x, y = xy_of(points, self.site)
        return self.weights_xy(x, y)

    def cap_weights(self, radius_deg, n_quad=100, shift=(0.0, 0.0)):
        """W_P = (1/A) int_cap w_P(Q + shift) dA over the cap of ``radius_deg``
        about the site (area quadrature on the sphere: Gauss-Legendre in the
        angular distance with weight sin(alpha), uniform in the bearing; about
        ``n_quad``^2 points).  ``shift`` (dx, dy) [deg] displaces each point in
        the azimuthal-equidistant coordinates before the bilinear weights."""
        xg, wg = np.polynomial.legendre.leggauss(n_quad)
        a = np.radians(radius_deg) * 0.5 * (xg + 1.0)
        wa = np.radians(radius_deg) * 0.5 * wg * np.sin(a)
        nb = 2 * n_quad
        b = 2 * np.pi * (np.arange(nb) + 0.5) / nb
        A, B = np.meshgrid(a, b, indexing="ij")
        WA = np.repeat(wa[:, None], nb, axis=1) * (2 * np.pi / nb)
        x = np.degrees(A) * np.sin(B) + shift[0]
        y = np.degrees(A) * np.cos(B) + shift[1]
        idx, w, inside = self.weights_xy(x.ravel(), y.ravel())
        if not np.all(inside):
            raise ValueError("cap (with its shift) leaves the location grid")
        W = np.zeros(self.n)
        np.add.at(W, idx.ravel(), (w * WA.ravel()[:, None]).ravel())
        return W / W.sum()


# ---------------------------------------------------------------------------
# line nodes and injection states
# ---------------------------------------------------------------------------
def line_nodes(ground_cosz=GROUND_COSZ, tangent_ht_km=TANGENT_HT_KM):
    """Line nodes of a sky table: ``kind`` (0 ground, 1 tangent), ``cosz`` (ground
    cos Z, 0 for tangent lines), ``ht_km`` (tangent altitude, 0 for ground lines)."""
    gc = np.asarray(ground_cosz, float)
    ht = np.asarray(tangent_ht_km, float)
    kind = np.concatenate([np.zeros(gc.size, int), np.ones(ht.size, int)])
    cosz = np.concatenate([gc, np.zeros(ht.size)])
    htk = np.concatenate([np.zeros(gc.size), ht])
    return {"kind": kind, "cosz": cosz, "ht_km": htk}


def arrival_velocity(up, north, east, cosz, az_deg):
    """Inward unit velocity of a particle arriving with local zenith cos ``cosz``
    from compass azimuth ``az_deg`` (``geomag_backtrace.arrival_direction``)."""
    a = np.radians(np.asarray(az_deg, float))
    c = np.asarray(cosz, float)
    s = np.sqrt(np.clip(1.0 - c * c, 0.0, None))
    return -(c[..., None] * up + s[..., None] * (np.cos(a)[..., None] * north
                                                  + np.sin(a)[..., None] * east))


def entry_state(x0, u, r_inj=R_INJ_CM):
    """Follow the line through ``x0`` (n, 3) [cm] with velocity ``u`` backwards to its
    first crossing of the sphere ``r_inj``.  Returns ``(r_inj (n, 3), never (n,))``;
    ``never``: the line does not reach that sphere (rows set to NaN)."""
    x0 = np.atleast_2d(np.asarray(x0, float))
    u = np.atleast_2d(np.asarray(u, float))
    pu = np.einsum("ij,ij->i", x0, u)
    c = np.einsum("ij,ij->i", x0, x0) - r_inj ** 2
    disc = pu * pu - c
    # impact parameter at or above the sphere (1 cm slack): no crossing
    b2 = np.einsum("ij,ij->i", x0, x0) - pu * pu
    never = (disc < 0.0) | (b2 >= (r_inj - 1.0) ** 2)
    t = pu + np.sqrt(np.maximum(disc, 0.0))
    r = x0 - t[:, None] * u
    r[never] = np.nan
    return r, never


def chain_entry_states(rh, nodes, az_deg, h_ref_km=0.0, r_e=R_EARTH_CM,
                       r_inj=R_INJ_CM):
    """Injection states of the lines attached at the location ``rh`` (unit vector).

    A ground line node (ground cos Z ``c``) is attached at the reference altitude
    ``h_ref_km`` above the location: it passes there with the local zenith of the
    ground line of cos Z ``c`` at that altitude (``sin Z_h = sin Z (R / (R + h))``)
    and arrival azimuth ``az``; ``h_ref_km = 0``: the line passes through the
    location.  A tangent line node passes horizontally through the point at its
    tangent altitude above the location.

    Parameters
    ----------
    nodes : dict
        :func:`line_nodes`.
    az_deg : array (n_nodes, ...)
        Arrival azimuths (any trailing shape).

    Returns
    -------
    r (n_nodes, ..., 3), u (n_nodes, ..., 3), never (n_nodes, ...)
    """
    rh = np.asarray(rh, float).reshape(3)
    up, north, east = (v[0] for v in local_frame(rh[None]))
    az = np.asarray(az_deg, float)
    nn = nodes["kind"].size
    if az.shape[0] != nn:
        raise ValueError("az_deg must have one row per line node")
    shp = az.shape
    cz = np.empty(shp)
    alt = np.empty(shp)
    for i in range(nn):
        if nodes["kind"][i] == 0:
            c = nodes["cosz"][i]
            s = np.sqrt(max(1.0 - c * c, 0.0)) * r_e / (r_e + h_ref_km * 1e5)
            cz[i] = np.sqrt(max(1.0 - s * s, 0.0))
            alt[i] = h_ref_km * 1e5
        else:
            cz[i] = 0.0
            alt[i] = nodes["ht_km"][i] * 1e5
    u = arrival_velocity(up, north, east, cz, az)
    x0 = (r_e + alt)[..., None] * rh
    r, never = entry_state(x0.reshape(-1, 3), u.reshape(-1, 3), r_inj)
    return r.reshape(shp + (3,)), u, never.reshape(shp)


# ---------------------------------------------------------------------------
# primaries: group weights on the rigidity ladder
# ---------------------------------------------------------------------------
def t_of_rigidity(R, a_over_z):
    """Kinetic energy per nucleon [GeV] of a nucleus of rigidity ``R`` [GV]."""
    p = np.asarray(R, float) / a_over_z
    return np.sqrt(p * p + M_NUC * M_NUC) - M_NUC


class GroupWeights:
    """Rigidity-ladder weights of the nucleon boundary.

    Parameters
    ----------
    e_bins : array (nE + 1,)
        Kinetic energy per nucleon bin edges of the kernel set [GeV].
    model : str
        ``primaries.primary_model`` name: ``"GSFG:GH"`` (the Monte Carlo of G2.3) or
        ``"H3A"`` (G2.2).
    e_range : (lo, hi), optional
        Primary range in kinetic energy per nucleon (default: the Monte Carlo's,
        ``sky.STRATA`` with ``--nstrata 7``: 1 to 10^3.5 GeV).
    edges : array, optional
        Ladder edges [GV] (:func:`ladder_edges`).

    Attributes
    ----------
    C : array (2, nE, n_rungs)
        ``C[x, k, r]``: share of the nucleons of type x (0 p, 1 n) of bin k whose
        nucleus has its rigidity in half-interval r, summed over the groups.
        Rigidities below the ladder are counted in rung 0.
    above : array (2, nE)
        Share above the last edge (60 GV: transmission 1).
    flux : array (2, nE)
        ``int_bin J_x dT / width`` from the same quadrature (compare ``phi_top``).
    """

    def __init__(self, e_bins, model="GSFG:GH", e_range=None, edges=None,
                 convention=None, n_gl=8, n_panel=2):
        self.e_bins = np.asarray(e_bins, float)
        self.model = model
        self.edges = ladder_edges() if edges is None else np.asarray(edges, float)
        if e_range is None:
            import sky
            e_range = (float(sky.STRATA[0]), float(sky.STRATA[7]))
        self.e_range = tuple(e_range)
        mdl = mcprim.primary_model(model)
        conv = mcprim.resolve_convention(convention or mcprim.ENERGY_CONVENTION,
                                         name=model)
        self.convention = conv
        nE = self.e_bins.size - 1
        nr = self.edges.size - 1
        C = np.zeros((2, nE, nr))
        above = np.zeros((2, nE))
        tot = np.zeros((2, nE))
        xg, wg = np.polynomial.legendre.leggauss(n_gl)
        groups = [(gid, a / z) for _n, a, z, gid in mcprim.SPECIES]
        for k in range(nE):
            lo = max(self.e_bins[k], self.e_range[0])
            hi = min(self.e_bins[k + 1], self.e_range[1])
            if hi <= lo * (1 + 1e-12):
                continue
            for gid, aoz in groups:
                tcut = t_of_rigidity(self.edges, aoz)
                cuts = np.unique(np.concatenate([[lo, hi],
                                                 tcut[(tcut > lo) & (tcut < hi)]]))
                for a, b in zip(cuts[:-1], cuts[1:]):
                    ed = np.linspace(np.log(a), np.log(b), n_panel + 1)
                    h = np.diff(ed)
                    u = ed[:-1, None] + 0.5 * h[:, None] * (xg[None] + 1.0)
                    t = np.exp(u).ravel()
                    w = (0.5 * h[:, None] * wg[None] * np.exp(u)).ravel()
                    jp, jn = mcprim.group_split(mdl, gid, t, conv)
                    ip, inn = float(np.sum(w * jp)), float(np.sum(w * jn))
                    rmid = cutoff.rigidity_gv(np.sqrt(a * b), aoz, 1.0, M_NUC)
                    r = int(np.searchsorted(self.edges, rmid) - 1)
                    tot[:, k] += (ip, inn)
                    if r >= nr:
                        above[:, k] += (ip, inn)
                    else:
                        C[:, k, max(r, 0)] += (ip, inn)
        width = np.diff(self.e_bins)
        ok = tot > 0
        self.flux = tot / width[None]
        self.C = np.where(ok[..., None], C / np.where(ok, tot, 1.0)[..., None], 0.0)
        self.above = np.where(ok, above / np.where(ok, tot, 1.0), 0.0)
        self.has_flux = ok


# ---------------------------------------------------------------------------
# sky table of one location
# ---------------------------------------------------------------------------
def _pool_worker(args):
    j, idx, r, u, R, fn_name = args
    t0 = time.process_time()
    if fn_name == "trace_gated":
        a, nt = cutoff.trace_gated(r, u, R, 1.0)
    else:                                   # test hook: a module-level function
        mod, _, name = fn_name.rpartition(".")
        __import__(mod)
        a, nt = getattr(sys.modules[mod], name)(r, u, R)
    return j, idx, np.asarray(a, bool), int(nt), time.process_time() - t0


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


class SkyTable:
    """Cutoff verdicts at one location (see the module docstring).

    Attributes (arrays of the npz): ``nodes_kind``, ``nodes_cosz``, ``nodes_ht_km``
    (line nodes); ``edges`` (ladder); ``az`` (n_nodes, 36, n_rungs) drawn azimuths;
    ``R`` (same) drawn rigidities; ``allowed`` (same, bool); ``never`` (n_nodes,)
    lines that never reach the injection sphere; ``rh`` the location; ``xy`` its
    grid coordinates; ``h_ref_km``; ``seed``.
    """

    def __init__(self, d, path=None):
        self.d = {k: np.asarray(v) for k, v in d.items()}
        self.path = path

    def __getattr__(self, k):
        d = self.__dict__.get("d")
        if d is not None and k in d:
            return d[k]
        raise AttributeError(k)

    @property
    def nodes(self):
        return {"kind": self.d["nodes_kind"], "cosz": self.d["nodes_cosz"],
                "ht_km": self.d["nodes_ht_km"]}

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            d = {k: f[k] for k in f.files}
        return cls(d, path)

    @staticmethod
    def states(xy=(0.0, 0.0), nodes=None, edges=None, n_sectors=N_SECTORS,
               h_ref_km=0.0, seed=1, site=SITE):
        """Draws and injection states of a table (no tracing)."""
        nodes = line_nodes() if nodes is None else nodes
        edges = ladder_edges() if edges is None else np.asarray(edges, float)
        nn, nr = nodes["kind"].size, edges.size - 1
        rng = np.random.default_rng(seed)
        az = (np.arange(n_sectors)[None, :, None]
              + rng.random((nn, n_sectors, nr))) * (360.0 / n_sectors)
        le = np.log(edges)
        R = np.exp(le[:-1][None, None, :] + rng.random((nn, n_sectors, nr))
                   * np.diff(le)[None, None, :])
        rh = pos_from_xy(xy[0], xy[1], site)[0]
        r, u, never = chain_entry_states(rh, nodes, az, h_ref_km)
        return {"nodes_kind": nodes["kind"], "nodes_cosz": nodes["cosz"],
                "nodes_ht_km": nodes["ht_km"], "edges": edges, "az": az, "R": R,
                "r_inj": r, "u": u, "never_state": never,
                "never": never.reshape(nn, -1).all(axis=1), "rh": rh,
                "xy": np.asarray(xy, float), "h_ref_km": np.asarray(h_ref_km),
                "seed": np.asarray(seed), "n_sectors": np.asarray(n_sectors),
                "site": np.asarray(site, float)}

    @classmethod
    def build(cls, path, xy=(0.0, 0.0), nodes=None, edges=None,
              n_sectors=N_SECTORS, h_ref_km=0.0, seed=1, n_jobs=1,
              min_chunk=10000, verdict="trace_gated", site=SITE, log=print):
        """Build (or resume) the table and write it to ``path``.

        The states are dealt round-robin into chunks of at least ``min_chunk``
        states (the bouncing tail of a call sets its cost); each chunk is one
        ``trace_gated`` call in a forked worker and is saved to
        ``path + ".work/chunk_NNN.npz"`` when done, so an interrupted build resumes.
        An existing finished table is never overwritten (``FileExistsError``).

        ``verdict``: ``"trace_gated"`` or the dotted name of a module-level function
        ``f(r, u, R) -> (allowed, n_traced)`` (tests).
        """
        if os.path.exists(path):
            raise FileExistsError(f"{path} exists; a finished table is never "
                                  "overwritten")
        d = cls.states(xy, nodes, edges, n_sectors, h_ref_km, seed, site)
        nn, ns, nr = d["az"].shape
        flat_r = d["r_inj"].reshape(-1, 3)
        flat_u = d["u"].reshape(-1, 3)
        flat_R = d["R"].ravel()
        live = np.flatnonzero(~d["never_state"].ravel())
        key = hashlib.sha256(np.ascontiguousarray(flat_r[live]).tobytes()
                             + np.ascontiguousarray(flat_R[live]).tobytes()
                             + verdict.encode()).hexdigest()[:16]
        work = path + ".work"
        os.makedirs(work, exist_ok=True)
        n_chunks = max(1, live.size // int(min_chunk))
        chunks = [live[j::n_chunks] for j in range(n_chunks)]
        allowed = np.zeros(flat_R.size, bool)
        n_traced, cpu = 0, 0.0
        todo = []
        for j, idx in enumerate(chunks):
            cf = os.path.join(work, f"chunk_{j:03d}.npz")
            if os.path.exists(cf):
                with np.load(cf) as f:
                    if str(f["key"]) == key and np.array_equal(f["idx"], idx):
                        allowed[idx] = f["allowed"]
                        n_traced += int(f["n_traced"])
                        cpu += float(f["cpu_s"])
                        log(f"chunk {j}: resumed")
                        continue
                raise RuntimeError(f"{cf} belongs to another table")
            todo.append((j, idx))
        t0 = time.time()
        args = [(j, idx, flat_r[idx], flat_u[idx], flat_R[idx], verdict)
                for j, idx in todo]

        def save(j, idx, a, nt, c):
            cf = os.path.join(work, f"chunk_{j:03d}.npz")
            tmp = cf + ".tmp.npz"
            np.savez(tmp, key=np.array(key), idx=idx, allowed=a,
                     n_traced=np.array(nt), cpu_s=np.array(c))
            os.replace(tmp, cf)

        if todo:
            if n_jobs <= 1 or len(todo) == 1:
                for a_ in args:
                    j, idx, a, nt, c = _pool_worker(a_)
                    allowed[idx] = a
                    n_traced += nt
                    cpu += c
                    save(j, idx, a, nt, c)
                    log(f"chunk {j}: {idx.size} states, {nt} traced, {c:.0f} s")
            else:
                import multiprocessing as mp
                with mp.get_context("fork").Pool(min(int(n_jobs), len(todo))) as pool:
                    for j, idx, a, nt, c in pool.imap_unordered(_pool_worker, args):
                        allowed[idx] = a
                        n_traced += nt
                        cpu += c
                        save(j, idx, a, nt, c)
                        log(f"chunk {j}: {idx.size} states, {nt} traced, {c:.0f} s")
        d["allowed"] = allowed.reshape(nn, ns, nr) & ~d["never_state"]
        d["n_traced"] = np.asarray(n_traced)
        d["n_states"] = np.asarray(live.size)
        d["cpu_s"] = np.asarray(cpu)
        d["wall_s"] = np.asarray(time.time() - t0)
        d["verdict"] = np.asarray(verdict)
        d["key"] = np.asarray(key)
        d["created"] = np.asarray(time.strftime("%Y-%m-%d %H:%M:%S"))
        tmp = path + ".tmp.npz"
        np.savez(tmp, **d)
        if os.path.exists(path):
            raise FileExistsError(path)
        os.replace(tmp, path)
        sha = _sha256(path)
        with open(path + ".sha256", "w") as fh:
            fh.write(f"{sha}  {os.path.basename(path)}\n")
        return cls.load(path)

    # ----------------------------------------------------------- statistics
    def summary(self):
        a = self.d["allowed"]
        live = ~self.d["never_state"]
        return {"line_nodes": int(a.shape[0]), "sectors": int(a.shape[1]),
                "rungs": int(a.shape[2]), "states": int(live.sum()),
                "never_lines": int(self.d["never"].sum()),
                "traced": int(self.d.get("n_traced", -1)),
                "allowed_fraction": float(a[live].mean()),
                "cpu_s": float(self.d.get("cpu_s", np.nan)),
                "wall_s": float(self.d.get("wall_s", np.nan))}


# ---------------------------------------------------------------------------
# deficits
# ---------------------------------------------------------------------------
def deficit_lines(table, gw, b0):
    """D_x(line node, sector, bin) = B0_x(k) sum_r C_x[k, r] (1 - allowed).

    Parameters
    ----------
    table : SkyTable
    gw : GroupWeights
    b0 : array (2, nE)
        The no-cutoff boundary of p and n (``phi_top`` rows of the kernel set).

    Returns
    -------
    array (n_nodes, n_sectors, nE, 2); lines that never reach the injection sphere
    carry the whole boundary (transmission 0).
    """
    a = np.asarray(table.allowed, float)                     # (nn, ns, nr)
    forb = 1.0 - a
    # forbidden share; the shares sum to 1 to rounding, so it is clipped to [0, 1]
    # (D never exceeds B0: no negative boundary)
    f = np.clip(np.einsum("ijr,xkr->ijkx", forb, gw.C), 0.0, 1.0)
    D = f * np.asarray(b0, float).T[None, None]
    nev = np.asarray(table.never, bool)
    D[nev] = np.asarray(b0, float).T[None]
    return D


def chain_weights(rays, nodes, r_inj=R_INJ_CM):
    """Line-node interpolation weights of every chain of a ray table.

    Ground chains: linear in the ground zenith angle between the ground line nodes,
    with the tangent node h_t = 0 as zenith 90 deg.  Tangent chains: linear in the
    tangent altitude between the tangent nodes, flat outside them.  Chains whose
    line never reaches the injection sphere (h_t >= 100 km) are flagged.

    Returns ``(i0, i1, w0, w1, never)``, each (Nl,).
    """
    kind = np.asarray(nodes["kind"])
    gi = np.flatnonzero(kind == 0)
    ti = np.flatnonzero(kind == 1)
    zg = np.degrees(np.arccos(np.clip(np.asarray(nodes["cosz"])[gi], -1, 1)))
    ht = np.asarray(nodes["ht_km"])[ti]
    h0 = ti[np.flatnonzero(ht == 0.0)]
    if h0.size:
        zg = np.append(zg, 90.0)
        gi = np.append(gi, h0[0])
    og, ot = np.argsort(zg), np.argsort(ht)
    zg, gi, ht, ti = zg[og], gi[og], ht[ot], ti[ot]
    Nl = rays.Nl
    i0 = np.zeros(Nl, int)
    i1 = np.zeros(Nl, int)
    w1 = np.zeros(Nl)
    never = np.zeros(Nl, bool)
    R = rays.r_e
    for j in range(Nl):
        b = rays.line_b[j]
        if rays.line_kind[j] == 0:
            x, xs, ids = np.degrees(np.arcsin(min(b / R, 1.0))), zg, gi
        else:
            x, xs, ids = (b - R) / 1e5, ht, ti
            if b >= r_inj - 1.0:
                never[j] = True
        p = int(np.clip(np.searchsorted(xs, x) - 1, 0, xs.size - 2))
        t = float(np.clip((x - xs[p]) / (xs[p + 1] - xs[p]), 0.0, 1.0))
        i0[j], i1[j], w1[j] = ids[p], ids[p + 1], t
    return i0, i1, 1.0 - w1, w1, never


def deficit_chains(D_lines, cw, b0):
    """D on the chains: (Nl, n_sectors, nE, 2) from :func:`deficit_lines`."""
    i0, i1, w0, w1, never = cw
    D = (w0[:, None, None, None] * D_lines[i0] + w1[:, None, None, None]
         * D_lines[i1])
    bt = np.asarray(b0, float).T
    D = np.clip(D, 0.0, bt[None, None])           # rounding of w0 + w1 = 1
    D[never] = bt[None]
    return D


# ---------------------------------------------------------------------------
# azimuthal orders
# ---------------------------------------------------------------------------
def sector_centres(n_sectors=N_SECTORS):
    """Centres of the sectors [deg]."""
    return (np.arange(n_sectors) + 0.5) * 360.0 / n_sectors


def _sinc(x):
    return np.where(np.abs(x) < 1e-12, 1.0, np.sin(x) / np.where(x == 0, 1.0, x))


def fourier_sectors(V, M, axis=1):
    """Orders 0..M of a function given by its sector averages.

    ``V`` holds the means over the ``n`` sectors (along ``axis``); returns
    ``(c, s)``, each with ``axis`` replaced by M + 1, such that the function is
    ``sum_m c_m cos(m phi) + s_m sin(m phi)`` (phi the arrival azimuth, radians from
    north through east): discrete Fourier sum over the sector centres divided by
    ``sinc(m pi / n)`` (the sector average of cos(m phi)).  ``M <= n / 2``; at
    ``M = n / 2`` the sector values are reproduced exactly by :func:`synth_sectors`.
    """
    V = np.moveaxis(np.asarray(V, float), axis, -1)
    n = V.shape[-1]
    if M > n // 2:
        raise ValueError("M must be at most n_sectors / 2")
    ph = np.radians(sector_centres(n))
    m = np.arange(M + 1)
    cs = np.cos(m[:, None] * ph[None])
    sn = np.sin(m[:, None] * ph[None])
    f = np.where((m == 0) | (2 * m == n), 1.0 / n, 2.0 / n)
    sc = _sinc(m * np.pi / n)
    c = np.einsum("...j,mj->...m", V, cs) * (f / sc)
    s = np.einsum("...j,mj->...m", V, sn) * (f / sc)
    if 2 * M == n:
        # Nyquist order: only the component that is non-zero at the centres
        nz_c = np.max(np.abs(cs[M])) > 1e-9
        if not nz_c:
            c[..., M] = 0.0
        else:
            s[..., M] = 0.0
    s[..., 0] = 0.0
    return np.moveaxis(c, -1, axis), np.moveaxis(s, -1, axis)


def synth_sectors(c, s, n_sectors=N_SECTORS, axis=1):
    """Sector averages from orders (inverse of :func:`fourier_sectors`)."""
    c = np.moveaxis(np.asarray(c, float), axis, -1)
    s = np.moveaxis(np.asarray(s, float), axis, -1)
    M = c.shape[-1] - 1
    ph = np.radians(sector_centres(n_sectors))
    m = np.arange(M + 1)
    sc = _sinc(m * np.pi / n_sectors)
    V = (np.einsum("...m,mj->...j", c * sc, np.cos(m[:, None] * ph[None]))
         + np.einsum("...m,mj->...j", s * sc, np.sin(m[:, None] * ph[None])))
    return np.moveaxis(V, -1, axis)


def bin_mean_basis(M, edges_deg):
    """Means of cos(m phi) and sin(m phi) over azimuth bins (analytic).

    Returns ``(Cb, Sb)``, each (n_bins, M + 1)."""
    e = np.radians(np.asarray(edges_deg, float))
    a, b = e[:-1, None], e[1:, None]
    m = np.arange(M + 1)[None]
    w = b - a
    ms = np.where(m == 0, 1.0, m)
    Cb = np.where(m == 0, 1.0, (np.sin(ms * b) - np.sin(ms * a)) / (ms * w))
    Sb = np.where(m == 0, 0.0, (np.cos(ms * a) - np.cos(ms * b)) / (ms * w))
    return Cb, Sb


# ---------------------------------------------------------------------------
# G2.0: transmission of arbitrary injection states
# ---------------------------------------------------------------------------
def state_location(r_cm, u, r_e=R_EARTH_CM):
    """Location of the line of each state: first ground crossing, or the point
    under the tangent point for a line that misses the ground.

    Returns ``(q (n, 3) unit positions, hit (n,), cosz (n,) at the location
    (0 for misses), az_deg (n,) arrival azimuth there, ht_km (n,) tangent
    altitude for misses (0 for hits))``.
    """
    r = np.atleast_2d(np.asarray(r_cm, float))
    u = np.atleast_2d(np.asarray(u, float))
    b = np.einsum("ij,ij->i", r, u)
    c = np.einsum("ij,ij->i", r, r) - r_e ** 2
    disc = b * b - c
    s = -b - np.sqrt(np.maximum(disc, 0.0))
    hit = (disc >= 0.0) & (s > 0.0)
    st = -b                                   # closest approach
    p = np.where(hit[:, None], r + s[:, None] * u, r + st[:, None] * u)
    q = p / np.linalg.norm(p, axis=1)[:, None]
    up, north, east = local_frame(q)
    cosz = np.where(hit, -np.einsum("ij,ij->i", u, up), 0.0)
    az = np.degrees(np.arctan2(-np.einsum("ij,ij->i", u, east),
                               -np.einsum("ij,ij->i", u, north))) % 360.0
    ht = np.where(hit, 0.0, (np.linalg.norm(p, axis=1) - r_e) / 1e5)
    return q, hit, cosz, az, ht


def _node_weights_one(nodes, hit, cosz, ht_km):
    """Two line nodes and weights for one line coordinate (as chain_weights)."""
    kind = np.asarray(nodes["kind"])
    gi = np.flatnonzero(kind == 0)
    ti = np.flatnonzero(kind == 1)
    if hit:
        zg = np.degrees(np.arccos(np.clip(np.asarray(nodes["cosz"])[gi], -1, 1)))
        h0 = ti[np.asarray(nodes["ht_km"])[ti] == 0.0]
        if h0.size:
            zg, gi = np.append(zg, 90.0), np.append(gi, h0[0])
        o = np.argsort(zg)
        xs, ids, x = zg[o], gi[o], np.degrees(np.arccos(np.clip(cosz, -1, 1)))
    else:
        ht = np.asarray(nodes["ht_km"])[ti]
        o = np.argsort(ht)
        xs, ids, x = ht[o], ti[o], ht_km
    p = int(np.clip(np.searchsorted(xs, x) - 1, 0, xs.size - 2))
    t = float(np.clip((x - xs[p]) / (xs[p + 1] - xs[p]), 0.0, 1.0))
    return (ids[p], ids[p + 1]), (1.0 - t, t)


def sky_transmission(r_cm, u, R, tables, grid, cover_deg=10.0):
    """Transmission the sky tables give injection states (the G2.0 interpolation).

    For each state: its line's location Q (:func:`state_location`), the bilinear
    weights of Q on ``grid``, and in each of those tables the verdict at (line node,
    sector of the azimuth at Q, rigidity half-interval of R), linear in the line
    node; 1 above the last ladder edge; 0 on lines that never reach the injection
    sphere.

    Parameters
    ----------
    tables : dict
        grid node index -> :class:`SkyTable`.
    grid : LocationGrid

    Returns
    -------
    T (n,), covered (n,) bool (location within ``cover_deg`` of the site and its
    four tables present), hit (n,) bool.
    """
    r = np.atleast_2d(np.asarray(r_cm, float))
    u = np.atleast_2d(np.asarray(u, float))
    R = np.broadcast_to(np.asarray(R, float), (r.shape[0],))
    q, hit, cosz, az, ht = state_location(r, u)
    x, y = xy_of(q, grid.site)
    idx, w, inside = grid.weights_xy(x, y)
    cover = inside & (np.hypot(x, y) <= cover_deg)
    T = np.full(r.shape[0], np.nan)
    for i in range(r.shape[0]):
        if not cover[i]:
            continue
        tabs = []
        for p, wp in zip(idx[i], w[i]):
            if wp == 0.0:
                continue
            if int(p) not in tables:
                break
            tabs.append((tables[int(p)], wp))
        else:
            val = 0.0
            for tab, wp in tabs:
                edges = tab.edges
                ns = int(tab.n_sectors)
                if R[i] >= edges[-1]:
                    val += wp
                    continue
                rr = int(np.clip(np.searchsorted(edges, R[i]) - 1, 0,
                                 edges.size - 2))
                sec = int(np.floor(az[i] / (360.0 / ns))) % ns
                (n0, n1), (a0, a1) = _node_weights_one(tab.nodes, hit[i],
                                                       cosz[i], ht[i])
                al = tab.allowed
                v = a0 * al[n0, sec, rr] + a1 * al[n1, sec, rr]
                val += wp * v
            T[i] = val
            continue
        cover[i] = False
    return T, cover, hit


def parse_spacing(text):
    """"5" -> 5.0, "5,1.25" -> (5.0, 1.25) (command-line option of the gates)."""
    v = [float(t) for t in str(text).split(",")]
    if len(v) == 1:
        return v[0]
    if len(v) != 2:
        raise ValueError("spacing: one value or two (x,y)")
    return tuple(v)


def table_path(sky_dir, xy, h_ref_km=0.0, spacing_tag=""):
    """Canonical file name of a sky table."""
    return os.path.join(sky_dir, f"sky_h{h_ref_km:g}_x{xy[0]:+05.1f}_y{xy[1]:+05.1f}"
                        f"{spacing_tag}.npz")


def main(argv=None):
    """Build one sky table: ``boundary.py build OUT_DIR [--x X --y Y --h-ref H
    --seed S --jobs N]``."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build",))
    ap.add_argument("out_dir")
    ap.add_argument("--x", type=float, default=0.0)
    ap.add_argument("--y", type=float, default=0.0)
    ap.add_argument("--h-ref", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--min-chunk", type=int, default=10000)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    path = table_path(a.out_dir, (a.x, a.y), a.h_ref)
    t0, c0 = time.time(), time.process_time()
    tab = SkyTable.build(path, (a.x, a.y), h_ref_km=a.h_ref, seed=a.seed,
                         n_jobs=a.jobs, min_chunk=a.min_chunk,
                         log=lambda s: print(s, flush=True))
    s = tab.summary()
    s.update({"path": path, "sha256": _sha256(path),
              "driver_wall_s": time.time() - t0,
              "driver_cpu_s": time.process_time() - c0, "seed": a.seed,
              "affinity": sorted(os.sched_getaffinity(0))})
    print(json.dumps(s, indent=1), flush=True)
    with open(path + ".json", "w") as fh:
        json.dump(s, fh, indent=1)
    return s


if __name__ == "__main__":
    main()
