"""Atmosphere, altitude levels and the line family of ``cascade3d`` (WP 1.1).

Everything here is plain numpy and computed once per geometry.  Units: cm, g, g/cm^3.

Conventions (``DESIGN.md`` section 2)
-------------------------------------
* Spherical Earth of radius ``R_E_CM`` = 6391 km and top of the atmosphere
  ``H_TOP_CM`` = 112.8 km: MCEq's ``config.r_E`` and ``config.h_atm``.
* :class:`CorsikaAtmosphere` reproduces MCEq's ``CorsikaAtmosphere.get_density``
  (``corsikaatm.c``) for ``BK_USStd``: layer ``i`` is used when ``h > hl[i]``
  (strict), ``rho = b_i / c_i exp(-h / c_i)`` in layers 0 to 3 and ``b_4 / c_4``
  (1e-9 g/cm^3) above 100 km.
* A straight line keeps its impact parameter ``b`` (distance of closest approach to
  the Earth's centre).  Ground lines have ``b < R`` and are labelled by the cosine of
  their zenith angle at the ground, ``mu0``; tangent lines have ``b = R + h_t``.
"""

import numpy as np

#: MCEq ``config.r_E`` [cm]
R_E_CM = 6391.0e5
#: MCEq ``config.h_atm`` [cm]
H_TOP_CM = 112.8e5

#: CORSIKA ``BK_USStd`` (MCEq ``atmosphere_parameters``)
_BK_USSTD = {
    "a": np.array([-149.801663, -57.932486, 0.63631894, 4.3545369e-4, 0.01128292]),
    "b": np.array([1183.6071, 1143.0425, 1322.9748, 655.69307, 1.0]),
    "c": np.array([954248.34, 800005.34, 629568.93, 737521.77, 1.0e9]),
    "hl": np.array([0.0, 7.0e5, 1.14e6, 3.7e6, 1.0e7]),
}

#: G1.1 directions (cos Z at the ground)
G11_COSZ = (1.0, 0.75, 0.5, 0.25, 0.15, 0.05)
#: G1.3 cos Z bin edges (ten bins from 0 to 1)
G13_COSZ_EDGES = tuple(np.round(np.linspace(0.0, 1.0, 11), 10))


class CorsikaAtmosphere:
    """MCEq's CORSIKA 5-layer density, ``BK_USStd`` by default.

    Parameters
    ----------
    params : dict, optional
        Arrays ``a``, ``b``, ``c``, ``hl`` of the CORSIKA parameterisation.
    h_top : float
        Top of the atmosphere [cm].
    """

    name = "CORSIKA BK_USStd"

    def __init__(self, params=None, h_top=H_TOP_CM, r_e=R_E_CM):
        p = _BK_USSTD if params is None else params
        self.b = np.asarray(p["b"], float)
        self.c = np.asarray(p["c"], float)
        self.hl = np.asarray(p["hl"], float)
        self.h_top = float(h_top)
        self.r_e = float(r_e)
        #: altitudes at which the density formula changes (become levels)
        self.boundaries = self.hl[1:].copy()

    def layer(self, h):
        """MCEq's layer index: the largest ``i`` with ``h > hl[i]`` (0 if none)."""
        h = np.asarray(h, float)
        lay = np.zeros(h.shape, dtype=np.int64)
        for i in range(5):
            lay = np.where(h > self.hl[i], i, lay)
        return lay

    def density(self, h, layer=None):
        """Density [g/cm^3] at altitude ``h`` [cm].

        ``layer`` forces the layer index (used to evaluate a segment's interior
        formula at its end points); by default MCEq's rule.
        """
        h = np.asarray(h, float)
        lay = self.layer(h) if layer is None else np.asarray(layer)
        b, c = self.b[lay], self.c[lay]
        return np.where(lay == 4, b / c, b / c * np.exp(-h / c))

    def vertical_depth(self, h):
        """Vertical grammage above ``h`` [g/cm^2] up to ``h_top`` (exact integral)."""
        h = np.atleast_1d(np.asarray(h, float))
        out = np.zeros(h.shape)
        edges = np.concatenate([self.hl, [self.h_top]])
        for i in range(5):
            lo = np.clip(h, edges[i], edges[i + 1])
            hi = edges[i + 1]
            if i == 4:
                out += self.b[4] / self.c[4] * (hi - lo)
            else:
                out += self.b[i] * (np.exp(-lo / self.c[i]) - np.exp(-hi / self.c[i]))
        return out


class ExponentialAtmosphere:
    """Isothermal test atmosphere ``rho = rho0 exp(-h / H)`` up to ``h_top``.

    Parameters
    ----------
    rho0 : float
        Density at the ground [g/cm^3].
    scale : float
        Scale height [cm].
    h_top : float
        Top [cm].
    """

    name = "exponential"

    def __init__(self, rho0=1.225e-3, scale=7.0e5, h_top=80.0e5, r_e=R_E_CM):
        self.rho0, self.scale = float(rho0), float(scale)
        self.h_top, self.r_e = float(h_top), float(r_e)
        self.boundaries = np.zeros(0)

    def layer(self, h):
        return np.zeros(np.shape(h), dtype=np.int64)

    def density(self, h, layer=None):
        h = np.asarray(h, float)
        return self.rho0 * np.exp(-h / self.scale)

    def vertical_depth(self, h):
        h = np.asarray(h, float)
        return self.rho0 * self.scale * (
            np.exp(-h / self.scale) - np.exp(-self.h_top / self.scale))


def make_levels(atm, spec=None, refine=1):
    """Altitude levels [cm] from 0 to ``atm.h_top``.

    Parameters
    ----------
    atm : atmosphere
        Supplies ``h_top`` and ``boundaries`` (always included as levels).
    spec : sequence of (h_upper_km, dh_km), optional
        Piecewise-constant spacing: ``dh`` up to ``h_upper``.  Default
        ``((50, 0.4), (70, 1.0), (inf, 3.0))``.
    refine : int
        Every interval is split into ``refine`` equal parts (``refine=2`` is the
        G1.2 doubling of the altitude grid).

    Returns
    -------
    numpy.ndarray
        Strictly increasing levels, first 0, last ``h_top``.
    """
    if spec is None:
        spec = ((50.0, 0.4), (70.0, 1.0), (np.inf, 3.0))
    top = atm.h_top
    pts = [0.0]
    lo = 0.0
    for h_up, dh in spec:
        hi = min(h_up * 1e5, top)
        if hi <= lo:
            continue
        n = max(1, int(np.ceil((hi - lo) / (dh * 1e5) - 1e-9)))
        pts.extend(np.linspace(lo, hi, n + 1)[1:])
        lo = hi
        if hi >= top:
            break
    lev = np.array(pts)
    lev = np.union1d(lev, [b for b in atm.boundaries if 0.0 < b < top])
    # merge levels closer than 1 m (a boundary next to a grid point)
    keep = np.concatenate([[True], np.diff(lev) > 100.0])
    lev = lev[keep]
    lev[-1] = top
    if refine > 1:
        sub = [lev[:1]]
        for a, b in zip(lev[:-1], lev[1:]):
            sub.append(np.linspace(a, b, refine + 1)[1:])
        lev = np.concatenate(sub)
    return lev


def ground_mu0(n=48, extra=G11_COSZ + G13_COSZ_EDGES):
    """Ground directions (cos Z at the ground), ``0 < mu0 <= 1``.

    Half the nodes uniform in cos Z, half uniform in zenith angle (dense near the
    horizon in angle), plus ``extra`` (the G1.1 directions and the G1.3 bin edges);
    ``mu0 = 0`` is not a ground line (it is the tangent line at h = 0).
    """
    n1 = n // 2
    a = np.linspace(0.0, 1.0, n1 + 1)[1:]
    th = np.linspace(0.0, 0.5 * np.pi, n - n1 + 1)[1:-1]
    base = np.unique(np.round(np.concatenate([a, np.cos(th)]), 12))
    ex = np.unique(np.asarray(extra, float))
    ex = ex[(ex > 1e-6) & (ex <= 1.0)]
    # base nodes closer than 2e-3 to a requested direction are dropped
    if ex.size:
        near = np.min(np.abs(base[:, None] - ex[None, :]), axis=1) < 2e-3
        base = base[~near]
    mu = np.union1d(base[(base > 1e-6) & (base <= 1.0)], ex)
    # thin remaining base near-duplicates
    keep = np.concatenate([[True], np.diff(mu) > 1e-3])
    keep |= np.isin(mu, ex)
    return mu[keep]


class LineFamily:
    """The set of straight lines whose crossings with the levels are the nodes.

    Parameters
    ----------
    mu0 : array_like
        Ground directions of the ground lines (``0 < mu0 <= 1``).
    h_tangent : array_like
        Tangent altitudes [cm] of the tangent lines (``0 <= h_t < h_top``).  Those
        equal to a level get a node at the tangent point.
    r_e : float
        Earth radius [cm].
    """

    def __init__(self, mu0, h_tangent, r_e=R_E_CM):
        self.mu0 = np.sort(np.asarray(mu0, float))
        self.h_tangent = np.sort(np.asarray(h_tangent, float))
        self.r_e = float(r_e)

    @property
    def b(self):
        """Impact parameters [cm]: ground lines first, then tangent lines."""
        bg = self.r_e * np.sqrt(np.clip(1.0 - self.mu0 ** 2, 0.0, None))
        return np.concatenate([bg, self.r_e + self.h_tangent])

    @property
    def n_ground(self):
        return self.mu0.size


def make_line_family(levels, n_ground=48, refine=1, extra_mu0=None, r_e=R_E_CM):
    """Default line family: ``n_ground`` ground lines and one tangent line per level.

    ``refine=2`` doubles the family (G1.2): twice the ground nodes and one extra
    pass-through tangent line midway between consecutive levels.
    """
    extra = G11_COSZ + G13_COSZ_EDGES if extra_mu0 is None else extra_mu0
    mu0 = ground_mu0(n_ground * refine, extra)
    ht = list(levels[:-1])
    if refine > 1:
        for a, b in zip(levels[:-1], levels[1:]):
            ht.extend(a + (b - a) * np.arange(1, refine) / refine)
    return LineFamily(mu0, np.array(sorted(ht)), r_e=r_e)
