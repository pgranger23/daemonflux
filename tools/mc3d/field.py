"""The geomagnetic field inside the atmosphere, fast, for the bending of the
primary and of the muons (milestone 3b).

The cutoff back-trace (``cutoff.py``) evaluates IGRF exactly with
``cutoff.FastIGRF`` at every RK4 node.  The *cascade* needs the field at
millions of points that all lie inside a thin shell (0-130 km altitude) and
for which the back-trace's bit-for-bit reproducibility is irrelevant, so it
reads a **pre-tabulated grid** instead:

* nodes every ``dlat = dlon = 1 deg`` (geocentric latitude / longitude, the
  whole sphere, longitude periodic) and every ``dh = 10 km`` of altitude from
  0 to 130 km above the ``R_E = 6371 km`` sphere;
* the geocentric **Cartesian** components of B [T] at each node, from
  ``FastIGRF(model)`` at ``date`` (IGRF-14, 2020-01-01 by default -- the same
  model and epoch as the cutoff test, milestone-3b decision);
* trilinear interpolation in ``(lat, lon, h)``.

Accuracy (``test_m3b.py::test_field_grid_matches_igrf``): <= 6e-4 of |B| at
random points of the shell (median 1.5e-4), i.e. < 20 nT -- far below what moves a bend (the
muon bend is 5.4 deg per lifetime per 0.5 G; 3e-4 of it is 0.002 deg).  The
grid is built once per (model, date, spacing) in ~5 s and cached as an
``.npz`` (``cache_dir`` or ``$MC3D_FIELD_CACHE``); a forked worker inherits
the parent's instance.

Above the grid top (130 km) the value at the top node is returned scaled by
``(r_top/r)^3`` (dipole fall-off); nothing in the cascade goes there (the
atmosphere ends at 112.8 km).
"""

from __future__ import annotations

import os

import numpy as np

from constants import R_EARTH_CM

#: unit charge * c in GeV / (T m): d(theta)/ds = Q_C * B[T] / p[GeV]  [rad/m]
Q_C = 0.299792458


class FieldGrid:
    """Tabulated IGRF in the atmosphere; ``__call__(r_cm) -> B [T]``."""

    def __init__(self, model=None, date=None, dlat=1.0, dlon=1.0, dh_km=10.0,
                 h_max_km=130.0, cache_dir=None):
        import cutoff
        self.model = (model or cutoff.DEFAULT_MODEL).upper()
        self.date = cutoff._as_datetime(date or cutoff.DEFAULT_DATE)
        self.dlat, self.dlon, self.dh = float(dlat), float(dlon), float(dh_km)
        self.lat = np.arange(-90.0, 90.0 + 0.5 * dlat, dlat)
        self.lon = np.arange(-180.0, 180.0 + 0.5 * dlon, dlon)
        self.h = np.arange(0.0, h_max_km + 0.5 * dh_km, dh_km)
        cache_dir = cache_dir or os.environ.get("MC3D_FIELD_CACHE")
        fp = None
        if cache_dir:
            fp = os.path.join(cache_dir, f"field_{self.model}_"
                              f"{self.date:%Y%m%d}_{dlat:g}_{dlon:g}_{dh_km:g}_"
                              f"{h_max_km:g}.npz")
            if os.path.exists(fp):
                self.B = np.load(fp)["B"]
                self._prep()
                return
        self.B = self._build()
        if fp:
            os.makedirs(cache_dir, exist_ok=True)
            tmp = fp + f".{os.getpid()}.part"
            with open(tmp, "wb") as fh:
                np.savez(fh, B=self.B)
            os.replace(tmp, fp)
        self._prep()

    def _build(self):
        import cutoff
        f = cutoff.fast_igrf(self.model)
        LA, LO, H = np.meshgrid(self.lat, self.lon, self.h, indexing="ij")
        r_km = R_EARTH_CM * 1e-5 + H.ravel()
        # the pole rows are evaluated a hair off the pole (ppigrf divides by
        # sin(colat)); nothing in the cascade comes near a pole
        colat = np.clip(90.0 - LA.ravel(), 1e-6, 180.0 - 1e-6)
        lon = LO.ravel()
        Br, Bt, Bp = (np.ravel(c) * 1e-9 for c in f(r_km, colat, lon, self.date))
        th, ph = np.deg2rad(colat), np.deg2rad(lon)
        st, ct, sp, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
        rh = np.stack([st * cp, st * sp, ct], 1)
        thh = np.stack([ct * cp, ct * sp, -st], 1)
        phh = np.stack([-sp, cp, np.zeros_like(sp)], 1)
        B = Br[:, None] * rh + Bt[:, None] * thh + Bp[:, None] * phh
        return B.reshape(LA.shape + (3,))

    def _prep(self):
        self._nla, self._nlo, self._nh = self.B.shape[:3]
        self._flat = self.B.reshape(-1, 3)
        self._r_top = R_EARTH_CM + self.h[-1] * 1e5

    def __call__(self, r_cm):
        """B [T], shape (n, 3), at geocentric positions ``r_cm`` (n, 3) [cm]."""
        r = np.atleast_2d(np.asarray(r_cm, float))
        x, y, z = r[:, 0], r[:, 1], r[:, 2]
        rn = np.sqrt(x * x + y * y + z * z)
        lat = np.degrees(np.arcsin(np.clip(z / rn, -1.0, 1.0)))
        lon = np.degrees(np.arctan2(y, x))
        h = (np.minimum(rn, self._r_top) - R_EARTH_CM) * 1e-5
        fa = np.clip((lat + 90.0) / self.dlat, 0.0, self._nla - 1.000001)
        fo = np.clip((lon + 180.0) / self.dlon, 0.0, self._nlo - 1.000001)
        fh = np.clip(h / self.dh, 0.0, self._nh - 1.000001)
        ia, io, ih = fa.astype(int), fo.astype(int), fh.astype(int)
        ta, to, th = fa - ia, fo - io, fh - ih
        s_a, s_o = self._nlo * self._nh, self._nh
        base = ia * s_a + io * s_o + ih
        out = np.zeros((len(rn), 3))
        F = self._flat
        for da, wa in ((0, 1.0 - ta), (1, ta)):
            for do, wo in ((0, 1.0 - to), (1, to)):
                for dh, wh in ((0, 1.0 - th), (1, th)):
                    w = wa * wo * wh
                    out += w[:, None] * F[base + da * s_a + do * s_o + dh]
        above = rn > self._r_top
        if np.any(above):
            out[above] *= ((self._r_top / rn[above]) ** 3)[:, None]
        return out


def rotate(v, axis_unit, ang):
    """Rodrigues rotation of rows ``v`` (n, 3) by ``ang`` (n,) [rad] about the
    unit rows ``axis_unit`` (n, 3) (right-handed)."""
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    kv = np.einsum("ij,ij->i", axis_unit, v)[:, None]
    return v * c + np.cross(axis_unit, v) * s + axis_unit * kv * (1.0 - c)


def bend_rotation(u, B, q, p_gev, ds_cm):
    """``(axis, angle)`` of the magnetic rotation of direction ``u`` over the
    arc ``ds``: ``du/ds = (q c / p) u x B`` is a right-handed rotation about
    ``B_hat`` by ``-q Q_C |B| ds / p`` (``q`` the charge sign, ``p`` in GeV,
    ``B`` in T, ``ds`` in cm).  Rows with ``B = 0`` get a zero angle."""
    bn = np.linalg.norm(B, axis=1)
    ok = bn > 0
    axis = np.where(ok[:, None], B / np.where(ok, bn, 1.0)[:, None], 0.0)
    axis[~ok, 2] = 1.0
    ang = -q * Q_C * bn * (ds_cm * 1e-2) / p_gev
    return axis, np.where(ok, ang, 0.0)
