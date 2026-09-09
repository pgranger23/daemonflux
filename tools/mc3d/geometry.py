"""Spherical-Earth geometry: local frames, injection sphere, detector cap.

Conventions (identical to ``tools/mceq3d/geomag_backtrace.py`` so the IGRF
back-tracer can be reused unchanged):

* geocentric Cartesian, z through the geographic north pole, x through
  (lat 0, lon 0);
* local frame ``(up, north, east)`` at a site;
* ``azimuth`` is the *arrival* (compass "from") azimuth: N = 0, E = 90;
  a particle arriving from azimuth ``az`` at zenith ``zen`` has velocity
  ``-cos(zen) up - sin(zen) (cos(az) north + sin(az) east)``.

Geodetic corrections.  The WGS84 ellipsoid differs from a sphere by
``f = 1/298.26`` (21 km equator-to-pole).  Three places could care:
(i) the slant depth of a near-horizontal ray -- the local radius of curvature
enters as ``sqrt(2 R H)``, so a 0.33% radius change is a 0.17% path change,
i.e. <0.2% on the horizon flux, an order below the statistical target;
(ii) the local vertical (deflection of the vertical, <=0.2 deg between geodetic
and geocentric latitude at mid-latitude) -- this shifts the zenith/azimuth
labels by <=0.2 deg, small against the 5 deg zenith bins and against the
~3-5 deg muon bend; (iii) the geomagnetic cutoff -- ``ppigrf`` is evaluated in
*geocentric* coordinates and ``geomag_backtrace`` already launches from a
spherical surface, so using the same sphere here keeps the two consistent.
**Decision: spherical Earth, R = 6371 km, no geodetic correction**, matching
``offaxis_mc``/``geomag_backtrace``.  Honda uses R_e = 6378.18 km; the 0.11%
difference is inside the same budget.
"""

from __future__ import annotations

import numpy as np

from constants import R_EARTH_CM, R_EARTH_KM


def local_frame(lat_deg, lon_deg):
    """(up, north, east) unit vectors at a site, geocentric Cartesian."""
    la, lo = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    up = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    north = np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)])
    east = np.array([-np.sin(lo), np.cos(lo), 0.0])
    return up, north, east


def site_position(lat_deg, lon_deg, h_cm=0.0):
    up, _, _ = local_frame(lat_deg, lon_deg)
    return (R_EARTH_CM + h_cm) * up


def arrival_direction(lat_deg, lon_deg, zenith_deg, azimuth_deg):
    """Velocity unit vector of a particle arriving from (zenith, azimuth)."""
    up, north, east = local_frame(lat_deg, lon_deg)
    th, az = np.deg2rad(zenith_deg), np.deg2rad(azimuth_deg)
    return -np.cos(th) * up - np.sin(th) * (np.cos(az) * north + np.sin(az) * east)


def zenith_azimuth(u, up, north, east):
    """Arrival zenith/azimuth [deg] of a particle whose *velocity* is ``u``."""
    d = -np.asarray(u, float)          # direction the particle came FROM
    cz = float(np.dot(d, up))
    zen = np.degrees(np.arccos(np.clip(cz, -1.0, 1.0)))
    az = np.degrees(np.arctan2(float(np.dot(d, east)), float(np.dot(d, north))))
    return zen, az % 360.0


def latlon_of(r_vec):
    r = np.asarray(r_vec, float)
    n = np.linalg.norm(r)
    return np.degrees(np.arcsin(r[2] / n)), np.degrees(np.arctan2(r[1], r[0]))


# ---------------------------------------------------------------------------
# Detector: Earth-concentric shell at ground level, Honda-style virtual cap
# ---------------------------------------------------------------------------
class DetectorCap:
    """Honda's 'virtual detector': the ground-sphere cap of angular radius
    ``theta_D`` centred on the site.

    Honda (astro-ph/0404457 sec. IV) uses ``theta_D = 10 deg`` (1117 km).  The
    cap area is ``A = 2 pi R^2 (1 - cos theta_D)``; a neutrino crossing it is
    scored with weight ``w / A`` so the tally is a flux per unit area.  The bias
    from the finite cap (the cutoff and the field vary across it) is removed by
    running **nested caps** ``theta_D in {2.5, 5, 7.5, 10} deg`` *on the same
    showers* -- every neutrino is scored into every cap it falls in, so the four
    estimates are nested subsets and maximally correlated -- and extrapolating
    the observable linearly in the cap solid angle ``Omega_D = 2 pi (1 - cos
    theta_D)`` to ``Omega_D -> 0``.  The leading bias is second order in
    ``theta_D`` for a smooth field (the first-order term averages out over the
    ring), so a linear fit in ``Omega_D ~ pi theta_D^2`` is the right form.
    """

    def __init__(self, lat_deg, lon_deg, theta_deg=10.0):
        self.lat, self.lon = float(lat_deg), float(lon_deg)
        self.theta_deg = float(theta_deg)
        self.cos_theta = float(np.cos(np.deg2rad(theta_deg)))
        self.centre = site_position(lat_deg, lon_deg)
        self.up, self.north, self.east = local_frame(lat_deg, lon_deg)

    @property
    def area_cm2(self):
        return 2.0 * np.pi * R_EARTH_CM ** 2 * (1.0 - self.cos_theta)

    @property
    def solid_angle(self):
        """Cap solid angle seen from the Earth centre [sr] (the extrapolation
        variable for the nested-cap limit)."""
        return 2.0 * np.pi * (1.0 - self.cos_theta)

    def contains(self, r_vec):
        r = np.asarray(r_vec, float)
        return float(np.dot(r, self.up)) / np.linalg.norm(r) >= self.cos_theta

    def crossing(self, r0, u):
        """Where the ray ``r0 + s u`` (s > 0) first meets the ground sphere.

        Returns ``(s, r_hit)`` or ``(nan, None)`` if it never does.
        """
        from atmosphere import ray_sphere
        s0, s1 = ray_sphere(r0, u, R_EARTH_CM)
        if not np.isfinite(s0):
            return np.nan, None
        s = s0 if s0 > 0.0 else s1
        if s <= 0.0:
            return np.nan, None
        return s, np.asarray(r0, float) + s * np.asarray(u, float)

    def score(self, r0, u):
        """Does a neutrino emitted at ``r0`` along ``u`` cross the cap?

        Returns ``(hit, zenith_deg, azimuth_deg, cos_incidence)``.
        """
        s, r_hit = self.crossing(r0, u)
        if r_hit is None:
            return False, np.nan, np.nan, np.nan
        n_hat = r_hit / np.linalg.norm(r_hit)
        if float(np.dot(n_hat, self.up)) < self.cos_theta:
            return False, np.nan, np.nan, np.nan
        # local frame at the *hit* point (Honda scores in the local frame of the
        # crossing, not of the site centre -- these differ by up to theta_D)
        lat, lon = latlon_of(r_hit)
        up, north, east = local_frame(lat, lon)
        zen, az = zenith_azimuth(u, up, north, east)
        return True, zen, az, abs(float(np.dot(u, n_hat)))


# ---------------------------------------------------------------------------
# Injection sphere
# ---------------------------------------------------------------------------
def sample_injection(rng, r_inj_cm, n=1):
    """Sample ``n`` inward-going states on the injection sphere.

    Position uniform on the sphere; direction from the **cosine (Lambert)**
    inward distribution ``p(mu) = 2 mu``, ``mu = -u.n_hat in (0, 1]``, which is
    the correct sampling for an isotropic external flux crossing a sphere.  The
    associated rate normalisation is
    ``N_dot = Phi_iso * pi * 4 pi R_inj^2`` (per unit energy), i.e. each sample
    carries weight ``Phi(E) * pi * 4 pi R_inj^2 / n_samples`` times the energy
    importance weight.
    """
    r_hat = rng.normal(size=(n, 3))
    r_hat /= np.linalg.norm(r_hat, axis=1)[:, None]
    mu = np.sqrt(rng.random(n))            # p(mu) = 2 mu
    phi = 2.0 * np.pi * rng.random(n)
    # build a tangent basis at each point
    a = np.zeros((n, 3))
    a[:, 2] = 1.0
    flip = np.abs(r_hat[:, 2]) > 0.9
    a[flip] = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(r_hat, a)
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(r_hat, e1)
    st = np.sqrt(np.maximum(1.0 - mu ** 2, 0.0))
    u = (-mu[:, None] * r_hat
         + st[:, None] * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2))
    return r_inj_cm * r_hat, u


def great_circle_deg(lat1, lon1, lat2, lon2):
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dl = np.deg2rad(lon2 - lon1)
    return np.degrees(np.arccos(np.clip(
        np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl), -1, 1)))


def km_of_deg(deg):
    return np.deg2rad(deg) * R_EARTH_KM
