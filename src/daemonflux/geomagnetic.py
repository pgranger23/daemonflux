"""Geomagnetic (3D) corrections for daemonflux.

The baseline daemonflux fluxes are computed with MCEq, which solves the
*one-dimensional* coupled cascade equations. The 1D approximation is excellent
above a few GeV but breaks down at low energy (below ~2 GeV), where the
trajectories of primary cosmic rays and low-energy secondaries are bent by the
geomagnetic field. The dominant low-energy effect is the **geomagnetic rigidity
cutoff**: primaries with magnetic rigidity below a direction-dependent threshold
cannot reach the top of the atmosphere, suppressing the lepton flux. The cutoff
depends on the geomagnetic latitude of the detector and on the *arrival
direction* (zenith **and** azimuth), the latter producing the well-known
East--West asymmetry.

This module implements a fast, analytic first-cut of that effect as a
multiplicative *admittance* factor ``G(E, theta, phi) in [0, 1]`` applied on top
of the 1D flux::

    flux_3D(E, theta, phi) = flux_1D(E, theta) * G(E, theta, phi)

By construction ``G -> 1`` at high energy, so the calibrated high-energy
behaviour of daemonflux is left untouched and the 3D result reduces exactly to
the published 1D result above the cutoff region.

Physics scope and limitations
-----------------------------
* **Two cutoff implementations are available and selectable.** By default the
  cutoff is the analytic **Stoermer** approximation (dipole field): fast,
  dependency-free, capturing the latitude dependence and East--West asymmetry but
  only approximate near the cutoff. Passing ``cutoff_source=`` to
  :class:`GeomagneticModel` instead uses a **first-principles back-traced cutoff**
  in the real **IGRF** field (``tools/mceq3d/geomag_backtrace.py``), which resolves
  the real-field structure and penumbra and matches the literature site cutoffs
  (e.g. Kamioka 11.3 GV). Both feed the same admittance machinery.
* The mapping from a *lepton* energy to the *primary* rigidity that produced it
  uses a single effective inelasticity ``x_eff`` (the mean fraction of the
  primary energy carried by the observed lepton). A full treatment folds the
  cutoff into the primary spectrum *before* the cascade.
* Charge-dependent effects on the East--West *charge ratio* are second order and
  are not modelled here (the same ``G`` multiplies all species, so it cancels in
  ratio quantities, which are therefore left unmodified).

All of the above are the well-defined "performance" simplifications that a full
3D Monte-Carlo (or a 3D-enabled MCEq) would remove. ``GeomagneticModel`` is the
single plug-point: replace :meth:`GeomagneticModel.admittance` (or load
MC-derived admittance ratios) and the rest of daemonflux is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np

# Stoermer constant at the Earth's surface (r = 1 Earth radius), in GV.
# The vertical cutoff at geomagnetic latitude ``lat`` is
# ``_STOERMER_GV * cos^4(lat) / 4``  (== 14.9 * cos^4 lat GV).
_STOERMER_GV = 59.6


@dataclass(frozen=True)
class GeomagneticSite:
    """Geomagnetic description of a detector location.

    Parameters
    ----------
    name : str
        Human-readable site name.
    geomagnetic_latitude_deg : float
        Geomagnetic latitude (dipole) in degrees. Drives the cutoff strength.
    declination_deg : float, optional
        Magnetic declination in degrees (offset between geographic and
        geomagnetic south). Small; defaults to 0.
    """

    name: str
    geomagnetic_latitude_deg: float
    declination_deg: float = 0.0

    @property
    def vertical_cutoff_GV(self) -> float:
        """Approximate Stoermer vertical cutoff rigidity in GV."""
        lat = np.deg2rad(self.geomagnetic_latitude_deg)
        return _STOERMER_GV * np.cos(lat) ** 4 / 4.0


# Representative geomagnetic latitudes, tuned so that ``vertical_cutoff_GV``
# roughly matches the literature vertical cutoff at each site. These are
# first-cut dipole values; replace with IGRF-derived cutoffs for production use.
KNOWN_SITES: Dict[str, GeomagneticSite] = {
    # Super-Kamiokande / Hyper-Kamiokande, Japan (vertical cutoff ~11 GV)
    "kamioka": GeomagneticSite("kamioka", 21.0),
    # IceCube / South Pole (cutoff ~0, essentially no suppression)
    "southpole": GeomagneticSite("southpole", -80.0),
    # INO/ICAL, India, near the geomagnetic equator (highest cutoff ~15 GV)
    "ino": GeomagneticSite("ino", 0.0),
    # Gran Sasso, Italy (vertical cutoff ~5 GV)
    "gransasso": GeomagneticSite("gransasso", 36.0),
    # SNOLAB, Canada (high latitude, low cutoff ~1 GV)
    "snolab": GeomagneticSite("snolab", 57.0),
}


def _is_ratio_quantity(quantity: str) -> bool:
    """Ratio quantities are dimensionless and unaffected by a common
    multiplicative admittance, so they must not be modulated."""
    return "ratio" in quantity


class GeomagneticModel:
    """Analytic geomagnetic admittance model.

    Parameters
    ----------
    site : GeomagneticSite or str
        The detector site. A string is looked up in :data:`KNOWN_SITES`.
    x_eff : float, optional
        Effective fraction of the primary energy carried by the observed lepton,
        used to map lepton energy to primary rigidity
        (``R_primary ~ E_lepton / x_eff``). Default 0.1.
    penumbra_width : float, optional
        Width (in natural log of rigidity) of the smooth transition across the
        cutoff. Default 0.5.
    n_azimuth_avg : int, optional
        Number of azimuth samples used when computing the azimuth-averaged
        admittance (``azimuth_deg=None``). Default 24.
    cutoff_source : callable, optional
        Selects **which cutoff implementation** to use. If ``None`` (default) the
        analytic **Stoermer** dipole formula (:meth:`cutoff_rigidity_GV`) is used.
        Pass a callable ``f(zenith_deg, azimuth_deg) -> R_c [GV]`` to use a
        **first-principles back-traced** (or Monte-Carlo) cutoff instead -- e.g. a
        closure around ``geomag_backtrace.cutoff_igrf`` for this site/epoch. This
        is the documented plug-point: the real-field cutoff then drives the same
        admittance machinery (the only remaining approximation being ``x_eff``;
        for the fully cascade-folded treatment use ``tools/mceq3d/mceq3d_flux``).
    """

    def __init__(
        self,
        site: Union[GeomagneticSite, str],
        x_eff: float = 0.1,
        penumbra_width: float = 0.5,
        n_azimuth_avg: int = 24,
        cutoff_source=None,
    ) -> None:
        if isinstance(site, str):
            key = site.lower()
            if key not in KNOWN_SITES:
                raise KeyError(
                    f"Unknown geomagnetic site '{site}'. "
                    f"Known sites: {sorted(KNOWN_SITES)}. "
                    "Pass a GeomagneticSite instance for a custom location."
                )
            site = KNOWN_SITES[key]
        self.site = site
        self.x_eff = float(x_eff)
        self.penumbra_width = float(penumbra_width)
        self.n_azimuth_avg = int(n_azimuth_avg)
        self.cutoff_source = cutoff_source

    def __repr__(self) -> str:
        return (
            f"GeomagneticModel(site={self.site.name!r}, "
            f"vertical_cutoff={self.site.vertical_cutoff_GV:.2f} GV, "
            f"x_eff={self.x_eff}, penumbra_width={self.penumbra_width})"
        )

    # ------------------------------------------------------------------
    # Cutoff rigidity
    # ------------------------------------------------------------------
    def cutoff_rigidity_GV(
        self,
        zenith_deg: Union[float, np.ndarray],
        azimuth_deg: Union[float, np.ndarray],
    ) -> np.ndarray:
        """Directional Stoermer cutoff rigidity in GV.

        Parameters
        ----------
        zenith_deg : float or np.ndarray
            Zenith angle in degrees (0 = down-going from the local zenith).
        azimuth_deg : float or np.ndarray
            Geographic azimuth in degrees measured from North, increasing
            towards the East (N=0, E=90, S=180, W=270).

        Returns
        -------
        np.ndarray
            Cutoff rigidity in GV with the broadcast shape of the inputs.

        Notes
        -----
        Stoermer cutoff for a positive particle::

            R_c = 59.6 cos^4(lat)
                  / [1 + sqrt(1 - sin(eps) sin(xi) cos^3(lat))]^2   [GV]

        with ``eps`` the angle from the local (magnetic) zenith, here
        approximated by the zenith angle, and ``xi`` the azimuth measured from
        geomagnetic south, positive towards the East. Because primaries are
        positively charged, the cutoff is lowest for arrival from the West,
        which is the origin of the East--West effect.

        If a ``cutoff_source`` was supplied (e.g. a back-traced IGRF cutoff), it
        is used instead of the analytic Stoermer formula below.
        """
        if self.cutoff_source is not None:
            return np.asarray(self.cutoff_source(zenith_deg, azimuth_deg), dtype=float)
        lat = np.deg2rad(self.site.geomagnetic_latitude_deg)
        eps = np.deg2rad(np.asarray(zenith_deg, dtype=float))
        # Geographic azimuth (from North, +East) -> angle from geomagnetic
        # south, +East:  xi = 180 - A - declination.
        xi = np.deg2rad(
            180.0 - np.asarray(azimuth_deg, dtype=float) - self.site.declination_deg
        )

        cos_lat = np.cos(lat)
        disc = 1.0 - np.sin(eps) * np.sin(xi) * cos_lat**3
        # Numerical guard: the discriminant is physically in [0, 2].
        disc = np.clip(disc, 0.0, None)
        denom = (1.0 + np.sqrt(disc)) ** 2
        return _STOERMER_GV * cos_lat**4 / denom

    # ------------------------------------------------------------------
    # Admittance
    # ------------------------------------------------------------------
    def _admittance_from_cutoff(
        self, energy: np.ndarray, cutoff_GV: np.ndarray
    ) -> np.ndarray:
        """Smooth admittance in [0, 1] given energy (GeV) and cutoff (GV).

        ``energy`` has shape ``(nE,)`` and ``cutoff_GV`` has shape ``(...,)``;
        the result has shape ``(..., nE)``.
        """
        from scipy.special import erf

        # Effective primary rigidity feeding a lepton of energy E (protons).
        r_eff = energy / self.x_eff  # (nE,) in GV
        cutoff_GV = np.atleast_1d(cutoff_GV).astype(float)

        # No appreciable cutoff (e.g. polar site): full admittance.
        out_shape = cutoff_GV.shape + energy.shape
        result = np.ones(out_shape)
        active = cutoff_GV > 1e-3
        if not np.any(active):
            return result

        # log-rigidity distance from the cutoff, broadcast (..., nE).
        log_r = np.log(r_eff)[np.newaxis, :]  # (1, nE)
        log_c = np.log(cutoff_GV[active])[:, np.newaxis]  # (nactive, 1)
        arg = (log_r - log_c) / (np.sqrt(2.0) * self.penumbra_width)
        result.reshape(-1, energy.shape[0])[active.ravel()] = 0.5 * (1.0 + erf(arg))
        return result

    def admittance(
        self,
        quantity: str,
        energy: Union[float, np.ndarray],
        zenith_deg: Union[float, np.ndarray],
        azimuth_deg: Optional[Union[float, np.ndarray]] = None,
    ) -> np.ndarray:
        """Geomagnetic admittance factor for a given quantity and direction.

        Parameters
        ----------
        quantity : str
            daemonflux quantity name. Ratio quantities return 1 (unmodulated).
        energy : float or np.ndarray
            Lepton energy/momentum in GeV.
        zenith_deg : float
            Zenith angle in degrees.
        azimuth_deg : float, np.ndarray or None
            Geographic azimuth in degrees (N=0, E=90, S=180, W=270). If ``None``
            the azimuth-averaged admittance is returned.

        Returns
        -------
        np.ndarray
            Admittance broadcastable against the 1D flux. For a scalar or
            ``None`` azimuth the shape matches ``energy``; for an array of
            ``naz`` azimuths the shape is ``(naz, nE)``.
        """
        if _is_ratio_quantity(quantity):
            return np.array(1.0)

        energy = np.atleast_1d(np.asarray(energy, dtype=float))

        if azimuth_deg is None:
            # Average the admittance over a uniform azimuth grid.
            az_grid = np.linspace(0.0, 360.0, self.n_azimuth_avg, endpoint=False)
            cutoff = self.cutoff_rigidity_GV(zenith_deg, az_grid)  # (naz,)
            adm = self._admittance_from_cutoff(energy, cutoff)  # (naz, nE)
            return adm.mean(axis=0)  # (nE,)

        scalar_az = np.ndim(azimuth_deg) == 0
        cutoff = self.cutoff_rigidity_GV(zenith_deg, azimuth_deg)
        adm = self._admittance_from_cutoff(energy, cutoff)  # (naz, nE) or (1, nE)
        if scalar_az:
            return adm[0]  # (nE,)
        return adm  # (naz, nE)
