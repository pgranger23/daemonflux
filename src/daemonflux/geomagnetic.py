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


# ----------------------------------------------------------------------
# Tabulated, per-species 3D factor (e.g. from a Monte Carlo or the engine)
# ----------------------------------------------------------------------
#: species a table may carry; the first four are the neutrino species of daemonflux
TABLE_SPECIES = ("numu", "antinumu", "nue", "antinue", "mu+", "mu-")
_SUM_COMPONENTS = {
    "numuflux": ("numu", "antinumu"),
    "nueflux": ("nue", "antinue"),
    "muflux": ("mu+", "mu-"),
}
_SPECIES_QUANTITY = set(TABLE_SPECIES)
#: ratio quantity -> (numerator species, denominator species); a species *tuple*
#: stands for the flux-weighted combination of the components
_RATIOS = {
    "numuratio": (("numu",), ("antinumu",)),
    "nueratio": (("nue",), ("antinue",)),
    "muratio": (("mu+",), ("mu-",)),
    "flavorratio": (("numu", "antinumu"), ("nue", "antinue")),
}


def _split_quantity(quantity: str):
    """``('total_' or '', core)`` of a daemonflux quantity name."""
    if quantity.startswith("total_"):
        return "total_", quantity[len("total_") :]
    return "", quantity


class Tabulated3DModel:
    """Per-species, tabulated 3D factor ``F_s(E, cos Z, azimuth)`` for daemonflux.

    ``flux_3D_s = flux_1D_s * F_s``.  Unlike the analytic
    :class:`GeomagneticModel`, ``F`` is **not** confined to [0, 1] (the 3D
    production geometry raises the near-horizon flux, so ``F`` can exceed 1) and it
    differs between species (the charge/flavour fan-out of the East-West effect).
    The factor is the 3D flux with cutoff and bending (Monte Carlo or engine) divided
    by a common 1D no-cutoff reference (MCEq, curved atmosphere, same primary model),
    so that ``daemonflux_1D * F`` is the 3D flux.

    Parameters
    ----------
    table : str, pathlib.Path or dict
        An ``.npz`` file (or the dict of its arrays) with

        ``species``      (S,)   names from :data:`TABLE_SPECIES`
        ``e_edges``      (nE+1,) energy bin edges [GeV], ascending
        ``cosz_edges``   (nC+1,) cos Z bin edges, ascending in (-1, 1); cos Z < 0 is
                         up-going
        ``az_edges``     (nA+1,) azimuth bin edges [deg, N=0, E=90], spanning 360
        ``factor``       (S, nE, nC, nA)  the factor, finite
        ``factor_error`` (S, nE, nC, nA)  its 1-sigma statistical uncertainty
                         (absolute, same units as ``factor``)

        and optionally ``label``, ``status`` (a string such as ``"PROVISIONAL"``,
        shown in ``repr``), ``site`` and ``provenance``.
    below_emin : {"warn", "raise"}
        Behaviour for energies below ``e_edges[0]``: warn (default) and use the
        lowest bin, or raise ``ValueError``.
    blend_decades : float
        Above ``E_max = e_edges[-1]`` the factor is blended **linearly in log10 E** to 1
        over this many decades (default 1): ``F = (1 - w) F(E_max) + w``,
        ``w = clip(log10(E / E_max) / blend_decades, 0, 1)``; the factor error is
        blended to 0 the same way.

    Interpolation
    -------------
    Linear in ``ln E`` between bin centres (geometric means of the edges), linear in
    cos Z between bin centres, linear and **periodic** in azimuth between bin centres.
    Outside the outermost centres (but inside the edges) the outermost value is used.
    Requests outside the table's cos Z coverage raise ``ValueError``.

    If the table has a cos Z edge at 0 with bins on both sides, the two hemispheres are
    interpolated **separately**: the up- and down-going patterns are physically distinct
    (different production points and cutoffs), so they are never blended across the
    horizon; between the horizon and the first bin centre of a hemisphere that bin's
    value is used. A request at exactly cos Z = 0 uses the down-going hemisphere.

    Quantities
    ----------
    * a single species (``numu`` ... , also with the ``total_`` prefix) is multiplied
      by its own factor;
    * sums (``numuflux``, ``nueflux``, ``muflux``) by the flux-weighted factor of
      their components, ``(F_a Phi_a + F_b Phi_b) / (Phi_a + Phi_b)``, with the 1D
      component fluxes as weights;
    * ratios (``numuratio``, ``nueratio``, ``muratio``, ``flavorratio``) by the
      quotient of the
      numerator and denominator factors (the same weighted factor for the sums in
      ``flavorratio``), i.e. they are recomputed from the corrected components;
    * quantities the table does not cover (e.g. ``muflux``, ``mu+``, ``muratio`` for a
      neutrino table) pass through with factor 1 and zero table error (no warning, no
      zenith-coverage check); a sum or ratio with only one of its species in the table
      uses 1 for the missing one;
    * the factor is applied to the ``total_`` quantities as well, i.e. also to their
      prompt component (the table is a conventional-flux ratio); the prompt flux is
      negligible below ~10 GeV, and above the table's last energy edge the factor
      blends to 1.

    Errors: the 1D error is multiplied by the same scale as the flux (relative error of
    the 1D model preserved).  The table's own uncertainty is **not** folded in; see
    :meth:`factor_error` and :meth:`scale_error`.
    """

    #: marker used by daemonflux to route per-species handling and up-going zeniths
    per_species = True

    def __init__(self, table, below_emin: str = "warn", blend_decades: float = 1.0):
        if below_emin not in ("warn", "raise"):
            raise ValueError("below_emin must be 'warn' or 'raise'")
        if isinstance(table, (str, bytes)) or hasattr(table, "__fspath__"):
            with np.load(table, allow_pickle=False) as d:
                table = {k: d[k] for k in d.files}
        t = dict(table)
        self.species = tuple(str(s) for s in np.atleast_1d(t["species"]))
        for s in self.species:
            if s not in TABLE_SPECIES:
                raise ValueError(f"unknown species {s!r}; allowed: {TABLE_SPECIES}")
        self.e_edges = np.asarray(t["e_edges"], float)
        self.cosz_edges = np.asarray(t["cosz_edges"], float)
        self.az_edges = np.asarray(t["az_edges"], float)
        self._f = np.asarray(t["factor"], float)
        self._df = np.asarray(t["factor_error"], float)
        shape = (
            len(self.species),
            len(self.e_edges) - 1,
            len(self.cosz_edges) - 1,
            len(self.az_edges) - 1,
        )
        if self._f.shape != shape or self._df.shape != shape:
            raise ValueError(f"factor/factor_error must have shape {shape}")
        if not (np.all(np.isfinite(self._f)) and np.all(np.isfinite(self._df))):
            raise ValueError("factor and factor_error must be finite (fill empty bins)")
        for name, ax in (
            ("e_edges", self.e_edges),
            ("cosz_edges", self.cosz_edges),
            ("az_edges", self.az_edges),
        ):
            if np.any(np.diff(ax) <= 0):
                raise ValueError(f"{name} must be strictly increasing")
        if self.e_edges[0] <= 0:
            raise ValueError("e_edges must be positive")
        if not np.isclose(self.az_edges[-1] - self.az_edges[0], 360.0):
            raise ValueError("az_edges must span 360 degrees")
        self.below_emin = below_emin
        self.blend_decades = float(blend_decades)
        self.label = str(t.get("label", ""))
        self.status = str(t.get("status", ""))
        self.site = str(t.get("site", ""))
        self.provenance = str(t.get("provenance", ""))
        self._lne = np.log(np.sqrt(self.e_edges[:-1] * self.e_edges[1:]))
        self._cz = 0.5 * (self.cosz_edges[:-1] + self.cosz_edges[1:])
        self._az = 0.5 * (self.az_edges[:-1] + self.az_edges[1:])
        #: interpolate the hemispheres separately (edge at cos Z = 0, bins on both sides)
        self._split = bool(
            self.cosz_edges[0] < 0.0 < self.cosz_edges[-1]
            and np.any(np.isclose(self.cosz_edges, 0.0, rtol=0.0, atol=1e-12))
        )

    def __repr__(self) -> str:
        return (
            f"Tabulated3DModel(label={self.label!r}, status={self.status!r}, "
            f"site={self.site!r}, species={self.species}, "
            f"E=[{self.e_edges[0]:g}, {self.e_edges[-1]:g}] GeV, "
            f"cosZ=[{self.cosz_edges[0]:g}, {self.cosz_edges[-1]:g}], "
            f"n_az={len(self._az)})"
        )

    @property
    def e_min(self) -> float:
        return float(self.e_edges[0])

    @property
    def e_max(self) -> float:
        return float(self.e_edges[-1])

    # ------------------------------------------------------------------
    def _interp(self, arr, species, energy, zenith_deg, azimuth_deg, blend_to):
        """Interpolate one (S, nE, nC, nA) array for one species.

        Returns shape (nE,) for ``azimuth_deg`` None/scalar, (naz, nE) for an array.
        """
        energy = np.atleast_1d(np.asarray(energy, float))
        if species not in self.species:
            return np.full(
                (
                    energy.shape
                    if (azimuth_deg is None or np.ndim(azimuth_deg) == 0)
                    else (len(np.atleast_1d(azimuth_deg)),) + energy.shape
                ),
                blend_to,
            )
        cz = float(np.cos(np.deg2rad(float(zenith_deg))))
        tol = 1e-12
        if cz < self.cosz_edges[0] - tol or cz > self.cosz_edges[-1] + tol:
            raise ValueError(
                f"cos Z = {cz:.4f} (zenith {float(zenith_deg):g} deg) is outside the "
                f"table's coverage [{self.cosz_edges[0]:g}, {self.cosz_edges[-1]:g}]"
            )
        k = self.species.index(species)
        a = arr[k]  # (nE, nC, nA)
        # cos Z (linear, flat outside the outermost centres; per hemisphere if split)
        cidx = np.arange(a.shape[1])
        if self._split:
            cidx = cidx[self._cz > 0.0] if cz >= 0.0 else cidx[self._cz < 0.0]
        w1 = np.interp(cz, self._cz[cidx], np.arange(len(cidx)))  # fractional index
        j0 = int(np.floor(w1))
        j1 = min(j0 + 1, len(cidx) - 1)
        fr = w1 - j0
        a2 = (1.0 - fr) * a[:, cidx[j0], :] + fr * a[:, cidx[j1], :]  # (nE, nA)
        # azimuth (linear, periodic)
        if azimuth_deg is None:
            a3 = a2.mean(axis=1)[None, :]  # (1, nE)
            scalar = True
        else:
            scalar = np.ndim(azimuth_deg) == 0
            az = np.atleast_1d(np.asarray(azimuth_deg, float))
            c = self._az
            ext_c = np.concatenate([[c[-1] - 360.0], c, [c[0] + 360.0]])
            ext = np.concatenate([a2[:, -1:], a2, a2[:, :1]], axis=1)  # (nE, nA+2)
            azw = (az - c[0]) % 360.0 + c[0]
            idx = np.interp(azw, ext_c, np.arange(len(ext_c)))
            i0 = np.floor(idx).astype(int)
            i1 = np.minimum(i0 + 1, len(ext_c) - 1)
            f = idx - i0
            a3 = (1.0 - f)[:, None] * ext[:, i0].T + f[:, None] * ext[
                :, i1
            ].T  # (naz, nE)
        # energy: linear in ln E, flat in the half bins, blend to ``blend_to``
        # (1 for the factor, 0 for its error) above E_max
        if np.any(energy < self.e_edges[0]):
            msg = (
                f"energy {energy.min():g} GeV is below the table's lowest edge "
                f"{self.e_edges[0]:g} GeV"
            )
            if self.below_emin == "raise":
                raise ValueError(msg)
            import warnings

            warnings.warn(msg + "; the lowest bin is used", stacklevel=4)
        le = np.log(energy)
        out = np.empty((a3.shape[0], len(energy)))
        for i in range(a3.shape[0]):
            out[i] = np.interp(le, self._lne, a3[i])
        w = np.clip(np.log10(energy / self.e_max) / self.blend_decades, 0.0, 1.0)
        out = (1.0 - w) * out + w * blend_to
        return out[0] if scalar else out

    def factor(self, species, energy, zenith_deg, azimuth_deg=None):
        """The factor ``F_s`` (broadcast like :meth:`GeomagneticModel.admittance`)."""
        return self._interp(self._f, species, energy, zenith_deg, azimuth_deg, 1.0)

    def factor_error(self, species, energy, zenith_deg, azimuth_deg=None):
        """Interpolated 1-sigma statistical uncertainty of ``F_s`` (absolute)."""
        return self._interp(self._df, species, energy, zenith_deg, azimuth_deg, 0.0)

    # ------------------------------------------------------------------
    def _combo(self, species, weights, which, energy, zenith_deg, azimuth_deg):
        """Flux-weighted factor (``which`` = factor or factor_error) of components."""
        fn = self.factor if which == "factor" else self.factor_error
        num = 0.0
        den = 0.0
        for s, w in zip(species, weights):
            num = num + fn(s, energy, zenith_deg, azimuth_deg) * w
            den = den + w
        return num / den

    def covers(self, quantity: str) -> bool:
        """True if any species behind ``quantity`` is in the table; quantities that are
        not (muons for a neutrino table) pass through with factor 1."""
        _, core = _split_quantity(quantity)
        if core in _SPECIES_QUANTITY:
            sp = (core,)
        elif core in _SUM_COMPONENTS:
            sp = _SUM_COMPONENTS[core]
        elif core in _RATIOS:
            sp = _RATIOS[core][0] + _RATIOS[core][1]
        else:
            return False
        return any(s in self.species for s in sp)

    def _ones(self, energy, azimuth_deg, value):
        energy = np.atleast_1d(np.asarray(energy, float))
        if azimuth_deg is None or np.ndim(azimuth_deg) == 0:
            return np.full(energy.shape, value)
        return np.full((len(np.atleast_1d(azimuth_deg)),) + energy.shape, value)

    def scale(self, quantity, energy, zenith_deg, azimuth_deg, component_flux):
        """Multiplier of a daemonflux quantity (see the class docstring).

        ``component_flux(name)`` must return the 1D flux of species ``name`` (with the
        ``total_`` prefix of ``quantity`` where relevant) at the requested energy and
        zenith (used as weights for sums and flavour ratios only).
        """
        if not self.covers(quantity):
            return self._ones(energy, azimuth_deg, 1.0)
        pre, core = _split_quantity(quantity)
        if core in _SPECIES_QUANTITY:
            return self.factor(core, energy, zenith_deg, azimuth_deg)
        if core in _SUM_COMPONENTS:
            sp = _SUM_COMPONENTS[core]
            wts = [np.asarray(component_flux(pre + s), float) for s in sp]
            return self._combo(sp, wts, "factor", energy, zenith_deg, azimuth_deg)
        if core in _RATIOS:
            (nsp, dsp) = _RATIOS[core]
            out = []
            for sp in (nsp, dsp):
                wts = [np.asarray(component_flux(pre + s), float) for s in sp]
                out.append(
                    self._combo(sp, wts, "factor", energy, zenith_deg, azimuth_deg)
                )
            return out[0] / out[1]
        raise ValueError(f"quantity {quantity!r} is not known to the 3D table")

    def scale_error(self, quantity, energy, zenith_deg, azimuth_deg, component_flux):
        """Statistical uncertainty of :meth:`scale` from the table (components
        combined in quadrature with the flux weights, ratios by relative errors;
        the components' errors are treated as uncorrelated, which neglects the
        correlation through the common 1D reference of the Monte Carlo)."""
        if not self.covers(quantity):
            return self._ones(energy, azimuth_deg, 0.0)
        pre, core = _split_quantity(quantity)
        if core in _SPECIES_QUANTITY:
            return self.factor_error(core, energy, zenith_deg, azimuth_deg)

        def comb_err(sp):
            wts = [np.asarray(component_flux(pre + s), float) for s in sp]
            tot = sum(wts)
            v = 0.0
            for s, w in zip(sp, wts):
                v = (
                    v
                    + (self.factor_error(s, energy, zenith_deg, azimuth_deg) * w / tot)
                    ** 2
                )
            return np.sqrt(v)

        def comb_val(sp):
            wts = [np.asarray(component_flux(pre + s), float) for s in sp]
            return self._combo(sp, wts, "factor", energy, zenith_deg, azimuth_deg)

        if core in _SUM_COMPONENTS:
            return comb_err(_SUM_COMPONENTS[core])
        if core in _RATIOS:
            nsp, dsp = _RATIOS[core]
            n, d = comb_val(nsp), comb_val(dsp)
            r = n / d
            return r * np.sqrt((comb_err(nsp) / n) ** 2 + (comb_err(dsp) / d) ** 2)
        raise ValueError(f"quantity {quantity!r} is not known to the 3D table")
