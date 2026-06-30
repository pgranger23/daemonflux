"""mceq3d_flux -- a trustable, absolute, directional atmospheric-neutrino flux.

This is the production engine: it returns the **absolute** flux
``Phi(E, cos zenith, azimuth)`` for all four species (numu, antinumu, nue,
antinue) at a detector site, down to ~0.5 GeV, built so that every piece comes
from a *trusted* source and the 3D corrections are *validated against Honda*.

Construction (each factor trusted / validated)
---------------------------------------------
``Phi_3D(E, cosZ, az, s) = Phi_MCEq(E, cosZ, s) * G_s(E, R_c(cosZ, az))``

* **Phi_MCEq** -- MCEq solved per zenith with its **curved atmosphere** (the same
  engine daemonflux is built on). This supplies the absolute normalization, the
  full flavour/charge content, the spectra, *and* the sec(theta) horizon
  enhancement (a 1D-per-direction-with-curvature effect). Down-going hemisphere.
* **G_s(E, R_c)** -- the geomagnetic suppression = ``MCEq(primary cut at R_c) /
  MCEq(full)``, i.e. the **cascade-correct** response to removing primaries below
  the rigidity cutoff (NO ``x_eff`` hack). Precomputed on a small R_c grid (the
  suppression *ratio* is ~zenith-independent) and interpolated.
* **R_c(cosZ, az)** -- the first-principles **back-traced full-IGRF cutoff**
  (:mod:`geomag_backtrace`, validated: Kamioka 11.3 GV).

The rigidity cut is applied per-nucleus to the primary nucleons: the proton flux
is split into **free protons** (A/Z=1, R=E) and **bound protons** (in nuclei,
A/Z~2, R~2E) via MCEq's own p,n fluxes (free p = p-n by isospin), and neutrons are
all bound (R~2E).

The 1D base can be raw MCEq (default) or daemonflux's **muon-calibrated** flux
(``base_model="daemonflux"``) for a data-anchored absolute normalization.

Validation (`validate_honda.py`, and ``--validate`` here): the absolute Phi(E,
cosZ, az) for numu and nue matches the Honda HKKM2014 Kamioka tables in
normalization, zenith (sec theta) and azimuth (East-West).

Full sky
--------
* **Down-going (cosZ >= 0):** detector geomagnetic cutoff.
* **Up-going (cosZ < 0):** the production is up/down symmetric (same atmosphere at
  |cosZ|), but the geomagnetic cutoff is evaluated at the **far-side production
  point** (:func:`farside_production`) -- a global geomagnetic treatment, so the
  up-going hemisphere is included (essential for oscillation analyses).

Trust boundary (documented, not hidden)
---------------------------------------
* Up-going uses a single representative far-side production point per direction
  (the production region has finite extent; this is the leading geometric term).
* Nuclei: per-nucleus rigidity via the free/bound split (above); the bound part
  uses <A/Z>=2.0 (Fe is 2.08, a ~1% sub-component).
* Inter-direction streaming residual (~1-2%) and muon-bending E-W (charge-split,
  ~3 deg sub-GeV) are separate small corrections (see `spherical_streaming`,
  `muon_bending`).

Run::

    python mceq3d_flux.py --validate --plot
"""

from __future__ import annotations

import argparse

import numpy as np

CM2_PER_M2 = 1.0e4  # MCEq flux is per cm^2; Honda/this engine report per m^2
SPECIES = ("total_numu", "total_antinumu", "total_nue", "total_antinue")
SP_LABEL = {
    "total_numu": "numu",
    "total_antinumu": "antinumu",
    "total_nue": "nue",
    "total_antinue": "antinue",
}


def _transmission(e_grid, rc_gv, penumbra=0.5, az_over_z=1.0):
    """Smooth rigidity-cutoff transmission at R = az_over_z * E (erf step)."""
    from scipy.special import erf

    r = az_over_z * np.maximum(e_grid, 1e-9)
    return 0.5 * (1.0 + erf((np.log(r) - np.log(rc_gv)) / (np.sqrt(2) * penumbra)))


def farside_production(lat, lon, cos_zenith, azimuth, h_prod_km=20.0):
    """Far-side production point for an UP-going arrival (cos_zenith < 0).

    Up-going neutrinos are produced on the *opposite* side of the Earth and travel
    straight through it. The atmospheric production is up/down symmetric (same
    atmosphere at |cosZ|), but the **geomagnetic cutoff is set at the far-side
    production point**, not at the detector. This returns that point's
    ``(lat, lon)`` and the local primary arrival ``(cosZ, azimuth)`` there, so the
    detector's up-going geomagnetics can be evaluated globally.
    """
    from geomag_backtrace import _local_frame, arrival_direction, RE

    up = _local_frame(lat, lon)[0]
    P = RE * up
    d = arrival_direction(
        lat, lon, np.degrees(np.arccos(np.clip(cos_zenith, -1, 1))), azimuth
    )  # neutrino velocity (up-going for cosZ<0)
    Rh = RE + h_prod_km * 1e3
    Pd = P @ d
    s = Pd + np.sqrt(max(Pd**2 + (Rh**2 - RE**2), 0.0))  # far-side intersection
    Q = P - s * d
    qhat = Q / np.linalg.norm(Q)
    lat_q = np.degrees(np.arcsin(np.clip(qhat[2], -1, 1)))
    lon_q = np.degrees(np.arctan2(qhat[1], qhat[0]))
    upq, northq, eastq = _local_frame(lat_q, lon_q)
    src = -d  # primary source direction at Q
    cosz_q = float(src @ upq)
    horiz = src - cosz_q * upq
    az_q = np.degrees(np.arctan2(horiz @ eastq, horiz @ northq)) % 360.0
    return lat_q, lon_q, cosz_q, az_q


class MCEq3DFlux:
    """Absolute directional flux engine (full sky: down-going + up-going)."""

    def __init__(
        self,
        interaction_model="SIBYLL23D",
        primary=("HillasGaisser2012", "H3a"),
        e_min=0.1,
        atmosphere=None,
        base_model="mceq",
        daemonflux_location="generic",
    ):
        """``atmosphere`` is an MCEq ``density_model`` tuple. Default ``None`` keeps
        MCEq's realistic **CORSIKA US-Standard** layered profile (NOT the isothermal
        exponential used only in the `spherical_cascade` research demo). For
        seasonal/site tracking pass e.g. ``("MSIS00", ("SoudanMine", "January"))``.

        ``base_model`` selects the 1D base the geomagnetic factor multiplies:
        ``"mceq"`` (default) uses raw MCEq; ``"daemonflux"`` uses daemonflux's
        **muon-calibrated, data-anchored** 1D flux (numuflux/nueflux split by the
        ratios), which removes the ~10-20% hadronic-model normalization offset.
        """
        import crflux.models as crf
        from MCEq.core import MCEqRun
        import mceq_config as config

        self.base_model = base_model
        self._df = None
        if base_model == "daemonflux":
            from daemonflux import Flux

            self._df = Flux(location=daemonflux_location)
        config.e_min = e_min
        pm = (getattr(crf, primary[0]), primary[1])
        self.mceq = MCEqRun(
            interaction_model=interaction_model, primary_model=pm, theta_deg=0.0
        )
        if atmosphere is not None:
            self.mceq.set_density_model(atmosphere)
        self.e = self.mceq.e_grid
        self._phi0_std = self.mceq._phi0.copy()
        p = self.mceq.pman[(2212, 0)]
        n = self.mceq.pman[(2112, 0)]
        self._p_sl = slice(p.lidx, p.uidx)
        self._n_sl = slice(n.lidx, n.uidx)
        # Per-nucleus rigidity bookkeeping. The geomagnetic cutoff is on RIGIDITY
        # R = (A/Z)*E_nucleon, so free protons (A/Z=1) and bound nucleons
        # (He/CNO/Fe, A/Z~2) are cut at different energies. By isospin the bound
        # protons ~ the neutron flux, so from MCEq's own p/n nucleon fluxes:
        #   free protons  = p - n  (A/Z=1),   bound protons = n (A/Z~2).
        p_arr = self._phi0_std[self._p_sl]
        n_arr = self._phi0_std[self._n_sl]
        with np.errstate(invalid="ignore", divide="ignore"):
            self._f_free = np.where(
                p_arr > 0, np.clip((p_arr - n_arr) / p_arr, 0.0, 1.0), 1.0
            )
        # <A/Z> of bound nucleons, **nucleon-flux-weighted over the real primary
        # composition** (He/CNO/Si A/Z=2, Fe 2.077) instead of a fixed 2.0, so the
        # ~0.2-0.5% Fe sub-component is included and energy-dependent (Fe rises with
        # E). Per species the nucleon flux at per-nucleon energy E is A^2*Phi(A*E).
        cr = pm[0](pm[1])
        num = np.zeros_like(self.e)
        den = np.zeros_like(self.e)
        for cid in cr.nucleus_ids:
            Z, A = cr.Z_A(cid)
            if A <= 1:
                continue  # free protons handled separately (A/Z=1)
            w = A * A * np.ravel(cr.nucleus_flux(cid, A * self.e))
            num += (A / Z) * w
            den += w
        self._az_bound = np.where(den > 0, num / den, 2.0)

    # -- base: MCEq per zenith, curved atmosphere, no geomag --
    def base(self, cos_zeniths):
        """Phi_MCEq[species, |cosZ|, E] in /(m^2 s sr GeV) (curved, no geomag).

        Uses ``|cosZ|`` (production is up/down symmetric: an up-going neutrino is
        produced on the far side at the conjugate down-going slant).
        """
        if self.base_model == "daemonflux":
            return self._base_daemonflux(cos_zeniths)
        self.mceq._phi0[:] = self._phi0_std
        out = {s: np.zeros((len(cos_zeniths), len(self.e))) for s in SPECIES}
        for i, cz in enumerate(cos_zeniths):
            self.mceq.set_theta_deg(np.degrees(np.arccos(np.clip(abs(cz), 1e-3, 1))))
            self.mceq.solve()
            for s in SPECIES:
                out[s][i] = self.mceq.get_solution(s, 0) * CM2_PER_M2
        return out

    def _base_daemonflux(self, cos_zeniths):
        """Muon-calibrated 1D base from daemonflux [/(m^2 s sr GeV)], all flavours.

        daemonflux reports E^3-weighted *sums* (numuflux = nu_mu+nubar_mu) and the
        ratios (numuratio = nu_mu/nubar_mu); we de-weight by E^3, convert cm->m,
        and split into species with the ratios. Uses |cosZ| (up/down symmetric).

        Validity: this base matches Honda to ~10 % for E >~ 1 GeV but
        **over-predicts below ~0.3 GeV** (up to ~2x at 0.1 GeV), where it
        extrapolates past daemonflux's muon-calibration region; there the raw MCEq
        base is closer to Honda (see `base_comparison.py`). For sub-0.3-GeV work,
        cross-check both bases.
        """
        e = self.e
        m = e <= 1.0e9  # daemonflux splines are valid to ~1e9 GeV (>> 3D regime)
        ev = e[m]
        out = {s: np.zeros((len(cos_zeniths), len(e))) for s in SPECIES}
        for i, cz in enumerate(cos_zeniths):
            zen = float(np.degrees(np.arccos(np.clip(abs(cz), 1e-3, 1))))
            tot_mu = self._df.flux(ev, zen, "numuflux") / ev**3 * CM2_PER_M2
            r_mu = self._df.flux(ev, zen, "numuratio")  # nu_mu / nubar_mu
            tot_e = self._df.flux(ev, zen, "nueflux") / ev**3 * CM2_PER_M2
            r_e = self._df.flux(ev, zen, "nueratio")
            out["total_numu"][i, m] = tot_mu * r_mu / (1.0 + r_mu)
            out["total_antinumu"][i, m] = tot_mu / (1.0 + r_mu)
            out["total_nue"][i, m] = tot_e * r_e / (1.0 + r_e)
            out["total_antinue"][i, m] = tot_e / (1.0 + r_e)
        return out

    # -- geomagnetic response G_s(E, R_c) from cascade-correct cut/full --
    def geomag_response(self, rc_grid, cz_ref=1.0):
        """G[species, R_c, E] = MCEq(primary cut at R_c)/MCEq(full) at one zenith."""
        self.mceq.set_theta_deg(np.degrees(np.arccos(np.clip(cz_ref, 1e-3, 1))))
        self.mceq._phi0[:] = self._phi0_std
        self.mceq.solve()
        full = {s: self.mceq.get_solution(s, 0).copy() for s in SPECIES}
        G = {s: np.ones((len(rc_grid), len(self.e))) for s in SPECIES}
        for j, rc in enumerate(rc_grid):
            # proper per-nucleus rigidity: split the proton flux into free
            # (A/Z=1) and bound (A/Z~2); neutrons are all bound.
            t1 = _transmission(self.e, rc, az_over_z=1.0)
            t2 = _transmission(self.e, rc, az_over_z=self._az_bound)
            tp = self._f_free * t1 + (1.0 - self._f_free) * t2
            tn = t2
            self.mceq._phi0[:] = self._phi0_std
            self.mceq._phi0[self._p_sl] *= tp
            self.mceq._phi0[self._n_sl] *= tn
            self.mceq.solve()
            for s in SPECIES:
                cut = self.mceq.get_solution(s, 0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    G[s][j] = np.where(full[s] > 0, cut / full[s], 1.0)
        self.mceq._phi0[:] = self._phi0_std
        return G, rc_grid

    def cutoff_grid(
        self, lat, lon, cos_zeniths, azimuths, date, n_scan=14, r_lo=0.5, r_hi=20.0
    ):
        """R_c[cosZ, az] [GV]: detector cutoff (down-going), far-side (up-going).

        Both hemispheres use a single batched trajectory back-trace. For up-going,
        the primary's velocity at the far-side production point equals the
        (straight-line) neutrino direction ``d``, so the cutoff is a back-trace
        from the production point ``Q`` with ``u0 = -d`` -- the global geomagnetic
        treatment, batched like :func:`geomag_backtrace.cutoff_map`.
        """
        import geomag_backtrace as gb

        rc = np.zeros((len(cos_zeniths), len(azimuths)))
        down = cos_zeniths >= 0
        if np.any(down):
            zen = np.degrees(np.arccos(np.clip(cos_zeniths[down], 1e-3, 1)))
            rc[down] = gb.cutoff_map(
                lat, lon, date, zen, azimuths, n_scan=n_scan, r_lo=r_lo, r_hi=r_hi
            )
        ups = np.where(~down)[0]
        if len(ups):
            m_hat = gb.dipole_axis()
            rs = np.linspace(r_hi, r_lo, n_scan)
            r0, u0, R, idx = [], [], [], []
            for iz in ups:
                zdeg = np.degrees(np.arccos(np.clip(cos_zeniths[iz], -1, 1)))
                for ia, az in enumerate(azimuths):
                    latq, lonq, _, _ = farside_production(lat, lon, cos_zeniths[iz], az)
                    Q = (gb.RE + 20e3) * gb._local_frame(latq, lonq)[0]
                    d = gb.arrival_direction(lat, lon, zdeg, az)  # neutrino velocity
                    for Ri in rs:
                        r0.append(Q)
                        u0.append(-d)
                        R.append(Ri)
                    idx.append((iz, ia))
            allowed = gb.backtrace_vec(
                np.array(r0), np.array(u0), np.array(R), date, m_hat
            ).reshape(len(idx), n_scan)
            for k, (iz, ia) in enumerate(idx):
                forb = np.where(~allowed[k])[0]
                rc[iz, ia] = (
                    r_lo
                    if len(forb) == 0
                    else (
                        r_hi if forb[0] == 0 else 0.5 * (rs[forb[0] - 1] + rs[forb[0]])
                    )
                )
        return rc

    def solve(
        self,
        lat,
        lon,
        cos_zeniths,
        azimuths,
        date=None,
        n_scan=12,
        rc_grid=None,
        zenith_dependent_geomag=False,
    ):
        """Absolute Phi[species, cosZ, az, E] for a site (full sky).

        Down-going (cosZ>=0): detector geomagnetic cutoff. Up-going (cosZ<0):
        far-side production-point cutoff (global treatment, :func:`farside_production`).

        ``zenith_dependent_geomag``: by default the suppression ratio ``G_s(E,R_c)``
        is precomputed at the vertical column and reused at all zeniths -- validated
        zenith-independent to <=2% (sub-GeV horizon) by `geomag_zenith_check.py`,
        since the slant-depth shower effects cancel in the cut/full ratio. Set True
        to instead recompute ``G_s`` at *every* zenith (removes that <=2%
        approximation at the cost of one cascade pair per zenith band).
        """
        cos_zeniths = np.asarray(cos_zeniths, float)
        azimuths = np.asarray(azimuths, float)
        base = self.base(cos_zeniths)

        import datetime as _dt

        date = date or _dt.datetime(2020, 1, 1)
        rc_map = self.cutoff_grid(lat, lon, cos_zeniths, azimuths, date, n_scan)

        # Build the G(R_c) interpolation grid to *span the actual cutoff map*, so the
        # suppression is never clamped: a fixed floor (e.g. 2 GV) would apply spurious
        # suppression to low-cutoff directions/sites (polar R_c<2 GV) where G->1.
        if rc_grid is None:
            lo = max(0.1, float(np.min(rc_map)) * 0.9)
            hi = max(lo + 0.5, float(np.max(rc_map)) * 1.05)
            rc_grid = np.linspace(lo, hi, 12)
        if zenith_dependent_geomag:
            G_by_z = []
            for cz in cos_zeniths:
                Gz, rc_grid = self.geomag_response(rc_grid, cz_ref=max(abs(cz), 1e-3))
                G_by_z.append(Gz)
        else:
            G0, rc_grid = self.geomag_response(rc_grid)
            G_by_z = [G0] * len(cos_zeniths)

        flux = {
            s: np.zeros((len(cos_zeniths), len(azimuths), len(self.e))) for s in SPECIES
        }
        for s in SPECIES:
            for ia in range(len(azimuths)):
                for iz in range(len(cos_zeniths)):
                    g = _interp_rc(rc_map[iz, ia], rc_grid, G_by_z[iz][s])
                    flux[s][iz, ia] = base[s][iz] * g
        return dict(
            e=self.e,
            cos_zeniths=cos_zeniths,
            azimuths=azimuths,
            flux=flux,
            base=base,
            cutoff=rc_map,
        )


def _interp_rc(rc, rc_grid, gmat):
    """Linear interp of G at scalar ``rc`` along axis 0 of ``gmat`` (n_rc, n_e).

    Vectorised over energy (replaces a per-energy ``np.interp`` loop); clamps to the
    grid endpoints exactly like ``np.interp``.
    """
    rc = min(max(float(rc), rc_grid[0]), rc_grid[-1])
    j = int(np.searchsorted(rc_grid, rc))
    j = min(max(j, 1), len(rc_grid) - 1)
    w = (rc - rc_grid[j - 1]) / (rc_grid[j] - rc_grid[j - 1])
    return (1.0 - w) * gmat[j - 1] + w * gmat[j]


def horizon_grid(n_horizon=9, n_bulk=6, cz_max=0.95, horizon_half_width=0.2):
    """Full-sky cosθ grid **refined near the horizon** (cosθ→0).

    The flux (and the sec θ enhancement) vary fastest near cosθ=0, so a uniform
    grid interpolates poorly there. This packs ``n_horizon`` points (per
    hemisphere) into |cosθ| < ``horizon_half_width`` and ``n_bulk`` into the rest,
    symmetric in up/down. Pass the result as ``cos_zeniths`` to :meth:`solve`.
    """
    bulk = np.linspace(horizon_half_width, cz_max, n_bulk)
    horiz = np.linspace(0.02, horizon_half_width, n_horizon, endpoint=False)
    down = np.unique(np.concatenate([horiz, bulk]))
    return np.unique(np.concatenate([-down[::-1], down]))


def interp_flux(result, energy_gev, cos_zenith, azimuth_deg, species="total_numu"):
    """Evaluate a solved grid at (E, cosZ, azimuth) [/(m^2 s sr GeV)].

    Trilinear in (log E, azimuth, cosZ) on the grid from :meth:`MCEq3DFlux.solve`;
    azimuth wraps modulo 360.
    """
    e, cz, az = result["e"], result["cos_zeniths"], result["azimuths"]
    f = result["flux"][species]  # (cosZ, az, E)
    le = np.log(np.maximum(energy_gev, e[0]))
    # interpolate in log E for every (cosZ, az)
    fz = np.array(
        [
            [np.interp(le, np.log(e), f[i, j]) for j in range(len(az))]
            for i in range(len(cz))
        ]
    )  # (cosZ, az)
    az_ext = np.r_[az, az[0] + 360.0]
    row = np.array(
        [
            np.interp(azimuth_deg % 360.0, az_ext, np.r_[fz[i], fz[i, 0]])
            for i in range(len(cz))
        ]
    )  # (cosZ,) at this E, azimuth
    order = np.argsort(cz)
    return float(np.interp(cos_zenith, cz[order], row[order]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lat", type=float, default=36.43)
    p.add_argument("--lon", type=float, default=137.31)
    p.add_argument("--validate", action="store_true", help="compare vs Honda")
    p.add_argument("--plot", action="store_true")
    p.add_argument(
        "--base",
        choices=["mceq", "daemonflux"],
        default="mceq",
        help="1D base the geomag factor multiplies (daemonflux = data-anchored)",
    )
    args = p.parse_args(argv)

    df_loc = "kamioka" if abs(args.lat - 36.43) < 1 else "generic"
    eng = MCEq3DFlux(base_model=args.base, daemonflux_location=df_loc)
    cz = np.array([-0.95, -0.55, -0.15, 0.15, 0.55, 0.95])  # full sky
    az = np.array([0, 45, 90, 135, 180, 225, 270, 315], float)
    r = eng.solve(args.lat, args.lon, cz, az)
    e = r["e"]
    ie = int(np.argmin(np.abs(e - 1.0)))
    print(
        f"Absolute numu flux at 1 GeV (vertical, az-avg): "
        f"{r['flux']['total_numu'][0].mean(0)[ie]:.3g} /(m^2 s sr GeV)"
    )

    if args.validate or args.plot:
        _validate(r, args)


def _validate(r, args):
    import os

    if not os.path.exists("honda_kam.npz"):
        print("honda_kam.npz not present; run validate_honda.py first.")
        return
    h = dict(np.load("honda_kam.npz"))
    He, Hcz, nm = h["E"], h["czlo"], h["numu"]  # Honda numu[cosZ,az,E]
    e = r["e"]

    def _at(yvals, xgrid, E):
        """Log-log interpolate a falling flux to the exact energy E.

        Essential: the MCEq grid point nearest 1 GeV is 0.89 GeV, so comparing by
        nearest-index would pit our 0.89 GeV flux against Honda's 1.0 GeV bin (a
        ~1.5x energy mismatch on a steeply falling spectrum).
        """
        y = np.maximum(np.asarray(yvals), 1e-300)
        return float(np.exp(np.interp(np.log(E), np.log(xgrid), np.log(y))))

    # Dense low-E grid: 0.1-100 GeV is the region that matters (sub-GeV oscillation
    # physics), with extra points below 1 GeV.
    EGRID = (0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0)
    # az-averaged numu vs Honda: vertical (down 0.95) and up-going (-0.95)
    print("\nABSOLUTE numu (az-avg) vs Honda HKKM2014 [/(m^2 s sr GeV)], 0.1-100 GeV:")
    print("  E[GeV]   vert(this)   vert(Honda)  ratio | up(this) up(Honda) ratio")
    mine = {s: r["flux"][s] for s in SPECIES}
    izd = int(np.argmin(np.abs(r["cos_zeniths"] - 0.95)))
    izu = int(np.argmin(np.abs(r["cos_zeniths"] + 0.95)))
    ihzd = int(np.argmin(np.abs(Hcz - 0.9)))  # Honda down bin lo edge
    ihzu = int(np.argmin(np.abs(Hcz - (-1.0))))  # Honda up bin lo edge
    for E in EGRID:
        md = _at(mine["total_numu"][izd].mean(0), e, E)
        hd = _at(nm[ihzd].mean(0), He, E)
        mu = _at(mine["total_numu"][izu].mean(0), e, E)
        hu = _at(nm[ihzu].mean(0), He, E)
        print(
            f"  {E:6.2f}  {md:10.3g}  {hd:10.3g}  {md/hd:5.2f} |"
            f" {mu:8.3g} {hu:8.3g} {mu/hu:5.2f}"
        )

    # flavour ratio (nue+nuebar)/(numu+numubar) -- robust across models
    if "nue" in h:
        nue_h = h["nue"] + h["nuebar"]
        numu_h = h["numu"] + h["numubar"]
        print("\nFLAVOUR RATIO (nue+nuebar)/(numu+numubar) vs Honda (vertical):")
        print("  E[GeV]   this work   Honda")
        iz = int(np.argmin(np.abs(r["cos_zeniths"] - 0.95)))
        ihz = int(np.argmin(np.abs(Hcz - 0.9)))
        for E in (0.1, 0.2, 0.3, 0.5, 1.0, 3.0, 10.0):
            rm = (
                _at(mine["total_nue"][iz].mean(0), e, E)
                + _at(mine["total_antinue"][iz].mean(0), e, E)
            ) / (
                _at(mine["total_numu"][iz].mean(0), e, E)
                + _at(mine["total_antinumu"][iz].mean(0), e, E)
            )
            rh = _at(nue_h[ihz].mean(0), He, E) / _at(numu_h[ihz].mean(0), He, E)
            print(f"  {E:5.1f}     {rm:6.3f}     {rh:6.3f}")
    if args.plot:
        _plot(r, h)


def _plot(r, h):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e = r["e"]
    He, Hcz, nm = h["E"], h["czlo"], h["numu"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.4))
    # spectrum, down-going vertical: this work vs Honda, focused on 0.1-100 GeV
    iz = int(np.argmin(np.abs(r["cos_zeniths"] - 0.95)))
    mv = r["flux"]["total_numu"][iz].mean(0)
    ihz = int(np.argmin(np.abs(Hcz - 0.9)))
    s = (e >= 0.1) & (e <= 100)
    axL.loglog(e[s], (mv * e**3)[s], "C3o-", ms=3, label="this work (3D engine)")
    sH = (He >= 0.1) & (He <= 100)
    axL.loglog(He[sH], (nm[ihz].mean(0) * He**3)[sH], "k--", label="Honda HKKM2014")
    axL.set_xlabel("E [GeV]")
    axL.set_ylabel(r"$E^3\,\Phi_{\nu_\mu}$  [GeV$^2$/(m$^2$ s sr)]")
    axL.set_title("Absolute numu spectrum, vertical (Kamioka)")
    axL.legend()
    # zenith dependence at 1 GeV
    ie = int(np.argmin(np.abs(e - 1.0)))
    ih = int(np.argmin(np.abs(He - 1.0)))
    axR.plot(
        r["cos_zeniths"],
        r["flux"]["total_numu"][:, :, ie].mean(1),
        "C3o-",
        label="this work",
    )
    czc = Hcz + 0.05
    axR.plot(czc[czc > 0], nm[:, :, ih].mean(1)[czc > 0], "k--", label="Honda")
    axR.set_xlabel(r"$\cos\theta_z$")
    axR.set_ylabel(r"$\Phi_{\nu_\mu}$ at 1 GeV  [/(m$^2$ s sr GeV)]")
    axR.set_title("Zenith dependence (azimuth-averaged)")
    axR.legend()
    fig.tight_layout()
    fig.savefig("mceq3d_flux.png", dpi=110)
    print("\nsaved plot -> mceq3d_flux.png")


if __name__ == "__main__":
    main()
