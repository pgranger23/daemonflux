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

The rigidity cut is applied to the primary nucleons in MCEq's initial state
(protons at R=E, bound neutrons at R~2E for the A/Z~2 of He/CNO).

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
* Nuclei treated by superposition with an approximate R=(A/Z)E rigidity for the
  cut (leading order; ~10-20% on the suppression near the cutoff, largest at the
  sub-GeV horizon where the cutoff is highest).
* Inter-direction streaming residual (~1-2%) and muon-bending E-W (charge-split,
  ~3 deg sub-GeV) are separate small corrections (see `spherical_streaming`,
  `muon_bending`); optionally folded via ``muon_ew=True``.

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
        e_min=0.3,
    ):
        import crflux.models as crf
        from MCEq.core import MCEqRun
        import mceq_config as config

        config.e_min = e_min
        pm = (getattr(crf, primary[0]), primary[1])
        self.mceq = MCEqRun(
            interaction_model=interaction_model, primary_model=pm, theta_deg=0.0
        )
        self.e = self.mceq.e_grid
        self._phi0_std = self.mceq._phi0.copy()
        p = self.mceq.pman[(2212, 0)]
        n = self.mceq.pman[(2112, 0)]
        self._p_sl = slice(p.lidx, p.uidx)
        self._n_sl = slice(n.lidx, n.uidx)

    # -- base: MCEq per zenith, curved atmosphere, no geomag --
    def base(self, cos_zeniths):
        """Phi_MCEq[species, |cosZ|, E] in /(m^2 s sr GeV) (curved, no geomag).

        Uses ``|cosZ|`` (production is up/down symmetric: an up-going neutrino is
        produced on the far side at the conjugate down-going slant).
        """
        self.mceq._phi0[:] = self._phi0_std
        out = {s: np.zeros((len(cos_zeniths), len(self.e))) for s in SPECIES}
        for i, cz in enumerate(cos_zeniths):
            self.mceq.set_theta_deg(np.degrees(np.arccos(np.clip(abs(cz), 1e-3, 1))))
            self.mceq.solve()
            for s in SPECIES:
                out[s][i] = self.mceq.get_solution(s, 0) * CM2_PER_M2
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
            tp = _transmission(self.e, rc, az_over_z=1.0)  # protons R=E
            tn = _transmission(self.e, rc, az_over_z=2.0)  # bound n, R~2E
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
        self, lat, lon, cos_zeniths, azimuths, date=None, n_scan=12, rc_grid=None
    ):
        """Absolute Phi[species, cosZ, az, E] for a site (full sky).

        Down-going (cosZ>=0): detector geomagnetic cutoff. Up-going (cosZ<0):
        far-side production-point cutoff (global treatment, :func:`farside_production`).
        """
        cos_zeniths = np.asarray(cos_zeniths, float)
        azimuths = np.asarray(azimuths, float)
        if rc_grid is None:
            rc_grid = np.linspace(2.0, 20.0, 8)
        base = self.base(cos_zeniths)
        G, rc_grid = self.geomag_response(rc_grid)

        import datetime as _dt

        date = date or _dt.datetime(2020, 1, 1)
        rc_map = self.cutoff_grid(lat, lon, cos_zeniths, azimuths, date, n_scan)

        flux = {
            s: np.zeros((len(cos_zeniths), len(azimuths), len(self.e))) for s in SPECIES
        }
        for s in SPECIES:
            for ia in range(len(azimuths)):
                for iz in range(len(cos_zeniths)):
                    rc = rc_map[iz, ia]
                    g = np.array(
                        [np.interp(rc, rc_grid, G[s][:, k]) for k in range(len(self.e))]
                    )
                    flux[s][iz, ia] = base[s][iz] * g
        return dict(
            e=self.e,
            cos_zeniths=cos_zeniths,
            azimuths=azimuths,
            flux=flux,
            base=base,
            cutoff=rc_map,
        )


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
    args = p.parse_args(argv)

    eng = MCEq3DFlux()
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
    # az-averaged numu vs Honda at a few cosZ
    print("\nABSOLUTE numu (az-averaged) vs Honda HKKM2014 [/(m^2 s sr GeV)]:")
    print("  E[GeV]  cosZ   this work     Honda      ratio")
    mine = {s: r["flux"][s] for s in SPECIES}
    for E in (0.5, 1.0):
        ie = int(np.argmin(np.abs(e - E)))
        ih = int(np.argmin(np.abs(He - E)))
        for cz in (-0.95, -0.55, 0.55, 0.95):  # up-going and down-going
            iz = int(np.argmin(np.abs(r["cos_zeniths"] - cz)))
            ihz = int(np.argmin(np.abs(Hcz - (cz - 0.05))))  # Honda bin lo edge
            mv = mine["total_numu"][iz].mean(0)[ie]
            hv = nm[ihz].mean(0)[ih]
            print(f"  {E:5.1f}   {cz:5.2f}   {mv:9.3g}  {hv:9.3g}   {mv/hv:5.2f}")
    if args.plot:
        _plot(r, h)


def _plot(r, h):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e = r["e"]
    He, Hcz, nm = h["E"], h["czlo"], h["numu"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.4))
    # spectrum vertical: this work vs Honda
    iz = 0
    mv = r["flux"]["total_numu"][iz].mean(0)
    ihz = int(np.argmin(np.abs(Hcz - 0.9)))
    s = (e > 0.3) & (e < 1e3)
    axL.loglog(e[s], (mv * e**3)[s], "C3-", label="this work (MCEq+geomag)")
    sH = (He > 0.3) & (He < 1e3)
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
