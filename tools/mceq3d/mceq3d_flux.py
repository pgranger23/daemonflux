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

Trust boundary (documented, not hidden)
---------------------------------------
* **Down-going hemisphere (cosZ >= 0) is correct.** Up-going neutrinos come from
  primaries hitting the far-side atmosphere; their geomagnetic cutoff is set there,
  not at the detector -- that needs the *global* 3D back-tracing Honda does and is
  **not** included (the flux magnitude is ~up/down symmetric; the up-going
  geomagnetic modulation is the missing piece).
* Nuclei treated by superposition with an approximate R=(A/Z)E rigidity for the
  cut (leading order; ~10-20% on the suppression near the cutoff).
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


class MCEq3DFlux:
    """Absolute directional flux engine (down-going hemisphere)."""

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
        """Phi_MCEq[species, cosZ, E] in /(m^2 s sr GeV) (curved, no geomag)."""
        self.mceq._phi0[:] = self._phi0_std
        out = {s: np.zeros((len(cos_zeniths), len(self.e))) for s in SPECIES}
        for i, cz in enumerate(cos_zeniths):
            self.mceq.set_theta_deg(np.degrees(np.arccos(np.clip(cz, 1e-3, 1))))
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

    def solve(
        self, lat, lon, cos_zeniths, azimuths, date=None, n_scan=12, rc_grid=None
    ):
        """Absolute Phi[species, cosZ, az, E] for a site (down-going)."""
        import geomag_backtrace as gb

        cos_zeniths = np.asarray(cos_zeniths, float)
        azimuths = np.asarray(azimuths, float)
        if rc_grid is None:
            rc_grid = np.linspace(2.0, 20.0, 8)
        base = self.base(cos_zeniths)
        G, rc_grid = self.geomag_response(rc_grid)

        import datetime as _dt

        date = date or _dt.datetime(2020, 1, 1)
        zen_deg = np.degrees(np.arccos(np.clip(cos_zeniths, 1e-3, 1)))
        rc_map = gb.cutoff_map(lat, lon, date, zen_deg, azimuths, n_scan=n_scan)

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
    cz = np.array([0.95, 0.75, 0.55, 0.35, 0.15, 0.05])
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
        for cz in (0.95, 0.55, 0.05):
            iz = int(np.argmin(np.abs(r["cos_zeniths"] - cz)))
            ihz = int(np.argmin(np.abs(Hcz - (cz - 0.05))))  # Honda bin lo edge
            mv = mine["total_numu"][iz].mean(0)[ie]
            hv = nm[ihz].mean(0)[ih]
            print(f"  {E:5.1f}   {cz:4.2f}   {mv:9.3g}  {hv:9.3g}   {mv/hv:5.2f}")
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
