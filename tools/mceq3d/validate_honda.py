"""Quantitative cross-check against the Honda (HKKM2014) 3D flux tables.

This validates the 3D layer against the authoritative 3D atmospheric-neutrino
calculation: M. Honda et al., *Atmospheric neutrino flux calculation using the
NRLMSISE-00 atmospheric model*, Phys. Rev. D 92, 023004 (2014). The
azimuth-dependent Kamioka table (``kam-ally-20-12-solmin``) gives the full 3D
flux ``Phi(E, cos zenith, azimuth)`` on a 20 x 12 (cosZ x az) grid.

Two directional observables are compared -- both are *ratios*, so they test the
directional physics independently of absolute normalization and of Honda's
particular hadronic/atmosphere choices:

* **East-West amplitude** ``max/min`` over azimuth, near the horizon, vs energy
  -- the geomagnetic rigidity-cutoff signature. Compared to
  :func:`directional_flux.solve_directional` (back-traced full-IGRF cutoff).
* **sec(theta) horizon enhancement** ``Phi(horizon)/Phi(vertical)``
  (azimuth-averaged) vs energy -- the curved-cascade structure. Compared to
  :func:`spherical_cascade.solve`.

Findings (Kamioka, numu):
* E-W at 1 GeV: Honda ~2.1, this work ~2.3 (~5-10%); both peak sub-GeV
  (Honda ~2.5 at 0.5 GeV) and vanish above ~10 GeV -- the correct rigidity
  behaviour.
* sec(theta): both finite and growing with energy, agreeing best at 10-100 GeV
  (Honda ~2.2 vs this work ~2.3 at 100 GeV); the toy cascade somewhat overshoots
  the TeV saturation.

The parsed table is cached in ``honda_kam.npz`` (committed) so this runs offline;
pass ``--refresh`` to re-download from the Honda site.

Run::

    python validate_honda.py --plot
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np

CACHE = "honda_kam.npz"
URL = (
    "http://www-rccn.icrr.u-tokyo.ac.jp/mhonda/public/nflx2014/"
    "kam-ally-20-12-solmin.d.gz"
)


def fetch_honda_cache(path=CACHE, refresh=False):
    """Return parsed Honda Kamioka table: dict(E, czlo, azlo, numu[cosZ,az,E])."""
    if os.path.exists(path) and not refresh:
        d = np.load(path)
        return {k: d[k] for k in d}

    import gzip
    import urllib.request

    raw = gzip.decompress(urllib.request.urlopen(URL, timeout=60).read()).decode()
    blocks, cz, az, rows = {}, None, None, []

    def flush():
        if cz is not None and rows:
            blocks[(cz, az)] = np.array(rows)

    for ln in raw.splitlines():
        m = re.match(
            r"average flux in \[cosZ =\s*([\-0-9.]+) --.*phi_Az =\s*([0-9]+) --", ln
        )
        if m:
            flush()
            cz, az, rows = float(m.group(1)), int(m.group(2)), []
            continue
        p = ln.split()
        if len(p) == 5 and p[0][0].isdigit():
            rows.append([float(x) for x in p])
    flush()
    czs = sorted({k[0] for k in blocks})
    azs = sorted({k[1] for k in blocks})
    E = blocks[(czs[0], azs[0])][:, 0]
    numu = np.array([[blocks[(c, a)][:, 1] for a in azs] for c in czs])
    np.savez(path, E=E, czlo=np.array(czs), azlo=np.array(azs), numu=numu)
    return dict(E=E, czlo=np.array(czs), azlo=np.array(azs), numu=numu)


def honda_observables(h, horizon_cz=0.0):
    """Honda E-W amplitude (near horizon) and sec(theta) ratio, both vs E."""
    E, cz, numu = h["E"], h["czlo"], h["numu"]
    ic = int(np.argmin(np.abs(cz - horizon_cz)))
    ew = numu[ic].max(axis=0) / numu[ic].min(axis=0)  # max/min over azimuth
    azavg = numu.mean(axis=1)  # [cosZ, E]
    ih, iv = int(np.argmin(np.abs(cz - 0.05))), int(np.argmin(np.abs(cz - 0.95)))
    sec = azavg[ih] / azavg[iv]
    return E, ew, sec


def _engine():
    """The delivered production engine (recommended config: hybrid base + GSF
    primary), memoised so E-W and sec-theta reuse one instance/cache."""
    from mceq3d_flux import MCEq3DFlux

    if not hasattr(_engine, "_e"):
        _engine._e = MCEq3DFlux(
            base_model="hybrid", primary=("GlobalSplineFitBeta", None),
            daemonflux_location="kamioka",
        )
    return _engine._e


CACHE_DIR = ".cache3d"  # persists the (one-time) geomagnetic cone map per site


def my_east_west(energies, lat=36.43, lon=137.31, zenith=75.0):
    """This work's West/East nu_mu ratio vs energy, from the delivered engine.

    Uses the production-cone-averaged geomagnetic cutoff (``cone_cutoff=True``):
    the single detector cutoff gives the correct *contrast* but a maximal, sharp
    East-West (single sight-line), while the parent cosmic rays arrive over a
    production cone; averaging the cutoff over that cone reproduces Honda's
    cone-smeared amplitude (4-9% vs HKKM2014 over 0.5-2 GeV)."""
    eng = _engine()
    cz = np.array([np.cos(np.radians(zenith))])
    r = eng.solve(lat, lon, cz, np.array([90.0, 270.0]), offaxis=True,
                  use_cache=True, cone_cutoff=True,
                  cache_dir=CACHE_DIR)  # az: 90=E, 270=W
    e = r["e"]
    we = r["flux"]["total_numu"][0, 1] / r["flux"]["total_numu"][0, 0]  # W/E
    return np.interp(energies, e, we)


def my_sec_theta(energies, lat=36.43, lon=137.31):
    """This work's horizon/vertical nu_mu ratio vs energy, from the delivered
    engine (azimuth-averaged, WITH the off-axis 3D factor)."""
    eng = _engine()
    az = np.array([0.0, 90.0, 180.0, 270.0])
    # cone_cutoff is unnecessary here: the horizon/vertical ratio is
    # azimuth-averaged, and cone smearing only redistributes flux in azimuth
    # (it leaves the azimuth mean, hence sec-theta, essentially unchanged).
    r = eng.solve(lat, lon, np.array([0.95, 0.05]), az, offaxis=True,
                  use_cache=True, cache_dir=CACHE_DIR)
    e = r["e"]
    f = r["flux"]["total_numu"].mean(1)  # azimuth-average -> (cz, E)
    return np.interp(energies, e, f[1] / f[0])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--refresh", action="store_true", help="re-download the table")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args(argv)

    h = fetch_honda_cache(refresh=args.refresh)
    E, ew_h, sec_h = honda_observables(h)

    eg = np.geomspace(0.1, 30.0, 20)
    ew_mine = my_east_west(eg)
    print("EAST-WEST amplitude (numu, near horizon) -- Honda vs this work:")
    print("  E[GeV]   Honda max/min   this work W/E")
    for e, m in zip(eg, ew_mine):
        print(f"  {e:6.1f}      {np.interp(e, E, ew_h):6.2f}        {m:6.2f}")

    eg2 = np.geomspace(0.3, 100.0, 18)
    sec_mine = my_sec_theta(eg2)
    print("\nsec(theta) horizon/vertical (azimuth-avg) -- Honda vs this work:")
    print("  E[GeV]   Honda   this work")
    for e, m in zip(eg2, sec_mine):
        print(f"  {e:7.1f}   {np.interp(e, E, sec_h):5.2f}    {m:5.2f}")

    if args.plot:
        _plot(E, ew_h, sec_h, eg, ew_mine, eg2, sec_mine)


def _plot(E, ew_h, sec_h, eg, ew_mine, eg2, sec_mine):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.4))
    s = (E >= 0.3) & (E <= 30)
    axL.semilogx(E[s], ew_h[s], "k-", lw=2, label="Honda HKKM2014 (max/min az)")
    axL.semilogx(eg, ew_mine, "C3o", ms=7,
                 label="this work (cone-averaged cutoff)")
    axL.axhline(1, color="gray", ls=":", lw=0.7)
    axL.axvspan(0.3, 2.0, color="orange", alpha=0.12)
    axL.set_xlabel("E [GeV]")
    axL.set_ylabel("East-West amplitude")
    axL.set_title("Geomagnetic East-West (Kamioka, near horizon)")
    axL.legend()

    s2 = (E >= 1) & (E <= 1e3)
    axR.loglog(E[s2], sec_h[s2], "k-", lw=2, label="Honda HKKM2014")
    axR.loglog(eg2, sec_mine, "C0s", ms=7, label="this work (delivered engine)")
    axR.axhline(1, color="gray", ls=":", lw=0.7)
    axR.set_xlabel("E [GeV]")
    axR.set_ylabel(r"$\Phi_{\rm horizon}/\Phi_{\rm vertical}$")
    axR.set_title(r"sec$\theta$ horizon enhancement (azimuth-avg)")
    axR.legend()
    fig.tight_layout()
    fig.savefig("validate_honda.png", dpi=110)
    print("\nsaved plot -> validate_honda.png")


if __name__ == "__main__":
    main()
