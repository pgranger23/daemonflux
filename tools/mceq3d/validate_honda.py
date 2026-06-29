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


def my_east_west(energies, lat=36.43, lon=137.31, zenith=75.0, n_scan=12):
    """This work's West/East flux ratio vs energy (directional_flux)."""
    import directional_flux as df

    r = df.solve_directional(
        lat, lon, np.array([zenith]), np.array([90.0, 270.0]), n_scan=n_scan
    )
    e, we = r["e"], r["flux"][0, 1] / r["flux"][0, 0]
    return np.interp(energies, e, we)


def my_sec_theta(energies):
    """This work's horizon/vertical ratio vs energy (spherical_cascade)."""
    from spherical_cascade import solve as scs

    r = scs(np.array([0.95, 0.05]), n_e=60, nsteps=3000)
    e, ratio = r["e"], r["flux"][1] / r["flux"][0]
    return np.interp(energies, e, ratio)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--refresh", action="store_true", help="re-download the table")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args(argv)

    h = fetch_honda_cache(refresh=args.refresh)
    E, ew_h, sec_h = honda_observables(h)

    eg = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    ew_mine = my_east_west(eg)
    print("EAST-WEST amplitude (numu, near horizon) -- Honda vs this work:")
    print("  E[GeV]   Honda max/min   this work W/E")
    for e, m in zip(eg, ew_mine):
        print(f"  {e:6.1f}      {np.interp(e, E, ew_h):6.2f}        {m:6.2f}")

    eg2 = np.array([1.0, 10.0, 100.0, 1000.0])
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
    axL.semilogx(eg, ew_mine, "C3o", ms=7, label="this work (directional_flux W/E)")
    axL.axhline(1, color="gray", ls=":", lw=0.7)
    axL.axvspan(0.3, 2.0, color="orange", alpha=0.12)
    axL.set_xlabel("E [GeV]")
    axL.set_ylabel("East-West amplitude")
    axL.set_title("Geomagnetic East-West (Kamioka, near horizon)")
    axL.legend()

    s2 = (E >= 1) & (E <= 1e3)
    axR.loglog(E[s2], sec_h[s2], "k-", lw=2, label="Honda HKKM2014")
    axR.loglog(eg2, sec_mine, "C0s", ms=7, label="this work (spherical_cascade)")
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
