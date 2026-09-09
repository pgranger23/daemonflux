"""Compare the directional engine to the Bartol (Barr et al. 2004) 3D flux tables.

Bartol/Oxford 3D atmospheric-neutrino fluxes (TARGET-2.1, ICRC01 primary, full 3D
with geomagnetic bending) at Kamioka, from astro-ph/0403630. The files give
dN/dlnE [/m^2/s/sr] on an (E, cos zenith) grid; we convert to dN/dE = (dN/dlnE)/E
and compare the vertical numu flux (and the flavour ratio) of this work and Honda
HKKM2014 against Bartol, over the overlap 0.1-10 GeV. Bartol solar-min and
solar-max bracket the solar cycle and are shown as a band.

This is a genuine inter-calculation comparison (different code, hadronic model,
primary flux, and 3D treatment), complementing the MCEq hadronic-model spread
(`hadronic_spread.py`). Run::

    python validate_bartol.py
"""

from __future__ import annotations

import glob
import os

import numpy as np

BDIR = "bartol/0403i"
SITE = "kam"
RC_KAMIOKA = 11.3  # GV, back-traced vertical cutoff


CACHE = "bartol_kam.npz"  # small committed cache of the Kamioka subset


def _parse_raw(species, phase="fmin", nbins="20"):
    pat = os.path.join(BDIR, f"{phase}{nbins}_*z.{SITE}_{species}")
    d = np.loadtxt(sorted(glob.glob(pat))[0], comments="#")
    E = np.unique(d[:, 0])
    cz = np.unique(d[:, 1])
    grid = np.full((len(cz), len(E)), np.nan)
    ei = {v: i for i, v in enumerate(E)}
    ci = {v: i for i, v in enumerate(cz)}
    for row in d:
        grid[ci[row[1]], ei[row[0]]] = row[2] / row[0]  # dN/dlnE -> dN/dE
    return E, cz, grid


def _build_cache():
    """Parse the raw Bartol 0403i Kamioka tables into a small npz (all species/phases)."""
    out = {}
    E = cz = None
    for sp in ("num", "nbm", "nue", "nbe"):
        for ph in ("fmin", "fmax"):
            E, cz, g = _parse_raw(sp, ph)
            out[f"{sp}_{ph}"] = g
    np.savez(CACHE, E=E, cz=cz, **out)
    return E, cz, out


def load_bartol(species, phase="fmin", nbins="20"):
    """Return (E[GeV], cosZ, dNdE[cosZ,E]); uses the committed npz cache if present."""
    if not os.path.exists(CACHE):
        _build_cache()
    d = np.load(CACHE)
    return d["E"], d["cz"], d[f"{species}_{phase}"]


def _at(y, x, X):
    return float(np.exp(np.interp(np.log(X), np.log(x), np.log(np.maximum(y, 1e-300)))))


def engine_vertical_numu():
    """This work (daemonflux base) vertical numu at Kamioka [/(m^2 s sr GeV)]."""
    from mceq3d_flux import MCEq3DFlux

    eng = MCEq3DFlux(base_model="daemonflux", daemonflux_location="kamioka")
    e = eng.e
    base = eng.base(np.array([0.95]))["total_numu"][0]
    rcg = np.linspace(0.1, 20.0, 24)
    G, rcg = eng.geomag_response(rcg)
    g = np.array(
        [np.interp(RC_KAMIOKA, rcg, G["total_numu"][:, k]) for k in range(len(e))]
    )
    return e, base * g


def main():
    EREPORT = (0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0)

    # Bartol: vertical (cosZ 0.95) numu, solar min & max
    Eb, czb, gmin = load_bartol("num", "fmin")
    _, _, gmax = load_bartol("num", "fmax")
    iv = int(np.argmin(np.abs(czb - 0.95)))
    b_min, b_max = gmin[iv], gmax[iv]

    # Honda vertical numu
    h = dict(np.load("honda_kam.npz"))
    He, Hcz, nm = h["E"], h["czlo"], h["numu"]
    ihz = int(np.argmin(np.abs(Hcz - 0.9)))
    hv = nm[ihz].mean(0)

    # This work vertical numu
    e, mv = engine_vertical_numu()

    print("Vertical numu at Kamioka [/(m^2 s sr GeV)] -- this work vs Honda vs Bartol:")
    print(
        "  E[GeV]  this work   Honda   Bartol(max-min)   this/Bartol  Honda/Bartol"
    )
    for E in EREPORT:
        tw = _at(mv, e, E)
        ho = _at(hv, He, E)
        bmn = _at(b_min, Eb, E)
        bmx = _at(b_max, Eb, E)
        print(
            f"  {E:6.2f}  {tw:9.1f}  {ho:8.1f}   {bmx:6.1f}-{bmn:<6.1f}   "
            f"{tw / bmn:6.2f}          {ho / bmn:6.2f}"
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))
    sb = (Eb >= 0.1) & (Eb <= 10)
    se = (e >= 0.1) & (e <= 10)
    sh = (He >= 0.1) & (He <= 10)
    axL.fill_between(
        Eb[sb],
        (b_max * Eb**3)[sb],
        (b_min * Eb**3)[sb],
        color="C2",
        alpha=0.3,
        label="Bartol 2004 (solar min–max)",
    )
    axL.loglog(He[sh], (hv * He**3)[sh], "k--", label="Honda HKKM2014")
    axL.loglog(e[se], (mv * e**3)[se], "C3-", lw=2, label="this work (daemonflux base)")
    axL.set_xlabel("E [GeV]")
    axL.set_ylabel(r"$E^3\,\Phi_{\nu_\mu}$ [GeV$^2$/(m$^2$ s sr)]")
    axL.set_title("Vertical numu at Kamioka")
    axL.legend(fontsize=8)

    bmid = np.sqrt(b_min * b_max)
    axR.semilogx(Eb[sb], np.ones(sb.sum()), "C2-", label="Bartol (mid)")
    axR.fill_between(
        Eb[sb], (b_max / bmid)[sb], (b_min / bmid)[sb], color="C2", alpha=0.3
    )
    axR.semilogx(
        He[sh],
        [_at(hv, He, x) / _at(bmid, Eb, x) for x in He[sh]],
        "k--",
        label="Honda / Bartol",
    )
    axR.semilogx(
        e[se],
        [_at(mv, e, x) / _at(bmid, Eb, x) for x in e[se]],
        "C3-",
        lw=2,
        label="this work / Bartol",
    )
    axR.axhline(1.0, color="0.5", lw=0.7)
    axR.set_xlabel("E [GeV]")
    axR.set_ylabel("ratio to Bartol (solar mid)")
    axR.set_title("Inter-calculation comparison")
    axR.set_ylim(0.5, 2.0)
    axR.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("validate_bartol.png", dpi=110)
    print("saved plot -> validate_bartol.png")


if __name__ == "__main__":
    main()
