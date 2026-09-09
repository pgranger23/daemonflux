"""Flux-level East-West asymmetry at Kamioka, in the Super-Kamiokande
observable, for (a) the Honda HKKM2014 table and (b) this work's engine.

Motivation
----------
Honda is a *model*, not data.  The residual E-W discrepancy between this
engine and Honda's tables (nue W/E at the horizon ~30% low, numu ~12-21%
high) can only be arbitrated by a measurement.  Super-Kamiokande has
published the atmospheric East-West asymmetry (Futagami et al., PRL 82
(1999) 5194; SK-I..IV updates), so this script computes the closest
*flux-level* analogue of that observable from both fluxes.

Observable
----------
SK bins events by the reconstructed **lepton** direction.  Their asymmetry

    A = (N_E - N_W) / (N_E + N_W)

counts events whose momentum *points* East / West, i.e. particles that
*arrived from* the West / East.  Both this engine and Honda tabulate flux
against the **arrival** direction, so at flux level

    A_flux = (Phi_from-W - Phi_from-E) / (Phi_from-W + Phi_from-E)

is the sign-matched quantity.  ``--arrival-sign`` flips it if wanted.

Cuts (as close as the flux level allows):
  * |cos z| < 0.5  -> Honda's 10 cosZ bins with centres +-0.05..+-0.45
  * East sector  = arrival azimuth in [45, 135) deg (compass, 0=N, 90=E)
    West sector  = arrival azimuth in [225, 315) deg
    i.e. exactly 3 of Honda's 12 30-deg bins per sector.
  * neutrino energy window standing in for the 400 < p_lep < 3000 MeV/c
    lepton-momentum cut (default 0.5-5 GeV; scanned).

Weights: interaction rate ~ Phi(E) * sigma_CC(E) * (nu / nubar split).
sigma_CC ~ E above ~1 GeV; nubar/nu cross-section ratio ``--rbar``
(default 0.45).  ``--wpow`` sets the sigma exponent (1 = linear, 0 = flat).

Stages
------
``solve``   run the delivered engine on the +-|cosZ|<0.5 grid x 12 compass
            azimuth bin centres, saving after every zenith band.
``report``  post-process the saved grid + honda_kam.npz into the tables.

Run from tools/mceq3d with PYTHONPATH=$PWD.  New file; touches nothing
tracked.
"""

from __future__ import annotations

import argparse
import importlib.util  # noqa: F401  (MCEq config touches importlib.util)
import os
import time
import warnings
from datetime import datetime

import numpy as np

warnings.filterwarnings("ignore")

LAT, LON = 36.43, 137.31
DATE = datetime(2020, 1, 1)
CACHE = ".cache3d"
SP = ("total_numu", "total_antinumu", "total_nue", "total_antinue")
HKEY = {"total_numu": "numu", "total_antinumu": "numubar",
        "total_nue": "nue", "total_antinue": "nuebar"}

# Honda's cosZ bin centres with |cosZ| < 0.5
CZ = tuple(np.round(np.concatenate([-np.arange(0.05, 0.5, 0.1)[::-1],
                                    np.arange(0.05, 0.5, 0.1)]), 2))
# compass azimuth bin centres, matching Honda's 12 x 30 deg bins after
# az_compass = (180 - az_Honda) % 360
AZ = tuple((np.arange(12) + 0.5) * 30.0)

# East/West sectors are centred on 90 / 270 deg (compass, arrival-from) with
# half-width HW.  Honda's 12 bins have compass centres 15,45,...,345, so the
# selections symmetric about the E-W axis are HW = 30 (2 bins), 60 (4 bins)
# and 90 (6 bins = the full East / West hemisphere split at the N-S meridian).
HW = 90.0


def honda_azimuth_to_compass(az_honda_deg):
    """Honda az (0=S, 90=E, 180=N, 270=W) -> compass (0=N, 90=E)."""
    return (180.0 - np.asarray(az_honda_deg, float)) % 360.0



# --------------------------------------------------------------------------
# optional oscillation weighting (up-going numu disappearance)
R_EARTH = 6371.0
H_PROD_KM = 20.0
DM2 = 2.5e-3          # eV^2, |Delta m^2_32|
S2T = 1.0             # sin^2(2 theta_23), maximal


def path_length_km(cz):
    """Neutrino path length from a production point 20 km up to the detector."""
    r = R_EARTH + H_PROD_KM
    return np.sqrt(r ** 2 - (R_EARTH * np.sqrt(1 - cz ** 2)) ** 2) - R_EARTH * cz


def numu_survival(cz, E):
    """Two-flavour vacuum P(numu->numu); (ncz, nE)."""
    L = path_length_km(np.asarray(cz, float))[:, None]
    x = 1.267 * DM2 * L / np.asarray(E, float)[None, :]
    return 1.0 - S2T * np.sin(x) ** 2


# --------------------------------------------------------------------------
def stage_solve(a):
    from mceq3d_flux import MCEq3DFlux

    eng = MCEq3DFlux(base_model="hybrid",
                     primary=("GlobalSplineFitBeta", None),
                     daemonflux_location="kamioka")
    cz_all = np.array([float(x) for x in a.cz.split(",")]) if a.cz else np.array(CZ)
    az = np.array(AZ)
    acc = {}
    e = None
    done = []
    for czv in cz_all:
        t = time.time()
        r = eng.solve(LAT, LON, np.array([czv]), az, use_cache=True,
                      cache_dir=CACHE, date=DATE, n_jobs=a.n_jobs)
        e = r["e"]
        for s in SP:
            acc.setdefault(s, []).append(r["flux"][s][0])
        done.append(czv)
        np.savez(a.out, e=e, cz=np.array(done), az=az,
                 **{s: np.array(acc[s]) for s in SP})
        print(f"cz={czv:+.2f} done in {time.time()-t:.0f}s "
              f"-> {a.out} ({len(done)}/{len(cz_all)})", flush=True)
    print("SOLVE_DONE", flush=True)


# --------------------------------------------------------------------------
def _sector_mask(az, centre, hw):
    d = (np.asarray(az, float) - centre + 180.0) % 360.0 - 180.0
    return np.abs(d) < hw


def _rate_weight(E, elo, ehi, wpow):
    """dN/dE weight = sigma(E) inside the window, 0 outside (trapz on lnE)."""
    w = np.where((E >= elo) & (E <= ehi), E ** wpow, 0.0)
    return w


def _integrate(flux, E, elo, ehi, wpow):
    """int Phi(E) sigma(E) dE over the window; flux is (..., nE)."""
    w = _rate_weight(E, elo, ehi, wpow)
    m = w > 0
    if m.sum() < 2:
        return np.zeros(flux.shape[:-1])
    return np.trapezoid(flux[..., m] * w[m], E[m], axis=-1)


def _asym(fW, fE):
    return (fW - fE) / (fW + fE)


def _bands(cz):
    cz = np.asarray(cz)
    return {"[-0.5,-0.2]": (cz >= -0.5) & (cz < -0.2),
            "[-0.2,+0.2]": (cz >= -0.2) & (cz < 0.2),
            "[+0.2,+0.5]": (cz >= 0.2) & (cz < 0.5),
            "all |cz|<0.5": np.ones_like(cz, bool)}


def _species(fluxes, E, az, cz, elo, ehi, wpow, hw=HW, band=None):
    """Per-species asymmetry A_s = (Phi_fromW - Phi_fromE)/(sum)."""
    mE = _sector_mask(az, 90.0, hw)
    mW = _sector_mask(az, 270.0, hw)
    mcz = np.ones(len(cz), bool) if band is None else band
    out = {}
    for sp in SP:
        r = _integrate(fluxes[sp], E, elo, ehi, wpow)[mcz]
        fE = r[:, mE].mean()
        fW = r[:, mW].mean()
        out[sp] = _asym(fW, fE)
    return out


def _collect(fluxes, E, az, cz, elo, ehi, wpow, rbar, band=None, osc=False,
             hw=HW):
    """Return dict flavour -> (A, W, Eint) for one energy window / band.

    ``fluxes``: dict species -> array (ncz, naz, nE), all on grid ``E``.
    Combines nu + nubar with the CC cross-section ratio ``rbar`` and
    averages over the cosZ bins in ``band`` (equal solid angle per bin).
    """
    mE = _sector_mask(az, 90.0, hw)
    mW = _sector_mask(az, 270.0, hw)
    mcz = np.ones(len(cz), bool) if band is None else band
    out = {}
    P = numu_survival(cz, E)[:, None, :] if osc else None
    for fl, (snu, sbar) in (("nue", ("total_nue", "total_antinue")),
                            ("numu", ("total_numu", "total_antinumu"))):
        fnu, fbar = fluxes[snu], fluxes[sbar]
        if osc and fl == "numu":
            fnu, fbar = fnu * P, fbar * P
        r = (_integrate(fnu, E, elo, ehi, wpow)
             + rbar * _integrate(fbar, E, elo, ehi, wpow))
        r = r[mcz]                                  # (ncz_sel, naz)
        fE = r[:, mE].mean(axis=1).mean()
        fW = r[:, mW].mean(axis=1).mean()
        out[fl] = (_asym(fW, fE), fW, fE)
    return out


def _honda_grid(h, cz, E):
    """Honda table -> dict species -> (ncz, naz, nE) on compass azimuths AZ
    and the requested cosZ bin centres, interpolated in lnE onto ``E``."""
    az_c = honda_azimuth_to_compass(h["azlo"].astype(float) + 15.0)
    order = np.argsort(az_c)
    out = {}
    for s in SP:
        tab = h[HKEY[s]]                            # (ncz, naz, nEh)
        rows = []
        for czv in cz:
            i = int(np.argmin(np.abs(h["czlo"] - round(czv - 0.05, 2))))
            r = tab[i][order]                       # (naz, nEh) compass-sorted
            rows.append(np.exp(np.array(
                [np.interp(np.log(E), np.log(h["E"]), np.log(np.maximum(x, 1e-300)))
                 for x in r])))
        out[s] = np.array(rows)
    return out, az_c[order]


def stage_report(a):
    """``--grid`` may be a comma-separated list of partial grids (e.g. the
    down-going and up-going runs); they are concatenated in cosZ."""
    parts = [np.load(g) for g in a.grid.split(",")]
    e_m = parts[0]["e"]
    az = parts[0]["az"]
    for d in parts[1:]:
        assert np.allclose(d["e"], e_m) and np.allclose(d["az"], az)
    cz = np.concatenate([d["cz"] for d in parts])
    o = np.argsort(cz)
    cz = cz[o]
    mine = {s: np.concatenate([d[s] for d in parts])[o] for s in SP}
    h = dict(np.load("honda_kam.npz"))

    # common energy grid: engine grid restricted to Honda's coverage
    ok = (e_m >= h["E"].min()) & (e_m <= h["E"].max())
    E = e_m[ok]
    mine = {s: mine[s][..., ok] for s in SP}
    hond, az_h = _honda_grid(h, cz, E)
    assert np.allclose(np.sort(az), np.sort(az_h)), (az, az_h)
    # reorder mine to az_h order
    idx = [int(np.argmin(np.abs(az - x))) for x in az_h]
    mine = {s: mine[s][:, idx] for s in SP}
    az = az_h

    sgn = -1.0 if a.arrival_sign else 1.0
    wins = [tuple(float(x) for x in w.split(":")) for w in a.windows.split(",")]

    print(f"# grid: cosZ centres {list(np.round(cz,2))}")
    print(f"# azimuth (compass, arrival-from) bin centres {list(az)}")
    print(f"# East/West sectors: centred on 90/270 deg, half-width {a.hw} deg")
    print(f"# weight sigma ~ E^{a.wpow}, sigma(nubar)/sigma(nu) = {a.rbar}")
    print(f"# A = (Phi_fromW - Phi_fromE)/(sum)  [= SK's (N_E-N_W)/(N_E+N_W),"
          f" lepton momentum direction]  sign factor {sgn:+.0f}")

    for (elo, ehi) in wins:
        print(f"\n=== E_nu in [{elo}, {ehi}] GeV, |cosZ|<0.5 (all bands) ===")
        print(f"{'flavour':8s} {'A_Honda':>9s} {'A_engine':>9s} "
              f"{'A_eng/A_Hon':>12s} {'(W/E)_Hon':>10s} {'(W/E)_eng':>10s}")
        H = _collect(hond, E, az, cz, elo, ehi, a.wpow, a.rbar, hw=a.hw)
        M = _collect(mine, E, az, cz, elo, ehi, a.wpow, a.rbar, hw=a.hw)
        for fl in ("nue", "numu"):
            ah, am = sgn * H[fl][0], sgn * M[fl][0]
            rh, rm = H[fl][1] / H[fl][2], M[fl][1] / M[fl][2]
            print(f"{fl:8s} {ah:9.4f} {am:9.4f} {am/ah:12.3f} "
                  f"{rh:10.4f} {rm:10.4f}")

        print("  -- per species (no nu/nubar mixing) --")
        Hs = _species(hond, E, az, cz, elo, ehi, a.wpow, hw=a.hw)
        Ms = _species(mine, E, az, cz, elo, ehi, a.wpow, hw=a.hw)
        print(f"  {'species':16s} {'A_Honda':>9s} {'A_engine':>9s}")
        for sp in ("total_nue", "total_antinue", "total_numu",
                   "total_antinumu"):
            print(f"  {sp:16s} {sgn*Hs[sp]:9.4f} {sgn*Ms[sp]:9.4f}")

        print(f"  -- zenith bands --")
        print(f"  {'band':14s} {'flavour':8s} {'A_Honda':>9s} {'A_engine':>9s}"
              f" {'ratio':>8s}")
        for name, m in _bands(cz).items():
            if name.startswith("all"):
                continue
            H = _collect(hond, E, az, cz, elo, ehi, a.wpow, a.rbar, band=m, hw=a.hw)
            M = _collect(mine, E, az, cz, elo, ehi, a.wpow, a.rbar, band=m, hw=a.hw)
            for fl in ("nue", "numu"):
                ah, am = sgn * H[fl][0], sgn * M[fl][0]
                print(f"  {name:14s} {fl:8s} {ah:9.4f} {am:9.4f} "
                      f"{am/ah:8.3f}")

    # full azimuth profile at the nominal window (for the dipole phase)
    elo, ehi = wins[0]
    print(f"\n=== azimuth profile, E in [{elo},{ehi}] GeV, |cosZ|<0.5, "
          f"normalised to the azimuth mean ===")
    print(f"{'az_compass':>10s} " + " ".join(f"{f'{k}_{w}':>12s}"
          for k in ("nue", "numu") for w in ("Hon", "eng")))
    prof = {}
    for tag, F in (("Hon", hond), ("eng", mine)):
        for fl, (snu, sbar) in (("nue", ("total_nue", "total_antinue")),
                                ("numu", ("total_numu", "total_antinumu"))):
            r = (_integrate(F[snu], E, elo, ehi, a.wpow)
                 + a.rbar * _integrate(F[sbar], E, elo, ehi, a.wpow))
            v = r.mean(axis=0)
            prof[(fl, tag)] = v / v.mean()
    for j, azv in enumerate(az):
        print(f"{azv:10.1f} " + " ".join(
            f"{prof[(k, w)][j]:12.4f}" for k in ("nue", "numu")
            for w in ("Hon", "eng")))

    # sensitivity to rbar / wpow
    print("\n=== sensitivity (nominal window) ===")
    for rbar in (0.0, 0.40, 0.45, 0.50, 1.0):
        H = _collect(hond, E, az, cz, elo, ehi, a.wpow, rbar, hw=a.hw)
        M = _collect(mine, E, az, cz, elo, ehi, a.wpow, rbar, hw=a.hw)
        print(f"  rbar={rbar:4.2f}  A_Hon(e,mu)=({sgn*H['nue'][0]:.4f},"
              f"{sgn*H['numu'][0]:.4f})  A_eng=({sgn*M['nue'][0]:.4f},"
              f"{sgn*M['numu'][0]:.4f})")
    for wp in (0.0, 0.5, 1.0):
        H = _collect(hond, E, az, cz, elo, ehi, wp, a.rbar, hw=a.hw)
        M = _collect(mine, E, az, cz, elo, ehi, wp, a.rbar, hw=a.hw)
        print(f"  wpow={wp:4.2f}  A_Hon(e,mu)=({sgn*H['nue'][0]:.4f},"
              f"{sgn*H['numu'][0]:.4f})  A_eng=({sgn*M['nue'][0]:.4f},"
              f"{sgn*M['numu'][0]:.4f})")

    print("\n=== with 2-flavour numu disappearance (dm2=2.5e-3, max mixing) ===")
    H = _collect(hond, E, az, cz, elo, ehi, a.wpow, a.rbar, osc=True, hw=a.hw)
    M = _collect(mine, E, az, cz, elo, ehi, a.wpow, a.rbar, osc=True, hw=a.hw)
    for fl in ("nue", "numu"):
        print(f"  {fl:6s} A_Hon={sgn*H[fl][0]:8.4f}  A_eng={sgn*M[fl][0]:8.4f}"
              f"  ratio={M[fl][0]/H[fl][0]:6.3f}")

    print("\n=== sector half-width scan (nominal window) ===")
    for hw in (30.0, 60.0, 90.0):
        H = _collect(hond, E, az, cz, elo, ehi, a.wpow, a.rbar, hw=hw)
        M = _collect(mine, E, az, cz, elo, ehi, a.wpow, a.rbar, hw=hw)
        print(f"  hw={hw:4.0f}  A_Hon(e,mu)=({sgn*H['nue'][0]:.4f},"
              f"{sgn*H['numu'][0]:.4f})  A_eng=({sgn*M['nue'][0]:.4f},"
              f"{sgn*M['numu'][0]:.4f})  ratio=({M['nue'][0]/H['nue'][0]:.3f},"
              f"{M['numu'][0]/H['numu'][0]:.3f})")

    # crude smearing test: dilute by the neutrino->lepton angle
    print("\n=== dilution by lepton-direction smearing (toy) ===")
    print("  A_obs ~ A_flux * <cos(dtheta)>; sub-GeV <dtheta> ~ 55-60 deg")
    for dil in (1.0, 0.7, 0.55, 0.5):
        H = _collect(hond, E, az, cz, elo, ehi, a.wpow, a.rbar, hw=a.hw)
        M = _collect(mine, E, az, cz, elo, ehi, a.wpow, a.rbar, hw=a.hw)
        print(f"  dil={dil:4.2f}  A_Hon(e,mu)=({dil*sgn*H['nue'][0]:.4f},"
              f"{dil*sgn*H['numu'][0]:.4f})  A_eng=("
              f"{dil*sgn*M['nue'][0]:.4f},{dil*sgn*M['numu'][0]:.4f})")


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="stage", required=True)
    s = sub.add_parser("solve")
    s.add_argument("--out", default="sk_ew_grid.npz")
    s.add_argument("--cz", default="")
    s.add_argument("--n-jobs", type=int, default=8)
    s.set_defaults(fn=stage_solve)
    r = sub.add_parser("report")
    r.add_argument("--grid", default="sk_ew_grid.npz")
    r.add_argument("--windows", default="0.5:5.0,0.4:3.0,0.6:10.0,0.3:1.5")
    r.add_argument("--rbar", type=float, default=0.45)
    r.add_argument("--wpow", type=float, default=1.0)
    r.add_argument("--arrival-sign", action="store_true")
    r.add_argument("--hw", type=float, default=HW)
    r.set_defaults(fn=stage_report)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
