"""Shift-sensitive East-West observable, and the charge-dependent E-W test.

WHY A NEW OBSERVABLE
--------------------
Every earlier charge-split diagnostic (`diag_full_comparison.py` section E,
`diag_ew_charge_scale.py`, `coupled_ew_charge_diag.py`) quantified the E-W
asymmetry as ``max/min`` over a uniform azimuth grid.  That statistic is *blind
to the mechanism it was meant to test*: the charge-dependent muon-bending term is
a coherent **rotation** of the azimuthal pattern (mu+ and mu- decay neutrinos
sample the cutoff at primary directions shifted by +/- ~5 deg), and for a pattern
that is even about the magnetic meridian a rotation by ``+delta`` and by
``-delta`` give the *identical* set of grid values -- hence the identical max and
min.  Scaling the shift by 30x therefore "saturated" rather than growing.

THE OBSERVABLE USED HERE
------------------------
Fit / project the azimuthal flux onto its first two harmonics,

    Phi(phi) ~= a0 * [ 1 + (a1/a0) cos(phi - phi_1) + (a2/a0) cos(2(phi - phi_2)) ]

with ``phi`` the arrival azimuth (compass convention, 0 = geographic North,
increasing clockwise through East).  On a uniform N-point grid the projection is
the discrete Fourier sum; when the samples are *bin averages* of width ``Delta``
(Honda's 30-deg azimuth bins) the m-th coefficient is divided by
``sinc(m Delta/2)`` to undo the bin smearing (a +1.1% correction for m=1,
+4.7% for m=2; the *phase* is unaffected).

Reported per species / energy / zenith band:

* ``a1_rel = a1/a0``  -- the E-W dipole amplitude (relative), and
* ``phi1``            -- its phase, i.e. the azimuth of the flux MAXIMUM;
* ``dphi``            -- ``phi1`` minus the geomagnetic **West** azimuth,
  wrapped to (-180, 180].  **This is the shift-sensitive quantity**: a coherent
  rotation of the pattern by ``delta`` moves ``dphi`` by ``delta``, with sign.
* ``s1_rel = (a1/a0) sin(dphi)`` -- the component of the dipole *transverse* to
  the geomagnetic E-W axis.  It is odd in ``delta`` and linear for small
  ``delta``; ``c1_rel = (a1/a0) cos(dphi)`` is the even, along-axis part that
  ``max/min`` already sees.
* ``a2_rel``, ``phi2`` -- the quadrupole (the E-W pattern is not a pure cosine).
* ``f_E``, ``f_W``, ``WE = f_W/f_E`` at the *fixed geomagnetic* East and West
  azimuths, from the 2-harmonic reconstruction (well defined on both grids), and
* ``maxmin`` -- the legacy grid ``max/min``, for continuity with the paper.

The paper's "W/E amplitude" is the legacy ``max/min`` over Honda's 12 azimuth
bins at the ``cosZ in [0, 0.1]`` band (87 deg); "splitting" is
``amplitude(nu) - amplitude(nubar)``.  Both that definition and the new ones are
printed side by side so the tables are comparable.

WHAT IS COMPARED
----------------
Honda HKKM2014 ``kam-ally-20-12-solmin`` (``honda_kam.npz``, 20 cosZ x 12 az x
101 E, four species) against the delivered factorised engine evaluated on
*exactly* Honda's bins: cosZ bins [0.2,0.3] / [0.1,0.2] / [0.0,0.1]
(75 / 81 / 87 deg) and the 12 30-deg azimuth bins, at 0.3, 0.5, 1 and 2 GeV.
**Honda's azimuth is NOT our azimuth.**  Honda et al. (arXiv:1102.2688) measure
it *counterclockwise from south* (0 = S, 90 = E, 180 = N, 270 = W); we use the
compass convention (0 = N, 90 = E, 180 = S, 270 = W).  The two are mirror images
about the E-W axis, ``az_compass = (180 - az_Honda) mod 360``, so every
azimuth-symmetric statistic ever compared against Honda here (``max/min``,
azimuth averages) was unaffected -- but anything ODD about the E-W axis, which is
precisely the charge-dependent bending signal, flips sign without the mapping.
``--convention`` prints the back-traced cutoff map's own azimuth structure beside
Honda's for a sanity check.

Run (from tools/mceq3d, PYTHONPATH=$PWD)::

    python diag_ew_charge_fourier.py --honda-only     # no engine needed
    python diag_ew_charge_fourier.py                  # + delivered engine
    python diag_ew_charge_fourier.py --analytic       # task-3 analytic check
"""

from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np

LAT, LON = 36.43, 137.31
DATE = datetime(2020, 1, 1)
CACHE = ".cache3d"

HONDA = "honda_kam.npz"
HKEY = {"total_numu": "numu", "total_antinumu": "numubar",
        "total_nue": "nue", "total_antinue": "nuebar"}
SPECIES = ("total_numu", "total_antinumu", "total_nue", "total_antinue")
PAIRS = (("numu", "total_numu", "total_antinumu"),
         ("nue", "total_nue", "total_antinue"))
E_POINTS = (0.3, 0.5, 1.0, 2.0)
CZ_BANDS = (0.25, 0.15, 0.05)          # bin centres of Honda's 0.1-wide cosZ bins
ZEN_LABEL = {0.25: "75 deg", 0.15: "81 deg", 0.05: "87 deg"}


# --------------------------------------------------------------------------
# the observable
# --------------------------------------------------------------------------
def azimuth_harmonics(flux, az_deg, bin_width_deg=None, n_harm=2):
    """Harmonic decomposition of ``flux`` sampled on a uniform azimuth grid.

    Parameters
    ----------
    flux : (n_az,) array
        Flux at the azimuths ``az_deg`` (uniformly spaced, one full period).
    az_deg : (n_az,) array
        Azimuths [deg] of the samples (bin *centres* if the samples are bin
        averages).
    bin_width_deg : float, optional
        If the samples are bin averages over this width, undo the bin smearing:
        the m-th harmonic of a bin-averaged cosine is reduced by
        ``sinc(m*Delta/2) = sin(m Delta/2)/(m Delta/2)``.
    n_harm : int
        Number of harmonics to return (>=1).

    Returns
    -------
    dict with ``a0`` (mean), ``a[m]`` (amplitude) and ``phi[m]`` (phase [deg],
    the azimuth of the m-th harmonic's maximum, in [0, 360/m)) for m=1..n_harm,
    plus the raw cosine/sine projections ``c[m]``/``s[m]``.
    """
    f = np.asarray(flux, float)
    ph = np.radians(np.asarray(az_deg, float))
    n = f.size
    if n < 2 * n_harm + 1:
        raise ValueError(f"need >= {2*n_harm+1} azimuth samples, got {n}")
    out = {"a0": float(f.mean()), "a": {}, "phi": {}, "c": {}, "s": {}}
    for m in range(1, n_harm + 1):
        c = 2.0 / n * float(np.sum(f * np.cos(m * ph)))
        s = 2.0 / n * float(np.sum(f * np.sin(m * ph)))
        if bin_width_deg:
            half = 0.5 * m * np.radians(bin_width_deg)
            sinc = np.sin(half) / half
            c, s = c / sinc, s / sinc
        out["c"][m], out["s"][m] = c, s
        out["a"][m] = float(np.hypot(c, s))
        out["phi"][m] = float(np.degrees(np.arctan2(s, c)) % (360.0 / m))
    return out


def _wrap180(x):
    return (np.asarray(x, float) + 180.0) % 360.0 - 180.0


def harmonic_value(h, az_deg, n_harm=2):
    """Reconstruct the flux at ``az_deg`` from the harmonics ``h``."""
    ph = np.radians(np.asarray(az_deg, float))
    v = h["a0"] * np.ones_like(ph)
    for m in range(1, n_harm + 1):
        if m in h["a"]:
            v = v + h["a"][m] * np.cos(m * (ph - np.radians(h["phi"][m])))
    return v


def ew_observables(flux, az_deg, ew_axis_deg, bin_width_deg=None):
    """Shift-sensitive East-West observables about a geomagnetic E-W axis.

    ``ew_axis_deg`` is the azimuth of geomagnetic **East** (the direction of
    maximum rigidity cutoff for positive primaries); geomagnetic West is
    ``ew_axis_deg + 180``.  See the module docstring for the definitions.
    """
    h = azimuth_harmonics(flux, az_deg, bin_width_deg=bin_width_deg, n_harm=2)
    a0 = max(h["a0"], 1e-300)
    west = (ew_axis_deg + 180.0) % 360.0
    dphi = float(_wrap180(h["phi"][1] - west))
    a1_rel = h["a"][1] / a0
    f = np.asarray(flux, float)
    return {
        "a0": h["a0"],
        "a1_rel": float(a1_rel),
        "phi1": h["phi"][1],
        "dphi": dphi,                                     # shift-sensitive
        "c1_rel": float(a1_rel * np.cos(np.radians(dphi))),
        "s1_rel": float(a1_rel * np.sin(np.radians(dphi))),  # odd in the shift
        "a2_rel": float(h["a"][2] / a0),
        "phi2": h["phi"][2],
        "f_E": float(harmonic_value(h, ew_axis_deg)),
        "f_W": float(harmonic_value(h, west)),
        "WE": float(harmonic_value(h, west) / max(harmonic_value(h, ew_axis_deg),
                                                  1e-300)),
        "maxmin": float(f.max() / max(f.min(), 1e-300)),   # legacy (paper)
    }


def geomagnetic_ew_axis(lat=LAT, lon=LON, date=DATE, h_km=15.0):
    """Azimuth [deg, compass] of geomagnetic East = declination + 90.

    Positive primaries arriving from magnetic East see the highest Stormer
    cutoff, so this is the axis the E-W asymmetry is organised about.  At
    Kamioka the IGRF-13 declination is ~-8.1 deg, so magnetic East is at
    ~81.9 deg and magnetic West at ~261.9 deg (not 90/270).
    """
    from muon_bending import local_field_enu

    b = local_field_enu(lat, lon, date, h_km=h_km)
    decl = np.degrees(np.arctan2(b[0], b[1]))
    return float((90.0 + decl) % 360.0)


# --------------------------------------------------------------------------
# Honda
# --------------------------------------------------------------------------
def _log_at(y, x, X):
    return float(np.exp(np.interp(np.log(X), np.log(x),
                                  np.log(np.maximum(y, 1e-300)))))


def honda_table(path=HONDA):
    d = dict(np.load(path))
    return d


def honda_azimuth_to_compass(az_honda_deg):
    """Honda's azimuth -> our compass azimuth (0 = geographic North, clockwise).

    Honda et al. (arXiv:1102.2688, Sec. II): "the azimuth angle is measured
    **counterclockwise from south** in the local coordinate system", i.e.
    0 = South, 90 = East, 180 = North, 270 = West -- confirmed in the same paper
    by "the deficit of neutrino flux due to the rigidity cutoff from the East
    direction (phi ~ 90 deg)".  Our engine (``geomag_backtrace.arrival_direction``)
    uses the compass convention 0 = North, 90 = East, 180 = South, 270 = West.
    The two agree on East and West and are **mirror images** about the E-W axis:

        az_compass = (180 - az_Honda) mod 360.

    Every azimuth-symmetric comparison made previously (``max/min``, azimuth
    averages) is unaffected by this; anything ODD about the E-W axis -- which is
    exactly the charge-dependent bending signal -- flips sign without it.
    """
    return (180.0 - np.asarray(az_honda_deg, float)) % 360.0


def honda_ew(h, hkey, cz_centre, E, ew_axis_deg):
    """Honda's E-W observables for one species / cosZ bin / energy.

    Honda's rows are already bin averages over the 0.1-wide cosZ bin and the
    30-deg azimuth bin, so the azimuth bin-width deconvolution is applied; the
    azimuths are mapped to our compass convention first
    (:func:`honda_azimuth_to_compass`).
    """
    cz_lo = round(cz_centre - 0.05, 2)
    icz = int(np.argmin(np.abs(h["czlo"] - cz_lo)))
    az_c = honda_azimuth_to_compass(h["azlo"].astype(float) + 15.0)
    rows = h[hkey][icz]                              # (n_az, nE)
    f = np.array([_log_at(rows[j], h["E"], E) for j in range(rows.shape[0])])
    return ew_observables(f, az_c, ew_axis_deg, bin_width_deg=30.0)


# --------------------------------------------------------------------------
# the delivered engine
# --------------------------------------------------------------------------
LEGACY_MAP = ".cache3d/finerc_8e83b6248bd62681.npz"


def use_legacy_map(path=LEGACY_MAP):
    """Pin ``finemap_rc`` to a *specific cached* cutoff map (a context manager).

    Used to produce a like-for-like BASELINE against the map the paper's numbers
    were computed on (13+13 uniform zenith nodes, 24-point linear rigidity scan,
    vertical R_c = 8.79 GV, six near-horizon East cells saturated, 9.5 GV seam at
    the 89/90 deg stitch) while the working tree already carries the repaired
    map-builder.  Without this the two cannot be compared, because the cache key
    changes with the builder.
    """
    import contextlib

    from mceq3d_flux import MCEq3DFlux

    @contextlib.contextmanager
    def _ctx():
        d = np.load(path)
        out = (d["zen"], d["az"], d["rc"])
        orig = MCEq3DFlux.finemap_rc
        MCEq3DFlux.finemap_rc = lambda self, *a, **k: out
        try:
            yield out
        finally:
            MCEq3DFlux.finemap_rc = orig

    return _ctx()


def model_flux(muon_bending=True, cz_sub=1, cache_dir=CACHE, naz=12,
               engine=None, legacy_map=False, **solve_kw):
    """Delivered-engine flux on Honda's bins.

    Returns ``(e, {species: flux[cz_band, az, E]}, az_centres)``.  With
    ``cz_sub > 1`` each 0.1-wide cosZ bin is averaged over ``cz_sub`` equally
    spaced sub-points (Honda's rows are cosZ-bin averages); ``cz_sub=1`` uses the
    bin centre.
    """
    from mceq3d_flux import MCEq3DFlux

    eng = engine or MCEq3DFlux(base_model="hybrid",
                               primary=("GlobalSplineFitBeta", None),
                               daemonflux_location="kamioka")
    az = (np.arange(naz) + 0.5) * 360.0 / naz
    sub = (np.arange(cz_sub) + 0.5) / cz_sub - 0.5     # in units of the bin width
    cz = np.concatenate([np.asarray(CZ_BANDS)[:, None] + 0.1 * sub[None, :]]).ravel()
    import contextlib
    with (use_legacy_map() if legacy_map else contextlib.nullcontext()):
        r = eng.solve(LAT, LON, cz, az, use_cache=True, cache_dir=cache_dir,
                      date=DATE, muon_bending=muon_bending, **solve_kw)
    e = r["e"]
    out = {s: r["flux"][s].reshape(len(CZ_BANDS), cz_sub, naz, len(e)).mean(1)
           for s in SPECIES}
    return e, out, az


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
_HDR = (f"{'E[GeV]':>6} {'a1/a0':>7} {'dphi':>7} {'s1/a0':>8} {'c1/a0':>7} "
        f"{'a2/a0':>7} {'W/E':>6} {'max/min':>8}")


def _row(E, o):
    return (f"{E:6.2f} {o['a1_rel']:7.3f} {o['dphi']:+7.1f} {o['s1_rel']:+8.4f} "
            f"{o['c1_rel']:7.3f} {o['a2_rel']:7.3f} {o['WE']:6.2f} "
            f"{o['maxmin']:8.2f}")


def report_honda(ew_axis, h=None):
    h = h or honda_table()
    print("=" * 78)
    print("HONDA HKKM2014 (kam-ally-20-12-solmin), Fourier E-W observables")
    print(f"geomagnetic East axis = {ew_axis:.2f} deg (declination "
          f"{ew_axis-90:.2f} deg); dphi is measured from geomagnetic WEST")
    print("=" * 78)
    res = {}
    for cz in CZ_BANDS:
        print(f"\n--- cosZ bin [{cz-0.05:.1f},{cz+0.05:.1f}]  ({ZEN_LABEL[cz]}) ---")
        for sp in SPECIES:
            print(f"  [{HKEY[sp]}]  {_HDR}")
            for E in E_POINTS:
                o = honda_ew(h, HKEY[sp], cz, E, ew_axis)
                res[(sp, cz, E)] = o
                print(f"{'':>12}{_row(E, o)}")
    return res


def report_splitting(res_a, res_b, label_a, label_b):
    """nu-vs-nubar splitting of the three observables, model vs Honda."""
    print("\n" + "=" * 78)
    print(f"CHARGE SPLITTING  nu minus nubar:  {label_a}  vs  {label_b}")
    print("  d(max/min) is the PAPER's 'W/E amplitude' splitting;")
    print("  d(dphi) [deg] and d(s1/a0) are the new shift-sensitive ones.")
    print("=" * 78)
    for cz in CZ_BANDS:
        print(f"\n--- cosZ bin [{cz-0.05:.1f},{cz+0.05:.1f}]  ({ZEN_LABEL[cz]}) ---")
        print(f"{'flavour':>8} {'E':>5} | "
              f"{'d(max/min)':>21} | {'d(W/E)':>17} | {'d(dphi) deg':>17} | "
              f"{'d(s1/a0)':>19}")
        print(f"{'':>8} {'':>5} | {label_a:>10}{label_b:>11} | "
              f"{label_a:>8}{label_b:>9} | {label_a:>8}{label_b:>9} | "
              f"{label_a:>9}{label_b:>10}")
        for fl, sp_nu, sp_nb in PAIRS:
            for E in E_POINTS:
                cells = []
                for res in (res_a, res_b):
                    a, b = res.get((sp_nu, cz, E)), res.get((sp_nb, cz, E))
                    if a is None or b is None:
                        cells.append((np.nan,) * 4)
                        continue
                    cells.append((a["maxmin"] - b["maxmin"],
                                  a["WE"] - b["WE"],
                                  _wrap180(a["dphi"] - b["dphi"]),
                                  a["s1_rel"] - b["s1_rel"]))
                x, y = cells
                print(f"{fl:>8} {E:5.2f} | {x[0]:+10.2f}{y[0]:+11.2f} | "
                      f"{x[1]:+8.2f}{y[1]:+9.2f} | {x[2]:+8.2f}{y[2]:+9.2f} | "
                      f"{x[3]:+9.4f}{y[3]:+10.4f}")


def report_model(e, F, az, ew_axis, tag):
    print("\n" + "=" * 78)
    print(f"DELIVERED ENGINE ({tag}), same bins")
    print("=" * 78)
    res = {}
    for icz, cz in enumerate(CZ_BANDS):
        print(f"\n--- cosZ bin [{cz-0.05:.1f},{cz+0.05:.1f}]  ({ZEN_LABEL[cz]}) ---")
        for sp in SPECIES:
            print(f"  [{HKEY[sp]}]  {_HDR}")
            for E in E_POINTS:
                f = np.array([_log_at(F[sp][icz, j], e, E)
                              for j in range(F[sp].shape[1])])
                o = ew_observables(f, az, ew_axis)   # engine samples are pointwise
                res[(sp, cz, E)] = o
                print(f"{'':>12}{_row(E, o)}")
    return res


# --------------------------------------------------------------------------
# convention check
# --------------------------------------------------------------------------
def check_convention(ew_axis, n_scan=56, r_hi=55.0):
    """Fix Honda's azimuth convention empirically against our own back-trace.

    Our compass convention (0 = geographic North, increasing clockwise through
    East) is fixed by ``geomag_backtrace.arrival_direction``.  Honda's table only
    labels bins 0..330.  Comparing the azimuth of Honda's flux MINIMUM (and the
    N-vs-S ordering) with the azimuth of our cutoff MAXIMUM settles both the
    E-W orientation and the *sense* of rotation, which is what the phase
    observable needs.
    """
    import geomag_backtrace as gb

    az = np.arange(12) * 30.0 + 15.0
    zen = np.array([87.0, 81.0, 75.0])
    rc = gb.cutoff_map(LAT, LON, DATE, zen, az, n_scan=n_scan, r_hi=r_hi)
    h = honda_table()
    print("=" * 78)
    print("AZIMUTH CONVENTION CHECK (back-traced R_c vs Honda's flux shape)")
    print("=" * 78)
    for i, z in enumerate(zen):
        hrc = azimuth_harmonics(rc[i], az, n_harm=2)
        # cutoff is ANTI-correlated with flux: its maximum is geomagnetic East
        print(f"  zenith {z:.0f}: R_c max at az {hrc['phi'][1]:6.1f} deg "
              f"(geomag East = {ew_axis:.1f}); R_c = "
              + " ".join(f"{a:.0f}:{v:.0f}" for a, v in zip(az, rc[i])))
    for cz in CZ_BANDS:
        o = honda_ew(h, "numu", cz, 1.0, ew_axis)
        print(f"  Honda numu cosZ~{cz:.2f}: flux max at az {o['phi1']:6.1f} deg, "
              f"i.e. {o['dphi']:+.1f} deg from geomag West")
    print("\n  A flux maximum within a few deg of geomagnetic West (and a minimum")
    print("  at geomagnetic East, where R_c peaks) confirms Honda's bins share our")
    print("  compass convention (90 = East, 270 = West) AND its sense of rotation.")
    return rc


# --------------------------------------------------------------------------
# task-3 analytic check (independent of the delivered engine)
# --------------------------------------------------------------------------
def _rodrigues(v, k, ang):
    """Rotate ``v`` (..., 3) about unit axis ``k`` by ``ang`` (...,) [rad]."""
    v = np.atleast_2d(np.asarray(v, float))
    ang = np.atleast_1d(np.asarray(ang, float))[:, None]
    c, s = np.cos(ang), np.sin(ang)
    kv = np.cross(np.broadcast_to(k, v.shape), v)
    kd = (v @ np.asarray(k, float))[:, None]
    return v * c + kv * s + np.broadcast_to(k, v.shape) * kd * (1 - c)


def _rc_interp_factory(zen_f, az_f, rc_f):
    def rc_at(th_deg, ph_deg):
        th = np.clip(np.asarray(th_deg, float), zen_f[0], zen_f[-1])
        ph = np.asarray(ph_deg, float) % 360.0
        iz = np.clip(np.searchsorted(zen_f, th) - 1, 0, len(zen_f) - 2)
        ja = np.clip(np.searchsorted(az_f, ph) - 1, 0, len(az_f) - 2)
        tz = (th - zen_f[iz]) / (zen_f[iz + 1] - zen_f[iz])
        ta = (ph - az_f[ja]) / (az_f[ja + 1] - az_f[ja])
        return (rc_f[iz, ja] * (1 - tz) * (1 - ta)
                + rc_f[iz + 1, ja] * tz * (1 - ta)
                + rc_f[iz, ja + 1] * (1 - tz) * ta
                + rc_f[iz + 1, ja + 1] * tz * ta)
    return rc_at


def _dir_angles(P):
    """(zenith, azimuth) [deg] of unit vectors ``P`` in the **ENU** frame.

    ENU (east, north, up) is right-handed, so ``np.cross`` and Rodrigues are
    valid in it.  The (north, east, up) ordering used inside ``cone_geff`` is
    LEFT-handed -- a cross product taken in that ordering comes out with the
    wrong sign, which silently reverses the charge assignment.  ``cone_geff``
    avoids this by taking the cross product in ENU and permuting afterwards;
    everything here simply stays in ENU.
    """
    th = np.degrees(np.arccos(np.clip(P[..., 2], -1, 1)))
    ph = np.degrees(np.arctan2(P[..., 0], P[..., 1])) % 360.0   # atan2(E, N)
    return th, ph


def _check_first_order(n_enu, b_enu, delta, k, tol=2e-3):
    """Assert the exact Rodrigues rotation agrees to first order with the
    deflection vector the delivered engine actually applies.

    ``cone_geff`` uses ``primary = n + bending_deflection(v, B, +1)``; here the
    primary direction is ``R(B_hat, +Delta) n``.  The two must agree to
    O(Delta^2), which pins the SIGN of the charge assignment -- the single
    easiest thing to get wrong, because (north, east, up) is a left-handed
    ordering and a cross product taken in it comes out negated.
    """
    from muon_bending import bending_deflection

    d = bending_deflection(-n_enu, b_enu, charge=+1)        # ENU, rad
    exact = _rodrigues(n_enu, k, np.array([delta]))[0] - n_enu
    err = np.linalg.norm(exact - d) / max(np.linalg.norm(d), 1e-30)
    assert err < 0.05, (f"rotation/deflection mismatch {err:.3f}: "
                        f"exact {exact}, first order {d}")
    return err


def analytic_check(n_mc=20000, seed=0, e_nu=(0.5, 1.0), zeniths=(87.0, 81.0),
                   cache_dir=CACHE, species="total_numu", engine=None):
    """mu+ vs mu- effective cutoff with the FULL exponential bend distribution.

    ``cone_geff`` shifts the muon-decay cone axis by the single *mean* bending
    vector (magnitude ``Delta = q B tau / m`` = 5.08 deg at Kamioka).  The
    physical bend accumulated before decay is however exponentially distributed:
    a muon decaying at lab time ``t`` has rotated about **B** by
    ``psi = Delta * (t / <t>)`` with ``t/<t> ~ Exp(1)``, so 5% of the muons have
    bent by more than 3x the mean (>15 deg) -- and the cutoff rises steeply and
    convexly with zenith near the limb.  ``<G_s(R_c(psi))>`` therefore need not
    equal ``G_s(R_c(<psi>))``, which is the approximation the delivered engine
    makes.

    Everything here comes from ``muon_bending.bending_angle`` + the cached cutoff
    map + the engine's own cascade-correct ``G_s`` response; no cone, no solve.
    """
    from mceq3d_flux import MCEq3DFlux, _interp_rc, RC_MAX_GV
    from muon_bending import bending_angle, local_field_enu, muon_decay_numu_fraction

    eng = engine or MCEq3DFlux(base_model="hybrid",
                               primary=("GlobalSplineFitBeta", None),
                               daemonflux_location="kamioka")
    zen_f, az_f, rc_f = eng.finemap_rc(LAT, LON, DATE, cache_dir=cache_dir)
    rc_at = _rc_interp_factory(zen_f, az_f, rc_f)
    b_enu = local_field_enu(LAT, LON, DATE)
    Bmag = float(np.linalg.norm(b_enu))
    k = b_enu / Bmag                                   # rotation axis, ENU
    delta = bending_angle(Bmag)                        # rad, mean rotation about B
    ew_axis = geomagnetic_ew_axis()
    rng = np.random.default_rng(seed)
    u = rng.exponential(1.0, n_mc)

    rc_grid = np.linspace(0.1, RC_MAX_GV, 40)
    print("=" * 78)
    print("ANALYTIC CHECK -- exponential bend distribution vs the single mean shift")
    print(f"  |B| = {Bmag:.4f} G, mean rotation about B  Delta = "
          f"{np.degrees(delta):.2f} deg (energy independent)")
    print(f"  geomagnetic East az = {ew_axis:.2f} deg; cutoff map "
          f"{len(zen_f)}x{len(az_f)}, vertical R_c = {rc_f[0].mean():.2f} GV, "
          f"max {rc_f.max():.1f} GV")
    print(f"  species response: {species};  {n_mc} exponential samples")
    print("=" * 78)

    for zdeg in zeniths:
        cz = np.cos(np.radians(zdeg))
        # snap cz_ref to the nearest Honda band centre so this shares the
        # engine's own cached G_s (otherwise one 40-solve cascade sweep per
        # zenith, ~10 min on a loaded machine, for a <1% difference in G_s)
        czr = min(CZ_BANDS, key=lambda b: abs(b - abs(cz)))
        G, _ = eng.geomag_response(rc_grid, cz_ref=czr, cache_dir=cache_dir)
        gmat = G[species]                              # (n_rc, n_e)
        gcurve = {E: np.array([_interp_rc(r, rc_grid, gmat)[
            int(np.argmin(np.abs(eng.e - E)))] for r in rc_grid]) for E in e_nu}

        def g_of_rc(rcv, E):
            return np.interp(rcv, rc_grid, gcurve[E])

        def rc_of_g(gv, E):
            c = gcurve[E]
            o = np.argsort(c)
            return float(np.interp(gv, c[o], rc_grid[o]))

        print(f"\n--- zenith {zdeg:.0f} deg (cosZ = {cz:.3f}) ---")
        for name, azd in (("East", ew_axis), ("West", (ew_axis + 180.0) % 360.0)):
            th = np.radians(zdeg)
            phi = np.radians(azd)
            n = np.array([np.sin(th) * np.sin(phi), np.sin(th) * np.cos(phi),
                          np.cos(th)])                  # arrival dir, ENU
            rc_axis = float(np.ravel(rc_at(*_dir_angles(n[None, :])))[0])
            _check_first_order(n, b_enu, delta, k)
            print(f"  [{name}] axis R_c = {rc_axis:6.2f} GV")
            samples, means = {}, {}
            for q, sgn in (("mu+", +1.0), ("mu-", -1.0)):
                n_mean = _rodrigues(n, k, np.array([sgn * delta]))[0]
                n_mean /= np.linalg.norm(n_mean)
                means[q] = float(np.ravel(
                    rc_at(*_dir_angles(n_mean[None, :])))[0])
                P = _rodrigues(np.broadcast_to(n, (n_mc, 3)), k, sgn * delta * u)
                P /= np.linalg.norm(P, axis=1)[:, None]
                samples[q] = np.asarray(rc_at(*_dir_angles(P)))
                th_s, _ = _dir_angles(P)
                print(f"    {q}: mean-shift R_c = {means[q]:6.2f} GV   "
                      f"<R_c>_exp = {samples[q].mean():6.2f} GV   "
                      f"(zenith {np.degrees(np.arccos(np.clip(n_mean[2],-1,1))):.2f} "
                      f"deg mean-shift, {th_s.mean():.2f} deg exp-mean)")
            for E in e_nu:
                gm = {q: float(g_of_rc(means[q], E)) for q in means}
                ge = {q: float(g_of_rc(samples[q], E).mean()) for q in samples}
                g0 = float(g_of_rc(rc_axis, E))
                w = float(muon_decay_numu_fraction(E, zenith_deg=zdeg))
                print(f"    E_nu={E:4.2f} GeV  G(axis)={g0:.4g}")
                print(f"      single mean shift : G(mu+)={gm['mu+']:.4g} "
                      f"G(mu-)={gm['mu-']:.4g}  ratio(mu-/mu+)="
                      f"{gm['mu-']/max(gm['mu+'],1e-300):7.3f}  "
                      f"R_eff(mu+)={rc_of_g(gm['mu+'],E):5.2f} "
                      f"R_eff(mu-)={rc_of_g(gm['mu-'],E):5.2f} GV")
                print(f"      exponential bend  : G(mu+)={ge['mu+']:.4g} "
                      f"G(mu-)={ge['mu-']:.4g}  ratio(mu-/mu+)="
                      f"{ge['mu-']/max(ge['mu+'],1e-300):7.3f}  "
                      f"R_eff(mu+)={rc_of_g(ge['mu+'],E):5.2f} "
                      f"R_eff(mu-)={rc_of_g(ge['mu-'],E):5.2f} GV")
                # what survives into the delivered species flux: the mu-decay
                # channel carries weight f_mu, the rest is the (charge-blind)
                # direct pion channel.
                for tag, gg in (("mean", gm), ("exp", ge)):
                    r = ((1 - w) * g0 + w * gg["mu-"]) / \
                        max((1 - w) * g0 + w * gg["mu+"], 1e-300)
                    print(f"      f_mu-blended ({tag:4s}) f_mu={w:.3f} -> "
                          f"flux ratio (from mu-)/(from mu+) = {r:7.4f}")
    print("\nANALYTIC_CHECK_DONE")


# --------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--honda-only", action="store_true")
    p.add_argument("--convention", action="store_true",
                   help="verify Honda's azimuth convention (slow back-trace)")
    p.add_argument("--analytic", action="store_true")
    p.add_argument("--no-bending", action="store_true",
                   help="also run the engine with muon_bending=False")
    p.add_argument("--cz-sub", type=int, default=1,
                   help="sub-points per cosZ bin for the model (Honda's rows are "
                        "bin averages)")
    p.add_argument("--cache-dir", default=CACHE)
    p.add_argument("--legacy-map", action="store_true",
                   help="pin the cutoff map to the cached pre-repair one "
                        "(the paper's baseline)")
    args = p.parse_args(argv)

    ew_axis = geomagnetic_ew_axis()
    if args.convention:
        check_convention(ew_axis)
    h_res = report_honda(ew_axis)
    if args.honda_only:
        print("\nDIAG_EW_CHARGE_FOURIER_DONE")
        return
    if args.analytic:
        import contextlib
        with (use_legacy_map() if args.legacy_map else contextlib.nullcontext()):
            analytic_check(cache_dir=args.cache_dir)

    tag = "legacy map" if args.legacy_map else "current map"
    e, F, az = model_flux(muon_bending=True, cz_sub=args.cz_sub,
                          cache_dir=args.cache_dir, legacy_map=args.legacy_map)
    m_res = report_model(e, F, az, ew_axis, f"muon_bending=ON, {tag}")
    report_splitting(m_res, h_res, "ours", "Honda")
    if args.no_bending:
        e0, F0, az0 = model_flux(muon_bending=False, cz_sub=args.cz_sub,
                                 cache_dir=args.cache_dir,
                                 legacy_map=args.legacy_map)
        m0 = report_model(e0, F0, az0, ew_axis, f"muon_bending=OFF, {tag}")
        report_splitting(m0, h_res, "ours-nb", "Honda")
    print("\nDIAG_EW_CHARGE_FOURIER_DONE")


if __name__ == "__main__":
    main()
