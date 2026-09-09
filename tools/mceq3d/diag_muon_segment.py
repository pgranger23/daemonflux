"""Deliverables of the muon-segment study: bending-angle tables, the implied
change in the muon-channel ``<G>``, and the implied change in W/E vs Honda.

Sections
--------
A  handedness / sign sanity + the Lipari cross-check.
B  bending-angle tables per charge, species, energy and direction:
   mean / RMS / 16-50-84 percentiles of the total bend and of its zenith and
   azimuth components, ``<T/tau>``, the mean momentum-loss factor ``p_i/p_f``,
   and the effective sample size.  Two variants:
     ``ray``  -- the production slant depth is read along the *arrival ray*, the
                 assumption the delivered engine makes when it evaluates the
                 muon channel's production on the unshifted cone direction
                 (``joint_cone.delivered_joint_factor``);  this is the
                 like-for-like replacement of the single ``+-Delta`` shift;
     ``muon`` -- the production slant depth is read along the muon's own
                 direction (the physical parent shower axis), which couples the
                 bend to how much atmosphere the primary had crossed.
C  forward companion: decay-in-flight vs ground vs range-out fractions.
D  the cutoff response: ``<G_s>`` for the muon-decay channel at the axis, at the
   delivered single mean shift, under the exponential-in-lifetime model and
   under the MC offset distribution; then the f_mu-blended flux and the implied
   W/E, against the current delivered values and Honda.

Run (from tools/mceq3d, PYTHONPATH=$PWD)::

    python diag_muon_segment.py                 # everything
    python diag_muon_segment.py --sections AB   # tables only, no engine
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import numpy as np

import muon_segment_mc as ms

LAT, LON = ms.LAT, ms.LON
DATE = datetime(2020, 1, 1)
CACHE = ".cache3d"
PAPER_MAP = ".cache3d/finerc_4b7188ee3e51511f.npz"

SPECIES = ("total_numu", "total_antinumu", "total_nue", "total_antinue")
MU_PLUS = ("total_antinumu", "total_nue")          # daughters of mu+
SP_LABEL = {"total_numu": "numu", "total_antinumu": "antinumu",
            "total_nue": "nue", "total_antinue": "antinue"}
MICHEL = {"total_numu": "numu", "total_antinumu": "numu",
          "total_nue": "nue", "total_antinue": "nue"}
CZ_BANDS = (0.25, 0.15, 0.05)
ZENITHS = (75.0, 81.0, 87.0)
E_POINTS = (0.3, 0.5, 1.0, 2.0)

# the delivered engine's W/E (max/min over Honda's 12 azimuth bins) and Honda's,
# from the current working tree -- scratchpad/fourier_newmap.log.
DELIVERED_WE = {   # (zenith, flavour, E) -> (ours, honda), max/min over the 12 bins
    (87.0, "numu", 0.5): (2.93, 2.51), (87.0, "numu", 1.0): (2.50, 2.09),
    (87.0, "antinumu", 0.5): (3.01, 3.81), (87.0, "antinumu", 1.0): (2.55, 3.62),
    (87.0, "nue", 0.5): (3.10, 4.74), (87.0, "nue", 1.0): (2.70, 4.90),
    (87.0, "antinue", 0.5): (2.67, 2.12), (87.0, "antinue", 1.0): (2.32, 1.63),
    (81.0, "numu", 0.5): (2.80, 2.44), (81.0, "numu", 1.0): (2.35, 2.02),
    (81.0, "antinumu", 0.5): (2.89, 3.50), (81.0, "antinumu", 1.0): (2.46, 2.98),
    (81.0, "nue", 0.5): (3.03, 4.30), (81.0, "nue", 1.0): (2.68, 3.89),
    (81.0, "antinue", 0.5): (2.62, 1.94), (81.0, "antinue", 1.0): (2.23, 1.65),
    (75.0, "numu", 0.5): (2.57, 2.28), (75.0, "numu", 1.0): (2.16, 1.92),
    (75.0, "antinumu", 0.5): (2.66, 3.02), (75.0, "antinumu", 1.0): (2.25, 2.45),
    (75.0, "nue", 0.5): (2.78, 3.49), (75.0, "nue", 1.0): (2.44, 2.99),
    (75.0, "antinue", 0.5): (2.40, 1.94), (75.0, "antinue", 1.0): (2.04, 1.67),
}


def hr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def ew_axis():
    """Compass azimuth of geomagnetic East = IGRF declination + 90 deg."""
    from muon_bending import local_field_enu

    b = local_field_enu(LAT, LON, DATE, h_km=15.0)
    return float((90.0 + np.degrees(np.arctan2(b[0], b[1]))) % 360.0)


# ---------------------------------------------------------------------------
def section_a():
    hr("A. sign / handedness / geometry sanity")
    err, delta, b = ms.check_handedness(87.0, ew_axis(), +1)
    print(f"  exact Rodrigues vs muon_bending.bending_deflection: rel. err "
          f"{err:.4f}  (must be O(Delta^2) = {0.5*np.radians(delta):.3f})")
    print(f"  Delta = q B tau / m = {delta:.2f} deg;  B_ENU = {np.round(b, 4)} G")
    e, n, u = np.eye(3)
    print(f"  ENU is right-handed: cross(E,N) = {np.cross(e, n)} = up;  the "
          f"(N,E,U) ordering used in cone_geff gives {np.cross(n, e)}")
    print(f"  geomagnetic East axis = {ew_axis():.2f} deg (compass)")


# ---------------------------------------------------------------------------
def _mc_grid(n, seed, prod_dir, energy_loss=True, polarisation=0.0,
             zeniths=ZENITHS, energies=E_POINTS, cache=None):
    """``{(zen, 'E'|'W', E_nu, charge): segment_mc(...)}``, memoised to ``cache``.

    Only the fields the tables and the cutoff lookup need are cached, so a
    re-run of section D costs seconds instead of ten minutes.
    """
    keep = ("w_numu", "w_nue", "bend_deg", "dzen", "daz", "n_prim", "T_p",
            "ratio", "e_dec", "e_prod")
    if cache is not None and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        out = {}
        for k in d["keys"]:
            key = (float(k[0]), str(k[1]), float(k[2]), int(k[3]))
            out[key] = {f: d[f"{k[0]}_{k[1]}_{k[2]}_{k[3]}_{f}"] for f in keep}
        return out
    rho, geom = ms.load_atmosphere()
    src = ms.source_interp(ms.muon_source_table())
    fld = ms.FieldGrid()
    axis = ew_axis()
    out = {}
    for zen in zeniths:
        for azd, tag in ((axis, "E"), ((axis + 180.0) % 360.0, "W")):
            for E in energies:
                for q in (+1, -1):
                    out[(zen, tag, E, q)] = ms.segment_mc(
                        zen, azd, E, q, n=n, seed=seed, field=fld, rho=rho,
                        geom=geom, source=src, prod_dir=prod_dir,
                        energy_loss=energy_loss, polarisation=polarisation)
    if cache is not None:
        blob, keys = {}, []
        for (zen, tag, E, q), r in out.items():
            k = (f"{zen:g}", tag, f"{E:g}", f"{q:d}")
            keys.append(k)
            for f in keep:
                blob["_".join(k) + f"_{f}"] = r[f]
        np.savez_compressed(cache, keys=np.array(keys, dtype=object), **blob)
    return out


def section_b(grids, delta):
    for prod_dir, g in grids.items():
        hr(f"B. bending-angle distributions  [prod_dir = {prod_dir}]"
           f"   (delivered single shift = {delta:.2f} deg)")
        print("  species        zen  az   E_nu | <bend>   RMS   p16   p50   p84 "
              "| <dzen>  <daz> | <T/tau> <p_i/p_f> | neff")
        for sp in SPECIES:
            q = +1 if sp in MU_PLUS else -1
            kind = MICHEL[sp]
            for zen in ZENITHS:
                for tag in ("E", "W"):
                    for E in E_POINTS:
                        r = g[(zen, tag, E, q)]
                        w = r[f"w_{kind}"]
                        sb = ms.wstats(r["bend_deg"], w)
                        sz = ms.wstats(r["dzen"], w)
                        sa = ms.wstats(r["daz"], w)
                        tt = ms.wstats(r["T_p"] / ms.TAU_MU, w)
                        rr = ms.wstats(r["ratio"], w)
                        print(f"  {SP_LABEL[sp]:9s} (mu{'+' if q > 0 else '-'}) "
                              f"{zen:3.0f} {tag}  {E:4.2f} |"
                              f"{sb['mean']:6.2f} {sb['rms']:5.2f} {sb['p16']:5.2f} "
                              f"{sb['p50']:5.2f} {sb['p84']:5.2f} |"
                              f"{sz['mean']:+6.2f} {sa['mean']:+6.2f} |"
                              f" {tt['mean']:6.3f} {rr['mean']:8.4f} |"
                              f"{sb['neff']:6.0f}")


def section_b_summary(grids, delta):
    hr("B-summary. mean bend / delivered 5.08 deg, by zenith and charge "
       "(nu_mu-type daughter, E_nu = 0.5 GeV)")
    print("  prod_dir  zen  az |  mu+ bend  ratio |  mu- bend  ratio")
    for prod_dir, g in grids.items():
        for zen in ZENITHS:
            for tag in ("E", "W"):
                row = []
                for q in (+1, -1):
                    r = g[(zen, tag, 0.5, q)]
                    m = ms.wstats(r["bend_deg"], r["w_numu"])["mean"]
                    row += [m, m / delta]
                print(f"  {prod_dir:8s} {zen:3.0f} {tag}  |{row[0]:8.2f} "
                      f"{row[1]:6.2f} |{row[2]:8.2f} {row[3]:6.2f}")


# ---------------------------------------------------------------------------
def section_e(n=30000, seed=3):
    """Ablations: energy loss on/off, muon polarisation, prod_dir -- on the mean
    bend of the nu_mu-type daughter at E_nu = 0.5 GeV."""
    hr("E. ablations on the mean bend [deg] (nu_mu-type daughter, E_nu = 0.5 GeV,"
       " geomagnetic East)")
    rho, geom = ms.load_atmosphere()
    src = ms.source_interp(ms.muon_source_table())
    fld = ms.FieldGrid()
    axis = ew_axis()
    print("  variant                    zen=87    zen=81    zen=75    zen=0")
    variants = (
        ("dE/dx ON  , prod_dir=ray ", dict(energy_loss=True, prod_dir="ray")),
        ("dE/dx OFF , prod_dir=ray ", dict(energy_loss=False, prod_dir="ray")),
        ("dE/dx ON  , prod_dir=muon", dict(energy_loss=True, prod_dir="muon")),
        ("dE/dx ON  , pol P=+1     ", dict(energy_loss=True, prod_dir="ray",
                                           polarisation=+1.0)),
        ("dE/dx ON  , pol P=-1     ", dict(energy_loss=True, prod_dir="ray",
                                           polarisation=-1.0)),
    )
    for name, kw in variants:
        row = []
        for zen in (87.0, 81.0, 75.0, 0.0):
            az = axis if zen > 0 else 0.0
            r = ms.segment_mc(zen, az, 0.5, +1, n=n, seed=seed, field=fld,
                              rho=rho, geom=geom, source=src, **kw)
            row.append(ms.wstats(r["bend_deg"], r["w_numu"])["mean"])
        print(f"  {name}  " + "  ".join(f"{v:8.2f}" for v in row))
    print("  (a mu+; the mu- values mirror it except in the prod_dir=muon row)")


# ---------------------------------------------------------------------------
def section_c(n=4000):
    hr("C. forward companion: decay in flight vs ground vs range-out")
    rho, geom = ms.load_atmosphere()
    src = ms.source_interp(ms.muon_source_table())
    fld = ms.FieldGrid()
    print("  zen   E_mu[GeV]  f_decay  f_ground  f_stop  <bend at decay>[deg]")
    for zen in (87.0, 81.0, 75.0, 0.0):
        az = ew_axis() if zen > 0 else 0.0
        for row in ms.forward_stats(zen, az, n=n, field=fld, rho=rho, geom=geom,
                                    source=src):
            print(f"  {zen:3.0f}  {row['e_lo']:5.1f}-{row['e_hi']:5.1f} "
                  f"  {row['f_decay']:7.3f} {row['f_ground']:8.3f} "
                  f"{row['f_stop']:7.3f}  {row['mean_psi_deg']:8.2f}")


# ---------------------------------------------------------------------------
def section_d(grids, n_exp=20000, cache_dir=CACHE, engine=None):
    """Cutoff response: <G> under the axis / mean shift / exponential / MC."""
    from mceq3d_flux import MCEq3DFlux, _interp_rc, RC_MAX_GV
    from joint_cone import load_rc_map, rc_bilinear
    from kinematic_kernel import channel_fractions
    from muon_bending import local_field_enu, bending_angle

    eng = engine or MCEq3DFlux(base_model="hybrid",
                               primary=("GlobalSplineFitBeta", None),
                               daemonflux_location="kamioka")
    zen_f, az_f, rc_f = load_rc_map(PAPER_MAP)

    def rc_at(th, ph):
        return rc_bilinear(zen_f, az_f, rc_f, th, ph)
    b_enu = local_field_enu(LAT, LON, DATE)
    bmag = float(np.linalg.norm(b_enu))
    khat = b_enu / bmag
    delta = bending_angle(bmag)
    rc_grid = np.linspace(0.1, RC_MAX_GV, 40)
    rng = np.random.default_rng(11)
    uexp = rng.exponential(1.0, n_exp)
    axis = ew_axis()

    hr(f"D. muon-channel <G> and the implied W/E  (map {PAPER_MAP}, "
       f"vertical R_c = {rc_f[0].mean():.2f} GV)")

    we = {}
    for zen in ZENITHS:
        cz = np.cos(np.radians(zen))
        czr = min(CZ_BANDS, key=lambda b: abs(b - cz))
        G, _ = eng.geomag_response(rc_grid, cz_ref=czr, cache_dir=cache_dir)
        gcurve = {}
        for sp in SPECIES:
            gm = G[sp]
            for E in E_POINTS:
                ie = int(np.argmin(np.abs(eng.e - E)))
                gcurve[(sp, E)] = np.array(
                    [_interp_rc(r, rc_grid, gm)[ie] for r in rc_grid])

        def g_of(rcv, sp, E):
            return np.interp(rcv, rc_grid, gcurve[(sp, E)])

        print(f"\n--- zenith {zen:.0f} deg (cosZ = {cz:.3f}) ---")
        for tag, azd in (("E", axis), ("W", (axis + 180.0) % 360.0)):
            n_enu = ms.arrival_enu(zen, azd)
            th0, ph0 = ms.enu_angles(n_enu[None, :])
            rc_axis = float(np.ravel(rc_at(th0, ph0))[0])
            print(f"  [{tag}] axis R_c = {rc_axis:6.2f} GV")
            for sp in SPECIES:
                q = +1 if sp in MU_PLUS else -1
                kind = MICHEL[sp]
                # delivered single mean shift
                nm = ms.rodrigues(n_enu[None, :], khat, np.array([q * delta]))[0]
                nm /= np.linalg.norm(nm)
                rc_mean = float(np.ravel(rc_at(*ms.enu_angles(nm[None, :])))[0])
                # exponential-in-lifetime model
                Pe = ms.rodrigues(np.broadcast_to(n_enu, (n_exp, 3)), khat,
                                  q * delta * uexp)
                Pe /= np.linalg.norm(Pe, axis=1)[:, None]
                rc_e = np.asarray(rc_at(*ms.enu_angles(Pe)))
                for E in E_POINTS:
                    f = float(channel_fractions(np.array([E]), SP_LABEL[sp],
                                                zenith_deg=zen)["mu"][0])
                    g0 = float(g_of(rc_axis, sp, E))
                    gmean = float(g_of(rc_mean, sp, E))
                    gexp = float(g_of(rc_e, sp, E).mean())
                    row = dict(axis=g0, mean=gmean, exp=gexp)
                    for pd, g in grids.items():
                        r = g[(zen, tag, E, q)]
                        w = r[f"w_{kind}"]
                        rc_mc = np.asarray(rc_at(*ms.enu_angles(r["n_prim"])))
                        gv = g_of(rc_mc, sp, E)
                        s = max(w.sum(), 1e-300)
                        row[pd] = float((w * gv).sum() / s)
                    print(f"    {SP_LABEL[sp]:9s} E={E:4.2f}  f_mu={f:.3f}  "
                          f"G: axis {row['axis']:.4f}  mean-shift {row['mean']:.4f}"
                          f"  exp {row['exp']:.4f}  MC-ray {row['ray']:.4f}"
                          f"  MC-muon {row['muon']:.4f}")
                    for k in ("mean", "exp", "ray", "muon"):
                        we.setdefault((zen, tag, sp, E), {})[k] = (
                            (1 - f) * g0 + f * row[k])
                    we[(zen, tag, sp, E)]["axis"] = g0

    hr("D-2. implied W/E of the geomagnetic factor, and the implied change in "
       "the delivered W/E")
    print("  The muon channel enters as (1-f_mu) G(axis) + f_mu G(shift); the")
    print("  table gives (W/E)_G for each treatment and the multiplicative")
    print("  change of the delivered W/E when the single mean shift is replaced.")
    print("\n  zen  species    E_nu | (W/E)_G  no-bend  mean-shift  exp     "
          "MC-ray   MC-muon | delivered -> MC-ray  MC-muon | Honda")
    for zen in ZENITHS:
        for sp in SPECIES:
            for E in E_POINTS:
                kE = we.get((zen, "E", sp, E))
                kW = we.get((zen, "W", sp, E))
                if not kE:
                    continue
                r = {k: kW[k] / max(kE[k], 1e-300)
                     for k in ("axis", "mean", "exp", "ray", "muon")}
                key = (zen, SP_LABEL[sp], E)
                dv, hv = DELIVERED_WE.get(key, (np.nan, np.nan))
                print(f"  {zen:3.0f}  {SP_LABEL[sp]:9s} {E:4.2f} |"
                      f" {r['axis']:7.3f} {r['mean']:10.3f} {r['exp']:7.3f}"
                      f" {r['ray']:8.3f} {r['muon']:8.3f} |"
                      f" {dv:8.2f} -> {dv * r['ray'] / r['mean']:6.2f} "
                      f"{dv * r['muon'] / r['mean']:7.2f} | {hv:6.2f}")


# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sections", default="ABCDE")
    p.add_argument("-n", type=int, default=30000)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--grid-cache", default=None,
                   help="prefix for a per-prod_dir sample cache (npz)")
    a = p.parse_args(argv)
    from muon_bending import local_field_enu, bending_angle

    delta = np.degrees(bending_angle(
        float(np.linalg.norm(local_field_enu(LAT, LON, DATE)))))
    if "A" in a.sections:
        section_a()
    grids = {}
    if set("BD") & set(a.sections):
        for pd in ("ray", "muon"):
            grids[pd] = _mc_grid(a.n, a.seed, pd,
                                 cache=None if a.grid_cache is None else
                                 f"{a.grid_cache}_{pd}_n{a.n}_s{a.seed}.npz")
    if "B" in a.sections:
        section_b(grids, delta)
        section_b_summary(grids, delta)
    if "C" in a.sections:
        section_c()
    if "E" in a.sections:
        section_e(n=a.n, seed=a.seed)
    if "D" in a.sections:
        section_d(grids)
    print("\nDIAG_MUON_SEGMENT_DONE")


if __name__ == "__main__":
    main()
