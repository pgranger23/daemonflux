"""Read a ``sky.py`` run and produce the milestone-2b tables.

    python analyse_sky.py pilot_total.npz [--collinear-total coll_total.npz]

Reported, in order:

(a) **cost**: effective entries per bin (``(sum w)^2 / sum w^2`` -- the right
    measure when the weights are spread, and the one that sets the statistical
    error), the measured 3D/1D correlation coefficient across shards, and the
    core-hours implied for a 3% East-West table and a 5% survey.
(b) **the 3D/1D ratio vs cos Z** at 0.3 / 0.5 / 1 GeV, azimuth-averaged, at the
    largest cap, plus the nested-cap linear extrapolation in the cap solid
    angle ``Omega_D -> 0``.
(c) **the two measures**: the plain solid-angle average of the ratio and the
    ``cos``-weighted (ground-crossing) average, versus energy -- the question
    raised in ``tools/mceq3d/PHASE1_RESULTS.md`` sec. 8.
"""

from __future__ import annotations

import argparse

import numpy as np

from scoring import SPECIES

NAME = {12: "nue", -12: "antinue", 14: "numu", -14: "antinumu"}


def neff(sw, sw2):
    sw2 = np.where(sw2 > 0, sw2, np.inf)
    return sw * sw / sw2


def ratio_err(a, a2, b, b2, rho=0.0):
    """Error on ``a/b`` from the two weight sums, with a correlation ``rho``."""
    ra = np.sqrt(a2) / np.where(a > 0, a, np.inf)
    rb = np.sqrt(b2) / np.where(b > 0, b, np.inf)
    v = ra ** 2 + rb ** 2 - 2.0 * rho * ra * rb
    return np.abs(a / np.where(b > 0, b, np.inf)) * np.sqrt(np.maximum(v, 0.0))


def load(fp, erebin=1, czrebin=1, azsum=False):
    """Load a run, optionally merging adjacent bins.

    The tally is always accumulated on Honda's full grid (20 E-bins/decade,
    ``Delta cos Z = 0.1``, 12 azimuth bins).  A pilot has ~20 raw entries per
    bin, so the *reported* tables merge bins after the fact -- which is exact,
    because every array in the file is a plain sum of weights (or of squared
    weights) and therefore additive.  ``erebin=2`` gives Honda's coarser
    10 E-bins/decade; ``azsum`` collapses the azimuth axis, which is free here
    because with ``B = 0`` and no cutoff the problem is exactly azimuthally
    symmetric.
    """
    d = dict(np.load(fp, allow_pickle=False))
    keys = ("s0", "s0sq", "s1", "s1sq", "t0", "t0sq", "t1", "t1sq")
    if erebin > 1:
        ne = (len(d["e_bins"]) - 1) // erebin * erebin
        for k in keys:
            a = d[k][:, :, :ne]
            d[k] = a.reshape(a.shape[0], a.shape[1], ne // erebin, erebin,
                             *a.shape[3:]).sum(axis=3)
        d["e_bins"] = d["e_bins"][:ne + 1:erebin]
    if czrebin > 1:
        nz = (len(d["cz_bins"]) - 1) // czrebin * czrebin
        for k in keys:
            a = d[k][:, :, :, :nz]
            d[k] = a.reshape(*a.shape[:3], nz // czrebin, czrebin,
                             a.shape[4]).sum(axis=4)
        d["cz_bins"] = d["cz_bins"][:nz + 1:czrebin]
    if azsum:
        for k in keys:
            d[k] = d[k].sum(axis=-1, keepdims=True)
        d["az_bins"] = np.array([0.0, 360.0])
    return d


def cz_index(cz_bins, c):
    return int(np.searchsorted(cz_bins, c + 1e-9) - 1)


def e_index(e_bins, e):
    return int(np.searchsorted(e_bins, e * (1 + 1e-9)) - 1)


def rebin_note(d):
    return (f"E bins/decade {1.0/np.log10(d['e_bins'][1]/d['e_bins'][0]):.0f}, "
            f"Delta cosZ {d['cz_bins'][1]-d['cz_bins'][0]:.2f}, "
            f"{len(d['az_bins'])-1} azimuth bins")


def probe_correlations(d):
    """Pearson r of the 3D and 1D shard sums, per probe bin."""
    from sky import PROBES
    p = d.get("probes")
    if p is None or len(p) < 4:
        return None
    p = np.asarray(p, float)
    out = []
    for j, pr in enumerate(PROBES):
        a, b = p[:, 4 * j], p[:, 4 * j + 1]
        m = (a > 0) | (b > 0)
        if m.sum() < 4:
            out.append((pr, np.nan, 0))
            continue
        r = np.corrcoef(a[m], b[m])[0, 1]
        out.append((pr, float(r), int(m.sum())))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("total")
    ap.add_argument("--collinear-total", default=None)
    ap.add_argument("--cap", type=int, default=-1)
    ap.add_argument("--erebin", type=int, default=1)
    ap.add_argument("--czrebin", type=int, default=1)
    a = ap.parse_args(argv)
    d = load(a.total, a.erebin, a.czrebin)
    eb, cz, az = d["e_bins"], d["cz_bins"], d["az_bins"]
    caps, areas = d["cap_theta"], d["cap_area"]
    n_prim, n_nuc, n_nu, n_hit3, n_hit1 = d["counts"]
    wall = float(d["wall_s"][0]) if "wall_s" in d else np.nan
    nproc = float(d["nproc"][0]) if "nproc" in d else np.nan
    print(f"# {a.total}")
    print(f"primaries {n_prim:.0f}   neutrinos >E_min {n_nu:.0f} "
          f"({n_nu/max(n_prim,1):.2f}/shower)   3D cap hits {n_hit3:.0f} "
          f"({n_hit3/max(n_nu,1):.4f})   1D cap hits {n_hit1:.0f}")
    if np.isfinite(wall):
        print(f"wall {wall:.0f} s on {nproc:.0f} cores -> "
              f"{n_prim/wall:.1f} showers/s, "
              f"{n_prim/wall/max(nproc,1):.2f} showers/s/core, "
              f"{nproc*wall/3600.0:.3f} core-h for {n_prim:.0f} showers")
    ic = a.cap
    print("binning: " + rebin_note(d))
    print(f"\ncaps {list(np.round(caps,2))} deg, "
          f"reporting cap {caps[ic]:.1f} deg (A = {areas[ic]:.3e} cm2)")

    # ---------------- (a) entries per bin, near the horizon ---------------
    print("\n=== (a) effective entries per bin (azimuth bin of 30 deg, "
          "cos Z bin of 0.1, 20 E-bins/decade) ===")
    print(f"{'species':10s} {'E [GeV]':>9s} {'cosZ':>10s} "
          f"{'N_eff 3D':>10s} {'N_eff 1D':>10s} {'per 1e6 prim':>13s} "
          f"{'rel.err':>8s}")
    rows = []
    for s in (14, -14, 12, -12):
        k = SPECIES.index(s)
        for e in (0.3, 0.5, 1.0):
            ie = e_index(eb, e)
            for c0 in (0.0, 0.1):
                iz = cz_index(cz, c0)
                a3 = d["s1"][ic, k, ie, iz, :].sum()
                a3s = d["s1sq"][ic, k, ie, iz, :].sum()
                b1 = d["t1"][ic, k, ie, iz, :].sum()
                b1s = d["t1sq"][ic, k, ie, iz, :].sum()
                n3 = neff(a3, a3s) / max(len(az) - 1, 1)
                n1 = neff(b1, b1s) / max(len(az) - 1, 1)
                rel = 1.0 / np.sqrt(n3) if n3 > 0 else np.nan
                rows.append((s, e, c0, n3, n1))
                print(f"{NAME[s]:10s} {e:9.2f} {c0:5.1f}-{c0+0.1:4.1f} "
                      f"{n3:10.1f} {n1:10.1f} "
                      f"{n3/max(n_prim,1)*1e6:13.1f} {rel:8.3f}")

    cr = probe_correlations(d)
    if cr:
        print("\n3D/1D correlation across shards (probe bins, azimuth-summed):")
        for pr, r, nsh in cr:
            print(f"  {NAME[pr[0]]:9s} E={pr[1]:4.1f} cosZ {pr[2]:.1f}-{pr[3]:.1f}"
                  f"  rho = {r:+.3f}  ({nsh} shards)")

    # ---------------- (b) azimuth-averaged 3D/1D vs cos Z -----------------
    print("\n=== (b) azimuth-averaged 3D/1D ratio vs cos Z ===")
    for s in (14, 12):
        k = SPECIES.index(s)
        print(f"\n  {NAME[s]}   (rows: cos Z band; cols: E; "
              f"'ext' = nested-cap linear extrapolation to Omega_D -> 0)")
        hdr = "  cosZ      "
        for e in (0.3, 0.5, 1.0):
            hdr += f"{e:>8.1f} GeV{'':>10s}"
        print(hdr)
        for iz in range(len(cz) - 1):
            if cz[iz] < -1e-9:
                continue
            line = f"  {cz[iz]:4.1f}-{cz[iz+1]:4.1f} "
            for e in (0.3, 0.5, 1.0):
                ie = e_index(eb, e)
                rr, ee = [], []
                for jc in range(len(caps)):
                    n3 = d["s1"][jc, k, ie, iz, :].sum()
                    n3s = d["s1sq"][jc, k, ie, iz, :].sum()
                    n1 = d["t1"][jc, k, ie, iz, :].sum()
                    n1s = d["t1sq"][jc, k, ie, iz, :].sum()
                    rr.append(n3 / n1 if n1 > 0 else np.nan)
                    ee.append(ratio_err(n3, n3s, n1, n1s, rho=0.6))
                rr, ee = np.array(rr), np.array(ee)
                m = np.isfinite(rr) & (ee > 0)
                if m.sum() >= 2:
                    om = 2 * np.pi * (1 - np.cos(np.deg2rad(caps)))
                    cf = np.polyfit(om[m], rr[m], 1, w=1.0 / ee[m])
                    ext = cf[1]
                else:
                    ext = np.nan
                line += f"{rr[-1]:7.3f}+-{ee[-1]:5.3f}/{ext:6.3f} "
            print(line)

    # ---------------- (c) the two measures --------------------------------
    print("\n=== (c) hemisphere-averaged 3D/1D vs energy: plain solid angle "
          "vs cos-weighted (ground crossing) ===")
    izs = [i for i in range(len(cz) - 1) if cz[i] >= -1e-9]
    print(f"{'E [GeV]':>9s} " + " ".join(
        f"{NAME[s]:>9s}_pl {NAME[s]:>7s}_cos" for s in (14, -14, 12, -12)))
    for e in (0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0):
        ie = e_index(eb, e)
        line = f"{e:9.2f} "
        for s in (14, -14, 12, -12):
            k = SPECIES.index(s)
            a1 = d["s1"][ic, k, ie][izs].sum()
            b1 = d["t1"][ic, k, ie][izs].sum()
            a0 = d["s0"][ic, k, ie][izs].sum()
            b0 = d["t0"][ic, k, ie][izs].sum()
            line += (f"{a1/b1 if b1>0 else np.nan:12.3f} "
                     f"{a0/b0 if b0>0 else np.nan:11.3f}")
        print(line)

    # ---------------- absolute flux sanity check --------------------------
    print("\n=== absolute azimuth-averaged flux "
          "[cm^-2 s^-1 sr^-1 GeV^-1] (3D, largest cap) ===")
    dcz = cz[1] - cz[0]
    print(f"{'E [GeV]':>9s} {'cosZ':>10s} " +
          " ".join(f"{NAME[s]:>11s}" for s in (14, -14, 12, -12)))
    for e in (0.3, 1.0, 3.0):
        ie = e_index(eb, e)
        de = eb[ie + 1] - eb[ie]
        for c0 in (0.0, 0.5, 0.9):
            iz = cz_index(cz, c0)
            line = f"{e:9.2f} {c0:5.1f}-{c0+0.1:4.1f} "
            for s in (14, -14, 12, -12):
                k = SPECIES.index(s)
                v = d["s1"][ic, k, ie, iz, :].sum() / (2 * np.pi * dcz * de)
                line += f"{v:12.4e}"
            print(line)

    if a.collinear_total:
        dc = load(a.collinear_total, a.erebin, a.czrebin)
        print("\n=== the in-run correlated 1D reference vs a true collinear "
              "run on the same primaries ===")
        print(f"{'E [GeV]':>9s} {'cosZ':>10s} " +
              " ".join(f"{NAME[s]:>9s}" for s in (14, 12)))
        for e in (0.3, 1.0):
            ie = e_index(eb, e)
            for c0 in (0.0, 0.5, 0.9):
                iz = cz_index(cz, c0)
                line = f"{e:9.2f} {c0:5.1f}-{c0+0.1:4.1f} "
                for s in (14, 12):
                    k = SPECIES.index(s)
                    corr = d["t1"][ic, k, ie, iz, :].sum()
                    true = dc["s1"][ic, k, ie, iz, :].sum()
                    line += f"{corr/true if true>0 else np.nan:10.3f}"
                print(line)


if __name__ == "__main__":
    main()
