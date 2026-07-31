"""In-depth comparison of the delivered flux against Honda (HKKM2014, Kamioka),
across every axis Honda's table supports: absolute normalisation, zenith shape,
flavour (nue/numu), charge (nu/nubar), and charge-dependent East-West.

Honda's kam-ally table is (cosZ=20, az=12, E=101), split into numu/numubar/nue/
nuebar -- richer than anything checked so far this session (which only ever used
the charge-SUMMED numu, and only a handful of (E,cosZ) spot points for E-W). This
requests the delivered flux on Honda's EXACT down-going (cosZ,az) grid (10x12,
cosZ centres 0.05-0.95, az centres 15-345 deg) for all 4 species in one solve()
call, so every comparison below is a direct cell-by-cell ratio, not an
interpolated spot-check.

Sections:
  A. Absolute flux ratio (ours/Honda), all 4 species, vertical/mid/horizon x
     0.15-100 GeV.
  B. Full zenith shape (az-averaged, relative to vertical) at 3 energies, all 10
     cosZ bins -- not just the horizon/vertical endpoint used elsewhere.
  C. Flavour ratio (nue+nuebar)/(numu+numubar), az-averaged, ours vs Honda.
  D. Charge ratio numu/numubar and nue/nuebar (az-averaged), ours vs Honda.
  E. CHARGE-DEPENDENT East-West: does the muon-bending charge split reproduce
     Honda's OWN nu-vs-nubar E-W amplitude difference? (not checked before.)
  F. Charge-summed E-W amplitude vs zenith x energy (consolidates earlier spot
     checks into the same grid/dataset).
  G. Headline summary: median/RMS |log10(ours/Honda)| over the full grid, per
     species, restricted to 0.1-100 GeV (the engine's validated range).
  H. Bartol cross-check (independent of Honda) on absolute vertical + zenith
     shape, numu and nue.

Run from tools/mceq3d (PYTHONPATH=$PWD). Needs the .cache3d / gs_* caches (should
mostly hit from this session's earlier hybrid/GSF/Kamioka solve() calls).
"""
from datetime import datetime

import numpy as np

from mceq3d_flux import MCEq3DFlux

LAT, LON = 36.43, 137.31
DATE = datetime(2020, 1, 1)
CACHE = ".cache3d"
EGRID = np.array([0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0])
SPECIES = ("total_numu", "total_antinumu", "total_nue", "total_antinue")
HKEY = {"total_numu": "numu", "total_antinumu": "numubar",
       "total_nue": "nue", "total_antinue": "nuebar"}


def log_at(y, x, X):
    X = np.atleast_1d(X)
    out = np.exp(np.interp(np.log(X), np.log(x), np.log(np.maximum(y, 1e-300))))
    return out if out.size > 1 else float(out[0])


def main():
    h = dict(np.load("honda_kam.npz"))
    He, Hcz, Haz = h["E"], h["czlo"], h["azlo"]
    cz = Hcz[Hcz >= 0.0] + 0.05          # down-going bin centres, 0.05..0.95 (10)
    az = Haz.astype(float) + 15.0        # bin centres, 15..345 (12)
    cz_idx0 = int(np.argmin(np.abs(Hcz - 0.0)))  # first down-going row in h arrays

    print(f"Requesting delivered flux on Honda's exact grid: {len(cz)} cosZ x "
          f"{len(az)} az x {len(SPECIES)} species ...")
    eng = MCEq3DFlux(base_model="hybrid", primary=("GlobalSplineFitBeta", None),
                     daemonflux_location="kamioka")
    r = eng.solve(LAT, LON, cz, az, offaxis=True, use_cache=True, cone_cutoff=True,
                 cache_dir=CACHE, date=DATE)
    e = r["e"]
    F = {s: r["flux"][s] for s in SPECIES}  # (n_cz, n_az, nE)
    print("... done.\n")

    def honda_grid(hkey):
        """Honda's down-going flux[cz,az,E] restricted to e's validated range."""
        arr = h[hkey][cz_idx0:cz_idx0 + len(cz)]  # (n_cz, n_az_honda, E)
        return arr

    def honda_at(hkey, icz, iaz, E):
        return log_at(h[hkey][cz_idx0 + icz, iaz], He, E)

    def ours_at(s, icz, iaz, E):
        return log_at(F[s][icz, iaz], e, E)

    iv, im, ih = 9, 5, 0  # cosZ=0.95(vert), 0.55(mid), 0.05(horizon) indices

    # ---------------- A. ABSOLUTE FLUX ----------------
    print("=" * 78)
    print("A. ABSOLUTE FLUX ratio (ours/Honda), azimuth-averaged")
    print("=" * 78)
    for s in SPECIES:
        hkey = HKEY[s]
        print(f"\n[{s}]")
        print(f"{'E[GeV]':>7} {'vert(0.95)':>11} {'mid(0.55)':>10} "
              f"{'horiz(0.05)':>12}")
        for E in EGRID:
            row = []
            for icz in (iv, im, ih):
                o = float(np.interp(E, e, F[s][icz].mean(0)))
                hn = log_at(h[hkey][cz_idx0 + icz].mean(0), He, E)
                row.append(o / max(hn, 1e-300))
            print(f"{E:7.2f} {row[0]:11.2f} {row[1]:10.2f} {row[2]:12.2f}")

    # ---------------- B. FULL ZENITH SHAPE ----------------
    print("\n" + "=" * 78)
    print("B. ZENITH SHAPE Phi(cosZ)/Phi(vertical), az-averaged, ours vs Honda")
    print("=" * 78)
    for s in ("total_numu", "total_nue"):
        hkey = HKEY[s]
        print(f"\n[{s}]  (o=ours, h=Honda; all 10 down-going cosZ bins)")
        for E in (0.3, 1.0, 10.0):
            o_v = float(np.interp(E, e, F[s][iv].mean(0)))
            h_v = log_at(h[hkey][cz_idx0 + iv].mean(0), He, E)
            devs = []
            line = f"  E={E:5.1f}  cz: "
            vals_o, vals_h = [], []
            for icz in range(len(cz)):
                o = float(np.interp(E, e, F[s][icz].mean(0))) / o_v
                hn = log_at(h[hkey][cz_idx0 + icz].mean(0), He, E) / h_v
                vals_o.append(o); vals_h.append(hn)
                devs.append(abs(o / hn - 1))
            print(f"  E={E:5.1f} GeV  ours:  " +
                  " ".join(f"{v:4.2f}" for v in vals_o))
            print(f"             Honda: " +
                  " ".join(f"{v:4.2f}" for v in vals_h))
            print(f"             max dev = {max(devs)*100:.0f}%  "
                  f"(at cosZ={cz[int(np.argmax(devs))]:.2f})")

    # ---------------- C. FLAVOUR RATIO ----------------
    print("\n" + "=" * 78)
    print("C. FLAVOUR RATIO (nue+nuebar)/(numu+numubar), az-averaged")
    print("=" * 78)
    print(f"{'E[GeV]':>7} {'ours(vert)':>10} {'Honda(vert)':>12} {'dev':>6} | "
          f"{'ours(horiz)':>11} {'Honda(horiz)':>12} {'dev':>6}")
    for E in EGRID:
        row = []
        for icz in (iv, ih):
            e_num = float(np.interp(E, e, F["total_nue"][icz].mean(0))
                          + np.interp(E, e, F["total_antinue"][icz].mean(0)))
            e_den = float(np.interp(E, e, F["total_numu"][icz].mean(0))
                          + np.interp(E, e, F["total_antinumu"][icz].mean(0)))
            o_r = e_num / max(e_den, 1e-300)
            h_num = (log_at(h["nue"][cz_idx0+icz].mean(0), He, E)
                    + log_at(h["nuebar"][cz_idx0+icz].mean(0), He, E))
            h_den = (log_at(h["numu"][cz_idx0+icz].mean(0), He, E)
                    + log_at(h["numubar"][cz_idx0+icz].mean(0), He, E))
            h_r = h_num / max(h_den, 1e-300)
            row += [o_r, h_r, (o_r/h_r-1)*100]
        print(f"{E:7.2f} {row[0]:10.3f} {row[1]:12.3f} {row[2]:+5.0f}% | "
              f"{row[3]:11.3f} {row[4]:12.3f} {row[5]:+5.0f}%")

    # ---------------- D. CHARGE RATIO ----------------
    print("\n" + "=" * 78)
    print("D. CHARGE RATIO nu/nubar (az-averaged), ours vs Honda")
    print("=" * 78)
    for fl, num, den, hnum, hden in (
        ("numu/numubar", "total_numu", "total_antinumu", "numu", "numubar"),
        ("nue/nuebar", "total_nue", "total_antinue", "nue", "nuebar"),
    ):
        print(f"\n[{fl}]")
        print(f"{'E[GeV]':>7} {'ours(vert)':>10} {'Honda(vert)':>12} | "
              f"{'ours(horiz)':>11} {'Honda(horiz)':>12}")
        for E in EGRID:
            row = []
            for icz in (iv, ih):
                o = (float(np.interp(E, e, F[num][icz].mean(0)))
                    / max(float(np.interp(E, e, F[den][icz].mean(0))), 1e-300))
                hn = (log_at(h[hnum][cz_idx0+icz].mean(0), He, E)
                     / max(log_at(h[hden][cz_idx0+icz].mean(0), He, E), 1e-300))
                row += [o, hn]
            print(f"{E:7.2f} {row[0]:10.3f} {row[1]:12.3f} | "
                  f"{row[2]:11.3f} {row[3]:12.3f}")

    # ---------------- E. CHARGE-DEPENDENT E-W ----------------
    print("\n" + "=" * 78)
    print("E. CHARGE-DEPENDENT East-West: nu vs nubar amplitude, near horizon")
    print("=" * 78)
    print("(tests whether the muon-bending charge split reproduces Honda's own")
    print(" nu-vs-nubar E-W difference -- not checked before this script)")
    for fl, sp_nu, sp_nubar, h_nu, h_nubar in (
        ("numu", "total_numu", "total_antinumu", "numu", "numubar"),
        ("nue", "total_nue", "total_antinue", "nue", "nuebar"),
    ):
        print(f"\n[{fl}]  W/E amplitude (max/min over az), cosZ=0.05 (87 deg)")
        print(f"{'E[GeV]':>7} {'ours nu':>8} {'ours nubar':>10} {'ours diff':>10} | "
              f"{'Honda nu':>9} {'Honda nubar':>11} {'Honda diff':>10}")
        for E in (0.3, 0.5, 1.0, 2.0):
            o_nu = float(np.interp(E, e, F[sp_nu][ih].max(0))) / \
                max(float(np.interp(E, e, F[sp_nu][ih].min(0))), 1e-300)
            o_nb = float(np.interp(E, e, F[sp_nubar][ih].max(0))) / \
                max(float(np.interp(E, e, F[sp_nubar][ih].min(0))), 1e-300)
            h_row_nu = h[h_nu][cz_idx0 + ih]
            h_row_nb = h[h_nubar][cz_idx0 + ih]
            hn = log_at(h_row_nu.max(0), He, E) / max(log_at(h_row_nu.min(0), He, E), 1e-300)
            hnb = log_at(h_row_nb.max(0), He, E) / max(log_at(h_row_nb.min(0), He, E), 1e-300)
            print(f"{E:7.2f} {o_nu:8.2f} {o_nb:10.2f} {o_nu-o_nb:+10.2f} | "
                  f"{hn:9.2f} {hnb:11.2f} {hn-hnb:+10.2f}")

    # ---------------- F. CHARGE-SUMMED E-W vs ZENITH x ENERGY ----------------
    print("\n" + "=" * 78)
    print("F. CHARGE-SUMMED numu East-West amplitude vs zenith x energy")
    print("=" * 78)
    print(f"{'E[GeV]':>7} " + " ".join(f"cz={c:.2f}" for c in cz[::2]))
    for E in (0.3, 0.5, 1.0, 2.0, 5.0):
        row = []
        for icz in range(0, len(cz), 2):
            o = float(np.interp(E, e, F["total_numu"][icz].max(0))) / \
                max(float(np.interp(E, e, F["total_numu"][icz].min(0))), 1e-300)
            hn_row = h["numu"][cz_idx0 + icz]
            hn = log_at(hn_row.max(0), He, E) / max(log_at(hn_row.min(0), He, E), 1e-300)
            row.append(f"{o:4.2f}/{hn:4.2f}")
        print(f"{E:7.2f} " + " ".join(f"{v:>9}" for v in row))
    print("(each cell: ours/Honda W-E amplitude)")

    # ---------------- G. HEADLINE SUMMARY ----------------
    print("\n" + "=" * 78)
    print("G. HEADLINE: median / 90th-pct |log10(ours/Honda)| over the FULL grid")
    print("   (0.1-100 GeV x all 10 cosZ x all 12 az, per species)")
    print("=" * 78)
    e_sel = (e >= 0.1) & (e <= 100.0)
    for s in SPECIES:
        hkey = HKEY[s]
        hgrid = honda_grid(hkey)  # (10, 12, E_honda)
        hgrid_on_e = np.array([[log_at(hgrid[i, j], He, e[e_sel])
                                for j in range(hgrid.shape[1])]
                               for i in range(hgrid.shape[0])])  # (10,12,nE_sel)
        ratio = F[s][:, :, e_sel] / np.maximum(hgrid_on_e, 1e-300)
        logr = np.abs(np.log10(np.maximum(ratio, 1e-300)))
        med = np.median(logr)
        p90 = np.percentile(logr, 90)
        print(f"  {s:18s}  median |log10 ratio| = {med:.3f} (~{10**med:.2f}x)   "
              f"90th pct = {p90:.3f} (~{10**p90:.2f}x)")

    # ---------------- H. BARTOL CROSS-CHECK ----------------
    print("\n" + "=" * 78)
    print("H. BARTOL cross-check (independent of Honda): vertical numu/nue, "
          "zenith shape")
    print("=" * 78)
    try:
        import validate_bartol as vb
        for fl, sp in (("num", "total_numu"), ("nue", "total_nue")):
            Eb, czb, gb = vb.load_bartol(fl, "fmin")
            ivb = int(np.argmin(np.abs(czb - 0.95)))
            print(f"\n[{sp}] ours/Bartol, vertical:")
            # NB the cached Bartol table is the LOW-energy (20-bin) file only:
            # it stops at 9.441 GeV. np.interp does not extrapolate -- it clamps
            # to the last value -- so a linear request at 10 GeV silently reused
            # the 9.441 GeV flux and overstated Bartol by ~21% on this steeply
            # falling (~E^-2.9) spectrum, faking a 0.79 ratio for numu. Use
            # log-log (consistent with the rest of the codebase) and refuse to
            # report points beyond the table's range.
            for E in (0.3, 0.5, 1.0, 3.0, 10.0):
                o = log_at(F[sp][iv].mean(0), e, E)
                if E > Eb.max():
                    print(f"  E={E:6.2f}  ours={o:.3e}  Bartol=  (out of table "
                          f"range, max {Eb.max():.2f} GeV -- skipped)")
                    continue
                b = log_at(gb[ivb], Eb, E)
                print(f"  E={E:6.2f}  ours={o:.3e}  Bartol={b:.3e}  "
                      f"ratio={o/max(b,1e-300):.2f}")
    except Exception as ex:
        print(f"  (Bartol cross-check skipped: {type(ex).__name__}: {ex})")

    print("\nDIAG_FULL_COMPARISON_DONE")


if __name__ == "__main__":
    main()
