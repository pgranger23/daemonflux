"""Diagnostic: how big is the dropped Cov_cone(p, G) in the delivered product?

Phase 1, step 3 of the post-audit plan.  The delivered engine multiplies two
*separately* cone-averaged factors, ``E_off = <p>_cone/p_axis`` (``offaxis_mc``)
and ``<G_s>_cone`` (``mceq3d_flux.cone_geff``).  :mod:`joint_cone` evaluates the
single joint integral ``J_s`` on the SAME quadrature and reports

    C_s = J_s / (<p>_cone <G_s>_cone)      (the covariance correction)
    F_s = J_s / p_axis                     (the joint factor replacing E_off <G>)

Because the 1D base and ``E_off`` are azimuth-independent, ``C_s`` is exactly the
multiplicative change the joint treatment would make to the delivered flux, so

    change in horizon/vertical  =  C_s(horizon) / C_s(vertical) - 1
    change in the W/E amplitude =  C_s(West)    / C_s(East)     - 1

Run (from ``tools/mceq3d``)::

    python diag_joint_cone.py [--map PATH] [--prod-cache PATH] [--gs-cache DIR]

Both heavy ingredients (the depth-resolved MCEq production profile and the
per-zenith ``G_s``) are memoised; the first run costs a few minutes.
"""

from __future__ import annotations

import argparse
import importlib.util  # noqa: F401  (mceq_config import shim; must precede MCEq)
import os

import numpy as np

import joint_cone as jc

SPECIES_SHOWN = ("total_numu", "total_nue")
SP_SHORT = {"total_numu": "numu", "total_nue": "nue "}
AZ = (("E", 90.0), ("W", 270.0), ("N", 0.0), ("S", 180.0))
#: uniform ring used for the azimuth-AVERAGED horizon/vertical ratio and for the
#: max/min E-W amplitude, matching the 12-24 point rings the delivered
#: diagnostics use (``diag_G_zenith.py``, ``diag_full_comparison.py``).
AZ_RING = tuple(np.linspace(0.0, 360.0, 25)[:-1])
E_PROBE = (0.3, 0.5, 1.0, 3.0)
# cosZ list: the five requested plus the exact 87 deg / 75 deg used for the
# horizon/vertical and W/E summaries.
CZ_MAIN = (0.05, 0.15, 0.35, 0.65, 0.95)
CZ_87 = float(np.cos(np.radians(87.0)))
CZ_75 = float(np.cos(np.radians(75.0)))
LAT, LON = 36.43, 137.31  # Kamioka


def _at(res, key, s, E):
    arr = res[key][s] if isinstance(res[key], dict) else res[key]
    return float(np.interp(np.log(E), np.log(res["e"]), arr))


def build(args):
    from mceq3d_flux import MCEq3DFlux

    prod = jc.load_production(cache=args.prod_cache)
    fine = jc.load_rc_map(args.map)
    print(f"cutoff map : {args.map}")
    print(f"             {len(fine[0])} zeniths x {len(fine[1])} azimuths, "
          f"R_c in [{fine[2].min():.2f}, {fine[2].max():.2f}] GV, "
          f"vertical = {fine[2][0].mean():.2f} GV")
    eng = MCEq3DFlux()
    rc_grid = np.linspace(0.1, args.rc_hi, 40)
    czs = sorted(set(CZ_MAIN) | {CZ_87, CZ_75}, reverse=True)
    G = {}
    if args.gs_cz is not None:
        # one G_s for every zenith: G_s is validated zenith-independent to <=2%
        # (paper Section 3.4), and C_s is a RATIO of two cone averages of the
        # same G_s, so the residual cancels to well below that.
        g0 = jc.gs_on_grid(eng, rc_grid, args.gs_cz, prod["ep_grid"],
                           cache_dir=args.gs_cache)
        print(f"  G_s: single reference column cz_ref={args.gs_cz:.4f}")
        G = {cz: g0 for cz in czs}
    else:
        for cz in czs:
            G[cz] = jc.gs_on_grid(
                eng, rc_grid, max(cz, 1e-3), prod["ep_grid"],
                cache_dir=args.gs_cache
            )
            print(f"  G_s ready for cosZ={cz:.4f}")
    res, ring = {}, {}
    for cz in czs:
        terms = jc.zenith_terms(cz, prod, n_alpha=args.n_alpha,
                                n_beta=args.n_beta,
                                sigma_scale=args.sigma_scale)
        for name, az in AZ:
            res[(cz, name)] = jc.joint_cone(
                cz, az, prod, fine, rc_grid, G[cz], terms=terms
            )
        ring[cz] = [jc.joint_cone(cz, a, prod, fine, rc_grid, G[cz], terms=terms)
                    for a in AZ_RING]
        print(f"  cone done for cosZ={cz:.4f}")
    return res, ring, czs, fine


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", default=jc.PAPER_MAP,
                    help="cutoff-map .npz (finemap_rc cache); default = the "
                         "committed/paper full-sphere Kamioka map")
    ap.add_argument("--prod-cache", default=os.environ.get(
        "JOINT_CONE_PROD_CACHE", "joint_cone_prod.npz"))
    ap.add_argument("--gs-cache", default=os.environ.get(
        "JOINT_CONE_GS_CACHE", ".cache3d"))
    ap.add_argument("--rc-hi", type=float, default=jc.RC_MAX_GV,
                    help="upper end of the G_s rigidity grid [GV]")
    ap.add_argument("--gs-cz", type=float, default=None,
                    help="use ONE G_s reference column at this cos(zenith) for "
                         "every direction instead of the per-zenith G_s")
    ap.add_argument("--sigma-scale", type=float, default=1.0,
                    help="scale every cone width (1.12/0.88 = the NA61 pull; "
                         "1.41 = the width cone_geff actually uses, per the audit)")
    ap.add_argument("--n-alpha", type=int, default=44)
    ap.add_argument("--n-beta", type=int, default=18)
    args = ap.parse_args(argv)

    res, ring, czs, fine = build(args)

    # ---------------- A. the covariance correction C_s -------------------
    print("\n" + "=" * 78)
    print("A. Covariance correction  C_s = J_s / (<p>_cone <G_s>_cone) - 1  [%]")
    print("   (>0: the joint integral gives MORE flux than the delivered product)")
    print("=" * 78)
    for s in SPECIES_SHOWN:
        print(f"\n[{SP_SHORT[s]}]  {'cosZ':>6} " +
              "  ".join(f"{a:>6}" for a, _ in AZ) + "     E [GeV]")
        for E in E_PROBE:
            for cz in CZ_MAIN:
                row = [100 * (_at(res[(cz, a)], "C", s, E) - 1) for a, _ in AZ]
                print(f"        {cz:6.2f} " +
                      "  ".join(f"{v:+6.2f}" for v in row) +
                      (f"     {E:g}" if cz == CZ_MAIN[0] else ""))
            print()

    # ---------------- B. joint factor vs product --------------------------
    print("=" * 78)
    print("B. Joint 3D factor F_s = J_s/p_axis  vs  the delivered product "
          "E_off * <G_s>_cone")
    print("=" * 78)
    for s in SPECIES_SHOWN:
        print(f"\n[{SP_SHORT[s]}]")
        print(f"{'cosZ':>6} {'az':>3} {'E':>5} {'E_off':>7} {'<G>':>7} "
              f"{'product':>8} {'joint':>8} {'C-1 %':>7} {'w_sub':>6} {'f_sub':>6}")
        for cz in CZ_MAIN:
            for a, _ in AZ:
                r = res[(cz, a)]
                for E in E_PROBE:
                    print(f"{cz:6.2f} {a:>3} {E:5.1f} "
                          f"{_at(r,'E_off',s,E):7.4f} {_at(r,'G_cone',s,E):7.4f} "
                          f"{_at(r,'product',s,E):8.4f} {_at(r,'joint',s,E):8.4f} "
                          f"{100*(_at(r,'C',s,E)-1):+7.2f} "
                          f"{_at(r,'w_sublimb',s,E):6.3f} "
                          f"{_at(r,'f_sublimb',s,E):6.3f}")
            print()

    # ---------------- C. implied shape changes ----------------------------
    print("=" * 78)
    print("C. Implied change in the DELIVERED observables")
    print("   horizon/vertical : C_s(cosZ) / C_s(0.95) - 1   [%]")
    print("   W/E amplitude    : C_s(W) / C_s(E) - 1         [%]")
    print("   reported residuals to beat: H/V deficit -11..-15% at 0.3-1 GeV,")
    print("                               E-W (W/E) overshoot +15..+24% at 87 deg")
    print("=" * 78)
    print("\n[azimuth-AVERAGED horizon/vertical, the observable behind the "
          "-11..-15% deficit]")
    print("   d(H/V) = [<J>_az/<product>_az](cosZ) / [same](0.95) - 1  [%], "
          f"{len(AZ_RING)} azimuths")
    print(f"{'sp':>5} " + " ".join(f"{'cz='+format(c,'.2f'):>9}" for c in CZ_MAIN)
          + "   E [GeV]")
    for s in SPECIES_SHOWN:
        for E in E_PROBE:
            row = []
            for cz in CZ_MAIN:
                num = np.mean([_at(r, "J", s, E) for r in ring[cz]])
                den = np.mean([_at(r, "p_axis", None, E) * _at(r, "product", s, E)
                               for r in ring[cz]])
                nv = np.mean([_at(r, "J", s, E) for r in ring[0.95]])
                dv = np.mean([_at(r, "p_axis", None, E) * _at(r, "product", s, E)
                              for r in ring[0.95]])
                row.append(100 * ((num / den) / (nv / dv) - 1))
            print(f"{SP_SHORT[s]:>5} " + " ".join(f"{v:+9.2f}" for v in row)
                  + f"   {E:g}")

    print("\n[E-W amplitude max/min over the azimuth ring, the observable behind "
          "the +15..+24% overshoot]")
    print(f"{'sp':>5} {'zen':>5} {'E':>5} {'A_old':>7} {'A_new':>7} "
          f"{'change %':>9}")
    for s in SPECIES_SHOWN:
        for zdeg, cz in ((87.0, CZ_87), (75.0, CZ_75)):
            for E in E_PROBE:
                g = np.array([_at(r, "G_cone", s, E) for r in ring[cz]])
                f = np.array([_at(r, "joint", s, E) for r in ring[cz]])
                a_old, a_new = g.max() / g.min(), f.max() / f.min()
                print(f"{SP_SHORT[s]:>5} {zdeg:5.0f} {E:5.1f} {a_old:7.3f} "
                      f"{a_new:7.3f} {100*(a_new/a_old-1):+9.2f}")

    for zdeg, cz in ((87.0, CZ_87), (75.0, CZ_75)):
        print(f"\n--- zenith {zdeg:g} deg (cosZ = {cz:.4f}) ---")
        print(f"{'sp':>5} {'E':>5} | " +
              " ".join(f"{'H/V '+a:>8}" for a, _ in AZ) +
              f" | {'W/E':>8} {'A_EW old':>9} {'A_EW new':>9}")
        for s in SPECIES_SHOWN:
            for E in E_PROBE:
                hv = [100 * (_at(res[(cz, a)], "C", s, E)
                             / _at(res[(0.95, a)], "C", s, E) - 1) for a, _ in AZ]
                cE = _at(res[(cz, "E")], "C", s, E)
                cW = _at(res[(cz, "W")], "C", s, E)
                gE = _at(res[(cz, "E")], "G_cone", s, E)
                gW = _at(res[(cz, "W")], "G_cone", s, E)
                jE = _at(res[(cz, "E")], "joint", s, E)
                jW = _at(res[(cz, "W")], "joint", s, E)
                a_old = (gW - gE) / (gW + gE)
                a_new = (jW - jE) / (jW + jE)
                print(f"{SP_SHORT[s]:>5} {E:5.1f} | " +
                      " ".join(f"{v:+8.2f}" for v in hv) +
                      f" | {100*(cW/cE-1):+8.2f} {a_old:9.4f} {a_new:9.4f}")

    # ---------------- D. what drives it ------------------------------------
    print("\n" + "=" * 78)
    print("D. Why: cone weight below the local limb (reads the far-side map in")
    print("   cone_geff, but carries p = 0 in offaxis_mc) vs the production it")
    print("   actually carries")
    print("=" * 78)
    print(f"{'cosZ':>6} {'E':>5} {'w_sublimb':>10} {'f_sublimb':>10} "
          f"{'w_blocked':>10} {'rho(p,G) E':>11} {'rho(p,G) W':>11}")
    for cz in CZ_MAIN:
        for E in E_PROBE:
            rE, rW = res[(cz, "E")], res[(cz, "W")]
            print(f"{cz:6.2f} {E:5.1f} {_at(rE,'w_sublimb',None,E):10.3f} "
                  f"{_at(rE,'f_sublimb',None,E):10.3f} "
                  f"{_at(rE,'w_blocked',None,E):10.3f} "
                  f"{_at(rE,'rho_pG','total_numu',E):11.3f} "
                  f"{_at(rW,'rho_pG','total_numu',E):11.3f}")
    print("\n(rho is the cone-measure correlation coefficient of p and G for the")
    print(" pion channel; C - 1 = rho * sigma_p sigma_G / (<p><G>).)")

    print("\n" + "=" * 78)
    print("E. The cutoff map across the limb (what the sub-limb cone weight reads)")
    print("=" * 78)
    zen_f, az_f, rc_f = fine
    lo = max(0, len(zen_f) - 18)
    sl = slice(lo, min(lo + 10, len(zen_f)))
    print(f"{'az':>4} " + " ".join(f"{z:6.1f}" for z in zen_f[sl]))
    for nm, a in AZ:
        j = int(np.argmin(np.abs(az_f - a)))
        print(f"{nm:>4} " + " ".join(f"{v:6.2f}" for v in rc_f[sl, j]))
    print("(zenith 89 -> 90 deg is the down-going/far-side stitch; the delivered")
    print(" cone reads the right-hand columns for ~30-43% of its weight at 87 deg,")
    print(" while those directions carry ~1-7% of the production.)")
    return res


if __name__ == "__main__":
    main()
