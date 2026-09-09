"""Cumulative S0 -> S3 measurement of the cone-consistency and joint-cone steps.

Each stage changes ONE thing in the delivered ``solve()`` and is measured with
the two observables the paper quotes, for ``nu_mu`` **and** ``nu_e``:

* horizon/vertical ``Phi(cosZ=0.05)/Phi(cosZ=0.95)`` (azimuth-averaged) at
  0.3/0.5/1/3 GeV versus Honda (``diag_shape_decompose.py`` / paper 4.5, 6);
* the West/East ratio at 87/81/75 deg at 0.5 and 1 GeV versus Honda's max/min
  over azimuth (``validate_ew_zenith.py`` / paper 4.1).

Stages
------
``S0``  the delivered engine as reported after the cutoff-map repairs:
        ``cone_kernel="legacy"`` -- the sampled ``k_spliced`` kernel on a
        0.5-70 deg cone grid with ``sigma`` (not ``sigma/sqrt2``) in both
        Gaussians.  This reproduces the PAPER_DRAFT / ``abl_after.log`` numbers.
``S1a`` cone geometry only: 0.5-89 deg grid (so ``pion_alpha_pdf``'s last bin is
        88.2-180 deg instead of 66.8-180 deg) and the per-axis ``sigma/sqrt2``
        in the Gaussian fallback and the muon-decay cone -- still the sampled
        kernel.
``S1``  + the NA61-validated **moment** ``sigma_pi`` Gaussian, i.e. exactly the
        cone ``offaxis_mc`` uses for ``E_off`` (``cone_kernel="moments"``).
``S1b`` + the **arcsin-corrected v2 generator moments** (``m_*_v2.npz``): the
        moment extraction used ``theta = arctan(pT/p_total)`` instead of
        ``arcsin``, so ``sigma_pi`` was 13/11/9/6/2% too narrow at
        0.2/0.3/0.5/1/3 GeV.  Passed explicitly through
        ``channel_shapes(moments=...)``; ``kinematic_kernel``'s own defaults are
        untouched.
``S2``  + the **joint** integral ``J_s/p_axis`` replacing ``E_off x <G>_cone``
        (``joint_cone=True``), still with the single ``channel_fractions``
        ``f_mu[s]`` blend and the biased ``mudecay_shape`` width
        (``joint_channels=False``) -- kept as an A/B row.
``S4``  + the **channel-resolved** joint: MCEq's depth-resolved per-species
        ``{s}_dir`` / ``{s}_k`` / ``{s}_mu`` production profiles as the blend
        weights, with the corrected ``mudecay_shape_mc`` cone per flavour
        (``joint_channels=True``, the default).  Its ``G == 1`` limit is
        ``offaxis_excess_channel_v2.npz`` to machine precision.

Run from ``tools/mceq3d`` with a warm ``.cache3d``::

    python diag_joint_stages.py [--stages S0,S1a,S1,S2] [--n-jobs 16]
"""

from __future__ import annotations

import argparse
import importlib.util  # noqa: F401  (mceq_config import shim; must precede MCEq)
import time
import warnings
from datetime import datetime

import numpy as np

import kinematic_kernel as kk
from mceq3d_flux import MCEq3DFlux, EOFF_TABLE_FLAT

LAT, LON = 36.43, 137.31
DATE = datetime(2020, 1, 1)
CACHE = ".cache3d"
HV_E = (0.3, 0.5, 1.0, 3.0)
EW_E = (0.5, 1.0)
EW_CZ = (0.05, 0.15, 0.25)  # 87 / 81 / 75 deg
CZ = (0.05, 0.15, 0.25, 0.95)
AZ = tuple(np.arange(8) * 45.0)  # includes 90 (E) and 270 (W)
FLAV = (("numu", "total_numu"), ("nue", "total_nue"))

#: arcsin-corrected generator moments (kernel agent, 2026-09-04).  These ARE
#: ``kinematic_kernel._MOMENTS`` since the defaults consolidation, so S1b/S2/S4
#: could equally pass ``cone_moments=None``; the pre-arcsin set has to be named
#: EXPLICITLY (``M_OLD``) for the S0/S1a/S1 rungs, otherwise the ladder would
#: start at the corrected moments and the "+v2 arcsin moments" step (S1 -> S1b)
#: would measure nothing.
M_V2 = dict(kk._MOMENTS)
M_OLD = dict(kk._MOMENTS_LEGACY)

#: Every rung of this ladder was measured with the ORIGINAL flat, pion-only,
#: pre-arcsin ``offaxis_excess.npz`` as the tabulated E_off (that was the only
#: table ``offaxis_factor`` could read).  It is now the non-default A/B table, so
#: it is named here to keep the cumulative numbers comparable.  On these grids
#: (all down-going, 0.3-3 GeV) it only feeds the out-of-range rows anyway.
_FLAT = EOFF_TABLE_FLAT

STAGES = {
    "S0": dict(cone_kernel="legacy", cone_moments=M_OLD, joint_cone=False,
               offaxis_table=_FLAT),
    "S1a": dict(cone_kernel="sampled", cone_moments=M_OLD, joint_cone=False,
                offaxis_table=_FLAT),
    "S1": dict(cone_kernel="moments", cone_moments=M_OLD, joint_cone=False,
               offaxis_table=_FLAT),
    "S1b": dict(cone_kernel="moments", cone_moments=M_V2, joint_cone=False,
                offaxis_table=_FLAT),
    "S2": dict(cone_kernel="moments", cone_moments=M_V2, joint_cone=True,
               joint_channels=False, offaxis_table=_FLAT),
    "S4": dict(cone_kernel="moments", cone_moments=M_V2, joint_cone=True,
               joint_channels=True, offaxis_table=_FLAT),
}


def log_at(y, x, X):
    return float(np.exp(np.interp(np.log(X), np.log(x),
                                  np.log(np.maximum(y, 1e-300)))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", default="S0,S1a,S1,S1b,S2,S4")
    ap.add_argument("--n-jobs", type=int, default=16)
    args = ap.parse_args(argv)
    tags = [t for t in args.stages.split(",") if t]

    warnings.filterwarnings("ignore")
    eng = MCEq3DFlux(base_model="hybrid", primary=("GlobalSplineFitBeta", None),
                     daemonflux_location="kamioka")
    h = dict(np.load("honda_kam.npz"))
    He, Hcz = h["E"], h["czlo"]
    HND = {"numu": h["numu"], "nue": h["nue"]}
    bt = dict(np.load("bartol_kam.npz"))  # Honda-independent cross-check

    cz, az = np.array(CZ), np.array(AZ)
    ie, iw = int(np.argmin(abs(az - 90.0))), int(np.argmin(abs(az - 270.0)))
    hv, ew, cost = {}, {}, {}
    for tag in tags:
        t0 = time.time()
        r = eng.solve(LAT, LON, cz, az, use_cache=True, cache_dir=CACHE,
                      date=DATE, n_jobs=args.n_jobs, **STAGES[tag])
        cost[tag] = time.time() - t0
        e = r["e"]
        for fl, sp in FLAV:
            f = r["flux"][sp]
            hv[(tag, fl)] = [log_at(f[0].mean(0), e, E) / log_at(f[-1].mean(0), e, E)
                             for E in HV_E]
            ew[(tag, fl)] = {
                E: [float(np.interp(E, e, f[i, iw] / np.maximum(f[i, ie], 1e-300)))
                    for i in range(len(EW_CZ))] for E in EW_E}
        print(f"  {tag} done in {cost[tag]:.0f}s", flush=True)

    # Honda references
    hv_h, ew_h = {}, {}
    ihz = int(np.argmin(abs(Hcz - 0.0)))
    ivz = int(np.argmin(abs(Hcz - 0.9)))
    for fl, _ in FLAV:
        nm = HND[fl]
        hv_h[fl] = [log_at(nm[ihz].mean(0), He, E) / log_at(nm[ivz].mean(0), He, E)
                    for E in HV_E]
        ew_h[fl] = {E: [float(np.interp(
            E, He, nm[int(np.argmin(abs(Hcz - c)))].max(0)
            / nm[int(np.argmin(abs(Hcz - c)))].min(0))) for c in EW_CZ]
            for E in EW_E}

    print("\n" + "=" * 78)
    print("horizon/vertical (az-averaged), value and ratio to Honda")
    print("=" * 78)
    for fl, _ in FLAV:
        print(f"\n[{fl}]  " + "".join(f"{E:>16.2f} GeV" for E in HV_E))
        print("Honda    " + "".join(f"{v:>20.3f}" for v in hv_h[fl]))
        for tag in tags:
            print(f"{tag:<8} " + "".join(
                f"{v:>11.3f} ({v / hv_h[fl][i]:5.3f})"
                for i, v in enumerate(hv[(tag, fl)])))

    # Bartol horizon/vertical (azimuth-averaged is not available: the table is
    # the fmin/fmax solar bracket, so we use their mean and the same cz bins)
    bhv = {}
    for fl, key in (("numu", "num"), ("nue", "nue")):
        y = 0.5 * (bt[f"{key}_fmin"] + bt[f"{key}_fmax"])
        ih = int(np.argmin(abs(bt["cz"] - 0.05)))
        iv = int(np.argmin(abs(bt["cz"] - 0.95)))
        bhv[fl] = [log_at(y[ih], bt["E"], E) / log_at(y[iv], bt["E"], E)
                   for E in HV_E]
    print("\n[Bartol horizon/vertical, same bins]")
    for fl, _ in FLAV:
        print(f"{fl:<8} " + "".join(f"{v:>20.3f}" for v in bhv[fl]))
        for tag in tags:
            print(f"  /{tag:<6} " + "".join(
                f"{hv[(tag, fl)][i] / bhv[fl][i]:>20.3f}" for i in range(len(HV_E))))

    print("\n" + "=" * 78)
    print("West/East vs Honda max/min over azimuth")
    print("=" * 78)
    for fl, _ in FLAV:
        for E in EW_E:
            print(f"\n[{fl}] E = {E} GeV     " + "".join(
                f"{np.degrees(np.arccos(c)):>13.0f} deg" for c in EW_CZ))
            print("Honda            " + "".join(f"{v:>17.2f}"
                                                for v in ew_h[fl][E]))
            for tag in tags:
                print(f"{tag:<16} " + "".join(
                    f"{v:>11.2f}({v / ew_h[fl][E][i] - 1:+5.0%})"
                    for i, v in enumerate(ew[(tag, fl)][E])))

    # ---- flux conservation of the joint 3D production factor --------------
    # <F>_Omega at G == 1 (i.e. the channel E_off the joint factor carries).
    # An isotropic primary flux demands ~1; Bartol's all-direction 3D/1D is
    # about +3%, Honda's about -2%.
    print("\n" + "=" * 78)
    print("Solid-angle average of the joint production factor at G == 1")
    print("(down-going hemisphere, uniform in cosZ = uniform in solid angle)")
    print("=" * 78)
    import joint_cone as jc
    import offaxis_mc as ox

    prod = eng.joint_prod(CACHE)
    ep = prod["ep_grid"]
    w = eng.joint_cone_widths(ep, moments=M_V2, cache_dir=CACHE)
    SP = ("total_numu", "total_antinumu", "total_nue", "total_antinue")
    chan = {sp: prod["chan"][jc.CHANNEL_SPECIES_MAP[sp]] for sp in SP}
    smu = {sp: w["mu_nue" if "nue" in sp else "mu_numu"] for sp in SP}
    rcg = np.linspace(0.1, 55.0, 40)
    G1 = {sp: np.ones((len(rcg), len(ep))) for sp in SP}
    fine = eng.finemap_rc(LAT, LON, DATE, cache_dir=CACHE, n_jobs=args.n_jobs)
    czs = np.round(np.arange(0.05, 1.0, 0.1), 2)
    acc = {sp: np.zeros(len(ep)) for sp in SP}
    for c in czs:
        rr = jc.delivered_joint_factor(c, [0.0], prod, fine, rcg, G1,
                                       sigma_pi=w["pi"], sigma_k=w["k"],
                                       channels_by_species=chan,
                                       sigma_mu_by_species=smu)
        for sp in SP:
            acc[sp] += rr["F"][sp][0] / len(czs)
    print(f"{'E [GeV]':>9} " + "".join(f"{sp.replace('total_', ''):>12}"
                                       for sp in SP))
    for E in (0.2, 0.3, 0.5, 1.0, 3.0, 10.0):
        print(f"{E:>9.2f} " + "".join(
            f"{np.interp(np.log(E), np.log(ep), acc[sp]):>12.4f}" for sp in SP))

    print("\ncost per solve [s] (4 zeniths x 8 azimuths, warm caches): " +
          ", ".join(f"{t}={cost[t]:.0f}" for t in tags))
    for a, b in (("S1", "S2"), ("S1b", "S2"), ("S1b", "S4")):
        if a in cost and b in cost:
            print(f"joint-cone overhead {b} vs {a}: "
                  f"{cost[b] / max(cost[a], 1e-9):.2f}x")
    print("DIAG_JOINT_STAGES_DONE")


if __name__ == "__main__":
    main()
