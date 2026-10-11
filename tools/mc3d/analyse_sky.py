"""Read a ``sky.py`` run and produce the milestone-2b tables.

    python analyse_sky.py pilot_total.npz [--collinear-total coll_total.npz]
    python analyse_sky.py RUNDIR --m2b [--tag3 p3d --tag1 p1d --gate g40 g55]
                                       [--out-txt F --out-npz F]

The second form is the 2026-10-08 report (``m2b_report``): it reads the
per-task shard files written by ``sky.run`` and attaches a stratified
delete-a-group jackknife error to every number (see ``JackRun``).

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
    """Pearson r of the 3D and 1D shard sums, per probe bin.

    Legacy (2026-09-10) diagnostic: the shards come from different strata, so
    this r mixes the between-stratum spread into the correlation.  The
    ``--m2b`` report replaces it with the within-stratum jackknife
    correlation (``_rho`` on ``JackRun`` replicates), as it replaces the
    fixed ``rho = 0.6`` of ``ratio_err`` in ``main``'s table (b)."""
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
    ap.add_argument("--m2b", action="store_true",
                    help="treat TOTAL as a run directory holding p3d (and "
                         "p1d, g40, g55) and print the milestone-2b report")
    ap.add_argument("--out-npz", default=None)
    ap.add_argument("--out-txt", default=None)
    ap.add_argument("--tag3", default="p3d")
    ap.add_argument("--tag1", default="p1d")
    ap.add_argument("--gate", nargs="*", default=["g40", "g55"],
                    help="the two patch-gate tags (none to skip)")
    a = ap.parse_args(argv)
    if a.m2b:
        m2b_report(a.total, out_npz=a.out_npz, out_txt=a.out_txt,
                   tag3=a.tag3, tag1=a.tag1, gate=tuple(a.gate))
        return
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



# ===========================================================================
# Milestone 2b report: stratified delete-a-group jackknife over task files
# ===========================================================================
#
# Why a jackknife and not the stored sums of squared weights.  ``s1sq`` etc.
# give the neutrino-level variance, which treats every neutrino as
# independent; neutrinos of one shower are not (same primary weight, nearby
# directions), and a ratio, a nested-cap fit or a cone RMS is not linear in
# the tallies anyway.  Every task of ``sky.run`` writes its own file
# (``<tag>_tasks/t<i>.npz``) and the tasks of a stratum are i.i.d., so the
# tasks of each stratum are dealt round-robin into ``G`` groups and the
# estimator is recomputed with each group deleted, the deleted stratum share
# being restored by ``M_k / (M_k - M_gk)`` (``M`` = showers).  Paired runs
# (``p3d``/``p1d``, ``g40``/``g55``) share the task list, hence the grouping,
# so their correlation is carried automatically.
import os  # noqa: E402

MEASURES = ("s0", "s1", "t0", "t1")
I_S0, I_S1, I_T0, I_T1 = range(4)
HZ_CZ = (10, 13)                  # cos Z bins [0, 0.3) of the 20-bin grid
SP_NAMES = ("nue", "antinue", "numu", "antinumu")   # == scoring.SPECIES order
ENGINE_SP = ("numu", "antinumu", "nue", "antinue")  # offaxis_excess_channel_v2


def _red_az(d):
    return np.stack([np.asarray(d[m]).sum(-1) for m in MEASURES])


def _red_hz(d):
    return np.stack([np.asarray(d[m])[-1, :, :, HZ_CZ[0]:HZ_CZ[1], :]
                     for m in MEASURES])


REDUCERS = {
    "az": _red_az,                                      # (4, cap, sp, E, cz)
    "hz": _red_hz,                                      # (4, sp, E, 3, az)
    "cone_m": lambda d: np.asarray(d["cone_m"]),        # (sp,e10,ch,zp,gen,5)
    "cone_h": lambda d: np.asarray(d["cone_h"]).sum(axis=(3, 4)),
    "hinj": lambda d: np.asarray(d["hinj"]),            # (2,sp,band,czb,th)
    "sh": lambda d: np.asarray(d["shcounts"], float),   # (5,)
}


class JackRun:
    """Per-(jackknife group, stratum) sums of reduced tallies of one run."""

    def __init__(self, outdir, tag, G=20, reducers=None):
        import sky
        self.tag, self.G = tag, int(G)
        tdir = sky.task_dir(outdir, tag)
        self.man = dict(np.load(os.path.join(tdir, "manifest.npz"),
                                allow_pickle=False))
        ntask = int(self.man["ntask"][0])
        reducers = reducers or REDUCERS
        acc, mgs, pos, sids = {}, {}, {}, {}
        cpu, nsh, wall = {}, {}, {}
        for i in range(ntask):
            d = np.load(sky.task_path(outdir, tag, i), allow_pickle=False)
            meta = d["meta"]
            sid = sids.setdefault((str(d["group"][0]), int(meta[1])),
                                  len(sids))
            j = pos.get(sid, 0)
            pos[sid] = j + 1
            key = (j % self.G, sid)
            a = acc.setdefault(key, {})
            for nm, fn in reducers.items():
                v = np.asarray(fn(d), float)
                a[nm] = v.copy() if nm not in a else a[nm] + v
            mgs[key] = mgs.get(key, 0.0) + float(meta[0])
            nsh[sid] = nsh.get(sid, 0.0) + float(meta[0])
            cpu[sid] = cpu.get(sid, 0.0) + float(meta[7])
            wall[sid] = wall.get(sid, 0.0) + float(meta[3])
        self.K = len(sids)
        self.strata = sorted(sids, key=lambda s: sids[s])
        self.M = np.zeros((self.G, self.K))
        for (g, s), v in mgs.items():
            self.M[g, s] = v
        self.S = {}
        for nm in reducers:
            shp = next(iter(acc.values()))[nm].shape
            arr = np.zeros((self.G, self.K) + shp)
            for (g, s), a in acc.items():
                arr[g, s] = a[nm]
            self.S[nm] = arr
        self.cpu = np.array([cpu[i] for i in range(self.K)])
        self.nsh = np.array([nsh[i] for i in range(self.K)])
        self.task_wall = np.array([wall[i] for i in range(self.K)])
        self.ntask = np.array([pos[i] for i in range(self.K)])

    @property
    def n_showers(self):
        return float(self.nsh.sum())

    @property
    def cpu_per_shower(self):
        return float(self.cpu.sum() / self.nsh.sum())

    def total(self, nm, strata=None):
        s = self.S[nm].sum(axis=0)
        return s.sum(axis=0) if strata is None else s[strata]

    def replicates(self, nm, stratum=None):
        """``(G, ...)`` leave-one-group-out totals (one stratum or all)."""
        cache = self.__dict__.setdefault("_rep_cache", {})
        if (nm, stratum) in cache:
            return cache[(nm, stratum)]
        cache[(nm, stratum)] = self._replicates(nm, stratum)
        return cache[(nm, stratum)]

    def _replicates(self, nm, stratum=None):
        S, M = self.S[nm], self.M
        Sk, Mk = S.sum(axis=0), M.sum(axis=0)
        bshape = (-1,) + (1,) * (S.ndim - 2)
        out = []
        for g in range(self.G):
            rem = Mk - M[g]
            fac = np.where(rem > 0, Mk / np.where(rem > 0, rem, 1.0), 0.0)
            r = (Sk - S[g]) * fac.reshape(bshape)
            if stratum is None:
                out.append(r.sum(axis=0))
            else:
                full = Sk.sum(axis=0) - Sk[stratum]
                out.append(full + r[stratum])
        return np.array(out)


def jack(f, runs_names, stratum=None):
    """Estimate, jackknife sigma and replicates of ``f(*arrays)``.

    ``runs_names`` is a list of ``(JackRun, reducer name)``; all runs must
    share the task list (same G, same strata)."""
    full = [r.total(nm) for r, nm in runs_names]
    reps = [r.replicates(nm, stratum) for r, nm in runs_names]
    G = reps[0].shape[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        est = f(*full)
        rep = np.array([f(*[x[g] for x in reps]) for g in range(G)])
    var = (G - 1.0) / G * np.nansum((rep - np.nanmean(rep, axis=0)) ** 2,
                                     axis=0)
    return est, np.sqrt(var), rep


def _sdiv(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b != 0, a / np.where(b != 0, b, 1.0), np.nan)


def _erebin(x, axis, fac=2):
    x = np.moveaxis(x, axis, -1)
    n = x.shape[-1] // fac * fac
    x = x[..., :n].reshape(x.shape[:-1] + (n // fac, fac)).sum(-1)
    return np.moveaxis(x, -1, axis)


def engine_tables(path=None):
    """The factorised engine's channel-v2 table (read-only)."""
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "..", "mceq3d",
                            "offaxis_excess_channel_v2.npz")
    return dict(np.load(path, allow_pickle=False))


def engine_kernel(sigma_deg, n_alpha=44):
    """``(<cos a>, RMS a [deg])`` of the engine's cone kernel: Gaussian on the
    sphere, ``sin a exp(-a^2/sigma^2)`` on ``a in [0.5, 89] deg`` (44 nodes),
    exactly ``joint_cone.gauss_alpha_weights`` on ``joint_cone.alpha_grid``.
    For a wide cone the 89-deg truncation and the ``sin a`` measure make the
    kernel's realised RMS differ from the nominal ``sigma``."""
    a = np.deg2rad(np.linspace(0.5, 89.0, n_alpha))
    out_c, out_r = [], []
    for s in np.atleast_1d(sigma_deg):
        s1 = np.deg2rad(s) / np.sqrt(2.0)
        wa = np.sin(a) * np.exp(-a ** 2 / (2.0 * s1 ** 2))
        wa = wa / wa.sum()
        out_c.append(float(wa @ np.cos(a)))
        out_r.append(float(np.degrees(np.sqrt(wa @ a ** 2))))
    return np.array(out_c), np.array(out_r)


def _rho(rep_a, rep_b):
    da = rep_a - np.nanmean(rep_a, axis=0)
    db = rep_b - np.nanmean(rep_b, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.nansum(da * db, axis=0)
                / np.sqrt(np.nansum(da * da, axis=0) * np.nansum(db * db, axis=0)))


def _cap_fit(r, w, om):
    """Weighted straight line in the cap solid angle; returns the intercept."""
    m = np.isfinite(r) & np.isfinite(w) & (w > 0)
    if m.sum() < 2:
        return np.nan
    return np.polyfit(om[m], r[m], 1, w=np.sqrt(w[m]))[1]


def m2b_report(rundir, G=20, out_npz=None, out_txt=None, cap=-1,
               tag3="p3d", tag1="p1d", gate=("g40", "g55")):
    """All milestone-2b tables from the task files of ``tag3`` (required),
    ``tag1`` (its collinear twin) and the ``gate`` pair (optional) under
    ``rundir``.  Per-azimuth-bin errors are the azimuth-summed jackknife
    errors times sqrt(12): with B = 0 and no cutoff the twelve azimuth bins
    are statistically equivalent, and the summed estimate is the stable one
    (a per-bin jackknife breaks down when a replicate empties a bin)."""
    import sky
    from scoring import CONE_BANDS, CONE_CHANNELS, CONE_EMOM

    lines, summ = [], {}

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    have = {t: os.path.exists(os.path.join(rundir, f"{t}_total.npz"))
            for t in (tag3, tag1) + tuple(gate)}
    P3 = JackRun(rundir, tag3, G)
    P1 = JackRun(rundir, tag1, G) if have[tag1] else None
    d3 = dict(np.load(os.path.join(rundir, f"{tag3}_total.npz"),
                      allow_pickle=False))
    caps, e20 = d3["cap_theta"], d3["e_bins"]
    e10 = e20[::2]
    om = 2.0 * np.pi * (1.0 - np.cos(np.deg2rad(caps)))
    n_prim, cps = P3.n_showers, P3.cpu_per_shower
    wall = float(d3["wall_s"][0])
    nproc = float(d3["nproc"][0])
    naz = d3["s1"].shape[-1]
    summ.update(n_prim=n_prim, cpu_per_shower=cps, wall_s=wall, nproc=nproc,
                caps=caps)

    # ------------------------------------------------------------- cost --
    out("=" * 78)
    out(f"(a) COST [{tag3}]: throughput, scored-shower fraction, entries, "
        "extrapolation")
    out("=" * 78)
    out(f"{tag3}: {n_prim:.0f} showers, CPU {P3.cpu.sum():.0f} s = "
        f"{P3.cpu.sum()/3600:.2f} core-h ({cps*1e3:.2f} ms/shower, "
        f"{1/cps:.0f} showers/s/core); wall {wall:.0f} s on {nproc:.0f} "
        f"workers = {wall*nproc/3600:.2f} core-h")
    sh = P3.S["sh"].sum(axis=0)                     # (K, 5)
    out("\nper stratum: E/nucleon [GeV], showers, tasks, ms/shower, "
        "fraction of showers with >=1 hit: 3D / 1D / either / 3D in E-W "
        "region (cosZ<0.3, 0.3-2 GeV)")
    rows = []
    for s, (grp, k) in enumerate(P3.strata):
        e1, e2 = sky.STRATA[k], sky.STRATA[k + 1]
        f = sh[s, 1:] / max(sh[s, 0], 1.0)
        rows.append([k, e1, e2, P3.nsh[s], P3.ntask[s],
                     P3.cpu[s] / P3.nsh[s] * 1e3, *f])
        out(f"  {grp} s{k}  {e1:7.1f}-{e2:7.1f}  {P3.nsh[s]:9.0f} "
            f"{P3.ntask[s]:5d} {P3.cpu[s]/P3.nsh[s]*1e3:8.2f}   "
            f"{f[0]:.4f} / {f[1]:.4f} / {f[2]:.4f} / {f[3]:.5f}")
    ft = sh.sum(axis=0)
    out(f"  all            {ft[0]:9.0f}        {cps*1e3:8.2f}   "
        f"{ft[1]/ft[0]:.4f} / {ft[2]/ft[0]:.4f} / {ft[3]/ft[0]:.4f} / "
        f"{ft[4]/ft[0]:.5f}")
    summ["strata_table"] = np.array(rows)
    summ["scored_fraction_all"] = ft[1:] / ft[0]

    out("\nnear-horizon bins, Honda's grid (20 E-bins/decade, dcosZ 0.1, "
        "30-deg azimuth), 10-deg cap.  N3/N1 = neutrino-level effective "
        "entries (sum w)^2/sum w^2 per azimuth bin, 3D / correlated 1D; "
        "sPhi, sR = jackknife rel. error per azimuth bin of Phi_3D and of "
        "R = 3D/1D; clus = N3 * sPhi^2 (1 = independent neutrinos); rho = "
        "3D/1D correlation; core-h to reach 3% per azimuth bin on Phi_3D / "
        "on R, without azimuthal reuse and with N_az = 12")
    out(f"{'species':9s} {'E':>5s} {'cosZ':>8s} {'N3':>6s} {'N1':>6s} "
        f"{'/1e6pr':>6s} {'sPhi':>6s} {'clus':>5s} {'rho':>5s} {'sR':>6s} "
        f"{'R':>6s} {'chPhi':>7s} {'chPhi12':>7s} {'chR':>7s} {'chR12':>7s}")
    s1, s1sq, t1, t1sq = d3["s1"], d3["s1sq"], d3["t1"], d3["t1sq"]
    cost_rows = []
    for s in (14, -14, 12, -12):
        k = SPECIES.index(s)
        for e in (0.3, 0.5, 1.0):
            ie = e_index(e20, e)
            for izz in range(3):
                iz = HZ_CZ[0] + izz
                n3 = float(neff(s1[cap, k, ie, iz].sum(),
                                s1sq[cap, k, ie, iz].sum())) / naz
                n1 = float(neff(t1[cap, k, ie, iz].sum(),
                                t1sq[cap, k, ie, iz].sum())) / naz
                a, sa, ra = jack(lambda h: h[I_S1, k, ie, izz, :].sum(),
                                 [(P3, "hz")])
                b, sb, rb_ = jack(lambda h: h[I_T1, k, ie, izz, :].sum(),
                                  [(P3, "hz")])
                rr, srr, _ = jack(lambda h: h[I_S1, k, ie, izz, :].sum()
                                  / h[I_T1, k, ie, izz, :].sum(),
                                  [(P3, "hz")])
                rho = float(_rho(ra, rb_))
                v_phi = naz * float(sa / a) ** 2       # per azimuth bin
                v_r = naz * float(srr / rr) ** 2
                kh = n_prim * cps / 3600.0 / 0.03 ** 2
                row = [s, e, iz, n3, n1, n3 / n_prim * 1e6, np.sqrt(v_phi),
                       n3 * v_phi, rho, np.sqrt(v_r), float(rr),
                       kh * v_phi, kh * v_phi / naz, kh * v_r, kh * v_r / naz]
                cost_rows.append(row)
                out(f"{NAME[s]:9s} {e:5.2f} {d3['cz_bins'][iz]:3.1f}-"
                    f"{d3['cz_bins'][iz+1]:3.1f} {n3:6.1f} {n1:6.1f} "
                    f"{n3/n_prim*1e6:6.2f} {np.sqrt(v_phi):6.3f} "
                    f"{n3*v_phi:5.2f} {rho:5.2f} {np.sqrt(v_r):6.3f} "
                    f"{float(rr):6.3f} {row[11]:7.0f} {row[12]:7.0f} "
                    f"{row[13]:7.0f} {row[14]:7.0f}")
    cost_rows = np.array(cost_rows)
    summ["cost_rows"] = cost_rows
    summ["cost_cols"] = np.array(
        ["pdg", "E", "iz", "N3_az", "N1_az", "N3_per_1e6", "sPhi_az",
         "clus", "rho", "sR_az", "R", "chPhi", "chPhi12", "chR", "chR12"])
    ew = cost_rows[(cost_rows[:, 2] <= HZ_CZ[0] + 1)]
    for col, what in ((11, "Phi_3D"), (13, "R")):
        j = int(np.nanargmax(ew[:, col]))
        out(f"BINDING E-W bin for 3% on {what} per azimuth bin (cosZ < 0.2, "
            f"0.3-1 GeV, 20/decade): {NAME[int(ew[j,0])]} {ew[j,1]} GeV "
            f"cosZ {0.1*(int(ew[j,2])-10):.1f}-{0.1*(int(ew[j,2])-9):.1f}: "
            f"{ew[j,col]:.0f} core-h without azimuthal reuse "
            f"({ew[j,col]*3600/cps:.2e} showers), {ew[j,col+1]:.0f} core-h "
            f"with N_az = 12; median over the E-W bins "
            f"{np.nanmedian(ew[:, col]):.0f} / {np.nanmedian(ew[:, col+1]):.0f}")
        summ[f"ew_binding_{what}"] = ew[j]

    # 5% survey: R per (species, 10/decade E, dcosZ 0.1, 30-deg azimuth)
    def r10(az):
        x = _erebin(az, 3)
        return x[I_S1, cap] / x[I_T1, cap]          # (sp, e10, cz)
    estR, sigR, _ = jack(r10, [(P3, "az")])
    rel = sigR / estR
    x3 = _erebin(np.stack([d3["s1"].sum(-1), d3["s1sq"].sum(-1)]), 3)
    x1 = _erebin(np.stack([d3["t1"].sum(-1), d3["t1sq"].sum(-1)]), 3)
    n3s = neff(x3[0, cap], x3[1, cap])
    n1s = neff(x1[0, cap], x1[1, cap])
    for lo, hi in ((0.1, 3.0), (0.1, 10.0)):
        ie0, ie1 = e_index(e10, lo), e_index(e10, hi * 0.999)
        sub = rel[:, ie0:ie1 + 1, 10:20]
        pop = (n3s[:, ie0:ie1 + 1, 10:20] >= 10) & \
            (n1s[:, ie0:ie1 + 1, 10:20] >= 10) & np.isfinite(sub)
        v = np.where(pop, sub, np.nan) ** 2
        jj = np.unravel_index(np.nanargmax(v), v.shape)
        vmax, v90 = float(v[jj]), float(np.nanpercentile(v, 90))
        kh = n_prim * cps / 3600.0 / 0.05 ** 2
        out(f"5% SURVEY ({lo}-{hi} GeV, 10 E-bins/decade, dcosZ 0.1, 12 "
            f"azimuth bins, 4 species, down-going; {int(pop.sum())} of "
            f"{pop.size} bins populated enough (N_eff >= 10) to measure): "
            f"binding bin {SP_NAMES[jj[0]]} "
            f"E={np.sqrt(e10[ie0+jj[1]]*e10[ie0+jj[1]+1]):.2f} GeV cosZ "
            f"{0.1*jj[2]:.1f}-{0.1*jj[2]+0.1:.1f}: {kh*vmax*naz:.0f} core-h "
            f"without reuse / {kh*vmax:.0f} with N_az = 12; 90th-percentile "
            f"bin {kh*v90*naz:.0f} / {kh*v90:.0f}")
        summ[f"survey_{lo}_{hi}"] = np.array([kh * vmax * naz, kh * vmax,
                                              kh * v90 * naz, kh * v90,
                                              pop.sum(), pop.size])
    summ["relerr_R10_az_summed"] = rel

    # Neyman re-allocation of the strata for the binding E-W bin (R)
    jb = summ["ew_binding_R"]
    s_b, e_b, iz_b = int(jb[0]), float(jb[1]), int(jb[2])
    kb, ieb, izzb = SPECIES.index(s_b), e_index(e20, e_b), iz_b - HZ_CZ[0]

    def rb(h):
        return h[I_S1, kb, ieb, izzb, :].sum() / h[I_T1, kb, ieb, izzb, :].sum()
    v_k = []
    for st in range(P3.K):
        _, sg, _ = jack(rb, [(P3, "hz")], stratum=st)
        v_k.append(float(sg) ** 2)
    v_k = np.array(v_k)
    c_k = P3.cpu / P3.nsh
    pershower = v_k * P3.nsh                     # variance x n_k
    c_now = float((c_k * P3.nsh).sum())
    c_opt = float(np.sqrt(pershower * c_k).sum() ** 2 / max(v_k.sum(), 1e-300))
    nk_opt = np.sqrt(pershower / c_k)
    nk_opt = nk_opt / nk_opt.sum()
    out(f"per-stratum share of var(R) in that bin: "
        f"{np.round(v_k / v_k.sum(), 3).tolist()}; Neyman allocation "
        f"n_k/N = {np.round(nk_opt, 3).tolist()} (now "
        f"{np.round(P3.nsh / P3.nsh.sum(), 3).tolist()}) would cut its cost "
        f"by x{c_now / c_opt:.2f}")
    summ["neyman"] = np.array([c_now / c_opt, *nk_opt, *(v_k / v_k.sum())])

    # ------------------------------------------- 3D/1D vs cos Z ---------
    out("\n" + "=" * 78)
    out("(b) azimuth-averaged 3D/1D (specific intensity) vs cos Z, 10-deg cap;"
        " 10 E-bins/decade")
    out("=" * 78)
    eng = engine_tables()
    ee = eng["e"]
    E_B = (0.3, 0.5, 1.0)

    def rcap(az, ie, sp):
        x = _erebin(az, 3)
        return x[I_S1, :, sp, ie, 10:20] / x[I_T1, :, sp, ie, 10:20]
    btab = {}
    for s in (14, 12, -14, -12):
        sp = SPECIES.index(s)
        esp = ENGINE_SP.index(NAME[s])
        for e in E_B:
            ie = e_index(e10, e)
            iee = int(np.argmin(np.abs(ee - np.sqrt(e10[ie] * e10[ie + 1]))))
            r_all, s_all, _ = jack(lambda az: rcap(az, ie, sp), [(P3, "az")])
            wfit = 1.0 / np.maximum(s_all, 1e-12) ** 2       # (cap, cz)

            def ext(az):
                r = rcap(az, ie, sp)
                return np.array([_cap_fit(r[:, i], wfit[:, i], om)
                                 for i in range(r.shape[1])])
            r_ext, s_ext, _ = jack(ext, [(P3, "az")])
            if P1 is not None:
                def rtrue(a3, a1):
                    x3, x1 = _erebin(a3, 3), _erebin(a1, 3)
                    return x3[I_S1, cap, sp, ie, 10:20] / \
                        x1[I_S1, cap, sp, ie, 10:20]

                def resid(a3, a1):
                    x3, x1 = _erebin(a3, 3), _erebin(a1, 3)
                    return x3[I_T1, cap, sp, ie, 10:20] / \
                        x1[I_S1, cap, sp, ie, 10:20]
                r_tr, s_tr, _ = jack(rtrue, [(P3, "az"), (P1, "az")])
                r_rs, s_rs, _ = jack(resid, [(P3, "az"), (P1, "az")])
            else:
                r_tr = s_tr = r_rs = s_rs = np.full(10, np.nan)
            btab[(s, e)] = np.array([r_all[cap], s_all[cap], r_ext, s_ext,
                                     r_tr, s_tr, r_rs, s_rs,
                                     eng["E_off_s"][esp, :, iee]])
    summ["btab_keys"] = np.array([[s, e] for (s, e) in btab])
    summ["btab"] = np.array(list(btab.values()))
    for s in (14, 12):
        out(f"\n{NAME[s]}: R = in-run 3D/1D | ext = nested-cap Omega_D->0 | "
            f"true = {tag3}/{tag1} (both complete runs) | 1Dres = in-run 1D / "
            f"true 1D | engine E_off_s(G=1)")
        for e in E_B:
            t = btab[(s, e)]
            out(f"  E = {e} GeV  (10/decade bin centre "
                f"{np.sqrt(e10[e_index(e10,e)]*e10[e_index(e10,e)+1]):.3f})")
            out(f"  {'cosZ':>8s} {'R':>13s} {'ext':>13s} {'true':>13s} "
                f"{'1Dres':>13s} {'engine':>7s}")
            for i in range(10):
                out(f"  {0.1*i:3.1f}-{0.1*i+0.1:3.1f} "
                    f"{t[0][i]:6.3f}+-{t[1][i]:5.3f} "
                    f"{t[2][i]:6.3f}+-{t[3][i]:5.3f} "
                    f"{t[4][i]:6.3f}+-{t[5][i]:5.3f} "
                    f"{t[6][i]:6.3f}+-{t[7][i]:5.3f} {t[8][i]:7.3f}")

    # horizon/vertical contrast of the 3D/1D factor (what moves H/V)
    out("\nhorizon/vertical contrast R(cosZ 0-0.1) / R(cosZ 0.9-1.0): MC "
        "true (p3d/p1d) | MC in-run | engine E_off_s")
    hv = []
    for s in (14, 12, -14, -12):
        sp = SPECIES.index(s)
        esp = ENGINE_SP.index(NAME[s])
        for e in E_B:
            ie = e_index(e10, e)
            iee = int(np.argmin(np.abs(ee - np.sqrt(e10[ie] * e10[ie + 1]))))

            def c_in(az):
                x = _erebin(az, 3)
                r = x[I_S1, cap, sp, ie, 10:20] / x[I_T1, cap, sp, ie, 10:20]
                return r[0] / r[-1]
            ci, sci, _ = jack(c_in, [(P3, "az")])
            if P1 is not None:
                def c_tr(a3, a1):
                    x3, x1 = _erebin(a3, 3), _erebin(a1, 3)
                    r = x3[I_S1, cap, sp, ie, 10:20] / x1[I_S1, cap, sp, ie,
                                                          10:20]
                    return r[0] / r[-1]
                ct, sct, _ = jack(c_tr, [(P3, "az"), (P1, "az")])
            else:
                ct = sct = np.nan
            ce = eng["E_off_s"][esp, 0, iee] / eng["E_off_s"][esp, -1, iee]
            hv.append([s, e, float(ct), float(sct), float(ci), float(sci), ce])
            out(f"  {NAME[s]:9s} {e:4.1f} GeV: {float(ct):6.3f}+-{float(sct):5.3f}"
                f" | {float(ci):6.3f}+-{float(sci):5.3f} | {ce:6.3f}")
    summ["hv_contrast"] = np.array(hv)

    # Honda 2004's statement is band-integrated: horizontal 3D/1D > 1.5 at
    # 0.1-0.32 GeV and <~1.1 at 1-3.2 GeV (his numbers include the cutoff)
    out("\nHonda bands: horizontal 3D/1D integrated over E_nu (in-run | true)")
    hb = []
    for lo, hi in ((0.1, 0.3163), (0.3163, 1.0), (1.0, 3.163)):
        i0, i1 = e_index(e20, lo * 1.0001), e_index(e20, hi * 0.999)
        for spn, sps in (("numu+antinumu", (14, -14)), ("nue+antinue", (12, -12)),
                         ("all four", (14, -14, 12, -12))):
            ks = [SPECIES.index(x) for x in sps]
            for iz in (10, 11):
                def fin(az):
                    return az[I_S1, cap][ks][:, i0:i1 + 1, iz].sum() / \
                        az[I_T1, cap][ks][:, i0:i1 + 1, iz].sum()
                v1, s1_, _ = jack(fin, [(P3, "az")])
                if P1 is not None:
                    def ftr(a3, a1):
                        return a3[I_S1, cap][ks][:, i0:i1 + 1, iz].sum() / \
                            a1[I_S1, cap][ks][:, i0:i1 + 1, iz].sum()
                    v2, s2_, _ = jack(ftr, [(P3, "az"), (P1, "az")])
                else:
                    v2 = s2_ = np.nan
                hb.append([lo, hi, len(sps), iz, float(v1), float(s1_),
                           float(v2), float(s2_)])
                out(f"  {lo:.2f}-{hi:.2f} GeV {spn:14s} cosZ "
                    f"{0.1*(iz-10):.1f}-{0.1*(iz-9):.1f}: {float(v1):6.3f}+-"
                    f"{float(s1_):5.3f} | {float(v2):6.3f}+-{float(s2_):5.3f}")
    summ["honda_bands"] = np.array(hb)

    # G7: nested-cap consistency.  With B = 0 and no cutoff nothing depends
    # on the cap size, so the hemisphere-summed R must be cap-independent.
    out("\nG7 nested caps: hemisphere-summed in-run R(cap) / R(10 deg) at "
        "0.3 / 0.5 / 1 GeV (B = 0: must be 1)")
    g7 = []
    for s in (14, 12):
        sp = SPECIES.index(s)
        for e in E_B:
            ie = e_index(e10, e)

            def rc(az):
                x = _erebin(az, 3)
                r = x[I_S1, :, sp, ie, 10:20].sum(-1) / \
                    x[I_T1, :, sp, ie, 10:20].sum(-1)
                return r / r[-1]
            v, sv, _ = jack(rc, [(P3, "az")])
            g7.append([s, e, *v, *sv])
            out(f"  {NAME[s]:5s} {e:4.1f} GeV: " + "  ".join(
                f"{caps[i]:4.1f}deg {v[i]:5.3f}+-{sv[i]:5.3f}"
                for i in range(len(caps))))
    summ["g7"] = np.array(g7)

    # ------------------------------------------- the two measures -------
    out("\n" + "=" * 78)
    out("(c) solid-angle averages of 3D/1D vs energy (down-going, 10-deg cap)")
    out("=" * 78)
    E_C = (0.2, 0.3, 0.5, 1.0, 2.0, 3.0)
    czc = np.arange(10) * 0.1 + 0.05

    def avgs(a3, a1=None):
        x3 = _erebin(a3, 3)
        if a1 is None:
            num1, den1 = x3[I_S1, cap, :, :, 10:20], x3[I_T1, cap, :, :, 10:20]
            num0, den0 = x3[I_S0, cap, :, :, 10:20], x3[I_T0, cap, :, :, 10:20]
        else:
            x1 = _erebin(a1, 3)
            num1, den1 = x3[I_S1, cap, :, :, 10:20], x1[I_S1, cap, :, :, 10:20]
            num0, den0 = x3[I_S0, cap, :, :, 10:20], x1[I_S0, cap, :, :, 10:20]
        F = num1 / den1                                     # (sp, e10, cz)
        return np.stack([F.mean(-1), (F * czc).sum(-1) / czc.sum(),
                         num1.sum(-1) / den1.sum(-1),
                         num0.sum(-1) / den0.sum(-1), F[..., -1]])
    a_in, sa_in, _ = jack(avgs, [(P3, "az")])
    if P1 is not None:
        a_tr, sa_tr, _ = jack(lambda x, y: avgs(x, y), [(P3, "az"), (P1, "az")])
    else:
        a_tr = sa_tr = np.full_like(a_in, np.nan)

    def mcos(m):
        w = m[..., 0].sum(axis=(2, 4))                     # (sp, e10, zp)
        c = m[..., 4].sum(axis=(2, 4))
        return np.concatenate([(c / w), (c.sum(-1) / w.sum(-1))[..., None]],
                              axis=-1)                     # zp0..2, all
    mc_cos, smc_cos, _ = jack(mcos, [(P3, "cone_m")])
    eo = eng["E_off_s"]                                      # (4, 10, 30)
    e_avg = np.stack([eo.mean(1), (eo * czc[None, :, None]).sum(1) / czc.sum(),
                      eo[:, -1, :]])                         # (3, 4, 30)
    summ.update(avg_inrun=a_in, avg_inrun_err=sa_in, avg_true=a_tr,
                avg_true_err=sa_tr, cone_mean_cos=mc_cos,
                cone_mean_cos_err=smc_cos, engine_avgs=e_avg)
    lab = ("<F>_Omega (plain, unweighted over cosZ)",
           "<F>_cos (cosZ-weighted)",
           "sum s1/sum t1 (Phi_1D-weighted intensity)",
           "sum s0/sum t0 (neutrinos through the ground)",
           "F(vertical, cosZ 0.9-1)")
    for s in (14, 12):
        sp = SPECIES.index(s)
        esp = ENGINE_SP.index(NAME[s])
        out(f"\n{NAME[s]}   (true = p3d/p1d; in-run = vs correlated 1D; "
            f"engine = offaxis_excess_channel_v2 E_off_s at G == 1)")
        out("  " + f"{'E [GeV]':40s}" + "".join(f"{e:>15.2f}" for e in E_C))
        for q in range(5):
            row_t, row_i, row_e = "", "", ""
            for e in E_C:
                ie = e_index(e10, e)
                row_t += f"{a_tr[q, sp, ie]:9.3f}+-{sa_tr[q, sp, ie]:4.3f}"
                row_i += f"{a_in[q, sp, ie]:9.3f}+-{sa_in[q, sp, ie]:4.3f}"
                iee = int(np.argmin(np.abs(ee - np.sqrt(e10[ie]
                                                        * e10[ie + 1]))))
                qe = {0: 0, 1: 1, 4: 2}.get(q)
                row_e += (f"{e_avg[qe, esp, iee]:15.3f}" if qe is not None
                          else f"{'':>15s}")
            out(f"  {lab[q]:40s}")
            out(f"  {'   true':40s}{row_t}")
            out(f"  {'   in-run':40s}{row_i}")
            if q in (0, 1, 4):
                out(f"  {'   engine':40s}{row_e}")
        row = ""
        for e in E_C:
            ie = e_index(CONE_EMOM, e)
            row += f"{mc_cos[sp, ie, 2]:9.3f}+-{smc_cos[sp, ie, 2]:4.3f}"
        out(f"  {'<cos alpha> MC, near-vertical primaries':40s}{row}")
        row = ""
        for e in E_C:
            ie = e_index(CONE_EMOM, e)
            row += f"{mc_cos[sp, ie, 3]:9.3f}+-{smc_cos[sp, ie, 3]:4.3f}"
        out(f"  {'<cos alpha> MC, all primaries':40s}{row}")

    # ------------------------------------------- the cone ---------------
    out("\n" + "=" * 78)
    out("(d) production cone: RMS space angle primary->neutrino [deg]")
    out("=" * 78)

    def rms_ch(m):
        # (sp, e10, ch, [all gens, gen<=1]) ; plus all channels
        w = m[..., 0].sum(axis=3)                            # (sp,e,ch,gen)
        a2 = m[..., 3].sum(axis=3)
        r_all = np.sqrt(a2.sum(-1) / w.sum(-1))
        r_g1 = np.sqrt(a2[..., 0] / w[..., 0])
        r_tot = np.sqrt(a2.sum(axis=(-1, -2)) / w.sum(axis=(-1, -2)))
        frac = w.sum(-1) / w.sum(axis=(-1, -2))[..., None]
        fg1 = w[..., 0].sum(-1) / w.sum(axis=(-1, -2))
        return np.concatenate([r_all, r_g1, frac, r_tot[..., None],
                               fg1[..., None]], axis=-1)
    rc, src, _ = jack(rms_ch, [(P3, "cone_m")])
    nch = len(CONE_CHANNELS)
    summ.update(cone_rms=rc, cone_rms_err=src,
                # the measured kernel itself, for the factorised engine:
                # h[sp, band, channel, alpha 1-deg] (summed over primary
                # zenith class and generation) and the exact moments
                # m[sp, e10, channel, zp, gen, (w, w2, wa, wa2, wcos)]
                cone_hist=P3.total("cone_h").astype(np.float32),
                cone_moments=P3.total("cone_m").astype(np.float32),
                cone_bands=np.asarray(CONE_BANDS),
                cone_emom=np.asarray(CONE_EMOM),
                cone_channels=np.array(CONE_CHANNELS),
                species=np.array(SP_NAMES))
    eng_sig = {"pi": eng["sigma_pi"], "K": eng["sigma_k"],
               "mu_numu": eng["sigma_mu_numu"], "mu_nue": eng["sigma_mu_nue"]}
    E_D = (0.141, 0.224, 0.282, 0.355, 0.447, 0.708, 1.12, 1.78, 2.82)
    for s in (14, -14, 12, -12):
        sp = SPECIES.index(s)
        mu_key = "mu_nue" if abs(s) == 12 else "mu_numu"
        out(f"\n{NAME[s]}:  MC all generations / MC first-interaction only / "
            f"engine nominal sigma / engine kernel realised RMS; f = channel "
            f"weight fraction")
        for chn, ek in (("pi", "pi"), ("K", "K"), ("mu", mu_key)):
            c = CONE_CHANNELS.index(chn)
            ename = ("sigma_" + ek).replace("sigma_K", "sigma_k")
            out(f"  channel {chn} (engine {ename})")
            for e in E_D:
                ie = e_index(CONE_EMOM, e)
                iee = int(np.argmin(np.abs(ee - e)))
                sg = eng_sig[ek][iee]
                _, rr = engine_kernel(sg)
                out(f"    E={e:5.3f}  MC {rc[sp, ie, c]:6.2f}+-{src[sp, ie, c]:4.2f}"
                    f"  gen1 {rc[sp, ie, nch + c]:6.2f}+-"
                    f"{src[sp, ie, nch + c]:4.2f}  engine {sg:6.2f} "
                    f"(realised {rr[0]:6.2f})  ratio MC/realised "
                    f"{rc[sp, ie, c]/rr[0]:5.3f}  f={rc[sp, ie, 2*nch + c]:.3f}")
        out(f"  all channels: " + "  ".join(
            f"E={e:.3f}: {rc[sp, e_index(CONE_EMOM, e), 3*nch]:5.2f}"
            for e in E_D))
        out(f"  first-interaction fraction of the neutrinos: " + "  ".join(
            f"{rc[sp, e_index(CONE_EMOM, e), 3*nch+1]:.2f}" for e in E_D))
    # shape, not just width: quantiles of the alpha distribution per band
    # against the engine's Gaussian-on-the-sphere kernel at the band centre
    out("\n  kernel SHAPE per E_nu band: MC median / q68 / q90 / RMS [deg] vs "
        "the engine kernel's median / q68 / q90 (Gaussian: median/RMS = 0.83)")
    hall = P3.total("cone_h")                      # (sp, band, ch, alpha)
    acen = np.arange(hall.shape[-1]) + 0.5
    qrows = []
    for s, chans in ((14, (("pi", "pi"), ("mu", "mu_numu"))),
                     (12, (("mu", "mu_nue"),))):
        sp = SPECIES.index(s)
        for chn, ek in chans:
            c = CONE_CHANNELS.index(chn)
            for ib in range(len(CONE_BANDS) - 1):
                row = hall[sp, ib, c]
                if row.sum() <= 0:
                    continue
                cdf = np.cumsum(row) / row.sum()
                mq = [float(np.interp(f, cdf, acen)) for f in (0.5, 0.68, 0.9)]
                rms = float(np.sqrt((row * acen ** 2).sum() / row.sum()))
                ec = np.sqrt(CONE_BANDS[ib] * CONE_BANDS[ib + 1])
                sg = float(np.interp(np.log(ec), np.log(ee), eng_sig[ek]))
                al = np.linspace(0.5, 89.0, 2000)
                s1_ = np.deg2rad(sg) / np.sqrt(2.0)
                wk = np.sin(np.deg2rad(al)) * np.exp(-np.deg2rad(al) ** 2
                                                     / (2 * s1_ ** 2))
                ck = np.cumsum(wk) / wk.sum()
                eq = [float(np.interp(f, ck, al)) for f in (0.5, 0.68, 0.9)]
                qrows.append([s, c, ib, *mq, rms, *eq])
                out(f"  {NAME[s]:5s} {chn:3s} {CONE_BANDS[ib]}-"
                    f"{CONE_BANDS[ib+1]} GeV: MC {mq[0]:5.1f}/{mq[1]:5.1f}/"
                    f"{mq[2]:5.1f}/{rms:5.1f} (median/RMS {mq[0]/rms:.2f}) | "
                    f"engine {eq[0]:5.1f}/{eq[1]:5.1f}/{eq[2]:5.1f} | MC/engine "
                    f"{mq[0]/eq[0]:.2f} / {mq[1]/eq[1]:.2f} / {mq[2]/eq[2]:.2f}")
    summ["cone_quantiles"] = np.array(qrows)
    # engine kernel <cos alpha> on its own blend is not available without
    # its channel fractions; report per-channel engine <cos a>
    ec = {kk: engine_kernel(v)[0] for kk, v in eng_sig.items()}
    summ["engine_kernel_cos"] = np.array([ec["pi"], ec["K"], ec["mu_numu"],
                                          ec["mu_nue"]])
    summ["engine_e"] = ee

    # ------------------------------------------- the patch gate ---------
    if gate and have[gate[0]] and have[gate[1]]:
        out("\n" + "=" * 78)
        out("(e) injection-patch saturation (G8): theta_inj 40 vs 55 deg, "
            "2.5e5 showers each, same seeds")
        out("=" * 78)
        A = JackRun(rundir, gate[0], G)
        B = JackRun(rundir, gate[1], G)
        grows = []
        for spn, sps in (("numu+antinumu", (14, -14)), ("nue+antinue", (12, -12))):
            ks = [SPECIES.index(x) for x in sps]
            for lo, hi in ((0.3, 0.5), (0.5, 1.0), (1.0, 3.0)):
                i0, i1 = e_index(e20, lo), e_index(e20, hi * 0.999)
                for izz in range(3):
                    def f(h40, h55):
                        a = h40[I_S1][ks][:, i0:i1 + 1, izz, :].sum()
                        b = h55[I_S1][ks][:, i0:i1 + 1, izz, :].sum()
                        return b / a
                    r, sr, _ = jack(f, [(A, "hz"), (B, "hz")])
                    grows.append([lo, hi, izz, float(r), float(sr)])
                    out(f"  {spn:14s} {lo}-{hi} GeV cosZ {0.1*izz:.1f}-"
                        f"{0.1*izz+0.1:.1f}: Phi3D(55)/Phi3D(40) = "
                        f"{float(r):.3f} +- {float(sr):.3f}")
        summ["gate_ratio"] = np.array(grows)
        # direct measurement: where do the near-horizon 3D hits come from?
        out("\n  injection-point distance from the site for 3D cap hits "
            "(weight fraction beyond theta, all species):")
        thr = (20, 25, 30, 35, 40, 45, 50)
        tail = []
        th_of = {gate[1]: float(B.man["theta_inj"][0]),
                 tag3: float(P3.man["theta_inj"][0]),
                 gate[0]: float(A.man["theta_inj"][0])}
        for tg, run in ((gate[1], B), (tag3, P3), (gate[0], A)):
            h = run.total("hinj")[0].sum(axis=0)       # (band, czb, th)
            for ib in range(len(CONE_BANDS) - 1):
                for izb in range(4):
                    row = h[ib, izb]
                    tot = row.sum()
                    if tot <= 0:
                        continue
                    fr = [row[t:].sum() / tot for t in thr]
                    cum = np.cumsum(row) / tot
                    q99 = int(np.searchsorted(cum, 0.99))
                    q999 = int(np.searchsorted(cum, 0.999))
                    tail.append([th_of[tg], ib,
                                 izb, *fr, q99, q999])
                    if izb <= 1 or tg == gate[1]:
                        out(f"  {tg} {CONE_BANDS[ib]}-{CONE_BANDS[ib+1]} GeV "
                            f"cosZ {DualCapScorerCZ[izb]}: "
                            + " ".join(f">{t}:{x:.1e}" for t, x in
                                       zip(thr, fr))
                            + f"  99%<{q99+1} deg, 99.9%<{q999+1} deg")
        summ["gate_tail"] = np.array(tail)

    # ------------------------------------------- absolute flux ----------
    hp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                      "mceq3d", "honda_kam.npz")
    if os.path.exists(hp):
        out("\n" + "=" * 78)
        out("(f) absolute normalisation: MC Phi_3D (no cutoff, GSF, E_N <= "
            "top stratum) / HKKM2014 Kamioka (with cutoff), azimuth-averaged")
        out("=" * 78)
        h = np.load(hp)
        hname = {14: "numu", -14: "numubar", 12: "nue", -12: "nuebar"}
        absr = []
        for s_ in (14, 12, -14, -12):
            k = SPECIES.index(s_)

            def phi(az):
                x = _erebin(az, 3)[I_S1, cap, k]          # (e10, cz)
                return x / (2.0 * np.pi * 0.1 * np.diff(e10)[:, None])
            f, sf, _ = jack(phi, [(P3, "az")])
            row = []
            for e in (0.3, 0.5, 1.0, 2.0, 3.0, 5.0):
                ie = e_index(e10, e)
                for iz in (10, 14, 19):
                    ee = np.geomspace(e10[ie], e10[ie + 1], 50)
                    fh = np.exp(np.interp(np.log(ee), np.log(h["E"]), np.log(
                        h[hname[s_]][iz].mean(axis=0) * 1e-4)))
                    ph = np.trapezoid(fh, ee) / (e10[ie + 1] - e10[ie])
                    row.append((f[ie, iz] / ph, sf[ie, iz] / ph))
                    absr.append([s_, e, iz, f[ie, iz] / ph, sf[ie, iz] / ph])
            out(f"  {NAME[s_]:9s} cosZ 0-0.1 / 0.4-0.5 / 0.9-1 at E = 0.3, 0.5,"
                " 1, 2, 3, 5 GeV:")
            out("     " + " | ".join(" ".join(f"{v:4.2f}+-{e:4.2f}" for v, e in
                                             row[3 * i:3 * i + 3])
                                    for i in range(6)))
        summ["abs_vs_honda"] = np.array(absr)

    if out_txt:
        with open(out_txt, "w") as fh:
            fh.write("\n".join(lines) + "\n")
    if out_npz:
        np.savez_compressed(out_npz, **{k: np.asarray(v) for k, v in
                                        summ.items()})
    return summ, lines


DualCapScorerCZ = ("0.0-0.1", "0.1-0.2", "0.2-0.5", "0.5-1.0")


if __name__ == "__main__":
    main()
