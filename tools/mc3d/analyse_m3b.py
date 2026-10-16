"""Milestone-3b analysis: the East-West species pattern, with jackknife errors,
against Honda (HKKM2014) and the factorised engine; the 3D/1D ratios against
both 1D references; the 1D-cutoff gate; the allocation (task B).

    python analyse_m3b.py RUNDIR --tag m3b [--engine ENGINE.npz]
                          [--out-txt F --out-npz F]

Observables (``tools/mceq3d/diag_ew_charge_fourier``'s helpers, Honda's 12
azimuth bins, compass convention ``az = 180 - az_Honda`` applied to Honda's
table): for each species, energy band and ``cos Z`` band the azimuthal
pattern of the band-integrated specific intensity is decomposed into
``a0 [1 + a1 cos(phi - phi1) + a2 cos 2(phi - phi2)]`` (bin-width
deconvolved), and reported as ``a1/a0`` (the East-West dipole amplitude),
``dphi = phi1 - geomagnetic West`` (the shift-sensitive phase), ``a2/a0``,
``W/E`` (two-harmonic reconstruction at the geomagnetic West and East) and
the legacy ``max/min`` over the 12 bins, plus North and South relative to
the mean (``f_N/a0`` from the bins centred at 345 and 15 deg, ``f_S/a0`` from
165 and 195 deg).

Energy bands (edges on the 20/decade grid, centres on Honda's points):
0.3 GeV = [10^-0.65, 10^-0.40], 0.5 GeV = [10^-0.40, 10^-0.20],
1.0 GeV = [10^-0.10, 10^0.10].  Honda's table is integrated over the same
bands (log-log), so the comparison is like for like.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import analyse_sky as A
from scoring import SPECIES

SP_NAMES = ("nue", "antinue", "numu", "antinumu")      # == scoring.SPECIES
HONDA_KEY = {"nue": "nue", "antinue": "nuebar", "numu": "numu",
             "antinumu": "numubar"}
E_BANDS = {0.3: (7, 12), 0.5: (12, 16), 1.0: (18, 22)}  # [i0, i1) of E_BINS
CZ_BANDS = ((0.0, 0.1), (0.1, 0.2), (0.2, 0.3))
CZ0 = 10                                               # cos Z bin of [0, 0.1)
AZ_C = 15.0 + 30.0 * np.arange(12)
TAG_IDX = {"s": 0, "n": 1, "t": 2, "v": 3}
OBS = ("a1_rel", "dphi", "a2_rel", "WE", "maxmin", "fN", "fS", "NS", "c1_rel",
       "s1_rel")


def _red_ew(d):
    """(tag, species, E, cz in [0, 0.3), az): measure 1, largest cap."""
    return np.stack([np.asarray(d[t + "1"])[-1, :, :, CZ0:CZ0 + 3, :]
                     for t in ("s", "n", "t", "v")])


def _red_az(d):
    """(tag x measure, species, E, cz): azimuth-summed, largest cap."""
    return np.stack([np.asarray(d[t + m])[-1].sum(-1)
                     for t in ("s", "n", "t", "v") for m in ("0", "1")])


def _red_caps(d):
    """(tag, cap, species, E, cz in [0, 0.3)): measure 1, az-summed."""
    return np.stack([np.asarray(d[t + "1"])[:, :, :, CZ0:CZ0 + 3, :].sum(-1)
                     for t in ("s", "t", "v")])


REDUCERS = {"ew": _red_ew, "az": _red_az, "caps": _red_caps,
            "cnt": lambda d: np.asarray(d["m3b_counts"], float),
            "sh": lambda d: np.array([1.0])}


def load(rundir, tag, G=20):
    return A.JackRun(rundir, tag, G=G, reducers=REDUCERS)


def ew_axis():
    import diag_ew_charge_fourier as F
    return F.geomagnetic_ew_axis()


def observables(f, axis_deg):
    """The observables of one 12-bin azimuth pattern (our bin order)."""
    import diag_ew_charge_fourier as F
    o = F.ew_observables(f, AZ_C, axis_deg, bin_width_deg=30.0)
    a0 = max(o["a0"], 1e-300)
    if not (o["f_E"] > 0 and o["f_W"] > 0):
        o["WE"] = np.nan
    if not f.min() > 0:
        o["maxmin"] = np.nan
    fN = 0.5 * (f[0] + f[11]) / a0
    fS = 0.5 * (f[5] + f[6]) / a0
    return np.array([o["a1_rel"], o["dphi"], o["a2_rel"], o["WE"], o["maxmin"],
                     fN, fS, fN / max(fS, 1e-300), o["c1_rel"], o["s1_rel"]])


def band_patterns(ew):
    """(tag, species, E band, cz band, az) band-summed patterns."""
    out = np.zeros(ew.shape[:2] + (len(E_BANDS), 3, 12))
    for ib, (i0, i1) in enumerate(E_BANDS.values()):
        out[:, :, ib] = ew[:, :, i0:i1].sum(2)
    return out


def ew_table(J, axis_deg, tags=("s", "n", "t", "v")):
    """``{tag: (est, err)}`` with arrays (species, E band, cz band, OBS)."""
    pats = band_patterns(J.total("ew"))
    reps = np.array([band_patterns(r) for r in J.replicates("ew")])
    G = reps.shape[0]
    res = {}
    for tg in tags:
        it = TAG_IDX[tg]
        est = np.zeros((4, len(E_BANDS), 3, len(OBS)))
        rep = np.zeros((G,) + est.shape)
        for k in range(4):
            for ie in range(len(E_BANDS)):
                for iz in range(3):
                    est[k, ie, iz] = observables(pats[it, k, ie, iz], axis_deg)
                    for g in range(G):
                        rep[g, k, ie, iz] = observables(reps[g, it, k, ie, iz],
                                                        axis_deg)
        dev = rep - np.nanmean(rep, 0)
        dev[..., 1] = (dev[..., 1] + 180.0) % 360.0 - 180.0     # phase wraps
        err = np.sqrt((G - 1.0) / G * np.nansum(dev ** 2, 0))
        res[tg] = (est, err, rep)
    return res


def honda_patterns(path=None):
    """Honda's band-integrated patterns (species, E band, cz band, az) in our
    compass bin order."""
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "..", "mceq3d", "honda_kam.npz")
    import sky
    h = dict(np.load(path))
    eb = sky.E_BINS
    az_c = (180.0 - (h["azlo"].astype(float) + 15.0)) % 360.0
    order = [int(np.argmin(np.abs(((az_c - c + 180) % 360) - 180))) for c in AZ_C]
    out = np.zeros((4, len(E_BANDS), 3, 12))
    lE = np.log(h["E"])
    for k, sp in enumerate(SP_NAMES):
        tab = h[HONDA_KEY[sp]]
        for ib, (i0, i1) in enumerate(E_BANDS.values()):
            x = np.logspace(np.log10(eb[i0]), np.log10(eb[i1]), 81)
            for iz, (c0, _c1) in enumerate(CZ_BANDS):
                icz = int(np.argmin(np.abs(h["czlo"] - c0)))
                for ja, jh in enumerate(order):
                    y = np.exp(np.interp(np.log(x), lE,
                                         np.log(np.maximum(tab[icz, jh], 1e-300))))
                    out[k, ib, iz, ja] = np.trapezoid(y, x)
    return out


def table_obs(pats, axis_deg):
    out = np.zeros(pats.shape[:3] + (len(OBS),))
    for idx in np.ndindex(pats.shape[:3]):
        out[idx] = observables(pats[idx], axis_deg)
    return out


def engine_patterns(path):
    """The factorised engine's patterns from an ``engine_ew.npz`` written by
    ``engine_ew`` below: ``flux[species]`` (cz band, az, E) on its own E
    grid, band-integrated like Honda's."""
    import sky
    d = dict(np.load(path))
    e = d["e"]
    out = np.zeros((4, len(E_BANDS), 3, 12))
    keymap = {"nue": "total_nue", "antinue": "total_antinue",
              "numu": "total_numu", "antinumu": "total_antinumu"}
    eb = sky.E_BINS
    for k, sp in enumerate(SP_NAMES):
        F = d[keymap[sp]]
        for ib, (i0, i1) in enumerate(E_BANDS.values()):
            x = np.logspace(np.log10(eb[i0]), np.log10(eb[i1]), 81)
            for iz in range(3):
                for ja in range(12):
                    y = np.exp(np.interp(np.log(x), np.log(e),
                                         np.log(np.maximum(F[iz, ja], 1e-300))))
                    out[k, ib, iz, ja] = np.trapezoid(y, x)
    return out


def engine_ew(out_path, cache_dir, cz_sub=3, primary=None):
    """Run the delivered factorised engine (read-only use of tools/mceq3d) on
    Honda's bins and save ``flux[species] (cz band, az, E)``.  ``cache_dir``
    must be outside tools/mceq3d (copy its ``.cache3d`` there for a warm
    start)."""
    import diag_ew_charge_fourier as F
    eng = None
    if primary is not None:
        from mceq3d_flux import MCEq3DFlux
        eng = MCEq3DFlux(base_model="hybrid", primary=primary,
                         daemonflux_location="kamioka")
    e, Fl, az = F.model_flux(muon_bending=True, cz_sub=cz_sub,
                             cache_dir=cache_dir, engine=eng)
    # model_flux orders the cz bands as CZ_BANDS = (0.25, 0.15, 0.05)
    order = [2, 1, 0]
    np.savez(out_path, e=e, az=az,
             **{s: Fl[s][order] for s in Fl})
    return out_path


def fmt_row(lab, v, e=None, fmt="%6.3f"):
    if e is None:
        return lab + " " + " ".join(fmt % x for x in v)
    return lab + " " + " ".join((fmt % x) + "+-" + (fmt % y).strip()
                                for x, y in zip(v, e))


def report(rundir, tag, engine=None, G=20, out_txt=None, out_npz=None):
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    J = load(rundir, tag, G)
    ax = ew_axis()
    tab = ew_table(J, ax)
    hp = honda_patterns()
    ho = table_obs(hp, ax)
    eo = None
    if engine:
        eo = table_obs(engine_patterns(engine), ax)
    cnt = J.total("cnt")
    import sky3d
    out(f"# milestone 3b East-West report: run {tag}, {J.n_showers:.3g} primaries,"
        f" {J.cpu.sum() / 3600:.1f} core-h, geomagnetic East axis {ax:.2f} deg")
    out("# counters: " + ", ".join(f"{n}={v:.4g}" for n, v in
                                   zip(sky3d.COUNT_NAMES, cnt)))
    summ = {"honda_obs": ho, "axis": ax, "obs_names": np.array(OBS),
            "e_bands": np.array(list(E_BANDS)), "cz_bands": np.array(CZ_BANDS)}
    if eo is not None:
        summ["engine_obs"] = eo
    for tg, lab in (("s", "3D (cutoff at injection)"),
                    ("v", "1D with the site cutoff"),
                    ("n", "3D no cutoff (copy 0)"),
                    ("t", "1D no cutoff")):
        est, err, _ = tab[tg]
        summ[f"{tg}_obs"] = est
        summ[f"{tg}_err"] = err
        out(f"\n## {lab} [{tg}]: a1/a0, dphi [deg], W/E, max/min, f_N/a0, "
            "f_S/a0 (MC +- jackknife | Honda | engine)")
        for ie, E in enumerate(E_BANDS):
            for iz, (c0, c1) in enumerate(CZ_BANDS):
                out(f"E={E} GeV cosZ {c0:.1f}-{c1:.1f}")
                for k, sp in enumerate(SP_NAMES):
                    m, s = est[k, ie, iz], err[k, ie, iz]
                    h = ho[k, ie, iz]
                    row = (f"  {sp:9s} a1 {m[0]:.3f}+-{s[0]:.3f} (H {h[0]:.3f}"
                           + (f" E {eo[k, ie, iz, 0]:.3f}" if eo is not None else "")
                           + f") dphi {m[1]:+6.1f}+-{s[1]:.1f} (H {h[1]:+6.1f}"
                           + (f" E {eo[k, ie, iz, 1]:+6.1f}" if eo is not None else "")
                           + f") W/E {m[3]:.2f}+-{s[3]:.2f} (H {h[3]:.2f}"
                           + (f" E {eo[k, ie, iz, 3]:.2f}" if eo is not None else "")
                           + f") max/min {m[4]:.2f}+-{s[4]:.2f} (H {h[4]:.2f}"
                           + (f" E {eo[k, ie, iz, 4]:.2f}" if eo is not None else "")
                           + f") N {m[5]:.3f}+-{s[5]:.3f} (H {h[5]:.3f}) S "
                           f"{m[6]:.3f}+-{s[6]:.3f} (H {h[6]:.3f})")
                    out(row)
    # species ordering of the 3D amplitude vs Honda
    out("\n## species fan-out, 3D [s]: a1/a0 per species and ratios to numu")
    est, err, rep = tab["s"]
    for ie, E in enumerate(E_BANDS):
        for iz, (c0, c1) in enumerate(CZ_BANDS):
            a = est[:, ie, iz, 0]
            ra = rep[:, :, ie, iz, 0]
            r = a / a[2]
            rr = ra / ra[:, 2:3]
            G_ = rr.shape[0]
            re_ = np.sqrt((G_ - 1.0) / G_ * ((rr - rr.mean(0)) ** 2).sum(0))
            h = ho[:, ie, iz, 0]
            out(f"E={E} cz {c0:.1f}-{c1:.1f}: MC a1/a1(numu) nue {r[0]:.2f}+-"
                f"{re_[0]:.2f} antinue {r[1]:.2f}+-{re_[1]:.2f} antinumu "
                f"{r[3]:.2f}+-{re_[3]:.2f} | Honda nue {h[0] / h[2]:.2f} "
                f"antinue {h[1] / h[2]:.2f} antinumu {h[3] / h[2]:.2f}"
                + (" | engine nue %.2f antinue %.2f antinumu %.2f" % tuple(
                    eo[[0, 1, 3], ie, iz, 0] / eo[2, ie, iz, 0])
                   if eo is not None else ""))
    # 3D/1D ratios against both references (az-averaged)
    out("\n## 3D/1D, azimuth-averaged specific intensity: s/t (vs no-cutoff 1D) "
        "and s/v (vs site-cutoff 1D)")
    az = J.total("az")
    reps = J.replicates("az")
    names = [t + m for t in ("s", "n", "t", "v") for m in ("0", "1")]
    i_s1, i_t1, i_v1, i_n1 = (names.index(x) for x in ("s1", "t1", "v1", "n1"))
    for ie, E in enumerate(E_BANDS):
        i0, i1 = E_BANDS[E]
        for nm, ia, ib in (("s/t", i_s1, i_t1), ("s/v", i_s1, i_v1),
                           ("n/t", i_n1, i_t1), ("v/t", i_v1, i_t1)):
            for k, sp in enumerate(SP_NAMES):
                num = az[ia, k, i0:i1].sum(0)
                den = az[ib, k, i0:i1].sum(0)
                rr = np.array([r_[ia, k, i0:i1].sum(0) / np.where(
                    r_[ib, k, i0:i1].sum(0) > 0, r_[ib, k, i0:i1].sum(0), np.nan)
                    for r_ in reps])
                with np.errstate(invalid="ignore", divide="ignore"):
                    est_ = num / np.where(den > 0, den, np.nan)
                G_ = rr.shape[0]
                e_ = np.sqrt((G_ - 1.0) / G_ * np.nansum(
                    (rr - np.nanmean(rr, 0)) ** 2, 0))
                cz = slice(10, 20)
                out(f"E={E} {nm} {sp:9s} cz 0.0..1.0: " + " ".join(
                    f"{x:.3f}+-{y:.3f}" for x, y in zip(est_[cz], e_[cz])))
                summ[f"ratio_{nm.replace('/', '')}_{sp}_{E}"] = np.stack(
                    [est_, e_])
    if out_txt:
        with open(out_txt, "w") as fh:
            fh.write("\n".join(lines) + "\n")
    if out_npz:
        np.savez(out_npz, **summ)
    return tab, ho, eo


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir")
    ap.add_argument("--tag", default="m3b")
    ap.add_argument("--engine", default=None)
    ap.add_argument("--G", type=int, default=20)
    ap.add_argument("--out-txt", default=None)
    ap.add_argument("--out-npz", default=None)
    a = ap.parse_args(argv)
    report(a.rundir, a.tag, a.engine, a.G, a.out_txt, a.out_npz)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# task B: allocation;  gate C5: the bias of the rotation reuse
# ---------------------------------------------------------------------------
class _Mix:
    """A JackRun-like view of per-(group, stratum) sums built by hand."""

    def __init__(self, base, S):
        self.G, self.K, self.M = base.G, base.K, base.M
        self.S = S

    total = A.JackRun.total
    replicates = A.JackRun.replicates
    _replicates = A.JackRun._replicates


def a1_obs(J, axis_deg, tag="s", stratum=None):
    """(est, replicates) of a1/a0 and dphi, (species, E band, cz band, 2)."""
    it = TAG_IDX[tag]
    pats = band_patterns(J.total("ew"))[it]
    reps = np.array([band_patterns(r)[it]
                     for r in J.replicates("ew", stratum)])

    def f(p):
        o = np.zeros(p.shape[:3] + (2,))
        for idx in np.ndindex(p.shape[:3]):
            v = observables(p[idx], axis_deg)
            o[idx] = v[[0, 1]]
        return o
    return f(pats), np.array([f(r) for r in reps])


def jvar(rep, phase_axis=None):
    G = rep.shape[0]
    dev = rep - np.nanmean(rep, 0)
    if phase_axis is not None:
        dev[..., phase_axis] = (dev[..., phase_axis] + 180.0) % 360.0 - 180.0
    return (G - 1.0) / G * np.nansum(dev ** 2, 0)


def allocation(J, axis_deg, cost_per_primary, floor_frac=0.002):
    """Neyman allocation over the (group, stratum) cells that minimises the
    summed *relative* variance of the 3D East-West amplitudes ``a1/a0``
    (4 species x 3 energies x 3 zenith bands): ``n_h ~ sqrt(a_h / c_h)``,
    ``a_h = n_h sum_q V_hq / V_q`` with ``V_hq`` stratum ``h``'s jackknife
    variance contribution to observable ``q`` at the current ``n_h`` and
    ``V_q`` its total.  ``cost_per_primary[h]`` in CPU s.  Returns
    ``(frac, a_h, V_hq)``."""
    est, rep = a1_obs(J, axis_deg)
    Vq = jvar(rep[..., 0])
    a = np.zeros(J.K)
    Vh = []
    for h in range(J.K):
        _e, r_h = a1_obs(J, axis_deg, stratum=h)
        v = jvar(r_h[..., 0])
        Vh.append(v)
        with np.errstate(invalid="ignore", divide="ignore"):
            a[h] = J.nsh[h] * np.nansum(np.where(Vq > 0, v / Vq, 0.0))
    w = np.sqrt(np.maximum(a, 0.0) / np.asarray(cost_per_primary, float))
    frac = w / w.sum()
    frac = np.maximum(frac, floor_frac)
    return frac / frac.sum(), a, np.array(Vh)


def rotation_bias(Jrot, Janc, axis_deg, above):
    """Gate C5: the change of the 3D observables when the cascades of the
    strata flagged ``above`` (reused by rotation) are re-anchored on each
    copy's own exact bent primary -- the bias of the rotation reuse there.
    Paired runs (same seeds), so the jackknife of the difference is honest.
    Returns ``(d_est, d_err)`` with arrays (species, E band, cz band, 2)
    for ``a1/a0`` and ``dphi``."""
    mask = np.zeros(Jrot.K, bool)
    mask[list(above)] = True
    S = Jrot.S["ew"] + mask[None, :, None, None, None, None, None] * (
        Janc.S["ew"] - Jrot.S["ew"])
    mix = _Mix(Jrot, {"ew": S})
    e0, r0 = a1_obs(Jrot, axis_deg)
    e1, r1 = a1_obs(mix, axis_deg)
    d = e1 - e0
    dr = r1 - r0
    G = dr.shape[0]
    dev = dr - dr.mean(0)
    dev[..., 1] = (dev[..., 1] + 180.0) % 360.0 - 180.0
    d[..., 1] = (d[..., 1] + 180.0) % 360.0 - 180.0
    return d, np.sqrt((G - 1.0) / G * (dev ** 2).sum(0))


def realisations_per_primary(groups, nstrata, thr, n=4000, naz=12, seed=3,
                             theta_inj=40.0, model="H3A"):
    """Expected hadronic cascades per primary for each (group, stratum) when
    copies below ``thr`` GV are re-simulated (one straight + copy 0 + every
    copy not gated forbidden), from a sample of the actual primaries (no
    cascade run).  Also returns the fraction of primaries below ``thr``."""
    import cutoff
    import primaries as prim
    import sky
    mdl = prim.primary_model(model)
    axis = cutoff.site_axis(*sky.SITES["kamioka"])
    out = {}
    rng = np.random.default_rng(seed)
    for g in groups:
        _n, a_mass, z, _gid = sky.GROUPS[g]
        for k in range(nstrata):
            e, _isp, _w, r0, u0 = sky.primary_sample(rng, mdl, g, k, n, n,
                                                     theta_inj, axis)
            R = cutoff.rigidity_gv(e, a_mass, z)
            rr, uu = cutoff.copy_states(r0, u0, axis, naz, 0.0)
            gate = cutoff.gate(rr.reshape(-1, 3), uu.reshape(-1, 3),
                               np.repeat(R, naz)).reshape(n, naz)
            kept = 1 + (gate[:, 1:] != -1).sum(1)
            nreal = np.where(R < thr, 1 + kept, 1)
            out[(g, k)] = (float(nreal.mean()), float((R < thr).mean()),
                           float(np.median(R)))
    return out


# ---------------------------------------------------------------------------
# gate C3: the 1D site-cutoff reference against the engine's G_s
# ---------------------------------------------------------------------------
C3_BINS = (("vertical", 19, None), ("87E 60-90", 10, 2), ("87E 90-120", 10, 3),
           ("87W 240-270", 10, 8), ("87W 270-300", 10, 9))


def gate_c3(J, gs_path, sitemap, n_cz=5, n_az=6, out=print):
    """MC 1D transmission ``v/t`` (site map, same primaries) against the
    factorised engine's ``G_s(R_c, E)`` (H3a, SIBYLL-2.3d, sharp cutoff,
    ``geomag_response``) averaged over the bin's directions with the site
    map's ``R_c`` and weighted by the engine's own no-cutoff 1D flux at each
    direction, band-integrated in energy like the MC."""
    import sky
    from mceq3d_flux import MCEq3DFlux
    d = dict(np.load(gs_path))
    e = d["e"]
    eng = MCEq3DFlux(interaction_model="SIBYLL23D",
                     primary=("HillasGaisser2012", "H3a"), base_model="mceq")
    az = J.total("az")
    reps_az = J.replicates("az")
    names = [t + m for t in ("s", "n", "t", "v") for m in ("0", "1")]
    it1, iv1 = names.index("t1"), names.index("v1")
    ew = J.total("ew")
    reps_ew = J.replicates("ew")
    rows = []
    sp_mceq = {"nue": "total_nue", "antinue": "total_antinue",
               "numu": "total_numu", "antinumu": "total_antinumu"}
    eb = sky.E_BINS
    for lab, iz, ia in C3_BINS:
        cz_lo = -1.0 + 0.1 * iz
        czs = cz_lo + (np.arange(n_cz) + 0.5) / n_cz * 0.1
        if ia is None:
            azs = (np.arange(4 * n_az) + 0.5) * 360.0 / (4 * n_az)
            czr = 0.95
        else:
            azs = 30.0 * ia + (np.arange(n_az) + 0.5) * 30.0 / n_az
            czr = 0.05
        flux = {}
        for c in czs:
            eng.mceq.set_theta_deg(float(np.degrees(np.arccos(c))))
            eng.mceq._phi0[:] = eng._phi0_std
            eng.mceq.solve()
            flux[c] = {sp: eng.mceq.get_solution(sp_mceq[sp], 0).copy()
                       for sp in SP_NAMES}
        for k, sp in enumerate(SP_NAMES):
            G = d[f"{sp_mceq[sp]}|{czr}"]                  # (n_rc, n_e)
            rc = d.get(f"rc|{czr}", d.get("rc"))
            for ib, (E, (i0, i1)) in enumerate(E_BANDS.items()):
                sel = (e >= eb[i0]) & (e < eb[i1])
                num = den = 0.0
                for c in czs:
                    zen = np.degrees(np.arccos(c))
                    for a_ in azs:
                        rcv = float(sitemap(zen, a_))
                        g = np.array([np.interp(rcv, rc, G[:, j])
                                      for j in np.where(sel)[0]])
                        f = flux[c][sp][sel] * np.gradient(e)[sel]
                        num += (f * g).sum()
                        den += f.sum()
                g_eng = num / den
                if ia is None:
                    t = az[it1, k, i0:i1, iz].sum()
                    v = az[iv1, k, i0:i1, iz].sum()
                    rr = np.array([r[iv1, k, i0:i1, iz].sum()
                                   / r[it1, k, i0:i1, iz].sum() for r in reps_az])
                else:
                    t = ew[2, k, i0:i1, 0, ia].sum()
                    v = ew[3, k, i0:i1, 0, ia].sum()
                    rr = np.array([r[3, k, i0:i1, 0, ia].sum()
                                   / r[2, k, i0:i1, 0, ia].sum() for r in reps_ew])
                G_ = len(rr)
                err = np.sqrt((G_ - 1.0) / G_ * np.nansum((rr - np.nanmean(rr)) ** 2))
                g_mc = v / t if t > 0 else np.nan
                rows.append((lab, sp, E, g_mc, err, g_eng))
                out(f"C3 {lab:12s} {sp:9s} E={E}: MC v/t = {g_mc:.4f} +- {err:.4f}"
                    f"   engine G_s = {g_eng:.4f}   MC/engine = "
                    f"{g_mc / g_eng:.3f} +- {err / g_eng:.3f}")
    return rows
