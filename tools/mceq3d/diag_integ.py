"""Engine side of the daemonflux 3D integration (``INTEGRATION.md``).

    maps-dn      build the down-going cutoff maps of the validation grid (cache)
    engine       F_s = engine 3D / engine 1D no-cutoff (same definition as the MC's s/t),
                 up-going (10 cos Z x 12 az) and down-going (3 horizon bands x 12 az,
                 7 cos Z x 6 az), GSF primary, muon_bending off -> out/engine_F_fine.npz
    table        provisional Kamioka table (PROVISIONAL): pilot-2 down-going s/t +
                 engine up-going F on the coarse grid -> $W/runs/integ/kamioka_3D_provisional.npz
    validate     engine vs Honda (solmin, solmax) for up-going: up/down, azimuth observables

``F`` is the ratio of ``MCEq3DFlux.solve`` (cone-averaged cutoff, production-point
anchor, far-side cutoffs for cos Z < 0, 3D production factor) to
``MCEq3DFlux.base(|cos Z|)``, the curved-atmosphere 1D flux without cutoff: the engine
analogue of the MC's s/t (3D flux over the same showers' 1D no-cutoff flux).
Outputs under ``$W/runs/integ``; caches via ``$MCEQ3D_CACHE``.
"""
import importlib.util  # noqa: F401
import os
import sys
import time
import warnings
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
W = os.environ.get("MC3D_WORK", "/home/pigrange/mc3d_work")
RUN = os.path.join(W, "runs", "integ")
CACHE_REQ = os.path.join(W, "runs", "dbv142", "cache_old")  # database-independent maps
LAT, LON = 36.43, 137.31
DATE = datetime(2020, 1, 1)
SPECIES = ("total_nue", "total_antinue", "total_numu", "total_antinumu")
CZ_UP = -np.array([0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05])
CZ_UP = np.sort(CZ_UP)  # ascending: -0.95 ... -0.05
CZ_HOR = np.array([0.05, 0.15, 0.25])  # cached 12-azimuth maps
CZ_DN6 = np.array([0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
AZ12 = 15.0 + 30.0 * np.arange(12)
AZ6 = AZ12[::2]


def _bare():
    import mceq3d_flux as mf

    class Bare(mf.MCEq3DFlux):
        def __init__(self):
            pass

    return Bare()


def cmd_maps_dn(n_jobs=4):
    e = _bare()
    t = time.time()
    e.cutoff_grid(LAT, LON, CZ_DN6, AZ6, DATE, None, cache_dir=CACHE_REQ, n_jobs=n_jobs)
    print("down maps done", time.time() - t, flush=True)


def cmd_engine(n_jobs=4):
    from mceq3d_flux import MCEq3DFlux

    warnings.simplefilter("ignore")
    eng = MCEq3DFlux(base_model="mceq", primary=("GlobalSplineFitBeta", None))
    out = {"e": np.array(eng.e), "e_widths": np.array(eng.mceq.e_widths)}
    for name, cz, az in (("up", CZ_UP, AZ12), ("hor", CZ_HOR, AZ12), ("dn6", CZ_DN6, AZ6)):
        t = time.time()
        r = eng.solve(LAT, LON, cz, az, date=DATE, use_cache=True, cache_dir=CACHE_REQ,
                      muon_bending=False, n_jobs=n_jobs)
        base = eng.base(np.abs(cz))
        out[f"cz_{name}"], out[f"az_{name}"] = cz, az
        for sp in SPECIES:
            out[f"base_{name}_{sp}"] = base[sp]  # (ncz, nE) 1D no-cutoff
            out[f"flux_{name}_{sp}"] = r["flux"][sp]  # (ncz, naz, nE) 3D
        print(name, "done", f"{time.time() - t:.0f}s", flush=True)
    np.savez(os.path.join(RUN, "out", "engine_F_fine.npz"), **out)


# ---------------------------------------------------------------------------
# coarse grid of the MC table: E 5/dec 0.1-10 GeV, cos Z 0.2 bins over (-1, 1), az 60 deg
# ---------------------------------------------------------------------------
E_EDGES = 10.0 ** (-1.0 + 0.2 * np.arange(11))
CZ_EDGES = np.linspace(-1.0, 1.0, 11)
AZ_EDGES = np.linspace(0.0, 360.0, 7)
MC_SPECIES = ("nue", "antinue", "numu", "antinumu")  # order of the MC arrays


MIRROR_CZMAX = 0.35  # rows beyond are left as the single far-side cutoff (U/D validated to here)
MIRROR = False  # apply the up-going cone correction (INTEGRATION.md 11); set by `table`/`validate mirror`


def mirror_ratio(sp):
    """Cone correction F_cone/F_single of the mirrored down-going directions for the 10
    up-going rows (cos Z -0.95 ... -0.05) x 12 azimuths x nE (``diag_upgoing_fix.py modes``).
    Rows 0.05-0.25: computed at 12 azimuths; rows 0.35-0.95: at 6 azimuths (cached maps),
    interpolated periodically in azimuth."""
    m = np.load(os.path.join(RUN, "out", "updiag_modes.npz"))
    m6 = np.load(os.path.join(RUN, "out", "updiag_modes_dn6.npz"))
    with np.errstate(invalid="ignore", divide="ignore"):
        r_h = np.where(m[f"single_{sp}"] > 0, m[f"cone_{sp}"] / m[f"single_{sp}"], 1.0)  # (3, 12, nE)
        r_6 = np.where(m6[f"dn6_single_{sp}"] > 0, m6[f"dn6_cone_{sp}"] / m6[f"dn6_single_{sp}"], 1.0)
    az6 = m6["dn6_az"]
    r_6_12 = np.zeros((r_6.shape[0], 12, r_6.shape[2]))
    for i in range(r_6.shape[0]):
        for ie in range(r_6.shape[2]):
            ext_a = np.concatenate([[az6[-1] - 360.0], az6, [az6[0] + 360.0]])
            ext_v = np.concatenate([[r_6[i, -1, ie]], r_6[i, :, ie], [r_6[i, 0, ie]]])
            r_6_12[i, :, ie] = np.interp(AZ12, ext_a, ext_v)
    down = np.concatenate([r_h, r_6_12])  # |cos Z| = 0.05 ... 0.95
    down[np.arange(10) * 0.1 + 0.05 > MIRROR_CZMAX + 1e-9] = 1.0  # validated range only
    return down[::-1]  # up-going rows -0.95 ... -0.05


def _fz(fine, name, sp):
    """(F[ncz, 6 az, nE], base[ncz, nE]) of one entry; 12-azimuth entries are
    pair-averaged to the 60-degree bins, 6-azimuth ones are used as they are."""
    base = fine[f"base_{name}_{sp}"]
    fl = fine[f"flux_{name}_{sp}"] / base[:, None, :]
    if name == "up" and MIRROR:
        fl = fl * mirror_ratio(sp)
    if fl.shape[1] == 12:
        fl = 0.5 * (fl[:, 0::2] + fl[:, 1::2])
    return fl, base


def kappa_and_B():
    """kappa_s(E) = MC/MCEq yield ratio from t/B over cos Z 0.35-1 (t is reliable there), per
    species and coarse E bin, with errors and a flatness check; and the coarse B cells."""
    b = np.load(os.path.join(RUN, "out", "base1d_kamioka.npz"))
    t = np.load(os.path.join(W, "runs", "m3e", "sky", "pilot2_total.npz"))
    B = b["B_t1"]  # (4, 60, 20), az-summed t1 scale
    T = t["t1"][-1].sum(-1)  # (4, 60, 20) largest cap, measure 1, az-summed
    T2 = t["t1sq"][-1].sum(-1)
    cz = slice(13, 20)  # cos Z bins 0.3-1.0 ... (bin 13 = [0.3, 0.4]); use 0.35-1 centres
    cz = slice(13, 20)
    kap = np.zeros((4, 10))
    dk = np.zeros((4, 10))
    chi2 = np.zeros((4, 10))
    slope = np.zeros((4, 10))
    for k in range(4):
        for ie in range(10):
            es = slice(4 * ie, 4 * ie + 4)
            tt, bb, t2 = T[k, es, cz].sum(0), B[k, es, cz].sum(0), T2[k, es, cz].sum(0)
            r = tt / bb  # per cos Z bin
            er = np.sqrt(t2) / bb
            kap[k, ie] = tt.sum() / bb.sum()
            dk[k, ie] = np.sqrt(t2.sum()) / bb.sum()
            chi2[k, ie] = np.sum(((r - kap[k, ie]) / er) ** 2) / (len(r) - 1)
            x = 0.35 + 0.1 * np.arange(len(r))
            slope[k, ie] = np.polyfit(x, r / kap[k, ie], 1, w=1.0 / (er / kap[k, ie]))[0]
    # the per-bin scatter of t/B is far larger than the sum-of-squares error (chi2/dof 6-66: the
    # tallies are correlated and heavy tailed): scale the kappa error by sqrt(chi2/dof)
    dk_stat = dk.copy()
    dk = dk * np.sqrt(np.maximum(chi2, 1.0))
    # coarse B cells (E 5/dec: 4 of the 60 bins; cos Z pairs; az pairs = 2/12 of the azimuth)
    Bc = B[:, :40].reshape(4, 10, 4, 10, 2).sum((2, 4)) * (2.0 / 12.0)  # (4, 10 E, 10 cz)
    return kap, dk, chi2, slope, Bc, b


def pilot_common():
    """Down-going G = s / (kappa B) on the coarse grid, jackknife errors (s) + kappa error."""
    sys.path.insert(0, os.path.join(os.environ["R"], "tools", "mc3d", "results", "m3c", "scripts"))
    import m3c_lib as L

    kap, dk, chi2, slope, Bc, _ = kappa_and_B()
    d = np.load(os.path.join(W, "runs", "m3f", "work", "design_pilot2.npz"))
    S, Mg = d["x"], d["M"]
    s = S.shape
    x = S.reshape(s[:-3] + (s[-3] // 2, 2, s[-2] // 2, 2, s[-1] // 2, 2)).sum((-1, -3, -5))
    tot = x.sum((0, 1))
    rep = L.replicates(x, Mg)  # (G, 3, sp, E, cz, az)
    ref = (kap[:, :, None, None] * Bc[:, :, :, None])  # (4, 10, 10, 1)
    g0 = tot[0] / ref
    gr = rep[:, 0] / ref[None]
    n = len(gr)
    err_s = np.sqrt((n - 1.0) / n * np.nansum((gr - np.nanmean(gr, 0)) ** 2, 0))
    err = np.sqrt(err_s**2 + (g0 * dk[:, :, None, None] / kap[:, :, None, None]) ** 2)
    return g0, err, kap, dk, chi2, slope, float(Mg.sum())


def engine_coarse(fine, hemisphere):
    """Engine F on the MC coarse grid (E 5/dec x 5 cos Z bins x 6 az per hemisphere).

    The weights are base(|cos Z|, E) * dE (the engine analogue of the ratio of summed
    tallies s/t); cos Z bins of 0.2 are the pair-average of two 0.1 rows."""
    e, de = fine["e"], fine["e_widths"]
    out = np.full((4, 10, 5, 6), np.nan)
    for k, sp in enumerate(SPECIES):
        if hemisphere == "up":
            fl, base = _fz(fine, "up", sp)  # rows -0.95 ... -0.05
        else:
            f1, b1 = _fz(fine, "hor", sp)
            f2, b2 = _fz(fine, "dn6", sp)
            fl, base = np.concatenate([f1, f2]), np.concatenate([b1, b2])  # 0.05 ... 0.95
        for ie in range(10):
            m = (e >= E_EDGES[ie]) & (e < E_EDGES[ie + 1])
            for ic in range(5):
                sel = slice(2 * ic, 2 * ic + 2)
                w = base[sel][:, m] * de[m]  # (2, nEin)
                out[k, ie, ic, :] = (fl[sel][:, :, m] * w[:, None, :]).sum((0, 2)) / w.sum()
    return out


def pilot_coarse():
    """Down-going pilot-2 s/t, coarse (E 5/dec, cz 0.2, az 60): G, jackknife error, t."""
    sys.path.insert(0, os.path.join(os.environ["R"], "tools", "mc3d", "results", "m3c", "scripts"))
    import m3c_lib as L

    d = np.load(os.path.join(W, "runs", "m3f", "work", "design_pilot2.npz"))
    S, Mg = d["x"], d["M"]  # (G, K, 3 [s, t, v], sp, E20, cz20, az12, )
    s = S.shape
    x = S.reshape(s[:-3] + (s[-3] // 2, 2, s[-2] // 2, 2, s[-1] // 2, 2)).sum((-1, -3, -5))
    tot = x.sum((0, 1))
    rep = L.replicates(x, Mg)  # (G, 3, sp, E, cz, az)
    with np.errstate(divide="ignore", invalid="ignore"):
        g0 = tot[0] / tot[1]
        gr = rep[:, 0] / rep[:, 1]
        n = len(gr)
        err = np.sqrt((n - 1.0) / n * np.nansum((gr - np.nanmean(gr, 0)) ** 2, 0))
    return g0, err, tot[1], float(Mg.sum())


HONDA = {
    "solmin": os.path.join(HERE, "honda_kam.npz"),
    "solmax": os.path.join(W, "runs", "m3f", "honda", "honda_kam_solmax.npz"),
}
HKEY = {"nue": "nue", "antinue": "nuebar", "numu": "numu", "antinumu": "numubar"}
E_PTS = (0.3, 0.5, 1.0, 2.0)


def _lat(y, x, X):
    return float(np.exp(np.interp(np.log(X), np.log(x), np.log(np.maximum(y, 1e-300)))))


def _honda_cell(h, key, cz_c, az_idx=None):
    """Honda flux rows (naz, nE) of the cos Z bin centred on ``cz_c`` (up or down)."""
    i = int(np.argmin(np.abs(h["czlo"] - round(cz_c - 0.05, 2))))
    rows = h[key][i]
    return rows if az_idx is None else rows[az_idx]


def cmd_validate(flag=""):
    """Engine up-going vs Honda solmin/solmax -> out/validate_up.md (``mirror``: with the
    up-going cone correction -> out/validate_up_mirror.md)."""
    global MIRROR
    MIRROR = flag == "mirror"
    import analyse_m3b as M
    from diag_rigidity_scan import _honda_pattern
    import diag_ew_charge_fourier as fz

    fine = np.load(os.path.join(RUN, "out", "engine_F_fine.npz"))
    e = fine["e"]
    H = {k: dict(np.load(v)) for k, v in HONDA.items()}
    ax = fz.geomagnetic_ew_axis()
    L = []

    def F(name, sp, icz):  # F pattern (naz, nE)
        out = fine[f"flux_{name}_total_{sp}"][icz] / fine[f"base_{name}_total_{sp}"][icz][None, :]
        if name == "up" and MIRROR:
            out = out * mirror_ratio(f"total_{sp}")[icz]
        return out

    # A. up/down ratio, azimuth averaged
    L += ["**A. up/down ratio at equal |cos Z|, azimuth-averaged; engine F_up/F_dn (muon_bending off) "
          "vs Honda solmin / solmax**", "",
          "| species | |cos Z| | E [GeV] | engine | Honda solmin | Honda solmax | engine/solmin - 1 | engine/solmax - 1 |",
          "|---|---|---|---|---|---|---|---|"]
    res = {}
    bands = [("hor", i, c, None) for i, c in enumerate(CZ_HOR)]
    bands += [("dn6", i, c, [0, 2, 4, 6, 8, 10]) for i, c in enumerate(CZ_DN6)]
    for sp in MC_SPECIES:
        for name, i, c, azi in bands:
            iu = int(np.argmin(np.abs(fine["cz_up"] + c)))
            fu, fd = F("up", sp, iu), F(name, sp, i)
            if azi is not None:
                fu = fu[azi]
            for E in E_PTS:
                eng = _lat(fu.mean(0), e, E) / _lat(fd.mean(0), e, E)
                row = [eng]
                for k in ("solmin", "solmax"):
                    h = H[k]
                    up = _honda_cell(h, HKEY[sp], -c, azi)
                    dn = _honda_cell(h, HKEY[sp], c, azi)
                    row.append(_lat(up.mean(0), h["E"], E) / _lat(dn.mean(0), h["E"], E))
                res[f"{sp}|{c}|{E}"] = row
                L.append(f"| {sp} | {c} | {E} | {row[0]:.3f} | {row[1]:.3f} | {row[2]:.3f} | "
                         f"{100 * (row[0] / row[1] - 1):+.1f}% | {100 * (row[0] / row[2] - 1):+.1f}% |")
    # B. azimuth observables, up-going horizon bands
    L += ["", "**B. up-going azimuth observables (12 bins, analyse_m3b definitions), engine | Honda solmin | "
          "Honda solmax**", "",
          "| species | cos Z | E | a1/a0 | dphi [deg] | fN | fS | N/S | W/E |", "|---|---|---|---|---|---|---|---|---|"]
    names = list(M.OBS)
    sel = ["a1_rel", "dphi", "fN", "fS", "NS", "WE"]
    for sp in MC_SPECIES:
        for iu, c in ((9, -0.05), (8, -0.15), (7, -0.25)):
            for E in E_PTS[:3]:
                eng = np.array([_lat(F("up", sp, iu)[j], e, E) for j in range(12)])
                oe = M.observables(eng, ax)
                cells = []
                for o in sel:
                    vals = [oe[names.index(o)]]
                    for k in ("solmin", "solmax"):
                        fh = _honda_pattern(H[k], HKEY[sp], c, E, fz)
                        vals.append(M.observables(fh, ax)[names.index(o)])
                    cells.append(" / ".join(f"{v:.3f}" for v in vals))
                L.append(f"| {sp} | {c} | {E} | " + " | ".join(cells) + " |")
    open(os.path.join(RUN, "out", "validate_up_mirror.md" if MIRROR else "validate_up.md"), "w").write(
        "\n".join(L)
    )
    print("\n".join(L))

ENGINE_REL_ERR = 0.05  # floor of the engine relative error (see INTEGRATION.md 11.5)
E_BANDS = ((0.0, 0.3), (0.3, 1.0), (1.0, 3.0), (3.0, 10.0))  # GeV, by coarse-bin centre


def engine_rel_err():
    """Relative error array (4 species, 10 E, 10 cos Z) of the engine-filled bins:
    max(5%, |median shape residual vs Honda solmax|) per species x E band x coarse
    cos Z bin (``cmd_residual``); plain 5% while ``out/engine_sigma.npz`` does not exist."""
    p = os.path.join(RUN, "out", "engine_sigma.npz")
    out = np.full((4, 10, 10), ENGINE_REL_ERR)
    if os.path.exists(p):
        res = np.load(p)["sigma"]  # (4, 4 bands, 10 cz)
        ec = np.sqrt(E_EDGES[:-1] * E_EDGES[1:])
        for ie, e in enumerate(ec):
            ib = [i for i, (lo, hi) in enumerate(E_BANDS) if lo <= e < hi][0]
            out[:, ie, :] = np.maximum(ENGINE_REL_ERR, res[:, ib, :])
    return out
E_MC_MAX = 7  # MC (pilot-2) down-going bins 0..6 (< 2.5 GeV); engine above


def cmd_table():
    """PROVISIONAL Kamioka table in the format of ``Tabulated3DModel``."""
    global MIRROR
    MIRROR = True
    fine = np.load(os.path.join(RUN, "out", "engine_F_fine.npz"))
    g0, err, kap, dk, chi2, slope, nprim = pilot_common()
    eu, ed = engine_coarse(fine, "up"), engine_coarse(fine, "down")
    fac = np.zeros((4, 10, 10, 6))
    dfac = np.zeros_like(fac)
    source = np.zeros((10, 10), dtype=int)  # 0 MC down-going, 1 engine
    fac[:, :, :5] = eu
    rel = engine_rel_err()
    dfac[:, :, :5] = rel[:, :, :5, None] * np.abs(eu)
    source[:, :5] = 1
    fac[:, :, 5:] = g0[:, :, 5:]
    dfac[:, :, 5:] = err[:, :, 5:]
    fac[:, E_MC_MAX:, 5:] = ed[:, E_MC_MAX:]
    dfac[:, E_MC_MAX:, 5:] = rel[:, E_MC_MAX:, 5:, None] * np.abs(ed[:, E_MC_MAX:])
    source[E_MC_MAX:, 5:] = 1
    assert np.all(np.isfinite(fac)) and np.all(np.isfinite(dfac))
    prov = (
        "PROVISIONAL. Down-going (cos Z > 0, E < 2.5 GeV): pilot-2 (%.2g primaries, 34.2 core-h, "
        "MC milestones 3e/3f; its per-species s/t contains the muon-bend fan-out) s/t = 3D flux with cutoff and bending over the same showers' "
        "1D no-cutoff flux, jackknife G = 20 errors, coarse binning (E 5/dec, cos Z 0.2, az 60 deg). "
        "Common 1D reference for both hemispheres: G = s / (kappa B), B = mceq3d engine base(|cos Z|) "
        "(base1d_kamioka.npz), kappa_s(E) = MC/MCEq yield ratio from t/B over cos Z 0.35-1. "
        "Down-going E >= 2.5 GeV and all up-going bins: mceq3d engine (GSF, v142p, muon_bending "
        "off, rigidity fixed, up-going rows with the cone correction of the mirrored down-going "
        "direction) F = 3D/1D no-cutoff, factor_error = max(5%%, |shape residual vs Honda solmax|) of F per species/E band/cos Z bin "
        "(a cross-model systematic, Honda is a comparison model). "
        "Honda/solmax epoch matches GSF. Replace by the production table." % nprim
    )
    np.savez(
        os.path.join(RUN, "kamioka_3D_provisional.npz"),
        species=np.array(MC_SPECIES),
        e_edges=E_EDGES,
        cosz_edges=CZ_EDGES,
        az_edges=AZ_EDGES,
        factor=fac,
        factor_error=dfac,
        source=source,
        label="kamioka_3D_provisional",
        status="PROVISIONAL",
        site="kamioka",
        provenance=prov,
        engine_down_coarse=ed,
        engine_up_coarse=eu,
        mc_down_coarse=g0[:, :, 5:],
        mc_down_error=err[:, :, 5:],
        kappa=kap,
        kappa_error=dk,
    )
    print("wrote", os.path.join(RUN, "kamioka_3D_provisional.npz"))
    # variant: the engine on both hemispheres (one definition of the 1D reference throughout)
    fac3, dfac3 = fac.copy(), dfac.copy()
    fac3[:, :, 5:], dfac3[:, :, 5:] = ed, rel[:, :, 5:, None] * np.abs(ed)
    dfac3[:, :, :5] = rel[:, :, :5, None] * np.abs(eu)
    np.savez(
        os.path.join(RUN, "kamioka_3D_provisional_engine.npz"),
        species=np.array(MC_SPECIES), e_edges=E_EDGES, cosz_edges=CZ_EDGES, az_edges=AZ_EDGES,
        factor=fac3, factor_error=dfac3, source=np.ones((10, 10), dtype=int),
        label="kamioka_3D_provisional_engine", status="PROVISIONAL", site="kamioka",
        provenance=prov + " VARIANT 'engine': engine F on both hemispheres, muon_bending off "
        "(no fan-out); the MC is not used.",
    )
    print("wrote", os.path.join(RUN, "kamioka_3D_provisional_engine.npz"))

def cmd_demo():
    """daemonflux(Kamioka) x table vs Honda solmin/solmax: ratio maps and figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from daemonflux import Flux
    from daemonflux.geomagnetic import Tabulated3DModel
    import diag_ew_charge_fourier as fz

    tabs = {
        "provisional": os.path.join(RUN, "kamioka_3D_provisional.npz"),
        "engine": os.path.join(RUN, "kamioka_3D_provisional_engine.npz"),
    }
    H = {k: dict(np.load(v)) for k, v in HONDA.items()}
    h0 = H["solmin"]
    az_c = fz.honda_azimuth_to_compass(h0["azlo"].astype(float) + 15.0)
    czc = h0["czlo"] + 0.05
    flux = Flux(location="kamioka", debug=0)
    quant = {"nue": "nue", "antinue": "antinue", "numu": "numu", "antinumu": "antinumu"}
    models = {k: Tabulated3DModel(v) for k, v in tabs.items()}
    models["1D"] = None

    def eval_map(name, sp, E):
        flux.set_geomagnetic_model(models[name])
        out = np.zeros((len(czc), len(az_c)))
        for i, c in enumerate(czc):
            zen = float(np.degrees(np.arccos(c)))
            if models[name] is None and zen > 90.0:
                zen = 180.0 - zen  # the 1D model is up/down symmetric
            for j, a in enumerate(az_c):
                v = flux.flux(np.array([E]), zen, quant[sp], azimuth_deg=float(a))
                out[i, j] = float(np.squeeze(v)) / E**3 * 1e4  # /(m^2 s sr GeV)
        return out

    def honda_map(h, sp, E):
        out = np.zeros((len(czc), len(az_c)))
        for i in range(len(czc)):
            for j in range(len(az_c)):
                out[i, j] = _lat(h[HKEY[sp]][i][j], h["E"], E)
        return out

    R, RN = {}, {}
    L = ["**model/Honda ratio, medians over azimuth bins (rms over azimuth in brackets); "
         "model = daemonflux(kamioka) x table (PROVISIONAL)**", "",
         "| table | Honda | species | E | up |cz| 0.5-1 | up 0.2-0.5 | up 0-0.2 | down 0-0.2 | down 0.2-0.5 | down 0.5-1 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    LN = ["", "**same, normalised to each case's median ratio at down-going cos Z > 0.5 (removes "
          "the absolute normalisation of daemonflux's 1D flux at these energies: shape only)**", "",
          L[2], L[3]]
    regions = [(-1.0, -0.5), (-0.5, -0.2), (-0.2, 0.0), (0.0, 0.2), (0.2, 0.5), (0.5, 1.0)]
    for E in (0.3, 0.5, 1.0):
        for sp in MC_SPECIES:
            maps = {n: eval_map(n, sp, E) for n in models}
            for hk in ("solmin", "solmax"):
                hm = honda_map(H[hk], sp, E)
                for n, m in maps.items():
                    r = m / hm
                    R[(n, hk, sp, E)] = r
                    RN[(n, hk, sp, E)] = r / np.median(r[czc > 0.5])
                    cells = []
                    for lo, hi in regions:
                        sel = (czc > lo) & (czc < hi)
                        rr = r[sel]
                        cells.append(f"{np.median(rr):.3f} ({rr.std():.3f})")
                    L.append(f"| {n} | {hk} | {sp} | {E} | " + " | ".join(cells) + " |")
                    cn = []
                    for lo, hi in regions:
                        sel = (czc > lo) & (czc < hi)
                        rr = RN[(n, hk, sp, E)][sel]
                        cn.append(f"{np.median(rr):.3f} ({rr.std():.3f})")
                    LN.append(f"| {n} | {hk} | {sp} | {E} | " + " | ".join(cn) + " |")
    L += LN
    open(os.path.join(RUN, "out", "demo_ratio_table.md"), "w").write("\n".join(L))
    os.makedirs(os.path.join(RUN, "figs"), exist_ok=True)
    for n in ("provisional", "engine", "1D"):
        for hk in ("solmax", "solmin"):
            fig, axs = plt.subplots(4, 3, figsize=(12, 11), sharex=True, sharey=True)
            for i, sp in enumerate(MC_SPECIES):
                for j, E in enumerate((0.3, 0.5, 1.0)):
                    ax = axs[i, j]
                    im = ax.pcolormesh(np.sort(az_c), czc, RN[(n, hk, sp, E)][:, np.argsort(az_c)],
                                       vmin=0.7, vmax=1.3, cmap="RdBu_r", shading="nearest")
                    ax.set_title(f"{sp}, {E} GeV", fontsize=9)
                    if j == 0:
                        ax.set_ylabel("cos Z")
                    if i == 3:
                        ax.set_xlabel("azimuth [deg]")
            fig.colorbar(im, ax=axs, label=f"model / Honda {hk}, normalised (down cos Z > 0.5 median = 1)")
            fig.suptitle(f"daemonflux(kamioka) x {n} 3D table vs Honda {hk}   [PROVISIONAL]")
            fig.savefig(os.path.join(RUN, "figs", f"ratio_map_{n}_{hk}.png"), dpi=90)
            plt.close(fig)
    # azimuth-averaged profiles vs cos Z
    fig, axs = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for j, E in enumerate((0.3, 0.5, 1.0)):
        for i, sp in enumerate(("numu", "nue")):
            ax = axs[i, j]
            for n, st in (("1D", ":"), ("provisional", "-"), ("engine", "-.")):
                ax.plot(czc, RN[(n, "solmax", sp, E)].mean(1), st, label=f"{n} / solmax")
            ax.plot(czc, RN[("provisional", "solmin", sp, E)].mean(1), "-", alpha=0.4, label="provisional / solmin")
            ax.axhline(1.0, color="k", lw=0.5)
            ax.set_title(f"{sp}, {E} GeV [PROVISIONAL]", fontsize=9)
            ax.set_ylim(0.4, 1.6)
            if i == 1:
                ax.set_xlabel("cos Z")
            if j == 0:
                ax.set_ylabel("model / Honda, normalised (az. mean)")
    axs[0, 0].legend(fontsize=7)
    fig.savefig(os.path.join(RUN, "figs", "profile_vs_cosz.png"), dpi=90)
    plt.close(fig)
    print("\n".join(L))

def cmd_residual():
    """Shape residual of the engine table vs Honda solmax -> out/engine_sigma.npz.

    model = daemonflux(kamioka) x ``kamioka_3D_provisional_engine.npz``; per species and coarse
    energy bin the ratio model/Honda is normalised to its median over the down-going cos Z > 0.5
    cells (removes daemonflux's absolute level), then |median - 1| over azimuth, the two Honda
    cos Z bins of a coarse bin and the energy bins of the band (E_BANDS)."""
    from daemonflux import Flux
    from daemonflux.geomagnetic import Tabulated3DModel
    import diag_ew_charge_fourier as fz

    h = dict(np.load(HONDA["solmax"]))
    az_c = fz.honda_azimuth_to_compass(h["azlo"].astype(float) + 15.0)
    czc = h["czlo"] + 0.05
    flux = Flux(location="kamioka", debug=0)
    flux.set_geomagnetic_model(Tabulated3DModel(os.path.join(RUN, "kamioka_3D_provisional_engine.npz")))
    ec = np.sqrt(E_EDGES[:-1] * E_EDGES[1:])
    ratio = np.zeros((4, 10, 20, 12))
    for k, sp in enumerate(MC_SPECIES):
        for ie, E in enumerate(ec):
            for i, c in enumerate(czc):
                zen = float(np.degrees(np.arccos(c)))
                for j, a in enumerate(az_c):
                    v = float(np.squeeze(flux.flux(np.array([E]), zen, sp, azimuth_deg=float(a)))) / E**3 * 1e4
                    ratio[k, ie, i, j] = v / _lat(h[HKEY[sp]][i][j], h["E"], E)
            ratio[k, ie] /= np.median(ratio[k, ie][czc > 0.5])
    sigma = np.zeros((4, 4, 10))
    for k in range(4):
        for ib, (lo, hi) in enumerate(E_BANDS):
            ies = [ie for ie, e in enumerate(ec) if lo <= e < hi]
            for ic in range(10):
                r = ratio[k][ies][:, 2 * ic : 2 * ic + 2, :]
                sigma[k, ib, ic] = abs(np.median(r) - 1.0)
    np.savez(os.path.join(RUN, "out", "engine_sigma.npz"), sigma=sigma, ratio=ratio, species=np.array(MC_SPECIES))
    L = ["| species | E band [GeV] | " + " | ".join(f"cos Z {-1 + 0.2 * i:+.1f}..{-0.8 + 0.2 * i:+.1f}" for i in range(10)) + " |",
         "|---|---|" + "---|" * 10]
    for k, sp in enumerate(MC_SPECIES):
        for ib, (lo, hi) in enumerate(E_BANDS):
            L.append(f"| {sp} | {lo:g}-{hi:g} | " + " | ".join(f"{100 * x:.0f}%" for x in sigma[k, ib]) + " |")
    open(os.path.join(RUN, "out", "engine_sigma.md"), "w").write("\n".join(L))
    print("\n".join(L))


def main():
    c, a = sys.argv[1], sys.argv[2:]
    if c == "validate":
        return cmd_validate(*a)
    if c == "table":
        return cmd_table()
    if c == "demo":
        return cmd_demo()
    if c == "residual":
        return cmd_residual()
    {"maps-dn": cmd_maps_dn, "engine": cmd_engine}[c](*[int(x) for x in a])


if __name__ == "__main__":
    main()
