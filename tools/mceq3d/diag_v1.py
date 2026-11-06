"""Final Kamioka table v1 (INTEGRATION.md 12):  ``$W/runs/integ/out/kamioka_3D_v1.npz``.

    python diag_v1.py fit        muon-bend model test on the down-going R table -> out/rfit.npz
    python diag_v1.py build      the table
    python diag_v1.py validate   Honda shape tables, continuity, azimuth observables

Down-going E < 2.5 GeV: MC G = s/(kappa B) with G_err_total; E >= 2.5 GeV: engine (the two are compared, not blended).
Up-going: engine (cone mirror) x R_up, sigma = sigma_engine (+) sigma_R.
R_up (route a, see 12.2): R_s - 1 = delta_s(E, cos Z) q_s [beta . grad ln F_off], beta = d x B (at the production point Q for
up-going), q = parent muon charge sign, delta fitted on the down-going R table.
"""

import importlib.util  # noqa: F401
import hashlib
import os
import sys
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import diag_integ as D  # noqa: E402

W = D.W
RUN = D.RUN
OUT = os.path.join(RUN, "out")
G_FILE = os.path.join(W, "runs", "m4", "out", "kamioka_G_down_v1.npz")
R_FILE = os.path.join(W, "runs", "m4", "out", "kamioka_Rmu_v1.npz")
Q_SIGN = {
    "nue": +1,
    "antinue": -1,
    "numu": -1,
    "antinumu": +1,
}  # parent-muon charge sign
LAT, LON = D.LAT, D.LON
DATE = datetime(2020, 1, 1)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def field_enu_uT(lat, lon, h_km=20.0):
    """IGRF (ppigrf) field (N, E, U) in uT at (lat, lon, h)."""
    from muon_bending import local_field_enu

    be, bn, bu = (
        np.asarray(local_field_enu(lat, lon, DATE, h_km=h_km)) * 100.0
    )  # gauss -> uT
    return np.array([bn, be, bu])


def tangent(cz, az_deg):
    th = np.arccos(cz)
    ph = np.radians(az_deg)
    n = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    eth = np.array([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)])
    eph = np.array([-np.sin(ph), np.cos(ph), 0.0])
    return n, eth, eph, th


def bend_vector(cz, az_deg, B_nev):
    """beta = d x B with d = -n (neutrino travel direction), components on the detector (e_theta, e_phi)."""
    n, eth, eph, th = tangent(cz, az_deg)
    beta = np.cross(-n, B_nev)
    return np.linalg.norm(beta), beta @ eth, beta @ eph, th


# ---------------------------------------------------------------- fit of delta on the down-going R table
def cmd_fit():
    d = np.load(R_FILE, allow_pickle=True)
    R, Re, Fon = d["R"], d["R_err"], d["F_on"]
    Foff = Fon / R
    ee = d["E_edges"]
    ec = np.sqrt(ee[:-1] * ee[1:])
    cze = d["cosZ_edges"]
    czc = 0.5 * (cze[:-1] + cze[1:])
    azc = 30 + 60 * np.arange(6)
    sp = [str(s) for s in d["species"]]
    BK = field_enu_uT(LAT, LON, 20.0)
    delta = np.zeros((4, 7, 5))
    ddel = np.zeros_like(delta)
    P = np.zeros((4, 7, 5, 6))
    th = np.arccos(czc)
    for k, s in enumerate(sp):
        for ie in range(7):
            lf = np.log(Foff[k, ie])
            dth = np.gradient(lf, th, axis=0)
            for ic in range(5):
                for ia in range(6):
                    dph = (lf[ic, (ia + 1) % 6] - lf[ic, (ia - 1) % 6]) / np.radians(
                        120.0
                    )
                    bn, bth, bph, t = bend_vector(czc[ic], azc[ia], BK)
                    P[k, ie, ic, ia] = Q_SIGN[s] * (
                        bth * dth[ic, ia] + bph * dph / np.sin(t)
                    )
            y = R[k, ie] - 1
            w = 1.0 / np.maximum(Re[k, ie], 1e-3) ** 2
            for ic in range(5):
                den = (w[ic] * P[k, ie, ic] ** 2).sum()
                delta[k, ie, ic] = (w[ic] * P[k, ie, ic] * y[ic]).sum() / den
                ddel[k, ie, ic] = 1.0 / np.sqrt(den)
    pred = delta[..., None] * P
    chi0 = (((R - 1) / np.maximum(Re, 1e-3)) ** 2).sum()
    chi1 = (((R - 1 - pred) / np.maximum(Re, 1e-3)) ** 2).sum()
    # fractional unexplained part where the effect is significant
    sig = np.abs(R - 1) > 2 * Re
    frac = np.sqrt(np.mean((R - 1 - pred)[sig] ** 2) / np.mean((R - 1)[sig] ** 2))
    # leave-one-azimuth-out predictive check (delta from the other 5 azimuths)
    loo = 0.0
    n = 0
    for k in range(4):
        for ie in range(7):
            for ic in range(5):
                for ia in range(6):
                    m = np.arange(6) != ia
                    w = 1.0 / np.maximum(Re[k, ie, ic], 1e-3) ** 2
                    dl = (w[m] * P[k, ie, ic][m] * (R[k, ie, ic][m] - 1)).sum() / (
                        w[m] * P[k, ie, ic][m] ** 2
                    ).sum()
                    loo += (
                        (R[k, ie, ic, ia] - 1 - dl * P[k, ie, ic, ia])
                        / Re[k, ie, ic, ia]
                    ) ** 2
                    n += 1
    # B_perp alone (no gradient): correlation of (R-1) with q |beta| at fixed (E, cz)
    bperp = np.array(
        [[bend_vector(czc[ic], azc[ia], BK)[0] for ia in range(6)] for ic in range(5)]
    )
    corr = []
    for k, s in enumerate(sp):
        for ie in range(7):
            for ic in range(5):
                a, b = (R[k, ie, ic] - 1), Q_SIGN[s] * bperp[ic]
                if a.std() > 0:
                    corr.append(np.corrcoef(a, b)[0, 1])
    res = dict(
        delta=delta,
        delta_err=ddel,
        E_centres=ec,
        cz_centres=czc,
        species=np.array(sp),
        chi2_null=chi0,
        chi2_model=chi1,
        n_data=R.size,
        n_par=delta.size,
        frac_unexplained=frac,
        loo_chi2=loo,
        loo_n=n,
        corr_bperp_median=np.median(corr),
        corr_bperp_absmean=np.mean(np.abs(corr)),
        B_K_nev_uT=BK,
    )
    np.savez(os.path.join(OUT, "rfit.npz"), **res)
    print(
        f"chi2 null (R=1) {chi0:.0f}; model {chi1:.0f} for {R.size} points, {delta.size} parameters; "
        f"leave-one-azimuth-out chi2 {loo:.0f}/{n}; unexplained fraction of |R-1| (>2 sigma cells) {frac:.2f}"
    )
    print(
        f"correlation of R-1 with q*B_perp over azimuth: median {np.median(corr):.2f}, mean|r| {np.mean(np.abs(corr)):.2f}"
    )
    print("delta[species][E] at cos Z bins:")
    for k, s in enumerate(sp):
        print(s, [np.round(delta[k, ie], 4).tolist() for ie in (2, 4, 6)])
    print(
        "species ratio check: median over (E,cz) of delta_numu/delta_nue =",
        np.median(delta[2] / delta[0]),
        " antinue/nue",
        np.median(delta[1] / delta[0]),
        " antinumu/numu",
        np.median(delta[3] / delta[2]),
    )


# ---------------------------------------------------------------- R_up and the table
def _delta_interp(rf, sp, e, czq):
    """delta_s and its error at engine energies ``e`` and local zenith cosine czq (bilinear; flat outside)."""
    ec, czc = rf["E_centres"], rf["cz_centres"]
    k = list(rf["species"]).index(sp)
    out, err = np.zeros(len(e)), np.zeros(len(e))
    le = np.log(e)
    for tab, o in ((rf["delta"][k], out), (rf["delta_err"][k], err)):
        # first cos Z (linear, clamped), then log E (linear, clamped)
        v = np.array([np.interp(czq, czc, tab[ie]) for ie in range(len(ec))])
        o[:] = np.interp(le, np.log(ec), v)
    out[e > 2.5119] = 0.0  # no response measured above the R table: R_up = 1
    return out, err


def r_up_fine(fine, F_up, rf, mirror_R=None):
    """R_up(row, az, E) per species (route a) and its statistical error; F_up[sp] (10, 12, nE) = fine up-going
    engine F (cone mirror applied, muon bend off)."""
    from mceq3d_flux import farside_production
    import geomag_backtrace as gb

    e = fine["e"]
    cz = fine["cz_up"]
    th = np.arccos(cz)
    up_d, north_d, east_d = gb._local_frame(LAT, LON)
    R = {}
    Rerr = {}
    geo = {}
    for i, c in enumerate(cz):
        for j, a in enumerate(D.AZ12):
            lat_q, lon_q, cq, aq = farside_production(LAT, LON, c, a)
            Bq = field_enu_uT(lat_q, lon_q, 20.0)  # (N, E, U) at Q
            uq, nq, eq = gb._local_frame(lat_q, lon_q)
            bv = Bq[0] * nq + Bq[1] * eq + Bq[2] * uq
            geo[(i, j)] = (
                np.array([bv @ north_d, bv @ east_d, bv @ up_d]),
                cq,
                lat_q,
                lon_q,
            )
    for sp in D.MC_SPECIES:
        lf = np.log(F_up[sp])  # (10, 12, nE)
        dth = np.gradient(lf, th, axis=0)
        dph = (np.roll(lf, -1, axis=1) - np.roll(lf, 1, axis=1)) / np.radians(60.0)
        r = np.ones_like(lf)
        re = np.zeros_like(lf)
        for i, c in enumerate(cz):
            for j, a in enumerate(D.AZ12):
                Bdet, cq, _, _ = geo[(i, j)]
                bn, bth, bph, t = bend_vector(c, a, Bdet)
                dl, de = _delta_interp(rf, sp, e, cq)
                # the muon bends at Q, where the production direction is the reflection of the detector direction
                # (theta_Q = 180 - theta_det along the chord family, azimuth unchanged): the theta-gradient with respect to
                # the muon direction at Q has the opposite sign to d/dtheta_det, the azimuth gradient is unchanged.
                P = Q_SIGN[sp] * (-bth * dth[i, j] + bph * dph[i, j] / np.sin(t))
                r[i, j] = 1.0 + dl * P
                re[i, j] = np.abs(de * P)
        R[sp], Rerr[sp] = r, re
    return R, Rerr, geo


def r_mirror_fine(fine, rd):
    """Plain mirror (route b): the down-going R table at the same |cos Z|, azimuth (coarse cell), interpolated in E."""
    e = fine["e"]
    ee = rd["E_edges"]
    ec = np.sqrt(ee[:-1] * ee[1:])
    out = {}
    for k, sp in enumerate(D.MC_SPECIES):
        r = np.ones((10, 12, len(e)))
        for i, c in enumerate(fine["cz_up"]):
            ic = min(int(abs(c) / 0.2), 4)
            for j in range(12):
                ia = j // 2
                r[i, j] = np.interp(np.log(e), np.log(ec), rd["R"][k, :, ic, ia])
                r[i, j, e > 2.5119] = 1.0
        out[sp] = r
    return out


def coarse_up(fine, arr):
    """Weighted (base x dE) coarse average of a fine up-going array arr[sp] (10, 12, nE) -> (4, 10, 5, 6)."""
    e, de = fine["e"], fine["e_widths"]
    out = np.zeros((4, 10, 5, 6))
    for k, sp in enumerate(D.SPECIES):
        short = sp[6:]
        base = fine[f"base_up_{sp}"]
        a = arr[short]
        for ie in range(10):
            m = (e >= D.E_EDGES[ie]) & (e < D.E_EDGES[ie + 1])
            for ic in range(5):
                sel = slice(2 * ic, 2 * ic + 2)
                w = base[sel][:, m] * de[m]
                for ia in range(6):
                    v = 0.5 * (
                        a[sel][:, 2 * ia, :][:, m] + a[sel][:, 2 * ia + 1, :][:, m]
                    )
                    out[k, ie, ic, ia] = (v * w).sum() / w.sum()
    return out


def cmd_build():
    from daemonflux.geomagnetic import Tabulated3DModel  # noqa: F401

    D.MIRROR = True
    fine = np.load(os.path.join(OUT, "engine_F_fine.npz"))
    rf = dict(np.load(os.path.join(OUT, "rfit.npz")))
    rd = dict(np.load(R_FILE, allow_pickle=True))
    gd = dict(np.load(G_FILE, allow_pickle=True))
    # fine up-going engine F with the cone mirror
    F_up = {}
    for sp in D.SPECIES:
        F_up[sp[6:]] = (
            fine[f"flux_up_{sp}"] / fine[f"base_up_{sp}"][:, None, :]
        ) * D.mirror_ratio(sp)
    Ra, Rerr, geo = r_up_fine(fine, F_up, rf)
    Rb = r_mirror_fine(fine, rd)
    F_a = {sp: F_up[sp] * Ra[sp] for sp in F_up}
    eu0 = coarse_up(fine, F_up)  # engine mirror, no muon bend
    eu = coarse_up(fine, F_a)  # route a
    R_a_c = eu / eu0
    R_b_c = coarse_up(fine, {sp: F_up[sp] * Rb[sp] for sp in F_up}) / eu0
    s_stat = coarse_up(fine, {sp: Rerr[sp] for sp in Rerr})
    # sigma_R = stat (+) 32% of the effect (Kamioka closure residual) (+) |route a - route b|
    f_clos = float(rf["frac_unexplained"])
    sig_R = np.sqrt(s_stat**2 + (f_clos * (R_a_c - 1.0)) ** 2 + (R_a_c - R_b_c) ** 2)
    # engine rel. error (max(5%, Honda residual)) and total
    rel_eng = D.engine_rel_err()
    ed = D.engine_coarse(fine, "down")
    g = gd["G_coarse"]  # (4, 10 E, 5 cz, 6 az)
    gerr = gd["G_err_total_coarse"]
    fac = np.zeros((4, 10, 10, 6))
    dfac = np.zeros_like(fac)
    source = np.zeros(
        (10, 10), dtype=int
    )  # 0 MC down, 1 engine down (E >= 2.5), 3 engine mirror x R_up (up-going)
    fac[:, :, :5] = eu
    dfac[:, :, :5] = np.sqrt((rel_eng[:, :, :5, None]) ** 2 + sig_R**2) * eu
    source[:, :5] = 3
    fac[:, :, 5:] = g
    dfac[:, :, 5:] = gerr
    fac[:, 7:, 5:] = ed[:, 7:]
    dfac[:, 7:, 5:] = rel_eng[:, 7:, 5:, None] * ed[:, 7:]
    source[7:, 5:] = 1
    assert np.all(np.isfinite(fac)) and np.all(np.isfinite(dfac))
    # MC vs engine in the 2.5-4 GeV bin (index 7), per cos Z band, species-wise, azimuth mean
    ratio72 = g[:, 7].mean(-1) / ed[:, 7].mean(-1)  # (4, 5)
    err72 = np.sqrt((gerr[:, 7] ** 2).sum(-1)) / 6 / ed[:, 7].mean(-1)
    prov = (
        "Kamioka 3D factor v1. Down-going E < 2.5 GeV: MC G = s/(kappa B) (milestone 4, G_err_total = stat (+) reference-shape "
        "systematic); down-going E >= 2.5 GeV: mceq3d engine F = 3D/B (not blended; MC/engine compared in INTEGRATION.md 12.1). "
        "Up-going: engine F (production cone from the mirrored down-going direction, |cos Z| <= 0.35) x R_up (muon-bend response "
        "transferred with the projection model, route a) ; errors sqrt(sigma_engine^2 + sigma_R^2) with sigma_engine = "
        "max(5%, |shape residual vs Honda solmax|) (a cross-model systematic). Reference B = engine curved-atmosphere 1D "
        "(GSF, SIBYLL23D, v142p, no cutoff), both hemispheres."
    )
    p = os.path.join(OUT, "kamioka_3D_v1.npz")
    np.savez(
        p,
        species=np.array(D.MC_SPECIES),
        e_edges=D.E_EDGES,
        cosz_edges=D.CZ_EDGES,
        az_edges=D.AZ_EDGES,
        factor=fac,
        factor_error=dfac,
        source=source,
        label="kamioka_3D_v1",
        status="v1 (up-going and E >= 2.5 GeV from the engine; see INTEGRATION.md 12)",
        site="kamioka",
        provenance=prov,
        input_sha256_G_down=sha(G_FILE),
        input_sha256_Rmu=sha(R_FILE),
        input_sha256_base1d=sha(os.path.join(OUT, "base1d_kamioka.npz")),
        input_sha256_engine_F_fine=sha(os.path.join(OUT, "engine_F_fine.npz")),
        input_sha256_updiag_modes=sha(os.path.join(OUT, "updiag_modes.npz")),
        input_sha256_updiag_modes_dn6=sha(os.path.join(OUT, "updiag_modes_dn6.npz")),
        db_label="v142p",
        db_sha256=str(gd["db_sha256"][0]),
        primary_model_mc=str(gd["primary_model"][0]),
        primary_model_engine="GlobalSplineFitBeta (GSF), SIBYLL23D",
        phi0_GV=0.55,
        phi0_note=str(gd["phi0_note"][0]),
        field="IGRF-14 (ppigrf 2.1.0, IGRF14.shc), 2020-01-01, 20 km at the production point (up-going bend)",
        build_date=datetime.now().isoformat(timespec="seconds"),
        route_R_up=(
            "hybrid: R_mirror (azimuth mean always from the measured R) + w(E,|cosZ|) x azimuth-odd difference (a' - mirror), w = 1 at |cosZ| <= 0.15 / E <= 0.8 GeV, linear to 0 at 0.25 / 1.25; a' = "
            if tag == "v1c"
            else ""
        )
        + "a: projection model R-1 = delta_s(E, cosZ_local) q_s (d x B_Q) . grad ln F_off, delta fitted on the "
        "down-going R table; the literal B_perp-only model fails (12.2)",
        route_down_high_E="engine for E >= 2.5 GeV (MC/engine compared, not blended)",
        route_up_cone="mirror cone correction F_cone/F_single, |cosZ| <= 0.35",
        kappa_coarse=gd["kappa_coarse"],
        kappa_err_coarse=gd["kappa_err_coarse"],
        R_up_minus_1=R_a_c - 1.0,
        R_up_mirror_minus_1=R_b_c - 1.0,
        sigma_R=sig_R,
        mc_over_engine_2p5_4=ratio72,
        mc_over_engine_2p5_4_err=err72,
        delta=rf["delta"],
        delta_err=rf["delta_err"],
    )
    print("wrote", p)
    ec = np.sqrt(D.E_EDGES[:-1] * D.E_EDGES[1:])
    L = [
        "**rms over azimuth and the E bins of the band of (R_up - 1): route a | route b (plain mirror) | median sigma_R, per species x E band x up cos Z bin (R_up - 1 is odd in azimuth, its median is ~0)**",
        "",
        "| species | E band | "
        + " | ".join(
            f"cos Z {-1 + 0.2 * i:+.1f}..{-0.8 + 0.2 * i:+.1f}" for i in range(5)
        )
        + " |",
        "|---|---|" + "---|" * 5,
    ]
    for k, sp in enumerate(D.MC_SPECIES):
        for lo, hi in D.E_BANDS:
            ies = [ie for ie, e in enumerate(ec) if lo <= e < hi and e < 2.5]
            if not ies:
                continue
            cells = []
            for ic in range(5):
                a = np.sqrt(np.mean((R_a_c[k, ies, ic] - 1) ** 2))
                b = np.sqrt(np.mean((R_b_c[k, ies, ic] - 1) ** 2))
                sg = np.median(sig_R[k, ies, ic])
                cells.append(f"{100 * a:+.1f} / {100 * b:+.1f} / {100 * sg:.1f}%")
            L.append(f"| {sp} | {lo:g}-{hi:g} | " + " | ".join(cells) + " |")
    L += [
        "",
        "**MC / engine, 2.5-4 GeV bin, azimuth mean, per cos Z band (0-0.2 ... 0.8-1)**",
        "",
        "| species | "
        + " | ".join(f"{0.2 * i:.1f}-{0.2 * i + 0.2:.1f}" for i in range(5))
        + " |",
        "|---|---|---|---|---|---|",
    ]
    for k, sp in enumerate(D.MC_SPECIES):
        L.append(
            f"| {sp} | "
            + " | ".join(f"{ratio72[k, i]:.3f} +- {err72[k, i]:.3f}" for i in range(5))
            + " |"
        )
    open(os.path.join(OUT, "v1_build_tables.md"), "w").write("\n".join(L))
    print("\n".join(L))


def cmd_validate(tag="v1"):
    import analyse_m3b as M
    import diag_ew_charge_fourier as fz
    from daemonflux import Flux
    from daemonflux.geomagnetic import Tabulated3DModel
    from diag_rigidity_scan import _honda_pattern

    path = os.path.join(OUT, f"kamioka_3D_{tag}.npz")
    model = Tabulated3DModel(path)
    H = {k: dict(np.load(v)) for k, v in D.HONDA.items()}
    ax = fz.geomagnetic_ew_axis()
    flux = Flux(location="kamioka", debug=0)
    flux.set_geomagnetic_model(path)  # load through the Flux API
    L = []
    # ---- A. shape tables
    h0 = H["solmin"]
    az_c = fz.honda_azimuth_to_compass(h0["azlo"].astype(float) + 15.0)
    czc = h0["czlo"] + 0.05
    regions = [
        (-1.0, -0.5),
        (-0.5, -0.2),
        (-0.2, 0.0),
        (0.0, 0.2),
        (0.2, 0.5),
        (0.5, 1.0),
    ]
    L += [
        "**A. daemonflux(kamioka) x v1 / Honda, normalised to the median ratio at down-going cos Z > 0.5; median over azimuth "
        "(columns: up 0.5-1 | up 0.2-0.5 | up 0-0.2 | down 0-0.2 | down 0.2-0.5 | down 0.5-1); Honda solmax [solmin]**",
        "",
        "| species | E | up .5-1 | up .2-.5 | up 0-.2 | down 0-.2 | down .2-.5 | down .5-1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for E in (0.3, 0.5, 1.0):
        for sp in D.MC_SPECIES:
            mm = np.zeros((20, 12))
            for i, c in enumerate(czc):
                zen = float(np.degrees(np.arccos(c)))
                for j, a in enumerate(az_c):
                    mm[i, j] = (
                        float(
                            np.squeeze(
                                flux.flux(np.array([E]), zen, sp, azimuth_deg=float(a))
                            )
                        )
                        / E**3
                        * 1e4
                    )
            cells = []
            for lo, hi in regions:
                vals = []
                for hk in ("solmax", "solmin"):
                    hm = np.array(
                        [
                            [
                                D._lat(H[hk][D.HKEY[sp]][i][j], H[hk]["E"], E)
                                for j in range(12)
                            ]
                            for i in range(20)
                        ]
                    )
                    r = mm / hm
                    r = r / np.median(r[czc > 0.5])
                    vals.append(np.median(r[(czc > lo) & (czc < hi)]))
                cells.append(f"{vals[0]:.3f} [{vals[1]:.3f}]")
            L.append(f"| {sp} | {E} | " + " | ".join(cells) + " |")
    # ---- B. continuity
    t = np.load(path, allow_pickle=True)
    fac = t["factor"]
    ec = np.sqrt(D.E_EDGES[:-1] * D.E_EDGES[1:])
    nz = fac.shape[2]
    fine_t = nz == 20
    L += [
        "",
        "**B1. horizon continuity at bin values: table az-mean factor at cos Z = -|c| and +|c| (up/down ratio), the interpolated model at the "
        "same cos Z (sin 0 cut: each hemisphere separately) and the Honda up/down flux ratio (solmax, same |cos Z|)**",
        "",
        "| species | E | |cos Z| | F(up) | F(down) | table up/down | model F(-c)/F(+c) | Honda U/D |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k, sp in enumerate(D.MC_SPECIES):
        for E in (0.3, 0.5, 1.0, 2.0):
            ie = int(np.argmin(abs(ec - E)))
            for ac in (0.05, 0.15, 0.25) if fine_t else (0.1,):
                i = int(round(ac * 10 - 0.5)) if fine_t else int(ac * 5)
                fu = fac[k, ie, nz // 2 - 1 - i].mean()
                fd = fac[k, ie, nz // 2 + i].mean()
                Ee = ec[ie]
                mu = float(
                    np.mean(
                        model.factor(
                            sp, np.array([Ee]), float(np.degrees(np.arccos(-ac))), None
                        )
                    )
                )
                md = float(
                    np.mean(
                        model.factor(
                            sp, np.array([Ee]), float(np.degrees(np.arccos(ac))), None
                        )
                    )
                )
                ih = int(round(ac * 10 - 0.5))
                hu = np.mean(
                    [
                        D._lat(H["solmax"][D.HKEY[sp]][9 - ih][j], H["solmax"]["E"], Ee)
                        for j in range(12)
                    ]
                )
                hd = np.mean(
                    [
                        D._lat(
                            H["solmax"][D.HKEY[sp]][10 + ih][j], H["solmax"]["E"], Ee
                        )
                        for j in range(12)
                    ]
                )
                L.append(
                    f"| {sp} | {Ee:.2f} | {ac} | {fu:.3f} | {fd:.3f} | {fu / fd:.3f} | {mu / md:.3f} | {hu / hd:.3f} |"
                )
    L += [
        "",
        "**B2. continuity at the switches: model F(2.4)/F(2.6 GeV) (R_up -> 1 up-going; MC -> engine down-going for v1) ; F(3.8)/F(4.2 GeV) (MC -> engine, v1b), az mean, per cos Z (down 0.1 / 0.3 / 0.5 / 0.7 / 0.9; up -0.1 ... -0.9); the "
        "table switches MC -> engine (down) and R_up -> 1 (up) there**",
        "",
        "| species | down 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | up -0.1 | -0.3 | -0.5 | -0.7 | -0.9 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k, sp in enumerate(D.MC_SPECIES):
        row = []
        for c in (0.1, 0.3, 0.5, 0.7, 0.9, -0.1, -0.3, -0.5, -0.7, -0.9):
            z = float(np.degrees(np.arccos(c)))
            a = np.mean(model.factor(sp, np.array([2.4]), z, None))
            b = np.mean(model.factor(sp, np.array([2.6]), z, None))
            a2 = np.mean(model.factor(sp, np.array([3.8]), z, None))
            b2 = np.mean(model.factor(sp, np.array([4.2]), z, None))
            row.append(f"{a / b:.3f} ; {a2 / b2:.3f}")
        L.append(f"| {sp} | " + " | ".join(row) + " |")
    # ---- C. azimuth observables
    names = list(M.OBS)
    sel = ["a1_rel", "dphi", "NS", "WE"]
    L += [
        "",
        "**C. azimuth observables of the pattern at cos Z = +-0.05, +-0.15, +-0.25 (12 azimuth bins, `analyse_m3b` definitions): v1 | Honda solmax | Honda solmin**",
        "",
        "| species | cos Z | E | a1/a0 | dphi [deg] | N/S | W/E |",
        "|---|---|---|---|---|---|---|",
    ]
    for sp in D.MC_SPECIES:
        for c in (0.05, -0.05, 0.15, -0.15, 0.25, -0.25):
            zen = float(np.degrees(np.arccos(c)))
            for E in (0.3, 0.5, 1.0):
                pat = np.array(
                    [
                        float(
                            np.squeeze(model.factor(sp, np.array([E]), zen, float(a)))
                        )
                        for a in D.AZ12
                    ]
                )
                om = M.observables(pat, ax)
                cells = []
                for o in sel:
                    v = [om[names.index(o)]]
                    for hk in ("solmax", "solmin"):
                        v.append(
                            M.observables(
                                _honda_pattern(H[hk], D.HKEY[sp], c, E, fz), ax
                            )[names.index(o)]
                        )
                    cells.append(" / ".join(f"{x:.3f}" for x in v))
                L.append(f"| {sp} | {c:+.2f} | {E} | " + " | ".join(cells) + " |")
    # ---- D. Flux API check
    z = [80.0, 100.0]
    Eg = np.array([0.5, 1.0, 3.0, 20.0])
    base = Flux(location="kamioka", debug=0)
    L += [
        "",
        "**D. Flux.set_geomagnetic_model(npz) check: numu at azimuth 100 deg, ratio to the 1D flux; numuratio, flavorratio recomputed; errors scaled**",
        "",
        "| zenith | E | numu 3D/1D | numuflux 3D/1D | numuratio 3D/1D | flavorratio 3D/1D | error/1D error (numu) | rel. table error (numu) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for zz in z:
        z1 = 180.0 - zz if zz > 90 else zz
        for k, E in enumerate(Eg):
            e1 = np.array([E])
            row = []
            for q in ("numu", "numuflux", "numuratio", "flavorratio"):
                row.append(
                    float(
                        np.squeeze(
                            flux.flux(e1, zz, q, azimuth_deg=100.0)
                            / base.flux(e1, z1, q)
                        )
                    )
                )
            er = float(
                np.squeeze(
                    flux.error(e1, zz, "numu", azimuth_deg=100.0)
                    / base.error(e1, z1, "numu")
                )
            )
            te = float(
                np.squeeze(
                    flux.geomagnetic_factor_error(e1, zz, "numu", azimuth_deg=100.0)
                )
            )
            L.append(
                f"| {zz:g} | {E} | "
                + " | ".join(f"{x:.3f}" for x in row)
                + f" | {er:.3f} | {te:.3f} |"
            )
    open(os.path.join(OUT, f"{tag}_validation.md"), "w").write("\n".join(L))
    print("\n".join(L))


# ---------------------------------------------------------------- v1b: E 5/dec x cos Z 0.1 x az 12
G_FILE_B = os.path.join(W, "runs", "m4", "out", "kamioka_G_down_v1b.npz")
E_MC_B = 8  # MC down-going bins 0..7 (< 3.98 GeV); engine above


def _harm12(v6, order=2):
    """6 values at az 15 + 60 i  ->  12 values at az 15 + 30 j (harmonics 0..order; last axis = azimuth)."""
    c = np.fft.rfft(v6, axis=-1) / 6.0
    ph0, ph = np.radians(15.0), np.radians(15.0 + 30.0 * np.arange(12))
    out = np.zeros(v6.shape[:-1] + (12,))
    for m in range(order + 1):
        w = 1.0 if m == 0 else 2.0
        # sample i at az 15+60i: v = sum_m w Re[c_m exp(i m (60 i deg))]; evaluate at az (relative to 15 deg)
        out += w * np.real(c[..., m, None] * np.exp(1j * m * (ph - ph0)))
    return out


def r_mirror_harm(fine, rd):
    """Plain mirror (route b) on the 12-azimuth grid: the measured down-going R table at the same |cos Z| (linear between
    the 0.2-wide bin centres, clamped), azimuth by a harmonic fit (orders 0-2) of the 6 measured bins, log E interpolation.
    """
    e = fine["e"]
    ee = rd["E_edges"]
    ec = np.sqrt(ee[:-1] * ee[1:])
    czc = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    out = {}
    for k, sp in enumerate(D.MC_SPECIES):
        # azimuth bins of the table are 0-60, 60-120 ... : centres 30 + 60 i; shift to the 15 + 60 i convention by harmonics
        Rm = rd["R"][k] - 1.0  # (7 E, 5 cz, 6 az)
        c = np.fft.rfft(Rm, axis=-1) / 6.0
        az12 = 15.0 + 30.0 * np.arange(12)
        h = np.zeros(Rm.shape[:-1] + (12,))
        for m in range(3):
            w = 1.0 if m == 0 else 2.0
            h += w * np.real(c[..., m, None] * np.exp(1j * m * np.radians(az12 - 30.0)))
        r = np.ones((10, 12, len(e)))
        for i, cz in enumerate(fine["cz_up"]):
            a = abs(cz)
            for j in range(12):
                v = np.array([np.interp(a, czc, h[ie, :, j]) for ie in range(len(ec))])
                r[i, j] = 1.0 + np.interp(np.log(e), np.log(ec), v)
                r[i, j, e > 2.5119] = 1.0
        out[sp] = r
    return out


def fine_avg(fine, arr, base):
    """E-band average on the table grid of a fine array arr (nrow, naz, nE) with weights base (nrow, nE) * dE -> (10 E, nrow, naz)."""
    e, de = fine["e"], fine["e_widths"]
    out = np.zeros((10, arr.shape[0], arr.shape[1]))
    for ie in range(10):
        m = (e >= D.E_EDGES[ie]) & (e < D.E_EDGES[ie + 1])
        w = base[:, m] * de[m]
        out[ie] = (arr[:, :, m] * w[:, None, :]).sum(-1) / w.sum(-1)[:, None]
    return out


def engine_down12(fine):
    """Engine down-going F on the v1b grid (4, 10 E, 10 cos Z rows 0.05 ... 0.95, 12 az): rows 0.05-0.25 are native (12 az); rows
    0.35-0.95 have 6 az (15 + 60 i) and are extended to the 12 centres with harmonics 0-2.
    """
    out = np.zeros((4, 10, 10, 12))
    for k, sp in enumerate(D.SPECIES):
        fh = fine[f"flux_hor_{sp}"] / fine[f"base_hor_{sp}"][:, None, :]
        fd = fine[f"flux_dn6_{sp}"] / fine[f"base_dn6_{sp}"][:, None, :]
        ah = fine_avg(fine, fh, fine[f"base_hor_{sp}"])  # (10, 3, 12)
        ad = fine_avg(fine, fd, fine[f"base_dn6_{sp}"])  # (10, 7, 6)
        out[k, :, :3] = ah
        out[k, :, 3:] = _harm12(ad)
    return out


W_CZ = (0.15, 0.25)  # weight 1 up to |cos Z| 0.15, 0 from 0.25 (closure test: 12.x)
W_E = (0.8, 1.25)  # weight 1 up to 0.8 GeV, 0 from 1.25 GeV


def hybrid_weight(cz_rows, e):
    """w(E, |cos Z|) = w_cz * w_E, linear tapers; shape (nrow, 1, nE)."""
    wc = np.clip((W_CZ[1] - np.abs(cz_rows)) / (W_CZ[1] - W_CZ[0]), 0.0, 1.0)
    we = np.clip((W_E[1] - e) / (W_E[1] - W_E[0]), 0.0, 1.0)
    return wc[:, None, None] * we[None, None, :]


def r_up_hybrid(fine, Ra, Rb):
    """v1c: R_up = R_b + w [(R_a - <R_a>_az) - (R_b - <R_b>_az)]; the azimuth mean always from the measured (mirror) R."""
    w = hybrid_weight(fine["cz_up"], fine["e"])
    return {
        sp: Rb[sp]
        + w
        * (
            (Ra[sp] - Ra[sp].mean(1, keepdims=True))
            - (Rb[sp] - Rb[sp].mean(1, keepdims=True))
        )
        for sp in Ra
    }


def cmd_build_v1b(tag="v1b"):
    D.MIRROR = True
    fine = np.load(os.path.join(OUT, "engine_F_fine.npz"))
    rf = dict(np.load(os.path.join(OUT, "rfit.npz")))
    rd = dict(np.load(R_FILE, allow_pickle=True))
    gd = dict(np.load(G_FILE_B, allow_pickle=True))
    assert list(gd["E_edges"]) == list(D.E_EDGES) or np.allclose(
        gd["E_edges"], D.E_EDGES
    )
    F_up = {}
    for sp in D.SPECIES:
        F_up[sp[6:]] = (
            fine[f"flux_up_{sp}"] / fine[f"base_up_{sp}"][:, None, :]
        ) * D.mirror_ratio(sp)
    Ra, Rerr, geo = r_up_fine(fine, F_up, rf)
    Rb = r_mirror_harm(fine, rd)
    eu0 = np.zeros((4, 10, 10, 12))
    eu = np.zeros_like(eu0)
    eub = np.zeros_like(eu0)
    era = np.zeros_like(eu0)
    sst = np.zeros_like(eu0)
    Rc = {}
    if tag == "v1c":
        Rc = r_up_hybrid(fine, Ra, Rb)
    for k, sp in enumerate(D.SPECIES):
        s = sp[6:]
        base = fine[f"base_up_{sp}"]
        eu0[k] = fine_avg(fine, F_up[s], base).transpose(0, 1, 2)
        if tag == "v1c":
            eu[k] = fine_avg(fine, F_up[s] * Rc[s], base)
        else:
            eu[k] = fine_avg(fine, F_up[s] * Ra[s], base)
        era[k] = fine_avg(fine, F_up[s] * Ra[s], base)
        eub[k] = fine_avg(fine, F_up[s] * Rb[s], base)
        sst[k] = fine_avg(fine, Rerr[s], base)
    R_a, R_b, R_c = era / eu0, eub / eu0, eu / eu0
    f_clos = float(rf["frac_unexplained"])
    sig_R = np.sqrt(sst**2 + (f_clos * (R_a - 1.0)) ** 2 + (R_a - R_b) ** 2)
    rel = D.engine_rel_err()  # (4, 10, 10 coarse cz)
    ed = engine_down12(fine)
    g, gerr = gd["G"], gd["G_err_total"]  # (4, 10 E, 10 cz, 12 az)
    fac = np.zeros((4, 10, 20, 12))
    dfac = np.zeros_like(fac)
    cz_c = np.arange(20) // 2
    fac[:, :, :10] = eu
    dfac[:, :, :10] = np.sqrt(rel[:, :, cz_c[:10], None] ** 2 + sig_R**2) * eu
    fac[:, :, 10:] = g
    dfac[:, :, 10:] = gerr
    fac[:, E_MC_B:, 10:] = ed[:, E_MC_B:]
    dfac[:, E_MC_B:, 10:] = rel[:, E_MC_B:, cz_c[10:], None] * ed[:, E_MC_B:]
    source = np.zeros((10, 20), dtype=int)
    source[:, :10] = 3
    source[E_MC_B:, 10:] = 1
    # sigma_model (12.z): (i) down-going, MC-filled (E < 3.98 GeV), every cos Z band of 0.2: r = <G>_az/<F>_az per species x E bin; if
    # |r - 1| > 2 sigma_r (sigma_r = MC G_err_total of the cells in quadrature / n / <F>; MC error only, no engine error), sigma_mod = |r - 1|
    # in quadrature (relative) on all azimuths of the band. (ii) up-going (engine-filled): sigma_level(species, E) = |r_vert - 1|,
    # r_vert from cos Z 0.6-1 (48 cells), on all up-going bins at that E if |r_vert - 1| > 2 sigma.
    sigma_model = np.zeros_like(dfac)
    sm_bins, lv_bins = [], []
    for ic in range(5):
        c0 = slice(2 * ic, 2 * ic + 2)
        rows = slice(10 + 2 * ic, 12 + 2 * ic)
        for ie in range(E_MC_B):
            fm = ed[:, ie, c0].mean((-1, -2))
            r = g[:, ie, c0].mean((-1, -2)) / fm
            sr = np.sqrt((gerr[:, ie, c0] ** 2).sum((-1, -2))) / 24.0 / fm
            for k in range(4):
                if abs(r[k] - 1.0) > 2.0 * sr[k]:
                    sigma_model[k, ie, rows] = abs(r[k] - 1.0)
                    sm_bins.append((D.MC_SPECIES[k], ie, 0.2 * ic, r[k], sr[k]))
    for ie in range(E_MC_B):
        v = slice(6, 10)
        fm = ed[:, ie, v].mean((-1, -2))
        r = g[:, ie, v].mean((-1, -2)) / fm
        sr = np.sqrt((gerr[:, ie, v] ** 2).sum((-1, -2))) / 48.0 / fm
        for k in range(4):
            if abs(r[k] - 1.0) > 2.0 * sr[k]:
                sigma_model[k, ie, :10] = np.sqrt(
                    sigma_model[k, ie, :10] ** 2 + (r[k] - 1.0) ** 2
                )
                lv_bins.append((D.MC_SPECIES[k], ie, r[k], sr[k]))
    dfac = np.sqrt(dfac**2 + (sigma_model * fac) ** 2)
    print(
        "sigma_model bins (down-going):",
        len(sm_bins),
        " sigma_level (up-going) species x E:",
        len(lv_bins),
    )
    for sp_, ie_, c_, r_, sr_ in sm_bins:
        print(
            f"  {sp_:9s} E {D.E_EDGES[ie_]:.2f}-{D.E_EDGES[ie_ + 1]:.2f}  cos Z {c_:.1f}-{c_ + 0.2:.1f}  r = {r_:.3f} +- {sr_:.3f}"
        )
    for sp_, ie_, r_, sr_ in lv_bins:
        print(
            f"  LEVEL {sp_:9s} E {D.E_EDGES[ie_]:.2f}-{D.E_EDGES[ie_ + 1]:.2f}  r_vert = {r_:.3f} +- {sr_:.3f}"
        )
    assert np.all(np.isfinite(fac)) and np.all(np.isfinite(dfac)), (
        np.isnan(fac).sum(),
        np.isnan(dfac).sum(),
    )
    # MC vs engine in the overlap bins (2.5-4, 4-6.3, 6.3-10 GeV), species-wise, cos Z bands of 0.2, azimuth mean
    mc_over_eng = g[:, 7:].reshape(4, 3, 5, 2, 12).mean((-1, -2)) / ed[:, 7:].reshape(
        4, 3, 5, 2, 12
    ).mean((-1, -2))
    e_edges = np.asarray(D.E_EDGES)
    gm = g[:, 7:].reshape(4, 3, 5, 2, 12).mean((-1, -2))
    em = ed[:, 7:].reshape(4, 3, 5, 2, 12).mean((-1, -2))
    sig_mc = np.sqrt((gerr[:, 7:].reshape(4, 3, 5, 24) ** 2).sum(-1)) / 24.0 / em
    sig_en = (
        rel[:, 7:, ::2][..., :5] if False else rel[:, 7:].reshape(4, 3, 5, 2).mean(-1)
    )
    zval = (mc_over_eng - 1.0) / np.sqrt(sig_mc**2 + sig_en**2)
    prov = (
        "Kamioka 3D factor v1b (E 5/dec x cos Z 0.1 x az 12). Down-going E < 3.98 GeV: MC G = s/(kappa B) (milestone 4 v1b, "
        "G_err_total = stat (+) reference-shape systematic); E >= 3.98 GeV: mceq3d engine F = 3D/B (rows 0.35-0.95 extended "
        "from 6 to 12 azimuths by harmonics 0-2). Up-going: engine F (production cone from the mirrored down-going direction, "
        "|cos Z| <= 0.35) x R_up (muon-bend response transferred with the projection model to the production point Q, route a "
        "with the theta-gradient reflected, see INTEGRATION.md 12); errors sqrt(sigma_engine^2 + sigma_R^2), sigma_engine = "
        "max(5%, |shape residual vs Honda solmax|), sigma_R = stat (+) 32% closure (+) |a - mirror|. Reference B = engine "
        "curved-atmosphere 1D (GSF, SIBYLL23D, v142p, no cutoff), both hemispheres."
    )
    p = os.path.join(OUT, f"kamioka_3D_{tag}.npz")
    cz_edges = np.round(np.linspace(-1, 1, 21), 6)
    np.savez(
        p,
        species=np.array(D.MC_SPECIES),
        e_edges=e_edges,
        cosz_edges=cz_edges,
        az_edges=np.arange(13) * 30.0,
        factor=fac,
        factor_error=dfac,
        source=source,
        label=f"kamioka_3D_{tag}",
        status=f"{tag} (up-going and E >= 3.98 GeV down-going from the engine; see INTEGRATION.md 12)",
        site="kamioka",
        provenance=prov,
        input_sha256_G_down=sha(G_FILE_B),
        input_sha256_Rmu=sha(R_FILE),
        input_sha256_base1d=sha(os.path.join(OUT, "base1d_kamioka.npz")),
        input_sha256_engine_F_fine=sha(os.path.join(OUT, "engine_F_fine.npz")),
        db_label="v142p",
        db_sha256=str(gd["db_sha256"][0]),
        primary_model_mc=str(gd["primary_model"][0]),
        primary_model_engine="GlobalSplineFitBeta (GSF), SIBYLL23D",
        phi0_GV=0.55,
        field="IGRF-14 (ppigrf 2.1.0, IGRF14.shc), 2020-01-01, 20 km at the production point (up-going bend)",
        build_date=datetime.now().isoformat(timespec="seconds"),
        route_R_up="a: projection model R-1 = delta_s(E, cosZ_local) q_s [(d x B_Q) . (-d/dtheta, +d/dphi) ln F_off], theta term "
        "reflected for the production point; delta fitted on the down-going R table (12.2)",
        route_down_high_E="engine for E >= 3.98 GeV (MC/engine compared, not blended)",
        route_up_cone="mirror cone correction F_cone/F_single, |cosZ| <= 0.35",
        kappa=gd["kappa"],
        kappa_err=gd["kappa_err"],
        R_up_minus_1=R_c - 1.0,
        R_up_a_minus_1=R_a - 1.0,
        R_up_mirror_minus_1=R_b - 1.0,
        sigma_R=sig_R,
        mc_over_engine_overlap=mc_over_eng,
        mc_over_engine_overlap_z=zval,
        sigma_model_rel=sigma_model,
        sigma_model="down-going MC-filled bins, all cosZ 0.2 bands: |<G>/<F>-1| if > 2 sigma_r; up-going: sigma_level(species,E)=|r_vert-1| (cosZ 0.6-1) if > 2 sigma; sigma_r = MC G_err_total only; relative, added in quadrature",
        delta=rf["delta"],
        delta_err=rf["delta_err"],
    )
    print("wrote", p, fac.shape)
    ec = np.sqrt(e_edges[:-1] * e_edges[1:])
    L = [
        "**rms over azimuth and E bands of (R_up - 1) (route a | route b plain mirror | median sigma_R), species x E band x up cos Z pair (0.2 bands)**",
        "",
        "| species | E band | "
        + " | ".join(
            f"cos Z {-1 + 0.2 * i:+.1f}..{-0.8 + 0.2 * i:+.1f}" for i in range(5)
        )
        + " |",
        "|---|---|" + "---|" * 5,
    ]
    for k, sp in enumerate(D.MC_SPECIES):
        for lo, hi in D.E_BANDS:
            ies = [ie for ie, e in enumerate(ec) if lo <= e < hi and e < 2.5]
            if not ies:
                continue
            cells = []
            for ic in range(5):
                sl = slice(2 * ic, 2 * ic + 2)
                a = np.sqrt(np.mean((R_a[k][ies][:, sl] - 1) ** 2))
                b = np.sqrt(np.mean((R_b[k][ies][:, sl] - 1) ** 2))
                sg = np.median(sig_R[k][ies][:, sl])
                cells.append(f"{100 * a:+.1f} / {100 * b:+.1f} / {100 * sg:.1f}%")
            L.append(f"| {sp} | {lo:g}-{hi:g} | " + " | ".join(cells) + " |")
    L += [
        "",
        "**z = (MC/engine - 1)/sigma in the overlap E bins (sigma = G_err_total of the 24 cells in quadrature /24 / engine mean, plus the engine relative error, both in quadrature; the MC systematic part is correlated between cells so z is somewhat optimistic)**",
        "",
        "| species | E bin | 0.0-0.2 | 0.2-0.4 | 0.4-0.6 | 0.6-0.8 | 0.8-1.0 |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, sp in enumerate(D.MC_SPECIES):
        for b in range(3):
            L.append(
                f"| {sp} | {b} | "
                + " | ".join(f"{zval[k, b, i]:+.1f}" for i in range(5))
                + " |"
            )
    L += [
        "",
        "**MC / engine in the overlap E bins (2.5-4 | 4-6.3 | 6.3-10 GeV), azimuth mean, per cos Z band (0-0.2 ... 0.8-1)**",
        "",
        "| species | E bin | "
        + " | ".join(f"{0.2 * i:.1f}-{0.2 * i + 0.2:.1f}" for i in range(5))
        + " |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, sp in enumerate(D.MC_SPECIES):
        for b in range(3):
            L.append(
                f"| {sp} | {b} | "
                + " | ".join(f"{mc_over_eng[k, b, i]:.3f}" for i in range(5))
                + " |"
            )
    open(os.path.join(OUT, f"{tag}_build_tables.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    {
        "fit": cmd_fit,
        "build": cmd_build,
        "validate": cmd_validate,
        "build_v1b": cmd_build_v1b,
        "validate_v1b": lambda: cmd_validate("v1b"),
        "build_v1c": lambda: cmd_build_v1b("v1c"),
        "validate_v1c": lambda: cmd_validate("v1c"),
    }[sys.argv[1]]()
