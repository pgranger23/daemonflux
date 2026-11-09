"""Kamioka 3D table v2 and its hybrid-level twin v2h (candidate).

    python build_table_v2.py regress   rebuild the v1c factor blocks from v1c's inputs
                                       with this code path (must agree bit for bit)
    python build_table_v2.py build     $W/runs/v2h/out/kamioka_3D_v2.npz and
                                       kamioka_3D_v2h.npz (refuses to overwrite)
    python build_table_v2.py gates     gate G1 (hybrid identities through daemonflux)
                                       -> $W/runs/v2h/out/gates_G1.md

Draft documentation: ``$W/runs/v2h/INTEGRATION_13_draft.md`` (INTEGRATION.md section 13).

v2 = the v1c pipeline (``diag_v1.cmd_build_v1b("v1c")``: same grid, R_up, sigma_R,
sigma_model) with

* (a) MC down-going cells (source 0) multiplied by C'' = (kappa/kappa_ref)/(1+b) x gf
  (``$W/runs/fix1/REPORT.md``): fix1's per-cell values (``t4_closure.npy``) below 1 GeV;
  from the definition at 1-3.98 GeV (b measured, kappa/kappa_ref = 1 assumed, gf
  computed);
* (b) every engine cell (down-going E >= 3.98 GeV, up-going mirror cone x R_up) from the
  fixed engine numerator F = Phi_3D^PHYS / B_ref with ``eoff_model="v4"`` (fix2 variant
  D); the E_off v3 cells (fix1) give the engine-version spread;
* (c) R_up, sigma_R, sigma_model and the grid exactly as v1c (v1c relative error kept).

v2h = v2 x H.  Hybrid 1D level per species and |cos Z| (azimuth-independent):
L = (1 - w) B_ref + w Phi_df, w = ``mceq3d_flux.hybrid_weight(E, e0=1.7)``, B_ref = the
engine base in the REF population (the denominator of F), Phi_df = daemonflux(kamioka),
calibrated, nominal.  H(E bin, cos Z bin) = int L / int Phi_df (energy: engine 10/decade
grid points inside the bin with weights e_widths, the convention of ``diag_v1.fine_avg``;
cos Z: the 3-point Gauss rule of ``base1d_kamioka.npz``); up-going bins use |cos Z|.

Errors (relative, in quadrature; ``factor_error`` stays absolute): v1c terms (v1c
factor_error / factor), C'' (MC cells), engine spread |F_v4 - F_v3| / F_v4 (engine
cells), near-horizon MC / v4 term (MC cells, cos Z 0-0.2), and for v2h only the
hybrid-level term [(1 - w) B_ref / L] sigma_B, sigma_B^2 = sigma_had^2 + sigma_prim^2.
"""

import importlib.util  # noqa: F401  (before any MCEq import)
import os
import subprocess
import sys
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import diag_v1 as V  # noqa: E402

D = V.D
W = D.W
FIX1 = os.path.join(W, "runs", "fix1", "out")
FIX2 = os.path.join(W, "runs", "fix2", "out")
LOWE = os.path.join(W, "runs", "lowE", "out")
V2H = os.path.join(W, "runs", "v2h", "out")
INTEG = V.OUT
SP = D.MC_SPECIES  # nue antinue numu antinumu (table order)
ESP = D.SPECIES  # total_<sp>, engine order = SP order
E_EDGES = np.asarray(D.E_EDGES)
NE = 10
E_MC = V.E_MC_B  # MC down-going bins 0..7 (< 3.98 GeV)
HYBRID_E0 = 1.7
W_FORMULA = (
    "w(E) = 0.5 * (1 + tanh(ln(E / e0) / ln 2)), e0 = 1.7 GeV "
    "(mceq3d_flux.hybrid_weight)"
)
#: relative systematic on the scoring-cap bias b (theta^2 form; see cpp_map)
B_FORM_SYS = 0.30
PHI0_GV = 0.55

#: fix1 t4_closure.npy keys per down-going table row (cos Z 0.05 ... 0.95)
CLOSURE_KEY = (
    ["0-0.2 (near-horizon)"] * 2 + ["0.2-0.3"] + ["0.3-0.6"] * 3 + ["0.6-1"] * 4
)
#: s1d_cap.npz cos Z band per down-going row and its energy-band labels per table E bin
CAP_BAND = ["0-0.2"] * 2 + ["0.2-0.6"] * 4 + ["0.6-1"] * 4
CAP_E = (
    "0.10-0.16",
    "0.16-0.25",
    "0.25-0.40",
    "0.40-0.63",
    "0.63-1.0",
    "1.0-1.6",
    "1.6-2.5",
    "2.5-4",
)


def p_in(*parts):
    return os.path.join(*parts)


INPUTS = {
    "v1c": p_in(INTEG, "kamioka_3D_v1c.npz"),
    "G_down": V.G_FILE_B,
    "Rmu": V.R_FILE,
    "base1d": p_in(INTEG, "base1d_kamioka.npz"),
    "engine_F_fine": p_in(INTEG, "engine_F_fine.npz"),
    "rfit": p_in(INTEG, "rfit.npz"),
    "engine_sigma": p_in(INTEG, "engine_sigma.npz"),
    "updiag_modes": p_in(INTEG, "updiag_modes.npz"),
    "updiag_modes_dn6": p_in(INTEG, "updiag_modes_dn6.npz"),
    "t4_closure": p_in(FIX1, "t4_closure.npy"),
    "b_gridfloor": p_in(FIX1, "b_gridfloor.npz"),
    "s1d_cap": p_in(LOWE, "s1d_cap.npz"),
    "s11_ref": p_in(LOWE, "s11_ref.npz"),
    "F_v3_hor": p_in(FIX1, "t4_C_hor.npz"),
    "F_v3_dn6": p_in(FIX1, "t4_C_dn6.npz"),
    "F_v3_up": p_in(FIX1, "t4_C_up.npz"),
    "F_v4_hor": p_in(FIX2, "t4_D_hor.npz"),
    "F_v4_dn6": p_in(FIX2, "t4_D_dn6.npz"),
    "F_v4_up": p_in(V2H, "t4_D_up.npz"),
    "t5_phi_p03": p_in(FIX1, "t5_phi+0.3.npz"),
    "t5_phi_m03": p_in(FIX1, "t5_phi-0.3.npz"),
    "daemonflux_eval": p_in(V2H, "e3_df.npz"),
    "sigma_prim": p_in(V2H, "e4_sigma_prim.npz"),
    "eoff_table_v4": p_in(HERE, "offaxis_excess_channel_v4.npz"),
}
E2_FILES = [p_in(V2H, f"e2_{t}.npz") for t in "abcdef"]


# ------------------------------------------------------------------------------ inputs
def load_e2():
    """B_ref (phi 0, +-0.3 GV) and the DDM base at the row centres (and extra cos Z)."""
    rows = {}
    for f in E2_FILES:
        d = np.load(f)
        for i, c in enumerate(d["cz"]):
            r = {}
            for key in ("B", "Bp03", "Bm03", "Bddm"):
                if f"{key}_{ESP[0]}" in d.files:
                    r[key] = np.array([d[f"{key}_{s}"][i] for s in ESP])  # (4, nE)
            rows[round(float(c), 4)] = r
    return rows


def engine_cells_down(src_hor, src_dn6, fine):
    """Engine F (``Fnew`` = Phi_3D^PHYS/B_ref) on the table's down-going grid
    (4, 10, 10, 12), v1c convention (``diag_v1.engine_down12``: base_REF x dE weights,
    rows 0.35-0.95 from 6 to 12 azimuths by harmonics 0-2)."""
    fk = {"e": fine["e"], "e_widths": fine["e_widths"]}
    for sp in ESP:
        for nm, src in (("hor", src_hor), ("dn6", src_dn6)):
            b = fine[f"base_{nm}_{sp}"]
            fk[f"flux_{nm}_{sp}"] = src[f"Fnew_{sp}"] * b[:, None, :]
            fk[f"base_{nm}_{sp}"] = b
    return V.engine_down12(fk)


def engine_cells_up(F_up, Rc, fine):
    """Up-going cells (4, 10, 10, 12) = fine_avg(F x mirror cone ratio x R_up, base_up),
    as in v1c."""
    out = np.zeros((4, NE, 10, 12))
    for k, sp in enumerate(ESP):
        s = sp[6:]
        out[k] = V.fine_avg(
            fine, F_up[s] * D.mirror_ratio(sp) * Rc[s], fine[f"base_up_{sp}"]
        )
    return out


def r_up_v1c(fine):
    """R_up of v1c (route a' azimuth-odd part on the measured mirror,
    ``diag_v1.r_up_hybrid``), computed exactly as in ``cmd_build_v1b("v1c")`` (gradients
    of the delivered engine F)."""
    D.MIRROR = True
    rf = dict(np.load(INPUTS["rfit"]))
    rd = dict(np.load(INPUTS["Rmu"], allow_pickle=True))
    F_up = {
        sp[6:]: (fine[f"flux_up_{sp}"] / fine[f"base_up_{sp}"][:, None, :])
        * D.mirror_ratio(sp)
        for sp in ESP
    }
    Ra, _, _ = V.r_up_fine(fine, F_up, rf)
    Rb = V.r_mirror_harm(fine, rd)
    return V.r_up_hybrid(fine, Ra, Rb)


def old_up_F(fine):
    """Delivered (v1c) engine up-going F without the cone ratio (applied in
    engine_cells_up)."""
    return {
        sp[6:]: fine[f"flux_up_{sp}"] / fine[f"base_up_{sp}"][:, None, :] for sp in ESP
    }


# ------------------------------------------------------------------------------ C''
def cpp_map(gd):
    """C'' (4, 10 E, 10 down rows) and its relative error, MC bins 0..7.

    E bins 0-4 (0.1-1 GeV): fix1's values, C'' = kappa/kappa_ref/(1 + b) x gf per
    closure key (cos Z 0.2-1: measured cells; 0-0.2: same kappa window, b of the 0-0.2
    cap band).  E bins 5-7 (1-3.98 GeV): kappa/kappa_ref = 1 (assumed: the T < 2 GeV/n
    content of t, dB/B, is 0.05-0.24% at 0.63-1 GeV and falls with E), b from s1d_cap,
    gf from b_gridfloor over the kappa window (rows 0.65-0.95).
    Error: kappa_ref estimator |kappa_ref/kappa_h1 - 1| (bins 0-4) or dB/B of 0.63-1 GeV
    (bins 5-7); b: fit error (+) 30% of b (theta^2 form; the engine-based cap estimate
    of lowE s1e is ~1.5x the fitted b at 0.1-0.45 GeV, so the form is uncertain at that
    level).
    """
    clo = np.load(INPUTS["t4_closure"], allow_pickle=True).item()
    cap = np.load(INPUTS["s1d_cap"])
    s11 = np.load(INPUTS["s11_ref"])
    gfl = np.load(INPUTS["b_gridfloor"])
    fine = np.load(INPUTS["engine_F_fine"])
    e, de = fine["e"], fine["e_widths"]
    C = np.ones((4, NE, 10))
    err = np.zeros_like(C)
    parts = {k: np.zeros_like(C) for k in ("kk", "b", "gf", "err_kref", "err_b")}
    for ie in range(E_MC):
        for k in range(4):
            if ie < 5:
                kref, kh1 = s11[f"{ie}|{k}"][9], s11[f"{ie}|{k}"][10]
                e_kref = abs(kref / kh1 - 1.0)
            else:
                e_kref = float(s11[f"4|{k}"][0])
            m = (e >= E_EDGES[ie]) & (e < E_EDGES[ie + 1])
            rows = [3, 4, 5, 6]  # dn6 rows 0.65-0.95 = the kappa window
            gf_def = (gfl[f"B30_dn6_{ESP[k]}"][rows][:, m] * de[m]).sum() / (
                fine[f"base_dn6_{ESP[k]}"][rows][:, m] * de[m]
            ).sum()
            for ic in range(10):
                cq = cap[f"{CAP_BAND[ic]}|{CAP_E[ie]}"]
                b, b_fit = float(cq[6]), float(cq[7])
                if ie < 5:
                    r = clo[f"{CLOSURE_KEY[ic]}|{ie}|{k}"]
                    assert abs(r["b"] - b) < 1e-12, (ie, k, ic)
                    kk, gf = r["kappa"] / r["kref"], r["gf"]
                    assert abs(gf / gf_def - 1.0) < 1e-12
                else:
                    kk, gf = 1.0, gf_def
                C[k, ie, ic] = kk / (1.0 + b) * gf
                e_b = np.hypot(b_fit, B_FORM_SYS * b) / (1.0 + b)
                err[k, ie, ic] = np.hypot(e_kref, e_b)
                parts["kk"][k, ie, ic], parts["b"][k, ie, ic] = kk, b
                parts["gf"][k, ie, ic] = gf
                parts["err_kref"][k, ie, ic], parts["err_b"][k, ie, ic] = e_kref, e_b
    return C, err, parts


# ------------------------------------------------------------------------------ hybrid
def hybrid(fine):
    """H (4, 10, 20), the hybrid-level error (4, 10, 20) and the bin diagnostics."""
    from mceq3d_flux import hybrid_weight

    b1 = np.load(INPUTS["base1d"])
    df = np.load(INPUTS["daemonflux_eval"])
    sp = np.load(INPUTS["sigma_prim"])
    e2 = load_e2()
    ndf = len(df["e"])  # daemonflux evaluated to 1 TeV (prefix of the engine grid)
    assert np.allclose(b1["engine_e"], fine["e"]) and np.allclose(
        df["e"], fine["e"][:ndf]
    )
    e, de = fine["e"][:ndf], fine["e_widths"][:ndf]
    w = hybrid_weight(e, e0=HYBRID_E0)
    nodes = b1["nodes_cz"]
    assert np.allclose(df["cz_all"][:30], nodes)
    gw = np.array([5.0, 8.0, 5.0]) / 18.0  # base1d's Gauss-Legendre weights
    H = np.ones((4, NE, 20))
    err = np.zeros_like(H)
    diag = {
        k: np.zeros((4, NE, 10))
        for k in ("sigma_had", "sigma_prim", "D_bin", "wfac_lvl", "wB_over_L")
    }
    for k, s in enumerate(SP):
        Bn = b1["engine_flux_per_cm2"][k][:, :ndf]  # (30, nE) per cm^2, REF
        Pn = df[f"phi_df_{s}"][:30]  # (30, nE) per cm^2
        ww = w
        Ln = (1.0 - ww) * Bn + ww * Pn
        for ie in range(NE):
            m = (e >= E_EDGES[ie]) & (e < E_EDGES[ie + 1])
            for j in range(10):
                n3 = slice(3 * j, 3 * j + 3)
                num = (gw[:, None] * Ln[n3][:, m] * de[m]).sum()
                den = (gw[:, None] * Pn[n3][:, m] * de[m]).sum()
                H[k, ie, 10 + j] = num / den
                H[k, ie, 9 - j] = num / den  # up-going: 1D flux at |cos Z|
                c = round(0.05 + 0.1 * j, 4)
                Bc = e2[c]["B"][k][:ndf] / 1e4  # per m^2 -> per cm^2
                assert np.allclose(
                    Bc, Bn[3 * j + 1], rtol=1e-12, atol=0
                ), "e2 B != base1d"
                s_had = np.abs(e2[c]["Bddm"][k][:ndf] / e2[c]["B"][k][:ndf] - 1.0)
                s_pri = sp[f"prim_gsf_cal_{s}"][j][:ndf]
                sB = np.hypot(s_had, s_pri)
                Pc, Lc = Pn[3 * j + 1], Ln[3 * j + 1]
                eh = ((1.0 - ww[m]) * Bc[m] * sB[m] * de[m]).sum() / (
                    Lc[m] * de[m]
                ).sum()
                err[k, ie, 10 + j] = eh
                err[k, ie, 9 - j] = eh
                diag["sigma_had"][k, ie, j] = (Bc[m] * s_had[m] * de[m]).sum() / (
                    Bc[m] * de[m]
                ).sum()
                diag["sigma_prim"][k, ie, j] = (Bc[m] * s_pri[m] * de[m]).sum() / (
                    Bc[m] * de[m]
                ).sum()
                diag["D_bin"][k, ie, j] = (Pc[m] * de[m]).sum() / (Bc[m] * de[m]).sum()
                diag["wfac_lvl"][k, ie, j] = (ww[m] * Pc[m] * de[m]).sum() / (
                    Lc[m] * de[m]
                ).sum()
                diag["wB_over_L"][k, ie, j] = ((1 - ww[m]) * Bc[m] * de[m]).sum() / (
                    Lc[m] * de[m]
                ).sum()
    return H, err, diag


def solar_response(fine, wfac_lvl):
    """dlnproduct_dphi (4, 10, 20) [1/GV], azimuth-averaged.

    d ln(product)/d phi = d ln Phi_3D/d phi - (w Phi_df / L) d ln B_ref/d phi, i.e. the
    brief's "dlnF/dphi + dlnB_ref/dphi below the switch, dlnF/dphi above" with
    F = Phi_3D/B_ref at the same phi.  fix1's t5 files hold
    F_t5 = Phi_3D^PHYS(phi)/B_ref(0) (B_ref unmodulated), so their slope IS
    d ln Phi_3D/d phi.  Central difference at +-0.3 GV; t5 rows (down 0.15, 0.35, 0.95;
    up -0.15, -0.35, -0.95; 6 azimuths, E_off v3, reduced config) interpolated linearly
    in |cos Z| (flat outside); d ln B_ref/d phi from the REF base at the row centres
    (e2).  Phi_df is held fixed (calibrated at its own epoch).
    """
    tp, tm = np.load(INPUTS["t5_phi_p03"]), np.load(INPUTS["t5_phi_m03"])
    et = tp["e"]
    det = np.gradient(np.log(et)) * et
    e, de = fine["e"], fine["e_widths"]
    e2 = load_e2()
    out = np.zeros((4, NE, 20))
    dln3d = {"dn": np.zeros((4, NE, 3)), "up": np.zeros((4, NE, 3))}
    czr = np.array([0.15, 0.35, 0.95])
    for k, sp in enumerate(ESP):
        for ie in range(NE):
            m = (et >= E_EDGES[ie]) & (et < E_EDGES[ie + 1])
            for hemi, srcs in (
                ("dn", (("dn", 0), ("dn", 1), ("v", 0))),
                ("up", (("up", 2), ("up", 1), ("up", 0))),
            ):
                for r, (nm, i) in enumerate(srcs):
                    a = (tp[f"flux_{nm}_{sp}"][i].mean(0)[m] * det[m]).sum()
                    b = (tm[f"flux_{nm}_{sp}"][i].mean(0)[m] * det[m]).sum()
                    dln3d[hemi][k, ie, r] = np.log(a / b) / 0.6
            mm = (e >= E_EDGES[ie]) & (e < E_EDGES[ie + 1])
            for j in range(10):
                c = round(0.05 + 0.1 * j, 4)
                bp = (e2[c]["Bp03"][k][mm] * de[mm]).sum()
                bm = (e2[c]["Bm03"][k][mm] * de[mm]).sum()
                dlnB = np.log(bp / bm) / 0.6
                d_dn = np.interp(c, czr, dln3d["dn"][k, ie])
                d_up = np.interp(c, czr, dln3d["up"][k, ie])
                out[k, ie, 10 + j] = d_dn - wfac_lvl[k, ie, j] * dlnB
                out[k, ie, 9 - j] = d_up - wfac_lvl[k, ie, j] * dlnB
    return out, dln3d


# ------------------------------------------------------------------------------- build
def sha(path):
    return V.sha(path)


def git_rev():
    try:
        return subprocess.run(
            ["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        return "unknown"


def build_arrays():
    """All arrays of v2 / v2h (dict); nothing is written."""
    fine = np.load(INPUTS["engine_F_fine"])
    v1 = dict(np.load(INPUTS["v1c"]))
    gd = dict(np.load(INPUTS["G_down"], allow_pickle=True))
    fac1, dfac1 = v1["factor"], v1["factor_error"]
    rel_v1c = dfac1 / fac1
    Rc = r_up_v1c(fine)
    # engine cells: v4 (central) and v3 (alternative)
    v4 = {g: np.load(INPUTS[f"F_v4_{g}"]) for g in ("hor", "dn6", "up")}
    v3 = {g: np.load(INPUTS[f"F_v3_{g}"]) for g in ("hor", "dn6", "up")}
    ed4 = engine_cells_down(v4["hor"], v4["dn6"], fine)
    ed3 = engine_cells_down(v3["hor"], v3["dn6"], fine)
    eu4 = engine_cells_up({s[6:]: v4["up"][f"Fnew_{s}"] for s in ESP}, Rc, fine)
    eu3 = engine_cells_up({s[6:]: v3["up"][f"Fnew_{s}"] for s in ESP}, Rc, fine)
    # MC cells with C''
    C, C_err, C_parts = cpp_map(gd)
    G = gd["G"]
    fac = np.zeros_like(fac1)
    fac[:, :, :10] = eu4
    fac[:, :E_MC, 10:] = G[:, :E_MC] * C[:, :E_MC, :, None]
    fac[:, E_MC:, 10:] = ed4[:, E_MC:]
    source = np.array(v1["source"])
    assert np.all(source[:E_MC, 10:] == 0) and np.all(source[E_MC:, 10:] == 1)
    assert np.all(source[:, :10] == 3)
    # error terms (relative)
    err_cpp = np.zeros_like(fac)
    err_cpp[:, :E_MC, 10:] = C_err[:, :E_MC, :, None]
    spread = np.zeros_like(fac)
    spread[:, :, :10] = np.abs(eu4 - eu3) / eu4
    spread[:, E_MC:, 10:] = np.abs(ed4[:, E_MC:] - ed3[:, E_MC:]) / ed4[:, E_MC:]
    # near-horizon MC/v4 term (rows 0.05, 0.15): q = <G''>/(<F_v4> <R>); the part of
    # |q - 1| not covered by the v1c sigma_model of the band
    rd = np.load(INPUTS["Rmu"])
    Rm = rd["R"].mean(-1)  # (4, 7, 5)
    err_hor = np.zeros_like(fac)
    q_hor = np.ones((4, E_MC, 2))
    for ie in range(E_MC):
        for ic in range(2):
            for k in range(4):
                rr = Rm[k, ie, 0] if ie < Rm.shape[1] else 1.0
                q = fac[k, ie, 10 + ic].mean() / (ed4[k, ie, ic].mean() * rr)
                sm = v1["sigma_model_rel"][k, ie, 10 + ic].mean()
                q_hor[k, ie, ic] = q
                err_hor[k, ie, 10 + ic] = np.sqrt(max((q - 1.0) ** 2 - sm**2, 0.0))
    rel_v2 = np.sqrt(rel_v1c**2 + err_cpp**2 + spread**2 + err_hor**2)
    # hybrid
    H, err_hyb, hdiag = hybrid(fine)
    fac_h = fac * H[..., None]
    err_hyb4 = np.broadcast_to(err_hyb[..., None], fac.shape).copy()
    rel_v2h = np.sqrt(rel_v2**2 + err_hyb4**2)
    dphi, dln3d = solar_response(fine, hdiag["wfac_lvl"])
    assert np.all(np.isfinite(fac_h)) and np.all(np.isfinite(rel_v2h))
    assert np.all(fac > 0) and np.all(rel_v2 > 0)
    return dict(
        fine=fine,
        v1=v1,
        gd=gd,
        fac=fac,
        fac_h=fac_h,
        rel_v1c=rel_v1c,
        rel_v2=rel_v2,
        rel_v2h=rel_v2h,
        err_cpp=err_cpp,
        spread=spread,
        err_hor=err_hor,
        q_hor=q_hor,
        err_hyb=err_hyb4,
        H=H,
        hdiag=hdiag,
        C=C,
        C_err=C_err,
        C_parts=C_parts,
        ed4=ed4,
        ed3=ed3,
        eu4=eu4,
        eu3=eu3,
        Rc=Rc,
        dphi=dphi,
        dln3d=dln3d,
        source=source,
    )


PROV_V2 = (
    "Kamioka 3D factor v2 (E 5/dec 0.1-10 GeV x cos Z 0.1 x az 12; grid, R_up, "
    "sigma_R, sigma_model as v1c). Down-going E < 3.98 GeV: MC G = s/(kappa B) "
    "(milestone 4 v1b) x C'' = (kappa/kappa_ref)/(1+b) x gf (fix1 per-cell values "
    "below 1 GeV, definition above; source 0); E >= 3.98 GeV: engine F = "
    "Phi_3D^PHYS/B_ref, eoff_model v4 (fix2 variant D; source 1). Up-going: engine "
    "v4 F x mirror cone ratio (|cos Z| <= 0.35) x R_up of v1c (source 3). B_ref = "
    "engine curved-atmosphere 1D, REF population (MCEq floor 3.0 GeV, e_min 0.1), "
    "GSF, SIBYLL23D, v142p, no cutoff. factor_error (absolute) = factor x "
    "sqrt(err_v1c_rel^2 + err_Cpp_rel^2 + err_engine_spread_rel^2 + "
    "err_horizon_rel^2). Tool: tools/mceq3d/build_table_v2.py; doc: INTEGRATION.md "
    "section 13 (draft $W/runs/v2h/INTEGRATION_13_draft.md)."
)
PROV_V2H = (
    "Kamioka 3D factor v2h = v2 x hybrid_correction H. Hybrid 1D level L = (1 - w) "
    "B_ref + w Phi_df per species and |cos Z|, "
    + W_FORMULA
    + "; Phi_df = daemonflux Flux(location='kamioka') calibrated nominal; H = int L / "
    "int Phi_df per bin (engine 10/dec grid points x e_widths; 3-point Gauss in "
    "cos Z), up-going at |cos Z|. daemonflux x factor_v2h = factor_v2 x L: below "
    "~0.44 GeV (w < 0.02) the product is Phi_3D^PHYS of the engine/MC, above ~6.6 GeV "
    "(w > 0.98) daemonflux x factor_v2. factor_3d holds factor_v2 (undo: factor / "
    "hybrid_correction). factor_error adds err_hybrid_rel = [(1-w) B_ref/L] sigma_B, "
    "sigma_B^2 = sigma_had^2 + sigma_prim^2 (sigma_had = |DDM/B_ref - 1|, MCEq 1.4 "
    "DDM, same atmosphere and population; sigma_prim = GSF block of daemonflux's "
    "calibrated covariance). No solar-modulation error (see dlnproduct_dphi). "
    + PROV_V2
)


CPP_NOTE = (
    "C'' (4, E, 10 down-going rows cos Z 0.05..0.95), applied to source-0 cells; E bins "
    "0-4 from fix1 t4_closure.npy, 5-7 from the definition with kappa/kappa_ref = 1 "
    "(assumed; error dB/B of 0.63-1 GeV), bins 8-9 unused (=1)"
)
HOR_NOTE = (
    "cos Z 0-0.2 MC cells: sqrt(max((q-1)^2 - sigma_model_v1c^2, 0)), q = "
    "<G''>_az/(<F_v4>_az <R>_az) per species x E bin x row (R = measured muon-bend "
    "response; R = 1 for 2.5-3.98 GeV)"
)
DPHI_NOTE = (
    "d ln(daemonflux x factor)/d phi [1/GV], azimuth mean, force field relative to the "
    "GSF epoch (phi0); = dlnPhi3D/dphi - (w Phi_df/L) dlnB_ref/dphi, i.e. dlnF/dphi + "
    "dlnB_ref/dphi below the switch and dlnF/dphi above (F = Phi_3D/B_ref at the same "
    "phi); dlnPhi3D/dphi from fix1 t5 (E_off v3, reduced config, rows +-0.15/0.35/0.95 "
    "interpolated in |cos Z|), dlnB_ref/dphi from REF MCEq at +-0.3 GV; daemonflux fixed"
)
PRIM_NOTE = (
    "the brief's recipe (prior total minus hadronic-only, use_calibration=False) gives "
    "~1e-4 because the prior covariance's GSF block is zero; used instead: GSF block of "
    "the calibrated covariance (0.4-1%)"
)


def cmd_build():
    a = build_arrays()
    v1 = a["v1"]
    gd = a["gd"]
    os.makedirs(V2H, exist_ok=True)
    shas = {f"input_sha256_{k}": sha(p) for k, p in INPUTS.items()}
    for f in E2_FILES:
        shas[f"input_sha256_{os.path.basename(f)[:-4]}"] = sha(f)
    common = dict(
        species=np.array(SP),
        e_edges=E_EDGES,
        cosz_edges=v1["cosz_edges"],
        az_edges=v1["az_edges"],
        source=a["source"],
        source_codes="0 = MC down-going x C'' (E < 3.98 GeV); 1 = engine v4 down-going "
        "(E >= 3.98 GeV); 3 = engine v4 up-going x mirror cone x R_up",
        site="kamioka",
        db_label="v142p",
        db_sha256=str(gd["db_sha256"][0]),
        primary_model_mc=str(gd["primary_model"][0]),
        primary_model_engine="GlobalSplineFitBeta (GSF), SIBYLL23D",
        engine_eoff_model="v4 (fix2 variant D, V4_CONFIG all fixes; table "
        "offaxis_excess_channel_v4.npz); alternative v3 (fix1) -> err_engine_spread_rel",
        engine_numerator="Phi_3D^PHYS (primary_floor 1.9, e_min 0.03, "
        "production='engine', "
        "muon_bending=False, date 2020-01-01) / B_ref (REF: floor 3.0, e_min 0.1)",
        phi0_GV=PHI0_GV,
        phi0_note="assumed force-field potential of GSF as shipped (AMS-02 2011-13), not "
        "verified; solar modulation is not in factor_error",
        field=str(v1["field"]),
        route_R_up=str(v1["route_R_up"]),
        route_up_cone=str(v1["route_up_cone"]),
        build_date=datetime.now().isoformat(timespec="seconds"),
        code_git_rev=git_rev(),
        kappa=v1["kappa"],
        kappa_err=v1["kappa_err"],
        R_up_minus_1=v1["R_up_minus_1"],
        sigma_R=v1["sigma_R"],
        sigma_model_rel=v1["sigma_model_rel"],
        sigma_model=str(v1["sigma_model"]),
        delta=v1["delta"],
        delta_err=v1["delta_err"],
        factor_v1c=v1["factor"],
        Cpp=a["C"],
        Cpp_parts_kappa_over_kref=a["C_parts"]["kk"],
        Cpp_parts_b=a["C_parts"]["b"],
        Cpp_parts_gf=a["C_parts"]["gf"],
        Cpp_note=CPP_NOTE,
        err_v1c_rel=a["rel_v1c"],
        err_Cpp_rel=a["err_cpp"],
        err_engine_spread_rel=a["spread"],
        err_horizon_rel=a["err_hor"],
        q_horizon=a["q_hor"],
        err_horizon_note=HOR_NOTE,
        dlnproduct_dphi=a["dphi"],
        dlnproduct_dphi_note=DPHI_NOTE,
        **shas,
    )
    for tag, fac, rel, prov, status in (
        (
            "v2",
            a["fac"],
            a["rel_v2"],
            PROV_V2,
            "candidate (3D factor without hybrid level)",
        ),
        ("v2h", a["fac_h"], a["rel_v2h"], PROV_V2H, "candidate"),
    ):
        p = os.path.join(V2H, f"kamioka_3D_{tag}.npz")
        if os.path.exists(p):
            raise FileExistsError(p)
        extra = dict(common)
        if tag == "v2h":
            extra.update(
                hybrid_correction=a["H"],
                factor_3d=a["fac"],
                err_hybrid_rel=a["err_hyb"],
                sigma_had_rel=a["hdiag"]["sigma_had"],
                sigma_prim_rel=a["hdiag"]["sigma_prim"],
                D_df_over_Bref_bin=a["hdiag"]["D_bin"],
                hybrid_wB_over_L=a["hdiag"]["wB_over_L"],
                hybrid_e0=HYBRID_E0,
                hybrid_w_formula=W_FORMULA,
                hybrid_level="L = (1 - w) B_ref + w Phi_df; B_ref = denominator of "
                "factor_3d",
                sigma_prim_note=PRIM_NOTE,
            )
        np.savez(
            p,
            factor=fac,
            factor_error=fac * rel,
            label=f"kamioka_3D_{tag}",
            status=status,
            provenance=prov,
            **extra,
        )
        print("wrote", p, sha(p))


# ----------------------------------------------------------------------------- regress
def cmd_regress():
    """The v1c factor blocks rebuilt with this module's helpers from v1c's own inputs."""
    fine = np.load(INPUTS["engine_F_fine"])
    v1 = np.load(INPUTS["v1c"])
    gd = np.load(INPUTS["G_down"], allow_pickle=True)
    Rc = r_up_v1c(fine)
    up = engine_cells_up(old_up_F(fine), Rc, fine)
    dn = V.engine_down12(fine)
    blocks = (
        ("up-going (source 3)", up, v1["factor"][:, :, :10]),
        ("engine down E>=3.98 (source 1)", dn[:, E_MC:], v1["factor"][:, E_MC:, 10:]),
        ("MC down (source 0)", gd["G"][:, :E_MC], v1["factor"][:, :E_MC, 10:]),
    )
    ok = True
    for name, a, b in blocks:
        d = np.abs(a / b - 1.0).max()
        print(f"{name:32s} max |rebuilt/v1c - 1| = {d:.3e}")
        ok &= d < 1e-12
    print("REGRESSION", "PASS" if ok else "FAIL")
    return ok


# ------------------------------------------------------------------------------- gates
def _flux(tab):
    import warnings

    from daemonflux import Flux

    warnings.simplefilter("ignore")
    fl = Flux(location="kamioka")
    fl.set_geomagnetic_model(tab)
    return fl


def cmd_gates():
    """G1 through the real API, G3 v2/v1c per cell."""
    import warnings

    from daemonflux import Flux
    from mceq3d_flux import hybrid_weight

    warnings.simplefilter("ignore")
    p2, p2h = (os.path.join(V2H, f"kamioka_3D_{t}.npz") for t in ("v2", "v2h"))
    f2, f2h = _flux(p2), _flux(p2h)
    f1d = Flux(location="kamioka")
    fine = np.load(INPUTS["engine_F_fine"])
    e = fine["e"]
    e2 = load_e2()
    L = [
        "# G1: hybrid identities through daemonflux.Flux(location='kamioka') + table",
        "",
    ]
    az = np.arange(0.0, 360.0, 15.0)  # bin centres and edges
    lo = (e >= 0.1122 * 0.999) & (hybrid_weight(e, HYBRID_E0) < 0.02)
    hi = (hybrid_weight(e, HYBRID_E0) > 0.98) & (e <= 60.0)
    worst = {"lo": [], "hi": []}
    for c in sorted(e2):
        Bref = e2[c]["B"]  # (4, nE) per m^2
        for sgn in (1, -1):
            zen = float(np.degrees(np.arccos(sgn * c)))
            zen1d = zen if zen <= 90 else 180.0 - zen
            for k, s in enumerate(SP):
                P1 = f1d.flux(e[lo | hi], zen1d, s)
                Pv2 = np.array(
                    [f2.flux(e[lo | hi], zen, s, azimuth_deg=float(a)) for a in az]
                )
                Pv2h = np.array(
                    [f2h.flux(e[lo | hi], zen, s, azimuth_deg=float(a)) for a in az]
                )
                F2 = Pv2 / P1
                el = e[lo | hi]
                ml, mh = lo[lo | hi], hi[lo | hi]
                b = Bref[k][lo | hi] / 1e4 * el**3  # E^3 per cm^2, daemonflux units
                d_lo = Pv2h[:, ml] / (F2[:, ml] * b[ml]) - 1.0
                d_hi = Pv2h[:, mh] / (P1[mh] * F2[:, mh]) - 1.0
                worst["lo"].append(
                    (
                        np.abs(d_lo).max(),
                        s,
                        sgn * c,
                        el[ml][np.argmax(np.abs(d_lo).max(0))],
                    )
                )
                worst["hi"].append(
                    (
                        np.abs(d_hi).max(),
                        s,
                        sgn * c,
                        el[mh][np.argmax(np.abs(d_hi).max(0))],
                    )
                )
    for key, txt in (
        ("lo", "w < 0.02: product_v2h/(factor_v2 B_ref) - 1"),
        ("hi", "w > 0.98: product_v2h/(Phi_df factor_v2) - 1"),
    ):
        arr = sorted(worst[key], reverse=True)
        L.append(
            f"## {txt}: max |dev| = {arr[0][0]:.4f} "
            f"({'PASS' if arr[0][0] < 0.01 else 'FAIL'} at 1%)"
        )
        L.append("")
        L.append("| max abs dev | species | cos Z | at E [GeV] |")
        L.append("|---|---|---|---|")
        for row in arr[:12]:
            L.append(f"| {row[0]:.4f} | {row[1]} | {row[2]:+.2f} | {row[3]:.3f} |")
        L.append("")
    # where the low-side deviation sits: flat half-bin below the first centre, between
    # centres, and at the bin centres (B_ref log-log interpolated there, half a step)
    ec = np.sqrt(E_EDGES[:-1] * E_EDGES[1:])
    regions = {
        "E < first bin centre (flat)": [],
        "between centres": [],
        "at bin centres": [],
    }
    for c in [round(0.05 + 0.1 * j, 4) for j in range(10)]:
        for sgn in (1, -1):
            zen = float(np.degrees(np.arccos(sgn * c)))
            zen1d = zen if zen <= 90 else 180.0 - zen
            for k, s in enumerate(SP):
                for name, el in (
                    ("grid", e[lo]),
                    ("at bin centres", ec[hybrid_weight(ec, HYBRID_E0) < 0.02]),
                ):
                    b = np.exp(np.interp(np.log(el), np.log(e), np.log(e2[c]["B"][k])))
                    b = b / 1e4 * el**3
                    P1 = f1d.flux(el, zen1d, s)
                    d = np.array(
                        [
                            f2h.flux(el, zen, s, azimuth_deg=float(a))
                            / (f2.flux(el, zen, s, azimuth_deg=float(a)) / P1 * b)
                            - 1.0
                            for a in az
                        ]
                    )
                    if name == "grid":
                        flat = el < ec[0]
                        regions["E < first bin centre (flat)"].append(
                            np.abs(d[:, flat]).max()
                        )
                        regions["between centres"].append(np.abs(d[:, ~flat]).max())
                    else:
                        regions[name].append(np.abs(d).max())
    L.append("## w < 0.02, by region (max |dev| over species, rows, 24 azimuths)")
    L.append("")
    for k, v in regions.items():
        L.append(f"- {k}: {max(v):.4f}")
    L.append("")
    open(os.path.join(V2H, "gates_G1.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    {"regress": cmd_regress, "build": cmd_build, "gates": cmd_gates}[sys.argv[1]]()
