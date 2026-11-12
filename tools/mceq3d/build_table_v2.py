"""Kamioka 3D table v2, its hybrid-level twin v2h and the near-horizon v2i (candidate).

    python build_table_v2.py regress   rebuild the v1c factor blocks from v1c's inputs
                                       with this code path (must agree bit for bit)
    python build_table_v2.py build     $W/runs/v2h/out/kamioka_3D_v2.npz and
                                       kamioka_3D_v2h.npz (refuses to overwrite)
    python build_table_v2.py gates     gate G1 (hybrid identities through daemonflux)
                                       -> $W/runs/v2h/out/gates_G1.md

    python build_table_v2.py build-v2i [CORR.npz]
                                       $W/runs/v2i/out/kamioka_3D_v2i.npz = v2h x the
                                       near-horizon corrections (refuses to overwrite;
                                       CORR defaults to $V2I_CORR or
                                       $W/runs/v2i/out/corrections_v2i.npz)
    python build_table_v2.py gates-v2i identities (V1) and errors (V2) of v2i vs v2h
                                       -> $W/runs/v2i/out/gates_V1V2.md

    python build_table_v2.py build-v2j [CORR.npz]
                                       $W/runs/v2j/out/kamioka_3D_v2j.npz = v2i with the
                                       up-going correction carried as an error only
                                       (c_up = 1, species-resolved err_upcorr; refuses
                                       to overwrite; CORR defaults to $V2J_CORR or
                                       $W/runs/v2j/out/corrections_v2j.npz)
    python build_table_v2.py gates-v2j identities and errors of v2j vs v2h and v2i
                                       -> $W/runs/v2j/out/gates.md (section "builder")

``$V2_OUTDIR`` redirects the output directory of build-v2i / build-v2j (reproducibility
checks); the default is the run directory of the table.

Draft documentation: ``$W/runs/v2h/INTEGRATION_13_draft.md`` (INTEGRATION.md section
13), ``$W/runs/v2i/INTEGRATION_14_draft.md`` (section 14, v2i).

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

v2i = v2h with three near-horizon changes from the hzn study (``$W/runs/hzn/REPORT.md``;
numbers produced by ``$W/runs/v2i/scripts/c0_corrections.py`` -> corrections npz, whose
sha256 is recorded): (a) MC horizon cells (source 0, cos Z 0.05 / 0.15, E bins 0-6)
x f_b = (1 + b1) / (1 + b0), the cap bias of C'' re-measured without the 1/cos weight,
new term err_capb_rel; (b) err_horizon_rel re-assigned on those cells from hzn's
decomposition; (c) up-going source-3 cells at cos Z -0.05 / -0.15, E bins 0-2, x c_up
(template / MC field-free measurement), new term err_upcorr_rel (E bins 3-4: no
correction, error only).  factor_3d_v2i = factor_3d_v2h x corr_v2i, factor = factor_3d x
hybrid_correction as before.

v2j = v2i with (c) not applied as a central value: c_up = 1 (the up-going cells keep
their v2h values) and err_upcorr_rel becomes species-resolved, carrying |1 - c_up_v2i|
in quadrature with the jackknife error of the species-mean MC ratio and the
template-MC difference (``$W/runs/v2j/scripts/c0_corrections_v2j.py``); (a) and
(b) are v2i's.
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
V2I = os.path.join(W, "runs", "v2i", "out")
V2J = os.path.join(W, "runs", "v2j", "out")
V2H_SHA256 = "e3db406733bbe64fd9534c3366214b423f622f6a702b95518e4731a342a711ce"
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


# ----------------------------------------------------------------------------- v2i
#: hzn arrays hold E bins 0-6 and the |cos Z| rows 0.05, 0.15
V2I_NE = 7
ROW_DN = (10, 11)  # cos Z +0.05, +0.15 (source 0 below 3.98 GeV)
ROW_UP = (9, 8)  # cos Z -0.05, -0.15 (source 3)

PROV_V2I = (
    "Kamioka 3D factor v2i = v2h x corr_v2i (near-horizon changes from the hzn study, "
    "$W/runs/hzn/REPORT.md): (a) MC horizon cells (source 0, cos Z 0.05/0.15, E bins "
    "0-6) x f_b = (1 + b1)/(1 + b0), b1 = measure-1 and b0 = measure-0 cap bias of the "
    "0-0.2 band (production B), because C'' divided these rows by the 1/cos-heavy-tail "
    "b1; (b) err_horizon_rel re-assigned on those cells (E bins 0-2 |species-mean "
    "D_q| of the engine-Z decomposition, bins 3-6 |species-mean (D_Bd + D_Cut)| of the "
    "v4 decomposition, antinue 0.25-0.40 GeV row 0.05 keeps the v2h value); (c) "
    "up-going source-3 cells at cos Z -0.05/-0.15: E bins 0-2 x c_up (row -0.05 "
    "species-resolved v4/Z template, row -0.15 species-mean MC field-free), E bins 3-4 "
    "no correction but an error |1 - mean template|. factor_3d = factor_3d_v2h x "
    "corr_v2i, factor = factor_3d x hybrid_correction (unchanged). factor_error = "
    "factor x sqrt(err_v1c_rel^2 + err_Cpp_rel^2 + err_engine_spread_rel^2 + "
    "err_horizon_rel^2 + err_hybrid_rel^2 + err_capb_rel^2 + err_upcorr_rel^2). "
    "Tool: tools/mceq3d/build_table_v2.py build-v2i; doc: INTEGRATION.md section 14 "
    "(draft $W/runs/v2i/INTEGRATION_14_draft.md). v2h provenance follows. "
)
CORR_NOTE = (
    "corr_v2i (4, 10, 20) = corr_fb x corr_cup, azimuth-independent; exactly 1.0 "
    "outside the MC horizon rows (E bins 0-6) and the up-going rows -0.05/-0.15 "
    "(E bins 0-2)"
)
ERR_CAPB_NOTE = (
    "MC horizon cells (source 0, cos Z 0.05/0.15, E bins 0-6): sqrt((|f_b - 1|/2)^2 + "
    "(3 sigma_b0)^2); sigma_b0 = fit error of b0, x3 because hzn found jackknife "
    "errors of b to be 2-4x the per-primary ones; 0 elsewhere"
)
ERR_UPCORR_NOTE = (
    "up-going cos Z -0.05/-0.15: E bins 0-2 sqrt(se_mean^2 + (mean_k template - "
    "mean_k mc)^2), se_mean = sqrt(sum_k mc_err^2)/4; E bins 3-4 |1 - mean_k "
    "template| (no correction applied, evidence not established); 0 elsewhere"
)
HOR_NOTE_V2I = (
    "MC cells cos Z 0.05/0.15, E bins 0-6 (replaces the v2h term, kept as "
    "err_horizon_rel_v2h): E bins 0-2 |species-mean D_q| (hzn a2_decomp_Z), bins 3-6 "
    "|species-mean (D_Bd + D_Cut)| (hzn a2_decomp, v4), D_f = ln f - mean over rows "
    "0.35-0.95 of ln f; azimuth- and species-independent except antinue 0.25-0.40 GeV "
    "row 0.05 (v2h value kept); E bin 7 as v2h"
)
DPHI_NOTE_V2I = (
    "UNCHANGED from v2h (v2i's corrections are not propagated into the solar "
    "response). " + DPHI_NOTE
)
PROV_V2J = (
    "Kamioka 3D factor v2j = v2i with the up-going correction (c) carried as an "
    "uncertainty only: (a) cap-bias factor f_b on the MC horizon cells (source 0, "
    "cos Z 0.05/0.15, E bins 0-6) and (b) err_horizon_rel as in v2i; (c) c_up = 1 "
    "everywhere, "
    "i.e. the up-going source-3 cells (cos Z -0.05/-0.15) keep their v2h factor_3d and "
    "factor, and err_upcorr_rel becomes species-resolved: E bins 0-2 sqrt(se_jk^2 + "
    "(mean_k template - mean_k mc)^2 + (1 - c_up_v2i[species])^2), se_jk = jackknife "
    "error of the species-mean MC field-free ratio (hzn b4 machinery), E bins 3-4 "
    "|1 - mean_k template| as in v2i. Reason: with c_up applied v2i's U/D at |cos Z| "
    "0.05/0.15 fell 3-6% below the Honda/Bartol envelope at 0.1-0.25 GeV "
    "(INTEGRATION.md 14.6). factor = factor_3d x hybrid_correction (unchanged). "
    "factor_error = factor x sqrt(err_v1c_rel^2 + err_Cpp_rel^2 + "
    "err_engine_spread_rel^2 + err_horizon_rel^2 + err_hybrid_rel^2 + err_capb_rel^2 + "
    "err_upcorr_rel^2). Tool: tools/mceq3d/build_table_v2.py build-v2j; doc: "
    "INTEGRATION.md section 15 (draft $W/runs/v2j/INTEGRATION_15_draft.md). v2h "
    "provenance follows. "
)
CORR_NOTE_V2J = (
    "corr_v2j (4, 10, 20) = corr_fb x corr_cup, azimuth-independent; corr_cup == 1 "
    "everywhere (up-going correction carried as an error only); != 1 only on the MC "
    "horizon rows (E bins 0-6)"
)
ERR_UPCORR_NOTE_V2J = (
    "up-going cos Z -0.05/-0.15, species-resolved: E bins 0-2 sqrt(se_jk^2 + "
    "(mean_k template - mean_k mc)^2 + (1 - c_up_v2i[species])^2), se_jk = jackknife "
    "error of the species-mean MC field-free ratio; E bins 3-4 |1 - mean_k template|; "
    "0 elsewhere. No correction is applied (c_up = 1)"
)
V2I_TERMS = (
    "err_v1c_rel",
    "err_Cpp_rel",
    "err_engine_spread_rel",
    "err_horizon_rel",
    "err_hybrid_rel",
    "err_capb_rel",
    "err_upcorr_rel",
)


def v2i_arrays(h, c):
    """All new v2i / v2j arrays from the v2h npz ``h`` and the corrections npz ``c``.

    ``c["err_upcorr"]`` may have shape (7, 2) (v2i) or (4, 7, 2) (v2j,
    species-resolved).
    """
    assert np.array_equal(h["species"], c["species"])
    assert np.allclose(h["cosz_edges"][8:13], [-0.2, -0.1, 0.0, 0.1, 0.2])
    src = h["source"]
    assert np.all(src[:V2I_NE, list(ROW_DN)] == 0) and np.all(src[:, list(ROW_UP)] == 3)
    ne = h["factor"].shape[1]
    corr_fb = np.ones((4, ne, 20))
    corr_cup = np.ones((4, ne, 20))
    capb = np.zeros((4, ne, 20))
    upc = np.zeros((4, ne, 20))
    hor = h["err_horizon_rel"].copy()
    # (7, 2) species-mean or (4, 7, 2) species-resolved
    eu = np.asarray(c["err_upcorr"])
    assert eu.shape in ((V2I_NE, 2), (4, V2I_NE, 2))
    eu = np.broadcast_to(eu, (4, V2I_NE, 2))
    for j, (rd, ru) in enumerate(zip(ROW_DN, ROW_UP)):
        corr_fb[:, :V2I_NE, rd] = c["f_b"][None, :]
        capb[:, :V2I_NE, rd] = c["err_capb"][None, :]
        corr_cup[:, :3, ru] = c["c_up"][:, :3, j]
        upc[:, :5, ru] = eu[:, :5, j]
        new = np.broadcast_to(c["err_horizon"][None, :, j], (4, V2I_NE)).copy()
        keep = c["keep_v2h_mask"][:, :, j]
        new[keep] = h["err_horizon_rel"][:, :V2I_NE, rd, 0][keep]
        hor[:, :V2I_NE, rd, :] = new[..., None]
    corr = corr_fb * corr_cup
    f3d = h["factor_3d"] * corr[..., None]
    fac = f3d * h["hybrid_correction"][..., None]
    rel2 = (
        h["err_v1c_rel"] ** 2
        + h["err_Cpp_rel"] ** 2
        + h["err_engine_spread_rel"] ** 2
        + hor**2
        + h["err_hybrid_rel"] ** 2
        + capb[..., None] ** 2
        + upc[..., None] ** 2
    )
    touched = (
        (corr != 1.0)[..., None]
        | (capb != 0)[..., None]
        | (upc != 0)[..., None]
        | (hor != h["err_horizon_rel"])
    )
    ferr = np.where(touched, fac * np.sqrt(rel2), h["factor_error"])
    assert np.all(np.isfinite(fac)) and np.all(fac > 0) and np.all(np.isfinite(ferr))
    return dict(
        corr=corr,
        corr_fb=corr_fb,
        corr_cup=corr_cup,
        capb=capb,
        upc=upc,
        hor=hor,
        f3d=f3d,
        fac=fac,
        ferr=ferr,
    )


def _load_h():
    p = os.path.join(V2H, "kamioka_3D_v2h.npz")
    if sha(p) != V2H_SHA256:
        raise RuntimeError(f"v2h sha256 differs from {V2H_SHA256}: {p}")
    return p, np.load(p, allow_pickle=False)


def _corr_path(env, default):
    if len(sys.argv) > 2:
        return sys.argv[2]
    return os.environ.get(env, default)


def _build_candidate(tag):
    """Build kamioka_3D_<tag>.npz (tag v2i or v2j) from v2h and the corrections npz."""
    v2j = tag == "v2j"
    ph, h = _load_h()
    own = V2J if v2j else V2I
    pc = _corr_path(
        "V2J_CORR" if v2j else "V2I_CORR",
        os.path.join(own, "corrections_%s.npz" % tag),
    )
    c = np.load(pc, allow_pickle=False)
    outdir = os.environ.get("V2_OUTDIR", own)
    out = os.path.join(outdir, "kamioka_3D_%s.npz" % tag)
    if os.path.exists(out):
        raise FileExistsError(out)
    os.makedirs(outdir, exist_ok=True)
    a = v2i_arrays(h, c)
    if v2j:
        assert np.all(a["corr_cup"] == 1.0)
    shape = a["fac"].shape
    d = {k: h[k] for k in h.files}
    for k in ("factor", "factor_error", "factor_3d", "err_horizon_rel"):
        d[k + "_v2h"] = h[k]
    d.update(
        factor=a["fac"],
        factor_error=a["ferr"],
        factor_3d=a["f3d"],
        label="kamioka_3D_%s" % tag,
        status="candidate",
        provenance=(PROV_V2J if v2j else PROV_V2I) + str(h["provenance"]),
        build_date=datetime.now().isoformat(timespec="seconds"),
        code_git_rev=git_rev(),
        err_horizon_rel=a["hor"],
        err_horizon_note=HOR_NOTE_V2I,
        err_horizon_note_v2h=str(h["err_horizon_note"]),
        err_capb_rel=np.broadcast_to(a["capb"][..., None], shape).copy(),
        err_capb_note=ERR_CAPB_NOTE,
        err_upcorr_rel=np.broadcast_to(a["upc"][..., None], shape).copy(),
        err_upcorr_note=ERR_UPCORR_NOTE_V2J if v2j else ERR_UPCORR_NOTE,
        dlnproduct_dphi_note=DPHI_NOTE_V2I,
        input_sha256_v2h=sha(ph),
    )
    d["corr_" + tag] = a["corr"]
    d["corr_" + tag + "_note"] = CORR_NOTE_V2J if v2j else CORR_NOTE
    d["corr_fb"] = a["corr_fb"]
    d["corr_cup"] = a["corr_cup"]
    d["input_sha256_corrections_" + tag] = sha(pc)
    for k in c.files:
        if k.startswith("input_sha256_"):
            d["input_sha256_hzn_" + k[len("input_sha256_") :]] = str(c[k])
        elif k in ("b1", "b0", "sigma_b0", "f_b", "c_up", "err_upcorr", "err_capb"):
            d[tag + "_" + k] = c[k]
        elif v2j and k in ("c_up_v2i", "err_upcorr_v2i", "se_jk_species_mean"):
            d[k] = c[k]
    np.savez(out, **d)
    print("wrote", out, sha(out))


def cmd_build_v2i():
    _build_candidate("v2i")


def cmd_build_v2j():
    _build_candidate("v2j")


def cmd_gates_v2i():
    ph, h = _load_h()
    p = os.path.join(V2I, "kamioka_3D_v2i.npz")
    n = np.load(p, allow_pickle=False)
    L = ["# v2i gates V1 (identities) and V2 (errors)", "", f"table sha256 {sha(p)}"]
    L.append("")
    ok = True
    # V1
    corr = n["corr_v2i"]
    one = corr == 1.0
    one4 = np.broadcast_to(one[..., None], n["factor"].shape)
    same = np.array_equal(n["factor_3d"][one4], h["factor_3d"][one4])
    dev = np.abs(n["factor_3d"] / h["factor_3d"] / corr[..., None] - 1).max()
    fh = np.abs(n["factor"] / (n["factor_3d"] * n["hybrid_correction"][..., None]) - 1)
    hsame = np.array_equal(n["hybrid_correction"], h["hybrid_correction"])
    exp = np.zeros(corr.shape, bool)
    exp[:, :V2I_NE, list(ROW_DN)] = True
    exp[:, :3, list(ROW_UP)] = True
    L += ["## V1 identities", ""]
    for name, val, good in (
        ("max |factor_3d_v2i / factor_3d_v2h / corr_v2i - 1|", dev, dev < 1e-14),
        ("max |factor / (factor_3d x hybrid_correction) - 1|", fh.max(), fh.max() == 0),
        ("factor_3d bit-identical to v2h where corr == 1.0", float(same), same),
        ("hybrid_correction bit-identical to v2h", float(hsame), hsame),
        ("corr != 1 only inside the expected set", float(np.all(~one <= exp)), True),
    ):
        good = bool(good) if name[:4] != "corr" else bool(np.all(~one <= exp))
        L.append(f"- {name}: {val:.3g} -> {'PASS' if good else 'FAIL'}")
        ok &= good
    L.append(f"- cells with corr != 1: {int((~one).sum())} of {one.size}")
    # V2
    s2 = sum(n[t] ** 2 for t in V2I_TERMS)
    dd = np.abs((n["factor_error"] / n["factor"]) ** 2 / s2 - 1).max()
    chg = n["factor_error"] != h["factor_error"]
    touch = (
        (n["err_horizon_rel"] != h["err_horizon_rel"])
        | (n["err_capb_rel"] != 0)
        | (n["err_upcorr_rel"] != 0)
        | (corr != 1.0)[..., None]
    )
    chg_ok = bool(np.all(chg <= touch))
    unch = all(
        np.array_equal(n[k], h[k])
        for k in (
            "err_v1c_rel",
            "err_Cpp_rel",
            "err_engine_spread_rel",
            "err_hybrid_rel",
            "dlnproduct_dphi",
        )
    )
    L += ["", "## V2 errors", ""]
    L.append(
        f"- max |(factor_error/factor)^2 / sum(terms^2) - 1| = {dd:.2e} -> "
        f"{'PASS' if dd <= 1e-12 else 'FAIL'}"
    )
    L.append(f"- factor_error changes only in touched cells ({int(chg.sum())} cells "
             f"changed): {'PASS' if chg_ok else 'FAIL'}")
    L.append(f"- unchanged v2h arrays bit-identical: {'PASS' if unch else 'FAIL'}")
    ok &= bool(dd <= 1e-12 and chg_ok and unch)
    L += [
        "",
        "Relative factor_error (median / max over species x azimuth), v2h -> v2i",
        "",
        "| row cos Z | "
        + " | ".join(f"E{i} {E_EDGES[i]:.2f}-{E_EDGES[i + 1]:.2f}" for i in range(8))
        + " |",
        "|---|" + "---|" * 8,
    ]
    for r, lab in ((10, "+0.05"), (11, "+0.15"), (9, "-0.05"), (8, "-0.15")):
        cells = []
        for i in range(8):
            b = h["factor_error"][:, i, r] / h["factor"][:, i, r]
            a2 = n["factor_error"][:, i, r] / n["factor"][:, i, r]
            cells.append(
                f"{np.median(b):.3f}/{b.max():.3f} -> "
                f"{np.median(a2):.3f}/{a2.max():.3f}"
            )
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    L += ["", f"**V1+V2 overall: {'PASS' if ok else 'FAIL'}**", ""]
    open(os.path.join(V2I, "gates_V1V2.md"), "w").write("\n".join(L))
    print("\n".join(L))


#: sha256 of the protected inputs (runs/v2i/out/sha256.txt, INTEGRATION.md 14.5, 13)
PROTECTED_SHA = {
    "v1c": (
        os.path.join(W, "runs", "integ", "out", "kamioka_3D_v1c.npz"),
        "ffa41e0c4edb26f9b1a00bf0e1a250a68a227c86f03310dbabc085eab280d511",
    ),
    "v2h": (
        os.path.join(V2H, "kamioka_3D_v2h.npz"),
        V2H_SHA256,
    ),
    "v2": (
        os.path.join(V2H, "kamioka_3D_v2.npz"),
        "576ea3ead035e2e06bfed208ae503c90a2741dd390a603059d207005dd563c64",
    ),
    "v2i": (
        os.path.join(V2I, "kamioka_3D_v2i.npz"),
        "e4029e6ce3baede6e4634dcdb34f9f54ebde3ecb7b716f93cb892353127e3be5",
    ),
    "corrections_v2i": (
        os.path.join(V2I, "corrections_v2i.npz"),
        "a5fb8fe3608fb115c0d1b4fa58d60936bdbfaf349d85058d95f10c5f642e200a",
    ),
}


def cmd_gates_v2j():
    ph, h = _load_h()
    L = ["# v2j gates (builder): identities and errors vs v2h and v2i", ""]
    ok = True
    L += ["## Protected files (sha256 must be unchanged; a mismatch is a STOP)", ""]
    for k, (pp, ref) in PROTECTED_SHA.items():
        good = sha(pp) == ref
        ok &= good
        L.append(f"- {k}: {sha(pp)} -> {'PASS' if good else 'FAIL (STOP)'}")
        if not good:
            print("\n".join(L))
            raise RuntimeError(f"protected file changed: {pp}")
    pj = os.environ.get("V2_TABLE", os.path.join(V2J, "kamioka_3D_v2j.npz"))
    n = np.load(pj, allow_pickle=False)
    i = np.load(PROTECTED_SHA["v2i"][0], allow_pickle=False)
    L += ["", f"table {pj}", f"sha256 {sha(pj)}", ""]

    def gate(name, val, good):
        nonlocal ok
        ok &= bool(good)
        L.append(f"- {name}: {val} -> {'PASS' if good else 'FAIL'}")

    shp = n["factor"].shape
    # rows / E masks on the (species, E, cos Z, azimuth) grid
    rowmask = np.zeros(shp, bool)
    dn = rowmask.copy()
    up = rowmask.copy()
    dn[:, :, 10:12] = True
    up[:, :, 8:10] = True
    e7 = np.zeros(shp, bool)
    e7[:, :V2I_NE] = True
    e5 = np.zeros(shp, bool)
    e5[:, :5] = True
    L += ["## Identities", ""]
    gate(
        "factor == factor_3d x hybrid_correction (max abs diff)",
        float(
            np.abs(
                n["factor"] - n["factor_3d"] * n["hybrid_correction"][..., None]
            ).max()
        ),
        np.array_equal(n["factor"], n["factor_3d"] * n["hybrid_correction"][..., None]),
    )
    for k in ("factor", "factor_3d", "factor_error"):
        gate(
            f"down-going rows 10/11, all E: {k} bit-identical to v2i",
            "equal",
            np.array_equal(n[k][dn], i[k][dn]),
        )
    for k in ("factor", "factor_3d"):
        gate(
            f"up-going rows 8/9, all E: {k} bit-identical to v2h",
            "equal",
            np.array_equal(n[k][up], h[k][up]),
        )
    chg = n["factor_error"] != h["factor_error"]
    gate(
        "up-going factor_error differs from v2h only in E bins 0-4 and rows 8/9",
        f"{int(chg.sum())} cells differ in total, {int((chg & up).sum())} up-going",
        bool(np.all(chg[up] <= e5[up])) and bool(np.all(chg[up & ~e5] == 0)),
    )
    gate(
        "up-going factor_error differs from v2h in all E bins 0-4 cells",
        int((chg & up & e5).sum()),
        bool(np.all(chg[up & e5])),
    )
    touched = dn | up
    touched &= e7
    ngrid = 0
    bad = []
    for k in h.files:
        a = h[k]
        if a.ndim >= 3 and a.shape[:3] == (4, shp[1], shp[2]):
            tm = touched[..., 0].reshape(touched.shape[:3] + (1,) * (a.ndim - 3))
            m = ~np.broadcast_to(tm, a.shape)
            ngrid += 1
            if not np.array_equal(a[m], n[k][m]):
                bad.append(k)
        elif k not in n.files:
            bad.append(k + " (missing)")
        elif a.dtype.kind in "fiub" and not np.array_equal(a, n[k]):
            bad.append(k)
    gate(
        "every cell outside rows 8-11 / E bins 0-6 bit-identical to v2h, all "
        f"{ngrid} grid arrays (+ non-grid numeric arrays); differing: {bad or 'none'}",
        "",
        not bad,
    )
    s2 = sum(n[t] ** 2 for t in V2I_TERMS)
    dd = float(np.abs((n["factor_error"] / n["factor"]) ** 2 / s2 - 1).max())
    gate(
        "max |(factor_error/factor)^2 / sum(err_*_rel^2) - 1| (limit 1e-12)",
        f"{dd:.2e}",
        dd <= 1e-12,
    )
    gate(
        "label / status",
        f"{n['label']} / {n['status']}",
        str(n["label"]) == "kamioka_3D_v2j" and str(n["status"]) == "candidate",
    )
    gate("corr_cup == 1 everywhere", "", np.all(n["corr_cup"] == 1.0))
    gate(
        "hybrid_correction bit-identical to v2h",
        "",
        np.array_equal(n["hybrid_correction"], h["hybrid_correction"]),
    )
    L += [
        "",
        "Relative factor_error (median / max over species x azimuth), v2h | v2i -> v2j",
        "",
        "| row cos Z | "
        + " | ".join(f"E{k} {E_EDGES[k]:.2f}-{E_EDGES[k + 1]:.2f}" for k in range(8))
        + " |",
        "|---|" + "---|" * 8,
    ]
    for r, lab in ((10, "+0.05"), (11, "+0.15"), (9, "-0.05"), (8, "-0.15")):
        cells = []
        for k in range(8):
            q = {
                t: m["factor_error"][:, k, r] / m["factor"][:, k, r]
                for t, m in (("h", h), ("i", i), ("j", n))
            }
            cells.append(
                " -> ".join(f"{np.median(q[t]):.3f}/{q[t].max():.3f}" for t in "hij")
            )
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    L += ["", f"**Builder gates overall: {'PASS' if ok else 'FAIL'}**", ""]
    out = os.environ.get("V2_GATES", os.path.join(V2J, "gates.md"))
    if os.path.exists(out):
        raise FileExistsError(out)
    open(out, "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    {
        "regress": cmd_regress,
        "build": cmd_build,
        "gates": cmd_gates,
        "build-v2i": cmd_build_v2i,
        "gates-v2i": cmd_gates_v2i,
        "build-v2j": cmd_build_v2j,
        "gates-v2j": cmd_gates_v2j,
    }[sys.argv[1]]()
