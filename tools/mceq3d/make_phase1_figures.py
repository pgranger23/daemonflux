#!/usr/bin/env python3
"""Regenerate every PNG in ``figures_phase1/``.

Thirteen figures documenting the Phase-1 audit and repair of the 3D /
geomagnetic extension (``PHASE1_RESULTS.md``).  One PNG per figure, 900 px wide
at 150 dpi, light surface, one palette across the whole set.

Data provenance
---------------
Almost every number is a literal transcribed from the Phase-1 session logs and
is annotated with its source file at the point of use.  Arrays come from:

* repository files (read-only): ``honda_kam.npz``, ``offaxis_excess.npz``,
  ``offaxis_excess_channel_v2.npz``, ``m_spliced.npz``, ``m_spliced_v2.npz``
* ``figures_phase1/data/*.npz`` -- small extracts of the session scratchpad
  made once by ``figures_phase1/data/extract_scratchpad.py``
* ``figures_phase1/data/grid_delivered.npz`` -- the ONE live solve in this set
  (figure 12), produced by ``figures_phase1/data/solve_grid.py``

Nothing here writes to a tracked file.

Run
---
    export PY=/cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc14-opt/bin/python3.12
    export PYTHONPATH=<venv site-packages>:<repo>/src:$PWD
    $PY make_phase1_figures.py            # all figures
    $PY make_phase1_figures.py 03 07      # just these
"""
from __future__ import annotations

import os
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures_phase1")
DATA = os.path.join(OUT, "data")

# ---------------------------------------------------------------------------
# Design system.  Light surface only (these are PNGs).  Palette values are the
# dataviz reference instance; the categorical order and the four-slot species
# set were validated with scripts/validate_palette.js --mode light --pairs all
# (worst CVD dE 9.2, worst normal-vision dE 16.3; aqua carries a sub-3:1
# contrast WARN, so every aqua series is direct-labelled or in a legend).
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
SEQ3 = ["#86b6ef", "#2a78d6", "#104281"]          # ordinal ramp, 3 steps
# Diverging: the palette's blue <-> red poles with the neutral gray midpoint.
# Polarity encoding (log ratio around 0), never a rainbow.
DIVCMAP = LinearSegmentedColormap.from_list(
    "phase1_div", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec",
                   "#f2b0ae", "#e34948", "#8e1f1f"])

# Roles held fixed across the whole set -------------------------------------
C_NEW = BLUE       # repaired / delivered / v2 / dense-node / sharp cutoff
C_OLD = ORANGE     # paper-era: legacy map, legacy moments, sigma=0.5, detector
C_ALT = AQUA       # a third variant in the same comparison
C_TRUTH = INK      # direct back-trace / measured ladder / NA61 data / Honda

SPEC = {"numu": BLUE, "antinumu": ORANGE, "nue": AQUA, "antinue": VIOLET}
SPEC_LBL = {"numu": r"$\nu_\mu$", "antinumu": r"$\bar\nu_\mu$",
            "nue": r"$\nu_e$", "antinue": r"$\bar\nu_e$"}

LW = 1.6
LWR = 1.0          # reference / secondary lines
MS = 4.0

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.handlelength": 2.2,
    "lines.solid_capstyle": "round",
})


def style(ax, xlabel=None, ylabel=None, title=None, grid="y"):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", pad=6)
    if grid:
        ax.grid(True, axis=grid, zorder=0)
        ax.set_axisbelow(True)
    return ax


def note(fig, text, dy=-0.012):
    """Record the provenance line; ``save`` renders it under the figure."""
    fig._src_note = text
    fig._src_dy = dy


def save(fig, name):
    src = getattr(fig, "_src_note", None)
    if src:
        w = int(fig.get_size_inches()[0] * 21)      # ~5.8 pt chars per inch
        fig.text(0.004, getattr(fig, "_src_dy", -0.012),
                 "\n".join(textwrap.wrap(src, w)), ha="left",
                 va="top", fontsize=5.8, color=MUTED, linespacing=1.4)
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    print("wrote", os.path.relpath(path, HERE), flush=True)


def logi(x, xg, y):
    """log-log interpolation of y(xg) at x."""
    return np.exp(np.interp(np.log(x), np.log(xg),
                            np.log(np.maximum(np.asarray(y, float), 1e-300))))


# ===========================================================================
# 01  cutoff_scan
# ===========================================================================
# scratchpad/buildmap.log ("bilinear interpolation error at azimuth 90 (East)")
BM_ZEN = [81.0, 84.0, 84.75, 85.5, 86.25, 87.0, 87.5, 88.0, 88.25, 89.0, 89.5]
BM_DIRECT = [36.02, 38.62, 39.30, 39.98, 40.79, 41.72, 43.02, 44.44, 45.25,
             48.09, 50.63]
BM_DENSE = [36.05, 38.62, 39.30, 39.98, 40.85, 41.72, 43.08, 44.44, 45.28,
            48.09, 50.63]
BM_LEGACY = [36.11, 40.30, 41.47, 42.64, 43.81, 44.98, 45.76, 46.54, 46.93,
             48.09, 45.82]


def fig01():
    d = np.load(os.path.join(DATA, "penumbra_ladders.npz"))
    R, A = d["vertical_R"], d["vertical_A"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.0, 2.9))

    ax.plot(BM_ZEN, BM_LEGACY, "--", color=C_OLD, lw=LW, marker="s", ms=MS - 1,
            mfc=SURFACE, mew=1.1, label="old map, 13 uniform nodes", zorder=3)
    ax.plot(BM_ZEN, BM_DENSE, "-", color=C_NEW, lw=LW, marker="o", ms=MS - 1,
            label="new map, dense limb nodes", zorder=4)
    ax.plot(BM_ZEN, BM_DIRECT, color=C_TRUTH, lw=LWR, ls=(0, (1, 1.4)),
            marker="+", ms=MS + 1, mew=1.0, label="direct back-trace",
            zorder=5)
    i = BM_ZEN.index(87.0)
    ax.annotate(f"+{BM_LEGACY[i] - BM_DIRECT[i]:.2f} GV",
                xy=(87.0, BM_LEGACY[i]), xytext=(85.0, 47.6),
                fontsize=7, color=C_OLD,
                arrowprops=dict(arrowstyle="-", color=C_OLD, lw=0.7))
    ax.annotate("89$^\\circ$ seam", xy=(89.5, 45.82), xytext=(88.15, 42.9),
                fontsize=6.8, color=C_OLD, ha="left",
                arrowprops=dict(arrowstyle="->", color=C_OLD, lw=0.7))
    style(ax, "arrival zenith [deg]", r"$R_c$ [GV]",
          "a  Cutoff vs zenith, geographic East")
    ax.set_xlim(80.5, 90.0)
    ax.legend(loc="upper left", borderpad=0.2)

    # -- b: the rigidity ladder at the vertical -----------------------------
    bx.fill_between(R, 0, A, step="mid", color=C_NEW, alpha=0.18, lw=0)
    bx.step(R, A, where="mid", color=C_NEW, lw=LW, label="back-traced admittance")
    for x, c, lbl, xt, yt in (
            (8.79, C_OLD, "old scan\nstops at 8.79 GV", 8.68, 0.74),
            (11.37, C_TRUTH, "bisection\n11.37 GV", 11.25, 0.30)):
        bx.axvline(x, color=c, lw=LWR, ls="--", zorder=5)
        bx.text(xt, yt, lbl, fontsize=6.8, color=c, ha="right", va="center",
                linespacing=1.25)
    bx.annotate("", xy=(9.50, 1.13), xytext=(10.70, 1.13),
                arrowprops=dict(arrowstyle="|-|,widthA=0.25,widthB=0.25",
                                color=INK2, lw=0.8))
    bx.text(10.1, 1.21, "allowed islands", fontsize=6.5, color=INK2,
            ha="center")
    style(bx, "rigidity [GV]", "admitted (1) / forbidden (0)",
          "b  Rigidity ladder at the vertical")
    bx.set_xlim(6.6, 13.0)
    bx.set_ylim(-0.06, 1.32)
    bx.set_yticks([0, 1])

    note(fig, "a: scratchpad/buildmap.log  ·  b: scratchpad/pen/penumbra.npz "
              "(0.1 GV ladder, 72 traces), scan.log")
    save(fig, "01_cutoff_scan.png")


# ===========================================================================
# 02  penumbra
# ===========================================================================
def _erf_T(R, rc, sigma):
    from scipy.special import erf
    return 0.5 * (1.0 + erf(np.log(np.maximum(R, 1e-12) / rc)
                            / (np.sqrt(2.0) * sigma)))


def fig02():
    d = np.load(os.path.join(DATA, "penumbra_ladders.npz"))
    # scratchpad/pen/penumbra.json: R_U (highest forbidden rigidity)
    cases = [("vertical", d["vertical_R"], d["vertical_A"], 11.453,
              "vertical  ($R_U$ = 11.45 GV)"),
             ("87E", d["87E_R"], d["87E_A"], 41.644,
              r"87$^\circ$ East  ($R_U$ = 41.64 GV)")]

    # sub-cutoff leak of the legacy erf, on an E^-2.7 primary spectrum
    def leak(sigma, rc=1.0):
        r = np.logspace(np.log10(rc) - 2.5, np.log10(rc) + 2.5, 20001)
        w = r ** -2.7
        T = _erf_T(r, rc, sigma)
        below = np.trapezoid(T[r < rc] * w[r < rc], r[r < rc])
        return below / np.trapezoid(T * w, r)

    lk = leak(0.5)

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.9), sharey=True)
    for ax, (_, R, A, ru, title) in zip(axes, cases):
        x = R / ru
        m = (x > 0.55) & (x < 1.45)
        ax.step(x[m], A[m], where="mid", color=C_TRUTH, lw=LW,
                label="measured")
        xf = np.linspace(0.55, 1.45, 400)
        ax.plot(xf, _erf_T(xf, 1.0, 0.5), color=C_OLD, lw=LW, ls="--",
                label=r"erf $\sigma_{\ln R}$=0.5 (old)")
        ax.plot(xf, (xf >= 1.0).astype(float), color=C_NEW, lw=LW,
                label=r"$\sigma_{\ln R}$=0 (delivered)")
        style(ax, r"$R\,/\,R_U$", None, title)
        ax.set_ylim(-0.05, 1.12)
        ax.set_xlim(0.55, 1.45)
    axes[0].set_ylabel("transmission / admittance")
    axes[0].set_yticks([0, 0.5, 1])
    axes[1].legend(loc="lower right", borderpad=0.2,
                   bbox_to_anchor=(1.02, -0.02))
    axes[1].fill_between([0.55, 1.0], 0, 1.12, color=C_OLD, alpha=0.07, lw=0)
    axes[1].text(0.575, 1.06,
                 f"the legacy erf puts\n{lk:.0%} of the admitted\n"
                 r"flux BELOW $R_U$",
                 fontsize=6.8, color=C_OLD, va="top", linespacing=1.35)
    axes[1].text(1.055, 0.66, "measured penumbra\n"
                             r"$\sigma_{16/84}=0.007$", fontsize=6.5,
                 color=INK2, va="top", linespacing=1.3)
    axes[0].text(0.575, 1.06, "two allowed\nislands\n"
                              r"$\sigma_{16/84}=0.13$", fontsize=6.5,
                 color=INK2, va="top", linespacing=1.35)

    note(fig, "ladders: scratchpad/pen/penumbra.npz + penumbra.json (0.1 GV "
              "steps, 1013 traces)  ·  transmission: mceq3d_flux._transmission")
    save(fig, "02_penumbra.png")


# ===========================================================================
# 03  moments_arcsin
# ===========================================================================
# scratchpad/na61_angle.log, pi+ block: NA61 <theta> [mrad], UrQMD v2 <theta>,
# and the ratio the OLD (arctan) moments gave.
NA61_P = [0.35, 0.46, 0.60, 0.80, 1.06, 1.40, 1.85, 2.45, 3.24, 4.29, 5.67,
          7.51, 9.93, 13.14, 17.39]
NA61_TH = [276.4, 264.4, 254.7, 242.1, 214.9, 167.4, 152.1, 120.6, 110.2, 92.0,
           68.9, 59.6, 51.1, 42.9, 34.5]
URQMD_TH_V2 = [262.8, 255.4, 248.0, 234.5, 217.8, 194.9, 171.0, 145.1, 120.0,
               96.6, 78.7, 62.9, 50.7, 41.6, 34.1]
URQMD_RATIO_OLD = [0.908, 0.923, 0.932, 0.930, 0.976, 1.127, 1.094, 1.176,
                   1.071, 1.038, 1.134, 1.051, 0.989, 0.968, 0.988]


def fig03():
    o = np.load(os.path.join(HERE, "m_spliced.npz"))
    v = np.load(os.path.join(HERE, "m_spliced_v2.npz"))
    pe = o["proj_energies"]
    ip = int(np.argmin(np.abs(pe - 20.0)))

    es = o["e_sec"][ip]
    th_o = np.degrees(np.sqrt(o["theta_sq"][ip]))
    th_v = np.degrees(np.sqrt(v["theta_sq"][ip]))
    m = np.isfinite(th_o) & np.isfinite(th_v) & (es > 0.08) & (es < 15.0)

    fig, axes = plt.subplots(1, 3, figsize=(6.0, 2.7))
    fig.subplots_adjust(wspace=0.42)

    ax = axes[0]
    ax.plot(es[m], th_o[m], color=C_OLD, lw=LW, ls="--", label="old")
    ax.plot(es[m], th_v[m], color=C_NEW, lw=LW, label="v2")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_yticks([3, 10, 30, 70])
    ax.set_yticklabels(["3", "10", "30", "70"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([0.1, 1, 10])
    ax.set_xticklabels(["0.1", "1", "10"])
    style(ax, r"secondary $E_\pi$ [GeV]",
          r"$\sqrt{\langle\theta^2\rangle}$ [deg]", "a  cone width")
    ax.legend(loc="lower left", borderpad=0.2)
    ax.text(0.97, 0.95, r"$\pi^+$, $E_p$ = 20 GeV", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.8, color=INK2)

    ax = axes[1]
    ax.plot(es[m], th_v[m] / th_o[m], color=C_NEW, lw=LW, label="v2 / old")
    tan = np.tan(np.radians(th_o[m]))
    ok = tan < 0.999
    ax.plot(es[m][ok], np.degrees(np.arcsin(tan[ok])) / th_o[m][ok],
            color=MUTED, lw=LWR, ls=(0, (3, 2)),
            label=r"$\arcsin/\arctan$ only")
    ax.axhline(1.0, color=AXIS, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 1, 10])
    ax.set_xticklabels(["0.1", "1", "10"])
    style(ax, r"secondary $E_\pi$ [GeV]", "width ratio", "b  size of the fix")
    ax.set_ylim(0.96, 1.80)
    ax.legend(loc="upper right", borderpad=0.2)

    ax = axes[2]
    old_th = np.array(NA61_TH) * np.array(URQMD_RATIO_OLD)
    ax.plot(NA61_P, old_th, color=C_OLD, lw=LW, ls="--")
    ax.plot(NA61_P, URQMD_TH_V2, color=C_NEW, lw=LW)
    ax.plot(NA61_P, NA61_TH, color=C_TRUTH, lw=0, marker="o", ms=MS,
            mfc=SURFACE, mew=1.2, label="NA61 data")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_yticks([30, 50, 100, 200, 300])
    ax.set_yticklabels(["30", "50", "100", "200", "300"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([0.3, 1, 3, 10])
    ax.set_xticklabels(["0.3", "1", "3", "10"])
    style(ax, r"$p_{\rm lab}$ [GeV]", r"$\langle\theta\rangle$ [mrad]",
          "c  NA61 closure")
    ax.legend(loc="lower left", borderpad=0.2, bbox_to_anchor=(-0.02, -0.02))
    ax.text(0.03, 0.24, "mean $\\langle\\theta\\rangle$ ratio to NA61\n"
                        "v2 1.043  ·  old 1.020", transform=ax.transAxes,
            fontsize=6.5, color=INK2, va="bottom", linespacing=1.35)

    note(fig, "a,b: m_spliced.npz vs m_spliced_v2.npz  ·  "
              "c: scratchpad/na61_angle.log (validate_na61_angle.py, p+C at "
              "31 GeV/c, 120k events)")
    save(fig, "03_moments_arcsin.png")


# ===========================================================================
# 04  cone_widths
# ===========================================================================
# scratchpad/w2/cone_widths.log, row "(c0) mudecay_shape (existing)"
OLD_MU_E = [0.20, 0.30, 0.50, 1.00, 3.00]
OLD_MU_W = [26.45, 19.76, 13.77, 8.83, 5.20]


def fig04():
    d = np.load(os.path.join(HERE, "offaxis_excess_channel_v2.npz"))
    e = d["e"]
    m = (e >= 0.11) & (e <= 20.0)

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for key, col, lbl in (("sigma_k", C_OLD, r"kaon cone  $\sigma_K$"),
                          ("sigma_mu_numu", C_NEW,
                           r"$\mu$-decay $\to\nu_\mu$  (v2, + bend)"),
                          ("sigma_mu_nue", C_ALT,
                           r"$\mu$-decay $\to\nu_e$  (v2, + bend)"),
                          ("sigma_pi", VIOLET,
                           r"direct $\pi\to\mu\nu$  $\sigma_\pi$")):
        ax.plot(e[m], d[key][m], color=col, lw=LW, label=lbl)
    ax.plot(OLD_MU_E, OLD_MU_W, color=C_TRUTH, lw=LWR, ls=(0, (4, 2)),
            marker="s", ms=MS - 1, mfc=SURFACE, mew=1.0,
            label=r"old flavour-blind $\mu$-decay cone")
    ax.set_xscale("log")
    ax.set_yscale("log")
    style(ax, r"$E_\nu$ [GeV]", "space-angle RMS of the cone [deg]",
          "Parent-channel cone widths, delivered (v2) engine", grid="both")
    ax.set_xlim(0.11, 20)
    ax.legend(loc="upper right", borderpad=0.3)
    ax.annotate("the corrected $\\mu$-decay cone is WIDER below\n"
                r"$\sim$0.4 GeV and NARROWER above 1 GeV",
                xy=(3.0, 5.2), xytext=(0.3, 1.35), fontsize=6.8, color=INK2,
                linespacing=1.35,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.7,
                                connectionstyle="arc3,rad=-0.15"))
    note(fig, "offaxis_excess_channel_v2.npz (sigma_pi / sigma_k / "
              "sigma_mu_numu / sigma_mu_nue)  ·  old curve: "
              "scratchpad/w2/cone_widths.log row (c0)")
    save(fig, "04_cone_widths.png")


# ===========================================================================
# 05  eoff_species
# ===========================================================================
def fig05():
    flat = np.load(os.path.join(HERE, "offaxis_excess.npz"))
    ch = np.load(os.path.join(HERE, "offaxis_excess_channel_v2.npz"))
    e, cz = ch["e"], ch["cz"]
    sp = list(ch["species"])
    E_PTS = (0.3, 0.5, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.9), sharey=True)
    for ax, s in zip(axes, ("numu", "nue")):
        js = sp.index(s)
        for E, col in zip(E_PTS, SEQ3):
            y_ch = [logi(E, e, ch["E_off_s"][js, i]) for i in range(len(cz))]
            y_fl = [logi(E, e, flat["E_off"][i]) for i in range(len(cz))]
            ax.plot(cz, y_fl, color=col, lw=LWR, ls="--")
            ax.plot(cz, y_ch, color=col, lw=LW, marker="o", ms=MS - 1.2,
                    label=f"{E:g} GeV")
        style(ax, r"$\cos Z$", None, SPEC_LBL[s])
        ax.axhline(1.0, color=AXIS, lw=0.8)
        ax.set_xlim(0.0, 1.0)
    axes[0].set_ylabel(r"off-axis production excess  $E_{\rm off}$")
    h = [Line2D([], [], color=INK2, lw=LW, label="channel_v2 (per species)"),
         Line2D([], [], color=INK2, lw=LWR, ls="--", label="flat table (old)")]
    axes[0].legend(loc="upper right", borderpad=0.2, title="colour = $E_\\nu$",
                   title_fontsize=6.5)
    axes[1].legend(handles=h, loc="upper right", borderpad=0.2)
    axes[1].text(0.34, 1.68,
                 "the flat table is species-blind:\nthe dashed curves are the "
                 "same\nin both panels", fontsize=6.5, color=INK2, va="top",
                 linespacing=1.35)
    note(fig, "offaxis_excess.npz (E_off, flat)  vs  "
              "offaxis_excess_channel_v2.npz (E_off_s, per species)")
    save(fig, "05_eoff_species.png")


# ===========================================================================
# 06  stage_ladder
# ===========================================================================
STAGES = ["S0", "S1", "S1b", "S2", "S4", r"$\sigma$=0", "anchor"]
# scratchpad/jc/stages_s4.log (S0..S4) + pen/table_all.txt (sigma=0)
# + anchor/hv_after.log (anchor, numu only).
HV_HONDA = {                       # model / Honda, azimuth-averaged H/V
    ("numu", 0.3): [0.841, 0.850, 0.853, 0.921, 1.040, 1.037, 1.031],
    ("numu", 0.5): [0.871, 0.877, 0.879, 0.941, 1.019, 1.013, 0.998],
    ("numu", 1.0): [0.929, 0.935, 0.935, 0.981, 1.001, 0.988, 0.971],
    ("nue", 0.3): [0.842, 0.857, 0.857, 0.886, 1.075, 1.073, np.nan],
    ("nue", 0.5): [0.844, 0.858, 0.858, 0.906, 1.010, 1.006, np.nan],
    ("nue", 1.0): [0.897, 0.908, 0.908, 0.957, 0.955, 0.946, np.nan],
}
HV_BARTOL = {                      # model / Bartol, same stages
    ("numu", 0.3): [0.915, 0.925, 0.928, 1.002, 1.131, 1.128, 1.122],
    ("numu", 0.5): [0.908, 0.915, 0.917, 0.982, 1.063, 1.057, 1.041],
    ("numu", 1.0): [0.911, 0.916, 0.917, 0.962, 0.981, 0.968, 0.952],
    ("nue", 0.3): [0.876, 0.892, 0.892, 0.922, 1.118, 1.117, np.nan],
    ("nue", 0.5): [0.873, 0.887, 0.887, 0.936, 1.044, 1.040, np.nan],
    ("nue", 1.0): [0.851, 0.861, 0.861, 0.908, 0.907, 0.898, np.nan],
}
# W/E deviation from Honda, numu 0.5 GeV, zenith 87/81/76 deg.
# S0..S4: stages_s4.log; sigma=0: pen/table_all.txt (87 and 76 measured, 81
# absent from that table -> np.nan); anchor: anchor/ewz_after.log (87, 81).
WE_DEV = {
    87: [15, 13, 12, 6, 2, 13, 12],
    81: [14, 10, 9, 4, 0, np.nan, 11],
    76: [11, 9, 9, 3, 0, 11, np.nan],
}
E_LS = {0.3: "-", 0.5: (0, (5, 2)), 1.0: (0, (1.2, 1.4))}


def fig06():
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 3.3), layout="constrained")
    x = np.arange(len(STAGES))
    for k, (ax, tab, ref) in enumerate(((axes[0], HV_HONDA, "Honda"),
                                        (axes[1], HV_BARTOL, "Bartol"))):
        for (sp, E), y in tab.items():
            ax.plot(x, y, color=SPEC[sp], lw=LW, ls=E_LS[E],
                    marker="o", ms=MS - 1.4, mfc=SURFACE, mew=1.0)
        ax.axhline(1.0, color=AXIS, lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(STAGES, rotation=55, ha="right")
        style(ax, None, None, f"{'ab'[k]}  H/V  ÷ {ref}")
        ax.set_ylim(0.80, 1.22)
    axes[0].set_ylabel("model / reference, horizon-to-vertical")

    ax = axes[2]
    for zen, col in zip((87, 81, 76), SEQ3[::-1]):
        ax.plot(x, WE_DEV[zen], color=col, lw=LW, marker="o", ms=MS - 1.4,
                mfc=SURFACE, mew=1.0, label=f"{zen}$^\\circ$")
    ax.axhline(0.0, color=AXIS, lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(STAGES, rotation=55, ha="right")
    style(ax, None, "W/E minus Honda [%]", r"c  W/E, $\nu_\mu$ 0.5 GeV")
    ax.set_ylim(-1.5, 18.5)
    ax.set_yticks([0, 5, 10, 15])
    ax.legend(loc="upper right", borderpad=0.2, title="zenith",
              title_fontsize=6.5, labelspacing=0.3)

    h = ([Line2D([], [], color=SPEC[sp], lw=LW, label=SPEC_LBL[sp])
          for sp in ("numu", "nue")]
         + [Line2D([], [], color=INK2, lw=LW, ls=E_LS[E], label=f"{E:g} GeV")
            for E in (0.3, 0.5, 1.0)])
    axes[0].legend(handles=h, loc="upper left", borderpad=0.2,
                   labelspacing=0.35)
    axes[1].text(0.03, 0.97, r"no $\nu_e$ point was" "\n"
                             "measured at the\nanchor stage",
                 transform=axes[1].transAxes, ha="left", va="top",
                 fontsize=6.2, color=MUTED, linespacing=1.35)
    note(fig, "S0-S4: scratchpad/jc/stages_s4.log · sigma=0: pen/table_all.txt "
              "· anchor: anchor/hv_after.log (nu_mu; the /Bartol column "
              "rescaled by the published Honda/Bartol ratio) and "
              "anchor/ewz_after.log · gaps in c = not tabulated at that stage")
    save(fig, "06_stage_ladder.png")


# ===========================================================================
# 07  azimuth_pattern
# ===========================================================================
SP4 = ("numu", "antinumu", "nue", "antinue")
HK = {"numu": "numu", "antinumu": "numubar", "nue": "nue", "antinue": "nuebar"}
EW_AXIS = 81.88          # geomagnetic East, compass deg (diag_ew_charge_fourier)


def _honda_pattern(E=0.5, cz=0.05):
    """Honda's 12 azimuth bins at one (E, cosZ), on OUR compass axis.

    Honda's azimuth is counterclockwise from South (arXiv:1102.2688 II);
    az_compass = (180 - az_Honda) mod 360.
    """
    h = dict(np.load(os.path.join(HERE, "honda_kam.npz")))
    icz = int(np.argmin(np.abs(h["czlo"] - round(cz - 0.05, 2))))
    az_h = h["azlo"].astype(float) + 15.0
    az_c = (180.0 - az_h) % 360.0
    out = {}
    for s in SP4:
        y = np.array([logi(E, h["E"], h[HK[s]][icz, j]) for j in range(12)])
        o = np.argsort(az_c)
        out[s] = (az_c[o], y[o])
    return out


def fig07():
    d = np.load(os.path.join(DATA, "pattern_cz005.npz"))
    e, az = d["e"], d["az"]
    E = 0.5
    hon = _honda_pattern(E, 0.05)

    def norm(y):
        return y / y.mean()

    model = {}
    for s in SP4:
        model[s] = {
            "prod": norm(np.array([logi(E, e, d[f"prod_total_{s}"][j])
                                   for j in range(len(az))])),
            "det": norm(np.array([logi(E, e, d[f"det_total_{s}"][j])
                                  for j in range(len(az))])),
        }
    honda = {s: norm(hon[s][1]) for s in SP4}
    az_h = hon["numu"][0]

    fig = plt.figure(figsize=(6.0, 6.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.05], hspace=0.55,
                          wspace=0.28)
    for k, s in enumerate(SP4):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        ax.axvline(EW_AXIS, color=GRID, lw=1.4, zorder=0)
        ax.plot(az_h, honda[s], color=C_TRUTH, lw=LWR, ls="--", marker="o",
                ms=MS - 1, mfc=SURFACE, mew=1.0, label="Honda")
        ax.plot(az, model[s]["det"], color=SPEC[s], lw=LWR, ls=(0, (1.2, 1.4)),
                label="detector anchor")
        ax.plot(az, model[s]["prod"], color=SPEC[s], lw=LW,
                label="delivered (prod-point)")
        style(ax, None, None, f"{SPEC_LBL[s]}")
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xticklabels(["N", "E", "S", "W", "N"])
        ax.set_ylim(0.30, 1.95)
        ax.text(0.03, 0.98,
                f"max/min   Honda {honda[s].max()/honda[s].min():.2f}\n"
                f"{'':13s}model {model[s]['prod'].max()/model[s]['prod'].min():.2f}",
                transform=ax.transAxes, va="top", fontsize=6.3, color=INK2,
                linespacing=1.3, family="monospace")
        if k == 0:
            hleg = ax.get_legend_handles_labels()
        if k % 2 == 0:
            ax.set_ylabel(r"$\phi(\rm az)\,/\,\langle\phi\rangle_{\rm az}$")

    for j, (lbl, getter) in enumerate(
            (("Honda: the four species spread", lambda s: (az_h, honda[s])),
             ("delivered model: they collapse",
              lambda s: (az, model[s]["prod"])))):
        ax = fig.add_subplot(gs[2, j])
        ax.axvline(EW_AXIS, color=GRID, lw=1.4, zorder=0)
        for s in SP4:
            xx, yy = getter(s)
            ax.plot(xx, yy, color=SPEC[s], lw=LW, label=SPEC_LBL[s])
        style(ax, "arrival azimuth (compass)", None, f"{'de'[j]}  {lbl}")
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xticklabels(["N", "E", "S", "W", "N"])
        ax.set_ylim(0.30, 1.95)
        if j == 0:
            ax.set_ylabel(r"$\phi(\rm az)\,/\,\langle\phi\rangle_{\rm az}$")
            ax.legend(loc="lower right", ncol=2, borderpad=0.2,
                      columnspacing=1.0)
    fig.legend(*hleg, loc="upper left", bbox_to_anchor=(0.06, 0.972), ncol=3,
               columnspacing=1.6, borderpad=0.2)
    fig.suptitle(r"Azimuth pattern at $\cos Z$ = 0.05, $E_\nu$ = 0.5 GeV  "
                 "(Honda's 12 bins, compass convention)", x=0.012, y=1.005,
                 ha="left", fontsize=9, color=INK)
    note(fig, "model: scratchpad/anchor/pat_after.npz (prod_point) and "
              "pattern/pat_a.npz (detector), both diag_ew_pattern.py  ·  "
              "Honda: honda_kam.npz, az_compass = 180 - az_Honda  ·  grey rule "
              "= geomagnetic East 81.9 deg")
    save(fig, "07_azimuth_pattern.png")


# ===========================================================================
# 08  charge_split
# ===========================================================================
# scratchpad/anchor/fourier_after.log, "CHARGE SPLITTING nu minus nubar"
# (delivered engine, cutoff_anchor='prod_point').  Rows: E, ours, Honda.
CS_E = [0.3, 0.5, 1.0, 2.0]
CHARGE_SPLIT = {
    (87, "numu", "dmm"): ([-0.09, -0.16, -0.19, -0.10],
                          [-0.91, -1.30, -1.53, -1.21]),
    (87, "nue", "dmm"): ([+0.33, +0.50, +0.61, +0.48],
                         [+1.69, +2.62, +3.26, +2.43]),
    (87, "numu", "dphi"): ([+4.56, +4.30, +3.78, +3.21],
                           [+5.95, +4.36, +1.70, -2.63]),
    (87, "nue", "dphi"): ([-9.12, -8.53, -7.62, -7.16],
                          [-12.22, -9.21, +0.44, +11.36]),
    (81, "numu", "dmm"): ([-0.13, -0.16, -0.14, -0.06],
                          [-0.80, -1.06, -0.95, -0.56]),
    (81, "nue", "dmm"): ([+0.42, +0.55, +0.58, +0.40],
                         [+1.76, +2.36, +2.24, +1.38]),
    (81, "numu", "dphi"): ([+4.33, +3.83, +3.37, +3.00],
                           [+3.49, +2.90, +0.59, -1.47]),
    (81, "nue", "dphi"): ([-8.86, -8.15, -7.43, -7.12],
                          [-7.81, -5.85, -0.78, +1.82]),
}
PAIR_LBL = {"numu": r"$\nu_\mu-\bar\nu_\mu$", "nue": r"$\nu_e-\bar\nu_e$"}
PAIR_C = {"numu": BLUE, "nue": AQUA}


def fig08():
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 4.4), sharex=True,
                             sharey="row")
    rows = (("dphi", r"$\Delta(\delta\varphi)$  [deg]", "phase splitting"),
            ("dmm", r"$\Delta(\max/\min)$", "amplitude splitting"))
    for i, (obs, ylab, oname) in enumerate(rows):
        for j, zen in enumerate((87, 81)):
            ax = axes[i, j]
            for pair in ("numu", "nue"):
                ours, hon = CHARGE_SPLIT[(zen, pair, obs)]
                ax.plot(CS_E, hon, color=PAIR_C[pair], lw=LWR, ls="--",
                        marker="o", ms=MS, mfc=SURFACE, mew=1.1)
                ax.plot(CS_E, ours, color=PAIR_C[pair], lw=LW, marker="o",
                        ms=MS - 1.2)
            ax.axhline(0.0, color=AXIS, lw=0.9)
            ax.set_xscale("log")
            ax.set_xticks(CS_E)
            ax.set_xticklabels([f"{v:g}" for v in CS_E])
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.xaxis.set_minor_locator(plt.NullLocator())
            style(ax, r"$E_\nu$ [GeV]" if i == 1 else None,
                  ylab if j == 0 else None,
                  f"{'ab'[i]}{j + 1}  {oname}, {zen}$^\\circ$")
    h = ([Line2D([], [], color=PAIR_C[p], lw=LW, label=PAIR_LBL[p])
          for p in ("numu", "nue")]
         + [Line2D([], [], color=INK2, lw=LW, label="delivered model"),
            Line2D([], [], color=INK2, lw=LWR, ls="--", marker="o", ms=MS,
                   mfc=SURFACE, mew=1.1, label="Honda")])
    fig.subplots_adjust(top=0.86)
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.06, 0.985),
               ncol=4, columnspacing=1.4, borderpad=0.2)
    fig.suptitle("Charge-splitting Fourier observables, "
                 r"$\cos Z$ = 0.05 / 0.15" "\n"
                 "the model reproduces 60-90% of Honda's phase splitting but "
                 "only 13-24% of the amplitude splitting",
                 x=0.012, y=1.085, ha="left", va="top", fontsize=9, color=INK,
                 linespacing=1.5)
    note(fig, "scratchpad/anchor/fourier_after.log "
              "(diag_ew_charge_fourier.py, cutoff_anchor='prod_point')")
    save(fig, "08_charge_split.png")


# ===========================================================================
# 09  bend_scale
# ===========================================================================
# scratchpad/pattern/channels_tables.txt, block "cz 005", section C.
BS_SCALE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
BS = {
    "numu": [2.971, 2.927, 2.886, 2.846, 2.807, 2.733, 2.599, 2.470],
    "antinumu": [2.974, 3.030, 3.090, 3.149, 3.210, 3.335, 3.540, 3.655],
    "nue": [3.104, 3.208, 3.322, 3.439, 3.562, 3.827, 4.305, 5.015],
    "antinue": [2.883, 2.791, 2.706, 2.626, 2.550, 2.434, 2.325, 2.179],
}
BS_HONDA = {"numu": 2.507, "antinumu": 3.812, "nue": 4.740, "antinue": 2.117}
BS_RATIO = {r"$\nu_e/\bar\nu_e$": ([1.077, 1.150, 1.228, 1.309, 1.397, 1.572,
                                    1.851, 2.302], 2.239, AQUA),
            r"$\bar\nu_\mu/\nu_\mu$": ([1.001, 1.035, 1.070, 1.106, 1.144,
                                        1.220, 1.362, 1.480], 1.520, ORANGE)}


def fig09():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.0, 3.0))
    for s in SP4:
        ax.plot(BS_SCALE, BS[s], color=SPEC[s], lw=LW, marker="o",
                ms=MS - 1.4, label=SPEC_LBL[s])
        ax.axhline(BS_HONDA[s], color=SPEC[s], lw=LWR, ls=(0, (4, 2)),
                   alpha=0.85)
        ax.text(8.25, BS_HONDA[s], f" {SPEC_LBL[s]}", color=SPEC[s],
                fontsize=6.8, va="center")
    ax.axvline(1.0, color=INK2, lw=0.9, ls=":")
    ax.text(1.15, 2.16, "physical\nscale", fontsize=6.5, color=INK2,
            linespacing=1.3, va="bottom")
    style(ax, "muon-bending scale factor", "W/E  (max/min over azimuth)",
          "a  W/E per species vs bend scale")
    ax.set_xlim(-0.3, 9.6)
    ax.legend(loc="upper left", borderpad=0.2, ncol=2, columnspacing=1.0)
    ax.text(0.03, 0.74, "dashed = Honda", transform=ax.transAxes,
            fontsize=6.5, color=INK2)

    for lbl, (y, hon, col) in BS_RATIO.items():
        bx.plot(BS_SCALE, y, color=col, lw=LW, marker="o", ms=MS - 1.4,
                label=lbl)
        bx.axhline(hon, color=col, lw=LWR, ls=(0, (4, 2)), alpha=0.85)
    bx.axvline(1.0, color=INK2, lw=0.9, ls=":")
    style(bx, "muon-bending scale factor", "ratio of the two W/E amplitudes",
          "b  Charge splitting vs bend scale")
    bx.set_xlim(-0.3, 8.4)
    bx.legend(loc="upper left", borderpad=0.2)
    bx.text(0.97, 0.05, "dashed = Honda\nreaching Honda needs a\n"
                        r"$\sim$5-8$\times$ bend", transform=bx.transAxes,
            ha="right", va="bottom", fontsize=6.5, color=INK2,
            linespacing=1.35)
    fig.suptitle(r"Bending-scale response, $\cos Z$ = 0.05, $E_\nu$ = 0.5 GeV",
                 x=0.012, y=0.995, ha="left", fontsize=9, color=INK)
    note(fig, "scratchpad/pattern/channels_tables.txt, block 'cz 005' "
              "sections C (diag_ew_pattern.py --stage channels)")
    save(fig, "09_bend_scale.png")


# ===========================================================================
# 10  muon_segment
# ===========================================================================
# scratchpad/muonseg.log, section B (prod_dir = ray), zenith 87, E_nu = 0.5.
MS_STAT = {"E": dict(mean=4.76, rms=4.74, p16=0.84, p50=3.29, p84=8.64),
           "W": dict(mean=4.92, rms=4.92, p16=0.87, p50=3.40, p84=8.92)}
DELIVERED_SHIFT = 5.08          # deg, Delta = q B tau / m (muonseg.log line 6)


def fig10():
    d = np.load(os.path.join(DATA, "muonseg_bend.npz"))
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.9), sharey=True)
    bins = np.linspace(0, 22, 89)
    xc = 0.5 * (bins[1:] + bins[:-1])
    for ax, az, col in zip(axes, ("E", "W"), (C_NEW, C_ALT)):
        b = np.asarray(d[f"bend_87_{az}_m"], float)      # mu- (nu_mu daughter)
        ax.hist(b, bins=bins, density=True, color=col, alpha=0.30, lw=0)
        h, _ = np.histogram(b, bins=bins, density=True)
        ax.step(xc, h, where="mid", color=col, lw=LW,
                label="MC, 30k samples")
        m = MS_STAT[az]["mean"]
        ax.plot(xc, np.exp(-xc / m) / m, color=C_TRUTH, lw=LWR, ls="--",
                label=f"exponential, {m:.2f}$^\\circ$")
        ax.axvline(DELIVERED_SHIFT, color=C_OLD, lw=LW, ls=":")
        ax.text(DELIVERED_SHIFT + 0.4, 0.128,
                f"delivered single\nshift {DELIVERED_SHIFT:.2f}$^\\circ$",
                fontsize=6.6, color=C_OLD, linespacing=1.3, va="top")
        for q in ("p16", "p50", "p84"):
            ax.axvline(MS_STAT[az][q], color=MUTED, lw=0.7, ymax=0.12)
        ax.text(0.97, 0.62,
                "p16 / p50 / p84\n"
                f"{MS_STAT[az]['p16']:.2f} / {MS_STAT[az]['p50']:.2f} / "
                f"{MS_STAT[az]['p84']:.2f}$^\\circ$",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
                color=INK2, linespacing=1.3)
        style(ax, "bending angle at decay [deg]", None,
              f"87$^\\circ$, geomagnetic {'East' if az == 'E' else 'West'}")
        ax.set_xlim(0, 22)
        ax.legend(loc="upper right", borderpad=0.2)
    axes[0].set_ylabel("probability density [1/deg]")
    fig.suptitle(r"In-flight muon bending, backward MC ($\nu_\mu$ daughter, "
                 r"$E_\nu$ = 0.5 GeV)", x=0.012, y=0.995, ha="left",
                 fontsize=9, color=INK)
    note(fig, "samples: scratchpad/msgrid_ray_n30000_s3.npz (muon_segment_mc, "
              "continuous dE/dx + IGRF-13)  ·  statistics and the 5.08 deg "
              "shift: scratchpad/muonseg.log section B")
    save(fig, "10_muon_segment.png")


# ===========================================================================
# 11  anchor_pattern
# ===========================================================================
# 24 azimuth nodes, 0..345 deg (compass).  Rows transcribed from
# scratchpad/dipole/prodpoint.log ('det') and scratchpad/anchor/validate.log
# ('family' = engine, prod_point-anchored; 'direct30' = direct back-trace at
# h_prod = 30 km, the published table).
AZ24 = np.arange(24) * 15.0
RC24 = {
    (80, "det"): [21.3, 25.4, 29.0, 31.6, 33.2, 35.5, 35.2, 32.5, 27.0, 18.7,
                  12.9, 10.7, 9.8, 8.8, 8.3, 7.9, 7.9, 8.5, 7.5, 7.2, 8.5,
                  9.2, 12.9, 16.9],
    (80, "family"): [19.7, 23.6, 27.0, 29.4, 31.6, 33.6, 34.0, 31.8, 27.2,
                     20.3, 14.9, 12.0, 10.5, 9.5, 8.8, 8.7, 8.4, 8.4, 7.5,
                     7.2, 8.0, 8.7, 11.8, 15.7],
    (80, "direct"): [19.5, 23.3, 26.7, 29.1, 30.9, 33.5, 33.7, 31.7, 27.2,
                     20.6, 14.7, 11.6, 10.3, 9.5, 8.5, 8.2, 9.1, 8.3, 7.2,
                     7.5, 7.4, 8.9, 11.7, 15.5],
    (87, "det"): [26.8, 32.4, 37.5, 41.5, 43.8, 44.1, 41.7, 38.4, 31.6, 19.0,
                  12.3, 10.6, 9.8, 9.4, 8.5, 7.7, 8.9, 7.7, 7.5, 8.0, 8.8,
                  11.6, 16.1, 21.2],
    (87, "family"): [22.6, 27.3, 31.5, 34.7, 36.4, 37.1, 37.8, 36.0, 31.3,
                     24.2, 17.8, 13.7, 11.4, 10.3, 9.5, 8.5, 9.1, 8.0, 7.2,
                     7.4, 7.6, 9.6, 13.4, 17.9],
    (87, "direct"): [22.5, 27.2, 31.3, 34.5, 36.3, 36.7, 37.6, 36.1, 31.8,
                     24.9, 17.5, 13.0, 11.0, 10.1, 9.7, 9.5, 9.4, 8.5, 7.3,
                     7.7, 7.2, 9.4, 13.4, 17.7],
}
NS_PHASE = {80: [(1.83, 66.89), (1.55, 71.96), (1.58, 72.57)],
            87: [(2.32, 61.74), (1.59, 71.98), (1.65, 72.43)]}


def fig11():
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.1), sharey=True)
    for ax, zen in zip(axes, (80, 87)):
        ax.axvline(EW_AXIS, color=GRID, lw=1.4, zorder=0)
        ax.plot(AZ24, RC24[(zen, "det")], color=C_OLD, lw=LW, ls="--",
                label="detector-anchored (old default)")
        ax.plot(AZ24, RC24[(zen, "family")], color=C_NEW, lw=LW,
                label="production-point anchored (engine)")
        ax.plot(AZ24, RC24[(zen, "direct")], color=C_TRUTH, lw=0, marker="+",
                ms=MS + 2, mew=1.0,
                label="direct back-trace, $h_{\\rm prod}$ = 30 km")
        ax.set_xticks([0, 90, 180, 270])
        ax.set_xticklabels(["N", "E", "S", "W"])
        style(ax, "arrival azimuth (compass)", None,
              f"zenith {zen}$^\\circ$")
        ax.set_xlim(-8, 352)
        ax.set_ylim(2.5, 47.0)
        ax.text(EW_AXIS + 4, 3.6, "geomagnetic E", fontsize=6.3, color=MUTED)
        rows = NS_PHASE[zen]
        ax.text(0.97, 0.97,
                "          N/S   phase\n"
                f"detector {rows[0][0]:.2f}  {rows[0][1]:.1f}$^\\circ$\n"
                f"engine   {rows[1][0]:.2f}  {rows[1][1]:.1f}$^\\circ$\n"
                f"direct   {rows[2][0]:.2f}  {rows[2][1]:.1f}$^\\circ$",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.2,
                color=INK2, linespacing=1.3, family="monospace")
    axes[0].set_ylabel(r"$R_c$ [GV]")
    fig.subplots_adjust(top=0.80)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left",
               bbox_to_anchor=(0.06, 0.975), ncol=3, columnspacing=1.4,
               borderpad=0.2)
    fig.suptitle("Cutoff vs azimuth: where the back-trace is launched",
                 x=0.012, y=1.03, ha="left", fontsize=9, color=INK)
    note(fig, "detector: scratchpad/dipole/prodpoint.log  ·  engine + direct: "
              "scratchpad/anchor/validate.log section 3 "
              "(diag_cutoff_anchor.py)  ·  N/S and phase read at the "
              "geomagnetic cardinals")
    save(fig, "11_anchor_pattern.png")


# ===========================================================================
# 12  gridwide      (the ONE figure that needs a live solve)
# ===========================================================================
def fig12():
    p = os.path.join(DATA, "grid_delivered.npz")
    if not os.path.exists(p):
        print("SKIP 12: run figures_phase1/data/solve_grid.py first")
        return
    g = np.load(p)
    h = dict(np.load(os.path.join(HERE, "honda_kam.npz")))
    e, cz = g["e"], g["cz"]
    icz0 = int(np.argmin(np.abs(h["czlo"] - 0.0)))
    Eg = np.logspace(np.log10(0.1), np.log10(100.0), 60)

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.8), sharey=True)
    ims = []
    for ax, sp in zip(axes, ("numu", "nue")):
        M = np.zeros((len(cz), len(Eg)))
        cell = []
        for i in range(len(cz)):
            ours = logi(Eg, e, g[f"total_{sp}"][i].mean(0))
            hon = logi(Eg, h["E"], h[HK[sp]][icz0 + i].mean(0))
            M[i] = np.log10(ours / hon)
            for j in range(12):                      # per-cell, all azimuths
                cell.append(np.log10(logi(Eg, e, g[f"total_{sp}"][i, j])
                                     / logi(Eg, h["E"], h[HK[sp]][icz0 + i, j])))
        cell = np.abs(np.array(cell))
        im = ax.pcolormesh(Eg, cz, M, cmap=DIVCMAP, vmin=-0.08, vmax=0.08,
                           shading="nearest", rasterized=True)
        ims.append(im)
        ax.set_xscale("log")
        style(ax, r"$E_\nu$ [GeV]", None, SPEC_LBL[sp], grid=None)
        ax.text(0.035, 0.05,
                "per cell, all 12 az:\n"
                f"median {np.median(cell):.3f}\n"
                f"90th pct {np.percentile(cell, 90):.3f}",
                transform=ax.transAxes, fontsize=6.4, color=INK, ha="left",
                va="bottom", linespacing=1.3,
                bbox=dict(fc=SURFACE, ec="none", alpha=0.88, pad=2.0))
    axes[0].set_ylabel(r"$\cos Z$")
    cb = fig.colorbar(ims[0], ax=axes, fraction=0.045, pad=0.02,
                      extend="both", ticks=[-0.08, -0.04, 0, 0.04, 0.08])
    cb.set_label(r"$\log_{10}$(model / Honda), azimuth-averaged", fontsize=7)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5, color=MUTED, labelcolor=INK2)
    fig.suptitle("Grid-wide agreement, delivered engine on Honda's own bins",
                 x=0.012, y=1.02, ha="left", fontsize=9, color=INK)
    note(fig, "live solve: figures_phase1/data/solve_grid.py (bare "
              "MCEq3DFlux.solve, 10 cosZ x 12 az, Kamioka, 2020-01-01) · "
              "reference honda_kam.npz · the per-cell numbers reproduce "
              "scratchpad/anchor/full_after.log section G (0.032/0.061 numu, "
              "0.032/0.090 nue) · the 0.1 GeV column is the engine grid edge",
         dy=-0.14)
    save(fig, "12_gridwide.png")


# ===========================================================================
# 13  conservation
# ===========================================================================
# scratchpad/jc/stages_s4.log, "Solid-angle average of the joint production
# factor at G == 1 (down-going hemisphere)".
CONS_E = [0.20, 0.30, 0.50, 1.00, 3.00, 10.00]
CONS = {"numu": [1.0652, 1.0458, 1.0242, 1.0084, 1.0013, 1.0001],
        "antinumu": [1.0665, 1.0478, 1.0250, 1.0077, 1.0009, 1.0000],
        "nue": [1.0686, 1.0521, 1.0287, 1.0091, 1.0009, 0.9996],
        "antinue": [1.0692, 1.0525, 1.0288, 1.0090, 1.0008, 0.9996]}


def fig13():
    fig, ax = plt.subplots(figsize=(6.0, 3.1))
    ax.axhspan(0.98, 1.03, color=GRID, alpha=0.55, lw=0, zorder=0)
    ax.axhline(1.0, color=AXIS, lw=1.0, zorder=1)
    for y, lbl, va in ((1.03, "Bartol  +3%", "bottom"),
                       (0.98, "Honda  -2%", "top")):
        ax.axhline(y, color=INK2, lw=LWR, ls=(0, (4, 2)), zorder=2)
        ax.text(15.5, y, lbl, fontsize=6.8, color=INK2, va=va, ha="right")
    for sp, mk in zip(SP4, ("o", "s", "^", "D")):
        ax.plot(CONS_E, CONS[sp], color=SPEC[sp], lw=LW, marker=mk,
                ms=MS - 0.6, mfc=SURFACE, mew=1.2, label=SPEC_LBL[sp],
                zorder=4)
    ax.set_xscale("log")
    style(ax, r"$E_\nu$ [GeV]",
          r"$\langle\,$production factor$\,\rangle_{\Omega}$  at $G\equiv 1$",
          "Down-going solid-angle average of the joint production factor",
          grid="both")
    ax.set_xlim(0.17, 22)
    ax.set_ylim(0.975, 1.078)
    ax.legend(loc="upper right", borderpad=0.25, ncol=2, columnspacing=1.0,
              bbox_to_anchor=(1.0, 0.80))
    ax.text(1.35, 1.072,
            "a pure redistribution would sit on 1.000;\n"
            "the delivered cone adds a net 4.6-6.9%\nbelow 0.3 GeV, and the "
            "four species agree\nto better than 0.5%",
            fontsize=6.8, color=INK2, va="top", linespacing=1.35)
    note(fig, "scratchpad/jc/stages_s4.log, conservation table "
              "(diag_joint_stages.py, S4 configuration)")
    save(fig, "13_conservation.png")


# ---------------------------------------------------------------------------
FIGS = {"01": fig01, "02": fig02, "03": fig03, "04": fig04, "05": fig05,
        "06": fig06, "07": fig07, "08": fig08, "09": fig09, "10": fig10,
        "11": fig11, "12": fig12, "13": fig13}


def main(argv):
    os.makedirs(OUT, exist_ok=True)
    keys = argv[1:] or sorted(FIGS)
    for k in keys:
        FIGS[k]()


if __name__ == "__main__":
    main(sys.argv)
