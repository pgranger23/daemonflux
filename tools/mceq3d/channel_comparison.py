"""Compare the validated angular spread across production channels.

Loads the per-channel spliced moments files (pi+, pi-, K+, K-) produced by
``kernel_regeneration.py --moments`` + ``splice_kernels.py`` and plots the mean
production angle <theta>(E_sec). Kaons are heavier and carry harder p_T than
pions, so their angular behaviour differs -- relevant because kaons dominate the
neutrino flux at higher energies.

Run::

    python channel_comparison.py --plot
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from angular_kernel import pool_moments_by_energy

CHANNELS = {
    "piplus": ("m_spliced.npz", r"$\pi^+$", "C0"),
    "piminus": ("m_piminus.npz", r"$\pi^-$", "C1"),
    "Kplus": ("m_Kplus.npz", r"$K^+$", "C2"),
    "Kminus": ("m_Kminus.npz", r"$K^-$", "C3"),
}


def channel_curve(npz, e_edges):
    d = dict(np.load(npz))
    mom = {k: d[k] for k in ("e_sec", "theta_mean", "theta_sq", "dndx")}
    pooled = pool_moments_by_energy(mom, e_edges)
    return pooled["e_sec"], np.degrees(pooled["theta_mean"]), pooled["pt_eff"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args(argv)

    e_edges = np.logspace(np.log10(0.6), 3.5, 22)
    curves = {}
    for name, (npz, label, color) in CHANNELS.items():
        if not os.path.exists(npz):
            print(f"  (missing {npz}; skip {name})")
            continue
        e, th, pt = channel_curve(npz, e_edges)
        curves[name] = (e, th, pt, label, color)

    print("mean production angle <theta> [deg] by channel:")
    hdr = "  E[GeV] " + "".join(f"{CHANNELS[c][1]:>9}" for c in curves)
    print(hdr)
    any_e = next(iter(curves.values()))[0]
    for ie in range(0, len(any_e), 3):
        row = f"  {any_e[ie]:6.1f} "
        for c in curves:
            row += f"{curves[c][1][ie]:9.2f}"
        print(row)

    if args.plot:
        _plot(curves)


def _plot(curves):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))
    for name, (e, th, pt, label, color) in curves.items():
        g = np.isfinite(th)
        axL.loglog(e[g], th[g], "o-", ms=3, color=color, label=label)
        gp = np.isfinite(pt)
        axR.semilogx(e[gp], pt[gp], "o-", ms=3, color=color, label=label)
    axL.set_xlabel(r"secondary energy $E_{\rm sec}$ [GeV]")
    axL.set_ylabel(r"$\langle\theta\rangle$ [deg]")
    axL.set_title("Production angle by channel")
    axL.legend()
    axR.axhspan(0.30, 0.50, color="gray", alpha=0.12)
    axR.set_xlabel(r"secondary energy $E_{\rm sec}$ [GeV]")
    axR.set_ylabel(r"$\langle p_T\rangle_{\rm eff}$ [GeV]")
    axR.set_title(r"$\langle p_T\rangle$ by channel (K harder than $\pi$)")
    axR.set_ylim(0, 0.8)
    axR.legend()
    fig.tight_layout()
    fig.savefig("channel_comparison.png", dpi=110)
    print("saved plot -> channel_comparison.png")


if __name__ == "__main__":
    main()
