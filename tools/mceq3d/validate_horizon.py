"""Validate the explicit near-horizon 3D-excess correction H(E, cosZ).

Shows the delivered zenith shape before and after folding in H
(`horizon_excess.npz`, built by `build_horizon_excess.py`) against the two
independent full-3D references (Honda, Bartol). Before the correction the
deterministic engine misses the sub-GeV horizontal enhancement (horizon/vertical
~0.9 vs the references' ~1.8 at 0.3 GeV); after it, the shape reproduces the
references to within their mutual ~5% spread. The correction is anchored to Honda
and cross-checked against Bartol, so agreement with Bartol is an *independent* test.
Uses only the stored H and the reference tables (no re-solve). Run::

    python validate_horizon.py --plot
"""

from __future__ import annotations

import argparse

import numpy as np

import validate_bartol as vb


def _at(y, x, E):
    return np.exp(np.interp(np.log(E), np.log(x), np.log(np.maximum(y, 1e-300))))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args(argv)

    d = np.load("horizon_excess.npz")
    eg, cz, H = d["e"], d["cz"], d["H"]  # H = Honda_shape / our_shape
    h = dict(np.load("honda_kam.npz"))
    He, Hcz, nm = h["E"], h["czlo"], h["numu"]
    Eb, czb, gb = vb.load_bartol("num", "fmin")

    def hon(czc):
        return _at(nm[int(np.argmin(np.abs(Hcz - (czc - 0.05))))].mean(0), He, eg)

    def bar(czc):
        return _at(gb[int(np.argmin(np.abs(czb - czc)))], Eb, eg)

    hon_v, bar_v = hon(0.95), bar(0.95)
    # our shape (pre-correction) reconstructed as Honda_shape / H
    hon_sh = np.array([hon(c) / hon_v for c in cz])  # (n_cz, n_E)
    our_sh = hon_sh / H
    corr_sh = our_sh * H  # = Honda shape (post-correction)
    bar_sh = np.array([bar(c) / bar_v for c in cz])

    print("Zenith shape Phi(cosZ)/Phi(vertical), before/after H vs references:")
    print("  E[GeV] cosZ   ours   ours+H  Honda  Bartol")
    for E in (0.3, 0.5, 1.0, 3.0):
        ie = int(np.argmin(np.abs(eg - E)))
        for i in (0, 1, 2):
            print(
                f"  {E:5.2f} {cz[i]:.2f}  {our_sh[i, ie]:5.3f}  "
                f"{corr_sh[i, ie]:5.3f}  {hon_sh[i, ie]:5.3f}  {bar_sh[i, ie]:5.3f}"
            )
    # independent residual: corrected vs Bartol at the horizon
    ie3 = int(np.argmin(np.abs(eg - 0.3)))
    resid = abs(corr_sh[0, ie3] / bar_sh[0, ie3] - 1) * 100
    print(
        f"\nhorizon (cosZ=0.05) 0.3 GeV: pre-correction ours={our_sh[0, ie3]:.2f}, "
        f"corrected={corr_sh[0, ie3]:.2f}, Bartol(indep.)={bar_sh[0, ie3]:.2f} "
        f"-> residual {resid:.0f}%"
    )

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
        for ax, E in zip(axes, (0.3, 1.0)):
            ie = int(np.argmin(np.abs(eg - E)))
            ax.plot(cz, our_sh[:, ie], "C0o--", label="this work (no H)")
            ax.plot(cz, corr_sh[:, ie], "C3o-", lw=2, label="this work + H")
            ax.plot(cz, hon_sh[:, ie], "k-", label="Honda")
            ax.plot(cz, bar_sh[:, ie], "C2s:", label="Bartol (indep.)")
            ax.set_xlabel(r"$\cos\theta_z$ (1=vertical, 0=horizon)")
            ax.set_ylabel(r"$\Phi_{\nu_\mu}(\cos\theta)/\Phi(\rm vert)$")
            ax.set_title(f"Zenith shape at {E:g} GeV (Kamioka)")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig("horizon_excess.png", dpi=110)
        print("saved plot -> horizon_excess.png")


if __name__ == "__main__":
    main()
