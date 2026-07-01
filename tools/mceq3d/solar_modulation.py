"""Solar-modulation knob: force-field modulation of the primary -> flux factor.

The 11-year solar cycle modulates the sub-GeV primary cosmic-ray flux (higher solar
activity -> stronger heliospheric field -> lower flux at Earth). This is a real
~10-20% sub-GeV effect that Bartol and Honda model and that a fixed primary omits.

`MCEq3DFlux.solar_factor(phi)` applies the Gleeson-Axford force-field modulation
(potential phi [GV]) to the primary and returns the resulting neutrino-energy
factor S(E) = numu(modulated)/numu(baseline); `solve(solar_modulation=phi)` folds
it in. phi is *relative to the H3a baseline* (0 = baseline); the physical
solar-cycle effect is the difference between two potentials -- we use ~0.4 GV
(solar min) and ~1.0 GV (solar max), whose ratio is the solar-cycle band. Run::

    python solar_modulation.py --plot
"""

from __future__ import annotations

import argparse

import numpy as np

from mceq3d_flux import MCEq3DFlux

PHI = {"solar-min (0.4 GV)": 0.4, "mid (0.7 GV)": 0.7, "solar-max (1.0 GV)": 1.0}
EREPORT = (0.1, 0.2, 0.3, 0.5, 1.0, 3.0, 10.0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args(argv)

    eng = MCEq3DFlux(base_model="mceq")
    e = eng.e
    S = {k: eng.solar_factor(v) for k, v in PHI.items()}

    def at(y, E):
        return float(np.interp(E, e, y))

    print("Solar-modulation factor S(E) = numu(phi)/numu(baseline), vertical:")
    print("  E[GeV]  " + "".join(f"{k.split()[0]:>13s}" for k in PHI))
    for E in EREPORT:
        print(f"  {E:6.2f} " + "".join(f"{at(S[k], E):13.3f}" for k in PHI))

    smin = S["solar-min (0.4 GV)"]
    smax = S["solar-max (1.0 GV)"]
    span = smin / smax  # solar-cycle amplitude (min flux / max flux)
    print("\nSolar-cycle span  S(min)/S(max)  [= flux(solar-min)/flux(solar-max)]:")
    for E in EREPORT:
        print(f"  {E:6.2f}   {at(span, E):.3f}  ({(at(span, E) - 1) * 100:+.0f}%)")
    print("(Honda/Bartol report ~10-20% sub-GeV solar-cycle variation.)")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sel = (e >= 0.1) & (e <= 100)
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))
        for k in PHI:
            axL.semilogx(e[sel], S[k][sel], label=k)
        axL.axhline(1.0, color="k", lw=0.6)
        axL.set_xlabel("E [GeV]")
        axL.set_ylabel(r"$S(E)=\Phi_{\nu_\mu}(\phi)/\Phi_{\nu_\mu}$(baseline)")
        axL.set_title("Solar-modulation factor (force-field on primary)")
        axL.legend(fontsize=8)
        axR.fill_between(
            e[sel],
            smax[sel],
            smin[sel],
            color="C1",
            alpha=0.3,
            label="solar min–max band",
        )
        axR.semilogx(e[sel], (smin / smax)[sel], "C3-", label="flux(min)/flux(max)")
        axR.axhline(1.0, color="k", lw=0.6)
        axR.set_xlabel("E [GeV]")
        axR.set_ylabel("solar-cycle flux ratio")
        axR.set_title("Solar-cycle amplitude vs energy")
        axR.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig("solar_modulation.png", dpi=110)
        print("saved plot -> solar_modulation.png")


if __name__ == "__main__":
    main()
