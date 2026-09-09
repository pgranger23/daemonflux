"""Demonstrate the 3D / geomagnetic extension of daemonflux.

Produces a two-panel figure:

* left  -- directional Stoermer cutoff rigidity vs azimuth (the East--West
           asymmetry) for a few zenith angles at Kamioka;
* right -- the geomagnetic admittance G(E) from the East vs from the West,
           showing the suppression of the low-energy flux and the convergence
           to the 1D result (G -> 1) above the cutoff.

The admittance physics needs no flux tables, so this script runs offline. The
commented block at the end shows how to apply it to a real ``Flux`` object.

Run:  python examples/geomagnetic_example.py
"""

import numpy as np
import matplotlib.pyplot as plt

from daemonflux.geomagnetic import GeomagneticModel

model = GeomagneticModel("kamioka")
print(model)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

# --- Left: cutoff rigidity vs azimuth -------------------------------------
azimuth = np.linspace(0, 360, 361)
for zen in (30.0, 60.0, 85.0):
    rc = model.cutoff_rigidity_GV(zen, azimuth)
    axL.plot(azimuth, rc, label=f"zenith = {zen:.0f}$^\\circ$")
for a, name in [(90, "E"), (180, "S"), (270, "W"), (0, "N")]:
    axL.axvline(a, color="gray", ls=":", lw=0.7)
axL.set_xlabel("geographic azimuth [deg]  (N=0, E=90, S=180, W=270)")
axL.set_ylabel("Stoermer cutoff rigidity [GV]")
axL.set_title("East--West cutoff asymmetry (Kamioka)")
axL.set_xticks([0, 90, 180, 270, 360])
axL.legend()

# --- Right: admittance from East vs West ----------------------------------
E = np.logspace(-1, 2.5, 200)  # 0.1 - ~300 GeV
g_east = model.admittance("numuflux", E, 85.0, 90.0)
g_west = model.admittance("numuflux", E, 85.0, 270.0)
g_avg = model.admittance("numuflux", E, 85.0, None)
axR.semilogx(E, g_west, label="from West")
axR.semilogx(E, g_east, label="from East")
axR.semilogx(E, g_avg, "k--", label="azimuth-averaged")
axR.axhline(1.0, color="gray", ls=":", lw=0.7)
axR.set_xlabel("lepton energy [GeV]")
axR.set_ylabel("geomagnetic admittance  G")
axR.set_title("Low-energy suppression (zenith 85$^\\circ$)")
axR.set_ylim(0, 1.05)
axR.legend()

fig.tight_layout()
out = "geomagnetic_example.png"
fig.savefig(out, dpi=110)
print("saved", out)

# --- How to use with a real Flux object -----------------------------------
# from daemonflux import Flux
# flux = Flux(location="generic", geomag_location="kamioka")
# egrid = np.logspace(-0.5, 2, 50)        # 0.3 - 100 GeV
# fE = flux.flux(egrid, 70.0, "numuflux", azimuth_deg=90.0)    # from East
# fW = flux.flux(egrid, 70.0, "numuflux", azimuth_deg=270.0)   # from West
# # fW / fE > 1 at low energy is the East--West effect.
