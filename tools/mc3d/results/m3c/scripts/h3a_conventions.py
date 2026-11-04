"""Gate C3 candidate (c): tools/mc3d's H3a nucleon flux and rigidity against
MCEq's / the engine's, per mass group, at 2-30 GeV/nucleon (crflux only).
    python h3a_conventions.py -> $M/reports/h3a_conventions.txt"""
import os
import numpy as np
import crflux.models as crf
import primaries as prim
import cutoff
M = os.environ["M"]
MP, MN = 0.938272, 0.939565
MBAR = 0.5 * (MP + MN)
h = crf.HillasGaisser2012("H3a")
mdl = prim.primary_model("H3A")
L = []
L.append("# H3a per group: nucleon flux MC (crflux at A*T, primaries.nucleon_intensity) / "
         "MCEq (crflux at A*(T+m), MCEqRun.set_primary_model); rigidity of the cut "
         "MC (A/Z)p (cutoff.rigidity_gv) vs engine (A/Z)T (mceq3d_flux._transmission)")
for T in (2.0, 3.0, 4.0, 6.0, 10.0, 15.0, 30.0, 100.0):
    row = [f"T={T:5.1f} GeV/n:"]
    for g, a, z, gid in prim.SPECIES:
        mc = prim.nucleon_intensity(mdl, gid, np.array([T]), a)[0]
        mq = 1e-4 * a * a * h.nucleus_flux(gid, a * (T + MBAR))[0]
        row.append(f"{g} flux {mc / mq:.3f} R {cutoff.rigidity_gv(T, a, z):.2f}/{a / z * T:.2f}")
    L.append("  ".join(row))
# the rigidity at which each convention cuts, for the site values
L.append("\n# kinetic energy per nucleon cut at a nucleus rigidity R_c: MC (true p) vs engine (R = (A/Z) T)")
for rc in (7.5, 11.5, 42.0):
    for aoz, lab in ((1.0, "A/Z=1"), (2.0, "A/Z=2"), (56 / 26, "Fe")):
        p = rc / aoz
        L.append(f"R_c {rc:5.1f} GV {lab:6s}: MC cuts T < {np.sqrt(p * p + MP * MP) - MP:6.2f}, "
                 f"engine cuts T < {rc / aoz:6.2f} GeV/n")
open(M + "/reports/h3a_conventions.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
