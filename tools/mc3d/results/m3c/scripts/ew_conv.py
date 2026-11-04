"""The pilot's East-West table under the corrected H3a energy convention
(``reweight_replay.py`` -> ``pilot_tot``) against the original (``pilot``):
pooled cos Z 0-0.3 ``a1/a0`` per species and band, paired jackknife errors of
the difference, chi2/12 against Honda for both, and the fan-out differences.

    python ew_conv.py [TAG TAG_TOT] -> $M/reports/ew_conv.txt
"""
import os
import sys

import numpy as np

import analyse_m3b as a
import m3c_lib as L

M = os.environ["M"]
t0, t1 = (sys.argv[1:3] if len(sys.argv) > 2 else ("pilot", "pilot_tot"))
A, B = L.Run(t0), L.Run(t1)
oa, ob = A.obs("s", True), B.obs("s", True)
d, de = L.diff(ob, oa)
ea, eb = L.jk_err(oa[1]), L.jk_err(ob[1])
hp = a.honda_patterns().sum(2)
ho = np.array([[L.obs_pattern(hp[k, ib]) for ib in range(3)] for k in range(4)])
lines = [f"# {t1} (crflux at the total energy) vs {t0} (at the kinetic energy): "
         "pooled cos Z 0-0.3 a1/a0, paired jackknife of the difference"]
for ib, E in enumerate(L.EB):
    lines.append(f"E {E}: " + "  ".join(
        f"{sp} {oa[0][k, ib, 0]:.3f}+-{ea[k, ib, 0]:.3f} -> {ob[0][k, ib, 0]:.3f}+-"
        f"{eb[k, ib, 0]:.3f} (d {d[k, ib, 0]:+.3f}+-{de[k, ib, 0]:.3f}; Honda "
        f"{ho[k, ib, 0]:.3f})" for k, sp in enumerate(L.SP)))
for lab, o, e in ((t0, oa[0], ea), (t1, ob[0], eb)):
    z = (o[..., 0] - ho[..., 0]) / e[..., 0]
    lines.append(f"chi2/12 vs Honda (MC errors): {lab} {np.sum(z ** 2) / 12:.2f}")
lines.append(f"mean d(a1/a0) {np.mean(d[..., 0]):+.4f}; chi2(d=0)/12 "
             f"{np.sum((d[..., 0] / de[..., 0]) ** 2) / 12:.2f}")
for (i, j) in ((0, 3), (3, 2), (2, 1)):
    x0 = oa[0][i, :, 0] - oa[0][j, :, 0]
    x1 = ob[0][i, :, 0] - ob[0][j, :, 0]
    lines.append(f"fan-out {L.SP[i]}-{L.SP[j]}: " + " ".join(
        f"{u:+.3f}->{v:+.3f}" for u, v in zip(x0, x1)) + "  (Honda "
        + " ".join(f"{h:+.3f}" for h in ho[i, :, 0] - ho[j, :, 0]) + ")")
# azimuth-averaged 3D flux ratio new/old, cos Z 0-0.3 (the absolute effect)
r = B.tot[0].sum((2, 3)) / A.tot[0].sum((2, 3))
lines.append("3D flux (s, cos Z 0-0.3, az-summed) new/old: " + "  ".join(
    f"{sp} " + " ".join(f"{x:.3f}" for x in r[k]) for k, sp in enumerate(L.SP)))
open(f"{M}/reports/ew_conv.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
