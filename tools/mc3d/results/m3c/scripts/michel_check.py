"""Decay-pipeline check: bend.michel_decay (field-on pipeline) against
decays.michel_sample (the cascade's own decays): neutrino energy fractions in
the analysis bands and mean cos to the muon direction, per flavour, for
mu+-/mu- at several energies and polarisations."""
import math
import numpy as np
import bend, decays
from constants import M_MU
rng = np.random.default_rng(11)
bands = ((0.2239, 0.3981), (0.3981, 0.631), (0.7943, 1.2589))
n_v, n_s = 400000, 60000
print("E_mu pol pdg flavour: band fractions vectorised | scalar (rel. diff, sigma)")
worst = 0.0
for E in (0.6, 1.0, 2.0, 4.0):
    for pdg, pol in ((-13, -1.0), (-13, -0.3), (13, 0.3), (13, 1.0)):
        u = np.tile([0, 0, 1.0], (n_v, 1))
        pv, ev, dv = bend.michel_decay(np.full(n_v, pdg), np.full(n_v, E), u,
                                       np.full(n_v, pol), rng.random((n_v, 6)),
                                       np.array([1.0, 0, 0]))
        p = math.sqrt(E * E - M_MU * M_MU)
        p4 = np.array([E, 0, 0, p])
        ch = +1 if pdg < 0 else -1
        for j, kd in enumerate(("numu", "nue")):
            q = np.array([decays.michel_sample(rng, p4, ch, pol, kd) for _ in range(n_s)])
            es = q[:, 0]
            row = []
            for lo, hi in bands:
                fv = np.mean((ev[:, j] >= lo) & (ev[:, j] < hi))
                fs = np.mean((es >= lo) & (es < hi))
                sig = math.sqrt(fs * (1 - fs) / n_s + fv * (1 - fv) / n_v) or 1e-9
                row.append(f"{fv:.4f}|{fs:.4f} ({(fv / fs - 1) if fs else 0:+.3f}, {(fv - fs) / sig:+.1f})")
                worst = max(worst, abs(fv - fs) / sig)
            print(f"{E} {pol:+.1f} {pdg} {kd}: " + "  ".join(row))
print("largest pull", round(worst, 1))
