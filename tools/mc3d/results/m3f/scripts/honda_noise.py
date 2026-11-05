"""Statistical noise of Honda's kam-ally-20-12-solmin table in our 12 cells.

Two independent estimates of the relative per-bin noise sigma(species, cos Z bin):
 (1) E-roughness: the table has 101 energy points (20/decade); for each (cos Z, az)
     series of log flux the second difference d2 = y[i-1] - 2 y[i] + y[i+1] has
     variance 6 sigma^2 for independent noise (smooth physics contributes a small
     curvature term, so this is an upper bound); sigma = 1.4826 MAD(d2)/sqrt(6),
     over 0.2-3 GeV.
 (2) az-roughness: the azimuth pattern of a cutoff-limited flux is smooth; its
     harmonics k = 5, 6 (and the Nyquist term) at fixed (cos Z, E) are ~noise:
     sigma^2 ~ mean power in harmonics 5-6 (12 bins -> 2 harmonics + Nyquist).
Then 300 Monte-Carlo copies of the table with independent Gaussian noise (log-normal)
of the estimated sigma are pushed through ``analyse_m3b.honda_patterns`` /
``observables`` (same band integration), giving the noise on the 12-cell observables.
Output: $M/reports/honda_noise.txt and honda_noise.npz (est, err (4,3,5)).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "m3c", "scripts"))
import analyse_m3b as a
import m3c_lib as L

M = os.environ["M"]; R = os.environ["R"]
path = R + "/tools/mceq3d/honda_kam.npz"
h = dict(np.load(path))
E = h["E"]
keys = [a.HONDA_KEY[s] for s in a.SP_NAMES]
sel = (E > 0.2) & (E < 3.0)
rng = np.random.default_rng(5)
lines = ["# Honda table noise (relative, log flux)", "species  cosZ_lo   sigma(E-roughness)  sigma(az-harmonics 5-6)"]
sig = {}
for k, sp in zip(keys, a.SP_NAMES):
    t = np.log(np.maximum(h[k], 1e-300))                    # (20, 12, 101)
    d2 = t[..., :-2] - 2 * t[..., 1:-1] + t[..., 2:]
    Ec = E[1:-1]
    m = (Ec > 0.2) & (Ec < 3.0)
    for icz in range(10, 13):                               # cos Z 0-0.3
        x = d2[icz][:, m].ravel()
        s1 = 1.4826 * np.median(abs(x - np.median(x))) / np.sqrt(6)
        f = np.fft.rfft(t[icz][:, sel], axis=0)             # (7, nE)
        p = (abs(f[5]) ** 2 + abs(f[6]) ** 2 / 2) / 12.0 ** 2 * 4   # power in the two highest harmonics
        s2 = np.sqrt(np.mean(p) / 3.0)
        sig[(sp, icz)] = s1
        lines.append(f"{sp:9s} {h['czlo'][icz]:+.1f}      {s1:.4f}              {s2:.4f}")
s_all = np.array([sig[k] for k in sig])
lines.append(f"E-roughness sigma: median {np.median(s_all):.4f}, range {s_all.min():.4f}-{s_all.max():.4f}")
# Monte Carlo through the same machinery
base = a.honda_patterns(path).sum(2)
def obs(pats):
    return np.array([[a.observables(pats[k, ib], L.axis())[[0, 1, 5, 6, 7]] for ib in range(3)] for k in range(4)])
o0 = obs(base)
draws = []
for _ in range(300):
    hh = dict(h)
    for k, sp in zip(keys, a.SP_NAMES):
        n = np.ones_like(h[k])
        for icz in range(10, 13):
            n[icz] = np.exp(rng.normal(0, sig[(sp, icz)], h[k][icz].shape))
        hh[k] = h[k] * n
    np.savez(M + "/work/_honda_tmp.npz", **hh)
    draws.append(obs(a.honda_patterns(M + "/work/_honda_tmp.npz").sum(2)))
draws = np.array(draws)
dev = draws - o0
dev[..., 1] = (dev[..., 1] + 180) % 360 - 180
err = dev.std(0)
names = ("a1/a0", "dphi", "fN", "fS", "N/S")
for j, nm in enumerate(names):
    lines.append(f"honda noise on {nm:6s}: median {np.median(err[..., j]):.4f}, max {err[..., j].max():.4f}   (cells: " + " ".join(f"{x:.3f}" for x in err[..., j].ravel()) + ")")
lines.append("mean a1/a0 noise over 12 cells (independent bins): %.4f" % (np.sqrt((err[..., 0] ** 2).sum()) / 12))
open(M + "/reports/honda_noise.txt", "w").write("\n".join(lines) + "\n")
np.savez(M + "/reports/honda_noise.npz", est=o0, err=err, sigma=np.array(list(sig.values())))
print("\n".join(lines))
os.remove(M + "/work/_honda_tmp.npz")
