"""Approximate correction factor for the energy-guard bug (pool.draw, fixed in
milestone 3d): g(E_p, E_nu) = (1D neutrino yield per primary proton, fixed
guard) / (same, old guard), all four flavours summed, from the milestone-1
closure column (collinear, vertical; ``closure_floor.py`` with and without
``GUARD_MASS=0.94``), smoothed over E_nu (3 bins) and interpolated in log E_p.

    python build_gfix.py OUT.npz OLD1.npz NEW1.npz [OLD2.npz NEW2.npz ...]

Output ``ep`` (primary kinetic energies), ``ec`` (E_nu bin centres), ``g``
(len(ep), len(ec)), plus the raw ratio and its error."""
import sys
import numpy as np

out = sys.argv[1]
pairs = list(zip(sys.argv[2::2], sys.argv[3::2]))
SP = (14, -14, 12, -12)
acc = {}
for fo, fn in pairs:
    o, n = np.load(fo), np.load(fn)
    eb = o["e_bins"]
    for k in o.files:
        if k.startswith("mc_") and k.endswith("_14"):
            ep = float(k.split("_")[1])
            so = sum(o[f"mc_{ep}_{s}"] for s in SP)
            sn = sum(n[f"mc_{ep}_{s}"] for s in SP)
            eo = np.sqrt(sum(o[f"err_{ep}_{s}"] ** 2 for s in SP))
            en = np.sqrt(sum(n[f"err_{ep}_{s}"] ** 2 for s in SP))
            acc[ep] = (so, sn, eo, en)
eps = np.array(sorted(acc))
ec = np.sqrt(eb[1:] * eb[:-1])
w = np.diff(eb)
# band ratios (sums over wide E_nu bands: statistically robust)
BE = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 100.0])
bc = np.sqrt(BE[1:] * BE[:-1])
braw = np.full((len(eps), len(bc)), np.nan)
berr = np.full_like(braw, np.nan)
for i, ep in enumerate(eps):
    so, sn, eo, en = acc[ep]
    for b in range(len(bc)):
        m = (ec >= BE[b]) & (ec < BE[b + 1]) & (ec < 0.6 * ep)   # kinematic reach
        if not m.any():
            continue
        a_, b_ = (so[m] * w[m]).sum(), (sn[m] * w[m]).sum()
        ea = np.sqrt(((eo[m] * w[m]) ** 2).sum())
        eb_ = np.sqrt(((en[m] * w[m]) ** 2).sum())
        if a_ > 8 * ea and b_ > 8 * eb_:
            braw[i, b] = b_ / a_
            berr[i, b] = braw[i, b] * np.sqrt((ea / a_) ** 2 + (eb_ / b_) ** 2)
# unmeasured bands: the nearest lower measured one (E_nu >> E_p/3 does not
# occur), then 3-point smoothing along E_p
gb = braw.copy()
for i in range(len(eps)):
    for b in range(len(bc)):
        if not np.isfinite(gb[i, b]):
            lower = [gb[i, k] for k in range(b - 1, -1, -1) if np.isfinite(gb[i, k])]
            gb[i, b] = lower[0] if lower else 1.0
gs = gb.copy()
for i in range(1, len(eps) - 1):
    gs[i] = 0.25 * gb[i - 1] + 0.5 * gb[i] + 0.25 * gb[i + 1]
g = np.stack([np.interp(np.log(ec), np.log(bc), gs[i]) for i in range(len(eps))])
np.savez(out, ep=eps, ec=ec, g=g, band_ratio=braw, band_err=berr, band_edges=BE,
         band_smoothed=gs)
np.set_printoptions(linewidth=200, precision=3, suppress=True)
print("E_p", eps)
print("band edges", BE)
print("raw band ratios (fixed / old guard) rows = E_p:")
print(braw)
print("errors:")
print(berr)
print("used (smoothed along E_p):")
print(gs)
