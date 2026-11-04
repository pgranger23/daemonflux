"""Task 2 (milestone 3c): the muon-bend mechanism.

Six runs on the same seeds, tasks and allocation (1e6 primaries, E^-0.8,
r_thr 0, re-anchored), paired cascade by cascade:

  a1anc    cutoff on,  muon field on            (m3b)
  a1nomu   cutoff on,  muon field off           (m3b)
  a1nc     cutoff off, muon field on
  a1ncnomu cutoff off, muon field off
  a1vert   cutoff on,  vertical component of each muon's bend only
  a1horiz  cutoff on,  horizontal component only

Observables: the 3D tally ``s`` (all 12 copies; with the cutoff off every copy
is allowed, so it is the all-copy no-cutoff tally), pooled cos Z 0-0.3:
``a1/a0``, ``c1/a0`` (the dipole component towards geomagnetic West, signed),
``s1/a0``, ``a0``.  The muon effect is the paired difference field on - off;
its size relative to the field-off ``a0`` of the same cutoff setting is

  eps = [c1 a0 (on) - c1 a0 (off)] / a0 (off).

To first order a cutoff-independent zenith migration (H1) gives the same eps
with and without the cutoff (the cutoff multiplies the flux by tau(cz, az);
the migration's dipole is carried over with tau's monopole); a change of the
effective cutoff through the parent direction (H2) gives eps = 0 without the
cutoff.  Plus the muon-bend histograms ``mub_m`` of the parent muon of the
neutrinos scored in cos Z 0-0.3 (no-cutoff run).

    python mech.py -> $M/reports/mech.txt, mech.npz
"""
import os

import numpy as np

import m3c_lib as L
import sky3d

M = os.environ["M"]
lines = []


def out(s=""):
    print(s, flush=True)
    lines.append(s)


tags = [t for t in ("a1anc", "a1nomu", "a1nc", "a1ncnomu", "a1vert", "a1horiz")
        if os.path.exists(f"{M}/work/strata_{t}.npz")]
R = {t: L.Run(t) for t in tags}
O = {t: R[t].obs("s", pooled=True) for t in tags}          # (sp, band, OBS)
Oz = {t: R[t].obs("s", pooled=False) for t in tags}        # (sp, band, cz, OBS)
ax = L.axis()
out(f"# milestone 3c task 2: muon-bend mechanism; runs {tags}; geomagnetic East {ax:.2f}")
out("\n## 1. pooled cos Z 0-0.3, 3D tally s: a1/a0, c1/a0 (towards W), s1/a0, dphi")
for t in tags:
    e, r = O[t]
    er = L.jk_err(r)
    out(f"### {t}")
    for ib, E in enumerate(L.EB):
        out(f"E {E}: " + "  ".join(
            f"{sp} a1 {e[k, ib, 0]:.3f}+-{er[k, ib, 0]:.3f} c1 {e[k, ib, 2]:+.3f}+-"
            f"{er[k, ib, 2]:.3f} s1 {e[k, ib, 3]:+.3f}+-{er[k, ib, 3]:.3f} dphi "
            f"{e[k, ib, 1]:+.0f}" for k, sp in enumerate(L.SP)))


def eps(on, off, Odict=O):
    """eps = d(c1 a0)/a0(off) and d(s1 a0)/a0(off), d a0/a0(off); paired."""
    (e1, r1), (e0, r0) = Odict[on], Odict[off]

    def f(x1, x0):
        a0 = x0[..., 4]
        return np.stack([(x1[..., 2] * x1[..., 4] - x0[..., 2] * x0[..., 4]) / a0,
                         (x1[..., 3] * x1[..., 4] - x0[..., 3] * x0[..., 4]) / a0,
                         x1[..., 4] / a0 - 1.0], -1)
    d = f(e1, e0)
    dr = np.array([f(x1, x0) for x1, x0 in zip(r1, r0)])
    return d, L.jk_err(dr), dr


PAIRS = (("a1anc", "a1nomu", "muon bend, cutoff ON"),
         ("a1nc", "a1ncnomu", "muon bend, cutoff OFF"),
         ("a1vert", "a1nomu", "vertical part of the bend only, cutoff ON"),
         ("a1horiz", "a1nomu", "horizontal part only, cutoff ON"))
res = {}
out("\n## 2. the muon effect, paired: eps_c = d(c1 a0)/a0(off) [West dipole], "
    "eps_s = d(s1 a0)/a0(off), d a0/a0 (pooled cos Z 0-0.3)")
for on, off, lab in PAIRS:
    if on not in O or off not in O:
        continue
    d, er, dr = eps(on, off)
    res[(on, off)] = (d, er, dr)
    out(f"### {on} - {off}: {lab}")
    for ib, E in enumerate(L.EB):
        out(f"E {E}: " + "  ".join(
            f"{sp} eps_c {d[k, ib, 0]:+.3f}+-{er[k, ib, 0]:.3f} eps_s "
            f"{d[k, ib, 1]:+.3f}+-{er[k, ib, 1]:.3f} da0 {d[k, ib, 2]:+.3f}+-"
            f"{er[k, ib, 2]:.3f}" for k, sp in enumerate(L.SP)))
    # charge-odd and charge-even parts: mu+ species (nue, antinumu) vs mu-
    # (antinue, numu)
    for ib, E in enumerate(L.EB):
        x = d[:, ib, 0]
        xr = dr[:, :, ib, 0]
        odd = 0.5 * ((x[0] - x[1]))
        oddr = 0.5 * (xr[:, 0] - xr[:, 1])
        odd2 = 0.5 * ((x[3] - x[2]))
        odd2r = 0.5 * (xr[:, 3] - xr[:, 2])
        out(f"   E {E}: charge-odd (nue-antinue)/2 {odd:+.3f}+-{L.jk_err(oddr):.3f}"
            f"   (antinumu-numu)/2 {odd2:+.3f}+-{L.jk_err(odd2r):.3f}")

np.savez(M + "/reports/mech_eps.npz", **{f"{a}|{b}": res[(a, b)][0] for a, b in res},
         **{f"{a}|{b}|err": res[(a, b)][1] for a, b in res},
         **{f"{a}|{b}|rep": res[(a, b)][2] for a, b in res})
# 3. H1 test: eps with the cutoff vs without
if ("a1anc", "a1nomu") in res and ("a1nc", "a1ncnomu") in res:
    dc, ec, rc = res[("a1anc", "a1nomu")]
    dn, en, rn = res[("a1nc", "a1ncnomu")]
    out("\n## 3. H1 test: eps_c with the cutoff (fan-out) vs without (pure "
        "geometry); paired difference and ratio")
    for ib, E in enumerate(L.EB):
        row = []
        for k, sp in enumerate(L.SP):
            dd = dc[k, ib, 0] - dn[k, ib, 0]
            de = L.jk_err(rc[:, k, ib, 0] - rn[:, k, ib, 0])
            row.append(f"{sp} cut {dc[k, ib, 0]:+.3f} nocut {dn[k, ib, 0]:+.3f} "
                       f"diff {dd:+.3f}+-{de:.3f}")
        out(f"E {E}: " + "  ".join(row))
    z = []
    for ib in range(3):
        for k in range(4):
            dd = dc[k, ib, 0] - dn[k, ib, 0]
            de = L.jk_err(rc[:, k, ib, 0] - rn[:, k, ib, 0])
            z.append(dd / de)
    z = np.array(z)
    out(f"chi2(eps_cut = eps_nocut)/12 = {np.sum(z ** 2) / 12:.2f}")
    sl = np.sum(dc[..., 0] * dn[..., 0] / (ec[..., 0] ** 2 + en[..., 0] ** 2)) / \
        np.sum(dn[..., 0] ** 2 / (ec[..., 0] ** 2 + en[..., 0] ** 2))
    out(f"least-squares slope eps_cut = k eps_nocut: k = {sl:.3f}")

# 4. additivity of the decomposition and its weights
if all(p in res for p in (("a1anc", "a1nomu"), ("a1vert", "a1nomu"),
                          ("a1horiz", "a1nomu"))):
    dF, eF, rF = res[("a1anc", "a1nomu")]
    dV, eV, rV = res[("a1vert", "a1nomu")]
    dH, eH, rH = res[("a1horiz", "a1nomu")]
    out("\n## 4. decomposition with the cutoff: full vs vertical-only + "
        "horizontal-only (eps_c, paired)")
    for ib, E in enumerate(L.EB):
        row = []
        for k, sp in enumerate(L.SP):
            nl = dF[k, ib, 0] - dV[k, ib, 0] - dH[k, ib, 0]
            nle = L.jk_err(rF[:, k, ib, 0] - rV[:, k, ib, 0] - rH[:, k, ib, 0])
            row.append(f"{sp} full {dF[k, ib, 0]:+.3f} vert {dV[k, ib, 0]:+.3f}+-"
                       f"{eV[k, ib, 0]:.3f} horiz {dH[k, ib, 0]:+.3f}+-{eH[k, ib, 0]:.3f}"
                       f" nonadd {nl:+.3f}+-{nle:.3f}")
        out(f"E {E}: " + "  ".join(row))

# 5. zenith migration per cos Z band: d a0/a0 and eps per band
out("\n## 5. per cos Z band (0-0.1, 0.1-0.2, 0.2-0.3): eps_c and d a0/a0")
for on, off, lab in PAIRS[:2]:
    if (on, off) not in res:
        continue
    d, er, _ = eps(on, off, Oz)
    out(f"### {on} - {off}: {lab}")
    for ib, E in enumerate(L.EB):
        for k, sp in enumerate(L.SP):
            out(f"E {E} {sp:9s}: " + "  ".join(
                f"cz{iz}: eps_c {d[k, ib, iz, 0]:+.3f}+-{er[k, ib, iz, 0]:.3f} da0 "
                f"{d[k, ib, iz, 2]:+.3f}+-{er[k, ib, iz, 2]:.3f}" for iz in range(3)))

# 6. the bend of the parent muons, no-cutoff run (cos Z 0-0.3)
mub = {}
for t in ("a1nc", "a1anc_none", "a1vert", "a1horiz"):
    if t in R and R[t].d["mub_m"].ndim > 1:
        mub[t] = R[t].d["mub_m"].sum((0, 1))          # (sp, band, cz, az, 5)
        mub[t + "_h"] = R[t].d["mub_h"].sum((0, 1))
if "a1nc" in mub:
    m = mub["a1nc"]
    out("\n## 6. parent-muon bend of the muon-born 3D neutrinos (a1nc, no cutoff, "
        "measure 0): mean bv = d cosZ_arrival (> 0: away from the horizon) and "
        "bh = d az [deg] per arrival azimuth (12 bins, compass centres 15..345)")
    azc = 15.0 + 30.0 * np.arange(12)
    W = (ax + 180.0) % 360.0
    for ib, E in enumerate(L.EB):
        for k, sp in enumerate(L.SP):
            w = m[k, ib, :, :, 0].sum(0)
            bv = m[k, ib, :, :, 1].sum(0) / np.maximum(w, 1e-300)
            bh = m[k, ib, :, :, 3].sum(0) / np.maximum(w, 1e-300)
            sv = np.sqrt(np.maximum(m[k, ib, :, :, 2].sum(0) / np.maximum(w, 1e-300)
                                    - bv ** 2, 0))
            c = np.cos(np.radians(azc - W))
            s = np.sin(np.radians(azc - W))
            b1c = 2 * np.sum(w * bv * c) / np.sum(w) if w.sum() else np.nan
            b1s = 2 * np.sum(w * bv * s) / np.sum(w) if w.sum() else np.nan
            out(f"E {E} {sp:9s}: <bv> W-component {b1c:+.4f} perp {b1s:+.4f} "
                f"mean {np.sum(w * bv) / max(w.sum(), 1e-300):+.4f} rms(bv) "
                f"{np.sum(w * sv) / max(w.sum(), 1e-300):.4f}; <bh> "
                f"{np.sum(w * bh) / max(w.sum(), 1e-300):+.2f} deg; bv(az) "
                + " ".join(f"{x:+.3f}" for x in bv))
np.savez(M + "/reports/mech.npz",
         **{f"obs|{t}": O[t][0] for t in tags},
         **{f"obserr|{t}": L.jk_err(O[t][1]) for t in tags},
         **{f"eps|{a}|{b}": res[(a, b)][0] for a, b in res},
         **{f"epserr|{a}|{b}": res[(a, b)][1] for a, b in res},
         **{f"mub|{t}": v for t, v in mub.items()}, axis=ax)
open(M + "/reports/mech.txt", "w").write("\n".join(lines) + "\n")
