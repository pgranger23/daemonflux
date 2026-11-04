"""Milestone-3e analysis of pilot-2 and its muon-field-off companion.

    python analyse_m3e.py [--engine ENGINE_NOMU.npz] [--ref-gfix M3D_NPZ]

Reads ``$M/work/strata_pilot2.npz`` and ``strata_comp.npz``
(``results/m3c/scripts/extract_strata.py``).  Output ``$M/reports/m3e_table.txt``.

Per species (nue, antinue, numu, antinumu) and energy band (0.3, 0.5, 1 GeV),
pooled cos Z 0-0.3, largest cap, tag s: a1/a0, dphi, fN, fS, N/S with
jackknife errors (G = 20) against Honda HKKM2014 (kam-ally-20-12-solmin);
species-pooled means; fan-out antinumu-numu and nue-antinue; chi2/12 (MC errors
only).  The companion (``--no-muon-field``) is compared with pilot-2 (paired
seeds: the companion's chunks are the first chunks of pilot-2's) and, if given,
with the engine run with ``muon_bending=False``.

ENGINE: either the rigfix flux files (``/home/pigrange/mc3d_work/runs/rigfix/out/m3b_engine_<h3a|gsf|ghnative>_mb0.npz``,
read through ``analyse_m3b.engine_patterns``) or, alternatively, a plain npz in this format: keys ``a1_rel``, ``dphi``,
``fN``, ``fS``, ``NS`` each of shape (4, 3) = (nue, antinue, numu, antinumu) x
(0.3, 0.5, 1.0 GeV), pooled cos Z 0-0.3 / all azimuth bins as ``honda_kam.npz``
bins, same band integration; optional ``<key>_err`` of the same shape.
``--ref-gfix`` = an ``ew_models*.npz`` of milestone 3d (key ``<label>_est``;
label via ``--ref-label``, default GSFG_GH_gfix) for the +gfix check."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "m3c", "scripts"))
import analyse_m3b as a  # noqa: E402
import m3c_lib as L  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--engine")
ap.add_argument("--ref-gfix")
ap.add_argument("--ref-label", default="GSFG_GH_gfix")
ap.add_argument("--main", default="pilot2")
ap.add_argument("--comp", default="comp")
ap.add_argument("--where", default=None)
A = ap.parse_args()
M = os.environ["M"]
NAMES = ("a1/a0", "dphi", "fN", "fS", "N/S")
IDX = (0, 1, 5, 6, 7)
SP = L.SP


def cells(ew):
    x = ew[0].sum(2)
    out = np.zeros(x.shape[:2] + (5,))
    for k in range(4):
        for ib in range(3):
            out[k, ib] = a.observables(x[k, ib], L.axis())[list(IDX)]
    return out


def wrap(d):
    d = d.copy()
    d[..., 1] = (d[..., 1] + 180.0) % 360.0 - 180.0
    return d


def jk(rep, ref=None):
    G = rep.shape[0]
    dd = wrap(rep - (rep.mean(0) if ref is None else ref))
    dd = dd - np.nanmean(dd, 0) if ref is not None else dd
    return np.sqrt((G - 1.0) / G * np.nansum(dd ** 2, 0))


def load(tag):
    r = L.Run(tag, A.where)
    est = cells(r.tot)
    rep = np.array([cells(x) for x in r.rep])
    return r, est, rep, jk(rep)


hp = a.honda_patterns().sum(2)
hon = np.array([[a.observables(hp[k, ib], L.axis())[list(IDX)] for ib in range(3)] for k in range(4)])
L_ = []
def out(s=""):
    print(s); L_.append(s)

R1, e1, r1, s1 = load(A.main)
out(f"# pilot-2 ({A.main}): {R1.M.sum():.4g} primaries; Honda = kam-ally-20-12-solmin; errors = jackknife G=20")
def table(lab, est, err, extra=None):
    out(f"\n## {lab}")
    for j, nm in enumerate(NAMES):
        for ib, E in enumerate(L.EB):
            out(f"{nm:6s} E {E}: " + "  ".join(f"{sp} {est[k, ib, j]:.3f}+-{err[k, ib, j]:.3f} ({hon[k, ib, j]:.3f})" for k, sp in enumerate(SP)))
    z = wrap(est - hon) / np.where(err > 0, err, np.nan)
    out("chi2/12 vs Honda: " + "  ".join(f"{nm} {np.nansum(z[..., j] ** 2) / 12:.2f}" for j, nm in enumerate(NAMES)))
    out("species-pooled (mean over the 3 bands) a1/a0: " + "  ".join(f"{sp} {est[k, :, 0].mean():.3f} (Honda {hon[k, :, 0].mean():.3f})" for k, sp in enumerate(SP)))
    out("all 12 cells: <a1/a0> %.4f (Honda %.4f)  <dphi> %.2f (%.2f)  <fN> %.3f (%.3f)  <fS> %.3f (%.3f)  <N/S> %.3f (%.3f)" % (
        est[..., 0].mean(), hon[..., 0].mean(), est[..., 1].mean(), hon[..., 1].mean(), est[..., 2].mean(), hon[..., 2].mean(),
        est[..., 3].mean(), hon[..., 3].mean(), est[..., 4].mean(), hon[..., 4].mean()))
    for (i, j, nm) in ((3, 2, "antinumu-numu"), (0, 1, "nue-antinue")):
        x = est[i, :, 0] - est[j, :, 0]
        out(f"fan-out {nm}: " + " ".join(f"{u:+.3f}" for u in x) + "  (Honda " + " ".join(f"{h:+.3f}" for h in hon[i, :, 0] - hon[j, :, 0]) + ")")
table(f"pilot-2 ({A.main})", e1, s1)
if A.ref_gfix:
    z = np.load(A.ref_gfix)
    ref = z[A.ref_label + "_est"]
    d = wrap(e1 - ref)
    out(f"\n## +gfix check vs M3d {A.ref_label}: <a1/a0> pilot-2 {e1[...,0].mean():.4f} vs {ref[...,0].mean():.4f} "
        f"(diff {d[...,0].mean():+.4f}); per-cell diff median |d| {np.median(abs(d[...,0])):.3f}; "
        f"per-cell pilot-2 error median {np.median(s1[...,0]):.3f}")
    out(f"   <a1/a0> shift pilot-2 - M3d(no gfix) cannot be formed here; M3d quoted GSFG-GH {0.4304:.4f} -> {0.4422:.4f} (+0.0118)")
try:
    R2, e2, r2, s2 = load(A.comp)
    table(f"companion ({A.comp}, muon field off)", e2, s2)
    d = wrap(e2 - e1)
    out(f"\n## companion - pilot-2 (different primary samples; companion is 0.4x of pilot-2's allocation, "
        f"first chunks shared): <d a1/a0> {d[...,0].mean():+.4f}; per cell " + " ".join(f"{x:+.3f}" for x in d[..., 0].ravel()))
    dr = wrap(r2 - r1)
    G = dr.shape[0]
    dd = np.sqrt((G - 1.0) / G * np.nansum((dr - np.nanmean(dr, 0)) ** 2, 0))
    out("   jackknife error of the difference (assuming pairing is only partial; the plain quadrature sum would be "
        f"{np.sqrt(np.nanmedian(s1[...,0])**2 + np.nanmedian(s2[...,0])**2):.3f}): median {np.nanmedian(dd[...,0]):.3f}")
    if A.engine:
        en = np.load(A.engine)
        if "total_nue" in en.files:          # rigfix engine cells (flux format)
            pats = a.engine_patterns(A.engine).sum(2)          # (4, 3, 12) pooled cos Z 0-0.3
            eng = np.array([[a.observables(pats[k, ib], L.axis())[list(IDX)] for ib in range(3)] for k in range(4)])
            ee = np.zeros_like(eng)
        else:                                 # plain (4, 3) arrays
            eng = np.stack([en[k] for k in ("a1_rel", "dphi", "fN", "fS", "NS")], -1)
            ee = np.stack([en[k + "_err"] if k + "_err" in en.files else np.zeros_like(en[k]) for k in ("a1_rel", "dphi", "fN", "fS", "NS")], -1)
        dz = wrap(e2 - eng)
        sig = np.sqrt(s2 ** 2 + ee ** 2)
        out(f"\n## companion vs ENGINE (muon_bending=False): {os.path.basename(A.engine)}")
        for j, nm in enumerate(NAMES):
            out(f"{nm:6s} MC-engine per cell: " + " ".join(f"{x:+.3f}" for x in dz[..., j].ravel()) +
                f"   mean {dz[..., j].mean():+.4f}  chi2/12 {np.nansum((dz[..., j]/np.where(sig[..., j]>0, sig[..., j], np.nan))**2)/12:.2f}")
except Exception as ex:   # companion not available
    out(f"\n(companion not analysed: {ex!r})")
open(f"{M}/reports/m3e_table.txt", "w").write("\n".join(L_) + "\n")
np.savez(f"{M}/reports/m3e_table.npz", est=e1, err=s1, honda=hon)
