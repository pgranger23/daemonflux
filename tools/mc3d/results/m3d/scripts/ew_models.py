"""The pilot's pooled East-West table (4 species x 3 energy bands, cos Z 0-0.3
pooled, largest cap, tag s) under several primary models, from the per-stratum
sums of ``results/m3c/scripts/extract_strata.py`` (``$M/work/strata_<tag>.npz``).

    python ew_models.py LABEL=TAG[@WORKDIR] ... -> $M/reports/ew_models.txt (+ .npz)

The first entry is the reference for the paired differences.  Reports, per
run, a1/a0, dphi [deg], fN, fS, N/S in the 12 cells (jackknife errors, G = 20),
the species fan-out, chi2/12 vs Honda (kam-ally-20-12-solmin; MC errors only)
for a1/a0 and the other observables, and the az-summed flux relative to the
reference."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "m3c", "scripts"))
import analyse_m3b as a  # noqa: E402
import m3c_lib as L  # noqa: E402

M = os.environ["M"]
ENTRIES = []
for s in sys.argv[1:]:
    lab, _, rest = s.partition("=")
    tag, _, where = rest.partition("@")
    ENTRIES.append((lab, tag, where or None))
NAMES = ("a1/a0", "dphi", "fN", "fS", "N/S")
IDX = (0, 1, 5, 6, 7)
SP = L.SP


def cells(ew):
    """(sp, band, 5) observables of the cos Z-pooled patterns of an ``ew``
    total (tag, sp, band, cz, az), tag s."""
    x = ew[0].sum(2)
    out = np.zeros(x.shape[:2] + (len(IDX),))
    for k in range(x.shape[0]):
        for ib in range(x.shape[1]):
            out[k, ib] = a.observables(x[k, ib], L.axis())[list(IDX)]
    return out


def jk(run):
    est = cells(run.tot)
    rep = np.array([cells(r) for r in run.rep])
    return est, rep


def wrap(d):
    d = d.copy()
    d[..., 1] = (d[..., 1] + 180.0) % 360.0 - 180.0
    return d


hp = a.honda_patterns().sum(2)
hon = np.array([[a.observables(hp[k, ib], L.axis())[list(IDX)]
                 for ib in range(3)] for k in range(4)])
runs = {}
for lab, tag, where in ENTRIES:
    r = L.Run(tag, where)
    est, rep = jk(r)
    G = rep.shape[0]
    dev = wrap(rep - np.nanmean(rep, 0))
    runs[lab] = (r, est, rep, np.sqrt((G - 1.0) / G * np.nansum(dev ** 2, 0)))
ref = ENTRIES[0][0]
lines = [f"# pooled cos Z 0-0.3, largest cap, tag s; bands {L.EB} GeV; "
         f"jackknife G=20; Honda = kam-ally-20-12-solmin"]
for lab, tag, _w in ENTRIES:
    r, est, rep, err = runs[lab]
    lines.append(f"\n## {lab}  ({tag}, {r.M.sum():.3g} primaries)")
    for j, nm in enumerate(NAMES):
        for ib, E in enumerate(L.EB):
            lines.append(f"{nm:6s} E {E}: " + "  ".join(
                f"{sp} {est[k, ib, j]:.3f}+-{err[k, ib, j]:.3f} ({hon[k, ib, j]:.3f})"
                for k, sp in enumerate(SP)))
    z = wrap(est - hon) / np.where(err > 0, err, np.nan)
    lines.append("chi2/12 vs Honda (MC errors): " + "  ".join(
        f"{nm} {np.nansum(z[..., j] ** 2) / 12:.2f}" for j, nm in enumerate(NAMES)))
    lines.append("mean a1/a0 over the 12 cells: %.4f  (Honda %.4f)"
                 % (est[..., 0].mean(), hon[..., 0].mean()))
    lines.append("mean (MC - Honda) a1/a0 %+.4f, N/S %+.3f, dphi %+.2f deg" % (
        (est[..., 0] - hon[..., 0]).mean(), (est[..., 4] - hon[..., 4]).mean(),
        wrap(est[..., 1] - hon[..., 1]).mean()))
    for (i, j) in ((0, 3), (3, 2), (2, 1)):
        x = est[i, :, 0] - est[j, :, 0]
        lines.append(f"fan-out {SP[i]}-{SP[j]}: " + " ".join(f"{u:+.3f}" for u in x)
                     + "  (Honda " + " ".join(
                         f"{h:+.3f}" for h in hon[i, :, 0] - hon[j, :, 0]) + ")")
    if lab != ref:
        r0, e0, rep0, _ = runs[ref]
        d = wrap(est - e0)
        dd = wrap(rep - rep0)
        de = np.sqrt((G - 1.0) / G * np.nansum((dd - np.nanmean(dd, 0)) ** 2, 0))
        lines.append(f"paired d(a1/a0) vs {ref}: mean {d[..., 0].mean():+.4f} "
                     f"(per cell +-{np.median(de[..., 0]):.4f}); range "
                     f"{d[..., 0].min():+.3f}..{d[..., 0].max():+.3f}")
        q = r.tot[0].sum((2, 3)) / r0.tot[0].sum((2, 3))
        lines.append(f"3D flux (s, cos Z 0-0.3, az-summed) / {ref}: " + "  ".join(
            f"{sp} " + " ".join(f"{x:.3f}" for x in q[k]) for k, sp in enumerate(SP)))
open(f"{M}/reports/ew_models.txt", "w").write("\n".join(lines) + "\n")
np.savez(f"{M}/reports/ew_models.npz", **{f"{lab}_est": runs[lab][1] for lab in runs},
         **{f"{lab}_err": runs[lab][3] for lab in runs}, honda=hon)
print("\n".join(lines))
