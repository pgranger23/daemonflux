"""Summarise closure_floor npz files: all-flavour band yield ratio MC/MCEq per
primary kinetic energy, for every rung present in $M/reports/cl_<rung>_{lo,hi}.npz."""
import os, sys
import numpy as np
M = os.environ["M"]
RUNGS = sys.argv[1:] or ["A1", "A2", "A2f2", "B"]
BANDS = [(0.1, 0.3), (0.3, 1.0), (1.0, 3.0)]
SP = (14, -14, 12, -12)
def load(r):
    d = {}
    r, _, parts = r.partition(":")
    for part in (parts.split(",") if parts else ("lo", "hi")):
        p = f"{M}/reports/cl_{r}_{part}.npz"
        if os.path.exists(p):
            z = np.load(p); d.update({k: z[k] for k in z.files})
    return d
D = {r: load(r) for r in RUNGS}
eb = next(iter(D.values()))["e_bins"]; ec = np.sqrt(eb[1:] * eb[:-1]); w = np.diff(eb)
eps = sorted({float(k.split("_")[1]) for d_ in D.values() for k in d_ if k.startswith("mc_")})
L = ["# all-flavour (nue+antinue+numu+antinumu) yield per primary proton, MC / MCEq(1D), vertical column; E_p = kinetic [GeV]"]
L.append("band [GeV]  rung   " + "  ".join(f"{e:>13g}" for e in eps))
for lo, hi in BANDS:
    m = (ec >= lo) & (ec < hi)
    for r in RUNGS:
        row = []
        for e in eps:
            if f"mc_{e}_14" not in D[r]:
                row.append(" " * 13); continue
            num = sum((D[r][f"mc_{e}_{s}"][m] * w[m]).sum() for s in SP)
            den = sum((D[r][f"ref_{e}_{s}"][m] * w[m]).sum() for s in SP)
            err = np.sqrt(sum(((D[r][f"err_{e}_{s}"][m] * w[m]) ** 2).sum() for s in SP))
            row.append(f"{num / den:6.3f}+-{err / den:5.3f}")
        L.append(f"{lo:g}-{hi:g}      {r:6s} " + "  ".join(row))
    L.append("")
# absolute yields (per proton) for MCEq reference
for lo, hi in BANDS:
    m = (ec >= lo) & (ec < hi)
    L.append(f"MCEq absolute yield {lo:g}-{hi:g}: " + "  ".join(f"{e:g}: {sum((next(d_ for d_ in D.values() if f'ref_{e}_14' in d_)[f'ref_{e}_{s}'][m] * w[m]).sum() for s in SP):.4f}" for e in eps))
open(f"{M}/reports/ladder.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
