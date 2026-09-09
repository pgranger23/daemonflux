"""Summarise a closure npz: MC/MCEq ratio per species in energy bands."""

from __future__ import annotations

import argparse

import numpy as np

BANDS = [(0.1, 0.3), (0.3, 1.0), (1.0, 3.0), (3.0, 10.0), (10.0, 30.0)]
NAME = {12: "nu_e", -12: "antinu_e", 14: "nu_mu", -14: "antinu_mu"}


def summarize(path, energies=None):
    d = np.load(path)
    eb = d["e_bins"]
    ec = np.sqrt(eb[1:] * eb[:-1])
    eps = sorted({float(k.split("_")[1]) for k in d.files if k.startswith("mc_")})
    if energies:
        eps = [e for e in eps if e in energies]
    out = {}
    for ep in eps:
        for s in (14, -14, 12, -12):
            mc = d[f"mc_{ep}_{s}"]
            err = d[f"err_{ep}_{s}"]
            ref = d[f"ref_{ep}_{s}"]
            rows = []
            for lo, hi in BANDS:
                m = (ec >= lo) & (ec < hi) & (ref > 0)
                if not m.any():
                    continue
                w = np.diff(eb)[m]
                num = np.sum(mc[m] * w)
                den = np.sum(ref[m] * w)
                e = np.sqrt(np.sum((err[m] * w) ** 2))
                if den <= 0:
                    continue
                rows.append((lo, hi, num / den, e / den))
            out[(ep, s)] = rows
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args(argv)
    for p in a.paths:
        print(f"\n===== {p} =====")
        res = summarize(p)
        eps = sorted({k[0] for k in res})
        for ep in eps:
            print(f"\n E_p = {ep:g} GeV        " +
                  "  ".join(f"{lo:g}-{hi:g} GeV" for lo, hi in BANDS))
            for s in (14, -14, 12, -12):
                rows = res.get((ep, s), [])
                cells = []
                for lo, hi in BANDS:
                    hit = [r for r in rows if r[0] == lo]
                    cells.append(f"{hit[0][2]:.3f}+-{hit[0][3]:.3f}"
                                 if hit else "     ---     ")
                print(f"  {NAME[s]:>10s}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
