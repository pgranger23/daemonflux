"""Per-stratum diagnostics of the exact re-weighting (``reweight_replay.py``
-> ``$M/work/rwdiag_<TAG>_<n>.npz``): for each target and (group, stratum)

* ``f`` range over the pending primaries (min .. max of Phi_new / Phi_old),
* ``F/S``: the stratum's new / old contribution (sum of the primaries'
  no-cutoff 3D weights, ``S1`` -> ``F1``) = the stratum's flux ratio,
* effective sample size of the primaries' contributions c_i (their no-cutoff
  3D weight): ESS_old = (sum c)^2 / sum c^2, ESS_new = (sum c f)^2 / sum
  (c f)^2, and their ratio.  ESS_new / ESS_old < 1 = the new model weights the
  stratum's sample less evenly; ESS_new << ESS_old = poorly covered.

    python rw_diag.py RWDIAG.npz [min_share]   -> stdout (and $M/reports/rw_diag.txt)

Strata whose new contribution is below ``min_share`` (default 0.02) of the
target's total are summarised but not flagged.
"""
import os
import sys

import numpy as np

d = dict(np.load(sys.argv[1]))
min_share = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
GROUPS = ("p", "He", "CNO", "MgSi", "Fe")
L = []
for lab in sorted(set(d["label"])):
    m = d["label"] == lab
    tot_new = d["F1"][m].sum()
    tot_old = d["S1"][m].sum()
    L.append(f"\n## {lab}: total no-cutoff 3D weight new/old = {tot_new / tot_old:.4f}")
    L.append("group  k  stratum[GeV/n]   pend  f_min   f_max    F/S    share_new  ESS_old  ESS_new  ratio  flag")
    ks = sorted(set(d["k"][m]))
    for g in GROUPS:
        for k in ks:
            s = m & (d["group"] == g) & (d["k"] == k)
            if not s.any():
                continue
            S1, S2 = d["S1"][s].sum(), d["S2"][s].sum()
            F1, F2 = d["F1"][s].sum(), d["F2"][s].sum()
            if S1 <= 0:
                continue
            eo, en = S1 ** 2 / S2, (F1 ** 2 / F2 if F2 > 0 else 0.0)
            share = F1 / tot_new
            flag = ""
            if share >= min_share and (en < 0.5 * eo or en < 30):
                flag = "POOR"
            lo, hi = 10 ** (0.5 * k), 10 ** (0.5 * (k + 1))
            L.append(f"{g:5s} {k:2d} {lo:7.2f}-{hi:<7.2f} {int(d['n3'][s].sum()):6d} "
                     f"{np.nanmin(d['fmin'][s]):6.3f} {np.nanmax(d['fmax'][s]):7.3f} "
                     f"{F1 / S1:6.3f} {share:9.4f} {eo:8.1f} {en:8.1f} {en / eo:6.3f}  {flag}")
txt = "\n".join(L)
open(os.path.join(os.environ["M"], "reports", "rw_diag.txt"), "w").write(txt + "\n")
print(txt)
