"""Per-(jackknife group, (group, stratum)) sums of a run's tallies, reduced to
what milestone 3c's analyses need, saved to ``$M/work/strata_<tag>.npz``:

* ``ew``  (G, K, tag s/n/t/v, species, E band, cz 0-0.1/0.1-0.2/0.2-0.3, az)
  measure 1, largest cap, band-summed over ``analyse_m3b.E_BANDS``;
* ``ew0`` the same in measure 0;
* ``vert`` (G, K, tag, species, E band) measure 1, cos Z 0.9-1, az-summed;
* ``M`` (G, K) primaries, ``strata`` the (group, stratum) of each K;
* ``mub_m``, ``mub_h`` the muon-bend diagnostics (if present).

    python extract_strata.py RUNDIR TAG
"""
import os
import sys

import numpy as np

import analyse_m3b as a
import analyse_sky as A

rundir, tag = sys.argv[1], sys.argv[2]
EB = list(a.E_BANDS.values())


def _bands(x):            # (..., E, ...) with E at axis -3 -> bands
    return np.stack([x[..., i0:i1, :, :].sum(-3) for i0, i1 in EB], -3)


def red_ew(d, m="1"):
    x = np.stack([np.asarray(d[t + m])[-1, :, :, 10:13, :] for t in "sntv"])
    return _bands(x)


def red_vert(d):
    x = np.stack([np.asarray(d[t + "1"])[-1, :, :, 19, :].sum(-1)
                  for t in "sntv"])               # (tag, sp, E)
    return np.stack([x[..., i0:i1].sum(-1) for i0, i1 in EB], -1)


red = {"ew": red_ew, "ew0": lambda d: red_ew(d, "0"), "vert": red_vert}


def _mub(name):
    def f(d):
        if name in d.files:
            return np.asarray(d[name], float)
        return np.zeros(1)
    return f


red["mub_m"] = _mub("mub_m")
red["mub_h"] = _mub("mub_h")
red["mucf"] = _mub("mucf")
J = A.JackRun(rundir, tag, G=20, reducers=red)
out = {k: J.S[k] for k in red}
out["M"] = J.M
out["strata"] = np.array([f"{g}:{k}" for g, k in J.strata])
out["cpu"] = J.cpu
out["nsh"] = J.nsh
np.savez(os.path.join(os.environ["M"], "work", f"strata_{tag}.npz"), **out)
print("wrote", tag, {k: v.shape for k, v in out.items()})
