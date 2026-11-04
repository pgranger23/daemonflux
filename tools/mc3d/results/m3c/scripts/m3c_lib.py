"""Milestone-3c analysis helpers on the per-(jackknife group, stratum) sums
written by ``extract_strata.py`` (``$M/work/strata_<tag>.npz``).

Runs on the same tasks share the jackknife groups (task j of a stratum ->
group j % G), so the replicates of a difference are the differences of the
replicates: a paired jackknife."""
import os

import numpy as np

import analyse_m3b as a

M = os.environ["M"]
SP = a.SP_NAMES
EB = list(a.E_BANDS)
TAG = {"s": 0, "n": 1, "t": 2, "v": 3}
_AX = None


def axis():
    global _AX
    if _AX is None:
        _AX = a.ew_axis()
    return _AX


def load(tag, where=None):
    p = os.path.join(where or os.path.join(M, "work"), f"strata_{tag}.npz")
    return dict(np.load(p))


def replicates(S, Mg, per_stratum=False):
    """``(G, ...)`` leave-one-group-out totals (``analyse_sky.JackRun``);
    ``per_stratum`` keeps the stratum axis: ``(G, K, ...)``."""
    Sk, Mk = S.sum(0), Mg.sum(0)
    bshape = (-1,) + (1,) * (S.ndim - 2)
    out = []
    for g in range(S.shape[0]):
        rem = Mk - Mg[g]
        fac = np.where(rem > 0, Mk / np.where(rem > 0, rem, 1.0), 0.0)
        r = (Sk - S[g]) * fac.reshape(bshape)
        out.append(r if per_stratum else r.sum(0))
    return np.array(out)


def total(S):
    return S.sum((0, 1))


def jk_err(rep):
    G = rep.shape[0]
    return np.sqrt((G - 1.0) / G * np.nansum((rep - np.nanmean(rep, 0)) ** 2, 0))


def obs_pattern(f):
    """a1/a0, dphi, c1/a0, s1/a0, a0 of one 12-bin pattern."""
    o = a.observables(f, axis())
    return np.array([o[0], o[1], o[8], o[9], f.mean()])


OBS = ("a1_rel", "dphi", "c1_rel", "s1_rel", "a0")


def ew_obs(ew, tag="s", pooled=True, strata=None):
    """Observables (species, E band[, cz band], OBS) of an ``ew`` total
    ``(tag, sp, band, cz, az)``."""
    x = ew[TAG[tag]]
    if pooled:
        x = x.sum(2)
    out = np.zeros(x.shape[:-1] + (len(OBS),))
    for idx in np.ndindex(x.shape[:-1]):
        out[idx] = obs_pattern(x[idx])
    return out


class Run:
    def __init__(self, tag, where=None, key="ew"):
        d = load(tag, where)
        self.tag, self.d = tag, d
        self.S, self.M = d[key], d["M"]
        self.tot = total(self.S)
        self.rep = replicates(self.S, self.M)

    def obs(self, tag="s", pooled=True):
        est = ew_obs(self.tot, tag, pooled)
        rep = np.array([ew_obs(r, tag, pooled) for r in self.rep])
        return est, rep


def diff(o1, o2):
    """Paired difference of two ``(est, rep)``: ``(d, err)``; the phase
    wraps."""
    d = o1[0] - o2[0]
    dr = o1[1] - o2[1]
    d[..., 1] = (d[..., 1] + 180.0) % 360.0 - 180.0
    dr[..., 1] = (dr[..., 1] + 180.0) % 360.0 - 180.0
    return d, jk_err(dr)


def chi2(d, e, mask=None):
    z = (d / np.where(e > 0, e, np.nan)) ** 2
    if mask is not None:
        z = z[mask]
    z = z[np.isfinite(z)]
    return float(z.sum()), int(z.size)
