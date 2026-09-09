"""NA61/SHINE validation of the **production angle** (not just p_T).

``validate_na61.py`` compares ``<p_T>(p_lab)`` against the NA61/SHINE p+C
31 GeV/c data. ``p_T`` comes straight from the generator, so that test is blind
to how the pipeline *converts* ``(p_T, p)`` into a production angle -- which is
precisely the quantity the moment files (``m_*.npz``) store and the 3D engine
consumes. The pre-2026-09 bug (``theta = arctan(p_T / p)`` with ``p`` the total
momentum instead of the longitudinal one) therefore slipped through.

NA61 publishes ``d sigma / dp`` in bins of momentum ``p`` for ten polar-angle
ranges, so the data give ``<theta>(p_lab)`` directly. This script forms that
yield-weighted mean angle from the tables and compares it with the same quantity
from UrQMD-3.4, evaluated **both** ways:

* ``arcsin(p_T / p)``  -- the correct relation, used by the ``*_v2.npz`` moments;
* ``arctan(p_T / p)``  -- the old, biased convention behind the original ``m_*.npz``.

Run::

    python validate_na61_angle.py --nevents 200000 --nproc 24
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np

from validate_na61 import (
    M_PION,
    PIMINUS_TABLES,
    PIPLUS_TABLES,
    _theta_center_rad,
    _yield_value,
    load_na61,
)

THETA_ACC = 0.420  # rad, NA61 forward acceptance
E_LAB = 31.0  # GeV/c beam on carbon


def na61_angle_moments(data, tables, p_edges):
    """NA61 ``<theta>`` and ``sqrt(<theta^2>)`` [mrad] vs p_lab, yield-weighted."""
    p_c, th_c, wt = [], [], []
    for name in tables:
        tab = data[name]
        th = _theta_center_rad(tab["theta"])
        for v in tab["values"]:
            lo = float(v["x"][0]["low"])
            hi = float(v["x"][0]["high"])
            dsig = _yield_value(v["y"][0])
            if not np.isfinite(dsig) or dsig <= 0:
                continue
            p_c.append(0.5 * (lo + hi))
            th_c.append(th)
            wt.append(dsig * (hi - lo))
    p_c, th_c, wt = map(np.asarray, (p_c, th_c, wt))
    idx = np.digitize(p_c, p_edges) - 1
    centers = np.sqrt(p_edges[:-1] * p_edges[1:])
    mean = np.full(len(centers), np.nan)
    rms = np.full(len(centers), np.nan)
    for b in range(len(centers)):
        sel = (idx == b) & (th_c < THETA_ACC)
        if sel.any() and wt[sel].sum() > 0:
            w = wt[sel]
            mean[b] = np.sum(th_c[sel] * w) / w.sum()
            rms[b] = np.sqrt(np.sum(th_c[sel] ** 2 * w) / w.sum())
    return centers, mean * 1e3, rms * 1e3


_MODEL = None


def _init(seed):
    global _MODEL
    warnings.simplefilter("ignore")
    import chromo
    from chromo.kinematics import FixedTarget, GeV

    _MODEL = chromo.models.UrQMD34(
        FixedTarget(E_LAB * GeV, 2212, (12, 6)), seed=(seed + os.getpid()) % (2**31 - 1)
    )


def _shard(nev):
    pls, pts, pids = [], [], []
    for event in _MODEL(nev):
        fs = event.final_state()
        sel = (fs.pid == 211) | (fs.pid == -211)
        if not np.any(sel):
            continue
        pls.append(np.sqrt(np.maximum(fs.en[sel] ** 2 - M_PION**2, 0.0)))
        pts.append(fs.pt[sel])
        pids.append(fs.pid[sel])
    if not pls:
        return np.empty(0), np.empty(0), np.empty(0)
    return np.concatenate(pls), np.concatenate(pts), np.concatenate(pids)


def urqmd_angle_moments(p_edges, n_events, nproc, seed=17):
    """UrQMD-3.4 ``<theta>``/``rms`` [mrad] per pid, in both angle conventions."""
    import multiprocessing as mp

    shard = max(1000, n_events // (4 * nproc))
    tasks = [shard] * (n_events // shard)
    ctx = mp.get_context("fork")
    P, T, I = [], [], []
    with ctx.Pool(nproc, initializer=_init, initargs=(seed,)) as pool:
        for pl, pt, pid in pool.imap_unordered(_shard, tasks):
            P.append(pl)
            T.append(pt)
            I.append(pid)
    p_tot, p_t, pid = np.concatenate(P), np.concatenate(T), np.concatenate(I)
    p_t = np.minimum(p_t, p_tot)
    th_new = np.arcsin(np.clip(p_t / np.maximum(p_tot, 1e-12), 0, 1))
    th_old = np.arctan2(p_t, np.maximum(p_tot, 1e-12))
    acc = th_new < THETA_ACC  # true detector acceptance for both

    centers = np.sqrt(p_edges[:-1] * p_edges[1:])
    idx = np.digitize(p_tot, p_edges) - 1
    out = {}
    for q in (211, -211):
        res = {}
        for tag, th in (("new", th_new), ("old", th_old)):
            m = np.full(len(centers), np.nan)
            r = np.full(len(centers), np.nan)
            for b in range(len(centers)):
                s = (idx == b) & (pid == q) & acc
                if s.sum() > 50:
                    m[b] = th[s].mean() * 1e3
                    r[b] = np.sqrt((th[s] ** 2).mean()) * 1e3
            res[tag] = (m, r)
        out[q] = res
    return centers, out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nevents", type=int, default=200_000)
    ap.add_argument("--nproc", type=int, default=24)
    args = ap.parse_args(argv)

    data = load_na61()
    p_edges = np.logspace(np.log10(0.3), np.log10(20.0), 16)
    centers, gen = urqmd_angle_moments(p_edges, args.nevents, args.nproc)

    channels = (("pi+", PIPLUS_TABLES, 211), ("pi-", PIMINUS_TABLES, -211))
    for label, tables, q in channels:
        _, mean_d, rms_d = na61_angle_moments(data, tables, p_edges)
        m_new, r_new = gen[q]["new"]
        m_old, r_old = gen[q]["old"]
        print(
            f"\n{label} production angle vs NA61 p+C 31 GeV/c "
            f"(theta < {THETA_ACC * 1e3:.0f} mrad), {args.nevents} events"
        )
        print(
            "  p_lab[GeV]  NA61<th>  UrQMD<th>   ratio | "
            "NA61 rms  UrQMD rms   ratio | old ratio(<th>/rms)"
        )
        rn, ro, rrn, rro = [], [], [], []
        for i, p in enumerate(centers):
            if not (np.isfinite(mean_d[i]) and np.isfinite(m_new[i])):
                continue
            rn.append(m_new[i] / mean_d[i])
            ro.append(m_old[i] / mean_d[i])
            rrn.append(r_new[i] / rms_d[i])
            rro.append(r_old[i] / rms_d[i])
            print(
                f"  {p:9.2f}  {mean_d[i]:8.1f}  {m_new[i]:9.1f}  {rn[-1]:6.3f} | "
                f"{rms_d[i]:8.1f}  {r_new[i]:9.1f}  {rrn[-1]:6.3f} | "
                f"{ro[-1]:6.3f} / {rro[-1]:.3f}"
            )
        print(
            f"  MEAN RATIO  <theta>: new {np.mean(rn):.3f}  old {np.mean(ro):.3f}"
            f"   |  rms: new {np.mean(rrn):.3f}  old {np.mean(rro):.3f}"
        )


if __name__ == "__main__":
    main()
