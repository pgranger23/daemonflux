"""Quantify the pre-2026-09 production-angle bug on *identical* event samples.

Until 2026-09 ``kernel_regeneration`` computed the secondary production angle as
``arctan(p_T / p)`` with ``p = sqrt(E_sec^2 - m^2)`` the **total** momentum -- i.e.
it used the total momentum where the longitudinal one belongs. The correct
relation is ``sin(theta) = p_T / p``.

Because ``arctan(u) < arcsin(u)``, the old convention biased ``<theta^2>``
*low*, by an amount that grows as ``p_T`` becomes comparable to ``p``, i.e. at
low secondary energy -- exactly the sub-GeV region that sets the off-axis
production excess ``E_off``.

This script runs the production generators once and histograms **three**
conventions on the same secondaries, so the differences are pure systematics and
carry no Monte-Carlo scatter:

* ``old``    -- ``arctan(p_T / p)``, the bug;
* ``new``    -- ``arcsin(p_T / p)``, what the ``*_v2.npz`` moments now use;
* ``signed`` -- ``arctan2(p_T, p_z)`` with the generator's **signed** ``p_z``,
  the true space angle. It is not reproducible from ``(x_L, p_T)`` alone, so it
  is not what the moment files store; the ``signed/new`` column is the residual
  cost of folding backward-produced (target-fragmentation) secondaries into the
  forward hemisphere.

  Read the ``signed`` column only below ~1 GeV. Above that it is dominated by an
  unrelated artefact: ``ChromoSource.generate`` clips ``x_L`` up to the first bin
  edge (1e-4), so bin 0 of a high-``E_proj`` row is a pile-up of genuinely soft
  secondaries carrying the *nominal* energy ``1e-4 * E_proj``. Those include
  backward pions, and ``theta`` then reaches 180 deg instead of being capped at
  90 deg. Direct per-secondary measurements (no binning) give the honest number:
  the folding costs +11% in ``sqrt(<theta^2>)`` at ``E_proj = 4 GeV`` and +4% at
  ``E_proj = 80 GeV`` for ``E_sec`` in 0.3-1 GeV, and **nothing** above 1 GeV
  (UrQMD-3.4 and SIBYLL-2.3d both produce no backward secondary above 1 GeV).

It writes nothing.

Usage::

    python diag_angle_convention.py --nint 20000 --nproc 24
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np

from angular_kernel import pool_moments_by_energy
from kernel_regeneration import ChromoMultiSource, KernelGrid, MASS

SPECIES = ("piplus", "piminus", "Kplus", "Kminus")
E_REPORT = np.array([0.3, 0.5, 1.0, 2.0, 5.0, 10.0])

_SRC = None
_XL = None


def _init(model, xl_edges, seed):
    global _SRC, _XL
    warnings.simplefilter("ignore")
    _XL = np.asarray(xl_edges)
    _SRC = ChromoMultiSource(
        model=model, secondaries=SPECIES, seed=(seed + os.getpid()) % (2**31 - 1)
    )


def _shard(task):
    i, e_proj, nev = task
    batches = _SRC.generate_all(e_proj, nev, _XL[0])
    out = {}
    for s in SPECIES:
        b = batches[s]
        e_sec = b.x_L * e_proj
        p = np.sqrt(np.maximum(e_sec**2 - MASS[s] ** 2, 1e-12))
        pt = np.minimum(b.p_T, p)
        th_new = np.arcsin(np.clip(pt / p, 0.0, 1.0))  # correct, forward-folded
        th_old = np.arctan2(b.p_T, p)  # the bug
        # Reference: the true space angle from the generator's SIGNED p_z. Not
        # reproducible from (x_L, p_T) alone -- it is the residual cost of the
        # forward folding, dominated by target-fragmentation secondaries.
        th_sgn = np.arctan2(b.p_T, b.p_z)
        c, _ = np.histogram(b.x_L, bins=_XL)
        n1, _ = np.histogram(b.x_L, bins=_XL, weights=th_new)
        n2, _ = np.histogram(b.x_L, bins=_XL, weights=th_new**2)
        o1, _ = np.histogram(b.x_L, bins=_XL, weights=th_old)
        o2, _ = np.histogram(b.x_L, bins=_XL, weights=th_old**2)
        g1, _ = np.histogram(b.x_L, bins=_XL, weights=th_sgn)
        g2, _ = np.histogram(b.x_L, bins=_XL, weights=th_sgn**2)
        out[s] = (c.astype(float), n1, n2, o1, o2, g1, g2)
    return i, nev, out


def collect(model, energies, xl_edges, nint, nproc, shard, seed):
    import multiprocessing as mp

    nE, nxl = len(energies), len(xl_edges) - 1
    z = lambda: np.zeros((nE, nxl))  # noqa: E731
    acc = {s: [z() for _ in range(7)] for s in SPECIES}
    ndone = np.zeros(nE)
    tasks = []
    for i, e in enumerate(energies):
        left = nint
        while left > 0:
            n = min(shard, left)
            tasks.append((i, float(e), int(n)))
            left -= n
    tasks.sort(key=lambda t: -t[1])
    ctx = mp.get_context("fork")
    with ctx.Pool(nproc, initializer=_init, initargs=(model, xl_edges, seed)) as pool:
        for i, nev, res in pool.imap_unordered(_shard, tasks, chunksize=1):
            ndone[i] += nev
            for s in SPECIES:
                for k in range(7):
                    acc[s][k][i] += res[s][k]
    return acc, ndone


def to_moments(acc_s, energies, xl_edges, ndone, which="new"):
    nE, nxl = len(energies), len(xl_edges) - 1
    c, n1, n2, o1, o2, g1, g2 = acc_s
    s1, s2 = {"new": (n1, n2), "old": (o1, o2), "signed": (g1, g2)}[which]
    xl_c = np.sqrt(xl_edges[:-1] * xl_edges[1:])
    th_m = np.full((nE, nxl), np.nan)
    th_s = np.full((nE, nxl), np.nan)
    nz = c > 0
    th_m[nz] = s1[nz] / c[nz]
    th_s[nz] = s2[nz] / c[nz]
    return {
        "e_sec": xl_c[None, :] * np.asarray(energies, float)[:, None],
        "theta_mean": th_m,
        "theta_sq": th_s,
        "dndx": c / ndone[:, None] / np.diff(xl_edges)[None, :],
    }


def pooled_rms_deg(mom, e_query):
    """sqrt(<theta^2>) [deg] at ``e_query``, pooled the way the solver does."""
    e_edges = np.logspace(np.log10(0.25), np.log10(mom["e_sec"].max()), 40)
    p = pool_moments_by_energy(mom, e_edges)
    good = np.isfinite(p["theta_sq"]) & (p["theta_sq"] > 0)
    e, t2 = p["e_sec"][good], p["theta_sq"][good]
    lg = np.interp(np.log(e_query), np.log(e), np.log(t2))
    return np.degrees(np.sqrt(np.exp(lg)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nint", type=int, default=20000)
    ap.add_argument("--nproc", type=int, default=24)
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args(argv)

    xl_edges = np.logspace(-4, 0, 61)
    e_low = np.logspace(np.log10(4.0), np.log10(80.0), 12)
    e_high = np.logspace(np.log10(80.0), 6.0, 30)

    lo, nlo = collect(
        "UrQMD34", e_low, xl_edges, args.nint, args.nproc, 2000, args.seed
    )
    hi, nhi = collect(
        "Sibyll23d", e_high, xl_edges, args.nint, args.nproc, 5000, args.seed + 11
    )

    keep_lo = e_low < 80.0
    keep_hi = e_high >= 80.0

    def spl(a, b):
        return {
            k: np.concatenate([a[k][keep_lo], b[k][keep_hi]], axis=0)
            for k in ("e_sec", "theta_mean", "theta_sq", "dndx")
        }

    print("\nsqrt(<theta^2>)(E_sec) [deg], identical event samples")
    print("  species   E_sec  old arctan  new arcsin  new/old | signed p_z  /new")
    for s in SPECIES:
        v = {}
        for w in ("old", "new", "signed"):
            v[w] = pooled_rms_deg(
                spl(
                    to_moments(lo[s], e_low, xl_edges, nlo, w),
                    to_moments(hi[s], e_high, xl_edges, nhi, w),
                ),
                E_REPORT,
            )
        for i, e in enumerate(E_REPORT):
            print(
                f"  {s:8s} {e:6.1f}  {v['old'][i]:9.3f}   {v['new'][i]:9.3f}  "
                f"{v['new'][i] / v['old'][i]:6.3f} | {v['signed'][i]:9.3f} "
                f"{v['signed'][i] / v['new'][i]:6.3f}"
            )
        print()


if __name__ == "__main__":
    main()
