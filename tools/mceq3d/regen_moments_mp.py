"""Multiprocessing driver that reproduces the ``m_*.npz`` moments production.

This is the shared-memory equivalent of the SLURM job-array recipe in
``KERNEL_GENERATION.md`` (sections 3 and 6): the same generators (UrQMD-3.4 below
80 GeV, SIBYLL-2.3d above), the same energy grid, the same ``x_L`` binning and
the same 2e5 interactions per projectile energy -- but sharded across local
cores instead of batch tasks, and with all four secondary species histogrammed
from a **single** event sample (section 8), which makes it ~4x cheaper.

Same-energy shards are merged by summing ``counts``, ``sum_theta`` and
``sum_theta_sq`` -- the count-weighted merge section 6 warns about, not a naive
average of ``theta_sq``.

The production angle is ``theta = arcsin(p_T / p)``
(:func:`kernel_regeneration.production_angle`); the files this writes are the
``*_v2.npz`` set, distinct from the pre-2026-09 ``m_*.npz`` which used the
incorrect ``arctan(p_T / p)``.

Usage::

    python regen_moments_mp.py --nint 200000 --nproc 46 --suffix _v2
"""

from __future__ import annotations

import argparse
import os
import time
import warnings

import numpy as np

from kernel_regeneration import (
    ChromoMultiSource,
    KernelGrid,
    MASS,
    moments_from_batch,
)

SPECIES = ("piplus", "piminus", "Kplus", "Kminus")
# Names the downstream modules load (kinematic_kernel._MOMENTS).
OUTNAME = {
    "piplus": "m_spliced",
    "piminus": "m_piminus",
    "Kplus": "m_Kplus",
    "Kminus": "m_Kminus",
}

_SRC = None
_GRID = None


def _init(model, xl_edges, seed_base):
    global _SRC, _GRID
    warnings.simplefilter("ignore")
    _GRID = np.asarray(xl_edges)
    ident = os.getpid()
    _SRC = ChromoMultiSource(
        model=model, secondaries=SPECIES, seed=(seed_base + ident) % (2**31 - 1)
    )


def _run_shard(task):
    """One shard: (i_energy, e_proj, n_events) -> per-species raw sums."""
    i, e_proj, nev = task
    grid = KernelGrid(
        xl_edges=_GRID, pt_edges=np.linspace(0, 3, 2), proj_energies=np.array([e_proj])
    )
    batches = _SRC.generate_all(e_proj, nev, _GRID[0])
    out = {}
    for s in SPECIES:
        m = moments_from_batch(batches[s], grid, MASS[s], e_proj)
        out[s] = (m["counts"], m["sum_theta"], m["sum_theta_sq"])
    return i, nev, out


def run_model(model, energies, xl_edges, nint, nproc, shard_events, seed_base):
    """Return {species: moments dict} for one generator over ``energies``."""
    import multiprocessing as mp

    nxl = len(xl_edges) - 1
    nE = len(energies)
    # ``_a`` / ``_b`` are two statistically independent halves (shards split by
    # arrival parity). Their difference is a direct, assumption-free estimate of
    # the Monte-Carlo error on <theta^2> -- see ``--halves`` in the output file.
    def _blank():
        return dict(
            counts=np.zeros((nE, nxl)),
            s1=np.zeros((nE, nxl)),
            s2=np.zeros((nE, nxl)),
        )

    acc = {s: _blank() for s in SPECIES}
    half = {s: (_blank(), _blank()) for s in SPECIES}
    nint_done = np.zeros(nE)

    tasks = []
    for i, e in enumerate(energies):
        left = nint
        while left > 0:
            n = min(shard_events, left)
            tasks.append((i, float(e), int(n)))
            left -= n
    # longest (highest-energy UrQMD / SIBYLL) shards first for load balance
    tasks.sort(key=lambda t: -t[1])

    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(
        nproc, initializer=_init, initargs=(model, xl_edges, seed_base)
    ) as pool:
        for k, (i, nev, res) in enumerate(
            pool.imap_unordered(_run_shard, tasks, chunksize=1)
        ):
            nint_done[i] += nev
            h = k % 2
            for s in SPECIES:
                c, s1, s2 = res[s]
                for tgt in (acc[s], half[s][h]):
                    tgt["counts"][i] += c
                    tgt["s1"][i] += s1
                    tgt["s2"][i] += s2
            if (k + 1) % 25 == 0 or k + 1 == len(tasks):
                el = time.time() - t0
                eta = el / (k + 1) * (len(tasks) - k - 1)
                print(
                    f"  {model}: {k + 1}/{len(tasks)} shards, "
                    f"{el / 60:.1f} min, eta {eta / 60:.1f} min",
                    flush=True,
                )

    dxl = np.diff(xl_edges)
    xl_centers = np.sqrt(xl_edges[:-1] * xl_edges[1:])
    out = {}
    for s in SPECIES:
        c = acc[s]["counts"]
        nz = c > 0
        th_mean = np.full((nE, nxl), np.nan)
        th_sq = np.full((nE, nxl), np.nan)
        th_mean[nz] = acc[s]["s1"][nz] / c[nz]
        th_sq[nz] = acc[s]["s2"][nz] / c[nz]
        out[s] = dict(
            is_moments=True,
            proj_energies=np.asarray(energies, float),
            xl_edges=xl_edges,
            xl_centers=xl_centers,
            e_sec=xl_centers[None, :] * np.asarray(energies, float)[:, None],
            theta_mean=th_mean,
            theta_sq=th_sq,
            dndx=c / nint_done[:, None] / dxl[None, :],
            counts=c,
        )
        for tag, hh in zip(("a", "b"), half[s]):
            hc = hh["counts"]
            hnz = hc > 0
            t2 = np.full((nE, nxl), np.nan)
            t2[hnz] = hh["s2"][hnz] / hc[hnz]
            out[s][f"theta_sq_{tag}"] = t2
            out[s][f"counts_{tag}"] = hc
    return out


def merge_model_files(out_path, *paths):
    """Merge several per-model moment files produced by separate runs.

    Sums the raw ``counts`` and the reconstructed ``sum_theta`` /
    ``sum_theta_sq`` (count-weighted, as ``KERNEL_GENERATION.md`` section 6
    requires), and adds the per-energy interaction totals so ``dndx`` stays a
    per-interaction density. The two half-samples are also merged (run A halves
    into A, B into B), so the half-difference error estimate survives.
    """
    ds = [dict(np.load(p)) for p in paths]
    ref = ds[0]
    dxl = np.diff(ref["xl_edges"])
    for d in ds[1:]:
        assert np.allclose(d["proj_energies"], ref["proj_energies"])
        assert np.allclose(d["xl_edges"], ref["xl_edges"])

    def _nint(d):
        # dndx = counts / n_int / dxl  ->  n_int = counts / (dndx * dxl)
        num = d["counts"]
        den = d["dndx"] * dxl[None, :]
        ok = den > 0
        return np.array(
            [np.median((num[i][ok[i]] / den[i][ok[i]])) for i in range(num.shape[0])]
        )

    out = dict(ref)
    for tag in ("", "_a", "_b"):
        c = sum(d["counts" + tag] for d in ds)
        s2 = sum(d["counts" + tag] * np.nan_to_num(d["theta_sq" + tag]) for d in ds)
        nz = c > 0
        t2 = np.full(c.shape, np.nan)
        t2[nz] = s2[nz] / c[nz]
        out["counts" + tag] = c
        out["theta_sq" + tag] = t2
        if tag == "":
            s1 = sum(d["counts"] * np.nan_to_num(d["theta_mean"]) for d in ds)
            t1 = np.full(c.shape, np.nan)
            t1[nz] = s1[nz] / c[nz]
            out["theta_mean"] = t1
            n_int = sum(_nint(d) for d in ds)
            out["dndx"] = c / n_int[:, None] / dxl[None, :]
    np.savez(out_path, **out)
    print(f"merged {len(ds)} files -> {out_path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nint", type=int, default=200_000, help="interactions / energy")
    p.add_argument(
        "--nint-high",
        type=int,
        default=None,
        help="interactions / energy for SIBYLL (default: --nint). SIBYLL is "
        "~100x cheaper per event than UrQMD near 80 GeV, so it is usual to run "
        "it at the full 2e5 while trimming the low-energy model.",
    )
    p.add_argument("--nproc", type=int, default=46)
    p.add_argument("--shard-low", type=int, default=4000, help="events / UrQMD shard")
    p.add_argument(
        "--shard-high", type=int, default=20000, help="events / SIBYLL shard"
    )
    p.add_argument("--suffix", default="_v2")
    p.add_argument("--seed", type=int, default=20260903)
    p.add_argument("--only", choices=("low", "high", "both"), default="both")
    p.add_argument("--tmp", default=".", help="directory for the per-model files")
    args = p.parse_args(argv)

    xl_edges = np.logspace(-4, 0, 61)
    e_low = np.logspace(np.log10(4.0), np.log10(80.0), 12)
    e_high = np.logspace(np.log10(80.0), 6.0, 30)

    if args.only in ("low", "both"):
        t = time.time()
        low = run_model(
            "UrQMD34", e_low, xl_edges, args.nint, args.nproc, args.shard_low, args.seed
        )
        for s in SPECIES:
            np.savez(os.path.join(args.tmp, f"mom_{s}_low{args.suffix}.npz"), **low[s])
        print(f"UrQMD34 done in {(time.time() - t) / 60:.1f} min", flush=True)

    if args.only in ("high", "both"):
        t = time.time()
        high = run_model(
            "Sibyll23d",
            e_high,
            xl_edges,
            args.nint_high or args.nint,
            args.nproc,
            args.shard_high,
            args.seed + 7717,
        )
        for s in SPECIES:
            np.savez(
                os.path.join(args.tmp, f"mom_{s}_high{args.suffix}.npz"), **high[s]
            )
        print(f"Sibyll23d done in {(time.time() - t) / 60:.1f} min", flush=True)

    if args.only == "both":
        from splice_kernels import splice

        for s in SPECIES:
            splice(
                os.path.join(args.tmp, f"mom_{s}_low{args.suffix}.npz"),
                os.path.join(args.tmp, f"mom_{s}_high{args.suffix}.npz"),
                80.0,
                f"{OUTNAME[s]}{args.suffix}.npz",
            )


if __name__ == "__main__":
    main()
