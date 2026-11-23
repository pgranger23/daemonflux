"""Statistics of gate G2.3 (C2): one definition, used for the Monte Carlo reference,
for the null counts (`null_count_c2.py`) and for the comparison with the solver.

Tallies are arrays ``(..., species, band, cos Z row, azimuth bin)`` in the flux
measure: ``s`` with the cutoff, ``n`` with no cutoff.  The no-cutoff flux of every
azimuth bin is the azimuth mean of ``n`` (the Monte Carlo fills ``n`` from copy 0 of
each primary only; with no bending the twelve copies are rotations of it).  A solver
enters through :func:`solver_tallies`, so that both sides go through the same code.

The number of runs is free (the reference of G2.3 has twenty).  Gate texts:
``GATES.md``, section C2.  Nothing here reads a file.
"""

import numpy as np

#: tolerance floors of G2.3 (a), (b), (c) and of the level trigger (e)
FLOOR_A = 0.03
FLOOR_B = 0.01
FLOOR_C = 0.02
FLOOR_E = 0.02
#: cos Z rows of "horizontal" (0 to 0.2) and "vertical" (0.8 to 1), of ten rows 0 to 1
ROWS_H = (0, 1)
ROWS_V = (8, 9)
#: a cell enters part (a) if `s` is filled in at least this fraction of the runs
MIN_RUN_FRACTION = 0.5


def az_centres(n_az):
    """Centres of the azimuth bins [rad]; azimuth from north through east."""
    return np.deg2rad((np.arange(n_az) + 0.5) * 360.0 / n_az)


def solver_tallies(phi_cut, phi_0):
    """Solver cells as a pair of tallies: ``phi_cut`` (species, band, cz, az) with the
    cutoff, ``phi_0`` (species, band, cz) with the transmission set to 1."""
    phi_cut = np.asarray(phi_cut, float)
    n = np.repeat(np.asarray(phi_0, float)[..., None], phi_cut.shape[-1], axis=-1)
    return phi_cut, n


def stat_k(s, n):
    """(a) Cutoff factor per cell, K = s / <n>_az."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return s / n.mean(-1, keepdims=True)


def stat_kbar(s, n):
    """Azimuth-averaged cutoff factor per (species, band, cos Z)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return s.sum(-1) / n.sum(-1)


def stat_harm(s, n=None):
    """(b) First azimuthal harmonic of the flux with the cutoff, the species summed:
    (..., band, cz, 2) holding sum(Phi cos az) / sum(Phi) and the same with sin az."""
    f = s.sum(-4)
    az = az_centres(f.shape[-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        tot = f.sum(-1)
        return np.stack([(f * np.cos(az)).sum(-1) / tot,
                         (f * np.sin(az)).sum(-1) / tot], -1)


def _kband(s, n, rows):
    rows = list(rows)
    return s[..., rows, :].sum((-4, -2, -1)) / n[..., rows, :].sum((-4, -2, -1))


def stat_hv(s, n):
    """(c) Horizontal over vertical of the cutoff factor per band, species, azimuth
    and the two rows of each side summed."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return _kband(s, n, ROWS_H) / _kband(s, n, ROWS_V)


def stat_level(s, n):
    """(e) Cutoff factor per band, everything else summed."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return s.sum((-4, -2, -1)) / n.sum((-4, -2, -1))


def stat_flux(s, n=None):
    """(d) The flux with the cutoff itself."""
    return s


STATS = {"a": stat_k, "b": stat_harm, "c": stat_hv, "d": stat_flux, "e": stat_level,
         "f": stat_kbar}


def jackknife(fn, s_runs, n_runs):
    """Value on the mean of the runs and its delete-one-run jackknife error."""
    m = s_runs.shape[0]
    ts, tn = s_runs.sum(0), n_runs.sum(0)
    val = fn(ts / m, tn / m)
    loo = np.array([fn((ts - s_runs[j]) / (m - 1), (tn - n_runs[j]) / (m - 1))
                    for j in range(m)])
    err = np.sqrt((m - 1.0) / m * ((loo - loo.mean(0)) ** 2).sum(0))
    return val, err


def used_cells(s_runs):
    """Cells of part (a) and (d): `s` filled in at least half of the runs."""
    return (s_runs > 0).sum(0) >= MIN_RUN_FRACTION * s_runs.shape[0]


def inside(part, test, ref, err):
    """Boolean array: is the tested value inside the tolerance of the part?

    a, d, f: |test - ref| <= max(3% of ref, 2 err), per cell.
    b: both harmonic components within max(0.01, 2 err), per (band, row).
    c: |test - ref| <= max(2% of ref, 2 err), per band.
    e: as c (the trigger fires where this is False).
    """
    d = np.abs(test - ref)
    if part in ("a", "d", "f"):
        return d <= np.maximum(FLOOR_A * np.abs(ref), 2 * err)
    if part == "b":
        return (d <= np.maximum(FLOOR_B, 2 * err)).all(-1)
    if part == "c":
        return d <= np.maximum(FLOOR_C * np.abs(ref), 2 * err)
    if part == "e":
        return d <= np.maximum(FLOOR_E * np.abs(ref), 2 * err)
    raise ValueError(part)


def reference(s_runs, n_runs):
    """All reference statistics: {part: (value, error, mask of what is counted)}."""
    out = {}
    cells = used_cells(s_runs)
    for part, fn in STATS.items():
        val, err = jackknife(fn, s_runs, n_runs)
        ok = np.isfinite(val) & np.isfinite(err)
        if part == "b":
            ok = ok.all(-1)
        elif part in ("a", "d"):
            ok &= cells
        elif part == "f":
            ok &= cells.all(-1)
        out[part] = (val, err, ok)
    return out


def count(part, test, ref):
    """Number of counted items of the part that are inside."""
    val, err, ok = ref[part]
    with np.errstate(invalid="ignore"):
        ins = inside(part, test, val, err)
    return int((ins & ok).sum())


def n5(counts):
    """Largest n with P(count < n) <= 0.05: an exactly right solver reaches n or more
    in at least 95% of the cases."""
    counts = np.sort(np.asarray(counts))
    n = int(counts.max())
    while n > 0 and (counts < n).mean() > 0.05:
        n -= 1
    return n


def null_gauss(part, ref, nsamp, rng):
    """Counts of an exactly right solver, independent Gaussian errors of the size of the
    jackknife error on every counted item."""
    val, err, ok = ref[part]
    out = np.empty(nsamp, int)
    for i in range(nsamp):
        test = val + err * rng.standard_normal(val.shape)
        with np.errstate(invalid="ignore"):
            out[i] = (inside(part, test, val, err) & ok).sum()
    return out


def null_resample(parts, ref, s_runs, n_runs, nsamp, rng):
    """Counts of an exactly right solver, the runs resampled with replacement (keeps
    the correlations between cells): {part: counts}."""
    m = s_runs.shape[0]
    sm, nm = s_runs.mean(0), n_runs.mean(0)
    f = np.sqrt(m / (m - 1.0))
    out = {p: np.empty(nsamp, int) for p in parts}
    for i in range(nsamp):
        idx = rng.integers(0, m, m)
        sb = sm + f * (s_runs[idx].mean(0) - sm)
        nb = nm + f * (n_runs[idx].mean(0) - nm)
        for p in parts:
            val, err, ok = ref[p]
            with np.errstate(invalid="ignore", divide="ignore"):
                out[p][i] = (inside(p, STATS[p](sb, nb), val, err) & ok).sum()
    return out


def null_split(parts, ref, s_runs, n_runs, nsamp, rng):
    """Counts of an exactly right solver, half of the runs against the other half
    (amendment of 2026-11-23 in ``GATES.md``): {part: counts}.

    For a random half A of the runs and its complement B, d = (stat(A) - stat(B)) / 2
    stands for the deviation of the reference from the truth: it has the same variance
    and it keeps the correlations between the items.  The error that goes with it is
    the jackknife error with d taken out, e^2 = (nu err^2 - d^2) / (nu - 1), nu = runs
    - 1; for a mean of Gaussian runs e is independent of d and has nu - 1 degrees of
    freedom.  A draw counts a solver at the reference value against a reference at
    value + d with error e.  The number of runs must be even.
    """
    m = s_runs.shape[0]
    if m % 2 or m < 4:
        raise ValueError("null_split needs an even number of runs, at least four")
    h, nu = m // 2, m - 1
    ts, tn = s_runs.sum(0), n_runs.sum(0)
    out = {p: np.empty(nsamp, int) for p in parts}
    for i in range(nsamp):
        idx = rng.permutation(m)[:h]
        sa, na = s_runs[idx].sum(0), n_runs[idx].sum(0)
        for p in parts:
            val, err, ok = ref[p]
            with np.errstate(invalid="ignore", divide="ignore"):
                d = (STATS[p](sa / h, na / h)
                     - STATS[p]((ts - sa) / h, (tn - na) / h)) / 2
                e = np.sqrt(np.clip((nu * err**2 - d**2) / (nu - 1), 0, None))
                out[p][i] = (inside(p, val, val + d, e) & ok).sum()
    return out
