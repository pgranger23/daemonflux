"""Measure the intrinsic rigidity penumbra per direction, i.e. ``SIGMA_LNR``.

This is where ``mceq3d_flux.SIGMA_LNR`` (the width of the Eq. 7 cutoff
transmission) comes from: it is measured, not tuned.

For a FIXED direction the back-traced admittance A(R) is binary (0/1): the
penumbra is the chaotic alternation of allowed/forbidden bands between the
Stoermer (lower) cutoff R_L and the upper cutoff R_U.  The erf transmission
T_a(E;R_c) of PAPER_DRAFT Eq. 7 models that band with a smooth step of width
sigma in ln R, so the sigma that the physics supports is set by the ln-R extent
of the penumbral band and by how much allowed measure it contains.

Reported per direction:
  R_U      highest forbidden rigidity (what cutoff_map returns)
  R_L      lowest allowed rigidity in the scan window (= main/Stoermer cone edge)
  dbw      total ALLOWED rigidity measure strictly below R_U  [GV] and as a
           fraction of R_U (the "leakage" an erf tail must mimic)
  sig_band ln(R_U/R_L)/2   -- band edges mapped onto the erf 16%/84% points
  sig_16_84 from a smoothed admittance (window w_smooth), with the resolution
           floor for a perfect step quoted alongside
  sig_fit  least-squares erf fit (both centre and width free) to the smoothed
           admittance

A second pass repeats the scan for a small BUNDLE of directions around the axis
(+-0.5, +-1 deg) to separate the intrinsic penumbra from the apparent one
created by angular smearing -- the latter is already handled downstream by the
production-cone average, so it must NOT be folded into sigma as well.
Run from ``tools/mceq3d`` (needs ppigrf); ~30 min on 40 cores, the cost being
the eight coarse ``R_U`` scans (stage 1), not the fine ladder.
"""
import argparse, json, os, time
from datetime import datetime
import multiprocessing as mp
import numpy as np
import geomag_backtrace as gb

LAT, LON = 36.43, 137.31
DATE = datetime(2020, 1, 1)
OUT = os.environ.get("PENUMBRA_OUT", "")   # set to save the raw admittance

DIRS = [("vertical", 0.0, 0.0), ("60E", 60.0, 90.0), ("75E", 75.0, 90.0),
        ("81E", 81.0, 90.0), ("87E", 87.0, 90.0), ("87W", 87.0, 270.0),
        ("87N", 87.0, 0.0), ("87S", 87.0, 180.0)]

_G = {}

def _worker(args):
    i0, r0, u0, R = args
    return i0, gb.backtrace_vec(r0, u0, R, DATE, _G["m_hat"], +1)

def trace_pairs(dirs, Rs, n_jobs=32):
    """dirs: list of (zen, az); Rs: list of arrays (one per direction)."""
    up = gb._local_frame(LAT, LON)[0]
    r0_all, u0_all, R_all, owner = [], [], [], []
    for k, (z, a) in enumerate(dirs):
        u = -gb.arrival_direction(LAT, LON, z, a)
        for R in Rs[k]:
            r0_all.append(gb.RE * up)
            u0_all.append(u)
            R_all.append(R)
            owner.append(k)
    r0_all = np.array(r0_all)
    u0_all = np.array(u0_all)
    R_all = np.array(R_all)
    n = len(R_all)
    # interleave so every worker gets a mix of cheap (forbidden) and dear (allowed)
    order = np.argsort(np.arange(n) % n_jobs, kind="stable")
    chunks = np.array_split(order, n_jobs)
    args = [(c, r0_all[c], u0_all[c], R_all[c]) for c in chunks if len(c)]
    with mp.get_context("fork").Pool(len(args)) as pool:
        res = pool.map(_worker, args)
    A = np.zeros(n, bool)
    for idx, a in res:
        A[idx] = a
    owner = np.array(owner)
    return [(np.array(Rs[k]), A[owner == k]) for k in range(len(dirs))]


def smooth(R, A, w):
    """Running mean of the binary admittance over a +-w/2 GV window."""
    out = np.empty(len(R))
    for i, r in enumerate(R):
        m = np.abs(R - r) <= 0.5 * w
        out[i] = A[m].mean()
    return out


def cross(R, T, lvl):
    """Highest rigidity at which the smoothed T crosses lvl going upward."""
    hi = np.where(T >= lvl)[0]
    if not len(hi):
        return np.nan
    i = hi[0]
    lo = np.where(T[:i] < lvl)[0]
    if not len(lo):
        return float(R[i])
    j = lo[-1]
    if T[i] == T[j]:
        return float(R[i])
    return float(R[j] + (lvl - T[j]) * (R[i] - R[j]) / (T[i] - T[j]))


def erf_fit(R, T):
    from scipy.optimize import least_squares
    from scipy.special import erf
    x = np.log(R)
    def res(p):
        return 0.5 * (1 + erf((x - p[0]) / (np.sqrt(2) * np.exp(p[1])))) - T
    p0 = [np.log(R[np.argmin(np.abs(T - 0.5))]), np.log(0.05)]
    s = least_squares(res, p0, method="lm", max_nfev=5000)
    return float(np.exp(s.x[0])), float(np.exp(s.x[1]))


def analyse(label, R, A, w_smooth, verbose=True):
    o = np.argsort(R); R, A = R[o], A[o].astype(float)
    dR = np.gradient(R)
    forb = R[A < 0.5]
    R_U = float(forb.max()) if forb.size else float(R.min())
    allw = R[A >= 0.5]
    R_L = float(allw.min()) if allw.size else float(R.max())
    below = (R < R_U) & (A >= 0.5)
    dbw = float(np.sum(dR[below]))
    n_isl = int(np.sum(np.diff(np.concatenate([[0], below.astype(int)])) == 1))
    sig_band = 0.5 * np.log(R_U / max(R_L, 1e-9))
    T = smooth(R, A, w_smooth)
    r16, r84 = cross(R, T, 0.16), cross(R, T, 0.84)
    sig_1684 = 0.5 * np.log(r84 / r16) if np.isfinite(r16 * r84) and r16 > 0 else np.nan
    try:
        rc_fit, sig_fit = erf_fit(R, T)
    except Exception:
        rc_fit, sig_fit = np.nan, np.nan
    # resolution floor: what a PERFECT step at R_U returns under the same pipeline
    Astep = (R > R_U).astype(float)
    Ts = smooth(R, Astep, w_smooth)
    f16, f84 = cross(R, Ts, 0.16), cross(R, Ts, 0.84)
    sig_floor = 0.5 * np.log(f84 / f16)
    return dict(label=label, R_U=R_U, R_L=R_L, dbw=dbw, dbw_frac=dbw / R_U,
                islands=n_isl, sig_band=float(sig_band), sig_1684=float(sig_1684),
                sig_floor=float(sig_floor), rc_fit=rc_fit, sig_fit=sig_fit,
                n=len(R))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--span", type=float, default=0.45,
                    help="scan window low edge as a fraction below R_U")
    ap.add_argument("--n-jobs", type=int, default=32)
    ap.add_argument("--smooth", type=float, default=1.0, help="GV smoothing window")
    ap.add_argument("--bundle", action="store_true")
    a = ap.parse_args()
    _G["m_hat"] = gb.dipole_axis()

    t0 = time.time()
    # stage 1: cheap R_U per direction (coarse ladder + bisection, as delivered)
    RU0 = {}
    for lab, z, az in DIRS:
        rc = gb.cutoff_map(LAT, LON, DATE, [z], [az], n_jobs=4, warn_saturated=False)
        RU0[lab] = float(rc[0, 0])
    print("stage1 R_U (delivered scheme): " +
          " ".join(f"{k}={v:.2f}" for k, v in RU0.items()), flush=True)
    print(f"  {time.time()-t0:.0f}s", flush=True)

    # stage 2: fine ladder around each R_U
    Rs, dirs = [], []
    for lab, z, az in DIRS:
        ru = RU0[lab]
        lo = max(0.3, ru * (1.0 - a.span))
        hi = min(gb.RC_MAX_GV, ru + 2.0)
        Rs.append(np.arange(lo, hi + 1e-9, a.step))
        dirs.append((z, az))
    npair = sum(len(r) for r in Rs)
    print(f"stage2: {npair} traces, step {a.step} GV", flush=True)
    t1 = time.time()
    res = trace_pairs(dirs, Rs, a.n_jobs)
    print(f"  {time.time()-t1:.0f}s", flush=True)

    rows = []
    store = {}
    for (lab, z, az), (R, A) in zip(DIRS, res):
        d = analyse(lab, R, A, a.smooth)
        d.update(zen=z, az=az, R_U_coarse=RU0[lab])
        rows.append(d)
        store[lab + "_R"] = R
        store[lab + "_A"] = A.astype(float)
    hdr = (f"{'dir':>9} {'zen':>5} {'az':>5} {'R_U':>7} {'R_L':>7} "
           f"{'dbw[GV]':>8} {'dbw/R_U':>8} {'isl':>4} {'sig_band':>9} "
           f"{'sig16_84':>9} {'floor':>7} {'sig_fit':>8}")
    print("\n" + hdr)
    for d in rows:
        print(f"{d['label']:>9} {d['zen']:5.0f} {d['az']:5.0f} {d['R_U']:7.2f} "
              f"{d['R_L']:7.2f} {d['dbw']:8.2f} {d['dbw_frac']:8.4f} "
              f"{d['islands']:4d} {d['sig_band']:9.4f} {d['sig_1684']:9.4f} "
              f"{d['sig_floor']:7.4f} {d['sig_fit']:8.4f}")

    if a.bundle:
        print("\nBUNDLE (angular smearing, NOT intrinsic): 87E +-0.5,+-1 deg zenith "
              "and azimuth")
        offs = [(-1.0, 0.0), (-0.5, 0.0), (0.0, 0.0), (0.5, 0.0), (1.0, 0.0),
                (0.0, -1.0), (0.0, -0.5), (0.0, 0.5), (0.0, 1.0)]
        for base_lab, bz, ba in (("87E", 87.0, 90.0), ("vertical", 0.0, 0.0)):
            ru = RU0[base_lab]
            lo = max(0.3, ru * (1.0 - a.span)); hi = min(gb.RC_MAX_GV, ru + 2.0)
            grid = np.arange(lo, hi + 1e-9, a.step)
            bdirs = [(bz + dz, (ba + daz) % 360.0) for dz, daz in offs]
            r2 = trace_pairs(bdirs, [grid] * len(bdirs), a.n_jobs)
            Ab = np.mean([x[1].astype(float) for x in r2], axis=0)
            T = Ab  # already in [0,1]
            r16, r84 = cross(grid, T, 0.16), cross(grid, T, 0.84)
            sg = 0.5 * np.log(r84 / r16)
            try:
                rcf, sf = erf_fit(grid, T)
            except Exception:
                rcf, sf = np.nan, np.nan
            print(f"  {base_lab}: bundle-averaged R16={r16:.2f} R84={r84:.2f} "
                  f"sigma_1684={sg:.4f}  erf-fit Rc={rcf:.2f} sigma={sf:.4f}")
            store[f"bundle_{base_lab}_R"] = grid
            store[f"bundle_{base_lab}_A"] = Ab

    if OUT:
        np.savez(os.path.join(OUT, "penumbra.npz"), **store)
        with open(os.path.join(OUT, "penumbra.json"), "w") as f:
            json.dump(rows, f, indent=1)
    print(f"\ntotal {time.time()-t0:.0f}s\nSCAN_PENUMBRA_DONE")


if __name__ == "__main__":
    main()
