"""Milestone-3b studies (``python study_m3b.py <study> --outdir DIR``).

``bend``    the primary bend between injection and the first interaction vs
            rigidity and injection zenith, and the error of reusing a cascade
            by rotation: the angle between the rotated base copy's
            first-interaction direction and each copy's own (exact) one.
``muoncost`` the CPU of the exact per-copy muon re-transport against the
            cascade, on c-files of a run.
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np

import bend
import cutoff
import geometry as geo
from constants import R_EARTH_CM

LAT, LON = 36.4267, 137.31
R_INJ = R_EARTH_CM + 100.0e5


def cap_pointing_states(seed, n, cap_deg=10.0, theta_inj=40.0):
    """Injection states whose straight ray reaches the ground inside the cap."""
    rng = np.random.default_rng(seed)
    axis = cutoff.site_axis(LAT, LON)
    out_r, out_u = [], []
    have = 0
    while have < n:
        r, u = geo.sample_injection_patch(rng, R_INJ, axis, theta_inj, 40 * n)
        h = cutoff.first_ground_crossing(r, u)
        ok = np.isfinite(h[:, 0])
        ca = np.full(len(r), -2.0)
        ca[ok] = (h[ok] @ axis) / np.linalg.norm(h[ok], axis=1)
        keep = np.where(ca >= math.cos(math.radians(cap_deg)))[0][: n - have]
        out_r.append(r[keep])
        out_u.append(u[keep])
        have += len(keep)
    return np.concatenate(out_r), np.concatenate(out_u), rng


def study_bend(outdir, n=3000, n_az=12, seed=5, lam=85.0, **_):
    """Bend and rotation error vs R (log-uniform 2-300 GV) and zenith."""
    import field
    fld = field.FieldGrid(cache_dir=os.environ.get("MC3D_FIELD_CACHE"))
    r0, u0, rng = cap_pointing_states(seed, n)
    R = np.exp(rng.uniform(np.log(2.0), np.log(300.0), n))
    dX = -np.log(rng.random(n)) * lam
    axis = cutoff.site_axis(LAT, LON)
    rr, uu = cutoff.copy_states(r0, u0, axis, n_az, 0.0)
    st, ri, ui, L, b = bend.bend_primary(rr.reshape(-1, 3), uu.reshape(-1, 3),
                                         np.repeat(R, n_az), np.repeat(dX, n_az),
                                         fld)
    st = st.reshape(n, n_az)
    ri = ri.reshape(n, n_az, 3)
    ui = ui.reshape(n, n_az, 3)
    b = b.reshape(n, n_az)
    # rotation reuse: rotate copy 0's interaction state into copy j
    shifts = 360.0 * np.arange(n_az) / n_az
    eps = np.zeros((n, n_az))
    dpos = np.zeros((n, n_az))
    for j in range(n_az):
        ur = cutoff.rotate_about(ui[:, 0], axis, -math.radians(shifts[j]))
        rr_ = cutoff.rotate_about(ri[:, 0], axis, -math.radians(shifts[j]))
        eps[:, j] = np.arccos(np.clip(np.einsum("ij,ij->i", ur, ui[:, j]),
                                      -1, 1))
        dpos[:, j] = np.linalg.norm(rr_ - ri[:, j], axis=1) * 1e-5
    ok = (st == bend.P_INTERACT).all(1)
    mu = -np.einsum("ij,ij->i", u0, r0) / np.linalg.norm(r0, axis=1)
    zen = np.degrees(np.arccos(mu))
    lines = [f"# primary bend and rotation-reuse error, {n} cap-pointing "
             f"states x {n_az} copies, dX = Exp x {lam} g/cm2, IGRF-14 grid",
             "zen_band R_band n | bend median p90 [deg] | eps(copies>0) median "
             "p90 p99 [deg] | dpos median p90 [km] | L median [km]"]
    redges = (2, 5, 10, 15, 20, 30, 45, 60, 100, 300)
    tab = []
    for z0, z1 in ((0, 45), (45, 70), (70, 90)):
        for r_lo, r_hi in zip(redges[:-1], redges[1:]):
            m = ok & (zen >= z0) & (zen < z1) & (R >= r_lo) & (R < r_hi)
            if m.sum() < 5:
                continue
            bb = np.degrees(b[m]).ravel()
            ee = np.degrees(eps[m][:, 1:]).ravel()
            dd = dpos[m][:, 1:].ravel()
            row = (z0, z1, r_lo, r_hi, int(m.sum()), np.median(bb),
                   np.percentile(bb, 90), np.median(ee), np.percentile(ee, 90),
                   np.percentile(ee, 99), np.median(dd), np.percentile(dd, 90),
                   np.median(L.reshape(n, n_az)[m]) * 1e-5)
            tab.append(row)
            lines.append("%2d-%2d %5.0f-%-5.0f %4d | %6.2f %6.2f | %6.2f %6.2f "
                         "%6.2f | %6.2f %6.2f | %5.0f" % row)
    # all zeniths, per R band, the quantity that sets the bias
    lines.append("all zeniths:")
    for r_lo, r_hi in zip(redges[:-1], redges[1:]):
        m = ok & (R >= r_lo) & (R < r_hi)
        ee = np.degrees(eps[m][:, 1:]).ravel()
        bb = np.degrees(b[m]).ravel()
        lines.append("  R %5.0f-%-5.0f n %4d bend med %5.2f p90 %5.2f | eps med "
                     "%5.2f p90 %5.2f p99 %5.2f" % (
                         r_lo, r_hi, m.sum(), np.median(bb),
                         np.percentile(bb, 90), np.median(ee),
                         np.percentile(ee, 90), np.percentile(ee, 99)))
    # R * eps is ~constant if the bend is ~1/R: fit
    m = ok & (R > 5)
    k = np.median((np.degrees(eps[m][:, 1:]) * R[m][:, None]).ravel())
    lines.append(f"median eps x R = {k:.1f} deg GV (eps ~ {k:.0f}/R deg)")
    txt = "\n".join(lines)
    print(txt)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "bend.txt"), "w") as fh:
        fh.write(txt + "\n")
    np.savez(os.path.join(outdir, "bend.npz"), R=R, zen=zen, bend=b, eps=eps,
             dpos=dpos, st=st, tab=np.array(tab))
    return tab


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("study", choices=["bend"])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n", type=int, default=3000)
    a = ap.parse_args(argv)
    {"bend": study_bend}[a.study](a.outdir, n=a.n)


if __name__ == "__main__":
    main()
