"""Independent test of the 3D machinery: two-generation toy against direct integration.

Problem (``kernelset.toy_cascade``): exponential atmosphere rho0 exp(-h/H) on a
sphere (R = 6391 km, top 80 km); parent P isotropic at the top for every down-going
direction, interaction length ``LAM`` [g/cm^2], no decay; each interaction gives one
child C with the angular law ``g delta + (1 - g) vMF(kappa)`` (density
exp(kappa cos theta) per solid angle); C does not interact and decays with length
``DC``; each decay gives one neutrino along C.  The neutrino intensity at a point
in direction Omega is

    Phi_nu = int_line q_C(s) (1 - exp(-(L - s)/DC)) ds,
    q_C(x, Omega) = rho(x)/LAM [g f_P(x, Omega) + (1-g) int dOmega' K f_P(x, Omega')],
    f_P(x, Omega') = exp(-X_back(x, Omega')/LAM), 0 if the backward ray hits the ground.

The reference below evaluates these integrals by Gauss-Legendre quadrature; it shares
no code with the solver (its own density, line geometry and angular integrals).  The
solver must agree to 1% at the ground (including within 5 degrees of the horizon) and
in up-going directions at altitude.

Run as a script to write ``$N3/c1/gates/TOY3D.md``::

    taskset -c 0-3 $PY tools/cascade3d/test_toy3d.py
"""

import os
import sys
import time

import numpy as np

R = 6391.0e5
TOP = 80.0e5
RHO0, HS = 1.225e-3, 7.0e5
LAM, DC = 90.0, 4.0e5
G, KAPPA = 0.25, 2.0
#: (altitude [cm], mu) of the test points; mu > 0 down-going
POINTS = [(0.0, 1.0), (0.0, 0.5), (0.0, 0.2), (0.0, np.cos(np.radians(85.0))),
          (0.0, 0.05), (0.0, 0.02), (15.0e5, 0.3), (15.0e5, 0.05),
          (15.0e5, -0.05), (15.0e5, -0.3), (30.0e5, -0.6)]
TOL = 0.01


# --------------------------------------------------------------- reference

def pinned():
    """The process affinity as a taskset list, e.g. "0-3"."""
    c = sorted(os.sched_getaffinity(0))
    parts, start = [], c[0]
    for a, b in zip(c, c[1:] + [None]):
        if b != a + 1:
            parts.append(f"{start}-{a}" if a > start else f"{a}")
            start = b
    return ",".join(parts)

def _rho(h):
    return np.where((h >= 0) & (h <= TOP), RHO0 * np.exp(-h / HS), 0.0)


def _gl(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1), 0.5 * w


def _grammage(b, t0, t1, n=96):
    """Grammage along a line of impact b between t0 < t1 (split at t = 0)."""
    x, w = _gl(n)
    out = np.zeros(np.broadcast(b, t0, t1).shape)
    for lo, hi in ((t0, np.minimum(t1, 0.0)), (np.maximum(t0, 0.0), t1)):
        ok = hi > lo
        lo_, hi_ = np.where(ok, lo, 0.0), np.where(ok, hi, 0.0)
        tt = lo_[..., None] + (hi_ - lo_)[..., None] * x
        h = np.sqrt(np.asarray(b)[..., None] ** 2 + tt ** 2) - R
        out += np.where(ok, np.sum(_rho(h) * w, axis=-1) * (hi_ - lo_), 0.0)
    return out


def f_parent(r, mu):
    """Parent intensity at radius r moving with direction cosine mu (broadcast)."""
    r, mu = np.broadcast_arrays(np.asarray(r, float), np.asarray(mu, float))
    b = r * np.sqrt(np.clip(1 - mu ** 2, 0, None))
    tp = -mu * r
    ts = -np.sqrt((R + TOP) ** 2 - b ** 2)
    ground = (mu < 0) & (b < R)
    X = _grammage(b, ts, tp)
    return np.where(ground, 0.0, np.exp(-X / LAM))


def _vmf(ct):
    return KAPPA / (4 * np.pi * np.sinh(KAPPA)) * np.exp(KAPPA * ct)


def q_child(r, mu, n_mu=240, n_phi=48, chunk=64):
    """Child emissivity per unit length at radius r, direction mu (arrays)."""
    r = np.atleast_1d(np.asarray(r, float))
    mu = np.broadcast_to(np.asarray(mu, float), r.shape)
    if r.size > chunk:
        return np.concatenate([q_child(r[i:i + chunk], mu[i:i + chunk], n_mu, n_phi,
                                       chunk) for i in range(0, r.size, chunk)])
    # parent directions: dense around the horizon (feature of f_P at mu' ~ 0)
    xs, ws = _gl(n_mu)
    pieces = [(-1.0, -0.2), (-0.2, -0.02), (-0.02, 0.0), (0.0, 0.02), (0.02, 0.2),
              (0.2, 1.0)]
    mup = np.concatenate([a + (b - a) * xs for a, b in pieces])
    wmu = np.concatenate([(b - a) * ws for a, b in pieces])
    xp, wp = _gl(n_phi)
    phi, wphi = np.pi * xp, np.pi * wp * 2.0  # [0, pi], symmetric
    fp = f_parent(r[:, None], mup[None, :])                     # (n, M)
    s = np.sqrt(np.clip(1 - mu ** 2, 0, None))
    sp_ = np.sqrt(np.clip(1 - mup ** 2, 0, None))
    ct = (mu[:, None, None] * mup[None, :, None]
          + s[:, None, None] * sp_[None, :, None] * np.cos(phi)[None, None, :])
    kern = np.sum(_vmf(ct) * wphi, axis=-1)                     # (n, M)
    smooth = np.sum(kern * fp * wmu, axis=-1)
    direct = f_parent(r, mu)
    h = r - R
    return _rho(h) / LAM * (G * direct + (1 - G) * smooth)


def reference_nu(h_d, mu_d, n_seg=60, n_gl=16):
    """Neutrino intensity (per unit P intensity) at altitude h_d, direction mu_d."""
    rd = R + h_d
    b = rd * np.sqrt(1 - mu_d ** 2)
    td = -mu_d * rd
    if mu_d < 0 and b < R:
        ts = np.sqrt(R ** 2 - b ** 2)  # from the ground, going up
    else:
        ts = -np.sqrt((R + TOP) ** 2 - b ** 2)
    L = td - ts
    # break points: geometric towards the detector, plus the tangent point
    d = np.concatenate([[0.0], DC * np.geomspace(1e-3, 50.0, n_seg)])
    d = d[d < L]
    tb = np.unique(np.concatenate([td - d, [ts], np.linspace(ts, td, n_seg)]))
    if ts < 0 < td:
        tb = np.unique(np.concatenate([tb, [0.0]]))
    tb = tb[(tb >= ts) & (tb <= td)]
    x, w = _gl(n_gl)
    t = (tb[:-1, None] + np.diff(tb)[:, None] * x).ravel()
    wt = (np.diff(tb)[:, None] * w).ravel()
    r = np.sqrt(b ** 2 + t ** 2)
    mu = -t / r
    q = q_child(r, mu)
    return float(np.sum(q * (1 - np.exp(-(td - t) / DC)) * wt))


# ------------------------------------------------------------------ solver
def solver_nu(levels_dh=0.5, n_ground=48, n_sub=4, L=32, refine_family=1,
              refine_levels=1, **toy_kw):
    """Same quantities from cascade3d."""
    import assemble
    import grid
    import kernelset as K
    import rays
    import solver as S

    atm = grid.ExponentialAtmosphere(RHO0, HS, TOP, R)
    lev = grid.make_levels(atm, spec=((np.inf, levels_dh),), refine=refine_levels)
    base = grid.make_levels(atm, spec=((np.inf, levels_dh),))
    mu0, ht = [], []
    for h, m in POINTS:
        r = R + h
        b = r * np.sqrt(1 - m * m)
        if b < R:
            mu0.append(np.sqrt(1 - (b / R) ** 2))
        else:
            ht.append(b - R)
    fam = grid.make_line_family(base, n_ground=n_ground, refine=refine_family,
                                extra_mu0=tuple(mu0) + grid.G13_COSZ_EDGES, r_e=R)
    fam = grid.LineFamily(fam.mu0, np.concatenate([fam.h_tangent, ht]), r_e=R)
    ry = rays.Rays(atm, lev, fam, n_sub=n_sub)
    kw = dict(lam_p=LAM, dec_c=1.0 / DC, phi_top=1.0, g=G, kappa=KAPPA, L=L)
    kw.update(toy_kw)
    ks = K.toy_cascade(**kw)
    sv = S.Solver(ry, ks)
    out = sv.solve()
    nu = np.asarray(assemble.sink_intensity(sv, out["emis"]))[0, 0]  # bin 0
    w = ks.e_widths
    res = []
    for h, m in POINTS:
        lvl = int(np.argmin(np.abs(ry.levels - h)))
        cand = np.flatnonzero((ry.node_level == lvl))
        j = cand[np.argmin(np.abs(ry.node_mu[cand] - m))]
        assert abs(ry.node_mu[j] - m) < 1e-9, (h, m, ry.node_mu[j])
        # per-energy intensity of bin 0 times its width = number; P had w[1]
        res.append(nu[j] * w[0] / w[1])
    return np.array(res), ry.summary()


def compare(**kw):
    t = time.time()
    ref = np.array([reference_nu(h, m) for h, m in POINTS])
    t_ref = time.time() - t
    t = time.time()
    sol, summ = solver_nu(**kw)
    t_sol = time.time() - t
    return ref, sol, summ, t_ref, t_sol


def test_toy3d():
    ref, sol, _, _, _ = compare()
    rel = sol / ref - 1
    assert np.all(np.abs(rel) < TOL), rel


def test_loss_parent_angular_decay():
    """Angular law on a channel whose parent is a loss species (muon -> nu path).

    C decays with 0.3 delta + 0.7 vMF(kappa = 3).  C as a hadron (segment transport;
    its moments validated by ``test_toy3d``) and C as a loss species (loss points,
    dedx = 0: the same physics) must agree; the angular law must matter.
    Measured (2026-11-19): agreement 2.2e-5; the law changes the 11 points by
    -17% to +114%.
    """
    had, _ = solver_nu(dec_g=0.3, dec_kappa=3.0)
    los, _ = solver_nu(dec_g=0.3, dec_kappa=3.0, c_loss=True)
    fwd, _ = solver_nu(c_loss=True)
    rel = np.abs(los / had - 1)
    assert rel.max() < 2e-3, rel
    assert np.max(np.abs(los / fwd - 1)) > 0.01


def main():
    ref, sol, summ, t_ref, t_sol = compare()
    rel = sol / ref - 1
    lines = ["# Independent 3D toy test (cascade3d, C1)", "",
             __doc__.split("Run as")[0].strip(), "",
             f"Parameters: rho0 {RHO0} g/cm3, H {HS / 1e5} km, top {TOP / 1e5} km, "
             f"lambda_P {LAM} g/cm2, decay length C {DC / 1e5} km, g {G}, "
             f"kappa {KAPPA}.  Solver grid: {summ}.", "",
             "| altitude [km] | mu (down > 0) | reference | solver | rel. diff |",
             "|---|---|---|---|---|"]
    for (h, m), a, b, c in zip(POINTS, ref, sol, rel):
        lines.append(f"| {h / 1e5:.0f} | {m:+.4f} | {a:.6e} | {b:.6e} | {c:+.3%} |")
    worst = int(np.argmax(np.abs(rel)))
    ok = bool(np.all(np.abs(rel) < TOL))
    lines += ["", f"Worst: {rel[worst]:+.3%} at h = {POINTS[worst][0] / 1e5:.0f} km, "
              f"mu = {POINTS[worst][1]:+.4f}.  Tolerance 1%: "
              f"**{'PASS' if ok else 'FAIL'}**.",
              f"Time: reference {t_ref:.0f} s, solver {t_sol:.0f} s.", "",
              f"Command: `taskset -c {pinned()} $PY tools/cascade3d/test_toy3d.py`"]
    n3 = os.environ.get("N3", "/home/pigrange/mc3d_work/runs/native3d")
    outdir = os.environ.get("C3D_OUTDIR", os.path.join(n3, "c1", "cycle2"))
    path = os.path.join(outdir, "gates", "TOY3D.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
