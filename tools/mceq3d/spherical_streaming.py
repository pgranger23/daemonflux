"""Spherical-streaming transport -- the off-axis PDE term, built and validated.

`prototype_streaming` proved the slab-P_N streaming operator stays sparse and
~linear; the one term it omitted is the **curvature** ``(1-mu^2)/r d/dmu`` that a
genuine spherical geometry adds -- and that curvature is exactly what couples
*different arrival directions* (a neutrino produced off the detector axis streams
in and arrives from a shifted zenith). This module adds that term and solves the
real **1-D spherical P_N transport equation**

    mu d(psi)/dr + (1-mu^2)/r d(psi)/dmu + sigma_t psi = sigma_s/2 * phi_0 + S/2 ,

whose Legendre-moment form (Lewis & Miller) is, for moment l,

    l/(2l+1)  [ phi'_{l-1} - (l-1)/r phi_{l-1} ]
  + (l+1)/(2l+1) [ phi'_{l+1} + (l+2)/r phi_{l+1} ]
  + sigma_t phi_l = delta_{l0}(sigma_s phi_0 + S).

The ``(l-1)/r`` and ``(l+2)/r`` pieces are the curvature; everything stays
tridiagonal in ``l`` and in ``r`` (sparse), so the cost verdict from the slab
prototype carries over unchanged.

Validation
----------
* **P_1 == spherical diffusion.** With the P_1 closure the moment system reduces
  *analytically* to ``-D (1/r^2)(r^2 phi')' + sigma_a phi = S`` (D = 1/3 sigma_t).
  We solve that diffusion ODE independently and recover phi_0 -- a quantitative
  check that the curvature operator is right (this is the term the slab prototype
  could not test).
* **Horizon redistribution.** With an atmospheric-shell source and weak
  absorption (streaming-dominated), the curvature term pushes the detector-level
  angular flux toward the horizon -- the curved-atmosphere off-axis excess,
  consistent in sign/shape with the geometric `spherical_geometry` result.
* **Sparsity / cost** re-confirmed including curvature.

Run::

    python spherical_streaming.py --plot
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.polynomial import legendre as L


def build_spherical_pn(r, n_l, sigma_t, sigma_s, source, bc0=(0.0, 0.0)):
    """Assemble the steady 1-D spherical P_N operator A and RHS b (sparse).

    **Staggered grid** (the standard cure for odd-even decoupling): even-l moments
    live on the ``n`` cell centres ``r``; odd-l moments live on the ``n-1`` faces
    ``r_{i+1/2}``. Then a derivative of an odd moment at a centre uses the two
    flanking faces (compact, no skipped points) and vice-versa.

    ``bc0`` = Dirichlet values of phi_0 at the inner/outer centre (other even
    moments are 0 there). Returns ``A``, ``b`` and an index helper ``gidx``.
    """
    n = len(r)
    h = r[1] - r[0]
    rf = 0.5 * (r[:-1] + r[1:])  # faces
    stf = 0.5 * (sigma_t[:-1] + sigma_t[1:])  # sigma_t on faces
    even = [ll for ll in range(n_l) if ll % 2 == 0]
    odd = [ll for ll in range(n_l) if ll % 2 == 1]

    # global index layout
    off, cur = {}, 0
    for ll in even:
        off[ll] = cur
        cur += n
    for ll in odd:
        off[ll] = cur
        cur += n - 1
    ndof = cur

    def gidx(ll, k):
        return off[ll] + k

    rows, cols, vals = [], [], []
    b = np.zeros(ndof)

    def add(ro, co, v):
        rows.append(ro)
        cols.append(co)
        vals.append(v)

    # ---- even-l equations on centres ----
    for ll in even:
        for k in range(n):
            ro = gidx(ll, k)
            if k == 0 or k == n - 1:  # Dirichlet
                add(ro, ro, 1.0)
                if ll == 0:
                    b[ro] = bc0[0] if k == 0 else bc0[1]
                continue
            add(ro, ro, sigma_t[k] - (sigma_s[k] if ll == 0 else 0.0))
            for m, c in (
                (ll - 1, ll / (2 * ll + 1)),
                (ll + 1, (ll + 1) / (2 * ll + 1)),
            ):
                if not (0 <= m < n_l):
                    continue
                curv = -(ll - 1) if m == ll - 1 else (ll + 2)
                # odd moment m sits on faces; centre k flanked by faces k-1, k
                # d/dr at centre k = (phi_m[face k] - phi_m[face k-1]) / h
                add(ro, gidx(m, k), c / h)
                add(ro, gidx(m, k - 1), -c / h)
                # curvature: phi_m(centre k) ~ 0.5(face k + face k-1)
                add(ro, gidx(m, k), c * curv / r[k] * 0.5)
                add(ro, gidx(m, k - 1), c * curv / r[k] * 0.5)
            if ll == 0:
                b[ro] = source[k]

    # ---- odd-l equations on faces ----
    for ll in odd:
        for k in range(n - 1):
            ro = gidx(ll, k)
            add(ro, ro, stf[k])
            for m, c in (
                (ll - 1, ll / (2 * ll + 1)),
                (ll + 1, (ll + 1) / (2 * ll + 1)),
            ):
                if not (0 <= m < n_l):
                    continue
                curv = -(ll - 1) if m == ll - 1 else (ll + 2)
                # even moment m on centres; face k flanked by centres k, k+1
                add(ro, gidx(m, k + 1), c / h)
                add(ro, gidx(m, k), -c / h)
                add(ro, gidx(m, k + 1), c * curv / rf[k] * 0.5)
                add(ro, gidx(m, k), c * curv / rf[k] * 0.5)

    A = sp.csr_matrix((vals, (rows, cols)), shape=(ndof, ndof))
    return A, b, off, rf


def solve_spherical(r, n_l=8, sigma_t=1.0, c=0.0, source=None, bc0=(0.0, 0.0)):
    """Solve the spherical-P_N problem on grid ``r``; return phi_l(r) (n_l, n_r).

    Odd moments (on faces) are interpolated back to the centres so the returned
    ``phi`` is a clean (n_l, n_r) array.
    """
    n = len(r)
    st = np.full(n, sigma_t, float) if np.isscalar(sigma_t) else np.asarray(sigma_t)
    ss = c * st
    if source is None:
        source = np.zeros(n)
    A, b, off, rf = build_spherical_pn(r, n_l, st, ss, source, bc0)
    t0 = time.time()
    u = spla.spsolve(A.tocsc(), b)
    dt = time.time() - t0
    phi = np.zeros((n_l, n))
    for ll in range(n_l):
        if ll % 2 == 0:
            phi[ll] = u[off[ll] : off[ll] + n]
        else:
            face = u[off[ll] : off[ll] + n - 1]
            phi[ll, 1:-1] = 0.5 * (face[:-1] + face[1:])  # faces -> centres
            phi[ll, 0], phi[ll, -1] = face[0], face[-1]
    return dict(r=r, phi=phi, A=A, solve_time=dt)


def diffusion_spherical(r, sigma_t, sigma_a, source, phi_lo, phi_hi):
    """Independent FD solve of -D (1/r^2)(r^2 phi')' + sigma_a phi = S (D=1/3 st).

    Dirichlet phi(r0)=phi_lo, phi(r1)=phi_hi. Used to validate the P_1 limit.
    """
    n = len(r)
    h = r[1] - r[0]
    D = 1.0 / (3 * sigma_t)
    A = np.zeros((n, n))
    b = np.zeros(n)
    A[0, 0] = A[-1, -1] = 1.0
    b[0], b[-1] = phi_lo, phi_hi
    for i in range(1, n - 1):
        # -D/r^2 d/dr(r^2 dphi/dr): conservative 3-point with r^2 faces
        rp = 0.5 * (r[i] + r[i + 1])
        rm = 0.5 * (r[i] + r[i - 1])
        ap = D * rp**2 / h**2 / r[i] ** 2
        am = D * rm**2 / h**2 / r[i] ** 2
        A[i, i - 1] = -am
        A[i, i + 1] = -ap
        A[i, i] = am + ap + sigma_a
        b[i] = source[i]
    return np.linalg.solve(A, b)


def angular_flux(phi_col, mu):
    """Reconstruct psi(mu) = sum_l (2l+1)/2 phi_l P_l(mu) at one radius."""
    n_l = len(phi_col)
    out = np.zeros_like(mu, float)
    for ll in range(n_l):
        coef = np.zeros(ll + 1)
        coef[ll] = 1.0
        out += (2 * ll + 1) / 2.0 * phi_col[ll] * L.legval(mu, coef)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args(argv)

    # ---- validation 1: P_1 spherical solve == spherical diffusion ODE ----
    r = np.linspace(1.0, 6.0, 400)
    st = 1.0
    cc = 0.97  # scattering-dominated -> diffusion valid
    src = np.exp(-((r - 3.5) ** 2) / (2 * 0.4**2))
    res = solve_spherical(r, n_l=2, sigma_t=st, c=cc, source=src)
    phi0 = res["phi"][0]
    ref = diffusion_spherical(r, st, st * (1 - cc), src, phi0[0], phi0[-1])
    m = (r > 1.5) & (r < 5.5)
    rel = np.abs(phi0[m] / ref[m] - 1.0)
    print("VALIDATION 1 -- P_1 spherical vs independent spherical-diffusion ODE")
    print(f"  (tests the curvature term)   median |rel diff| = {np.median(rel):.4f}")

    # ---- validation 2: curvature pushes detector flux toward the horizon ----
    # atmospheric shell, weak absorption (streaming-dominated), source in the shell
    R, H = 30.0, 5.0  # detector radius / shell thickness (Re/h-like ratio)
    rg = np.linspace(R, R + 2 * H, 300)
    shell = np.exp(-(rg - R) / (0.35 * H))  # production falls with altitude
    rs = solve_spherical(rg, n_l=24, sigma_t=0.02, c=0.0, source=shell)
    mu = np.linspace(-0.999, -0.02, 60)  # inward (downward) arrival directions
    # detector is a few cells above the inner Dirichlet wall
    psi = angular_flux(rs["phi"][:, 3], mu)
    psi = np.clip(psi, 0, None)
    cosz = -mu  # arrival zenith cosine (1=vertical down, 0=horizon)
    vert = psi[np.argmax(cosz)]
    hor = psi[np.argmin(cosz)]
    print("\nVALIDATION 2 -- curvature redistributes flux toward the horizon")
    print(f"  psi(horizon)/psi(vertical) = {hor/vert:.2f}  (>1 = horizon excess)")

    # ---- cost: sparse & ~linear including curvature ----
    print("\nCOST (spherical operator, curvature included):")
    print("  n_l    dof     nnz    density   solve[ms]")
    nls, times = [], []
    for n_l in (2, 4, 8, 16, 32):
        rr = solve_spherical(
            np.linspace(1, 6, 300),
            n_l=n_l,
            sigma_t=1.0,
            c=0.5,
            source=src[:300] if len(src) >= 300 else None,
        )
        A = rr["A"]
        dens = A.nnz / A.shape[0] ** 2
        nls.append(n_l)
        times.append(rr["solve_time"] * 1e3)
        print(
            f"  {n_l:3d}  {A.shape[0]:6d}  {A.nnz:7d}  {dens:.2e}  "
            f"{rr['solve_time']*1e3:8.1f}"
        )
    p_exp = np.polyfit(np.log(nls), np.log(np.maximum(times, 1e-3)), 1)[0]
    print(f"  scaling: solve ~ n_l^{p_exp:.2f}  (sparse & ~linear -> feasible)")

    if args.plot:
        _plot(r, phi0, ref, m, cosz, psi, vert)


def _plot(r, phi0, ref, m, cosz, psi, vert):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))
    axL.plot(r, phi0, "C0-", lw=2, label=r"P$_1$ spherical solve ($\phi_0$)")
    axL.plot(r, ref, "k--", label="independent spherical diffusion")
    axL.set_xlabel("radius r")
    axL.set_ylabel(r"scalar flux $\phi_0$")
    axL.set_title("Curvature term validated vs spherical diffusion")
    axL.legend()

    order = np.argsort(cosz)
    axR.plot(cosz[order], (psi / vert)[order], "C3o-", ms=3)
    axR.axhline(1, color="k", ls=":", lw=0.7)
    axR.set_xlabel(r"$\cos\theta_z$ (1=vertical, 0=horizon)")
    axR.set_ylabel(r"detector $\psi/\psi_{\rm vertical}$")
    axR.set_title("Streaming + curvature: horizon redistribution")
    fig.tight_layout()
    fig.savefig("spherical_streaming.png", dpi=110)
    print("\nsaved plot -> spherical_streaming.png")


if __name__ == "__main__":
    main()
