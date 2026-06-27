"""Streaming feasibility prototype: does spatial transport stay cheap?

The single-column prototype (:mod:`prototype_3d_cascade`) showed the angular
*production* operator is free in the Legendre basis because, with no spatial
transport, the multipoles decouple. The remaining open question for a real
3D-MCEq is the **streaming** term ``Omega . grad``: in any genuine geometry it
couples neighbouring multipoles (``l <-> l +/- 1``) and neighbouring spatial
cells, so the multipoles no longer decouple. The worry: does that re-coupling
blow the deterministic solve up to Monte-Carlo cost, or stay sparse and cheap?

This prototype answers it concretely with the textbook **slab P_N transport**
(the same physics structure as the spherical case, minus curvature):

    mu d(psi)/dx + sigma_t psi = (sigma_s/2) integral psi dmu' + Q/2

In the P_N (Legendre) representation the streaming derivative gives the standard
recursion that couples ``phi_l`` to ``phi_{l-1}`` and ``phi_{l+1}`` -- exactly the
multipole re-coupling we need to test. We assemble the full
(space x multipole) operator as a sparse matrix, solve it, and measure:

* **sparsity** (nnz, density, bandwidth) -- is it sparse?
* **solve time and scaling** with n_l and n_x -- linear, or blow-up?
* **physics validation** at the diffusion (P_1, scattering-dominated) limit,
  where phi_0 has a closed-form analytic solution.

The curved (spherical) case adds one more tridiagonal-in-l term (the curvature
``(1-mu^2)/r d/dmu``); it does not change the sparsity class, so the cost verdict
here carries over. Energy is a trivial batch dimension (x n_E).

Run::

    python prototype_streaming.py --plot
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def build_pn_slab(n_x, n_l, sigma_t, sigma_s, source, length):
    """Assemble the slab-P_N steady transport operator A and RHS b (sparse).

    Unknowns u[l*n_x + i] = phi_l(x_i). Central differences for the streaming
    derivative; Dirichlet phi_l=0 at both ends (large slab -> interior is the
    physical region). Isotropic scattering and source enter the l=0 equation.
    """
    x = np.linspace(0, length, n_x)
    h = x[1] - x[0]

    def idx(ll_, i):
        return ll_ * n_x + i

    rows, cols, vals = [], [], []
    b = np.zeros(n_x * n_l)

    for ll in range(n_l):
        for i in range(n_x):
            r = idx(ll, i)
            if i == 0 or i == n_x - 1:  # Dirichlet boundary: phi_l = 0
                rows.append(r)
                cols.append(r)
                vals.append(1.0)
                continue
            # collision term: sigma_t phi_l  (minus sigma_s phi_0 for l=0)
            diag = sigma_t - (sigma_s if ll == 0 else 0.0)
            rows.append(r)
            cols.append(r)
            vals.append(diag)
            # streaming: l/(2l+1) d phi_{l-1}/dx + (l+1)/(2l+1) d phi_{l+1}/dx
            if ll - 1 >= 0:
                cf = ll / (2 * ll + 1) / (2 * h)
                rows += [r, r]
                cols += [idx(ll - 1, i + 1), idx(ll - 1, i - 1)]
                vals += [cf, -cf]
            if ll + 1 < n_l:
                cf = (ll + 1) / (2 * ll + 1) / (2 * h)
                rows += [r, r]
                cols += [idx(ll + 1, i + 1), idx(ll + 1, i - 1)]
                vals += [cf, -cf]
            if ll == 0:
                b[r] = source[i]

    A = sp.csr_matrix((vals, (rows, cols)), shape=(n_x * n_l, n_x * n_l))
    return A, b, x


def solve_pn(n_x=400, n_l=8, sigma_t=1.0, c=0.99, length=40.0, source_width=1.0):
    """Solve the slab-P_N problem; return x, phi_0, the matrix, and timing."""
    sigma_s = c * sigma_t
    x = np.linspace(0, length, n_x)
    x0 = length / 2
    source = np.exp(-((x - x0) ** 2) / (2 * source_width**2))
    source /= np.trapezoid(source, x)  # unit-integral source

    A, b, x = build_pn_slab(n_x, n_l, sigma_t, sigma_s, source, length)
    t0 = time.time()
    u = spla.spsolve(A.tocsc(), b)
    dt = time.time() - t0
    phi0 = u[:n_x]  # l=0 moment = scalar flux
    return dict(
        x=x,
        phi0=phi0,
        A=A,
        solve_time=dt,
        source=source,
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        length=length,
    )


def diffusion_analytic(x, sigma_t, sigma_s, length):
    """P_1 / diffusion-limit scalar flux for a point source at the centre.

    -D phi'' + sigma_a phi = delta(x-x0)  ->  phi = exp(-kappa|x-x0|)/(2 D kappa).
    """
    D = 1.0 / (3 * sigma_t)
    sigma_a = sigma_t - sigma_s
    kappa = np.sqrt(sigma_a / D)
    x0 = length / 2
    return np.exp(-kappa * np.abs(x - x0)) / (2 * D * kappa)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args(argv)

    # --- physics validation at the diffusion limit (P_1, scattering-dominated) ---
    r = solve_pn(n_x=400, n_l=2, c=0.99, length=40.0)
    ana = diffusion_analytic(r["x"], r["sigma_t"], r["sigma_s"], r["length"])
    m = (r["x"] > 8) & (r["x"] < 32)  # interior, away from Dirichlet walls
    scale = np.trapezoid(r["phi0"][m], r["x"][m]) / np.trapezoid(ana[m], r["x"][m])
    rel = np.abs(r["phi0"][m] / (ana[m] * scale) - 1.0)
    print("PHYSICS (P_1 vs analytic diffusion, scattering-dominated):")
    print(f"  median |rel diff| in interior = {np.median(rel):.3f}")

    # --- sparsity + cost + scaling ---
    print("\nFEASIBILITY (does streaming keep it sparse/cheap?):")
    print("  n_l   dof     nnz     density   solve[ms]   ms/dof")
    rows = []
    for n_l in (2, 4, 8, 16, 32, 64):
        rr = solve_pn(n_x=400, n_l=n_l, c=0.9, length=40.0)
        A = rr["A"]
        dof = A.shape[0]
        dens = A.nnz / dof**2
        rows.append((n_l, dof, A.nnz, dens, rr["solve_time"] * 1e3))
        print(
            f"  {n_l:3d}  {dof:6d}  {A.nnz:7d}   {dens:.2e}   "
            f"{rr['solve_time']*1e3:8.1f}   {rr['solve_time']*1e3/dof:.2e}"
        )

    nls = np.array([x[0] for x in rows])
    times = np.array([x[4] for x in rows])
    dens = np.array([x[3] for x in rows])
    # scaling exponent: time ~ n_l^p
    p_exp = np.polyfit(np.log(nls), np.log(times), 1)[0]
    print(
        f"\n  scaling: solve time ~ n_l^{p_exp:.2f}  (1.0 = linear; <<2 means no blow-up)"
    )
    print(
        f"  matrix density falls ~1/n_l (stays sparse): "
        f"{dens[0]:.1e} -> {dens[-1]:.1e}"
    )
    verdict = (
        "SPARSE & ~LINEAR -> deterministic 3D is cost-feasible"
        if p_exp < 1.6
        else "super-linear -> potential blow-up"
    )
    print(f"  VERDICT: {verdict}")

    if args.plot:
        _plot(r, ana, scale, nls, times, dens, args)


def _plot(r, ana, scale, nls, times, dens, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))
    axL.plot(r["x"], r["phi0"], "C0-", lw=2, label=r"P$_1$ solve ($\phi_0$)")
    axL.plot(r["x"], ana * scale, "k--", label="analytic diffusion")
    axL.set_yscale("log")
    axL.set_xlabel("x")
    axL.set_ylabel(r"scalar flux $\phi_0$")
    axL.set_title("Physics check: streaming+scattering vs diffusion limit")
    axL.legend()

    axR.loglog(nls, times, "C0o-", label="solve time")
    axR.loglog(nls, times[0] * nls / nls[0], "k:", label="linear ref")
    axR.set_xlabel("number of multipoles $n_l$")
    axR.set_ylabel("sparse solve time [ms]")
    axR.set_title("Cost scaling: sparse & ~linear (no blow-up)")
    axR.legend()
    fig.tight_layout()
    fig.savefig("prototype_streaming.png", dpi=110)
    print("\nsaved plot -> prototype_streaming.png")


if __name__ == "__main__":
    main()
