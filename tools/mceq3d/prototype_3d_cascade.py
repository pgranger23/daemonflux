"""De-risking prototype: a coupled (energy x angle) deterministic cascade.

This is the minimal "3D-MCEq" core requested to test two things before committing
to a full build:

  (1) ASSEMBLY  -- can the energy cascade and the (validated) angular kernels be
      solved *together*, self-consistently, reducing to the 1D result when the
      production angle -> 0?
  (2) PERFORMANCE -- does the deterministic solve stay cheap, or does the angular
      dimension blow it up?

Key idea (why it is cheap)
--------------------------
Represent each species' angular distribution about the column axis in Legendre
modes ``c_l`` (mu = cos of deviation angle). A forward production kick of RMS
angle ``theta1`` is, on the sphere, the heat kernel -- it multiplies the
coefficients by ``exp(-l(l+1) theta1^2/4)``. Interaction/decay losses are
direction-independent (diagonal in ``l``). In a single column (no spatial
streaming) nothing mixes different ``l``. **So the coupled (E, mu) solve
decouples into independent 1D-energy cascades, one per multipole ``l``**, with
the production yields scaled by the per-``l`` kernel factor. Cost =
``n_l x (1D cascade)``, vectorized over ``l``.

The ``l=0`` mode is exactly the standard 1D flux. The higher modes carry the
angular structure; the neutrino angular distribution at the ground follows from
``c_l(E)``. The *effective number of production generations* contributing to a
neutrino's angle is **not assumed** -- it emerges from the cascade.

Physics scope (prototype)
-------------------------
Minimal but genuine conventional-numu chain: N -> (leading N) + pi ; pi -> numu
(decay) competing with pi interaction loss; numu terminal. Single isothermal
column, depth-independent decay length via the critical energy eps_pi. Charge,
kaons, muon-decay neutrinos, and the spatial/streaming term (which would re-
couple the l modes -- the genuinely hard part of full 3D) are omitted.
Longitudinal yields are simple scaling forms; the angular kernel uses the
NA61-validated <theta^2>(E).

Run::

    python prototype_3d_cascade.py --moments m_spliced.npz --plot
"""

from __future__ import annotations

import argparse
import time

import numpy as np

# Species
N, PI, NUMU = 0, 1, 2
LAM_N, LAM_PI = 90.0, 120.0  # interaction lengths [g/cm^2]
EPS_PI = 115.0  # pion critical energy [GeV]
X_VERT = 1030.0  # vertical column depth [g/cm^2]
R_PI = (0.10566 / 0.13957) ** 2  # (m_mu/m_pi)^2 -> numu energy fraction in [0,1-r]


def log_grid(n_e=60, e_lo=0.3, e_hi=1e6):
    e_edges = np.logspace(np.log10(e_lo), np.log10(e_hi), n_e + 1)
    e = np.sqrt(e_edges[:-1] * e_edges[1:])
    dlnE = np.log(e_edges[1] / e_edges[0])
    return e, dlnE


def scaling_matrix(e, dlnE, dNdx_func, x_min=1e-4):
    """Y[i_daughter, j_parent] = dN/dlnE_daughter for a scaling yield dN/dx
    (x = E_d/E_p); Toeplitz in the log grid."""
    n = len(e)
    Y = np.zeros((n, n))
    for j in range(n):
        x = e[: j + 1] / e[j]
        good = x >= x_min
        Y[: j + 1, j][good] = (x * dNdx_func(x))[good] * dlnE
    return Y


def run_cascade(
    moments="m_spliced.npz", lmax=80, n_e_bins=60, nsteps=1200, collimated=False
):
    """Solve the coupled (E, l) cascade; return a results dict.

    Keys: ``e`` (GeV), ``c0`` (l=0 numu flux, dN/dlnE arb.), ``sigma_theta``
    (deg, = arccos(c_1/c_0)), ``theta1`` (rad, single-production angle),
    ``runtime`` (s), ``state_shape``.
    """
    from fokker_planck_3d import load_theta2, theta2_interp

    e, dlnE = log_grid(n_e_bins)
    n_e = len(e)
    ell = np.arange(lmax + 1)

    e_sig, t2 = load_theta2(moments) if isinstance(moments, str) else moments
    theta1 = np.sqrt(theta2_interp(e, e_sig, t2))
    theta_dec = 0.030 / np.maximum(e, 1e-3)  # pi->munu decay kick ~ p*/E [rad]
    if collimated:
        theta1 = np.zeros_like(theta1)
        theta_dec = np.zeros_like(theta_dec)

    def kernel_l(theta_E):  # (n_e, n_l) heat-kernel factor exp(-l(l+1) theta^2/4)
        kappa = theta_E[:, None] ** 2 / 4.0
        return np.exp(-ell[None, :] * (ell[None, :] + 1) * kappa)

    K_prod = kernel_l(theta1)
    K_dec = kernel_l(theta_dec)

    Y_NN = scaling_matrix(e, dlnE, lambda x: 0.8 * np.ones_like(x))
    Y_Npi = scaling_matrix(e, dlnE, lambda x: 5.0 * (1 - x) ** 3 / x)
    Y_pinu = scaling_matrix(
        e, dlnE, lambda x: np.where(x < (1 - R_PI), 1.0 / (1 - R_PI), 0.0)
    )

    lam_dec = X_VERT * e / EPS_PI
    inv_lam_pi_tot = 1.0 / LAM_PI + 1.0 / lam_dec
    f_dec = (1.0 / lam_dec) / inv_lam_pi_tot

    n_l = lmax + 1
    Phi = [np.zeros((n_e, n_l)) for _ in range(3)]
    Phi[N][:] = (e ** (-1.7))[:, None]  # collimated primary (c_l equal)

    dX = X_VERT / nsteps
    surv_N = np.exp(-dX / LAM_N)
    surv_pi = np.exp(-dX * inv_lam_pi_tot)[:, None]
    t0 = time.time()
    for _ in range(nsteps):
        dN_int = Phi[N] * (1.0 - surv_N)
        src_N = Y_NN @ dN_int
        src_pi = (Y_Npi @ dN_int) * K_prod
        loss_pi = Phi[PI] * (1.0 - surv_pi)
        src_nu = (Y_pinu @ (loss_pi * f_dec[:, None])) * K_dec
        Phi[N] = Phi[N] - dN_int + src_N
        Phi[PI] = Phi[PI] - loss_pi + src_pi
        Phi[NUMU] = Phi[NUMU] + src_nu
    runtime = time.time() - t0

    c = Phi[NUMU]
    c0 = c[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma_theta = np.degrees(np.arccos(np.clip(c[:, 1] / c0, -1, 1)))
    sigma_theta[~(c0 > 0)] = np.nan
    return dict(
        e=e,
        c0=c0,
        sigma_theta=sigma_theta,
        theta1=theta1,
        runtime=runtime,
        state_shape=(n_e, n_l),
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--moments", default="m_spliced.npz")
    p.add_argument("--lmax", type=int, default=80)
    p.add_argument("--nE", type=int, default=60)
    p.add_argument("--nsteps", type=int, default=1200)
    p.add_argument("--collimated", action="store_true")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args(argv)

    r = run_cascade(args.moments, args.lmax, args.nE, args.nsteps, args.collimated)
    e, st = r["e"], r["sigma_theta"]
    nl = r["state_shape"][1]
    print(f"ASSEMBLY: coupled (E x l) solve, state {r['state_shape']} x 3 species")
    print(
        f"PERFORMANCE: {args.nsteps} steps, lmax={args.lmax}  ->  "
        f"{r['runtime']*1e3:.0f} ms  ({r['runtime']*1e3/nl:.2f} ms / multipole)"
    )

    if args.collimated:
        sig = np.nanmax(st[r["c0"] > r["c0"].max() * 1e-6])
        print(f"REDUCTION-TO-1D CHECK: max sigma_theta = {sig:.3e} deg (should be ~0)")
        return

    print("\nself-consistent numu angular spread (NO assumed N_chain):")
    print("  E[GeV]   sigma_theta[deg]   theta1[deg]   sigma/theta1")
    for i in range(0, len(e), 4):
        if 0.5 < e[i] < 3000 and np.isfinite(st[i]):
            t1d = np.degrees(r["theta1"][i])
            print(
                f"  {e[i]:8.2f}    {st[i]:7.2f}        {t1d:6.2f}      "
                f"{st[i]/t1d:.2f}"
            )
    if args.plot:
        _plot(r, args)


def _plot(r, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e, c0, st, th1 = r["e"], r["c0"], r["sigma_theta"], r["theta1"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))
    sel = (e > 0.5) & (e < 1e5) & (c0 > 0)
    axL.loglog(e[sel], (c0 * e**2)[sel], "C0-")
    axL.set_xlabel("neutrino energy [GeV]")
    axL.set_ylabel(r"$E^2\,\Phi_{\nu_\mu}$ (l=0 mode, arb.)")
    axL.set_title(r"1D flux (l=0) with spectral break $\sim\epsilon_\pi$")

    s = (e > 0.5) & (e < 3000) & np.isfinite(st)
    axR.loglog(e[s], st[s], "C0o-", ms=3, label=r"$\sigma_\theta$ (coupled solve)")
    axR.loglog(e[s], np.degrees(th1)[s], "C1--", label=r"$\theta_1$ (single prod.)")
    axR.loglog(
        e[s],
        np.degrees(np.sqrt(2) * th1)[s],
        "k:",
        label=r"$\sqrt{2}\,\theta_1$ (old N_chain=2 guess)",
    )
    axR.axvspan(0.5, 2.0, color="orange", alpha=0.15)
    axR.set_xlabel("neutrino energy [GeV]")
    axR.set_ylabel(r"$\sigma_\theta$ [deg]")
    axR.set_title("Angular spread emerges self-consistently (below the guess)")
    axR.legend()
    fig.tight_layout()
    fig.savefig("prototype_3d_cascade.png", dpi=110)
    print("\nsaved plot -> prototype_3d_cascade.png")


if __name__ == "__main__":
    main()
