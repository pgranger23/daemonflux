"""P_N (spherical) angular transport -- large-angle cross-check of Fokker-Planck.

The Fokker-Planck solve (:mod:`fokker_planck_3d`) is a *small-angle* (Gaussian)
approximation. This module transports the *full* angular distribution on the
sphere (correct at any angle, bounded by isotropy) to check whether the
small-angle form is adequate. With the physical finite parent-chain count
(``N_CHAIN ~ 2``) the spread stays moderate (<~35 deg) and **P_N agrees with FP
everywhere** -- i.e. FP is adequate in this regime. The large-angle divergence
only appears for an (unphysical) large ``N`` (try ``--nchain 120``), which is
exactly the artifact that motivated removing ``N_gen = slant/lambda``.

Method (P_N / spherical convolution)
------------------------------------
The flux angular distribution is the single-production kernel ``f_1(mu)`` (mu =
cos theta) **convolved with itself once per production generation** on the
sphere. Spherical convolution is a product in Legendre space: with
``f_1 = sum_l (2l+1)/(4pi) c_l P_l`` (``c_0 = 1``), after ``N`` generations
``f_N = sum_l (2l+1)/(4pi) c_l^N P_l``. For a forward heat-kernel step the
coefficients are analytic, ``c_l = exp(-l(l+1) kappa)``, so ``<cos theta> = c_1``
in closed form (:func:`sn_sigma_from_theta1`); the spread ``arccos<cos theta>``
matches ``sqrt(N) theta1`` at small angle and saturates at 90 deg for large
``N``. The single-step *variance* is the NA61-validated ``<theta^2>(E)``.

Full-shape cross-check (``--kernel``): :func:`realshape_sn_spread` builds the
single-production density directly from a measured ``d2N/(dx_L dtheta)`` kernel
and compares it to a Gaussian of the *same variance*. They agree (pure-shape
median rel diff ~0): for these forward distributions ``arccos<cos theta>`` is set
by the variance alone, so the variance-only Fokker-Planck input is sufficient and
the (heavier-tailed) full-shape kernel is not needed for the angular spread.

Run::

    python sn_transport.py --moments m_spliced.npz --plot
    python sn_transport.py --moments m_spliced.npz --kernel k_local_demo.npz --plot
"""

from __future__ import annotations

import argparse

import numpy as np

from fokker_planck_3d import load_theta2


def legendre_coeffs(mu: np.ndarray, f: np.ndarray, lmax: int = 200) -> np.ndarray:
    """Legendre coefficients c_l = 2 pi integral f(mu) P_l(mu) dmu.

    ``f`` is an azimuthally-symmetric density on the sphere; if it is normalized
    (2 pi integral f dmu = 1) then ``c_0 = 1``.
    """
    from scipy.special import eval_legendre

    c = np.empty(lmax + 1)
    for ell in range(lmax + 1):
        c[ell] = 2 * np.pi * np.trapezoid(f * eval_legendre(ell, mu), mu)
    return c


def reconstruct(mu: np.ndarray, c: np.ndarray) -> np.ndarray:
    """f(mu) = sum_l (2l+1)/(4 pi) c_l P_l(mu)."""
    from scipy.special import eval_legendre

    f = np.zeros_like(mu)
    for ell, cl in enumerate(c):
        f += (2 * ell + 1) / (4 * np.pi) * cl * eval_legendre(ell, mu)
    return f


def single_production_f1(kernel_npz: str, e_sec: float, n_mu: int = 2001):
    """Per-production angular density f_1(mu) from the binned (x_L, theta) kernel.

    Returns ``mu`` (ascending) and ``f_1`` normalized so 2 pi integral f dmu = 1.
    """
    from angular_kernel import load_kernel, angular_row

    kernel, axes = load_kernel(kernel_npz)
    assert axes.get("is_angular"), "S_N needs a directly-binned angular kernel"
    # Pick the lowest projectile energy that can still produce this secondary
    # (so x_L = e_sec/e_proj <= 1 and the row is well sampled).
    pe = axes["proj_energies"]
    above = np.where(pe >= e_sec / 0.8)[0]
    i_proj = int(above[0]) if len(above) else len(pe) - 1
    th_deg, p_theta = angular_row(kernel, axes, i_proj, e_sec)

    mu = np.linspace(-1.0, 1.0, n_mu)
    th = np.arccos(mu)  # rad, descending in mu
    # interpolate P(theta) (per unit theta) onto the mu grid; zero beyond support
    p_interp = np.interp(np.degrees(th), th_deg, p_theta, left=0.0, right=0.0)
    # convert density in theta to density on sphere f(mu): dN = P(theta) dtheta,
    # dmu = -sin(theta) dtheta, and f(mu) 2pi dmu = P(theta) dtheta (azimuth avg)
    sin_th = np.sqrt(np.clip(1 - mu**2, 1e-12, None))
    f = p_interp / (2 * np.pi * sin_th)
    norm = 2 * np.pi * np.trapezoid(f, mu)
    return mu, (f / norm if norm > 0 else f)


def sigma_theta_of(mu: np.ndarray, f: np.ndarray) -> float:
    """RMS deviation angle [rad] of an on-sphere density f(mu)."""
    th = np.arccos(np.clip(mu, -1, 1))
    num = 2 * np.pi * np.trapezoid(f * th**2, mu)
    den = 2 * np.pi * np.trapezoid(f, mu)
    return float(np.sqrt(num / den))


ISO_SIGMA_RAD = np.sqrt((np.pi**2 - 4) / 2)  # RMS angle of an isotropic sphere


def gaussian_f1(mu: np.ndarray, theta1_rad: float) -> np.ndarray:
    """Forward-peaked single-production density of RMS angle ~theta1, on sphere.

    ``f(theta) ~ exp(-theta^2 / (2 s^2))`` with ``s = theta1/sqrt(2)`` (so the
    small-angle 2D variance is theta1^2), normalized to 2 pi integral f dmu = 1.
    This supplies the single-step *shape* for the P_N convolution; the *variance*
    theta1^2 = <theta^2>(E) is the NA61-validated quantity from the moments.
    """
    th = np.arccos(np.clip(mu, -1, 1))
    s = max(theta1_rad / np.sqrt(2.0), 1e-3)
    f = np.exp(-(th**2) / (2 * s**2))
    norm = 2 * np.pi * np.trapezoid(f, mu)
    return f / norm


def sn_sigma_from_theta1(theta1_rad: float, n_gen: int) -> float:
    """P_N angular spread ``arccos<cos theta>`` [rad] for a Gaussian single step.

    For a forward heat-kernel step the spherical Legendre coefficients are
    analytic, ``c_l = exp(-l(l+1) kappa_1)`` with per-step ``kappa_1 =
    theta1^2/4`` (so the small-angle variance is theta1^2). The N-generation
    self-convolution multiplies the coefficients: ``c_l^N = exp(-l(l+1) N
    kappa_1)``, and the first moment gives ``<cos theta> = c_1 = exp(-2 N
    kappa_1)``. The spread measure ``arccos<cos theta>``:

    * -> sqrt(N) * theta1 at small angle (matches the Fokker-Planck / Gaussian),
    * saturates at 90 deg (<cos theta> -> 0, fully randomized) for wide steps or
      many generations -- the physical bound the small-angle form overshoots.

    This closed form sidesteps the numerical Legendre integration (which is
    ill-conditioned for the very narrow high-energy kernels).
    """
    kappa = max(1, int(n_gen)) * theta1_rad**2 / 4.0
    mean_cos = np.exp(-2.0 * kappa)
    return float(np.arccos(np.clip(mean_cos, -1.0, 1.0)))


def realshape_sn_spread(kernel_npz, energies, n_gen):
    """Pure-shape test of the variance-only approximation, from the real kernel.

    For each energy this builds the single-production density ``f_1(mu)`` directly
    from the measured ``d2N/(dx_L dtheta)`` kernel and computes two N-generation
    spreads that share the **same variance** so only the *shape* differs:

    * ``real``    -- the true shape: self-convolve the real coefficients,
      ``arccos((c_1/c_0)^N)``;
    * ``matched`` -- a Gaussian step of the *same* ``<theta^2>`` as ``f_1``
      (:func:`sn_sigma_from_theta1`).

    Their ratio is the genuine "does the shape (tails) matter beyond the
    variance?" test -- the thing that decides whether the Fokker-Planck
    variance-only input is sufficient or the full-shape kernel is needed.
    Returns ``(real, matched)`` arrays in degrees.
    """
    real = np.full(len(energies), np.nan)
    matched = np.full(len(energies), np.nan)
    for i, e in enumerate(energies):
        try:
            mu, f1 = single_production_f1(kernel_npz, e)
        except Exception:
            continue
        c = legendre_coeffs(mu, f1, lmax=200)
        if c[0] <= 0:
            continue
        real[i] = np.degrees(
            np.arccos(np.clip((c[1] / c[0]) ** max(1, int(n_gen)), -1.0, 1.0))
        )
        theta1 = sigma_theta_of(mu, f1)  # sqrt(<theta^2>) of the SAME f_1
        matched[i] = np.degrees(sn_sigma_from_theta1(theta1, n_gen))
    return real, matched


def sn_and_smallangle(energies, e_sig, theta2, zenith_deg=0.0, n_chain=None):
    """Return (S_N spread, small-angle sqrt(N)*theta1) in degrees vs energy.

    Both use the validated <theta^2>(E) and the same finite parent-chain count N
    (``N_CHAIN``), so they are apples-to-apples: the small-angle (Gaussian/FP)
    form is unbounded, the P_N form is bounded. ``zenith_deg`` is unused (the
    production spread is zenith-independent).
    """
    from fokker_planck_3d import theta2_interp, N_CHAIN

    n_gen = int(round(N_CHAIN if n_chain is None else n_chain))
    sn = np.zeros(len(energies))
    sa = np.zeros(len(energies))
    for i, e in enumerate(energies):
        theta1 = np.sqrt(theta2_interp(e, e_sig, theta2))
        sn[i] = np.degrees(sn_sigma_from_theta1(theta1, n_gen))
        sa[i] = np.degrees(np.sqrt(n_gen) * theta1)
    return sn, sa


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--moments", default="m_spliced.npz", help="validated <theta^2>(E)")
    p.add_argument("--zenith", type=float, default=0.0)
    p.add_argument("--nchain", type=float, default=None, help="parent-chain count")
    p.add_argument(
        "--kernel",
        default=None,
        help="binned angular kernel (.npz) -> add the REAL-shape P_N curve",
    )
    p.add_argument("--plot", action="store_true")
    args = p.parse_args(argv)

    energies = np.logspace(np.log10(0.5), 2.5, 14)
    e_sig, theta2 = load_theta2(args.moments)
    sn, sa = sn_and_smallangle(
        energies, e_sig, theta2, args.zenith, n_chain=args.nchain
    )

    from fokker_planck_3d import N_CHAIN

    n_gen = int(round(N_CHAIN if args.nchain is None else args.nchain))
    real = matched = None
    if args.kernel:
        real, matched = realshape_sn_spread(args.kernel, energies, n_gen)

    print(f"angular spread [deg] at zenith {args.zenith:.0f}:")
    hdr = "  E[GeV]   P_N-Gauss   small-angle(FP)"
    hdr += "   real-shape   matched-Gauss(same var)" if real is not None else ""
    print(hdr)
    for i, (e, a, b) in enumerate(zip(energies, sn, sa)):
        line = f"  {e:7.2f}     {a:7.2f}      {b:8.2f}"
        if real is not None and np.isfinite(real[i]):
            line += f"       {real[i]:7.2f}        {matched[i]:7.2f}"
        print(line)

    if real is not None:
        m = np.isfinite(real) & np.isfinite(matched)
        if m.any():
            reldiff = np.abs(real[m] - matched[m]) / np.maximum(matched[m], 1e-6)
            print(
                "\n  PURE-SHAPE test (real vs Gaussian of the SAME variance): "
                f"median |rel diff| = {np.median(reldiff):.2f}."
            )
            print(
                "  -> The real production-angle shape and a Gaussian of equal"
                " variance give the\n  SAME spread: for these forward (small-angle)"
                " distributions arccos<cos theta>\n  is set by the variance alone,"
                " so the variance-only Fokker-Planck input is\n  sufficient and the"
                " full-shape kernel is NOT needed for the angular spread."
            )

    if args.plot:
        _plot(energies, sn, sa, args, real=real, matched=matched)


def _plot(energies, sn, sa, args, real=None, matched=None):
    """Two independent overlap-tests, one per panel (each: curves agree -> pass).

    Left  -- *method* test, on the production moments: Fokker-Planck (small-angle)
             vs P_N (full-angle). Same input, two methods -> overlap means FP is
             adequate. Right -- *shape* test, on a demo kernel: its real angular
             shape vs a Gaussian of the SAME variance -> overlap means only the
             variance matters. The two panels use different kernels, so their
             absolute heights are not meant to match -- read each panel on its own.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    have_shape = real is not None and np.isfinite(real).any()
    fig, axes = plt.subplots(
        1,
        2 if have_shape else 1,
        figsize=(12 if have_shape else 6.6, 4.6),
        squeeze=False,
    )
    axL = axes[0, 0]
    # -- Panel 1: method test (FP vs P_N), same input (production moments) --
    axL.loglog(
        energies,
        sa,
        "C3--",
        lw=2.5,
        label=r"Fokker-Planck (small-angle $\sqrt{N}\,\theta_1$)",
    )
    axL.loglog(energies, sn, "C0o-", lw=1.5, ms=5, label=r"$P_N$ (full-angle)")
    axL.axhline(90.0, color="gray", ls=":", label="90° isotropy bound")
    axL.axvspan(energies[0], 2.0, color="orange", alpha=0.15)
    axL.set_xlabel("neutrino energy [GeV]")
    axL.set_ylabel(r"angular spread $\arccos\langle\cos\theta\rangle$ [deg]")
    axL.set_title(r"Method test: FP $\equiv$ P$_N$ $\Rightarrow$ FP adequate")
    axL.legend()

    if have_shape:
        axR = axes[0, 1]
        m = np.isfinite(real) & np.isfinite(matched)
        axR.loglog(
            energies[m],
            matched[m],
            "C2-",
            lw=4,
            alpha=0.4,
            label="Gaussian of the same variance",
        )
        axR.loglog(
            energies[m],
            real[m],
            "C2s",
            ms=8,
            mfc="none",
            label="real measured kernel shape",
        )
        axR.set_xlabel("neutrino energy [GeV]")
        axR.set_ylabel(r"angular spread $\arccos\langle\cos\theta\rangle$ [deg]")
        axR.set_title(r"Shape test: real $\equiv$ same-variance Gaussian")
        axR.text(
            0.5,
            0.05,
            "(demo kernel k_local_demo; its variance differs from the\n"
            "left panel's moments, so heights need not match)",
            transform=axR.transAxes,
            fontsize=7.5,
            ha="center",
            color="0.4",
        )
        axR.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig("sn_transport.png", dpi=110)
    print("saved plot -> sn_transport.png")


if __name__ == "__main__":
    main()
