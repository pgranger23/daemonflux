"""First 3D solve: a Fokker-Planck angular-transport prototype.

This is the deterministic 3D step the whole pipeline was building toward. MCEq
solves the 1D (collinear) cascade in energy; here we add the *angular* transport
that the 1D matrices throw away, driven by the **validated** production angular
variance ``<theta^2>(E)`` from :mod:`kernel_regeneration` (``--moments``, checked
against NA61 in :mod:`validate_na61`).

Physics model (prototype)
-------------------------
A secondary lepton acquires a transverse kick at each production step; over the
development of the shower these accumulate into an angular spread of the flux
about the primary direction. In the small-angle limit this is a diffusion in the
*projected* deviation angle ``phi`` as a function of slant depth ``X`` [g/cm^2]::

    d g(phi, X) / dX = D(E, X) d^2 g / dphi^2  +  q(X) delta(phi)

* ``D(E, X) = <theta^2>_prod(E) / (4 lambda)`` -- the (projected) angular
  diffusion coefficient per unit depth; ``<theta^2>_prod`` is the validated
  per-production variance, ``lambda`` the interaction/production length.
* ``q(X)`` -- the shower-development profile: new (collinear) secondaries are
  injected along the column and then diffuse over the *remaining* depth, so the
  detector distribution is an integral over production depths (the march does
  this automatically).

The energy axis is treated as a parameter here (each energy solved independently)
-- the energy redistribution is exactly the existing 1D MCEq operator, which this
prototype is designed to be multiplied onto, not to replace. This is the first
deterministic solve of the *new* (angular) operator, end-to-end from validated
kernels.

Limiting checks
---------------
* top-only source + constant D -> g is an exact Gaussian, sigma^2 = 2 D X
  (:func:`analytic_sigma_proj`), reproduced by the solver (correctness gate).
* sigma_theta -> 0 at high E (1D recovered). It is ~zenith-independent: the
  spread is set by production kinematics accumulated over the *finite* parent
  chain (N_CHAIN generations, effective depth N_CHAIN*lambda), NOT by the full
  column depth. Genuine geometric zenith effects (Honda horizontal excess) are a
  separate ingredient, not included here.

Run::

    python fokker_planck_3d.py --moments m_spliced.npz --plot
"""

from __future__ import annotations

import argparse

import numpy as np

X_VERT = 1030.0  # vertical atmospheric depth [g/cm^2]

# Number of hadronic generations in a neutrino's parent chain (p -> meson -> nu,
# plus a sub-dominant decay/secondary contribution). This is *small and finite*
# -- NOT slant_depth/lambda, which counts interaction lengths along the whole
# column and wrongly grows without bound toward the horizon. The production
# transverse momentum (~0.3 GeV, NA61-validated) dominates the angular kick over
# the two-body decay (~0.03 GeV), so the spread is essentially the production
# angle accumulated over ~N_CHAIN steps and is ~zenith-independent (set by
# kinematics, not column depth). Any genuine zenith dependence is geometric
# (Honda horizontal excess) and is a separate, not-yet-included ingredient.
N_CHAIN = 2.0


# ---------------------------------------------------------------------------
# Validated angular variance  <theta^2>(E)
# ---------------------------------------------------------------------------
def load_theta2(moments_npz: str):
    """Return (E_centers [GeV], theta2 [rad^2]) from a validated moments file."""
    from angular_kernel import pool_moments_by_energy

    d = dict(np.load(moments_npz))
    assert "is_moments" in d, "expected a --moments file (e.g. m_spliced.npz)"
    mom = {k: d[k] for k in ("e_sec", "theta_mean", "theta_sq", "dndx")}
    e_edges = np.logspace(
        np.log10(max(d["e_sec"][d["e_sec"] > 0].min(), 0.3)),
        np.log10(d["e_sec"].max()),
        40,
    )
    pooled = pool_moments_by_energy(mom, e_edges)
    # drop the leading-particle / statistics-starved edge bins (see
    # pool_moments_by_energy): they bias theta downward at the highest E_sec.
    good = np.isfinite(pooled["theta_sq"]) & pooled["reliable"]
    return pooled["e_sec"][good], pooled["theta_sq"][good]


def theta2_interp(e_query, e_grid, theta2):
    """Interpolate <theta^2>(E) in log-log, clamped to the measured range."""
    lg = np.interp(
        np.log(e_query),
        np.log(e_grid),
        np.log(theta2),
        left=np.log(theta2[0]),
        right=np.log(theta2[-1]),
    )
    return np.exp(lg)


# ---------------------------------------------------------------------------
# Shower-development source profile
# ---------------------------------------------------------------------------
def shower_profile(x_grid: np.ndarray, x_max: float, lam: float) -> np.ndarray:
    """Normalized production profile q(X): rises then attenuates with depth.

    A simple Gaisser-Hillas-like shape (production tracks the primary cascade and
    then dies out). Normalized to unit integral over the column.
    """
    xm = 0.30 * x_max  # shower maximum ~30% into the column
    prof = np.clip(x_grid, 1e-6, None) ** 2 * np.exp(-x_grid / lam)
    prof[x_grid > x_max] = 0.0
    _ = xm
    s = np.trapezoid(prof, x_grid)
    return prof / s if s > 0 else prof


# ---------------------------------------------------------------------------
# Fokker-Planck solve (projected angle vs slant depth)
# ---------------------------------------------------------------------------
def solve_projected_fp(
    d_coeff: float,
    x_max: float,
    lam: float,
    phi_max: float = 1.4,
    n_phi: int = 401,
    n_steps: int = 800,
    top_source: bool = False,
):
    """Solve d g/dX = D d^2 g/dphi^2 + q(X) delta(phi) by implicit time-stepping.

    Backward Euler with a tridiagonal solve per depth step: unconditionally
    stable (no CFL limit), so it handles the large slant depths near the horizon.
    This is the prototype of the angular-transport operator; it generalizes
    directly to depth-dependent D and added sinks/sources.

    Parameters
    ----------
    d_coeff : float
        Diffusion coefficient D [rad^2 / (g/cm^2)] (constant in this prototype).
    x_max : float
        Total slant depth [g/cm^2].
    lam : float
        Production length for the source profile [g/cm^2].
    top_source : bool
        If True, inject all production at X=0 (delta in depth) -> analytic
        Gaussian limit for the correctness check.

    Returns
    -------
    phi, g : np.ndarray
        Projected-angle grid [rad] and the normalized detector distribution.
    """
    from scipy.linalg import solve_banded

    phi = np.linspace(-phi_max, phi_max, n_phi)
    dphi = phi[1] - phi[0]
    x_grid = np.linspace(0.0, x_max, n_steps + 1)
    dx = x_grid[1] - x_grid[0]
    r = d_coeff * dx / dphi**2

    # Backward-Euler tridiagonal operator (I - dx D L), interior rows; the two
    # edge rows are identity (Dirichlet g=0 at +-phi_max, chosen wide).
    ab = np.zeros((3, n_phi))
    ab[0, 2:] = -r  # upper diagonal
    ab[1, :] = 1.0 + 2.0 * r  # main diagonal
    ab[2, :-2] = -r  # lower diagonal
    ab[1, 0] = ab[1, -1] = 1.0
    ab[0, 1] = ab[2, -2] = 0.0

    delta0 = np.zeros_like(phi)
    delta0[n_phi // 2] = 1.0 / dphi  # collimated injection at phi=0

    if top_source:
        q = np.zeros_like(x_grid)
        g = delta0.copy()
    else:
        q = shower_profile(x_grid, x_max, lam)
        g = np.zeros_like(phi)

    for k in range(n_steps):
        rhs = g + dx * q[k] * delta0
        rhs[0] = rhs[-1] = 0.0
        g = solve_banded((1, 1), ab, rhs)

    area = np.trapezoid(g, phi)
    return phi, (g / area if area > 0 else g)


def sigma_from_dist(phi: np.ndarray, g: np.ndarray) -> float:
    """RMS projected deviation angle [rad] of a normalized distribution."""
    return float(np.sqrt(np.trapezoid(g * phi**2, phi) / np.trapezoid(g, phi)))


def analytic_sigma_proj(d_coeff: float, x_max: float) -> float:
    """Projected sigma for a top-only source with constant D: sqrt(2 D X)."""
    return float(np.sqrt(2.0 * d_coeff * x_max))


# ---------------------------------------------------------------------------
# Driver: sigma_theta(E, zenith) from the validated kernels
# ---------------------------------------------------------------------------
R_EARTH = 6371.0  # km
H_ATM_EFF = 40.0  # km, effective atmospheric/production shell thickness


def slant_depth(zenith_deg, curved: bool = True) -> float:
    """Slant grammage [g/cm^2] along a line of sight at the given zenith angle.

    The flat ``sec theta`` law diverges at the horizon, which over-estimates the
    near-horizon path (and hence the accumulated angular spread). With
    ``curved=True`` (default) we use the spherical-shell geometry: for a detector
    at the surface looking through an atmosphere of effective thickness
    ``H_ATM_EFF`` around an Earth of radius ``R_EARTH``, the path length is

        L(theta) = sqrt((R+H)^2 - R^2 sin^2 theta) - R cos theta,

    so the grammage ratio L(theta)/L(0) -> sec theta for small theta but
    *saturates* at the horizon (L(90)/H ~ sqrt(2R/H) ~ 18), as in real curved-
    atmosphere airmass (Chapman-function) treatments. This is the defensible
    geometric refinement; the full off-axis production-volume enhancement
    (Honda horizontal excess) is a separate ingredient and not included here.
    """
    zenith_deg = np.asarray(zenith_deg, dtype=float)
    if not curved:
        return X_VERT / np.cos(np.deg2rad(np.minimum(zenith_deg, 89.0)))
    s = np.sin(np.deg2rad(zenith_deg))
    c = np.cos(np.deg2rad(zenith_deg))
    r, h = R_EARTH, H_ATM_EFF
    path = np.sqrt((r + h) ** 2 - (r * s) ** 2) - r * c  # km
    return X_VERT * path / h  # ratio to vertical path (= h) times vertical grammage


def sigma_theta_vs_energy(
    e_grid,
    theta2_grid,
    energies,
    zenith_deg=0.0,
    lam=120.0,
    n_chain=N_CHAIN,
    **solve_kw,
):
    """Full-space RMS deviation angle sigma_theta(E) [deg].

    Uses the validated <theta^2>(E); the angular variance accumulates over
    ``n_chain`` production generations (effective depth ``n_chain * lambda``),
    NOT over the full slant column -- so sigma_theta = sqrt(n_chain) * theta1 and
    is **zenith-independent** (production kinematics, not column depth). The
    ``zenith_deg`` argument is retained for API compatibility but no longer
    scales the production spread; geometric zenith effects are separate.
    """
    x_max = n_chain * lam  # effective depth of the finite parent chain
    out = np.zeros(len(energies))
    for i, e in enumerate(energies):
        t2 = theta2_interp(e, e_grid, theta2_grid)
        d_coeff = t2 / (4.0 * lam)
        phi, g = solve_projected_fp(d_coeff, x_max, lam, **solve_kw)
        out[i] = np.degrees(np.sqrt(2.0) * sigma_from_dist(phi, g))
    return out


# ---------------------------------------------------------------------------
# CLI / demo
# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--moments", default="m_spliced.npz")
    p.add_argument("--lam", type=float, default=120.0, help="prod. length [g/cm^2]")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args(argv)

    e_grid, theta2 = load_theta2(args.moments)

    # Correctness gate: top-source + constant D must match the Gaussian.
    Dtest = 1e-4
    phi, g = solve_projected_fp(Dtest, X_VERT, args.lam, top_source=True)
    num = sigma_from_dist(phi, g)
    ana = analytic_sigma_proj(Dtest, X_VERT)
    print(
        f"correctness (top-source Gaussian): numeric sigma={num:.4f} rad, "
        f"analytic={ana:.4f} rad, ratio={num / ana:.3f}"
    )

    energies = np.logspace(np.log10(0.7), 3.5, 16)
    print(
        f"\nsigma_theta(E) from FP solve [deg]  (N_chain={N_CHAIN:.0f}, "
        "zenith-independent):"
    )
    print("  E[GeV]   sigma_theta")
    s0 = sigma_theta_vs_energy(e_grid, theta2, energies, lam=args.lam)
    for e, a in zip(energies, s0):
        print(f"  {e:7.2f}   {a:6.2f}")

    if args.plot:
        _plot(e_grid, theta2, energies, s0, args)


def _plot(e_grid, theta2, energies, s0, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))

    axL.loglog(energies, s0, "o-", color="C0")
    axL.axvspan(energies[0], 2.0, color="orange", alpha=0.15)
    axL.set_xlabel("neutrino energy [GeV]")
    axL.set_ylabel(r"$\sigma_\theta$ of flux [deg]")
    axL.set_title(
        rf"3D angular spread (validated $D_\theta$, $N_{{\rm chain}}$={N_CHAIN:.0f})"
    )

    x_max = N_CHAIN * args.lam  # finite parent-chain depth
    for e in (1.0, 5.0, 100.0):
        t2 = theta2_interp(e, e_grid, theta2)
        phi, g = solve_projected_fp(t2 / (4.0 * args.lam), x_max, args.lam)
        axR.plot(np.degrees(phi), g / g.max(), label=f"E={e:.0f} GeV")
    axR.set_xlim(-40, 40)
    axR.set_xlabel(r"projected deviation $\phi$ [deg]")
    axR.set_ylabel("flux (normalized)")
    axR.set_title("Vertical angular distribution at detector")
    axR.legend()

    fig.tight_layout()
    fig.savefig("fokker_planck_3d.png", dpi=110)
    print("saved plot -> fokker_planck_3d.png")


if __name__ == "__main__":
    main()
