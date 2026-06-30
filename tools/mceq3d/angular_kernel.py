"""Turn the regenerated (x_L, p_T) kernel into angular transport objects.

This is the bridge from :mod:`kernel_regeneration` to a deterministic 3D
solver. A secondary produced with transverse momentum ``p_T`` and longitudinal
momentum ``p_L`` leaves the interaction at a production angle

    theta = arctan(p_T / p_L),   p_L = sqrt(E_sec^2 - m^2),  E_sec = x_L E_proj

relative to the projectile axis. The double-differential kernel
``d2N/(dx_L dp_T)`` therefore encodes the angular redistribution that a 1D
matrix cascade throws away. Here we extract:

* the production-angle distribution per (E_proj, secondary energy);
* its moments ``<theta>`` and ``<theta^2>`` -- the latter is the
  angular-diffusion coefficient ``D_theta`` of a Fokker-Planck solver;
* the discrete-ordinates scattering row ``P(mu)`` (``mu = cos theta``) -- the
  input to an S_N solver;
* the energy at which the production angle drops below a detector-relevant
  scale -- i.e. the quantitative boundary of the "3D matters" regime.

The angular structure is **independent of the overall yield normalization**
(the ~4.4 MCEq-convention factor discussed in the README cancels here), so these
objects are usable directly from the validated kernels.

Run::

    python angular_kernel.py --kernel kernel_real_piplus.npz --plot
"""

from __future__ import annotations

import argparse
from typing import Dict, Tuple

import numpy as np

M_PION = 0.13957


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_kernel(path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Load a kernel .npz saved by :mod:`kernel_regeneration`.

    Returns
    -------
    kernel : np.ndarray, shape (n_proj, n_xl, n_pt)
        d2N/(dx_L dp_T) density.
    axes : dict
        ``proj_energies``, ``xl_centers``, ``pt_centers``, ``xl_edges``,
        ``pt_edges``.
    """
    d = np.load(path)
    xl_edges = d["xl_edges"]
    axes = {
        "proj_energies": d["proj_energies"],
        "xl_edges": xl_edges,
        "xl_centers": np.sqrt(xl_edges[:-1] * xl_edges[1:]),
    }
    if "theta_edges" in d:
        # Directly-binned angular kernel d2N/(dx_L dtheta), theta in degrees.
        th = d["theta_edges"]
        axes["is_angular"] = True
        axes["theta_edges"] = th
        axes["theta_centers"] = 0.5 * (th[:-1] + th[1:])
    else:
        pt_edges = d["pt_edges"]
        axes["is_angular"] = False
        axes["pt_edges"] = pt_edges
        axes["pt_centers"] = 0.5 * (pt_edges[:-1] + pt_edges[1:])
    return d["kernel"], axes


# ---------------------------------------------------------------------------
# Production angle
# ---------------------------------------------------------------------------
def production_angle(axes: Dict[str, np.ndarray], mass: float = M_PION) -> np.ndarray:
    """Production angle [rad] at each (E_proj, x_L, p_T) bin center.

    Shape ``(n_proj, n_xl, n_pt)``.
    """
    e_proj = axes["proj_energies"][:, None, None]
    xl = axes["xl_centers"][None, :, None]
    pt = axes["pt_centers"][None, None, :]
    e_sec = xl * e_proj
    p_l = np.sqrt(np.maximum(e_sec**2 - mass**2, 1e-12))
    return np.arctan2(pt, p_l)


def angle_moments(
    kernel: np.ndarray, axes: Dict[str, np.ndarray], mass: float = M_PION
) -> Dict[str, np.ndarray]:
    """Per (E_proj, x_L) production-angle moments, p_T-weighted.

    Returns dict with arrays of shape ``(n_proj, n_xl)``:
      ``e_sec``     -- secondary energy x_L * E_proj [GeV]
      ``theta_mean``-- <theta> [rad]
      ``theta_rms`` -- sqrt(<theta^2>) [rad]
      ``yield``     -- dN/dx_L (p_T-integrated weight), for weighting.
    """
    if axes.get("is_angular"):
        # Directly-binned (x_L, theta) kernel: theta axis is explicit.
        dth = np.diff(axes["theta_edges"])
        theta = np.radians(axes["theta_centers"])[None, None, :]
        w = kernel * dth[None, None, :]
    else:
        dpt = np.diff(axes["pt_edges"])
        theta = production_angle(axes, mass)  # (nE, nxl, npt)
        w = kernel * dpt[None, None, :]  # (nE, nxl, npt)
    norm = np.sum(w, axis=2)  # (nE, nxl) = dN/dx_L
    with np.errstate(invalid="ignore", divide="ignore"):
        th_mean = np.sum(w * theta, axis=2) / norm
        th2 = np.sum(w * theta**2, axis=2) / norm
    e_sec = axes["xl_centers"][None, :] * axes["proj_energies"][:, None]
    return {
        "e_sec": e_sec,
        "theta_mean": th_mean,
        "theta_rms": np.sqrt(np.clip(th2, 0, None)),
        "yield": norm,
    }


def mean_angle_vs_energy(
    kernel: np.ndarray,
    axes: Dict[str, np.ndarray],
    e_sec_edges: np.ndarray,
    mass: float = M_PION,
) -> Dict[str, np.ndarray]:
    """Aggregate the production angle as a function of *secondary* energy.

    All (E_proj, x_L) cells are pooled and binned by their secondary energy
    ``E_sec = x_L E_proj``; the yield-weighted mean angle is returned per bin.
    This is the quantitative "when does 3D matter" curve: <theta>(E_sec).
    """
    mom = angle_moments(kernel, axes, mass)
    e = mom["e_sec"].ravel()
    th = mom["theta_mean"].ravel()
    wt = mom["yield"].ravel()
    good = np.isfinite(th) & np.isfinite(wt) & (wt > 0)
    e, th, wt = e[good], th[good], wt[good]

    idx = np.digitize(e, e_sec_edges) - 1
    nb = len(e_sec_edges) - 1
    centers = np.sqrt(e_sec_edges[:-1] * e_sec_edges[1:])
    mean_th = np.full(nb, np.nan)
    for b in range(nb):
        sel = idx == b
        if np.any(sel) and wt[sel].sum() > 0:
            mean_th[b] = np.sum(th[sel] * wt[sel]) / np.sum(wt[sel])
    return {"e_sec": centers, "theta_mean": mean_th}


def crossover_energy(
    e_sec: np.ndarray, theta_mean_deg: np.ndarray, threshold_deg: float
) -> float:
    """Secondary energy [GeV] where <theta> falls below ``threshold_deg``.

    Linear interpolation in log-energy on the (decreasing) <theta>(E) curve.
    """
    good = np.isfinite(theta_mean_deg)
    e, t = e_sec[good], theta_mean_deg[good]
    order = np.argsort(e)
    e, t = e[order], t[order]
    below = t < threshold_deg
    if not np.any(below):
        return np.nan
    i = np.argmax(below)  # first crossing
    if i == 0:
        return float(e[0])
    # interpolate in log E between i-1 (above) and i (below)
    le = np.log(e[i - 1 : i + 1])
    tt = t[i - 1 : i + 1]
    frac = (threshold_deg - tt[0]) / (tt[1] - tt[0])
    return float(np.exp(le[0] + frac * (le[1] - le[0])))


# ---------------------------------------------------------------------------
# Discrete-ordinates scattering row
# ---------------------------------------------------------------------------
def discrete_ordinate_row(
    kernel: np.ndarray,
    axes: Dict[str, np.ndarray],
    i_proj: int,
    e_sec_target: float,
    mu_edges: np.ndarray,
    mass: float = M_PION,
) -> np.ndarray:
    """Normalized P(mu) for secondaries near ``e_sec_target`` at projectile bin
    ``i_proj``. ``mu = cos(theta)`` measured from the projectile axis.

    This is one row of the discrete-ordinates angular-scattering operator: the
    probability of being deflected into each mu bin. Rows are normalized to sum
    to 1 so they conserve probability.
    """
    e_proj = axes["proj_energies"][i_proj]
    xl = axes["xl_centers"]
    e_sec = xl * e_proj
    j = int(np.argmin(np.abs(e_sec - e_sec_target)))

    dpt = np.diff(axes["pt_edges"])
    pt = axes["pt_centers"]
    p_l = np.sqrt(max(e_sec[j] ** 2 - mass**2, 1e-12))
    mu = np.cos(np.arctan2(pt, p_l))  # (npt,)
    w = kernel[i_proj, j, :] * dpt  # (npt,)

    hist, _ = np.histogram(mu, bins=mu_edges, weights=w)
    total = hist.sum()
    return hist / total if total > 0 else hist


def angular_row(
    kernel: np.ndarray,
    axes: Dict[str, np.ndarray],
    i_proj: int,
    e_sec_target: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Smooth P(theta) row from a directly-binned (x_L, theta) kernel.

    No resampling: the theta axis is the one the kernel was histogrammed on, so
    the row is free of the comb artifact that ``discrete_ordinate_row`` shows
    when it converts a coarse p_T kernel.

    Returns
    -------
    theta_centers_deg, P_theta : np.ndarray
        Angle bin centers [deg] and the normalized angular density.
    """
    if not axes.get("is_angular"):
        raise ValueError("angular_row requires a directly-binned angular kernel")
    e_proj = axes["proj_energies"][i_proj]
    j = int(np.argmin(np.abs(axes["xl_centers"] * e_proj - e_sec_target)))
    dth = np.diff(axes["theta_edges"])
    row = kernel[i_proj, j, :]
    area = np.sum(row * dth)
    return axes["theta_centers"], (row / area if area > 0 else row)


# ---------------------------------------------------------------------------
# Gridless moments (Fokker-Planck input)
# ---------------------------------------------------------------------------
def pool_moments_by_energy(
    mom: Dict[str, np.ndarray], e_sec_edges: np.ndarray, mass: float = M_PION
) -> Dict[str, np.ndarray]:
    """Yield-weighted pooling of the (E_proj, x_L) moments by secondary energy.

    Because the moments were computed gridless (per secondary), the resulting
    ``<theta>(E_sec)`` and ``<theta^2>(E_sec)`` follow the 1/E law with no
    high-energy bin floor. ``D_theta = <theta^2>/2`` is the Fokker-Planck
    angular-diffusion coefficient and -> 0 as E_sec -> inf, recovering 1D.

    Also returns ``pt_eff = <theta> * p_L`` [GeV], the effective mean transverse
    momentum. ``<theta>`` deviates from a naive ``<p_T>/E`` with constant
    ``<p_T>`` precisely because ``pt_eff`` rises with energy (and turns over at
    low energy from mass/large-angle effects) -- this is physical, and is the
    quantity to validate against fixed-target data (NA61), not against MCEq
    (whose 1D kernels contain no angular information).
    """
    e = mom["e_sec"].ravel()
    th = mom["theta_mean"].ravel()
    th2 = mom["theta_sq"].ravel()
    wt = mom["dndx"].ravel()
    good = np.isfinite(th) & np.isfinite(th2) & (wt > 0)
    e, th, th2, wt = e[good], th[good], th2[good], wt[good]

    idx = np.digitize(e, e_sec_edges) - 1
    nb = len(e_sec_edges) - 1
    centers = np.sqrt(e_sec_edges[:-1] * e_sec_edges[1:])
    mean_th = np.full(nb, np.nan)
    mean_th2 = np.full(nb, np.nan)
    weight = np.zeros(nb)
    for b in range(nb):
        sel = idx == b
        w = wt[sel]
        if sel.any() and w.sum() > 0:
            mean_th[b] = np.sum(th[sel] * w) / w.sum()
            mean_th2[b] = np.sum(th2[sel] * w) / w.sum()
            weight[b] = w.sum()
    p_l = np.sqrt(np.maximum(centers**2 - mass**2, 1e-12))
    # Reliability mask: the highest-E_sec bins are reached only via the x_L -> 1
    # corner (the *leading particle*: p_T -> 0 by kinematics, vanishing yield), so
    # they are leading-particle-biased and statistics-starved -- not physical. A
    # bin is reliable only if lower-x_L (bulk-yield) projectiles can populate it,
    # i.e. E_sec < E_proj_max / 8 (so x_L can reach <~0.12), with data present.
    # ``e_sec.max() ~ x_L_max * E_proj_max``.
    e_proj_max = float(mom["e_sec"].max())
    reliable = np.isfinite(mean_th) & (centers < e_proj_max / 8.0) & (weight > 0)
    return {
        "e_sec": centers,
        "theta_mean": mean_th,
        "theta_sq": mean_th2,
        "d_theta": 0.5 * mean_th2,
        "pt_eff": mean_th * p_l,
        "weight": weight,
        "reliable": reliable,
    }


# ---------------------------------------------------------------------------
# CLI / demo
# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kernel", default="kernel_real_piplus.npz")
    p.add_argument("--threshold-deg", type=float, default=3.0)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args(argv)

    raw = np.load(args.kernel)
    if "is_moments" in raw:
        _run_moments_analysis(dict(raw), args)
        return

    kernel, axes = load_kernel(args.kernel)
    e_sec_edges = np.logspace(-0.3, 3.5, 24)  # ~0.5 GeV .. 3 TeV
    curve = mean_angle_vs_energy(kernel, axes, e_sec_edges)
    theta_deg = np.degrees(curve["theta_mean"])

    e_x = crossover_energy(curve["e_sec"], theta_deg, args.threshold_deg)
    print("Mean pion production angle vs secondary energy:")
    for e, t in zip(curve["e_sec"], theta_deg):
        if np.isfinite(t):
            print(f"  E_sec = {e:8.2f} GeV   <theta> = {t:6.2f} deg")
    print(
        f"\n<theta> crosses {args.threshold_deg:.0f} deg at "
        f"E_sec ~ {e_x:.1f} GeV  -> below this, 3D production geometry matters."
    )

    if args.plot:
        _plot(kernel, axes, curve, theta_deg, args)


def _run_moments_analysis(mom, args):
    e_sec_edges = np.logspace(-0.3, 4.0, 28)
    pooled = pool_moments_by_energy(mom, e_sec_edges)
    theta_deg = np.degrees(pooled["theta_mean"])
    e_x = crossover_energy(pooled["e_sec"], theta_deg, args.threshold_deg)
    print("Gridless <theta>(E_sec) from event moments (no theta-grid floor):")
    print("  E_sec [GeV]   <theta> [deg]   <pT>_eff [GeV]")
    for e, t, pt in zip(pooled["e_sec"], theta_deg, pooled["pt_eff"]):
        if np.isfinite(t):
            print(f"  {e:9.2f}     {t:7.3f}        {pt:.3f}")
    print(
        f"\n<theta> crosses {args.threshold_deg:.0f} deg at "
        f"E_sec ~ {e_x:.1f} GeV  -> below this, 3D production geometry matters."
    )
    if args.plot:
        _plot_moments(pooled, args)


def _plot_moments(pooled, args):
    import matplotlib.pyplot as plt

    fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(15, 4.3))
    e = pooled["e_sec"]
    th = np.degrees(pooled["theta_mean"])
    rel = pooled.get("reliable", np.isfinite(th))
    edge = np.isfinite(th) & ~rel  # leading-particle / low-stat edge bins

    def draw(ax, y, fmt, color, label, log=True):
        """Plot reliable bins solid; grey the unreliable edge bins."""
        plot = ax.loglog if log else ax.semilogx
        plot(e[rel], y[rel], fmt, color=color, label=label)
        if edge.any():
            plot(
                e[edge],
                y[edge],
                "x",
                color="0.6",
                ms=6,
                label="edge bins ($x_L\\!\\to\\!1$, unreliable)",
            )

    # Panel 1: <theta>(E). Deviates from the const-<pT> guide for real reasons.
    draw(axL, th, "o-", "C0", r"$\langle\theta\rangle$ (event moments)")
    axL.loglog(
        e[rel],
        np.degrees(0.30 / e[rel]),
        "k--",
        lw=1,
        label=r"const $\langle p_T\rangle{=}0.3$ (reference, not a fit)",
    )
    axL.axhline(args.threshold_deg, color="r", ls=":", lw=1)
    axL.axvspan(e[rel][0], 2.0, color="orange", alpha=0.15)
    axL.set_xlabel(r"secondary energy $E_{\rm sec}$ [GeV]")
    axL.set_ylabel(r"$\langle\theta\rangle$ [deg]")
    axL.set_title("Gridless: clean 1/E (edge bins greyed)")
    axL.legend()

    # Panel 2: the reason for the deviation -- <pT> rises with energy.
    pt = pooled["pt_eff"]
    draw(axM, pt, "s-", "C1", r"$\langle p_T\rangle_{\rm eff}$", log=False)
    axM.axhspan(0.30, 0.45, color="green", alpha=0.12, label="typical inclusive $\\pi$")
    axM.set_xlabel(r"secondary energy $E_{\rm sec}$ [GeV]")
    axM.set_ylabel(r"$\langle p_T\rangle_{\rm eff}=\langle\theta\rangle p_L$ [GeV]")
    axM.set_ylim(0, 0.7)
    axM.set_title("Why it deviates: $\\langle p_T\\rangle$ rises (validate vs NA61)")
    axM.legend()

    # Panel 3: Fokker-Planck diffusion coefficient -> 0 at high E.
    d = pooled["d_theta"]
    draw(axR, d, "o-", "C2", r"$D_\theta=\langle\theta^2\rangle/2$")
    axR.loglog(e[rel], (0.30 / e[rel]) ** 2 / 2, "k--", lw=1, label=r"$\propto E^{-2}$")
    axR.set_xlabel(r"secondary energy $E_{\rm sec}$ [GeV]")
    axR.set_ylabel(r"Fokker-Planck $D_\theta$ [rad$^2$]")
    axR.set_title(r"Diffusion coeff $\to 0$ at high E (recovers 1D)")
    axR.legend()

    fig.tight_layout()
    out = args.kernel.replace(".npz", ".png")
    fig.savefig(out, dpi=110)
    print(f"saved plot -> {out}")


def _plot(kernel, axes, curve, theta_deg, args):
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))

    good = np.isfinite(theta_deg)
    axL.loglog(
        curve["e_sec"][good],
        theta_deg[good],
        "o-",
        label=r"$\langle\theta\rangle$ (regenerated kernel)",
    )
    # analytic ~ <pT>/E guide
    e = curve["e_sec"][good]
    axL.loglog(
        e, np.degrees(0.30 / e), "k--", lw=1, label=r"$\langle p_T\rangle/E\simeq0.3/E$"
    )
    axL.axhline(args.threshold_deg, color="r", ls=":", lw=1)
    axL.axvspan(curve["e_sec"][0], 2.0, color="orange", alpha=0.15)
    axL.set_xlabel(r"secondary (lepton) energy $E_{\rm sec}$ [GeV]")
    axL.set_ylabel(r"mean production angle $\langle\theta\rangle$ [deg]")
    axL.set_title("Where 3D production geometry matters")
    axL.legend()

    i = len(axes["proj_energies"]) // 2
    if axes.get("is_angular"):
        for e_sec in (2.0, 10.0, 100.0):
            th, row = angular_row(kernel, axes, i, e_sec)
            if np.any(row > 0):
                axR.plot(th, row, label=f"$E_{{\\rm sec}}={e_sec:.0f}$ GeV")
        axR.set_title("Angular scattering row (direct binning, smooth)")
    else:
        # Coarse mu grid to limit aliasing from the discrete p_T binning.
        mu_edges = np.cos(np.radians(np.linspace(30.0, 0.0, 16)))
        mu_c = 0.5 * (mu_edges[:-1] + mu_edges[1:])
        for e_sec in (2.0, 10.0, 100.0):
            row = discrete_ordinate_row(kernel, axes, i, e_sec, mu_edges)
            if row.sum() > 0:
                axR.plot(
                    np.degrees(np.arccos(mu_c)),
                    row,
                    label=f"$E_{{\\rm sec}}={e_sec:.0f}$ GeV",
                )
        axR.set_title("Angular scattering row (resampled p_T, spiky)")
    axR.set_xlabel(r"deflection angle $\theta$ [deg]")
    axR.set_ylabel(r"$P(\theta)$ (discrete-ordinates row)")
    axR.legend()

    fig.tight_layout()
    out = args.kernel.replace(".npz", "_angular.png")
    fig.savefig(out, dpi=110)
    print(f"saved plot -> {out}")


if __name__ == "__main__":
    main()
