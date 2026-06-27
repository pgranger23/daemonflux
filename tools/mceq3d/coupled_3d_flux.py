"""Couple the validated angular operator to the real 1D MCEq flux -> 3D flux.

This closes the loop: MCEq provides the calibrated 1D *energy* cascade
(Phi_1D(E, cos theta), azimuthally symmetric), and the Fokker-Planck angular
operator (driven by the NA61-validated <theta^2>(E)) redistributes it in
arrival direction. A neutrino observed from direction n_o was produced along a
spread of directions n_s about it, so

    Phi_3D(n_o, E) = integral  Phi_1D(n_s, E)  K(n_o . n_s; sigma_theta(E)) dOmega_s

with K a normalized angular-spread kernel (2D Gaussian of width sigma_theta).
The convolution is done on the sphere by Monte Carlo (exact rotation, valid for
any sigma), preserving total flux. At high energy sigma_theta -> 0, so
Phi_3D -> Phi_1D and the calibrated 1D result (daemonflux) is recovered; the
sub-GeV regime is where the redistribution bites -- strongest near the horizon,
where the 1D flux varies fastest and the slant depth (hence sigma_theta) is
largest.

Run::

    python coupled_3d_flux.py --moments m_spliced.npz --plot
"""

from __future__ import annotations

import argparse

import numpy as np

from fokker_planck_3d import load_theta2, sigma_theta_vs_energy


# ---------------------------------------------------------------------------
# 1D flux from MCEq
# ---------------------------------------------------------------------------
def mceq_1d_flux(quantity="total_numu", zeniths_deg=None, model="SIBYLL23D"):
    """Solve MCEq at several zenith angles -> Phi_1D(E, cos theta).

    Returns ``e_grid``, ``cos_full`` (ascending, mirrored to [-1, 1] using the
    up/down symmetry of the conventional flux), and ``flux`` of shape
    ``(n_cos_full, n_E)``.
    """
    from MCEq.core import MCEqRun
    import crflux.models as crf

    if zeniths_deg is None:
        zeniths_deg = np.array([0, 25, 40, 55, 70, 80, 85, 89.0])
    mceq = MCEqRun(
        interaction_model=model,
        primary_model=(crf.HillasGaisser2012, "H3a"),
        theta_deg=0.0,
    )
    e_grid = mceq.e_grid
    cos_pos = np.cos(np.deg2rad(zeniths_deg))  # down-going, in (0, 1]
    fl = np.zeros((len(zeniths_deg), len(e_grid)))
    for i, th in enumerate(zeniths_deg):
        mceq.set_theta_deg(float(th))
        mceq.solve()
        fl[i] = mceq.get_solution(quantity, mag=0)

    # Mirror to the full sphere (Phi even in cos theta for the conventional flux).
    order = np.argsort(cos_pos)
    cos_pos, fl = cos_pos[order], fl[order]
    cos_full = np.concatenate([-cos_pos[::-1], cos_pos])
    flux_full = np.vstack([fl[::-1], fl])
    return e_grid, cos_full, flux_full


# ---------------------------------------------------------------------------
# Spherical convolution with the validated angular spread
# ---------------------------------------------------------------------------
def sigma_theta_grid(e_grid, e_sig, theta2, zeniths_deg):
    """sigma_theta(E, zenith) [rad].

    The production-angle spread accumulates over the finite parent chain and is
    **zenith-independent** (set by kinematics, not column depth), so the same
    sigma_theta(E) applies at every zenith. (The earlier sqrt(sec theta) scaling
    came from the unphysical slant/lambda generation count and has been removed;
    genuine geometric zenith effects are a separate, not-yet-included ingredient.)
    """
    sig_vert = np.deg2rad(sigma_theta_vs_energy(e_sig, theta2, e_grid, zenith_deg=0.0))
    return np.tile(sig_vert, (len(np.atleast_1d(zeniths_deg)), 1))  # (n_zen, n_E)


def convolve_sphere(
    e_grid, cos_full, flux_full, sigma_EZ, out_cos, e_indices, n_mc=4000, seed=0
):
    """Phi_3D(out_cos, E) by MC convolution on the sphere for selected energies.

    ``sigma_EZ`` is sigma_theta on (out_cos, E) [rad]. Returns array
    ``(len(out_cos), len(e_indices))``.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros((len(out_cos), len(e_indices)))
    for ic, c_o in enumerate(out_cos):
        th_o = np.arccos(np.clip(c_o, -1, 1))
        sin_o, cos_o = np.sin(th_o), np.cos(th_o)
        for je, ie in enumerate(e_indices):
            s = sigma_EZ[ic, ie] / np.sqrt(2.0)  # per-component scale
            if s < 1e-4:
                out[ic, je] = np.interp(c_o, cos_full, flux_full[:, ie])
                continue
            u = rng.random(n_mc)
            alpha = s * np.sqrt(-2.0 * np.log(np.clip(u, 1e-12, 1)))  # Rayleigh
            beta = rng.uniform(0, 2 * np.pi, n_mc)
            cos_s = np.cos(alpha) * cos_o - np.sin(alpha) * np.cos(beta) * sin_o
            phi_s = np.interp(cos_s, cos_full, flux_full[:, ie])
            out[ic, je] = phi_s.mean()
    return out


# ---------------------------------------------------------------------------
# CLI / demo
# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--moments", default="m_spliced.npz")
    p.add_argument("--quantity", default="total_numu")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args(argv)

    e_sig, theta2 = load_theta2(args.moments)
    e_grid, cos_full, flux_full = mceq_1d_flux(args.quantity)

    out_cos = np.linspace(0.02, 1.0, 25)  # down-going: horizon -> vertical
    sigma_EZ = sigma_theta_grid(e_grid, e_sig, theta2, np.arccos(out_cos) * 180 / np.pi)

    e_targets = [0.5, 1.0, 3.0, 10.0, 100.0]
    e_idx = [int(np.argmin(np.abs(e_grid - e))) for e in e_targets]
    phi3d = convolve_sphere(e_grid, cos_full, flux_full, sigma_EZ, out_cos, e_idx)
    phi1d = np.array([np.interp(out_cos, cos_full, flux_full[:, ie]) for ie in e_idx]).T

    print("Phi_3D / Phi_1D  vs cos(zenith):")
    print("  cosZ  " + "".join(f"{e:>8.1f}GeV" for e in e_targets))
    for ic in range(0, len(out_cos), 3):
        ratios = phi3d[ic] / phi1d[ic]
        print(f"  {out_cos[ic]:4.2f}  " + "".join(f"{r:11.3f}" for r in ratios))

    if args.plot:
        _plot(out_cos, phi3d, phi1d, e_grid, e_idx, e_targets, args)


def _plot(out_cos, phi3d, phi1d, e_grid, e_idx, e_targets, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))
    for je, e in enumerate(e_targets):
        axL.plot(out_cos, phi3d[:, je] / phi1d[:, je], "o-", ms=3, label=f"{e:.1f} GeV")
    axL.axhline(1.0, color="k", lw=0.8, ls=":")
    axL.set_xlabel(r"$\cos\theta_z$ (1=vertical, 0=horizon)")
    axL.set_ylabel(r"$\Phi_{3D}/\Phi_{1D}$")
    axL.set_title("3D angular redistribution of the numu flux")
    axL.legend()

    je = e_targets.index(1.0)
    axR.plot(out_cos, phi1d[:, je] / phi1d[:, je].max(), "k--", label="1D (MCEq)")
    axR.plot(out_cos, phi3d[:, je] / phi1d[:, je].max(), "C0-", label="3D (coupled)")
    axR.set_xlabel(r"$\cos\theta_z$")
    axR.set_ylabel(r"$\Phi$ (norm.)  at 1 GeV")
    axR.set_title("Zenith shape smeared by sub-GeV angular spread")
    axR.legend()

    fig.tight_layout()
    fig.savefig("coupled_3d_flux.png", dpi=110)
    print("saved plot -> coupled_3d_flux.png")


if __name__ == "__main__":
    main()
