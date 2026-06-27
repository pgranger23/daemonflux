"""Production 3D engine: MCEq solved per multipole (full physics + angular layer).

This is the production-grade architecture. Instead of reimplementing the cascade
with toy yields, it **wraps MCEq**: the angular distribution of each species about
the shower axis is expanded in Legendre modes ``c_l``, and -- because a forward
production kick of RMS angle ``theta1`` multiplies the mode by
``exp(-l(l+1) theta1^2/4)`` (the heat kernel) -- each multipole is obtained by
running MCEq with its meson-production yields scaled by that factor (via
``MCEqRun.set_mod_pprod``). So:

* ``l = 0``  -> the *unmodified* MCEq solve == standard MCEq / daemonflux flux
  (full species, charge, flavour, real atmosphere, real hadronic yields);
* ``l > 0``  -> the same production-grade cascade with the validated
  (NA61-checked) angular kernel applied at every modified production vertex,
  giving the self-consistent angular structure.

The whole flavour/charge/atmosphere physics therefore comes from MCEq for free;
our contribution is the validated angular layer ``theta1(E)``. Cost = n_l x (MCEq
regenerate+solve).

What still needs the cluster / external inputs (not here): the high-statistics
double-differential kernels (only ``theta1(E)`` variance is used now), IGRF
back-tracing (Stoermer scaffold used for the primary cutoff), and the muon-
bending / full off-axis geometric horizon excess.

Run::

    python mceq3d_production.py --lmax 8 --plot
"""

from __future__ import annotations

import argparse

import numpy as np

# proton/neutron -> charged pi/K are the dominant angular-kick production vertices
_CHANNELS = [
    (2212, 211),
    (2212, -211),
    (2212, 321),
    (2212, -321),
    (2112, 211),
    (2112, -211),
    (2112, 321),
    (2112, -321),
]


def _theta1_on(e_grid):
    from fokker_planck_3d import load_theta2, theta2_interp

    e_sig, t2 = load_theta2("m_spliced.npz")
    return np.sqrt(theta2_interp(e_grid, e_sig, t2))


# module globals so set_mod_pprod's args stay hashable (scalar l only)
_TH1 = None  # theta1(E) grid [rad]
_EREF = None  # energy grid [GeV]


def _kernel_func(xmat, e_c, ll, *rest):
    """modmat for set_mod_pprod: multiply production by exp(-l(l+1) theta1^2/4)
    at the secondary energy E_sec = x * E_prim. (MCEq's Barr machinery passes a
    second scalar arg and may rescale it for K0 isospin; we depend only on l, so
    that rescaling is harmless and isospin auto-propagation just applies the same
    angular kernel to the partner channel -- which is what we want.)"""
    e_sec = xmat * e_c[None, :]
    th = np.interp(
        np.log(np.maximum(e_sec, 1e-9)),
        np.log(_EREF),
        _TH1,
        left=_TH1[0],
        right=_TH1[-1],
    )
    return np.exp(-ll * (ll + 1) * th**2 / 4.0)


def solve_multipoles(interaction_model="SIBYLL23D", lmax=8, quantity="total_numu"):
    """Return (e_grid, c_l[n_l, n_e]) for the given neutrino quantity."""
    from MCEq.core import MCEqRun
    import crflux.models as crf

    mceq = MCEqRun(
        interaction_model=interaction_model,
        primary_model=(crf.HillasGaisser2012, "H3a"),
        theta_deg=0.0,
    )
    e = mceq.e_grid
    global _TH1, _EREF
    _EREF = e
    _TH1 = _theta1_on(e)

    n_l = lmax + 1
    c = np.zeros((n_l, len(e)))
    for ll in range(n_l):
        if ll == 0:
            mceq.unset_mod_pprod(dont_fill=True)
        else:
            for pp, ss in _CHANNELS:
                # second arg is a dummy required by MCEq's Barr-style API
                mceq.set_mod_pprod(pp, ss, _kernel_func, (ll, 1.0), delay_init=True)
        mceq.regenerate_matrices()
        mceq.solve()
        c[ll] = mceq.get_solution(quantity, mag=0)
        mceq.unset_mod_pprod(dont_fill=True)
    return e, c


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="SIBYLL23D")
    p.add_argument("--lmax", type=int, default=8)
    p.add_argument("--quantity", default="total_numu")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args(argv)

    e, c = solve_multipoles(args.model, args.lmax, args.quantity)

    # validation: l=0 must equal the plain MCEq solve
    from MCEq.core import MCEqRun
    import crflux.models as crf

    ref = MCEqRun(
        interaction_model=args.model,
        primary_model=(crf.HillasGaisser2012, "H3a"),
        theta_deg=0.0,
    )
    ref.solve()
    f0 = ref.get_solution(args.quantity, mag=0)
    m = f0 > 0
    rel = np.nanmax(np.abs(c[0][m] / f0[m] - 1.0))
    print(
        f"VALIDATION  l=0 vs plain MCEq:  max rel diff = {rel:.2e}  "
        "(0 => the wrapper is exact; full MCEq physics in l=0)"
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        sigma = np.degrees(np.arccos(np.clip(c[1] / c[0], -1, 1)))
    print("\nself-consistent angular spread of the MCEq flux:")
    print("  E[GeV]    sigma_theta[deg]")
    for i in range(0, len(e), 6):
        if 0.5 < e[i] < 5000 and np.isfinite(sigma[i]):
            print(f"  {e[i]:9.2f}    {sigma[i]:7.2f}")

    if args.plot:
        _plot(e, c, f0, sigma, args)


def _plot(e, c, f0, sigma, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))
    s = (e > 0.3) & (e < 1e6) & (f0 > 0)
    axL.loglog(e[s], (c[0] * e**3)[s], "C0-", lw=2, label="l=0 (= MCEq flux)")
    axL.set_xlabel("E [GeV]")
    axL.set_ylabel(r"$E^3\,\Phi_{\nu_\mu}$")
    axL.set_title("Production-grade flux (full MCEq) in l=0")
    axL.legend()

    a = (e > 0.5) & (e < 5000) & np.isfinite(sigma)
    axR.loglog(e[a], sigma[a], "C0o-", ms=3)
    axR.axvspan(0.5, 2.0, color="orange", alpha=0.15)
    axR.set_xlabel("E [GeV]")
    axR.set_ylabel(r"$\sigma_\theta$ [deg]")
    axR.set_title("Validated angular layer on the MCEq flux")
    fig.tight_layout()
    fig.savefig("mceq3d_production.png", dpi=110)
    print("\nsaved plot -> mceq3d_production.png")


if __name__ == "__main__":
    main()
