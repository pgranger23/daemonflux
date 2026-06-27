"""Geomagnetic muon bending -- the non-cluster part of the off-axis 3D physics.

A muon produced in the cascade bends in the geomagnetic field before it decays
(mu -> e nu_e nubar_mu), so the decay neutrinos inherit a *deflected* direction.
This is a genuine driver of the sub-GeV up/down and East-West asymmetries that the
production-angle spread alone does not capture, and -- unlike the high-statistics
kernels -- it needs no cluster: the deflection is a closed-form quantity.

Key result (energy independence)
--------------------------------
The bending over the muon's in-flight decay is

    Delta_phi = (d / r_g) = (gamma c tau) * (q B / (gamma beta m c)) = q B tau / m

i.e. **independent of muon energy** (the longer flight of an energetic muon is
exactly offset by its larger gyroradius). Numerically ~3-5 deg for B ~ 0.3-0.5 G.
What *does* depend on energy is whether the muon decays in flight at all: low-E
muons decay in the atmosphere (and bend); high-E muons reach the ground first and
contribute no decay neutrino.

Two consequences for the neutrino flux, both delivered here:
* an **angular spread** (over the exponential decay-time distribution) that adds
  ``<Delta_phi^2>`` to the mu-decay-neutrino angular kernel
  (:func:`numu_bending_sigma2`, wired into ``mceq3d_solver``);
* a **coherent, charge-dependent shift** (mu+ and mu- bend oppositely) -> the
  East-West asymmetry of the mu-decay neutrinos (:func:`bending_deflection`,
  :func:`coherent_ew_shift_deg`, :func:`numu_ew_asymmetry`): mu+ shift ~+3 deg
  east and mu- ~-3 deg west, so the nu/nubar (charge-separated) flux carries a
  ~3 deg E-W split at sub-GeV, while the summed displacement is small (~0.3 deg,
  since the charge ratio R~1.27 is near unity).

Run::

    python muon_bending.py --plot
"""

from __future__ import annotations

import argparse

import numpy as np

TAU_MU = 2.1969811e-6  # s
M_MU = 0.1056584  # GeV/c^2
M_MU_KG = 1.883531e-28  # kg
Q = 1.602176634e-19  # C
C = 2.99792458e8  # m/s
CTAU_MU = C * TAU_MU  # ~659 m


def bending_angle(b_gauss=0.45):
    """Mean in-flight bending angle [rad] = q B tau / m (energy-independent)."""
    b_tesla = b_gauss * 1e-4
    return Q * b_tesla * TAU_MU / M_MU_KG


def decay_length_km(e_mu_gev):
    """Muon lab decay length [km] = gamma beta c tau ~ (E/m) c tau."""
    gamma = e_mu_gev / M_MU
    beta = np.sqrt(np.maximum(1 - 1 / gamma**2, 0.0))
    return gamma * beta * CTAU_MU / 1e3


def decay_in_flight_fraction(e_mu_gev, path_km=15.0):
    """Fraction of muons of energy E that decay within ``path_km`` of atmosphere
    (the rest reach the ground and produce no atmospheric decay neutrino)."""
    return 1.0 - np.exp(-path_km / decay_length_km(e_mu_gev))


def numu_bending_sigma2(e_nu_gev, b_gauss=0.45, path_km=15.0):
    """Angular-spread variance <theta^2> [rad^2] that muon bending adds to a
    mu-decay neutrino of energy ``e_nu_gev``.

    The neutrino carries ~1/3 of the muon energy on average, so E_mu ~ 3 E_nu.
    The spread is the bending angle weighted by the decay-in-flight fraction
    (only decaying muons contribute); the RMS over the exponential decay-time
    distribution equals the mean bending angle (Delta_phi), so var ~ f * Dphi^2.
    """
    e_mu = 3.0 * np.asarray(e_nu_gev, dtype=float)
    f = decay_in_flight_fraction(e_mu, path_km)
    return f * bending_angle(b_gauss) ** 2


def bending_deflection(vel_hat, b_gauss_enu, charge=+1):
    """Coherent in-flight deflection *vector* [rad] of a muon, ``(q tau/m)(v x B)``.

    ``vel_hat`` and ``b_gauss_enu`` are in the local (east, north, up) frame; the
    result is perpendicular to ``vel_hat`` and its magnitude is the (energy-
    independent) bending angle for the component of B transverse to the velocity.
    The decay neutrino inherits this deflection, so it is also the arrival-
    direction shift of the mu-decay neutrino.
    """
    b_t = np.asarray(b_gauss_enu, float) * 1e-4  # gauss -> tesla
    coeff = charge * Q * TAU_MU / M_MU_KG  # rad per tesla (B perp v)
    return coeff * np.cross(np.asarray(vel_hat, float), b_t)


def coherent_ew_shift_deg(b_north_gauss=0.30, charge=+1):
    """East-West (azimuthal) arrival shift [deg] of mu-decay nu for a *vertical*
    muon: only the horizontal (north) field bends it east/west.

    For a downward muon ``v=-up``, ``(v x B)`` has east-component ``+B_north``, so
    mu+ shifts east and mu- shifts west -- the coherent charge-dependent E-W
    deflection (Honda). Magnitude = (q tau/m) B_north, energy-independent.
    """
    d = bending_deflection([0, 0, -1.0], [0.0, b_north_gauss, 0.0], charge)
    return np.degrees(d[0])  # east component


def numu_ew_asymmetry(e_nu_gev, b_north_gauss=0.30, charge_ratio=1.27, path_km=15.0):
    """Net East-West azimuthal displacement [deg] of the *summed* mu-decay
    neutrino flux from coherent muon bending.

    mu+ and mu- deflect oppositely (+/- the coherent shift); with a muon charge
    ratio ``R = N+/N-`` the net displacement of the (nu_mu + nubar_mu) sum is
    ``shift * (R-1)/(R+1)``, gated by the decay-in-flight fraction (sub-GeV only).
    The charge-*separated* shift (full +/- the value, for nu vs nubar) is the
    larger, flavour-dependent effect; return both.
    """
    e_mu = 3.0 * np.asarray(e_nu_gev, dtype=float)
    f = decay_in_flight_fraction(e_mu, path_km)
    shift = coherent_ew_shift_deg(b_north_gauss, charge=+1)  # mu+ east
    net = shift * (charge_ratio - 1) / (charge_ratio + 1) * f
    return dict(net_shift_deg=net, charge_separated_deg=shift * f)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args(argv)

    print("muon in-flight bending angle (energy-independent):")
    for b in (0.30, 0.45, 0.60):
        print(
            f"  B = {b:.2f} G  ->  Delta_phi = {np.degrees(bending_angle(b)):.2f} deg"
        )

    print("\ndecay-in-flight fraction (15 km path) & neutrino bending spread:")
    print("  E_nu[GeV]   E_mu[GeV]   f_decay   sqrt<theta^2>_bend [deg]")
    for enu in (0.3, 1.0, 3.0, 10.0, 30.0):
        emu = 3 * enu
        f = decay_in_flight_fraction(emu)
        s = np.degrees(np.sqrt(numu_bending_sigma2(enu)))
        print(f"  {enu:7.1f}    {emu:7.1f}    {f:6.3f}     {s:6.2f}")

    print("\ncoherent charge-dependent E-W shift (B_north=0.30 G, vertical muon):")
    print(
        f"  mu+ shifts {coherent_ew_shift_deg(charge=+1):+.2f} deg (east), "
        f"mu- {coherent_ew_shift_deg(charge=-1):+.2f} deg (west)"
    )
    print("  net (nu_mu+nubar_mu) E-W displacement, charge ratio 1.27:")
    print("  E_nu[GeV]   net_shift[deg]   charge-separated[deg]")
    for enu in (0.3, 1.0, 3.0):
        a = numu_ew_asymmetry(enu)
        print(
            f"  {enu:7.1f}    {a['net_shift_deg']:+8.2f}       "
            f"+/-{a['charge_separated_deg']:.2f}"
        )

    if args.plot:
        _plot()


def _plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    enu = np.logspace(-0.7, 2, 60)
    fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(15, 4.3))
    axL.semilogx(enu, decay_in_flight_fraction(3 * enu), "C0-")
    axL.set_xlabel(r"$E_\nu$ [GeV]")
    axL.set_ylabel("muon decay-in-flight fraction")
    axL.set_title("Only low-E muons decay (and bend) in the atmosphere")

    s = np.degrees(np.sqrt(numu_bending_sigma2(enu)))
    axM.semilogx(enu, s, "C3-", label=r"$\sqrt{\langle\theta^2\rangle}_{\rm bend}$")
    axM.axhline(
        np.degrees(bending_angle()),
        color="k",
        ls=":",
        label=r"$\Delta\phi$ (energy-indep.)",
    )
    axM.axvspan(0.3, 2.0, color="orange", alpha=0.15)
    axM.set_xlabel(r"$E_\nu$ [GeV]")
    axM.set_ylabel("bending angular spread [deg]")
    axM.set_title(r"$\mu$-decay-$\nu$ bending: sub-GeV only")
    axM.legend()

    sep = np.array([numu_ew_asymmetry(e)["charge_separated_deg"] for e in enu])
    net = np.array([numu_ew_asymmetry(e)["net_shift_deg"] for e in enu])
    axR.semilogx(enu, sep, "C0-", label=r"$\nu$ vs $\bar\nu$ (charge-separated)")
    axR.semilogx(enu, -sep, "C0-")
    axR.semilogx(enu, net, "C3--", label=r"net ($\nu_\mu+\bar\nu_\mu$), R=1.27")
    axR.axhline(0, color="k", lw=0.6)
    axR.axvspan(0.3, 2.0, color="orange", alpha=0.15)
    axR.set_xlabel(r"$E_\nu$ [GeV]")
    axR.set_ylabel("coherent E-W azimuthal shift [deg]")
    axR.set_title("Coherent muon-bending East-West shift")
    axR.legend()
    fig.tight_layout()
    fig.savefig("muon_bending.png", dpi=110)
    print("\nsaved plot -> muon_bending.png")


if __name__ == "__main__":
    main()
