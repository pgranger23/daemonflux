"""
[PROTOTYPE] Research/de-risking scaffolding -- NOT part of the delivered flux
(mceq3d_flux). Kept for the record; do not depend on it in the paper. See
ARCHITECTURE.md.

Curved-atmosphere production geometry -- the Honda horizontal excess.

This supplies the one genuinely-3D ingredient that the multiplicative-correction
chain and the single-column cascade could not: the **off-axis / curved-shell
production geometry** that enhances the near-horizon flux at low energy.

Neutrinos free-stream in straight lines, so the geometric flux at the detector
from arrival zenith ``theta`` is the production integrated along the backward
line of sight through the spherical atmosphere::

    Phi_geom(theta)  ~  integral  rho(alt(l))  w(E, alt(l))  dl

where, for a detector at Earth radius ``R`` looking at zenith ``theta``, a point a
distance ``l`` up the ray sits at radius ``r(l) = sqrt(R^2 + l^2 + 2 R l cos theta)``
(altitude ``r-R``), ``rho`` is the air density and ``w`` the meson decay-to-neutrino
weight (->1 at low energy where mesons decay; -> meson interaction-dominated at
high energy).

Limits / validation
-------------------
* Vertical: matches the plane-parallel column (ratio 1 by construction).
* Small zenith: reproduces the flat ``sec theta`` enhancement.
* Horizon: the flat ``sec theta`` **diverges**, but the spherical integral
  **saturates** at a finite value ~ ``sqrt(R/h0)`` -- this finite near-horizon
  rise is the curved-atmosphere horizontal enhancement. At low (sub-GeV) energy,
  where mesons fully decay, the enhancement is largest; at high energy it is
  suppressed because mesons interact before decaying.

This is the geometric (neutrino-free-streaming) core of the effect; the full
treatment also spreads the *parent* directions (the angular kernels) and is
deferred. Magnitudes here are geometric/indicative, validated against the limits
above rather than against Honda tables (which require the full cascade).

Run::

    python spherical_geometry.py --plot
"""

from __future__ import annotations

import argparse

import numpy as np

R_EARTH = 6371.0  # km
H0 = 6.4  # density scale height [km]
H_TOP = 100.0  # top of atmosphere [km]


def density(alt_km):
    """Isothermal exponential air density (arbitrary normalization)."""
    return np.exp(-np.clip(alt_km, 0.0, None) / H0)


def _altitude_along_ray(l_km, zenith_deg):
    """Altitude [km] a distance l up the line of sight at the given zenith."""
    c = np.cos(np.deg2rad(zenith_deg))
    r = np.sqrt(R_EARTH**2 + l_km**2 + 2 * R_EARTH * l_km * c)
    return r - R_EARTH


def production_integral(zenith_deg, decay_weight=None, spherical=True, n=4000):
    """Line-of-sight production integral ~ integral rho(alt) w(alt) dl.

    ``decay_weight`` is an optional callable of altitude [km] (the meson
    decay-to-neutrino probability, energy dependent); default 1 (full decay,
    sub-GeV limit). ``spherical=False`` uses the plane-parallel alt = l cos theta.
    """
    # integrate to where the ray exits the atmosphere
    if spherical:
        c = np.cos(np.deg2rad(zenith_deg))
        # solve r(l)=R+H_TOP  ->  l = -R c + sqrt((R c)^2 + 2 R H + H^2)
        lmax = -R_EARTH * c + np.sqrt(
            (R_EARTH * c) ** 2 + 2 * R_EARTH * H_TOP + H_TOP**2
        )
    else:
        lmax = H_TOP / max(np.cos(np.deg2rad(zenith_deg)), 1e-3)
    ll = np.linspace(0, lmax, n)
    if spherical:
        alt = _altitude_along_ray(ll, zenith_deg)
    else:
        alt = ll * np.cos(np.deg2rad(zenith_deg))
    w = 1.0 if decay_weight is None else decay_weight(alt)
    return np.trapezoid(density(alt) * w, ll)


def horizon_enhancement(cos_zenith, decay_weight=None, spherical=True):
    """Phi_geom(theta)/Phi_geom(vertical) vs cos(zenith)."""
    cz = np.atleast_1d(cos_zenith).astype(float)
    vert = production_integral(0.0, decay_weight, spherical)
    out = np.array(
        [
            production_integral(
                np.degrees(np.arccos(np.clip(c, -1, 1))), decay_weight, spherical
            )
            for c in cz
        ]
    )
    return out / vert


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args(argv)

    cz = np.linspace(0.02, 1.0, 25)

    sph = horizon_enhancement(cz, spherical=True)
    flat = horizon_enhancement(cz, spherical=False)

    print("curved-atmosphere horizon enhancement (full-decay / sub-GeV limit):")
    print("  cosZ    spherical   flat(sec th)")
    for c, s, f in zip(cz, sph, flat):
        print(f"  {c:4.2f}    {s:8.2f}    {f:8.2f}")
    print(f"\n  flat at horizon (diverges):    {flat[0]:.1f}")
    print(
        f"  spherical at horizon (finite): {sph[0]:.1f}  ~ sqrt(R/h0)="
        f"{np.sqrt(R_EARTH/H0):.0f}"
    )
    print(f"  vertical (both = 1):           {sph[-1]:.3f}")

    if args.plot:
        _plot(cz, sph, flat)


def _plot(cz, sph, flat):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.plot(cz, sph, "C0o-", label="spherical (3D, finite at horizon)")
    ax.plot(cz, flat, "C3--", label=r"flat sec$\theta$ (diverges)")
    ax.axvspan(0.0, 0.1, color="orange", alpha=0.15)
    ax.set_xlabel(r"$\cos\theta_z$ (1=vertical, 0=horizon)")
    ax.set_ylabel(r"$\Phi_{\rm geom}/\Phi_{\rm vertical}$")
    ax.set_title("Curved-atmosphere horizontal enhancement (sub-GeV limit)")
    ax.set_ylim(0, min(20, flat[0] * 1.1))
    ax.legend()
    fig.tight_layout()
    fig.savefig("spherical_geometry.png", dpi=110)
    print("\nsaved plot -> spherical_geometry.png")


if __name__ == "__main__":
    main()
