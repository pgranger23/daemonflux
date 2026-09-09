"""How much production-cone weight is Earth-shadowed, and where the samples land.

For a set of down-going neutrino directions it reproduces the cone sampling of
``MCEq3DFlux.cone_geff`` and reports, per energy:

* the fraction of the cone weight that falls below the DETECTOR's local horizon
  (the samples the legacy code sent to the antipodal far-side map);
* the fraction still below the horizon at the PRODUCTION point, i.e. genuinely
  Earth-shadowed and blocked by ``sublimb="prod_point"``;
* the mean local-zenith shift between the two frames.

Run::

    python diag_cutoff_block.py
"""

from __future__ import annotations

import numpy as np

from mceq3d_flux import RE_KM, H_PROD_KM

E_PROBE = (0.3, 0.5, 1.0, 3.0)
ZEN = (60.0, 75.0, 81.0, 84.0, 87.0, 89.0)
SIGMA_PI = {0.3: 38.9, 0.5: 14.6, 1.0: 8.9, 3.0: 4.1}  # space-angle RMS [deg]


def prod_frame(n_vec, h_km=H_PROD_KM):
    cz_n = float(n_vec[2])
    r = RE_KM + h_km
    disc = (RE_KM * cz_n) ** 2 + (r * r - RE_KM * RE_KM)
    L = -RE_KM * cz_n + np.sqrt(max(disc, 0.0))
    P = np.array([L * n_vec[0], L * n_vec[1], RE_KM + L * n_vec[2]])
    up_p = P / np.linalg.norm(P)
    north_p = np.array([1.0, 0.0, 0.0])
    north_p = north_p - np.dot(north_p, up_p) * up_p
    return L, up_p, north_p / np.linalg.norm(north_p)


def main():
    alpha_deg = np.linspace(0.5, 70.0, 12)
    beta = np.linspace(0.0, 2 * np.pi, 12, endpoint=False)
    print("zen  L[km] tilt[deg]  E[GeV]  w(below det. horizon)  w(blocked at P)")
    for zdeg in ZEN:
        th, phi = np.radians(zdeg), np.radians(90.0)  # from the East
        n = np.array([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi),
                      np.cos(th)])
        L, up_p, north_p = prod_frame(n)
        e1 = np.array([0.0, 0.0, 1.0]) - n[2] * n
        e1 = e1 / np.linalg.norm(e1) if np.linalg.norm(e1) > 1e-9 else \
            np.array([1.0, 0.0, 0.0])
        e2 = np.cross(n, e1)
        th_det = np.empty((len(alpha_deg), len(beta)))
        th_prod = np.empty_like(th_det)
        for ka, a in enumerate(np.radians(alpha_deg)):
            for kb, b in enumerate(beta):
                v = np.cos(a) * n + np.sin(a) * (np.cos(b) * e1 + np.sin(b) * e2)
                th_det[ka, kb] = np.degrees(np.arccos(np.clip(v[2], -1, 1)))
                th_prod[ka, kb] = np.degrees(
                    np.arccos(np.clip(float(np.dot(v, up_p)), -1, 1)))
        for E in E_PROBE:
            s = SIGMA_PI[E]
            w = (np.exp(-0.5 * (alpha_deg / s) ** 2)
                 * np.sin(np.radians(alpha_deg)))
            w = w / w.sum()
            f_det = float(np.sum(w * (th_det > 90.0).mean(1)))
            f_pro = float(np.sum(w * (th_prod > 90.0).mean(1)))
            print(f"{zdeg:4.0f} {L:6.0f} {np.degrees(L / RE_KM):8.2f}  {E:6.2f}"
                  f"  {f_det:20.3f}  {f_pro:14.3f}")
    print("\nDIAG_CUTOFF_BLOCK_DONE")


if __name__ == "__main__":
    main()
