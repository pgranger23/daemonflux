"""Task-8 fix attempt: the PRODUCTION-REGION-INTEGRATED cutoff. Instead of evaluating
the geomagnetic cutoff at the detector (cone_geff) we average G_s(R_c) over the
actual production points along the near-horizontal line of sight, weighted by the
local neutrino production -- the effect Honda's full-3D MC has and cone_geff lacks.

For 87 deg East and West: sample production points P_m at altitudes along the
arrival ray; weight by rho(h) * p(X_slant(h,psi_o), E) * dl (the local numu
production); compute R_c(P_m) at each point's location + local primary direction
(IGRF back-trace); and form the production-weighted effective East/West cutoff via
G_s. Compare to the detector cutoff (~42/7), our delivered cone_geff-effective
(33-39/rising), and Honda's implied (~28/flat, diag_ew_invert).

If the production-region-effective East cutoff -> ~28 flat, this is the fix (wire it
into cone_geff). If it stalls near ~38, the production-region spatial averaging is
NOT the missing softening. Run from tools/mceq3d (PYTHONPATH=$PWD). Needs ppigrf.
"""
from datetime import datetime

import numpy as np

import offaxis_mc as om
from offaxis_mc import (production_profile, slant_depth_table, _rho_of_h,
                        _interp_xslant, _p_at, R_EARTH_CM)
import geomag_backtrace as gb
from geomag_backtrace import _local_frame, arrival_direction, RE

LAT, LON = 36.43, 137.31
DATE = datetime(2020, 1, 1)


def latlon_of(c):
    r = c / np.linalg.norm(c)
    return np.degrees(np.arcsin(np.clip(r[2], -1, 1))), np.degrees(np.arctan2(r[1], r[0]))


def local_dir(lat, lon, vel):
    u, n, e = _local_frame(lat, lon)
    f = -vel / np.linalg.norm(vel)
    cz = float(f @ u); h = f - cz * u
    return float(np.degrees(np.arccos(np.clip(cz, -1, 1)))), \
        np.degrees(np.arctan2(h @ e, h @ n)) % 360.0


def main():
    print("MCEq production profile p(X,E) + slant geometry ...")
    x_grid, ep_grid, p, dm = production_profile()
    om._RHO = _rho_of_h(dm)
    geom = slant_depth_table(om._RHO)
    hg, pg, tbl = geom
    E = 1.0
    jE = int(np.argmin(np.abs(ep_grid - E)))
    ptot = p["tot"]

    from mceq3d_flux import MCEq3DFlux
    eng = MCEq3DFlux(interaction_model="SIBYLL23D",
                     primary=("HillasGaisser2012", "H3a"),
                     base_model="mceq", daemonflux_location="kamioka")
    rc_grid = np.linspace(4.0, 60.0, 40)
    G, rc_grid = eng.geomag_response(rc_grid, cache_dir=".cache3d")
    gcol = G["total_numu"][:, int(np.argmin(np.abs(eng.e - E)))]  # G_s(rc) at E

    def eff_cutoff(rc_vals, w):
        gs = np.interp(rc_vals, rc_grid, gcol)
        gbar = np.sum(w * gs) / np.sum(w)
        order = np.argsort(gcol)
        return float(np.interp(gbar, gcol[order], rc_grid[order]))

    theta = np.arccos(0.05)  # 87 deg
    up = _local_frame(LAT, LON)[0]
    P_det = RE * up                        # RE is in METERS (geomag_backtrace)
    h_km = np.array([4, 7, 10, 14, 18, 24, 32, 45])  # production-point altitudes
    h_m = h_km * 1e3                       # metres, for the geomag ray geometry
    h_cm = h_km * 1e5                      # cm, for om._RHO / _interp_xslant

    print(f"\nProduction-region-integrated cutoff at 87 deg (E={E} GeV):")
    print(f"{'dir':>4} {'Rc_detector':>12} {'Rc_prodregion(eff)':>19}  Honda~28(E)/~? (W)")
    for name, az in (("E", 90.0), ("W", 270.0)):
        d = arrival_direction(LAT, LON, theta * 180 / np.pi, az)  # velocity (down)
        r = RE + h_m                       # metres (was RE[m] + h_cm[cm]: 100x bug)
        disc = (RE * np.cos(theta)) ** 2 + (r ** 2 - RE ** 2)
        ell = -RE * np.cos(theta) + np.sqrt(np.clip(disc, 0, None))  # detector->alt [m]
        cos_psi_o = np.clip((ell + RE * np.cos(theta)) / r, -1, 1)
        psi_o = np.arccos(cos_psi_o)
        # slant depth to each point + local production weight
        X = np.array([_interp_xslant(h_cm[i], psi_o[i], hg, pg, tbl)
                      for i in range(len(h_cm))])
        p_o = _p_at(X, x_grid, ep_grid, ptot)[:, jE]
        dl = np.gradient(ell)
        w = om._RHO(h_cm) * np.abs(dl) * np.maximum(p_o, 0.0)
        # cutoff at each production point (its location + local primary direction)
        rc_vals = []
        for i in range(len(h_cm)):
            Pm = P_det - ell[i] * d
            laq, loq = latlon_of(Pm)
            zp, ap = local_dir(laq, loq, d)
            rc_vals.append(gb.cutoff_map(laq, loq, DATE, [zp], [ap], n_scan=28)[0, 0])
        rc_vals = np.array(rc_vals)
        rc_det = gb.cutoff_map(LAT, LON, DATE, [theta * 180 / np.pi], [az],
                               n_scan=28)[0, 0]
        rc_eff = eff_cutoff(rc_vals, w)
        print(f"{name:>4} {rc_det:12.1f} {rc_eff:19.1f}")
        print(f"       R_c along path (h={list(h_km)} km): "
              f"{np.round(rc_vals,1).tolist()}")
        print(f"       prod weight (norm):                 "
              f"{np.round(w/max(w.max(),1e-30),2).tolist()}")
    print("\nrc_eff -> ~28 (E) flat => production-region integration is the fix.")
    print("rc_eff stalls near ~38 => spatial averaging is NOT the missing softening.")
    print("DIAG_EW_PRODREGION_DONE")


if __name__ == "__main__":
    main()
