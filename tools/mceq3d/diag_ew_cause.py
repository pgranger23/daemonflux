"""Task-7: what causes the extreme-horizon E-W overshoot? Test candidate (a): the
cutoff at the DISPLACED production point vs at the detector.

Since the delivered G_s(R_c,E) is cascade-correct and shared by East and West, the
W/E ratio at fixed E is set by the two cutoff values R_c(E), R_c(W). At 87 deg the
neutrino is produced ~270 km away along the near-horizontal line of sight, where
the local geomagnetic cutoff differs from the detector's. If the East cutoff drops
at the production point, the East is less suppressed -> W/E falls toward Honda.

This computes R_c at the detector and at the production point (h_prod=20 km) for the
87 deg East and West arrival directions, and the implied change in the cutoff
contrast. Run from tools/mceq3d (PYTHONPATH=$PWD). Needs ppigrf.
"""
from datetime import datetime

import numpy as np

import geomag_backtrace as gb
from geomag_backtrace import _local_frame, arrival_direction, RE

LAT, LON = 36.43, 137.31
DATE = datetime(2020, 1, 1)
H_PROD_KM = 20.0


def latlon_of(cart):
    r = cart / np.linalg.norm(cart)
    return np.degrees(np.arcsin(np.clip(r[2], -1, 1))), np.degrees(np.arctan2(r[1], r[0]))


def local_dir(lat, lon, vel_cart):
    """(zenith, azimuth) of a primary with velocity ``vel_cart`` in the local frame
    at (lat, lon). Primary arrives from -vel; zenith is of the from-direction."""
    up, north, east = _local_frame(lat, lon)
    frm = -vel_cart / np.linalg.norm(vel_cart)  # from-direction
    cz = float(frm @ up)
    horiz = frm - cz * up
    az = np.degrees(np.arctan2(horiz @ east, horiz @ north)) % 360.0
    return float(np.degrees(np.arccos(np.clip(cz, -1, 1)))), az


def main():
    up = _local_frame(LAT, LON)[0]
    P_det = RE * up
    Rh = RE + H_PROD_KM * 1e3
    zen = 87.0
    print(f"E-W cutoff at 87 deg: detector vs displaced production point "
          f"(h={H_PROD_KM:.0f} km)")
    print(f"{'dir':>4} {'az':>5} {'L[km]':>7} {'dlat':>6} {'dlon':>6} "
          f"{'Rc_det':>7} {'Rc_prod':>8}")
    res = {}
    for name, az in (("E", 90.0), ("W", 270.0)):
        d = arrival_direction(LAT, LON, zen, az)      # neutrino velocity (downward)
        # production point Q = P_det - L d (back up the ray) with |Q| = Rh.
        # |Q|^2 = RE^2 - 2 L (P.d) + L^2 = Rh^2 -> L = (P.d) +/- sqrt((P.d)^2+(Rh^2-RE^2)).
        # P.d < 0 (d down, P up), so the ONLY positive (near, up-the-ray) root is
        # L = (P.d) + sqrt(...). (The earlier max()-of-roots picked the spurious
        # far-side root ~938 km and overstated the displacement -> fixed.)
        Pd = P_det @ d
        L = Pd + np.sqrt(max(Pd**2 + (Rh**2 - RE**2), 0.0))
        Q = P_det - L * d
        latq, lonq = latlon_of(Q)
        zprim, azprim = local_dir(latq, lonq, d)      # primary (=neutrino dir) at Q
        rc_det = gb.cutoff_map(LAT, LON, DATE, [zen], [az], n_scan=40)[0, 0]
        rc_prod = gb.cutoff_map(latq, lonq, DATE, [zprim], [azprim], n_scan=40)[0, 0]
        res[name] = (rc_det, rc_prod)
        print(f"{name:>4} {az:5.0f} {L/1e3:7.1f} {latq-LAT:+6.2f} {lonq-LON:+6.2f} "
              f"{rc_det:7.1f} {rc_prod:8.1f}")

    # cutoff contrast (proxy for the sharpness of the E-W): larger Rc_E/Rc_W -> sharper
    cdet = res["E"][0] / max(res["W"][0], 1e-9)
    cprod = res["E"][1] / max(res["W"][1], 1e-9)
    print(f"\ncutoff contrast Rc(E)/Rc(W): detector {cdet:.2f} -> production {cprod:.2f}")
    print("If production < detector (East cutoff drops), candidate (a) softens W/E")
    print("toward Honda -> the overshoot is the production-point displacement.")
    print("DIAG_EW_CAUSE_DONE")


if __name__ == "__main__":
    main()
