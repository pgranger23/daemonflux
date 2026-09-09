"""Physical constants and particle data for the mc3d Monte Carlo.

Masses in GeV, lifetimes in s, c*tau in cm.  PDG 2024 values.
"""

from __future__ import annotations

import numpy as np

# --- fundamental ---------------------------------------------------------
C_CM_S = 2.99792458e10
C_M_S = 2.99792458e8
GEV_PER_KG = 5.6095886e26
AMU_G = 1.66053907e-24

# --- Earth / atmosphere --------------------------------------------------
# Honda (astro-ph/0404457) uses R_e = 6378.180 km (equatorial); tools/mceq3d
# uses the mean radius 6371 km everywhere.  We keep 6371 km for continuity with
# `offaxis_mc` / `geomag_backtrace` (see PHASE2_PLAN.md sec. 2 for why the
# 0.11% difference and geodetic flattening are irrelevant here).
R_EARTH_KM = 6371.0
R_EARTH_CM = R_EARTH_KM * 1e5
R_EARTH_M = R_EARTH_KM * 1e3

H_TOP_KM = 112.8          # top of the CORSIKA atmosphere (rho -> 0 above)
H_INJ_KM = 100.0          # Honda's injection sphere
R_INJ_KM = R_EARTH_KM + H_INJ_KM

A_AIR = 14.5              # MCEq's effective air mass number (config.A_target)
M_AIR_G = A_AIR * AMU_G   # mean air "nucleus" mass [g]

# --- particle data -------------------------------------------------------
M_P = 0.9382720813
M_N = 0.9395654133
M_PI = 0.13957039
M_PI0 = 0.1349768
M_K = 0.493677
M_K0 = 0.497611
M_MU = 0.1056583745
M_E = 0.000510998950

TAU_PI = 2.6033e-8
TAU_K = 1.2380e-8
TAU_KL = 5.116e-8
TAU_KS = 8.954e-11
TAU_MU = 2.1969811e-6
TAU_N = 878.4
TAU_LAM = 2.617e-10

CTAU_CM = {
    211: C_CM_S * TAU_PI,
    -211: C_CM_S * TAU_PI,
    321: C_CM_S * TAU_K,
    -321: C_CM_S * TAU_K,
    130: C_CM_S * TAU_KL,
    310: C_CM_S * TAU_KS,
    13: C_CM_S * TAU_MU,
    -13: C_CM_S * TAU_MU,
    2112: C_CM_S * TAU_N,
    -2112: C_CM_S * TAU_N,
    3122: C_CM_S * TAU_LAM,
    -3122: C_CM_S * TAU_LAM,
}

MASS = {
    2212: M_P, -2212: M_P,
    2112: M_N, -2112: M_N,
    211: M_PI, -211: M_PI,
    111: M_PI0,
    321: M_K, -321: M_K,
    130: M_K0, 310: M_K0,
    13: M_MU, -13: M_MU,
    11: M_E, -11: M_E,
    12: 0.0, -12: 0.0, 14: 0.0, -14: 0.0, 16: 0.0, -16: 0.0,
    22: 0.0,
    # hyperons / charm are dropped by the tracker (see PHASE2_PLAN sec. 4.4)
    3122: 1.115683, -3122: 1.115683,   # Lambda (tracked)
    3112: 1.197449, 3222: 1.18937,
    411: 1.86966, -411: 1.86966, 421: 1.86484, -421: 1.86484,
    431: 1.96835, -431: 1.96835, 4122: 2.28646,
}

CHARGE = {
    2212: +1, -2212: -1, 2112: 0, -2112: 0,
    211: +1, -211: -1, 111: 0,
    321: +1, -321: -1, 130: 0, 310: 0,
    13: -1, -13: +1,          # mu- is PDG 13 and has charge -1
    11: -1, -11: +1,
    12: 0, -12: 0, 14: 0, -14: 0, 16: 0, -16: 0, 22: 0,
    3122: 0, -3122: 0,
}

NEUTRINOS = (12, -12, 14, -14, 16, -16)
NU_NAME = {12: "nue", -12: "antinue", 14: "numu", -14: "antinumu",
           16: "nutau", -16: "antinutau"}

# Species tracked by the cascade (everything else is discarded, see plan).
# Lambda matters: MCEq's SIBYLL-2.3d tables give 0.20 Lambda + 0.03 Lambda-bar
# per p-air interaction at 89 GeV, and Lambda -> p pi- (63.9%) feeds the pion
# cascade.  Dropping it costs ~2% of the neutrino yield (measured in the
# milestone-1 closure).  Sigma+- have zero yield in these tables.
HADRONS = (2212, -2212, 2112, -2112, 211, -211, 321, -321, 130, 310,
           3122, -3122)
LEPTONS = (13, -13)
TRACKED = HADRONS + LEPTONS


def gamma_of(pdg: float, e_tot: float) -> float:
    m = MASS.get(int(pdg), 0.0)
    return e_tot / m if m > 0 else np.inf


def momentum(pdg: int, e_tot):
    m = MASS.get(int(pdg), 0.0)
    return np.sqrt(np.maximum(e_tot ** 2 - m ** 2, 0.0))
