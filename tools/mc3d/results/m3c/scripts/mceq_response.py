"""Gate C3 (milestone 3c): MCEq's 1D neutrino response per primary-nucleon
energy cell, so that the engine's G_s can be re-derived under any convention
(rigidity of the cut, energy variable of the H3a flux, lowest primary energy)
by reweighting -- MCEq is linear in the primary flux.

For cos Z in (0.05, 0.95) and nucleon in (p, n): for every cell (single
kinetic-energy bins of MCEq's grid from 1 to 31.6 GeV, then lumps 31.6-100,
100-316, 316-1000, 1000-3162, > 3162 GeV), MCEq is solved with ``_phi0`` equal
to the engine's standard (H3a, crflux at E_tot = T + m) nucleon flux in that
cell only -- and, below MCEq's own minimal primary energy (E_tot = 3 GeV,
where the standard flux is zero), with the same H3a expression extended down
to T = 1 GeV.  Output: ``$M/work/mceq_resp_h3a.npz``.

Read-only use of tools/mceq3d (``MCEq3DFlux`` builds the identical MCEqRun).
Database: whatever ``tools/mceq3d/mceq3d_db`` resolves (``$MCEQ3D_DB``; its
default became the patched v1.4.2 file on 2026-11-04, while this script ran);
``$RESP_SUFFIX`` is appended to the output name.
"""
import os
import sys
import time

import numpy as np

os.chdir(os.environ["M"] + "/work")
from mceq3d_flux import MCEq3DFlux  # noqa: E402

t0 = time.time()
eng = MCEq3DFlux(interaction_model="SIBYLL23D",
                 primary=("HillasGaisser2012", "H3a"), base_model="mceq")
mq = eng.mceq
e = np.asarray(mq.e_grid)
eb = np.asarray(mq.e_bins)
phi_std = eng._phi0_std.copy()
psl, nsl = eng._p_sl, eng._n_sl
# the H3a nucleon flux at E_tot = T + m for every bin (as MCEq's
# set_primary_model, but without the minimal-energy floor)
m_avg = 0.5 * (mq.pman[(2212, 0)].mass + mq.pman[(2112, 0)].mass)
_f, p_ext, n_ext = mq.pmodel.p_and_n_flux(e + m_avg)
p_ext, n_ext = 1e-4 * np.asarray(p_ext), 1e-4 * np.asarray(n_ext)
SPECIES = ("total_nue", "total_antinue", "total_numu", "total_antinumu")

ilo = int(np.searchsorted(eb, 1.0 - 1e-9))          # first bin with lower edge 1
ihi = int(np.searchsorted(eb, 31.6228 - 1e-3))       # edge 31.6
cells = [[i] for i in range(ilo, ihi)]
lump_edges = (31.6228, 100.0, 316.228, 1000.0, 3162.28, 1e12)
for a, b in zip(lump_edges[:-1], lump_edges[1:]):
    idx = np.where((e > a) & (e < b))[0]
    cells.append(list(idx))
print("cells", [(round(e[c[0]], 3), round(e[c[-1]], 3)) for c in cells],
      flush=True)
out = {"db": np.array([str(getattr(eng, "_db", "unknown"))]), "e": e, "e_bins": eb, "phi_std": phi_std, "p_ext": p_ext,
       "n_ext": n_ext, "p_sl": np.array([psl.start, psl.stop]),
       "n_sl": np.array([nsl.start, nsl.stop]),
       "cells": np.array([[c[0], c[-1] + 1] for c in cells]),
       "f_free": eng._f_free, "az_bound": eng._az_bound}
czs = [float(x) for x in (sys.argv[1:] or ["0.95", "0.05"])]
for cz in czs:
    mq.set_theta_deg(float(np.degrees(np.arccos(cz))))
    mq._phi0[:] = phi_std
    mq.solve()
    for s in SPECIES:
        out[f"full|{cz}|{s}"] = mq.get_solution(s, 0).copy()
    for nuc, sl, ext in (("p", psl, p_ext), ("n", nsl, n_ext)):
        R = np.zeros((len(cells), 4, len(e)))
        for ic, c in enumerate(cells):
            mq._phi0[:] = 0.0
            seg = np.asarray(c)
            # the extended H3a flux (= the standard one wherever that is
            # non-zero, checked below)
            mq._phi0[sl.start + seg] = ext[seg]
            mq.solve()
            for k, s in enumerate(SPECIES):
                R[ic, k] = mq.get_solution(s, 0)
            print(f"cz {cz} {nuc} cell {ic} ({e[c[0]]:.3g}-{e[c[-1]]:.3g} GeV)"
                  f" {time.time() - t0:.0f} s", flush=True)
        out[f"resp|{cz}|{nuc}"] = R
    np.savez(os.environ["M"] + "/work/mceq_resp_h3a" + os.environ.get("RESP_SUFFIX", "")
             + ".npz", **out)
nz = phi_std[psl] > 0
print("standard vs extended p flux where non-zero: max rel diff",
      float(np.max(np.abs(phi_std[psl][nz] / p_ext[nz] - 1))), flush=True)
print("done", time.time() - t0, flush=True)
