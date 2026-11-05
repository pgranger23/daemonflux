"""The common 1D no-cutoff reference ``B_s(E, |cos Z|)`` of the Kamioka 3D table (INTEGRATION.md 10).

    nodes I N     solve the engine 1D flux (``MCEq3DFlux.base``: MCEq curved atmosphere,
                  GSF primary, v142p, no cutoff) at the Gauss-Legendre cos Z nodes of the
                  MC's cos Z bins (3 per bin) -> $W/runs/integ/base1d/nodes_<I>of<N>.npz
    assemble      integrate over cos Z within each bin and over energy within each MC E bin
                  (same measure as the MC tallies) -> $W/runs/integ/out/base1d_kamioka.npz

Output ``B``: ``B[species, E bin (60, sky.E_BINS), cos Z bin (20, sky.CZ_BINS)]`` =
``int dE <Phi>_cosZ-bin`` in /(cm^2 s sr) (the per-bin integral of the intensity), the same for
both hemispheres (1D production is up/down symmetric). ``B_t1`` = ``B * 2 pi * dcosZ``
is directly comparable to the azimuth-summed MC tally ``t1`` (measure 1): checked against
``pilot2_total.npz`` in the assembly (flat t1/B_t1 in cos Z 0.35-1).
"""
import importlib.util  # noqa: F401
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
W = os.environ.get("MC3D_WORK", "/home/pigrange/mc3d_work")
RUN = os.path.join(W, "runs", "integ")
SPECIES = ("total_nue", "total_antinue", "total_numu", "total_antinumu")
NAMES = ("nue", "antinue", "numu", "antinumu")
E_BINS = np.logspace(-1, 2, 61)  # sky.E_BINS
CZ_BINS = np.linspace(-1.0, 1.0, 21)  # sky.CZ_BINS
GL_X = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)])  # 3-point Gauss-Legendre on [-1, 1]
GL_W = np.array([5.0, 8.0, 5.0]) / 18.0  # weights of the bin average


def nodes():
    """(|cos Z| of every node, bin index 0..9 of |cos Z|, node index) -- 10 x 3."""
    out = []
    for j in range(10):
        c = 0.05 + 0.1 * j
        for k in range(3):
            out.append((c + 0.05 * GL_X[k], j, k))
    return out


def cmd_nodes(i, n):
    from mceq3d_flux import MCEq3DFlux

    i, n = int(i), int(n)
    nd = nodes()
    order = sorted(range(len(nd)), key=lambda q: nd[q][0])  # horizontal (slow) first
    mine = order[i::n]
    eng = MCEq3DFlux(base_model="mceq", primary=("GlobalSplineFitBeta", None))
    cz = np.array([nd[q][0] for q in mine])
    base = eng.base(cz)
    d = {"idx": np.array(mine), "cz": cz, "e": np.array(eng.e), "tag": eng._tag,
         "db_label": eng._db.label, "db_sha256": eng._db.sha256,
         "atmosphere": str(eng.mceq.density_model.model if hasattr(eng.mceq.density_model, "model") else "CORSIKA"),
         "e_min": 0.1}
    for sp in SPECIES:
        d[sp] = base[sp] / 1e4  # engine reports /(m^2 s sr GeV); MC tallies are per cm^2
    os.makedirs(os.path.join(RUN, "base1d"), exist_ok=True)
    np.savez(os.path.join(RUN, "base1d", f"nodes_{i}of{n}.npz"), **d)
    print("done", i, n, len(mine))


def cmd_assemble():
    import glob

    files = sorted(glob.glob(os.path.join(RUN, "base1d", "nodes_*of*.npz")))
    nd = nodes()
    parts = [np.load(f) for f in files]
    e = parts[0]["e"]
    flux = {sp: np.full((len(nd), len(e)), np.nan) for sp in SPECIES}
    for p in parts:
        for sp in SPECIES:
            flux[sp][p["idx"]] = p[sp]
    assert all(np.all(np.isfinite(flux[sp])) for sp in SPECIES), "missing nodes"
    # per-bin integral of the intensity over each MC E bin (log-log interpolation of the
    # engine's 10/decade grid, 4-point Gauss-Legendre in ln E), cos Z-averaged with the GL weights
    gx, gw = np.polynomial.legendre.leggauss(6)
    B = np.zeros((4, 60, 20))
    le = np.log(e)
    for k, sp in enumerate(SPECIES):
        for q, (c, j, kk) in enumerate(nd):
            lf = np.log(np.maximum(flux[sp][q], 1e-300))
            for ib in range(60):
                a, b = np.log(E_BINS[ib]), np.log(E_BINS[ib + 1])
                x = 0.5 * (a + b) + 0.5 * (b - a) * gx
                # linear extrapolation in log-log below the first engine grid point
                f = np.interp(x, le, lf)
                lo = x < le[0]
                if lo.any():
                    s = (lf[1] - lf[0]) / (le[1] - le[0])
                    f[lo] = lf[0] + s * (x[lo] - le[0])
                integral = 0.5 * (b - a) * np.sum(gw * np.exp(f + x))  # int Phi dE = int Phi E dlnE
                val = GL_W[kk] * integral
                B[k, ib, 10 + j] += val  # down-going bin of |cos Z| = bin j
                B[k, ib, 9 - j] += val  # up-going mirror
    out = {
        "B": B,
        "B_t1": B * 2.0 * np.pi * 0.1,
        "species": np.array(NAMES),
        "e_bins": E_BINS,
        "cz_bins": CZ_BINS,
        "nodes_cz": np.array([n[0] for n in nd]),
        "engine_e": e,
        "engine_flux_per_cm2": np.stack([flux[sp] for sp in SPECIES]),
        "units": "B: int dE Phi in /(cm^2 s sr), cos Z-bin average (3-point Gauss), identical for both "
        "hemispheres; B_t1 = B * 2 pi * 0.1 is the azimuth-summed MC tally t1 (measure 1) scale",
        "database": str(parts[0]["db_label"]),
        "database_sha256": str(parts[0]["db_sha256"]),
        "primary": "GlobalSplineFitBeta (GSF)",
        "interaction_model": "SIBYLL23D",
        "atmosphere": "CORSIKA US standard (BK_USStd)",
        "engine_tag": str(parts[0]["tag"]),
        "reference": "MCEq3DFlux.base(|cos Z|): curved-atmosphere 1D flux, no cutoff, total (conventional + prompt)",
    }
    p = os.path.join(RUN, "out", "base1d_kamioka.npz")
    np.savez(p, **out)
    print("wrote", p)


if __name__ == "__main__":
    c, a = sys.argv[1], sys.argv[2:]
    {"nodes": cmd_nodes, "assemble": lambda: cmd_assemble()}[c](*a)
