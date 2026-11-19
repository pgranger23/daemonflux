"""Independent straight-ahead reference: dense/sparse march of the full kernel matrix.

Given a :class:`kernelset.KernelSet`, solve the one-dimensional cascade (every
secondary along its parent's direction: the g = 1 limit) along a straight slant path
from the top of the atmosphere to the ground,

    dN/dl = [rho(l) A_int + A_dec] N,     N[s, k] = f[s, k] w_k,

with ``A_int[(c,m),(p,k)] = inv_lambda[p,k] (n_mk - delta)`` and ``A_dec`` likewise
with ``inv_dec`` (``n_mk = Y[m,k] w_m / w_k``, children per interaction or decay),
over the **full** matrix: every channel, every child bin including those above the
parent bin (up-scatter), the short-lived species (K_S, Lambda decay within
centimetres) and the sinks (neutrinos accumulate their source; no attenuation).
Muon energy loss is not included: the set must have ``dedx = 0`` (or
``--zero-dedx``).  Angular laws are ignored (straight ahead).

Scheme: TR-BDF2 (Bank et al. 1985; gamma = 2 - sqrt 2, L-stable, second order) in
the path length ``l``, with the density evaluated at the three stage points (one
layer formula per step: steps never straddle a CORSIKA layer boundary).  Each
stage is one sparse LU solve (SuperLU, natural order, unknowns sorted by energy from
the top down, so the factor is block-triangular except the up-scatter entries).
Steps are uniform in ``u = X(l)/dX0 + l/dl0`` inside each layer; the grid is
refined by 2 and 4 (nested) and the result is the finest level, with the
convergence (largest relative change from level 2 to level 4, and the second-order
Richardson estimate of the remaining error) reported.

Geometry: spherical Earth of radius ``r_e``, ground at altitude 0, top
``h_top`` = 112.8 km; density ``grid.CorsikaAtmosphere`` (imported read-only; it is
the solver's atmosphere, CORSIKA BK_USStd with MCEq's layer rule, which is also
``tools/mc3d/atmosphere.py``'s parameterisation).  ``r_e`` = 6391 km is MCEq's and
the solver's default, 6371 km the Monte Carlo's (``constants.R_EARTH_KM``).
Does not import ``solver.py``.

Usage::

    source $N3/env.sh; cd $R/tools/cascade3d
    taskset -c 8-11 $PY gates/ref1d_dense.py validate          # against MCEq
    taskset -c 8-11 $PY gates/ref1d_dense.py run --set <ks.npz> --r-e 6371

Outputs under ``$N3/c1/ref1d/``.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from grid import CorsikaAtmosphere  # noqa: E402
from kernelset import KIND_INT, ROLE_SINK, KernelSet  # noqa: E402

N3 = os.environ.get("N3", "/home/pigrange/mc3d_work/runs/native3d")
OUT_DIR = os.path.join(N3, "c1", "ref1d")
GAMMA = 2.0 - np.sqrt(2.0)
NU = ("numu", "antinumu", "nue", "antinue")


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------
class Operators:
    """Sparse ``A_int`` [per g/cm^2] and ``A_dec`` [per cm] on a common pattern.

    Unknowns ``N[s, k] = f[s, k] w_k`` (number per bin), ordered by energy bin from
    the top down, species inside: ``index = (n_top - 1 - k) * nS + s``.
    """

    def __init__(self, ks, n_top=None):
        d = ks.d
        self.ks = ks
        nE = ks.nE if n_top is None else int(n_top)
        nS = ks.nS
        self.nE, self.nS, self.N = nE, nS, nE * nS
        w = d["e_widths"][:nE]
        rows, cols, vi, vd = [], [], [], []
        for j in range(d["ch_kind"].size):
            p, c, kind = d["ch_parent"][j], d["ch_child"][j], d["ch_kind"][j]
            Y = d["Y"][j][:nE, :nE]
            m, k = np.nonzero(Y)
            n = Y[m, k] * w[m] / w[k]
            rate = (d["inv_lambda"] if kind == KIND_INT else d["inv_dec"])[p][k]
            rows.append(self.index(c, m))
            cols.append(self.index(p, k))
            vi.append(n * rate if kind == KIND_INT else np.zeros(n.size))
            vd.append(n * rate if kind != KIND_INT else np.zeros(n.size))
        s = np.repeat(np.arange(nS), nE)
        k = np.tile(np.arange(nE), nS)
        diag = self.index(s, k)
        rows.append(diag)
        cols.append(diag)
        vi.append(-d["inv_lambda"][:, :nE].ravel())
        vd.append(-d["inv_dec"][:, :nE].ravel())
        r, c = np.concatenate(rows), np.concatenate(cols)
        vi, vd = np.concatenate(vi), np.concatenate(vd)
        # one csc pattern (all entries plus the diagonal), three aligned data vectors
        N = self.N
        key = c.astype(np.int64) * N + r
        dkey = np.arange(N, dtype=np.int64) * (N + 1)
        keys = np.unique(np.concatenate([key, dkey]))
        self.indices = (keys % N).astype(np.int32)
        cnt = np.bincount((keys // N).astype(np.int64), minlength=N)
        self.indptr = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int32)
        pos = np.searchsorted(keys, key)
        self.ai = np.zeros(keys.size)
        self.ad = np.zeros(keys.size)
        np.add.at(self.ai, pos, vi)
        np.add.at(self.ad, pos, vd)
        self.eye = np.zeros(keys.size)
        self.eye[np.searchsorted(keys, dkey)] = 1.0
        self.Ai = sp.csc_matrix((self.ai, self.indices, self.indptr), shape=(N, N))
        self.Ad = sp.csc_matrix((self.ad, self.indices, self.indptr), shape=(N, N))
        self.n_up = int(np.count_nonzero(r < c))

    def index(self, s, k):
        return (self.nE - 1 - np.asarray(k)) * self.nS + np.asarray(s)

    def state(self, f):
        """Pack ``f[s, k]`` (dPhi/dE) into ``N``."""
        w = self.ks.d["e_widths"][:self.nE]
        y = np.zeros(self.N)
        for s in range(self.nS):
            y[self.index(s, np.arange(self.nE))] = f[s, :self.nE] * w
        return y

    def unpack(self, y):
        """``f[s, k]`` (dPhi/dE at the bin centre) from ``N``."""
        w = self.ks.d["e_widths"][:self.nE]
        f = np.zeros((self.nS, self.nE))
        for s in range(self.nS):
            f[s] = y[self.index(s, np.arange(self.nE))] / w
        return f

    def system(self, c):
        """``I - c_int A_int - c_dec A_dec`` as csc."""
        ci, cd = c
        data = self.eye - ci * self.ai - cd * self.ad
        return sp.csc_matrix((data, self.indices, self.indptr), shape=(self.N, self.N))


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
class SlantPath:
    """Straight line from the top (l = 0) to the ground (l = length).

    Parameters
    ----------
    cos_z : float
        Cosine of the zenith angle at the ground.
    r_e, h_top : float
        Earth radius and top of the atmosphere [cm].
    """

    def __init__(self, cos_z, r_e=6391.0e5, h_top=112.8e5, atm=None):
        self.cos_z, self.r_e, self.h_top = float(cos_z), float(r_e), float(h_top)
        self.atm = atm or CorsikaAtmosphere(h_top=h_top, r_e=r_e)
        R, c = self.r_e, self.cos_z
        self.length = -R * c + np.sqrt(R * R * c * c + (R + h_top) ** 2 - R * R)
        # break points: crossings of the layer boundaries
        hb = [h for h in self.atm.boundaries if 0.0 < h < h_top]
        ls = sorted({0.0, self.length} | {self.l_of_h(h) for h in hb})
        self.breaks = np.array(ls)
        self._table()

    def h(self, l):
        """Altitude at path length ``l`` from the top [cm]."""
        s = self.length - np.asarray(l, float)          # distance from the ground
        R = self.r_e
        return np.sqrt(R * R + s * s + 2.0 * R * s * self.cos_z) - R

    def l_of_h(self, h):
        R, c = self.r_e, self.cos_z
        s = -R * c + np.sqrt(R * R * c * c + (R + h) ** 2 - R * R)
        return self.length - s

    def rho(self, l, layer):
        return self.atm.density(self.h(l), layer=layer)

    def piece_layer(self, i):
        """CORSIKA layer of piece i (between breaks i and i+1)."""
        lm = 0.5 * (self.breaks[i] + self.breaks[i + 1])
        return int(self.atm.layer(self.h(lm)))

    def _table(self, dh=2.0e3, dl=2.0e5, n_gl=8):
        """Fine table of the slant grammage X(l) per piece (Gauss-Legendre)."""
        xg, wg = np.polynomial.legendre.leggauss(n_gl)
        self.tab = []
        X0 = 0.0
        for i in range(len(self.breaks) - 1):
            a, b = self.breaks[i], self.breaks[i + 1]
            lay = self.piece_layer(i)
            dhab = abs(self.h(b) - self.h(a))
            n = int(max(np.ceil((b - a) / dl), np.ceil(dhab / dh), 4))
            ls = np.linspace(a, b, n + 1)
            Xs = [X0]
            for j in range(n):
                lm = 0.5 * (ls[j] + ls[j + 1]) + 0.5 * (ls[j + 1] - ls[j]) * xg
                Xs.append(Xs[-1] + 0.5 * (ls[j + 1] - ls[j])
                          * np.sum(wg * self.rho(lm, lay)))
            Xs = np.array(Xs)
            self.tab.append((ls, Xs, lay))
            X0 = Xs[-1]
        self.X_total = X0

    def nodes(self, dX0, dl0, refine):
        """Step nodes: per piece, uniform in u = X/dX0 + l/dl0, base count
        ``ceil(u_piece)``, each base step split into ``refine``.

        Returns a list of ``(l_nodes, layer)`` per piece.
        """
        out = []
        for ls, Xs, lay in self.tab:
            u = (Xs - Xs[0]) / dX0 + (ls - ls[0]) / dl0
            nb = max(int(np.ceil(u[-1])), 1)
            uu = np.linspace(0.0, u[-1], nb * refine + 1)
            out.append((np.interp(uu, u, ls), lay))
        return out


# ---------------------------------------------------------------------------
# march
# ---------------------------------------------------------------------------
def march(ops, path, y0, dX0, dl0, refine):
    """TR-BDF2 from the top to the ground; returns the state at the ground."""
    y = y0.copy()
    d = GAMMA / 2.0
    a1 = 1.0 / (GAMMA * (2.0 - GAMMA))
    a0 = (1.0 - GAMMA) ** 2 / (GAMMA * (2.0 - GAMMA))
    n_steps = 0
    for ln, lay in path.nodes(dX0, dl0, refine):
        for i in range(len(ln) - 1):
            l0, l1 = ln[i], ln[i + 1]
            h = l1 - l0
            lg = l0 + GAMMA * h
            r0, rg, r1 = path.rho(np.array([l0, lg, l1]), lay)
            rhs = y + d * h * (r0 * (ops.Ai @ y) + ops.Ad @ y)
            lu = spl.splu(ops.system((d * h * rg, d * h)), permc_spec="NATURAL",
                          diag_pivot_thresh=0.0)
            yg = lu.solve(rhs)
            lu = spl.splu(ops.system((d * h * r1, d * h)), permc_spec="NATURAL",
                          diag_pivot_thresh=0.0)
            y = lu.solve(a1 * yg - a0 * y)
            n_steps += 1
    return y, n_steps


def solve_direction(ks, cos_z, r_e=6391.0e5, h_top=112.8e5, dX0=2.0, dl0=2.0e5,
                    levels=(1, 2, 4), n_top=None, ops=None):
    """Ground fluxes ``f[s, k]`` at each refinement level, and step counts."""
    ops = ops or Operators(ks, n_top)
    path = SlantPath(cos_z, r_e, h_top)
    y0 = ops.state(ks.d["phi_top"])
    res, steps = [], []
    for r in levels:
        y, n = march(ops, path, y0, dX0, dl0, r)
        res.append(ops.unpack(y))
        steps.append(n)
    return res, steps, path


def convergence(fs, sel_species, e_grid, emin=0.1, emax=100.0, floor=1e-12):
    """Largest relative change between the two finest levels and the second-order
    Richardson error estimate, over the selected species and energies."""
    f2, f4 = fs[-2], fs[-1]
    ksel = (e_grid >= emin * 0.999) & (e_grid <= emax * 1.001)
    out = {}
    for s in sel_species:
        a, b = f2[s, ksel], f4[s, ksel]
        ok = np.abs(b) > floor * np.abs(b).max()
        rel = np.abs(b[ok] / a[ok] - 1.0)
        out[int(s)] = (float(rel.max()), float(rel.max() / 3.0))
    return out


# ---------------------------------------------------------------------------
# drivers
# ---------------------------------------------------------------------------
def _load_set(path, zero_dedx):
    ks = KernelSet.load(path)
    if np.any(ks.d["dedx"] != 0):
        if not zero_dedx:
            raise SystemExit(f"{path} has muon energy loss on; use a loss-off set "
                             "or --zero-dedx")
        ks.d["dedx"] = np.zeros_like(ks.d["dedx"])
    return ks


def run_set(path, r_e_km, cos_list, dX0, dl0, levels, zero_dedx, out_dir=OUT_DIR):
    ks = _load_set(path, zero_dedx)
    ops = Operators(ks)
    sp_idx = [ks.index(n) for n in NU]
    name = os.path.basename(path).replace(".npz", "")
    save = {"e_grid": ks.d["e_grid"], "cos_z": np.array(cos_list),
            "r_e_cm": np.array(r_e_km * 1e5), "h_top_cm": np.array(112.8e5),
            "levels": np.array(levels), "dX0": np.array(dX0), "dl0": np.array(dl0)}
    conv = {}
    t0 = time.time()
    for i, cz in enumerate(cos_list):
        fs, steps, path_ = solve_direction(ks, cz, r_e_km * 1e5, dX0=dX0, dl0=dl0,
                                           levels=levels, ops=ops)
        for n, s in zip(NU, sp_idx):
            save[f"{n}_{i}"] = fs[-1][s]
            save[f"{n}_{i}_coarse"] = fs[-2][s]
        save[f"steps_{i}"] = np.array(steps)
        save[f"X_total_{i}"] = np.array(path_.X_total)
        c = convergence(fs, sp_idx, ks.d["e_grid"], 0.1, 10.0)
        conv[str(cz)] = {NU[j]: c[s] for j, s in enumerate(sp_idx)}
        print(f"{name} R={r_e_km} cosZ={cz}: steps {steps}, X={path_.X_total:.2f}, "
              f"max rel change 2->4 (0.1-10 GeV) "
              f"{max(v[0] for v in conv[str(cz)].values()):.2e}, "
              f"{time.time() - t0:.0f} s", flush=True)
    save["convergence_json"] = np.array(json.dumps(conv))
    save["set_meta_json"] = np.array(json.dumps(ks.meta))
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"ref1d_{name}_R{int(r_e_km)}.npz")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; never overwritten")
    np.savez_compressed(out, **save)
    print("wrote", out)
    return out


def validate(dX0, dl0, levels, n_top, out_dir=OUT_DIR,
             step_file=os.path.join(N3, "c1", "review", "mceq_step_PHYS_lossoff.npz")):
    """``from_mceq("PHYS", loss=False, keep_upscatter=True)`` against MCEq's
    step-converged limit (first-order Richardson on n = 4 and 16)."""
    from kernelset import from_mceq

    ks = from_mceq("PHYS", loss=False, keep_upscatter=True)
    ops = Operators(ks, n_top)
    z = np.load(step_file)
    E = ks.d["e_grid"][:ops.nE]
    np.testing.assert_allclose(z["e_grid"][:ops.nE], E, rtol=1e-12)
    quant = {"numu": ["numu"], "antinumu": ["antinumu"], "nue": ["nue"],
             "antinue": ["antinue"],
             "mu": ["mu+_l", "mu+_r", "mu+", "mu-_l", "mu-_r", "mu-"]}
    dirs = {0: 1.0, 3: 0.25, 5: 0.05}
    save = {"e_grid": E, "levels": np.array(levels), "dX0": np.array(dX0),
            "dl0": np.array(dl0), "n_top": np.array(ops.nE)}
    lines = []
    ksel = (E >= 0.0999) & (E <= 100.1)
    t0 = time.time()
    summary = {}
    for di, cz in dirs.items():
        fs, steps, path = solve_direction(ks, cz, 6391.0e5, dX0=dX0, dl0=dl0,
                                          levels=levels, ops=ops)
        save[f"steps_{di}"] = np.array(steps)
        save[f"X_total_{di}"] = np.array(path.X_total)
        for q, names in quant.items():
            idx = [ks.index(n) for n in names]
            vals = [sum(f[i] for i in idx) for f in fs]
            for lv, v in zip(levels, vals):
                save[f"{q}_r{lv}_{di}"] = v
            lim = z[f"{q}_limit_{di}"][:ops.nE]
            n4, n16 = z[f"{q}_n4_{di}"][:ops.nE], z[f"{q}_n16_{di}"][:ops.nE]
            mine = vals[-1]
            ratio = mine[ksel] / lim[ksel]
            lim_err = np.abs(n16 - n4)[ksel] / 3.0 / np.abs(lim[ksel])
            conv = np.abs(vals[-1][ksel] / vals[-2][ksel] - 1.0)
            summary[(q, di)] = (ratio, lim_err, conv)
            j = int(np.argmax(np.abs(ratio - 1.0)))
            lines.append(f"| {q} | {cz} | {np.abs(ratio - 1).max():.2e} at "
                         f"{E[ksel][j]:.3g} GeV | {np.median(np.abs(ratio - 1)):.1e} "
                         f"| {lim_err.max():.1e} | {lim_err[j]:.1e} | "
                         f"{np.count_nonzero(np.abs(ratio - 1) > lim_err + 1e-4)} "
                         f"/ {ratio.size} | {conv.max():.1e} |")
        print(f"validate cosZ={cz}: steps {steps}, X={path.X_total:.3f}, "
              f"{time.time() - t0:.0f} s", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "validation_mceq_PHYS_lossoff")
    for ext in (".npz", ".md"):
        if os.path.exists(out + ext):
            raise SystemExit(f"{out + ext} exists; never overwritten")
    np.savez_compressed(out + ".npz", **save)
    hdr = ["# ref1d_dense against step-converged MCEq (PHYS, loss off)", "",
           "Set: `kernelset.from_mceq('PHYS', loss=False, keep_upscatter=True)`, "
           f"bins below index {ops.nE} (E < {ks.d['e_bins'][ops.nE]:.3g} GeV).",
           f"Steps: dX0 = {dX0} g/cm2, dl0 = {dl0 / 1e5} km, levels {levels}; "
           "R = 6391 km, top 112.8 km, ground 0.",
           f"MCEq limit: `{step_file}` (first-order Richardson on n = 4, 16).",
           "Cells: every MCEq bin from 0.1 to 100 GeV.", "",
           "| quantity | cos Z | max abs(ref/limit - 1) | median | max limit error "
           "abs(n16-n4)/3 | limit error at the max | cells outside limit error "
           "+ 1e-4 | max rel change level 2->4 |", "|---|---|---|---|---|---|---|---|"]
    with open(out + ".md", "w") as fh:
        fh.write("\n".join(hdr + lines) + "\n")
    print("wrote", out + ".md")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--n-top", type=int, default=86)
    r = sub.add_parser("run")
    r.add_argument("--set", required=True)
    r.add_argument("--r-e", type=float, default=6371.0, help="Earth radius [km]")
    r.add_argument("--cosz", default="1,0.5,0.15")
    r.add_argument("--zero-dedx", action="store_true")
    for p in (v, r):
        p.add_argument("--dX0", type=float, default=2.0)
        p.add_argument("--dl0", type=float, default=2.0, help="[km]")
        p.add_argument("--levels", default="1,2,4")
    a = ap.parse_args(argv)
    levels = tuple(int(x) for x in a.levels.split(","))
    if a.cmd == "validate":
        validate(a.dX0, a.dl0 * 1e5, levels, a.n_top)
    else:
        run_set(a.set, a.r_e, [float(x) for x in a.cosz.split(",")], a.dX0,
                a.dl0 * 1e5, levels, a.zero_dedx)


if __name__ == "__main__":
    main()
