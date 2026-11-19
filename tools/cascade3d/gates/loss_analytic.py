"""Continuous-loss schemes against an exact solution (G1.1 loss-on diagnosis).

Problem on MCEq's PHYS energy grid, along the G1.1 lines cos Z = 1 and 0.05
(CORSIKA BK_USStd):
a parent P with a power-law top intensity E^-gamma and an interaction rate LAM
(default 1e-6 cm2/g: source along the whole line; ``LOSS_LAM=0.01`` in the
environment: production concentrated high up, so that low-energy ground muons are
slowed-down muons, the regime of the failing G1.1 loss-on cells) produces one
muon-like particle in the same energy bin per interaction
(source per grammage s(E) = lam E^-gamma exp(-lam X)); the particle loses energy at
the rate beta(E) and, optionally, decays like a muon.  With psi = beta phi:

    psi(E, X) = int_0^X beta(E') s(E', x) exp(-int_x^X D(E'') ds) dx,
    R(E') = R(E) + X - x   (CSDA range R),

evaluated by Gauss-Legendre quadrature (exact up to 1e-6).  Compared:

* cascade3d (``solver.Solver``, same code path as G1.1, its default settings);
* MCEq's scheme: forward Euler on MCEq's own steps and densities with MCEq's loss
  operator ``(I + 0.1 Op)^10 - I``, ``Op = -diag(1/E) D diag(dEdX)`` (D =
  ``loss_diff_op`` of the kernel file), rebuilt here for the given beta and
  checked against MCEq's stored operator for MCEq's dedx;
* the same MCEq scheme with 4x finer Euler steps.

Cases: constant beta = 2 MeV cm2/g, MCEq's muon dedx, each without and with decay.
Writes ``$N3/c1/gates/LOSS_ANALYTIC.md``.

Usage::

    taskset -c 0-3 $PY tools/cascade3d/gates/loss_analytic.py
    LOSS_LAM=0.01 taskset -c 0-3 $PY tools/cascade3d/gates/loss_analytic.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import grid  # noqa: E402
import kernels_mceq as km  # noqa: E402
import kernelset as K  # noqa: E402
import mceq_euler as ME  # noqa: E402
import rays  # noqa: E402
import solver as S  # noqa: E402

N3 = os.environ.get("N3", "/home/pigrange/mc3d_work/runs/native3d")
GAMMA = 2.7
LAM = float(os.environ.get("LOSS_LAM", "1e-6"))
M_MU, CTAU_MU = 0.1056583755, 65865.4  # GeV, cm


def toy_set(E, eb, w, beta, decay):
    nE = E.size
    dec = M_MU / (CTAU_MU * np.sqrt(E * (E + 2 * M_MU))) if decay else np.zeros(nE)
    d = {
        "e_grid": E, "e_bins": eb, "e_widths": w,
        "species": np.array(["P", "mu", "nu"]), "pdg": np.array([2212, 13, 14]),
        "mass": np.array([0.938, M_MU, 0.0]), "charge": np.array([1, -1, 0]),
        "role": np.array([K.ROLE_HADRON, K.ROLE_LOSS, K.ROLE_SINK]),
        "active": np.ones((3, nE), bool),
        "inv_lambda": np.array([np.full(nE, LAM), np.zeros(nE), np.zeros(nE)]),
        "inv_dec": np.array([np.zeros(nE), dec, np.zeros(nE)]),
        "dedx": np.array([np.zeros(nE), beta, np.zeros(nE)]),
        "phi_top": np.array([E ** -GAMMA, np.zeros(nE), np.zeros(nE)]),
        "ch_kind": np.array([K.KIND_INT]), "ch_parent": np.array([0]),
        "ch_child": np.array([1]), "Y": np.eye(nE)[None],
    }
    return K.KernelSet(d)


def mceq_loss_operator(E, diff_op, dEdX):
    raw = -np.diag(1.0 / E) @ diff_op @ np.diag(dEdX)
    I = np.eye(E.size)
    return np.linalg.matrix_power(I + 0.1 * raw, 10) - I


def mceq_march(E, diff_op, beta, decay, dX, rho_inv, refine=1):
    nE = E.size
    Lop = mceq_loss_operator(E, diff_op, -beta)
    dec = M_MU / (CTAU_MU * np.sqrt(E * (E + 2 * M_MU))) if decay else np.zeros(nE)
    A = np.zeros((2 * nE, 2 * nE))
    A[:nE, :nE] = -LAM * np.eye(nE)
    A[nE:, :nE] = LAM * np.eye(nE)
    A[nE:, nE:] = Lop
    B = np.zeros_like(A)
    B[nE:, nE:] = -np.diag(dec)
    if refine > 1:
        xs = np.concatenate([[0.0], np.cumsum(dX)])[:-1]
        lr = np.log(rho_inv)
        ri = np.stack([np.exp(np.interp(xs + dX * j / refine, xs, lr))
                       for j in range(refine)], axis=1).ravel()
        dX = np.repeat(dX / refine, refine)
        rho_inv = ri
    phi = np.concatenate([E ** -GAMMA, np.zeros(nE)])
    for k in range(dX.size):
        phi = phi + (A @ phi + B @ (rho_inv[k] * phi)) * dX[k]
    return phi[nE:]


def exact(E, beta, decay, X_tot, rho_of_X):
    """Quadrature of the exact solution at depth X_tot along the vertical line."""
    lb = np.log(beta)
    le = np.log(E)
    # fine range table
    lef = np.linspace(le[0], le[-1] + 3.0, 20001)
    bf = np.exp(np.interp(lef, le, lb, right=lb[-1]))
    Ef = np.exp(lef)
    Rf = np.concatenate([[0.0], np.cumsum(0.5 * (Ef[1:] / bf[1:] + Ef[:-1] / bf[:-1])
                                          * np.diff(lef))])
    xg, wg = np.polynomial.legendre.leggauss(64)
    out = np.zeros(E.size)
    edges = np.linspace(0.0, X_tot, 401)
    for i, e in enumerate(E):
        R0 = np.interp(np.log(e), lef, Rf)
        tot = 0.0
        for a, b in zip(edges[:-1], edges[1:]):
            x = a + (b - a) * 0.5 * (xg + 1)
            Rx = R0 + X_tot - x
            ok = Rx <= Rf[-1]
            lex = np.interp(Rx, Rf, lef)
            Ex = np.exp(lex)
            bx = np.exp(np.interp(lex, le, lb, right=lb[-1]))
            s = LAM * Ex ** -GAMMA * np.exp(-LAM * x)
            att = np.ones_like(x)
            if decay:
                # int_x^X D(E(x')) ds, ds = dX / rho  (all x of the piece at once)
                xp = x[:, None] + (X_tot - x)[:, None] * 0.5 * (xg[None] + 1)
                Ep = np.exp(np.interp(R0 + X_tot - xp, Rf, lef))
                D = M_MU / (CTAU_MU * np.sqrt(Ep * (Ep + 2 * M_MU)))
                att = np.exp(-np.sum(D / rho_of_X(xp) * wg, axis=1) * 0.5 * (X_tot - x))
            tot += np.sum(np.where(ok, bx * s * att, 0.0) * wg) * 0.5 * (b - a)
        out[i] = tot / beta[i]
    return out


def main():
    d = km.load("PHYS")
    E, eb, w = d["e_grid"], d["e_bins"], d["e_widths"]
    diff_op = d["loss_diff_op"]
    j = list(d["loss_species"]).index(km.species_index(d, "mu-"))
    beta_mceq = -d["dedx"][j]
    chk = mceq_loss_operator(E, diff_op, d["dedx"][j])
    ref_op = d["loss_blocks"][j]
    op_err = np.max(np.abs(chk - ref_op)) / np.max(np.abs(ref_op))
    z = np.load(ME.ref_path("PHYS", True))
    rows_all, summary_all = [], []
    for iline in (0, 5):
        rr, sm = run_line(z, iline, E, eb, w, diff_op, beta_mceq)
        rows_all += rr
        summary_all += sm
    finish(op_err, rows_all, summary_all)


def run_line(z, iline, E, eb, w, diff_op, beta_mceq):
    cz = float(z["cos_z"][iline])
    dX, ri = z[f"path_dX_{iline}"], z[f"path_rho_inv_{iline}"]
    atm = grid.CorsikaAtmosphere()
    lev = grid.make_levels(atm)
    fam = grid.LineFamily(np.array([cz]), np.zeros(0))
    r = rays.Rays(atm, lev, fam)
    X_tot = r.node_Xc[r.ground_node[0]]
    bline = r.line_b[0]

    hgrid = np.linspace(0, 112.8e5, 2001)
    Xg = rays.slant_depth(atm, bline, hgrid)
    lXg = np.log(np.maximum(Xg, 1e-300))

    def rho_of_X(x):
        h = np.interp(np.log(np.maximum(x, 1e-300)), lXg[::-1], hgrid[::-1])
        return atm.density(h)

    sel = (E >= 0.03) & (E <= 100)
    rows = []
    summary = []
    for bname, beta in (("const 2 MeV cm2/g", np.full(E.size, 2.0e-3)),
                        ("MCEq mu dedx", beta_mceq)):
        for decay in (False, True):
            ex = exact(E, beta, decay, X_tot, rho_of_X)
            sv = S.Solver(r, toy_set(E, eb, w, beta, decay))
            mine = np.asarray(sv.solve()["F_ground"])[:, 1, 0]
            m1 = mceq_march(E, diff_op, beta, decay, dX, ri)
            m4 = mceq_march(E, diff_op, beta, decay, dX, ri, refine=4)
            rows += [f"## cos Z {cz}: beta {bname}, decay {'on' if decay else 'off'}",
                     "",
                     "| E [GeV] | exact | cascade3d / exact - 1 | MCEq scheme | "
                     "MCEq scheme, 4x steps |", "|---|---|---|---|---|"]
            for k in np.flatnonzero(sel)[::3]:
                rows.append(f"| {E[k]:.3g} | {ex[k]:.4e} | "
                            f"{mine[k] / ex[k] - 1:+.4f} | "
                            f"{m1[k] / ex[k] - 1:+.4f} | {m4[k] / ex[k] - 1:+.4f} |")
            q = sel & (E >= 0.1)
            summary.append((cz, bname, decay, np.max(np.abs(mine[q] / ex[q] - 1)),
                            np.max(np.abs(m1[q] / ex[q] - 1)),
                            np.max(np.abs(m4[q] / ex[q] - 1))))
            rows.append("")
    return rows, summary


def finish(op_err, rows_in, summary):
    rows = ["# Continuous energy loss: cascade3d and MCEq's scheme against an exact "
            "solution", "", __doc__.split("Usage")[0].strip(), "",
            f"MCEq loss operator rebuilt from `loss_diff_op` and MCEq's mu- dedx "
            f"equals the stored `loss_blocks` to {op_err:.1e} (relative to the "
            f"largest entry).",
            ""] + rows_in
    rows += ["## Summary: largest |ratio - 1| for 0.1 to 100 GeV", "",
             "| cos Z | beta | decay | cascade3d | MCEq scheme | MCEq 4x steps |",
             "|---|---|---|---|---|---|"]
    for cz, b, dcy, a1, a2, a3 in summary:
        rows.append(f"| {cz} | {b} | {'on' if dcy else 'off'} | {a1:.4f} | {a2:.4f} | "
                    f"{a3:.4f} |")
    env = "" if LAM == 1e-6 else f"LOSS_LAM={LAM:g} "
    rows += ["", f"Command: `{env}taskset -c 0-3 $PY "
             "tools/cascade3d/gates/loss_analytic.py`", ""]
    tag = "" if LAM == 1e-6 else f"_lam{LAM:g}"
    path = os.path.join(N3, "c1", "gates", f"LOSS_ANALYTIC{tag}.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(rows))
    print("\n".join(rows[-12:]))


if __name__ == "__main__":
    main()
