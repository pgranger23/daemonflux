"""MCEq's forward-Euler march, re-implemented on the extracted matrices (diagnosis).

Used by gate G1.1 to tell the solver's discretisation errors from MCEq's: the march
``phi += (int_m phi + dec_m (rho_inv phi)) dX`` is run on ``kernels_mceq.rebuild``'s
matrices restricted to the physical species (tracking species never feed them), with

* ``mode="mceq"``: MCEq's own steps and densities (``path_dX_i``, ``path_rho_inv_i``
  of the G1.1 reference file).  Must reproduce the reference fluxes;
* ``refine=n``: every step split into ``n`` equal sub-steps, ``1/rho`` interpolated
  log-linearly in X between MCEq's step starts (Euler-error study, Richardson);
* ``exact_end=True``: the last step is shortened so that the march ends at ``max_X``
  (MCEq overshoots the ground by up to one step);
* ``drop_upscatter=True``: entries with child bin above parent bin removed for
  transported (non-neutrino) children, as the solver does.

Usage::

    $PY tools/cascade3d/gates/mceq_euler.py PHYS lossoff 5 [refine]
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import kernels_mceq as km  # noqa: E402

N3 = os.environ.get("N3", "/home/pigrange/mc3d_work/runs/native3d")
REF_DIR = os.path.join(N3, "c1", "ref")
QUANTITIES = ("numu", "antinumu", "nue", "antinue", "mu")
MUONS = ("mu+_l", "mu+", "mu+_r", "mu-_l", "mu-", "mu-_r")


def ref_path(pop, loss):
    tag = "losson" if loss else "lossoff"
    return os.path.join(REF_DIR, f"mceq_g11_{pop}_{tag}.npz")


class EulerMarch:
    """Physical-species sub-system of MCEq's matrices for one population/loss."""

    def __init__(self, pop, loss, drop_upscatter=False, d=None):
        from scipy.sparse import csr_matrix

        d = km.load(pop) if d is None else d
        self.d = d
        nE = d["e_grid"].size
        self.nE = nE
        A, B = km.rebuild(d, loss=False)
        phys = np.flatnonzero(~d["is_tracking"])
        idx = (phys[:, None] * nE + np.arange(nE)[None, :]).ravel()
        A = A[idx][:, idx].tocoo()
        B = B[idx][:, idx].tocoo()
        if drop_upscatter:
            # yields only: the loss operator (added below) is not touched
            sink_rows = np.isin(np.abs(d["pdg"][phys]), (12, 14, 16))
            for M in (A, B):
                keep = ((M.row % nE) <= (M.col % nE)) | sink_rows[M.row // nE]
                M.data, M.row, M.col = M.data[keep], M.row[keep], M.col[keep]
        A = csr_matrix(A)
        if loss:
            Aw, _ = km.rebuild(d, loss=True)
            Ao, _ = km.rebuild(d, loss=False)
            A = A + (Aw - Ao)[idx][:, idx]
        self.A = csr_matrix(A)
        self.B = csr_matrix(B)
        self.phys = phys
        self.names = list(d["species"][phys])
        self.phi0 = d["phi0"][idx].copy()

    def run(self, dX, rho_inv, refine=1, exact_end=None):
        """March; returns the final state (physical species, flattened)."""
        dX = np.asarray(dX, float)
        ri = np.asarray(rho_inv, float)
        if exact_end is not None:
            cum = np.cumsum(dX)
            n = int(np.searchsorted(cum, exact_end) + 1)
            dX = dX[:n].copy()
            dX[-1] -= cum[n - 1] - exact_end
            ri = ri[:n]
        if refine > 1:
            xs = np.concatenate([[0.0], np.cumsum(dX)])[:-1]
            lr = np.log(ri)
            sub = []
            for j in range(refine):
                x = xs + dX * j / refine
                sub.append(np.exp(np.interp(x, xs, lr)))
            ri = np.stack(sub, axis=1).ravel()
            dX = np.repeat(dX / refine, refine)
        phi = self.phi0.copy()
        A, B = self.A, self.B
        for k in range(dX.size):
            phi += (A @ phi + B @ (ri[k] * phi)) * dX[k]
        return phi

    def quantities(self, phi):
        """G1.1 quantities at the ground: dict name -> (nE,)."""
        nE = self.nE
        out = {}
        for q in QUANTITIES:
            names = MUONS if q == "mu" else (q,)
            v = np.zeros(nE)
            for n in names:
                i = self.names.index(n)
                v += phi[i * nE:(i + 1) * nE]
            out[q] = v
        return out


def main(argv):
    pop, loss_s, i = argv[1], argv[2], int(argv[3])
    refine = int(argv[4]) if len(argv) > 4 else 1
    loss = loss_s == "losson"
    z = np.load(ref_path(pop, loss))
    em = EulerMarch(pop, loss)
    phi = em.run(z[f"path_dX_{i}"], z[f"path_rho_inv_{i}"], refine=refine)
    q = em.quantities(phi)
    E = em.d["e_grid"]
    sel = (E >= 0.1) & (E <= 100.0)
    for name in QUANTITIES:
        ref = z[name][i]
        r = q[name][sel] / ref[sel] - 1.0
        print(f"{name:9s} max|rel| {np.max(np.abs(r)):.3e}")


if __name__ == "__main__":
    main(sys.argv)
