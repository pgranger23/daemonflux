"""Export MCEq's own inclusive yield / cross-section / decay tables to an npz.

Rationale (PHASE2_PLAN.md sec. 4.2).  Milestone-1's closure gate has to separate
*geometry and transport* errors from *generator* differences.  Running the
cascade off MCEq's **own** hadronic yield matrices makes the first gate a pure
test of the geometry, the decay/interaction competition, the energy-loss
treatment and the scorer: if the MC and MCEq disagree there, the MC is wrong.
The chromo backend (`interactions.ChromoBackend`) is then gated against the same
MCEq run in the same collinear mode, which isolates the generator difference.

Sampling convention.  MCEq stores ``M[i, j] = dN/dx_i * dlnE``, so the *number*
of daughters produced in energy bin ``i`` per parent interaction/decay at bin
``j`` is ``N[i, j] = M[i, j] * E_i / E_j`` (verified: the pi -> mu nu decay
matrix sums to 1.000 in this convention).

Usage::

    python build_tables.py --model SIBYLL23D --emin 0.05 --out mceq_SIBYLL23D.npz
"""

from __future__ import annotations

import argparse

import numpy as np

# parents whose interactions we track
PARENTS = (2212, -2212, 2112, -2112, 211, -211, 321, -321, 130, 310,
           3122, -3122)
# daughters we keep from an interaction
CHILDREN = (2212, -2212, 2112, -2112, 211, -211, 321, -321, 130, 310,
            3122, -3122, 13, -13)
# decay daughters (for the optional 'mceq decay tables' collinear mode)
DEC_CHILDREN = (13, -13, 12, -12, 14, -14, 211, -211, 111, 321, -321,
                2212, -2212, 2112, -2112)

A_TARGET = 14.6568            # MCEq's average air mass number
M_TARGET_G = A_TARGET * 1.672621e-24


def build(model="SIBYLL23D", emin=0.05, primary="H3a", out="mceq_tables.npz"):
    import importlib.util  # noqa: F401

    import MCEq.config as cfg

    cfg.e_min = emin
    cfg.debug_level = 0
    import crflux.models as crf
    from MCEq.core import MCEqRun

    mc = MCEqRun(
        interaction_model=model,
        primary_model=(crf.HillasGaisser2012, primary),
        theta_deg=0.0,
    )
    eg, ew, eb = mc.e_grid, mc.e_widths, mc.e_bins
    I, D = mc._interactions, mc._decays
    xr = eg[:, None] / eg[None, :]        # E_i / E_j

    data = {"e_grid": eg, "e_widths": ew, "e_bins": eb,
            "model": np.array([model]), "emin": np.array([emin])}
    for p in PARENTS:
        data[f"cs_{p}"] = mc._int_cs.get_cs(p)
        for c in CHILDREN:
            try:
                M = I.get_matrix((p, 0), (c, 0))
            except Exception:
                continue
            N = M * xr
            if N.sum() > 1e-12:
                data[f"y_{p}_{c}"] = N.astype(np.float32)
    # Decay matrices.  NB: MCEq keeps *helicity-resolved* muon states
    # ((-13,-1) "pi_mu+_l", (-13,+1) "pi_mu+_r", ...), so ``(211,0) -> (-13,0)``
    # is EMPTY and a naive export silently loses every muon (and with it half
    # the nu_mu and essentially all the nu_e).  Sum over the helicity index for
    # muon daughters; the resulting table is the *unpolarised* muon yield, and
    # the matching MCEq reference for that rung must therefore be run with
    # ``config.muon_helicity_dependence = False``.
    for p in (211, -211, 321, -321, 130, 310, 13, -13, 3122, -3122):
        for c in DEC_CHILDREN:
            N = np.zeros_like(xr)
            hels = (-1, 0, 1) if abs(c) == 13 else (0,)
            for h in hels:
                try:
                    M = D.get_matrix((p, 0), (c, h))
                except Exception:
                    continue
                N = N + M * xr
            if N.sum() > 1e-12:
                data[f"d_{p}_{c}"] = N.astype(np.float32)
    np.savez_compressed(out, **data)
    print(f"wrote {out}: {len(data)} arrays")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="SIBYLL23D")
    ap.add_argument("--emin", type=float, default=0.05)
    ap.add_argument("--primary", default="H3a")
    ap.add_argument("--out", default="mceq_tables_SIBYLL23D.npz")
    a = ap.parse_args(argv)
    build(a.model, a.emin, a.primary, a.out)


if __name__ == "__main__":
    main()
