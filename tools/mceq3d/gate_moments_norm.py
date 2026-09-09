"""Normalisation / shape gate for a moments file (``m_*.npz``).

``KERNEL_GENERATION.md`` section 4 runs the consistency gate on a separate,
non-``--moments`` kernel build. But a moments file already stores the
``x_L``-marginal ``dN/dx_L`` (``dndx``), so the gate can be applied to the
delivered artefact itself -- which is strictly better evidence.

The reference is MCEq's own stored ``hadr_yields`` for the same
(projectile, secondary, model) channel, divided by MCEq's log-energy bin width
``Delta(lnE)`` -- the storage convention that used to show up as an unexplained
~4.4x offset (see ``REVIEW.md``, "Resolution status"). With that divided out the
norm factor is O(1).

Only the SIBYLL rows (``E_proj >= 80 GeV``) are testable: MCEq ships no table for
UrQMD-3.4.

Usage::

    python gate_moments_norm.py m_spliced_v2.npz --sec piplus
"""

from __future__ import annotations

import argparse

import numpy as np

from kernel_regeneration import KernelGrid, compare_to_reference, load_mceq_reference


def gate(path, secondary="piplus", model="SIBYLL23D", emin=80.0, xl_min=5e-3):
    d = dict(np.load(path))
    assert "is_moments" in d, f"{path} is not a moments file"
    e = d["proj_energies"]
    sel = e >= emin
    grid = KernelGrid(
        xl_edges=d["xl_edges"], pt_edges=np.linspace(0, 3, 2), proj_energies=e[sel]
    )
    ref = load_mceq_reference(grid, model, secondary=secondary)
    if ref is None:
        return None
    marg = d["dndx"][sel]
    # restrict to the bulk-yield region: below x_L ~ 5e-3 SIBYLL is at its
    # kinematic turn-on while MCEq extrapolates its matrix (documented in
    # KERNEL_PRODUCTION_REPORT.md section 5).
    keep = grid.xl_centers >= xl_min
    stats = compare_to_reference(marg[:, keep], ref[:, keep], rtol=0.05)
    stats["n_energies"] = int(sel.sum())
    stats["n_xl"] = int(keep.sum())
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--sec", default="piplus")
    ap.add_argument("--model", default="SIBYLL23D")
    ap.add_argument("--xl-min", type=float, default=5e-3)
    args = ap.parse_args(argv)
    for f in args.files:
        s = gate(f, args.sec, args.model, xl_min=args.xl_min)
        if s is None:
            print(f"{f}: No reference available (MCEq missing or no table)")
            continue
        print(
            f"{f}  [{args.sec}, {args.model}, x_L >= {args.xl_min:g}, "
            f"{s['n_energies']} E x {s['n_xl']} x_L]\n"
            f"  norm factor (MCEq convention) = {s['norm_factor']:.3f}\n"
            f"  shape agreement within 5%     = {s['frac_within_rtol']:.3f}\n"
            f"  raw agreement within 5%       = {s['raw_frac_within_rtol']:.3f}"
        )


if __name__ == "__main__":
    main()
