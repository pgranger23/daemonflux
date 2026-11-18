"""Gate G0.1: MCEq blocks (see tools/cascade3d/GATES.md).

Evaluates, for PHYS and REF: the interaction and decay matrices rebuilt from the
stored per-(parent, child) blocks (``kernels_mceq.rebuild``) against the matrices of
a freshly built ``MCEqRun`` (``int_m``, ``dec_m``): largest absolute difference
divided by the largest absolute entry must be <= 1e-12.  Also reports (not gated) the
loss-off check and the database file MCEq actually loaded.

Usage::

    source $N3/env.sh; $PY tools/cascade3d/gates/g0_1.py [--rebuild]

Writes ``$N3/c0/gates/G0.1.md``.  ``--rebuild`` rebuilds the kernel files first.
"""

import hashlib
import importlib.util  # noqa: F401
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import kernels_mceq as K  # noqa: E402

TOL = 1e-12
GATE_TEXT = (
    "G0.1 MCEq blocks (gated). The interaction and decay matrices rebuilt from the\n"
    "extracted per-(parent, child) blocks equal MCEq's `int_m` and `dec_m`: largest\n"
    "absolute difference divided by the largest absolute entry <= 1e-12, for PHYS and\n"
    "for REF. The result file names the database file that MCEq actually loaded."
)


def rel_diff(a, b):
    """max|a-b| / max|a| for sparse matrices (a is MCEq's)."""
    diff = (a - b).tocoo()
    num = np.abs(diff.data).max() if diff.nnz else 0.0
    den = np.abs(a.tocoo().data).max()
    return float(num / den), float(num), float(den)


def main(argv):
    N3 = K.N3
    lines = []
    ok = True
    for pop in ("PHYS", "REF"):
        if "--rebuild" in argv or not os.path.exists(K.kernel_path(pop)):
            K.save(K.build(pop), pop)
        d = K.load(pop)
        run = K.make_mceq(pop)  # fresh, independent of the stored file
        im, dm = run.int_m, run.dec_m
        ri, rd = K.rebuild(d, loss=True)
        r_int, a_int, m_int = rel_diff(im, ri)
        r_dec, a_dec, m_dec = rel_diff(dm, rd)
        # same matrix structure?
        nnz = (im.nnz, ri.nnz, dm.nnz, rd.nnz)
        # extra: loss off against MCEq built with enable_energy_loss=False
        run0 = K.make_mceq(pop, energy_loss=False)
        ri0, _ = K.rebuild(d, loss=False)
        r_loss0, _, _ = rel_diff(run0.int_m, ri0)
        r_dec0, _, _ = rel_diff(run0.dec_m, rd)
        loaded = run._mceq_db.had_fname
        sha = hashlib.sha256()
        with open(loaded, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 24), b""):
                sha.update(blk)
        meta = d["meta"]
        p_ok = r_int <= TOL and r_dec <= TOL
        ok &= p_ok
        lines += [
            f"### {pop}: {'PASS' if p_ok else 'FAIL'}",
            "",
            f"* population: floor {K.POPULATIONS[pop]['primary_floor']} GeV, "
            f"e_min {K.POPULATIONS[pop]['e_min']} GeV; nE = {meta['nE']}, "
            f"nS = {meta['nS']}, state dimension {meta['nE'] * meta['nS']}",
            f"* blocks stored: {meta['n_int_blocks']} interaction, "
            f"{meta['n_dec_blocks']} decay, {meta['n_loss_species']} loss species",
            f"* int_m: max|diff| = {a_int:.3e}, max|entry| = {m_int:.3e}, "
            f"relative = **{r_int:.3e}** (limit {TOL:.0e})",
            f"* dec_m: max|diff| = {a_dec:.3e}, max|entry| = {m_dec:.3e}, "
            f"relative = **{r_dec:.3e}** (limit {TOL:.0e})",
            f"* nnz (MCEq int, rebuilt int, MCEq dec, rebuilt dec): {nnz}",
            f"* not gated: loss off, rebuild(loss=False) vs MCEq with "
            f"enable_energy_loss=False: int_m relative {r_loss0:.3e}, "
            f"dec_m relative {r_dec0:.3e}",
            f"* database file MCEq loaded: `{loaded}`",
            f"* sha256 of that file (computed now): `{sha.hexdigest()}`; "
            f"stored in kernel file: `{meta['db_sha256']}`; equal: "
            f"{sha.hexdigest() == meta['db_sha256']}",
            f"* MCEq {meta['mceq_version']} (version.py; dist "
            f"{meta['mceq_dist_version']}), "
            f"model {meta['interaction_model']}, primary {meta['primary_model_name']}",
            "",
        ]
        del run, run0
    head = [
        "# Gate G0.1 result",
        "",
        "```",
        GATE_TEXT,
        "```",
        "",
        f"**Overall: {'PASS' if ok else 'FAIL'}**",
        "",
    ]
    cmd = "source $N3/env.sh; $PY tools/cascade3d/gates/g0_1.py" + (
        " --rebuild" if "--rebuild" in argv else "")
    tail = ["Command:", "", "```", cmd, "```", ""]
    out = os.path.join(N3, "c0", "gates", "G0.1.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("\n".join(head + lines + tail))
    print("\n".join(head + lines))
    print("written", out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
