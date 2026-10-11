"""MCEq database hole at E_kin ~ 56 GeV: confirm it, patch it, measure its effect.

Finding (see ``MCEQ_DB_HOLE.md``).  In the shipped MCEq database
``mceq_db_lext_dpm193_v140.h5`` every non-EPOS interaction model is extended
below ~80 GeV with DPMJET-III-19.3 ("lext").  The extension ramps from the low-
energy model (DPMJET) to the high-energy model over a few bins around 80 GeV.
In the projectile bin centred on E_kin = 56.2 GeV (grid index 37) every yield
column AND (since v1.4.0) the inelastic cross section are exactly 0.800 x the
DPMJET value, i.e. the DPMJET column enters the blend with weight 0.8 against an
empty high-energy column (our reading; the build script is not available).
Only that bin is affected (the 70.8 GeV bin is a normal blend: sum rule OK).

Sub-commands (all outputs go to ``WORK``, never to /tmp; the shipped database is
only ever opened read-only)::

    confirm [--db FILE]  task 1: show the hole (projectiles, secondaries, models);
                         ``--db`` checks another file (e.g. the v1.4.2 database)
    patch                task 2: write the patched database copies to WORK/data_<variant>/
    run VARIANT PRIMARY  task 3: MCEq fluxes (3 zenith angles) for one DB variant;
                         PRIMARY = H3a | GSF | GSF_ddm (GSF + DDM yields injected,
                         the daemonflux-like baseline)
    gs VARIANT PRIMARY   G_s(E, R_c) response of tools/mceq3d for one DB variant
    olddb FILE           task 4: same diagnostic for every model of another database
                         file (e.g. the 2020 v1.2.0 file, downloaded separately)
    tables               assemble the ratio tables (markdown to WORK/tables.md)
    all                  patch + run/gs for every variant (<= 6 processes) + tables

Variants::

    orig           the shipped database (read-only, untouched)
    noop           shipped content re-packed with no change (checks the writer)
    dpm            56.2 GeV SIBYLL23D yields AND cross section := DPMJETIII193
    interp         yields: x-aligned log-E interpolation between 44.7 and 89.1
                   GeV columns (weight 1/3 on the 89.1 GeV column); cross section:
                   log-E interpolation of the same two bins
    interp_yonly   as interp, yields only (cross section left as shipped)
    interp_csonly  as interp, cross section only (yields left as shipped)

Environment (see the memory recipe): cvmfs LCG_108 python3.12 with
``PYTHONPATH=<repo>/.venv/lib/python3.12/site-packages:<repo>/src``.
"""

import importlib.util  # noqa: F401  (must precede the MCEq import)
import argparse
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

WORK = "/home/pigrange/mc3d_work/runs/dbhole"
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "SIBYLL23D"
REF_MODEL = "DPMJETIII193"
VARIANTS = ["orig", "noop", "dpm", "interp", "interp_yonly", "interp_csonly"]
# name -> (crflux class, argument, inject the data-driven model DDM?)
PRIMARIES = {
    "H3a": ("HillasGaisser2012", "H3a", False),
    "GSF": ("GlobalSplineFitBeta", None, False),
    # daemonflux-like baseline: GSF primary + DDM yields (MCEq ``inject_ddm``)
    "GSF_ddm": ("GlobalSplineFitBeta", None, True),
}
COSZ = (1.0, 0.5, 0.1)
SPECIES = ("numu", "antinumu", "nue", "antinue", "mu+", "mu-")
E_TABLE = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
E_EXTRA = (300.0, 1000.0)
J_HOLE = 37  # projectile bin E_kin = 56.2 GeV (found by `confirm`, asserted below)
RC_GRID = (1.0, 2.0, 3.0, 5.0, 7.1, 11.0, 15.0, 20.0, 40.0)  # GV, for G_s
GS_CZ = (1.0, 0.1)
RC_SHOW = (3.0, 7.1, 11.0, 40.0)  # rows printed in the G_s tables


# --------------------------------------------------------------------------
# raw database access
# --------------------------------------------------------------------------
def orig_db():
    import MCEq.config as cfg

    return str(cfg.data_dir), cfg.mceq_db_fname


def orig_db_path():
    d, n = orig_db()
    return os.path.join(d, n)


class Table:
    """Interaction matrices of one model/medium, straight from the HDF5 file.

    ``mats[(parent, child)]`` is the dense ``M[i, j]`` (secondary bin i, projectile
    bin j) exactly as MCEq builds it (``HDF5Backend._gen_db_dictionary``), and
    ``raw[(parent, child)]`` the CSR triplet ``(data, indices, indptr)``.
    """

    def __init__(self, f, model, medium="air"):
        # medium=None: the old (v1.2) layout without a medium level
        base = f"hadronic_interactions/{medium}" if medium else "hadronic_interactions"
        g = f[f"{base}/{model}"]
        ip = f[f"{base}/{model}_indptrs"][:]
        dat = g[:, :]
        self.len_data = g.attrs["len_data"]
        self.tuples = np.asarray(g.attrs["tuple_idcs"])
        self.keys, self.raw, self._cache = [], {}, {}
        r = 0
        for k, t in enumerate(self.tuples):
            key = tuple(int(x) for x in t)
            n = int(self.len_data[k])
            self.raw[key] = (
                dat[0, r : r + n].copy(),
                dat[1, r : r + n].copy(),
                ip[k].copy(),
            )
            self.keys.append(key)
            r += n
        self.dim = ip.shape[1] - 1

    def dense(self, key):
        from scipy.sparse import csr_matrix

        if key not in self._cache:
            d, i, p = self.raw[key]
            self._cache[key] = csr_matrix(
                (d, i.astype(int), p), shape=(self.dim, self.dim)
            ).toarray()
        return self._cache[key]


def energy_fraction(tab, parent, eg):
    """sum_i E_i N_ij / E_j for each projectile bin j (N = M E_i/E_j)."""
    xr = eg[:, None] / eg[None, :]
    out = np.zeros(len(eg))
    for key in tab.keys:
        if key[0] == parent:
            out += (tab.dense(key) * xr**2).sum(0)
    return out


def _open(path=None):
    import h5py

    return h5py.File(path or orig_db_path(), "r")


# --------------------------------------------------------------------------
# task 1: confirm
# --------------------------------------------------------------------------
def cmd_confirm(args):
    path = getattr(args, "db", None)
    f = _open(path)
    eg = f["common"].attrs["e_grid"]
    print("database:", path or orig_db_path())
    print(
        "file attrs:",
        {k: (v.item() if hasattr(v, "item") else v) for k, v in f.attrs.items()},
    )
    print(
        f"energy grid: {len(eg)} bins, {eg[0]:.4g} .. {eg[-1]:.4g} GeV kinetic, "
        f"ratio {eg[1] / eg[0]:.5f} (= 10^0.1)"
    )
    S = Table(f, MODEL)
    D = Table(f, REF_MODEL)
    jj = [j for j in range(len(eg)) if 20 < eg[j] < 160]

    print(
        f"\n[1] tuples: {MODEL} {len(S.keys)}, {REF_MODEL} {len(D.keys)}; "
        f"{MODEL}-only: {len(set(S.keys) - set(D.keys))}, "
        f"{REF_MODEL}-only: {len(set(D.keys) - set(S.keys))}"
    )

    parents = sorted({k[0] for k in S.keys})
    print(
        f"\n[2] per projectile: energy sum rule sum_i E_i N_ij/E_j in "
        f"{MODEL}, and {MODEL}/{REF_MODEL}, vs E_kin of the projectile bin"
    )
    print("    ('=' bit-identical to DPMJET for every common secondary)")
    hole_bins = set()
    print("    parent " + " ".join(f"{eg[j]:>11.1f}" for j in jj))
    for p in parents:
        eS, eD = energy_fraction(S, p, eg), energy_fraction(D, p, eg)
        cells = []
        for j in jj:
            ident = all(
                np.array_equal(S.dense(k)[:, j], D.dense(k)[:, j])
                for k in S.keys
                if k[0] == p and k in D.raw
            )
            r = eS[j] / eD[j] if eD[j] > 0 else np.nan
            if eD[j] > 0 and r < 0.9:
                hole_bins.add(j)
            cells.append(f"{'=' if ident else ' '}{eS[j]:.3f}/{r:.3f}")
        if eD.max() > 0:
            print(f"    {p:6d} " + " ".join(f"{c:>11s}" for c in cells))
    print(
        f"\n    -> bins with a >10% energy-sum-rule deficit (any projectile), "
        f"all 20-160 GeV: {sorted(hole_bins)} = "
        f"{[round(float(eg[j]), 1) for j in sorted(hole_bins)]} GeV"
    )
    if path is None:
        assert hole_bins == {J_HOLE}, hole_bins
        print(
            "    -> the 70.8 GeV bin (index 38) satisfies the sum rule (regular blend)"
        )
    else:
        print(f"    -> (foreign database; shipped-v1.4.0 expectation is [{J_HOLE}])")

    print(
        f"\n[3] column-by-column ratio {MODEL}/{REF_MODEL} at E_kin = "
        f"{eg[J_HOLE]:.1f} GeV, per secondary (rows with "
        f"{REF_MODEL} > 0; min .. max over the rows)"
    )
    for p in (2212, 2112, -2212, 211, -211, 321, -321, 130):
        out = []
        for c in (2212, 2112, 211, -211, 111, 321, -321, 130, 13, -13, 3122):
            k = (p, 0, c, 0)
            if k not in S.raw or k not in D.raw:
                continue
            s, d = S.dense(k)[:, J_HOLE], D.dense(k)[:, J_HOLE]
            m = d > 0
            if m.any():
                r = s[m] / d[m]
                out.append(f"{c}:{r.min():.4f}..{r.max():.4f}")
        print(f"    p={p:5d}: " + "  ".join(out))
    allr = []
    for k in S.keys:
        if k in D.raw:
            s, d = S.dense(k)[:, J_HOLE], D.dense(k)[:, J_HOLE]
            m = d > 0
            allr.extend(s[m] / d[m])
    allr = np.array(allr)
    print(
        f"    all {len(S.keys)} {MODEL} tuples, all non-empty cells in bin {J_HOLE}: "
        f"ratio min {allr.min():.12f} max {allr.max():.12f} (n={len(allr)})"
    )
    nid = sum(
        not np.array_equal(S.dense(k)[:, :J_HOLE], D.dense(k)[:, :J_HOLE])
        for k in S.keys
        if k in D.raw
    )
    print(
        f"    tuples whose columns j < {J_HOLE} (E_kin <= {eg[J_HOLE - 1]:.1f} GeV) "
        f"differ from {REF_MODEL}: {nid}"
    )

    print(
        f"\n[4] inelastic cross sections cross_sections/air/{MODEL} / {REF_MODEL} "
        "(columns = parents attr of the dataset)"
    )
    cs_s = f[f"cross_sections/air/{MODEL}"][:]
    cs_d = f[f"cross_sections/air/{REF_MODEL}"][:]
    par = list(f[f"cross_sections/air/{MODEL}"].attrs["parents"])
    print("    parent  " + " ".join(f"{eg[j]:>8.1f}" for j in jj[:9]))
    for c, pid in enumerate(par):
        if cs_d[J_HOLE, c] > 0:
            print(
                f"    {pid:6d}  "
                + " ".join(f"{cs_s[j, c] / cs_d[j, c]:8.4f}" for j in jj[:9])
            )
    ip = par.index(2212)
    sl = slice(J_HOLE - 1, J_HOLE + 2)
    fmt = lambda a: "/".join(f"{x:.2f}" for x in a[sl, ip])  # noqa: E731
    print(
        f"    p-air at {eg[J_HOLE - 1]:.1f}/{eg[J_HOLE]:.1f}/{eg[J_HOLE + 1]:.1f} GeV: "
        f"{MODEL} {fmt(cs_s)} mb, {REF_MODEL} {fmt(cs_d)} mb"
    )

    print(
        "\n[5] same diagnostic for every interaction model/medium in the file "
        "(proton projectile, sum rule at E_kin = 44.7 / 56.2 / 70.8 GeV):"
    )
    for med in ("air", "hydrogen"):
        for m in sorted(
            x for x in f[f"hadronic_interactions/{med}"] if "indptrs" not in x
        ):
            t = Table(f, m, med)
            e = energy_fraction(t, 2212, eg)
            flag = (
                "HOLE"
                if e[J_HOLE] < 0.9 * 0.5 * (e[J_HOLE - 1] + e[J_HOLE + 1])
                else "ok"
            )
            print(
                f"    {med:8s} {m:20s} {e[J_HOLE - 1]:.3f} {e[J_HOLE]:.3f} "
                f"{e[J_HOLE + 1]:.3f}  {flag}"
            )

    j0_, j1_, jm_ = J_HOLE - 1, J_HOLE + 2, J_HOLE + 1
    print(
        f"\n[6] how bin {jm_} ({eg[jm_]:.1f} GeV) is built: least-squares fit "
        f"N_{jm_} = a N_{j0_} + b N_{j1_} (neighbours x-shifted onto bin {jm_}); "
        "a linear interpolation in bin index would give (a, b) = (1/3, 2/3)"
    )
    for k in (
        (2212, 0, 211, 0),
        (2212, 0, -211, 0),
        (2212, 0, 2212, 0),
        (2212, 0, 321, 0),
        (211, 0, 211, 0),
        (2112, 0, 2112, 0),
    ):
        xr_ = eg[:, None] / eg[None, :]
        n_ = S.dense(k) * xr_
        a_ = np.stack(
            [shifted(n_[:, J_HOLE - 1], 2), shifted(n_[:, J_HOLE + 2], -1)], 1
        )
        co, *_ = np.linalg.lstsq(a_, n_[:, J_HOLE + 1], rcond=None)
        rel = np.linalg.norm(a_ @ co - n_[:, J_HOLE + 1]) / np.linalg.norm(
            n_[:, J_HOLE + 1]
        )
        print(
            f"    {k[0]:5d} -> {k[2]:5d}: a = {co[0]:.3f}, b = {co[1]:.3f}, "
            f"residual {100 * rel:.1f}%"
        )
    f.close()


# --------------------------------------------------------------------------
# task 2: patched databases
# --------------------------------------------------------------------------
def shifted(col, k):
    """out[i] = col[i - k] (x-aligned shift on the log-uniform grid)."""
    out = np.zeros_like(col)
    if k > 0:
        out[k:] = col[:-k]
    elif k < 0:
        out[:k] = col[-k:]
    else:
        out[:] = col
    return out


def new_column_interp(M, eg, j, j0, j1):
    """Interpolated column j from columns j0 < j < j1 at fixed x = E_i/E_j.

    N_ij = M_ij E_i/E_j is the number of secondaries per interaction in bin i; on
    a log-uniform grid it depends on i - j only (Feynman scaling), so the neighbours
    are shifted by j - j0 / j - j1 bins before the log-E linear interpolation
    (t = ln(E_j/E_j0)/ln(E_j1/E_j0) = 1/3).
    """
    xr = eg / eg[j]
    t = np.log(eg[j] / eg[j0]) / np.log(eg[j1] / eg[j0])
    n0 = M[:, j0] * eg / eg[j0]
    n1 = M[:, j1] * eg / eg[j1]
    nn = (1 - t) * shifted(n0, j - j0) + t * shifted(n1, j - j1)
    return nn / xr


def make_variant(variant, outfile):
    import shutil

    import h5py
    from scipy.sparse import csr_matrix

    src = orig_db_path()
    shutil.copyfile(src, outfile)
    f = h5py.File(outfile, "r+")
    eg = f["common"].attrs["e_grid"]
    j, j0, j1 = J_HOLE, J_HOLE - 1, J_HOLE + 2  # 56.2, 44.7, 89.1 GeV
    S = Table(f, MODEL)
    D = Table(f, REF_MODEL)
    do_y = variant in ("dpm", "interp", "interp_yonly")
    do_cs = variant in ("dpm", "interp", "interp_csonly")
    nchg = 0
    new_raw = {}
    for key in S.keys:
        d, i, p = S.raw[key]
        if do_y:
            M = S.dense(key)
            if variant == "dpm":
                newc = D.dense(key)[:, j] if key in D.raw else M[:, j] / 0.8
            else:
                newc = new_column_interp(M, eg, j, j0, j1)
            if not np.array_equal(newc, M[:, j]):
                M2 = M.copy()
                M2[:, j] = newc
                m = csr_matrix(M2)
                m.sort_indices()
                d, i, p = (
                    m.data.astype(d.dtype),
                    m.indices.astype(i.dtype),
                    m.indptr.astype(p.dtype),
                )
                nchg += 1
        new_raw[key] = (d, i, p)

    # --- repack the SIBYLL23D yield dataset (data row, index row, indptrs) ---
    grp = f["hadronic_interactions/air"]
    old = grp[MODEL]
    old_ip = grp[MODEL + "_indptrs"]
    ld = np.array([len(new_raw[k][0]) for k in S.keys], dtype=S.len_data.dtype)
    data = np.zeros((2, int(ld.sum())), dtype=old.dtype)
    ipt = np.zeros(old_ip.shape, dtype=old_ip.dtype)
    r = 0
    for n, k in enumerate(S.keys):
        d, i, p = new_raw[k]
        data[0, r : r + len(d)] = d
        data[1, r : r + len(d)] = i
        ipt[n] = p
        r += len(d)
    attrs = dict(old.attrs)
    ip_attrs = dict(old_ip.attrs)
    kw = dict(
        chunks=old.chunks,
        compression=old.compression,
        compression_opts=old.compression_opts,
        shuffle=old.shuffle,
    )
    kw_ip = dict(
        chunks=old_ip.chunks,
        compression=old_ip.compression,
        compression_opts=old_ip.compression_opts,
        shuffle=old_ip.shuffle,
    )
    if kw["chunks"] is not None:  # a chunk may not exceed the new dataset shape
        kw["chunks"] = tuple(min(c, s) for c, s in zip(kw["chunks"], data.shape))
    del grp[MODEL]
    del grp[MODEL + "_indptrs"]
    ds = grp.create_dataset(MODEL, data=data, **kw)
    for k, v in attrs.items():
        ds.attrs[k] = v
    ds.attrs["len_data"] = ld
    if variant != "orig":
        ds.attrs["description"] = (
            str(attrs.get("description", ""))
            + f"\n[PATCHED diag_mceq_db_hole variant={variant}]"
        )
    dsi = grp.create_dataset(MODEL + "_indptrs", data=ipt, **kw_ip)
    for k, v in ip_attrs.items():
        dsi.attrs[k] = v

    # --- cross section ---
    if do_cs:
        cs = f[f"cross_sections/air/{MODEL}"]
        a = cs[:]
        if variant == "dpm":
            a[j] = f[f"cross_sections/air/{REF_MODEL}"][j]
        else:
            t = np.log(eg[j] / eg[j0]) / np.log(eg[j1] / eg[j0])
            a[j] = (1 - t) * a[j0] + t * a[j1]
        cs[:] = a
    f.close()
    return nchg


def verify_variant(variant, path):
    """Re-read the patched file; every column except the hole column must be
    bit-identical to the shipped one; report the hole column's sum rule."""
    f0, f1 = _open(), _open(path)
    eg = f0["common"].attrs["e_grid"]
    S0, S1 = Table(f0, MODEL), Table(f1, MODEL)
    assert S0.keys == S1.keys
    nbad = 0
    for k in S0.keys:
        a, b = S0.dense(k), S1.dense(k)
        c = np.ones(a.shape[1], bool)
        c[J_HOLE] = False
        if not np.array_equal(a[:, c], b[:, c]):
            nbad += 1
        if variant in ("noop", "interp_csonly") and not np.array_equal(a, b):
            nbad += 1
    ef0 = energy_fraction(S0, 2212, eg)[J_HOLE - 1 : J_HOLE + 3]
    ef1 = energy_fraction(S1, 2212, eg)[J_HOLE - 1 : J_HOLE + 3]
    cs0 = f0[f"cross_sections/air/{MODEL}"][J_HOLE - 1 : J_HOLE + 3, 8]
    cs1 = f1[f"cross_sections/air/{MODEL}"][J_HOLE - 1 : J_HOLE + 3, 8]
    # the other datasets must be unchanged
    for name in (
        f"cross_sections/air/{REF_MODEL}",
        f"hadronic_interactions/air/{REF_MODEL}",
    ):
        assert np.array_equal(f0[name][:], f1[name][:]), name
    print(
        f"  [{variant}] tuples with changed columns outside bin {J_HOLE}: {nbad}; "
        f"p sum rule (44.7,56.2,70.8,89.1) {np.round(ef0, 4)} -> {np.round(ef1, 4)}; "
        f"p-air cs {np.round(cs0, 2)} -> {np.round(cs1, 2)}"
    )
    f0.close()
    f1.close()
    assert nbad == 0


def variant_files(variant):
    """(data_dir, db_fname) to hand to MCEq.config for a variant."""
    if variant == "orig":
        return orig_db()
    d = os.path.join(WORK, f"data_{variant}")
    return d, f"mceq_db_lext_dpm193_v140_dbhole_{variant}.h5"


def cmd_patch(args):
    for v in VARIANTS:
        if v == "orig" or (args.only and v not in args.only):
            continue
        d, n = variant_files(v)
        os.makedirs(d, exist_ok=True)
        ddm = os.path.join(d, "DDM_1.0.npy")
        if not os.path.exists(ddm):
            os.symlink(os.path.join(orig_db()[0], "DDM_1.0.npy"), ddm)
        path = os.path.join(d, n)
        t0 = time.time()
        nchg = make_variant(v, path)
        print(
            f"[{v}] wrote {path} ({os.path.getsize(path) / 1e6:.0f} MB, "
            f"{nchg} tuples re-packed, {time.time() - t0:.0f} s)"
        )
        verify_variant(v, path)


# --------------------------------------------------------------------------
# task 3: MCEq runs
# --------------------------------------------------------------------------
def _setup_mceq(variant):
    import MCEq.config as cfg

    d, n = variant_files(variant)
    cfg.data_dir = pathlib.Path(d)
    cfg.mceq_db_fname = n
    cfg.e_min = 0.1
    cfg.debug_level = 0
    if getattr(cfg, "has_mkl", False):
        cfg.set_mkl_threads(1)
    return cfg


def _primary(name):
    import crflux.models as crf

    cls, arg, _ = PRIMARIES[name]
    return getattr(crf, cls), arg


def cmd_run(args):
    cfg = _setup_mceq(args.variant)
    from MCEq.core import MCEqRun

    t0 = time.time()
    mc = MCEqRun(
        interaction_model=MODEL, primary_model=_primary(args.primary), theta_deg=0.0
    )
    if PRIMARIES[args.primary][2]:
        from MCEq.ddm import DataDrivenModel

        mc.inject_ddm(DataDrivenModel())
    print(f"db: {cfg.data_dir}/{cfg.mceq_db_fname}; atmosphere {cfg.density_model}")
    out = {"e": np.array(mc.e_grid)}
    for cz in COSZ:
        mc.set_theta_deg(float(np.degrees(np.arccos(cz))))
        mc.solve()
        for s in SPECIES:
            out[f"{s}_cz{cz}"] = np.array(mc.get_solution("total_" + s, 0))
    fn = os.path.join(WORK, f"flux_{args.variant}_{args.primary}.npz")
    np.savez(fn, **out)
    print(f"wrote {fn} ({time.time() - t0:.0f} s)")


def cmd_gs(args):
    cfg = _setup_mceq(args.variant)
    sys.path.insert(0, HERE)
    from mceq3d_flux import MCEq3DFlux, SPECIES as GS_SPECIES

    t0 = time.time()
    eng = MCEq3DFlux(
        interaction_model=MODEL, primary=PRIMARIES[args.primary][:2], base_model="mceq"
    )
    out = {"e": np.array(eng.e), "rc": np.array(RC_GRID)}
    for cz in GS_CZ:
        G, _ = eng.geomag_response(np.array(RC_GRID), cz_ref=cz)
        for s in GS_SPECIES:
            out[f"{s}_cz{cz}"] = G[s]
    fn = os.path.join(WORK, f"gs_{args.variant}_{args.primary}.npz")
    np.savez(fn, **out)
    print(f"wrote {fn} ({time.time() - t0:.0f} s)")


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------
def _at(e, y, e0):
    return float(np.interp(np.log(e0), np.log(e), y))


def ratio_tables(variants, lines):
    for prim in PRIMARIES:
        ref = os.path.join(WORK, f"flux_orig_{prim}.npz")
        if not os.path.exists(ref):
            continue
        o = np.load(ref)
        e = o["e"]
        sel = (e >= 0.1 * 0.999) & (e <= 1000 * 1.001)
        for v in variants:
            fn = os.path.join(WORK, f"flux_{v}_{prim}.npz")
            if not os.path.exists(fn):
                continue
            p = np.load(fn)
            lines.append(
                f"\n### {prim}, variant `{v}`: flux(patched)/flux(original), SIBYLL23D\n"
            )
            for cz in COSZ:
                lines.append(f"\ncos Z = {cz}\n")
                hdr = (
                    "| species | "
                    + " | ".join(f"{x:g}" for x in E_TABLE + E_EXTRA)
                    + " | max dev [%] | at E [GeV] |"
                )
                lines.append(hdr)
                lines.append("|" + "---|" * (len(E_TABLE + E_EXTRA) + 3))
                for s in SPECIES:
                    k = f"{s}_cz{cz}"
                    r = np.where(o[k] > 0, p[k] / np.where(o[k] > 0, o[k], 1), 1.0)
                    vals = [_at(e, r, x) for x in E_TABLE + E_EXTRA]
                    ii = np.argmax(np.abs(r[sel] - 1))
                    lines.append(
                        f"| {s} | "
                        + " | ".join(f"{x:.4f}" for x in vals)
                        + f" | {100 * (r[sel][ii] - 1):+.2f} | {e[sel][ii]:.3g} |"
                    )


def gs_tables(variants, lines):
    for prim in PRIMARIES:
        ref = os.path.join(WORK, f"gs_orig_{prim}.npz")
        if not os.path.exists(ref):
            continue
        o = np.load(ref)
        e, rc = o["e"], o["rc"]
        for v in variants:
            fn = os.path.join(WORK, f"gs_{v}_{prim}.npz")
            if not os.path.exists(fn):
                continue
            p = np.load(fn)
            for cz in GS_CZ:
                lines.append(
                    f"\n### G_s, {prim}, `{v}`, cz_ref={cz}: G_s(patched)/G_s(orig), "
                    "total_numu (and total_nue in brackets)\n"
                )
                lines.append(
                    "| R_c [GV] | "
                    + " | ".join(f"{x:g}" for x in E_TABLE + E_EXTRA)
                    + " | max dev [%] (E<=300) |"
                )
                lines.append("|" + "---|" * (len(E_TABLE + E_EXTRA) + 2))
                for ir, r_c in enumerate(rc):
                    if r_c not in RC_SHOW:
                        continue
                    cells, mx = [], 0.0
                    for x in E_TABLE + E_EXTRA:
                        vals = []
                        for sp in ("total_numu", "total_nue"):
                            k = f"{sp}_cz{cz}"
                            g0, g1 = o[k][ir], p[k][ir]
                            r = np.where(g0 > 0, g1 / np.where(g0 > 0, g0, 1), 1.0)
                            vals.append(_at(e, r, x))
                        cells.append(f"{vals[0]:.4f} ({vals[1]:.4f})")
                    k = f"total_numu_cz{cz}"
                    g0, g1 = o[k][ir], p[k][ir]
                    r = np.where(g0 > 0, g1 / np.where(g0 > 0, g0, 1), 1.0)
                    m = (e >= 0.1) & (e <= 300)
                    mx = 100 * (r[m][np.argmax(np.abs(r[m] - 1))] - 1)
                    lines.append(
                        f"| {r_c:g} | " + " | ".join(cells) + f" | {mx:+.2f} |"
                    )
                # W/E-like double ratio at Kamioka-like cutoffs
                k = f"total_numu_cz{cz}"
                i_w, i_e = list(rc).index(7.1), list(rc).index(40.0)
                we0 = o[k][i_w] / np.where(o[k][i_e] > 0, o[k][i_e], 1)
                we1 = p[k][i_w] / np.where(p[k][i_e] > 0, p[k][i_e], 1)
                lines.append(
                    "\nW/E-type double ratio G_s(7.1 GV)/G_s(40 GV), patched/orig: "
                    + ", ".join(
                        f"{x:g} GeV: {_at(e, we1 / we0, x):.4f}"
                        for x in E_TABLE + E_EXTRA
                    )
                )


def cmd_tables(_args):
    lines = ["# generated by diag_mceq_db_hole.py tables\n"]
    ratio_tables([v for v in VARIANTS if v != "orig"], lines)
    gs_tables([v for v in VARIANTS if v in ("dpm", "interp")], lines)
    txt = "\n".join(lines)
    with open(os.path.join(WORK, "tables.md"), "w") as fh:
        fh.write(txt)
    print(txt)


# --------------------------------------------------------------------------
# task 4 helper: the older database
# --------------------------------------------------------------------------
def cmd_olddb(args):
    """Proton-projectile sum rule and p cross section of every model in an
    older (v1.2, no medium level) or newer database file."""
    import h5py

    f = h5py.File(args.path, "r")
    print(
        "file:",
        args.path,
        "attrs:",
        {k: (v.item() if hasattr(v, "item") else v) for k, v in f.attrs.items()},
    )
    eg = f["common"].attrs["e_grid"]
    jj = [j for j in range(len(eg)) if 20 < eg[j] < 160]
    old = "air" not in f["hadronic_interactions"]
    root = "hadronic_interactions" if old else "hadronic_interactions/air"
    models = sorted(x for x in f[root] if "indptrs" not in x)
    print("models:", models)
    print("E_kin [GeV]:           " + " ".join(f"{eg[j]:7.1f}" for j in jj))
    for m in models:
        t = Table(f, m, None if old else "air")
        e = energy_fraction(t, 2212, eg)
        print(f"  {m:18s} p sum rule: " + " ".join(f"{e[j]:7.3f}" for j in jj))
        csn = f"cross_sections/{m}" if old else f"cross_sections/air/{m}"
        if csn in f:
            cs = f[csn][:]
            par = list(f[csn].attrs["projectiles" if old else "parents"])
            if 2212 in par:
                c = par.index(2212)
                print(
                    f"  {'':18s} p-air cs: " + " ".join(f"{cs[j, c]:7.1f}" for j in jj)
                )
    f.close()


# --------------------------------------------------------------------------
def cmd_all(args):
    py = sys.executable
    cmd_patch(argparse.Namespace(only=None))
    jobs = []
    for v in VARIANTS:
        for p in PRIMARIES:
            jobs.append([py, os.path.abspath(__file__), "run", v, p])
    for v in ("orig", "dpm", "interp"):
        for p in PRIMARIES:
            if not PRIMARIES[p][2]:
                jobs.append([py, os.path.abspath(__file__), "gs", v, p])
    env = dict(
        os.environ, MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
    )
    running = []
    while jobs or running:
        while jobs and len(running) < args.nproc:
            j = jobs.pop(0)
            logf = open(os.path.join(WORK, "log_" + "_".join(j[2:]) + ".txt"), "w")
            running.append(
                (subprocess.Popen(j, env=env, stdout=logf, stderr=subprocess.STDOUT), j)
            )
        time.sleep(2)
        for pr, j in list(running):
            if pr.poll() is not None:
                print(
                    "done" if pr.returncode == 0 else f"FAILED({pr.returncode})",
                    " ".join(j[2:]),
                )
                running.remove((pr, j))
    cmd_tables(args)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sp = ap.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("confirm")
    q.add_argument(
        "--db",
        default=None,
        help="check another database file (default: the shipped one)",
    )
    q.set_defaults(fn=cmd_confirm)
    q = sp.add_parser("patch")
    q.add_argument("--only", nargs="*")
    q.set_defaults(fn=cmd_patch)
    for name, fn in (("run", cmd_run), ("gs", cmd_gs)):
        q = sp.add_parser(name)
        q.add_argument("variant", choices=VARIANTS)
        q.add_argument("primary", choices=list(PRIMARIES))
        q.set_defaults(fn=fn)
    q = sp.add_parser("olddb")
    q.add_argument("path")
    q.set_defaults(fn=cmd_olddb)
    sp.add_parser("tables").set_defaults(fn=cmd_tables)
    q = sp.add_parser("all")
    q.add_argument("--nproc", type=int, default=6)
    q.set_defaults(fn=cmd_all)
    args = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    args.fn(args)


if __name__ == "__main__":
    main()
