"""MCEq's 1D kernels, extracted exactly (work package 0.1 of ``cascade3d``).

The 3D solver must reproduce MCEq in its straight-ahead limit, so its 1D kernels are
MCEq's own matrices.  This module builds an ``MCEqRun`` for one of the two populations
of ``tools/mceq3d/INTEGRATION.md``, cuts MCEq's sparse ``int_m`` and ``dec_m`` into
dense per-(parent, child) energy blocks, stores them with everything else MCEq's
update uses in one ``.npz`` file (no pickles), and reassembles the full matrices
(:func:`rebuild`) so that gate G0.1 can compare them with MCEq.

Populations (the two MCEq configurations of INTEGRATION.md)
-----------------------------------------------------------
``PHYS``  primary floor 1.9 GeV, ``e_min`` 0.03 GeV.
``REF``   primary floor 3.0 GeV (MCEq's own default), ``e_min`` 0.1 GeV.

Both use the *same* database, the v1.4.2 file with the 56.2 GeV hole patched
(``mceq_db_lext_dpm193_v142_dbhole_interp.h5``, chosen by ``mceq3d_db.make_run``).
The "floor" is ``MCEq.config.minimal_primary_energy`` [GeV, *total* energy per
nucleon]: ``MCEqRun.set_primary_model`` zeroes the nucleon initial condition in all
grid bins whose total nucleon energy ``e_grid + (m_p + m_n)/2`` lies below the bin
nearest to the floor.  It changes only the initial vector ``phi0``, never the
matrices.  ``e_min`` (``MCEq.config.e_min``) cuts the energy grid, hence the matrices.

Energy grid and state vector
----------------------------
Energies are **kinetic** energies.  ``e_grid[i]`` is the geometric centre
``sqrt(e_bins[i] e_bins[i+1])`` of bin ``i`` (10 bins per decade), ``e_widths[i] =
e_bins[i+1] - e_bins[i]`` [GeV].  The state vector is ``phi`` of length ``nS * nE``;
species ``s`` (MCEq index ``s``, name ``species[s]``) occupies
``phi[s*nE:(s+1)*nE]``.  ``phi[s*nE+i]`` is the differential intensity
``dPhi/dE_kin`` at the centre of bin ``i`` [GeV^-1 cm^-2 s^-1 sr^-1] (not a number
per bin, not E^3-weighted).  Number per bin is ``phi * e_widths``.  MCEq returns
``get_solution(name, mag=0)`` in these units (sum over helicities for muons).

The equation (MCEq's forward Euler update, ``solvers.solv_numpy``)
-----------------------------------------------------------------
With ``X`` the slant depth [g/cm^2], ``rho_inv = 1/rho(X)`` [cm^3/g] and ``dX`` the
step::

    phi += (int_m . phi + dec_m . (rho_inv * phi)) * dX

starting from ``phi0`` at ``X = X_start = 0`` and ending at the observation level
(``max_X``).  The step sizes ``dX`` come from MCEq's own integration path
(``dX <= 1 g/cm^2``, stability bound ``0.95 / max(lambda_int^-1, rho^-1 inv_dec_len)``).

Block convention (derived from ``MCEq/core.py``, ``MatrixBuilder.construct_matrices``)
--------------------------------------------------------------------------------------
Blocks are stored for MCEq index pairs ``(parent p, child c)`` as arrays of shape
``(nE_child, nE_parent) = (nE, nE)``: ``int_blocks[j][m, k]`` is the entry of
``int_m`` at row ``c*nE + m`` and column ``p*nE + k`` with ``p = int_parent[j]``,
``c = int_child[j]``; the same for ``dec_blocks``.  Only blocks with a non-zero entry
are kept.

``int_m`` block ``(p, c)``, column ``k`` is ``Lam_p[k] * (Y_pc[m, k] - delta_pc *
delta_mk)`` with ``Lam_p = inv_lambda_int[p]`` [cm^2/g] = ``sigma_inel / (<A_air> m_p)``
(``<A_air>`` = ``meta["A_target"]``, ``m_p`` = 1.672621e-24 g) and ``Y_pc`` MCEq's
yield matrix.  ``Y`` is the number of secondaries ``c`` produced into child bin
``m`` per interaction of a parent in bin ``k``, **per unit child energy and per
unit parent bin width**: ``Y[m, k] =
n_mk * w_k / w_m`` with ``n_mk`` the number of children in bin ``m`` per interacting
parent in bin ``k`` and ``w`` = ``e_widths``.  Hence the multiplicity of a parent at
bin ``k`` is ``sum_m Y[m, k] w_m / w_k`` (checked numerically: p(100 TeV) -> pi+ 12.6).
So ``Lam_p[k]`` (1/lambda_int) **is inside** the block, the density does **not**
enter (``int_m`` is per g/cm^2).  The parent's disappearance, ``-Lam_p[k]``, is on
the diagonal of the ``(p, p)`` block, present only if the parent can interact
(``can_interact``).  Elements of ``int_m`` therefore mix ``dN/dE`` per g/cm^2.

``dec_m`` block ``(p, c)``: column ``k`` is ``Dec_p[k] * (D_pc[m, k] - delta_pc *
delta_mk)`` with ``Dec_p = inv_dec_len[p]`` = ``m / (c tau (E_kin + m))`` [1/cm]
(zero below ``mix_idx``, see below) and ``D`` the decay distribution, normalised like
``Y`` (pi+ -> mu+ summed over both helicities: 1.000).  ``Dec_p`` **is inside** the
block, but ``1/rho`` is **outside**: the update applies ``dec_m`` to ``rho_inv * phi``
(1/cm times cm^3/g times g/cm^2 = decay probability).  Stable species have no ``(p, p)``
diagonal.  Only ``decay_dists`` of non-tracking, non-stable particles are filled;
chains through resonances (particles in the "resonance regime", below) are already
folded into the blocks (``MatrixBuilder._follow_chains``).

Same-bin (diagonal) entries.  ``m == k`` entries exist in the ``(p, c)`` blocks of
both matrices.  In ``int_m`` the ``(p, p)`` diagonal is ``-Lam_p[k]`` plus the same-bin
regeneration of the leading particle (measured p -> p: diagonal / ``Lam_p`` =
-0.989 at 10 GeV, -0.954 at 100 GeV, -0.917 at 1 TeV, i.e. +1% to +8% regenerated
in the same bin) and for ``(N, N')`` charge exchange.  Cross-species
same-bin entries (pi -> mu, p -> n, ...) are ordinary non-zero ``m == k`` elements.
MCEq's Euler march never solves a system: it is explicit, so these terms enter with
the previous step's ``phi``.

Blocks are essentially upper triangular (``m <= k``, child not above parent): the
largest entry with ``m > k`` is 2e-4 against a largest entry of 5.8e9 (REF), and
1e-6 in the decay blocks.  The entries span 1e-26 to 6e9 because of the factor
``w_k / w_m`` (bin widths run from 0.02 GeV to 1e10 GeV).

Continuous energy loss.  In ``MatrixBuilder`` the loss operator is **added to the
diagonal block ``(p, p)`` of ``int_m``** for every species with ``|pdg| == 13`` (and
``|pdg| == 11`` if ``enable_em_ion``; ``generic_losses_all_charged`` is off) when
``config.enable_energy_loss`` is on.  This includes helicity muons and the tracking
muons.  The operator is ``(I + h Op)^(1/h) - I`` with ``h = loss_step_for_average =
0.1`` (``average_loss_operator``) and ``Op = -diag(1/E) . D . diag(dEdX)``, where ``D``
is a 6th-order finite-difference d/d(ln E) (``loss_diff_op``, row ``i`` divided by
``ln(e_bins[i+1]/e_bins[i])``) and ``dEdX`` [GeV cm^2/g, **negative**, ionisation plus
radiative, PDG table, function of kinetic energy at the bin centre].  It acts on
``dN/dE`` per g/cm^2.  This module stores ``int_blocks`` **without** the loss and the
loss as separate arrays (``loss_species``, ``loss_blocks`` = the averaged operator
actually added, ``loss_blocks_raw`` = ``Op``, ``dedx``); ``rebuild(d, loss=True)`` adds
it back.  ``rebuild(d, loss=False)`` equals MCEq run with ``enable_energy_loss =
False`` (tested).

Resonance regime and mixing.  Species with ``mix_idx > 0`` (``is_mixed``) behave as
"resonances" in bins ``0 .. mix_idx-1`` (decay instantly, ``Dec`` set to 0 there, their
products are attached to the parent via ``_follow_chains``) and as normal hadrons above
(``config.hybrid_crossover`` = 0.5, ``config.max_density``).  There are no
separate resonance species in the system (``resonances`` is empty in this database).
``adv_set`` entries are recorded in ``meta``.

Initial condition: ``phi0`` is MCEq's ``_phi0`` after ``set_primary_model``: for the
proton and neutron slices ``1e-4 * p_top`` / ``1e-4 * n_top`` (the primary model's
nucleon flux at total energy per nucleon ``e_grid + (m_p+m_n)/2``, per GeV, with the
1e-4 converting m^-2 to cm^-2) from bin ``phi0_min_idx`` upward; zero below.

Everything is also available in ``meta`` (a dict, stored as JSON text in the npz).
"""

import hashlib
import importlib.util  # noqa: F401  (MCEq.config must be imported after this)
import json
import os
import contextlib
from types import SimpleNamespace

import numpy as np

N3 = os.environ.get("N3", "/home/pigrange/mc3d_work/runs/native3d")
KERNEL_DIR = os.path.join(N3, "kernels")

#: The two MCEq populations of tools/mceq3d/INTEGRATION.md (floor in GeV of TOTAL
#: nucleon energy = MCEq ``minimal_primary_energy``; ``e_min`` in GeV kinetic).
POPULATIONS = {
    "PHYS": {"primary_floor": 1.9, "e_min": 0.03},
    "REF": {"primary_floor": 3.0, "e_min": 0.1},
}
DEFAULT_MODEL = "SIBYLL23D"
DEFAULT_PRIMARY = ("HillasGaisser2012", "H3a")

_CONFIG_KEYS = [
    "e_min", "e_max", "minimal_primary_energy", "enable_energy_loss",
    "enable_cont_rad_loss", "generic_losses_all_charged", "enable_em_ion",
    "average_loss_operator", "loss_step_for_average", "muon_helicity_dependence",
    "use_isospin_sym", "hybrid_crossover", "max_density", "dXmax", "stability_margin",
    "leading_process", "enable_default_tracking", "interaction_medium", "density_model",
    "h_obs", "h_atm", "r_E", "X_start", "kernel_config", "integrator", "enable_em",
    "assume_nucleon_interactions_for_exotics", "fallback_to_air_cs",
    "low_energy_extension", "return_as", "adv_set", "mceq_db_fname", "A_target",
]


def kernel_path(population):
    """File ``$N3/kernels/mceq_<population>.npz``."""
    _check_population(population)
    return os.path.join(KERNEL_DIR, f"mceq_{population}.npz")


def _check_population(population):
    if population not in POPULATIONS:
        raise ValueError(f"population must be one of {sorted(POPULATIONS)}")


@contextlib.contextmanager
def _config(**kw):
    """Temporarily set ``MCEq.config`` attributes; restored on exit."""
    from MCEq import config as cfg

    old = {k: getattr(cfg, k) for k in kw}
    for k, v in kw.items():
        setattr(cfg, k, v)
    try:
        yield cfg
    finally:
        for k, v in old.items():
            setattr(cfg, k, v)


def make_mceq(population, energy_loss=True, interaction_model=DEFAULT_MODEL,
              primary=DEFAULT_PRIMARY, theta_deg=0.0):
    """Build the ``MCEqRun`` of a population exactly as ``tools/mceq3d`` does.

    ``mceq3d_db.make_run`` selects the database (default ``v142p``); the floor and
    ``e_min`` are set in ``MCEq.config`` while the run is constructed (the floor is
    read in ``set_primary_model``, ``e_min`` when the energy grid is read) and
    restored afterwards.  ``energy_loss=False`` sets ``enable_energy_loss`` False
    during construction (the switch is read by ``construct_matrices``).

    Returns
    -------
    MCEq.core.MCEqRun
        With ``.mceq3d_db`` (the ``DBSpec`` used) attached.
    """
    _check_population(population)
    from mceq3d_db import make_run
    from mceq3d_primary import primary_class

    pop = POPULATIONS[population]
    pm = (primary_class(primary[0]), primary[1])
    with _config(e_min=pop["e_min"], minimal_primary_energy=pop["primary_floor"],
                 enable_energy_loss=bool(energy_loss)) as cfg:
        run = make_run(interaction_model=interaction_model, primary_model=pm,
                       theta_deg=theta_deg)
        # snapshot of the settings in force while the matrices were built
        run.c3d_config = SimpleNamespace(
            **{k: getattr(cfg, k) for k in _CONFIG_KEYS if hasattr(cfg, k)})
    return run


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def _blocks_from_sparse(mat, nE, nS):
    """Dense non-zero (nE, nE) blocks of a sparse matrix: {(parent, child): array}."""
    coo = mat.tocoo()
    keep = coo.data != 0.0
    row, col, val = coo.row[keep], coo.col[keep], coo.data[keep]
    child, parent = row // nE, col // nE
    key = parent.astype(np.int64) * nS + child
    order = np.argsort(key, kind="stable")
    key, row, col, val = key[order], row[order], col[order], val[order]
    cuts = np.flatnonzero(np.diff(key)) + 1
    out = {}
    for sel in np.split(np.arange(len(key)), cuts):
        if len(sel) == 0:
            continue
        p, c = divmod(int(key[sel[0]]), nS)
        blk = np.zeros((nE, nE))
        blk[row[sel] % nE, col[sel] % nE] = val[sel]
        out[(p, c)] = blk
    return out


def _loss_applies(p, cfg):
    pid = abs(p.pdg_id[0])
    return bool(
        p.has_contloss and cfg.enable_energy_loss
        and (pid == 13 or (cfg.enable_em_ion and pid == 11)
             or (cfg.generic_losses_all_charged and pid != 11))
    )


def _dist_version():
    try:
        from importlib.metadata import version

        return version("MCEq")
    except Exception:  # pragma: no cover
        return None


def _json_safe(x):
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def extract(run, population, energy_loss=True, primary=DEFAULT_PRIMARY,
            interaction_model=DEFAULT_MODEL):
    """Cut a constructed ``MCEqRun`` into the kernel dictionary (see module docs)."""
    from MCEq import config as live_cfg

    cfg = getattr(run, "c3d_config", live_cfg)  # settings used to build the run
    pop = POPULATIONS[population]
    nE, P = int(run.dim), run.pman
    parts = sorted(P.cascade_particles, key=lambda p: p.mceqidx)
    nS = len(parts)
    assert [p.mceqidx for p in parts] == list(range(nS))
    assert run.int_m.shape == (nS * nE, nS * nE)
    c = np.asarray(run.e_grid, float)

    def arr(f, dtype=float):
        return np.array([f(p) for p in parts], dtype=dtype)

    def row(f):
        out = np.zeros((nS, nE))
        for i, p in enumerate(parts):
            try:
                v = f(p)
            except (TypeError, AttributeError, ZeroDivisionError):
                v = None
            if v is not None:
                out[i] = np.asarray(v, float)
        return out

    d = {
        "e_grid": c, "e_bins": np.asarray(run.e_bins, float),
        "e_widths": np.asarray(run.e_widths, float),
        "species": np.array([p.name for p in parts]),
        "pdg": arr(lambda p: p.pdg_id[0], np.int64),
        "helicity": arr(lambda p: p.helicity, np.int64),
        "mass": arr(lambda p: p.mass), "charge": arr(lambda p: p.charge, np.int64),
        "ctau": arr(lambda p: p.ctau),
        "is_hadron": arr(lambda p: p.is_hadron, bool),
        "is_lepton": arr(lambda p: p.is_lepton, bool),
        "is_stable": arr(lambda p: p.is_stable, bool),
        "is_tracking": arr(lambda p: p.is_tracking, bool),
        "is_resonance": arr(lambda p: p.is_resonance, bool),
        "is_mixed": arr(lambda p: p.is_mixed, bool),
        "mix_idx": arr(lambda p: p.mix_idx, np.int64),
        "e_mix": arr(lambda p: p.E_mix),
        "can_interact": arr(lambda p: p.can_interact, bool),
        "is_projectile": arr(lambda p: p.is_projectile, bool),
        "has_contloss": arr(lambda p: p.has_contloss, bool),
        "is_em": arr(lambda p: p.is_em, bool),
        "lidx": arr(lambda p: p.lidx, np.int64),
        "uidx": arr(lambda p: p.uidx, np.int64),
        "cs": row(lambda p: p.cs if p.can_interact else None),
        "inv_lambda_int": row(
            lambda p: p.inverse_interaction_length() if p.can_interact else None),
        "inv_dec_len": row(lambda p: None if p.is_stable
                           else p.inverse_decay_length(cut=True)),
        "inv_dec_len_uncut": row(lambda p: None if p.is_stable
                                 else p.inverse_decay_length(cut=False)),
        "phi0": np.asarray(run._phi0, float).copy(),
    }
    nuc = 0.5 * (P[(2212, 0)].mass + P[(2112, 0)].mass)
    e_tot = c + nuc
    d["phi0_min_idx"] = np.array(
        int(np.argmin(np.abs(e_tot - pop["primary_floor"]))))
    d["e_tot_nucleon"] = e_tot

    # matrices: interaction (loss split off) and decay
    mb = run.matrix_builder
    int_b = _blocks_from_sparse(run.int_m, nE, nS)
    dec_b = _blocks_from_sparse(run.dec_m, nE, nS)
    loss_sp, loss_blk, loss_raw, dedx = [], [], [], []
    for p in parts:
        if not _loss_applies(p, cfg):
            continue
        op = np.asarray(mb.cont_loss_operator(p.pdg_id), float)
        raw = -np.diag(1.0 / c) @ mb.op_matrix @ np.diag(p.dEdX)
        key = (p.mceqidx, p.mceqidx)
        diff = int_b.get(key, np.zeros((nE, nE))) - op
        if np.any(diff != 0.0):
            int_b[key] = diff
        else:
            int_b.pop(key, None)
        loss_sp.append(p.mceqidx)
        loss_blk.append(op)
        loss_raw.append(raw)
        dedx.append(np.asarray(p.dEdX, float))
    d["loss_species"] = np.array(loss_sp, dtype=np.int64)
    d["loss_blocks"] = np.array(loss_blk).reshape(len(loss_sp), nE, nE)
    d["loss_blocks_raw"] = np.array(loss_raw).reshape(len(loss_sp), nE, nE)
    d["dedx"] = np.array(dedx).reshape(len(loss_sp), nE)
    d["loss_diff_op"] = np.asarray(mb.op_matrix, float)
    for name, blocks in (("int", int_b), ("dec", dec_b)):
        keys = sorted(blocks)
        d[f"{name}_parent"] = np.array([k[0] for k in keys], dtype=np.int32)
        d[f"{name}_child"] = np.array([k[1] for k in keys], dtype=np.int32)
        d[f"{name}_blocks"] = np.array([blocks[k] for k in keys]).reshape(
            len(keys), nE, nE)

    dbspec = run.mceq3d_db
    db_path = run._mceq_db.had_fname
    sha = _sha256_file(db_path)
    meta = {
        "population": population, "primary_floor": pop["primary_floor"],
        "e_min_population": pop["e_min"], "nE": nE, "nS": nS,
        "mceq_version": __import__("MCEq.version", fromlist=["x"]).__version__,
        "mceq_dist_version": _dist_version(),
        "A_target": float(P[(2212, 0)].A_target),
        "interaction_model": interaction_model, "primary_model": list(primary),
        "primary_model_name": run.pmodel.name,
        "db_label": dbspec.label, "db_file": os.path.basename(db_path),
        "db_path_loaded": db_path, "db_sha256": sha,
        "db_sha256_mceq3d_db": dbspec.sha256,
        "db_hdf_version": str(run._mceq_db.version),
        "energy_loss": bool(energy_loss),
        "config": {k: _json_safe(getattr(cfg, k)) for k in _CONFIG_KEYS
                   if hasattr(cfg, k)},
        "max_lint": float(mb.max_lint), "max_ldec": float(mb.max_ldec),
        "n_int_blocks": int(len(int_b)), "n_dec_blocks": int(len(dec_b)),
        "n_loss_species": len(loss_sp),
        "state_units": "dPhi/dE_kin [1/(GeV cm2 s sr)] on e_grid (kinetic energy)",
        "recipe": "tools/mceq3d/mceq3d_db.make_run; config.e_min and "
                  "config.minimal_primary_energy set during construction",
    }
    assert sha == dbspec.sha256, "database hash differs from mceq3d_db's"
    d["meta_json"] = np.array(json.dumps(meta))
    return d


def build(population, energy_loss=True, primary=DEFAULT_PRIMARY,
          interaction_model=DEFAULT_MODEL, run=None):
    """Build the kernel dictionary of ``population`` ("PHYS" or "REF") from MCEq."""
    if run is None:
        run = make_mceq(population, energy_loss, interaction_model, primary)
    return extract(run, population, energy_loss, primary, interaction_model)


def save(d, population=None, path=None):
    """Write ``d`` to ``$N3/kernels/mceq_<population>.npz`` (atomic, compressed)."""
    meta = json.loads(str(d["meta_json"]))
    population = population or meta["population"]
    path = path or kernel_path(population)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **d)
    os.replace(tmp, path)
    return path


def load(population, path=None):
    """Read the kernel file (``allow_pickle=False``); adds ``d["meta"]`` (a dict)."""
    path = path or kernel_path(population)
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta_json"]))
    return d


def species_index(d, name):
    """MCEq index of the species called ``name`` (e.g. ``"numu"``, ``"mu+_l"``)."""
    idx = np.flatnonzero(d["species"] == name)
    if len(idx) != 1:
        raise KeyError(name)
    return int(idx[0])


def get_block(d, kind, parent, child, loss=True):
    """Dense ``(nE_child, nE_parent)`` block of ``kind`` ("int" or "dec").

    Zeros if the pair is not stored.  For ``kind == "int"`` and ``parent == child``
    the continuous-loss operator is added if ``loss``.
    """
    nE = d["e_grid"].size
    out = np.zeros((nE, nE))
    sel = np.flatnonzero(
        (d[f"{kind}_parent"] == parent) & (d[f"{kind}_child"] == child))
    if len(sel):
        out += d[f"{kind}_blocks"][sel[0]]
    if kind == "int" and loss and parent == child:
        j = np.flatnonzero(d["loss_species"] == parent)
        if len(j):
            out += d["loss_blocks"][j[0]]
    return out


def rebuild(d, loss=True):
    """Reassemble ``(int_m, dec_m)`` (scipy CSR, ``nS*nE`` square) from the blocks.

    With ``loss=True`` (default) the continuous-loss blocks are added to the diagonal
    blocks as MCEq does (``int_m`` then equals MCEq's, ``enable_energy_loss`` on);
    with ``loss=False`` the result equals MCEq built with ``enable_energy_loss =
    False``.
    """
    from scipy.sparse import coo_matrix

    nE, nS = d["e_grid"].size, d["species"].size
    n = nS * nE
    out = []
    for kind in ("int", "dec"):
        par, chi, blk = d[f"{kind}_parent"], d[f"{kind}_child"], d[f"{kind}_blocks"]
        parts = {(int(p), int(c)): blk[j] for j, (p, c) in enumerate(zip(par, chi))}
        if kind == "int" and loss:
            for j, s in enumerate(d["loss_species"]):
                key = (int(s), int(s))
                parts[key] = parts.get(key, 0.0) + d["loss_blocks"][j]
        rows, cols, vals = [], [], []
        for (p, c), b in parts.items():
            r, k = np.nonzero(b)
            rows.append(c * nE + r)
            cols.append(p * nE + k)
            vals.append(b[r, k])
        m = coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n))
        out.append(m.tocsr())
    return out[0], out[1]
