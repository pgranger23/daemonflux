"""Sensitivity of the engine 3D/1D factor F_s(E, cosZ, az) to hadronic / primary /
cone / solar variations (study of 2026-11-07, run dir $W/runs/sens).

F is the ``joint_cone_factor`` (down-going) or Eoff x Geff (up-going) returned by
``MCEq3DFlux.solve``; ``solve`` multiplies it onto ``base`` so F = flux / base is
insensitive to the 1D response by construction (the 1D response cancels).

This module WRAPS the engine, it changes no tracked code:

* a hadronic variation is applied as MCEq ``set_mod_pprod`` yield blocks (Barr-style
  scale of one secondary's dN/dx_lab, energy-weighted by hat functions in ln E_proj
  between the daemonflux beam-energy nodes) -- a PROXY for the daemonflux blocks;
* a primary variation (GSF proxy, solar) multiplies the nucleon fluxes ``_phi0``;
* the SAME variation is applied to (a) the engine run (G_s, base) and (b) the
  H3a production-profile run behind ``joint_prod`` (via a patched
  ``mceq3d_db.make_run``), so G_s and the pi/K/mu parent-channel mix both respond;
* disk caches: ``mceq3d_db.CACHE_ROOT`` is pointed at a private directory per
  variation; only database-independent maps are symlinked in (read-only).

Frozen: cutoff maps (finerc/rc/prodfam), cone widths, E_off table, static
``channel_fractions`` (only used by the up-going/product path), muon bending off.
"""
import glob
import os
import shutil
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
W = os.environ.get("MC3D_WORK", "/home/pigrange/mc3d_work")
RUN = os.path.join(W, "runs", "sens")
SHARED = os.path.join(W, "mceq3d_cache")  # read-only
BASE_COPY = os.path.join(RUN, "cache", "base")  # private copy of SHARED
GRIDS = None  # filled by grids()

# beam-energy nodes [GeV] per secondary family (daemonflux block labels)
NODES = {
    "pi": (("31G", 31.0), ("158G", 158.0), ("20T", 2.0e4), ("2P", 2.0e6)),
    "K": (("31G", 31.0), ("158G", 158.0), ("2P", 2.0e6)),
    "N": (("158G", 158.0), ("2P", 2.0e6)),
}
SEC_PDG = {"K+": 321, "K-": -321, "pi+": 211, "pi-": -211, "p": 2212, "n": 2112}


def hadronic_names():
    out = []
    for s in ("K+", "K-", "n", "p", "pi+", "pi-"):
        fam = "pi" if s.startswith("pi") else ("K" if s.startswith("K") else "N")
        for lab, _ in NODES[fam]:
            out.append(f"{s}_{lab}")
    return out


def block_weight(e_proj, fam, label):
    """Hat function in ln E_proj of one node; flat beyond the end nodes."""
    nodes = NODES[fam]
    x = np.log(np.asarray(e_proj, float))
    en = np.log([n[1] for n in nodes])
    i = [n[0] for n in nodes].index(label)
    w = np.zeros_like(x)
    if i > 0:
        m = (x >= en[i - 1]) & (x <= en[i])
        w[m] = (x[m] - en[i - 1]) / (en[i] - en[i - 1])
    else:
        w[x <= en[i]] = 1.0
    if i < len(nodes) - 1:
        m = (x > en[i]) & (x <= en[i + 1])
        w[m] = (en[i + 1] - x[m]) / (en[i + 1] - en[i])
    else:
        w[x > en[i]] = 1.0
    return w


def blockmod(x, E, key, delta):
    """MCEq ``x_func``: 1 + delta * w(E_proj) for all x_lab (columns = projectile).
    ``key`` = "fam:label" (MCEq's isospin code wants args = (name, value))."""
    fam, label = key.split(":")
    w = block_weight(E, fam, label)  # (nE,) projectile energy
    return 1.0 + delta * np.broadcast_to(w[None, :], x.shape).copy()


def apply_yield_mods(run, mods):
    """``mods``: list of (parameter name, delta). Replaces any previous mods."""
    run.unset_mod_pprod(dont_fill=True)
    for name, delta in mods:
        sec, lab = name.split("_", 1)
        fam = "pi" if sec.startswith("pi") else ("K" if sec.startswith("K") else "N")
        run.set_mod_pprod(2212, SEC_PDG[sec], blockmod, (f"{fam}:{lab}", float(delta)))
    run.regenerate_matrices()


def primary_factors(eng, phi_gv=None, extra=None):
    """(fp, fn) per-bin multiplicative factors on the p / n slices of ``_phi0``.

    ``phi_gv``: force-field potential DIFFERENCE added to the baseline
    (``_modulate_phi0``); ``extra``: callable(E_kin_grid) -> factor on both.
    """
    fp = np.ones(eng._p_sl.stop - eng._p_sl.start)
    fn = np.ones(eng._n_sl.stop - eng._n_sl.start)
    if phi_gv:
        m = eng._modulate_phi0(phi_gv)
        std = eng._phi0_std
        with np.errstate(invalid="ignore", divide="ignore"):
            fp = np.where(std[eng._p_sl] > 0, m[eng._p_sl] / std[eng._p_sl], 1.0)
            fn = np.where(std[eng._n_sl] > 0, m[eng._n_sl] / std[eng._n_sl], 1.0)
    if extra is not None:
        g = extra(eng.e)
        fp, fn = fp * g, fn * g
    return fp, fn


class Variation:
    """One perturbation: yield mods, primary factors, cone scale."""

    def __init__(self, tag, mods=(), phi_gv=0.0, prim_extra=None, cone_sigma_scale=1.0):
        self.tag = tag
        self.mods = list(mods)
        self.phi_gv = phi_gv
        self.prim_extra = prim_extra
        self.cone_sigma_scale = cone_sigma_scale


_ORIG_MAKE_RUN = None
_CURRENT = {"var": None, "fp": None, "fn": None}


def _patched_make_run(**kw):
    """make_run that applies the current variation to every NEW MCEqRun (the H3a
    production-profile run of ``offaxis_mc.production_profile``)."""
    run = _ORIG_MAKE_RUN(**kw)
    var = _CURRENT["var"]
    if var is not None:
        if var.mods:
            apply_yield_mods(run, var.mods)
        fp, fn = _CURRENT["fp"], _CURRENT["fn"]
        if fp is not None:
            p, n = run.pman[(2212, 0)], run.pman[(2112, 0)]
            run._phi0[p.lidx:p.uidx] *= fp
            run._phi0[n.lidx:n.uidx] *= fn
    return run


def install_patch():
    global _ORIG_MAKE_RUN
    import mceq3d_db

    if _ORIG_MAKE_RUN is None:
        _ORIG_MAKE_RUN = mceq3d_db.make_run
        mceq3d_db.make_run = _patched_make_run


def build_engine():
    from mceq3d_flux import MCEq3DFlux

    warnings.simplefilter("ignore")
    eng = MCEq3DFlux(base_model="mceq", primary=("GlobalSplineFitBeta", None))
    eng._phi0_base = eng._phi0_std.copy()
    return eng


def private_cache(tag, link_maps=True):
    """Fresh private cache root for one variation (db-independent maps symlinked)."""
    import mceq3d_db

    d = os.path.join(RUN, "cache", tag)
    os.makedirs(d, exist_ok=True)
    if link_maps:
        for pat in ("finerc_*", "rc_*", "prodfam_*", "conewidths_*"):
            for f in glob.glob(os.path.join(BASE_COPY, pat)):
                dst = os.path.join(d, os.path.basename(f))
                if not os.path.exists(dst):
                    os.symlink(f, dst)
    mceq3d_db.CACHE_ROOT = d
    return d


def set_variation(eng, var):
    """Put the engine into the state of ``var`` (None = baseline)."""
    install_patch()
    if var is None:
        var = Variation("baseline")
    fp, fn = primary_factors(eng, var.phi_gv, var.prim_extra)
    _CURRENT.update(var=var, fp=fp, fn=fn)
    apply_yield_mods(eng.mceq, var.mods)
    eng._phi0_std = eng._phi0_base.copy()
    eng._phi0_std[eng._p_sl] *= fp
    eng._phi0_std[eng._n_sl] *= fn
    eng.mceq._phi0[:] = eng._phi0_std
    eng._joint_prod = None
    eng._joint_q = {}
    eng._channel_by_zen = {}


def cpu_s():
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


def solve_F(eng, var, cache_tag=None, grids=("hor", "dn6", "up"), log=print):
    """F[name][species] = flux/base (ncz, naz, nE) for each grid under ``var``."""
    import diag_integ as di

    set_variation(eng, var)
    d = private_cache(cache_tag or var.tag)
    G = {
        "hor": (di.CZ_HOR, di.AZ12),
        "dn6": (di.CZ_DN6, di.AZ6),
        "up": (di.CZ_UP, di.AZ12),
    }
    out = {}
    for name in grids:
        cz, az = G[name]
        t = time.time()
        c0 = cpu_s()
        r = eng.solve(di.LAT, di.LON, cz, az, date=di.DATE, use_cache=True,
                      cache_dir=di.CACHE_REQ, muon_bending=False, n_jobs=2,
                      cone_sigma_scale=var.cone_sigma_scale)
        base = eng.base(np.abs(cz))
        out[name] = {sp: r["flux"][sp] / base[sp][:, None, :] for sp in di.SPECIES}
        out[name]["base"] = base
        log(f"  [{var.tag}] {name}: {time.time() - t:.0f}s wall, {cpu_s() - c0:.0f}s cpu",
            flush=True)
    return out, d


def assert_shared_untouched(snapshot):
    now = {f: os.path.getmtime(os.path.join(SHARED, f)) for f in os.listdir(SHARED)}
    assert now == snapshot, "SHARED cache modified!"


def snapshot_shared():
    return {f: os.path.getmtime(os.path.join(SHARED, f)) for f in os.listdir(SHARED)}
