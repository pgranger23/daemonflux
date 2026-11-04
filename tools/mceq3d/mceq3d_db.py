"""The ONE place where ``tools/mceq3d`` chooses the MCEq interaction database.

Why: MCEq's yields and inelastic cross sections come from an HDF5 file
(``MCEq.config.data_dir / mceq_db_fname``).  The file that ships with MCEq 1.4.1,
``mceq_db_lext_dpm193_v140.h5``, was withdrawn upstream (SIBYLL neutrons simulated
as protons, wrong DPMJET-III-19.3 antiparticle cross sections); the v1.4.2 file
fixes those but still has the 56.2 GeV hole of ``MCEQ_DB_HOLE.md``.  The
engine's default is therefore the **v1.4.2 file with the hole patched**
(variant ``interp`` of ``diag_mceq_db_hole.py patch``).

Selection (``MCEQ3D_DB`` environment variable, read when an ``MCEqRun`` is made)::

    v142p  (default)  $MCEQ3D_DB_DIR/mceq_db_lext_dpm193_v142_dbhole_interp.h5
    v142              unpatched v1.4.2 (hole present)
    v140p             v1.4.0/1.4.1 file with the hole patched (interp)
    v140              the file shipped with the installed MCEq 1.4.1 (the
                      pre-2026-11 engine behaviour, bit for bit)
    <path>            any HDF5 file (its directory should hold DDM_1.0.npy if the
                      data-driven model is used)

Every ``MCEqRun`` in ``tools/mceq3d`` is built with :func:`make_run`, which calls
:func:`select` first.  Nothing else in the package may set ``data_dir`` or
``mceq_db_fname``.

Cache identity.  Disk caches that depend on MCEq yields or cross sections
(``gs_*`` G_s responses, ``jointprod_chan_*``, ``channel_fractions``) carry
:attr:`DBSpec.tag` in their key.  The tag is ``""`` for the shipped v1.4.0 file
(recognised by its sha256), so the caches built before the switch (in
``.cache3d``) stay valid, and ``"_db<label>-<sha8>"`` for any other file: the
sha256 of the actual file, so a file replaced under the same name changes the key.
Pure geomagnetic maps (``finerc_*``, ``prodfam_*``, ``rc_*``) and the cone widths
do not depend on the database and are never re-keyed; for a non-legacy database
they are read from the requested cache directory, copied (not moved) to
:func:`cache_dir` and written there.  :func:`cache_dir` is ``$MCEQ3D_CACHE``
(default ``/home/pigrange/mc3d_work/mceq3d_cache``) -- never AFS.
"""

import hashlib
import importlib.util  # noqa: F401  (MCEq.config needs it imported first)
import json
import os
import shutil
from dataclasses import dataclass

W = os.environ.get("MC3D_WORK", "/home/pigrange/mc3d_work")
DB_DIR = os.environ.get("MCEQ3D_DB_DIR", os.path.join(W, "mceq_db"))
CACHE_ROOT = os.environ.get("MCEQ3D_CACHE", os.path.join(W, "mceq3d_cache"))
DEFAULT = "v142p"
#: sha256 of the file shipped with MCEq 1.4.1 (cache keys for it stay unsuffixed)
LEGACY_SHA256 = "5da415e9bcf81926b1061d5792d75cb3aceb9de173beccb4695fd3909a0bfdd0"

_NAMED = {
    "v142p": (DB_DIR, "mceq_db_lext_dpm193_v142_dbhole_interp.h5"),
    "v142": (os.path.join(DB_DIR, "v142_raw"), "mceq_db_lext_dpm193_v142.h5"),
    "v140p": (
        os.path.join(W, "runs", "dbhole", "data_interp"),
        "mceq_db_lext_dpm193_v140_dbhole_interp.h5",
    ),
    "v140": None,  # MCEq's own data_dir / mceq_db_fname, as imported
}
_SHIPPED = {}  # filled on first use: MCEq's pristine (data_dir, mceq_db_fname)
_SHA_MEMO = os.path.join(DB_DIR, ".sha256_memo.json")


@dataclass(frozen=True)
class DBSpec:
    label: str
    data_dir: str
    fname: str
    sha256: str

    @property
    def path(self):
        return os.path.join(self.data_dir, self.fname)

    @property
    def legacy(self):
        return self.sha256 == LEGACY_SHA256

    @property
    def tag(self):
        """Cache-key suffix: ``""`` for the shipped v1.4.0 file, else ``_db<label>-<sha8>``."""
        return "" if self.legacy else f"_db{self.label}-{self.sha256[:8]}"


def _sha256(path):
    """sha256 of ``path``, memoised on disk by (realpath, size, mtime_ns)."""
    real = os.path.realpath(path)
    st = os.stat(real)
    key = f"{real}|{st.st_size}|{st.st_mtime_ns}"
    try:
        with open(_SHA_MEMO) as fh:
            memo = json.load(fh)
    except Exception:
        memo = {}
    if key in memo:
        return memo[key]
    h = hashlib.sha256()
    with open(real, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    memo[key] = h.hexdigest()
    try:
        os.makedirs(os.path.dirname(_SHA_MEMO), exist_ok=True)
        with open(_SHA_MEMO, "w") as fh:
            json.dump(memo, fh)
    except OSError:
        pass  # memo is an optimisation only
    return memo[key]


def _shipped():
    if not _SHIPPED:
        import importlib.util  # noqa: F401  (must precede the MCEq import)
        import MCEq.config as cfg

        _SHIPPED["dir"], _SHIPPED["fname"] = str(cfg.data_dir), cfg.mceq_db_fname
    return _SHIPPED["dir"], _SHIPPED["fname"]


_resolved = {}


def resolve(spec=None):
    """:class:`DBSpec` for ``spec`` (default: ``$MCEQ3D_DB`` or :data:`DEFAULT`)."""
    spec = spec or os.environ.get("MCEQ3D_DB") or DEFAULT
    if spec in _resolved:
        return _resolved[spec]
    if spec in _NAMED:
        loc, label = _NAMED[spec], spec
        if loc is None:
            loc = _shipped()
    else:
        p = os.path.abspath(os.path.expanduser(spec))
        loc, label = (os.path.dirname(p), os.path.basename(p)), None
        label = os.path.splitext(os.path.basename(p))[0]
    d, n = loc
    path = os.path.join(d, n)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"MCEQ3D_DB={spec!r}: database file {path} not found "
            "(see tools/mceq3d/MCEQ_DB_HOLE.md section 10 for how to build it)"
        )
    out = DBSpec(label, str(d), n, _sha256(path))
    _resolved[spec] = out
    return out


def select(spec=None):
    """Point ``MCEq.config`` at the chosen database; return its :class:`DBSpec`.

    Must run before an ``MCEqRun`` is created (:func:`make_run` does it).  Only
    ``data_dir`` and ``mceq_db_fname`` are touched.
    """
    s = resolve(spec)
    import pathlib

    import importlib.util  # noqa: F401
    import MCEq.config as cfg

    cfg.data_dir = pathlib.Path(s.data_dir)
    cfg.mceq_db_fname = s.fname
    return s


def make_run(**kwargs):
    """``MCEqRun(**kwargs)`` on the selected database (the only constructor to use)."""
    s = select()
    from MCEq.core import MCEqRun

    run = MCEqRun(**kwargs)
    run.mceq3d_db = s
    return run


def db_tag(spec=None):
    """Cache-key suffix of the selected database (see :attr:`DBSpec.tag`)."""
    return resolve(spec).tag


def cache_dir(requested=None):
    """Directory for DB-dependent cache files.

    ``requested`` unchanged for the legacy database (old caches stay valid);
    :data:`CACHE_ROOT` otherwise.
    """
    if requested is not None and resolve().legacy:
        return requested
    d = CACHE_ROOT
    os.makedirs(d, exist_ok=True)
    return d


def indep_path(requested, name):
    """Path of a database-INDEPENDENT cache file (geomagnetic map, cone widths).

    Legacy database: ``requested/name`` as before.  Otherwise
    ``cache_dir()/name``, first copied from ``requested/name`` when only the
    latter exists (copy, never move; no re-keying).
    """
    if requested is None or resolve().legacy:
        return os.path.join(requested if requested is not None else "", name)
    dst = os.path.join(cache_dir(requested), name)
    src = os.path.join(requested, name)
    if not os.path.exists(dst) and os.path.exists(src):
        shutil.copy2(src, dst)
    return dst


_WARNED = set()


def check_static_table(name, built_with="", sha256=None):
    """Warn (once) when a checked-in MCEq-derived table is used with another database.

    ``built_with`` is the database tag recorded in the table ("" = the shipped
    v1.4.0 file, i.e. every table that predates the 2026-11 switch).  These tables
    (``offaxis_excess*.npz``, ``channel_fractions.npz``) are static artefacts, not
    memos; they are not rebuilt automatically.
    """
    cur = db_tag()
    if sha256 and sha256 != resolve().sha256 and cur == built_with:
        cur = cur + "(sha256 differs)"
    if cur != built_with and (name, cur) not in _WARNED:
        import warnings

        _WARNED.add((name, cur))
        warnings.warn(
            f"static table {name} was built with MCEq database "
            f"{built_with or 'v140 (shipped 1.4.1 file)'}, the engine uses "
            f"{cur or 'v140 (shipped 1.4.1 file)'}; rebuild it for exactness "
            "(MCEQ_DB_HOLE.md section 10).",
            stacklevel=3,
        )


def dep_path(requested, name):
    """Path of a yield-/cross-section-DEPENDENT cache file (G_s, production profile,
    channel fractions, ...).

    Reading: for the legacy (shipped v1.4.0) database an existing
    ``requested/name`` is used as it is (the old ``.cache3d`` entries stay valid).
    Writing: everything else -- including every *new* file of the legacy database --
    goes to :data:`CACHE_ROOT` (``$MCEQ3D_CACHE``, never the repository / AFS).
    """
    if requested is not None and resolve().legacy:
        p = os.path.join(requested, name)
        if os.path.exists(p):
            return p
    os.makedirs(CACHE_ROOT, exist_ok=True)
    return os.path.join(CACHE_ROOT, name)
