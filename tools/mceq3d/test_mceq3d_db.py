"""The MCEq database switch (mceq3d_db.py) and the cache keys that depend on it.

These tests never build an MCEqRun: they check the identity bookkeeping only
(file sha256 -> tag -> cache key / cache directory), which is what keeps a cache
built with one database from being served for another.
"""
import importlib.util  # noqa: F401
import glob
import os
import re

import numpy as np
import pytest

import mceq3d_db as db
from mceq3d_flux import MCEq3DFlux, engine_tag, gs_cache_name, SPECIES

HERE = os.path.dirname(os.path.abspath(__file__))
RC = np.array([7.0, 11.0, 40.0])


def _have(spec):
    try:
        db.resolve(spec)
        return True
    except FileNotFoundError:
        return False


need_new = pytest.mark.skipif(not _have("v142p") or not _have("v140"),
                              reason="v140 / v142p database files not present")


def test_default_is_patched_v142(monkeypatch):
    monkeypatch.delenv("MCEQ3D_DB", raising=False)
    assert db.DEFAULT == "v142p"
    if _have("v142p"):
        s = db.resolve()
        assert s.label == "v142p" and "v142" in s.fname and not s.legacy
        assert s.tag.startswith("_dbv142p-") and len(s.tag) == len("_dbv142p-") + 8


@need_new
def test_env_selects_database(monkeypatch):
    monkeypatch.setenv("MCEQ3D_DB", "v140")
    old = db.resolve()
    assert old.legacy and old.sha256 == db.LEGACY_SHA256 and old.tag == ""
    monkeypatch.setenv("MCEQ3D_DB", "v142p")
    new = db.resolve()
    assert new.sha256 != old.sha256 and new.tag != old.tag
    # an explicit path works and is identified by file content, not by name
    monkeypatch.setenv("MCEQ3D_DB", new.path)
    assert db.resolve().sha256 == new.sha256


@need_new
def test_cache_key_changes_with_database():
    """Same model/primary/e_min/atmosphere/grid: different database -> different
    engine tag and different G_s cache file name; the legacy key is unchanged."""
    t_old = db.resolve("v140").tag
    t_hole = db.resolve("v142").tag      # v1.4.2 unpatched
    t_new = db.resolve("v142p").tag      # v1.4.2 patched
    assert len({t_old, t_hole, t_new}) == 3
    args = ("SIBYLL23D", ("GlobalSplineFitBeta", None), 0.1, None)
    tags = [engine_tag(*args, t) for t in (t_old, t_hole, t_new)]
    assert tags[0] == "SIBYLL23D_GlobalSplineFitBeta-None_emin0.1_atmstd"  # legacy
    names = [gs_cache_name(t, 1.0, RC, 0.5) for t in tags]
    assert len(set(names)) == 3
    # prefix property used by offaxis_factor(): the table tag is still a prefix
    assert all(t.startswith("SIBYLL23D_GlobalSplineFitBeta") for t in tags)


@need_new
def test_stale_gs_cache_never_served(tmp_path, monkeypatch):
    """A G_s file stored under the OLD database's key is found for the old
    database and is not found for the new one (which must recompute)."""
    monkeypatch.setattr(db, "CACHE_ROOT", str(tmp_path / "newcache"))
    legacy_dir = tmp_path / "oldcache"
    legacy_dir.mkdir()
    e = np.logspace(-1, 3, 21)

    class Boom(Exception):
        pass

    class Bare(MCEq3DFlux):
        def __init__(self, spec):
            self.e = e
            self._db = db.resolve(spec)
            self._tag = self._gs_tag = engine_tag(
                "SIBYLL23D", ("GlobalSplineFitBeta", None), 0.1, None, self._db.tag)

        def _gs_engine(self):
            raise Boom

    monkeypatch.setenv("MCEQ3D_DB", "v140")
    old = Bare("v140")
    sigma = 0.5
    fake = {s: np.full((len(RC), len(e)), 0.5) for s in SPECIES}
    np.savez(legacy_dir / gs_cache_name(old._gs_tag, 1.0, RC, sigma), e=e, **fake)
    G, _ = old.geomag_response(RC, 1.0, cache_dir=str(legacy_dir), sigma_lnr=sigma)
    assert np.all(G[SPECIES[0]] == 0.5)                  # legacy hit, same dir

    for spec in ("v142p", "v142"):
        monkeypatch.setenv("MCEQ3D_DB", spec)
        new = Bare(spec)
        with pytest.raises(Boom):                        # not served: recompute
            new.geomag_response(RC, 1.0, cache_dir=str(legacy_dir), sigma_lnr=sigma)
        # even if the OLD file is copied into the new cache directory under the
        # old name, the new key does not match it
        os.makedirs(db.CACHE_ROOT, exist_ok=True)
        np.savez(os.path.join(db.CACHE_ROOT, gs_cache_name(old._gs_tag, 1.0, RC, sigma)),
                 e=e, **fake)
        with pytest.raises(Boom):
            new.geomag_response(RC, 1.0, cache_dir=str(legacy_dir), sigma_lnr=sigma)
        assert not os.path.exists(legacy_dir / gs_cache_name(new._gs_tag, 1.0, RC, sigma))


@need_new
def test_geomagnetic_maps_not_rekeyed_but_relocated(tmp_path, monkeypatch):
    """finerc_/prodfam_/rc_/conewidths_ files do not depend on the database: same
    name for any database; a non-legacy database copies (not moves) them from the
    requested directory to mceq3d_db.cache_dir() and the legacy one leaves the
    requested directory untouched."""
    monkeypatch.setattr(db, "CACHE_ROOT", str(tmp_path / "new"))
    src = tmp_path / "old"
    src.mkdir()
    (src / "finerc_0123456789abcdef.npz").write_bytes(b"map")
    monkeypatch.setenv("MCEQ3D_DB", "v140")
    assert db.indep_path(str(src), "finerc_0123456789abcdef.npz") == str(
        src / "finerc_0123456789abcdef.npz")
    assert db.cache_dir(str(src)) == str(src)
    monkeypatch.setenv("MCEQ3D_DB", "v142p")
    p = db.indep_path(str(src), "finerc_0123456789abcdef.npz")
    assert os.path.dirname(p) == str(tmp_path / "new")
    assert os.path.basename(p) == "finerc_0123456789abcdef.npz"   # name unchanged
    assert open(p, "rb").read() == b"map"
    assert (src / "finerc_0123456789abcdef.npz").exists()         # copied, not moved


def test_every_mceqrun_goes_through_the_switch():
    """No module of tools/mceq3d may construct MCEqRun directly (that would
    bypass the database selection and the cache identity)."""
    bad = []
    for f in sorted(glob.glob(os.path.join(HERE, "*.py"))):
        name = os.path.basename(f)
        if name in ("mceq3d_db.py", "test_mceq3d_db.py"):
            continue
        for i, line in enumerate(open(f), 1):
            code = line.split("#")[0]
            if re.search(r"(^|[^\w`.])MCEqRun\(", code) or re.search(
                    r"from MCEq\.core import MCEqRun", code):
                bad.append(f"{name}:{i}: {line.strip()}")
    assert not bad, bad


@need_new
def test_channel_fractions_cache_is_keyed(tmp_path, monkeypatch):
    """channel_fractions (MCEq-derived, static name) must not read the legacy
    channel_fractions.npz, built with the shipped v1.4.0 file, for another database."""
    import kinematic_kernel as kk

    monkeypatch.setattr(db, "CACHE_ROOT", str(tmp_path))
    monkeypatch.setenv("MCEQ3D_DB", "v140")
    assert kk.channel_fractions_cache() == "channel_fractions.npz"
    names = {}
    for spec in ("v142", "v142p"):
        monkeypatch.setenv("MCEQ3D_DB", spec)
        p = kk.channel_fractions_cache()
        assert os.path.dirname(p) == str(tmp_path)
        assert db.resolve(spec).tag in os.path.basename(p)
        names[spec] = p
    assert names["v142"] != names["v142p"]
