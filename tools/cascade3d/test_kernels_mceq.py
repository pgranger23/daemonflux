"""Tests of ``kernels_mceq`` (WP 0.1).  Run: ``$PY -m pytest test_kernels_mceq.py -q``.

Each test builds MCEq in memory (about 10 s per population, 1.3 GB), so the REF run
and kernel dictionary are shared by all tests.
"""

import importlib.util  # noqa: F401
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernels_mceq as K  # noqa: E402


@pytest.fixture(scope="module")
def ref_run():
    return K.make_mceq("REF")


@pytest.fixture(scope="module")
def ref(ref_run):
    return K.build("REF", run=ref_run)


def _rel(a, b):
    diff = (a - b).tocoo()
    num = np.abs(diff.data).max() if diff.nnz else 0.0
    return num / np.abs(a.tocoo().data).max()


def test_roundtrip_without_pickle(ref, tmp_path):
    path = K.save(ref, "REF", path=str(tmp_path / "k.npz"))
    with np.load(path, allow_pickle=False) as z:
        for key in ("e_grid", "e_bins", "e_widths", "species", "pdg", "helicity",
                    "mass", "charge", "ctau", "is_hadron", "is_lepton", "is_stable",
                    "int_parent", "int_child", "int_blocks", "dec_parent",
                    "dec_child", "dec_blocks", "phi0", "meta_json"):
            assert key in z.files
            assert z[key].dtype != object
    d = K.load("REF", path=path)
    assert d["meta"]["population"] == "REF"
    assert np.array_equal(d["int_blocks"], ref["int_blocks"])


def test_rebuild_equals_mceq(ref, ref_run):
    im, dm = K.rebuild(ref)
    assert _rel(ref_run.int_m, im) <= 1e-12
    assert _rel(ref_run.dec_m, dm) <= 1e-12


def test_loss_off_equals_mceq_without_loss(ref):
    run0 = K.make_mceq("REF", energy_loss=False)
    im0, _ = K.rebuild(ref, loss=False)
    assert _rel(run0.int_m, im0) <= 1e-12
    # and the loss really is a separate, non-trivial operator on the muons
    assert ref["loss_species"].size > 0
    mu = K.species_index(ref, "mu+_l")
    assert mu in ref["loss_species"]
    assert not np.any((ref["int_parent"] == mu) & (ref["int_child"] == mu))


def test_blocks_only_nonzero_and_dense_shape(ref):
    nE = ref["e_grid"].size
    assert ref["int_blocks"].shape[1:] == (nE, nE)
    assert np.all(np.abs(ref["int_blocks"]).max(axis=(1, 2)) > 0)
    assert np.all(np.abs(ref["dec_blocks"]).max(axis=(1, 2)) > 0)
    pairs = set(zip(ref["int_parent"].tolist(), ref["int_child"].tolist()))
    assert len(pairs) == ref["int_parent"].size


def test_grid_conventions(ref):
    b, c, w = ref["e_bins"], ref["e_grid"], ref["e_widths"]
    assert np.allclose(c, np.sqrt(b[:-1] * b[1:]))
    assert np.allclose(w, np.diff(b))
    assert c.size == 121 and np.isclose(b[0], 10 ** -1.1, rtol=1e-3)


def test_yield_and_decay_normalisation(ref):
    w, nE = ref["e_widths"], ref["e_grid"].size
    pip = K.species_index(ref, "pi+")
    dl = ref["inv_dec_len"][pip]
    k = 60
    tot = 0.0
    for h in ("mu+_l", "mu+_r", "mu+"):
        i = K.species_index(ref, h)
        col = K.get_block(ref, "dec", pip, i)[:, k] / dl[k]
        tot += (col * w).sum() / w[k]
    assert abs(tot - 1.0) < 2e-3  # pi+ -> mu+ nu, all helicities
    assert K.get_block(ref, "dec", pip, pip)[k, k] == pytest.approx(-dl[k])
    # proton: -1/lambda on the diagonal plus same-bin regeneration of a few percent
    p = K.species_index(ref, "p+")
    lam = ref["inv_lambda_int"][p]
    diag = np.diag(K.get_block(ref, "int", p, p))[k] / lam[k]
    assert -1.0 < diag < -0.8
    # energy: hadronic secondaries carry 0.5-1 of the parent's total energy (leptons
    # from resonance chains are excluded: binning makes their sum overshoot by <1%)
    mass, c = ref["mass"], ref["e_grid"]
    k = 100
    etot = 0.0
    for ch in range(ref["species"].size):
        if ref["is_tracking"][ch] or not ref["is_hadron"][ch]:
            continue
        Y = K.get_block(ref, "int", p, ch, loss=False)[:, k] / lam[k]
        if ch == p:
            Y[k] += 1.0
        etot += (Y * w * (c + mass[ch])).sum() / w[k]
    assert 0.5 < etot / (c[k] + mass[p]) < 1.0


def test_species_flags(ref):
    names = list(ref["species"])
    assert names[:3] == ["antinue", "nue", "mu+_l"]
    for n in ("mu+_l", "mu+_r", "mu+", "mu-_l", "mu-_r", "mu-"):
        assert n in names
    mu = K.species_index(ref, "mu+_l")
    assert ref["helicity"][mu] == -1 and ref["charge"][mu] == 1
    assert ref["is_stable"][K.species_index(ref, "p+")]
    assert ref["is_tracking"].sum() == 50
    assert ref["mix_idx"][K.species_index(ref, "pi+")] == 20
    assert ref["species"].size == 82


def test_end_to_end_euler_march(ref, ref_run):
    """The rebuilt matrices, MCEq's path and phi0 reproduce MCEq's own solve."""
    ref_run.set_theta_deg(0.0)
    ref_run.solve()
    nsteps, dX, rho_inv, _ = ref_run.integration_path
    im, dm = K.rebuild(ref)
    phi = ref["phi0"].copy()
    for s in range(nsteps):
        phi += (im.dot(phi) + dm.dot(rho_inv[s] * phi)) * dX[s]
    sol = np.asarray(ref_run._solution)
    assert np.abs(phi - sol).max() / np.abs(sol).max() < 1e-12
    nE = ref["e_grid"].size
    i = K.species_index(ref, "numu")
    got = ref_run.get_solution("numu", mag=0)
    assert np.allclose(phi[i * nE:(i + 1) * nE], got, rtol=1e-12, atol=0)


def test_populations_match_mceq3d_flux_constants():
    try:
        import mceq3d_flux as mf
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"mceq3d_flux not importable: {exc}")
    assert K.POPULATIONS["PHYS"] == mf.PHYS_POPULATION
    assert K.POPULATIONS["REF"] == mf.REF_POPULATION


def test_phys_floor_and_grid():
    run = K.make_mceq("PHYS")
    d = K.build("PHYS", run=run)
    meta = json.loads(str(d["meta_json"]))
    assert meta["config"]["e_min"] == 0.03
    assert meta["config"]["minimal_primary_energy"] == 1.9
    nE = d["e_grid"].size
    p = K.species_index(d, "p+")
    sl = d["phi0"][p * nE:(p + 1) * nE]
    first = int(np.flatnonzero(sl)[0])
    assert first == int(d["phi0_min_idx"])
    mn = d["mass"][K.species_index(d, "n0")]
    e_tot = d["e_grid"][first] + 0.5 * (d["mass"][p] + mn)
    assert abs(e_tot - 1.9) < abs(d["e_tot_nucleon"][first - 1] - 1.9) + 1e-12
    im, dm = K.rebuild(d)
    assert _rel(run.int_m, im) <= 1e-12 and _rel(run.dec_m, dm) <= 1e-12
