"""Offline tests for the kernel splice utility."""

import numpy as np

from splice_kernels import splice


def _write(path, proj_energies, fill):
    xl_edges = np.logspace(-3, 0, 11)
    th_edges = np.linspace(0.0, 40.0, 21)
    n = (len(proj_energies), len(xl_edges) - 1, len(th_edges) - 1)
    np.savez(
        path,
        kernel=np.full(n, fill, dtype=float),
        marginal=np.full((n[0], n[1]), fill, dtype=float),
        xl_edges=xl_edges,
        theta_edges=th_edges,
        proj_energies=np.asarray(proj_energies, float),
    )


def test_splice_selects_by_transition(tmp_path):
    low = tmp_path / "low.npz"
    high = tmp_path / "high.npz"
    out = tmp_path / "spliced.npz"
    _write(low, [4.0, 10.0, 30.0, 80.0], fill=1.0)  # low model, value 1
    _write(high, [50.0, 80.0, 200.0, 1000.0], fill=2.0)  # high model, value 2

    splice(str(low), str(high), transition_gev=65.0, out_path=str(out))
    d = np.load(out)
    proj = d["proj_energies"]
    # below 65 from low (4,10,30), at/above 65 from high (80,200,1000)
    assert np.allclose(proj, [4.0, 10.0, 30.0, 80.0, 200.0, 1000.0])
    # sorted ascending
    assert np.all(np.diff(proj) > 0)
    # values: low energies tagged 1, high energies tagged 2
    k = d["kernel"]
    assert np.allclose(k[proj < 65].ravel(), 1.0)
    assert np.allclose(k[proj >= 65].ravel(), 2.0)


def test_splice_preserves_axes(tmp_path):
    low = tmp_path / "low.npz"
    high = tmp_path / "high.npz"
    out = tmp_path / "spliced.npz"
    _write(low, [4.0, 50.0], 1.0)
    _write(high, [80.0, 1000.0], 2.0)
    splice(str(low), str(high), 65.0, str(out))
    d = np.load(out)
    assert "theta_edges" in d
    assert d["kernel"].shape[0] == d["proj_energies"].shape[0]
