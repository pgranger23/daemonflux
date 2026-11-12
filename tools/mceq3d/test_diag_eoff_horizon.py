"""Unit tests of the E_off horizon gate (:mod:`diag_eoff_horizon`) on synthetic tallies (no MC files, < 1 s)."""

from __future__ import annotations

import numpy as np
import pytest

import diag_eoff_horizon as dh


def test_e5_from20_merges_groups_of_four():
    x = np.arange(2 * 40 * 3, dtype=float).reshape(2, 40, 3)
    y = dh.e5_from20(x, 1)
    assert y.shape == (2, 10, 3)
    assert np.allclose(y[:, 0], x[:, :4].sum(1))
    assert y.sum() == pytest.approx(x.sum())


def test_replicates_are_unbiased_for_uniform_groups():
    rng = np.random.default_rng(1)
    G, K = 20, 3
    S = np.ones((G, K, 5)) * rng.uniform(1, 2, (1, K, 1))
    M = np.full((G, K), 100.0)
    r = dh.replicates(S, M)
    assert r.shape == (G, 5)
    assert np.allclose(r, S.sum((0, 1)))


def _synthetic(scale_row0=1.0):
    rng = np.random.default_rng(2)
    G, K = 20, 2
    Bc = rng.uniform(1.0, 2.0, (4, 10, 10))
    Bk = Bc[:, :, 6:10].sum(-1)
    eoff = rng.uniform(1.0, 2.5, (4, 10, 10))
    kappa = 0.7
    s = kappa * Bc * eoff
    s[:, :, 0] *= scale_row0
    noise = rng.normal(1.0, 0.01, (G, K, 4, 10, 10))
    A = {"s": s[None, None] / (G * K) * noise, "mu_on": np.zeros((G, K, 4, 10, 10)),
         "mu_cf": np.zeros((G, K, 4, 10, 10)), "t": (kappa * Bc)[None, None] / (G * K) * np.ones((G, K, 1, 1, 1))}
    return A, np.full((G, K), 50.0), Bc, Bk, eoff


def test_exact_shape_gives_zero_departure_and_passes():
    A, M, Bc, Bk, eoff = _synthetic()
    d, sig, ok, n = dh.horizon_test(A, M, Bc, Bk, eoff)
    assert np.abs(d[:5][:, :3]).max() < 0.01
    assert n == 15 and ok.all()
    assert (sig > 0).all()


def test_horizon_deficit_is_detected():
    A, M, Bc, Bk, eoff = _synthetic(scale_row0=0.85)
    d, sig, ok, n = dh.horizon_test(A, M, Bc, Bk, eoff)
    assert d[:5, 0] == pytest.approx(np.full(5, np.log(0.85)), abs=0.02)
    assert not ok[:, 0].any() and ok[:, 1:].all()
    assert n == 10


def test_muon_counterfactual_enters_with_the_right_sign():
    A, M, Bc, Bk, eoff = _synthetic()
    bump = A["s"] * 0.0
    bump[..., 0] = 0.1 * A["s"][..., 0]
    A2 = dict(A, s=A["s"] + bump, mu_on=bump)          # the muon bend's extra flux, removed by mu_on - mu_cf
    d, _, _, _ = dh.horizon_test(A2, M, Bc, Bk, eoff)
    assert np.abs(d[:5][:, :3]).max() < 0.01
