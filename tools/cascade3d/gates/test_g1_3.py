"""Tests of the G1.3 and G1.4 gate scripts on synthetic references (no Monte Carlo).

``make_ref`` writes a file with the structure of ``c1/mcref2/mcref2_fieldfree.npz``
(brief ``c1/G13_BRIEF.md``): the reference is the solver's cells divided by a known
distortion D, with independent noise per run, so that r = D (up to the noise) and the
script must recover D and give the verdict the gate text implies.
"""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g1_3 as G  # noqa: E402
import g1_4 as G4  # noqa: E402

BAND = 0.1 * 10.0 ** (np.arange(11) / 5.0)
COSZ = np.round(np.linspace(0.0, 1.0, 11), 10)
CC = 0.5 * (COSZ[1:] + COSZ[:-1])
EC = np.sqrt(BAND[1:] * BAND[:-1])


def analytic_solver():
    """(4, 10, 10) smooth positive cells."""
    amp = np.array([1.0, 0.8, 2.0, 1.9])
    return (amp[:, None, None] * EC[None, :, None] ** -2.0
            * (0.15 + CC[None, None, :]) ** -0.6)


def make_ref(sol, D, sig_run=0.03, seed=1, n_runs=10, t_ratio=None, zero=None,
             err_scale=1.0, run_common=0.0):
    """Reference dict: s_true = sol / D; run k: s_true (1 + sig_run eps + common_k)."""
    rng = np.random.default_rng(seed)
    s_true = sol / D
    caps = np.array([2.5, 5.0, 7.5, 10.0])
    out = {}
    for name in ("s", "t", "s_cross", "t_cross"):
        base = s_true if name in ("s", "s_cross") else (
            s_true / (t_ratio if t_ratio is not None else 0.97))
        per_cap = np.stack([base * (1.0 + 0.001 * (3 - c)) for c in range(4)])
        runs = []
        for _k in range(n_runs):
            eps = rng.standard_normal(per_cap.shape)
            com = run_common * rng.standard_normal((4, 1, 1, per_cap.shape[-1]))
            runs.append(per_cap * (1.0 + sig_run * eps + com))
        runs = np.array(runs)
        if zero is not None and name == "s":
            runs[(slice(None), 3) + zero] = 0.0
        m = runs.mean(0)
        out[name] = m
        out[name + "_runs"] = runs
        e = per_cap * sig_run / np.sqrt(n_runs) * err_scale
        out[name + "_err_task"] = e
        out[name + "_err_sq"] = e / 3.4
        out[name + "_err_seed"] = runs.std(0, ddof=1) / np.sqrt(n_runs)
    out.update(species=np.array(list(G.SPECIES)), e_band_edges_GeV=BAND,
               cosz_edges=COSZ, cap_theta_deg=caps,
               cap_area_cm2=2 * np.pi * 6.371e8 ** 2 * (1 - np.cos(np.radians(caps))),
               n_primaries=np.array(10 ** 7), n_primaries_per_run=np.full(n_runs,
                                                                          10 ** 6),
               meta_normalisation=np.array("synthetic"),
               meta_cosz=np.array("synthetic"))
    return out


def as_ref(d):
    r = dict(d)
    r["meta"] = {}
    return r


def test_null_passes():
    sol = analytic_solver()
    a = G.part_a(sol, as_ref(make_ref(sol, np.ones_like(sol))))
    assert a["pass"] and a["count_ok"] and a["rows_ok"].all()
    assert not a["b_any"]
    assert np.max(np.abs(a["rbar"] - 1)) < 0.02


def test_constant_factor_is_normalised_out_and_triggers_b():
    sol = analytic_solver()
    a = G.part_a(sol, as_ref(make_ref(sol, np.full_like(sol, 1.07))))
    assert a["pass"]
    assert a["b_any"] and a["b_trigger"].all()
    assert np.allclose(a["rbar"], 1.07, atol=0.02)


def test_zenith_tilt_is_recovered_and_fails():
    sol = analytic_solver()
    D = np.broadcast_to(1.0 + 0.04 * (CC - 0.5), sol.shape).copy()
    a = G.part_a(sol, as_ref(make_ref(sol, D, sig_run=0.005)))
    expect = D / D.mean(axis=-1, keepdims=True)
    assert np.max(np.abs(a["rn"] - expect)) < 0.01
    assert not a["pass"]
    assert not a["rows_ok"][0] and not a["rows_ok"][-1]
    assert np.allclose(a["rows"], expect[0, 0], atol=0.003)


def test_horizon_bump_fails_on_the_row_not_the_count():
    sol = analytic_solver()
    D = np.ones_like(sol)
    D[:, :3, 0] = 1.10                       # cos Z 0-0.1, bands below 0.4 GeV
    a = G.part_a(sol, as_ref(make_ref(sol, D, sig_run=0.005)))
    assert a["count_ok"] and a["n_inside"] >= 380
    assert not a["rows_ok"][0] and a["rows_ok"][1:].all()
    assert not a["pass"]
    assert a["rows"][0] - 1 > 0.04


def test_zero_cell_counted_outside():
    sol = analytic_solver()
    a = G.part_a(sol, as_ref(make_ref(sol, np.ones_like(sol), zero=(1, 4, 5))))
    assert a["zero"][1, 4, 5] and not a["inside"][1, 4, 5]
    assert a["n_inside"] <= 399
    assert np.isfinite(a["rbar"]).all() and np.isfinite(a["rows"]).all()


def test_count_fails_with_scatter_beyond_errors_and_passes_with_honest_errors():
    sol = analytic_solver()
    D = 1.0 + 0.04 * np.random.default_rng(5).standard_normal(sol.shape)
    small = G.part_a(sol, as_ref(make_ref(sol, D, sig_run=0.0003)))
    assert not small["count_ok"] and not small["pass"]
    # the same scatter with a declared sigma of ~6%: inside
    big = G.part_a(sol, as_ref(make_ref(sol, D, sig_run=0.0003, err_scale=650)))
    assert big["count_ok"]
    assert np.median(big["tol"]) > 0.1


def test_threshold_is_95_percent_of_cells():
    sol = analytic_solver()
    for n_out, ok in ((20, True), (21, False)):
        D = np.ones_like(sol)
        # at most one outlier per (species, band), so the others stay inside
        rows = np.random.default_rng(3).permutation(40)[:n_out]
        for k, rw in enumerate(rows):
            D[rw // 10, rw % 10, k % 10] = 1.15 if k % 2 else 0.85
        a = G.part_a(sol, as_ref(make_ref(sol, D, sig_run=0.0003)))
        assert a["n_needed"] == 380
        assert a["count_ok"] is ok, (n_out, a["n_inside"])


def test_count_min_replaces_the_95_percent():
    """Closing cycle of C1: the count needed is given, the original one is kept."""
    sol = analytic_solver()
    for n_out, ok in ((27, True), (28, False)):
        D = np.ones_like(sol)
        rows = np.random.default_rng(3).permutation(40)[:n_out]
        for k, rw in enumerate(rows):
            D[rw // 10, rw % 10, k % 10] = 1.15 if k % 2 else 0.85
        a = G.part_a(sol, as_ref(make_ref(sol, D, sig_run=0.0003)), n_min=373)
        assert a["n_inside"] == 400 - n_out
        assert a["n_needed"] == 373 and a["n_needed_original"] == 380
        assert a["count_ok"] is ok and a["count_ok_original"] is False
    D = np.ones_like(sol)
    a = G.part_a(sol, as_ref(make_ref(sol, D, sig_run=0.0003)), n_min=373)
    assert a["count_ok"] and a["count_ok_original"]


def test_row_sigma_from_runs_includes_correlation():
    sol = analytic_solver()
    ind = G.part_a(sol, as_ref(make_ref(sol, np.ones_like(sol), sig_run=0.01)))
    cor = G.part_a(sol, as_ref(make_ref(sol, np.ones_like(sol), sig_run=0.01,
                                        run_common=0.01)))
    # independent noise: sigma of a run's row mean ~ 0.01 / sqrt(20)
    assert 0.5 < np.median(ind["rows_sigma"]) / (0.01 / np.sqrt(20) / np.sqrt(10)) < 1.6
    assert np.median(cor["rows_sigma"]) > 2 * np.median(ind["rows_sigma"])


def test_part_d_recovers_q():
    sol = analytic_solver()
    straight = sol / (1.0 + 0.1 * (1 - CC))[None, None, :]
    Dq = np.broadcast_to(1.0 + 0.05 * CC, sol.shape)
    # s/t must equal (sol/straight) / Dq
    t_ratio = (sol / straight) / Dq
    ref = as_ref(make_ref(sol, np.ones_like(sol), sig_run=0.002, t_ratio=t_ratio))
    d = G.part_d(sol, straight, ref)
    assert np.max(np.abs(d["q"] - Dq)) < 0.005


def test_diagnostics_see_cosz_power_and_shift():
    sol = analytic_solver()
    D = np.broadcast_to(CC, sol.shape).copy()
    ref = as_ref(make_ref(sol, D, sig_run=0.002))
    a = G.part_a(sol, ref)
    dg = G.diagnostics(sol, ref, a)
    assert abs(dg["cosz_power"] - 1.0) < 0.02
    # reference made from the solver shifted by one cos Z bin
    shifted = np.roll(sol, -1, axis=2)
    ref2 = as_ref(make_ref(sol, sol / shifted, sig_run=0.002))
    a2 = G.part_a(sol, ref2)
    dg2 = G.diagnostics(sol, ref2, a2)
    v = dg2["shifts"]["cosz+1"]
    assert v["chi2_shifted"] < 0.1 * v["chi2_unshifted"]


def _save(path, d):
    np.savez(path, **d)


def _solver_file(path, cells):
    np.savez(path, cell_band_edges=BAND, cell_cosz_edges=COSZ,
             cost_json=np.array('{"ks": "synthetic", "total_cpu_s": 60, '
                                '"peak_rss_GB": 1.0, "r_earth": 6371, '
                                '"straight": false}'),
             **{f"cell_{q}": cells[i] for i, q in enumerate(G.SPECIES)})


def test_cli_end_to_end(tmp_path):
    sol = analytic_solver()
    _solver_file(tmp_path / "pool.npz", sol)
    _solver_file(tmp_path / "fresh.npz", sol * 1.001)
    _solver_file(tmp_path / "straight.npz", sol * 0.98)
    _save(tmp_path / "ok.npz", make_ref(sol, np.ones_like(sol)))
    D = np.broadcast_to(1.0 + 0.04 * (CC - 0.5), sol.shape)
    _save(tmp_path / "bad.npz", make_ref(sol, D, sig_run=0.005))
    common = ["--pool", str(tmp_path / "pool.npz"), "--fresh",
              str(tmp_path / "fresh.npz"),
              "--straight", str(tmp_path / "straight.npz"), "--outdir", str(tmp_path)]
    assert G.main(["--ref", str(tmp_path / "ok.npz"), "--name", "ok"] + common) == 0
    assert G.main(["--ref", str(tmp_path / "bad.npz"), "--name", "bad"] + common) == 1
    txt = (tmp_path / "bad.md").read_text()
    assert "**FAIL**" in txt and "meta_normalisation" in txt
    assert (tmp_path / "ok.json").exists()


def test_input_checks(tmp_path):
    sol = analytic_solver()
    _solver_file(tmp_path / "pool.npz", sol)
    d = make_ref(sol, np.ones_like(sol))
    d["species"] = np.array(["numu", "antinumu", "nue", "antinue"])
    _save(tmp_path / "sp.npz", d)
    assert G.main(["--ref", str(tmp_path / "sp.npz"), "--pool",
                   str(tmp_path / "pool.npz"),
                   "--outdir", str(tmp_path)]) == 2
    d = make_ref(sol, np.ones_like(sol))
    d["cap_theta_deg"] = np.array([2.5, 5.0, 10.0, 7.5])
    _save(tmp_path / "cap.npz", d)
    assert G.main(["--ref", str(tmp_path / "cap.npz"), "--pool",
                   str(tmp_path / "pool.npz"),
                   "--outdir", str(tmp_path)]) == 2
    d = make_ref(sol, np.ones_like(sol))
    d["cosz_edges"] = np.linspace(-1, 0, 11)
    _save(tmp_path / "cz.npz", d)
    assert G.main(["--ref", str(tmp_path / "cz.npz"), "--pool",
                   str(tmp_path / "pool.npz"),
                   "--outdir", str(tmp_path)]) == 2


def test_g1_4_verdicts(tmp_path):
    sol = analytic_solver()
    _solver_file(tmp_path / "h0.npz", sol)
    b = sol.copy()
    b[2, 7, 0] *= 1.0049
    _solver_file(tmp_path / "h1.npz", b)
    assert G4.main(["--h0", str(tmp_path / "h0.npz"), "--h1", str(tmp_path / "h1.npz"),
                    "--outdir", str(tmp_path)]) == 0
    b[2, 7, 0] = sol[2, 7, 0] * 1.0052
    _solver_file(tmp_path / "h1.npz", b)
    assert G4.main(["--h0", str(tmp_path / "h0.npz"), "--h1", str(tmp_path / "h1.npz"),
                    "--outdir", str(tmp_path)]) == 1
    r = G4.evaluate(sol, b)
    assert r["n_inside"] == 399 and not r["inside"][2, 7, 0]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
