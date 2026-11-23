"""Tests of the G2.3 gate script on a synthetic reference (no Monte Carlo data, no
solver output)."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c2_stats as st  # noqa: E402
import g2_3 as G  # noqa: E402
import null_count_c2 as N  # noqa: E402

BAND = 0.1 * 10.0 ** (np.arange(11) / 5.0)
COSZ = np.round(np.linspace(0.0, 1.0, 11), 10)
AZ = np.linspace(0.0, 360.0, 13)
CAPS = np.array([2.5, 5.0, 7.5, 10.0])
NRUN = 20


def truth():
    """Flux with no cutoff (4, 10, 10) and cutoff factor (4, 10, 10, 12)."""
    sp = np.array([1.0, 0.8, 2.0, 1.7])[:, None, None]
    phi_0 = sp * np.logspace(0, -1, 10)[None, :, None] * np.linspace(1.5, 1.0, 10)
    az = st.az_centres(12)
    amp = np.linspace(0.3, 0.02, 10)[None, :, None, None]
    k = (0.4 + 0.06 * np.arange(10))[None, :, None, None] * (1 - amp * np.sin(az))
    k = k * np.linspace(0.9, 1.0, 10)[None, None, :, None] * np.ones((4, 1, 1, 1))
    return phi_0, k


def write_inputs(d, seed=1, nev=4000.0):
    """Reference with Poisson noise per run and cap, its thresholds, the true cells."""
    rng = np.random.default_rng(seed)
    phi_0, k = truth()
    scale = nev / phi_0.max()
    shape = (NRUN, 4) + k.shape
    ref = {"s_runs": rng.poisson(phi_0[..., None] * k * scale, shape) / scale,
           "n_runs": rng.poisson(phi_0[..., None] * scale * np.ones(12), shape) / scale,
           "t_runs": rng.poisson(phi_0[..., None] * 1.1 * scale * np.ones(12), shape)
           / scale}
    ref["us_runs"], ref["un_runs"] = ref["s_runs"][::-1], ref["n_runs"][::-1]
    axes = {"e_band_edges_GeV": BAND, "azimuth_edges_deg": AZ, "cap_theta_deg": CAPS,
            "species": np.array(G.SPECIES)}
    np.savez(os.path.join(d, "ref.npz"), cosz_edges_down=COSZ, **ref, **axes)
    N.main([os.path.join(d, "ref.npz"), "--outdir", os.path.join(d, "null"),
            "--nsamp", "300"])
    cells = {"phi_cut": np.broadcast_to(phi_0[..., None] * k, (4,) + k.shape),
             "phi_0": np.broadcast_to(phi_0, (4,) + phi_0.shape),
             "phi_straight_0": 1.1 * phi_0, "cosz_edges": COSZ}
    cells["phi_cut_up"] = cells["phi_cut"][:, :, :, 8:, :]
    cells["phi_0_up"] = cells["phi_0"][:, :, :, 8:]
    return cells, axes


def run(d, cells, axes, name):
    np.savez(os.path.join(d, name + ".npz"), **cells, **axes)
    rc = G.main(["--ref", os.path.join(d, "ref.npz"), "--solver",
                 os.path.join(d, name + ".npz"), "--null",
                 os.path.join(d, "null", "null_count_c2.json"), "--outdir", d,
                 "--name", name])
    return rc, json.load(open(os.path.join(d, name + ".json")))


def test_exact_and_wrong_solvers(tmp_path):
    d = str(tmp_path)
    cells, axes = write_inputs(d)
    rc, res = run(d, cells, axes, "exact")
    # the truth is a typical draw of the null model: far above a wrong solver's counts
    assert res["items"] == {"a": 4800, "b": 100, "c": 10}
    assert res["counts"]["a"] >= 0.93 * 4800 and res["counts"]["c"] >= 9
    assert rc == (0 if res["pass"] else 1)
    # the threshold is that of the half-against-half null model; the first rule is
    # carried next to it and decides nothing
    null = json.load(open(os.path.join(d, "null", "null_count_c2.json")))
    for p in "abc":
        part = null["parts"][p]
        assert part["threshold"] == part["split_n5"] == res["thresholds"][p]
        assert part["threshold_first_rule"] == max(part["gauss_n5"],
                                                   part["resample_n5"])
        assert res["thresholds_first_rule"][p] == part["threshold_first_rule"]
        assert res["passed_first_rule"][p] == (
            res["counts"][p] >= part["threshold_first_rule"])
    assert res["pass_first_rule"] == all(res["passed_first_rule"].values())
    # no calibration at the reference's own event counts was given: flagged
    assert 0 <= null["a_floor_share"] <= 1 and res["a_calibration_flag"]
    assert res["a_calibration"] is None and not res["a_failed_within_delta"]
    assert not res["e_trigger"]
    assert abs(np.mean(res["d_mean_r"]) - 1) < 0.01
    assert abs(np.mean(res["f"]["mean_q"]) - 1) < 0.01
    assert res["h"]["counted"] == 4 * 10 * 2 * 12 and res["g"]["a"]["counted"] == 4800
    with pytest.raises(SystemExit):     # the result files are never overwritten
        run(d, cells, axes, "exact")
    # part (a) alone (used for parts f and h) is part (a) of the full evaluation
    r = np.load(os.path.join(d, "ref.npz"))
    args = (r["s_runs"][:, 3], r["n_runs"][:, 3], cells["phi_cut"][3],
            cells["phi_0"][3])
    one, full = G.cutoff_factor(*args), G.evaluate(*args)["a"]
    assert all(np.array_equal(one[k], full[k], equal_nan=True) for k in full)

    az = st.az_centres(12)
    high = dict(cells, phi_cut=cells["phi_cut"] * 1.08)
    rc, bad = run(d, high, axes, "high")
    assert rc == 1 and not bad["passed"]["a"] and bad["e_trigger"]
    assert bad["counts"]["a"] < res["counts"]["a"] - 1000

    skew = dict(cells, phi_cut=cells["phi_cut"] * (1 + 0.1 * np.cos(az)))
    rc, bad = run(d, skew, axes, "skew")
    assert rc == 1 and not bad["passed"]["b"]

    tilt = np.linspace(1.08, 1.0, 10)[:, None]
    rc, bad = run(d, dict(cells, phi_cut=cells["phi_cut"] * tilt), axes, "tilt")
    assert rc == 1 and not bad["passed"]["c"]


def test_calibration_at_the_reference(tmp_path):
    """calib_null_c2.py (mode ref) -> --calib -> null_count_c2.py -> g2_3.py: the
    calibration is carried next to the threshold and changes nothing in it."""
    d = str(tmp_path)
    cells, axes = write_inputs(d)
    ref = os.path.join(d, "ref.npz")
    outs = [os.path.join(d, "calib_s%d.json" % seed) for seed in (5, 6)]
    script = os.path.join(HERE, "calib_null_c2.py")
    for seed, out in zip((5, 6), outs):
        subprocess.run([sys.executable, script, "nested", "0", "0", "2", str(seed), out,
                        ref, "ref"], check=True, stdout=subprocess.DEVNULL)
    cal = os.path.join(d, "calib.json")
    subprocess.run([sys.executable, script, "--calib", cal] + outs, check=True,
                   stdout=subprocess.DEVNULL)
    c = json.load(open(cal))
    assert c["references"] == 4 and c["seeds"] == [5, 6] and c["delta_a"] >= 0
    assert c["rate_after"] <= 0.05 <= 1 and c["cap_index"] == 3
    # the same two runs twice, or a second file of the same name, are refused
    assert subprocess.run([sys.executable, script, "--calib",
                           os.path.join(d, "twice.json"), outs[0], outs[0]],
                          capture_output=True).returncode != 0
    assert subprocess.run([sys.executable, script, "--calib", cal] + outs,
                          capture_output=True).returncode != 0

    N.main([ref, "--outdir", os.path.join(d, "null_cal"), "--nsamp", "300", "--calib",
            cal])
    plain = json.load(open(os.path.join(d, "null", "null_count_c2.json")))
    null = json.load(open(os.path.join(d, "null_cal", "null_count_c2.json")))
    assert null["parts"] == plain["parts"]
    a = null["a_calibration"]
    assert a["delta_a"] == c["delta_a"] and a["references"] == 4
    assert a["threshold_less_delta"] == null["parts"]["a"]["threshold"] - c["delta_a"]
    assert null["a_calibration_flag"] == (c["rate_split"] > N.RATE_FLAG)

    np.savez(os.path.join(d, "exact.npz"), **cells, **axes)
    rc = G.main(["--ref", ref, "--solver", os.path.join(d, "exact.npz"), "--null",
                 os.path.join(d, "null_cal", "null_count_c2.json"), "--outdir", d,
                 "--name", "exact"])
    res = json.load(open(os.path.join(d, "exact.json")))
    assert rc == (0 if res["pass"] else 1) and res["a_calibration"] == a
    low = res["thresholds"]["a"] - a["delta_a"]
    assert res["a_failed_within_delta"] == (low <= res["counts"]["a"]
                                            < res["thresholds"]["a"])
    assert "own event counts" in open(os.path.join(d, "exact.md")).read()

    # a calibration made on another reference is refused
    c["sha256"] = "0" * 64
    json.dump(c, open(os.path.join(d, "other.json"), "w"))
    with pytest.raises(SystemExit):
        N.main([ref, "--outdir", os.path.join(d, "null_other"), "--nsamp", "300",
                "--calib", os.path.join(d, "other.json")])


def test_input_errors(tmp_path):
    d = str(tmp_path)
    cells, axes = write_inputs(d)
    with pytest.raises(SystemExit) as e:    # cells of another binning
        run(d, dict(cells, cosz_edges=COSZ * 0.99), axes, "edges")
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:    # thresholds of another reference
        null = os.path.join(d, "null", "null_count_c2.json")
        j = json.load(open(null))
        j["sha256"] = "0" * 64
        json.dump(j, open(null, "w"))
        run(d, cells, axes, "sha")
    assert e.value.code == 2
