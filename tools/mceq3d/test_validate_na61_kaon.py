"""Offline tests for the NA61 K± parsing/weighting (no network, no chromo)."""

import os

import numpy as np

from validate_na61_kaon import _theta_center_rad, load_kaon, na61_ptmean


def test_theta_center_mrad():
    assert np.isclose(_theta_center_rad("0.0-20.0"), 0.010)
    assert np.isclose(_theta_center_rad("100.0-140.0"), 0.120)


def test_single_theta_gives_p_sin_theta():
    th = "80.0-120.0"  # center 0.1 rad
    data = {
        "Table23": {
            "reaction": "P C --> K+ X",
            "theta": th,
            "p": [{"low": 0.9, "high": 1.1}, {"low": 1.9, "high": 2.1}],
            "values": [{"value": 10.0}, {"value": 5.0}],
        }
    }
    p_edges = np.array([0.5, 1.5, 2.5])
    _, pt = na61_ptmean(data, ["Table23"], p_edges)
    assert np.isclose(pt[0], 1.0 * np.sin(0.1), atol=1e-3)
    assert np.isclose(pt[1], 2.0 * np.sin(0.1), atol=1e-3)


def test_two_theta_yield_weighting():
    data = {
        "A": {
            "reaction": "P C --> K+ X",
            "theta": "80.0-120.0",  # 0.1 rad
            "p": [{"low": 0.9, "high": 1.1}],
            "values": [{"value": 3.0}],
        },
        "B": {
            "reaction": "P C --> K+ X",
            "theta": "280.0-320.0",  # 0.3 rad
            "p": [{"low": 0.9, "high": 1.1}],
            "values": [{"value": 1.0}],
        },
    }
    p_edges = np.array([0.5, 1.5])
    _, pt = na61_ptmean(data, ["A", "B"], p_edges)
    expected = (3.0 * np.sin(0.1) + 1.0 * np.sin(0.3)) / 4.0
    assert np.isclose(pt[0], expected, atol=1e-3)


def test_real_kaon_tables_present_and_sane():
    # If the hepdata-cli-fetched YAML tables are present, <p_T> should rise with p
    # and sit at the kaon scale (heavier than pions).
    if not os.path.exists("na61_k/Table23.yaml"):
        return  # not fetched; skip silently
    data = load_kaon()
    p_edges = np.logspace(np.log10(0.8), np.log10(16.0), 12)
    _, pt = na61_ptmean(data, [f"Table{n}" for n in range(23, 31)], p_edges)
    good = np.isfinite(pt)
    assert pt[good][0] < pt[good][-1]  # rises with p
    assert 0.1 < pt[good][0] < 0.3
    assert 0.4 < pt[good][-1] < 0.7
