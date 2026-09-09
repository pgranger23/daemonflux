"""Offline tests for the NA61 parsing/weighting logic (no network, no chromo)."""

import numpy as np

from validate_na61 import _theta_center_rad, _yield_value, na61_ptmean


def test_theta_center():
    assert np.isclose(_theta_center_rad("0.0-20.0 Mrad"), 0.010)
    assert np.isclose(_theta_center_rad("100.0-140.0 Mrad"), 0.120)


def test_yield_value_parsing():
    assert _yield_value({"value": "10.5"}) == 10.5
    assert np.isnan(_yield_value({"value": "-"}))


def test_single_theta_gives_p_sin_theta():
    # One polar-angle table at theta=100 mrad: <p_T>(p) must equal p*sin(theta).
    th = "80.0-120.0 Mrad"  # center 0.1 rad
    data = {
        "Table2": {
            "reaction": "P C --> PI+ X",
            "theta": th,
            "values": [
                {"x": [{"low": "0.9", "high": "1.1"}], "y": [{"value": "10.0"}]},
                {"x": [{"low": "1.9", "high": "2.1"}], "y": [{"value": "5.0"}]},
            ],
        }
    }
    p_edges = np.array([0.5, 1.5, 2.5])
    centers, pt = na61_ptmean(data, ["Table2"], p_edges)
    assert np.isclose(pt[0], 1.0 * np.sin(0.1), atol=1e-3)
    assert np.isclose(pt[1], 2.0 * np.sin(0.1), atol=1e-3)


def test_two_theta_yield_weighting():
    # Same momentum, two angles: <p_T> is yield-weighted (yield = dsig/dp * dp).
    data = {
        "A": {
            "reaction": "P C --> PI+ X",
            "theta": "80.0-120.0 Mrad",  # 0.1 rad
            "values": [{"x": [{"low": "0.9", "high": "1.1"}], "y": [{"value": "3.0"}]}],
        },
        "B": {
            "reaction": "P C --> PI+ X",
            "theta": "280.0-320.0 Mrad",  # 0.3 rad
            "values": [{"x": [{"low": "0.9", "high": "1.1"}], "y": [{"value": "1.0"}]}],
        },
    }
    p_edges = np.array([0.5, 1.5])
    _, pt = na61_ptmean(data, ["A", "B"], p_edges)
    expected = (3.0 * np.sin(0.1) + 1.0 * np.sin(0.3)) / 4.0
    assert np.isclose(pt[0], expected, atol=1e-3)


def test_ptmean_rises_with_momentum_on_real_cache():
    # If the real HEPData cache is present, <p_T> should rise with momentum.
    import os
    import json

    if not os.path.exists("na61_886780_cache.json"):
        return  # offline / not fetched; skip silently
    data = json.load(open("na61_886780_cache.json"))
    p_edges = np.logspace(np.log10(0.3), np.log10(15.0), 12)
    _, pt = na61_ptmean(data, [f"Table{n}" for n in range(2, 12)], p_edges)
    good = np.isfinite(pt)
    assert pt[good][0] < pt[good][-1]  # rises with p
    assert 0.05 < pt[good][0] < 0.2
    assert 0.4 < pt[good][-1] < 0.8
