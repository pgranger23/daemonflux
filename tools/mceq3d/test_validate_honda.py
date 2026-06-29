"""Offline test of the Honda cross-check (uses the committed cache, no network)."""

import os

import numpy as np
import pytest

import validate_honda as vh

CACHE = os.path.join(os.path.dirname(__file__), vh.CACHE)


@pytest.mark.skipif(not os.path.exists(CACHE), reason="Honda cache not present")
def test_honda_cache_schema():
    h = vh.fetch_honda_cache(CACHE)
    assert h["numu"].shape == (len(h["czlo"]), len(h["azlo"]), len(h["E"]))
    assert len(h["azlo"]) == 12 and len(h["czlo"]) == 20
    assert h["E"][0] < 0.2 and h["E"][-1] > 1e3  # 0.1 GeV .. >=1 TeV
    assert np.all(h["numu"] > 0)


@pytest.mark.skipif(not os.path.exists(CACHE), reason="Honda cache not present")
def test_honda_east_west_signature():
    h = vh.fetch_honda_cache(CACHE)
    E, ew, sec = vh.honda_observables(h)
    # East-West is a strong sub-GeV effect that vanishes above the cutoff
    assert ew[np.argmin(np.abs(E - 0.5))] > 2.0  # large sub-GeV
    assert ew[np.argmin(np.abs(E - 30.0))] < 1.1  # gone above ~10 GeV
    # sec(theta) horizon enhancement grows with energy and exceeds 1
    assert sec[np.argmin(np.abs(E - 1000.0))] > sec[np.argmin(np.abs(E - 10.0))] > 1.0
