"""Offline tests for offaxis_mc: geometry limits + delivered-table integrity."""

import os

import numpy as np
import pytest

import offaxis_mc as ox


@pytest.fixture(scope="module")
def toy():
    """Small synthetic setup: exponential atmosphere + one-bump production."""
    H = 7.0e5  # scale height [cm]
    rho0 = 1.2e-3  # sea-level density [g/cm3]

    def rho(h):
        h = np.asarray(h, dtype=float)
        return rho0 * np.exp(-np.clip(h, 0, None) / H)

    geom = ox.slant_depth_table(rho, n_h=40, n_psi=80, n_step=400)
    x_grid = np.linspace(5.0, 1000.0, 60)
    e_grid = np.array([0.3, 3.0])
    # production peaked shallow (young shower), same for both energies
    prof = np.exp(-((x_grid - 100.0) ** 2) / (2 * 60.0**2))
    p = np.tile(prof[:, None], (1, len(e_grid)))
    ox._RHO = rho
    return x_grid, e_grid, p, geom


def test_collimated_limit_is_unity(toy):
    """sigma -> 0 must recover the 1D flux exactly (E_off = 1) at every zenith."""
    x_grid, e_grid, p, geom = toy
    sig = np.full(len(e_grid), 1e-6)
    for cz in (0.05, 0.5, 0.95):
        eo = ox.e_off_for_zenith(cz, e_grid, sig, x_grid, e_grid, p, geom, 20, 8)
        assert np.allclose(eo, 1.0, atol=5e-3)


def test_wide_cone_builds_horizon_excess(toy):
    """A wide cone must enhance the horizon and not the vertical (net excess)."""
    x_grid, e_grid, p, geom = toy
    sig = np.full(len(e_grid), 20.0)  # degrees
    e_h = ox.e_off_for_zenith(0.05, e_grid, sig, x_grid, e_grid, p, geom, 30, 12)
    e_v = ox.e_off_for_zenith(0.95, e_grid, sig, x_grid, e_grid, p, geom, 30, 12)
    assert np.all(e_h > 1.1 * e_v)


def test_blocked_rays_give_zero_production(toy):
    x_grid, e_grid, p, geom = toy
    h_grid, psi_grid, table = geom
    # a horizontal-at-sea-level ray pointing below the horizon hits the Earth
    assert not np.isfinite(table[0, -1]) or table[0, -1] > 1e6


TABLE = "offaxis_excess.npz"


@pytest.mark.skipif(not os.path.exists(TABLE), reason="table not built")
class TestDeliveredTable:
    d = np.load(TABLE) if os.path.exists(TABLE) else None

    def test_schema_and_tag(self):
        for k in ("e", "cz", "E_off", "E_off_hi", "E_off_lo", "E_off_mu", "tag"):
            assert k in self.d, f"missing key {k}"
        assert str(self.d["tag"]) == ox.TAG

    def test_high_energy_limit(self):
        e, E = self.d["e"], self.d["E_off"]
        ie = int(np.argmin(abs(e - 30.0)))
        assert np.allclose(E[:, ie], 1.0, atol=0.02)

    def test_subgev_horizon_excess(self):
        e, cz, E = self.d["e"], self.d["cz"], self.d["E_off"]
        ie = int(np.argmin(abs(e - 0.3)))
        ih, iv = int(np.argmin(cz)), int(np.argmax(cz))
        assert E[ih, ie] / E[iv, ie] > 1.5  # the references give ~1.8-1.9

    def test_na61_variants_bracket_central(self):
        e, cz = self.d["e"], self.d["cz"]
        ie = int(np.argmin(abs(e - 0.3)))
        ih = int(np.argmin(cz))
        lo, ce, hi = (
            self.d["E_off_lo"][ih, ie],
            self.d["E_off"][ih, ie],
            self.d["E_off_hi"][ih, ie],
        )
        assert lo < ce < hi  # wider pion angle -> larger horizon excess

    def test_muon_calibration_closure(self):
        """E_off with the muon kernel must be ~1 where daemonflux calibrates:
        vertical-to-moderate zenith muons at E_mu >= 5 GeV (the basis of the
        shape-only convention). Near the horizon the muon still carries the
        parent-pion production angle at 5 GeV, so only require closure by 20 GeV
        (horizontal muon data enter the calibration at high energy)."""
        e, Emu = self.d["e"], self.d["E_off_mu"]  # rows: [horizon, vertical]
        assert np.all(np.abs(Emu[1, e >= 5.0] - 1.0) < 0.03)  # vertical
        assert np.all(np.abs(Emu[0, e >= 20.0] - 1.0) < 0.05)  # horizon
