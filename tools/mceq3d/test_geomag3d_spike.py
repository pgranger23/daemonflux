"""De-risking-spike checks: the geomagnetic force operator is flux-conserving,
the sharp cutoff Gibbs-rings under raw P_N truncation, and the production-cone
low-pass tames it at feasible l_max (the go signal for a deterministic 3D
geomagnetic cascade)."""
import numpy as np

from geomag3d_spike import (
    sphere_grid, sh_forward, sh_reconstruct, heat_lowpass, ew_suppression,
    force_apply, rotate_dirs,
)


def test_force_operator_conserves_flux():
    TH, PH, W = sphere_grid(48, 96)
    f0 = np.exp(-((TH - 0.7) ** 2) / (2 * 0.3**2))
    f1 = force_apply(f0, TH, PH, W, np.array([0.0, 1.0, 0.3]), 0.6)
    assert abs(np.sum(f1 * W) / np.sum(f0 * W) - 1) < 5e-3


def test_force_operator_matches_analytic_rotation():
    TH, PH, W = sphere_grid(64, 128)
    f0 = np.exp(-((TH - 0.6) ** 2) / (2 * 0.25**2))
    ax, ang = np.array([0.0, 1.0, 0.3]), 0.6
    f1 = force_apply(f0, TH, PH, W, ax, ang)
    thb, phb = rotate_dirs(TH, PH, ax, -np.linalg.norm(ax) * ang)
    ref = np.exp(-((thb - 0.6) ** 2) / (2 * 0.25**2))
    assert np.sqrt(np.mean((f1 - ref) ** 2)) / ref.max() < 3e-2


def test_raw_cutoff_gibbs_persists_but_cone_tames_it():
    TH, PH, W = sphere_grid(64, 128)
    G = ew_suppression(TH, PH, e_gev=1.0)
    a = sh_forward(G, TH, PH, W, 40)
    # raw P_N truncation keeps ringing (overshoot does not vanish by l_max=24)
    rec_raw = sh_reconstruct(a, TH, PH, 24)
    ov_raw = max(rec_raw.max() - G.max(), G.min() - rec_raw.min(), 0.0)
    assert ov_raw > 0.03  # persistent Gibbs overshoot
    # the production-cone low-pass tames it to <1% at feasible l_max=16
    a_cone = heat_lowpass(a, np.deg2rad(12.0))
    truth = sh_reconstruct(a_cone, TH, PH, 40)
    rec_cone = sh_reconstruct(a_cone, TH, PH, 16)
    ov_cone = max(rec_cone.max() - truth.max(), truth.min() - rec_cone.min(), 0.0)
    assert ov_cone < 0.01
