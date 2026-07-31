"""Fast regression test on the DELIVERED zenith shape ingredients vs Honda.

Motivation (independent review, 2026-07-23): the whole test suite was
sub-module/synthetic -- nothing pinned the delivered model's agreement with the
reference tables, so a regression in the physics (as opposed to a crash) would
pass CI silently. The moment-cone E_off change and the 11-15% zenith-shape
finding were both caught by hand-run multi-minute scripts, not by tests.

SCOPE AND HONEST LIMITS. A full `solve()` cannot be a fast test: `MCEq3DFlux.base()`
alone takes ~107 s (an MCEq cascade solve per zenith) and engine construction ~35 s.
This test therefore pins the two *3D-specific delivered ingredients* that load
instantly from committed/cached artifacts, with **no MCEq run at all**:

  * `E_off` (offaxis_excess.npz)  -- the off-axis production factor;
  * Honda's own table (honda_kam.npz) -- the reference.

and checks their relation using a **recorded** 1D base shape (measured
2026-07-23 via diag_shape_decompose.py). The base itself is NOT tested here -- it
is the daemonflux/MCEq 1D flux, validated separately. What this test does catch is
a regression in `E_off` (e.g. the sampled-vs-moment cone-kernel bug, which shifted
the sub-GeV horizon/vertical by 12-15%) and any corruption of the shipped table.

The full absolute/zenith/flavour/charge comparison remains
`diag_full_comparison.py` (minutes, not CI).
"""
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
EOFF = os.path.join(HERE, "offaxis_excess.npz")
HONDA = os.path.join(HERE, "honda_kam.npz")

# 1D base horizon/vertical ratio, measured through the delivered configuration
# (hybrid base + GSF primary + Kamioka) by diag_shape_decompose.py on 2026-07-23.
# Recorded, not computed -- computing it needs the ~107 s base() call.
HV_BASE_RECORDED = {0.3: 1.124, 0.5: 1.148, 1.0: 1.218}

# Per-energy tolerance on (base x E_off) / Honda. These are set from MEASURED
# values of both the delivered moment-cone table and the legacy sampled one, so
# the thresholds genuinely discriminate rather than passing everything:
#
#            0.3 GeV   0.5 GeV   1.0 GeV
#   moment    0.975     1.021     1.090     <- delivered, must PASS
#   sampled   1.009     1.094     1.138     <- legacy over-wide cone, must FAIL
#
# Note the 0.3 GeV point cannot discriminate (the legacy table is *closer* to
# Honda there); the 0.5 GeV point is the real discriminator, with 1.021 vs 1.094
# giving ~7 points of separation. Tolerances leave >=3 points of headroom for the
# delivered values while still rejecting the legacy build.
TOL_BY_E = {0.3: 0.10, 0.5: 0.06, 1.0: 0.12}


def _log_at(y, x, X):
    return float(np.exp(np.interp(np.log(X), np.log(x),
                                  np.log(np.maximum(y, 1e-300)))))


@pytest.fixture(scope="module")
def tables():
    if not (os.path.exists(EOFF) and os.path.exists(HONDA)):
        pytest.skip("offaxis_excess.npz / honda_kam.npz not present")
    d = np.load(EOFF)
    h = dict(np.load(HONDA))
    return d, h


def _honda_hv(h, key, E):
    """Honda horizon/vertical (azimuth-averaged) for a species at energy E."""
    He, Hcz = h["E"], h["czlo"]
    ih = int(np.argmin(np.abs(Hcz - 0.0)))   # cosZ bin 0.0-0.1 -> horizon
    iv = int(np.argmin(np.abs(Hcz - 0.9)))   # cosZ bin 0.9-1.0 -> vertical
    return _log_at(h[key][ih].mean(0), He, E) / _log_at(h[key][iv].mean(0), He, E)


def _eoff_hv(d, E):
    """E_off horizon/vertical from the delivered table."""
    cz, e, E_off = d["cz"], d["e"], d["E_off"]
    ih, iv = int(np.argmin(cz)), int(np.argmax(cz))
    return _log_at(E_off[ih], e, E) / _log_at(E_off[iv], e, E)


def test_eoff_plus_base_reproduces_honda_zenith_shape(tables):
    """base x E_off must track Honda's sub-GeV zenith shape.

    This is the check that would have caught the over-wide sampled cone kernel:
    with it, the 0.5 GeV horizon/vertical was ~12% high; with the corrected
    moment sigma_pi it lands within a few % of Honda.
    """
    d, h = tables
    for E, hv_base in HV_BASE_RECORDED.items():
        got = hv_base * _eoff_hv(d, E)
        want = _honda_hv(h, "numu", E)
        tol = TOL_BY_E[E]
        assert abs(got / want - 1.0) < tol, (
            f"zenith shape (base x E_off) off Honda at {E} GeV: "
            f"{got:.3f} vs {want:.3f} (ratio {got/want:.3f}, tol {tol})"
        )


def test_eoff_horizon_excess_is_sub_gev(tables):
    """E_off's horizon excess must be large sub-GeV and vanish at high E --
    the qualitative signature of the off-axis production effect."""
    d, _ = tables
    hv_low = _eoff_hv(d, 0.3)
    hv_high = _eoff_hv(d, 10.0)
    assert hv_low > 1.3, f"sub-GeV horizon excess too weak: {hv_low:.3f}"
    assert abs(hv_high - 1.0) < 0.05, (
        f"E_off should vanish at 10 GeV, got horizon/vertical {hv_high:.3f}"
    )


def test_eoff_table_is_the_moment_cone_build(tables):
    """Guard the delivered table against silently reverting to the sampled
    kernel, which was root-caused as 16-37% too wide (a theta-binning artefact).

    Measured horizon/vertical of E_off at 0.5 GeV: moment (delivered) = 1.288,
    sampled (legacy, over-wide) = 1.380. 0.5 GeV is used because it separates the
    two builds cleanly (~7%), unlike 0.3 GeV where they nearly coincide
    (1.583 vs 1.638).
    """
    d, _ = tables
    hv_05 = _eoff_hv(d, 0.5)
    assert 1.24 < hv_05 < 1.33, (
        f"E_off horizon/vertical at 0.5 GeV = {hv_05:.3f}, outside the "
        "moment-cone range [1.24, 1.33] -- has offaxis_excess.npz been rebuilt "
        "with cone_kernel='sampled' (legacy, over-wide)?"
    )
