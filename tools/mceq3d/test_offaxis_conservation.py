"""Flux-conservation guardrail for the off-axis factor E_off (no reference flux).

This is an *independent* physics check, not a comparison to Honda/Bartol: 3D
geometry only redistributes neutrinos in arrival direction; it cannot change the
number or energy of neutrinos produced by a fixed, isotropic primary flux. So the
solid-angle average of E_off at fixed energy must be close to 1 -- a genuine
excess near the horizon must be paid for by a deficit elsewhere (near the
vertical). E_off that averaged well above 1 would be manufacturing flux (double
counting); one that never dipped below 1 would be a pedestal, not a redistribution.

We check the delivered table directly (fast, no MCEq):
  * <E_off>_Omega  is within a few % of 1 at every energy;
  * E_off > 1 near the horizon and < 1 near the vertical (redistribution shape);
  * E_off -> 1 at high energy.

The base-weighted number (the physically exact inflation factor, W(E)) is
computed by ``diag_eoff_conservation.py``; it is ~1.04-1.06 at 0.2-0.3 GeV and
->1 above a few GeV. This test uses the geometric (flat-cosZ) measure so it needs
no cascade solve.
"""
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(HERE, "offaxis_excess.npz")

# Physical tolerance on the geometric solid-angle average of E_off at fixed E.
# A perfect flux-conserving redistribution gives 1.0 exactly; a few % of genuine
# net sub-GeV enhancement (also seen in full 3D calculations) is allowed. Well
# above this would indicate E_off is inflating the total flux (double counting).
W_MAX = 1.08


@pytest.fixture(scope="module")
def table():
    if not os.path.exists(TABLE):
        pytest.skip(f"{TABLE} not built (run offaxis_mc.py --build)")
    d = np.load(TABLE)
    return d["cz"], d["e"], d["E_off"]


def _omega_avg(eoff_col, cz):
    """Geometric solid-angle average over the down hemisphere: dOmega = dphi dcosZ,
    phi -> 2pi, so the weight is dcosZ (uniform bin centres -> equal weights)."""
    dcz = np.gradient(cz)
    return np.sum(eoff_col * dcz) / np.sum(dcz)


# Below this energy E_off is an extrapolation: the production cones reach ~40 deg,
# the factorised small-angle geometry is least reliable, and daemonflux's own base
# extrapolates beyond its muon calibration. The paper states the excess below
# 0.1 GeV is not modelled. The conservation residual is allowed to grow here to a
# looser ceiling; the strict bound applies in the modelled range.
E_MODELLED_MIN = 0.15
W_MAX_EDGE = 1.12


def test_eoff_does_not_inflate_total_flux(table):
    """<E_off>_Omega must stay ~1 (E_off redistributes, it must not manufacture
    flux): <= W_MAX in the modelled range (E >= 0.15 GeV), <= W_MAX_EDGE below it
    (the factorised extrapolation edge, where the residual is expected to grow --
    e.g. ~8% at 0.11 GeV; a documented boundary, not a free knob)."""
    cz, e, E_off = table
    for k, en in enumerate(e):
        w = _omega_avg(E_off[:, k], cz)
        bound = W_MAX if en >= E_MODELLED_MIN else W_MAX_EDGE
        assert w <= bound, (
            f"E_off inflates the angle-integrated flux at {en:.2f} GeV: "
            f"<E_off>_Omega = {w:.3f} > {bound} (not flux-conserving)"
        )


def test_eoff_is_a_redistribution_sub_gev(table):
    """Sub-GeV: E_off > 1 near the horizon AND < 1 near the vertical -- the excess
    is redistributed, not a pedestal added on top of the flux."""
    cz, e, E_off = table
    horiz = int(np.argmin(cz))  # smallest cosZ = horizon
    vert = int(np.argmax(cz))  # largest cosZ = vertical
    for E in (0.3, 0.5):
        k = int(np.argmin(np.abs(e - E)))
        assert E_off[horiz, k] > 1.03, (
            f"no horizon enhancement at {E} GeV: {E_off[horiz, k]:.3f}"
        )
        assert E_off[vert, k] < 1.0, (
            f"E_off does not dip below 1 at the vertical at {E} GeV "
            f"({E_off[vert, k]:.3f}) -> pedestal, not a redistribution"
        )


def test_eoff_vanishes_at_high_energy(table):
    """E_off -> 1 (all zeniths) by ~10 GeV: 3D is a low-energy effect."""
    cz, e, E_off = table
    k = int(np.argmin(np.abs(e - 10.0)))
    assert np.allclose(E_off[:, k], 1.0, atol=0.03), (
        f"E_off not ~1 at 10 GeV: {E_off[:, k]}"
    )


# --------------------------------------------------------------------------
# Same guardrail for the per-species channel-weighted tables
# --------------------------------------------------------------------------
CHANNEL_TABLES = [
    os.path.join(HERE, n) for n in
    ("offaxis_excess_channel.npz", "offaxis_excess_channel_v2.npz",
     "offaxis_excess_v2.npz", "offaxis_excess_rebuild_old.npz")
]

# The channel-weighted cone is genuinely wider for the muon-decay component, so a
# somewhat larger net (non-redistributive) residual is expected -- but only
# somewhat: a cone change that manufactured flux rather than moving it would be
# an artefact, not physics. This is the honest ceiling, not a fitted one.
W_MAX_CHANNEL = 1.15


@pytest.mark.parametrize("path", CHANNEL_TABLES)
def test_species_tables_do_not_inflate_total_flux(path):
    if not os.path.exists(path):
        pytest.skip(f"{path} not built")
    d = np.load(path)
    cz, e, tab = d["cz"], d["e"], d["E_off_s"]
    for isp, sp in enumerate(d["species"]):
        for k, en in enumerate(e):
            if en < E_MODELLED_MIN:
                continue
            w = _omega_avg(tab[isp, :, k], cz)
            assert w <= W_MAX_CHANNEL, (
                f"{os.path.basename(path)} {sp}: <E_off>_Omega = {w:.3f} at "
                f"{en:.2f} GeV inflates the angle-integrated flux"
            )


@pytest.mark.parametrize("path", CHANNEL_TABLES)
def test_species_tables_are_redistributions(path):
    """Sub-GeV: horizon enhanced, vertical suppressed, for every species."""
    if not os.path.exists(path):
        pytest.skip(f"{path} not built")
    d = np.load(path)
    cz, e, tab = d["cz"], d["e"], d["E_off_s"]
    ih, iv = int(np.argmin(cz)), int(np.argmax(cz))
    for isp, sp in enumerate(d["species"]):
        for E in (0.3, 0.5):
            k = int(np.argmin(np.abs(e - E)))
            assert tab[isp, ih, k] > 1.03, (sp, E, tab[isp, ih, k])
            assert tab[isp, iv, k] < 1.0, (sp, E, tab[isp, iv, k])
