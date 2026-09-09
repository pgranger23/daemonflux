"""Gates for the conservation diagnostics and the options they added.

Three kinds of check:

1. **No-op gates.**  Every option added for ``diag_conservation.py``
   (``arrival_ray(r_earth=...)``, ``cone_production(ray=...)``,
   ``delivered_joint_factor(ray=..., renorm=...)``) must leave the delivered
   result bit-for-bit unchanged when it is off.

2. **The flat-atmosphere gate.**  Eq. (8) is the exact straight-line 3D transport
   of an isotropic primary flux, so in a *flat* atmosphere it has a closed form:
   ``F = <cos alpha>_K`` at the vertical (see ``diag_conservation`` module
   docstring for the two-line derivation).  Reproducing that number is a much
   stronger statement about the implementation than any comparison to a
   reference flux -- it would break on a missing ``sin alpha`` measure, a wrong
   cone normalisation, a reciprocity/Jacobian factor, or a mis-built ray
   quadrature.

3. **The Liouville constraint hits its target.**  ``liouville_renorm`` is a
   constraint, not a derivation; the gate is only that it does what it says and
   that, being a function of energy alone, it cancels in every zenith ratio.

Run from ``tools/mceq3d``; needs the cached production profile
(``.cache3d/jointprod_chan_*.npz``) -- skipped if absent.
"""

import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def ctx():
    import diag_conservation as dc
    import joint_cone as jc

    if not os.path.exists(dc.PRODCACHE):
        pytest.skip(f"{dc.PRODCACHE} not built (run diag_joint_stages.py once)")
    prod, widths = dc.load()
    return dc, jc, prod, widths


@pytest.fixture(scope="module")
def flat_g1(ctx):
    """``(rc_grid, {species: G == 1})`` and a degenerate cutoff map: the cone
    integral with the geomagnetic side switched off."""
    _, jc, prod, _ = ctx
    ep = prod["ep_grid"]
    rc = np.linspace(0.1, 55.0, 6)
    G = {s: np.ones((len(rc), len(ep))) for s in jc.SPECIES}
    fine = (np.array([0.0, 45.0, 89.0]), np.array([0.0, 180.0, 360.0]),
            np.ones((3, 3)))
    return rc, G, fine


def _F(jc, prod, widths, cz, *, rc, G, fine, **kw):
    chan = {s: prod["chan"][jc.CHANNEL_SPECIES_MAP[s]] for s in jc.SPECIES}
    smu = {s: widths["mu_nue" if "nue" in s else "mu_numu"] for s in jc.SPECIES}
    return jc.delivered_joint_factor(
        cz, [0.0], prod, fine, rc, G, sigma_pi=widths["pi"], sigma_k=widths["k"],
        channels_by_species=chan, sigma_mu_by_species=smu, **kw)["F"]


# ---------------------------------------------------------------------------
# 1. no-op gates
# ---------------------------------------------------------------------------
def test_r_earth_default_is_the_real_earth(ctx):
    _, jc, prod, _ = ctx
    a = jc.arrival_ray(0.35)
    b = jc.arrival_ray(0.35, r_earth=jc.R_EARTH_CM)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_ray_option_off_is_identical(ctx, flat_g1):
    """Passing the ray that ``arrival_ray`` would have built changes nothing."""
    dc, jc, prod, widths = ctx
    rc, G, fine = flat_g1
    for cz in (0.05, 0.95):
        ref = _F(jc, prod, widths, cz, rc=rc, G=G, fine=fine)
        got = _F(jc, prod, widths, cz, rc=rc, G=G, fine=fine,
                 ray=jc.arrival_ray(cz, n_ray=260))
        for s in jc.SPECIES:
            assert np.array_equal(ref[s], got[s])


def test_renorm_off_is_identical(ctx, flat_g1):
    dc, jc, prod, widths = ctx
    rc, G, fine = flat_g1
    ref = _F(jc, prod, widths, 0.45, rc=rc, G=G, fine=fine)
    got = _F(jc, prod, widths, 0.45, rc=rc, G=G, fine=fine, renorm=None)
    for s in jc.SPECIES:
        assert np.array_equal(ref[s], got[s])


# ---------------------------------------------------------------------------
# 2. the flat-atmosphere gate
# ---------------------------------------------------------------------------
def test_flat_atmosphere_returns_mean_cos_alpha(ctx):
    """In a flat atmosphere the exact kernel gives ``F = <cos alpha>_K`` at the
    vertical, for every parent channel separately.

    ``<cos alpha>`` runs from 0.60 (kaon cone at 0.2 GeV) to 0.999, so this is a
    gate on the cone normalisation and the ray quadrature over a factor-1.7 range
    of the answer, not a near-unity tautology.
    """
    dc, jc, prod, widths = ctx
    ep = prod["ep_grid"]
    fg = dc.flat_geom(prod)
    N, D = dc.terms(1.0, prod, widths, geom=fg, ray=dc.flat_ray(1.0))
    for s in ("numu", "nue"):
        mca = dc.mean_cos_alpha(widths, ep, s)
        for ch in dc.CHANNELS:
            got = N[(s, ch)] / np.maximum(D[(s, ch)], 1e-300)
            # tolerance: the ray quadrature + the linear p(X) interpolation.
            assert np.allclose(got, mca[ch], rtol=2e-3), (
                s, ch, got[:6], mca[ch][:6])
        dtot = np.maximum(sum(D[(s, c)] for c in dc.CHANNELS), 1e-300)
        blend = sum(N[(s, c)] for c in dc.CHANNELS) / dtot
        pred = sum(D[(s, c)] * mca[c] for c in dc.CHANNELS) / dtot
        assert np.allclose(blend, pred, rtol=2e-3)


def test_flat_atmosphere_factor_is_below_one(ctx):
    """The flat-atmosphere factor is a *deficit* at the vertical (``<cos alpha>``),
    not an excess -- the sanity check that the sub-GeV excess of the delivered
    factor is a curvature/horizon effect and not a cone-normalisation error."""
    dc, jc, prod, widths = ctx
    ep = prod["ep_grid"]
    fg = dc.flat_geom(prod)
    f = dc.factor(1.0, prod, widths, geom=fg, ray=dc.flat_ray(1.0))
    for s in dc.SPECIES:
        for E in (0.2, 0.3, 0.5, 1.0):
            v = dc.at(ep, f[s], E)
            assert 0.5 < v < 1.0, (s, E, v)


# ---------------------------------------------------------------------------
# 3. the Liouville constraint
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("measure", ("cos", "omega"))
def test_liouville_renorm_hits_its_target(ctx, measure):
    dc, jc, prod, widths = ctx
    czs = np.round(np.arange(0.1, 1.0, 0.2), 2)  # 5 zeniths: fast, same measure
    g = jc.liouville_renorm(prod, widths, czs=czs, measure=measure, target=1.0)
    _, F = jc.sky_scan(prod, widths, czs, renorm=g)
    w = (czs / czs.sum()) if measure == "cos" else np.full(len(czs), 1.0 / len(czs))
    for s in jc.SPECIES:
        assert np.allclose(w @ F[s], 1.0, atol=1e-10)


def test_liouville_renorm_cancels_in_zenith_ratios(ctx):
    """It is an energy-only rescaling, so it cannot change H/V (or E/W)."""
    dc, jc, prod, widths = ctx
    czs = np.array([0.05, 0.95])
    g = jc.liouville_renorm(prod, widths,
                            czs=np.round(np.arange(0.1, 1.0, 0.2), 2))
    _, F0 = jc.sky_scan(prod, widths, czs)
    _, F1 = jc.sky_scan(prod, widths, czs, renorm=g)
    for s in jc.SPECIES:
        assert np.allclose(F1[s][0] / F1[s][1], F0[s][0] / F0[s][1], rtol=1e-12)


def test_liouville_scale_is_a_few_percent(ctx):
    """Sub-GeV the constraint is a <=10% renormalisation and it vanishes above a
    few GeV -- if it ever became large, the factor would not be a redistribution
    at all."""
    dc, jc, prod, widths = ctx
    ep = prod["ep_grid"]
    g = jc.liouville_renorm(prod, widths,
                            czs=np.round(np.arange(0.1, 1.0, 0.2), 2))
    for s in jc.SPECIES:
        assert 1.0 < np.interp(np.log(0.3), np.log(ep), g[s]) < 1.12
        assert abs(np.interp(np.log(5.0), np.log(ep), g[s]) - 1.0) < 0.01
