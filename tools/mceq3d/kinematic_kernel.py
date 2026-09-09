"""Neutrino production angle from the generator moments + exact decay kinematics.

The off-axis excess (``offaxis_mc.py``) is driven by the angle between the
primary/shower axis and the produced neutrino. Following the MCEq/daemonflux
philosophy -- the production angle must come from the *same* hadronic interaction
model that provides the yields, and be anchored to data rather than hand-set -- we
build the per-channel angle from:

  * the **generator meson production angle** ``<theta^2>(E_meson)`` measured from
    the SIBYLL-2.3d / UrQMD-3.4 moments (regenerated through ``chromo`` by
    ``kernel_regeneration.py``, pooled by ``fokker_planck_3d.load_theta2``, and
    validated against NA61/SHINE by ``validate_na61.py``). This is the actual
    inclusive production angle of the generator -- **no hand-set p_T or x_F**.
  * **exact two-body decay** ``pi/K -> mu nu_mu`` for the direct channels,
  * the **H3a primary spectrum** (``crflux``) to weight which meson energies feed
    each neutrino energy -- the same primary MCEq uses (no hand-set spectral index).

The neutrino production-angle variance is the sum of the (uncorrelated) generator
meson-production and decay contributions, ``<theta_nu^2> = <theta_meson^2> +
theta_decay^2``, binned by ``E_nu``. ``offaxis_mc`` uses the **pion** angle
``sigma_pi`` as the flavour-independent excess kernel (the near-horizon excess is
a pion-production-rate effect inherited by all neutrinos); ``sigma_K`` is provided
for reference (kaons are a subdominant parent). ``channel_fractions`` (MCEq nu_mu/
nu_e channel fractions) is retained as a utility for cross-checks.

The moments are single-interaction. Because the excess is set by the pion
production geometry (not by multi-generation angular accumulation), this is
adequate: the resulting E_off reproduces the Honda/Bartol nu_mu and nu_e zenith
shapes to ~5% sub-GeV (``offaxis_mc.py --validate``) with no hand-tuning. Run
``python kinematic_kernel.py`` to print the per-species angles and the fractions.
"""

from __future__ import annotations

import numpy as np

from fokker_planck_3d import load_theta2, theta2_interp

M_PI, M_MU, M_K = 0.13957, 0.10566, 0.49368
# nu_mu CM energy in two-body meson -> mu nu decay
ESTAR_PI = (M_PI**2 - M_MU**2) / (2 * M_PI)
ESTAR_K = (M_K**2 - M_MU**2) / (2 * M_K)
# mu energy/momentum in the pion rest frame (for the muon closure check)
EMU_PI = (M_PI**2 + M_MU**2) / (2 * M_PI)
PSTAR_PI = ESTAR_PI

# Charge-combined generator moment files per meson species (chromo: UrQMD+SIBYLL).
#
# DEFAULT (2026-09-04): the **arcsin-corrected** ``*_v2`` regeneration
# (``regen_moments_mp.py``; same UrQMD-3.4 <=80 GeV / SIBYLL-2.3d >80 GeV splice
# recipe as ``KERNEL_GENERATION.md``).  The original files extracted the
# production angle as ``arctan(p_T/p_total)`` instead of ``arcsin(p_T/p)``,
# biasing ``sqrt(<theta^2>)`` ~16% too narrow at 0.3 GeV (<1% above 3 GeV);
# ``kernel_regeneration.production_angle`` is the canonical definition.
# Until 2026-09-04 the corrected set was opt-in through the ``moments=`` /
# ``cone_moments=`` keyword chain, so every default call silently used the old
# files.  ``_MOMENTS_LEGACY`` keeps them reachable for A/B work.
_MOMENTS = {
    "pi": ["m_spliced_v2.npz", "m_piminus_v2.npz"],
    "k": ["m_Kplus_v2.npz", "m_Kminus_v2.npz"],
}

#: the pre-2026-09 (``arctan``) moment files -- A/B only, never the default.
_MOMENTS_LEGACY = {
    "pi": ["m_spliced.npz", "m_piminus.npz"],
    "k": ["m_Kplus.npz", "m_Kminus.npz"],
}


def meson_theta2(species, moments=None):
    """Charge-combined generator meson production-angle variance <theta^2>(E_mes)
    [rad^2] vs meson energy [GeV], from the SIBYLL/UrQMD moments (NA61-validated).

    ``moments`` optionally overrides the default per-species file list (e.g. to
    swap in an alternative generator for the hadronic-model spread)."""
    files = (moments or _MOMENTS)[species]
    e, t = load_theta2(files[0])
    acc = [theta2_interp(e, e, t)]
    for f in files[1:]:
        e2, t2 = load_theta2(f)
        acc.append(theta2_interp(e, e2, t2))
    return e, np.mean(acc, axis=0)


def _sigma_powerlaw(e_nu, th2, w, e_query, emin=0.2, emax=15.0, nbin=18, floor=2000):
    """Flux-weighted RMS neutrino angle [deg] in log-E bins, returned as a
    power-law fit at ``e_query``."""
    edges = np.geomspace(emin, emax, nbin + 1)
    ec, sg = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (e_nu >= lo) & (e_nu < hi)
        if m.sum() < floor:
            continue
        ec.append(np.sqrt(lo * hi))
        sg.append(np.degrees(np.sqrt(np.average(th2[m], weights=w[m]))))
    ec, sg = np.asarray(ec), np.asarray(sg)
    b, lnA = np.polyfit(np.log(ec), np.log(sg), 1)
    return np.exp(lnA) * np.asarray(e_query, dtype=float) ** b


def channel_shapes(e_query, n=6_000_000, seed=1, primary="H3a", moments=None):
    """Per-channel neutrino production angle sigma_c(E_nu) [deg] for the collimated
    direct channels, from the generator meson angle + exact two-body decay,
    weighted by the H3a primary spectrum. Returns dict {'pi': ..., 'k': ...}.

    ``moments`` optionally overrides the default SIBYLL/UrQMD moment files (used to
    swap in an alternative generator for the hadronic-model spread)."""
    import crflux.models as crf

    pm = crf.HillasGaisser2012(primary)
    rng = np.random.default_rng(seed)
    out = {}
    for name, mass, estar in (("pi", M_PI, ESTAR_PI), ("k", M_K, ESTAR_K)):
        e_m, t2 = meson_theta2(name, moments=moments)
        e_mes = np.exp(rng.uniform(np.log(0.5), np.log(300.0), n))
        w = pm.tot_nucleon_flux(e_mes) * e_mes  # dN/dlnE (log-uniform sampling)
        ct = rng.uniform(-1, 1, n)  # cos(theta*) isotropic two-body decay
        gamma = e_mes / mass
        e_nu = gamma * estar * (1 + ct)  # beta ~ 1
        th_dec = np.arctan2(np.sin(np.arccos(ct)), gamma * (ct + 1.0))
        th2 = theta2_interp(e_mes, e_m, t2) + th_dec**2  # generator prod. + decay
        out[name] = _sigma_powerlaw(e_nu, th2, w, e_query)
    return out


def muon_shape(e_query, n=6_000_000, seed=2, primary="H3a", moments=None):
    """Muon angle w.r.t. the primary axis, sigma_mu(E_mu) [deg]: the generator pion
    production angle folded with exact pi -> mu nu two-body kinematics, H3a-weighted.
    Used by ``offaxis_mc.build`` for the muon-calibration closure check (E_off
    evaluated with this kernel must be ~1 for E_mu >~ 5 GeV, daemonflux's
    calibration region)."""
    import crflux.models as crf

    pm = crf.HillasGaisser2012(primary)
    rng = np.random.default_rng(seed)
    e_m, t2 = meson_theta2("pi", moments=moments)
    e_mes = np.exp(rng.uniform(np.log(0.5), np.log(300.0), n))
    w = pm.tot_nucleon_flux(e_mes) * e_mes
    ct = rng.uniform(-1, 1, n)
    gamma = e_mes / M_PI
    # E_lab = gamma (E* + beta p* ct) ; tan(th_lab) = sin th* / (gamma (ct + E*/p*))
    # (massive daughter: the boost ratio is E*/p* = 1/beta*, NOT p*/E* as for the
    # massless neutrino -- the muon is always boosted forward, beta_frame > beta*).
    e_mu = gamma * EMU_PI * (1 + (PSTAR_PI / EMU_PI) * ct)
    th_mu = np.arctan2(np.sin(np.arccos(ct)), gamma * (ct + EMU_PI / PSTAR_PI))
    th2 = theta2_interp(e_mes, e_m, t2) + th_mu**2
    return _sigma_powerlaw(e_mu, th2, w, e_query)


def mudecay_shape(e_query, b_gauss=0.45, zenith_deg=80.0):
    """RMS angle sigma_mudecay(E_nu) [deg] of a **muon-decay** neutrino w.r.t. the
    primary -- wider than the direct pion cone (:func:`channel_shapes`` ``['pi']``)
    and carrying an **energy-independent floor** from in-flight muon bending.

    Combines three uncorrelated contributions in quadrature (with E_mu ~ 3 E_nu,
    the Michel mean):

      * ``sigma_mu(E_mu)`` -- the muon direction w.r.t. the primary axis
        (:func:`muon_shape`: generator pion angle folded with exact pi->mu nu),
      * the **Michel decay** angle of the neutrino w.r.t. the muon (isotropic
        rest-frame emission of a massless nu, exact boost), and
      * the **in-flight bending** angle ``Delta_phi = qB tau/m`` -- energy
        INDEPENDENT (:func:`muon_bending.bending_angle`), weighted by the
        square-root of the decay-in-flight fraction (only muons that decay in
        flight contribute the coherent bend).

    ~40% of sub-GeV nu_mu and ~all nu_e come from muon decay
    (:func:`channel_fractions`), so applying only the narrow pion cone to them
    (as the geomagnetic cone did) under-smears the near-horizon E-W; this width is
    the channel-weighted correction.
    """
    import muon_bending as mb

    e_mu = 3.0 * np.asarray(e_query, dtype=float)
    sig_mu = muon_shape(e_mu)  # muon direction vs primary [deg]
    # Michel neutrino angle w.r.t. the muon: isotropic rest-frame ct, massless-nu
    # boost tan(th) = sin th* / (gamma (ct + 1)); deterministic quadrature over ct.
    ct = np.linspace(-0.9995, 0.9995, 4000)
    gamma = (e_mu / M_MU)[:, None]
    th = np.arctan2(np.sqrt(1.0 - ct[None, :] ** 2), gamma * (ct[None, :] + 1.0))
    sig_numu = np.degrees(np.sqrt(np.mean(th**2, axis=1)))
    # energy-independent bending, weighted by sqrt(decay-in-flight fraction)
    sig_bend0 = np.degrees(mb.bending_angle(b_gauss))
    fdec = mb.decay_in_flight_fraction(e_mu, zenith_deg=zenith_deg)
    sig_bend = sig_bend0 * np.sqrt(np.clip(fdec, 0.0, 1.0))
    return np.sqrt(sig_mu**2 + sig_numu**2 + sig_bend**2)


def _sigma_binned(e_nu, th2, w, e_query, emin=0.15, emax=15.0, nbin=20,
                  floor=2000):
    """Flux-weighted space-angle RMS [deg] in log-E bins, returned by **log-log
    interpolation** of the binned values (clamped at the ends).

    Unlike :func:`_sigma_powerlaw` this makes no power-law assumption. That
    matters for the muon-decay channel, whose width is the quadrature sum of a
    falling (kinematic) part and an energy-*independent* geomagnetic-bending
    floor: a single power-law fit through such a curve is badly biased at both
    ends (it under-predicts the high-E floor and over-predicts nothing at low E,
    or vice versa depending on the lever arm). The pion channel is close to a
    power law, so the two agree there to <2%; :func:`channel_shapes` keeps the
    power-law fit so the delivered table is bit-for-bit unchanged.
    """
    edges = np.geomspace(emin, emax, nbin + 1)
    ec, sg = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (e_nu >= lo) & (e_nu < hi)
        if m.sum() < floor:
            continue
        ec.append(np.sqrt(lo * hi))
        sg.append(np.degrees(np.sqrt(np.average(th2[m], weights=w[m]))))
    ec, sg = np.asarray(ec), np.asarray(sg)
    lg = np.interp(np.log(np.asarray(e_query, dtype=float)), np.log(ec),
                   np.log(sg), left=np.log(sg[0]), right=np.log(sg[-1]))
    return np.exp(lg)


def _michel_x(rng, n, kind):
    """Rest-frame x = 2E_nu/m_mu for an **unpolarised** muon decay:
    dN/dx ~ 2x^2(3-2x) for the nu_mu, 12x^2(1-x) for the nu_e (Michel).
    (Only the rest-frame *angle* enters the lab opening angle for a massless
    daughter -- x fixes which lab energy bin the event lands in.)"""
    out = np.empty(n)
    fmax = 2.0 if kind == "numu" else 1.78
    filled = 0
    while filled < n:
        m = n - filled
        x = rng.random(m)
        f = 2 * x**2 * (3 - 2 * x) if kind == "numu" else 12 * x**2 * (1 - x)
        acc = rng.random(m) * fmax < f
        k = int(acc.sum())
        out[filled:filled + k] = x[acc]
        filled += k
    return out


def direct_shape_mc(e_query, n=4_000_000, seed=11, primary="H3a", moments=None,
                    species="numu"):
    """**Exact** direct-channel cone: space-angle RMS [deg] of a ``pi -> mu nu_mu``
    neutrino w.r.t. the primary axis, vs E_nu.

    Same physics as :func:`channel_shapes` ``['pi']`` but with the exact two-body
    boost (``beta_pi < 1`` in both E_nu and the opening angle, where
    :func:`channel_shapes` uses beta = 1) and the binned interpolation of
    :func:`_sigma_binned` instead of a power-law fit. Provided as the correctness
    reference for the delivered ``sigma_pi``; the two agree to ~5%.
    """
    import crflux.models as crf

    pm = crf.HillasGaisser2012(primary)
    rng = np.random.default_rng(seed)
    e_m, t2 = meson_theta2("pi", moments=moments)
    e_pi = np.exp(rng.uniform(np.log(0.2), np.log(300.0), n))
    w = pm.tot_nucleon_flux(e_pi) * e_pi
    ct = rng.uniform(-1, 1, n)
    g = e_pi / M_PI
    b = np.sqrt(np.maximum(1 - 1 / g**2, 0.0))
    e_nu = g * ESTAR_PI * (1 + b * ct)
    th = np.arctan2(np.sqrt(1 - ct**2), g * (ct + b))
    th2 = theta2_interp(e_pi, e_m, t2) + th**2
    return _sigma_binned(e_nu, th2, w, e_query)


def mudecay_shape_mc(e_query, species="numu", n=4_000_000, seed=12,
                     primary="H3a", moments=None, bending=False,
                     b_gauss=0.45, zenith_deg=80.0):
    """**Exact** muon-decay-channel cone: space-angle RMS [deg] of a
    ``pi -> mu -> e nu nu`` neutrino w.r.t. the **primary** axis, vs E_nu.

    This is the cone that Eq. (8) of the off-axis construction needs for the
    muon-decay component: the distribution of the primary direction given the
    *neutrino* direction accumulates over the whole decay chain (Lipari 2000),

        theta_tot^2 = theta_had^2(E_pi) + theta_(pi->mu)^2 + theta_(mu->nu)^2
                      [ + theta_bend^2 ] ,

    added in quadrature (uncorrelated azimuths). Improvements over the earlier
    :func:`mudecay_shape`:

      * the muon energy is **sampled** from exact ``pi -> mu nu`` two-body
        kinematics and the neutrino energy from the exact Michel boost, instead of
        the fixed ``E_mu = 3 E_nu`` surrogate. On a steeply falling spectrum the
        correct convolution puts more low-E_mu (wide-angle) muons under a given
        E_nu, so the cone comes out wider;
      * the massless-daughter boost uses ``beta_mu < 1`` (the old code used
        ``ct + 1``, which loses the backward-emitted tail entirely);
      * the width is returned by log-log interpolation of the binned RMS
        (:func:`_sigma_binned`), not a power-law fit, which the energy-independent
        bending floor breaks.

    ``bending`` is **off by default**: the in-flight bending angle depends on the
    local |B|, and E_off is meant to be a site-independent production factor.
    Switch it on to bound the effect (it adds an energy-flat ~5 deg in
    quadrature, i.e. it matters only above ~1 GeV).

    The rest-frame decay is treated as unpolarised. Muons from pion decay are
    fully polarised, but for a *massless* daughter the lab opening angle depends
    only on (theta*, gamma) -- not on E* -- so the polarisation enters only
    through the (x*, theta*) correlation that decides which lab energy bin an
    event falls in; see ``--polarisation-scan`` for the bound.
    """
    import crflux.models as crf
    import muon_bending as mb

    pm = crf.HillasGaisser2012(primary)
    rng = np.random.default_rng(seed)
    e_m, t2 = meson_theta2("pi", moments=moments)
    e_pi = np.exp(rng.uniform(np.log(0.2), np.log(300.0), n))
    w = pm.tot_nucleon_flux(e_pi) * e_pi
    # pi -> mu nu, exact (massive daughter)
    ct1 = rng.uniform(-1, 1, n)
    g1 = e_pi / M_PI
    b1 = np.sqrt(np.maximum(1 - 1 / g1**2, 0.0))
    e_mu = g1 * (EMU_PI + b1 * PSTAR_PI * ct1)
    th1 = np.arctan2(PSTAR_PI * np.sqrt(1 - ct1**2),
                     g1 * (PSTAR_PI * ct1 + b1 * EMU_PI))
    # mu -> e nu nu, exact boost of a massless neutrino
    e_mu = np.maximum(e_mu, M_MU * (1 + 1e-7))
    g2 = e_mu / M_MU
    b2 = np.sqrt(np.maximum(1 - 1 / g2**2, 0.0))
    ct2 = rng.uniform(-1, 1, n)
    xs = _michel_x(rng, n, "nue" if "nue" in species else "numu")
    e_nu = g2 * (0.5 * M_MU * xs) * (1 + b2 * ct2)
    th2a = np.arctan2(np.sqrt(1 - ct2**2), g2 * (ct2 + b2))
    tot = theta2_interp(e_pi, e_m, t2) + th1**2 + th2a**2
    if bending:
        sb = mb.bending_angle(b_gauss)
        tot = tot + sb**2 * np.clip(
            mb.decay_in_flight_fraction(e_mu, zenith_deg=zenith_deg), 0, 1)
    return _sigma_binned(e_nu, tot, w, e_query)


def pion_alpha_pdf(e_grid, alpha_deg, n=3_000_000, seed=4, scale=1.0,
                   kernels="k_spliced.npz", primary="H3a", floor=800):
    """Sampled nu angular distribution W[nE, n_alpha] from the generator's full
    (x_L, theta) pion kernel + exact pi->mu nu decay (no Gaussian assumption).

    Cells of the 2D kernel are sampled with H3a-flux-weighted projectile rows,
    the decay is exact two-body, and the hadronic and decay angles are combined
    on the sphere with a random azimuth. Rows are per-``e_grid`` energy
    histograms over ``alpha_deg`` (weights include the measure); rows with fewer
    than ``floor`` samples are zeroed (the cone then falls back to the Gaussian
    second-moment kernel). ``scale`` stretches the angles (NA61 +-12% pull)."""
    import crflux.models as crf

    d = dict(np.load(kernels))
    pe, xe, te, K = d["proj_energies"], d["xl_edges"], d["theta_edges"], d["kernel"]
    pm = crf.HillasGaisser2012(primary)
    wrow = pm.tot_nucleon_flux(pe) * pe  # flux weight per projectile row
    prob = K * wrow[:, None, None]
    prob = (prob / prob.sum()).ravel()
    rng = np.random.default_rng(seed)
    idx = rng.choice(prob.size, size=n, p=prob)
    ip, ix, it = np.unravel_index(idx, K.shape)
    u1, u2 = rng.random(n), rng.random(n)
    x = xe[ix] + u1 * (xe[ix + 1] - xe[ix])
    th_had = np.deg2rad(te[it] + u2 * (te[it + 1] - te[it]))
    e_mes = np.maximum(x * pe[ip], 0.15)
    ct = rng.uniform(-1, 1, n)
    gamma = e_mes / M_PI
    e_nu = gamma * ESTAR_PI * (1 + ct)
    th_dec = np.arctan2(np.sin(np.arccos(ct)), gamma * (ct + 1.0))
    phi = rng.uniform(0, 2 * np.pi, n)  # random relative azimuth on the sphere
    cth = np.cos(th_had) * np.cos(th_dec) - np.sin(th_had) * np.sin(th_dec) * np.cos(
        phi
    )
    th = np.degrees(np.arccos(np.clip(cth, -1, 1))) * scale
    # histogram per e_grid point (nearest in log E) over the alpha grid
    ae = np.concatenate([[0.0], 0.5 * (alpha_deg[1:] + alpha_deg[:-1]), [180.0]])
    le = np.log(e_grid)
    edges = np.concatenate(
        [[-np.inf], 0.5 * (le[1:] + le[:-1]), [np.inf]]
    )
    ie = np.searchsorted(edges, np.log(np.maximum(e_nu, 1e-9))) - 1
    W = np.zeros((len(e_grid), len(alpha_deg)))
    for k in range(len(e_grid)):
        m = ie == k
        if m.sum() < floor:
            continue  # thin statistics -> Gaussian fallback in the cone
        W[k], _ = np.histogram(th[m], bins=ae)
    return W


def channel_fractions(e_query, flavour="numu", cache="channel_fractions.npz",
                      zenith_deg=0.0):
    """Energy-dependent flux fractions f_pi, f_k, f_mu from MCEq (cached).

    ``flavour`` is "numu" or "nue". For nu_e the direct pi channel is helicity-
    suppressed (~0) and the flux is dominated by muon decay.

    ``zenith_deg`` (default 0 = vertical) selects the zenith at which the MCEq
    cascade is solved. This matters: the muon-decay fraction ``f_mu`` rises
    strongly toward the horizon (longer slant path -> more decay in flight), so
    applying the vertical value at all zeniths under-weights the (wide)
    muon-decay production cone exactly where it is largest. Fractions are cached
    per zenith in ``<cache>`` under a ``<key>_<zen>`` naming scheme; the vertical
    (0 deg) entry stays backward-compatible with the original flat layout.
    """
    import os

    ztag = f"{float(zenith_deg):.0f}"
    suffix = "" if ztag == "0" else f"_z{ztag}"

    def _has(d, fl):
        return all(f"{k}_{fl}{suffix}" in d for k in ("pi", "k", "mu"))

    d = dict(np.load(cache)) if os.path.exists(cache) else {}
    if d and "tag" in d and str(d["tag"]) != "SIBYLL23D_HillasGaisser2012-H3a":
        raise ValueError(
            f"{cache} was built for {d['tag']}; delete it to regenerate"
        )
    if not (d and _has(d, flavour)):
        import importlib.util  # noqa: F401  (MCEq config touches importlib.util)
        import mceq_config as cfg

        cfg.e_min = 0.1
        from MCEq.core import MCEqRun
        import crflux.models as crf

        mc = MCEqRun(
            interaction_model="SIBYLL23D",
            primary_model=(crf.HillasGaisser2012, "H3a"),
            theta_deg=float(zenith_deg),
        )
        mc.solve()
        e = mc.e_grid
        d.setdefault("e", e)
        d["tag"] = "SIBYLL23D_HillasGaisser2012-H3a"
        # Include the ANTI-species: the muon-decay fraction differs between a
        # neutrino and its antineutrino (the muon charge ratio is ~1.27), and by
        # lepton-flavour conservation the neutrino species already fixes the parent
        # muon charge -- mu_numu is 100% from mu-, mu_antinumu 100% from mu+
        # (verified: the opposite combinations are identically zero). So reading
        # both gives the charge-resolved fractions with no extra machinery.
        for fl in ("numu", "nue", "antinumu", "antinue"):
            tot = mc.get_solution(f"total_{fl}", 0)
            for k in ("pi", "k", "mu"):
                d[f"{k}_{fl}{suffix}"] = mc.get_solution(f"{k}_{fl}", 0) / np.maximum(
                    tot, 1e-300
                )
        np.savez(cache, **d)
    le = np.log(d["e"])
    return {
        k: np.interp(np.log(e_query), le, d[f"{k}_{flavour}{suffix}"])
        for k in ("pi", "k", "mu")
    }


def numu_fractions(e_query, **kw):
    """Backward-compatible nu_mu fractions (see :func:`channel_fractions`)."""
    return channel_fractions(e_query, flavour="numu", **kw)


if __name__ == "__main__":
    e = np.array([0.3, 0.5, 1.0, 2.0, 3.0, 10.0])
    sh = channel_shapes(e)
    fr = numu_fractions(e)
    print("Generator-sourced production angle sigma_c(E_nu) [deg] + MCEq fractions:")
    print("  E[GeV]  sig_pi  sig_K  | f_pi  f_K  f_mu")
    for i, ee in enumerate(e):
        print(
            f"  {ee:6.2f}  {sh['pi'][i]:6.2f} {sh['k'][i]:6.2f}  "
            f"| {fr['pi'][i]:.2f} {fr['k'][i]:.2f} {fr['mu'][i]:.2f}"
        )
    print("\nsig_pi is the flavour-independent excess kernel (offaxis_mc); the")
    print("excess is a pion-production effect inherited by all neutrinos. Meson")
    print("angle: generator moments (NA61-validated); decay: exact two-body.")
