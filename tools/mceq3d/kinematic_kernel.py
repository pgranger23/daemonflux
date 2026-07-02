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

# charge-combined generator moment files per meson species (chromo: UrQMD+SIBYLL)
_MOMENTS = {
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


def channel_fractions(e_query, flavour="numu", cache="channel_fractions.npz"):
    """Energy-dependent flux fractions f_pi, f_k, f_mu from MCEq (cached).

    ``flavour`` is "numu" or "nue". For nu_e the direct pi channel is helicity-
    suppressed (~0) and the flux is dominated by muon decay."""
    import os

    if not os.path.exists(cache):
        import importlib.util  # noqa: F401  (MCEq config touches importlib.util)
        import mceq_config as cfg

        cfg.e_min = 0.1
        from MCEq.core import MCEqRun
        import crflux.models as crf

        mc = MCEqRun(
            interaction_model="SIBYLL23D",
            primary_model=(crf.HillasGaisser2012, "H3a"),
            theta_deg=0.0,
        )
        mc.solve()
        e = mc.e_grid
        out = {"e": e, "tag": "SIBYLL23D_HillasGaisser2012-H3a"}
        for fl in ("numu", "nue"):
            tot = mc.get_solution(f"total_{fl}", 0)
            for k in ("pi", "k", "mu"):
                out[f"{k}_{fl}"] = mc.get_solution(f"{k}_{fl}", 0) / np.maximum(
                    tot, 1e-300
                )
        np.savez(cache, **out)
    d = np.load(cache)
    if "tag" in d and str(d["tag"]) != "SIBYLL23D_HillasGaisser2012-H3a":
        raise ValueError(
            f"{cache} was built for {d['tag']}; delete it to regenerate"
        )
    le = np.log(d["e"])
    return {
        k: np.interp(np.log(e_query), le, d[f"{k}_{flavour}"]) for k in ("pi", "k", "mu")
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
