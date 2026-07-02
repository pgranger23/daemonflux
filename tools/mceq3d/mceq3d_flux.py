"""mceq3d_flux -- a trustable, absolute, directional atmospheric-neutrino flux.

This is the production engine: it returns the **absolute** flux
``Phi(E, cos zenith, azimuth)`` for all four species (numu, antinumu, nue,
antinue) at a detector site, down to ~0.5 GeV, built so that every piece comes
from a *trusted* source and the 3D corrections are *validated against Honda*.

Construction (each factor trusted / validated)
---------------------------------------------
``Phi_3D = Phi_base(E,|cosZ|,s) * E_off(E,cosZ) * G_s(E,R_c(cosZ,az)) * S(E)``

* **Phi_base** -- MCEq (or daemonflux) solved per zenith with the **curved
  atmosphere**: absolute normalization, flavour/charge content, spectra, and the
  sec(theta) horizon enhancement (1D-per-direction-with-curvature).
* **E_off(E, cosZ)** -- the **first-principles off-axis 3D-production factor**
  ``Phi_3D/Phi_1D`` (:meth:`offaxis_factor`, ``offaxis=True``): the complete
  genuine-3D/1D ratio that both redistributes flux in zenith and produces the
  sub-GeV near-horizon excess (horizon/vertical ~1.8 at 0.3 GeV). Built in
  `offaxis_mc.py` from MCEq per-parent depth-resolved production (pi-geometry +
  kaon term), the curved-atmosphere slant geometry, and the **sampled** angular
  distribution from the generator's (x_L, theta) pion kernel folded with exact
  decay (Gaussian sigma_K for the kaon term). **No reference flux is used**;
  validated to reproduce Honda and Bartol (numu and nue) to their mutual ~5-15%.
  ->1 at high E and at the vertical. Off (=1) by default.
* **G_s(E, R_c)** -- the geomagnetic suppression = ``MCEq(primary cut at R_c) /
  MCEq(full)``, the **cascade-correct** response to removing sub-cutoff primaries
  (NO ``x_eff`` hack); ~zenith-independent, interpolated on a small R_c grid.
* **R_c(cosZ, az)** -- the first-principles **back-traced full-IGRF cutoff**
  (:mod:`geomag_backtrace`, validated: Kamioka 11.3 GV).
* **S(E)** -- optional solar-modulation factor (:meth:`solar_factor`).

With ``offaxis=True`` this is the complete first-principles 3D construction (curved
per-zenith cascade + off-axis 3D production + geomagnetic + solar), validated
absolutely against Honda/Bartol; ``offaxis=False`` drops E_off (the fast
factorised path, ~1.8x lower near the horizon sub-GeV). The legacy ``full_3d``
option applies only the flux-conserving redistribution R (:meth:`angular_factor`)
and is superseded by ``offaxis`` (do not combine them).

The rigidity cut is applied per-nucleus to the primary nucleons: the proton flux
is split into **free protons** (A/Z=1, R=E) and **bound protons** (in nuclei,
A/Z~2, R~2E) via MCEq's own p,n fluxes (free p = p-n by isospin), and neutrons are
all bound (R~2E).

The 1D base can be raw MCEq (default) or daemonflux's **muon-calibrated** flux
(``base_model="daemonflux"``) for a data-anchored absolute normalization.

Validation (`validate_honda.py`, and ``--validate`` here): the absolute Phi(E,
cosZ, az) for numu and nue matches the Honda HKKM2014 Kamioka tables in
normalization, zenith (sec theta) and azimuth (East-West).

Full sky
--------
* **Down-going (cosZ >= 0):** detector geomagnetic cutoff.
* **Up-going (cosZ < 0):** the production is up/down symmetric (same atmosphere at
  |cosZ|), but the geomagnetic cutoff is evaluated at the **far-side production
  point** (:func:`farside_production`) -- a global geomagnetic treatment, so the
  up-going hemisphere is included (essential for oscillation analyses).

Trust boundary (documented, not hidden)
---------------------------------------
* Up-going uses a single representative far-side production point per direction
  (the production region has finite extent; this is the leading geometric term).
* Nuclei: per-nucleus rigidity via the free/bound split (above); the bound part
  uses <A/Z>=2.0 (Fe is 2.08, a ~1% sub-component).
* Inter-direction streaming residual (~1-2%) and muon-bending E-W (charge-split,
  ~3 deg sub-GeV) are separate small corrections (see `spherical_streaming`,
  `muon_bending`).

Run::

    python mceq3d_flux.py --validate --plot
"""

from __future__ import annotations

import argparse
import hashlib
import os

import numpy as np

# Site-independent G_s and per-site cutoff maps are cached here (see solve(use_cache)).
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flux_cache")

CM2_PER_M2 = 1.0e4  # MCEq flux is per cm^2; Honda/this engine report per m^2
SPECIES = ("total_numu", "total_antinumu", "total_nue", "total_antinue")
SP_LABEL = {
    "total_numu": "numu",
    "total_antinumu": "antinumu",
    "total_nue": "nue",
    "total_antinue": "antinue",
}


def _transmission(e_grid, rc_gv, penumbra=0.5, az_over_z=1.0):
    """Smooth rigidity-cutoff transmission at R = az_over_z * E (erf step)."""
    from scipy.special import erf

    r = az_over_z * np.maximum(e_grid, 1e-9)
    return 0.5 * (1.0 + erf((np.log(r) - np.log(rc_gv)) / (np.sqrt(2) * penumbra)))


def farside_production(lat, lon, cos_zenith, azimuth, h_prod_km=20.0):
    """Far-side production point for an UP-going arrival (cos_zenith < 0).

    Up-going neutrinos are produced on the *opposite* side of the Earth and travel
    straight through it. The atmospheric production is up/down symmetric (same
    atmosphere at |cosZ|), but the **geomagnetic cutoff is set at the far-side
    production point**, not at the detector. This returns that point's
    ``(lat, lon)`` and the local primary arrival ``(cosZ, azimuth)`` there, so the
    detector's up-going geomagnetics can be evaluated globally.
    """
    from geomag_backtrace import _local_frame, arrival_direction, RE

    up = _local_frame(lat, lon)[0]
    P = RE * up
    d = arrival_direction(
        lat, lon, np.degrees(np.arccos(np.clip(cos_zenith, -1, 1))), azimuth
    )  # neutrino velocity (up-going for cosZ<0)
    Rh = RE + h_prod_km * 1e3
    Pd = P @ d
    s = Pd + np.sqrt(max(Pd**2 + (Rh**2 - RE**2), 0.0))  # far-side intersection
    Q = P - s * d
    qhat = Q / np.linalg.norm(Q)
    lat_q = np.degrees(np.arcsin(np.clip(qhat[2], -1, 1)))
    lon_q = np.degrees(np.arctan2(qhat[1], qhat[0]))
    upq, northq, eastq = _local_frame(lat_q, lon_q)
    src = -d  # primary source direction at Q
    cosz_q = float(src @ upq)
    horiz = src - cosz_q * upq
    az_q = np.degrees(np.arctan2(horiz @ eastq, horiz @ northq)) % 360.0
    return lat_q, lon_q, cosz_q, az_q


class MCEq3DFlux:
    """Absolute directional flux engine (full sky: down-going + up-going)."""

    def __init__(
        self,
        interaction_model="SIBYLL23D",
        primary=("HillasGaisser2012", "H3a"),
        e_min=0.1,
        atmosphere=None,
        base_model="mceq",
        daemonflux_location="generic",
        hybrid_e0=1.7,
    ):
        """``atmosphere`` is an MCEq ``density_model`` tuple. Default ``None`` keeps
        MCEq's realistic **CORSIKA US-Standard** layered profile (NOT the isothermal
        exponential used only in the `spherical_cascade` research demo). For
        seasonal/site tracking pass e.g. ``("MSIS00", ("SoudanMine", "January"))``.

        ``base_model`` selects the 1D base the geomagnetic factor multiplies:
        ``"mceq"`` (default) uses raw MCEq; ``"daemonflux"`` uses daemonflux's
        **muon-calibrated, data-anchored** 1D flux (numuflux/nueflux split by the
        ratios), which removes the ~10-20% hadronic-model normalization offset.
        """
        import crflux.models as crf
        from MCEq.core import MCEqRun
        import mceq_config as config

        self.base_model = base_model
        # hybrid-blend centre [GeV]. Physics window: bounded above by the
        # daemonflux muon-calibration floor mapped to neutrinos
        # (E_mu >= 5 GeV -> E_nu ~ E_mu/3 ~ 1.7 GeV, above which the muon
        # calibration fully constrains the flux) and below by where the
        # GSF-anchored MCEq base is validated (~0.15 GeV). The delivered
        # ratios are insensitive to E0 within this window (<=6% below 1 GeV,
        # scan in docs); the default 1.7 GeV IS the derived calibration floor
        # (fixed before the scan) -- NOT tuned to any reference.
        self._hybrid_e0 = float(hybrid_e0)
        # identity for the G_s disk cache (G_s depends only on these + rc_grid, cz_ref)
        self._tag = (
            f"{interaction_model}_{primary[0]}-{primary[1]}_emin{e_min:g}"
            f"_atm{'std' if atmosphere is None else str(atmosphere)}"
        )
        self._df = None
        if base_model in ("daemonflux", "hybrid"):
            from daemonflux import Flux

            self._df = Flux(location=daemonflux_location)
        config.e_min = e_min
        pm = (getattr(crf, primary[0]), primary[1])
        self.mceq = MCEqRun(
            interaction_model=interaction_model, primary_model=pm, theta_deg=0.0
        )
        if atmosphere is not None:
            self.mceq.set_density_model(atmosphere)
        self.e = self.mceq.e_grid
        self._phi0_std = self.mceq._phi0.copy()
        p = self.mceq.pman[(2212, 0)]
        n = self.mceq.pman[(2112, 0)]
        self._p_sl = slice(p.lidx, p.uidx)
        self._n_sl = slice(n.lidx, n.uidx)
        # Per-nucleus rigidity bookkeeping. The geomagnetic cutoff is on RIGIDITY
        # R = (A/Z)*E_nucleon, so free protons (A/Z=1) and bound nucleons
        # (He/CNO/Fe, A/Z~2) are cut at different energies. By isospin the bound
        # protons ~ the neutron flux, so from MCEq's own p/n nucleon fluxes:
        #   free protons  = p - n  (A/Z=1),   bound protons = n (A/Z~2).
        p_arr = self._phi0_std[self._p_sl]
        n_arr = self._phi0_std[self._n_sl]
        with np.errstate(invalid="ignore", divide="ignore"):
            self._f_free = np.where(
                p_arr > 0, np.clip((p_arr - n_arr) / p_arr, 0.0, 1.0), 1.0
            )
        # <A/Z> of bound nucleons, **nucleon-flux-weighted over the real primary
        # composition** (He/CNO/Si A/Z=2, Fe 2.077) instead of a fixed 2.0, so the
        # ~0.2-0.5% Fe sub-component is included and energy-dependent (Fe rises with
        # E). Per species the nucleon flux at per-nucleon energy E is A^2*Phi(A*E).
        cr = pm[0](pm[1])
        num = np.zeros_like(self.e)
        den = np.zeros_like(self.e)
        for cid in cr.nucleus_ids:
            Z, A = cr.Z_A(cid)
            if A <= 1:
                continue  # free protons handled separately (A/Z=1)
            w = A * A * np.ravel(cr.nucleus_flux(cid, A * self.e))
            num += (A / Z) * w
            den += w
        self._az_bound = np.where(den > 0, num / den, 2.0)

    # -- base: MCEq per zenith, curved atmosphere, no geomag --
    def base(self, cos_zeniths):
        """Phi_MCEq[species, |cosZ|, E] in /(m^2 s sr GeV) (curved, no geomag).

        Uses ``|cosZ|`` (production is up/down symmetric: an up-going neutrino is
        produced on the far side at the conjugate down-going slant).
        """
        if self.base_model == "daemonflux":
            return self._base_daemonflux(cos_zeniths)
        self.mceq._phi0[:] = self._phi0_std
        out = {s: np.zeros((len(cos_zeniths), len(self.e))) for s in SPECIES}
        for i, cz in enumerate(cos_zeniths):
            self.mceq.set_theta_deg(np.degrees(np.arccos(np.clip(abs(cz), 1e-3, 1))))
            self.mceq.solve()
            for s in SPECIES:
                out[s][i] = self.mceq.get_solution(s, 0) * CM2_PER_M2
        if self.base_model == "hybrid":
            # Muon-calibrated daemonflux where the calibration is valid
            # (E >~ 1 GeV), data-anchored MCEq below (recommended with the
            # GSF primary: AMS-02/BESS/PAMELA-fitted, which fixes the sub-GeV
            # primary that H3a extrapolates poorly). Smooth log-blend around
            # E0=0.8 GeV (one octave wide): the two bases agree with the 3D
            # references in complementary domains (GSF-MCEq 0.15-0.5 GeV,
            # daemonflux >=1 GeV), so the blend tracks the better one.
            df = self._base_daemonflux(cos_zeniths)
            w = hybrid_weight(self.e, e0=self._hybrid_e0)
            for s in SPECIES:
                out[s] = (1.0 - w)[None, :] * out[s] + w[None, :] * df[s]
        return out

    def _base_daemonflux(self, cos_zeniths):
        """Muon-calibrated 1D base from daemonflux [/(m^2 s sr GeV)], all flavours.

        daemonflux reports E^3-weighted *sums* (numuflux = nu_mu+nubar_mu) and the
        ratios (numuratio = nu_mu/nubar_mu); we de-weight by E^3, convert cm->m,
        and split into species with the ratios. Uses |cosZ| (up/down symmetric).

        Validity: this base matches Honda to ~10 % for E >~ 1 GeV but
        **over-predicts below ~0.3 GeV** (up to ~2x at 0.1 GeV), where it
        extrapolates past daemonflux's muon-calibration region; there the raw MCEq
        base is closer to Honda (see `base_comparison.py`). For sub-0.3-GeV work,
        cross-check both bases.
        """
        e = self.e
        m = e <= 1.0e9  # daemonflux splines are valid to ~1e9 GeV (>> 3D regime)
        ev = e[m]
        out = {s: np.zeros((len(cos_zeniths), len(e))) for s in SPECIES}
        for i, cz in enumerate(cos_zeniths):
            zen = float(np.degrees(np.arccos(np.clip(abs(cz), 1e-3, 1))))
            tot_mu = self._df.flux(ev, zen, "numuflux") / ev**3 * CM2_PER_M2
            r_mu = self._df.flux(ev, zen, "numuratio")  # nu_mu / nubar_mu
            tot_e = self._df.flux(ev, zen, "nueflux") / ev**3 * CM2_PER_M2
            r_e = self._df.flux(ev, zen, "nueratio")
            out["total_numu"][i, m] = tot_mu * r_mu / (1.0 + r_mu)
            out["total_antinumu"][i, m] = tot_mu / (1.0 + r_mu)
            out["total_nue"][i, m] = tot_e * r_e / (1.0 + r_e)
            out["total_antinue"][i, m] = tot_e / (1.0 + r_e)
        return out

    # daemonflux quantity name per engine species (conventional charge/flavour)
    _DF_Q = {
        "total_numu": "numu",
        "total_antinumu": "antinumu",
        "total_nue": "nue",
        "total_antinue": "antinue",
    }

    def _daemonflux_relerr(self, cos_zeniths, only_hadronic=False):
        """Fractional calibration uncertainty sigma/Phi[species, cosZ, E] from
        daemonflux's nuisance-parameter covariance (its ``error()``).

        The directional flux is ``Phi_df * G * S`` with ``G, S`` independent of the
        daemonflux nuisance parameters, so the *fractional* error is preserved and
        propagates unchanged: ``sigma(Phi_3D)/Phi_3D = sigma(Phi_df)/Phi_df``.
        ``only_hadronic`` isolates the hadronic-production part of the covariance.
        """
        e = self.e
        m = e <= 1.0e9
        ev = e[m]
        out = {s: np.zeros((len(cos_zeniths), len(e))) for s in SPECIES}
        for i, cz in enumerate(cos_zeniths):
            zen = float(np.degrees(np.arccos(np.clip(abs(cz), 1e-3, 1))))
            for s in SPECIES:
                q = self._DF_Q[s]
                f = np.asarray(self._df.flux(ev, zen, q))
                er = np.asarray(self._df.error(ev, zen, q, only_hadronic=only_hadronic))
                with np.errstate(invalid="ignore", divide="ignore"):
                    out[s][i, m] = np.where(f > 0, er / f, 0.0)
        return out

    def calib_jacobian(self, cos_zeniths):
        """Full correlated calibration Jacobian for propagation into a fit.

        Returns ``dict(params, corr, cov, jac)`` where ``jac[species]`` has shape
        ``(n_param, cosZ, E)`` and is the **fractional** flux response to a +1-sigma
        pull of each daemonflux nuisance parameter, ``R_i = Phi(pull_i=+1)/Phi - 1``.
        Because ``G, S`` are parameter-independent, ``R_i`` is identical for the base
        and the directional flux (and azimuth-independent). ``corr`` is the parameter
        correlation matrix (unit-variance pulls). The directional-flux calibration
        covariance for any direction is then ``(Phi*R)^T corr (Phi*R)``
        (see :func:`calib_covariance`); by construction ``sqrt(R^T corr R)`` equals
        the fractional ``error()``. Enables an oscillation fit to carry daemonflux's
        *correlated* nuisance parameters on the 3D flux, not just the 1-sigma band.
        """
        if self.base_model != "daemonflux":
            raise ValueError("calib_jacobian requires base_model='daemonflux'")
        names = list(self._df.params.known_parameters)
        cov = np.asarray(self._df.params.cov)
        sig = np.sqrt(np.diag(cov))
        corr = cov / np.outer(sig, sig)
        e = self.e
        m = e <= 1.0e9
        ev = e[m]
        jac = {s: np.zeros((len(names), len(cos_zeniths), len(e))) for s in SPECIES}
        for iz, cz in enumerate(cos_zeniths):
            zen = float(np.degrees(np.arccos(np.clip(abs(cz), 1e-3, 1))))
            for s in SPECIES:
                q = self._DF_Q[s]
                c = np.asarray(self._df.flux(ev, zen, q))
                for ip, name in enumerate(names):
                    sh = np.asarray(self._df.flux(ev, zen, q, params={name: 1.0}))
                    with np.errstate(invalid="ignore", divide="ignore"):
                        jac[s][ip, iz, m] = np.where(c > 0, sh / c - 1.0, 0.0)
        return dict(params=names, corr=corr, cov=cov, jac=jac)

    # -- geomagnetic response G_s(E, R_c) from cascade-correct cut/full --
    def geomag_response(self, rc_grid, cz_ref=1.0, cache_dir=None):
        """G[species, R_c, E] = MCEq(primary cut at R_c)/MCEq(full) at one zenith.

        ``G_s`` depends only on the interaction model / primary / atmosphere / e_min
        (via ``self._tag``), the rigidity grid and ``cz_ref`` -- it is **site- and
        (validated) zenith-independent**. With ``cache_dir`` set it is memoised to
        ``<cache_dir>/gs_*.npz`` and reused across sites and calls.
        """
        rc_grid = np.asarray(rc_grid, float)
        fpath = None
        if cache_dir is not None:
            h = hashlib.md5(
                f"{self._tag}|{cz_ref:.4f}|{rc_grid.tobytes()}".encode()
            ).hexdigest()[:16]
            fpath = os.path.join(cache_dir, f"gs_{h}.npz")
            if os.path.exists(fpath):
                d = np.load(fpath)
                if d["e"].shape == self.e.shape and np.allclose(d["e"], self.e):
                    return {s: d[s] for s in SPECIES}, rc_grid
        self.mceq.set_theta_deg(np.degrees(np.arccos(np.clip(cz_ref, 1e-3, 1))))
        self.mceq._phi0[:] = self._phi0_std
        self.mceq.solve()
        full = {s: self.mceq.get_solution(s, 0).copy() for s in SPECIES}
        G = {s: np.ones((len(rc_grid), len(self.e))) for s in SPECIES}
        for j, rc in enumerate(rc_grid):
            # proper per-nucleus rigidity: split the proton flux into free
            # (A/Z=1) and bound (A/Z~2); neutrons are all bound.
            t1 = _transmission(self.e, rc, az_over_z=1.0)
            t2 = _transmission(self.e, rc, az_over_z=self._az_bound)
            tp = self._f_free * t1 + (1.0 - self._f_free) * t2
            tn = t2
            self.mceq._phi0[:] = self._phi0_std
            self.mceq._phi0[self._p_sl] *= tp
            self.mceq._phi0[self._n_sl] *= tn
            self.mceq.solve()
            for s in SPECIES:
                cut = self.mceq.get_solution(s, 0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    G[s][j] = np.where(full[s] > 0, cut / full[s], 1.0)
        self.mceq._phi0[:] = self._phi0_std
        if fpath is not None:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez(fpath, e=self.e, **{s: G[s] for s in SPECIES})
        return G, rc_grid

    # -- solar modulation: force-field on the primary -> neutrino-energy factor --
    def _modulate_phi0(self, phi_gv):
        """Force-field (Gleeson-Axford) modulation of the primary nucleon _phi0.

        Potential ``phi_gv`` [GV] shifts each nucleon by its per-nucleon energy loss
        Phi = (Z/A)*phi and applies the flux Jacobian. Free protons (Z/A=1) lose the
        full phi; bound nucleons (Z/A=1/<A/Z>~0.5) lose half -- the standard
        rigidity-dependent modulation. ``phi_gv=0`` returns the baseline unchanged.
        """
        m_n = 0.938272
        E = self.e
        out = self._phi0_std.copy()

        def ff(f0, z_over_a):
            phi = z_over_a * phi_gv
            lo = np.log(np.maximum(f0, 1e-300))
            # Negative phi = DE-modulation (e.g. to a solar-minimum epoch):
            # E+phi can drop below the grid; clip to the grid floor. This is
            # harmless for the neutrino flux: sub-threshold primaries
            # (E_kin below the single-pion production threshold ~0.29 GeV)
            # cannot produce the E_nu >= 0.1 GeV flux modelled here.
            es = np.clip(E + phi, E[0], None)
            shifted = np.exp(np.interp(np.log(es), np.log(E), lo))
            jac = (E * (E + 2 * m_n)) / (es * (es + 2 * m_n))
            return shifted * jac

        p = self._phi0_std[self._p_sl]
        n = self._phi0_std[self._n_sl]
        zoa_bound = 1.0 / self._az_bound  # Z/A of bound nucleons (~0.5)
        out[self._p_sl] = ff(p * self._f_free, 1.0) + ff(
            p * (1 - self._f_free), zoa_bound
        )
        out[self._n_sl] = ff(n, zoa_bound)
        return out

    def solar_factor(self, phi_gv, cz_ref=1.0):
        """Neutrino-energy solar-modulation factor S(E) = numu(modulated)/numu(base).

        Runs the cascade with the force-field-modulated primary (`_modulate_phi0`)
        over the unmodulated one; the ratio maps the primary modulation to neutrino
        energy through the shower (species- and ~zenith-independent, like G_s), so it
        multiplies whichever 1D base is used. ``phi_gv`` is *relative to the H3a
        baseline* (0 = baseline); the physical solar-cycle effect is the difference
        between two potentials (e.g. ~0.4 GV solar-min vs ~1.0 GV solar-max).
        """
        if phi_gv == 0:
            return np.ones(len(self.e))
        self.mceq.set_theta_deg(np.degrees(np.arccos(np.clip(cz_ref, 1e-3, 1))))
        self.mceq._phi0[:] = self._phi0_std
        self.mceq.solve()
        ref = self.mceq.get_solution("total_numu", 0).copy()
        self.mceq._phi0[:] = self._modulate_phi0(phi_gv)
        self.mceq.solve()
        mod = self.mceq.get_solution("total_numu", 0)
        self.mceq._phi0[:] = self._phi0_std
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(ref > 0, mod / ref, 1.0)

    # -- genuine-3D production-angle redistribution R(E, cosZ) = Phi_3D/Phi_1D --
    def angular_factor(
        self, cos_zeniths, moments="m_spliced.npz", n_dense=41, n_mc=4000
    ):
        """Production-angle 3D redistribution factor R[species][cosZ, E].

        The 1D base is collinear (neutrino along the primary); in 3D a neutrino from
        direction n_o is produced along a spread of parent directions about it. We
        convolve the per-zenith base with the **NA61-validated** production-angle
        spread ``sigma_theta(E)`` (from the high-statistics kernel moments) on the
        sphere (`coupled_3d_flux.convolve_sphere`), flux-conserving. R->1 at high E
        (sigma->0); sub-GeV it is a ~1-2% zenith redistribution (slight horizon
        deficit / vertical excess). Site- and azimuth-independent; the curved-
        atmosphere sec-theta rise is already in the per-zenith base and is *not*
        touched here (no double counting).
        """
        from fokker_planck_3d import load_theta2, sigma_theta_vs_energy
        from coupled_3d_flux import convolve_sphere

        e = self.e
        e_sig, theta2 = load_theta2(moments)
        sig = np.deg2rad(sigma_theta_vs_energy(e_sig, theta2, e, zenith_deg=0.0))
        dense = np.linspace(-1.0, 1.0, n_dense)  # mirrored full sphere
        base = self.base(dense)  # per species (n_dense, n_E), symmetric in cosZ
        out_cos = np.clip(np.abs(np.asarray(cos_zeniths, float)), 1e-3, 1.0)
        sigma_EZ = np.tile(sig, (len(out_cos), 1))
        eidx = list(range(len(e)))
        R = {}
        for s in SPECIES:
            phi3d = convolve_sphere(
                e, dense, base[s], sigma_EZ, out_cos, eidx, n_mc=n_mc
            )
            phi1d = np.array(
                [np.interp(out_cos, dense, base[s][:, ie]) for ie in eidx]
            ).T
            with np.errstate(invalid="ignore", divide="ignore"):
                R[s] = np.where(phi1d > 0, phi3d / phi1d, 1.0)  # (n_cos, n_E)
        return R

    def offaxis_factor(self, cos_zeniths, path=None, shape_only=False,
                       which="E_off"):
        """First-principles off-axis 3D-production factor E_off[species][cosZ, E].

        The complete genuine-3D/1D production ratio: it both redistributes flux in
        zenith and produces the sub-GeV near-horizon excess (horizon/vertical ~1.8
        at 0.3 GeV) that the per-zenith cascade cannot. Derived *from first
        principles* in `offaxis_mc.py`: MCEq depth-resolved production p(X,E),
        curved-atmosphere slant depth, and the **pion production angle from the
        SIBYLL/UrQMD generator moments** (chromo, NA61-validated) folded with exact
        pi->mu nu decay. The excess is a pion-production-rate effect inherited by
        every daughter neutrino (muon-decay neutrinos included -- NOT a flat
        pedestal), so E_off is **flavour-independent**: one table for all species;
        the nu_e-vs-nu_mu flux difference lives entirely in the per-flavour base.
        No reference flux is used; validated to reproduce the Honda and Bartol
        nu_mu *and* nu_e zenith shapes to ~5% sub-GeV (`offaxis_mc.py --validate`).
        E_off->1 at high E; applied to ``|cosZ|``; =1 outside the tabulated E range
        (0.1-100 GeV -- below 0.1 GeV the excess is large and NOT modelled).

        ``shape_only`` (opt-in, off by default): divide out the vertical value,
        E_off(cosZ)/E_off(vert). This was once the default on the daemonflux base
        out of a double-counting worry, but the muon closure shows it is
        unnecessary: daemonflux's neutrino flux is genuinely 1D and E_off_mu ~ 1
        in its muon-calibration region (E_mu >~ 5 GeV), so the **full** factor is
        self-consistent with the calibration and tracks Honda better (the vertical
        sub-GeV flux lands at ~1.0x Honda with the full factor vs ~1.07x
        shape-only). Kept only for A/B comparison.

        ``which``: "E_off" (central) | "E_off_hi"/"E_off_lo" (NA61 +-12%
        pion-angle variants, used for the sigma_pi_NA61 covariance pull).
        """
        if path is None:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "offaxis_excess.npz")
        e = self.e
        cz = np.abs(np.asarray(cos_zeniths, float))
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found; run 'python offaxis_mc.py --build' first"
            )
        d = np.load(path)
        if "tag" in d:
            tag = str(d["tag"])
            if not self._tag.startswith(tag):
                # interaction model must match (the kernel and p(X,E) are its
                # yields); a primary-spectrum mismatch only reweights the ratio
                # mildly -> warn (E_off is primary-insensitive to first order).
                if not self._tag.startswith(tag.split("_")[0]):
                    raise ValueError(
                        f"offaxis table built for {tag} but engine is "
                        f"{self._tag}; rebuild with offaxis_mc.py --build"
                    )
                import warnings

                warnings.warn(
                    f"offaxis table primary ({tag}) differs from engine "
                    f"({self._tag}); E_off is a primary-insensitive ratio, "
                    "but rebuild for exactness."
                )
        Etab = d[which]  # (n_cz, n_Ee)
        Ecz, Ee = d["cz"], d["e"]
        if shape_only:
            Etab = Etab / Etab[np.argmax(Ecz)]  # normalise to the vertical row
        # interpolate E_off(cz, E) -> (len(cz), len(e)); 1 outside the tabulated E
        table = np.ones((len(cz), len(e)))
        lo, hi = Ee[0], Ee[-1]
        for k, en in enumerate(e):
            if en < lo or en > hi:
                continue
            col = np.array([np.interp(np.log(en), np.log(Ee), Etab[i]) for i in
                            range(len(Ecz))])
            table[:, k] = np.interp(cz, Ecz, col)
        return {s: table for s in SPECIES}  # same (geometric) factor for all species

    def cutoff_grid(
        self,
        lat,
        lon,
        cos_zeniths,
        azimuths,
        date,
        n_scan=14,
        r_lo=0.5,
        r_hi=20.0,
        cache_dir=None,
    ):
        """R_c[cosZ, az] [GV]: detector cutoff (down-going), far-side (up-going).

        Both hemispheres use a single batched trajectory back-trace. For up-going,
        the primary's velocity at the far-side production point equals the
        (straight-line) neutrino direction ``d``, so the cutoff is a back-trace
        from the production point ``Q`` with ``u0 = -d`` -- the global geomagnetic
        treatment, batched like :func:`geomag_backtrace.cutoff_map`.

        This is the dominant cost (trajectory integration). With ``cache_dir`` set
        the resulting map is memoised to ``<cache_dir>/rc_*.npz``, keyed by site,
        date, grid and scan parameters, so repeat evaluations are ~instant.
        """
        import geomag_backtrace as gb

        fpath = None
        if cache_dir is not None:
            dtag = date.isoformat() if hasattr(date, "isoformat") else str(date)
            cz_b = np.asarray(cos_zeniths).tobytes()
            az_b = np.asarray(azimuths).tobytes()
            tag = f"{lat:.4f}_{lon:.4f}_{dtag}_{cz_b}_{az_b}_{n_scan}_{r_lo}_{r_hi}"
            h = hashlib.md5(tag.encode()).hexdigest()[:16]
            fpath = os.path.join(cache_dir, f"rc_{h}.npz")
            if os.path.exists(fpath):
                d = np.load(fpath)
                if d["rc"].shape == (len(cos_zeniths), len(azimuths)):
                    return d["rc"]

        rc = np.zeros((len(cos_zeniths), len(azimuths)))
        down = cos_zeniths >= 0
        if np.any(down):
            zen = np.degrees(np.arccos(np.clip(cos_zeniths[down], 1e-3, 1)))
            rc[down] = gb.cutoff_map(
                lat, lon, date, zen, azimuths, n_scan=n_scan, r_lo=r_lo, r_hi=r_hi
            )
        ups = np.where(~down)[0]
        if len(ups):
            m_hat = gb.dipole_axis()
            rs = np.linspace(r_hi, r_lo, n_scan)
            r0, u0, R, idx = [], [], [], []
            for iz in ups:
                zdeg = np.degrees(np.arccos(np.clip(cos_zeniths[iz], -1, 1)))
                for ia, az in enumerate(azimuths):
                    latq, lonq, _, _ = farside_production(lat, lon, cos_zeniths[iz], az)
                    Q = (gb.RE + 20e3) * gb._local_frame(latq, lonq)[0]
                    d = gb.arrival_direction(lat, lon, zdeg, az)  # neutrino velocity
                    for Ri in rs:
                        r0.append(Q)
                        u0.append(-d)
                        R.append(Ri)
                    idx.append((iz, ia))
            allowed = gb.backtrace_vec(
                np.array(r0), np.array(u0), np.array(R), date, m_hat
            ).reshape(len(idx), n_scan)
            for k, (iz, ia) in enumerate(idx):
                forb = np.where(~allowed[k])[0]
                rc[iz, ia] = (
                    r_lo
                    if len(forb) == 0
                    else (
                        r_hi if forb[0] == 0 else 0.5 * (rs[forb[0] - 1] + rs[forb[0]])
                    )
                )
        if fpath is not None:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez(fpath, rc=rc)
        return rc

    def solve(
        self,
        lat,
        lon,
        cos_zeniths,
        azimuths,
        date=None,
        n_scan=12,
        rc_grid=None,
        zenith_dependent_geomag=True,
        use_cache=False,
        cache_dir=None,
        solar_modulation=0.0,
        with_calib_error=False,
        calib_hadronic_only=False,
        with_calib_jacobian=False,
        full_3d=False,
        moments="m_spliced.npz",
        offaxis=False,
        offaxis_shape_only=None,
        with_eoff_jacobian=False,
        solar_sigma_gv=0.0,
        with_base_spread=False,
    ):
        """Absolute Phi[species, cosZ, az, E] for a site (full sky).

        Down-going (cosZ>=0): detector geomagnetic cutoff. Up-going (cosZ<0):
        far-side production-point cutoff (global treatment, :func:`farside_production`).

        ``zenith_dependent_geomag`` (default **True**): recompute the suppression
        ratio ``G_s(E,R_c)`` at *every* zenith band, so no slant-depth
        approximation enters. G_s is site-independent and cached per (cz_ref,
        rc_grid), so the extra cost (one cascade pair per unique |cosZ|) is a
        one-time precompute. Set False to reuse the vertical-column G_s at all
        zeniths -- validated zenith-independent to <=2% (sub-GeV horizon) by
        `geomag_zenith_check.py` -- for a faster first (uncached) evaluation.

        ``use_cache``: memoise the two heavy ingredients to ``cache_dir`` (default
        ``flux_cache/`` next to this module) -- the **site-independent** ``G_s`` and
        the **per-site** cutoff map. The first call to a site pays the full cost;
        repeat calls (and other sites, for ``G_s``) load from disk in milliseconds.
        Matches the default path to interpolation precision (a fixed wide ``rc_grid``
        is used so one cached ``G_s`` serves every site without clamping). Off by
        default so the validated path is untouched.

        ``solar_modulation`` [GV]: force-field modulation potential applied to the
        primary (`solar_factor`), rescaling the flux toward low energy; 0 (default) =
        H3a baseline. The physical solar-cycle span is the *difference* between two
        potentials (~0.4 GV solar-min vs ~1.0 GV solar-max).

        ``with_calib_error`` (daemonflux base only): also return the propagated
        muon-**calibration** 1-sigma uncertainty from daemonflux's nuisance-parameter
        covariance -- ``result["flux_err"]`` (absolute) and ``["flux_relerr"]``
        (fractional). Since ``G`` and ``S`` are parameter-independent, the fractional
        error carries through unchanged. ``calib_hadronic_only`` isolates the
        hadronic-production component.

        ``offaxis`` (recommended for 3D): fold in the first-principles off-axis
        3D-production factor ``E_off(E, cosZ)`` (`offaxis_factor`) -- the complete
        genuine-3D/1D production ratio, which both redistributes flux in zenith and
        produces the sub-GeV near-horizon excess (horizon/vertical ~1.8 at 0.3 GeV).
        Derived from first principles (MCEq production + curved geometry + decay
        kinematics; no reference flux) and validated against Honda/Bartol to their
        mutual ~5-15%. This supersedes ``full_3d`` (the two must not be combined --
        that double-counts the production angle).

        ``full_3d`` (legacy): fold in only the flux-conserving production-angle
        **redistribution** ``R(E, cosZ)`` (`angular_factor`) -- the ~1-2% sub-GeV
        zenith redistribution without the net horizontal excess. Kept for
        comparison; use ``offaxis`` for the complete 3D flux. ``full_3d=False`` and
        ``offaxis=False`` (default) is the pure factorised path.

        ``offaxis_shape_only``: opt-in E_off/E_off(vertical) instead of the full
        factor (default ``None`` -> ``False``, i.e. the full factor on every
        base). daemonflux's neutrino flux is genuinely 1D, and the build-time
        muon closure (E_off with the muon kernel ~1 for E_mu >= 5 GeV,
        daemonflux's calibration region) shows the full 1D->3D factor does not
        double-count the calibration -- it tracks Honda better than shape-only
        (vertical sub-GeV ~1.0x Honda vs ~1.07x). Kept only for A/B studies.

        Additional nuisance pulls (appended to ``calib_params/corr/jac`` so
        `calib_covariance` carries the **full** uncertainty, and folded into
        ``flux_relerr``/``flux_err`` in quadrature when ``with_calib_error``):
        ``with_eoff_jacobian`` -- the hadronic E_off pull ``sigma_pi_NA61``
        (NA61 +-12% pion production angle; ~+-8% on the sub-GeV horizon excess);
        ``solar_sigma_gv`` -- a +1-sigma solar-potential pull of this size [GV];
        ``with_base_spread`` -- the fully-correlated base-model-choice pull
        (half log-spread MCEq vs daemonflux; the dominant sub-GeV systematic).
        """
        if offaxis and full_3d:
            raise ValueError(
                "offaxis supersedes full_3d (E_off already contains the "
                "production-angle redistribution); set only one."
            )
        cos_zeniths = np.asarray(cos_zeniths, float)
        azimuths = np.asarray(azimuths, float)
        base = self.base(cos_zeniths)

        import datetime as _dt

        date = date or _dt.datetime(2020, 1, 1)
        if use_cache and cache_dir is None:
            cache_dir = _CACHE_DIR
        rc_map = self.cutoff_grid(
            lat, lon, cos_zeniths, azimuths, date, n_scan, cache_dir=cache_dir
        )

        # Build the G(R_c) interpolation grid to *span the actual cutoff map*, so the
        # suppression is never clamped: a fixed floor (e.g. 2 GV) would apply spurious
        # suppression to low-cutoff directions/sites (polar R_c<2 GV) where G->1.
        if rc_grid is None:
            if use_cache:
                # fixed wide grid -> one cached G_s is reusable across all sites
                rc_grid = np.linspace(0.1, 20.0, 24)
            else:
                lo = max(0.1, float(np.min(rc_map)) * 0.9)
                hi = max(lo + 0.5, float(np.max(rc_map)) * 1.05)
                rc_grid = np.linspace(lo, hi, 12)
        if zenith_dependent_geomag:
            G_by_z = []
            for cz in cos_zeniths:
                Gz, rc_grid = self.geomag_response(
                    rc_grid, cz_ref=max(abs(cz), 1e-3), cache_dir=cache_dir
                )
                G_by_z.append(Gz)
        else:
            G0, rc_grid = self.geomag_response(rc_grid, cache_dir=cache_dir)
            G_by_z = [G0] * len(cos_zeniths)

        # solar modulation: neutrino-energy factor S(E), applied to all species/dirs
        smod = self.solar_factor(solar_modulation) if solar_modulation else 1.0
        # legacy flux-conserving production-angle redistribution R[species][cosZ,E]
        R3d = self.angular_factor(cos_zeniths, moments=moments) if full_3d else None
        # first-principles complete off-axis 3D-production factor E_off[cosZ,E].
        # The FULL factor is applied on every base (default). daemonflux's
        # neutrino flux is genuinely 1D, and the muon-calibration region
        # (E_mu >~ 5 GeV) has E_off_mu ~ 1 (build-time closure check), so the
        # full 1D->3D factor is self-consistent with the calibration and does
        # not double-count -- verified to track Honda better than the earlier
        # shape-only heuristic (kept only as an opt-in for A/B studies).
        if offaxis_shape_only is None:
            offaxis_shape_only = False
        Eoff = (
            self.offaxis_factor(cos_zeniths, shape_only=offaxis_shape_only)
            if offaxis
            else None
        )

        flux = {
            s: np.zeros((len(cos_zeniths), len(azimuths), len(self.e))) for s in SPECIES
        }
        for s in SPECIES:
            r3 = R3d[s] if R3d is not None else None
            eo = Eoff[s] if Eoff is not None else None
            for ia in range(len(azimuths)):
                for iz in range(len(cos_zeniths)):
                    g = _interp_rc(rc_map[iz, ia], rc_grid, G_by_z[iz][s])
                    f = base[s][iz] * g * smod
                    if r3 is not None:
                        f = f * r3[iz]
                    if eo is not None:
                        f = f * eo[iz]
                    flux[s][iz, ia] = f
        result = dict(
            e=self.e,
            cos_zeniths=cos_zeniths,
            azimuths=azimuths,
            flux=flux,
            base=base,
            cutoff=rc_map,
        )
        if with_calib_error:
            if self.base_model != "daemonflux":
                raise ValueError("with_calib_error requires base_model='daemonflux'")
            relerr = self._daemonflux_relerr(
                cos_zeniths, only_hadronic=calib_hadronic_only
            )
            result["flux_relerr"] = relerr  # sigma/Phi [species, cosZ, E]
        if with_calib_jacobian:
            cj = self.calib_jacobian(cos_zeniths)
            result["calib_params"] = cj["params"]  # 24 nuisance-parameter names
            result["calib_corr"] = cj["corr"]  # parameter correlation (n_par, n_par)
            result["calib_cov"] = cj["cov"]
            result["calib_jac"] = cj[
                "jac"
            ]  # R[species]=dPhi/Phi per +1sig [par,cosZ,E]

        # -- extra nuisance pulls (uncorrelated with the daemonflux calibration) --
        # Each is a fractional flux response R[species][cosZ, E] to a +1-sigma pull,
        # appended to calib_params/corr/jac so calib_covariance() carries the FULL
        # uncertainty (calibration + hadronic E_off + solar + base spread).
        extras = []  # (name, R[species])
        if with_eoff_jacobian:
            if not offaxis:
                raise ValueError("with_eoff_jacobian requires offaxis=True")
            E_hi = self.offaxis_factor(
                cos_zeniths, shape_only=offaxis_shape_only, which="E_off_hi"
            )
            r = {
                s: np.where(Eoff[s] > 0, E_hi[s] / Eoff[s] - 1.0, 0.0)
                for s in SPECIES
            }
            extras.append(("sigma_pi_NA61", r))
        if solar_sigma_gv:
            s_ref = smod if solar_modulation else np.ones(len(self.e))
            s_up = self.solar_factor(solar_modulation + solar_sigma_gv)
            r1 = np.where(s_ref > 0, s_up / s_ref - 1.0, 0.0)[None, :] * np.ones(
                (len(cos_zeniths), 1)
            )
            extras.append(("solar_phi", {s: r1 for s in SPECIES}))
        if with_base_spread:
            # evaluate the *other* base at the same zeniths; the half log-spread
            # is a single fully-correlated model-choice pull (base_comparison).
            if self.base_model == "daemonflux":
                saved = self.base_model
                self.base_model = "mceq"
                try:
                    other = self.base(cos_zeniths)
                finally:
                    self.base_model = saved
            else:
                if self._df is None:
                    from daemonflux import Flux

                    self._df = Flux(location="generic")
                other = self._base_daemonflux(cos_zeniths)
            r = {}
            for s in SPECIES:
                with np.errstate(invalid="ignore", divide="ignore"):
                    r[s] = np.where(
                        (base[s] > 0) & (other[s] > 0),
                        0.5 * np.log(other[s] / base[s]),
                        0.0,
                    )
            extras.append(("base_model_spread", r))
        if extras:
            names = list(result.get("calib_params", []))
            n0 = len(names)
            jac = result.get(
                "calib_jac",
                {s: np.zeros((0, len(cos_zeniths), len(self.e))) for s in SPECIES},
            )
            for name, r in extras:
                names.append(name)
                for s in SPECIES:
                    jac[s] = np.concatenate([jac[s], r[s][None]], axis=0)
            corr = np.eye(len(names))
            if n0:
                corr[:n0, :n0] = result["calib_corr"]
            result["calib_params"] = names
            result["calib_corr"] = corr
            result["calib_jac"] = jac
            # total fractional error incl. extras (extras mutually uncorrelated)
            if with_calib_error:
                for s in SPECIES:
                    q = result["flux_relerr"][s] ** 2
                    for _, r in extras:
                        q = q + r[s] ** 2
                    result["flux_relerr"][s] = np.sqrt(q)
        if with_calib_error:
            result["flux_err"] = {
                s: flux[s] * result["flux_relerr"][s][:, None, :] for s in SPECIES
            }
        return result


def _interp_rc(rc, rc_grid, gmat):
    """Linear interp of G at scalar ``rc`` along axis 0 of ``gmat`` (n_rc, n_e).

    Vectorised over energy (replaces a per-energy ``np.interp`` loop); clamps to the
    grid endpoints exactly like ``np.interp``.
    """
    rc = min(max(float(rc), rc_grid[0]), rc_grid[-1])
    j = int(np.searchsorted(rc_grid, rc))
    j = min(max(j, 1), len(rc_grid) - 1)
    w = (rc - rc_grid[j - 1]) / (rc_grid[j] - rc_grid[j - 1])
    return (1.0 - w) * gmat[j - 1] + w * gmat[j]


def hybrid_weight(e, e0=1.7):
    """Daemonflux weight w(E) of the ``base_model="hybrid"`` blend.

    Smooth one-octave log-transition centred at ``e0`` [GeV]: w->0 below (the
    GSF-anchored MCEq base), w->1 above (the muon-calibrated daemonflux base);
    w(e0)=0.5. Default e0=1.7 GeV is the daemonflux muon-calibration floor
    (E_mu >= 5 GeV per its README) mapped to neutrinos via E_nu ~ E_mu/3 --
    a physics-derived choice, not a fit to any reference.
    """
    return 0.5 * (1.0 + np.tanh(np.log(np.asarray(e, float) / e0) / np.log(2.0)))


def horizon_grid(n_horizon=9, n_bulk=6, cz_max=0.95, horizon_half_width=0.2):
    """Full-sky cosθ grid **refined near the horizon** (cosθ→0).

    The flux (and the sec θ enhancement) vary fastest near cosθ=0, so a uniform
    grid interpolates poorly there. This packs ``n_horizon`` points (per
    hemisphere) into |cosθ| < ``horizon_half_width`` and ``n_bulk`` into the rest,
    symmetric in up/down. Pass the result as ``cos_zeniths`` to :meth:`solve`.
    """
    bulk = np.linspace(horizon_half_width, cz_max, n_bulk)
    horiz = np.linspace(0.02, horizon_half_width, n_horizon, endpoint=False)
    down = np.unique(np.concatenate([horiz, bulk]))
    return np.unique(np.concatenate([-down[::-1], down]))


def interp_flux(result, energy_gev, cos_zenith, azimuth_deg, species="total_numu"):
    """Evaluate a solved grid at (E, cosZ, azimuth) [/(m^2 s sr GeV)].

    Trilinear in (log E, azimuth, cosZ) on the grid from :meth:`MCEq3DFlux.solve`;
    azimuth wraps modulo 360.
    """
    e, cz, az = result["e"], result["cos_zeniths"], result["azimuths"]
    f = result["flux"][species]  # (cosZ, az, E)
    le = np.log(np.maximum(energy_gev, e[0]))
    # interpolate in log E for every (cosZ, az)
    fz = np.array(
        [
            [np.interp(le, np.log(e), f[i, j]) for j in range(len(az))]
            for i in range(len(cz))
        ]
    )  # (cosZ, az)
    az_ext = np.r_[az, az[0] + 360.0]
    row = np.array(
        [
            np.interp(azimuth_deg % 360.0, az_ext, np.r_[fz[i], fz[i, 0]])
            for i in range(len(cz))
        ]
    )  # (cosZ,) at this E, azimuth
    order = np.argsort(cz)
    return float(np.interp(cos_zenith, cz[order], row[order]))


def calib_covariance(result, species="total_numu", iz=0, ia=0):
    """Full (n_E, n_E) muon-calibration covariance of the directional flux for one
    direction, from ``solve(with_calib_jacobian=True)``.

    ``Cov = A^T . corr . A`` with the absolute per-parameter gradient
    ``A = Phi * R`` (``R`` = fractional response per +1-sigma pull). ``sqrt(diag(Cov))``
    is the absolute 1-sigma; dividing by the flux recovers the fractional ``error()``.
    An oscillation fit uses the daemonflux pulls as nuisance parameters with unit
    priors and correlation ``result["calib_corr"]``, and modifies the flux by
    ``Phi * (1 + sum_i eta_i R_i)``.
    """
    R = result["calib_jac"][species][:, iz, :]  # (n_par, n_E) fractional response
    phi = result["flux"][species][iz, ia]  # (n_E,)
    A = phi[None, :] * R  # absolute gradient (n_par, n_E)
    return A.T @ result["calib_corr"] @ A


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lat", type=float, default=36.43)
    p.add_argument("--lon", type=float, default=137.31)
    p.add_argument("--validate", action="store_true", help="compare vs Honda")
    p.add_argument("--plot", action="store_true")
    p.add_argument(
        "--base",
        choices=["mceq", "daemonflux", "hybrid"],
        default="mceq",
        help="1D base the geomag factor multiplies (daemonflux = data-anchored)",
    )
    p.add_argument(
        "--offaxis", action="store_true",
        help="fold in the first-principles off-axis 3D-production factor E_off "
        "(complete genuine-3D zenith shape; recommended)",
    )
    p.add_argument(
        "--full3d", action="store_true",
        help="legacy: flux-conserving production-angle redistribution only "
        "(superseded by --offaxis; cannot be combined)",
    )
    args = p.parse_args(argv)

    df_loc = "kamioka" if abs(args.lat - 36.43) < 1 else "generic"
    eng = MCEq3DFlux(base_model=args.base, daemonflux_location=df_loc)
    cz = np.array([-0.95, -0.55, -0.15, 0.15, 0.55, 0.95])  # full sky
    az = np.array([0, 45, 90, 135, 180, 225, 270, 315], float)
    r = eng.solve(
        args.lat, args.lon, cz, az, full_3d=args.full3d, offaxis=args.offaxis
    )
    e = r["e"]
    ie = int(np.argmin(np.abs(e - 1.0)))
    print(
        f"Absolute numu flux at 1 GeV (vertical, az-avg): "
        f"{r['flux']['total_numu'][0].mean(0)[ie]:.3g} /(m^2 s sr GeV)"
    )

    if args.validate or args.plot:
        _validate(r, args)


def _validate(r, args):
    import os

    if not os.path.exists("honda_kam.npz"):
        print("honda_kam.npz not present; run validate_honda.py first.")
        return
    h = dict(np.load("honda_kam.npz"))
    He, Hcz, nm = h["E"], h["czlo"], h["numu"]  # Honda numu[cosZ,az,E]
    e = r["e"]

    def _at(yvals, xgrid, E):
        """Log-log interpolate a falling flux to the exact energy E.

        Essential: the MCEq grid point nearest 1 GeV is 0.89 GeV, so comparing by
        nearest-index would pit our 0.89 GeV flux against Honda's 1.0 GeV bin (a
        ~1.5x energy mismatch on a steeply falling spectrum).
        """
        y = np.maximum(np.asarray(yvals), 1e-300)
        return float(np.exp(np.interp(np.log(E), np.log(xgrid), np.log(y))))

    # Dense low-E grid: 0.1-100 GeV is the region that matters (sub-GeV oscillation
    # physics), with extra points below 1 GeV.
    EGRID = (0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0)
    # az-averaged numu vs Honda: vertical (down 0.95) and up-going (-0.95)
    print("\nABSOLUTE numu (az-avg) vs Honda HKKM2014 [/(m^2 s sr GeV)], 0.1-100 GeV:")
    print("  E[GeV]   vert(this)   vert(Honda)  ratio | up(this) up(Honda) ratio")
    mine = {s: r["flux"][s] for s in SPECIES}
    izd = int(np.argmin(np.abs(r["cos_zeniths"] - 0.95)))
    izu = int(np.argmin(np.abs(r["cos_zeniths"] + 0.95)))
    ihzd = int(np.argmin(np.abs(Hcz - 0.9)))  # Honda down bin lo edge
    ihzu = int(np.argmin(np.abs(Hcz - (-1.0))))  # Honda up bin lo edge
    for E in EGRID:
        md = _at(mine["total_numu"][izd].mean(0), e, E)
        hd = _at(nm[ihzd].mean(0), He, E)
        mu = _at(mine["total_numu"][izu].mean(0), e, E)
        hu = _at(nm[ihzu].mean(0), He, E)
        print(
            f"  {E:6.2f}  {md:10.3g}  {hd:10.3g}  {md/hd:5.2f} |"
            f" {mu:8.3g} {hu:8.3g} {mu/hu:5.2f}"
        )

    # flavour ratio (nue+nuebar)/(numu+numubar) -- robust across models
    if "nue" in h:
        nue_h = h["nue"] + h["nuebar"]
        numu_h = h["numu"] + h["numubar"]
        print("\nFLAVOUR RATIO (nue+nuebar)/(numu+numubar) vs Honda (vertical):")
        print("  E[GeV]   this work   Honda")
        iz = int(np.argmin(np.abs(r["cos_zeniths"] - 0.95)))
        ihz = int(np.argmin(np.abs(Hcz - 0.9)))
        for E in (0.1, 0.2, 0.3, 0.5, 1.0, 3.0, 10.0):
            rm = (
                _at(mine["total_nue"][iz].mean(0), e, E)
                + _at(mine["total_antinue"][iz].mean(0), e, E)
            ) / (
                _at(mine["total_numu"][iz].mean(0), e, E)
                + _at(mine["total_antinumu"][iz].mean(0), e, E)
            )
            rh = _at(nue_h[ihz].mean(0), He, E) / _at(numu_h[ihz].mean(0), He, E)
            print(f"  {E:5.1f}     {rm:6.3f}     {rh:6.3f}")
    if args.plot:
        _plot(r, h)


def _plot(r, h):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e = r["e"]
    He, Hcz, nm = h["E"], h["czlo"], h["numu"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.4))
    # spectrum, down-going vertical: this work vs Honda, focused on 0.1-100 GeV
    iz = int(np.argmin(np.abs(r["cos_zeniths"] - 0.95)))
    mv = r["flux"]["total_numu"][iz].mean(0)
    ihz = int(np.argmin(np.abs(Hcz - 0.9)))
    s = (e >= 0.1) & (e <= 100)
    axL.loglog(e[s], (mv * e**3)[s], "C3o-", ms=3, label="this work (3D engine)")
    sH = (He >= 0.1) & (He <= 100)
    axL.loglog(He[sH], (nm[ihz].mean(0) * He**3)[sH], "k--", label="Honda HKKM2014")
    axL.set_xlabel("E [GeV]")
    axL.set_ylabel(r"$E^3\,\Phi_{\nu_\mu}$  [GeV$^2$/(m$^2$ s sr)]")
    axL.set_title("Absolute numu spectrum, vertical (Kamioka)")
    axL.legend()
    # zenith dependence at 1 GeV
    ie = int(np.argmin(np.abs(e - 1.0)))
    ih = int(np.argmin(np.abs(He - 1.0)))
    axR.plot(
        r["cos_zeniths"],
        r["flux"]["total_numu"][:, :, ie].mean(1),
        "C3o-",
        label="this work",
    )
    czc = Hcz + 0.05
    axR.plot(czc[czc > 0], nm[:, :, ih].mean(1)[czc > 0], "k--", label="Honda")
    axR.set_xlabel(r"$\cos\theta_z$")
    axR.set_ylabel(r"$\Phi_{\nu_\mu}$ at 1 GeV  [/(m$^2$ s sr GeV)]")
    axR.set_title("Zenith dependence (azimuth-averaged)")
    axR.legend()
    fig.tight_layout()
    fig.savefig("mceq3d_flux.png", dpi=110)
    print("\nsaved plot -> mceq3d_flux.png")


if __name__ == "__main__":
    main()
