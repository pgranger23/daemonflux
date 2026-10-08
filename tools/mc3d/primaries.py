"""Primary cosmic-ray sampling with importance weights (PHASE2_PLAN.md sec. 3).

Five species groups, superposition to nucleons, ``E^-1`` (log-uniform) energy
sampling with the weight that restores the physical spectrum, and a Lambert
inward direction on the injection sphere.

Weight bookkeeping (all per sample, multiply together):

``w_E``      ``J_s(E_nuc) * E_nuc * ln(E2/E1) / n``  -- restores ``dN/dE``
``w_geom``   ``pi * A_inj``                          -- isotropic flux through a
                                                        sphere/patch
``w_cutoff`` geomagnetic admittance in [0, 1]
``w_solar``  ``J_s(E; phi_1) / J_s(E; phi_0)``

so that ``sum(w)`` over a bin is the physical rate [s^-1] into it.
"""

from __future__ import annotations

import numpy as np

# (name, A, Z, crflux pdg id for the group)
SPECIES = (
    ("p", 1, 1, 14),
    ("He", 4, 2, 402),
    ("CNO", 14, 7, 1407),
    ("MgSi", 25, 12, 2512),
    ("Fe", 56, 26, 5626),
)


class GSFNucleons:
    """GSF at the **nucleon** level -- the only thing ``crflux`` exposes for it.

    ``crflux.models.GlobalSplineFitBeta`` is a *spline interface* version: it
    carries the total nucleon flux and the proton/neutron split
    (``p_and_n_flux``) but no per-nucleus splines at all
    (``nucleus_ids == []``, ``_nucleus_flux`` returns zeros).  So the plan's
    "GSF, p and He, superposition" cannot be built from it species by species.

    That costs nothing for milestone 2b.  Superposition means the cascade only
    ever sees free nucleons at ``E/A``, so with **no geomagnetic cutoff** the
    neutrino flux is a functional of the all-nucleon intensity and its p/n
    split alone -- exactly what GSF provides -- and a species decomposition is
    redundant.  It stops being redundant in milestone 3, where the cutoff test
    needs each nucleus's rigidity ``(A/Z) p_nucleon``; there the per-species
    model (``H3A``) or a GSF species table from outside ``crflux`` is required.

    Units are converted to cm^-2 s^-1 sr^-1 GeV^-1 here.
    """

    name = "GSF"
    nucleon_level = True

    def __init__(self):
        import crflux.models as crf
        self._m = crf.GlobalSplineFitBeta()

    def nucleon_flux(self, e_nucleon):
        """``(total nucleon intensity, proton fraction)`` at ``e_nucleon``."""
        frac, pf, nf = self._m.p_and_n_flux(np.atleast_1d(
            np.asarray(e_nucleon, float)))
        return (pf + nf) * M2_TO_CM2, frac


def primary_model(name="GSF"):
    """A ``crflux`` primary model instance (or the GSF nucleon wrapper)."""
    import crflux.models as crf
    if name.upper() in ("GSF", "GLOBALSPLINEFITBETA"):
        return GSFNucleons()
    if name.upper() == "GSF_RAW":
        return crf.GlobalSplineFitBeta()
    if name.upper() == "H3A":
        return crf.HillasGaisser2012("H3a")
    if name.upper() in ("GH", "GAISSERHONDA"):
        return crf.GaisserHonda()
    raise ValueError(name)


# crflux returns fluxes per (m^2 s sr GeV); everything downstream works in cm^2
M2_TO_CM2 = 1.0e-4


def nucleon_intensity(model, pdg, e_nucleon, a_mass):
    """dJ/dE_nucleon [cm^-2 s^-1 sr^-1 GeV^-1] of *nucleons* carried by the
    group ``pdg``, at energy-per-nucleon ``e_nucleon``.

    ``crflux`` returns ``dJ/dE_total`` per *nucleus* **per m^2**; converting to
    per-nucleon-energy multiplies by ``A`` twice (once for ``dE_tot = A dE_n``,
    once because each nucleus carries ``A`` nucleons), and the ``1e-4`` puts it
    on cm^2.  (Check: GSF's all-nucleon flux at 10 GeV is 34.4 per m^2 s sr
    GeV = 3.44e-3 per cm^2, against the textbook 1.8 E^-2.7 = 3.6e-3.)
    """
    e_tot = np.atleast_1d(e_nucleon) * a_mass
    return model.nucleus_flux(pdg, e_tot) * a_mass * a_mass * M2_TO_CM2


def sample_energy(rng, n, e1=1.0, e2=1.0e4):
    """Log-uniform (``E^-1``) energy-per-nucleon sample and its log range."""
    ln1, ln2 = np.log(e1), np.log(e2)
    return np.exp(rng.uniform(ln1, ln2, n)), (ln2 - ln1)


def energy_weights(model, e_nucleon, a_mass, pdg, ln_range, n):
    """``w_E`` for a log-uniform sample (see the module docstring)."""
    return (nucleon_intensity(model, pdg, e_nucleon, a_mass)
            * e_nucleon * ln_range / n)


def sample_species(rng, model, n, e1=1.0, e2=1.0e4):
    """Sample ``n`` nucleon-level primaries across all five groups.

    Returns ``(pdg_nucleon, e_nucleon, rigidity_GV, weight)`` arrays, with the
    weight already including the ``A`` nucleons per nucleus (superposition) and
    the ``A/Z`` rigidity used in the cutoff test -- Honda's "all nucleons ...
    treated as protons with double rigidity".
    """
    per = max(1, n // len(SPECIES))
    pdgs, es, rig, w = [], [], [], []
    for _name, a_mass, z, gid in SPECIES:
        e, lr = sample_energy(rng, per, e1, e2)
        we = energy_weights(model, e, a_mass, gid, lr, per)
        # split the A nucleons into Z protons and A-Z neutrons
        is_p = rng.random(per) < (z / a_mass)
        pdgs.append(np.where(is_p, 2212, 2112))
        es.append(e)
        # rigidity of the PARENT nucleus at this energy per nucleon
        p_tot = np.sqrt((e * a_mass) ** 2 - (a_mass * 0.938) ** 2)
        rig.append(p_tot / max(z, 1))
        w.append(we)
    return (np.concatenate(pdgs), np.concatenate(es),
            np.concatenate(rig), np.concatenate(w))
