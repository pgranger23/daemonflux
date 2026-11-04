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
    native_energy = "total"

    def __init__(self):
        import crflux.models as crf
        self._m = crf.GlobalSplineFitBeta()

    def nucleon_flux(self, e_nucleon, convention=None):
        """``(total nucleon intensity, proton fraction)`` at kinetic energy
        per nucleon ``e_nucleon`` (crflux evaluated at the energy variable of
        :data:`ENERGY_CONVENTION`, see :func:`crflux_energy`)."""
        e = crflux_energy(e_nucleon, 1, convention, model=self)
        frac, pf, nf = self._m.p_and_n_flux(e)
        return (pf + nf) * M2_TO_CM2, frac

    def proton_neutron(self, e_nucleon, convention=None):
        """``(J_p, J_n)`` [cm^-2 s^-1 sr^-1 GeV^-1], per kinetic energy per
        nucleon, protons and neutrons (free and bound)."""
        e = crflux_energy(e_nucleon, 1, convention, model=self)
        _f, pf, nf = self._m.p_and_n_flux(e)
        return pf * M2_TO_CM2, nf * M2_TO_CM2


class GSFGroups:
    """GSF split into the MC's five groups -- an **approximation**.

    ``crflux``'s GSF file holds three splines only (proton fraction, proton
    and neutron nucleon flux vs ``ln E``); the p / He / O / Fe splines of the
    published fit are not in it (and nothing else is installed).  The
    nucleon-level numbers are therefore exact GSF, and the split into groups
    (which only matters through the rigidity ``(A/Z) p`` of the cutoff test
    and through the p/n identity of the cascade's projectile) is taken from a
    reference group model ``ref`` (default H3a at its native energy):

    ``N_{j,x}(T) = J^GSF_x(T) * g_{j|x}(T)``, ``x`` = p or n,

    ``g_{j|x}`` = the share of nucleon type ``x`` that ``ref`` puts in group
    ``j`` at the same kinetic energy per nucleon.  Summed over ``j`` this
    reproduces GSF's proton and neutron fluxes exactly; neutrons come only
    from ``A/Z ~ 2`` groups, protons from the p group and from bound protons.
    ``group_split(gid, T)`` returns ``(J_p, J_n)`` of group ``gid`` in the
    units of :func:`nucleon_intensity`.
    """

    nucleon_level = False
    native_energy = "total"

    def __init__(self, ref="H3A"):
        self.name = "GSF-groups(" + str(ref) + ")"
        self.gsf = GSFNucleons()
        self.ref = primary_model(ref) if isinstance(ref, str) else ref

    def _ref_split(self, gid, t, convention):
        a_mass, z = _AZ[gid]
        j = nucleon_intensity(self.ref, gid, t, a_mass, convention)
        return j * z / a_mass, j * (a_mass - z) / a_mass

    def group_split(self, gid, e_nucleon, convention=None):
        t = np.atleast_1d(np.asarray(e_nucleon, float))
        conv = convention or ENERGY_CONVENTION
        # the reference model is evaluated at ITS native variable unless a
        # fixed convention was asked for
        rc = "native" if conv in ("native", "total") else conv
        mine = self._ref_split(gid, t, rc)
        tot = [0.0, 0.0]
        for _n, _a, _z, g in SPECIES:
            r = self._ref_split(g, t, rc)
            tot[0] = tot[0] + r[0]
            tot[1] = tot[1] + r[1]
        jp, jn = self.gsf.proton_neutron(t, conv)
        sh = [np.where(tot[i] > 0, mine[i] / np.where(tot[i] > 0, tot[i], 1.0),
                       0.0) for i in (0, 1)]
        return jp * sh[0], jn * sh[1]


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
    if name.upper() in ("GST", "GST3"):
        return crf.GaisserStanevTilav("3-gen")
    if name.upper() == "GST4":
        return crf.GaisserStanevTilav("4-gen")
    if name.upper().startswith("GSFG"):          # GSFG, GSFG:GST, GSFG:GH
        ref = name.split(":", 1)[1] if ":" in name else "H3A"
        return GSFGroups(ref)
    raise ValueError(name)


# crflux returns fluxes per (m^2 s sr GeV); everything downstream works in cm^2
M2_TO_CM2 = 1.0e-4

#: The energy variable at which the ``crflux`` spectra are evaluated, for a
#: primary of kinetic energy per nucleon ``T`` (what the drivers sample and
#: the cascade starts from: ``e_tot = T + m``):
#:
#: * ``"total"`` -- crflux's documented argument, "laboratory energy of the
#:   nucleus", i.e. the total energy ``A (T + m_N)``; what MCEq does
#:   (``MCEqRun.set_primary_model`` evaluates ``p_and_n_flux(e_grid + m)``);
#: * ``"kinetic"`` -- ``A T``: milestones 1-3b (for H3a, a bug: sec. 1.1 of
#:   MILESTONE3C_RESULTS.md);
#: * ``"native"`` (default since milestone 3d) -- each model in the variable
#:   its formula is a fit in (:data:`NATIVE_ENERGY`): H3a, GST and GSF fit
#:   the total energy per particle, Gaisser-Honda the *kinetic* energy per
#:   nucleon (``crflux``'s ``GaisserHonda._nucleus_flux`` feeds ``E/A`` to
#:   the formula of Gaisser & Honda 2002, whose argument is the kinetic
#:   energy per nucleon -- so ``A T`` is its right argument).
#:
#: Module-level so that every driver (``sky``, ``sky3d``) and its forked
#: workers see one value; ``sky3d --primary-energy`` sets it (and records it
#: in the run manifest, resolved to ``total`` / ``kinetic`` for the model).
ENERGY_CONVENTION = "native"
ENERGY_CONVENTIONS = ("kinetic", "total", "native")
#: native variable by model name (:func:`native_for_name`)
NATIVE_ENERGY = {"H3A": "total", "H4A": "total", "GST": "total",
                 "GST3": "total", "GST4": "total", "GSF": "total",
                 "GSF_RAW": "total", "GSFG": "total", "GH": "kinetic",
                 "GAISSERHONDA": "kinetic", "GLOBALSPLINEFITBETA": "total"}
#: the nucleon mass MCEq adds to its kinetic grid for the primary flux
M_NUCLEON = 0.5 * (0.938272 + 0.939565)
_AZ = {gid: (a_, z_) for _n, a_, z_, gid in SPECIES}


def native_for_name(name):
    """``"total"`` / ``"kinetic"``: the native energy variable of the model
    called ``name`` (``primary_model`` names; ``GSFG:REF`` is GSF's)."""
    n = name.upper().split(":")[0]
    if n not in NATIVE_ENERGY:
        raise ValueError(f"no native energy variable known for {name!r}; "
                         "pass --primary-energy kinetic|total")
    return NATIVE_ENERGY[n]


def native_convention(model):
    """``"total"`` / ``"kinetic"`` for a model instance (checked against
    ``crflux``'s source, see ``MILESTONE3D_RESULTS.md`` sec. 1)."""
    n = getattr(model, "native_energy", None)
    if n:
        return n
    import crflux.models as crf
    if isinstance(model, crf.GaisserHonda):
        return "kinetic"
    if isinstance(model, (crf.HillasGaisser2012, crf.GaisserStanevTilav,
                          crf.GlobalSplineFitBeta)):
        return "total"
    raise ValueError(f"no native energy variable known for "
                     f"{type(model).__name__}")


def resolve_convention(convention, model=None, name=None):
    """``"kinetic"`` / ``"total"`` for ``convention`` (``None`` = the module
    default); ``"native"`` needs the model (instance or name)."""
    conv = convention or ENERGY_CONVENTION
    if conv == "native":
        if model is not None:
            return native_convention(model)
        if name is not None:
            return native_for_name(name)
        raise ValueError('convention "native" needs the model')
    if conv not in ENERGY_CONVENTIONS:
        raise ValueError(f"energy convention {conv!r} not in "
                         f"{ENERGY_CONVENTIONS}")
    return conv


def crflux_energy(e_kin_nucleon, a_mass, convention=None, model=None):
    """The ``crflux`` argument (energy of the whole nucleus) for kinetic
    energy per nucleon ``e_kin_nucleon`` under ``convention`` (default
    :data:`ENERGY_CONVENTION`; ``"native"`` needs ``model``)."""
    conv = resolve_convention(convention, model)
    t = np.atleast_1d(np.asarray(e_kin_nucleon, float))
    if conv == "total":
        return a_mass * (t + M_NUCLEON)
    return a_mass * t


def _group_ids(model, pdg):
    """The crflux ids that make up MC group ``pdg`` in ``model``: ``[pdg]``
    (``crflux`` maps it to its nearest group) except where that fails: GST
    has C and O but no Mg/Si group (its Fe group carries the heavy nuclei)."""
    import crflux.models as crf
    if isinstance(model, crf.GaisserStanevTilav):
        if pdg == 1407:
            return [1206, 1608]
        if pdg == 2512:
            return []
    return [pdg]


def nucleon_intensity(model, pdg, e_nucleon, a_mass, convention=None):
    """dJ/dE_nucleon [cm^-2 s^-1 sr^-1 GeV^-1] of *nucleons* carried by the
    group ``pdg``, at kinetic energy per nucleon ``e_nucleon`` (crflux
    evaluated at :func:`crflux_energy`).

    ``crflux`` returns ``dJ/dE_total`` per *nucleus* **per m^2**; converting to
    per-nucleon-energy multiplies by ``A`` twice (once for ``dE_tot = A dE_n``,
    the same for the kinetic and the total energy, once because each nucleus
    carries ``A`` nucleons), and the ``1e-4`` puts it on cm^2.  (Check: GSF's
    all-nucleon flux at 10 GeV is 34.4 per m^2 s sr GeV = 3.44e-3 per cm^2,
    against the textbook 1.8 E^-2.7 = 3.6e-3.)
    """
    ids = _group_ids(model, pdg)
    if ids == [pdg]:                     # milestones 1-3c: the MC group's A
        e_tot = crflux_energy(e_nucleon, a_mass, convention, model)
        return model.nucleus_flux(pdg, e_tot) * a_mass * a_mass * M2_TO_CM2
    out = np.zeros(np.shape(np.atleast_1d(e_nucleon)))
    for cid in ids:                      # GST's C + O: each at its own A
        _z, a_c = model.Z_A(cid)
        e_tot = crflux_energy(e_nucleon, a_c, convention, model)
        out = out + model.nucleus_flux(cid, e_tot) * a_c * a_c * M2_TO_CM2
    return out


def group_split(model, pdg, e_nucleon, convention=None):
    """``(J_p, J_n)``: the proton and neutron nucleon intensity of group
    ``pdg`` (same units as :func:`nucleon_intensity`).  Group models:
    ``Z/A`` and ``(A-Z)/A`` of it; :class:`GSFGroups` its own split."""
    if hasattr(model, "group_split"):
        return model.group_split(pdg, e_nucleon, convention)
    a_mass, z = _AZ[pdg]
    j = nucleon_intensity(model, pdg, e_nucleon, a_mass, convention)
    return j * z / a_mass, j * (a_mass - z) / a_mass


def reweight_factor(new, old, pdg, e_nucleon, is_p=None):
    """The exact importance-weight ratio of a primary of group ``pdg`` at
    kinetic energy per nucleon ``e_nucleon`` when the primary model changes
    from ``old`` to ``new``; each is ``(model, convention)``.  Needs the
    nucleon identity ``is_p`` only if one of the models splits p/n by itself
    (:class:`GSFGroups`)."""
    (mn, cn), (mo, co) = new, old
    a_mass, _z = _AZ[pdg]
    if not (hasattr(mn, "group_split") or hasattr(mo, "group_split")):
        return (nucleon_intensity(mn, pdg, e_nucleon, a_mass, cn)
                / nucleon_intensity(mo, pdg, e_nucleon, a_mass, co))
    if is_p is None:
        raise ValueError("this re-weighting needs the nucleon identity")
    pn, nn = group_split(mn, pdg, e_nucleon, cn)
    po, no = group_split(mo, pdg, e_nucleon, co)
    is_p = np.asarray(is_p, bool)
    num = np.where(is_p, pn, nn)
    den = np.where(is_p, po, no)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


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
