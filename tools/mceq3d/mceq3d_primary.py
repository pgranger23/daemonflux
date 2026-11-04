"""Primary-flux classes for the engine: which energy variable each crflux model wants.

MCEq evaluates every primary model at the *total* energy per nucleon
``E_N = e_grid + m_N`` (``MCEqRun.set_primary_model``; ``pmodel.p_and_n_flux(E_N)``,
nucleus energy ``A * E_N``).  That is the native variable of ``GlobalSplineFitBeta``
and ``HillasGaisser2012``.  crflux's ``GaisserHonda`` formula, however,
``K/A (E/A + b exp(-c sqrt(E/A)))^-alpha``, takes ``E/A`` as the **kinetic**
energy per nucleon (Gaisser & Honda 2002), so MCEq's call mis-evaluates it by a shift of
one nucleon mass (all-nucleon flux / GSF at 3 GeV/n: 0.64 instead of 1.05).

``primary_class(name)`` is the engine's single resolver of ``primary[0]``:

``GaisserHondaNative``    ``GaisserHonda`` with its native variable (kinetic energy
                          per nucleon): the nucleus energy is shifted by ``A m_N``
                          before the crflux formula.  Use this for physics.
``GaisserHondaAtTotal``   crflux ``GaisserHonda`` unchanged, i.e. evaluated at the
                          total energy as MCEq does; kept only as the bridge to the
                          Phase-1 primary scan (``PHASE1_RESULTS.md`` 8.3).
``GaisserHonda``          refused: ambiguous (the name used to mean ``AtTotal``).

Every other name is looked up in ``crflux.models`` as before.  The class name is
part of the G_s cache key, so the two GH variants never share a cache entry.
"""

M_NUCLEON = 0.938272  # GeV, proton mass (momentum of a nucleon from its kinetic energy)


def _gh_native():
    import crflux.models as crf

    class GaisserHondaNative(crf.GaisserHonda):
        """GaisserHonda evaluated at kinetic energy per nucleon (its native variable)."""

        def _nucleus_flux(self, corsika_id, E):
            import numpy as np

            A = self.Z_A(self._find_nearby_id(corsika_id))[1]
            return super()._nucleus_flux(corsika_id, np.asarray(E, float) - A * M_NUCLEON)

    GaisserHondaNative.__name__ = "GaisserHondaNative"
    return GaisserHondaNative


_CACHE = {}


def primary_class(name):
    """crflux class (or engine wrapper) for ``primary[0]``."""
    import crflux.models as crf

    if name == "GaisserHonda":
        raise ValueError(
            "primary 'GaisserHonda' is ambiguous since 2026-11 (MCEq passes total "
            "energy, the GH formula wants kinetic): use 'GaisserHondaNative' "
            "(physics) or 'GaisserHondaAtTotal' (the old, mis-evaluated behaviour)"
        )
    if name == "GaisserHondaAtTotal":
        return crf.GaisserHonda
    if name == "GaisserHondaNative":
        if name not in _CACHE:
            _CACHE[name] = _gh_native()
        return _CACHE[name]
    return getattr(crf, name)
