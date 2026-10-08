"""Batched event pools for the ``chromo`` interaction backend.

Why a pool at all
-----------------
``chromo`` generators pay a large *per-call* setup cost.  Measured here on one
core, SIBYLL-2.3d p--air at 100 GeV:

===========================  ===============
``model(1)`` in a loop        73 events/s
``model(3000)`` in one call   1845 events/s
===========================  ===============

a factor 25.  A cascade needs one interaction at a time, at an arbitrary
energy, so the only way to keep the generator batched is to **pre-generate a
pool of events per (projectile, energy bin) and draw from it**.

The bin representative and the rescaling
----------------------------------------
Events in a pool are generated at the geometric centre ``E0`` of a log energy
bin (20 bins/decade for SIBYLL, 10/decade for the much slower DPMJET, see
``BINS_PER_DECADE``).  When the cascade asks for an interaction at ``E != E0``
the drawn event is rescaled by keeping ``p_T`` and ``x_L = p_z / p_beam``
fixed and scaling ``p_z`` by ``f = p_beam(E) / p_beam(E0)``.  That is Feynman
scaling, exact in the limit the generator itself scales, and the residual is
bounded by the half-bin width (5.9% for 20/decade, 12% for 10/decade).  ``p_T``
is deliberately *not* scaled: its distribution is energy-independent to a good
approximation, and it is the whole source of the 3D opening angle.

Reuse
-----
A pool holds ``n_pool`` events and is drawn from with replacement.  Re-using an
event does not bias any inclusive (linear) tally -- and the scorer tallies
nothing else -- it only caps the effective statistics of that one
(projectile, energy-bin) channel at ``n_pool`` events.  Pools are cached on
disk (``cache_dir``) so that all workers of all runs share them and the
generation cost is paid once.

Air target
----------
Nitrogen and oxygen with **atomic** number fractions.  Dry air is 78.09% N2,
20.95% O2, 0.93% Ar by molecule, i.e. 0.7848 / 0.2105 / 0.0047 by atom.
SIBYLL-2.3d does not accept an Ar target (``Sibyll23d.targets`` stops below
A=40), so argon's 0.47% is folded into oxygen: ``f_N = 0.7848``,
``f_O = 0.2152``, ``<A> = 14.430`` against real air's 14.542 (a 0.8%
difference, which enters only through the interaction length).  The same
composition is used for DPMJET so the two sides of the 80 GeV splice see the
same target.

Cross sections
--------------
``air_lambda`` builds ``lambda = <A> m_u / sum_i f_i sigma_prod,i`` from
**chromo's own** per-component production cross sections
(``model.cross_section(kin).prod``; ``.prod`` and not ``.inelastic`` is the
right quantity, since quasi-elastic events make no new secondaries).
``chromo``'s own ``CompositeTarget`` averaging is *not* used: it draws the
target from the number fractions alone rather than from ``f_i sigma_i``, and
its composite ``.prod`` accumulator returns a wrong value for DPMJET.

Two defects of the chromo accessors were measured here and are worked around:

* **DPMJET-III-19.3 returns a constant.**  ``cross_section().prod`` is
  293.158 mb for *every* projectile (p, pi+, K+, K_L) at *every* energy
  (20 and 79 GeV) -- it is the value frozen at initialisation.  It is therefore
  unusable, and below the 80 GeV splice the interaction length is taken from
  MCEq's tabulated SIBYLL-2.3d/DPMJET-III-19.3 cross sections instead (the
  same database, correctly differentiated by projectile and energy).
* **SIBYLL-2.3d returns NaN for a Lambda projectile**, handled by falling back
  to the number fractions for the target draw and to the MCEq table for
  ``lambda``.  Lambda's decay length in air is ~10^-3 of its interaction
  length, so the choice is immaterial.

SIBYLL's own cross sections *are* correct and are tabulated above the splice;
``ChromoBackend(xs=...)`` selects ``"mceq"`` (default -- the only source valid
over the whole range, and the one that makes rung B a pure *yield*
comparison), ``"hybrid"`` (chromo above the splice, MCEq below) or
``"chromo"``.  Measured difference at 100 GeV p-air: chromo/SIBYLL
lambda = 87.76 g/cm2 against MCEq's 89.44, i.e. 1.9%.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from constants import MASS

M_U_G = 1.66053907e-24          # atomic mass unit [g]

# atomic number fractions; argon folded into oxygen (see the module docstring)
AIR_COMPONENTS = (("N", 14, 7, 0.7848), ("O", 16, 8, 0.2152))
A_AIR_MEAN = sum(f * a for _n, a, _z, f in AIR_COMPONENTS)     # 14.430

# projectiles that are re-interacted (the rest decay only)
POOL_PROJECTILES = (2212, -2212, 2112, -2112, 211, -211, 321, -321,
                    130, 310, 3122, -3122)

# species kept from a generated final state (everything else -- pi0, gammas,
# electrons, Sigma, charm -- is dropped exactly as in milestone 1)
KEEP = frozenset((2212, -2212, 2112, -2112, 211, -211, 321, -321,
                  130, 310, 3122, -3122, 13, -13))

E_SWITCH = 80.0                 # GeV: DPMJET below, SIBYLL above
E_MIN_POOL = 2.0                # GeV total energy of the projectile
E_MAX_POOL = 1.0e6
BINS_PER_DECADE = {"lo": 10, "hi": 20}


def _p_beam(pdg, e_tot):
    m = MASS.get(int(pdg), 0.938)
    return float(np.sqrt(max(e_tot * e_tot - m * m, 1e-12)))


class EnergyGrid:
    """Log energy bins, coarser below ``E_SWITCH`` where DPMJET is 16x slower."""

    def __init__(self, e_min=E_MIN_POOL, e_max=E_MAX_POOL, e_switch=E_SWITCH):
        self.e_switch = float(e_switch)
        n_lo = int(round(np.log10(e_switch / e_min) * BINS_PER_DECADE["lo"]))
        n_hi = int(round(np.log10(e_max / e_switch) * BINS_PER_DECADE["hi"]))
        lo = np.logspace(np.log10(e_min), np.log10(e_switch), n_lo + 1)
        hi = np.logspace(np.log10(e_switch), np.log10(e_max), n_hi + 1)
        self.edges = np.concatenate([lo, hi[1:]])
        self.centres = np.sqrt(self.edges[1:] * self.edges[:-1])
        self.n = len(self.centres)

    def index(self, e_tot):
        i = int(np.searchsorted(self.edges, e_tot) - 1)
        return int(np.clip(i, 0, self.n - 1))

    def centre(self, i):
        return float(self.centres[i])


GRID = EnergyGrid()


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
def _model_name(e_tot, model_hi="Sibyll23d", model_lo="DpmjetIII193"):
    return model_hi if e_tot >= E_SWITCH else model_lo


class _Generators:
    """Process-wide ``chromo`` model instances, one per model name.

    Two hard constraints of the wrapped fortran libraries force this to be a
    **process-level singleton**, both of which cost a full debugging round:

    1. ``chromo`` asserts on a second construction of the same generator in one
       process ("Don't run initialization multiple times for the same
       generator ... all symbols are by default in global scope").  So a new
       ``EventPool`` in a worker must find the already-built model, not build
       its own.
    2. DPMJET-III sizes its internal arrays at *initialisation* from the
       projectile and target mass, then refuses anything heavier
       ("Maximal initialization mass exceeded 1/1, 16/14").  It must therefore
       be initialised with the **heaviest** target it will ever be handed --
       oxygen, not nitrogen -- and at the top of the energy range.
    """

    _shared = {}

    def __init__(self, seed=1):
        self.seed = int(seed)
        self._m = _Generators._shared

    def get(self, name, pdg, e_tot, target):
        import chromo
        from chromo.kinematics import FixedTarget, GeV
        if name not in self._m:
            heaviest = max(AIR_COMPONENTS, key=lambda c: c[1])
            init = FixedTarget(1.0e5 * GeV, 2212, (heaviest[1], heaviest[2]))
            self._m[name] = getattr(chromo.models, name)(init, seed=self.seed)
        m = self._m[name]
        kin = FixedTarget(float(e_tot) * GeV, int(pdg), tuple(target))
        if m.kinematics != kin:
            m.kinematics = kin
        return m


def air_lambda(gens, pdg, e_tot, model=None):
    """Interaction length [g/cm2] from chromo's own production cross sections.

    ``lambda = <A> m_u / sum_i f_i sigma_prod,i`` with the atomic fractions of
    ``AIR_COMPONENTS``.  Combining the components explicitly (rather than
    through ``chromo.util.CompositeTarget``) also sidesteps DPMJET's broken
    composite ``.prod`` accumulator.
    """
    name = model or _model_name(e_tot)
    sig = 0.0
    for _nm, a, z, f in AIR_COMPONENTS:
        m = gens.get(name, pdg, e_tot, (a, z))
        cs = m.cross_section()
        s = cs.prod
        if not np.isfinite(s) or s <= 0:
            s = cs.inelastic
        sig += f * float(s)            # mb
    if not np.isfinite(sig) or sig <= 0:
        return None            # caller falls back to the MCEq table
    return A_AIR_MEAN * M_U_G / (sig * 1e-27)


def _target_fractions(gens, pdg, e_tot, model):
    """Per-component target probabilities, ``p_i ∝ f_i sigma_prod,i``.

    ``chromo``'s ``CompositeTarget`` draws the target from the *number*
    fractions alone, which over-weights nitrogen by ~1%: the struck nucleus
    must be drawn from ``n_i sigma_i``.
    """
    w = []
    for _nm, a, z, f in AIR_COMPONENTS:
        cs = gens.get(model, pdg, e_tot, (a, z)).cross_section()
        s = cs.prod
        if not np.isfinite(s) or s <= 0:
            s = cs.inelastic
        w.append(f * float(s))
    w = np.asarray(w, float)
    if not np.all(np.isfinite(w)) or w.sum() <= 0:
        # SIBYLL-2.3d returns NaN for the Lambda-air cross section; fall back
        # to the plain number fractions.  Lambda decays ~10^3 times more often
        # than it interacts at these energies, so the choice is immaterial.
        w = np.array([f for _n, _a, _z, f in AIR_COMPONENTS], float)
    return w / w.sum()


def generate_pool(pdg, e0, n_pool, seed=1, gens=None, model=None):
    """Generate ``n_pool`` final states of ``pdg`` + air at total energy ``e0``.

    Returns the flat-array pool dict.  The targets are drawn from
    ``f_i sigma_i`` and each sub-batch is produced with a single ``model(n)``
    call, which is where the 25x throughput comes from.
    """
    gens = gens or _Generators(seed)
    name = model or _model_name(e0)
    rng = np.random.default_rng(seed)
    p = _target_fractions(gens, pdg, e0, name)
    counts = rng.multinomial(n_pool, p)
    pid_l, px_l, py_l, pz_l, nper = [], [], [], [], []
    for (_nm, a, z, _f), k in zip(AIR_COMPONENTS, counts):
        if k == 0:
            continue
        m = gens.get(name, pdg, e0, (a, z))
        for ev in m(int(k)):
            fs = ev.final_state()
            pid = np.asarray(fs.pid, np.int32)
            keep = np.array([int(q) in KEEP for q in pid], bool)
            pid_l.append(pid[keep])
            px_l.append(np.asarray(fs.px, np.float32)[keep])
            py_l.append(np.asarray(fs.py, np.float32)[keep])
            pz_l.append(np.asarray(fs.pz, np.float32)[keep])
            nper.append(int(keep.sum()))
    if not nper:
        raise RuntimeError(f"no events generated for {pdg} at {e0} GeV")
    off = np.zeros(len(nper) + 1, np.int64)
    off[1:] = np.cumsum(nper)
    return {"pid": np.concatenate(pid_l) if pid_l else np.zeros(0, np.int32),
            "px": np.concatenate(px_l), "py": np.concatenate(py_l),
            "pz": np.concatenate(pz_l), "off": off,
            "e0": np.array([e0]), "pdg": np.array([pdg]),
            "n": np.array([len(nper)]),
            "model": np.array([name])}


# ---------------------------------------------------------------------------
# the pool object used by the backend
# ---------------------------------------------------------------------------
class EventPool:
    """On-disk-cached pools of pre-generated events, keyed by (pdg, E-bin)."""

    def __init__(self, cache_dir, n_pool=2000, seed=1, grid=None,
                 model_hi="Sibyll23d", model_lo="DpmjetIII193", readonly=False):
        self.cache_dir = cache_dir
        self.n_pool = int(n_pool)
        self.seed = int(seed)
        self.grid = grid or GRID
        self.model_hi, self.model_lo = model_hi, model_lo
        self.readonly = bool(readonly)
        self._pools = {}
        self._lam = {}
        self._gens = None
        self.n_draw = 0
        self.n_gen = 0
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    # -- housekeeping ------------------------------------------------------
    @property
    def gens(self):
        if self._gens is None:
            self._gens = _Generators(self.seed)
        return self._gens

    def model_for(self, e_tot):
        return self.model_hi if e_tot >= self.grid.e_switch else self.model_lo

    def path(self, pdg, i):
        return os.path.join(self.cache_dir,
                            f"pool_{int(pdg)}_{i:03d}_{self.n_pool}.npz")

    @property
    def lam_path(self):
        return os.path.join(self.cache_dir, "lambda_table.npz")

    def _lam_table(self):
        """Pre-tabulated interaction lengths, so a ``readonly`` worker never
        constructs a chromo generator at all (each construction costs ~1 s and
        a few hundred MB of fortran tables per process)."""
        if getattr(self, "_lamtab", None) is None:
            self._lamtab = {}
            if self.cache_dir and os.path.exists(self.lam_path):
                d = np.load(self.lam_path)
                for k in d.files:
                    if k.startswith("lam_"):
                        self._lamtab[int(k[4:])] = d[k]
        return self._lamtab

    def build_lambda_table(self, species=POOL_PROJECTILES, emax=np.inf):
        out = {}
        for s in species:
            v = np.full(self.grid.n, np.nan)
            for i in range(self.grid.n):
                e0 = self.grid.centre(i)
                if e0 > emax:
                    break
                if e0 < self.grid.e_switch:
                    continue          # chromo's DPMJET sigma is unusable
                lam = air_lambda(self.gens, s, e0, model=self.model_for(e0))
                v[i] = np.inf if lam is None else lam
            out[f"lam_{int(s)}"] = v
        _atomic_savez(self.lam_path, out)
        return out

    # -- pool access -------------------------------------------------------
    def _load_or_make(self, pdg, i):
        key = (int(pdg), int(i))
        if key in self._pools:
            return self._pools[key]
        e0 = self.grid.centre(i)
        fp = self.path(pdg, i) if self.cache_dir else None
        if fp and os.path.exists(fp):
            try:
                d = dict(np.load(fp, allow_pickle=False))
                self._pools[key] = d
                return d
            except Exception:
                pass
        if self.readonly:
            raise KeyError(f"pool {key} missing from {self.cache_dir}")
        d = generate_pool(pdg, e0, self.n_pool,
                          seed=self.seed + 7919 * i + 104729 * abs(int(pdg)),
                          gens=self.gens, model=self.model_for(e0))
        self.n_gen += self.n_pool
        if fp:
            _atomic_savez(fp, d)
        self._pools[key] = d
        return d

    def lambda_int(self, pdg, e_tot):
        """Air interaction length [g/cm2], cached on the pool energy grid.

        Below ``grid.edges[0] = 2 GeV`` the particle is declared
        non-interacting (``inf``).  That is safe rather than lazy: a nucleon
        below 2 GeV total energy has 0.23 GeV of kinetic energy at most and
        cannot make a pion at all, while a pion or kaon that low decays long
        before it interacts (a 1 GeV pion's decay length is 0.1 g/cm2 at 30 km
        and 7 g/cm2 at sea level, against an interaction length of 110).
        """
        if e_tot < self.grid.edges[0]:
            return np.inf
        i = self.grid.index(e_tot)
        if e_tot < self.grid.e_switch:
            return None      # chromo's DPMJET cross section is a constant
        tab = self._lam_table().get(int(pdg))
        if tab is not None and np.isfinite(tab[i]):
            return float(tab[i]) if np.isfinite(tab[i]) else None
        key = (int(pdg), i)
        if key not in self._lam:
            e0 = self.grid.centre(i)
            self._lam[key] = air_lambda(self.gens, pdg, e0,
                                        model=self.model_for(e0))
        return self._lam[key]

    def lambda_available(self, pdg, e_tot):
        """``lambda_int`` or ``None`` when chromo has no cross section."""
        if e_tot < self.grid.edges[0]:
            return np.inf
        return self.lambda_int(pdg, e_tot)

    def bin_choice(self, rng, e_tot):
        """Pick the pool bin, interpolating stochastically between neighbours.

        Drawing always from the bin that *contains* ``e_tot`` freezes the
        multiplicity at the bin centre and leaves a first-order bias of order
        the bin width (~1% per 20-bins-per-decade bin, since
        ``d<n>/dln E ~ 1.5`` against ``<n> ~ 18``).  Choosing between the two
        bins whose centres bracket ``e_tot``, with probability linear in
        ``ln E``, makes the mean multiplicity linear in ``ln E`` and leaves
        only the second-order term.
        """
        c = self.grid.centres
        j = int(np.searchsorted(c, e_tot))
        if j <= 0:
            return 0
        if j >= len(c):
            return len(c) - 1
        t = np.log(e_tot / c[j - 1]) / np.log(c[j] / c[j - 1])
        return j if rng.random() < t else j - 1

    def draw(self, rng, pdg, e_tot):
        """One final state at total energy ``e_tot``.

        Returns ``[(pdg, E, p_vec), ...]`` with ``p_vec`` in the projectile
        frame (z along the projectile).  ``p_z`` is rescaled by
        ``p_beam(E)/p_beam(E0)``, ``p_T`` is kept.
        """
        if e_tot < self.grid.edges[0]:
            return []
        i = self.bin_choice(rng, e_tot)
        d = self._load_or_make(pdg, i)
        n = int(d["n"][0])
        k = int(rng.integers(n))
        a, b = int(d["off"][k]), int(d["off"][k + 1])
        self.n_draw += 1
        if b <= a:
            return []
        f = _p_beam(pdg, e_tot) / _p_beam(pdg, float(d["e0"][0]))
        pid = d["pid"][a:b]
        px = d["px"][a:b].astype(np.float64)
        py = d["py"][a:b].astype(np.float64)
        pz = d["pz"][a:b].astype(np.float64) * f
        out = []
        for q, x, y, z in zip(pid, px, py, pz):
            m = MASS.get(int(q), 0.0)
            e = float(np.sqrt(m * m + x * x + y * y + z * z))
            out.append((int(q), e, np.array([x, y, z])))
        # energy-conservation guard: with 20 bins/decade this never fires,
        # but a pathological rescale must not create energy.
        tot = sum(o[1] for o in out)
        avail = e_tot + 0.94
        if tot > avail:
            s = avail / tot
            out = [(q, e * s, p * s) for q, e, p in out]
        return out


def _atomic_savez(path, d):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".npz")
    os.close(fd)
    try:
        np.savez(tmp, **d)
        os.replace(tmp + ".npz" if not tmp.endswith(".npz") else tmp, path)
    except Exception:
        for p in (tmp, tmp + ".npz"):
            if os.path.exists(p):
                os.remove(p)
        raise
