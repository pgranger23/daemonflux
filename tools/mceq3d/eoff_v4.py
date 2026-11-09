"""E_off v4: the v3 primary-stratified production cone with the muon-channel and
multi-generation physics that v3 leaves out (opt-in: ``MCEq3DFlux(eoff_model="v4")``).

Why (2026-11-09, ``$W/runs/fix2/REPORT.md``).  v3 (:mod:`eoff_v3`) builds the cone
``W_b(alpha | E_nu)`` from ONE generator interaction of the primary and unpolarised
muon decays at the muon's production energy.  The depth profiles it is combined
with come from MCEq, which carries every hadronic generation, the helicity of the
muons and their energy loss, so the kernel and the profile describe different
populations of neutrinos.  v4 builds the kernel from the same generator pools and
the same per-band GSF weighting as v3, with, each switchable:

* ``multigen``: every nucleon of a final state above the pool floor (2 GeV total)
  interacts again (all generations), drawn from the pool bin bracketing its energy
  (stochastic linear-in-ln-E choice between the two bracketing bins), ``p_z``
  rescaled by ``p_beam(E)/p_beam(E_bin)`` with ``p_T`` kept, and rotated into the
  nucleon's direction.  Pions and kaons always decay (their re-interaction
  probability is <~1% per GeV of meson energy at the production altitudes of
  sub-GeV neutrinos: not carried).
* ``polarisation``: the muon from ``pi/K -> mu nu`` carries the exact longitudinal
  polarisation ``P_L`` (helicity -1 for mu+, +1 for mu- in the parent rest frame,
  spin 4-vector boosted to the lab and projected on the muon momentum); its decay
  uses the polarised Michel distributions ``n(x) [1 + s A(x) P_L cos theta*]``
  with ``A = (1 - 2x)/(3 - 2x)`` for the e-like neutrino (nu_mu from mu-,
  anti-nu_mu from mu+) and ``A = 1`` for the other one, ``s = +1`` for mu-,
  ``-1`` for mu+ (V-A: ``|M|^2 ~ ((p_mu - m s).p_nubar_e)(p_e.p_nu_mu)`` for mu-).
* ``kmu``: muons from ``K+- -> mu nu`` (BR 0.6356) enter the mu channel (MCEq's
  muon category contains them; v3's mu kernel had pion muons only).
* ``eloss_rho``: muon continuous energy loss (PDG dE/dX in air) before the decay
  in a medium of constant density ``eloss_rho`` [g/cm3] (``None`` = off).  The
  decay proper time is exponential; a muon that ranges out decays at rest
  (E_nu < 53 MeV, outside the kernel grid).  The muon direction and ``P_L`` are
  unchanged by the loss.  The density of the decay region depends on the zenith
  and the depth, so a single value is an approximation: off by default.

Everything else (band profiles, Green's functions, the joint integral) is
:mod:`eoff_v3`'s, unchanged; the v3 Green's-function and profile caches are reused.
No MC tally enters the kernel or the table.
"""

from __future__ import annotations

import glob
import hashlib
import os

import numpy as np

import eoff_v3 as v3
import mceq3d_db as _db

SCHEME = "v4a"  # part of every v4 kernel cache key
M_PI, M_MU, M_K, M_N = 0.13957, 0.105658, 0.493677, 0.938272
CTAU_MU_CM = 65865.4
BR_KMU2 = 0.6356
POOL_FLOOR_ETOT = 2.0      # GeV: the pool grid starts here (no interaction below)
GUARD_MASS = 16 * 0.931494  # energy budget of a rescaled final state: E + struck nucleus
NUCLEONS = (2212, 2112)
#: PDG muon stopping power in air [MeV cm2/g] vs muon momentum [GeV/c]
_DEDX_P = np.array([0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0,
                    100.0, 1000.0])
_DEDX_V = np.array([9.20, 6.05, 3.62, 2.35, 2.05, 1.88, 1.84, 1.92, 2.06, 2.19, 2.32,
                    2.51, 2.66, 3.30])
DEFAULTS = dict(multigen=True, polarisation=True, kmu=True, eloss_rho=None)
#: ground zeniths [deg] of the zenith-dependent production profiles (``zdep``)
ZNODES_DEG = (0.0, 45.0, 60.0, 70.0, 77.0, 83.0, 87.0, 89.0)
#: air densities [g/cm3] of the energy-loss kernel family (``eloss="row"``)
RHO_FAMILY = (0.3e-4, 0.6e-4, 1.2e-4, 2.5e-4, 5.0e-4)
#: v4 configuration of the engine path (``MCEq3DFlux(eoff_model="v4")``):
#: kernel physics switches, energy loss per row ("row": the family RHO_FAMILY
#: interpolated in ln rho at the row's muon-decay density, None: off) and the
#: zenith-dependent profiles
V4_CONFIG = dict(multigen=True, polarisation=True, kmu=True, eloss="row", zdep=True)


# ---------------------------------------------------------------------------
# kinematics helpers (vectorised)
# ---------------------------------------------------------------------------
def _basis(u):
    """Orthonormal (x, y) completing unit vectors ``u`` (n, 3)."""
    a = np.where(np.abs(u[:, 2:3]) < 0.9, np.array([[0.0, 0.0, 1.0]]),
                 np.array([[1.0, 0.0, 0.0]]))
    x = np.cross(a, u)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = np.cross(u, x)
    return x, y


def rotate_from_z(u, v):
    """Vectors ``v`` (n, 3) given in a frame whose z is (0,0,1) -> frame whose z is ``u``."""
    x, y = _basis(u)
    return v[:, 0:1] * x + v[:, 1:2] * y + v[:, 2:3] * u


def muon_pl(E_mu, p_mu, beta_parent, n_star, m_parent, charge):
    """Longitudinal polarisation of muons from ``parent -> mu nu`` (vectorised).

    ``n_star`` = muon direction in the parent rest frame, ``beta_parent`` the parent
    lab velocity, ``charge`` +1 (mu+) / -1 (mu-) per row."""
    h = np.where(charge > 0, -1.0, 1.0)
    p_star = (m_parent ** 2 - M_MU ** 2) / (2 * m_parent)
    e_star = np.sqrt(p_star ** 2 + M_MU ** 2)
    s0 = h * p_star / M_MU
    sv = (h * e_star / M_MU)[:, None] * n_star
    _, s_lab = v3._boost(s0, sv, beta_parent)
    pn = np.linalg.norm(p_mu, axis=1)
    pl = (M_MU / E_mu) * np.einsum("ij,ij->i", s_lab, p_mu) / np.maximum(pn, 1e-300)
    return np.clip(pl, -1.0, 1.0)


def michel_polarised(rng, E_mu, p_mu, pl, charge, kind):
    """Lab 4-vectors of one Michel neutrino per muon.  ``kind`` 'numu' = the e-like
    neutrino (nu_mu from mu-, anti-nu_mu from mu+), 'nue' = the other one
    (anti-nu_e from mu-, nu_e from mu+)."""
    n = len(E_mu)
    x = v3._michel(rng, n, kind)
    if kind == "numu":
        a = (1.0 - 2.0 * x) / np.maximum(3.0 - 2.0 * x, 1e-12)
    else:
        a = np.ones(n)
    s = np.where(charge > 0, -1.0, 1.0)
    aa = s * a * pl
    v = rng.random(n)
    small = np.abs(aa) < 1e-9
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(small, 2 * v - 1,
                     (-1.0 + np.sqrt(np.maximum(1.0 + 2.0 * aa * (2 * v - 1) + aa * aa, 0.0)))
                     / np.where(small, 1.0, aa))
    c = np.clip(c, -1.0, 1.0)
    ph = 2 * np.pi * rng.random(n)
    st = np.sqrt(1 - c * c)
    loc = np.stack([st * np.cos(ph), st * np.sin(ph), c], -1)
    pn = np.linalg.norm(p_mu, axis=1, keepdims=True)
    axis = np.where(pn > 1e-12, p_mu / np.maximum(pn, 1e-300), np.array([[0.0, 0.0, 1.0]]))
    nst = rotate_from_z(axis, loc)
    es = 0.5 * M_MU * x
    return v3._boost(es, es[:, None] * nst, p_mu / E_mu[:, None])


class _MuonRange:
    """Q(E) = int_m^E dE'/(p a(E')) for the decay-in-flight-with-loss sampling."""

    def __init__(self):
        e = M_MU + np.geomspace(1e-6, 2e3, 6000)
        p = np.sqrt(e * e - M_MU * M_MU)
        a = 1e-3 * np.exp(np.interp(np.log(p), np.log(_DEDX_P), np.log(_DEDX_V)))
        f = 1.0 / (p * a)
        q = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(e))])
        # the first point: int_m^{m+1e-6} dE/p ~ sqrt(2 dE/m) * m / a -> tiny; keep 0
        self.e, self.q = e, q

    def decay_energy(self, rng, E0, rho):
        """Energy at decay (nan where the muon stops first) for density ``rho``."""
        t = -np.log(rng.random(len(E0)))
        qd = np.interp(E0, self.e, self.q) - t * rho * CTAU_MU_CM / M_MU
        out = np.full(len(E0), np.nan)
        ok = qd > 0
        out[ok] = np.interp(qd[ok], self.q, self.e)
        return out


# ---------------------------------------------------------------------------
# pools
# ---------------------------------------------------------------------------
class _Pools:
    """All nucleon pools in memory (for the re-interactions)."""

    def __init__(self, pools, grid):
        self.grid = grid
        self.c = np.asarray(grid.centres, float)
        self.d = {}
        for pdg in NUCLEONS:
            for f in sorted(glob.glob(os.path.join(pools, f"pool_{pdg}_*_2000.npz"))):
                z = np.load(f)
                i = grid.index(float(z["e0"][0]))
                self.d[(pdg, i)] = dict(pid=z["pid"].astype(np.int64),
                                        p=np.stack([z[k].astype(float) for k in ("px", "py", "pz")], -1),
                                        off=z["off"].astype(np.int64), e0=float(z["e0"][0]),
                                        n=int(z["n"][0]))
        self.avail = {pdg: np.array(sorted(i for (q, i) in self.d if q == pdg)) for pdg in NUCLEONS}

    def bins(self, rng, e):
        j = np.searchsorted(self.c, e)
        jlo = np.clip(j - 1, 0, len(self.c) - 1)
        jhi = np.clip(j, 0, len(self.c) - 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.where(jhi > jlo, np.log(e / self.c[jlo]) / np.log(self.c[jhi] / self.c[jlo]), 0.0)
        return np.where(rng.random(len(e)) < t, jhi, jlo)

    def interact(self, rng, pdg, E, u, w, band):
        """Final states of nucleons (pdg, E, direction u, weight w, band) -> flat
        product arrays (pid, E, p_lab, w, band)."""
        outs = []
        ib = self.bins(rng, E)
        for q in NUCLEONS:
            for i in np.unique(ib[pdg == q]):
                if (q, int(i)) not in self.d:
                    # above the highest pool: nearest available bin
                    av = self.avail[q]
                    ii = int(av[np.argmin(np.abs(av - i))])
                else:
                    ii = int(i)
                P = self.d[(q, ii)]
                sel = np.where((pdg == q) & (ib == i))[0]
                k = rng.integers(P["n"], size=len(sel))
                a, b = P["off"][k], P["off"][k + 1]
                L = b - a
                if L.sum() == 0:
                    continue
                par = np.repeat(np.arange(len(sel)), L)
                idx = np.repeat(a - np.cumsum(np.concatenate([[0], L[:-1]])), L) + np.arange(L.sum())
                pid = P["pid"][idx]
                pv = P["p"][idx].copy()
                Ep = E[sel]
                f = np.sqrt(np.maximum(Ep ** 2 - M_N ** 2, 1e-12)) / np.sqrt(P["e0"] ** 2 - M_N ** 2)
                pv[:, 2] *= f[par]
                m = _mass(pid)
                ee = np.sqrt(m * m + np.sum(pv * pv, 1))
                # energy guard (projectile + struck nucleus), per final state
                tot = np.bincount(par, weights=ee, minlength=len(sel))
                sc = np.where(tot > Ep + GUARD_MASS, (Ep + GUARD_MASS) / np.maximum(tot, 1e-300), 1.0)
                ee *= sc[par]
                pv *= sc[par][:, None]
                plab = rotate_from_z(u[sel][par], pv)
                outs.append((pid, ee, plab, w[sel][par], band[sel][par]))
        if not outs:
            return None
        return tuple(np.concatenate([o[j] for o in outs]) for j in range(5))


_MASSES = {2212: 0.938272, 2112: 0.939565, 211: M_PI, 321: M_K, 130: 0.497611, 310: 0.497611,
           3122: 1.115683, 13: M_MU}


def _mass(pid):
    a = np.abs(pid)
    out = np.zeros(len(pid))
    for q, m in _MASSES.items():
        out[a == q] = m
    return out


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------
def kernels(primary=("GlobalSplineFitBeta", None), t_floor=0.7943282, n_dec=8,
            seed=20261109, pools=None, cache=True, multigen=True, polarisation=True,
            kmu=True, eloss_rho=None, max_gen=12):
    """``H[band, species, channel, E, alpha]`` (weights) like :func:`eoff_v3.kernels`,
    with the v4 physics switches (module docstring).  ``N_gen[band, gen, E]`` = the
    weight per hadronic generation (all species and channels, diagnostic).  Cached
    in ``$MCEQ3D_CACHE`` under a key that contains every switch."""
    import sys

    from mceq3d_primary import primary_class

    pools = pools or v3.pool_dir()
    opts = f"mg{int(multigen)}|pol{int(polarisation)}|kmu{int(kmu)}|el{eloss_rho}|g{max_gen}"
    key = hashlib.md5(f"{SCHEME}|{primary}|{t_floor:.6g}|{n_dec}|{seed}|{opts}|"
                      f"{os.path.realpath(pools)}".encode()).hexdigest()[:16]
    path = os.path.join(_db.CACHE_ROOT, f"eoffv4_kern_{key}.npz")
    if cache and os.path.exists(path):
        return dict(np.load(path))
    here = os.path.dirname(os.path.abspath(__file__))
    mc3d = os.path.join(os.path.dirname(here), "mc3d")
    if mc3d not in sys.path:
        sys.path.insert(0, mc3d)
    from pool import GRID  # the pool energy binning (tools/mc3d/pool.py)

    pm = primary_class(primary[0])(primary[1])
    rng = np.random.default_rng(seed)
    nE = len(v3.KERNEL_E)
    e_edges = np.sqrt(v3.KERNEL_E[1:] * v3.KERNEL_E[:-1])
    e_edges = np.concatenate([[v3.KERNEL_E[0] ** 2 / e_edges[0]], e_edges,
                              [v3.KERNEL_E[-1] ** 2 / e_edges[-1]]])
    NB = v3.N_BANDS
    nA = len(v3.ALPHA_NODES_DEG)
    H = np.zeros((NB, 4, 3, nE, nA))
    N = np.zeros((NB, 4, 3, nE))
    NG = np.zeros((NB, max_gen + 1, nE))
    SP = v3.SPECIES
    rng_loss = _MuonRange() if eloss_rho else None
    P = _Pools(pools, GRID) if multigen else None

    def tally(band, gen, si, ci, Enu, pnu, w):
        ca = np.clip(pnu[:, 2] / np.maximum(np.linalg.norm(pnu, axis=1), 1e-300), -1, 1)
        a = np.degrees(np.arccos(ca))
        ie = np.searchsorted(e_edges, Enu) - 1
        ia = np.clip(np.searchsorted(v3.ALPHA_EDGES_DEG, a) - 1, 0, nA - 1)
        ok = (ie >= 0) & (ie < nE)
        np.add.at(H, (band[ok], si, ci, ie[ok], ia[ok]), w[ok])
        np.add.at(N, (band[ok], si, ci, ie[ok]), 1.0)
        np.add.at(NG, (band[ok], np.minimum(gen[ok], max_gen), ie[ok]), w[ok])

    def muon_decay(band, gen, Emu, pmu, pl, charge, w):
        """mu -> e nu nu, both neutrinos, mu channel."""
        if rng_loss is not None:
            Ed = rng_loss.decay_energy(rng, Emu, float(eloss_rho))
            ok = np.isfinite(Ed)
            if not ok.any():
                return
            u = pmu[ok] / np.linalg.norm(pmu[ok], axis=1, keepdims=True)
            Emu = Ed[ok]
            pmu = np.sqrt(np.maximum(Emu ** 2 - M_MU ** 2, 0.0))[:, None] * u
            band, gen, pl, charge, w = band[ok], gen[ok], pl[ok], charge[ok], w[ok]
        plu = pl if polarisation else np.zeros(len(Emu))
        for kind in ("numu", "nue"):
            En, pn = michel_polarised(rng, Emu, pmu, plu, charge, kind)
            # mu+ (from pi+/K+): e-like = anti-nu_mu, other = nu_e;  mu-: nu_mu, anti-nu_e
            sp_plus = "antinumu" if kind == "numu" else "nue"
            sp_minus = "numu" if kind == "numu" else "antinue"
            for sp, msk in ((sp_plus, charge > 0), (sp_minus, charge < 0)):
                if msk.any():
                    tally(band[msk], gen[msk], SP.index(sp), 2, En[msk], pn[msk], w[msk])

    def mesons(pid, E, p, w, band, gen):
        """Decays of the charged pions and kaons of a product list."""
        for mes, M in ((211, M_PI), (-211, M_PI), (321, M_K), (-321, M_K)):
            sel = pid == mes
            if not sel.any():
                continue
            pp = np.repeat(p[sel], n_dec, 0)
            Em = np.sqrt(np.sum(pp ** 2, -1) + M * M)
            ww = np.repeat(w[sel], n_dec) / n_dec
            bb = np.repeat(band[sel], n_dec)
            gg = np.repeat(gen[sel], n_dec)
            Enu, pnu, Emu, pmu = v3._two_body(rng, Em, pp, M, M_MU)
            beta = pp / Em[:, None]
            # muon direction in the parent rest frame = - neutrino direction there
            # (recovered by the inverse boost of the neutrino)
            _, pst = v3._boost(Enu, pnu, -beta)
            nst = -pst / np.maximum(np.linalg.norm(pst, axis=1, keepdims=True), 1e-300)
            charge = np.full(len(Em), 1.0 if mes > 0 else -1.0)
            if abs(mes) == 211:
                tally(bb, gg, 0 if mes > 0 else 1, 0, Enu, pnu, ww)
                pl = muon_pl(Emu, pmu, beta, nst, M_PI, charge) if polarisation else np.zeros(len(Em))
                muon_decay(bb, gg, Emu, pmu, pl, charge, ww)
            else:
                for sp in (("numu", "nue") if mes > 0 else ("antinumu", "antinue")):
                    tally(bb, gg, SP.index(sp), 1, Enu, pnu, ww)
                if kmu:
                    pl = muon_pl(Emu, pmu, beta, nst, M_K, charge) if polarisation else np.zeros(len(Em))
                    muon_decay(bb, gg, Emu, pmu, pl, charge, ww * BR_KMU2)

    t_edges = np.asarray(v3.BAND_T)
    for pdg in NUCLEONS:
        for f in sorted(glob.glob(os.path.join(pools, f"pool_{pdg}_*_2000.npz"))):
            d = np.load(f)
            e0 = float(d["e0"][0])
            i = GRID.index(e0)
            lo, hi = float(GRID.edges[i]) - M_N, float(GRID.edges[i + 1]) - M_N
            if i == 0:
                lo = min(lo, t_floor)
            lo = max(lo, t_floor)
            if hi <= lo:
                continue
            nev = int(d["n"][0])
            pid0 = d["pid"].astype(np.int64)
            p0 = np.stack([d[k].astype(float) for k in ("px", "py", "pz")], -1)
            for b in range(NB):
                a0, a1 = max(lo, t_edges[b]), min(hi, t_edges[b + 1])
                if a1 <= a0:
                    continue
                tt = np.geomspace(a0, a1, 9)
                r = pm.p_and_n_flux(tt + M_N)
                phi = float(np.trapezoid(r[1] if pdg == 2212 else r[2], tt))
                if phi <= 0:
                    continue
                w = np.full(len(pid0), phi / nev)
                band = np.full(len(pid0), b, np.int64)
                gen = np.ones(len(pid0), np.int64)
                pid, p = pid0, p0
                m = _mass(pid)
                E = np.sqrt(m * m + np.sum(p * p, 1))
                g = 1
                while True:
                    mesons(pid, E, p, w, band, gen)
                    if not multigen or g >= max_gen:
                        break
                    nuc = np.isin(pid, NUCLEONS) & (E >= POOL_FLOOR_ETOT)
                    if not nuc.any():
                        break
                    u = p[nuc] / np.linalg.norm(p[nuc], axis=1, keepdims=True)
                    res = P.interact(rng, pid[nuc], E[nuc], u, w[nuc], band[nuc])
                    if res is None:
                        break
                    pid, E, p, w, band = res
                    g += 1
                    gen = np.full(len(pid), g, np.int64)
    out = dict(H=H, N=N, N_gen=NG, e=v3.KERNEL_E, e_edges=e_edges,
               alpha_edges=v3.ALPHA_EDGES_DEG, alpha_nodes=v3.ALPHA_NODES_DEG,
               band_T=np.asarray(v3.BAND_T), species=np.array(SP),
               channels=np.array(v3.CHANNELS), primary=np.array(str(primary)),
               t_floor=t_floor, n_dec=n_dec, scheme=np.array(SCHEME), options=np.array(opts))
    if cache:
        os.makedirs(_db.CACHE_ROOT, exist_ok=True)
        np.savez(path, **out)
    return out


def engine_kernels(engine, **kw):
    """v4 kernels for ``engine``'s primary and population floor (``DEFAULTS`` switches)."""
    o = dict(DEFAULTS)
    o.update(kw)
    return kernels(primary=engine._spec["primary"], t_floor=v3.engine_floor_edge(engine), **o)


def build_table(engine, out_path, cz_rows=None, cache_dir=None, kern=None):
    """Write the v4 E_off table (G = 1) for ``engine``'s population: same schema as
    :func:`eoff_v3.build_table`, v4 kernels.  Never overwrites an existing file."""
    import offaxis_mc as ox

    if os.path.exists(out_path):
        raise FileExistsError(f"{out_path} exists; tables are never overwritten")
    cz_rows = np.round(np.arange(0.05, 1.0, 0.1), 2) if cz_rows is None else np.asarray(cz_rows)
    prods = v3.band_profiles(engine, cache_dir)
    if kern is None:
        kern = engine_kernels(engine)
    N, D = v3.table_rows(cz_rows, prods, kern)
    with np.errstate(invalid="ignore", divide="ignore"):
        Eb = np.where(D.sum(2) > 0, N.sum(2) / np.maximum(D.sum(2), 1e-300), 1.0)
        Es = np.where(D.sum((0, 2)) > 0, N.sum((0, 2)) / np.maximum(D.sum((0, 2)), 1e-300), 1.0)
    np.savez(out_path, e=prods[0]["ep_grid"], cz=cz_rows, E_off_s=Es, E_off=Es[0],
             E_off_band_s=Eb, D_band_s=D.sum(2), N_bsc=N, D_bsc=D,
             species=np.array(v3.SPECIES), band_T=np.asarray(v3.BAND_T),
             alpha_nodes=v3.ALPHA_NODES_DEG, kernel_H=kern["H"], kernel_e=kern["e"],
             tag=np.array(engine._tag), cone_mode=np.array("v4-sampled-stratified"),
             scheme=np.array(SCHEME), kernel_options=np.array(str(kern.get("options", ""))),
             provenance=np.array(
                 "eoff_v4.build_table: v3 construction (generator pools, exact decays, full "
                 "sphere, stratified in primary T/n) with all nucleon generations, polarised "
                 "muon decays and K->mu muons in the mu channel; MCEq depth profiles per band "
                 "in the engine population; no reference flux, no MC tally used."),
             **ox._db_keys())
    return out_path


# ---------------------------------------------------------------------------
# zenith-dependent production profiles
# ---------------------------------------------------------------------------
def zenith_profile(engine, band, theta_deg, cache_dir=None):
    """``dict(x_grid, ep_grid, theta, chan={species: (p_dir, p_k, p_mu)})`` for the
    engine's population restricted to ``band`` (``-1`` = all primaries), solved by
    MCEq at ground zenith ``theta_deg``.

    Same channel split as ``offaxis_mc.production_profile``; the depth grid
    resolves the top of the atmosphere at every zenith (150 points to 600 g/cm2,
    then 70 log-spaced to X_max) and the gradient is anchored at X = 0
    (``production_profile``'s 80 linear points put the first node at 190 g/cm2
    at 87 deg and lose the early production).  Cached per engine tag, band and
    zenith."""
    import mceq_config as cfg
    import offaxis_mc as ox
    from mceq3d_db import make_run

    path = _db.dep_path(cache_dir, f"jointprodz_{engine._tag}_{SCHEME}b{band}_t{theta_deg:g}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return dict(x_grid=d["x_grid"], ep_grid=d["ep_grid"], theta=float(d["theta"]),
                    chan={sp: (d[f"p_{sp}_dir"], d[f"p_{sp}_k"], d[f"p_{sp}_mu"])
                          for sp in ox.CHANNEL_SPECIES})
    s = engine._spec
    old_emin = cfg.e_min
    old_floor = getattr(cfg, "minimal_primary_energy", 3.0)
    cfg.e_min = engine.e_min
    cfg.minimal_primary_energy = float(engine.primary_floor)
    try:
        mc = make_run(interaction_model=s["interaction_model"], primary_model=engine._pm,
                      theta_deg=float(theta_deg))
    finally:
        cfg.minimal_primary_energy = old_floor
    if s["atmosphere"] is not None:
        mc.set_density_model(s["atmosphere"])
        mc.set_theta_deg(float(theta_deg))
    if band >= 0:
        e = mc.e_grid
        keep = (e >= v3.BAND_T[band] * 0.999) & (e < v3.BAND_T[band + 1] * 0.999)
        for pid in ((2212, 0), (2112, 0)):
            part = mc.pman[pid]
            mc._phi0[part.lidx:part.uidx] *= keep
    xmax = float(mc.density_model.max_X)
    x = np.unique(np.concatenate([np.linspace(1.0, min(600.0, xmax), 150),
                                  np.geomspace(min(600.0, xmax), xmax, 70)]))
    mc.solve(int_grid=x, grid_var="X")
    e = mc.e_grid
    sel = (e >= engine.e_min) & (e <= 100.0)
    xx = np.concatenate([[0.0], x])

    def prof(key):
        acc = np.array([mc.get_solution(key, 0, grid_idx=i)[sel] for i in range(len(x))])
        aa = np.concatenate([np.zeros((1, acc.shape[1])), acc])
        return np.clip(np.gradient(aa, xx, axis=0), 0.0, None)[1:]

    out = {}
    chan = {}
    for sp in ox.CHANNEL_SPECIES:
        tot = prof(f"total_{sp}")
        k = np.minimum(prof(f"k_{sp}"), tot)
        mu = np.minimum(prof(f"mu_{sp}"), tot)
        dr = np.clip(tot - k - mu, 0.0, None)
        chan[sp] = (dr, k, mu)
        out.update({f"p_{sp}_dir": dr, f"p_{sp}_k": k, f"p_{sp}_mu": mu})
    cfg.e_min = old_emin
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + f".{os.getpid()}.part.npz"
    np.savez(tmp, x_grid=x, ep_grid=e[sel], theta=float(theta_deg), **out)
    os.replace(tmp, path)
    return dict(x_grid=x, ep_grid=e[sel], theta=float(theta_deg), chan=chan)


def band_profiles_z(engine, cache_dir=None, nodes=ZNODES_DEG):
    """Per band: ``dict(cg, x=[x_k], chan=[{species: (dir, k, mu)}], ep_grid)``, the
    ``znodes`` argument of :func:`joint_cone.delivered_joint_factor`.  Species keys
    as ``offaxis_mc.CHANNEL_SPECIES``; :func:`znodes_engine` re-keys them."""
    out = []
    for b in range(v3.N_BANDS):
        prs = [zenith_profile(engine, b, th, cache_dir) for th in nodes]
        out.append(dict(cg=np.cos(np.radians(np.asarray(nodes, float))),
                        x=[q["x_grid"] for q in prs], chan=[q["chan"] for q in prs],
                        ep_grid=prs[0]["ep_grid"]))
    return out


def znodes_engine(zb, species_map):
    """Re-key a :func:`band_profiles_z` entry from channel-species names to the
    engine species (``species_map`` = ``joint_cone.CHANNEL_SPECIES_MAP``)."""
    return dict(cg=zb["cg"], x=zb["x"],
                chan=[{es: ch[cs] for es, cs in species_map.items()} for ch in zb["chan"]])


def rho_dec(cos_theta, zb_all, geom, rho_of_h, e_band=(0.1, 0.4), n_ray=260):
    """Muon-decay-weighted air density [g/cm3] along the arrival line of a row: the
    1D mu-channel production (all bands, nu_e, ``e_band``) at the row zenith
    weights rho(h) on the line of sight.  The density at which the decaying muons
    lost their energy (the eloss kernel family's argument)."""
    import joint_cone as jc
    import offaxis_mc as ox

    h_ray, _, psi_o, w = jc.arrival_ray(cos_theta, n_ray=n_ray)
    h_grid, psi_grid, table = geom
    x_o = ox._interp_xslant(h_ray, psi_o, h_grid, psi_grid, table)
    ep = zb_all[0]["ep_grid"]
    m = (ep >= e_band[0]) & (ep < e_band[1])
    num = den = 0.0
    for zb in zb_all:
        zn = dict(cg=zb["cg"], x=zb["x"], p=[ch["nue"][2] for ch in zb["chan"]])
        p = jc._p_lookup(x_o, h_ray, psi_o, None, ep, None, zn)[:, m].sum(1)
        num += np.sum(w * p * rho_of_h(h_ray))
        den += np.sum(w * p)
    return float(num / den) if den > 0 else float(RHO_FAMILY[2])


def kernel_family(engine, rhos=RHO_FAMILY, **switches):
    """``[(rho, kern)]`` energy-loss kernels for ``rhos`` (``switches`` as
    :func:`kernels`)."""
    o = dict(DEFAULTS)
    o.update(switches)
    o.pop("eloss_rho", None)
    return [(float(r), kernels(primary=engine._spec["primary"],
                               t_floor=v3.engine_floor_edge(engine), eloss_rho=float(r), **o))
            for r in rhos]


def kernel_W_rho(family, rho, band, species, channel, e_grid):
    """Normalised kernel at density ``rho``: linear interpolation in ln rho between
    the family members (clamped to the end members)."""
    rs = np.array([f[0] for f in family])
    lr = np.log(np.clip(rho, rs[0], rs[-1]))
    j = int(np.clip(np.searchsorted(np.log(rs), lr) - 1, 0, len(rs) - 2))
    t = (lr - np.log(rs[j])) / (np.log(rs[j + 1]) - np.log(rs[j]))
    W0 = v3.kernel_W(family[j][1], band, species, channel, e_grid)
    W1 = v3.kernel_W(family[j + 1][1], band, species, channel, e_grid)
    return (1 - t) * W0 + t * W1


def table_rows_v4(cz_rows, prods, zbands, Wfun, n_beta=18, n_ray=260):
    """``N[b, s, c, iz, E]``, ``D[...]`` at G = 1 with zenith-node profiles
    (``zbands``, or ``None`` for the single profile) and row-dependent kernels
    ``Wfun(iz, cz, b, species, channel, e_grid)``."""
    import joint_cone as jc

    ep = prods[0]["ep_grid"]
    alpha = np.deg2rad(v3.ALPHA_NODES_DEG)
    beta = np.linspace(0.0, 2 * np.pi, n_beta, endpoint=False)
    N = np.zeros((v3.N_BANDS, 4, 3, len(cz_rows), len(ep)))
    D = np.zeros_like(N)
    for iz, cz in enumerate(cz_rows):
        for b in range(v3.N_BANDS):
            plist = [prods[b]["chan"][s][ic] for s in v3.SPECIES for ic in range(3)]
            zn = None
            if zbands is not None:
                zb = zbands[b]
                zn = dict(cg=zb["cg"], x=zb["x"],
                          profiles=[[ch[s][ic] for s in v3.SPECIES for ic in range(3)]
                                    for ch in zb["chan"]])
            qs, qas = jc.cone_production_multi(float(cz), alpha, beta, prods[0]["x_grid"],
                                               ep, plist, prods[0]["geom"], n_ray=n_ray,
                                               znodes=zn)
            i = 0
            for si, s in enumerate(v3.SPECIES):
                for ic, c in enumerate(v3.CHANNELS):
                    W = Wfun(iz, float(cz), b, s, c, ep)
                    N[b, si, ic, iz] = np.einsum("ea,ae->e", W, qs[i].mean(axis=1))
                    D[b, si, ic, iz] = qas[i]
                    i += 1
    return N, D


def engine_inputs(engine, cache_dir=None, config=None):
    """Everything the v4 engine path needs: ``prods`` (eoff_v3 band profiles: geometry
    and the theta = 0 profile), ``zbands`` (zenith nodes or None), the kernels
    (single, or the eloss family) and ``Wfun``."""
    import offaxis_mc as ox

    cfgv = dict(V4_CONFIG if config is None else config)
    prods = v3.band_profiles(engine, cache_dir)
    zbands = band_profiles_z(engine, cache_dir) if cfgv.get("zdep") else None
    sw = {k: cfgv[k] for k in ("multigen", "polarisation", "kmu")}
    if cfgv.get("eloss") == "row":
        fam = kernel_family(engine, **sw)
        rho_cache = {}

        def rho_row(cz):
            key = round(float(cz), 6)
            if key not in rho_cache:
                zb_all = zbands if zbands is not None else [
                    dict(cg=np.array([1.0]), x=[prods[b]["x_grid"]],
                         chan=[prods[b]["chan"]], ep_grid=prods[b]["ep_grid"])
                    for b in range(v3.N_BANDS)]
                rho_cache[key] = rho_dec(float(cz), zb_all, prods[0]["geom"], ox._RHO)
            return rho_cache[key]

        def Wfun(iz, cz, b, s, c, e_grid):
            return kernel_W_rho(fam, rho_row(abs(cz)), b, s, c, e_grid)
        kern = fam
    else:
        er = cfgv.get("eloss")
        kern = kernels(primary=engine._spec["primary"], t_floor=v3.engine_floor_edge(engine),
                       eloss_rho=(None if er in (None, False) else float(er)), **sw)
        rho_row = None

        def Wfun(iz, cz, b, s, c, e_grid):
            return v3.kernel_W(kern, b, s, c, e_grid)
    return dict(prods=prods, zbands=zbands, kern=kern, Wfun=Wfun, rho_row=rho_row, config=cfgv)


def build_table_v4(engine, out_path, cz_rows=None, cache_dir=None, config=None):
    """The v4 E_off table (G = 1) with :data:`V4_CONFIG` (zenith-node profiles and
    row-dependent energy-loss kernels).  Same schema as :func:`eoff_v3.build_table`
    plus ``rho_dec`` per row.  Never overwrites an existing file."""
    import offaxis_mc as ox

    if os.path.exists(out_path):
        raise FileExistsError(f"{out_path} exists; tables are never overwritten")
    cz_rows = np.round(np.arange(0.05, 1.0, 0.1), 2) if cz_rows is None else np.asarray(cz_rows)
    inp = engine_inputs(engine, cache_dir, config)
    N, D = table_rows_v4(cz_rows, inp["prods"], inp["zbands"], inp["Wfun"])
    with np.errstate(invalid="ignore", divide="ignore"):
        Eb = np.where(D.sum(2) > 0, N.sum(2) / np.maximum(D.sum(2), 1e-300), 1.0)
        Es = np.where(D.sum((0, 2)) > 0, N.sum((0, 2)) / np.maximum(D.sum((0, 2)), 1e-300), 1.0)
    rr = np.array([inp["rho_row"](c) if inp["rho_row"] else np.nan for c in cz_rows])
    np.savez(out_path, e=inp["prods"][0]["ep_grid"], cz=cz_rows, E_off_s=Es, E_off=Es[0],
             E_off_band_s=Eb, D_band_s=D.sum(2), N_bsc=N, D_bsc=D,
             species=np.array(v3.SPECIES), band_T=np.asarray(v3.BAND_T),
             alpha_nodes=v3.ALPHA_NODES_DEG, rho_dec=rr, znodes_deg=np.asarray(ZNODES_DEG),
             tag=np.array(engine._tag), cone_mode=np.array("v4-sampled-stratified-zdep"),
             scheme=np.array(SCHEME), config=np.array(str(inp["config"])),
             provenance=np.array(
                 "eoff_v4.build_table_v4: generator-pool kernels with all nucleon generations, "
                 "polarised muon decays, K->mu muons and muon energy loss at the row's "
                 "muon-decay air density (kernel family in ln rho); zenith-dependent MCEq depth "
                 "profiles per band (ground-zenith nodes ZNODES_DEG); no reference flux, no MC "
                 "tally used."),
             **ox._db_keys())
    return out_path
