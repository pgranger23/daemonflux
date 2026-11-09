"""E_off v3: a first-principles, primary-stratified production cone.

Why (2026-11-08, ``$W/runs/fix1/REPORT.md``).  The v2 cone (``offaxis_mc.build_channel``
-> ``offaxis_excess_channel_v2.npz`` and the joint path of ``mceq3d_flux``) parametrises
the distribution of the primary direction given the neutrino direction by a
Gaussian-on-the-sphere of space-angle RMS ``sigma(E)``, truncated at 89 deg.  Below
0.2 GeV that width is not computed but clamped:

* ``kinematic_kernel._sigma_binned`` starts at 0.15 GeV and clamps below its first bin
  centre (0.168 GeV): sigma_mu is pinned at 35-37 deg;
* ``fokker_planck_3d.load_theta2`` pools the generator meson angle only from 0.3 GeV
  and clamps below 0.36 GeV (40 deg), while the generator gives 45-60 deg at
  0.15-0.3 GeV;
* ``channel_shapes`` samples pions only above 0.5 GeV and uses the beta = 1 decay
  (no backward neutrinos), and extrapolates a power law below 0.2 GeV;
* the 89-deg truncation renormalises away the tail beyond 90 deg (8-13% of the
  muon-decay neutrinos at 0.1-0.3 GeV);
* one kernel for all primaries: the cone of a 0.1-GeV neutrino is far wider for a
  2-GeV/n primary than for a 50-GeV/n one, so the cutoff, which removes the
  low-energy primaries, changes the effective cone ("primary-dependent E_off").

v3 replaces the width by the **sampled angular distribution** and stratifies it by
primary energy:

* kernel ``W_b(alpha | E_nu)`` per primary band ``b`` (T/n edges :data:`BAND_T`),
  species and parent channel (dir = pi -> nu_mu, k = K -> nu, mu = pi -> mu -> nu), on
  a full-sphere alpha grid (0-180 deg).  Single interaction, as the engine's cone has
  always been: a primary nucleon (GSF p/n flux) interacts with air; the exclusive
  final state is the generator event pool ``$MC3D_POOL_CACHE`` (DPMJET-III-19.3 below
  80 GeV, SIBYLL-2.3d above: the same models as the MCEq database ``lext_dpm193``, i.e.
  the engine's own yields); the TRUE lab angle atan2(p_T, p_z) is kept (backward
  production included); pi/K -> mu nu exact two-body (beta < 1); mu -> e nu nu with
  the unpolarised Michel spectra; directions composed as 3-vectors.
* depth profiles ``p_b(X, E)`` per band, species and channel: MCEq depth-resolved,
  in the engine's OWN population (primary, floor, e_min) with the primary restricted
  to band ``b``.
* band-resolved cutoff response ``G_b(E, R_c)``: exact, from per-primary-bin MCEq
  Green's functions (MCEq is linear in ``_phi0``), :func:`band_green` /
  :func:`band_G`.  ``sum_b Phi_b G_b = Phi G_s`` reproduces ``geomag_response``.

The joint factor becomes ``F_s = sum_b J_{b,s} / sum_b D_{b,s}`` with each band's own
profile, kernel and cutoff response (``MCEq3DFlux(eoff_model="v3")``).

Not included (documented limits): multiple hadronic generations (the cone is a
single-interaction kernel; the MC's multi-generation share at 0.1 GeV is 35-40% with
a ~6 deg wider RMS), muon polarisation, muons from kaon decays in the mu channel,
K_l3 kinematics (the k channel uses K_mu2; its share is <= 0.5% below 0.3 GeV),
backward angles beyond what the pools contain, pool Feynman rescaling inside a
pool bin (bin-centre kinematics are used).
"""

from __future__ import annotations

import glob
import hashlib
import os

import numpy as np

import mceq3d_db as _db

#: primary bands in kinetic energy per nucleon [GeV]: [0.794, 1) (below the MC's
#: pool floor; populated only for floors < 2 GeV), then the MC strata h = 0..4
#: (half decades from 1 GeV), then everything above 316 GeV/n.
BAND_T = (0.7943282, 1.0, 10 ** 0.5, 10.0, 10 ** 1.5, 100.0, 10 ** 2.5, np.inf)
N_BANDS = len(BAND_T) - 1
#: MC stratum h <-> band h + 1 (h = 0..4); band 6 = MC strata 5 and 6 (and above)
ALPHA_EDGES_DEG = np.linspace(0.0, np.sqrt(180.0), 61) ** 2
ALPHA_NODES_DEG = 0.5 * (ALPHA_EDGES_DEG[1:] + ALPHA_EDGES_DEG[:-1])
#: kernel energy nodes = MCEq's grid centres (10/decade), 0.0282-100 GeV
KERNEL_E = 10 ** (np.arange(-15.5, 20.5) / 10.0)
SPECIES = ("numu", "antinumu", "nue", "antinue")  # = offaxis_mc.CHANNEL_SPECIES
CHANNELS = ("dir", "k", "mu")
SCHEME = "v3a"  # part of every v3 cache key
M_PI, M_MU, M_K, M_N = 0.13957, 0.105658, 0.493677, 0.938272
#: T/n below which every MCEq primary bin gets its own Green's function (above it no
#: cutoff <= RC_MAX_GV can act: free protons are transmitted from T = 54 GeV at 55 GV)
T_SINGLE_MAX = 100.0


def band_index(t_gev):
    """Band of kinetic energy per nucleon ``t_gev`` (-1 below the first edge)."""
    return int(np.searchsorted(np.asarray(BAND_T), float(t_gev), side="right") - 1)


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------
def _boost(E, p, beta_vec):
    b2 = np.sum(beta_vec ** 2, -1)
    g = 1.0 / np.sqrt(np.maximum(1.0 - b2, 1e-300))
    bp = np.sum(beta_vec * p, -1)
    gm1 = np.where(b2 > 0, (g - 1.0) / np.maximum(b2, 1e-300), 0.0)
    return g * (E + bp), p + (gm1 * bp + g * E)[..., None] * beta_vec


def _iso(rng, n):
    c = rng.uniform(-1, 1, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    s = np.sqrt(1 - c * c)
    return np.stack([s * np.cos(ph), s * np.sin(ph), c], -1)


def _michel(rng, n, kind):
    out = np.empty(n)
    f = 0
    fmax = 2.0 if kind == "numu" else 1.78
    while f < n:
        m = n - f
        x = rng.random(m)
        val = 2 * x ** 2 * (3 - 2 * x) if kind == "numu" else 12 * x ** 2 * (1 - x)
        acc = rng.random(m) * fmax < val
        k = int(acc.sum())
        out[f:f + k] = x[acc]
        f += k
    return out


def _two_body(rng, Ep, pvec, M, m1):
    n = len(Ep)
    estar = (M * M - m1 * m1) / (2 * M)
    d = _iso(rng, n)
    beta = pvec / Ep[:, None]
    Enu, pnu = _boost(np.full(n, estar), estar * d, beta)
    E1, p1 = _boost(np.full(n, M - estar), -estar * d, beta)
    return Enu, pnu, E1, p1


def pool_dir():
    return os.environ.get("MC3D_POOL_CACHE",
                          os.path.join(_db.W, "pools_fix"))


def kernels(primary=("GlobalSplineFitBeta", None), t_floor=0.7943282, n_dec=8,
            seed=20261108, pools=None, cache=True):
    """Tallied kernels ``H[band, species, channel, E, alpha]`` (weights, not yet
    normalised) for the primary population ``primary`` above ``t_floor`` [GeV/n,
    the LOWER EDGE of MCEq's first populated bin].  Cached in ``$MCEQ3D_CACHE``.

    Each pool bin (projectile p or n, total energy bin [E_lo, E_hi]) is weighted by
    the primary nucleon flux integrated over its overlap with each band (and with
    ``T >= t_floor``); its events are equally weighted.  The lowest pool bin also
    carries the primaries between ``t_floor`` and the pool floor (2 GeV total).
    """
    import sys

    from mceq3d_primary import primary_class

    pools = pools or pool_dir()
    key = hashlib.md5(f"{SCHEME}|{primary}|{t_floor:.6g}|{n_dec}|{seed}|"
                      f"{os.path.realpath(pools)}".encode()).hexdigest()[:16]
    path = os.path.join(_db.CACHE_ROOT, f"eoffv3_kern_{key}.npz")
    if cache and os.path.exists(path):
        return dict(np.load(path))
    here = os.path.dirname(os.path.abspath(__file__))
    mc3d = os.path.join(os.path.dirname(here), "mc3d")
    if mc3d not in sys.path:
        sys.path.insert(0, mc3d)
    from pool import GRID  # the pool energy binning (tools/mc3d/pool.py)

    pm = primary_class(primary[0])(primary[1])
    rng = np.random.default_rng(seed)
    nE = len(KERNEL_E)
    e_edges = np.sqrt(KERNEL_E[1:] * KERNEL_E[:-1])
    e_edges = np.concatenate([[KERNEL_E[0] ** 2 / e_edges[0]], e_edges,
                              [KERNEL_E[-1] ** 2 / e_edges[-1]]])
    H = np.zeros((N_BANDS, 4, 3, nE, len(ALPHA_NODES_DEG)))
    N = np.zeros((N_BANDS, 4, 3, nE))

    def tally(b, si, ci, Enu, pnu, w):
        ca = np.clip(pnu[:, 2] / np.maximum(np.linalg.norm(pnu, axis=1), 1e-300), -1, 1)
        a = np.degrees(np.arccos(ca))
        ie = np.searchsorted(e_edges, Enu) - 1
        ia = np.clip(np.searchsorted(ALPHA_EDGES_DEG, a) - 1, 0, len(ALPHA_NODES_DEG) - 1)
        ok = (ie >= 0) & (ie < nE)
        np.add.at(H[b, si, ci], (ie[ok], ia[ok]), w[ok])
        np.add.at(N[b, si, ci], ie[ok], 1.0)

    t_edges = np.asarray(BAND_T)
    for pdg in (2212, 2112):
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
            pid = d["pid"]
            px, py, pz = (d[k].astype(float) for k in ("px", "py", "pz"))
            for b in range(N_BANDS):
                a0, a1 = max(lo, t_edges[b]), min(hi, t_edges[b + 1])
                if a1 <= a0:
                    continue
                tt = np.geomspace(a0, a1, 9)
                r = pm.p_and_n_flux(tt + M_N)
                phi = float(np.trapezoid(r[1] if pdg == 2212 else r[2], tt))
                if phi <= 0:
                    continue
                wev = phi / nev
                for mes, M in ((211, M_PI), (-211, M_PI), (321, M_K), (-321, M_K)):
                    sel = pid == mes
                    if not sel.any():
                        continue
                    p = np.repeat(np.stack([px[sel], py[sel], pz[sel]], -1), n_dec, 0)
                    Em = np.sqrt(np.sum(p ** 2, -1) + M * M)
                    w = np.full(len(Em), wev / n_dec)
                    Enu, pnu, Emu, pmu = _two_body(rng, Em, p, M, M_MU)
                    if abs(mes) == 211:
                        tally(b, 0 if mes > 0 else 1, 0, Enu, pnu, w)
                        beta = pmu / Emu[:, None]
                        pairs = ((("antinumu", "numu"), ("nue", "nue")) if mes > 0
                                 else (("numu", "numu"), ("antinue", "nue")))
                        for sp, kind in pairs:
                            es = 0.5 * M_MU * _michel(rng, len(Emu), kind)
                            En2, pn2 = _boost(es, es[:, None] * _iso(rng, len(Emu)), beta)
                            tally(b, SPECIES.index(sp), 2, En2, pn2, w)
                    else:
                        for sp in (("numu", "nue") if mes > 0 else ("antinumu", "antinue")):
                            tally(b, SPECIES.index(sp), 1, Enu, pnu, w)
    out = dict(H=H, N=N, e=KERNEL_E, e_edges=e_edges, alpha_edges=ALPHA_EDGES_DEG,
               alpha_nodes=ALPHA_NODES_DEG, band_T=np.asarray(BAND_T),
               species=np.array(SPECIES), channels=np.array(CHANNELS),
               primary=np.array(str(primary)), t_floor=t_floor, n_dec=n_dec,
               scheme=np.array(SCHEME))
    if cache:
        os.makedirs(_db.CACHE_ROOT, exist_ok=True)
        np.savez(path, **out)
    return out


def kernel_W(kern, band, species, channel, e_grid):
    """Normalised ``W[len(e_grid), n_alpha]`` (sin-measure included: tallies per
    alpha bin).  ``band=None`` sums all bands.  Fallbacks: an empty (band, E) cell
    takes the all-band kernel at that E, then the nearest populated E; nu_e /
    anti-nu_e ``dir`` (helicity-suppressed, MCEq's p_dir ~ 0) takes the pion cone of
    nu_mu / anti-nu_mu."""
    if channel == "dir" and "nue" in species:
        species = "numu" if species == "nue" else "antinumu"
    sp = [str(s) for s in kern["species"]]
    si, ci = sp.index(species), CHANNELS.index(channel)
    H = kern["H"]
    hall = H[:, si, ci].sum(0)
    h = hall if band is None else H[band, si, ci]
    ke = np.asarray(kern["e"])
    W = np.zeros((len(e_grid), H.shape[-1]))
    pop = np.where(hall.sum(1) > 0)[0]
    for j, e in enumerate(np.asarray(e_grid, float)):
        i = int(np.argmin(np.abs(np.log(ke / e))))
        row = h[i] if h[i].sum() > 0 else hall[i]
        if row.sum() <= 0:
            row = hall[pop[np.argmin(np.abs(pop - i))]]
        W[j] = row / row.sum()
    return W


# ---------------------------------------------------------------------------
# band-resolved depth profiles (engine population)
# ---------------------------------------------------------------------------
def band_profiles(engine, cache_dir=None):
    """``[prod_b]`` for b in bands: ``dict(x_grid, ep_grid, chan={species: (dir,
    k, mu)}, p_tot, p_k, geom)`` from MCEq depth-resolved solves in the ENGINE's
    population with the primary restricted to band ``b``.  Cached per engine tag."""
    import offaxis_mc as ox
    import joint_cone as jc

    s = engine._spec
    out = []
    for b in range(N_BANDS):
        lo, hi = BAND_T[b], BAND_T[b + 1]

        def hook(mc, lo=lo, hi=hi):
            e = mc.e_grid
            keep = (e >= lo * 0.999) & (e < hi * 0.999)
            for pid in ((2212, 0), (2112, 0)):
                part = mc.pman[pid]
                mc._phi0[part.lidx:part.uidx] *= keep

        path = _db.dep_path(cache_dir, f"jointprod_chan_{engine._tag}_{SCHEME}b{b}.npz")
        out.append(jc.load_production(
            cache=path, species=ox.CHANNEL_SPECIES, e_lo=engine.e_min,
            interaction_model=s["interaction_model"], primary_model=engine._pm,
            primary_floor=engine.primary_floor, atmosphere=s["atmosphere"],
            phi0_hook=hook))
    return out


# ---------------------------------------------------------------------------
# band-resolved cutoff response: per-primary-bin Green's functions
# ---------------------------------------------------------------------------
def band_green(engine, cz_ref, cache_dir=None):
    """Per-primary-bin neutrino fluxes at zenith ``cz_ref`` for the band split.

    For every populated MCEq primary bin ``k`` with T < :data:`T_SINGLE_MAX`: the
    flux from its protons alone ``Phi_p[k]`` and its neutrons alone ``Phi_n[k]``;
    plus the bands entirely above (never cut for R_c <= 55 GV) and the total.  MCEq
    is linear in ``_phi0``, so any cutoff response follows exactly (:func:`band_G`).
    Cached (``gsb_*.npz``) per engine tag and zenith.
    """
    from mceq3d_flux import SPECIES as ESP

    mq, phi0_std, p_sl, n_sl, f_free, az_bound = engine._gs_engine()
    e = np.asarray(engine.e, float)
    h = hashlib.md5(f"{engine._gs_tag}|{float(cz_ref):.4f}|{SCHEME}|{T_SINGLE_MAX}"
                    .encode()).hexdigest()[:16]
    fpath = _db.dep_path(cache_dir, f"gsb_{h}.npz") if cache_dir is not None else None
    if fpath is not None and os.path.exists(fpath):
        d = np.load(fpath)
        if d["e"].shape == e.shape and np.allclose(d["e"], e):
            return {k: d[k] for k in d.files}
    mq.set_theta_deg(np.degrees(np.arccos(np.clip(cz_ref, 1e-3, 1))))

    def run(phi):
        mq._phi0[:] = phi
        mq.solve()
        return np.array([mq.get_solution(s, 0).copy() for s in ESP])

    tot = run(phi0_std)
    pop = np.where(phi0_std[p_sl] > 0)[0]
    ks = np.array([k for k in pop if e[k] < T_SINGLE_MAX], int)
    Pp = np.zeros((len(ks), len(ESP), len(e)))
    Pn = np.zeros_like(Pp)
    for j, k in enumerate(ks):
        phi = np.zeros_like(phi0_std)
        phi[p_sl.start + k] = phi0_std[p_sl.start + k]
        Pp[j] = run(phi)
        phi = np.zeros_like(phi0_std)
        phi[n_sl.start + k] = phi0_std[n_sl.start + k]
        Pn[j] = run(phi)
    # bands entirely above T_SINGLE_MAX
    hi_b = [b for b in range(N_BANDS) if BAND_T[b] >= T_SINGLE_MAX * 0.999]
    Pb = np.zeros((N_BANDS, len(ESP), len(e)))
    for b in hi_b[:-1]:
        keep = (e >= BAND_T[b] * 0.999) & (e < BAND_T[b + 1] * 0.999)
        phi = phi0_std.copy()
        phi[p_sl] *= keep
        phi[n_sl] *= keep
        Pb[b] = run(phi)
    if hi_b:
        Pb[hi_b[-1]] = tot - Pp.sum(0) - Pn.sum(0) - Pb[hi_b[:-1]].sum(0)
    mq._phi0[:] = phi0_std
    out = dict(e=e, k=ks, T_k=e[ks], band_k=np.array([band_index(t) for t in e[ks]]),
               Phi_p=Pp, Phi_n=Pn, Phi_band_hi=Pb, Phi_tot=tot,
               f_free=np.asarray(f_free), az_bound=np.asarray(az_bound),
               cz_ref=float(cz_ref))
    if fpath is not None:
        os.makedirs(os.path.dirname(fpath) or ".", exist_ok=True)
        np.savez(fpath, **out)
    return out


def band_G(green, rc_grid, sigma_lnr=None):
    """``(G_b, Phi_b, G_tot)`` from :func:`band_green`:

    * ``G_b[b][species][n_rc, nE]`` = cut / uncut flux of band ``b`` (1 where the band
      has no flux),
    * ``Phi_b[b][species][nE]`` = uncut band flux (the 1D band weights),
    * ``G_tot[species][n_rc, nE]`` = ``sum_b Phi_b G_b / Phi_tot`` -- identical to
      ``MCEq3DFlux.geomag_response`` (same bin-averaged transmission).
    """
    from mceq3d_flux import SPECIES as ESP, _nucleon_transmission

    e = green["e"]
    ks, bk = green["k"], green["band_k"]
    f_free, az_bound = green["f_free"], green["az_bound"]
    Pp, Pn, Pb, tot = green["Phi_p"], green["Phi_n"], green["Phi_band_hi"], green["Phi_tot"]
    rc_grid = np.asarray(rc_grid, float)
    Phi_b = np.zeros((N_BANDS, len(ESP), len(e)))
    for b in range(N_BANDS):
        m = bk == b
        Phi_b[b] = Pp[m].sum(0) + Pn[m].sum(0) + Pb[b]
    cut = np.repeat(Phi_b[:, :, None, :], len(rc_grid), axis=2)  # (b, s, rc, E)
    for j, rc in enumerate(rc_grid):
        t1 = _nucleon_transmission(e, rc, sigma_lnr, az_over_z=1.0)
        t2 = _nucleon_transmission(e, rc, sigma_lnr, az_over_z=az_bound)
        tp = f_free * t1 + (1.0 - f_free) * t2
        for b in range(N_BANDS):
            m = bk == b
            if not m.any():
                continue
            cut[b, :, j] = (np.einsum("k,kse->se", tp[ks[m]], Pp[m])
                            + np.einsum("k,kse->se", t2[ks[m]], Pn[m]) + Pb[b])
    G_b, Phi = [], []
    with np.errstate(invalid="ignore", divide="ignore"):
        for b in range(N_BANDS):
            G_b.append({s: np.where(Phi_b[b, i][None, :] > 0,
                                    cut[b, i] / Phi_b[b, i][None, :], 1.0)
                        for i, s in enumerate(ESP)})
            Phi.append({s: Phi_b[b, i] for i, s in enumerate(ESP)})
        G_tot = {s: np.where(tot[i][None, :] > 0, cut[:, i].sum(0) / tot[i][None, :], 1.0)
                 for i, s in enumerate(ESP)}
    return G_b, Phi, G_tot


# ---------------------------------------------------------------------------
# G = 1 table (up-going path, out-of-range energies, gates)
# ---------------------------------------------------------------------------
def table_rows(cz_rows, prods, kern, n_beta=18, n_ray=260):
    """``N[b, s, c, iz, E]``, ``D[b, s, c, iz, E]`` at G = 1 on the band profiles'
    common energy grid, with the v3 kernels (full-sphere alpha grid)."""
    import joint_cone as jc

    ep = prods[0]["ep_grid"]
    alpha = np.deg2rad(ALPHA_NODES_DEG)
    beta = np.linspace(0.0, 2 * np.pi, n_beta, endpoint=False)
    Ws = {(b, s, c): kernel_W(kern, b, s, c, ep) for b in range(N_BANDS)
          for s in SPECIES for c in CHANNELS}
    N = np.zeros((N_BANDS, 4, 3, len(cz_rows), len(ep)))
    D = np.zeros_like(N)
    for iz, cz in enumerate(cz_rows):
        plist = [prods[b]["chan"][s][ic] for b in range(N_BANDS) for s in SPECIES
                 for ic in range(3)]
        qs, qas = jc.cone_production_multi(float(cz), alpha, beta, prods[0]["x_grid"],
                                           ep, plist, prods[0]["geom"], n_ray=n_ray)
        i = 0
        for b in range(N_BANDS):
            for si, s in enumerate(SPECIES):
                for ic, c in enumerate(CHANNELS):
                    N[b, si, ic, iz] = np.einsum("ea,ae->e", Ws[(b, s, c)],
                                                 qs[i].mean(axis=1))
                    D[b, si, ic, iz] = qas[i]
                    i += 1
    return N, D


def build_table(engine, out_path, cz_rows=None, cache_dir=None, kern=None):
    """Write the v3 E_off table (G = 1) for ``engine``'s population.

    Keys: ``e``, ``cz``, ``E_off_s[species, cz, E]`` (all primaries), ``E_off``
    (= the numu row, flavour-blind slot of the old schema), ``E_off_band_s[band,
    species, cz, E]``, ``D_band_s`` (the band's 1D production, same shape), the
    kernels, ``band_T``, ``tag`` (engine tag), database keys and provenance.
    Never overwrites an existing file.
    """
    import offaxis_mc as ox

    if os.path.exists(out_path):
        raise FileExistsError(f"{out_path} exists; v3 tables are never overwritten")
    cz_rows = np.round(np.arange(0.05, 1.0, 0.1), 2) if cz_rows is None else np.asarray(cz_rows)
    prods = band_profiles(engine, cache_dir)
    if kern is None:
        kern = kernels(primary=engine._spec["primary"],
                       t_floor=engine_floor_edge(engine))
    N, D = table_rows(cz_rows, prods, kern)
    with np.errstate(invalid="ignore", divide="ignore"):
        Eb = np.where(D.sum(2) > 0, N.sum(2) / np.maximum(D.sum(2), 1e-300), 1.0)
        Es = np.where(D.sum((0, 2)) > 0, N.sum((0, 2)) / np.maximum(D.sum((0, 2)), 1e-300), 1.0)
    np.savez(out_path, e=prods[0]["ep_grid"], cz=cz_rows, E_off_s=Es, E_off=Es[0],
             E_off_band_s=Eb, D_band_s=D.sum(2), N_bsc=N, D_bsc=D,
             species=np.array(SPECIES), band_T=np.asarray(BAND_T),
             alpha_nodes=ALPHA_NODES_DEG, kernel_H=kern["H"], kernel_e=kern["e"],
             tag=np.array(engine._tag), cone_mode=np.array("v3-sampled-stratified"),
             scheme=np.array(SCHEME),
             provenance=np.array(
                 "eoff_v3.build_table: first-principles single-interaction kernels from the "
                 "generator pools (DPMJET-III-19.3 < 80 GeV, SIBYLL-2.3d above), exact decays, "
                 "full sphere, stratified in primary T/n (band_T); MCEq depth profiles per band "
                 "in the engine population; no reference flux, no MC tally used."),
             **ox._db_keys())
    return out_path


def engine_floor_edge(engine):
    """Lower edge [GeV/n] of the engine's first populated primary bin."""
    e = np.asarray(engine.e, float)
    k = engine._floor_index()
    dl = np.log(e[1] / e[0])
    return float(e[k] * np.exp(-0.5 * dl))
