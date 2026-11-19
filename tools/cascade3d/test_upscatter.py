"""Up-scatter into transported species and the features of the MC-matched sets.

Reference: in a constant-density atmosphere the straight-ahead problem along a
straight line has constant coefficients in grammage X,
``d phi / dX = M phi`` with ``M = sum_channels rate (Y - delta)`` over all species and
bins (sinks included, not attenuated), so ``phi(X) = expm(M X) phi(0)`` exactly.  This
is independent of the solver: no march, no source interpolation, up-scatter (child
bin above parent bin) included as ordinary matrix entries.

The kernel set (``upscatter_set``) is small but has the features of the Monte
Carlo-matched sets (``DESIGN.md`` section 3):

* exact decay kinematics on the kinetic-energy grid from ``$N3/kernels/decay.npz``
  (K+ -> pi+, mu+_l, mu+_r, numu, nue; K_S -> pi+; pi+ -> mu+_l, mu+_r, numu;
  mu+_l, mu+_r -> antinumu, nue): up to 84% of a yield above the parent's bin;
* K+ and pi+ transported at all energies (nothing folded), K_S with a decay length of
  centimetres (thick), K+ with an interaction of zero yield (pure absorption), pi+
  with zero interaction rate below a threshold;
* a nucleon-like parent with a top intensity only between 1 and 1000 GeV;
* a neutron-like species with a decay rate and no decay channel (zero yield) and an
  interaction of zero yield;
* helicity muons (loss species, dedx = 0 so that expm is exact) decaying into sinks.

Grid: levels every 50 m plus 21 levels geometric from 1 m to 3 km below the top.
Constant density puts a large share of the grammage just below the top, where
parents with decay lengths of 0.1-1 km are still rising from zero (boundary layer);
with the default top layer that costs up to 0.5% (first order in the spacing).  This
is a property of the test geometry: the real atmosphere has 1e-5 g/cm2 in its top
200 m.

Run: ``taskset -c 0-3 $PY -m pytest tools/cascade3d/test_upscatter.py -q``
"""

import os

import numpy as np
import pytest

import grid
import kernelset as K
import rays
import solver as S

N3 = os.environ.get("N3", "/home/pigrange/mc3d_work/runs/native3d")
DECAY = os.path.join(N3, "kernels", "decay.npz")
NB = 40                     # PHYS bins 0..39: 0.028 to 220 GeV
RHO, HTOP = 1.0e-3, 10.0e5  # X(vertical) = 1000 g/cm2
M_K, M_PI, M_MU, M_N = 0.493677, 0.13957, 0.10565837, 0.938272
CTAU = {"K+": 371.2, "K_S0": 2.6844, "pi+": 780.45, "mu+_l": 65865.4,
        "mu+_r": 65865.4}
NAMES = ["P", "K+", "K_S0", "pi+", "mu+_l", "mu+_r", "numu", "antinumu", "nue", "n0"]
DECAYS = [("K+", "pi+"), ("K+", "mu+_l"), ("K+", "mu+_r"), ("K+", "numu"),
          ("K+", "nue"), ("K_S0", "pi+"), ("pi+", "mu+_l"), ("pi+", "mu+_r"),
          ("pi+", "numu"), ("mu+_l", "antinumu"), ("mu+_l", "nue"),
          ("mu+_r", "antinumu"), ("mu+_r", "nue")]


def _decay_yield(z, parent, child, nb):
    pp, pc, ptr = z["pair_parent"], z["pair_child"], z["pair_ptr"]
    i = int(np.flatnonzero((pp == parent) & (pc == child))[0])
    s = slice(ptr[i], ptr[i + 1])
    k, m, y0 = z["k"][s], z["m"][s], z["mom"][s, 0]
    Y = np.zeros((nb, nb))
    sel = (k < nb) & (m < nb)
    np.add.at(Y, (m[sel], k[sel]), y0[sel])
    return Y


def upscatter_set():
    """The small kernel set described in the module docstring."""
    z = np.load(DECAY, allow_pickle=False)
    eb = z["e_bins"][:NB + 1]
    E = z["e_grid"][:NB]
    w = z["e_widths"][:NB]
    idx = {n: i for i, n in enumerate(NAMES)}
    nS = len(NAMES)
    mass = np.array([M_N, M_K, M_K, M_PI, M_MU, M_MU, 0, 0, 0, M_N])
    role = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2, 0])
    lam = np.zeros((nS, NB))
    dec = np.zeros((nS, NB))
    lam[idx["P"]] = 1.0 / 90.0
    lam[idx["K+"]] = 1.0 / 140.0                      # absorption, zero yield
    lam[idx["pi+"]] = np.where(E > 0.7, 1.0 / 120.0, 0.0)   # threshold
    for n, ct in CTAU.items():
        m_ = mass[idx[n]]
        dec[idx[n]] = m_ / (ct * np.sqrt(E * (E + 2 * m_)))
    # "n0": produced, decays with no decay channel (zero yield) and is absorbed;
    # decay length shortened (1 km x p/m) so that the decay matters
    dec[idx["n0"]] = M_N / (1.0e5 * np.sqrt(E * (E + 2 * M_N)))
    lam[idx["n0"]] = 1.0 / 100.0
    phi = np.zeros((nS, NB))
    phi[idx["P"]] = np.where((E >= 1.0) & (E <= 1000.0), 1e4 * E ** -2.7, 0.0)
    kinds, par, chi, Ys = [], [], [], []
    # P -> K+, K_S0, pi+: smooth inclusive yields below the parent's bin
    for c, mult in (("K+", 0.12), ("K_S0", 0.06), ("pi+", 1.0), ("n0", 0.3)):
        Y = np.zeros((NB, NB))
        for k in range(NB):
            x = E[:k] / E[k]
            dn = mult * (1 - x) ** 3 / np.maximum(x, 1e-3) * 0.23  # number per bin
            dn[:max(0, k - 30)] = 0.0
            Y[:k, k] = dn * w[k] / w[:k]
        kinds.append(K.KIND_INT)
        par.append(idx["P"])
        chi.append(idx[c])
        Ys.append(Y)
    for p, c in DECAYS:
        kinds.append(K.KIND_DEC)
        par.append(idx[p])
        chi.append(idx[c])
        Ys.append(_decay_yield(z, p, c, NB))
    d = {"e_grid": E, "e_bins": eb, "e_widths": w, "species": np.array(NAMES),
         "pdg": np.array([2212, 321, 310, 211, -13, -13, 14, -14, 12, 2112]),
         "mass": mass, "charge": np.array([1, 1, 0, 1, 1, 1, 0, 0, 0, 0]),
         "role": role, "active": np.ones((nS, NB), bool), "inv_lambda": lam,
         "inv_dec": dec, "dedx": np.zeros((nS, NB)), "phi_top": phi,
         "ch_kind": np.array(kinds), "ch_parent": np.array(par),
         "ch_child": np.array(chi), "Y": np.array(Ys)}
    ks = K.KernelSet(d)
    ks.validate()
    return ks


def expm_reference(ks, X):
    """phi at grammage X: expm of the full straight-ahead matrix (constant rho)."""
    from scipy.linalg import expm

    nS, nE = ks.nS, ks.nE
    M = np.zeros((nS * nE, nS * nE))
    for j in range(ks.ch_kind.size):
        p, c = ks.ch_parent[j], ks.ch_child[j]
        rate = (ks.inv_lambda[p] if ks.ch_kind[j] == K.KIND_INT
                else ks.inv_dec[p] / RHO)
        M[c * nE:(c + 1) * nE, p * nE:(p + 1) * nE] += ks.Y[j] * rate[None, :]
    for s in range(nS):
        att = ks.inv_lambda[s] + ks.inv_dec[s] / RHO
        M[s * nE:(s + 1) * nE, s * nE:(s + 1) * nE] -= np.diag(att)
    phi = expm(M * X) @ ks.phi_top.reshape(-1)
    return phi.reshape(nS, nE)


def upscatter_fraction(ks):
    """Largest fraction of a transported child's yield above the parent's bin."""
    w = ks.e_widths
    out = 0.0
    for j in range(ks.ch_kind.size):
        if ks.role[ks.ch_child[j]] == K.ROLE_SINK:
            continue
        n = ks.Y[j] * w[:, None] / w[None, :]
        tot = n.sum(axis=0)
        up = np.tril(n, -1).sum(axis=0)
        f = np.where(tot > 0, up / np.where(tot > 0, tot, 1.0), 0.0)
        out = max(out, float(f.max()))
    return out


def solve_set(ks, mu0s, n_up, spec=((np.inf, 0.05),)):
    atm = grid.ConstantAtmosphere(RHO, HTOP)
    lev = grid.make_levels(atm, spec=spec,
                           top_layer=tuple(np.geomspace(1.0, 3000.0, 21)))
    fam = grid.LineFamily(np.array(mu0s), np.zeros(0))
    r = rays.Rays(atm, lev, fam, dX_max=10.0)
    sv = S.Solver(r, ks, n_up=n_up)
    return sv, r


def ground(sv, r, out):
    import assemble

    nu = np.asarray(assemble.sink_intensity(sv, out["emis"]))[:, :, r.ground_node]
    Fg = np.asarray(out["F_ground"])
    res = {}
    for i, n in enumerate(sv.ks.species):
        if i in sv.Sk:
            res[str(n)] = nu[:, sv.Sk.index(i), :].T
        else:
            res[str(n)] = Fg[:, sv.T.index(i), :].T
    return res


def compare(ks, n_up, mu0s=(1.0, 0.25)):
    sv, r = solve_set(ks, mu0s, n_up)
    out = sv.solve()
    g = ground(sv, r, out)
    worst = {}
    for j, mu0 in enumerate(r.ground_mu0):
        X = r.node_Xc[r.ground_node[j]]
        ref = expm_reference(ks, X)
        for i, n in enumerate(NAMES):
            if n in ("P",):
                continue
            a, b = g[n][j], ref[i]
            ok = b > 1e-6 * b.max()
            rel = np.max(np.abs(a[ok] / b[ok] - 1)) if ok.any() else 0.0
            worst[(n, float(mu0))] = rel
    return worst, np.asarray(out["up_res"]), sv


@pytest.fixture(scope="module")
def ks():
    if not os.path.exists(DECAY):
        pytest.skip("decay kernels not available")
    return upscatter_set()


def test_set_has_large_upscatter(ks):
    assert upscatter_fraction(ks) > 0.5


def test_upscatter_against_expm(ks):
    """With the up-scatter passes, against expm: 1.5e-3 vertical, 5e-3 at cos Z 0.25.

    Measured (2026-11-19): K+ and K_S 3e-9, pi+ 5e-5, muons and neutrinos 1.0e-3
    (vertical) and 3.3e-3 (cos Z 0.25).  The latter is made in the top 100 m of the
    slab (thick kaons and intermediate pions rising from their zero boundary value)
    and falls at first order with the level spacing (1.8e-3 at 25 m, 0.9e-3 at
    12.5 m): a boundary layer of the constant-density geometry, not of the
    up-scatter.  Pass sizes 0.62, 0.024, 0, 0: the chain has depth two.
    """
    worst, up_res, sv = compare(ks, n_up=4)
    assert sv.K_up > 0
    vert = max(v for (n, mu0), v in worst.items() if mu0 == 1.0)
    incl = max(v for (n, mu0), v in worst.items() if mu0 == 0.25)
    assert vert < 1.5e-3 and incl < 5e-3, worst
    assert up_res[1] < 0.05 and up_res[-1] < 1e-12, up_res


def test_without_passes_is_biased(ks):
    """Without the passes the low-energy fluxes are visibly wrong (sanity)."""
    worst, _, _ = compare(ks, n_up=0)
    assert max(worst.values()) > 0.05, worst


def test_jvp_through_passes(ks):
    """jvp w.r.t. an up-scattering decay yield (K+ -> pi+) equals central FD."""
    import jax
    import jax.numpy as jnp

    sv, r = solve_set(ks, (1.0,), n_up=4)
    prm = {k: jnp.asarray(v) for k, v in ks.params().items()}
    j = ks.channel("dec", "K+", "pi+")

    def f(Yb):
        p = dict(prm)
        p["Y"] = prm["Y"].at[j].set(Yb)
        out = sv.solve(p)
        return jnp.concatenate([out["emis"].ravel(), out["F_ground"].ravel()])

    Y0 = prm["Y"][j]
    val, tan = jax.jvp(f, (Y0,), (Y0,))
    eps = 1e-4
    fd = (f(Y0 * (1 + eps)) - f(Y0 * (1 - eps))) / (2 * eps)
    tan, fd, val = np.asarray(tan), np.asarray(fd), np.asarray(val)
    sig = np.abs(fd) > 1e-8 * np.abs(val).max()
    rel = np.abs(tan[sig] - fd[sig]) / np.abs(fd[sig])
    assert sig.sum() > 100 and rel.max() < 1e-6, rel.max()
