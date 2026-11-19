"""Kernel sets: the only physics input of the ``cascade3d`` solver (WP 1.2).

A :class:`KernelSet` is a dictionary of plain numpy arrays (npz-serialisable, no
pickles).  The layout and its physics meaning are fixed in ``DESIGN.md`` section 3;
in short, for every production channel (parent p -> child c, interaction or decay)
the yield ``Y[m, k]`` in MCEq's convention (child bin m, parent bin k,
``Y = n_mk w_k / w_m``, rate and ``-delta`` not included) and, for channels and parent
bins with an angular law, the delta-M form: forward fraction ``g[m, k]`` and reduced
Legendre moments ``a[l, m, k]`` (``a_0 = 1``) of the rest, so that the angular kernel
is ``g delta + (1 - g) sum_l (2l + 1)/(4 pi) a_l P_l(cos theta)``.

Builders here
-------------
:func:`from_mceq`        straight-ahead set from ``kernels_mceq`` (g = 1 everywhere).
:func:`synthetic_angular`  an analytic angular law attached to a set (tests, G1.2).
:func:`toy_cascade`      tiny sets for the independent 3D toy test.
"""

import json

import numpy as np

ROLE_HADRON, ROLE_LOSS, ROLE_SINK = 0, 1, 2
KIND_INT, KIND_DEC = 0, 1

_REQUIRED = ("e_grid", "e_bins", "e_widths", "species", "pdg", "mass", "charge",
             "role", "active", "inv_lambda", "inv_dec", "dedx", "phi_top",
             "ch_kind", "ch_parent", "ch_child", "Y", "ang_channel", "n_ang", "g", "a")


class KernelSet:
    """Container of the kernel arrays (see module and ``DESIGN.md`` section 3).

    Parameters
    ----------
    d : dict
        Arrays named as in ``DESIGN.md``; ``meta`` (dict) is optional.
    """

    def __init__(self, d):
        self.d = dict(d)
        self.d.setdefault("meta", {})
        if "ang_channel" not in self.d:
            nE = self.d["e_grid"].size
            self.d["ang_channel"] = np.zeros(0, dtype=np.int64)
            self.d["n_ang"] = np.array(0)
            self.d["g"] = np.zeros((0, 0, 0))
            self.d["a"] = np.zeros((0, 1, 0, 0))
            del nE

    def __getattr__(self, name):
        try:
            return self.__dict__["d"][name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    @property
    def nE(self):
        return int(self.d["e_grid"].size)

    @property
    def nS(self):
        return int(self.d["species"].size)

    @property
    def L(self):
        return int(self.d["a"].shape[1]) - 1

    @property
    def nA(self):
        return int(self.d["n_ang"])

    def index(self, name):
        """Species index of ``name``."""
        idx = np.flatnonzero(self.d["species"] == name)
        if idx.size != 1:
            raise KeyError(name)
        return int(idx[0])

    def channel(self, kind, parent, child):
        """Channel index for (``kind`` 0/1 or "int"/"dec", parent, child names)."""
        if isinstance(kind, str):
            kind = {"int": KIND_INT, "dec": KIND_DEC}[kind]
        p, c = self.index(parent), self.index(child)
        sel = np.flatnonzero((self.d["ch_kind"] == kind) & (self.d["ch_parent"] == p)
                             & (self.d["ch_child"] == c))
        if sel.size != 1:
            raise KeyError((kind, parent, child))
        return int(sel[0])

    def params(self):
        """The differentiable arrays (dict of numpy arrays) passed to the solver."""
        keys = ("Y", "inv_lambda", "inv_dec", "phi_top", "g", "a")
        return {k: np.asarray(self.d[k], float) for k in keys}

    def validate(self):
        """Check shapes and ranges; raises ``ValueError``."""
        d = self.d
        for k in _REQUIRED:
            if k not in d:
                raise ValueError(f"missing {k}")
        nE, nS = self.nE, self.nS
        nCh = d["ch_kind"].size
        shapes = {"e_bins": (nE + 1,), "e_widths": (nE,), "pdg": (nS,),
                  "mass": (nS,), "charge": (nS,), "role": (nS,),
                  "active": (nS, nE), "inv_lambda": (nS, nE), "inv_dec": (nS, nE),
                  "dedx": (nS, nE), "phi_top": (nS, nE), "ch_parent": (nCh,),
                  "ch_child": (nCh,), "Y": (nCh, nE, nE)}
        for k, s in shapes.items():
            if tuple(np.shape(d[k])) != s:
                raise ValueError(f"{k} has shape {np.shape(d[k])}, expected {s}")
        lne = np.log(d["e_grid"])
        if not np.allclose(np.diff(lne), lne[1] - lne[0], rtol=1e-9, atol=0):
            raise ValueError("e_grid must be uniform in ln E")
        if np.any(d["dedx"] < 0):
            raise ValueError("dedx is the loss rate -dE/dX and must be >= 0")
        if np.any(d["dedx"][d["role"] != ROLE_LOSS] != 0):
            raise ValueError("dedx > 0 only for loss species")
        if np.any(d["role"][d["ch_parent"]] == ROLE_SINK):
            raise ValueError("sinks cannot be parents")
        nA, nAng = self.nA, d["ang_channel"].size
        if nAng:
            if d["g"].shape != (nAng, nA, nA):
                raise ValueError("g must be (nAng, nA, nA)")
            if d["a"].shape[0] != nAng or d["a"].shape[2:] != (nA, nA):
                raise ValueError("a must be (nAng, L+1, nA, nA)")
            if np.any((d["g"] < 0) | (d["g"] > 1)):
                raise ValueError("g outside [0, 1]")
            if not np.allclose(d["a"][:, 0], 1.0):
                raise ValueError("a_0 must be 1")
            if np.any(np.abs(d["a"]) > 1 + 1e-9):
                raise ValueError("|a_l| > 1")
        return True

    def save(self, path):
        """Write to ``path`` (npz, compressed, atomic)."""
        import os

        out = {k: np.asarray(v) for k, v in self.d.items() if k != "meta"}
        out["meta_json"] = np.array(json.dumps(self.d.get("meta", {})))
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, **out)
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        d["meta"] = json.loads(str(d.pop("meta_json", "{}")))
        return cls(d)


# ---------------------------------------------------------------------- MCEq
def _yield_from_block(block, rate, same):
    """Invert ``block = rate[k] (Y - delta)``; zero where the rate is zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.where(rate[None, :] > 0, block / rate[None, :], 0.0)
    if same:
        idx = np.arange(rate.size)
        y[idx, idx] = np.where(rate > 0, y[idx, idx] + 1.0, 0.0)
    return y


def from_mceq(population, loss=True, d=None, keep_upscatter=False):
    """Straight-ahead kernel set from ``kernels_mceq`` (all g = 1).

    Parameters
    ----------
    population : {"PHYS", "REF"}
    loss : bool
        Continuous loss of the muons on (MCEq ``dedx``) or off.
    d : dict, optional
        An already loaded ``kernels_mceq.load(population)``.
    keep_upscatter : bool
        Keep entries with child bin above parent bin for transported children (the
        solver ignores them there; kept only for diagnostics).  Up-scatter into
        sinks (neutrinos) is always kept: a neutrino can carry more than the
        parent's kinetic energy, and the solver accumulates sink emissivity until the
        end of the march, so it uses these entries.

    Notes
    -----
    Kept: the 32 physical species minus gamma (no child of interest); neutrinos are
    sinks, the six helicity muons are loss species (``dedx = 0`` if ``loss`` is
    False), everything else hadron.  ``active[s, k] = k >= mix_idx[s]``.
    """
    import kernels_mceq as km

    if d is None:
        d = km.load(population)
    sp = d["species"]
    keep = (~d["is_tracking"]) & (sp != "gamma")
    old = np.flatnonzero(keep)
    new_of = -np.ones(sp.size, dtype=np.int64)
    new_of[old] = np.arange(old.size)
    nE = d["e_grid"].size
    pdg = d["pdg"][old]
    role = np.where(np.isin(np.abs(pdg), (12, 14, 16)), ROLE_SINK,
                    np.where(np.abs(pdg) == 13, ROLE_LOSS, ROLE_HADRON))
    lam = d["inv_lambda_int"][old].copy()
    dec = d["inv_dec_len"][old].copy()
    dedx = np.zeros((old.size, nE))
    if loss:
        for j, s in enumerate(d["loss_species"]):
            if new_of[s] >= 0:
                dedx[new_of[s]] = -d["dedx"][j]
    mix = d["mix_idx"][old]
    active = np.arange(nE)[None, :] >= mix[:, None]
    phi_top = d["phi0"].reshape(sp.size, nE)[old].copy()
    kinds, par, chi, Ys = [], [], [], []
    up_max = 0.0
    up_where = None
    for kind, rate_all in ((KIND_INT, d["inv_lambda_int"]),
                           (KIND_DEC, d["inv_dec_len"])):
        name = "int" if kind == KIND_INT else "dec"
        for p, c, blk in zip(d[f"{name}_parent"], d[f"{name}_child"],
                             d[f"{name}_blocks"]):
            if new_of[p] < 0 or new_of[c] < 0:
                continue
            y = _yield_from_block(blk, rate_all[p], p == c)
            sink = abs(int(d["pdg"][c])) in (12, 14, 16)
            if not sink:
                # number fraction of children above the parent's bin
                n = y * d["e_widths"][:, None] / d["e_widths"][None, :]
                tot = n.sum(axis=0)
                up = np.tril(n, -1).sum(axis=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    f = np.where(tot > 0, up / tot, 0.0)
                if f.max() > up_max:
                    up_max = float(f.max())
                    up_where = (str(sp[p]), str(sp[c]), kind)
                if not keep_upscatter:
                    y = np.triu(y)
            if not np.any(y != 0):
                continue
            kinds.append(kind)
            par.append(new_of[p])
            chi.append(new_of[c])
            Ys.append(y)
    Y = np.array(Ys)
    ch_child = np.array(chi, dtype=np.int64)
    inactive_rows = 0.0
    for j, c in enumerate(ch_child):
        bad = ~active[c]
        if bad.any():
            inactive_rows = max(inactive_rows, float(np.abs(Y[j][bad]).max()))
            Y[j][bad] = 0.0
    meta = dict(d["meta"])
    meta.update({"builder": "kernelset.from_mceq", "population": population,
                 "loss": bool(loss),
                 "upscatter_dropped_max_number_fraction": up_max,
                 "upscatter_dropped_worst_channel": up_where,
                 "upscatter_kept_for_sinks": True,
                 "max_yield_in_inactive_child_bins_zeroed": inactive_rows})
    ks = KernelSet({
        "e_grid": d["e_grid"], "e_bins": d["e_bins"], "e_widths": d["e_widths"],
        "species": sp[old], "pdg": pdg, "mass": d["mass"][old],
        "charge": d["charge"][old], "role": role, "active": active,
        "inv_lambda": lam, "inv_dec": dec, "dedx": dedx, "phi_top": phi_top,
        "ch_kind": np.array(kinds, dtype=np.int64),
        "ch_parent": np.array(par, dtype=np.int64), "ch_child": ch_child, "Y": Y,
        "meta": meta,
    })
    ks.validate()
    return ks


# ------------------------------------------------------------ angular laws
def legendre_table(x, L):
    """``P_l(x)`` for l = 0..L, shape ``(L + 1,) + x.shape`` (recurrence)."""
    x = np.asarray(x, float)
    P = np.zeros((L + 1,) + x.shape)
    P[0] = 1.0
    if L >= 1:
        P[1] = x
    for ll in range(1, L):
        P[ll + 1] = ((2 * ll + 1) * x * P[ll] - ll * P[ll - 1]) / (ll + 1)
    return P


_GL_CACHE = {}


def _leggauss(n):
    if n not in _GL_CACHE:
        _GL_CACHE[n] = np.polynomial.legendre.leggauss(n)
    return _GL_CACHE[n]


def moments_of_law(w_of_cos, L, n=2000, theta_max=np.pi):
    """Legendre moments ``<P_l>`` (l = 0..L) of an angular density.

    ``w_of_cos(mu)`` is the (unnormalised) density per unit cos theta (equivalently
    per solid angle).  Gauss-Legendre with ``n`` nodes in theta on
    ``[0, theta_max]`` (the density must be negligible beyond ``theta_max``).
    """
    x, w = _leggauss(n)
    th = 0.5 * theta_max * (x + 1.0)
    mu = np.cos(th)
    f = w_of_cos(mu) * np.sin(th) * w
    P = legendre_table(mu, L)
    return (P @ f) / np.sum(f)


def delta_m(b, L):
    """Delta-M split of raw moments ``b`` (l = 0..L+1): ``(g, a)``, ``a_0 = 1``."""
    g = float(np.clip(b[L + 1], 0.0, 1.0 - 1e-12))
    a = (np.asarray(b[:L + 1]) - g) / (1.0 - g)
    a[0] = 1.0
    return g, np.clip(a, -1.0, 1.0)


def gaussian_theta_law(theta0):
    """Density in cos theta of ``exp(-theta^2 / (2 theta0^2))`` per solid angle."""
    def w(mu):
        th = np.arccos(np.clip(mu, -1.0, 1.0))
        return np.exp(-0.5 * (th / theta0) ** 2)
    return w


def synthetic_angular(ks, L=32, e_cut=3.0e3, pt0=0.35, pt0_dec=0.1, channels="int",
                      theta_max=None):
    """Attach a pT-like Gaussian angular law to a kernel set (tests, G1.2).

    For channel (p -> c) and child bin m the secondaries' angle to the parent is
    Gaussian in theta (per solid angle) with ``theta0 = pt / p_child`` (``pt0`` for
    interactions, ``pt0_dec`` for decays), independent of the parent bin; parent bins
    with ``E_k < e_cut`` [GeV] get the law (``n_ang`` = number of such bins),
    above it everything stays straight ahead.  Moments by quadrature, delta-M split
    at order L+1.

    Parameters
    ----------
    channels : {"int", "dec", "all"}
        Which channels receive the law (hadron -> any child).
    """
    d = dict(ks.d)
    E = d["e_grid"]
    nA = int(np.searchsorted(E, e_cut))
    sel = []
    for j, (kind, p) in enumerate(zip(d["ch_kind"], d["ch_parent"])):
        if d["role"][p] != ROLE_HADRON:
            continue
        if channels == "int" and kind != KIND_INT:
            continue
        if channels == "dec" and kind != KIND_DEC:
            continue
        sel.append(j)
    sel = np.array(sel, dtype=np.int64)
    # law per (child species, child bin, kind)
    cache = {}
    g = np.ones((sel.size, nA, nA))
    a = np.ones((sel.size, L + 1, nA, nA))
    for i, j in enumerate(sel):
        c, kind = int(d["ch_child"][j]), int(d["ch_kind"][j])
        key = (c, kind)
        if key not in cache:
            m_c = float(d["mass"][c])
            pt = pt0 if kind == KIND_INT else pt0_dec
            gm = np.ones(nA)
            am = np.ones((L + 1, nA))
            for m in range(nA):
                p = np.sqrt(E[m] * (E[m] + 2.0 * m_c)) if m_c > 0 else E[m]
                th0 = pt / p
                if theta_max is not None:
                    th0 = min(th0, theta_max)
                if th0 < 1e-3:
                    continue  # effectively straight ahead
                b = moments_of_law(gaussian_theta_law(th0), L + 1,
                                   theta_max=min(np.pi, 12.0 * th0))
                gm[m], am[:, m] = delta_m(b, L)
            cache[key] = (gm, am)
        gm, am = cache[key]
        g[i] = gm[:, None] * np.ones(nA)[None, :]
        a[i] = am[:, :, None] * np.ones(nA)[None, None, :]
    d["ang_channel"] = sel
    d["n_ang"] = np.array(nA)
    d["g"] = g
    d["a"] = a
    meta = dict(d.get("meta", {}))
    meta.update({"angular": "synthetic gaussian theta", "pt0": pt0, "pt0_dec": pt0_dec,
                 "e_cut": e_cut, "L": L, "channels": channels})
    d["meta"] = meta
    out = KernelSet(d)
    out.validate()
    return out


# ------------------------------------------------------------------- toys
def toy_cascade(lam_p=90.0, dec_c=1.0 / 4.0e5, phi_top=1.0, g=0.25, kappa=2.0, L=32,
                law="vmf", dec_g=1.0, dec_kappa=None, dedx_c=0.0):
    """Two-generation toy: P (bin 1) -> C (bin 0) -> nu (bin 0).

    P: hadron, interaction length ``1/lam_p``... given as ``lam_p`` [g/cm^2], no decay,
    isotropic top intensity ``phi_top``.  C: no interaction, decay rate ``dec_c``
    [1/cm].  P -> C: one C per interaction with angular law ``g delta + (1 - g)
    vMF(kappa)`` (von Mises-Fisher, density ``exp(kappa cos theta)`` per solid angle,
    moments ``i_l(kappa)/i_0(kappa)``), or isotropic if ``law == "iso"``.  C -> nu:
    one neutrino per decay, straight ahead if ``dec_g == 1`` else ``dec_g delta +
    (1 - dec_g) vMF(dec_kappa)``.

    The energy grid has two bins of equal width so that the yields are plain
    multiplicities.
    """
    e_bins = np.array([1.0, np.e, np.e ** 2])
    e_grid = np.sqrt(e_bins[1:] * e_bins[:-1])
    # equal widths are impossible on a log grid; the solver needs uniform ln E, so
    # the yields carry the width ratio: Y = n w_k / w_m
    w = np.diff(e_bins)
    names = np.array(["P", "C", "nu"])
    nE = 2
    role = np.array([ROLE_HADRON, ROLE_HADRON if dedx_c == 0 else ROLE_LOSS,
                     ROLE_SINK])
    active = np.array([[False, True], [True, False], [True, False]])
    inv_lambda = np.array([[0.0, 1.0 / lam_p], [0.0, 0.0], [0.0, 0.0]])
    inv_dec = np.array([[0.0, 0.0], [dec_c, 0.0], [0.0, 0.0]])
    dedx = np.zeros((3, nE))
    dedx[1] = dedx_c
    phi = np.array([[0.0, phi_top], [0.0, 0.0], [0.0, 0.0]])
    Y = np.zeros((2, nE, nE))
    Y[0, 0, 1] = 1.0 * w[1] / w[0]   # P(bin 1) -> C(bin 0)
    Y[1, 0, 0] = 1.0                 # C(bin 0) -> nu(bin 0)

    def vmf_moments(kap, LL):
        if kap == 0:
            b = np.zeros(LL + 1)
            b[0] = 1.0
            return b
        return moments_of_law(lambda mu: np.exp(kap * (mu - 1.0)), LL)

    ang, gs, As = [], [], []
    for j, (gg, kap, lw) in enumerate(((g, kappa, law), (dec_g, dec_kappa, "vmf"))):
        if gg >= 1.0:
            continue
        b = np.zeros(L + 1)
        if lw == "iso":
            b[0] = 1.0
        else:
            b = vmf_moments(kap, L)
        A = np.zeros((L + 1, nE, nE))
        A[:, :, :] = b[:, None, None]
        G = np.full((nE, nE), gg)
        ang.append(j)
        gs.append(G)
        As.append(A)
    d = {
        "e_grid": e_grid, "e_bins": e_bins, "e_widths": w, "species": names,
        "pdg": np.array([2212, 211, 14]), "mass": np.array([0.938, 0.1396, 0.0]),
        "charge": np.array([1, 1, 0]), "role": role, "active": active,
        "inv_lambda": inv_lambda, "inv_dec": inv_dec, "dedx": dedx, "phi_top": phi,
        "ch_kind": np.array([KIND_INT, KIND_DEC]), "ch_parent": np.array([0, 1]),
        "ch_child": np.array([1, 2]), "Y": Y,
        "ang_channel": np.array(ang, dtype=np.int64), "n_ang": np.array(nE),
        "g": np.array(gs).reshape(len(ang), nE, nE),
        "a": np.array(As).reshape(len(ang), L + 1, nE, nE),
        "meta": {"builder": "kernelset.toy_cascade", "lam_p": lam_p, "dec_c": dec_c,
                 "g": g, "kappa": kappa, "law": law, "dec_g": dec_g,
                 "dec_kappa": dec_kappa},
    }
    ks = KernelSet(d)
    ks.validate()
    return ks
