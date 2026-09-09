"""Where does the +5% sub-GeV excess of the delivered production factor come from?

The delivered joint factor at ``G == 1``,

    F_s(E, cosZ) = J_s / p_axis        (Eq. 8b with G == 1, i.e. Eq. 8)

has a down-going solid-angle average of 1.065 / 1.046 / 1.024 / 1.008 at
0.2 / 0.3 / 0.5 / 1 GeV.  A full 3D calculation is constrained by Liouville, so
the question is which part of that is physics (curvature) and which is an
artefact of the construction.  This module decomposes it.

WHAT THE EXACT EXPRESSION IS
----------------------------
Let the primary flux be isotropic and let ``p(X, E) = dPhi_nu/dX`` be MCEq's
depth-resolved production *per unit slant depth and per steradian of primary
direction* (that is what a 1D MCEq run returns).  The number of neutrinos made
per unit time in a volume ``dV`` at ``P`` by primaries within ``dOmega_p`` is
then ``rho(P) p(X_slant(P, Omega_p), E) dV dOmega_p`` -- the volume element does
not depend on the primary direction, so no Jacobian appears here.  Those
neutrinos are emitted with the decay-chain kernel ``K(alpha)``, ``alpha`` the
angle to the primary, normalised over the *neutrino* sphere,
``Int dOmega_nu K = 1``.  Because ``K`` is azimuthally symmetric about the
primary axis, normalising over ``dOmega_p`` at fixed ``Omega_nu`` gives the same
constant, so the *conditioning* (primary direction given the neutrino direction
vs the reverse) does NOT introduce a reciprocity factor.  The emissivity per
unit volume and per unit neutrino solid angle is therefore

    eps(P, Omega_nu, E) = rho(P) Int dOmega_p K(alpha) p(X_slant(P, Omega_p), E)

and, since neutrinos free-stream, the specific intensity at the detector is the
line integral of the emissivity along the arrival ray (the ``1/r^2`` of a point
source is exactly cancelled by the solid-angle growth of the beam):

    Phi_nu(Omega_nu, E) = Int dl rho(h) Int dOmega_p K(alpha)
                              p(X_slant(l, psi_p), E).                    (EXACT)

That **is** Eq. (8)'s numerator, and its denominator is the same integral with
``K -> delta``.  So Eq. (8) is the exact straight-line 3D transport of an
isotropic primary flux through a curved atmosphere, not an approximation of it,
and no reciprocity/Jacobian factor is missing.

WHAT IS THEN CONSTRAINED
------------------------
Not ``<F>_Omega``.  Evaluate the exact expression in a **flat** atmosphere,
where ``X_slant = X_v(h)/cos psi`` and translation invariance makes the answer
known in closed form.  Using ``Int_0^{X_ground} dX_v p(X_v/u_p) = u_p P`` with
``P = Int p dX`` the fully developed yield,

    D(u_o) = (1/u_o) Int dX_v p(X_v/u_o)                       = P
    N(u_o) = (1/u_o) Int dOmega_p K <Int dX_v p(X_v/u_p)>       = (P/u_o) <u_p>_K

and, for a cone that does not reach the local horizon, ``<u_p>_K = <cos alpha> u_o``,
hence

    F_flat = <cos alpha>_K   (uniform in zenith, < 1).                 (FLAT)

The deficit ``1 - <cos alpha>`` is not lost flux: it is the flux a flat
atmosphere emits into the *upper* hemisphere near the horizon, which a spherical
Earth returns.  So the exact kernel does **not** predict ``<F>_Omega = 1``; the
number that is conserved is the flux through the ground,
``Int_down dOmega cos psi Phi``, and even that only up to the genuine
curvature term.  Both averages are reported below.

WHAT THIS SCRIPT MEASURES
-------------------------
``--decompose``  <F>_Omega and <F>_cos, split by parent channel (direct pi, K,
                 muon decay), by cone scale (0.5/0.75/1/1.25), by quadrature
                 (n_alpha, n_beta, n_ray), by the slant-depth table resolution
                 (the limb), and by the depth at which ``p(X)`` is truncated.
``--flat``       the (FLAT) gate above: the same machinery in a flat atmosphere
                 must return ``<cos alpha>_K`` at the vertical.
``--hv``         horizon/vertical of the delivered engine for the variants.

RESULTS (2026-09-09, delivered v2 moments, channel-resolved cone)
----------------------------------------------------------------
nu_mu, at 0.2 / 0.3 / 0.5 / 1 GeV::

    <F>_Omega   curved   1.0652  1.0458  1.0242  1.0084   <- the number quoted
    <F>_cos     curved   0.9125  0.9453  0.9717  0.9891
    <F>_Omega   FLAT     1.2059  1.1545  1.0950  1.0421   <- exact, conserving
    <F>_cos     FLAT     0.9276  0.9565  0.9785  0.9920
    F(vertical) curved   0.8563  0.9138  0.9578  0.9847
    <cos alpha> (FLAT)   0.8557  0.9135  0.9576  0.9846

So the exact kernel, evaluated in a flat atmosphere where it provably conserves
flux, ALREADY gives +15% in the plain solid-angle measure at 0.3 GeV -- more than
the +4.6% the curved calculation gives -- because ``F ~ 1/cos psi_o`` at the
horizon and that measure weights the horizon like the vertical (the flat number
is grid-dependent; it grows without bound as the horizon bin is refined).  In the
measure that counts neutrinos through the ground the delivered factor is a 5.5%
*deficit*, of which 4.4% is already in the flat limit (large-alpha emission
leaving the atmosphere upward) and 1.1% is genuine sphericity (near-horizontal
neutrinos skimming over the limb).  Nothing is manufactured.

Everything else is small: quadrature <= 0.02%, ``p(X)`` depth truncation <= 0.4%,
limb resolution -0.44% on F(horizon), cone SHAPE at fixed second moment (von
Mises-Fisher vs Gaussian) -0.27%, and using ``p(X,E)`` solved at 85 deg instead
of the vertical +1.7% on F(horizon).  The excess is set by ONE number, the cone
second moment: scaling every width by 0.5/0.75/1/1.25 gives <F>_Omega =
1.014/1.030/1.046/1.059 at 0.3 GeV, and a factor H/V of 1.54/1.86/2.20 at
0.75/1/1.25 -- the excess and the zenith shape move together and cannot be
separated.

Run from ``tools/mceq3d`` with a warm ``.cache3d``.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import joint_cone as jc
import offaxis_mc as ox

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".cache3d")
PRODCACHE = os.path.join(CACHE, f"jointprod_chan_{ox.TAG}.npz")
WIDTHCACHE = os.path.join(CACHE, "conewidths_1f0c6cd501e88f41.npz")

#: engine species names, and the (dir, K, mu) channel order of
#: ``offaxis_mc.production_profile`` / ``joint_cone.load_production``.
SPECIES = ("numu", "antinumu", "nue", "antinue")
CHANNELS = ("dir", "k", "mu")
E_REPORT = (0.2, 0.3, 0.5, 1.0, 3.0)


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def load(prod_cache=PRODCACHE, width_cache=WIDTHCACHE):
    """``(prod, widths)`` -- the delivered production profile and cone widths."""
    prod = jc.load_production(cache=prod_cache, species=ox.CHANNEL_SPECIES)
    if os.path.exists(width_cache):
        widths = {k: v for k, v in np.load(width_cache).items()}
    else:  # 4M-event decay MC per flavour
        widths = ox.channel_cone_widths(prod["ep_grid"], bending=False)
    return prod, widths


def channel_widths(widths, species, sigma_scale=1.0):
    """``{channel: sigma_deg[nE]}`` for one species, scaled by ``sigma_scale``."""
    mu = widths["mu_nue" if "nue" in species else "mu_numu"]
    return {"dir": np.asarray(widths["pi"], float) * sigma_scale,
            "k": np.asarray(widths["k"], float) * sigma_scale,
            "mu": np.asarray(mu, float) * sigma_scale}


# ---------------------------------------------------------------------------
# geometry variants
# ---------------------------------------------------------------------------
def flat_geom(prod, n_psi=721):
    """A FLAT-atmosphere slant-depth table on the same altitude grid.

    ``X_slant(h, psi) = X_v(h)/cos psi`` for ``psi < 90 deg`` and ``inf`` above,
    with ``X_v`` read off the delivered curved table at ``psi = 0`` (identical by
    construction there).  Used by the (FLAT) gate.
    """
    h_grid, _, table = prod["geom"]
    x_v = np.asarray(table)[:, 0]
    psi = np.linspace(0.0, np.deg2rad(179.0), n_psi)
    c = np.cos(psi)
    out = np.where(c[None, :] > 1e-9, x_v[:, None] / np.where(c > 1e-9, c, 1.0),
                   np.inf)
    return h_grid, psi, np.minimum(out, 1e30)


def flat_ray(cos_theta, n_ray=260, h_max_cm=80e5):
    """``(h_ray, ell, psi_o, w)`` for a straight ray in a flat atmosphere."""
    h = np.linspace(0.0, h_max_cm, n_ray)
    ell = h / max(float(cos_theta), 1e-9)
    psi_o = np.full(n_ray, np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    return h, ell, psi_o, ox._RHO(h) * np.gradient(ell)


def fine_geom(prod, n_h=180, n_psi=901, n_step=4000, limb_deg=8.0):
    """Rebuild ``slant_depth_table`` with a limb-refined ``psi`` grid.

    The delivered table is ``linspace(0, 179 deg, 140)`` -- 1.29 deg steps, over
    which ``X_slant`` changes by orders of magnitude at the limb, and blocked
    (``inf``) nodes are bilinearly mixed with their unblocked neighbours.  This
    variant puts half the ``psi`` nodes inside ``+-limb_deg`` of the local
    horizon and doubles the altitude and path-integration resolution.
    """
    rho = ox._RHO
    h_grid = np.concatenate([np.linspace(0.0, 40e5, n_h - 40),
                             np.linspace(41e5, ox.H_TOP_CM, 40)])
    lo, hi = np.deg2rad(90.0 - limb_deg), np.deg2rad(90.0 + limb_deg)
    n_limb = n_psi // 2
    psi = np.unique(np.concatenate([
        np.linspace(0.0, lo, (n_psi - n_limb) // 2, endpoint=False),
        np.linspace(lo, hi, n_limb),
        np.linspace(hi, np.deg2rad(179.0), (n_psi - n_limb) // 2 + 1),
    ]))
    table = np.zeros((len(h_grid), len(psi)))
    r_top = ox.R_EARTH_CM + ox.H_TOP_CM
    for i, h in enumerate(h_grid):
        r0 = ox.R_EARTH_CM + h
        for j, ps in enumerate(psi):
            cp = np.cos(ps)
            disc = (r0 * cp) ** 2 - (r0**2 - r_top**2)
            s_top = -r0 * cp + np.sqrt(max(disc, 0.0))
            s = np.linspace(0.0, s_top, n_step)
            alt = np.sqrt(r0**2 + s**2 + 2 * r0 * s * cp) - ox.R_EARTH_CM
            if np.any(alt < -1.0):
                table[i, j] = np.inf
                continue
            table[i, j] = np.trapezoid(rho(alt), s)
    return h_grid, psi, np.minimum(table, 1e30)


def truncate_p(prod, x_max):
    """A copy of ``prod`` whose per-channel profiles are zeroed beyond ``x_max``.

    Probes the sensitivity of ``F`` to the fact that ``p(X, E)`` is tabulated
    only out to the *vertical* atmospheric depth (1034 g/cm2): every cone sample
    (and, near the horizon, the axis itself) that goes deeper is charged zero
    production.
    """
    out = dict(prod)
    m = (prod["x_grid"] <= x_max)[:, None]
    out["chan"] = {s: tuple(np.where(m, p, 0.0) for p in ch)
                   for s, ch in prod["chan"].items()}
    return out


# ---------------------------------------------------------------------------
# cone SHAPE at fixed second moment (the tail systematic)
# ---------------------------------------------------------------------------
def vmf_alpha_weights(sigma_deg, alpha_rad):
    """Von Mises-Fisher cone weights matched to the Gaussian's SECOND MOMENT.

    The delivered cone is ``W ~ sin(alpha) exp(-alpha^2 / 2 s1^2)`` -- a
    Gaussian in the space angle wrapped onto the sphere.  Only its second moment
    is measured (the generator ``<theta^2>``); the *shape* of the large-alpha
    tail is an assumption, and the tail is what sets both the horizon
    enhancement and the fraction emitted into the upper hemisphere.  The von
    Mises-Fisher distribution ``W ~ sin(alpha) exp(kappa cos alpha)`` is the
    natural Gaussian ON a sphere; ``kappa`` is solved per energy so that
    ``<alpha^2>`` equals the Gaussian's on the SAME quadrature grid.  Comparing
    the two is a direct measure of the tail systematic at fixed measured moment.
    """
    G = jc.gauss_alpha_weights(sigma_deg, alpha_rad)
    a2 = G @ (alpha_rad**2)
    a = alpha_rad[1:]
    W = np.zeros_like(G)
    kap = np.exp(np.linspace(np.log(1e-3), np.log(1e5), 900))
    def _a2(k):
        w = np.sin(a) * np.exp(k * (np.cos(a) - 1.0))
        return (w / w.sum()) @ (a**2)

    tab = np.array([_a2(k) for k in kap])  # decreasing in kappa
    for i in range(len(a2)):
        if G[i, 0] == 1.0:  # collimated short circuit
            W[i, 0] = 1.0
            continue
        k = float(np.interp(a2[i], tab[::-1], kap[::-1]))
        w = np.sin(a) * np.exp(k * (np.cos(a) - 1.0))
        W[i, 1:] = w / w.sum()
    return W


# ---------------------------------------------------------------------------
# the factor, channel-resolved
# ---------------------------------------------------------------------------
def terms(cos_theta, prod, widths, *, sigma_scale=1.0, n_alpha=44, n_beta=18,
          n_ray=260, geom=None, ray=None, cone_shape="gauss"):
    """``(N, D)`` -- per-``(species, channel)`` numerators and denominators of
    Eq. (8) at ``G == 1``, so that ``F_s = sum_c N / sum_c D``.

    This is ``joint_cone.delivered_joint_factor`` with ``G == 1``, opened up so
    that each parent channel can be read separately; the ``--decompose`` gate
    asserts the two agree.
    """
    ep = prod["ep_grid"]
    alpha = jc.alpha_grid(n_alpha)
    beta = np.linspace(0.0, 2 * np.pi, n_beta, endpoint=False)
    plist = [pc for s in SPECIES for pc in prod["chan"][s]]
    qs, qas = jc.cone_production_multi(
        cos_theta, alpha, beta, prod["x_grid"], ep, plist,
        geom if geom is not None else prod["geom"], n_ray=n_ray, ray=ray)
    N, D = {}, {}
    for i, s in enumerate(SPECIES):
        sg = channel_widths(widths, s, sigma_scale)
        for j, ch in enumerate(CHANNELS):
            W = (jc.gauss_alpha_weights(sg[ch], alpha) if cone_shape == "gauss"
                 else vmf_alpha_weights(sg[ch], alpha))
            q = qs[3 * i + j]
            N[(s, ch)] = np.einsum("ea,ae->e", W, q.mean(axis=1))
            D[(s, ch)] = qas[3 * i + j]
    return N, D


def factor(cos_theta, prod, widths, **kw):
    """``{species: F[nE]}`` at ``G == 1``."""
    N, D = terms(cos_theta, prod, widths, **kw)
    return {s: sum(N[(s, c)] for c in CHANNELS)
            / np.maximum(sum(D[(s, c)] for c in CHANNELS), 1e-300)
            for s in SPECIES}


def mean_cos_alpha(widths, ep, species, sigma_scale=1.0, n_alpha=44):
    """``{channel: <cos alpha>_W}`` -- the (FLAT) prediction, per channel."""
    alpha = jc.alpha_grid(n_alpha)
    sg = channel_widths(widths, species, sigma_scale)
    return {ch: jc.gauss_alpha_weights(sg[ch], alpha) @ np.cos(alpha)
            for ch in CHANNELS}


# ---------------------------------------------------------------------------
# solid-angle averages
# ---------------------------------------------------------------------------
def sky_average(prod, widths, czs=None, **kw):
    """``(F_of_cz, avg_omega, avg_cos)`` over the down-going hemisphere.

    ``avg_omega``  = Int F dOmega / Int dOmega          (the number the paper quotes)
    ``avg_cos``    = Int F cos psi dOmega / Int cos psi dOmega  -- the weight that
                     counts neutrinos *through the ground*, i.e. the one a
                     conservation argument actually constrains.
    """
    czs = np.round(np.arange(0.05, 1.0, 0.1), 2) if czs is None else np.asarray(czs)
    F = {s: [] for s in SPECIES}
    for c in czs:
        f = factor(float(c), prod, widths, **kw)
        for s in SPECIES:
            F[s].append(f[s])
    F = {s: np.array(v) for s, v in F.items()}  # (n_cz, nE)
    wc = czs / czs.sum()
    return (czs, F,
            {s: F[s].mean(axis=0) for s in SPECIES},
            {s: wc @ F[s] for s in SPECIES})


def channel_average(prod, widths, czs=None, **kw):
    """Per-channel ``<N/D>_Omega``: the factor each parent channel would carry on
    its own, plus its share of the denominator (its blend weight)."""
    czs = np.round(np.arange(0.05, 1.0, 0.1), 2) if czs is None else np.asarray(czs)
    accF, accW = {}, {}
    for c in czs:
        N, D = terms(float(c), prod, widths, **kw)
        for s in SPECIES:
            dtot = np.maximum(sum(D[(s, ch)] for ch in CHANNELS), 1e-300)
            for ch in CHANNELS:
                k = (s, ch)
                accF[k] = accF.get(k, 0.0) + N[k] / np.maximum(D[k], 1e-300) / len(czs)
                accW[k] = accW.get(k, 0.0) + D[k] / dtot / len(czs)
    return accF, accW


def at(ep, arr, e_val):
    return float(np.interp(np.log(e_val), np.log(ep), arr))


def _row(tag, ep, arr, es=E_REPORT):
    return f"{tag:<26}" + "".join(f"{at(ep, arr, e):>10.4f}" for e in es)


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------
def report_flat(prod, widths, n_alpha=44, n_beta=18, n_ray=260):
    """(FLAT) gate: in a flat atmosphere the exact kernel gives ``<cos alpha>``."""
    ep = prod["ep_grid"]
    fg = flat_geom(prod)
    print("=" * 84)
    print("FLAT-ATMOSPHERE GATE  --  exact kernel must give F = <cos alpha>_K")
    print("(vertical arrival; deviation = quadrature + p(X) truncation, not physics)")
    print("=" * 84)
    hdr = f"{'':<26}" + "".join(f"{e:>10.2f}" for e in E_REPORT)
    for s in ("numu", "nue"):
        N, D = terms(1.0, prod, widths, n_alpha=n_alpha, n_beta=n_beta,
                     n_ray=n_ray, geom=fg, ray=flat_ray(1.0, n_ray))
        mca = mean_cos_alpha(widths, ep, s, n_alpha=n_alpha)
        dtot = np.maximum(sum(D[(s, ch)] for ch in CHANNELS), 1e-300)
        pred = sum(D[(s, ch)] * mca[ch] for ch in CHANNELS) / dtot
        got = sum(N[(s, ch)] for ch in CHANNELS) / dtot
        print(f"\n[{s}]  E [GeV]" + hdr[16:])
        for ch in CHANNELS:
            print(_row(f"  {ch}: flat F", ep, N[(s, ch)]
                       / np.maximum(D[(s, ch)], 1e-300)))
            print(_row(f"  {ch}: <cos alpha>", ep, mca[ch]))
        print(_row("  blended flat F", ep, got))
        print(_row("  blended <cos alpha>", ep, pred))
        print(_row("  ratio (must be 1)", ep, got / pred))
    # and the CURVED factor at the same (vertical) direction, for scale
    print("\ncurved-atmosphere F at the vertical (same quadrature):")
    fc = factor(1.0, prod, widths, n_alpha=n_alpha, n_beta=n_beta, n_ray=n_ray)
    for s in ("numu", "nue"):
        print(_row(f"  {s}", ep, fc[s]))
    print("DIAG_FLAT_DONE")


def report_decompose(prod, widths, args):
    ep = prod["ep_grid"]
    hdr = f"{'':<26}" + "".join(f"{e:>10.2f}" for e in E_REPORT)

    print("=" * 84)
    print("A. SOLID-ANGLE MEASURE:  <F>_Omega  vs  <F>_cos (flux through the ground)")
    print("=" * 84)
    print("E [GeV]" + hdr[7:])
    czs, F, avo, avc = sky_average(prod, widths)
    for s in SPECIES:
        print(_row(f"  {s}: <F>_Omega", ep, avo[s]))
        print(_row(f"  {s}: <F>_cos", ep, avc[s]))

    print("\n" + "=" * 84)
    print("B. BY PARENT CHANNEL: <N_c/D_c>_Omega (the channel's own factor) and")
    print("   <D_c/D>_Omega (its blend weight).  numu and nue only.")
    print("=" * 84)
    print("E [GeV]" + hdr[7:])
    accF, accW = channel_average(prod, widths)
    for s in ("numu", "nue"):
        for ch in CHANNELS:
            print(_row(f"  {s} {ch}: <F_c>", ep, accF[(s, ch)]))
            print(_row(f"  {s} {ch}: weight", ep, accW[(s, ch)]))

    print("\n" + "=" * 84)
    print("C. BY CONE SCALE: <F>_Omega (numu / nue)")
    print("=" * 84)
    print("E [GeV]" + hdr[7:])
    for sc in (0.0, 0.5, 0.75, 1.0, 1.25):
        _, _, a_o, a_c = sky_average(prod, widths, sigma_scale=sc)
        for s in ("numu", "nue"):
            print(_row(f"  scale {sc:.2f} {s} Omega", ep, a_o[s]))
            print(_row(f"  scale {sc:.2f} {s} cos", ep, a_c[s]))

    print("\n" + "=" * 84)
    print("D. QUADRATURE: <F>_Omega (numu)")
    print("=" * 84)
    print("E [GeV]" + hdr[7:])
    for tag, kw in (("default 44/18/260", {}),
                    ("n_alpha 88", dict(n_alpha=88)),
                    ("n_alpha 176", dict(n_alpha=176)),
                    ("n_beta 36", dict(n_beta=36)),
                    ("n_beta 72", dict(n_beta=72)),
                    ("n_ray 520", dict(n_ray=520)),
                    ("all refined", dict(n_alpha=176, n_beta=72, n_ray=520))):
        _, _, a_o, a_c = sky_average(prod, widths, **kw)
        print(_row(f"  {tag} Omega", ep, a_o["numu"]))
        print(_row(f"  {tag} cos", ep, a_c["numu"]))

    print("\n" + "=" * 84)
    print("E. PRODUCTION-DEPTH TRUNCATION: p(X) is zeroed beyond x_max [g/cm2].")
    print("   The delivered table stops at the VERTICAL depth, 1034 g/cm2.")
    print("=" * 84)
    print("E [GeV]" + hdr[7:])
    for xmax in (1034.0, 800.0, 600.0, 400.0):
        pr = truncate_p(prod, xmax)
        _, _, a_o, a_c = sky_average(pr, widths)
        print(_row(f"  x_max {xmax:6.0f} Omega", ep, a_o["numu"]))
        print(_row(f"  x_max {xmax:6.0f} cos", ep, a_c["numu"]))

    if args.fine_geom:
        print("\n" + "=" * 84)
        print("F. SLANT-DEPTH TABLE RESOLUTION (the limb): <F>_Omega (numu / nue)")
        print("=" * 84)
        print("E [GeV]" + hdr[7:])
        fg = fine_geom(prod)
        _, _, a_o, a_c = sky_average(prod, widths, geom=fg)
        for s in ("numu", "nue"):
            print(_row(f"  limb-refined {s} Omega", ep, a_o[s]))
            print(_row(f"  limb-refined {s} cos", ep, a_c[s]))
    print("DIAG_DECOMPOSE_DONE")


def report_zenith(prod, widths, args):
    """F(cosZ) itself, so the shape change of each variant is visible."""
    ep = prod["ep_grid"]
    czs = np.round(np.arange(0.05, 1.0, 0.1), 2)
    print("=" * 84)
    print("G. F(cosZ) at G == 1, numu, and the horizon/vertical of the FACTOR")
    print("=" * 84)
    for E in (0.3, 0.5, 1.0):
        print(f"\nE = {E} GeV   cosZ:" + "".join(f"{c:>8.2f}" for c in czs))
        for tag, kw in (("default", {}),
                        ("scale 0.75", dict(sigma_scale=0.75)),
                        ("scale 1.25", dict(sigma_scale=1.25))):
            _, F, _, _ = sky_average(prod, widths, czs=czs, **kw)
            row = [at(ep, F["numu"][i], E) for i in range(len(czs))]
            print(f"{tag:<16}" + "".join(f"{v:>8.3f}" for v in row)
                  + f"   H/V={row[0] / row[-1]:.3f}")
    print("DIAG_ZENITH_DONE")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flat", action="store_true")
    ap.add_argument("--decompose", action="store_true")
    ap.add_argument("--zenith", action="store_true")
    ap.add_argument("--hv", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=14)
    ap.add_argument("--fine-geom", action="store_true",
                    help="also rebuild the slant-depth table with a limb-refined "
                         "psi grid (~2 min)")
    args = ap.parse_args(argv)
    if not (args.flat or args.decompose or args.zenith or args.hv):
        args.flat = args.decompose = True
    prod, widths = load()
    if args.flat:
        report_flat(prod, widths)
    if args.decompose:
        report_decompose(prod, widths, args)
    if args.zenith:
        report_zenith(prod, widths, args)
    if args.hv:
        report_hv(prod, widths, args)


# ---------------------------------------------------------------------------
# H/V of the delivered engine, for each variant
# ---------------------------------------------------------------------------
LAT, LON = 36.43, 137.31
HV_E = (0.3, 0.5, 1.0)
HV_CZ = (0.05, 0.95)
HV_AZ = tuple(np.arange(8) * 45.0)


def _log_at(y, x, X):
    return float(np.exp(np.interp(np.log(X), np.log(x),
                                  np.log(np.maximum(y, 1e-300)))))


def engine_pieces(n_jobs=16, cache=CACHE):
    """Everything ``MCEq3DFlux.solve`` builds before the joint cone integral.

    Mirrors mceq3d_flux.py:2270-2345 for the delivered defaults
    (``cone_cutoff=True``, ``cutoff_anchor='prod_point'``, ``muon_bending=True``,
    ``zenith_dependent_geomag``), so the variants below differ from the delivered
    engine ONLY inside :func:`joint_cone.delivered_joint_factor`.
    """
    from datetime import datetime

    from mceq3d_flux import MCEq3DFlux, RC_MAX_GV, SPECIES as ENG_SP
    from muon_bending import local_field_enu

    date = datetime(2020, 1, 1)
    eng = MCEq3DFlux(base_model="hybrid", primary=("GlobalSplineFitBeta", None),
                     daemonflux_location="kamioka")
    cz = np.array(HV_CZ)
    base = eng.base(cz)
    rc_grid = np.linspace(0.1, RC_MAX_GV, 40)
    G_by_z = [eng.geomag_response(rc_grid, cz_ref=max(abs(c), 1e-3),
                                  cache_dir=cache)[0] for c in cz]
    fine = eng.finemap_rc(LAT, LON, date, cache_dir=cache, n_jobs=n_jobs)
    fam = eng.prod_family_rc(LAT, LON, date, cache_dir=cache, n_jobs=n_jobs)
    return dict(eng=eng, cz=cz, base=base, rc_grid=rc_grid, G_by_z=G_by_z,
                fine=fine, rc_family=fam, b_enu=local_field_enu(LAT, LON, date),
                species=ENG_SP)


def hv_of(pieces, prod, widths, *, sigma_scale=1.0, geom=None, n_alpha=44,
          n_beta=18, n_ray=260, renorm=None):
    """``{species: [H/V at HV_E]}`` for one variant of the joint factor.

    ``renorm``: optional ``{species: g[nE]}`` multiplying ``F`` (the Liouville
    constraint).  It is a function of energy only, so it cancels in H/V by
    construction -- reported to make that explicit.
    """
    import muon_bending as _mb
    from mceq3d_flux import ox_regrid

    eng, cz = pieces["eng"], pieces["cz"]
    ENG_SP = pieces["species"]
    ep = prod["ep_grid"]
    chan = {s: prod["chan"][jc.CHANNEL_SPECIES_MAP[s]] for s in ENG_SP}
    smu = {s: widths["mu_nue" if "nue" in s else "mu_numu"] for s in ENG_SP}
    MU_PLUS = ("total_antinumu", "total_nue")
    pr = dict(prod)
    if geom is not None:
        pr["geom"] = geom
    q_cache = {}
    F = {s: np.zeros((len(cz), len(HV_AZ), len(ep))) for s in ENG_SP}
    for iz, c in enumerate(cz):
        zen = float(np.degrees(np.arccos(np.clip(c, -1, 1))))
        Gz = {s: ox_regrid(pieces["G_by_z"][iz][s], eng.e, ep) for s in ENG_SP}
        for ia, azd in enumerate(HV_AZ):
            v = _mb.muon_velocity_enu(zen, float(azd))
            d0 = _mb.bending_deflection(v, pieces["b_enu"], charge=+1)
            d_cone = np.array([d0[1], d0[0], d0[2]])
            r = jc.delivered_joint_factor(
                c, [azd], pr, pieces["fine"], pieces["rc_grid"], Gz,
                sigma_pi=widths["pi"] , sigma_k=widths["k"],
                channels_by_species=chan, sigma_mu_by_species=smu,
                d_cone=d_cone, mu_plus=MU_PLUS, prod_frame=True,
                n_alpha=n_alpha, n_beta=n_beta, n_ray=n_ray,
                sigma_scale=sigma_scale, q_cache=q_cache,
                rc_family=pieces["rc_family"], site_lat=LAT)
            for s in ENG_SP:
                F[s][iz, ia] = r["F"][s][0] * (1.0 if renorm is None else renorm[s])
    out = {}
    for s in ENG_SP:
        f = np.array([[np.exp(np.interp(np.log(eng.e), np.log(ep),
                                        np.log(np.maximum(F[s][iz, ia], 1e-300))))
                       * pieces["base"][s][iz] for ia in range(len(HV_AZ))]
                      for iz in range(len(cz))])
        out[s] = [_log_at(f[0].mean(0), eng.e, E) / _log_at(f[-1].mean(0), eng.e, E)
                  for E in HV_E]
    return out


def report_hv(prod, widths, args):
    ref = dict(np.load(os.path.join(HERE, "honda_kam.npz")))
    bt = dict(np.load(os.path.join(HERE, "bartol_kam.npz")))
    He, Hcz = ref["E"], ref["czlo"]
    ihz, ivz = int(np.argmin(abs(Hcz - 0.0))), int(np.argmin(abs(Hcz - 0.9)))
    hond, bart = {}, {}
    for fl, key in (("numu", "num"), ("nue", "nue")):
        nm = ref[fl]
        hond[fl] = [_log_at(nm[ihz].mean(0), He, E) / _log_at(nm[ivz].mean(0), He, E)
                    for E in HV_E]
        y = 0.5 * (bt[f"{key}_fmin"] + bt[f"{key}_fmax"])
        ih = int(np.argmin(abs(bt["cz"] - 0.05)))
        iv = int(np.argmin(abs(bt["cz"] - 0.95)))
        bart[fl] = [_log_at(y[ih], bt["E"], E) / _log_at(y[iv], bt["E"], E)
                    for E in HV_E]

    pieces = engine_pieces(n_jobs=args.n_jobs)
    variants = [("default", {}),
                ("cone x0.75", dict(sigma_scale=0.75)),
                ("cone x1.25", dict(sigma_scale=1.25)),
                ("quad refined", dict(n_alpha=176, n_beta=72, n_ray=520))]
    if args.fine_geom:
        variants.append(("limb-refined geom", dict(geom=fine_geom(prod))))
    print("=" * 84)
    print("H. HORIZON/VERTICAL of the delivered flux (base x joint factor),")
    print("   cz = 0.05 / 0.95, azimuth-averaged, Kamioka, hybrid base")
    print("=" * 84)
    print(f"{'':<22}" + "".join(f"{E:>10.2f} GeV" for E in HV_E))
    for fl in ("numu", "nue"):
        print(f"\n[{fl}]")
        print(f"{'Honda':<22}" + "".join(f"{v:>14.3f}" for v in hond[fl]))
        print(f"{'Bartol':<22}" + "".join(f"{v:>14.3f}" for v in bart[fl]))
        for tag, kw in variants:
            hv = hv_of(pieces, prod, widths, **kw)[f"total_{fl}"]
            print(f"{tag:<22}" + "".join(
                f"{v:>8.3f}({v / hond[fl][i]:4.2f}/{v / bart[fl][i]:4.2f})"
                for i, v in enumerate(hv)))
    print("\n(x.xx/y.yy) = ratio to Honda / to Bartol")
    print("DIAG_HV_DONE")


if __name__ == "__main__":
    main()
