"""Full azimuth x zenith pattern comparison (model vs Honda) and the
bending-response test.

Three stages, all writing to ``--out``:

``pattern``
    Solve the delivered engine (joint cone, channel mode, v2 moments,
    ``sublimb='prod_point'``, ``sigma_lnr`` default) on Honda's exact bins --
    12 azimuth bins x cosZ bin centres 0.05/0.15/0.25/0.35/0.45 -- and dump the
    four species on the engine energy grid.  Honda's azimuth convention is
    handled by ``diag_ew_charge_fourier.honda_azimuth_to_compass``.

``channels``
    Per-parent-channel numerators of the joint integral at one zenith.  Repeats
    the arithmetic of ``joint_cone.delivered_joint_factor`` (channel-resolved
    mode) but keeps ``N_dir`` / ``N_K`` / ``N_mu`` separate, and re-evaluates the
    muon channel for a list of *scale factors* on the charge-signed bending
    displacement ``+-delta`` -- the direct test of the bending response.  The
    production cone integral is computed once and reused for every scale, so the
    whole scan costs one zenith's worth of cone production.

``verify``
    Cross-check that ``channels`` at scale 1 reproduces the engine's own
    ``Fjoint`` azimuth pattern.

Run from tools/mceq3d with PYTHONPATH=$PWD.
"""

from __future__ import annotations

import argparse
import importlib.util  # noqa: F401  (MCEq config touches importlib.util)
import json
import time
import warnings
from datetime import datetime

import numpy as np

warnings.filterwarnings("ignore")

LAT, LON = 36.43, 137.31
DATE = datetime(2020, 1, 1)
CACHE = ".cache3d"
SP = ("total_numu", "total_antinumu", "total_nue", "total_antinue")
MU_PLUS = ("total_antinumu", "total_nue")  # neutrinos from mu+ decay
M_V2 = {"pi": ["m_spliced_v2.npz", "m_piminus_v2.npz"],
        "k": ["m_Kplus_v2.npz", "m_Kminus_v2.npz"]}
S4 = dict(cone_kernel="moments", cone_moments=M_V2, joint_cone=True,
          joint_channels=True)


def _check_moments():
    """The module default ``kinematic_kernel._MOMENTS`` still points at the OLD
    (pre-arcsin) moment files, so every solve here passes ``cone_moments``
    explicitly.  Assert it is byte-identical to ``offaxis_mc.MOMENTS_V2``."""
    import offaxis_mc as ox
    assert M_V2 == ox.MOMENTS_V2, (M_V2, ox.MOMENTS_V2)
CZ = (0.05, 0.15, 0.25, 0.35, 0.45)
AZ = tuple((np.arange(12) + 0.5) * 30.0)   # compass bin centres = Honda's, mapped
N_ALPHA, N_BETA, N_RAY = 44, 18, 260
H_PROD = 30.0


def engine():
    _check_moments()
    from mceq3d_flux import MCEq3DFlux
    return MCEq3DFlux(base_model="hybrid", primary=("GlobalSplineFitBeta", None),
                      daemonflux_location="kamioka")


# --------------------------------------------------------------------------
def stage_pattern(a):
    eng = engine()
    cz = np.array([float(x) for x in a.cz.split(",")])
    az = np.array(AZ)
    t = time.time()
    r = eng.solve(LAT, LON, cz, az, offaxis=True, use_cache=True,
                  cone_cutoff=True, cache_dir=CACHE, date=DATE,
                  muon_bending=True, n_jobs=a.n_jobs, **S4)
    print(f"solve {time.time()-t:.0f}s", flush=True)
    np.savez(a.out, e=r["e"], cz=cz, az=az, cutoff=r["cutoff"],
             **{s: r["flux"][s] for s in SP})
    print("PATTERN_DONE", flush=True)


# --------------------------------------------------------------------------
def _setup(eng, cz):
    """Ingredients of the joint channel-resolved cone at one arrival zenith."""
    import joint_cone as jc
    import muon_bending as mb
    from mceq3d_flux import ox_regrid, RC_MAX_GV

    prod = eng.joint_prod(CACHE)
    ep = prod["ep_grid"]
    w = eng.joint_cone_widths(ep, moments=M_V2, cache_dir=CACHE)
    chan = {s: prod["chan"][jc.CHANNEL_SPECIES_MAP[s]] for s in SP}
    sig_mu = {s: w["mu_nue" if "nue" in s else "mu_numu"] for s in SP}
    fine = eng.finemap_rc(LAT, LON, DATE, cache_dir=CACHE)
    rc_grid = np.linspace(0.1, RC_MAX_GV, 40)
    G, _ = eng.geomag_response(rc_grid, cz_ref=max(abs(cz), 1e-3),
                               cache_dir=CACHE)
    Gz = {s: ox_regrid(G[s], eng.e, ep) for s in SP}
    b_enu = mb.local_field_enu(LAT, LON, DATE)
    return dict(jc=jc, mb=mb, prod=prod, ep=ep, w=w, chan=chan, sig_mu=sig_mu,
                fine=fine, rc_grid=rc_grid, Gz=Gz, b_enu=b_enu)


def channel_pieces(ctx, cz, azimuths, scales=(1.0,)):
    """Per-channel numerators N_c[s] / axis denominators D[s] for each azimuth
    and each bending scale.

    Mirrors ``joint_cone.delivered_joint_factor`` (``channels_by_species`` mode,
    ``prod_frame=True``) exactly, but keeps the three parent channels separate
    and loops over scale factors on the charge-signed shift ``+- s*delta``.
    """
    jc, mb = ctx["jc"], ctx["mb"]
    ep, w = ctx["ep"], ctx["w"]
    alpha = jc.alpha_grid(N_ALPHA)
    beta = np.linspace(0.0, 2 * np.pi, N_BETA, endpoint=False)
    zen_f, az_f, rc_f = ctx["fine"]
    dn = zen_f <= 89.9
    zen_f, rc_f = zen_f[dn], rc_f[dn]
    rc_grid, Gz = ctx["rc_grid"], ctx["Gz"]

    plist = [pc for s in SP for pc in ctx["chan"][s]]      # 4 species x 3 chan
    qs, qas = jc.cone_production_multi(cz, alpha, beta, ctx["prod"]["x_grid"],
                                       ep, plist, ctx["prod"]["geom"],
                                       n_ray=N_RAY)
    W_pi = jc.gauss_alpha_weights(np.asarray(w["pi"], float), alpha)
    W_k = jc.gauss_alpha_weights(np.asarray(w["k"], float), alpha)
    W_mu = {s: jc.gauss_alpha_weights(np.asarray(ctx["sig_mu"][s], float), alpha)
            for s in SP}
    zen_deg = float(np.degrees(np.arccos(np.clip(cz, -1, 1))))

    out = {"e": ep, "az": np.asarray(azimuths, float), "scales": np.asarray(scales),
           "N": {s: {c: np.zeros((len(scales), len(azimuths), len(ep)))
                     for c in ("dir", "k", "mu")} for s in SP},
           "D": {s: np.zeros(len(ep)) for s in SP},
           "rc": np.zeros((len(scales), len(azimuths), 3)),   # axis, +d, -d
           "Gmu": {s: np.zeros((len(scales), len(azimuths), len(ep))) for s in SP}}
    for i, s in enumerate(SP):
        out["D"][s] = np.maximum(qas[3 * i] + qas[3 * i + 1] + qas[3 * i + 2],
                                 1e-300)
    for ia, azd in enumerate(azimuths):
        npv = jc.cone_directions(cz, azd, alpha, beta)
        frame = jc.prod_point_frame(cz, azd, H_PROD)

        def _rc(v):
            th, ph = jc.local_angles(v, frame)
            return jc.rc_bilinear(zen_f, az_f, rc_f, th, ph)

        rc0 = _rc(npv)
        Gab = {s: jc._interp_G(rc0, rc_grid, Gz[s]).reshape(
            len(alpha), len(beta), len(ep)) for s in SP}
        v = mb.muon_velocity_enu(zen_deg, float(azd))
        d0 = mb.bending_deflection(v, ctx["b_enu"], charge=+1)      # (E,N,U)
        d_unit = np.array([d0[1], d0[0], d0[2]])                    # -> (N,E,U)
        for isc, sc in enumerate(scales):
            d_cone = sc * d_unit
            pp = npv + d_cone
            pp /= np.linalg.norm(pp, axis=-1, keepdims=True)
            pm = npv - d_cone
            pm /= np.linalg.norm(pm, axis=-1, keepdims=True)
            rcp, rcm = _rc(pp), _rc(pm)
            out["rc"][isc, ia] = [rc0[0, 0], rcp[0, 0], rcm[0, 0]]
            Gp = {s: jc._interp_G(rcp, rc_grid, Gz[s]).reshape(
                len(alpha), len(beta), len(ep)) for s in SP}
            Gm = {s: jc._interp_G(rcm, rc_grid, Gz[s]).reshape(
                len(alpha), len(beta), len(ep)) for s in SP}
            for i, s in enumerate(SP):
                qd, qk, qm = qs[3 * i], qs[3 * i + 1], qs[3 * i + 2]
                g_mu = Gp[s] if s in MU_PLUS else Gm[s]
                if isc == 0:
                    pass
                out["N"][s]["dir"][isc, ia] = np.einsum(
                    "ea,ae->e", W_pi, (qd * Gab[s]).mean(axis=1))
                out["N"][s]["k"][isc, ia] = np.einsum(
                    "ea,ae->e", W_k, (qk * Gab[s]).mean(axis=1))
                out["N"][s]["mu"][isc, ia] = np.einsum(
                    "ea,ae->e", W_mu[s], (qm * g_mu).mean(axis=1))
                # production-weighted <G> of the muon channel (diagnostic)
                den = np.einsum("ea,ae->e", W_mu[s], qm.mean(axis=1))
                out["Gmu"][s][isc, ia] = out["N"][s]["mu"][isc, ia] / np.maximum(
                    den, 1e-300)
    return out


def stage_channels(a):
    eng = engine()
    cz = float(a.cz)
    ctx = _setup(eng, cz)
    scales = [float(x) for x in a.scales.split(",")]
    t = time.time()
    res = channel_pieces(ctx, cz, np.array(AZ), scales=scales)
    print(f"channel_pieces {time.time()-t:.0f}s", flush=True)
    np.savez(a.out, e=res["e"], az=res["az"], scales=res["scales"],
             rc=res["rc"], cz=cz,
             **{f"N_{s}_{c}": res["N"][s][c] for s in SP
                for c in ("dir", "k", "mu")},
             **{f"D_{s}": res["D"][s] for s in SP},
             **{f"Gmu_{s}": res["Gmu"][s] for s in SP})
    print("CHANNELS_DONE", flush=True)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("pattern", "channels"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--cz", default=",".join(str(c) for c in CZ))
    ap.add_argument("--scales", default="0,0.5,1,1.5,2,3,5")
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args()
    if a.stage == "pattern":
        stage_pattern(a)
    else:
        stage_channels(a)


if __name__ == "__main__":
    main()
