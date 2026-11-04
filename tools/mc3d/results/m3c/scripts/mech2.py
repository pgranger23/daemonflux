"""Task 2 (milestone 3c): the in-run decomposition of the muon effect
(``--mu-cf`` runs).  For every re-transported muon the run also decays its
zero-field counterfactual (same proper time, same decay numbers), so

  field on - field off = migration (muons that decay either way; only the
                          neutrino directions differ: class 0, on - cf)
                       + saved     (decay only with the field: class 1, on)
                       - lost      (decay only without it:      class 2, cf)

per (species, E band, cos Z bin, az bin), measure 1, the s weight (with the
copy's cutoff verdict).  Linear Fourier components about geomagnetic West
(bin-width deconvolved like analyse_m3b): eps = c1(piece) / a0(field off),
da0 = a0(piece) / a0(field off), a0(field off) = s - muons(on) + muons(cf).

    python mech2.py TAG [TAG...] -> $M/reports/mech2_<TAG>.txt
"""
import os
import sys

import numpy as np

import m3c_lib as L

M = os.environ["M"]
AZ = 15.0 + 30.0 * np.arange(12)
SINC = np.sin(np.radians(15.0)) / np.radians(15.0)


def four(f, W):
    """(a0, c1, s1) of patterns (..., 12)."""
    c = np.cos(np.radians(AZ - W))
    s = np.sin(np.radians(AZ - W))
    return np.stack([f.mean(-1), 2 * (f * c).mean(-1) / SINC,
                     2 * (f * s).mean(-1) / SINC], -1)


W = (L.axis() + 180.0) % 360.0
for tag in sys.argv[1:]:
    d = L.load(tag)
    # mucf holds raw weights (every entry once, no 1/area): = the largest
    # cap's tally x its area, the cap that ``ew`` is reduced on
    area = float(np.load(f"{M}/sky/{tag}_total.npz")["cap_area"][-1])
    S = {"s": d["ew"][:, :, 0], "cf": d["mucf"] / area}   # (G,K,sp,band,cz,az)
    lines = [f"# {tag}: decomposition of the muon effect (pooled cos Z 0-0.3 and "
             "per cos Z bin); eps = c1/a0(off), towards geomagnetic West"]

    def pieces(s, cf):
        """dict of patterns (sp, band, cz, az)."""
        on_mu = cf[0].sum(0)
        off_mu = cf[1].sum(0)
        off = s - on_mu + off_mu
        return {"off": off, "on": s, "migration": cf[0, 0] - cf[1, 0],
                "saved": cf[0, 1], "lost": -cf[1, 2], "total": on_mu - off_mu,
                "mu_off": off_mu}

    def stats(P):
        a0off = four(P["off"].sum(2), W)[..., 0]          # pooled (sp, band)
        out = {}
        for k in ("total", "migration", "saved", "lost", "mu_off"):
            fo = four(P[k].sum(2), W)
            out[k] = np.stack([fo[..., 1] / a0off, fo[..., 2] / a0off,
                               fo[..., 0] / a0off], -1)    # eps_c, eps_s, da0
        return out

    tot = {k: L.total(v) for k, v in S.items()}
    rep = {k: L.replicates(v, d["M"]) for k, v in S.items()}
    est = stats(pieces(tot["s"], tot["cf"]))
    reps = [stats(pieces(rep["s"][g], rep["cf"][g])) for g in range(rep["s"].shape[0])]
    for k in ("total", "migration", "saved", "lost", "mu_off"):
        e = est[k]
        er = L.jk_err(np.array([r[k] for r in reps]))
        lab = {"mu_off": "muon-born share of the field-off flux (da0 = share)"}.get(k, k)
        lines.append(f"## {lab}: eps_c, eps_s, da0")
        for ib, E in enumerate(L.EB):
            lines.append(f"E {E}: " + "  ".join(
                f"{sp} {e[kk, ib, 0]:+.3f}+-{er[kk, ib, 0]:.3f} {e[kk, ib, 1]:+.3f}+-{er[kk, ib, 1]:.3f}"
                f" {e[kk, ib, 2]:+.3f}+-{er[kk, ib, 2]:.3f}" for kk, sp in enumerate(L.SP)))
    # charge-odd / -even parts of the total (mu+ species: nue, antinumu)
    e = est["total"][..., 0]
    er = np.array([r["total"][..., 0] for r in reps])
    for lab_, (i, j) in (("e: (nue -+ antinue)/2", (0, 1)),
                         ("mu: (antinumu -+ numu)/2", (3, 2))):
        odd, even = 0.5 * (e[i] - e[j]), 0.5 * (e[i] + e[j])
        oe = L.jk_err(0.5 * (er[:, i] - er[:, j]))
        ee = L.jk_err(0.5 * (er[:, i] + er[:, j]))
        lines.append(f"charge-odd {lab_}: " + " ".join(
            f"{odd[ib]:+.3f}+-{oe[ib]:.3f}" for ib in range(3))
            + "   charge-even: " + " ".join(f"{even[ib]:+.3f}+-{ee[ib]:.3f}"
                                         for ib in range(3)))
    np.savez(f"{M}/reports/mech2_{tag}.npz",
             **{f"{k}": est[k] for k in est},
             **{f"{k}_err": L.jk_err(np.array([r[k] for r in reps])) for k in est})
    # per-bin azimuth patterns of the pieces for nue/antinue 0.3 GeV, cz 0-0.3
    P = pieces(tot["s"], tot["cf"])
    a0 = P["off"].sum(2).mean(-1)
    for kk in (0, 1):
        for k in ("migration", "saved", "lost"):
            lines.append(f"pattern {L.SP[kk]} 0.3 GeV {k:9s} / a0(off): " + " ".join(
                f"{x / a0[kk, 0]:+.3f}" for x in P[k][kk, 0].sum(0)))
    open(f"{M}/reports/mech2_{tag}.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
