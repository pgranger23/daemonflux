"""[milestone 3e adaptation of m3c c3_conventions.py] Gate C3 West (milestone 3c, task 1): the engine's G_s re-derived from
MCEq's per-cell responses (``mceq_response.py``) under the MC's and the
engine's conventions, against the MC's v/t (pilot), plus the per-stratum
no-cutoff yields (candidate b).

Conventions that differ between ``tools/mc3d`` and ``mceq3d_flux.geomag_response``:

  rigidity of the cut   MC: R = (A/Z) p_nucleon(T)        engine: R = (A/Z) T
                        (cutoff.rigidity_gv)               (_transmission on the
                                                            kinetic grid e_grid)
  H3a energy variable   MC: crflux at A*T                 MCEq: crflux at A*(T+m)
                        (primaries.nucleon_intensity)      (set_primary_model)
  lowest primary        MC: T = 1 GeV/n (sky.STRATA)       MCEq: E_tot >= 3 GeV
                                                            (minimal_primary_energy)
  nuclei                MC: five groups, own A/Z           engine: free p = p - n,
                                                            bound <A/Z>

    python c3_conventions.py -> $M/reports/c3_conventions.txt, c3_conventions.npz
"""
import os

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "m3c", "scripts"))
import m3c_lib as L
import sky
import sky3d
import cutoff

M = os.environ["M"]
W = os.environ["W"]
MP, MN = 0.938272, 0.939565
MBAR = 0.5 * (MP + MN)
lines = []


def out(s=""):
    print(s, flush=True)
    lines.append(s)


d = dict(np.load(M + "/work/mceq_resp_h3a.npz"))   # mceq_response.py on the v142p database
e, ebins = d["e"], d["e_bins"]
cells = d["cells"]
nsingle = int(np.sum(cells[:, 1] - cells[:, 0] == 1))
jsingle = cells[:nsingle, 0]
psl = slice(*d["p_sl"])
phi_std_p = d["phi_std"][psl]
p_ext, n_ext = d["p_ext"], d["n_ext"]
f_free, az_bound = d["f_free"], d["az_bound"]
eb = sky.E_BINS
de = np.gradient(e)
BANDS = list(L.EB)


def band(x, ib):
    """Band integral of a flux on MCEq's grid: each MCEq bin's density times
    its overlap with the band [eb[i0], eb[i1]) (the gate itself weights the
    bin centres by np.gradient, fine for G, ~20% off for absolute values)."""
    i0, i1 = list(sky3d.MUB_EBANDS)[ib]
    lo, hi = eb[i0], eb[i1]
    w = np.clip(np.minimum(ebins[1:], hi) - np.maximum(ebins[:-1], lo), 0, None)
    return (x * w).sum(-1)


import crflux.models as crf  # noqa: E402
H3A = crf.HillasGaisser2012("H3a")
GROUPS = [(g, a_, z_, gid) for g, a_, z_, gid in
          (("p", 1, 1, 14), ("He", 4, 2, 402), ("CNO", 14, 7, 1407),
           ("MgSi", 25, 12, 2512), ("Fe", 56, 26, 5626))]


import primaries as _prim
_MC = _prim.primary_model(os.environ.get("M3E_MODEL", "GSFG:GH"))


def phi_group(T, a_, z_, gid, conv):
    """(p, n) nucleon flux [cm^-2 s^-1 sr^-1 GeV^-1] of a group at kinetic
    energy per nucleon T: crflux at A*(T+m) (MCEq) or A*T (MC)."""
    if conv == "mc":                       # the MC's own primaries, native variable
        jp, jn = _prim.group_split(_MC, {1: 14, 4: 402, 14: 1407, 25: 2512, 56: 5626}[a_], T)
        return np.ravel(jp), np.ravel(jn)
    E = a_ * (T + MBAR)
    f = 1e-4 * a_ * np.ravel(H3A.nucleus_flux(gid, E))
    return z_ * f, (a_ - z_) * f


# check: the per-group MCEq-convention fluxes add up to MCEq's own
pp = sum(phi_group(e, a_, z_, g, "mceq")[0] for _, a_, z_, g in GROUPS)
nz = phi_std_p > 0
out(f"# check: sum of groups vs MCEq p flux where non-zero: max rel diff "
    f"{np.max(np.abs(pp[nz] / phi_std_p[nz] - 1)):.2e}; MCEq's first non-zero "
    f"primary bin T = {e[np.argmax(nz)]:.3f} GeV (lower edge {ebins[np.argmax(nz)]:.3f})")


def rig(T, aoz, kind):
    return aoz * (T if kind == "kin" else np.sqrt(T * (T + 2 * MP)))


class Resp:
    """Band-integrated responses at one cos Z: per unit T at the single-bin
    centres (interpolated linearly in ln T inside), and the lumps."""

    def __init__(self, cz):
        self.cz = cz
        self.y = {}
        self.lump = {}
        for nuc, ext in (("p", p_ext), ("n", n_ext)):
            R = d[f"resp|{cz}|{nuc}"]                     # (cell, sp, e)
            Rb = np.stack([band(R, ib) for ib in range(3)], -1)  # (cell, sp, band)
            j = jsingle
            dT = ebins[j + 1] - ebins[j]
            self.y[nuc] = Rb[:nsingle] / (ext[j] * dT)[:, None, None]
            self.lump[nuc] = Rb[nsingle:]
        self.full = np.stack([[band(d[f"full|{cz}|{s}"], ib) for ib in range(3)]
                              for s in ("total_nue", "total_antinue", "total_numu",
                                        "total_antinumu")])          # (sp, band)

    def yield_T(self, nuc, T):
        """(len(T), sp, band) per-unit-T yield, linear in ln T between the
        single-bin centres (constant beyond)."""
        x = np.log(e[jsingle])
        y = self.y[nuc]
        out_ = np.empty((len(T),) + y.shape[1:])
        for idx in np.ndindex(y.shape[1:]):
            out_[(slice(None),) + idx] = np.interp(np.log(T), x, y[(slice(None),) + idx])
        return out_


# the single-bin region as a fine grid, plus lump flux ratios MC/MCEq
T_lo, T_hi = ebins[jsingle[0]], ebins[jsingle[-1] + 1]
TF = np.exp(np.linspace(np.log(T_lo), np.log(T_hi), 3000))
dTF = np.gradient(TF)
lump_edges = [(ebins[c0], ebins[c1]) for c0, c1 in cells[nsingle:]]


def lump_ratio(conv, a_, z_, gid, nuc, lo, hi):
    Tq = np.exp(np.linspace(np.log(lo), np.log(min(hi, 1e7)), 400))
    k = 0 if nuc == "p" else 1
    num = np.trapezoid(phi_group(Tq, a_, z_, gid, conv)[k], Tq)
    den = np.trapezoid(phi_group(Tq, a_, z_, gid, "mceq")[k], Tq)
    return num / den if den > 0 else 0.0


def yields(resp, conv, kind, t_min, rc, t_max_mc=3162.28):
    """(cut, full) band yields (sp, band) summed over groups, for a cut at
    rigidity ``rc`` (None: no cut).  Lumps are never cut (all above 31.6
    GeV/n, R >= 63 GV for A/Z >= 2 -- valid for the vertical and the West)."""
    tot_cut = np.zeros((4, 3))
    tot = np.zeros((4, 3))
    yT = {nuc: resp.yield_T(nuc, TF) for nuc in ("p", "n")}
    for g, a_, z_, gid in GROUPS:
        fp, fn = phi_group(TF, a_, z_, gid, conv)
        ok = TF >= t_min
        cut = ok if rc is None else ok & (rig(TF, a_ / z_, kind) > rc)
        for nuc, f in (("p", fp), ("n", fn)):
            w = (f * dTF)[:, None, None] * yT[nuc]
            tot += (w * ok[:, None, None]).sum(0)
            tot_cut += (w * cut[:, None, None]).sum(0)
            # lumps: flux-reweighted from MCEq's (all groups together) by the
            # group's share -- each lump's response is for the summed flux, so
            # weight it by this group's fraction of the MCEq flux in the lump
            for il, (lo, hi) in enumerate(lump_edges):
                if conv == "mc" and lo >= t_max_mc:
                    continue
                Tq = np.exp(np.linspace(np.log(lo), np.log(min(hi, 1e7)), 200))
                k = 0 if nuc == "p" else 1
                share = np.trapezoid(phi_group(Tq, a_, z_, gid, "mceq")[k], Tq) / \
                    np.trapezoid(sum(phi_group(Tq, a2, z2, g2, "mceq")[k]
                                     for _, a2, z2, g2 in GROUPS), Tq)
                r = lump_ratio(conv, a_, z_, gid, nuc, lo, hi)
                x = resp.lump[nuc][il] * share * r
                tot += x
                tot_cut += x
    return tot_cut, tot


def g_engine(resp, rc):
    """The engine's own G_s (discrete, bin-averaged hard step at R = (A/Z) T,
    free/bound split) from the cell responses, for validation."""
    dl = np.log(ebins[1:] / ebins[:-1])
    num = np.zeros((4, 3))
    den = np.zeros((4, 3))
    for nuc in ("p", "n"):
        for ic in range(nsingle):
            j = jsingle[ic]
            if phi_std_p[j] <= 0:
                continue
            r = resp.y[nuc][ic] * (p_ext if nuc == "p" else n_ext)[j] * \
                (ebins[j + 1] - ebins[j])

            def tr(aoz):
                x = np.log(aoz * e[j]) - np.log(rc)
                return float(np.clip((x + 0.5 * dl[j]) / dl[j], 0.0, 1.0))
            t = (f_free[j] * tr(1.0) + (1 - f_free[j]) * tr(az_bound[j])) \
                if nuc == "p" else tr(az_bound[j])
            num += r * t
            den += r
        num += resp.lump[nuc].sum(0)
        den += resp.lump[nuc].sum(0)
    return num / den



VARIANTS = (("B  MCEq flux, R=(A/Z)p,   T>=2", ("mceq", "mom", 1.99)),
            ("C  MC flux,   R=(A/Z)p,   T>=1", ("mc", "mom", 1.0)),
            ("D  MC flux,   R=(A/Z)T,   T>=1", ("mc", "kin", 1.0)))
smap = sky3d.site_map(36.4267, 137.31, "IGRF14", cutoff.DEFAULT_DATE, 1, M + "/work")
BINS = (("vertical", 0.95, 19, None), ("87W 240-270", 0.05, 10, 8),
        ("87W 270-300", 0.05, 10, 9))
TAG = os.environ.get("M3E_TAG", "pilot2")
if os.environ.get("M3E_BINS"):
    BINS = tuple(b for b in BINS if b[0] in os.environ["M3E_BINS"].split(","))
pil = L.load(TAG)
reps_ew = L.replicates(pil["ew"], pil["M"])
reps_v = L.replicates(pil["vert"], pil["M"])
tot_ew, tot_v = L.total(pil["ew"]), L.total(pil["vert"])
summ = {}
for lab, czr, iz, ia in BINS:
    resp = Resp(czr)
    czs = -1.0 + 0.1 * iz + (np.arange(5) + 0.5) / 5 * 0.1
    azs = ((np.arange(24) + 0.5) * 15.0) if ia is None else 30.0 * ia + (np.arange(6) + 0.5) * 5.0
    Z, A_ = np.meshgrid(np.degrees(np.arccos(czs)), azs, indexing="ij")
    rcs = smap(Z.ravel(), A_.ravel())
    out(f"\n## {lab}: cos Z ref {czr}, site-map R_c over the bin {rcs.min():.2f}-{rcs.max():.2f} GV (mean {rcs.mean():.2f})")
    if ia is None:
        t, v = tot_v[2], tot_v[3]
        rr = reps_v[:, 3] / reps_v[:, 2]
    else:
        t, v = tot_ew[2, :, :, 0, ia], tot_ew[3, :, :, 0, ia]
        rr = reps_ew[:, 3, :, :, 0, ia] / reps_ew[:, 2, :, :, 0, ia]
    g_mc, g_mce = v / t, L.jk_err(rr)
    G = {}
    for name, spec in VARIANTS:
        acc = np.zeros((4, 3))
        cache = {}
        for r in np.unique(np.round(rcs, 2)):
            c_, f_ = yields(resp, spec[0], spec[1], spec[2], r)
            cache[r] = c_ / f_
        for r in rcs:
            acc += cache[np.round(r, 2)]
        G[name] = acc / len(rcs)
    out("species E:  MC v/t (%s) | " % TAG + " | ".join(G) + " | MC/C")
    for k, sp in enumerate(L.SP):
        for ib, E in enumerate(BANDS):
            c = G["C  MC flux,   R=(A/Z)p,   T>=1"][k, ib]
            out(f"{sp:9s} {E}: {g_mc[k, ib]:.3f}+-{g_mce[k, ib]:.3f} | " + " | ".join(f"{G[n][k, ib]:.3f}" for n in G)
                + f" | {g_mc[k, ib] / c:.3f}+-{g_mce[k, ib] / c:.3f}")
    c = G["C  MC flux,   R=(A/Z)p,   T>=1"]
    z = (g_mc / c - 1) / (g_mce / c)
    out(f"MC/C: mean {np.mean(g_mc / c):.3f}; chi2/12 vs 1 (MC errors) {np.nansum(z ** 2) / 12:.2f}")
    summ[lab] = dict(g_mc=g_mc, g_mce=g_mce, C=c, B=G["B  MCEq flux, R=(A/Z)p,   T>=2"], D=G["D  MC flux,   R=(A/Z)T,   T>=1"])
np.savez(M + f"/reports/gate_c3_mceq_{TAG}{os.environ.get("M3E_SUF", "")}.npz", **{f"{lab}|{k}": v for lab, dd in summ.items() for k, v in dd.items()})
open(M + f"/reports/gate_c3_mceq_{TAG}{os.environ.get("M3E_SUF", "")}.txt", "w").write("\n".join(lines) + "\n")
