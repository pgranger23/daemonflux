"""Build the deliverables from the merged production (milestone 4).

    python make_table.py G  -> $M/out/kamioka_G_down_v1.npz   (needs B)
    python make_table.py R  -> $M/out/kamioka_Rmu_v1.npz      (needs addon)

G table: per species (nue, antinue, numu, antinumu), down-going cos Z 0-1 (5 bins of 0.2) x az 6 (60 deg) x E 5/decade
0.1-10 GeV (10 bins; bins 0-7 = 0.1-3.98 GeV are the deliverable range, the last two are the 3-10 GeV blend-check bins), plus
the same on a FINE grid (E 10/dec 0.1-10, cos Z 10 bins of 0.1, az 12).  G = s/t with s = 3D tally (cutoff, bending), t = 1D
no-cutoff reference of the same showers, v = 1D with the site cutoff (the MC's own per-primary verdict), n = 3D without
cutoff; jackknife (G = 20, groups = task index mod 20 per stratum) errors.  G is NOT clipped at 1.  Tallies are the largest
cap (theta_D = 10 deg), measure 1 per unit solid angle (cap-area divided, as ``sky3d``)."""
import os, sys, glob, hashlib
import numpy as np
sys.path.insert(0, os.environ["R"] + "/tools/mc3d")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "m3c", "scripts"))
import analyse_sky as A
import m3c_lib as L
M = os.environ["M"]; W = os.environ["W"]
which = sys.argv[1]
SP = ("nue", "antinue", "numu", "antinumu")
E_FINE_EDGES = 0.1 * 10 ** (np.arange(21) / 10.0)          # 10/dec, 0.1-10 GeV
E_COARSE = 0.1 * 10 ** (np.arange(11) / 5.0)               # 5/dec, 0.1-10 GeV
CZ_FINE = np.linspace(0, 1, 11); CZ_COARSE = np.linspace(0, 1, 6)
AZ_FINE = np.arange(13) * 30.0; AZ_COARSE = np.arange(7) * 60.0


def rebin(x, fe, fc, fa):
    s = x.shape
    return x.reshape(s[:-3] + (s[-3] // fe, fe, s[-2] // fc, fc, s[-1] // fa, fa)).sum((-1, -3, -5))


def jk(f, S, Mg):
    """estimate and jackknife error of f(totals) where S is (G, K, ...)."""
    tot = S.sum((0, 1))
    rep = L.replicates(S.reshape(S.shape[:2] + (-1,)), Mg).reshape((S.shape[0],) + S.shape[2:])
    e0 = f(tot); er = np.array([f(r) for r in rep]); G = len(er)
    return e0, np.sqrt((G - 1.0) / G * np.nansum((er - np.nanmean(er, 0)) ** 2, 0))


if which == "G":
    tag, rundir = "B", M + "/sky"
    base = np.load(W + "/runs/integ/out/base1d_kamioka.npz")
    Bt1 = base["B_t1"]                                       # (4, 60 [20/dec, 0.1-100], 20 cz) az-summed, measure 1
    assert str(base["database"]) == "v142p"
    def red(d):
        out = []
        for t in ("s1", "n1", "t1", "v1"):
            x = np.asarray(d[t])[-1][:, :40, 10:, :]        # (sp, 40 [20/dec 0.1-10], 10 cz down, 12 az)
            out.append(x.reshape(4, 20, 2, 10, 12).sum(2))   # E 10/dec
        return np.stack(out)                                  # (4tag, sp, 20, 10, 12)
    def red_tk(d):                                            # t1 for kappa: az-summed, cos Z 0.6-1, E 20/dec 0.1-10
        return np.asarray(d["t1"])[-1][:, :40, 16:20, :].sum(-1)      # (sp, 40, 4)
    J = A.JackRun(rundir, tag, G=20, reducers={"x": red, "tk": red_tk})
    S, Mg, TK = J.S["x"], J.M, J.S["tk"]
    man = dict(np.load(f"{rundir}/{tag}_tasks/manifest.npz"))
    # B per design cell: (sp, E 10/dec, cz down 10, az 12): az-summed B_t1 / 12; E pairs summed
    Bc = Bt1[:, :40, 10:].reshape(4, 20, 2, 10).sum(2)[..., None] / 12.0 * np.ones(12)       # (sp,20,10,12)
    Bk = Bt1[:, :40, 16:20]                                   # (sp, 40, 4)
    out = {"E_edges_fine": E_FINE_EDGES, "cosZ_edges_fine": CZ_FINE, "az_edges_fine": AZ_FINE,
           "E_edges_coarse": E_COARSE, "cosZ_edges_coarse": CZ_COARSE, "az_edges_coarse": AZ_COARSE,
           "species": np.array(SP),
           "note": np.array(["G = s / (kappa_s(E) B_s(E,|cosZ|)), not clipped; B = engine curved-atmosphere 1D (GSF, v142p, no cutoff), "
                             "kappa = t/B fitted over cos Z 0.6-1 (where bent primaries interact with probability >= 0.995); s/t and v/t are "
                             "diagnostics only (t = bent showers laid on straight lines: deficient at the horizon). E bins 0-7 (0.1-3.98 GeV) "
                             "= deliverable, 8-9 = 3-10 GeV blend check. cosZ = down-going 0-1. az = the bin order of sky.AZ_BINS (30 deg bins, "
                             "compass-angle of the arrival direction, 0 = first bin)."])}
    for name, (fe, fc, fa) in (("fine", (1, 1, 1)), ("coarse", (2, 2, 2))):
        Sx = rebin(S, fe, fc, fa)                             # (G,K,4tag,sp,E,cz,az)
        Bx = rebin(Bc[None, None], fe, fc, fa)[0, 0]           # (sp,E,cz,az)
        # kappa per (sp, E bin of this binning) from t over cos Z 0.6-1 (replicate-wise)
        def kap_of(tk, fe=fe):
            t = tk.reshape(4, 20 // fe, 2 * fe, 4).sum((2, 3))     # E 10/dec merged by fe: (sp, E)
            b = Bk.reshape(4, 20 // fe, 2 * fe, 4).sum((2, 3))
            return t / b
        tot_tk = TK.sum((0, 1)); k0 = kap_of(tot_tk)
        reps_s = L.replicates(Sx.reshape(Sx.shape[:2] + (-1,)), Mg).reshape((Sx.shape[0],) + Sx.shape[2:])
        reps_tk = L.replicates(TK.reshape(TK.shape[:2] + (-1,)), Mg).reshape((TK.shape[0],) + TK.shape[2:])
        def Gof(x, k):
            return x[0] / (k[:, :, None, None] * Bx)
        tot = Sx.sum((0, 1))
        G0 = Gof(tot, k0)
        Gr = np.array([Gof(rs, kap_of(rt)) for rs, rt in zip(reps_s, reps_tk)])
        Gn = len(Gr); Ge = np.sqrt((Gn - 1.0) / Gn * np.nansum((Gr - np.nanmean(Gr, 0)) ** 2, 0))
        kr = np.array([kap_of(rt) for rt in reps_tk]); ke = np.sqrt((Gn - 1.0) / Gn * np.nansum((kr - np.nanmean(kr, 0)) ** 2, 0))
        out[f"G_{name}"] = G0; out[f"G_err_{name}"] = Ge
        out[f"kappa_{name}"] = k0; out[f"kappa_err_{name}"] = ke
        out[f"B_{name}"] = Bx
        out[f"s_{name}"], out[f"n_{name}"], out[f"t_{name}"], out[f"v_{name}"] = tot[0], tot[1], tot[2], tot[3]
        # diagnostics (internal): s/t, v/t
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"s_over_t_{name}"] = tot[0] / tot[2]; out[f"v_over_t_{name}"] = tot[3] / tot[2]
    cpu = float(J.cpu.sum()) / 3600.0
    tabs = np.load("/afs/cern.ch/work/p/pigrange/daemonflux/tools/mc3d/mceq_tables_SIBYLL23D_v142p.npz")
    out.update(db_label=tabs["db_label"], db_sha256=tabs["db_sha256"],
               base1d_sha256=np.array([__import__("hashlib").sha256(open(W + "/runs/integ/out/base1d_kamioka.npz", "rb").read()).hexdigest()]),
               primary_model=np.array(["GSFG:GH (GSF nucleon level, GH composition), native=total energy"]),
               phi0_GV=np.array([0.55]), phi0_note=np.array(["assumed solar modulation potential of GSF (AMS-02 2011-13), not verified"]),
               seed0=man["seed0"], n_primaries=np.array([Mg.sum()]), cpu_core_h=np.array([cpu]),
               hybrid=np.array(["--r-thr 65 --resim-rmin 32 --resim-groups p --reanchor"]), site=man["site"], date=man["date"],
               field_model=man["field_model"], theta_inj=man["theta_inj"], caps=man["caps"])
    np.savez(M + "/out/kamioka_G_down_v1.npz", **out)
    print("wrote G table; primaries", Mg.sum(), "CPU(core-h) from the tasks:", cpu)
else:
    tag, rundir = "addon", M + "/sky"
    sys.argv = [sys.argv[0], rundir, tag]
    exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "addon_extract.py")).read())
    d = np.load(f"{M}/work/addon_{tag}.npz")
    S, Mg = d["x"], d["M"]
    def coarse(x):
        x = x[..., :28, 10:, :]
        s = x.shape
        return x.reshape(s[:-3] + (7, 4, 5, 2, 6, 2)).sum((-1, -3, -5))
    Sc = coarse(S)
    R0, Re = jk(lambda x: x[0] / (x[0] - x[1] + x[2]), Sc, Mg)
    man = dict(np.load(f"{rundir}/{tag}_tasks/manifest.npz"))
    # local field geometry at Kamioka (IGRF-14, 2020-01-01; the field the MC used)
    import field, sky, sky3d
    from constants import R_EARTH_CM
    lat, lon = sky.SITES["kamioka"]; o = sky3d.Options()
    f = field.FieldGrid(o.model, o.date, cache_dir=M + "/work")
    la, lo = np.radians(lat), np.radians(lon)
    up = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    east = np.array([-np.sin(lo), np.cos(lo), 0.0]); north = np.cross(up, east)
    Bv = np.atleast_2d(f(R_EARTH_CM * up[None, :]))[0]
    bn, be, bu = Bv @ north, Bv @ east, Bv @ up
    geom = dict(B_total_uT=np.array([np.linalg.norm(Bv) * 1e6]), B_north_east_up_uT=np.array([bn, be, bu]) * 1e6,
                declination_deg=np.array([np.degrees(np.arctan2(be, bn))]), dip_deg=np.array([np.degrees(np.arctan2(-bu, np.hypot(bn, be)))]),
                field_model=np.array([str(o.model)]), field_date=np.array([str(o.date)]), site_lat_lon=np.array([lat, lon]))
    out = {"E_edges": 0.1 * 10 ** (np.arange(8) / 5.0), "cosZ_edges": CZ_COARSE, "az_edges": AZ_COARSE, "species": np.array(SP),
           "R": R0, "R_err": Re, "F_on": Sc.sum((0, 1))[0], "mu_on": Sc.sum((0, 1))[1], "mu_cf": Sc.sum((0, 1))[2],
           "n_primaries": np.array([Mg.sum()]), "cpu_core_h": np.array([float(d["cpu"].sum()) / 3600.0]), "seed0": man["seed0"],
           "definition": np.array(["R_s = F_on/F_off, F_off = F_on - (muon-born, field on) + (same muon decays, zero field); down-going, no cutoff, no primary bend, "
                                   "from --mu-cf on the same showers; F_on = s tally of the largest cap"])}
    out.update(geom)
    np.savez(M + "/out/kamioka_Rmu_v1.npz", **out)
    print("wrote R table; primaries", Mg.sum())
