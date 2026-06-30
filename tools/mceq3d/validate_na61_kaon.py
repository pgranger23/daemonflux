"""Validate the kaon production p_T against NA61/SHINE K+- data.

Companion to ``validate_na61.py`` (pions). Kaons matter for the atmospheric flux
because K -> mu nu is the dominant nu_mu source above a few GeV and the leading
nu_e source at high energy, and the K+/K- yield asymmetry (proton-beam valence
effect) drives part of the nu/nubar ratio. As with pions, MCEq's 1D kernels carry
no angular information, so the production p_T must be checked against fixed-target
data.

Reference: NA61/SHINE K+- production in p+C at 31 GeV/c (HEPData ins1397003, the
comprehensive pi/K/p paper). K+ = Tables 23-30, K- = Tables 31-37, each a
double-differential d2(sigma)/dp/dtheta [mb/rad/(GeV/c)] vs lab momentum p in a
polar-angle bin theta (mrad). We form <p_T>(p) by weighting each (p, theta)
cell's yield by p*sin(theta_center), and compare to UrQMD-3.4 (the low-energy
model used to regenerate the kernels) under the same forward acceptance.

Data access: the tables are fetched with **hepdata-cli** (``pip install
hepdata-cli``) into ``na61_k/`` as HEPData YAML and cached there; delete the
folder to refetch. Run::

    python validate_na61_kaon.py --plot
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess

import numpy as np

M_KAON = 0.493677
RECORD = "1397003"  # inspire id
KDIR = "na61_k"
KPLUS_TABLES = [f"Table{n}" for n in range(23, 31)]  # K+
KMINUS_TABLES = [f"Table{n}" for n in range(31, 38)]  # K-
THETA_ACC = 0.300  # rad; max polar angle covered by the K+- tables (~300 mrad)


def fetch_kaon_tables(kdir=KDIR):
    """Download the NA61 K+- tables via hepdata-cli into ``kdir`` (YAML)."""
    os.makedirs(kdir, exist_ok=True)
    for n in range(23, 38):
        subprocess.run(
            [
                "hepdata-cli", "download", RECORD, "-i", "inspire",
                "-f", "yaml", "-t", f"Table {n}", "-d", kdir,
            ],
            check=True,
        )
    # hepdata-cli names files HEPData-ins...-v1-Table_N.yaml; normalise to TableN.yaml
    for f in glob.glob(os.path.join(kdir, "*Table_*.yaml")):
        n = f.split("Table_")[1].split(".")[0]
        os.replace(f, os.path.join(kdir, f"Table{n}.yaml"))


def load_kaon(kdir=KDIR):
    import yaml

    if not glob.glob(os.path.join(kdir, "Table2*.yaml")):
        fetch_kaon_tables(kdir)
    out = {}
    for n in range(23, 38):
        d = yaml.safe_load(open(os.path.join(kdir, f"Table{n}.yaml")))
        dv = d["dependent_variables"][0]
        q = {x["name"]: x["value"] for x in dv["qualifiers"]}
        out[f"Table{n}"] = {
            "reaction": q["RE"],
            "theta": q["THETA"],  # "lo-hi" in mrad
            "p": d["independent_variables"][0]["values"],
            "values": dv["values"],
        }
    return out


def _theta_center_rad(theta_str):
    lo, hi = theta_str.split("-")
    return 0.5 * (float(lo) + float(hi)) * 1e-3  # mrad -> rad


def na61_ptmean(data, tables, p_edges):
    """NA61 <p_T>(p_lab) for one charge, yield-weighted over polar angle."""
    p_c, p_t, wt = [], [], []
    for name in tables:
        tab = data[name]
        th = _theta_center_rad(tab["theta"])
        for pbin, v in zip(tab["p"], tab["values"]):
            lo, hi = float(pbin["low"]), float(pbin["high"])
            p = 0.5 * (lo + hi)
            dsig = v.get("value", np.nan)
            try:
                dsig = float(dsig)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(dsig) or dsig <= 0:
                continue
            p_c.append(p)
            p_t.append(p * np.sin(th))
            wt.append(dsig * (hi - lo))  # yield in the (p, theta) cell
    p_c, p_t, wt = map(np.array, (p_c, p_t, wt))
    idx = np.digitize(p_c, p_edges) - 1
    centers = np.sqrt(p_edges[:-1] * p_edges[1:])
    out = np.full(len(centers), np.nan)
    for b in range(len(centers)):
        sel = idx == b
        if sel.any() and wt[sel].sum() > 0:
            out[b] = np.sum(p_t[sel] * wt[sel]) / np.sum(wt[sel])
    return centers, out


def urqmd_ptmean(
    p_edges, pids=(321, -321), n_events=8000, e_lab=31.0, theta_max_rad=THETA_ACC
):
    """UrQMD-3.4 <p_T>(p_lab) for p+C at ``e_lab`` GeV/c, same binning/acceptance."""
    import chromo
    from chromo.kinematics import FixedTarget, GeV

    model = chromo.models.UrQMD34(FixedTarget(e_lab * GeV, 2212, (12, 6)))
    pl_acc = {pid: [] for pid in pids}
    pt_acc = {pid: [] for pid in pids}
    for event in model(n_events):
        fs = event.final_state()
        for pid in pids:
            sel = fs.pid == pid
            en = fs.en[sel]
            pl = np.sqrt(np.maximum(en**2 - M_KAON**2, 0.0))
            pt = fs.pt[sel]
            if theta_max_rad is not None:
                acc = (
                    np.arctan2(pt, np.sqrt(np.maximum(pl**2 - pt**2, 0.0)))
                    < theta_max_rad
                )
                pl, pt = pl[acc], pt[acc]
            pl_acc[pid].append(pl)
            pt_acc[pid].append(pt)

    centers = np.sqrt(p_edges[:-1] * p_edges[1:])
    out = {}
    for pid in pids:
        p_lab = np.concatenate(pl_acc[pid])
        p_t = np.concatenate(pt_acc[pid])
        idx = np.digitize(p_lab, p_edges) - 1
        pm = np.full(len(centers), np.nan)
        for b in range(len(centers)):
            sel = idx == b
            if sel.any():
                pm[b] = np.mean(p_t[sel])
        out[pid] = (centers, pm)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nevents", type=int, default=20000)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args(argv)

    data = load_kaon()
    p_edges = np.logspace(np.log10(0.8), np.log10(16.0), 12)

    e_kp, pt_kp = na61_ptmean(data, KPLUS_TABLES, p_edges)
    e_km, pt_km = na61_ptmean(data, KMINUS_TABLES, p_edges)
    ur = urqmd_ptmean(p_edges, (321, -321), n_events=args.nevents)
    e_ur, pt_ur = ur[321]
    e_urm, pt_urm = ur[-321]

    print("K+ <p_T>(p_lab)  [p+C, 31 GeV/c, theta < 300 mrad acceptance]")
    print("  p_lab[GeV]   NA61[GeV]   UrQMD[GeV]   ratio")
    for e, a, b in zip(e_kp, pt_kp, pt_ur):
        if np.isfinite(a) and np.isfinite(b):
            print(f"  {e:8.2f}    {a:8.3f}    {b:8.3f}     {b / a:5.2f}")
    print("K- <p_T>(p_lab)")
    print("  p_lab[GeV]   NA61[GeV]   UrQMD[GeV]   ratio")
    for e, a, b in zip(e_km, pt_km, pt_urm):
        if np.isfinite(a) and np.isfinite(b):
            print(f"  {e:8.2f}    {a:8.3f}    {b:8.3f}     {b / a:5.2f}")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        ax.plot(e_kp, pt_kp, "ko", label=r"NA61 $K^+$ (data)")
        ax.plot(e_ur, pt_ur, "C0-", lw=2, label=r"UrQMD $K^+$")
        ax.plot(e_km, pt_km, "ks", mfc="none", label=r"NA61 $K^-$ (data)")
        ax.plot(e_urm, pt_urm, "C3-", lw=2, label=r"UrQMD $K^-$")
        ax.set_xscale("log")
        ax.set_xlabel(r"kaon lab momentum $p$ [GeV]")
        ax.set_ylabel(r"$\langle p_T\rangle$ [GeV]")
        ax.set_title("p+C at 31 GeV/c: kaon production vs NA61/SHINE")
        ax.set_ylim(0, 0.7)
        ax.legend()
        fig.tight_layout()
        fig.savefig("validate_na61_kaon_pt.png", dpi=110)
        print("saved plot -> validate_na61_kaon_pt.png")


if __name__ == "__main__":
    main()
