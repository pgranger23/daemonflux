"""Validate the regenerated angular/p_T content against NA61/SHINE data.

MCEq's 1D kernels carry no angular information, so the production p_T cannot be
checked against MCEq. The right reference is fixed-target data. Here we compare
the mean transverse momentum <p_T>(p_lab) of charged pions from our regenerated
kernel (UrQMD-3.4, the low-energy model) against the NA61/SHINE measurement of
pi+- production in p+C interactions at 31 GeV/c (HEPData ins886780, the T2K
thin-target data) -- the same energy/target regime, where 3D effects matter.

NA61 reports d(sigma)/dp [mb/GeV] in bins of momentum p for ten polar-angle
ranges theta. We form <p_T>(p) by weighting each (p, theta) cell's yield
(d(sigma)/dp * dp) by p*sin(theta_center).

The NA61 tables are cached in ``na61_886780_cache.json`` (fetched via HEPData);
delete it to refetch. Run::

    python validate_na61.py --plot
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

M_PION = 0.13957
CACHE = "na61_886780_cache.json"
RECORD = "ins886780"
# Table -> charge mapping (from the record: Tables 2-11 = pi+, 12-21 = pi-).
PIPLUS_TABLES = [f"Table{n}" for n in range(2, 12)]
PIMINUS_TABLES = [f"Table{n}" for n in range(12, 22)]


def fetch_na61_cache(path=CACHE):
    """Download the NA61 p+C 31 GeV/c tables to a local JSON cache."""
    import requests

    base = "https://www.hepdata.net/download/table/%s/Table %d/json"
    out = {}
    for n in range(2, 22):
        t = requests.get(base % (RECORD, n), timeout=60).json()
        q = t["qualifiers"]
        th_key = [k for k in q if k.startswith("THETA")][0]
        out[f"Table{n}"] = {
            "reaction": q["RE"][0]["value"],
            "theta": q[th_key][0]["value"],
            "values": t["values"],
        }
    json.dump(out, open(path, "w"))
    return out


def load_na61(path=CACHE):
    if not os.path.exists(path):
        return fetch_na61_cache(path)
    return json.load(open(path))


def _theta_center_rad(theta_str):
    # e.g. "0.0-20.0 Mrad" -> mean in radians
    lo, hi = theta_str.replace("Mrad", "").split("-")
    return 0.5 * (float(lo) + float(hi)) * 1e-3


def _yield_value(yentry):
    v = yentry.get("value", yentry.get("y"))
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def na61_ptmean(data, tables, p_edges):
    """NA61 <p_T>(p_lab) for one charge, yield-weighted over polar angle."""
    p_c, p_t, wt = [], [], []
    for name in tables:
        tab = data[name]
        th = _theta_center_rad(tab["theta"])
        for v in tab["values"]:
            lo = float(v["x"][0]["low"])
            hi = float(v["x"][0]["high"])
            p = 0.5 * (lo + hi)
            dsig = _yield_value(v["y"][0])
            if not np.isfinite(dsig) or dsig <= 0:
                continue
            p_c.append(p)
            p_t.append(p * np.sin(th))
            wt.append(dsig * (hi - lo))  # yield in the (p, theta) cell
    p_c = np.array(p_c)
    p_t = np.array(p_t)
    wt = np.array(wt)
    idx = np.digitize(p_c, p_edges) - 1
    centers = np.sqrt(p_edges[:-1] * p_edges[1:])
    out = np.full(len(centers), np.nan)
    for b in range(len(centers)):
        sel = idx == b
        if sel.any() and wt[sel].sum() > 0:
            out[b] = np.sum(p_t[sel] * wt[sel]) / np.sum(wt[sel])
    return centers, out


def urqmd_ptmean(
    p_edges, pids=(211, -211), n_events=8000, e_lab=31.0, theta_max_rad=None
):
    """UrQMD-3.4 <p_T>(p_lab) for p+C at ``e_lab`` GeV/c, same binning.

    Runs the generator *once* (chromo allows only one Fortran model per process)
    and returns ``{pid: (centers, ptmean)}`` for each requested pid.

    ``theta_max_rad`` applies NA61's forward angular acceptance (the tables
    cover theta < 420 mrad); without it, UrQMD's full-4pi <p_T> is biased high
    at low momentum where NA61 cannot see the large-angle pions.
    """
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
            pl = np.sqrt(np.maximum(en**2 - M_PION**2, 0.0))
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
    p.add_argument("--nevents", type=int, default=8000)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args(argv)

    data = load_na61()
    p_edges = np.logspace(np.log10(0.3), np.log10(20.0), 16)

    # NA61 forward acceptance is theta < 420 mrad; apply the same cut to UrQMD.
    theta_acc = 0.420
    e_na61, pt_na61 = na61_ptmean(data, PIPLUS_TABLES, p_edges)
    e_na61m, pt_na61m = na61_ptmean(data, PIMINUS_TABLES, p_edges)
    ur = urqmd_ptmean(
        p_edges, (211, -211), n_events=args.nevents, theta_max_rad=theta_acc
    )
    e_ur, pt_ur = ur[211]
    e_urm, pt_urm = ur[-211]

    print("pi+ <p_T>(p_lab)  [p+C, 31 GeV/c, theta < 420 mrad acceptance]")
    print("  p_lab[GeV]   NA61[GeV]   UrQMD[GeV]   ratio")
    for e, a, b in zip(e_na61, pt_na61, pt_ur):
        if np.isfinite(a) and np.isfinite(b):
            print(f"  {e:8.2f}    {a:8.3f}    {b:8.3f}     {b / a:5.2f}")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        ax.plot(e_na61, pt_na61, "ko", label=r"NA61 $\pi^+$ (data)")
        ax.plot(e_ur, pt_ur, "C0-", lw=2, label=r"UrQMD $\pi^+$ (regen)")
        ax.plot(e_na61m, pt_na61m, "ks", mfc="none", label=r"NA61 $\pi^-$ (data)")
        ax.plot(e_urm, pt_urm, "C3-", lw=2, label=r"UrQMD $\pi^-$ (regen)")
        ax.set_xscale("log")
        ax.set_xlabel(r"pion lab momentum $p$ [GeV]")
        ax.set_ylabel(r"$\langle p_T\rangle$ [GeV]")
        ax.set_title("p+C at 31 GeV/c: regenerated kernel vs NA61/SHINE")
        ax.set_ylim(0, 0.5)
        ax.legend()
        fig.tight_layout()
        fig.savefig("validate_na61_pt.png", dpi=110)
        print("saved plot -> validate_na61_pt.png")


if __name__ == "__main__":
    main()
