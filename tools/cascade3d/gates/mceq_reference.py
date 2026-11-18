"""MCEq reference fluxes for gate G1.1 (straight-ahead limit of ``cascade3d``).

Runs MCEq itself (database, populations, H3a and CORSIKA BK_USStd atmosphere exactly
as ``kernels_mceq.make_mceq`` / ``tools/mc3d/atmosphere.py``; default observation
level h_obs = 0) at cos Z = 1, 0.75, 0.5, 0.25, 0.15, 0.05 for each population and
with continuous energy loss on and off (``MCEq.config.enable_energy_loss``, read when
the matrices are built, so a separate run is constructed for each setting).

Output ``$N3/c1/ref/mceq_g11_<PHYS|REF>_<losson|lossoff>.npz`` (no pickles):

``cos_z`` (6,), ``theta_deg`` (6,), ``e_grid``, ``e_bins``, ``e_widths`` [GeV, kinetic].
Fluxes at the ground, shape (6, nE), dPhi/dE_kin [1/(GeV cm^2 s sr)] (E-weighting is
left to the user):

``numu``, ``antinumu``, ``nue``, ``antinue``: MCEq ``get_solution("total_<name>")``,
    i.e. the state-vector species of that name.  This is everything MCEq produces
    with this setup: conventional (pi, K, mu decays, ...) plus the prompt (charm)
    channels that SIBYLL-2.3d carries; prompt is below 1e-4 of the total at
    E < 100 GeV.  ``conv_<name>`` is ``total - pr`` (MCEq's definition) and
    ``pr_<name>`` the prompt part (tracking species ``prcas_``, ``prres_``).
``mu``: mu+ + mu-, all helicities summed (``total_mu+`` + ``total_mu-``); ``mu_plus``,
    ``mu_minus`` separately; ``conv_mu`` / ``pr_mu`` analogously.

Geometry used by MCEq for each angle ``i`` (so that the solver can be checked to use
identical depth, altitude and density): ``path_X_i`` (slant depth at the END of each
Euler step, g/cm^2), ``path_dX_i``, ``path_rho_inv_i`` (cm^3/g, at the step start),
``path_h_i`` (altitude [cm] at ``path_X_i`` from MCEq's spline ``X2h``), ``max_X``
(depth at the observation level), ``h_tab`` (altitude grid [cm]) and ``X_tab`` (6, nh)
= MCEq's ``h2X(h)`` (slant depth along the ray down to altitude h), the analytic
ingredients ``h_obs_cm``, ``h_atm_cm``, ``r_E_cm`` and the ray length ``path_len_cm``.

Usage (one core each)::

    source $N3/env.sh
    $PY tools/cascade3d/gates/mceq_reference.py PHYS     # both loss settings
    $PY tools/cascade3d/gates/mceq_reference.py REF
    $PY tools/cascade3d/gates/mceq_reference.py --summary
"""

import importlib.util  # noqa: F401
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import kernels_mceq as K  # noqa: E402

COS_Z = np.array([1.0, 0.75, 0.5, 0.25, 0.15, 0.05])
OUT_DIR = os.path.join(K.N3, "c1", "ref")
H_TAB = np.linspace(0.0, 110.0e5, 441)  # cm


def out_path(pop, loss):
    tag = "losson" if loss else "lossoff"
    return os.path.join(OUT_DIR, f"mceq_g11_{pop}_{tag}.npz")


def run_one(pop, loss):
    """Solve all six angles for one (population, loss) setting; return the dict."""
    t0 = time.time()
    run = K.make_mceq(pop, energy_loss=loss)
    from MCEq import config as cfg

    nE = run.dim
    theta = np.degrees(np.arccos(COS_Z))
    names = ["numu", "antinumu", "nue", "antinue"]
    d = {"cos_z": COS_Z, "theta_deg": theta, "e_grid": np.asarray(run.e_grid, float),
         "e_bins": np.asarray(run.e_bins, float),
         "e_widths": np.asarray(run.e_widths, float)}
    for n in names + ["mu", "mu_plus", "mu_minus"]:
        for pre in ("", "conv_", "pr_"):
            d[pre + n] = np.zeros((len(COS_Z), nE))
    d["X_tab"] = np.zeros((len(COS_Z), H_TAB.size))
    d["h_tab"] = H_TAB
    d["max_X"] = np.zeros(len(COS_Z))
    d["path_len_cm"] = np.zeros(len(COS_Z))
    d["tab_roundtrip_max_dh_cm"] = np.zeros(len(COS_Z))
    steps = []
    for i, th in enumerate(theta):
        run.set_theta_deg(float(th))
        run.solve()
        dm = run.density_model
        nsteps, dX, rho_inv, _ = run.integration_path
        X = np.cumsum(dX)
        d[f"path_X_{i}"] = X
        d[f"path_dX_{i}"] = np.asarray(dX, float)
        d[f"path_rho_inv_{i}"] = np.asarray(rho_inv, float)
        d[f"path_h_{i}"] = np.asarray(dm.X2h(X), float)
        d["max_X"][i] = dm.max_X
        d["path_len_cm"][i] = dm.geom.path_len(dm.thrad)
        d["X_tab"][i] = dm.h2X(H_TAB)
        d["tab_roundtrip_max_dh_cm"][i] = np.abs(
            dm.X2h(d["X_tab"][i][H_TAB < 100e5]) - H_TAB[H_TAB < 100e5]).max()
        steps.append(int(nsteps))
        for n in names:
            d[n][i] = run.get_solution(n, mag=0)
            d["conv_" + n][i] = run.get_solution("conv_" + n, mag=0)
            d["pr_" + n][i] = run.get_solution("pr_" + n, mag=0)
        for tag, sp in (("mu_plus", "mu+"), ("mu_minus", "mu-")):
            d[tag][i] = run.get_solution("total_" + sp, mag=0)
            d["conv_" + tag][i] = run.get_solution("conv_" + sp, mag=0)
            d["pr_" + tag][i] = run.get_solution("pr_" + sp, mag=0)
        for pre in ("", "conv_", "pr_"):
            d[pre + "mu"][i] = d[pre + "mu_plus"][i] + d[pre + "mu_minus"][i]
        print(f"  {pop} loss={loss} cosZ={COS_Z[i]}: {nsteps} steps, max_X "
              f"{dm.max_X:.1f}, {time.time() - t0:.0f} s", flush=True)
    geom = run.density_model.geom
    d["h_obs_cm"] = np.array(float(geom.h_obs))
    d["h_atm_cm"] = np.array(float(geom.h_atm))
    d["r_E_cm"] = np.array(float(geom.r_E))
    meta = {
        "population": pop, "energy_loss": bool(loss),
        "enable_energy_loss_in_config_at_build": bool(loss),
        "primary_floor": K.POPULATIONS[pop]["primary_floor"],
        "e_min": K.POPULATIONS[pop]["e_min"], "interaction_model": K.DEFAULT_MODEL,
        "primary": list(K.DEFAULT_PRIMARY), "db_file": os.path.basename(
            run._mceq_db.had_fname),
        "db_sha256": run.mceq3d_db.sha256,
        "density_model": [cfg.density_model[0], list(cfg.density_model[1])],
        "steps": steps, "kernel_config": cfg.kernel_config,
        "dXmax": cfg.dXmax, "X_start": cfg.X_start,
        "units": "dPhi/dE_kin [1/(GeV cm2 s sr)] on e_grid (kinetic energy)",
        "includes": "numu etc = total_<name> (conventional + prompt as MCEq "
                    "builds it); conv_ = total - pr; mu = mu+ + mu-, all helicities",
    }
    d["meta_json"] = np.array(json.dumps(meta))
    return d


def save(d, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **d)
    os.replace(tmp, path)


def summary():
    """Check that energy loss changes the muon flux; write ``loss_check.md``."""
    lines = ["# Energy-loss switch check for the G1.1 MCEq reference", "",
             "Ratio mu(loss off) / mu(loss on) (mu+ + mu-) and the same for the "
             "neutrinos, at selected energies. `enable_energy_loss` (MCEq config, "
             "read in `MatrixBuilder.construct_matrices`) is the switch.", ""]
    ok = True
    for pop in ("PHYS", "REF"):
        a = np.load(out_path(pop, True), allow_pickle=False)
        b = np.load(out_path(pop, False), allow_pickle=False)
        e = a["e_grid"]
        lines += [f"## {pop}", "", "| E_kin [GeV] | cos Z | mu off/on | numu off/on |",
                  "|---|---|---|---|"]
        for ie in [np.argmin(abs(e - x)) for x in (0.3, 1.0, 3.0, 10.0, 100.0)]:
            for ic in (0, 4):
                lines.append(
                    f"| {e[ie]:.3g} | {COS_Z[ic]} | "
                    f"{b['mu'][ic, ie] / a['mu'][ic, ie]:.4f} | "
                    f"{b['numu'][ic, ie] / a['numu'][ic, ie]:.4f} |")
        sel = (e > 0.5) & (e < 20)
        dev = np.abs(b["mu"][:, sel] / a["mu"][:, sel] - 1).max()
        devn = np.abs(b["numu"][:, sel] / a["numu"][:, sel] - 1).max()
        lines += ["", f"max |off/on - 1| for mu, 0.5-20 GeV: {dev:.3f}; for numu: "
                  f"{devn:.4f}", ""]
        ok &= dev > 0.01
    lines.append(f"Loss switch changes the muon flux by more than 1%: "
                 f"**{'YES' if ok else 'NO'}**")
    path = os.path.join(OUT_DIR, "loss_check.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if ok else 1


def main(argv):
    if argv and argv[0] == "--summary":
        return summary()
    pops = argv or ["PHYS", "REF"]
    for pop in pops:
        for loss in (True, False):
            d = run_one(pop, loss)
            save(d, out_path(pop, loss))
            print("written", out_path(pop, loss), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
