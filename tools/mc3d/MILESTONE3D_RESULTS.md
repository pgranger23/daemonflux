# `tools/mc3d` milestone 3d -- per-model primary energy variable, primary-model scan by exact re-weighting, low-energy yield deficit

Factual record, written as the work proceeds (started 2026-11-04).  Raw
outputs under `/home/pigrange/mc3d_work/runs/m3d/` (never `/tmp`); small
summaries copied to `results/m3d/`, scripts to `results/m3d/scripts/`.

## 0. Status / resume here

| step | state | where |
|---|---|---|
| task 1: per-model native energy variable (`primaries.py`, `sky3d.py`, `sky.py`), GST and GSF-groups models, generalised `reweight_factor`; tests | **done**; `test_m3d.py` (9 tests), `test_m3c.py` adapted | sec. 1 |
| task 2: exact re-weighting of the m3b pilot to H3a-total (check), GH, GST, GSF-groups x 3 composition references | **done** (6 targets, 108 trace groups) | sec. 2, `$M/sky/pilot_rw_*`, `results/m3d/ew_models.txt` |
| task 2b: same, with an approximate correction for the pool energy-guard bug of sec. 3 (`+gfix`) | **done** | sec. 2.6, `$M/sky/pilot_rw_*_gfix_tasks` |
| task 3: yield deficit | **done: mc3d bookkeeping bug (energy-conservation guard in `pool.EventPool.draw`), fixed + test**; residual ~10% | sec. 3 |
| task 4: full suite | **106 passed** | sec. 4 |

**Resume:** `source /home/pigrange/mc3d_work/runs/m3d/env.sh` (= m3c's env with
`M=$W/runs/m3d`); task-3 scripts in `$M/scripts`, task-2 scripts in
`tools/mc3d/results/m3d/scripts`, logs in `$M/logs`, text reports in `$M/reports`.

**Headline.**

1. Per-model native energy variable is now the default (`--primary-energy native`);
   H3a, GST, GSF = total, Gaisser-Honda = kinetic.  m3b (`kinetic`) and m3c
   (`total`) manifests/signatures are unchanged (checked by tests).
2. Pilot East-West amplitude, mean a1/a0 of the 12 pooled cells (Honda 0.4538):
   m3b H3a-kinetic 0.4427 (chi2/12 1.95), m3c H3a-total 0.4020 (12.85),
   **GH 0.4210 (6.37), GSF-groups 0.427-0.430 (4.2-5.1)**, GST 0.4010 (12.64).
   The primary spectrum accounts for about half of the m3c gap (+0.025 of the
   0.052); a residual -0.023..-0.027 stays.  Species fan-out is unchanged by the
   model (sec. 2.4).  With an approximate correction for the guard bug of item 3
   (+0.012 on every model, sec. 2.6): m3b 0.4563 (1.41), m3c 0.4141 (8.34),
   GH 0.4328 (3.64), **GSF-groups 0.439-0.442 (2.2-2.8)**, GST 0.4128 (8.49).
3. The "generator difference" at low energy is **an mc3d bug**: the
   energy-conservation guard of the event pool compared the sum of the kept
   final-state energies with `E_tot + 0.94 GeV`, ignoring the struck nucleus'
   fragments (2.5-3 GeV of rest mass in the kept products), and fired on 56% of
   p + air draws at 1.5 GeV, 33% at 10 GeV, 18% at 20 GeV, rescaling every
   product's energy (mean 0.87 at 1.5 GeV).  It lowers the 1D neutrino yield by
   a factor 1.46-1.35 (0.1-0.3 GeV, E_p 1.2-3.5 GeV), 1.1 at 10 GeV, 1.07 at
   40 GeV.  Fixed (`pool.GUARD_MASS`), with a test.  The old B/MCEq ratios
   (0.5 / 0.67 / 0.83 / 0.78 / 0.87 / 0.90 at E_p 1.5 / 2.5 / 4 / 6 / 10 / 20 GeV,
   0.3-1 GeV band) are reproduced and become 0.74 / 1.13 / 0.94 / 0.90 / 0.94 / 0.95.
   **Every pool-based run so far (m1 closure B, m2, m3a-c) carries the bug.**

---

## 1. Task 1 -- the energy variable of each primary model

### 1.1 What crflux does (checked against `crflux/models.py` 2.0.0)

| model | `_nucleus_flux(id, E)` | argument | native variable |
|---|---|---|---|
| `HillasGaisser2012` (H3a, H4a) | `sum_i a_i E^(-g_i-1) exp(-E / (Z R_c,i))` | "laboratory energy of nucleus" | **total** energy of the nucleus |
| `GaisserStanevTilav` (3-/4-gen) | same functional form, rigidity cutoffs per generation | same | **total** |
| `GlobalSplineFitBeta` | `p_and_n_flux(E)`: splines in `ln E` of p and n flux, "E = laboratory energy of nucleons"; `MCEqRun.set_primary_model` calls it at `e_grid + m` | total energy per nucleon | **total** |
| `GaisserHonda` | `K/A (E/A + b exp(-c sqrt(E/A)))^-alpha` | `E/A` fed to the Gaisser & Honda 2002 formula, whose argument is the **kinetic** energy per nucleon | **kinetic** |

The GH row rests on the formula's definition (kinetic energy per nucleon, with
`b exp(-c sqrt(E_k))` the low-energy modulation term) and on the numbers
below; I could not open the paper from here.  All-nucleon flux of each model in
its own variable and in the other one, over GSF at total energy
(`reports/native_table.txt`; GST uses C+O for the CNO group and has no Mg/Si
group, see 2.2):

| T/n [GeV] | 1 | 2 | 3 | 5 | 7 | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H3a, total (native) | 1.48 | 1.06 | 0.88 | 0.74 | 0.69 | 0.67 | 0.67 | 0.68 | 0.72 | 0.77 | 0.87 |
| H3a, kinetic (m3b) | 8.5 | 2.93 | 1.80 | 1.17 | 0.97 | 0.85 | 0.79 | 0.77 | 0.78 | 0.81 | 0.89 |
| GST 3-gen, total (native) | 1.27 | 0.91 | 0.76 | 0.64 | 0.61 | 0.59 | 0.59 | 0.61 | 0.64 | 0.70 | 0.80 |
| GH, kinetic (native) | 1.17 | 1.11 | 1.05 | 1.00 | 0.98 | 0.98 | 0.98 | 0.99 | 1.01 | 1.05 | 1.11 |
| GH, total | 0.54 | 0.61 | 0.64 | 0.70 | 0.74 | 0.79 | 0.84 | 0.88 | 0.93 | 0.99 | 1.08 |

This reproduces the project manager's table (H3a, GH rows; the 1% of the
H3a total-energy row is the `A_MC` vs `A_crflux` group definition, 2.1/m3c
sec. 1).  In their native variables GH and GSF agree to 2-5% at 3-30 GeV/n; H3a is
25-33% low, GST 36-41% low there.  **H3a and GST are high-energy
parametrisations used 1-2 decades below their range**; that, not only the
energy variable, is what the m3b->m3c change exposed.

### 1.2 Code

* `primaries.py`: `ENERGY_CONVENTIONS = ("kinetic", "total", "native")`,
  `ENERGY_CONVENTION = "native"`; `NATIVE_ENERGY` (by model name),
  `native_for_name`, `native_convention(model)` (checks the crflux class; an
  unknown model raises -- the user must pass `kinetic|total`),
  `resolve_convention`; `crflux_energy(..., model=)` and
  `nucleon_intensity` resolve "native" per model.  New models: `GST` (3-gen),
  `GST4`, `GSFG[:REF]` (= `GSFGroups`, sec. 2.2).  `group_split` /
  `reweight_factor` (the exact weight ratio between two (model, variable)
  pairs, used by the replay).  GST's CNO group = C-12 + O-16 each at its own A;
  its Mg/Si group is empty (crflux's GST has none).  `GSFNucleons`
  (nucleon-level, no cutoff) now evaluates at its native variable too.
* `sky3d.py`: `Options(primary_model=...)` resolves `native` to `kinetic` /
  `total` **at construction**, so `signature()` and the manifest are what they
  were: H3a default (`native` = `total`) -> the m3c 12-entry signature; GH
  `native` = `kinetic` -> the m3b 9-entry one; `--primary-energy kinetic|total`
  still overrides.  `run()` re-resolves with the actual `--model`; the module
  default is no longer written by `run()` in the parent process (workers set it in
  `_init`, as before).  `--primary-energy native` is the CLI default.
* `sky.py`: `primary_sample` takes the p/n split of a `group_split` model
  (GSF-groups) from the model instead of `Z/A`; the group-model branch (the
  only one the pilot used) is bit-for-bit unchanged.
* Tests (`test_m3d.py`): native variable per model and per name; "native"
  needs the model; GST CNO/MgSi mapping; GSF-groups reproduce GSF's p and n
  flux to 1e-12 for three composition references and the group sampler uses
  the split; `reweight_factor` equals the ratio of `primary_sample` weights of
  two seeded runs (exactness of the replay), identity handling; Options
  signatures (H3a -> m3c's, GH -> m3b's, forced `total`); the draw order the
  replay regenerates; the energy-guard fix (sec. 3.5).  `test_m3c.py`'s
  convention test was adapted (default is `native`; signature equalities).

---

## 2. Task 2 -- primary-model scan by exact re-weighting of the m3b pilot

### 2.1 Method

`results/m3d/scripts/reweight_replay.py SRC TAG NPROC TARGET...` (generalises
m3c's H3a-only script).  The source is `$W/runs/m3b/sky`, tag `pilot` (5641
c-files, 1.44e6 primaries, H3a at `kinetic`, read from the manifest).  The
weight of a primary is linear in the primary flux, so every entry of a primary is
multiplied by `f = J_new(group, T [, identity]) / J_old(...)`.  `T` and the
nucleon identity of each pending primary are regenerated from the c-file's seed
(same two vectorised draws as `sky.primary_sample`, tested) and matched to the
pending list by the stored rigidity (exact float equality, asserted).  The
cutoff verdicts of the traced copies are recomputed once per group of files
(shared by all targets); tallies are replayed with `sky3d.tally_entries`.
Checks: `--check` (old model as target, f = 1) reproduces the original t-files
to 1e-12 on 7 files; the H3a-total target reproduces m3c's `pilot_tot`
**digit for digit** (`ew_models.txt`, rows `m3c_H3a_tot` and `replay_H3a_tot`).

### 2.2 The per-group GSF (approximation)

crflux's `GlobalSplineFitBeta` file (`GSF_spline_20171007.pkl.bz2`) contains
three splines and nothing else: the proton fraction, the proton flux and the
neutron flux versus `ln E` (knots 1 GeV to 1e12 GeV).  The published fit's p /
He / O / Fe splines are **not in it and not installed anywhere** I could find
(crflux 2.0.0 in the venv, MCEq 1.4.1, nothing under `$W`; nothing was
installed from the internet).  `primaries.GSFGroups` therefore keeps the
nucleon-level GSF exact and takes only the *split into groups* from a
reference group model `ref` at the same kinetic energy per nucleon:
`N_{j,x}(T) = J^GSF_x(T) * g_{j|x}(T)`, `x` = p or n, `g_{j|x}` = the share of
nucleon type `x` that `ref` puts in group `j`.  Summed over groups it
reproduces GSF's proton and neutron fluxes to 1e-16 (tested); neutrons exist
only in `A/Z ~ 2` groups, protons in the p group and bound in nuclei.  What
the split changes in the MC: the rigidity `(A/Z) p` of the cutoff test and the
p/n identity of the projectile.  Group shares of the nucleon flux with
`ref` = H3a (T = 3 / 10 / 30 GeV/n): p 0.72 / 0.74 / 0.73, He 0.19, CNO
0.04-0.05, MgSi 0.02, Fe 0.02.

**Effect of the approximation**, estimated by changing `ref` between three
group models with different compositions (H3a, GST -- p 0.75, He 0.22-0.23, no
MgSi, Fe 0.004-0.007 --, GH -- p 0.72-0.75, He 0.18-0.21, MgSi 0.02-0.03):
mean a1/a0 0.4269 / 0.4300 / 0.4304, i.e. a **spread of 0.0035 (0.8%)**, a
fraction of one paired jackknife error per cell (0.003-0.005); chi2/12 5.05 /
4.20 / 4.15.  The three references are all five-group fits to the same
low-energy data, so they bracket the composition ambiguity only loosely: GSF's
own He/CNO/Fe shares at 3-30 GeV/n are not known to me to better than that.

### 2.3 Results

12 pooled cells (4 species x 3 bands 0.3 / 0.5 / 1 GeV, cos Z 0-0.3, largest
cap, tag `s`), jackknife G = 20, chi2/12 against Honda with MC errors only.
Full per-cell tables with errors: `results/m3d/ew_models.txt`.

| model (energy variable) | <a1/a0> | <dphi> [deg] | <fN> | <fS> | <N/S> | chi2/12 a1/a0 | dphi | fN | fS | N/S | 3D flux / m3b (0.3 / 0.5 / 1 GeV) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Honda HKKM14 `kam-ally-20-12-solmin` | 0.4538 | -0.10 | 0.932 | 1.105 | 0.846 | | | | | | |
| **m3b**: H3a (kinetic) | 0.4427 | -2.42 | 0.894 | 1.069 | 0.838 | 1.95 | 1.67 | 2.60 | 1.93 | 0.75 | 1 |
| **m3c**: H3a (total) | 0.4020 | -3.16 | 0.909 | 1.071 | 0.850 | 12.85 | 2.70 | 2.14 | 1.91 | 0.94 | 0.83 / 0.86 / 0.89 |
| GST 3-gen (total) | 0.4010 | -2.90 | 0.910 | 1.071 | 0.851 | 12.64 | 2.51 | 2.12 | 1.80 | 0.88 | 0.73 / 0.75 / 0.79 |
| **GH (kinetic)** | 0.4210 | -3.20 | 0.903 | 1.075 | 0.841 | 6.37 | 2.78 | 2.56 | 1.60 | 1.13 | 1.15 / 1.18 / 1.21 |
| **GSF-groups, ref H3a (total)** | 0.4269 | -3.12 | 0.903 | 1.077 | 0.839 | 5.05 | 2.96 | 2.37 | 1.37 | 1.12 | 1.16 / 1.18 / 1.20 |
| GSF-groups, ref GST | 0.4300 | -2.90 | 0.903 | 1.079 | 0.838 | 4.20 | 2.65 | 2.39 | 1.13 | 1.04 | 1.15 / 1.17 / 1.19 |
| GSF-groups, ref GH | 0.4304 | -3.19 | 0.901 | 1.078 | 0.837 | 4.15 | 2.82 | 2.64 | 1.36 | 1.19 | 1.15 / 1.18 / 1.19 |

(`replay_H3a_tot` = `m3c_H3a_tot` exactly; not repeated.)  a1/a0 per cell
(nue / antinue / numu / antinumu at 0.3, 0.5, 1 GeV: 3 values each):

| | nue | antinue | numu | antinumu |
|---|---|---|---|---|
| Honda | 0.583 0.643 0.615 | 0.315 0.310 0.226 | 0.400 0.410 0.332 | 0.525 0.569 0.518 |
| m3b | 0.602 0.620 0.607 | 0.299 0.273 0.167 | 0.386 0.360 0.308 | 0.531 0.577 0.584 |
| m3c | 0.559 0.576 0.575 | 0.256 0.237 0.129 | 0.342 0.314 0.271 | 0.489 0.534 0.543 |
| GH | 0.570 0.594 0.598 | 0.272 0.253 0.153 | 0.357 0.335 0.294 | 0.502 0.553 0.571 |
| GSF-groups (H3a ref) | 0.573 0.597 0.603 | 0.276 0.260 0.163 | 0.362 0.341 0.303 | 0.506 0.559 0.579 |
| GST | 0.558 0.573 0.573 | 0.253 0.243 0.133 | 0.337 0.309 0.271 | 0.485 0.527 0.550 |

### 2.4 Reading

* **The primary spectrum moves a1/a0 by +0.019 (GH) to +0.025..+0.028 (GSF) relative
  to H3a-total**, i.e. about half of the 0.052 gap that m3c opened against
  Honda; the paired shift vs m3b is -0.022 (GH) and -0.012..-0.016 (GSF).  The
  hypothesis "the residual gap to Honda is largely the H3a primary spectrum"
  is **half confirmed**: GSF/GH remove ~half, **a1/a0 stays 0.023-0.027 (GSF) /
  0.033 (GH) below Honda, chi2/12 4-5 (GSF), 6.4 (GH)**, with unchanged
  sign in 11 of 12 cells (nue 0.3 and 1 GeV within 0.012 of Honda; antinumu
  1 GeV, +-0.1, is above it).  GST is as bad as H3a-total (0.4010 vs 0.4020): its 3-gen
  fit is 40% below GSF at 3-30 GeV/n.
* **Why**: the amplitude is a ratio, so a harder / softer spectrum changes it
  through the mix of primary energies sampled by the cutoff, not through the
  normalisation (the 3D flux changes by up to +-20%).  GSF and GH are +15-21%
  in flux relative to m3b (H3a kinetic) at 0.3-1 GeV while their a1/a0 differ
  from m3b by -0.012..-0.022 (H3a-total: -0.041 at -11..-17% flux): amplitude
  and normalisation are not tied together.  I did not decompose which energy
  range of primaries drives the amplitude change.
* **Species fan-out is not changed by the model**: antinumu-numu at 0.3 / 0.5 /
  1 GeV is +0.147/+0.219/+0.271 (H3a total), +0.145/+0.218/+0.276 (GH),
  +0.144/+0.219/+0.277 (GSF-groups); Honda +0.125/+0.158/+0.186 (the MC's
  fan-out is too large by 0.02-0.09, growing with energy, **as in m3c**).
  nue-antinumu: +0.07/+0.04/+0.03 (all models) vs Honda +0.059/+0.075/+0.097.
  The residual is therefore **not removed by the primary model**; the candidates
  remain those of m3b/m3c (the muon-bend / charge fan-out pattern).  The GSF
  compositions differ in fan-out by <= 0.01.
* `<N/S>`, `<fS>`, `<fN>`: unchanged by the model to 0.01 (N/S: 0.837-0.851
  vs Honda 0.846).  `dphi`: -2.9..-3.2 deg for all models vs Honda -0.1: the
  3 deg phase offset is not a primary-model effect either.

### 2.5 Re-weighting diagnostics per stratum

(`results/m3d/rw_diag.txt`, `rw_diag.py` on `rw_diag_run.py`'s output: the same
per-primary records as the replay, from the c-files.  Per (group, stratum):
`f` range over the pending primaries, `F/S` = the stratum's new / old
contribution, effective sample size of the primaries' no-cutoff 3D weights
before and after, `ESS = (sum c)^2 / sum c^2`.  The strata are 7 half-decades
of kinetic energy per nucleon, 1 GeV - 3.2 TeV, for each of the 5 groups.)

* **Weight range**: `f` is a smooth deterministic function of `T`, so inside a
  stratum it varies by the model's local slope ratio, typically a factor
  1.1-2: GH, p 3.16-10 GeV/n 0.60-1.18, 10-31.6 GeV/n 1.18-1.35, 31.6-100
  1.29-1.35; GSF (H3a composition), p 0.56-1.16 / 1.16-1.30 / 1.13-1.29,
  He 0.57-1.25 / 1.17-1.31; GST, p 0.46-0.73 / 0.73-0.87 / 0.87-0.93; H3a-total
  0.50-0.79 / 0.79-0.93 / 0.93-0.98.  Largest ranges: the lowest stratum of
  every group (He, CNO, MgSi, Fe at 1-3.16 GeV/n: f 0.03-1.0, e.g. GSF-GST
  CNO 0.029-0.30), and the heavy groups where a reference model puts a
  different share (GH's MgSi at 10-31.6 GeV/n f = 1.7-2.0, 547 pending
  primaries).
* **Effective sample size**: ESS(new)/ESS(old) = 0.85-1.06 in all 196
  populated (target, group, stratum) cells (GST has no Mg/Si group); in the 49
  cells that carry >= 2% of their target's weight the lowest is 0.85 (He
  1-3.16 GeV/n of GSF-GST, share 2.6%, ESS 4300) and ESS(new) >= 1650 in all of
  them.  No cell meets my "poor" criterion (share >= 2% and ESS(new) < 0.5
  ESS(old) or < 30); `rw_diag.py` flags none.  The worst ratios are the
  lowest strata of He / CNO / Fe / MgSi (0.85-0.90, all with share < 3%).
* **Poorly covered, and not hidden**: the pilot's Neyman allocation (tuned
  for H3a-kinetic's 3D amplitudes) sampled the **p 1-3.16 GeV/n stratum**
  with 2850 primaries of which only **47 are pending** (reach a cap; ESS
  33-35); its weight is 0.9-1.3% of the total in every model.  Everything
  the models do below 3 GeV/n (H3a and GSF differ by x 1.5 there) rests on
  those 47 primaries and on the He/CNO strata (ESS 1300-5000); m3c showed the
  stratum has no 3D entries in the 0.3-1 GeV bands, so it does not enter the
  amplitudes above, but it enters any absolute comparison of the 3D flux
  (e.g. at the lowest energy bins) with a large Poisson error.  Likewise
  the strata above 1 TeV/n (ESS 26-36) carry <= 0.3% each.
* The re-weighted results are **not independent samples**: all targets re-use
  the 1.44e6 primaries, so differences between targets are paired (small
  jackknife errors, 0.003-0.005 per cell) while each absolute number keeps
  the pilot's own 0.01-0.05 per-cell error.

### 2.6 With the pool energy-guard bug corrected (approximate)

The pool energy-guard bug (sec. 3.5) lowered the neutrino yield most at the
lowest primary energies, which are exactly the primaries below the cutoff that
make the East-West amplitude.  A full re-run at production size is outside
this task; as an **approximation** the pilot is re-weighted entry by entry with
`g(T_primary, E_nu)` (`+gfix` targets of `reweight_replay.py`): the ratio of
the 1D yield with the fixed guard to the old one, from the vertical collinear
closure column at 14 primary energies 1.2-300 GeV (`build_gfix.py`, `gfix.txt`;
four flavours summed, three wide E_nu bands 0.1-0.3 / 0.3-1 / 1-3 GeV, smoothed over E_p; 8000-12000
showers per point below 10 GeV, 2000-4000 above).  It assumes the effect is a
yield factor that depends on the primary energy and the neutrino energy but not
on the arrival direction, which is first-order true (the guard rescales
magnitudes, not directions) and untested at second order; I take **+-30% of the
shift** as its uncertainty.  Same 12 cells (`results/m3d/ew_models_gfix.txt`,
`ew_summary_gfix.txt`):

| model | <a1/a0> without -> with | shift | chi2/12 a1/a0 without -> with |
|---|---|---|---|
| Honda | 0.4538 | | |
| m3b: H3a (kinetic) | 0.4427 -> 0.4563 | +0.0136 | 1.95 -> 1.41 |
| m3c: H3a (total) | 0.4020 -> 0.4141 | +0.0121 | 12.85 -> 8.34 |
| GST (total) | 0.4010 -> 0.4128 | +0.0118 | 12.64 -> 8.49 |
| GH (kinetic) | 0.4210 -> 0.4328 | +0.0118 | 6.37 -> 3.64 |
| **GSF-groups, ref H3a** | 0.4269 -> 0.4386 | +0.0117 | 5.05 -> 2.79 |
| GSF-groups, ref GST | 0.4300 -> 0.4416 | +0.0116 | 4.20 -> 2.32 |
| GSF-groups, ref GH | 0.4304 -> 0.4422 | +0.0118 | 4.15 -> 2.20 |

The correction raises every a1/a0 by +0.012 (nearly model independent), the
3D flux ratio between models is unchanged (GSF/m3b 1.15-1.20), the fan-out is
unchanged (antinumu-numu +0.144/+0.218/+0.279 for GSF-groups(H3a); Honda
+0.125/+0.158/+0.186), `<N/S>` 0.833-0.848 (Honda 0.846), `dphi` -2.2..-3.0
deg.  **With both corrections (GSF primaries, fixed guard) the pilot's mean
a1/a0 is 0.439-0.442 against Honda's 0.454: -0.012..-0.015, chi2/12 2.2-2.8.**
m3b's agreement with Honda (0.4563, chi2/12 1.41) was a cancellation of the H3a
energy-variable error and the guard bug, as m3c suspected for the vertical.

---

## 3. Task 3 -- is the low-energy yield deficit the generator or the bookkeeping?

**Verdict.**  It is **mc3d bookkeeping: the energy-conservation guard in
`pool.EventPool.draw`**.  After fixing it the MC's 1D no-cutoff yield is
within 5-15% of MCEq's at E_p = 1.2-4 GeV (0.82-0.94 at 0.1-0.3 GeV,
0.74-1.13 +- 0.05-0.1 at 0.3-1 GeV, noisy) instead of 0.5-0.7; the rest is
decomposed below into a decay-module term (-9% at 0.1-0.3, ~0 at 0.3-1 GeV),
a closure/MCEq-bookkeeping term (-5%) and a small generator term (-3 +- 3%).
The re-interaction floor, the Sigma drop and the per-interaction pion/kaon
yields are not the cause.

### 3.1 Per-interaction yields, pool vs table (not the cause)

`$M/scripts/yield_cmp.py` (`results/m3d/yield_cmp.txt`): raw DPMJET-III-19.3 pool
events at their bin energy against the MC's own MCEq tables (v140), at matched
projectile kinetic energy (table interpolated log-linearly between its two
columns around the pool's `E_kin`; the sampler's own within-bin shape `E^-g`),
p + air at E_kin = 1.3, 1.9, 2.6, 3.5, 4.7, 6.2, 8.0, 10.3, 13.3, 16.9, 21.6,
27.4 GeV and pi+- + air at 2.1-28 GeV, 2000 events per pool.  Yield per
interaction above a total-energy threshold, pool/table:

* **charged pions, all energies above 0.2 GeV**: 0.98-1.08 (p+air, 1.3-4.7 GeV, the 1.08
  is pi- at 1.3 GeV, 0.18 per interaction; 1.09/1.14 at 6.2 GeV, 1.01/1.05 at 8),
  0.97-1.00 at >= 10 GeV; pi+ + air 0.97-1.04 above 2 GeV.
* **pions above 0.7 GeV** (the ones that can make a 0.3-1 GeV neutrino
  directly, `E_nu <= 0.427 E_pi`): p + air T = 2.6: 1.13 / 0.70 (pi+ / pi-,
  yields 0.064 / 0.021 +- 0.003 per interaction: statistics), T = 3.5: 0.98 /
  0.98, 4.7: 0.97 / 0.98, 6.2: 0.94 / 0.93, 8: 0.90 / 0.95, >= 10: 0.94-0.98.
  So -3 to -10% at 6-20 GeV, consistent with the 5% soft deficit of
  MILESTONE2 sec. 2.6, parity below.
* **kaons**: K+ 0.8-1.3 (yield 0.001-0.2), too few events below 5 GeV.
* **nucleons**: pool +14-20% in total (slow target fragments, as in
  MILESTONE2), 0.97-1.05 above 1 GeV total, 0.9-1.0 above 2 GeV.

Per-interaction yields therefore differ by <= 5% in the neutrino-relevant
range, with noise of 10-30% in the near-threshold bins.  A factor two cannot
come from here.

### 3.2 The re-interaction floor (not the cause)

`pool.draw` returns nothing for a hadron below `E_MIN_POOL = 2 GeV` total
(`lambda_int` = inf there), so the MC never re-interacts sub-2-GeV
nucleons/pions, whereas MCEq does.  (Note the docstring of
`EventPool.lambda_int` is wrong: a nucleon of 2 GeV total energy has 1.06 GeV
kinetic energy, not 0.23, and can make pions.)  Quantified with the MCEq tables
as the generator and the floor imposed on them (`closure_floor.py kinematic
2.0`; same seeds as the no-floor run, 24000 showers, collinear vertical column,
all four flavours): MC/MCEq, floor 2 GeV vs none, at E_p = 4 / 6 / 10 GeV:
0.1-0.3 GeV 0.899 / 0.931 / 0.941 vs 0.896 / 0.931 / 0.941; 0.3-1 GeV 0.950 /
0.989 / 0.990 vs 0.953 / 0.981 / 0.992 (+-0.006-0.013): **the floor costs <
1%**; at E_p = 1.2-2.5 GeV the two are identical (no sub-2-GeV
secondary makes a neutrino above 0.1 GeV).

### 3.3 Sigma+- (not the cause; still not fixed)

`pool.KEEP` still has no 3112/3222 (item (i) of MILESTONE2 is **not fixed**).
Measured in raw chromo DPMJET events (`sigma_yield.py`, 3000 events per
energy, Sigma -> N pi isotropic in the Sigma frame): Sigma+- per p+air
interaction 0 (E_tot 3.6 GeV), 0.009 (5.6), 0.023 (7.1), 0.047 (11.3),
0.043 (17.9); their decay nucleons above 2 GeV: 0, 0.003, 0.009, 0.016, 0.025
per interaction (1-3% of the 0.5-0.8 nucleons above 2 GeV that the pool makes),
decay pi+- above 0.7 GeV 0, 0, 0.002, 0.005, 0.011 (<= 1% of the pions above
0.7 GeV).  Worth < 1% of the neutrino yield at E_p < 7 GeV and ~1-2% at
10-20 GeV.  Not fixed here (it needs new pools); the number is small.

### 3.4 The decomposition and the guard (the cause)

The milestone-1 closure ladder, extended to low energy
(`closure_floor.py`; vertical collinear column, four flavours summed,
MC/MCEq 1D; `results/m3d/ladder*.txt`, errors are statistical):

| rung | meaning |
|---|---|
| A1 | MCEq tables + MCEq decay tables (isolates transport / atmosphere / dE/dx / primary representation) |
| A2f2 | MCEq tables + full decay kinematics + the 2 GeV floor |
| B | chromo DPMJET pools (the MC's backend) + kinematics + floor; "B old" = the guard of milestones 1-3c |

0.1-0.3 GeV band (E_p kinetic, GeV):

| rung | 1.2 | 1.5 | 1.8 | 2.2 | 2.5 | 3 | 3.5 | 4 | 6 | 10 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | 0.96 | 0.98 | 0.95 | 0.85 | 0.89 | 0.98 | 0.96 | 0.99 | 0.98 | 0.99 | 1.00 |
| A2f2 | 0.87 | 0.91 | 0.87 | 0.78 | 0.80 | 0.89 | 0.88 | 0.90 | 0.94 | 0.95 | 0.98 |
| **B old** | | 0.61 | | | 0.59 | | | 0.70 | 0.85 | 0.83 | 0.85 |
| **B fixed** | 0.82 | 0.82 | 0.82 | 0.80 | 0.82 | 0.87 | 0.88 | 0.88 | 0.94 | 0.90 | 0.92 |

0.3-1 GeV band (statistical errors 0.02-0.1 below 4 GeV, the MCEq yield there
is 0.003-0.25 per primary):

| rung | 1.2 | 1.5 | 1.8 | 2.2 | 2.5 | 3 | 3.5 | 4 | 6 | 10 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | 0.74 | 0.90 | 0.93 | 0.90 | 0.89 | 0.96 | 0.95 | 0.95 | 0.99 | 1.01 | 1.00 |
| A2f2 | 0.70 | 0.96 | 0.88 | 0.87 | 0.96 | 0.95 | 0.95 | 0.92 | 0.96 | 0.99 | 0.99 |
| **B old** | | 0.50 | | | 0.67 | | | 0.83 | 0.78 | 0.87 | 0.90 |
| **B fixed** | 0.57 | 0.74 | 0.79 | 0.98 | 1.13 | 1.01 | 0.94 | 0.94 | 0.90 | 0.94 | 0.95 |

("B old" at 1.5-20 GeV = the numbers of the m3c sec. 1 table, 0.5 / 0.75-0.8 /
0.9, reproduced: that table's "generator difference" is this.)  Read as a
product at 0.1-0.3 GeV, E_p 1.2-4 GeV: MCEq 1 x **A1 0.95 +- 0.04** (the
dip at 2.2-2.5 GeV is MCEq's 3-bin primary representation / table grid) x
**A2f2/A1 0.91** (decay kinematics + floor; ~1.0 at 0.3-1 GeV) x **B/A2f2 0.97
+- 0.03** (generator, pool resampling) = 0.84, the fixed B.  At E_p >= 6: 0.9-0.95.

### 3.5 The cause: `EventPool.draw`'s energy guard

`draw` rescales `p_z` of a pool event by `p_beam(E)/p_beam(E_0)` and then
guards against creating energy: if `sum E_i > E_tot + 0.94` it multiplies
**every** product's energy and momentum by `(E_tot + 0.94)/sum E_i`.  The
budget `E_tot + 0.94` is one nucleon mass.  But the kept final state of
p + N-14/O-16 contains the target's evaporation / fragmentation nucleons: on
average 2.5-3.1 GeV of rest mass at **every** energy (`guard_check.txt`,
"slow nucleon rest-mass share").  For E_tot = 2.2-3.6 GeV the raw, unrescaled
events exceed the budget in 47-50% of cases (mean scale 0.86-0.89), at 11 GeV
28%, at 45 GeV 7%.  At the MC's own energies (`guard_check.txt`): the guard
fires on 56% of draws at E_kin = 1.5 GeV (mean scale 0.87, minimum 0.31), 53%
at 2.5, 46% at 4, 40% at 6, 33% at 10, 18% at 20, 9% at 40 GeV.  The docstring
said it "never fires" -- true for the 20-bins-per-decade SIBYLL pools it was
written for, false for the 10-per-decade DPMJET pools below 80 GeV and for
the slow fragments that DPMJET adds.

First-generation products with the MC's own sampler, p + air, charged pions
above 0.2 GeV, pool/table: **old guard 0.75 (E_p = 1.5), 0.91, 0.88 (4), 1.12
(6)** [`first_gen_oldguard.txt`] -> **fixed 0.96, 1.06, 0.94, 1.20**
[`first_gen.txt`] (the 6 GeV excess is the table, 10-20% below the pool's raw
yield there, sec. 3.1).

**Fix**: `pool.GUARD_MASS = 16 * 0.931494` (the heaviest air nucleus) replaces
the single nucleon mass in `avail = e_tot + GUARD_MASS`; the guard still fires
for a pathological rescale.  Test:
`test_m3d.py::test_energy_guard_budget_includes_the_struck_nucleus` (a
physical low-energy event is returned unscaled; a pathological one is scaled to
the budget).  Setting `pool.GUARD_MASS = 0.94` reproduces milestones 1-3c.
**Cost of the old guard**: 1D yield ratio fixed/old, four flavours, per
neutrino-energy band (`gfix.txt`; errors 1-3% at 0.1-0.3 GeV, 4-12% at 1-3):

| E_p [GeV] | 1.2 | 1.5 | 1.8 | 2.5 | 3.5 | 4 | 5 | 6 | 7 | 10 | 20 | 40 | 100 | 300 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1-0.3 GeV | 1.46 | 1.34 | 1.26 | 1.38 | 1.30 | 1.26 | 1.17 | 1.11 | 1.14 | 1.09 | 1.09 | 1.07 | 1.05 | 1.07 |
| 0.3-1 GeV | | 1.49 | 1.25 | 1.59 | 1.32 | 1.14 | 1.14 | 1.14 | 1.12 | 1.08 | 1.06 | 1.03 | 1.01 | 1.01 |

i.e. **every pool-based neutrino yield so far was 5-45% too low, most at the
lowest primary energies**; it affects absolute flux and the weighting of
primary energies, not the angular pattern.  Not yet re-run at production size
(rule: no new production runs); sec. 2.6 propagates it approximately into the
pilot.

**What I did not resolve.**  (a) The ~5-10% remaining pool/table difference
near 6-8 GeV and the 6-20% excess of pool over table at E_p = 6 GeV in the
first-generation test is a feature of the table (MCEq's DPMJET configuration;
the slow-nucleon excess of the pool is the same effect), at the per-interaction
level +-10%; it is not the origin of the deficit.  (b) The dip of A1/MCEq at
E_p = 2.2-2.5 GeV (0.85-0.89, 0.1-0.3 GeV band) is a closure artefact of
MCEq's 3-bin primary representation or table grid; not investigated further.
(c) The decay module (A2f2/A1 = 0.91 at 0.1-0.3 GeV) is a milestone-1 rung-A2
result carried to low E_p; it was not examined here.

---

## 4. Full test suite, CPU

Full suite from `$M/work/pytest` (not the repo):
`$PY -m pytest $R/tools/mc3d -q`: **106 passed**, 11 warnings, 8 min 52 s under
load (`results/m3d/pytest_final.log`) = 97 (m3c) + 9 in `test_m3d.py`
(the 8 before the last one passed in an earlier full run, 105 passed).
`test_m3c.py::test_primary_energy_convention_total_is_mceqs` was edited (the default is `native`).

**CPU** (estimates; the machine was oversubscribed -- load 15-70 on 48 cores --
so wall times are not informative; the 24-core cap was respected: <= 16 replay
workers + 8 closure workers + 1-2 single-process jobs at any time):
task 3: closure showers 1.7 core-h (wall x 8 workers, from the logs) + ~45
MCEq reference solves (~0.7) + Sigma generation (0.25) + per-interaction
tests (0.3) = ~3.0 core-h; task 2: two replays of 5641 c-files (the
vectorised tracer dominates: 7e5 copy states at ~2-4 ms) ~2.5 core-h, 13
`extract_strata` jobs ~2.5 core-h, one aborted timing run, merges and diagnostics
~0.7 core-h; test suites 0.3 core-h.  Total ~9 core-h.  (The merged `_total.npz`
of the re-weighted runs were not written -- `extract_strata` reads the task
files; the replay script's own merge is slow and was stopped.)

Run directories: `$W/runs/m3d/sky/pilot_rw_*_tasks` (13 x ~0.9 GB: delete
after use if space is needed), `$W/runs/m3d/work/strata_*.npz`,
`reports/`.  Repo: `results/m3d/` (144 kB of text), no git commit made.

## 5. Open questions / what I would do next

1. **The per-group GSF is an approximation** (sec. 2.2): the p / He / O / Fe
   splines of the published fit are not in crflux's `GSF_spline_20171007`; a
   proper split needs the authors' tables.  The composition ambiguity moves
   a1/a0 by 0.004 (0.8%).
2. **All pool-based results to date carry the guard bug** (m1 closure B, m2
   rung B and 3D, m3a-c).  Absolute neutrino yields were 5-45% too low, most at
   the lowest primary energies; the 1D normalisation of every closure /
   engine comparison with a pool should be re-read.  The fixed code is in
   place; the next production run uses it.  The `+gfix` numbers are an
   approximation (+-30% of the 0.012 shift).
3. After both corrections the pilot is still 0.012-0.015 below Honda in mean
   a1/a0 (chi2/12 2.2-2.8 with MC errors only) and the species fan-out
   (antinumu-numu +0.02..+0.09 too large) and the 3 deg phase offset are not
   touched by the primary model or the guard.  Remaining candidates are
   outside this milestone: the muon-bend / charge fan-out (m3c sec. 2), the
   magnetic-field model, Honda's own primaries (HKKM's GH-based spectrum with
   solar-minimum modulation: the GH row above is the closest to it, and GH
   alone leaves -0.033 / -0.021 with the guard fix).
4. H3a and GST are used here 1-2 decades below their intended range (25-40%
   low vs GSF/GH at 3-30 GeV/n); any `tools/mc3d` result with them at
   < 100 GeV/n should say so.  `GaisserHonda` is native-kinetic by my reading
   of the formula, not by the paper in hand.
5. The decay module is 9% below MCEq's decay matrices at 0.1-0.3 GeV (A2f2/A1)
   at all E_p tested, and A1/MCEq dips to 0.85-0.89 at E_p = 2.2-2.5 GeV:
   milestone-1 rung-A items not touched here.
6. Sigma+- still not in `pool.KEEP` (<= 1-2% effect); the docstring of
   `EventPool.lambda_int` about 2 GeV nucleons is wrong (not changed).
7. The pilot samples p 1-3.16 GeV/n with 47 pending primaries; the next
   allocation should cover it (m3c sec. 1 said the same for the 1D `t`).

