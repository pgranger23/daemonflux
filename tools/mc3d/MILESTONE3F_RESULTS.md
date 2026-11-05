# `tools/mc3d` milestone 3f -- engine closure, the East bookkeeping offset, solar modulation, Honda noise, production design

Factual record, written as the work proceeds (started 2026-11-05).  Raw outputs under
`/home/pigrange/mc3d_work/runs/m3f/`; summaries and scripts in `results/m3f/`.

## 0. Status / resume here

| step | state |
|---|---|
| 1. engine-MC closure, muon field off (GSF, GH-native); `mb1` vs pilot-2 | **done** (sec. 1) |
| 2. East 60-90 bookkeeping offset | **diagnosed, not a bug in the tally; physical definition mismatch** (sec. 2) |
| 3a. Honda solmax | not reachable by the agent; **done by PM 2026-11-05** (sec. 8.1) |
| 3b. force-field re-weighting of pilot-2 | **done** (5 phi values, sec. 3.2) |
| 4. Honda statistical noise | **done** (sec. 4) |
| 5. production design | **done** (sec. 5; options table) |

**Resume:** `source /home/pigrange/mc3d_work/runs/m3f/env.sh`; scripts `$M/scripts` and `results/m3f/scripts`.

## 1. Engine-MC closure, muon field off

Companion (`comp`, 0.8e6 primaries, `--no-muon-field`, GSF-groups GH composition) against the
rigidity-fixed engine (`rigfix/out/m3b_engine_<gsf|ghnative>_mb0.npz`; band-integrated like Honda by
`analyse_m3b.engine_patterns`), 12 pooled cells, errors = MC jackknife only
(`engine_compare.py`; `results/m3f/engine_compare_comp_gsf.txt`, `..._ghnative.txt`).
MC minus engine, GSF engine (GH-native in brackets):

| observable | mean (12 cells) | chi2/12 | by species chi2/3 (nue, antinue, numu, antinumu) |
|---|---|---|---|
| a1/a0 | **+0.012 +- 0.010** (+0.021 +- 0.010) | **1.11** (1.38) | 4.4, 3.4, 4.0, 1.6 |
| dphi [deg] | -2.0 +- 1.2 (-2.1) | 2.28 (2.36) | 1.9, 4.7, 6.4, 14.3 |
| fN | **-0.043 +- 0.012** (-0.042) | 3.13 (3.07) | 3.1, 17.9, 6.5, 10.1 |
| fS | -0.024 +- 0.016 (-0.019) | 1.74 (1.48) | 1.6, 5.0, 8.2, 6.1 |
| N/S | -0.020 +- 0.017 (-0.023) | 1.89 (1.94) | 1.4, 10.2, 4.5, 6.6 |

* **The amplitude closes**: a1/a0 agrees with the fixed engine to +0.012 +- 0.010, chi2/12 = 1.11 (M3b: 1.03 with
  three bugs; here none of them is present).  The GSF engine is the like-for-like one; GH-native is +0.009 further
  from the MC.
* **The azimuthal shape does not fully close**: the MC has a 4% lower north fraction (fN -0.043 +- 0.012, 3.6 sigma;
  chi2/12 3.1) and -2 deg phase relative to the engine, in all four species; fS -0.024, N/S -0.020 (1.2 sigma).  The
  N/S and fN offsets have the same sign as the East-bin offset of sec. 2 (the MC's near-horizon cutoffs differ from the
  ground-anchored map): a candidate common cause, not shown.
* Fan-out with the field off: MC antinumu-numu +0.031 / -0.057 / +0.048 (+-0.04), engine ~0 (-0.003 / -0.001 / +0.002);
  nue-antinue MC +0.001 / +0.026 / +0.078, engine +0.022 / +0.033 / +0.048.  Consistent within the MC errors.
* **`mb1` (field on) vs pilot-2** (GSF engine, `engine_compare_pilot2_gsf.txt`; GH-native in brackets): pooled a1/a0
  MC - engine **+0.005 +- 0.007** (+0.014) but chi2/12 = **29** (29): by species nue +0.148 +- 0.016, antinue -0.156,
  numu -0.076, antinumu +0.105 (each chi2/3 70-100) -- the engine's `muon_bending=True` produces almost none of the
  fan-out (engine antinumu-numu +0.010 / +0.018 / +0.022, nue-antinue +0.05 / +0.07 / +0.09 against the MC's
  +0.168 / +0.167 / +0.259 and +0.264 / +0.348 / +0.509); fN -0.040 +- 0.010 (chi2/12 7.5), fS -0.042 (5.3),
  N/S -0.005, dphi -0.4 +- 0.9 deg (1.6).  The pooled amplitude agrees only because the four species' errors cancel in
  the mean; per species the engine with the muon bend is wrong by +-0.08-0.16, as M3c sec. 2 said.

## 2. The East 60-90 bookkeeping offset (v/v_pred = 1.035-1.047)

Diagnostics (`east_diag.py`, `east_states.py`, `states_bin.py`; pilot-2 c-files of p 3.2-100 GeV/n, the strata that
dominate the East bin):

* **Not statistics, allocation, bin edges or interpolation.** The offset is the same in 12/12 cells and all caps
  with errors 1-3% each (v/v_pred 1.047 +- 0.002 from the 1449 p-stratum files alone); refining the sub-grid of the
  prediction from 5x6 to 20x30 directions changes v_pred by 0.1%; the site map's bilinear interpolation reproduces
  the direct ground-launch cutoff at each state's angles to 0.2% (rms 0.5%).
* **Where it comes from**: all of the excess v sits at primary rigidity R / <R_c(bin)> = 0.95-1.05 (45% of the
  weight below the nominal cut is already allowed): the traced verdict is 1.000 there while the sharp prediction is
  0.47-0.61.  Below 0.8 <R_c> the verdict is 0, above 1.1 it is 1: the MC's effective cutoff for its primaries lies
  at 0.8-0.95 of the site-map mean, not at the map's R_c = 39-52 GV spread.
* **Cause**: the exact upper cutoff R_U of MC-like injection states that arrive in the bin (4e6 sampled states on the
  100 km patch, those crossing the nested caps in the bin, 224 in East) differs from the ground-launched site map
  evaluated at the same crossing-frame (zenith, azimuth): **East -28% (rms 16%), North -36%, West +8% (rms 21%)**,
  already for the 2.5 deg cap (13 East states: -24%).  The floor rule ("injection" vs "sea_level") moves it by 2%.
  So for horizon directions the cutoff that the MC's gate applies to a primary's straight line through the
  injection point is not the site-anchored line-of-sight cutoff of the 1D reference (the map), with an azimuth
  dependent sign and size; vertical (cutoffs 11 GV) and West (7.8 GV, small penumbra) are insensitive at the 1%
  level, East (R_c 40-50 GV) and North are not.  Why a line crossing the cap centre at the same angles has a different
  R_U from the ground launch (field gradients along 10^3 km of near-horizontal path, or a frame/definition
  difference between `NestedCaps.locate`'s crossing frame and `state_above_site`) I did **not** establish; no
  code bug was found in the tally arithmetic (`gate_c3_m3e.py` reproduces v, t, v_pred exactly from the entries).
* **Consequence / recommendation**: the bookkeeping gate is met (<= 1%) in the vertical and West; in East 60-90 it
  measures the difference between two cutoff definitions, +3.5% (+4.7% for the p 3.2-100 GeV/n strata alone,
  v/t 0.04 higher than the map) and should be quoted as such.  For the production table the 1D reference `v`
  must be the MC's own per-primary verdict (as it is) and any comparison with the engine's `G_s` needs the engine's
  cutoff evaluated for the same lines (open item for the engine side).

(See sec. 7 for the data.)

## 3. Solar modulation

### 3.1 Honda solmax: not reachable

`curl` to `http://www-rccn.icrr.u-tokyo.ac.jp/mhonda/public/nflx2014/kam-ally-20-12-solmax.d.gz` (the directory of
the solmin file in `validate_honda.py`) timed out after 50 s (HTTP code 000, 0 bytes; this machine has no outbound
access to it); I did not look for mirrors.  `honda/` under `$W/runs/m3f/` is empty.  Honda's own solmin -> solmax
shift of the 12 cells is therefore **not computed**.

### 3.2 Force-field re-weighting of pilot-2 (exact, per primary)

Method (`rw_ff.py`, `FFModel`): J_1(T) = J_0(T + dPhi) p^2(T) / p^2(T + dPhi), dPhi = (Z/A)(phi_1 - phi_0) per
nucleon, p^2(x) = x^2 + 2 m x, applied per group and nucleon type to the `GSFG:GH` primaries (exact for a flux that is
the Gleeson-Axford modulation of a LIS at phi_0); `T` and the nucleon identity of every pending primary are
regenerated from the c-file seed (identity of the replay validated: phi_1 = phi_0 reproduces the t-files bit for bit on
160 files); one shared trace for all targets (5.1 core-h).

**phi_0 (GSF's epoch) is an assumption, not something I could verify**: GSF is a fit to data including AMS-02
(2011-2013, cycle-24 maximum, force-field phi ~ 0.5-0.6 GV in the usual reconstructions) plus older low-energy sets, and
the ICRC-2017 spline in crflux carries no modulation parameter; I take phi_0 = 0.55 GV and show phi_1 as a grid.  The
shift of a1/a0 is linear in (phi_1 - phi_0): **d<a1/a0>/dphi = -0.054 per GV** (-0.0146 for -0.25 GV, -0.0295 for
+0.55 GV).  I do not know Honda's solmin / solmax phi from the table itself (HKKM2014 documents solar-minimum and
-maximum fluxes of the cosmic-ray primaries, which I cannot read from here), so no phi is "matched to Honda": the
grid is the bracket.

| phi_1 [GV] | <a1/a0> | shift | chi2/12 a1/a0 | <fN> | <fS> | <N/S> | 3D flux / phi_0 (0.3 GeV, cos Z 0-0.3) | antinumu-numu fan-out (0.3/0.5/1) |
|---|---|---|---|---|---|---|---|---|
| Honda solmin | 0.4538 | | | 0.932 | 1.105 | 0.846 | | +0.125 / +0.158 / +0.186 |
| 0.30 | 0.4563 | +0.0146 | 2.62 | 0.918 | 1.075 | 0.856 | 1.06 | +0.169 / +0.166 / +0.259 |
| 0.45 | 0.4475 | +0.0057 | 3.12 | 0.921 | 1.076 | 0.858 | 1.02 | +0.169 / +0.166 / +0.259 |
| **0.55 (pilot-2)** | **0.4417** | 0 | **3.65** | 0.922 | 1.076 | 0.859 | 1 | +0.168 / +0.167 / +0.259 |
| 0.70 | 0.4333 | -0.0084 | 4.73 | 0.925 | 1.077 | 0.861 | 0.97 | +0.168 / +0.167 / +0.259 |
| 0.90 | 0.4226 | -0.0192 | 6.70 | 0.928 | 1.077 | 0.864 | 0.92 | +0.167 / +0.168 / +0.258 |
| 1.10 | 0.4122 | -0.0295 | 9.24 | 0.931 | 1.078 | 0.866 | 0.88-0.90 | +0.167 / +0.169 / +0.258 |

(errors: <a1/a0> +- 0.009 for every row (the same sample); the paired shifts have errors 0.0003-0.0016.)

Reading: **the direction is right** -- a lower phi raises the amplitude -- **and the size is relevant**: the residual
-0.012 in <a1/a0> is closed by phi_1 = 0.55 - 0.012/0.054 = **0.33 GV**, i.e. by phi_0 - 0.22 GV; if GSF's effective
epoch value were 0.45-0.65 GV instead of 0.55, the same Honda-equivalent phi would be 0.23-0.43 GV.  Whether Honda's
solmin sits at that value I cannot say; typical solar-minimum values (0.3-0.5 GV) are in this range and typical
GSF-epoch values (0.5-0.6 GV) above it, so **the solar-modulation difference is a plausible, quantitatively adequate
candidate for the whole residual, not a demonstrated one**.  Not tuned.  fN, fS, N/S, dphi and the fan-out do not
respond (<= 0.01), so modulation does not touch the engine-MC shape differences of sec. 1 or the fan-out excess.

## 4. Honda's statistical noise in our 12 cells

`honda_noise.py` (`honda_noise.txt`).  Two independent estimates of the relative per-bin noise of
`kam-ally-20-12-solmin` for cos Z 0-0.3: (1) roughness along energy: second differences of the log flux in the 101
energy points, sigma = 1.4826 MAD(d2)/sqrt(6) over 0.2-3 GeV (an upper bound, smooth curvature is counted as noise);
(2) the power in azimuth harmonics 5-6 of the 12-bin pattern at fixed (cos Z, E).  Results: sigma = 0.4-0.8% for cos Z
0.1-0.3 and 0.8-1.4% for cos Z 0-0.1 (E-roughness; the azimuth estimate is 10-40% lower, 0.3-1.2%).  A Monte Carlo of
300 noisy copies through the same band integration and observables gives the noise of the 12-cell observables:
**a1/a0 0.0009-0.0011 per cell (0.0003 on the 12-cell mean), dphi 0.1-0.3 deg, fN 0.001-0.002, fS 0.002, N/S 0.002-0.003**
-- 1/20 of our errors (0.01-0.05).  Quoting chi2/12 with Honda's errors added in quadrature changes nothing (every
chi2 above shrinks by < 0.3%).  Robustness: the two estimates agree within 40% and the answer is insensitive to a
factor 3 (0.003 on a1/a0 would still be 1/6 of our error); the assumption is that bins are statistically independent
and that Honda's MC noise is not much larger than the table's own bin-to-bin scatter (a correlated component, e.g.
from a finite shower sample reused across bins, would not show in a roughness estimate; Honda's documented
statistics were not available to me).  Honda's *systematic* uncertainty (hadronic model, solar phase, primary flux)
is separate and larger; it is not estimated here.

## 5. Production design: the Kamioka table G_s(E, cos Z, az) for `GeomagneticModel`

### 5.1 What daemonflux consumes
`GeomagneticModel.admittance(E, zenith, azimuth) = G in [0, 1]` multiplies the 1D flux of every species alike
(`flux_3D = flux_1D * G`); G -> 1 above the cutoff region, ratio quantities are left unchanged, and the module names
itself the plug-point for "admittance ratios from a full 3D Monte Carlo".  The MC's natural object is
`G_s(E, cos Z, az) = s / t`: the 3D flux with cutoff and bending over the same showers' no-cutoff 1D flux (`t`, tally
per bin), per species (nue, antinue, numu, antinumu); a species-summed G is the simplest drop-in (the charge-odd part is
the separate muon-bend table, 5.5).  Note: `s/t` is the MC's own ratio; the engine's 1D reference must use the same
cutoff definition (sec. 2) and the MC's absolute yield deficit (closure B/MCEq 0.82-0.95) cancels.

### 5.2 Binning, range, species, reference
* **Energy**: 0.1-3 GeV from the MC (E >= 3 GeV: pilot-2's s/t errors are 15-33% per fine bin because the allocation
  does not serve them and G -> 1 within the cutoff scale; blend to the engine / 1 above 3 GeV with a sanity band), 10
  bins per decade (Honda's 20/decade pairs) = 15 bins; coarse option 5/decade = 8 bins.
* **cos Z**: 20 bins of 0.1 over (-1, 1) (Honda's), coarse 10 bins; the **upward** half is the same showers (the table is
  filled for all cos Z; cos Z < 0 is the up-going neutrinos which pass through the Earth cap -- only the horizon band
  cos Z 0-0.3 down is quantified here; upward bins to be validated, no pilot-2 statistics were examined).
* **Azimuth**: 12 bins (30 deg); the table is smooth in az, so the production can store the first harmonics
  (a0, a1c, a1s, a2c, a2s; 5 numbers) per (E, cos Z) bin, which reduces the number of independent degrees of freedom
  per bin from 12 to 5: the **coarse-binning errors below correspond to merging az pairs (6 bins)**; harmonic fitting
  is expected to be at least as good (not measured).
* **Reference**: `t` = 1D no-cutoff tally of the same primaries/showers in the same bin (correlated reference, as in
  `sky3d`), plus the MC-internal gate `v/t` as the bookkeeping check (sec. 2).

### 5.3 Errors of the present sample and what 3% / 5% cost
Pilot-2 (2.0e6 primaries, 34.2 core-h = 0.062 core-s per primary; its allocation targets the 0.3-1 GeV E-W cells and
the vertical, not this table) gives, per design bin, the jackknife relative error of G = s/t
(`design_errors.py`, `design_errors.txt`; median over the bins of a class, horizon = cos Z 0-0.3 down):

| class (E band, GeV) | fine bins (10/dec, 20 cos Z, 12 az): median / 90th | coarse (5/dec, 10 cos Z, 6 az) |
|---|---|---|
| horizon 0.1-0.3 | 12.9% / 33% | 10.7% / 21% |
| horizon 0.3-1 | 10.5% / 26% | 7.8% / 13% |
| horizon 1-3 | 16.7% / 52% | 14.5% / 45% |
| mid cos Z 0.3-0.8, 0.1-1 | 5.5% / 8.5% | 3.0% / 4.3% |
| vertical 0.1-1 | 4.5% / 7% | 2.0% / 3.5% |
| vertical 1-3 | 7.0% / 12% | 2.3% / 5.0% |

Scaling is 1/sqrt(N) at fixed allocation: **primaries needed relative to pilot-2 = (error / target)^2**.  The
allocation can be re-optimised for the table; I assume (not demonstrated) up to -30% from that.

| target (median per bin) | fine, horizon 0.1-1 GeV | fine, horizon 0.1-3 GeV | coarse, horizon 0.1-1 | coarse, horizon 0.1-3 | vertical (fine) |
|---|---|---|---|---|---|
| 5% | x4.4-6.6 | x11 | x2.4-4.6 | x8.4 | x0.8 (met) |
| 3% | x12-18 | x31 | x6.8-13 | x23 | x2.3 |

(90th-percentile bins need x3-5 more; the sparsest horizon bins at 1-3 GeV are the cost driver.)

### 5.4 Options (cost = x pilot-2 x 34.2 core-h x 0.7 for the re-optimised allocation; wall = core-h / (cores x
0.85 at 24, 0.8 at 40; pilot-2 ran at ~0.9 efficiency); disk = 14 GB per 2e6 primaries with the c-files (8 GB of that),
6 GB t-files only; free on `/home`: 1.1 TB)

| option | content | precision | primaries | CPU | wall 24 / 40 cores | disk (with / without c-files) |
|---|---|---|---|---|---|---|
| **A: provisional** | the existing pilot-2 tallies (coarse bins, E 0.1-3, GSF-GH, v142p, fixed guard) delivered as is | horizon 8-11% (0.1-1 GeV), 14% (1-3); mid 3%, vertical 2% | 2e6 (done) | 34 core-h (spent) | -- | 14 GB (exists) |
| **B: 5% coarse** | coarse bins (5/dec, 10 cos Z, 6 az or 5 harmonics), 0.1-3 GeV, re-optimised allocation | 5% median horizon 0.1-3 GeV, <= 3% elsewhere | ~1.2e7 | ~200 core-h | 9.8 h / 6.3 h | 85 / 35 GB |
| **C: 3% fine** | fine bins (10/dec, 20 cos Z, 12 az), 0.1-3 GeV | 3% median horizon bins, vertical 2.5% | ~4.5e7 | ~740 core-h | 36 h / 23 h | 300 / 130 GB |
| (C') 3% fine, 0.1-1 GeV only | as C, E-range 0.1-1 | 3% median horizon bins | ~2.6e7 | ~420 core-h | 20 h / 13 h | 175 / 75 GB |

Add-on **muon-bend response table** `R_s(E, cos Z, az) = F(field on)/F(field off)` (M3c sec. 2.3, route 1: one
`--mu-cf` run with no cutoff and no primary bend): the pooled version (cos Z 0-0.3) costs 4 core-h for 1e6 primaries with
errors 0.02-0.06 (done in M3c); resolved in (cos Z, az) it needs ~10x the statistics, **~40 core-h per site** (M3c's
estimate; the companion/pilot-2 per-primary cost ratio is 0.82, so the muon pipeline adds ~20%).  A smooth form
(1 + m_s + q_s A_s cos(az - W)) fitted to the pooled run is the cheap alternative (4 core-h).  Given sec. 1 (the
engine with `muon_bending=True` reproduces ~10% of the fan-out) this table is the only route to correct species
ratios; **recommended with B or C**.

### 5.5 Checkpoint / resume plan
* Every unit of work is a task file: a `c*.pkl` (per (group, stratum, chunk), deterministic from `--seed`) and a `t*.npz`;
  `sky3d.py` skips tasks whose files exist and refuses to resume a job whose manifest (options signature, allocation
  signature, `theta_inj`, caps, model) differs.  Re-running the same command after any interruption resumes.
* Run in blocks of <= 3 h wall (chunk 1600 primaries, a task ~1-4 core-min; the p 32-100 GeV/n strata, 0.4 s/primary,
  are the long tail: schedule them first), with a driver script that logs `progress.txt`, checks free disk (>= 150 GB) and
  re-launches until all tasks exist; merging after the last task (the serial merge of ~8500 t-files took ~10 min; merge
  per block and sum).
* Keep the c-files until the replays needed for the final systematics (primary model, phi, composition: 5 core-h each
  set) are done, then delete (halves the disk).  Back up the `strata_*.npz` reductions (200 MB) off `/home`.
* Jackknife groups = task index mod 20: a task-file checksum list per block makes the error estimate reproducible.

### 5.6 Validation gates (all must pass before delivery)
1. Closure ladder A1/A2/B vs MCEq (`closure_floor.py`): B/MCEq within the documented 0.82-0.95 band and a1/a0
   unaffected; tables record database label + sha256 (`interactions.check_database` warns on mismatch).
2. Bookkeeping gate `gate_c3_m3e.py`: v/v_pred <= 1% in the vertical and West; East/North quoted with the sec. 2
   definition difference until it is resolved (open item), not silently.
3. MC vs MCEq with the true rigidity (`gate_c3_mceq.py`): vertical chi2/12 <= 1.5 (M3e: 1.11); West needs a dedicated
   sample (M3e: statistics-limited).
4. Muon-field-off companion vs the rigidity-fixed engine: a1/a0 chi2/12 <= 2 (M3f: 1.11 with the GSF engine); fN / N/S
   offsets of sec. 1 understood or quoted.
5. Overlap with pilot-2: production cells agree with pilot-2 within 2 sigma (pilot-2 stays as an independent check).
6. G -> 1 within 1% for E >= 3 GeV (blend region), and G_s <= 1 bins flagged (statistical fluctuation above 1 is not
   clipped, reported).
7. Statistical calibration: split the production in halves; per-bin differences have chi2 ~ 1 against the jackknife errors.
8. Systematics sheet (not part of the errors): primary model / phi (sec. 3: -0.054 per GV in <a1/a0>), composition
   (+-0.0035), the pool guard, Honda-comparison offsets (-0.012 residual).

## 6. Code, tests, CPU, open questions

* **Code**: no change to `tools/mc3d` modules (the sec. 2 diagnostics found no bug).  New scripts in
  `results/m3f/scripts/`: `engine_compare.py`, `east_diag.py` (no c-file filtering for the EAST caps), `east_states.py` /
  `states_bin.py`, `honda_noise.py`, `rw_ff.py` (generalised replay with the force-field `FFModel`, identity regeneration for
  group-split models), `design_extract.py`, `design_errors.py`.
* **Tests**: full suite, **108 passed** (no code change), 7 min 35 s.
* **CPU (user+sys)**: force-field replay 5.1 core-h; Honda noise, design extraction/errors, east diagnostics,
  5 extractions and engine comparison ~1.5 core-h (not individually timed: of the order of 10-15 core-min each);
  test suite 0.14 core-h: **~7 core-h**, <= 18 workers at any time.
* **Open questions**
  1. The East/North horizon cutoff definition (sec. 2): why do exact injection states differ from the ground-anchored
     map by -28% / -36% (E / N) and +8% (W) at the same crossing-frame angles?  It is the likely common cause of the
     engine-MC fN / N/S / dphi offsets (sec. 1) and decides which 1D reference the production table must use.
  2. Honda's solmax file (not reachable) and its solmin / solmax phi: needed to turn the force-field bracket into a
     number; GSF's effective modulation epoch phi_0 (assumed 0.55 GV).
  3. Upward (cos Z < 0) bins of the table and the 3-10 GeV blend are not characterised by pilot-2.
  4. The engine with `muon_bending=True` carries 10% of the MC's fan-out (sec. 1): production needs the muon-bend table.

## 8. PM review addendum (2026-11-05)

### 8.1 Honda solmax: fetched, and it accounts for the amplitude residual
The file is reachable over plain HTTP via the CERN proxy given explicitly (`curl -x http://epnu-web-proxy.cern.ch:3128 ...`;
curl ignores the upper-case `HTTP_PROXY` for http URLs, which is why sec. 3.1 timed out).  `kam-ally-20-12-solmax.d.gz`
(internal name `kam-nu-20-12-n3650.d`) is in `$W/runs/m3f/honda/`; `results/m3f/scripts/honda_solmax.py` parses it and
puts it through the same pooled 12-cell observables (`results/m3f/honda_solmax.txt`).  The script reproduces the stored
solmin reference exactly (max |diff| 0).

| | <a1/a0> | pilot-2 - Honda | chi2/12 a1/a0 | dphi | fN | fS | N/S |
|---|---|---|---|---|---|---|---|
| Honda solmin | 0.4538 | -0.0120 | 3.65 | 0.64 | 2.29 | 3.33 | 0.94 |
| Honda solmax | 0.4398 | **+0.0019 +- 0.009** | 2.62 | 0.64 | 2.66 | 3.89 | 0.92 |

* Honda's own solmin -> solmax shift of <a1/a0> is **-0.014**, the same in all four species and three energy bands
  (-0.013 to -0.015).  The 3D flux drops 3.4% / 3.1% / 2.4% at 0.3 / 0.5 / 1 GeV, and by 10 GeV it is 0%.  On the
  force-field slope of sec. 3.2 this is a span of 0.26 GV.
* GSF is fit to AMS-02 2011-13, a solar-maximum epoch (cycle 24, a weak maximum).  Against Honda solmax the mean
  amplitude residual vanishes (+0.002 +- 0.009).  **The -0.012 residual of M3d/M3e is therefore the solar epoch of
  the reference, not a 3D defect.**  This is not tuned: no phi was fitted.
* What survives either epoch is the per-species pattern.  Mean a1/a0 over 3 bands, MC vs Honda solmax:
  nue 0.621/0.601, antinue 0.247/0.269, numu 0.350/0.366, antinumu 0.548/0.523.  So the MC charge-split
  (fan-out) is 13-26% larger than Honda's: antinumu-numu 0.198 vs 0.157, nue-antinue 0.374 vs 0.332.  The
  fN/fS shape also survives (chi2/12 2.3-3.9).  Modulation does not touch either (sec. 3.2).

### 8.2 Review notes on secs. 1, 2, 5
* **Sec. 1, "candidate common cause" is not supported by the sign.**  In North the injection-state cutoffs are 36%
  *below* the map (sec. 2).  A lower cutoff means more North flux in the MC, i.e. a higher fN, but the MC's fN is
  *lower* than the engine's (-0.043).  The fN/dphi engine-MC offset stays unexplained; the cone tails (M2) or the
  primary bend are other candidates.  This is not a blocker for an MC-produced table.
* **Sec. 2, interpretation (PM, not demonstrated).**  For a near-horizontal arrival, the ground-launched map
  back-traces from the site through ~10^3 km of near-horizontal path.  A primary that interacts some 400-700 km
  out never travels that path.  The injection state at 100 km is the physical object.  Its cutoff is evaluated
  ~1000 km away, at a different geomagnetic latitude (North: roughly cos^4 of latitude, 26 -> 36 deg gives about
  -35%) and at a smaller local zenith.  So the MC is right and the site map is the 1D approximation.  Production
  must keep the MC's own per-primary verdict, as sec. 2 recommends.
* **Sec. 5, up-going hemisphere: not covered by this MC.**  In `design_pilot2.npz` the 1D tally `t` is identically
  0 for cos Z < 0.  The 3D tally `s` there is 1e-3 of the down-going content.  With `--theta-inj 40` the injection
  patch reaches up-going arrivals only to cos Z ~ -0.34 (chord exit at 2 arcsin|cos Z|), and the 1D reference is
  defined only for down-going.  Options B/C are therefore **down-going tables**.  The up-going hemisphere must come
  from the engine (far-side production-point cutoffs, `mceq3d_flux.farside_production`) or from a whole-globe
  injection (not costed; efficiency falls as 1/d^2 to the caps).
* **Sec. 5, G is not in [0, 1].**  s/t is 1.22 at cos Z 0.05, summed over E: the 3D horizon enhancement is included.
  `daemonflux.GeomagneticModel` documents G in [0, 1] and multiplies every species alike.  The integration needs a
  per-species factor that may exceed 1.  That is an interface change in `src/daemonflux/geomagnetic.py`.
* **Sec. 5.4, muon-bend add-on.**  A production with the muon field on and per-species s/t already contains the
  fan-out at Kamioka; the design-error medians are per species.  The add-on response table is needed only for the
  engine route: the up-going hemisphere and other sites.
