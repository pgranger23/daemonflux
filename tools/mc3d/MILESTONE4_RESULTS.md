# `tools/mc3d` milestone 4 -- production of the Kamioka down-going table G = s/t (option B) and the muon-bend response table

Record written as the work proceeds (started 2026-11-05).  Outputs under `/home/pigrange/mc3d_work/runs/m4/`
(`out/` = deliverables), scripts in `results/m4/scripts/`.

## 0. Status / resume

| step | state |
|---|---|
| step 0: re-optimised allocation + prediction | **done** (sec. 1) |
| B production (8.73e6 primaries) | **done** 2026-11-05 20:55 (phase 1 15:18-~19:50, phase 2 20:14-20:55), CPU 75.3 core-h (task sums) |
| add-on (mu-cf R_s), 4.36e6 primaries | **done** 20:11, CPU 21.7 core-h |
| `kamioka_G_down_v1.npz`, `kamioka_Rmu_v1.npz` (`$W/runs/m4/out/`) | **delivered** |
| gates 5-8, full test suite | gates done (sec. 7); suite: see sec. 8 |

**Resume:** `source /home/pigrange/mc3d_work/runs/m4/env.sh`; B: `nohup $M/scripts/launch_B.sh` (the driver
re-launches the same `sky3d.py` command, tasks are files; `cat $M/logs/progress_B.txt`).

## 1. Step 0: allocation and prediction

`alloc_m4.py` (from pilot-2's per-stratum jackknife information, `design_pilot2.npz`):
* observables: relative error of G = s/t per species in the coarse down-going bins (E 5/dec over 0.1-2.5 GeV = 7 bins
  x cos Z 5 bins of 0.2 x az 6) = 840 bins; the 2.5-10 GeV bins are kept in the tallies for the blend check but are
  not constrained.
* model: Var_q = sum_h a_hq / n_h, a_hq = n_h^pilot x (jackknife variance of the ratio when only stratum h varies).
  Check at the pilot-2 allocation: the additive model reproduces the jackknife medians (horizon 0.1-0.25 GeV 6.07% vs
  6.07%; 0.25-1 5.03% vs 5.19%; 1-2.5 7.84% vs 8.39%; other 2.4-3.1% vs 2.4-3.2%): conservative by <= 7%.
* targets (median over the bins of each class, per E-band 0.1-0.25 / 0.25-1 / 1-2.5 GeV): 5% in the horizon bins
  (cos Z 0-0.4 coarse bins 0-1), 3% elsewhere; class weights iterated to the cheapest allocation meeting all six.
* cost: pilot-2's measured CPU per primary per stratum (hybrid, muon field on) x 1.14 for the tracing.
* result: **6.71e6 primaries, 109.7 core-h** meet the targets (predicted medians: horizon 2.7 / 2.9 / 5.0%, other 1.1 /
  1.3 / 2.0%; **90th percentile of the horizon bins 5.7 / 6.3 / 12.3%**: the 1-2.5 GeV horizon tail does not reach 5%).
  This is far below the 285 core-h cap (and below the 200 core-h of sec. 5.4 of M3f, whose classes were per band
  and per fine bin).  **I launched with a x1.3 safety margin: 8.73e6 primaries, predicted 143 core-h** (floors
  of 1600 primaries per stratum; `reports/alloc_B.npz`).
* Seeds: `--seed 31415926`; the task seeds (`sky3d.build_tasks`) are 41336 distinct values with 0 overlap with pilot-2's
  (the first choice 20271105 overlapped in 1727, rejected).

## 2. Production configuration
Identical to pilot-2: `--model GSFG:GH` (native), v142p tables (default), fixed guard, `--r-thr 65 --resim-rmin 32
--resim-groups p --reanchor`, muon field and primary bend on, IGRF-14, site Kamioka, caps and `--theta-inj 40` default,
chunk 1600.  Tallies per task are the full fine grid (s, n, t, v x 2 measures, caps, species, E (sky.E_BINS 20/dec),
20 cos Z, 12 az).  Driver `m4_driver.py`: blocks <= 3 h (SIGTERM), `SKY3D_TAIL_FIRST` (high-energy-per-nucleon strata first),
a watcher every 60 s that stops the job if the CPU sum of finished tasks (sum of t-file `meta[7]`) exceeds the cap, free
disk < 150 GB or the RSS of our python processes > 35 GB; workers = 20 - the add-on's workers.

## 3. Operations log (2026-11-05)
* 15:18 B launched (14 workers), 15:27 add-on (6 workers): 20 workers in total.  My own RSS watcher summed per-process
  RSS (double-counting copy-on-write pages shared by the forked workers: 55-57 GB read vs 14 GB used by `free`) and halted
  both drivers at 18:15 without data loss; replaced by PSS (PM confirmed PSS is the right reading of the 35 GB cap) and a
  per-process CPU counter.  A second flaw (each driver's CPU counter also counted the OTHER job's processes) was found at
  18:48 and fixed (`--tag`-filtered), both jobs restarted at 18:52 from the task files (in-flight tasks lost, ~minutes).
  CPU book-keeping after these restarts uses `work/cpu_prior_*.txt` (B 49.5, add-on 20.5 core-h at 18:52: upper-bound
  estimates from wall x workers for the periods the old counters got wrong); true totals will be the sum of the task `meta[7]`
  from the t-files (printed at the end).  At no time did more than 20 workers run (one extra single-core analysis process
  at a time, within the 1-core allowance).

## 4. The 1D tally t is "bent showers laid on straight lines" (mechanism, 2026-11-05)
PM finding: at the horizon the production's `t` falls (numu 0.3 GeV, t/t(cos Z 0.5-1) = 0.67 / 0.76 / 0.92 at cos Z
0.05 / 0.15 / 0.25) while `s` follows Honda's zenith shape and the curved-atmosphere 1D flux is ~1.05; `s1/t1` at cos Z 0.05
depends on the cap (1.05 / 1.62 / 1.49 / 1.26 for 2.5 / 5 / 7.5 / 10 deg).

**Mechanism** (code: `sky3d.cascade_task`, `bend.bend_primary`, `sky3d.build_entries`, `DualCapScorer`): with the primary bend
on, each primary is transported in the geomagnetic field to its first interaction and the shower is run from that bent
state; the 1D (`t`, `v`) entries are laid on the primary's STRAIGHT injection line (`locate_vec(r0, u0)`) with the weight of
the shower that actually happened.  A primary whose straight line crosses the caps but whose bent trajectory escapes the
atmosphere or hits the ground before interacting makes no shower, and that line contributes zero to `t`.  `bend_t_diag.py` /
`bend_t_diag2.py` (4.5e5 and 1.2e6 proton states whose straight line crosses the caps, same sampled interaction depths for
both; `results/m4/bend_t_diag_full.txt`): fraction that make a shower, bent (the straight line is 1.0, 0.92 at R 2-4 GV which
is the pool floor):

| cos Z at the crossing | R 2-4 GV | 4-8 | 8-16 | 16-60 |
|---|---|---|---|---|
| 0.0-0.1 | 0.51 | 0.63 | 0.60 | 0.78 |
| 0.1-0.2 | 0.58 | 0.62 | 0.68 | 0.93 |
| 0.2-0.3 | 0.64 | 0.69 | 0.87 | 1.00 |
| 0.3-0.6 | 0.82 | 0.95 | 1.00 | 1.00 |
| **0.6-0.8** | **0.995** | 1.000 | 1.000 | 1.000 |
| **0.8-1.0** | **1.001** | 1.000 | 1.000 | 1.000 |

(bent / straight ratios; R 1-2 GV is below the pool's interaction floor for both.)  Weighted by the 0.3 GeV neutrino parents
(R ~ 5-30 GV) this reproduces the t deficit; stored runs agree (numu 0.25-0.5 GeV, 10 deg cap, cos Z 0.05 / 0.15 / 0.25: M3c
`a1nc` with primary bend 0.57 / 0.77 / 0.87 like pilot-2's 0.62 / 0.78 / 0.92; `a1ncnb` without 1.34 +- 0.14 / 1.01 / 1.12,
noisy and with a different energy sampling).

**What it touches / does not touch**
* `s`, `n`: actual 3D rays of the bent showers: correct (and Honda-like in zenith shape).  Azimuth observables (a1/a0, dphi, fN,
  fS, N/S, fan-out) use `s` only: **untouched** (M3b-M3e tables, M3f engine comparison, force-field bracket).
* `t` is not a field-free 1D reference; `v = t x verdict` inherits it.  Everything stated as "s/t", "v/t" or `G = s/t` is affected
  at cos Z < 0.6: the M3b/M3d/M3e G or "3D/1D" statements, the M3f design errors for G = s/t (sec. 5 of M3f, relative errors
  of s/t per bin, which include t's own noise and bias), and the pilot-2 `s/t` horizon enhancement.  The M3e/M3f gate-C3
  bookkeeping tests (`v/t` vs the sharp cut `v_pred/t`, vertical chi2/12 1.11, West 1.00 +- 0.012) compare two quantities built on
  the same set of surviving lines, so the arithmetic closes; but the surviving set is biased against low-rigidity, strongly
  bent lines, so the *physical* meaning of v/t at the horizon is not "the no-cutoff 1D transmission".  The M3f sec. 2 East
  offset (v/v_pred +3.5%) was traced to the exact-state cutoff vs the ground-anchored map (sec. 2 of M3f); whether the
  selection of surviving lines contributes is not tested.  The MC-vs-MCEq vertical gate (cos Z 0.95, M3e) is unaffected (>= 0.6).
* The closure ladder, the guard fix, the force-field bracket, the engine-MC amplitude comparison: untouched.

## 5. Amended production definition (PM, 2026-11-05)
G_s(E, cos Z, az) = s / (kappa_s(E) x B_s(E, |cos Z|)); B = the engine's MCEq curved-atmosphere 1D (GSF, v142p, no cutoff),
`$W/runs/integ/out/base1d_kamioka.npz` (engine agent, `sky.E_BINS` x `sky.CZ_BINS`, measure 1), common reference of both
hemispheres.  kappa_s(E) = t / B fitted only where `bend_t_diag` gives an interacting fraction >= 0.99 for every feeding
rigidity: **cos Z 0.6-1** (the table above: 0.995-1.001 for R >= 2 GV).  Gate: t/B flat in cos Z over 0.6-1 within errors.
`s/t`, `v/t` stay as internal diagnostics only.  The deliverable npz gets kappa_s(E) with jackknife errors.
Validation of kappa: the add-on run IS a no-primary-bend, no-cutoff run of the same configuration (`--no-cutoff
--no-bend-primary --mu-cf`, GSFG:GH, same pools and tables, 4.36e6 primaries, seeds 27182818), so its `t` is the requested
no-bend 1D reference over the full cos Z 0-1, with more primaries than the 1e6 asked for: I use it instead of a separate 1e6
run (saves ~5 core-h).  Gates: kappa(no-bend, cos Z 0-1) = kappa(bent, 0.6-1) within errors; t_nobend/B flat in cos Z 0-1.  If either
fails, kappa comes from the no-bend run.

### Amended sec. 5.6 validation gates
1. closure ladder as before; 2. ~~bookkeeping gate~~: v/v_pred as before (diagnostic; vertical and West pass, East/North quoted as the
cutoff-definition difference, not "t-clean" -- see sec. 4); 3. MC vs MCEq with the true rigidity as before (vertical); 4. companion vs
engine mb0 (amplitude); 5. overlap with pilot-2 (s only, azimuth observables and s per bin; NOT s/t); 6. G -> 1 in the blend bins
and G not clipped; 7. split-half statistical calibration of s/(kappa B); **8a. t/B flat over cos Z 0.6-1 (kappa gate);
8b. kappa(no-bend 0-1) vs kappa(bent 0.6-1); 8c. t_nobend/B flat 0-1;** 9. systematics sheet (phi_0 0.55 GV, composition, guard).

## 6. Results
### 6.1 kappa_s(E) = t / B (bent production B, cos Z 0.6-1; B = engine curved-atmosphere 1D, GSF, v142p)
Per band, mean over species in brackets of the per-species values (nue / antinue / numu / antinumu; jackknife errors 0.3-1.8%):

| E band [GeV] | kappa_B (0.6-1) | kappa_addon (no bend, 0.6-1) | flatness chi2/3 of t/B in the 4 cos Z bins (B) |
|---|---|---|---|
| 0.1-0.3 | 1.024 / 1.001 / 0.999 / 1.002 | 1.018 / 0.995 / 0.999 / 1.000 | 3.1 / 4.0 / 3.6 / 0.8 |
| 0.3-1 | 0.950 / 0.943 / 0.935 / 0.932 | 0.948 / 0.942 / 0.927 / 0.929 | 0.8 / 4.4 / 7.6 / 4.2 |
| 1-3 | 0.950 / 0.942 / 0.952 / 0.940 | 0.935 / 0.959 / 0.950 / 0.929 | 0.7 / 3.8 / 3.2 / 2.0 |
| 3-10 | 0.961 / 0.925 / 0.945 / 0.938 | 0.958 / 0.931 / 0.929 / 0.965 | 0.4 / 2.5 / 1.9 / 5.2 |

kappa < 1 above 0.3 GeV (0.93-0.95): the MC's absolute 1D yield vs MCEq (closure B/MCEq 0.88-0.95 at those energies),
kappa ~ 1.00 at 0.1-0.3 GeV.  Per-E-bin kappa (with errors) is in the deliverable (`kappa_fine`, `kappa_coarse`, `kappa_err_*`).

### 6.2 Typical errors of G = s / (kappa B) (median relative error per species and bin, coarse bins, down-going; 8.73e6 primaries)
| region | 0.1-0.25 GeV | 0.25-1 | 1-2.5 | 2.5-4 | 4-10 (blend) |
|---|---|---|---|---|---|
| horizon cos Z 0-0.4 (90th pct) | 1.3% (2.6) | 1.7% (3.5) | 2.8% (5.0) | 3.9% (7.3) | 7.5% (12.6) |
| mid 0.4-0.8 | 0.8% (1.2) | 0.9% (1.4) | 1.6% (2.4) | 2.3% (3.6) | 4.5% (9.2) |
| vertical 0.8-1 | 0.7% (1.0) | 0.7% (1.2) | 1.6% (2.2) | 2.0% (3.3) | 4.1% (9.6) |

Targets (5% horizon, 3% elsewhere, median per band) met with a factor 2-4 to spare in 0.1-2.5 GeV (the model of sec. 1 was
conservative: predicted medians 2.7 / 2.9 / 5.0% horizon).  Mean G (all species, az): horizon 0.54 / 0.64 / 0.78 / 0.89 / 0.96, mid
0.35 / 0.60 / 0.83 / 0.95 / 1.00, vertical 0.34 / 0.61 / 0.85 / 0.95 / 1.02 (same bands); G is not clipped (individual bins
exceed 1 above ~2.5 GeV within the errors).

## 7. Gates (amended list of sec. 5)
| gate | result |
|---|---|
| 1 closure ladder (M3e, v142p) | B/MCEq 0.82-0.95 (0.1-0.3 GeV), 0.89-0.95 (0.3-1) as quoted in M3e; unchanged |
| 2 bookkeeping v/v_pred (M3e/M3f; diagnostic) | vertical, West <= 1%; East/North quoted as the cutoff-definition difference; v inherits the t deficit (sec. 4), not re-measured |
| 3 MC vs MCEq true rigidity (vertical) | M3e: MC/C 0.993, chi2/12 1.11 (not rerun) |
| 4 companion vs engine mb0 | M3f: a1/a0 +0.012 +- 0.010, chi2/12 1.11 (not rerun) |
| **5 overlap with pilot-2 (s per coarse bin, E 0.1-2.5, 840 bins, independent samples)** | **chi2/ndf 1.20**, mean B/pilot-2 - 1 = +0.0023 +- 0.0013; horizon 1.03 (+1.3%), mid 1.47 (-0.9%), vertical 1.01 |
| **6 G -> 1 at 3.2-10 GeV (fine bins, all species/az), cos Z 0.05...0.95** | 0.89 0.97 0.99 1.03 1.03 0.99 1.01 0.98 1.01 1.04 (mean bin error 0.12): consistent with 1 within the errors; the horizon bin 0.89 +- ~0.1; not clipped |
| **7 split halves (groups 0-9 vs 10-19), G coarse E 0.1-2.5 GeV, 840 bins** | **chi2/ndf 0.67**, mean (G1 - G2)/G = -0.0008 (errors consistent, slightly over-estimated) |
| **8a t/B flat over cos Z 0.6-1 (B)** | chi2 0.4-7.6 for 3 dof in all 16 (species x band); worst numu 0.3-1 GeV 7.6/3 (p = 0.055): **pass** |
| **8b kappa(no bend, cos Z 0-1) vs kappa(bent, 0.6-1)** | chi2/16 = 0.56, all within 1.5 sigma: **pass** (kappa from the bent run stands) |
| **kappa_B vs kappa_addon over 0.6-1** | chi2/16 = 1.22, largest +2.0 sigma (numu 0.3-1 GeV, +0.011): agree |
| **8c t_nobend/B flat over cos Z 0-1 (add-on)** | flat within the errors (0.92-1.05 in cos Z 0-0.1 with 5-9% errors, 0.99-1.05 at 0.1-0.2, +1-3% (1-2 sigma) at 0.2-0.4) down to the horizon; **one exception: antinue 1-3 GeV, cos Z 0-0.1: 0.76 +- 0.08 (3 sigma low)**, a statistically poor cell; antinue 0.3-1 GeV flatness chi2 47/9 over the ten 0.1-wide bins (single bins at 0.95-1.02); a mild +1-3% bump at cos Z 0.2-0.6 is not excluded.  The primary-bend mechanism explains the deficit of the bent runs (0.51-0.78 -> 1.0), but 8c does not prove it at the 3% level at the horizon |
| 9 systematics sheet | phi_0 = 0.55 GV assumed; M3f: d<a1/a0>/dphi = -0.054 per GV; composition +-0.0035; guard: fixed |
Not done / not applicable: **no up-going bins** (the MC tally t is zero for cos Z < 0; the engine supplies them against the same B); the
gate "pool guard" etc. are as in M3e.

## 8. Deliverables, CPU, tests
* `$W/runs/m4/out/kamioka_G_down_v1.npz` (795 kB): axes (`E_edges_fine/coarse`, `cosZ_edges_*`, `az_edges_*`), species, per species `G_*`,
  `G_err_*`, `s_*`, `n_*`, `t_*`, `v_*`, `B_*`, `kappa_*`, `kappa_err_*`, diagnostics `s_over_t_*`, `v_over_t_*`; metadata: DB label/sha
  (v142p / 7de8275ff3...), sha256 of `base1d_kamioka.npz`, primary model, phi_0 = 0.55 GV (assumed), seed 31415926, N = 8 725 685, CPU 75.3
  core-h, hybrid options, site/date/field model/caps.  Fine grid: E 10/dec 0.1-10 GeV (20 bins), cos Z down 10 bins, az 12; coarse: E 5/dec
  (10 bins; bins 0-7 = 0.1-3.98 GeV deliverable, 8-9 blend check), cos Z 5, az 6.
* `$W/runs/m4/out/kamioka_Rmu_v1.npz` (40 kB): R_s (4 sp x 7 E 0.1-2.5 x 5 cos Z x 6 az) with jackknife errors (median 0.47% outside
  the horizon, 1.9% in cos Z < 0.4; median |R - 1| 1-2%), F_on, mu_on, mu_cf; field geometry at Kamioka (IGRF-14 2020-01-01): |B| 47.87 uT
  (N 29.60, E -4.30, Up -37.37), declination -8.27 deg, dip 51.33 deg; N = 4 362 833 (seed 27182818), CPU 21.7 core-h.
* **CPU** (user+sys, from the task `meta[7]` sums; the drivers' running figures were upper bounds): B 75.3, add-on 21.7, calibration 0.54,
  analysis extractions (kappa x3, G, R, gates, bend diagnostics) ~5: **~103 core-h** (caps 285 / 50; prediction 143 / 27 -- the allocation margin
  was generous and tracing cheaper than in pilot-2).  Wall: 15:18-20:55.
* **Operations incidents** (sec. 3): the RSS-watcher halt (18:15, no data loss), the driver CPU double count (fixed 18:52), and the **20:12-20:14
  double B pool (up to 40 workers for ~1-2 min)**, reported to the PM at once.
* Tests: see the final line below.
* **Tests**: full suite from `$W/runs/m4/work/pytest`: **108 passed**, 25 warnings, 7 min 11 s (0.11 core-h) after the code changes of M4
  (`Tally.mucf_full`, `--no-keep-cfiles`, `SKY3D_TAIL_FIRST`; the `mucf` test in `test_m3c.py` extended).  `results/m4/pytest_final.log`.

## 9. Open items
1. Gate 8c: antinue 1-3 GeV at cos Z 0-0.1 (0.76 +- 0.08, 3 sigma) and the +1-3% bump at cos Z 0.2-0.6 of the no-bend t/B; more no-bend statistics at the
   horizon would settle whether the bend mechanism is complete at the few-% level.
2. G in the 2.5-10 GeV bins is a blend/consistency region (errors 4-12%), not a deliverable range.
3. phi_0 = 0.55 GV (solar epoch of GSF) assumed; the force-field slope of M3f (-0.054 per GV in <a1/a0>) applies.
4. Up-going hemisphere: engine, against the same B.

## 10. Reference-shape factor c and the total error (PM request 2026-11-05 22:00)
c(E band, cos Z group) = t_addon / (kappa_addon B), kappa over cos Z 0.6-1 (`cref.py`, `results/m4/cref.txt`); jackknife (G = 20)
errors with the species taken together replicate by replicate, so the shared showers are accounted for.  Species-combined, PM groups:

| E band [GeV] | cos Z 0-0.1 | 0.1-0.2 | 0.2-0.4 | 0.4-0.6 | 0.6-1 |
|---|---|---|---|---|---|
| 0.1-0.3 | 1.009 +- 0.047 | 1.018 +- 0.019 | 1.013 +- 0.011 | 0.994 +- 0.008 | 1 (fit) |
| 0.3-1 | 0.964 +- 0.036 | 1.031 +- 0.021 | 1.020 +- 0.012 | 1.015 +- 0.009 | 1 |
| 1-3 | 0.934 +- 0.045 | 1.018 +- 0.027 | 1.014 +- 0.009 | 1.008 +- 0.012 | 1 |
| 3-10 | 0.945 +- 0.104 | 1.017 +- 0.070 | 1.036 +- 0.031 | 0.996 +- 0.026 | 1 |

* **Species compatibility** (coarse bins): max |c_sp - c_comb| / err_sp is 1.4-2.1, mean 0.5: compatible, so the combined c is used.
* **Significance of the mid-zenith bump** (cos Z 0.2-0.6, 0.1-3 GeV): +1.3..+2.0% with errors 0.9-1.2% per group: 1.1-1.7 sigma per cell, the
  neighbouring cells and bands are correlated (same showers); a joint significance of ~2 sigma at most.  It is not established.
* **sigma_ref(E coarse bin, cos Z coarse bin) = max(|c - 1|, err(c))** (err(c) alone for cos Z 0-0.2, where c is noise-dominated, and for 0.8-1,
  part of the kappa window): 0.1-0.25 GeV: 2.8 / 1.3 / 0.8 / 0.5 / 0.5%; 0.25-1: 2.0 / 2.0 / 1.5 / 0.5 / 0.6%; 1-2.5: 2.8 / 1.4 / 1.2 / 0.5 / 0.6%; 2.5-10:
  5.5 / 3.6 / 2.6 / 1.1 / 1.2% (cos Z bins 0-0.2 ... 0.8-1).
* **Not applied as a correction in v1.**  The deliverable gets `sigma_ref_coarse`, `c_coarse`, `c_err_coarse`, `G_err_total_coarse` and `G_err_total_fine`
  = sqrt(G_err^2 + (sigma_ref G)^2) (`G_err_*` stay statistical).  Median relative error of G, statistical -> total, coarse down-going:

| region | 0.1-0.25 | 0.25-1 | 1-2.5 | 2.5-4 | 4-10 (blend) |
|---|---|---|---|---|---|
| horizon cos Z 0-0.4 | 1.3 -> 2.4% | 1.7 -> 2.6% | 2.8 -> 3.5% | 3.9 -> 6.0% | 7.5 -> 9.0% |
| mid 0.4-0.8 | 0.8 -> 1.0% | 0.9 -> 1.6% | 1.6 -> 1.9% | 2.3 -> 3.1% | 4.5 -> 5.1% |
| vertical 0.8-1 | 0.7 -> 0.9% | 0.7 -> 0.9% | 1.6 -> 1.7% | 2.0 -> 2.4% | 4.1 -> 4.3% |

**Outliers** (one line each).
* antinue 1-3 GeV, cos Z 0-0.1 (0.761 +- 0.081): a statistics-poor cell, not a heavy-tail artefact at the 1D-weights level: N_eff = 1512 (vs 5.8e4 for antinue
  0.3-1 GeV in the vertical bin), and removing one jackknife group moves t by -7.7% / +3.3%, i.e. single groups carry ~8% of the cell (1/cos psi weights at the horizon);
  the same species/band at 0.1-0.2 and 0.2-0.4 is 0.99 +- 0.03, so I treat it as a ~3 sigma fluctuation of a noisy cell (one of 48 per-species cells below cos Z 0.2).
* antinue 0.3-1 GeV, flatness chi2 19.7/3 (0.6-1) and 46.9/9 (0-1): not driven by heavy weights (vertical N_eff 5.8e4, leave-one-group-out spread
  +-0.4%); the deviation sits in the single cos Z 0.9-1 bin (t/B / kappa 0.96 +- ~0.01, a 4% dip) of the add-on and is absent from the bent production
  (same cells: chi2 4.4/3 in the 0.6-1 window), so it is an add-on-sample (statistical or per-bin B-average) effect; not explained further.  Its size (4%) is not carried
  into the G error because the production G uses kappa of B (bent), not of the add-on.

**v2 option.**  A larger add-on lets v2 apply c as a correction instead of an error: the current err(c) is 2.0-2.8% in the horizon bin (cos Z 0-0.2), 1.0-1.2% at
cos Z 0.2-0.4 and 0.5-0.9% above (0.1-3 GeV).  err(c) scales as 1/sqrt(N): for err(c) <= 1% everywhere at cos Z >= 0.2 about x1.5 the present add-on,
**~33 core-h (6.5e6 primaries)**; for err(c) <= 1% also at cos Z 0-0.2 about x(2.8)^2 = 7.8, **~170 core-h (3.4e7 primaries)** with the present allocation (less with an
allocation tuned to the horizon low-energy strata; not computed); at 2.5-10 GeV (5.5%) x30, ~650 core-h, not worth it.
