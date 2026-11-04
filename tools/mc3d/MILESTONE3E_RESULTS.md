# `tools/mc3d` milestone 3e -- consolidated pilot-2 (v142p tables, fixed guard, GSF primaries, hybrid transport)

Factual record, written as the work proceeds (started 2026-11-04).  Raw outputs
under `/home/pigrange/mc3d_work/runs/m3e/`; summaries in `results/m3e/`, scripts
in `results/m3e/scripts/` (plus `$M/scripts`).

## 0. Status / resume here

| step | state | where |
|---|---|---|
| 1. MC tables on patched v1.4.2 (`mceq_tables_SIBYLL23D_v142p.npz`, label + sha256 stored, mismatch warning, old npz kept) | **done**, tested | sec. 1 |
| 2. closure ladder after all fixes vs MCEq on v142p | **done** (baseline; x5 statistics rerun, 1.3 core-h total) | sec. 2 |
| 3a. hybrid transport option (`--resim-rmin`, `--resim-groups`), allocation for s + t + v with floors | **done**, tested | sec. 3 |
| 3b. pilot-2 (2.0e6 primaries) | **done** (34.2 core-h) | `$M/sky/pilot2_tasks`, `$M/logs/pilot2.log` |
| 3c. companion `--no-muon-field` (0.8e6, shared first chunks) | **done** (11.3 core-h) | `$M/sky/comp_tasks` |
| 4. analysis (`analyse_m3e.py`, `gate_c3_m3e.py`, `gate_c3_mceq.py`) | **done**, engine comparison waits for the GSF mb0 file | sec. 4 |
| 5. full test suite | **108 passed** | sec. 6 |

**Resume:** `source /home/pigrange/mc3d_work/runs/m3e/env.sh` (= m3d's with `M=$W/runs/m3e`);
`cat $M/logs/progress.txt`; queue script `$M/scripts/run_pilot2.sh`.
Tasks are resumable from their files: re-run the same command.

---

## 1. MC tables on the patched v1.4.2 database

* `build_tables.py::build()` now builds the run with `mceq3d_db.make_run`
  (read-only import of `tools/mceq3d/mceq3d_db.py`; database = `$MCEQ3D_DB`,
  default `v142p`) and stores `db_label` and `db_sha256` in the npz.  Default
  output `mceq_tables_SIBYLL23D_v142p.npz` (237 arrays; 9 s).  Check: `cs_2212`
  at 56.2 GeV is 2.737e-25 cm^2 (273.7 mb) against 2.185e-25 (218.5 mb) in the
  old table; the neighbouring bins 273.5 / 273.9 mb.
* The old table `mceq_tables_SIBYLL23D.npz` (v1.4.0, hole, no database record)
  is **kept untouched**.  `interactions.DEFAULT_TABLES` is the v142p table
  (falls back to the old file if the new one is missing); `MC3D_TABLES=v140`
  (or a path) selects the old one.  To reproduce m1-m3d use
  `MC3D_TABLES=v140 MCEQ3D_DB=v140`, `--primary-energy kinetic` (m3b) or
  `total` (m3c/m3d H3a), and `pool.GUARD_MASS = 0.94` (m1-m3c; m3d runs used
  the fixed guard only for the closure studies).
* `MCEqYieldBackend` (and thereby `ChromoBackend(xs="mceq")`, which owns one)
  calls `check_database`: a `RuntimeWarning` if the table's sha256 differs from
  `mceq3d_db.resolve().sha256`; a table without the record counts as the shipped
  v1.4.0 file.  `closure.mceq_reference` also uses `make_run`, so the references
  of rungs A1/A2 use the same database as the tables.
* Tests: `test_m3d.py::test_mc_tables_record_and_check_their_database` (record
  present, old table without, warning on mismatch, no warning on match, the
  hole is gone in the new table and present in the old one).

## 2. Closure after all fixes (v142p tables, fixed guard, MCEq on v142p)

`closure_floor.py` (collinear, vertical column, four flavours summed); A1 = table
yields + MCEq decay tables, A2 = table yields + decay kinematics, B = chromo
DPMJET/SIBYLL pools (the MC's own backend, guard fixed).  MC/MCEq 1D, errors
statistical (8000 showers per point at E_p <= 4 GeV, 4000 at 6-20, 1500 at 100).
CPU 0.25 core-h (wall x workers; `$M/logs/cpu_closure.txt`).
Per-species, per-band numbers: `results/m3e/closure_species.txt`; summary
(`ladder_m3e.txt`):

0.1-0.3 GeV band:

| rung | 1.5 | 2.5 | 4 | 6 | 10 | 20 | 100 GeV |
|---|---|---|---|---|---|---|---|
| A1 | 0.98 | 0.89 | 0.98 | 0.96 | 0.98 | 0.97 | |
| A2 | 0.91 | 0.80 | 0.90 | 0.94 | 0.93 | 0.98 | 0.98 |
| **B** | **0.82** | **0.82** | **0.88** | **0.94** | **0.91** | **0.92** | **0.90** |

0.3-1 GeV band (errors 0.07 / 0.05 / 0.02 / 0.02 / 0.014 / 0.01 / 0.009):

| rung | 1.5 | 2.5 | 4 | 6 | 10 | 20 | 100 GeV |
|---|---|---|---|---|---|---|---|
| A1 | 0.90 | 0.89 | 0.94 | 0.97 | 0.98 | 0.97 | |
| A2 | 0.96 | 0.95 | 0.93 | 0.98 | 0.96 | 1.01 | 0.99 |
| **B** | **0.74** | **1.13** | **0.94** | **0.90** | **0.93** | **0.95** | **0.95** |

1-3 GeV band at E_p >= 10 GeV: B 0.95 +- 0.04 (10), 0.92 +- 0.02 (20), A2
0.99 +- 0.02 at 100 GeV (B 0.95 +- 0.02).  These are the **baseline numbers the
production table should quote: B/MCEq = 0.91 +- 0.01 (0.1-0.3 GeV) and 0.93-0.95
+- 0.01 (0.3-1 GeV) at 10-100 GeV, 0.82-0.88 (0.1-0.3) and 0.74-1.1 (0.3-1,
errors 0.05-0.07) below 5 GeV.**  Compared with m3d (v140 tables, same guard
fix) nothing moves beyond the errors (e.g. B 0.1-0.3 GeV at 4 / 6 / 10 / 20 GeV:
0.88 / 0.94 / 0.91 / 0.92 vs 0.88 / 0.94 / 0.90 / 0.92).  The E_p = 2.5 GeV dip
of A1 and the 9% A2/A1 decay-module deficit at 0.1-0.3 GeV are as in m3d.
The residual MC/MCEq deficit at 10-100 GeV is split about equally between the
table-vs-pool generator terms (B/A2 = 0.93 / 0.96 per band at 100 GeV) and
nothing else (A2 = 0.98-0.99).  Per species (100 GeV, B): nu_mu 0.90 / 0.96,
antinu_mu 0.91 / 0.95, nu_e 0.87 / 0.94, antinu_e 0.89 / 0.93 (0.1-0.3 / 0.3-1 GeV, +-0.01-0.02).

## 3. Pilot-2 design

* **Primaries**: `--model GSFG:GH` (GSF nucleon-level exact, GH group split;
  native = total energy); the +-0.0035 (0.8%) composition systematic of m3d
  sec. 2.2 applies to the a1/a0 below.
* **Fixed guard, v142p tables, native convention**: defaults.
* **Hybrid transport** (new, `sky3d.Options(resim_rmin, resim_groups)`,
  CLI `--resim-rmin`, `--resim-groups`): a primary is re-simulated in every
  azimuthal copy iff `resim_rmin <= R < r_thr` and its group is allowed; the rest
  is rotated and (with `--reanchor`) re-anchored.  Pilot-2 uses
  `--r-thr 65 --resim-rmin 32 --resim-groups p --reanchor`: protons of 32-65 GV
  re-simulated (the one 3-sigma re-anchoring bias of m3c sec. 3), everything
  else re-anchored.  The signature records the window (tail `[-1, rmin,
  group indices]`) only when it is set; m3b-m3d manifests are unchanged (test).
* **Allocation** (`results/m3e/scripts/alloc_m3e.py`, `reports/alloc_m3e.npz`):
  Neyman over the 35 (group, stratum) cells for the summed relative
  variance of the 12 pooled `a1/a0` (tag s) **and** of the vertical (cos Z 0.9-1)
  `t` and `v` rates (4 species x 3 bands), from the m3d GSF-GH+gfix re-weighted
  pilot, with a floor of 300 expected 3D-scoring primaries and 0.2% of the total
  per cell (the pilot's p 1-3.16 GeV/n had 47 of 2850: now 43 500 primaries,
  ~700 expected).  2.0e6 primaries (pilot 1.44e6), estimated 25 core-h.
* Seed 20270201, chunk 1600, `--nproc 24`.  Companion: same command + `--no-muon-field`,
  allocation x 0.4 (0.8e6 primaries, chunks of the same seeds are the same
  streams as pilot-2's first chunks), `$M/sky/comp_tasks`.
* Calibration (1% allocation, 2e4 primaries, 8 workers): 0.24 core-h phase 1 + 1.1 core-h
  extrapolated tracing -> 26 core-h for 2e6.  Observed per-primary CPU
  (hybrid): p 1-3.2 GeV/n 0.006 s, p 3.2-10 0.019, p 10-32 0.045, p 32-100 0.40 s.

## 3b. Production record

| run | primaries | CPU (user+sys, incl. tracing, merge) | wall |
|---|---|---|---|
| pilot-2 | 2.000e6 | **34.2 core-h** (phase 1 4434 s x 24 workers = 29.6; tracing 1.02e6 copy states ~4) | ~1.9 h |
| companion (`--no-muon-field`) | 0.800e6 | **11.3 core-h** (target 8-10: over by ~1) | 0.7 h |

(m3b pilot: 1.44e6 primaries, 58 core-h; the hybrid transport is ~x1.7 cheaper per primary
for the same primaries.)  My cost model underestimated the p 32-100 GeV/n stratum (0.40
s/primary, 8 core-h alone: the 32-65 GV window is re-simulated in all 12 azimuthal copies).

## 4. Analysis

(`results/m3e/m3e_table.txt`, `analyse_m3e.py`; jackknife G = 20; Honda HKKM2014
`kam-ally-20-12-solmin`; 12 pooled cells = 4 species x 3 bands (0.3, 0.5, 1 GeV), cos Z 0-0.3.)

### 4.1 Pilot-2, 12 pooled cells

a1/a0 (Honda in brackets):

| | 0.3 GeV | 0.5 GeV | 1 GeV |
|---|---|---|---|
| nue | 0.552 +- 0.020 (0.583) | 0.618 +- 0.030 (0.643) | 0.692 +- 0.030 (0.615) |
| antinue | 0.289 +- 0.026 (0.315) | 0.270 +- 0.026 (0.310) | 0.183 +- 0.030 (0.226) |
| numu | 0.343 +- 0.012 (0.400) | 0.398 +- 0.019 (0.410) | 0.310 +- 0.023 (0.332) |
| antinumu | 0.511 +- 0.017 (0.525) | 0.565 +- 0.021 (0.569) | 0.569 +- 0.026 (0.518) |

Species-pooled (mean of the three bands): nue 0.621 (Honda 0.614), antinue 0.247 (0.284), numu 0.350
(0.381), antinumu 0.548 (0.537).  All 12 cells: **<a1/a0> = 0.4417 +- 0.0089** (Honda 0.4538),
<dphi> +0.20 deg (-0.10), <fN> 0.922 (0.932), <fS> 1.076 (1.105), <N/S> 0.859 (0.846).
chi2/12 vs Honda (MC errors only): **a1/a0 3.65**, dphi 0.64, fN 2.29, fS 3.33, N/S 0.94.
(m3b pilot: 1.95, 1.67, 2.60, 1.93, 0.75; M3d GSF-GH + gfix re-weighting: 2.20, 2.44, 2.71, 1.31, 1.13.)

Fan-out (0.3 / 0.5 / 1 GeV): antinumu-numu **+0.168 / +0.167 / +0.259** (Honda +0.125 / +0.158 /
+0.186; M3d re-weighted: +0.144 / +0.217 / +0.279); nue-antinue +0.264 / +0.348 / +0.509 (Honda
+0.268 / +0.333 / +0.389).  The 0.3-0.5 GeV fan-out now matches Honda within ~0.04 (errors 0.03);
the 1 GeV excess (+0.07..+0.12) stays.  The cells' fine structure moved from M3d (numu 0.5 GeV
0.341 -> 0.398, nue 1 GeV 0.608 -> 0.692), see 4.4.

**Statistical errors vs the m3b pilot** (target: no worse): error of the 12-cell mean
**0.0089 vs 0.0135** (better, x1.5), rms of the cell errors 0.024 vs 0.039 (better), but the
**median cell error 0.024 vs 0.017 is worse**, and 8 of the 12 cells have a larger error than
the pilot's (ratios 0.24-1.98).  The target is therefore met for the pooled quantity and the worst
cells, **not cell by cell**: the allocation moved primaries to the 1D references (t, v) and to the
low-energy floors, and re-anchored copies are ~1.3x noisier per primary than re-simulated ones.

### 4.2 Companion (`--no-muon-field`, 0.8e6 primaries)

a1/a0: nue 0.415 / 0.508 / 0.478 (+-0.026-0.046), antinue 0.414 / 0.482 / 0.400, numu 0.412 / 0.507 /
0.408, antinumu 0.443 / 0.450 / 0.456 (errors 0.02-0.05).  <a1/a0> 0.4477 +- 0.0115, <dphi> -1.38,
<fN> 0.922, <fS> 1.093, <N/S> 0.846.  **The species pattern is gone**: antinumu-numu fan-out
+0.031 / -0.057 / +0.048 (+-0.04), nue-antinue +0.001 / +0.026 / +0.078; the mean amplitude is
unchanged (companion - pilot-2 +0.006 +- 0.014).  So the fan-out of pilot-2 (and of Honda) is the
muon-bend mechanism (M3c) and nothing in the primary or cutoff chain.  chi2/12 vs Honda a1/a0 16.1
(meaningless: Honda has the field-on pattern).  **Engine table not yet available**: `analyse_m3e.py
--engine FILE.npz` takes `a1_rel, dphi, fN, fS, NS` (4, 3) arrays (+ `*_err`) and prints
MC-engine per cell with chi2/12; run it when the other agent's `muon_bending=False` numbers exist.

### 4.3 Redefined gate C3

**(a) Bookkeeping test** (`gate_c3_m3e.py`, `gate_c3_bookkeeping_pilot2.txt`): the MC's `v/t` against
`v_pred/t`, the sharp site-map R_c cut applied to each 1D entry at its own primary's nucleus rigidity
(a 5 x 24 / 5 x 6 sub-grid of directions per bin), same entries, same weights, 60 cells (5 bins x 4
species x 3 bands).  **Vertical (cos Z 0.9-1): v/v_pred = 0.996-1.001 +- 0.001-0.003 in all 12
cells.  87 deg West (two 30 deg bins): 1.000 +- 0.012 (cells 0.975-1.025).  87 deg East 60-90: 1.04
+- 0.01 (all 12 cells 1.01-1.055, same sign), East 90-120: 0.99 +- 0.03.**  Mean over the 60 cells
1.006, median |dev| 1.0%, 4 cells beyond 5% (all with errors 0.03-0.06).  **Passes at the few-% level**;
the only systematic is +3.5% in the East 60-90 bin (traced verdicts allow more than the sharp
R_c at R_c ~ 44 GV: penumbra / geometry of the entries), not understood in detail.

**(b) MC vs MCEq with the true rigidity** (`gate_c3_mceq.py`, `gate_c3_mceq_pilot2.txt`; MCEq per-cell
responses regenerated on v142p, `mceq_response.py`, 1.9 core-h; the prediction uses the MC's own primaries
(GSF-GH, native) at the nucleus rigidity (A/Z) p, T >= 1 GeV/n, the site-map R_c averaged over the bin's
directions: variant "C"; "B" = MCEq's own H3a flux, T >= 2; "D" = rigidity (A/Z) T as the old engine):
**vertical: MC/C = 0.993 mean, cells 0.88-1.04, chi2/12 = 1.11 (MC errors 1-8%)**; m3c's same test failed
(chi2/12 31).  Tolerance from step 2: the closure B/MCEq is 0.82-0.95 in *yield* but cancels in the ratio
v/t, so the passing criterion is the statistical one.  Worst vertical cells: antinue 1 GeV 0.88 +- 0.08,
antinue 0.3 GeV 0.96 +- 0.03, nue 0.3/0.5 GeV 1.04 +- 0.03.  Variant D (rigidity (A/Z) T, the engine's
old convention) is 6-8% low in the vertical, B and C agree to <1%.  **87 deg West (two bins): MC/C 1.03 /
1.05 mean, but each cell has 5-17% statistical error and several cells saturate at v/t = 1 with zero
error; the cells with a defined error are within 1.8 sigma except antinumu 1 GeV in 240-270 (1.113 +- 0.015,
saturated, error not meaningful).**  The West gate is statistics-limited at the 10% level: it neither
confirms nor excludes a few-% bias there.

### 4.4 The +gfix check and consistency with M3d

M3d's approximate correction predicted for GSF-groups (GH composition) <a1/a0> 0.4304 -> **0.4422**
(+0.0118, +-30% of the shift).  **Pilot-2 gives 0.4417 +- 0.0089: -0.0005 from the prediction** (the
pilot-2 error of the mean is 0.009, so a +0.012 shift is confirmed only at ~1.3 sigma, and "no shift"
is not excluded by this number alone; pilot-2 also differs from M3d in tables, transport hybrid and
allocation).  Cell by cell pilot-2 minus M3d(+gfix): mean -0.0004, chi2/12 = 0.92 with quadrature
errors (no contradiction beyond the errors); the largest cell differences are nue 1 GeV +0.073
(1.7 sigma), numu 0.3 GeV -0.032 (-1.9), numu 0.5 GeV +0.039 (+1.7).  I do not claim a
contradiction, nor a sharp confirmation: the check is limited by the errors of both runs.

## 5. Code changes (tools/mc3d; tools/mceq3d untouched)

* `build_tables.py` (database via `mceq3d_db.make_run`; label + sha256 stored),
  `interactions.py` (`TABLES_V140/V142P`, `MC3D_TABLES`, `table_database`, `check_database`,
  warning from `MCEqYieldBackend`), `closure.py` (`mceq_reference` via `make_run`).
* `sky3d.py`: `Options(resim_rmin, resim_groups)`, `--resim-rmin`, `--resim-groups`, signature tail.
* new `mceq_tables_SIBYLL23D_v142p.npz`; `test_m3d.py` (+2 tests: hybrid option/signatures, table database).
* `results/m3e/scripts/`: `alloc_m3e.py`, `analyse_m3e.py`, `gate_c3_m3e.py`, `gate_c3_mceq.py`,
  closure/queue scripts.


## 6. Engine comparison hook, tests, CPU, open questions

**Engine comparison**: `analyse_m3e.py --engine FILE` reads the rigfix cell files directly
(`/home/pigrange/mc3d_work/runs/rigfix/out/m3b_engine_<h3a|gsf|ghnative>_mb0.npz`, keys `e`, `total_<sp>` as
`analyse_m3b.engine_patterns` expects; band integration identical to Honda's) or a plain npz of (4, 3)
arrays.  **Dry run only** with the H3a file (H3a engine vs the GSF-GH companion is not a physics comparison:
MC - engine mean a1/a0 +0.043, chi2/12 2.75 is meaningless); the GSF `mb0` file did not exist when this was
written.  Run: `python analyse_m3e.py --engine .../m3b_engine_gsf_mb0.npz` (or `ghnative`; the companion uses
GSF-groups with the GH composition, so the GSF file is the like-for-like one).

**Tests**: full suite from `$M/work/pytest`: **108 passed**, 25 warnings, 8 min 30 s (106 + 2 new).

**CPU (tracked with `/usr/bin/time`, user+sys)**: pilot-2 34.2 core-h; companion 11.3; MCEq responses 1.9;
closure ladder (baseline + x5) 1.3 (inc. MCEq references); two test runs 0.3; extractions, bookkeeping gate,
allocation, diagnostics ~1 (not individually timed): **~50 core-h**.  <= 24 cores at any time (21-24 production
workers + 1-2 single jobs).

**Open questions / caveats**
1. Pilot-2 does **not** contradict M3d beyond the errors (chi2/12 0.92 cell by cell, mean -0.0004 +- 0.014), but
   the M3d +gfix shift is confirmed only at ~1.3 sigma; the per-cell error target of the m3b pilot is met for the
   12-cell mean and rms, not for the median cell (sec. 4.1).
2. After all fixes the pilot is still 0.012 below Honda in <a1/a0> (0.4417 +- 0.009 vs 0.4538; chi2/12 3.65) and
   the 1 GeV fan-out is too large (+0.26 vs +0.19 antinumu-numu); the companion shows the whole fan-out is
   the muon bend.  The composition systematic (+-0.0035) is not folded into the errors above.
3. West-bin gate C3 is statistics-limited (sec. 4.3b); a dedicated West sample would be needed.
4. East 60-90 bin v/v_pred +3.5% not explained.
5. The companion cost 11.3 core-h (budget 8-10).
6. Old tables/runs: reproduce with `MC3D_TABLES=v140 MCEQ3D_DB=v140`; the m3d `--primary-energy kinetic` and
   `GUARD_MASS = 0.94` as described in sec. 1.
