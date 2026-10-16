# `tools/mc3d` milestone 3b — cutoff, primary and muon bending, azimuthal reuse, first East-West pilot

Factual record, written as the work proceeds (2026-10-11).  Raw outputs under
`/home/pigrange/mc3d_work/runs/m3b/` (never `/tmp`); small summaries copied to
`results/m3b/`.  **Status / resume point: see §0.**

---

## 0. Status (resume here)

| step | state | where |
|---|---|---|
| IGRF-14 default (`cutoff.DEFAULT_MODEL`) | done | `cutoff.py` |
| field grid, primary bend, muon re-transport, vectorised Michel | done, tested | `field.py`, `bend.py` |
| driver (cascade phase + pooled trace/replay phase) | done, tested | `sky3d.py` |
| gate C1 (regression vs 2b) | **passed** (test) | `test_m3b.py::test_regression_reproduces_milestone_2b` |
| gate C2 (charge mirror of the per-copy weights) | **passed** (test) | `test_m3b.py::test_copy_verdicts_are_charge_mirrored_in_a_dipole` |
| pending-test leakage | **0** (bit-identical, test + one chromo task) | `test_m3b.py::test_pending_test_loses_nothing` |
| primary-bend study | done | `results/m3b/bend.txt` |
| allocation runs `a1rot` + paired re-anchored twin `a1anc` (fixed pools) | done | `results/m3b/alloc_a1.txt`, `a1anc_ew.txt` |
| gate C3 (1D site cutoff vs engine `G_s`) | **passes vertical (<= 2%) and East; FAILS West at 0.5-1 GeV (MC +8-22%, up to 6 sigma)** -- open (sec. 3.1) | `results/m3b/gate_c3_pilot.txt` |
| gate C5 (bend-threshold bias) | **passed**: bias of the reuse above 65 GV `<~ 0.01` | `results/m3b/bias_c5.txt` |
| East-West pilot (1.44e6 primaries, 58 core-h) | **done -- reproduces Honda's species fan-out** (sec. 4) | `results/m3b/pilot_ew.txt` |
| mechanism decomposition (muon field off / primary bend off) | done -- **the fan-out is the muon bend** (sec. 5) | `results/m3b/a1nomu_ew.txt`, `a1nobend_ew.txt` |
| tests | 90 pass (`$PY -m pytest $R/tools/mc3d -q`, 9.5 min, 2026-10-16) | |

The agent that built this milestone was stopped by an API spend limit on
2026-10-12 01:22 after launching the pilot and the decomposition runs; the
runs completed unattended, and the PM ran the analyses (`final_report.py`,
`gate_c3.py`, `analyse_m3b.py`) and wrote secs. 3.1 and 4-8 on 2026-10-16.

Environment: see `MILESTONE2_RESULTS.md` §7; in addition
`export MC3D_FIELD_CACHE=$W/runs/m3b/work/fieldcache` (the 20 MB IGRF-14 grid,
built in 10 s if absent).

### 0.1 Pools and environment: the SIBYLL nucleon-projectile fix (2026-10-11)

`chromo` 0.11.0 routes every nucleon/antinucleon projectile (p, n, pbar,
nbar) to `sibnuc(A=1)`, so SIBYLL-2.3d simulates neutrons and antinucleons as
protons (upstream chromo PR #275, unmerged; PM finding: SIBYLL n+air at
158 GeV gave pi+/pi- = 1.137, the proton's 1.134; with the fix 0.958 vs
1.138).  In the mc3d pools this affects every nucleon/antinucleon pool above
the 80 GeV DPMJET->SIBYLL switch -- a charge-ratio bias in exactly the
East-West observable.  The PM built `$W/pylib_fix` (two-line fix in
`chromo/models/sibyll.py`) and `$W/pools_fix` (all `pool_+-2212_*`,
`pool_+-2112_*` regenerated, the other pools byte copies).

| runs | environment |
|---|---|
| every physics run quoted here (`a1rot`, `a1anc`, the pilot, gate C3) | `$W/pylib_fix`, `MC3D_POOL_CACHE=$W/pools_fix` (`runs/m3b/env.sh`) |
| gate C1 (regression vs 2b, `test_m3b.py`) | backend-independent (MCEq-yield backend in the test); `runs/m3b/env_oldpools.sh` for the chromo version |
| `a0rot` (first allocation pilot, superseded by `a1rot`, not quoted) | old `$W/pylib`, `$W/pools` |

**Caveat on milestone 2b**: every number in `MILESTONE2_RESULTS.md` was
produced with the buggy pools.  The 3D/1D ratio is expected to move little
(the defect is mostly a nu/nubar charge-ratio effect carried by >80 GeV
nucleon primaries, and numerator and denominator share the cascade), but the
per-species absolute yields above ~1 GeV and anything charge-odd in 2b are
affected.  **Caveat on the interaction lengths**: `ChromoBackend(xs="mceq")`
takes them from the shipped MCEq database (`mceq_db_lext_dpm193_v140.h5`,
withdrawn upstream for v1.4.2), which carries wrong DPMJET antiparticle cross
sections and the 56.2 GeV hole (`tools/mceq3d/MCEQ_DB_HOLE.md`); the
antinucleon/antimeson interaction lengths of the cascade inherit that.

**Measured effect of the fix on this milestone's observables**
(`results/m3b/poolfix.txt`): `a0rot` (old pools) and `a1rot` (fixed pools)
are the same job on the same seeds (1e6 primaries, everything on, every
cascade reused by rotation); they stay paired until the first SIBYLL nucleon
draw.  3D East-West amplitude: mean d(a1/a0) = -0.010 over the 36 cells,
chi2(d = 0)/36 = 1.03 -- no effect visible at this statistic.  Azimuth-
integrated 3D fluxes and 3D/1D ratios at cos Z 0.3-1: equal within the 2-5%
errors (e.g. numu 0.5 GeV 3D flux new/old 0.978 +- 0.018, 3D/1D 1.002 +-
0.026).  The defect lives in >80 GeV nucleon projectiles, which make few of
the sub-GeV neutrinos; the milestone-2b caveat above stands as "small, not
measured below the few-% level".

---

## 1. What was implemented

| file:function | what it does |
|---|---|
| `cutoff.py:DEFAULT_MODEL` | **IGRF-14** (DGRF-2020 at 2020-01-01) everywhere: the 3D test, the 1D site map, the primary and the muon bending (PM decision 1; the 19 `test_cutoff.py` tests pass unchanged with it) |
| `field.py:FieldGrid` | IGRF-14 tabulated at 1 deg x 1 deg x 10 km over 0-130 km (Cartesian B, trilinear), cached `.npz`; <= 5e-4 of `|B|` (17 nT) against `FastIGRF` |
| `field.py:bend_rotation`, `rotate` | the magnetic rotation `du/ds = (q c/p) u x B` as an exact rotation about `B_hat` by `-q Q_C |B| ds / p` |
| `bend.py:bend_primary` | the primary from the injection sphere to its first interaction in the field (rigidity of the parent nucleus, charge +1), vectorised over all the primaries / copies of a task: exact rotation per step, chord advance, 5-point Gauss-Legendre chord grammage, bisection for the interaction point; with `field=None` it is `atmosphere.advance_grammage` to 20 m |
| `bend.py:transport_muons` | the muon segment with `dE/dx` and the field, vectorised (all copies of all muons of a task in lock-step); with `field=None` it reproduces `shower.transport_muon` step for step (same outcome 300/300, decay point within 100 m, energy within 5e-3) |
| `bend.py:michel_decay` | the polarised Michel decay vectorised, with the decay frame built on the site vertical so that, with common random numbers, the copies of a field-free decay are exact rotations of one another |
| `shower.py:run_shower(first=...)`, `scorer.add_muon` | hooks: a pre-computed (bent) transport of the primary to its first interaction; a callback per muon segment (start state, polarisation, outcome, `int ds/p`, proper decay time).  Both default off: the RNG stream and every 2b result are unchanged |
| `shower.py:transport_muon` | bend constant fixed (`2.998e-4` -> `0.2998` rad/m/T/GeV; the scalar field branch had never run) and the `stats` hook |
| `sky3d.py:cascade_task` | phase 1 per (species, stratum, chunk): H3a nucleus rigidity, the 12 azimuthal copies (rotations about the site vertical, copy 0 = the sampled primary), the exact gate per copy, the realisations (`B` reused by rotation above `r_thr`; `S` straight + `C_j` per copy below), the bent primaries in one vectorised call, the cascades, the conservative pending test, then **every per-copy cap entry** (`build_entries`) and the list of copies that need a back-trace |
| `sky3d.py:build_entries` | rotations of the non-muon neutrinos, **exact per-copy re-transport of every pending muon segment in the copy's own field** + Michel decay with common random numbers, cap crossings; the 1D entries on each copy's straight line with the site-map verdict |
| `sky3d.py:pending_3d`, `soft_hits`, `cone_may_hit_cap` | the pending test: exact for non-muon neutrinos; for each muon a cone of half-angle `widest decay angle above e_nu_min + Q_C |B|_max int ds/p + 2 deg` tested against the cap's direction disc and the horizon |
| `sky3d.py:trace_replay_task` | phase 2 per group of c-files (>= 1e4 copy states): **one pooled `cutoff.trace_gated` call**, the verdict matrix, the tallies |
| `sky3d.py:Tally` | the four tallies `s` (3D, injection-point cutoff), `n` (3D, no cutoff, copy 0), `t` (1D, no cutoff), `v` (1D, site-line-of-sight cutoff), both measures, squares summed per primary, `hinj` |
| `sky3d.py:SiteMap`, `site_map` | `cutoff.SiteCutoffMap` (IGRF-14, ground launch, sea-level floor) built once (51 s on 30 cores) and cached |
| `sky3d.py:_reanchor_table` (`--reanchor`) | the bias measurement of the rotation reuse: every rotated copy of a reused cascade re-anchored on the copy's own exact bent primary (same sampled depth) |
| `sky3d.py:run`, `merge`, `build_tasks(alloc=...)` | the job (resumable at both phases), the merge, an explicit per-(species, stratum) allocation |
| `study_m3b.py bend` | the primary bend and the rotation-reuse error vs rigidity and zenith |
| `analyse_m3b.py` | the East-West tables (Fourier `a1/a0`, phase, `W/E`, `max/min`, N/S) with stratified jackknife errors, Honda band-integrated on the same bins, the engine, the 3D/1D ratios, the allocation (`allocation`), the rotation bias (`rotation_bias`), gate C3 (`gate_c3`) |
| `test_m3b.py` | 15 tests (sec. 8) |

---

## 2. Decisions, with the measurements behind them

### 2.1 Field model (PM decision 1)

`cutoff.DEFAULT_MODEL = "IGRF14"`, epoch 2020-01-01, for the 3D cutoff test
(`trace_gated`), the 1D site map (`SiteCutoffMap`: vertical 11.51 GV, 87 deg
East/West 41.67/7.48 GV, North/South 26.8/9.8 GV -- the Phase-1 numbers), the
primary bend and the muon bend (`FieldGrid`, built from the same `FastIGRF`
coefficients).  All 19 `test_cutoff.py` tests pass unchanged (the Kamioka
vertical island and the East-West verdicts are the same with IGRF-14).

### 2.2 The primary bend and the re-simulation threshold (PM decision 2; gate C5)

**Implementation.**  `bend.bend_primary`, vectorised over every bent
realisation of a task (0.17 ms per primary; 2.2% of the CPU of `a1rot`).
**Superposition approximation** (documented in `bend.py`): the nucleon handed
to the cascade, proton or neutron, bends with its *parent nucleus's* rigidity
`R = (A/Z) p_nucleon` and charge +1 until its own first interaction, drawn from
the nucleon interaction length.  A real nucleus fragments earlier, after which
its neutrons go straight and its protons bend at `p_nucleon` (half the
rigidity for A/Z = 2); the error is confined to the part of the first path
beyond the nucleus's own (shorter) interaction length, for the ~26% of
nucleons that arrive in nuclei.  A primary bent out of the atmosphere is lost
(no re-entry): 2.8-4.3% of 8-15 GV patch primaries against 1.6% for straight
lines (`study`: grazing directions); most are below the Kamioka cutoff.

**Bend vs rigidity** (`study_m3b.py bend`, 3000 cap-pointing primaries x 12
copies, `results/m3b/bend.txt`): median bend at injection zenith < 45 deg
15.4 / 6.6 / 3.7 / 2.0 / 1.3 / 0.9 / 0.3 deg at 2-5 / 5-10 / 10-15 / 20-30 /
30-45 / 45-60 / 100-300 GV (~60/R deg), twice that at 45-70 deg and four times
at 70-85 deg (MILESTONE3A sec. 9 confirmed).  The **error of reusing a
cascade by rotation** -- the angle between the rotated copy-0 interaction
direction and each copy's own one -- is as large as the bend itself:
median `eps ~ 67/R` deg (19 / 8.8 / 5.4 / 2.7 / 1.8 / 1.3 / 0.9 / 0.4 deg in
the bands above; p90 1.7-1.9x the median), with a km-scale displacement of
the interaction point.

**Bias of the rotation reuse, measured** (`sky3d --reanchor`,
`analyse_m3b.rotation_bias`, `results/m3b/bias_c5.txt`).  Two runs on the same
seeds, 1e6 primaries each, everything on, every cascade reused by rotation
(`r_thr = 0`): `a1rot` (rotation) and `a1anc`, where every rotated copy is
*re-anchored* on the copy's own exact bent primary (same sampled depth, the
copy's own field: the cascade is moved rigidly onto the exact first-interaction
point and axis, then its muons are re-transported).  Paired, so the difference
is nearly free of cascade noise.  Its effect on the 3D East-West amplitude
`a1/a0` (36 cells: 4 species x 0.3/0.5/1 GeV x cos Z 0-0.1/0.1-0.2/0.2-0.3):

| strata still reused by rotation | mean d(a1/a0) | rms d | chi2(d = 0)/36 |
|---|---|---|---|
| all (no re-simulation at all) | **+0.150** | 0.169 | 7.15 |
| R_lo >= 10.9 GV (p >= 10 GeV, nuclei >= 3.2 GeV/n) | +0.042 | 0.067 | 2.19 |
| R_lo >= 65 GV (p >= 100 GeV, nuclei >= 31.6 GeV/n) | **+0.003** | 0.015 | 1.25 |
| R_lo >= 101 GV | +0.001 | 0.009 | 0.97 |

Per stratum (each reused alone): p 3.2-10 GeV +0.013, **p 10-32 GeV +0.062**,
p 32-100 GeV +0.016 (chi2/36 = 1.22), He 3.2-10 GeV/n +0.019, He 10-32 GeV/n
+0.015; every stratum with `R_lo >= 65 GV` <= 0.003 in the mean and consistent
with zero.

* **The rotation reuse is badly biased below ~60 GV, always low**: reusing
  every cascade by rotation lowers the East-West amplitude by 0.15 on
  average -- ~30% of the amplitude itself.  The rotated copies carry the
  *base* copy's bend rotated with them, i.e. a bend that is wrong in
  direction for every other copy; summed over the copies that smears the
  coherent, geomagnetically oriented bend and with it the dipole.  **The
  primary bend is a first-order ingredient of the East-West amplitude.**
* **Threshold: 65 GV.**  Copies of primaries below 65 GV are re-simulated
  with their own bent primary; above, the reuse by rotation is kept.  The
  measured bias of the reused part is +0.003 +- 0.003 (mean over the 36
  cells; rms 0.015 and chi2/36 = 1.25, i.e. compatible with zero) for the
  strata wholly above 65 GV; the upper part (65-101 GV) of the p 32-100 GeV
  stratum, also reused, carries at most ~+0.006 by the stratum's +0.016
  scaled with the `1/R` error over its rigidity distribution.  **Bias bound
  at 65 GV: |d(a1/a0)| <~ 0.01**, against pilot statistical errors of
  0.02-0.05.  A threshold at 20 GV would leave a +0.04 bias (chi2/36 = 2.2).
* **Cost** (`analyse_m3b.realisations_per_primary`): below the threshold a
  primary costs 1 straight + up to 12 bent cascades (copies gated forbidden
  are skipped): x10.5 for p 3.2-10 GeV, x13 for p 10-32 GeV and He 3.2-32
  GeV/n, x8.2 for p 32-100 GeV.  At equal CPU the threshold costs a factor
  (0.535/0.240)^2 = 5.0 in variance against reusing everything (task B
  table, sec. 2.6) -- the price of an unbiased amplitude.
* **Re-anchoring is the cheap alternative for the next milestone**: it removes
  the first-order bias at ~1.1x the CPU instead of ~5x.  Its residual
  (the cascade's dependence on the changed zenith of its axis) is second
  order and is measured by comparing the pilot (re-simulated below 65 GV)
  with `a1anc` (re-anchored everywhere) in sec. 4.4.

### 2.3 The muon bend: exact per-copy re-transport (task A)

**Chosen: exact.**  Every muon segment of a cascade that can reach the cap is
re-transported in each copy's own field (`bend.transport_muons`), from its
production point to its decay at the *same proper time* as in the base cascade
(common random numbers across the copies, also for the Michel decay), then
decayed.  Measured cost in `a1rot` (1e6 primaries, everything on): **9.9e6
muon segment-copies, 27 us each, 0.074 core-h = 1.7% of the run (2.6% of the
cascade CPU)**.  The rotated-bend approximation would save at most that 1.7%
and would need its own gate (MILESTONE2_RESULTS 5.2 item 4); it was not
implemented, so gate C4 does not apply.  The scalar `shower.transport_muon`
field branch had a bend constant 1000x too small (never exercised; fixed).

Gates of the transport: `field=None` reproduces `shower.transport_muon` (300/300
outcomes, decay point < 100 m, energy < 5e-3); in a uniform 0.5 G field the
bend is 5.354 deg per lifetime at every energy, right-hand-rule sign
(`test_m3b.py`).

### 2.4 The two 1D references (PM decision 3)

Both on the same primaries, laid on each copy's **straight** injection line
(no bending; below `r_thr` from the dedicated straight realisation `S`, above
it from the reused cascade, whose primary bend is < 1 deg):

* `t` -- no cutoff: the correction that multiplies daemonflux;
* `v` -- the site line-of-sight cutoff (`SiteMap`, IGRF-14, sharp at `R_U`):
  isolates the genuine 3D part.

`DualCapScorer`'s single weight is replaced by separate 3D and 1D weights in
`sky3d.Tally` (`s` = `w/12 x allowed(copy)`, `t` = `w/12`,
`v` = `w/12 x [R > R_c(site; 1D direction of the copy)]`).

### 2.5 Lazy, pooled cutoff tracing (PM decision 5)

Only copies with at least one 3D cap entry are traced, after the cascade, in
one `cutoff.trace_gated` call per group of >= 1e4 states.  In `a1rot`:
0.256 traced states per primary (2.6e5), 10.6 ms each (17.5% of the CPU);
the gate decided another 0.08 forbidden and 0.04 allowed per primary.
The pending test (sec. 1) is conservative: replaying every realisation gives
bit-identical tallies (`test_m3b.py::test_pending_test_loses_nothing`, and a
SIBYLL-pool task of 150 p primaries at 32-100 GeV).

### 2.6 Task B: the stratum allocation with the cutoff on

From `a1rot` (1e6 primaries, `E^-0.8` per group, 7 strata 1-3162 GeV/nucleon,
`results/m3b/alloc_a1.txt`): the share of the summed relative variance of the
36 3D amplitudes `a1/a0` carried by each (species, stratum):

| | 1-3.2 | 3.2-10 | 10-32 | 32-100 | 100-316 | > 316 GeV/n |
|---|---|---|---|---|---|---|
| p | 0.000 | 0.150 | 0.248 | 0.089 | 0.014 | 0.002 |
| He | 0.006 | 0.149 | 0.139 | 0.046 | 0.010 | 0.001 |
| CNO | 0.002 | 0.041 | 0.021 | 0.008 | 0.000 | 0.000 |
| MgSi | 0.001 | 0.015 | 0.006 | 0.001 | 0.000 | 0.000 |
| Fe | 0.001 | 0.035 | 0.012 | 0.001 | 0.001 | 0.000 |

* The cutoff empties the lowest stratum (protons below 3.2 GeV are all
  forbidden near the site; 43% of the `E^-0.8` samples) and the high strata
  do not feed the sub-GeV horizon: **93% of the variance comes from
  3.2-100 GeV/nucleon** -- the North-patch protons the brief anticipated are
  there but small (He 1-3.2 GeV/n 0.6%).
* Neyman (`analyse_m3b.allocation`, `n_h ~ sqrt(a_h / c_h)` with the measured
  CPU per primary including the re-simulation multiplier) at 65 GV, for
  55 core-h: **1.44e6 primaries; p 26% / 13% / 4.8% in 3.2-10 / 10-32 /
  32-100 GeV, He 10% / 14% / 5% / 4% in 1-3.2 / 3.2-10 / 10-32 / 32-100
  GeV/n, CNO+MgSi+Fe 15%** (`results/m3b/alloc_thr65.txt`).  Against
  `E^-0.8` at the same CPU the rms relative error of the amplitudes falls by
  x0.91 at this threshold (x0.88 with no re-simulation).

---

## 3. Gates

| gate | how | result |
|---|---|---|
| **C1** regression vs milestone 2b | `test_m3b.py::test_regression_reproduces_milestone_2b` (cutoff off, bending off, B = 0; one task, same seed; MCEq-yield backend so it runs without pools) and `runs/m3b/scripts/regress.py` (the same with the chromo pools, old `pylib`/`pools`) | **passed**: with one copy every tally equals `sky._shard`'s bin by bin (`rtol` 1e-10; the chromo task: 2e-15); with 12 copies the azimuth-summed `s`, `t` (and `n`, `v` vs 2b's `s`, `t`) are equal to 3e-14; the cone tally is identical |
| **C2** charge mirror of the cutoff weights | `test_m3b.py::test_copy_verdicts_are_charge_mirrored_in_a_dipole`: the verdict of copy `j` of 40 primaries x 12 copies (8-30 GV) with charge +1 equals that of copy `-j` of the mirrored (`y -> -y`) primaries with charge -1 in an aligned dipole -- pins the copy rotation sense against the back-trace's charge reversal; in full IGRF the 3D tally's flux maximum sits at geomagnetic West for every species (`dphi` within +-10 deg of 0, sec. 4) and the 1D site-map tally has its maximum 14-20 deg off it, as the detector-anchored map of Phase 1 (sec. 6.3 there: -12.7 deg) | **passed** (480/480 verdicts mirrored; 42% allowed) |
| **C3** 1D site-cutoff reference vs the engine's `G_s` | `analyse_m3b.gate_c3`: MC `v/t` (same primaries) against `MCEq3DFlux.geomag_response` (H3a, SIBYLL-2.3d, sharp cutoff) averaged over the bin with the site map's `R_c`, weighted by the engine's own no-cutoff flux | vertical and East pass; **West fails at 0.5-1 GeV** (sec. 3.1) |
| **C4** rotated muon bend | not applicable: the exact per-copy re-transport is used (sec. 2.3) | -- |
| **C5** primary-bend threshold | paired `a1rot`/`a1anc` (sec. 2.2) | bias of the reuse above 65 GV: mean +0.003, `|d(a1/a0)| <~ 0.01` |
| pending-test leakage | `pend_all` vs pending on the same seeds | bit-identical (test + one chromo task) |
| transport | `test_m3b.py` (bend vs `advance_grammage`, uniform-field rotation, muon transport vs `shower.transport_muon`, 5.354 deg/lifetime, Michel vs `decays`, covariance of the decay frame) | passed |

### 3.1 Gate C3: the MC's 1D site-cutoff transmission against the engine's `G_s`

`results/m3b/gate_c3_pilot.txt`.  On the pilot, the MC transmission `v/t`
(1D tallies, same primaries: site line-of-sight cutoff over no cutoff) against
the factorised engine's `G_s(R_c, E)` (H3a, SIBYLL-2.3d, sharp cutoff)
averaged over the bin with the same site map.  This checks the MC's yields
*per rigidity* and its 1D cutoff bookkeeping independently of anything 3D.

Vertical (cos Z 0.95), MC/engine: nue 1.016 / 1.044 / 0.937, antinue 1.011 /
1.026 / 1.003, numu 0.978 / 1.018 / 1.004, antinumu 0.80 +- 0.26 (one
outlier stratum) / 1.002 / 1.014 at 0.3 / 0.5 / 1 GeV.  Errors 0.017-0.038,
so every bin is within 2.2 sigma of 1.

At 87 deg (cos Z 0-0.1, two 30-deg azimuth bins on each side; the engine at
cos Z 0.05):

* **East** (site R_c ~ 42 GV): MC/engine 0.78-1.27, errors 0.10-0.25, every
  bin within 1.6 sigma of 1 -- passes, weakly constraining.
* **West** (site R_c ~ 7.5 GV): **the MC exceeds the engine by 8-22% at
  0.5-1 GeV**. E.g. antinumu 1 GeV 1.145 +- 0.023 and 1.148 +- 0.022
  (6 sigma); nue 0.5 GeV 1.218 +- 0.072 and 1.144 +- 0.107; numu 1 GeV
  1.085 +- 0.040 and 1.099 +- 0.040.  At 0.3 GeV it is consistent with 1
  (0.80-1.00, errors 0.05-0.18).

**Gate C3: passes in the vertical and the East, FAILS in the West at
0.5-1 GeV.**  The disagreement grows as the cutoff falls (vertical 11.5 GV:
<= 2%; West 7.5 GV: ~13%). It therefore points at primaries of roughly 7-12 GV
near the horizon.  Candidates, not yet separated:

* (a) the cutoff definition: the MC's `v` is sharp at the site map's `R_U`,
  and the gate interpolates the engine's `G_s` at the same map value. The
  West penumbra is the widest.
* (b) the 1D yields of 4-12 GeV/nucleon primaries at the horizon: chromo
  DPMJET pools vs the MCEq-database matrices, including the database's
  cross-section defects;
* (c) the H3a implementation at low rigidity (`primaries.py` vs MCEq's
  `HillasGaisser2012`).

A first discriminant needs no new MC: compare the MC's no-cutoff `t` with
the engine's no-cutoff 1D flux in the same West bins. If those agree, it is
(a) or (c); if they differ, it is (b).

**Impact on sec. 4-5.**  The 3D amplitudes do not use `v`.  The muon-field-off
MC reproduces the engine's 3D a1/a0 (chi2/12 = 1.03, sec. 5), so whatever
inflates the 1D West transmission is not visible in the 3D amplitude at the
pilot's precision.  It also cannot produce the species fan-out, which is
charge-odd and switches on and off with the muon field alone.  It does bear
on the absolute 3D/1D tables (sec. 4.5) and must be resolved before
production.

---

## 4. The East-West pilot

**Run.**  1.444e6 primaries (H3a, five groups), Neyman allocation over
(species, stratum) from `a1rot` (sec. 2.6, `alloc_pilot.npz`).  Every primary
below 65 GV is re-simulated per azimuthal copy with its own bent primary;
above 65 GV the cascades are reused by rotation.  Muon segments are re-transported
exactly per copy.  Fixed SIBYLL pools.  **58.2 core-h**: 1.93 h wall on
34 cores, 7.1 showers/s/core in phase 1, 7.1e5 copy states back-traced in
phase 2 (298 s).  Honda 2011 is band-integrated on the same bins.  The engine
is the delivered factorised `mceq3d` (GSF, `muon_bending=True`, prod-point
maps).  Errors are stratified jackknifes, G = 20.  Full tables in
`results/m3b/pilot_ew.txt` and `pilot_ew_summary.npz`.

### 4.1 Dipole amplitude a1/a0, pooled over cos Z 0-0.3

| E [GeV] | | nue | antinue | numu | antinumu |
|---|---|---|---|---|---|
| 0.3 | **MC** | **0.602 +- 0.021** | **0.299 +- 0.015** | **0.386 +- 0.012** | **0.531 +- 0.009** |
| | Honda | 0.583 | 0.315 | 0.400 | 0.525 |
| | engine | 0.460 | 0.414 | 0.434 | 0.444 |
| 0.5 | **MC** | **0.620 +- 0.015** | **0.273 +- 0.044** | **0.360 +- 0.014** | **0.577 +- 0.016** |
| | Honda | 0.643 | 0.310 | 0.410 | 0.569 |
| | engine | 0.489 | 0.422 | 0.448 | 0.465 |
| 1.0 | **MC** | **0.607 +- 0.030** | **0.167 +- 0.048** | **0.308 +- 0.018** | **0.584 +- 0.108** |
| | Honda | 0.615 | 0.226 | 0.332 | 0.518 |
| | engine | 0.466 | 0.379 | 0.403 | 0.424 |

χ²/12 on the pooled table, using MC errors only (Honda's MC errors are unknown and not
included): **MC vs Honda 1.95**, engine vs Honda 38.  χ²/36 over the
individual cells (3 zenith bands): MC vs Honda 2.14, engine vs Honda 18.8.
Per species, χ²/9 vs Honda: nue 1.07, antinue 1.39, numu 3.43, antinumu 2.67.

**The species fan-out is reproduced.**  The ordering nue > antinumu > numu >
antinue is significant. Pooled differences of a1/a0, MC (Honda):

| E | nue - antinumu | antinumu - numu | numu - antinue |
|---|---|---|---|
| 0.3 | +0.071 +- 0.021, 3.4 sigma (+0.059) | +0.146 +- 0.017, 8.6 sigma (+0.125) | +0.087 +- 0.024, 3.6 sigma (+0.084) |
| 0.5 | +0.043 +- 0.020, 2.2 sigma (+0.075) | +0.217 +- 0.020, 10.7 sigma (+0.158) | +0.087 +- 0.044, 2.0 sigma (+0.100) |
| 1.0 | +0.024 +- 0.095 (+0.097) | +0.275 +- 0.115, 2.4 sigma (+0.186) | +0.141 +- 0.056, 2.5 sigma (+0.106) |

In the conventional West/East ratio (two harmonics at geomagnetic W/E),
cos Z 0.1-0.2, 0.5 GeV, MC | Honda | engine: nue 5.41 +- 1.03 | 4.40 | 2.95,
antinue 1.89 +- 0.11 | 1.97 | 2.60, numu 2.16 +- 0.07 | 2.47 | 2.75,
antinumu 3.30 +- 0.18 | 3.50 | 2.82.

### 4.2 Phase and North/South

* **Dipole phase** (pooled, degrees from geomagnetic West). The MC is within
  1-3 sigma of Honda in 11 of 12 entries. Examples at 0.5 GeV: nue -1.5 +- 2.1 (Honda -2.8),
  antinumu -3.2 +- 1.5 (-2.0).  The largest pull is numu 0.3 GeV,
  +0.4 +- 1.9 (Honda +5.7).
* **North/South.** fS (South bins over the azimuthal mean) agrees with Honda
  to 1-3 sigma.  fN is systematically lower than Honda's, by up to 7%: at 0.3 GeV
  nue 0.824 +- 0.026 (0.892), numu 0.938 +- 0.020 (0.998), antinumu
  0.867 +- 0.020 (0.930), each about 3 sigma.  This is not understood yet (sec. 6).

### 4.3 What does not agree

* **numu amplitude 4-12% below Honda**: MC/Honda pooled 0.965 +- 0.031,
  0.877 +- 0.035, 0.929 +- 0.053 at 0.3 / 0.5 / 1 GeV. This drives the larger
  antinumu/numu and nue/numu ratios at 0.5 GeV (1.60 +- 0.07 vs 1.39;
  1.72 +- 0.08 vs 1.57).  Candidate inputs, none tested yet:
  * the primary spectrum (H3a vs Honda's), which Phase 1 found pulls the E-W
    amplitude by +-10-15%;
  * the hadronic model below 80 GeV (DPMJET-III-19.3 vs Honda's JAM/DPMJET-III),
    which sets the pi/mu share of numu;
  * the MCEq-database cross sections the cascade inherits (sec. 0.1).
* **antinue at 1 GeV, 0.74 +- 0.21 of Honda**: 1.2 sigma, statistics.
* **The 1D site-cutoff reference `v`** (the detector-anchored 1D cutoff on the
  copies' straight lines) gives a1/a0 ~ 0.69 for *every* species at 0.3 GeV,
  with a phase of -17 deg.  This repeats Phase 1's detector-anchored result (-12.7 deg) and shows
  why the 1D cutoff picture cannot be used for the azimuthal structure.

### 4.4 Re-anchoring vs re-simulation (the cheap production option)

`a1anc` (1e6 primaries, every rotated copy re-anchored on its own bent primary,
`E^-0.8` allocation, **5.4 core-h**) against the pilot (re-simulated below
65 GV, Neyman, 58.2 core-h). Different seeds, so the comparison is unpaired:

* pooled 12 values: mean (pilot - a1anc) = -0.022, chi2/12 = 1.04;
* 36 cells: mean -0.027, chi2/36 = 1.52.

So re-anchoring has no bias visible at this precision. The -0.02 mean
leaves room for a positive second-order bias of re-anchoring up to ~0.03.
**Precision per core-hour**: the pooled variances of `a1anc` are 8-12 times the
pilot's at 10.8 times less CPU. So re-anchoring with a *flat* allocation already
matches re-simulation with Neyman per core-hour. Combined with Neyman it should
gain the factor ~5 of sec. 2.2.  A **paired** re-anchor vs re-simulate
test on the same seeds is needed before production uses it (sec. 7).

### 4.5 Azimuth-averaged 3D/1D (the daemonflux correction)

`results/m3b/pilot_full.txt`, last section. Ten cos Z bins, 0-1.

**3D over the site-cutoff 1D (`s/v`)**:
* at the horizon (cos Z 0-0.1): 2.6 / 1.9 / 2.2 / 2.3 (nue / antinue / numu
  / antinumu) at 0.3 GeV and 1.3-2.0 at 0.5 GeV. This is the 3D horizon
  enhancement.
* cos Z 0.1-0.2: 1.14-1.42 at 0.3 GeV.
* consistent with 1 within ~5% for cos Z > 0.4, at every energy.

**3D over the no-cutoff 1D (`s/t`)**: the full multiplicative correction to a
1D flux.

**Statistics.** A few bins carry a single heavy-weight primary: errors
0.11-0.27, values off their neighbours by 30-40%, e.g. numu 0.3 GeV cos Z
0.4-0.5, 0.361 +- 0.141.  The jackknife catches them. A production table
needs either more primaries in the strata that make them or a weight cap
that is checked for bias.

---

## 5. Mechanism: the fan-out is the muon bend

Two runs, paired with `a1anc` (same seeds, tasks and allocation, 1e6 primaries,
re-anchored): `a1nomu` (muon field off, 3.1 core-h) and `a1nobend` (primary
bend off, 3.7 core-h).  Pooled cos Z 0-0.3 a1/a0. The errors are unpaired and
therefore conservative, since the runs share seeds:

| E | run | nue | antinue | numu | antinumu |
|---|---|---|---|---|---|
| 0.3 | all on (`a1anc`) | 0.613 +- 0.069 | 0.307 +- 0.041 | 0.368 +- 0.034 | 0.536 +- 0.031 |
| | muon field off | 0.421 +- 0.030 | 0.301 +- 0.053 | 0.428 +- 0.038 | 0.444 +- 0.025 |
| | primary bend off | 0.398 +- 0.108 | 0.245 +- 0.049 | 0.310 +- 0.023 | 0.440 +- 0.017 |
| | engine | 0.460 | 0.414 | 0.434 | 0.444 |
| 0.5 | all on | 0.702 +- 0.052 | 0.260 +- 0.047 | 0.431 +- 0.034 | 0.565 +- 0.046 |
| | muon field off | 0.451 +- 0.061 | 0.465 +- 0.047 | 0.484 +- 0.021 | 0.430 +- 0.041 |
| | primary bend off | 0.551 +- 0.054 | 0.263 +- 0.060 | 0.286 +- 0.040 | 0.450 +- 0.024 |
| | engine | 0.489 | 0.422 | 0.448 | 0.465 |
| 1.0 | all on | 0.742 +- 0.046 | 0.179 +- 0.089 | 0.300 +- 0.035 | 0.578 +- 0.042 |
| | muon field off | 0.431 +- 0.097 | 0.408 +- 0.074 | 0.366 +- 0.037 | 0.423 +- 0.038 |
| | primary bend off | 0.581 +- 0.039 | 0.145 +- 0.069 | 0.264 +- 0.035 | 0.426 +- 0.029 |
| | engine | 0.466 | 0.379 | 0.403 | 0.424 |

χ²/12 on the pooled table (MC errors):

| run | vs engine | vs Honda |
|---|---|---|
| muon field off | **1.03** | 8.4 |
| primary bend off | 8.7 | 8.1 |
| all on | 10.1 | **1.23** |

* **With the muon field off, the four species collapse onto a common amplitude.**
  The MC then reproduces the delivered engine to chi2/12 = 1.03.  Everything
  else in the engine (the production cone, the prod-point cutoff, the joint
  integral, the primary spectrum) is therefore consistent with the MC at the
  E-W amplitude level. **The engine's defect is its muon-bending term.** Its
  `muon_bending=True` shifts the decay-cone axis by the mean bend
  (`diag_ew_charge_fourier.model_flux`), and that is not what the bend does in
  the full 3D transport.
* **The splitting is symmetric about the no-bend value**: (nue + antinue)/2 =
  0.46 / 0.48 / 0.46 against 0.44 / 0.46 / 0.42 for the engine. The ordering
  follows the mu+ minus mu- content of each species: nue is all mu+, antinumu
  half mu+ and half pi-, numu half mu- and half pi+, antinue all mu-.  This is
  the charge-dependent muon-bending E-W mechanism (Lipari 2000).
* **The primary bend leaves the ordering intact but raises every amplitude
  coherently**, by 15-30% (e.g. antinumu 0.440 -> 0.536 at 0.3 GeV, nue 0.581 -> 0.742 at 1 GeV).  This is the same
  effect gate C5 measured as the -0.15 bias of reusing cascades by rotation.
* **Open: which part of the muon bend matters.**  The bend per lifetime is only
  5.35 deg at 0.5 G (energy-independent), and Phase 1 excluded the exponential
  bend distribution and energy-loss enhancement *as cutoff-axis shifts*.
  Hypothesis: near the horizon, the bend of an East- or West-going muon in the
  mostly northward field is largely **vertical**. It moves the daughter
  neutrinos across the cos Z 0-0.3 bins and across the horizon, with the sign
  set by charge x direction, independently of the cutoff.  This is cheap
  to test (sec. 7): a run with the cutoff off and the muon field on, all
  copies tallied.  The `n` tally (no cutoff) of the pilot keeps copy 0 only
  and is too noisy to decide.

---

## 6. Conclusions

1. **The genuine 3D MC reproduces Honda's East-West species pattern**
   (nue > antinumu > numu > antinue, amplitudes and phases) at 0.3-1 GeV near
   the horizon: χ²/12 = 1.95 on the pooled table and 2.14/cell over 36 cells, MC errors
   only.  The factorised engine gives χ²/12 = 38. The pattern Phase 1 classified as "ours" is
   now explained: **it is the muon bend in the full 3D transport**, which the
   engine's mean-axis-shift term does not capture.
2. Super-K 2016 preferred Honda over the engine by ~2 sigma (Phase 1,
   `SK_EW_NOTES.md`). The MC now sits with Honda, so the comparison should be
   redone with MC tables.
3. **Residual differences with Honda**:
   * the numu amplitude is 4-12% low;
   * fN is ~6% low at 0.3 GeV.
   Both are 3-4 sigma and of the size the primary-spectrum and hadronic-model
   systematics of Phase 1 produce. Neither is classified yet.
4. **Gate C3 fails in the West** at 0.5-1 GeV: the MC's 1D cutoff transmission
   is ~13% above the engine's at R_c ~ 7.5 GV (sec. 3.1). This is unexplained.  It does
   not touch the fan-out, but it must be resolved before any absolute 3D/1D
   table is delivered.
5. **Cost.** The pilot's pooled errors are 0.01-0.02 at 0.3-0.5 GeV (2-5%) for
   58 core-h. Production with re-anchoring plus Neyman should cost about 5x less per unit
   variance (sec. 4.4). That is pending the paired bias test.

## 7. Next steps (proposed)

0. **Gate C3 West** (sec. 3.1): the no-cutoff `t` vs the engine's no-cutoff 1D
   flux in the West bins (no new MC), then whichever of (a)-(c) it points to.
   This blocks the production 3D/1D tables, not the E-W conclusions.
1. **Paired re-anchor vs re-simulate test** (same seeds, ~10 core-h). If it
   passes, the production E-W table uses re-anchoring.
2. **Muon-bend mechanism run**: cutoff off, muon field on, all copies (~5 core-h).
   It decides between the vertical-migration and cutoff-axis pictures. That in turn
   decides whether the engine can be repaired analytically or must take the
   MC's per-species azimuthal response as a table.
3. **Input scan on the residuals**: GSF and Gaisser-Honda primaries in the MC
   on the pilot allocation (2 x ~20 core-h with re-anchoring), aimed at the
   numu and fN differences.
4. **daemonflux integration**: MC ratio tables (3D `s` over 1D no-cutoff `t`) per
   species, E, zenith and azimuth, delivered through the `GeomagneticModel` plug-point.
5. MCEq database v1.4.2 with the 56 GeV hole patched (PM recommendation to the
   PI, `tools/mceq3d/MCEQ_DB_HOLE.md`).  It touches the MC only through the
   interaction lengths (sec. 0.1).

## 8. Tests

`test_m3b.py`: 15 tests, covering the gates C1 and C2, pending-test leakage,
bend vs `advance_grammage`, uniform-field rotation, muon transport vs
`shower.transport_muon`, 5.354 deg per lifetime, Michel vs `decays` and the
decay-frame covariance.  Whole `tools/mc3d` suite: **90 passed** (2026-10-16, 9.5 min).

---

## 9. Reproducing

```sh
source /home/pigrange/mc3d_work/runs/m3b/env.sh      # PY, R, W, M; pylib_fix + pools_fix
cd $M/work                                            # chromo writes tables.dat into the cwd
SK=$R/tools/mc3d/sky3d.py
# primary-bend study (sec. 2.2), seconds
$PY $R/tools/mc3d/study_m3b.py bend --outdir $M/gates/bend --n 3000
# allocation pilot + its re-anchored twin (sec. 2.2, 2.6): 1e6 primaries each,
# ~4.5 core-h each (12 min on 34 cores, unloaded)
C="--nproc 34 --cache $MC3D_POOL_CACHE --chunk 10000 --nstrata 7 --outdir $M/sky --work $M/work --seed 31001"
$PY $SK --nshower 1000000 $C --tag a1rot --r-thr 0
$PY $SK --nshower 1000000 $C --tag a1anc --r-thr 0 --reanchor
$PY $M/scripts/bias.py                    # gate C5 -> reports/bias_c5.txt
$PY $M/scripts/make_alloc.py 65 55        # task B -> reports/alloc_thr65.{txt,npz}
# the East-West pilot (sec. 4): Neyman allocation, re-simulation below 65 GV
$PY $SK --nshower 0 --alloc $M/reports/alloc_pilot.npz --nproc 34 \
    --cache $MC3D_POOL_CACHE --chunk 1600 --outdir $M/sky --work $M/work \
    --seed 20261012 --tag pilot --r-thr 65
$PY $M/scripts/final_report.py pilot      # -> reports/pilot_ew.{txt,npz}
$PY $R/tools/mc3d/analyse_m3b.py $M/sky --tag pilot \
    --engine $M/work/engine_ew_gsf.npz --out-txt $M/reports/pilot_full.txt
# the factorised engine on Honda's bins (read-only use of tools/mceq3d; its
# cache copied to $M/work/engine_cache) and its G_s for gate C3
$PY $M/scripts/engine_ew.py gsf ; $PY $M/scripts/engine_gs3.py
$PY $M/scripts/gate_c3.py pilot                # -> reports/gate_c3_pilot.txt (~1 h: engine at cos Z 0.05)
# mechanism decomposition (sec. 5), paired with a1anc: ~3-4 core-h each
$M/scripts/run_decomp.sh
for t in a1anc a1nomu a1nobend; do $PY $M/scripts/final_report.py $t; done
$PY -m pytest $R/tools/mc3d -q
```

Every job resumes from its task files when re-launched with the same
arguments (phase-1 `c<i>.pkl` and phase-2 `t<i>.npz` are both skipped when
present).  The scripts above live in `$M/scripts/` (copied to
`results/m3b/scripts/`).
