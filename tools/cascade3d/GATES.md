# Pre-registered gates, Phase 3 (`cascade3d`)

Stages C0 and C1. Written 2026-11-18, before any Phase 3 code or run exists.
The plan is `PLAN.md`; this file fixes what each gate measures and when it passes.

## Rules

1. A gate is written here before the run that tests it. The commit of this file is the
   time stamp.
2. A failed gate is reported as failed. No gate is redefined after its run without the PI.
3. Each gate has one script that evaluates it and writes one result file. The result file
   quotes the gate text, the numbers, PASS or FAIL, and the exact command.
4. **Freeze before comparison.** The solver's numerical settings (grids, L, sweeps, ray
   sampling, kernel sets) are written to `RESULTS_C1.md` after G1.0 to G1.2 and before the
   solver's 3D output is compared with the Monte Carlo for the first time. Any later change
   is listed there with its reason and the gate is re-run in full.
5. The comparison with the Monte Carlo (G1.3) is run by an agent that did not write the
   solver.

Notation: PHYS and REF are the two populations of `tools/mceq3d/INTEGRATION.md`.
σ_MC is the Monte Carlo statistical error from the tallies' per-primary squared sums,
cross-checked against the spread over independent seeds.

## C0. Foundations

**G0.1 MCEq blocks (gated).** The interaction and decay matrices rebuilt from the extracted
per-(parent, child) blocks equal MCEq's `int_m` and `dec_m`: largest absolute difference
divided by the largest absolute entry ≤ 1e-12, for PHYS and for REF.
The result file names the database file that MCEq actually loaded.

**G0.2 Decay kernels (gated for two-body).** For π± → μν and K± → μν, the l = 0 part of our
decay kernel, binned on MCEq's grid, against MCEq's decay block: for parent energies from
0.1 GeV to 1 TeV, every entry holding more than 1% of its column's sum agrees within 0.5%.
Three-body kaon decays and muon decay: the same table is reported, not gated (flat Dalitz
plots are a known approximation of `tools/mc3d/decays.py`).

**G0.3 Generator yields against MCEq's (reported).** For each projectile class and
secondary, the l = 0 generator kernel against MCEq's block, as spectrum-weighted moments
(Z factors) at projectile energies 3, 10, 30, 100, 1000 GeV. Not gated: this is the known
generator-versus-database difference.

**G0.4 Kernel statistics (reported).** For each (projectile, energy) node, moments from two
disjoint halves of the events; the distribution of the half-difference in units of its
expected error. Feeds G1.4.

## C1. Field off, azimuthally symmetric

**G1.0 JAX set-up (gated, before any physics gate).**
(a) A test fails if 64-bit mode is off.
(b) Measured CPU time over wall time of a solve does not exceed the number of pinned cores.
(c) `jax.jvp` of the straight-ahead solve with respect to one kernel block agrees with a
central finite difference to 1e-6 relative.

**G1.1 Straight-ahead limit against MCEq (gated).** Same database, population, atmosphere,
primary model and observation level in both. Quantities: ν_μ, anti-ν_μ, ν_e, anti-ν_e and
μ+ + μ− at the ground. Energies: every MCEq bin from 0.1 to 100 GeV. Directions:
cos Z = 1, 0.75, 0.5, 0.25, 0.15, 0.05.
- Energy loss off in both: every cell within 0.3%.
- Energy loss on in both: every cell within 2%.

**G1.2 Numerical convergence (gated).** Doubling, one at a time, the altitude grid, the μ
grid, the Legendre order L and the ray sampling moves no G1.1 or G1.3 cell by more than 0.5%.
The same-bin sweep residual is below 1e-4 in every energy bin.

**G1.3 Three-dimensional transport against the Monte Carlo.**
Reference: a field-free run of `tools/mc3d/sky3d.py` (`--no-cutoff --no-bend-primary
--no-muon-field`), about 10^7 primaries, down-going tally `s`. Cells: the four neutrino
species × the Monte Carlo's energy bands inside 0.1 to 10 GeV × its ten cos Z bins from
0 to 1. The solver is averaged over each cell.

Refinement of the plan's wording, made here before any run. The plan compares "3D/1D"
with the Monte Carlo's `s/t`. The Monte Carlo's `t` is not a one-dimensional cascade: it is
the three-dimensional cascade with its neutrinos laid on the primary's line. So the gate is
put on the 3D flux itself, with like-for-like yields:

- **(a) Zenith shape, like for like (gated).** Solver with kernels built from the Monte
  Carlo's own event pools (the directory the reference run reads,
  `/home/pigrange/mc3d_work/pools_fix`; see Corrections), so both use the same events.
  r = Φ_solver / s_MC per cell, divided by its solid-angle-weighted mean over the ten cos Z
  bins of the same species and energy band. Pass: at least 95% of cells within
  max(2%, 2σ_MC) of 1; and for each cos Z row, the mean over species and bands below 1 GeV
  within max(1%, 2σ_MC of that mean) of 1.
- **(b) Level, like for like (reported, with a trigger).** The mean of r before
  normalisation, per species and band. A deviation above 5% triggers one diagnosis cycle
  before decision point 1.
- **(c) Kernel statistics (reported).** (a) and (b) repeated with the 10^5-event kernels.
  The difference from the pool kernels measures the effect of the 2000-event pools on the
  Monte Carlo.
- **(d) Plan wording (reported).** Solver 3D over its own straight-ahead limit, against
  `s/t`, with the caveat above.

Reported cell by cell in every case: rows cos Z 0 to 0.1 and 0.1 to 0.2 below 0.4 GeV (the
engine's E_off v4 is about 8% high there) and the band 0.10 to 0.16 GeV (engine 5 to 9% low).

**G1.4 Kernel statistics in the flux (gated).** Kernels from two disjoint halves of the
10^5-event samples give fluxes within 0.5% in every G1.3 cell. If not, the nodes responsible
are regenerated at 10^6 events and the gate is re-run.

**G1.5 Cost (gated).** One C1 solve, all species, measured: at most 10 core-minutes and 8 GB.

## Decision point 1

After C1 the PI decides on C2. G1.1 or G1.3(a) failing after one diagnosis cycle stops the phase.

## Corrections

- 2026-11-18, before any G1.3 run or solver output existed. G1.3(a) named the pool
  directory `/home/pigrange/mc3d_work/pools`. That is the pre-fix pool set (above 80 GeV
  its n, pbar and nbar events were generated as protons, `tools/mc3d/MILESTONE3_RESULTS.md`
  section 0.1). The reference run, like every physics run since 2026-10-11, reads
  `pools_fix`. The gate's content, "the Monte Carlo's own event pools", is unchanged; only
  the path is corrected.
- 2026-11-18, same status. The reference run uses production B's primary model
  (`--model GSFG:GH`) and `--thresholds chain`. The solver run of G1.3 uses the same
  primary spectrum, taken from `tools/mc3d/primaries.py`. G1.1 keeps H3a in both MCEq and
  the solver; the primary model only sets the initial condition.
- 2026-11-19, after the reference run and before any solver output was compared with it.
  The Notation section takes σ_MC from the per-primary squared sums, "cross-checked against
  the spread over independent seeds". The cross-check fails: the squared-sum error is 3.4
  times smaller than the spread over the ten runs, because the 12 azimuthal copies of a
  primary are correlated. σ_MC of a cell is therefore the chunk-spread estimate
  (`*_err_task` in `c1/mcref/mcref_fieldfree.npz`; its ratio to the seed spread is 0.99 and
  the ten runs scatter about their mean with rms 0.98 in its units). For the row means of
  G1.3(a), σ is the spread over the ten runs of the row mean itself, divided by √10, which
  includes the correlation between cells. Both are wider than the squared sums would give;
  they were fixed without sight of any solver output.
- 2026-11-19, same status. For like-for-like in G1.3 the solver's initial condition is the
  reference run's: GSF proton and neutron fluxes (`primaries.py`, `GSFG:GH`, energy
  convention "total"), kinetic energy per nucleon from 1 to 1000 GeV, nothing outside.
- 2026-11-19, same status. The range in the previous entry is wrong: the reference run
  covers kinetic energy per nucleon from 1 to 3162 GeV. `--nstrata 7` takes the first
  seven strata of `sky.STRATA = logspace(0, 4, 9)`, whose upper edge is 10^3.5
  (`tools/mc3d/sky.py`, `build_tasks`; confirmed by calling it). `PLAN_CONFIG.txt` and
  the reference's report say 1 to 1000; the run itself is what the source says. The
  gate's content, the reference run's own initial condition, is unchanged: the kernel
  sets use 1 to 3162 GeV. Found by the kernel-set builder in the source, without sight
  of any Monte Carlo output. Cutting at 1000 GeV would have lowered the solver by 1.8
  to 3.5% at 3 to 10 GeV (builder's measurement, straight ahead).

## C1 closing cycle

Written 2026-11-20, before any run of the cycle (decision of the PI of 2026-11-20, last
section). The comparison is no longer blind: the solver's cells were compared with the
reference on 2026-11-19. What follows is fixed before the changed code or the changed
kernel sets exist.

**Changes to the frozen state (rule 4), and nothing else.**
1. `solver._interp_log`: where the four stencil values span more than a factor 1e3, the
   log-cubic value is bounded by the smallest and the largest of them
   (`c1/diag_straight/proposed.diff`, as proposed on 2026-11-19).
2. Monte Carlo-matched kernel sets: the cross section of the nucleons, which in the
   Monte Carlo turns on inside the bin of 1.26 to 1.58 GeV. The treatment is derived
   from the Monte Carlo's own interaction-length function and from the distribution
   of the nucleons inside the bin, not from any tally and not from the size of the
   offset seen in G1.3(b). Its check needs no tally: the probability that a nucleon of
   that bin interacts over X g/cm2, kernel set against the direct integration of the
   Monte Carlo's rule, for X from 100 to 10 000 g/cm2 (reported).

Grids, L, sweeps, ray sampling, cell averaging and the reference stay as frozen on
2026-11-19. A gate script changes only in its paths, and `g1_3.py` in the count below.

**Gates.** G1.0, G1.1 and G1.2 are re-run in full, as written, on the changed code. The
new state (commit, kernel sets with their hashes) is then written to `RESULTS_C1.md`,
and G1.3, G1.4 and G1.5 are re-run in full on it, by an agent that wrote neither the
solver nor the kernel sets. All are evaluated as written, with one amendment. No change
of this cycle addresses the failures of G1.1 and G1.2 (ground muons); they are reported
as they come out.

**G1.3(a), amended.** r and its normalisation as written. Pass: at least 373 of the 400
cells within max(2%, 2σ_MC) of 1; and the row-mean condition as written. 373 is the
count that an exactly right solver reaches or exceeds in 95% of the cases with this
reference. It is computed from the reference alone (`c1/close/manager/null_count.py`:
independent Gaussian errors of size σ_MC, 200 000 samples, P(count < 373) = 0.050,
P(count < 380) = 0.36). Resampling the ten runs instead gives a wider distribution and
a lower number (344); the stricter one is taken. The count is also reported against the
380 of the original text. The amendment is made with knowledge of the first result (377
on the state of 2026-11-19).

Reported with G1.3, not gated: the chi2 of the 400 cells, and the part of the shape
difference that is common to the ten runs with its 2σ bound, both as in part A of the
diagnosis cycle.

**Test of the candidate for the level above 0.4 GeV (reported, not a gate).** Candidate:
the energy discretisation of the kernels (parent at the bin centre; `LIKE_FOR_LIKE.md`,
section 3, item 2). Variant kernel sets are built for this test only and used for no
gate: the parent of the interaction and decay channels is spread over its bin with a
weight T^-γ, (A) γ = 2.7, (B) γ = 1, (C) γ per species and bin from the slope of the
production spectrum that the kernel set itself gives for the primary spectrum in use.
Straight-ahead solves, variant against the changed set, for the full primary spectrum
and for the strata of 3.16 to 10, 10 to 31.6 and 31.6 to 100 GeV per nucleon. The
builder reports the change of the flux without sight of any tally. The manager compares
it with the excess over `t` (bands from 0.4 to 10 GeV, cos Z 0.1 to 1, the four species;
in part C: full spectrum +1.74 +- 0.20%, strata +2.2 +- 0.5%, +1.5 +- 0.4%, +0.9 +-
0.5%), re-measured on the changed set with the script of part C. Wording fixed in
advance, on variant (C) and the full spectrum: "explained" if the excess left is within
0.6% of zero; "not explained" if more than two thirds of it is left; "partly explained"
in between. The strata and the variants (A) and (B) are reported next to it. A variant
that cannot be built within the budget of the cycle is reported as not built, and the
candidate stays untested to that extent.

Correction of 2026-11-20, made before any solve on the changed kernel sets was compared
with a tally. The +1.74 +- 0.20% quoted above for the full spectrum was computed with
nine of the ten runs; the three strata values are from all ten. The same statistic with
the ten runs gives +1.52 +- 0.20% on the state of 2026-11-19
(`c1/close/manager/level_stat.py`, which fixes the statistic: per band the flux summed
over the four species and cos Z 0.1 to 1, solver over `t`; mean over the seven bands
from 0.4 to 10 GeV weighted by the inverse variances, the bands taken as independent, so
that its error is a lower limit). The wording does not depend on that number: it refers
to the excess re-measured on the changed set.

**After the cycle** the manager reports to the PI. The recommendation given before the
cycle stands: stop the phase if G1.3(a) as amended fails, or if the level above 0.4 GeV
is "not explained". Compute: about 15 core-h, no new Monte Carlo run.

## Results log

Result files are under `/home/pigrange/mc3d_work/runs/native3d/` (`c0/gates/`, `c1/gates/`).

- **G0.1: PASS** (2026-11-18). Rebuilt `int_m` and `dec_m` equal MCEq's exactly (relative
  difference 0), PHYS and REF. Command: `gates/g0_1.py`.
- **G0.2: FAIL** (2026-11-18, reproduced twice). 1502 of 4678 gated entries differ by more
  than 0.5%, up to 100%. All failures are at low parent energy: pions above 10 GeV and
  kaons above 100 GeV pass (largest deviation 0.036% and 0.055%).
  Diagnosis: MCEq's decay tables treat the parent as ultra-relativistic (momentum set equal
  to total energy, ultra-relativistic muon polarisation) and use rounded masses; our kernels
  use exact kinematics, as the Monte Carlo does. With MCEq's approximations switched on in
  our code, 0 of 4572 analytic entries differ by more than 0.5% (largest 0.012%).
  Independent check: the plateau of MCEq's π+ → ν_μ column gives a parent momentum equal to
  T + m to four digits at every energy, 20% above the true momentum at T = 0.11 GeV and
  0.9% above at 0.89 GeV.
  The gate is not redefined. How it is treated is the PI's decision. Command: `gates/g0_2.py`.
- **G0.3: reported** (2026-11-19, kernel set `gen_fresh1e5`, 10^5 events per node, 708
  nodes). 553 channel rows with Z above 0.001: median generator/MCEq 0.969, 67% within 10%,
  81% within 20%. At 3, 9, 28 and 890 GeV the main nucleon and pion channels agree within
  1 to 6%. At 89 GeV, next to the generator switch at 80 GeV, they differ by up to 17%
  (π → π) and 40% (π → nucleon); the database file in use is interpolated across that
  region. The pool kernels agree with the 10^5-event kernels within their errors (pull rms
  1.005). Command: `gates/g0_3.py --tag fresh1e5 --also pool2000`.
- **G0.4: reported** (2026-11-19, `gen_fresh1e5`). Half-sample differences in units of their
  expected error: rms 0.998 to 1.002 for l = 0 to 64 (276 213 entries with at least 200
  particles), fraction beyond 3: 0.0024 to 0.0028, per-node rms from 0.87 to 1.11. The
  quoted kernel errors are therefore reliable. Median relative error of such an entry:
  2.4% (l = 0), 2.6% (l = 1). Command: `gates/g0_4.py --tag fresh1e5`.
- **G1.0: PASS** (2026-11-19). (a) The test fails when 64-bit mode is off. (b) CPU time
  over wall time 1.77 on 2 pinned cores. (c) `jax.jvp` against a central finite difference:
  1.2e-7 relative. Command: `gates/g1_0.py`.
- **G1.1: FAIL** (2026-11-19, PHYS and REF alike; solver settings of `DESIGN.md`).
  Neutrinos: all 720 cells of each case inside the tolerance (worst −0.28% with loss off,
  ν_e at 89 GeV; −0.30% with loss on, ν_e at 4.5 GeV).
  Muons, loss on (2%): 1 cell of 180 outside, +2.04% (PHYS) and +2.03% (REF), vertical,
  0.141 GeV. Muons, loss off (0.3%): 107 cells of 180 outside, in every direction, worst
  +101% at cos Z = 0.05, 0.112 GeV (+2.3% at the vertical). Command: `gates/g1_1.py`.
  Diagnosis cycle (one, by the solver's author; `c1/WP12_REPORT.md`): MCEq's references are
  not converged in MCEq's own step size, and the solver is compared with that step error.
  Independent check by the manager, with MCEq's own solver and its two step controls
  divided by 4 and 16 (PHYS; cos Z = 1, 0.25, 0.05; `gates/mceq_step_check.py`, result
  files `c1/review/mceq_step_PHYS_*.md`):
  - the stored references sit below their own step limit by 2.3%, 11.0% and 35.2% for
    loss-off muons at 0.112 GeV (cos Z = 1, 0.25, 0.05), by 1.45% in the failing loss-on
    cell, and above it by 0.22 to 0.28% for neutrinos at 89 GeV;
  - against the step limit the solver is within 0.01% (loss off) and 0.10% (loss on) for
    neutrinos, within 0.78% for loss-on muons, and for loss-off muons within 0.02% at the
    vertical and 0.20% at cos Z = 0.25;
  - a solver error remains: loss-off muons at cos Z = 0.05 are above the step limit by 30%
    at 0.112 GeV, 6.5% at 0.18 GeV, 0.8% at 0.36 GeV and less than 0.3% above 0.5 GeV
    (the limit itself is known to 2.6% at 0.112 GeV). It shrinks with a finer altitude
    grid (see G1.2).
  Not checked independently: REF, and cos Z = 0.75, 0.5, 0.15.
  The gate is not redefined. Under decision point 1 the phase stops here until the PI
  decides; nothing of stage B or G1.3 has been run.
- **G1.2: FAIL** (2026-11-19; G1.1 cells only, no G1.3 cell exists yet). Largest move of a
  cell under each doubling, against 0.5%:
  - straight-ahead, loss on: altitude grid 0.02%, ray sampling 0.02% (inside);
  - straight-ahead, loss off: altitude grid −20% (muons, cos Z = 0.05, 0.112 GeV), ray
    sampling 0.07%;
  - synthetic angular kernels (Gaussian p_T of 0.35 GeV, L = 32, loss on): altitude grid
    0.07%, ray sampling 0.70% (muons, cos Z = 0.05, 0.18 GeV), μ grid (line family) −8.4%
    (muons, vertical, 3.5 GeV; neutrinos −6 to −7%), Legendre order L +30% (muons,
    vertical, 8.9 GeV).
  Same-bin sweep residual 4.5e-7 (inside 1e-4). Cause of the two large moves, from the
  author's follow-up runs: near the vertical the ground lines are 0.065 rad apart, too
  coarse for L = 32; with 96 and 192 ground lines the vertical cells move by −6 to −8% and
  then −1.3 to −1.9%. Directions at cos Z ≤ 0.5 are converged to 0.2%.
  Command: `gates/g1_2.py`.
- **Not gates, same date.** Three-dimensional toy problem against direct integration
  (`test_toy3d.py`, reproduced by the manager): worst −0.043% over 11 points, including
  1° above the horizon and up-going directions at 15 and 30 km. Cost of one angular solve
  on the default grid, synthetic kernels: 9.8 core-minutes, 4.7 GB (G1.5, to be measured on
  the real kernel set, allows 10 and 8); with 192 ground lines 17.5 core-minutes, 7.1 GB.
- **Open before G1.3 (2026-11-19; found by the kernel-set builder in the source, nothing
  compared yet).** The Monte Carlo's muon transport (`shower.transport_muon`, and
  likewise `bend.transport_muons`) advances the decay clock of a step with βγ at the
  start of the step, in steps of up to 30 g/cm². A slowing muon ages too slowly, so it
  decays later and at lower energy. Manager's check, the Monte Carlo's own function
  against a fine integration of the same stopping power and atmosphere
  (`gates/mc_muon_clock_check.py`, `c1/review/mc_muon_clock.md`): neutrinos above
  0.1 GeV per muon, Monte Carlo below exact by 1.2, 1.8, 2.9 and 3.9% for vertical muons
  of kinetic energy 1.2, 0.6, 0.3 and 0.15 GeV starting at 15 km; by 2.1, 3.4, 6.0 and
  9.2% starting at 8 km; by 0.2 to 0.5% starting at 25 km; by 0.5, 1.0, 2.0 and 3.5% at
  15 km and cos θ = 0.3. The effect on the reference flux is not measured. The builder
  estimates the reference 1 to 4% low at 0.1 to 1 GeV, with a zenith dependence of up to
  1%, the size of the row tolerance of G1.3(a). The reference is compared with nothing
  until the PI has decided how to treat this.
- **Kernel sets for G1.3 and G1.4 (2026-11-19, not a gate).** `kernelset_mc.py` builds
  them from the generator and decay kernels with the Monte Carlo's own functions for
  rates, thresholds, stopping power and primaries; rule-by-rule table in
  `c1/LIKE_FOR_LIKE.md`. Its 23 tests pass (re-run by the manager). The independent
  dense straight-ahead reference `gates/ref1d_dense.py` agrees with step-converged MCEq
  within 6.2e-5 for neutrinos (0.1 to 100 GeV, cos Z 1, 0.25, 0.05). Straight ahead, the
  two half-sample sets differ by up to 0.6% per bin at 1 to 10 GeV (G1.4 allows 0.5% per
  cell; not yet run).
- **Monte Carlo muon clock, acceptance met (2026-11-19, not a gate; decision of the PI
  below).** `tools/mc3d` has the opt-in clock `converged`: the step is shortened until
  the muon loses at most 5% of its momentum, the proper time of a step is the mean of
  1/(βγ) for a momentum linear in the path, the decay point inside a step is the inverse
  of the same law, and the energy there gets the mid-point correction of a full step.
  - Acceptance as written, `gates/mc_muon_clock_check.py 4000 converged`: largest
    deviation 0.07% over the 160 entries, against 0.2% (`c1/review/mc_muon_clock_converged.md`).
    Two earlier schemes failed it and the criterion was not changed: trapezoid rule 0.28%,
    logarithmic mean 0.25%. The cause was the decay point, which both still interpolated
    linearly inside the step.
  - Default unchanged. The check table of the default clock is byte-identical to the one
    made with the code before the change (largest deviation 9.19%). A run of 60 000
    primaries with the reference's options gives the same bits with the code at af178e2
    and with the new code and no switch, in every array of the 62 output files; only the
    run-time entries differ (`c1/mcref2/pilot/CMP_before_vs_legacy.txt`).
  - Tests: `tools/mc3d/test_muclock.py` (17: defaults, run signature, the default clock
    pinned to numbers from af178e2, both transports against a fine integration within
    0.2% in path and kinetic energy, scalar against vectorised). All 171 tests of
    `tools/mc3d` pass.
  - Cost, same pilot: 588 CPU s with the switch against 531 without (1.11). The re-run of
    the reference is therefore about 27 core-h, not 25.
- **Second solver cycle: G1.0 to G1.2 re-run in full, as written (2026-11-19; results in
  `c1/cycle2/gates/`, author's report `c1/WP12_CYCLE2_REPORT.md`).** Changes of the cycle:
  extra passes for up-scatter into transported species; Legendre moments of the
  piecewise-linear interpolant in μ at each level; 48 ground lines uniform in zenith
  angle; loss step 5 g/cm² (was 10); an error in the mid-point of the loss step
  corrected; highest bin marched set from the primaries; L = 32. The 18 unit tests pass
  (re-run by the manager), among them the three-dimensional toy (worst +0.05% over 11
  points).
  - **G1.0: PASS.** `jax.jvp` against a finite difference 1.4e-7; CPU over wall 2.60 on
    4 pinned cores.
  - **G1.1: FAIL**, unchanged in substance. Neutrinos inside in the four cases (worst
    −0.28% loss off, −0.30% loss on). Muons, loss on: worst +2.035% (PHYS) and +2.028%
    (REF), vertical, 0.141 GeV, against 2%. Muons, loss off: worst +101% at cos Z = 0.05,
    0.112 GeV, against 0.3%. Not a gate, against step-converged MCEq: neutrinos within
    1.5e-4 (loss off) and 1.1e-3 (loss on), loss-on muons within 0.79%, loss-off muons
    within 0.07% at the vertical, 0.65% at cos Z = 0.25 (0.02% and 0.17% against the
    manager's own step limit) and still +30% at cos Z = 0.05, 0.112 GeV (the limit
    documented under the PI's decision of this date, point 2).
    The header of `G1.1.md` gives the loss step as 10 g/cm²; the run used 5 (the label
    in the script was not updated).
  - **G1.2: FAIL as a whole.** Largest move of a cell under each doubling, against 0.5%:
    - A, straight ahead, loss on: inside (at most 0.02%). Loss off: altitude grid −20.3%
      (muons, cos Z = 0.05, 0.112 GeV), ray sampling 0.11%.
    - B, synthetic angular kernels, G1.1 cells: inside under the four doublings
      (altitude grid 0.03%, line family 0.27%, L 0.05%, ray sampling 0.42%). In the first
      cycle the family and L moved cells by 8% and 30%.
    - C, the Monte Carlo-matched kernel set (`pool2000`). G1.3 cells (four neutrino
      species, ten bands, ten cos Z bins): inside, at most 0.16% (altitude grid 0.02%,
      line family 0.16%, L 0.08%, ray sampling 0.05%). G1.1 neutrino cells: inside, at
      most 0.38% (ν_e, vertical, 71 GeV, under the L doubling). G1.1 muon cells: outside
      at cos Z = 0.05 (altitude grid +0.72%, 3 cells of 180; line family −3.19%, 7 cells;
      L +2.29%, 8 cells; ray sampling +1.07%, 6 cells), up to 0.7 GeV, and in one cell at
      cos Z = 0.15 below 0.2 GeV. Recomputed by the manager from the stored solver
      outputs (`c1/cycle2/g12/C_*.npz`).
    - Same-bin sweep residual 4.5e-7 (1.5e-8 on the real set), against 1e-4.
    What fails is therefore muons at the ground near the horizon below 0.7 GeV, with and
    without loss; every neutrino cell of G1.1 and G1.3 is inside. The gate is not
    redefined.
  - Cost of one solve in the configuration of G1.5 (not the gate's run): 11.9
    core-minutes, 5.0 GB; G1.5 allows 10 and 8.
- **Solver cycle 2b: cost, a diagnostic near the horizon, G1.0 to G1.2 re-run on the
  final code (2026-11-19; results in `c1/cycle2b/`, author's report
  `c1/WP12_CYCLE2B_REPORT.md`).** Decided by the manager inside the PI's "Continue", bounded
  at 8 core-h (used: about 8). No numerical setting was changed.
  - Adopted: the push contraction is restricted to the parents that have channels and the
    children that receive one (the terms dropped are exact zeros). Acceptance, fixed
    before the change: every stored quantity of the four base runs within 1e-9 relative
    of the cycle-2 file, zeros staying zero. Met: worst 1.0e-13 (recomputed by the
    manager); before and after on the same code are identical. The 90 unit tests of
    `tools/cascade3d` pass (re-run by the manager).
  - G1.0 to G1.2 re-run in full on the final code: **G1.0 PASS** (1.4e-7), **G1.1 FAIL**
    and **G1.2 FAIL**, with the numbers of the second cycle. The 19 runs of G1.2 reproduce
    the second cycle's fluxes to 3e-11, except the altitude doubling of part C (8e-8):
    that run of the second cycle was made before the levels below the top were included
    in the doubling (335 levels then, 341 in the committed code). `G1.1.md` now prints the
    settings read from the run's own objects (loss step 5 g/cm²).
  - Diagnostic, not adopted: a ground layer of levels (10, 30, 100, 300 m) on the Monte
    Carlo-matched set, base and the four doublings. It does not bring the muon cells
    inside 0.5%: largest moves at cos Z = 0.05 are +1.74% (altitude grid), −0.76% (line
    family), +0.65% (L), +0.80% (ray sampling), against +0.72%, −3.19%, +2.29%, +1.07%
    on the default grid; and the base itself moves by +14% at cos Z = 0.05, 0.112 GeV.
    The ground muon flux near the horizon below about 0.4 GeV is therefore not converged
    on either grid, with loss on. Neutrinos do not see it: the ground layer moves the
    neutrino cells of G1.1 by at most 4e-6 and the G1.3 cells by 1e-5, and under the
    doublings the G1.3 cells stay within 0.16%. A second candidate (tangent lines passed
    through below 2 km) was not run, for the budget.
  - Cost of one solve in the configuration of G1.5, cold, on four cores while the node
    was shared with the reference re-run: 15.3 core-minutes before the change and 14.0
    after (−8%), 5.0 GB; a second solve in the same process 12.4. The same code before
    the change measured 11.9 in the second cycle on a quieter node. G1.5 is measured as
    written by the agent that runs G1.3; the expectation is that it fails.
- **Freeze (2026-11-19).** The settings and the code state are written to
  `RESULTS_C1.md`, before any solver output is compared with the Monte Carlo. G1.3 to
  G1.5 go ahead with G1.1 and G1.2 on record as failed: the PI's "Continue" covers G1.1;
  G1.2 failing again after it, on ground muons near the horizon only, is carried forward
  on the manager's reading of the same decision and is reported to the PI as such.

- **G1.4, and the evaluation of G1.3 fixed before the reference existed (2026-11-19;
  agent that wrote neither the solver nor the kernel sets; report `c1/WP14_REPORT.md`,
  results in `c1/g13/`).** Done without opening any Monte Carlo tally: the combined
  reference file did not exist yet.
  - **G1.4 PASS**, with little margin: the two half-sample sets differ by at most 0.479%
    in the 400 cells (0.5% allowed; recomputed by the manager from `h0.npz` and
    `h1.npz`). The largest differences are one level per band (anti-nu_e, 2.5 to 4 GeV,
    +0.47 to +0.48% in every cos Z bin). Each half is within 0.24% of the full set. No
    kernel regeneration.
  - `gates/g1_3.py` (sha256 `2c8a2edd55cca63b...`) is committed in this state before the
    reference is combined. It was run on synthetic references built from the solver's
    own cells: it recovers the distortion put in and returns FAIL for a 2% zenith tilt,
    for +8% in the lowest cos Z row below 0.4 GeV and for 1.5% of cell-to-cell scatter.
  - Like for like, from the source with file and line: the tally `s` and
    `assemble.cell_averages` are the same quantity (mean dN/dE over the band, mean over
    2 pi dcos Z, weight 1/cos psi with a floor of 1e-6 only, cap area on the 6371 km
    sphere, total energy, same species order). No conversion factor. Not checked yet:
    the reference's own `meta_*` strings, and its caps against each other.
  - The solve of the G1.3(a) configuration (`c1/g13/pool2000.npz`) is identical in all
    400 cells to `c1/cycle2b/g12/C_base.npz`.
  - 10^5-event kernels against the 2000-event pools (solver only, part of G1.3(c)):
    levels per species and band differ by up to 1.4% (rms 0.5%); after division by the
    cos Z mean, at most 0.36% (rms 0.08%, largest in the lowest cos Z row).
  - **Defect found: `run_solve.py --straight` with the full line family is wrong at the
    horizontal node.** Its cells of the row cos Z 0 to 0.1 below 1.6 GeV are 10^3 to
    10^12 times too large (the ground lines from cos Z 0.033 up are normal; the
    up-scatter passes of that run do not converge: 1, 1, 4e-3). G1.1 does not use this
    path (six lines, no horizontal node), and the 3D solves are not affected (the 3D
    horizontal node agrees with the doubled line family to four digits). Consequence:
    G1.3(d) has no valid number in that row. Not diagnosed yet; nothing changed.
  - The sweep residual of the `h1` solve is 2.6e-5 (the other 3D solves: 1.5e-8 to
    2.9e-8; G1.2 allows 1e-4).
  - Cost of the 3D solves on the shared node: 13.1 to 15.4 core-minutes, 5.1 GB.

- **Straight-ahead defect diagnosed; nothing adopted (2026-11-19; diagnosis by the
  solver's author on reduced problems, record `c1/diag_straight/MANAGER_NOTE.md`,
  written by the manager from the author's messages; exposure of the 3D solve recomputed
  by the manager).**
  - Cause: muons are interpolated in energy with a log-cubic stencil
    (`solver._interp_log`). In the up-scatter passes of a straight-ahead run the solution
    on grazing lines consists only of up-scatter from parents that are extinct after
    about 36 000 g/cm2; neighbouring muon bins then differ by tens to hundreds of
    decades, the cubic extrapolates without bound and the march compounds it. With
    linear interpolation, or with the cubic bounded by its own stencil values, the
    horizontal node agrees with the neighbouring ground line to 3e-4.
  - Extent in the straight-ahead run: the lines tangent up to about 10 km are wrong (at
    29 km: 4e-4); the ground lines from cos Z 0.033 up are right to 2e-4. So its cells for cos Z 0.1 to 1
    stand, and the row cos Z 0 to 0.1 does not.
  - Exposure of the frozen 3D solve, measured with a bound that acts only where the four
    stencil values span more than a factor 1000 (`c1/diag_straight/C_base_wide.npz`
    against `c1/cycle2b/g12/C_base.npz`): ground muons 4.4e-4, nu_e on the ground lines
    2.8e-4, the G1.3 cells 2.5e-4, the zenith-normalised cells that G1.3(a) gates
    1.0e-4 (2% allowed).
  - The fix (`c1/diag_straight/proposed.diff`) is **not adopted before G1.3**: the 3D
    results would not be bit-identical, so rule 4 would require G1.0 to G1.5 to be
    re-run in full, for 1.0e-4 in the gated quantity. G1.3 to G1.5 run on the frozen
    code. Adopting it afterwards, with the full re-run, is put to the PI at decision
    point 1. An unconditional bound is not a correct fix (it caps genuine maxima of the
    muon spectrum: 3.2% in the ground muons).
  - G1.3(d) therefore has no valid number in the row cos Z 0 to 0.1 on the frozen code.
    A supplement, labelled as such and not a gate: one straight-ahead solve with the
    bound, in the diagnostic copy of the code (`c1/diag_straight/straight_wide.npz`).
    Its up-scatter passes converge (0.11, 1.8e-4, 1.4e-9), it agrees with the frozen
    straight-ahead run within 2.5e-4 for cos Z 0.1 to 1, and its ratio to the 3D solve
    in the row cos Z 0 to 0.1 for nu_mu runs from 0.43 at 0.10 to 0.16 GeV to 0.996 at
    6.3 to 10 GeV.
- **Reference re-run complete (2026-11-19, 14:44).** Ten runs of 10^6 primaries with
  `--muon-clock converged`, seeds and settings of the first run, all with exit status 0;
  29.8 core-h (25 quoted; the node was shared), 0.45 GB per process. Combined into
  `c1/mcref2/mcref2_fieldfree.npz` (sha256 `2bc6afe7ceec46f6...`): same arrays as the
  first run plus `meta_muon_clock`, no empty cell in the 10 degree cap, 52 fallback
  (group, stratum) pairs per run as in the first run. The comparison of the two
  references (the clock's effect) is kept with the manager until G1.3 has run.
- **G1.3 and G1.5 (2026-11-19; the same independent agent, after the reference was
  complete; report `c1/WP14_REPORT.md`, results `c1/g13/G1.3.md`, `G1.3.json`,
  `c1/g13/g15/G1.5.md`).** `gates/g1_3.py` ran unchanged (sha256 `2c8a2edd55cca63b...`,
  checked by the manager) on `c1/mcref2/mcref2_fieldfree.npz`, 10 degree cap. The count,
  the row means and the levels below were recomputed by the manager with a separate script
  (`c1/g13/manager/recheck_g13.py`) and agree.
  - **G1.3(a) FAIL.** 377 of the 400 cells are within max(2%, 2 sigma_MC) of 1; 380 are
    needed. The second condition is met: the ten row means below 1 GeV differ from 1 by
    -0.73% to +0.53%, each inside max(1%, 2 sigma). Cells outside, by species (nu_e,
    anti-nu_e, nu_mu, anti-nu_mu): 2, 12, 3, 6; by band from 0.10 GeV up: 3, 4, 2, 2, 0, 1,
    1, 7, 1, 2; by cos Z row from the horizon up: 2, 1, 3, 1, 1, 6, 3, 2, 2, 2. The largest:
    2.51 to 3.98 GeV, cos Z 0 to 0.1, anti-nu_e +46% (3.1 sigma) and anti-nu_mu +42%
    (4.2 sigma). The 2% floor is the tolerance in 46 cells; elsewhere it is 2 sigma_MC
    (median sigma_MC per band: 1.0% at 0.10 GeV to 11% at 6.3 to 10 GeV).
  - **G1.3(b): the trigger fired.** The mean of r - 1 per species and band is above 5% in 4
    of the 40: nu_e at 0.10 to 0.16 GeV, +5.08 +- 0.49%; anti-nu_e at 2.51 to 3.98 GeV,
    +7.4 +- 4.3%; nu_e at 6.31 to 10 GeV, +10.9 +- 8.9%; anti-nu_mu at 6.31 to 10 GeV,
    +6.7 +- 2.7% (errors from the ten runs). The solver is above the Monte Carlo in 36 of
    the 40: by +4.2 to +5.1% in the four species at 0.10 to 0.16 GeV, +2.8 to +3.3% at 0.16
    to 0.25 GeV, -0.5 to +2.2% from 0.25 to 1 GeV (errors 0.2 to 1.4%), and -1.5 to +11%
    above 1 GeV (errors 0.8 to 9%).
  - G1.3(c), reported: with the 10^5-event kernels 379 of 400, ten rows of ten, and the
    trigger fires as well. Levels against the pool kernels: -1.4% to +0.8%; normalised
    shape within 0.36%.
  - G1.3(d), reported: (3D / straight-ahead) against s/t. Frozen code: 349 of 400 inside
    the tolerance of (a); the row cos Z 0 to 0.1 has no valid number (straight-ahead
    defect, entry above); rows 0.1 to 1: mean +0.17%, rms 3.5%. Supplement with the
    bounded interpolation, in the diagnostic copy and not the frozen code: 373 of 400; row
    cos Z 0 to 0.1 below 1 GeV -4.5% on average. With the bound the count of (a) is 377
    as well.
  - Not gates: the 7.5 and 10 degree caps of the reference agree (chi2 467.5 for 400
    cells); no shift by one bin and no factor cos Z between solver and reference (fitted
    power of cos Z -0.006 +- 0.0015); the quadrature of the cell mean in the lowest cos Z
    bin is good to 4e-4.
  - **G1.5 FAIL.** One cold solve of the G1.3(a) configuration in a fresh process on four
    cores: 13.23 core-minutes (user 710.5 s, system 83.6 s; 10 allowed), peak memory
    5.38 GB (8 allowed), 5 min 13 s elapsed. Load average of the node 5.3 before, 9.6
    after. A second solve in the same process: 12.20 core-minutes (not the gate).
    `gates/g1_5.py` is the agent's; the manager corrected its reading of the elapsed time
    (the result file shows "None" there) and checked that the edited script rebuilds the
    stored result file from the stored measurement, that cell and the paths apart.
  - After the run, lines over 88 characters in `gates/g1_3.py`, `g1_4.py` and
    `test_g1_3.py` were wrapped. The syntax trees are identical to the versions that ran,
    two docstrings of `g1_3.py` apart; the quoted gate texts keep their long lines.
    `g1_3.py` is now sha256 `7531abb540b8af99...`; the version that ran is in commit `512a119`.
- **Diagnosis cycle after G1.3, parts A and B (2026-11-19, manager; scripts and outputs in
  `c1/g13/manager/`).** The gate text gives one diagnosis cycle to a failed G1.3(a) and to
  a fired trigger of (b). Nothing was changed in the solver, and no gate is redefined.
  - A, on the count of (a). The error model of the reference holds: the spread of the ten
    runs squared over sigma_MC squared, averaged over the cells, is 0.996 +- 0.024. With
    it, a solver that is exactly right was put through the gate 4000 times: it gives 380.8
    +- 4.8 cells inside for Gaussian errors and 379.5 +- 18 when the ten runs are
    resampled, and it reaches 380 in 62% and 60% of the cases. **The count criterion as
    written by the manager is failed by a perfect solver four times in ten**; 377 is
    inside that distribution (24% and 36% of the cases give 377 or fewer). The chi2 of the
    400 cells is 489.5 against 423 +- 54 expected (p = 0.06 Gaussian, 0.28 resampled); the
    rms of (r - 1)/sigma is 1.105 against 1.03 expected. This says the count does not
    discriminate; it does not turn the verdict into a pass.
  - A, what the data do bound. The part of the shape difference that is common to the ten
    runs (noise subtracted, error by jackknife over the runs): below 1 GeV and cos Z 0.1
    to 1 (180 cells), rms 0.41%, below 1.06% at 2 sigma; with the lowest row (200 cells)
    below 1.3%; the lowest row alone below 3.1%; 1 to 10 GeV below 5.9%, where the noise
    is 7.4%. Per species below 1 GeV, cos Z 0.1 to 1: below 1.2%, 2.2%, 0.9%, 1.4%. The row
    means show a small tilt: +0.3 to +0.5% for cos Z 0.1 to 0.6, -0.4 to -0.7% for 0.6
    to 1.
  - B, on the level of (b). Flux summed over cos Z 0.1 to 1, the four species together,
    solver / Monte Carlo - 1 per band from 0.10 GeV up, in %: the 3D solve against `s`,
    +4.8, +2.9, +1.0, +1.1, +1.4, +2.4, +1.8, +1.2, +1.8, +1.9 (errors 0.2 to 0.5 below
    1 GeV, 0.3 to 2.1 above); the straight-ahead solve against `t`, +4.9, +3.1, +0.7,
    +1.3, +1.5, +1.8, +1.7, +0.9, +1.8, +1.8. Their ratio run by run differs from 1 by
    at most 0.6 +- 0.4%. **The level offset is the same in the straight-ahead and in the
    3D calculation: it is in the yields they share, not in the angular transport, and it
    cancels in a ratio of 3D to straight-ahead.** It is the same for the four species
    ((nu_mu + anti-nu_mu)/(nu_e + anti-nu_e), solver over Monte Carlo: +0.6 +- 1.2% in
    the lowest band), so its origin is upstream of the muon decay, and it depends little
    on cos Z (lowest band: +5.4% inclined, +3.8% vertical).
  - Excluded as the cause: kernel statistics (10^5-event kernels: +4.6% in the lowest
    band), the Monte Carlo's muon clock (the reference has the corrected one), hadron
    energy loss (none on either side), a cause in the muon chain alone (flavour ratio).
    The builder's prior estimates (`c1/LIKE_FOR_LIKE.md`, section 3) do not add up to +5%.
- **Diagnosis cycle after G1.3, part C: cause of the level offset (2026-11-19; the
  independent agent, brief `c1/g13/LEVEL_BRIEF.md`, report
  `c1/g13/diag_level/REPORT.md`; the numbers below recomputed by the manager,
  `c1/g13/manager/check_c_rate.py`, `check_c_var.py`, `check_c_shape.py`).** Nothing
  frozen was changed and no fix was applied.
  - Where, by primary energy. The reference was rebuilt per primary-energy stratum from
    its per-task files (the strata sum to the combined file to 1e-15, in `s` and in `t`;
    all ten runs), and the frozen straight-ahead solve was repeated with the primary
    spectrum restricted to one stratum. Excess of the solver over `t`, cos Z 0.1 to 1,
    the four species, in the bands 0.10 to 0.16, 0.16 to 0.25 and 0.25 to 0.40 GeV:
    primaries of 1 to 3.16 GeV per nucleon +11.6 +- 0.5%, +9.2 +- 0.5%, +3.3 +- 0.8%
    (they make 24%, 17% and 10% of the band); 3.16 to 10 GeV +3.6 +- 0.4%, +2.5 +- 0.3%,
    +1.1 +- 0.5%; 10 to 31.6 GeV +2.2 +- 0.6%, +1.5 +- 0.7%, +0.0 +- 0.5%; 31.6 to 100
    GeV +1.1 +- 0.9%, +0.6 +- 0.9%, -1.3 +- 0.9%. The seven stratum solves add up to the
    full solve within -0.39% to +0.17%: the solve is not exactly linear in the primary
    flux (the interpolations in log space, read from the code, not tested).
  - **Cause in the two lowest bands, measured: the nucleon cross section turns on inside
    one energy bin.** The Monte Carlo interpolates its table (MCEq's, log-log, zero at
    the node 1.12 GeV, full at 1.41 GeV): a nucleon interacts only above a kinetic
    energy of about 1.39 GeV. The kernel set gives the whole bin 1.26 to 1.58 GeV the
    rate of its centre. Over 1030 g/cm2, 58% of the nucleons of that bin interact in the
    Monte Carlo (61% over 3000 g/cm2) and all of them in the solver. Two diagnostic
    straight-ahead solves on a copy of the set, with the interaction yields of protons
    and neutrons in that bin scaled by f = 0.60 and f = 0.54 (rate kept), change the
    flux by -4.7% and -5.4% at 0.10 to 0.16 GeV, -2.9% and -3.3% at 0.16 to 0.25 GeV,
    -1.1% and -1.3% at 0.25 to 0.40 GeV, -0.2% at 0.40 to 0.63 GeV and nothing above.
    With f = 0.60 the level against `t` becomes -0.0%, +0.2%, -0.5% in the three lowest
    bands (errors 0.2%), and each species is within 1.4%. f is the interacting fraction
    read from the table (0.54 to 0.65 according to the path), not a fit.
  - The builder's estimate of this effect (`c1/LIKE_FOR_LIKE.md`, section 3, item 3: at
    most 0.45%) was too low by a factor of ten.
  - What it is and is not. It is a difference between the two calculations in where the
    nucleon interactions stop inside one bin (1.26 against 1.39 GeV), not an error of
    the transport; neither cut is physical. It shows how much the lowest band depends on
    that cut in a field-free sky: nucleons of 1.26 to 1.58 GeV make about 12% of the
    flux at 0.10 to 0.16 GeV in the solver [from the two variants: 4.7% / 0.40 and 5.4%
    / 0.46]. Interactions below the cut are absent from the solver, from the Monte Carlo
    and from MCEq.
  - Shape. The variant's change depends little on cos Z (lowest band: -5.0% at cos Z 0.1
    to 0.2, -4.4% at 0.9 to 1). Applied cell by cell to the frozen 3D cells (an
    estimate: no 3D solve of the variant was made), the count of G1.3(a) is 378 with f =
    0.60 and 380 with f = 0.54, and the row means move by less than 0.2 points. No
    verdict follows: this is the scatter of the count shown in part A.
  - **Not explained: the level of +1 to +2% above 0.4 GeV**, which the variants do not
    change. By stratum, mean over the bands above 0.4 GeV: +2.2 +- 0.5% for primaries of
    3.16 to 10 GeV per nucleon, +1.5 +- 0.4% for 10 to 31.6 GeV, +0.9 +- 0.5% for 31.6
    to 100 GeV, -0.5 +- 0.7% for 100 to 316 GeV, -0.7 +- 0.7% for 316 to 1000 GeV and
    -2.6 +- 1.3% above. The bands are taken as independent although they share showers,
    so the errors are lower limits (above 100 GeV the error of one band is 2 to 3). The
    excess falls with the primary energy; the strata above 100 GeV are each compatible
    with zero. A candidate, not tested here: the energy discretisation of the kernels
    (parent at the bin centre), for which the builder gave +-1 to 2% in level before the
    comparison (`c1/LIKE_FOR_LIKE.md`, section 3, item 2). For the two-body decays the
    builder measured it: the centre in place of a parent spread over the bin raises nu_e
    and anti-nu_e by 0.6 to 2.2% and nu_mu and anti-nu_mu by 0.3 to 1.2% at 0.4 to 10
    GeV (row 7); for the interactions it is an estimate. A decay effect would not depend
    on the primary energy. MCEq has the same discretisation.
  - Deviations of the agent: two errors in its rebuild script (wall time lost, no data),
    one process wait with `pgrep -f` (read only). 45 core-minutes.
- **The muon clock's effect, from the two references (2026-11-19, manager;
  `c1/mcref2/CLOCK_PAIR.md`, `CLOCK_NOTE.md`).** Opened after G1.3 had run. Corrected
  clock over legacy clock, field free, 0.10 to 1 GeV, the four species together: `s`
  +0.84 +- 0.30%, `t` +0.34 +- 0.22% (nu_e and anti-nu_e +1.4 +- 0.6% in `s`). The
  quantity the product table uses, s/t, moves by less than 0.3%, 0.5% and 0.8% (value plus
  2 sigma) in cos Z 0.6 to 1, 0.3 to 0.6 and 0.1 to 0.3; per cell the two runs agree
  within noise. The row cos Z 0 to 0.1 is bounded only at 3 to 11%, and the field-on and
  up-going cases are not covered by the pair. The first run was not compared with the
  solver.
- **C1 closing cycle, before the comparison (2026-11-20, manager; outputs in
  `c1/close/`).**
  - Change 1, commit `1e5d2d8`: the bound in `solver._interp_log`, as proposed, with a
    unit test.
  - Change 2, commit `ab8bda7`: rule M3 of `kernelset_mc.py` and six new sets in
    `kernels/mc_close/` (hashes in `RESULTS_C1.md`, section 5). Built by an agent that
    opened no tally, from a brief that fixed the rule before any set existed
    (`c1/close/K_BRIEF.md`, report `WP_K1_REPORT.md`). For protons and neutrons in the
    bin of 1.26 to 1.58 GeV: sources into the bin times the fraction that interacts over
    2000 g/cm2 (primaries 0.602, 0.594; secondaries 0.605, 0.607), rate 9.46e-3 and
    9.62e-3 cm2/g. Manager's check (`c1/close/manager/check_k1.txt`): each set differs
    from its predecessor in those entries only, by those factors; own quadrature agrees
    to 4e-11.
  - The check fixed above, set against direct integration of the Monte Carlo's rule,
    probability to interact over X: exact at 2000 g/cm2; +4.2%, +6.4%, +6.4%, +3.2% at
    100, 200, 500, 1030 g/cm2; -4.1%, -7.0% at 5000 and 10 000 g/cm2 (the old sets:
    +55 to +87%). One fraction per bin does not follow the depth.
  - Straight-ahead solve, new set over old, no tally: -4.71%, -2.89%, -1.13%, -0.23%,
    -0.02% in the bands from 0.10 to 1 GeV, nothing above.
  - Not treated, reported by the builder: the same kind of bin for the other
    projectiles and the part of two bins that the Monte Carlo serves and the sets leave
    empty; at most 0.03% per band and 0.06% per cell on variants of the set.
  - **G1.0 PASS** (`c1/close/gates/G1.0.md`).
  - **G1.1 FAIL**, with the numbers of 2026-11-19: +2.035% (mu, vertical, 0.141 GeV,
    loss on) and +101% (mu, cos Z 0.05, 0.112 GeV, loss off).
  - **G1.2 FAIL.** Parts A and B as on 2026-11-19 (A fails in the loss-off muon cell at
    cos Z 0.05, B is inside). Part C on the new sets: G1.1 muon cells at cos Z 0.05 move
    by up to 3.2% (family), 2.3% (L), 1.1% (sampling), 0.7% (levels); the G1.3 cells by
    at most 0.16%. Base run of part C: 14.0 core-minutes, 5.1 GB.
  - G1.0, G1.1 and G1.2 A and B ran on `1e5d2d8`, G1.2 C on `ab8bda7`; nothing that
    those solves import differs between the two. Unit tests on `ab8bda7`: 116 passed.
  - Variants of the candidate test, built and solved before the freeze, compared with
    the base solve only (`c1/close/WP_K2_REPORT.md`): (C) changes the straight-ahead
    flux by -1.2 to -2.5% per band from 0.4 to 10 GeV and by +0.3% or less below; mean
    of the seven bands (C) -1.89%, (A) -3.21%, (B) -0.02%. With the parent at the bin
    centre the builder's assembly reproduces the base set to 7e-16. Sixteen points per
    bin against eight: at most 0.12% in a cell.
  - The builder's stop condition "a variant changes the flux by more than 5%" fired: two
    cells of (A) at 5.2%, and the strata near their kinematic end (up to 132% where the
    stratum makes 0.04% of the flux). Read by the manager as binning, not as an error,
    for the reasons above.
  - Manager's check of the variants (`c1/close/manager/k2check/CHECK.txt`). The
    residual of 8.3e-3 that the builder reported for the stratum of 10 to 31.6 GeV is in
    the stratum's top bin alone; twelve sweeps in place of six bring it to 1e-14 and
    change no cell beyond 3e-13. The builder changed the slope of (C) in the turn-on bin
    once, before any (C) solve had finished; its first version differs by at most 0.03%
    in a band and by 0.002 points in the mean above 0.4 GeV.
  - Second freeze written to `RESULTS_C1.md`, section 5. No solve on the changed sets
    had been compared with a tally.
- **G1.3 to G1.5 on the second frozen state (2026-11-20; an agent that wrote neither the
  solver nor the kernel sets; report `c1/close/WP_G13_REPORT.md`, results
  `c1/close/g13/G1.3.md`, `G1.3.json`, `G1.4.md`, `g15/G1.5.md`).** The agent checked
  before its runs: HEAD `b77b065`, clean tree, code of `tools/cascade3d` equal to
  `ab8bda7` (the diff lists this file and `RESULTS_C1.md` only), the six kernel sets and
  the reference by sha256. `gates/g1_3.py` ran as committed (sha256
  `9cab9b3cc7d10773...`) with `--count-min 373`. Its pool solve is bit-identical to the
  manager's `c1/close/g12/C_base.npz`. The manager recomputed the counts, the row means,
  the levels and G1.4 from the stored cells with separate scripts
  (`c1/close/manager/recheck_pool2000.txt`, `recheck_fresh1e5.txt`,
  `recheck_g14_close.txt`): they agree. After the run the tree is clean and the kernel
  sets verify against their hashes.
  - **G1.3(a) as amended: PASS.** 380 of the 400 cells are within max(2%, 2 sigma_MC)
    of 1; 373 are needed. The 380 of the original text is reached with no margin. The
    ten row means below 1 GeV differ from 1 by -0.70% to +0.45%, each inside
    max(1%, 2 sigma). On the state of 2026-11-19 the count was 377. Cells outside, by
    species (nu_e, anti-nu_e, nu_mu, anti-nu_mu): 2, 10, 3, 5; by band from 0.10 GeV up:
    2, 2, 2, 2, 0, 1, 1, 7, 1, 2; by cos Z row from the horizon up: 2, 1, 2, 1, 1, 5, 3,
    2, 1, 2. The net change against 2026-11-19 is in the two bands below 0.25 GeV.
  - Reported with (a), not gated (`c1/close/manager/diag_a_pool2000.txt`): chi2 of the
    400 shape pulls 466.3 (489.5 on 2026-11-19). An exactly right solver gives a larger
    chi2 in 12% of the cases with Gaussian errors and in 33% when the ten runs are
    resampled, and it reaches 380 cells or more in about six cases of ten. Shape
    difference common to the ten runs, noise subtracted: not different from zero in any
    block; 2 sigma bounds 0.98% below 1 GeV at cos Z 0.1 to 1, 3.1% below 1 GeV in the
    row cos Z 0 to 0.1, 5.9% from 1 to 10 GeV. The agent's report gives 467.5 for the
    chi2 of the pulls; that number is the agreement of the two caps of the reference
    (`G1.3.md`, "Not gates"), the solver's is 466.3.
  - **G1.3(b): the trigger fired.** The mean of r - 1 per species and band is above 5%
    in 3 of the 40: anti-nu_e at 2.51 to 3.98 GeV, +7.4 +- 4.3%; nu_e at 6.31 to 10 GeV,
    +10.9 +- 8.9%; anti-nu_mu at 6.31 to 10 GeV, +6.7 +- 2.7%. These are the values of
    2026-11-19: the changes of this cycle do not act above 1 GeV. The fourth of
    2026-11-19, nu_e at 0.10 to 0.16 GeV, is now +0.45 +- 0.47%. The four species
    together, per band from 0.10 GeV up: +0.26, +0.20, -0.19, +0.62, +1.09, +2.66,
    +1.79, +3.72, +1.54, +4.53%. In the anti-nu_e mean the cell of the row cos Z 0 to
    0.1 stands at +57% (error from the ten runs 13%); the nine other rows give
    +1.8%. No diagnosis was
    started: the one cycle that the gate text gives was used on 2026-11-19, and this
    closing cycle is what the PI added to it.
  - The level in the statistic of `level_stat.py`, 3D solve over `s`
    (`c1/close/manager/level3d_close.txt`; computed by the manager after the run): bands
    below 0.4 GeV +0.22 +- 0.20%, +0.17 +- 0.31%, -0.07 +- 0.23% (2026-11-19: +4.75,
    +2.94, +0.98%); mean of the bands from 0.4 to 10 GeV +1.77 +- 0.18% (+1.83 +-
    0.18%). With the 10^5-event kernels +1.23 +- 0.18%.
  - G1.3(c), reported: with the 10^5-event kernels 382 of 400, ten rows of ten, the
    trigger fires as in (b); normalised shape within 0.37% of the pool kernels.
  - G1.3(d), reported: 372 of 400 inside the tolerance of (a) (349 on 2026-11-19, 373
    there with the bound in a diagnostic copy). The row cos Z 0 to 0.1 holds valid
    numbers: q from 0.63 to 1.42. With the fluxes summed over the four species (manager,
    after the run, not fixed in advance; `c1/close/manager/ratio_d_close.txt`):
    (3D / straight-ahead) over s/t differs from 1 by -0.13 to +0.56% per band at cos Z
    0.1 to 1 (errors 0.2 to 0.8%); in the row cos Z 0 to 0.1 by -1.7 +- 2.9% below 1 GeV
    and -5 +- 11% above. By zenith below 1 GeV: -0.9 +- 0.8% (cos Z 0.1 to 0.3), +0.76
    +- 0.29% (0.3 to 0.6), +0.34 +- 0.14% (0.6 to 1).
  - **G1.4 PASS**, with the margin of 2026-11-19: 400 of 400 cells within 0.5%, largest
    0.4788% (anti-nu_e, 2.51 to 3.98 GeV).
  - **G1.5 FAIL.** Cold solve on four cores: 12.48 core-minutes (user 688 s, system
    61 s; 10 allowed), peak memory 5.48 GB (8 allowed), 4 min 20 s elapsed. Load average
    of the node 1.8 before, 4.5 after. A second solve in a warm process: 12.09
    core-minutes (not the gate). 2026-11-19: 13.23. No work was done on the cost.
  - The five solves: 710 to 864 CPU seconds each, 4.8 to 5.4 GB, largest sweep residual
    2.6e-5 (second half-sample set; the others 3e-8 or less); every cell finite and
    positive. Compute of the agent: about 1.75 core-h.
- **Test of the candidate for the level above 0.4 GeV (2026-11-20, manager; run after
  the freeze commit; `c1/close/manager/candidate_cmp.py`, output `candidate_cmp.txt`,
  per species `candidate_species.txt`).** Straight-ahead solve over `t`, statistic of
  `level_stat.py`, ten runs.
  - Changed set (parent at the bin centre): +1.44 +- 0.20% (2026-11-19: +1.52 +-
    0.20%).
  - Variant (C): -0.18 +- 0.20%. By the wording fixed in advance the level is
    **"explained"**: the excess left is within 0.6% of zero.
  - Variant (A), slope 2.7: -1.73 +- 0.19%. Variant (B), slope 1: +1.16 +- 0.20%.
  - Strata, changed set and (C): 3.16 to 10 GeV per nucleon +2.14 +- 0.55% and +0.57 +-
    0.54%; 10 to 31.6: +1.48 +- 0.37% and +0.58 +- 0.37%; 31.6 to 100: +0.88 +- 0.48%
    and +0.07 +- 0.48%.
  - Per species, changed set and (C): nu_e +0.96 +- 0.37% and -0.77 +- 0.36%; anti-nu_e
    +1.64 +- 0.43% and -0.34 +- 0.42%; nu_mu +1.48 +- 0.29% and +0.11 +- 0.28%;
    anti-nu_mu +1.59 +- 0.33% and -0.02 +- 0.33%.
  - Bands below 0.4 GeV on the changed set: -0.04 +- 0.18%, +0.17 +- 0.24%, -0.48 +-
    0.23% (2026-11-19: +4.91, +3.14, +0.66%). The turn-on rule removed that offset.
  - Limits. The test is straight-ahead: no 3D solve of a variant exists. The errors take
    the bands as independent and are lower limits. The outcome depends on the slope: (A)
    over-corrects by more than the excess and (B) leaves four fifths of it, so
    "explained" rests on the slopes that the set itself gives. The sets of the gates
    keep the parent at the bin centre and carry the excess (3D over `s`: +1.77 +-
    0.18%).
- **Compute of the closing cycle:** about 10 core-h of the 15 (kernel sets and variants
  about 3, G1.0 to G1.2 and unit tests about 5, G1.3 to G1.5 1.75, checks 0.3). No
  Monte Carlo was run.

## Decisions of the PI

- **2026-11-19, after the report of G1.0 to G1.2 (interim use of decision point 1).** The PI
  was given the failed G1.1 and G1.2, the manager's check, and three recommendations, and
  answered "Continue". The manager reads this as acceptance of the three:
  1. C1 continues. G1.1 stays on record as failed and is not redefined.
  2. The solver's error for loss-off muons near the horizon below 0.3 GeV is documented as
     a known limit of an unphysical configuration; no grid nodes are spent on it.
  3. Next: a second solver cycle (near-vertical angular grid, up-scatter into transported
     species, cost), the Monte Carlo-matched kernel sets, the freeze of rule 4, then G1.3
     and G1.4 by an agent that wrote neither.
  After the second cycle G1.0 to G1.2 are re-run in full, as written. The comparison with
  step-converged MCEq is reported next to G1.1 and is not a gate.
- **2026-11-19, after the report of the kernel sets and of the Monte Carlo's muon decay
  clock.** The PI was given the finding, its estimated effect on the G1.3 reference and on
  the product table, and the manager's recommendation, and answered "Follow your
  recommendations". That is:
  1. The clock is corrected behind an opt-in switch of `tools/mc3d`
     (`Config(mu_clock="converged")`, `sky3d.py --muon-clock converged`); the default
     stays bit-identical to every earlier run. Acceptance, fixed here before the code is
     written: `gates/mc_muon_clock_check.py 4000 converged`, every entry within 0.2%.
  2. The field-free reference is re-run with the switch on, same settings and seeds
     (about 25 core-h, into `c1/mcref2/`). The re-run becomes the reference of G1.3. The
     first run is kept and is not compared with the solver.
  3. The pair of runs measures the clock's effect per cell. Whether the product table
     needs a correction is put to the PI after that.
  The compute of C0 and C1 rises from about 110 to about 115 core-h.
- **2026-11-20, after the decision-point-1 report.** The PI was given the verdicts of
  G1.3 to G1.5, the diagnosis cycle and three options (stop the phase; one closing
  cycle of C1; go on to C2), and answered "Go ahead with option 2." Option 2 as it was
  put: apply the two fixes that were not applied (the bound of the energy
  interpolation; the cross-section turn-on in the kernel sets), test the candidate for
  the level above 0.4 GeV, and re-run G1.0 to G1.5 in full; about 15 core-h, no new
  Monte Carlo; the comparison no longer blind; the criterion of G1.3(a) to be set
  before the re-run, for which the manager proposed the row means as written plus a
  count at or above the 5th percentile of a correct solver; and the manager's
  recommendation to stop if the re-run fails again or the level stays unexplained. The
  manager reads the answer as acceptance of the proposed criterion as well, and said so
  to the PI at the start of the cycle. The section "C1 closing cycle" fixes it. Not
  decided by this answer and left as they are: G1.5 (cost), the muon clock in the
  product table, the form of G2.3.
- **2026-11-20, after the report of the closing cycle (decision point 1).** The PI was
  given the gates on the second frozen state (G1.0 pass; G1.1 and G1.2 fail on ground
  muons as before; G1.3(a) as amended pass with 380 of 400 against 373; the trigger of
  G1.3(b) fired above 2.5 GeV; G1.4 pass; G1.5 fail at 12.48 core-minutes), the
  manager's reading and six points, and answered "Follow your recommendations". Point
  by point:
  1. **C2 is authorised.** The manager recommended going on, since neither of his stop
     conditions occurred. Budget as in the plan: about 100 core-h, of which about 60 for
     one new Monte Carlo run, over one to two weeks. A PI decision follows C2; C3 is not
     authorised.
  2. G1.1 is accepted as a known limit for ground muons until C3, where muon intensities
     are used. It stays on record as failed and is not redefined.
  3. G2.3: its thresholds are fixed from the reference alone, before any solver output
     with the cutoff exists (section C2).
  4. G1.5: the cost is accepted for now and stays on record as failed. The cost of a C2
     solve is measured at the start of C2 and brought back to the PI.
  5. Kernel sets for C2: the report put two options (keep the parent at the bin centre
     and carry the level of +1.8%; or switch to the spread-parent variant after one 3D
     check solve) and gave **no recommendation**, so the answer decides nothing here.
     The manager takes the option that changes nothing: C2 uses the kernel sets frozen
     in the closing cycle. No check solve is run, and the spread-parent variant stays an
     open item. This default is said to the PI as such.
  6. Muon clock: `kamioka_3D_v2l.npz` is left as it is, with one documentation entry
     (`tools/mceq3d/INTEGRATION.md`, 18.7). New Monte Carlo runs use
     `--muon-clock converged`.
