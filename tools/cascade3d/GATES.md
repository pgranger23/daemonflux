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
