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
