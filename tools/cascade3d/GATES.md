# Pre-registered gates, Phase 3 (`cascade3d`)

Stages C0 and C1: written 2026-11-18, before any Phase 3 code or run exists.
Stage C2: written 2026-11-20, before any C2 code or run exists (section C2).
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

## C2. Azimuth and the cutoff sky, no bending

Written 2026-11-20, after the PI's decision of that date (last section) and before any
C2 code, Monte Carlo run or solve exists. The plan's text of the stage is `PLAN.md`,
section 4, C2. Rules 1 to 3 apply as written.

- Rule 4 applies with `RESULTS_C2.md`. The settings (grids in μ, φ and location, L,
  sweeps, ray sampling, kernel sets, cutoff skies with their hashes, reference altitude)
  are written there after G2.0, G2.1, G2.2, G2.4 and G2.5, and before the solver's
  output with the cutoff is compared with a Monte Carlo tally for the first time.
- Rule 5 applies to G2.3: it is run by an agent that wrote neither the solver, the
  kernel sets nor the cutoff skies.
- The agents that write the solver and the cutoff skies do not open the reference of
  G2.3 before the freeze. The agent that runs the reference compares nothing with the
  solver.

**Carried from C1.**
- G1.1 and G1.2 are on record as failed for ground muons (accepted by the PI until C3).
  The gates of C2 are on the four neutrino species. Ground muons are reported where a
  gate has them.
- G1.5 is on record as failed. The cost of C2 is measured first (G2.6) and brought to
  the PI.
- Kernel sets: those frozen in the closing cycle (`RESULTS_C1.md`, section 5; parent at
  the bin centre). In an absolute comparison with the Monte Carlo the solver is
  therefore expected about 1.8% high from 0.4 to 10 GeV, as measured in C1. G2.3 is
  gated on quantities in which this level cancels.

**Notation.** Cells: the four neutrino species × the ten energy bands of G1.3 (0.1 to
10 GeV) × ten cos Z bins from 0 to 1 × twelve azimuth bins of 30°, 4800 in all. Zenith
and azimuth are those of `tools/mc3d/scoring.py`: in the local frame of the point where
the neutrino crosses the ground; azimuth of the direction the neutrino comes from, from
north through east. Φ_cut is the flux with the cutoff. Φ_0 is the flux with the
transmission set to 1, averaged over azimuth. K = Φ_cut / Φ_0 is the cutoff factor of a
cell.

**G2.0 Cutoff sky against direct back-tracing (reported, with a trigger).** The
transmission that `boundary.py` gives a primary (nucleus group, kinetic energy per
nucleon, point and direction on the injection sphere) against the verdict of
`tools/mc3d/cutoff.py` (`gate`, then `trace_gated`) for the same state. States: at least
20 000, drawn as the reference run draws its primaries (`cutoff.sample_test_states`),
rigidity below 60 GV; states that the sky does not cover are counted and left out.
Classes: rigidity 0 to 5, 5 to 10, 10 to 20 and 20 to 60 GV × cos of the zenith at
injection 0 to 0.25, 0.25 to 0.5 and 0.5 to 1 × four azimuth quadrants centred on
north, east, south and west. Per class: mean transmission of the sky against the
allowed fraction of the verdicts. A class in which they differ by more than
max(0.02, 2σ), σ the binomial error, triggers a check of the sky before the freeze.

**G2.1 Transmission 1 reproduces C1 (gated).** The C2 solver with the transmission set
to 1 everywhere, assembled exactly as for G2.3 (location grid, interpolation, average
over the 10° cap), against the C1 solve `c1/close/g13/pool2000.npz`: every G1.3 cell, in
each of the twelve azimuth bins, within 0.2%.

**G2.2 Straight-ahead limit with the cutoff against MCEq (gated).** At Kamioka
(36.4267° N, 137.31° E), cos Z = 1, 0.75, 0.5, 0.25, 0.15, 0.05, azimuth 0°, 90°, 180°
and 270° (21 directions; the vertical once). Solver: straight-ahead limit, boundary from
the cutoff sky for that line of sight. MCEq: same zenith, the spectrum of each nucleus
group multiplied by the same binned transmission of that line. Otherwise as G1.1 with
energy loss on (same database, population, atmosphere, H3a, observation level).
Pass: the four neutrino species within 1% in every MCEq bin from 0.1 to 100 GeV.
Ground muons are reported.

**G2.3 Cutoff and azimuth against the Monte Carlo.**

Reference (WP 2.3): `tools/mc3d/sky3d.py` at Kamioka with the cutoff on and no bending.
The command is that of the field-free reference of C1 (`c1/mcref2/scripts/run_all.sh`:
`--model GSFG:GH --nstrata 7 --chunk 10000 --r-thr 0 --no-bend-primary --no-muon-field
--thresholds chain --muon-clock converged`, pools `pools_fix`) without `--no-cutoff`.
- Twenty runs of equal size, seeds 900000000 + 20000000 j, j = 0 to 19 (no seed of an
  earlier run).
- Size: 2×10^5 primaries per run, the plan's 4×10^6 in all. 5×10^5 per run instead if
  a pilot (own seed, not part of the reference) projects at most 55 core-h in total
  for that. The size is chosen on cost alone, before the runs.
- `--upgoing` is added if the pilot shows that it costs less than 10% more.
- Tallies used: `s` (each of the twelve azimuthal copies of a primary with its own
  cutoff verdict) and `n` (no cutoff, copy 0), in the flux measure, in the cap of 10°
  around the site.

Amended on 2026-11-23 by decision of the PI, before any solver output with the cutoff
was compared with a tally: the reference is replaced by twenty runs with the event
plane turned (`--event-roll`, seeds 1500000000 + 20000000 j). The bullets above are
kept as first written. The reasons and the amendment are in "C2, dated entries".

Statistics: `gates/c2_stats.py`, committed with this text. The same functions serve
the reference, the count thresholds and the solver.
- K_MC = `s`(cell) / (Σ_az `n` / 12).
- σ of any statistic: delete-one-run jackknife of that statistic over the twenty runs.

Refinement of the plan's wording, made here before any run. The plan gates "F", the
table factor `s/t`. Two reasons against that, both from C1. The Monte Carlo's `t` is not
what the solver's straight-ahead limit computes (G1.3). And an absolute comparison
carries the 1.8% level and the full sampling error of the cascades. The cutoff factor K
isolates what C2 adds. In the Monte Carlo its numerator and its denominator come from
the same showers, so most of the sampling error cancels. Measured on the field-free
reference of C1 with the azimuth kept, where K is 1 by construction: K scatters over
the 4800 cells with an rms of 1.25% at 10^7 primaries and 2.1% at 4×10^6, against 1 to
10% for the flux of one azimuth cell at 10^7. With the cutoff the cancellation is
weaker; the jackknife measures what is left.

Solver, like for like:
- kernel set `kernels/mc_close/ks_mc_pool2000_L32_losson.npz` (sha256 beginning
  `32b4215d5bf0de6d`);
- initial condition of the reference run: the nucleon fluxes of each nucleus group
  (`tools/mc3d/primaries.py`, `GSFG:GH`), kinetic energy per nucleon from 1 to 3162 GeV,
  superposition, rigidity R = (A/Z) p (`cutoff.rigidity_gv`);
- transmission of a primary line from `tools/mc3d/cutoff.py` with its defaults (IGRF-14
  at 2020-01-01, injection sphere at 100 km, re-entry rule at that sphere), at the state
  where the line enters the injection sphere, as the Monte Carlo tests each copy;
- Earth radius 6371 km;
- flux averaged over the 10° cap, each ground point with its own local frame, and over
  each cell;
- Φ_0 from the same assembly with the transmission set to 1 (the solve of G2.1).

Parts:
- **(a) Cutoff factor per cell (gated).** |K_solver − K_MC| ≤ max(0.03 K_MC, 2σ) in at
  least N_a of the cells counted. A cell is counted if `s` is filled in at least half
  of the runs; the others are listed.
- **(b) East-West (gated).** Per energy band and cos Z row, the four species summed:
  a_c = Σ Φ_cut cos(az) / Σ Φ_cut and a_s = Σ Φ_cut sin(az) / Σ Φ_cut over the twelve
  azimuth bins, az at the bin centres. Both within max(0.01, 2σ) of the Monte Carlo's,
  in at least N_b of the 100 pairs of band and row. The floor of 0.01 is an addition to
  the plan's "within 2σ_MC"; it is a 2% error of the modulation, of the size of the 3%
  of (a).
- **(c) Horizontal over vertical (gated).** Per energy band: H = K(cos Z 0 to 0.2) /
  K(cos Z 0.8 to 1), each K from the sums over the species, the azimuth and the two
  rows. Within max(2%, 2σ) of the Monte Carlo's in at least N_c of the 10 bands.

N_a, N_b and N_c are the counts that an exactly right solver reaches or exceeds in 95%
of the cases. They are computed from the reference alone by `gates/null_count_c2.py`
(committed with this text, before the reference exists): the larger of two null models,
independent Gaussian errors of size σ and resampling of the runs, as in the closing
cycle of C1. The three numbers are written here as a dated entry before any solver
output with the cutoff is compared with a tally. The count of (a) is also reported
against 95% of the cells counted, the plan's wording.

Amended on 2026-11-23, before the reference was combined or compared with anything: the
thresholds are those of a third null model, half of the runs against the other half.
The paragraph above and the one below are kept as first written. The amendment, its
reasons and what it costs are in "C2, dated entries".

What the choice costs. With three gated parts at 95% each, an exactly right solver
fails at least one of them in up to 14% of the cases. Where the two null models
differ, taking the larger number makes a right solver fail that part more often than
5% if resampling is the truer model. Both numbers of each part are reported.

Reported, not gated:
- **(d) Absolute.** r = Φ_cut,solver / `s` per cell, and its mean per species and band
  (expected level: about +1.8% above 0.4 GeV).
- **(e) Level of K per band**, the species, the zenith and the azimuth summed, solver
  against Monte Carlo. A band outside max(2%, 2σ) triggers one diagnosis cycle before
  decision point 2.
- **(f) Plan wording.** Solver 3D with the cutoff over its own straight-ahead limit with
  no cutoff, against `s/t`, with the caveat of G1.3.
- **(g) Smaller cap.** (a) to (c) for the cap of 5°.
- **(h) Up-going rows near the horizon**, if the reference has the tallies `us` and
  `un` and the assembly covers the far side: (a) for cos Z from −0.2 to 0.

**G2.4 Homogeneity (gated).** Halving the spacing of the location grid, and moving the
reference altitude at which a line of sight is attached to a location, each move no
cell of Φ_cut of G2.3 (cap of 10°) by more than 0.5%. The two variations (the spacings,
the altitudes) are fixed here as a dated entry once `DESIGN.md` names the location grid
and the reference altitude, before any C2 solve with the cutoff. Fixed on 2026-11-23
("C2, dated entries"), together with two additions: G2.7 (gated) and G2.8 (reported).

**G2.5 Numerical convergence with the azimuth (gated; an addition to the plan).** At
one location, the grid location nearest the site, with the cutoff, on the ground flux
of that location in the cells of G2.3:
- (a) doubling, one at a time, the φ nodes (or azimuthal modes), the μ grid and the
  Legendre order L moves no cell by more than 0.5%;
- (b) the cutoff sky rotated by 30° about the vertical gives the flux moved by one
  azimuth bin, every cell within 0.5%;
- (c) the same-bin sweep residual is below 1e-4 in every energy bin.

**G2.6 Cost (reported, no threshold).** One location solve with the cutoff, and the full
set of solves of G2.3, in core-minutes and GB (`/usr/bin/time -v`, pinned cores). The
first measurement goes to the PI before the solves of G2.3, with what it projects for
G2.4 and for the global grid of C5.

**Decision point 2.** After C2 the manager reports to the PI, who decides on C3. The
manager's recommendation is fixed in advance: stop the phase if, after one diagnosis
cycle, G2.3(a) fails, or if G2.4 does not converge on a finer location grid (the plan's
stop rule). Compute: the plan's about 100 core-h. What goes materially beyond is brought
to the PI before it is spent.

## C2, dated entries

The entries that section C2 announces ("written here as a dated entry"), and the
amendments of its gates. Each is committed before the run it governs. Files named
`c2/...` are under `/home/pigrange/mc3d_work/runs/native3d/`.

### 2026-11-23. Count thresholds of G2.3 (N_a, N_b, N_c): amendment

**Standing.** The fault was found on 2026-11-20 while the reference ran, and the session
ended before the amendment was written. The twenty runs ended on 2026-11-20 at 19:24.
So this text is committed after the reference exists, which the first text did not
intend. What holds at the time of this commit:
- the runs are not combined, and no file of G2.3 statistics exists;
- the manager, who wrote the amendment and its scripts, has opened no tally of
  `c2/mcref/` (only the progress log, the run script and directory listings);
- no C2 solver code exists.

The PI was given the fault and the rule below and approved it on 2026-11-23 ("Proceed as
proposed"; last section).

**What changes.** N_a, N_b and N_c are the 5th-percentile counts of a third null model,
half of the runs against the other half (`c2_stats.null_split`). They are no longer the
larger of the Gaussian and the resampling numbers. Everything else of G2.3 stands: the
statistics, the tolerances, the cells counted, the 95%, the reported parts.

- For a random half A of the twenty runs and its complement B,
  d = (stat(A) − stat(B)) / 2 stands for the deviation of the reference from the truth.
- The error that goes with it is the jackknife error with d taken out:
  e² = (ν σ² − d²) / (ν − 1), ν = 19.
- One draw counts the items in which a solver at the reference value is inside the
  tolerance of a reference at value + d with error e. 20 000 draws.
- N is the largest count that at most 5% of the draws fall below.

The numbers of the first rule are computed and reported next to the new ones
(`threshold_first_rule`). They decide nothing.

**Why.** Before any tally was opened, the manager tested the first rule on synthetic
references with known truth (`gates/calib_null_c2.py`). A synthetic reference has twenty
runs of Poisson events. The exactly right solver is the truth itself. Three models of
how the twelve azimuth cells of a ring share their primaries: `indep` (not at all),
`corr` (same neutrinos, independent verdicts), `nested` (same neutrinos, verdicts nested
in rigidity, which is what twelve rotated copies of one primary give). The table gives
how often the exactly right solver falls below the threshold of part (a); 5% is
intended. "Events" is the number of events per azimuth cell and run with no cutoff in
the best-filled cell; the counts fall by a factor 100 over the ten bands. 200 references
per line unless stated. A negative number in the last column: the threshold could be
raised by that much.

| Model, events | Cells on the 3% floor | First rule | Half against half | Cells to lower it for 5% |
|---|---|---|---|---|
| indep, 300 | 0.1% | 100% | 6.0% | 2 |
| indep, 1500 | 14% | 99% | 3.0% | −3 |
| corr, 60 | 14% | 100% | 38% | 20 |
| corr, 300 | 72% | 39% | 3.5% | −1 |
| nested, 60 | 14% | 88% | 11.5% | 12 |
| nested, 150 | 44% | 66% | 5.5% | 6 |
| nested, 300 | 72% | 43% | 5.5% | 2 |
| nested, 1000 | 99% | 16% | 7.5% | 4 |
| nested, counts of the dry run (300 references) | 36% | 82% | 13.3% | 22 |
| nested, the same with counts 29% too high (300) | 44% | 72% | 11.0% | 22 |
| corr, the same with counts 29% too high (150) | 44% | 95% | 24% | 13 |

"Counts of the dry run": the event counts per ring of the field-free reference of C1
with the azimuth kept (10^7 primaries, the size of the reference of G2.3), with a
synthetic cutoff factor. The first of the three lines is the nearest to what the
reference is expected to be. The other two were run with an estimate of the counts that
is too high by 29% on average (corrected since in `calib_null_c2.py`). Statistical
error of a rate: 1.5 to 3 points.

Causes of the first rule's failure:
1. The Gaussian null model treats σ as known. It is a jackknife estimate with 19 degrees
   of freedom, and 2σ then covers 94.0%, not 95.45%.
2. With independent items the null model understates the spread of the count when the
   twelve azimuth cells of a ring share their primaries.
3. In part (a) the resampling number never exceeded the Gaussian one, in any reference.
   "The larger of the two" was always the Gaussian number.

**What is left in the amended rule.** At the expected event counts an exactly right
solver fails part (a) in about 13% of the cases with the `nested` model, not in 5%. In
cells with few events and a cutoff factor near 1, the deviation of a cell and its error
estimate are correlated (skewed counts). The half-against-half model makes them
independent and counts about 12 cells of about 4770 as inside too often. A threshold
about 20 cells lower would give 5%. Parts (b) and (c): 0 to 4% in every test.

**Calibration at the reference's own event counts (reported).** The strictness that is
left is measured on the reference and reported. It is not corrected. Commands, fixed
here before the reference is combined (REF is the combined reference):

    calib_null_c2.py nested 0 0 150 SEED OUT_SEED.json REF.npz ref    SEED = 51 52 53 54
    calib_null_c2.py --calib CALIB.json OUT_51.json OUT_52.json OUT_53.json OUT_54.json
    null_count_c2.py REF.npz --outdir DIR --calib CALIB.json

That is 600 synthetic references of twenty runs, model `nested`, drawn at the event
counts and the cutoff factors estimated from the reference (cap of 10°), with 400 draws
of each null model per reference. Output: the rate at which the exactly right solver
falls below the half-against-half threshold, and Δ_a, the smallest number of cells by
which the threshold would have to be lowered for a rate of at most 5%.

How it is read, fixed here:
- The gate decides on N_a of the half-against-half model of the reference itself. The
  calibration changes no threshold.
- A count of (a) below N_a and at or above N_a − Δ_a is a FAIL. It goes to the PI with
  Δ_a and the measured rate.
- If the measured rate is above 10%, or if there is no calibration, the result of (a)
  carries that as a caveat, pass or fail.
- Parts (b) and (c), and the thresholds of the 5° cap (part g, reported): no
  calibration.

Validation of the calibration (2026-11-23, `c2/manager/calib/valid/`). Truth known,
`nested`, counts of the dry run: rate 13.3 ± 2.0%, 22 ± 7 cells for 5% (300
references). The calibration was then run on three synthetic references drawn from that
truth, from their tallies alone, 150 references each: 12.7%, 9.3% and 17.3% (each ± 3
points; pooled 13.1 ± 1.6%), Δ_a = 26, 10 and 20. It recovers the rate within the
statistics. Δ_a from 150 references is uncertain by 5 to 14 cells; with the 600 fixed
above about 4 are expected. Not modelled by the calibration: event weights, and several
neutrinos of one shower falling in different cells. The half-against-half model itself
works on the runs and keeps both.

**What N_a detects.** Fraction of synthetic references on which a wrong solver fails
(`nested`, 150 to 1000 events, 60 references each):
- cutoff factor 1% too high everywhere: (a) fails in 95 to 97%;
- 2% or more too high: 100%;
- East-West error of 2% cos(az): (a) and (b) 100%;
- horizon 2% too high, falling to 0 at the vertical: (a) 100%, (c) 93%.

The plan's wording, 95% of the cells counted, detects a uniform error of about 2%. N_a
is about twice as strict. `g2_3.py` reports both counts. The gate is N_a.

**What the choice costs, restated.** With (a) at about 13% and (b) and (c) at up to 4%
each, an exactly right solver fails at least one gated part in 15 to 20% of the cases.
The first text said up to 14%.

**Check on real tallies.** The scripts were run on the field-free reference of C1 with
the azimuth kept (`c2/manager/dryrun_fieldfree_az.npz`, ten runs; K = 1 by construction;
not the reference of G2.3). The solver K = 1 counts 4766 of 4800 cells. Half against
half: median 4768, N5 4758. First rule: 4750. In this file 86% of the cells are on the
3% floor, where both rules behave.

### 2026-11-23. Entries before the first C2 solve with the cutoff

The design of C2 is `DESIGN_C2.md` (the design agent's note of 2026-11-20 as delivered,
sha256 beginning `d56a0733ff04f05a`). The manager's review is `REVIEW_DESIGN_C2.md`. The
design is accepted with the conditions below. Where the note and this section differ,
this section holds.

In short: azimuth as Fourier orders m = 0 to M about the local vertical. The flux with
the cutoff is the C1 solve minus a linear map of the boundary deficit,
Φ_cut = Φ_C1 − J[D], D = B⁰ − B. J is the tangent of the C1 scheme with guarded
derivative rules. The cap average is one boundary. The cutoff skies come from
`cutoff.trace_gated`, per location of a grid about the site.

**G2.7 Linearisation (gated; an addition to the plan).** The exact derivative of the C1
solve is unstable below 1.6 GeV (`DESIGN_C2.md`, 1.6; measured on a reduced geometry).
The guarded tangent is not the exact derivative of anything, so its accuracy is known
only against nonlinear solves. Test, on the full geometry and settings of the frozen C1
solve (`RESULTS_C1.md`, section 5; kernel set of G2.3), with two deficits that the
nonlinear C1 scheme can solve because they are symmetric in azimuth:
1. the isotropic step of the design's test 1: a smooth step in rigidity at 11.5 GV, 15%
   wide, R = p for protons and 2p for neutrons;
2. the azimuth average (order m = 0) of the deficit of the cutoff sky at the site, from
   `boundary.py`.

For each, Φ_lin = Φ_C1 − J[D] against Φ_nl, the nonlinear solve with the boundary
B⁰ − D, in the 400 cells of G1.3. Pass, for both deficits:
- (i) |Φ_lin − Φ_nl| ≤ 0.005 Φ_nl in every cell;
- (ii) |Φ_lin − Φ_nl| ≤ 0.01 (Φ_C1 − Φ_nl) in every cell with Φ_nl ≤ 0.95 Φ_C1.

Ground muons are reported. First setting: the prototype's (ρ = 0.1 on the source shape,
the source integral and the loss interpolation; the Simpson correction not guarded). On
the reduced geometry it gave 0.42% for (i) and 0.23% for (ii), so the margin of (i) is
small.

If it fails: one tuning cycle (guard on the Simpson correction; ρ per piece). Every
variant is run in full on both deficits and every result is reported. The setting that
passes is frozen in `RESULTS_C2.md`. If none passes, the gate is reported as failed. The
fallback of the design (chains in φ, two to three times the cost per boundary) is not
started without the PI. No solve of the mode path with a cutoff sky is run before
G2.7 has passed.

**G2.8 Mode path in the straight-ahead limit (reported, with a trigger; an addition).**
The orders m ≥ 1 have no nonlinear counterpart on the same code path. This check tests
their bookkeeping (frames, signs, phases, normalisation of the deficit). At the site,
the six zenith lines of G2.2 × 36 azimuth sectors of 10°, straight-ahead limit:
- path A: the nonlinear solve of each direction with that direction's own boundary (the
  path of G2.2);
- path B: the mode path with M = 18, which holds 36 sector values exactly, synthesised
  at the sector centres.

A difference above 1% for any of the four neutrino species, in any MCEq bin from 0.1 to
100 GeV, triggers a check of the mode path before the freeze. The check does not test
the kernel moments of order m ≥ 1: in the straight-ahead limit every order has the same
transport. If the mode path cannot be run in the straight-ahead limit at reasonable
cost, the agent says so and the check is reported as not done.

**G2.0, the draw and the frames.** The gate text leaves two points open.
- Draw: `cutoff.sample_test_states(seed, n)` with its default (`cap_deg=None`),
  rigidity below 60 GV. With `cap_deg=10` the lines that miss the ground would be left
  out, and they have their own code path in `boundary.py`.
- Location of a state: its line's first ground crossing or, for a line that misses the
  ground, the point under its tangent point. A state is covered if that location is
  within 10° of the site. Only covered states are traced. The others are counted.
- Size: at least 20 000 covered states, of which at least 2 000 on lines that miss the
  ground.
- Quadrants: in the local frame of the injection point.
- The classes and the trigger of the gate are evaluated on all covered states and,
  separately, on those that hit and those that miss the ground. For the last subset the
  classes are the four rigidity classes only.
- Reported in addition: 500 states above 60 GV on lines that miss the ground, traced.

**G2.4, the two variations.**
- (a) Spacing of the location grid: 5° (25 nodes on [−10°, 10°]² in azimuthal
  equidistant coordinates about the site) against 2.5° (81 nodes, which contain the 25).
- (b) Reference altitude: h_ref = 0 (a line of sight belongs to the location under its
  ground point) against h_ref = 15 km with the displaced location (the location of a
  ground cell is the point under its line's crossing of 15 km). Evaluated as in
  `DESIGN_C2.md`, 3.4: six shifted cap averages (shift 0, ±4° east, ±4° north, and 4°
  east with 4° north), interpolated quadratically in the shift, from sky tables at
  15 km on a 5° grid over [−15°, 15°]².
- Before the solves of (b), the interpolation error of the six-shift stencil is
  measured on the deficit itself, with no solve, against deficits built directly at the
  displacement of a set of test directions. If it exceeds 0.001 of the no-cutoff
  boundary anywhere, shifts are added first.

Pass for each, as the gate says: no cell of Φ_cut of G2.3 (cap of 10°, 4800 cells)
moves by more than 0.5%.

**G2.5(a), how "doubling the φ nodes (or azimuthal modes)" is read.** M against 2M,
from the partial sums of one run at 2M. In the design the orders are independent. The
agent shows this once on a reduced geometry: a run at M equals the partial sum of a run
at 2M to 1e-8. The μ grid and L are doubled in separate runs. The first solve (the
site, M = 12) gives the test for M = 6. If M = 6 fails it, M = 12 for all solves takes
C2 to or beyond the plan's compute and goes to the PI with G2.6.

**G2.1 and G2.5(b)** hold by construction in this design (D = 0 returns the C1 solve;
a rotation of the sky is a phase of the orders). They are run as written and test the
bookkeeping of the assembly only. G2.7 is the test of the m = 0 part.

**G2.2, how "H3a" is read.** The boundary of this gate uses the group weights of H3a,
as G1.1 did, on both sides. The solver side is the straight-ahead nonlinear path, one
chain per direction, not the mode path.

**G2.3(h)** is not evaluated in C2. The assembly does not cover the far side
(`DESIGN_C2.md`, 3.5).

**Conventions.** Transmission 0 on the lines of the ray family that never reach the
injection sphere. Transmission 1 above 60 GV. The azimuth of the cells is the arrival
azimuth of `tools/mc3d/scoring.py`.

**Reported with the skies:** the singular values of the deficits of the grid locations
(no solve). They say how many independent boundaries a set of locations needs, which
sets the cost of C5.

**Order.** G2.7, then the first solve with the full sky (the site, M = 12), whose cost
goes to the PI (G2.6) before the solves of G2.3. G2.0 is evaluated on the skies before
that solve.

### 2026-11-23. The reference of G2.3 and its count thresholds

**Reference.** `c2/mcref/c2ref_cutoff.npz`, sha256
`d293001c5e766f7ae0dadeb0a9b5ea2962185b796e2ab9431612322c01de0d39`, combined on
2026-11-23 from the twenty runs (report: `c2/mcref/WP23_REPORT.md`).
- Size: 5×10^5 primaries per run, 10^7 in all, with `--upgoing`. Cost: 36.93 core-h of
  CPU.
- Level of `n` against the field-free reference of C1 (`c1/mcref2`, tally `s`):
  0.9997 ± 0.0043.
- Cutoff factor per band, cap of 10°: 0.4135, 0.4657, 0.5449, 0.6327, 0.7103, 0.7866,
  0.8681, 0.9354, 0.9653, 0.9948 (errors 0.0014 to 0.0034; band 8: 0.0105).

**Thresholds.** Computed from the reference alone with the commands fixed in the
amendment above. Files in `c2/manager/thresholds/`: `calib.json` (sha256 beginning
`b8cbf29302ad1441`), `cap10/null_count_c2.json` (`c75056921859f12d`),
`cap5/null_count_c2.json` (`5e7ff953009ad9fb`).

| Part | Items counted | Threshold | First rule (decides nothing) |
|---|---|---|---|
| (a) cutoff factor per cell | 4800 | **N_a = 4639** | 4655 |
| (b) East-West harmonic per band and row | 100 | **N_b = 98** | 97 |
| (c) horizontal over vertical per band | 10 | **N_c = 8** | 8 |

Reported parts, cap of 10°: (d) 4504 of 4800, (f) 391 of 400. Cap of 5° (part g): (a)
4533 of 4776, (b) 95 of 100, (c) 8 of 10. In part (a), 2562 of the 4800 cells have the
3% floor as their tolerance.

**Calibration of (a) at the reference's own event counts.** 600 synthetic references,
model `nested`. An exactly right solver falls below N_a in 14.3 ± 1.4% of them (5%
intended; first rule 60.2%). Δ_a = 13: at 4626 the rate is 4.2%. As fixed in the
amendment:
- the gate decides on N_a = 4639;
- the rate is above 10%, so the result of (a) carries it as a caveat, pass or fail;
- a count from 4626 to 4638 is a FAIL that goes to the PI with these numbers.

**Standing.** These thresholds belong to this reference file. If the reference is
replaced (next entry), they are recomputed for the new file with the same commands,
before any comparison.
- The commands were fixed in commit `529b596`, before the reference was combined. The
  numbers follow from them and from the file.
- The threshold files were written between 15:31 and 15:38. The first solver output
  with a cutoff, a nonlinear solve of G2.7, was written at 15:43. This entry is
  committed after G2.7 (results log), so the file times are the record of that order.
- No output of the C2 solver has been compared with a tally. The manager opened the
  reference after commit `529b596` and has since read the result of G2.7. The manager
  compares neither with the other.
- This entry and the next hold more of the reference than the three counts (the cap
  radius, the azimuth of `n`, the effect of the roll). The solver agent is told not to
  read them before the freeze.

**Cap radius (reported).** The cutoff factor of the 10° cap is higher than that of the
smaller caps below 1 GeV: by 4.6 ± 1.2%, 3.8 ± 0.8% and 2.1 ± 0.3% against the caps of
2.5°, 5° and 7.5° in the lowest band, and by nothing above 1 GeV. The three ratios
follow the square of the cap radius in every band (χ² 0.1 to 2.1 for two degrees of
freedom; `c2/manager/dipole/caps.out`). That is what the average of a smooth, convex
function of the location gives: with K ∝ R_c^-1.7 and R_c ∝ cos⁴ of the geomagnetic
latitude, the estimate for the 10° cap is about 5%. Read as physical. It says that the
solver's cap average must carry the second order in the location, which G2.4 (a) tests.

### 2026-11-23. Azimuth artefact of the Monte Carlo: the event plane was not turned

**Found.** With no cutoff and no bending the flux at the site is flat in azimuth. The
tally `n` of the reference (copy 0 of each primary) is not. Sine component of `n`,
species and rows summed, cap of 10°, mean of the twenty runs:
- per band from 0.1 GeV up: −0.0507 ± 0.0027, −0.0360 ± 0.0031, −0.0163 ± 0.0031,
  −0.0064 ± 0.0034, −0.0070 ± 0.0036; the cosine component is zero within the errors;
- pooled over the three lowest bands: −0.0361 ± 0.0020 for all rows, −0.0552 ± 0.0039
  for cos Z 0 to 0.3.

More flux arrives from the west, most of all near the horizon and below 0.4 GeV
(`c2/manager/dipole/roll_test_before.out`, `rows.out`).

**Cause.** The event generators return every event with the same reaction plane, and
the Monte Carlo did not turn it.
- In the event pools the sum of the meson p_x per event is negative in every group of
  DPMJET-III pools (5 to 15 standard deviations per group) and for the meson
  projectiles of SIBYLL above 100 GeV (22 to 31). The sum of p_y is zero within the
  errors. The mean cos φ of the mesons about the beam goes down to −0.086 for
  DPMJET-III and to −0.013 for SIBYLL
  (`c2/manager/dipole/pool_azimuth_pools_fix_full.out`).
- `lorentz.rotate_to` puts the generator x axis on a fixed geographic direction: the
  cross product of the Earth's axis and the projectile direction. Every interaction
  therefore pushes its mesons toward the same side.
- This is an error of `tools/mc3d`. No run before today turned the events.

**Test, with its criterion written before the run was read**
(`c2/manager/dipole/CRITERION.md`). One run with the new option `--event-roll`: seed
1500000000, 5×10^5 primaries, otherwise the command of the reference; 2.64 core-h of
CPU; `c2/mcref_roll/sky/c2roll_s00_total.npz`, sha256 beginning `4c55f4eca615bb06`.
Statistic S: the sine component of `n` pooled over the three lowest bands. One run
without the roll scatters by 0.0090. Criterion: confirmed if |S| ≤ 0.018.
- Result: S = −0.0085, which is 0.9 standard deviations from zero and 3.1 from the mean
  without the roll. **Confirmed.**
- Reported: cos Z 0 to 0.3, −0.0159 (0.9 from zero, 2.3 from the mean without the
  roll); the lowest band alone, −0.0221 (1.8 from zero, 2.4 from the mean without the
  roll).
- What one run shows: the roll removes most or all of the dipole. The remainder is
  0.24 ± 0.25 of the value without the roll (`roll_test_after.out`).

**Effect on G2.3.** `n` enters G2.3 only through its azimuth mean, so its dipole does
not enter. `s` is filled from twelve copies of each shower, turned about the vertical of
the site. By argument, not measured: over the copies the sideways push averages out
except for a turn about the vertical (proportional to the sine of the latitude, 0.59),
which turns the East-West pattern of `s` by a small angle.
- Estimated size: 0.002 to 0.005 in the cosine component near the horizon below
  0.4 GeV (floor of (b): 0.01); up to about 1% in the cutoff factor of a cell there
  (floor of (a): 3%); a few to about ten cells in the count of (a). For comparison:
  Δ_a = 13, and the median of the null count is 28 cells above N_a.
- Measured with the one run (reported, no criterion; `roll_s_after.out`; in brackets the
  standard deviations of one run). Cosine component of `s` pooled over the three lowest
  bands, roll minus no roll: −0.0019 (1.8) for all rows, −0.0020 (0.7) for cos Z 0 to
  0.3. Sine component: −0.0017 (1.1) and +0.0034 (0.8). Cutoff factor per band: no band
  beyond 1.6. Horizontal over vertical: no band beyond 1.5. The sign and the size of the
  cosine shift are those of the estimate. It is not significant: one run cannot measure
  an effect of this size, and these deviations are wider than Gaussian (rms 1.15 to
  1.22 on the reference runs).
- Flux level of the run with the roll in the band 4 to 6.3 GeV: 23% (`n`) and 26% (`s`)
  above the mean of the reference runs (2.1 and 3.5 standard deviations). The cutoff
  factor does not move (0.7). The reference has one run 34% above its mean in `n` in
  that band. Read as rare heavy showers, not as an effect of the roll. Not followed up.

**Where the artefact is and where it is not.**
- Not in the solver. Its kernel sets are Legendre moments about the projectile
  (`kernelset_mc.py`), which is the average over the angle about the beam. The
  reference and the solver therefore differ by this artefact.
- Not at first order in the comparisons of C1. They used azimuth averages, and a turn
  about the vertical does not change an azimuth average (argument, not measured).
- In every tally with azimuth bins that `tools/mc3d` has produced, among them the Monte
  Carlo behind the product table `kamioka_3D_v2l.npz`. Its size there is not measured
  and is expected to be that of the estimate above. It becomes a known limitation in
  `tools/mceq3d/INTEGRATION.md` once it is measured.

**Code.** `sky3d.py --event-roll` (`pool.EventPool(roll=True)`): every drawn event is
turned by a random angle about its projectile. Off by default: every earlier run, its
signature and its random stream are unchanged. Test:
`test_pool.py::test_event_roll_is_a_rigid_random_rotation_about_the_beam`.

**Standing of the reference: open, for the PI.** The gate text names the twenty seeds
of the reference, and a second Monte Carlo reference is beyond the compute authorised.
Put to the PI on 2026-11-23:
- (A) keep the reference, with the artefact on record at its estimated size;
- (B) replace it by twenty runs with `--event-roll` (seeds 1500000000 + 20000000 j; the
  run above is j = 0). The thresholds are then recomputed with the same commands before
  any comparison. The present reference is kept and is compared with nothing but the
  new one.

Until the PI has answered, no solver output with the cutoff is compared with a tally.

### 2026-11-23. Reference of G2.3 replaced: twenty runs with the event plane turned

Decision of the PI of 2026-11-23 (last section), on the entry above. Written before the
runs j = 1 to 19 exist and before any solver output with the cutoff is compared with a
tally.

**What changes in G2.3.** Only the reference.
- Command: that of the first reference with `--event-roll` added
  (`c2/mcref_roll/scripts/run_roll.sh`). Code: the snapshot `c2/mcref_roll/code/`, equal
  file by file to `tools/mc3d` at commit `97a87be`.
- Twenty runs of 5×10^5 primaries, seeds 1500000000 + 20000000 j, j = 0 to 19. No
  seed of an earlier run: the task seeds of a run lie within 5×10^6 above its seed
  (`sky3d.build_tasks`), and the first reference ends at 1280000000.
- The tallies, the statistics, the parts (a) to (g) and their floors are unchanged.

**Run j = 0** is the run of the test in the entry above. It was made with this command
and this seed before this amendment, and the manager has read its tallies (same entry).
It enters as it is. No run is left out or repeated for what it contains. A run that
ends with an error is repeated with its seed.

**Runs j = 1 to 19:** `run_roll.sh 1 19 8-11,20,22-23 7`. Run 0 used six processes
and these use seven. By the code the number of processes sets no random stream: the
seed of a task is fixed by the seed of the run, the nucleus group, the stratum and the
chunk.

**Combination and checks.** `c2/mcref_roll/scripts/combine_roll.py` is `combine_c2.py`
of the first reference with the paths, tags, seeds and command of this one. Output:
`c2/mcref_roll/c2roll_cutoff.npz`, same arrays and units. Checks as for the first
reference: integrity of the twenty totals, the number of showers, the level of `n`
against the field-free reference of C1.

**Thresholds.** N_a, N_b and N_c are recomputed on the new file with the commands of
the amendment of 2026-11-23, outputs in `c2/manager/thresholds_roll/`. They are written
here as a dated entry before any comparison. The thresholds of the first reference no
longer gate anything and stay on record.

**The first reference** (`c2/mcref/c2ref_cutoff.npz`) is kept. No solver output is
compared with it. It is compared with the new one only, as follows.

**The two references compared (reported, not a gate; fixed here).** By the manager,
script `c2/manager/dipole/two_refs.py`, cap of 10°. Error of a difference: the
jackknife errors of the two references in quadrature.
1. `n`, the species summed: sine and cosine components per band and pooled over the
   three lowest bands, for all rows and for cos Z 0 to 0.3, in both references. The
   roll is said to remove the artefact if the pooled sine component of the new
   reference is within two errors of zero in both sets of rows. Otherwise the remainder
   is reported as unexplained.
2. `s`: a_c and a_s pooled over the three lowest bands, for all rows and for cos Z 0
   to 0.3, new minus first. This is the size of the artefact in the quantity of
   part (b). It is set against the estimate of the entry above (sign and size).
3. Per band and row: the differences of a_c and a_s. Reported: the largest, the number
   above the floor of 0.01, and the rms of the differences over their errors.
4. K per cell: the rms of the differences over their errors, the number beyond 2 and
   beyond 3, the number of cells that differ by more than 3% of K, and the mean
   relative difference per band for cos Z 0 to 0.3.
5. K per band and H per band (the statistics of (e) and (c)): the differences over
   their errors.

**Product table.** Items 2 to 4 measure the artefact in this configuration (cutoff on,
no bending). For `kamioka_3D_v2l.npz`, made with bending, they are an estimate and not
a measurement. The entry in `tools/mceq3d/INTEGRATION.md` is written with them.

**Compute.** 39 to 50 core-h for the nineteen runs and about 2 for the thresholds,
authorised by the decision. C2 then stands at about 112 to 135 core-h at the design's
solve costs, against the plan's about 100. The cost of the solves is open (G2.6).

**Blindness.** The agents that write the solver and the cutoff skies do not open
`c2/mcref_roll/` before the freeze, as for `c2/mcref/`.

### 2026-11-23. G2.0 triggered: the check of the sky before the freeze

**Outcome of G2.0: TRIGGER in one class of 100** (solver agent, milestone B; result
file `c2/solver/gates/G2.0/G2.0.md`, sha256 beginning `298f839300be9ea1`).
- 64 916 covered states below 60 GV, 2 000 of them on lines that miss the ground. All
  states: sky 0.1741, verdicts 0.1718.
- The class: 5 to 10 GV, cos Z at injection 0.25 to 0.5, south, 659 states. Sky
  0.1062, verdicts 0.0774, difference +0.0288, tolerance 0.0208.
- The gate text asks for a check of the sky before the freeze. The agent stopped there
  and handed back.

**Checked by the manager** (`c2/manager/g20/recheck.out`, from the stored states; no
tracing). The class table of the agent is reproduced. The difference is not one
fluctuation. Mean of d = sky − verdict over the states, with the error of that mean:
- 4 to 5 GV +0.0037 ± 0.0007; 5 to 10 GV +0.0108 ± 0.0017, positive in 10 of 12
  classes; 10 to 20 GV −0.0020 ± 0.0025; 20 to 60 GV −0.0015 ± 0.0010.
- In counts: the sky allows about 150 states more than the 11 150 of the verdicts
  (+1.3%), about +170 at 4 to 9 GV and −20 above. At 5 to 10 GV it is +7.4%.
- Where the sky gives a value between 0.1 and 1, the verdicts are lower by 0.04 to
  0.06.
- These errors hold the test sample only. The draws of the tables (one azimuth and one
  rigidity per cell, the same at every location) are a second source of scatter that
  is frozen in the sky. Its size is not known.

**Question of the check.** Is the excess at the foot of the cutoff edge (a) a bias of
the interpolation between locations, (b) a bias of another interpolation, (c) the
frozen scatter of the table draws? How large is it in the quantity the solver uses?
- Why (a) is the first suspect: at fixed direction and rigidity the transmission is a
  step in the location. The cutoff rigidity changes by about 18% over 5° of latitude.
  The bilinear value between two tables is a ramp over that range, so the edge is
  smeared: too much transmission at its foot, too little at its top.

**Steps, each written down before it was run** (`c2/manager/g20/CHECK.md`: steps A to
C before step A; step D after step A). This entry is committed before the tables of
B, C and D are read.
- **A. Stored states, no tracing.** Least squares of d on c_x = 4 t_x (1 − t_x), c_y
  and c_n for 4 to 10 GV, with robust errors; t_x, t_y the weights of the state's
  location on the grid, t_n its weight between two line nodes. Predicted slope on c_y
  if the latitude makes all of the excess: about 0.013. Expected errors about 0.005.
  A can point; it cannot decide.
- **B. Five tables at mid locations**, with the draws of the base tables (same seed,
  nodes, sectors, ladder; h_ref = 0): (0, ±2.5°), (±2.5°, 0), (2.5°, 2.5°). The
  bilinear value of the 5° tables against the direct table, state by state.
  - B1: allowed count over the ground line nodes and the sectors,
    E = N_bilinear / N_direct − 1, error by a jackknife over the 36 sectors. Ranges:
    foot 4.27 to 10.73 GV, middle 10.73 to 19.07 GV, top 19.07 to 60 GV, all. Tangent
    nodes apart.
  - Prediction if (a) makes the excess: about +11% in the foot range at (2.5°, 2.5°),
    negative at the top. A scale, not a value: the weights of G2.0 and of this count
    differ.
  - **Criterion.** E_foot at (2.5°, 2.5°) of +5.5% or more and at least two errors
    from zero: (a) explains the excess. Below +2.75%: it does not. Between: in part.
  - B2, reported: the same comparison in the solver's quantity. Transmission of the
    proton group per energy bin (group weights of the solver, mean over the sectors),
    bilinear over direct minus 1, mean over the ground line nodes, per location. Read
    against 0.5% (G2.4) and 3% (floor of G2.3(a)).
- **C. A second realisation of the site table**, seed 20261124. Against the base site
  table: the allowed count per range of B1 and the quantity of B2. The difference over
  √2 is the scatter of one realisation at one location. Reported. It does not say
  whether the excess of G2.0 is frozen scatter; that needs a second set of 25 tables
  (2.4 core-h), built only if B says (a) does not explain the excess.
- **D. A site table on the midpoints of the line nodes**, seed 20261125: the 30
  midpoints in ground zenith angle (the horizon included) and the 14 midpoints in
  tangent altitude. For each midpoint, the mean of its two neighbours in the base site
  table against the direct table.
  - E = N_interpolated / N_direct − 1 over the ground midpoints, in the ranges of B1
    and for ground zenith 0 to 60°, 60° to 80° and 80° to 90°; jackknife over the
    sectors. The draws of different nodes are independent, so the expected error is
    about 0.3% for all ranges together.
  - If the line-node term of step A (below) is a bias of the interpolation: E_all
    about −3%. If |E_all| is below 1% and within two errors of zero, the term is read
    as chance. Reported, with the quantity of B2.
- The seven tables are built by the manager with `boundary.py build` as committed
  (`c2/manager/g20/sky/`, `sky_seed2/`, `sky_mid/`), about 0.7 core-h. They are check
  tables. No solve uses them.

**What follows, fixed before the steps.**
- G2.0 stays on record as TRIGGER whatever the check finds.
- If (a) explains the excess: the 56 further tables of the 2.5° grid are built now
  (planned for G2.4(a), 5.4 core-h), by the solver agent. G2.0 is evaluated again on
  the 2.5° grid with the same stored states and verdicts, no tracing, and reported
  next to the first. Prediction: the excess at 5 to 10 GV falls to half or less. The
  cap-averaged deficits of the two grids are compared with no solve. The spacing for
  the solves is chosen on that comparison and goes to the PI with the cost (G2.6).
  The cost of a solve does not depend on the spacing. G2.4(a) is run as written in
  any case.
- If (a) does not explain it: a second set of tables with another seed, then the line
  nodes. Written down before it is run.
- The solve of the site at M = 12 waits for the outcome. The cost measurements on
  short runs do not depend on the content of the sky and go on.
- The manager has opened the reference of G2.3. Nothing in this check uses it: every
  comparison is between the sky and direct back-tracing.

**Step A, result** (`c2/manager/g20/step_a.out`, `step_a2.out`). The sky value
recomputed from the tables equals the stored one.
- 4 to 10 GV, 19 132 states, mean d +0.0088 ± 0.0013. Slopes: c_y +0.0141 ± 0.0041,
  c_x +0.0044 ± 0.0044, c_n −0.0085 ± 0.0043; constant +0.0019 ± 0.0052.
- d by the place between two locations in y: +0.0033 ± 0.0020 near a location,
  +0.0100 ± 0.0023 between, +0.0127 ± 0.0024 at mid distance.
- The allowed fraction of the verdicts against y is strongly convex: 0.0034, 0.0159,
  0.0337, 0.0813, 0.1504, 0.2622 in bands of 2.5° from −7.5° to +7.5°.
- Reading: A points to the latitude. The slope on c_y is the predicted one and the
  constant is compatible with zero. B decides.
- Not predicted: the slope on c_n is negative, −0.0085 ± 0.0043 at 4 to 10 GV,
  −0.0219 ± 0.0083 at 10 to 20 GV, −0.0049 ± 0.0015 over all states. Near a line node
  the sky is 0.010 above the verdicts, between the nodes 0.004. Not understood. The
  errors leave out the frozen draws of the tables, which states near the same node
  share. Step D was added to measure it.
- d against the place in the rigidity half-interval, 4 to 10 GV, in quarters from low
  to high: +0.0288, +0.0156, −0.0001, −0.0091 (± 0.0026). A gradient is expected with
  one draw per half-interval, since the transmission rises with rigidity. Its mean is
  the excess.

### 2026-11-23. The new reference of G2.3 (event roll) and its count thresholds

**Reference.** `c2/mcref_roll/c2roll_cutoff.npz`, sha256
`fb226760abacc2f40963d5e48f8264830a5349baf47a9b589592965c9775fc50`, combined on
2026-11-23 at 22:06 from the twenty runs with `--event-roll`
(`c2/mcref_roll/scripts/combine_roll.py`). From here on it is the reference of G2.3.
- Runs 1 to 19 ended at 22:05, each with exit 0. Size: 5×10^5 primaries per run, 10^7
  in all, with `--upgoing`. Cost: 37.14 core-h of CPU for the twenty (run 0: 2.64;
  runs 1 to 19: 34.50, against the 39 to 50 of the entry above).
- Integrity (`scripts/roll_integrity.py`, output `logs/roll_integrity.out`): twenty
  totals of 47350185 bytes with the same keys, shapes and edges; seeds 1500000000 +
  20000000 j; 500000 showers and 312 tasks in each; no two totals equal; all tallies
  finite; `event_roll` set in the twenty manifests, the other options as in the first
  reference (no bending of the primaries, no field on the muons).
- Level of `n` against the field-free reference of C1 (`c1/mcref2`, tally `s`), cap of
  10°: 1.0031 ± 0.0045 (`logs/roll_checks.out`). Per band, nine of ten are within 1.5
  errors of 1. The band 0.63 to 1 GeV is at 0.9798 ± 0.0063 (−3.2 errors). Over the 400
  cells of species, band and row: rms of the pulls 1.11, one beyond 3 (+3.19). Smaller
  caps: 0.9980 ± 0.0063 (7.5°), 0.9932 ± 0.0114 (5°), 0.9882 ± 0.0200 (2.5°).
- Cutoff factor per band, cap of 10°: 0.4144, 0.4669, 0.5439, 0.6263, 0.7141, 0.7884,
  0.8699, 0.9434, 0.9760, 0.9929 (errors 0.0013 to 0.0037).

**Thresholds.** Computed from the reference alone with the commands fixed in the
amendment of 2026-11-23 (`c2/manager/thresholds_roll.sh`). Files in
`c2/manager/thresholds_roll/`: `calib.json` (sha256 beginning `b3596e8d2f53be08`),
`cap10/null_count_c2.json` (`a18f7d8c73d1783a`), `cap5/null_count_c2.json`
(`db22225063607e27`).

| Part | Items counted | Threshold | First rule (decides nothing) |
|---|---|---|---|
| (a) cutoff factor per cell | 4800 | **N_a = 4640** | 4654 |
| (b) East-West harmonic per band and row | 100 | **N_b = 97** | 97 |
| (c) horizontal over vertical per band | 10 | **N_c = 8** | 9 |

Reported parts, cap of 10°: (d) 4499 of 4800, (f) 391 of 400. Cap of 5° (part g): (a)
4543 of 4788, (b) 94 of 100, (c) 8 of 10; twelve cells of the 5° cap are filled in
fewer than half of the runs and are not counted. In part (a), 2631 of the 4800 cells
have the 3% floor as their tolerance. The thresholds of the first reference were 4639,
98 and 8.

**Calibration of (a) at the reference's own event counts.** 600 synthetic references,
model `nested`. An exactly right solver falls below N_a in 17.3 ± 1.5% of them (5%
intended; first rule 61.0%). Δ_a = 18: at 4622 the rate is 4.8%. As fixed in the
amendment:
- the gate decides on N_a = 4640;
- the rate is above 10%, so the result of (a) carries it as a caveat, pass or fail;
- a count from 4622 to 4639 is a FAIL that goes to the PI with these numbers.

**Standing.**
- The commands were fixed in commit `529b596`. `gates/c2_stats.py`,
  `gates/null_count_c2.py` and `gates/calib_null_c2.py` were at their committed state
  (sha256 beginning `0b54bc056c47b405`, `fe9160e1fa9d78bc`, `0baedfc127e49f44`; working
  tree clean at `5f586a8`). The numbers follow from them and from the file. Run from
  22:08:32 to 22:12:26, 0.2 core-h.
- What the manager had read of the new reference before the thresholds existed: the
  tallies of run 0 (entry on the artefact), the integrity check and the checks of
  `logs/roll_checks.out` and `logs/roll_azcheck.out`. While the threshold commands ran,
  the manager looked at the azimuth of `n` and of the up-going tallies per row
  (`c2/manager/dipole/roll_n_rows.out`, `roll_n_rows_up.out`; results log). None of
  this enters a threshold.
- Solver outputs with the cutoff that exist at the time of this entry: the solves of
  G2.7, the sky values of G2.0 and of the check of the sky, and the tangent solves of
  the site at M = 12 (three batches of five written, the fourth running). None has
  been compared with a tally of either reference. The manager has read the results of
  G2.7 and of G2.0. Of the site solve the manager has read the log of times only.
- The two references have not yet been compared by `two_refs.py`. That comparison is
  run after the commit of this entry.
- This entry holds content of the reference. The solver agent is told not to read it
  before the freeze.

### 2026-11-24. After the site solve: azimuth order, location grid, G2.4(a), two stages

Decision of the PI of 2026-11-24 (last section), on the Results-log entry of 2026-11-23
"Site solve at M = 12". Committed before every run it governs. At this commit there is
no tangent of an order above 12, no cap solve, no run of G2.1, G2.2, G2.4, G2.5(b) or
G2.8, and no table of a new grid node in the solver's sky directory. (The manager's
check tables of step E at (0, ±1.25°) exist; they serve as a cross-check below.) The
entries of 2026-11-23 hold where this one does not change them.

**1. Azimuth order: M = 8 for every solve of C2.**
- G2.5(a) on the orders stays FAILED at M = 6 (0.507% against 0.5%). M = 6 is not used.
- The test of M = 8 is the gate's, as read on 2026-11-23: M = 8 against M = 16 at the
  site, from the partial sums of one run at 16. That run is the site solve of
  2026-11-23 (orders 0 to 12, kept as it is) plus the eight tangents of the orders 13
  to 16: same solver code (unchanged since commit d7ede51), same settings, same sky
  table (sha256 beginning `0a382659062ef54d`).
- Cells and evaluation as on 2026-11-23: `gates/g2_5.py orders --M 16`. The script is
  committed with this entry. With `--M 12` it gives the numbers of the result file of
  2026-11-23 again.
- Pass: no cell of the 4800 moves by more than 0.5% between M = 8 and M = 16.
- If it fails, it is reported as failed, and no cap solve is run before the PI has
  decided.
- What was seen before this entry: the partial sums of the site run up to order 12.
  M = 8 against 12 moves at most 0.137%, and the choice of 8 was made with that in
  view. The orders 13 to 16 are not solved yet. Expected for the test, from the entry
  of 2026-11-23: 0.16 to 0.30%, most of it draw noise of the table.
- What the test does not cover: the sector width of the sky tables (36 sectors of 10°,
  orders up to 18). No gate doubles it. On record for it: G2.0 by quadrant, G2.8 at
  M = 18, and the three seeds of the site table (in the boundary no signal above the
  draw noise beyond order 10).
- A boundary at M = 8 has 17 tangents.

**2. Location grid of the cap solves: 5° in x (east), 1.25° in y (north).**
- Grid B: x = −10, −5, 0, 5, 10 and y from −10 to 10 in steps of 1.25, in the
  azimuthal equidistant coordinates about the site; 85 nodes. 45 of its tables exist
  (those of the 2.5° grid at these x). 40 are built: the same seed (the same draws),
  h_ref = 0, the same line nodes, sectors and rungs as the 81.
- Reason (entry "G2.0 triggered" and the Results log of 2026-11-23): only the spacing
  in latitude counts. By the square law the cap-averaged proton boundary of the 2.5°
  grid is off by about 0.55% by number (cap of 10°) and that of grid B by about 0.14%.
- `LocationGrid` takes one spacing per axis. With equal spacings it gives the weights
  of the present class (unit test).
- Checks by the manager: the cap weights of grid B against his own quadrature
  (`c2/manager/g20/cap_compare.py`); the new tables at (0, ±1.25°) against those of
  step E (equal allowed arrays expected).
- G2.0 is evaluated again on grid B from the stored states and verdicts of the gate
  (no tracing), classes and trigger as written. The evaluation on the 5° grid
  (TRIGGER) stays on record.
- Reported with the tables: the singular values of the 85 deficits.

**3. G2.4(a): the pair is replaced before its run.**
- Not run: 5° against 2.5° (entry of 2026-11-23). G2.0 triggered on the 5° grid, the
  cap-averaged proton boundaries of the two differ by 1.65% ± 0.12% by number (cap of
  10°), so a failure was foreseen; and no solve uses the 5° grid.
- The pair now: grid A (5° in x, 2.5° in y, 45 nodes) against grid B (85 nodes, which
  contain the 45). Cap of 10°, M = 8, everything else equal.
- Pass, as the gate says: no cell of Φ_cut of G2.3 (4800 cells) moves by more than
  0.5% between A and B.
- Reading. B is the finer grid of the pair and the grid of the solves. If the error
  goes with the square of the spacing (step E supports this roughly), the error of B
  is a third of the move.
- Expected, written before the run: the cap-averaged boundaries of A and B differ by
  about 0.4% by number. The largest cell may come close to 0.5%.
- A FAIL is reported as failed and goes to the PI (stop rule of decision point 2). The
  next finer grid, 0.625° in y, would need 80 more tables.
- The spacing in x is not tested by a solve. On record for it, at the level of the
  boundary only: 5° against 2.5° in x, at 2.5° in y, moves the cap-averaged proton
  boundary by −0.027% ± 0.026% by number (cap of 10°). This narrows "halving the
  spacing of the location grid" to the latitude. The PI was asked in these words
  ("2.5° against 1.25° in latitude").

**4. Order of the work: two stages.** From here on this replaces the paragraph "Order"
of 2026-11-23, and it amends the bullet of section C2 on rule 4.
- Stage 1, released by the decision, in this order:
  1. the orders 13 to 16 at the site, and G2.5(a) on the orders for M = 8;
  2. the grid code, the 40 tables, G2.0 on grid B, the report of the skies; G2.2 and
     G2.8, which need no cap solve;
  3. if 1 passed, the cap solves at M = 8: cap of 10° on B (G2.3), cap of 5° on B
     (G2.3(g)), cap of 10° on A (G2.4(a)); with them G2.1, G2.4(a), G2.5(b), G2.5(c)
     for every solve, and G2.6;
  4. the freeze, `RESULTS_C2.md`;
  5. G2.3, by an agent that wrote neither the solver, the kernel sets nor the skies.
- Stage 2, not released: G2.4(b) (the reference altitude) and the doublings of the μ
  grid and of L of G2.5(a). The PI decides on it with the result of G2.3. Until they
  have run, G2.4 and G2.5 are incomplete and are reported as such.
- Rule 4, as amended. The settings are written in `RESULTS_C2.md` after G2.0, G2.1,
  G2.2, G2.4(a), G2.5(a) on the orders, G2.5(b), G2.5(c), G2.7 and G2.8, and before the
  solver's output with the cutoff is compared with a Monte Carlo tally for the first
  time. G2.4(b) and the two doublings are run on the frozen state. If one of them
  fails and a setting is changed, the change is listed there with its reason and G2.3
  is re-run in full.
- What the staging costs. G2.3 is first evaluated on a state whose μ grid, L and
  reference altitude are tested without the cutoff only (C1, G1.2: every neutrino cell
  within 0.16%). A re-run of G2.3 after a change is a second comparison with the same
  reference. Both results are then reported.
- Guard. The agents that write the solver and the skies do not open the reference,
  its thresholds or the result files of G2.3 until stage 2 has run, unless the PI
  decides otherwise. A setting changed after the freeze follows from the numerical
  gate that failed and from nothing else.
- After G2.3 the manager reports to the PI before anything else is run (stage 2, or a
  diagnosis cycle).
- Compute of stage 1: about 17 to 18 core-h (orders 1.7; tables 3.9; three cap
  boundaries of 17 tangents 10.5; the other checks 1 to 2). C2 then stands at about
  116 core-h. What stage 1 needs beyond 20 core-h goes to the PI before it is spent.

**5. Not changed.** G2.1, G2.2, G2.3, G2.5(b), G2.5(c), G2.6, G2.7 and G2.8 as written.
The kernel sets, h_ref = 0, the line nodes, the 36 sectors. One draw per state in each
table: the draw noise measured on 2026-11-23 is accepted for C2. It is small against
the 3% floor of G2.3(a), and the tables of grids A and B hold the same draws.

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
- **G2.7: PASS on the first setting** (2026-11-23; solver agent, milestone A; result
  file `c2/solver/gates/G2.7/G2.7.md`, sha256 beginning `7a94fbd6f5d97ec3`; report
  `c2/solver/WP_SOLVER_REPORT_A.md`). Guard: ρ = 0.1 on the source shape, the source
  integral and the loss interpolation; the Simpson correction not guarded. The tuning
  cycle was not run.
  - Deficit 1 (isotropic step): (i) 0.439% against 0.5%; (ii) 0.237% against 1%, over
    291 cells. Deficit 2 (site sky, order 0): (i) 0.386%; (ii) 0.237%, over 308 cells.
    Every worst value is in one cell: anti-nu_e, 0.10 to 0.16 GeV, cos Z 0 to 0.1.
  - The margin of (i) is small, as the gate text expected. Only the two lowest bands
    are above 0.011%.
  - Ground muons (reported): up to 4.9% and 4.0% of the flux, 7.6% and 7.7% of the
    deficit, both at 0.224 GeV and cos Z 0.05.
  - Checked by the manager. Both criteria recomputed from the stored cells give the
    same numbers. The C1 cells of the gate files equal the stored cells of the frozen
    C1 solve bit for bit. The 26 unit tests pass in the agent's log; the manager re-ran
    the eleven of `test_boundary.py`. The manager did not re-run the solves.
  - Deviations from the design, as the agent reports them: three up-scatter passes in
    the tangent (design: two); Φ_lin in the cells built as in section 3.3 of the
    design; the deficit clipped to [0, B⁰] against a rounding excess of 1e-15.
  - Cost. Nonlinear solves: 17.5 core-min and 5.2 GB each. Tangent run (primal and two
    tangents): 47.9 core-min and 11.3 GB, about 15 core-min per tangent. The design had
    0.33 to 0.41 of a C1 solve per tangent (4 to 5 core-min). To be measured at the
    batch size of the full-sky solve (G2.6). Site sky table: 50 220 states, 290 core-s.
    Compute of the agent: about 2.5 core-h of 4.
  - The setting goes into `RESULTS_C2.md` at the freeze.
- **2026-11-23. G2.0: TRIGGER in one class of 100** (solver agent, milestone B; result
  file `c2/solver/gates/G2.0/G2.0.md`, sha256 beginning `298f839300be9ea1`; report
  `c2/solver/WP_SOLVER_REPORT_B.md`).
  - 64 916 covered states below 60 GV: 62 916 hit the ground, 2 000 miss it. Sky
    0.1741, verdicts 0.1718.
  - Trigger: 5 to 10 GV, cos Z at injection 0.25 to 0.5, south, 659 states. Sky
    0.1062, verdicts 0.0774, difference +0.0288, tolerance 0.0208. The same class
    triggers among the states that hit. No class triggers among those that miss.
  - 500 states above 60 GV on lines that miss the ground: all allowed.
  - Checked by the manager. The class table recomputed from the stored states gives
    the same numbers. The difference is systematic at the foot of the cutoff edge:
    +0.0108 ± 0.0017 at 5 to 10 GV. The manager did not trace the states again.
  - The check of the sky that the gate text asks for is the dated entry "G2.0
    triggered". The solve of the site at M = 12 waits for it.
  - Milestone B so far, besides the gate: the 25 sky tables of the 5° grid
    (`c2/solver/sky/`, `SKIES.md`); `assemble3d.py` with six tests, which the manager
    ran again (6 passed); timing output and a compilation cache in `solver3d.py`.
    `test_solver3d.py` has not been run in full since that change.
- **2026-11-23. Check of the sky after G2.0: the 5° spacing in latitude makes the
  excess** (manager; `c2/manager/g20/step_bcd.out`, sha256 beginning
  `e8de2c38d91fc9f4`; steps and criterion in the dated entry "G2.0 triggered",
  committed at 16:55 before any check table was read; the last table was finished at
  17:00). Seven check tables, built with `boundary.py build` as committed, 0.5 core-h.
  Nothing here uses the reference or a solve.
  - **B1, the criterion.** Bilinear value of the 5° tables against the direct table
    with the same draws: allowed count at 4.27 to 10.73 GV over the ground line nodes.
    At (2.5°, 2.5°): +15.3% ± 3.1%, 4.9 errors from zero. The criterion was +5.5% and
    two errors: **(a) explains the excess.** The prediction was about +11%.
  - It is the latitude. (0, +2.5°): +15.6% ± 3.2%. (0, −2.5°): +22% ± 7%.
    (+2.5°, 0): −1.3% ± 0.7%. (−2.5°, 0): +1.0% ± 0.9%.
  - Above the foot, at (2.5°, 2.5°): +0.8% ± 1.0% at 10.73 to 19.07 GV and
    −0.05% ± 0.12% above. The predicted deficit at the top of the edge is not seen.
    Tangent line nodes at the foot: +3.1% ± 2.4%.
  - **B2, in the solver's quantity**: transmission of the proton boundary per energy
    bin, mean over the ground line nodes and the sectors. At (0, +2.5°) the bilinear
    value is too high by 9% to 23% per bin from 3.2 to 7.9 GeV (errors 3% to 5%), by
    4% ± 3% at 7.9 to 10 GeV and by less than 1.3% above. Below 3.2 GeV, where the
    transmission is 2% or less, by 29% ± 10% and more.
  - B2 summed over the bins below 60 GV, by number and by energy. (0, +2.5°):
    +6.2% ± 0.8% and +2.7% ± 0.5% (neutrons +5.9% and +2.3%). (0, −2.5°):
    +3.3% ± 0.7% and +1.3% ± 0.4%. (+2.5°, 0): −0.45% ± 0.16% by number.
    (−2.5°, 0): +0.27% ± 0.24%.
  - Read against the scales (estimate, not measured). The error is zero at a table
    and largest half-way, so a cap average carries about two thirds of the mean of
    north and south: about 3% by number and 1.3% by energy at 5°. That is above the
    0.5% of G2.4 and of the size of the 3% floor of G2.3(a). The 5° grid is not fine
    enough in latitude. The comparison of the two grids below measures it.
  - **C, a second realisation of the site table** (seed 20261124) against the base:
    foot count −2.1% ± 2.0%, all rigidities −0.18% ± 0.20%, boundary by number
    −0.5% ± 0.5%. The frozen draws of a table are not the cause.
  - **D, the line nodes.** Mean of two neighbouring ground nodes against the direct
    table on the 30 midpoints: allowed count −0.07% ± 0.20% over all rigidities,
    −1.3% ± 2.5% at the foot. By the reading fixed before, the line-node term of
    step A is chance. Tangent midpoints: −7.0% ± 2.8% at the foot (2.5 errors),
    −0.5% ± 0.3% over all rigidities. Noted, not followed up yet.
  - **Consequence, as fixed before.** G2.0 stays TRIGGER. The solver agent was told
    the outcome at 17:03 and builds the 56 further tables of the 2.5° grid. G2.0 is
    then evaluated again on that grid from the stored states, and the cap-averaged
    deficits of the two grids are compared with no solve. The spacing for the solves
    goes to the PI with G2.6. The site solve at M = 12 waits.
  - **Step E, added before it was run** (`c2/manager/g20/CHECK.md`, sha256 beginning
    `d37e86f5845ed30d`): the error of the spacings 2.5° and 1.25° in latitude, from
    direct tables at (0, ±1.25°) and (0, ±0.625°) with the draws of the base tables.
    Prediction: a quarter and a sixteenth of the values at 5°. Reported, no criterion.
- **2026-11-23. Step E of the check of the sky: the error falls with the square of
  the spacing in latitude** (manager; `c2/manager/g20/step_e.out`, sha256 beginning
  `d6acd04ad15ff097`; `step_e.py`). Four direct tables at (0, ±1.25°) and
  (0, ±0.625°) in `c2/manager/g20/sky_fine/` (sha256 beginning `a5d082b223b54158`,
  `1702ab39e18fd879`, `b0ff59aeb6145aaf`, `74c7e6b5400f9060`), with the draws of the
  base tables, 0.32 core-h. No reference, no solve. Compared: the mean of two tables
  one spacing apart in latitude against the direct table half-way, ground line nodes.
  - Control: the rows of 5° repeat step B (+15.6% and +22% at the foot).
  - Allowed count at 4.27 to 10.73 GV. Spacing 2.5°: +3.1% ± 1.6% north and
    +6.2% ± 2.4% south (predicted +4% and +5.5%). Spacing 1.25°: +1.9% ± 2.1% north
    and −0.9% ± 1.8% south (predicted +1% and +1.4%).
  - Proton boundary summed over the bins below 60 GV, by number. Spacing 2.5°:
    +0.72% ± 0.44% north and +1.47% ± 0.48% south (predicted +1.5% and +0.8%).
    Spacing 1.25°: +0.48% ± 0.55% north and +0.21% ± 0.38% south (predicted +0.4%
    and +0.2%).
  - Mean of north and south, by number and by energy. 5°: +4.75% ± 0.54% and
    +1.97% ± 0.29%. 2.5°: +1.10% ± 0.33% and +0.49% ± 0.15%. 1.25°: +0.35% ± 0.34%
    and +0.16% ± 0.15%. Ratios to 5° by number: 0.23 and 0.07, against 0.25 and
    0.0625 predicted.
  - Reading. The mean of the two sides follows the square law. Taken alone at 2.5°,
    the north is 1.7 errors below its prediction and the south 1.3 errors above. At
    1.25° no value differs from zero: the tables bound the error there at about 1%
    (two errors); the 0.3% is the square law carried down from 5°, not a
    measurement.
  - Per energy bin at 2.5°, 3.2 to 10 GeV: between −0.4% and +3.3% in the north
    (errors 1.2% to 2.5%), between −1.9% and +13% in the south (errors 1.1% to
    4.2%; the +13% ± 4% is the bin of 3.2 to 4.0 GeV, transmission 0.02). Above
    10 GeV 0.6% or less on both sides.
  - Tangent line nodes at the foot, 2.5°: −3.4% ± 1.4% north, +2.9% ± 2.9% south;
    1.25°: −1.0% ± 2.0% and −3.6% ± 1.7%. No common sign.
  - Estimate for a cap average (two thirds of the mean of the two sides; not
    measured), by number and by energy: 5° 3.2% and 1.3%; 2.5° 0.7% and 0.3%; 1.25°
    in latitude 0.2% and 0.1% by the square law. The 2.5° grid is at the 0.5% of
    G2.4, not clearly under it. The comparison of the cap averages on the 81 tables
    measures the difference between 5° and 2.5°.
  - State. The usage limit stopped the session and the solver agent from about
    17:25 to 20:15. None of the 56 tables of the 2.5° grid existed then. The agent
    was told at 20:15 to build them and was given the numbers above (they hold
    nothing of the reference). Its cost runs with 1, 2 and 4 tangents had ended
    before the stop and are not yet in its report.
- **2026-11-23. Step F of the check of the sky: the tangent nodes stay; the errors
  of the ground nodes hold** (manager; `c2/manager/g20/step_f.out`, sha256 beginning
  `481dc91db566509a`; `step_f.py`, `2a811c6cc60e1392`, written while the first of
  its tables was built; fixed before in `CHECK.md`, section of 20:20, sha256 of the
  file beginning `ac3716c319f04e6e`). Three more tables, 0.22 core-h on one core:
  the midpoints of the line nodes with seeds 20261126 and 20261127 (`sky_mid/`,
  sha256 beginning `b36863d4bceeaf0d`, `13bd1189cf0ff32a`) and the site with seed
  20261128 (`sky_seed3/`, `28d30df71977bbec`). With the tables of C and D: three
  realisations of the site table and three of the midpoint table. No reference, no
  solve.
  - F2, tangent midpoints at 4.27 to 10.73 GV: interpolated 412.8 counts against
    426.3 direct, −3.2% ± 1.4% (error from the scatter of the three, as fixed). The
    reading fixed before asks for beyond −4% and beyond two errors: not met. The
    node set stays; no further table waits on it.
  - What that reading does not say. The value is two errors from zero and has the
    sign of D. Over all rigidities the tangent midpoints give −0.42% ± 0.17%, at 19
    to 60 GV −0.26% ± 0.13% (errors from the states, below). A small error of the
    linear value in the tangent altitude is likely, about 3% of the count at the
    foot at the midpoints. The −7.0% of D was that and chance. Ground midpoints:
    +0.8% ± 1.3% at the foot, 0.00% ± 0.10% over all rigidities.
  - F1 as written cannot decide: the scatter of three totals has two degrees of
    freedom. Against the jackknife it gives 1.7 and 1.8 for the ground nodes at the
    foot in the two sets, 1.1 and 1.0 for the tangent nodes at the foot, 0.1 and
    0.5 for the ground nodes at 10.73 to 19.07 GV.
  - Two additions, written after `step_f.out` was read (not in `CHECK.md`):
    `step_f_add.py`, the same test on the proton boundary sums (1.3 to 1.7;
    `step_f_add.out`, `5cfa6242ed586926`), and `step_f_add2.py`
    (`step_f_add2.out`, `28b71a1ccf0e2e5e`). The second takes the variance of a
    count from the states. The draws of a state are independent of all others, so
    the variance is the sum of p (1 − p), estimated for each state from its three
    realisations: thousands of states in place of two degrees of freedom.
  - Jackknife over the sectors against the value from the states. Ground nodes:
    0.89 to 1.08 for the counts in the three ranges, 0.91 to 0.99 for the proton
    boundary sums. Tangent nodes: 0.75 to 0.90. Scatter of the three totals against
    the states: mean square 1.17 over the twelve sets, 1 ± 0.29 expected; the 1.5
    and 1.9 of the ground nodes at the foot are the largest of them.
  - Reading. The jackknife is right for the ground nodes to about 10%. The errors
    of B, C, D and E for the ground nodes stand; nothing is requoted. For the
    tangent nodes it is 10% to 25% too small: the tangent values of C, D and E are
    read with errors larger by that. Step C, tangent nodes at the foot: +3.7% with
    ± 2.7% from the states, 1.3 errors.
  - F2 with the errors from the states (`step_f_add3.out`, `e7bab89eb4fac938`), at
    the foot, at 10.73 to 19.07 GV, above, and over all. Tangent: −3.2% ± 1.6%,
    −0.03% ± 0.37%, −0.26% ± 0.13%, −0.42% ± 0.17%. Ground: +0.8% ± 1.3%,
    −0.21% ± 0.21%, 0.00% ± 0.05%, 0.00% ± 0.10%.
  - Consequences (manager). The check of the tables themselves is closed: they are
    right at their locations and nodes, the ground line nodes interpolate within
    the errors, the tangent nodes within the bound fixed before. What is left of
    the check is the spacing in latitude (the 81 tables, the cap comparison), which
    goes to the PI. The effect of the line nodes on the flux belongs to G2.5(a),
    the doubling of the μ grid. The solver agent is asked how that doubling treats
    the tangent altitudes; a refinement, if one is needed, is written as a dated
    entry before that run.
  - The site solve at M = 12 is released (manager, after step F). It was held on
    2026-11-23 until the check of the sky. It reads the site table alone, with
    weight 1 (`c2/solver/b4cost/B1.log`), so it does not depend on the spacing of
    the location grid. Its cost goes to the PI (G2.6) with the spacing, before any
    solve of G2.3.
- **2026-11-23. G2.0 evaluated again on the 2.5° grid: no trigger. The cap averages
  of the two grids: only the spacing in latitude counts** (manager;
  `c2/manager/g20/g20_2p5.out`, sha256 beginning `19a594b5feed018c`, script
  `g20_grid.py` `08a353691db69185`, values `g20_2p5_states.npz` `7c156cfe48280211`;
  `cap_compare.out` `df8f6fddc42d8907`, script `cap_compare.py` `fd158d79afb8c3f1`).
  No tracing, no solve, no reference; one core, minutes.
  - The tables. The solver agent built the 56 further tables of the 2.5° grid
    between 20:15 and 20:40 (4.98 core-h; `c2/solver/sky/SKIES_2p5.md`), with the
    seed and settings of the 25. All 81 load with seed 20261123 and h_ref = 0. At
    the five locations of the manager's check tables (`c2/manager/g20/sky/`) the
    arrays `allowed`, `R` and `az` are equal to the agent's.
  - G2.0 stays on record as TRIGGER: the gate was evaluated on the skies of the 5°
    grid. What follows is the evaluation fixed in the bullet of the check of the
    sky, with the 64 916 stored states and verdicts of the gate. Control: the stored sky
    value recomputed on the 5° grid for 1755 states, largest difference 0.
  - On the 2.5° grid no class triggers (5°: one). Largest |difference| over
    tolerance 0.93, in the class that triggered: R 5 to 10 GV, cos Z at injection
    0.25 to 0.5, S, n = 659, +0.0194 against 0.0208 (5°: +0.0288).
  - Prediction fixed before (the excess at 5 to 10 GV falls to half or less): met.
    Mean of (sky − verdict): +0.0108 ± 0.0017 at 5°, +0.0032 ± 0.0016 at 2.5°,
    ratio 0.29 (square law: 0.25). At 4 to 10 GV: +0.0088 ± 0.0013 and
    +0.0027 ± 0.0012.
  - By the place between the 5° locations in latitude, 4 to 10 GV, near a node,
    between, half-way: +0.0014, +0.0069, +0.0117 at 5°; +0.0025, +0.0031, +0.0026
    at 2.5° (errors 0.0016 to 0.0028). The dependence on the place is gone.
  - Not predicted. At 10 to 20 GV the 2.5° grid gives −0.0047 ± 0.0022 (5°:
    −0.0020 ± 0.0025). One class carries it: W, cos Z 0.5 to 1, n = 1218,
    −0.0149 ± 0.0068, and −0.0154 on the 5° grid, so it is not the spacing. It is
    2.2 errors in one class of about a hundred, and one of twelve rigidity ranges
    printed. Noted, not followed; the cap comparison below shows no difference
    between the grids in that range (−0.17% ± 0.18%).
  - Cap averages (`cap_compare.out`). The boundary of a cap is linear in the
    `allowed` arrays, so the cap-averaged array of a test grid is compared with
    that of the 2.5° grid; ground line nodes; the draws are the same in all
    tables; errors by the jackknife over the 36 sectors. The cap weights were
    checked against `LocationGrid.cap_weights` for both grids. Test over 2.5°
    grid minus 1: allowed count at 4.27 to 10.73 GV, proton boundary by number,
    by energy.

    | cap | test grid, east × north | foot count | by number | by energy |
    |---|---|---|---|---|
    | 10° | 5° × 5° | +3.86% ± 0.51% | +1.65% ± 0.12% | +0.51% ± 0.07% |
    | 10° | 5° × 2.5° | −0.06% ± 0.08% | −0.03% ± 0.03% | −0.03% ± 0.01% |
    | 10° | 2.5° × 5° | +3.95% ± 0.51% | +1.69% ± 0.11% | +0.55% ± 0.07% |
    | 5° | 5° × 5° | +8.4% ± 1.5% | +2.50% ± 0.29% | +1.01% ± 0.17% |
    | 5° | 5° × 2.5° | −0.26% ± 0.23% | −0.09% ± 0.06% | −0.05% ± 0.03% |
    | 5° | 2.5° × 5° | +8.6% ± 1.5% | +2.59% ± 0.30% | +1.07% ± 0.17% |
    | 2.5° | 5° × 5° | +7.4% ± 1.4% | +2.09% ± 0.25% | +0.83% ± 0.15% |
    | 2.5° | 5° × 2.5° | −0.15% ± 0.22% | −0.05% ± 0.06% | −0.04% ± 0.03% |
    | 2.5° | 2.5° × 5° | +7.5% ± 1.4% | +2.15% ± 0.25% | +0.87% ± 0.14% |

  - Cap of 10°, 5° grid against 2.5° grid, further numbers. Neutron boundary:
    +1.47% ± 0.11% by number, +0.40% ± 0.07% by energy. Proton transmission per
    energy bin: +4% to +9% at 2.5 to 6.3 GeV (transmission 0.02 to 0.17), +1.3% at
    6.3 to 7.9 GeV, +0.6% at 7.9 to 10 GeV, within 0.25% above. Tangent line
    nodes at the foot: +1.2% ± 0.8%. First harmonic over the sectors of the
    proton boundary by number, sine part: −0.5130 against −0.5180, difference
    +0.0049 (cap of 5°: +0.0167; cap of 2.5°: +0.0132); with 5° × 2.5° the
    differences are −0.0004 to −0.0008. No error was computed for the harmonics.
  - Reading. The spacing in longitude does nothing at this level: 5° in the east
    direction reproduces the 2.5° grid to 0.03% (cap of 10°) and 0.09% (cap of 5°).
    The whole difference comes from the spacing in latitude. For the cap of 10° it
    is 1.65% by number, below the 2.4% that the estimate of step E gave (three
    quarters of 3.2%); the measurement replaces the estimate.
  - Error of the 2.5° grid itself, by the square law of step E (a third of the
    difference; not measured): cap of 10°, 0.55% by number and 0.17% by energy;
    cap of 5°, 0.83% and 0.34%. With 1.25° in latitude (a twelfth): 0.14% and
    0.04%; 0.21% and 0.08%. These are numbers of the boundary, not of the flux
    cells of G2.4.
  - Consequences (manager). G2.4(a) as written, 5° against 2.5°, is expected to
    fail; it is run as written and reported as it comes. The 2.5° grid is not shown
    to be under 0.5%. The spacing for the cap solves goes to the PI with the cost
    (G2.6). The options: the 2.5° grid as built; or 1.25° in latitude with 5° in
    longitude (85 tables, 40 of them new, about 3.9 core-h, and a change of
    `LocationGrid` that the solver agent puts at half a day), with a convergence
    test of its own written here before its run. No table finer than 2.5° is
    built and the grid code is not changed before that decision.
- **2026-11-23. Milestone B after the trigger: cost split from short runs; the site
  solve at M = 12 started** (solver agent, hand-back at 20:43;
  `c2/solver/WP_SOLVER_REPORT_B.md`, sections after "Continuation"). Code
  checkpoint `d7ede51` (`assemble3d.grid_deficit`, `run_solve3d.py --cap/--spacing`,
  a test on 81 mock tables); no solve with the full sky in it.
  - Tests: `test_solver3d.py` 15 passed (762.83 s); `test_assemble3d.py` 7 passed
    (288.85 s).
  - Cost runs at M = 12 with 1, 2 and 4 tangent-linear solves in a batch, one
    process on 4 cores, site table, frozen C1 geometry and kernel set, guard of
    G2.7 (`c2/solver/b4cost/`): 23.1, 31.9 and 52.1 core-min; 8.6, 11.6 and
    17.2 GB. Sweep residual 1.5e-8 or below.
  - Split: 13.3 core-min fixed per process (the primal march and the compilation),
    9.7 core-min per tangent, 5.7 GB plus 2.9 GB per tangent. The design had 0.33
    to 0.41 of a C1 solve per tangent; measured 0.78. Four cores per process are
    about 1.5 times as efficient as eight. The primal is recomputed by each batch.
  - Projected, not measured: the site solve at M = 12 (25 tangents) about 5 core-h,
    at M = 6 about 2.5; the same for each cap boundary, whatever the number of
    tables. All solves of C2 without G2.8: about 34 core-h at M = 6 and 68 at
    M = 12, against 10 to 20 in the plan. G2.4(b) alone (six caps at 15 km) is 15
    to 31. This is the first input to G2.6 and goes to the PI with the measured
    site solve.
  - Released to the agent at 20:46 (manager): the site solve at M = 12 in batches
    of 5 on 4 cores (stop above 8 core-h or 24 GB per process); G2.5(a) for the
    azimuthal orders (M = 6 against 12 from the partial sums) and G2.5(c), with a
    script written before the fluxes are read; the report with G2.6. Not
    released: the doubled μ grid and L, G2.5(b), G2.8, any cap solve, G2.3, G2.4.
  - Asked, for the report: how the doubling of the μ grid treats the tangent
    altitudes of the sky table; whether tests that compare two boundaries about
    1% apart can solve the difference alone at lower cost with a known error.
- **2026-11-23. The two references of G2.3 compared: the roll removes the artefact
  from `n`; in `s` it moves a_c by −0.002 below 0.4 GeV and nothing else is seen**
  (manager; `c2/manager/dipole/two_refs.py`, output `two_refs.out`, run at 22:17
  after commit `80db4c7`; the fixed list of the entry "Reference of G2.3 replaced";
  cap of 10°; reported, not a gate). First reference `c2/mcref/c2ref_cutoff.npz`, new
  reference `c2/mcref_roll/c2roll_cutoff.npz`, twenty runs each. No solver output was
  read. This bullet holds content of the references: the solver agent is told not to
  read it before the freeze.
  - Item 1, `n`, pooled over the three lowest bands. All rows: sine −0.0360 ± 0.0020
    in the first reference and −0.0021 ± 0.0022 in the new one (1.0 errors from zero).
    cos Z 0 to 0.3: −0.0550 ± 0.0039 and −0.0024 ± 0.0042 (0.6). **The criterion is met
    in both sets of rows: the roll removes the artefact.** New minus first:
    +0.0339 ± 0.0030 and +0.0526 ± 0.0057. The cosine component is within 1.1 errors
    of zero in both references. Per band in the new reference, all rows, from 0.1 GeV
    up: sine −0.0031 ± 0.0029, +0.0025 ± 0.0043, −0.0055 ± 0.0048, +0.0060 ± 0.0056,
    +0.0023 ± 0.0033.
  - Item 2, `s`, pooled over the three lowest bands, new minus first. a_c:
    −0.00213 ± 0.00041 for all rows (5.2 errors) and −0.00250 ± 0.00090 for cos Z 0 to
    0.3 (2.8). a_s: −0.00030 ± 0.00063 and +0.00072 ± 0.00146. The shift of a_c has the
    sign and the size of the estimate (0.002 to 0.005) and of the one-run measure
    (−0.0019 and −0.0020). It is a fifth to a quarter of the floor of part (b).
  - Item 3, per band and row, new minus first. a_c: largest 0.0143 ± 0.0067 (band 4,
    row 0); rms of the differences over their errors 1.27; 4 above 0.01, 2.3 expected.
    a_s: largest 0.0212 ± 0.0074 (band 8, row 1); rms 0.85; 4 above 0.01, 6.0 expected.
    Either component above 0.01: 7 pairs of 100. The excess of a_c is in the three
    lowest bands: the sum of the squared differences over their errors is 87.0 for 30
    pairs there (mean −1.25) and 73.1 for 70 pairs above (`two_refs_tails.out`). That
    is the shift of item 2 seen pair by pair.
  - Item 4, K per cell, 4800 cells counted in both. Rms of the differences over their
    errors 1.036; beyond 2: 250; beyond 3: 24. Gaussian errors give 218 and 13.0;
    errors from two sets of twenty runs (Student t, 38 degrees of freedom) give rms
    1.027, 253 and 22.8. Cells that differ by more than 3% of K: 1107 (1073 expected).
    Mean relative difference per band for cos Z 0 to 0.3, from 0.1 GeV up:
    −0.07 ± 0.70%, +1.02 ± 1.26%, −0.25 ± 1.05%, −1.64 ± 1.44%, +2.28 ± 1.23%,
    +0.42 ± 1.54%, +0.76 ± 1.47%, +1.51 ± 1.01%, +1.93 ± 2.10%, +0.05 ± 0.44%. No band
    is beyond 2 errors. The estimate "up to about 1% in the cutoff factor of a cell"
    below 0.4 GeV is neither seen nor excluded: the errors there are 0.7 to 1.3%.
  - Item 5, differences over their errors per band. K: +0.5, +0.4, −0.4, −1.7, +1.2,
    +0.4, +0.4, +2.1, +1.0, −0.9 (sum of squares 11.2 for 10). H: −0.2, +0.1, −0.5,
    −1.1, +2.2, +0.8, −0.1, +1.7, +1.1, −1.1 (12.5 for 10).
  - Outside the fixed list: flux level, new over first, cap of 10° (`roll_levels.out`).
    `n` 1.0034 ± 0.0042 and `s` 1.0047 ± 0.0048 over all bands; per band nothing beyond
    1.9 errors (sums of squares 10.1 and 12.9 for 10).
  - Reading. The artefact was a dipole of 3.6% (5.5% near the horizon) in a tally the
    gate uses only through its azimuth mean. In the tally the gate compares cell by
    cell it is a shift of a_c by 0.002 below 0.4 GeV. No change of a_s, K, H or the
    level is seen at the precision of two references of this size. G2.3 is evaluated
    against the new reference only, as decided.
  - **Two observations outside the fixed list, on the references alone.** They were
    found by looking through 100 pairs of band and row in several tallies after the
    data were seen. Their significances are overstated by that choice.
    1. Sine component of `n` near the vertical in the new reference
       (`roll_n_rows.out`, `roll_n_rows_up.out`). Pooled over the bands: row 9 (cos Z
       0.9 to 1) −0.0065 ± 0.0016, row 8 −0.0059 ± 0.0024; rows 0 to 7 within 1.4
       errors of zero. It sits above 0.25 GeV (rows 8 and 9, bands 2 to 9: −0.0096
       and −0.0106, 3.0 and 5.2 errors). At row 9 the sign is the same in all eight
       bands from 0.25 GeV up; the one significant band is 0.25 to 0.4 GeV
       (−0.0099 ± 0.0024, negative in 17 of 20 runs). The bands share their showers,
       so the common sign is weaker evidence than eight independent values. Nothing
       in the two lowest bands, where the artefact was largest. The first reference
       shows nothing there: row 9, bands 2 to 9, −0.0024 ± 0.0033 against
       −0.0115 ± 0.0025 in the new one with the same weights (`roll_s_rows.out`); new
       minus first is 2.2 errors.
       - Read of the code of the run (`c2/mcref_roll/code/tools/mc3d/pool.py`,
         `decays.py`): the roll angle is uniform and drawn independently of the
         event; the decays are isotropic about the parent. With the roll, no cutoff
         and no bending, the set-up is mirror symmetric about the meridian of the
         site, so a sine component of `n` has no source that we know.
       - Test that bears on the gate (`roll_s_rows.out`): `s`, filled from twelve
         turned copies of each shower, does not carry it. a_s of `s`, new minus first,
         bands 2 to 9: −0.0005 ± 0.0009 at row 8 and −0.0002 ± 0.0007 at row 9. No row
         from 5 to 9 is beyond 1.2 errors in either group of bands.
       - Standing: a fluctuation (of the order of 1 to 2% after the choice of the row,
         by a rough count) or unexplained. It does not reach a quantity of G2.3: `n`
         enters only through its azimuth mean. A second set of runs would settle it
         (about 37 core-h). Not proposed.
    2. North-south component of the up-going tally without cutoff, `un`, at its edge
       row 6 (cos Z −0.4 to −0.3), in both references (`roll_un_caps.out`). Cosine
       component, three lowest bands, caps of 2.5, 5, 7.5 and 10°: new
       −0.0077 ± 0.0169, −0.0015 ± 0.0079, +0.0193 ± 0.0063, +0.0273 ± 0.0040; first
       −0.0125 ± 0.0239, −0.0049 ± 0.0104, +0.0085 ± 0.0058, +0.0207 ± 0.0037. It
       grows with the radius of the cap, compatible with its square. Row 7 at 10°:
       +0.0050 ± 0.0020 (new), −0.0008 ± 0.0017 (first).
       - Rows 8 and 9, the rows part (h) uses, cap of 10°: cosine −0.0041 ± 0.0043
         (new) and −0.0057 ± 0.0055 (first); sine −0.0612 ± 0.0048 in the first
         reference (the artefact) and −0.0043 ± 0.0044 in the new one.
       - Reading, plausible and not established: the injection patch ends 40° from the
         site, and a neutrino of |cos Z| 0.35 entered the Earth 41° away. Row 6 sits
         on the edge of the patch, and a ground point off the site then loses a part
         of its sky that depends on the azimuth. It is not an effect of the roll.
       - Rows 6 and 7 of the up-going tallies are used by no part of G2.3. They are
         not to be used for a statement on the azimuth. Rows 8 and 9 are inside the
         patch.
  - The known limitation for the product table is written in
    `tools/mceq3d/INTEGRATION.md`, section 18.7, with items 2 to 4 as an estimate.
  - Compute: single-core minutes.
- **2026-11-23. Site solve at M = 12: first measurement of G2.6; G2.5(a) on the
  azimuthal orders FAIL at M = 6; G2.5(c) PASS; the orders above about 8 of one sky
  table are draw noise** (solver agent, hand-back at 22:50;
  `c2/solver/WP_SOLVER_REPORT_C.md`, sha256 `cba66113…`; the manager's checks in
  `c2/manager/g25/`). No repository file was changed by the agent.
  - Solve: the location (0, 0) with its own sky table (seed 20261123, sha256
    `0a382659…`), mode path, the 25 tangents of the orders 0 to 12, five processes
    of five tangents one after the other on cores 0 to 3, frozen C1 geometry and
    kernel set, guard of G2.7, code state `d7ede51`. Output `c2/solver/site_M12/`
    (`tangents_b1.npz` to `tangents_b5.npz`, logs, `batches.out`). Command of a
    batch:

        taskset -c 0-3 /usr/bin/time -v $PY tools/cascade3d/gates/run_solve3d.py \
            run $N3/c2/solver/site_M12 --xy 0,0 --M 12 --orders <five orders> \
            --batch 5 --tag b<i>

  - **G2.6, first measurement** (`/usr/bin/time -v`, user plus system time):

    | batch | orders | core-min | wall | peak memory |
    |---|---|---|---|---|
    | b1 | 0c 1c 1s 2c 2s | 60.4 | 24:33 | 19 861 112 kB |
    | b2 | 3c 3s 4c 4s 5c | 55.8 | 22:56 | 19 888 492 kB |
    | b3 | 5s 6c 6s 7c 7s | 54.9 | 22:36 | 19 969 476 kB |
    | b4 | 8c 8s 9c 9s 10c | 56.2 | 22:59 | 19 895 004 kB |
    | b5 | 10s 11c 11s 12c 12s | 65.0 | 26:54 | 19 995 640 kB |

    Total 292.3 core-min = **4.87 core-h**, peak **20.0 GB**, wall 2 h 00 (20:47 to
    22:47). Batch b5 ran while other projects loaded the node (load average 32 to
    48); b2 to b4 are the quiet numbers, about 56 core-min for five tangents. The
    split of the short runs holds: it gives 61.8 core-min and 20.2 GB for a batch
    of five. G1.5 asked for 2 core-min and 2 GB per location.
  - **G2.5(a), azimuthal orders: FAIL.** M = 6 (partial sum) against M = 12 on the
    4800 cells at the site: largest relative move **5.067e-3** (limit 0.005), at
    nue, 1.58 to 2.51 GeV, cos Z 0.5 to 0.6, azimuth 30° to 60°. It is the only
    cell above 0.5%; 700 cells are above 0.1%. Worst per band, 0.1 to 10 GeV:
    2.7e-4, 7.3e-4, 1.7e-3, 3.0e-3, 4.1e-3, 4.8e-3, 5.1e-3, 4.0e-3, 2.4e-3, 1.2e-3.
    Reported: the order 0 alone against M = 12 moves a cell by up to 0.911.
    Result file `c2/solver/gates/G2.5/G2.5_orders.md` (sha256 `a457a7ab…`; `.json`
    `185f1970…`; cells `G2.5_orders_cells.npz` `dd97120f…`). Script `g2_5.py` of
    commit `5f586a8` (sha256 `49cc3381…`, checked before the run), run once, exit
    status 1:

        $PY tools/cascade3d/gates/g2_5.py orders --solve-dir $N3/c2/solver/site_M12 \
            --tags b1,b2,b3,b4,b5 --outdir $N3/c2/solver/gates/G2.5 \
            --g27 $N3/c2/solver/gates/G2.7/cells_s0.1_i0.1_l0.1_mx_D2.npz

  - **G2.5(c): PASS.** Largest same-bin sweep residual 1.49e-8 (energy bin 22;
    primal 1.49e-8, tangents 1.28e-8; limit 1e-4 in every bin).
  - Reported check of the path: the order-0 tangent gives the linearised cells of
    deficit 2 of G2.7 to 4.4e-16, and the C1 cells exactly.
  - The μ grid and L doublings of (a) and part (b) are not run. G2.5 as a whole is
    failed by its part (a) on the orders at M = 6.
  - Checks of the manager, no solve and nothing of the reference
    (`c2/manager/g25/orders_series.py`, `boundary_orders.py` and their `.out`):
    1. The comparison again from `G2.5_orders_cells.npz`: the same cell, 5.0668e-3.
       Cells above 0.5%, 0.4%, 0.3%, 0.1%: 1, 12, 53, 700. Root mean square of the
       move over the cells 8.0e-4. Largest per species: nue 5.07e-3, antinue
       3.73e-3, numu 4.19e-3, antinumu 4.35e-3.
    2. Every partial sum 0..M against 0..12, from the stored tangents:

       | M | largest move | cells > 0.5% | cells > 0.1% | largest step from M − 1 |
       |---|---|---|---|---|
       | 4 | 1.41e-2 | 200 | 1791 | 1.14e-2 |
       | 5 | 7.21e-3 | 18 | 1271 | 7.4e-3 |
       | 6 | 5.07e-3 | 1 | 700 | 4.6e-3 |
       | 7 | 3.37e-3 | 0 | 151 | 3.9e-3 |
       | 8 | 1.37e-3 | 0 | 19 | 2.1e-3 |
       | 9 | 8.8e-4 | 0 | 0 | 8.9e-4 |
       | 10 | 2.8e-4 | 0 | 0 | 6.5e-4 |
       | 11 | 0 | 0 | 0 | 2.8e-4 |

       The agent's reported numbers for M = 8 and M = 10 (1.37e-3, 2.8e-4) are
       confirmed.
    3. The order 12 does not reach the cells: the mean of cos 12φ and of sin 12φ
       over a bin of 30° is zero. "M = 12" on these cells is M = 11. The bins also
       weight the orders: largest bin mean 0.51 at order 7, 0.41 at 8, 0.21 at 9,
       0.19 at 10, 0.09 at 11, 0 at 12, then 0.07, 0.14, 0.13, 0.21, 0.21 at the
       orders 13 to 17. The small steps at the orders 10 and 11 come from these
       weights, not from the series.
    4. Amplitude of an order in a cell before the bin average, over the azimuth
       mean of the cell, largest cell: 4.6e-1, 8.6e-2, 4.4e-2, 1.0e-2, 9.1e-3,
       6.6e-3, 5.2e-3, 4.1e-3, 2.1e-3, 2.7e-3, 2.8e-3, 3.0e-3 at the orders 1 to 12
       (99th percentile from order 9: 1.8e-3, 2.2e-3, 2.5e-3, 2.5e-3). The series
       stops falling at order 9.
    5. The sky tables hold 36 sectors of 10°, so their largest order is 18
       (`fourier_sectors` refuses more). A test of M = 12 against M = 24 cannot be
       made with them; the agent's projection row for it would need a table with
       48 sectors or more. M = 8 against 16 can be made.
    6. Signal against draw noise in the boundary, from three realisations of the
       site table (seeds 20261123, 20261124, 20261128; the last two were built for
       the sky check, `c2/manager/g20/sky_seed2/`, `sky_seed3/`). A table holds one
       draw per line node, sector and rigidity rung (45 × 36 × 31). Quantity: the
       forbidden share of the proton boundary summed by number over the 18 energy
       bins the cutoff touches, order m over the sectors, root mean square over
       the 30 ground line nodes; signal from the products of two seeds, noise from
       the rest:

       | order | signal | noise |
       |---|---|---|
       | 0 | 0.886 | 0.0025 |
       | 1 | 0.0774 | 0.0048 |
       | 2 | 0.0150 | 0.0051 |
       | 3 | 0.0125 | 0.0050 |
       | 4 | 0.0059 | 0.0048 |
       | 5 | 0.0034 | 0.0046 |
       | 6 | 0.0025 | 0.0052 |
       | 7 | 0.0022 | 0.0056 |
       | 8 | 0.0027 | 0.0050 |
       | 9 | 0.0012 | 0.0052 |

       From order 10 to 18 the signal is zero within its scatter (−0.0020 to
       +0.0022) and the noise is 0.0039 to 0.0078. The noise is flat in the
       order, apart from the factor that undoes the sector average (1.5 at order
       17), as for independent draws per sector. It exceeds the signal from
       order 5 on. Per energy bin the signal falls more slowly (0.0115 at order 7,
       0.0060 at order 12, noise 0.017 to 0.019) and is below the noise from
       order 6 on. Tangent nodes: signal above the noise to order 6, about equal
       to it from 7 to 10.
    7. Reading. The move of 0.507% between M = 6 and M = 12 is mostly the draw
       noise that the orders 7 to 11 of one table carry, not azimuthal structure
       of the cutoff. With the signal share of the power at the orders 7 and 8
       (13 to 31%), the truncation of the true series at M = 6 is about 0.1 to
       0.25% in the worst cell, and below 0.1% at M = 8. This is an estimate with
       an error of its own size. It changes no verdict: G2.5(a) on the orders is
       failed at M = 6 as written.
    8. What follows for a test of M against 2M at one location with one table:
       the orders M + 1 to 2M add noise of 0.2 to 0.3% of a cell (largest cells)
       times the bin weights. Expected for M = 8 against 16: 0.16 to 0.30%, from
       the measured 0.137% to order 12 and the orders 13 to 16. The agent's
       "factor 3.6" of margin for M = 8 is against M = 12 only.
    9. The same noise is in every order the solves use, the order 0 included
       (0.0025 of the boundary by number at the site, on a transmitted share of
       0.114). Its size in a cap average is not measured: the tables of a grid
       hold the same draws, so it does not fall as for independent tables. More
       draws per table lower it at about 0.1 core-h per table and realisation;
       the solve cost does not change (the boundary is linear in the tables).
  - Consequence by the dated entry on G2.5(a): "If M = 6 fails it, M = 12 for all
    solves takes C2 to or beyond the plan's compute and goes to the PI with
    G2.6." The order for the solves of G2.3 and G2.4 goes to the PI with this
    entry. No cap solve, no solve of G2.3 or G2.4 and no new table before the
    decision.
  - Projected, not measured (agent's table with the measured batch; core-h at
    M = 6 and at M = 12): one boundary 2.6 and 4.87 (measured); G2.3, caps of 10°
    and 5°, 5.2 and 9.8; G2.4(a) 2.6 and 4.9 plus tables; G2.4(b), six shifted
    caps at 15 km, 15.4 and 29.3 plus 49 tables (about 4.7); the μ grid doubled
    6.7 and 12.1 (1.9 times the CPU and 1.75 times the memory per tangent,
    batches of 3); L doubled about 3 and 5 to 6; G2.5(b) none if assembled from
    the orders (a phase), else one solve; G2.8 at M = 18 about 6.6 for the full
    line family. All solves of C2 still to run at M = 12: about 78 core-h.
    At M = 8 (manager, derived from the split: 17 tangents, about 3.5 core-h per
    boundary): about 57 core-h. One process of five tangents fills 20 GB, so the
    solves run one after the other under the memory cap.
  - Global grid of C5, projected: 2.6 (M = 6), 3.5 (M = 8), 4.9 (M = 12) core-h
    per location on CPU; 1450 locations are about 3800, 5100 and 7100 core-h.
  - Answers of the agent to the manager's questions (report, section "Answers"):
    - The doubled μ grid reads the same 45 line nodes of the sky table by
      interpolation. It does not test the spacing of the line nodes. Step F of
      the sky check and the ground midpoints are the only tests of it. A doubled
      set of tangent altitudes costs about 1.5 core-min per location (about 2
      core-h for 81) and a few lines of code.
    - The map from the boundary to the flux is linear, so a test of two
      boundaries about 1% apart can solve their difference alone, at a lower
      order with its own truncation bound: G2.4(a) at 2.6 core-h, G2.4(b) at
      about 15, or 5 to 8 with two or three singular vectors and a bound that
      needs a positivity test. Such a reading has to be written here before its
      run. Not decided.
    - Batches of 6 or 7 tangents save at most 4% and peak near 26 GB. Not run.
  - Blindness, one disclosure. In check 1 the manager printed the solver's
    azimuth mean of the cutoff factor per band for one row at the site. The
    level of the reference per band was known to the manager from the threshold
    report. No comparison was made or written, nothing of it went to the solver
    agent, and the recommendations to the PI were formed before it, on cost and
    convergence alone.
  - Compute: milestone B about 9.7 core-h of 12 released, the site solve
    included; the manager's checks single-core minutes. C2 so far about 98 to 99
    core-h of the plan's 100.

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
- **2026-11-23, after the manager's report on the count thresholds of G2.3.** The PI was
  given three findings, the manager's intended rule and five remaining steps.
  - Findings: the threshold rule of the first text fails an exactly right solver in 68
    to 95% of synthetic references at the expected event counts; the half-against-half
    rule gives about 11% there in the most realistic model and 24% in a less realistic
    one; and the threshold from a null model detects a uniform error of 1% of the
    cutoff factor, where the plan's "95% of the cells" detects about 2%.
  - Rule: the half-against-half threshold decides. The calibration at the reference's
    own event counts is reported next to it. A failure by fewer cells than the measured
    bias is still a FAIL and is brought to the PI with that number.
  - Steps: the dated amendment; the calibration wired into `null_count_c2.py` and
    `g2_3.py`, with tests and one validation on a synthetic reference; commit and push;
    the entries before the first solve (G2.7, G2.8, the draw of G2.0, the variations of
    G2.4); a new solver agent.

  The PI answered "Proceed as proposed". Recorded as approval of the rule and of the
  five steps. The report also said that the amendment could no longer be committed
  before the reference was complete, and why. Not decided by this answer: M = 12 for all
  solves, the fallback of G2.7, the far side, anything of C3.
- **2026-11-23, after the manager's report on the azimuth artefact of the Monte Carlo.**
  The PI was given the finding and its test, the estimated effect on G2.3, and three
  options: (A) keep the reference with the artefact on record; (B1) nineteen more runs
  with `--event-roll` at the same size, 39 to 50 core-h; (B2) twenty runs at the plan's
  first size of 2×10^5 primaries, 17 to 21 core-h. The manager recommended B1 and said
  that it takes C2 beyond the plan's compute: about 112 to 135 core-h at the design's
  solve costs, and near 100 to 135 even without it if the measured cost of the
  linearised solves holds (three to four times the design's estimate; G2.6).

  The PI answered "GO with B1 as recommended". Recorded as: the reference of G2.3 is
  replaced as in the dated entry "Reference of G2.3 replaced", and its compute is
  authorised. Not decided by this answer: the cost of the solves (G2.6 goes to the
  PI before the solves of G2.3, as before), M = 12 for all solves, the fallback of
  G2.7, anything of C3, anything about the product table.
- **2026-11-24, after the manager's report on the site solve (G2.6) and on G2.5(a).**
  The PI was given:
  - the measured cost: 4.87 core-h and 20 GB for one location at M = 12, about four
    times the design's estimate; about 2.6 core-h at M = 6 and 3.5 at M = 8 (derived);
  - G2.5(a) on the orders FAILED at M = 6 (0.507% against 0.5%, one cell of 4800),
    with the manager's reading (mostly draw noise of one sky table; true truncation
    estimated at 0.1 to 0.25%) and the fact that M = 12 cannot be tested by doubling
    with tables of 36 sectors;
  - the compute used, about 98 to 99 of the plan's 100 core-h;
  - the outlook for C5 (1450 locations, about 5000 core-h on CPU at M = 8), the table
    noise, and the disclosure on blindness of the Results-log entry of 2026-11-23.

  Three recommendations were put:
  1. M = 8 for all solves, tested as the gate reads against M = 16 at the site (1.7
     core-h; expected move 0.16 to 0.30%); if it fails, back to the PI.
  2. Location grid of 5° in longitude by 1.25° in latitude (40 new tables, about 3.9
     core-h). The pair of G2.4(a) replaced, before its run, by 2.5° against 1.25° in
     latitude; otherwise the pair as written would be run and its foreseen failure
     recorded.
  3. Two stages. Stage 1: the new tables, the test of the orders, the two cap solves,
     the grid test, the cheap checks, then G2.3 by an independent agent; 17 to 18
     core-h, C2 at about 116. Stage 2: G2.4(b) and the doublings of the μ grid and of
     L; 30 to 38 core-h; decided with the result of G2.3. The catch was said: if a test
     of stage 2 later forces a change, G2.3 is re-run in full.

  The PI answered "Follow your recommendations". Recorded as approval of the three,
  with the pair of G2.4(a) replaced. The dated entry of 2026-11-24 fixes the details.
  Not decided by this answer: stage 2, an order above 8, more draws per table, the
  fallback of G2.7, anything of C3, anything about the product table.
