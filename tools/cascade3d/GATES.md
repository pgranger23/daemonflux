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
