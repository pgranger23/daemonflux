# Milestone 3a: the geomagnetic cutoff at each primary's injection point

Factual record (2026-10-08).  Every number comes from a study in `cutoff.py`
(sec. 10); raw outputs are in `/home/pigrange/mc3d_work/runs/m3a/`.

## 0. Summary

* **What.**  `cutoff.py` applies Honda's injection-point test -- the
  time-reversed trajectory must reach 10 R_E "without touching the injection
  sphere again" -- batched, to arbitrary injection states (position,
  inward velocity, nucleus rigidity, charge).  The integrator is
  `geomag_backtrace.backtrace_vec`, unchanged.  New: an exact IGRF accelerator
  (`ppigrf`'s sum from cached coefficients, equal to 2e-15, bitwise independent
  of the batch; 11-228x per field call, ~15x per trace); an explicit field
  model (`ppigrf` 2.1.0 silently defaults to IGRF-14); a step table and path
  budget fixed by convergence studies; an exact pre-filter; the azimuthal
  copies; an optional `R_U` cache; the 1D reference's site map.  19 tests.
* **Re-entry rule: Honda's, measured.**  Forbidden on re-entering the
  injection sphere -- the only rule under which each physical primary is
  sampled once (at its first inward crossing).  `geomag_backtrace`'s sea-level
  floor accepts 0.8-0.9% more of the allowed primaries (all below 20 GV;
  grazing back-traces bottoming out at 20-100 km).
* **Validation.**  (a) With Phase-1 settings the machinery reproduces the
  cached Phase-1 Kamioka map exactly (504/504 cells).  At the injection point
  above Kamioka the vertical cutoff is 11.16 GV (11.10 on a 0.05 GV ladder,
  island [9.20, 9.75]) against 11.4-11.5 on the ground: the Stoermer
  `(R_E/R_inj)^2 = 0.969` launch-altitude factor.  Honda's floor then raises
  near-horizon cutoffs (+0.45 GV at 87 deg), so the injection map sits 3.7%
  below `cutoff_map` on average.  East >> West (E/W 2.8/4.7/5.6 at 60/80/87 deg).
  (b) Pure dipole: Stoermer to 1e-4 at the equator, the `r^-2` launch scaling
  exact, `R_U >= R_St` everywhere, no allowed state below Stoermer above 0.59 GV.
  Below that the bound region is larger than the 10 R_E escape sphere: a limit
  of the test, outside the MC's 1.7 GV domain.  (c) The charge-mirror symmetry
  holds in 3000/3000 states; for negative primaries the East-West pattern
  reverses (E/W 0.36/0.22/0.15 at Kamioka).
* **Cost.**  Forbidden, quasi-trapped back-traces are the expensive ones, and
  a call waits for its slowest trajectory: **9.3 ms per copy state at 1e4
  states per call, 54 ms at 1e3, 149 ms at 100** (gated, machine fully
  loaded).  Splitting one batch over workers does not scale (1.44x at 12).
* **Strategy.**  Trace lazily, after the cascade, every azimuthal copy of the
  showers that scored (18% of cap-pointing primaries have copies with
  different verdicts, 48-75% at 5-20 GV); gate; pool >= 1e4 states per call.
  `CPU ~ N_prim x f x N_az x 9.3 ms`: **+45% / +90% of the cascade (N_az 12 /
  24) if all scored showers are traced (f = 3.6%), +5% / +10% for East-West-
  region showers only (f = 0.4%)**; per-shard calls of a few hundred states
  cost several times more.  The `R_U` cache (option iii) is error-free at
  5/10/25% margins on 14 400 states and saves a further 1.6x; it pays above
  ~1.2e7 traced states.
* **For the PM** (sec. 9): the primary is transported straight from the
  injection point, but it bends by 6-27 deg (median, 5-15 GV) before its
  first interaction -- Honda bends it; near the site the lowest stratum's
  protons are 100% forbidden (re-derive the allocation); GSF has no species.

---

## 1. What was implemented (`cutoff.py`, `test_cutoff.py`)

| function | what it does |
|---|---|
| `trace_allowed(r_cm, u, R, charge, ...)` | binary cutoff verdict for N injection states, serial; the integrator is `geomag_backtrace.backtrace_vec`, unchanged |
| `gate`, `trace_gated` | the exact pre-filter (Stoermer floor below, 60 GV above), then `trace_allowed` on the rest -- **the call to use in production** |
| `admittance(..., n_jobs)` | `trace_allowed` over forked workers (one chunk per worker); `n_jobs=1` inside an existing pool worker |
| `FastIGRF`, `igrf_field(model, fast)` | cached-coefficient IGRF evaluator equal to `ppigrf.igrf_gc` to <= 2e-15 and bitwise independent of the batch size, swapped in by a context manager for the duration of a trace; selects IGRF-13 (default) or IGRF-14 explicitly |
| `rigidity_gv(e_kin_nucleon, A, Z)` | nucleus rigidity `R = (A/Z) p_nucleon` (superposition) |
| `step_m(R)`, `STEP_TABLE`, `MAX_S_M` | RK4 step (25/100/200 km below 5 / 5-10 / above 10 GV) and path budget (2e8 m) |
| `copy_states`, `copy_shifts_deg`, `rotate_about`, `admittance_copies` | the azimuthal copies: rotations about the site vertical |
| `upper_cutoff(r, u, ...)` | `R_U` per state with `geomag_backtrace.scan_upper_cutoff`, every trace through `trace_allowed` (optionally gated) |
| `SiteCutoffMap` | detector-anchored `R_c(zenith, azimuth)` map at the site: the 1D reference's cutoff |
| `RUCache` | the cached-`R_U` gate (acceleration option iii) |
| `stoermer_gv`, `stoermer_floor_gv` | Stoermer cutoff of a state / of the most favourable direction at a position |
| `python cutoff.py {steps,budget,highr,floor,escape,kamioka,dipole,cost,cache,cachebatch,bend}` | the studies behind every number below (sec. 10) |

## 2. Conventions and design decisions

### 2.1 The state and the reversal

A primary is `(r, u, R, q)`: position on the injection sphere `|r| = R_E + 100 km`
[cm, mc3d units], its **inward** physical velocity `u` (as sampled by
`geometry.sample_injection_patch`), the nucleus rigidity `R` [GV] and the
charge sign `q` (+1 for every nucleus).  It is tested as
`backtrace_vec(r, -u, R, ..., charge=q)`: the launch velocity is the reversed
velocity and the antiparticle reversal happens inside `backtrace_vec` through
`k = -charge c/R` -- the Phase-1 convention (`cutoff_map` launches
`-arrival_direction` with `charge=+1`), unchanged.  Azimuths are compass
bearings of the *source*, clockwise from North (`geometry`/`geomag_backtrace`).

### 2.2 The re-entry rule: forbidden on touching the injection sphere

Honda, astro-ph/0404457 sec. IV (p. 8), verbatim: *"For a sampled cosmic ray,
the 'history' is examined by solving the equation of motion in the negative
time direction.  When the cosmic ray reaches the escape sphere without
touching the injection sphere again in the inverse direction of time, the
cosmic ray can pass through the magnetic barrier"*, with `R_inj = R_e + 100 km`
and `R_esc = 10 R_e`.

`cutoff.py` adopts exactly that (`floor="injection"`, the default): the
back-trace is absorbed at the launch sphere itself.  Three reasons, in order of
weight:

1. **It is the only rule consistent with sampling the injection sphere.**  The
   MC samples every inward crossing of the 100 km sphere as an independent
   primary with an isotropic (Lambert) weight.  A physical cosmic ray whose
   past trajectory dipped below 100 km crossed that sphere inward *before* --
   at a point and direction the MC also samples.  Counting the later crossing
   as well counts the same particle twice.  Absorbing at `R_inj` keeps exactly
   one representative per particle: its *first* inward crossing.
2. **Below 100 km is atmosphere.**  The MC's own world starts at `R_inj`
   (`X_v(100 km) = 0.00128 g/cm2`; everything below is simulated matter).  A
   trajectory that comes back through it has traversed (part of) the
   atmosphere: deep dips (tens of g/cm2 of slant column) would have interacted;
   shallow ones are the double counting of point 1.
3. It is Honda's convention, and the deliverable is a ratio compared with his.

`geomag_backtrace`'s default floor is sea level (`r < R_E`): appropriate for
a *detector-anchored* cutoff map launched from the ground, too permissive for
an injection-point test.  "Top of the atmosphere" cannot be the CORSIKA
`H_TOP_KM = 112.8 km` here: it lies above the launch sphere.  The measured
difference between the floors is in sec. 4.

### 2.3 Escape radius, path budget

`r_escape = 10 R_E` (Honda; `R_E = 6371 km` here vs Honda's 6378.18 km, 0.11%).
`geomag_backtrace.cutoff_map`'s implicit default is 15 R_E (`backtrace_vec`'s
default).  A trace still bound after `max_s` of path is forbidden (trapped).
Both are measured in sec. 4.

### 2.4 The field: explicit model, cached coefficients

`geomag_backtrace._bfield_igrf_cart` calls `ppigrf.igrf_gc` without a
coefficient file.  **`ppigrf` 2.1.0 (the one in the venv) defaults to
`IGRF14.shc`**, so every Phase-1 map was in fact built with IGRF-14 (DGRF-2020
at 2020-01-01) below `r_switch = 4 R_E` and the hard-coded IGRF-13 dipole
above -- not "IGRF-13" as the docstrings say.  The two models differ by at most
6.9 nT (2.2e-4 of |B|) at 2020-01-01; the cutoff consequences are in sec. 5.1.
`cutoff.py` makes the model explicit (`model="IGRF13"` default, `"IGRF14"`
available).

`ppigrf.igrf_gc` re-reads and re-parses the coefficient file with pandas on
every call (14.6 ms) and then costs ~30 us per point; it was 97-98% of a
back-trace.  `FastIGRF` evaluates the same sum from coefficients parsed and
time-interpolated once (ppigrf's own pandas interpolation, run once per date),
equal to ppigrf to <= 2e-15 relative for both models at both tested dates.  It
has three internal evaluators -- scalar Python for <= 4 points (the bouncing
tails, where numpy's per-call overhead dominates: 0.15 vs 0.97 ms at one
point), packed arrays up to 1024 points, a per-degree loop above -- made
**bitwise identical** (same Legendre recursion, same `cos/sin(m phi)`, same
`a^(n+2)` product sequence, same sequential summation order; `np.add.reduceat`
is *not* sequential and was replaced; the scalar path relies on numpy's
float64 `sin`/`cos` being libm's, true on this AVX2 build and pinned by a
test), so a point's field, hence a trajectory's verdict, never depends on how
many other trajectories share its batch.  It is
installed by `igrf_field(model, fast=True)`, a context manager that swaps
`ppigrf.igrf_gc` for the duration of a trace and restores it; `fast=False`
runs the stock ppigrf pinned to the chosen model (the reference path).

### 2.5 Step size

`backtrace_vec` integrates with a fixed arc-length step (`ds = 100 km` in
Phase 1).  The gyroradius near the surface is `rho ~ 55.6 km x R[GV]` at 60 uT,
so 100 km exceeds it for a 1.7 GV proton -- and the 40 deg injection patch
reaches geomagnetic latitudes (up to ~65 deg) where such primaries are
allowed.  `step_m` uses the table **25 km below 5 GV, 100 km at 5-10 GV,
200 km above** (`STEP_TABLE`).  Measured (sec. 3), it reproduces fine-step
references in 5999 of 6000 states; 200 km above 10 GV halves the cost of the
high-rigidity traces relative to Phase 1's 100 km.  (The validation studies of
secs. 4-5 ran with a provisional rule that was finer at 5-20 GV -- 50 km at
5-10, 100 km at 10-20 -- before the step study showed the coarser classes to
give identical verdicts; nothing in them depends on the difference.)

### 2.6 Rigidity of nuclei

`R = (A/|Z|) sqrt(T (T + 2 m_p))` with `T` the kinetic energy per nucleon
(`sky._shard` starts the cascade at `e_tot = e_nuc + m`, so its `e_nuc` is
kinetic).  All `A` nucleons of a nucleus share one injection state and one
verdict.

---

## 3. Step-size convergence (`python cutoff.py steps`)

Two test sets of 3000 states each, sampled the way `sky.py` samples primaries
(position uniform per area on the 40 deg patch around Kamioka, inward Lambert
directions, energy per nucleon stratified in ln E over 1-1e4 GeV with
`n_k ~ E_k^-0.8`, p/He samples 0.8/0.2, `R = (A/Z) p`): **A** the whole patch
(median R 4 GV), **B** only states whose straight primary ray reaches the ground
inside the 10 deg cap (the geometric proxy for showers that score; positions
within 15 deg of the site, median 7 deg).  Verdicts at fixed steps 400 ... 6.25
km, injection-sphere floor, 4e8 m path budget; the number of verdicts that
differ from the finest step, per rigidity bin:

| set | R [GV] | n | 400 km | 200 km | 100 km | 50 km | 25 km | 12.5 km |
|---|---|---|---|---|---|---|---|---|
| A | < 3 | 969 | 7 | 7 | 4 | 5 | 0 | 0 |
| A | 3-5 | 784 | 8 | 1 | 1 | 2 | 0 | 0 |
| A | 5-10 | 592 | 5 | 0 | 0 | 0 | 0 | 1 |
| A | >= 10 | 655 | 0 | 0 | 0 | 0 | 0 | 0 |
| B | < 3 | 1011 | 0 | 0 | 0 | 0 | 0 | ref |
| B | 3-5 | 750 | 1 | 0 | 1 | 1 | 1 | ref |
| B | 5-10 | 618 | 3 | 2 | 0 | 0 | 0 | ref |
| B | >= 10 | 621 | 0 | 0 | 0 | 0 | 0 | ref |

(A against 6.25 km, B against 12.5 km.)  The low-rigidity flips of set A all
sit 18-40 deg from the site, in the northern patch where 2-4 GV primaries are
allowed; the one persistent set-B flip (4.8 GV, 10.7 deg, `mu = 0.30`) changes
at every step >= 25 km and is a chaotic penumbral trajectory.  The allowed
fraction moves by <= 0.0017 (A) and <= 0.0010 (B) over the whole range of
steps.  Every state was traced at every step, so any rule can be scored
exactly: the adopted table gives 0/3000 (A) and 1/3000 (B, the chaotic state)
differences, the same as a finer provisional rule (50 km at 5-10, 100 km at
10-20 GV) with 8-9% fewer steps over the sampled mix and half the steps at
5-20 GV, where the traced cost lives.  Coarsening further (50 km at 2.5-5 GV)
adds 3 flips in A and was rejected.

## 4. Re-entry rule, escape radius and path budget

### 4.1 Path budget (`python cutoff.py budget`)

A trace that has neither escaped nor fallen below the floor after `max_s` of
path is declared forbidden (trapped).  The minimum path to escape radially from
the injection sphere to 10 R_E is 5.7e7 m, so 5e7 m allows nothing.  Same two
test sets (seeds 61/62), provisional step rule (sec. 2.5), injection floor:

| `max_s` [m] | 5e7 | 1e8 | 2e8 | 4e8 | 8e8 |
|---|---|---|---|---|---|
| A allowed | 0 | 0.3417 | 0.3603 | 0.3660 | 0.3687 |
| A flips vs 8e8 (R < 5 / 5-10 / >= 10 GV) | -- | 52/26/3 | 22/3/0 | 8/0/0 | ref |
| B allowed | 0 | 0.1907 | 0.2040 | 0.2043 | 0.2043 |
| B flips vs 8e8 (R < 5 / 5-10 / >= 10 GV) | -- | 0/35/6 | 0/1/0 | 0/0/0 | ref |

Above 10 GV nothing changes beyond 2e8 m.  Every flip is a late escape
(forbidden with the shorter budget, allowed at 8e8 m): quasi-trapped
trajectories.  The slow convergence of set A is entirely below 10 GV and far
from the site (the 25 flips at 2e8 m sit at a median 29 deg, minimum 6 deg):
low-rigidity primaries of the northern patch, which hardly ever make a
neutrino in the cap.  For the cap-pointing set the verdicts are converged at
4e8 m and differ by one state (5.8 GV, 9 deg) at 2e8 m (3/3000 on the
floor-study set, `0.2130` vs `0.2140` at 8e8 m).  **Default `MAX_S_M = 2e8 m`
(31 R_E of path)**: the budget also sets the length of the tail a call waits
for (sec. 6), and 2e8 m halves it relative to 4e8 m for a <= 0.1% change in
the relevant verdicts.

### 4.2 High-rigidity gate (`python cutoff.py highr`)

3000 cap-pointing states with `R` log-uniform in 20-2000 GV: 12 forbidden with
the injection floor (11 with the sea-level floor), the highest at 26.6 GV;
**none above 60 GV**.  Scored showers above 60 GV can therefore be accepted
without a trace (`R_SKIP = 60 GV`, exact for this geometry; the maximum
`R_U` over the cap-pointing domain in sec. 5 is the cross-check).  Near-
tangent states (injection zenith > ~80 deg) are still shadowed at higher `R`,
but they do not reach the dense atmosphere and never score.

### 4.3 The re-entry rule, measured (`python cutoff.py floor`)

Test sets of seeds 21/22, provisional step rule, 2e8 m budget, floors nested
from sea level to the injection sphere.  Every trace allowed with a higher floor is
allowed with a lower one (same integration), so the counts below also locate
the lowest point of the trajectories that the rules disagree on:

| | sea level | 20 km | 50 km | 80 km | injection (100 km) |
|---|---|---|---|---|---|
| A allowed | 0.3577 | 0.3577 | 0.3567 | 0.3553 | **0.3543** |
| B allowed | 0.2147 | 0.2143 | 0.2140 | 0.2137 | **0.2130** |

* `geomag_backtrace`'s sea-level floor accepts 10/3000 (A) and 5/3000 (B)
  states that Honda's rule rejects: **0.9% (A) and 0.8% (B) of the allowed
  primaries, all below 20 GV** (near the cutoff, where back-traces graze the
  atmosphere).
* Their lowest altitude: A -- 20-50 km: 3, 50-80 km: 4, 80-100 km: 3;
  B -- below 20 km: 1, 20-50: 1, 50-80: 1, 80-100: 2.  The deep ones crossed
  tens to hundreds of g/cm2 of air on a grazing chord and could not have
  survived; the shallow ones are the double counting of sec. 2.2.  Either way
  the sea-level rule over-counts.
* Escape radius: within the 2e8 m budget, 15 R_E instead of 10 flips 10/3000
  (A) and 6/3000 (B), all allowed -> forbidden (25 R_E: 64 and 48): the extra
  path to the larger sphere runs out of budget.  With a budget long enough for
  both (8e8 m, `python cutoff.py escape`) **10 vs 15 R_E gives 0/3000 flips**
  on set B (allowed 0.2140 both).  At 10 R_E the field is ~30 nT (gyroradius
  ~1e6 km at 10 GV), so a trajectory there is free; 10 R_E (Honda) is kept.

Per-stratum allowed fractions with the injection floor (the input for
re-deriving the stratum allocation, item 4 of sec. 9):

| E_kin/nucleon [GeV] | 1-3.2 | 3.2-10 | 10-32 | 32-100 | > 100 |
|---|---|---|---|---|---|
| p, whole patch (A) | 0.166 | 0.344 | 0.807 | 1.000 | 1.000 |
| He, whole patch (A) | 0.312 | 0.604 | 1.000 | 1.000 | 1.000 |
| p, cap-pointing (B) | **0.000** (0/1448) | 0.158 | 0.864 | 1.000 | 1.000 |
| He, cap-pointing (B) | 0.049 | 0.634 | 0.968 | 1.000 | 1.000 |

The lowest stratum is 60% of the samples under `n_k ~ E_k^-0.8`; near the
site its protons are all forbidden.

## 5. Validation

### 5.1 Injection points above Kamioka on the cutoff-map grid (`python cutoff.py kamioka`)

`R_U` (the highest forbidden rigidity, `scan_upper_cutoff`: 1 GV ladder +
bisection to 0.1 GV) on `mceq3d`'s delivered cutoff-map grid (21 zeniths with
the limb nodes x 24 azimuths), for six variants that change one thing at a time:

| variant | launch | floor | field, settings |
|---|---|---|---|
| `ph1` | ground | sea level | Phase 1 exactly: IGRF-14 (ppigrf default), 15 R_E, 100 km steps, 1.6e9 m, 36.43 N |
| `V0_14` | ground | sea level | IGRF-14, this module's defaults (10 R_E, step table, 2e8 m), 36.4267 N |
| `V0` | ground | sea level | IGRF-13, defaults |
| `V1` | 100 km | sea level | IGRF-13, defaults |
| **`V2`** | **100 km** | **injection** | **IGRF-13, defaults: the injection-point test** |
| `V2neg` | 100 km | injection | as V2, charge -1 |

| comparison (504 cells) | mean | rms | max abs | cells > 0.2 GV |
|---|---|---|---|---|
| `ph1` vs the cached Phase-1 map (`prodfam` centre) | 0.000 | 0.000 | 0.000 GV | 0 |
| IGRF-13 - IGRF-14 (`V0 - V0_14`) | -0.003 | 0.108 | 1.75 GV | 7 |
| launch 100 km - ground (`V1 - V0`) | -0.872 (-4.9%) | 1.40 | 7.75 GV | 467 |
| injection floor - sea level (`V2 - V1`) | +0.296 (+1.3%) | 0.82 | 7.25 GV | 108 |
| **injection-point test - `cutoff_map` (`V2 - V0`)** | **-0.577 (-3.7%)** | 0.81 | 2.75 GV | 464 |

| variant | vertical | 60 deg N/E/S/W | 80 deg N/E/S/W | 87 deg N/E/S/W | E/W at 60/80/87 |
|---|---|---|---|---|---|
| `ph1` | 11.37 | 10.3/23.6/10.0/8.4 | 21.3/35.2/9.8/7.5 | 26.8/41.7/9.8/7.5 | 2.81/4.72/5.59 |
| `V0` | 11.53 | 10.2/23.6/9.8/8.3 | 21.3/35.2/9.8/7.5 | 26.8/41.7/9.8/7.5 | 2.83/4.72/5.59 |
| `V1` | 11.16 | 9.8/22.9/9.7/8.1 | 19.7/34.2/9.5/7.2 | 25.0/40.2/9.5/7.2 | 2.83/4.77/5.57 |
| **`V2`** | **11.16** | 9.8/22.9/9.7/8.2 | 20.5/34.2/9.5/7.2 | 25.8/40.5/9.5/7.2 | **2.81/4.73/5.61** |
| `V2neg` | 10.78 | 9.2/7.8/9.5/21.8 | 15.0/6.8/9.4/31.2 | 18.8/6.8/9.4/44.8 | 0.36/0.22/0.15 |

* **The machinery reproduces Phase 1 exactly.**  With Phase 1's settings the
  map equals the cached Phase-1 map in every one of the 504 cells (vertical
  11.37, 87 deg East 41.72, 87 deg West 7.47 GV -- the published values).  That
  pins `FastIGRF`, the launch-state reversal, the charge handling, the units
  and the scan wrapper all at once.
* **Vertical 11.16 GV at the injection point, ~11.4-11.5 at the ground; East
  >> West near the horizon** (E/W = 2.8 at 60 deg, 4.7 at 80, 5.6 at 87).  The
  ground value moves between 11.37 and 11.53 GV with the ladder's rungs: the
  vertical has a penumbra (Phase-1 fine ladder: `R_U = 11.45`, allowed islands
  down to 9.55 GV), so a 1 GV ladder + bisection lands within the band's fine
  structure, +-0.1 GV.
* **The difference to `cutoff_map` is the launch altitude, then the floor.**
  Launching 100 km higher lowers every cutoff by the Stoermer factor
  `(R_E/R_inj)^2 = 0.969` (vertical 11.53 -> 11.16) and by more near the
  horizon, where a given local zenith at 100 km and at the ground are
  different trajectories.  Honda's floor then raises the near-horizon cutoffs
  (zero at the vertical; mean +0.12 GV at 60 deg, +0.45 at 87, +1.6 at 89.5,
  up to +7 GV in single cells): grazing back-traces that dip into the
  atmosphere are now forbidden, a larger Earth shadow.  Net: the injection-point
  map is 3.7% below `cutoff_map` on average.
* **Model.**  IGRF-13 vs IGRF-14 changes the map by 0.003 GV on average; the 7
  cells that move by more than 0.2 GV are penumbral (chaotic) cells, e.g. 87
  deg West 7.47 vs 7.22 GV.
* **Charge.**  For negative primaries the East-West pattern reverses (E/W
  0.36/0.22/0.15), and the vertical moves to 10.78 GV (the non-dipole field
  breaks the dipole's charge symmetry of the vertical).
* The largest cutoff on the map at the injection sphere is 50.5 GV (89.5 deg,
  az 75); none saturates the 70 GV ladder.


### 5.2 Pure centred dipole: Stoermer (`python cutoff.py dipole`)

`r_switch = 0`, `m_hat = z`: a centred aligned dipole of `geomag_backtrace`'s
`B0 = 3.07e-5 T`, whose Stoermer constant is `C = B0 R_E c = 58.64 GV` (vertical
equatorial cutoff 14.66 GV on the ground).

| check | result |
|---|---|
| vertical `R_U` at the magnetic equator, ground launch / sea-level floor | 14.660 GV vs Stoermer 14.659 (1.0001) |
| same, injection-sphere launch and floor | 14.207 vs 14.210 (0.9998) |
| launch-altitude ratio `R_U(100 km)/R_U(0)` at 0/10/20 deg | 0.9691/0.9693/0.9692 vs `(R_E/R_inj)^2 = 0.9693` |
| vertical `R_U/R_St` at 10/20/30/40/50 deg (injection) | 1.033/1.131/1.256/1.088/1.145 -- `R_U >= R_St` everywhere (penumbra above the Stoermer edge) |
| lat 30, zenith 60, E/W, q = +1 | `R_U` 20.66/7.22 GV (Stoermer 11.58/6.32) |
| lat 30, zenith 80, E/W, q = +1 | 30.91/6.22 GV (Stoermer 12.48/6.15) |
| same, q = -1 | 7.22/20.66 and 6.22/30.91: **exactly mirrored** |
| mirror symmetry `(q, state) <-> (-q, y -> -y)`, 3000 random states, 3-80 GV | **0 disagreements** (allowed fraction 0.689) |
| Stoermer theorem: 3000 random states (`|lat| <= 70`, both floors) at `R/R_St` in [0.30, 0.97] | 9 allowed, **all at 0.17-0.25 GV**; 0 of the 2379 states above 1.7 GV |

The 9 "violations" are not integration error: they stay allowed with 5 km and
1 km steps (gyroradius 9-14 km).  They are a limit of the escape test itself.
A bound (Stoermer inner-region) orbit extends to the Stoermer length
`R_E sqrt(C/R)` = 15-18 R_E at 0.2 GV, beyond the 10 R_E escape sphere, so
reaching 10 R_E does not prove escape there.  The test is valid for
`R > C / r_escape^2 = 0.59 GV` (Stoermer length < 10 R_E).  The MC's
primaries start at 1.7 GV (Stoermer length 5.9 R_E), where 0 of 2379 states
violate the theorem.  Confirmed directly: the 9 states have Stoermer lengths
of 15.3-18.8 R_E, and re-traced with the escape sphere at 25 R_E (8e8 m budget)
or 40 R_E (1.6e9 m), all 9 are forbidden.

Above the Stoermer edge the shadow cone raises `R_U` far above Stoermer near
the horizon (30.9 vs 12.5 GV from the East at 80 deg).  Stoermer is the
*lower* edge of the penumbra, exact as such, and `R_U` the upper edge, which
only a trace gives.

### 5.3 Charge sign

Positive primaries are cut harder from the East: dipole above (East/West 20.7/7.2
at 60 deg and 30.9/6.2 GV at 80 deg), Kamioka in sec. 5.1.  The exact
mirror-symmetry test pins the antiparticle reversal inside `backtrace_vec`
(`k = -charge c/R`): a sign error there would swap East and West and fail
3000 of 3000 comparisons, not 0.

### 5.4 The exact pre-filter (`gate`)

Two bounds that decide a state without a trace, both measured rather than
assumed:

* **Lower: `R < 0.70 x R_St,min(position)`  ->  forbidden.**  `R_St,min` is
  the Stoermer cutoff of the most favourable direction (West horizon) at the
  injection point, in the IGRF's tilted centred dipole: 6.50 GV above Kamioka.
  In a dipole nothing below it is allowed; in IGRF allowed states were found
  down to 0.98 x it within 15 deg of the site (set B) and down to 0.77 x it
  anywhere on the patch (set A, 8e8 m budget), hence the 0.70.
* **Upper: `R > 60 GV` and `mu > 0.17`  ->  allowed.**  No cap-pointing state is
  forbidden above 26.6 GV (sec. 4.2); near-tangent states (`mu < 0.17`) can be
  shadowed at higher rigidity but never reach the dense atmosphere.

On five test sets (15 000 states; steps, budget and high-R sets, A and B):
**0 errors**.  On the cap-pointing mix it decides 53% of the states as
forbidden and 5% as allowed, leaving 41% to trace.  The rejected states
include the deepest bouncers (sec. 6: deep below the cutoff the back-trace
bounces in the inner magnetosphere until the path budget), but the measured
CPU gain is 1.5-1.7x (secs. 6, 6.1), not the ~3.7x the per-band costs suggest:
the undecided states just below their cutoff bounce too and keep the per-call
tail alive.  `trace_gated` = gate + trace of the rest.
Used before the cascade, it can skip the cascade of primaries whose every
azimuthal copy is gated forbidden: 6% of all primaries at `N_az = 12-24` (10%
of the lowest stratum).

## 6. Cost (`python cutoff.py cost`)

Machine: 2 x 12-core Xeon E5-2680 v3 (Haswell, 2.5 GHz, 48 hyperthreads),
shared with another agent's 32-process Monte Carlo: **every number is process
CPU per hyperthread under full machine load** (44 busy threads on 24 physical
cores), the condition the production run will see.

**The field call** (`ppigrf.igrf_gc` vs `FastIGRF`, ms per call):

| points per call | 1 | 4 | 100 | 1000 |
|---|---|---|---|---|
| `ppigrf.igrf_gc` | 33.1 | 18.1 | 23.2 | 50.0 |
| `FastIGRF` | 0.145 | 0.571 | 0.964 | 4.50 |
| speed-up | 228 | 32 | 24 | 11 |

**CPU per trace** (cap-pointing geometry, 2e8 m budget; this table and the
next were measured with the provisional step rule, sec. 2.5):

| case | batch | `FastIGRF` | stock `ppigrf` |
|---|---|---|---|
| allowed, 80 GV | 100 | 13.5 ms | 198 ms |
| allowed, 80 GV | 3000 | 2.8 ms | -- |
| forbidden, 4 GV (bouncing) | 100 | 447 ms | 7175 ms |
| forbidden, 4 GV (bouncing) | 3000 | 33 ms | -- |

| R band [GV] (batch 1000) | 1.7-5 | 5-10 | 10-20 | 20-40 | 40-100 | 100-1000 |
|---|---|---|---|---|---|---|
| allowed fraction | 0.000 | 0.155 | 0.712 | 0.972 | 1.000 | 1.000 |
| CPU per trace | 69.0 ms | 40.9 ms | 12.7 ms | 4.5 ms | 3.7 ms | 3.4 ms |

**The plan's premise is reversed for injection states: forbidden traces are
the dear ones.**  An allowed high-rigidity trace runs ~290 steps of 200 km to
10 R_E and costs ~3 ms; a state deep below its cutoff is quasi-trapped and
bounces until the 2e8 m budget (8000 steps of 25 km).

**Realistic mix vs batch size** (cap-pointing set, sky.py rigidity
distribution, allowed fraction ~0.2):

| states per call | 100 | 300 | 1000 | 3000 | 10000 |
|---|---|---|---|---|---|
| ungated (provisional step rule), ms CPU per state | 207 | 111 | 60 | 24 | 13.9 |
| **gated (`trace_gated`, adopted rule), ms CPU per state** | **149** | **132** | **54** | **27** | **9.3** |
| gated, ms per *traced* state (42-45% traced) | 338 | 293 | 134 | 64 | 22 |

The batch dependence is the per-step fixed overhead of the bouncing tail: a
call lives as long as its slowest member, and that tail (thousands of steps at
a handful of points, ~1 ms per step with the scalar field path) is paid once
per call and step class -- 15-90 s per call here, whatever the number of
states.  At small batches it dominates, which is also why the two rows are
within noise of each other below 3000 states per call.  The pre-filter's gain
is real but modest (1.5x at 1e4 states): the states it cannot decide -- just
below their cutoff -- are themselves bouncers and keep the tail alive.
Whole-patch mix (set A), batch 3000, ungated: 32.9 ms per state.

**Multiprocessing scaling.**  One batch of 6000 cap-pointing states (ungated)
split over `admittance(n_jobs)` workers, one chunk each:

| workers | 1 | 2 | 4 | 8 | 12 |
|---|---|---|---|---|---|
| wall | 108 s | 77 s | 102 s | 82 s | 75 s |
| total CPU | 108 s | 139 s | 363 s | 596 s | 804 s |
| speed-up / efficiency | 1 / 1 | 1.40 / 0.70 | 1.07 / 0.27 | 1.33 / 0.17 | 1.44 / 0.12 |

Splitting one batch does not scale: every chunk pays its own tail, so the
total CPU grows with the number of chunks and the wall time is floored by the
longest-lived trajectory (~1 min).  **Parallelise across independent large
batches** (shards, or pooled trace-and-replay tasks), never by splitting a
batch.  The single-process numbers in this section were taken with the whole
machine busy, so they are the per-core throughput of that design.

### 6.1 Acceleration (iii): cached `R_U` gate (`python cutoff.py cache`)

`RUCache`: `R_U` on a 4-D grid of injection states around the site --
positions in azimuthal-equidistant coordinates about the site every 2.5 deg to
+-7.5 deg (7 x 7), and at each the arrival direction in that point's own local
frame, zenith {0, 10, ..., 60, 65, 70, 74, 77, 80, 83} deg x azimuth every 15
deg -- **15 288 nodes**, each a full `scan_upper_cutoff` ladder (1 GV from
70 GV, bisection to 0.1 GV) with the exact pre-filter deciding the rungs it
can.  A state's 4-D cell has 16 corners; with `lo, hi` the min/max of their
`R_U`, the gate accepts `R > hi (1 + m_hi)`, rejects `R < lo (1 - m_lo)` (with
a wider `m_lo` = islands margin where `lo < 20 GV`, the low-cutoff vertical and
West directions where allowed islands exist below `R_U`) and traces the rest.

* **Build: 7930 s CPU (874 s on 11 workers), 0.52 s per node**; `R_U` from 4.91 to
  45.53 GV over the grid, none saturated.
* **Test: 600 cap-pointing primaries within 6 deg of the site x 24 azimuthal
  copies = 14 400 states** (exact allowed fraction 0.202), all traced exactly
  for reference:

| variant | decided allowed | decided forbidden | traced | errors | CPU (11 workers) | speed-up |
|---|---|---|---|---|---|---|
| all traced | -- | -- | 1.000 | -- | 982 s | 1 |
| exact pre-filter (sec. 5.4) | 0.042 | 0.549 | 0.409 | **0** | 796 s | 1.23 |
| cache gate, margins 5/10/25% | 0.163 | 0.696 | 0.141 | **0** | 566 s | 1.74 |
| pre-filter + cache | 0.163 | 0.696 | 0.141 | **0** | 559 s | 1.76 |

  Margin scan (fraction traced, errors of 14 400): 2/5/15% -> 0.102, **3**
  (gate-forbidden but allowed: islands); 5/10/25% -> 0.141, 0; 10/15/35% ->
  0.189, 0; 20/25/50% -> 0.298, 0.  Lookup 2.2 us per state.
* The CPU speed-up is far below the 7x reduction in traced states, for two
  measured reasons: the states left to trace are the near-cutoff, penumbral
  ones -- the longest-lived of all -- and each call still waits for its
  bouncing tail (the traced subsets shrink to 185-535 states per worker).
* **Per-state saving, one call per variant, one process** (`cachebatch`):

| variant | traced states (one call) | CPU | per state of the 14 400 | per traced state |
|---|---|---|---|---|
| all traced | 14 400 | 173.7 s | 12.1 ms | 12.1 ms |
| exact pre-filter | 5 895 | 101.3 s | **7.0 ms** (x1.7) | 17.2 ms |
| cache gate | 2 035 | 64.1 s | **4.45 ms** (x2.7) | 31.5 ms |

  The cache saves 2.6 ms per copy state beyond the pre-filter.  Its build
  cost (0.52 CPU-s per node) is amortised after `7930 s / 2.6 ms = 3e6`
  states for this +-7.5 deg grid, and after ~1.2e7 states for the +-15 deg
  grid that would cover every scored shower (~4x the nodes).

## 7. Strategy and cost of the 3D run

### 7.1 What sets the cost

* **States to trace per injected primary = `f x N_az`.**  `f` = the
  scored-shower fraction, measured by the other agent (MILESTONE2 sec. 4.1):
  **3.6%** of injected showers have a 3D cap hit or a 1D crossing in the 10 deg
  cap (1.9% / 5.8% / 6.8-7.0% in the 1-3 / 3-10 / >10 GeV-per-nucleon strata),
  **0.40%** have a 3D hit in the East-West region (`cos Z < 0.3`, 0.3-2 GeV).
  `N_az = 12-24`.
* **Every copy needs its own trace.**  Of the 600 cap-pointing primaries of the
  cache test (within 6 deg of the site), 106 (18%) have copies with different
  verdicts -- 48% of those at 5-10 GV and 75% at 10-20 GV, i.e. exactly the
  rigidities that make the sub-GeV East-West effect.  One verdict per shower
  would be wrong where it matters.
* **CPU per traced state** depends on the batch size and on how many deep
  bouncers the batch holds (sec. 6): the exact pre-filter removes most of
  them, and pooling the copies of many showers into one call amortises the
  tail.

### 7.2 Recommendation

1. **Lazy, per-copy, after the cascade** (sec. 8): trace the `N_az` copies of
   the showers that scored, nothing else -- `0.43` states per primary at
   `N_az = 12` (`f = 3.6%`) instead of 12 for eager tracing of every copy of
   every primary (28x fewer).
2. **Defaults of `cutoff.py`**: `FastIGRF` (15x per trace against stock
   ppigrf), the step table, the 2e8 m budget, Honda's floor, 10 R_E.
3. **Always `trace_gated`** (exact pre-filter, zero errors).
4. **Pool the traces**: one call per >= 1e4 states.  Per-shard calls of a few
   hundred states pay the bouncing tail per call (sec. 6 table: up to 10x
   per state).  Either make the shard large enough that `f x n_shard x N_az
   >= 1e4`, or return the pending (scored) showers from the shards and run
   trace-and-replay tasks on pooled batches (all tallies merge additively,
   so the replay can happen in any task).
5. **Trace what the run measures.**  A dedicated East-West run can keep only
   showers with a 3D hit in the East-West region (`f = 0.40%`): 9x fewer
   traces.
6. **The `R_U` cache (option iii) pays only for large trace counts.**  On top
   of the pre-filter it saves 2.6 ms per copy state (x1.6, single calls; x1.4
   in small chunks) at the price of a one-off build of 0.52 CPU-s per node
   (2.2 core-h for the +-7.5 deg test grid, ~9 core-h for the +-15 deg region
   scored showers come from), a new approximation (zero errors at 5/10/25%
   margins on 14 400 states, 3 island errors at 2/5/15%) and a grid to rebuild
   per site and epoch.  Break-even ~1.2e7 copy states: **use it if every
   scored shower is traced in a full-size run** (the 3% East-West table at
   `f = 3.6%`, `N_az = 12`: ~2.6e8 states, ~190 core-h saved), with a running
   exact-trace check on ~1% of the gate's decisions; **skip it for an
   East-West-only run** (`f = 0.4%`: ~3e7 states, ~10 core-h net).
7. **The 1D reference uses a site map, not traces** (`SiteCutoffMap`, built
   once: 504 directions, ~0.6 core-h, sec. 8.5).

### 7.3 Core-hours

`CPU_trace = N_prim x f x N_az x c(n) / 3600` core-h, with `c(n)` the CPU per
copy state through `trace_gated` at `n` states per call (sec. 6).

Per **1e7 injected primaries** (the cascade costs 24.7 core-h at the measured
8.9 ms/shower, MILESTONE2 sec. 4), from the measured gated `c(n)` above
(log-log interpolation; below 100 states a call costs its 15 s tail), for the
cap-pointing rigidity mix:

| f | N_az | states | per-shard calls (2000 primaries/shard) | pooled calls (1e4 states) |
|---|---|---|---|---|
| 1e-3 | 12 / 24 | 1.2e5 / 2.4e5 | 20.6 / 20.6 core-h (24-48 states/call) | 0.3 / 0.6 core-h |
| **4e-3 (E-W region)** | 12 / 24 | 4.8e5 / 9.6e5 | 20.6 / 36.9 core-h | **1.2 / 2.5 core-h** |
| 1e-2 | 12 / 24 | 1.2e6 / 2.4e6 | 45 / 62 core-h | 3.1 / 6.2 core-h |
| **3.6e-2 (all scored)** | 12 / 24 | 4.3e6 / 8.6e6 | 72 / 92 core-h | **11.2 / 22.4 core-h** |
| 1e-1 | 12 / 24 | 1.2e7 / 2.4e7 | 103 / 118 core-h | 31 / 62 core-h |

* **Formula** (pooled): `CPU_trace ~ N_prim x f x N_az x 9.3 ms`, i.e.
  `CPU_trace / CPU_cascade ~ f x N_az x 1.05` at today's 8.9 ms/shower --
  **+45% (N_az = 12) to +90% (N_az = 24) for all scored showers, +5-10% for
  East-West-region showers only.**  Per shard instead: add `~15-30 s x
  N_shards` of tails.  For the 3% East-West table (MILESTONE2 sec. 4.7:
  ~1.4-1.9e3 core-h, ~6e8 primaries at N_az = 12) that is ~70 core-h pooled with
  `f = 0.4%`, ~670 core-h with `f = 3.6%`, and up to 4000+ core-h per-shard.
* The scored showers' rigidity mix is harder than the sampled one (scoring
  probability rises with energy): weighting the cap-pointing states by the
  measured `f` per stratum moves the per-state cost by +21% (estimate from
  the per-band costs of sec. 6) -- within the uncertainty of this table.
* Both the cascade share and `f` change when the stratum allocation is
  re-derived (sec. 9 item 4): fewer cheap 1-3 GeV/nucleon showers raise the
  average cascade cost per shower, so the relative tracing cost falls.

## 8. Integration spec for `sky._shard` (for the agent wiring 3b)

Everything below uses only `cutoff.py`'s public functions; nothing in
`geomag_backtrace` or `tools/mceq3d` needs to change.

### 8.1 Prerequisite: a per-species primary

The cutoff needs the **nucleus** rigidity, so the primary model must carry
`(A, Z)` per sample: `--model H3A --groups p He CNO MgSi Fe` (already in
`sky.GROUPS`).  The GSF "N" group (nucleon level, `primaries.GSFNucleons`) has
no `A/Z` and cannot be used with a cutoff (MILESTONE2 sec. 5.1 item 1) unless a
GSF species table is brought in from outside `crflux`.

### 8.2 Where to call, and which state to pass

In `_shard`, right after `e_nuc, is_p, w, r0, u0 = primary_sample(...)`
(before the shower loop):

```python
_name, a_mass, z, gid = GROUPS[gname]
R = cutoff.rigidity_gv(e_nuc, a_mass, z)    # (n,) nucleus rigidity; e_nuc is kinetic/nucleon
```

The state passed to the test is **the primary's own injection state, exactly as
sampled**: `r0[i]` [cm, on R_E + 100 km], `u0[i]` (inward velocity), `R[i]`,
`charge=+1`.  Not the site line of sight, not the 1D crossing point, not the
first-interaction point.  The test is applied *after* the cascade (lazy, sec.
7): only showers that scored need it.

### 8.3 The shower loop: buffer, then trace, then replay

```python
N_AZ = 12 or 24
axis = np.array(caps.axis)                      # site vertical (Earth centre -> site)
rng_ph = np.random.default_rng(seed + 7919)     # separate stream: keeps the cascade RNG unchanged
pending = []                                    # (i, buffered neutrino entries)
for i in range(n):
    buf = ShowerBuffer(caps)                    # records every add(pdg, e, r, u, w,
    buf.begin_shower(r0[i], u0[i])              #   parent, gen) call (wants_parent)
    run_shower(rng, pdg, e_tot, r0[i], u0[i], backend, cfg, buf, 1.0)
    if buf.scored:                              # >= 1 3D cap hit or a 1D crossing in
        pending.append((i, buf.entries))        #   the cap (DualCapScorer's _f3 / _f1)
idx = np.array([p[0] for p in pending], int)
phase = rng_ph.uniform(0.0, 360.0 / N_AZ, len(idx))
rr, uu = cutoff.copy_states(r0[idx], u0[idx], axis, N_AZ, phase)       # (m, N_AZ, 3)
ok, n_traced = cutoff.trace_gated(rr.reshape(-1, 3), uu.reshape(-1, 3),
                                  np.repeat(R[idx], N_AZ), charge=1,
                                  date=DATE, model="IGRF13")   # serial (n_jobs=1)
ok = ok.reshape(-1, N_AZ)
shifts = np.radians(cutoff.copy_shifts_deg(N_AZ, phase))               # (m, N_AZ)
for (i, entries), ok_i, sh_i in zip(pending, ok, shifts):
    for j in np.where(ok_i)[0]:
        rot = lambda v: cutoff.rotate_about(v, axis, -sh_i[j])
        sc.begin_shower(rot(r0[i]), rot(u0[i]))
        for (pdg, e, r, u, parent, gen) in entries:
            u_j = rot(u)                         # 3b: + per-copy muon-bend correction
            sc.add(pdg, e, rot(r), u_j, w[i] / N_AZ, parent=parent, gen=gen)
        sc.end_shower()                          # 3D tally weight: sec. 8.5
```

* `ShowerBuffer` is a ~20-line proxy scorer (not written here: `scoring.py` is
  owned elsewhere).  It stores `(pdg, e, r, u, parent, gen)` of every neutrino
  `run_shower` hands it and sets `scored` when `caps.locate(r, u)` hits any cap
  or `caps.locate(r0, u0)` (the 1D crossing) is in the cap.  Under a pure
  rotation about the site vertical the cap maps onto itself, so a shower scores
  in every copy or in none; with 3b's per-copy muon-bend correction a neutrino
  near the cap edge can move in or out, so test `scored` against a cap ~1 deg
  larger than the largest scored cap.
* The rotation for copy `j` is `rotate_about(v, axis, -delta)` with
  `delta = phase + 360 j / N_AZ` **compass degrees, clockwise seen from above**:
  it is applied identically to the injection point, the injection direction,
  and every buffered production point and neutrino direction.  `copy_states`
  is exactly that rotation for the primary.
* `trace_gated` = the exact pre-filter (`gate`: rejects `R < 0.70 R_St,min`
  and accepts `R > 60 GV` with `mu > 0.17` without a trace; zero errors on
  15 000 test states, sec. 5.4) followed by `trace_allowed` on the rest.  Use
  `trace_allowed` directly for an ungated reference run.
* A forbidden copy is simply skipped (weight 0); the copies of one shower are
  equally weighted, `w/N_AZ`, so the sum over copies of an all-allowed shower
  is the original weight.  The estimator is unbiased because the injection
  distribution (patch about the site, isotropic directions) is invariant under
  rotations about the site vertical; only B is not, which is what the
  per-copy verdict carries.

### 8.4 Batching

The trace call is **serial** inside a pool worker (`_shard` already runs in a
pool; daemonic workers cannot fork), and its cost per state falls steeply with
the number of states per call (sec. 6: 149 ms at 100 states, 54 at 1000, 9.3 at
1e4), because a call lives as long as its slowest bouncing trajectory.  So:

* **Per shard**, as in the pseudo-code above, only if a shard holds
  `>= ~1e4` copy states (`f x n_shard x N_AZ`; e.g. `n_shard >= 2.3e4`
  primaries at `f = 3.6%`, `N_AZ = 12`).  Never per shower.
* Otherwise **pool**: `_shard` returns its pending showers (`(i, r0, u0, R, w,
  entries)`, ~2-3 kB each) instead of replaying them, the parent accumulates
  them, and submits a `trace_and_replay` task per >= 1e4 copy states to the
  same pool.  That task calls `trace_gated` once, replays the allowed copies
  into a fresh `DualCapScorer` and returns it; all tallies merge additively,
  so it does not matter which task scores a shower.
* Parallelism comes from many such tasks, never from splitting one batch
  (`admittance(n_jobs > 1)` scales to 1.4x at 12 workers, sec. 6).

The trace is deterministic and draws no random numbers, and a state's verdict
does not depend on its batch (bitwise, sec. 2.4, gated in
`test_batching_equivalence`), so pooling changes no result.

### 8.5 How the weight multiplies the tallies

* **3D tally** (`s0, s1`): `w_i / N_AZ x allowed_ij` -- the injection-point
  verdict of copy `j`.
* **1D reference** (`t0, t1`): the 1D convention is a cutoff at the *site*
  line of sight (PHASE2_PLAN sec. 5.2), i.e. a detector map.  Use
  `cmap = cutoff.SiteCutoffMap(lat, lon)` (built once, ~500 directions; ground
  launch, sea-level floor, same field model and date) and weight
  `w_i / N_AZ x cmap.allowed(R_i, zen1_ij, az1_ij)`, with `(zen1, az1)` the
  copy's 1D crossing direction that `begin_shower` already computes.  That is
  the sharp-`R_U` convention of the delivered `mceq3d` engine, so `R_s` is the
  factor to apply on top of a 1D flux times a detector-cutoff `G`.  This
  needs `DualCapScorer.add` to take separate 3D and 1D weights (or two passes),
  a small change in `scoring.py`.
* **Variance.**  All copies of a shower (and all neutrinos of one copy) are
  correlated; per-entry `w^2` sums underestimate the variance.  With the
  replay loop above the fix is cheap: accumulate the shower's contributions per
  bin in a dict, then add `sum w` and `(sum w)^2` per bin once per shower.

### 8.6 Bookkeeping to record per shard (`shard_meta`)

`n_scored`, `n_states = n_scored x N_AZ`, `n_traced` (returned by
`trace_gated`), `n_allowed`, CPU seconds of the call.  These give the measured `f` and cost per trace that turn
sec. 7's formula into a number.

## 9. Open items and caveats for the PM

1. **The primary is not bent between injection and its first interaction.**
   `shower.transport_hadron` moves every hadron, the primary included, in a
   straight line (`r + s u`).  Honda tracks it ("all the nucleons carried by the
   cosmic ray nuclei are treated as protons with double rigidity ... before
   the first interaction with an air-nucleus", astro-ph/0404457 sec. IV).  The
   bend `L c B_perp / R` over the path `L` from 100 km to `X = 85 g/cm2`,
   evaluated in IGRF along the straight path for 3000 cap-pointing states
   (`python cutoff.py bend`, all verdicts):

   | injection zenith | R [GV] | median L [km] | median bend | 90% |
   |---|---|---|---|---|
   | 0-45 deg | 5-15 | 95 | 6.2 deg | 12.3 deg |
   | 0-45 deg | 15-40 | 95 | 2.2 deg | 3.8 deg |
   | 45-70 deg | 5-15 | 143 | 12.0 deg | 21.5 deg |
   | 45-70 deg | 15-40 | 140 | 3.9 deg | 7.1 deg |
   | 70-85 deg | 5-15 | 280 | 27.1 deg | 41.9 deg |
   | 70-85 deg | 15-40 | 273 | 8.3 deg | 15.2 deg |

   Coherent (one charge sign), and as large as the production cone that
   PHASE2_PLAN sec. 4.3 compares hadron bending to (14.6 deg RMS at 0.5 GeV).
   The plan's sec. 4.3 argument covers pions and kaons over a *decay* length; the
   primary (and leading charged nucleons) travel an *interaction* length, tens to
   hundreds of km at altitude, at the lowest rigidities in the cascade.  This
   needs a decision before the East-West run: bend charged nucleons in
   `transport_hadron` (the muon stepper already integrates in the field).  The
   cutoff test is unaffected: Honda's test is at the injection sphere, whatever
   happens below it.
2. **IGRF-13 vs IGRF-14.**  `ppigrf` 2.1.0 defaults to IGRF-14, so Phase-1
   maps used IGRF-14 below 4 R_E (sec. 2.4).  `cutoff.py` defaults to IGRF-13
   as the plan asks; the cutoff difference is in sec. 5.1.  Pick one and use it
   for both the 3D test and the 1D reference map.
3. **GSF has no species.**  The cutoff needs `(A, Z)`: run `--model H3A`, or
   bring a GSF species table (sec. 8.1).
4. **Stratum allocation.**  With the cutoff in, the lowest ln E stratum
   (1-3.2 GeV/nucleon, 60% of the samples under `n_k ~ E_k^-0.8`) is mostly
   forbidden near the site (sec. 4.3 per-stratum fractions); the allocation of
   PHASE2_PLAN sec. 3.2 must be re-derived (MILESTONE2 sec. 5.1 item 5).
5. **Forbidden traces are the expensive ones, not the allowed ones.**  The
   plan's "forbidden ~30 RK4 steps, allowed ~900" (from launch points on the
   ground at Kamioka near the cutoff) does not hold for injection states: deep
   below the cutoff the back-trace is quasi-trapped and bounces until the path
   budget (sec. 6).  The exact pre-filter (sec. 5.4) removes the deepest ones;
   pooled calls amortise the tails of the rest.

## 10. Reproducing

```sh
PY=/cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc14-opt/bin/python3.12
R=<repo>; W=/home/pigrange/mc3d_work
export PYTHONPATH=$W/pylib:$R/.venv/lib/python3.12/site-packages:$R/src:$R/tools/mceq3d:$R/tools/mc3d
export OMP_NUM_THREADS=1
C="$PY $R/tools/mc3d/cutoff.py"
O=$W/runs/m3a/out                    # outputs: <study>/*.txt (tables) + *.npz
$C steps   --outdir $O/steps   --n 3000 --n-jobs 12   # sec. 3   (~45 min)
$C budget  --outdir $O/budget  --n 3000 --n-jobs 12   # sec. 4.1 (~20 min)
$C highr   --outdir $O/highr   --n 3000 --n-jobs 12   # sec. 4.2
$C floor   --outdir $O/floor   --n 3000 --n-jobs 11   # sec. 4.3 (~20 min)
$C escape  --outdir $O/escape  --n 3000 --n-jobs 12   # sec. 4.3
$C kamioka --outdir $O/kamioka --n-jobs 12            # sec. 5.1 (~17 min)
$C dipole  --outdir $O/dipole  --n 3000 --n-jobs 11   # sec. 5.2
$C cost    --outdir $O/cost    --n-jobs 1 --parts field,band,single,batch,mixA
$C cost    --outdir $O/cost    --n-jobs 12 --parts gated,scaling
$C cache   --outdir $O/cache   --n-jobs 11            # sec. 6.1 (~25 min)
$C cachebatch --outdir $O/cache                       # sec. 6.1
$C bend    --outdir $O/bend                           # sec. 9 item 1
$PY -m pytest $R/tools/mc3d/test_cutoff.py -q         # 19 tests, ~40 s
```

All studies sample their test states from fixed seeds (`sample_test_states`),
so the sets are reproducible bit for bit; timings depend on the machine load
(here: a 24-core / 48-thread node shared with a 32-process Monte Carlo).  Raw
outputs of this milestone: `/home/pigrange/mc3d_work/runs/m3a/out/` and
`logs/` (persistent local disk).

## 11. Tests (`test_cutoff.py`, 19 tests, ~40 s)

| test | what it pins |
|---|---|
| `test_fast_igrf_matches_ppigrf` (x4) | `FastIGRF` = `ppigrf.igrf_gc` to 1e-13 for IGRF-13/14 at two dates, 1500 random points |
| `test_field_value_does_not_depend_on_the_batch` | the scalar, packed and loop evaluators are bitwise equal (and numpy's sin/cos are libm's) |
| `test_igrf_field_context_selects_the_model_and_restores` | the context manager routes to the chosen model and restores ppigrf; ppigrf's own default is IGRF-14 |
| `test_rigidity_superposition` | `R = (A/Z) p_nucleon` for p, He, Fe |
| `test_step_rule` | the step table |
| `test_copy_states_are_rotations_about_the_site_vertical` | isometry (radius, distance to the site, local zenith) and the compass shift of the arrival azimuth |
| `test_reentry_rule_sets_the_absorbing_radius` | `floor` -> `r_floor` passed to `backtrace_vec` (launch sphere, sea level, altitude), `r_escape = 10` |
| `test_batching_equivalence` | one batch = reversed order = pairs = two forked workers, verdict by verdict |
| `test_charge_mirror_symmetry_in_a_dipole` | `(q, state) <-> (-q, mirrored state)` in an aligned dipole: the antiparticle reversal |
| `test_stoermer_floor_values` | the most-favourable Stoermer cutoff: equator, pole, `r^-2`, minimum over directions |
| `test_gate_rules_and_physics` | the pre-filter's rules; what it rejects is forbidden by trace |
| `test_pure_dipole_vertical_cutoff_is_stoermer_at_the_equator` | verdicts at 0.98/1.02 x Stoermer (measured `R_U` 14.207 vs 14.210 GV) |
| `test_pure_dipole_nothing_below_stoermer_is_allowed` | Stoermer's theorem for `R >= 2 GV` |
| `test_east_west_ordering_and_charge_sign_in_a_dipole` | East/West verdicts reverse with the charge |
| `test_kamioka_east_west_ordering_and_charge_sign` | full IGRF above Kamioka, 80 deg: East cut for `q > 0`, West for `q < 0` at 20 GV |
| `test_kamioka_vertical_cutoff_and_island` | Kamioka vertical: allowed above `R_U = 11.10`, forbidden in the band below, allowed in the island [9.20, 9.75] GV |
