# `tools/mc3d` milestone 2 — chromo in the driver, and the first 3D run

Factual record.  Every number comes from a run in this directory; the raw
outputs are under `scratchpad/mc3d/` (`pools/`, `closure_B*.npz`,
`sky/*.npz`, and the `*.log` files) and are reproduced by the commands in §7.

---

## 1. What was implemented

| file:function | what it does |
|---|---|
| `pool.py:EventPool` | on-disk-cached pools of pre-generated `chromo` events, keyed by (projectile, energy bin); `draw` returns one exclusive final state with full 3-momenta |
| `pool.py:EventPool.bin_choice` | stochastic choice between the two bin centres bracketing `E`, linear in `ln E` — removes the first-order multiplicity bias of a fixed bin representative |
| `pool.py:generate_pool` | one `model(n)` call per target component; the batching is the whole point (§2) |
| `pool.py:_Generators` | **process-level singleton** generator cache, initialised with the heaviest target and the top of the energy range (§2.2) |
| `pool.py:air_lambda`, `_target_fractions` | N/O air mix; `lambda = <A> m_u / sum_i f_i sigma_prod,i`; struck nucleus drawn from `f_i sigma_i` |
| `pool.py:EventPool.build_lambda_table` | tabulates `lambda` into the cache so physics workers never construct a generator |
| `prebuild_pools.py:main` | parallel, resumable cache fill (708 pools, 321 MB, 90 s on 40 cores) |
| `interactions.py:ChromoBackend` | the pool behind the milestone-1 backend interface (`has_interaction`, `lambda_int`, `interact`); `xs` selects the cross-section source |
| `closure.py` (`--mode chromo`) | rung B; `_init` builds the `ChromoBackend`, everything else is the milestone-1 code path |
| `geometry.py:sample_injection_patch`, `patch_area_cm2` | injection-patch sampler: position uniform per unit area on a cap of angular radius `theta_inj`, direction from the inward Lambert law |
| `scoring.py:NestedCaps` | concentric ground-sphere caps intersected **once** per neutrino; returns the smallest containing cap index plus the zenith/azimuth in the local frame of the crossing point |
| `scoring.py:DualCapScorer` | the 3D tally and the correlated collinear 1D reference on the same neutrinos, in both measures (`s0 = sum w`, `s1 = sum w/cos psi`), with squared sums, merging additively |
| `sky.py:run`, `_shard` | the 3D driver: one multiprocessing task per (species, `ln E` stratum, chunk); per-stratum accumulators written to disk |
| `sky.py` (`--collinear`) | the same primary sample with a collinear cascade — a genuine 1D run, used to bound the correlated-1D residual |
| `primaries.py:GSFNucleons` | GSF at the nucleon level (crflux's GSF has no per-nucleus splines); also fixes the m^2 -> cm^2 unit conversion |
| `analyse_sky.py` | the milestone-2b tables, with exact post-hoc bin merging |
| `test_pool.py`, `test_sky.py` | the new gates (§6) |

---

## 2. Milestone 2a — chromo in the driver

### 2.1 The pool, and why it exists

Measured on one core, SIBYLL-2.3d p+air at 100 GeV: **73 events/s** calling
`model(1)` per interaction, **1845 events/s** calling `model(3000)` once — a
factor 25, all of it per-call setup.  `pool.EventPool` therefore pre-generates
2000 events per (projectile, energy bin) and draws from the pool; the draw
costs **13 000 draws/s** on one core, i.e. it is no longer visible against the
transport.  Pools live in a disk cache, so every worker of every run shares
them and generation is paid once (`prebuild_pools.py`: **708 pools, 321 MB,
90 s on 40 cores**).

Bin representative and rescaling: 20 bins/decade above the 80 GeV splice,
10/decade below (DPMJET is 16x slower, see §2.3).  A drawn event is put on the
requested energy by keeping `p_T` and `x_L = p_z/p_beam` and scaling `p_z` by
`p_beam(E)/p_beam(E0)` — Feynman scaling, and `p_T` is deliberately untouched
because it is the whole source of the 3D opening angle.  The *multiplicity* is
frozen at the bin representative, which would leave a ~1% first-order bias
(`d<n>/dln E ~ 1.5` against `<n> ~ 18`); `EventPool.bin_choice` removes it by
choosing between the two bin centres bracketing `E` with a probability linear
in `ln E`, so the mean `ln E` of the drawn representative equals the requested
`ln E` exactly (gated in `test_pool.py::test_bin_choice_is_linear_in_log_e`).

Reuse: a pool is drawn from with replacement.  That biases nothing the scorer
tallies (every tally is linear in the event) but caps the effective statistics
of one (projectile, energy-bin) channel at 2000 events; the pilot below drew
each pool ~10^2-10^3 times, and a production run should raise `--n-pool`.

### 2.2 Two hard constraints of the fortran libraries

Both were found by the prebuild failing, and both are now handled in
`pool._Generators`, which is a **process-level singleton**:

* `chromo` asserts on a second construction of the same generator in one
  process ("all symbols are by default in global scope").  A new `EventPool`
  in a worker must find the already-built model.
* DPMJET-III-19.3 sizes its arrays at initialisation and then refuses anything
  heavier: `Maximal initialization mass exceeded 1/1, 16/14`.  It must be
  initialised with the **heaviest** target it will ever see (oxygen, not
  nitrogen) at the top of the energy range.

### 2.3 The air target and the cross sections

Atomic number fractions of dry air (78.09% N2, 20.95% O2, 0.93% Ar by
molecule → 0.7848 / 0.2105 / 0.0047 by atom).  SIBYLL-2.3d does not accept an
Ar target, so argon's 0.47% is folded into oxygen: **`f_N = 0.7848`,
`f_O = 0.2152`, `<A> = 14.430`** against real air's 14.542 (0.8%, and it enters
only through `lambda`).  The struck nucleus is drawn from **`f_i sigma_i`**,
not from the number fractions: `chromo`'s own `CompositeTarget` uses
`rng.multinomial(n, fractions)`, which over-weights nitrogen by ~1% because
`sigma_prod(O)/sigma_prod(N) = 300.4/265.7 = 1.13` at 100 GeV.

Cross-section source: **chromo's own** `cross_section().prod` per component
(the production cross section; quasi-elastic events make no new secondaries),
combined as `lambda = <A> m_u / sum_i f_i sigma_i`.  Two accessor defects were
measured and are worked around:

* **DPMJET-III-19.3 returns a constant.**  `cross_section().prod` is
  **293.158 mb for every projectile (p, pi+, K+, K_L) at every energy (20 and
  79 GeV)** — the value frozen at initialisation.  Unusable.  Below the splice
  the interaction length is therefore taken from MCEq's tabulated
  SIBYLL-2.3d/DPMJET-III-19.3 cross sections (the same database, correctly
  differentiated).
* **SIBYLL-2.3d returns NaN for a Lambda projectile**; the number fractions and
  the MCEq `lambda` are used instead.  Lambda's decay length in air is ~10^-3
  of its interaction length, so it never matters.

Where SIBYLL does answer, its `lambda` agrees with MCEq's to **1.9%**
(87.76 vs 89.44 g/cm2, p+air at 100 GeV); the tabulated SIBYLL values are
ordered as they must be (p 87.7 < pi 115.8 < K 132.9 g/cm2 at 85 GeV).
`ChromoBackend(xs=...)` selects `"mceq"` (the default, and what rung B uses so
that the closure is a pure *yield* comparison), `"hybrid"` or `"chromo"`.

**Projectiles re-interacted:** p, pbar, n, nbar, pi+-, K+-, K_L, K_S, Lambda,
Lambdabar — every hadron the cascade tracks (`pool.POOL_PROJECTILES`), each
with its own pool.  Muons and neutrinos decay only.  K_S and Lambda are offered
to the generator even though they decay essentially always, so the competition
is resolved by the transport rather than by an assumption.  Below **2 GeV total
energy** a projectile is declared non-interacting: a nucleon there has at most
0.23 GeV of kinetic energy and cannot make a pion, and a pion or kaon that low
decays long before it interacts (a 1 GeV pion's decay length is 0.1 g/cm2 at
30 km and 7 g/cm2 at sea level against `lambda ~ 110`).  Species kept from a
final state: the 12 hadrons above plus muons; pi0, gammas, electrons, Sigma+-
(0.08 pions/interaction of feed-down, ~1.3% of the pions) and charm are
dropped, exactly as in milestone 1 and as in MCEq's own tables.

### 2.4 The air mix fixes the milestone-1 yield discrepancies

Inclusive yields per interaction, chromo pool (N/O mix) against the MCEq
SIBYLL-2.3D tables at the same projectile energy, 4000 draws:

| E_p [GeV] | pi+ | pi- | K+ | K- | K_L | K_S | Lambda | p | n |
|---|---|---|---|---|---|---|---|---|---|
| 95 | 0.986 | 0.973 | 0.892 | 0.800 | **0.956** | 0.950 | 0.874 | 0.938 | 0.893 |
| 300 | 1.022 | 1.024 | 1.021 | 1.058 | 1.053 | 1.008 | 1.028 | 1.058 | 0.979 |
| 1000 | 0.997 | 0.996 | 0.996 | 1.053 | 1.006 | 1.306 | 0.939 | 1.025 | 0.946 |
| 3000 | 0.999 | 0.999 | 0.996 | 1.018 | 1.004 | 1.040 | 0.937 | 1.036 | 0.957 |
| 50 (DPMJET) | 1.029 | 1.037 | 1.147 | 1.068 | 1.054 | 1.159 | 1.065 | 1.178 | 1.188 |
| 20 (DPMJET) | 1.050 | 1.054 | 1.046 | 0.952 | 1.057 | 1.108 | 0.975 | 1.161 | 1.205 |

Milestone 1 flagged `K_L` at **0.160 vs 0.226 = 0.71** with a nitrogen-only
target.  With the air mix it is **0.216 vs 0.226 = 0.96**: the target mix *was*
the explanation, as expected.  Pions now agree to 0.3% at 1-3 TeV.  Below the
splice DPMJET-III-19.3 gives 3-5% more pions and 16-21% more nucleons than
MCEq's SIBYLL23D table — a real model difference, since MCEq's `SIBYLL23D`
tables run SIBYLL all the way down while we switch to DPMJET at 80 GeV.

### 2.5 Rung B — chromo vs MCEq, collinear, B = 0

`MC / MCEq`, `dN/dE` integrated over each band; 200 000 showers at 20 and
100 GeV, 50 000 at 1000 GeV; errors are the MC statistics only.

| E_p | species | 0.1-0.3 GeV | 0.3-1 GeV | 1-3 GeV | 3-10 GeV | 10-30 GeV |
|---|---|---|---|---|---|---|
| 20 GeV | nu_mu | 0.869±0.001 | 0.911±0.002 | 0.943±0.005 | 0.910±0.019 | — |
| | antinu_mu | 0.864±0.001 | 0.909±0.002 | 0.918±0.005 | 0.914±0.025 | — |
| | nu_e | 0.864±0.002 | 0.908±0.003 | 0.941±0.007 | 0.953±0.032 | — |
| | antinu_e | 0.847±0.002 | 0.916±0.003 | 0.923±0.008 | 0.877±0.041 | — |
| 100 GeV | nu_mu | 0.867±0.001 | 0.922±0.001 | 0.953±0.002 | 0.995±0.004 | 0.983±0.013 |
| | antinu_mu | 0.864±0.001 | 0.920±0.001 | 0.952±0.002 | 0.973±0.004 | 0.932±0.016 |
| | nu_e | 0.851±0.001 | 0.919±0.002 | 0.953±0.003 | 0.979±0.008 | 1.017±0.030 |
| | antinu_e | 0.846±0.001 | 0.921±0.002 | 0.954±0.004 | 0.945±0.009 | 0.871±0.037 |
| 1000 GeV | nu_mu | 0.868±0.001 | 0.897±0.001 | 0.909±0.002 | 0.957±0.003 | 0.974±0.005 |
| | antinu_mu | 0.867±0.001 | 0.896±0.001 | 0.908±0.002 | 0.961±0.003 | 0.971±0.006 |
| | nu_e | 0.851±0.002 | 0.895±0.002 | 0.918±0.003 | 0.957±0.007 | 0.963±0.015 |
| | antinu_e | 0.843±0.002 | 0.893±0.002 | 0.909±0.003 | 0.964±0.007 | 0.962±0.016 |

The comparison that isolates the **generator** is B/A2 (rung A2 shares B's
decay module, so the decay-kinematics residual cancels).  `closure_A2.npz` and
a new `closure_A2_1000.npz` give:

| E_p | species | 0.1-0.3 GeV | 0.3-1 GeV | 1-3 GeV | 3-10 GeV |
|---|---|---|---|---|---|
| 20 GeV | nu_mu | 0.900±0.002 | 0.926±0.003 | 0.968±0.006 | 0.908±0.026 |
| | nu_e | 0.917±0.003 | 0.926±0.004 | 0.957±0.010 | 0.957±0.044 |
| 100 GeV | nu_mu | 0.884±0.001 | 0.934±0.002 | 0.976±0.003 | 0.991±0.006 |
| | antinu_mu | 0.883±0.001 | 0.934±0.002 | 0.979±0.003 | 0.976±0.006 |
| | nu_e | 0.895±0.002 | 0.937±0.003 | 0.967±0.005 | 0.994±0.011 |
| | antinu_e | 0.887±0.002 | 0.939±0.003 | 0.979±0.005 | 0.972±0.013 |
| 1000 GeV | nu_mu | 0.862±0.001 | 0.907±0.002 | 0.944±0.002 | 0.956±0.004 |
| | nu_e | 0.876±0.002 | 0.907±0.003 | 0.950±0.005 | 0.967±0.009 |

**Above 1 GeV the generator difference is 2-6%, i.e. inside the expectation.
Below 0.3 GeV it is 10-14%, above the 5% threshold, so it was diagnosed rather
than tuned.**

### 2.6 Diagnosis of the sub-0.3 GeV deficit

Same 100 GeV vertical proton, both backends, instrumented per generation
(`scratchpad/mc3d/gendiag.py`, 800 showers each; the RNG stream is identical up
to the first interaction):

| per shower | MCEq yields | chromo | chromo/MCEq |
|---|---|---|---|
| interactions | 7.236 | 6.609 | **0.913** |
| pi+- tracked | 20.148 | 19.034 | 0.945 |
| `<E_pi>` [GeV] | 2.452 | 2.846 | **1.161** |
| K tracked | 1.994 | 1.995 | 1.001 |
| mu+- | 20.435 | 19.297 | 0.944 |
| `<E_mu>` [GeV] | 1.824 | 2.109 | 1.156 |
| nu (> 0.1 GeV) | 32.771 | 31.740 | 0.969 |
| nu, 0.1-0.3 GeV | 16.254 | 15.106 | **0.929** |
| nu, > 0.3 GeV | 16.517 | 16.634 | 1.007 |
| nucleons reaching the ground | 7.559 | 5.855 | 0.775 |

Interactions by generation (per shower): MCEq 1.000 / 1.677 / 1.794 / 1.480 /
0.805 / 0.340 / 0.106; chromo 1.000 / 1.516 / 1.691 / 1.344 / 0.711 / 0.261 /
0.071 — chromo is **lower in every generation beyond the first**, by 6-10%
each, compounding.  Decay fractions are stable (`mu_decay` 15.961 vs 14.945,
`mu_ground` 2.458 vs 2.649, `h_decay` 22.246 vs 21.035), so the
interaction/decay competition is not the culprit.

So the deficit is **not** a missing yield: the spectrum-weighted moments
`Z = sum x^(gamma-1)` at `gamma = 2.7` are *higher* in chromo than in the MCEq
tables at every energy checked (`pi+` 1.01-1.13, `pi-` 1.06-1.21, `p` 1.03-1.20
of MCEq at 20-1000 GeV).  It is a **harder, less multiplicative cascade**:
chromo's secondaries carry more energy each (`<E_pi>` +16%) and there are 8.7%
fewer interactions, so the low-energy tail of the pion spectrum — which is
what makes 0.1-0.3 GeV neutrinos — is thinner.  Two candidate origins, both
genuine model differences and neither a wiring error:

1. **The 80 GeV splice.**  Sub-GeV neutrinos come from the last generations,
   i.e. from projectiles at 10-80 GeV, where we run DPMJET-III-19.3 and MCEq's
   `SIBYLL23D` yield tables run SIBYLL.  §2.4 shows the two differ by 3-21%
   there.
2. **SIBYLL just above its validity edge.**  `Sibyll23d._ecm_min = 10 GeV` is
   `E_lab ~ 53 GeV` for a proton, and the 95 GeV row of §2.4 (0.89-0.99 across
   the board, against 1.00-1.06 at 300-3000 GeV) is chromo's SIBYLL being
   noticeably softer than MCEq's SIBYLL table right at that edge.

**Open item for milestone 6.**  Whether MCEq's shipped `SIBYLL23D` yield
matrices are pure SIBYLL below 80 GeV or already the DPMJET-spliced
(`_lext_dpm193`) variety was not established here; if they are spliced, the
attribution flips from "generator difference" to "chromo's DPMJET differs from
MCEq's tabulated DPMJET", which is a database-version question.  The decisive
test is one rung-B run with `--mode chromo` and the splice moved to 53 GeV.

---

## 3. Milestone 2b — the first 3D run

### 3.1 What runs

`sky.py`, Kamioka (36.4267 N, 137.31 E), **no geomagnetic cutoff and no field**
(that is milestone 3), so this run measures the **pure 3D production
geometry**.  Nested caps `theta_D in {2.5, 5, 7.5, 10} deg` scored
simultaneously on every neutrino; injection patch `theta_inj = theta_D_max +
30 deg = 40 deg` on the 100 km sphere; Honda's grid (20 E-bins/decade from
0.1 GeV, `Delta cos Z = 0.1`, 12 azimuth bins of 30 deg); GSF primaries;
SIBYLL-2.3d / DPMJET-III-19.3 through the pool.

**Primary model.**  `crflux`'s `GlobalSplineFitBeta` is a *spline interface*
version: it carries the total nucleon flux and the p/n split but
`nucleus_ids == []` and `_nucleus_flux` returns zeros, so the plan's "GSF, p
and He, superposition" cannot be built from it species by species.  That costs
nothing here: superposition means the cascade only ever sees free nucleons at
`E/A`, so **with no cutoff the neutrino flux is a functional of the all-nucleon
intensity and its p/n split alone**, and a species decomposition is redundant.
It stops being redundant in milestone 3, where the cutoff needs each nucleus's
rigidity — see §5.1.  `sky.py --model H3A --groups p He` runs the per-species
version (`HillasGaisser2012:H3a` does have per-nucleus splines) and is the
milestone-3 path.

A **units bug** was found on the way: `primaries.nucleon_intensity` returned
`crflux`'s native per-m^2 intensity while its docstring and every consumer
assumed cm^2 — a factor 10^4 on any absolute flux (it cancels in a ratio, which
is why milestone 1 never saw it).  Fixed, with a check against the textbook
all-nucleon flux (`1.8 E^-2.7`): GSF gives 3.44e-3 vs 3.6e-3 cm^-2 s^-1 sr^-1
GeV^-1 at 10 GeV.

Note also that `PHASE2_PLAN.md` sec. 7's cost estimate uses
`A_patch = 6.2e19 cm^2` for `theta_inj = 40 deg`; the correct value is
`2 pi R_inj^2 (1 - cos 40 deg) = 6.15e17 cm^2`, a factor 100 smaller.  That is
one reason its "entries per injected nucleon" number has to be replaced by a
measurement.

### 3.2 Stratification, and the chunking bug it exposed

Eight strata of `10^0.5` in `ln E` over `[1, 10^4] GeV/nucleon`, allocated
`n_k ~ E_k^-0.8` (`sky.STRATA_FRAC`) — a compromise between Neyman's
variance-optimal `E_k^-1.7` for this weight (`w ~ J(E) E ~ E^-1.7`) and a flat
allocation, which would starve `E_nu > 3 GeV`.  Each (species, stratum, chunk)
is one task, so the per-stratum yield and cost come out of the run for free
(`shard_meta`).

A flat *chunk* size does not work with a steep allocation: the top stratum then
becomes a single task of ~10^3 showers at 10^4 GeV, each ~100x the cost of a
2 GeV shower, and it holds the pool open for over an hour after every other
task has finished.  Measured the hard way (the first 1.2e6-shower attempt sat
at 480/485 tasks for 20 minutes and was killed).  `build_tasks` now scales the
chunk as `chunk * E_0/E_k` with a floor of 100, so every task costs roughly the
same wall time.

The pilot below is truncated at `--nstrata 6`, i.e. primaries up to
**316 GeV/nucleon**.  That is complete for `E_nu <~ 3 GeV` (the deliverable
band) and progressively incomplete above it; the *ratio* `R` is far less
sensitive to the truncation than the flux is, since numerator and denominator
lose the same primaries.

### 3.3 The 1D reference and the two measures

`scoring.DualCapScorer` scores every neutrino twice: **3D** along its own
direction from its own production point, and **1D** along the *primary's*
direction from the injection point (that ray's ground crossing and local
zenith/azimuth are computed once per shower in `begin_shower`).  Both tallies
keep two sums:

* `s0 = sum w` — the ground-crossing measure, `int dOmega cos psi Phi = s0/(A dE)`
* `s1 = sum w / cos psi` — the specific intensity, `Phi = s1/(A dOmega dE)`

plus their squared sums.  Both are reported because `PHASE1_RESULTS.md` sec. 8.2
shows the whole "conservation excess" question is a choice of measure.

### 3.4 Up-going neutrinos: how they are treated

`NestedCaps.crossing` takes only the **first (near) ground-sphere root**.  For a
straight ray starting above the ground sphere that is the only physical
crossing: `r(s)` has a single minimum, so a ray with `dr/ds > 0` never comes
back.  The only way to populate `cos Z < 0` at the detector is a shower on the
far side of the Earth whose neutrino crosses the ground sphere twice — once,
down-going, at its own production side (the root we take) and once, up-going,
under the detector.  That needs a second injection patch at the antipode and
the *second* root.

**This run therefore scores down-going neutrinos only, and the `cos Z < 0` bins
are empty by construction rather than by physics.**  The scorer asserts this in
`test_sky.py::test_cap_scorer_reproduces_an_isotropic_field`.  Adding up-going
is independent of milestone 3's cutoff and bending work (§5.2, item 5).

---

## 5. Explicit list for milestone 3

### 5.1 Cutoff acceptance at the injection point (3a)

1. `sky._shard` already carries the nucleus rigidity per sample in
   principle — but **not with the GSF nucleon-level model**, which has no
   per-nucleus splines.  Milestone 3 must either switch to `--model H3A`
   (per-species `p, He, CNO, MgSi, Fe`, exact `R = (A/Z) p_nucleon`, already
   coded in `sky.GROUPS`) or bring in a GSF species table from outside
   `crflux`.  This is a prerequisite, not a detail: the cutoff is the only
   place the species decomposition matters, and it is the whole East-West
   effect.
2. A batched `geomag_backtrace.backtrace_vec` call at the **injection point**
   (not the site line of sight) per chunk of primaries, before any cascade is
   run, returning `w_cutoff in {0, 1}`; the chunked task structure of
   `sky.build_tasks` is already the right granularity (a chunk is 1000-2500
   primaries with a common seed).
3. A cached admittance map `A(R, zenith, azimuth)` on the injection sphere per
   site, built with `cutoff_from_states`, storing the **admittance fraction
   over the rigidity bin** and not just `R_U` — the allowed islands below `R_U`
   are real (`[9.63, 10.13] GV` at the Kamioka vertical).
4. The gate: on-the-fly binary vs the map at ~500 directions.
5. Because the cutoff kills the low-rigidity end, the stratum allocation of
   §3.2 must be re-derived *after* the cutoff is in: the `E^-0.8` weighting is
   optimal for a no-cutoff response and will be wrong once the sub-10 GV
   primaries are removed at Kamioka.

### 5.2 Azimuthal reuse with the rotated muon bend (3b)

1. `N_az = 12-24` copies of each shower, rotated about the site vertical.  With
   `B = 0` and no cutoff (this milestone) the azimuth dependence is *exactly*
   flat and the reuse is identical to summing the azimuth axis — which is why
   milestone 2b's azimuth-summed tables are free and why the reuse buys
   nothing yet.  It starts paying the moment the cutoff enters, because then
   the only azimuth dependence is the per-copy scalar `w_cutoff(R, n_p^(psi))`.
2. Per-copy cutoff weight: read the admittance map at the rotated primary
   direction.  One map lookup per copy, no re-simulation.
3. The muon bend: transport the muon with `B = 0`, record the decay point and
   the accumulated `Phi = int ds / r_gyro`, then per copy rotate the muon's
   direction *at decay* by `Phi` about the **copy's own** local `B`.  The
   displacement error is `(1 - cos Delta_theta) L <~ 0.004 L`, under 50 m over
   a 10 km muon path, against a 1117 km cap.  `shower.transport_muon` already
   integrates the direction in the field; it needs a `bend_only=True` mode that
   accumulates `Phi` without rotating.
4. **The exact re-transport gate**: re-run the muon segment per copy in the
   copy's own field at low statistics and compare the East-West amplitude to
   the rotated-bend approximation.  This is the one gate that cannot be skipped
   — the rotated bend is the single approximation the whole East-West
   measurement rests on.  Hadron bending stays off (`pi` 0.048 deg, `K` 0.0064
   deg per lifetime against a 4-39 deg production cone) with an A/B run as the
   check.
5. Up-going neutrinos (§4.4) need a second injection patch at the antipode and
   the *second* ground-sphere root; that is independent of 3a/3b and can be
   done whenever the up-going bins are wanted.
---

## 6. Tests

`test_pool.py` (8) and `test_sky.py` (8), on top of the 34 milestone-1 tests.

| test | what it pins down |
|---|---|
| `test_pool.py::test_pool_reproduces_direct_chromo_multiplicities` | the pool's per-species mean multiplicities against a **direct** `chromo` call at the same energy with the same air mix and the same `f_i sigma_i` target draw, for p and pi+ projectiles |
| `::test_pool_draw_conserves_species_and_energy` | a rescaled draw never creates energy and every particle stays on its mass shell |
| `::test_bin_choice_is_linear_in_log_e` | the mean `ln E` of the chosen bin representative equals the requested `ln E` to 0.01 — the thing that removes the frozen-multiplicity bias |
| `::test_air_mix_and_mean_mass` | the composition normalises and `<A> = 14.43` |
| `::test_chromo_sibyll_lambda_is_sane` | chromo's own SIBYLL `lambda` is physical for p, pi, K |
| `test_sky.py::test_patch_positions_are_uniform_per_unit_area` | `cos alpha` uniform on `[cos theta_inj, 1]` to 0.5% at five quantiles, and azimuthal symmetry about the axis |
| `::test_patch_directions_are_lambert_and_inward` | strictly inward, unit norm, `<mu> = 2/3`, `<mu^2> = 1/2`, `CDF = mu^2` — all to 0.4% |
| `::test_patch_area` | the cap area formula |
| `::test_cap_scorer_reproduces_an_isotropic_field` | **a synthetic isotropic field of unit intensity comes back as unit intensity in every down-going bin** (pull test, mean over 120 bins to 3%).  Positions uniform on a source sphere with inward-Lambert directions is an exact sampling of an isotropic external intensity, and every ray reaching the ground must cross that sphere inward, so the test closes the whole normalisation chain at once: cap area, `1/cos psi`, solid-angle measure, `cos Z`/azimuth binning.  Because the field is isotropic it *is* the "equal counts per unit solid angle" test.  It also asserts the up-going bins are empty (§4.4) |
| `::test_nested_caps_are_nested_and_monotone` | containment is monotone in the cap index and the hit counts scale as `1 - cos theta_D` to `5 sqrt(N)` |
| `::test_cap_scorer_local_frame_is_at_the_crossing_point` | a ray vertical at the cap *edge* is scored at `cos Z = 1`, not at `cos theta_D` |
| `::test_collinear_cascade_makes_the_two_tallies_identical` | in collinear mode the 3D and the correlated-1D tallies agree **bit for bit** (`rtol = 1e-12`) in both measures — the sharpest available test of the 1D scorer, and the reduced-statistics version of the milestone-1 collinear closure since `sky.py --collinear` is the same code path |
| `::test_chromo_collinear_yield_matches_the_mceq_yield_backend` (slow) | rung B at 1/2000 statistics: the chromo cascade's `nu_mu` yield inside 30% of the MCEq-yield cascade, which bounds any wiring error in the pool backend (skipped without `MC3D_POOL_CACHE`) |

Result: **16 passed** (`test_sky.py`, `test_pool.py`), plus the 34
milestone-1 tests unchanged.
---

## 7. Reproducing

```sh
cd tools/mc3d
export PY=/cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc14-opt/bin/python3.12
export PYTHONPATH=<scratchpad>/pylib:<repo>/.venv/lib/python3.12/site-packages:\
<repo>/src:<repo>/tools/mceq3d:$PWD
cd <scratchpad>/mc3d        # chromo writes tables.dat into the cwd

# 1. the event-pool cache (708 pools, 321 MB, 90 s on 40 cores; resumable)
$PY <repo>/tools/mc3d/prebuild_pools.py --cache ./pools --emax 1.1e4 \
    --npool 2000 --nproc 40

# 2. rung B and its A2 partner at 1000 GeV
$PY <repo>/tools/mc3d/closure.py --energies 20 100 --nshower 200000 --nproc 40 \
    --mode chromo --cache ./pools --shard 500 --out closure_B.npz
$PY <repo>/tools/mc3d/closure.py --energies 1000 --nshower 50000 --nproc 40 \
    --mode chromo --cache ./pools --shard 200 --out closure_B1000.npz
$PY <repo>/tools/mc3d/closure.py --energies 1000 --nshower 50000 --nproc 40 \
    --mode kinematic --shard 200 --out closure_A2_1000.npz
$PY <repo>/tools/mc3d/summarize_closure.py closure_B.npz closure_B1000.npz

# 3. the diagnosis of the sub-0.3 GeV deficit
$PY gendiag.py 800                  # per-generation instrumented shower

# 4. the 3D pilot, the patch gate and the collinear twin
SK=<repo>/tools/mc3d/sky.py
C="--nproc 40 --cache ./pools --chunk 6000 --nstrata 6 --outdir ./sky"
$PY $SK --nshower 1500000 $C --tag p3d --seed 20260910
$PY $SK --nshower 250000  $C --tag g40 --seed 777 --theta-inj 40
$PY $SK --nshower 250000  $C --tag g55 --seed 777 --theta-inj 55
$PY $SK --nshower 1500000 $C --tag p1d --seed 20260910 --collinear

$PY <repo>/tools/mc3d/analyse_sky.py ./sky/p3d_total.npz \
    --collinear-total ./sky/p1d_total.npz --erebin 2 --czrebin 2

# 5. tests
MC3D_POOL_CACHE=$PWD/pools $PY -m pytest <repo>/tools/mc3d -q -m "not slow"
```
