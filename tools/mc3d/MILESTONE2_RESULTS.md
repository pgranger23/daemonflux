# `tools/mc3d` milestone 2 — chromo in the driver, and the first 3D run

Factual record.  Every number comes from a run of the code in this directory
and is reproduced by the commands in §7.  §2 (milestone 2a) was measured on
2026-09-10; its raw outputs lived on a scratch area that was wiped by a reboot,
so its tables are quoted as recorded then.  §2.4/§2.6 were corrected and §3-§4
(milestone 2b) were re-measured on **2026-10-08**; their raw outputs (per-task
shard files, logs) are on local disk under `/home/pigrange/mc3d_work/runs/m2b/`
and the small summary tables are copied to `results/m2b/` in this directory.

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
| **2026-10-08 (2b re-run)** | |
| `scoring.py:ConeScorer` | histogram + exact moments of the space angle between the **primary** and every produced **neutrino**, by species, `E_nu`, parent channel (direct pi / K / mu decay / other), primary zenith class and generation (§3.5) |
| `scoring.py:DualCapScorer` (`wants_parent`, `end_shower`, `hinj`, `shcounts`) | feeds the cone tally; shower-level scored counters (the milestone-3 cutoff-tracing cost); the injection-point distance of every cap hit (gate G8 measured directly); `cos Z = 1` now lands in the top bin instead of being dropped |
| `shower.py:run_shower`, `_child` | every stack entry carries the PDG id of its decaying parent and its interaction generation; a scorer with `wants_parent = True` receives both (`add(..., parent=, gen=)`); other scorers unchanged |
| `sky.py:primary_sample` | **normalisation fix**: the importance weight is divided by the stratum total, not by the chunk size (§3.2) |
| `sky.py:run`, `_shard`, `merge_tasks`, `task_path` | every task writes its own compressed shard file; a job re-launched with the same arguments skips finished tasks (manifest-guarded); totals are merged from disk |
| `analyse_sky.py:JackRun`, `jack`, `m2b_report`, `engine_kernel` | stratified delete-a-group jackknife over the task files (honest errors for ratios, cap fits and cone moments, shower correlations included); the §4 report; the factorised engine's cone kernel as the engine evaluates it |
| `diag_m2b.py` (`db`, `splice`, `dpm`, `sigma`, `ptpool`) | the §2.4/§2.6 database and generator diagnosis and the §4.4 check of the factorised engine's cone input, kept in the repo (the 2026-09-10 `gendiag.py` was lost with the scratch area) |

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
splice the pool gives 3-5% more pions and 16-21% more nucleons than the MCEq
table.

**Correction (2026-10-08): below ~50 GeV this is DPMJET against DPMJET.**  The
2026-09-10 text read the low rows as SIBYLL (table) vs DPMJET (pool).  That is
wrong.  `mceq_tables_SIBYLL23D.npz` is exported by `build_tables.py` with
MCEq's default configuration, whose database (`config.mceq_db_fname`, and the
only file in `MCEq/data/`) is `mceq_db_lext_dpm193_v140.h5`: a fresh export
reproduces the shipped npz with `max|diff| = 0`.  In that database the
`SIBYLL23D` yield columns are **bit-identical** to the `DPMJETIII193` ones
(ratio 1.00000, every cell equal) at every projectile kinetic energy up to
**44.7 GeV**, for every channel checked (p -> pi+-, p, n, K+; pi+ -> pi+-).
The splice sits between the 44.7 and 56.2 GeV kinetic bins, not at 80 GeV:
at 56.2 GeV every SIBYLL23D column is exactly **0.800 x DPMJET** (all three
channels checked -- a blend weight against an empty column, i.e. a 20% yield
hole in MCEq's own cascade for projectiles at ~50-63 GeV), 70.8 GeV is a
blend, and the columns are pure SIBYLL from ~89 GeV.  So the 20 and 50 GeV
rows above compare chromo's DPMJET-III-19.3 with MCEq's *tabulated*
DPMJET-III-19.3 (`scripts`: `db_check.py`, `splice_bin.py`, §7).  §2.6 says
what actually differs.

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
(`gendiag.py`, 800 showers each, 2026-09-10 -- the script was lost with the
scratch area; the RNG stream is identical up to the first interaction):

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
what makes 0.1-0.3 GeV neutrinos — is thinner.  The 2026-09-10 record listed
two candidate origins: (1) the 80 GeV splice, on the premise that MCEq's
`SIBYLL23D` tables run SIBYLL below 80 GeV, and (2) SIBYLL just above its
validity edge (`Sibyll23d._ecm_min = 10 GeV`, `E_lab ~ 53 GeV`), with the
database question left open.

**Resolution of the open item (2026-10-08).**  The premise of (1) is false
(§2.4 correction): below ~50 GeV both sides run DPMJET-III-19.3, so the deficit
is chromo's DPMJET against MCEq's *tabulated* DPMJET plus the SIBYLL first
interactions.  A per-interaction comparison, p + air, pool against table at
matched projectile energy, `x = E_kin(secondary)/E_kin(projectile)`
(`dpm_diag.py`, 2000 pool events per bin, 17.9 / 22.5 / 44.9 GeV):

| per interaction | 17.9 GeV | 22.5 GeV | 44.9 GeV |
|---|---|---|---|
| pi+ / pi- total, pool / table | 0.999 / 0.996 | 0.984 / 0.988 | 1.001 / 1.001 |
| pi+ at x = 0.05-0.3, pool / table | 0.88-0.95 | 0.91-1.01 | 0.85-1.02 |
| leading nucleon x > 0.9 (quasi-elastic-like), pool vs table | 0.013 vs 0.014 | 0.015 vs 0.016 | 0.018 vs 0.019 |
| slow nucleons (E_tot < 2 GeV, x < 0.02), pool vs table | 1.79 vs 1.22 | 1.75 vs 1.21 | 1.77 vs 1.21 |
| **re-interacting nucleons (E_tot > 2 GeV)**, pool vs table | **1.37 vs 1.48** | **1.40 vs 1.50** | **1.50 vs 1.59** |
| Sigma+- per interaction in raw chromo (dropped by `pool.KEEP`) | -- | 0.050 | 0.054 |
| their decay nucleons above 2 GeV | -- | 0.031 | 0.037 |

* The **"+16-21% nucleons" of §2.4 is entirely slow target-side nucleons**
  below the 2 GeV re-interaction cut (an evaporation/fragmentation setting of
  the generator); they cannot reach the neutrinos.
* Charged-pion multiplicities agree to 0.1-1.6%, and the leading
  (quasi-elastic-like) nucleons agree: the **quasi-elastic treatment is not
  the difference**.  **Energy definitions** are consistent (kinetic table grid,
  total-energy cascade, converted at every lookup; the comparison above is in
  kinetic `x`).
* What differs is the **re-interacting** part of the final state: 6-7% fewer
  nucleons above 2 GeV (3% at matched bin edges at 44 GeV) and ~5% fewer
  charged pions above ~1 GeV per interaction in the pool.  The **Sigma+-** that
  `pool.KEEP` drops (MCEq's tables decay them: their Sigma+- columns are
  empty) account for about half of the nucleon part, and all of it at 44 GeV
  (SIBYLL at 95 GeV: 0.081 Sigma+- and 0.066 decay nucleons per interaction,
  the same direction as its 0.89-0.94 nucleon row in §2.4).
* Compounded over 3-5 generations, a few-% deficit of re-interacting
  particles per interaction is exactly the per-generation pattern of the
  table above (-6 to -10% interactions per generation, -8.7% per shower), and
  through it the -13% at 0.1-0.3 GeV; the soft-pion yield per interaction is
  at parity.
* The splice-position difference that does exist acts the **other** way:
  between `E_kin ~ 50` and 79 GeV MCEq is in its blend / SIBYLL region (the
  20% hole at 56 GeV) while the pool still runs DPMJET (pool / table at
  56.6 GeV: pi+- 1.24, nucleons 1.47), which makes chromo *higher*.

**Classification** (not tuned): not a wiring error and not a model-family
difference.  (i) **Ours, bookkeeping** -- Sigma+- are dropped instead of
decayed; worth ~3% of the re-interacting nucleons per interaction (the fix is
to add 3112/3222 to `pool.KEEP` and give `decays.py` the Sigma -> N pi
channels).  (ii) **Database/configuration** -- the remainder of the
re-interacting-particle deficit is chromo's DPMJET-III-19.3 against the
DPMJET-III-19.3 configuration MCEq tabulated (the slow-nucleon excess shows
the two configurations differ).  (iii) The database splice at ~50 GeV and its
56 GeV blend hole are real features of `mceq_db_lext_dpm193_v140.h5`, opposite
in sign; the hole also sits in every MCEq/daemonflux 1D flux built on that
database (flagged for `tools/mceq3d`, not acted on here).  The 3D/1D *ratio*
is insensitive to all three at the few-% level, since numerator and
denominator share the cascade.

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

**The chunking left a normalisation bug (found and fixed 2026-10-08).**  The
log-uniform importance weight `J(E) E ln(E2/E1) / n * pi A_patch` was divided
by the *chunk* size `n`, so every chunk was a complete estimate of its stratum
and the merged sum counted each stratum once per chunk: **152 / 191 / 240 /
303 / 228 / 91 times** for strata 0-5 at the §7 pilot settings.  Every
absolute flux was therefore 90-300x too high, and the 3D/1D ratio was a
chunk-count-weighted mix of strata (over-weighting the high-energy strata by up
to 2x relative to stratum 0).  `sky.primary_sample` now divides by the stratum
total `n_tot`, carried in the task tuple; regression
`test_sky.py::test_stratum_weights_do_not_depend_on_the_chunking` (the summed
weight equals `pi A_patch int J dE` to 2% for two different chunkings).  All
numbers in §4 are post-fix; nothing from the lost 2026-09-10 pilot survives
to be corrected.

The pilot below is truncated at `--nstrata 6`, i.e. six half-decade strata,
primaries of **1-1000 GeV/nucleon** (the 2026-09-10 text said 316, which is
five).  §4.6 measures the effect on the flux: it is complete to a few % up to
`E_nu ~ 3 GeV` (the deliverable band) and 1-22% low at 5 GeV (MC/HKKM2014
0.78-0.99 in the well-measured bins); the *ratio*
`R` is far less sensitive to the truncation than the flux is, since numerator
and denominator lose the same primaries.

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

### 3.5 The production-cone tally and the auxiliary tallies (2026-10-08)

`scoring.ConeScorer` records, for **every neutrino produced** above
`e_nu_min` (whether or not it reaches the cap -- the kernel is a property of
the production, and conditioning on the cap would fold in the detector
geometry), the space angle `alpha` between the **primary** direction and the
neutrino direction, with its physical weight.  That is precisely the kernel
`K(alpha)` of the factorised engine's Eq. (8) ("the distribution of the
primary direction given the neutrino direction", which, `K` being
azimuthally symmetric about the primary axis, is the distribution of `alpha`
over the produced neutrinos).  Axes: species; `E_nu` bands 0.1-0.3, 0.3-0.5,
0.5-1, 1-3 GeV (1-deg `alpha` bins) and exact moments `sum w`, `w alpha`,
`w alpha^2`, `w cos alpha` on 10 bins/decade whose centres are the engine's
energy grid; **parent channel** (direct pi, K = K+- / K_L / K_S including
K_l3, muon decay, other), tagged for free because `run_shower` now carries
the decaying parent's PDG id on the stack; the primary's local zenith class;
and the **generation** (neutrino whose parent meson came from the primary's
first interaction, or later), which separates the single-interaction cone the
engine assumes from multi-generation accumulation.  Gate:
`test_sky.py::test_cone_of_a_decaying_pion_matches_two_body_kinematics`
(a non-interacting 2 GeV pi+: the measured `<cos alpha>` and `<alpha^2>` match
the exact two-body integral in every 10/decade bin, the nu_mu is tagged
`pi` and the muon-decay neutrinos `mu`).

`DualCapScorer` also keeps (i) `hinj`, the angular distance between each cap
hit's injection point and the site -- the patch truncation measured directly
-- and (ii) `shcounts`, the number of showers with at least one 3D / 1D /
either cap hit and with a 3D hit in the East-West target region
(`cos Z < 0.3`, 0.3-2 GeV): the scored-shower fraction that sets the cost of a
per-shower cutoff back-trace in milestone 3a.

---

## 4. Milestone 2b — results of the first 3D run (2026-10-08)

Kamioka, no field and no cutoff, GSF nucleons, SIBYLL-2.3d / DPMJET-III-19.3
through the pool, `theta_inj = 40 deg`, caps `{2.5, 5, 7.5, 10} deg`,
`--nstrata 6` (1-1000 GeV/nucleon, `n_k ~ E_k^-0.8`), 30 workers on a shared
48-core node.  Every error below is a stratified delete-a-group jackknife
(20 groups) over the per-task shard files (`analyse_sky.JackRun`; gated in
`test_sky.py::test_stratified_jackknife_matches_the_true_variance`), so it
includes the shower-to-shower correlations that the stored `sum w^2` misses.
Full tables: `results/m2b/{pilot,x10}.txt` and `*_summary.npz`.

| tag | showers | seed | patch | mode | tasks | wall | CPU [core-h] | ms/shower |
|---|---|---|---|---|---|---|---|---|
| `p3d` (§7 pilot) | 1.5e6 | 20260910 | 40 deg | 3D | 1205 | 385 s (+72 s merge) | 3.08 | 7.38 |
| `g40` | 2.5e5 | 777 | 40 deg | 3D | 203 | 71 s | 0.51 | 7.38 |
| `g55` | 2.5e5 | 777 | 55 deg | 3D | 203 | 70 s | 0.50 | 7.25 |
| `p1d` | 1.5e6 | 20260910 | 40 deg | collinear | 1205 | 407 s | 3.25 | 7.80 |
| `p3dx` (10x) | 1.5e7 | 20261008 | 40 deg | 3D | 1745 | 4502 s (+117 s) | 37.1 | 8.90 |
| `p1dx` (10x) | 1.5e7 | 20261008 | 40 deg | collinear | 1745 | 4193 s (+122 s) | 33.9 | 8.14 |

The §7 pilot sizes ran ~10x faster than the hour the plan budgeted (the
`E^-0.8` allocation makes the average shower cheap), so the hour was spent on
a 10x pair (`p3dx`/`p1dx`, new seeds, chunk x10).  The pilot numbers are
consistent with the 10x ones wherever both are measured; the 10x run is quoted
unless stated.  Per-shower CPU is 2.1 / 7.1 / 19 / 43 / 88 / 192 ms in the six
strata (10x run; 1.6 / 6.0 / 16 / 36 / 73 / 152 ms in the pilot on a less
loaded node), i.e. **11-14 showers/s/core at 100-316 GeV/nucleon**, inside
the plan's 10-20 budget, and **112-135 showers/s/core averaged over the
stratified sample**.

### 4.1 (a) Cost: entries per primary, the 3D/1D correlation, core-hours

**Scored showers** (all strata: 3.21% with a 3D cap hit, 3.38% with a 1D
crossing in the cap, **3.58% either**, **0.40% with a 3D hit in the East-West
region** `cos Z < 0.3`, 0.3-2 GeV):

| stratum [GeV/nucleon] | showers | ms/shower | scored, either | 3D in E-W region |
|---|---|---|---|---|
| 1-3.2 | 9.06e6 | 2.1 | 0.0187 | 0.00026 |
| 3.2-10 | 3.61e6 | 7.1 | 0.0577 | 0.0046 |
| 10-32 | 1.44e6 | 19.0 | 0.0681 | 0.0145 |
| 32-100 | 5.7e5 | 42.5 | 0.0686 | 0.0208 |
| 100-316 | 2.3e5 | 88.4 | 0.0700 | 0.0252 |
| 316-1000 | 9.1e4 | 191.7 | 0.0702 | 0.0302 |

**Near-horizon entries and cost** (Honda's grid: 20 E-bins/decade, `dcos Z =
0.1`, 30-deg azimuth bins; 10-deg cap; per azimuth bin).  `N_eff/1e6` is the
neutrino-level `(sum w)^2/sum w^2` per million injected primaries;
`clus = N_eff x err^2` compares it with the jackknife (1 = the neutrinos of a
shower are independent); `rho` is the 3D/1D correlation of the bin.
Core-hours to reach 3% per azimuth bin at 8.9 ms/shower, without azimuthal
reuse / with `N_az = 12` (the reuse makes every shower fill all twelve
azimuth bins, so the per-bin error becomes the azimuth-summed one):

| species | E [GeV] | cos Z | N_eff/1e6 | err Phi_3D | clus | rho | err R | core-h 3% Phi_3D | core-h 3% R |
|---|---|---|---|---|---|---|---|---|---|
| numu | 0.3 | 0.0-0.1 | 1.19 | 0.19 | 0.63 | -0.14 | 0.28 | 1449 / 121 | 3136 / 261 |
| numu | 0.3 | 0.1-0.2 | 6.19 | 0.09 | 0.67 | +0.50 | 0.09 | 298 / 25 | 302 / 25 |
| numu | 0.5 | 0.0-0.1 | 0.24 | 0.45 | 0.75 | -0.30 | 0.56 | 8406 / 700 | 13012 / 1084 |
| numu | 0.5 | 0.1-0.2 | 3.73 | 0.13 | 1.01 | -0.23 | 0.17 | 742 / 62 | 1188 / 99 |
| numu | 1.0 | 0.0-0.1 | 0.33 | 0.35 | 0.60 | +0.19 | 0.57 | 5007 / 417 | 13615 / 1135 |
| numu | 1.0 | 0.1-0.2 | 1.68 | 0.21 | 1.12 | +0.42 | 0.21 | 1829 / 152 | 1897 / 158 |
| nue | 0.3 | 0.0-0.1 | 1.00 | 0.28 | 1.17 | -0.32 | 0.56 | 3206 / 267 | 13029 / 1086 |
| nue | 0.3 | 0.1-0.2 | 3.63 | 0.16 | 1.48 | +0.01 | 0.23 | 1120 / 93 | 2149 / 179 |
| nue | 0.5 | 0.0-0.1 | 0.48 | 0.46 | 1.53 | -0.11 | 0.71 | 8793 / 733 | 20518 / 1710 |
| nue | 0.5 | 0.1-0.2 | 2.18 | 0.14 | 0.66 | -0.17 | 0.28 | 827 / 69 | 3189 / 266 |
| nue | 1.0 | 0.0-0.1 | 0.24 | 0.63 | 1.42 | +0.47 | 0.58 | 16385 / 1365 | 13950 / 1163 |
| nue | 1.0 | 0.1-0.2 | 1.01 | 0.28 | 1.16 | +0.29 | 0.37 | 3158 / 263 | 5704 / 475 |

(antineutrinos in `results/m2b/x10.txt`; they bracket the same range, the
worst being antinue 1 GeV horizontal: 7898 / 658 and 22184 / 1849 core-h.)

* **Entries.**  0.2-1.2 effective entries per million primaries per azimuth
  bin in the horizontal bin and 1-6 in `cos Z = 0.1-0.2`, at 0.3-1 GeV.
* **The 3D/1D correlation is weak near the horizon**: `rho = -0.3 ... +0.5`.
  A near-horizontal sub-GeV neutrino and its primary's collinear entry land in
  different `cos Z` bins, so the correlated in-run reference does **not**
  reduce the variance there and the ratio costs 1.5-4x the 3D flux.  The
  plan's "÷~4 from the correlated ratio" does not materialise for the
  East-West table (§5.1 item 7).
* **Clustering is mild** (`clus` 0.5-1.5): near the horizon the neutrinos of
  one shower rarely share a bin, so `(sum w)^2/sum w^2` is a fair proxy there.
  The jackknife is still required for ratios, cap fits and the cone moments.
* **3% East-West table** (four species, 0.3-1 GeV, `cos Z < 0.2`, 20
  E-bins/decade, 30-deg azimuth): binding bin nue 1 GeV horizontal at
  **16 400 core-h without and 1 370 with `N_az = 12`** for the 3D flux
  (median over these bins 2 470 / 205), and antinue 1 GeV horizontal at
  **22 200 / 1 850 core-h** for the ratio (median 5 300 / 440).  Halving to
  10 E-bins/decade halves these.
* **5% survey** (0.1-3 GeV, 10 E-bins/decade, `dcos Z = 0.1`, 12 azimuth bins,
  four species, down-going; all 600 bins measurable): binding bin nue
  2.8 GeV horizontal at **29 700 core-h without / 2 480 with `N_az = 12`**;
  the 90th-percentile bin needs **870 / 73 core-h**.  The cost is set by a
  handful of horizontal bins.
* **Stratum allocation.**  In the binding E-W bin 25% / 71% of `var(R)` come
  from the 3-10 / 10-32 GeV strata and nothing from 1-3 GeV, which carries 60%
  of the showers; the Neyman allocation `n_k/N = (0, 0.58, 0.38, 0.03, 0.007,
  0.001)` would cut that bin's cost **x1.9**.
* **Against the plan's analytic estimate** (`~1e3` core-h for the 3% E-W table,
  `~300` for the 5% survey, with `N_az = 12`): the measured binding numbers are
  1.4-1.9e3 and 2.5e3 core-h, the median ones 0.2-0.4e3 and 73.  The analytic
  estimate landed within a factor ~2 of the median despite the 100x patch-area
  slip (§3.1), because the other factors in it were off in the opposite
  direction; it should not be reused.
* **Pilot statistics under-estimate the cost** in the sparsest bins by up to
  2-4x (numu 0.5 GeV horizontal: 291 core-h from `p3d`, 1084 from `p3dx`): the
  weights are heavy-tailed and a small sample misses the rare high-weight
  showers.  Plan production runs from the 10x numbers.
* **Caveat: no cutoff yet.**  At Kamioka the cutoff removes most of the
  1-10 GeV/nucleon primaries (84% of the showers, 33% of the CPU here); the
  allocation and the binding bins must be re-measured with the cutoff on (§5.1
  items 5-6).

### 4.2 (b) The 3D/1D ratio vs cos Z

Azimuth-averaged specific intensity, 10-deg cap, 10 E-bins/decade (bin
centres 0.282 / 0.447 / 1.12 GeV, exactly the factorised engine's grid
points), in-run ratio `s1/t1` of the 10x run, against the factorised engine's
production factor at `G == 1`, `E_off_s` of `offaxis_excess_channel_v2.npz`
(which, like this run, has no cutoff in it):

| numu | 0.0-0.1 | 0.1-0.2 | 0.2-0.3 | 0.3-0.4 | 0.4-0.5 | 0.5-0.6 | 0.6-0.7 | 0.7-0.8 | 0.8-0.9 | 0.9-1.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.28 GeV MC | 1.83±0.16 | 1.233±0.031 | 1.054±0.023 | 0.922±0.021 | 0.886±0.013 | 0.888±0.014 | 0.900±0.017 | 0.874±0.011 | 0.886±0.010 | 0.901±0.011 |
| engine | 1.743 | 1.246 | 1.047 | 0.968 | 0.933 | 0.918 | 0.912 | 0.908 | 0.907 | 0.907 |
| 0.45 GeV MC | 1.59±0.15 | 1.172±0.039 | 1.028±0.027 | 0.939±0.020 | 0.905±0.018 | 0.940±0.019 | 0.950±0.016 | 0.941±0.016 | 0.938±0.014 | 0.952±0.010 |
| engine | 1.443 | 1.128 | 1.012 | 0.974 | 0.960 | 0.955 | 0.953 | 0.952 | 0.952 | 0.951 |
| 1.12 GeV MC | 1.06±0.13 | 1.064±0.046 | 0.991±0.034 | 1.006±0.033 | 0.962±0.031 | 0.952±0.020 | 0.993±0.024 | 0.989±0.021 | 0.984±0.023 | 0.990±0.014 |
| engine | 1.128 | 1.026 | 0.997 | 0.990 | 0.989 | 0.988 | 0.988 | 0.987 | 0.987 | 0.987 |

| nue | 0.0-0.1 | 0.1-0.2 | 0.2-0.3 | 0.3-0.4 | 0.4-0.5 | 0.5-0.6 | 0.6-0.7 | 0.7-0.8 | 0.8-0.9 | 0.9-1.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.28 GeV MC | 1.71±0.21 | 1.284±0.048 | 1.004±0.035 | 0.901±0.024 | 0.880±0.020 | 0.875±0.015 | 0.865±0.016 | 0.862±0.015 | 0.857±0.017 | 0.887±0.017 |
| engine | 1.811 | 1.268 | 1.054 | 0.967 | 0.930 | 0.912 | 0.905 | 0.902 | 0.901 | 0.901 |
| 0.45 GeV MC | 1.58±0.17 | 1.053±0.058 | 0.988±0.031 | 1.008±0.033 | 0.921±0.020 | 0.911±0.023 | 0.900±0.020 | 0.976±0.023 | 0.933±0.013 | 0.939±0.016 |
| engine | 1.499 | 1.145 | 1.015 | 0.970 | 0.957 | 0.951 | 0.950 | 0.948 | 0.949 | 0.949 |
| 1.12 GeV MC | 1.02±0.20 | 0.996±0.066 | 0.927±0.042 | 1.005±0.045 | 0.971±0.049 | 1.032±0.037 | 0.948±0.030 | 0.953±0.018 | 0.990±0.025 | 0.998±0.018 |
| engine | 1.129 | 1.021 | 0.995 | 0.990 | 0.990 | 0.990 | 0.990 | 0.988 | 0.989 | 0.989 |

The **true** ratio, `p3dx` over its collinear twin `p1dx` (two complete
cascades on the same primaries, paired jackknife), confirms the in-run one,
e.g. at 0.28 GeV:

| 0.28 GeV, `cos Z` | 0.0-0.1 | 0.1-0.2 | 0.2-0.3 | 0.3-0.4 | 0.4-0.5 | 0.5-0.6 | 0.6-0.7 | 0.7-0.8 | 0.8-0.9 | 0.9-1.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| numu true | 1.72±0.11 | 1.215±0.033 | 1.070±0.025 | 0.918±0.017 | 0.920±0.018 | 0.897±0.019 | 0.909±0.018 | 0.874±0.010 | 0.883±0.011 | 0.909±0.014 |
| numu in-run 1D / true 1D | 0.94±0.09 | 0.986±0.030 | 1.015±0.022 | 0.996±0.024 | 1.039±0.018 | 1.009±0.019 | 1.010±0.016 | 1.000±0.015 | 0.996±0.011 | 1.009±0.014 |
| nue true | 1.84±0.18 | 1.258±0.050 | 1.006±0.029 | 0.902±0.018 | 0.901±0.021 | 0.842±0.020 | 0.908±0.020 | 0.852±0.017 | 0.863±0.019 | 0.891±0.020 |
| nue in-run 1D / true 1D | 1.08±0.09 | 0.980±0.040 | 1.002±0.033 | 1.001±0.027 | 1.024±0.031 | 0.962±0.016 | 1.050±0.016 | 0.988±0.021 | 1.006±0.017 | 1.005±0.020 |

The in-run 1D reference (neutrinos of the 3D-transported cascade laid on the
primary's line) agrees with a genuinely collinear cascade to 1-4% per bin
away from the horizon and within the 9% errors at it, and to 0.1-1% in the
hemisphere sums (§4.3): the secondaries' different slant depths in a 3D
cascade do not bias the reference measurably.

* **Agreement at the 5% level in shape.**  The MC reproduces the engine's
  production factor everywhere within ~5% except in a systematic way at
  0.28 GeV: from `cos Z = 0.3` to the vertical the MC sits **1-7% below** the
  engine (numu 1-5%, nue 2-7%), i.e. a deeper 3D deficit, while the
  horizontal bins agree within their 9-12% errors.
* **Which cone does the MC prefer?**  Against the engine's own tables built
  with its cone scaled by 0.88 / 1 / 1.12 (`E_off_lo_s`, `E_off_s`,
  `E_off_hi_s`), all four species, three energies and ten zeniths:

  | `chi^2` | all 120 bins | 36 bins `cos Z < 0.3` | 84 bins `cos Z >= 0.3` |
  |---|---|---|---|
  | in-run ratio, cone x0.88 / x1 / x1.12 | 483 / 295 / 254 | 56 / 68 / 114 | 427 / 227 / 140 |
  | true ratio, cone x0.88 / x1 / x1.12 | 360 / 232 / 220 | **40** / 53 / 100 | 320 / 178 / **120** |

  The MC wants a **wider cone away from the horizon** (the deeper deficit)
  and a **nominal-or-narrower cone near it** (the true ratio's 36
  near-horizon bins fit the x0.88 table at `chi^2 = 40/36`).  No single
  Gaussian width does both, and §4.4 shows why: the MC kernel is
  heavy-tailed, with a core (median, 68% quantile) like the engine's at
  0.3-1 GeV and a tail far wider; the tail drives `<cos alpha>` (the deficit),
  the core drives the horizon excess.
* **Horizon / vertical contrast** `R(0-0.1)/R(0.9-1)`, true ratio (in-run in
  brackets): numu 1.89±0.13 (2.04±0.18) / 1.63±0.18 (1.67±0.15) / 1.10±0.12
  (1.07±0.13) at 0.28 / 0.45 / 1.12 GeV against the engine's 1.92 / 1.52 /
  1.14; nue 2.06±0.21 (1.92±0.23) / 1.46±0.22 (1.69±0.19) / 1.20±0.19
  (1.02±0.20) against 2.01 / 1.58 / 1.14.  **The factorised engine's H/V
  production factor is confirmed within the 6-15% errors** at every point.
* **Honda 2004** (horizontal 3D/1D > 1.5 at 0.1-0.32 GeV, <~1.1 at
  1-3.2 GeV, *with* his cutoff): the MC's horizontal (`cos Z` 0-0.1), all four
  species, band-integrated, true ratio (in-run): **1.99 ± 0.05** (2.04) at
  0.1-0.32 GeV, **1.43 ± 0.06** (1.45) at 0.32-1 GeV, **1.03 ± 0.05** (1.07)
  at 1-3.16 GeV; at `cos Z = 0.1-0.2`: 1.34 / 1.08 / 0.99.  Same pattern and
  magnitude as Honda's; his cutoff modifies it near the horizon, so the
  closer comparison waits for milestone 3.
* **G7 (nested caps).**  With `B = 0` and no cutoff nothing depends on the cap
  size, and the hemisphere-summed ratio is cap-independent:
  `R(7.5 deg)/R(10 deg) = 0.987-1.001 ± 0.007-0.021` and `R(5 deg)/R(10 deg) =
  0.986-1.034 ± 0.02-0.05` at 0.3-1 GeV.  Per bin, the linear `Omega_D -> 0`
  extrapolation only amplifies noise here (the 2.5-deg cap has 1/16 of the
  statistics), so the 10-deg cap is the estimator until the field is on; the
  extrapolation column is in `results/m2b/x10.txt`.

### 4.3 (c) The two measures, and the flat-atmosphere argument

Down-going averages of the 3D/1D ratio vs energy (10-deg cap, 10x run;
"MC" = in-run reference, "MC true" = `p3dx/p1dx`), against the factorised engine at `G == 1` (its numbers
recomputed here from `E_off_s` on the same bins; `PHASE1_RESULTS.md` §8.2
quotes 1.046 / 0.945 at 0.30 GeV):

| numu | 0.2 GeV | 0.28 GeV | 0.45 GeV | 1.1 GeV | 2.0 GeV | 2.8 GeV |
|---|---|---|---|---|---|---|
| plain `<F>_Omega` (unweighted over `cos Z`), MC | 1.023±0.012 | 1.038±0.016 | 1.035±0.015 | 0.999±0.012 | 0.980±0.013 | 0.996±0.017 |
| plain `<F>_Omega`, MC true (p3dx/p1dx) | 1.011±0.013 | 1.031±0.009 | 1.033±0.018 | 0.989±0.018 | 0.989±0.025 | 1.006±0.032 |
| plain `<F>_Omega`, engine | 1.060 | 1.049 | 1.028 | 1.007 | 1.002 | 1.001 |
| `<F>_cos` (`cos Z`-weighted), MC | 0.883±0.002 | 0.920±0.002 | 0.958±0.002 | 0.987±0.002 | 0.991±0.002 | 0.996±0.002 |
| `<F>_cos`, MC true | 0.886±0.007 | 0.925±0.006 | 0.962±0.006 | 0.982±0.010 | 0.994±0.016 | 1.007±0.020 |
| `<F>_cos`, engine | 0.923 | 0.941 | 0.968 | 0.991 | 0.996 | 0.997 |
| `sum s0 / sum t0` (neutrinos through the ground), MC | 0.882±0.001 | 0.919±0.002 | 0.956±0.002 | 0.985±0.002 | 0.994±0.002 | 0.996±0.002 |
| `sum s0 / sum t0`, MC true | 0.886±0.007 | 0.924±0.006 | 0.961±0.006 | 0.981±0.009 | 0.993±0.016 | 1.004±0.021 |
| F(vertical, `cos Z` 0.9-1), MC | 0.830±0.007 | 0.901±0.011 | 0.952±0.010 | 0.990±0.014 | 0.976±0.018 | 1.000±0.015 |
| F(vertical), MC true | 0.844±0.011 | 0.909±0.014 | 0.948±0.016 | 0.951±0.020 | 1.013±0.038 | 1.064±0.043 |
| F(vertical), engine | 0.876 | 0.907 | 0.951 | 0.987 | 0.995 | 0.997 |
| `<cos alpha>` of the MC cone | 0.809 | 0.873 | 0.940 | 0.982 | 0.993 | 0.995 |

| nue | 0.2 GeV | 0.28 GeV | 0.45 GeV | 1.1 GeV | 2.0 GeV | 2.8 GeV |
|---|---|---|---|---|---|---|
| plain `<F>_Omega`, MC | 0.986±0.017 | 1.012±0.021 | 1.021±0.019 | 0.984±0.018 | 0.990±0.018 | 0.990±0.034 |
| plain `<F>_Omega`, MC true | 1.022±0.022 | 1.026±0.021 | 1.007±0.021 | 0.990±0.022 | 0.977±0.022 | 0.981±0.061 |
| plain `<F>_Omega`, engine | 1.065 | 1.055 | 1.033 | 1.007 | 1.002 | 1.001 |
| `<F>_cos`, MC | 0.858±0.002 | 0.900±0.003 | 0.951±0.003 | 0.982±0.002 | 0.987±0.003 | 0.997±0.005 |
| `<F>_cos`, MC true | 0.870±0.005 | 0.905±0.006 | 0.951±0.009 | 0.969±0.014 | 0.973±0.025 | 0.986±0.032 |
| `<F>_cos`, engine | 0.921 | 0.938 | 0.966 | 0.992 | 0.998 | 0.999 |
| `sum s0 / sum t0`, MC | 0.861±0.002 | 0.900±0.002 | 0.950±0.003 | 0.983±0.002 | 0.992±0.003 | 0.995±0.003 |
| `sum s0 / sum t0`, MC true | 0.871±0.005 | 0.904±0.005 | 0.952±0.009 | 0.968±0.013 | 0.974±0.021 | 0.983±0.033 |
| F(vertical), MC | 0.807±0.009 | 0.887±0.017 | 0.939±0.016 | 0.998±0.018 | 0.983±0.027 | 0.966±0.021 |
| F(vertical), MC true | 0.832±0.015 | 0.891±0.020 | 0.950±0.018 | 0.993±0.048 | 1.020±0.067 | 0.991±0.085 |
| F(vertical), engine | 0.870 | 0.901 | 0.949 | 0.989 | 0.997 | 0.998 |
| `<cos alpha>` of the MC cone | 0.774 | 0.843 | 0.932 | 0.983 | 0.994 | 0.996 |

* **Nothing is manufactured, in either measure.**  The plain average sits
  *at or below* the engine's (numu 1.038 ± 0.016 vs 1.049 at 0.28 GeV; nue
  1.012 vs 1.055), and the number of neutrinos through the ground is a
  **deficit of 8.1% (numu) / 10.0% (nue) at 0.28 GeV**, against the engine's
  5.9% / 6.2%.  The MC deficit equals the engine's own number for a cone
  scaled x1.25 (`<F>_cos = 0.921` at 0.3 GeV, `PHASE1_RESULTS.md` §8.2) --
  the MC cone *is* that much wider (§4.4), and its backward tail, which the
  engine truncates at 89 deg, is emitted upward and never reaches the ground.
* **The flat-atmosphere argument holds in the MC** once written as
  `F(vertical) = <max(cos alpha, 0)>` (a primary cannot come from below the
  horizon): per `E_nu` band, near-vertical primaries, all channels,
  F(vertical) = 0.926 / 0.973 / 0.986 against `<max(cos alpha,0)>` = 0.922 /
  0.965 / 0.987 at 0.3-0.5 / 0.5-1 / 1-3 GeV (numu; nue 0.931 / 0.952 /
  0.986 against 0.908 / 0.964 / 0.987), and 0.739 vs 0.719 at 0.1-0.3 GeV,
  where 7% of the neutrinos are emitted backward (`alpha > 90 deg`) and
  `<cos alpha>` alone (0.693) would miss by 6%.  So the MC's vertical
  suppression is the engine's `<cos alpha>` mechanism with the wider cone:
  0.873 at 0.28 GeV against the engine kernel's 0.914.
* **Caveat (i) of `PHASE1_RESULTS.md` §8.2** ("the ~9% all-sky suppression
  away from the horizon ... a direct comparison against a published 3D MC in
  both measures is still owed"): a genuine 3D cascade with no cone model at
  all gives the suppression, and it is *larger* (F(vertical) 0.90 / 0.89 for
  numu / nue at 0.28 GeV, `<max(cos alpha,0)>`); in the plain measure the
  hemisphere is +3.8% (numu) / +1.2% (nue), against Bartol's +3%, Honda's
  -2% and the engine's +4.9%, and in the ground-crossing measure it is the
  8-10% deficit above.

### 4.4 (d) The production cone, channel by channel

Space-angle RMS [deg] between the primary and the neutrino, all generations /
first-interaction only, against the factorised engine's cone **as the engine
evaluates it** (`sin alpha exp(-alpha^2/sigma^2)` on `[0.5, 89] deg`; its
realised RMS is below the nominal `sigma` for wide cones), 10x run, ±0.02-0.4
deg statistical:

| numu | 0.22 GeV | 0.28 GeV | 0.45 GeV | 0.71 GeV | 1.12 GeV | 1.78 GeV |
|---|---|---|---|---|---|---|
| direct pi: MC / gen-1 / engine (`sigma_pi`) | 32.2 / 31.3 / 26.8 | 24.9 / 24.5 / 21.8 | 16.6 / 16.4 / 14.4 | 11.8 / 11.7 / 9.4 | 8.4 / 8.3 / 6.2 | 5.7 / 5.7 / 4.1 |
| MC / engine | **1.20** | **1.14** | **1.15** | **1.25** | **1.35** | **1.41** |
| mu decay: MC / gen-1 / engine (`sigma_mu_numu`) | 40.6 / 39.3 / 31.9 | 33.3 / 32.4 / 28.2 | 21.7 / 21.6 / 20.3 | 15.1 / 15.1 / 14.1 | 10.9 / 10.8 / 9.6 | 7.7 / 7.7 / 6.3 |
| MC / engine | **1.27** | **1.18** | **1.07** | **1.07** | **1.13** | **1.23** |
| K: MC / engine (`sigma_k`) | 79.0 / 52.7 | 57.4 / 48.9 | 38.8 / 38.3 | 26.3 / 27.0 | 18.1 / 18.6 | 12.3 / 12.7 |
| channel weight pi / mu / K | 0.53/0.45/0.02 | 0.52/0.45/0.02 | 0.52/0.44/0.04 | 0.51/0.42/0.07 | 0.50/0.40/0.11 | 0.48/0.36/0.16 |
| first-interaction fraction | 0.74 | 0.76 | 0.79 | 0.81 | 0.82 | 0.82 |

| nue | 0.22 GeV | 0.28 GeV | 0.45 GeV | 0.71 GeV | 1.12 GeV | 1.78 GeV |
|---|---|---|---|---|---|---|
| mu decay: MC / gen-1 / engine (`sigma_mu_nue`) | 41.5 / 40.6 / 29.9 | 33.7 / 33.1 / 26.0 | 21.4 / 21.2 / 18.5 | 14.7 / 14.6 / 12.7 | 10.6 / 10.6 / 8.6 | 7.6 / 7.6 / 5.6 |
| MC / engine | **1.39** | **1.30** | **1.16** | **1.15** | **1.23** | **1.37** |
| K (K_e3): MC / engine (`sigma_k`) | 47.8 / 52.7 | 38.3 / 48.9 | 27.0 / 38.3 | 18.3 / 27.0 | 13.1 / 18.6 | 9.4 / 12.7 |
| channel weight mu / K | 0.99/0.01 | 0.99/0.01 | 0.99/0.01 | 0.98/0.02 | 0.97/0.03 | 0.96/0.04 |

(antineutrinos: the same within a few %; `results/m2b/x10.txt`.)

* **The MC cone is wider than the engine's in the two channels that matter**:
  direct pi **+14 to +41%**, muon decay **+7 to +27% (numu) and +15 to +39%
  (nue)**, smallest at 0.4-0.7 GeV and growing toward both ends.  The kaon
  cone agrees within 3% above 0.4 GeV for numu; for nue the engine applies the
  `K -> mu nu` cone to `K_e3`, which the MC finds 22-32% narrower (a 1-4%
  channel).
* **It is not multi-generation accumulation.**  70-85% of the neutrinos come
  from mesons made in the primary's first interaction, and the
  first-interaction-only cone is only 0.2-3.5% narrower than the full one.
* **It is not the hadronic model either.**  Per interaction, the pion
  production angle `arcsin(p_T/p)` of the chromo pools agrees with the
  engine's own generator moment rows (`m_spliced_v2.npz`) to 0.90-1.03 at
  every projectile energy checked (9-1000 GeV) and pion energy (0.3-5 GeV)
  (`diag_m2b.py ptpool`).
* **It is the engine's pooling of those moments.**  `sigma_pi(E)` is built
  from `fokker_planck_3d.load_theta2` -> `angular_kernel.pool_moments_by_energy`,
  which averages `<theta^2>` over all (projectile energy, `x_L`) cells at fixed
  secondary energy with weight `dndx` -- a density in `x_L` on log-spaced
  `x_L` bins, without the bin width and without any projectile spectrum.  At
  fixed `E_pi` the high-`E_proj` rows sit at tiny `x_L`, where `dndx` is
  largest, so **rows above 1 TeV carry 85% (0.3-0.5 GeV pions) to 99% (2-5
  GeV) of the weight**, and those rows have the smallest angles.  Weighting by
  pions per cell (`dndx x dx_L`) and an `E_proj^-1.7` interaction spectrum
  instead gives a pion production angle **x1.25 / x1.38 / x1.36 / x1.47**
  wider at `E_pi` = 0.3-0.5 / 0.5-1 / 1-2 / 2-5 GeV (`m_spliced_v2`; x1.29 -
  x1.48 for `m_piminus_v2`) -- the size and the energy trend of the MC/engine
  ratio above.  This is a construction defect of the factorised engine's
  input, classified, not fixed here (`tools/mceq3d` is read-only for this
  milestone).  `PHASE1_RESULTS.md` §8.2 noted the dominance of the highest
  rows but not its effect on `sigma`.
* **What it does to the observables** (§4.2-4.3): the wider cone gives the
  deeper sub-GeV deficit away from the horizon (`<max(cos alpha,0)>` 0.87 vs
  0.91) and the larger ground-crossing deficit (8-10% vs 6%); the engine's own
  sensitivity would then predict a +12-15% larger horizon excess at 0.3 GeV,
  which the MC does **not** show beyond its 9-12% errors -- the question §4.2
  leaves open for the cutoff run's statistics.  The measured kernel
  `K(alpha)` per species / channel / band (and its exact moments on the
  engine's energy grid) is exported in `results/m2b/x10_summary.npz`
  (`cone_hist`, `cone_moments`) for the engine to adopt directly.

### 4.5 (e) The injection-patch saturation gate (G8)

* **Direct measurement** (`hinj`, the injection-point distance of every cap
  hit).  In the 55-deg run (`N_eff ~ 1000` near-horizon 3D entries) **no
  3D-hit weight comes from injection points more than 22 deg from the site**
  in any `cos Z` or energy band (0.1-3 GeV).  In the 10x run (`N_eff ~
  1.2e5` near-horizon entries, `cos Z < 0.2`) the farthest contributing
  injection point is 25 deg from the site (cap edge 10 deg plus up to ~1700 km
  of primary path and neutrino flight), the weight beyond 25 deg is 1.4e-5 of
  the horizontal bin and there is none beyond 30 deg.  The 40-deg patch is
  saturated with a >= 15-deg margin: its truncation is < 3e-5 of the
  near-horizon flux (95% CL), far inside the gate's 1%.
* **Flux comparison** (paired jackknife, 2.5e5 showers each):
  `Phi_3D(55)/Phi_3D(40)` hemisphere-summed = 1.011 ± 0.034 / 0.951 ± 0.043 /
  1.091 ± 0.048 at 0.1-0.3 / 0.3-1 / 1-3 GeV, vertical (`cos Z > 0.7`)
  1.020 ± 0.023 / 1.035 ± 0.029 / 0.995 ± 0.060 -- the patch-area
  normalisation (`A` 1.82x larger at 55 deg) is right.  Per near-horizon bin
  the ratio scatters 0.4-1.9 with 10-70% errors: at 2.5e5 showers that
  comparison cannot reach the gate's 1%, which is why the direct measurement
  above is the gate.  Scored-shower fractions confirm it: 3.57% at 40 deg,
  1.95% at 55 deg = 3.57%/1.82.

### 4.6 Absolute normalisation (a check of the §3.2 fix)

`Phi_3D` of the 10x run against HKKM2014 Kamioka (`tools/mceq3d/honda_kam.npz`,
azimuth-averaged, same bins): **2.1-2.4 at 0.3 GeV, 1.7-2.0 at 0.5 GeV,
1.3-1.5 at 1 GeV, 1.03-1.15 at 2 GeV, 0.91-1.06 at 3 GeV, 0.78-0.99 at 5 GeV**
(well-measured bins, all species, `cos Z` 0.4-1).  Exactly the expected
pattern: no geomagnetic cutoff here (Kamioka's suppresses the sub-GeV flux by
~2x), agreement where neither the cutoff nor the 1000 GeV/nucleon truncation
matters, and the truncation showing at 5 GeV.  Before the fix every number
would have been 90-300x too large.

### 4.7 Summary of milestone 2b

The 3D engine runs end to end with honest errors; the patch is saturated; the
nested caps are consistent; the absolute flux is right.  The production
geometry agrees with the factorised engine's to ~5% in shape, but the
factorised cone is 7-41% too narrow because of how it pools its generator
moments, and the MC's ground-crossing deficit is correspondingly larger
(8-10% vs 6% at 0.3 GeV).  The East-West table at 3% costs ~1.4-1.9e3 core-h
per site with 12-fold azimuthal reuse (before the cutoff re-allocation), and
the correlated in-run 1D reference buys nothing near the horizon.

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
6. **The scored-shower fraction (measured, §4.1) sets the tracing cost.**  At
   `theta_inj = 40 deg` and the 10-deg cap, **3.6% of injected showers score**
   (>= 1 neutrino crossing the cap in 3D, or a 1D crossing in the cap) --
   1.9% in the 1-3 GeV/nucleon stratum, 5.8% at 3-10, 6.8-7.1% above 10 GeV --
   and **0.40%** have a 3D hit in the East-West target region (`cos Z < 0.3`,
   0.3-2 GeV; 0.03% / 0.5% / 1.5% / 2.1% / 2.5% / 3.0% by stratum, 10x run).  A lazy
   back-trace of scored showers only (the `MILESTONE3A_NOTES.md` design) costs
   `0.036 x N_az` traces per injected primary, i.e. **0.43 per primary at
   N_az = 12**, against 12 for eager tracing of every copy of every primary;
   an East-West production run that keeps only E-W-region showers needs 0.048
   per primary.
   With the cutoff in, strata 0-1 (84% of today's showers, 33% of the CPU)
   feed the E-W region at 0.03-0.5% per shower and are mostly forbidden at
   Kamioka, so item 5's re-allocation comes first.
7. **Budget the 3D tally, not the in-run ratio, near the horizon.**  The
   measured 3D/1D correlation in the near-horizon bins is weak (§4.1:
   `rho` between -0.3 and +0.5), because a near-horizontal 3D neutrino and its primary's
   collinear entry land in different `cos Z` bins at sub-GeV, so the
   correlated in-run reference does not reduce the variance there; the ratio
   error is the 3D and 1D errors in quadrature.  The 1D reference is
   azimuth-symmetric up to the site cutoff and smooth in zenith, so a
   dedicated collinear run (or MCEq scaled by the rung-B ratio) supplies it
   far more cheaply than the 3D run.

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

Result (2026-09-10): **16 passed** (`test_sky.py`, `test_pool.py`), plus the
34 milestone-1 tests unchanged.

Added 2026-10-08 (`test_sky.py`, 5 tests):

| test | what it pins down |
|---|---|
| `::test_cone_scorer_bins_the_primary_neutrino_angle` | synthetic: neutrinos at known angles to a known primary land in the right (species, band, channel, generation, `alpha`) cell with exact moments; the shower counters and the injection-angle histogram see exactly the cap hits they should; `to_dict`/`from_dict`/`merge` are additive |
| `::test_cone_of_a_decaying_pion_matches_two_body_kinematics` | physics gate of the cone tally and of the parent tagging: a non-interacting 2 GeV pi+ (stub backend) -- the direct nu_mu's measured `<cos alpha>` and `<alpha^2>` match the exact two-body integral (`cos a = (c+b)/(1+bc)`, `c` uniform) in every 10/decade bin to 5 sigma, the per-band counts match, the nu_mu is tagged `pi` only and the muon-decay neutrinos `mu` only |
| `::test_stratum_weights_do_not_depend_on_the_chunking` | regression for the §3.2 normalisation bug: the summed primary weight of the strata equals `pi A_patch int J dE` to 2% for a single chunk and for 10 chunks |
| `::test_sky_run_resumes_from_task_files` | a job with one shard file deleted re-runs only that task and reproduces the merged totals bit for bit (`s1`, `t1`, cone moments, `hinj`, counters); skipped without `MC3D_POOL_CACHE` |
| `::test_stratified_jackknife_matches_the_true_variance` | the §4 error machinery: on synthetic task sums from two strata of unequal size, weight and spread, the delete-a-group jackknife (`analyse_sky.JackRun`/`jack`) reproduces the analytic stratified variance to 8% (mean of 300 realisations), and its per-stratum pieces add up to it |

Full suite on 2026-10-08 (including the parallel milestone-3a agent's
`test_cutoff.py`): **68 passed** (`$PY -m pytest tools/mc3d -q`, 12.5 min
with `MC3D_POOL_CACHE` set, so the slow tests run).
---

## 7. Reproducing

Raw outputs go to local disk (`/home/pigrange/mc3d_work`, never `/tmp`, which
is wiped on reboot); only the summary tables are copied to `results/m2b/`.

```sh
export PY=/cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc14-opt/bin/python3.12
export R=/afs/cern.ch/work/p/pigrange/daemonflux W=/home/pigrange/mc3d_work
export PYTHONPATH=$W/pylib:$R/.venv/lib/python3.12/site-packages:$R/src:\
$R/tools/mceq3d:$R/tools/mc3d
export MC3D_POOL_CACHE=$W/pools OMP_NUM_THREADS=1
M=$W/runs/m2b; mkdir -p $M && cd $M     # chromo writes tables.dat into the cwd

# 1. the event-pool cache (708 pools, 321 MB, 90 s on 40 cores; resumable)
$PY $R/tools/mc3d/prebuild_pools.py --cache $W/pools --emax 1.1e4 \
    --npool 2000 --nproc 30

# 2. rung B and its A2 partner at 1000 GeV (2026-09-10; outputs not preserved)
$PY $R/tools/mc3d/closure.py --energies 20 100 --nshower 200000 --nproc 30 \
    --mode chromo --cache $W/pools --shard 500 --out $M/closure_B.npz
$PY $R/tools/mc3d/closure.py --energies 1000 --nshower 50000 --nproc 30 \
    --mode chromo --cache $W/pools --shard 200 --out $M/closure_B1000.npz
$PY $R/tools/mc3d/closure.py --energies 1000 --nshower 50000 --nproc 30 \
    --mode kinematic --shard 200 --out $M/closure_A2_1000.npz
$PY $R/tools/mc3d/summarize_closure.py $M/closure_B.npz $M/closure_B1000.npz

# 3. the database and generator diagnosis (sec. 2.4, 2.6) and the input of
#    the factorised engine's cone (sec. 4.4); seconds to a minute each
for w in db splice dpm sigma ptpool; do $PY $R/tools/mc3d/diag_m2b.py $w; done

# 4. the 3D pilot, the patch gate and the collinear twin (sec. 4); every job
#    writes one shard per task under $M/sky/<tag>_tasks and resumes if
#    re-launched with the same arguments
SK=$R/tools/mc3d/sky.py
C="--nproc 30 --cache $W/pools --chunk 6000 --nstrata 6 --outdir $M/sky"
$PY $SK --nshower 1500000 $C --tag p3d --seed 20260910             # 7.6 min
$PY $SK --nshower 250000  $C --tag g40 --seed 777 --theta-inj 40    # 1.4 min
$PY $SK --nshower 250000  $C --tag g55 --seed 777 --theta-inj 55    # 1.4 min
$PY $SK --nshower 1500000 $C --tag p1d --seed 20260910 --collinear  # 8.1 min
#    the 10x statistics pair (chunk x10 keeps ~1700 tasks)
C10="--nproc 30 --cache $W/pools --chunk 60000 --nstrata 6 --outdir $M/sky"
$PY $SK --nshower 15000000 $C10 --tag p3dx --seed 20261008
$PY $SK --nshower 15000000 $C10 --tag p1dx --seed 20261008 --collinear

# 5. the sec. 4 tables (stratified jackknife over the shard files)
cd $R/tools/mc3d
$PY analyse_sky.py $M/sky --m2b --out-txt $M/reports/pilot.txt \
    --out-npz $M/reports/pilot_summary.npz
$PY analyse_sky.py $M/sky --m2b --tag3 p3dx --tag1 p1dx --gate \
    --out-txt $M/reports/x10.txt --out-npz $M/reports/x10_summary.npz

# 6. tests (the slow ones need MC3D_POOL_CACHE)
$PY -m pytest $R/tools/mc3d -q
```
