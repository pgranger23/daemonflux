# Phase 2 — `tools/mc3d`: a lightweight genuine 3D atmospheric-neutrino Monte Carlo

**Purpose.** Produce per-site 3D/1D correction tables
`R_s(E, cos Z, phi) = Phi_3D_s / Phi_1D_s` for `daemonflux`, and — first — settle
the **East–West species pattern** that the factorised engine in `tools/mceq3d`
cannot reproduce (`PHASE1_RESULTS.md` §5, §8.1, §8.4: Honda orders the four
species' W/E amplitude `nu_e > antinu_mu > nu_mu > antinu_e` = 4.74/3.81/2.51/2.12
at 0.5 GeV/87°, the factorised engine compresses all four onto ~2.6–3.3, and the
Super-K East–West data prefer Honda's ordering at ≈2σ).  The factorised engine
localises the failure in the **cone average**, not in the hadronic input or the
cutoff: the un-cone-averaged `(W/E)_G` reproduces Honda's ordering.  A genuine MC
has no cone average, so it either reproduces Honda's ordering or it does not —
and either answer is decisive.

This file is the design.  It is meant to be complete enough that another agent
can implement any module in it from this text alone.  Milestone 1 (geometry,
atmosphere, primary sampler, shower driver, scorer, and the collinear/B=0 closure
against MCEq) is **implemented and measured**; see §9 and `MILESTONE1_RESULTS.md`.

---

## 1. Scope, and what is deliberately *not* done

| in | out |
|---|---|
| p/He/CNO/MgSi/Fe primaries, superposition to nucleons | full nucleus–air fragmentation |
| pi, K, K_L, K_S, Lambda, nucleons, muons | EM cascade, charm (irrelevant below 10^5 GeV) |
| full decay kinematics, muon polarisation | radiative corrections, muon spin precession beyond the boost |
| IGRF-13 cutoff by back-tracing at the injection point | time-dependent field, magnetospheric storms |
| continuous muon dE/dx | stochastic muon energy loss (irrelevant below ~100 GeV) |
| curved-Earth spherical geometry | geodetic (ellipsoidal) Earth — see §2.1 |
| 3D/1D **ratio** tables | an absolute flux prediction competing with Honda |

The deliverable is a **ratio**.  Everything that cancels in the ratio (the
absolute primary normalisation, the overall hadronic yield scale) may be
approximate; everything that does not (the geometry, the cutoff, the decay
kinematics, the muon bending) must be right.

---

## 2. Geometry

### 2.1 Earth

Spherical, `R_E = 6371 km` (the mean radius; `constants.R_EARTH_KM`).  Honda uses
the equatorial `R_e = 6378.18 km`; `tools/mceq3d` (`offaxis_mc`,
`geomag_backtrace`) uses 6371 km, and we keep that so the back-tracer can be
reused with no convention change.

**Geodetic corrections do not matter here**, for three independent reasons, all
recorded in `geometry.py`'s module docstring:

1. *Slant depth.*  The near-horizontal path through a shell of thickness `H`
   scales as `sqrt(2 R H)`; the WGS84 flattening `f = 1/298.26` changes `R` by at
   most 0.33% between equator and pole, i.e. the horizon slant depth by 0.17%.
   That is an order below the statistical target (§7) and below the ±12% NA61
   cone systematic that already dominates the horizon shape
   (`PHASE1_RESULTS.md` §8.2).
2. *Local vertical.*  The geodetic–geocentric latitude difference peaks at 0.19°
   at 45°.  It relabels zenith/azimuth by <0.2°, against 5°-wide zenith bins and
   a 5.4°/lifetime muon bend.
3. *Geomagnetic cutoff.*  `ppigrf.igrf_gc` is evaluated in **geocentric**
   coordinates and `geomag_backtrace` already launches from a spherical surface;
   using the same sphere keeps the MC and the existing cutoff machinery
   consistent, which matters more than absolute geodetic fidelity.

### 2.2 Spheres (Honda's scheme, astro-ph/0404457 §IV)

| sphere | Honda | here | why |
|---|---|---|---|
| injection | `R_E + 100 km` | `R_E + 100 km` (`constants.H_INJ_KM`) | above essentially all the mass (`X_v(100 km) = 0.00128 g/cm2`) |
| atmosphere top | — | `R_E + 112.8 km` (CORSIKA `X_v -> 0`) | the cascade only exists below this |
| simulation | `R_E + 3000 km` | `R_E + 3000 km` (bookkeeping only) | bounds the "escaped and re-entered" population |
| escape | `10 R_E` | `10 R_E` (`geomag_backtrace.r_escape = 25 R_E` default; use 10) | the back-trace termination radius |

Honda showed a 300 km simulation sphere is already good to ~1%; escaping
secondaries are counted (`n_escape` in `shower.run_shower`) and, for milestone 1
(down-going primaries), are <0.1%.  Milestone 3 adds explicit re-entry for
particles that leave the atmosphere upward: in a straight line a ray leaving a
sphere outward never returns, so re-entry exists **only** for charged particles
bent by the field, and only muons bend appreciably (§4.3).  The muon path is
integrated in the field anyway, so re-entry falls out of the existing stepper
with no extra machinery — it just needs the `escape` branch replaced by
"continue the field integration outside the atmosphere until `r > R_sim` or the
particle comes back below `R_top`".

### 2.3 Atmosphere

CORSIKA US-standard (Keilhauer `BK_USStd`), the **same parameterisation MCEq
uses**, re-implemented analytically in `atmosphere.py` (five layers,
`X_v(h) = a_i + b_i exp(-h/c_i)` for `i<4`, linear above 100 km).  Verified
bit-identical against `MCEq.geometry.density_profiles.CorsikaAtmosphere` at every
altitude (`test_geometry.py::test_density_matches_mceq`).

**Why not reuse `offaxis_mc`'s machinery.**  `offaxis_mc._rho_of_h` tabulates
MCEq's density model and `offaxis_mc.slant_depth_table` builds a 90×140 bilinear
`X_slant(h, psi)` table for the *deterministic* cone integral.  A Monte Carlo
needs the **inverse** operation — "advance this particle until it has traversed
`dX` g/cm²" — millions of times at arbitrary positions and directions, and needs
it invertible so the interaction/decay competition is unbiased.  A bilinear table
is neither invertible nor accurate near the limb (it is capped at `1e7` to keep
the interpolation finite).  `atmosphere.grammage` /
`atmosphere.advance_grammage` are an adaptive Simpson integrator and its
bisection inverse along an arbitrary straight ray; they reproduce the vertical
column to `2.6e-5` relative.  `test_geometry.py` gates them against
`offaxis_mc.slant_depth_table` at a grid of `(h, psi)`.

Scalar fast paths (`atmosphere.rho_s`, `hscale_s`) exist because the numpy
versions cost ~2 µs per call through `asarray/searchsorted/where` and dominated
the whole run; they are gated against the array versions in the tests.

### 2.4 Detector: an Earth-concentric shell with a Honda-style virtual cap

The detector is the **ground sphere** `|r| = R_E`; a neutrino counts when the
straight ray from its production point crosses that sphere inside a cap of
angular radius `theta_D` around the site (`geometry.DetectorCap`).  Honda uses
`theta_D = 10°` (1117 km radius, ~1/6 of his previous belt).

Scoring is in the **local frame of the crossing point**, not of the site centre
(they differ by up to `theta_D`), because that is what "the flux at a detector at
that latitude" means.

**Finite-size bias and its removal.**  A 10° cap averages the cutoff and the
field over ±1117 km, which is exactly the systematic Honda warns about.  We
remove it by **nested caps**: `theta_D in {2.5, 5, 7.5, 10}°` scored
*simultaneously on the same neutrino* (`scoring.CapScorer` loops over the caps),
so the four estimates are nested subsets and maximally correlated.  Any observable
`O(theta_D)` is then extrapolated to `Omega_D -> 0` linearly in the cap solid
angle `Omega_D = 2 pi (1 - cos theta_D)`: for a field smooth across the cap the
first-order term averages out over the ring, so the leading bias is
`O(theta_D^2) ~ Omega_D` and a straight line in `Omega_D` is the right form.  The
extrapolation is done on the **ratio** `R_s`, not on the flux, so the
primary-normalisation cancels first.

Cap areas: `A(theta_D) = 2 pi R_E^2 (1 - cos theta_D)`; `A(10°) = 3.87e16 cm²`.
Weights carry `1/A` so the tally is a flux per cm²; the scorer stores the plain
count so either the "crossing the shell" (`cos`-weighted) or the "specific
intensity" normalisation can be formed downstream — the two differ by the
`<cos psi>` measure factor that `PHASE1_RESULTS.md` §8.2 shows is the whole
"conservation excess" story, so **both must be reported**.

### 2.5 Azimuthal reuse — the central variance reduction

Everything in the problem is azimuthally symmetric about the site's local
vertical **except the geomagnetic field**: the Earth is a sphere, the atmosphere
depends only on altitude, and the primary flux is isotropic.  Therefore a shower
simulated at position/direction `(P, u)` relative to the site can be **rotated by
any angle `psi` about the site vertical** and remains a valid shower — with two
consequences:

1. The cutoff acceptance must be re-evaluated for the rotated primary direction.
   This is a **per-copy scalar weight** `G(R, n_p^(psi))`, not a re-simulation.
2. The muon bending is *not* symmetric, because `B` does not rotate with the
   copy.  §4.3 shows the fix: hadrons bend by ≤0.05° and can be treated with
   `B = 0`; only the muon matters, and the muon's bend is a rotation of its
   direction by `Delta_theta = 0.3 B c tau / m_mu = 5.4°` per lifetime about the
   local `B`, essentially independent of energy and of `dE/dx`
   (`PHASE1_RESULTS.md` §4, gated to 1e-6 in
   `muon_segment_mc.test_bend_is_energy_independent`).  So: transport the muon
   with `B = 0`, record the decay point and the accumulated
   `Phi = int ds / r_gyro`, then per copy rotate the muon's direction at decay by
   `Phi` about the *copy's own* `B`.  The displacement error is
   `(1 - cos Delta_theta) L <~ 0.004 L`, i.e. <50 m over a 10 km muon path —
   negligible against the 1117 km cap.  The exact fallback (re-run the muon
   segment per copy) costs `N_az x` the muon transport only, and should be run
   once at low statistics as a gate on the approximation.

With `N_az = 12`–24 copies this multiplies the azimuthal statistics by 12–24 at
~zero cost, and it is what makes the East–West measurement affordable (§7).

---

## 3. Primaries

### 3.1 Species and spectrum

Five groups, as in `crflux`: p (A=1, Z=1), He (4,2), CNO (14,7), Mg–Si (25,12),
Fe (56,26).  Spectrum from `crflux.models`: `GlobalSplineFitBeta` (GSF, the
`tools/mceq3d` delivered choice) or `HillasGaisser2012:H3a`.
`PHASE1_RESULTS.md` §8.3 measured that the **primary choice moves the East–West
amplitude by ±10–15%** — an order of magnitude more than it moves H/V — so the
primary is a first-class systematic and both must be run.

**Superposition** (Honda does the same, astro-ph/0404457 §IV): a nucleus of mass
`A` and charge `Z` at total energy `E` is `Z` protons + `A-Z` neutrons at
`E/A`, all launched from the same point in the same direction, and — critically —
**the rigidity used in the cutoff test is that of the whole nucleus**,
`R = p_total / (Z e) = (A/Z) (p_nucleon)`, i.e. Honda's "all nucleons carried by
the cosmic-ray nuclei are treated as protons with double rigidity".  The A/Z
split is already handled the same way in `mceq3d_flux._nucleon_split`.

### 3.2 Energy sampling and weights

Sample `ln E_nucleon` uniformly on `[E_1, E_2] = [1, 10^4] GeV`
(`primaries.sample_energy`), i.e. `dN/dE ∝ E^-1`.  Weight

```
w_E = J_s(E) * E * ln(E_2/E_1) / n_samples
```

with `J_s` the per-nucleon differential intensity of species `s`
[cm^-2 s^-1 sr^-1 GeV^-1].

**Stratify.**  A single `E^-1` proposal is *not* optimal: measured against the
natural `E^-2.7`, it gains only ~2× in the number of entries feeding a 0.5 GeV
horizon bin (because the cutoff already removes the low-energy end) while adding
weight spread; it gains a lot for `E_nu > 3 GeV`.  The right scheme is
**stratified sampling in `ln E`** with per-stratum allocation tuned by a pilot
run (milestone 2b): 8 strata of one decade... one half-decade each, each with its
own sample count `n_k` chosen to equalise the statistical error in the *target*
bins.  This is the single most cost-effective knob after azimuthal reuse.

### 3.3 Direction and position on the injection sphere

Position uniform on the injection sphere; direction from the **cosine (Lambert)
inward** law `p(mu) = 2 mu`, `mu = -u . n_hat` (`geometry.sample_injection`).
The rate normalisation for an isotropic external intensity `Phi` is

```
N_dot = Phi * pi * A_inj          (A_inj = the sampled injection area)
```

so each sample carries `w_geom = pi * A_inj / n_samples` and the total weight is
`w = w_E * w_geom * w_cutoff * w_solar`.

**Injection patch.**  Sampling the whole `4 pi R_inj^2` wastes ~99% of the CPU on
showers that cannot reach the cap.  Restrict the injection to a cap of angular
radius `theta_inj` about the site.  `theta_inj` must cover the horizontal
displacement between the primary's entry point and the neutrino's arrival point:
~100–300 km for down-going, up to ~1500 km at the exact horizon
(`PHASE1_RESULTS.md` §6.2 quotes 617 km up the ray at `h_prod = 30 km`, and
neutrinos travel further).  **Use `theta_inj = theta_D + 30°` and verify
saturation empirically** by re-running one configuration with `theta_D + 45°` and
checking the near-horizon bins move by <1%.  This is a required gate, not an
option: an under-sized patch silently truncates exactly the horizon bins the
project exists to measure.

### 3.4 Geomagnetic cutoff as an acceptance weight

For each sampled primary, back-trace from the **injection point** with its own
rigidity and charge and ask whether the trajectory reaches the escape sphere
without re-entering (`geomag_backtrace.backtrace_vec` / `is_allowed`; this is
literally Honda's test).  Two implementations, both supported:

* **On the fly, binary (`w_cutoff in {0,1}`).**  One `backtrace_vec` call per
  primary.  Cost: a *forbidden* trajectory returns in ~30 RK4 steps, an *allowed*
  one in ~900 (`PHASE1_RESULTS.md` §6.2).  At ~45 µs per field call and ~16 ms
  fixed overhead per batch, back-tracing must be **batched** (thousands of
  primaries at once) or it dominates.  This is the physically exact route and the
  only one that is correct in the penumbra.
* **Cached admittance map.**  Precompute, per site and per azimuth-copy geometry,
  `A(R, zenith, azimuth)` on the injection sphere by `cutoff_from_states`
  (already parallel, already bisecting to 0.1 GV) and use `w_cutoff = A`.  This
  is what makes the **azimuthal reuse** cheap: the map is a function of the
  primary direction in the *geomagnetic* frame, so all `N_az` copies read the
  same map at different arguments.  `PHASE1_RESULTS.md` §5 warns that the
  penumbra is *not* well described by a smooth `sigma_lnR`; the measured width is
  `sigma_lnR = 0` (a sharp cutoff at `R_U`), so a binary map at `R_U` is a good
  approximation — but the **allowed islands** below `R_U` are real
  (`diag_admittance_fine.py` found `[9.63, 10.13] GV` at the Kamioka vertical),
  so the map must store the *admittance fraction over the rigidity bin*, not just
  `R_U`.  `scan_upper_cutoff(..., return_admittance=True)` already returns the
  0/1 ladder needed to build that.

**Recommendation.** Milestone 3 uses the on-the-fly binary test for the reference
run (it is exact and its cost is bounded by batching), and the cached admittance
map for the azimuth copies, with a gate comparing the two at a few hundred
directions.

### 3.5 Solar modulation

A weight, not a re-sampling: `w_solar = J_s(E; phi_1) / J_s(E; phi_0)` using the
force-field parameter of the chosen `crflux` model.  This lets one shower set
serve several solar epochs.  (Honda tabulates solar-min and solar-max
separately; `honda1d/kam-ally-*-solmin.d` in the scratchpad are the solar-min
tables we compare against.)

---

## 4. The cascade

### 4.1 Transport

`shower.run_shower` is a stack-based tracker.  For each particle:

* **hadrons** (`shower.transport_hadron`): sample `dX = -ln(u) * lambda_int(E)`
  and a decay path `s_dec = -ln(u) * (p/m) c tau`; advance with
  `atmosphere.advance_grammage` up to `min(s_dec, s_boundary)`; whichever comes
  first wins.  `lambda_int = <A> m_p / sigma_inel(E)` with `<A> = 14.6568` —
  byte-for-byte MCEq's `ParticleManager.inverse_interaction_length`, so the
  interaction depth cannot differ from MCEq's by construction.
* **muons** (`shower.transport_muon`): stepped, with continuous `dE/dx` (PDG
  total stopping power in air, the same table as
  `muon_segment_mc.dedx_air`), a midpoint energy correction per step, proper-time
  accumulation for the decay, and — in 3D mode — the Lorentz rotation of the
  direction per step.  The step is adaptive on the **energy loss**
  (`_MU_DX_MAX = 30 g/cm²` per step) rather than on the density scale height, so
  a muon high above the atmosphere crosses tens of km in one step; that alone is
  a 5× speed-up.
* **boundaries**: hitting the ground absorbs; leaving the top escapes (§2.2).

> **Bug to never re-introduce.**  `atmosphere.path_to_exit` must take only the
> **near** root of the ground sphere.  Taking the far root when the near one is
> `<= 0` (which happens the instant a particle sits exactly on the ground) sends
> it straight through the Earth and out the other side.  In the milestone-1
> closure every muon then "decayed" after thousands of steps *inside the Earth*
> and the muon-decay neutrino yield came out ~2× too high.

### 4.2 Interactions — two backends

`interactions.py` provides one interface with two implementations:

* **`MCEqYieldBackend`** — samples secondaries from MCEq's **own inclusive yield
  matrices**, exported once by `build_tables.py` into
  `mceq_tables_SIBYLL23D.npz` (3.6 MB, loads in ms in every worker).  Inclusive
  and therefore collinear in energy: it carries no `p_T`.  Its purpose is the
  **closure gate**: running the cascade off MCEq's own physics makes a
  disagreement with MCEq unambiguously *our* error.  This is not a toy — Honda's
  own 3D calculation uses an inclusive interaction code for exactly the same
  reason (astro-ph/0404457 §III: "the inclusive interaction code is only valid
  for the calculation of a time averaged quantity, such as the fluxes of
  atmospheric neutrinos").
* **`ChromoBackend`** — a real generator through `chromo`: SIBYLL-2.3d above
  `e_switch = 80 GeV`, DPMJET-III-19.3 below (SIBYLL is not valid below ~10 GeV
  lab, and DPMJET-III-19.3 is what MCEq's shipped `lext_dpm193` database splices
  in below 80 GeV — Honda's own generator family).  Returns the exclusive final
  state with full 3-momenta; this is the backend the 3D physics needs.
  Smoke-tested against the MCEq tables (p–N at 100 GeV, per interaction):
  `pi+` 3.393 vs 3.399, `pi-` 2.860 vs 2.971, `K+` 0.257 vs 0.279,
  `Lambda` 0.197 vs 0.201, `p` 1.323 vs 1.404 — the residual is the
  nitrogen-only target versus MCEq's N/O air mix, which milestone 2a fixes by
  drawing the target from the air composition (78.1% N₂, 20.9% O₂, 0.9% Ar
  → `<A> = 14.66`).

  > **Performance requirement.**  Calling `model(1)` once per interaction costs
  > **73 events/s/core** for SIBYLL-2.3d at 100 GeV; the same generator called
  > as `model(3000)` runs at **1068 events/s/core** (`<n_final> = 18.4`).  The
  > 15× is pure per-call setup overhead.  Milestone 2a must therefore keep a
  > **pre-generated event pool per (projectile, energy bin)**, refilled in
  > batches and drawn from at random, rather than one generator call per
  > interaction.  Re-using an event more than once introduces a correlation that
  > is harmless for inclusive means provided the pool is large (>=10^4 per bin)
  > and re-drawn independently of the cascade history.

**Two conventions that bite** (both cost a factor ~2 if got wrong, both were hit
in milestone 1):

1. MCEq stores `M[i,j] = dN/dx_i * dlnE`, so the *number* of daughters in bin `i`
   per parent event is `N[i,j] = M[i,j] * E_i / E_j`.
2. MCEq's energy grid is **kinetic** energy (`etot_grid = e_grid + m`), while the
   cascade tracks **total** energy.  Every table lookup must convert both ways.
3. MCEq keeps **helicity-resolved muon states** (`(-13,-1)` `pi_mu+_l`,
   `(-13,+1)` `pi_mu+_r`, and for kaons additionally `(-13,0)`), so
   `D.get_matrix((211,0), (-13,0))` is **empty**.  A naive export loses every
   muon, hence half the `nu_mu` and essentially all the `nu_e`.  `build_tables`
   sums over the helicity index for muon daughters, which produces the
   *unpolarised* yield — so the matching MCEq reference for that rung must be run
   with `config.muon_helicity_dependence = False`.

**Superposition for nuclei** is applied *before* the cascade (§3.1): the nucleus
never enters `interact()`; `A` independent nucleons at `E/A` do, sharing one
injection point, one direction and one cutoff weight.  Nuclear fragmentation and
the difference between a real `A`–air collision and `A` superposed nucleon–air
collisions is a known ~few-% systematic on the absolute flux which largely
cancels in the 3D/1D ratio (both sides use the same primaries).

### 4.3 Charged-particle bending — how much matters

The in-flight bend over one decay length is
`Delta_theta = L / r_gyro = 0.3 B c tau / m`, **independent of energy** (the
`gamma` in `L` cancels the `p` in `r_gyro`) — the result `muon_segment_mc`
established analytically and gated to `1e-6`.  With `B = 0.5 G = 5e-5 T`:

| particle | `c tau` | `Delta_theta` per lifetime |
|---|---|---|
| pi± | 7.80 m | **0.048°** |
| K± | 3.71 m | **0.0064°** |
| mu± | 659 m | **5.4°** |

Against a production cone whose RMS space angle is 38.9/14.6/8.9/4.1° at
0.3/0.5/1/3 GeV (`PHASE1_RESULTS.md` §4), **pion and kaon bending are ≤0.3% of
the relevant angular scale and are dropped** (they are switched on only as an
A/B).  The muon bend is the whole charge-dependent East–West mechanism and is
carried exactly in the stepper.

### 4.4 Decays

`decays.py`, full kinematics: `pi -> mu nu`; `K± -> mu nu / pi pi0 / 3pi /
pi0 l nu`; `K_L -> pi l nu / 3 pi0 / pi+ pi- pi0`; `K_S -> pi pi`;
`Lambda -> p pi- / n pi0`; `mu -> e nu nu` with the **polarised Michel**
spectra (`n(x)`, `A(x)` identical to `muon_segment_mc.michel_n/michel_asym`).

Muon polarisation is carried exactly: the muon from `pi/K -> mu nu` has helicity
`-1` (`mu+`) / `+1` (`mu-`) in the parent rest frame; the spin 4-vector is
boosted to the lab and projected back into the muon rest frame to give the
longitudinal `P_L` (`decays.muon_polarisation`).  The two Michel neutrinos are
sampled **independently from their exact marginals** — unbiased for every
single-particle (linear) observable, which is all the scorer tallies.

Known approximations, all contained and all listed so they can be upgraded:

* three-body decays use **flat Dalitz phase space** (constant matrix element).
  For `K_l3` this misstates the neutrino spectrum by a few per cent; `K_l3` is
  <8% of kaon decays and kaons make <15% of the sub-GeV neutrinos, so the effect
  on the total is <0.2%.  Upgrading means putting the measured `f_+(t)` into
  `decays.three_body`.
* muons from three-body kaon decays are treated as unpolarised.
* `Lambda` decay is two-body with the correct BRs but no `Lambda` polarisation.

### 4.5 Tracking thresholds and what is dropped

A particle is dropped when it *cannot* make a neutrino above `e_nu_min` (default
0.1 GeV) — the bounds are exact, not heuristics:

| species | threshold on total energy | bound used |
|---|---|---|
| pi± | `e_nu_min / (1 - m_mu²/m_pi²) = e_nu_min / 0.4270` | `E_nu <= (E+p)/2 (1-r) <= E (1-r)` |
| K±, K_L, K_S | `e_nu_min / 0.9544` | same with `r_K` |
| mu± | `max(m_mu, e_nu_min)` | `E_nu <= (E+p)/2 <= E` |
| nucleons | `m_N + e_nu_min/0.4270` | via the pion daughter |
| Lambda | `m_Lambda + e_nu_min/0.4270` | via the pion daughter |

> The muon threshold was originally `e_nu_min + m_mu`, which **throws away muons
> that can make a neutrino above threshold** (a 0.15 GeV muon can give a 0.13 GeV
> neutrino).  Fixed.

Dropped outright: `pi0`, photons, electrons, charm.  The EM component feeds
nothing back into the hadronic cascade in MCEq either, so dropping it keeps the
two calculations comparable; charm matters only above ~10^5 GeV.

**Lambda is *not* dropped.**  MCEq's SIBYLL-2.3d tables give 0.20 `Lambda` +
0.03 `Lambda-bar` per p–air interaction at 89 GeV, and `Lambda -> p pi-` (63.9%)
feeds the pion cascade.  Dropping it cost ~2% of the neutrino yield in the
milestone-1 closure — measurable, so it is tracked.  `Sigma±` have zero yield in
these tables.

**No thinning.**  Thinning is a `>10^6 GeV` technique; at `10^1`–`10^4 GeV`
primaries the particle count per shower is `O(100)` above threshold and the
thresholds above already remove ~90% of it.

---

## 5. Scoring

### 5.1 The 3D tally

`scoring.CapScorer`: for each neutrino above `e_nu_min`, for each nested cap, the
ray is intersected with the ground sphere; if the crossing is inside the cap the
entry goes into `(cap, species, E, cos Z, azimuth)` with weight `w / A_cap`.  Sum
of weights and sum of squared weights are both accumulated, so every number
carries a statistical error, and both are additive across multiprocessing shards.

Binning: Honda's own grid — `E` log-spaced 20 bins/decade from 0.1 GeV,
`cos Z` in 20 bins of 0.1, azimuth in 12 bins of 30°, with the **`az_compass =
180 - az_Honda` convention** (`PHASE1_RESULTS.md` §5; his azimuth is
counterclockwise from South, arXiv:1102.2688 §II).  Anything odd about the E–W
axis flips sign without this mapping.

### 5.2 The 1D reference on the same showers

This is what makes the ratio cheap.  In the **same run**, each shower is *also*
scored collinearly: the neutrino is assigned the **primary's** direction, no
bending is applied, and the cutoff is evaluated at the site line-of-sight rather
than at the injection point.  That is exactly the 1D approximation, evaluated on
the same primaries with the same hadronic sample, so `R_s = Phi_3D / Phi_1D` has
strongly correlated numerator and denominator and its variance is far below the
naive sum in quadrature.  (`shower.Config(collinear=True)` is the same code
path — the 1D run is the `sigma_theta -> 0, B -> 0` limit of the 3D one, not a
different program.)

Two normalisations must both be reported, because `PHASE1_RESULTS.md` §8.2 shows
the entire "conservation excess" controversy is a choice of measure: the plain
solid-angle average `<F>_Omega` and the ground-crossing average
`<F>_cos = int_down dOmega cos psi Phi`.  The scorer keeps the information for
both.

---

## 6. Validation gates

In order; each is a hard gate, not a plot.

* **G1 — closure (`sigma_theta -> 0`, `B -> 0`).**  Collinear mode, `B = 0`,
  vertical column, protons.  The neutrino yield per primary must reproduce MCEq's
  1D solution for the same primary and hadronic model to a few %.  **Done, §9.**
  Three rungs isolate the pieces: A1 (MCEq yields + MCEq decay tables) tests
  geometry/transport/competition/`dE/dx`; A2 (MCEq yields + our decay kinematics)
  tests the decay module; B (chromo + our kinematics) *measures* the generator
  difference rather than testing it.
* **G2 — geometry.**  `atmosphere.grammage` vs `offaxis_mc.slant_depth_table` on
  a `(h, psi)` grid; cap acceptance vs analytic solid angle; the vertical column
  = 1033.805 g/cm².
* **G3 — sampler.**  The weighted primary sample reproduces `crflux`'s input
  spectrum per species to the MC error; the yield sampler reproduces the MCEq
  yield column bin-by-bin (measured: 0.3% over the whole `x_L` range).
* **G4 — decay kinematics.**  `pi -> mu nu` gives a flat lab energy spectrum for
  the neutrino on `[0, (1-r) E_pi]`; the Michel spectra integrate to the known
  `n(x)` and reproduce `<x>`; the polarised asymmetry has the right sign against
  MCEq's helicity-resolved decay matrices.
* **G5 — azimuth-averaged flux vs Honda.**  `honda_kam.npz` (HKKM2014) on
  Honda's grid; the 3D/1D ratio compared to Honda's own 3D/1D where he publishes
  both.
* **G6 — the physics target: East–West per species.**  W/E amplitude for all four
  species at 87/81/76° and 0.3–2 GeV, against (i) Honda, (ii) the factorised
  engine (`diag_ew_charge_fourier.py`'s first-two-harmonic observable — the
  `max/min` observable is blind to the sign of a coherent shift and must not be
  used), (iii) the Super-K asymmetry through `diag_sk_ew.py`.
* **G7 — nested-cap extrapolation.**  The `Omega_D -> 0` extrapolation must be
  consistent (linear) across the four caps; if it is not, the cap is too large or
  the site interpolation of the cutoff is failing.
* **G8 — injection-patch saturation.**  `theta_inj + 15°` must move the horizon
  bins by <1%.

---

## 7. Cost and variance

**Measured throughput (milestone 1, `MCEqYieldBackend`, collinear, 40 cores):**
2713 showers/s at 20 GeV, ~900/s at 100 GeV — i.e. **23 showers/s/core at
100 GeV**.  For the chromo backend, SIBYLL-2.3d at 100 GeV runs at
**1068 events/s/core when batched** (73/s if called one event at a time — see
the performance note in §4.2), and a 100 GeV shower needs ~6 hadronic
interactions above threshold, so the generator allows ~180 showers/s/core: the
**transport stays the bottleneck**.  Budget **10–20 showers/s/core** for full
3D, i.e. **1.4–2.9e6 showers/hour on 40 cores**.  If that turns out to be the
binding constraint, the next 3–5× is in the transport inner loop (it is pure
Python scalar arithmetic; numba or a vectorised multi-particle stepper are both
straightforward).

**Entries per primary in a hard bin.**  Take the near-horizon 0.5 GeV `nu_mu`
bin: `Phi ≈ 2 (cm² s sr GeV)^-1`, `Delta E = 0.0575 GeV` (20/decade),
`Delta Omega = 0.05 x 30° = 0.0262 sr`, `A_cap(10°) = 3.87e16 cm²` gives a
physical rate of `1.2e14 nu/s` into that bin.  The nucleon flux above 5 GeV is
`~0.25 cm^-2 s^-1 sr^-1`; through an injection patch of `theta_inj = 40°`
(`A_patch = 6.2e19 cm²`, times `pi` sr) that is `4.8e19 nucleons/s`.  So

```
entries per injected nucleon (natural spectrum)  ~=  2.4e-6
```

For **1e4 entries (1%)** that is `4.2e9` nucleons; for **1.1e3 entries (3%)**,
`4.6e8`.  Then:

| lever | factor | note |
|---|---|---|
| azimuthal reuse, `N_az = 12` | ÷12 | §2.5; one shower fills all 12 azimuth bins |
| correlated 3D/1D ratio | ÷~4 in entries | numerator and denominator share the primary sample |
| coarser grid (`Delta cos Z = 0.1`, 10 E-bins/decade) | ÷4 | for the survey table only |
| stratified `ln E` allocation | ÷~2 | §3.2; larger at high `E_nu` |

**Bottom line.** A 3%-level near-horizon East–West table at 0.3–1 GeV needs
`~1e7` showers per site → `~1e3` core-hours → **~25 h on 40 cores**.  A 1%-level
version, or the full fine grid, is `1e4`–`1e5` core-hours — consistent with the
audit's Phase-2 estimate.  The *survey* table (all zeniths and energies at 5%) is
`~300` core-hours, **~8 h on 40 cores**.  These are estimates from the flux
argument above; **milestone 2b must replace them with a measured entries/primary
from a pilot run** before any long job is launched.

---

## 8. Work breakdown (each ≤3 h of compute, each leaves resumable state on disk)

State lives in `tools/mc3d/` (code, plan, results) and
`scratchpad/mc3d/` (npz shards, logs).  Every shard is a self-contained
`YieldScorer`/`CapScorer` `to_dict()` npz that merges additively, so an
interrupted run resumes by re-launching the missing shards.

| # | milestone | compute | deliverable |
|---|---|---|---|
| **1** | geometry, atmosphere, decays, MCEq-yield backend, shower driver, scorer; **collinear/B=0 closure vs MCEq** | ~1 h | **DONE** — `MILESTONE1_RESULTS.md`, `closure_A{1,2}.npz` |
| **2a** | `ChromoBackend` wired into the shower driver; rung B (chromo vs MCEq in collinear mode) at 20/100/1000 GeV; per-species yield comparison | ~2 h | the measured generator difference, `closure_B.npz` |
| **2b** | 3D mode end-to-end at low statistics: `CapScorer`, nested caps, injection patch, **pilot measurement of entries/primary** and of the 3D/1D correlation; G8 saturation gate | ~2 h | `pilot.npz`, a *measured* cost table replacing §7's estimate |
| **3a** | cutoff acceptance: batched `backtrace_vec` at the injection point + cached admittance map; gate map vs on-the-fly at ~500 directions | ~3 h (map build) | `admittance_<site>_<date>.npz` |
| **3b** | azimuthal reuse (`N_az` copies, per-copy cutoff weight, rotated muon bend); gate against the exact per-copy muon re-transport at low statistics | ~1 h | the reuse machinery + its gate |
| **4** | survey run: full sky, 5% target, GSF primary, SIBYLL-2.3d | ~8 h on 40 cores (split into 3 h chunks by seed) | `R_s(E, cosZ, phi)` survey table |
| **5** | **the East–West run**: near-horizon, 0.3–2 GeV, four species, 2–3% | ~25 h on 40 cores (chunked) | the answer to §0; G6 |
| **6** | systematics: H3a vs GSF primary, EPOS/DPMJET vs SIBYLL, solar epoch, nested-cap extrapolation, patch size | ~6 h | the error budget on `R_s` |
| **7** | integration: ship `R_s` as `daemonsplines_<site>_3d_<rev>.pkl`, applied multiplicatively in `_FluxEntry` mirroring `set_geomagnetic_model`/`_apply_geomag` | ~0 | the Phase-3 hand-off |

---

## 9. Milestone 1 status

Implemented: `constants.py`, `atmosphere.py`, `geometry.py`, `lorentz.py`,
`decays.py`, `interactions.py` (both backends), `shower.py`, `scoring.py`,
`build_tables.py`, `closure.py`, and `test_*.py`.

The closure result, the three bugs it caught (Earth-crossing `path_to_exit`,
kinetic-vs-total energy at the MCEq table lookup, the helicity-resolved muon
decay matrices), and the per-species/per-energy agreement table are in
`MILESTONE1_RESULTS.md`.

**Not yet implemented** (milestone 2 onwards): `CapScorer` is written but not
exercised end-to-end; the chromo backend is written and smoke-tested but not
gated; the injection-patch sampler, the cutoff acceptance weight, the azimuthal
reuse and the muon-bend rotation are designed here but not coded.
