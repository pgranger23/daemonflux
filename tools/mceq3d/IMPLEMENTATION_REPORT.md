# A 3D / low-energy extension of daemonflux — complete implementation report

**For:** the daemonflux author.
**What this is:** a full, deliberately self-critical account of a research-grade
3D extension built on top of daemonflux/MCEq — every method, every validation
(with its plot), and an exhaustive list of the simplifications and outright
"cheats" so you can judge exactly what is trustworthy and what is scaffolding.

**Scope honesty up front.** One small, optional change is wired into the
*installed package* (`src/daemonflux/geomagnetic.py`, a directional admittance
factor on the 1D flux). **Everything else lives in `tools/mceq3d/` as a separate
research layer** — it is an architecture and a set of validated physics
ingredients for a deterministic 3D-MCEq, *not* a drop-in replacement for the
published spline model. Where I write "flux", the absolute normalization is only
meaningful for the MCEq-backed engine (§5.6); the parametrized engines produce
*ratios/shapes* only.

All code is on the fork `pgranger23/daemonflux`, branch `3d-extension`.
Inventory: **28 modules, 21 test files, 118 passing offline tests, 26 validation
plots, 4 companion docs.** Tooling: Black + flake8 clean, NumPy docstrings.

---

## 1. Executive summary

* **Goal:** let daemonflux be used below ~2 GeV, where the 1D assumption breaks
  and geomagnetic + geometric 3D effects matter.
* **What was built:** (i) regeneration of the production *angle* that 1D MCEq
  integrates away, validated against NA61; (ii) angular transport (Fokker-Planck
  + P_N); (iii) a first-principles geomagnetic cutoff (full IGRF back-tracing) and
  a directional flux Φ(E, zenith, azimuth); (iv) muon bending; (v) a spherical
  streaming PDE operator; (vi) the cascade coupled to the curved atmosphere; and
  (vii) a high-statistics kernel set produced on a cluster.
* **Headline physics findings (with the magnitudes):**
  * The dominant low-energy 3D effect is **geomagnetic** — at Kamioka, 1 GeV, the
    flux is suppressed to ~0.5 of the bare value vertically and shows an
    **East–West ratio ≈ 2** near the horizon.
  * The **production-angle 3D effect on the conventional flux is small** (~1–2%).
  * **Muon bending** gives a coherent **±3° charge-dependent East–West shift**
    sub-GeV (large for ν vs ν̄ separately, ~0.3° for the sum).
  * The **sec θ horizon enhancement is a high-energy effect** (≈5× near the
    horizon at TeV, finite thanks to curvature); sub-GeV the conventional flux is
    **near-isotropic**. This is captured by 1D-per-direction with curved columns;
    the genuinely-3D inter-direction residual is the small (~1–2%) part.
* **Validated against Honda HKKM2014** (§8): the directional observables match the
  authoritative 3D tables — East–West amplitude at Kamioka **2.1 (Honda) vs 2.4
  (this work)** at 1 GeV with the correct sub-GeV peak and >10 GeV vanishing, and
  the sec θ horizon enhancement **2.19 vs 2.17** at 100 GeV.
* **A trustable *absolute, full-sky* directional engine** (`mceq3d_flux.py`,
  §5.11): returns the absolute Φ(E, cosθ, azimuth) for all four flavours over the
  **whole sky across 0.1–100 GeV** as **selectable 1D base × cascade-correct
  geomagnetic factor** (no `x_eff`), with **up-going handled by a global far-side
  geomagnetic treatment**. The focus is the **100 MeV–100 GeV** band (oscillation
  physics); validation now uses a **dense low-energy grid**. With the **data-anchored
  daemonflux base** (`base_model="daemonflux"`) the **absolute** νμ flux matches
  Honda at Kamioka to **~10 % for E ≳ 1 GeV** (to 100 GeV), but the two bases
  **cross over** below ~1 GeV: daemonflux over-predicts toward 0.1 GeV (~2× there)
  while raw MCEq runs ~25–30 % low over 0.3–10 GeV — and Honda sits between them, so
  **0.3–1 GeV carries an irreducible ~1.8× model spread** that must be taken as a
  systematic. It **reproduces the up/down asymmetry** independent of the base, and
  the **flavour ratio (νe/νμ) matches Honda to 0.7–4%**. (Validation numbers are
  **energy-matched** — an earlier nearest-index comparison overstated the raw-MCEq
  agreement; §5.11.)
* **Net message:** the value for sub-GeV daemonflux is overwhelmingly in the
  **geomagnetic layer** (production-ready as an admittance factor) plus muon
  bending; the deterministic-3D cascade machinery is validated as an architecture
  and quantifies that the remaining geometric/angular 3D corrections are small.

---

## 1b. Response to an independent review

An external reviewer assessed this work; the substantive items have been addressed
in code, with the remainder documented honestly:

1. **Per-nucleus rigidity** (flagged *High*, ~20% sub-GeV horizon) — **fixed**. The
   geomagnetic cut now splits the primary nucleon flux into free protons (A/Z=1,
   R=E) and bound nucleons (He/CNO/Fe, A/Z≈2, R≈2E) using MCEq's own p,n fluxes
   (free p = p−n by isospin), replacing the all-protons-R=E treatment, so bound
   nucleons (cut at the higher rigidity R≈2E) are no longer over-suppressed near the
   sub-GeV horizon. (§5.11.)
2. **Muon-bending idealizations** — **improved**. (a) the fixed 15 km path is
   replaced by the curved **zenith-dependent slant** (`path_length_km`: ~15 km
   vertical → hundreds of km near the horizon); (b) the bending is now
   **restricted to the muon-decay ν_μ channel** (weighted by
   `muon_decay_numu_fraction = f/(1+f)`) instead of bleeding into every decay
   neutrino. (§5.5.)
3. **Atmosphere** — **clarified + extended**. The production engine (`mceq3d_flux`)
   already uses MCEq's realistic **CORSIKA US-Standard layered** atmosphere, *not*
   the isothermal exponential — that profile is only in the `spherical_cascade`
   research demo (where the TeV-horizon overshoot lives). Seasonal/site tracking is
   now exposed: `MCEq3DFlux(atmosphere=("MSIS00", (site, month)))`. (§5.11.)
4. **K± validation** — **done**. The NA61 K± data (HEPData `ins1397003`, p+C at
   31 GeV/c) is fetched with `hepdata-cli` and compared in `validate_na61_kaon.py`:
   UrQMD reproduces the kaon `⟨p_T⟩` scale and momentum dependence, ~10–20 % harder
   than NA61 — a small, documented production-angle effect. (§5.2.)
5. **Nitrogen-only / proton-only targets** — small, documented effects; the
   forward-looking "when these bite" analysis is §9b.

**Absolute normalization — now data-anchored (the structural improvement).** The
reviewer's central concern was that the absolute scale rode on the SIBYLL23D/H3a
hadronic model. With daemonflux's muon-calibrated spline data now available, the
engine offers `base_model="daemonflux"`, which multiplies the **muon-calibrated 1D
flux** by the validated geomagnetic factor. Energy-matched against Honda this moves
the 1 GeV νμ normalization from **~0.73 (raw MCEq) to ~1.1**, i.e. into the
daemonflux↔Honda model spread — the deficit is no longer a methodological gap but
the irreducible sub-GeV flux uncertainty. Fixing this also surfaced and corrected
an **energy-binning artifact** in `--validate` (nearest-index vs log-log
interpolation), which had flattered the raw-MCEq numbers (§5.11).

**Migration (the reviewer's closing question).** The full-IGRF back-tracer is now
**selectable through the package API**: `GeomagneticModel(cutoff_source=…)` accepts
the back-traced cutoff via `geomag_backtrace.cutoff_source(lat, lon, date)`, so it
is no longer a standalone-only utility (see §1c for the now-fast cached path).

## 1c. Response to the second-round review

The second review confirmed the major fixes and flagged four remaining soft spots;
all are now addressed in code:

1. **Cutoff-map cache (the verdict's "remaining engineering task")** — **done**.
   `geomag_backtrace.cached_cutoff_source(lat, lon, date)` batches the full
   (zenith×azimuth) sky map **once** with `cutoff_map`, caches it to disk
   (`cutoff_cache/*.npz`), and returns a callable that **bilinearly interpolates**
   it in microseconds (azimuth-periodic). The first-principles IGRF cutoff is thus
   fast enough to be the standard path: `GeomagneticModel(cutoff_source=
   cached_cutoff_source(...))`. (§5.4.)
2. **Sub-0.3 GeV volatility → an explicit systematic** — **done**.
   `base_comparison.model_envelope` returns a **data-grounded fractional systematic
   (half log-spread of the two bases)** and a convenience central. The band is
   **±37 % at 0.1 GeV, ±31 % at 0.5 GeV, ±21 % at 1 GeV, ±10 % at 10 GeV, ±4 % at
   100 GeV** (`base_comparison.png`), delivered as a number to carry rather than a
   caveat. Honest scope (see audit, §1e): the bases **bracket Honda only below
   ~1 GeV** (geometric-mean central ~Honda there); above ~1 GeV both lie below Honda
   so daemonflux is the central, and the band is an intra-framework floor (it does
   not by itself span Honda/Bartol/FLUKA). (§5.11.)
3. **Muon-bending vertical-muon approximation** — **fixed**. `coherent_shift_deg`
   uses the **actual muon direction** (`muon_velocity_enu`) and the **full local
   field** (`local_field_enu`, IGRF degree-13 via ppigrf: Kamioka ≈ (−0.04, 0.30,
   −0.37) G, inclination ~49°), so vertical *and* both horizontal components and the
   off-vertical geometry all enter. The old vertical/north-only result is recovered
   as the special case. (§5.5.)
4. **Primary composition ⟨A/Z⟩** — **fixed**. The fixed 2.0 is replaced by the
   **nucleon-flux-weighted ⟨A/Z⟩ over the real composition** (He/CNO/Si 2.0, Fe
   2.077), computed per energy from crflux (≈2.005, rising slightly with E as Fe
   grows). The ~0.2–0.5 % Fe sub-component is no longer dropped. (§5.11.)

The remaining **SIBYLL soft-pion turn-on** (~64 % shape agreement at the √s floor)
is a generator-threshold artifact confined to the vanishing-yield region; the bulk
(x_L ≥ 0.005) matches to 1–3 % and the angular moments are unaffected, so those
kernels are fine for the *angular spread* but not for standalone absolute yields —
documented, not "fixable" without a higher-energy generator (§5.1).

## 1d. Response to the third-round review

The third review flagged two subtle spots and two suggested cross-checks; all are
now resolved, with the cross-checks added as reproducible scripts.

1. **Zenith-independence of `G_s` — measured and bounded** (the reviewer's
   "zenith-insensitivity validation"). `geomag_zenith_check.py` recomputes the
   suppression ratio `G_s = Φ_cut/Φ_full` in raw MCEq at zeniths from vertical to
   ~84° (cosθ=0.1) for fixed cutoffs. **`G_s` is zenith-independent to ≤2.1 %** at
   the worst point (0.3 GeV, R_c=11 GV) and **≤0.4 % above 1 GeV** — because the
   slant-depth shower effects the reviewer noted (extra meson decay, muon energy
   loss) act on numerator *and* denominator and **cancel in the ratio**. So the
   vertical precomputation is safe at the ~1–2 % level; for studies that want even
   that removed, `solve(zenith_dependent_geomag=True)` now recomputes `G_s` at every
   zenith. (`geomag_zenith_check.png`, §5.11.)
2. **Horizon grid exposed** (the verdict's gating item). `horizon_grid()` returns a
   full-sky cosθ grid **refined near cosθ=0** (most points inside |cosθ|<0.2),
   pass-through to `solve(cos_zeniths=...)`, so the sharp sec θ horizon region is
   sampled finely without wasting points in the isotropic bulk. (§5.11.)
3. **Extreme-latitude spot-check** (verification #1). `latitude_check.py` runs the
   central estimate + systematic from a high-cutoff equatorial site to the polar
   limit. **R_c falls monotonically equator→pole (17.2 → 9.0 → 1.8 → 0.8 GV)**, the
   flux rises smoothly with no discontinuity at the no-cutoff polar limit, and the
   **fractional systematic is site-robust (0 % spread** — the cutoff cancels in the
   base ratio). The composition ⟨A/Z⟩ is site-independent by construction.
   (`latitude_check.png`.)

With these, the two open technical spots are closed and the horizon grid is
exposed — the conditions the verdict set for production-readiness.

---

## 1e. Internal audit (code + analysis)

A self-critical pass over the whole engine, looking for numerical/physics corners
beyond the review points. Three items found; the validated Kamioka results are
**unchanged** by all of them (verified by re-running `--validate`).

1. **Rigidity-grid clamping (fixed).** `solve()` interpolates the suppression
   `G_s` on an `R_c` grid that previously had a fixed floor of 2 GV, while the
   cutoff map can return `R_c` as low as 0.5 GV. Low-cutoff directions (polar, or
   near-horizon at mid-latitude) were therefore clamped to `G_s(2 GV)`, applying a
   **spurious ~3.5 % suppression at 0.3 GeV** (less above). The grid is now built to
   **span the actual cutoff map** (`rc_grid` from `R_c^min` to `R_c^max`), so no
   clamping occurs at any site. Kamioka (R_c≈11) was inside the old grid, hence the
   headline numbers are identical.
2. **Vectorised `G_s` interpolation (optimisation).** The per-energy `np.interp`
   loop in `solve()` is replaced by a single vectorised linear interpolation along
   the rigidity axis (`_interp_rc`), bit-identical results, ~100× fewer Python ops
   in the inner triple loop (helps fine horizon grids).
3. **Systematic-band scope clarified (analysis).** The earlier text said the two
   bases "bracket Honda" and the geometric-mean central "tracks Honda to ~15 % over
   0.3–100 GeV". On inspection this is only true **below ~1 GeV** (where daemonflux
   > Honda > MCEq). **Above ~1 GeV both bases lie below Honda** (daemonflux ≈0.91,
   MCEq ≈0.73), so they do *not* bracket it and the geometric mean runs ~10–18 %
   **below** Honda — there the data-anchored **daemonflux base is the better
   central**. The half-log-spread band remains a valid *intra-framework* systematic
   but is a **floor**, not a full inter-calculation envelope (Honda sits above it at
   multi-GeV). All three reports and the `base_comparison` docstrings were corrected
   accordingly; the recommendation is now: **daemonflux central for E≳0.5 GeV**,
   geometric-mean only in the sub-GeV crossover, band as the model systematic.

**Bounded limitation (documented, not changed):** the back-traced cutoff is capped
at `r_hi` (default 20 GV), so for very-high-cutoff **equatorial** sites the
near-horizon-East cutoff (which can exceed 20 GV) is under-estimated, slightly
under-suppressing those directions. Kamioka's maximum (~14 GV) is well inside the
cap, so the validated results are unaffected; raise `r_hi` for equatorial sites.

No other correctness issues were found: the per-nucleus free/bound split, the
cascade-correct `G_s` ratio, the far-side up-going geometry, the species
reconstruction, and `interp_flux` (which sorts cosθ and wraps azimuth) all check out.

**Performance (`profile_3d.py`, single core).** One MCEq cascade solve ≈ **1.5 s**
(the atomic unit). A full directional `solve()` over a 6×8 = 48-direction sky grid
≈ **6 min**, i.e. **~240× one MCEq solve** (~14× the equivalent 1D MCEq flux at the
same zeniths). Breakdown: cutoff back-tracing **~315 s (~85 %, dominant)**, `G_s`
response ~21 s (13 cascade solves), 1D base ~26 s (MCEq base) or **~10 ms
(daemonflux base)**. Both heavy terms are **reusable**: the cutoff map is a
one-time per-site precompute (`cached_cutoff_source`; warm re-evaluation
**~0.3 µs/direction**), and `G_s` is site- and zenith-independent (computable once).
So the engine is a one-time per-site precompute then near-instant evaluation —
vs the CPU-weeks of a full 3D Monte-Carlo.

**Caching wired in (`solve(use_cache=True)`).** Both heavy ingredients are now
memoised to disk (`flux_cache/`): the **site-independent** `G_s` (keyed by
model/primary/atmosphere/e_min + rigidity grid) and the **per-site** cutoff map
(keyed by site/date/grid). The first evaluation of a site pays the full cost;
repeat calls — and *any* other site, for `G_s` — load in **milliseconds**
(measured: a warm `solve()` of a 2×2 grid is **4.7 ms** vs **60 s** cold, a
**>10³–10⁴× speed-up** on re-use), matching the default path to ~0.1 % (the cached
path uses a finer fixed rigidity grid); cold-vs-warm cache is byte-identical. Off by
default so the validated path is untouched.

---

## 2. Motivation and scope

daemonflux is a muon-calibrated, data-driven spline model of the **1D** inclusive
atmospheric flux. Below ~2 GeV the physics it cannot represent is (a) the
geomagnetic rigidity cutoff and East–West effect, (b) the curved-atmosphere
geometry, and (c) the spread of production/decay directions. You noted that the
1D method makes assumptions that are violated at low energy but are *removable*
(performance simplifications, not fundamental). This work removes them one by one
in a separable way and measures how much each matters.

---

## 3. Background: what is "1D"

MCEq solves the cascade equations in **one dimension** (column depth) per zenith:
its interaction matrices are inclusive yields `dN/dx_L` with the transverse
momentum **integrated out**, and each arrival direction is an independent vertical
column. daemonflux is a spline-evaluation + muon-calibration layer on top of MCEq
1D fluxes. So the genuinely new ingredient for 3D is the **angular** information
(production p_T → angle) and the **directional** physics (geomagnetic, curved
geometry). That is exactly what this layer adds.

---

## 4. Architecture

```
 primaries ──[geomagnetic cutoff]──► cascade ──[angular production]──► transport ──► Φ(E,θ,φ)
              §5.4 geomagnetic.py        §5.6 mceq3d_*        §5.1-5.3          §5.7-5.9
              geomag_backtrace.py        cascade engines      kernels+FP/P_N    directional/spherical
```

| layer | module(s) | status |
|---|---|---|
| production-angle kernels | `kernel_regeneration`, `splice_kernels`, `angular_kernel` | validated vs NA61 |
| NA61 validation | `validate_na61` | ~10% (pions) |
| angular transport | `fokker_planck_3d`, `sn_transport` | FP shown adequate |
| geomagnetic | `geomagnetic.py` (pkg), `geomag_backtrace` | back-traced, lit-checked |
| muon bending | `muon_bending` | analytic, validated |
| cascade engines | `mceq3d_solver` (parametrized), `mceq3d_production` (MCEq) | l=0 ≡ MCEq exactly |
| directional flux | `directional_flux`, `unified_3d_flux` | Φ(E,θ,φ) ratios |
| **absolute 3D engine** | **`mceq3d_flux`** | **Φ(E,θ,φ) absolute, all flavours, Honda-validated** |
| spherical PDE / coupling | `spherical_streaming`, `spherical_cascade` | curvature validated |
| de-risking prototypes | `prototype_3d_cascade`, `prototype_streaming`, `spherical_geometry`, `coupled_3d_flux`, `channel_comparison` | feasibility |

---

## 5. The ingredients

Each subsection: **idea → method → implementation → validation (plot) →
simplifications**.

### 5.1 Production-angle kernels — `kernel_regeneration.py`, `splice_kernels.py`

**Idea.** Recover the production angle `θ = arctan(p_T/p_L)` that 1D MCEq drops,
by re-running the *same* generators MCEq uses (via `chromo`) and histogramming in
`(x_L, θ)` (or computing gridless moments `<θ>`, `<θ²>` per `(E_proj, x_L)`).

**Method/implementation.** Single-interaction inclusive mode; `θ` computed
per-secondary from its own `(x_L, p_T)` with `p_L = sqrt((x_L E)² − m²)`. Two
models spliced (no single model spans the range): UrQMD-3.4 (E_lab ≲ 80 GeV) +
SIBYLL-2.3d (≳ 80 GeV).

**Consistency gate (the key check).** Integrating the regenerated kernel over θ
must reproduce MCEq's *own* stored `dN/dx_L`. Once MCEq's per-bin storage
convention is accounted for (its `hadr_yields` are densities multiplied by the
log-energy bin width `Δ(lnE)`), the regenerated yield matches MCEq's to **~0.3%**
in normalization, with good shape agreement — validating the whole pipeline
against MCEq's database. Plots: `kernel_real_piplus.png`, `angular_smoothness_demo.png`
(why direct-angle binning beats resampling a coarse p_T grid); the pooled
production-angle moments are discussed just below.

**Simplifications.**
* **Nitrogen target only** — justified: the production *angle* is
  target-independent to ~1.3% (C vs N checked).
* **Proton projectile only** via the CLI; meson re-interaction projectiles (π, K
  beams) need a one-line `ChromoSource(projectile=…)` change and are *not* in the
  high-stat set.
* The angular quantities used downstream (`⟨θ⟩`, `⟨θ²⟩`) are normalization-
  independent ratios, so the absolute yield convention never enters them.

**Reading the moments plot (`m_spliced.png`).** The three panels are the
production-angle moments pooled by secondary energy `E_sec`, and they answer a
natural question — *why does `⟨θ⟩` track 1/E but sit off the "0.3" guide, and what
is the odd behaviour of the highest-energy bin?*

* **Panel 1 — `⟨θ⟩(E_sec)` vs the 0.3 guide.** Geometrically `⟨θ⟩ = ⟨p_T⟩/p_L ≈
  ⟨p_T⟩/E_sec`. The dashed line is a *constant* `⟨p_T⟩ = 0.3 GeV` reference (a
  guide, **not a fit**) — it is exactly `∝1/E`. The data follow `1/E` but sit
  *above* the guide and with a slightly shallower slope, because the real `⟨p_T⟩`
  is **not** constant.
* **Panel 2 — why: `⟨p_T⟩` rises.** The effective `⟨p_T⟩ = ⟨θ⟩·p_L` climbs from
  ~0.2 GeV sub-GeV (limited phase space) through ~0.3–0.5 GeV over GeV→TeV — the
  well-known logarithmic growth of `⟨p_T⟩` (scaling violation). This is the
  *physical* reason for the offset, and it is precisely the quantity validated
  against NA61 (§5.2) — not against MCEq, whose 1D kernels carry no angle.
* **Panel 3 — Fokker-Planck `D_θ = ⟨θ²⟩/2`.** Built from the *same* moments, so it
  shows the same trend (`∝E⁻²`, →0 at high E, recovering 1D). Anything downstream
  that reads these moments (the FP solver, `load_theta2`) inherits panels 1/3
  identically — which is why a feature in the moments shows up "in Fokker-Planck"
  too: it is literally the same data.
* **The highest-`E_sec` bins are masked (greyed/dropped).** A secondary at the top
  of the range can only come from the `x_L → 1` corner — the *leading particle*,
  which by kinematics is forward (`p_T → 0`) and has vanishing yield. So those
  bins are leading-particle-biased and statistics-starved; their `⟨p_T⟩` collapses
  artificially toward the 0.3 guide. An earlier low-statistics kernel showed this
  as a visible "snap-back" of the last point; the high-statistics kernels remove
  it, and `pool_moments_by_energy` now additionally flags bins with
  `E_sec > E_proj_max/8` as unreliable so they are greyed in the plot and dropped
  by `load_theta2`. **The bins are physically irrelevant anyway** — there the
  angle is ~0.005°, deep in the 1D limit.

### 5.2 NA61 validation — `validate_na61.py`, `validate_na61_kaon.py`

**Idea/method.** Validate the regenerated production kinematics against real data:
`<p_T>(p_lab)` of charged pions in p+C at 31 GeV/c (HEPData `ins886780`, the
T2K-replica record), applying the experiment's angular acceptance.

**Result.** ~**10%** agreement in `<p_T>` over the measured range
(`validate_na61_pt.png`).

**Kaons — `validate_na61_kaon.py`.** Kaons matter directly for the flux (K→μν is
the dominant ν_μ source above a few GeV and the leading high-energy ν_e source, and
the K⁺/K⁻ asymmetry feeds the ν/ν̄ ratio), so the kaon production p_T is checked the
same way against NA61 K± in p+C at 31 GeV/c (**HEPData `ins1397003`**; K⁺ Tables
23–30, K⁻ Tables 31–37). The tables are fetched reproducibly with **`hepdata-cli`**
(`pip install hepdata-cli`) into `na61_k/`. UrQMD-3.4 **reproduces the kaon `⟨p_T⟩`
scale (0.2–0.6 GeV) and its rise with momentum**, running **~10–20 % harder** than
NA61 (K⁺ ratio 1.09–1.22, K⁻ 1.02–1.34; `validate_na61_kaon_pt.png`). As for pions
this maps to a small over-estimate of the kaon production angle in the K→μν channel;
since that angle is already <1° at the multi-GeV energies where kaons dominate, the
impact on the directional flux is a sub-percent rider on the yield (taken from the
1D base). This closes the one data cross-check the reviewer flagged as open.

**π⁺/π⁻ charge asymmetry (a known, understood limitation).** NA61 shows a real
π⁺/π⁻ difference in `⟨p_T⟩(p)`: π⁻ is *harder* than π⁺ at intermediate-to-high
momentum (up to ~20% at p~2–3 GeV, ~10% averaged), agreeing only below ~1 GeV.
This is the **forward leading-particle charge asymmetry** — the proton's valence
*u* quarks fragment into *leading* π⁺ that go forward and beam-like with low p_T,
pulling `⟨p_T⟩(π⁺)` down, while π⁻ (no valence-leading channel in a proton beam)
is more central and harder. Our sim (UrQMD-3.4, the low-energy model covering this
regime) reproduces the *sign* but **under-predicts the magnitude by ~3×** (~3–4%
vs ~10%) — a known shortfall of low-energy hadronic models in the forward
valence-fragmentation region, not a pipeline bug (both sides use the same
acceptance/binning/weighting). **Impact on the 3D flux is negligible:** it maps to
a small π⁺/π⁻ → ν_μ/ν̄_μ *production-angle* difference, riding on the ~1–2% angular
effect at multi-GeV energies where the angle is already <0.5°; the dominant
ν_μ/ν̄_μ difference is in the *yield*, taken from MCEq in `mceq3d_flux`. It would
matter only for a high-precision charge-resolved directional study.

**Simplifications / limitations.**
* Pions only with a full HEPData fit. **K± was not fitted against HEPData**
  because this environment has no network (SSL). Instead the kaon kinematics are
  cross-checked *internally*: `channel_comparison.py` shows kaon `<p_T>` ≈
  0.4–0.65 GeV vs pion ≈ 0.2–0.46 GeV — kaons harder, consistent with the
  well-established NA61 kaon scale (~0.5 GeV). A proper HEPData K± fit is left as
  future work (needs network).
* UrQMD acceptance applied as a simple θ<420 mrad cut.

### 5.3 Angular transport — `fokker_planck_3d.py`, `sn_transport.py`

**What it does.** MCEq gives the flux vs energy assuming everything is collinear
(1D). This layer adds back the small spread of the neutrino's arrival direction
about the incoming cosmic-ray direction, which 1D discards. The spread comes from
a **transverse-momentum kick at each production vertex** (`θ ≈ p_T/p_L`, the
NA61-validated quantity).

**The parent chain and `N_chain`.** A neutrino is made by a *short chain of
vertices*, each giving one angular kick:

```
primary nucleon ──(interaction)──► meson (π/K) ──(decay)──► ν
                   kick #1: production           kick #2: decay
```

`N_chain` is **how many kick-giving generations feed one neutrino**, and the kicks
add in quadrature: `σ_θ ≈ √(N_chain)·θ₁`. For a conventional ν that is **~2** (a
production kick, the big one at `⟨p_T⟩`~0.3 GeV, plus a decay kick, small at
~0.03 GeV). It is *not* `slant_depth/λ` (the interaction lengths the shower
crosses, ~10–20 and growing toward the horizon): each neutrino traces back through
only those ~2 vertices, and the leading particles stay nearly collinear, so
crossing more atmosphere does **not** keep adding angle. Using `slant/λ` produced
an earlier runaway, zenith-growing spread; `N_chain≈2` makes it finite and
~zenith-independent (set by production *kinematics*, not column *depth*).

**Is 2 reasonable?** Yes. Direct π/K→ν is production+decay = 2; muon-decay ν
(π→μ→ν) is one step longer (≈3) but its extra physics (muon bending) is handled
separately and the conventional ν is direct-decay-dominated. Since `σ_θ∝√N`, 2 vs
3 is a `√(3/2)≈1.22` (~22%) change in an effect that is itself only ~1–2% of the
flux — i.e. ~0.3% on the flux. So 2 is sound and slightly conservative; the exact
value barely matters.

**The two methods, and how they relate.**
* **Fokker-Planck (FP)** — the small-angle limit: the spread is a Gaussian that
  diffuses with depth, `σ_θ = √N·θ₁`. Analytic, fast; valid while the spread ≪ 1
  rad; **unbounded** (would grow past isotropy at large spread).
* **P_N / S_N** — the *full* angular distribution on the sphere: the single-
  production kernel is self-convolved `N` times (a product in Legendre space,
  `c_l → c_l^N`, so `<cos θ> = c₁^N` in closed form). Exact at any angle and
  **bounded**: the spread `arccos<cos θ>` saturates at 90° (isotropy), never
  unphysical.

**P_N is the principled one; FP is what we use** — because Fokker-Planck is
*literally the small-angle limit of P_N*, and in our regime (`N_chain=2`, the
measured `θ₁`) the spread is moderate (~30–35° at ~0.5 GeV, sub-degree above a few
GeV), where the two coincide. P_N's job is to **bound FP's error and prove it
adequate**: they only diverge at an *unphysical* large `N`. A further pure-shape
test (`realshape_sn_spread`: the real measured kernel vs a Gaussian of the *same
variance*, median rel diff ≈ 0) shows the answer depends on the **variance only**,
not the kernel shape — so FP's variance-only input (`<θ²>(E)`) is sufficient and
the full-shape kernel is not needed.

**How to read `sn_transport.png`** — it is **two independent overlap-tests, one
per panel** (the spread `arccos<cos θ>` vs E). In each panel the *two curves lying
on top of each other is the result*; the two panels are separate checks on
*different* kernels, so **read each panel on its own** — their absolute heights are
not meant to match.

* **Left panel — the *method* test** (same input: the production moments). Red
  dashed = Fokker-Planck (small-angle `√N·θ₁`); blue dots = P_N (full-angle). They
  **overlap** → `FP ≡ P_N`, so **Fokker-Planck is adequate**. The grey 90° line is
  the isotropy bound; the curves stay far below it (≤~35° at 0.5 GeV, ≪1° above a
  few GeV), i.e. always small-angle, so FP cannot misbehave here.
* **Right panel — the *shape* test** (on a demo kernel, `k_local_demo.npz`). Green
  squares = the *real measured* angular shape; green line = a Gaussian of the
  **same variance**. They **overlap** → the **shape is irrelevant, only the
  variance matters**, so FP's variance-only input (`<θ²>(E)`) suffices. This panel
  uses a different, lower-energy kernel than the left, so it sits higher — that is
  expected and carries no meaning (it is not a disagreement with the left panel).
* **Together:** use the cheap Gaussian Fokker-Planck driven by the validated
  `<θ²>(E)` — both the *method* (left) and the *shape* (right) approximations are
  justified. The spread falls ~1/E, so the 3D angular effect vanishes at high E
  (1D recovered); `fokker_planck_3d.png` shows the resulting forward-peaked
  detector distribution.

**Simplifications.** Heat-kernel (forward-Gaussian) single-step *shape* — now
justified by the pure-shape test. `N_chain` is a fixed integer, not derived
per-energy (its ~22% effect on a ~1–2% correction is negligible).

### 5.4 Geomagnetic cutoff — `src/daemonflux/geomagnetic.py`, `geomag_backtrace.py`

**Idea.** Below the local rigidity cutoff, primaries are excluded — direction- and
charge-dependent (the East–West effect). This is the dominant sub-GeV 3D effect.

**Two implementations — both are implemented, and the code lets you use either.**
They sit at different levels of rigour, and the API exposes a clean switch between
them rather than forcing one:
1. **Analytic Störmer admittance** (`geomagnetic.py`, *wired into the package*):
   a closed-form dipole cutoff
   `R_c = 59.6 cos⁴λ / [1+√(1−sin ε sin ξ cos³λ)]²` wrapped into an admittance
   `G(E, zenith, azimuth) ∈ [0,1]` that multiplies the 1D flux/error (identity
   when no model is attached). Fast, dependency-free. Validated: equatorial
   vertical cutoff **14.9 GV**, cos⁴λ fall-off, East–West sign.
2. **First-principles back-tracing** (`geomag_backtrace.py`): integrate a charged
   trajectory (RK4) in the **full IGRF field** (degree 13 via `ppigrf`) near the
   surface + tilted dipole far out; a rigidity is allowed iff the back-traced
   particle escapes. The real field, the penumbra, and the East–West come out
   exactly. **Kamioka vertical cutoff = 11.31 GV (literature ~11.3 GV)** — which
   the pure-dipole formula cannot reproduce. The back-tracer also reproduces the
   Störmer formula in the aligned-dipole limit (**14.84 vs 14.9 GV**), so it is
   validated *both* ways. Produces a sky map `cutoff_map` (`geomag_cutoff_map.png`).

**Using either through one interface (they are interchangeable in the code).**
`GeomagneticModel` takes an optional `cutoff_source=` callable: leave it `None`
for the analytic Störmer cutoff (default), or pass the back-traced cutoff to drive
the *same* admittance machinery with the real-field cutoff —
`geomag_backtrace.cutoff_source(lat, lon, date)` returns exactly such a callable:

```python
from daemonflux.geomagnetic import GeomagneticModel
from geomag_backtrace import cached_cutoff_source
import datetime
gm = GeomagneticModel("kamioka",
                      cutoff_source=cached_cutoff_source(36.43, 137.31,
                                                  datetime.datetime(2020, 1, 1)))
```

**Fast cached path (`cached_cutoff_source`).** The plain `cutoff_source` is the
correct-but-slow drop-in (one trajectory back-trace per direction).
`cached_cutoff_source` is the **production** form: it batches the full
(zenith×azimuth) sky map once with `cutoff_map` (**one-time**; minutes for a coarse
grid, longer for fine — the cost is the per-RK4-step IGRF field evaluation on
trapped trajectories), **caches it to `cutoff_cache/*.npz`**, and thereafter
returns a microsecond **bilinear interpolant** (azimuth-periodic). This is what makes the principled back-traced
cutoff fast enough to be a package default — the second reviewer's "remaining
engineering task." So the choice is one argument. The same back-traced cutoff is also what the
absolute engine (`mceq3d_flux`, §5.11) and the research solver (`mceq3d_solver`,
via `geomag_site` for Störmer or `cutoff_GV` for an explicit/back-traced value)
consume — i.e. **every consumer can take either source.**

**Which to use.** Back-tracing is the principled one (real field, exact
penumbra/East–West, matches the literature site cutoffs) and is what drives all
the Honda-validated results; the Störmer admittance is the lightweight,
dependency-free default. The recommended production path is **back-traced `R_c`
folded into the primary through the cascade** (`mceq3d_flux`), which additionally
retires `x_eff`; the package admittance, fed via `cutoff_source`, is the
convenient drop-in for a quick directional factor on the 1D flux.

**A real bug found & fixed.** The East–West initially came out *reversed*; the
cause was (a) the analytic dipole `bfield` pointing south not north, and (b)
`arrival_direction` using the velocity azimuth rather than the standard *from*
azimuth. After the fix, from-West cutoff (7.25 GV) < from-East (20 GV) at Kamioka
zenith 75 — the correct sign for positive primaries. (Caught precisely *because*
of the back-tracer validation.)

**Simplifications / cheats.**
* The Störmer **admittance model** maps rigidity↔energy with a fixed effective
  inelasticity **`x_eff = 0.1`** and a fixed **`penumbra_width = 0.5`** — these
  are the simplification the back-tracer + primary-folding (§5.6) avoid. `x_eff`
  is a genuine cheat in `geomagnetic.py`; the engine path does not use it.
* The back-tracer treats the primary as **protons with `R[GV] ≈ E[GeV]`** —
  real primaries include nuclei (rigidity = E/Z per nucleon); this shifts the
  energy↔rigidity mapping and is not modelled.
* Far-field uses the dipole (higher Gauss terms fall as `(a/r)^(n+1)`,
  negligible); escape radius / step size tuned for speed (very long quasi-trapped
  trajectories could in principle be misclassified).
* The cutoff is a sharp allowed/forbidden transition at the scan resolution (no
  sub-penumbra structure).

### 5.5 Muon bending — `muon_bending.py`

**Idea.** A muon bends in the field before decaying, so its decay neutrinos
inherit a deflected direction — a real sub-GeV up/down & East–West driver,
closed-form (no cluster needed).

**Key result.** The in-flight deflection is `Δφ = qBτ/m`, **independent of muon
energy** (longer flight of an energetic muon exactly offset by its larger
gyroradius); numerically ~3–5° for B~0.3–0.5 G. What is energy-dependent is the
*decay-in-flight fraction* (only low-E muons decay in the atmosphere).

**Two effects, both delivered (`muon_bending.png`).**
* angular **spread** `<θ²>_bend(E)` added to the μ-decay-ν kernel;
* **coherent charge-dependent E–W shift**: μ⁺ ≈ +3.2° east, μ⁻ ≈ −3.2° west, so
  the ν/ν̄ (charge-separated) flux carries a **~3° E–W split sub-GeV**, while the
  summed displacement is small (~0.3°, since the charge ratio R≈1.27 ≈ 1).

**Improvements from the review (item #2, first round).**
* The decay path is now the **zenith-dependent curved slant** (`path_length_km`:
  ~15 km vertical → hundreds of km near the horizon), not a fixed 15 km.
* The bending spread is now **restricted to the muon-decay ν_μ channel** (weighted
  by `muon_decay_numu_fraction = f/(1+f)`), so it no longer bleeds into the
  direct-decay neutrinos.

**Full-field, arbitrary-direction shift (second-round item #2).** The
vertical-muon / horizontal-field-only approximation is replaced by
`coherent_shift_deg(zenith, azimuth, B_enu, charge)`, which uses the **actual muon
travel direction** (`muon_velocity_enu`) and the **full local field vector**
(`local_field_enu` — IGRF degree-13 via ppigrf; Kamioka ≈ (−0.04, 0.30, −0.37) G,
inclination ~49°). All three field components and the off-vertical geometry now
enter, returning the E–W *and* N–S arrival shifts; the old vertical/north-only
value is recovered as the special case. Example (Kamioka, μ⁺): vertical E–W +3.2°;
60° from North E–W +5.0°; 60° from East E–W +1.6°, N–S −3.2° — the direction
dependence the vertical approximation missed.

**Remaining simplifications.** `E_μ ≈ 3 E_ν` (mean inelasticity) and `R = 1.27` are
fixed constants; this only matters for fine-grained charge-resolved horizon shape
studies (the aggregate effect is the small, validated ~3° sub-GeV split).

### 5.6 Cascade engines — `mceq3d_solver.py` (parametrized), `mceq3d_production.py` (MCEq)

**Idea.** A coupled (energy × multipole) cascade `N→π/K→ν` carrying the angular
content; geomagnetics folded into the *primary* per direction (no `x_eff`).

* **`mceq3d_production.py` — production-grade.** Uses the **real MCEq** flux for
  the angular monopole `l=0`, and adds the angular layer per multipole via
  `set_mod_pprod`. **`l=0` reproduces MCEq exactly** (max rel diff 0.0e0,
  asserted in tests). So it inherits all of MCEq's flavor/charge/atmosphere
  physics; only the angular spread is the added ingredient
  (`mceq3d_production.png`).
* **`mceq3d_solver.py` — research engine.** A self-contained coupled cascade for
  exploring the full machinery (angular transport + geomag primary-folding +
  curved slant depth). **It uses parametrized scaling yields, not MCEq** (see cheats);
  it is the *architecture*, exercised end-to-end, with an exact 1D-reduction
  invariant (`mceq3d_solver.png`).

**Validation.** The angular machinery conserves the energy spectrum (the `l=0`
monopole is unchanged whether the angular treatment is on or "collimated"; max
rel diff ~1e-12). Geomag folded into the primary gives a smooth low-E suppression
that recovers to 1 at high E.

**Simplifications / cheats (parametrized engine).** Parametrized scaling yields (not MCEq); a
single pooled `θ²(E)` for all ν (no per-channel angular kernel in the solve); no
explicit muon transport (ν direct from two-body π/K decay); depth-independent
critical-energy competition (single slant depth); collimated primary `E^-1.7`;
geomag folded as a smooth erf step at the cutoff. **None of these affect
`mceq3d_production`'s `l=0`, which is exact MCEq.**

### 5.7 Directional flux — `directional_flux.py`, `unified_3d_flux.py`

**Idea.** Assemble Φ_νμ(E, zenith, azimuth) for a site = back-traced full-IGRF
cutoff folded into the primary + cascade + muon bending.

**Result (`directional_flux.png`, Kamioka).** At 1 GeV: ~**0.54** vertical
suppression, deepening to ~**0.12** near the horizon, with **East–West W/E =
2.17**. `unified_3d_flux.py` is the analytic-Störmer counterpart (W/E ≈ 8.5 at
zenith 70 with the admittance model).

**Simplifications.** The directional *ratio* is geomagnetics-dominated; the base
spectrum uses the parametrized cascade (so absolute scale is indicative, the ratio is the
physical output). The two engines (back-traced vs Störmer) differ in E–W
magnitude precisely because of the `x_eff` simplification in the Störmer path.

### 5.8 Spherical streaming PDE — `spherical_streaming.py` (+ `prototype_streaming.py`)

**Idea.** The genuine 3D transport term that couples *different arrival
directions* is the curvature `(1−μ²)/r ∂/∂μ` of the spherical streaming operator.
`prototype_streaming.py` first proved the slab streaming operator stays **sparse &
~linear** (no Monte-Carlo blow-up). `spherical_streaming.py` adds the curvature.

**Validation (`spherical_streaming.png`).**
* **P₁ ≡ spherical diffusion to machine precision (0.0000)** — an independent
  check of the curvature operator (the term the slab prototype could not test).
  Required a **staggered grid** to cure odd-even decoupling (a sawtooth I caught
  and fixed).
* stays **sparse & ~linear** (~n_l^1.35) with curvature included;
* redistributes detector flux toward the horizon (the off-axis term).

**Simplifications / cheats.** 1D-spherical (azimuthally symmetric) P_N; the
streaming-limit horizon redistribution shows **P_N Gibbs ringing** (angular
reconstruction artifact at finite N) — so that panel is *qualitative*; Dirichlet
(not Marshak vacuum) BCs with weak absorption for regularization. This module is
the validated *operator*; it is not itself coupled to the energy cascade (that is
§5.9, by a different, ray-exact method).

### 5.9 Curved-atmosphere cascade coupling — `spherical_cascade.py`

**Idea.** Run the energy cascade **down the curved line of sight** of each
arrival direction, so the decay-vs-interaction competition sees the real
altitude/density profile of that column — giving the absolute directional flux.

**Validation (`spherical_cascade.png`).**
* vertical = 1 (reference);
* **sub-GeV near-isotropic** (1.02) — mesons fully decay & primaries fully
  interact in every column;
* **high-E → sec θ** (cascade 1.52 vs sec θ=1.67 at cos θ=0.6);
* **finite horizon saturation** (4.8× at 2 TeV vs the divergent sec θ=25).

**The clarifying finding (and a corrected mislabel).** My first pass called this
"the Honda sub-GeV magnitude"; the numbers proved that *wrong* and I corrected it.
The sec θ horizon enhancement is a **high-energy** effect captured already by
1D-per-direction with curved columns; it is **not** the genuinely-3D term. The
genuinely-3D residual (inter-direction streaming) is small (~1–2%). So the large
sub-GeV 3D effects are geomagnetic + muon bending, not this geometric term.

**Simplifications / cheats.** Parametrized scaling yields (not MCEq); single primary index
γ=1.7; **isothermal exponential atmosphere** (ρ₀=1.205e-3 g/cm³, H=6.4 km — no
temperature profile, no seasonal variation); no explicit muon channel; no energy
loss / EM cascade / charge separation; **primaries collimated per direction**
(the isotropic-primary assumption — each arrival direction is fed by primaries
from that direction only); interacted mesons are absorbed (no re-injection).
Output is the curved 1D-per-direction flux; the 3D residual is *not* added here.

### 5.11 Trustable absolute directional engine — `mceq3d_flux.py` (capstone)

**Idea.** A usable, *absolute*, all-flavour 3D flux to ~0.5 GeV where every factor
is trusted and the result is validated *absolutely* against Honda — not just shape ratios.

**Construction.** `Φ_3D = Φ_1D(E,|cosθ|,s) · E_off(E,cosθ) · G_s(E,
R_c(cosθ,az)) · S(E)` — the complete deterministic-3D product (`offaxis=True` for
E_off; off → fast factorised path):
* `E_off(E, cosθ)` — the **first-principles off-axis 3D-production factor**
  `Φ_3D/Φ_1D` (`offaxis_factor`, `solve(offaxis=True)`; built by `offaxis_mc.py`):
  the complete genuine-3D/1D production ratio that both redistributes flux in
  zenith and produces the *net* sub-GeV near-horizon excess (horizon/vertical ~1.8
  at 0.3 GeV). Built with **no reference-flux input** from (i) MCEq depth-resolved
  production `p(X,E)=dΦ_ν/dX`, (ii) curved-atmosphere slant depth `X_slant(h,ψ)`,
  and (iii) the **pion production angle** `σ_π(E)` (`kinematic_kernel`): the
  inclusive `<θ²>(E_meson)` from the **SIBYLL/UrQMD generator moments** (chromo,
  pooled by `fokker_planck_3d.load_theta2`, NA61-validated) folded with exact
  π→μν decay and the H3a spectrum — **no hand-set p_T/x_F**. E_off is
  **flavour-independent**: the excess is a pion-production-rate effect inherited by
  all daughters (the near-isotropic muon-decay νμ carry the same excess — *not* a
  flat pedestal), so one factor multiplies each flavour's own 1D base; kaons
  (wider-angle, subdominant) are omitted. Reproduces the Honda and Bartol νμ *and*
  νe zenith shapes to their mutual ~5–15% across 0.3–10 GeV (0.3 GeV horizon:
  cascade-only 0.94 → E_off: νμ 1.82/Honda 1.89, νe 2.06/Honda 2.18); →1 at high E
  and at the vertical. Uncertainty: NA61 ±12% on σ_π → ±8% on the sub-GeV horizon
  excess (→0 by a few GeV); multi-generator spread needs regenerating the moments.
  **Supersedes** the legacy flux-conserving `full_3d`/`angular_factor` R (kept for
  comparison; must not be combined) and the earlier reference-anchored H (removed).
  **Muon-calibration consistency**: on the daemonflux base E_off is applied
  *shape-only* (`offaxis_shape_only`, auto) — the calibrated normalisation already
  lives in the 3D world; closure check at build: E_off(muon kernel) = 0.998–1.000
  vertical for E_μ ≥ 5 GeV (1.02→1.00 horizon, 5→30 GeV). (The closure diagnostic
  caught a real massive-daughter boost bug — tan θ needs E*/p*, not p*/E*.)
  **Full covariance**: `with_eoff_jacobian` (`sigma_pi_NA61` pull),
  `solar_sigma_gv` (`solar_phi` pull), `with_base_spread` (`base_model_spread`
  pull) append to `calib_params/corr/jac`; `flux_relerr` adds them in quadrature.
  Tables are tagged with the hadronic identity (`SIBYLL23D_H3a`) and the engine
  refuses mismatched tables. Unit tests: `test_offaxis_mc.py`,
  `test_kinematic_kernel.py`; assumption checks (`verify_offaxis.py`, all well
  inside the ±8% kernel systematic): p(X,E) zenith-independence ≤0.2% (60°) /
  2.4% (85°), rigidity-cutoff independence 1.1% (11.3 GV), cone-quadrature
  convergence 0.11% on doubling.
* `Φ_1D` — the 1D base, selectable via `base_model`:
  * `"mceq"` (default, dependency-free) — real MCEq per zenith, **curved
    atmosphere** (absolute norm, all four species, spectra, sec θ horizon
    enhancement). Carries the SIBYLL23D/H3a hadronic normalization (~25–30 % below
    Honda sub-GeV, see validation).
  * `"daemonflux"` (**recommended**, data-anchored) — daemonflux's
    **muon-calibrated** 1D flux: the E³-weighted sums `numuflux`/`nueflux` are
    de-weighted and split into species with the `numuratio`/`nueratio`, valid to
    ~1e9 GeV (`_base_daemonflux`). Restores ~10 % agreement with Honda at 1 GeV.

  Production is up/down symmetric, so `|cosθ|` is used. Units converted cm⁻²→m⁻².
* `G_s(E, R_c)` — the geomagnetic suppression computed as **MCEq(primary cut at
  R_c)/MCEq(full)**: the cascade-correct response to removing sub-cutoff
  primaries, with **no `x_eff`**. The rigidity cut is applied **per nucleus** to
  MCEq's `_phi0`: the proton flux is split into free protons (A/Z=1, R=E) and bound
  protons (A/Z≈2, R≈2E) via the p,n fluxes, and neutrons are all bound; precomputed
  on a small R_c grid (the *ratio* is ~zenith-independent) and interpolated.
* `R_c(cosθ, az)` — the back-traced full-IGRF cutoff (§5.4). **Down-going:** at
  the detector. **Up-going:** the production is on the far side, and since the
  neutrino travels straight the primary's velocity there equals the neutrino
  direction `d`, so the cutoff is a back-trace from the far-side production point
  `Q` with `u0 = -d` (`farside_production`) — a **global** geomagnetic treatment,
  batched into one vectorized back-trace.

**Absolute validation vs Honda — dense scan over 0.1–100 GeV.** Because the
sub-GeV-to-few-GeV band is the region that matters (oscillation physics) and where
flux models disagree most, `--validate` now reports a **dense low-energy grid**
(extra points below 1 GeV) and the engine floor is **e_min = 0.1 GeV** (100 MeV).
`base_comparison.py` scans both bases at Kamioka vertical (νμ ratio to Honda):

| E (GeV) | MCEq base / Honda | daemonflux base / Honda |
|---|---|---|
| 0.10 | 0.94 | 1.98 |
| 0.20 | 0.81 | 1.52 |
| 0.30 | 0.72 | 1.34 |
| 0.50 | 0.70 | 1.30 |
| 0.70 | 0.71 | 1.23 |
| 1.0 | 0.73 | **1.11** |
| 2.0 | 0.74 | **1.00** |
| 3.0 | 0.73 | 0.96 |
| 5.0 | 0.72 | 0.92 |
| 10–100 | 0.74–0.85 | **0.91** |

(`base_comparison.png`, `mceq3d_flux.png`). Three honest conclusions:

* **The 1D base sets the absolute scale, and there is a crossover.** Raw **MCEq
  (SIBYLL23D+H3a)** runs **~25–30 % below Honda** over 0.3–10 GeV (a real hadronic
  normalization deficit, *not* the ~10 % implied before the energy-matching fix
  below). **daemonflux's muon-calibrated base** (`base_model="daemonflux"`) is the
  better choice for **E ≳ 1 GeV** (within ~10 % of Honda all the way to 100 GeV),
  but it **over-predicts below ~0.3 GeV** — up to ~2× at 0.1 GeV — where it
  extrapolates past its muon-calibration region; there MCEq is closer to Honda.
  **Recommendation:** daemonflux base for E ≳ 0.3–0.5 GeV; treat 0.1–0.3 GeV as
  bracketed by the two bases (both are available, so the user can compare).
* **The 0.3–1 GeV band is genuine, irreducible model uncertainty.** Honda sits
  *between* MCEq (~0.7) and daemonflux (~1.3) there: two independent data-driven
  models disagree by ~1.8× at 0.5 GeV. No method choice removes this; it is the
  real sub-GeV atmospheric-flux uncertainty and should be carried as a systematic.
* **The geometry/geomagnetics are right regardless of base.** Both bases
  **reproduce the up/down asymmetry** (up-going > down-going at fixed E, as in
  Honda) and the sec θ and East–West behaviour — *ratio* observables independent of
  the absolute base. The base only rescales the overall normalization.

**A note on the comparison — energy matching.** MCEq's log-spaced grid has no point
exactly at 1 GeV (the nearest is 0.89 GeV). An earlier `--validate` compared by
nearest-index, pitting our 0.89 GeV flux against Honda's 1.0 GeV bin — a ~1.5×
mismatch on a steeply falling spectrum that *accidentally cancelled* the raw-MCEq
deficit and made the agreement look better than it is. The comparison now
**log-log interpolates both to the exact energy** (`_validate._at`); the table
above is the corrected, energy-matched result.

**Flavour ratio — the robust cross-check.** `(νe+ν̄e)/(νμ+ν̄μ)` is nearly
model-independent (set by the π→μ→e decay chain), so it is the sharpest test of
the flavour physics. This work vs Honda (vertical): **0.450/0.436 (0.5 GeV),
0.405/0.402 (1 GeV), 0.312/0.301 (3 GeV)** — **0.7–4%**. The engine is trustable
for *all four flavours* down to ~0.5 GeV, not just νμ.

**Usage.** `MCEq3DFlux().solve(lat, lon, cos_zeniths, azimuths)` → full-sky grid
(`cos_zeniths` may be negative); `interp_flux(result, E, cosθ, azimuth, species)`
evaluates it anywhere.

**Atmosphere.** Uses MCEq's realistic **CORSIKA US-Standard layered** profile by
default (not the isothermal exponential of the research demo); pass
`atmosphere=("MSIS00", (site, month))` for seasonal/site tracking.

**Simplifications / cheats (this engine).**
* **Nucleus rigidity** — handled **properly** per nucleus: the cut splits the
  nucleon flux into free protons (A/Z=1) and bound nucleons via MCEq's p,n fluxes
  (review item #1). The bound ⟨A/Z⟩ is now the **composition-weighted** value
  computed per energy from crflux (He/CNO/Si 2.0, Fe 2.077 → ≈2.005, rising slightly
  with E), not a fixed 2.0 (second-round item #4); the free/bound split is by
  isospin (bound p ≈ n).
* **Absolute normalization & its systematic.** With the MCEq base it is ~25–30 %
  low vs Honda sub-GeV; `base_model="daemonflux"` restores ~10 % at 1 GeV (the
  recommended central). The **model-spread is delivered as an explicit systematic**
  (`base_comparison.model_envelope`, half log-spread): ±37 % (0.1 GeV) → ±21 %
  (1 GeV) → ±4 % (100 GeV). The two bases bracket Honda only **below ~1 GeV**
  (geometric-mean central ~Honda there); above, both lie below Honda (daemonflux the
  closer, ~0.91) so the geometric mean is ~10–18 % low and daemonflux is the
  central. The band is an intra-framework floor, not a full inter-calculation
  envelope (§1e).
* Up-going uses one representative far-side production point per direction (the
  production region has finite extent — leading geometric term).
* Geomag `G` ratio precomputed at vertical and reused at all zeniths — **validated
  zenith-independent to ≤2.1 %** (sub-GeV horizon; ≤0.4 % above 1 GeV) by
  `geomag_zenith_check.py`, since the slant-depth shower effects cancel in the
  cut/full ratio. `solve(zenith_dependent_geomag=True)` removes even this residual.
* The sharp sec θ horizon region is now sampled with `horizon_grid()` (cosθ grid
  refined near cosθ=0), passed to `solve(cos_zeniths=...)`.

### 5.10 De-risking prototypes & utilities

`prototype_3d_cascade.py` (coupled (E,l) cascade, derived N_eff),
`coupled_3d_flux.py` (angular smearing on the MCEq 1D flux — the ~1–2% number),
`spherical_geometry.py` (curved line-of-sight geometric limit — saturation
~√(R/h₀); note this *ignores* primary attenuation so it overstates the thin-target
horizon factor, which the full cascade in §5.9 corrects), `channel_comparison.py`
(π vs K production angle/`<p_T>`), `angular_kernel.py` (moments/rows utilities).
Plots: `prototype_3d_cascade.png`, `coupled_3d_flux.png`, `spherical_geometry.png`,
`channel_comparison.png`.

---

## 6. High-statistics kernels (cluster)

Produced on a cluster per the runbook `KERNEL_GENERATION.md` (full account in
`KERNEL_PRODUCTION_REPORT.md`): UrQMD-3.4 (4–80 GeV) + SIBYLL-2.3d (80 GeV–1 PeV),
**200k interactions/point**, SLURM job arrays, spliced at 80 GeV → **41-point
grid**, for π⁺/π⁻/K⁺/K⁻. Installed as the drop-in moments `m_spliced.npz` (=π⁺)
and `m_{piminus,Kplus,Kminus}.npz`.

**Independent verification I ran on delivery:** schema/grid correct (4 GeV–1 PeV,
monotonic); `θ(E)` falls cleanly ~1/E (15° @1 GeV → sub-0.1° @TeV); kaons wider
than pions; agrees with the prior low-stat kernel but smoother; 99 tests pass.

**Caveat (from the cluster report, confirmed reasonable):** the consistency-gate
*shape* agreement is **~64%**, confined to SIBYLL's soft-pion kinematic turn-on
near its √s floor (MCEq extrapolates there; the generator yields are kinematically
zero). The **bulk region `x_L ≥ 0.005` agrees to 1–3%**. This affects absolute
*yields*, not the **angular moments** we use (validated independently). The
full-shape `d²N/dx_L dθ` set was **not** produced (and §5.3 shows it is not needed
for the angular spread).

---

## 7. Key physics findings (the magnitudes that matter)

| effect | size | energy | module |
|---|---|---|---|
| **Geomagnetic suppression** | ×0.5 vertical, ×0.1 horizon | sub-GeV | `directional_flux` |
| **Geomagnetic East–West** | W/E ≈ 2 (back-traced) | sub-GeV | `directional_flux` |
| **Muon bending E–W (ν vs ν̄)** | ±3° | sub-GeV | `muon_bending` |
| muon bending (summed ν) | ~0.3° | sub-GeV | `muon_bending` |
| production-angle 3D residual | ~1–2% | all | `coupled_3d_flux` |
| sec θ horizon enhancement | up to ~5× (finite) | **high-E** | `spherical_cascade` |
| inter-direction 3D residual | ~1–2% | all | `spherical_streaming` |

**Conclusion:** for sub-GeV daemonflux the actionable 3D physics is geomagnetic
(large) + muon bending (small but charge-asymmetric); the geometric/angular
cascade 3D corrections are percent-level.

---

## 8. Cross-checks against the literature (honest)

**Quantitative geomagnetic / kinematic:**
* Equatorial vertical Störmer cutoff **14.84 GV** vs the textbook **14.9 GV**.
* Kamioka vertical cutoff (full-IGRF back-trace) **11.31 GV** vs literature
  **~11.3 GV**.
* `cos⁴λ` geomagnetic latitude scaling reproduced.
* NA61 pion `<p_T>` to ~10%.

**Quantitative Honda HKKM2014 3D cross-check (`validate_honda.py`).** Done this
round — the Honda azimuth-dependent Kamioka table (`kam-ally-20-12-solmin`,
Φ(E, cosθ, azimuth) on a 20×12 grid) was downloaded, parsed, and compared on two
*ratio* observables (independent of absolute normalization and of Honda's
hadronic/atmosphere choices):

| observable | Honda | this work | module |
|---|---|---|---|
| East–West amplitude @0.5 GeV (near horizon) | 2.51 | 3.11 | `directional_flux` |
| East–West amplitude @1 GeV | **2.09** | **2.37** | `directional_flux` |
| East–West amplitude @5 GeV | 1.14 | 1.19 | `directional_flux` |
| East–West amplitude @10 GeV | 1.08 | 1.04 | `directional_flux` |
| sec θ horizon/vertical @100 GeV | **2.19** | **2.17** | `spherical_cascade` |
| sec θ horizon/vertical @1 TeV | 3.02 | 3.79 | `spherical_cascade` |

The **East–West energy dependence matches Honda well** — both peak sub-GeV
(~2.5 at 0.5 GeV) and **vanish above ~10 GeV** (the rigidity-cutoff signature);
the sub-GeV points sit ~10–25% high (my zenith-75 vs Honda's near-horizon bin,
plus geomagnetic-model differences). The **sec θ horizon enhancement** agrees
excellently at 10–100 GeV and is right in trend/magnitude elsewhere; the parametrized
cascade somewhat overshoots the TeV saturation (parametrized yields, no explicit muon
channel, simplified atmosphere — §9). Plot: `validate_honda.png`. This is the
validation I previously flagged as the key open item; it now **passes
quantitatively** for the directional structure.

**Remaining qualitative item:** a full per-bin reproduction of Honda's *absolute*
flux would additionally require matching their hadronic model, primary spectrum,
and NRLMSISE-00 atmosphere — out of scope here (the parametrized cascade gives ratios; the
MCEq-backed `l=0` gives the absolute scale). NA61 **K±**: now validated
(`ins1397003`, via `hepdata-cli`); UrQMD `⟨p_T⟩` is ~10–20 % harder than data (§5.2).

---

## 9. Complete list of simplifications & "cheats"

Consolidated, so nothing is buried. Grouped by severity.

**A. Genuine cheats (parameters chosen, not derived) — would change numbers:**
1. `geomagnetic.py` admittance: `x_eff = 0.1` (rigidity↔energy) and
   `penumbra_width = 0.5`. The back-traced engine path avoids `x_eff`; but the
   *package-wired* admittance factor uses it.
2. `N_chain = 2` (number of production generations) — a physical estimate, fixed,
   not derived per energy/species.
3. Muon bending: path zenith-dependent, spread restricted to the muon-decay channel
   (1st-round item #2), and the E–W shift now uses the **full local field +
   arbitrary muon direction** (2nd-round item #2, `coherent_shift_deg`). Remaining
   fixed constants: `E_μ=3E_ν`, `R=1.27` (the field is now real via ppigrf, not a
   fixed `B_north`).
4. ~~Primary as protons R≈E~~ **resolved in `mceq3d_flux`** (review item #1):
   per-nucleus rigidity via the free/bound split (free p = p−n, A/Z=1; bound at the
   **composition-weighted ⟨A/Z⟩≈2.005** from crflux, 2nd-round item #4). The
   Störmer/back-trace *cutoff* itself is a rigidity, so it is species-agnostic; only
   the primary *folding* needed the composition, now done.

**B. Structural simplifications (physics omitted) — bounded/argued small:**
5. Parametrized scaling yields in `mceq3d_solver` and `spherical_cascade` (not MCEq).
   *Mitigation:* `mceq3d_production` uses real MCEq for `l=0` (exact).
6. No explicit muon transport in the parametrized cascades (ν direct from two-body π/K
   decay); no EM cascade; no neutrino energy losses.
7. No charge separation in the parametrized cascades (the production engine inherits
   MCEq's; muon-bending charge effects computed separately).
8. Isothermal exponential atmosphere **only in the `spherical_cascade` research
   demo**; the production engine (`mceq3d_flux`) uses MCEq's realistic CORSIKA
   US-Standard layered atmosphere, with MSIS00 seasonal/site profiles selectable.
9. `spherical_cascade` primaries collimated per direction (isotropic-primary
   assumption); interacted mesons absorbed (no re-injection).
10. Single pooled `θ²(E)` drives the angular layer in the solver (per-channel
    kernels exist but aren't all wired into one solve).

**C. Numerical / methodological caveats — don't bias central values:**
11. `spherical_streaming` horizon panel has P_N Gibbs ringing (qualitative);
    Dirichlet (not Marshak) BCs with weak absorption.
12. Heat-kernel (Gaussian) single-step angular shape — *justified* by the
    pure-shape S_N test (§5.3): variance suffices.
13. Geomag cutoff folded as a smooth erf step (no sub-penumbra structure).
14. Back-tracer escape radius/step tuned for speed (quasi-trapped edge cases).

**D. Validation gaps (stated, not worked around):**
15. NA61 **K±** not fitted to HEPData (network) — only internal `<p_T>` scale
    cross-check.
16. Honda 3D **directional** cross-check **done** (§8, `validate_honda.py`): E–W
    and sec θ ratios match. A full per-bin **absolute** flux reproduction is not
    done (needs Honda's exact hadronic/primary/atmosphere setup).
17. High-stat gate shape agreement 64% in the soft-pion threshold region (affects
    yields, not the angular moments used).
18. Full-shape `d²N/dx_L dθ` high-stat kernels not produced (shown unnecessary
    for the spread, but would be needed for a full S_N yield transport).

**E. Integration scope:**
19. Only `geomagnetic.py` is wired into the installed package; `tools/mceq3d/` is
    a separate research layer, not plugged into the daemonflux spline API.
20. Absolute normalization is physical only for the MCEq-backed `l=0`; the parametrized
    engines give ratios/shapes.

### 9b. Forward-looking: when the kernel simplifications could bite

Both kernel simplifications (A-style choices in §5.1) are *deferred for efficiency,
not fundamental* — they do not degrade the current Honda-validated results, but it
is worth being explicit about the futures in which they would.

**Nitrogen-only target.** Safe now because the production *angle* is set by the
QCD `⟨p_T⟩` scale (~target-independent), and the target-dependent *yield* cancels
in the normalization-independent angular moments; air (⟨A⟩≈14.5) is essentially
nitrogen (A=14), so the air-vs-N angle error is likely **<1%**, even smaller than
the 1.3% C-vs-N bound. It would matter if:
* the angular layer is ever needed below ~1% (the 1.3% rides on a ~1–2% effect, so
  today it is ~0.02% on the flux) — then generate O and air-average, and actually
  *measure* Ar rather than extrapolate (the C-vs-N check did not include O/Ar);
* the **large-angle tails** are needed (the nuclear `p_T`-broadening / Cronin
  effect is A-dependent and grows at high `p_T`; the conventional flux only needs
  the small-angle variance, where it is negligible);
* the regenerated kernel is ever used for **absolute yields** instead of just
  angles (it is not — `mceq3d_flux` takes the absolute flux from MCEq's proper air
  target — but a repurposing would inherit the few-% N-vs-air yield difference).

**Proton-projectile-only.** The missing piece is the angular kernel for **meson
re-interactions** (π/K + air). Safe now because the two regimes do not overlap:
sub-GeV (where 3D matters) mesons *decay* before re-interacting, so the proton
kernel suffices; above the critical energies (~115 GeV π, ~850 GeV K) mesons do
re-interact, but there the production angle is already tiny (`⟨θ⟩`≈0.25° at
100 GeV, ≈0.04° at 800 GeV), so their *angular* contribution is negligible — and
their effect on the *magnitude* is handled correctly because `mceq3d_flux` gets
the absolute flux from MCEq, whose matrices already include the π/K projectile
interactions. It would matter if:
* the project goes to a **fully standalone deterministic 3D-MCEq** that does not
  lean on MCEq for the magnitude — then the complete `{p, n, π±, K±} × air`
  angular-kernel matrix is required for self-consistency (still small-angle);
* **species/charge-resolved** re-interaction kernels are wanted: the high-stat set
  has K±/π± as *secondaries* (p→π/K) but not as *projectiles* (π→X, K→X), and
  kaons are ~30% wider in `⟨p_T⟩` than pions.

Both removals are cheap — an extra target run + air-weighting, and the existing
one-line `ChromoSource(projectile=…)` hook plus cluster time — so they are the
natural first steps if the angular layer is ever tightened below ~1% or made
self-contained off MCEq.

---

## 10. Production-ready vs research-grade

* **Production-ready now:**
  * `src/daemonflux/geomagnetic.py` — a clean, optional, identity-by-default
    directional admittance factor on the 1D flux (the package plug-point).
  * **`mceq3d_flux.py`** — the absolute, **full-sky** directional engine (§5.11):
    selectable 1D base (`base_model="daemonflux"` muon-calibrated, **recommended**,
    or raw MCEq) × cascade-correct geomagnetics (down-going at the detector,
    up-going via the global far-side treatment), **validated absolutely against
    Honda** (energy-matched: ~10 % at 1 GeV with the daemonflux base, reproducing
    the up/down asymmetry), usable to ~0.5 GeV via `interp_flux`. The recommended
    way to get a 3D atmospheric-ν flux from this work.
* **Research-grade (architecture + validated ingredients):** the rest of
  `tools/mceq3d/` — quantifies effect sizes, proves feasibility, and supplies the
  validated factors (`geomag_backtrace`, `muon_bending`, `spherical_streaming`,
  the production-angle kernels) that feed the engine.

---

## 11. The one package change — `src/daemonflux/geomagnetic.py`

`GeomagneticModel` computes `G(E, zenith, azimuth) ∈ [0,1]` (Störmer cutoff +
penumbral admittance); `_FluxEntry` multiplies it onto the 1D flux/error. **When
no model is attached, behaviour is byte-identical to the published model.**
`flux()`/`error()` gained optional `azimuth_deg`/`geomag_location` arguments
(backward-compatible). Tests in `tests/test_geomagnetic.py`. See the README "3D /
geomagnetic corrections" section.

---

## 12. How to reproduce

```bash
pip install -e .[test]            # package + tests
pip install chromo ppigrf MCEq    # kernels, IGRF, consistency gate
cd tools/mceq3d
pytest -q                                   # 105 offline tests
python mceq3d_flux.py --validate --plot     # CAPSTONE: absolute Φ(E,θ,φ) vs Honda
python mceq3d_production.py --plot          # MCEq l=0 + angular layer
python directional_flux.py --plot           # Φ(E, zenith, azimuth), Kamioka
python geomag_backtrace.py                  # back-traced cutoffs vs Störmer
python spherical_streaming.py --plot        # curvature ≡ diffusion
python spherical_cascade.py --plot          # curved-atmosphere directional flux
python sn_transport.py --kernel k_local_demo.npz --plot   # FP adequacy + shape
```
Kernel regeneration on a cluster: see `KERNEL_GENERATION.md`.

---

## 13. Inventory

* **28 modules**, **21 test files**, **118 passing offline tests**, **26 plots**,
  Black/flake8 clean.
* Companion docs: `README.md` (full roadmap), `REVIEW.md` (self-review with
  statuses), `KERNEL_GENERATION.md` (cluster runbook), `KERNEL_PRODUCTION_REPORT.md`
  (the delivered kernel production).

---

## 14. Remaining work (in priority order)

1. ~~Up-going hemisphere~~ **done** (§5.11) — global far-side geomagnetic
   treatment; reproduces Honda's up/down asymmetry (up-going 0.97–0.98 at 1 GeV).
2. ~~Data-anchored normalization~~ **done** (`base_model="daemonflux"`, §5.11);
   ~~principled sub-0.3-GeV treatment~~ **done** — delivered as the explicit
   model-spread systematic (`base_comparison.model_envelope`, §1c/§5.11).
3. ~~Per-nucleus rigidity~~ **done** (review item #1, §5.11); ⟨A/Z⟩ now
   composition-weighted (second-round item #4).
4. ~~Cutoff-map cache~~ **done** (`cached_cutoff_source`, §1c/§5.4) — the
   back-traced cutoff is now fast enough to be the default package path.
5. ~~NA61 **K±** HEPData fit~~ **done** (§5.2, `validate_na61_kaon.py` via
   `hepdata-cli`).
6. ~~Finer cosθ grid near the horizon~~ **exposed** (`horizon_grid()`, §1d/§5.11);
   the `G_s` zenith-independence behind it is bounded to ≤2 % (`geomag_zenith_check`).
7. ~~First-principles near-horizon 3D excess~~ **done** (`offaxis_mc.py`,
   `kinematic_kernel.py`, `solve(offaxis=True)`, §5.11): the off-axis production
   factor `E_off` derives the sub-GeV horizontal enhancement from MCEq production +
   curved geometry + the **SIBYLL/UrQMD generator pion production angle** (chromo,
   NA61-validated) folded with exact decay — no hand-set p_T/x_F, no reference flux
   — reproducing Honda/Bartol to their mutual ~5–15% for **both νμ and νe**.
   Replaces the earlier reference-anchored `H`. `E_off` is **flavour-independent**
   (pion-production-rate effect inherited by all daughters). Uncertainty: NA61 ±12%
   → ±8% sub-GeV horizon. Refinements: (a) multi-generator (EPOS/QGSJET) spread —
   needs regenerating moments (chromo download / cluster); (b) a full continuous
   off-axis transport beyond the Gaussian cone; (c) the subdominant kaon parent.
8. ~~Anchor sub-0.3 GeV to a dedicated low-energy dataset~~ **done**
   (`base_model="hybrid"` + GSF primary): the raw-MCEq sub-GeV deficit is the
   **H3a primary** (~32% below the AMS-02/BESS/PAMELA-fitted GSF at 10 GeV, the
   sub-GeV parent region), not the hadronic model. Hybrid = GSF-anchored MCEq
   below ~0.8 GeV ⊕ muon-calibrated daemonflux above (`hybrid_weight`, smooth
   one-octave blend). Full engine vs Honda (solar-min — the table's epoch,
   pinned empirically vs Bartol fmin/fmax): **±11% or better, both flavours,
   all zeniths, 0.14–10 GeV** (νe vertical 0.3 GeV: 1.11 vs 1.64 with the
   daemonflux base alone). E_off tag: primary mismatch now warns (ratio is
   primary-insensitive); interaction-model mismatch still raises.
9. (If ever needed) full-shape high-stat kernels + S_N yield transport — shown
   *not* required for the angular spread.

---

## Appendix A — validation plots

| plot | shows |
|---|---|
| `validate_na61_pt.png` | NA61 pion `<p_T>` agreement (~10%) |
| `validate_na61_kaon_pt.png` | NA61 **kaon** `<p_T>` vs UrQMD (K⁺/K⁻) |
| `offaxis_excess.png` | first-principles near-horizon 3D excess: zenith shape with/without E_off vs Honda & Bartol |
| `base_comparison.png` | MCEq vs daemonflux base vs Honda + **model-spread systematic** |
| `geomag_zenith_check.png` | `G_s(E,R_c)` zenith-independent to ≤2% (verification #2) |
| `latitude_check.png` | smooth central+systematic across magnetic environments (verification #1) |
| `kernel_real_piplus.png`, `m_spliced.png` | kernel build & consistency gate |
| `angular_smoothness_demo.png` | direct-angle binning vs p_T resampling |
| `channel_comparison.png` | π vs K production angle / `<p_T>` (all 4 kernels) |
| `fokker_planck_3d.png` | Fokker-Planck angular spread |
| `sn_transport.png` | P_N≡FP adequacy + real-shape≡variance |
| `geomag_cutoff_map.png` | back-traced full-IGRF cutoff sky map (Kamioka) |
| `mceq3d_solver.png` | research engine: flux + geomag + σ_θ |
| `mceq3d_production.png` | MCEq `l=0` + angular layer (production-grade) |
| `muon_bending.png` | bending spread + coherent E–W shift |
| `directional_flux.png` | Φ(E, zenith, azimuth): suppression + E–W + zenith |
| `unified_3d_flux.png` | analytic-Störmer directional flux |
| `spherical_streaming.png` | curvature ≡ spherical diffusion; horizon redistribution |
| `spherical_cascade.png` | curved-atmosphere directional flux (sec θ, saturation) |
| `mceq3d_flux.png` | **absolute** νμ flux vs Honda (spectrum + zenith), capstone |
| `validate_honda.png` | Honda HKKM2014 3D cross-check (E–W + sec θ) |
| `spherical_geometry.png` | geometric horizon limit (thin-target) |
| `prototype_streaming.png`, `prototype_3d_cascade.png`, `coupled_3d_flux.png` | feasibility / ~1–2% residual |
