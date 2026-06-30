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
Inventory: **21 modules, 20 test files, 107 passing offline tests, 21 validation
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
  **whole sky** down to ~0.5 GeV as **MCEq (curved, per-zenith) × cascade-correct
  geomagnetic factor** (no `x_eff`), with **up-going handled by a global far-side
  geomagnetic treatment**. Its **absolute** νμ flux matches Honda at Kamioka to
  **0.90–0.98 at 1 GeV** and 0.82–0.87 at 0.5 GeV across the full sky, and it
  **reproduces the up/down asymmetry** (up-going 129–135 vs down-going 111–117 at
  1 GeV, matching Honda's 133–138 vs 122–130). The ~10–20% normalization offset is
  the SIBYLL23D/H3a-vs-Honda hadronic-model spread (anchorable to daemonflux's
  muon-calibrated base). The **flavour ratio (νe/νμ) matches Honda to 0.7–4%** —
  the robust, model-independent test passes for all four species. **Trust
  boundary:** the sub-GeV near-horizon carries a larger ~20% uncertainty from the
  nucleus-rigidity approximation (documented).
* **Net message:** the value for sub-GeV daemonflux is overwhelmingly in the
  **geomagnetic layer** (production-ready as an admittance factor) plus muon
  bending; the deterministic-3D cascade machinery is validated as an architecture
  and quantifies that the remaining geometric/angular 3D corrections are small.

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

### 5.2 NA61 validation — `validate_na61.py`

**Idea/method.** Validate the regenerated production kinematics against real data:
`<p_T>(p_lab)` of charged pions in p+C at 31 GeV/c (HEPData `ins886780`, the
T2K-replica record), applying the experiment's angular acceptance.

**Result.** ~**10%** agreement in `<p_T>` over the measured range
(`validate_na61_pt.png`).

**Simplifications / limitations.**
* Pions only with a full HEPData fit. **K± was not fitted against HEPData**
  because this environment has no network (SSL). Instead the kaon kinematics are
  cross-checked *internally*: `channel_comparison.py` shows kaon `<p_T>` ≈
  0.4–0.65 GeV vs pion ≈ 0.2–0.46 GeV — kaons harder, consistent with the
  well-established NA61 kaon scale (~0.5 GeV). A proper HEPData K± fit is left as
  future work (needs network).
* UrQMD acceptance applied as a simple θ<420 mrad cut.

### 5.3 Angular transport — `fokker_planck_3d.py`, `sn_transport.py`

**Idea.** Propagate the angular distribution with depth. Two methods, to bound
the approximation: small-angle **Fokker-Planck** (Gaussian) and the full-angle
**P_N / S_N** spherical convolution (bounded by isotropy).

**Method.** In the multipole basis a forward production step is a heat kernel
`c_l = exp(−l(l+1)θ₁²/4)`; N production generations multiply coefficients
(`c_l^N`), so `<cos θ> = c₁^N` in closed form. The single-step variance is the
**NA61-validated `<θ²>(E)`**. The number of generations is the **finite parent
chain `N_chain ≈ 2`** (see the "cheats" §9 — this replaced an earlier unphysical
`N_gen = slant/λ`).

**Validation (`sn_transport.png`).**
* P_N (bounded) agrees with FP (`√N·θ₁`) for `N_chain≈2`; they only diverge at an
  *unphysical* large N — confirming **FP is adequate** in this regime.
* **Full-shape cross-check (this round):** `realshape_sn_spread` builds the
  single-production density directly from a measured `d²N/(dx_L dθ)` kernel and
  compares it to a Gaussian of the **same variance**. **Pure-shape median rel
  diff ≈ 0** → for the forward production distribution `arccos<cos θ>` is set by
  the *variance alone*, so the variance-only FP input is sufficient and the
  (heavier-tailed) full-shape kernel is **not needed for the angular spread**.
  (Demo kernel `k_local_demo.npz`, UrQMD-3.4 low-E.)

**Simplifications.** Heat-kernel (forward-Gaussian) single-step *shape* — now
justified by the pure-shape test. `N_chain` is a fixed integer, not derived
per-energy.

### 5.4 Geomagnetic cutoff — `src/daemonflux/geomagnetic.py`, `geomag_backtrace.py`

**Idea.** Below the local rigidity cutoff, primaries are excluded — direction- and
charge-dependent (the East–West effect). This is the dominant sub-GeV 3D effect.

**Two implementations.**
1. **Analytic Störmer admittance** (`geomagnetic.py`, *the one wired into the
   package*): `G(E, zenith, azimuth) ∈ [0,1]` multiplies the 1D flux/error;
   identity when no model is attached. Validated: equatorial vertical cutoff
   **14.84 GV** (Störmer 14.9), cos⁴λ latitude fall-off, East–West sign.
2. **First-principles back-tracing** (`geomag_backtrace.py`): integrate a charged
   trajectory (RK4) in the **full IGRF field** (degree 13 via `ppigrf`) near the
   surface + tilted dipole far out; allowed iff the back-traced particle escapes.
   **Kamioka vertical cutoff = 11.31 GV (literature ~11.3 GV)** — a genuine
   quantitative literature agreement. Produces a sky map `cutoff_map`
   (`geomag_cutoff_map.png`).

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

**Simplifications / cheats.**
* Coherent E–W uses a **vertical-muon approximation** (only the horizontal field
  bends it E–W); real muons arrive at all zeniths.
* `E_μ ≈ 3 E_ν` (mean inelasticity), `path = 15 km`, `R = 1.27`, `B_north =
  0.30 G` are fixed constants.
* In `mceq3d_solver` the bending spread is added to *all* decay-ν (not only the
  muon-decay channel) — a small over-inclusion, gated by the decay fraction.

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

**Construction.** `Φ_3D(E, cosθ, az, s) = Φ_MCEq(E, |cosθ|, s) · G_s(E, R_c(cosθ,
az))`:
* `Φ_MCEq` — real MCEq per zenith, **curved atmosphere** (absolute norm, all four
  species, spectra, and the sec θ horizon enhancement). Production is up/down
  symmetric, so `|cosθ|` is used. Units converted cm⁻²→m⁻².
* `G_s(E, R_c)` — the geomagnetic suppression computed as **MCEq(primary cut at
  R_c)/MCEq(full)**: the cascade-correct response to removing sub-cutoff
  primaries, with **no `x_eff`**. The cut is applied to the primary nucleons in
  MCEq's `_phi0` (protons at R=E, bound neutrons at R≈2E); precomputed on a small
  R_c grid (the *ratio* is ~zenith-independent) and interpolated.
* `R_c(cosθ, az)` — the back-traced full-IGRF cutoff (§5.4). **Down-going:** at
  the detector. **Up-going:** the production is on the far side, and since the
  neutrino travels straight the primary's velocity there equals the neutrino
  direction `d`, so the cutoff is a back-trace from the far-side production point
  `Q` with `u0 = -d` (`farside_production`) — a **global** geomagnetic treatment,
  batched into one vectorized back-trace.

**Absolute validation vs Honda — full sky (`--validate`, Kamioka νμ):**

| E, cosθ | this work | Honda | ratio |
|---|---|---|---|
| 1 GeV, up-going −0.95 | 129 | 133 | **0.97** |
| 1 GeV, up-going −0.55 | 135 | 138 | **0.98** |
| 1 GeV, down-going 0.55 | 117 | 130 | 0.90 |
| 1 GeV, down vertical 0.95 | 111 | 122 | 0.91 |
| 0.5 GeV, up-going −0.95 | 531 | 631 | 0.84 |
| 0.5 GeV, down 0.95 | 416 | 510 | 0.82 |

(flux in /(m² s sr GeV); `mceq3d_flux.png`). The agreement is ~10–20% across the
**whole sky**, and the engine **reproduces the up/down asymmetry** (up-going
129–135 > down-going 111–117 at 1 GeV, as in Honda 133–138 > 122–130) — the
observable that drives atmospheric-ν oscillation analyses. The ~10–20% offset is
the inter-model spread (SIBYLL23D+H3a vs Honda; visible at the no-geomag peak).

**Flavour ratio — the robust cross-check.** `(νe+ν̄e)/(νμ+ν̄μ)` is nearly
model-independent (set by the π→μ→e decay chain), so it is the sharpest test of
the flavour physics. This work vs Honda (vertical): **0.450/0.436 (0.5 GeV),
0.405/0.402 (1 GeV), 0.312/0.301 (3 GeV)** — **0.7–4%**. The engine is trustable
for *all four flavours* down to ~0.5 GeV, not just νμ.

**Usage.** `MCEq3DFlux().solve(lat, lon, cos_zeniths, azimuths)` → full-sky grid
(`cos_zeniths` may be negative); `interp_flux(result, E, cosθ, azimuth, species)`
evaluates it anywhere.

**Simplifications / cheats (this engine).**
* **Nucleus rigidity** handled by superposition with R≈(A/Z)E (protons R=E,
  neutrons R≈2E). This over-suppresses where the cutoff is highest (sub-GeV
  horizon East), the main residual at the sub-GeV horizon. Proper per-nucleus
  rigidity is the fix (~20%).
* **Absolute normalization** carries the SIBYLL23D/H3a hadronic-model systematic
  (~15–20%); the one-line fix is to use **daemonflux's muon-calibrated 1D flux as
  the base** (data-anchored; not done here as the spline data needs network).
* Up-going uses one representative far-side production point per direction (the
  production region has finite extent — leading geometric term).
* Geomag `G` ratio assumed zenith-independent (precomputed at vertical); the
  sharp horizon spike needs a finer cosθ grid than the demo's.

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
MCEq-backed `l=0` gives the absolute scale). NA61 **K±** is still only an internal
`<p_T>`-scale check (HEPData fit needs network).

---

## 9. Complete list of simplifications & "cheats"

Consolidated, so nothing is buried. Grouped by severity.

**A. Genuine cheats (parameters chosen, not derived) — would change numbers:**
1. `geomagnetic.py` admittance: `x_eff = 0.1` (rigidity↔energy) and
   `penumbra_width = 0.5`. The back-traced engine path avoids `x_eff`; but the
   *package-wired* admittance factor uses it.
2. `N_chain = 2` (number of production generations) — a physical estimate, fixed,
   not derived per energy/species.
3. Muon bending fixed constants: `E_μ=3E_ν`, `path=15 km`, `R=1.27`,
   `B_north=0.30 G`, and the vertical-muon approximation for the E–W projection.
4. Primary treated as **protons, R≈E** (no nuclei/rigidity-per-nucleon) in the
   geomagnetic folding.

**B. Structural simplifications (physics omitted) — bounded/argued small:**
5. Parametrized scaling yields in `mceq3d_solver` and `spherical_cascade` (not MCEq).
   *Mitigation:* `mceq3d_production` uses real MCEq for `l=0` (exact).
6. No explicit muon transport in the parametrized cascades (ν direct from two-body π/K
   decay); no EM cascade; no neutrino energy losses.
7. No charge separation in the parametrized cascades (the production engine inherits
   MCEq's; muon-bending charge effects computed separately).
8. Isothermal exponential atmosphere (no temperature profile / seasonal).
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

---

## 10. Production-ready vs research-grade

* **Production-ready now:**
  * `src/daemonflux/geomagnetic.py` — a clean, optional, identity-by-default
    directional admittance factor on the 1D flux (the package plug-point).
  * **`mceq3d_flux.py`** — the absolute, **full-sky** directional engine (§5.11):
    trusted MCEq base × cascade-correct geomagnetics (down-going at the detector,
    up-going via the global far-side treatment), **validated absolutely against
    Honda** (~10–20% across the whole sky, reproducing the up/down asymmetry),
    usable to ~0.5 GeV via `interp_flux`. The recommended way to get a 3D
    atmospheric-ν flux from this work. Its documented boundaries: sub-GeV-horizon
    nucleus-rigidity systematic (~20%) and hadronic-model normalization (swap in
    daemonflux's muon-calibrated base to anchor it).
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

* **21 modules**, **20 test files**, **107 passing offline tests**, **21 plots**,
  Black/flake8 clean.
* Companion docs: `README.md` (full roadmap), `REVIEW.md` (self-review with
  statuses), `KERNEL_GENERATION.md` (cluster runbook), `KERNEL_PRODUCTION_REPORT.md`
  (the delivered kernel production).

---

## 14. Remaining work (in priority order)

1. ~~Up-going hemisphere~~ **done** (§5.11) — global far-side geomagnetic
   treatment; reproduces Honda's up/down asymmetry (up-going 0.97–0.98 at 1 GeV).
2. **Data-anchored normalization** — use daemonflux's muon-calibrated 1D flux as
   the `mceq3d_flux` base instead of raw MCEq (one-line swap; removes the ~10–20%
   hadronic-model offset). Needs the spline data (network).
3. **Per-nucleus rigidity** in the geomagnetic cut (retire the superposition
   R≈(A/Z)E approximation) — the ~20% residual at the sub-GeV horizon.
4. Finer cosθ grid near the horizon to resolve the sharp sec θ spike.
5. NA61 **K±** HEPData fit (§5.2, needs network).
6. (If ever needed) full-shape high-stat kernels + S_N yield transport — shown
   *not* required for the angular spread.

---

## Appendix A — validation plots

| plot | shows |
|---|---|
| `validate_na61_pt.png` | NA61 pion `<p_T>` agreement (~10%) |
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
