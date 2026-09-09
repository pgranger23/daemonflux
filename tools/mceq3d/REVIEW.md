# Critical review of the 3D / geomagnetic work

A precise, self-critical account of what was built, where it is approximate or
cuts corners, every validation actually run, and the leftover issues. Read this
before trusting any number downstream.

---

## ⚠️ CURRENT STATUS (2026-09-04) — read this first

**This block supersedes everything below it, including the 2026-07-23 block
that used to open this file (now kept under "Superseded 2026-07-31 status").**
A 2026-09-03 critical audit (`scratchpad/audit_report.md`) and a 2026-09-04
Phase-1 repair sprint found that the two headline conclusions of the old
"CURRENT STATUS" block were wrong. Full detail, every number sourced, is in
`PHASE1_RESULTS.md`; the module map is `ARCHITECTURE.md`.

* **Retracted: "The one systematic that remains ... the factorised cone-average
  under-softens the sharp back-traced cutoff relative to Honda's full 3D
  treatment."** This was dominated by numerical defects — a coarse rigidity
  scan, bilinear interpolation across the convex near-limb rise, a 9.5 GV seam
  at 89 deg stitching the down-going cutoff to the antipodal far-side cutoff,
  and an under-measured production-point displacement — not an intrinsic
  under-softening. Repairing them (`geomag_backtrace.scan_upper_cutoff`, dense
  limb nodes, `sublimb="prod_point"`) plus replacing the product
  `E_off x <G>_cone` by the single joint cone integral (`joint_cone.py`) and
  correcting the generator-moment `arctan` -> `arcsin` bug closes most of the
  sub-GeV zenith-shape deficit and the 0.5 GeV E-W overshoot (nu_mu H/V 0.841 ->
  1.040 x Honda; nu_mu 87 deg/0.5 GeV E-W +15% -> +2%). See
  `PHASE1_RESULTS.md` §3.1-3.2 for the full S0->S4 table.
* **Retracted: "Confirmed structural gap: charge-dependent East-West."** The
  diagnostic used (`max/min` over azimuth) is exactly even under a sign flip of
  a coherent shift, and Honda's azimuth convention (counterclockwise from
  South) was never mapped to ours — both bugs, not physics. With the corrected
  observable (`diag_ew_charge_fourier.py`) and convention, the model produces a
  charge-dependent E-W split with the right sign and 60-90% of Honda's *phase*
  magnitude at 0.3-0.5 GeV; what remains genuinely missing is the *amplitude*
  half (13-30% of Honda's) and the species-ordering pattern. See
  `PHASE1_RESULTS.md` §3.3 and §5.
* Negative results from the same sprint: energy-loss-enhanced muon bending is
  refuted (wrong sign); the fully-coupled charged-transport march is a
  structural dead end for the E-W observable (`mceq3d_real.march_checkpoints`,
  `coupled_ew_charge_diag.py` — kept for the record only, see
  `ARCHITECTURE.md`); the cone-too-narrow hypothesis is refuted; sharpening the
  geomagnetic penumbra (`SIGMA_LNR` 0.5 -> 0.0) fixes the East-side level but
  neither the Bartol H/V overshoot nor the charge-splitting amplitude.
* Open items carried forward (not fixed by Phase 1): the nu_e/antinu_e/antinu_mu
  E-W species pattern, the nu_mu 1 GeV E-W overshoot, a 6-13% H/V overshoot vs
  Bartol with a genuine ~5% sub-GeV solid-angle excess, `with_eoff_jacobian`
  being first-order only, and the v2 (arcsin-corrected) moments not being wired
  as the engine default.
* **2026-09-05 addendum.** A follow-up investigation closed the North/South
  and dipole-phase item that this block did not yet list: the apparent ~20 deg
  rotation of the back-traced cutoff's azimuthal pattern is the Earth-shadow
  cone biasing the first-harmonic phase estimator, not a frame or convention
  bug (a real but small dipole-axis longitude bug was found and fixed en
  route, with no measurable effect on the map); the genuine remaining physics
  was that the cutoff map is anchored at the detector while the primary of a
  near-horizon neutrino enters the atmosphere up to 617 km away, at a
  different geomagnetic latitude. Anchoring the cone samples' cutoff at the
  neutrino's own production point instead (`mceq3d_flux.CUTOFF_ANCHOR =
  "prod_point"`, a cached per-site displaced-site map family, one-time cost
  4.1 h) removes essentially the whole effect with no free parameter — the
  North/South model-over-Honda pair moves from 0.86-0.92 / 1.07-1.23 to
  0.93-1.02 / 0.91-1.05 and the dipole-phase offset from 12-14 deg to
  <=1.3 deg (0.5 GeV, all four species), confirmed by a grid-wide improvement
  in the 90th-percentile |log10(ours/Honda)| for every species — and it was
  made the **default** on 2026-09-05, since the family build is a one-time,
  cached precompute like the cutoff map itself (`"detector"` remains the fast
  A/B path). It leaves the East-West species pattern and its amplitude
  splitting untouched: that item remains open. Full detail in
  `PHASE1_RESULTS.md` §6; the module map is in `ARCHITECTURE.md`.
* **2026-09-09 addendum: residual classification.** Three further
  investigations classified every remaining residual as our approximation, a
  different input, or unknown, and checked one of them against real data. The
  East-West species-compression pattern (nu_e > antinu_mu > nu_mu > antinu_e
  in Honda, compressed to ~2.7-3.3 here) is confirmed as **ours**: it stays
  flat across nine hadronic-model/primary combinations, and it is now
  **corroborated by Super-Kamiokande data** — the measured East-West
  asymmetry wants more nu_e and less nu_mu splitting than Honda's own Monte
  Carlo, the same direction our compression predicts, at ~2.0-2.2 sigma with
  no free parameters (no later SK azimuthal analysis exists beyond
  arXiv:1510.08127). The nu_mu East-West overshoot is **mostly a different
  input** (the primary spectrum) at 0.5 GeV (+12% -> +1% under Honda's own
  primary proxy) and **partly unknown** at 1-2 GeV, where a genuine +10-16%
  residual survives every primary and hadronic model tried. The H/V overshoot
  against Bartol and the "~5% conservation excess" are **reclassified: not a
  defect**. A first-principles re-derivation shows Eq. (8) is the exact
  straight-line 3D transport of an isotropic primary flux (gated to its own
  flat-atmosphere closed form to 0.02-0.12%); the "~5% excess" was the plain
  solid-angle average, which is measure-dependent (the same exact kernel gives
  15%, not 5%, for that average in a flat atmosphere), while the physically
  conserved ground-crossing average shows a 5.5% *deficit* — nothing is
  manufactured. The real, quantified culprit is the NA61 +-12% uncertainty on
  the production-cone width, which alone moves H/V by ~+-20% at 0.3 GeV and
  spans the Honda-Bartol gap; every other ingredient moves it <=2%. Absolute
  sub-1.7-GeV normalisation remains attributed to input choice (primary
  spectrum/base), as before; Honda's charge-splitting phase reversal above
  ~1 GeV remains genuinely unexplained. Full detail, every number sourced, in
  `PHASE1_RESULTS.md` §8; restated in `PAPER_DRAFT.md` §3.6/§4.7/§5/§6.

---

## Superseded 2026-07-31 status

Everything from here down to "## Resolution status" is the block that used to
open this file. It predates the 2026-09-03 audit and is superseded where it
conflicts with the block above; kept verbatim for history.

### ⚠️ CURRENT STATUS (2026-07-23) — read this first

**Everything below the "Resolution status" heading predates commit `71dd9d9` and is
superseded where it conflicts with this block.** Authoritative sources, in order:
`ARCHITECTURE.md` (module map), `PAPER_DRAFT.md` §4-6 (physics + validation), then
this file (history and the older approximation inventory).

#### What changed since the old review

* **`E_off` cone kernel (delivered physics change).** The sampled `(x_L,θ)` kernel
  (`k_spliced.npz`) was root-caused as **16-37% too wide** in per-secondary
  meson-angle RMS — a 0.667° θ-binning artefact that grows with energy. The
  delivered cone now uses the exact, NA61-validated **moment `σ_π`**
  (`offaxis_mc.py`, `cone_kernel="moments"`, the default; legacy via
  `--cone-kernel sampled`). Brings νμ and νe horizon/vertical onto **both** Honda
  and Bartol. See `diag_kernel_consistency.py`, `diag_cone_fix.py`.
* **Production-vertex closure — the old "no coupled solve" item is CLOSED.**
  `mceq3d_prodvertex.py` reproduces the delivered `E_off` to **~1-2%** from an
  independently hand-marched real-matrix cascade, with both gates passing
  (cone→0 ⇒ `E_off`→1 to 5e-4; →1 at high E). Physics reframe: neutrinos
  free-stream, so there is **no transport coupling** for them — the entire 3D
  neutrino effect is production geometry, which `E_off` already integrates.
  Caveat: independent *code path*, same underlying model — not independent physics.
* **Absolute validation now exists across all four species.** `diag_full_comparison.py`
  compares against Honda's full 10×12×101 table: absolute flux **0.82-1.10×** Honda
  everywhere 0.15-100 GeV; grid-wide median |log10 ratio| ~0.03 (~8%), 90th pct
  ~17-23%. Flavour ratio within a few %, charge ratios within ~5%. Bartol
  (Honda-independent) confirms 0.94-1.04× over most of the range.

#### The one systematic that remains (and it is ONE, not several)

The extreme-horizon **East-West overshoot** (+15-24% at 87°) and the **sub-GeV
zenith-shape deficit** (11-15% at 0.3-1 GeV) are the **same underlying systematic**:
the factorised cone-average **under-softens the sharp back-traced cutoff** relative
to Honda's full 3D treatment. Evidence: inverting each observable to an effective
cutoff gives a consistent, nearly **energy-flat** downward shift (E-W: Honda ~27-29
GV East vs ours ~33-39; zenith: Honda ~11-17 GV horizon vs ours ~14-20, i.e. ~3 GV).
Honda's implied *vertical* cutoff matches ours, so the vertical is correct and only
the high-cutoff directions are under-softened.

Ruled out as the cause, each with a dedicated diagnostic: inter-direction coupling
(<1%, `coupled_ew_diag.py`), production-point displacement (~4%, `diag_ew_cause.py`
— note an earlier version of that script had a root-selection bug that overstated
it), penumbra sharpness (the real cutoff is *sharper* than our erf,
`diag_ew_penumbra.py`), hadronic interaction model (~1/6, DPMJET-III-19.3,
`diag_ew_dpmjet.py`), cutoff averaging over the production region (~2 GV,
`diag_ew_prodregion.py`), and the muon-decay channel width (~4%,
`diag_ew_muon.py`). For the zenith shape specifically, `diag_G_zenith.py` shows the
per-zenith cascade **response curve is correct** (0.97-1.00) and the cone averaging
already *helps* (lifts HV_G 0.598→0.855) — the residual is the cutoff values.

**Do not tune this to Honda.** Quote it as a systematic of the fast factorised
treatment. Any future fix should target the softening of high-cutoff directions
(energy-flat) and would improve both observables at once.

#### Confirmed structural gap: charge-dependent East-West

The model predicts ≈zero ν-vs-ν̄ E-W amplitude splitting; Honda shows a factor 2-3
(νμ diff -0.91 to -1.53; νe +1.7 to +3.3). Verified via **two independent
implementations** — the factorised coherent shift scaled to 30× (saturates at ~20%
of Honda) and the fully coupled multi-species transport with charge-signed Lorentz
bending (exact null) — with the latter's null confirmed **not** to be a
checkpoint-resolution artefact by a 10× scan (`diag_checkpoint_resolution.py`).
π⁺/π⁻ cone widths are equal to 1-3% (refuted directly from data). K⁺/K⁻ widths do
differ (~8-9% at 1-2 GeV) but `cone_geff` has **no kaon-parent cone term at all** —
a real, currently-missing piece, too small alone to close the gap. The one untested
lever is a spatially-varying **B(r)** along the ~1000 km near-horizon path (both
implementations use a single detector-point field vector).

#### Known-and-accepted approximations (still true)

* `channel_fractions`/`mudecay_shape` were vertical-only; now zenith-dependent, but
  this is worth only ~0.3% (the two cone widths it blends are nearly identical).
* Below ~0.15 GeV the factorised geometry is an extrapolation: the `E_off`
  flux-conservation residual grows to ~8% at 0.11 GeV (guardrail deliberately
  loosened there, `test_offaxis_conservation.py`).
* Honda/Bartol are **models, not data**. All "validation" against them is
  model-self-consistency. The genuine data-level test (Super-K measured E-W) is
  still not done — this remains the single biggest gap in the validation story.

---

## Resolution status (update, pre-`71dd9d9` — historical)

Worked through the action items in priority order:

* **(A) Kernel normalization — RESOLVED.** The "~4.4×" was *my bug*: MCEq stores
  `hadr_yields = (dN/dx) · Δ(lnE_grid)`; I read it as dN/dx without dividing by
  the log-bin width (ΔlnE = 0.2303 ⇒ 1/ΔlnE = 4.34). Fixed in
  `load_mceq_reference`; the gate now gives **norm = 1.005** and raw (absolute)
  agreement within 5% went 0% → 51%. The kernels are correctly normalized.
* **(C) `N_gen = slant/λ` — RESOLVED.** Replaced by a finite parent-chain count
  `N_CHAIN = 2` (production p_T dominates the kick; the chain is short). The
  spread is now **zenith-independent** and moderate (σ_θ ≈ 18° at 0.7 GeV, was
  44°). The S_N solver now **confirms FP is adequate** (P_N ≈ FP for N≈2); the
  old large-angle divergence was purely the `slant/λ` artifact (reproducible with
  `--nchain 120`).
* **(G) Target N vs C — RESOLVED.** Production ⟨p_T⟩ differs by **1.3%** between
  carbon and nitrogen (measured), so the angular content is target-independent;
  validating on C (NA61) and applying on N (≈air) transfers.
* **(D) σ_θ magnitude validation — PARTIAL.** σ_θ is now anchored to the
  NA61-validated θ₁(E) × √N_CHAIN, so the dominant input is data-validated; the
  residual uncertainty is the N_CHAIN factor (√2-level) and the single-step shape.
  A direct Honda-3D comparison is still not done.
* **(M) S_N real-kernel path — DOCUMENTED.** `single_production_f1` + numerical
  Legendre are kept as a hook but explicitly marked statistics-limited and not on
  the headline path (which uses the analytic heat-kernel).
* **(E) Coupled energy×angle solve — DE-RISKED (prototype).**
  `prototype_3d_cascade.py` solves the coupled (E, multipole) cascade and shows:
  (i) it **reduces exactly to 1D** when the production angle → 0 (σ_θ = 0.000°);
  (ii) it is **cheap** — in the Legendre representation the single-column solve
  decouples per multipole, so the whole thing runs in ~40 ms (0.5 ms/multipole),
  i.e. the angular-production dimension is essentially free; (iii) the effective
  parent-chain count is **not** 2 — the solve gives σ_θ(ν) ≈ 0.3–0.5·θ₁(E_ν)
  (the parent meson is more energetic/forward), so the conventional-flux angular
  effect is **<1%**, even smaller than the ~1–2% estimate. **The remaining
  performance unknown is the spatial *streaming* term** (off-axis geometry),
  which re-couples the multipoles and is deliberately not in the single-column
  prototype — that is the next de-risking step.
* **(E-streaming) Spatial streaming cost — DE-RISKED (prototype).**
  `prototype_streaming.py` adds the streaming term that re-couples the multipoles
  (slab P_N) and measures the cost. Verdict: the operator stays **sparse**
  (~5 nonzeros/row; density falls as 1/dof), the direct sparse solve scales as
  **~n_l^1.5** (sub-quadratic; iterative solvers do better), and the physics is
  validated against the analytic diffusion limit (~2% in the interior). So the
  streaming-coupled deterministic solve does **not** blow up to MC scale — a real
  3D-MCEq is computationally feasible. The full spherical solver (curvature term
  + proper BCs + multi-species + energy batch) is now an engineering build, not a
  feasibility gamble.
* **(B) geomagnetic — largely RESOLVED.** Primary-folding done in
  `mceq3d_solver`/`mceq3d_production` (no x_eff hack); the analytic Störmer is
  replaced by first-principles **trajectory back-tracing** in an IGRF-dipole
  field (`geomag_backtrace.py`, validated vs Störmer); **muon bending** added
  (`muon_bending.py`). Remaining: full-IGRF higher Gauss terms (back-tracer
  ready), and the coherent charge-dependent muon-bending shift wired spatially.
* **(B) geomagnetic — now essentially complete.** Full **IGRF** (degree 13, via
  ppigrf) back-traced cutoff (`geomag_backtrace.py`; Kamioka 11.31 GV), a
  vectorized sky `cutoff_map`, and an **integrated directional flux**
  `Phi(E, zenith, azimuth)` (`directional_flux.py`; Kamioka 1 GeV E-W W/E=2.17)
  folding the back-traced cutoff into the primary + cascade + muon bending. (A
  sign/azimuth-convention bug in the dipole path was found and fixed via the
  East-West validation.)
* **(E/off-axis) spherical streaming — built.** `spherical_streaming.py` solves
  the 1-D spherical P_N transport with the **curvature term** `(1-mu^2)/r d/dmu`
  (the piece the slab prototype omitted) on a staggered grid; the P_1 limit
  matches an independent spherical-diffusion ODE to machine precision (0.0000),
  it stays sparse & ~linear, and it redistributes the detector flux toward the
  horizon. The **coherent charge-dependent muon-bending E-W shift** is delivered
  in `muon_bending.py` (mu± ~±3°, ~3° ν/ν̄ split sub-GeV).
* **(off-axis coupling) done.** `spherical_cascade.py` runs the cascade down the
  curved line of sight per direction (depth-resolved, local density), giving the
  absolute `Phi_nu(E, cos zenith)`: near-isotropic sub-GeV, sec θ horizon
  enhancement at high E with finite curved saturation (≈5× at 2 TeV vs the
  divergent sec θ=25). The genuinely-3D residual (inter-direction streaming)
  is small (~1–2%). The clarifying physics finding: the large directional
  structure is the sec θ effect (high-E, 1D-per-direction), **not** a sub-GeV
  term — sub-GeV 3D is dominated by geomagnetics + muon bending.
* **High-statistics kernels done** (cluster: UrQMD34+Sibyll23d, 200k evt/pt,
  spliced; installed as the drop-in moments). See `KERNEL_PRODUCTION_REPORT.md`.
* **(D) Honda cross-check — done.** `validate_honda.py` parses the Honda
  HKKM2014 azimuth-dependent Kamioka table and matches this work on the
  directional observables: with the production-cone-averaged cutoff, at a
  *matched* near-horizon zenith (cosZ 0.25, ~75°) the East–West amplitude sits
  ≈11–14 % above Honda across 0.4–2 GeV (1.85 Honda vs 2.10 this work at
  1.1 GeV; correct sign/peak/vanishing), a systematic overshoot growing toward
  the extreme horizon (out-of-sample: ~6 % at 63° → ~26 % at 87°, cone truncates
  at the limb); sec θ horizon 2.19 vs 2.33 at 100 GeV.
* **(full-shape S_N) resolved** — real kernel shape ≡ same-variance Gaussian, so
  the variance-only FP input suffices (no full-shape kernel needed for the spread).
* **Still open:** absolute per-bin Honda reproduction (needs their full setup);
  NA61 K± HEPData fit (network).

The body below is the original review; read it together with the statuses above.

## 0. Two separate deliverables

1. **`daemonflux.geomagnetic`** (in the installed package) + `examples/geomagnetic_example.py`
   — an *analytic* geomagnetic admittance layer bolted onto the daemonflux 1D
   flux. Fully backward compatible (no model ⇒ identical 1D).
2. **`tools/mceq3d/`** — a research pipeline that regenerates production kernels
   with transverse momentum and builds a deterministic 3D angular treatment.
   This is prototype/research code, *not* part of the daemonflux package.

These were built largely independently and only joined at the very end
(`unified_3d_flux.py`); see issue (L).

---

## 1. What each piece does + its main simplification

| module | does | main simplification / cheat |
|---|---|---|
| `daemonflux/geomagnetic.py` | Störmer cutoff `R_c(λ,θ,ξ)` + smooth admittance `G(E,θ,φ)`; multiplies 1D flux | **(B)** cutoff applied to *lepton* energy via single `x_eff=0.1` and to the *final* flux (not folded into the primary before the cascade); dipole (not IGRF); site λ hand-tuned; `G` cancels in ratios ⇒ no charge-ratio E–W |
| `kernel_regeneration.py` | re-runs SIBYLL/UrQMD via chromo → `d²N/dx_L dp_T`; consistency gate vs MCEq | **(A)** absolute normalization vs MCEq unresolved (~4.4×); target = N₂ `(14,7)` not air mix |
| `angular_kernel.py` | moments `⟨θ⟩,⟨θ²⟩`; `D_θ`; binned angular rows | gridless moments are clean; binned-row `discrete_ordinate_row` aliases (cosmetic, superseded) |
| `splice_kernels.py` | merge low-E (UrQMD) + high-E (SIBYLL) at a transition energy | hard switch at 65 GeV, no overlap blending |
| `validate_na61.py` | `⟨p_T⟩(p)` vs NA61 p+C 31 GeV/c data | target = C `(12,6)` (matches NA61) — *different from the kernels*, see (G) |
| `fokker_planck_3d.py` | marches angular-diffusion PDE; `σ_θ(E,zenith)` | **(C)** `N_gen = slant/λ`; **(E)** energy is a parameter, not solved; source profile ad hoc |
| `coupled_3d_flux.py` | MCEq 1D flux ⊗ angular spread on the sphere | **(E)** post-hoc convolution, not a coupled solve; **(H)** Rayleigh small-angle kernel; **(I)** up/down symmetry assumed |
| `unified_3d_flux.py` | `Φ₃D/Φ₁D = R(E,θ)·G(E,θ,φ)` | **(L)** product of two separately-validated factors; the product itself is unvalidated |
| `sn_transport.py` | P_N spherical self-convolution, large-angle | **(C)** same `N_gen` proxy; headline uses analytic heat-kernel + forward-Gaussian *shape* assumption; real-kernel path present but fragile/unused (see issue M) |
| `channel_comparison.py` | `⟨θ⟩(E)` for π±, K± | descriptive; K± not data-validated |

---

## 2. Where I simplified or cheated — ranked by how much it matters

**(A) Kernel absolute normalization is unresolved (~4.4× vs MCEq).** The
consistency gate validated only the **shape** of `dN/dx_L` (≈constant ratio over
~3 decades; quantitatively only **52% of populated bins agree within 5%** after
removing the constant — statistics + low-x feed-down). I showed empirically the
4.4 is *not* resonance feed-down (tracked-stable changed the count ~3%), so it is
an MCEq storage-normalization convention I never tracked down. **Consequence: the
regenerated kernels are not usable for absolute yields, only shapes/angles.**

**(C) `N_gen = slant_depth/λ` is a non-physical proxy** (both FP and S_N). The
hadronic parent chain that sets a neutrino's angular spread is *finite* (a few
generations: p→π/K→μ→ν), but `slant/λ` reaches ~150 at the horizon. This almost
certainly **over-states the angular randomization**, especially near the horizon
(S_N → ~isotropic). The absolute `σ_θ` magnitudes therefore carry a large model
uncertainty.

**(D) The absolute `σ_θ(E)` values are order-of-magnitude, not quantitative**, and
are **not validated against any 3D reference** (Honda/FLUKA/data). Only the
*scaling* (∝1/E, → 0 at high E, grows toward horizon) is trustworthy.

**(B) The geomagnetic layer is a first-cut scaffold**, not the physics the
daemonflux authors would accept: single `x_eff` lepton→rigidity map, applied to
the final flux rather than the primary spectrum, dipole Störmer instead of IGRF
back-tracing, hand-tuned site latitudes, and no charge-ratio E–W (cancels in
ratios). The E–W *sign and rough size* are right; the magnitudes are not
production-grade.

**(E) There is no self-consistent coupled energy×angle solve.** The "first 3D
solve" (FP) is angular-only with energy as a parameter; `coupled_3d_flux` is a
post-hoc angular convolution of the MCEq 1D flux, not a transport solve of the
coupled system.

**(F) The Honda horizontal (off-axis production-volume) enhancement is absent.**
`slant_depth(curved=True)` only fixes the path-length saturation; the genuinely
3D source-geometry enhancement is not modelled.

**(G) Target inconsistency.** The kernels/moments use nitrogen `(14,7)`; the NA61
validation uses carbon `(12,6)` to match the data. Both ≈ air, but the validated
object is therefore not *exactly* the one feeding the FP/coupled/unified chain.

**(H) Small-angle tangent approximation** in the spherical convolution (Rayleigh
deflection) is marginal at sub-GeV where `σ_θ ~ 40–50°`.

**(I) Up/down symmetry** `Φ(cosθ)=Φ(−cosθ)` assumed in `coupled_3d_flux` (valid
for the conventional GeV flux without oscillations/Earth absorption).

**(J) Hand-set parameters, not fit:** `x_eff=0.1`, `penumbra_width=0.5`, `λ=120`,
`H_ATM_EFF=40 km`, transition=65 GeV.

**(K) Most tests are offline *synthetic* (code-logic) tests, not physics
validation.** See §3 for the few real ones.

---

## 3. Validations actually run (with plot + honest grade)

Three tiers: **[DATA]** = against external measurement; **[ANALYTIC]** = against a
closed-form limit (code correctness); **[INTERNAL]** = self-consistency /
qualitative physics only.

| # | what | tier | result | plot |
|---|---|---|---|---|
| 1 | regenerated `dN/dx_L` shape vs MCEq stored SIBYLL kernel | **[DATA]**-ish (vs MCEq) | shape ≈const ratio over 3 decades; 52% bins <5% after norm; norm 4.4 unresolved | `kernel_real_piplus.png` |
| 2 | MCEq `hadr_yields` is dN/dx in scaling form | [ANALYTIC] | confirmed (energy-independent at fixed offset; physical multiplicities) | — |
| 3 | feed-down is *not* the 4.4× | [DATA] | tracked-stable changes π⁺ count ~3% | — |
| 4 | `⟨p_T⟩(p)` vs **NA61 p+C 31 GeV/c** (π⁺,π⁻) | **[DATA]** | **~10% after matching θ<420 mrad acceptance** (ratio 0.97–1.15) | `validate_na61_pt.png` |
| 5 | direct-angle binning removes the comb artifact | [INTERNAL] | smooth vs spiky shown | `angular_smoothness_demo.png` |
| 6 | gridless moments: `⟨θ⟩∝1/E`, `D_θ→0` high-E | [INTERNAL] | clean 1/E, no floor | `m_spliced.png` |
| 7 | FP solver vs analytic Gaussian (top-source) | [ANALYTIC] | ~2% | `fokker_planck_3d.png` |
| 8 | `σ_θ(E,zenith)`: →0 high-E, grows to horizon | [INTERNAL] | qualitatively correct; magnitudes unvalidated (C,D) | `fokker_planck_3d.png` |
| 9 | coupled `Φ₃D/Φ₁D`→1 high-E, ~5–20% sub-GeV | [INTERNAL] | recovers 1D; zenith smearing | `coupled_3d_flux.png` |
| 10 | unified E–W: R cancels in W/E ratio | [ANALYTIC] | W/E = G's W/E (R azimuth-flat) | `unified_3d_flux.png` |
| 11 | P_N vs small-angle: agree high-E, P_N bounded sub-GeV | [ANALYTIC/INTERNAL] | matches FP high-E, saturates 90° | `sn_transport.png` |
| 12 | channels: K ~40% wider angle than π | [INTERNAL] | shown; K not data-validated | `channel_comparison.png` |
| 13 | Kamioka cutoff ~11 GV; West<East | [DATA]-ish (vs literature) | 11.3 GV; W/E asymmetry correct sign | `geomagnetic_example.png` (reproducible) |

**Bottom line on validation:** only #4 is a clean quantitative match to
independent data. #1 validates shape (not normalization) against MCEq. Everything
about the *absolute* 3D angular magnitudes (#8–#12) is internal/qualitative.

---

## 4. Leftover issues caught in this review (action items)

1. **Reconcile the 4.4× kernel normalization** against MCEq's matrix-assembly
   code before any absolute use (issue A).
2. **Replace `N_gen = slant/λ`** with a finite hadronic-parent-chain count; this
   is the single biggest fix for the 3D-spread magnitudes (issue C). Until then,
   treat all `σ_θ` numbers as indicative.
3. **Validate `σ_θ`/`Φ₃D` against Honda-3D or FLUKA** — currently nothing pins the
   absolute angular magnitudes (issue D).
4. **Geomagnetic: move the cutoff into the primary spectrum and use IGRF
   back-tracing**, add charge dependence (issue B). The current layer is the
   "scaffold" promised at the start, not production.
5. **Unify the target** (use an air mix, or carbon throughout) so the validated
   object matches the one used downstream (issue G).
6. **`sn_transport.py` carries an unused, fragile real-kernel path**
   (`single_production_f1` + numerical Legendre) that the headline plot bypasses
   in favour of the analytic heat-kernel; either harden it (denser-near-pole grid,
   empty-row fallback) or remove it (issue M).
7. **No coupled energy×angle solve yet** — the "3D solve" is angular-only; a real
   MCEq-style coupled matrix march is the next genuine milestone (issue E).
8. **Honda horizontal excess still missing** (issue F).
9. Minor: `arccos⟨cosθ⟩` saturates at 90° while the RMS isotropic angle is 98°
   (two different spread measures in the narrative); the spherical convolution's
   Rayleigh kernel is small-angle (issue H); up/down symmetry assumed (issue I).

## 5. What you can trust vs not

* **Trust:** the *method* and pipeline (regenerate kernels → validate vs MCEq
  shape & NA61 ⟨p_T⟩ → moments → angular transport); the qualitative results
  (3D matters below ~2 GeV; ⟨θ⟩∝1/E; kaons wider; E–W sign; everything →1D at high
  E). The NA61 ⟨p_T⟩ match (~10%) is solid.
* **Don't trust (yet):** any *absolute* number — kernel normalization, absolute
  `σ_θ` magnitudes, the near-horizon randomization, the geomagnetic suppression
  magnitudes, and the unified `Φ₃D/Φ₁D` values. These are prototype-level and
  hinge on the unfixed items (A), (C), (B).
