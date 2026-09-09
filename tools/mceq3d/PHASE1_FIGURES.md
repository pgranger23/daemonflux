# Phase-1 figure set (`figures_phase1/`)

Thirteen figures for the Phase-1 audit and repair of the 3D / geomagnetic
extension. The authoritative narrative and every quoted number is
`PHASE1_RESULTS.md`; this file says what each PNG shows, how to read it, which
number it illustrates, and where the data came from.

Regenerate everything with

```bash
cd tools/mceq3d
export PY=/cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc14-opt/bin/python3.12
export PYTHONPATH=/afs/cern.ch/work/p/pigrange/daemonflux/.venv/lib/python3.12/site-packages:/afs/cern.ch/work/p/pigrange/daemonflux/src:$PWD
$PY make_phase1_figures.py            # all 13, ~20 s
$PY make_phase1_figures.py 03 07      # selected
```

Only figure 12 needs the engine. Its solve is cached in
`figures_phase1/data/grid_delivered.npz`; to rebuild it (≈ 7 min on warm
`.cache3d`) run `$PY figures_phase1/data/solve_grid.py` first. Everything else
is transcribed from the Phase-1 session logs or read from repository `.npz`
files; the small array extracts under `figures_phase1/data/` were pulled out of
the (temporary) session scratchpad once by
`figures_phase1/data/extract_scratchpad.py`, so the set is self-contained.

## Reading conventions used everywhere

The whole set shares one palette and one set of roles, so a colour means the
same thing in every figure.

| role | colour | used for |
|---|---|---|
| repaired / delivered / v2 / sharp | blue `#2a78d6` | the state of the engine after Phase 1 |
| paper-era / legacy | orange `#eb6834` | 13-node map, `arctan` moments, σ=0.5, detector anchor |
| third variant | aqua `#1baf7a` | the remaining arm of a three-way comparison |
| ground truth | near-black `#0b0b0b` | direct back-traces, measured ladders, NA61 data |

Where the *species* is the thing being distinguished, the four slots are fixed
instead: ν<sub>μ</sub> blue, ν̄<sub>μ</sub> orange, ν<sub>e</sub> aqua,
ν̄<sub>e</sub> violet `#4a3aa7`. Honda is then a dashed black line with open
circles, and the detector-anchored model a dotted line in the species colour.
The palette was validated against the light chart surface with the dataviz
validator (`--pairs all`, worst CVD ΔE 9.2, worst normal-vision ΔE 16.3); aqua
is below 3:1 contrast, so every aqua series carries a legend entry or a direct
label. Each PNG has its own source line printed underneath it.

Azimuths are always **compass** (0 = geographic North, clockwise), so Honda's
tables are mapped with `az_compass = (180 − az_Honda) mod 360`
(arXiv:1102.2688 § II). Geomagnetic East at Kamioka is 81.88°, marked with a
grey rule where relevant.

---

## 01 — `01_cutoff_scan.png` · the two numerical defects in the cutoff map

**Shows.** (a) The East-side cutoff `R_c(zenith)` at azimuth 90° from three
sources: the direct back-trace (black crosses), the delivered dense-node map
(blue) and the old 13-uniform-node map (orange). (b) The full 0.1 GV rigidity
ladder at the vertical — 1 = the trajectory escapes, 0 = it is forbidden.

**How to read.** In (a) the blue line sits on the crosses everywhere; the orange
line bulges above them through the whole near-limb rise and then *collapses*
past 89°. In (b) the admittance is not a single step: there are two isolated
allowed islands at 9.55–10.05 GV and near 10.55 GV before the ladder becomes
permanently allowed above 11.45 GV. A coarse top-down scan stops at the first
allowed rigidity it meets and therefore lands inside an island.

**Numbers illustrated.** Legacy bilinear interpolation error at azimuth 90°:
+1.68 GV at 84°, **+3.26 GV at 87°**, and −4.82 GV at 89.5° (the 89° seam
stitching the down-going map to the antipodal far-side cutoff); the dense-node
map is within **0.06 GV** of the direct trace everywhere. Vertical cutoff:
**8.79 GV** from the working-tree scan at audit time versus **11.369 GV** from
the coarse-ladder + bisection scheme (literature 11.3–11.5 GV). Highest
forbidden rigidity at the vertical `R_U` = 11.45 GV, band width 0.70 GV,
2 islands.

**Source.** (a) `scratchpad/buildmap.log` (`diag_cutoff_buildmap.py`).
(b) `scratchpad/pen/penumbra.npz` (72 traces at 0.1 GV steps) and
`scratchpad/pen/scan.log`. No solve.

---

## 02 — `02_penumbra.png` · the penumbra is not 0.5 in ln R

**Shows.** The measured admittance ladder around `R_U`, rescaled to `R/R_U`, for
the vertical and for 87° East, with the legacy erf transmission
(σ<sub>lnR</sub> = 0.5, orange) and the delivered sharp bin-averaged step
(σ<sub>lnR</sub> = 0, blue) drawn on top.

**How to read.** At 87° East the back-traced ladder *is* a step: nothing is
admitted below `R_U`, everything above. The legacy erf, by contrast, is still
transmitting at half strength at the cutoff itself and does not reach zero until
well below it — the shaded band is the region it wrongly admits. The vertical
(left) is the one direction with a real penumbra, and even there the structure
is two narrow islands, not a Gaussian smear.

**Numbers illustrated.** Measured 16/84 width σ<sub>16/84</sub> = **0.007** at
87° East (0.0073–0.0140 at the East and North horizon directions; the
low-cutoff West and South directions, which do have islands, reach
0.043–0.065) against **0.13** at the
vertical; the default was changed **0.5 → 0.0**. Integrated over an E<sup>−2.7</sup>
primary spectrum the legacy erf puts **45 %** of the admitted flux below `R_U`
(computed in the script, not transcribed). Sharpening the cutoff drops `G` at
the 41.7 GV East cutoff by 12.5 % at 0.5 GeV and raises every species' W/E by a
nearly uniform 12–15 points (`PHASE1_RESULTS.md` § 3.2, § 4).

**Source.** `scratchpad/pen/penumbra.npz` + `penumbra.json`
(`diag_penumbra_width` / `scan_penumbra.py`, 1013 traces at 0.1 GV);
transmission curve from `mceq3d_flux._transmission`. No solve.

---

## 03 — `03_moments_arcsin.png` · the `arctan` → `arcsin` production-angle fix

**Shows.** (a) The RMS production angle √⟨θ²⟩ of π<sup>+</sup> versus secondary
energy at a 20 GeV projectile, old (`m_spliced.npz`) versus v2
(`m_spliced_v2.npz`). (b) Their ratio, with the pure `arcsin/arctan` factor
applied to the same RMS as a grey reference. (c) The mean production angle
⟨θ⟩(p<sub>lab</sub>) against the NA61 p+C 31 GeV/c measurement, old and v2.

**How to read.** Panel b separates two effects: the grey curve is what the bug
fix *alone* would do (≤ 1.37 at the softest secondaries, → 1 above ~1 GeV); the
blue curve is what the delivered v2 set actually does. The gap between them is
the wholesale regeneration of the moments (UrQMD-3.4 + SIBYLL-2.3d), not the
`arcsin` correction. Panel c is the external closure test: both sets bracket
NA61 to a few percent, and v2 is *not* uniformly better in the mean.

**Numbers illustrated.** `kernel_regeneration.production_angle` used
`θ = arctan(p_T/p)` instead of `arcsin(p_T/p)`, quoted in `PHASE1_RESULTS.md`
§ 1 as biasing the generator moments **~16 % too narrow at 0.3 GeV**. Note the
caveat the figure makes visible: the *pointwise* v2/old ratio at the relevant
secondary energies is 1.2–1.6, larger than 16 %, because the v2 files were
regenerated from scratch rather than merely re-projected — the 16 % is the
folded effect on the delivered cone, not the ratio of the two moment tables.
NA61 mean ⟨θ⟩ ratio for π<sup>+</sup>: **v2 1.043, old 1.020** (π<sup>−</sup>:
0.967 / 0.945).

**Source.** `m_spliced.npz`, `m_spliced_v2.npz` (repository);
`scratchpad/na61_angle.log` (`validate_na61_angle.py`, 120k events). No solve.

---

## 04 — `04_cone_widths.png` · the channel-resolved cones

**Shows.** The space-angle RMS of each parent channel's cone versus E<sub>ν</sub>
in the delivered (v2) engine — direct π→μν, kaon, and the muon-decay cone split
by daughter flavour — with the old flavour-blind muon-decay cone overlaid.

**How to read.** The kaon cone is a factor 2.5–3.2 wider than the pion cone at
every energy, which is why treating the two channels together was wrong. The two
muon-decay curves (ν<sub>μ</sub> and ν<sub>e</sub> daughters) differ by ~6–7 %,
small but systematic. Against the old flavour-blind curve the corrected cone
crosses over: **wider** below ~0.4 GeV and **narrower** above ~1 GeV, which is
why fixing the cone alone (stage S1 in figure 06) moved the residual the wrong
way before the joint integral compensated.

**Numbers illustrated.** At 0.2 / 0.3 / 0.5 / 1 / 3 GeV: σ<sub>π</sub> =
26.75 / 18.72 / 11.94 / 6.48 / 2.46°; σ<sub>K</sub> = 67.05 / 48.68 / 32.52 /
18.82 / 7.90°; μ-decay → ν<sub>μ</sub> (with bend) 26.66 / 21.54 / 16.47 /
11.44 / 6.42°; μ-decay → ν<sub>e</sub> 24.85 / 20.12 / 15.41 / 10.74 / 6.06°;
old flavour-blind μ-decay 26.45 / 19.76 / 13.77 / 8.83 / 5.20°. The related
open item in `PHASE1_RESULTS.md` § 5 (W3) quotes the corrected μ cone as 34.6°
against the old 26.5° at 0.2 GeV — a different (pre-bend, pre-realisation)
definition of the same quantity; the values plotted here are the ones the
delivered engine uses.

**Source.** `offaxis_excess_channel_v2.npz` (`sigma_pi`, `sigma_k`,
`sigma_mu_numu`, `sigma_mu_nue`, repository); old curve from
`scratchpad/w2/cone_widths.log` row `(c0)`. No solve.

---

## 05 — `05_eoff_species.png` · flat versus channel-resolved off-axis excess

**Shows.** The off-axis production excess `E_off` versus cosZ at 0.3, 0.5 and
1 GeV, for ν<sub>μ</sub> (left) and ν<sub>e</sub> (right). Solid = the delivered
per-species `channel_v2` table, dashed = the old flat table. Colour is energy
(one-hue ordinal ramp, light → dark).

**How to read.** The dashed curves are identical in the two panels — that is the
point: the flat table has no species axis at all. The solid curves separate,
most at the horizon and at the lowest energy, and both lie *above* the flat
table there. Above cosZ ≈ 0.35 the excess turns into a small deficit (the
redistribution taking flux away from the vertical).

**Numbers illustrated.** At cosZ = 0.05 and 0.3 GeV the flat table gives 1.496
while the channel table gives 1.697 (ν<sub>μ</sub>) … 1.767 (ν̄<sub>e</sub>) —
a **4.1 % spread across species** that the flat table cannot represent, falling
to 3.4 % at 0.5 GeV and 1.6 % at 1 GeV. `PHASE1_RESULTS.md` § 5 records that
this table is still only reachable through `joint_cone_factor`'s `G≡1` limit:
`offaxis_factor()` (and hence the `with_eoff_jacobian` NA61 pull) still reads
the flat table.

**Source.** `offaxis_excess.npz` (`E_off`) and
`offaxis_excess_channel_v2.npz` (`E_off_s`), both repository files. No solve.

---

## 06 — `06_stage_ladder.png` · what each repair step bought

**Shows.** The cumulative effect of the Phase-1 steps on two observables.
(a) azimuth-averaged horizon/vertical divided by Honda; (b) the same divided by
Bartol; (c) the ν<sub>μ</sub> 0.5 GeV West/East amplitude, as a deviation from
Honda, at three zenith angles. The x axis is the stage ladder: `S0` repaired map
only, `S1` + moment σ<sub>π</sub> cone, `S1b` + v2 arcsin moments, `S2` + joint
integral, `S4` + channel-resolved joint (the delivered configuration),
`σ=0` + sharp cutoff, `anchor` + production-point cutoff anchor.

**How to read.** In (a)/(b) the horizontal line at 1.00 is perfect agreement.
Almost all of the sub-GeV H/V deficit is closed in one step — `S2`→`S4`, the
channel-resolved joint integral — and it overshoots Bartol while landing on
Honda. Panel (c) tells the opposite story: the same ladder drives the W/E
overshoot to zero at `S4`, and then sharpening the cutoff (`σ=0`) puts 11–13
points straight back. That non-monotonicity is real, not a plotting artefact:
the legacy σ = 0.5 was partly cancelling the E-W overshoot.

**Numbers illustrated.** ν<sub>μ</sub> H/V ÷ Honda at 0.3 GeV:
0.841 → 0.850 → 0.853 → 0.921 → **1.040** → 1.037 → 1.031; ÷ Bartol:
0.915 → … → **1.131** → 1.128 → 1.122 (the 6–13 % Bartol overshoot recorded as
an open item in § 5). ν<sub>e</sub> at 0.3 GeV: 0.842 → 0.857 → 0.886 →
**1.075** → 1.073. W/E deviation, ν<sub>μ</sub> 0.5 GeV, 87°:
+15 % → +13 % → +12 % → +6 % → **+2 %** → +13 % → +12 %.

**Gaps.** No ν<sub>e</sub> H/V was measured at the anchor stage
(`anchor/hv_after.log` covers ν<sub>μ</sub> only), so those two lines stop at
`σ=0`; the anchor `/Bartol` points are the measured `/Honda` values rescaled by
the published Honda/Bartol ratio. In (c) the 81° `σ=0` and 76° `anchor` cells
are not tabulated in their respective logs, which is why those lines break.

**Source.** `scratchpad/jc/stages_s4.log` (S0–S4 and the Honda/Bartol
references), `scratchpad/pen/table_all.txt` (σ=0), `scratchpad/anchor/
hv_after.log` and `ewz_after.log` (anchor). No solve.

---

## 07 — `07_azimuth_pattern.png` · the open species-pattern problem

**Shows.** Flux versus arrival azimuth on Honda's own 12 bins at cosZ = 0.05 and
E<sub>ν</sub> = 0.5 GeV, normalised to its own azimuth average so only the
*shape* is compared. Top four panels: one species each, Honda (dashed black,
open circles) against the delivered production-point-anchored model (solid) and
the detector-anchored model (dotted). Bottom row: all four Honda curves overlaid
(d) and all four model curves overlaid (e).

**How to read.** Panels d and e are the figure. Honda's four species fan out —
ν<sub>e</sub> has by far the deepest East minimum and highest West maximum,
ν̄<sub>e</sub> the shallowest — while the model's four curves lie almost on top
of one another. The per-panel `max/min` box quantifies each pair. The detector
versus production-point anchor (dotted versus solid) changes the *phase* of the
pattern, visibly around North and South, but not its amplitude.

**Numbers illustrated.** `max/min` at cosZ 0.05, 0.5 GeV — Honda
**2.51 / 3.81 / 4.74 / 2.12** for ν<sub>μ</sub> / ν̄<sub>μ</sub> / ν<sub>e</sub> /
ν̄<sub>e</sub>, model **2.75 / 2.91 / 3.14 / 2.64**: Honda spans a factor 2.2,
the model a factor 1.19. This is the unresolved open item in
`PHASE1_RESULTS.md` § 5; the single-direction (uncone-averaged) `(W/E)_G`
reproduces Honda's ordering (3.79 / 4.08 / 4.35 / 3.45), so it is the cone
average that destroys the species differentiation.

**Source.** `scratchpad/anchor/pat_after.npz` (production-point anchor) and
`scratchpad/pattern/pat_a.npz` (detector anchor), both produced by
`diag_ew_pattern.py --stage pattern` and read here read-only via the extract in
`figures_phase1/data/pattern_cz005.npz`; Honda from `honda_kam.npz` with the
compass mapping. No new solve.

---

## 08 — `08_charge_split.png` · phase splitting works, amplitude splitting does not

**Shows.** The two Fourier charge-splitting observables — the ν-minus-ν̄
difference in first-harmonic phase Δ(δφ) (top row) and in azimuthal amplitude
Δ(max/min) (bottom row) — versus energy, at 87° (left) and 81° (right), for the
ν<sub>μ</sub>–ν̄<sub>μ</sub> and ν<sub>e</sub>–ν̄<sub>e</sub> pairs. Solid =
delivered engine, dashed with open circles = Honda.

**How to read.** In the top row the solid and dashed lines of the same colour
sit close together at 0.3–0.5 GeV — the model reproduces most of Honda's phase
splitting and gets the sign right for both pairs. In the bottom row the solid
lines are pressed against zero while Honda's are several times larger: the
amplitude splitting, which is what the retracted paper claim measured, is
largely missing. Honda's phase curves cross zero and change sign above ~1 GeV
while the model's stay flat; that divergence is a separate, unexplained feature
of Honda's table at these energies.

**Numbers illustrated.** At 87°, 0.5 GeV: Δ(δφ) ν<sub>μ</sub>-pair
**+4.30° model vs +4.36° Honda**, ν<sub>e</sub>-pair **−8.53° vs −9.21°**;
Δ(max/min) ν<sub>μ</sub>-pair **−0.16 vs −1.30**, ν<sub>e</sub>-pair
**+0.50 vs +2.62**. The phase splitting is 60–90 % of Honda's, the amplitude
splitting 13–24 % (§ 3.3; the anchor flip improved the phase half only,
3.52 → 4.30 and −7.32 → −8.53).

**Source.** `scratchpad/anchor/fourier_after.log`
(`diag_ew_charge_fourier.py`, `cutoff_anchor='prod_point'`). No solve.

---

## 09 — `09_bend_scale.png` · how much bending would be needed

**Shows.** (a) The W/E amplitude of each species as the charge-signed muon
bending displacement is scaled by a factor 0 → 8, with Honda's measured values
as horizontal dashed lines in the same species colour. (b) The two ratio
observables — ν<sub>e</sub>/ν̄<sub>e</sub> and ν̄<sub>μ</sub>/ν<sub>μ</sub> W/E
amplitude — against the same scale. The dotted vertical rule marks scale 1, the
physically correct value.

**How to read.** At scale 0 the four species are nearly degenerate
(2.97 / 2.97 / 3.10 / 2.88 — the small residual spread is the
flavour-dependent cone width, not the bending). Turning the bending on splits them in the right
*direction* — ν<sub>e</sub> and ν̄<sub>μ</sub> up, ν<sub>μ</sub> and
ν̄<sub>e</sub> down, matching Honda's ordering — but at scale 1 the splitting is
a small fraction of what is needed. Reading across to where each solid curve
meets its dashed line gives the scale that would be required.

**Numbers illustrated.** At scale 1 the model gives
**2.886 / 3.090 / 3.322 / 2.706** against Honda's
**2.507 / 3.812 / 4.740 / 2.117**; the ν<sub>e</sub>/ν̄<sub>e</sub> ratio is
1.228 against Honda's 2.239 and only reaches it near scale **5–8**. Figure 10
shows independently that the physical bend is if anything ~3–14 % *smaller*
than the delivered 5.08°, so this scale factor is not available.

**Source.** `scratchpad/pattern/channels_tables.txt`, block `cz 005`, section C
(`diag_ew_pattern.py --stage channels`). Note the task brief pointed at
`shiftscan.txt`; that file holds the `d ln N_μ / d(shift)` derivative scan, not
the 0–8 amplitude scan, which lives in `channels_tables.txt`. No solve.

---

## 10 — `10_muon_segment.png` · the bending angle is a distribution, not a shift

**Shows.** The distribution of the muon's total bending angle at decay from the
backward muon-segment Monte Carlo (30 000 samples per point, continuous dE/dx,
full IGRF-13 along the muon's own trajectory) at zenith 87°, for geomagnetic
East and West, ν<sub>μ</sub>-type daughter at E<sub>ν</sub> = 0.5 GeV. The
dashed black curve is a pure exponential with the measured mean; the dotted
orange rule is the single shift the engine actually applies.

**How to read.** The histogram follows the exponential almost exactly, which is
the point: the bend is `qBT_proper/m` with `T_proper` exponentially distributed,
so the *mean* is a fair summary but half the muons bend by less than 3.3°. The
delivered single shift sits above the median and close to the mean — a
reasonable, slightly generous, one-number stand-in. The grey ticks at the bottom
mark p16/p50/p84.

**Numbers illustrated.** Mean bend **4.76°** East / **4.92°** West at 87°
(p16/p50/p84 = 0.84 / 3.29 / 8.64° East) against the delivered single shift
**Δ = qBτ/m = 5.08°**. `PHASE1_RESULTS.md` § 4 records that propagating the full
MC instead of the single shift is a **−3 % to −14 %** correction at 75–87° — the
wrong sign for the energy-loss-enhanced-bending hypothesis, which is refuted.

**Source.** raw samples `scratchpad/msgrid_ray_n30000_s3.npz`
(`muon_segment_mc.py`, extracted to `figures_phase1/data/muonseg_bend.npz`);
statistics and the 5.08° shift from `scratchpad/muonseg.log` section B; the
W/E consequences from `scratchpad/muonseg_D2.log`. No solve.

---

## 11 — `11_anchor_pattern.png` · where the back-trace is launched

**Shows.** `R_c` versus arrival azimuth (24 nodes, 15° apart) at zenith 80° and
87°: the old detector-anchored map (orange dashed), the delivered engine's
production-point-anchored family (blue), and direct back-traces launched from
the production point at h = 30 km (black crosses). The grey rule is geomagnetic
East (81.9°).

**How to read.** The blue curve lies on the crosses; the orange curve is
systematically too high across the whole East lobe and too low in the South.
The detector map is anchored several degrees of geomagnetic latitude away from
where the primary actually enters the atmosphere, and the effect grows with
zenith angle — compare the two panels. The inset table gives the North/South
contrast and the first-harmonic phase, the two summary numbers the anchor was
introduced to fix.

**Numbers illustrated.** At 87°: N/S **2.32 → 1.59** (direct 1.65) and
first-harmonic phase **61.7° → 72.0°** (direct 72.4°); at 80°, 1.83 → 1.55
(direct 1.58) and 66.9° → 72.0° (direct 72.6°). Residual budget at 87° over all
24 azimuths: site interpolation ≤ 1.23 GV (12.7 %), launch-altitude convention
≤ 0.50 GV (6.9 %). The vertical is an exact no-op (0.000e+00 GV) by
construction. Downstream, this moved the model/Honda North/South pair from
0.86–0.92 / 1.07–1.23 to 0.93–1.02 / 0.91–1.05 and the dipole-phase offset from
12–14° to ≤ 1.3° for all four species (§ 6.3).

**Source.** detector curve from `scratchpad/dipole/prodpoint.log`; engine and
direct curves from `scratchpad/anchor/validate.log` section 3
(`diag_cutoff_anchor.py`). No solve.

---

## 12 — `12_gridwide.png` · grid-wide agreement (**the one live solve**)

**Shows.** log<sub>10</sub>(model / Honda), azimuth-averaged, over the full
(E<sub>ν</sub>, cosZ) grid for ν<sub>μ</sub> and ν<sub>e</sub>, on Honda's own
10 cosZ × 12 azimuth down-going bins. Diverging blue↔red scale with a neutral
midpoint at zero, clipped at ±0.08 (arrows mark the out-of-range cells).

**How to read.** Blue = the model is below Honda, red = above. The map is
mostly pale blue: a broad, smooth few-percent deficit rather than any localised
structure, with a near-zero band around 1–2 GeV and the strongest deficits at
the horizon below 0.3 GeV and in the 3–30 GeV range. The leftmost column is the
engine's energy-grid edge (the engine grid starts at 0.089 GeV) and is not
meaningful. The boxes give the median and 90th percentile of |log₁₀ ratio| over
**all 12 azimuths cell by cell**, not over this azimuth-averaged map.

**Numbers illustrated.** Per cell over 0.1–100 GeV × 10 cosZ × 12 az this run
gives median / 90th percentile **0.033 / 0.062** (ν<sub>μ</sub>) and
**0.032 / 0.092** (ν<sub>e</sub>), reproducing the published
`anchor/full_after.log` section G values 0.032 / 0.061 and 0.032 / 0.090 and
confirming the solve is the delivered configuration. On the azimuth-averaged
absolute-ratio table (section A), the pre-repair run of 2026-09-03 spans
**0.84–1.10** and the delivered engine spans **0.90–1.10** — the "paper-era band
versus now" comparison. (The brief quoted 0.82 for the lower edge; the value
measured in `scratchpad/diag_full_comparison.log` is 0.84, which is what is
stated here.)

**Source.** live solve —
`figures_phase1/data/solve_grid.py`, a bare
`MCEq3DFlux(base_model="hybrid", primary=("GlobalSplineFitBeta", None),
daemonflux_location="kamioka").solve(36.43, 137.31, cz, az, use_cache=True,
cache_dir=".cache3d", date=2020-01-01)` on Honda's grid, 436 s on warm caches,
cached to `figures_phase1/data/grid_delivered.npz`. Reference `honda_kam.npz`.

---

## 13 — `13_conservation.png` · the joint cone is not a pure redistribution

**Shows.** The down-going solid-angle average of the joint production factor
evaluated at `G ≡ 1` (geomagnetic cutoff switched off), versus E<sub>ν</sub>,
for all four species. Dashed lines mark Bartol's own +3 % and Honda's −2 %; the
grey band between them is the range the two published models occupy.

**How to read.** If the off-axis cone only *moved* flux between directions, this
average would be exactly 1.000 at every energy. It is not: below ~0.5 GeV it
rises above the band, so the delivered cone adds net production. All four
species track each other to better than 0.5 %, so this is a property of the
cone geometry, not of any flavour channel.

**Numbers illustrated.** The average is **1.065–1.069 at 0.2 GeV** and
**1.046–1.053 at 0.3 GeV**, falling to 1.008–1.009 at 1 GeV and 1.000 by 3 GeV
— opposite in sign to Bartol's +3 % and Honda's −2 %. This is the quantitative
form of the open item in `PHASE1_RESULTS.md` § 5: *"H/V 6–13 % above Bartol with
a net sub-GeV solid-angle excess ~5 %"*, with the likely cause per W3 being that
the corrected muon-decay cone is now slightly too wide (see figure 04).

**Source.** `scratchpad/jc/stages_s4.log`, conservation table
(`diag_joint_stages.py`, S4 configuration). No solve.

---

## Coverage and caveats

* **Live solve:** figure 12 only (436 s). Every other figure is built from the
  Phase-1 logs or from repository `.npz` files.
* **Figure 06** is missing two ν<sub>e</sub> points (no anchor-stage H/V was
  measured for ν<sub>e</sub>) and two W/E cells (not tabulated at those
  stages); the gaps are marked in the figure and explained above rather than
  filled by interpolation.
* **Figure 09** uses `channels_tables.txt` rather than `shiftscan.txt`, which
  does not contain a 0–8 bend-scale scan.
* **Figure 03** deliberately does *not* claim to show the "16 % too narrow"
  number directly; see its entry.
* Nothing in this set modifies a tracked file. `make_phase1_figures.py`,
  `PHASE1_FIGURES.md` and everything under `figures_phase1/` are new.
