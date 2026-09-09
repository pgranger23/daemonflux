# Phase 1 results — 2026-09-03 audit and 2026-09-04 repair (`tools/mceq3d`)

Factual record of the critical audit and the same-branch Phase-1 fixes it triggered.
Every number below is quoted from `scratchpad/audit_report.md`, the seven Phase-1
agent reports (`scratchpad/tasks/{w1..w7}` transcripts, referenced below by their
short label), or the logs `scratchpad/jc/stages_s4.log` and
`scratchpad/pen/{table_all.txt,ewtable.txt}`. Nothing here is invented; where a
number could not be traced it is omitted.

## 1. What was wrong (2026-09-03 audit)

`REVIEW.md`'s "CURRENT STATUS" block made two headline claims. Both are **retracted**:

1. *"The one systematic that remains ... the factorised cone-average under-softens
   the sharp back-traced cutoff relative to Honda's full 3D treatment."* The audit
   found this was dominated by **numerical defects**, not an intrinsic softening:
   a coarse rigidity scan (20-24 pts to 40-55 GV, +-1 GV/cell quantisation), bilinear
   interpolation across the convex near-limb rise (+2.6 to +4.2 GV at 84-88deg E),
   a 9.5 GV seam at 89deg stitching the down-going detector cutoff to the antipodal
   far-side cutoff, and production-point displacement measured ~2.5x too small.
   Working-tree defaults at audit time (`r_hi=55` without a matching `n_scan`) made
   it *worse*: Kamioka vertical cutoff came out 8.79 GV instead of ~11.5 GV.
2. *"Confirmed structural gap: charge-dependent East-West"* (model predicts ~zero
   nu-vs-nubar E-W amplitude split; Honda shows a factor 2-3). **Unsupported**: the
   diagnostic (`max/min` over a uniform azimuth grid) is exactly even under a sign
   flip of the shift (355.234 vs 355.234 at +-30deg, audit 3.4.1); the coupled march
   had a 10x gyroradius unit bug (gauss read as tesla, 1172deg/checkpoint at
   0.3 GeV/87deg) plus a non-bijective `argmax` resample; and the map's 89deg seam
   moved both charges to a *lower* cutoff at 87deg E, killing the mechanism exactly
   where it was tested.

Other defects found (audit sections 3.2, 3.5, 3.6): the geomagnetic cone in
`cone_geff` used the un-corrected `k_spliced.npz` kernel and `sigma` instead of
`sigma/sqrt(2)` (41% too wide in the affected path); the joint production x cutoff
covariance `Cov_cone(p,G)` was never computed (product of two separately-averaged
factors instead of one integral); `kernel_regeneration.py` used
`theta=arctan(p_T/p)` instead of the correct `arcsin(p_T/p)`, biasing the generator
moments ~16% too narrow at 0.3 GeV; `cutoff_grid`'s default `n_scan=12` silently
saturated cached maps at 40 GV.

## 2. What was changed

| step | module / function | default changed |
|---|---|---|
| 1 | `geomag_backtrace.scan_upper_cutoff`, `cutoff_from_states` | coarse top-down ladder (step <=1 GV) + bisection to 0.1 GV, replacing the fixed linear `n_scan`; `r_hi` 40->55 GV |
| 2 | `mceq3d_flux._zenith_nodes` (`LIMB_ZENITH_NODES`), `finemap_rc` | dense nodes at 82/84/85.5/87/88/88.5/89/89.5deg; interpolation error at 84-88deg E drops from +1.7..+3.3 GV to <=0.06 GV |
| 3 | `mceq3d_flux.cone_geff(sublimb="prod_point")` (new default) | down-going cone samples transformed into the production-point frame (`H_PROD_KM=30`) and read off the down-going map only; Earth-shadowed samples blocked in numerator *and* normalisation instead of reading the antipodal far-side cutoff (`sublimb="farside"` kept as legacy A/B) |
| 4a | `offaxis_mc.production_profile(species=...)`, `cone_numden_multi`, `build_channel` | per-species channel-resolved (`{s}_dir`/`{s}_k`/`{s}_mu`) off-axis excess, replacing the single flavour-blind pion cone |
| 4b | `kernel_regeneration.production_angle` (new canonical `arcsin` definition), `regen_moments_mp.py` | arcsin bug fixed at every call site; regenerated moments `m_{spliced,piminus,Kplus,Kminus}_v2.npz` (UrQMD-3.4 + SIBYLL-2.3d, same recipe as `KERNEL_GENERATION.md`) |
| 3+4 | `joint_cone.py` (new module), `mceq3d_flux.joint_cone_factor` | the single joint integral `J_s/p_axis` replacing the product `E_off x <G_s>_cone`; `joint_cone` defaults **on** whenever `offaxis` and `cone_cutoff` are both set; `joint_channels=True` default uses MCEq's own depth-resolved per-species channel profiles with the corrected `mudecay_shape_mc` cone per flavour |
| 5 | `diag_ew_charge_fourier.py` (new, now the E-W observable) | first-two-harmonic azimuthal decomposition (`a1/a0`, `dphi`, `s1/a0`) replacing `max/min`; Honda's azimuth convention mapped (`az_compass = 180 - az_Honda`, arXiv:1102.2688 § II) |
| 5 | `mceq3d_real._force_checkpoint` | 1e5->1e6 gyroradius unit fix retained; `argmax` nearest-neighbour pull replaced by a row-normalised inverse-square (Shepard) interpolation, exact at zero rotation |
| penumbra | `mceq3d_flux.SIGMA_LNR`, `_transmission` | measured penumbra width in ln R; default changed **0.5 -> 0.0** (analytically bin-averaged over the primary energy bin so sigma=0 stays continuous); `gs_cache_name` keys on sigma |

> **Superseded 2026-09-04 (defaults consolidation).** Both items below were
> subsequently made the defaults: `kinematic_kernel._MOMENTS` is the v2 set
> (old = `_MOMENTS_LEGACY`), `offaxis_factor()` reads
> `offaxis_excess_channel_v2.npz` (`mceq3d_flux.EOFF_TABLE`; flat table =
> `EOFF_TABLE_FLAT`, still selectable via `solve(offaxis_table=...)`), and
> `solve(offaxis=...)` defaults to `True`. A bare
> `solve(lat, lon, cz, az, use_cache=True, cache_dir=..., date=...)` reproduces
> the sigma_lnR=0 column of §3.1/§3.2 below to <=0.0005 (H/V) and <=0.005 (W/E).
> See `ARCHITECTURE.md`. The paragraph below records the state at the time of
> the Phase-1 write-up.

> **Superseded 2026-09-05 (cutoff-anchor default).** `mceq3d_flux.CUTOFF_ANCHOR`
> now defaults to `"prod_point"` (was `"detector"`): every `cone_cutoff` sample
> is read off a cutoff map built at the neutrino's own production point rather
> than at the detector, interpolated from a cached per-site displaced-site map
> family (`MCEq3DFlux.prod_family_rc`). This closes the North/South and
> dipole-phase mismatch identified in the dipole-phase investigation with no
> free parameter; the East-West species pattern and its amplitude splitting are
> untouched (that is the cone average, not the cutoff anchor). See §6 below for
> the full account, `ARCHITECTURE.md` for the module map, and the code comment
> above `CUTOFF_ANCHOR` (`mceq3d_flux.py:383-408`) for the engineering
> trade-off that delayed the flip by one day. `cutoff_anchor="detector"` remains
> available as the fast A/B path; every other `solve()` default is unchanged.

Two things explicitly **not** changed as engine defaults, found while tracing the
code for this record: (i) `solve()`'s `cone_moments=None` still resolves to the
**old, pre-arcsin** `m_spliced.npz`/`m_piminus.npz` (`kinematic_kernel._MOMENTS`);
the v2 moments are only used when a caller passes `cone_moments={"pi": [...v2],
"k": [...v2]}` explicitly, as `diag_joint_stages.py`'s `S1b`/`S2`/`S4` stages do.
(ii) `offaxis_factor()` still reads the **original, un-repaired** `offaxis_excess.npz`
(flat pion-only, July build) — this is the value actually used for down-going
in-range flux whenever `joint_cone=False` (`cone_cutoff=False`, or `joint_cone`
explicitly forced off); when `joint_cone=True` (the default) it is used only as
the fallback for up-going directions and energies outside 0.1-100 GeV, and for the
`with_eoff_jacobian` NA61 pull (`sigma_pi_NA61`), which is therefore first-order
only (ratio of two `offaxis_factor` evaluations), not a re-evaluation of the joint
factor at sigma_pi +-12%.

## 3. Measured effect

### 3.1 Cumulative H/V, azimuth-averaged, ratio to Honda (Bartol in parentheses)

Source: `scratchpad/jc/stages_s4.log` (S0..S4) + `scratchpad/pen/table_all.txt`
(sigma=0, same repaired map/joint-cone/channel/v2-moments configuration as S4,
sigma=0.5 reproduces S4 to 3 decimals).

**nu_mu**

| stage | 0.3 GeV | 0.5 GeV | 1.0 GeV | 3.0 GeV |
|---|---|---|---|---|
| S0 (repaired map only) | 0.841 (0.915) | 0.871 (0.908) | 0.929 (0.911) | 0.984 (0.981) |
| S1 (+moment sigma_pi cone) | 0.850 (0.925) | 0.877 (0.915) | 0.935 (0.916) | 0.985 (0.982) |
| S1b (+v2 arcsin moments) | 0.853 (0.928) | 0.879 (0.917) | 0.935 (0.917) | 0.985 (0.982) |
| S2 (+joint integral, single f_mu) | 0.921 (1.002) | 0.941 (0.982) | 0.981 (0.962) | 1.008 (1.004) |
| S4 (+channel-resolved joint, delivered) | 1.040 (1.131) | 1.019 (1.063) | 1.001 (0.981) | 0.993 (0.989) |
| sigma_lnR=0 (sharp cutoff) | 1.037 (1.128) | 1.013 (1.057) | 0.988 (0.968) | 0.985 (0.982) |

**nu_e**

| stage | 0.3 GeV | 0.5 GeV | 1.0 GeV | 3.0 GeV |
|---|---|---|---|---|
| S0 | 0.842 (0.876) | 0.844 (0.873) | 0.897 (0.851) | 0.992 (0.912) |
| S1 / S1b | 0.857 (0.892) | 0.858 (0.887) | 0.908 (0.861) | 0.998 (0.917) |
| S2 | 0.886 (0.922) | 0.906 (0.936) | 0.957 (0.908) | 1.028 (0.945) |
| S4 (delivered) | 1.075 (1.118) | 1.010 (1.044) | 0.955 (0.907) | 0.975 (0.897) |
| sigma_lnR=0 | 1.073 (1.117) | 1.006 (1.040) | 0.946 (0.898) | 0.962 (0.885) |

Honda/Bartol reference values (`stages_s4.log`): nu_mu 1.825/1.677, 1.448/1.388,
1.241/1.266, 1.346/1.350; nu_e 2.119/2.036, 1.830/1.770, 1.770/1.865, 2.471/2.687
(0.3/0.5/1/3 GeV).

### 3.2 W/E vs Honda, max/min over azimuth (87 / 81 / 76 deg)

Source: `scratchpad/jc/stages_s4.log`.

**nu_mu, 0.5 GeV** — Honda 2.51/2.44/2.28 — S0 +15/+14/+11% -> S1 +13/+10/+9% ->
S1b +12/+9/+9% -> S2 +6/+4/+3% -> **S4 +2/+0/+0%**.

**nu_mu, 1.0 GeV** — Honda 2.09/2.02/1.92 — S0 +19/+16/+11% -> S1 +17/+14/+11% ->
S1b +17/+14/+11% -> S2 +12/+10/+7% -> **S4 +12/+9/+7%** (essentially unchanged
from S2 — see open items).

**nu_e, 0.5 GeV** — Honda 4.74/4.30/3.49 — S0 -37/-32/-23% -> S1/S1b -42/-40/-30%
(the moment-cone steps make it *worse*) -> S2 -37/-34/-25% -> **S4 -40/-37/-28%**.

**nu_e, 1.0 GeV** — Honda 4.90/3.89/2.99 — S0 -46/-33/-20% -> S1/S1b -48/-39/-24%
-> S2 -46/-36/-23% -> **S4 -45/-35/-22%**.

At the penumbra step (`scratchpad/pen/table_all.txt`, all four species, 87/76deg,
0.5 GeV, sigma_lnR default 0.5 -> 0.0): nu_mu +1%/+0% -> +13%/+11%; antinu_mu
-30%/-21% -> -22%/-13%; nu_e -40%/-28% -> -33%/-20%; antinu_e +14%/+11% ->
+27%/+22%. Sharpening the cutoff raises every species' W/E by a nearly uniform
12-15 points; it improves nu_e and antinu_mu (closer to Honda) and worsens nu_mu
and antinu_e (already overshooting) by the same amount.

### 3.3 Charge-splitting observables (87 deg / cosZ 0.05, delivered engine)

Source: `scratchpad/pen/table_all.txt` section C (`diag_ew_charge_fourier.py`,
Honda's azimuth convention applied). nu - nubar difference in the phase (`D(dphi)`,
deg) and amplitude (`D(max/min)`) observables, delivered config (sigma=0.5) vs
sharp cutoff (sigma=0):

| pair, E | Honda D(dphi) | delivered D(dphi) | sigma=0 D(dphi) | Honda D(max/min) | delivered D(max/min) | sigma=0 D(max/min) |
|---|---|---|---|---|---|---|
| numu-antinumu 0.3 | +5.95 | +4.08 | +4.10 | -0.91 | -0.12 | -0.15 |
| numu-antinumu 0.5 | +4.36 | +3.50 | +3.52 | -1.31 | -0.17 | -0.20 |
| nue-antinue 0.3 | -12.22 | -8.14 | -8.12 | +1.69 | +0.39 | +0.46 |
| nue-antinue 0.5 | -9.21 | -7.36 | -7.32 | +2.62 | +0.51 | +0.62 |

The phase splitting (`dphi`) is already 60-90% of Honda's and essentially
sigma-independent (W7). The amplitude splitting (`max/min`, what the retracted
paper claim measured) is 13-24% of Honda's and only weakly helped by the sharper
cutoff.

## 4. Negative results

* **Energy-loss-enhanced muon bending — refuted analytically and numerically**
  (W6, `muon_segment_mc.py`). The in-flight bend angle is `qB*T_proper/m`,
  exactly energy- and dE/dx-independent (`test_bend_is_energy_independent`, 1e-6).
  A full backward Monte Carlo (30k samples/point, continuous dE/dx, full IGRF-13
  along the muon's own trajectory) gives a correction to the delivered 5.08deg of
  **-3% to -14%** at 75-87deg — the *wrong sign* for the hypothesis — plus a
  charge-asymmetric +-20% from coupling the bend to the parent's slant depth
  (which the engine does not model). Propagated through the cutoff map this
  recovers <=8% of the missing nu_e W/E amplitude and none of the nu_mu overshoot.
* **Coupled march — dead end** (W4 section 6, `mceq3d_real.py`,
  `coupled_ew_charge_diag.py`). After fixing the 10x unit bug and the resample,
  a full-hemisphere run (240 directions, real charge-signed Lorentz rotation on
  pi+-/K+-/mu+- at every checkpoint) moves the phase splitting by <=0.3deg and
  `s1/a0` by ~0.001 — a few percent of Honda's, at the level of interpolation
  noise. Structural reason, not numerical: near the horizon the atmospheric
  columns a bent muon moves between are hundreds of km apart, but the operator
  exchanges only direction labels at one altitude; angular-binned operator
  splitting cannot represent that at any checkpoint resolution. Recommendation:
  stop developing this route (the factorised `cone_geff` shift, which re-reads
  the cutoff at the displaced primary direction, is the physically correct cheap
  treatment; a genuine Monte Carlo is Phase 2).
* **Cone-too-narrow (multi-generation spread) — refuted** (audit 3.2, confirmed
  W1/W5). The delivered space-angle RMS (38.9/14.6/8.9/4.1deg at 0.3/0.5/1/3 GeV)
  is consistent with `p_T/E_pi`; extra decay generations add <5% in quadrature.
  Fixing the actual cone defects (moment kernel, `sigma/sqrt(2)`, tail bin) made
  the zenith-shape residual *worse* before the joint-integral and channel steps
  compensated (W1: pion cone `<G>` at 0.3 GeV/87deg E, 0.278 -> 0.193).
* **Penumbra width does not fix H/V or the charge amplitude** (W7). Sharpening
  `sigma_lnR` from the legacy 0.5 to the measured 0.0 explains only the East-side
  *level* (G at the 41.7 GV East cutoff drops 12.5% at 0.5 GeV) — it does not
  touch H/V (Bartol overshoot 1.131 -> 1.128, unchanged) and the charge-splitting
  amplitude improves by only ~21% while remaining ~4x short of Honda; the
  `dlnG/dR_c` slope at the East cutoff moves only +3%, so the weak near-limb
  response is not a penumbra-width effect.

## 5. Open items

* ~~**North/South dipole-phase mismatch**~~ — **CLOSED 2026-09-05** by the
  production-point-anchored cutoff (§6): the model/Honda North/South pair at
  0.5 GeV/cosZ 0.05 moves from 0.86-0.92 / 1.07-1.23 to 0.93-1.02 / 0.91-1.05,
  and the dipole-phase offset from 12-14deg to <=1.3deg, for all four species,
  with no free parameter. It does not touch the item below.
* **nu_e/antinu_e/antinu_mu E-W species pattern** (W6, W7): Honda orders the four
  species' W/E amplitude nu_e > antinu_mu > nu_mu > antinu_e (4.74/3.81/2.51/2.12
  at 0.5 GeV, 87deg); the delivered engine compresses all four onto ~2.7-3.3. The
  single-direction (uncone-averaged) `(W/E)_G` reproduces Honda's ordering and
  rough magnitude (3.79/4.08/4.35/3.45), so the *cone average*, not the bending
  mechanism, destroys the species differentiation — unresolved. The
  production-point cutoff anchor (§6) leaves this pattern essentially unchanged
  (W/E amplitude moves by ~1 point; the four species stay compressed onto
  2.6-3.1 against Honda's 2.1-4.7) and the charge-splitting *amplitude* (Table 4
  of `PAPER_DRAFT.md`) is likewise untouched — only the *phase* half improved.
  **2026-09-09 (§8, classification A/B/C): confirmed as ours** — flat across
  every hadronic model and primary tried (§8.3 Table B) and **now corroborated
  by Super-K data** (§8.4): the flux-level East-West asymmetry pulls *more*
  $\nu_e$ and *less* $\nu_\mu$ split than Honda's own MC, the same direction our
  compression predicts, at $\approx2$-$2.2\sigma$ with no free parameters.
* **nu_mu 1 GeV E-W overshoot** (W3): +12/+9/+7% at S4, essentially unchanged from
  S2; the covariance/channel fixes have faded at 1 GeV, so this residual sits in
  the cutoff values or cascade response, not the factorisation.
  **2026-09-09 (§8.3):** at 0.5 GeV this is mostly a *different input* — the
  primary spectrum removes $-9$ to $-11$ of the +12 points, the hadronic model
  0 to $-5$; at 1-2 GeV the primary removes about half, and a genuine +10-16%
  residual survives every input tried — that part remains open, cause unknown.
* **H/V 6-13% above Bartol with a net sub-GeV solid-angle excess ~5%** (W3): S4
  lands on Honda at 0.3-0.5 GeV but overshoots Bartol by 6-13%; the down-going
  solid-angle average of the joint production factor at G≡1 is 1.046-1.065 at
  0.2-0.3 GeV (`stages_s4.log`), i.e. a genuine net enhancement rather than a pure
  redistribution, opposite in sign to Bartol's own +3% and Honda's -2%. Likely
  cause per W3: the corrected muon-decay cone (34.6deg vs the old 26.5deg at
  0.2 GeV) may now be slightly too wide, or reaching production it should not.
  **Reclassified 2026-09-09 (§8.2): not a defect.** The exact kernel is gated to
  0.02-0.12% against its own flat-atmosphere closed form; the "1.046" plain
  solid-angle average is a measure-dependent number (the same exact kernel gives
  1.15 there in a flat atmosphere), and in the physically conserved
  ground-crossing measure the delivered factor is a 5.5% *deficit*, not an
  excess. Every ingredient except the cone width moves H/V by $\le2\%$; the cone
  width alone moves it by $\pm20\%$ for $\pm25\%$, i.e. the honest systematic
  here is the NA61 $\pm12\%$ uncertainty on $\langle\theta^2\rangle$, and it
  spans the Honda-Bartol difference.
* **`with_eoff_jacobian` is first-order only** (§2): it derives the `sigma_pi_NA61`
  pull from the ratio of two `offaxis_factor()` (old flat table) evaluations, not
  from re-evaluating the joint channel-resolved factor at sigma_pi +-12%.
* **Honda's azimuth convention must be mapped** in any future per-azimuth
  comparison: `az_compass = 180 - az_Honda` (his azimuth is counterclockwise from
  South; arXiv:1102.2688 § II). Symmetric observables (`max/min`, zenith-only
  averages) are unaffected; anything odd about the E-W axis flips sign without it.
* **v2 (arcsin-corrected) moments are not the engine default**: `solve()` needs
  `cone_moments={"pi": ["m_spliced_v2.npz","m_piminus_v2.npz"], "k": [...]}`
  passed explicitly to get the corrected cone; left unset it silently uses the
  pre-arcsin moments (§2 above).
* **Channel-weighted/species-dependent E_off is not wired into `offaxis_factor()`**:
  the delivered per-species table (`offaxis_excess_channel_v2.npz`) is only
  reachable through `joint_cone_factor`'s `G==1` limit; `offaxis_factor()` (used
  whenever `joint_cone=False`, and for the jacobian pull) remains the single
  flavour-blind table, and antinu_e overshoots Honda/Bartol (1.05/1.12) when the
  channel+v2 correction is applied standalone (W5).

## 6. 2026-09-05: dipole-phase investigation and the production-point-anchored cutoff

Source: two agent reports (transcripts `tasks/a4f1e74cc437bb1c6.output`, the
dipole-phase investigation, and `tasks/ad6416244f67cd3bb.output`, the anchor
implementation) plus the logs `scratchpad/dipole/{prodpoint,pytest_full}.log`
and `scratchpad/anchor/{hv,fourier,full,ewz}_{before,after}.log`,
`scratchpad/anchor/ab_tables.txt`. Both are quoted directly; nothing here is
invented.

### 6.1 Dipole-phase investigation: what the ~20deg rotation actually was

Following up on the open item above, an agent verified the IGRF field,
declination, handedness and every azimuth/rotation convention in
`geomag_backtrace.py` and `muon_bending.py` from first principles (aligned-dipole
closure tests: vertical cutoff vs `14.9 cos^4(lambda)`, E/W = 4.9 at 30deg
latitude/80deg zenith, published IGRF-13 declination at Kamioka to 0.1deg) and
found all of them correct. It fixed one real bug, with no measurable effect on
the delivered map: `geomag_backtrace.dipole_axis()` located the geomagnetic
north pole at 107.32E instead of the published 80.59N/72.68W (a longitude
inherited from the *south*-pole coefficients while the latitude was flipped to
the north pole); the rebuilt map differs by <=0.557 GV (rms 0.061 GV, 26/1050
cells > 0.2 GV) because the tilted-dipole term only matters beyond
`r_switch = 4 R_E`.

The actual root cause of the apparent rotation is **not** a frame, axis or
convention error: repeating the back-traced map in a pure, centred, *aligned*
dipole at 30deg latitude -- where geomagnetic East *is* azimuth 90deg by
construction and no declination or tilt can exist -- reproduces the same
skew (zenith 80: first-harmonic phase 69.7deg, N/S = 2.02, against the
Stormer-exact main-cone phase of 90.0deg, N/S = 1.00). $R_c$ here is $R_U$, the
*highest forbidden* rigidity, and above the Stormer main cone the Earth-shadow
cone suppresses the North sector's $R_U$ far more than it moves the East peak;
that skews a single first-harmonic fit even though the map's actual maximum
sits exactly at geomagnetic East. Launch altitude was tested and excluded
(<2% effect, N/S and phase unchanged for launch 0-100 km in the pure dipole).

What the same investigation *did* identify as a genuine, uncorrected physical
effect: the map is anchored at the **detector**, while the primary of a
near-horizon neutrino enters the atmosphere at the production point, several
degrees of geomagnetic latitude away. Launching the back-trace from the
production point instead of the detector (48 directions, `h_prod=30 km`):

| zen | launch | N | E | S | W | N/S | 1st-harm phase |
|---|---|---|---|---|---|---|---|
| 80 | detector | 18.94 | 35.35 | 10.32 | 8.00 | 1.83 | 66.89 |
| 80 | prod_point | 17.31 | 33.60 | 10.99 | 7.82 | 1.57 | 72.12 |
| 87 | detector | 23.78 | 42.99 | 10.26 | 7.60 | 2.32 | 61.74 |
| 87 | prod_point | 19.91 | 37.13 | 12.10 | 7.95 | 1.65 | 72.36 |

+10.6deg of phase and -29% of N/S contrast at zenith 87 -- the same size and
sign as the model-vs-Honda gap. `cone_geff(sublimb="prod_point")` already
rotates the production point's local *vertical* but still reads this
detector-anchored map, so the site-latitude half of the displacement was
unmodelled. This is the physical explanation handed to the anchor
implementation below. (Tests: `test_geomag_backtrace.py` +6 tests; full suite
**241 passed, 2 skipped** after the dipole-axis fix, `scratchpad/dipole/
pytest_full.log`. `CUTOFF_SCHEME` was re-keyed, forcing a full-sphere map
rebuild; every printed flux number is unchanged to print precision.)

### 6.2 The production-point-anchored cutoff (`cutoff_anchor="prod_point"`)

**Physics.** The primary of a zenith-87deg neutrino enters the atmosphere
368 km up the arrival ray (617 km at the exact horizon, `h_prod=30 km`), at a
measurably different geomagnetic latitude than the detector.

**Scheme.** A 3x3 Cartesian family of full down-going cutoff maps at site
offsets `(d_north, d_east) in {-617, 0, +617} km` (the horizon displacement),
each on the unmodified 21-zenith x 25-azimuth limb-node grid, bilinear in the
site offset only -- the direction axes keep the family's own fine nodes and are
never interpolated. The centre node is the already-cached detector map, so the
vertical is an exact no-op by construction. Cone samples are read in the
production point's TRUE local frame (`prod_frame_exact`), which -- unlike the
approximate `prod_point_frame` used by `sublimb="prod_point"` alone -- includes
the convergence of the meridians, up to 4.0deg of azimuth for an east-west
displacement.

**Cost.** 8 x 525 = 4200 extra back-traced cutoffs. A naive scan is dominated
by *allowed* rigidities above the cutoff (~900 RK4 steps vs ~30 for a forbidden
one) starting from a 55 GV ceiling for every cell. A seeded band scan exploits
that a displaced site's cutoff is within ~20% of the *same direction's*
detector cutoff: directions are split into 16 bands by that seed and each band
is scanned over a window around it, with any cell landing within one coarse
step of its band edge re-scanned full-range. Measured build: **14 927 s
(4.1 h) on 46 cores**, cached per site/date as `prodfam_*.npz` under scheme tag
`cartNxN-bandseed-v1` (part of the cache key, so a family built with a
different node layout, launch altitude or `r_hi` is never silently reused).
7 of 4200 cells saturate at the 55 GV ceiling (far-south-east limb, already
clamped in `G` at the same ceiling).

**Validation** (`diag_cutoff_anchor.py`, vs. direct production-point
back-traces at the same 48 directions used in §6.1):

| zen | source | N | E | S | W | N/S | 1st-harm |
|---|---|---|---|---|---|---|---|
| 87 | family (engine) | 20.07 | 37.43 | 12.61 | 7.66 | 1.59 | 71.98 |
| 87 | direct back-trace, h=30 (published) | 19.91 | 37.13 | 12.10 | 7.95 | 1.65 | 72.36 |

Cardinals agree to 0.16-0.51 GV, N/S 1.59 vs 1.65, phase 71.98 vs 72.36deg.
Residual budget over all 24 azimuths at 87deg: site interpolation <=1.23 GV
(12.7%), launch-altitude convention <=0.50 GV (6.9%). **Vertical: exactly
0.000e+00 GV change** at every azimuth (by construction). Continuity is
*better* than the detector-anchored curve: max |Delta R_c| per 0.5deg of
zenith is 0.06-0.33 GV against 2.54 GV for the detector map at azimuth 90.

### 6.3 Before/after vs Honda (detector -> prod_point, delivered engine)

Sources: `scratchpad/anchor/{hv,fourier,full,ewz}_after.log` (and their
`_before.log` counterparts) and `ab_tables.txt`.

**North/South, model/Honda at 0.5 GeV, cosZ 0.05, all four species**
(`ab_tables.txt`):

| species | N (det -> prod) | S (det -> prod) |
|---|---|---|
| nu_mu | 0.87 -> 0.95 | 1.07 -> 0.91 |
| antinu_mu | 0.86 -> 0.93 | 1.14 -> 0.98 |
| nu_e | 0.92 -> 1.02 | 1.23 -> 1.05 |
| antinu_e | 0.91 -> 0.98 | 1.12 -> 0.95 |

i.e. N: 0.86-0.92 -> 0.93-1.02, S: 1.07-1.23 -> 0.91-1.05.

**First-harmonic phase offset vs Honda, deg, 0.5 GeV** (`ab_tables.txt`):
nu_mu -12.7 -> +1.3, antinu_mu -11.8 -> +1.3, nu_e -12.8 -> +0.5, antinu_e
-14.7 -> -0.2.

**W/E vs Honda, nu_mu** (`scratchpad/anchor/ewz_{before,after}.log`,
`validate_ew_zenith.py`, charge-summed): 87deg +13->+12% (0.5 GeV), +22->+21%
(1 GeV), +26->+25% (2 GeV); 81deg +12->+11%, +19->+19%, +16->+16%; 69deg
+8->+8%, +10->+10%, +9->+8%; 63deg +5->+4%, +9->+9%, +7->+6%. Essentially
unchanged -- this observable is the cone average, not the cutoff anchor
(§5 above).

**Charge splitting, D(dphi) deg at 87deg** (`scratchpad/anchor/
fourier_{before,after}.log`, `diag_ew_charge_fourier.py`): nu_mu-antinu_mu
0.5 GeV 3.52 -> 4.30 (Honda 4.36); nu_e-antinu_e 0.5 GeV -7.32 -> -8.53 (Honda
-9.21). The *amplitude* splitting D(max/min) is unchanged and remains ~5x
short of Honda's.

**H/V, nu_mu, full/Honda** (`scratchpad/anchor/hv_{before,after}.log`,
`diag_shape_decompose.py`): 0.3 GeV 1.038 -> 1.031; 0.5 GeV 1.011 -> 0.998;
1.0 GeV 0.982 -> 0.971; 3.0 GeV 0.977 -> 0.978. Bartol vertical unchanged to
3 digits, as required (the anchor acts on `G`, not on the base or `E_off`).

**Grid-wide, 90th-pct |log10(ours/Honda)| over 0.1-100 GeV x 10 cosZ x 12 az**
(`scratchpad/anchor/full_{before,after}.log`, `diag_full_comparison.py`
section G): nu_mu 0.067 -> 0.061, antinu_mu 0.086 -> 0.076, nu_e 0.101 -> 0.090,
antinu_e 0.066 -> 0.061. Medians unchanged (0.032-0.036, antinu_e 0.022-0.023).

### 6.4 Default decision and remaining limitation

`mceq3d_flux.CUTOFF_ANCHOR` was flipped from `"detector"` to `"prod_point"` on
2026-09-05 (project decision, see the comment block at `mceq3d_flux.py:383-408`
and §2 above): the anchor is the physically correct treatment (it is what
Honda does -- back-tracing each primary from its own injection point) and the
family build is a one-time, cached, per-site/date precompute like the cutoff
map itself, so it does not change the cost of any subsequent evaluation.
`cutoff_anchor="detector"` remains the fast A/B path.

**Open item carried forward.** The bilinear site interpolation has measured
8-22% outliers in the low-cutoff West/South-West cells (the median cell is
~1%; `diag_cutoff_anchor.py` section 3, `test_cutoff_anchor.py`). An
`n_side=5` family (5x5 site grid instead of 3x3) would quarter the outliers at
~4x the one-time build cost; not built as part of this work.

**Test count.** `test_cutoff_anchor.py` adds 18 tests (geometry/frame
exactness including meridian convergence, site interpolation, family centre ==
`cutoff_map`, vertical no-op, direct-back-trace agreement, sign of the horizon
response). Combined with the 241 pre-existing passing tests (§6.1), the full
suite is **241 + 18 = 259, verified**; a first run against the working tree
(before three initially-wrong test expectations were corrected to the measured
truth) showed 256 passed / 3 failed / 2 skipped
(`scratchpad/anchor/pytest_full.log`), all three failures in
`test_cutoff_anchor.py` itself and traced to guessed rather than measured
expectations (the low-cutoff West lobe does not follow the cos^4(lambda)
latitude scaling, and the site interpolation has the 8-22% outliers noted
above) -- not to any engine defect. A confirmation run of the full suite after the `CUTOFF_ANCHOR` default flip (`scratchpad/tests_final.log`, 2026-09-05) completed with **259 passed, 2 skipped** in 29 min, so the total is verified post-flip.

## 7. Run recipe

* `.venv` has no interpreter any more (python3.12 removed from the machine).
  Working recipe: cvmfs `LCG_108` python3.12 with `PYTHONPATH` set to the venv's
  site-packages, `src/`, and `tools/mceq3d`. chromo is not installed in the venv;
  the cp312 wheel installs cleanly. 48 cores available.
* Run tests from `tools/mceq3d` (`pytest -q`); the delivered-path tests need a
  warm `.cache3d` (cutoff maps, `G_s` families) to run at reasonable speed —
  a cold 42x25 full-sphere map build is ~2.4 hours (down+up hemispheres run
  concurrently).
* AFS tokens expire mid-run on long builds; the failure mode is a
  `PermissionError` on the cache write at the very end of a build. Recover with
  `aklog` and re-run (W7 lost one wave of five `G_s` builds this way).
## 8. 2026-09-09: residual classification

Three parallel agent investigations were run to classify every residual left open
by §5/§6 as **our approximation**, **a different input**, or **unknown** — and to
check whether the East-West species-pattern deficiency (§6(i) of `PAPER_DRAFT.md`)
is corroborated by real data. Sources, quoted directly: the model/primary/epoch
scan (`tasks/a635b3c0e12da22e3.output`, new file `diag_model_scan.py`, logs
`scratchpad/modelscan/{logs,json,json2,epoch.json}`), the flux-conservation
derivation (`tasks/a4954116deaf98d50.output`, new files `diag_conservation.py`
— derivation in its module docstring — and `test_conservation.py`, log
`scratchpad/conserve/ALL_RESULTS.log`), and the Super-K East-West arbiter
(`tasks/aa32a9034feafdb2b.output`, new files `diag_sk_ew.py` and
`SK_EW_NOTES.md`, logs under `scratchpad/sk/`). No `.py`/`.npz` file on the
delivered path was touched by any of the three.

### 8.1 Classification table

| residual | size | cause | evidence | status |
|---|---|---|---|---|
| E-W species pattern (§6(i)): Honda orders $\nu_e>\bar\nu_\mu>\nu_\mu>\bar\nu_e$ (4.74/3.81/2.51/2.12 at 0.5 GeV/87°), engine compresses to ~2.6-3.3 | amplitude at 15-24% of Honda's | **ours** — the cone average, corroborated by SK data | §8.4: SK's own asymmetry pulls *more* $\nu_e$, *less* $\nu_\mu$ split than Honda's MC — the same direction the species-compression defect predicts; $\Delta\chi^2=4.2$-$4.6$ (~2$\sigma$) in Honda's favour with no free parameters | open, now data-corroborated, not merely model-internal |
| $\nu_\mu$ E-W overshoot, 0.5-2 GeV: Honda 2.51/2.09/1.52 (87°), ours +12/+21/+25% | +12% (0.5 GeV) down to a genuine +10-16% (1-2 GeV) | **mostly a different input** (primary spectrum) at 0.5 GeV; **partly unknown** at 1-2 GeV | §8.3 Table A: primary choice alone removes -9 to -11 points at 0.5 GeV (H3a +3%, Gaisser-Honda +1%), hadronic model 0 to -5; at 1-2 GeV the primary removes about half, a +10-16% residual survives every input tried | 0.5 GeV: resolved (input, not defect); 1-2 GeV: open, cause unknown |
| H/V 6-13% above Bartol with a "conservation excess" (§6(iii)) | $\langle F\rangle_\Omega$ 1.046-1.065 sub-GeV, previously read as manufactured flux | **not a defect** — the plain solid-angle average is measure-dependent; the correct (ground-crossing) measure shows a 5.5% *deficit*. The real, honest systematic is the NA61 cone-width uncertainty | §8.2: Eq. 8 gate reproduces $\langle\cos\alpha\rangle$ to 0.02-0.12%; everything but cone width moves H/V $\le2\%$; cone width alone moves H/V $\pm20\%$ for $\pm25\%$, i.e. the measured NA61 $\pm12\%$ on $\langle\theta^2\rangle$ | reclassified: input uncertainty (NA61 cone), not a construction defect |
| Absolute normalisation below 1.7 GeV (`PAPER_DRAFT.md` §4.1/§5, Eq. 11 band) | $\pm14$-$20\%$ at 0.1-0.5 GeV | **different input** (primary-spectrum/base choice) | Table 2 and the GSF-hybrid vs `daemonflux`-base spread, §4.1/§5 of `PAPER_DRAFT.md`; reinforced by §8.3: the primary alone moves the E-W amplitude by $\pm10$-$15\%$ and the vertical base by tens of percent | already quantified and carried as the model-spread systematic (Eq. 11); unchanged by this work |
| Honda's charge-splitting phase reversal above $\sim\!1\,\mathrm{GeV}$ (`PAPER_DRAFT.md` Table 4) | sign flip, ours stays flat | **unknown** | none of the three investigations targeted this observable | open, no new evidence |
| Housekeeping | — | — | see §8.5 | test count updated, new default-off options documented |

### 8.2 (A) Conservation: Eq. (8) is exact, the plain average is the wrong measure

**Derivation** (full text in the `diag_conservation.py` module docstring). Let the
primary flux be isotropic and $p(X,E)=d\Phi_\nu/dX$ MCEq's depth-resolved
production per unit slant depth per steradian of primary direction. The number of
neutrinos made per unit time in $dV$ at $P$ by primaries in $d\Omega_p$ is
$\rho(P)\,p(X_{\rm slant}(P,\Omega_p),E)\,dV\,d\Omega_p$ — the volume element does
not depend on the primary direction, so **no Jacobian enters**. They are emitted
with the decay-chain kernel $K(\alpha)$, azimuthally symmetric about the primary
axis, so normalising over $d\Omega_p$ at fixed $\Omega_\nu$ gives the same
constant as normalising over $d\Omega_\nu$ at fixed $\Omega_p$ — **no reciprocity
factor**. Since neutrinos free-stream, the specific intensity is the line integral
of the emissivity along the arrival ray, and this integral **is** Eq. (8)'s
numerator (denominator: $K\to\delta$). **Eq. (8) is the exact straight-line 3D
transport of an isotropic primary flux through a curved atmosphere; nothing is
missing.** The implementation was gated against this: in a flat atmosphere the
exact kernel has the closed form $F_{\rm flat}=\langle\cos\alpha\rangle_K$ at the
vertical, and the machinery reproduces it to **0.02-0.12%** per channel over
$\langle\cos\alpha\rangle=0.604$-$0.999$ (`test_conservation.py::test_flat_atmosphere_returns_mean_cos_alpha`).

**What is actually constrained is not $\langle F\rangle_\Omega$.** Neutrinos
crossing the ground are $\int_{\rm down}d\Omega\,\cos\psi\,\Phi$, i.e.
$\langle F\rangle_{\cos}$. Measured ($\nu_\mu$, `diag_conservation.py --decompose`):

| | 0.20 GeV | 0.30 GeV | 0.50 GeV | 1.00 GeV |
|---|---|---|---|---|
| $\langle F\rangle_\Omega$ curved (the number quoted in `PAPER_DRAFT.md` §3.6) | 1.0652 | 1.0458 | 1.0242 | 1.0084 |
| $\langle F\rangle_{\cos}$ curved | 0.9125 | 0.9453 | 0.9717 | 0.9891 |
| $\langle F\rangle_\Omega$ **flat** (exact, provably conserving) | 1.2059 | 1.1545 | 1.0950 | 1.0421 |
| $\langle F\rangle_{\cos}$ **flat** | 0.9276 | 0.9565 | 0.9785 | 0.9920 |
| $F$(vertical) curved | 0.8563 | 0.9138 | 0.9578 | 0.9847 |
| $\langle\cos\alpha\rangle$ (flat prediction) | 0.8557 | 0.9135 | 0.9576 | 0.9846 |

The exact kernel in a *flat* atmosphere — provably conserving — gives **+15% at
0.3 GeV in the plain-$\Omega$ measure**, more than the curved +4.6%, because
$F\sim1/\cos\psi_o$ at the horizon and that measure weights the horizon like the
vertical. In the ground-crossing measure the delivered factor is a **5.5%
deficit**: 4.4% already present in the flat limit (large-$\alpha$ emission leaving
the atmosphere upward) and 1.1% genuine sphericity (near-horizontal neutrinos
skimming over the limb and escaping). **Nothing is manufactured; if anything the
construction removes ~5%.**

**Decomposition of the residual** (`diag_conservation.py --decompose`, effect on
$F(\cos Z=0.05)$ at 0.3 GeV and on $\langle F\rangle_\Omega$):

| effect | on $F$(horizon) | on $\langle F\rangle_\Omega$ |
|---|---|---|
| quadrature ($n_\alpha$ 44→176, $n_\beta$ 18→72, $n_{\rm ray}$ 260→520) | — | $\le0.02\%$ |
| $p(X)$ depth truncation (1034→400 g/cm²) | — | $\le0.4\%$ |
| limb/$\psi$-grid refinement (1.29°→0.036°) | $-0.44\%$ | $-0.07\%$ |
| cone shape at fixed 2nd moment (von Mises-Fisher vs Gaussian) | $-0.27\%$ | $-0.04\%$ |
| $p(X,E)$ solved at 85° instead of 0° (fixed-density assumption) | $+1.67\%$ | $+0.23\%$ |
| **cone width $\times0.75$ / $\times1.25$** | 1.464 / 1.920 (vs 1.699) | 1.030 / 1.059 |

Per channel ($\langle N_c/D_c\rangle_\Omega$ at 0.3 GeV, weight): $\nu_\mu$ direct
1.035 (0.55), K 1.074 (0.013), $\mu$-decay 1.058 (0.434); $\nu_e$ is 99%
$\mu$-decay, hence its larger 1.052. Cone scale 0/0.5/0.75/1/1.25 gives
$\langle F\rangle_\Omega=1.000/1.014/1.030/1.046/1.059$ and
$\langle F\rangle_{\cos}=1.000/0.984/0.967/0.945/0.921$. **Verdict: neither
numerical nor a conditioning artefact — the whole effect is set by one measured
number, the cone second moment $\langle\theta^2\rangle$.**

**H/V (delivered engine settings, `diag_conservation.py --hv`)**, $\nu_\mu$/$\nu_e$
at 0.3/0.5/1.0 GeV: default 1.887/1.451/1.207 and 2.269/1.819/1.644, against Honda
1.825/1.448/1.241 and 2.119/1.830/1.770, Bartol 1.677/1.388/1.266 and
2.036/1.770/1.865. Cone $\times0.75$: 1.514/1.236/1.113 and 1.796/1.525/1.507;
cone $\times1.25$: 2.289/1.689/1.318 and 2.783/2.148/1.809; quadrature refined:
1.888/1.452/1.208 and 2.271/1.821/1.647 — indistinguishable from default.
**Everything except the cone width moves H/V by $\lesssim2\%$; the cone width
moves it by $\pm20\%$ for a $\pm25\%$ change.** Since NA61 constrains
$\langle\theta^2\rangle$ (hence the cone width) to $\pm12\%$, **the honest
sub-GeV zenith-shape systematic is this NA61 pull, worth roughly $\pm20\%$ on
H/V, and it spans the Honda-Bartol difference.**

**Generator spread on the cone** (`chromo` 0.11.0, `regen_moments_mp.run_model`):
UrQMD-3.4→DPMJET-III-3.07 (low-energy half, 4-80 GeV): $\sigma$ changes
$\le0.2\%$, $F$(horizon) $\le0.06\%$ — negligible because `fokker_planck_3d.load_theta2`
pools by secondary energy with no primary-spectrum weight, so the pool is
dominated by the highest-projectile-energy rows. SIBYLL-2.3d→EPOS-LHC
(high-energy half, $>80$ GeV): $\sigma_\pi$ +0.9…+2.6%, $\sigma_K$ +7.4% (0.2 GeV)
to $-4.6\%$ (3 GeV), $\sigma_\mu$ +1.0…+2.4%; $F$(horizon, 0.3 GeV) +0.59%
($\nu_\mu$) / +0.55% ($\nu_e$); $\langle F\rangle_\Omega$ +0.07%. **The
hadronic-model spread on the cone is ~1-2% on $\sigma$, well inside the NA61
$\pm12\%$ pull already carried.**

**New, default-off options** (`joint_cone.py`, all no-op unless invoked): a
Liouville normalisation constraint `liouville_renorm()` (:942) built on
`sky_scan()` (:913), plumbed into `delivered_joint_factor(renorm=...)` (:688,
applied at :875/:890); `arrival_ray(r_earth=...)` (:145), the flat-atmosphere
plug-point; and `ray=` on `cone_production`/`cone_production_multi`/
`delivered_joint_factor` (:357/:393/:687). The scale needed to force
$\langle F\rangle_{\cos}=1$ is 1.096/1.058/1.029/1.011 at 0.2/0.3/0.5/1 GeV; to
force $\langle F\rangle_\Omega=1$: 0.939/0.956/0.976/0.992. **Being a function of
energy alone, any such constraint cancels identically in H/V and E-W**
(`test_liouville_renorm_cancels_in_zenith_ratios`, gated to $10^{-12}$) — so no
normalisation constraint can buy or unbuy the zenith shape.

**Caveats not removed.** (i) The $\langle\cos\alpha\rangle\approx0.914$ plateau at
0.3 GeV implies an ~9% all-sky suppression relative to 1D away from the horizon —
larger than either Honda's $-2\%$ or Bartol's $+3\%$ angle-averaged numbers — so a
direct comparison against a published 3D MC in *both* measures is still owed. (ii)
The cone weight is energy-only, not depth-resolved; using the $\theta=85^\circ$
production profile instead of $\theta=0^\circ$ moves $F$(horizon) by $+1.7\%$, a
real ~2% depth/zenith-coupling systematic that *grows* the excess.

**Tests.** New: `test_conservation.py` (9 tests, all pass) and the RESULTS block
embedded in `diag_conservation.py`'s docstring. Regression: `test_joint_cone.py`,
`test_offaxis_conservation.py`, `test_offaxis_mc.py`, `test_kinematic_kernel.py`,
`test_delivered_shape.py`, `test_mceq3d_flux.py` → 72 passed, 2 skipped
(`scratchpad/conserve/ALL_RESULTS.log`). Only `joint_cone.py` was modified, and
every new option is default-off/no-op.

### 8.3 (B) Model/primary/epoch scan: is the hadronic model or the primary to blame?

**Structural finding.** MCEq's shipped database (`mceq_db_lext_dpm193_v140.h5`)
splices **DPMJET-III-19.3 into every high-energy model below ~80 GeV** projectile
energy: the $p+{\rm air}\to\pi^\pm$ yield matrices of SIBYLL-2.3d and
DPMJET-III-19.3 are **bit-identical** (ratio 1.00000) at 2.8/4.5/8.9/17.8/44.7 GeV,
diverging only above 89 GeV ($\pi^+$ 0.877, $\pi^-$ 0.830, charge ratio $+5.6\%$).
Consistently, vertical $\nu/\bar\nu$ charge ratios at 0.3-2 GeV are identical to
$\le0.5\%$ across five models; absolute $\nu_\mu$ moves $+1.7\%$ (DPMJET),
$+0.1\%$ (QGSJET-II-04, SIBYLL-2.1), $-4.6\%$ (EPOS-LHC). **So for the 5-40 GV
primaries that make sub-GeV neutrinos, the low-energy hadronic physics already
is Honda's own code family (DPMJET-III), and MCEq's model knob has almost no
lever there.** Models available offline (no network): `SIBYLL23D/23E/21`,
`DPMJETIII193`, `EPOSLHC(R)`, `QGSJETII04`, `QGSJETIII`; primaries: `GlobalSplineFitBeta`
(GSF, delivered), `HillasGaisser2012:H3a`, `GaisserHonda` (the closest available
proxy to Honda's own primary; his actual refit is not distributed with `crflux`).

**Code.** `interaction_model`/`primary` were already constructor arguments
(`_nucleon_split`, `mceq3d_flux.py:556`); new `gs_interaction_model=None`,
`gs_primary=None` (:611-612) let the **response** model ($G_s$) be overridden
independently of the base, via `_gs_spec`/`_gs_tag` (:687-696) and a lazily-built
second `MCEqRun` (`_gs_engine()`, :697); `geomag_response` keys on `_gs_tag`
(:886, :892). Defaults are unchanged and verified identical (`_gs_tag == _tag`,
cache hits in 0.00 s, `G_numu(0.5 GeV, R_c=42)=0.16957` unchanged). New CLI flags
`--interaction-model`/`--primary` (:2601, :2608). New file: `diag_model_scan.py`.

**Table A — $\nu_\mu$ W/E vs Honda (bending ON)**, `diag_model_scan.py measure`:

| | SIB23D/GSF | DPMJET/GSF | EPOS/GSF | QGSJET/GSF | SIB21/GSF | SIB23D/H3a | SIB23D/GH | Honda |
|---|---|---|---|---|---|---|---|---|
| 87° 0.5 GeV | 2.81 (+12%) | 2.73 (+9%) | 2.68 (+7%) | 2.81 (+12%) | 2.80 (+12%) | 2.58 (+3%) | **2.52 (+1%)** | 2.51 |
| 87° 1.0 GeV | 2.54 (+21%) | 2.47 (+18%) | 2.38 (+14%) | 2.52 (+21%) | 2.52 (+21%) | 2.30 (+10%) | 2.29 (+10%) | 2.09 |
| 87° 2.0 GeV | 1.89 (+25%) | 1.86 (+22%) | 1.79 (+18%) | 1.88 (+24%) | 1.90 (+25%) | 1.73 (+14%) | 1.76 (+16%) | 1.52 |
| 81° 0.5 GeV | 2.71 (+11%) | 2.63 (+8%) | 2.59 (+6%) | 2.70 (+11%) | 2.70 (+11%) | 2.49 (+2%) | **2.44 (−0%)** | 2.44 |
| 81° 1.0 GeV | 2.40 (+19%) | 2.34 (+16%) | 2.27 (+12%) | 2.39 (+18%) | 2.39 (+18%) | 2.19 (+8%) | 2.18 (+8%) | 2.02 |
| 81° 2.0 GeV | 1.80 (+16%) | 1.77 (+14%) | 1.71 (+10%) | 1.78 (+15%) | 1.80 (+16%) | 1.66 (+7%) | 1.68 (+8%) | 1.55 |

The baseline column reproduces the published +12/+21/+25% and +11/+19/+16%
exactly. At 0.5 GeV the primary choice removes essentially all of the residual
(+12%→+1% at 87°, +11%→$-0\%$ at 81°); at 1-2 GeV it removes about half, and a
**genuine +10-16% residual survives every input choice tried.**

**Table B — four-species W/E, 87°/0.5 GeV, bending OFF** (the charge-ratio
component):

| | SIB23D/GSF | DPMJET | EPOS | QGSJET | SIB21 | H3a | GH | Honda |
|---|---|---|---|---|---|---|---|---|
| $\nu_\mu$ | 2.87 | 2.79 | 2.73 | 2.87 | 2.86 | 2.62 | 2.57 | 2.51 |
| $\bar\nu_\mu$ | 2.88 | 2.79 | 2.73 | 2.87 | 2.86 | 2.62 | 2.57 | 3.81 |
| $\nu_e$ | 2.99 | 2.90 | 2.87 | 2.99 | 2.98 | 2.73 | 2.68 | 4.74 |
| $\bar\nu_e$ | 2.79 | 2.70 | 2.66 | 2.77 | 2.78 | 2.55 | 2.49 | 2.12 |
| **$\nu_e/\bar\nu_e$** | **1.07** | 1.08 | 1.08 | 1.08 | 1.07 | 1.07 | 1.08 | **2.24** |
| **$\nu_\mu/\bar\nu_\mu$** | **1.00** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **0.66** |

Bending ON: $\nu_e/\bar\nu_e=1.18\pm0.01$ across all seven configurations,
$\nu_\mu/\bar\nu_\mu=0.95$-$0.96$, against Honda's 2.24/0.66. **This residual is
not a different input**: the spread across five hadronic models and three
primaries is $\pm1\%$ of a required $+108\%$. The underlying $G_s$ charge
response moves $\le1.8\%$ ($\nu_e/\bar\nu_e$) and $\le0.6\%$ ($\nu_\mu/\bar\nu_\mu$)
across everything. The "near-threshold $\pi^+/\pi^-\gg1$" mechanism cannot even
be tested in MCEq below 80 GeV — every model there is the same bit-identical
DPMJET-III-19.3 tables. This is consistent with the cone-average hypothesis of
`PAPER_DRAFT.md` §6(i): the un-cone-averaged $(W/E)_G$ reproduces Honda's
ordering (3.79/4.08/4.35/3.45), so the compression happens in the cone average,
not in the hadronic input.

**Table C — horizon/vertical (azimuth-averaged)**, `diag_model_scan.py gsresp`:
$\nu_\mu$ 1.89/1.45/1.21 at 0.3/0.5/1 GeV (ratio to Honda 1.03/1.00/0.97; to
Bartol 1.125/1.045/0.954); $\nu_e$ 2.27/1.82/1.64 (Honda 1.07/0.99/0.93; Bartol
1.114/1.028/0.881). **Every model and primary sits within 1-2% of these**, EPOS
the only mild outlier (+1 to +2%). The 6-13% Bartol overshoot is untouched by
any input choice — consistent with §8.2's finding that it is a cone-width
(NA61), not an input, effect.

**$G_s$ charge-ratio response** ($E_\nu=0.5$ GeV, 87° band): the single-direction
W/E predicted by $G(R_c=7.1)/G(R_c=42.3)$ is 4.64 (SIB23D/GSF), 4.38 (DPMJET),
4.18 (EPOS), 4.01 (SIB23D/H3a), 3.96 (SIB23D/GH) — **the primary moves it $-15\%$,
the hadronic model at most $-10\%$**; the *charge* ratio itself moves $\le1.8\%$
($\nu_e/\bar\nu_e$) and $\le0.6\%$ ($\nu_\mu/\bar\nu_\mu$) across everything.

**Geomagnetic epoch** (IGRF back-trace, 12 directions × 4 epochs): vertical
cutoff drifts 3.2% over 2000-2020, but the East horizon cutoff at 87° drifts only
**0.7%** and the E/W contrast is $\pm1\%$, non-monotonic (5.49/5.60/5.50/5.59 at
2000/2005/2010/2020) — two orders below the +12/+21/+25% residual. Epoch is
negligible.

**Verdict and recommendation.** Residual (a), the $\nu_\mu$ W/E overshoot, is
**mostly a different input — the primary spectrum, not the hadronic model**: of
the +12% at 0.5 GeV, the primary is worth $-9$ to $-11$ points (H3a +3%,
Gaisser-Honda +1%), the hadronic model 0 to $-5$ (QGSJET/SIB21 0, DPMJET $-3$,
EPOS $-5$). At 1-2 GeV the primary removes about half; a genuine +10-16% residual
at 1-2 GeV remains ours/unknown. Residual (b), the charge-ratio component, is
**not** a different input — every lever is flat, so it stays classified as ours
(the cone average). **Recommendation: keep SIBYLL-2.3d** — below 80 GeV it *is*
DPMJET-III-19.3, Honda's own code family, so the choice is nearly vacuous
sub-GeV. Quote the hadronic spread as a small systematic: **$\le7$ points on the
E-W amplitude** (EPOS the envelope), **$\le2\%$ on H/V**, **$\le2\%$ on the
charge ratios**. The new, currently-unquantified-elsewhere systematic this scan
exposes is the **primary-spectrum dependence of the East-West amplitude,
$\pm10$-$15\%$** — far larger than its effect on H/V ($\le1\%$) — which belongs
in the error budget as a distinct pull.

**Tests.** `pytest -q` with the `gs_interaction_model`/`gs_primary` changes in
`mceq3d_flux.py`: 259 passed, 2 skipped (2883 s). A confirmation run 90 minutes
later gave **268 passed, 2 skipped** (1816 s) — the +9 is `test_conservation.py`
(§8.2, created independently at 18:45, exactly 9 tests collected), not this
agent's own changes, which add no tests and break none.

### 8.4 (C) Super-Kamiokande East-West: does the species-compression defect show up in data?

**Literature, corrected against the source papers (arXiv full text).** Futagami
*et al.*, PRL 82 (1999) 5194, **astro-ph/9901139** (not `hep-ex`), "Observation
of the East-West anisotropy…", **publishes no $A$ values**, only $\chi^2$/Kuiper
probabilities. The familiar $A_e=0.21\pm0.04$, $A_\mu=0.08\pm0.04$ (statistical
only) come from **Lipari, Astropart. Phys. 14 (2000) 171**, which also quotes
$A_e^{\rm HKKM}=0.13$, $A_\mu^{\rm HKKM}=0.11$ (HKKM95, 1D) and Bartol 0.17/0.15.
**SK-2016 (arXiv:1510.08127, PRD 94 052001)**, the definitive measurement: $A_\mu
=0.108\pm0.014\pm0.004$ (6.0$\sigma$), $A_e=0.153\pm0.015\pm0.004$ (8.0$\sigma$),
cuts $0.4<E_{\rm rec}<3.0\,$GeV, $|\cos\theta_{\rm rec}|<0.6$. Both papers define
$A$ over hemispheres on the **lepton momentum direction** (SK "east-going" =
neutrino arriving from the West). **No later SK azimuthal analysis exists**
(arXiv:1710.09126, 2311.05105 both state the oscillation baseline does not
depend on azimuth). Caveat: SK-2016's quoted $A$ does not reproduce its own
Fig. 22 (histogram gives $A=0.1306/0.0877$ against MC $0.1145/0.0929$, with
stated errors implying $N\approx4340/5043$ vs the $6547/8162$ plotted) — so
everything below is quoted **as a ratio to SK's own HKKM11 MC**, immune to that
offset.

**Grid.** Honda's ten $\cos Z$ bins with $|\cos Z|<0.5$, hemisphere sectors
(East/West, matching SK's definition), $E_\nu\in[0.5,3]\,$GeV (Lipari's window),
weight $\Phi\cdot\sigma$ with $\sigma\propto E$ and $\sigma(\bar\nu)/\sigma(\nu)
=0.45$ (cross-checked against Lipari's Table 1). Engine: delivered 3D
(`MCEq3DFlux(base_model="hybrid", primary=GlobalSplineFitBeta,
daemonflux_location="kamioka")`), bare `solve()`, Kamioka, IGRF epoch
2020-01-01; Honda = `honda_kam.npz` (HKKM2014).

**Flux-level results.**

| | Honda HKKM2014 | this engine | ours/Honda |
|---|---|---|---|
| $\nu_e$ | 0.2852 | 0.2278 | 0.80 |
| $\bar\nu_e$ | 0.1078 | 0.1977 | 1.84 |
| $\nu_\mu$ | 0.1532 | 0.2009 | 1.31 |
| $\bar\nu_\mu$ | 0.2349 | 0.2038 | 0.87 |
| **e-like** | **0.2375** | **0.2196** | **0.925** |
| **mu-like** | **0.1781** | **0.2018** | **1.133** |

Stable against sector width, $\bar r$ (the $\nu/\bar\nu$ cross-section weight),
and neutrino oscillations. Zenith bands: e-like ratio 0.98/0.89/1.01 (up/horizon/
down) — the engine under-produces the horizon peak; mu-like flat at ~1.13.

**Confrontation with the data**, transferring ours/Honda onto SK's own HKKM11 MC
(the ratio survives to lepton level to good approximation, nearly energy-flat):

| | SK data | SK's HKKM11 MC | this engine (MC × ratio) |
|---|---|---|---|
| $A_e$, optimised sample | $0.1306\pm0.0129$ | $0.1145$ ($+1.25\sigma$) | $0.1059$ ($+1.92\sigma$) |
| $A_\mu$, optimised sample | $0.0877\pm0.0117$ | $0.0929$ ($-0.44\sigma$) | $0.1053$ ($-1.50\sigma$) |
| $A_e$, $0.40<E_{\rm rec}<1.33$ | $0.1346\pm0.0140$ | $0.1227$ ($+0.85\sigma$) | $0.1150$ ($+1.40\sigma$) |
| $A_e$, $1.33<E_{\rm rec}<3.00$ | $0.1178\pm0.0251$ | $0.0885$ ($+1.17\sigma$) | $0.0783$ ($+1.57\sigma$) |
| $A_\mu$, $0.40<E_{\rm rec}<1.33$ | $0.0930\pm0.0125$ | $0.0988$ ($-0.47\sigma$) | $0.1112$ ($-1.46\sigma$) |
| $A_\mu$, $1.33<E_{\rm rec}<3.00$ | $0.0690\pm0.0235$ | $0.0718$ ($-0.12\sigma$) | $0.0836$ ($-0.62\sigma$) |

$\sum\Delta\chi^2$ (ours $-$ Honda) $=4.15$ on the optimised sample and $4.63$ on
the four $E_{\rm rec}$ bins — a **$\approx2.0$-$2.2\sigma$ preference for Honda
over this engine, with no free parameters**. Suggestive, not decisive: SK's own
statement that re-weighting HKKM11 to Bartol changes $\chi^2$ by 1.0 sets the
scale — current precision cannot cleanly separate two 3D flux models differing
by ~10% in $A$.

**The direction is unambiguous and matters.** The data want **more** $\nu_e$
asymmetry than HKKM11 ($+9$ to $+14\%$) and **slightly less** $\nu_\mu$
asymmetry ($-4$ to $-6\%$); our engine goes the **opposite way on both**
($-7.5\%$ e-like, $+13\%$ mu-like), because it compresses all four species'
asymmetries onto 0.198-0.204 while HKKM2014 spans 0.108-0.285 and Lipari's
independent 3D calculation spans $-0.065$ to $+0.335$, both with the ordering
$\nu_e>\bar\nu_\mu>\nu_\mu>\bar\nu_e$. **Two independent calculations and the
sub-GeV SK data agree on this ordering that our engine erases — the species
compression is a real deficiency, corroborated by data, not a Honda artefact.**

**What a real test needs.** Event-level modelling of the $\nu\to$lepton kernel
(SK's modal opening angle ~70° below 400 MeV, 38° at 0.4-0.7 GeV, 8° above
1.33 GeV — dilutes $A$ by roughly $\times0.7$), the CCQE/RES/DIS mix with the
correct $\nu/\bar\nu$ ratio, NC contamination (9% e-like, 1.3% mu-like),
acceptance, and three-flavour oscillations for the up-going half. A cheaper,
sharper target than reproducing SK's $A$ directly is SK's **dipole phase $B$ vs
zenith** (Fig. 23, a 2.2$\sigma$ effect) — a shape observable largely immune to
the smearing dilution, and one where our engine already tracks Honda's phase to
$\le1.3^\circ$ (`PAPER_DRAFT.md` §6(ii)) after the production-point cutoff
anchor.

**Files.** New: `tools/mceq3d/diag_sk_ew.py`, `tools/mceq3d/SK_EW_NOTES.md`. No
tracked file touched; no git commands run.

### 8.5 Housekeeping

* **Test count.** The model-scan agent's confirmation run completed **268
  passed, 2 skipped**, up from the 259 recorded at the end of §6.4; the +9 tests
  are entirely `test_conservation.py` (§8.2), not a product of the model-scan
  agent's own (default-preserving) changes.
* **New default-off surface.** `mceq3d_flux.MCEq3DFlux(gs_interaction_model=...,
  gs_primary=...)` (§8.3) and `joint_cone`'s `liouville_renorm`/`sky_scan`/
  `arrival_ray(r_earth=...)`/`ray=` (§8.2) are all opt-in; every default-path
  number in this file and in `PAPER_DRAFT.md` is unaffected.
* **No `.py`/`.npz` file on the delivered path was modified by the SK arbiter
  agent** (§8.4: two new files only). The conservation agent (§8.2) touched only
  `joint_cone.py` (default-off additions). The model-scan agent (§8.3) touched
  only `mceq3d_flux.py` (default-off additions per its own report).

* Test counts recorded during Phase 1 (not all directly comparable — different
  subsets/working-tree states): W1 targeted 34 passed / full suite 189 passed,
  1 skipped (460s); W2/W3 (joint cone) 229 passed, 0 failed (7m50s); W3 (kernel)
  57 passed in the dependent set; W4 (E-W observable + coupled march) 21 passed
  in its own files, 177 passed/1 skipped/4 failed in the rest of `tools/mceq3d`
  (the 4 failures are a stub in `test_cutoff_sublimb.py` unrelated to W4's files);
  W5 (channel E_off) 32 passed; W6 (muon-segment MC) 15 passed; W7 (penumbra)
  full suite 233 passed, 2 skipped.
