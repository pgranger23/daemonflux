# `tools/mceq3d` architecture — what is delivered vs. what is a prototype

> **2026-09-04 update.** The delivered engine changed materially in Phase 1
> (see `PHASE1_RESULTS.md`): `joint_cone.py` is now part of the delivered path,
> the product equation below replaces the one this file described on
> 2026-07-16, and several modules move between "delivered" and "retired". This
> revision reflects the working tree as of the Phase-1 fixes (uncommitted).

> **2026-09-05 update.** `mceq3d_flux.CUTOFF_ANCHOR` now defaults to
> `"prod_point"` (was `"detector"`): the geomagnetic cutoff `cone_cutoff`
> samples is read off a map built at the neutrino's own production point,
> interpolated from a new cached per-site "displaced-site map family"
> (`MCEq3DFlux.prod_family_rc`), rather than at the detector. See the new
> subsection below and `PHASE1_RESULTS.md` §6 for the physics and validation.

This directory grew as a research sprint and accumulated ~60 modules, many of
them overlapping feasibility prototypes for a "real 3D MCEq". This file is the
authoritative map of **which single code path produces the delivered flux** and
which modules are research/de-risking scaffolding that the delivered flux does
**not** use.

## The canonical delivered engine

**`mceq3d_flux.MCEq3DFlux`** is the one and only delivered solver
(`solve()`, `offaxis_factor()`, `cone_geff()`, `joint_cone_factor()`). It is a
*factorised deterministic product*, not a coupled Monte-Carlo. With the
recommended (and default-on) settings `offaxis=True, cone_cutoff=True`
(`joint_cone` then defaults to `True` automatically), the delivered product is

```
Phi_s(E, cosZ, phi) = Phi_1D_s(E, |cosZ|)  x  F_s(E, cosZ, phi)  x  S(E)

F_s = J_s / p_axis      (joint_cone.py / MCEq3DFlux.joint_cone_factor)
    = single integral over the production altitude AND the primary-direction
      cone of  p(X_slant, E) * G_s(E, R_c(n_p))  -- replaces the product
      E_off(E,cosZ) x <G_s>_cone(E,n) and restores the cone covariance
      Cov_cone(p, G) the product drops (zero at the vertical, largest at the
      horizon).
```

`F_s` is built with `joint_channels=True` (default): MCEq's own depth-resolved
per-species parent-channel production profiles (`{s}_dir`/`{s}_k`/`{s}_mu`),
each folded with its own cone width — the direct pion cone `sigma_pi`, the kaon
cone `sigma_K`, and the corrected muon-decay cone `mudecay_shape_mc` per flavour
(the neutrino's own decay-opening angle enters for the muon-decay component, so
`F_s` is genuinely species-dependent, unlike the flavour-blind `E_off` it
supersedes). The charge-signed muon-bending shift (`muon_bending`, default on)
displaces the muon channel's cutoff lookup by `+-delta` per species. The
geomagnetic cutoff transmission `G_s(E,R_c)` uses the measured penumbra width
`sigma_lnR = 0.0` (`SIGMA_LNR`, `mceq3d_flux.py:154`, changed from the legacy
0.5) analytically bin-averaged over the primary energy bin. Down-going cone
samples below the local horizon are evaluated in the **production-point** frame
(`sublimb="prod_point"`, default; ~30 km up the arrival ray) and blocked
(`p->0`) rather than reading the antipodal far-side cutoff — the 89 deg seam
that used to sit in the map is gone from the down-going path. As of 2026-09-05
each such sample's cutoff is additionally read off a map **built at the
production point** rather than at the detector (`cutoff_anchor="prod_point"`,
default — see below). Up-going neutrino directions and energies outside the
tabulated 0.1-100 GeV range fall back to the legacy product `E_off * <G>_cone`.

**The production-point-anchored cutoff (`cutoff_anchor`, default
`"prod_point"` since 2026-09-05).** `sublimb="prod_point"` alone rotates the
local *vertical* of a cone sample to the production point P but still reads
the cutoff off the detector-anchored map; `cutoff_anchor="prod_point"` supplies
the missing half, the *site* displacement. The primary of a near-horizon
neutrino enters the atmosphere at P, up to 617 km from the detector at the
exact horizon (`H_PROD_KM=30`), at a measurably different geomagnetic
latitude — closing this gap removes essentially the whole North/South and
dipole-phase mismatch (`PHASE1_RESULTS.md` §6) with no free parameter. The
implementation:

* `mceq3d_flux.prod_point_offset_km(zen, az, h_prod_km)` — the ground
  displacement `(d_north, d_east)` of P, exactly zero at the vertical, 617 km
  at the exact horizon.
* `MCEq3DFlux.prod_family_rc(lat, lon, date, ...)` — builds and caches (as
  `.cache3d/prodfam_*.npz`, scheme tag `PROD_FAMILY_SCHEME =
  "cartNxN-bandseed-v1"`) a `3x3` Cartesian **family** of full down-going
  cutoff maps at site offsets `(d_north, d_east) in {-D, 0, +D}^2` (`D` = the
  horizon displacement); the centre node is byte-identical to the ordinary
  `finemap_rc` detector map and is reused rather than rebuilt. Cost: 8 extra
  maps (4200 back-traced cutoffs) using a **seeded band scan**
  (`_banded_cutoff_scan`, ~4x faster than a naive scan by exploiting that a
  displaced site's cutoff is within ~20% of the same direction's detector
  cutoff), measured at **4.1 h on 46 cores** for the Kamioka family — a
  one-time, cached, per-site/date precompute exactly like `finemap_rc` itself.
* `mceq3d_flux.interp_site_rc(family, dn_km, de_km)` — bilinear interpolation
  of the family in the *site* offset only; the direction (zenith/azimuth) axes
  always keep the family's own fine limb-node grid and are never interpolated.
* `mceq3d_flux.prod_frame_exact(cos_theta, az_deg, h_prod_km, lat_deg)` — the
  cone samples' TRUE local frame at P (in the detector's N/E/U frame),
  including the convergence of the meridians (up to 4.0 deg of azimuth for an
  east-west displacement) that the approximate `prod_point_frame` /
  `cone_geff`'s own vertical-only rotation omits — harmless while the map was
  detector-anchored, but not once the map is built at P.
* `joint_cone.delivered_joint_factor(..., rc_family, site_lat)` and
  `mceq3d_flux.cone_geff` select, per arrival azimuth, the map/frame pair
  above whenever `rc_family is not None` (i.e. `cutoff_anchor="prod_point"`).

Validated against direct production-point back-traces
(`diag_cutoff_anchor.py`): cardinals agree to 0.16-0.51 GV, N/S 1.59 vs 1.65,
first-harmonic phase 71.98 vs 72.36 deg; the vertical is an exact no-op
(`0.000e+00` GV change) by construction. Known limitation: the bilinear site
interpolation has measured 8-22% outliers in the low-cutoff West/South-West
cells (median cell ~1%); an `n_side=5` family would quarter them at ~4x the
build cost (not built). `cutoff_anchor="detector"` (the pre-2026-09-05
default) remains available as the fast A/B path — it needs no family build at
all.

**Both of those defaults were consolidated on 2026-09-04** (they were the last
place where the delivered physics depended on what the caller typed):
* `kinematic_kernel._MOMENTS` **is** the arcsin-corrected `m_*_v2.npz` set; the
  pre-arcsin files live on as `kinematic_kernel._MOMENTS_LEGACY` (and
  `offaxis_mc.MOMENTS_LEGACY`) for A/B only, and `offaxis_mc.MOMENTS_V2` is now
  an alias of `_MOMENTS`. `solve(cone_moments=None)`, `channel_shapes()`,
  `mudecay_shape*()`, `channel_cone_widths()` and `joint_cone_widths()`
  therefore all resolve to v2. `joint_cone_widths` keys its `.cache3d`
  `conewidths_*.npz` file on the **actual moment file names** (it used to key on
  "was an override passed", which would have served stale widths after the
  default flip).
* `offaxis_factor()` reads `mceq3d_flux.EOFF_TABLE` =
  **`offaxis_excess_channel_v2.npz`**, the species-resolved channel-weighted v2
  build, and returns a genuinely per-species factor. The flat July table stays
  reachable as `EOFF_TABLE_FLAT` / `solve(offaxis_table=...)` /
  `offaxis_factor(path=...)`. With `G == 1` the table path and the joint-cone
  path now agree to machine precision (`test_mceq3d_flux.
  test_table_path_matches_joint_path_at_unit_G`), i.e. every place E_off enters
  — `cone_cutoff=False`, the up-going hemisphere, energies outside 0.1-100 GeV,
  the `sigma_pi_NA61` pull — is the same physics as the down-going joint factor.
* `solve()` itself defaults to the delivered configuration: `offaxis=True`
  (changed from `False`), `cone_cutoff=True` → `joint_cone=True`,
  `joint_channels=True`, `cone_kernel="moments"`, `sublimb="prod_point"`,
  `sigma_lnr=None` → `SIGMA_LNR=0`. A bare
  `solve(lat, lon, cz, az, use_cache=True, cache_dir=..., date=...)` reproduces
  the S4/sigma=0 numbers of `PHASE1_RESULTS.md`. `offaxis=False` remains the
  fast geomagnetic-only path.

### Modules the engine actually depends on (transitive import closure)

| module | role in the delivered flux |
|---|---|
| `mceq3d_flux.py`      | the engine (base reconstruction, product assembly, covariance) |
| `joint_cone.py`        | the joint production x cutoff cone integral `F_s` (delivered default) |
| `geomag_backtrace.py` | IGRF-13 rigidity cutoff `R_c` by trajectory back-tracing, now a coarse ladder (<=1 GV step) + bisection to 0.1 GV (`scan_upper_cutoff`), `r_hi=55` GV; `dipole_axis()` fixed 2026-09-05 (geomagnetic north pole was at 107.32E instead of 80.59N/72.68W — no measurable effect on the delivered map, see `PHASE1_RESULTS.md` §6.1) |
| `offaxis_mc.py`        | production profiles (`production_profile`, per-species channels), the legacy `offaxis_factor` table builder, `channel_cone_widths` |
| `kinematic_kernel.py` | pion/kaon/muon production-angle kernels (NA61-anchored); `mudecay_shape_mc` / `direct_shape_mc` / `_sigma_binned` (corrected, binned, replacing the biased power-law `mudecay_shape` fit in the channel path) |
| `muon_bending.py`     | charge-signed in-flight muon-decay cone shift for the E-W geomagnetic term |
| `fokker_planck_3d.py` | supplies `load_theta2` / `theta2_interp` (generator moment loader) |
| `coupled_3d_flux.py`  | supplies `convolve_sphere` (legacy `full_3d=True` path only) |
| `angular_kernel.py`   | angular-moment helpers (`production_angle`, now `arcsin`-based) used by kernel regeneration |

### Build-time input (not imported at runtime, but the engine reads its output)

| module | produces | read by the engine when |
|---|---|---|
| `offaxis_mc.build_channel` | `offaxis_excess_channel_v2.npz` (species-resolved, channel-weighted, v2 moments) | **always**, via `offaxis_factor()` (`EOFF_TABLE`), whenever `offaxis=True` (the default). It is the value actually multiplied into the down-going in-range flux only when `joint_cone=False` (i.e. `cone_cutoff=False`, or `joint_cone` forced `False`). When `joint_cone=True` (default) it is used (a) as the fallback for up-going directions and E outside 0.1-100 GeV, and (b) for the `with_eoff_jacobian` NA61 pull (`sigma_pi_NA61`), which is therefore first-order (ratio of two table evaluations) but is now built on the same cone as `F_s`. It is also the `G==1` gate target for `joint_cone_factor` (`test_joint_cone.py`, `test_mceq3d_flux.py`). |
| `offaxis_mc.py --build` / `build_channel` (A/B outputs) | `offaxis_excess.npz` (flat, single-pion-cone, July build; `EOFF_TABLE_FLAT`), `offaxis_excess_rebuild_old.npz`, `offaxis_excess_v2.npz`, `offaxis_excess_channel.npz` | only through an explicit `solve(offaxis_table=...)` / `offaxis_factor(path=...)`, e.g. `diag_joint_stages.py`'s S0..S4 ladder, which pins the flat table so its cumulative numbers stay comparable. |
| `kernel_regeneration.py`, `splice_kernels.py` | the generator moment kernels `m_*.npz` (old, pre-arcsin convention) | only via `kinematic_kernel._MOMENTS_LEGACY` / an explicit `cone_moments=` |
| `regen_moments_mp.py` | `m_{spliced,piminus,Kplus,Kminus}_v2.npz` — arcsin-corrected, same UrQMD-3.4/SIBYLL-2.3d/80 GeV splice recipe as `KERNEL_GENERATION.md` | **default**, via `kinematic_kernel._MOMENTS` |

### Validation / analysis entry points (keep; each is a runnable script)

`validate_honda.py`, `validate_bartol.py`, `validate_na61.py`,
`validate_na61_kaon.py`, `validate_ew_zenith.py`, `validate_na61_angle.py` (new
— `<theta>(p_lab)` vs NA61 polar-angle tables, the angle-convention-sensitive
counterpart to `validate_na61.py`'s convention-blind `<p_T>`), `base_comparison.py`,
`channel_comparison.py`, `make_cutoff_map_fig.py`, `diag_eoff_conservation.py`,
`gate_moments_norm.py` (new — norm/shape acceptance gate on the delivered vs v2
moments against MCEq `hadr_yields`), `diag_angle_convention.py` (new — same-sample
arcsin-vs-arctan comparison, no MC scatter), plus the `verify_*`, `latitude_check`,
`geomag_zenith_check`, `profile_3d` diagnostics.

**`diag_ew_charge_fourier.py` (new) is now THE East-West / charge-splitting
observable.** It replaces the retired `max/min`-over-azimuth comparison (exactly
even under a sign flip of a coherent shift — the bug that hid the charge
splitting) with a first-two-harmonic azimuthal decomposition (`a1/a0`, `dphi`,
`s1/a0`) and applies Honda's azimuth convention (`az_compass = 180 - az_Honda`,
his azimuth is counterclockwise from South, arXiv:1102.2688 § II). Any future
per-azimuth comparison against Honda that is odd about the E-W axis must go
through this module or apply the same mapping.

**`diag_joint_stages.py` (new)** is the S0->S1->S1a->S1b->S2->S4 cumulative
regression harness for the joint-cone/channel/v2-moments steps — the source of
the cumulative tables in `PHASE1_RESULTS.md` §3.1-3.2.

**Cutoff-map diagnostics (new, W1):** `diag_cutoff_buildmap.py`,
`diag_cutoff_stage.py`, `diag_cutoff_ablation.py`, `diag_cutoff_block.py` —
build/stage/ablate the repaired `finemap_rc` and quantify the Earth-shadow
blocking fraction at the limb.

**`diag_dipole_phase.py` (new, 2026-09-05)** is the dipole-phase investigation
harness (`PHASE1_RESULTS.md` §6.1): field/convention closure tests against a
pure aligned dipole, the decisive experiment showing the ~20 deg "rotation" is
the Earth-shadow cone biasing the first-harmonic phase estimator (not a frame
or axis error), and the detector- vs production-point-launch comparison that
identified the real, remaining physical effect. Its `--sections prodpoint`
mode is the reference the anchor family is validated against.

**`diag_cutoff_anchor.py` (new, 2026-09-05)** builds and validates the
displaced-site cutoff family: reproduces `diag_dipole_phase.py`'s 48
production-point directions from the cached family and compares to direct
back-traces (§3 of that script covers the site-interpolation outliers),
measures the family build cost, and runs the delivered-engine before/after
comparison against Honda that produced the `cutoff_anchor` A/B numbers in
`PHASE1_RESULTS.md` §6.3. `test_cutoff_anchor.py` (new, 18 tests) covers the
geometry/frame exactness (including meridian convergence), site
interpolation, family-centre-equals-`cutoff_map`, the vertical no-op, and
direct-back-trace agreement.

**`diag_penumbra_width.py` (new, W7)** measured the actual penumbral band
`[R_L, R_U]` by direct back-tracing on a 0.1 GV ladder; it is the source of the
`SIGMA_LNR=0.0` default.

**`diag_conservation.py` (new, 2026-09-09)** is the flux-conservation
derivation and decomposition harness (`PHASE1_RESULTS.md` §8.2): its module
docstring carries the two-line first-principles proof that Eq. (8) is the
exact straight-line 3D transport of an isotropic primary flux (no Jacobian, no
reciprocity factor), the flat-atmosphere closed form `F_flat = <cos alpha>_K`
that the implementation is gated against, and the finding that the physically
conserved quantity is the ground-crossing (`<F>_cos`) average, not the plain
solid-angle one quoted in earlier drafts — in the latter the delivered factor
is a 5.5% deficit, not the previously-reported ~5% excess. `--flat`,
`--decompose`, `--zenith`, `--hv`, `--fine-geom` reproduce every number in
that section. `test_conservation.py` (new, 9 tests) gates the no-op options
below, the flat-atmosphere closed form, and the Liouville-constraint identity.
Only `joint_cone.py` was modified to support it, with every addition
default-off: `arrival_ray(r_earth=...)` (`joint_cone.py:145`, the
flat-atmosphere plug-point), `ray=` on `cone_production` /
`cone_production_multi` / `delivered_joint_factor`
(`joint_cone.py:357/393/687`), and a Liouville normalisation constraint
`sky_scan()` (`joint_cone.py:913`) + `liouville_renorm()`
(`joint_cone.py:942`) plumbed into `delivered_joint_factor(renorm=...)`
(`joint_cone.py:688`, applied at `:875`/`:890`) — being a function of energy
alone, it is proven to cancel identically in every H/V and E-W ratio
(`test_liouville_renorm_cancels_in_zenith_ratios`), so it cannot be used to
buy or unbuy the zenith shape. The NA61 `<theta^2>` uncertainty (±12%), not
this construction, is the resulting honest zenith-shape systematic (~±20% on
H/V at 0.3 GeV, `PHASE1_RESULTS.md` §8.2).

**`diag_model_scan.py` (new, 2026-09-09)** is the interaction-model /
primary-model / geomagnetic-epoch scan that classifies how much of the
Honda residual is a different input rather than an engine approximation
(`PHASE1_RESULTS.md` §8.3). It uses two new, default-`None` (hence inert)
constructor keywords on `MCEq3DFlux`: `gs_interaction_model=None`,
`gs_primary=None` (`mceq3d_flux.py:611-612`), which let the **response**
model `G_s` be built from a different interaction model / primary than the
base, via `_gs_spec`/`_gs_tag` (`:687-696`) and a lazily-constructed second
`MCEqRun` (`_gs_engine()`, `:697`); `geomag_response` keys its cache on
`_gs_tag`, not `_tag` (`:886`, `:892`), so an override never silently reuses
the default-model `G_s` cache, and when unset `_gs_tag == _tag` and behaviour
is bit-identical to before (verified: cache hits in 0.00 s,
`G_numu(0.5 GeV, R_c=42) = 0.16957` unchanged). New CLI flags
`--interaction-model`/`--primary` (`mceq3d_flux.py:2601`, `:2608`). Its
`build`/`measure`/`gsresp`/`epoch`/`table` sub-commands are the source of
Tables A-C in `PHASE1_RESULTS.md` §8.3: the structural finding that MCEq's
shipped `lext_dpm193` database splices DPMJET-III-19.3 (Honda's own generator
family) into every model below ~80 GeV, and the resulting primary-spectrum
pull on the East-West amplitude (±10-15%, an order of magnitude larger than
its effect on H/V) now carried in `PAPER_DRAFT.md` §5.

**`diag_sk_ew.py` (new, 2026-09-09)**, with its companion note
`SK_EW_NOTES.md`, computes the flux-level confrontation between this engine
and the Super-Kamiokande East-West asymmetry measurement (SK-2016,
arXiv:1510.08127; `PHASE1_RESULTS.md` §8.4, `PAPER_DRAFT.md` §4.7). It
evaluates the delivered engine and Honda HKKM2014 on Honda's own
`|cosZ|<0.5` zenith grid and hemisphere sectors, transfers the resulting
ours/Honda ratio onto SK's own published HKKM11 Monte Carlo, and reports the
summed chi-square difference against the SK data (~2.0-2.2 sigma in Honda's
favour, no free parameters) — corroborating, from real data rather than a
second model comparison, that the East-West species-compression limitation
(i) in `PAPER_DRAFT.md` §6 is a genuine engine deficiency. Touches no
tracked file; two new files only.

**`muon_segment_mc.py` + `diag_muon_segment.py` (new, W6) are a closed study,
not part of the engine.** A full backward Monte Carlo (continuous dE/dx, full
IGRF-13 along the muon's own trajectory) that refuted the "energy-loss enhances
bending" hypothesis (§4 of `PHASE1_RESULTS.md`); kept for the record, not
imported by `mceq3d_flux.py`.

## Research / de-risking PROTOTYPES — **not** part of the delivered flux

These were built to test feasibility of a genuinely coupled 3D transport (or, in
the 2026-09 batch, to test the charge-dependent E-W mechanism). The delivered
`mceq3d_flux` engine does **not** import any of them except the two named
exceptions above (`convolve_sphere`, `theta2_interp`/`load_theta2`). Each carries
a `PROTOTYPE` banner in its module docstring where noted.

| module | what it explored | status |
|---|---|---|
| `mceq3d_real.py`, function `march_profile` | deterministic march of MCEq's real matrices, used as the independent production-vertex cross-check (`mceq3d_prodvertex.py`) | **active** — reproduces `E_off` to 1-2%, unchanged by Phase 1 |
| `mceq3d_real.py`, function `march_checkpoints` | **DEAD END, KEPT FOR THE RECORD.** Angular-binned operator-splitting march with inter-column rotation, meant to carry the charge-signed muon bend through the cascade. The 2026-09 fix (10x gyroradius unit bug, `argmax` pull -> row-normalised inverse-square interpolation) makes it numerically correct but the *structural* conclusion (W4) is that it cannot represent muon bending near the horizon: bent muons end up in atmospheric columns hundreds of km away, not a neighbouring direction bin at the same altitude. A full-hemisphere run moves the E-W phase splitting by <=0.3 deg against Honda's 4-12 deg. **Do not develop further**; a genuine Monte Carlo (audit Phase 2) is the only route to this observable. |
| `coupled_ew_charge_diag.py` | **DEAD END, KEPT FOR THE RECORD.** Driver for `march_checkpoints`, above; rewritten in Phase 1 onto the corrected observable and the repaired map, confirming the null is structural, not a bug. |
| `diag_checkpoint_resolution.py` | **DEAD END, KEPT FOR THE RECORD.** The 10x checkpoint-resolution scan meant to certify `march_checkpoints`'s convergence; its premise was doubly invalid (it was chasing the 10x unit bug, and a lossy resample applied 140x is as destructive as applied 14x), so it cannot certify anything either way. Updated to the new observable in Phase 1 but not a source of trustworthy numbers. |
| `diag_ew_charge_scale.py` | **DEAD END, KEPT FOR THE RECORD.** Scaled the factorised coherent shift by up to 30x looking for the charge split; "saturated" because its `max/min` observable is blind to the shift's sign (root cause identified in Phase 1, not a real physical saturation). Updated to the new observable for continuity, not for new conclusions. |
| `validate_3d_deterministic.py` | emergent cone factor vs `E_off`/Honda | superseded — its closure question was answered by `mceq3d_prodvertex.py` |
| `mceq3d_deterministic.py` | parametrised (pre-real-matrix) 3D march | superseded by `mceq3d_real` |
| `mceq3d_solver.py`, `mceq3d_production.py` | earlier coupled-solve attempts | superseded |
| `spherical_streaming.py`, `spherical_cascade.py`, `spherical_geometry.py` | P_N spherical transport with curvature term | feasibility only |
| `coupled_3d_transport.py` | coupled energy x angle solve prototype | feasibility |
| `sn_transport.py` | discrete-ordinate angular spread | feasibility |
| `prototype_3d_cascade.py`, `prototype_streaming.py` | cost-scaling de-risking of the coupled solve | feasibility, done |
| `unified_3d_flux.py`, `directional_flux.py`, `geomag3d_spike.py` | early end-to-end assemblies | superseded by `mceq3d_flux` |
| `hadronic_spread.py`, `calib_uncertainty.py`, `solar_modulation.py` | one-off studies | standalone |
| `coupled_ew_diag.py` | inter-direction coupling estimate (<1%, distinct from `coupled_ew_charge_diag.py`) | feasibility, unchanged by Phase 1 |

## Moment files — old vs v2

`m_spliced.npz` / `m_piminus.npz` / `m_Kplus.npz` / `m_Kminus.npz` are the
original (pre-2026-09) generator moments: production angle extracted as
`theta = arctan(p_T/p_total)`, which the audit root-caused as wrong (should be
`arcsin(p_T/p)`), biasing `sqrt(<theta^2>)` ~16% too narrow at 0.3 GeV and <1%
above 3 GeV. `m_spliced_v2.npz` / `m_piminus_v2.npz` / `m_Kplus_v2.npz` /
`m_Kminus_v2.npz` are the corrected regeneration (`regen_moments_mp.py`, same
UrQMD-3.4 (<=80 GeV) / SIBYLL-2.3d (>80 GeV) splice as `KERNEL_GENERATION.md`).

**The engine reads the v2 files by default since 2026-09-04**:
`kinematic_kernel._MOMENTS` is the v2 dict, and every default call
(`channel_shapes(moments=None)`, `mudecay_shape_mc(moments=None)`,
`offaxis_mc.channel_cone_widths(moments=None)`,
`mceq3d_flux.joint_cone_widths(moments=None)`, `solve(cone_moments=None)`,
`cone_geff(cone_moments=None)`) resolves to them. The OLD files are reachable
only through `kinematic_kernel._MOMENTS_LEGACY` / `offaxis_mc.MOMENTS_LEGACY`
or an explicit `moments=`/`cone_moments=`; `offaxis_mc.build_channel` names both
axes explicitly, and `diag_joint_stages.py`'s S0/S1a/S1 rungs pin `M_OLD` so the
"+v2 arcsin moments" step (S1 → S1b) still measures something. Two things that
do **not** follow the flip, by design: `solve(moments="m_spliced.npz")` and
`angular_factor(moments=...)`, which drive only the retired `full_3d`
redistribution and are kept on the old file so that legacy comparison is
unchanged.

## Production-vertex closure — DONE, unchanged by Phase 1 (`mceq3d_prodvertex.py`)

`E_off` (the flat table, `offaxis_excess.npz`) is a near-flux-conserving
redistribution and is confirmed as the deterministic production-vertex result:
neutrinos free-stream, so the whole 3D neutrino effect is production geometry —
there is no transport coupling to solve for them (why `march_checkpoints`
always failed, see above). `mceq3d_prodvertex.py` takes the production profile
`p(X,E)` from the independent hand-march `mceq3d_real.march_profile` and
integrates it through the curved production-vertex cone, reproducing the
delivered `E_off` to ~1-2%.
