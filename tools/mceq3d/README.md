# MCEq 3D kernel regeneration

Tooling for the first step of a deterministic 3D MCEq: regenerating the
production kernels with **transverse-momentum** information that the 1D MCEq
database integrates away.

## Why

MCEq's interaction matrices are inclusive yields `dN/dx_L` in the lab energy
fraction `x_L = E_sec / E_proj`, with `p_T` integrated out — a 1D matrix
cascade has no angular index. A 3D solver (discrete ordinates / Fokker–Planck)
needs the double-differential kernel `d2N/(dx_L dp_T)`, because the production
angle `theta ~ p_T / p_L` is what redistributes secondaries in direction.

The original MCEq event-level output was never archived (these workflows
histogram on the fly). But the kernels are **regenerable**: re-run the same
generators MCEq uses, via `chromo` (formerly `impy`), and keep `p_T` this time.

## What `kernel_regeneration.py` does

1. Runs an event generator (default **SIBYLL-2.3d**, the model daemonflux uses)
   in inclusive mode via `chromo`, histogramming secondaries in `(x_L, p_T)`.
2. `marginalize_pt` integrates over `p_T` to recover `dN/dx_L`.
3. `compare_to_reference` checks that marginal against MCEq's **own stored**
   `dN/dx_L` (`load_mceq_reference`) — the consistency gate.
4. `to_angular_kernel` converts `(x_L, p_T)` to the production-angle density
   `d2N/(dx_L dtheta)` — the bridge to the 3D transport operator.

A `toy` backend (known analytic distributions, no external deps) exercises the
same pipeline offline and backs the unit tests.

```bash
# offline pipeline demo + plot
python kernel_regeneration.py --backend toy --plot

# real SIBYLL-2.3d kernel vs the real MCEq consistency gate
python kernel_regeneration.py --backend chromo --sec piplus \
    --emin 100 --emax 10000 --ne 5 --nint 12000 --plot

pytest test_kernel_regeneration.py -q          # offline tests
```

## Validated result

Real SIBYLL-2.3d (via chromo) vs MCEq's stored `SIBYLL23D` `dN/dx_L`:

* the regenerated p_T-marginal **reproduces the MCEq dN/dx_L shape across 3+
  decades in x_L** (see `kernel_real_piplus.png`, right panel);
* the **normalization is resolved and the gate now passes in absolute terms.**
  An apparent near-constant factor ~4.4 was a *reading* bug on our side, not an
  MCEq convention to be reconciled later: MCEq stores
  `hadr_yields = (dN/dx_L) · Δ(lnE_grid)`, and `Δ(lnE) = 0.2303`, i.e.
  `1/Δ(lnE) = 4.34`. `load_mceq_reference` divides it out (see `REVIEW.md`,
  "Resolution status" item (A)). Resonance feed-down was tested and ruled out
  independently: setting all MCEq-tracked species stable in chromo changes the
  π⁺ count by only ~3% (6.03 → 5.83 /event at 1 TeV).
* With the fix the gate gives **norm factor ≈ 1.0** — e.g. `gate_moments_norm.py
  m_spliced.npz --sec piplus` returns norm 0.97 with 86% of bulk-yield bins
  (`x_L ≥ 5e-3`) agreeing within 5%.

We also confirmed empirically that MCEq stores `hadr_yields` as `dN/dx_L` in
scaling form (the value at fixed log-offset `i-j` is energy independent), and
that integrating it gives physical π⁺ multiplicities (0.8 → 2.2 over 90 GeV →
9 TeV).

## Angular kernel: `angular_kernel.py`

Converts the regenerated `d2N/(dx_L dp_T)` into the transport objects a 3D
solver needs (this stage is **independent of the yield normalization**):

* production angle `theta = arcsin(p_T/p)` per `(E_proj, x_L, p_T)`, with `p` the
  **total** momentum `sqrt(E_sec^2 - m^2)` (see the note on the pre-2026-09
  `arctan` bug below);
* moments `<theta>`, `<theta^2>` (the Fokker–Planck diffusion coefficient);
* the discrete-ordinates scattering row `P(mu)` for an S_N solver;
* the crossover energy where `<theta>` drops below a detector scale.

**Quantitative result (`kernel_real_piplus_angular.png`), from real SIBYLL-2.3d:**
the mean π production angle follows `<theta> ~ <p_T>/E ~ 0.3 GeV / E`:

| secondary energy | mean production angle |
|---|---|
| ~1 GeV   | ~17° |
| 6.7 GeV  | 3° (crossover) |
| ~20 GeV  | ~1° |
| >250 GeV | <0.1° |

So below ~2 GeV the production angle is tens of degrees (3D essential) and above
~10–20 GeV it is sub-degree (1D fine). This is the rigorous, model-based version
of "why daemonflux/MCEq is accurate at high energy and needs 3D below 2 GeV".

```bash
python angular_kernel.py --kernel kernel_real_piplus.npz --plot
pytest test_angular_kernel.py -q
```

## Low-energy reach + direct-angle binning + splicing

Two upgrades remove the limitations found above:

* **`--angular` (direct-angle binning).** Instead of converting a coarse
  pre-binned p_T kernel (which aliases into a comb at low energy), the angle
  `theta = arctan(p_T/p_L)` is computed for *each secondary* and histogrammed
  once in `(x_L, theta)`. The result is smooth and is the correct input for the
  3D transport operator. `angular_kernel.py` auto-detects such a kernel
  (`theta_edges` present) and uses the smooth row directly.
* **Low-energy model.** `UrQMD-3.4` (in chromo) runs down to E_lab = 3 GeV and
  reaches the 1–2 GeV secondary regime that SIBYLL cannot. Use any chromo model
  via `--model`.
* **Splicing.** Because chromo's Fortran generators use global COMMON blocks,
  only one model runs per process, so each model's kernel is generated
  separately and merged by `splice_kernels.py` (low-energy model below a
  transition energy, high-energy model above) — the standard CORSIKA/MCEq
  approach.

```bash
# low-energy half (UrQMD) and high-energy half (SIBYLL), direct-angle
python kernel_regeneration.py --backend chromo --model UrQMD34  --angular \
    --emin 4  --emax 50    --ne 5 --out k_low.npz
python kernel_regeneration.py --backend chromo --model Sibyll23d --angular \
    --emin 80 --emax 10000 --ne 5 --out k_high.npz
python splice_kernels.py k_low.npz k_high.npz --transition 65 --out k_spliced.npz
python angular_kernel.py --kernel k_spliced.npz --plot
```

The spliced result (`k_spliced_angular.png`) now extends the `<theta>(E)` curve
down to ~1 GeV (reaching ~20°) with smooth angular rows. Note the linear theta
grid imposes a resolution floor at high energy (`<theta>` flattens near the
first-bin center); that regime is better resolved by the p_T kernel and is
physically `theta -> 0` anyway, so 1D is valid there.

## Angular moments (the Fokker–Planck input)

The `<theta>(E)` curve and the angular-diffusion coefficient should **not** be
taken from the binned (x_L, theta) kernel: a fixed theta grid floors `<theta^2>`
at the bin scale at high energy, which would inject spurious diffusion and break
the 1D limit in a Fokker–Planck solver. Instead, `--moments` computes the
moments **gridless** — `theta` is evaluated per secondary and averaged — so they
are exact at every energy and `<theta^2> -> 0` as `E -> inf`.

```bash
python kernel_regeneration.py --backend chromo --model UrQMD34  --moments \
    --emin 4  --emax 50    --ne 5 --out m_low.npz
python kernel_regeneration.py --backend chromo --model Sibyll23d --moments \
    --emin 80 --emax 10000 --ne 6 --out m_high.npz
python splice_kernels.py m_low.npz m_high.npz --transition 65 --out m_spliced.npz
python angular_kernel.py --kernel m_spliced.npz --plot   # -> m_spliced.png
```

The result (`m_spliced.png`) shows a clean 1/E law for `<theta>` from ~20° at
~0.6 GeV down to ~0.002° at 8 TeV (no high-E plateau), and the Fokker–Planck
coefficient `D_theta = <theta^2>/2` falling as E^-2 to ~1e-9 rad² — i.e. -> 0,
which is what makes the 3D solver recover the calibrated 1D flux at high energy.

**Use which product for what:** moments (gridless) for the Fokker–Planck solver
and for the `<theta>(E)` diagnostic; the binned `(x_L, theta)` kernel only for
the discrete-ordinates (S_N) scattering matrix, and then on the *solver's own*
angular grid (so its high-E "floor" just means "forward / no transfer", which is
correct for S_N).

## Validation against NA61/SHINE data

Since MCEq's 1D kernels contain no angular information, the regenerated p_T
content is validated against **data**: `validate_na61.py` compares the mean
transverse momentum `<p_T>(p_lab)` of charged pions from the UrQMD-3.4 kernel to
the NA61/SHINE measurement of π± production in **p+C at 31 GeV/c** (HEPData
`ins886780`, the T2K thin-target data — same energy/target as our low-energy
backend). NA61 tables are fetched once and cached in `na61_886780_cache.json`.

```bash
python validate_na61.py --plot      # -> validate_na61_pt.png
pytest test_validate_na61.py -q
```

Result (`validate_na61_pt.png`): once **NA61's forward acceptance (θ < 420 mrad)
is applied to UrQMD too**, the regenerated `<p_T>(p)` matches the data to ~10%
for both π⁺ and π⁻ across 0.3–8 GeV (ratio 0.97–1.15). Without the acceptance
cut, UrQMD's full-4π `<p_T>` is ~2× high at low momentum because NA61 cannot see
the large-angle pions — a textbook acceptance effect, and a reminder that
data/MC angular comparisons must match phase space. The residual ~10% (UrQMD
slightly high for π⁺ mid-momentum) is a genuine model–data difference, exactly
what a data-driven calibration would absorb.

## First 3D solve: Fokker–Planck angular transport

`fokker_planck_3d.py` is the first deterministic 3D solve — it adds the angular
transport MCEq omits, driven by the **validated** `<theta^2>(E)`. It marches
(implicit backward-Euler, tridiagonal, unconditionally stable) the projected-angle
diffusion `dg/dX = D(E) d2g/dphi2 + q(X) delta(phi)` through the atmospheric
column, with `D = <theta^2>/(4 lambda)` and a shower-development source `q(X)`.

```bash
python fokker_planck_3d.py --moments m_spliced.npz --plot   # -> fokker_planck_3d.png
pytest test_fokker_planck_3d.py -q
```

Result (`fokker_planck_3d.png`): the RMS arrival-direction spread `sigma_theta(E)`
of the flux is **~18° at 0.7 GeV, ~3° at ~7 GeV, <0.1° above ~300 GeV**, and is
**zenith-independent** — the spread is set by production kinematics over the
finite parent chain (`N_CHAIN ≈ 2`), not by column depth (see issue C in
`REVIEW.md`; the earlier "44–50°, grows toward the horizon" was the `slant/λ`
artifact). At high energy `sigma_theta -> 0`: the 1D limit (and daemonflux's
calibration) is recovered. The solver passes a correctness gate against the
analytic Gaussian (`sigma^2 = 2 D X`) for a top-only source to ~2%.

**Scope / caveats.** Energy is a parameter here (each E solved independently);
the energy redistribution is the existing 1D MCEq operator that this angular
operator multiplies onto — wiring the two together is the next step. The
magnitude is anchored to the NA61-validated `theta_1(E)`; the residual
uncertainty is the `N_CHAIN` factor (√2-level) and the single-step shape. Genuine
geometric zenith dependence (Honda horizontal excess) is a separate ingredient,
not included.

## Self-consistent 3D flux: coupling to the real MCEq 1D flux

`coupled_3d_flux.py` closes the loop. MCEq supplies the calibrated 1D *energy*
flux `Phi_1D(E, cos theta)` (azimuthally symmetric); the validated angular
operator redistributes it in arrival direction by an on-sphere convolution with
the FP spread kernel (width `sigma_theta(E, zenith)`), done by exact Monte-Carlo
rotation (flux-conserving, valid for any sigma):

```bash
python coupled_3d_flux.py --moments m_spliced.npz --plot   # -> coupled_3d_flux.png
pytest test_coupled_3d_flux.py -q
```

Result (`coupled_3d_flux.png`): at high energy `Phi_3D/Phi_1D = 1` everywhere
(sigma -> 0, the calibrated 1D result is recovered); at sub-GeV the production-
angle spread **redistributes the conventional numu flux by only ~1-2%** (≈2%
horizon depletion, ≈1% vertical enhancement at 0.5 GeV). **This is much smaller
than the earlier ~5-20%**, which used the inflated `slant/λ` spread (issue C):
with the physical `σ_θ ≈ 18°` the smearing of the (smooth) conventional zenith
distribution is modest. The take-away: for the conventional flux the dominant
sub-GeV 3D effects are **geomagnetic** (East-West, see below) and the
(not-yet-modelled) geometric horizon excess — not the production-angle smearing.

**Scope.** This is the *production-angle-spread* component of the 3D effect,
driven end-to-end by validated inputs and coupled to the real calibrated 1D
flux. It is complementary to (a) the **geomagnetic** rigidity-cutoff / East-West
layer already in `daemonflux.geomagnetic`, and (b) the curved-atmosphere
*geometric* source enhancement (Honda-style horizontal excess), which needs the
spherical production geometry and is not yet included. Folding all three on top
of the calibrated 1D flux is the path to a complete low-energy 3D model.

## Unified 3D flux: geomagnetic x angular

`unified_3d_flux.py` composes the project's two 3D strands into one directional
correction on top of the muon-calibrated 1D flux. Both are ratios to the same 1D
flux, so they multiply::

    Phi_3D(E, zenith, azimuth) / Phi_1D = R(E, zenith) * G(E, zenith, azimuth)

* `G` = geomagnetic rigidity-cutoff / East-West admittance
  (`daemonflux.geomagnetic`, the Stoermer scaffold — azimuth-dependent);
* `R` = production-angle zenith redistribution (`coupled_3d_flux`, NA61-validated
  `<theta^2>` — azimuth-independent).

```bash
python unified_3d_flux.py --moments m_spliced.npz --site kamioka --plot
pytest test_unified_3d_flux.py -q
```

Result (`unified_3d_flux.png`): at Kamioka, 1 GeV, zenith 70 deg the combined
correction is dominated by the geomagnetic **East-West asymmetry** (W/E ~ 8: flux
from the West far exceeds the East, the classic positive-primary effect); the
production-angle factor `R ≈ 1.00` here, so `R·G ≈ G`. A full-sky
`(azimuth, cos zenith)` map of `Phi_3D/Phi_1D` is shown. Everything -> 1 at high
energy (1D / daemonflux recovered). (The geomagnetic `G` is still the analytic
Störmer scaffold — issue B — so its magnitude is indicative, not final.)

This is the complete low-energy directional model *within current scope*: cutoff
+ East-West (geomagnetic) and zenith smearing (production-angle), both validated
end-to-end, on the calibrated 1D flux. The one remaining 3D ingredient is the
curved-atmosphere *geometric* source enhancement (Honda horizontal excess).

## Completing the model: channels, curved geometry, large-angle S_N

Three refinements (milestone #8):

* **Full channel set** (`channel_comparison.py`). Regenerated moments for
  `pi+/pi-/K+/K-`; kaons show **~40% larger production angles** than pions
  (heavier, harder `<p_T>` ~ 0.55 GeV vs ~0.45 GeV) — relevant since kaons
  dominate the flux at higher energies. `pi+ ~ pi-` as expected.
  (`channel_comparison.png`)

* **Curved-atmosphere slant depth** (`fokker_planck_3d.slant_depth`, `curved=True`).
  The flat `sec theta` diverges at the horizon; the spherical-shell geometry
  `L(theta)=sqrt((R+H)^2-R^2 sin^2 theta)-R cos theta` reduces to `sec theta`
  near vertical but **saturates** (~18x vertical at the horizon), as in real
  airmass treatments. This is the defensible geometric piece; the full off-axis
  production-volume enhancement (Honda horizontal excess) remains future work.

* **Large-angle P_N solver** (`sn_transport.py`). Cross-checks whether the
  small-angle Fokker-Planck form is adequate. It transports the full angular
  distribution as a spherical self-convolution; for a forward heat-kernel step
  the coefficients are analytic (`c_l = exp(-l(l+1) kappa)`), giving the bounded
  spread `arccos<cos theta>`. With the physical `N_CHAIN ≈ 2` the spread is
  moderate and **P_N agrees with FP everywhere** (`sn_transport.png`) — i.e. FP
  is adequate. The large-angle divergence (P_N bounded at 90°, FP unbounded) only
  appears for an unphysical large `N` (`--nchain 120`), which is the artifact that
  motivated fixing `N_gen` (issue C).

```bash
python channel_comparison.py --plot
python sn_transport.py --moments m_spliced.npz --zenith 80 --plot
pytest test_sn_transport.py test_fokker_planck_3d.py -q
```

## The integrated engine: `mceq3d_solver.py`

The culmination — every validated/de-risked component assembled into one coupled
solver:

* multi-species energy cascade `N -> pi/K -> numu` (matrix cascade in lnE);
* angular transport in the **Legendre/multipole basis** driven by the
  **NA61-validated** production angle `<theta^2>(E)`;
* the geomagnetic rigidity cutoff **folded into the primary spectrum** per
  arrival direction — the *proper* treatment (no `x_eff` hack; resolves REVIEW
  issue #4c). Low-rigidity primaries are removed before the cascade, so the
  low-energy neutrino suppression emerges self-consistently;
* curved-atmosphere slant depth.

```bash
python mceq3d_solver.py --site kamioka --zenith 0 --plot   # -> mceq3d_solver.png
pytest test_mceq3d_solver.py -q
```

Output: `Phi_numu(E)` for an arrival zenith with its self-consistent angular
spread `sigma_theta(E)`. Validated invariants:

* **exact reduction to 1D** — turning the angular machinery on leaves the l=0
  energy spectrum unchanged to **0.0e0** (the cascade is conserved);
* **geomagnetic suppression** folded through the primary: `Phi_3D/Phi_1D` rises
  from ~0.4 sub-GeV to **1.000** above the cutoff (`mceq3d_solver.png`, middle);
* the numu spectrum shows the expected `~epsilon_pi` steepening; `sigma_theta`
  emerges below the single-production angle (the parent-meson effect).

**Status — what this is and isn't.** It is the real *engine/architecture*,
runnable end-to-end, integrating four of the five 3D ingredients with their
correctness invariants tested. **Deferred to the documented next layer** (its
*feasibility* already proven by `prototype_streaming`): the **global spherical
streaming / curvature term** that couples different arrival directions — i.e. the
off-axis Honda horizontal excess. Also simplified: charge separation, muon-decay
neutrinos, EM cascade, and production-grade hadronic/atmosphere inputs. It is
**not** a drop-in replacement for production MCEq.

## Production engine: `mceq3d_production.py` (MCEq per multipole)

The production-grade architecture: instead of a toy cascade, **wrap MCEq**. The
angular distribution of each species is expanded in Legendre modes; a forward
production kick of RMS angle `theta1` multiplies mode `l` by `exp(-l(l+1)
theta1^2/4)`, so each multipole is just MCEq run with its meson-production scaled
by that factor (`MCEqRun.set_mod_pprod`). Therefore:

* **`l=0` is the unmodified MCEq solve == standard MCEq / daemonflux** — full
  species, charge, flavour, real atmosphere, real hadronic yields (validated:
  `l=0` vs plain MCEq, max rel diff **0.0e0**);
* `l>0` carries the validated angular structure through MCEq's *actual* cascade.

So the entire flavour/charge/atmosphere/hadronic physics comes from MCEq for
free; our contribution is the NA61-validated angular layer `theta1(E)`. Output:
the production-grade flux with its self-consistent `sigma_theta(E)`
(`mceq3d_production.png`; ~13 deg at 0.5 GeV -> sub-0.1 deg above ~100 GeV).

```bash
python mceq3d_production.py --lmax 8 --plot   # cost = (lmax+1) x MCEq solve
pytest test_mceq3d_production.py -q            # kernel logic (offline)
```

## Curved-atmosphere geometry: `spherical_geometry.py`

The off-axis / horizon-geometry ingredient, via the neutrino line-of-sight
production integral through the spherical shell. Validated against the limits:
vertical = 1, recovers flat `sec theta` near vertical, and **saturates finite at
the horizon** (`~sqrt(R/h0)`) where the flat law diverges — the curved-atmosphere
horizontal enhancement (`spherical_geometry.png`). Note MCEq's 1D already uses a
curved column, so this is the *geometric core*; the full off-axis excess also
needs the parent-direction spread (the angular kernels) wired into the spatial
transport, which together with muon bending is the remaining physics item.

## First-principles geomagnetics & muon bending

`geomag_backtrace.py` replaces the analytic Störmer cutoff with **trajectory
back-tracing** (RK4; allowed if the back-traced particle escapes, forbidden if it
returns). It uses the **full IGRF field (degree 13, via `ppigrf`)** near the
surface and the fast tilted dipole farther out, a **vectorized batched tracer**,
and a `cutoff_map` sky grid. Validated: against Störmer for an aligned dipole
(14.84 vs 14.9 GV equator, cos⁴λ, East-West), and the **full-IGRF Kamioka vertical
cutoff = 11.31 GV** (literature ~11.3). `geomag_cutoff_map.png` shows the
directional cutoff (low from the West, high from the East — the East-West effect).

`directional_flux.py` wires it together into `Phi_numu(E, zenith, azimuth)` for a
site: the back-traced full-IGRF cutoff folded into the *primary* + the cascade +
muon bending. At Kamioka, 1 GeV: ~0.54 vertical suppression, deepening to ~0.12 at
the horizon, with East-West **W/E = 2.17** (`directional_flux.png`).

`muon_bending.py` adds the muon-bending physics: the **energy-independent**
deflection `Delta_phi = q B tau / m ~ 3-5 deg`, the decay-in-flight weighting
(only sub-GeV muons decay in the atmosphere and bend), and the resulting
`<theta^2>` contribution to the μ-decay-neutrino angular kernel — a genuine
sub-GeV up/down & East-West driver. The coherent charge-dependent shift and its
wiring into the spatial solver remain the documented build.

```bash
python geomag_backtrace.py        # cutoff vs Stoermer (slow: trajectory integ.)
python muon_bending.py --plot
pytest test_geomag_backtrace.py test_muon_bending.py -q
```

## Findings that shape the production effort

* **The ~4.4 factor is RESOLVED** — it was a missing division by MCEq's
  log-energy bin width (`hadr_yields = dN/dx_L · Δ(lnE)`, `1/Δ(lnE) = 4.34`),
  fixed in `load_mceq_reference`; the gate now returns norm ≈ 1.0. Feed-down was
  separately tested and ruled out (tracked-stable changes the count ~3%).
* **Production-angle convention (fixed 2026-09).** The moments and the angular
  kernel used `theta = arctan(p_T/p)` with `p` the *total* momentum, i.e. the
  total momentum in the place of the longitudinal one. The correct relation is
  `sin(theta) = p_T/p`. The bug biased `<theta^2>` **low**. Measured on identical
  event samples (`diag_angle_convention.py`), the corrected π⁺ `sqrt(<theta^2>)`
  is **+46% at `E_sec` = 0.3 GeV**, +32% at 0.5, +13% at 1, +4.6% at 2, +1.1% at
  5 and +0.3% at 10 GeV — i.e. the bias sits exactly in the sub-GeV region that
  drives the off-axis excess `E_off`, where it dwarfs the ±12% NA61 systematic.
  Downstream, `kinematic_kernel.channel_shapes` gives `sigma_pi(E_nu)` 13% wider
  at 0.2 GeV, 11% at 0.3, 9% at 0.5, 6% at 1 and 2% at 3 GeV. It was invisible to
  `validate_na61.py`, which compares `<p_T>` (angle-convention-independent);
  `validate_na61_angle.py` closes that hole by comparing `<theta>(p_lab)` against
  the NA61 polar-angle tables. The corrected moment files are the `*_v2.npz` set
  (`regen_moments_mp.py`); the pre-fix `m_*.npz` are retained for comparison.
* **SIBYLL-2.3d has a hard floor at √s = 10 GeV (E_lab ≈ 53 GeV).** Below that —
  exactly where geomagnetic effects matter most (1–10 GeV) — a **low-energy
  model** is required (DPMJET / data-driven), as MCEq already does. The NA61
  double-differential `d²N/dx_F dp_T` data daemonflux relies on is the natural,
  data-driven source for the low-energy transverse kernels.
* **Decay kernels are analytic** (two-body `π/K → μν` kinematics) and are not
  covered here.

## Next steps

1. ~~Reconcile the storage-normalization factor~~ **done** — it was a missing
   `1/Δ(lnE)`; the gate returns norm ≈ 1.0 (`gate_moments_norm.py`).
2. ~~Low-energy backend~~ **done** — UrQMD-3.4 reaches E_lab = 3 GeV; splice via
   `splice_kernels.py`. ~~Cross-check against NA61 data~~ **done** —
   `validate_na61.py` matches NA61 p+C 31 GeV/c `<p_T>` to ~10%.
3. ~~Direct-angle binning~~ **done** — `--angular` removes the comb artifact.
4. ~~Gridless moments / Fokker–Planck input~~ **done** — `--moments` emits exact
   `<theta>`, `<theta^2>` and `D_theta(E)` with no high-E floor.
5. ~~Minimal Fokker–Planck transport prototype~~ **done** —
   `fokker_planck_3d.py` solves the angular transport from the validated
   `D_theta(E)`; first actual 3D solve.
6. ~~Couple the angular operator to the 1D MCEq flux~~ **done** —
   `coupled_3d_flux.py` produces a self-consistent 3D `Phi(E, cos theta)`,
   recovering 1D at high E and showing the ~5-20% sub-GeV zenith redistribution.
7. ~~Fold geomagnetic + angular into one directional flux~~ **done** —
   `unified_3d_flux.py` gives `Phi_3D/Phi_1D = R x G`, East-West + zenith smearing
   together on the calibrated 1D flux.
8. ~~Full channel set + curved geometry + large-angle S_N~~ **done** —
   `channel_comparison.py` (pi/K), `slant_depth(curved=True)`, `sn_transport.py`.
9. ~~Single self-consistent coupled (E × angle) solve with energy
   redistribution~~ **done** — `prototype_3d_cascade.py` (derives N_eff) and the
   integrated `mceq3d_solver.py` (multi-species + geomag primary-folding).
10. ~~Feasibility of the spatial streaming term~~ **de-risked** —
    `prototype_streaming.py` (sparse, ~linear).
11. ~~Production-grade flavour/charge/atmosphere/hadronic inputs~~ **done** —
    `mceq3d_production.py` wraps MCEq per multipole (`l=0` == MCEq exactly), so all
    of that physics is inherited; the angular layer rides on top.
12. ~~Curved-atmosphere horizon geometry (core)~~ **done** —
    `spherical_geometry.py` (validated against the sec-theta / saturation limits).
13. ~~IGRF geomagnetics (vs Störmer)~~ **done (dipole, first-principles)** —
    `geomag_backtrace.py` computes the cutoff by trajectory back-tracing in an
    IGRF-dipole field; validated against Störmer (14.84 vs 14.9 GV vertical
    equator, cos⁴λ, East-West). Pluggable `bfield` for full IGRF (add higher
    Gauss terms).
14. ~~Muon bending~~ **done (physics + angular spread)** — `muon_bending.py`:
    the energy-independent `q B tau / m ~ 3-5 deg` deflection, the decay-in-flight
    weighting (sub-GeV only), and the `<theta^2>` contribution to the μ-decay-ν
    angular kernel.
15. ~~Full IGRF + vectorized cutoff map~~ **done** — `geomag_backtrace.py` now
    uses the real IGRF (degree 13, via ``ppigrf``) near the surface, a vectorized
    batched tracer, and a `cutoff_map` sky grid (Kamioka full-IGRF vertical
    cutoff = 11.31 GV vs literature ~11.3; East-West correct after a sign fix).
16. ~~Off-axis pieces wired into a directional flux~~ **done (the non-cluster
    part)** — `directional_flux.py` produces `Phi_numu(E, zenith, azimuth)` for a
    site from the back-traced full-IGRF cutoff folded into the primary + the
    cascade + muon bending (Kamioka 1 GeV: ~0.54 vertical suppression, East-West
    W/E = 2.17). `geomag_cutoff_map.png`, `directional_flux.png`.
17. ~~Spherical-streaming PDE + coherent muon-bending E-W~~ **done** —
    `spherical_streaming.py` adds the **curvature term** `(1-mu^2)/r d/dmu` to the
    P_N transport (a staggered grid cures odd-even decoupling); the P_1 limit
    reproduces spherical diffusion **exactly**, the operator stays sparse &
    ~linear with curvature, and the detector flux is redistributed toward the
    horizon (off-axis excess). `muon_bending.py` now gives the **coherent
    charge-dependent E-W shift** (mu+ ~+3° east, mu- ~-3° west; ~3° ν/ν̄ split
    sub-GeV, ~0.3° summed). `spherical_streaming.png`, `muon_bending.png`.
18. ~~Couple the cascade to the curved atmosphere~~ **done** —
    `spherical_cascade.py` runs the energy cascade *down the curved line of sight*
    of each arrival direction (depth-resolved, local density), giving the absolute
    directional flux `Phi_nu(E, cos zenith)`. Reproduces the textbook structure:
    near-isotropic sub-GeV, **sec θ horizon enhancement at high E** made finite by
    the geometry (≈5× at 2 TeV near the horizon vs the divergent sec θ=25), 1D
    vertical reference. This is the *dominant* directional structure; the
    genuinely-3D residual (inter-direction streaming) is small (~1–2%, the
    `spherical_streaming`/`coupled_3d_flux` term). `spherical_cascade.png`.
19. ~~High-statistics kernels~~ **done** — produced on a cluster (UrQMD34 +
    Sibyll23d, 200k evt/pt, spliced at 80 GeV → 41-pt grid) and installed as the
    drop-in `m_spliced.npz` / `m_{piminus,Kplus,Kminus}.npz`; see
    `KERNEL_PRODUCTION_REPORT.md`.
20. ~~Full-shape S_N adequacy~~ **resolved** — `sn_transport.py --kernel` builds
    the single-production density from a real `d²N/dx_L dθ` kernel and compares to
    a Gaussian of the *same variance*: pure-shape median rel diff ≈ 0, so the
    variance-only Fokker-Planck input suffices and the full-shape kernel is **not
    needed for the angular spread**. (`sn_transport.png`, demo `k_local_demo.npz`.)
21. ~~Quantitative Honda 3D cross-check~~ **done** — `validate_honda.py` parses
    the Honda HKKM2014 azimuth-dependent Kamioka table and matches this work on
    the directional observables: with the production-cone-averaged cutoff, at a
    *matched* near-horizon zenith (cosZ 0.25, ~75°) the East–West amplitude sits
    ≈11–14 % above Honda across 0.4–2 GeV (1.85 Honda vs 2.10 this work at
    1.1 GeV; correct sign, peak, and >10 GeV vanishing), a systematic overshoot
    growing toward the extreme horizon (out-of-sample zenith scan: ~6 % at 63° →
    ~26 % at 87°, where the cone truncates at the limb); sec θ horizon
    enhancement 2.19 vs 2.33 at 100 GeV. (`validate_honda.png`.)
22. ~~Trustable absolute *full-sky* directional engine~~ **done** —
    `mceq3d_flux.py` returns the absolute Φ(E, cosθ, azimuth) for all 4 flavours
    over the whole sky to ~0.5 GeV as MCEq (curved) × cascade-correct geomag (no
    `x_eff`); up-going via the **global far-side** geomagnetic treatment.
    **Absolute** νμ matches Honda at Kamioka to **0.90–0.98 @1 GeV** and
    reproduces the **up/down asymmetry** (up 129–135 vs down 111–117, Honda
    133–138 vs 122–130). `interp_flux()` accessor. (`mceq3d_flux.png`.)
23. Genuinely remaining (refinements):
    * **daemonflux muon-calibrated base** to anchor the ~10–20% normalization
      (1-line swap, needs spline data);
    * **per-nucleus rigidity** in the cut (~20% at the sub-GeV horizon); finer
      cosθ grid near the horizon; **NA61 K±** HEPData fit.

> **See [`IMPLEMENTATION_REPORT.md`](IMPLEMENTATION_REPORT.md)** for the complete,
> self-critical account (every method, validation plot, and an exhaustive list of
> simplifications/cheats) — written to communicate this work to the package author.
