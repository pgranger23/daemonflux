# Integrating the 3D factor into daemonflux, and the up-going hemisphere (2026-11-05)

Status: **interface done and tested; the Kamioka table is PROVISIONAL** (option A of
`MILESTONE3F_RESULTS.md` sec. 5/8). Work area `/home/pigrange/mc3d_work/runs/integ/` (tables, figures, logs,
`NOTES.md`); caches via `$MCEQ3D_CACHE`. Nothing committed.

## 1. Summary

> **Update (same day, section 10):** finding 1 below was diagnosed (the MC's 1D reference `t` is wrong at the horizon, its `s` is fine) and the provisional table was rebuilt against one common 1D reference for both hemispheres. Sections 3 and 6 describe the first version (the `udscaled` variant is withdrawn); the current numbers are in section 10.

* `daemonflux.geomagnetic.Tabulated3DModel` (new) is a per-species, tabulated 3D factor `F_s(E, cos Z, az)`, not
  limited to [0, 1], with up-going zeniths. `Flux.set_geomagnetic_model` takes it (or a path to its npz). Sums
  and flavour/charge ratios are recomputed from the corrected components, the 1D error is scaled by the same factor,
  the factor's own uncertainty is exposed separately (`Flux.geomagnetic_factor_error`). No model: bit-identical to
  the 1D model (tested); the analytic `GeomagneticModel` and its tests are untouched.
* Engine up-going `F_s` at Kamioka (GSF, v142p, muon_bending off, rigidity fixed): computed on the MC coarse
  grid (10 energy bins 0.1-10 GeV, five 0.2-wide cos Z bins per hemisphere, 6 azimuth bins).
* **Findings that matter for the table**
  1. The engine's `F = solve / base(|cos Z|)` is the same object as the MC's `s/t` and **agrees with the MC's
     down-going G to 1-3% for cos Z > 0.35** (E 0.2-3 GeV), but is **25-35% lower at cos Z < 0.2** (engine/MC 0.65-0.76
     at cos Z 0.05-0.15, 0.85-0.92 at 0.25, E 0.2-0.5 GeV). Against Honda the engine is the one with the right
     horizon shape: with the MC's G at cos Z 0-0.2 the product with daemonflux's 1D flux is **+38% (nu_mu) / +23% (nu_e)
     above Honda's shape at 0.3 GeV** (+10% / +55% at 1 GeV), the engine's is within 5% (section 6). Either the MC's 1D
     reference `t` differs from MCEq's curved-atmosphere 1D flux near the horizon, or the MC's 3D horizon excess is
     too large; the table cannot be used across the horizon as it stands (the literal table jumps from 0.9 to 1.4 at
     cos Z = 0). This is not resolved here; it needs the MC's `t` against MCEq at cos Z 0-0.2.
  2. The engine up-going factor is good for |cos Z| > 0.2 (up/down within about 5%, section 5) and **poor in the
     near-horizon up-going bin** (up/down 17-21% low at 0.3 GeV, azimuth phase dphi ~2-4 deg where Honda has 7-52 deg).

## 2. Interface (`src/daemonflux/geomagnetic.py`, `flux.py`)

```python
from daemonflux import Flux
from daemonflux.geomagnetic import Tabulated3DModel
flux = Flux(location="kamioka")
flux.set_geomagnetic_model(Tabulated3DModel(path))     # or flux.set_geomagnetic_model(path)
flux.flux(E, 100.0, "numu", azimuth_deg=90.0)           # up-going: 90 < zenith <= 180
flux.geomagnetic_factor_error(E, 100.0, "numu", azimuth_deg=90.0)   # relative, from the table
```

* **Table (npz)**: `species` (S,) from `numu antinumu nue antinue mu+ mu-`; `e_edges` (nE+1) [GeV],
  `cosz_edges` (nC+1) ascending in (-1, 1) (cos Z < 0 up-going), `az_edges` (nA+1) [deg, N=0, E=90, spanning 360];
  `factor`, `factor_error` (S, nE, nC, nA) finite (absolute 1-sigma); optional `label`, `status`, `site`, `provenance`
  (and anything else, e.g. `source`). The loader refuses NaN, wrong shapes, unknown species, non-monotonic axes.
* **Interpolation**: linear in ln E and in cos Z between bin centres, linear and periodic in azimuth; flat between
  the outermost centre and the edge; cos Z outside the table raises. The factor is not clipped.
* **Outside the table**: above `E_max = e_edges[-1]` the factor (and its error) is blended linearly in log10 E to 1 (error to 0)
  over `blend_decades` (default 1); below `e_edges[0]` a `UserWarning` (default, the lowest bin is used) or
  `ValueError` (`below_emin="raise"`).
* **Quantities** (per prefix `total_` as well): species -> own factor; `numuflux`, `nueflux`, `muflux` ->
  flux-weighted factor of the components with the 1D component fluxes as weights; `numuratio`, `nueratio`,
  `muratio`, `flavorratio` -> quotient of the numerator and denominator factors (weighted for the sums in
  `flavorratio`), i.e. recomputed from the corrected components; species not in the table (muons) -> 1. Because
  the 1D splines of sums and ratios differ from their components by up to ~0.4%, the multiplier is applied to the
  spline value, not recomputed from the components (so a unit table is exact).
* **Up-going**: zenith in (90, 180] uses the 1D flux at `180 - zenith` (1D production is up/down symmetric).
* **Errors**: `error()` scales the 1D error by the same multiplier as the flux; the table's own uncertainty is
  not folded in (`Tabulated3DModel.factor_error`, `scale_error`, `Flux.geomagnetic_factor_error`; components are
  taken uncorrelated, which neglects the correlation through the MC's common 1D reference).
* **Tests** (`tests/test_tabulated3d.py`, 14, plus the 18 existing `test_geomagnetic.py` untouched): bit-identical
  no-model output and after clearing; unit table = 1D to 1e-12; closed-form toy table (linear in ln E and cos Z)
  reproduced to 1e-12; bin-centre values and flat edges; blend to 1 above E_max, warn/raise below E_min; cos Z
  coverage; azimuth periodicity and interpolation; recombination of species/sums/ratios (all quantities, with and
  without `total_`); equal factors leave ratios unchanged; error scaling and the separate table error; up-going; npz
  round trip and `set_geomagnetic_model(path)`; invalid tables; the provisional table if present. Black (88) and
  flake8 (setup.cfg) clean on `src` and `tests` (installed in `$W/pylib_lint`, not in the venv).
* Docs: README section "3D / geomagnetic corrections" (new subsection) and CLAUDE.md.

## 3. The provisional tables (PROVISIONAL) -- first version, superseded by section 10

`$W/runs/integ/kamioka_3D_provisional.npz` (PM's option A, literal) and two variants of the same format, all with
`status = "PROVISIONAL"` in the file and its `repr`; species nue, antinue, numu, antinumu; E 5/decade 0.1-10 GeV;
cos Z 10 bins of 0.2 over (-1, 1); az 6 bins of 60 deg; `source` marks the origin of each (E, cos Z) bin.

| file | down-going (cos Z > 0) | up-going |
|---|---|---|
| `kamioka_3D_provisional.npz` | pilot-2 s/t (E < 2.5 GeV, jackknife G=20 errors), engine F above | engine F |
| `kamioka_3D_provisional_udscaled.npz` | as above | MC/engine down-going factor at the mirrored |cos Z| x the engine up/down ratio |
| `kamioka_3D_provisional_engine.npz` | engine F (no MC) | engine F |

Errors: MC jackknife where the MC is used; **5% of F as a nominal placeholder** where the engine is used (the engine has
no statistical error; 5% is the order of the engine-vs-Honda shape residuals of section 6, not a derived number). The
pilot-2 errors per coarse cell are 2-3% (cos Z > 0.4, E < 3 GeV), 3-5% (0.2-0.4), 6-37% (horizon bin; nu_e 0.5 GeV
37%), up to 10-20% at 3-10 GeV (the engine replaces them there). The production table replaces these in the same format.

## 4. Engine up-going `F_s`: definition and the 1D reference

`F = MCEq3DFlux.solve(lat, lon, cos Z, az, muon_bending=False) / MCEq3DFlux.base(|cos Z|)` per species. `base` is
the engine's curved-atmosphere 1D flux without cutoff, up/down symmetric (production at the far side at the conjugate
slant); `solve` is base x (3D production factor E_off x cone-averaged G_s with the far-side production-point cutoffs for
cos Z < 0, `farside_production` / `farside_cutoff_map`). The ratio is independent of the base model, so it is the engine's
`s/t`: 3D flux with cutoff over the 1D no-cutoff flux. Grid: GSF primary (solar-maximum epoch like Honda solmax),
up-going cos Z = -0.95 ... -0.05 (0.1 spacing) x 12 azimuth centres (15 + 30 j deg), averaged to the coarse bins with
the weights base(|cos Z|, E) dE (the engine analogue of a ratio of summed tallies); down-going: cos Z 0.05, 0.15, 0.25 x 12 az and
0.35 ... 0.95 x 6 az (az = 15 + 60 k), used for validation and for the E >= 2.5 GeV down-going bins. New engine inputs:
the G_s families for |cos Z| = 0.35 ... 0.85 (GSF, `rigp1`), cutoff maps for 120 + 42 + 36 cells (database independent).
Check of consistency with the MC: engine F / MC G, azimuth mean, nu_mu (nu_e within ~5% of the same): cos Z 0.05 / 0.15 / 0.25 / 0.35
at E 0.2 GeV 0.73 / 0.76 / 0.92 / 0.95 and at 0.5 GeV 0.65 / 0.75 / 0.85 / 0.94; 0.97-1.05 at every cos Z > 0.45 (E 0.2-3 GeV).

## 5. Up-going validation against Honda (HKKM2014, solmin `honda_kam.npz` and solmax `honda_kam_solmax.npz`)

Details: `out/validate_up.md` (all species). **Up/down at equal |cos Z|, azimuth-averaged**: engine/Honda solmax - 1
(solmin differs by <= 3 points, solmax is closer for |cos Z| > 0.2), mean over the four species; Honda solmax U/D in brackets:

| |cos Z| | 0.3 GeV | 0.5 GeV | 1 GeV | 2 GeV |
|---|---|---|---|---|
| 0.05 | -21.2% (1.03) | -14.4% (1.01) | -6.6% (0.99) | -1.7% |
| 0.15 | -17.0% (1.11) | -10.4% (1.04) | -3.1% (0.99) | -0.2% |
| 0.25 | -7.0% (1.16) | -3.2% (1.07) | +1.1% (0.99) | +0.8% |
| 0.35 | -6.6% (1.16) | -4.2% (1.07) | -0.6% | +0.3% |
| 0.45 | -2.6% (1.18) | -1.2% (1.09) | +0.8% | +0.7% |
| 0.55 | -4.6% (1.30) | -4.2% (1.19) | -2.3% | -1.2% |
| 0.65 | -6.9% (1.42) | -5.0% (1.28) | -0.3% | +1.3% |
| 0.75 | -0.5% (1.40) | +0.8% (1.26) | +4.4% | +3.5% |
| 0.85 | +5.3% (1.28) | +5.1% (1.18) | +5.9% | +3.7% |
| 0.95 | +4.3% (1.31) | +3.9% (1.20) | +4.3% | +2.3% |

The engine reproduces the large low-energy up/down excess (1.2-1.4 at 0.3 GeV for |cos Z| 0.2-1: the far-side cutoffs are lower)
to 3-7% for |cos Z| > 0.2 and misses it near the horizon (the 0.05 and 0.15 bins) by up to -21%.
**Azimuth observables** (12 bins, `analyse_m3b` definitions, up-going cos Z -0.05 / -0.15 / -0.25, E 0.3 / 0.5 / 1 GeV;
engine | solmin | solmax in `out/validate_up.md`): a1/a0 agrees within ~0.08 (cos Z -0.25), 0.10 (-0.15) and is high by 0.14 on average
at -0.05 (engine 0.73-0.76 vs Honda 0.39-0.63 at 0.3 GeV); the phase dphi (geomagnetic West) is off by 5 deg (-0.25), 17 deg (-0.15) and 23 deg (-0.05)
(engine 2-4 deg at -0.05 for every species, where Honda has 7-52 deg); fN/fS/N/S follow at -0.25 (N/S 3.9 vs 3.8/3.3 for nu_mu 0.3 GeV)
but not at -0.05 (N/S 0.73 vs 1.4-1.6). So the engine's up-going azimuth pattern in the near-horizon bin is the down-going
one, not rotated as the far-side geometry requires; for |cos Z| > 0.2 it is usable. The up-going muon-bend term (section 7) is
not in these numbers (`muon_bending = False`).

## 6. Provisional end-to-end: daemonflux(Kamioka) x table vs Honda -- first version, superseded by section 10

Maps (cos Z x azimuth, per species, 0.3/0.5/1 GeV), normalised so that the median ratio at down-going cos Z > 0.5 is 1:
`figs/ratio_map_<provisional|udscaled|engine|1D>_<solmax|solmin>.png`; azimuth-mean profiles: `figs/profile_vs_cosz.png`; numbers:
`out/demo_ratio_table.md` (raw and normalised). **Absolute level**: daemonflux's 1D flux at these energies is
outside its muon-calibration range (E_nu < ~2 GeV) and overshoots Honda's 3D flux; with the table the vertical ratio is
1.37 / 1.36 / 1.17 (nu_mu, 0.3 / 0.5 / 1 GeV, solmax) instead of 2.7 / 2.2 / 1.5 without it. That normalisation is not the table's
business (Phase 1 used the hybrid base for this reason); the shape is. **Shape** (median, normalised; nu_mu, solmax; columns
up 0.5-1 / up 0.2-0.5 / up 0-0.2 / down 0-0.2 / down 0.2-0.5):

| table, E | up .5-1 | up .2-.5 | up 0-.2 | down 0-.2 | down .2-.5 |
|---|---|---|---|---|---|
| 1D only, 0.3 GeV | 0.77 | 1.09 | 0.66 | 0.68 | 0.94 |
| provisional, 0.3 GeV | 1.01 | 1.00 | 0.90 | **1.38** | 1.18 |
| udscaled, 0.3 GeV | 1.00 | 1.12 | 1.21 | 1.43 | 1.18 |
| engine only, 0.3 GeV | 1.00 | 1.00 | 0.83 | 0.97 | 1.04 |
| provisional, 1 GeV | 1.02 | 1.01 | 0.96 | 1.10 | 1.02 |
| engine only, 1 GeV | 1.02 | 1.01 | 0.92 | 0.96 | 0.99 |

(nu_e: provisional down 0-0.2 1.23 at 0.3 GeV; the rest similar. solmin differs by a few points.) With the engine on both
hemispheres the shape is within about 5% of Honda everywhere except the up-going horizon bin (-10 to -17%);
with the MC's down-going G the horizon bin is 38% (0.3 GeV) high and the bin 0.2-0.5 is 18% high, so the literal
provisional table has a +40% / -10% step across the horizon, and the `udscaled` variant, which carries the MC's horizon value to
the up-going side, is worse up-going. **Recommendation (provisional)**: use the MC table only for |cos Z| > 0.35 and the engine below,
or the engine alone, until the horizon definition difference of section 1 is resolved; the MC's fan-out is lost with the engine alone
(muon_bending off). The honest statement is that the provisional table is a plumbing test, not a physics result.

## 7. Design note: transferring the MC muon-bend response to the up-going hemisphere

(No computation; proposal for review. Input: `$W/runs/m4/out/kamioka_Rmu_v1.npz`, `R_s(E, cos Z, az)`, the
ratio of the full 3D flux with the muon field on to the same with it off, per species, down-going at Kamioka.)

**What R_s is.** Muons (and the pions/kaons that make them) are deflected in the geomagnetic field before
they decay; the neutrino direction is that of the parent at decay. For a muon of momentum p travelling a
path L the deflection is delta ~ 0.3 B_perp L / p (B in T, L in m, p in GeV/c). The charge sign decides the
direction of the shift, the neutrino species fix the parent charge (mu+ -> nubar_mu, nu_e; mu- -> nu_mu, nubar_e).
The effect on the flux at a *fixed* arrival direction is a shift of the production cone against the local
direction gradient of the (cutoff-modulated) flux, so to first order

    R_s(E, n) - 1  ~  - q_s * delta(E, n) * d ln Phi_s / d psi (E, n) ,

with psi the angle in the plane perpendicular to the muon direction and B (E-W like), and d ln Phi/dpsi the
cutoff/penumbra gradient of the down-going pattern. Both factors depend on position and direction: delta
through B_perp and the muon path length L(E, n) (the decay length gamma c tau ~ 6 km per GeV of p, cut off
by the atmosphere; L/p, hence delta, is roughly energy independent for decays well inside the atmosphere),
the gradient through the cutoff map.

**Why a direct copy is wrong up-going.** For cos Z < 0 the neutrino is produced on the far side of the
Earth, at a different geomagnetic latitude and longitude (`mceq3d_flux.farside_production` gives the
production point and the *local* down-going arrival direction there). B_perp, its orientation (declination,
inclination), the local zenith (hence L) and the local cutoff gradient all differ from Kamioka's. The up-going
R_s is therefore a property of the far-side site, evaluated in that site's frame.

**Proposed transfer (two steps, each testable).**

1. *Frame mapping.* For each Kamioka up-going bin (E, cos Z < 0, az) take the production point P and the local
   arrival direction (zenith', azimuth') from `farside_production` (the engine already does this for the
   cutoff map). Express the azimuth relative to the local magnetic meridian, az_rel' = az' - declination(P).
2. *Scaling of the measured table.* Read the down-going Kamioka table at the *geometrically equivalent*
   direction, i.e. the same angle between direction and B (and the same local zenith'),
       R_s^up(E, cz, az) = 1 + kappa(P, n') * ( R_s^dn(E, cz'_eq, az_eq) - 1 ),
   with `cz'_eq`, `az_eq` the Kamioka direction whose angle to B and zenith best match (zenith', az_rel' at P),
   and  kappa = B_perp(P, n') / B_perp(Kamioka, n_eq)  the ratio of the field components perpendicular to the
   direction (IGRF at 20 km altitude at both points; the engine's `muon_bending.local_field_enu`). Linear
   scaling in B_perp follows from delta being linear in B_perp at these small deflections (<~ 5 deg); the
   direction of the shift follows from using the table at the matched azimuth relative to the meridian, and
   the sign of the charge dependence is preserved because the species labels are the same.
   An additional correction factor g = [d ln Phi / d psi]_P / [d ln Phi / d psi]_Kamioka(eq) is needed
   when the far-side cutoff gradient differs; it can be taken from the engine's far-side cutoff map
   (`finemap_rc`, full sphere) with the same penumbra width, as a ratio of the engine's own bend response
   evaluated at the two sites: R_s^up = 1 + (R_s^dn - 1) * (r_eng(P) / r_eng(Kam,eq)), where
   r_eng(site) = engine R_s with muon_bending on over off (`solve(..., muon_bending=True/False)`) at that
   site/direction. This reuses the only existing quantity defined everywhere (the engine's term) as a
   *geometry transfer function*, not as a physics input: the MC provides the amplitude at Kamioka, the
   engine provides the ratio between sites. The 3D MC showed the engine's own bend term has the wrong
   magnitude/mechanism at Kamioka, so only the *ratio of two evaluations of the same flawed term* is used,
   and it is flagged as such.

**Validation (in order of cost).**
(a) Closure at Kamioka: transfer the table to a down-going bin of another azimuth/zenith (matched geometry)
and compare with the MC itself (the table is measured on the whole down-going sphere, so this tests kappa and
the mapping with no new MC); (b) a small MC with the injection patch around one far-side production region
(whole-globe injection is inefficient, ~1/d^2 to the caps, but a patch around the production point of one
cos Z ~ -0.2 bin is cheap) gives R_s^up for that bin; (c) compare the up/down muon-bend-induced charge
asymmetry with Honda's up-going nu/nubar azimuth patterns (solmin and solmax, `honda_kam*.npz`), which
contain the true far-side effect (see section 4 for the up-going Honda comparison without the muon term).

**When it matters.** The muon-bend term is a charge-odd (fan-out) effect; averaged over charge it vanishes
to first order. A table delivered with the muon field on for down-going (the MC's per-species s/t) and the
engine's up-going F with `muon_bending = False` is therefore consistent in the species-average flux but lacks the
up-going nu/nubar fan-out; the transfer above is the way to add it. The relevant far-side B_perp is of the same
order as at Kamioka (|B| is 25-65 uT over the globe), so no a-priori reason to neglect it. Magnitudes: to be
read from `kamioka_Rmu_v1.npz` once it exists (nothing is claimed here).

## 8. Reproduction, CPU, tests

```bash
source /home/pigrange/mc3d_work/runs/integ/env.sh ; export M=$W/runs/m4 ; cd $R/tools/mceq3d
# G_s for GSF at |cos Z| 0.35 ... 0.85: diag_rigidity_scan.py chunk gsf <cz> <i> 8  (i = 0..7), then assemble gsf <cz>
python mkmaps_up.py            # $W/runs/integ: cutoff maps of the up-going grid (4 cores, 1.4 h)
$PY diag_integ.py maps-dn 4 ; $PY diag_integ.py engine 4        # engine F, fine grid -> out/engine_F_fine.npz
$PY diag_integ.py table ; $PY diag_integ.py validate ; $PY diag_integ.py demo
$PY -m pytest -q tests ; (cd tools/mceq3d && $PY -m pytest -q)
```

CPU (real, 4 cores at most): up-going cutoff maps 5.8 core-h (5193 s x 4), down-going maps 0.6, G_s chunks 1.6, engine 0.6,
demo/validation/pilot ~0.4, tests ~1.5: about 10 core-h (of 20 allowed). Test results: `tests/` (daemonflux): **52 passed** (14 new in `test_tabulated3d.py`, 18 `test_geomagnetic.py` unchanged); `tools/mceq3d` full suite: **282 passed, 2 skipped, 4 warnings in 2410 s** (unchanged, the 2 skips are the usual `test_joint_cone.py` ones)

## 9. Open questions

1. Horizon: MC `t` against MCEq 1D at cos Z 0-0.2, or the MC's horizon enhancement (section 1). Blocks the use of the MC table there.
2. Up-going near-horizon azimuth pattern (phase, N/S) of the engine; the cone straddles the limb and reads the down-going pattern.
3. Engine uncertainty to put into `factor_error` (5% placeholder).
4. Down-going E >= 2.5 GeV: the MC cells are noisy (10-45%); the engine is used there; production should blend to 1 with the sanity band.
5. Up-going muon-bend response: section 7.

## 10. Common 1D reference for both hemispheres (2026-11-05, PM decision)

**Diagnosis (PM).** The MC's `t` (1D reference) is wrong near the horizon, its 3D flux `s` is fine: normalised to cos Z 0.5-1,
s/Honda solmax is 0.935 / 0.965 / 0.964 (nu_mu, 0.3 GeV, cos Z 0.05 / 0.15 / 0.25), while t is 0.67 / 0.76 / 0.92 of daemonflux's 1D
(about 1.05), inflating s/t by ~1.45. Definition adopted: **`G_s = s / (kappa_s(E) * B_s(E, |cos Z|))`**, with `B_s` the engine's
`MCEq3DFlux.base(|cos Z|)` (MCEq curved atmosphere, GSF, SIBYLL23D, v142p, no cutoff, total = conventional + prompt) for both hemispheres, and
`kappa_s(E)` the MC/MCEq absolute-yield ratio fitted from `t/B` where `t` is reliable (cos Z 0.35-1).

### 10.1 The reference file `out/base1d_kamioka.npz` (`diag_base1d.py`)

* `B` (4, 60, 20): species `nue antinue numu antinumu` (the MC's order, `scoring.SPECIES`), the 60 `sky.E_BINS` (0.1-100 GeV, 20/decade),
  the 20 `sky.CZ_BINS`; value = `int_{E bin} dE <Phi>` in /(cm^2 s sr) (per-bin integral of the intensity), cos Z-averaged over each bin with a
  3-point Gauss rule (nodes 0.05 + 0.1 j + {-0.0387, 0, +0.0387}; the lowest node, cos Z 0.011, is a near-horizontal MCEq solve), the same for both hemispheres
  (up/down symmetric 1D production). Energy integral: log-log interpolation of the engine's 10/decade grid, 6-point Gauss in ln E,
  linear log-log extrapolation below the first grid point.
* `B_t1 = B * 2 pi * 0.1` is on the scale of the MC tally `t1` (measure 1, largest cap) **summed over azimuth**, divided by 12 per azimuth cell.
  **Normalisation check**: `t1` is `int Phi dOmega dE` per bin (`analyse_sky` section (f) divides by `2 pi dcosZ dE` to get the intensity); with this
  convention `t/B_t1` is 0.93-0.97 at cos Z 0.35-1 for E 0.2-5 GeV (below), i.e. the expected yield ratio, so the normalisation is right.
* Metadata keys: `database` (`v142p`), `database_sha256`, `primary`, `interaction_model`, `atmosphere`, `engine_tag`
  (`SIBYLL23D_GlobalSplineFitBeta-None_emin0.1_atmstd_rigp1_dbv142p-7de8275f`), `reference`, `units`; also `species`, `e_bins`, `cz_bins`, `nodes_cz`,
  `engine_e`, `engine_flux_per_cm2` (the raw engine fluxes at the 30 nodes, /(cm^2 s sr GeV)) so the integration can be redone.
* Horizon check (why `t` is wrong there): `t/B_t1` at cos Z 0.05 / 0.15 / 0.25 is 0.34-0.71 / 0.59-0.88 / 0.74-0.99 for E 0.3-1 GeV (all species),
  against 0.9-1.0 for cos Z > 0.35.

### 10.2 kappa_s(E) from pilot-2 (`diag_integ.kappa_and_B`)

`kappa = sum t1 / sum B_t1` over cos Z bins 0.3-1.0 (centres 0.35-0.95) and the four 20/decade bins of each coarse energy bin. Result
(error = sum-of-squares error scaled by sqrt(chi2/dof) of the per-cos Z scatter, which is 6-66 because the tallies are correlated and heavy-tailed):

| species | 0.13 | 0.20 | 0.32 | 0.50 | 0.79 | 1.26 | 2.0 | 3.2 | 5.0 | 7.9 GeV |
|---|---|---|---|---|---|---|---|---|---|---|
| nue | 1.038 | 0.965 | 0.959 | 0.962 | 0.934 | 0.952 | 0.944 | 0.980 | 0.968 | 0.836 |
| antinue | 1.026 | 0.951 | 0.945 | 0.935 | 0.965 | 0.937 | 1.002 | 0.931 | 0.995 | 0.909 |
| numu | 1.005 | 0.969 | 0.944 | 0.935 | 0.947 | 0.939 | 0.968 | 0.947 | 0.944 | 0.927 |
| antinumu | 1.012 | 0.956 | 0.952 | 0.939 | 0.927 | 0.941 | 0.938 | 0.949 | 0.945 | 0.849 |

(Sum-of-squares errors 0.2-2%; scaled errors 1-4% below 3 GeV, up to ~9% at 3-8 GeV; the per-energy errors are stored in the table file as `kappa_error`; the weighted
mean for E < 3 GeV is 0.965 / 0.962 / 0.955 / 0.953.) The pilot-2 yield ratio is therefore 0.93-0.97 over 0.2-5 GeV and 1.0-1.04 at the lowest bin
(the PM's closure ~0.92 is the same quantity measured elsewhere). **Flatness of t/B in cos Z over 0.35-1**: the slope of the relative `t/B` per unit cos Z is
within +-0.2 (i.e. +-10% across the range) for 33 of 40 (species, E) cells (outliers: nue 5-8 GeV, antinue 0.2 GeV and 3-8 GeV, numu 8 GeV, antinumu 0.2 GeV)
(e.g. nu_mu 0.5-2 GeV: slopes +0.06, -0.09, 0.00, +0.01); the 0.13-0.32 GeV bins have a mild positive slope (+0.1 to +0.2),
and 3-8 GeV is noise-dominated. `kappa` itself is read from the whole range, so a +-10% tilt changes it by ~3%, covered by the scaled error.

### 10.3 The rebuilt provisional tables

`kamioka_3D_provisional.npz` (down-going E < 2.5 GeV: pilot-2 `s` over `kappa B`, jackknife `s` errors plus the `kappa` error, 2-4% at cos Z > 0.2, 3-8% in the horizon bin;
everything else the engine `F = solve/B`, 5% placeholder error) and `kamioka_3D_provisional_engine.npz` (engine on both hemispheres), both PROVISIONAL, same format. The
`udscaled` variant is withdrawn. The file also stores `kappa`, `kappa_error`. Azimuth-mean factor, nu_mu, 0.32 GeV, cos Z bins -1 ... 1: 0.66 0.69 0.60 0.56 0.62 | 0.71 0.52 0.51 0.53 0.52
(before: the step across cos Z = 0 was 0.9 -> 1.4; now 0.62 -> 0.71).

### 10.4 Section-6 shape table, redone (daemonflux(kamioka) x table / Honda solmax, normalised to the median ratio at down-going cos Z > 0.5; medians over azimuth in the columns)

| species | E | table | up .5-1 | up .2-.5 | up 0-.2 | down 0-.2 | down .2-.5 | down .5-1 |
|---|---|---|---|---|---|---|---|---|
| nu_mu | 0.3 | provisional (MC down, common ref) | 0.99 | 0.98 | 0.81 | **0.94** | **1.00** | 1 |
| nu_mu | 0.3 | engine only | 1.00 | 1.00 | 0.83 | 0.97 | 1.04 | 1 |
| nu_mu | 0.3 | *first version (MC s/t)* | 1.01 | 1.00 | 0.90 | 1.38 | 1.18 | 1 |
| nu_mu | 0.5 | provisional | 0.99 | 1.00 | 0.86 | 0.92 | 1.00 | 1 |
| nu_mu | 0.5 | engine only | 1.00 | 1.02 | 0.88 | 0.95 | 1.02 | 1 |
| nu_mu | 1.0 | provisional | 1.03 | 1.01 | 0.91 | 0.93 | 0.96 | 1 |
| nu_mu | 1.0 | engine only | 1.02 | 1.01 | 0.92 | 0.96 | 0.99 | 1 |
| nu_e | 0.3 | provisional | 1.00 | 1.05 | 0.82 | 0.93 | 1.00 | 1 |
| nu_e | 0.3 | engine only | 1.00 | 1.05 | 0.83 | 1.01 | 1.01 | 1 |
| nu_e | 0.5 | provisional | 1.02 | 1.04 | 0.85 | 1.01 | 0.98 | 1 |
| nu_e | 1.0 | provisional | 1.02 | 1.01 | 0.94 | 1.01 | 0.97 | 1 |

The horizon step is gone: the down-going 0-0.2 column is 0.90-1.02 (was 1.23-1.38 at 0.3 GeV) and 0.2-0.5 is 0.96-1.04 (was 1.18), within ~8% of Honda's shape. nubar_mu and nubar_e
are the same within 3-5 points (`out/demo_ratio_table.md`, `figs/ratio_map_*`, `figs/profile_vs_cosz.png`; solmin differs by a few points). What remains is the **engine's
up-going near-horizon bin** (0.81-0.94 of the Honda shape: -6 to -19%), unchanged since section 5. The MC and the engine now agree on the shape at the horizon within
the ~5% placeholder error, so the MC-engine joint is no longer a problem; the engine's up-going near-horizon pattern is.

### 10.5 Code fixes in the interface (`src/daemonflux`)

* `_zenith_1d`: with a tabulated model an array of zeniths (down- or up-going) is rejected up front with `NotImplementedError("A tabulated 3D model supports
  a single zenith angle per call ... loop over the zenith angles")` (it used to fail with an unrelated range error for arrays above 90 deg); scalar
  up-going zeniths are mapped as before.
* Quantities the table does not cover (`muflux`, `muratio`, `mu+`, `mu-`, with or without `total_`, and any sum/ratio none of whose species is in the table)
  pass through with factor 1 and zero table error, with no warning and no zenith-coverage check (`Tabulated3DModel.covers`); sums/ratios with only one species
  covered use 1 for the other. The docstring now says that the factor is also applied to `total_` quantities, i.e. to their prompt part (negligible below ~10 GeV;
  the factor blends to 1 above the table). README updated.
* Tests (`tests/test_tabulated3d.py`): `test_array_zenith_with_a_tabulated_model_is_rejected_clearly`, `test_uncovered_quantities_pass_through_with_factor_one`
  (muon quantities identical to 1D for scalar/None/array azimuth at down- and up-going zeniths, errors, zero table error; a partial-species table).
  daemonflux suite: **54 passed**; black/flake8 clean. The `tools/mceq3d` suite was not rerun (no `mceq3d` module or test changed; the new files are `diag_base1d.py`
  and edits to `diag_integ.py`): last result 282 passed, 2 skipped.

### 10.6 Reproduction and CPU

```bash
source $W/runs/integ/env.sh ; export M=$W/runs/m4 ; cd $R/tools/mceq3d
for i in 0 1 2 3; do $PY diag_base1d.py nodes $i 4 & done ; wait ; $PY diag_base1d.py assemble   # out/base1d_kamioka.npz
$PY diag_integ.py table ; $PY diag_integ.py demo
```

CPU: base1d 30 MCEq solves on 4 processes about 0.6 core-h; table/demo ~0.3: about 0.9 core-h in this step (cumulative about 11 core-h of 20).

## 11. The engine's up-going near-horizon bin: mechanism and fix (2026-11-05, later)

### 11.1 Mechanism

* **(a) Where the production point is, which map is used.** `farside_production` (h = 20 km) puts the production point Q on the chord exit,
  **8.2 deg = 915 km** from Kamioka at cos Z = -0.05, 5.1 deg (572 km) at -0.01, 13 deg (1451 km) at -0.10, 18 deg (2042 km) at -0.15 and 30 deg
  (3295 km) at -0.25; the local zenith of the primary at Q is cos = 0.093 / 0.127 / 0.169 / 0.261 (less grazing than |cos Z|: the chord
  is asymmetric because production is at 20 km altitude). There is **no threshold and no fallback**: `cutoff_grid` selects the far-side
  map purely by the sign of cos Z, and `farside_cutoff_map` launches the back-trace from Q at 20 km along the primary direction -d. This map
  already carries the production-point anchoring that the down-going path gets from `cutoff_anchor="prod_point"` (the primary's entry is
  displaced, here by 570-2000 km).
* **The actual defect: no production cone.** In `solve`, `cone_geff` (line "up-going: single far-side cutoff (no cone yet)") and
  `joint_cone_factor` ("up-going rows come back as the unchanged product") skip every cos Z < 0 row, so an up-going cell is
  `base x E_off(|cos Z|) x G_s(single far-side R_c)`. The cone average over the production angle is what compresses the East-West
  pattern and rotates it (PHASE1_RESULTS 6.1/6.2, 8.2). Experiment (`diag_upgoing_fix.py modes/report`, down-going rows 0.05/0.15/0.25 solved three
  ways, nu_mu 0.5 GeV, a1/a0 | dphi [deg] | N/S | W/E):

  | cos Z 0.05 | single (E_off x one detector cutoff) | + cone (detector anchor) | + cone + production-point anchor (delivered) | up-going engine (far-side single) | Honda up-going, solmax |
  |---|---|---|---|---|---|
  | a1/a0 | 0.71 | 0.49 | 0.48 | 0.66 | 0.45 |
  | dphi | -22.1 | -10.8 | +2.7 | +3.1 | +29.9 |
  | N/S | 0.44 | 0.70 | 0.87 | 0.76 | 1.32 |
  | W/E | 4.62 | 2.97 | 2.87 | 4.33 | 2.30 |

  The up-going engine has the anchor (dphi +3 instead of -22: Q is far from the site) but not the cone (a1/a0 0.66 vs 0.49; W/E 4.3 vs 3.0). The
  cone also rotates the pattern further (the +12 deg step from `single` to `cone`), which is why the delivered *down-going* dphi of +2.7
  deg is right for the down-going Honda pattern but the up-going pattern, whose Honda phase is +30 deg, needs the cone on top of the anchor.
* **(b) Azimuth frame.** No frame mapping is involved in the cutoff: the back-trace takes the 3D direction vector, and the table is indexed by the
  detector's arrival azimuth. The meridian rotation between Q and Kamioka is small in the near-horizon bins: the local azimuth at Q differs from the detector azimuth by 3.8 /
  6.0 / 9.5 / 13 / 20 deg (at detector azimuth 90 or 270) for cos Z -0.01 / -0.05 / -0.10 / -0.15 / -0.25 (N/S: 0 or 180 unchanged), so
  it is not a frame error. Not the cause.
* **(c) cos Z weighting.** `B` uses 3 Gauss nodes per MC bin (lowest node cos Z = 0.011) but the engine up-going `F` is a single row at
  the bin centre (-0.05, -0.15, ...), two rows averaged per 0.2 coarse bin. The row-to-row change of the (uncorrected) `F` is <= 10%
  (nu_mu 0.3 GeV: 0.62 at -0.1, 0.56 at -0.3), so the centre-row sampling is a few-percent effect at most; the node at cos Z = 0.011 was not
  computed for `F` (it needs a G_s family there, ~7 core-h), so a residual horizon-edge effect is not excluded but cannot be the -15..-21%.

### 11.2 Fix: `solve(upgoing_cone="mirror")`

New option (default `UPGOING_CONE = "none"`: legacy behaviour, no existing number changes; nothing in the G_s / production / map caches
depends on it): up-going rows with `|cos Z| <= upgoing_cone_czmax` (default 0.35) are multiplied by `F_cone / F_single` of the *mirrored down-going*
direction (same |cos Z| and azimuth; both **detector-anchored**, so only the cone effect is transferred, the production-point anchoring already
being in the far-side map; `F_cone` with the caller's cone/joint options, `F_single` = `cone_cutoff=False, joint_cone=False`). The
down-going maps and G_s of those directions are cached like for any down-going solve. Assumption (tested below): the cone correction of an
up-going direction at the far-side point Q is that of the down-going direction with the same |cos Z| and azimuth at the site. The returned dict
has `upgoing_cone_ratio`. Tests: `test_upgoing_cone.py` (3: row selection, the two extra solves' options, validation, default).

Honda validation (solmax; `out/validate_up_mirror.md`). **Caveat on what is independent**: the cut `czmax = 0.35` was chosen by looking at
this very comparison (beyond it the all-rows correction made U/D worse at |cos Z| 0.55, -4.6% to -11%), so only the rows **-0.05 and -0.15** are a test that does
not depend on that choice; the rows -0.25 and -0.35 are within the fitted range and are not independent evidence. The sign change of the correction
between |cos Z| 0.35 (about +5%) and 0.55 (-5..-11%) means the mirror is unreliable beyond the horizon region, not merely "least good" there
(11.5):

| | before | after |
|---|---|---|
| U/D engine/Honda - 1, |cos Z| 0.05 (0.3 / 0.5 / 1 GeV) | -21 / -14 / -7% | **-2.8 / -0.4 / +1.5%** |
| U/D, 0.15 | -17 / -10 / -3% | **-2.7 / -0.6 / +1.3%** |
| U/D, 0.25 | -7 / -3 / +1% | +6.6 / +5.0 / +4.1% |
| U/D, 0.35 | -6.6 / -4.2 / -0.6% | +1.0 / +0.6 / +1.3% |
| mean(engine - Honda) dphi at cos Z -0.05 / -0.15 / -0.25 | -22.8 / -17.1 / -4.9 deg | **+0.3 / +0.2 / +5.9 deg** |
| mean ln W/E (engine/Honda) at -0.05 / -0.15 / -0.25 | +0.44 / +0.31 / +0.25 | **-0.10 / +0.07 / -0.06** |
| mean ln N/S at -0.05 / -0.15 / -0.25 | +0.19 / +0.05 / -0.06 | +0.04 / -0.09 / -0.16 |
| mean a1/a0 (engine - Honda) at -0.05 / -0.15 / -0.25 | +0.143 / +0.103 / +0.081 | **-0.011** / +0.068 / +0.100 |

(mean over four species and E = 0.3, 0.5, 1 GeV; the rms spread of a1/a0 stays 0.10-0.13 and of dphi ~10 deg: the species-dependent
fan-out of the muon bend is not in the engine, as before.) Example, nu_mu, cos Z -0.05, 0.3 / 0.5 / 1 GeV, a1/a0 | dphi | N/S | W/E:
engine 0.74/0.66/0.52 | 3.9/3.1/2.0 | 0.73/0.76/0.81 | 5.8/4.3/3.1 -> corrected 0.53/0.50/0.42 | 35/26/16 | 1.41/1.20/1.02 | 2.5/2.6/2.4;
Honda solmax 0.45/0.45/0.34 | 40/30/19 | 1.54/1.32/1.09 | 2.1/2.3/2.0. Remaining: the +5..+7% U/D overshoot at |cos Z| = 0.25 and a1/a0
too high by 0.07-0.10 at -0.15/-0.25 (the mirror assumption is least good where the far-side point is far from Kamioka: Q is 18-30 deg away).

### 11.3 Provisional table rebuilt with the correction (rows <= 0.35)

Shape against Honda solmax (daemonflux x table, normalised at down-going cos Z > 0.5; medians over azimuth; before the correction in brackets):

| species | E | up .5-1 | up .2-.5 | up 0-.2 | down 0-.2 | down .2-.5 |
|---|---|---|---|---|---|---|
| nu_mu | 0.3 | 0.99 (0.99) | 1.07 (0.98) | **0.92 (0.81)** | 0.99 | 1.00 |
| nu_mu | 0.5 | 0.99 | 1.03 (1.00) | 0.93 (0.86) | 0.97 | 1.00 |
| nu_mu | 1.0 | 1.03 | 1.03 (1.01) | 0.97 (0.91) | 0.95 | 0.96 |
| nu_e | 0.3 | 1.00 | 1.14 (1.05) | **1.00 (0.82)** | 0.96 | 1.00 |
| nu_e | 0.5 | 1.02 | 1.11 (1.04) | 1.06 (0.85) | 1.01 | 0.98 |
| nu_e | 1.0 | 1.02 | 1.03 (1.01) | 1.01 (0.94) | 1.01 | 0.97 |
| nubar_mu | 0.3 | 1.00 | 1.12 | 1.02 | 1.03 | 1.02 |
| nubar_e | 0.3 | 1.01 | 1.10 | 0.99 | 0.98 | 1.01 |

The up-going 0-0.2 bin is now within -8% .. +6% (was -6 .. -19%), and the whole hemisphere is within about 10% of Honda's shape; the
cost is an +7..+13% overshoot of the 0.2-0.5 bin at 0.3-0.5 GeV (rows -0.25 and -0.35). `kamioka_3D_provisional.npz` and
`kamioka_3D_provisional_engine.npz` are rebuilt (`diag_integ.py table`; `MIRROR_CZMAX = 0.35`); `figs/` regenerated.

### 11.4 Reproduction, CPU, tests

```bash
$PY diag_upgoing_fix.py modes ; $PY diag_upgoing_fix.py modes dn6 ; $PY diag_upgoing_fix.py report   # mechanism experiment
$PY diag_integ.py validate ; $PY diag_integ.py validate mirror ; $PY diag_integ.py table ; $PY diag_integ.py demo
```
CPU: 4 solve passes (down-going cone / single, 12 and 6 azimuths) about 0.3 core-h; nothing needed new maps (all cached). Cumulative: about 11.5 of 20 core-h.
Tests (`tools/mceq3d`, full suite, workers capped at 4 through the pytest plugin `$W/runs/integ/capjobs.py`, which wraps `geomag_backtrace.cutoff_from_states`; no test file edited): **285 passed, 2 skipped** (282 + the 3 new `test_upgoing_cone.py`; the run was done in two parts, 280 passed + 5 failed from a bug of the first version of the plugin, then the two files with those 5 tests rerun: 34 passed). The first attempt ran the `n_jobs=8` back-trace tests with 8 workers (core-cap breach, killed after about 5 min and rerun capped).

**n_jobs.** `geomag_backtrace.cutoff_from_states` defaults to `os.cpu_count()//2` workers (24 on this node) when `n_jobs=None`, and `solve`/`cutoff_grid`/`finemap_rc`/`prod_family_rc` pass `None` through. In this work every back-trace call passed `n_jobs <= 5` (`mkmaps_up.py` 4, `diag_integ maps-dn/engine` 4, `diag_upgoing_fix` 1, the earlier `mkmaps` of the rigidity task 5, one-off over by 1); `_solve_upgoing_mirror` forwards the caller's `n_jobs` unchanged, so `n_jobs=None` with uncached maps would use 24. Proposal (not applied): in `cutoff_from_states` use `n_jobs = min(int(os.environ.get("MCEQ3D_NJOBS", 4)), cpu_count()//2)` when `None`, so the library default is 4 and the environment can raise it; explicit larger values in tests (8) would still be honoured unless capped by the same variable.

### 11.5 Step check at the cut (|cos Z| = 0.35 / 0.45)

`F_cone/F_single` of the down-going mirror directions (`updiag_modes*.npz`), median over azimuth, E = 0.3 / 0.5 / 1 GeV:

| species | 0.35: 0.3 | 0.5 | 1.0 | 0.45: 0.3 | 0.5 | 1.0 |
|---|---|---|---|---|---|---|
| nue | 1.073 | 1.044 | 1.015 | 1.002 | 1.007 | 1.008 |
| antinue | 1.067 | 1.038 | 1.011 | 0.999 | 1.006 | 1.006 |
| numu | 1.066 | 1.037 | 1.012 | 1.000 | 1.007 | 1.006 |
| antinumu | 1.067 | 1.037 | 1.011 | 1.000 | 1.007 | 1.006 |

The azimuth-mean ratio (nu_mu, 0.3 GeV) vs |cos Z| 0.05 ... 0.95 is 1.43, 1.25, 1.15, 1.09, 1.04, **0.976**, 1.02, 1.02, 1.02, 1.00: a smooth decline through 1 at
about 0.5, then a dip below 1 at 0.55 and a plateau at +2% that is not a cone effect I can interpret (noise or an interaction with the 6-azimuth sampling and the
zenith nodes of the map); so the cut removes a tail of +4% (0.45) and replaces the unreliable 0.55+ values by 1. The cut itself therefore creates a step of
at most the +4% at 0.3 GeV (+0.7% at 1 GeV) between the rows 0.35 and 0.45.

Up-going coarse-bin flux ratio (0.2-0.4)/(0.4-0.6) (engine flux, azimuth mean, nu_mu; rows -0.35,-0.25 over -0.55,-0.45), engine before | after the correction | Honda solmax | solmin:

| species | 0.3 GeV | 0.5 GeV | 1 GeV |
|---|---|---|---|
| nue | 1.004 / 1.133 / 1.044 / 1.041 | 0.980 / 1.055 / 1.013 / 1.010 | 1.023 / 1.051 / 1.041 / 1.038 |
| antinue | 1.010 / 1.130 / 1.036 / 1.035 | 0.987 / 1.055 / 1.008 / 1.006 | 1.033 / 1.058 / 1.042 / 1.040 |
| numu | 0.973 / 1.087 / 1.012 / 1.010 | 0.945 / 1.008 / 0.971 / 0.968 | 0.962 / 0.985 / 0.977 / 0.974 |
| antinumu | 0.979 / 1.094 / 1.017 / 1.016 | 0.949 / 1.013 / 0.977 / 0.975 | 0.973 / 0.994 / 0.992 / 0.990 |

Relative to Honda solmax the step is -3.9 / -2.7 / -1.5% (nu_mu, 0.3 / 0.5 / 1 GeV) before and **+7.4 / +3.8 / +0.8% after** (nu_e: -3.8 -> +8.5% at 0.3 GeV, -3.3 -> +4.2% at 0.5 GeV), i.e. beyond
Honda's by more than 3% at 0.3 and 0.5 GeV. Where it comes from: not the cut (the cut contributes at most the +4% tail of 0.45 and a 0 correction at 0.55, which acts in the
opposite direction on the bin 0.4-0.6), but the correction itself: the mirrored ratio at the rows 0.25/0.35 (about +15% / +9% mean at 0.3 GeV) is larger than the
cone effect that Honda's up-going flux shows there (U/D +6.6% at 0.25 after the correction). The mirror assumption (cone effect of the up-going direction = that of the
down-going direction at the same |cos Z| at the site) degrades with the distance of the far-side point Q (30 deg at |cos Z| 0.25, 40+ deg at 0.35).
**Proposal (not applied, nothing tuned against Honda):** taper the correction with the distance of Q, `r_eff = r^w`, `w = clip(1 - d_Q/d_max, 0, 1)`, with `d_Q` the
Kamioka-Q angle (8.2 / 18 / 30 deg for |cos Z| 0.05 / 0.15 / 0.25) and `d_max` fixed a priori by the scale over which the geomagnetic cutoff map changes (a few tens of
degrees; e.g. 45 deg), which replaces the hard cut at 0.35 by a continuous weight; or the local-zenith variant of 11.7.

### 11.6 Error of the engine-filled bins (replaces the 5% placeholder)

`sigma_rel = max(5%, |median shape residual vs Honda solmax|)` per species x E band (0.1-0.3, 0.3-1, 1-3, 3-10 GeV by coarse-bin centre) x coarse cos Z bin
(`diag_integ.py residual` -> `out/engine_sigma.md/.npz`): model = daemonflux(kamioka) x the engine table, ratio to Honda solmax per species and coarse energy bin normalised to its median over
the down-going cos Z > 0.5 cells (removes daemonflux's absolute level), then median over azimuth, the two Honda cos Z bins of a coarse bin and the E bins of the band. It is set on the
**engine-filled bins only** (all up-going bins, down-going E >= 2.5 GeV; 65 of the 100 (E, cos Z) bins of the main table; every bin of the engine-only variant); the MC bins keep their jackknife
errors. Result: sigma_rel 5% almost everywhere, 6-15% in the up-going bins -0.6..0 at 0.1-1 GeV (largest: nubar_mu 0.1-0.3 GeV, cos Z -0.4..-0.2: 15%; nu_e 13%; nu_mu 12%), median 5%, maximum 14.8%.
**This is a cross-model systematic**: Honda is a comparison model with its own ~5-10% shape uncertainty and is not the truth; the number measures the engine-Honda disagreement, not the engine's error.
Both provisional npz (`kamioka_3D_provisional.npz`, `..._engine.npz`) are rebuilt with it.

### 11.7 Cost of a principled up-going cone at the far-side point Q (estimate, not run)

* **Needed**: for each up-going row |cos Z| <= 0.35 and azimuth, the cutoff of the cone of primary directions `n'` at angle alpha, beta around the axis, **launched from Q** (not from the detector or the
  chord exit of `n'`), i.e. a site-Q map R_c(zenith', azimuth') of the neighbourhood of the axis, plus the production weights `q(alpha, beta)` of the mirrored geometry
  (these are the existing down-going `cone_production_multi` values for the local zenith at Q, but the correspondence beta <-> detector frame must be derived; a
  reflection error would flip the sign of the rotation). The delivered cone has 44 x 18 = 792 directions per cell.
* **Cost** (measured: 120 up-going cells took 5193 s on 4 cores = 2.9 core-min per cutoff scan): a coarse cone of 6 alpha x 8 beta + axis = 49 directions, 4 rows x 12 azimuths = 48 cells: 2352 back-traces =
  **~114 core-h = 28 h at 4 cores**. With 24 cells (6 azimuth bins of 60 deg) and a 4 x 6 + 1 = 25-direction cone: 600 scans = **~29 core-h = 7 h at 4 cores**. The delivered 792-direction cone would be ~1900 core-h. The band
  trick of `prod_family_rc` (seeding the rigidity scan from the neighbouring direction, ~4x) could bring the 25-direction variant to ~7 core-h (2 h at 4 cores). All of these exceed the 3 core-h of this step and need new
  code in `farside_cutoff_map` (arbitrary launch point and direction list), a site-Q interpolation or direct lookup in `joint_cone`, and the frame derivation.
* **Cheaper variants**: (i) *matched local zenith* (no new back-traces): read the mirror ratio at the down-going |cos Z'| equal to the local zenith cosine at Q (0.093 / 0.169 / 0.261 for |cos Z| 0.05 / 0.15 / 0.25)
  by interpolating the cached rows (0.05, 0.15, 0.25, 0.35) -- the geometry at the production point is then the right one; costs minutes. (ii) The distance taper of 11.5. (iii) Down-going maps at the three or four exact |cos Z'| values
  (12 azimuths each = 48 cells, ~2.3 core-h) plus their G_s families (the 0.09 one is the slow one, ~2 core-h) for variant (i) without interpolation. None of these is applied or tuned.


## 12. Final Kamioka table v1 (2026-11-06)

File: `$W/runs/integ/out/kamioka_3D_v1.npz` (`Tabulated3DModel` format, label `kamioka_3D_v1`, v142p database, IGRF14).
Build/validation tables: `out/v1_build_tables.md`, `out/v1_validation.md`. Reproduce with
`python tools/mceq3d/diag_v1.py fit`, `... build`, `... validate` (well under 1 core-h in total; no new engine or MC runs).

* **Sources per cell** (`source` key): 0 = MC down-going (E < 2.5 GeV), 1 = engine down-going (E >= 2.5 GeV),
  3 = engine mirror cone (up-going) multiplied by R_up. Common 1D reference B = engine `base(|cosZ|)` (section 10).
* **R_up route (a')**: the muon-bend response R = F_on/F_off is measured only down-going at Kamioka. The literal
  B_perp test failed (correlation 0.00). A projection model, R-1 = delta_s(E, cosZ) q_s [beta . grad ln F_off] with
  beta = d x B in the detector tangent basis, explains the Kamioka R table (chi2 689 for 840 points, 140 parameters).
  Route a' transfers delta_s to the up-going production point Q (field from ppigrf at 20 km). The transfer to Q is
  untested. sigma_R = stat + 32% closure systematic on (R_a - 1) + |R_a - R_b|, where R_b is the plain mirror of the
  down-going R. R_up is set to 1 above the R-table range (2.5 GeV); this leaves a step there.
* **Caveat seen in the last check**: at the up-going horizon bin (-0.2..0) the azimuth-mean (R-1) of route a' differs
  from the plain mirror (route b), which tracks the measured down-going row. E.g. nue at 0.5-1.3 GeV: a' -3.5 to -6.2%,
  b +0.2 to +2.2%, down +0.5 to +2.3%; antinue sign is opposite in places. The envelope term in sigma_R covers this
  by construction but the choice a' versus b is a model decision, not validated.
* **MC/engine at 2.5-4 GeV**: consistent except antinue in the vertical band (0.940 +- 0.012, about 5 sigma low).
* **Errors**: `factor_error` = MC total (stat + reference shape) or engine sigma, and for up-going also sigma_R.
  Max relative error 0.59, 282 of 2400 bins above 10%. Up-going near-horizon medians are 11-19% at 0.3-3 GeV.
* **Honda comparison** (shape only, nothing tuned): numu horizon up/down ratio is 2-5% above Honda at 0.3 and 0.5 GeV;
  azimuth phases differ in places (nue, 0.5 GeV, -0.05 in cos Z: 49.7 vs 15.8 deg).
* **API check**: `Flux` with the table loads (test `test_kamioka_v1_table_if_available`, skips if the file is absent);
  daemonflux suite 55 passed, black and flake8 clean. The `tools/mceq3d` suite was not rerun (nothing changed there).
* **Not validated**: R_up transfer beyond Kamioka; up-going patterns beyond the horizon rows; neglect of R for
  E >= 2.5 GeV; the engine-based E >= 2.5 GeV down-going bins.

### 12.x v1b (2026-11-06, after PM review): what changed and why

`kamioka_3D_v1.npz` is superseded by `$W/runs/integ/out/kamioka_3D_v1b.npz` (E 5/dec x cos Z 0.1 x az 12; label `kamioka_3D_v1b`).
Build: `diag_v1.py fit` (unchanged), `build_v1b`, `validate_v1b` -> `out/v1b_build_tables.md`, `out/v1b_validation.md`.

* **Bug in R_up (route a'): reflection of the theta gradient.** The muon bends at the production point Q, where the muon
  direction is the reflection of the detector direction (theta_Q = 180 - theta_det along the chord family, azimuth unchanged).
  A rotation of the ray by alpha changes both theta_Q and theta_det by +alpha, but the flux depends on the muon direction at
  Q, so d ln F / d theta_Q = - d ln F / d theta_det. v1 used the detector-frame gradient for both terms, which flipped the
  sign of the theta term for up-going rows (R-1 had the opposite azimuth pattern to the measured down-going R, hence the
  reversed mu+/mu- ordering of a1/a0). Fixed in `r_up_fine`: P = q (-b_theta d_theta + b_phi d_phi / sin theta). The
  azimuth term is unchanged. The sign was derived, not fitted; the ordering was checked afterwards (table C).
* **Route decision: a'** (the fix reproduces the measured mu+/mu- ordering at the horizon), sigma_R = stat (+) 32% closure
  (+) |a' - mirror|, with the mirror now the measured down-going R interpolated in cos Z and by harmonics 0-2 in azimuth.
  R_up is evaluated from the fitted model on the 12-azimuth grid (no linear interpolation of the 6 R bins).
* **Grid and sources**: up-going = engine rows -0.05 ... -0.95 x 12 az (mirror cone, |cos Z| <= 0.35) x R_up; down-going
  E < 3.98 GeV = MC (`kamioka_G_down_v1b.npz`), E >= 3.98 GeV = engine (rows 0.05-0.25 native 12 az, rows 0.35-0.95
  extended from 6 to 12 az by harmonics 0-2). R_up = 1 above 2.5 GeV (end of the R table).
* **Package**: validated with the patched `Tabulated3DModel` (separate hemispheres at the cos Z = 0 edge). B1 now reads the
  table up/down ratio and the model ratio at the bin values; they agree (the model no longer mixes hemispheres).
  Tests: 55 passed, 1 skipped (`DAEMONFLUX_3D_TABLE_V1` accepts v1 and v1b).

Validation of v1b (see `v1b_validation.md`):
* Table C, -0.05, 0.5 GeV, a1/a0 v1b vs Honda solmax: nue 0.700/0.685, antinumu 0.610/0.606, numu 0.397/0.447, antinue 0.300/0.360
  (ordering as in Honda and in the measured down-going pattern; v1: reversed). dphi at -0.05 agrees to 1-7 deg except nue
  1.0 GeV (21 vs 6 deg). At -0.25 a1/a0 and N/S are 5-25% above Honda (nue 0.3 GeV a1/a0 0.89/0.77, N/S 6.0/3.6).
* Closure test of the projection model with the cutoff gradient (down-going, engine F_off gradient vs the pattern of MC G /
  engine F, azimuth part only): correlation 0.6-0.9 and slope ~1 at |cos Z| < 0.2 and E < 1 GeV (numu, antinumu good,
  nue/antinue weaker), but no correlation (negative) at |cos Z| >= 0.25. So the cutoff-gradient part of R_up is supported
  only near the horizon; at |cos Z| 0.4-0.8 and 0.3-1 GeV R_up rms is 10-16% against 1.5-4% for the plain mirror, and
  sigma_R (median 6-10%) is dominated by that difference. NOT validated there.
* Az-mean bias of a': the az-mean of R_up at -0.05 is +4..+10% for mu+ daughters (nue, antinumu) and -2..-10% for mu-
  daughters, versus ~0 for the mirror. Consequently the table up/down ratio at |cos Z| 0.05, 0.5 GeV is nue 1.14, antinumu
  1.08 (Honda 1.00), antinue 0.99, numu 0.99 (Honda 1.02, 1.00). Not corrected; the choice (normalise R_up to the az-mean of
  the mirror, or use the mirror beyond |cos Z| ~ 0.3) is left to the PM.
* Continuity at 2.5 / 3.98 GeV: F(2.4)/F(2.6) = 0.97-1.00 (up and down), F(3.8)/F(4.2) = 0.975-0.997, i.e. the E falloff plus
  steps below 3%. MC/engine in the overlap bins (az mean): 2.5-4 GeV within 0.94-1.03; 4-6.3 GeV 0.89-1.05 (horizon band
  0.89-0.96 low); 6.3-10 GeV 0.86-1.13 (noisy).
* Error map: max relative error 0.73; 1276 of 9600 bins above 10%. Medians: up |cos Z|<0.2: 10% / 9% / 14% (0.1-0.3 / 0.3-1 / 1-3
  GeV); up -0.2..-0.5: 12% / 10% / 6%; up < -0.5: 8% / 5.5% / 5%; down 0-0.2: 4% / 4% / 6%; down 0.2-0.5: 2% / 2.5% / 3%;
  down > 0.5: 1% / 1.4% / 2.2%; all E 3-10 GeV engine bins 5%.
* Not validated: R_up transfer beyond Kamioka and beyond |cos Z| ~ 0.2; up-going patterns beyond the horizon rows; R for
  E >= 2.5 GeV (neglected); the engine-based E >= 3.98 GeV down-going bins.
* CPU: well under 0.2 core-h (no engine or MC runs).

### 12.y v1c (2026-11-06, PM decision on R_up): hybrid azimuth pattern

`$W/runs/integ/out/kamioka_3D_v1c.npz` supersedes v1b; everything else is as in v1b (grid, sources, sigma_R, errors).
Build: `diag_v1.py build_v1c`, `validate_v1c` -> `out/v1c_build_tables.md`, `out/v1c_validation.md`.

* **Central R_up**: R_up = R_m + w(E, |cos Z|) [(R_a' - <R_a'>_az) - (R_m - <R_m>_az)], R_m = measured down-going R mirrored
  (cos Z linear, azimuth harmonics 0-2, log E), <.>_az = mean over the 12 azimuths at fixed (row, E). The azimuth mean always
  comes from the measured R (the a' mean term is not validated). The azimuth-odd a' pattern enters with w = w_cz w_E:
  w_cz = 1 at |cos Z| <= 0.15, linear to 0 at 0.25; w_E = 1 at E <= 0.8 GeV, linear to 0 at 1.25 GeV.
* **Thresholds from the closure test** (down-going, engine F_off gradient with cutoff vs azimuth pattern of MC G / engine F;
  median over species of the correlation, slope in brackets):

  | E [GeV] | cos Z 0.05 | 0.15 | 0.25 | 0.55 |
  |---|---|---|---|---|
  | 0.20 | 0.68 (1.13) | 0.65 (0.97) | 0.46 (1.29) | -0.08 |
  | 0.32 | 0.70 (0.73) | 0.79 (0.86) | 0.59 (0.98) | -0.32 |
  | 0.50 | 0.77 (0.61) | 0.73 (0.84) | 0.20 (0.60) | -0.29 |
  | 0.79 | 0.75 (1.11) | 0.66 (0.85) | -0.21 | -0.20 |
  | 1.26 | 0.58 (0.72) | 0.23 (0.31) | -0.51 | -0.10 |
  | 2.00 | -0.02 | -0.19 | -0.53 | -0.16 |

  Correlation >= 0.65 up to 0.79 GeV at |cos Z| <= 0.15; it drops to 0.2 at 0.25 (0.5 GeV) and below 0 beyond, and at 1.26 GeV
  holds only at 0.05. The PM thresholds are consistent with this and were kept; none was taken from Honda.
* **Error**: sigma_R = stat (+) 32% closure (+) |a' - mirror| unchanged (full envelope everywhere), so the error map is the
  same as v1b (medians: up |cos Z| < 0.2 10% / 9% / 14% at 0.1-0.3 / 0.3-1 / 1-3 GeV; up -0.2..-0.5 12 / 10 / 6%; up < -0.5
  8 / 5.5 / 5%; down 0-0.2 4 / 4 / 6%; down > 0.5 1 / 1.4 / 2%; 3-10 GeV engine bins 5%; max 0.73; 1276 of 9600 above 10%).
* **Checks** (`v1c_validation.md`): a1/a0 at -0.05, 0.5 GeV (v1c / Honda solmax): nue 0.707/0.685, antinumu 0.614/0.606, numu
  0.398/0.447, antinue 0.308/0.360; at -0.15: nue 0.731/0.683, antinumu 0.671/0.617, numu 0.610/0.528, antinue 0.567/0.482.
  The Honda ordering nue > antinumu > numu > antinue is reproduced at both rows; at -0.15 the values are 5-18% above Honda
  (antinue +18%, numu +16%).
  Table up/down at |cos Z| 0.05, 0.5 GeV: nue 1.096, antinue 1.033, numu 1.002, antinumu 1.042 (Honda 1.003, 1.016, 0.999,
  1.003); v1b was 1.138 / 0.989 / 0.990 / 1.078. The nue +9% remains (it comes from the measured mean and the engine, not a').
  B2: F(2.4)/F(2.6) 0.976-1.005, F(3.8)/F(4.2) 0.975-0.996. Tests 55 passed, 1 skipped.
* **MC / engine z-values** (z = (ratio - 1)/sigma; sigma = G_err_total over the 24 cells in quadrature / 24 plus the engine
  relative error; correlated MC systematics make z optimistic), 4-6.3 GeV, horizon band cos Z 0-0.2: nue -1.5, antinue -1.7,
  numu -0.8, antinumu +0.1; other cos Z bands in this E bin |z| <= 1.0. 2.5-4 GeV |z| <= 1.2; 6.3-10 GeV |z| up to 2.3 (noisy).
  So the 0.89-0.96 horizon deficit is within about 1.7 sigma.
* Not validated: as in v1b (R_up beyond the horizon rows, R above 2.5 GeV, engine E >= 3.98 GeV bins). CPU well under 0.2 core-h.

### 12.z v1c model systematic sigma_model (2026-11-06, PM addition)

Reason: the MC agent's vertical antinue MC G / engine F is 1.032 +- 0.004 at 1-2.5 GeV and 0.943 +- 0.009 at 2.5-4 GeV (cause in the
MC's s, unexplained; MILESTONE4_RESULTS 11(c)). Rule (metadata `sigma_model`, array `sigma_model_rel`): in MC-filled down-going bins
(E < 3.98 GeV) at cos Z >= 0.6 (R - 1 ~ 0.5%, MC and engine should agree), r = <G>_az / <F>_az per species x E bin x cos Z 0.2 band;
if |r - 1| > 2 sigma_r (sigma_r = MC G_err_total of the 24 cells in quadrature / 24 / <F>, no engine error), sigma_mod = |r - 1| is added
in quadrature (relative) on all azimuths of the band. Not applied at cos Z < 0.6 (muon bend). Applied to 47 species x E x band bins
(file `out/v1c_sigma_model_bins.txt`, with r and sigma_r); max sigma_mod 0.149:
* 0.1-0.16 GeV, both bands, all species: r = 0.85-0.93 (nue 0.851/0.861, antinue 0.875/0.878, numu 0.925/0.925, antinumu 0.912/0.904);
* 0.16-0.25 GeV, both bands, all species: r = 0.95-0.99;
* 0.25-0.4 GeV: nue 0.985/0.990, antinue (0.8-1) 1.009, antinumu 1.009/1.015, numu 1.007/1.015;
* 0.4-1.0 GeV: r = 1.008-1.038 (antinue 1.015-1.033, antinumu 1.009-1.021, numu 1.008-1.038, nue 1.016-1.030 in part of the bins);
* 1-1.58 GeV: antinue 1.026 (0.6-0.8), 1.018 (0.8-1), antinumu 1.012 (0.6-0.8), nue 0.991 (0.8-1);
* 1.58-2.51 GeV: antinue 1.040 (0.8-1), antinumu 0.991 / 0.980, nue 1.017 (0.8-1);
* 2.51-3.98 GeV: antinue 0.940 (0.8-1) only.
Effect on the error-map medians (before -> after): down cos Z > 0.6: 0.1-0.3 GeV 1.10 -> 6.2%, 0.3-1 GeV 1.26 -> 1.9%, 1-3 GeV 2.05 ->
2.35%, 3-10 GeV unchanged (5%); down 0.5-1: 1.15 -> 3.7%, 1.38 -> 1.9%, 2.17 -> 2.4%; cos Z < 0.6 and up-going unchanged. Max relative
error unchanged (0.73); bins above 10%: 1276 -> 1372. The 0.1-0.25 GeV entries show that the MC/engine mismatch of 5-15% below
0.25 GeV is general (all species), not only antinue.

#### 12.z.1 Low-E MC/engine level, diagnosis and extended sigma_model (2026-11-06, PM 12:10)

Why it matters: the Honda checks normalise at down-going cos Z > 0.5, so a common level offset was invisible; the table has MC
down-going and engine up-going, so a MC/engine level disagreement at low E is an up/down level inconsistency (B1 up/down at |cos Z|
0.05: E 0.1-0.16 GeV nue 1.093, antinue 1.066, numu 1.055, antinumu 1.054; 0.16-0.25 GeV 1.107, 1.118, 0.991, 1.101).

**Diagnosis from cached cells only** (script `$W/runs/integ/lowE_diag.py`). r = <G_MC>_az / <F_engine>_az, all azimuths, sigma_r = MC
G_err_total only (no engine error). (a) GSF engine (`engine_F_fine.npz`, as in the table):

| E [GeV] | species | 0-0.2 | 0.2-0.4 | 0.4-0.6 | 0.6-0.8 | 0.8-1 |
|---|---|---|---|---|---|---|
| 0.10-0.16 | nue | 0.933 | 0.886 | 0.868 | 0.851 | 0.861 |
| | antinue | 0.941 | 0.896 | 0.876 | 0.875 | 0.878 |
| | numu | 0.926 | 0.932 | 0.925 | 0.925 | 0.925 |
| | antinumu | 0.941 | 0.932 | 0.911 | 0.912 | 0.904 |
| 0.16-0.25 | nue | 0.930 | 0.942 | 0.951 | 0.954 | 0.962 |
| | antinue | 0.907 | 0.930 | 0.966 | 0.972 | 0.991 |
| | numu | 0.950 | 0.972 | 0.980 | 0.981 | 0.980 |
| | antinumu | 0.943 | 0.969 | 0.969 | 0.978 | 0.981 |
| 0.25-0.40 | nue | 0.934 | 0.969 | 0.979 | 0.985 | 0.990 |
| | antinue | 0.882 | 0.950 | 0.984 | 1.000 | 1.009 |
| | numu | 0.963 | 0.990 | 1.001 | 1.007 | 1.015 |
| | antinumu | 0.963 | 0.997 | 1.000 | 1.009 | 1.015 |

(sigma_r is 0.002-0.013.) The mismatch is general, 7-15% at 0.1-0.16 GeV in every species and band, falling to a few % at 0.25-0.4 GeV
(except near the horizon, where antinue is 0.88). The nue/antinue deficit is deepest at large cos Z (0.85-0.88), numu/antinumu are flat in
cos Z (0.91-0.93).

(b) GH-native engine (`rigfix/out/m3b_engine_ghnative_mb0.npz`, v142p after the rigidity fix, hybrid base, no muon bend). These cells
cover only cos Z rows 0.05, 0.15, 0.25, so only the band 0-0.2 and the row 0.25 (cos Z 0.2-0.3) can be compared; nothing at cos Z >= 0.3.
r with GSF (same file) | with GH-native (cos Z 0-0.2, then the 0.25 row):

| E [GeV] | nue | antinue | numu | antinumu |
|---|---|---|---|---|
| 0.10-0.16 (0-0.2) | 0.928 \| 0.954 | 0.937 \| 0.966 | 0.921 \| 0.949 | 0.936 \| 0.965 |
| 0.10-0.16 (0.25 row) | 0.892 \| 0.916 | 0.901 \| 0.929 | 0.927 \| 0.955 | 0.930 \| 0.958 |
| 0.16-0.25 (0-0.2) | 0.924 \| 0.947 | 0.902 \| 0.927 | 0.946 \| 0.970 | 0.939 \| 0.963 |
| 0.25-0.40 (0-0.2) | 0.929 \| 0.944 | 0.878 \| 0.895 | 0.958 \| 0.974 | 0.959 \| 0.975 |

F_GHn/F_GSF = 0.970-0.984, i.e. (b) - (a) = +0.015 to +0.03 in r: the composition/primary change closes only 1/5 to 1/3 of the gap at
0.1-0.16 GeV, and less above. It does not explain the effect (nor does it reorder species).
**Composition of the engine GSF run**: it uses GSF's own nucleon-level p and n fluxes only (crflux's GSF file holds proton fraction, proton
and neutron nucleon fluxes; no group splines): free protons = p - n (A/Z = 1), bound = n (A/Z <~ 2, <A/Z> nucleon-flux weighted). The MC
primaries are GSFG:GH, i.e. the same GSF all-nucleon flux split into groups with the GH composition (own A/Z mix per group, Fe 2.08), so the
A/Z mix near the cutoff differs between the two. The GH-native engine changes both the spectrum and the composition, so (b) - (a) is only a
first, joint test of that hypothesis.

**<R - 1>_az of the measured muon-bend response** (R table, E < 0.4 GeV): median 0.003; max over species and E: 0.012 at cos Z 0.2-1; at cos Z
0-0.2 up to 0.095 (antinue, 0.25-0.4 GeV) and 0.048 at 0.4-2.5 GeV. So the az-mean comparison is clean for cos Z >= 0.2 and carries up to 5-10%
bend contribution in the horizon band; |r - 1| there is therefore partly physics, which the (conservative) sigma_mod covers.

**Extended sigma_model** (metadata `sigma_model`, array `sigma_model_rel`, bins in `out/v1c_sigma_model_bins.txt`; sigma_r is the MC error
only):
* down-going, MC-filled (E < 3.98 GeV), all cos Z 0.2 bands: sigma_mod = |r - 1| where |r - 1| > 2 sigma_r, in quadrature, all azimuths of
  the band: 98 of 160 species x E x band bins;
* up-going (engine-filled): sigma_level(species, E) = |r_vert - 1| with r_vert over cos Z 0.6-1 (48 cells), on all up-going bins at that E
  where |r_vert - 1| > 2 sigma: 26 of 32 species x E bins (E < 3.98 GeV; r_vert = 0.856 / 0.877 / 0.925 / 0.908 for nue / antinue / numu /
  antinumu at 0.1-0.16 GeV, 0.96-0.98 at 0.16-0.25, 0.99-1.01 at 0.25-0.4, 1.01-1.03 at 0.4-1 GeV, antinue 1.022 and 1.018 at 1-2.5 GeV, 0.979 at
  2.5-4 GeV). Not applied to E >= 3.98 GeV (engine-filled both hemispheres, no MC level there).

Error-map medians (no sigma_model -> first sigma_model (cos Z >= 0.6) -> extended), in %; bins above 10%: 1276 -> 1372 -> 1743; max 0.73:

| region | 0.1-0.16 | 0.16-0.25 | 0.25-0.4 | 0.4-1 | 1-3 | 3-10 |
|---|---|---|---|---|---|---|
| up \|cos Z\| < 0.2 | 9.8 -> 9.8 -> 14.4 | 10.2 -> 10.2 -> 10.4 | 8.8 -> 8.8 -> 8.8 | 9.8 -> 9.8 -> 10.0 | 13.6 | 5 |
| up -0.2..-0.6 | 12.3 -> 12.3 -> 15.7 | 12.4 -> 12.4 -> 12.6 | 9.7 | 10.0 -> 10.0 -> 10.1 | 5.8 -> 5.9 | 5 |
| up < -0.6 | 6.8 -> 6.8 -> 13.7 | 6.9 -> 6.9 -> 7.7 | 5.9 -> 5.9 -> 6.1 | 5.1 -> 5.1 -> 5.7 | 5.0 -> 5.2 | 5 |
| down 0-0.2 | 3.6 -> 3.6 -> 7.6 | 3.7 -> 3.7 -> 7.6 | 3.5 -> 3.5 -> 7.2 | 4.0 -> 4.0 -> 4.6 | 6.1 -> 6.3 | 5 |
| down 0.2-0.6 | 1.6 -> 1.6 -> 9.8 | 1.7 -> 1.7 -> 3.6 | 2.3 -> 2.3 -> 2.6 | 2.5 -> 2.5 -> 2.6 | 2.9 -> 3.1 | 5 |
| down > 0.6 | 1.1 -> 11.0 -> 11.0 | 1.1 -> 2.3 -> 2.3 | 1.2 -> 1.7 -> 1.7 | 1.3 -> 2.3 -> 2.3 | 2.0 -> 2.3 | 5 |

**Hypotheses not yet tested** (the cause is open): (1) composition (A/Z mix, GSF-own vs GH groups; the GH-native cells test it only jointly with
the spectrum and only at cos Z < 0.3); (2) phi_0 / solar-modulation treatment: both sides nominally use GSF as shipped (phi_0 = 0.55 GV is the epoch label in the MC
metadata); to check that the production applied no `w_solar` reweighting (phi_1 = phi_0) and that the engine applies no modulation -- a
modulation difference would act mostly below ~1 GeV, i.e. with this E dependence *(PM wording, 2026-11-06)*; (3) low-E hadronic yields (kappa_s
absorbs the MC/MCEq yield ratio over cos Z 0.6-1 in the M4 production (0.35-1 was the provisional pilot-2 window of sec. 10), so a cos Z-dependent yield difference at 0.1-0.3 GeV survives in G); (4) the engine
cone/cutoff treatment at 0.1 GeV (grid edge, min of the E range, the up-going mirror cone) and the MC's own lowest bins. Nothing here was tuned.
CPU: negligible (cached cells only).

### 12.h Table checksums (PM, 2026-11-06)

The v1c table and its inputs as committed in 1716a8d (sha256):
`kamioka_3D_v1c.npz` ffa41e0c4edb26f9b1a00bf0e1a250a68a227c86f03310dbabc085eab280d511;
`kamioka_G_down_v1b.npz` 43ec21fb88031491699a57e9ed3863cb67a27dcfb17208c063d67064aea238aa;
`kamioka_Rmu_v1.npz` 27ffb4b20386331e297709f1e1e4b414b64a39088923ceca66e66010ef40f9a1.
**Erratum:** the commit message of 1716a8d quotes the prefixes 74b4d8a9 / 47e3dc47; they are wrong (typed before the hashes were computed). The values above are the correct ones.

## 13. Hybrid table v2h and its 3D twin v2 (2026-11-09, CANDIDATE)

Files: `$W/runs/v2h/out/kamioka_3D_v2h.npz` (label `kamioka_3D_v2h`, status `candidate`) and
`$W/runs/v2h/out/kamioka_3D_v2.npz` (label `kamioka_3D_v2`, the same 3D factor without the hybrid level). Both use the
`Tabulated3DModel` format of section 2, unchanged: `factor`, `factor_error` (absolute 1 sigma) on the v1c grid (E 5/dec 0.1-10 GeV
x cos Z 0.1 x az 12). Builder: `tools/mceq3d/build_table_v2.py`.
`regress` rebuilds the three v1c source blocks from v1c's inputs, bit for bit; `build` writes the tables and refuses to
overwrite; `gates` runs gate G1. The run directory is `$W/runs/v2h` (`REPORT.md`, `scripts/`, `logs/`, `out/`).
sha256: v2h `e3db406733bbe64fd9534c3366214b423f622f6a702b95518e4731a342a711ce`,
v2 `576ea3ead035e2e06bfed208ae503c90a2741dd390a603059d207005dd563c64`,
new engine grid `out/t4_D_up.npz` `14a7cbed9728accf32955490342f7d7ae7bba080fbc1686d5918bb6c0f8b6098`.

### 13.1 What the hybrid is

The PI decided that the production 1D level is the hybrid: daemonflux above about 2 GeV, the engine 1D (B_ref) below,
blended smoothly, times the 3D factor. For each species and |cos Z| (no azimuth dependence):

    L_s(E, |cos Z|) = (1 - w(E)) B_ref,s + w(E) Phi_df,s,     w(E) = 0.5 (1 + tanh(ln(E/1.7 GeV) / ln 2))

* `w` is `mceq3d_flux.hybrid_weight(E, e0=1.7)`. Then w < 0.02 below 0.44 GeV, w = 0.18 at 1 GeV, 0.5 at 1.7 GeV, 0.75 at
  2.5 GeV, and w > 0.98 above 6.6 GeV.
* `B_ref` is the engine base in the REF population (MCEq floor 3.0 GeV, e_min 0.1, GSF, SIBYLL23D, v142p, CORSIKA US-Std). It is
  the denominator of the 3D factor, so below the switch daemonflux x factor = F x B_ref = Phi_3D^PHYS.
* `Phi_df` is `daemonflux.Flux(location="kamioka")`, calibrated, with nominal parameters.

The table carries the level as `hybrid_correction` H (S, nE, nC) = int L / int Phi_df per bin, with factor_v2h = factor_v2 x H.
The energy integral uses the engine's 10/decade grid points inside the bin with weights `e_widths` (the convention of
`diag_v1.fine_avg`). The cos Z integral uses the 3-point Gauss rule of `base1d_kamioka.npz`. Up-going bins take H at |cos Z|,
like the 1D flux. `factor_3d` stores factor_v2, so a user can undo the hybrid: factor / hybrid_correction.

H is 0.61-0.66 at 0.10-0.16 GeV (that is 1/D, with D = Phi_df/B_ref = 1.53-1.64). It rises to 0.75-0.84 at 0.40-0.63 GeV,
0.85-0.95 at 0.63-1 GeV and 0.96-1.03 at 1-1.6 GeV, and is 0.993-1.003 from 1.6 GeV upward. The cos Z dependence is at most 1.5%.

### 13.2 The 3D factor v2 (v1c pipeline with three changes)

* (a) MC down-going cells (source 0, E < 3.98 GeV) are multiplied by C'' = (kappa/kappa_ref)/(1+b) x gf
  (fix1 REPORT, PM block).
  * 0.1-1 GeV: fix1's per-cell values (`t4_closure.npy`). This includes the "0-0.2" key: kappa window 0.6-1, b of the
    0-0.2 cap band.
  * 1-3.98 GeV, from the definition: b from `s1d_cap.npz` (measured in every band), gf = sum B(3.0, e_min 0.03) / sum B_ref
    over the kappa window (computed), and kappa/kappa_ref = 1 (assumed).
  * The assumption kappa/kappa_ref = 1 rests on the T < 2 GeV/n content of t, dB/B, which is only 0.05-0.24% at 0.63-1 GeV
    and falls with E.
  * Values: C'' = 1.09-1.17 at 0.10-0.16 GeV, 0.99-1.02 at 0.16-0.25, 0.97-1.00 at 0.25-1, 0.95-1.01 at 1-2.5 and 0.99-1.04
    at 2.5-3.98 GeV (b = -0.012 to -0.034 there).
* Near-horizon MC cells (fix2's production-B finding): **kept**.
  * The production-B artefact is in t, the straight-line reference, near the horizon. G = s/(kappa B) uses t only in the
    kappa window cos Z 0.6-1, where t_B/t_A = 0.99-1.005 (fix2 PM check). The 3D tally s is a physical flux (fix2: "MC
    tallies change by 0").
  * Check (`scripts/c1_horizon.py`), q = <G''>/(<F_v4><R>) with R = the measured muon-bend response:
    * cos Z 0.2-0.3: q is within -7%/+3% of the cos Z 0.35-0.95 median (mostly within 2-4%).
    * cos Z 0-0.2: q is 5-20% low at E <= 0.4 GeV and at 1-1.6 GeV (antinue at 0.25-0.4 GeV and cos Z 0.05: 0.78).
  * The near-horizon difference is not the B artefact, and its cause is open. It is carried as `err_horizon_rel` =
    sqrt(max((q-1)^2 - sigma_model^2, 0)) on cos Z 0-0.2. 552 of 768 cells are non-zero, maximum 0.18.
* (b) Engine cells come from F = Phi_3D^PHYS/B_ref with `eoff_model="v4"` (fix2 variant D, solve kwargs as fix2). This covers
  source 1 (down-going, E >= 3.98 GeV) and source 3 (up-going, mirror-cone ratio x R_up of v1c).
  * Inputs: the hor/dn6 grids come from `fix2/out/t4_D_*`. The up-going 10 x 12 grid is new: `v2h/out/t4_D_up.npz`, 421 s CPU.
  * The fix1 v3 cells give `err_engine_spread_rel` = |F_v4 - F_v3| / F_v4: median 0.9% and max 9.6% up-going, at most 4.2%
    down-going.
* (c) R_up, sigma_R, sigma_model and the grid are as in v1c. The v1c relative error (factor_error/factor) is kept as
  `err_v1c_rel`.

`source` codes are unchanged in layout: 0 = MC x C'', 1 = engine v4 down-going, 3 = engine v4 up-going x mirror cone x R_up.

### 13.3 Errors (`factor_error` absolute; terms stored as relative arrays)

    v2 : rel^2 = err_v1c_rel^2 + err_Cpp_rel^2 + err_engine_spread_rel^2 + err_horizon_rel^2
    v2h: rel^2 = (v2 terms) + err_hybrid_rel^2,   err_hybrid_rel = [(1 - w) B_ref / L] sigma_B,  sigma_B^2 = sigma_had^2 + sigma_prim^2

* `err_Cpp_rel` combines two pieces.
  * The kappa_ref estimator: |kappa_ref/kappa_h1 - 1| below 1 GeV, or dB/B of 0.63-1 GeV above.
  * b: its fit error (+) 30% of b. The 30% covers the theta^2 form; lowE s1e's engine estimate is about 1.5x the fitted b.
  * Result: median 1.0%, max 2.2%.
* `sigma_had` = |DDM/B_ref - 1|. The base is recomputed with `MCEqRun.inject_ddm(DataDrivenModel())` (MCEq 1.4 DDM), same
  atmosphere and REF population, at the 10 row centres.
  * Values: 0.20-0.28 at 0.10-0.25 GeV, 0.13-0.23 at 0.4-0.63 GeV, 0.03-0.15 at 0.63-1 GeV, 0.01-0.05 at 1-2.5 GeV.
  * It dips where DDM/B_ref crosses 1 (1-2 GeV). That is a weakness of a single-model spread.
* `sigma_prim`: the prescribed recipe gives about 1e-4. That recipe is the prior error of `Flux(location="kamioka",
  use_calibration=False)` with the hadronic-only part removed in quadrature. It is near zero because the prior covariance in
  the spline file has a zero GSF block (sigma(GSF_1) = 0, sigma(GSF_2..6) = 0.012).
  * Used instead: the GSF block of the calibrated covariance (diagonal 0.93-1.00, i.e. essentially unconstrained). It gives
    0.4-1.0%, so sigma_B is in practice sigma_had.
* Solar modulation is not in `factor_error`. The optional `dlnproduct_dphi` (S, nE, nC), in 1/GV, is the azimuth-mean
  response of daemonflux x factor to the force-field potential:

      d ln P/d phi = d ln Phi_3D/d phi - (w Phi_df/L) d ln B_ref/d phi

  * This is the brief's "dlnF/dphi + dlnB_ref/dphi below the switch, dlnF/dphi above", with F = Phi_3D/B_ref at the same phi.
    fix1's t5 files define F with an unmodulated B_ref, so their slope is d ln Phi_3D/d phi itself.
  * Epoch convention: phi relative to the GSF epoch (phi0 = 0.55 GV label, not verified). The force field acts on the
    un-floored primary, with the MCEq floor applied after. daemonflux is held fixed. The slope is a central difference at
    +-0.3 GV.
  * Inputs: t5 (E_off v3, reduced config, rows +-0.15/0.35/0.95, interpolated in |cos Z|) and REF MCEq bases at +-0.3 GV.
  * Species-mean values below the switch: -0.22/GV (down 0.95) to -0.39/GV (up -0.95) at 0.10-0.16 GeV, and -0.14/-0.21 at
    0.63-1 GeV. Above 3 GeV it is about 0, because the numerator and B_ref respond alike.

Error budget, v2h (median / max relative error; variance share of each term):

| band | hemisphere | median | max | v1c | C'' | engine spread | horizon | hybrid | v2 median |
|---|---|---|---|---|---|---|---|---|---|
| 0.1-0.25 | down | 0.243 | 0.305 | 11% | 1% | 0% | 1% | 87% | 0.075 |
| 0.1-0.25 | up | 0.262 | 0.646 | 31% | 0% | 3% | 0% | 66% | 0.137 |
| 0.25-1 | down | 0.158 | 0.500 | 7% | 0% | 0% | 2% | 91% | 0.027 |
| 0.25-1 | up | 0.189 | 0.636 | 42% | 0% | 1% | 0% | 57% | 0.082 |
| 1-2.5 | down | 0.036 | 0.343 | 69% | 2% | 0% | 19% | 10% | 0.031 |
| 1-2.5 | up | 0.061 | 0.729 | 97% | 0% | 1% | 0% | 2% | 0.056 |
| 2.5-10 | down | 0.050 | 0.251 | 93% | 1% | 1% | 5% | 0% | 0.050 |
| 2.5-10 | up | 0.050 | 0.138 | 98% | 0% | 2% | 0% | 0% | 0.050 |

### 13.4 Validation (`v2h/REPORT.md`; `out/gates_G1.md`, `out/validation.md`, `out/ext_condensed.md`, `out/analysis.md`)

* **G1, identities through the API**, checked at 20 cos Z (row centres plus 0.02/0.10/.../0.98, both hemispheres) x 24
  azimuths at engine grid energies:
  * w > 0.98 (6.6-60 GeV): product_v2h/(Phi_df factor_v2) - 1 <= 0.20%. **PASS.**
  * w < 0.02 (0.112-0.36 GeV): product_v2h/(factor_v2 B_ref) - 1 reaches 2.0%. **FAIL at 1%.**
    * At the bin centres it is at most 0.18%.
    * Below the first bin centre (0.112-0.126 GeV, where the table is flat in E) it reaches 1.6% (numu/antinumu).
    * Between the 0.32 and 0.50 GeV centres it reaches 2.0% (nue near the horizon). The 0.40-0.63 GeV bin's H carries
      w = 0.02-0.05, and linear interpolation of F x H is not F x H interpolated.
  * Pointwise H does not fix either. Both are properties of the 5/dec grid with linear interpolation, applied to a ratio
    that changes 4-10% per bin.
* **G2, API**: PASS.
  * Both tables load. `flux`, `error` and `geomagnetic_factor_error` are finite and positive on the full grid, including
    up-going and zeniths 89.9/90/90.1/180: 1944 (zenith x azimuth x quantity) calls per table.
  * daemonflux suite: 54 passed, 3 skipped. `test_tabulated3d.py` with the table env vars: 18 passed, 1 skipped, including
    the new `test_kamioka_v2h_table_if_available`.
  * mceq3d set (flux, deterministic, production, offaxis_mc, joint_cone + test_tabulated3d): 75 passed, 5 skipped, 0 failed.
    The PM's last count was 67/12. Mine has one extra test (the new skip-if-absent one), and the split difference is not
    explained.
* **G3, v2/v1c**: every change above 5% has an identified source.
  * MC cells: only 0.10-0.16 GeV, where v2/v1c is exactly C'' = 1.089-1.168.
  * Engine down-going: at most 1.8%.
  * Up-going: 284 cells, all at E <= 0.25 GeV:
    * 0.10-0.16 GeV, |cos Z| 0.6-1: -5 to -9%, from E_off v4 (v4/v3 = 0.95-0.96).
    * 0.10-0.16 GeV, |cos Z| 0-0.3: up to +20% (nue/antinue), from the PHYS population (v3/delivered = 1.05-1.14: primaries of
      T 0.9-2.2 GeV/n at low far-side cutoffs, plus the e_min 0.03 grid).
    * 0.16-0.25 GeV: -7.5% to +5%, from v4/v3.
* **External (ext1 method, 4-reference envelope Honda solmin/solmax + Bartol fmin/fmax, flag at 3% in ln):**
  * Level: below 1 GeV the species-mean level of Pv2h is within +5.5% of the envelope edges (written expectation: +-10%,
    met). The largest single species is +8.3% (nue, 0.11-0.16 GeV, down-going cos Z 0.6-1). P1/P2/Pv2 sit at +8 to +63%.
  * Level at 2.5-10 GeV: -1.4 to -3.1% for all products (daemonflux's own level).
  * Flag counts P1 / Pv2 / Pv2h: level 130/128/27, nu/nubar 49/50/20, flavour 16/19/6, U/D 0/0/0, H/V 9/11/12, a1/a0
    12/16/16, phase 2/2/2.
  * The hybrid does not touch the EW shape. Down-going a1/a0 is 0.03-0.04 below Honda at 0.11-0.6 GeV. Up-going 0-0.3 is
    +0.06 to +0.11 above Honda at 0.11-0.25 GeV.
  * H/V (down 0-0.3 over 0.6-1) is 3-7% below the envelope at 0.4-2.5 GeV. It is slightly lower than P2 because of the 0-0.2
    cap bias in C''.

### 13.5 Not established / open

* The G1 low-side tolerance needs a decision. Options:
  * accept a 2% interpolation tolerance (well below err_hybrid, 13-28% there);
  * test at bin centres;
  * use a finer E grid or a lower first edge, which changes the v1c grid;
  * carry the hybrid as a smooth 1D level option in daemonflux rather than in the table. That needs a `src/daemonflux` change
    and is not implemented.
* The near-horizon (cos Z 0-0.2) MC vs v4 difference of 5-20%.
* Up-going v4 rows have no MC validation.
* sigma_had is a single-model spread that vanishes at the DDM/B_ref crossing.
* The v1c sigma_model is kept although C'' now corrects part of the mismatch it covers, so the budget is conservative at
  0.1-0.25 GeV.
* C'' above 1 GeV assumes kappa/kappa_ref = 1.
* `dlnproduct_dphi` uses E_off v3 and 3 rows per hemisphere.

### 13.6 PM review (2026-11-09)

* **G1 decision: the 2% low-side tolerance is accepted for the candidate.** The deviation is interpolation between bin
  centres (0.18% at the centres) of a ratio that changes 4-10% per bin, and it is an order of magnitude below
  `err_hybrid_rel` there (13-28%). If v2h is promoted, revisit with a finer E grid below 1 GeV or with the hybrid as a
  smooth 1D-level option inside daemonflux (a `src/daemonflux` change, not made).
* **Test counts, PM rerun** with `DAEMONFLUX_3D_TABLE_V1` (v1c) and `DAEMONFLUX_3D_TABLE_V2H` set: daemonflux suite
  56 passed, 1 skipped; the mceq3d set above 77 passed, 3 skipped when run from `tools/mceq3d`, 69 passed, 11 skipped when
  run from the repo root. The split difference is the working directory: several mceq3d tests open data files by relative
  name and skip elsewhere. No failures either way.
* **Checked by the PM:** v1c sha256 unchanged; factor = factor_3d x hybrid_correction exactly; factor_3d = v2 factor
  exactly; H at cos Z 0.95 as quoted in 13.1; median relative factor_error per E bin 0.26/0.24/0.21/0.17/0.08/0.06/0.05/
  0.05/0.05/0.05.
* **`dlnproduct_dphi`:** the PM brief said "dlnF/dphi + dlnB_ref/dphi below the switch"; that double-counts, because fix1's
  t5 slopes already hold B_ref fixed. The formula in 13.3 is the correct one (d ln P = d ln Phi_3D - (w Phi_df/L) d ln B_ref).
* **Status:** candidate. v1c remains the last committed table; nothing in `src/daemonflux` changed.

## 14. Near-horizon candidate v2i (2026-11-10, CANDIDATE)

File: `$W/runs/v2i/out/kamioka_3D_v2i.npz` (label `kamioka_3D_v2i`, status `candidate`), `Tabulated3DModel` format unchanged.
It is v2h (section 13) plus three near-horizon changes found by the hzn study (`$W/runs/hzn/REPORT.md`). Builder:
`tools/mceq3d/build_table_v2.py build-v2i` / `gates-v2i`; the correction numbers come from
`$W/runs/v2i/scripts/c0_corrections.py` (corrections npz, sha256 recorded in the table). v2h is untouched and still the reference
candidate. sha256: v2i `e4029e6c...27e3be5`, corrections `a5fb8fe3...42e200a` (full values in 14.5).

### 14.1 The hzn finding (and its caveats)

* 0.10-0.40 GeV: the MC is right and the engine (E_off v4) is too high at the horizon. The multigeneration part of the v4 kernel
  inside the factorised cone integral over-broadens at grazing angles. Engine-only horizon departure relative to the zenith-profile
  variant Z at 0.10-0.16 GeV, rows 0.05 / 0.15: multigeneration +13.3% / +10.2%, net v4 +8.7% / +5.6% [m]. The MC's field-free
  shape sides with Z (-0.008 +- 0.028 / +0.025 +- 0.014) and rejects v4 (-0.096 +- 0.028 / -0.031 +- 0.014) [m].
  Honda H/V and E-W a1/a0 support this independently, but Honda and Bartol differ by 24/15/6% in H/V at 0.1-0.4 GeV, so that check is coarse.
* Strength: 3.4 sigma at 0.10-0.16 GeV, row 0.05; about 2 sigma at 0.25-0.40; not discriminating at 0.16-0.25. The finding is
  POST HOC relative to the pre-registered criterion (by the letter only Bd passes). The mechanism is NOT established.
* 0.4-1.0 GeV: MC and engine agree within about 3%, not arbitrated. 1.0-1.6 GeV: most of the -7% is a definition effect on the
  MC-table side (C'' divided the horizon rows by 1 + b with b from the "0-0.2" cap band, measured with the 1/cos flux weight; a
  heavy-tail artefact), the rest about -6% at row 0.05 is a bend residual at 1-2 sigma, not closed.
* Up-going near-horizon cells are engine-only and carry the same v4 excess. The correction is a template exp(-D[E_off_v4/E_off_Z]);
  Z is not a validated engine (its cos Z >= 0.2 pass is a cancellation). It is an engineering patch backed by the MC's field-free
  measurement at 0.10-0.16 GeV. The mirror-cone (cutoff part) and far-side in-atmosphere bend of the up-going cells are untested.
* Beyond Kamioka: every site's up-going hemisphere is engine-only, so the v4 horizon defect should be fixed in the engine (or masked
  at |cos Z| < 0.3) before the engine route is used elsewhere.

### 14.2 The v2i definition

factor_3d_v2i = factor_3d_v2h x corr_v2i; factor_v2i = factor_3d_v2i x hybrid_correction (unchanged).
`corr_v2i` (4, 10, 20) = `corr_fb` x `corr_cup`, azimuth-independent, exactly 1.0 (bit-identical cells) outside the 80 touched cells.
Rows: cos Z +0.05 / +0.15 are table indices 10 / 11 (source 0), -0.05 / -0.15 are 9 / 8 (source 3); verified from `cosz_edges` and `source`.

**(a) Cap-bias key, MC horizon rows** (source 0, cos Z 0.05 / 0.15, E bins 0-6), all species and azimuths:
f_b = (1 + b1)/(1 + b0), b = production-B cap bias of the "0-0.2" band, b1 = measure 1 (1/cos flux weights, what C'' used), b0 = measure 0
(crossing count). Re-run of `hzn/scripts/a15_b_measure0.py`. b1 equals `Cpp_parts_b` of v2h at these rows and `lowE/s1d_cap.npz`
(max difference 0.0, species-independent). E bin 7 unchanged.

| E bin / GeV | 0 / 0.10-0.16 | 1 / 0.16-0.25 | 2 / 0.25-0.40 | 3 / 0.40-0.63 | 4 / 0.63-1.0 | 5 / 1.0-1.6 | 6 / 1.6-2.5 |
|---|---|---|---|---|---|---|---|
| b1 | +0.0677 | +0.0622 | +0.0364 | +0.0413 | +0.0297 | +0.0506 | +0.0145 |
| b0 | +0.0726 | +0.0559 | +0.0342 | +0.0266 | +0.0133 | +0.0264 | +0.0178 |
| sigma_b0 | 0.0021 | 0.0024 | 0.0029 | 0.0034 | 0.0043 | 0.0054 | 0.0070 |
| f_b | 0.9955 | 1.0060 | 1.0021 | 1.0143 | 1.0162 | 1.0235 | 0.9967 |
| err_capb_rel | 0.0068 | 0.0077 | 0.0087 | 0.0126 | 0.0152 | 0.0200 | 0.0211 |

**(c) Up-going near-horizon correction** (source 3, cos Z -0.05 / -0.15), E bins 0-2; c_up from `hzn/out/b4_upgoing_template.npz`
(template exp(-D[E_off_v4/E_off_Z]), species-resolved; mc = MC field-free measurement, species-mean used):

| cos Z | species | 0.10-0.16 | 0.16-0.25 | 0.25-0.40 |
|---|---|---|---|---|
| -0.05 | nue | 0.9073 | 0.9133 | 0.9283 |
| -0.05 | antinue | 0.9084 | 0.9043 | 0.9233 |
| -0.05 | numu | 0.9214 | 0.9391 | 0.9773 |
| -0.05 | antinumu | 0.9282 | 0.9468 | 0.9820 |
| -0.15 | all (species-mean MC) | 0.9701 | 0.9676 | 0.9791 |

E bins 3-4 (0.4-1 GeV): no correction (evidence not established), only an error. E bins 5-9: nothing.

### 14.3 Errors

`factor_error` stays absolute. (factor_error/factor)^2 = err_v1c^2 + err_Cpp^2 + err_engine_spread^2 + err_horizon^2 + err_hybrid^2
+ err_capb^2 + err_upcorr^2, all stored as relative arrays (4, 10, 20, 12); v2h's `err_horizon_rel` is kept as `err_horizon_rel_v2h`.
Cells outside the touched set keep v2h's `factor_error` bit for bit.

* `err_capb_rel` (MC horizon cells, E bins 0-6): sqrt((|f_b - 1|/2)^2 + (3 sigma_b0)^2). The 3 is because hzn found jackknife errors
  of b to be 2-4x the per-primary ones. Table above.
* `err_horizon_rel` re-assigned on the MC rows 0.05 / 0.15, azimuth- and species-independent. D_f = ln f - <ln f> over rows 0.35-0.95
  (hzn `a2_decomp.py`), used as |D| relative error. E bins 0-2: |species-mean D_q| of the engine-Z decomposition
  (`a2_decomp_Z.npz`); bins 3-6: |species-mean (D_Bd + D_Cut)| of the v4 decomposition (`a2_decomp.npz`). E bin 7: v2h's value.
  Exception: antinue 0.25-0.40 GeV, row 0.05 keeps its v2h value (unexplained cell).

| E bin / GeV | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| row 0.05 | 0.0069 | 0.0157 | 0.0281 | 0.0433 | 0.0904 | 0.0339 | 0.0266 |
| row 0.15 | 0.0198 | 0.0336 | 0.0041 | 0.0266 | 0.0331 | 0.0140 | 0.0565 |

* `err_upcorr_rel` (up-going rows -0.05 / -0.15): E bins 0-2 sqrt(se_mean^2 + (mean_k template - mean_k mc)^2), se_mean = sqrt(sum_k
  mc_err^2)/4; E bins 3-4 |1 - mean_k template|; species-independent.

| E bin / GeV | 0 / 0.10-0.16 | 1 / 0.16-0.25 | 2 / 0.25-0.40 | 3 / 0.40-0.63 | 4 / 0.63-1.0 |
|---|---|---|---|---|---|
| row -0.05 | 0.0263 | 0.0804 | 0.0643 | 0.0171 | 0.0107 |
| row -0.15 | 0.0294 | 0.0192 | 0.0159 | 0.0040 | 0.0013 |

  The se_mean here uses the quadrature sum / 4 as specified (0.0257 / 0.0603 / 0.0458 at row -0.05; 0.0159 / 0.0156 / 0.0159 at
  row -0.15). hzn's log quotes the jackknife error of the species mean, 0.028 / 0.065 / 0.049 and 0.014 / 0.016 / 0.019, slightly different
  because the species are correlated in the jackknife.
* `dlnproduct_dphi` is unchanged from v2h (the corrections are not propagated into the solar response); its note says so.

Relative factor_error (median / max over species x azimuth), v2h -> v2i, E bins 0-7:

| row cos Z | E0 0.10-0.16 | E1 0.16-0.25 | E2 0.25-0.40 | E3 0.40-0.63 | E4 0.63-1.00 | E5 1.00-1.58 | E6 1.58-2.51 | E7 2.51-3.98 |
|---|---|---|---|---|---|---|---|---|
| +0.05 | 0.233/0.288 -> 0.225/0.260 | 0.225/0.272 -> 0.218/0.256 | 0.229/0.325 -> 0.231/0.325 | 0.173/0.500 -> 0.174/0.502 | 0.095/0.260 -> 0.132/0.276 | 0.114/0.338 -> 0.094/0.337 | 0.104/0.343 -> 0.105/0.344 | 0.151/0.251 -> 0.151/0.251 |
| +0.15 | 0.227/0.251 -> 0.228/0.252 | 0.218/0.249 -> 0.220/0.251 | 0.215/0.235 -> 0.214/0.232 | 0.156/0.210 -> 0.159/0.212 | 0.066/0.123 -> 0.075/0.128 | 0.096/0.129 -> 0.061/0.075 | 0.062/0.085 -> 0.082/0.102 | 0.095/0.154 -> 0.095/0.154 |
| -0.05 | 0.263/0.299 -> 0.265/0.300 | 0.239/0.272 -> 0.252/0.284 | 0.214/0.316 -> 0.224/0.323 | 0.209/0.304 -> 0.210/0.305 | 0.148/0.379 -> 0.148/0.379 | 0.166/0.535 -> 0.166/0.535 | 0.154/0.496 -> 0.154/0.496 | 0.062/0.138 -> 0.062/0.138 |
| -0.15 | 0.256/0.305 -> 0.258/0.306 | 0.231/0.295 -> 0.232/0.296 | 0.209/0.374 -> 0.210/0.374 | 0.188/0.456 -> 0.188/0.456 | 0.127/0.579 -> 0.127/0.579 | 0.104/0.729 -> 0.104/0.729 | 0.121/0.535 -> 0.121/0.535 | 0.055/0.125 -> 0.055/0.125 |

### 14.4 Gates and external results

* V1 identities (`out/gates_V1V2.md`, `build_table_v2.py gates-v2i`) PASS: factor_3d_v2i / factor_3d_v2h / corr_v2i - 1 = 2.2e-16;
  factor = factor_3d x hybrid_correction exactly; factor_3d bit-identical where corr = 1; hybrid_correction bit-identical; 80 of 800
  (species, E, cos Z) cells have corr != 1, all inside the expected set.
* V2 errors PASS: max |(factor_error/factor)^2 / sum(terms^2) - 1| = 6.7e-16; factor_error changes in 1152 cells, all touched
  (672 MC horizon + 480 up-going).
* V3 tests: daemonflux `pytest tests` with `DAEMONFLUX_3D_TABLE_V1` (v1c), `_V2H`, `_V2I`: 57 passed, 1 skipped (was 56 / 1; the new
  `test_kamioka_v2i_table_if_available`). mceq3d from `tools/mceq3d`: 78 passed, 3 skipped (was 77 / 3; set: test_mceq3d_flux, test_mceq3d_deterministic, test_mceq3d_production, test_offaxis_mc, test_joint_cone + ../../tests/test_tabulated3d.py, run from `tools/mceq3d` with `MC3D_POOL_CACHE`; the 3 skips need a cached real production profile or `DAEMONFLUX_3D_TABLE`).
* V4 external (Honda solmin/solmax, Bartol fmin/fmax; scripts copied from v2h's x1-x4 into `$W/runs/v2i/scripts`, plus `x5_near_horizon.py`).
  Flag count (outside the envelope by > 0.03 in ln, a1/a0 by > 0.05, phase > 15 deg), all tags:

| tag | P1 (v1c) | Pv2h | Pv2i |
|---|---|---|---|
| H/V | 9 | 12 | 7 |
| a1/a0 | 12 | 16 | 16 |
| flavour | 16 | 6 | 5 |
| level | 130 | 27 | 23 |
| nu/nubar | 49 | 20 | 20 |
| phase | 2 | 2 | 2 |
| total | 218 | 83 | 73 |

  The x2 flag set has no row-resolved U/D at |cos Z| < 0.3; the near-horizon detail below is from `x5_near_horizon.py` (`out/near_horizon.md`).

Near-horizon detail, species means, E bands 0.11-1 GeV (Pv2h -> Pv2i; Honda/Bartol values in `out/near_horizon.md`). Expectations were
stated before the run; no tuning.

1. **Down-going unchanged below 0.4 GeV within 1%: MET.** Max |Pv2i/Pv2h - 1| = 0.41% over all 10 rows x 4 species x 3 bands (0.40% /
   0.41% at rows 0.05 / 0.15; this is f_b = 0.9955 / 1.006).
2. **Up-going H/V at |cos Z| < 0.2 below 0.4 GeV moves 3-8% toward Honda: PARTLY MET.**
   * Row 0.05: Pv2i/Pv2h - 1 = -8.3% / -7.4% / -4.8% at 0.11-0.16 / 0.16-0.25 / 0.25-0.40 GeV (H/V 2.100 -> 1.927, 1.759 -> 1.629,
     1.452 -> 1.383; Honda 1.951-2.085, 1.627-1.722, 1.377-1.438). The gap to Honda's mean was 3.9 / 4.8 / 3.1%, so the move closes it
     and overshoots by 1-4%; Pv2i sits at or just below Honda's lower edge but inside the Honda-Bartol envelope.
   * Row 0.15: -3.0% / -3.1% / -2.0%. Pv2h was already below the Honda mean there (Honda higher by 3.7 / 2.2 / 2.5%), so this moves AWAY
     from Honda (Pv2i is 4.8-4.9% below Honda's lower edge at 0.11-0.25 GeV, inside the envelope only because Bartol is lower).
3. **U/D at |cos Z| 0.05 below 0.4 GeV drops about 8%: MET at the two lowest bands** (-8.0% / -7.7%; -5.1% at 0.25-0.40 GeV).
   **WARNING, new U/D flags against the Honda/Bartol envelope** (ln distance beyond the nearest edge, > 0.03 flagged):

| U/D cell | Pv2h | Pv2i |
|---|---|---|
| 0.05, 0.11-0.16 GeV | +0.014 (inside) | **-0.046** (NEW, now below the envelope: U/D 0.996 vs 1.043-1.067) |
| 0.05, 0.16-0.25 GeV | +0.030 | -0.026 |
| 0.05, 0.25-0.40 GeV | +0.037 | inside |
| 0.05, 0.40-0.63 GeV | +0.042 | +0.023 |
| 0.15, 0.11-0.16 GeV | -0.035 | **-0.063** (worse) |
| 0.15, 0.16-0.25 GeV | inside | **-0.033** (NEW) |

   Net: 4 flagged U/D cells in Pv2h (3 high, 1 low), 3 in Pv2i (all low): the high-side excess at 0.25-0.63 GeV and row 0.05 is gone,
   but the 0.1-0.25 GeV cells overshoot to the low side. The x2 flag set does not include these rows (it has U/D only for |cos Z|
   0.3-1), so the flag table above is unaffected.

Other: down-going H/V at rows 0.05 / 0.15 rises 1.3-1.7% at 0.40-1 GeV (f_b) and keeps one flag per row at 0.63-1.0 GeV (-0.044) and
at 0.40-0.63 GeV row 0.15 (-0.034); E-W a1/a0 is unchanged to 0.001 (corrections are azimuth-independent): down-going rows still 0.01-0.04 below the Honda mean, up-going
0.02-0.14 above it at 0.11-0.63 GeV (pre-existing in v2h). Species-mean level excursions below 1 GeV stay within 5.5% of the envelope
(Pv2h 5.5%).

### 14.5 Reproduce and sha256

    source $W/runs/dbv142/env.sh; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        MCEQ3D_CACHE=$W/mceq3d_cache TMPDIR=$W/tmp MCEQ3D_NJOBS=4 HZN=$W/runs/hzn V2I=$W/runs/v2i
    $PY $W/runs/v2i/scripts/c0_corrections.py                       # -> out/corrections_v2i.npz
    cd $R/tools/mceq3d
    $PY build_table_v2.py build-v2i [out/corrections_v2i.npz]       # -> out/kamioka_3D_v2i.npz (needs v2h sha below)
    $PY build_table_v2.py gates-v2i                                  # -> out/gates_V1V2.md
    cd $W/runs/v2i; $PY scripts/x1_eval_products.py Pv2i; for x in x2_analyze x3_summarize x4_tables x5_near_horizon; do $PY scripts/$x.py; done
    DAEMONFLUX_3D_TABLE_V1=... _V2H=... _V2I=$W/runs/v2i/out/kamioka_3D_v2i.npz $PY -m pytest tests

Full sha256 (computed):

    e4029e6ce3baede6e4634dcdb34f9f54ebde3ecb7b716f93cb892353127e3be5  out/kamioka_3D_v2i.npz
    a5fb8fe3608fb115c0d1b4fa58d60936bdbfaf349d85058d95f10c5f642e200a  out/corrections_v2i.npz
    e3db406733bbe64fd9534c3366214b423f622f6a702b95518e4731a342a711ce  ../v2h/out/kamioka_3D_v2h.npz
    ffa41e0c4edb26f9b1a00bf0e1a250a68a227c86f03310dbabc085eab280d511  ../integ/out/kamioka_3D_v1c.npz
    a8b50a860df456e8dcc85f86e02afcaa7e7b82106514c096e6fe1a3ed2ad69dd  ../hzn/out/a2_decomp_Z.npz
    287b053803ead7bb86c4315aeb830fd919b9602676fdabcbe032133e94846e3a  ../hzn/out/a2_decomp.npz
    7b6bf0eacad101109f78edccdeca0b251c2bb6896304095f7cb08e04afba1183  ../hzn/out/b4_upgoing_template.npz
    d855057d057af77e8e6b41491cc0cddf6a0f52a6b536d1c50eff7f78d3f4e224  ../m4/sky/B_total.npz
    f760923535b9c93c60fe9feda96a00fc94301e9ed4fca9eda7a6a72b096ff1ca  ../lowE/out/s1d_cap.npz
    9179f71be6aa074a777ccccd3fa9c1f741ddaddc18372ffedb7b044f82b921bd  ../hzn/scripts/a15_b_measure0.py
    d86abe5bc729085ea2d5a5a1a15624b1aab52e9d2d18b7cb45f54a23647d3c6a  ../hzn/scripts/hzn_lib.py

### 14.6 PM review (2026-11-10)

* **PM checks:**
  * sha256 of v2i, the corrections file, v2h, v2 and v1c recomputed: all as in 14.5.
  * Identities recomputed independently: factor_3d_v2i / factor_3d_v2h / corr_v2i - 1 <= 2.2e-16; cells outside the 80 touched ones are
    bit-identical; factor = factor_3d x hybrid_correction exactly; (factor_error/factor)^2 = sum of the seven terms to 6.7e-16.
  * f_b, c_up and the touched rows (10/11 down, 9/8 up) are as specified.
  * daemonflux tests rerun with `DAEMONFLUX_3D_TABLE_V1/_V2H/_V2I`: 57 passed, 1 skipped.
* **Spec issue (PM):** se_mean = sqrt(sum_k mc_err^2)/4 does not reproduce hzn's jackknife species-mean errors exactly (6-10% low at
  row -0.05, slightly high at -0.15). The effect on err_upcorr_rel is <= 0.3 points. Left as built.
* **The pre-stated U/D warning fired.**
  * Background: Honda and Bartol agree on U/D at the horizon to about 2%, although they differ by up to 24% in H/V. In those models the
    up/down-symmetric geometry largely cancels in U/D, and what remains is mainly the cutoff asymmetry.
  * In this table the two hemispheres come from different calculations (MC down, engine up), so U/D is a direct test of their consistency.
  * Before the correction, the up-going cells carried the v4 excess of about +8% at |cos Z| 0.05. Yet U/D was only +1.4 to +4.2% high at
    row 0.05, and already 3.5% low at row 0.15, 0.11-0.16 GeV.
  * With the excess removed, U/D is 3-6% below the envelope at 0.1-0.25 GeV in both rows.
  * Reading: the up-going near-horizon cells very likely carry a second error of the opposite sign, about 4-7% at 0.1-0.25 GeV.
    Untested candidates: the v1c-era mirror-cone ratio, the single far-side cutoff, R_up, and the missing far-side in-atmosphere bend.
    This is the situation the lowE study warned about: two up-going errors partly cancel, so do not fix one alone.
  * The correction does improve 0.25-0.63 GeV at row 0.05 (three high-side flags removed).
* **PM recommendation:**
  * Keep changes (a) and (b) on the down-going MC cells; they are not in question.
  * Do not apply the up-going correction (c) as a central value. Carry it as an uncertainty, |1 - c_up| combined with err_upcorr_rel,
    until the up-going near-horizon side is tested directly (a far-side check, estimated earlier at 7-29 core-h).
  * Decision pending with the PI. As built, v2i applies (c).
* **Status:** candidate. v2h stays the reference candidate until the PI decides. v1c is unchanged; nothing in `src/daemonflux` changed.

## 15. Candidate v2j: v2i with the up-going correction carried as an error only (2026-11-12, CANDIDATE)

File: `$W/runs/v2j/out/kamioka_3D_v2j.npz` (label `kamioka_3D_v2j`, status `candidate`), format unchanged. Builder:
`tools/mceq3d/build_table_v2.py build-v2j` / `gates-v2j` (the same `_build_candidate` code path as `build-v2i`). Correction numbers:
`$W/runs/v2j/scripts/c0_corrections_v2j.py` (reads `v2i/out/corrections_v2i.npz` and `v2j/out/b4_jk_v2j.npz`). v2h, v2i untouched.

### 15.1 Definition
Reason (14.6): with c_up applied, v2i's U/D at |cos Z| 0.05/0.15 fell 3-6% below the Honda/Bartol envelope at 0.1-0.25 GeV.
* Kept from v2i: (a) cap-bias factor f_b on the MC horizon cells (source 0, rows +0.05/+0.15, E bins 0-6) with err_capb_rel; (b) err_horizon_rel.
* Changed: (c) c_up = 1 everywhere (`corr_cup` == 1). The up-going rows (-0.05/-0.15, table indices 9/8) keep their v2h `factor_3d` and
  `factor` bit for bit. `corr_v2j` = `corr_fb`. Down-going rows 10/11: `factor`, `factor_3d`, `factor_error` bit-identical to v2i.
* `err_upcorr_rel` is species-resolved (4 species, E bins 0-4, rows -0.05/-0.15): E bins 0-2
  sqrt(se_jk^2 + (mean_k template - mean_k mc)^2 + (1 - c_up_v2i[species])^2); E bins 3-4 |1 - mean_k template| (v2i value, species-broadcast); 0 elsewhere.
* Fix of v2i's spec error: se_jk is now the jackknife error of the species-mean MC field-free ratio (hzn b4 machinery, replicates
  in `hzn_lib.replicates`), not sqrt(sum mc_err^2)/4 (6-10% low at row 0.05). Reproduces hzn's quoted values [m]:
  row 0.05 0.0284 / 0.0654 / 0.0488 (bands 0.10-0.16 / 0.16-0.25 / 0.25-0.40), MC 0.909 +- 0.028 at 0.10-0.16; row 0.15 0.0143 / 0.0155 / 0.0189.

### 15.2 Errors
(factor_error/factor)^2 = sum of the seven err_*_rel^2 (same terms as 14.3), max deviation 6.7e-16 [m].
err_upcorr_rel (E bins 0-4):

| row | species | 0.10-0.16 | 0.16-0.25 | 0.25-0.40 | 0.40-0.63 | 0.63-1.0 |
|---|---|---|---|---|---|---|
| -0.05 | nue | 0.0971 | 0.1210 | 0.0978 | 0.0171 | 0.0107 |
| -0.05 | antinue | 0.0961 | 0.1276 | 0.1015 | 0.0171 | 0.0107 |
| -0.05 | numu | 0.0837 | 0.1040 | 0.0702 | 0.0171 | 0.0107 |
| -0.05 | antinumu | 0.0774 | 0.0997 | 0.0689 | 0.0171 | 0.0107 |
| -0.15 | all | 0.0414 | 0.0376 | 0.0282 | 0.0040 | 0.0013 |

Median relative factor_error, v2h -> v2j (rows -0.05 / -0.15, E bins 0-2): 0.263/0.256 -> 0.278/0.260; 0.239/0.231 -> 0.264/0.234;
0.214/0.209 -> 0.232/0.211. Down-going rows equal v2i.

### 15.3 Gates (builder, `out/gates.md`), all PASS [m]
Protected sha256 (v1c, v2h, v2, v2i, corrections_v2i) unchanged; factor == factor_3d x hybrid_correction exactly; rows 10/11 factor,
factor_3d, factor_error == v2i; rows 8/9 factor, factor_3d == v2h; up-going factor_error differs from v2h in all 480 cells of E bins 0-4
and nowhere else (1152 cells in total including the 672 down-going cells, as v2i); every cell outside rows 8-11 / E bins 0-6 is
bit-identical to v2h in all 12 grid arrays; `build-v2i` rebuilt into `v2j/work` has arrays bit-identical to v2i (only build_date and code_git_rev differ).

### 15.4 External check (Honda solmin/solmax, Bartol fmin/fmax; flags > 0.03 in ln beyond the envelope)
Pre-stated expectation (written into gates.md before the run): up-going categories = Pv2h, down-going = Pv2i, U/D = Pv2h up to f_b.
Result: up-going level/flavour/nu-nubar/a1/a0/phase flags = Pv2h exactly (0 flag-set differences, 0 value differences); down-going
H/V, level, flavour, nu/nubar flags = Pv2i exactly. Flag totals P1 218 / Pv2h 83 / Pv2i 73 / Pv2j 77 (H/V 7, a1/a0 16, flavour 6, level 26, nu/nubar 20, phase 2).
U/D at |cos Z| 0.3-1 (x2): 0 flags for all products. Near-horizon U/D, species mean, 0.1-1 GeV, with the table error [e]:
see HANDBACK.md; Pv2j has 2 flagged cells (0.05 at 0.25-0.40: +0.033; 0.15 at 0.11-0.16: -0.032; Pv2h had 4, Pv2i 3), no new ones, and every
miss is 0.00-0.11 of the table's own U/D error (about 0.2-0.36).
The U/D change relative to Pv2h comes only from f_b in the down-going denominator (<= 0.4% below 0.4 GeV, up to 1.8% at 0.4-1 GeV).

### 15.5 Reproduce and sha256
    cd $W/runs/v2j     # env: runs/dbv142/env.sh, OMP_NUM_THREADS=1 etc.
    PYTHONPATH=$PYTHONPATH:$W/runs/hzn/scripts $PY scripts/b4_jk_v2j.py      # -> out/b4_jk_v2j.npz (hzn b4 log reproduced identically)
    $PY scripts/c0_corrections_v2j.py                                       # -> out/corrections_v2j.npz
    $PY $R/tools/mceq3d/build_table_v2.py build-v2j                          # -> out/kamioka_3D_v2j.npz
    $PY $R/tools/mceq3d/build_table_v2.py gates-v2j                          # -> out/gates.md
    V2_OUTDIR=$W/runs/v2j/work $PY $R/tools/mceq3d/build_table_v2.py build-v2i   # arrays == v2i
    a99adaa6485ab6e8118b949cdd5174bfbc8bc0e83b0fba539add51e3bd09ac2d  out/kamioka_3D_v2j.npz
    ab858d7be9bf8cba7923ce30fc3c620a7f228d3f533904c609bc4c51e5737fc8  out/corrections_v2j.npz
    fb51afd03dedde1ed62603bf6dc8d8a1a0ee238ee1ee46b85a0093044d1966a2  out/b4_jk_v2j.npz
Tests: `DAEMONFLUX_3D_TABLE_{V1,V2H,V2I,V2J}` set, `pytest tests`: 58 passed / 1 skipped (new `test_kamioka_v2j_table_if_available`).

### 15.6 PM review (2026-11-12)

* **PM checks:**
  * sha256 of v1c, v2h, v2, v2i and corrections_v2i recomputed: unchanged. v2j and corrections_v2j are as in 15.5.
  * Identities recomputed independently, all exact: factor = factor_3d x hybrid_correction; rows 8/9 factor and factor_3d equal v2h;
    rows 10/11 factor, factor_3d and factor_error equal v2i; every cell outside rows 8-11 / E bins 0-6 equals v2h (factor and
    factor_error); the error quadrature closes to 6.7e-16.
  * daemonflux tests rerun with `DAEMONFLUX_3D_TABLE_{V1,V2H,V2I,V2J}`: 58 passed, 1 skipped.
  * The PM wrapped seven lines over 88 characters in `build_table_v2.py` (formatting only; black and flake8 are not installed in
    the LCG environment). A v2j rebuild into a scratch directory after the reformat gives identical arrays and provenance.
* **Brief erratum (PM):** the brief's expectation "the U/D cells reproduce Pv2h's flags exactly" ignored that U/D has the
  down-going cells, which carry f_b, in its denominator. The agent wrote the corrected expectation into gates.md before the run.
  The result matches it: 4 -> 2 near-horizon U/D flags, none new.
* **The table's per-cell errors cannot test U/D at the few-% level (PM caveat on 15.4):**
  * The statement that every U/D miss is "0.00-0.11 of the table's U/D error" treats the up and down cells as uncorrelated.
  * err_hybrid_rel (16-22% below 0.63 GeV) is identical in the two hemispheres and cancels in U/D. Parts of err_v1c_rel
    (sigma_model's general 0.1-0.25 GeV level term) are common as well.
  * Even without the hybrid term, the uncorrelated U/D error is 14-20%. Most of it is err_v1c_rel: 10-14% up-going, 7-9% down-going.
  * The table stores no up/down correlation. The external U/D check therefore stays the operative test of hemisphere consistency.
  * Open item: a correlation model, or a stored U/D-relevant error.
* **err_upcorr may double-count:** it has both the template-minus-MC term and |1 - c_up|, giving 8-13% at |cos Z| 0.05 below
  0.4 GeV. This is conservative by intent while the up-going side is unresolved. It is to be replaced by the MC up-going
  measurement (run `mcup`, in progress).
* **Status:** v2j supersedes v2i. It is the recommended candidate following the PI's go-ahead (2026-11-12) on the recommendation in
  14.6. v2h stays in place as the parent; v1c is unchanged; nothing in `src/daemonflux` changed.
* **In progress:**
  * `mgfix`: the engine's multigeneration horizon excess, as a new opt-in E_off model.
  * `mcup`: an MC second-root up-going scorer, which measures rows 8/9 directly.

  If mcup closes, rows 8/9 become MC cells like rows 10/11 (planned v2k).

## 16. E_off v5 and the near-horizon mechanism of the engine (2026-11-12, DIAGNOSTIC: v5 fails its pre-registered gate)

Run directory `$W/runs/mgfix` (`PREREG.md` with addenda A/B, `HANDBACK.md`, `scripts/`, `logs/`, `out/`). New repo files:
`tools/mceq3d/eoff_v5.py`, `tools/mceq3d/diag_eoff_horizon.py`, `tools/mceq3d/test_eoff_v5.py`,
`tools/mceq3d/test_diag_eoff_horizon.py`; `tools/mceq3d/mceq3d_flux.py` gets the `eoff_model="v5"` plumbing only.
`eoff_model="v4"` is unchanged (bit-identical engine output, 16.5). Labels [m] measured, [d] derived, [e] estimate.

### 16.1 What was asked, what was found

The hzn study (section 14.1) isolated the multigeneration kernel of E_off v4 as the ingredient that raises the engine's horizon
departure (+13% at |cos Z| 0.05, 0.10-0.16 GeV) and found the MC's field-free shape below v4 (D(P') = -0.096 +- 0.028). Its
hypothesis: later-generation vertices displaced from the primary line, misplaced by the factorised cone. Result:

* **The factorisation is not the main cause, and displacement is not a cause.** A field-free, spherically symmetric toy that runs
  the MC's own cascade code (tools/mc3d physics, imported read-only) and scores every neutrino reproduces the MC's horizon shape
  (6/6 cells, e.g. D = -0.104 +- 0.012 against v4 at row 0.05, 0.10-0.16 GeV; MC -0.096 +- 0.028) [m]. On the same chains, moving
  every vertex onto its primary line changes the horizon by <= 1-2% (displacement), crediting the chains the atmosphere truncates by
  <= 0.2%, decorrelating depth and angle by ~1%; the whole product form (vertex on the primary line, kernel independent of depth and
  symmetric about the primary) errs by +2.2 +- 1.6% / +4.2 +- 2.0% at row 0.05 for 0.10-0.16 / 0.16-0.25 GeV, ~0 at row 0.15 and
  0.25-0.40 GeV, and about -2% at 0.4-1 GeV (two independent toy runs combined) [m]. The later-generation part of every step is
  <= 2%: the multigeneration kernel itself is not wrong.
* **The main cause is an input: the density of the muon energy loss.** v4 evaluates its mu-channel kernel at ONE density per row,
  the decay-weighted density of the ON-AXIS production (0.26e-4 g/cm3 at |cos Z| 0.05). The horizon enhancement is carried by
  off-axis samples (steep primaries feeding deep production points); for the mu channel at 0.10-0.18 GeV and row 0.05 their
  log-mean density is 1.3e-4, 3.6x the on-axis value; at row 0.65 the two agree (1.1e-4 / 1.2e-4) [m]. The row kernel is
  therefore too wide exactly where the horizon is made; a wide (all-generation) kernel makes this visible.
  Evaluating the kernel at each sample's own density (v5) closes the engine-vs-toy factorised gap (pre-registered reading 6/6) [m].
* **The MC reference has a small artefact at 0.10-0.25 GeV.** tools/mc3d drops charged pions below e_nu_min/(1 - m_mu^2/m_pi^2)
  = 0.234 GeV (exact for pi -> nu, not for pi -> mu -> nu: it removes muon-decay neutrinos up to ~0.21 GeV, 4.5% of the 0.10-0.16
  GeV flux at row 0.05, 1.9% at 0.95). Paired on the toy, the cut lowers the field-free horizon departure by 0.021 +- 0.003 /
  0.014 +- 0.001 / 0.013 +- 0.001 (rows 0.05 / 0.15 / 0.25, 0.10-0.16 GeV) and 0.017 (row 0.05, 0.16-0.25 GeV); 0 above [m].
  It also explains about 2/3 of the 6% mu-channel width difference between the engine kernel and the MC's realised cone at
  0.10-0.16 GeV.
* Not causes [m]: escape/ground truncation (M1: the MC's later-generation share is HIGHER for near-horizontal primaries, opposite to
  the truncation prediction; toy <= 0.2%), the cone quadrature (doubling n_beta and n_ray moves the horizon by <= 0.1%), the
  engine's on-axis 1D vs B (<= 0.5% below 0.4 GeV).

### 16.2 The v5 definition (`eoff_v5.py`)

    J_s = sum_c sum_j sum_alpha W_{s,c,j}(alpha) mean_beta[ q_{s,c,j}(alpha, beta) G_s(alpha, beta) ]
    q_{s,c,j}(alpha, beta) = Int dl rho(h) lambda_j(rho(h)) p_{s,c}(X_slant(l, psi_p(alpha, beta)), E; theta_g)

* `lambda_j(rho)`: linear-in-ln-rho weights over the density nodes `rho_j` = eoff_v4.RHO_FAMILY + RHO_EXTRA = (0.3, 0.6, 1.2, 2.5,
  5.0, 10) x 1e-4 g/cm3, clamped at the ends; evaluated at every arrival-ray point (the neutrino production point = the muon decay
  point). Sum over j = 1, so sum_j q_j is v4's q.
* `W_{s,mu,j}`: the v4 energy-loss kernel (all generations, polarisation, K -> mu) at rho_j (eoff_v4.kernel_family; the 1e-3 member
  is new, cache key `el0.001`). Direct and kaon channels: v4's kernels, unchanged.
* Everything else is v4: zenith-node depth profiles, geometry, cutoff, production-point frame, band stratification. Up-going rows use
  the v5 G = 1 table (`offaxis_excess_channel_v5.npz`, built by `eoff_v5.build_table_v5`, scheme `v5a`, `table_tag` with the nodes);
  the engine refuses a non-v5 table under `eoff_model="v5"`.
* Use: `MCEq3DFlux(..., eoff_model="v5")`, `solve(..., offaxis_table=<v5 table>)` (default `EOFF_TABLE_V5`, not committed: the
  table is in `$W/runs/mgfix/out/`, sha256 below). The v5 joint q's are memoised separately (`_joint_q_v5`).
* No free parameter; nothing is fitted to the MC.

### 16.3 The E_off horizon gate (`diag_eoff_horizon.py`)

    $PY tools/mceq3d/diag_eoff_horizon.py --addon $W/runs/m4/work/addon_addon.npz --sums $W/runs/hzn/out/a1_addon_sums.npz \
        --base $W/runs/integ/out/base1d_kamioka.npz --fine $W/runs/integ/out/engine_F_fine.npz TABLE [TABLE ...]

The field-free P' test of hzn (b3): species-mean D(P') at rows 0.05/0.15/0.25 x 5 bands 0.10-1.0 GeV, gate max(3%, 2 sigma_jk).
It reproduces hzn's numbers exactly (v4 11/15, Z 14/15). Together with fix2's r'' closure (cos Z 0.2-1) it is the E_off gate:
fix2 tested only cos Z >= 0.2. Its reference carries the soft-pion cut (16.1): at 0.10-0.16 GeV the MC's horizon is 1.3-2.1%
low.

### 16.4 Results (criteria pre-registered in `PREREG.md`; numbers in `HANDBACK.md` sec. 4)

| criterion | v4 | **v5 (pre-registered)** | v4c (interim, post hoc) |
|---|---|---|---|
| (i) P' horizon gate, 15 cells | 11/15 | **12/15 FAIL**: 0.10-0.16 row 0.05 -0.069(28); 0.63-1.0 rows 0.05/0.15 +0.094(39), +0.040(19) | 15/15 (not independent: the toy shares the MC's physics) |
| (ii) fix2 r'' gate, cos Z 0.2-1 | 6/60 fail | **13/60 FAIL**, 7 new, 20 cells worse than v4 by > 1 sigma (engine 1-3% lower at cos Z 0.2-0.6 below 0.63 GeV) | 7/60 FAIL, 1 new, 6 worse (all at cos Z 0.2-0.3, 0.4-1 GeV, from c(0.25) = 1.02-1.03) |
| (iii) Honda/Bartol H/V (40 cells) and E-W | 20 cells fail; a1/a0 -0.053/-0.035 | **21 fail; a1/a0 -0.037/-0.023 (MC -0.026/-0.022) FAIL** | 16 fail; a1/a0 unchanged FAIL |

* (iii)'s H/V failures at 0.11-0.25 GeV come from the engine's cos Z 0.6-1 rows (fix2's R1: r'' 1.07-1.09 at 0.10-0.16 GeV), which
  normalise H/V. No horizon-only E_off change can pass that cell set.
* Reading: the local-density energy loss is the right direction at the horizon, but the vertex density over-corrects at
  intermediate zenith and at 0.4-1 GeV, where v4 was right. **v5 is not recommended for production.**
* The interim factor `out/c_horizon_toy.npz` is first-principles (MC physics, field-free, every pion) and site-independent. If the
  PM wants an engine horizon correction for up-going cells or other sites, it is the better-founded replacement for hzn's
  E_off_v4/E_off_Z template at |cos Z| < 0.2.
  * At |cos Z| 0.25, its 0.4-1 GeV values (1.02-1.03 +- 0.012) conflict with the MC closure.
  * Using it only at |cos Z| < 0.2 is a post hoc restriction and is not applied here.

### 16.5 v4 bit-identity, tests, CPU, files

* `eoff_model="v4"` after the edits: the engine solve on the hor grid (fix2 t4_engine recipe) equals `fix2/out/t4_D_hor.npz` in
  every array (flux, Fnew, Fphys, G1, cutoff, base). `eoff_v4.py` is unchanged.
* Tests (named files, `MC3D_POOL_CACHE`, from `tools/mceq3d`): 86 passed, 2 skipped.
  * test_mceq3d_flux 16, test_mceq3d_deterministic 3, test_mceq3d_production 3, test_offaxis_mc 13, test_joint_cone 24 + 2
    skipped, test_upgoing_cone 3;
  * new test_eoff_v5 19, new test_diag_eoff_horizon 5.
* CPU: 4.0 core-h, `$W/runs/mgfix/logs/ledger.txt`.
* sha256:
  * `out/offaxis_excess_channel_v5.npz` ca6999de910ca40dcb5cc6b68c36ae1173d69fdda6816cb3013de50640ead77d;
  * `out/c_horizon_toy.npz` edaf02c3d6c91b9c912c688daee6cedbb9b2481c4d4fb6918bdc5942d9339c3a;
  * `out/t4_v5_up.npz` 53bc99d92460ec84d4b38a29bdc9867d78f27f7be47a40b44260a53323abf2aa;
  * `out/t4_v4c_up.npz` 0ef38baaaaa1d69b75c0efab6e1be76ba8fde2f9dd56c57a0feaadf91b320ede.
* Reproduce. Cores 4-11. Engine jobs need cwd `tools/mceq3d` (`m_spliced_v2.npz` is read from the cwd).

      source $W/runs/mgfix/env.sh; cd $W/runs/mgfix
      ./run.sh toy_main 4-11 $PY scripts/toy3d.py run $MG/work/toy_main 8 20261112 450000 280000 67000 9600 2900 1000 300
      TOY_PION_THR=none ./run.sh toy_nocut 4-9 $PY scripts/toy3d.py run $MG/work/toy_nocut 6 20261113 315000 196000 47000 6700 2000 700 210
      $PY scripts/toy_analyse.py work/toy_main out/toy_H_main.npz 8; $PY scripts/toy_analyse.py work/toy_nocut out/toy_H_nocut.npz 6
      $PY scripts/toy_final.py; $PY scripts/i6_report.py; $PY scripts/k_interim.py
      RUNCWD=$R/tools/mceq3d ./run.sh b_build_v5 10 $PY scripts/b_build_v5.py
      RUNCWD=$R/tools/mceq3d ./run.sh e_v5_up 8 $PY scripts/e_engine.py v5 up    # also hor, dn6; "v4 hor" = the identity check

### 16.6 Up-going engine rows (step 5)

`out/t4_v5_up.npz` and `out/t4_v4c_up.npz` follow the v2h recipe (10 rows x 12 az). Ratios to v2h's v4 rows, species mean, at
|cos Z| 0.05 / 0.15:

| E [GeV] | v5 / v4 | v4c / v4 | hzn template | MC field-free |
|---|---|---|---|---|
| 0.10-0.16 | 0.949 / 0.973 | 0.939 / 0.983 | 0.916 / 0.945 | 0.909(28) / 0.970 |
| 0.16-0.25 | 0.941 / 0.965 | 0.940 / 0.978 | 0.926 / 0.956 | 0.977(65) / 0.967 |
| 0.25-0.40 | 0.943 / 0.965 | 0.928 / 0.998 | 0.952 / 0.978 | 0.903(49) / 0.979 |
| 0.40-0.63 | 0.961 / 0.976 | 0.958 / 0.999 | 0.983 / 0.996 | 1.01 / 1 |
| 0.63-1.0 | 0.974 / 0.984 | 1.014 / 1.018 | 0.989 / 0.999 | 1.07 / 1 |

* v5 also lowers |cos Z| 0.35 by 0.7-1.5% and 0.4-1 GeV by 2-4%. v4c leaves |cos Z| >= 0.35 untouched.
* Neither addresses the second up-going error flagged in 14.6 (mirror cone, far-side cutoff, R_up).

### 16.7 Not established / open

* **The effective density for the muon energy loss.** The vertex density over-corrects. The path-average along the muon flight is
  the natural next candidate; it is untested, and the engine has no per-sample muon direction.
* **The residual product-form error.** It is +2-4% at row 0.05 below 0.25 GeV and -2% at 0.4-1 GeV, with no single step.
* **The mu kernel at 0.10-0.16 GeV** is still 2-3.5% wider than the MC-physics cone.
* **The cutoff part of the joint integral is untested by the field-free toy.**
* **The MC's soft-pion cut (tools/mc3d `shower.thresholds`, pions below 0.234 GeV).**
  * It biases every MC product at 0.10-0.25 GeV, including the v1c/v2h/v2i MC cells. For the horizon shape: -2.1 / -1.4 / -1.3%
    at rows 0.05 / 0.15 / 0.25, 0.10-0.16 GeV.
  * Suggested fix (not applied, other agents' files): pion threshold at m_pi; a muon-decay neutrino above e_nu_min needs
    E_pi > ~0.15 GeV (kinematic limit).
* **The row-0.05 measure-1 estimators** (toy and MC) have a log-divergent variance (cos Z -> 0); their jackknife errors are lower
  bounds.

### 16.8 PM review (2026-11-12)

* **PM checks:**
  * The named engine tests were rerun by the PM (test_eoff_v5, test_diag_eoff_horizon, test_mceq3d_flux, test_joint_cone,
    test_upgoing_cone): 67 passed, 2 skipped, the same as the agent's counts for those files.
  * The v4 identity was recomputed: `mgfix/out/t4_v4_hor.npz` equals `fix2/out/t4_D_hor.npz` in all 26 arrays except `cpu_s`.
  * `eoff_v4.py` is unchanged (sha256 8252a5af...).
  * The diff of `mceq3d_flux.py` was read. It is plumbing only: the v5 branch and a table-scheme guard. Default and v2-v4 paths are
    untouched.
  * The MC pion threshold was confirmed in the code: `tools/mc3d/shower.py` `thresholds()`, pi+- below
    e_nu_min / (1 - m_mu^2/m_pi^2).
* **Reading:**
  * The brief's hypothesis (vertex displacement of later generations) is refuted. The "multigeneration" label from hzn named an
    ingredient switch, not the mechanism.
  * The supported cause is an input: v4 applies one mu-channel energy-loss kernel per row, at the on-axis decay density. The
    off-axis samples that make the horizon sit in air about 3.6x denser.
  * The toy (the MC's own cascade code, field-free, full sphere) reproduces the MC's horizon shape in 6/6 cells at 2-3x smaller
    error. This independently confirms that v4 is about 8% high at |cos Z| 0.05 below 0.4 GeV. The toy shares the MC's physics, so
    it is not independent of the MC's physics modelling.
* **v5 fails all three pre-registered criteria. It stays an opt-in diagnostic, not a candidate; its table is not committed.**
* **v4c (v4 x c_toy) is post hoc.** It fails (ii) narrowly, at cos Z 0.2-0.3, 0.4-1 GeV. It is not adopted. Its numbers are kept
  for comparison only.
* **Criterion (iii) as briefed could not be passed by any horizon-only change (PM brief error).** H/V divides by the cos Z 0.6-1
  rows, where the engine is 7-9% low at 0.10-0.16 GeV (fix2's R1). A future gate needs a horizon-relative external metric, written
  down before the numbers.
* **New MC defect (affects every MC product at 0.10-0.25 GeV):** `shower.thresholds` sets the pion threshold for the direct
  channel only. It drops pions whose muons can still give neutrinos above e_nu_min.
  * The soft-ancestry share of the 0.10-0.16 GeV flux is 4.5% at |cos Z| 0.05 and 1.9% at 0.95.
  * In the table convention G = s/(kappa B), the level largely cancels through kappa. The horizon rows keep a bias of about -1.3 to
    -2.1% at 0.10-0.16 GeV.
  * Not fixed yet. tools/mc3d is in use by the `mcup` run, which keeps production B's threshold so that the up- and down-going rows
    share one convention. The fix (threshold e_nu_min for pi and K, opt-in or with a version tag) and its correction of the MC cells
    are scheduled after mcup.
* **Up-going engine rows:** v5/v4 and v4c/v4 are 0.93-0.95 at |cos Z| 0.05 below 0.4 GeV, consistent with hzn's c_up template.
  The second, opposite up-going error (14.6) is not addressed here; that is mcup's job.
* **Status:**
  * The engine horizon defect is located but not fixed.
  * v4 stays the engine default for eoff_model="v4". The Kamioka table is unaffected: rows +0.05/+0.15 are MC cells, and rows
    -0.05/-0.15 carry the error-only correction of v2j.
  * Before the engine is used on its own near the horizon at any site, it needs a model that passes the gate, or the masked or
    error-carried treatment of v2j.
