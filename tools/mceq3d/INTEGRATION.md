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
