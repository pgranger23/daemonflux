# Rigidity convention of G_s and the primary-model energy variable (2026-11-04)

Status: fixed in `tools/mceq3d`; measured in the delivered engine path
(`MCEq3DFlux(base_model="hybrid", ..., daemonflux_location="kamioka")`, default database
`v142p`, Kamioka, 2020-01-01). Raw outputs, logs and `NOTES.md`:
`/home/pigrange/mc3d_work/runs/rigfix/` (tables: `out/tables_rig.md`, regenerate with
`diag_rigidity_scan.py tables`). Nothing committed.

## 1. Headline

* **The bug.** `MCEq3DFlux.geomag_response` applied the cutoff to `R = (A/Z) * T` with `T`
  the kinetic energy per nucleon of MCEq's grid, instead of `R = (A/Z) p_N`,
  `p_N = sqrt(T (T + 2 m_N))`. At R_c = 7.5 GV protons below T = 7.50 GeV were removed
  instead of 6.62 GeV, bound nucleons (A/Z = 2) below 3.75 instead of 2.93 GeV/n.
* **Effect on G_s** (GSF, nu_mu, cos Z_ref = 1; before -> after, new/old - 1):
  R_c = 7 GV: +14.5% (0.1 GeV), +8.7% (0.3), +3.3% (1), +0.5% (3);
  11 GV: +11.0 / +8.5 / +4.6 / +1.1%; 40 GV: +4.8 / +4.3 / +3.3 / +1.7% (+0.4% at 10 GeV).
  nu_e is the same within ~1 point; H3a within ~1 point. So G_s was 3-15% too low
  below ~3 GeV at Kamioka-type cutoffs (the other agent's 3.5-10% estimate is the
  right order; the 7 GV, 0.1 GeV corner is larger).
* **Effect on the delivered observables** (table A, v142p before | after): H/V changes by
  -0.6..+0.2%; W/E nu_mu 0.5 GeV +1.4..+1.6% (87-75 deg; the overshoot of Honda at 87 deg
  goes from +10.4 to +12.2%), 1 GeV < 0.3%; W/E nu_e 0.5 GeV +2.2..+2.5%; charge
  splitting D(dphi) < 0.09 deg, D(max/min) <= 0.04; first-harmonic phase offsets
  -0.14..-0.22 deg; **the absolute level rises: N/S model/Honda at 0.5 GeV, 87 deg
  moves up by +6.5..+7.5% in every species** (nu_mu N 0.963 -> 1.026, nu_e N 1.065 -> 1.139),
  with N and S moving together (the N-S contrast is unchanged). N/S model/Honda was 0.93-1.07
  before and is 1.00-1.14 after; nu_e is now +6..+14% above Honda at 0.5 GeV, 87 deg.
* **The Phase-1 primary scan was wrong for Gaisser-Honda**, as the PM found: MCEq passes
  total energy per nucleon, the GH formula wants kinetic. The engine now has
  `GaisserHondaNative`; the rigidity-fixed scan (section 5) gives, for the nu_mu W/E at
  87 deg, 0.5 GeV (muon_bending = True, Phase-1 definition, Honda 2.507): GSF 2.813
  (+12%), H3a 2.587 (+3%), **GH-native 2.742 (+9%)**, GH-at-total (the Phase-1 "GH")
  2.516 (+0%). The statement of PHASE1_RESULTS 8.3 that Gaisser-Honda removes
  essentially all of the 0.5 GeV E-W residual is an artefact of the mis-evaluated GH;
  with its native variable GH removes about 3 of the 12 points, H3a (evaluated correctly
  at total energy) removes 9. At 1 GeV: GSF +19%, H3a +8%, GH-native +16%, GH-at-total +8%.
* Tests: full `pytest -q` from `tools/mceq3d` (default `v142p`): **280 passed, 2 skipped, 6 warnings in 2539 s**
  (275 + 5 new `test_rigidity.py` tests; the 2 skips are the usual `test_joint_cone.py` ones; log
  `runs/rigfix/logs/pytest_rig.log`)
* CPU: about 30 core-hours in total (G_s: 19 (primary, cos Z) families x 40 rigidity points
  as 152 chunk jobs = 26.6 core-h of process time, inflated by a loaded machine;
  engine and scan runs ~1.5 core-h; static-table rebuild 0.4 core-h; test suite 0.9 h).
  At most 16 cores at a time.

## 2. Code changes (`tools/mceq3d`)

* `mceq3d_flux.py`
  * `_nucleon_transmission(e_kin, rc_gv, sigma_lnr, az_over_z, m, dlne, nsub)`: transmission
    of nucleons on a *kinetic* grid with `R = az * sqrt(T (T + 2 m))`, `m = 0.938272 GeV`.
    The bin average is taken in the variable the bins are uniform in, `ln T`. Hard step
    (`sigma_lnr = 0`, the engine default `SIGMA_LNR`): exact, `clip((ln T + dlnT/2 - ln T_c)
    / dlnT, 0, 1)` with `T_c = sqrt(m^2 + (R_c/az)^2) - m`. Smooth step: 32-point midpoint
    rule per bin (integrand smooth on the bin scale for sigma >~ 0.03; agrees with the
    analytic momentum-grid form to < 3e-3 at T >> m). `az_over_z` may be per-bin
    (`az_bound(E)`). `geomag_response` calls it for the free (A/Z = 1) and bound
    (A/Z = `az_bound`) fractions. The old `_transmission` (momentum grid) is kept and
    still tested.
  * `_nucleon_split`: the `<A/Z>` weights `A^2 Phi(A E)` are now evaluated at the nucleus energy
    `A (e_grid + m)` with `m` the mean nucleon mass MCEq uses (`MCEqRun.set_primary_model`),
    i.e. consistent with the flux MCEq itself injects (before: `A * e_grid`).
  * `RIGIDITY_TAG = "rigp1"` is part of `engine_tag` and hence of `_tag`, `_gs_tag` and every
    `gs_*` file name: `SIBYLL23D_GlobalSplineFitBeta-None_emin0.1_atmstd_rigp1<db tag>`. No
    G_s built with the old convention can be read (the old key is a strict prefix of the new
    one, which keeps `offaxis_factor()`'s `startswith` check working). **Side effect:**
    even `MCEQ3D_DB=v140` now looks for `gs_*` under the new key (the shipped-database *tag*
    and the DB behaviour of `mceq3d_db` are unchanged; `.cache3d` was not touched and its
    old G_s files are simply never matched). Other caches (`jointprod_chan_*`,
    `channel_fractions*`, `muon_source_*`, maps, cone widths) do not depend on the rigidity
    convention and keep their keys.
* `mceq3d_primary.py` (new): `primary_class(name)`, the single resolver of `primary[0]`.
  `GaisserHondaNative` (nucleus energy shifted by `A m_N` before the crflux formula; used for
  the MCEq primary flux *and* `_nucleon_split`), `GaisserHondaAtTotal` (crflux GH unchanged,
  the bridge to Phase 1), `GaisserHonda` raises (ambiguous). `mceq3d_flux.py` and
  `mceq3d_real.py` use it. `diag_model_scan.py`'s `GH` entry now names `GaisserHondaAtTotal`
  so its published numbers remain reproducible. The class name is in the G_s key.
  Verification (all-nucleon flux / GSF at total energy, T/n = 3/5/7/10/15/20/30/50/100 GeV;
  PM's table in brackets):
  * GH native: 1.052 1.000 0.983 0.975 0.977 0.985 1.006 1.046 1.108 (1.05 1.00 0.98 0.98 0.98 0.99 1.01 1.05 1.11)
  * GH at total: 0.643 0.697 0.740 0.787 0.839 0.876 0.928 0.995 1.080 (0.64 0.70 0.74 0.79 0.84 0.88 0.93 1.00 1.08)
  * H3a at total: 0.880 0.742 0.697 0.675 0.674 0.686 0.717 0.775 0.872 (0.88 0.74 0.70 0.68 0.67 0.69 0.72 0.78 0.87)
* `test_rigidity.py` (new, 5 tests): the T at which a 7.5 GV cut acts (6.62 GeV / 2.93 GeV/n,
  and that the old convention would put it at 7.5), exact hard-step bin overlap and
  continuity in R_c, agreement with the momentum-grid form at high energy and monotonicity of
  the smooth step, the rigidity token in the G_s key (a key built the pre-fix way never equals
  a current one), and the GH variable against PM's numbers (atol 0.015) plus
  `GaisserHondaNative(T + m) == GaisserHonda(T)` exactly. `test_mceq3d_db.py` expects the
  new engine-tag string.
* `diag_rigidity_scan.py` (new): chunked G_s priming (the 40 rigidity points are independent
  solves; the assembled file was checked equal to a direct `geomag_response`, max
  difference 0.0), the scan, the `analyse_m3b` cells, and the tables. `analyse_m3b` is imported
  read-only; `tools/mc3d` is untouched.
* Not changed (flagged): `_modulate_phi0` (solar modulation) shifts the *kinetic* grid by
  the per-nucleon energy loss and uses `m_n` in the Jacobian; it does not use rigidity
  and was left alone. `offaxis_mc.production_profile(rc_cut_gv=...)` also calls the old
  momentum-grid `_transmission` on MCEq's kinetic grid (only used in a site-independence check).

## 3. Observable definitions

* Scan (C): `diag_ew_charge_fourier.model_flux` on Honda's 12 azimuth bins (cos Z bands
  0.25 / 0.15 / 0.05 = 75 / 81 / 87 deg, `cz_sub = 1`), values at the energy points, and
  `analyse_m3b.observables` for model and Honda alike (a1/a0 = E-W dipole amplitude, dphi =
  phase of the first harmonic minus geomagnetic West, fN/fS = flux at N / S relative to the
  mean (bins 345+15 / 165+195), NS = fN/fS, WE = two-harmonic W/E, maxmin = 12-bin max/min).
  Honda is HKKM2014 `kam-ally-20-12-solmin` in the same convention. Only 87 deg is
  tabulated here; 81 and 75 deg are in the json files.
* (D): Phase-1 definition, `solve(...)` at cos Z = 0.05, azimuths 90 and 270: W/E = f(270)/f(90)
  against Honda's max/min.
* (E): the 36 cells (4 species x E bands 0.3/0.5/1.0 x cos Z bands 0-0.1/0.1-0.2/0.2-0.3) of
  `analyse_m3b`: `cz_sub = 3` (so 9 values of cos Z), band-integrated over the same energy
  bands as Honda, H3a at total energy, `muon_bending = False`, species order of
  `analyse_m3b.SP_NAMES` (nue, antinue, numu, antinumu). File:
  `out/m3b_engine_h3a_mb0.npz` (the fluxes) and `out/m3b_obs_h3a_mb0.npz` (observables).

## 4. Part C: static tables on v142p

`offaxis_mc.build_channel(n_jobs=8)` takes 171 s wall (about 0.4 core-h) with the production
profile from v142p. The rebuilt tables (in `runs/rigfix/eoff/`, with the `db` key, **not**
copied over the tracked ones) differ from the tracked v1.4.0 ones by at most 0.10% in `E_off`
and `E_off_s` for 0.1-10 GeV (all four `offaxis_excess*` variants; the flat `offaxis_excess.npz`
is built by `build()` and was not rebuilt). The `channel_fractions` entries for the
new database are built on first use (and exist in `$W/mceq3d_cache`). Replacing the tracked
files would remove the two static-table `UserWarning`s; the effect on any number is
< 0.1%. Decision for the PM.

## 5. Tables (from `out/tables_rig.md`)

A: before/after on the headline observables; B: G_s; C: the primary scan (muon_bending False,
then True); D: W/E in the Phase-1 definition; E: the `analyse_m3b` cells for the MC comparison.


**A. v142p before | after the rigidity fix (delivered engine, hybrid base, GSF, Kamioka)**

| observable | before | after | after/before - 1 | Honda |
|---|---|---|---|---|
| H/V numu 0.3 GeV | 1.889 | 1.887 | -0.08% | 1.825 |
| H/V numu 0.5 GeV | 1.454 | 1.448 | -0.44% | 1.448 |
| H/V numu 1.0 GeV | 1.211 | 1.206 | -0.41% | 1.241 |
| H/V numu 3.0 GeV | 1.323 | 1.325 | +0.16% | 1.346 |
| H/V nue 0.3 GeV | 2.267 | 2.271 | +0.16% | 2.119 |
| H/V nue 0.5 GeV | 1.819 | 1.812 | -0.37% | 1.830 |
| H/V nue 1.0 GeV | 1.644 | 1.634 | -0.61% | 1.770 |
| H/V nue 3.0 GeV | 2.358 | 2.357 | -0.07% | 2.471 |
| W/E numu 0.5 GeV 87 deg | 2.768 | 2.813 | +1.64% | 2.507 |
| W/E numu 0.5 GeV 81 deg | 2.668 | 2.710 | +1.55% | 2.441 |
| W/E numu 0.5 GeV 75 deg | 2.490 | 2.525 | +1.39% | 2.285 |
| W/E numu 1.0 GeV 87 deg | 2.488 | 2.488 | -0.03% | 2.090 |
| W/E numu 1.0 GeV 81 deg | 2.363 | 2.361 | -0.09% | 2.024 |
| W/E numu 1.0 GeV 75 deg | 2.183 | 2.178 | -0.21% | 1.919 |
| W/E nue 0.5 GeV 87 deg | 3.098 | 3.176 | +2.51% | 4.740 |
| W/E nue 0.5 GeV 81 deg | 2.956 | 3.026 | +2.37% | 4.302 |
| W/E nue 0.5 GeV 75 deg | 2.733 | 2.792 | +2.15% | 3.488 |
| W/E nue 1.0 GeV 87 deg | 2.911 | 2.935 | +0.86% | 4.895 |
| W/E nue 1.0 GeV 81 deg | 2.729 | 2.749 | +0.74% | 3.887 |
| W/E nue 1.0 GeV 75 deg | 2.492 | 2.506 | +0.57% | 2.990 |
| numu-anti 0.3 GeV D(dphi) deg | +4.576 | +4.565 | -0.011 | +5.953 |
| numu-anti 0.3 GeV D(max/min) | -0.088 | -0.095 | -0.007 | -0.906 |
| numu-anti 0.5 GeV D(dphi) deg | +4.313 | +4.314 | +0.001 | +4.355 |
| numu-anti 0.5 GeV D(max/min) | -0.156 | -0.164 | -0.008 | -1.305 |
| nue-anti 0.3 GeV D(dphi) deg | -9.119 | -9.032 | +0.087 | -12.218 |
| nue-anti 0.3 GeV D(max/min) | +0.328 | +0.361 | +0.033 | +1.694 |
| nue-anti 0.5 GeV D(dphi) deg | -8.533 | -8.449 | +0.084 | -9.209 |
| nue-anti 0.5 GeV D(max/min) | +0.487 | +0.527 | +0.040 | +2.623 |
| N numu model/Honda 0.5 GeV | 0.963 | 1.026 | +6.57% | 1.000 |
| S numu model/Honda 0.5 GeV | 0.945 | 1.011 | +6.98% | 1.000 |
| phase offset numu deg | +1.28 | +1.07 | -0.209 | +0.00 |
| N antinumu model/Honda 0.5 GeV | 0.994 | 1.058 | +6.46% | 1.000 |
| S antinumu model/Honda 0.5 GeV | 0.933 | 0.998 | +6.98% | 1.000 |
| phase offset antinumu deg | +1.32 | +1.11 | -0.210 | +0.00 |
| N nue model/Honda 0.5 GeV | 1.065 | 1.139 | +6.90% | 1.000 |
| S nue model/Honda 0.5 GeV | 0.983 | 1.057 | +7.48% | 1.000 |
| phase offset nue deg | +0.45 | +0.32 | -0.135 | +0.00 |
| N antinue model/Honda 0.5 GeV | 0.993 | 1.059 | +6.60% | 1.000 |
| S antinue model/Honda 0.5 GeV | 1.005 | 1.074 | +6.91% | 1.000 |
| phase offset antinue deg | -0.22 | -0.44 | -0.220 | +0.00 |

**B. G_s(E, R_c), cos Z_ref = 1, before | after (ratio after/before - 1)**

| primary, species, R_c | 0.1 GeV | 0.3 GeV | 1 GeV | 3 GeV | 10 GeV |
|---|---|---|---|---|---|
| GSF numu 7 GV | 0.5381 -> 0.6161 (+14.5%) | 0.7118 -> 0.7739 (+8.7%) | 0.9018 -> 0.9319 (+3.3%) | 0.9892 -> 0.9939 (+0.5%) | 1.0000 -> 1.0000 (+0.0%) |
| GSF numu 11 GV | 0.3593 -> 0.3989 (+11.0%) | 0.5220 -> 0.5665 (+8.5%) | 0.7664 -> 0.8020 (+4.6%) | 0.9567 -> 0.9672 (+1.1%) | 0.9999 -> 1.0000 (+0.0%) |
| GSF numu 40 GV | 0.0815 -> 0.0854 (+4.8%) | 0.1410 -> 0.1471 (+4.3%) | 0.2936 -> 0.3032 (+3.3%) | 0.6044 -> 0.6150 (+1.7%) | 0.9336 -> 0.9370 (+0.4%) |
| GSF nue 7 GV | 0.5417 -> 0.6168 (+13.9%) | 0.6705 -> 0.7348 (+9.6%) | 0.8665 -> 0.9036 (+4.3%) | 0.9853 -> 0.9921 (+0.7%) | 1.0000 -> 1.0000 (-0.0%) |
| GSF nue 11 GV | 0.3634 -> 0.4022 (+10.7%) | 0.4822 -> 0.5246 (+8.8%) | 0.7127 -> 0.7509 (+5.4%) | 0.9359 -> 0.9512 (+1.6%) | 1.0000 -> 1.0000 (+0.0%) |
| GSF nue 40 GV | 0.0826 -> 0.0866 (+4.8%) | 0.1237 -> 0.1292 (+4.5%) | 0.2480 -> 0.2568 (+3.5%) | 0.5303 -> 0.5413 (+2.1%) | 0.9217 -> 0.9260 (+0.5%) |
| H3a numu 7 GV | 0.4843 -> 0.5553 (+14.7%) | 0.6800 -> 0.7398 (+8.8%) | 0.8975 -> 0.9271 (+3.3%) | 0.9899 -> 0.9942 (+0.4%) | 1.0000 -> 1.0000 (+0.0%) |
| H3a numu 11 GV | 0.3263 -> 0.3602 (+10.4%) | 0.5037 -> 0.5438 (+8.0%) | 0.7705 -> 0.8029 (+4.2%) | 0.9616 -> 0.9705 (+0.9%) | 0.9999 -> 1.0000 (+0.0%) |
| H3a numu 40 GV | 0.0853 -> 0.0889 (+4.2%) | 0.1559 -> 0.1616 (+3.7%) | 0.3322 -> 0.3414 (+2.8%) | 0.6546 -> 0.6640 (+1.4%) | 0.9485 -> 0.9512 (+0.3%) |
| H3a nue 7 GV | 0.4872 -> 0.5557 (+14.1%) | 0.6315 -> 0.6928 (+9.7%) | 0.8586 -> 0.8953 (+4.3%) | 0.9858 -> 0.9922 (+0.6%) | 1.0000 -> 1.0000 (-0.0%) |
| H3a nue 11 GV | 0.3296 -> 0.3629 (+10.1%) | 0.4583 -> 0.4962 (+8.3%) | 0.7132 -> 0.7483 (+4.9%) | 0.9418 -> 0.9552 (+1.4%) | 1.0000 -> 1.0000 (+0.0%) |
| H3a nue 40 GV | 0.0863 -> 0.0899 (+4.1%) | 0.1347 -> 0.1399 (+3.8%) | 0.2800 -> 0.2885 (+3.0%) | 0.5803 -> 0.5903 (+1.7%) | 0.9376 -> 0.9411 (+0.4%) |

**C. primary scan, muon_bending = False; rows: observable of the 12-bin azimuth pattern (analyse_m3b definitions), cos Z = 0.05 (87 deg); columns: engine primary, Honda**

| species, E | observable | GSF | H3a | GH-native | GH-at-total | Honda |
|---|---|---|---|---|---|---|
| numu 0.3 | a1_rel | 0.461 | 0.436 | 0.454 | 0.416 | 0.412 |
| numu 0.3 | dphi | 9.861 | 10.067 | 10.019 | 9.508 | 10.347 |
| numu 0.3 | fN | 1.032 | 1.030 | 1.031 | 1.031 | 1.026 |
| numu 0.3 | fS | 1.041 | 1.036 | 1.037 | 1.043 | 1.025 |
| numu 0.3 | NS | 0.992 | 0.994 | 0.994 | 0.988 | 1.001 |
| numu 0.3 | WE | 2.756 | 2.566 | 2.691 | 2.477 | 2.402 |
| numu 0.3 | maxmin | 2.738 | 2.566 | 2.685 | 2.455 | 2.415 |
| numu 0.5 | a1_rel | 0.475 | 0.443 | 0.466 | 0.428 | 0.434 |
| numu 0.5 | dphi | 2.697 | 2.858 | 2.842 | 2.460 | 3.596 |
| numu 0.5 | fN | 0.955 | 0.958 | 0.955 | 0.964 | 0.951 |
| numu 0.5 | fS | 1.104 | 1.094 | 1.099 | 1.100 | 1.080 |
| numu 0.5 | NS | 0.865 | 0.875 | 0.869 | 0.876 | 0.881 |
| numu 0.5 | WE | 2.867 | 2.621 | 2.783 | 2.563 | 2.527 |
| numu 0.5 | maxmin | 2.795 | 2.579 | 2.730 | 2.501 | 2.507 |
| numu 1.0 | a1_rel | 0.433 | 0.393 | 0.422 | 0.389 | 0.357 |
| numu 1.0 | dphi | -4.392 | -4.283 | -4.288 | -4.510 | -1.935 |
| numu 1.0 | fN | 0.905 | 0.914 | 0.906 | 0.920 | 0.939 |
| numu 1.0 | fS | 1.156 | 1.140 | 1.150 | 1.144 | 1.115 |
| numu 1.0 | NS | 0.783 | 0.802 | 0.788 | 0.804 | 0.842 |
| numu 1.0 | WE | 2.570 | 2.323 | 2.491 | 2.315 | 2.160 |
| numu 1.0 | maxmin | 2.473 | 2.255 | 2.408 | 2.238 | 2.090 |
| antinumu 0.3 | a1_rel | 0.457 | 0.432 | 0.450 | 0.413 | 0.556 |
| antinumu 0.3 | dphi | 10.189 | 10.396 | 10.347 | 9.835 | 4.393 |
| antinumu 0.3 | fN | 1.036 | 1.034 | 1.035 | 1.035 | 0.943 |
| antinumu 0.3 | fS | 1.038 | 1.033 | 1.035 | 1.041 | 1.049 |
| antinumu 0.3 | NS | 0.998 | 1.001 | 1.000 | 0.994 | 0.899 |
| antinumu 0.3 | WE | 2.735 | 2.548 | 2.672 | 2.459 | 3.378 |
| antinumu 0.3 | maxmin | 2.717 | 2.548 | 2.666 | 2.438 | 3.322 |
| antinumu 0.5 | a1_rel | 0.475 | 0.442 | 0.466 | 0.428 | 0.625 |
| antinumu 0.5 | dphi | 2.801 | 2.961 | 2.941 | 2.559 | -0.759 |
| antinumu 0.5 | fN | 0.958 | 0.960 | 0.957 | 0.966 | 0.864 |
| antinumu 0.5 | fS | 1.104 | 1.094 | 1.099 | 1.100 | 1.102 |
| antinumu 0.5 | NS | 0.868 | 0.878 | 0.871 | 0.878 | 0.785 |
| antinumu 0.5 | WE | 2.871 | 2.623 | 2.788 | 2.565 | 4.266 |
| antinumu 0.5 | maxmin | 2.800 | 2.580 | 2.733 | 2.503 | 3.812 |
| antinumu 1.0 | a1_rel | 0.438 | 0.397 | 0.427 | 0.393 | 0.609 |
| antinumu 1.0 | dphi | -4.621 | -4.509 | -4.517 | -4.734 | -3.630 |
| antinumu 1.0 | fN | 0.899 | 0.909 | 0.900 | 0.914 | 0.833 |
| antinumu 1.0 | fS | 1.157 | 1.141 | 1.151 | 1.145 | 1.129 |
| antinumu 1.0 | NS | 0.777 | 0.796 | 0.782 | 0.798 | 0.738 |
| antinumu 1.0 | WE | 2.585 | 2.334 | 2.505 | 2.328 | 4.023 |
| antinumu 1.0 | maxmin | 2.495 | 2.271 | 2.428 | 2.255 | 3.619 |
| nue 0.3 | a1_rel | 0.470 | 0.446 | 0.464 | 0.426 | 0.619 |
| nue 0.3 | dphi | 11.198 | 11.417 | 11.358 | 10.827 | 2.771 |
| nue 0.3 | fN | 1.045 | 1.042 | 1.044 | 1.042 | 0.903 |
| nue 0.3 | fS | 1.029 | 1.024 | 1.026 | 1.033 | 1.055 |
| nue 0.3 | NS | 1.016 | 1.017 | 1.017 | 1.009 | 0.856 |
| nue 0.3 | WE | 2.804 | 2.615 | 2.742 | 2.524 | 4.363 |
| nue 0.3 | maxmin | 2.807 | 2.635 | 2.757 | 2.516 | 3.733 |
| nue 0.5 | a1_rel | 0.494 | 0.462 | 0.486 | 0.447 | 0.696 |
| nue 0.5 | dphi | 4.024 | 4.198 | 4.171 | 3.766 | -0.624 |
| nue 0.5 | fN | 0.964 | 0.966 | 0.964 | 0.972 | 0.816 |
| nue 0.5 | fS | 1.094 | 1.085 | 1.090 | 1.092 | 1.079 |
| nue 0.5 | NS | 0.881 | 0.890 | 0.884 | 0.890 | 0.756 |
| nue 0.5 | WE | 2.989 | 2.733 | 2.906 | 2.673 | 6.044 |
| nue 0.5 | maxmin | 2.941 | 2.712 | 2.876 | 2.624 | 4.740 |
| nue 1.0 | a1_rel | 0.472 | 0.431 | 0.461 | 0.426 | 0.714 |
| nue 1.0 | dphi | -3.856 | -3.734 | -3.749 | -3.992 | -3.970 |
| nue 1.0 | fN | 0.889 | 0.898 | 0.890 | 0.905 | 0.784 |
| nue 1.0 | fS | 1.158 | 1.142 | 1.152 | 1.147 | 1.145 |
| nue 1.0 | NS | 0.767 | 0.786 | 0.773 | 0.789 | 0.684 |
| nue 1.0 | WE | 2.787 | 2.511 | 2.701 | 2.502 | 6.333 |
| nue 1.0 | maxmin | 2.701 | 2.454 | 2.630 | 2.428 | 4.895 |
| antinue 0.3 | a1_rel | 0.447 | 0.422 | 0.439 | 0.401 | 0.327 |
| antinue 0.3 | dphi | 10.742 | 10.946 | 10.883 | 10.348 | 14.989 |
| antinue 0.3 | fN | 1.044 | 1.041 | 1.043 | 1.042 | 1.062 |
| antinue 0.3 | fS | 1.035 | 1.030 | 1.032 | 1.038 | 1.017 |
| antinue 0.3 | NS | 1.009 | 1.011 | 1.011 | 1.004 | 1.045 |
| antinue 0.3 | WE | 2.675 | 2.494 | 2.610 | 2.398 | 1.980 |
| antinue 0.3 | maxmin | 2.655 | 2.491 | 2.600 | 2.376 | 2.039 |
| antinue 0.5 | a1_rel | 0.459 | 0.427 | 0.449 | 0.411 | 0.331 |
| antinue 0.5 | dphi | 3.471 | 3.628 | 3.601 | 3.210 | 8.585 |
| antinue 0.5 | fN | 0.972 | 0.973 | 0.972 | 0.979 | 1.022 |
| antinue 0.5 | fS | 1.099 | 1.089 | 1.094 | 1.095 | 1.059 |
| antinue 0.5 | NS | 0.884 | 0.893 | 0.888 | 0.894 | 0.966 |
| antinue 0.5 | WE | 2.786 | 2.548 | 2.700 | 2.479 | 2.011 |
| antinue 0.5 | maxmin | 2.707 | 2.497 | 2.637 | 2.414 | 2.117 |
| antinue 1.0 | a1_rel | 0.422 | 0.382 | 0.409 | 0.375 | 0.220 |
| antinue 1.0 | dphi | -4.371 | -4.263 | -4.271 | -4.477 | -4.415 |
| antinue 1.0 | fN | 0.910 | 0.919 | 0.911 | 0.925 | 0.979 |
| antinue 1.0 | fS | 1.154 | 1.137 | 1.147 | 1.141 | 1.124 |
| antinue 1.0 | NS | 0.788 | 0.808 | 0.795 | 0.811 | 0.871 |
| antinue 1.0 | WE | 2.506 | 2.263 | 2.421 | 2.246 | 1.609 |
| antinue 1.0 | maxmin | 2.409 | 2.195 | 2.338 | 2.172 | 1.632 |

**D. W/E, Phase-1 definition (engine flux at az 270 / az 90, cos Z = 0.05) vs Honda max/min, muon_bending = False**

| species, E | GSF | H3a | GH-native | GH-at-total | Honda |
|---|---|---|---|---|---|
| numu 0.3 | 2.800 (+16%) | 2.611 (+8%) | 2.739 (+13%) | 2.507 (+4%) | 2.415 |
| numu 0.5 | 2.873 (+15%) | 2.636 (+5%) | 2.798 (+12%) | 2.564 (+2%) | 2.507 |
| numu 1.0 | 2.536 (+21%) | 2.303 (+10%) | 2.465 (+18%) | 2.290 (+10%) | 2.090 |
| nue 0.3 | 2.865 (-23%) | 2.675 (-28%) | 2.807 (-25%) | 2.565 (-31%) | 3.733 |
| nue 0.5 | 3.018 (-36%) | 2.768 (-42%) | 2.943 (-38%) | 2.687 (-43%) | 4.740 |
| nue 1.0 | 2.771 (-43%) | 2.506 (-49%) | 2.693 (-45%) | 2.485 (-49%) | 4.895 |

**C. primary scan, muon_bending = True; rows: observable of the 12-bin azimuth pattern (analyse_m3b definitions), cos Z = 0.05 (87 deg); columns: engine primary, Honda**

| species, E | observable | GSF | H3a | GH-native | GH-at-total | Honda |
|---|---|---|---|---|---|---|
| numu 0.3 | a1_rel | 0.456 | 0.431 | 0.449 | 0.411 | 0.412 |
| numu 0.3 | dphi | 12.056 | 12.262 | 12.212 | 11.714 | 10.347 |
| numu 0.3 | fN | 1.048 | 1.045 | 1.046 | 1.045 | 1.026 |
| numu 0.3 | fS | 1.021 | 1.017 | 1.019 | 1.026 | 1.025 |
| numu 0.3 | NS | 1.026 | 1.027 | 1.027 | 1.019 | 1.001 |
| numu 0.3 | WE | 2.711 | 2.529 | 2.650 | 2.438 | 2.402 |
| numu 0.3 | maxmin | 2.714 | 2.547 | 2.663 | 2.436 | 2.415 |
| numu 0.5 | a1_rel | 0.467 | 0.435 | 0.458 | 0.421 | 0.434 |
| numu 0.5 | dphi | 4.667 | 4.828 | 4.810 | 4.437 | 3.596 |
| numu 0.5 | fN | 0.971 | 0.972 | 0.970 | 0.977 | 0.951 |
| numu 0.5 | fS | 1.086 | 1.077 | 1.081 | 1.083 | 1.080 |
| numu 0.5 | NS | 0.894 | 0.903 | 0.898 | 0.902 | 0.881 |
| numu 0.5 | WE | 2.821 | 2.583 | 2.741 | 2.523 | 2.527 |
| numu 0.5 | maxmin | 2.748 | 2.539 | 2.686 | 2.462 | 2.507 |
| numu 1.0 | a1_rel | 0.424 | 0.385 | 0.413 | 0.380 | 0.357 |
| numu 1.0 | dphi | -2.844 | -2.734 | -2.742 | -2.953 | -1.935 |
| numu 1.0 | fN | 0.917 | 0.925 | 0.918 | 0.930 | 0.939 |
| numu 1.0 | fS | 1.142 | 1.127 | 1.136 | 1.131 | 1.115 |
| numu 1.0 | NS | 0.803 | 0.820 | 0.808 | 0.822 | 0.842 |
| numu 1.0 | WE | 2.534 | 2.294 | 2.458 | 2.284 | 2.160 |
| numu 1.0 | maxmin | 2.408 | 2.203 | 2.348 | 2.185 | 2.090 |
| antinumu 0.3 | a1_rel | 0.464 | 0.439 | 0.457 | 0.419 | 0.556 |
| antinumu 0.3 | dphi | 7.491 | 7.695 | 7.649 | 7.122 | 4.393 |
| antinumu 0.3 | fN | 1.016 | 1.014 | 1.015 | 1.016 | 0.943 |
| antinumu 0.3 | fS | 1.061 | 1.055 | 1.057 | 1.061 | 1.049 |
| antinumu 0.3 | NS | 0.957 | 0.962 | 0.960 | 0.958 | 0.899 |
| antinumu 0.3 | WE | 2.780 | 2.587 | 2.714 | 2.500 | 3.378 |
| antinumu 0.3 | maxmin | 2.809 | 2.625 | 2.752 | 2.513 | 3.322 |
| antinumu 0.5 | a1_rel | 0.486 | 0.452 | 0.476 | 0.438 | 0.625 |
| antinumu 0.5 | dphi | 0.353 | 0.510 | 0.493 | 0.098 | -0.759 |
| antinumu 0.5 | fN | 0.937 | 0.941 | 0.937 | 0.948 | 0.864 |
| antinumu 0.5 | fS | 1.126 | 1.115 | 1.121 | 1.120 | 1.102 |
| antinumu 0.5 | NS | 0.832 | 0.844 | 0.836 | 0.846 | 0.785 |
| antinumu 0.5 | WE | 2.921 | 2.665 | 2.834 | 2.609 | 4.266 |
| antinumu 0.5 | maxmin | 2.912 | 2.671 | 2.838 | 2.594 | 3.812 |
| antinumu 1.0 | a1_rel | 0.450 | 0.408 | 0.438 | 0.404 | 0.609 |
| antinumu 1.0 | dphi | -6.634 | -6.524 | -6.527 | -6.761 | -3.630 |
| antinumu 1.0 | fN | 0.883 | 0.894 | 0.885 | 0.900 | 0.833 |
| antinumu 1.0 | fS | 1.176 | 1.158 | 1.169 | 1.162 | 1.129 |
| antinumu 1.0 | NS | 0.751 | 0.772 | 0.757 | 0.775 | 0.738 |
| antinumu 1.0 | WE | 2.628 | 2.369 | 2.545 | 2.365 | 4.023 |
| antinumu 1.0 | maxmin | 2.595 | 2.350 | 2.521 | 2.337 | 3.619 |
| nue 0.3 | a1_rel | 0.482 | 0.457 | 0.475 | 0.437 | 0.619 |
| nue 0.3 | dphi | 6.527 | 6.741 | 6.690 | 6.122 | 2.771 |
| nue 0.3 | fN | 1.007 | 1.006 | 1.007 | 1.009 | 0.903 |
| nue 0.3 | fS | 1.069 | 1.062 | 1.066 | 1.070 | 1.055 |
| nue 0.3 | NS | 0.942 | 0.947 | 0.945 | 0.943 | 0.856 |
| nue 0.3 | WE | 2.881 | 2.682 | 2.815 | 2.594 | 4.363 |
| nue 0.3 | maxmin | 2.976 | 2.777 | 2.916 | 2.654 | 3.733 |
| nue 0.5 | a1_rel | 0.513 | 0.480 | 0.504 | 0.465 | 0.696 |
| nue 0.5 | dphi | -0.307 | -0.138 | -0.156 | -0.599 | -0.624 |
| nue 0.5 | fN | 0.926 | 0.930 | 0.926 | 0.937 | 0.816 |
| nue 0.5 | fS | 1.136 | 1.124 | 1.131 | 1.130 | 1.079 |
| nue 0.5 | NS | 0.815 | 0.827 | 0.819 | 0.829 | 0.756 |
| nue 0.5 | WE | 3.078 | 2.810 | 2.990 | 2.752 | 6.044 |
| nue 0.5 | maxmin | 3.158 | 2.889 | 3.078 | 2.801 | 4.740 |
| nue 1.0 | a1_rel | 0.496 | 0.453 | 0.484 | 0.448 | 0.714 |
| nue 1.0 | dphi | -7.666 | -7.549 | -7.553 | -7.836 | -3.970 |
| nue 1.0 | fN | 0.855 | 0.868 | 0.857 | 0.875 | 0.784 |
| nue 1.0 | fS | 1.196 | 1.178 | 1.189 | 1.182 | 1.145 |
| nue 1.0 | NS | 0.715 | 0.737 | 0.721 | 0.740 | 0.684 |
| nue 1.0 | WE | 2.874 | 2.585 | 2.783 | 2.579 | 6.333 |
| nue 1.0 | maxmin | 2.929 | 2.634 | 2.841 | 2.615 | 4.895 |
| antinue 0.3 | a1_rel | 0.438 | 0.414 | 0.430 | 0.392 | 0.327 |
| antinue 0.3 | dphi | 15.558 | 15.759 | 15.692 | 15.214 | 14.989 |
| antinue 0.3 | fN | 1.077 | 1.073 | 1.075 | 1.071 | 1.062 |
| antinue 0.3 | fS | 0.995 | 0.992 | 0.993 | 1.002 | 1.017 |
| antinue 0.3 | NS | 1.083 | 1.081 | 1.084 | 1.069 | 1.045 |
| antinue 0.3 | WE | 2.575 | 2.412 | 2.518 | 2.310 | 1.980 |
| antinue 0.3 | maxmin | 2.615 | 2.458 | 2.563 | 2.344 | 2.039 |
| antinue 0.5 | a1_rel | 0.442 | 0.412 | 0.433 | 0.395 | 0.331 |
| antinue 0.5 | dphi | 8.142 | 8.298 | 8.265 | 7.928 | 8.585 |
| antinue 0.5 | fN | 1.006 | 1.005 | 1.005 | 1.009 | 1.022 |
| antinue 0.5 | fS | 1.058 | 1.051 | 1.054 | 1.058 | 1.059 |
| antinue 0.5 | NS | 0.952 | 0.956 | 0.954 | 0.954 | 0.966 |
| antinue 0.5 | WE | 2.674 | 2.457 | 2.598 | 2.381 | 2.011 |
| antinue 0.5 | maxmin | 2.631 | 2.436 | 2.567 | 2.353 | 2.117 |
| antinue 1.0 | a1_rel | 0.399 | 0.361 | 0.387 | 0.354 | 0.220 |
| antinue 1.0 | dphi | -0.079 | 0.030 | 0.012 | -0.139 | -4.415 |
| antinue 1.0 | fN | 0.940 | 0.946 | 0.941 | 0.952 | 0.979 |
| antinue 1.0 | fS | 1.117 | 1.104 | 1.112 | 1.108 | 1.124 |
| antinue 1.0 | NS | 0.842 | 0.856 | 0.846 | 0.859 | 0.871 |
| antinue 1.0 | WE | 2.407 | 2.184 | 2.331 | 2.161 | 1.609 |
| antinue 1.0 | maxmin | 2.305 | 2.112 | 2.243 | 2.086 | 1.632 |

**D. W/E, Phase-1 definition (engine flux at az 270 / az 90, cos Z = 0.05) vs Honda max/min, muon_bending = True**

| species, E | GSF | H3a | GH-native | GH-at-total | Honda |
|---|---|---|---|---|---|
| numu 0.3 | 2.752 (+14%) | 2.571 (+6%) | 2.694 (+12%) | 2.467 (+2%) | 2.415 |
| numu 0.5 | 2.813 (+12%) | 2.587 (+3%) | 2.742 (+9%) | 2.516 (+0%) | 2.507 |
| numu 1.0 | 2.488 (+19%) | 2.265 (+8%) | 2.420 (+16%) | 2.250 (+8%) | 2.090 |
| nue 0.3 | 2.975 (-20%) | 2.767 (-26%) | 2.909 (-22%) | 2.655 (-29%) | 3.733 |
| nue 0.5 | 3.176 (-33%) | 2.896 (-39%) | 3.089 (-35%) | 2.815 (-41%) | 4.740 |
| nue 1.0 | 2.935 (-40%) | 2.636 (-46%) | 2.845 (-42%) | 2.619 (-46%) | 4.895 |

**E. cross-check cells for tools/mc3d/analyse_m3b: engine (rigidity fixed, H3a at total energy, muon_bending = False, cz_sub = 3, band-integrated like Honda; analyse_m3b.observables) | Honda**

| species | E band | cos Z band | a1_rel | dphi | WE | maxmin | fN | fS | NS |
|---|---|---|---|---|---|---|---|---|---|
| nue | 0.3 | 0-0.1 | 0.442 | 0.609 | 13.167 | 3.163 | 2.578 | 4.129 | 2.610 | 3.653 | 1.056 | 0.909 | 1.011 | 1.048 | 1.045 | 0.868 |
| nue | 0.3 | 0.1-0.2 | 0.434 | 0.588 | 3.194 | -2.396 | 2.599 | 3.731 | 2.589 | 3.631 | 0.986 | 0.869 | 1.086 | 1.107 | 0.908 | 0.785 |
| nue | 0.3 | 0.2-0.3 | 0.413 | 0.538 | 0.815 | -3.509 | 2.487 | 3.241 | 2.460 | 3.233 | 0.979 | 0.890 | 1.102 | 1.127 | 0.888 | 0.790 |
| nue | 0.5 | 0-0.1 | 0.460 | 0.700 | 5.438 | -0.607 | 2.714 | 6.111 | 2.694 | 4.757 | 0.977 | 0.816 | 1.075 | 1.081 | 0.909 | 0.755 |
| nue | 0.5 | 0.1-0.2 | 0.445 | 0.645 | -1.085 | -4.472 | 2.672 | 4.396 | 2.652 | 4.220 | 0.945 | 0.825 | 1.130 | 1.149 | 0.836 | 0.718 |
| nue | 0.5 | 0.2-0.3 | 0.416 | 0.562 | -1.511 | -4.556 | 2.522 | 3.502 | 2.498 | 3.469 | 0.963 | 0.879 | 1.132 | 1.153 | 0.850 | 0.762 |
| nue | 1.0 | 0-0.1 | 0.431 | 0.719 | -2.826 | -4.314 | 2.509 | 6.766 | 2.450 | 4.894 | 0.905 | 0.780 | 1.135 | 1.142 | 0.797 | 0.683 |
| nue | 1.0 | 0.1-0.2 | 0.404 | 0.610 | -6.455 | -6.712 | 2.382 | 3.882 | 2.364 | 3.894 | 0.905 | 0.827 | 1.162 | 1.179 | 0.779 | 0.701 |
| nue | 1.0 | 0.2-0.3 | 0.364 | 0.492 | -4.849 | -5.763 | 2.211 | 2.918 | 2.209 | 2.961 | 0.945 | 0.896 | 1.147 | 1.158 | 0.824 | 0.774 |
| antinue | 0.3 | 0-0.1 | 0.419 | 0.323 | 12.723 | 15.753 | 2.461 | 1.962 | 2.471 | 2.014 | 1.055 | 1.068 | 1.018 | 1.014 | 1.037 | 1.053 |
| antinue | 0.3 | 0.1-0.2 | 0.411 | 0.317 | 2.888 | 5.559 | 2.471 | 1.988 | 2.452 | 1.955 | 0.990 | 1.009 | 1.088 | 1.069 | 0.910 | 0.944 |
| antinue | 0.3 | 0.2-0.3 | 0.390 | 0.307 | 0.556 | 3.514 | 2.361 | 1.952 | 2.334 | 1.934 | 0.983 | 1.013 | 1.101 | 1.081 | 0.892 | 0.937 |
| antinue | 0.5 | 0-0.1 | 0.425 | 0.318 | 4.883 | 9.661 | 2.533 | 1.955 | 2.484 | 2.021 | 0.983 | 1.025 | 1.080 | 1.052 | 0.911 | 0.974 |
| antinue | 0.5 | 0.1-0.2 | 0.411 | 0.310 | -1.487 | 1.465 | 2.478 | 1.973 | 2.447 | 1.935 | 0.955 | 0.993 | 1.129 | 1.099 | 0.846 | 0.904 |
| antinue | 0.5 | 0.2-0.3 | 0.382 | 0.301 | -1.863 | 0.611 | 2.335 | 1.943 | 2.310 | 1.951 | 0.971 | 1.005 | 1.129 | 1.109 | 0.860 | 0.906 |
| antinue | 1.0 | 0-0.1 | 0.381 | 0.216 | -3.346 | -0.869 | 2.266 | 1.598 | 2.195 | 1.639 | 0.924 | 0.994 | 1.131 | 1.103 | 0.817 | 0.901 |
| antinue | 1.0 | 0.1-0.2 | 0.356 | 0.229 | -6.846 | -5.276 | 2.145 | 1.651 | 2.122 | 1.639 | 0.924 | 0.992 | 1.151 | 1.126 | 0.803 | 0.881 |
| antinue | 1.0 | 0.2-0.3 | 0.319 | 0.233 | -5.209 | -4.647 | 1.995 | 1.663 | 1.992 | 1.678 | 0.960 | 0.994 | 1.136 | 1.124 | 0.845 | 0.884 |
| numu | 0.3 | 0-0.1 | 0.432 | 0.411 | 11.873 | 10.697 | 2.532 | 2.391 | 2.546 | 2.403 | 1.045 | 1.028 | 1.022 | 1.025 | 1.022 | 1.003 |
| numu | 0.3 | 0.1-0.2 | 0.426 | 0.405 | 2.393 | 2.546 | 2.554 | 2.410 | 2.536 | 2.375 | 0.981 | 0.975 | 1.093 | 1.079 | 0.898 | 0.904 |
| numu | 0.3 | 0.2-0.3 | 0.405 | 0.381 | 0.329 | 0.927 | 2.448 | 2.297 | 2.420 | 2.280 | 0.977 | 0.978 | 1.107 | 1.092 | 0.883 | 0.896 |
| numu | 0.5 | 0-0.1 | 0.440 | 0.432 | 4.033 | 3.977 | 2.606 | 2.519 | 2.564 | 2.524 | 0.968 | 0.957 | 1.085 | 1.076 | 0.892 | 0.889 |
| numu | 0.5 | 0.1-0.2 | 0.428 | 0.411 | -2.160 | -0.955 | 2.565 | 2.471 | 2.538 | 2.418 | 0.939 | 0.952 | 1.136 | 1.117 | 0.827 | 0.852 |
| numu | 0.5 | 0.2-0.3 | 0.399 | 0.381 | -2.259 | -0.799 | 2.423 | 2.312 | 2.401 | 2.306 | 0.960 | 0.970 | 1.135 | 1.115 | 0.846 | 0.870 |
| numu | 1.0 | 0-0.1 | 0.393 | 0.358 | -3.402 | -1.718 | 2.325 | 2.159 | 2.254 | 2.113 | 0.920 | 0.941 | 1.134 | 1.112 | 0.811 | 0.847 |
| numu | 1.0 | 0.1-0.2 | 0.369 | 0.330 | -6.963 | -5.993 | 2.208 | 2.038 | 2.185 | 2.003 | 0.917 | 0.942 | 1.156 | 1.135 | 0.794 | 0.830 |
| numu | 1.0 | 0.2-0.3 | 0.331 | 0.304 | -5.262 | -4.500 | 2.050 | 1.936 | 2.048 | 1.940 | 0.954 | 0.968 | 1.140 | 1.129 | 0.837 | 0.857 |
| antinumu | 0.3 | 0-0.1 | 0.429 | 0.547 | 12.200 | 5.058 | 2.514 | 3.275 | 2.528 | 3.211 | 1.048 | 0.952 | 1.020 | 1.048 | 1.028 | 0.908 |
| antinumu | 0.3 | 0.1-0.2 | 0.422 | 0.529 | 2.573 | -0.923 | 2.534 | 3.152 | 2.517 | 3.154 | 0.984 | 0.908 | 1.091 | 1.096 | 0.902 | 0.829 |
| antinumu | 0.3 | 0.2-0.3 | 0.402 | 0.487 | 0.413 | -2.252 | 2.428 | 2.895 | 2.400 | 2.876 | 0.979 | 0.919 | 1.105 | 1.112 | 0.886 | 0.827 |
| antinumu | 0.5 | 0-0.1 | 0.440 | 0.620 | 4.165 | 0.445 | 2.607 | 4.168 | 2.565 | 3.734 | 0.970 | 0.876 | 1.085 | 1.089 | 0.895 | 0.805 |
| antinumu | 0.5 | 0.1-0.2 | 0.426 | 0.569 | -2.007 | -3.752 | 2.559 | 3.500 | 2.531 | 3.487 | 0.942 | 0.869 | 1.135 | 1.141 | 0.830 | 0.761 |
| antinumu | 0.5 | 0.2-0.3 | 0.397 | 0.500 | -2.156 | -3.840 | 2.413 | 3.016 | 2.389 | 3.025 | 0.962 | 0.909 | 1.134 | 1.142 | 0.849 | 0.795 |
| antinumu | 1.0 | 0-0.1 | 0.397 | 0.609 | -3.630 | -4.218 | 2.335 | 4.086 | 2.270 | 3.597 | 0.914 | 0.826 | 1.135 | 1.139 | 0.806 | 0.725 |
| antinumu | 1.0 | 0.1-0.2 | 0.371 | 0.514 | -7.038 | -6.636 | 2.210 | 3.041 | 2.190 | 2.988 | 0.915 | 0.860 | 1.156 | 1.168 | 0.792 | 0.737 |
| antinumu | 1.0 | 0.2-0.3 | 0.331 | 0.414 | -5.269 | -6.442 | 2.046 | 2.441 | 2.046 | 2.470 | 0.953 | 0.918 | 1.139 | 1.152 | 0.836 | 0.797 |

## 6. Reproducing

```bash
source /home/pigrange/mc3d_work/runs/dbv142/env.sh ; cd $R/tools/mceq3d
# G_s families (19 (primary, cos Z) pairs, 8 chunks each; ~27 core-h); then assemble each
$PY diag_rigidity_scan.py chunk gsf 0.05 0 8 ; ... ; $PY diag_rigidity_scan.py assemble gsf 0.05
$PY diag_db_switch.py engine v142p --cache-dir $W/runs/dbv142/cache_old   # "after" headline (out/before/ holds "before")
for p in gsf h3a ghn ght; do for mb in 0 1; do $PY diag_rigidity_scan.py scan $p $mb; done; done
$PY diag_rigidity_scan.py m3b h3a 0 ; $PY diag_rigidity_scan.py tables
```

(`chunk`/`assemble` only parallelise `geomag_response`; with enough time
`diag_rigidity_scan.py scan` builds the same G_s files by itself.) The cutoff maps for the
`cz_sub = 3` grid and for the W/E pair were not cached and were built once (108 + 2 cells,
8-core, ~1 h); they are in `$W/mceq3d_cache` as ordinary `rc_*` files.

## 7. Open questions

1. N/S level: after the fix the engine is above Honda at 0.5 GeV in N/S (nu_e +6..+14%). Whether
   the hybrid's sub-1.7 GeV MCEq half (GSF) or the cone/E_off part is responsible was not
   investigated.
2. `MCEQ3D_DB=v140` runs now look for `gs_*` under the `rigp1` key and would write them into
   `.cache3d` (AFS); route them to `$MCEQ3D_CACHE` if that is unwanted.
3. Whether to replace the tracked `offaxis_excess*.npz` by the v142p rebuild (changes < 0.1%).
4. The MC response table that will replace the engine's muon-bend term is still pending; the
   `muon_bending = True` columns are given only as requested.
5. Solar modulation (`_modulate_phi0`) and `offaxis_mc`'s `rc_cut_gv` check use kinetic-grid
   conventions of their own (section 2); unchanged.

## 8. Follow-up (2026-11-04, later)

### 8.1 Engine cells for the MC pilot (GSF, GH-native; muon_bending off and on)

Same recipe as section 3 (E): `diag_rigidity_scan.py m3b <gsf|ghn> <0|1>`; hybrid base,
`cz_sub = 3`, band-integrated, rigidity fix, v142p. Files in `/home/pigrange/mc3d_work/runs/rigfix/out/`:

| file | engine |
|---|---|
| `m3b_engine_gsf_mb0.npz`, `m3b_engine_gsf_mb1.npz` | GSF, `muon_bending` False / True |
| `m3b_engine_ghnative_mb0.npz`, `m3b_engine_ghnative_mb1.npz` | `GaisserHondaNative`, False / True |
| `m3b_engine_h3a_mb0.npz` | H3a (section E) |
| `m3b_obs_<same>.npz` | the observables of the corresponding engine file |

Keys of `m3b_engine_*.npz` (the format `analyse_m3b.engine_patterns(path)` reads):
`e` (121,) engine energy grid [GeV, kinetic neutrino energy]; `az` (12,) azimuth bin centres
[deg, compass: 0 = N, 90 = E; 15 + 30 j]; `total_numu`, `total_antinumu`, `total_nue`,
`total_antinue`, each (3, 12, 121) = flux[cos Z band, azimuth, E] in /(m^2 s sr GeV) with the
cos Z bands ordered **[0-0.1, 0.1-0.2, 0.2-0.3]** (the engine's own order 0.25/0.15/0.05
is reversed on writing). Plug-in: `P = analyse_m3b.engine_patterns(path)` gives (species in
`SP_NAMES` = nue, antinue, numu, antinumu; E band 0.3/0.5/1.0; cos Z band; az 12).
Keys of `m3b_obs_*.npz`: `engine` and `honda`, each (4 species in `species`
order nue/antinue/numu/antinumu, 3 E bands 0.3/0.5/1.0, 3 cos Z bands, 10 observables in
the order of `obs` = a1_rel, dphi, a2_rel, WE, maxmin, fN, fS, NS, c1_rel, s1_rel), i.e.
`analyse_m3b.table_obs(...)` with the geomagnetic axis of `analyse_m3b.ew_axis()`; `honda` is
`table_obs(honda_patterns())`. Example (nu_mu, 0.5 GeV, cos Z 0-0.1; W/E, Honda 2.519):
GSF 2.848 (off) / 2.801 (on); GH-native 2.766 / 2.723; H3a 2.606 (off).

### 8.2 Cache routing

`mceq3d_db.dep_path(requested, name)`: reads an existing `requested/name` only for the legacy
(shipped v1.4.0) database (old `.cache3d` entries stay valid), and otherwise always returns a
path in `$MCEQ3D_CACHE`. Used for `gs_*` (`geomag_response`) and `jointprod_chan_*`; the
channel fractions use `channel_fractions_cache()` (tracked file if it carries the selected
database, else `$MCEQ3D_CACHE/channel_fractions<db tag>.npz`, `_dbv140` for the shipped file).
Geomagnetic maps (`finerc_*` ...) are database-independent and unchanged. Test:
`test_legacy_database_never_writes_new_caches_into_the_given_directory`.

### 8.3 Static tables replaced by the v142p rebuild

The tracked `offaxis_excess.npz`, `_v2`, `_channel`, `_channel_v2`, `_rebuild_old` and
`channel_fractions.npz` are now rebuilt with `v142p` and store `db` (cache tag),
`db_label` and `db_sha256` (`7de8275f...`); `offaxis_mc.py --build/--build-channel` writes the
same keys, `check_static_table` also compares the sha256, and the two warnings are gone for the
default database (they now fire for the others). Changes with respect to the v1.4.0 tables:
`E_off`, `E_off_s` <= 0.10% (0.1-10 GeV, all five files); `channel_fractions` (not a
ratio, an absolute fraction f_pi, f_K, f_mu) <= 0.021 absolute at 0.1-10 GeV (largest at
zenith 90 deg, 9 GeV) and <= 0.037 at 10-100 GeV: these come mostly from the v1.4.2
changes, not the hole, and only feed the legacy blend (`joint_channels=False`) and the cone
fractions. The v1.4.0 versions are in `/home/pigrange/mc3d_work/runs/rigfix/eoff/v140/` (copy back to
reproduce the old tables; with `MCEQ3D_DB=v140` the code then finds the file matching that
database; with the replaced files `MCEQ3D_DB=v140` builds its own
`channel_fractions_dbv140.npz` in `$MCEQ3D_CACHE`). Rebuild: `build_channel` 171 s with 8 jobs,
`build()` 375 s, channel fractions 12 zeniths ~40 min single core.
`test_static_tables_carry_the_default_database` guards it.

### 8.4 Level offset (record, not a fix)

GSF, hybrid base, delivered configuration, Honda's 12-bin azimuth average at the same cos Z
band and E (linear mean over the 12 bins). Column 4 divides the engine's **1D, no-cutoff**
base by Honda's azimuth average; because Honda's own average contains its geomagnetic
cutoff, that factor is not a 1D disagreement (it is 1.2-2.1 because the cutoff removes
10-50% at these energies); the column that carries the level information is the product
(last), and column 5 (engine az-average over its own 1D base) is the engine's
cutoff x cone x E_off factor.

| species | cos Z | E [GeV] | 1D base / Honda az-avg | engine az-avg / 1D base | engine az-avg / Honda az-avg |
|---|---|---|---|---|---|
| numu | 0.25 | 0.3 | 1.858 | 0.532 | 0.989 |
| numu | 0.25 | 0.5 | 1.676 | 0.596 | 0.999 |
| numu | 0.25 | 1.0 | 1.490 | 0.703 | 1.047 |
| numu | 0.15 | 0.3 | 1.592 | 0.616 | 0.981 |
| numu | 0.15 | 0.5 | 1.548 | 0.639 | 0.989 |
| numu | 0.15 | 1.0 | 1.470 | 0.705 | 1.037 |
| numu | 0.05 | 0.3 | 1.197 | 0.843 | 1.009 |
| numu | 0.05 | 0.5 | 1.282 | 0.784 | 1.005 |
| numu | 0.05 | 1.0 | 1.356 | 0.765 | 1.038 |
| antinumu | 0.25 | 0.3 | 1.868 | 0.521 | 0.973 |
| antinumu | 0.25 | 0.5 | 1.678 | 0.581 | 0.975 |
| antinumu | 0.25 | 1.0 | 1.481 | 0.689 | 1.020 |
| antinumu | 0.15 | 0.3 | 1.596 | 0.606 | 0.967 |
| antinumu | 0.15 | 0.5 | 1.539 | 0.623 | 0.959 |
| antinumu | 0.15 | 1.0 | 1.454 | 0.687 | 0.999 |
| antinumu | 0.05 | 0.3 | 1.203 | 0.834 | 1.004 |
| antinumu | 0.05 | 0.5 | 1.275 | 0.766 | 0.976 |
| antinumu | 0.05 | 1.0 | 1.314 | 0.740 | 0.972 |
| nue | 0.25 | 0.3 | 2.116 | 0.486 | 1.028 |
| nue | 0.25 | 0.5 | 1.836 | 0.546 | 1.003 |
| nue | 0.25 | 1.0 | 1.588 | 0.647 | 1.027 |
| nue | 0.15 | 0.3 | 1.798 | 0.569 | 1.023 |
| nue | 0.15 | 0.5 | 1.673 | 0.591 | 0.990 |
| nue | 0.15 | 1.0 | 1.546 | 0.647 | 1.001 |
| nue | 0.05 | 0.3 | 1.350 | 0.796 | 1.074 |
| nue | 0.05 | 0.5 | 1.354 | 0.741 | 1.004 |
| nue | 0.05 | 1.0 | 1.380 | 0.706 | 0.975 |
| antinue | 0.25 | 0.3 | 1.911 | 0.552 | 1.055 |
| antinue | 0.25 | 0.5 | 1.675 | 0.615 | 1.030 |
| antinue | 0.25 | 1.0 | 1.460 | 0.714 | 1.043 |
| antinue | 0.15 | 0.3 | 1.655 | 0.645 | 1.067 |
| antinue | 0.15 | 0.5 | 1.553 | 0.665 | 1.032 |
| antinue | 0.15 | 1.0 | 1.461 | 0.715 | 1.045 |
| antinue | 0.05 | 0.3 | 1.240 | 0.900 | 1.116 |
| antinue | 0.05 | 0.5 | 1.292 | 0.832 | 1.075 |
| antinue | 0.05 | 1.0 | 1.336 | 0.780 | 1.042 |

Reading: engine az-average / Honda az-average is 0.96-1.12 over the 36 cells (nu_mu
0.98-1.05, nubar_mu 0.96-1.02, nu_e 0.98-1.07, nubar_e 1.03-1.12; the largest, 1.116, is nubar_e
at 0.3 GeV, 87 deg). So the level offset is a genuine shift of the *average*, not of the
N-S contrast, and it grows toward low energy and the horizon for nubar_e. Whether it comes
from the MCEq half of the base or from the cone / E_off factor cannot be separated with this
table: `honda_kam.npz` has no no-cutoff reference, so the "1D part" (column 4) cannot be
compared with Honda's own 1D flux.

### 8.5 Tests and CPU of the follow-up

Full `pytest -q` (default `v142p`): `282 passed, 2 skipped, 4 warnings in 1951.74s (0:32:31)` (log `runs/rigfix/logs/pytest_rig3.log`). New
tests: `test_legacy_database_never_writes_new_caches_into_the_given_directory`,
`test_static_tables_carry_the_default_database` (and the channel-fractions test was
rewritten). `test_joint_cone.py` and `test_mceq3d_flux.py` took their production profile
from the legacy `.cache3d` file; with tables now built on v142p they use the profile of the
active database (`mceq3d_db.dep_path`), and their machine-precision gates (`G = 1` limit
equals the table, rtol 1e-9) pass against the rebuilt tables. CPU: G_s chunks for 12 more
(primary, cos Z) pairs 96 chunks about 15 core-h of process time, four `m3b` engine runs about 1
core-h, tables 0.7 core-h, level offset 0.2 core-h, test suite 0.5 h.

