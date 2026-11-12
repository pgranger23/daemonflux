# `tools/mc3d` run mcup -- the up-going (second-root) scorer and the Kamioka up-going near-horizon cells (2026-11-12)

MILESTONE-style record for `tools/mc3d` (draft; the PM merges). Run directory `/home/pigrange/mc3d_work/runs/mcup`
(`BRIEF.md`, `PREREG.md`, `HANDBACK.md`, `scripts/`, `logs/`, `out/`). Labels [m] measured, [d] derived, [e] estimate.

## 1. What was added (opt-in, default off)

* `scoring.NestedCaps.crossing2` / `locate2`: the second (far) root of a straight ray with the ground sphere -- the exit
  point -- with cap index, local cos zenith (< 0, up-going) and azimuth of the arrival direction in the exit point's frame.
  It exists only for an origin above the sphere and an entering ray (two positive roots). `locate` now calls the shared
  `_locate_at` (arithmetic unchanged).
* `sky3d.locate_vec2` (vectorised), `soft_hits_up` (exact second-root test for rigidly rotated neutrinos),
  `cone_may_hit_cap_up` (conservative muon test: below-horizon reach + disc around the site direction; `beta = pi` when the
  emission point's horizon circle reaches the cap).
* `sky3d.Options(upgoing=True)` / `--upgoing`: every 3D entry is also scored at the second root into the new tallies
  `us` (the `s` weight, i.e. with the copy's cutoff verdict) and `un` (copy 0, no cutoff, like `n`), both measures, per-primary
  squares, same grid and caps; `hinj_up[measure, smallest cap, species, E 5/dec 0.1-3.98, |cos Z| band, theta]`; the pending
  test keeps realisations/muons that can score at either root; copies with an up-going hit are traced. The manifest signature
  gets `[-2, 1]` appended only when on.
* Tests: `tools/mc3d/test_upgoing.py` (34): chord geometry (a ray entering at Q, 2 arcsin(c) from the site, leaves at the site
  with cos Z = -c, azimuth = Q's direction, cos psi = c, at Q it was down-going with cos c), cap membership, no-root cases,
  vector = scalar, an isotropic field of lines gives U = D in both measures, the driver (off = outputs unchanged; on = down-going
  bit-identical field free), the up-going pending test loses nothing (bend + muon field, no re-anchor), the muon cone test is
  conservative.

## 2. Pending test and re-anchoring (finding)

With `--reanchor` (production B's hybrid), the pending test is evaluated on copy 0 but the copies are rotated *and*
re-anchored onto the copy's own bent primary, so it is not exact. Paired `--pend-all` twin (same seed, 1970 primaries,
production-B physics) [m]: the normal test drops 0.4% of `s1` in total and 0.8% at cos Z 0-0.2 below 0.4 GeV, and ~0.9% of
`us1`; the copy-0 tallies `n1`/`un1` are exact. All mcup production-physics runs therefore use `--pend-all` (+6% CPU).
Production B (m4) did not, so its down-going horizon cells are expected to be low by about this amount (see sec. 5).

## 3. Field-free closure (U = D) and coverage

Field free (`--no-cutoff --no-bend-primary --no-muon-field`), GSFG:GH, theta_inj 50 deg, 10-deg cap, measure 1, az-summed
(`out/closure_C1.md`, `closure_C2.md`, `closure_C1+C2.md`; `scripts/a1_closure.py`). Pre-registered gate: `PREREG.md` sec. 1.

* **C1** (1.6e6 primaries, p/He strata 1-4, 4.56 core-h): **gate FAILED** by the letter [m]. Row 0.05 closes
  (ln sum U/sum D = +0.003 +- 0.011); row 0.15 is low, -0.027 +- 0.011 (C3 fails), with two cells beyond 3 sigma
  (numu 0.10-0.16 GeV 0.910 +- 0.020, antinumu 0.16-0.25 GeV 0.897 +- 0.024). C2 (species mean) and C4 (cap structure)
  pass. The deficit was confined to row 0.15 below 0.25 GeV (U/D 1.005 / 0.958 / 0.999 at rows 0.05 / 0.15 / 0.25) and
  spread over the jackknife groups.
* Scorer alone (post hoc diagnostic, `scripts/d2_isotropic.py`): 2.4e7 isotropic straight lines through the production code
  path give U/D(row 0.15) = 1.0000 +- 0.0029 [m] (rows 0.05 / 0.25: 0.997 +- 0.009 / 1.003 +- 0.002): no geometric bias.
  Atmosphere spherical, sampler Lambert about the local normal, pools azimuthally isotropic within statistics, decays
  covariant: no mechanism found.
* **Replication C2** (pre-registered in `PREREG.md` A2 before it ran: p:1 only, 1.5e6 primaries, 2.56 core-h):
  U/D(row 0.15, 0.10-0.25 GeV, species summed) = 1.013 +- 0.015 against C1's p:1 value 0.955 +- 0.017 (2.6 sigma apart)
  [m] -> **not reproduced**; by the A2 rule C1's failure is classified a fluctuation (post hoc classification).
* Pool C1 + C2 (3.1e6): C2, C3, C4 pass; C3 rows 0.05 / 0.15 / both = +0.003 +- 0.009 / -0.016 +- 0.009 /
  -0.005 +- 0.006 [m]; C1 by the letter still fails on one cell at 3.0 sigma (antinumu 0.16-0.25 GeV row 0.15, 0.928 +- 0.024).
  A third sample C3 (pre-registered in A6.2 to set the systematic: p:1/p:2/He:1, 1.33e6, 3.2 core-h) passes the gate alone
  (C3 rows 0.05 / 0.15: -0.012 +- 0.014 / -0.002 +- 0.017). **Pool C1 + C2 + C3 (4.43e6 primaries): all four criteria
  pass** [m] (C1: 3/56 beyond max(2%, 2 sigma), none beyond 3 sigma; C2 1/14; C3 rows 0.05 / 0.15 / both
  -0.0002 +- 0.0075 / -0.0114 +- 0.0088 / -0.0048 +- 0.0059; C4 7/7). Closure systematic carried on the up-going cells
  (A3 formula sqrt(C3^2 + sigma^2)): 0.75% (row -0.05), 1.44% (row -0.15).
* **Coverage** [m]: with a 50-deg patch, the share of up-going weight injected beyond 40 deg is 0.00% at rows 0.05 and
  0.15 in every band (10-deg cap; median / 99.9% injection distance 11 / 25 deg and 20 / 36 deg), and 7-11% at row 0.25
  (0.2-0.6% in the 2.5-deg cap). **The 40-deg production patch fully covers rows -0.05 / -0.15; row -0.25 needs ~50 deg.**
* Field free, the 12 azimuthal copies add nothing to az-summed tallies: rotations about the site vertical map the cap onto
  itself, so every copy scores the same neutrinos (`un1/n1` copy 0 = `us1/s1` all copies, to 1e-4).

## 4. Production-physics measurement (production B physics + `--upgoing --pend-all`)

`--model GSFG:GH --r-thr 65 --resim-rmin 32 --resim-groups p --reanchor`, cutoff/bend/muon field on, IGRF-14, theta_inj 40.
Pilot P1 1.23e5 primaries (1.54 core-h, 42.6 ms/primary: +20-25% over production B per primary), production P2 1.80e6
(19.98 core-h, allocation re-optimised on P1's own up-going variances, prediction 19.6), pooled per stratum: 1.92e6 primaries.
Analysis `scripts/a2_prod.py P1,P2` -> `out/prod_P1+P2.md/.npz`, deliverable `out/mcup_rows_P1+P2.npz`.

* Gates [m]: G1 kappa_run vs production B chi2/32 = 25.5 (p 0.79), species-mean ratio 0.99-1.02 per E bin; G2 this run's
  down-going G'' vs the table's MC cells chi2/56 = 76.8 (p 0.034), 12/14 species means within 2 sigma: both pass. Absolute
  down-going level vs production B (no kappa) at rows +0.05/+0.15 below 0.4 GeV: +0.7 to +4.8% (1-2.4 sigma per cell,
  correlated cells, mean ~ +3%), not the pending loss (sec. 5).
* Jackknife errors of the az-mean cells, E bins 0-2: per species 2.1-4.2% (row -0.05), 1.6-3.1% (row -0.15), as predicted.

## 5. Production B's pending loss, measured (paired)
P1 again without `--pend-all` (same seeds; P1np, 1.43 core-h) [m]: the copy-0 first-root test of production B drops
0.19 +- 0.19% of s1 at row +0.05, 0.10-0.16 GeV, and < 0.03% elsewhere (0.04% of all s1). Negligible for the down-going
table cells (the 0.4-0.8% of the 1970-primary bitid pair was a small-sample number). The union (first + second root) test
would have dropped 0.49 +- 0.13% of us1 at row -0.15 (1% at 0.10-0.16 GeV); the mcup runs used `--pend-all`.

## 6. Other facts worth keeping
* The pilot's kappa was 8% low at 0.16-0.40 GeV: its up-going-optimised allocation gave stratum p:0 (10% of the vertical
  1D tally t there) 300 primaries in one task, so the jackknife could not see the heavy-tailed shortfall. Floors of one task
  per stratum are not enough for kappa; P2 (28 p:0 tasks) is fine.
* The m4 driver's live CPU must use utime + stime only (cutime/cstime of the parent re-add the reaped pool workers).
* `shower.thresholds` drops charged pions below e_nu_min/0.427 = 0.234 GeV (PM note, another agent): a common, up/down
  symmetric MC artefact at 0.10-0.25 GeV, kept identical to production B here.
