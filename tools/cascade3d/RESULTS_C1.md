# Results of stage C1 (`cascade3d`)

## 1. Freeze of the numerical settings (rule 4 of `GATES.md`)

Frozen 2026-11-19 by the manager, after G1.0 to G1.2 were re-run on the final code and
before any output of the solver was compared with the Monte Carlo.

**Code state: commit `70d020d69ec76683e93d10564274b08c17d48429` of branch `phase3-cascade3d`.** The gates G1.3 to G1.5
run on this state. A later change of anything in this section is listed in section 2 with
its reason, and the gate concerned is re-run in full.

Who had seen what at the freeze. The author of the solver and the builder of the kernel
sets opened no Monte Carlo flux output (their reports say so; they were given bin edges
only). The manager read the reference tallies' structure and metadata, and compared the
reference with earlier Monte Carlo runs; no solver output was put next to a Monte Carlo
tally by anyone.

### Status of the gates at the freeze

| Gate | Verdict | Where |
|---|---|---|
| G1.0 | PASS | `c1/cycle2b/gates/G1.0.md` |
| G1.1 | **FAIL** (ground muons: +2.035% at the vertical, 0.141 GeV, loss on, against 2%; +101% at cos Z 0.05, 0.112 GeV, loss off, against 0.3%; every neutrino cell inside) | `c1/cycle2b/gates/G1.1.md` |
| G1.2 | **FAIL** (ground muons at cos Z 0.05 below 0.7 GeV and one cell at cos Z 0.15; every neutrino cell of G1.1 and every G1.3 cell inside, G1.3 cells at most 0.16%) | `c1/cycle2b/gates/G1.2.md` |

The PI's "Continue" of 2026-11-19 keeps G1.1 on record as failed. G1.2 failed again after
that decision; going on to G1.3 with it on record as failed is the manager's reading of the
same decision and is reported to the PI as such (`GATES.md`, Results log).

### Geometry and transport (defaults of the code state above)

| Setting | Value | Where |
|---|---|---|
| Atmosphere | CORSIKA `BK_USStd`, five layers, top 112.8 km | `grid.CorsikaAtmosphere` |
| Earth radius | 6371 km for G1.3 to G1.5 (the Monte Carlo's); 6391 km for G1.1 (MCEq's) | `run_solve.py --r-earth` |
| Altitude levels | 0.4 km steps to 50 km, 1 km to 70 km, 3 km to the top, the layer boundaries, and 1, 3, 10, 30, 100, 300 m below the top: 171 levels | `grid.make_levels(atm)` |
| Ground layer | off | `ground_layer=False` |
| Line family | 48 ground lines uniform in zenith angle plus the G1.1 directions and the G1.3 bin edges (57 ground lines), one tangent line per level: 227 lines, 48 734 nodes | `grid.make_line_family(n_ground=48, mode="theta")` |
| Ray sampling | 4 sub-points per segment; loss step at most 5 g/cm² (530 683 loss points); `n_buckets` 8 | `rays.Rays(n_sub=4, dX_max=5.0, n_buckets=8)` |
| Legendre order | L = 32, delta-M truncation as built into the kernel sets | kernel sets |
| Angular projection | exact Legendre moments of the piecewise-linear interpolant in μ | `Solver(projection="hat")` |
| Same-bin sweeps | 6, `gamma0` = 2 | `Solver(n_sweeps=6, gamma0=2)` |
| Up-scatter passes | 3 | `Solver(n_up=3)` |
| Highest bin marched | set from the primaries (51 for the Monte Carlo-matched sets) | automatic `k_top` |
| Energy grid | MCEq's, ten bins per decade, kinetic energy, lowest edge 25 MeV | kernel sets |
| Energy loss | on (muons, table `dedx_air` of the Monte Carlo at the bin centres) | kernel sets `*_losson` |

### Cell averages for G1.3 and G1.4

`assemble.cell_averages(solver, out)` with its defaults: mean over the cos Z bin from the
shape-preserving cubic through all ground directions and the horizontal node; mean of
dN/dE over the band as Σ φ_i w_i / Σ w_i over the solver's bins inside the band
(`energy="bins"`). Bands 0.1 × 10^(j/5) GeV, j = 0 to 10; cos Z edges 0 to 1 in steps of
0.1; species ν_e, anti-ν_e, ν_μ, anti-ν_μ.

### Kernel sets (`/home/pigrange/mc3d_work/runs/native3d/kernels/mc/`)

| Use | File | sha256 (first 16) |
|---|---|---|
| G1.3(a), (b), (d); G1.5 | `ks_mc_pool2000_L32_losson.npz` | `40c721af020961c1` |
| G1.3(c) | `ks_mc_fresh1e5_L32_losson.npz` | `700ca7d557d270ec` |
| G1.4, first half | `ks_mc_fresh1e5_h0_L32_losson.npz` | `80cd8d39f05623a0` |
| G1.4, second half | `ks_mc_fresh1e5_h1_L32_losson.npz` | `b807516d1813ce56` |

Primary spectrum and energy range: from the set (`phi_top`: GSF protons and neutrons,
kinetic energy per nucleon 1 to 3162 GeV). Builder's account of what is and is not like for
like: `c1/LIKE_FOR_LIKE.md`.

### Command of the G1.3(a) solve

```
taskset -c <4 cores> $PY tools/cascade3d/gates/run_solve.py MC losson --r-earth 6371 \
    --ks $N3/kernels/mc/ks_mc_pool2000_L32_losson.npz --out <file.npz>
```

`--straight` added for the straight-ahead limit of G1.3(d). An existing run of exactly this
configuration and code state: `c1/cycle2b/g12/C_base.npz` (same file as
`c1/cycle2b/costB/C_base_after.npz`).

### Reference

`c1/mcref2/mcref2_fieldfree.npz`: the field-free run of `tools/mc3d/sky3d.py` at commit
`7fac9b3` with `--muon-clock converged`, ten runs of 10^6 primaries (decision of the PI of
2026-11-19). σ_MC as in the Corrections of `GATES.md`, taken from this file. The first run
(`c1/mcref/`) is not compared with the solver.

### Known limits carried into G1.3

- Ground muons near the horizon below 0.7 GeV are not converged in the grids (up to 3% under
  a doubling, 14% between the default grid and one with a ground layer at cos Z 0.05,
  0.11 GeV). The neutrino cells do not depend on it at the level of 1e-5.
- ν_e at the vertical near 70 GeV moves by 0.38% under the L doubling (inside 0.5%; above
  the range of G1.3).
- Cost of one solve: 11.9 core-minutes in cycle 2, cut by 8% in cycle 2b; G1.5 allows 10.
  G1.5 is measured as written by the agent that runs G1.3.

## 2. Changes after the freeze

None before the gates of section 4 ran.

After them, on 2026-11-20, the PI authorised one closing cycle. Its two changes and the
second freeze are in section 5; the gates re-run in full on it follow in section 6. Sections 1, 3
and 4 stay as written: they describe the first frozen state.

## 3. Found after the freeze, nothing changed

- `run_solve.py --straight` with the full line family is wrong in the row cos Z 0 to 0.1
  (lines tangent up to about 10 km). Cause: unbounded log-cubic energy interpolation of the
  muons where neighbouring bins differ by many decades (`GATES.md`, Results log;
  `c1/diag_straight/MANAGER_NOTE.md`). The same mechanism moves the frozen 3D solve by
  at most 2.5e-4 in the G1.3 cells and 1.0e-4 in the zenith-normalised cells. A fix is
  proposed (`c1/diag_straight/proposed.diff`) and not applied; G1.3 to G1.5 run on the
  code state of section 1.
- The kernel sets give the nucleon bin of 1.26 to 1.58 GeV (kinetic energy) the
  interaction rate of its centre; the cross section of the Monte Carlo turns on at 1.39
  GeV, inside that bin. Against the Monte Carlo this raises the solver by 4.7 to 5.4% at
  0.10 to 0.16 GeV and by 2.9 to 3.3% at 0.16 to 0.25 GeV, in the straight-ahead and in
  the 3D solve alike (`GATES.md`, Results log, part C of the diagnosis cycle). No kernel
  set was changed.

## 4. Gates G1.3 to G1.5 on the frozen state

Run 2026-11-19 by an agent that wrote neither the solver nor the kernel sets; recomputed by
the manager from the stored cells. Details: `GATES.md`, Results log.

| Gate | Verdict | Numbers | Where |
|---|---|---|---|
| G1.3(a) | **FAIL** | 377 of 400 cells within max(2%, 2 sigma_MC); 380 needed. The ten row means below 1 GeV are inside (-0.73% to +0.53%) | `c1/g13/G1.3.md` |
| G1.3(b) | trigger fired | level above 5% in 4 of 40 (species, band); nu_e at 0.10 to 0.16 GeV +5.08 +- 0.49% | `c1/g13/G1.3.md` |
| G1.3(c) | reported | 10^5-event kernels: 379 of 400; level -1.4% to +0.8% against the pool kernels, shape within 0.36% | `c1/g13/G1.3.md` |
| G1.3(d) | reported | no valid number in the row cos Z 0 to 0.1 (section 3); rows 0.1 to 1: mean +0.17%, rms 3.5% | `c1/g13/G1.3.md`, `c1/g13/supp/` |
| G1.4 | PASS | half-sample sets within 0.479% (0.5% allowed) | `c1/g13/G1.4.md` |
| G1.5 | **FAIL** | 13.23 core-minutes (10 allowed), 5.38 GB (8 allowed) | `c1/g13/g15/G1.5.md` |

After the diagnosis cycle (`GATES.md`, Results log; `c1/g13/manager/`): the count
criterion of G1.3(a) is failed by an exactly right solver in four cases of ten with this
reference, so 377 does not separate a correct solver from a wrong one; the shape
difference common to the ten runs is 0.4% rms (below 1.1% at 2 sigma) below 1 GeV for
cos Z 0.1 to 1, below 3.1% in the row cos Z 0 to 0.1, and below 5.9% from 1 to 10 GeV.
The level offset of G1.3(b) is the same in the straight-ahead solve against `t` as in
the 3D solve against `s`, and the same for the four species. Below 0.25 GeV its cause is
measured: the Monte Carlo's nucleon cross section turns on at 1.39 GeV, inside the
solver's bin 1.26 to 1.58 GeV, which the kernel set gives one rate; a diagnostic set
with the interacting fraction of that bin (0.54 to 0.60) brings the level to within 1%
in the three lowest bands. The +1 to +2% above 0.4 GeV is not explained. Nothing was
changed in the frozen state, and the verdicts above stand as written.

## 5. C1 closing cycle: second freeze (rule 4 of `GATES.md`)

Frozen 2026-11-20 by the manager, after G1.0 to G1.2 were re-run in full on the changed
state and before any solve on the changed kernel sets was compared with a Monte Carlo
tally. Decision of the PI of 2026-11-20 and what was fixed before the runs: `GATES.md`,
"C1 closing cycle".

**Code state: commit `ab8bda740f0e6508331735854fd2439bcc1511a1` of branch
`phase3-cascade3d`.** G1.3 to G1.5 are re-run in full on this state.

### The two changes, with their reasons

1. **Solver, commit `1e5d2d8`.** `solver._interp_log` bounds the log-cubic value by the
   smallest and the largest of the four stencil values where these span more than a
   factor 1e3. Reason: section 3, first item (the straight-ahead solve had no valid
   number in the row cos Z 0 to 0.1). The diff is the one proposed on 2026-11-19,
   unchanged, plus a unit test.
2. **Kernel sets, commit `ab8bda7`.** The Monte Carlo-matched sets treat the turn-on of
   the nucleon cross section inside the bin of 1.26 to 1.58 GeV (rule M3 of
   `kernelset_mc.py`). Reason: section 3, second item. The rule was fixed in the
   builder's brief before any set was built (`c1/close/K_BRIEF.md`), from the Monte
   Carlo's interaction-length function and table: for protons and neutrons the sources
   into that bin are multiplied by the fraction that interacts over 2000 g/cm2
   (primaries 0.602 and 0.594; secondaries 0.605 and 0.607, weight T^-g with g 0.91 and
   0.87), and the bin's rate is that of its interacting population (9.46e-3 and
   9.62e-3 cm2/g in place of 1.080e-2 and 1.101e-2). 2000 g/cm2 is not adjusted.
   Nothing else in the sets changes: against each predecessor only `phi_top` (2
   entries), `inv_lambda` (2 entries) and the rows 17 of the 26 channels into p and n
   differ, by these factors (builder's report `c1/close/WP_K1_REPORT.md`; manager's own
   comparison and quadrature `c1/close/manager/check_k1.txt`).

Everything of section 1 stays as frozen on 2026-11-19: geometry and transport, cell
averages, reference. The gate scripts changed in the paths of the kernel sets
(`g1_2.py` part C, `g1_5.py`) and `g1_3.py` in the count option of the amended G1.3(a)
(`--count-min 373`; without it the script reproduces the result of 2026-11-19 from the
stored cells).

### Kernel sets (`/home/pigrange/mc3d_work/runs/native3d/kernels/mc_close/`)

| Use | File | sha256 (first 16) | Predecessor |
|---|---|---|---|
| G1.3(a), (b), (d); G1.5; G1.2 C | `ks_mc_pool2000_L32_losson.npz` | `32b4215d5bf0de6d` | `40c721af020961c1` |
| G1.3(c) | `ks_mc_fresh1e5_L32_losson.npz` | `dcb4455805e9164a` | `700ca7d557d270ec` |
| G1.4, first half | `ks_mc_fresh1e5_h0_L32_losson.npz` | `dc46a30ecc401927` | `80cd8d39f05623a0` |
| G1.4, second half | `ks_mc_fresh1e5_h1_L32_losson.npz` | `eafe25654d7b2902` | `b807516d1813ce56` |
| G1.2 C, L doubled | `ks_mc_pool2000_L64_losson.npz` | `24b2f161303522fc` | `ecfb53df4840c94b` |
| G1.2 C, diagnostic | `ks_mc_pool2000_L16_losson.npz` | `17d8691224e6d2be` | `6e3cee950c360991` |

Full hashes: `c1/close/kbuild/k1/sha256_mc_close.txt`. The sets of 2026-11-19 are
untouched (`kernels/mc/`, `c1/cycle2/kernels/`); `build(turn_on=False)` reproduces them
array by array.

### What the kernel change does, without any tally

- Probability that a nucleon of the bin interacts over X g/cm2, set against the direct
  integration of the Monte Carlo's rule (the check fixed in `GATES.md`): exact at 2000
  g/cm2; +4.2%, +6.4%, +6.4%, +3.2% at 100, 200, 500, 1030 g/cm2; -4.1% and -7.0% at
  5000 and 10 000 g/cm2 (protons, primary weight; neutrons and the secondary weight
  within 0.2 points of these). The sets of 2026-11-19 were +55 to +87% high. One
  fraction per bin cannot follow the depth; this residual is carried into G1.3.
- Straight-ahead solve, new set over old, four species, cos Z 0.1 to 1: -4.71%, -2.89%,
  -1.13%, -0.23%, -0.02% in the five bands from 0.10 to 1 GeV, nothing above.

Reported by the builder and not treated (`WP_K1_REPORT.md`, section 7): every other
projectile has one such bin (antinucleons 17, pions 14, kaons 15, Lambda 18), and the
Monte Carlo serves part of the pion bin 18 and of the kaon bin 17 that the sets leave
empty. Estimated on variants of the set, each moves no band by more than 0.03% and no
cell by more than 0.06%.

### Who had seen what at this freeze

The comparison is not blind any more: the manager and the agent of section 4 saw the
solver next to the reference on 2026-11-19. The builder of the kernel sets opened no
Monte Carlo flux tally (report); the rule, its depth and its check were fixed before
the sets existed. The solver change is the diff proposed before the PI's decision. No
solve on the changed kernel sets was compared with a tally before this freeze; the
statistic and the wording of the candidate test are fixed in `GATES.md`.

The variant sets of the candidate test were built and solved before this freeze
(`c1/close/WP_K2_REPORT.md`). They were compared with the base solve only, by the builder
and by the manager (`c1/close/manager/k2check/CHECK.txt`).

### Status of the gates at this freeze

| Gate | Verdict | Where |
|---|---|---|
| G1.0 | PASS | `c1/close/gates/G1.0.md` |
| G1.1 | **FAIL**, with the numbers of 2026-11-19 (ground muons: +2.035% at the vertical, 0.141 GeV, loss on, against 2%; +101% at cos Z 0.05, 0.112 GeV, loss off, against 0.3%) | `c1/close/gates/G1.1.md` |
| G1.2 | **FAIL**. Parts A and B: the numbers of 2026-11-19 (A fails in the loss-off muon cell at cos Z 0.05, 0.112 GeV; B is inside). Part C, on the new kernel sets: ground muons at cos Z 0.05 move by up to 3.2% (line family), 2.3% (L), 1.1% (ray sampling) and 0.7% (levels) against 0.5%; the G1.3 cells move by at most 0.16% | `c1/close/gates/G1.2.md` |

G1.0, G1.1 and G1.2 parts A and B ran on commit `1e5d2d8`. From there to the frozen
commit the diff touches `kernelset_mc.py` and its test, `gates/g1_2.py` (paths of part
C), `gates/g1_3.py` and its test, `gates/g1_5.py` (path) and `GATES.md`; none of them
is imported by the solves of those gates (they use `kernelset.py`, the MCEq and the
synthetic kernels). G1.2 part C ran on the frozen commit. Unit tests of
`tools/cascade3d` on the frozen commit: 116 passed (`c1/close/logs/pytest_ab8bda7.log`).

No change of this cycle addresses the failures of G1.1 and G1.2 (ground muons).

### Command of the G1.3(a) solve

```
taskset -c <4 cores> $PY tools/cascade3d/gates/run_solve.py MC losson --r-earth 6371 \
    --ks $N3/kernels/mc_close/ks_mc_pool2000_L32_losson.npz --out <file.npz>
```

`--straight` added for G1.3(d). An existing run of exactly this configuration and code
state: `c1/close/g12/C_base.npz`.
