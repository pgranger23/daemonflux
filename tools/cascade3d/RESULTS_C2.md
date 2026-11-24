# Results of stage C2 (`cascade3d`)

## 1. Freeze of the numerical settings (rule 4 of `GATES.md`, as amended 2026-11-24)

Frozen 2026-11-24 by the manager, at the end of items 1 to 3 of stage 1 (`GATES.md`,
"After the site solve", item 4): after G2.0, G2.1, G2.2, G2.4(a), G2.5(a) on the
orders, G2.5(b), G2.5(c), G2.7 and G2.8, and before any output of the solver with the
cutoff was compared with a Monte Carlo tally.

**Code state: commit `954abfb1c6c2a64bff554f4d9ca711f1acd619d0` of branch
`phase3-cascade3d`.** The cap solves and their gates ran from an exported copy of it
(`git archive`), `c2/solver/code_954abfb/`; the manager compared 15 modules of the
copy with the commit, all equal. G2.3 is evaluated on the cells listed below. A later
change of anything in this section is listed in section 2 with its reason, and G2.3
is re-run in full.

**Not run at this freeze** (stage 2, not released by the PI): G2.4(b), the reference
altitude, and the doublings of the μ grid and of the Legendre order of G2.5(a). G2.4
and G2.5 are incomplete and are reported as such. G2.3 is first evaluated on a state
whose μ grid, L and reference altitude are tested without the cutoff only (C1, G1.2:
every neutrino cell within 0.16%). If a test of stage 2 fails and a setting is
changed, G2.3 is run again in full and both results are reported.

Paths below are relative to `/home/pigrange/mc3d_work/runs/native3d/` unless they
begin with `gates/` or name a module (then `tools/cascade3d/`).

### Who had seen what at the freeze

- The agents that wrote the solver and built the cutoff skies were told not to open
  the references of G2.3 (`c2/mcref/`, `c2/mcref_roll/`), the thresholds, the
  manager's files on the azimuth of the reference, nor the parts of `GATES.md` that
  hold their content. Every brief carried this rule. Report A says that nothing
  under `c2/mcref/` and `c2/manager/` was opened, report D that no off-limits file
  was read; reports B and C make no statement. The solver agent stopped on a usage
  limit on 2026-11-24 at about 14:07, during the cap solves and before a report E.
  This is the word of the agents and of their briefs; it was not checked otherwise.
- The manager read the new reference (integrity, level, azimuth), computed the
  thresholds from it and compared the two references with each other. The manager
  has compared no output of the solver with a tally and has passed nothing of the
  reference to a solver agent.
- One disclosure (`GATES.md`, Results log, 2026-11-23, "Site solve at M = 12"): the
  manager printed the solver's azimuth mean of the cutoff factor per band for one row
  at the site, knowing the level of the reference per band from the threshold
  report. No comparison was made or written.
- The manager's checks of the cap solves (`c2/manager/stage1/check_caps.py`) print no
  level of the cutoff factor per band or cell: its largest excess over 1, the count
  of cells above 1, and differences between solves. The result file of G2.4(a)
  prints the smallest and the largest cutoff factor of the 4800 cells on grid B; the
  manager read it.
- No setting of this section was chosen with a tally in view. M = 8 was chosen on
  cost and on the partial sums of the site solve; grid B on the square law of the
  cap-averaged boundary (step E) and on G2.0.

### Status of the gates at the freeze

| Gate | Verdict | Where (`c2/solver/gates/`) |
|---|---|---|
| G2.0 | reported: **TRIGGER** on the 5° grid (on record); **no trigger** on grid B, the grid of the solves (largest difference 0.78 of its tolerance) | `G2.0/G2.0.md`, `G2.0/G2.0_gridB.md` |
| G2.1 | PASS: Φ_0 of the cap solve equals the C1 cells exactly (bookkeeping only; holds by construction) | `G2.1/G2.1.md` |
| G2.2 | PASS: worst neutrino cell −0.302% against 1% | `G2.2/G2.2.md` |
| G2.4(a) | PASS: largest move 0.392% against 0.5%, grid A against grid B | `G2.4/G2.4a.md` |
| G2.4(b) | **not run** (stage 2) | |
| G2.5(a), orders | **FAILED** at M = 6 (0.507% against 0.5%, M = 6 against 12; on record, M = 6 not used). PASS at M = 8 (0.183%, M = 8 against 16) | `G2.5/G2.5_orders.md`, `G2.5/G2.5_orders_M8.md` |
| G2.5(a), μ grid and L | **not run** (stage 2) | |
| G2.5(b) | PASS: 7.8e-16 against 0.005 (holds by construction) | `G2.5/G2.5_rotation.md` |
| G2.5(c) | PASS: largest residual 1.49e-8 against 1e-4, the site solve and each of the three cap solves | `G2.5/G2.5_c_cap10_B.md`, `G2.5_c_cap5_B.md`, `G2.5_c_cap10_A.md`; site: Results log |
| G2.6 | reported: see "Cost" below | |
| G2.7 | PASS on the first setting: 0.439% against 0.5% and 0.237% against 1% | `G2.7/G2.7.md` |
| G2.8 | reported: **TRIGGERED** under its text (point values at the sector centres −4.68% against 1%); sector means 0.040%; the check the gate asks for is done and finds no fault of the mode path | `G2.8/G2.8.md`, `c2/manager/stage1/check_g28.out` |

Carried from C1 and unchanged: G1.1 and G1.2 failed for ground muons, G1.5 failed
(cost). The gates of C2 are on the four neutrino species.

Who ran what. G2.0, G2.2, G2.5(a), G2.7, G2.8, G2.1, G2.5(b) and G2.5(c) of
`cap10_B`: the solver agents. G2.5(c) of `cap5_B` and of `cap10_A`, and G2.4(a): the
manager, with the committed scripts from the exported copy, after the solver agent
had stopped (`GATES.md`, Results log, 2026-11-24, "Stage 1, the cap solves"). Unit
tests on the tree that became the frozen commit: `test_solver3d.py` 16 passed,
`test_boundary.py`, `test_assemble3d.py`, `gates/test_g2_3.py` and
`gates/test_c2_stats.py` 33 passed (`c2/manager/stage1/pytest_mgr*.log`).

### Geometry and transport

Everything of `RESULTS_C1.md`, sections 1 and 5, as frozen there (the solve of
G1.3(a) of the closing cycle): atmosphere, 171 levels, 227 lines, ray sampling,
L = 32, projection, sweeps, up-scatter passes, energy grid, energy loss, Earth radius
6371 km, cell averages in energy and cos Z. Built by `gates/g2_7.build_geometry()`.
Added by C2:

| Setting | Value | Where |
|---|---|---|
| Kernel set | `kernels/mc_close/ks_mc_pool2000_L32_losson.npz`, sha256 `32b4215d5bf0de6d2b4600d9aaa991566907aee6e211121ace1b3ca735829bd6` (the set of G1.3(a) of the closing cycle) | `g2_7.KS_PATH` |
| Flux with the cutoff | Φ_cut = Φ_0 − J[D]: Φ_0 the C1 solve, D = B⁰ − B the deficit of the nucleon boundary at the top, J the derivative of the C1 solve at Φ_0 | `solver3d.TangentSolver` |
| Guard of the derivative | ρ = 0.1 on the source shape, the source integral and the loss interpolation; the Simpson correction not guarded (the first setting of G2.7) | `solver3d.GUARD_PROTOTYPE` |
| Up-scatter passes in the tangent | 3 | `TangentSolver` |
| Azimuth | Fourier orders 0 to M = 8 about the local vertical, cosine and sine parts: 17 tangents per boundary | `run_solve3d.py --M 8` |
| Cells in azimuth | means of the series over each of the twelve bins of 30° | `boundary.bin_mean_basis`, `assemble3d.azimuth_cells` |
| Location grid | grid B: x (east) −10 to 10° in steps of 5°, y (north) −10 to 10° in steps of 1.25°, azimuthal equidistant about the site; 85 nodes | `boundary.LocationGrid((5, 1.25), 10)` |
| Cap average | Σ_P W_P D_P with the cap weights of the bilinear interpolation on the grid; cap of 10°: 81 tables with weight; cap of 5°: 27 | `LocationGrid.cap_weights` |
| Sky table of a node | 45 line nodes (30 ground directions, 15 tangent altitudes from 0 to 99 km) × 36 sectors of 10° × 31 rigidity rungs (1.7 to 60 GV, 20 per decade); transmission 1 above 60 GV; one draw per state, seed 20261123; injection sphere at 100 km | `boundary.SkyTable`, `gates/c2_skies.py` |
| Reference altitude | h_ref = 0 | `c2_skies.py` |
| Nucleus groups to the nucleon boundary | primary model `GSFG:GH` (the Monte Carlo's), 1 to 10^3.5 GeV per nucleon: share of the nucleons of each energy bin per rigidity rung | `boundary.GroupWeights` |
| Lines to chains | 227 chains (57 ground, 170 tangent) from the 45 line nodes; the 11 tangent chains above 100 km never reach the injection sphere and have transmission 0 with the cutoff | `boundary.chain_weights` |
| Processes | four per boundary, of 5, 4, 4 and 4 tangents (orders 0c 1c 1s 2c 2s; 3c 3s 4c 4s; 5c 5s 6c 6s; 7c 7s 8c 8s), `--batch 5`, four cores | `gates/run_caps_M8.sh` |

### Cutoff skies (`c2/solver/sky/`)

121 tables, all with the same draws. The 85 of grid B: report `SKIES_5x1p25.md`
(sha256 `06f5cfe1ebd096d5770c37de17782906e081f779bf83718b984d9d0c5d821df0`). Each
tangents file stores the path, the weight and the sha256 of every table it used; at
the freeze the tables on disk have these hashes (`c2/manager/stage1/check_caps.out`).

### The cap solves (`c2/solver/caps_M8/`)

| Solve | Cap | Grid | Use | `cells_M8.npz`, sha256 |
|---|---|---|---|---|
| `cap10_B` | 10° | B (5° × 1.25°) | G2.3 (a) to (f), G2.1, G2.4(a) | `f653022ca5363cc5cd55f8d2c16b4f55398968fb59c44a1ab5dbb041ea8f22ec` |
| `cap5_B` | 5° | B | G2.3(g) | `7cb9e8a1ee0bdae900d226c1c1e0ad574d571d41623045864885e4255d0d7b98` |
| `cap10_A` | 10° | A (5° × 2.5°, 45 nodes) | G2.4(a) only | `5a7fa19103e6c8d3b733e2963c8526858360063f83f616b70c844f51b84834b0` |

Hashes of the twelve tangents files: `c2/manager/stage1/check_caps.sha256`. Command
of one process and of the cells (driver `c2/solver/caps_M8/run_caps_M8_export.sh`, the
committed `gates/run_caps_M8.sh` with the paths of the exported copy):

```
taskset -c 0-3 /usr/bin/time -v -o $D/b$i.time $PY gates/run_solve3d.py run $D \
    --cap <10|5> --spacing <5,1.25|5,2.5> --M 8 --orders <orders> --batch 5 --tag b$i
$PY gates/run_solve3d.py combine $D --tags b1,b2,b3,b4 --M 8 --partial "" \
    --out cells_M8.npz
```

Checked by the manager, without any tally (`check_caps.out`): the settings stored
with each of the twelve files; the cap weights against an own quadrature (within
1.4e-16); an own sum of the tangents against each `cells_M8.npz` (equal); Φ_0
against the C1 cells (equal); the primal equal in the three solves; no cell of the
cutoff factor negative or above 1 + 1e-3.

G2.4(a), cap of 10°, grid A against grid B: **PASS.** Largest move
|Φ_A − Φ_B| / Φ_B: 0.392% (limit 0.5%; 0.390% with Φ_A as the denominator), at ν_e,
0.251 to 0.398 GeV, cos Z 0 to 0.1, azimuth 60 to 90°. 1767 cells of 4800 move by
more than 0.1%, 133 by more than 0.3%, none by more than 0.5%. Worst cell per band,
0.1 to 10 GeV: 0.35, 0.38, 0.39, 0.37, 0.33, 0.27, 0.19, 0.15, 0.12, 0.06%. If the
error goes with the square of the spacing, that of grid B is a third of the move,
about 0.13%. Written before the run: "The largest cell may come close to 0.5%."
The manager's own difference of the two cell files gives the same numbers.

### The file of G2.3

Made and evaluated by the agent of G2.3, after this freeze:

```
$PY gates/g2_3_file.py --cap10 $N3/c2/solver/caps_M8/cap10_B \
    --cap5 $N3/c2/solver/caps_M8/cap5_B --out <dir>/solver_g23.npz
$PY gates/g2_3.py --ref $N3/c2/mcref_roll/c2roll_cutoff.npz \
    --null $N3/c2/manager/thresholds_roll/cap10/null_count_c2.json \
    --null5 $N3/c2/manager/thresholds_roll/cap5/null_count_c2.json \
    --solver <dir>/solver_g23.npz --outdir <dir>
```

Scripts at the frozen commit (sha256):

- `gates/g2_3.py` `a410685bf7b45a713f53eae87c4cbf5b06fc4cfad7f1b2c039075e44cd572211`
- `gates/g2_3_file.py`
  `4075f2a5eeab76f271e60d55bf005a455bc96a97fbc29e2c234ce749ade95391`
- `gates/c2_stats.py`
  `0b54bc056c47b405b4376950f00e62d1d48ebaf394a3267953e16896845d0ba2`

Part (f) takes the straight-ahead flux from `c1/close/g13/pool2000_straight.npz`.
Part (h) is not evaluated (`GATES.md`, entry "Before the cap solves", item 6).

### Reference and thresholds

`c2/mcref_roll/c2roll_cutoff.npz`, sha256
`fb226760abacc2f40963d5e48f8264830a5349baf47a9b589592965c9775fc50`: twenty runs of
`sky3d.py` with `--event-roll`, cutoff on, no bending (decision of the PI of
2026-11-23). Thresholds: computed from the reference alone on 2026-11-23, before any
cap solve existed; `c2/manager/thresholds_roll/cap10/null_count_c2.json` (sha256
`a18f7d8c73d1783ac67b16ea7586c79ae5ecafb9ff9000312125d885a509a1d0`) and
`cap5/null_count_c2.json`
(`db22225063607e27d2e5d027d7ddde356be28c2469082c0063d01e04a620ad48`). Their values,
the calibration of part (a) and the rule for a narrow failure are in `GATES.md`,
entry of 2026-11-23 "The new reference of G2.3 (event roll) and its count
thresholds". The first reference (`c2/mcref/`) is not used by G2.3.

### Cost (G2.6)

`/usr/bin/time -v`, user plus system time, four pinned cores, one process at a time.
Peak memory is the largest resident set of a process, in units of 10^6 kB as in the
earlier entries.

| Solve | b1 (5 tangents) | b2 | b3 | b4 | Total | Peak memory |
|---|---|---|---|---|---|---|
| `cap10_B` | 64.3 core-min, 32:36 wall | 48.5, 21:55 | 47.8, 22:30 | 48.4, 18:20 | 208.9 core-min = 3.48 core-h | 19.7 GB |
| `cap5_B` | 57.8, 24:41 | 45.5, 16:54 | 46.5, 16:59 | 46.9, 17:02 | 196.8 = 3.28 | 20.9 GB |
| `cap10_A` | 57.6, 23:54 | 49.5, 18:14 | 47.8, 17:22 | 47.5, 17:21 | 202.3 = 3.37 | 19.8 GB |

The set of solves of G2.3 (`cap10_B` and `cap5_B`): 405.7 core-min = 6.76 core-h,
peak 20.9 GB. The three cap solves: 608.0 core-min = 10.13 core-h (projected: 10.9).
One location solve at M = 8 is one boundary of 17 tangents: about 3.4 core-h and
20 GB. G1.5 asked for 2 core-min and 2 GB per location.

### Known limits carried into G2.3

- Stage 2 has not run (above).
- The absolute level is that of C1: the solver is expected about 1.8% high from 0.4
  to 10 GeV. G2.3 is gated on quantities in which this level cancels.
- G2.7 has a small margin in one cell (anti-ν_e, 0.10 to 0.16 GeV, cos Z 0 to 0.1:
  0.439% against 0.5%).
- G2.5(a) on the orders: M = 8 against 16 moves a cell by up to 0.183%, most of it
  draw noise of the table. The sector width of the tables (36 sectors) is not doubled
  by any gate.
- G2.8 fired under its literal text; the kernel moments of the orders m ≥ 1 are not
  tested by it.
- G2.2 does not exercise `GroupWeights` nor the cap assembly. The manager checked
  these against a second computation (`c2/manager/stage1/check_gw.out`), within
  2.8e-5 of the uncut boundary.
- G2.0 on grid B: the class 10 to 20 GV, cos Z 0.5 to 1, west, is at 0.78 of its
  tolerance, on the 5° grid as on grid B.
- Location grid: the spacing in x is not tested by a solve (boundary level only:
  −0.027% ± 0.026% by number). G2.4(a) tests the spacing in y.
- One draw per state in each table; the draw noise is accepted for C2. Grids A and B
  hold the same draws, so G2.4(a) does not see it.
- Φ_0 has the uncut boundary on the 11 chains that never reach the injection sphere,
  where the tally `n` has no primary. Estimated below 1e-4 of a ground cell, not
  measured.
- G2.3(h), the up-going rows, is not evaluated.
- Ground muons: the failures of G1.1 and G1.2 are carried.

## 2. Changes after the freeze

None.

## 3. G2.3 on the frozen state

This section holds the result of G2.3. It is closed to an agent that writes the solver
or the skies until the PI decides otherwise.

**G2.3: FAIL** (2026-11-24, cap of 10°). Run by an agent that wrote nothing of the
solver, of the kernel sets or of the skies (rule 5), with the two commands of "The file
of G2.3" above, from the code state of the freeze. No setting was changed between the
freeze and the run.

| Part (gated) | Items | Inside | Needed | Result | First rule: needed, result |
|---|---|---|---|---|---|
| (a) cutoff factor per cell | 4800 | 4481 | 4640 | **FAIL** | 4654, fail |
| (b) harmonics per band and row | 100 | 88 | 97 | **FAIL** | 97, fail |
| (c) horizontal over vertical per band | 10 | 9 | 8 | **PASS** | 9, pass |

- The threshold of (a) has the caveat of the entry of 2026-11-23: an exactly right
  solver has 4675 ± 22 cells inside and falls below 4640 in 17.3% of the cases. The
  count is 8.7 standard deviations below 4675, so the caveat does not bear on it.
- Reported parts: (d) 4404 of 4800; (e) the ten bands inside, the trigger did not
  fire; (f) 4369 of 4800; (g) cap of 5°: (a) 4420 of 4788 against 4543, (b) 84 of 100
  against 94, (c) 9 of 10 against 8; (h) not evaluated.
- Result files, `c2/g23/`, never overwritten:
  - `solver_g23.npz`, sha256
    `9d8299472b2298644caaea14a987e84ec81979ae6466820610b86189d8abf0d1`;
  - `G2.3.md`, sha256
    `9ba59497c76310e4b6a80e6f44e3de38fac6357e1067c409b173eae61bdbb020`
    (`g23.log` holds the same bytes);
  - `G2.3.json`, sha256
    `f335e3b4e0ca21aead34396ea660273e2541ac834c0d44a34f13c88573a539c3`;
  - `g23_file.log`, sha256
    `a446205dc9aae5d58252e71a772e4b59438fdcddc5acc724ba16c477ef521c75`.
- Verification by the manager, `c2/manager/g23/` (files read, nothing solved):
  `verify_g23.py` (`a42cb907…`) and `.out` (`ec8a0903…`). The file of G2.3 holds the
  frozen cells exactly; the manager's own formulas on the runs of the reference give
  the same 4481 and 88 with no flag differing.
- Description of the cells outside: `describe_a.py` (`71804fd6…`), `.out`
  (`378b31ff…`), `describe_b.py` (`71e63083…`), `.out` (`ce5e3fad…`). The text is in
  the Results log of `GATES.md`, entry "Stage 1, item 5", with the manager's reading.
- Stage 2 has not run. G2.4 and G2.5 stay incomplete. Nothing was run after G2.3; the
  result goes to the PI first (decision point 2).
