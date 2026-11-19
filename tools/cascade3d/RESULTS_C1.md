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

None.

## 3. Found after the freeze, nothing changed

- `run_solve.py --straight` with the full line family is wrong in the row cos Z 0 to 0.1
  (lines tangent up to about 10 km). Cause: unbounded log-cubic energy interpolation of the
  muons where neighbouring bins differ by many decades (`GATES.md`, Results log;
  `c1/diag_straight/MANAGER_NOTE.md`). The same mechanism moves the frozen 3D solve by
  at most 2.5e-4 in the G1.3 cells and 1.0e-4 in the zenith-normalised cells. A fix is
  proposed (`c1/diag_straight/proposed.diff`) and not applied; G1.3 to G1.5 run on the
  code state of section 1.

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

After the diagnosis cycle (`GATES.md`, Results log; `c1/g13/manager/`): the count criterion
of G1.3(a) is failed by an exactly right solver in four cases of ten with this reference, so
377 does not separate a correct solver from a wrong one; the shape difference common to the
ten runs is 0.4% rms (below 1.1% at 2 sigma) below 1 GeV for cos Z 0.1 to 1, below 3.1% in
the row cos Z 0 to 0.1, and below 5.9% from 1 to 10 GeV. The level offset of G1.3(b) is the
same in the straight-ahead solve against `t` as in the 3D solve against `s`, and the same
for the four species. The verdicts above stand as written.
