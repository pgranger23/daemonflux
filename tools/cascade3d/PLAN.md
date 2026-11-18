# Phase 3 plan: `cascade3d`, a deterministic 3D cascade solver

v0.2, 2026-11-18. **Approved by the PI on 2026-11-18**: C0 and C1 are authorised; decision
point 1 (after C1) and every later stage need a PI decision. Pre-registered gates: `GATES.md`.
v0.2: solver core in JAX at the PI's suggestion (sections 3, C1, C4, 7, 8).
Background and equations: `THEORY_NOTE.md` in this directory (route B, staged as C).
Tags: **[m]** measured or recorded in this project, **[e]** my estimate, **[d]** design choice.

## 0. What "done" means

| PI goal | Deliverable | Test |
|---|---|---|
| Low-energy fluxes with all 3D effects | 3D/1D factor tables F(E, cos Z, az, species) from one solver, any site | Agrees with our MC within its statistical error (gates G1.3, G2.3, G3.3); Honda/Bartol as consistency check |
| Easy, correct hadronic-uncertainty propagation | ∂F/∂θ for the 24 daemonflux parameters plus angular-shape parameters, applied inside `daemonflux.Flux.error()` | 1D limit reproduces daemonflux's own Jacobians (G4.1); `error()` matches brute-force re-solves (G5.3) |

The current product is untouched while this runs: `kamioka_3D_v2l.npz` stays the recommended
table, `tools/mceq3d` is frozen except for bug fixes, `tools/mc3d` stays the reference.

## 1. Scope

In scope: hadrons, muons and neutrinos from 0.03 GeV to 10^4 GeV primaries; curved
atmosphere; geomagnetic cutoff; bending of primaries and muons; polarised muon decay;
muon energy loss; parameter derivatives.

Not in scope: electromagnetic cascades and prompt leptons (as in MCEq and the MC); a better
low-energy hadronic model (the 1.9 / 3.0 GeV projectile floors and their uncertainty are
inherited unchanged); bending of pions and kaons (bend over a decay length: pions (5 to 8)×10^-4 rad
for 0.3 to 0.47 G, kaons 7 times less [e]); neutrino absorption in the Earth.

## 2. Physics definition (frozen conventions)

- Unknown at one location P: f_i(h, μ, φ; E) = intensity of species i at altitude h, with
  μ the cosine of the angle between the direction of motion and the local downward
  vertical, φ the azimuth about the vertical. Horizontal position is integrated out
  (THEORY_NOTE section 3); homogeneity is used only for collided sources, i.e. within a few
  interaction lengths.
- Atmosphere: CORSIKA BK_USStd, the parameterisation shared by MCEq and
  `tools/mc3d/atmosphere.py`, top at 112.8 km [m].
- Species and energy grid: MCEq's own (helicity-labelled muons), so the straight-ahead
  limit is MCEq [d]. Species without generator angular shapes are transported straight ahead.
- Populations PHYS and REF, the hybrid level and the product definition
  F = Φ_3D^PHYS / B_ref: exactly as in `tools/mceq3d/INTEGRATION.md`. No change.
- Primaries: nucleus groups by superposition, rigidity R = (A/Z) p for the cutoff, as in the
  MC (`bend.py`, `primaries.py`).

## 3. Numerical design

Each item is a choice [d] with its fallback. C1 tests all of them at low cost.

1. **March in energy, highest bin first.** Same-bin couplings (π→μ, leading nucleons,
   p↔n) are closed by a fixed number of Gauss-Seidel sweeps over species (fixed, so that the
   step compiles and differentiates); the residual after the last sweep is an output and
   must be below 1e-4; expected ≤ 6 sweeps [e]. Fallback: exact small-block solve per bin for the straight-ahead part.
2. **Transport by backward characteristics.** For each node, integrate source × survival
   back along the true trajectory: straight line with curvature (C1, C2), arc in the local
   field for muons and uncollided primaries (C3). Geometry arrays (altitude, grammage,
   local angles along each back-ray) are computed once and cached. Sources are interpolated
   per unit grammage, which is smooth where the density is not. Muon energy loss is carried
   along the trajectory (continuous slowing down), with the PDG table already used by
   `shower.py`.
3. **Collision source by Legendre moments with delta-M truncation.** The unresolved forward
   peak is kept in the parent's direction; moments 0..L carry the rest. Setting the forward
   fraction to 1 is the straight-ahead limit, on the same code path. L = 32 to start.
   Fallback: binned kernels and direct angular quadrature.
4. **Kernels.** c_l(E',E) = c_0(E',E) × a_l(E',E).
   - c_0, interaction lengths, decay matrices: blocks of MCEq's `int_m` and `dec_m`,
     extracted as `mceq3d_real.MCEqCascade3D` already does [m].
   - a_l: cos θ histograms of secondaries from DPMJET-III-19.3 (below 80 GeV) and
     SIBYLL-2.3d (above), generated through `tools/mc3d/pool.py`'s generator setup on the
     existing 708 (projectile, energy) nodes, mapped to the MCEq grid with the MC's own
     rescaling rule. Histograms are stored, not events.
   - A second, **generator-complete** kernel set (c_0 also from the generators) exists only
     for like-for-like comparison with the MC.
   - Decays: two-body angles are fixed by the energies (analytic a_l); three-body and
     polarised muon decay are sampled from `tools/mc3d/decays.py` at fixed helicity.
5. **Boundary.** Top: primary intensity × cutoff transmission T_P(direction, rigidity),
   T from back-tracing along the true line (`tools/mc3d/cutoff.py`). Ground: absorbing.
6. **Neutrinos are not transported.** The solver outputs the neutrino emissivity
   S_ν(P, h, μ, φ, E). The flux at a detector is a line-of-sight integral through S_ν on
   the near side and, through the Earth, the far side.
7. **Implementation: JAX for the solver core** (`solver.py`, `assemble.py`, `deriv.py`),
   64-bit, as pure functions on fixed-shape arrays. Reason: `jax.jvp` and `jax.vjp` give
   the tangent and the adjoint of the whole solve without hand-written derivative code,
   which is goal 2. Everything that does not depend on hadronic parameters stays in numpy
   and is computed once: kernels from MCEq and the generators, ray tables (indices,
   weights, grammage), cutoff skies.
   - Not a speed argument on this node: it has no GPU [m], and on CPU I expect JAX near
     vectorised numpy [e]. The same code runs on a GPU if one is available for C5.
   - JAX's CPU thread pool ignores `OMP_NUM_THREADS`; every run is pinned with `taskset`
     to stay inside the 24-core limit.
   - The energy loop stays in Python around a compiled per-bin step, so reverse mode
     stores each energy slice once.
   - `src/daemonflux` gets no new dependency: it only reads tables.
   - Install: JAX 0.6.1 and numba 0.61.2 ship in the LCG_108 view but are not visible to
     our environment [m]. Default: pip install of the current release (0.12.1 [m]) into
     `/home/pigrange/mc3d_work/pylib_jax`, never into the repo venv.

   Starting grids: 120 altitudes, 96 μ nodes (dense near the horizon), 16 φ nodes from C2,
   ≤ 200 samples per back-ray.

Code: new directory `tools/cascade3d/` with `kernels_mceq.py`, `kernels_gen.py`,
`kernels_decay.py`, `grid.py`, `rays.py`, `solver.py`, `boundary.py`, `assemble.py`,
`deriv.py`, tests, `GATES.md` (pre-registration), `RESULTS_C<n>.md`. NumPy docstrings,
88-column lines wrapped by hand (black and flake8 are not installed).

## 4. Stages, work packages, gates

Every gate is written into `GATES.md` before the run that tests it. A stage ends with a
report and a PI decision. Work packages are sized to one agent task each.

### C0. Foundations

| WP | Content |
|---|---|
| 0.1 | `kernels_mceq.py`: extract per-(parent, child) blocks, λ, decay lengths, loss table, for PHYS and REF |
| 0.2 | `kernels_gen.py`: parallel, resumable, atomic-write generation of angular histograms; moments and delta-M |
| 0.3 | `kernels_decay.py` |
| 0.4 | Run the generation: 10^5 events per node |

- G0.1 `int_m` and `dec_m` rebuilt from the blocks equal MCEq's to 1e-12 relative.
- G0.2 Two-body decay kernels match MCEq's decay blocks within 0.5%; three-body differences
  (flat Dalitz, a known approximation [m]) are reported, not gated.
- G0.3 Generator c_0 against MCEq's c_0: reported. This is the known generator-versus-database
  difference of the MC closure ladder, not an error of the solver.

Compute: about 55 core-h [e]: about 520 SIBYLL nodes at 1845 events/s batched [m] (8 core-h),
about 190 DPMJET nodes 16× slower [m] (46 core-h).
Disk: under 1 GB on `/home/pigrange/mc3d_work`.

### C1. Field off, azimuthally symmetric: f(h, μ, E)

| WP | Content |
|---|---|
| 1.1 | `grid.py`, `rays.py` (straight lines in a spherical shell) |
| 1.2 | `solver.py`: energy march, sweeps, source synthesis; `assemble.py` for the symmetric case |
| 1.3 | Field-free MC reference: `--no-cutoff --no-bend-primary --no-muon-field`, about 10^7 primaries, same composition as the solver run |
| 1.4 | Gate run and independent review |

- G1.0 JAX set-up, tested before any physics gate: 64-bit enforced by a test; measured CPU
  use inside the pinned cores; `jax.jvp` of the straight-ahead solve with respect to one
  kernel block agrees with a finite difference to 1e-6 relative.
- G1.1 Straight-ahead limit against MCEq, four neutrino species and muons at the ground,
  0.1 to 100 GeV, cos Z = 1, 0.75, 0.5, 0.25, 0.15, 0.05: within 0.3% with energy loss off
  in both, within 2% with it on (MCEq's loss scheme is a finite difference, ours is not).
- G1.2 Doubling the altitude grid, the μ grid, L and the ray sampling, one at a time, moves
  no gate cell by more than 0.5%.
- G1.3 Field-free 3D/1D ratio against the MC's s/t, 0.1 to 10 GeV, all cos Z rows, four
  species: at least 95% of cells within max(2%, 2σ_MC), and no row mean below 1 GeV off by
  more than 1%. The rows cos Z 0.05 and 0.15 below 0.4 GeV (engine v4 8% high) and the band
  0.10 to 0.16 GeV (engine 5 to 9% low) are reported individually.
- G1.4 Two kernel sets from disjoint halves of the events give fluxes within 0.5%. If not,
  the failing nodes are regenerated at 10^6 events (up to about 550 core-h more [e]).
- G1.5 Measured cost per solve, target ≤ 10 core-min and ≤ 8 GB.

Compute: MC reference 30 to 45 core-h [e] (2.8 to 4.6 core-h per 10^6 primaries [m]);
existing field-free products serve as a first look: the `runs/mcup` closure (4.43×10^6
primaries, horizon patch only) and `a1ncnbnomu` in `runs/m3c` (10^6 primaries).

**Decision point 1.** G1.1 or G1.3 failing after one diagnosis cycle stops the phase.

### C2. Azimuth and the cutoff sky, no bending: f(h, μ, φ, E) on a grid of locations

| WP | Content |
|---|---|
| 2.1 | `boundary.py`: cutoff skies on the location grid; nucleus groups |
| 2.2 | Solver in (μ, φ); `assemble.py` with location interpolation and the far side |
| 2.3 | MC reference at Kamioka: cutoff on, `--no-bend-primary --no-muon-field`, about 4×10^6 primaries |
| 2.4 | Gate run and review |

- G2.1 Transmission set to 1 everywhere reproduces C1 within 0.2%.
- G2.2 Straight-ahead with cutoff reproduces MCEq with a cut primary spectrum per direction
  within 1%.
- G2.3 Against the MC: F in at least 95% of cells within max(3%, 2σ_MC); East-West
  amplitude a1/a0 per energy band and row within 2σ_MC; horizontal/vertical within 2%.
- G2.4 Halving the location spacing, and moving the cutoff reference altitude, each move F
  by less than 0.5%.

Compute: MC about 60 core-h [e]; cutoff skies 10 to 30 core-h [e]; solves 10 to 20 core-h [e].

### C3. Bending of muons and uncollided primaries

| WP | Content |
|---|---|
| 3.1 | `rays.py`: arcs in the local field, per charge and rigidity bin, stepping scheme of `bend.py` |
| 3.2 | Gate run against existing MC products (production B, the v2k/v2l horizon rows, `runs/mcup`); no new MC |

- G3.1 Field set to zero reproduces C2 within 0.2%.
- G3.2 Single-muon check: bend per lifetime (5.35°, energy independent) and sign convention
  as fixed in `tools/mc3d/test_m3b.py`.
- G3.3 Per-species F and the charge-dependent East-West split against the MC: within
  max(2σ_MC, 20% of the MC split). The engine reaches about 10% of that split [m].
- G3.4 Up-going horizon rows against the MC rows of v2l within 2σ_MC.

Compute: about 50 core-h [e].

**Decision point 2.** With C3 passed the solver matches the MC everywhere we can test.
The PI decides whether to proceed to derivatives and production.

### C4. Derivatives

| WP | Content |
|---|---|
| 4.1 | Real parameter → yield map: DDM spline-knot variations through `MCEq.ddm` (`DataDrivenModel`, present in our MCEq [m]); GSF parameters as primary-flux variations |
| 4.2 | `deriv.py`: `jax.jvp` through the solver, one tangent per parameter; kernel tangents ∂K/∂θ enter as arrays built in numpy, so the parameter map is not ported |
| 4.3 | Angular-shape parameters: a width scale on a_l per projectile class, sized to the NA61 ±12% |
| 4.4 | One adjoint map (`jax.vjp`): which projectile energy, secondary energy and angle drive a sub-GeV horizon cell |

- G4.1 In the straight-ahead limit, ∂lnΦ/∂θ for all 24 parameters against daemonflux's
  shipped Jacobian splines above 5 GeV: within 10% of each Jacobian's peak. This is the test
  the Barr-style proxies failed for K±_31G [m].
- G4.2 Tangent against finite difference within 2%; second step size confirms linearity.
- G4.3 Cross-check against the sensitivity study [m]: largest σ_F 1.54% for the 24
  parameters; a ±12% cone width moves F by 5.4% at the down-going horizon, 0.1 to 0.25 GeV.
  Differences are explained, not gated.

Compute: 26 parameters × about 100 locations × 10 min ≈ 430 core-h [e].
Risk: if the knot definitions used for daemonflux are not recoverable from `MCEq.ddm` and
the package data, the map needs the authors. That contact is the PI's.

### C5. Production and daemonflux integration

| WP | Content |
|---|---|
| 5.1 | Global location grid, spacing set by G2.4; Kamioka table through `build_table_v2.py` with a new source code |
| 5.2 | `Tabulated3DModel`: optional arrays `dfactor_dparam` (S, nE, nC, nA, nP) and `param_names`; `_FluxEntry` applies F(θ) and adds Φ_1D × ∂F/∂θ to the Jacobian |
| 5.3 | Honda/Bartol comparison with the existing scripts |

- G5.1 The existing table gate suite passes on the solver table.
- G5.2 No table attached: bit-identical 1D output. Table without derivative arrays:
  identical to today's behaviour.
- G5.3 `Flux.error()` with the derivative arrays against brute-force re-solves at shifted
  parameters: within 5%.

Compute: global map 800 to 1300 core-h [e] (2600 locations at 20 to 30 core-min each in
plain numpy; the theory note's lower figure assumed compiled code), to be replaced by the
C3 measurement.

### C6. Later, not planned in detail

Muon intensities at altitude from the same solve, for a low-momentum muon calibration.

## 5. Validation ladder

| Limit of the solver | Reference | Exists? |
|---|---|---|
| Straight ahead, no field | MCEq | yes |
| Angles on, no field | MC, all three switches off | partly; C1 adds statistics |
| + cutoff | MC, bending off | no; C2 runs it |
| + bending | MC production B, v2k/v2l rows, mcup | yes |
| Derivatives, straight ahead | daemonflux Jacobian splines | yes |
| Full product | Honda, Bartol | yes |

## 6. Budget [e]

| Stage | Core-hours | Calendar |
|---|---|---|
| C0 | 60 | 3 to 5 days |
| C1 | 50 (up to 600 if G1.4 fails) | 1 week |
| C2 | 100 | 1 to 2 weeks |
| C3 | 50 | 1 to 2 weeks |
| C4 | 430 | 1 to 2 weeks |
| C5 | 800 to 1300 | 1 to 2 weeks |

Calendar figures assume the pace of Phase 2 and are the least certain numbers here.
All runs obey the standing limits: 24 cores, 35 GB, outputs under
`/home/pigrange/mc3d_work/runs/native3d/`, existing products never overwritten.

## 7. Risks and stop rules

| Risk | Sign | Response |
|---|---|---|
| Same-bin coupling or energy-loss scheme misses MCEq | G1.1 fails | One diagnosis cycle, then stop |
| Angular grid cannot hold the horizon | G1.2 fails | Finer μ grid near 0; if cost exceeds 10× target, stop |
| Kernel statistics | G1.4 fails | Regenerate failing nodes at 10^6 |
| Homogeneity assumption | G2.4 fails | Finer location grid; if no convergence, stop |
| Parameter map not recoverable | G4.1 fails | PI decides on contacting the authors |
| Solver matches the MC but costs far more than estimated | G1.5, C3 timing | Re-plan C5 with a coarser grid or fewer sites |
| JAX on CPU too slow or too memory-hungry | G1.5 fails | Ray integration in numba with hand-written tangents, or C5 on a GPU |

A failed gate is reported as failed. No gate is redefined after its run without the PI.

## 8. Decisions (answered by the PI, 2026-11-18)

1. C0 and C1 (about 110 core-h, 1.5 to 2 weeks), with decision point 1 after: **approved**.
2. Branch: new branch `phase3-cascade3d` on the fork, same commit rules as Phase 2
   (checkpoint commits by the project manager only, no force-push): **approved**.
3. Framework: JAX for the solver core, installed by pip into
   `/home/pigrange/mc3d_work/pylib_jax`: **approved**.
4. Contact with the MCEq / daemonflux authors: **no contact**.
5. GPU: lxplus GPUs may be used if needed (relevant for the global map in C5).
