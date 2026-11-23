# cascade3d, stage C2: design note and cost model (draft for the manager)

Status: draft, 2026-11-20. Design only: no C2 solver, boundary or assembly code exists.
Written before any C2 solve; the Monte Carlo reference of G2.3 (`c2/mcref/`) was not
opened, nor any other Monte Carlo tally. Tags: **[m]** measured (by me today, file
given, or recorded in this project), **[e]** my estimate, **[d]** design choice.
Study files: `$N3/c2/design/` (`lin_test.py`, `lin_diag.py`, `lin_diag2.py`,
`guard.py`, `trace_cost.py`, `logs/`, `out/`, `traces/`).

**The one finding that changes the plan.** The exact tangent (`jax.jvp`) of the frozen 3D
scheme is unstable at low energies: on a reduced geometry it is wrong by up to 10^17 in the
cells below 1.6 GeV, while it is right to 5e-5 above 2.5 GeV and right to 1e-3 everywhere
in the straight-ahead limit [m] (section 1.6). The cause is the log forms of the scheme
applied to the tiny, partly negative up-going fields left by the truncated Legendre
synthesis. A guarded tangent fixes it in the tests I could afford [m]. This matters for C2
(the chosen design uses tangents) and **for C4** (whose plan is `jax.jvp` of the 3D solve:
G1.0(c) checked the jvp only in the straight-ahead limit).

## Symbols (defined once)

- P, Q: points on the ground sphere (radius 6371 km). A *location* is a node of the
  location grid; a *ground point* is any point of the cap.
- h: altitude; μ: cosine of the angle between the direction of motion and the local
  downward vertical (μ > 0 down-going, as in C1); φ: arrival azimuth, the compass
  direction the particle comes from, from north through east, in the local frame of the
  point considered (the convention of `tools/mc3d/scoring.py`).
- ℓ: a chain, i.e. one straight line of the frozen C1 line family (57 ground lines, 170
  tangent lines [m]); k: an energy bin of the kernel set (52 bins, ten per decade, lowest
  edge 25 MeV [m]); x ∈ {p, n}: the nucleon species that carry a boundary; g: a nucleus
  group of the primary model (p, He, CNO, MgSi, Fe; A/Z = 1, 2, 2, 2.08, 2.15).
- T: Honda's verdict of `tools/mc3d/cutoff.py` (1 allowed, 0 forbidden); T̄: its average
  over an energy bin, weighted with the group's spectrum.
- B(ℓ, φ, k, x): boundary intensity at the top node of chain ℓ attached with azimuth φ.
  B⁰: the C1 boundary (`phi_top` of the kernel set, the same for every chain and azimuth).
  D = B⁰ − B ≥ 0: the deficit.
- m = 0 … M: azimuthal Fourier order; D_m^c, D_m^s: cosine and sine coefficients of D.
- L: Legendre order of the kernel sets (32).
- C1: the cost of the frozen C1 solve: 12.48 core-min cold, 12.09 warm, 5.48 GB, 4:20
  wall on 4 cores [m] (`c1/close/g13/g15/G1.5.md`).
- 𝓙: the tangent map from a boundary perturbation to the change of every output of the
  solve, at the C1 state; 𝓙̃: its guarded version (section 1.6).
- c_t: marginal cost of one extra tangent per energy step, as a fraction of a primal step.

## 0. The design in five lines

1. Azimuth by Fourier modes about the local vertical. The problem is linear and the
   operator commutes with rotations about the vertical, so each order m is a separate
   problem on the C1 chains, with the associated-Legendre moments of order m.
2. Φ_cut = Φ_C1 − 𝓙̃[D]: the frozen C1 solve (T = 1) is the base for every location; the
   cutoff enters as the guarded tangent response to the deficit, one tangent field per
   (order, cos/sin, boundary), all tangents of a process sharing one primal march. The C1
   code path is not touched; with T = 1 the C2 result is the C1 result bit for bit.
3. The tangent step is assembled from `jax.linearize` of the C1 step's pieces, with the
   guarded derivative rules of section 1.6 and the mode-m moment operators. The kernel sets
   are used as frozen; nothing new is needed from them.
4. 𝓙̃ is exactly linear and does not depend on the location, so a cap average of fluxes is
   𝓙̃ of the cap average of the deficits: G2.3, G2.3(g), G2.4 and G2.5 need about thirteen
   boundaries in all, independent of the number of locations.
5. Skies: Honda's verdict (`cutoff.trace_gated`) on a rigidity ladder for about 40 line
   nodes × 36 azimuths at every location of a 5° local grid (2.5° for G2.4 a), averaged
   over each energy bin with the group spectra.

Fallbacks, in order: (i) the guard tuned per piece, if the full-geometry linearisation
test of 1.6 fails its acceptance; (ii) azimuth nodes (φ-chains) with the nonlinear C1
scheme on every chain (no tangent at all): about 2M + 1 C1 per boundary instead of about
0.33 (2M + 1) C1 plus a primal share, about 20 GB per process with copy-batched
transport, and the cap average then only approximately linear (section 5).

## 1. Formulation

### 1.1 Symmetry, and what is conserved along a straight line

The horizontally homogeneous model (`THEORY_NOTE.md` section 3): at a point Q the unknown
is f(h, μ, φ, E), and every node of every chain at level h is treated as being above Q.

- A straight line stays in the plane through the Earth's centre that contains it. Its
  impact parameter is conserved (C1), and so is the angle between its direction and any
  frame parallel-transported along it: in that frame its azimuth is constant.
- Not conserved: μ (the local vertical turns along the line: the shell effect C1 has);
  the compass azimuth (the local north turns: meridian convergence, about Δλ sin(latitude),
  2.6° over 400 km of track at 36° N [e]); the altitude.
- Streaming in the spherical shell, attenuation, decay, continuous loss and the collision
  kernels (functions of Ω·Ω' only) commute with every rotation about the vertical through
  Q; such a rotation maps chains attached at Q onto chains attached at Q, and the chain
  geometry does not depend on φ. So a boundary e^{imφ} b(ℓ, k) gives a solution e^{imφ}
  f_m: each m is an independent problem on the same chains.
- Only the boundary depends on φ. It is computed from the true three-dimensional state of
  each line (section 2), so a chain's own primary has no frame error; the identification
  of each chain's transported frame with Q's frame affects only scattered contributions
  (a few degrees, against 30° bins [e]).

### 1.2 φ nodes against Fourier modes

| | φ nodes (N_φ copies of every chain) | Fourier modes m = 0 … M |
|---|---|---|
| Coupling | all copies coupled at every bin through the moments | orders independent |
| Scheme | the nonlinear C1 scheme on every chain | needs a linear scheme: mode fields change sign |
| Cost per boundary | about N_φ C1 (13 C1 for N_φ = 13) | 2M + 1 tangents at about 0.33 C1 each, plus one primal per process (4 to 8 C1 for M = 6) |
| Memory | carry and transients × N_φ in one process | +1.5 GB per tangent [m, scaled] |
| Cap average by one boundary | approximate (scheme nonlinear) | exact (tangent linear) |
| Sky rotated by 30° (G2.5 b) | exact permutation only if 30° is a multiple of the spacing | exact phase rotation |

**Choice [d]: modes, as tangents of the C1 scheme.**

### 1.3 Collision source with delta-M in order m, from the existing kernel sets

A channel's kernel is g δ(Ω − Ω') + (1 − g) Σ_l (2l + 1)/(4π) a_l P_l(Ω·Ω') (`DESIGN.md`
section 3). By the addition theorem, for a field f_m(μ) cos(mφ) (or sin) the non-forward
source is cos(mφ) Σ_{l ≥ m} a_l c_lm P_l^m(μ) ∫ P_l^m(μ') f_m(μ') dμ', with
c_lm = (2l + 1)/2 · (l − m)!/(l + m)! (m = 0 gives C1's (2l + 1)/2). The forward part
g δ keeps the direction and is the same for every m.

- Projection at a level: F_lm = Σ_n W^m_nl f_n, W^m_nl = ∫ hat_n(μ) P_l^m(μ) dμ: the C1
  hat projection with P_l^m. For odd m, P_l^m contains (1 − μ²)^{1/2}: the quadrature is
  done in θ = arccos μ, where P_l^m(cos θ) sin θ times the hat is a smooth trigonometric
  function. Normalised P̄_l^m in projection and synthesis for stability.
- **The kernel sets carry everything needed** (g, a_l to l = 32, yields, rates, decays).
  No new kernel content and no PI decision on kernels.
- Decays (two-body, three-body, polarised muon decay at fixed helicity) are rotation
  invariant: same treatment. Continuous muon loss keeps the direction: same operator for
  every m. Same-bin sweeps: as C1, with the order-m moments in the non-forward part.
  Up-scatter passes: as C1 (two non-zero passes; the third is exactly zero because muons
  feed only sinks, `DESIGN.md` section 4).

### 1.4 The boundary in the solver

B(ℓ, φ, k, x) = B⁰(k, x) T̄_x(ℓ, φ, k), T̄_x = Σ_g w_{g,x}(k) T̄_g(ℓ, φ, k), with w_{g,x}(k)
the spectrum-weighted share of group g in the nucleons of type x in bin k
(`primaries.GSFGroups.group_split` for GSFG:GH, which reproduces GSF's p and n exactly).
B⁰ keeps the closing-cycle factor of bin 17 (rule M3), so T̄ ≡ 1 is C1. Above the highest
bin with T̄ < 1 the deficit is zero: bin 33 for protons below 60 GV [m: grid]; bins 34 to
51 carry no tangent.

### 1.5 What is assumed homogeneous, over what distance, where it shows

- Exact: the atmosphere (altitude only, the same CORSIKA BK_USStd as the Monte Carlo);
  each chain's own transmission (true entry state of that line).
- Assumed: that the field at level h is, at the horizontal position of every node of that
  level, what it is above Q. A ground line attached at Q passes over points 26 km
  (cos Z 0.5), 94 km (0.15), 222 km (0.05) and 437 km (0) from Q where it crosses 15 km
  [m: geometry]. A parent reaching a node of that line from direction Ω' is taken from the
  chain of direction Ω' attached at Q; its true line passes at a distance of order
  (horizontal distance) × (scattering angle): 100 to 300 km for horizon neutrinos below
  1 GeV [e].
- Where the error would show: rows cos Z 0 to 0.2, energies below about 1 GeV, East and
  West bins where the cutoff changes fastest with position. G2.4(b) is defined to measure
  it (section 3.4).
- The neutrino line of sight is inside its chain, i.e. it uses Q's column (same
  assumption).

### 1.6 Why a tangent, and why a guarded one

The C1 scheme is not linear in the boundary: the source shape between nodes
(log-quadratic), the source function in optical depth (exponential) and the energy
interpolation of the loss species (log-cubic, plus a geometric-mean Simpson correction)
select log forms where values are positive and fall back to linear forms elsewhere
(`DESIGN.md` section 4). The log forms are what made G1.1 and the loss scheme work.
A mode field of order m ≥ 1 changes sign, so the C1 scheme cannot solve it as it is.
Hence the anisotropic part, and the whole deficit, as the tangent of the positive C1
scheme at its own state: the tangent of a log form interpolates the ratio δf/f with the
base's shape, a consistent linear scheme. Base [d]: the T = 1 state (C1), one for all
locations, so that the tangent map is exactly linear and location-independent.

**Test 1, exact tangent, reduced geometry [m]** (`lin_test.py`: 92 levels, 35 ground and
91 tangent lines, 14 903 nodes, 252 710 loss points; pool2000 L32 set; R 6371 km). An
isotropic Kamioka-like cut (smooth step in rigidity at 11.5 GV, 15% wide, R = p for
protons and 2p for neutrons; cutoff factors K from 0.30 to 1.00 in the cells), solved
(a) with the nonlinear scheme and (b) as C1 − 𝓙[D] by `jax.jvp` of the whole solve.
Cells (4 species × 10 bands × 10 cos Z): bands from 2.5 GeV up agree to 5e-5; below
1.6 GeV the linearised cells are wrong by 10 to 10^4 (ν_μ, anti-ν_e) and up to 10^17
(ν_e, anti-ν_μ); 7.5% of the linearised emissivity values are negative.

**Diagnosis [m]** (`lin_diag.py`, `lin_diag2.py`, the six G1.1 lines, full levels):
straight-ahead, loss off: linear and nonlinear agree to 5e-4; straight-ahead, loss on:
to 1.8e-3 (muons) and 1.1e-3 (neutrino emissivity); with the angular kernels: the
emissivity tangent is wrong by up to 10^6. The growth starts at **up-going** (outward)
nodes at 3 to 10 km and μ ≈ −0.25 to −0.5, in the muon species, at 1 to 1.4 GeV, where
the base field is tiny (on this six-line family 11% of the transported node values of
the base are ≤ 0: the ringing of the truncated Legendre synthesis on up-going nodes); from there it reaches the down-going field
through the moments. The C1 primal is not affected (it passed G1.2 and G1.3); its
derivative is.

**Guard [d, prototype `guard.py`].** The primal is unchanged (same functions, same
values); only the derivative rule changes: where a log form combines positive values whose
ratio is below ρ (source shape, source integral), or where the four energy-stencil values
span more than a factor 1/ρ (loss interpolation), the derivative of the scheme's own linear
fallback is used. With ρ = 1e-3 the explosion is reduced only by a factor 2 to 5; with
ρ = 0.1 it is gone on the six lines (inward emissivity, node by node: linear against
nonlinear within 0.3 to 2% below 0.3 GeV, within 5e-4 above bin 13, i.e. 0.5 GeV) [m].
**Test 2, guarded tangent, reduced geometry [m]:** the same cut and geometry as test 1,
ρ = 0.1, `jax.jvp` of the whole solve with the guarded rules (`logs/jvp1_guard01.log`, `out/guard3d_cells.txt`).
All 400 cells agree with the nonlinear solve within 0.42% (largest: anti-ν_e,
0.10 to 0.16 GeV, cos Z 0 to 0.1); rms over the cells 1.4e-4 (anti-ν_μ) to 8.2e-4
(anti-ν_e); 18 cells above 1e-3, all in the two lowest bands; from 0.25 GeV up every cell
within 1.0e-4. Relative to the deficit itself (Φ_C1 − Φ_cut) the largest difference is
0.23%. The guard costs nothing: 721 core-s cold against 738 for the exact jvp, 4.6 GB.
This is the difference between two consistent discretisations of the same linear
problem, not the error of either; in G2.4 and G2.5, which compare the tangent scheme
with itself, it largely cancels.

What the guard costs: 𝓙̃ is not the exact derivative of anything, so its accuracy is
established only by comparison with nonlinear solves (test 2 on the full geometry is the
first task of WP 2.2, with an acceptance fixed in advance; section 8). The Simpson
correction of the loss step is not guarded in the prototype; it is a candidate if the
full-geometry test fails.

## 2. Boundary and cutoff skies (WP 2.1, `boundary.py`)

### 2.1 Like for like with the Monte Carlo

The Monte Carlo tests each azimuthal copy of a primary at its injection state on the
100 km sphere (`cutoff.trace_gated`: `gate`, then `trace_allowed` with IGRF-14 at
2020-01-01, re-entry rule at the injection sphere, escape at 10 R_E, the step table, the
2e8 m budget), at R = (A/Z) p_nucleon (`cutoff.rigidity_gv`), and weights the shower by
w/N_AZ × allowed. The solver's counterpart for a chain is the expectation of that verdict
over the primaries on that line in that bin, T̄_g(ℓ, φ, k) = ∫_bin J_g T(state(ℓ, φ),
R_g(E)) dE / ∫_bin J_g dE; nucleons of a nucleus share the nucleus's verdict
(superposition); the nucleon boundary is Σ_g J_{g,x} T̄_g.

### 2.2 The sky of a location

- **Attachment.** A location P has its own frame (`geomag_backtrace._local_frame`). A
  ground chain is attached at the reference altitude h_ref: the line passes over P at
  h_ref with the chain's local zenith there and compass azimuth φ. Base h_ref = 0: the
  line passes through P. A tangent chain is attached at its tangent point above P.
- **State.** From the attachment point backwards along the line to its first crossing of
  the 100 km sphere: r_inj and the inward velocity u (`trace_cost.line_entry`; checked:
  radius 100.000 km; a cos Z 0.05 line enters 7.6° from P with injection μ 0.182 [m]).
  The solver's top is 112.8 km; the grammage above 100 km is below 0.1 g/cm² on any line
  [e] and stays as in C1.
- **Lines that never reach the injection sphere** (11 tangent lines, h_t ≥ 100 km [m]):
  no Monte Carlo primary travels on them. Convention [d]: T = 0 there (effect expected
  below 1e-4 in any cell [e], reported).
- **Direction grid.** Line nodes: the ground lines at the G1.1 and G1.3 cos Z values plus
  ground zenith every 5° (about 25), and tangent nodes at h_t = 0, 1, 2, 5, 10, 15, 20,
  30, 40, 50, 60, 70, 80, 90, 99 km (15). T̄ of the other chains: linear interpolation in
  ground zenith angle, resp. in h_t, after the bin average. Azimuth: 36 sectors of 10°,
  one azimuth drawn uniformly inside each sector per (line node, rigidity sample), fixed
  seed: an unbiased sector average, which keeps the sharp azimuthal edges from aliasing
  into the low orders.
- **Rigidity ladder.** Per direction, two samples of ln R per bin width (20 per decade),
  each uniform in its half-interval, 1.7 to 60 GV (31 samples). T̄_g(k) = J_g-weighted
  mean over the half-intervals that bin k covers in R_g. Above 60 GV, T = 1 on every line
  [d] (`cutoff.gate` accepts above 60 GV only for injection μ > 0.17; the tangent lines
  below that are spot-checked in G2.0).
- **Penumbra** is sampled, not modelled: each sample is an exact verdict; bin and sector
  means are unbiased, with binomial noise of at most 0.5/√n per (line node, sector, bin)
  (n ≈ 2 per group); about 10³ boundary values feed one cell [e].
- **Table frame**: P's own; nothing is rotated when tables are combined (3.2).
- **What the solver receives**: per boundary, D(ℓ, φ_j, k, x) on the 36 sectors, then
  D_m^c, D_m^s(ℓ, k, x), m ≤ M, by the discrete Fourier sum divided by sinc(m · 5°) (the
  sector average).

### 2.3 Number of traces and their cost

States per location: 40 line nodes × 36 sectors × 31 rigidities ≈ 44 600 [d]; the exact
pre-filter removes those below 0.7 × the most favourable Störmer cutoff (6.4 GV near
Kamioka [m]), i.e. below about 4.5 GV.

Measured (`trace_cost.py`, one core, one call per set, IGRF-14) [m]:

- Sky-type states at Kamioka itself (h_ref = 0): 9 line nodes (ground cos Z 1, 0.75, 0.5,
  0.25, 0.15, 0.05; tangent h_t 5, 20, 50 km) × 36 azimuths × 31 rigidities = 10 044
  states in one call: **6.6 ms CPU per state**, 72% traced (28% decided by the
  pre-filter), 42% allowed.
- The reference's own draw (`sample_test_states`, `cap_deg=10`, R < 60 GV, 3000 states):
  **15.8 ms per state**, 44% traced, 18% allowed (consistent with the 7 to 27 ms of
  `MILESTONE3A_NOTES.md` section 6).
- Highest forbidden rung of the ladder (a proxy of the upper cutoff) at Kamioka: 9.0 GV at
  the vertical; over the 36 azimuths 16 GV (cos Z 0.75), 21 GV (0.5), 26 GV (0.25),
  29 GV (0.15), 33 GV (0.05 and the tangent lines). So at the site the deficit vanishes
  above about 35 GV (bin 31 for protons); the ladder's 60 GV top is conservative for the
  cap and for lines entering far from it.
- **Azimuthal content of the deficit at Kamioka** (`out/sky_modes.txt`; fixed azimuths, so
  penumbral noise counts as high orders): share of the azimuthal power, summed over the
  bins with their deficit, for the inclined line nodes: m = 0 0.77 to 0.89, m = 1 0.08 to
  0.17, m = 2 to 6 0.024 to 0.059, **m = 7 to 12 0.005 to 0.011, m ≥ 13 0.002 to 0.005**.
  In the bins near the cutoff edge (small mean deficit) single orders 7 to 12 reach 0.1 to
  0.5 of the mean deficit. This decides M (section 4.2).

Per location: 44 600 states × 6.6 ms ≈ **4.9 core-min** [e from m]. Tables needed:
25 (5° grid) + 56 (the rest of the 2.5° grid) + 49 (h_ref = 15 km, 5° grid over
[−15°, 15°]², for G2.4 b) = 130 → **10.6 core-h**; G2.0 about 5 core-min (below). Skies in
all **about 11 core-h** [e from m], inside the plan's 10 to 30. Calls are pooled to
≥ 10⁴ states (the bouncing tail sets the cost per call, `MILESTONE3A_NOTES.md`).

### 2.4 How G2.0 is evaluated

1. ≥ 20 000 states from `cutoff.sample_test_states` (seed fixed in the script), R < 60 GV.
   **Ambiguity**: the gate does not say whether `cap_deg` is used. With the default (whole
   40° patch) most states fall outside the location grid ("not covered"). I propose
   `cap_deg=10`: the states whose straight line meets the ground in the 10° cap.
2. Direct verdict: `cutoff.trace_gated`, pooled in calls of ≥ 10⁴ states.
3. Sky transmission of the same state: its line's ground crossing Q (tangent point for a
   line that misses the ground), local zenith and azimuth there, location weights of Q
   (3.2), T̄ at (line node, sector, rigidity half-interval of its group and R). Q outside
   the grid: not covered (counted).
4. Classes R 0-5, 5-10, 10-20, 20-60 GV × injection cos Z 0-0.25, 0.25-0.5, 0.5-1 × N, E,
   S, W quadrants. **Ambiguity**: the frame of the quadrant. I propose the injection
   point's local frame (where the state is defined).
5. Per class, mean T̄ against the allowed fraction; trigger if the difference exceeds
   max(0.02, 2σ). Reported in addition: 500 tangent-line states above 60 GV, traced.

Cost [e from m]: 20 000 × 15.8 ms ≈ 5.3 core-min, plus 500 tangent-line states.

## 3. Locations and assembly (WP 2.2)

### 3.1 What G2.3 needs

Cells of four species × ten bands × ten cos Z rows × twelve 30° azimuth bins, averaged
over the 10° cap (and, reported, the 5° cap), each ground point in its own local frame;
Φ_0 = the T = 1 solve, i.e. the C1 solve `c1/close/g13/pool2000.npz`. K, a_c, a_s and H/V
through `gates/c2_stats.py` (`solver_tallies(phi_cut, phi_0)`).

### 3.2 Location grid, interpolation, cap average [d]

- Local azimuthal-equidistant coordinates about Kamioka (x east, y north, degrees of arc;
  as `cutoff.RUCache`), nodes at multiples of Δ covering [−10°, 10°]²: Δ = 5° → 25
  locations (base), Δ = 2.5° → 81 (G2.4 a). The node (0, 0) is Kamioka: G2.5's "grid
  location nearest the site" is the site.
- The sky at any ground point Q: bilinear combination, weights w_P(Q), of the four
  surrounding tables, each read in its own frame at the same local (line node, sector).
  Second-order error: the dipole part of the vertical cutoff curves by about −10 GV/rad²
  at Kamioka, i.e. about 0.04 GV (Δ = 10°) and 0.01 GV (Δ = 5°) of interpolation error
  against a cutoff near 11 GV [e].
- **Cap average, exactly.** The flux at Q is Φ_C1 − 𝓙̃[D_Q]; 𝓙̃ is linear and the same at
  every location, so the cap average is Φ_C1 − 𝓙̃[D_cap], D_cap = Σ_P W_P D_P,
  W_P = (1/A)∫_cap w_P(Q) dQ (area quadrature on the sphere, 10⁴ points). This is the same
  as solving every location and interpolating the solutions; it costs one boundary instead
  of 25 or 81.
- The Monte Carlo's flux measure (crossings weighted 1/|cos ψ|, per area and solid angle)
  is the area-uniform average of the intensity [read: `scoring.NestedCaps`].

### 3.3 From modes to cells

Per tangent the solve returns the sink emissivity at the nodes and the transported species
at the ground nodes. Neutrino intensity: the (guarded) tangent of `assemble.sink_intensity`
at the base emissivity (that line-of-sight integral uses the same log source shape). Per
ground direction μ₀ and azimuth bin b: Φ(μ₀, b) = Φ_C1(μ₀) − Σ_m [c_m(μ₀) ⟨cos mφ⟩_b +
s_m(μ₀) ⟨sin mφ⟩_b], ⟨·⟩_b the analytic bin mean; then `assemble.cell_table` (PCHIP in
cos Z, band mean in the bin convention) on each bin's values, unchanged from C1. Ground
muons the same way.

### 3.4 The two variations of G2.4 (proposed for `GATES.md`)

**(a) Spacing.** Base Δ = 5° (25 locations); variation Δ = 2.5° (81 locations, containing
the 25). D_cap recomputed; one boundary. Expected to pass easily in the cap average [e].
It does not test a single-point table, where interpolation errors do not average out.

**(b) Reference altitude.** Base h_ref = 0: a line of sight belongs to the location under
its ground point. Variation h_ref = 15 km (typical production altitude): every chain is
attached at 15 km above its location, and the location of a ground cell (Q, direction) is
the point under the line's crossing of 15 km, displaced from Q towards the arrival azimuth
by 0 (vertical), 26 km (cos Z 0.5), 94 km (0.15), 222 km (0.05), 437 km (horizontal)
[m: geometry]. In both, each line's own transmission is exact (same physical line); only
the centre of the homogeneous column moves, so the variation measures the homogeneity
approximation of the scattered part. Evaluation: the displacement depends on the
direction, so D_cap becomes the cap average shifted by the displacement d, interpolated
quadratically from six shifted cap averages (d = 0, ±4° east, ±4° north, (4°, 4°)) built
from h_ref = 15 km tables on a 5° grid over [−15°, 15°]² (49 locations); each direction
uses its own d inside the cell average. Six boundaries. The stencil's interpolation error
is checked on the deficit itself (no solve) before any solve.

Not recommended: "every chain attached at 15 km above the ground point". It moves the
primary line of a horizontal cell 437 km away from the line that reaches Q, i.e. it
breaks the straight-ahead part (exact at h_ref = 0) and would fail in the horizon row
whatever the homogeneity error [e]. Cheaper alternative if wanted: h_ref = 5 km
(displacements up to 252 km), linear stencil, three boundaries.

### 3.5 The far side (G2.3 h)

An up-going neutrino with |cos Z| = c entered the Earth 2 arcsin(c) away, up to 23° for
the row −0.2 to 0 [read: `sky3d.py`]. Each up-going (row, azimuth bin) cell sees its own
far-side region, so the cap trick gives one boundary per cell: 24 boundaries, plus about 50
more locations of tables out to 33°. About 1.5 to 2 times all the down-going solves
together [e]. **Recommendation: not in C2**: (h) is reported, not gated; the reference
has the up-going tallies only if `--upgoing` was cheap; and bending (C3) matters most
there.

## 4. Cost model

### 4.1 Units [m]

| quantity | value | source |
|---|---|---|
| C1 solve, frozen geometry, cold / warm | 12.48 / 12.09 core-min, 5.48 GB | `G1.5.md` |
| executed steps of a C1 solve | 52 bins + 2 × 25 up-scatter bins = 102 | code |
| C1 geometry doubled (line family) | 28.0 core-min, 9.4 GB (base 14.8, 5.4) | `WP12_CYCLE2B_REPORT.md` |
| reduced geometry, primal (cold / warm) | 410 / 383 core-s, 3.0 GB, 4 cores | `logs/primal.log` |
| reduced geometry, jvp, 1 tangent (cold / warm) | 738 / 624 core-s, 4.9 GB | `logs/jvp1.log` |
| reduced geometry, jvp, 2 tangents (vmap; cold) | 948 core-s, 5.6 GB | `logs/jvp2.log` |
| reduced geometry, guarded jvp, 1 tangent (cold) | 721 core-s, 4.6 GB (no extra cost) | `logs/jvp1_guard01.log` |
| reduced / full geometry | 0.31 of the nodes, 0.48 of the loss points; primal 0.53 of the C1 time | |
| c_t | 0.31 (warm, 1st tangent) to 0.41 (cold, 2nd tangent) | the rows above, with the jvp's 127 steps (all three passes run under tracing) against 102 |
| memory per tangent | +1.9 GB (1st, includes the jvp's residuals), +0.7 GB (2nd) on the reduced geometry; × 2.1 (loss points) on the full one: about +4 GB, then +1.5 GB each | |
| one cutoff trace | 6.6 ms CPU per sky state (72% traced); 15.8 ms per state of the reference draw | `traces/trace_cost.json` |

### 4.2 Model

- One tangent field: t = c_t · f_k · C1, f_k = 84/102 (no tangent above bin 33):
  t ≈ 0.4 × 0.82 × 12.1 ≈ 4 core-min, range 3 to 5 [e from m].
- One process: one primal march (C1) plus n_t tangents; memory ≈ 9.5 GB + 1.5 GB (n_t − 1)
  [e from m]: n_t = 5 at 16 GB, 13 at 28 GB.
- One boundary: 2M + 1 tangents.
- **M.** A 30° bin passes order m with weight sinc(15° m): 0.53 to 0.09 for m = 7 to 11,
  0 at 12, at most 0.08 above. The Kamioka sky has 0.5 to 1.1% of its azimuthal power in
  m = 7 to 12 and 0.2 to 0.5% above (2.3) [m]; nearly straight cascades (neutrinos of 5
  to 10 GeV near the horizon) keep that content. My estimate of the truncation error at
  M = 6 in such cells: up to 1 to 2% of the flux, i.e. G2.5(a) at 0.5% would fail; at
  M = 12: a few 0.1% [e]. **Decision rule [d]**: the first solve (the G2.6 cost run,
  Kamioka location) is done at M = 12. The orders are independent, so its partial sums to
  6 and to 12 *are* the G2.5(a) doubling for M = 6, at no extra cost. If M = 6 passes
  there, the other boundaries use M = 6; if not, M = 12, and G2.5(a) needs M = 24 at
  Kamioka (24 more tangents).
- One boundary, M = 6 (13 tangents): 12.5 + 13 t ≈ 65 core-min in one 28 GB process,
  ≈ 90 core-min in three 16 GB processes: **0.9 to 1.7 core-h** with t from 3 to 5.
  M = 12 (25 tangents): **1.7 to 3.3 core-h** (two 28 GB or five 16 GB processes).

### 4.3 Table (CPU unless stated; "unit" = one boundary at the production M)

| item | units | core-h, M = 6 | core-h, M = 12 | memory per process | note |
|---|---|---|---|---|---|
| one location (G2.6) | 1 | 0.9 to 1.7 [e] | 1.7 to 3.3 [e] | 16 to 28 GB | 4 to 8 C1, resp. 8 to 16 C1 |
| G2.6 run itself (Kamioka, M = 12; also G2.5 base, (a) for M = 6, (b), (c)) | | 1.7 to 3.3 [e] | 1.7 to 3.3 [e] | | first solve with the cutoff |
| G2.5(a) M = 12 → 24, if M = 12 is needed | | 0 | 1.7 to 3.3 [e] | | |
| G2.5 μ doubled, L doubled | 1.9 + 1 | 2.6 to 4.9 [e] | 4.9 to 9.6 [e] | μ doubled ≈ 1.75 × memory [m, C1] | C1: geometry doubled 1.9 × CPU [m]; L64 ≈ L32 [m] |
| G2.1 | 0 | 0 | 0 | | assembly of the existing C1 solve |
| G2.2 | 21 straight-ahead chains + 21 MCEq runs | < 0.5 [e] | < 0.5 [e] | 5 GB | nonlinear per-chain path |
| G2.3: 10° cap and 5° cap | 2 | 1.8 to 3.4 [e] | 3.4 to 6.6 [e] | | independent of the number of locations |
| G2.4 (a) + (b) at 15 km | 1 + 6 | 6.3 to 12 [e] | 12 to 23 [e] | | (b) at 5 km: 1 + 3 units |
| linearisation test, full geometry | 1 nonlinear + 1 tangent | 0.5 [e] | 0.5 [e] | 10 GB | first task of WP 2.2 |
| **all C2 solves** | | **13 to 25** [e] | **26 to 50** [e] | | plan: 10 to 20; with (b) at 5 km: 11 to 20, resp. 21 to 40 |
| skies: 130 tables + G2.0 | | about 11 [e from m] | same | 1 core per worker | plan: 10 to 30 |
| C5 per location | 1 | 0.9 to 1.7 [e] | 1.7 to 3.3 [e] | | |
| C5, global 5° grid: about 1650 nodes, about 1450 below 60° geomagnetic latitude | 1450 | **1300 to 2500** [e] | **2500 to 4800** [e] | | plan: 800 to 1300; skies about 120 core-h more |
| C5 at 2.5° (if G2.4 asks for it) | about 5800 | 5200 to 9900 [e] | 10 000 to 19 000 [e] | | |
| C5 on one lxplus GPU | | not measured | | 40 to 80 GB: 20 to 45 tangents per process | [e]: if a C1-sized march takes 1 to 3 min wall (chain scans latency-bound), 1450 locations take 2 to 6 GPU-days at M = 6; needs a pilot and a CUDA JAX install |

### 4.4 The manager's worry, checked

"Every location needs (1 + 2M) mode solves each costing about a C1 solve."
- **Per location: right in order of magnitude, smaller in unit.** A location is 2M + 1
  tangents at about 0.33 C1 each [m-derived] plus one primal per process: 4 to 8 C1
  (M = 6) or 8 to 16 C1 (M = 12), not 13 or 25 C1.
- **For the C2 gates the worry does not apply.** They need cap averages, and the cap
  average of a linear, location-independent map is one boundary: about 13 units whatever
  the number of locations. The number of locations drives only the skies (11 core-h).
  The solves are 13 to 25 core-h at M = 6 (the plan's 10 to 20, at its top) and 26 to
  50 at M = 12 (above the plan). Which one holds is known after the first solve.
- **For C5 the worry is confirmed in substance.** A global emissivity needs one boundary
  per location: 1300 to 2500 core-h at 5° (M = 6) on CPU, 1 to 3 times the plan's 800 to
  1300, twice that at M = 12, four times more at 2.5°. GPU, a reduced basis or adjoint
  tables (section 5) are what could bring it back; none is measured.

## 5. Economies, ranked (saving / accuracy risk / effect on gates / on derivatives)

1. **Cap average by one boundary** (linearity, location-independent operator) [chosen].
   Saving 25× (G2.3), 81× (G2.4 a). Accuracy: exact for the tangent scheme. Gates: G2.3
   and G2.4 computed as defined. Derivatives: unchanged (linear).
2. **All tangents of a boundary set in one process, one primal** [chosen]. Saving: a
   tangent costs c_t ≈ 0.3 to 0.5 of a primal step [m] instead of a whole solve per mode:
   about 2.5× against separate mode solves. Accuracy: none. Gates: none. Derivatives:
   none. Limit: memory, +1.5 GB per tangent.
3. **Deficit only, tangents from bin 33 down** [chosen]. Saving 18% per tangent. Exact.
   The level offset of C1 (+1.8% above 0.4 GeV) enters K only through 𝓙̃, which is a
   fraction (1 − K) of the flux.
4. **GPU** [for C5]. More tangents per process (40 to 80 GB), wall-time gain not
   measured. No accuracy change. Needs a CUDA JAX install and a pilot.
5. **Reduced basis of deficits across locations** [C5 option]. The global set of D_P
   spans some rank r; solving r basis tangents instead of one boundary per location saves
   1450/r. r is unknown: the moving cutoff edge makes it high [e]. It can be measured at
   no solve cost from the C2 tables (SVD of the 130 location deficits). Exact within the
   span; derivatives unchanged.
6. **Fewer orders where the anisotropy is weak** (M per energy band, e.g. M = 2 above
   10 GeV primaries) [possible]. Saving up to about 40% [e]. Risk: G2.5(a). Derivatives:
   none.
7. **Adjoint tables for fixed outputs** [C5 option]: one adjoint per (output cell,
   order), then any location or site is a dot product with its sky. 400 cells × 7 orders
   ≈ 2800 adjoints at a few C1 each ≈ 1500 to 2000 core-h [e]; reverse mode is not
   exercised and needs per-bin checkpointing. Worth it only if many sites or the far side
   are wanted.
8. **Reduced L or grids for m ≥ 1**: not possible inside a tangent (same grid as the
   primal); L alone would save little (the angular operations are about 15% of a step:
   straight-ahead 710 s against 825 s [m, `WP_G13_REPORT.md`]) and break the delta-M
   consistency. Not chosen.

Chosen: 1 + 2 + 3 (and 6 if the first solve allows); 4 and 5 or 7 for C5. Fallback (if
𝓙̃ cannot be validated): φ-chains with the nonlinear C1 scheme, N_φ = 13: about 13 C1 ≈
2.7 core-h per boundary, memory
about 20 GB with the copies transported in batches inside a step (the carry is about
1.3 GB per copy [e: sizes of S_acc, windows, S_up]); the cap average then holds only to
second order in the spread of the skies (a secant test, as in 1.6, is needed); all C2
solves about 40 core-h at N_φ = 13 [e].

## 6. Derivatives

- The design keeps everything in JAX with the energy loop in Python: the explicit
  tangent is an ordinary function of the kernel arrays, so an outer `jax.jvp` (and in
  principle `jax.vjp`, not exercised, with per-bin checkpointing) goes through it. The
  guard is written as explicit functions, not as a custom derivative rule, so nested
  differentiation sees ordinary operations.
- **Warning for C4** [m]: the exact `jax.jvp` of the 3D solve is the object that failed
  test 1; for kernel directions that change the angular shape (C4 WP 4.3: a width scale on
  a_l) the same near-zero up-going fields are perturbed. C4 should use the guarded rule and
  check it against finite differences (G4.2 does exactly that); G1.0(c) does not cover it.
- Cost per parameter direction, one location, CPU [e from m]: Φ_C1 part: one guarded
  tangent, 0.33 C1 ≈ 4 core-min (batched with the others: primal shared). Cutoff part,
  exact: a second-order stream per mode tangent, about 2 × 13 × 0.33 C1 ≈ 9 C1 ≈ 1.8
  core-h. For 26 parameters × 100 locations that is 4700 core-h, far above C4's 430.
  Options for the PI at C4: (i) differentiate only the m = 0 deficit (about 0.7 C1 per
  direction: 26 × 100 × 0.7 × 12.5 core-min ≈ 380 core-h on top of the C1 part); (ii)
  assume ∂K/∂θ = 0 (∂Φ_cut/∂θ = K ∂Φ_0/∂θ), tested at one location with the full
  second-order derivative.

## 7. Risks per gate, and what I would check first

- **Guarded tangent (all gates)**: the largest risk. Check first: test 2 on the full
  geometry, isotropic cut as in test 1 and the azimuth-averaged Kamioka deficit, with an
  acceptance fixed before the run (proposal: every G1.3 cell within 0.5% of the
  nonlinear solve and within 1% of the deficit Φ_C1 − Φ_cut; the reduced geometry gave
  0.42% and 0.23%). If it fails: guard the Simpson
  correction, then ρ per piece, then the fallback.
- **G2.1** is exact by construction (D = 0 gives Φ_C1 bit for bit through the same
  `cell_table`); it can only fail on bookkeeping (azimuth bins, frames). It tests nothing
  about the azimuth. Check: the azimuth-bin assembly on an analytic mode field.
- **G2.2** (1% for neutrinos, every MCEq bin 0.1 to 100 GeV): the straight-ahead per-chain
  path, nonlinear, one chain per direction: the boundary machinery only. G1.1 had the four
  neutrinos within 0.30% with loss on [m, `G1.1.md`], so the margin is good. Risks: MCEq's
  interface for per-group initial spectra; the gate's "H3a" (G1.1's primary model) means
  the group weights of H3a, not GSFG:GH, in this gate: `boundary.py` must take both.
  Check first: T̄ ≡ 1 reproduces G1.1 exactly.
- **G2.3 (a)**: mode truncation at the horizon and at 5 to 10 GeV, where the cascade is
  nearly straight and the boundary's azimuthal edges are sharp; homogeneity in the rows 0
  to 0.2 below 1 GeV; sky noise. The +1.8% level cancels to first order in K; a level that
  depends on the parent spectrum inside a bin (the C1 candidate) changes slightly with the
  cut, i.e. a bias of order (1 − K) × 1% [e]. **(b)**: dominated by m = 1, robust. **(c)**:
  m = 0, robust, but H uses the horizon rows. Check first: the boundary's mode spectrum
  (from the tables, no solve) to see how much lies above M = 6.
- **G2.4 (a)**: low risk in the cap average. **(b)**: genuine risk in the rows 0 to 0.2
  below 0.5 GeV; that is the homogeneity question itself. Check first: the deficit
  difference between h_ref 0 and 15 km tables (no solve).
- **G2.5 (a)**: the order doubling at 5 to 10 GeV near the horizon (sharp azimuthal
  edges, little smoothing; the Kamioka sky has about 1% of its azimuthal power in
  m = 7 to 12 [m]): M = 6 would probably fail, M = 12 should pass [e] (decision rule of
  4.2). μ and L doublings moved the G1.3 cells by at most 0.16% in C1 [m], so they
  should pass; note that G2.5 uses one location, without the smoothing of the cap. **(b)**: exact by construction (phase rotation of the orders; or a
  permutation of the 36 sectors, 30° being three of them), i.e. it tests only the
  bookkeeping of the azimuth convention. **(c)**: the tangent sweeps contract like the
  primal ones (residual 1e-8 [m, C1]); the tangent residual must be output explicitly
  (the jvp of the primal residual is not it).
- **G2.0**: penumbra sampling noise per class is far below 0.02 at 20 000 states; risk in
  the class 20-60 GV × injection cos Z 0-0.25 (tangent lines, T = 1 convention above
  60 GV) and in the interpolation between line nodes near the horizon.

## 8. Implementation plan (WP 2.1, WP 2.2)

Rule: `solver.py`, `assemble.py` (existing functions), `grid.py`, `rays.py`,
`kernelset*.py` are not modified; C2 code is new modules that import them. The frozen C1
path stays bit-identical (G2.1 compares against `c1/close/g13/pool2000.npz`).

**WP 2.1 `boundary.py`** (about 2 days of agent work [e])
1. `LocationGrid(centre, spacing, half_width)`: nodes, `weights(points)` (bilinear),
   `cap_weights(radius_deg, n_quad)`, `shifted_cap_weights(radius, d)`.
2. `chain_entry_states(location, rays, sectors, h_ref, seed)`: (r_inj, u) per (line node,
   sector), never-entering flag; line-node interpolation weights for every chain.
3. `GroupWeights(model, e_bins)`: w_{g,x}(k) and the J_g-weighted half-interval weights
   (GSFG:GH and H3a).
4. `SkyTable.build(location, …, n_jobs)`: rigidity samples, pooled `trace_gated` calls of
   ≥ 10⁴ states over forked workers (≤ the agent's cores), atomic resumable npz per
   location with sha256; `SkyTable.tbar(group, k)`.
5. `deficit_modes(tables, weights, rays, ks, M)`: D on (ℓ, sector, k, x) → D_m^c, D_m^s.
6. `sky_transmission(states)`: the G2.0 interpolation for arbitrary injection states.
7. Tests: entry geometry (radius, zenith at entry against the analytic value, tangent
   lines, never-entering lines); T̄ ≡ 1 gives D = 0 exactly and the group sums give
   `phi_top`; Fourier round trip; a 30° rotation of the sectors rotates the orders
   exactly; G2.0 harness on 200 states.
8. Runs: Kamioka table first (pilot, G2.0 on it), then the 25 + 56 + 49 tables, then G2.0.

**WP 2.2 `solver3d.py`, `assemble3d.py`** (about 4 days [e])
1. `mode_weights(rays, L, m)`: W^m and the synthesis Q^m in θ-quadrature; test against
   C1's `_hat_weights` for m = 0 (to 1e-13) and against an analytic Y_lm field.
2. `TangentSolver(solver)`: holds a frozen `Solver`; per bin, `jax.linearize` of the
   pieces of the C1 step (seg source, chain scan, sweeps, loss transport, push, up-scatter)
   with the guarded rules, the order-m moment operators per tangent, per-chain top
   tangents, n_up = 2, tangents batched with `vmap`; outputs tangent emissivity, tangent
   ground fluxes, tangent sweep residual. Tests: m = 0 tangent equals `jax.jvp` of
   `Solver.solve` where the guard is inactive (straight-ahead: to 1e-10); rotation
   invariance; 3D toy (`test_toy3d.py` style) with an azimuthal boundary against an
   independent φ-chain solution of the toy; primal inside equals `Solver` to 1e-12.
3. **Linearisation test on the full geometry** (section 7), acceptance written into
   `GATES.md` by the manager before the run. Stop and report if it fails.
4. `assemble3d.py`: tangent sink intensity, azimuth-bin means, `cells_3d` → phi_cut
   (species, band, cz, az) through the unchanged `cell_table`; cap and shifted-cap
   deficits.
5. `gates/run_solve3d.py` and `gates/g2_0.py` … `g2_6.py`, one result file each.
6. Order of runs: G2.6 cost of one location (Kamioka, M = 12) → manager and PI; then G2.1,
   G2.2, G2.5, G2.4 (variations fixed in `GATES.md` before the cost run, since that is
   already a solve with the cutoff); freeze `RESULTS_C2.md`; G2.3 by another agent.

Compute of the implementation phase: tests and the linearisation test about 3 core-h
[e]; skies about 11 core-h [e from m]; solves 13 to 25 core-h at M = 6, 26 to 50 at
M = 12 (section 4.3).

## 9. Open points for the manager and the PI

1. **The guarded tangent** is a new numerical element, needed by C2 as designed and by C4
   in any design. Its acceptance test (section 7) should be fixed in `GATES.md` before it
   runs. If it fails, the fallback (φ-chains) costs two to three times more per
   boundary.
2. **G2.4(b)**: choose between h_ref 0 vs 15 km with the displaced location (six
   boundaries, my proposal), 0 vs 5 km (three), or another definition. I advise against
   "chains attached at 15 km above the ground point" (section 3.4).
3. **G2.4(a)**: 5° vs 2.5° as proposed; it says little about single-point tables (C5).
4. **G2.0**: `cap_deg` of the draw and the frame of the quadrants are not specified.
5. **G2.1** and **G2.5(b)** are satisfied by construction in this design (no-cutoff
   identity; rotation invariance); they still test bookkeeping. If the manager wants a
   non-trivial check of the m = 0 part, the full-geometry linearisation test is it.
6. **G2.2**: "H3a" in the gate means H3a groups for the boundary in this gate, and the
   straight-ahead per-chain path (not the mode path, whose truncation would show in a
   straight-ahead problem).
7. **G2.5(a)**: "doubling the φ nodes (or azimuthal modes)" read as M → 2M, evaluated
   as the partial sums of one run at 2M (exact, since the orders are independent).
   The manager should confirm that this reading is acceptable.
8. **Far side (G2.3 h)**: not in C2 (section 3.5).
9. **C4 derivatives**: the m ≥ 1 and deficit derivatives are second order and expensive;
   the PI may prefer the m = 0 or the ∂K/∂θ = 0 option (section 6).
10. **C5 cost**: 1 to 3 times the plan at 5° on CPU with M = 6, 2 to 6 times with
    M = 12; a GPU pilot (and a JAX CUDA install) before C5 is planned.
11. Conventions to confirm: T = 0 on the 11 lines that never reach the injection sphere;
    T = 1 above 60 GV; the 10 cos Z × 12 azimuth cells use the arrival azimuth of
    `scoring.py`.
12. Memory: one 28 GB process per boundary is the efficient size; with the Monte Carlo of
    G2.3 running, 16 GB processes (about 40% more CPU) are the safe default within the
    35 GB project limit.
13. **C2 budget**: skies 11 + solves 13 to 25 (M = 6) or 26 to 50 (M = 12) + tests 3,
    next to the Monte Carlo's 60: within the plan's 100 core-h only with M = 6. The
    first solve (Kamioka, M = 12, 1.7 to 3.3 core-h) decides; it goes to the PI with
    G2.6 as the gate says.
