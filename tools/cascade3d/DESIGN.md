# cascade3d solver design (stage C1, work packages 1.1 and 1.2)

Status: v1, 2026-11-18. Kept current with the code. Physics conventions: `PLAN.md`
section 2; MCEq conventions: `kernels_mceq.py` docstring. Tags: **[exact]** no
discretisation error beyond floating point, **[approx]** controlled approximation
(knob named), **[choice]** convention.

## 1. Unknowns and units

For species s and MCEq energy bin k: `f_s,k(node)` = dPhi/dE_kin at the bin centre
[1/(GeV cm2 s sr)], the same quantity as MCEq's state vector, at the nodes of section 2.
mu = cos(angle between direction of motion and local downward vertical); mu > 0 is
down-going. Field off, azimuthal symmetry (C1).

Species roles (from the kernel set): **hadron** (transported, no continuous loss),
**loss** (transported with continuous loss: the six helicity muons), **sink** (not
transported: neutrinos; only their emissivity is accumulated). Species in none of
these (gamma, MCEq tracking species) are dropped.

## 2. Geometry (numpy, computed once: `grid.py`, `rays.py`)

* Sphere radius R = 6391 km, top 112.8 km (MCEq's `r_E`, `h_atm`); density CORSIKA
  BK_USStd with MCEq's layer rule (`rho = b/c exp(-h/c)`, layer i when h > hl[i];
  constant 1e-9 g/cm3 above 100 km). Any `Atmosphere` object with `density(h)` and
  `boundaries` can replace it (the toy test uses an exponential atmosphere).
* **Levels** h_0 = 0 < ... < h_top, always containing the layer boundaries. Default
  0.4 km to 50 km, 1 km to 70 km, 3 km to the top (`LevelSpec`).
* **Line family.** A straight line keeps its impact parameter b. Ground lines:
  b = R sqrt(1 - mu0^2) for a set of ground directions mu0 (default 48 nodes, half
  uniform in cos Z and half uniform in zenith angle, plus the G1.1 directions and the
  cos Z edges 0.1 ... 1.0). Tangent lines: b = R + h_t for every level h_t (default)
  plus optional extra tangent altitudes between levels ("pass-through" tangents, used
  when the line family is doubled).
* **Nodes**: the crossings of every line with every level it reaches, on the inward
  (down-going, mu > 0) and the outward (mu < 0) branch; a tangent line whose tangent
  altitude is a level has one node there with mu = 0. Every node sits on a level, so
  angular moments are sums over nodes of one level.
* **Chains**: each line is one chain of nodes in the direction of motion: top (inward)
  -> ground or tangent -> top (outward). Ground lines contain a *break* between the
  inward ground node and the outward ground node (transmission 0: ground absorbs,
  nothing comes up through it). Top inward node: boundary condition.
* **Segments** between consecutive chain nodes carry `n_sub` sub-intervals equally
  spaced in the line parameter t (distance from the point of closest approach);
  grammage and length of each sub-interval by 8-point Gauss-Legendre in t [exact to
  ~1e-12]; density, altitude, local mu at every sub-point. A pass-through tangent
  segment goes from the inward node at the lowest level above h_t to the outward node
  at the same level.
* **Loss points**: for loss species each segment's sub-intervals are split further so
  that no step exceeds `dX_max` grammage (default 10 g/cm2). The loss-species unknowns
  live on these points (nodes are a subset). Lines are grouped into length buckets for
  the scan.
* Angular quadrature at a level: trapezoid in mu over that level's nodes, sorted, on
  [-1, 1] (mu = +-1 always present: the vertical ground line). Moments
  F_l(level) = sum_n w_n P_l(mu_n) f_n.

The ray table is kept general (per segment: upstream node, downstream node, sub-point
arrays) so that bent lines (C3) can reuse it with interpolation weights.

## 3. Kernel set (`kernelset.py`): the solver's only physics input

One `KernelSet` (plain numpy arrays, npz-serialisable, no pickles):

| field | shape | meaning |
|---|---|---|
| `e_grid`, `e_bins`, `e_widths` | nE, nE+1, nE | kinetic energy grid [GeV] (uniform in ln E) |
| `species`, `pdg`, `mass`, `charge` | nS | species table |
| `role` | nS | 0 hadron, 1 loss, 2 sink |
| `active` | nS x nE bool | bins in which the species exists (transported or emitted); elsewhere f = 0 |
| `inv_lambda` | nS x nE | 1/lambda_int [cm2/g] |
| `inv_dec` | nS x nE | 1/decay length [1/cm] (0 for stable or folded bins) |
| `dedx` | nS x nE | continuous loss rate beta = -dE/dX > 0 [GeV cm2/g]; 0 = none |
| `phi_top` | nS x nE | isotropic down-going intensity at the top [1/(GeV cm2 s sr)] |
| `ch_kind`, `ch_parent`, `ch_child` | nCh | 0 interaction, 1 decay; species indices |
| `Y` | nCh x nE x nE | yield [m, k] (child bin m, parent bin k), MCEq's convention: Y = n_mk w_k / w_m, *without* the -delta and *without* the rate |
| `ang_channel` | nAng | indices into the channel list that have an angular law |
| `n_ang` | scalar nA | angular laws given for parent (and child) bins k < nA only |
| `g` | nAng x nA x nA | forward fraction [m, k] in [0, 1] |
| `a` | nAng x (L+1) x nA x nA | reduced Legendre moments [l, m, k], a_0 = 1 |

Physics of a channel (parent p, bin k -> child c, bin m), rate r = rho/lambda_p for
interactions and 1/d_p for decays (per cm):

    source_c,m(x, Omega) = r Y[m,k] { g f_p,k(x, Omega)
        + (1 - g) sum_l (2l+1)/(4 pi) a_l
                  int dOmega' P_l(Omega.Omega') f_p,k(x, Omega') }

i.e. the angular kernel is g delta + (1 - g) sum_l (2l+1)/(4 pi) a_l P_l(cos theta),
normalised to 1. Channels not in `ang_channel`, and every parent bin k >= nA, are
straight ahead (g = 1). The straight-ahead set is the same code path with nAng = 0.

Builder obligations: (1) MCEq's `int_m` block (p, c) equals
`inv_lambda[p, k] * (Y[m, k] - delta_pc delta_mk)` and `dec_m` likewise with `inv_dec`;
(2) instantly decaying (folded) mesons are already folded into Y and their angular law:
the solver only sees "parent bin k gives child bin m with this yield and this law";
(3) `active` is False where a species is folded (MCEq: below `mix_idx`) and Y has no
rows there; (4) up-scatter entries (m > k, child bin above parent bin) are used for sink
children (a neutrino can carry more than the parent's kinetic energy: up to 91% of a
0.03 GeV muon's antinumu yield; the solver keeps the sink emissivity open until the end
of the march) and ignored for transported children (in MCEq only antinucleon channels
have them: annihilation energy, up to the whole n-bar -> pi yield at 7 GeV, whose
effect on the G1.1 quantities is measured in `WP12_REPORT.md`); (5) delta-M: from
raw moments b_l (l = 0..L+1) the standard choice is g = b_{L+1},
a_l = (b_l - g)/(1 - g) (clip g to [0, 1)); any other split is allowed as long as
the kernel above is the intended one. The solver's angular grid must resolve the
reduced law (section 7).
`KernelSet.validate()` checks shapes, ranges and a_0 = 1.

`from_mceq(population, loss)` builds the straight-ahead set from `kernels_mceq`;
`synthetic_angular(ks, ...)` attaches an analytic angular law (pT-like Gaussian in
theta, moments by quadrature) to a set for tests; `toy_*` sets serve the 3D toy test.

## 4. Transport (`solver.py`, JAX, float64)

**Energy march.** Bins from the highest to the lowest. Energy only goes down; the
solve is one pass. Everything a bin needs from higher bins is accumulated in
`S_acc[c, m, node]` (forward sources per unit length) and, for angular channels,
`M_acc[c, m, level, l]` (Legendre moments of the non-forward source). After a bin is
final, its contribution to all lower bins is added ("push"); the add is linear, so
reverse mode stores only each bin's slice. The loop over bins is Python around jitted
per-bin functions; kernels are arguments (jvp/vjp work through them).

**Segment formal solution** [exact given the source representation]. On a
sub-interval with optical depth x = Lambda dX + D ds, the source function J = S/sigma
is taken **exponential** in optical depth (ln J linear in tau):

    f_out = f_in e^-x + ds S_in r_in e^-x E(y),   y = x + ln(J_out/J_in),
    E(y) = (e^y - 1)/y,  r = sbar/sigma,  sbar = x/ds      (y > 0: equivalent form
    ds S_out r_out E(-y), no overflow)

Exact in the thick limit (f_out -> S_out/sigma_out: 0.1 GeV pions, low-energy muons),
in the thin limit (exact integral of an exponential source), for any source
exponential along the step at constant sigma (decay of a thin particle fed by an
attenuating parent) and for rho-proportional sources at sigma proportional to rho.
If an end value is <= 0: J linear in optical depth (phi0/phi1 weights). sigma = 0
(stable, below threshold, neutrino line-of-sight) is the thin limit. History: J linear
in tau (the first version) left loss-off muons below 0.4 GeV at cos Z 0.25 and 0.05
3% to 560% high against MCEq's step-converged value (`WP12_REPORT.md`); the convexity
of exponential sources on long near-horizon steps was the cause.

**Source between nodes** [approx; knobs: levels, n_sub]. A segment knows its source
at its end nodes a, b and at a third chain node c (next downstream, else previous).
At sub-points it is `rho(x) * exp(q(u))` with q the quadratic in u through
(0, ln A_a), (1, ln A_b), (u_c, ln A_c), A = S/rho per unit grammage, u the grammage
fraction from a: exact for rho times an exponential in depth (interaction sources from
attenuating parents, decay sources from thick parents), third order otherwise.
Fallbacks: log-linear between a and b if A_c <= 0 or no third node; linear if A_a or
A_b <= 0. The branch is selected by the primal values, so derivatives are exact
derivatives of the discrete scheme. (The quadratic changed the G1.1 cells by < 0.3%;
it is kept because it is never worse and helps steep 3D sources.)

**Same-bin closure** [exact as sweeps -> infinity]. Forward self-regeneration in the
same bin (p -> p, about 1-9%) is moved into the attenuation:
sigma_eff = rho Lambda (1 - g Y_pp[k,k]) + D (1 - g Ydec_pp[k,k]). Cross-species and
non-forward same-bin terms (p <-> n, K <-> K, Lambda -> N, K_S -> pi, ...; at most a few
%) are closed by `n_sweeps` (default 6) Jacobi sweeps over all hadrons at once; the
same-bin part of the source is interpolated linearly in grammage (it is small). The
relative change in the last sweep is returned per bin (gate: < 1e-4). Loss species
are solved after the hadrons of their bin (they never feed hadrons); sinks last. A
kernel set with same-bin couplings among loss species is rejected.

**Continuous energy loss** (loss species only) [approx; knobs: dX_max]. Along a
loss step of grammage dX the particle that arrives at E_m left at E* with
R(E*) = R(E_m) + dX, R the CSDA range from `dedx` (ln beta linear in ln E between bin
centres). psi = beta f is conserved along the characteristic without sources, so

    f_m(p) = [ psi(E*, p-1) e^-x + ds corr ( (beta S)(E*, p-1) (sbar/sigma*) phi0
             + beta_m S_m(p) (sbar/sigma_m) phi1 ) ] / beta_m,

x = ds (sigma(E*) + sigma(E_m))/2. psi(E*) and (beta S)(E*) at the upstream point:
cubic Lagrange in ln E of **ln psi** and ln(beta S) over bins >= m (one-sided at the
lowest stencil position; linear in f E^2 if a stencil value is <= 0). Bins above m are
kept in a circular window of W bins at all loss points (W from the largest shift).
`corr` = (3-point Gauss-Legendre along the characteristic, source interpolated in
energy at both ends) / (trapezoid of the same): at 0.03 GeV one 10 g/cm2 step changes
the energy by ~70%, and a steep source is then far from linear along the step. Because
the bin-m value enters the stencil, the recursion is nonlinear in f_m and is a
`lax.scan` with a nonlinear body. With dedx = 0 the scheme is exactly the hadron
scheme on the loss points. Derivatives w.r.t. `dedx` are not provided (setup data).

History (kept because it is the G1.1 loss-on diagnosis): the first version
interpolated f E^2 linearly in the cubic weights and used a trapezoid along the
step; against an exact solution with production high up (`gates/loss_analytic.py`,
`LOSS_LAM=0.01`) it was +46% at cos Z 0.05 below 0.2 GeV (a ~1% bias per bin traversed,
accumulated over 25 bins of slowing down), while MCEq's finite-difference scheme was
within 0.5%. Interpolating ln f fixed the constant-beta case; interpolating ln psi
instead of ln f (f inherits the kinks of the piecewise-log-linear beta) fixed MCEq's
dedx (3.4% -> 0.2%).

**Non-forward sources** (angular channels, parent bins < nA). Moments of the parent at
each level, multiplied by r Y (1-g) a_l and accumulated into `M_acc`; synthesised at
nodes as sum_l (2l+1)/2 M_l P_l(mu). With nAng = 0 none of this is compiled.

**Sinks**: their emissivity receives every child bin, including up-scatter, and is
read once at the end of the march. **Push**: in blocks of 16 child bins; a block above
k + (largest sink up-scatter distance, 7 bins for PHYS) is skipped with `lax.cond`.

**Neutrinos** (`assemble.py`): not transported. Intensity at any node = line-of-sight
integral of the emissivity upstream along its chain (trapezoid on the sub-points of the
same interpolated source; the ground break resets it: no far side in C1). Cell averages:
log-log-linear in E between bin centres integrated over the band, trapezoid in cos Z
over the ground lines inside the bin (edges are lines).

## 5. What is exact and what is approximate

Exact: geometry (Gauss-Legendre grammage, MCEq's density rule); straight-ahead lines
decouple, each is a 1D cascade on its own slant-depth profile; attenuation, decay and
same-bin self-regeneration on every sub-interval; thick and thin limits. Approximate:
the source shape between nodes (levels, n_sub); same-bin cross terms (sweeps); angular
quadrature and truncation (line family, L); continuous loss (dX_max, cubic stencil);
up-scatter entries dropped. MCEq itself differs by its Euler step, its finite-difference
loss operator and its one-step overshoot of the ground (<= 0.04% in X).

## 6. Deviations from the guidance, and why

1. **Up-scatter is not negligible for neutrinos.** The guidance allowed dropping child
   bins above the parent bin (MCEq's largest such block entry is 2e-4, but that is
   small only because decay rates are small). Relative to the yield, up-scatter is up
   to 91% of a 0.03 GeV muon's antinumu yield; dropping it made nu fluxes 10-22% low at
   0.07-0.45 GeV at every angle. Sinks therefore keep all child bins and their
   emissivity is read at the end of the march. For transported children only
   antinucleon channels have up-scatter (annihilation energy); it is dropped, and
   MCEq's own march without it changes the G1.1 quantities by at most 8e-5 (loss off,
   `gates/euler_refine.py`, "noup").
2. **Source function exponential, not linear, in optical depth** (section 4): linear
   left loss-off muons below 0.4 GeV 3% (cos Z 0.25) to 560% (cos Z 0.05) above MCEq's
   step-converged value; the exponential form is exact in the same two limits.
3. **Source shape between nodes**: log-quadratic per grammage through three chain nodes
   (log-linear in the first version); small effect, never worse.
4. **Same-bin closure**: Jacobi over all hadrons at once (one vectorised transport per
   sweep) instead of Gauss-Seidel in species order; the couplings are <= a few % and 6
   sweeps leave a residual of 4.5e-7. Self-regeneration is in the attenuation.
5. **Continuous loss**: as recommended, along the characteristic, but on a finer point
   set than the nodes (steps <= dX_max), with psi = beta f interpolated in log, and a
   Simpson correction along each step; the first, simpler version was wrong by up to
   46% in the slowed-down-muon regime (section 4, "History").
6. **Tangent lines**: one per level (the tangent point is then a node with mu = 0) plus
   optional pass-through tangents between levels (used by the doubled family).
7. **G1.1 is solved on the six G1.1 lines alone**: in the straight-ahead limit lines do
   not couple; G1.2 part A verifies the equality with the full family.

## 7. Known limitations (from G1.1 / G1.2, 2026-11-19)

1. **Angular resolution near the vertical must match L.** Above cos Z ~ 0.5 the only
   directions are those of the ground lines (default 24 uniform in zenith angle,
   0.065 rad). A reduced law with structure finer than that (synthetic pT 0.35 GeV at a
   few GeV after delta-M at L = 32) is aliased: doubling the ground lines moves vertical
   cells by -8%, doubling L by +30% (G1.2 B). Requirement for builders and grid: the
   reduced law must be resolved by nodes spaced <= pi/(3L) in angle near the vertical,
   or g must take the unresolved part forward. Not yet changed in the defaults.
2. **Steep sources on long near-ground steps of inclined lines** (loss-off muons below
   0.4 GeV at cos Z <= 0.25: up to +31% against MCEq's step-converged value). Converges
   with finer levels; physical (loss-on) muons and all neutrinos are within 1% there.
3. **Cost**: 9.8 core-min for the default C1 angular solve (target 10); the push into
   lower bins and the loss-species transport dominate (`gates/profile_step.py`).
4. **Reverse mode** not exercised; the nonlinear pieces will need `jax.checkpoint`.
