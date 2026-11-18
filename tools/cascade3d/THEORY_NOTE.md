# Can the 3D effects live inside the cascade calculation? Theory note

Date: 2026-11-18. Paper study only: nothing was run. Each claim is tagged.

- **[fact]** published result, or a number from our own records
- **[derived]** follows from the equations by a short calculation, shown here
- **[estimate]** my order-of-magnitude figure, not a benchmark
- **[inference]** my reasoning, untested

## 1. The exact problem

For each species i, the density f_i(x, Ω, E) in position, direction and energy obeys a
linear Boltzmann equation with a Lorentz-force term:

```
Ω·∇_x f_i                                   streaming
 + (0.3 q_i B / p) (Ω × b)·∇_Ω f_i          magnetic deflection   [B in T, p in GeV/c, per metre]
 − ∂/∂E [ ρ(x) S_i(E) f_i ]                 continuous energy loss
 + [ ρ(x)/λ_i(E) + 1/d_i(E) ] f_i           loss by interaction and decay
 = Σ_j ∫dE' ∫dΩ' [ ρ(x)/λ_j · c_ji(E',E, Ω'·Ω) + 1/d_j · d_ji(E',E, Ω'·Ω) ] f_j(x, Ω', E')
```

d_i = (p/m_i) c τ_i is the decay length. The geomagnetic cutoff is not a term in the
equation. It is the boundary condition at the top of the atmosphere:
f_primary = Φ(E) × T(x, Ω, E), with T the cutoff transmission.

MCEq is this equation with (a) every secondary collinear with its parent, so the kernels
depend on energy only, (b) no magnetic term, (c) one straight line per solve.

## 2. Five structural facts

1. **Linear.** [fact] Showers do not interact with each other. Superposition, Green's
   functions and adjoint (importance) solutions are all exact tools.
2. **Energy only goes down.** [fact] Every secondary has less energy than its parent, so the
   system is block-triangular in energy. A solve is one pass from high to low energy.
3. **Collisions are diagonal in Legendre index l.** [fact] The kernels depend on Ω'·Ω only,
   so with c_l(E',E) = 2π ∫ c(E',E,cosθ) P_l(cosθ) dcosθ each l evolves on its own.
   l = 0 is MCEq's matrix. Collisions in 3D cost L times collisions in 1D.
   (2D MCEq uses the small-angle version of this, a Hankel transform. Our
   `prototype_3d_cascade.py` confirmed the n_l × 1D cost.)
4. **The magnetic term is a rotation.** [derived] It turns directions about B at rate
   0.3 q B/p. In harmonics about the B axis it is diagonal: a phase −i m ω. For a muon the
   bend over one decay length is 0.3 B c τ / m_μ = 0.056 rad (3.2°) per 0.3 G of
   perpendicular field, independent of momentum. Kamioka (0.47 G) gives about 5°, which is
   the mean bend our segment MC test finds.
5. **Streaming through a stratified, curved atmosphere is the odd one out.** [fact] It is
   exact along trajectories, but in the l basis it couples l to l±1 with altitude-dependent
   coefficients, and the density depends on altitude, not on depth along a shared axis.

No single basis diagonalises 3, 4 and 5 together. Each formulation below is a choice of
which one to treat approximately.

## 3. Scale separation: the problem is 4D, not 6D

| length | value |
|---|---|
| π± decay length | 56 m × p[GeV/c] |
| K± decay length | 7.5 m × p |
| μ decay length | 6.2 km × p |
| nucleon interaction length | 0.7 km at sea level, 10 km at 20 km, 50 km at 30 km |
| atmospheric scale height | 6 to 7 km |
| scale over which cutoff sky and field direction change | several hundred km |
| line of sight to 20 km altitude at the horizon | 505 km |

[derived] Secondaries live for at most tens of km; the cutoff sky and the field change over
hundreds. So horizontal position can be integrated out locally: at each place on Earth the
unknown is f(altitude, direction, energy), four continuous variables. The one long leg, the
primary skimming 500 to 900 km of thin air, carries no cascade and enters only through the
boundary condition, which back-tracing gives exactly.

## 4. Three formulations

### A. Axis frame (2D MCEq, plus the field)

March in slant depth along the primary's line, carrying the angular distribution about that
line in Hankel or Fourier modes. [fact] This is Kozynets, Fedynitch and Koskinen (2023).
[derived] In the small-angle limit the field is a uniform drift in the angle plane, so it
adds an imaginary diagonal term i (k·a)(0.3 q B⊥/p)/ρ(X) with the same 1/ρ(X) structure as
decay. Cost: (number of modes) × (1D MCEq).

Sacrificed: fact 5. All particles share the axis' density profile and path length; no
curvature; small angles; nothing going upward relative to the axis hemisphere.
[fact] Authors' own validation: 100 GeV primaries, neutrinos within 1 to 10% up to medium
inclination, up to 25% off at large angles for 80° showers. Not included, by their
statement: geomagnetic cutoff, deflection of secondaries, Earth's sphericity.

[inference] Our engine is already a hand-built approximation of "A plus exact geometric
assembly": the cone kernels play the role of the angular modes. Adopting A would replace
those kernels with solved ones, but its errors sit in the same place as ours (near the
horizon, lowest energies) and are not controlled there.

### B. Altitude frame, marched in energy

Unknown: f(h, Ω; E, species) at one location, horizontally homogeneous (section 3).

```
for E from highest to lowest:
    source(h, Ω)  = Σ_l  Y_l(Ω) · [ c_l , d_l ] · (moments of densities already solved at E' ≥ E)
    f(h, Ω, E)    = ∫ source × survival, along the particle's true backward trajectory
```

The backward trajectory uses the real equations of motion: Earth curvature, bending in the
local field, energy loss. Every particle therefore sees the density at its own altitude,
and a bent muon is simply a particle whose local direction changed. There is no "column".

- Triangularity (fact 2) means no global iteration. [inference] Within one energy bin there
  are same-bin couplings (π→μ with most of the energy, leading nucleons). They are ordered
  by species and need at most a few inner sweeps.
- Forward peaks narrower than the angular grid are treated as straight-ahead (the delta-M
  truncation of radiative transfer). [inference] The straight-ahead limit is then MCEq.
- The cutoff sky T at that location is the top boundary condition. Solving on a grid of
  locations gives a global neutrino emissivity S_ν(lat, lon, h, Ω, E); any detector is a
  line-of-sight integral through it. The product is site-independent.

Approximations: horizontal homogeneity over tens of km, angular and altitude grids, the
hadronic yields. All are controlled, i.e. they can be refined and checked.

[estimate] One location: 150 altitudes × 1500 directions × 50 energies × 20 species =
2×10^8 unknowns, a few CPU-minutes compiled. A 5° global grid (2600 locations): a few
hundred CPU-hours, about a day at our 24-core cap. The field-free, azimuth-symmetric
version has 10^7 unknowns: seconds.

### C. B in stages, inside the present architecture

1. Field off, isotropic illumination: f(h, ϑ, E). Gives the pure geometric 3D/1D ratio.
2. Add azimuth and the cutoff sky.
3. Add bending of charged particles (curved trajectories).
4. Add derivatives (section 6).

Each stage has an independent reference because our MC has matching switches:
`--no-cutoff`, `--no-bend-primary`, `--no-muon-field` [fact: sky3d.py CLI].

## 5. What exists

- [fact] All published 3D atmospheric-neutrino fluxes are Monte Carlo: Honda et al., Barr
  et al. 2004, Battistoni et al. (FLUKA). Lipari 2000 gives the analytic geometry.
- [fact] 2D MCEq (arXiv:2306.15263; code: github.com/kotania/MCEq, branch 2DShow) is the
  only deterministic angular cascade for this problem. 24 Hankel modes, 60 energy bins,
  21 species; seconds to a minute per shower. Angular yields from UrQMD, EPOS-LHC,
  SIBYLL-2.3d, DPMJET-III via chromo, 10 million events per projectile and energy.
- [unverified] A search snippet of A. Fedynitch's TeVPA 2025 slides places "fluxes below
  5 GeV with geomagnetic cutoff and 3D effects" at 2027 or later. I could not open the file.
- [fact] Deterministic Boltzmann solvers with a magnetic term exist in MRI-guided
  radiotherapy (St-Aubin et al. 2015, 2016; Yang et al. 2018). Lesson: discrete ordinates
  with source iteration can be unstable with a field; angular finite elements with upwinding
  fix it. [inference] B avoids the issue: it has no source iteration and treats the field by
  trajectory, not by angular differencing.
- [fact] Our own record (ARCHITECTURE.md): Legendre decoupling works per column
  (`prototype_3d_cascade`); the streaming operator stays sparse and was checked against the
  diffusion limit (`prototype_streaming`, `spherical_streaming`); none of these carried real
  yields. The one dead end, `march_checkpoints`, binned *arrival-direction columns* and
  rotated between them. [inference] That is a failure of the column frame (A-like), not of
  deterministic transport: in B's frame the stated obstacle does not arise.

## 6. Consequences for hadronic-uncertainty propagation

- Write c_l(E',E) = c_0(E',E) × a_l(E',E), with c_0 MCEq's own matrix and a_l = ⟨P_l(cosθ)⟩
  from generator events. [inference] daemonflux's parameters act on c_0 exactly as they do
  today, so the 1D limit keeps the published calibration, and F = Φ_3D/Φ_1D from one solver
  has a native ∂F/∂θ. Angular-shape uncertainties (the NA61 ±12% width, worth ±5% on F in
  our sensitivity study) become parameters on a_l. Today the cone kernels are frozen.
- [fact] For a linear problem, first-order sensitivity to any kernel change is
  δΦ = ⟨ψ†, δK ψ⟩, with ψ† the adjoint solution: one extra solve per observable gives the
  gradient with respect to all parameters, with no statistical noise.
- [inference] The solver computes muon densities at every altitude and direction as a
  by-product. Low-momentum muon data (where cutoff and bending matter) could then enter a
  daemonflux-style fit. That is the only route I see to *calibrating*, not just propagating,
  the sub-GeV level, which today is uncalibrated MCEq with a 20 to 28% model spread.
- Still needed either way: the real DDM parameter → yield map.

## 7. What could kill B

1. Angular yields. Our pools hold 2000 events per node (709 files, 321 MB): enough to
   sample from, not to build smooth c_l. [estimate] Regenerating at 10^6 to 10^7 events per
   node is 10^2 to 10^3 core-hours with chromo.
2. Generators below about 2 GeV projectile energy. Unchanged: same floor problem as now.
3. Numerics: resolving cutoff edges on the angular grid, non-conservation of interpolated
   backward tracing, the same-bin couplings. Each is checkable; none is proven fine.
4. It is a new solver, not a modification of MCEq. MCEq supplies c_0 and the 1D limit.
5. It does not remove the need for the MC as an independent reference.

## 8. Proposed gate (stage C1), if authorised

Build B with the field off and azimuthal symmetry, real c_0, a_l from enlarged pools.

- G1: straight-ahead limit reproduces MCEq within 1% at every zenith.
- G2: with angles on, the zenith shape matches an MC run with all three switches off, within
  the MC's statistical error, including the horizon rows where E_off v4 is 8% high.
- G3: measured cost per solve.

Compute needed: pool regeneration, and one field-off MC run. Not started.

## Sources

- Kozynets, Fedynitch, Koskinen, arXiv:2306.15263, https://arxiv.org/abs/2306.15263
- St-Aubin et al., Med. Phys. 2015, https://www.osti.gov/biblio/22413444
- Fedynitch, TeVPA 2025 slides (not opened),
  https://indico.ific.uv.es/event/7986/contributions/29113/attachments/14166/20523/20251104_TeVPa_Anatoli_pdf.pdf
- Lipari, Astropart. Phys. 14 (2000) 153; Barr et al., Phys. Rev. D 70 (2004) 023006
- `tools/mceq3d/ARCHITECTURE.md` (prototype table), `tools/mc3d/sky3d.py` (CLI switches)
