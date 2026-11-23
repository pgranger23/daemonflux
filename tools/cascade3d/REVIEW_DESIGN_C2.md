# Manager's review of the C2 design note

Date: 2026-11-20. Reviewed: `c2/design/DESIGN_C2_draft.md` (638 lines) and
`c2/design/WP_DESIGN_REPORT.md`, read in full. Nothing of the reference of G2.3
(`c2/mcref/`) was opened for this review.

Status of the statements below: "measured" means measured by the design agent and not
reproduced by anyone else.

## 1. Decision

The design is accepted for implementation, with the conditions of section 3.

- Azimuth as Fourier orders m = 0 to M about the local vertical.
- The flux with the cutoff as the C1 solve (transmission 1) minus a linear map of the
  boundary deficit: Φ_cut = Φ_C1 − J[D], D = B⁰ − B.
- J as the tangent of the C1 scheme with guarded derivative rules.
- The cap average as one boundary (J is linear and the same at every location).
- Cutoff skies from `cutoff.trace_gated` on a rigidity ladder, per location of a grid
  about the site.

## 2. What the review found

1. **The exact derivative of the C1 solve is unstable (measured, reduced geometry).**
   `jax.jvp` of the frozen solve gives cells wrong by factors of 10 to 10^17 below
   1.6 GeV and right to 5e-5 above 2.5 GeV. G1.0(c) had tested the derivative only in the
   straight-ahead limit, so C1 did not see it. Consequence beyond C2: the hadronic
   derivatives of C4 need the same guard. The guard changes the derivative rule only; the
   C1 values are unchanged.
2. **The guard is validated on a reduced geometry only.** 400 cells within 0.42% of the
   nonlinear solve with ρ = 0.1. The full geometry is not tested yet. This is the first
   task of the implementation and it gets a gate (section 3, G2.7).
3. **Two gates of C2 lose their power in this design.** G2.1 (transmission 1 reproduces
   C1) and G2.5(b) (rotation by 30°) hold by construction. They still test the
   bookkeeping of the assembly. They no longer test the solver.
4. **The anisotropic orders (m ≥ 1) have no nonlinear counterpart on the same code
   path.** Before the freeze they are tested by unit tests only (analytic Y_lm field, a
   toy problem against independent φ-chains, rotation). The first end-to-end test is the
   blind G2.3. A cheap check of the azimuth bookkeeping is added (section 3, G2.8). It
   does not test the m ≥ 1 kernel moments: in the straight-ahead limit every order has
   the same transport.
5. **G2.0 as proposed would not test the lines that miss the ground.**
   `cutoff.sample_test_states(cap_deg=10)` keeps only lines whose first ground crossing
   is in the cap. The tangent lines below 60 GV, which have their own code path in
   `boundary.py`, would be left out. Fixed in section 3.
6. **M is not decided.** The Kamioka sky has 0.5 to 1.1% of its azimuthal power in
   m = 7 to 12 (measured). The agent expects M = 6 to fail G2.5(a) in near-horizon cells
   at 5 to 10 GeV. The first solve (Kamioka, M = 12) decides, at no extra cost.
7. **Cost.** One boundary: 0.9 to 1.7 core-h (M = 6) or 1.7 to 3.3 (M = 12). All C2
   solves: 13 to 25 core-h (M = 6) or 26 to 50 (M = 12); the plan had 10 to 20. C5 (about
   1450 locations): 1300 to 2500 core-h (M = 6) or 2500 to 4800 (M = 12); the plan had
   800 to 1300. No GPU number exists.
8. **C4.** The derivative of the cutoff part with respect to a hadronic parameter is a
   second-order derivative: about 1.8 core-h per parameter direction and location. Two
   cheaper options exist (differentiate the m = 0 deficit only; assume the cutoff factor
   does not depend on the parameter and test it at one location). This is a decision for
   the PI at decision point 2, not now.

## 3. Conditions (written in `GATES.md` before any C2 solve with the cutoff)

| Item | Decision |
|---|---|
| Guard acceptance | New gate G2.7, full geometry, two deficits, see `GATES.md` |
| Mode path, straight-ahead | New reported check G2.8 with a trigger at 1% |
| G2.0 draw | Default draw of `sample_test_states` (no `cap_deg`); a state is covered if the location of its line is within 10° of the site; at least 20 000 covered states, at least 2 000 of them on lines that miss the ground |
| G2.0 quadrants | Local frame of the injection point |
| G2.4(a) | Spacing 5° against 2.5° |
| G2.4(b) | Reference altitude 0 against 15 km with the displaced location, six shifted cap averages |
| G2.5(a) | M against 2M from the partial sums of one run at 2M; μ grid and L doubled in separate runs |
| G2.2 | H3a groups for the boundary; straight-ahead per-chain path |
| G2.3(h) | Not evaluated in C2 (the assembly does not cover the far side) |
| Conventions | T = 0 on lines that never reach the injection sphere; T = 1 above 60 GV; arrival azimuth of `scoring.py` |
| Rank of the deficits | Reported: singular values of the location deficits (no solve) |
| Memory | 16 GB per process while the Monte Carlo runs |

## 4. Not accepted, or left to the PI

- The φ-chain fallback (2 to 3 times the cost per boundary) is not authorised. If G2.7
  fails after one tuning cycle, this goes to the PI.
- M = 12 for all solves takes C2 to 83 to 107 core-h, at or beyond the plan. It goes to
  the PI with the first cost measurement (G2.6).
- The far side (24 boundaries, about 50 more sky tables) is not in C2.

## 5. Budget of C2 as it stands (core-h)

| Item | M = 6 | M = 12 |
|---|---|---|
| Monte Carlo reference (projected from 14 min per run on 12 cores) | 47 | 47 |
| Design work spent | 1.6 | 1.6 |
| Manager's calibrations | 1 | 1 |
| Cutoff skies and G2.0 | 11 | 11 |
| Tests | 3 | 3 |
| Solves | 13 to 25 | 26 to 50 |
| Total | 77 to 89 | 90 to 114 |

The plan has about 100 core-h.
