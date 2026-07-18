# `tools/mceq3d` architecture — what is delivered vs. what is a prototype

This directory grew as a research sprint and accumulated ~35 modules, many of
them overlapping feasibility prototypes for a "real 3D MCEq". This file is the
authoritative map of **which single code path produces the delivered flux** and
which modules are research/de-risking scaffolding that the delivered flux does
**not** use. (Generated from the actual import graph, 2026-07-16.)

## The canonical delivered engine

**`mceq3d_flux.MCEq3DFlux`** is the one and only delivered solver. Everything a
user should call goes through it (`solve()`, `offaxis_factor()`, `cone_geff()`).
It is a *factorised deterministic product*, not a coupled Monte-Carlo:

```
Phi_s(E, cosZ, phi) = Phi_1D_s(E,|cosZ|)  x  E_off(E,cosZ)  x  G_s(E, R_c(cosZ,phi))
```

### Modules the engine actually depends on (transitive import closure)

| module | role in the delivered flux |
|---|---|
| `mceq3d_flux.py`      | the engine (base reconstruction, product assembly, covariance) |
| `geomag_backtrace.py` | IGRF-13 rigidity cutoff `R_c` by trajectory back-tracing (the strongest ingredient) |
| `kinematic_kernel.py` | pion/kaon/muon production-angle kernels (NA61-anchored) |
| `muon_bending.py`     | charge-signed in-flight muon-decay cone shift for the E-W geomagnetic term |
| `fokker_planck_3d.py` | supplies `load_theta2` / `sigma_theta_vs_energy` (the cone width sigma_theta(E)) |
| `coupled_3d_flux.py`  | supplies `convolve_sphere` (spherical convolution helper) |
| `angular_kernel.py`   | angular-moment helpers used by the kernel path |

### Build-time input (not imported at runtime, but the engine reads its output)

| module | produces |
|---|---|
| `offaxis_mc.py`     | `offaxis_excess.npz` — the `E_off` table (`python offaxis_mc.py --build`) |
| `kernel_regeneration.py`, `splice_kernels.py` | the generator moment kernels `m_*.npz` |

### Validation / analysis entry points (keep; each is a runnable script)

`validate_honda.py`, `validate_bartol.py`, `validate_na61.py`,
`validate_na61_kaon.py`, `validate_ew_zenith.py`, `base_comparison.py`,
`channel_comparison.py`, `make_cutoff_map_fig.py`, `diag_eoff_conservation.py`
(flux-conservation check on `E_off`, added 2026-07-16), plus the `verify_*`,
`latitude_check`, `geomag_zenith_check`, `profile_3d` diagnostics.

## Research / de-risking PROTOTYPES — **not** part of the delivered flux

These were built to test feasibility of a genuinely coupled 3D transport. The
delivered `mceq3d_flux` engine does **not** import any of them. They are kept
for the record and for the open production-vertex closure (see below), but no
result in the paper should depend on them without promotion into the engine.
Each carries a `PROTOTYPE` banner in its module docstring.

| module | what it explored | status |
|---|---|---|
| `mceq3d_real.py` | deterministic march of MCEq's real matrices per direction | **active** — vehicle for the production-vertex closure |
| `validate_3d_deterministic.py` | emergent cone factor vs `E_off`/Honda | **active** — the closure diagnostic |
| `mceq3d_deterministic.py` | parametrised (pre-real-matrix) 3D march | superseded by `mceq3d_real` |
| `mceq3d_solver.py`, `mceq3d_production.py` | earlier coupled-solve attempts | superseded |
| `spherical_streaming.py`, `spherical_cascade.py`, `spherical_geometry.py` | P_N spherical transport with curvature term | feasibility only |
| `coupled_3d_transport.py`, `coupled_3d_flux.py`* | coupled energy×angle solve prototypes | feasibility (*`coupled_3d_flux` also exports the `convolve_sphere` helper the engine uses) |
| `sn_transport.py`, `fokker_planck_3d.py`* | discrete-ordinate / Fokker–Planck angular spread | feasibility (*`fokker_planck_3d` also exports `sigma_theta` used by the engine) |
| `prototype_3d_cascade.py`, `prototype_streaming.py` | cost-scaling de-risking of the coupled solve | feasibility, done |
| `unified_3d_flux.py`, `directional_flux.py`, `geomag3d_spike.py` | early end-to-end assemblies | superseded by `mceq3d_flux` |
| `hadronic_spread.py`, `calib_uncertainty.py`, `solar_modulation.py` | one-off studies | standalone |

## Cone-kernel consistency fix (2026-07-16, delivered)

The delivered pion cone now uses the exact NA61-validated moment `sigma_pi`
(`m_spliced.npz`), not the sampled full kernel (`k_spliced.npz`), which was
root-caused as ~16-37% too wide (a 0.667-deg theta-binning artefact;
`diag_kernel_consistency.py`). This removed the 0.5-1 GeV horizon overshoot for
both flavours vs both references. `offaxis_excess.npz` was rebuilt with
`cone_kernel="moments"` (the new default); the legacy table is
`offaxis_excess.sampled.bak.npz`. Diagnostics added: `diag_isolate.py`,
`diag_overshoot.py`, `diag_cone_shape.py`, `diag_kernel_stats.py`,
`diag_cone_fix.py`, `diag_kernel_consistency.py`, `diag_eoff_conservation.py`,
`diag_prodvertex.py` (the refuted column-norm experiment).

## Production-vertex closure — DONE (`mceq3d_prodvertex.py`, 2026-07-17)

The delivered `E_off` is a near-flux-conserving redistribution (`diag_eoff_conservation.py`,
`<E_off>_Omega = 1.04–1.06` sub-GeV → 1 above a few GeV) **and is now confirmed as
the deterministic production-vertex result**. Reframe: neutrinos free-stream, so the
whole 3D neutrino effect is production geometry — there is no transport coupling to
solve (which is why the `march_checkpoints` arrival/parent-reweighting experiments,
kept as `diag_prodvertex.py`/`diag_prodvertex2.py`, all failed). `mceq3d_prodvertex.py`
takes the production profile `p(X,E)` from an independent hand-march of MCEq's real
matrices (`mceq3d_real.march_profile`) and integrates it through the curved
production-vertex cone, reproducing the delivered `E_off` to ~1–2% (gates: cone→0 ⇒
`E_off`→1 to 5e-4; →1 at high E). Untested refinement: the per-direction curved-density
profile (~1–2%). The remaining capstone is the **charged-sector** coupled solve for the
geomagnetic E–W with in-cascade muon bending, not the neutrino excess.
