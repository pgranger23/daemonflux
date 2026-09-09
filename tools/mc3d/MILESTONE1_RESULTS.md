# `tools/mc3d` milestone 1 — the collinear / `B = 0` closure against MCEq

Factual record.  Every number below comes from a run in this directory; the raw
outputs are `scratchpad/mc3d/closure_{A1,A2,A2nopol}.{npz,log}` and are
reproduced by the commands in §5.

## 1. What the gate is

Inject one proton of fixed **kinetic** energy straight down at the top of the
CORSIKA atmosphere (`h = 112.8 km`, `X_v = 0`) and tally every neutrino produced,
`dN/dE_nu` per primary.  MCEq run with `set_single_primary_particle(E, pdg_id=
2212)` at `theta = 0` and read at the ground gives exactly the same quantity for
the same hadronic model.  In collinear mode every neutrino goes straight down, so
"produced" and "crossing the ground" are the same set and nothing but the
atmosphere itself enters the comparison.

The primary is injected in **MCEq's own three-bin representation** — 
`set_single_primary_particle` does not put the proton in one bin, it spreads it
over `cenbin-1, cenbin, cenbin+1` with moment-matching weights whose third
entry is *negative* (+0.4184 / +0.6692 / **−0.0876** at 100 GeV).  Injecting at a
single bin centre instead compares two different primaries and shows up as a
growing high-energy discrepancy.  `closure.mceq_initial_condition` reproduces the
three signed weights exactly (gated in
`test_closure.py::test_initial_condition_matches_mceq`).

Three rungs isolate the pieces:

| rung | interactions | decays | MCEq reference |
|---|---|---|---|
| **A1** | MCEq yield matrices | MCEq decay matrices (helicity-summed) | `muon_helicity_dependence = False` |
| **A2** | MCEq yield matrices | our full kinematics, **polarised** Michel | `muon_helicity_dependence = True` |
| **A2-nopol** | MCEq yield matrices | our full kinematics, unpolarised | `muon_helicity_dependence = True` |
| B | chromo SIBYLL-2.3d / DPMJET-III | our full kinematics | milestone 2a |

A1 tests the geometry, the interaction/decay competition, the muon `dE/dx` and
the scorer; A2 adds our decay module; A2-nopol measures the size of the
polarisation effect.

## 2. Result

`MC / MCEq`, `dN/dE` integrated over each band, 200 000 showers per energy,
errors are the MC statistics only.

### A1 — geometry / transport / competition

| E_p | species | 0.1–0.3 GeV | 0.3–1 GeV | 1–3 GeV | 3–10 GeV | 10–30 GeV |
|---|---|---|---|---|---|---|
| 20 GeV | nu_mu | 0.996 ± 0.002 | 1.011 ± 0.002 | 1.044 ± 0.005 | 1.274 ± 0.025 | 1.40 ± 0.59 |
| | antinu_mu | 1.003 ± 0.002 | 1.001 ± 0.002 | 1.016 ± 0.005 | 1.102 ± 0.027 | 1.59 ± 1.13 |
| | nu_e | 0.983 ± 0.002 | 1.007 ± 0.003 | 1.014 ± 0.008 | 0.955 ± 0.038 | — |
| | antinu_e | 0.966 ± 0.002 | 0.982 ± 0.004 | 0.971 ± 0.009 | 1.039 ± 0.051 | — |
| 100 GeV | nu_mu | 1.017 ± 0.001 | 1.008 ± 0.001 | 1.015 ± 0.002 | 1.074 ± 0.004 | 1.202 ± 0.015 |
| | antinu_mu | 1.019 ± 0.001 | 1.002 ± 0.001 | 1.006 ± 0.002 | 1.045 ± 0.005 | 1.145 ± 0.019 |
| | nu_e | 0.995 ± 0.002 | 1.005 ± 0.002 | 1.010 ± 0.004 | 1.004 ± 0.009 | 0.978 ± 0.034 |
| | antinu_e | 0.985 ± 0.002 | 0.996 ± 0.002 | 0.989 ± 0.004 | 0.994 ± 0.010 | 1.050 ± 0.044 |

**Closure at 100 GeV over 0.1–3 GeV: 0.985–1.019, i.e. within ±2% for all four
species.**  At 20 GeV: 0.966–1.044.

### A2 — with our own decay kinematics and muon polarisation

| E_p | species | 0.1–0.3 GeV | 0.3–1 GeV | 1–3 GeV | 3–10 GeV | 10–30 GeV |
|---|---|---|---|---|---|---|
| 20 GeV | nu_mu | 0.965 ± 0.002 | 0.984 ± 0.002 | 0.974 ± 0.005 | 1.002 ± 0.020 | — |
| | antinu_mu | 0.961 ± 0.002 | 0.982 ± 0.002 | 0.974 ± 0.005 | 0.967 ± 0.025 | — |
| | nu_e | 0.942 ± 0.002 | 0.981 ± 0.003 | 0.983 ± 0.007 | 0.997 ± 0.032 | — |
| | antinu_e | 0.939 ± 0.002 | 0.986 ± 0.004 | 0.987 ± 0.008 | 0.922 ± 0.042 | — |
| 100 GeV | nu_mu | 0.981 ± 0.001 | 0.987 ± 0.001 | 0.977 ± 0.002 | 1.004 ± 0.004 | 1.003 ± 0.013 |
| | antinu_mu | 0.978 ± 0.001 | 0.985 ± 0.001 | 0.973 ± 0.002 | 0.997 ± 0.004 | 0.992 ± 0.017 |
| | nu_e | 0.951 ± 0.002 | 0.981 ± 0.002 | 0.985 ± 0.003 | 0.984 ± 0.008 | 0.989 ± 0.031 |
| | antinu_e | 0.953 ± 0.002 | 0.981 ± 0.002 | 0.975 ± 0.004 | 0.972 ± 0.009 | 0.910 ± 0.037 |

**Closure at 100 GeV over 0.1–3 GeV: 0.951–0.987, i.e. within 2–5%.**  The gate
("a few %") is met.

### A2-nopol — the size of the muon-polarisation effect

| E_p | species | 0.1–0.3 GeV | 0.3–1 GeV | 1–3 GeV | 3–10 GeV |
|---|---|---|---|---|---|
| 100 GeV | nu_mu | 0.982 ± 0.001 | 0.997 ± 0.001 | 0.990 ± 0.002 | 1.019 ± 0.004 |
| | antinu_mu | 0.980 ± 0.001 | 0.995 ± 0.001 | 0.989 ± 0.002 | 1.022 ± 0.004 |
| | **nu_e** | **0.924 ± 0.001** | **0.916 ± 0.002** | **0.891 ± 0.003** | **0.852 ± 0.007** |
| | **antinu_e** | **0.925 ± 0.002** | **0.920 ± 0.002** | **0.876 ± 0.003** | **0.842 ± 0.008** |

Switching the polarisation off moves `nu_e` from 0.95–0.99 to 0.89–0.92 and
leaves `nu_mu` essentially unchanged — the expected signature (a `mu+` from a
`pi+` is predominantly helicity −1 and makes a *harder* `nu_e`).  This is an
independent confirmation that the polarised Michel treatment is right, and it
shows the polarisation is worth **+7 to +9% on the sub-GeV `nu_e` yield** —
far too large to omit in a calculation aimed at the East–West `nu_e` asymmetry.

### Where the residual sits

* **A1 vs A2, ~2–3% on all species**: the difference is our decay module against
  MCEq's decay matrices.  One known contributor is identified: MCEq's
  `d_321_14 = 0.6353` is the `K_mu2` branch **alone** — it carries the `K_mu3`
  *muon* (`d_321_-13 = 0.6692 = K_mu2 + K_mu3`) but not the `K_mu3` *neutrino*,
  a 5% difference on the kaon `nu_mu` yield.  The others are the flat-Dalitz
  `K_l3` treatment and unpolarised muons from three-body kaon decays.
* **The `1.0`–`1.2` drift above 3 GeV at 20 GeV primary energy** is the
  kinematic edge: there `dN/dE` falls by orders of magnitude per bin, our
  histogram is an exact integral of a continuous spectrum while the MCEq
  reference is a log-log interpolation of a binned one, and the parent's
  position within its energy bin matters.  It is outside the 0.1–3 GeV band the
  3D/1D tables are for.
* **The residual ~1.5% at 0.1–0.3 GeV** is at the level of the atmosphere
  integrator (2.6e-5), the `dE/dx` table difference and the within-bin energy
  assignment; it is not yet decomposed.

## 3. Bugs the gate caught

All three were silent — each produced a *plausible* flux with a wrong
normalisation or shape.

1. **Particles passed through the Earth.**  `atmosphere.path_to_exit` took the
   *far* root of the ground sphere whenever the near root was `<= 0`, which
   happens the instant a particle sits exactly on the surface.  Muons then took
   thousands of steps *inside* the Earth and **every one of them decayed**,
   inflating the muon-decay neutrino yield by ~2× (the measured decay fraction
   was 1.000 at every energy against the correct 0.30 for a 50 GeV muon).
   Regression: `test_geometry.py::test_particles_do_not_pass_through_the_earth`,
   `test_closure.py::test_no_particle_crosses_the_earth`.
2. **Kinetic vs total energy at the MCEq table lookup.**  MCEq's energy grid is
   *kinetic* (`etot_grid = e_grid + m`); the cascade tracks *total* energy.
   Without the conversion, nucleons are misplaced by ~1 GeV and pions by
   0.14 GeV.  Regression:
   `test_sampler.py::test_kinetic_vs_total_energy_conversion`.
3. **MCEq's helicity-resolved muon states.**  `D.get_matrix((211,0), (-13,0))`
   is **empty**: pion→muon decay lives in `(-13,-1)` `pi_mu+_l` and `(-13,+1)`
   `pi_mu+_r`.  A naive export loses every muon, hence half the `nu_mu` and
   essentially all the `nu_e` (the closure came out at 0.26–0.55).
   Regression: `test_sampler.py::test_decay_tables_contain_the_muon`.

Plus two found by the unit tests rather than the closure:

4. **The muon rest-frame spin projection had the wrong form.**  Using
   `zeta = s - (s.p)/(m(E+m)) p` gives `P_L = (2m-E)/m (s.phat)` instead of the
   correct `P_L = (m/E)(s.phat)` — the **wrong sign** for every relativistic
   muon.  Regression:
   `test_decays.py::test_muon_from_pion_is_fully_polarised_at_the_kinematic_edges`.
5. **The Michel asymmetry sign was flipped.**  Measured against MCEq's
   helicity-resolved decay matrix: a helicity −1 `mu+` gives
   `<E_nue> = 7.170 GeV` at `E_mu(kin) = 17.78 GeV` versus 5.401 unpolarised,
   i.e. *harder*; our convention gave softer.  With this uncorrected the `nu_e`
   closure was 0.86.  Regression:
   `test_decays.py::test_michel_polarised_spectra_match_mceq`.

Two more issues were found and fixed while tuning rather than by a gate: the
muon tracking threshold was `e_nu_min + m_mu`, which throws away muons that
*can* make a neutrino above threshold; and `Lambda` (0.20 + 0.03 `Lambda-bar`
per p–air interaction at 89 GeV, `Lambda -> p pi-` 63.9%) was not tracked, worth
~2% of the neutrino yield.

## 4. Throughput

`MCEqYieldBackend`, collinear, 40 cores:

| rung | 20 GeV | 100 GeV |
|---|---|---|
| A1 | 2713 showers/s | 907 showers/s |
| A2 | 1583 showers/s | 570 showers/s |

i.e. **14–23 showers/s/core at 100 GeV**.  `chromo` SIBYLL-2.3d p–N at 100 GeV
runs at **1068 events/s/core when batched** (`model(3000)`) against 73/s when
called one event at a time — see the performance note in `PHASE2_PLAN.md` §4.2.

`chromo` yields against the MCEq tables (p–N, 100 GeV, per interaction):
`pi+` 3.393 vs 3.399, `pi-` 2.860 vs 2.971, `K+` 0.257 vs 0.279, `K-` 0.140 vs
0.162, `K_L` 0.160 vs 0.226, `Lambda` 0.197 vs 0.201, `p` 1.323 vs 1.404,
`n` 0.997 vs 1.158.  The residual is the nitrogen-only target versus MCEq's N/O
air mix; milestone 2a draws the target from the air composition.

## 5. Reproducing

```sh
cd tools/mc3d
export PY=/cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc14-opt/bin/python3.12
export PYTHONPATH=<scratchpad>/pylib:<repo>/.venv/lib/python3.12/site-packages:\
<repo>/src:<repo>/tools/mceq3d:$PWD

$PY build_tables.py --out mceq_tables_SIBYLL23D.npz          # ~8 s, 3.6 MB
$PY closure.py --energies 20 100 --nshower 200000 --nproc 40 --mode mceq      \
    --out closure_A1.npz
$PY closure.py --energies 20 100 --nshower 200000 --nproc 40 --mode kinematic \
    --out closure_A2.npz
$PY summarize_closure.py closure_A1.npz closure_A2.npz
$PY -m pytest -q -m "not slow"        # 34 passed, ~4 min
```

`chromo` writes `tables.dat` into the working directory; it is in `.gitignore`.

## 6. Test inventory (34 passing, 1 marked slow)

| file | n | covers |
|---|---|---|
| `test_geometry.py` | 12 | density and vertical depth vs MCEq (1e-10), scalar fast paths, vertical column, `advance_grammage` as the inverse of `grammage`, slant depth vs `offaxis_mc.slant_depth_table` at table nodes (3e-3), the Earth-crossing regression, cap area / acceptance / Monte-Carlo solid angle, Lambert injection sampling, the arrival-direction convention vs `geomag_backtrace`, zenith/azimuth round trip |
| `test_decays.py` | 13 | two-body CM momentum and 4-momentum conservation, flat `pi -> mu nu` neutrino spectrum, muon spectrum, full polarisation at the kinematic edges, Michel shape and `<x>`, the polarisation sign, polarised spectra vs MCEq's helicity matrices, kaon BRs vs PDG and vs MCEq, `Lambda` BRs, three-body 4-momentum conservation, stopped-muon isotropy, `michel_n/asym` identical to `muon_segment_mc` |
| `test_sampler.py` | 6 | primary energy weights reproduce `crflux` per decade (2%), species/rigidity superposition, the yield sampler reproduces the MCEq column bin-by-bin (6%) and its multiplicity (1%), interaction length, kinetic-vs-total conversion, the helicity-summed muon decay tables |
| `test_closure.py` | 4 (1 slow) | MCEq's three-bin initial condition, a sane neutrino yield, no particle crosses the Earth, and the reduced-statistics collinear closure |
