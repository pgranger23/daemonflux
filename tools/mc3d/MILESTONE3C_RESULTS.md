# `tools/mc3d` milestone 3c — gate C3 West, the muon-bend mechanism, paired re-anchor vs re-simulate

Factual record, written as the work proceeds (started 2026-11-04).  Raw
outputs under `/home/pigrange/mc3d_work/runs/m3c/` (never `/tmp`); small
summaries copied to `results/m3c/`, scripts to `results/m3c/scripts/`.

## 0. Status / resume here

| step | state | where |
|---|---|---|
| code: `--crn`, `--mu-bend vertical/horizontal/none`, `--mu-cf`, muon-bend histograms, `--primary-energy` / `primaries.ENERGY_CONVENTION` (default now `total`) | done, 7 tests in `test_m3c.py`; `a1ancd` reproduces `a1anc` bit for bit | `sky3d.py`, `primaries.py`, sec. 4 |
| runs `a1nc`, `a1ncnomu`, `a1ncnb`, `a1ancd`, `c3anc`, `c3res`, `a1ncnbnomu` | **all done** (2026-11-04 11:38-13:31), 33.5 core-h MC; + ~13 core-h analysis (MCEq responses 2.2 h on 1 core, pilot replay ~9 core-h incl. an aborted first attempt with per-file tracing, penumbra, re-weighting) | sec. 6, `$M/logs/progress.txt` |
| task 1: gate C3 West | **done**: engine rigidity bug (`G_s` 3.5-10% low; not edited) + mc3d H3a energy-variable bug (fixed, default switched) + generator yield difference + pilot under-sampling of p 1-3 GeV; vertical agreement was a cancellation; gate re-run fails vs the delivered engine (chi2/12 31 / 13); the fix lowers the pilot's a1/a0 by 0.04 (fan-out unchanged; chi2/12 vs Honda 1.95 -> 12.85) | sec. 1 |
| task 2: muon-bend mechanism | **done: H1** (cutoff-independent direction migration; H2 share 0.07+-0.09); engine-fix note | sec. 2 |
| task 3: re-anchor vs re-simulate | **done**: mean d -0.014+-0.022 paired, -0.020+-0.014 combined; p 32-65 GV -0.026+-0.008; precision goal not reached | sec. 3 |
| full test suite | **97 passed** (90 + 7 new), 2 warnings, 13.2 min under load (`$PY -m pytest $R/tools/mc3d -q` from `$M/work/pytest`, 2026-11-04 13:57) | sec. 5 |

**Resume:** `source /home/pigrange/mc3d_work/runs/m3c/env.sh`; `cat $M/logs/progress.txt`.
All runs are finished.  The default primary-energy convention changed after
them (sec. 1.1): re-running or resuming any of these tags needs
`--primary-energy kinetic` (without it the manifest check refuses the job as
"a different job"); with it every job resumes from its task files.

Environment of every run here: `runs/m3c/env.sh` = `runs/m3b/env.sh`
(`$W/pylib_fix`, `MC3D_POOL_CACHE=$W/pools_fix`, field cache of m3b) with
`M=$W/runs/m3c`.  **Database: unchanged from m3b** -- the interaction lengths
of the cascade come from `tools/mc3d/mceq_tables_SIBYLL23D.npz`, built from
the shipped MCEq database (`mceq_db_lext_dpm193_v140.h5`, with its
antiparticle cross-section defects and the 56 GeV hole); the other agent's
switch of `tools/mceq3d` to the patched v1.4.2 file does not reach the MC.
Kept on purpose: the new runs are paired with m3b's
(same seeds and tasks), and the pairing removes far more noise from the
differences than a database change could be worth here.


---

## 1. Gate C3 West (task 1)

Scripts (`$M/scripts/`, copied to `results/m3c/scripts/`): `penumbra.py`,
`h3a_conventions.py`, `mceq_response.py` (MCEq's 1D response per primary
energy cell at cos Z 0.95 and 0.05: linear, so any convention can be
re-weighted), `extract_strata.py` (per-(jackknife group, stratum) sums),
`c3_conventions.py`, `c3_reweight.py` (exact per-primary re-weighting of the
1D entries from the kept c-files).

**(a) Cutoff definition and penumbra.**  The MC's `v` and the gate's engine
value use the same quantity: the site map's `R_c` (= `R_U`, ground launch,
sea-level floor) at the bin direction, sharp.  The penumbra therefore does not
enter the gate.  Width measured anyway (`penumbra.py`, ladder 0.1 GV,
`reports/penumbra.txt`): in the two West bins (zenith 84.5-89.5, az 240-300)
`R_c` = 7.3-8.9 GV, `R_U - R_L` = 0-1.4 GV (mostly 0.1-0.7), `R_U - R_eff`
<= 0.5 GV; in the vertical `R_U - R_L` = 1.7-2.5 GV, `R_U - R_eff` = 0.4-1.0
GV.  **The West penumbra is narrower than the vertical one**, not the widest.
Candidate (a) is excluded.

**(c) H3a and the rigidity: the conventions differ on both sides**
(`reports/h3a_conventions.txt`):

| | `tools/mc3d` | engine (`mceq3d_flux` / MCEq) |
|---|---|---|
| rigidity of the cut | `(A/Z) p_nucleon(T)` (`cutoff.rigidity_gv`) | `(A/Z) T` -- kinetic energy (`_transmission(self.e, ...)`, `self.e = mceq.e_grid`, MCEq's grid is kinetic) |
| H3a energy variable | crflux at `A T` (`primaries.nucleon_intensity`) | crflux at `A (T + m)` (`MCEqRun.set_primary_model`; crflux: "laboratory energy of nucleus") |
| lowest primary | `T = 1 GeV/n` (`sky.STRATA`) | `E_tot = 3 GeV` (`minimal_primary_energy`; first bin T = 2.24 GeV) |
| nuclei | five groups, own `A/Z` | free p = p - n, bound `<A/Z>` |

At a cut of 7.5 GV the MC removes `T < 6.62` (A/Z = 1) / `2.93` (A/Z = 2)
GeV/n, the engine `T < 7.50` / `3.75`; at 11.5 GV 10.60 / 4.89 vs 11.50 /
5.75.  The MC's nucleon flux is the MCEq one x 2.78 / 2.06 / 1.75 / 1.47 / 1.27
/ 1.18 / 1.09 / 1.03 at T = 2 / 3 / 4 / 6 / 10 / 15 / 30 / 100 GeV/n (every
group alike).

**Crflux group definitions.**  `primaries.SPECIES` asks crflux's H3a for ids
1407 / 2512 / 5626 (N-14, Mg-25, Fe-56); crflux 2.0 defines H3a with 1206 /
2814 / 5426 (C-12, Si-28, Fe-54) and maps the MC's ids onto the same group
spectra.  The MC's nucleon flux of those groups (`A^2 Phi(A T)`) therefore
differs from MCEq's by `(A_MC/A_crflux)^-0.63` = 0.91 (CNO) / 1.07 (MgSi) /
0.98 (Fe): ~1% of the total nucleon flux.  Not changed; noted.

**(b) and the pilot's sampling of the no-cutoff `t`.**  Exact per-primary
re-weighting of the kept c-files (`c3_reweight.py`: each pending primary's
nucleus rigidity is stored, so its `T` and any flux convention's weight
follow; with the original weights it reproduces the recorded gate values
digit for digit).  The pilot's Neyman allocation (optimised for the 3D
amplitudes, to which the p 1-3.16 GeV stratum contributes nothing) gave that
stratum 2850 primaries, and none of their 1D entries falls in the 0.3-1 GeV
bands of the gate bins: the pilot's `t` there misses it entirely.  In `a1anc`
(`E^-0.8`, 3.7e5 such primaries) it is 12% (vertical) and 19% (87 deg) of
`t` for numu at 0.3 GeV, ~3% at 0.5 GeV, 0 at 1 GeV.  **The pilot's `v/t` at
0.3 GeV is therefore biased high by ~12-19%**; the 1D no-cutoff reference of
a production run needs an allocation that covers the lowest strata.
Azimuth-summed (`reports/t_p0_share.txt`, a1anc): the p 1-3.16 GeV share of
`t` is 0.09-0.21 at 0.3 GeV (cos Z 0-0.3 and 0.9-1), 0-0.14 at 0.5 GeV,
<= 0.025 at 1 GeV -- **the pilot's `s/t` (MILESTONE3 sec. 4.5) is high by
the same 10-25% at 0.3 GeV** (`s/v` is unaffected: those primaries never
pass a Kamioka cutoff).

**MCEq's response per primary-energy cell** (`mceq_response.py`, bands
integrated with exact bin overlaps): the engine's stored `G_s` is
reproduced to 0.1-4% (`eng-repro`, the 4% at 0.3 GeV from the band
integration of the stored table).  The MC's no-cutoff 1D yield over MCEq's,
**same primary flux** (the MC's convention), vertical, numu / nue at
0.3 / 0.5 / 1 GeV, per stratum (`a1anc`):

| stratum (GeV/n) | p | He |
|---|---|---|
| 1-3.16 | 0.48 / 0.50 / - , 0.54 / 0.46 / - | 0.47 / 0.89 / - , 0.58 / 0.42 / - |
| 3.16-10 | 0.80 / 0.77 / 0.76 , 0.75 / 0.72 / 1.06 | 0.89 / 0.61 / 0.77 , 0.71 / 0.75 / 1.05 |
| 10-31.6 | 0.92 / 0.90 / 0.87 , 0.85 / 0.96 / 0.88 | 0.80 / 0.85 / 1.05 , 0.95 / 1.05 / 0.95 |
| 31.6-100 | 0.94 / 0.92 / 0.94 , 0.98 / 0.89 / 0.86 | 0.82 / 0.78 / 0.90 , 0.95 / 0.90 / 0.77 |

All strata: 0.71-0.78 at 0.3 GeV, 0.79-0.81 at 0.5, 0.84-0.95 at 1 GeV.  The
MC's generator (chromo DPMJET-III-19.3 pools, pool floor `E_tot` = 2 GeV, no
floor artefact) yields fewer sub-GeV neutrinos than MCEq's tables, the more
so the lower the primary energy -- the milestone-2 rung-B deficit (0.85-0.93
at 20 GeV) continued downward, ~0.5 at 1-3 GeV/n.  Because the cut removes
exactly the low-energy primaries, this raises the MC's `v/t` above MCEq's.


**The gate with the conventions made explicit, vertical**
(`reports/c3_conventions.txt`; `G` averaged over the bin's site-map `R_c`
like `gate_c3`; numu at 0.3 / 0.5 / 1 GeV, the other species alike):

| quantity | numu |
|---|---|
| MC `v/t`, pilot, as run (= the recorded gate) | 0.483 / 0.604 / 0.751 |
| engine `G_s`, stored (m3b) | 0.497 / 0.597 / 0.752 |
| MCEq A: engine conventions (`R = (A/Z) T`, crflux at `A(T+m)`, T >= 2) | 0.472 / 0.593 / 0.747 |
| MCEq B: true rigidity `(A/Z) p`, otherwise as A | 0.511 / 0.631 / 0.778 |
| MCEq C: all the MC's conventions (true rigidity, crflux at `A T`, T >= 1) | 0.367 / 0.545 / 0.730 |
| MC `v/t`, a1anc (p 1-3.16 GeV well sampled), as run | 0.437 / 0.591 / 0.757 |
| MC re-weighted to MCEq's flux convention and T >= 2 (exact), pilot / a1anc | 0.572 / 0.670 / 0.794 ; 0.555 / 0.664 / 0.803 |

* The engine's rigidity convention alone (A -> B) raises `G` by 6-8% at
  0.3-0.5 GeV, 4% at 1 GeV: **the delivered `G_s` cuts too many primaries**.
* The MC's flux convention lowers it by 18 / 12 / 6% (crflux at `A T`, T >= 2:
  0.418 / 0.557 / 0.730), and its T = 1-2 GeV/n primaries by a further 12 / 2 / 0%
  (MCEq C).
* Harmonised (MC re-weighted to MCEq's convention vs B): MC/MCEq = 1.04-1.12
  at 0.3 GeV, 1.04-1.11 at 0.5 GeV, 0.93-1.06 at 1 GeV (a1anc: numu
  1.09+-0.02 / 1.05+-0.03 / 1.03+-0.03) -- the generator's low-energy yield
  deficit (b) seen through the cut.
* As run against the stored `G_s` these shifts (+7%, -14 to -28%, +5-10%,
  and the pilot's missing p 1-3.16 GeV stratum, +12% at 0.3 GeV) cancel to
  the recorded <= 2%: **the vertical agreement of gate C3 was a
  cancellation, not a validation.**


**The West** (`reports/c3_conventions.txt`, both bins alike; `R_c` over the
bin 7.3-8.8 GV, mean 7.8; numu / antinumu at 0.3 / 0.5 / 1 GeV, bin
240-270):

| quantity | numu | antinumu |
|---|---|---|
| MC `v/t`, pilot, as run (recorded) | 0.606 / 0.806 / 0.950 | 0.591 / 0.851 / 0.981 |
| engine `G_s`, stored | 0.634 / 0.737 / 0.867 | 0.622 / 0.722 / 0.858 |
| MCEq A (engine conventions; reproduces the stored `G_s` to 0.2% at 0.5-1 GeV) | 0.607 / 0.732 / 0.862 | 0.597 / 0.717 / 0.853 |
| MCEq B (true rigidity) | 0.665 / 0.780 / 0.893 | 0.654 / 0.764 / 0.884 |
| MCEq C' (MC flux convention, T >= 2) | 0.571 / 0.714 / 0.858 | 0.556 / 0.692 / 0.846 |
| MC re-weighted to MCEq's convention, T >= 2 (pilot) | 0.708 / 0.849 / 0.962 | 0.654 / 0.883 / 0.985 |
| ... / MCEq B | 1.07+-0.05 / 1.09+-0.05 / 1.08+-0.03 | 1.00+-0.12 / 1.16+-0.05 / 1.12+-0.02 |

* Engine rigidity (A -> B): +10 / +7 / +3.5%.  MC flux convention (C' vs B):
  -14 / -9 / -4%.  The residual after harmonising: **+7-16% at 0.5 GeV,
  +0-12% at 1 GeV** (vertical: +4-10% and -3-+3%).
* The residual is the generator (b): with the MC's lower sub-GeV yields from
  low-energy primaries (table above) the cut removes less of the MC's flux.
  It is larger at the horizon than in the vertical; per stratum the MC/MCEq
  yield ratio at 87 deg over the vertical one is 0.4-0.8 for p 3.2-10 and He
  1-32 GeV/n and ~1 above (`reports/c3_horizon_yield.txt`; each 1-3 sigma).
  A likely reason, not proven: the MC's 1D tallies lay the neutrinos of a
  **3D cascade** on the primary's line, so at the horizon they inherit the
  3D losses of the wide-angle secondaries of low-energy primaries, which
  MCEq's 1D yield does not have -- the gate is then not purely a 1D check
  there.
* Measure: `v/t` in measure 0 (no `1/cos`) is within 0-5% of measure 1 in
  the West (`reports/c3_measure0.txt`); the heavy `1/cos` tail of the
  horizon bin is not the cause.

**Gate C3 re-run** (`reports/c3_rerun.txt`; the MC with the corrected
convention, exact re-weighting of the pilot and of `a1anc`):

| | vertical | 87 deg West (2 bins) |
|---|---|---|
| MC (crflux at the total energy) / delivered engine `G_s` | 1.12-1.19 (0.3-0.5 GeV), 1.01-1.07 (1 GeV); **chi2/12 = 31** (pilot) | 1.15-1.27 (0.5 GeV), 1.03-1.15 (1 GeV); chi2/12 = 13 |
| same MC (T >= 2) / MCEq B (true rigidity) | pilot chi2/12 = 15.6, a1anc 6.1: 1.05-1.10 at 0.3 GeV (a1anc), 1.04-1.11 at 0.5, 0.93-1.06 at 1 | pilot chi2/12 = 7.3-7.4: 0.89-1.16 at 0.3-0.5 GeV, 1.00-1.12 at 1 GeV |

### 1.1 Verdict (task 1)

The West failure is not a bookkeeping error in the MC's 1D cutoff (same
`R_c`, same direction, sharp on both sides; candidate (a) excluded, and the
re-weighting reproduces the recorded `v/t` digit for digit).  It is the sum
of four effects, which in the vertical happen to cancel:

1. **Engine bug (tools/mceq3d, not edited):**
   `MCEq3DFlux.geomag_response` cuts the primary flux with
   `_transmission(self.e, rc, ..., az_over_z)`, i.e. at `R = (A/Z) E` with
   `E = self.e = mceq.e_grid`, MCEq's **kinetic** energy grid
   (`MCEqRun.xgrid`: `e_grid` is kinetic, `etot_grid = e_grid + m`).  The
   rigidity is `(A/Z) p = (A/Z) sqrt(T (T + 2 m))`: at 7.5 GV the engine cuts
   protons below T = 7.50 instead of 6.62 GeV and bound nucleons below 3.75
   instead of 2.93 GeV/n.  **`G_s` is too low by 3.5-10% at Kamioka's
   cutoffs** (more the lower the energy).  Fix:
   `_transmission(np.sqrt(e * (e + 2 m_p)), ...)`.  Smaller engine-side items:
   `_nucleon_split` weights `<A/Z>` with crflux at `A e_grid` (kinetic) while
   MCEq's flux uses `A (e_grid + m)`; MCEq's primary flux starts at
   `E_tot` = 3 GeV (T >= 2.2 GeV) while H3a is defined below; the free/bound
   split `p - n` and `<A/Z>`.
2. **mc3d bug (fixed):** `primaries.nucleon_intensity` evaluated crflux at
   `A T` instead of the nucleus' total energy `A (T + m)` (crflux's
   documented argument and MCEq's): the MC's low-energy primary flux was
   x1.27 at 10, x1.47 at 6, x2.06 at 3 GeV/n.  It lowered the MC's `v/t` by
   4-28%, opposite to (1).  **New default `primaries.ENERGY_CONVENTION =
   "total"`**, `sky3d --primary-energy kinetic` reproduces milestones 1-3b
   (recorded in the run signature);
   regression test `test_m3c.py::test_primary_energy_convention_total_is_mceqs`.
   **Impact on milestone 3b's headline** (exact: `reweight_replay.py`
   re-weights every pending primary of the pilot by `Phi(A(T+m))/Phi(A T)`,
   re-traces the stored copy states and replays the tallies -- with f = 1 it
   reproduces the original t-files exactly; `reports/ew_conv.txt`): **every
   pooled a1/a0 drops coherently by 0.033-0.045** (paired errors
   0.001-0.005; e.g. 0.3 GeV nue 0.602 -> 0.559, numu 0.386 -> 0.342,
   antinumu 0.531 -> 0.489), **the species fan-out is unchanged** (pairwise
   differences within 0.01), the 3D flux at cos Z 0-0.3 falls by 11-17%, and
   **chi2/12 against Honda rises from 1.95 to 12.85** (MC errors only).  The
   primary spectrum at 5-30 GeV/n is a first-order systematic of the
   absolute East-West amplitude (Phase 1: +-10-15%); the MC's agreement with
   Honda's amplitudes in milestone 3b leaned on the kinetic-energy H3a
   (closer to HKKM's own spectrum there, or a coincidence -- not
   separated).  The default switch is a one-line change to revert
   (`primaries.ENERGY_CONVENTION`; the test asserts the default) if the PM
   prefers the 3b convention; what must not happen is a 3D/1D table whose 1D
   reference (MCEq, daemonflux) and 3D numerator use different
   conventions.
3. **[Corrected 2026-11-04 by milestone 3d: this was NOT the generator but an mc3d
   bookkeeping bug, the energy-conservation guard in `pool.EventPool.draw`, see
   MILESTONE3D_RESULTS.md sec. 3.  The original text follows.]** **Generator (b), not a bug:** the chromo DPMJET-III-19.3 cascades yield
   fewer sub-GeV neutrinos than MCEq's tables from low-energy primaries (0.5
   at 1-3, 0.75-0.8 at 3-10, ~0.9 at 10-32 GeV/n in the vertical; possibly
   less still at the horizon), which raises the MC's `v/t` by 5-15%.
4. **Pilot sampling:** the Neyman allocation left the p 1-3.16 GeV stratum
   at 2850 primaries; the pilot's `t` (and so `v/t` and `s/t`) misses its
   9-21% share at 0.3 GeV.

**Gate C3 cannot pass at the few-% level with two different generators**:
after both convention defects are removed it measures the generator's
low-energy yield difference (+5-15% on the transmission).  Proposal: define
the gate as MC (corrected convention, an allocation covering T >= 1 GeV/n)
vs MCEq with the true rigidity, with a tolerance set by the measured
generator difference -- or, for a few-% test of the bookkeeping alone,
compare the MC with itself: the MC's `v/t` against its own per-primary-energy
`t` histogram cut at the site-map rigidity (`c3_reweight.py` writes the
histograms), which removes the generator from the comparison.

---

## 2. The muon-bend mechanism (task 2)

### 2.1 Design

All runs on `a1anc`'s seeds, tasks and allocation (1e6 primaries, `E^-0.8`,
`r_thr 0`), so the cascades are identical run to run; verified: the
cutoff-independent tallies `t` and `n` of `a1nc` equal `a1anc`'s, and those of
`a1ncnomu` equal `a1nomu`'s, **bit for bit**.  Observable: the 3D tally `s`,
which with `--no-cutoff` is the all-copy no-cutoff tally (the pilot's `n` was
copy 0 only), pooled over cos Z 0-0.3.  The muon effect is the paired
difference field on - off, expressed as

    eps = [c1 a0 (on) - c1 a0 (off)] / a0 (off)

(`c1` = the dipole component towards geomagnetic West, signed, bin-width
deconvolved as in `analyse_m3b`): the change of the West dipole in units of
the field-off monopole.  `eps` of the muon-born species is the fan-out.

| run | cutoff | primary bend | muon field | role |
|---|---|---|---|---|
| `a1anc` (m3b) | on | on (re-anchored) | on | |
| `a1nomu` (m3b) | on | on | off | muon effect with the cutoff |
| `a1nc` | off | on | on | |
| `a1ncnomu` | off | on | off | muon effect without the cutoff |
| `a1ncnb` | off | off | on, `--mu-cf` | near-flat field-off pattern (copies are rotations about the site vertical; measured a1/a0 <= 0.025 in every band); in-run counterfactual |
| `a1ancd` | on | on | on, `--mu-cf` | in-run counterfactual with the cutoff |
| `a1ncnbnomu` | off | off | off | decay-pipeline check against `a1ncnb`'s counterfactual |

`--mu-cf` (new): every re-transported muon is also transported and decayed
with zero field, same proper time, same decay random numbers; the
difference splits exactly into *migration* (muons that decay either way:
only the neutrino directions differ), *saved* (decay only with the field)
and *lost* (only without).  The planned vertical-only / horizontal-only
runs (`--mu-bend vertical|horizontal`, implemented and tested) were dropped
for the flat-pattern run, which is more decisive: on a flat pattern a
horizontal rotation does nothing and an effective-cutoff change has nothing
to act on.

Hypotheses as posed: **H1** = the bend moves the daughter neutrinos in
direction, independently of the cutoff; **H2** = the bend changes the
effective cutoff seen through the muon's parent direction (the engine's
mean cone-axis shift).  H2 predicts `eps` = 0 without the cutoff.

### 2.2 Results

**The muon-induced dipole does not need the cutoff**
(`reports/mech.txt`, pooled cos Z 0-0.3, `eps` per species, 0.3 / 0.5 / 1 GeV):

| | cutoff on (`a1anc-a1nomu`) | cutoff off (`a1nc-a1ncnomu`) |
|---|---|---|
| nue | +0.29+-0.07 / +0.38+-0.10 / +0.28+-0.13 | +0.23+-0.08 / +0.10+-0.20 / +0.28+-0.11 |
| antinue | +0.02+-0.07 / -0.18+-0.08 / -0.24+-0.10 | -0.02+-0.09 / -0.23+-0.08 / -0.27+-0.16 |
| numu | -0.05+-0.03 / -0.05+-0.04 / -0.05+-0.04 | -0.09+-0.04 / -0.05+-0.04 / -0.05+-0.04 |
| antinumu | +0.15+-0.04 / +0.19+-0.05 / +0.18+-0.06 | +0.10+-0.04 / +0.16+-0.06 / +0.15+-0.09 |

Paired test `eps(cut) = eps(no cut)`: **chi2/12 = 0.69**; least-squares slope
`eps(cut) = k eps(no cut)`: **k = 1.04**.  Charge-odd parts (the fan-out):

| | (nue - antinue)/2 | (antinumu - numu)/2 |
|---|---|---|
| cutoff on | +0.138+-0.047 / +0.278+-0.072 / +0.262+-0.077 | +0.102+-0.028 / +0.120+-0.030 / +0.113+-0.039 |
| cutoff off | +0.127+-0.057 / +0.162+-0.102 / +0.274+-0.101 | +0.095+-0.023 / +0.108+-0.039 / +0.101+-0.058 |
| flat pattern (`a1ncnb`, no cutoff, no primary bend) | +0.115+-0.031 / +0.069+-0.026 / +0.264+-0.065 | -0.05+-0.13 / +0.091+-0.021 / +0.111+-0.024 |

**Bound on the cutoff-mediated part (H2)** (`reports/mech_h2bound.txt`):
the paired difference of the charge-odd parts with and without the cutoff,
over the 6 values, is +0.011 +- 0.015 (chi2/6 = 0.29), against a mean
charge-odd fan-out of 0.169: **the cutoff-mediated share is +0.07 +- 0.09**
(< 0.25 at 2 sigma).  The cutoff adds a small *charge-even* shift instead,
+0.01 to +0.17 (weighted ~+0.03).

**It is a direction migration, not a change of the muons' fate**
(`reports/mech2_a1ncnb.txt`, flat pattern, in-run counterfactual): of the
total `eps`, the part from muons that decay with and without the field is
100% (e.g. nue 0.3 GeV +0.113 +- 0.063 of +0.113); muons saved from the
ground or lost to it by the bend contribute <= 0.003 in `eps` and in `da0`.

**The bend itself** (`mub_m` of `a1nc`, parent muons of the 3D neutrinos in
cos Z 0-0.3, measure 0): the change of the arrival cos Z has a dipole
`-q 0.043-0.047 cos(az - W)` (mu+ arriving from the West turn up -- `q v x
B` with B northward -- their neutrinos appear lower; from the East, higher),
energy-independent at 0.3-1 GeV, rms 0.026-0.034 about the mean at fixed
azimuth; plus a uniform azimuthal rotation of `-q 3.2-3.5 deg`.  (5.35
deg per lifetime at 0.5 G with Kamioka's horizontal ~0.30 G and vertical
~0.35 G components.)

**Sign: not a simple shift of a cos Z distribution.**  The field-off flux
in cos Z 0-0.1 / 0.1-0.2 / 0.2-0.3 falls with cos Z (nue 0.3 GeV, flat run:
0.0022 / 0.0016 / 0.0015), so lowering the West mu+ neutrinos by 0.045 would
*reduce* the West nue flux; the MC instead finds the West nue flux raised
(+0.12 to +0.19 of the field-off flux per cos Z bin, East ~0), and the same
for every muon-born species on its bend-up side.  The neutrinos arriving
near the horizon at the cap are not a translation-invariant radiance field
on that scale (source region 100-1000 km away, curved Earth, 10 deg cap), so
the migration acts on the full 3D geometry.  The monopole of cos Z 0-0.3
rises too (`da0` +0.09 / +0.14 for nue / antinue at 0.3 GeV, flat run): the
field brings muon-decay neutrinos into the near-horizon band.

**With the cutoff, in-run** (`a1ancd`, `reports/mech2_a1ancd.txt`; it
reproduces `a1anc`'s `s`, `t`, `n`, `v` bit for bit, so the new code and the
diagnostic leave the physics tallies untouched): again 100% migration
(saved / lost <= 0.003); charge-odd parts (nue - antinue)/2 +0.105+-0.048 /
+0.203+-0.038 / +0.243+-0.061, (antinumu - numu)/2 +0.098+-0.017 /
+0.082+-0.014 / +0.145+-0.024 -- the same as without the cutoff and on the
flat pattern within the errors; charge-even parts +0.05 / +0.02 / +0.03 (e)
and 0.00 / +0.02 / +0.02 (mu).  The in-run counterfactual is the cleaner
estimate (same decays); the separate-run difference `a1anc - a1nomu` also
carries the two decay pipelines (below).

**Decay pipelines** (`reports/pipeline_check.txt`, `michel_check.txt`).
With the field on, the muon-decay neutrinos are regenerated
(`bend.transport_muons` + `bend.michel_decay`, keyed random numbers); with it
off, the cascade's own decays are used.  At zero field the two should agree.
`a1ancd`'s counterfactual against `a1nomu` (same cascades): flux ratio in
cos Z 0-0.3 1.08 / 1.13 / 0.93 (nue), 1.01 / 1.15 / 0.96 (antinue), 1.01 /
0.99 / 1.07 (numu), 1.08 / 1.05 / 1.06 (antinumu), errors 0.03-0.11,
chi2/12 = 1.81 against 1; a1/a0 differences chi2/12 = 1.54.  The two Michel
samplers agree (4e5 vs 6e4 decays at 0.6-4 GeV, polarisation -1 to +1, 96
band fractions: largest pull 2.8, differences <= 1-4% without a sign
pattern).  So a pipeline difference of a few % in the near-horizon flux is
neither established nor excluded (1.5-2 sigma); it would enter m3b's
"muon field off" comparison (sec. 5 of MILESTONE3_RESULTS) as a
charge-even shift.  Repeated without the cutoff and the primary bend
(`a1ncnb`'s counterfactual vs `a1ncnbnomu`, `reports/pipeline_check_nocut.txt`):
weighted mean flux ratio **0.985 +- 0.021**, chi2/12 = 1.26.  The two checks
pull in opposite directions (+4% with the cutoff, -1.5% without): **no
decay-pipeline difference is established; it is below ~3% in the
near-horizon flux.**

**Verdict: H1 (a cutoff-independent geometric effect of the bend on the
decay neutrinos' directions); H2 at most ~25% (2 sigma) of the charge-odd
fan-out and consistent with zero.**  The engine's term -- the decay cone's
cutoff evaluated at the mean-bend-shifted axis -- is an H2 mechanism; it
cannot produce what the MC shows.

### 2.3 What an engine fix needs (design note; `tools/mceq3d` not edited)

What the engine does: `MCEq3DFlux.cone_geff(..., muon_bending=True)` centres
the muon-decay part of the production cone on the primary direction shifted
by the one-lifetime mean deflection `(q tau/m) v x B`
(`muon_bending.bending_deflection`), i.e. it changes *which cutoff* the
decay neutrinos see.  That is H2; the MC bounds its charge-odd contribution
to 0.07 +- 0.09 of the fan-out.  The engine therefore cannot be repaired by
re-tuning that shift (an exponential bend distribution, energy loss, a
larger mean: all act through the cutoff and vanish without one).

What the fix must contain: a charge x azimuth-dependent change of the
muon-decay neutrino flux itself, present with no cutoff at all.  Two routes:

1. **Per-species MC response table (recommended, cheap, exact for the
   factorisation measured here).**  `R_s(E, cos Z, az) = F_s(field on) /
   F_s(field off)` from one `--mu-cf` run with no cutoff and no primary bend
   (the counterfactual is in the same run: no second run, no decay-pipeline
   difference; `a1ncnb`: 1e6 primaries, 4 core-h, `eps` errors 0.02-0.06
   pooled over cos Z 0-0.3).  The engine runs with `muon_bending=False` and
   multiplies by `R_s`.  Justification: `eps` is the same with the cutoff
   (slope 1.04, chi2/12 = 0.69) and with or without the primary bend (flat
   run); the residual cross term is charge-even, ~+0.03 in `eps`.  A
   production table resolved in (cos Z, az) bins needs ~10x the statistics
   (~40 core-h per site), or the smooth form `R_s = 1 + m_s(E, cz) + q_s
   A_s(E, cz) cos(az - W) + ...` fitted to the pooled run.  Site
   dependence: scale with the horizontal field component that sets the
   vertical bend (`-q 0.045 cos(az - W)` in cos Z at Kamioka).
2. **Analytic.**  A zenith-shift kernel applied to the 1D zenith
   distribution has the wrong sign (sec. 2.2: the field-off flux falls with
   cos Z, the bend-up side nevertheless gains).  The bend must instead enter
   the engine's 3D production-angle machinery -- the muon-decay part of the
   kernel that maps production directions and points onto the cap
   (`angular_factor` / the off-axis tables) -- as a charge x azimuth
   dependent rotation of the decay-neutrino directions at the production
   point (vertical `-q 0.045 cos(az - W)`, horizontal `-q 3.3 deg`, rms
   0.03, energy-independent at 0.3-1 GeV), before the cap geometry.  This
   is a new piece of the engine, to be validated against `R_s` of route 1.

---

## 3. Paired re-anchor vs re-simulate (task 3)

### 3.1 Design

Two runs on the same seed (20261104), tasks and allocation -- the pilot's
Neyman allocation x 0.2 (`reports/alloc_m3c_0p2.npz`, 2.89e5 primaries,
every cell >= 120), chunk 1600 as the pilot -- both with `--crn --reanchor`:

* `c3anc`: `r_thr 0` -- every cascade reused by rotation and re-anchored on
  each copy's own exact bent primary (the production candidate);
* `c3res`: `r_thr 65` -- below 65 GV every copy re-simulated with its own
  bent primary (the pilot's scheme); above 65 GV re-anchored as in `c3anc`.

With common random numbers (sec. 4) the two runs are paired copy by copy:
copy 0 is the same cascade in both (same stream, same first interaction,
same muon decays: `test_m3c.py::test_crn_resimulated_copy_starts_where_the_
reanchored_copy_does`), every primary above 65 GV is identical in both, and a
re-simulated copy `j` starts from the same first-interaction state as the
re-anchored copy `j` (same sampled depth, the copy's own bent primary;
within 20 m) and draws the same random numbers until the cascades diverge
through the geometry.  The difference `c3res - c3anc` is therefore the
re-anchoring bias below 65 GV, with the cascade noise largely cancelled.
Jackknife: the same task -> group assignment in both runs, so the replicates
of the difference are the differences of the replicates.

### 3.2 Results (`reports/pair.txt`, `pair_combined.txt`, `pair_strata.txt`)

CPU: `c3anc` 2.78 core-h, `c3res` 11.62 core-h (x4.2, same primaries).

`d = a1/a0(re-simulated) - a1/a0(re-anchored)`, paired jackknife:

| | mean d | chi2(d = 0) | median error per value |
|---|---|---|---|
| 36 cells | **-0.014 +- 0.022** | 37.1 / 36 (1.03) | 0.084 (unpaired 0.100) |
| pooled cos Z 0-0.3 (12) | **-0.017 +- 0.025** | 21.1 / 12 (1.76) | 0.054 (unpaired 0.063) |

Pooled values (0.3 / 0.5 / 1 GeV): nue +0.07+-0.06 / -0.04+-0.05 /
-0.12+-0.08, antinue +0.10+-0.06 / -0.09+-0.10 / -0.01+-0.11, numu
+0.12+-0.04 / -0.05+-0.04 / -0.08+-0.05, antinumu +0.03+-0.04 / -0.06+-0.05
/ -0.07+-0.09.

* **The pairing bought only x1.1 in precision** (paired vs unpaired errors),
  not the >~5 hoped for.  Common random numbers keep copy 0 and everything
  above 65 GV identical, but a re-simulated copy below 65 GV, started from
  the same state with the same random numbers, decorrelates from the
  re-anchored one within its first few interactions: the axis zenith differs
  by the bend difference (1-20 deg), so the slant-depth profile and with it
  the first decay-vs-interaction decisions differ, and the stream
  desynchronises.  **The target precision (<~ 0.01) was not reached**; with
  this design it would need ~25x the CPU (~350 core-h).
* **Combined with the earlier unpaired comparison** (pilot - `a1anc`,
  independent samples; corrected by +0.003 for the pilot's rotation above
  65 GV, gate C5): pilot - a1anc -0.024 +- 0.017 (36 cells), this test
  -0.014 +- 0.022, **combined -0.020 +- 0.014 (36 cells), -0.018 +- 0.015
  (pooled): |bias of re-anchoring| < 0.047 at 2 sigma**, with a 1.4 sigma
  hint that re-anchoring sits ~0.02 high.
* **Per stratum** (the mean over the 12 pooled values when only stratum h is
  re-simulated, paired): **p 32-100 GeV (re-simulated below 65 GV) -0.026 +-
  0.008 (3.2 sigma)**; p 3.2-10 -0.007 +- 0.008, p 10-32 -0.006 +- 0.010, He
  3.2-10 +0.005 +- 0.004, CNO 3.2-10 +0.007 +- 0.003, every other stratum
  |d| <= 0.004.  The one significant contribution is where the bend is
  smallest (1-2 deg); not understood.
* **Cost per unit variance**: with the same primaries (Neyman) the pooled
  variances are 1.3x larger re-anchored, at 4.2x less CPU: re-anchoring is
  **x3.2 cheaper per unit variance** (milestone 3b estimated ~5 from the
  `E^-0.8` runs).

**Verdict.**  Re-anchoring shows no bias above +-0.05 (2 sigma) overall,
but a 3 sigma bias of +0.026 from protons of 32-65 GV, and the test is not
precise enough to certify it at the 0.01 level the pilot's statistics
(0.01-0.02) require.  Production options, in order of preference:
(1) the hybrid -- re-simulate protons of 32-65 GV (and, if affordable, all
p below 65 GV), re-anchor the rest: 3.9 core-h on this allocation vs 2.8
(all re-anchored) and 11.6 (pilot scheme), i.e. x1.4; (2) re-anchor
everything and carry +-0.03 on `a1/a0` as a systematic.  A sharper test
needs a better-paired design (e.g. common random numbers per *particle*,
not per primary, so a geometry-dependent decision does not desynchronise the
rest of the cascade) rather than more primaries.

---

## 4. Code changes (`tools/mc3d`; `tools/mceq3d` untouched)

| file | change | default behaviour |
|---|---|---|
| `sky3d.py` `Options(crn=...)`, `--crn` | common random numbers per primary: every realisation of primary `i` draws its cascade from the stream `(seed, CRN_KEY, i)`, every bent copy its first-interaction depth from `(seed, 7919, i)` (one depth per primary, as the re-anchoring assumes), every realisation its muon-decay numbers from `(seed, i, 0, 0)`.  Re-simulated copy 0 is then the re-anchored base cascade, and re-simulated copy `j` starts where the re-anchoring puts copy `j` | off; stream unchanged |
| `sky3d.py` `Options(mu_bend=...)`, `--mu-bend full\|vertical\|horizontal\|none`, `split_bend` | diagnostic: keep only the vertical / horizontal component of each muon's bend at its decay point; `none` = the muon-field pipeline (per-copy re-transport, keyed Michel decay) with zero field | `full` |
| `sky3d.py` `Options(mu_cf=...)`, `--mu-cf`, `Tally.mucf` | diagnostic: the zero-field counterfactual of every re-transported muon (same proper time and decay numbers) and the fate classes (both / saved / lost) of the muon-born 3D entries, field on and counterfactual, measure 1 | off |
| `sky3d.py` `bend_components`, `Tally.mub_m`, `mub_h` | always on: vertical (`d cos Z`) and horizontal (`d az`) bend of the parent muon of every muon-born 3D entry in cos Z 0-0.3 and the three analysis bands: moments and histograms (measure 0, the `s` weight) | new keys in the t-files; physics tallies unchanged |
| `sky3d.py` signature / manifest | the new options enter `Options.signature()` only when not at their default, so m3b manifests stay valid; a shape mismatch is reported as "a different job" | |
| `sky3d.py` `cascade_task` | the first-interaction state is kept for `C` realisations too (diagnostics) | |
| `primaries.py` `ENERGY_CONVENTION`, `crflux_energy`, `nucleon_intensity(..., convention)`, `GSFNucleons.nucleon_flux(..., convention)`; `sky3d --primary-energy` | the crflux energy variable: `"total"` = `A (T + m)` (crflux's documented argument and MCEq's), `"kinetic"` = `A T` (milestones 1-3b) | see sec. 1 |

Pairing checks of the default path (no option set): `a1nc`/`a1ncnomu` (new
code) reproduce `a1anc`/`a1nomu` (m3b code) bit for bit in every
cutoff-independent tally (`t`, `n`); `a1ancd` (`--mu-cf`) reproduces
`a1anc`'s tallies (sec. 2).

---

## 5. Tests

`test_m3c.py` (new, 7 tests): the vertical/horizontal split of a bend and
its components; the bend sign in a northward and a vertical field (right-hand
rule, mu+ moving East turns up); common random numbers pair a re-simulated
copy 0 with the re-anchored base (identical 3D entries) and a re-simulated
copy `j` with the re-anchoring's first-interaction state (one depth per
primary, within 20 m); the default stream / signature and the `mub`
histograms; `mu_bend="none"` records zero bend; the `mu_cf` counterfactual
equals the field-on entries bin by bin at zero bend and leaves the physics
tallies unchanged; the primary-energy convention (`"total"` summed over
crflux's groups is exactly MCEq's `p_and_n_flux(T + m)`, the drivers' weights
follow the module setting, the default is `"total"`, the signature records
it).  `test_m3b.py` 15/15 unchanged.  Full suite: see the status table.

---

## 6. Runs and environment

Every run: `source /home/pigrange/mc3d_work/runs/m3c/env.sh` (= m3b's
`env.sh`: `$W/pylib_fix` with the SIBYLL nucleon fix, `MC3D_POOL_CACHE=
$W/pools_fix`, `MC3D_FIELD_CACHE` = m3b's field cache, field grid and site map
copied into `$M/work`; `M=$W/runs/m3c`), `--nproc 24`, run from `$M/work`,
outputs in `$M/sky`.  Interaction lengths: `tools/mc3d/mceq_tables_SIBYLL23D.npz`
(built from the shipped v1.4.0 MCEq file) -- the m3b setup, unaffected by the
other agent's switch of `tools/mceq3d` to the patched v1.4.2 file (11:27 on
2026-11-04; the MC does not import `mceq3d_db`).  Primary flux: the
milestone-3b convention (crflux at `A T`) in every run (pairing); the
default was switched to the total energy only after the last run had
started.

| run | arguments (besides the common ones) | primaries | core-h | wall |
|---|---|---|---|---|
| `a1nc` | `--nshower 1e6 --chunk 10000 --nstrata 7 --seed 31001 --r-thr 0 --reanchor --no-cutoff` | 1e6 | 3.80 | 0.17 h |
| `a1ncnomu` | same + `--no-muon-field` | 1e6 | 3.05 | 0.14 h |
| `a1ncnb` | `... --r-thr 0 --no-cutoff --no-bend-primary --mu-cf` | 1e6 | 4.38 | 0.26 h |
| `a1ancd` | `... --r-thr 0 --reanchor --mu-cf` | 1e6 | 5.09 | 0.28 h |
| `c3anc` | `--nshower 0 --alloc $M/reports/alloc_m3c_0p2.npz --chunk 1600 --seed 20261104 --crn --reanchor --r-thr 0` | 2.89e5 | 2.78 | 0.15 h |
| `c3res` | same with `--r-thr 65` | 2.89e5 | 11.62 | 0.52 h |
| `a1ncnbnomu` | `... --r-thr 0 --no-cutoff --no-bend-primary --no-muon-field` | 1e6 | 2.77 | 0.12 h |
| **total MC** | | | **33.5** | |

Non-MC jobs (1 core each): `mceq_response.py` (MCEq per-cell responses,
cos Z 0.95 and 0.05; database: the mceq3d default at start time = **patched
v1.4.2** -- the response differs from v1.4.0's by <= 0.2% per cell below
31.6 GeV/n, +4.7% in 31.6-100 GeV/n (the 56 GeV hole), +2% above, i.e. <= 1%
on `G`: `reports/db_v140_vs_v142p.txt`, from a v1.4.0 re-run at cos Z 0.95),
`penumbra.py`, `c3_reweight.py`, `extract_strata.py`, `michel_check.py`,
`reweight_replay.py` (the pilot under the corrected primary convention,
sec. 1).

---

## 7. Reproducing

```sh
source /home/pigrange/mc3d_work/runs/m3c/env.sh       # PY, R, W, M; pylib_fix, pools_fix
cd $M/work                                             # chromo writes tables.dat into the cwd
export PYTHONPATH=$PYTHONPATH:$M/scripts               # the analysis helpers (m3c_lib)
cp $W/runs/m3b/work/{sitemap_*,field_IGRF14_*}.npz $M/work/
# the runs (sec. 6), all with the milestone-3b primary convention: queue 1 ran
# a1nc, a1ncnomu, c3anc (run_m3c.sh, stopped there: its a1vert/a1horiz lines were
# never run), queue 3 the rest.  Today add --primary-energy kinetic to every line.
$M/scripts/run_m3c.sh ; $M/scripts/run_m3c3.sh 0
# per-(jackknife group, stratum) sums of every run used
for t in pilot a1anc a1nomu; do $PY $M/scripts/extract_strata.py $W/runs/m3b/sky $t; done
for t in a1nc a1ncnomu a1ncnb a1ancd c3anc c3res a1ncnbnomu; do $PY $M/scripts/extract_strata.py $M/sky $t; done
# task 1 (gate C3)
$PY $M/scripts/penumbra.py                             # (a)
$PY $M/scripts/h3a_conventions.py                      # (c)
$PY $M/scripts/mceq_response.py 0.95 0.05              # MCEq per-cell response, ~2 h (87 deg: 2.4 min/solve)
MCEQ3D_DB=v140 RESP_SUFFIX=_v140 $PY $M/scripts/mceq_response.py 0.95   # database check
for t in pilot a1anc; do $PY $M/scripts/c3_reweight.py $W/runs/m3b/sky $t; done
$PY $M/scripts/c3_rew_report.py pilot a1anc
$PY $M/scripts/c3_conventions.py                       # -> reports/c3_conventions.txt
$PY $M/scripts/reweight_replay.py $W/runs/m3b/sky pilot 24   # pilot under crflux at A(T+m)
$PY $M/scripts/extract_strata.py $M/sky pilot_tot ; $PY $M/scripts/ew_conv.py
# task 2
$PY $M/scripts/mech.py ; $PY $M/scripts/mech2.py a1ncnb a1ancd
$PY $M/scripts/michel_check.py
# task 3
$PY $M/scripts/pair.py c3anc c3res
$PY $M/scripts/cpu_table.py
$PY -m pytest $R/tools/mc3d -q
```

Scripts in `$M/scripts/`, copied to `results/m3c/scripts/`; small reports in
`results/m3c/`.
