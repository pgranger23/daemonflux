# Super-Kamiokande East-West asymmetry as an arbiter between this engine and Honda

New files: `diag_sk_ew.py` (this directory).
Raw output: `<scratch>/sk/final_full.txt`, `sk_significance.txt`, `solve_{down,up,up2}.log`,
grids `grid_down.npz` + `grid_up.npz`.

## 1. What SK actually measured

**Futagami et al., "Observation of the east-west anisotropy of the atmospheric
neutrino flux", PRL 82 (1999) 5194, arXiv:astro-ph/9901139.** (Note: the paper is
`astro-ph/9901139`, not `hep-ex/…`, and the title is "Observation of…", not
"Evidence for…".) 45 kt·yr, FC single-ring, 400 < p_lepton < 3000 MeV/c,
|cos θ_lepton| < 0.5, 552 e-like / 633 mu-like events.
`A = (N_E − N_W)/(N_E + N_W)`, where φ is the direction the **lepton travels**
(caption of Fig. 2: "φ = 0, π/2, π, 3π/2 shows particles *going to* north, west,
south, east"), and the two bins are **hemispheres**, not narrow sectors. The paper
publishes **no numbers** — only χ² and Kuiper probabilities. The values quoted in
the literature come from Lipari, Astropart. Phys. 14 (2000) 171
(arXiv:hep-ph/0003013), §1:

    A_e   = 0.21 ± 0.04     A_mu   = 0.08 ± 0.04     (statistical only)
    A_e^HKKM = 0.13         A_mu^HKKM = 0.11         (HKKM 1995, 1D)
    A_e^Bartol = 0.17       A_mu^Bartol = 0.15

**SK-I..IV, PRD 94 (2016) 052001, arXiv:1510.08127**, §IV "Azimuthal Spectrum
Analysis". 4799 days, 13 079 e-like / 12 725 mu-like. Same hemisphere definition,
written explicitly: `A = (n_east − n_west)/(n_east + n_west)`, n_west = events with
azimuth 0–180° (**west-going**), n_east = 180–360°. Optimised sub-sample
0.4 < E_rec < 3.0 GeV, |cos θ_rec| < 0.6. Headline:

    A_mu = 0.108 ± 0.014(stat) ± 0.004(syst)   6.0 σ
    A_e  = 0.153 ± 0.015(stat) ± 0.004(syst)   8.0 σ

plus a 2.2 σ indication that the dipole phase B rotates with zenith
(HKKM11 predicts ≈ −7° down-going → +22° up-going). Goodness of fit against
HKKM11: χ² = 87.6/96 bins; re-weighting to Bartol-2003 changes χ² by 1.0, i.e.
SK cannot separate HKKM11 from Bartol.

**No later SK azimuthal analysis exists.** arXiv:1710.09126 and arXiv:2311.05105
(SK I–V) both state the oscillation baseline "does not depend on the azimuth" and
carry no E-W observable; no SK-Gd azimuthal paper. 1510.08127 remains definitive.

Two cautions carried over from the literature retrieval:
* SK-2016's **quoted A values do not reproduce its own Fig. 22**: the plotted
  histogram (6547 e-like, 8162 mu-like) gives A = 0.1306 / 0.0877 with MC 0.1145 /
  0.0929, and the quoted stat errors correspond to N ≈ 4340 / 5043, not the plotted
  totals. The ~15–20 % offset is unexplained. Everything below is therefore quoted
  **as a ratio to SK's own HKKM11 MC**, which is insensitive to that offset.
* Futagami's convention (0 = N, 90 = W) and SK-2016's (0 = S, 90 = W) are mirror
  images; they agree on the E/W hemispheres, so the split is common to both.

## 2. Sign and convention chain

SK "east-going" = neutrino **arriving from the west**. Honda's azimuth is measured
counter-clockwise **from South** on the **arrival** direction; our engine uses
compass arrival azimuth (0 = N, 90 = E), `az_compass = (180 − az_Honda) mod 360`.
Hence the flux-level quantity that carries SK's sign is

    A_flux = (Φ_from-W − Φ_from-E) / (Φ_from-W + Φ_from-E) .

All A below are positive, as in the data.

## 3. Grid and weights

* **cosZ**: Honda's ten 0.1-wide bins with |cosZ| < 0.5, evaluated at the bin
  centres ±0.05, ±0.15, ±0.25, ±0.35, ±0.45 (Futagami's exact cut; also Lipari's).
  Up-going bins use the engine's far-side production-point cutoff — the joint
  production cone / cone-averaged cutoff is **not** applied there (`solve()` falls
  back to a single far-side R_c for cosZ < 0), so the up-going half of our numbers
  is a coarser treatment than the down-going half.
* **azimuth**: Honda's twelve 30° bins, compass centres 15°, 45°, …, 345°. Our
  engine is evaluated at those centres (point samples vs Honda's bin averages —
  irrelevant for a 180°-wide sector).
* **sectors**: hemispheres, |Δaz| < 90° about 90° (East) and 270° (West), i.e. six
  Honda bins each — SK's definition. Half-widths 30°/60° also scanned.
* **energy**: E_ν ∈ [0.5, 3] GeV nominal (Lipari's window; the sensible proxy for
  400–3000 MeV/c lepton momentum), with [0.4, 3], [0.5, 5], [0.3, 1.5], [0.5, 1.6],
  [1.6, 3.3], [3.3, 10] also computed.
* **weights**: rate ∝ Φ(E)·σ_CC(E) with σ ∝ E and σ(ν̄)/σ(ν) = **0.45**; the ν and
  ν̄ fluxes are combined as Φ_ν + 0.45 Φ_ν̄. Cross-checked against Lipari's Table 1,
  whose per-species and per-flavour entries imply an effective ν̄ weight of
  0.38 (e) / 0.42 (mu) — consistent. Results are flat in this choice
  (rbar 0.35→0.50 moves A_e by 0.005).
* Engine: `MCEq3DFlux(base_model="hybrid", primary=GlobalSplineFitBeta,
  daemonflux_location="kamioka")`, bare `solve()` (delivered 3D: offaxis, joint
  cone, channel mode, v2 moments, `sublimb="prod_point"`, muon bending), Kamioka
  36.43 N / 137.31 E, IGRF epoch 2020-01-01. Honda = `honda_kam.npz` (HKKM2014).

## 4. Results, flux level (E_ν 0.5–3 GeV, |cosZ| < 0.5, hemispheres, rbar 0.45)

| species | Honda HKKM2014 | this engine | ours/Honda | Lipari 2000 3D (ref.) |
|---|---|---|---|---|
| ν_e      | +0.2852 | +0.2278 | 0.80 | +0.335 |
| ν̄_e      | +0.1078 | +0.1977 | 1.84 | −0.065 |
| ν_μ      | +0.1532 | +0.2009 | 1.31 | +0.028 |
| ν̄_μ      | +0.2349 | +0.2038 | 0.87 | +0.240 |
| **e-like**  | **+0.2375** | **+0.2196** | **0.925** | +0.224 |
| **mu-like** | **+0.1781** | **+0.2018** | **1.133** | +0.091 |

Zenith bands (same window):

| band | e-like Honda / ours (ratio) | mu-like Honda / ours (ratio) |
|---|---|---|
| cosZ ∈ [−0.5, −0.2] | 0.1431 / 0.1399 (0.98) | 0.1145 / 0.1322 (1.15) |
| cosZ ∈ [−0.2, +0.2] | 0.3197 / 0.2855 (0.89) | 0.2315 / 0.2665 (1.15) |
| cosZ ∈ [+0.2, +0.5] | 0.1999 / 0.2012 (1.01) | 0.1625 / 0.1807 (1.11) |

Robustness of the two flavour ratios: sector half-width 30/60/90° → e-like
0.950/0.935/0.925, mu-like 1.142/1.139/1.133. rbar 0→1 → e-like 0.80→1.04,
mu-like 1.31→1.05 (the flavour-summed ratios are stable near the physical
rbar ≈ 0.45). Adding 2-flavour ν_μ disappearance (Δm² 2.5e−3, maximal) changes
A_mu by −3 % and the ratio not at all (oscillation is azimuth-flat).

## 5. Confrontation with the data

Because the ν→lepton smearing kernel is a property of the interaction, not of the
flux, the **ratio** ours/Honda survives to the lepton level to good approximation
(and it is nearly energy-independent: 0.937 at 0.5–1.6 GeV, 0.885 at 1.6–3.3 GeV
for e-like; 1.125 / 1.163 for mu-like). So the lepton-level prediction of this
engine is SK's own HKKM11 MC scaled by that ratio.

| | SK data | SK's HKKM11 MC | this engine (MC × ratio) |
|---|---|---|---|
| A_e, optimised sample (Fig. 22) | 0.1306 ± 0.0129 | 0.1145 (+1.25 σ) | 0.1059 (+1.92 σ) |
| A_μ, optimised sample | 0.0877 ± 0.0117 | 0.0929 (−0.44 σ) | 0.1053 (−1.50 σ) |
| A_e, 0.40 < E_rec < 1.33 | 0.1346 ± 0.0140 | 0.1227 (+0.85 σ) | 0.1150 (+1.40 σ) |
| A_e, 1.33 < E_rec < 3.00 | 0.1178 ± 0.0251 | 0.0885 (+1.17 σ) | 0.0783 (+1.57 σ) |
| A_μ, 0.40 < E_rec < 1.33 | 0.0930 ± 0.0125 | 0.0988 (−0.47 σ) | 0.1112 (−1.46 σ) |
| A_μ, 1.33 < E_rec < 3.00 | 0.0690 ± 0.0235 | 0.0718 (−0.12 σ) | 0.0836 (−0.62 σ) |

Σ Δχ² (ours − Honda) = **4.15** on the optimised sample and **4.63** on the four
E_rec bins, i.e. a **≈ 2.0–2.2 σ preference for Honda over this engine**, with no
free parameters. The 1999 measurement adds essentially nothing:
A_e = 0.21 ± 0.04 and A_mu = 0.08 ± 0.04 have 20 % and 49 % relative errors, and at
the flux level (no smearing) both models sit above them by construction.

## 6. Verdict

* The SK azimuthal data **lean against this engine and toward Honda, but only at
  ≈ 2 σ**. That is suggestive, not a discrimination. SK's own statement that
  re-weighting HKKM11 to Bartol changes χ² by 1.0 sets the scale: the current
  precision cannot separate two 3D flux models that differ by ~10 % in A.
* The direction of the lean is unambiguous and matches the known internal defect.
  The data want **more** ν_e asymmetry than HKKM11 (+9 to +14 %) and **slightly
  less** ν_μ asymmetry (−4 to −6 %). Our engine goes the **opposite way on both**
  (−7.5 % on e-like, +13 % on mu-like), because it compresses the four species'
  asymmetries onto a common value (0.198–0.204 vs Honda's 0.108–0.285, Lipari's
  −0.065 to +0.335) instead of splitting them by the muon-bending charge sign. Two
  independent calculations (HKKM2014 and Lipari 2000) agree on the ordering
  ν_e > ν̄_μ > ν_μ > ν̄_e, and the sub-GeV SK data reinforce it. The species
  compression is a real deficiency, not a Honda artefact.
* The zenith trend is the same story: our A_e/Honda ratio is 1.01 down-going,
  0.89 at the horizon, 0.98 up-going — we under-produce the horizon peak — while
  SK's e-like A vs cos θ shows a slightly *steeper* horizon rise than HKKM11.
* **The flux-level comparison alone is inconclusive at the ~1 σ level per
  flavour.** To turn it into a real test one needs: (i) the ν→lepton kernel folded
  in with SK's own opening-angle distributions (modal angle ~70° below 400 MeV,
  38° at 0.4–0.7 GeV, 8° above 1.33 GeV) — this dilutes A by roughly ×0.7 and is
  the dominant effect; (ii) the CCQE/RES/DIS cross-section mix with the correct
  ν/ν̄ ratio, since the four species carry different (and for ν̄_e, per Lipari,
  opposite-sign) asymmetries; (iii) SK's fiducial/PID acceptance and the 9 %/1.3 %
  NC contamination, both of which dilute A; (iv) three-flavour oscillations with
  matter effects for the up-going half — azimuth-flat to first order, so harmless
  for A within a zenith bin, but it re-weights the up-going bands against the
  down-going ones; and (v) SK's |cos θ_rec| < 0.6 rather than the < 0.5 used here.
  A cleaner and cheaper target than reproducing SK's A is the **dipole phase B vs
  zenith** (SK Fig. 23): it is a shape observable, largely immune to the
  normalisation and smearing dilution, and our engine already matches Honda's
  phase to ≤1.3° after the production-point cutoff anchor.
