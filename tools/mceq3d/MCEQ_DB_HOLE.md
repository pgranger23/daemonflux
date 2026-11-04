# MCEq database hole at E_kin = 56.2 GeV (SIBYLL/QGSJET columns are 0.8 x DPMJET)

Status: measured 2026-10-11 with `tools/mceq3d/diag_mceq_db_hole.py` (this note is its
write-up). Nothing was posted anywhere; the issue text in section 7 is a draft only.
Raw outputs: `/home/pigrange/mc3d_work/runs/dbhole/` (`confirm*.txt`, `olddb.txt`,
`flux_*.npz`, `gs_*.npz`, `tables.md`, patched databases `data_<variant>/`).

## 1. Summary

* The `SIBYLL23D` interaction matrices of MCEq's shipped `mceq_db_lext_dpm193_v140.h5`
  (installed MCEq 1.4.1) have a **20% hole in the projectile bin E_kin = 56.2 GeV**
  (grid index 37, bin edges 50.1-63.1 GeV): every non-empty yield cell of every one of the
  448 tuples (16 projectile species: p, n, pbar, nbar, pi+-, pi0, K+-, K0L, Lambda,
  Lambdabar, D+-, D0, Dbar0; all secondaries) is exactly `0.800000000000 x` the
  `DPMJETIII193` cell (13,480 cells), **and so is the inelastic cross section of all 13
  tabulated projectiles** (p-air 218.5 mb against 273.4 / 273.9 mb in the neighbouring
  bins). Below 44.7 GeV the columns are bit-identical to DPMJET (0 of 448 tuples differ),
  so this is "DPMJET blended with weight 0.8 against an empty high-energy column"
  (our reading; the database build script is not available).
* It is **not specific to SIBYLL23D**: the same defect is in SIBYLL21, SIBYLL23E and its
  `STAR` variants, QGSJETII04 and QGSJETIII (air) and SIBYLL23E (hydrogen). DPMJETIII193,
  EPOSLHC and EPOSLHCR are clean. It is **not fixed in the v1.4.2 database**
  (`mceq_db_lext_dpm193_v142.h5`, "bugfix release of the database", 2026-09-08), and the
  yield half is already in the 2020 v1.2.0 database (`mceq_db_lext_dpm191_v12.h5`; there
  the cross sections are smooth, the cross-section half is new in v1.4.0).
* Effect on 1D fluxes (SIBYLL23D, CORSIKA US standard atmosphere, patched/original,
  section 5; ranges over the four neutrino species, three zenith angles, H3a and GSF):
  neutrinos **+0.25-0.35% at 0.1 GeV, +0.4-0.6% at 0.3 GeV, +1.0-1.3% at 1 GeV,
  +2.1-2.4% at 3 GeV, +1.2-2.7% at 10 GeV, -0.7..+0.2% at 30 GeV, < 0.05% above
  100 GeV**; muons +0.3-1.5% below 1 GeV, up to +2.7% at ~10 GeV, and a notch of
  -2.2/-2.3% at 45 GeV (cos Z = 1; -1.0% at cos Z = 0.5). Maximum over everything:
  **+3.05% (GSF, nu_e, cos Z = 1, 7 GeV); H3a +2.91% (nu_e, 5.6 GeV)**. The yield half
  dominates the neutrinos (and the +1-3% muons below 20 GeV); the cross-section half
  (longer interaction length, hence more pion/kaon decays at 50-63 GeV) makes the muon
  notch at 45 GeV and the -1.7% nubar_mu notch at 22 GeV.
* `tools/mceq3d`: the **hybrid base** is only affected through its MCEq half
  (weight 1-w, w = 0.5 [1 + tanh(ln(E/1.7)/ln 2)]): +0.27% (0.1 GeV), +0.49% (0.3),
  +0.91% (1), +0.37% (3 GeV), 0.01% (10 GeV) for nu_mu, i.e. sub-percent. The **G_s
  response** is changed by <= 0.7% for R_c <= 11 GV, but by **+3.1% at R_c = 40 GV for
  E <= 0.3 GeV** (+2.6-2.8% at 1 GeV, +1.2-1.5% at 3 GeV); the near-horizon
  East-West amplitude G(7.1 GV)/G(40 GV) is **2.7-2.8% too large at 0.5 GeV, 2.4-2.6% at
  1 GeV and 1.7-1.9% at 2 GeV** (2-3 points of the +12/+21/+25% nu_mu overshoot at
  0.5/1/2 GeV listed in PHASE1_RESULTS; a minor contributor).
* **daemonflux** (conventional flux = GSF primary + DDM yields): in a DDM-injected MCEq
  calculation the yield hole changes **no flux by more than 0.07%** (the DDM replaces the
  p and pi yields into p, n, pi, K); only the cross-section half survives (relevant only
  for databases that have it, i.e. v1.4.x): <= 1.7% for neutrinos (22 GeV) and
  <= 2.5% for muons (45 GeV). What cannot be concluded: which database the 2023 splines
  were built with (section 6).
* Side finding (not about this hole): the upstream asset for the v1.4.0 database was
  renamed `mceq_db_lext_dpm193_v140_do_not_use.h5`; the v1.4.2 release notes say that in the
  v1.4.0/v1.4.1 databases the Sibyll neutron projectiles were simulated as protons
  (chromo#275) and that DPMJET-III-19.3 antiparticle cross sections were wrong. The
  installed 1.4.1 therefore carries both problems; v1.4.2's database still has the hole.

## 2. The hole (task 1; `diag_mceq_db_hole.py confirm`)

Database attributes: `he_le_transition_gev = 80`, `low_energy_model_ext = DPMJETIII193`;
`mceq.config.low_energy_extension = {"he_le_transition": 80, "nbins_interp": 3, ...}`.
The grid is 10 bins/decade in kinetic energy (index 36/37/38/39 = 44.7/56.2/70.8/89.1 GeV).
Proton projectile, quantities per interaction (energy sum rule = sum_i E_i N_ij / E_j,
N = M E_i/E_j):

| E_kin proj [GeV] | sum rule SIBYLL23D | sum rule DPMJET | pi+ mult. SIBYLL23D | pi+ mult. DPMJET | sigma_inel(p-air) SIBYLL23D / DPMJET |
|---|---:|---:|---:|---:|---:|
| 44.7 | 0.9645 | 0.9645 | 3.0023 | 3.0023 | 1.0000 |
| **56.2** | **0.7753** | 0.9691 | **2.5658** | 3.2072 | **0.8000** (218.5 / 273.1 mb) |
| 70.8 | 0.9746 | 0.9735 | 3.2887 | 3.4367 | 0.9953 |
| 89.1 | 0.9792 | 0.9777 | 3.4284 | 3.6680 | 0.9911 |

* Only bin 37 deviates (scan of all 16 projectiles over 20-160 GeV: the sum rule drops
  by >10% in no other bin). All secondaries of every projectile have the same factor
  0.800000000000 (min = max over the 13,480 cells), so it is a column weight, not a
  physics difference.
* The 70.8 GeV column is a regular, energy-conserving blend: it equals
  (1/3) x col(44.7) + (2/3) x col(89.1) to 1-3% (least-squares fit of the x-shifted
  neighbours: a, b = 0.32, 0.68 for p -> pi+; residual 1.0%; `confirm`, item [6]). A
  linear interpolation in bin index between the last pure-DPMJET bin (44.7) and the first
  pure-SIBYLL bin (89.1) is therefore apparently what the build intends, and the 56.2 GeV
  column should be (2/3) col(44.7) + (1/3) col(89.1). The pi+ multiplicity of that column,
  3.1428, sits exactly on the trend 3.0023 / **3.1428** / 3.2887 / 3.4284 (steps
  0.140 / 0.146 / 0.140), whereas the shipped 2.5658 does not.
* 2020 database `mceq_db_lext_dpm191_v12.h5` (v1.2.0, downloaded from the `builds_on_azure`
  release; its sha256 6353f661... cannot be checked against MCEq's checksum constant, which
  is that of the v1.4.0 file): the proton sum rule of SIBYLL21, SIBYLL23, SIBYLL23C,
  SIBYLL23C03, SIBYLL23CPP, QGSJET01C, QGSJETII03 and QGSJETII04 drops from 0.914 (44.7 GeV)
  to 0.740 (56.2 GeV; DPMJET 0.926) and recovers to 0.944-0.949 (70.8 GeV); cross sections
  are smooth there. SIBYLL23D is not in that file. (SIBYLL23CPP's p-air cross section falls
  to 31.6 mb at 141 GeV in the same file, an unrelated oddity.)
* v1.4.2 database (sha256 247f4020...6946): same 0.800 columns and 0.800 cross sections.

## 3. The patched databases (task 2)

Built by `diag_mceq_db_hole.py patch` from the shipped file, which is only read (sha256
5da415e9...bfdd0 unchanged): `/home/pigrange/mc3d_work/runs/dbhole/data_<variant>/
mceq_db_lext_dpm193_v140_dbhole_<variant>.h5` (321 MB each; `DDM_1.0.npy` symlinked).
Only the `SIBYLL23D` air yield dataset (and indptrs) and its cross-section dataset are
rewritten, only column 37 changes (verified bit-identical elsewhere), the DPMJET data are
untouched. Only bin 37 needs patching: the 70.8 GeV bin passes the sum rule.

| variant | 56.2 GeV SIBYLL23D yields | 56.2 GeV cross section |
|---|---|---|
| `noop` | re-packed, unchanged (writer check: MCEq fluxes identical to the original, max abs deviation 0.0) | unchanged |
| `interp` (**preferred**) | x-aligned log-E interpolation between the 44.7 and 89.1 GeV columns (weight 1/3 on 89.1): N_ij depends on i-j only (Feynman scaling on the log-uniform grid), so the neighbours are shifted onto bin 37 before interpolating; a row-aligned interpolation would wipe out the leading-particle peak. Sum rule 0.9694. | linear in ln E between the same two bins (273.69 mb) |
| `dpm` | the `DPMJETIII193` column (sum rule 0.9691) | the `DPMJETIII193` value (273.07 mb) |
| `interp_yonly`, `interp_csonly` | decomposition variants | |

Why `interp` is preferred: it is what the database does for its own 70.8 GeV bin and it
puts the pion multiplicity on the neighbour trend. `dpm` is the simplest "DPMJET column"
choice (= the 44.7 GeV rule continued); it differs from `interp` by at most
H3a: 0.41%, GSF: 0.46%, GSF_ddm: 0.07% (max |interp/dpm - 1| over all species, angles, energies; H3a, GSF, GSF + DDM),
so the choice does not matter at the level of the effect.

Pointing MCEq at a patched file (MCEq 1.4.1 reads `config.data_dir / config.mceq_db_fname`;
use a file name other than the default, because `ensure_db_available()` only
checksum-verifies and re-downloads the default name):

```python
import importlib.util, pathlib
import MCEq.config as cfg
cfg.data_dir = pathlib.Path("/home/pigrange/mc3d_work/runs/dbhole/data_interp")
cfg.mceq_db_fname = "mceq_db_lext_dpm193_v140_dbhole_interp.h5"
from MCEq.core import MCEqRun   # then as usual
```

## 4. Method of the flux comparison (task 3)

MCEq 1.4.1, `SIBYLL23D`, `e_min` = 0.1 GeV, CORSIKA US-standard (`BK_USStd`) atmosphere,
`total_numu / total_antinumu / total_nue / total_antinue / total_mu+ / total_mu-`
(conventional + prompt; prompt is negligible here), cos Z = 1, 0.5, 0.1, primaries
H3a and GSF (`GlobalSplineFitBeta`), plus a daemonflux-like baseline "GSF + DDM":
`MCEqRun.inject_ddm(DataDrivenModel())` on top of the same database. Ratios are
patched/original on MCEq's grid (10 bins/decade), linearly interpolated in ln E to the
requested energies; "max dev" is the largest |ratio - 1| on the grid points inside
0.1-1000 GeV.

## 5. Results (task 3)

### 5.1 Patched (`interp`) / original, H3a primaries


cos Z = 1

| species | 0.1 | 0.3 | 1 | 3 | 10 | 30 | 100 | max dev | at E [GeV] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| numu | 1.0029 | 1.0055 | 1.0125 | 1.0228 | 1.0164 | 1.0016 | 1.0000 | +2.52% | 4.47 |
| antinumu | 1.0030 | 1.0055 | 1.0128 | 1.0235 | 1.0121 | 0.9998 | 1.0000 | +2.44% | 3.55 |
| nue | 1.0030 | 1.0050 | 1.0115 | 1.0239 | 1.0241 | 0.9944 | 1.0000 | +2.91% | 5.62 |
| antinue | 1.0035 | 1.0058 | 1.0129 | 1.0241 | 1.0188 | 0.9930 | 1.0000 | +2.65% | 4.47 |
| mu+ | 1.0034 | 1.0058 | 1.0105 | 1.0183 | 1.0256 | 1.0035 | 0.9997 | +2.56% | 8.91 |
| mu- | 1.0033 | 1.0060 | 1.0110 | 1.0180 | 1.0209 | 0.9995 | 0.9997 | -2.32% | 44.7 |

cos Z = 0.5

| species | 0.1 | 0.3 | 1 | 3 | 10 | 30 | 100 | max dev | at E [GeV] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| numu | 1.0028 | 1.0053 | 1.0123 | 1.0225 | 1.0160 | 1.0010 | 1.0000 | +2.46% | 4.47 |
| antinumu | 1.0029 | 1.0053 | 1.0125 | 1.0231 | 1.0134 | 0.9993 | 1.0000 | +2.42% | 4.47 |
| nue | 1.0029 | 1.0048 | 1.0115 | 1.0235 | 1.0230 | 0.9962 | 1.0000 | +2.80% | 5.62 |
| antinue | 1.0033 | 1.0056 | 1.0127 | 1.0233 | 1.0174 | 0.9951 | 1.0000 | +2.53% | 4.47 |
| mu+ | 1.0125 | 1.0135 | 1.0162 | 1.0205 | 1.0231 | 0.9991 | 0.9996 | +2.36% | 7.08 |
| mu- | 1.0121 | 1.0131 | 1.0155 | 1.0186 | 1.0168 | 0.9961 | 0.9996 | +1.95% | 5.62 |

cos Z = 0.1

| species | 0.1 | 0.3 | 1 | 3 | 10 | 30 | 100 | max dev | at E [GeV] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| numu | 1.0027 | 1.0051 | 1.0118 | 1.0215 | 1.0151 | 1.0005 | 1.0000 | +2.33% | 4.47 |
| antinumu | 1.0027 | 1.0050 | 1.0118 | 1.0220 | 1.0148 | 0.9995 | 1.0000 | +2.34% | 4.47 |
| nue | 1.0026 | 1.0044 | 1.0108 | 1.0218 | 1.0199 | 0.9991 | 1.0000 | +2.52% | 5.62 |
| antinue | 1.0030 | 1.0051 | 1.0118 | 1.0213 | 1.0144 | 0.9983 | 1.0000 | +2.26% | 4.47 |
| mu+ | 1.0127 | 1.0125 | 1.0116 | 1.0091 | 1.0010 | 0.9953 | 0.9999 | +1.27% | 0.112 |
| mu- | 1.0057 | 1.0056 | 1.0050 | 1.0035 | 0.9984 | 0.9953 | 0.9998 | -0.62% | 22.4 |

### 5.2 Patched (`interp`) / original, GSF primaries


cos Z = 1

| species | 0.1 | 0.3 | 1 | 3 | 10 | 30 | 100 | max dev | at E [GeV] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| numu | 1.0027 | 1.0049 | 1.0112 | 1.0222 | 1.0187 | 1.0020 | 1.0000 | +2.63% | 5.62 |
| antinumu | 1.0028 | 1.0050 | 1.0116 | 1.0232 | 1.0141 | 0.9998 | 1.0000 | +2.52% | 4.47 |
| nue | 1.0029 | 1.0046 | 1.0104 | 1.0229 | 1.0269 | 0.9948 | 1.0000 | +3.05% | 7.08 |
| antinue | 1.0032 | 1.0053 | 1.0116 | 1.0234 | 1.0212 | 0.9932 | 1.0000 | +2.76% | 5.62 |
| mu+ | 1.0028 | 1.0050 | 1.0093 | 1.0171 | 1.0270 | 1.0048 | 0.9997 | +2.73% | 11.2 |
| mu- | 1.0027 | 1.0052 | 1.0098 | 1.0171 | 1.0223 | 1.0000 | 0.9997 | -2.30% | 44.7 |

cos Z = 0.5

| species | 0.1 | 0.3 | 1 | 3 | 10 | 30 | 100 | max dev | at E [GeV] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| numu | 1.0026 | 1.0048 | 1.0111 | 1.0220 | 1.0183 | 1.0013 | 1.0000 | +2.57% | 5.62 |
| antinumu | 1.0027 | 1.0048 | 1.0113 | 1.0228 | 1.0156 | 0.9994 | 1.0000 | +2.51% | 4.47 |
| nue | 1.0027 | 1.0044 | 1.0103 | 1.0226 | 1.0258 | 0.9965 | 1.0000 | +2.93% | 7.08 |
| antinue | 1.0031 | 1.0051 | 1.0115 | 1.0227 | 1.0197 | 0.9953 | 1.0000 | +2.62% | 5.62 |
| mu+ | 1.0113 | 1.0124 | 1.0153 | 1.0202 | 1.0250 | 1.0001 | 0.9996 | +2.53% | 8.91 |
| mu- | 1.0112 | 1.0122 | 1.0148 | 1.0187 | 1.0183 | 0.9965 | 0.9996 | +2.04% | 5.62 |

cos Z = 0.1

| species | 0.1 | 0.3 | 1 | 3 | 10 | 30 | 100 | max dev | at E [GeV] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| numu | 1.0025 | 1.0046 | 1.0107 | 1.0211 | 1.0172 | 1.0008 | 1.0000 | +2.41% | 5.62 |
| antinumu | 1.0026 | 1.0045 | 1.0107 | 1.0216 | 1.0171 | 0.9997 | 1.0000 | +2.42% | 4.47 |
| nue | 1.0025 | 1.0041 | 1.0097 | 1.0211 | 1.0225 | 0.9994 | 1.0000 | +2.64% | 5.62 |
| antinue | 1.0028 | 1.0046 | 1.0107 | 1.0209 | 1.0165 | 0.9985 | 1.0000 | +2.33% | 4.47 |
| mu+ | 1.0145 | 1.0142 | 1.0132 | 1.0106 | 1.0018 | 0.9953 | 0.9999 | +1.44% | 0.112 |
| mu- | 1.0065 | 1.0063 | 1.0056 | 1.0040 | 0.9986 | 0.9953 | 0.9998 | +0.65% | 0.112 |

The energy range 300-1000 GeV is unchanged (ratio 1.0000 to 4 digits); the full tables
with 300 and 1000 GeV, and the `dpm`, `interp_yonly` and `interp_csonly` variants, are in
`/home/pigrange/mc3d_work/runs/dbhole/tables.md`.

### 5.3 Decomposition and daemonflux-like baseline

nu_mu at cos Z = 1; the "max" columns scan nu_mu, nubar_mu, nu_e, nubar_e (or mu+-) at
all three zenith angles over 0.1-1000 GeV.

| baseline | patch | numu(cos Z=1) at 1 / 3 / 10 GeV | max nu dev (0.1-1000 GeV) | max mu dev |
|---|---|---|---|---|
| H3a, SIBYLL23D | yields + cross section, interpolated | 1.0125 / 1.0228 / 1.0164 | +2.91% (nue, cos Z=1, 5.62 GeV) | +2.56% (mu+, cos Z=1, 8.91 GeV) |
| H3a, SIBYLL23D | yields + cross section := DPMJET | 1.0124 / 1.0225 / 1.0149 | +2.87% (antinue, cos Z=1, 5.62 GeV) | -2.48% (mu-, cos Z=1, 44.7 GeV) |
| H3a, SIBYLL23D | yields only | 1.0115 / 1.0206 / 1.0163 | +2.25% (nue, cos Z=0.1, 5.62 GeV) | +2.83% (mu+, cos Z=0.5, 3.55 GeV) |
| H3a, SIBYLL23D | cross section only | 1.0007 / 1.0018 / 0.9996 | -1.69% (antinumu, cos Z=1, 22.4 GeV) | -2.52% (mu-, cos Z=1, 44.7 GeV) |
| GSF, SIBYLL23D | yields + cross section, interpolated | 1.0112 / 1.0222 / 1.0187 | +3.05% (nue, cos Z=1, 7.08 GeV) | +2.73% (mu+, cos Z=1, 11.2 GeV) |
| GSF, SIBYLL23D | yields + cross section := DPMJET | 1.0111 / 1.0220 / 1.0170 | +3.00% (antinue, cos Z=1, 5.62 GeV) | +2.54% (mu-, cos Z=1, 8.91 GeV) |
| GSF, SIBYLL23D | yields only | 1.0104 / 1.0200 / 1.0182 | +2.36% (nue, cos Z=0.1, 5.62 GeV) | +2.92% (mu+, cos Z=0.5, 5.62 GeV) |
| GSF, SIBYLL23D | cross section only | 1.0006 / 1.0017 / 0.9999 | -1.67% (antinumu, cos Z=1, 22.4 GeV) | -2.51% (mu-, cos Z=1, 44.7 GeV) |
| GSF + DDM yields | yields + cross section, interpolated | 1.0009 / 1.0023 / 1.0009 | -1.70% (antinumu, cos Z=1, 22.4 GeV) | -2.46% (mu-, cos Z=1, 44.7 GeV) |
| GSF + DDM yields | yields + cross section := DPMJET | 1.0010 / 1.0024 / 1.0009 | -1.72% (antinumu, cos Z=1, 22.4 GeV) | -2.51% (mu-, cos Z=1, 44.7 GeV) |
| GSF + DDM yields | yields only | 1.0002 / 1.0002 / 1.0001 | +0.04% (antinue, cos Z=0.1, 2.24 GeV) | +0.07% (mu-, cos Z=0.5, 0.112 GeV) |
| GSF + DDM yields | cross section only | 1.0007 / 1.0021 / 1.0008 | -1.70% (antinumu, cos Z=1, 22.4 GeV) | -2.46% (mu-, cos Z=1, 44.7 GeV) |

* The yield half and the cross-section half act on different species: the yield half
  gives the +1-3% at 1-10 GeV (neutrinos and muons), the cross-section half the -1.7%
  (nubar_mu at 22 GeV) and -2.5% (mu at 45 GeV) notches and -0.5 to -1% for muons below
  3 GeV (cos Z = 1).
* With DDM injected, MCEq 1.4.1 replaces 13 channels (p -> p, n, pbar, pi+-, K+-;
  pi- -> p, pbar, pi+-, K+-; mirrored to n and pi+ by isospin, K0S/L from K+-) over the
  whole MCEq grid (`ddm_matrices` uses `e_min = e_bins[0]`, `e_max = e_bins[-1]`), which
  includes the nucleon-regeneration channels; the SIBYLL yields that remain (K, Lambda,
  pi0 projectiles, ...) change nothing visible (<= 0.07%).

### 5.4 G_s response (`MCEq3DFlux.geomag_response`, cos Z_ref = 1; 0.1 gives the same to 0.1%)

G_s(patched, `interp`)/G_s(original) for total_numu (nu_e responds 0.1-0.5 points more
at R_c = 40 GV); the last row of each block is the double ratio G(7.1 GV)/G(40 GV) that sets
the 87-degree West/East amplitude (the cutoffs of `diag_ew_dpmjet.py`). The `dpm` patch
agrees to 0.1-0.4%.

| primary | R_c [GV] | 0.1 | 0.3 | 1 | 3 | 10 GeV |
|---|---|---:|---:|---:|---:|---:|
| H3a | 3 | 1.0006 | 1.0003 | 1.0000 | 1.0000 | 1.0000 |
| H3a | 7.1 | 1.0031 | 1.0026 | 1.0015 | 1.0002 | 1.0000 |
| H3a | 11 | 1.0059 | 1.0054 | 1.0037 | 1.0009 | 1.0000 |
| H3a | 40 | 1.0316 | 1.0306 | 1.0257 | 1.0121 | 1.0009 |
| H3a | **W/E-type**: G(7.1)/G(40) | 0.9724 | 0.9729 | 0.9764 | 0.9883 | 0.9991 |
| GSF | 3 | 1.0004 | 1.0002 | 1.0000 | 1.0000 | 1.0000 |
| GSF | 7.1 | 1.0024 | 1.0021 | 1.0013 | 1.0002 | 1.0000 |
| GSF | 11 | 1.0048 | 1.0045 | 1.0034 | 1.0010 | 1.0000 |
| GSF | 40 | 1.0314 | 1.0311 | 1.0276 | 1.0147 | 1.0014 |
| GSF | **W/E-type**: G(7.1)/G(40) | 0.9718 | 0.9718 | 0.9743 | 0.9858 | 0.9986 |

At R_c = 3-11 GV (mid-latitude sites, all zeniths except near-horizon East) the change is
<= 0.7% and vanishes above 3-10 GeV. At R_c = 40 GV the response is made by primaries of
>= 40 GV, whose cascade passes through the hole bin, so the *cut* flux changes more than
the full flux; this is why W/E, not H/V, moves (by -1.2 to -2.8% over 0.1-3 GeV).

## 6. Could daemonflux itself be affected? (task 4)

What the repository and the paper say:

* `src/daemonflux/data/daemonsplines_*_202303_2.pkl` (revision `202303_2`) hold only the
  parameter names, zenith-angle splines and the covariance matrices: **no MCEq version
  or database identifier**. `flux.py` only states that the `total_` prompt part is
  computed with SIBYLL-2.3d.
* The paper (Yanez & Fedynitch, arXiv:2303.00022, cited in the README) says the fluxes are
  computed with "MCEq 1.4" (as printed; the footnote is the MCEq repository), uses the GSF
  primary model and the Data-Driven Model (DDM) for the hadronic yields (31/158 GeV
  p-C/p-p and pi--C data; "only 10 of the 23 hadron production cross sections ... are
  relevant"), SIBYLL-2.3d being the baseline for everything else (cross sections, the
  remaining yields, decays). The relevant range is ~5-10,000 GeV (muon data), i.e. it
  contains the 56 GeV bin.

What we can conclude:

1. The dominant yields in daemonflux's conventional flux are DDM, not SIBYLL23D, so
   the yield half of the hole is not inherited (provided the DDM was injected over the
   whole grid, as MCEq 1.4.1's `inject_ddm` does): with DDM injected the effect of the
   hole in the baseline yields is <= 0.07% for every species/angle in 0.1-1000 GeV
   (section 5.3), against 2-3% for pure SIBYLL23D.
2. The cross-section half (present only in v1.4.x databases) would give up to 1.7% in
   neutrinos at 22 GeV and 2.5% in muons at 45 GeV. A 2023 database would not contain it
   if it resembles v1.2.0, whose cross sections are smooth; but see below.
3. The prompt flux (SIBYLL23D, `total_` quantities) is not sensitive: the hole acts on
   projectiles of 50-63 GeV, far below the energies where prompt matters.

What cannot be concluded:

* Which database the 2023 tables were built with. The public v1.2.0 file contains no
  SIBYLL23D, so the splines came from an unreleased/development database. Every
  database we could retrieve (2020, v1.4.0, v1.4.2) has the yield hole in all models
  that use the DPMJET extension, so a SIBYLL23D database of the same pipeline very
  likely has it as well (inference, not a measurement).
* The effect on the *calibrated* daemonflux (fit parameters, Jacobians): that needs the
  gradient library of the paper rebuilt with the patched database. Our DDM test is an
  uncalibrated baseline and ignores that the fit may partly absorb a smooth 1%-level
  distortion; the DDM central values here are those of `DDM_1.0.npy`.
* Whether the splines contain the cross-section half. If they do, a ~2% effect in the
  20-50 GeV neutrino/muon flux would be inside the quoted total error but systematic.

## 7. Draft issue for the MCEq maintainers (not posted)

> **Title:** Database: yield and cross-section columns of the E_kin = 56.2 GeV projectile
> bin of SIBYLL/QGSJET models are exactly 0.8 x DPMJET
>
> **Versions:** MCEq 1.4.1; database files `mceq_db_lext_dpm193_v140.h5` (sha256
> 5da415e9...), `mceq_db_lext_dpm193_v142.h5` from the v1.4.2 release (sha256
> 247f4020...) and, for the yields only, `mceq_db_lext_dpm191_v12.h5` (v1.2.0).
>
> **What I see.** In the "lext" databases the low-energy part of every model is
> DPMJET-III, blended into the high-energy model around `he_le_transition_gev = 80`. In the
> projectile bin E_kin = 56.2 GeV (grid index 37, 50.1-63.1 GeV) all yield matrix columns
> of SIBYLL21, SIBYLL23D, SIBYLL23E(*), QGSJETII04 and QGSJETIII are exactly
> 0.800000000000 x the DPMJETIII193 column (checked for all 448 `SIBYLL23D` tuples and 13,480
> non-empty cells), and, in v1.4.x, so are the inelastic cross sections of all projectiles
> (p-air: 218.5 mb at 56.2 GeV against 273.4 and 273.9 mb in the 44.7 and 70.8 GeV bins).
> The energy sum rule sum_i E_i N_ij / E_j of the proton column falls from 0.965 (44.7 GeV)
> to 0.775 and recovers to 0.975 (70.8 GeV). The columns below 44.7 GeV are bit-identical to
> DPMJET, and the 70.8 GeV column is, within 1-3%, 1/3 x col(44.7) + 2/3 x col(89.1), so
> the 56.2 GeV column looks like the DPMJET weight (0.8) applied against an empty
> high-energy column. DPMJETIII193, EPOSLHC and EPOSLHCR are not affected.
>
> **Reproducer** (MCEq API only):
> ```python
> import importlib.util, numpy as np
> from MCEq.core import MCEqRun
> import crflux.models as pm
> def first_interaction(model):
>     mc = MCEqRun(interaction_model=model, primary_model=(pm.HillasGaisser2012, "H3a"),
>                  theta_deg=0.0)
>     eg = mc.e_grid
>     n = mc._interactions.get_matrix((2212, 0), (211, 0)) * eg[:, None] / eg[None, :]
>     return eg, n.sum(axis=0), mc._int_cs.get_cs(2212)   # <n_pi+> and sigma_inel
> eg, ys, cs_s = first_interaction("SIBYLL23D")
> _, yd, cs_d = first_interaction("DPMJETIII193")
> j = np.searchsorted(eg, 56.0)
> print(eg[j], ys[j] / yd[j], cs_s[j] / cs_d[j])    # 56.2  0.800  0.800
> ```
>
> **Impact.** Replacing the 56.2 GeV column (and cross section) by the interpolation of the
> 44.7 and 89.1 GeV bins changes 1D fluxes computed with SIBYLL23D (GSF or H3a primaries,
> US standard atmosphere, cos Z = 1, 0.5, 0.1) by +0.3% (0.1 GeV), +0.5% (0.3), +1.2% (1),
> +2.2% (3) and +1.2-2.7% (10 GeV) for the four neutrino species, up to +3.0% at 5-7 GeV
> for nu_e; muons change by +0.3-1.5% below 1 GeV, up to +2.7% at ~10 GeV and by -2.3% at
> 45 GeV (the latter from the cross-section part); nothing above 100 GeV. With the DDM
> injected (`inject_ddm`, which replaces the p and pi yields) the yield part drops to
> <= 0.07% and only the cross-section part remains (-1.7% nubar_mu at 22 GeV, -2.5% mu at
> 45 GeV).
>
> **Suggested fix.** Where the high-energy column is empty, give the low-energy model
> weight 1 (or interpolate between the last pure-LE and the first pure-HE bins) for both the
> yield matrices and the cross sections, and add a regression test that the energy sum rule
> of every projectile column differs by less than a few % from its neighbours. Script and
> patched-database builder: `tools/mceq3d/diag_mceq_db_hole.py`
> (to be attached; `confirm` and `patch` need only numpy, scipy, h5py and MCEq).

## 8. Reproducing

```bash
PY=/cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc14-opt/bin/python3.12
export PYTHONPATH=/afs/cern.ch/work/p/pigrange/daemonflux/.venv/lib/python3.12/site-packages:/afs/cern.ch/work/p/pigrange/daemonflux/src
S=/afs/cern.ch/work/p/pigrange/daemonflux/tools/mceq3d/diag_mceq_db_hole.py
cd /home/pigrange/mc3d_work/runs/dbhole
$PY $S confirm                                        # section 2 (seconds)
$PY $S confirm --db newdb_dl/mceq_db_lext_dpm193_v142.h5
$PY $S olddb olddb_dl/mceq_db_lext_dpm191_v12.h5
$PY $S all --nproc 6                                  # patch + 24 flux/G_s jobs, ~30 min
```

The two downloaded databases were fetched from the `mceq-project/MCEq` GitHub release
`builds_on_azure` (assets `mceq_db_lext_dpm191_v12.h5`, `mceq_db_lext_dpm193_v142.h5`) into
`olddb_dl/` and `newdb_dl/` and only read with h5py.

## 9. Addendum (2026-10-11): the chromo neutron bug, verified and patched locally

The MCEq v1.4.2 release notes attribute the database's neutron defect to chromo
([chromo#275](https://github.com/impy-project/chromo/pull/275), closed unmerged):
the SIBYLL wrapper routes any projectile with `PDGID.is_nucleus` to
`sibnuc(A=1)`, and `is_nucleus` is true for p, n, pbar and nbar alike, so all four
are simulated as protons. chromo 0.11.0 (the version used by `tools/mc3d`) has it:
SIBYLL-2.3d n + air at 158 GeV gave pi+/pi- = 1.137 against 1.134 for p, while
DPMJET-III-19.3 is correct (n: 0.877 at 40 GeV).

A local copy with the two-line fix (`is_real_nucleus(kin.p1)` instead of
`kin.p1.is_nucleus` in `chromo/models/sibyll.py`) lives in
`/home/pigrange/mc3d_work/pylib_fix`; it gives pi+/pi- = 0.958 for n against 1.138
for p at 158 GeV. The Monte Carlo's nucleon and antinucleon event pools were
rebuilt with it (`/home/pigrange/mc3d_work/pools_fix`; the n pool at 4.75 TeV went
from pi+/pi- 1.053 to 0.976). Phase-2 results from milestone 3 on use the fixed
pools; milestone-2b results used the buggy ones (a charge-ratio effect from
primaries above 80 GeV/nucleon, small in the 3D/1D ratio).

## 10. 2026-11: switch to the patched v1.4.2 database

Status: done 2026-11-04. Decision (PI): the factorised 3D engine (`MCEq3DFlux`) moves
from `mceq_db_lext_dpm193_v140.h5` (shipped with MCEq 1.4.1, withdrawn upstream) to the
**v1.4.2 database with the 56.2 GeV hole patched** (`interp` variant of sections 3/8).
Scripts: `diag_mceq_db_hole.py patch --src` (builder), `mceq3d_db.py` (the one switch),
`diag_db_switch.py` (verification and re-validation), `test_mceq3d_db.py`. Raw outputs:
`/home/pigrange/mc3d_work/runs/dbv142/` (`out/`, `logs/`, `NOTES.md`).

### 10.1 The files

| label (`MCEQ3D_DB=`) | file | sha256 |
|---|---|---|
| `v140` | `.venv/.../MCEq/data/mceq_db_lext_dpm193_v140.h5` (shipped with MCEq 1.4.1) | `5da415e9bcf81926b1061d5792d75cb3aceb9de173beccb4695fd3909a0bfdd0` |
| `v140p` | `runs/dbhole/data_interp/mceq_db_lext_dpm193_v140_dbhole_interp.h5` | `d3beb1e861634322f1df840b84f2f2341a9b98e3db72efce795aeca6b3de0d73` |
| `v142` | `mceq_db/v142_raw/mceq_db_lext_dpm193_v142.h5` (symlink to `runs/dbhole/newdb_dl/`, the upstream file) | `247f40203436c69431db4d393316940636badb2c231f68d51870fb49bf476946` |
| **`v142p` (default)** | `mceq_db/mceq_db_lext_dpm193_v142_dbhole_interp.h5` | `7de8275ff3cbc46b6cb63c745a88761d8fd4a65e23f7617bdd8e99cfddcabddb` |
| (check) | `mceq_db/mceq_db_lext_dpm193_v142_dbhole_noop.h5` | `d192a94b4d49d69c1b22fadd8d6ec4e3569ce99ed1dd059e53868a6a9c0b0754` |
| (variant) | `mceq_db/mceq_db_lext_dpm193_v142_dbhole_dpm.h5` | `15b2e24e45ceb638db57a54e88cafd296c432a7c713b1474bd4e0e2ca9597157` |

(`mceq_db` = `/home/pigrange/mc3d_work/mceq_db`; `DDM_1.0.npy` is symlinked into each
data directory.)  Built with

```bash
$PY tools/mceq3d/diag_mceq_db_hole.py patch \
    --src /home/pigrange/mc3d_work/runs/dbhole/newdb_dl/mceq_db_lext_dpm193_v142.h5 \
    --outdir /home/pigrange/mc3d_work/mceq_db \
    --name 'mceq_db_lext_dpm193_v142_dbhole_{variant}.h5' --only noop interp dpm
```

(without `--src`/`--outdir` the command still patches the installed v1.4.0 file into
`runs/dbhole/data_<variant>/`, as in section 3).  The distinct file names matter:
MCEq's `ensure_db_available()` only checksum-verifies and re-downloads its default name.

### 10.2 Verification of the v1.4.2 hole and of the patch (`diag_db_switch.py verify`)

Full output: `runs/dbv142/out/verify_v142.txt`.

* v1.4.2 hole = same bin, same factor: the only bin with a > 10% energy-sum-rule
  deficit between 20 and 160 GeV is index 37 (56.2 GeV); all 12,264 non-empty
  SIBYLL23D yield cells of that column (416 tuples) are `0.800000000000` x DPMJETIII193
  (min = max), the columns below are bit-identical to DPMJET (0 differing tuples), and
  the inelastic cross section of all 24 tabulated projectiles is `0.800000000000` x
  DPMJET in that bin (p-air 218.45 mb against 273.45 / 273.86 mb in the neighbours).
* Patch: of the 116 datasets of the file exactly three differ from the v1.4.2 source
  (`hadronic_interactions/air/SIBYLL23D`, its `_indptrs`, `cross_sections/air/SIBYLL23D`);
  decoded, 0 of 416 tuples differ outside column 37, column 37 changes in all 416, and
  the cross-section array differs in row 37 only. Root attributes are identical.
* Sum rule, column 37: for all 16 projectiles 0.775-0.787 -> 0.969-0.983, within 2% of
  the mean of the neighbour columns (proton 0.9692 against 0.9643 / 0.9744).
* `noop` is dataset-for-dataset bit-identical to the v1.4.2 source (including the
  yield storage) and reproduces its MCEq fluxes exactly (max |noop/v142 - 1| = 0.0,
  H3a and GSF, all species and zeniths).
* Still carrying the hole in the patched file: SIBYLL21, SIBYLL23E and its four `STAR`
  variants, QGSJETII04 and QGSJETIII (air), SIBYLL23E (hydrogen). Clean: DPMJETIII193
  (air and hydrogen), EPOSLHC, EPOSLHCR, and the patched SIBYLL23D. Only SIBYLL23D was
  patched; the engine's default model is SIBYLL23D.

### 10.3 Does MCEq 1.4.1 read the v1.4.2 file? Yes; the 1.4.2 package is not needed

* The GitHub compare `v1.4.1...v1.4.2` (`runs/dbv142/cmp.json`) changes the code only in
  `src/MCEq/config.py` (default `mceq_db_fname` -> `..._v142.h5`, the default checksum,
  and the name of the old file `ensure_db_available()` removes), plus
  version/changelog/docs. `data.py`, `core.py` etc. are untouched, so MCEq 1.4.1
  with `config.mceq_db_fname` pointing at the v1.4.2 file is the same code path as
  MCEq 1.4.2. **No recipe change** (no `pylib_mceq142`). Release notes
  (`runs/dbv142/rel142.json`): "Fixes the Sibyll hadronic interaction matrices ... and
  DPMJetIII-19.3 anti-particle cross sections".
* Layout differences, all read correctly: the cross-section datasets now have 24 parent
  columns (antiparticles explicit) instead of 13; fewer tuples (the `-> e+-` channels
  are gone); `version` attribute still `1.4.0`.
* Independent checks (MCEq 1.4.1 on the v1.4.2 file, SIBYLL23D, GSF, vertical):
  mu+/mu- = 1.267 at 100 GeV (v1.4.0 file: 1.467; the measured muon charge ratio is
  about 1.27-1.3), nu_mu/nubar_mu = 1.44 (v1.4.0: 1.60); this is the neutron fix and
  it moves in the physical direction. MCEq(GSF, SIBYLL23D)/daemonflux for numu+nubar_mu
  at 1/3/10/30/100 GeV = 0.925/0.969/0.883/0.843/0.866 (patched v1.4.2) against
  0.913/0.946/0.868/0.842/0.862 (v1.4.0): same level, as expected
  (`runs/dbv142/out/indep_check.txt`).
* **Surprise, not in the release notes:** the v1.4.2 file differs from v1.4.0 for *every*
  model, not only SIBYLL (`runs/dbv142/out/h5diff_v140_v142.txt`, `dbdiff_v140_v142.txt`).
  Cross sections: DPMJETIII193 identical for the 13 old parents (+11 antiparticle
  columns); EPOSLHC pions up to 1.7%, nucleons 0.9%, K up to 15% (56 TeV);
  SIBYLL23D K up to 12% (141 GeV). SIBYLL23D yields are regenerated (212 of 416 tuples differ by more than 1e-6, at most 7.9% in L1); EPOS and DPMJET yields are the same to < 1e-6 (QGSJET not checked). The 1D
  EPOSLHC flux changes by up to 5% (nubar_e at ~900 GeV), outside the 3D-relevant
  energy range but not nothing.

### 10.4 The switch and the cache identity

* **One place:** `tools/mceq3d/mceq3d_db.py`. `MCEq.config.data_dir` / `mceq_db_fname`
  are set only there, by `select()`, and every `MCEqRun` of the package is built with
  `mceq3d_db.make_run(**kw)` (10 modules, including `diag_mceq_db_hole.py`; `test_every_mceqrun_goes_through_the_switch`
  fails if a module constructs `MCEqRun(` or imports `MCEq.core.MCEqRun` directly).
* **Default** `v142p` = `mceq_db/mceq_db_lext_dpm193_v142_dbhole_interp.h5`. Override with
  the environment variable `MCEQ3D_DB` = `v142p` | `v142` | `v140p` | `v140` | a path to
  any HDF5 file (read when the first `MCEqRun` is made; set it before importing the
  engine). `MCEQ3D_DB=v140` restores the pre-switch behaviour bit for bit, including
  all cache keys. Optional: `MCEQ3D_DB_DIR` (where the named files live, default
  `$MC3D_WORK/mceq_db`), `MCEQ3D_CACHE` (default `$MC3D_WORK/mceq3d_cache`),
  `MC3D_WORK` (default `/home/pigrange/mc3d_work`).
* **Database identity** = sha256 of the file (memoised in `mceq_db/.sha256_memo.json` by
  path/size/mtime). `DBSpec.tag` is `""` for the shipped v1.4.0 file (sha
  `5da415e9...`) and `_db<label>-<sha8>` for any other file, e.g. `_dbv142p-7de8275f`.
  Keying by content, not by name, means a file replaced under the same name changes the key.
* **Caches that depend on MCEq yields / cross sections and now carry the tag**
  (before, none of them had a database identity; the last two had a fixed file name
  without even the model/primary in it):

  | cache | where | key |
  |---|---|---|
  | `G_s` response `gs_*.npz` | `geomag_response` | `_gs_tag` / `_tag` = `engine_tag(model, primary, e_min, atmosphere, db tag)` -> `gs_cache_name` |
  | joint production profile `jointprod_chan_<TAG>.npz` | `joint_prod` | `<TAG><db tag>` in the file name |
  | channel fractions `channel_fractions.npz` | `kinematic_kernel.channel_fractions_cache()` | `channel_fractions<db tag>.npz` in `cache_dir()`; legacy name for the shipped file |
  | muon source table `muon_source_<method>.npz` | `muon_segment_mc.source_cache_path()` | `muon_source_<method><db tag>.npz` |

  For the shipped v1.4.0 file the tag is empty, so `.cache3d`, `channel_fractions.npz`
  etc. remain valid and unchanged for `MCEQ3D_DB=v140`. `_tag` keeps the database as a
  *suffix* because `offaxis_factor()` compares `self._tag.startswith(table_tag)`.
* **Not re-keyed** (pure geomagnetic back-traces or moment-file Monte Carlo): `finerc_*`,
  `prodfam_*`, `rc_*`, `conewidths_*`. For a non-legacy database `cache_dir()` is
  `$MCEQ3D_CACHE` (never AFS), DB-dependent files are written there, and the
  DB-independent maps are *copied* (never moved) from the requested directory on first
  use (`mceq3d_db.indep_path`); for the legacy database the requested directory is used
  unchanged. `$W/mceq3d_cache` was pre-seeded with a copy of the 50 map files of
  `.cache3d`; `.cache3d` was not modified by the new-database runs (125 files, newest 2026-09-09; one stray `gs_*.npz` written by a first smoke test was deleted at once).
* **Static, checked-in MCEq-derived tables are NOT memos and were NOT rebuilt:**
  `offaxis_excess*.npz` (E_off), `channel_fractions.npz` (legacy name), `m_*.npz`
  moments are independent of the database except E_off, whose production profile
  `p(X,E)` came from the v1.4.0 file. They now record `db` when rebuilt
  (`offaxis_mc.py --build`); `offaxis_factor()` emits a `UserWarning` (once per table
  and database) when a table without that key is used with another database
  (`mceq3d_db.check_static_table`). The warning appears in the test log. E_off is a ratio
  (G == 1 limit of the joint factor); the joint-cone path that is the default
  recomputes its production profile from the selected database
  (`jointprod_chan_*_dbv142p-...`), so the delivered `solve()` does not use the old E_off
  where it matters. Rebuilding E_off with the new database is listed under "remaining".
* **Test** `test_mceq3d_db.py` (7 tests, 0.7 s): default is `v142p`; env selection;
  `test_cache_key_changes_with_database` (engine tag and `gs_cache_name` differ for
  v140 / v142 / v142p, legacy key string unchanged);
  `test_stale_gs_cache_never_served` (a G_s file stored under the old key is read for
  v140 and is never read for v142/v142p, even if copied into the new cache directory);
  maps keep their name and are copied, not moved; channel-fractions cache name;
  no direct `MCEqRun`.

### 10.5 Results: old | new | new/old - 1 | Honda

"old" = `MCEQ3D_DB=v140`, "new" = the default `v142p`. Every number is measured in the
delivered engine path (`MCEq3DFlux(base_model="hybrid", primary=GSF, daemonflux_location=
"kamioka")`, `solve(...)` with the `diag_joint_stages` stage-S4 arguments, Kamioka,
2020-01-01), one process per database, the same cutoff maps and the same code. The split
is by chain: v140 -> **v140p** (same file, hole patched: "hole patch") -> **v142p**
(v1.4.2 file, hole patched: "v1.4.2 fixes" = neutron-as-proton, antiparticle
cross sections and the other regenerated tables). The factors multiply. Source files:
`runs/dbv142/out/engine_<label>.json`, `tables_doc.md` (these tables), `tables_engine.md`
(all four databases, hybrid base at cos Z 1/0.5/0.1, G_s for GSF and H3a, nu and nubar).
Check of the old column against `PHASE1_RESULTS.md`: 3.1 (S4 + the 6.3 prod-point anchor)
H/V/Honda 1.034/1.002/0.973/0.979 (6.3: 1.031/0.998/0.971/0.978), 3.2 W/E vs Honda
+12/+21% at 87 deg (6.3: +12/+21%), 3.3 D(dphi) 4.30 / -8.53 deg (6.3: 4.30 / -8.53),
phase offsets +1.3/+1.3/+0.5/-0.2 deg (6.3: same).

**3.1 H/V numu (azimuth-averaged horizon/vertical)**

| observable | old (v140) | new (v142p) | new/old - 1 | of which hole patch (v140p/v140) | of which v1.4.2 fixes (v142p/v140p) | Honda |
|---|---|---|---|---|---|---|
| 0.3 GeV | 1.887 | 1.889 | +0.09% | +0.03% | +0.06% | 1.825 |
| 0.5 GeV | 1.451 | 1.454 | +0.19% | +0.09% | +0.11% | 1.448 |
| 1.0 GeV | 1.207 | 1.211 | +0.31% | +0.17% | +0.14% | 1.241 |
| 3.0 GeV | 1.318 | 1.323 | +0.32% | +0.19% | +0.13% | 1.346 |

**3.1 H/V nue (azimuth-averaged horizon/vertical)**

| observable | old (v140) | new (v142p) | new/old - 1 | of which hole patch (v140p/v140) | of which v1.4.2 fixes (v142p/v140p) | Honda |
|---|---|---|---|---|---|---|
| 0.3 GeV | 2.269 | 2.267 | -0.07% | -0.02% | -0.05% | 2.119 |
| 0.5 GeV | 1.819 | 1.819 | -0.02% | +0.04% | -0.06% | 1.830 |
| 1.0 GeV | 1.644 | 1.644 | +0.05% | +0.14% | -0.09% | 1.770 |
| 3.0 GeV | 2.356 | 2.358 | +0.09% | +0.25% | -0.17% | 2.471 |

**3.2 W/E numu (engine W/E at the E/W azimuths; Honda: max/min over azimuth)**

| observable | old (v140) | new (v142p) | new/old - 1 | of which hole patch (v140p/v140) | of which v1.4.2 fixes (v142p/v140p) | Honda |
|---|---|---|---|---|---|---|
| 0.5 GeV, 87 deg | 2.815 | 2.768 | -1.67% | -1.54% | -0.13% | 2.507 |
| 0.5 GeV, 81 deg | 2.711 | 2.668 | -1.58% | -1.47% | -0.12% | 2.441 |
| 0.5 GeV, 75 deg | 2.527 | 2.490 | -1.43% | -1.34% | -0.10% | 2.285 |
| 1.0 GeV, 87 deg | 2.536 | 2.488 | -1.87% | -1.77% | -0.09% | 2.090 |
| 1.0 GeV, 81 deg | 2.404 | 2.363 | -1.72% | -1.65% | -0.08% | 2.024 |
| 1.0 GeV, 75 deg | 2.216 | 2.183 | -1.51% | -1.46% | -0.06% | 1.919 |

**3.2 W/E nue (engine W/E at the E/W azimuths; Honda: max/min over azimuth)**

| observable | old (v140) | new (v142p) | new/old - 1 | of which hole patch (v140p/v140) | of which v1.4.2 fixes (v142p/v140p) | Honda |
|---|---|---|---|---|---|---|
| 0.5 GeV, 87 deg | 3.149 | 3.098 | -1.63% | -1.75% | +0.12% | 4.740 |
| 0.5 GeV, 81 deg | 3.003 | 2.956 | -1.55% | -1.66% | +0.11% | 4.302 |
| 0.5 GeV, 75 deg | 2.772 | 2.733 | -1.42% | -1.51% | +0.09% | 3.488 |
| 1.0 GeV, 87 deg | 2.966 | 2.911 | -1.88% | -2.16% | +0.28% | 4.895 |
| 1.0 GeV, 81 deg | 2.777 | 2.729 | -1.75% | -2.00% | +0.26% | 3.887 |
| 1.0 GeV, 75 deg | 2.531 | 2.492 | -1.56% | -1.77% | +0.21% | 2.990 |

**3.3 charge splitting, 87 deg (nu - nubar); the 'new/old' columns are absolute differences**

| observable | old (v140) | new (v142p) | new - old | hole patch | v1.4.2 fixes | Honda |
|---|---|---|---|---|---|---|
| numu-anti 0.3 GeV D(dphi) [deg] | +4.563 | +4.576 | +0.012 | +0.001 | +0.011 | +5.953 |
| numu-anti 0.3 GeV D(max/min) | -0.090 | -0.088 | +0.002 | +0.003 | -0.001 | -0.906 |
| numu-anti 0.5 GeV D(dphi) [deg] | +4.296 | +4.313 | +0.017 | -0.000 | +0.017 | +4.355 |
| numu-anti 0.5 GeV D(max/min) | -0.161 | -0.156 | +0.005 | +0.007 | -0.001 | -1.305 |
| nue-anti 0.3 GeV D(dphi) [deg] | -9.118 | -9.119 | -0.000 | +0.000 | -0.001 | -12.218 |
| nue-anti 0.3 GeV D(max/min) | +0.334 | +0.328 | -0.006 | -0.009 | +0.004 | +1.694 |
| nue-anti 0.5 GeV D(dphi) [deg] | -8.532 | -8.533 | -0.001 | +0.000 | -0.001 | -9.209 |
| nue-anti 0.5 GeV D(max/min) | +0.497 | +0.487 | -0.010 | -0.017 | +0.008 | +2.623 |

**6.3 N/S model/Honda and first-harmonic phase offset (deg), 0.5 GeV, cos Z 0.05**

| observable | old (v140) | new (v142p) | new/old - 1 | hole patch | v1.4.2 fixes | Honda |
|---|---|---|---|---|---|---|
| N numu | 0.948 | 0.963 | +1.58% | +1.25% | +0.32% | 1.000 |
| S numu | 0.932 | 0.945 | +1.39% | +1.12% | +0.27% | 1.000 |
| phase numu [deg] (abs. diff.) | +1.26 | +1.28 | +0.023 | +0.001 | +0.023 | +0.00 |
| N antinumu | 0.980 | 0.994 | +1.42% | +1.32% | +0.10% | 1.000 |
| S antinumu | 0.922 | 0.933 | +1.17% | +1.09% | +0.08% | 1.000 |
| phase antinumu [deg] (abs. diff.) | +1.31 | +1.32 | +0.007 | +0.001 | +0.006 | +0.00 |
| N nue | 1.053 | 1.065 | +1.21% | +1.30% | -0.09% | 1.000 |
| S nue | 0.974 | 0.983 | +0.98% | +1.06% | -0.07% | 1.000 |
| phase nue [deg] (abs. diff.) | +0.45 | +0.45 | +0.001 | +0.002 | -0.001 | +0.00 |
| N antinue | 0.980 | 0.993 | +1.34% | +1.20% | +0.13% | 1.000 |
| S antinue | 0.992 | 1.005 | +1.27% | +1.14% | +0.13% | 1.000 |
| phase antinue [deg] (abs. diff.) | -0.23 | -0.22 | +0.002 | +0.002 | +0.001 | +0.00 |

**hybrid base, 1D flux, cos Z = 1, GSF, SIBYLL23D: new/old - 1 (total | hole patch | v1.4.2 fixes)**

| species | 0.1 GeV | 0.3 GeV | 1 GeV | 3 GeV | 10 GeV |
|---|---|---|---|---|---|
| numu | +0.31 / +0.27 / +0.04% | +0.61 / +0.48 / +0.13% | +1.07 / +0.89 / +0.18% | +0.35 / +0.37 / -0.02% | +0.00 / +0.01 / -0.01% |
| antinumu | +0.30 / +0.28 / +0.02% | +0.53 / +0.48 / +0.05% | +1.06 / +0.92 / +0.14% | +0.49 / +0.38 / +0.11% | +0.03 / +0.01 / +0.02% |
| nue | +0.29 / +0.29 / +0.01% | +0.45 / +0.45 / +0.00% | +0.76 / +0.83 / -0.06% | +0.30 / +0.38 / -0.08% | +0.00 / +0.02 / -0.01% |
| antinue | +0.35 / +0.32 / +0.03% | +0.57 / +0.51 / +0.06% | +1.08 / +0.94 / +0.14% | +0.52 / +0.40 / +0.12% | +0.03 / +0.01 / +0.02% |

**MCEq half of the hybrid base, 1D flux, cos Z = 1, GSF, SIBYLL23D: new/old - 1 (total | hole patch | v1.4.2 fixes)**

| species | 0.1 GeV | 0.3 GeV | 1 GeV | 3 GeV | 10 GeV |
|---|---|---|---|---|---|
| numu | +0.31 / +0.27 / +0.04% | +0.62 / +0.49 / +0.13% | +1.32 / +1.09 / +0.22% | +2.09 / +2.19 / -0.10% | -0.11 / +1.96 / -2.03% |
| antinumu | +0.30 / +0.28 / +0.02% | +0.54 / +0.49 / +0.05% | +1.30 / +1.13 / +0.17% | +2.97 / +2.29 / +0.67% | +4.69 / +1.51 / +3.13% |
| nue | +0.29 / +0.29 / +0.01% | +0.46 / +0.45 / +0.00% | +0.94 / +1.01 / -0.07% | +1.76 / +2.25 / -0.48% | +0.63 / +2.77 / -2.08% |
| antinue | +0.35 / +0.32 / +0.03% | +0.58 / +0.52 / +0.06% | +1.31 / +1.14 / +0.17% | +2.99 / +2.31 / +0.67% | +5.01 / +2.21 / +2.74% |

**G_s response, GSF primary, cos Z_ref = 1: old value | new/old - 1 (total / hole patch / v1.4.2 fixes)**

| species, R_c | 0.1 GeV | 0.3 GeV | 1 GeV | 3 GeV |
|---|---|---|---|---|
| numu, 7 GV | 0.5367: +0.26 / +0.23 / +0.03% | 0.7101: +0.23 / +0.20 / +0.03% | 0.9006: +0.12 / +0.12 / +0.00% | 0.9890: +0.02 / +0.02 / -0.00% |
| numu, 11 GV | 0.3574: +0.53 / +0.48 / +0.04% | 0.5194: +0.50 / +0.45 / +0.05% | 0.7638: +0.34 / +0.34 / +0.00% | 0.9559: +0.08 / +0.10 / -0.02% |
| numu, 40 GV | 0.0789: +3.27 / +3.14 / +0.12% | 0.1366: +3.23 / +3.11 / +0.11% | 0.2860: +2.66 / +2.76 / -0.09% | 0.5976: +1.14 / +1.46 / -0.31% |
| nue, 7 GV | 0.5403: +0.25 / +0.24 / +0.01% | 0.6690: +0.23 / +0.23 / +0.00% | 0.8653: +0.15 / +0.16 / -0.01% | 0.9850: +0.03 / +0.03 / -0.01% |
| nue, 11 GV | 0.3615: +0.52 / +0.50 / +0.01% | 0.4798: +0.50 / +0.49 / +0.00% | 0.7100: +0.39 / +0.42 / -0.03% | 0.9348: +0.12 / +0.15 / -0.03% |
| nue, 40 GV | 0.0800: +3.35 / +3.29 / +0.06% | 0.1197: +3.35 / +3.36 / -0.01% | 0.2408: +2.97 / +3.23 / -0.25% | 0.5220: +1.59 / +2.04 / -0.44% |

G_s(7 GV)/G_s(40 GV), numu, GSF: new/old - 1 = 0.3 GeV -2.90% (hole -2.82%), 1 GeV -2.49% (hole -2.58%), 3 GeV -1.13% (hole -1.43%)

G_s(7 GV)/G_s(40 GV), nue, GSF: new/old - 1 = 0.3 GeV -3.02% (hole -3.03%), 1 GeV -2.75% (hole -2.98%), 3 GeV -1.57% (hole -1.99%)

**G_s response, H3a primary, cos Z_ref = 1: old value | new/old - 1 (total / hole patch / v1.4.2 fixes)**

| species, R_c | 0.1 GeV | 0.3 GeV | 1 GeV | 3 GeV |
|---|---|---|---|---|
| numu, 7 GV | 0.4827: +0.34 / +0.30 / +0.03% | 0.6779: +0.30 / +0.26 / +0.04% | 0.8962: +0.14 / +0.14 / +0.00% | 0.9897: +0.02 / +0.02 / -0.00% |
| numu, 11 GV | 0.3242: +0.64 / +0.59 / +0.05% | 0.5007: +0.60 / +0.54 / +0.06% | 0.7676: +0.37 / +0.37 / -0.00% | 0.9609: +0.07 / +0.09 / -0.02% |
| numu, 40 GV | 0.0826: +3.30 / +3.16 / +0.14% | 0.1511: +3.19 / +3.06 / +0.12% | 0.3243: +2.45 / +2.56 / -0.10% | 0.6488: +0.90 / +1.20 / -0.30% |
| nue, 7 GV | 0.4856: +0.33 / +0.32 / +0.01% | 0.6297: +0.29 / +0.29 / +0.00% | 0.8571: +0.17 / +0.19 / -0.02% | 0.9856: +0.03 / +0.03 / -0.01% |
| nue, 11 GV | 0.3275: +0.63 / +0.62 / +0.02% | 0.4556: +0.60 / +0.59 / +0.00% | 0.7102: +0.42 / +0.46 / -0.04% | 0.9407: +0.11 / +0.15 / -0.04% |
| nue, 40 GV | 0.0835: +3.36 / +3.29 / +0.06% | 0.1304: +3.30 / +3.32 / -0.02% | 0.2725: +2.76 / +3.04 / -0.27% | 0.5729: +1.29 / +1.74 / -0.44% |

G_s(7 GV)/G_s(40 GV), numu, H3a: new/old - 1 = 0.3 GeV -2.80% (hole -2.72%), 1 GeV -2.27% (hole -2.37%), 3 GeV -0.89% (hole -1.18%)

G_s(7 GV)/G_s(40 GV), nue, H3a: new/old - 1 = 0.3 GeV -2.91% (hole -2.93%), 1 GeV -2.53% (hole -2.78%), 3 GeV -1.27% (hole -1.69%)

**1D MCEq fluxes** (no geomagnetic factor, SIBYLL23D, one process per database; the
engine's hybrid base uses only the MCEq half of these below ~1.7 GeV): the full set
(H3a and GSF, cos Z 1 and 0.1, nu, nubar, mu+-, 0.1-1000 GeV) is in
`runs/dbv142/out/tables_1d.md`. Maximum over 0.1-1000 GeV, vertical, GSF: hole patch
+3.05% (nu_e, 0.1-10 GeV), up to -2% at ~45 GeV in muons (as in section 5); **v1.4.2 fixes
at fixed patch: nu_mu -4%, nubar_mu +7.8%, nu_e -2.9%, nubar_e +6.3% (10-100 GeV)**,
and the muon charge ratio at 100 GeV falls from 1.467 to 1.267 (the v1.4.0 value is
the neutron-as-proton artefact).

### 10.6 Verdict

Nothing in the delivered engine moved by more than MCEQ_DB_HOLE.md section 1 predicted,
with three marginal exceedances/deviations that are stated here rather than rounded away.
(i) *Hybrid base:* +0.31 / +0.61 / +1.07 / +0.35% for nu_mu at 0.1 / 0.3 / 1 / 3 GeV and
< 0.05% from 10 GeV on; the hole patch alone reproduces the earlier +0.27 / +0.49 / +0.89 /
+0.37%, the v1.4.2 fixes add +0.04 / +0.13 / +0.18 / -0.02%, so the prediction
"sub-percent" is exceeded only by 0.07 points at 1 GeV. (ii) *G_s at 40 GV:* +3.27% (nu_mu)
and +3.35% (nu_e) at E <= 0.3 GeV (predicted <= 3%; hole patch alone +3.14/+3.29%,
v1.4.2 fixes +0.1%), +2.7/+3.0% at 1 GeV, +1.1/+1.6% at 3 GeV; <= 0.64% for R_c <= 11 GV;
`G_s(7 GV)/G_s(40 GV)` falls by 2.9% (0.3 GeV), 2.5% (1 GeV) and 1.1% (3 GeV) for nu_mu,
as predicted. (iii) *Near-horizon E-W amplitude:* predicted 2-3% smaller; in the delivered
engine W/E is smaller by only **1.4-1.9% for nu_mu** (87-75 deg, 0.5 and 1 GeV; 1.4-1.9%
for nu_e), the cone average and the 1D daemonflux half diluting the 2.5-2.9% of
`G_s(7)/G_s(40)`. This lowers the nu_mu overshoot of Honda from +12.3 to +10.4% (0.5 GeV,
87 deg) and from +21.3 to +19.1% (1 GeV, 87 deg); the nu_e deficit gets 1.1-1.2 points
worse (-33.6 -> -34.6%, -39.4 -> -40.5%). Everything else is below one percent: H/V
changes by +0.1..+0.3% (nu_mu) and < 0.1% (nu_e), the charge-splitting observables by
< 0.02 deg in D(dphi) and < 0.02 in D(max/min) (so Phase-1's finding that the amplitude
splitting is ~5x short of Honda is untouched), the first-harmonic phase offsets by < 0.03
deg, and N/S model/Honda rises uniformly by +1.0..+1.6% (a level shift: N and S move
together; the N-S contrast does not change). **Separation:** the 56 GeV hole accounts for
essentially all of it (hybrid base, G_s, W/E, N/S); the neutron and antiparticle fixes of
v1.4.2 change no 3D observable by more than 0.3% (H/V <= 0.17%, W/E <= 0.28%, G_s <= 0.44%
at 40 GV/3 GeV, hybrid base <= 0.18%) because they act on the charge and flavour ratio
at E >~ 3 GeV, where the hybrid base is the muon-calibrated daemonflux (weight > 0.5 above
1.7 GeV); the same fixes are large where MCEq itself is the base (nubar_mu +3.1%, nu_mu
-2.0% at 10 GeV in the MCEq half; 1D nu_mu -4%, nubar_mu +8% above 10 GeV), so they matter
for `base_model="mceq"` studies above a few GeV, for the Monte Carlo (section 10.7) and
for anything that uses charge ratios. The v1.4.2 file also differs from v1.4.0 for models
other than SIBYLL (10.3), which this engine does not use by default.

**Test suite** (`pytest -q -p no:cacheprovider` from `tools/mceq3d`, new default, 2026-11-04,
log `runs/dbv142/logs/pytest_final.log`): **275 passed, 2 skipped, 5 warnings in 3141 s.**
Before the switch: 268 passed, 2 skipped; the 7 new tests are `test_mceq3d_db.py`; the two
skips are the `test_joint_cone.py` ones that need `JOINT_CONE_PROD_CACHE`. The run started
from a copy of the new cache directory, so it also built the missing
`channel_fractions` entries; most of the time is the back-trace tests
(`test_cutoff_anchor.py`, ~45 min). The warnings are the two static-table `UserWarning`s of
10.4 and two RuntimeWarnings that were already there. An earlier full run (272 passed,
5 skipped) had three more skips: `test_muon_segment_mc.py` needs the MCEq-derived muon source
table in `$MSMC_SCRATCH` (a `/tmp` session scratchpad that did not contain one, so the
three tests skip independently of the switch). Since that table depends on the database,
its name now carries the tag (`muon_segment_mc.source_cache_path()`, test updated) and
it was rebuilt for the default database (28 s). `test_conservation.py` and
`test_cutoff_anchor.py` read the old `.cache3d/jointprod_chan_*.npz` by fixed path (the v1.4.0
production profile); they test geometry and were not changed.

### 10.7 What the Monte Carlo (`tools/mc3d`) would need (not changed here)

Read-only inspection of `tools/mc3d/build_tables.py`, `interactions.py`, `closure.py`.

* The MC takes from MCEq only the table file `tools/mc3d/mceq_tables_SIBYLL23D.npz`
  (`build_tables.py`): the inelastic cross sections `cs_<pdg>` (interaction lengths via
  `ChromoBackend(xs="mceq")`, the default) and, for the closure rungs, the yield and decay
  matrices (`MCEqYieldBackend`). The event pools (`pools_fix`, chromo) do not depend on the
  MCEq database.
* **`build_tables.build()` constructs `MCEqRun` directly** after `import MCEq.config`, so it
  uses MCEq's default database = the shipped v1.4.0 file. The switch in `mceq3d_db.py` is
  *not* seen by it.
* **The cached table is not keyed by database.** The npz holds only `e_grid, e_widths,
  e_bins, model, emin` plus the data; no file name, no sha256. The committed
  `mceq_tables_SIBYLL23D.npz` (10 Sep) therefore silently carries the v1.4.0 content,
  including the 0.8 hole: `cs_2212` at 56.2 GeV is 218.45 mb against 273.45 / 273.86 mb in
  the neighbouring bins, i.e. **the MC's interaction length is 25% too long between 50.1
  and 63.1 GeV** (and its v1.4.0 SIBYLL neutron-as-proton yields are in the `y_*` arrays).
* What has to change (for the agent that owns `tools/mc3d`, or the PM):
  1. `build_tables.py::build`: replace `MCEqRun(...)` by `mceq3d_db.make_run(...)`
     (`tools/mceq3d` is already on `PYTHONPATH` in the recipe) and write
     `db_label=np.array([spec.label])`, `db_sha256=np.array([spec.sha256])` (from
     `mceq3d_db.resolve()`) into the npz.
  2. `MCEqYieldBackend.__init__`: read those keys and warn (or raise) if
     `mceq3d_db.resolve().sha256` differs; a file without them is a v1.4.0 table.
  3. Regenerate: `MCEQ3D_DB=v142p python build_tables.py --model SIBYLL23D --emin 0.05
     --primary H3a --out mceq_tables_SIBYLL23D.npz` (a few minutes, one MCEq init) and
     redo the closure references that compare against MCEq (`closure_A1.npz`,
     `closure_A2*.npz`: MCEq is the reference of rungs A1/A2 and must use the same
     database as the tables).
  4. Anything that cached an MC result computed with `xs="mceq"` (`results/`, the
     `MC3D_POOL_CACHE` pools are fine) is stale in the 50-63 GeV interaction length and,
     through the yields, in the charge ratios above ~10 GeV (section 10.5); nothing in
     those files records which database it used.
  5. Optional: use `xs="hybrid"`, which takes chromo's own SIBYLL cross sections above the
     80 GeV splice, but below it MCEq's table is used anyway, and the hole is at 56 GeV.
* No MC code reads `mceq3d_db`; the MC runs in the same Python recipe, so
  `MCEQ3D_DB=v140` reproduces the current table build.

### 10.8 Reproducing

```bash
PY=/cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc14-opt/bin/python3.12
R=/afs/cern.ch/work/p/pigrange/daemonflux ; W=/home/pigrange/mc3d_work
export PYTHONPATH=$W/pylib_fix:$R/.venv/lib/python3.12/site-packages:$R/src:$R/tools/mceq3d:$R/tools/mc3d
export OMP_NUM_THREADS=1
cd $R/tools/mceq3d
# 1. databases (section 10.1), then verification
$PY diag_db_switch.py verify                       # -> $W/runs/dbv142/out/verify_v142.txt
# 2. 1D fluxes on each database (SIBYLL23D; 2 min each) and the compatibility checks
for db in v140 v140p v142 v142p; do for pr in H3a GSF; do $PY diag_db_switch.py flux $db $pr; done; done
$PY diag_db_switch.py flux $W/mceq_db/mceq_db_lext_dpm193_v142_dbhole_noop.h5 GSF --label noop
$PY diag_db_switch.py compat                       # -> out/compat.txt
# 3. delivered-path observables, one process per database (hybrid base + GSF + Kamioka;
#    the first run of a new database builds ~4 G_s families, the joint production profile
#    and the channel fractions at 11 zeniths: ~1.5 h of mostly single-core time at
#    cos Z = 0.05; v140 uses the existing .cache3d, copy it first: cp -r .cache3d $W/runs/dbv142/cache_old)
for db in v140 v140p v142 v142p; do
  $PY diag_db_switch.py engine $db --cache-dir $W/runs/dbv142/cache_old; done
$PY diag_db_switch.py tables                       # -> out/tables_engine.md
# 4. tests (a full run takes ~50 min under load)
$PY -m pytest -q -p no:cacheprovider
# old behaviour of any script:  MCEQ3D_DB=v140 $PY <script>
```

The delivered-path observables are computed with the code of `diag_joint_stages.py` (stage
`S4`: 3.1 H/V, 3.2 W/E; same grid, same `solve` arguments) and `diag_ew_charge_fourier.py`
(`model_flux`, `ew_observables`, `honda_ew`: 3.3 and 6.3), called from one engine
instance per database. N/S in 6.3 is the model/Honda ratio averaged over the two 30-degree
Honda bins adjacent to geographic North (centres 345/15 deg, compass) or South (165/195
deg), nu at cos Z = 0.05 and 0.5 GeV; the phase offset is `phi1(model) - phi1(Honda)`
of the first azimuthal harmonic. The 6.3 table in `PHASE1_RESULTS.md` quotes the same
quantities from a lost log (`ab_tables.txt`); our *v1.4.0* column reproduces its phase
offsets (+1.3, +1.3, +0.5, -0.2 deg) but its N/S values only roughly, since the exact
binning of that log is not recoverable. The old/new comparison is unaffected, as both
columns use the same definition.

### 10.9 Recipe change

None. MCEq 1.4.1 reads the v1.4.2 file (10.3), so `pylib_mceq142` is not needed and the
recipe of `MEMORY.md` / `PHASE1_RESULTS.md` section 7 is unchanged. New optional
environment variables: `MCEQ3D_DB`, `MCEQ3D_DB_DIR`, `MCEQ3D_CACHE`, `MC3D_WORK`
(10.4). The default now requires the file
`$MC3D_WORK/mceq_db/mceq_db_lext_dpm193_v142_dbhole_interp.h5` (and `DDM_1.0.npy` beside
it, for the data-driven model) to exist; `MCEQ3D_DB=v140` needs nothing new.

### 10.10 Remaining

* Rebuild the static MCEq-derived tables with the new database
  (`offaxis_excess*.npz`, `channel_fractions.npz`; `offaxis_mc.py --build` now writes the
  `db` key) and quantify their effect. The delivered joint-cone path already uses the
  database-keyed production profile, so the expected effect is small, but it was not measured.
* Models other than SIBYLL23D still carry the 56.2 GeV hole in v1.4.2 (10.2); patch them
  with `diag_mceq_db_hole.py` (it is written for SIBYLL23D only) if they are to be used.
* `tools/mc3d` table rebuild and closure re-run (10.7).
* `test_conservation.py` and `test_cutoff_anchor.py` read `.cache3d/jointprod_chan_*.npz`
  by fixed path, i.e. the v1.4.0 production profile, whatever the default database is.
  They test geometry, not database-dependent numbers; they were left alone.
* The upstream issue text (section 7) is still unposted and should be amended: v1.4.2
  still has the hole, and the file differs from v1.4.0 for every model.

### 10.11 Erratum (2026-11-04): G_s numbers in this file predate the rigidity fix

All G_s values and G_s-based observables quoted in sections 5.4 and 10.5 were computed with the
kinetic-energy cutoff convention fixed in `RIGIDITY_FIX.md` (G_s 3-15% too low below ~3 GeV).
The *database-effect ratios* (hole patch, v1.4.2 fixes) are ratios at fixed convention and are
expected to change little, but they were not re-measured; the v142p-before | after columns of
the headline observables are in `RIGIDITY_FIX.md` table A. The database cache identity of this
section is unchanged; `gs_*` keys additionally carry `rigp1`.

