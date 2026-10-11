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
